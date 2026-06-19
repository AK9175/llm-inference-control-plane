package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
	pb "github.com/atharva/llm-serving-platform/proto"
	"google.golang.org/grpc"
)

func main() {
	addr := flag.String("addr", ":50051", "gRPC listen address")
	flag.Parse()

	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("failed to listen on %s: %v", *addr, err)
	}

	// Each worker gets its own deadline timer — no background sweep goroutine needed.
	reg := registry.New(registry.DefaultDeadTimeout)

	srv := grpc.NewServer()
	pb.RegisterWorkerRegistryServer(srv, reg)

	fmt.Printf("[control-plane] listening on %s\n", *addr)
	fmt.Printf("[control-plane] dead timeout: %s (per-worker timer)\n", registry.DefaultDeadTimeout)

	// Fleet status printer — every 15s prints all workers and their load metrics.
	// Gives visibility into what the router will see when picking a worker (CP6).
	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			workers := reg.ListWorkers()
			if len(workers) == 0 {
				continue
			}
			fmt.Printf("[control-plane] ── fleet (%d workers) ──────────────────────────\n", len(workers))
			for _, w := range workers {
				fmt.Printf("  %-22s %-10s queue=%-3d rps=%-6.1f latency=%-6.0fms vram=%dMB\n",
					w.Info.WorkerId, w.State,
					w.Load.QueueDepth, w.Load.RequestsPerSec, w.Load.AvgLatencyMs,
					w.Load.VramUsedBytes/1024/1024)
			}
		}
	}()

	// Graceful shutdown on SIGINT / SIGTERM.
	go func() {
		quit := make(chan os.Signal, 1)
		signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
		<-quit
		fmt.Println("[control-plane] shutting down...")
		srv.GracefulStop()
	}()

	if err := srv.Serve(lis); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}
