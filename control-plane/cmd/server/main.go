package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/api"
	"github.com/atharva/llm-serving-platform/control-plane/registry"
	"github.com/atharva/llm-serving-platform/control-plane/router"
	pb "github.com/atharva/llm-serving-platform/proto"
	"google.golang.org/grpc"
)

func main() {
	grpcAddr  := flag.String("grpc-addr", ":50051", "gRPC listen address (worker registration + heartbeat)")
	httpAddr  := flag.String("http-addr", ":8080", "HTTP listen address (inference router)")
	adminAddr := flag.String("admin-addr", ":9090", "HTTP listen address (admin API)")
	flag.Parse()

	// Each worker gets its own deadline timer — no background sweep goroutine needed.
	reg := registry.New(registry.DefaultDeadTimeout)

	// ── gRPC server (workers talk here) ──────────────────────────────────────
	lis, err := net.Listen("tcp", *grpcAddr)
	if err != nil {
		log.Fatalf("failed to listen on %s: %v", *grpcAddr, err)
	}
	grpcSrv := grpc.NewServer()
	pb.RegisterWorkerRegistryServer(grpcSrv, reg)

	fmt.Printf("[control-plane] gRPC  listening on %s  (worker registry)\n", *grpcAddr)
	fmt.Printf("[control-plane] HTTP  listening on %s  (inference router)\n", *httpAddr)
	fmt.Printf("[control-plane] admin listening on %s  (admin API)\n", *adminAddr)
	fmt.Printf("[control-plane] dead timeout: %s\n", registry.DefaultDeadTimeout)

	// ── HTTP router (clients send inference requests here) ───────────────────
	rtr := router.New(reg)
	go func() {
		if err := http.ListenAndServe(*httpAddr, rtr); err != nil {
			log.Fatalf("HTTP router failed: %v", err)
		}
	}()

	// ── Admin API (fleet visibility + manual control) ─────────────────────────
	go func() {
		if err := http.ListenAndServe(*adminAddr, api.NewAdminHandler(reg, rtr)); err != nil {
			log.Fatalf("admin API failed: %v", err)
		}
	}()

	// Fleet status printer — every 15s shows all workers and their load.
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
		grpcSrv.GracefulStop()
	}()

	if err := grpcSrv.Serve(lis); err != nil {
		log.Fatalf("gRPC server failed: %v", err)
	}
}
