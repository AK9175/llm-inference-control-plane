package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os/signal"
	"syscall"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/api"
	"github.com/atharva/llm-serving-platform/control-plane/provisioner"
	"github.com/atharva/llm-serving-platform/control-plane/registry"
	"github.com/atharva/llm-serving-platform/control-plane/router"
	"github.com/atharva/llm-serving-platform/control-plane/scaler"
	pb "github.com/atharva/llm-serving-platform/proto"
	"google.golang.org/grpc"
)

func main() {
	grpcAddr    := flag.String("grpc-addr", ":50051", "gRPC listen address (worker registration + heartbeat)")
	httpAddr    := flag.String("http-addr", ":8080", "HTTP listen address (inference router)")
	adminAddr   := flag.String("admin-addr", ":9090", "HTTP listen address (admin API)")
	minHealthy  := flag.Int("min-healthy", 1, "minimum healthy workers before scaler warns")
	deadGrace   := flag.Duration("dead-grace", scaler.DefaultDeadGrace, "how long to keep a DEAD worker in registry before evicting")
	dockerImage := flag.String("docker-image", "", "Docker image for auto-provisioning on worker death (empty = disabled)")
	dockerCPAddr := flag.String("docker-cp-addr", "host.docker.internal:50051", "gRPC address provisioned containers use to reach the control plane")
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

	// ── Fleet Scaler (dead worker cleanup + health monitoring) ────────────────
	scalerOpts := []scaler.Option{
		scaler.WithMinHealthy(*minHealthy),
		scaler.WithDeadGrace(*deadGrace),
	}
	if *dockerImage != "" {
		dp := &provisioner.DockerProvisioner{
			Image:        *dockerImage,
			ControlPlane: *dockerCPAddr,
		}
		scalerOpts = append(scalerOpts, scaler.WithOnWorkerEvicted(dp.OnEvicted))
		fmt.Printf("[control-plane] auto-provisioner: docker image=%s  cp=%s\n", *dockerImage, *dockerCPAddr)
	}
	sc := scaler.New(reg, scalerOpts...)
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()
	go sc.Run(ctx)

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

	// Graceful shutdown — ctx is already wired to SIGINT/SIGTERM above.
	go func() {
		<-ctx.Done()
		fmt.Println("[control-plane] shutting down...")
		grpcSrv.GracefulStop()
	}()

	if err := grpcSrv.Serve(lis); err != nil {
		log.Fatalf("gRPC server failed: %v", err)
	}
}
