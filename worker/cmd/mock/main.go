package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	pb "github.com/atharva/llm-serving-platform/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	cp := flag.String("control-plane", "localhost:50051", "control plane gRPC address")
	id := flag.String("id", "mock-worker-1", "unique worker ID")
	addr := flag.String("addr", "localhost:8080", "this worker's inference address (fake for mock)")
	models := flag.String("models", "llama3.2:3b", "comma-separated model IDs this worker serves")
	backend := flag.String("backend", "mock", "inference backend: mock | ollama | vllm")
	provider := flag.String("provider", "local", "cloud provider: local | gcp | aws | azure")
	hardware := flag.String("hardware", "cpu", "hardware type: cpu | apple_m2 | nvidia_l4 | nvidia_a100")
	vram := flag.Uint64("vram", 0, "VRAM in bytes (0 for CPU workers)")
	cost := flag.Float64("cost-per-hour", 0.0, "cost in $/hr (0 for local workers)")
	crash := flag.Bool("crash", false, "simulate a crash: exit without deregistering (tests dead detection)")
	flag.Parse()

	conn, err := grpc.NewClient(*cp, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("[mock-worker] failed to connect to %s: %v", *cp, err)
	}
	defer conn.Close()

	client := pb.NewWorkerRegistryClient(conn)
	ctx := context.Background()

	// Register
	modelList := strings.Split(*models, ",")
	resp, err := client.Register(ctx, &pb.WorkerInfo{
		WorkerId:     *id,
		Address:      *addr,
		GpuType:      *hardware,
		VramBytes:    *vram,
		ModelsLoaded: modelList,
		Backend:      *backend,
		Provider:     *provider,
		Hardware:     *hardware,
		CostPerHour:  *cost,
	})
	if err != nil {
		log.Fatalf("[mock-worker] registration failed: %v", err)
	}

	heartbeatInterval := time.Duration(resp.HeartbeatIntervalSecs) * time.Second
	fmt.Printf("[mock-worker] registered  id=%s  models=%v  heartbeat_every=%s\n",
		*id, modelList, heartbeatInterval)

	// Crash mode: exit immediately without deregistering.
	// Use this to test CP3 dead-worker detection.
	if *crash {
		fmt.Printf("[mock-worker] crash mode — exiting without deregistering\n")
		os.Exit(0)
	}

	// CP3: heartbeat loop — runs in background until stop signal.
	stopHB := make(chan struct{})
	go func() {
		ticker := time.NewTicker(heartbeatInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				_, err := client.Heartbeat(ctx, &pb.HeartbeatRequest{
					WorkerId: *id,
					State:    pb.WorkerState_READY,
				})
				if err != nil {
					fmt.Printf("[mock-worker] heartbeat error: %v\n", err)
				} else {
					fmt.Printf("[mock-worker] ♥ heartbeat sent  id=%s\n", *id)
				}
			case <-stopHB:
				return
			}
		}
	}()

	// Block until SIGINT / SIGTERM
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// Graceful shutdown: stop heartbeats, then deregister
	close(stopHB)
	fmt.Printf("[mock-worker] shutting down, deregistering id=%s\n", *id)
	if _, err := client.Deregister(ctx, &pb.DeregisterRequest{WorkerId: *id}); err != nil {
		log.Printf("[mock-worker] deregister error (non-fatal): %v", err)
	}
	fmt.Println("[mock-worker] done")
}
