package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"strings"
	"sync"
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
	modelLoadTime := flag.Duration("model-load-time", 3*time.Second, "simulated model load time before transitioning STARTING→READY")
	reportLoadEvery := flag.Duration("report-load-every", 10*time.Second, "how often to send load metrics to the control plane")
	flag.Parse()

	conn, err := grpc.NewClient(*cp, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("[mock-worker] failed to connect to %s: %v", *cp, err)
	}
	defer conn.Close()

	client := pb.NewWorkerRegistryClient(conn)
	ctx := context.Background()

	// Register — state stored as STARTING on the control plane.
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
	fmt.Printf("[mock-worker] registered  id=%s  models=%v  heartbeat_every=%s  load_time=%s\n",
		*id, modelList, heartbeatInterval, *modelLoadTime)

	// Crash mode: exit immediately without deregistering.
	if *crash {
		fmt.Printf("[mock-worker] crash mode — exiting without deregistering\n")
		os.Exit(0)
	}

	// currentState is the state the heartbeat loop reports.
	// Protected by stateMu — both the heartbeat goroutine and the
	// model-load timer goroutine read/write it.
	var stateMu sync.Mutex
	currentState := pb.WorkerState_STARTING

	getState := func() pb.WorkerState {
		stateMu.Lock()
		defer stateMu.Unlock()
		return currentState
	}
	setState := func(s pb.WorkerState) {
		stateMu.Lock()
		defer stateMu.Unlock()
		currentState = s
	}

	// stopHBChan is closed to stop the heartbeat goroutine.
	// sync.Once ensures we never close a closed channel.
	stopHBChan := make(chan struct{})
	var hbOnce sync.Once
	stopHB := func() { hbOnce.Do(func() { close(stopHBChan) }) }

	// drained is closed by the heartbeat goroutine when it receives drain=true
	// from the control plane, signalling main to deregister.
	drained := make(chan struct{})

	// Heartbeat loop — runs until stopHBChan is closed or drain is received.
	go func() {
		ticker := time.NewTicker(heartbeatInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				s := getState()
				hbResp, err := client.Heartbeat(ctx, &pb.HeartbeatRequest{
					WorkerId: *id,
					State:    s,
				})
				if err != nil {
					fmt.Printf("[mock-worker] heartbeat error: %v\n", err)
					continue
				}
				fmt.Printf("[mock-worker] ♥ heartbeat sent  id=%s  state=%s\n", *id, s)

				if hbResp.Drain {
					// Control plane asked us to drain.
					// Transition to DRAINING, send one final heartbeat, then signal main.
					fmt.Printf("[mock-worker] drain signal received — transitioning to DRAINING\n")
					setState(pb.WorkerState_DRAINING)
					client.Heartbeat(ctx, &pb.HeartbeatRequest{ //nolint:errcheck
						WorkerId: *id,
						State:    pb.WorkerState_DRAINING,
					})
					close(drained)
					return
				}
			case <-stopHBChan:
				return
			}
		}
	}()

	// Model load simulation — after the configured delay, transition to READY.
	// This mirrors what a real worker does: it registers (STARTING), loads the
	// model weights into VRAM, then signals readiness.
	go func() {
		if *modelLoadTime > 0 {
			fmt.Printf("[mock-worker] loading model (simulated %s)...\n", *modelLoadTime)
			time.Sleep(*modelLoadTime)
		}
		fmt.Printf("[mock-worker] model loaded → transitioning to READY\n")
		setState(pb.WorkerState_READY)
	}()

	// Load reporting loop — sends ReportLoad every --report-load-every seconds.
	// Metrics are fake but state-aware: READY workers report low load, BUSY workers
	// report higher queue depth and latency. This gives the router real data to
	// make least-loaded routing decisions (CP6).
	go func() {
		ticker := time.NewTicker(*reportLoadEvery)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				load := fakeLoad(*id, getState(), *vram)
				if _, err := client.ReportLoad(ctx, load); err != nil {
					fmt.Printf("[mock-worker] load report error: %v\n", err)
					continue
				}
				fmt.Printf("[mock-worker] ↑ load report   id=%s  queue=%d  rps=%.1f  latency=%.0fms  vram=%dMB\n",
					*id, load.QueueDepth, load.RequestsPerSec, load.AvgLatencyMs, load.VramUsedBytes/1024/1024)
			case <-stopHBChan:
				return
			}
		}
	}()

	// Block until SIGINT/SIGTERM (user shutdown) or drain (Fleet Scaler scale-down).
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case <-quit:
		stopHB()
		fmt.Printf("[mock-worker] signal received, deregistering id=%s\n", *id)
	case <-drained:
		stopHB()
		fmt.Printf("[mock-worker] drain complete, deregistering id=%s\n", *id)
	}

	if _, err := client.Deregister(ctx, &pb.DeregisterRequest{WorkerId: *id}); err != nil {
		log.Printf("[mock-worker] deregister error (non-fatal): %v", err)
	}
	fmt.Println("[mock-worker] done")
}

// fakeLoad generates plausible load metrics based on the worker's current state.
// READY → idle metrics; BUSY → higher queue and latency; anything else → zeros.
// A real worker would measure these from its actual inference queue.
func fakeLoad(workerID string, state pb.WorkerState, vramBytes uint64) *pb.LoadReport {
	switch state {
	case pb.WorkerState_READY:
		return &pb.LoadReport{
			WorkerId:       workerID,
			QueueDepth:     0,
			RequestsPerSec: rand.Float64() * 2,
			AvgLatencyMs:   80 + rand.Float64()*40,
			VramUsedBytes:  vramBytes * uint64(30+rand.Intn(20)) / 100,
		}
	case pb.WorkerState_BUSY:
		return &pb.LoadReport{
			WorkerId:       workerID,
			QueueDepth:     uint32(1 + rand.Intn(8)),
			RequestsPerSec: 5 + rand.Float64()*15,
			AvgLatencyMs:   200 + rand.Float64()*300,
			VramUsedBytes:  vramBytes * uint64(60+rand.Intn(30)) / 100,
		}
	default:
		return &pb.LoadReport{WorkerId: workerID}
	}
}
