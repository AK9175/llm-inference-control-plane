package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	pb "github.com/atharva/llm-serving-platform/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	cp := flag.String("control-plane", "localhost:50051", "control plane gRPC address")
	id := flag.String("id", "ollama-worker-1", "unique worker ID")
	ollamaAddr := flag.String("ollama-addr", "localhost:11434", "Ollama HTTP address")
	provider := flag.String("provider", "local", "cloud provider: local | gcp | aws | azure")
	hardware := flag.String("hardware", "cpu", "hardware type: cpu | apple_m2 | nvidia_l4 | nvidia_a100")
	vram := flag.Uint64("vram", 0, "total VRAM in bytes (0 = CPU worker)")
	cost := flag.Float64("cost-per-hour", 0.0, "cost in $/hr")
	reportLoadEvery := flag.Duration("report-load-every", 10*time.Second, "load reporting interval")
	flag.Parse()

	// Verify Ollama is reachable and discover which models are loaded.
	models, err := ollamaModels(*ollamaAddr)
	if err != nil {
		log.Fatalf("[ollama-worker] cannot reach Ollama at %s: %v\nIs 'ollama serve' running?", *ollamaAddr, err)
	}
	if len(models) == 0 {
		log.Fatalf("[ollama-worker] Ollama is running but no models are loaded.\nRun: ollama pull llama3.2:3b")
	}
	fmt.Printf("[ollama-worker] Ollama at %s — models: %v\n", *ollamaAddr, models)

	// Connect to control plane.
	conn, err := grpc.NewClient(*cp, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("[ollama-worker] failed to connect to control plane %s: %v", *cp, err)
	}
	defer conn.Close()

	client := pb.NewWorkerRegistryClient(conn)
	ctx := context.Background()

	resp, err := client.Register(ctx, &pb.WorkerInfo{
		WorkerId:     *id,
		Address:      *ollamaAddr, // router will forward inference requests here directly
		ModelsLoaded: models,
		Backend:      "ollama",
		Provider:     *provider,
		Hardware:     *hardware,
		VramBytes:    *vram,
		CostPerHour:  *cost,
	})
	if err != nil {
		log.Fatalf("[ollama-worker] registration failed: %v", err)
	}

	heartbeatInterval := time.Duration(resp.HeartbeatIntervalSecs) * time.Second
	fmt.Printf("[ollama-worker] registered  id=%s  models=%v  heartbeat_every=%s\n",
		*id, models, heartbeatInterval)

	// currentState tracks the worker lifecycle.
	var stateMu sync.Mutex
	currentState := pb.WorkerState_READY // Ollama is already running — no STARTING delay

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

	stopHBChan := make(chan struct{})
	var hbOnce sync.Once
	stopHB := func() { hbOnce.Do(func() { close(stopHBChan) }) }

	drained := make(chan struct{})

	// Heartbeat loop.
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
					fmt.Printf("[ollama-worker] heartbeat error: %v\n", err)
					continue
				}
				fmt.Printf("[ollama-worker] ♥ heartbeat sent  id=%s  state=%s\n", *id, s)

				if hbResp.Drain {
					fmt.Printf("[ollama-worker] drain signal received — transitioning to DRAINING\n")
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

	// Load reporting loop — uses Ollama's /api/ps endpoint for real VRAM usage.
	go func() {
		ticker := time.NewTicker(*reportLoadEvery)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				load := ollamaLoad(*id, *ollamaAddr)
				if _, err := client.ReportLoad(ctx, load); err != nil {
					fmt.Printf("[ollama-worker] load report error: %v\n", err)
					continue
				}
				fmt.Printf("[ollama-worker] ↑ load report   id=%s  vram=%dMB\n",
					*id, load.VramUsedBytes/1024/1024)
			case <-stopHBChan:
				return
			}
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case <-quit:
		stopHB()
		fmt.Printf("[ollama-worker] signal received, deregistering id=%s\n", *id)
	case <-drained:
		stopHB()
		fmt.Printf("[ollama-worker] drain complete, deregistering id=%s\n", *id)
	}

	if _, err := client.Deregister(ctx, &pb.DeregisterRequest{WorkerId: *id}); err != nil {
		log.Printf("[ollama-worker] deregister error (non-fatal): %v", err)
	}
	fmt.Println("[ollama-worker] done")
}

// ── Ollama API helpers ────────────────────────────────────────────────────────

// ollamaModels calls GET /api/tags to list all models Ollama has pulled.
func ollamaModels(ollamaAddr string) ([]string, error) {
	resp, err := http.Get("http://" + ollamaAddr + "/api/tags")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	names := make([]string, 0, len(result.Models))
	for _, m := range result.Models {
		names = append(names, m.Name)
	}
	return names, nil
}

// ollamaLoad calls GET /api/ps to get currently running models and their VRAM usage.
// If the call fails, it returns a zero-value report rather than crashing.
func ollamaLoad(workerID, ollamaAddr string) *pb.LoadReport {
	resp, err := http.Get("http://" + ollamaAddr + "/api/ps")
	if err != nil {
		return &pb.LoadReport{WorkerId: workerID}
	}
	defer resp.Body.Close()

	var result struct {
		Models []struct {
			SizeVram uint64 `json:"size_vram"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return &pb.LoadReport{WorkerId: workerID}
	}

	var totalVram uint64
	for _, m := range result.Models {
		totalVram += m.SizeVram
	}

	return &pb.LoadReport{
		WorkerId:      workerID,
		VramUsedBytes: totalVram,
	}
}
