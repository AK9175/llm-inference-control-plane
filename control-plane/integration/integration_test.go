// Package integration_test exercises the full Phase 1 control plane stack:
// gRPC worker registry + HTTP inference router + admin API + fleet scaler.
//
// Tests start all servers in-process on random ports, simulate workers via
// real gRPC client calls, and send real HTTP requests — no mocks of the
// control plane itself.
package integration_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/api"
	"github.com/atharva/llm-serving-platform/control-plane/metrics"
	"github.com/atharva/llm-serving-platform/control-plane/registry"
	"github.com/atharva/llm-serving-platform/control-plane/router"
	"github.com/atharva/llm-serving-platform/control-plane/scaler"
	pb "github.com/atharva/llm-serving-platform/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ── Test stack ────────────────────────────────────────────────────────────────

// stack is a fully wired Phase 1 control plane running on random ports.
// The scaler is NOT started here — tests that need eviction behaviour create
// their own scaler so they can inject callbacks without racing the stack's one.
type stack struct {
	GRPCAddr  string // workers register here
	HTTPAddr  string // clients send inference here
	AdminAddr string // admin API + /metrics

	reg *registry.Registry
}

// newStack starts gRPC + HTTP router + admin on random ports.
// Everything is torn down via t.Cleanup when the test ends.
func newStack(t *testing.T, deadTimeout time.Duration) *stack {
	t.Helper()

	reg := registry.New(deadTimeout)
	rtr := router.New(reg)
	metricsHandler, hook := metrics.Setup(reg, rtr)
	rtr.SetRequestHook(hook)

	// gRPC — worker registry.
	grpcLis := mustListen(t)
	grpcSrv := grpc.NewServer()
	pb.RegisterWorkerRegistryServer(grpcSrv, reg)
	go grpcSrv.Serve(grpcLis) //nolint:errcheck

	// HTTP — inference router.
	httpLis := mustListen(t)
	httpSrv := &http.Server{Handler: rtr}
	go httpSrv.Serve(httpLis) //nolint:errcheck

	// Admin — /admin/* + /metrics.
	adminLis := mustListen(t)
	adminSrv := &http.Server{Handler: api.NewAdminHandler(reg, rtr, metricsHandler)}
	go adminSrv.Serve(adminLis) //nolint:errcheck

	t.Cleanup(func() {
		grpcSrv.GracefulStop()
		httpSrv.Close()  //nolint:errcheck
		adminSrv.Close() //nolint:errcheck
	})

	return &stack{
		GRPCAddr:  grpcLis.Addr().String(),
		HTTPAddr:  httpLis.Addr().String(),
		AdminAddr: adminLis.Addr().String(),
		reg:       reg,
	}
}

func mustListen(t *testing.T) net.Listener {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	return l
}

// ── Worker simulation helpers ─────────────────────────────────────────────────

func workerClient(t *testing.T, grpcAddr string) pb.WorkerRegistryClient {
	t.Helper()
	conn, err := grpc.NewClient(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("connect to control plane gRPC: %v", err)
	}
	t.Cleanup(func() { conn.Close() }) //nolint:errcheck
	return pb.NewWorkerRegistryClient(conn)
}

// registerReady registers a worker and sends a READY heartbeat.
// inferAddr is the address of the fake inference HTTP server the router will proxy to.
func registerReady(t *testing.T, client pb.WorkerRegistryClient, id, inferAddr, model string) {
	t.Helper()
	ctx := t.Context()
	if _, err := client.Register(ctx, &pb.WorkerInfo{
		WorkerId:     id,
		Address:      inferAddr,
		ModelsLoaded: []string{model},
		Backend:      "mock",
		Provider:     "local",
		Hardware:     "cpu",
	}); err != nil {
		t.Fatalf("register %s: %v", id, err)
	}
	if _, err := client.Heartbeat(ctx, &pb.HeartbeatRequest{
		WorkerId: id,
		State:    pb.WorkerState_READY,
	}); err != nil {
		t.Fatalf("heartbeat %s: %v", id, err)
	}
}

// fakeInferenceServer starts a fake inference HTTP server that returns a fixed
// response and counts how many requests it received.
func fakeInferenceServer(t *testing.T) (addr string, hits *atomic.Int64) {
	t.Helper()
	var count atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		count.Add(1)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	t.Cleanup(srv.Close)
	return strings.TrimPrefix(srv.URL, "http://"), &count
}

// waitForState polls until the worker reaches the target state or the timeout expires.
func waitForState(t *testing.T, reg *registry.Registry, workerID string, want pb.WorkerState, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if e, ok := reg.GetWorker(workerID); ok && e.State == want {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	e, _ := reg.GetWorker(workerID)
	t.Fatalf("worker %s did not reach %s within %s (current: %s)", workerID, want, timeout, e.State)
}

// waitEvicted polls until the worker is no longer in the registry.
func waitEvicted(t *testing.T, reg *registry.Registry, workerID string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, ok := reg.GetWorker(workerID); !ok {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("worker %s was not evicted within %s", workerID, timeout)
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

func inferenceRequest(t *testing.T, httpAddr, model string) *http.Response {
	t.Helper()
	body := fmt.Sprintf(`{"model":%q,"messages":[]}`, model)
	resp, err := http.Post(
		"http://"+httpAddr+"/v1/chat/completions",
		"application/json",
		bytes.NewBufferString(body),
	)
	if err != nil {
		t.Fatalf("inference request: %v", err)
	}
	return resp
}

func adminGet(t *testing.T, adminAddr, path string) []byte {
	t.Helper()
	resp, err := http.Get("http://" + adminAddr + path)
	if err != nil {
		t.Fatalf("GET %s: %v", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s: status %d", path, resp.StatusCode)
	}
	b, _ := io.ReadAll(resp.Body)
	return b
}

// ── Integration tests ─────────────────────────────────────────────────────────

// TestIntegration_WorkerRegistersAndServesTraffic verifies the happy path:
// worker registers → router routes inference → admin API shows correct state.
func TestIntegration_WorkerRegistersAndServesTraffic(t *testing.T) {
	s := newStack(t, 15*time.Second)
	client := workerClient(t, s.GRPCAddr)
	inferAddr, hits := fakeInferenceServer(t)

	registerReady(t, client, "w1", inferAddr, "llama3.2:3b")

	// Router proxies inference to the fake upstream.
	resp := inferenceRequest(t, s.HTTPAddr, "llama3.2:3b")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("inference: got status %d, want 200", resp.StatusCode)
	}
	if hits.Load() != 1 {
		t.Errorf("fake upstream hit count: got %d, want 1", hits.Load())
	}

	// Admin API shows the worker as READY.
	body := adminGet(t, s.AdminAddr, "/admin/workers")
	var workers []map[string]any
	if err := json.Unmarshal(body, &workers); err != nil {
		t.Fatalf("decode /admin/workers: %v", err)
	}
	if len(workers) != 1 {
		t.Fatalf("worker count: got %d, want 1", len(workers))
	}
	if workers[0]["state"] != "READY" {
		t.Errorf("worker state: got %v, want READY", workers[0]["state"])
	}
}

// TestIntegration_NoWorker_Returns503 verifies the router returns 503 when
// no worker is available for the requested model.
func TestIntegration_NoWorker_Returns503(t *testing.T) {
	s := newStack(t, 15*time.Second)

	resp := inferenceRequest(t, s.HTTPAddr, "llama3.2:3b")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status: got %d, want 503", resp.StatusCode)
	}
}

// TestIntegration_DeadWorkerDetection verifies that a worker which stops
// heartbeating is marked DEAD by the per-worker deadline timer.
func TestIntegration_DeadWorkerDetection(t *testing.T) {
	// Very short dead timeout so the test completes quickly.
	s := newStack(t, 150*time.Millisecond)
	client := workerClient(t, s.GRPCAddr)
	inferAddr, _ := fakeInferenceServer(t)

	// Register but do NOT send further heartbeats — worker will time out.
	if _, err := client.Register(t.Context(), &pb.WorkerInfo{
		WorkerId:     "silent-worker",
		Address:      inferAddr,
		ModelsLoaded: []string{"llama3.2:3b"},
	}); err != nil {
		t.Fatalf("register: %v", err)
	}

	waitForState(t, s.reg, "silent-worker", pb.WorkerState_DEAD, 2*time.Second)

	// Router must 503 — DEAD workers are not eligible.
	resp := inferenceRequest(t, s.HTTPAddr, "llama3.2:3b")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("post-death routing: got %d, want 503", resp.StatusCode)
	}
}

// TestIntegration_ScalerEvictsDeadWorker verifies that after the grace period
// the scaler removes the DEAD entry and fires OnWorkerEvicted.
func TestIntegration_ScalerEvictsDeadWorker(t *testing.T) {
	s := newStack(t, 150*time.Millisecond)

	evicted := make(chan string, 1)
	sc := scaler.New(s.reg,
		scaler.WithSweepInterval(50*time.Millisecond),
		scaler.WithDeadGrace(0),
		scaler.WithOnWorkerEvicted(func(id string, _ []string) { evicted <- id }),
	)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go sc.Run(ctx)

	client := workerClient(t, s.GRPCAddr)
	if _, err := client.Register(t.Context(), &pb.WorkerInfo{
		WorkerId:     "doomed",
		Address:      "127.0.0.1:1",
		ModelsLoaded: []string{"llama3.2:3b"},
	}); err != nil {
		t.Fatalf("register: %v", err)
	}

	// Wait for DEAD, then for eviction from the registry.
	waitForState(t, s.reg, "doomed", pb.WorkerState_DEAD, 2*time.Second)
	waitEvicted(t, s.reg, "doomed", 2*time.Second)

	// OnWorkerEvicted must have fired.
	select {
	case id := <-evicted:
		if id != "doomed" {
			t.Errorf("OnWorkerEvicted: got %s, want doomed", id)
		}
	case <-time.After(time.Second):
		t.Error("OnWorkerEvicted was not called after eviction")
	}
}

// TestIntegration_MultiWorkerDistribution verifies that two workers both
// receive traffic when requests arrive concurrently.
func TestIntegration_MultiWorkerDistribution(t *testing.T) {
	s := newStack(t, 15*time.Second)
	client := workerClient(t, s.GRPCAddr)

	addr0, hits0 := fakeInferenceServer(t)
	addr1, hits1 := fakeInferenceServer(t)

	registerReady(t, client, "w0", addr0, "llama3.2:3b")
	registerReady(t, client, "w1", addr1, "llama3.2:3b")

	for range 10 {
		resp := inferenceRequest(t, s.HTTPAddr, "llama3.2:3b")
		resp.Body.Close() //nolint:errcheck
	}

	if hits0.Load() == 0 || hits1.Load() == 0 {
		t.Errorf("both workers should receive traffic: w0=%d w1=%d", hits0.Load(), hits1.Load())
	}
}

// TestIntegration_AdminModelCatalog verifies /admin/models reflects the fleet.
func TestIntegration_AdminModelCatalog(t *testing.T) {
	s := newStack(t, 15*time.Second)
	client := workerClient(t, s.GRPCAddr)

	addr, _ := fakeInferenceServer(t)
	registerReady(t, client, "w1", addr, "llama3.2:3b")

	// w2 serves a different model but is still STARTING (no READY heartbeat).
	client.Register(t.Context(), &pb.WorkerInfo{ //nolint:errcheck
		WorkerId:     "w2",
		Address:      addr,
		ModelsLoaded: []string{"mistral:7b"},
	})

	body := adminGet(t, s.AdminAddr, "/admin/models")
	var models []map[string]any
	if err := json.Unmarshal(body, &models); err != nil {
		t.Fatalf("decode /admin/models: %v", err)
	}

	byModel := make(map[string]map[string]any)
	for _, m := range models {
		byModel[m["model"].(string)] = m
	}

	if _, ok := byModel["llama3.2:3b"]; !ok {
		t.Error("llama3.2:3b missing from model catalog")
	}
	if byModel["llama3.2:3b"]["healthy_workers"].(float64) != 1 {
		t.Errorf("llama3.2:3b healthy_workers: got %v, want 1", byModel["llama3.2:3b"]["healthy_workers"])
	}
	if _, ok := byModel["mistral:7b"]; !ok {
		t.Error("mistral:7b missing from model catalog")
	}
	if byModel["mistral:7b"]["healthy_workers"].(float64) != 0 {
		t.Errorf("mistral:7b healthy_workers: got %v, want 0 (still STARTING)", byModel["mistral:7b"]["healthy_workers"])
	}
}

// TestIntegration_MetricsEndpoint verifies /metrics is scrapeable and contains
// the core fleet metrics.
func TestIntegration_MetricsEndpoint(t *testing.T) {
	s := newStack(t, 15*time.Second)
	client := workerClient(t, s.GRPCAddr)

	addr, _ := fakeInferenceServer(t)
	registerReady(t, client, "w1", addr, "llama3.2:3b")

	// Send one request so the counter is non-zero.
	resp := inferenceRequest(t, s.HTTPAddr, "llama3.2:3b")
	resp.Body.Close() //nolint:errcheck

	body := string(adminGet(t, s.AdminAddr, "/metrics"))

	wantMetrics := []string{
		`llm_workers{state="ready"} 1`,
		`llm_model_healthy_workers{model="llama3.2:3b"} 1`,
		`llm_requests_total{model="llama3.2:3b",result="success"}`,
		`go_goroutines`,
	}
	for _, want := range wantMetrics {
		if !strings.Contains(body, want) {
			t.Errorf("metric %q not found in /metrics output", want)
		}
	}
}

// TestIntegration_AdminStats verifies /admin/stats fleet-wide counts.
func TestIntegration_AdminStats(t *testing.T) {
	s := newStack(t, 15*time.Second)
	client := workerClient(t, s.GRPCAddr)

	addr, _ := fakeInferenceServer(t)
	registerReady(t, client, "w1", addr, "llama3.2:3b")
	registerReady(t, client, "w2", addr, "llama3.2:3b")

	body := adminGet(t, s.AdminAddr, "/admin/stats")
	var stats map[string]any
	if err := json.Unmarshal(body, &stats); err != nil {
		t.Fatalf("decode /admin/stats: %v", err)
	}

	if stats["total"].(float64) != 2 {
		t.Errorf("total: got %v, want 2", stats["total"])
	}
	if stats["healthy"].(float64) != 2 {
		t.Errorf("healthy: got %v, want 2", stats["healthy"])
	}
}
