package router

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
	pb "github.com/atharva/llm-serving-platform/proto"
)

// fakeUpstream spins up an httptest.Server that returns a fixed response.
// We register its address as the worker's inference endpoint.
func fakeUpstream(t *testing.T, response string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, response)
	}))
}

// registerReady adds a worker to the registry and advances it to READY state.
func registerReady(t *testing.T, reg *registry.Registry, id, addr, model string) {
	t.Helper()
	ctx := t.Context()
	reg.Register(ctx, &pb.WorkerInfo{ //nolint:errcheck
		WorkerId:     id,
		Address:      addr,
		ModelsLoaded: []string{model},
	})
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{ //nolint:errcheck
		WorkerId: id,
		State:    pb.WorkerState_READY,
	})
}

func TestRouter_ProxiesToWorker(t *testing.T) {
	upstream := fakeUpstream(t, `{"choices":[{"message":{"content":"hello"}}]}`)
	defer upstream.Close()

	// upstream.URL is "http://127.0.0.1:<port>" — strip the scheme for Address field
	addr := strings.TrimPrefix(upstream.URL, "http://")

	reg := registry.New(15 * time.Second)
	registerReady(t, reg, "w1", addr, "llama3.2:3b")

	rtr := New(reg)
	body := `{"model":"llama3.2:3b","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	rtr.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status: got %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "hello") {
		t.Errorf("body: got %q, want response containing 'hello'", rec.Body.String())
	}
}

func TestRouter_NoWorkers_Returns503(t *testing.T) {
	reg := registry.New(15 * time.Second)
	rtr := New(reg)

	body := `{"model":"llama3.2:3b","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	rtr.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status: got %d, want 503", rec.Code)
	}
}

func TestRouter_MissingModel_Returns400(t *testing.T) {
	reg := registry.New(15 * time.Second)
	rtr := New(reg)

	body := `{"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	rtr.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status: got %d, want 400", rec.Code)
	}
}

// ── pickWorker unit tests ──────────────────────────────────────────────────────

func makeWorker(id string, queue uint32, latencyMs float64) registry.WorkerEntry {
	return registry.WorkerEntry{
		Info:  &pb.WorkerInfo{WorkerId: id},
		Load:  &pb.LoadReport{QueueDepth: queue, AvgLatencyMs: latencyMs},
		State: pb.WorkerState_READY,
	}
}

func TestPickWorker_LeastQueue(t *testing.T) {
	workers := []registry.WorkerEntry{
		makeWorker("w-busy", 5, 200),
		makeWorker("w-idle", 1, 300),
	}
	var c atomic.Uint64
	got := pickWorker(workers, &c)
	if got.Info.WorkerId != "w-idle" {
		t.Errorf("expected w-idle (queue=1), got %s", got.Info.WorkerId)
	}
}

func TestPickWorker_LatencyTiebreak(t *testing.T) {
	// Both queue=0 but different latencies — pick the faster one.
	workers := []registry.WorkerEntry{
		makeWorker("w-slow", 0, 300),
		makeWorker("w-fast", 0, 100),
	}
	var c atomic.Uint64
	got := pickWorker(workers, &c)
	if got.Info.WorkerId != "w-fast" {
		t.Errorf("expected w-fast (latency=100ms), got %s", got.Info.WorkerId)
	}
}

func TestPickWorker_AllZero_RoundRobin(t *testing.T) {
	// No load data yet — should distribute via round-robin, not always pick index 0.
	workers := []registry.WorkerEntry{
		makeWorker("w0", 0, 0),
		makeWorker("w1", 0, 0),
	}
	hits := map[string]int{}
	var c atomic.Uint64
	for range 10 {
		w := pickWorker(workers, &c)
		hits[w.Info.WorkerId]++
	}
	if hits["w0"] == 0 || hits["w1"] == 0 {
		t.Errorf("expected both workers hit under round-robin, got %v", hits)
	}
}

func TestRouter_RoundRobin_DistributesAcrossWorkers(t *testing.T) {
	// Spin up two fake upstreams that record which one was hit.
	hitCount := [2]int{}
	upstreams := [2]*httptest.Server{}
	for i := range upstreams {
		i := i
		upstreams[i] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			hitCount[i]++
			fmt.Fprint(w, `{}`)
		}))
		defer upstreams[i].Close()
	}

	reg := registry.New(15 * time.Second)
	for i, us := range upstreams {
		addr := strings.TrimPrefix(us.URL, "http://")
		registerReady(t, reg, fmt.Sprintf("w%d", i), addr, "llama3.2:3b")
	}

	rtr := New(reg)
	body := `{"model":"llama3.2:3b","messages":[]}`

	// Send 10 requests — each worker should get ~5 (round-robin alternates).
	for range 10 {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
		rtr.ServeHTTP(httptest.NewRecorder(), req)
	}

	if hitCount[0] == 0 || hitCount[1] == 0 {
		t.Errorf("expected both workers to be hit, got %v", hitCount)
	}
	if hitCount[0]+hitCount[1] != 10 {
		t.Errorf("expected 10 total hits, got %d", hitCount[0]+hitCount[1])
	}
}
