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

func makeWorker(id string) registry.WorkerEntry {
	return registry.WorkerEntry{
		Info:  &pb.WorkerInfo{WorkerId: id},
		Load:  &pb.LoadReport{},
		State: pb.WorkerState_READY,
	}
}

// zeroFlight returns 0 for every worker — used in tests that don't care about in-flight.
func zeroFlight(_ string) int64 { return 0 }

// ── Router integration tests ───────────────────────────────────────────────────

func TestRouter_ProxiesToWorker(t *testing.T) {
	upstream := fakeUpstream(t, `{"choices":[{"message":{"content":"hello"}}]}`)
	defer upstream.Close()

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

func TestRouter_DistributesAcrossWorkers(t *testing.T) {
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
	for range 10 {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
		rtr.ServeHTTP(httptest.NewRecorder(), req)
	}

	if hitCount[0] == 0 || hitCount[1] == 0 {
		t.Errorf("expected both workers to be hit, got %v", hitCount)
	}
}

// TestRouter_RetryOnWorkerFailure verifies that when the first worker is
// unreachable, the router automatically retries on the second worker and
// the client gets a successful response.
func TestRouter_RetryOnWorkerFailure(t *testing.T) {
	// Worker A: immediately close the connection — simulates a crashed worker.
	deadUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Hijack and close the connection without sending a response.
		conn, _, _ := w.(http.Hijacker).Hijack()
		conn.Close()
	}))
	defer deadUpstream.Close()

	// Worker B: healthy, returns a valid response.
	liveUpstream := fakeUpstream(t, `{"choices":[{"message":{"content":"recovered"}}]}`)
	defer liveUpstream.Close()

	reg := registry.New(15 * time.Second)
	registerReady(t, reg, "w-dead", strings.TrimPrefix(deadUpstream.URL, "http://"), "llama3.2:3b")
	registerReady(t, reg, "w-live", strings.TrimPrefix(liveUpstream.URL, "http://"), "llama3.2:3b")

	rtr := New(reg)
	// Pin the dead worker first by giving it a lower in-flight count
	// (both start at 0 so the counter will pick w-dead first on counter=1).
	// We just send the request and verify it eventually succeeds.
	body := `{"model":"llama3.2:3b","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	rtr.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status: got %d, want 200 (retry should have succeeded)", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "recovered") {
		t.Errorf("body: got %q, want 'recovered'", rec.Body.String())
	}
}

// ── pickWorker unit tests ──────────────────────────────────────────────────────

func TestPickWorker_PrefersLowestInFlight(t *testing.T) {
	workers := []registry.WorkerEntry{
		makeWorker("w-busy"),
		makeWorker("w-idle"),
	}
	inFlight := map[string]int64{"w-busy": 5, "w-idle": 1}
	getInFlight := func(id string) int64 { return inFlight[id] }

	var c atomic.Uint64
	got := pickWorker(workers, &c, getInFlight)
	if got.Info.WorkerId != "w-idle" {
		t.Errorf("expected w-idle (inFlight=1), got %s", got.Info.WorkerId)
	}
}

func TestPickWorker_TiedInFlight_RoundRobin(t *testing.T) {
	// Both workers have the same in-flight count — should distribute via round-robin.
	workers := []registry.WorkerEntry{
		makeWorker("w0"),
		makeWorker("w1"),
	}
	hits := map[string]int{}
	var c atomic.Uint64
	for range 10 {
		w := pickWorker(workers, &c, zeroFlight)
		hits[w.Info.WorkerId]++
	}
	if hits["w0"] == 0 || hits["w1"] == 0 {
		t.Errorf("expected both workers hit when tied, got %v", hits)
	}
}

func TestPickWorker_SingleWorker_AlwaysPicked(t *testing.T) {
	workers := []registry.WorkerEntry{makeWorker("only")}
	var c atomic.Uint64
	got := pickWorker(workers, &c, zeroFlight)
	if got.Info.WorkerId != "only" {
		t.Errorf("got %s, want only", got.Info.WorkerId)
	}
}

// ── CP11: cost-aware selection ─────────────────────────────────────────────────

func TestPickWorker_PrefersCheaperWhenInFlightTied(t *testing.T) {
	workers := []registry.WorkerEntry{
		{Info: &pb.WorkerInfo{WorkerId: "cloud", CostPerHour: 3.50}, Load: &pb.LoadReport{}, State: pb.WorkerState_READY},
		{Info: &pb.WorkerInfo{WorkerId: "local", CostPerHour: 0.00}, Load: &pb.LoadReport{}, State: pb.WorkerState_READY},
	}
	var c atomic.Uint64
	// Both workers have zero in-flight — cost is the deciding factor.
	for range 5 {
		got := pickWorker(workers, &c, zeroFlight)
		if got.Info.WorkerId != "local" {
			t.Errorf("expected local (cost=0), got %s (cost=%.2f)", got.Info.WorkerId, got.Info.CostPerHour)
		}
	}
}

func TestPickWorker_HigherLoadOverridesCost(t *testing.T) {
	// local is cheaper but has 10 in-flight; cloud is expensive but idle.
	// Routing should pick cloud because latency (in-flight) takes priority over cost.
	workers := []registry.WorkerEntry{
		{Info: &pb.WorkerInfo{WorkerId: "local", CostPerHour: 0.00}, Load: &pb.LoadReport{}, State: pb.WorkerState_READY},
		{Info: &pb.WorkerInfo{WorkerId: "cloud", CostPerHour: 3.50}, Load: &pb.LoadReport{}, State: pb.WorkerState_READY},
	}
	inFlight := map[string]int64{"local": 10, "cloud": 0}
	getInFlight := func(id string) int64 { return inFlight[id] }

	var c atomic.Uint64
	got := pickWorker(workers, &c, getInFlight)
	if got.Info.WorkerId != "cloud" {
		t.Errorf("expected cloud (in-flight=0 beats cost), got %s", got.Info.WorkerId)
	}
}

func TestPickWorker_SameCostAndInFlight_RoundRobin(t *testing.T) {
	workers := []registry.WorkerEntry{
		{Info: &pb.WorkerInfo{WorkerId: "w0", CostPerHour: 1.0}, Load: &pb.LoadReport{}, State: pb.WorkerState_READY},
		{Info: &pb.WorkerInfo{WorkerId: "w1", CostPerHour: 1.0}, Load: &pb.LoadReport{}, State: pb.WorkerState_READY},
	}
	hits := map[string]int{}
	var c atomic.Uint64
	for range 10 {
		got := pickWorker(workers, &c, zeroFlight)
		hits[got.Info.WorkerId]++
	}
	if hits["w0"] == 0 || hits["w1"] == 0 {
		t.Errorf("same cost + same in-flight should round-robin, got %v", hits)
	}
}
