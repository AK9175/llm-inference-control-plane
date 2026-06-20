package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
	"github.com/atharva/llm-serving-platform/control-plane/router"
	pb "github.com/atharva/llm-serving-platform/proto"
)

func setupAdmin(t *testing.T) (http.Handler, *registry.Registry, *router.Router) {
	t.Helper()
	reg := registry.New(15 * time.Second)
	rtr := router.New(reg)
	return NewAdminHandler(reg, rtr, http.NotFoundHandler()), reg, rtr
}

func registerWorker(t *testing.T, reg *registry.Registry, id, model string) {
	t.Helper()
	ctx := t.Context()
	reg.Register(ctx, &pb.WorkerInfo{ //nolint:errcheck
		WorkerId:     id,
		Address:      "localhost:9999",
		ModelsLoaded: []string{model},
		Backend:      "ollama",
		Provider:     "local",
		Hardware:     "cpu",
	})
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{ //nolint:errcheck
		WorkerId: id,
		State:    pb.WorkerState_READY,
	})
}

func TestAdminListWorkers(t *testing.T) {
	h, reg, _ := setupAdmin(t)
	registerWorker(t, reg, "w1", "llama3.2:3b")
	registerWorker(t, reg, "w2", "llama3.2:3b")

	req := httptest.NewRequest(http.MethodGet, "/admin/workers", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}

	var workers []workerResp
	if err := json.NewDecoder(rec.Body).Decode(&workers); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(workers) != 2 {
		t.Errorf("got %d workers, want 2", len(workers))
	}
}

func TestAdminGetWorker(t *testing.T) {
	h, reg, _ := setupAdmin(t)
	registerWorker(t, reg, "w1", "llama3.2:3b")

	req := httptest.NewRequest(http.MethodGet, "/admin/workers/w1", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}

	var w workerResp
	if err := json.NewDecoder(rec.Body).Decode(&w); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if w.ID != "w1" {
		t.Errorf("id: got %s, want w1", w.ID)
	}
	if w.State != "READY" {
		t.Errorf("state: got %s, want READY", w.State)
	}
}

func TestAdminGetWorker_NotFound(t *testing.T) {
	h, _, _ := setupAdmin(t)

	req := httptest.NewRequest(http.MethodGet, "/admin/workers/ghost", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Errorf("status: got %d, want 404", rec.Code)
	}
}

func TestAdminDrainWorker(t *testing.T) {
	h, reg, _ := setupAdmin(t)
	registerWorker(t, reg, "w1", "llama3.2:3b")

	req := httptest.NewRequest(http.MethodPost, "/admin/workers/w1/drain", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}

	// Verify DrainRequested is now set on the worker entry.
	entry, _ := reg.GetWorker("w1")
	if !entry.DrainRequested {
		t.Error("expected DrainRequested=true after drain call")
	}
}

func TestAdminStats(t *testing.T) {
	h, reg, _ := setupAdmin(t)
	registerWorker(t, reg, "w1", "llama3.2:3b")
	registerWorker(t, reg, "w2", "llama3.2:3b")

	// Manually advance w2 to BUSY so stats show mixed states.
	reg.Heartbeat(t.Context(), &pb.HeartbeatRequest{ //nolint:errcheck
		WorkerId: "w2",
		State:    pb.WorkerState_BUSY,
	})

	req := httptest.NewRequest(http.MethodGet, "/admin/stats", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}

	var s statsResp
	if err := json.NewDecoder(rec.Body).Decode(&s); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if s.Total != 2 {
		t.Errorf("total: got %d, want 2", s.Total)
	}
	if s.Healthy != 2 {
		t.Errorf("healthy: got %d, want 2 (READY+BUSY both count)", s.Healthy)
	}
}

// ── CP13: model catalog ────────────────────────────────────────────────────────

func TestAdminListModels(t *testing.T) {
	h, reg, _ := setupAdmin(t)
	ctx := t.Context()

	// w1: llama3.2:3b only — READY
	reg.Register(ctx, &pb.WorkerInfo{WorkerId: "w1", Address: "x:1", ModelsLoaded: []string{"llama3.2:3b"}}) //nolint:errcheck
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "w1", State: pb.WorkerState_READY})                    //nolint:errcheck

	// w2: llama3.2:3b + mistral:7b — READY
	reg.Register(ctx, &pb.WorkerInfo{WorkerId: "w2", Address: "x:2", ModelsLoaded: []string{"llama3.2:3b", "mistral:7b"}}) //nolint:errcheck
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "w2", State: pb.WorkerState_READY})                                   //nolint:errcheck

	// w3: mistral:7b — still STARTING (not healthy)
	reg.Register(ctx, &pb.WorkerInfo{WorkerId: "w3", Address: "x:3", ModelsLoaded: []string{"mistral:7b"}}) //nolint:errcheck

	req := httptest.NewRequest(http.MethodGet, "/admin/models", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}

	var models []modelResp
	if err := json.NewDecoder(rec.Body).Decode(&models); err != nil {
		t.Fatalf("decode: %v", err)
	}

	byModel := make(map[string]modelResp)
	for _, m := range models {
		byModel[m.Model] = m
	}

	llama, ok := byModel["llama3.2:3b"]
	if !ok {
		t.Fatal("llama3.2:3b missing from model catalog")
	}
	if llama.TotalWorkers != 2 {
		t.Errorf("llama3.2:3b total_workers: got %d, want 2", llama.TotalWorkers)
	}
	if llama.HealthyWorkers != 2 {
		t.Errorf("llama3.2:3b healthy_workers: got %d, want 2", llama.HealthyWorkers)
	}

	mistral, ok := byModel["mistral:7b"]
	if !ok {
		t.Fatal("mistral:7b missing from model catalog")
	}
	if mistral.TotalWorkers != 2 {
		t.Errorf("mistral:7b total_workers: got %d, want 2", mistral.TotalWorkers)
	}
	if mistral.HealthyWorkers != 1 {
		t.Errorf("mistral:7b healthy_workers: got %d, want 1 (w3 is STARTING)", mistral.HealthyWorkers)
	}
}

func TestAdminStats_FleetCostPerHour(t *testing.T) {
	h, reg, _ := setupAdmin(t)

	// Register two healthy workers with different costs.
	ctx := t.Context()
	reg.Register(ctx, &pb.WorkerInfo{ //nolint:errcheck
		WorkerId: "local", Address: "localhost:9", ModelsLoaded: []string{"m"},
		CostPerHour: 0.0,
	})
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "local", State: pb.WorkerState_READY}) //nolint:errcheck

	reg.Register(ctx, &pb.WorkerInfo{ //nolint:errcheck
		WorkerId: "cloud", Address: "localhost:10", ModelsLoaded: []string{"m"},
		CostPerHour: 2.50,
	})
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "cloud", State: pb.WorkerState_READY}) //nolint:errcheck

	req := httptest.NewRequest(http.MethodGet, "/admin/stats", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}
	var s statsResp
	if err := json.NewDecoder(rec.Body).Decode(&s); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if s.FleetCostPerHour != 2.50 {
		t.Errorf("fleet_cost_per_hour: got %.2f, want 2.50 (local=0 + cloud=2.50)", s.FleetCostPerHour)
	}
}

func TestAdminListWorkers_ShowsInFlight(t *testing.T) {
	// Slow upstream — blocks until we explicitly release it.
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-release
		fmt.Fprint(w, `{}`)
	}))
	defer upstream.Close()

	reg := registry.New(15 * time.Second)
	rtr := router.New(reg)
	h := NewAdminHandler(reg, rtr, http.NotFoundHandler())

	addr := strings.TrimPrefix(upstream.URL, "http://")
	reg.Register(t.Context(), &pb.WorkerInfo{ //nolint:errcheck
		WorkerId: "w1", Address: addr, ModelsLoaded: []string{"m"},
	})
	reg.Heartbeat(t.Context(), &pb.HeartbeatRequest{ //nolint:errcheck
		WorkerId: "w1", State: pb.WorkerState_READY,
	})

	// Fire an inference request in the background — it blocks inside the upstream.
	done := make(chan struct{})
	go func() {
		defer close(done)
		body := `{"model":"m","messages":[]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(body))
		rtr.ServeHTTP(httptest.NewRecorder(), req)
	}()

	// Give the goroutine time to reach the upstream and increment in-flight.
	time.Sleep(50 * time.Millisecond)

	req := httptest.NewRequest(http.MethodGet, "/admin/workers", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	var workers []workerResp
	json.NewDecoder(rec.Body).Decode(&workers) //nolint:errcheck
	if len(workers) == 1 && workers[0].InFlight != 1 {
		t.Errorf("in_flight: got %d, want 1", workers[0].InFlight)
	}

	// Release the upstream first, then wait for the goroutine to finish.
	// Closing release before <-done avoids deadlock: done only closes once
	// the goroutine gets a response from the upstream.
	close(release)
	<-done
}
