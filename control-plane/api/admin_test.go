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
	return NewAdminHandler(reg, rtr), reg, rtr
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
	h := NewAdminHandler(reg, rtr)

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
