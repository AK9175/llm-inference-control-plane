package metrics

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
	"github.com/atharva/llm-serving-platform/control-plane/router"
	pb "github.com/atharva/llm-serving-platform/proto"
)

func setup(t *testing.T) (*registry.Registry, *router.Router, http.Handler, func(string, string)) {
	t.Helper()
	reg := registry.New(15 * time.Second)
	rtr := router.New(reg)
	h, hook := Setup(reg, rtr)
	rtr.SetRequestHook(hook)
	return reg, rtr, h, hook
}

func scrape(t *testing.T, h http.Handler) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("metrics status: got %d, want 200", rec.Code)
	}
	body, _ := io.ReadAll(rec.Body)
	return string(body)
}

func assertMetric(t *testing.T, body, want string) {
	t.Helper()
	if !strings.Contains(body, want) {
		t.Errorf("metric %q not found in output:\n%s", want, body)
	}
}

func TestMetrics_WorkerStateCounts(t *testing.T) {
	reg, _, h, _ := setup(t)
	ctx := t.Context()

	reg.Register(ctx, &pb.WorkerInfo{WorkerId: "w1", Address: "x:1", ModelsLoaded: []string{"llama3.2:3b"}}) //nolint:errcheck
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "w1", State: pb.WorkerState_READY})                    //nolint:errcheck

	body := scrape(t, h)
	assertMetric(t, body, `llm_workers{state="ready"} 1`)
	assertMetric(t, body, `llm_workers{state="starting"} 0`)
}

func TestMetrics_ModelHealthyWorkers(t *testing.T) {
	reg, _, h, _ := setup(t)
	ctx := t.Context()

	reg.Register(ctx, &pb.WorkerInfo{WorkerId: "w1", Address: "x:1", ModelsLoaded: []string{"llama3.2:3b"}}) //nolint:errcheck
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "w1", State: pb.WorkerState_READY})                    //nolint:errcheck

	body := scrape(t, h)
	assertMetric(t, body, `llm_model_healthy_workers{model="llama3.2:3b"} 1`)
}

func TestMetrics_FleetCostPerHour(t *testing.T) {
	reg, _, h, _ := setup(t)
	ctx := t.Context()

	reg.Register(ctx, &pb.WorkerInfo{ //nolint:errcheck
		WorkerId: "w1", Address: "x:1", ModelsLoaded: []string{"m"}, CostPerHour: 2.50,
	})
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "w1", State: pb.WorkerState_READY}) //nolint:errcheck

	body := scrape(t, h)
	assertMetric(t, body, "llm_fleet_cost_per_hour 2.5")
}

func TestMetrics_RequestsTotal(t *testing.T) {
	_, _, h, hook := setup(t)

	hook("llama3.2:3b", "success")
	hook("llama3.2:3b", "success")
	hook("llama3.2:3b", "error")

	body := scrape(t, h)
	assertMetric(t, body, `llm_requests_total{model="llama3.2:3b",result="success"} 2`)
	assertMetric(t, body, `llm_requests_total{model="llama3.2:3b",result="error"} 1`)
}

func TestMetrics_GoCollectorPresent(t *testing.T) {
	_, _, h, _ := setup(t)
	body := scrape(t, h)
	// Standard Go runtime metrics are always present.
	assertMetric(t, body, "go_goroutines")
}
