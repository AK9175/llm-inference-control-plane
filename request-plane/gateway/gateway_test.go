package gateway

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/dispatcher"
	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

// liveSystem wires a queue + dispatcher + gateway against a fake upstream —
// the full Phase 2 pipeline minus the real Phase 1 router.
func liveSystem(t *testing.T, upstreamURL string, queueCap, concurrency int) *Gateway {
	t.Helper()
	q := queue.New(queueCap)
	d := dispatcher.New(q, upstreamURL, concurrency)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	return New(q, 2*time.Second)
}

func TestGateway_EnqueuesAndReturnsResult(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"content":"hello"}}]}`)
	}))
	t.Cleanup(upstream.Close)

	gw := liveSystem(t, upstream.URL, 4, 2)

	body := `{"model":"llama3.2:3b","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200, body=%s", rec.Code, rec.Body.String())
	}
	if !bytes.Contains(rec.Body.Bytes(), []byte("hello")) {
		t.Errorf("body: got %q, want it to contain 'hello'", rec.Body.String())
	}
}

func TestGateway_QueueFull_Returns503(t *testing.T) {
	// No dispatcher running — queue fills up and stays full.
	q := queue.New(1)
	q.TryPush(&queue.Request{ID: "blocker", ResultCh: make(chan queue.Result, 1)})

	gw := New(q, time.Second)

	body := `{"model":"m","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status: got %d, want 503", rec.Code)
	}
}

func TestGateway_MissingModel_Returns400(t *testing.T) {
	q := queue.New(4)
	gw := New(q, time.Second)

	body := `{"messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status: got %d, want 400", rec.Code)
	}
}

func TestGateway_NoDispatcher_TimesOut(t *testing.T) {
	// Queue has room, but nothing ever pops from it — request must time out.
	q := queue.New(4)
	gw := New(q, 100*time.Millisecond)

	body := `{"model":"m","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	start := time.Now()
	gw.ServeHTTP(rec, req)
	elapsed := time.Since(start)

	if rec.Code != http.StatusGatewayTimeout {
		t.Errorf("status: got %d, want 504", rec.Code)
	}
	if elapsed < 100*time.Millisecond {
		t.Errorf("returned too early: %s", elapsed)
	}
}

func TestGateway_UnknownPath_Returns404(t *testing.T) {
	q := queue.New(4)
	gw := New(q, time.Second)

	req := httptest.NewRequest(http.MethodGet, "/nope", nil)
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Errorf("status: got %d, want 404", rec.Code)
	}
}

func TestGateway_Healthz(t *testing.T) {
	q := queue.New(4)
	gw := New(q, time.Second)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status: got %d, want 200", rec.Code)
	}
}

// TestGateway_HighPriorityHeader_BypassesFullNormalLane verifies that a
// request with X-Priority: high still gets queued even when the normal
// lane is completely full — the two lanes have independent capacity.
func TestGateway_HighPriorityHeader_BypassesFullNormalLane(t *testing.T) {
	q := queue.New(1) // capacity 1 per lane
	q.TryPush(&queue.Request{ID: "blocker", Priority: queue.PriorityNormal, ResultCh: make(chan queue.Result, 1)})

	// No dispatcher running — this will time out, but the only thing we're
	// asserting is that TryPush succeeded (no 503), so a short timeout is fine.
	gw := New(q, 50*time.Millisecond)

	body := `{"model":"m","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("X-Priority", "high")
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code == http.StatusServiceUnavailable {
		t.Error("high priority request got 503 — should have used the separate high-priority lane")
	}
}
