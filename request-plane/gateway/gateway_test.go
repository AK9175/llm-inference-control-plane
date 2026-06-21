package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/backpressure"
	"github.com/atharva/llm-serving-platform/request-plane/dispatcher"
	"github.com/atharva/llm-serving-platform/request-plane/queue"
	"github.com/atharva/llm-serving-platform/request-plane/slo"
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

// ── CP18: SLO headers ────────────────────────────────────────────────────────

func TestGateway_WithSLO_SetsEstimatedAndActualWaitHeaders(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{}`)
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := dispatcher.New(q, upstream.URL, 2)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	tracker := slo.NewLatencyTracker(time.Second)
	estimator := slo.NewEstimator(tracker, 2)
	gw := New(q, 2*time.Second, WithSLO(tracker, estimator))

	body := `{"model":"m","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}
	if rec.Header().Get("X-Estimated-Wait-Ms") == "" {
		t.Error("X-Estimated-Wait-Ms header missing")
	}
	if rec.Header().Get("X-Actual-Wait-Ms") == "" {
		t.Error("X-Actual-Wait-Ms header missing")
	}
}

func TestGateway_WithoutSLO_NoHeaders(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{}`)
	}))
	t.Cleanup(upstream.Close)

	gw := liveSystem(t, upstream.URL, 4, 1) // New(q, timeout) with no SLO option

	body := `{"model":"m","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Header().Get("X-Estimated-Wait-Ms") != "" {
		t.Error("X-Estimated-Wait-Ms should be absent when SLO is not configured")
	}
	if rec.Header().Get("X-Actual-Wait-Ms") != "" {
		t.Error("X-Actual-Wait-Ms should be absent when SLO is not configured")
	}
}

func TestGateway_HighPriorityRequest_LowerEstimateThanQueuedNormal(t *testing.T) {
	tracker := slo.NewLatencyTracker(time.Second)
	estimator := slo.NewEstimator(tracker, 1)

	q := queue.New(10)
	// Pre-load 3 normal-priority requests so a new high-priority request has
	// nothing ahead of it, while a new normal-priority request would have 3.
	for i := range 3 {
		q.TryPush(&queue.Request{ID: fmt.Sprintf("n%d", i), Priority: queue.PriorityNormal, ResultCh: make(chan queue.Result, 1)})
	}

	highReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"m","messages":[]}`))
	highReq.Header.Set("X-Priority", "high")
	highRec := httptest.NewRecorder()

	// No dispatcher running — request will time out, but we only care about
	// the estimate header set before the wait begins.
	gw := New(q, 20*time.Millisecond, WithSLO(tracker, estimator))
	gw.ServeHTTP(highRec, highReq)

	estimateMs := highRec.Header().Get("X-Estimated-Wait-Ms")
	if estimateMs != "0" {
		t.Errorf("high priority with nothing ahead: got estimate %sms, want 0", estimateMs)
	}
}

// ── CP19: backpressure ───────────────────────────────────────────────────────

func TestGateway_Backpressure_RejectsWhenEstimateExceedsThreshold(t *testing.T) {
	tracker := slo.NewLatencyTracker(20 * time.Second) // deliberately slow fallback
	estimator := slo.NewEstimator(tracker, 1)
	policy := backpressure.New(map[queue.Priority]time.Duration{
		queue.PriorityNormal: 5 * time.Second,
	})

	q := queue.New(4)
	// Pre-load enough normal requests that AheadOf pushes the estimate past 5s:
	// 1 ahead * 1 concurrency * 20s fallback = 20s estimate, well over the 5s cap.
	q.TryPush(&queue.Request{ID: "blocker", Priority: queue.PriorityNormal, ResultCh: make(chan queue.Result, 1)})

	gw := New(q, time.Second, WithSLO(tracker, estimator), WithBackpressure(policy))

	body := `{"model":"m","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status: got %d, want 503 (estimated wait should exceed threshold)", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("Retry-After header missing on backpressure rejection")
	}
}

func TestGateway_Backpressure_AcceptsWhenEstimateWithinThreshold(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{}`)
	}))
	t.Cleanup(upstream.Close)

	tracker := slo.NewLatencyTracker(100 * time.Millisecond) // fast fallback
	estimator := slo.NewEstimator(tracker, 4)
	policy := backpressure.New(backpressure.DefaultThresholds())

	q := queue.New(4)
	d := dispatcher.New(q, upstream.URL, 4)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	gw := New(q, time.Second, WithSLO(tracker, estimator), WithBackpressure(policy))

	body := `{"model":"m","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status: got %d, want 200 (fast fallback latency, nothing ahead — should be admitted)", rec.Code)
	}
}

func TestGateway_Backpressure_WithoutEstimator_FailsOpen(t *testing.T) {
	// Backpressure configured but SLO is not — gateway must not block
	// admission since there's no estimate to check against.
	policy := backpressure.New(map[queue.Priority]time.Duration{queue.PriorityNormal: time.Nanosecond})
	q := queue.New(4)
	gw := New(q, 50*time.Millisecond, WithBackpressure(policy)) // no WithSLO

	body := `{"model":"m","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code == http.StatusServiceUnavailable {
		t.Error("backpressure without an estimator should fail open (admit), not reject")
	}
}

func TestGateway_Backpressure_HighPriorityStricterThreshold(t *testing.T) {
	tracker := slo.NewLatencyTracker(8 * time.Second)
	estimator := slo.NewEstimator(tracker, 1)
	policy := backpressure.New(backpressure.DefaultThresholds()) // high=5s, low=60s

	q := queue.New(4)
	// Pre-load one high-priority request so a NEW request has something
	// ahead of it — with an empty queue, AheadOf is always 0 and the
	// estimate would be 0 regardless of latency, defeating this test.
	q.TryPush(&queue.Request{ID: "blocker", Priority: queue.PriorityHigh, ResultCh: make(chan queue.Result, 1)})

	gw := New(q, 50*time.Millisecond, WithSLO(tracker, estimator), WithBackpressure(policy))

	// 1 ahead / 1 concurrency * 8s fallback latency = 8s estimate.
	// Exceeds the high-priority 5s threshold but is fine for low's 60s.
	highReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"m","messages":[]}`))
	highReq.Header.Set("X-Priority", "high")
	highRec := httptest.NewRecorder()
	gw.ServeHTTP(highRec, highReq)

	if highRec.Code != http.StatusServiceUnavailable {
		t.Errorf("high priority: got %d, want 503 (8s estimate exceeds 5s threshold)", highRec.Code)
	}

	lowReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"m","messages":[]}`))
	lowReq.Header.Set("X-Priority", "low")
	lowRec := httptest.NewRecorder()
	gw.ServeHTTP(lowRec, lowReq)

	if lowRec.Code == http.StatusServiceUnavailable {
		t.Error("low priority: 8s estimate should be within the 60s threshold, got 503")
	}
}

// ── CP20: served-model header ────────────────────────────────────────────────

func TestGateway_SetsServedModelHeader_OnFallback(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed struct{ Model string }
		json.Unmarshal(body, &parsed) //nolint:errcheck
		if parsed.Model == "big" {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		fmt.Fprint(w, `{}`)
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := dispatcher.New(q, upstream.URL, 1,
		dispatcher.WithFallbacks(map[string][]string{"big": {"small"}}),
	)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	gw := New(q, time.Second)

	body := `{"model":"big","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}
	if got := rec.Header().Get("X-Served-Model"); got != "small" {
		t.Errorf("X-Served-Model: got %q, want small", got)
	}
}

// ── CP21: streaming ──────────────────────────────────────────────────────────

func TestGateway_Stream_ForwardsChunksToClient(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		for _, c := range []string{"data: a\n\n", "data: b\n\n", "data: [DONE]\n\n"} {
			fmt.Fprint(w, c)
			flusher.Flush()
		}
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := dispatcher.New(q, upstream.URL, 1)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	gw := New(q, 2*time.Second)

	body := `{"model":"m","stream":true,"messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("Content-Type: got %q, want text/event-stream", got)
	}
	want := "data: a\n\ndata: b\n\ndata: [DONE]\n\n"
	if rec.Body.String() != want {
		t.Errorf("body: got %q, want %q", rec.Body.String(), want)
	}
}

// TestGateway_Stream_DefaultsContentTypeWhenUpstreamOmitsIt exercises
// streamResponse directly rather than through a real httptest.Server —
// Go's net/http always sniffs and fills in SOME Content-Type on the wire,
// so an end-to-end test can never observe a truly empty header. The
// fallback logic itself is still worth covering as a unit.
func TestGateway_Stream_DefaultsContentTypeWhenUpstreamOmitsIt(t *testing.T) {
	q := queue.New(4)
	gw := New(q, time.Second)

	req := &queue.Request{ID: "r1", Chunks: make(chan []byte, 1)}
	close(req.Chunks)

	rec := httptest.NewRecorder()
	gw.streamResponse(rec, req, queue.Result{StatusCode: http.StatusOK, ContentType: ""})

	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("Content-Type: got %q, want text/event-stream (default when upstream sends none)", got)
	}
}

func TestGateway_Stream_ServedModelHeaderOnFallback(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed struct{ Model string }
		json.Unmarshal(body, &parsed) //nolint:errcheck
		if parsed.Model == "big" {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		fmt.Fprint(w, "data: ok\n\n")
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := dispatcher.New(q, upstream.URL, 1,
		dispatcher.WithFallbacks(map[string][]string{"big": {"small"}}),
	)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	gw := New(q, 2*time.Second)

	body := `{"model":"big","stream":true,"messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if got := rec.Header().Get("X-Served-Model"); got != "small" {
		t.Errorf("X-Served-Model: got %q, want small", got)
	}
}

func TestGateway_Stream_WaitTimeoutOnlyAppliesBeforeFirstByte(t *testing.T) {
	// Upstream takes longer than waitTimeout to send its FIRST byte —
	// must time out, since the timeout bounds time-to-first-signal.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(150 * time.Millisecond)
		fmt.Fprint(w, "data: late\n\n")
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := dispatcher.New(q, upstream.URL, 1)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	gw := New(q, 50*time.Millisecond) // shorter than upstream's delay

	body := `{"model":"m","stream":true,"messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusGatewayTimeout {
		t.Errorf("status: got %d, want 504 (upstream never sent its first byte in time)", rec.Code)
	}
}
