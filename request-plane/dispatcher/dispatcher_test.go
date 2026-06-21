package dispatcher

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

func fakeUpstream(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestDispatcher_ForwardsToUpstream(t *testing.T) {
	upstream := fakeUpstream(t, http.StatusOK, `{"choices":[{"message":{"content":"hi"}}]}`)

	q := queue.New(4)
	d := New(q, upstream.URL, 1)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go d.Run(ctx)

	req := &queue.Request{
		ID:       "r1",
		Body:     []byte(`{"model":"llama3.2:3b","messages":[]}`),
		ResultCh: make(chan queue.Result, 1),
	}
	q.TryPush(req)

	select {
	case result := <-req.ResultCh:
		if result.Err != nil {
			t.Fatalf("unexpected error: %v", result.Err)
		}
		if result.StatusCode != http.StatusOK {
			t.Errorf("status: got %d, want 200", result.StatusCode)
		}
		if string(result.Body) == "" {
			t.Error("body is empty")
		}
	case <-time.After(time.Second):
		t.Fatal("dispatcher did not deliver a result")
	}
}

func TestDispatcher_PropagatesUpstreamUnreachable(t *testing.T) {
	q := queue.New(4)
	// Point at a port nothing is listening on.
	d := New(q, "http://127.0.0.1:1", 1)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go d.Run(ctx)

	req := &queue.Request{
		ID:       "r1",
		Body:     []byte(`{"model":"m","messages":[]}`),
		ResultCh: make(chan queue.Result, 1),
	}
	q.TryPush(req)

	select {
	case result := <-req.ResultCh:
		if result.Err == nil {
			t.Error("expected error for unreachable upstream, got nil")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("dispatcher did not deliver a result")
	}
}

func TestDispatcher_MultipleWorkers_AllRequestsServed(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		fmt.Fprint(w, `{}`)
	}))
	t.Cleanup(srv.Close)

	q := queue.New(20)
	d := New(q, srv.URL, 4)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go d.Run(ctx)

	const n = 10
	chans := make([]chan queue.Result, n)
	for i := range n {
		ch := make(chan queue.Result, 1)
		chans[i] = ch
		q.TryPush(&queue.Request{
			ID:       fmt.Sprintf("r%d", i),
			Body:     []byte(`{"model":"m","messages":[]}`),
			ResultCh: ch,
		})
	}

	for i, ch := range chans {
		select {
		case result := <-ch:
			if result.Err != nil {
				t.Errorf("request %d: unexpected error %v", i, result.Err)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("request %d: no result delivered", i)
		}
	}

	if hits.Load() != n {
		t.Errorf("upstream hits: got %d, want %d", hits.Load(), n)
	}
}

func TestDispatcher_StopsOnContextCancel(t *testing.T) {
	q := queue.New(4)
	d := New(q, "http://127.0.0.1:1", 2)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		d.Run(ctx)
	}()

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Error("dispatcher did not stop after context cancel")
	}
}

// ── CP20: retry + fallback ─────────────────────────────────────────────────────

func TestSetModel_RewritesOnlyModelField(t *testing.T) {
	body := []byte(`{"model":"old-model","messages":[{"role":"user","content":"hi"}],"temperature":0.7}`)
	got, err := setModel(body, "new-model")
	if err != nil {
		t.Fatalf("setModel: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if parsed["model"] != "new-model" {
		t.Errorf("model: got %v, want new-model", parsed["model"])
	}
	if parsed["temperature"] != 0.7 {
		t.Errorf("temperature field lost: got %v, want 0.7", parsed["temperature"])
	}
	if _, ok := parsed["messages"]; !ok {
		t.Error("messages field lost during rewrite")
	}
}

func TestDispatcher_RetriesSameModelOnTransientFailure_ThenSucceeds(t *testing.T) {
	var calls atomic.Int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable) // fail once
			return
		}
		fmt.Fprint(w, `{"ok":true}`) // succeed on retry
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := New(q, upstream.URL, 1, WithMaxAttempts(3))
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	req := &queue.Request{
		ID:       "r1",
		Model:    "m",
		Body:     []byte(`{"model":"m","messages":[]}`),
		ResultCh: make(chan queue.Result, 1),
	}
	q.TryPush(req)

	select {
	case result := <-req.ResultCh:
		if result.StatusCode != http.StatusOK {
			t.Errorf("status: got %d, want 200 (retry should have succeeded)", result.StatusCode)
		}
		if calls.Load() != 2 {
			t.Errorf("upstream calls: got %d, want 2 (1 failure + 1 retry)", calls.Load())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no result delivered")
	}
}

func TestDispatcher_FallsBackToAlternateModel(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed struct{ Model string }
		json.Unmarshal(body, &parsed) //nolint:errcheck

		if parsed.Model == "big-model" {
			w.WriteHeader(http.StatusServiceUnavailable) // big-model never has capacity
			return
		}
		fmt.Fprintf(w, `{"served_by":%q}`, parsed.Model)
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := New(q, upstream.URL, 1,
		WithMaxAttempts(1),
		WithFallbacks(map[string][]string{"big-model": {"small-model"}}),
	)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	req := &queue.Request{
		ID:       "r1",
		Model:    "big-model",
		Body:     []byte(`{"model":"big-model","messages":[]}`),
		ResultCh: make(chan queue.Result, 1),
	}
	q.TryPush(req)

	select {
	case result := <-req.ResultCh:
		if result.StatusCode != http.StatusOK {
			t.Fatalf("status: got %d, want 200 (fallback should have succeeded)", result.StatusCode)
		}
		if result.ServedModel != "small-model" {
			t.Errorf("ServedModel: got %s, want small-model", result.ServedModel)
		}
		if !bytes.Contains(result.Body, []byte("small-model")) {
			t.Errorf("body should reflect fallback model, got %s", result.Body)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no result delivered")
	}
}

func TestDispatcher_NonRetryableStatus_NoFallbackAttempted(t *testing.T) {
	var calls atomic.Int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest) // not retryable
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := New(q, upstream.URL, 1,
		WithMaxAttempts(3),
		WithFallbacks(map[string][]string{"m": {"fallback-model"}}),
	)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	req := &queue.Request{
		ID:       "r1",
		Model:    "m",
		Body:     []byte(`{"model":"m","messages":[]}`),
		ResultCh: make(chan queue.Result, 1),
	}
	q.TryPush(req)

	select {
	case result := <-req.ResultCh:
		if result.StatusCode != http.StatusBadRequest {
			t.Errorf("status: got %d, want 400", result.StatusCode)
		}
		if calls.Load() != 1 {
			t.Errorf("upstream calls: got %d, want 1 (400 must not retry or fall back)", calls.Load())
		}
	case <-time.After(time.Second):
		t.Fatal("no result delivered")
	}
}

func TestDispatcher_ExhaustsFallbackChain_ReturnsLastFailure(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable) // every model fails
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := New(q, upstream.URL, 1,
		WithMaxAttempts(1),
		WithFallbacks(map[string][]string{"m": {"fallback-1", "fallback-2"}}),
	)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	req := &queue.Request{
		ID:       "r1",
		Model:    "m",
		Body:     []byte(`{"model":"m","messages":[]}`),
		ResultCh: make(chan queue.Result, 1),
	}
	q.TryPush(req)

	select {
	case result := <-req.ResultCh:
		if result.StatusCode != http.StatusServiceUnavailable {
			t.Errorf("status: got %d, want 503 (final failure after exhausting all fallbacks)", result.StatusCode)
		}
		if result.ServedModel != "fallback-2" {
			t.Errorf("ServedModel: got %s, want fallback-2 (last attempted model)", result.ServedModel)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no result delivered")
	}
}

func TestDispatcher_NoFallbackConfigured_SingleModelOnly(t *testing.T) {
	var calls atomic.Int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(upstream.Close)

	q := queue.New(4)
	d := New(q, upstream.URL, 1) // no options — default maxAttempts=1, no fallbacks
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go d.Run(ctx)

	req := &queue.Request{
		ID:       "r1",
		Model:    "m",
		Body:     []byte(`{"model":"m","messages":[]}`),
		ResultCh: make(chan queue.Result, 1),
	}
	q.TryPush(req)

	select {
	case <-req.ResultCh:
		if calls.Load() != 1 {
			t.Errorf("upstream calls: got %d, want 1 (no retry, no fallback configured)", calls.Load())
		}
	case <-time.After(time.Second):
		t.Fatal("no result delivered")
	}
}
