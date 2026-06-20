package dispatcher

import (
	"context"
	"fmt"
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
