// Package dispatcher pulls requests off the queue and forwards them to the
// Infrastructure Control Plane's HTTP router. The dispatcher knows nothing
// about workers, registries, or routing algorithms — that is entirely the
// upstream's job. This keeps Phase 2 (Request Control Plane) decoupled from
// Phase 1 (Infrastructure Control Plane): swap the upstream URL and the
// dispatcher works against any compatible router.
package dispatcher

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

// Dispatcher runs a pool of worker goroutines that drain the queue and
// forward each request to upstream via HTTP POST.
type Dispatcher struct {
	q           *queue.Queue
	upstream    string
	concurrency int
	client      *http.Client
}

// New creates a Dispatcher. concurrency is the number of goroutines pulling
// from the queue concurrently — this bounds how many requests are in flight
// to the upstream router at once.
func New(q *queue.Queue, upstream string, concurrency int) *Dispatcher {
	return &Dispatcher{
		q:           q,
		upstream:    upstream,
		concurrency: concurrency,
		client:      &http.Client{Timeout: 60 * time.Second},
	}
}

// Run starts the worker pool and blocks until ctx is cancelled.
func (d *Dispatcher) Run(ctx context.Context) {
	var wg sync.WaitGroup
	for range d.concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			d.workerLoop(ctx)
		}()
	}
	wg.Wait()
}

func (d *Dispatcher) workerLoop(ctx context.Context) {
	for {
		req, ok := d.q.Pop(ctx)
		if !ok {
			return // ctx cancelled — shut down cleanly
		}
		d.dispatch(req)
	}
}

// dispatch forwards one request to the upstream router and writes the
// outcome to the request's ResultCh. Always sends exactly one Result —
// the gateway is waiting on this channel and must not block forever.
func (d *Dispatcher) dispatch(req *queue.Request) {
	target := d.upstream + "/v1/chat/completions"
	httpReq, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(req.Body))
	if err != nil {
		req.ResultCh <- queue.Result{Err: fmt.Errorf("build upstream request: %w", err)}
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(httpReq)
	if err != nil {
		req.ResultCh <- queue.Result{Err: fmt.Errorf("upstream unreachable: %w", err)}
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		req.ResultCh <- queue.Result{Err: fmt.Errorf("read upstream response: %w", err)}
		return
	}

	req.ResultCh <- queue.Result{StatusCode: resp.StatusCode, Body: body}
}
