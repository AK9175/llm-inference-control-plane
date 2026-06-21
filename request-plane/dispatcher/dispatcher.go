// Package dispatcher pulls requests off the queue and forwards them to the
// Infrastructure Control Plane's HTTP router. The dispatcher knows nothing
// about workers, registries, or routing algorithms — that is entirely the
// upstream's job. This keeps Phase 2 (Request Control Plane) decoupled from
// Phase 1 (Infrastructure Control Plane): swap the upstream URL and the
// dispatcher works against any compatible router.
//
// CP20 adds two layers of resilience on top of the upstream's own
// worker-level retries (control-plane/router already retries across
// workers for the SAME model): a same-model retry budget for transient
// failures, and a fallback model chain for when a model has no capacity
// at all.
//
// CP21 adds streaming: instead of buffering the full upstream response
// before replying, a streaming request gets its bytes forwarded chunk by
// chunk as they arrive. Retry/fallback only apply before any bytes are
// read from upstream — once streaming has started, the dispatcher has
// already committed (the gateway may have written response headers), so
// there's no way to retry without corrupting the client's stream.
package dispatcher

import (
	"bytes"
	"context"
	"encoding/json"
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

	// client is used for buffered (non-streaming) requests — a finite
	// timeout is correct here since we wait for the full response anyway.
	client *http.Client

	// streamClient has no timeout: a legitimate streaming generation can
	// run for minutes, and http.Client.Timeout bounds the ENTIRE request
	// including body reads, which would otherwise cut a long stream off
	// mid-generation.
	streamClient *http.Client

	// maxAttempts is how many times to retry the SAME model on a retryable
	// failure (connection error, 502, 503) before moving to a fallback.
	maxAttempts int

	// fallbacks maps a requested model to an ordered list of alternate
	// models to try once the requested model's attempts are exhausted.
	// nil/missing entries mean no fallback — the original CP16 behaviour.
	fallbacks map[string][]string
}

// Option configures optional Dispatcher behaviour.
type Option func(*Dispatcher)

// WithMaxAttempts sets how many times to retry the same model on a
// retryable failure before falling back (or giving up). Values < 1 are
// treated as 1 (no retry, single attempt).
func WithMaxAttempts(n int) Option {
	return func(d *Dispatcher) {
		if n < 1 {
			n = 1
		}
		d.maxAttempts = n
	}
}

// WithFallbacks configures the fallback model chain. fallbacks maps a
// requested model name to an ordered list of alternate models to try if
// the requested model's attempts are all exhausted.
func WithFallbacks(fallbacks map[string][]string) Option {
	return func(d *Dispatcher) { d.fallbacks = fallbacks }
}

// New creates a Dispatcher. concurrency is the number of goroutines pulling
// from the queue concurrently — this bounds how many requests are in flight
// to the upstream router at once.
func New(q *queue.Queue, upstream string, concurrency int, opts ...Option) *Dispatcher {
	d := &Dispatcher{
		q:            q,
		upstream:     upstream,
		concurrency:  concurrency,
		client:       &http.Client{Timeout: 60 * time.Second},
		streamClient: &http.Client{}, // no timeout — see field doc
		maxAttempts:  1,
	}
	for _, o := range opts {
		o(d)
	}
	return d
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
		if req.Stream {
			d.dispatchStream(req)
		} else {
			d.dispatchBuffered(req)
		}
	}
}

// dispatchBuffered forwards one request to the upstream router, retrying
// the requested model up to maxAttempts times on a retryable failure, then
// falling through the configured fallback chain. Always sends exactly one
// Result — the gateway is waiting on this channel and must not block forever.
func (d *Dispatcher) dispatchBuffered(req *queue.Request) {
	chain := append([]string{req.Model}, d.fallbacks[req.Model]...)

	var final queue.Result
	for _, model := range chain {
		body := req.Body
		if model != req.Model {
			rewritten, err := setModel(req.Body, model)
			if err != nil {
				final = queue.Result{Err: fmt.Errorf("rewrite model for fallback %q: %w", model, err)}
				continue
			}
			body = rewritten
		}

		for attempt := 1; attempt <= d.maxAttempts; attempt++ {
			result, retryable := d.attempt(body, model)
			final = result
			if !retryable {
				req.ResultCh <- result
				return
			}
		}
		// Exhausted attempts for this model — fall through to the next
		// model in the chain (or exit the loop if this was the last one).
	}
	req.ResultCh <- final
}

// attempt makes one buffered HTTP call to the upstream router. retryable is
// true for connection failures and 502/503 responses (upstream temporarily
// can't serve this model) — false for anything else, including success.
func (d *Dispatcher) attempt(body []byte, model string) (result queue.Result, retryable bool) {
	httpReq, err := buildRequest(d.upstream, body)
	if err != nil {
		return queue.Result{Err: fmt.Errorf("build upstream request: %w", err), ServedModel: model}, false
	}

	resp, err := d.client.Do(httpReq)
	if err != nil {
		return queue.Result{Err: fmt.Errorf("upstream unreachable: %w", err), ServedModel: model}, true
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return queue.Result{Err: fmt.Errorf("read upstream response: %w", err), ServedModel: model}, false
	}

	result = queue.Result{StatusCode: resp.StatusCode, Body: respBody, ServedModel: model}
	return result, isRetryableStatus(resp.StatusCode)
}

// dispatchStream forwards a streaming request, retrying/falling back only
// while no bytes have been read from upstream yet. Once a non-retryable
// status arrives, it commits: sends a Streaming=true Result immediately
// (so the gateway can write headers) and pumps the response body into
// req.Chunks until upstream closes the connection or errors.
func (d *Dispatcher) dispatchStream(req *queue.Request) {
	defer close(req.Chunks)

	chain := append([]string{req.Model}, d.fallbacks[req.Model]...)

	var final queue.Result
	for _, model := range chain {
		body := req.Body
		if model != req.Model {
			rewritten, err := setModel(req.Body, model)
			if err != nil {
				final = queue.Result{Err: fmt.Errorf("rewrite model for fallback %q: %w", model, err)}
				continue
			}
			body = rewritten
		}

		for attempt := 1; attempt <= d.maxAttempts; attempt++ {
			httpReq, err := buildRequest(d.upstream, body)
			if err != nil {
				final = queue.Result{Err: fmt.Errorf("build upstream request: %w", err), ServedModel: model}
				continue
			}

			resp, err := d.streamClient.Do(httpReq)
			if err != nil {
				final = queue.Result{Err: fmt.Errorf("upstream unreachable: %w", err), ServedModel: model}
				continue
			}

			if isRetryableStatus(resp.StatusCode) {
				respBody, _ := io.ReadAll(resp.Body)
				resp.Body.Close() //nolint:errcheck
				final = queue.Result{StatusCode: resp.StatusCode, Body: respBody, ServedModel: model}
				continue
			}

			// Non-retryable — commit. The gateway is about to write
			// headers/status from this Result; no more retries possible
			// after this point regardless of what happens during the read.
			req.ResultCh <- queue.Result{
				StatusCode:  resp.StatusCode,
				ServedModel: model,
				ContentType: resp.Header.Get("Content-Type"),
				Streaming:   true,
			}
			streamBody(req.Chunks, resp.Body)
			resp.Body.Close() //nolint:errcheck
			return
		}
	}
	// Exhausted every model/attempt without ever committing to a stream.
	req.ResultCh <- final
}

// streamBody copies resp.Body into chunks as data arrives, one Read() call
// at a time. Each chunk is a fresh slice — Read may reuse buf's backing
// array on the next call, so sending buf directly would race with the
// reader overwriting it before the gateway consumes the chunk.
func streamBody(chunks chan<- []byte, body io.Reader) {
	buf := make([]byte, 4096)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			chunks <- chunk
		}
		if err != nil {
			return // io.EOF (normal end) or a real error — nothing more to forward either way
		}
	}
}

// buildRequest constructs the upstream POST request with the standard
// Content-Type header. Shared by both the buffered and streaming paths.
func buildRequest(upstream string, body []byte) (*http.Request, error) {
	target := upstream + "/v1/chat/completions"
	req, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

// isRetryableStatus reports whether a non-error HTTP response code
// represents a transient upstream condition worth retrying (or falling
// back from). 502/503 indicate the upstream router has no usable worker
// right now — that can change on retry, or be solved with a different model.
func isRetryableStatus(code int) bool {
	return code == http.StatusServiceUnavailable || code == http.StatusBadGateway
}

// setModel returns a copy of body with only the top-level "model" field
// replaced — every other field (messages, temperature, etc.) is preserved
// untouched, including fields the dispatcher itself never parses.
func setModel(body []byte, model string) ([]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, fmt.Errorf("unmarshal request body: %w", err)
	}
	encoded, err := json.Marshal(model)
	if err != nil {
		return nil, fmt.Errorf("marshal model name: %w", err)
	}
	fields["model"] = encoded
	return json.Marshal(fields)
}
