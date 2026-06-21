// Package gateway is the HTTP front door of the Request Control Plane.
// It accepts client inference requests, enqueues them, and blocks until the
// dispatcher delivers a result or the wait exceeds the configured timeout.
package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/queue"
	"github.com/atharva/llm-serving-platform/request-plane/slo"
)

// Gateway is an http.Handler that enqueues requests and waits for a result.
type Gateway struct {
	q           *queue.Queue
	waitTimeout time.Duration
	idCounter   atomic.Uint64

	// tracker/estimator are nil unless WithSLO is passed to New — SLO
	// headers are simply omitted when they're not configured.
	tracker   *slo.LatencyTracker
	estimator *slo.Estimator
}

// Option configures optional Gateway behaviour.
type Option func(*Gateway)

// WithSLO enables the X-Estimated-Wait-Ms and X-Actual-Wait-Ms response
// headers. tracker records observed latency per model; estimator predicts
// wait time for new requests using that history plus current queue depth.
func WithSLO(tracker *slo.LatencyTracker, estimator *slo.Estimator) Option {
	return func(g *Gateway) {
		g.tracker = tracker
		g.estimator = estimator
	}
}

// New creates a Gateway. waitTimeout bounds how long a request waits in the
// queue (or for dispatch to complete) before the client gets a 504.
func New(q *queue.Queue, waitTimeout time.Duration, opts ...Option) *Gateway {
	g := &Gateway{q: q, waitTimeout: waitTimeout}
	for _, o := range opts {
		o(g)
	}
	return g
}

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/v1/chat/completions", "/v1/completions":
		g.handleInference(w, r)
	case "/healthz":
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	default:
		http.Error(w, "not found", http.StatusNotFound)
	}
}

func (g *Gateway) handleInference(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || payload.Model == "" {
		http.Error(w, "request body must contain a valid 'model' field", http.StatusBadRequest)
		return
	}

	priority := parsePriority(r)

	// Snapshot queue depth BEFORE pushing — AheadOf must not count this
	// request itself, so the estimate reflects what the caller is actually
	// waiting behind.
	var ahead int
	if g.estimator != nil {
		ahead = g.q.AheadOf(priority)
	}

	req := &queue.Request{
		ID:         fmt.Sprintf("req-%d", g.idCounter.Add(1)),
		Body:       body,
		Model:      payload.Model,
		Priority:   priority,
		EnqueuedAt: time.Now(),
		ResultCh:   make(chan queue.Result, 1),
	}

	if !g.q.TryPush(req) {
		http.Error(w, "request queue full, try again later", http.StatusServiceUnavailable)
		return
	}

	if g.estimator != nil {
		estimate := g.estimator.Estimate(payload.Model, ahead)
		w.Header().Set("X-Estimated-Wait-Ms", strconv.FormatInt(estimate.Milliseconds(), 10))
	}

	select {
	case result := <-req.ResultCh:
		actualWait := time.Since(req.EnqueuedAt)
		if g.tracker != nil {
			g.tracker.Record(payload.Model, actualWait)
			w.Header().Set("X-Actual-Wait-Ms", strconv.FormatInt(actualWait.Milliseconds(), 10))
		}
		if result.Err != nil {
			http.Error(w, result.Err.Error(), http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(result.StatusCode)
		w.Write(result.Body) //nolint:errcheck
	case <-time.After(g.waitTimeout):
		http.Error(w, "request timed out waiting for dispatch", http.StatusGatewayTimeout)
	}
}

// parsePriority reads the X-Priority header. Unset or unrecognized values
// default to queue.PriorityNormal — callers that don't know about priority
// tiers keep working exactly as before CP17.
func parsePriority(r *http.Request) queue.Priority {
	switch r.Header.Get("X-Priority") {
	case "high":
		return queue.PriorityHigh
	case "low":
		return queue.PriorityLow
	default:
		return queue.PriorityNormal
	}
}
