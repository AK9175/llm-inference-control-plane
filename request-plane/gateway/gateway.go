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
	"strings"
	"sync/atomic"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/auth"
	"github.com/atharva/llm-serving-platform/request-plane/backpressure"
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

	// policy is nil unless WithBackpressure is passed to New. It requires
	// an estimator to function — without one there's no prediction to
	// check against, so the gateway fails open (admits everything) rather
	// than silently rejecting based on nothing.
	policy *backpressure.Policy

	// keyStore/rateLimiter are nil unless WithAuth is passed to New — when
	// nil, every request is admitted without an API key (matches pre-CP22
	// behaviour exactly).
	keyStore    *auth.KeyStore
	rateLimiter *auth.RateLimiter

	// onRequest is nil unless WithRequestHook is passed to New. Called once
	// per request with the final outcome — for the admin/dashboard request
	// counters (CP24). model is empty if the request failed before the
	// body was parsed (auth rejection, bad method).
	onRequest func(model string, success bool)
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

// WithBackpressure rejects requests up front when their predicted wait
// exceeds the policy's threshold for their priority — instead of queuing
// them toward a near-certain timeout. Requires WithSLO to also be set;
// without an estimator there's nothing to check the policy against, so
// the gateway admits everything (fails open).
func WithBackpressure(policy *backpressure.Policy) Option {
	return func(g *Gateway) { g.policy = policy }
}

// WithAuth requires a valid API key (Authorization: Bearer <key>) on every
// request and enforces that key's rate limit. Missing or unrecognized keys
// get 401; over-budget keys get 429. Checked before any other processing —
// the cheapest possible rejection point for unauthenticated traffic.
func WithAuth(keyStore *auth.KeyStore, rateLimiter *auth.RateLimiter) Option {
	return func(g *Gateway) {
		g.keyStore = keyStore
		g.rateLimiter = rateLimiter
	}
}

// WithRequestHook registers fn to be called once per request with the
// final outcome (model, success). Used by the admin package (CP24) to
// maintain request counters for the dashboard.
func WithRequestHook(fn func(model string, success bool)) Option {
	return func(g *Gateway) { g.onRequest = fn }
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
	var outcomeModel string
	success := false
	if g.onRequest != nil {
		defer func() { g.onRequest(outcomeModel, success) }()
	}

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var keyID string
	if g.keyStore != nil {
		apiKey, ok := extractAPIKey(r)
		if !ok {
			http.Error(w, "missing API key", http.StatusUnauthorized)
			return
		}
		info, found := g.keyStore.Lookup(apiKey)
		if !found {
			http.Error(w, "invalid API key", http.StatusUnauthorized)
			return
		}
		keyID = info.KeyID

		if g.rateLimiter != nil && !g.rateLimiter.Allow(info.KeyID, info.RequestsPerMin) {
			w.Header().Set("Retry-After", "1")
			http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
			return
		}
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	var payload struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || payload.Model == "" {
		http.Error(w, "request body must contain a valid 'model' field", http.StatusBadRequest)
		return
	}
	outcomeModel = payload.Model

	priority := parsePriority(r)

	// Snapshot queue depth BEFORE pushing — AheadOf must not count this
	// request itself, so the estimate reflects what the caller is actually
	// waiting behind. Computed once and reused for both the backpressure
	// admission decision and the X-Estimated-Wait-Ms header.
	var estimate time.Duration
	haveEstimate := g.estimator != nil
	if haveEstimate {
		ahead := g.q.AheadOf(priority)
		estimate = g.estimator.Estimate(payload.Model, ahead)
	}

	if g.policy != nil && haveEstimate && !g.policy.Admit(priority, estimate) {
		w.Header().Set("X-Estimated-Wait-Ms", strconv.FormatInt(estimate.Milliseconds(), 10))
		w.Header().Set("Retry-After", strconv.Itoa(int(estimate.Seconds())+1))
		http.Error(w, fmt.Sprintf(
			"request rejected: estimated wait %s exceeds SLO for %s priority", estimate, priority),
			http.StatusServiceUnavailable)
		return
	}

	req := &queue.Request{
		ID:         fmt.Sprintf("req-%d", g.idCounter.Add(1)),
		Body:       body,
		Model:      payload.Model,
		Priority:   priority,
		Stream:     payload.Stream,
		KeyID:      keyID,
		EnqueuedAt: time.Now(),
		ResultCh:   make(chan queue.Result, 1),
	}
	if payload.Stream {
		req.Chunks = make(chan []byte, 16)
	}

	if !g.q.TryPush(req) {
		http.Error(w, "request queue full, try again later", http.StatusServiceUnavailable)
		return
	}

	if haveEstimate {
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
		success = result.StatusCode >= 200 && result.StatusCode < 300
		if result.Streaming {
			g.streamResponse(w, req, result)
			return
		}
		if result.ServedModel != "" {
			w.Header().Set("X-Served-Model", result.ServedModel)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(result.StatusCode)
		w.Write(result.Body) //nolint:errcheck
	case <-time.After(g.waitTimeout):
		http.Error(w, "request timed out waiting for dispatch", http.StatusGatewayTimeout)
	}
}

// streamResponse writes headers/status once (the dispatcher has already
// committed — no more retries possible) then forwards each chunk from
// req.Chunks to the client, flushing immediately so bytes aren't held back
// by Go's default response buffering. No timeout applies here: a
// legitimate generation can take as long as it takes, only the wait for
// the FIRST signal (above) was bounded by waitTimeout.
func (g *Gateway) streamResponse(w http.ResponseWriter, req *queue.Request, result queue.Result) {
	if result.ServedModel != "" {
		w.Header().Set("X-Served-Model", result.ServedModel)
	}
	contentType := result.ContentType
	if contentType == "" {
		contentType = "text/event-stream"
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(result.StatusCode)

	flusher, canFlush := w.(http.Flusher)
	for chunk := range req.Chunks {
		w.Write(chunk) //nolint:errcheck
		if canFlush {
			flusher.Flush()
		}
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

// extractAPIKey reads the API key from "Authorization: Bearer <key>".
// Returns ok=false if the header is missing, malformed, or the key is empty.
func extractAPIKey(r *http.Request) (string, bool) {
	const prefix = "Bearer "
	header := r.Header.Get("Authorization")
	if !strings.HasPrefix(header, prefix) {
		return "", false
	}
	key := strings.TrimPrefix(header, prefix)
	if key == "" {
		return "", false
	}
	return key, true
}
