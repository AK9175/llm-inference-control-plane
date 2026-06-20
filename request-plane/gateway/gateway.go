// Package gateway is the HTTP front door of the Request Control Plane.
// It accepts client inference requests, enqueues them, and blocks until the
// dispatcher delivers a result or the wait exceeds the configured timeout.
package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

// Gateway is an http.Handler that enqueues requests and waits for a result.
type Gateway struct {
	q          *queue.Queue
	waitTimeout time.Duration
	idCounter  atomic.Uint64
}

// New creates a Gateway. waitTimeout bounds how long a request waits in the
// queue (or for dispatch to complete) before the client gets a 504.
func New(q *queue.Queue, waitTimeout time.Duration) *Gateway {
	return &Gateway{q: q, waitTimeout: waitTimeout}
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

	req := &queue.Request{
		ID:         fmt.Sprintf("req-%d", g.idCounter.Add(1)),
		Body:       body,
		Model:      payload.Model,
		Priority:   parsePriority(r),
		EnqueuedAt: time.Now(),
		ResultCh:   make(chan queue.Result, 1),
	}

	if !g.q.TryPush(req) {
		http.Error(w, "request queue full, try again later", http.StatusServiceUnavailable)
		return
	}

	select {
	case result := <-req.ResultCh:
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
