package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
)

// Router is an HTTP server that sits in front of the worker fleet.
//
// Routing algorithm (CP8):
//  1. Primary signal: in-flight request count per worker — always accurate,
//     never stale. Tracked by the router itself, not reported by the worker.
//  2. Tie-break: round-robin via atomic counter.
//  3. Retry: if a worker is unreachable (connection error before response
//     starts), the router picks the next-best worker and retries — up to
//     min(len(workers), 3) attempts total. The client never sees a 502
//     unless all retry attempts fail.
//
// Exposes an OpenAI-compatible surface:
//
//	POST /v1/chat/completions
//	POST /v1/completions
type Router struct {
	reg      *registry.Registry
	counter  atomic.Uint64
	ifMu     sync.RWMutex
	inFlight map[string]*atomic.Int64 // workerID → in-flight count
}

// New creates a Router backed by the given registry.
func New(reg *registry.Registry) *Router {
	return &Router{
		reg:      reg,
		inFlight: make(map[string]*atomic.Int64),
	}
}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	switch req.URL.Path {
	case "/v1/chat/completions", "/v1/completions":
		r.handleInference(w, req)
	case "/healthz":
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	default:
		http.Error(w, "not found", http.StatusNotFound)
	}
}

func (r *Router) handleInference(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read body once — inspect for model name, then forward unchanged.
	body, err := io.ReadAll(req.Body)
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

	workers := r.reg.HealthyWorkersForModel(payload.Model)
	if len(workers) == 0 {
		http.Error(w,
			fmt.Sprintf("no healthy workers available for model %q", payload.Model),
			http.StatusServiceUnavailable)
		return
	}

	// Cap retries at 3 — no point hammering a dead fleet.
	maxAttempts := min(len(workers), 3)
	tried := make(map[string]bool, maxAttempts)

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		// Exclude workers that already failed this request.
		candidates := make([]registry.WorkerEntry, 0, len(workers))
		for _, w := range workers {
			if !tried[w.Info.WorkerId] {
				candidates = append(candidates, w)
			}
		}
		if len(candidates) == 0 {
			break
		}

		chosen := pickWorker(candidates, &r.counter, r.getInFlight)
		tried[chosen.Info.WorkerId] = true

		target := "http://" + chosen.Info.Address + req.URL.Path
		if attempt == 1 {
			fmt.Printf("[router] → %s  model=%s  worker=%s\n",
				req.URL.Path, payload.Model, chosen.Info.WorkerId)
		} else {
			fmt.Printf("[router] ↺ retry %d  model=%s  worker=%s\n",
				attempt, payload.Model, chosen.Info.WorkerId)
		}

		r.inc(chosen.Info.WorkerId)
		retryable, err := r.proxy(w, req, target, body)
		r.dec(chosen.Info.WorkerId)

		if err == nil {
			return // success
		}

		fmt.Printf("[router] ✗ worker=%s err=%v  retryable=%v\n",
			chosen.Info.WorkerId, err, retryable)

		if !retryable {
			// Response already started streaming — can't recover, client sees the error.
			return
		}
		// Connection-level failure before any response was written — safe to retry.
	}

	http.Error(w, "all workers failed or unavailable", http.StatusBadGateway)
}

// ── Worker selection ───────────────────────────────────────────────────────────

// pickWorker selects the best worker using a three-level priority:
//  1. Fewest in-flight requests — latency first, never stale.
//  2. Lowest cost_per_hour — among equally-loaded workers, prefer cheaper.
//     Local workers (cost=0) are always preferred over cloud workers.
//  3. Round-robin — final tie-break for equal load and equal cost.
func pickWorker(
	workers []registry.WorkerEntry,
	counter *atomic.Uint64,
	getInFlight func(string) int64,
) registry.WorkerEntry {
	if len(workers) == 1 {
		return workers[0]
	}

	// Level 1: find minimum in-flight count.
	minFlight := getInFlight(workers[0].Info.WorkerId)
	for _, w := range workers[1:] {
		if f := getInFlight(w.Info.WorkerId); f < minFlight {
			minFlight = f
		}
	}

	flight := make([]registry.WorkerEntry, 0, len(workers))
	for _, w := range workers {
		if getInFlight(w.Info.WorkerId) == minFlight {
			flight = append(flight, w)
		}
	}
	if len(flight) == 1 {
		return flight[0]
	}

	// Level 2: among equally-loaded workers, prefer lowest cost_per_hour.
	minCost := flight[0].Info.CostPerHour
	for _, w := range flight[1:] {
		if w.Info.CostPerHour < minCost {
			minCost = w.Info.CostPerHour
		}
	}

	cheapest := make([]registry.WorkerEntry, 0, len(flight))
	for _, w := range flight {
		if w.Info.CostPerHour == minCost {
			cheapest = append(cheapest, w)
		}
	}

	// Level 3: round-robin among equally-loaded, equally-priced workers.
	idx := counter.Add(1) % uint64(len(cheapest))
	return cheapest[idx]
}

// ── In-flight counter helpers ─────────────────────────────────────────────────

func (r *Router) counter64(workerID string) *atomic.Int64 {
	r.ifMu.RLock()
	c, ok := r.inFlight[workerID]
	r.ifMu.RUnlock()
	if ok {
		return c
	}
	r.ifMu.Lock()
	defer r.ifMu.Unlock()
	if c, ok = r.inFlight[workerID]; ok {
		return c // another goroutine beat us
	}
	c = new(atomic.Int64)
	r.inFlight[workerID] = c
	return c
}

func (r *Router) inc(workerID string) { r.counter64(workerID).Add(1) }
func (r *Router) dec(workerID string) { r.counter64(workerID).Add(-1) }

func (r *Router) getInFlight(workerID string) int64 {
	r.ifMu.RLock()
	c, ok := r.inFlight[workerID]
	r.ifMu.RUnlock()
	if !ok {
		return 0
	}
	return c.Load()
}

// InFlightSnapshot returns a point-in-time copy of in-flight counts for all
// workers. Used by the admin API and the Fleet Scaler to observe router state.
func (r *Router) InFlightSnapshot() map[string]int64 {
	r.ifMu.RLock()
	defer r.ifMu.RUnlock()
	out := make(map[string]int64, len(r.inFlight))
	for id, c := range r.inFlight {
		out[id] = c.Load()
	}
	return out
}

// ── Proxy ─────────────────────────────────────────────────────────────────────

// proxy forwards the request to target and streams the response back.
// Returns (retryable=true, err) when the upstream was unreachable before
// any response bytes were written — the caller can safely try another worker.
// Returns (retryable=false, err) when streaming started and then broke —
// the response headers are already committed, so retrying is not possible.
func (r *Router) proxy(w http.ResponseWriter, orig *http.Request, target string, body []byte) (retryable bool, _ error) {
	proxyReq, err := http.NewRequestWithContext(orig.Context(), orig.Method, target, bytes.NewReader(body))
	if err != nil {
		return true, fmt.Errorf("build request: %w", err)
	}

	for _, h := range []string{"Content-Type", "Accept", "Authorization"} {
		if v := orig.Header.Get(h); v != "" {
			proxyReq.Header.Set(h, v)
		}
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	// Connection errors here are retryable — nothing written to w yet.
	resp, err := http.DefaultClient.Do(proxyReq)
	if err != nil {
		return true, fmt.Errorf("upstream unreachable: %w", err)
	}
	defer resp.Body.Close()

	// Once we write the status code the response is committed — no retry possible.
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)

	if _, err := io.Copy(w, resp.Body); err != nil {
		return false, fmt.Errorf("stream interrupted: %w", err)
	}
	return false, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
