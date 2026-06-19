package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
)

// Router is an HTTP server that sits in front of the worker fleet.
// It parses the model name from each request, finds healthy workers
// that have that model loaded, picks one via round-robin, and proxies
// the request to that worker's Ollama address.
//
// It exposes an OpenAI-compatible surface:
//   POST /v1/chat/completions
//   POST /v1/completions
//
// so any OpenAI SDK or curl command works out of the box.
type Router struct {
	reg     *registry.Registry
	counter atomic.Uint64
}

// New creates a Router backed by the given registry.
func New(reg *registry.Registry) *Router {
	return &Router{reg: reg}
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

	// Read body once — we need to inspect it for the model name
	// and then forward it unchanged to the worker.
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

	// Find workers that are READY or BUSY and have this model loaded.
	workers := r.reg.HealthyWorkersForModel(payload.Model)
	if len(workers) == 0 {
		http.Error(w,
			fmt.Sprintf("no healthy workers available for model %q", payload.Model),
			http.StatusServiceUnavailable)
		return
	}

	// Round-robin: atomic counter mod fleet size.
	// Safe under concurrent requests — each request gets its own index.
	idx := r.counter.Add(1) % uint64(len(workers))
	chosen := workers[idx]

	target := "http://" + chosen.Info.Address + req.URL.Path
	fmt.Printf("[router] → %s  model=%s  worker=%s\n", req.URL.Path, payload.Model, chosen.Info.WorkerId)

	if err := r.proxy(w, req, target, body); err != nil {
		fmt.Printf("[router] proxy error worker=%s: %v\n", chosen.Info.WorkerId, err)
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
	}
}

// proxy forwards the request to target and streams the response back.
// Streaming responses (SSE / chunked) work because we use io.Copy —
// bytes flow through as the upstream writes them.
func (r *Router) proxy(w http.ResponseWriter, orig *http.Request, target string, body []byte) error {
	proxyReq, err := http.NewRequestWithContext(orig.Context(), orig.Method, target, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	// Forward relevant headers from the original request.
	for _, h := range []string{"Content-Type", "Accept", "Authorization"} {
		if v := orig.Header.Get(h); v != "" {
			proxyReq.Header.Set(h, v)
		}
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(proxyReq)
	if err != nil {
		return fmt.Errorf("upstream call: %w", err)
	}
	defer resp.Body.Close()

	// Copy status + headers before writing body.
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	_, err = io.Copy(w, resp.Body)
	return err
}
