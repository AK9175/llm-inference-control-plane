package admin

import (
	"encoding/json"
	"net/http"

	"github.com/atharva/llm-serving-platform/request-plane/auth"
	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

// NewHandler returns an http.Handler for the request-plane admin API.
// Runs on its own port (separate from the gateway) so dashboard polling
// never competes with inference traffic.
//
//	GET /admin/queue — queued request count per priority lane
//	GET /admin/stats — total/success/error request counters
//	GET /admin/keys  — registered API keys (KeyID + rate limit + current
//	                   token level; the raw secret key is never exposed)
//
// keyStore/rateLimiter may be nil if auth isn't configured on the gateway —
// /admin/keys then returns an empty list rather than erroring, since "no
// auth configured" is a valid, common deployment.
func NewHandler(q *queue.Queue, stats *Stats, keyStore *auth.KeyStore, rateLimiter *auth.RateLimiter) http.Handler {
	h := &handler{q: q, stats: stats, keyStore: keyStore, rateLimiter: rateLimiter}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /admin/queue", h.handleQueue)
	mux.HandleFunc("GET /admin/stats", h.handleStats)
	mux.HandleFunc("GET /admin/keys", h.handleKeys)
	return withCORS(mux)
}

type handler struct {
	q           *queue.Queue
	stats       *Stats
	keyStore    *auth.KeyStore
	rateLimiter *auth.RateLimiter
}

// ── Response shapes ────────────────────────────────────────────────────────────

type queueResp struct {
	High     int `json:"high"`
	Normal   int `json:"normal"`
	Low      int `json:"low"`
	Total    int `json:"total"`
	Capacity int `json:"capacity"`
}

type statsResp struct {
	TotalRequests   int64 `json:"total_requests"`
	SuccessRequests int64 `json:"success_requests"`
	ErrorRequests   int64 `json:"error_requests"`
}

type keyResp struct {
	KeyID           string  `json:"key_id"`
	RequestsPerMin  int     `json:"requests_per_min"`
	TokensRemaining float64 `json:"tokens_remaining,omitempty"`
	Unlimited       bool    `json:"unlimited"`
}

// ── Handlers ───────────────────────────────────────────────────────────────────

func (h *handler) handleQueue(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, queueResp{
		High:     h.q.LenByPriority(queue.PriorityHigh),
		Normal:   h.q.LenByPriority(queue.PriorityNormal),
		Low:      h.q.LenByPriority(queue.PriorityLow),
		Total:    h.q.Len(),
		Capacity: h.q.Cap(),
	})
}

func (h *handler) handleStats(w http.ResponseWriter, _ *http.Request) {
	total, success, errors := h.stats.Snapshot()
	writeJSON(w, http.StatusOK, statsResp{
		TotalRequests:   total,
		SuccessRequests: success,
		ErrorRequests:   errors,
	})
}

func (h *handler) handleKeys(w http.ResponseWriter, _ *http.Request) {
	if h.keyStore == nil {
		writeJSON(w, http.StatusOK, []keyResp{})
		return
	}

	infos := h.keyStore.ListKeys()
	out := make([]keyResp, 0, len(infos))
	for _, info := range infos {
		resp := keyResp{KeyID: info.KeyID, RequestsPerMin: info.RequestsPerMin}
		if info.RequestsPerMin <= 0 {
			resp.Unlimited = true
		} else if h.rateLimiter != nil {
			tokens, _, _ := h.rateLimiter.Tokens(info.KeyID, info.RequestsPerMin)
			resp.TokensRemaining = tokens
		}
		out = append(out, resp)
	}
	writeJSON(w, http.StatusOK, out)
}

// ── Helpers ────────────────────────────────────────────────────────────────────

// withCORS allows the dashboard (served from a different origin/port by
// control-plane's admin API) to fetch these read-only, non-sensitive
// endpoints directly.
func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}
