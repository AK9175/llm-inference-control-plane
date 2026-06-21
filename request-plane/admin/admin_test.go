package admin

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/atharva/llm-serving-platform/request-plane/auth"
	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

func TestHandleQueue_ReportsDepthPerLane(t *testing.T) {
	q := queue.New(10)
	q.TryPush(&queue.Request{ID: "h1", Priority: queue.PriorityHigh, ResultCh: make(chan queue.Result, 1)})
	q.TryPush(&queue.Request{ID: "n1", Priority: queue.PriorityNormal, ResultCh: make(chan queue.Result, 1)})
	q.TryPush(&queue.Request{ID: "n2", Priority: queue.PriorityNormal, ResultCh: make(chan queue.Result, 1)})

	h := NewHandler(q, NewStats(), nil, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/admin/queue", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}
	var resp queueResp
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.High != 1 || resp.Normal != 2 || resp.Low != 0 {
		t.Errorf("got %+v, want High=1 Normal=2 Low=0", resp)
	}
	if resp.Total != 3 {
		t.Errorf("Total: got %d, want 3", resp.Total)
	}
	if resp.Capacity != 30 { // 10 per lane * 3 lanes
		t.Errorf("Capacity: got %d, want 30", resp.Capacity)
	}
}

func TestHandleStats_ReflectsHookUpdates(t *testing.T) {
	stats := NewStats()
	hook := stats.Hook()
	hook("model-a", true)
	hook("model-a", true)
	hook("model-b", false)

	h := NewHandler(queue.New(4), stats, nil, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/admin/stats", nil))

	var resp statsResp
	json.NewDecoder(rec.Body).Decode(&resp) //nolint:errcheck

	if resp.TotalRequests != 3 {
		t.Errorf("TotalRequests: got %d, want 3", resp.TotalRequests)
	}
	if resp.SuccessRequests != 2 {
		t.Errorf("SuccessRequests: got %d, want 2", resp.SuccessRequests)
	}
	if resp.ErrorRequests != 1 {
		t.Errorf("ErrorRequests: got %d, want 1", resp.ErrorRequests)
	}
}

func TestHandleKeys_NoAuthConfigured_ReturnsEmptyList(t *testing.T) {
	h := NewHandler(queue.New(4), NewStats(), nil, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/admin/keys", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d, want 200", rec.Code)
	}
	var resp []keyResp
	json.NewDecoder(rec.Body).Decode(&resp) //nolint:errcheck
	if len(resp) != 0 {
		t.Errorf("got %d keys, want 0 (auth not configured)", len(resp))
	}
}

func TestHandleKeys_NeverExposesRawKey(t *testing.T) {
	keyStore := auth.NewKeyStore()
	keyStore.AddKey("super-secret-raw-key", auth.KeyInfo{KeyID: "customer-a", RequestsPerMin: 60})

	h := NewHandler(queue.New(4), NewStats(), keyStore, auth.NewRateLimiter())
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/admin/keys", nil))

	body := rec.Body.String()
	if strings.Contains(body, "super-secret-raw-key") {
		t.Fatal("raw API key leaked in /admin/keys response")
	}

	var resp []keyResp
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp) != 1 || resp[0].KeyID != "customer-a" {
		t.Errorf("got %+v, want one entry with KeyID=customer-a", resp)
	}
}

func TestHandleKeys_ReportsRateLimitInfo(t *testing.T) {
	keyStore := auth.NewKeyStore()
	keyStore.AddKey("secret-1", auth.KeyInfo{KeyID: "customer-a", RequestsPerMin: 60})
	keyStore.AddKey("secret-2", auth.KeyInfo{KeyID: "customer-b", RequestsPerMin: 0}) // unlimited

	limiter := auth.NewRateLimiter()
	limiter.Allow("customer-a", 60) // consume one token

	h := NewHandler(queue.New(4), NewStats(), keyStore, limiter)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/admin/keys", nil))

	var resp []keyResp
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp) != 2 {
		t.Fatalf("got %d keys, want 2", len(resp))
	}

	byID := make(map[string]keyResp)
	for _, k := range resp {
		byID[k.KeyID] = k
	}

	a := byID["customer-a"]
	if a.Unlimited {
		t.Error("customer-a should not be marked unlimited")
	}
	// ~59 expected (60 capacity - 1 consumed), allowing a small tolerance
	// for the refill that happens during the microseconds elapsed since
	// Allow() was called — Tokens() is a live read, not a frozen snapshot.
	if a.TokensRemaining < 58.9 || a.TokensRemaining > 59.1 {
		t.Errorf("customer-a TokensRemaining: got %v, want ~59 (60 capacity - 1 consumed)", a.TokensRemaining)
	}

	b := byID["customer-b"]
	if !b.Unlimited {
		t.Error("customer-b should be marked unlimited")
	}
}

func TestCORS_HeadersPresent(t *testing.T) {
	h := NewHandler(queue.New(4), NewStats(), nil, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/admin/stats", nil))

	if rec.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("CORS header missing — dashboard on a different origin would be blocked")
	}
}

func TestCORS_OptionsPreflight_Returns204(t *testing.T) {
	h := NewHandler(queue.New(4), NewStats(), nil, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodOptions, "/admin/stats", nil))

	if rec.Code != http.StatusNoContent {
		t.Errorf("status: got %d, want 204", rec.Code)
	}
}
