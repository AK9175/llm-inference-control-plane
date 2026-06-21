package auth

import (
	"sync"
	"time"
)

// RateLimiter enforces a per-key requests-per-minute budget with a token
// bucket: each key's bucket holds up to requestsPerMin tokens, refilling
// continuously at requestsPerMin/60 tokens per second. A request is
// permitted only if at least one token is available, and consumes it
// immediately — bursts up to the full budget are allowed, sustained traffic
// is capped at the average rate.
type RateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*bucket
}

type bucket struct {
	tokens     float64
	lastRefill time.Time
}

// NewRateLimiter creates an empty RateLimiter — buckets are created lazily
// on first use per key.
func NewRateLimiter() *RateLimiter {
	return &RateLimiter{buckets: make(map[string]*bucket)}
}

// Allow reports whether a request for keyID is permitted right now under
// its requestsPerMin budget, consuming one token if so. requestsPerMin <= 0
// means unlimited — always allowed, no bucket is tracked for that key.
func (r *RateLimiter) Allow(keyID string, requestsPerMin int) bool {
	if requestsPerMin <= 0 {
		return true
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	capacity := float64(requestsPerMin)
	refillPerSec := capacity / 60

	b, ok := r.buckets[keyID]
	now := time.Now()
	if !ok {
		// First request for this key — start at full capacity minus the
		// token this request consumes.
		r.buckets[keyID] = &bucket{tokens: capacity - 1, lastRefill: now}
		return true
	}

	elapsed := now.Sub(b.lastRefill).Seconds()
	b.tokens = min(capacity, b.tokens+elapsed*refillPerSec)
	b.lastRefill = now

	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

// Tokens reports the current token level and capacity for keyID's bucket,
// for inspection (dashboards, debugging) only — it does NOT consume a
// token or mutate bucket state, unlike Allow. tracked is false when
// requestsPerMin <= 0 (unlimited; no bucket is ever created for that key).
func (r *RateLimiter) Tokens(keyID string, requestsPerMin int) (tokens, capacity float64, tracked bool) {
	if requestsPerMin <= 0 {
		return 0, 0, false
	}
	capacity = float64(requestsPerMin)

	r.mu.Lock()
	defer r.mu.Unlock()

	b, ok := r.buckets[keyID]
	if !ok {
		return capacity, capacity, true // never made a request yet — full bucket
	}

	refillPerSec := capacity / 60
	elapsed := time.Since(b.lastRefill).Seconds()
	current := min(capacity, b.tokens+elapsed*refillPerSec)
	return current, capacity, true
}
