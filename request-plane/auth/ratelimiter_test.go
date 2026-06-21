package auth

import (
	"testing"
	"time"
)

func TestRateLimiter_AllowsWithinBudget(t *testing.T) {
	r := NewRateLimiter()
	if !r.Allow("k1", 2) {
		t.Error("1st request should be allowed")
	}
	if !r.Allow("k1", 2) {
		t.Error("2nd request should be allowed (within budget of 2)")
	}
}

func TestRateLimiter_BlocksOverBudget(t *testing.T) {
	r := NewRateLimiter()
	r.Allow("k1", 2) // consume 1st token
	r.Allow("k1", 2) // consume 2nd token

	if r.Allow("k1", 2) {
		t.Error("3rd immediate request should be blocked — budget of 2 exhausted")
	}
}

func TestRateLimiter_RefillsOverTime(t *testing.T) {
	r := NewRateLimiter()
	const rpm = 120 // refill rate = 2 tokens/sec

	for range rpm {
		r.Allow("k1", rpm)
	}
	if r.Allow("k1", rpm) {
		t.Fatal("expected budget exhausted immediately after consuming the full burst")
	}

	time.Sleep(600 * time.Millisecond) // ~1.2 tokens should have refilled

	if !r.Allow("k1", rpm) {
		t.Error("expected at least one token to have refilled after 600ms")
	}
}

func TestRateLimiter_UnlimitedWhenZero(t *testing.T) {
	r := NewRateLimiter()
	for i := range 1000 {
		if !r.Allow("k1", 0) {
			t.Fatalf("call %d: requestsPerMin=0 should always allow", i)
		}
	}
}

func TestRateLimiter_IndependentPerKey(t *testing.T) {
	r := NewRateLimiter()
	r.Allow("k1", 1) // exhausts k1's single token

	if r.Allow("k1", 1) {
		t.Error("k1 should be exhausted")
	}
	if !r.Allow("k2", 1) {
		t.Error("k2 should be independent of k1's budget")
	}
}
