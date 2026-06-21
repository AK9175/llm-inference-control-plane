package slo

import (
	"testing"
	"time"
)

func TestEstimator_ZeroAhead_NoWait(t *testing.T) {
	tr := NewLatencyTracker(time.Second)
	e := NewEstimator(tr, 4)

	got := e.Estimate("m", 0)
	if got != 0 {
		t.Errorf("got %s, want 0 (nothing ahead means no wait)", got)
	}
}

func TestEstimator_ScalesWithAheadCount(t *testing.T) {
	tr := NewLatencyTracker(time.Second)
	e := NewEstimator(tr, 1) // concurrency=1 for simple arithmetic

	got := e.Estimate("m", 4)
	want := 4 * time.Second
	if got != want {
		t.Errorf("got %s, want %s (4 requests ahead, 1 worker, 1s avg latency)", got, want)
	}
}

func TestEstimator_ScalesInverseWithConcurrency(t *testing.T) {
	tr := NewLatencyTracker(time.Second)

	e1 := NewEstimator(tr, 1)
	e4 := NewEstimator(tr, 4)

	got1 := e1.Estimate("m", 8)
	got4 := e4.Estimate("m", 8)

	if got4 >= got1 {
		t.Errorf("higher concurrency should reduce wait: concurrency=1 -> %s, concurrency=4 -> %s", got1, got4)
	}
	// 8 ahead / 4 workers = 2 rounds * 1s = 2s
	if got4 != 2*time.Second {
		t.Errorf("got %s, want 2s", got4)
	}
}

func TestEstimator_ConcurrencyLessThanOne_TreatedAsOne(t *testing.T) {
	tr := NewLatencyTracker(time.Second)
	e := NewEstimator(tr, 0) // invalid input, should not divide by zero

	got := e.Estimate("m", 2)
	want := 2 * time.Second
	if got != want {
		t.Errorf("got %s, want %s (concurrency=0 should be treated as 1)", got, want)
	}
}

func TestEstimator_UsesPerModelLatency(t *testing.T) {
	tr := NewLatencyTracker(time.Second)
	tr.Record("slow-model", 4*time.Second)
	e := NewEstimator(tr, 1)

	got := e.Estimate("slow-model", 1)
	want := 4 * time.Second
	if got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}
