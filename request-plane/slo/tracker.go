// Package slo estimates how long a newly queued request will wait before a
// dispatcher worker picks it up, based on observed historical latency per
// model and current queue depth.
package slo

import (
	"sync"
	"time"
)

// alpha is the EWMA smoothing factor. Higher values make the average react
// faster to recent samples. 0.3 reacts to load shifts within a handful of
// requests without being noisy on every single sample — a model that just
// got overloaded should have its estimate catch up in seconds, not minutes.
const alpha = 0.3

// LatencyTracker maintains a per-model exponentially weighted moving
// average of end-to-end request latency (time from enqueue to result
// delivered). EWMA is used instead of a plain average so the estimate
// tracks CURRENT load rather than being dragged down by a long history.
type LatencyTracker struct {
	mu       sync.RWMutex
	avg      map[string]time.Duration
	fallback time.Duration
}

// NewLatencyTracker creates a tracker. fallback is returned for any model
// that has no recorded samples yet (cold start).
func NewLatencyTracker(fallback time.Duration) *LatencyTracker {
	return &LatencyTracker{avg: make(map[string]time.Duration), fallback: fallback}
}

// Record incorporates one observed latency sample for model.
func (t *LatencyTracker) Record(model string, d time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	cur, ok := t.avg[model]
	if !ok {
		t.avg[model] = d
		return
	}
	t.avg[model] = time.Duration(float64(cur)*(1-alpha) + float64(d)*alpha)
}

// Average returns the current EWMA latency for model, or the fallback
// duration if no samples have been recorded for it yet.
func (t *LatencyTracker) Average(model string) time.Duration {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if d, ok := t.avg[model]; ok {
		return d
	}
	return t.fallback
}
