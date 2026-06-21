// Package backpressure decides whether to admit a request based on its
// predicted wait time, not just whether the queue has physical room.
//
// A queue lane that isn't full can still be deep enough that a new request
// would wait far longer than acceptable — "there's room" and "you'll be
// served in time" are different questions. This package answers the second
// one, using the wait estimate from package slo.
package backpressure

import (
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

// Policy decides whether to admit a request given its predicted wait time.
// Different priorities get different tolerances — high-priority traffic is
// rejected quickly if it can't be served fast, low-priority batch traffic
// is allowed to wait much longer before being shed.
type Policy struct {
	maxWait map[queue.Priority]time.Duration
}

// New creates a Policy from per-priority max wait thresholds.
// A priority with no entry in maxWait is admitted unconditionally —
// callers who want backpressure on every tier must specify all three.
func New(maxWait map[queue.Priority]time.Duration) *Policy {
	return &Policy{maxWait: maxWait}
}

// DefaultThresholds are sensible starting points: tighter tolerance for
// high-priority traffic (it's urgent — fail fast if it can't be served
// quickly), looser tolerance for low-priority batch traffic (it can afford
// to wait longer before being shed).
func DefaultThresholds() map[queue.Priority]time.Duration {
	return map[queue.Priority]time.Duration{
		queue.PriorityHigh:   5 * time.Second,
		queue.PriorityNormal: 15 * time.Second,
		queue.PriorityLow:    60 * time.Second,
	}
}

// Admit returns true if a request with the given priority and estimated
// wait should be accepted into the queue. False means the caller should
// reject it (503) instead of queuing it toward a near-certain timeout.
func (p *Policy) Admit(priority queue.Priority, estimatedWait time.Duration) bool {
	max, ok := p.maxWait[priority]
	if !ok {
		return true // no threshold configured for this priority — admit by default
	}
	return estimatedWait <= max
}
