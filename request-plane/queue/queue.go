// Package queue provides the request queue that decouples request arrival
// from dispatch. This is the foundation CP17 (priority), CP18 (SLO
// estimation), and CP19 (backpressure) build on.
package queue

import (
	"context"
	"time"
)

// Priority classifies a request for queue ordering. Higher-priority lanes
// are always drained first — a flood of low-priority batch traffic can
// never starve a single high-priority request.
//
// PriorityNormal is the zero value, so any Request constructed without
// explicitly setting Priority behaves exactly as it did before CP17.
type Priority int

const (
	PriorityNormal Priority = iota // zero value — default for unset requests
	PriorityHigh
	PriorityLow
)

func (p Priority) String() string {
	switch p {
	case PriorityHigh:
		return "high"
	case PriorityLow:
		return "low"
	default:
		return "normal"
	}
}

// Request is one client inference request waiting to be dispatched.
type Request struct {
	ID         string
	Body       []byte
	Model      string
	Priority   Priority
	EnqueuedAt time.Time

	// ResultCh receives exactly one Result once the dispatcher finishes.
	// Buffered with capacity 1 so the dispatcher never blocks writing it.
	ResultCh chan Result
}

// Result is what the dispatcher sends back after forwarding to the upstream
// Infrastructure Control Plane router.
type Result struct {
	StatusCode int
	Body       []byte
	Err        error
}

// popOrder defines which lane Pop checks first: highest priority to lowest.
var popOrder = []Priority{PriorityHigh, PriorityNormal, PriorityLow}

// Queue is a priority-aware FIFO request queue. Each priority level has its
// own channel ("lane"); within a lane, order is strict FIFO. Across lanes,
// higher priority is always served first — Pop never returns a normal or
// low request while a high-priority one is waiting.
//
// capacityPerLane bounds how many requests each lane can hold before
// TryPush starts failing for that lane. That failure is the backpressure
// signal CP19 turns into a 503 response.
type Queue struct {
	lanes map[Priority]chan *Request
}

// New creates a Queue with capacityPerLane slots in each priority lane.
func New(capacityPerLane int) *Queue {
	return &Queue{
		lanes: map[Priority]chan *Request{
			PriorityHigh:   make(chan *Request, capacityPerLane),
			PriorityNormal: make(chan *Request, capacityPerLane),
			PriorityLow:    make(chan *Request, capacityPerLane),
		},
	}
}

// TryPush enqueues a request into its priority's lane without blocking.
// Returns false if that lane is at capacity — caller should reject the
// request (503) rather than wait indefinitely.
func (q *Queue) TryPush(r *Request) bool {
	select {
	case q.lanes[r.Priority] <- r:
		return true
	default:
		return false
	}
}

// Pop returns the next request, always preferring higher-priority lanes.
// Blocks until something is available or ctx is cancelled.
//
// The non-blocking fast path enforces strict priority order. The blocking
// select fallback (entered only when every lane was empty) lets Go's
// runtime fairly wake any one of several goroutines calling Pop
// concurrently — a manual "signal channel" approach would let one goroutine
// hog all the draining under bursty load instead of sharing it.
func (q *Queue) Pop(ctx context.Context) (req *Request, ok bool) {
	if r, ok := q.tryPopHighestPriority(); ok {
		return r, true
	}
	select {
	case r := <-q.lanes[PriorityHigh]:
		return r, true
	case r := <-q.lanes[PriorityNormal]:
		return r, true
	case r := <-q.lanes[PriorityLow]:
		return r, true
	case <-ctx.Done():
		return nil, false
	}
}

func (q *Queue) tryPopHighestPriority() (*Request, bool) {
	for _, p := range popOrder {
		select {
		case r := <-q.lanes[p]:
			return r, true
		default:
		}
	}
	return nil, false
}

// Len returns the total number of queued (not yet popped) requests across
// all priority lanes.
func (q *Queue) Len() int {
	n := 0
	for _, lane := range q.lanes {
		n += len(lane)
	}
	return n
}

// LenByPriority returns the number of queued requests in one priority lane.
func (q *Queue) LenByPriority(p Priority) int { return len(q.lanes[p]) }

// Cap returns the total capacity across all priority lanes.
func (q *Queue) Cap() int {
	n := 0
	for _, lane := range q.lanes {
		n += cap(lane)
	}
	return n
}
