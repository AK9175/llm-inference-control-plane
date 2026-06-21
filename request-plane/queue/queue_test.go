package queue

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestQueue_PushPop(t *testing.T) {
	q := New(4)
	req := &Request{ID: "r1"}

	if !q.TryPush(req) {
		t.Fatal("TryPush failed on empty queue")
	}
	if q.Len() != 1 {
		t.Errorf("Len: got %d, want 1", q.Len())
	}

	got, ok := q.Pop(t.Context())
	if !ok {
		t.Fatal("Pop returned ok=false")
	}
	if got.ID != "r1" {
		t.Errorf("Pop: got %s, want r1", got.ID)
	}
	if q.Len() != 0 {
		t.Errorf("Len after pop: got %d, want 0", q.Len())
	}
}

func TestQueue_TryPush_FullReturnsFalse(t *testing.T) {
	q := New(2)
	q.TryPush(&Request{ID: "r1"})
	q.TryPush(&Request{ID: "r2"})

	if q.TryPush(&Request{ID: "r3"}) {
		t.Error("TryPush succeeded on full queue, want false")
	}
	if q.Len() != 2 {
		t.Errorf("Len: got %d, want 2 (capacity)", q.Len())
	}
}

func TestQueue_FIFO_Order(t *testing.T) {
	q := New(4)
	q.TryPush(&Request{ID: "first"})
	q.TryPush(&Request{ID: "second"})
	q.TryPush(&Request{ID: "third"})

	for _, want := range []string{"first", "second", "third"} {
		got, _ := q.Pop(t.Context())
		if got.ID != want {
			t.Errorf("Pop order: got %s, want %s", got.ID, want)
		}
	}
}

func TestQueue_Pop_BlocksUntilPush(t *testing.T) {
	q := New(4)
	done := make(chan *Request, 1)

	go func() {
		req, _ := q.Pop(t.Context())
		done <- req
	}()

	// Pop should be blocked — nothing pushed yet.
	select {
	case <-done:
		t.Fatal("Pop returned before any Push — should have blocked")
	case <-time.After(50 * time.Millisecond):
		// expected: still blocked
	}

	q.TryPush(&Request{ID: "late"})

	select {
	case got := <-done:
		if got.ID != "late" {
			t.Errorf("got %s, want late", got.ID)
		}
	case <-time.After(time.Second):
		t.Fatal("Pop did not unblock after Push")
	}
}

func TestQueue_Pop_RespectsContextCancel(t *testing.T) {
	q := New(4)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled

	_, ok := q.Pop(ctx)
	if ok {
		t.Error("Pop on cancelled context returned ok=true, want false")
	}
}

// ── CP17: priority lanes ───────────────────────────────────────────────────────

func TestQueue_HighPriority_PoppedBeforeNormal(t *testing.T) {
	q := New(4)
	q.TryPush(&Request{ID: "normal-1", Priority: PriorityNormal})
	q.TryPush(&Request{ID: "normal-2", Priority: PriorityNormal})
	q.TryPush(&Request{ID: "high-1", Priority: PriorityHigh})

	got, _ := q.Pop(t.Context())
	if got.ID != "high-1" {
		t.Errorf("first pop: got %s, want high-1 (high priority must jump the line)", got.ID)
	}

	got, _ = q.Pop(t.Context())
	if got.ID != "normal-1" {
		t.Errorf("second pop: got %s, want normal-1 (FIFO within the normal lane)", got.ID)
	}
}

func TestQueue_LowPriority_NeverBlocksHigh(t *testing.T) {
	q := New(4)
	for i := range 3 {
		q.TryPush(&Request{ID: fmt.Sprintf("low-%d", i), Priority: PriorityLow})
	}
	q.TryPush(&Request{ID: "urgent", Priority: PriorityHigh})

	got, _ := q.Pop(t.Context())
	if got.ID != "urgent" {
		t.Errorf("got %s, want urgent — high priority must be served first regardless of arrival order", got.ID)
	}
}

func TestQueue_DefaultPriority_IsNormal(t *testing.T) {
	q := New(4)
	// Zero-value Request (no Priority set) must land in the normal lane —
	// existing callers that don't know about priority keep working unchanged.
	q.TryPush(&Request{ID: "unset"})

	if q.LenByPriority(PriorityNormal) != 1 {
		t.Errorf("normal lane: got %d, want 1 (unset Priority should default to normal)", q.LenByPriority(PriorityNormal))
	}
	if q.LenByPriority(PriorityHigh) != 0 || q.LenByPriority(PriorityLow) != 0 {
		t.Error("high/low lanes should be empty when priority was never set")
	}
}

func TestQueue_PriorityLanes_IndependentCapacity(t *testing.T) {
	q := New(1)
	if !q.TryPush(&Request{ID: "n1", Priority: PriorityNormal}) {
		t.Fatal("first normal push should succeed")
	}
	if q.TryPush(&Request{ID: "n2", Priority: PriorityNormal}) {
		t.Error("second normal push should fail — normal lane at capacity")
	}
	// High lane has its own independent capacity — unaffected by normal being full.
	if !q.TryPush(&Request{ID: "h1", Priority: PriorityHigh}) {
		t.Error("high priority push should succeed even though normal lane is full")
	}
}

func TestQueue_ConcurrentPop_AllItemsDelivered(t *testing.T) {
	q := New(50)
	const n = 30
	for i := range n {
		q.TryPush(&Request{ID: fmt.Sprintf("r%d", i), Priority: PriorityNormal})
	}

	results := make(chan *Request, n)
	for range 5 { // 5 concurrent consumers, mirroring multiple dispatcher workers
		go func() {
			for {
				r, ok := q.Pop(t.Context())
				if !ok {
					return
				}
				results <- r
				if q.Len() == 0 {
					return
				}
			}
		}()
	}

	seen := make(map[string]bool)
	for range n {
		select {
		case r := <-results:
			seen[r.ID] = true
		case <-time.After(2 * time.Second):
			t.Fatalf("only received %d/%d items", len(seen), n)
		}
	}
	if len(seen) != n {
		t.Errorf("got %d distinct items, want %d", len(seen), n)
	}
}

// ── CP18: AheadOf ────────────────────────────────────────────────────────────

func TestQueue_AheadOf_CountsHigherAndOwnLaneOnly(t *testing.T) {
	q := New(10)
	q.TryPush(&Request{ID: "h1", Priority: PriorityHigh})
	q.TryPush(&Request{ID: "h2", Priority: PriorityHigh})
	q.TryPush(&Request{ID: "n1", Priority: PriorityNormal})
	q.TryPush(&Request{ID: "l1", Priority: PriorityLow})
	q.TryPush(&Request{ID: "l2", Priority: PriorityLow})

	if got := q.AheadOf(PriorityHigh); got != 2 {
		t.Errorf("AheadOf(High): got %d, want 2 (only the 2 high items)", got)
	}
	if got := q.AheadOf(PriorityNormal); got != 3 {
		t.Errorf("AheadOf(Normal): got %d, want 3 (2 high + 1 normal)", got)
	}
	if got := q.AheadOf(PriorityLow); got != 5 {
		t.Errorf("AheadOf(Low): got %d, want 5 (2 high + 1 normal + 2 low)", got)
	}
}

func TestQueue_AheadOf_EmptyQueue(t *testing.T) {
	q := New(10)
	if got := q.AheadOf(PriorityNormal); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}
