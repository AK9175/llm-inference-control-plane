package scaler

import (
	"context"
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
	pb "github.com/atharva/llm-serving-platform/proto"
)

func newReg() *registry.Registry { return registry.New(15 * time.Second) }

func registerReady(t *testing.T, reg *registry.Registry, id string) {
	t.Helper()
	ctx := t.Context()
	reg.Register(ctx, &pb.WorkerInfo{WorkerId: id, Address: "localhost:9"}) //nolint:errcheck
	reg.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: id, State: pb.WorkerState_READY}) //nolint:errcheck
}

// markDead bypasses the heartbeat timer by calling Evict + re-adding with DEAD
// state. Since we can't easily fire the real deadline timer in a unit test,
// we use the registry's exported Evict + re-Register path to set up a DEAD
// entry with a controlled MarkedDeadAt.
//
// Simpler approach: expose a test helper on Registry. Instead, we directly
// call the internal markDead pathway by registering a worker, then calling
// Evict to clear it, then re-adding a fake DEAD entry — but Registry doesn't
// expose that. So we use a very short dead timeout and let the timer fire.
func waitForDead(t *testing.T, reg *registry.Registry, id string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		e, ok := reg.GetWorker(id)
		if ok && e.State == pb.WorkerState_DEAD {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("worker %s never reached DEAD within %s", id, timeout)
}

func TestSweep_EvictsDeadWorkerAfterGrace(t *testing.T) {
	// Use a very short dead timeout so the worker goes DEAD quickly.
	reg := registry.New(100 * time.Millisecond)
	reg.Register(t.Context(), &pb.WorkerInfo{WorkerId: "w1", Address: "localhost:9"}) //nolint:errcheck
	// Do NOT send a heartbeat — worker will go DEAD after 100ms.

	waitForDead(t, reg, "w1", 500*time.Millisecond)

	// Grace period = 0 so the first Sweep evicts immediately.
	sc := New(reg, WithDeadGrace(0))
	sc.Sweep()

	_, ok := reg.GetWorker("w1")
	if ok {
		t.Error("expected w1 to be evicted after sweep, but it still exists")
	}
}

func TestSweep_DoesNotEvictWithinGrace(t *testing.T) {
	reg := registry.New(100 * time.Millisecond)
	reg.Register(t.Context(), &pb.WorkerInfo{WorkerId: "w1", Address: "localhost:9"}) //nolint:errcheck

	waitForDead(t, reg, "w1", 500*time.Millisecond)

	// Grace = 10 minutes — sweep should NOT evict yet.
	sc := New(reg, WithDeadGrace(10*time.Minute))
	sc.Sweep()

	_, ok := reg.GetWorker("w1")
	if !ok {
		t.Error("expected w1 to still be in registry within grace period")
	}
}

func TestSweep_CallsOnWorkerEvicted(t *testing.T) {
	reg := registry.New(100 * time.Millisecond)
	reg.Register(t.Context(), &pb.WorkerInfo{ //nolint:errcheck
		WorkerId:     "w1",
		Address:      "localhost:9",
		ModelsLoaded: []string{"llama3.2:3b"},
	})

	waitForDead(t, reg, "w1", 500*time.Millisecond)

	evicted := make(chan string, 1)
	sc := New(reg,
		WithDeadGrace(0),
		WithOnWorkerEvicted(func(id string, _ []string) { evicted <- id }),
	)
	sc.Sweep()

	select {
	case id := <-evicted:
		if id != "w1" {
			t.Errorf("OnWorkerEvicted called with %s, want w1", id)
		}
	case <-time.After(time.Second):
		t.Error("OnWorkerEvicted was not called")
	}
}

func TestSweep_HealthyWorkersNotEvicted(t *testing.T) {
	reg := newReg()
	registerReady(t, reg, "w1")
	registerReady(t, reg, "w2")

	sc := New(reg, WithDeadGrace(0))
	sc.Sweep()

	if _, ok := reg.GetWorker("w1"); !ok {
		t.Error("healthy w1 should not be evicted")
	}
	if _, ok := reg.GetWorker("w2"); !ok {
		t.Error("healthy w2 should not be evicted")
	}
}

func TestRun_StopsOnContextCancel(t *testing.T) {
	reg := newReg()
	sc := New(reg, WithSweepInterval(10*time.Millisecond))

	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() {
		defer close(done)
		sc.Run(ctx)
	}()

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Error("scaler.Run did not stop after context cancel")
	}
}
