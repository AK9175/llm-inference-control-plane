package registry

import (
	"context"
	"testing"
	"time"

	pb "github.com/atharva/llm-serving-platform/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ── validTransition unit tests ─────────────────────────────────────────────────

func TestValidTransition(t *testing.T) {
	type tc struct {
		from, to pb.WorkerState
		want     bool
	}
	cases := []tc{
		// same-state: always valid
		{pb.WorkerState_STARTING, pb.WorkerState_STARTING, true},
		{pb.WorkerState_READY, pb.WorkerState_READY, true},
		{pb.WorkerState_BUSY, pb.WorkerState_BUSY, true},
		{pb.WorkerState_DRAINING, pb.WorkerState_DRAINING, true},
		{pb.WorkerState_DEAD, pb.WorkerState_DEAD, true},

		// valid forward transitions
		{pb.WorkerState_STARTING, pb.WorkerState_READY, true},
		{pb.WorkerState_READY, pb.WorkerState_BUSY, true},
		{pb.WorkerState_BUSY, pb.WorkerState_READY, true},
		{pb.WorkerState_READY, pb.WorkerState_DRAINING, true},
		{pb.WorkerState_BUSY, pb.WorkerState_DRAINING, true},

		// invalid: skip READY
		{pb.WorkerState_STARTING, pb.WorkerState_BUSY, false},
		{pb.WorkerState_STARTING, pb.WorkerState_DRAINING, false},
		{pb.WorkerState_STARTING, pb.WorkerState_DEAD, false},

		// invalid: DRAINING is terminal
		{pb.WorkerState_DRAINING, pb.WorkerState_READY, false},
		{pb.WorkerState_DRAINING, pb.WorkerState_BUSY, false},
		{pb.WorkerState_DRAINING, pb.WorkerState_STARTING, false},

		// invalid: DEAD is terminal
		{pb.WorkerState_DEAD, pb.WorkerState_READY, false},
		{pb.WorkerState_DEAD, pb.WorkerState_STARTING, false},
	}

	for _, c := range cases {
		got := validTransition(c.from, c.to)
		if got != c.want {
			t.Errorf("validTransition(%s → %s) = %v, want %v", c.from, c.to, got, c.want)
		}
	}
}

// ── Registry integration tests ─────────────────────────────────────────────────

func newTestRegistry() *Registry {
	return New(15 * time.Second)
}

func registerWorker(t *testing.T, r *Registry, id string) {
	t.Helper()
	_, err := r.Register(context.Background(), &pb.WorkerInfo{
		WorkerId:     id,
		Address:      "localhost:9999",
		ModelsLoaded: []string{"llama3.2:3b"},
	})
	if err != nil {
		t.Fatalf("Register(%s) failed: %v", id, err)
	}
}

func TestHeartbeat_ValidLifecycle(t *testing.T) {
	r := newTestRegistry()
	registerWorker(t, r, "w1")

	steps := []pb.WorkerState{
		pb.WorkerState_STARTING,
		pb.WorkerState_READY,
		pb.WorkerState_BUSY,
		pb.WorkerState_READY,
		pb.WorkerState_DRAINING,
	}

	for _, s := range steps {
		_, err := r.Heartbeat(context.Background(), &pb.HeartbeatRequest{WorkerId: "w1", State: s})
		if err != nil {
			t.Errorf("Heartbeat(→%s) unexpected error: %v", s, err)
		}
	}
}

func TestHeartbeat_InvalidTransitionRejected(t *testing.T) {
	r := newTestRegistry()
	registerWorker(t, r, "w2")

	// Worker is freshly registered → STARTING.
	// Attempting to jump directly to BUSY must be rejected.
	_, err := r.Heartbeat(context.Background(), &pb.HeartbeatRequest{
		WorkerId: "w2",
		State:    pb.WorkerState_BUSY,
	})
	if err == nil {
		t.Fatal("expected error for STARTING→BUSY, got nil")
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", status.Code(err))
	}
}

func TestHeartbeat_DrainingTerminal(t *testing.T) {
	r := newTestRegistry()
	registerWorker(t, r, "w3")

	// Advance to DRAINING
	for _, s := range []pb.WorkerState{pb.WorkerState_READY, pb.WorkerState_DRAINING} {
		r.Heartbeat(context.Background(), &pb.HeartbeatRequest{WorkerId: "w3", State: s}) //nolint:errcheck
	}

	// Attempting to go back to READY from DRAINING must be rejected.
	_, err := r.Heartbeat(context.Background(), &pb.HeartbeatRequest{
		WorkerId: "w3",
		State:    pb.WorkerState_READY,
	})
	if err == nil {
		t.Fatal("expected error for DRAINING→READY, got nil")
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", status.Code(err))
	}
}

func TestRequestDrain_SignaledInHeartbeatResponse(t *testing.T) {
	r := newTestRegistry()
	registerWorker(t, r, "w4")

	// Advance to READY first.
	r.Heartbeat(context.Background(), &pb.HeartbeatRequest{WorkerId: "w4", State: pb.WorkerState_READY}) //nolint:errcheck

	// Request drain — next heartbeat response must carry drain=true.
	if err := r.RequestDrain("w4"); err != nil {
		t.Fatalf("RequestDrain: %v", err)
	}

	resp, err := r.Heartbeat(context.Background(), &pb.HeartbeatRequest{
		WorkerId: "w4",
		State:    pb.WorkerState_READY,
	})
	if err != nil {
		t.Fatalf("Heartbeat after drain request: %v", err)
	}
	if !resp.Drain {
		t.Error("expected drain=true in HeartbeatResponse, got false")
	}
}
