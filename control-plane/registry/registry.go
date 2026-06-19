package registry

import (
	"context"
	"fmt"
	"sync"
	"time"

	pb "github.com/atharva/llm-serving-platform/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	// DefaultHeartbeatIntervalSecs is sent to workers in RegisterResponse.
	// Workers use this to know how often to call Heartbeat().
	DefaultHeartbeatIntervalSecs = 5

	// DefaultDeadTimeout is how long a worker can go silent before being
	// marked DEAD. Three missed heartbeats is a reasonable threshold —
	// tolerates transient network hiccups without being too slow to detect crashes.
	DefaultDeadTimeout = 15 * time.Second
)

// WorkerEntry is everything the control plane knows about one worker.
type WorkerEntry struct {
	Info          *pb.WorkerInfo
	State         pb.WorkerState
	LastHeartbeat time.Time
	RegisteredAt  time.Time
	Load          *pb.LoadReport

	// deadlineTimer fires deadTimeout after the last heartbeat.
	// It is reset on every Heartbeat() call and stopped on Deregister().
	// Owned by the registry — not exposed outside.
	deadlineTimer *time.Timer
}

// Registry implements the WorkerRegistry gRPC service.
// It is the single source of truth for fleet membership.
type Registry struct {
	pb.UnimplementedWorkerRegistryServer

	mu          sync.RWMutex
	workers     map[string]*WorkerEntry
	deadTimeout time.Duration
}

// New creates an empty Registry. deadTimeout is how long a worker can go
// silent before being considered dead.
func New(deadTimeout time.Duration) *Registry {
	return &Registry{
		workers:     make(map[string]*WorkerEntry),
		deadTimeout: deadTimeout,
	}
}

// ── gRPC handlers ─────────────────────────────────────────────────────────────

// Register adds a worker to the fleet and starts its per-worker deadline timer.
// State starts as STARTING — the router will not route here until the worker
// transitions to READY via a subsequent Heartbeat.
func (r *Registry) Register(_ context.Context, info *pb.WorkerInfo) (*pb.RegisterResponse, error) {
	if info.WorkerId == "" {
		return nil, status.Error(codes.InvalidArgument, "worker_id is required")
	}
	if info.Address == "" {
		return nil, status.Error(codes.InvalidArgument, "address is required")
	}

	now := time.Now()

	// Build the entry first, then start the timer.
	// The timer callback closes over workerID and calls markDead —
	// it must not close over `entry` directly to avoid a data race.
	workerID := info.WorkerId
	entry := &WorkerEntry{
		Info:          info,
		State:         pb.WorkerState_STARTING,
		LastHeartbeat: now,
		RegisteredAt:  now,
		Load:          &pb.LoadReport{WorkerId: workerID},
	}

	// time.AfterFunc fires the callback in its own goroutine after deadTimeout.
	entry.deadlineTimer = time.AfterFunc(r.deadTimeout, func() {
		r.markDead(workerID)
	})

	r.mu.Lock()
	r.workers[workerID] = entry
	r.mu.Unlock()

	fmt.Printf("[registry] + registered   id=%-20s addr=%-22s backend=%-10s provider=%-6s hardware=%-14s models=%v\n",
		info.WorkerId, info.Address, info.Backend, info.Provider, info.Hardware, info.ModelsLoaded)

	return &pb.RegisterResponse{
		Ok:                    true,
		Message:               "registered",
		HeartbeatIntervalSecs: DefaultHeartbeatIntervalSecs,
	}, nil
}

// Heartbeat resets the worker's deadline timer and updates its self-reported state.
// This is the only thing keeping the worker alive in the registry.
func (r *Registry) Heartbeat(_ context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, ok := r.workers[req.WorkerId]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "worker %s not registered; re-register", req.WorkerId)
	}

	// Reset the deadline timer — worker is alive.
	// Stop + Reset is safe here because we hold the write lock,
	// so markDead cannot run concurrently and mutate entry.State.
	entry.deadlineTimer.Reset(r.deadTimeout)
	entry.LastHeartbeat = time.Now()
	entry.State = req.State

	return &pb.HeartbeatResponse{Ok: true, Drain: false}, nil
}

// Deregister removes a worker immediately (graceful shutdown path).
// Stops the deadline timer so markDead never fires for this worker.
func (r *Registry) Deregister(_ context.Context, req *pb.DeregisterRequest) (*pb.Empty, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, ok := r.workers[req.WorkerId]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "worker %s not registered", req.WorkerId)
	}

	entry.deadlineTimer.Stop()
	delete(r.workers, req.WorkerId)

	fmt.Printf("[registry] - deregistered id=%s\n", req.WorkerId)
	return &pb.Empty{}, nil
}

// ReportLoad stores the latest load metrics for a worker.
// Used by the router for least-connections decisions (CP7)
// and by the SLO estimator (CP18).
func (r *Registry) ReportLoad(_ context.Context, report *pb.LoadReport) (*pb.Empty, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, ok := r.workers[report.WorkerId]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "worker %s not registered", report.WorkerId)
	}

	entry.Load = report
	return &pb.Empty{}, nil
}

// ── Internal ──────────────────────────────────────────────────────────────────

// markDead is called by each worker's deadline timer goroutine when the timer
// fires. It marks the worker DEAD only if no heartbeat arrived after the timer
// was set — handles the race where a heartbeat arrives just as the timer fires.
func (r *Registry) markDead(workerID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, ok := r.workers[workerID]
	if !ok || entry.State == pb.WorkerState_DEAD {
		return
	}

	// A heartbeat may have arrived after the timer fired but before we acquired
	// the lock — in that case Heartbeat() already called timer.Reset(), so this
	// worker is still alive. Don't mark it dead.
	if time.Since(entry.LastHeartbeat) < r.deadTimeout {
		return
	}

	entry.State = pb.WorkerState_DEAD
	fmt.Printf("[registry] ✗ timed out   id=%-20s silent_for=%.1fs → DEAD\n",
		workerID, time.Since(entry.LastHeartbeat).Seconds())
}

// ── Read methods (router, scaler, UI) ─────────────────────────────────────────

// GetWorker returns a snapshot of one worker entry.
func (r *Registry) GetWorker(workerID string) (WorkerEntry, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.workers[workerID]
	if !ok {
		return WorkerEntry{}, false
	}
	return *e, true
}

// ListWorkers returns a snapshot of all workers. Used by the UI and scaler.
func (r *Registry) ListWorkers() []WorkerEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]WorkerEntry, 0, len(r.workers))
	for _, e := range r.workers {
		out = append(out, *e)
	}
	return out
}

// HealthyWorkersForModel returns READY or BUSY workers that have modelID loaded.
// Called by the router on every incoming inference request.
func (r *Registry) HealthyWorkersForModel(modelID string) []WorkerEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var out []WorkerEntry
	for _, e := range r.workers {
		if e.State != pb.WorkerState_READY && e.State != pb.WorkerState_BUSY {
			continue
		}
		for _, m := range e.Info.ModelsLoaded {
			if m == modelID {
				out = append(out, *e)
				break
			}
		}
	}
	return out
}

// WorkerCount returns total and healthy worker counts. Used by the scaler.
func (r *Registry) WorkerCount() (total, healthy int) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, e := range r.workers {
		total++
		if e.State == pb.WorkerState_READY || e.State == pb.WorkerState_BUSY {
			healthy++
		}
	}
	return
}
