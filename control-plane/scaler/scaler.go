package scaler

import (
	"context"
	"fmt"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
	pb "github.com/atharva/llm-serving-platform/proto"
)

const (
	DefaultSweepInterval = 30 * time.Second
	DefaultDeadGrace     = 30 * time.Second
	DefaultMinHealthy    = 1
)

// Scaler watches the worker fleet and enforces fleet health.
//
// On every sweep it does two things:
//  1. Evicts workers that have been DEAD for longer than deadGrace.
//     Dead workers that are evicted too quickly waste the grace window
//     operators use to inspect the admin API before entries disappear.
//  2. Warns when the number of healthy (READY/BUSY) workers drops below
//     minHealthy. CP12 will hook into OnWorkerEvicted to auto-provision
//     a replacement — for now it just logs.
type Scaler struct {
	reg        *registry.Registry
	sweep      time.Duration
	deadGrace  time.Duration
	minHealthy int

	// OnWorkerEvicted is called after a dead worker is removed from the registry.
	// CP12 (cloud provisioner) will set this to spin up a replacement worker.
	// nil means log-only.
	OnWorkerEvicted func(workerID string, models []string)
}

// New creates a Scaler with default settings. Use the functional options
// to override sweep interval, grace period, or minimum healthy count.
func New(reg *registry.Registry, opts ...Option) *Scaler {
	s := &Scaler{
		reg:        reg,
		sweep:      DefaultSweepInterval,
		deadGrace:  DefaultDeadGrace,
		minHealthy: DefaultMinHealthy,
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// Option is a functional option for Scaler.
type Option func(*Scaler)

func WithSweepInterval(d time.Duration) Option { return func(s *Scaler) { s.sweep = d } }
func WithDeadGrace(d time.Duration) Option     { return func(s *Scaler) { s.deadGrace = d } }
func WithMinHealthy(n int) Option              { return func(s *Scaler) { s.minHealthy = n } }
func WithOnWorkerEvicted(fn func(string, []string)) Option {
	return func(s *Scaler) { s.OnWorkerEvicted = fn }
}

// Run starts the scaler loop. It runs until ctx is cancelled.
// Call in a goroutine: go scaler.Run(ctx)
func (s *Scaler) Run(ctx context.Context) {
	ticker := time.NewTicker(s.sweep)
	defer ticker.Stop()
	s.Sweep() // immediate first pass on startup
	for {
		select {
		case <-ticker.C:
			s.Sweep()
		case <-ctx.Done():
			return
		}
	}
}

// Sweep is one pass of the scaler loop. It is exported so tests can call
// it directly without waiting for the ticker.
func (s *Scaler) Sweep() {
	workers := s.reg.ListWorkers()

	var healthy int
	for _, w := range workers {
		switch w.State {
		case pb.WorkerState_DEAD:
			s.handleDead(w)
		case pb.WorkerState_READY, pb.WorkerState_BUSY:
			healthy++
		}
	}

	// Re-count after potential evictions so the warning reflects current reality.
	total, currentHealthy := s.reg.WorkerCount()
	if total > 0 && currentHealthy < s.minHealthy {
		fmt.Printf("[scaler] ⚠  fleet below minimum: %d healthy, want %d — scale-up needed\n",
			currentHealthy, s.minHealthy)
	}
}

func (s *Scaler) handleDead(w registry.WorkerEntry) {
	if w.MarkedDeadAt.IsZero() {
		// markDead hasn't stamped MarkedDeadAt yet — skip this sweep.
		return
	}
	deadFor := time.Since(w.MarkedDeadAt)
	if deadFor < s.deadGrace {
		// Still within grace period — leave it visible in the admin API.
		return
	}

	if err := s.reg.Evict(w.Info.WorkerId); err != nil {
		return // already evicted by a concurrent sweep
	}

	fmt.Printf("[scaler] 🗑  evicted      id=%-20s dead_for=%.0fs  models=%v\n",
		w.Info.WorkerId, deadFor.Seconds(), w.Info.ModelsLoaded)

	if s.OnWorkerEvicted != nil {
		s.OnWorkerEvicted(w.Info.WorkerId, w.Info.ModelsLoaded)
	}
}
