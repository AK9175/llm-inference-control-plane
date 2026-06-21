// Package admin exposes introspection endpoints for the Request Control
// Plane — queue depth, request counters, and (masked) API key usage. This
// is the data source for the dashboard (CP25); it runs on its own port,
// separate from the client-facing gateway, so dashboard traffic never
// competes with inference traffic.
package admin

import "sync/atomic"

// Stats holds in-memory request counters. Updated via Hook(), which is
// wired into gateway.WithRequestHook so every request — success or
// failure — is counted exactly once.
type Stats struct {
	total   atomic.Int64
	success atomic.Int64
	errors  atomic.Int64
}

// NewStats creates an empty counter set.
func NewStats() *Stats { return &Stats{} }

// Hook returns the func gateway.WithRequestHook expects.
func (s *Stats) Hook() func(model string, success bool) {
	return func(_ string, success bool) {
		s.total.Add(1)
		if success {
			s.success.Add(1)
		} else {
			s.errors.Add(1)
		}
	}
}

// Snapshot returns the current counter values.
func (s *Stats) Snapshot() (total, success, errors int64) {
	return s.total.Load(), s.success.Load(), s.errors.Load()
}
