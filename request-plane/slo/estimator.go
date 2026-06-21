package slo

import "time"

// Estimator predicts how long a newly queued request will wait before a
// dispatcher worker starts serving it.
type Estimator struct {
	tracker     *LatencyTracker
	concurrency int // number of dispatcher workers draining the queue in parallel
}

// NewEstimator creates an Estimator. concurrency should match the
// dispatcher's worker pool size — it's the number of requests that can be
// "in service" simultaneously.
func NewEstimator(tracker *LatencyTracker, concurrency int) *Estimator {
	if concurrency < 1 {
		concurrency = 1
	}
	return &Estimator{tracker: tracker, concurrency: concurrency}
}

// Estimate predicts wait time for a request given how many equal-or-higher
// priority requests are already ahead of it (aheadCount, not including
// itself) and the model it targets.
//
// Model: with C dispatcher workers each processing requests of average
// latency L, a request sitting behind N others must wait for roughly N/C
// "rounds" of dispatch to complete before a worker frees up for it.
func (e *Estimator) Estimate(model string, aheadCount int) time.Duration {
	avg := e.tracker.Average(model)
	rounds := float64(aheadCount) / float64(e.concurrency)
	return time.Duration(rounds * float64(avg))
}
