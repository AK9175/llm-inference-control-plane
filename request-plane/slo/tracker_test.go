package slo

import (
	"testing"
	"time"
)

func TestLatencyTracker_FallbackWhenNoSamples(t *testing.T) {
	tr := NewLatencyTracker(2 * time.Second)
	if got := tr.Average("never-seen"); got != 2*time.Second {
		t.Errorf("got %s, want fallback 2s", got)
	}
}

func TestLatencyTracker_FirstSample_BecomesAverage(t *testing.T) {
	tr := NewLatencyTracker(2 * time.Second)
	tr.Record("llama3.2:3b", 500*time.Millisecond)

	if got := tr.Average("llama3.2:3b"); got != 500*time.Millisecond {
		t.Errorf("got %s, want 500ms (first sample sets the average directly)", got)
	}
}

func TestLatencyTracker_EWMA_ConvergesTowardRecentSamples(t *testing.T) {
	tr := NewLatencyTracker(2 * time.Second)
	tr.Record("m", 100*time.Millisecond)

	// Feed a much larger latency repeatedly — average should climb toward it,
	// not jump there in one step (that's the point of EWMA smoothing).
	for range 5 {
		tr.Record("m", time.Second)
	}

	got := tr.Average("m")
	if got <= 100*time.Millisecond {
		t.Errorf("average did not move up after slower samples: %s", got)
	}
	if got >= time.Second {
		t.Errorf("average jumped all the way to the new sample instantly (not smoothed): %s", got)
	}
}

func TestLatencyTracker_PerModelIndependent(t *testing.T) {
	tr := NewLatencyTracker(2 * time.Second)
	tr.Record("fast-model", 50*time.Millisecond)
	tr.Record("slow-model", 5*time.Second)

	if got := tr.Average("fast-model"); got != 50*time.Millisecond {
		t.Errorf("fast-model: got %s, want 50ms", got)
	}
	if got := tr.Average("slow-model"); got != 5*time.Second {
		t.Errorf("slow-model: got %s, want 5s", got)
	}
}
