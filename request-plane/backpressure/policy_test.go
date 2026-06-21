package backpressure

import (
	"testing"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

func TestPolicy_Admit_WithinThreshold_Accepts(t *testing.T) {
	p := New(map[queue.Priority]time.Duration{queue.PriorityNormal: 10 * time.Second})
	if !p.Admit(queue.PriorityNormal, 5*time.Second) {
		t.Error("estimate within threshold should be admitted")
	}
}

func TestPolicy_Admit_ExceedsThreshold_Rejects(t *testing.T) {
	p := New(map[queue.Priority]time.Duration{queue.PriorityNormal: 10 * time.Second})
	if p.Admit(queue.PriorityNormal, 11*time.Second) {
		t.Error("estimate exceeding threshold should be rejected")
	}
}

func TestPolicy_Admit_ExactlyAtThreshold_Accepts(t *testing.T) {
	p := New(map[queue.Priority]time.Duration{queue.PriorityNormal: 10 * time.Second})
	if !p.Admit(queue.PriorityNormal, 10*time.Second) {
		t.Error("estimate exactly at threshold should be admitted (inclusive boundary)")
	}
}

func TestPolicy_DifferentThresholdsPerPriority(t *testing.T) {
	p := New(DefaultThresholds())

	// 10s estimate: too slow for high priority, fine for normal and low.
	if p.Admit(queue.PriorityHigh, 10*time.Second) {
		t.Error("10s estimate should exceed the high-priority threshold (5s)")
	}
	if !p.Admit(queue.PriorityNormal, 10*time.Second) {
		t.Error("10s estimate should be within the normal-priority threshold (15s)")
	}
	if !p.Admit(queue.PriorityLow, 10*time.Second) {
		t.Error("10s estimate should be well within the low-priority threshold (60s)")
	}
}

func TestPolicy_NoThresholdConfigured_AdmitsByDefault(t *testing.T) {
	p := New(map[queue.Priority]time.Duration{queue.PriorityHigh: time.Second})
	// PriorityLow has no entry — should fail open, not reject.
	if !p.Admit(queue.PriorityLow, time.Hour) {
		t.Error("priority with no configured threshold should be admitted unconditionally")
	}
}

func TestDefaultThresholds_HighStricterThanLow(t *testing.T) {
	d := DefaultThresholds()
	if d[queue.PriorityHigh] >= d[queue.PriorityLow] {
		t.Errorf("high priority threshold (%s) should be stricter (smaller) than low priority (%s)",
			d[queue.PriorityHigh], d[queue.PriorityLow])
	}
}
