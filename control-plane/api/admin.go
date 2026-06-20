package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
	"github.com/atharva/llm-serving-platform/control-plane/router"
	pb "github.com/atharva/llm-serving-platform/proto"
)

// NewAdminHandler returns an http.Handler for the admin API.
// All endpoints are under /admin/* and run on a separate port from the
// inference router so management traffic never competes with inference traffic.
//
//	GET  /admin/workers           — list all workers + state + load + in-flight
//	GET  /admin/workers/{id}      — single worker detail
//	POST /admin/workers/{id}/drain — signal a worker to drain and exit
//	GET  /admin/stats             — fleet-wide summary counts
//	GET  /admin/models            — per-model worker counts
//	GET  /metrics                 — Prometheus scrape endpoint
func NewAdminHandler(reg *registry.Registry, rtr *router.Router, metricsHandler http.Handler) http.Handler {
	h := &adminHandler{reg: reg, rtr: rtr}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /admin/workers", h.listWorkers)
	mux.HandleFunc("GET /admin/workers/{id}", h.getWorker)
	mux.HandleFunc("POST /admin/workers/{id}/drain", h.drainWorker)
	mux.HandleFunc("GET /admin/stats", h.stats)
	mux.HandleFunc("GET /admin/models", h.listModels)
	mux.Handle("GET /metrics", metricsHandler)
	return mux
}

type adminHandler struct {
	reg *registry.Registry
	rtr *router.Router
}

// ── Response shapes ────────────────────────────────────────────────────────────

type workerResp struct {
	ID             string    `json:"id"`
	Address        string    `json:"address"`
	State          string    `json:"state"`
	Backend        string    `json:"backend"`
	Provider       string    `json:"provider"`
	Hardware       string    `json:"hardware"`
	Models         []string  `json:"models"`
	CostPerHour    float64   `json:"cost_per_hour"`
	RegisteredAt   time.Time `json:"registered_at"`
	LastHeartbeat  time.Time `json:"last_heartbeat"`
	DrainRequested bool      `json:"drain_requested"`
	InFlight       int64     `json:"in_flight"`
	Load           loadResp  `json:"load"`
}

type loadResp struct {
	QueueDepth     uint32  `json:"queue_depth"`
	RequestsPerSec float64 `json:"requests_per_sec"`
	AvgLatencyMs   float64 `json:"avg_latency_ms"`
	VramUsedMB     uint64  `json:"vram_used_mb"`
}

type modelResp struct {
	Model          string `json:"model"`
	TotalWorkers   int    `json:"total_workers"`
	HealthyWorkers int    `json:"healthy_workers"`
	TotalInFlight  int64  `json:"total_in_flight"`
}

type statsResp struct {
	Total              int     `json:"total"`
	Healthy            int     `json:"healthy"`
	Starting           int     `json:"starting"`
	Draining           int     `json:"draining"`
	Dead               int     `json:"dead"`
	TotalInFlight      int64   `json:"total_in_flight"`
	FleetCostPerHour   float64 `json:"fleet_cost_per_hour"` // sum of all healthy workers' $/hr
}

// ── Handlers ───────────────────────────────────────────────────────────────────

func (h *adminHandler) listWorkers(w http.ResponseWriter, _ *http.Request) {
	inFlight := h.rtr.InFlightSnapshot()
	workers := h.reg.ListWorkers()

	out := make([]workerResp, 0, len(workers))
	for _, e := range workers {
		out = append(out, toWorkerResp(e, inFlight[e.Info.WorkerId]))
	}
	writeJSON(w, http.StatusOK, out)
}

func (h *adminHandler) getWorker(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	entry, ok := h.reg.GetWorker(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "worker not found"})
		return
	}
	inFlight := h.rtr.InFlightSnapshot()
	writeJSON(w, http.StatusOK, toWorkerResp(entry, inFlight[id]))
}

func (h *adminHandler) drainWorker(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	if err := h.reg.RequestDrain(id); err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "drain requested", "worker_id": id})
}

func (h *adminHandler) stats(w http.ResponseWriter, _ *http.Request) {
	workers := h.reg.ListWorkers()
	inFlight := h.rtr.InFlightSnapshot()

	var s statsResp
	s.Total = len(workers)
	for _, e := range workers {
		switch e.State {
		case pb.WorkerState_READY, pb.WorkerState_BUSY:
			s.Healthy++
			s.FleetCostPerHour += e.Info.CostPerHour
		case pb.WorkerState_STARTING:
			s.Starting++
		case pb.WorkerState_DRAINING:
			s.Draining++
		case pb.WorkerState_DEAD:
			s.Dead++
		}
		s.TotalInFlight += inFlight[e.Info.WorkerId]
	}
	writeJSON(w, http.StatusOK, s)
}

func (h *adminHandler) listModels(w http.ResponseWriter, _ *http.Request) {
	stats := h.reg.ModelsServed()
	inFlight := h.rtr.InFlightSnapshot()

	// Build per-model in-flight by summing across all workers that serve it.
	workersByModel := make(map[string][]string) // model → workerIDs
	for _, e := range h.reg.ListWorkers() {
		for _, m := range e.Info.ModelsLoaded {
			workersByModel[m] = append(workersByModel[m], e.Info.WorkerId)
		}
	}

	out := make([]modelResp, 0, len(stats))
	for _, s := range stats {
		var totalIF int64
		for _, wid := range workersByModel[s.Model] {
			totalIF += inFlight[wid]
		}
		out = append(out, modelResp{
			Model:          s.Model,
			TotalWorkers:   s.TotalWorkers,
			HealthyWorkers: s.HealthyWorkers,
			TotalInFlight:  totalIF,
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// ── Helpers ────────────────────────────────────────────────────────────────────

func toWorkerResp(e registry.WorkerEntry, inFlight int64) workerResp {
	return workerResp{
		ID:             e.Info.WorkerId,
		Address:        e.Info.Address,
		State:          e.State.String(),
		Backend:        e.Info.Backend,
		Provider:       e.Info.Provider,
		Hardware:       e.Info.Hardware,
		Models:         e.Info.ModelsLoaded,
		CostPerHour:    e.Info.CostPerHour,
		RegisteredAt:   e.RegisteredAt,
		LastHeartbeat:  e.LastHeartbeat,
		DrainRequested: e.DrainRequested,
		InFlight:       inFlight,
		Load: loadResp{
			QueueDepth:     e.Load.QueueDepth,
			RequestsPerSec: e.Load.RequestsPerSec,
			AvgLatencyMs:   e.Load.AvgLatencyMs,
			VramUsedMB:     e.Load.VramUsedBytes / 1024 / 1024,
		},
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}
