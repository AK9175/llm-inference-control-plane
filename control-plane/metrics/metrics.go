package metrics

import (
	"net/http"

	"github.com/atharva/llm-serving-platform/control-plane/registry"
	"github.com/atharva/llm-serving-platform/control-plane/router"
	pb "github.com/atharva/llm-serving-platform/proto"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// FleetCollector implements prometheus.Collector.
// It reads live state from the registry and router on every Prometheus scrape,
// so metrics are always accurate — no background goroutine, no staleness.
type FleetCollector struct {
	reg *registry.Registry
	rtr *router.Router

	workersDesc   *prometheus.Desc // llm_workers{state}
	inFlightDesc  *prometheus.Desc // llm_worker_in_flight{worker_id}
	modelDesc     *prometheus.Desc // llm_model_healthy_workers{model}
	fleetCostDesc *prometheus.Desc // llm_fleet_cost_per_hour
}

func newCollector(reg *registry.Registry, rtr *router.Router) *FleetCollector {
	return &FleetCollector{
		reg: reg,
		rtr: rtr,
		workersDesc: prometheus.NewDesc(
			"llm_workers",
			"Number of workers in each lifecycle state.",
			[]string{"state"}, nil,
		),
		inFlightDesc: prometheus.NewDesc(
			"llm_worker_in_flight",
			"Current in-flight request count per worker (router-tracked, never stale).",
			[]string{"worker_id"}, nil,
		),
		modelDesc: prometheus.NewDesc(
			"llm_model_healthy_workers",
			"Number of READY or BUSY workers that have this model loaded.",
			[]string{"model"}, nil,
		),
		fleetCostDesc: prometheus.NewDesc(
			"llm_fleet_cost_per_hour",
			"Sum of cost_per_hour for all healthy (READY/BUSY) workers.",
			nil, nil,
		),
	}
}

// Describe sends all metric descriptors to the channel — required by Collector.
func (c *FleetCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.workersDesc
	ch <- c.inFlightDesc
	ch <- c.modelDesc
	ch <- c.fleetCostDesc
}

// Collect is called on every Prometheus scrape. It reads live state so the
// values are always current without any background update goroutine.
func (c *FleetCollector) Collect(ch chan<- prometheus.Metric) {
	workers := c.reg.ListWorkers()
	inFlight := c.rtr.InFlightSnapshot()

	// Worker counts by state.
	counts := map[string]float64{
		"starting": 0, "ready": 0, "busy": 0, "draining": 0, "dead": 0,
	}
	var fleetCost float64
	for _, w := range workers {
		switch w.State {
		case pb.WorkerState_STARTING:
			counts["starting"]++
		case pb.WorkerState_READY:
			counts["ready"]++
			fleetCost += w.Info.CostPerHour
		case pb.WorkerState_BUSY:
			counts["busy"]++
			fleetCost += w.Info.CostPerHour
		case pb.WorkerState_DRAINING:
			counts["draining"]++
		case pb.WorkerState_DEAD:
			counts["dead"]++
		}
	}
	for state, n := range counts {
		ch <- prometheus.MustNewConstMetric(c.workersDesc, prometheus.GaugeValue, n, state)
	}
	ch <- prometheus.MustNewConstMetric(c.fleetCostDesc, prometheus.GaugeValue, fleetCost)

	// Per-worker in-flight.
	for workerID, n := range inFlight {
		ch <- prometheus.MustNewConstMetric(c.inFlightDesc, prometheus.GaugeValue, float64(n), workerID)
	}

	// Per-model healthy worker count.
	for _, ms := range c.reg.ModelsServed() {
		ch <- prometheus.MustNewConstMetric(c.modelDesc, prometheus.GaugeValue,
			float64(ms.HealthyWorkers), ms.Model)
	}
}

// RequestsCounter wraps a Prometheus CounterVec for per-model request outcomes.
// The router calls Record() after each request — this is the only cross-package
// coupling: router receives a func(model, result string) hook, not this type.
type RequestsCounter struct {
	vec *prometheus.CounterVec
}

func newRequestsCounter(reg prometheus.Registerer) *RequestsCounter {
	vec := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_requests_total",
		Help: "Total inference requests routed, by model and result (success|error).",
	}, []string{"model", "result"})
	reg.MustRegister(vec)
	return &RequestsCounter{vec: vec}
}

// Hook returns a func the router calls after every request.
func (r *RequestsCounter) Hook() func(model, result string) {
	return func(model, result string) {
		r.vec.WithLabelValues(model, result).Inc()
	}
}

// Setup registers all metrics with the given Prometheus registry and returns:
//   - http.Handler for the /metrics endpoint
//   - func(model, result) to pass to the router as a request hook
func Setup(reg *registry.Registry, rtr *router.Router) (http.Handler, func(model, result string)) {
	promReg := prometheus.NewRegistry()
	promReg.MustRegister(prometheus.NewGoCollector())
	promReg.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	promReg.MustRegister(newCollector(reg, rtr))

	rc := newRequestsCounter(promReg)

	h := promhttp.HandlerFor(promReg, promhttp.HandlerOpts{Registry: promReg})
	return h, rc.Hook()
}
