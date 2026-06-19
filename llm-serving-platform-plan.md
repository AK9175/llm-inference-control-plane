# LLM Serving Platform — Project Plan

A production-style, self-hosted LLM inference platform built in two phases.
Phase 1 builds the infrastructure control plane (how GPUStack works internally).
Phase 2 builds the request control plane on top — the SLO-aware scheduling layer
that Google and Snowflake build above the infrastructure, and that GPUStack does not have.

Runs almost entirely on a MacBook Air M2. GPU rental only for the benchmark phase.

---

## The two-layer architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Request Control Plane  (Phase 2)        │
│  Priority Queue → SLO Estimator → Model Selector → Router   │
│  Tenant Registry · Cost Metering · Admission Control        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                 Infrastructure Control Plane  (Phase 1)     │
│  Worker Registry · Request Router · Fleet Scaler            │
│  (Worker Registry = gRPC + heartbeating + dead detection)   │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
    ┌──────▼──────┐                    ┌──────▼──────┐
    │  Worker     │                    │  Worker     │  ...
    │  (Go svc)   │                    │  (Go svc)   │
    └──────┬──────┘                    └──────┬──────┘
           │                                  │
    ┌──────▼──────┐                    ┌──────▼──────┐
    │  Ollama     │                    │  vLLM       │
    │  llama3.2:3b│                    │  llama3.1:8b│
    └─────────────┘                    └─────────────┘
```

The control plane never talks to Ollama/vLLM directly.
It talks to the worker (Go service), which wraps the inference engine.
This decoupling is the core design — swap the engine, nothing above changes.

---

## Why two layers?

GPUStack is an infrastructure control plane — it manages resources:
which worker has capacity, is the worker healthy, scale the fleet up/down.

What Google and Snowflake build on top is a request control plane — it makes
per-request economic and SLO decisions:
- Can this request meet its p99 target given current load?
- Should I downgrade from 70B to 8B to hit the SLO?
- Is this tenant's budget exhausted?

These are different planes. In Google's actual architecture (Borg/Kubernetes +
a separate request scheduler), they are separate systems. You build both.

---

## Assumptions

- Pace: ~10 weeks at 8–10 hours/week alongside coursework. ~4–5 weeks full-time.
- Background: 3 yrs SWE, Java/Kafka, learning Go.
- Already have: Ollama installed with `llama3.2:3b` pulled.
- GPU spend: ~$10–15 total (a few hours on a rented RTX 4090, benchmark phase only).

---

## Reuse from Raft Implementation

The heartbeat/dead-node detection in Raft is structurally identical to worker
health tracking. Directly reusable:

| Raft source | Reuse as |
|---|---|
| `resetTimer()` in `raft/node.go` | Worker deadline timer reset |
| `lastHeardFrom map[string]time.Time` in `raft/node.go` | Worker last-heartbeat timestamp |
| `checkQuorum()` in `raft/node.go` | `markDeadWorkers()` — same pattern, different threshold |
| `notifyHeartbeat()` in `raft/node.go` | Worker heartbeat signal to control loop |
| `GRPCTransport` + `Serve()` + `Close()` in `raft/transport_grpc.go` | Control plane gRPC server |
| `dial()` helper in `raft/transport_grpc.go` | Worker → control plane client connection |
| `stopCh + done` lifecycle in `raft/node.go` | Graceful shutdown |
| Main event loop `select` structure in `raft/node.go` | Control plane health check loop |
| React dashboard scaffold in `dashboard/` | Fleet UI + chaos panel |

Proto definitions, business logic, and worker state machine are rewritten from scratch.

---

## Model progression

| Phase | Model | Where |
|---|---|---|
| Phase 1 (mock) | No model — mock worker returns fake responses | Local |
| Phase 1 (real) | `llama3.2:3b` via Ollama | Local (M2 Metal) |
| Phase 2 | `llama3.2:3b` + `llama3.1:8b` via Ollama | Local (M2 Metal) |
| Benchmark | `llama3.1:8b` + `llama3.1:70b` via vLLM | Rented GPU |

Two models are needed only in Phase 2, when the model selector needs two sizes
to demonstrate SLO-based downgrade. Phase 1 works fine with one model or a mock.

---

## Phase 1 — Infrastructure Control Plane

### Week 1 — Worker Registry

Build the registry that tracks which workers exist and whether they are alive.
Workers register on startup, send heartbeats every N seconds, deregister on shutdown.
Control plane marks a worker dead after missed heartbeats.

**Proto definitions (write from scratch):**
```protobuf
service WorkerRegistry {
  rpc Register   (WorkerInfo)       returns (RegisterResponse);
  rpc Heartbeat  (HeartbeatRequest) returns (HeartbeatResponse);
  rpc Deregister (DeregisterRequest) returns (Empty);
  rpc ReportLoad (LoadReport)       returns (Empty);
}
```

`WorkerInfo` carries: ID, GPU type, VRAM, models loaded, listen address.
`LoadReport` carries: queue depth, requests/sec, avg latency.

**Control plane health check loop** (adapted from Raft's event loop):
```
select {
  case <-deadlineTimer.C  → sweep lastHeartbeat map, mark stale workers dead
  case <-heartbeatSignal  → reset timer for that worker
  case <-stopCh           → shutdown
}
```

**Done when:** multiple mock workers register, one stops sending heartbeats,
control plane marks it dead within the configured timeout.

**Learn:** Go gRPC server/client, protobuf, Go channels for concurrent state.

---

### Week 2 — Request Router

Given a request for model X, find a healthy worker that has it loaded and forward
the request. Implement two routing strategies selectable by config:

- Round robin — equal distribution regardless of load
- Least connections — route to worker with shortest current queue

Handle mid-request worker failure: mark worker suspect, retry on another worker.

**Done when:** requests route correctly across multiple workers, killing a worker
mid-load causes automatic failover with no dropped requests.

**Learn:** load balancing algorithms, `sync.RWMutex` for concurrent worker state reads.

---

### Week 3 — Fleet Scaler + Kubernetes

Deploy workers as Kubernetes pods (k3d locally). Control plane watches aggregate
queue depth across all workers. When queue depth crosses a high watermark, call
the Kubernetes API to scale the worker Deployment up. When idle, scale down.

This is more production-like than HPA — your control plane owns the scaling
decision. Same control loop pattern as Kubernetes controllers internally.

```go
clientset.AppsV1().Deployments(namespace).UpdateScale(ctx, name, newScale)
```

Ollama runs on the Mac host (Metal-accelerated). Workers inside k3d reach it via
`host.docker.internal:11434`.

**Done when:** under rising load, the scaler adds worker pods. Under idle, it removes them.

**Learn:** `client-go` (Go k8s client), control loops, k8s Deployment + Service manifests.

---

### Week 4 — Observability + Fleet UI

**Metrics (Prometheus):**
- Per-worker: queue depth, p50/p95/p99 latency, tokens/sec
- Fleet-level: total throughput, active workers, failed requests
- Scaler: scale-up/down events, current replica count

**Grafana dashboard:** fleet-wide latency percentiles, throughput, worker count over time.

**Fleet UI (React — reuse Raft dashboard scaffold):**
- Worker cards showing status, queue depth, model loaded (adapt `NodeCard.tsx`)
- Live routing log — each request decision streamed via SSE
- Chaos panel — kill a worker, watch failover (reuse `ChaosPanel.tsx`)

**Control plane HTTP API for the UI:**
```
GET  /api/workers         → all workers + state
GET  /api/routing/recent  → last N routing decisions
POST /api/chaos/kill/:id  → kill a worker
GET  /api/events          → SSE stream for live updates
```

**Done when:** Grafana shows live fleet metrics, UI shows workers and routing decisions,
killing a worker in the UI triggers visible failover.

**Learn:** Prometheus Go client, Grafana panels, SSE in Go.

---

## Phase 2 — Request Control Plane

Sits on top of Phase 1. Does not replace it — extends the request path before
the router makes its decision.

### Week 5 — Tenant Registry + Priority Queues

Tenants have API keys, SLO tiers, rate limits, and token budgets.

| Tier | p99 SLO | Rate limit | Token budget |
|---|---|---|---|
| free | best-effort | 5 req/min | 10k tokens/day |
| pro | 500ms | 60 req/min | 500k tokens/day |
| enterprise | 200ms | unlimited | unlimited |

Requests enter a per-tenant priority queue. Enterprise requests are scheduled
ahead of pro, pro ahead of free. Free requests are queued or shed under load.

**Done when:** under load, enterprise requests consistently get lower latency
than free-tier requests for identical prompts.

**Learn:** priority queues in Go, per-tenant rate limiting (token bucket).

---

### Week 6 — SLO Feasibility Estimator

For each incoming request, estimate whether the fleet can meet the tenant's SLO:

```
estimated_latency = avg_latency_per_token(worker) × estimated_output_tokens
                  + current_queue_depth(worker) × avg_request_latency(worker)

admit if estimated_latency < tenant.SLOTarget
```

If no worker can meet the SLO: queue (if within deadline budget) or reject with
backpressure. The estimator reads live load data from the worker registry.

**Done when:** under increasing load, the estimator starts rejecting or queuing
free-tier requests while enterprise requests continue to be admitted.

**Learn:** queuing theory basics (M/M/1 model), latency estimation under load.

---

### Week 7 — Model Selector

If the primary model (70B) cannot meet the tenant's SLO given current load,
automatically downgrade to the smaller model (8B):

```
try 70B workers → estimate latency → misses SLO?
  → try 8B workers → estimate latency → meets SLO?
    → serve on 8B (log downgrade event)
  → queue or reject
```

This requires two Ollama workers locally: one loaded with `llama3.2:3b`,
one with `llama3.1:8b`. The model selector picks between them.

**Done when:** under 70B saturation, requests automatically route to 8B,
downgrade events appear in the routing log and UI.

**Learn:** multi-model routing, cost vs quality tradeoffs in practice.

---

### Week 8 — Economic Scheduler + Tenant UI

**Economic scheduler:** when a free-tier tenant exhausts their token budget,
their requests are deferred (not rejected) until the next budget window.
This affects scheduling decisions, not just billing.

**Tenant UI panel (add to existing dashboard):**
- Per-tenant SLO compliance rate (gauge: % of requests meeting p99 target)
- Token budget usage (progress bar)
- Queue depth per tenant
- p50/p95/p99 latency per tenant

**Done when:** a free-tier tenant hitting their budget sees requests deferred,
visible in the UI. Grafana shows SLO compliance per tier diverging under load.

---

### Week 9 — GPU Benchmark

Develop everything locally first. Rent GPU only to run finished scripts.

**Rent:** RTX 4090 on RunPod or Vast.ai (~$0.20–0.44/hr).

**Deploy:** vLLM with `llama3.1:8b` and `llama3.1:70b`. Point your load tester at them.

**Benchmark matrix** (vary one parameter at a time):
- Model size: 8B vs 70B
- Quantization: FP16 vs INT8 vs INT4
- Concurrency: 1 / 4 / 16 / 32 concurrent requests
- SLO enforcement: enterprise vs free tier compliance rate under saturation

Capture results into `bench/results/`. Shut the instance down immediately after.

**Done when:** results table/CSV quantifies tradeoffs.
e.g. "8B INT4 gives 3.2× throughput vs 70B FP16 at +12ms p99."

---

### Week 10 — Write-up + Polish

- `docs/architecture.md` with full two-layer architecture diagram
- `docs/benchmark-report.md` — tradeoff curves, decisions, why
- Clean top-level `README.md` with demo GIF, architecture diagram, one-command setup
- `Makefile` with targets: `up`, `deploy`, `bench`, `ui`

**Done when:** a stranger can read the README, understand both layers, and
see the benchmark findings in 5 minutes.

---

## Repo layout

```
llm-serving-platform/
├── README.md
├── Makefile
│
├── control-plane/               # Infrastructure + Request control plane (Go)
│   ├── cmd/
│   │   └── server/main.go       # entrypoint
│   ├── registry/
│   │   ├── registry.go          # worker registry — register, heartbeat, health
│   │   └── registry_test.go
│   ├── router/
│   │   ├── router.go            # request routing — round-robin, least-conn
│   │   └── router_test.go
│   ├── scaler/
│   │   └── scaler.go            # fleet scaler — talks to k8s API
│   ├── scheduler/               # Phase 2 — request control plane
│   │   ├── tenant.go            # tenant registry, SLO tiers, token budgets
│   │   ├── queue.go             # per-tenant priority queues
│   │   ├── estimator.go         # SLO feasibility estimator
│   │   ├── selector.go          # model selector (downgrade logic)
│   │   └── metering.go          # per-tenant cost metering
│   ├── api/
│   │   └── http.go              # REST API for the UI (SSE, chaos endpoints)
│   └── metrics/
│       └── metrics.go           # Prometheus instrumentation
│
├── worker/                      # Go wrapper around Ollama / vLLM
│   ├── cmd/main.go
│   ├── worker.go                # register, heartbeat, forward inference requests
│   └── mock/main.go             # mock worker for testing control plane mechanics
│
├── proto/
│   ├── registry.proto           # WorkerRegistry service
│   └── (generated .pb.go files)
│
├── deploy/
│   ├── k3d-cluster.yaml
│   ├── k8s/
│   │   ├── control-plane-deployment.yaml
│   │   ├── worker-deployment.yaml
│   │   └── services.yaml
│   └── monitoring/
│       ├── prometheus.yaml
│       └── grafana-dashboard.json
│
├── ui/                          # React dashboard (adapted from Raft dashboard)
│   ├── src/
│   │   ├── components/
│   │   │   ├── WorkerCard.tsx   # adapted from NodeCard.tsx
│   │   │   ├── FleetGraph.tsx   # adapted from ClusterGraph.tsx
│   │   │   ├── RoutingLog.tsx   # adapted from LogViewer.tsx
│   │   │   ├── ChaosPanel.tsx   # reused directly from Raft
│   │   │   └── TenantPanel.tsx  # new — SLO compliance, budget gauges
│   │   ├── api.ts               # adapted from Raft api.ts
│   │   └── types.ts             # Worker, Tenant, RoutingEvent types
│   └── package.json
│
├── vllm/
│   ├── Dockerfile
│   └── run.sh                   # launch flags (model, quant, batch)
│
├── bench/
│   ├── loadtest.go              # async load generator
│   ├── scenarios/               # concurrency / quant / model configs
│   └── results/                 # CSVs + plots
│
└── docs/
    ├── architecture.md
    ├── architecture-diagram.png
    └── benchmark-report.md
```

---

## Stack

| Layer | Tool | Phase | Runs on |
|---|---|---|---|
| Language | Go | All | Everywhere |
| Worker ↔ Control plane | gRPC + protobuf | 1 | Cluster |
| Model backend (local) | Ollama + llama3.2:3b / llama3.1:8b | 1 + 2 | Mac host (Metal) |
| Model backend (benchmark) | vLLM | 9 | Rented NVIDIA GPU |
| Packaging | Docker | 1+ | Everywhere |
| Orchestration | Kubernetes via k3d | 1+ | Mac (local cluster) |
| Fleet scaling | Control plane → k8s API | 3 | Cluster |
| State store | Redis | 1+ | Cluster |
| Metrics | Prometheus | 4 | Cluster |
| Dashboards | Grafana | 4 | Cluster |
| UI | React + TypeScript (Vite) | 4 + 8 | Browser |
| Load testing | Go load generator / k6 | 4 + 9 | Mac → target |
| GPU rental | RunPod / Vast.ai | 9 | Cloud |

---

## Portfolio deliverables

1. A two-layer LLM serving platform: infrastructure control plane + request control plane.
2. Worker registry with gRPC heartbeating and dead worker detection.
3. SLO-aware scheduler: enterprise vs free tier diverges under load, visible in benchmarks.
4. Model selector: automatic 70B → 8B downgrade when SLO feasibility fails.
5. Fleet UI showing live worker state, routing decisions, and per-tenant SLO compliance.
6. Benchmark report quantifying model size × quantization × concurrency tradeoffs on real GPU.
7. Architecture write-up referencing the Orca paper and Google's disaggregated serving model.

---

## Showcasing

**GitHub README:** architecture diagram + demo GIF (30s clip: worker dies → failover) +
benchmark table + one-command setup. Consumable in 5 minutes.

**Demo video (3–4 min):** open the UI, run a load test, kill a worker live, watch
enterprise SLO stay green while free tier queues. Upload to YouTube (unlisted), link in resume.

**Technical blog post:** "Building the layer above the inference engine" — what GPUStack
does, what Google/Snowflake build on top, how you built it. Publish on Medium or personal site.

**Resume line:**
> LLM Serving Control Plane — Go, gRPC, Kubernetes
> - Built infrastructure control plane (worker registry, gRPC heartbeating, fleet scaler)
>   and request control plane (SLO-aware scheduler, model selector, tenant metering)
> - Demonstrated SLO enforcement under load: enterprise tier maintained p99 < 200ms
>   while free tier queued; validated on rented RTX 4090 with 8B/70B model pair

**Interview angle (Google/Snowflake system design):** propose this architecture in the
design round. You've already reasoned through every tradeoff and have benchmark numbers.

---

## Cost

- Weeks 1–8 and 10: $0 — everything on the M2.
- Week 9: ~$10–15 total. RTX 4090 ~$0.20–0.44/hr on RunPod/Vast.ai.
  Develop locally, rent only to execute finished scripts, shut down immediately after.

---

## Stretch goals

- Add a **prefix cache registry** — track which KV blocks are hot on which worker,
  route requests to the worker with the most prefix overlap. Open research problem.
- Compare **two engines** (Ollama vs vLLM) on identical hardware.
- Add **speculative decoding** orchestration: draft model on one worker, verify on another.
- Add **disaggregated prefill/decode**: separate worker pools for each phase,
  control plane manages KV cache transfer between them.
