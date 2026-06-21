// Command server runs the Request Control Plane: the client-facing HTTP
// gateway that queues inference requests and dispatches them to the
// Infrastructure Control Plane's router (CP1-15) running separately.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/admin"
	"github.com/atharva/llm-serving-platform/request-plane/auth"
	"github.com/atharva/llm-serving-platform/request-plane/backpressure"
	"github.com/atharva/llm-serving-platform/request-plane/dispatcher"
	"github.com/atharva/llm-serving-platform/request-plane/gateway"
	"github.com/atharva/llm-serving-platform/request-plane/queue"
	"github.com/atharva/llm-serving-platform/request-plane/slo"
)

func main() {
	listenAddr := flag.String("addr", ":9000", "HTTP listen address for the request control plane")
	adminAddr := flag.String("admin-addr", ":9001", "HTTP listen address for the request-plane admin/introspection API")
	upstream := flag.String("upstream", "http://localhost:8080", "Infrastructure Control Plane router address")
	queueCap := flag.Int("queue-capacity", 1000, "max queued requests before returning 503")
	concurrency := flag.Int("concurrency", 10, "number of dispatcher goroutines forwarding to upstream")
	waitTimeout := flag.Duration("wait-timeout", 30*time.Second, "max time a request waits for dispatch before returning 504")
	sloFallback := flag.Duration("slo-fallback-latency", 2*time.Second, "default latency estimate for a model with no observed history yet")
	maxWaitHigh := flag.Duration("max-wait-high", 5*time.Second, "reject high-priority requests whose estimated wait exceeds this")
	maxWaitNormal := flag.Duration("max-wait-normal", 15*time.Second, "reject normal-priority requests whose estimated wait exceeds this")
	maxWaitLow := flag.Duration("max-wait-low", 60*time.Second, "reject low-priority requests whose estimated wait exceeds this")
	maxAttempts := flag.Int("max-attempts-per-model", 2, "retry budget for the same model before falling back")
	fallbackMap := flag.String("fallback-map", "", `fallback chains, e.g. "llama3:70b=llama3:8b,llama3:3b;mistral:7b=llama3:3b"`)
	apiKeys := flag.String("api-keys", "", `API keys and rate limits, e.g. "key1=600,key2=60" (requests/min, 0=unlimited); empty disables auth`)
	flag.Parse()

	q := queue.New(*queueCap)
	d := dispatcher.New(q, *upstream, *concurrency,
		dispatcher.WithMaxAttempts(*maxAttempts),
		dispatcher.WithFallbacks(parseFallbackMap(*fallbackMap)),
	)

	tracker := slo.NewLatencyTracker(*sloFallback)
	estimator := slo.NewEstimator(tracker, *concurrency)
	policy := backpressure.New(map[queue.Priority]time.Duration{
		queue.PriorityHigh:   *maxWaitHigh,
		queue.PriorityNormal: *maxWaitNormal,
		queue.PriorityLow:    *maxWaitLow,
	})
	stats := admin.NewStats()
	gwOpts := []gateway.Option{
		gateway.WithSLO(tracker, estimator),
		gateway.WithBackpressure(policy),
		gateway.WithRequestHook(stats.Hook()),
	}
	keyStore := parseAPIKeys(*apiKeys)
	rateLimiter := auth.NewRateLimiter()
	if keyStore != nil {
		gwOpts = append(gwOpts, gateway.WithAuth(keyStore, rateLimiter))
		fmt.Println("[request-plane] API key auth enabled")
	}
	gw := gateway.New(q, *waitTimeout, gwOpts...)
	adminHandler := admin.NewHandler(q, stats, keyStore, rateLimiter)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	go d.Run(ctx)

	srv := &http.Server{Addr: *listenAddr, Handler: gw}
	adminSrv := &http.Server{Addr: *adminAddr, Handler: adminHandler}
	go func() {
		<-ctx.Done()
		fmt.Println("[request-plane] shutting down...")
		srv.Close()      //nolint:errcheck
		adminSrv.Close() //nolint:errcheck
	}()

	go func() {
		fmt.Printf("[request-plane] admin API listening on %s\n", *adminAddr)
		if err := adminSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("request-plane admin server failed: %v", err)
		}
	}()

	fmt.Printf("[request-plane] listening on %s\n", *listenAddr)
	fmt.Printf("[request-plane] forwarding to upstream %s (concurrency=%d, queue-capacity=%d)\n",
		*upstream, *concurrency, *queueCap)

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("request-plane server failed: %v", err)
	}
}

// parseFallbackMap parses "modelA=modelB,modelC;modelD=modelE" into
// {"modelA": ["modelB", "modelC"], "modelD": ["modelE"]}. Empty input
// returns an empty (non-nil) map — no fallbacks configured.
func parseFallbackMap(s string) map[string][]string {
	out := make(map[string][]string)
	if s == "" {
		return out
	}
	for _, pair := range strings.Split(s, ";") {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 || kv[0] == "" || kv[1] == "" {
			continue
		}
		out[kv[0]] = strings.Split(kv[1], ",")
	}
	return out
}

// parseAPIKeys parses "key1=600,key2=60" into a populated KeyStore. Returns
// nil for empty input — auth stays disabled, matching pre-CP22 behaviour.
func parseAPIKeys(s string) *auth.KeyStore {
	if s == "" {
		return nil
	}
	store := auth.NewKeyStore()
	for _, pair := range strings.Split(s, ",") {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 || kv[0] == "" {
			continue
		}
		rpm, _ := strconv.Atoi(kv[1]) // non-numeric -> 0 -> unlimited
		store.AddKey(kv[0], auth.KeyInfo{KeyID: kv[0], RequestsPerMin: rpm})
	}
	return store
}
