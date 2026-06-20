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
	"syscall"
	"time"

	"github.com/atharva/llm-serving-platform/request-plane/dispatcher"
	"github.com/atharva/llm-serving-platform/request-plane/gateway"
	"github.com/atharva/llm-serving-platform/request-plane/queue"
)

func main() {
	listenAddr := flag.String("addr", ":9000", "HTTP listen address for the request control plane")
	upstream := flag.String("upstream", "http://localhost:8080", "Infrastructure Control Plane router address")
	queueCap := flag.Int("queue-capacity", 1000, "max queued requests before returning 503")
	concurrency := flag.Int("concurrency", 10, "number of dispatcher goroutines forwarding to upstream")
	waitTimeout := flag.Duration("wait-timeout", 30*time.Second, "max time a request waits for dispatch before returning 504")
	flag.Parse()

	q := queue.New(*queueCap)
	d := dispatcher.New(q, *upstream, *concurrency)
	gw := gateway.New(q, *waitTimeout)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	go d.Run(ctx)

	srv := &http.Server{Addr: *listenAddr, Handler: gw}
	go func() {
		<-ctx.Done()
		fmt.Println("[request-plane] shutting down...")
		srv.Close() //nolint:errcheck
	}()

	fmt.Printf("[request-plane] listening on %s\n", *listenAddr)
	fmt.Printf("[request-plane] forwarding to upstream %s (concurrency=%d, queue-capacity=%d)\n",
		*upstream, *concurrency, *queueCap)

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("request-plane server failed: %v", err)
	}
}
