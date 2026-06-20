# Build stage — compiles the mock worker binary.
# CGO_ENABLED=0 produces a fully static binary that runs in scratch/alpine.
FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /mock-worker ./worker/cmd/mock

# Runtime stage — minimal image, just the binary.
FROM alpine:3.20
RUN apk add --no-cache ca-certificates
COPY --from=builder /mock-worker /usr/local/bin/mock-worker
ENTRYPOINT ["mock-worker"]

# Default flags — override at `docker run` time:
#   docker run llm-worker:latest \
#     --control-plane host.docker.internal:50051 \
#     --id my-worker-1 \
#     --models llama3.2:3b \
#     --cost-per-hour 0
CMD ["--help"]
