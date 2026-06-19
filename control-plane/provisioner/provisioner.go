package provisioner

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// WorkerProvisioner is called by the Fleet Scaler when a dead worker is evicted
// and needs a replacement. The provisioner starts a new worker that self-registers
// with the control plane — no manual intervention required.
//
// Implementations shipped here:
//   - DockerProvisioner  — local dev / Docker Compose
//   - NoopProvisioner    — placeholder when no provisioning is configured
//
// Future (CP beyond scope): KubernetesProvisioner, GCPProvisioner, AWSProvisioner.
// All share this interface — the scaler and server never import a concrete type.
type WorkerProvisioner interface {
	Provision(workerID string, models []string) error
}

// DockerProvisioner starts a replacement worker as a Docker container.
// The new container runs our worker binary and self-registers with the
// control plane — the scaler evicts the dead entry, Docker brings a new one up.
//
// Typical local setup (Mac):
//
//	p := &DockerProvisioner{
//	    Image:        "llm-worker:latest",
//	    ControlPlane: "host.docker.internal:50051",
//	}
//	sc := scaler.New(reg, scaler.WithOnWorkerEvicted(p.OnEvicted))
//
// "host.docker.internal" is the special DNS name Docker on Mac/Windows resolves
// to the host machine, so containers can reach the control plane running locally.
type DockerProvisioner struct {
	// Image is the Docker image that contains our worker binary.
	Image string
	// ControlPlane is the gRPC address provisioned containers register with.
	// On Mac/Windows: "host.docker.internal:50051" reaches the host.
	// In Kubernetes: the in-cluster service DNS name.
	ControlPlane string
	// ExtraArgs are appended to the docker run command after the image name.
	// Use for --backend, --hardware, --cost-per-hour, --vram, etc.
	ExtraArgs []string

	// run is the command executor — defaults to exec.Command("docker", ...).
	// Inject a stub in tests to avoid requiring a live Docker daemon.
	run func(args []string) error
}

// OnEvicted satisfies the scaler.WithOnWorkerEvicted callback signature.
// It generates a fresh worker ID (avoids collision with the evicted registry entry)
// and calls Provision to start the replacement container.
func (p *DockerProvisioner) OnEvicted(evictedID string, models []string) {
	// Append a short timestamp suffix so the new ID is unique even if the
	// old registry entry hasn't been fully flushed yet.
	newID := evictedID + "-r" + strconv.FormatInt(time.Now().UnixMilli()%100000, 10)

	if err := p.Provision(newID, models); err != nil {
		fmt.Printf("[docker-provisioner] ✗ failed to provision replacement  evicted=%s  err=%v\n",
			evictedID, err)
		return
	}
	fmt.Printf("[docker-provisioner] ✓ replacement started  new_id=%-24s  replaced=%s  models=%v\n",
		newID, evictedID, models)
}

// Provision starts one replacement worker container and returns once Docker
// has accepted the request (the container may still be starting up).
func (p *DockerProvisioner) Provision(workerID string, models []string) error {
	args := []string{
		"run", "--rm", "--detach",
		"--name", workerID,
		p.Image,
		"--control-plane", p.ControlPlane,
		"--id", workerID,
		"--models", strings.Join(models, ","),
	}
	args = append(args, p.ExtraArgs...)
	return p.executor()(args)
}

func (p *DockerProvisioner) executor() func([]string) error {
	if p.run != nil {
		return p.run
	}
	return func(args []string) error {
		out, err := exec.Command("docker", args...).CombinedOutput() //nolint:gosec
		if err != nil {
			return fmt.Errorf("docker %s: %w\noutput: %s", args[0], err, out)
		}
		return nil
	}
}

// NoopProvisioner logs but takes no action. Use it when the fleet should
// alert on worker loss but not attempt automatic replacement.
type NoopProvisioner struct{}

func (NoopProvisioner) Provision(evictedID string, models []string) error {
	fmt.Printf("[noop-provisioner] worker evicted id=%-20s models=%v — no replacement configured\n",
		evictedID, models)
	return nil
}
