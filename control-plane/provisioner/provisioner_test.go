package provisioner

import (
	"strings"
	"testing"
)

func TestDockerProvisioner_BuildsCorrectCommand(t *testing.T) {
	var captured []string
	p := &DockerProvisioner{
		Image:        "llm-worker:test",
		ControlPlane: "localhost:50051",
		ExtraArgs:    []string{"--backend", "mock"},
		run: func(args []string) error {
			captured = args
			return nil
		},
	}

	if err := p.Provision("w1-reprov", []string{"llama3.2:3b"}); err != nil {
		t.Fatalf("Provision: %v", err)
	}

	wantArgs := []string{
		"run", "--rm", "--detach",
		"llm-worker:test",
		"--control-plane", "localhost:50051",
		"--id", "w1-reprov",
		"--models", "llama3.2:3b",
		"--backend", "mock",
	}
	for _, want := range wantArgs {
		found := false
		for _, got := range captured {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing arg %q in docker command: %s", want, strings.Join(captured, " "))
		}
	}
}

func TestDockerProvisioner_OnEvicted_GeneratesNewID(t *testing.T) {
	var provisionedID string
	p := &DockerProvisioner{
		Image:        "llm-worker:test",
		ControlPlane: "localhost:50051",
		run: func(args []string) error {
			for i, a := range args {
				if a == "--id" && i+1 < len(args) {
					provisionedID = args[i+1]
				}
			}
			return nil
		},
	}

	p.OnEvicted("worker-abc", []string{"llama3.2:3b"})

	if provisionedID == "" {
		t.Fatal("no --id arg found in docker command")
	}
	if provisionedID == "worker-abc" {
		t.Error("new worker ID must differ from evicted ID to avoid registry conflict")
	}
	if !strings.HasPrefix(provisionedID, "worker-abc-r") {
		t.Errorf("new ID %q should be prefixed with evicted ID for traceability", provisionedID)
	}
}

func TestDockerProvisioner_OnEvicted_PassesModels(t *testing.T) {
	var gotModels string
	p := &DockerProvisioner{
		Image:        "llm-worker:test",
		ControlPlane: "localhost:50051",
		run: func(args []string) error {
			for i, a := range args {
				if a == "--models" && i+1 < len(args) {
					gotModels = args[i+1]
				}
			}
			return nil
		},
	}

	p.OnEvicted("w1", []string{"llama3.2:3b", "mistral:7b"})

	if !strings.Contains(gotModels, "llama3.2:3b") || !strings.Contains(gotModels, "mistral:7b") {
		t.Errorf("--models %q does not contain all evicted models", gotModels)
	}
}

// TestNoopProvisioner_ImplementsInterface is a compile-time check that
// NoopProvisioner satisfies WorkerProvisioner.
func TestNoopProvisioner_ImplementsInterface(t *testing.T) {
	var _ WorkerProvisioner = NoopProvisioner{}
	if err := (NoopProvisioner{}).Provision("w1", nil); err != nil {
		t.Errorf("NoopProvisioner.Provision returned error: %v", err)
	}
}

// TestDockerProvisioner_ImplementsInterface is a compile-time check.
func TestDockerProvisioner_ImplementsInterface(t *testing.T) {
	var _ WorkerProvisioner = &DockerProvisioner{}
}
