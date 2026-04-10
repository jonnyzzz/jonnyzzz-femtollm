package balancer

import (
	"sync"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/health"
)

// Balancer selects backends using round-robin among healthy instances.
type Balancer struct {
	mu      sync.Mutex
	counter map[string]uint64
	checker *health.Checker
}

// NewBalancer creates a balancer. If checker is nil, all backends are treated as healthy.
func NewBalancer(checker *health.Checker) *Balancer {
	return &Balancer{
		counter: make(map[string]uint64),
		checker: checker,
	}
}

// Select filters backends to healthy ones and rotates the order via round-robin.
// The returned slice preserves fallback semantics: the starting backend rotates,
// but all healthy backends remain in the list for fallback.
// If all backends are unhealthy, returns all of them (fail-open).
func (b *Balancer) Select(backends []config.Backend, model string) []config.Backend {
	if len(backends) <= 1 {
		return backends
	}

	// Filter to healthy backends
	var healthy []config.Backend
	for _, be := range backends {
		if b.checker == nil || b.checker.IsAlive(be.Name) {
			healthy = append(healthy, be)
		}
	}

	// Fail-open: if all are dead, return original list
	if len(healthy) == 0 {
		return backends
	}

	// Round-robin: rotate starting point
	b.mu.Lock()
	idx := b.counter[model] % uint64(len(healthy))
	b.counter[model]++
	b.mu.Unlock()

	rotated := make([]config.Backend, len(healthy))
	for i := range healthy {
		rotated[i] = healthy[(int(idx)+i)%len(healthy)]
	}
	return rotated
}
