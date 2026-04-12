package balancer

import (
	"sort"
	"sync"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/health"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/prefixtrie"
)

// KVCacheThreshold is the KV-cache usage level above which a backend is considered
// overloaded and non-preferred backends should be tried instead.
const KVCacheThreshold = 0.9

// PrefixMatchBonus is the score reduction per matched prefix chunk.
// Each chunk is ~128 chars (~32 tokens). A 10-chunk match (-0.5 total)
// is enough to outweigh moderate load differences.
const PrefixMatchBonus = 0.05

// Balancer selects backends using load-aware + prefix-cache-aware routing.
type Balancer struct {
	mu      sync.Mutex
	counter map[string]uint64
	checker *health.Checker
	trie    *prefixtrie.Trie
}

// NewBalancer creates a balancer. If checker is nil, all backends are treated as healthy.
// If trie is nil, prefix-cache routing is disabled.
func NewBalancer(checker *health.Checker, trie *prefixtrie.Trie) *Balancer {
	return &Balancer{
		counter: make(map[string]uint64),
		checker: checker,
		trie:    trie,
	}
}

// Select returns backends ordered by load-aware priority (no prefix matching).
func (b *Balancer) Select(backends []config.Backend, model string) []config.Backend {
	return b.SelectWithPrompt(backends, model, "")
}

// SelectWithPrompt returns backends ordered by combined load + prefix-cache scoring.
// If prompt is non-empty and the trie is configured, prefix matches reduce scores.
// After the caller routes a request, it should call RecordRoute to update the trie.
func (b *Balancer) SelectWithPrompt(backends []config.Backend, model string, prompt string) []config.Backend {
	if len(backends) <= 1 {
		return backends
	}

	// Build set of healthy backend names for trie lookup
	healthySet := make(map[string]bool, len(backends))
	for _, be := range backends {
		if b.checker == nil || b.checker.IsAlive(be.Name) {
			healthySet[be.Name] = true
		}
	}

	// Query trie for prefix matches
	var prefixMatches map[string]int // backend name -> match depth
	if b.trie != nil && prompt != "" {
		matched, depth := b.trie.Match(prompt, healthySet)
		if depth > 0 {
			prefixMatches = make(map[string]int, len(matched))
			for _, name := range matched {
				prefixMatches[name] = depth
			}
		}
	}

	type scored struct {
		backend config.Backend
		score   float64
		order   int
	}

	var healthy []scored
	for i, be := range backends {
		if !healthySet[be.Name] {
			continue
		}
		s := b.score(be)
		// Apply prefix match bonus
		if depth, ok := prefixMatches[be.Name]; ok {
			s -= PrefixMatchBonus * float64(depth)
		}
		healthy = append(healthy, scored{backend: be, score: s, order: i})
	}

	if len(healthy) == 0 {
		return backends // fail-open
	}

	// Round-robin counter for tie-breaking
	b.mu.Lock()
	rr := b.counter[model]
	b.counter[model]++
	b.mu.Unlock()

	sort.SliceStable(healthy, func(i, j int) bool {
		si, sj := healthy[i].score, healthy[j].score
		if diff := si - sj; diff < -0.05 || diff > 0.05 {
			return si < sj
		}
		ri := (uint64(healthy[i].order) + rr) % uint64(len(healthy))
		rj := (uint64(healthy[j].order) + rr) % uint64(len(healthy))
		return ri < rj
	})

	result := make([]config.Backend, len(healthy))
	for i, h := range healthy {
		result[i] = h.backend
	}
	return result
}

// RecordRoute records that a prompt was routed to a backend, updating the prefix trie.
func (b *Balancer) RecordRoute(prompt string, backend string) {
	if b.trie != nil && prompt != "" {
		b.trie.Insert(prompt, backend)
	}
}

// TrieStats returns the prefix trie statistics, or nil if trie is disabled.
func (b *Balancer) TrieStats() *prefixtrie.TrieStats {
	if b.trie == nil {
		return nil
	}
	s := b.trie.Stats()
	return &s
}

// score computes a routing score for a backend (lower = better).
func (b *Balancer) score(be config.Backend) float64 {
	s := float64(0)

	if b.checker != nil {
		st := b.checker.GetStatus(be.Name)
		if st != nil {
			s += st.KVCacheUsage
			s += 0.01 * float64(st.RequestsRunning+st.RequestsWaiting)

			if be.Preferred && st.KVCacheUsage < KVCacheThreshold {
				s -= 1.0
			}
			return s
		}
	}

	if be.Preferred {
		s -= 1.0
	}
	return s
}
