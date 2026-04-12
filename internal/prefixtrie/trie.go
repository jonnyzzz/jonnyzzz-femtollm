// Package prefixtrie implements a hash trie for prefix-cache-aware routing.
//
// Incoming prompts are split into fixed-size text chunks, each hashed with FNV-64a.
// The trie records which backends have been routed prompts through each node.
// On lookup, the trie finds the deepest node whose backend set intersects with the
// set of available backends, yielding the best prefix-cache match.
//
// This follows the vLLM Production Stack PrefixAwareRouter design:
// text-level matching, no tokenizer needed, no backend coordination.
package prefixtrie

import (
	"hash/fnv"
	"sync"
	"sync/atomic"
	"time"
)

const DefaultChunkSize = 128

// Trie is a thread-safe hash trie for prefix-cache-aware routing.
type Trie struct {
	mu        sync.RWMutex
	root      *node
	chunkSize int
	// Stats (atomic for lock-free updates during read-locked Match)
	lookups atomic.Uint64
	hits    atomic.Uint64 // lookups that matched at depth > 0
	inserts atomic.Uint64
	nodes   atomic.Uint64
}

type node struct {
	children map[uint64]*node
	backends map[string]time.Time // backend name -> last routed time
}

func newNode() *node {
	return &node{
		children: make(map[uint64]*node),
		backends: make(map[string]time.Time),
	}
}

// New creates a trie with the given chunk size (0 = default 128 chars).
func New(chunkSize int) *Trie {
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	t := &Trie{
		root:      newNode(),
		chunkSize: chunkSize,
	}
	t.nodes.Store(1)
	return t
}

// Match walks the trie for the given prompt and returns the backends at the
// deepest matching node that intersects with the available set.
// Returns (matched backends, match depth in chunks). Depth 0 means no match.
func (t *Trie) Match(prompt string, available map[string]bool) (matched []string, depth int) {
	chunks := chunkHashes(prompt, t.chunkSize)

	t.mu.RLock()
	defer t.mu.RUnlock()

	t.lookups.Add(1)

	cur := t.root
	var best []string
	bestDepth := 0

	for i, h := range chunks {
		child, ok := cur.children[h]
		if !ok {
			break
		}
		candidates := intersect(child.backends, available)
		if len(candidates) == 0 {
			break
		}
		best = candidates
		bestDepth = i + 1
		cur = child
	}

	if bestDepth > 0 {
		t.hits.Add(1)
	}
	return best, bestDepth
}

// Insert records that the given prompt was routed to the given backend.
func (t *Trie) Insert(prompt string, backend string) {
	chunks := chunkHashes(prompt, t.chunkSize)
	now := time.Now()

	t.mu.Lock()
	defer t.mu.Unlock()

	t.inserts.Add(1)

	cur := t.root
	for _, h := range chunks {
		child, ok := cur.children[h]
		if !ok {
			child = newNode()
			cur.children[h] = child
			t.nodes.Add(1)
		}
		child.backends[backend] = now
		cur = child
	}
}

// Prune removes trie nodes older than maxAge. Returns the number of nodes removed.
func (t *Trie) Prune(maxAge time.Duration) int {
	cutoff := time.Now().Add(-maxAge)

	t.mu.Lock()
	defer t.mu.Unlock()

	removed := pruneNode(t.root, cutoff)
	t.nodes.Add(^uint64(removed - 1)) // subtract
	return removed
}

// Stats returns trie statistics.
func (t *Trie) Stats() TrieStats {
	return TrieStats{
		Nodes:   t.nodes.Load(),
		Lookups: t.lookups.Load(),
		Hits:    t.hits.Load(),
		Inserts: t.inserts.Load(),
	}
}

// TrieStats holds trie usage statistics.
type TrieStats struct {
	Nodes   uint64 `json:"nodes"`
	Lookups uint64 `json:"lookups"`
	Hits    uint64 `json:"hits"`    // lookups with depth > 0
	Inserts uint64 `json:"inserts"`
}

// pruneNode recursively removes expired backends and empty children.
// Returns the number of nodes removed.
func pruneNode(n *node, cutoff time.Time) int {
	removed := 0
	for h, child := range n.children {
		// Prune expired backends from this child
		for backend, lastUsed := range child.backends {
			if lastUsed.Before(cutoff) {
				delete(child.backends, backend)
			}
		}
		// Recurse into children
		removed += pruneNode(child, cutoff)
		// Remove empty children
		if len(child.children) == 0 && len(child.backends) == 0 {
			delete(n.children, h)
			removed++
		}
	}
	return removed
}

// chunkHashes splits text into fixed-size chunks and returns their FNV-64a hashes.
func chunkHashes(text string, chunkSize int) []uint64 {
	if len(text) == 0 {
		return nil
	}
	n := (len(text) + chunkSize - 1) / chunkSize
	hashes := make([]uint64, 0, n)
	for i := 0; i < len(text); i += chunkSize {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}
		// Only hash full chunks (partial trailing chunk is ignored for matching
		// since the next request may extend it differently)
		if end-i < chunkSize {
			break
		}
		h := fnv.New64a()
		h.Write([]byte(text[i:end]))
		hashes = append(hashes, h.Sum64())
	}
	return hashes
}

// intersect returns backend names present in both the node's backend map and the available set.
func intersect(nodeBackends map[string]time.Time, available map[string]bool) []string {
	var result []string
	for name := range nodeBackends {
		if available[name] {
			result = append(result, name)
		}
	}
	return result
}
