package prefixtrie

import (
	"strings"
	"testing"
	"time"
)

func available(names ...string) map[string]bool {
	m := make(map[string]bool)
	for _, n := range names {
		m[n] = true
	}
	return m
}

// pad returns a string of exactly n characters.
func pad(s string, n int) string {
	if len(s) >= n {
		return s[:n]
	}
	return s + strings.Repeat(".", n-len(s))
}

func TestMatch_EmptyTrie(t *testing.T) {
	trie := New(128)
	matched, depth := trie.Match("hello world", available("a", "b"))
	if depth != 0 {
		t.Errorf("expected depth 0, got %d", depth)
	}
	if len(matched) != 0 {
		t.Errorf("expected no matches, got %v", matched)
	}
}

func TestInsertAndMatch_ExactPrefix(t *testing.T) {
	trie := New(128)
	// Create a prompt with at least 2 full chunks (256 chars)
	prefix := pad("system: you are a helpful assistant", 128)
	prompt1 := prefix + pad("user: hello", 128)
	prompt2 := prefix + pad("user: how are you", 128)

	trie.Insert(prompt1, "backend-a")

	// prompt2 shares the first chunk with prompt1
	matched, depth := trie.Match(prompt2, available("backend-a", "backend-b"))
	if depth != 1 {
		t.Errorf("expected depth 1 (shared prefix chunk), got %d", depth)
	}
	if len(matched) != 1 || matched[0] != "backend-a" {
		t.Errorf("expected [backend-a], got %v", matched)
	}
}

func TestMatch_LongerPrefixWins(t *testing.T) {
	trie := New(128)

	chunk1 := pad("system prompt", 128)
	chunk2 := pad("conversation history part 1", 128)
	chunk3a := pad("user message A", 128)
	chunk3b := pad("user message B", 128)

	// backend-a has chunks 1+2+3a
	trie.Insert(chunk1+chunk2+chunk3a, "backend-a")
	// backend-b has only chunk 1
	trie.Insert(chunk1+pad("different chunk2", 128), "backend-b")

	// Query with chunks 1+2+3b: backend-a matches at depth 2 (chunks 1+2), backend-b at depth 1
	matched, depth := trie.Match(chunk1+chunk2+chunk3b, available("backend-a", "backend-b"))
	if depth != 2 {
		t.Errorf("expected depth 2, got %d", depth)
	}
	if len(matched) != 1 || matched[0] != "backend-a" {
		t.Errorf("expected [backend-a] (deeper match), got %v", matched)
	}
}

func TestMatch_FiltersUnavailableBackends(t *testing.T) {
	trie := New(128)
	prompt := pad("some prompt text here", 128) + pad("more text", 128)

	trie.Insert(prompt, "backend-a")
	trie.Insert(prompt, "backend-b")

	// Only backend-b is available
	matched, depth := trie.Match(prompt, available("backend-b"))
	if depth == 0 {
		t.Fatal("expected a match")
	}
	if len(matched) != 1 || matched[0] != "backend-b" {
		t.Errorf("expected [backend-b], got %v", matched)
	}
}

func TestMatch_NoAvailableBackend_ReturnsEmpty(t *testing.T) {
	trie := New(128)
	prompt := pad("cached prompt", 128) + pad("more", 128)
	trie.Insert(prompt, "backend-a")

	matched, depth := trie.Match(prompt, available("backend-c"))
	if depth != 0 {
		t.Errorf("expected depth 0 (no available backend matches), got %d", depth)
	}
	if len(matched) != 0 {
		t.Errorf("expected empty, got %v", matched)
	}
}

func TestMatch_ShortPrompt_NoFullChunk(t *testing.T) {
	trie := New(128)
	trie.Insert(pad("full chunk", 128)+pad("another", 128), "backend-a")

	// Short prompt (< 128 chars) has no full chunks
	_, depth := trie.Match("short", available("backend-a"))
	if depth != 0 {
		t.Errorf("expected depth 0 for sub-chunk prompt, got %d", depth)
	}
}

func TestMatch_MultipleBackendsSamePrefix(t *testing.T) {
	trie := New(128)
	prefix := pad("shared system prompt", 128) + pad("shared context", 128)
	trie.Insert(prefix+pad("msg-a", 128), "backend-a")
	trie.Insert(prefix+pad("msg-b", 128), "backend-b")

	matched, depth := trie.Match(prefix+pad("msg-c", 128), available("backend-a", "backend-b"))
	if depth != 2 {
		t.Errorf("expected depth 2, got %d", depth)
	}
	if len(matched) != 2 {
		t.Errorf("expected 2 backends, got %v", matched)
	}
}

func TestPrune_RemovesExpired(t *testing.T) {
	trie := New(128)
	prompt := pad("old prompt", 128) + pad("continuation", 128)
	trie.Insert(prompt, "backend-a")

	before := trie.Stats()
	if before.Nodes <= 1 {
		t.Fatalf("expected nodes > 1 after insert, got %d", before.Nodes)
	}

	// Prune with zero TTL removes everything
	removed := trie.Prune(0)
	if removed == 0 {
		t.Error("expected some nodes removed")
	}

	after := trie.Stats()
	if after.Nodes != 1 {
		t.Errorf("expected only root node after full prune, got %d", after.Nodes)
	}
}

func TestPrune_KeepsRecent(t *testing.T) {
	trie := New(128)
	prompt := pad("recent prompt", 128) + pad("more", 128)
	trie.Insert(prompt, "backend-a")

	// Prune with generous TTL should keep everything
	removed := trie.Prune(1 * time.Hour)
	if removed != 0 {
		t.Errorf("expected 0 removed with 1h TTL, got %d", removed)
	}
}

func TestStats_TracksOperations(t *testing.T) {
	trie := New(128)
	prompt := pad("test prompt for stats", 128) + pad("more content", 128)

	trie.Insert(prompt, "a")
	trie.Insert(prompt, "b")
	trie.Match(prompt, available("a"))
	trie.Match("nomatch", available("a"))

	s := trie.Stats()
	if s.Inserts != 2 {
		t.Errorf("expected 2 inserts, got %d", s.Inserts)
	}
	if s.Lookups != 2 {
		t.Errorf("expected 2 lookups, got %d", s.Lookups)
	}
	if s.Hits != 1 {
		t.Errorf("expected 1 hit, got %d", s.Hits)
	}
}

func TestChunkHashes_Deterministic(t *testing.T) {
	text := pad("test input", 128) + pad("second chunk", 128)
	h1 := chunkHashes(text, 128)
	h2 := chunkHashes(text, 128)

	if len(h1) != 2 || len(h2) != 2 {
		t.Fatalf("expected 2 chunks, got %d and %d", len(h1), len(h2))
	}
	if h1[0] != h2[0] || h1[1] != h2[1] {
		t.Error("hashes should be deterministic")
	}
	if h1[0] == h1[1] {
		t.Error("different chunks should produce different hashes")
	}
}

func TestChunkHashes_IgnoresTrailingPartial(t *testing.T) {
	text := pad("full chunk", 128) + "partial"
	h := chunkHashes(text, 128)
	if len(h) != 1 {
		t.Errorf("expected 1 hash (partial trailing chunk ignored), got %d", len(h))
	}
}

func TestChunkHashes_EmptyString(t *testing.T) {
	h := chunkHashes("", 128)
	if len(h) != 0 {
		t.Errorf("expected 0 hashes for empty string, got %d", len(h))
	}
}

func TestConcurrentAccess(t *testing.T) {
	trie := New(128)
	prompt := pad("concurrent test prompt", 128) + pad("more data", 128)

	done := make(chan bool, 20)
	for i := 0; i < 10; i++ {
		go func() {
			trie.Insert(prompt, "backend-a")
			done <- true
		}()
		go func() {
			trie.Match(prompt, available("backend-a"))
			done <- true
		}()
	}
	for i := 0; i < 20; i++ {
		<-done
	}
	// No race detector failures = pass
}
