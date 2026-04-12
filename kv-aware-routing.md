# KV-Cache Aware Routing — Research & Design

Research into prefix-cache-aware request routing for LLM inference proxies, with a design proposal for femtollm.

## Background

When multiple vLLM backends serve the same model, each maintains its own KV-cache of previously computed attention states. If a new request shares a prefix with a cached request (e.g., same system prompt, same conversation history), routing it to the backend that has that prefix cached avoids redundant computation and dramatically reduces TTFT (time to first token).

The challenge: the router must know which prefixes are cached where, and make fast routing decisions accordingly.

## Industry Approaches

### 1. vLLM Production Stack

Two strategies, from simple to complex:

#### A. PrefixAwareRouter (router-only, no backend coordination)

**How it works**: The router maintains an in-process **HashTrie** — a prefix tree where edges are hashes of fixed-size text chunks (128 characters, hashed with xxhash64). Each trie node stores which backend endpoints have been routed requests passing through that node.

**Routing algorithm**:
1. Extract prompt text from the request (concatenate message contents).
2. Walk the trie following chunk hashes, intersecting node endpoint sets with currently-healthy endpoints.
3. Stop at the deepest matching node. Pick randomly among matched endpoints.
4. Insert the full prompt path into the trie associated with the chosen endpoint.

**Key properties**:
- Operates on **raw text**, not tokens. 128-char chunk granularity.
- **Assumes no cache eviction** — the trie grows monotonically. If the backend evicts cached KV, the router doesn't know and may route suboptimally.
- Zero infrastructure beyond the router process itself.
- No tokenization needed.

**Trade-offs**: Simple, fast, no external dependencies. But coarse-grained (128-char chunks), blind to actual cache state, and the "no eviction" assumption breaks down under memory pressure.

#### B. KvawareRouter (ground-truth via LMCache)

**How it works**: Each vLLM instance runs with LMCache, which reports cached token ranges to a centralized controller via ZeroMQ. The router tokenizes each request and queries the controller for which instance has the longest matching prefix at the **token level**.

**Routing algorithm**:
1. Tokenize the prompt (local `AutoTokenizer` or backend `/tokenize` API).
2. Query LMCache controller: returns `{instance_id: matched_token_count}`.
3. If best match < `len(tokens) - threshold` (default 2000), fall back to session-based (consistent hash) or QPS-based routing.
4. Otherwise, route to the instance with the longest match.

**Trade-offs**: Accurate (reflects actual cache state), token-level granularity. But requires LMCache infrastructure on every backend + ZeroMQ coordination.

---

### 2. llm-d (Kubernetes-native)

**Architecture**: An Endpoint Picker (EPP) intercepts requests via Envoy ext-proc. Pipeline: Filter → Score → Select. Multiple scorer plugins contribute weighted scores.

**KV-Cache Indexer** (`llm-d-kv-cache-manager`):
- **Event ingestion**: vLLM pods emit `BlockStored`/`BlockRemoved` events over ZMQ (msgpack). Events sharded by FNV-1a hash of pod-id for ordered processing.
- **Token processor**: Tokens chunked into fixed blocks (default **16 tokens/block**). Each block hashed with **FNV-64a** over CBOR-encoded `[parentHash, tokenChunk, extra]`. Hash chain — each block's hash depends on the full prefix, not just its own tokens.
- **Index backends**: In-memory LRU (100M keys), cost-aware (Ristretto, 2GB budget), or Redis/Valkey.
- **Scoring**: Longest consecutive prefix match per pod. Combined with load-aware scores (queue depth, active requests) via configurable weights.

**Speculative indexing**: When a routing decision is made, predicted cache entries are written immediately (TTL 2s), so the next request sees them before the actual KVEvent arrives.

**Performance** (8 pods × 2 H100s, Qwen-32B, 3-60 QPS):

| Metric | Precise | Approximate | Random |
|--------|---------|-------------|--------|
| TTFT p90 | **0.54s** | 31.1s | 92.6s |
| Throughput (tok/s) | **8,730** | 6,944 | 4,429 |

**57x faster TTFT** and **2x throughput** vs. cache-blind routing. Critically, prefix+load combined scoring is essential — prefix-only causes hot-spotting at high load.

**KV Offloading**: Three-tier hierarchy: GPU HBM → CPU DRAM → Shared Filesystem. Enables cross-node cache sharing via shared filesystem paths.

---

### 3. NVIDIA Dynamo

**Architecture**: Rust-based KV router with Python bindings. Frontend tokenizes, router scores, workers execute.

**Cost function**:
```
cost(worker) = overlap_score_weight * prefill_blocks + decode_blocks
```
Where `prefill_blocks` = tokens that must be computed from scratch (total - cached), `decode_blocks` = active generation load. Lowest cost wins. Optional softmax temperature for load distribution.

**Three indexer implementations** (all Rust):

| Indexer | Structure | Threading | Use Case |
|---------|-----------|-----------|----------|
| RadixTree | Rc<RefCell<RadixBlock>> | Single-threaded | Small scale, supports TTL/pruning |
| ConcurrentRadixTree | Arc<RwLock> + DashMap | Multi-threaded, sticky worker routing | Production, high event throughput |
| PositionalIndexer | DashMap<(Position, LocalBlockHash), Entry> | Multi-threaded with jump optimization | Highest throughput (>10M ops/sec, p99 <10μs) |

**Token hashing**: XXH3 with seed 1337. Tokens chunked into fixed blocks (e.g., 64 tokens). Each block has a `LocalBlockHash` (just its tokens) and a `SeqHash` (rolling hash including the full prefix). The RadixTree navigates by `LocalBlockHash`, the PositionalIndexer uses position+LocalBlockHash for O(D/J) jump lookups.

**Event protocol**: Workers publish KV stored/removed events via NATS Core (default), JetStream, or ZMQ. Gap recovery via sequence numbers — missing events fetched from worker's local indexer.

**Approximate mode** (`--no-router-kv-events`): No event infrastructure. Router predicts cache state from its own routing decisions with TTL expiration (default 120s). Simpler but drifts over time.

**Disaggregated prefill/decode**: Separate GPU pools for prefill and decode. KV transferred via NIXL (GPU-to-GPU over RDMA/NVLink).

---

## Comparison Matrix

| Feature | vLLM PrefixAware | vLLM Kvaware | llm-d | Dynamo |
|---------|-----------------|--------------|-------|--------|
| Matching level | Text (128-char chunks) | Tokens | Tokens (16/block) | Tokens (64/block) |
| Data structure | HashTrie (xxhash) | LMCache controller | FNV-64a hash chain + LRU index | RadixTree or PositionalIndexer (XXH3) |
| Cache truth | Assumed (no eviction) | Ground truth (LMCache) | Ground truth (ZMQ events) | Ground truth (NATS/ZMQ) or approximate |
| Infra required | None | LMCache + ZeroMQ | ZMQ + K8s Envoy | NATS or ZMQ (or none for approximate) |
| Load balancing | Random among matches | Session hash or lowest-QPS | Weighted scorer pipeline | Cost function (cache + decode load) |
| Language | Python + Go | Python | Go | Rust |

---

## Design for femtollm

### Requirements

femtollm is a minimal Go proxy with 2-3 vLLM backends. We need:
1. Route requests to the backend most likely to have the prefix cached.
2. No changes to the vLLM backends (no LMCache, no ZMQ events).
3. Keep it simple — pure Go, zero external dependencies.

### Proposed approach: Router-side HashTrie (vLLM PrefixAware style)

This is the right fit because:
- No backend coordination needed (we can't add LMCache/ZMQ to our vLLMs).
- Works at the text level (no tokenizer needed in the proxy).
- Simple to implement in Go.
- Good enough for 2-3 backends with stable workloads (system prompts, conversation prefixes).

### Data structure: HashTrie

```
HashTrie
  root: *TrieNode

TrieNode
  children: map[uint64]*TrieNode  // chunk hash -> child
  backends: map[string]struct{}   // which backends have been routed here
```

**Text chunking**: Split the prompt text into fixed-size chunks (e.g., 128 chars). Hash each chunk with xxhash64. Walk the trie to find the deepest node whose backend set intersects with healthy backends.

**Prompt extraction**: For `/v1/chat/completions`, concatenate all message `content` fields. For `/v1/completions`, use the `prompt` field directly.

### Algorithm

```
func (t *HashTrie) Route(prompt string, healthyBackends []string) string:
    chunks = splitIntoChunks(prompt, 128)
    node = t.root
    bestMatch = healthyBackends  // start with all healthy
    
    for each chunk in chunks:
        h = xxhash64(chunk)
        child = node.children[h]
        if child == nil:
            break
        candidates = intersect(child.backends, healthyBackends)
        if len(candidates) == 0:
            break
        bestMatch = candidates
        node = child
    
    // Pick from bestMatch, weighted by load score
    chosen = selectByLoad(bestMatch)
    
    // Record the route: insert full path associated with chosen backend
    t.Insert(prompt, chosen)
    
    return chosen
```

### Eviction handling

Since we don't get eviction events, two mitigations:
1. **TTL on trie nodes**: Nodes older than `N` minutes (configurable, e.g., matching vLLM's `--max-num-seqs` cycle time) are pruned.
2. **Max trie size**: If the trie exceeds a memory limit, prune least-recently-used branches.
3. **Combine with existing load metrics**: Even if the trie routes to a backend that evicted the prefix, the load-aware scoring (KV-cache usage from `/metrics`) will detect the cache miss indirectly (higher KV-cache usage = more recomputation happening).

### Integration with existing femtollm

The HashTrie slots into the existing balancer as an additional signal:

```
score(backend) = 
    kvCacheUsage                           // from /metrics (existing)
    + 0.01 * (requestsRunning + waiting)   // from /metrics (existing)
    - prefixMatchBonus                     // NEW: from HashTrie
    - preferredBonus                       // existing preferred flag
```

Where `prefixMatchBonus` is proportional to the depth of the trie match for this backend (deeper match = more cached prefix = lower score = preferred).

### Token-level vs text-level

**Text-level is sufficient for our use case** because:
- Our workloads have strong prefix locality (Hermes agent with system prompts, conversation history).
- 128-char chunks give ~32-token granularity (at ~4 chars/token for English), which is good enough.
- Avoids embedding a tokenizer in the Go proxy (Gemma 4 uses a SentencePiece tokenizer, which would need cgo or a separate service).
- The vLLM PrefixAwareRouter uses exactly this approach and it's been validated in production.

**If token-level is needed later**: Could add a `/tokenize` endpoint call to the backend (vLLM supports this) and switch to token-block hashing. But this adds latency to every request.

### Chunk size trade-off

| Chunk size | Granularity | Trie depth for 4K prompt | Notes |
|-----------|-------------|-------------------------|-------|
| 64 chars | ~16 tokens | 62 nodes | Fine-grained but deeper trie |
| 128 chars | ~32 tokens | 31 nodes | Good balance (vLLM default) |
| 256 chars | ~64 tokens | 15 nodes | Coarser, might miss short shared prefixes |

**Recommendation**: Start with 128 chars (matching vLLM Production Stack).

### Estimated implementation scope

1. `internal/prefixtrie/trie.go` — HashTrie with insert, match, prune (~150 lines)
2. `internal/prefixtrie/trie_test.go` — unit tests (~100 lines)
3. Update `internal/balancer/balancer.go` — integrate trie match into scoring (~30 lines)
4. Update `internal/proxy/proxy.go` — extract prompt text from request body (~40 lines)
5. Config: `prefix_routing` boolean in config.json to enable/disable

### References

- [vLLM Production Stack — PrefixAwareRouter](https://docs.vllm.ai/projects/production-stack/en/latest/)
- [vLLM HashTrie source](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/prefix/hashtrie.py)
- [llm-d KV-Cache Indexer](https://github.com/llm-d/llm-d-kv-cache-manager)
- [llm-d Precise Scheduling blog](https://llm-d.ai/blog/precise-scheduling-for-llms-with-kv-caching)
- [NVIDIA Dynamo KV Router](https://github.com/ai-dynamo/dynamo/tree/main/lib/kv-router)
- [Dynamo Router Design Doc](https://github.com/ai-dynamo/dynamo/blob/main/docs/design-docs/router-design.md)
