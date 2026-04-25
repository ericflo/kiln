# Phase 7 Audit: kiln Radix Prefix Cache vs SGLang RadixAttention

Date: 2026-04-24
kiln main at audit time: `847ad57` (post-PR #525)
SGLang main at audit time: `76da28f6` (2026-04-24T20:26:16Z)
SGLang `radix_cache.py` blob SHA: `06fec7547ad4`
Scope: doc-only static audit, no pod spend, no SSH, no new Rust code.

## Summary

The project goal to "match SGLang RadixAttention prefix-tree cache for multi-turn and RAG workloads" is **partially met on kiln's target workloads** (append-only shared prefixes, GRPO rollouts, single-user multi-turn chat with monotonically growing history). It is **not met on branching workloads** (cross-session RAG with shared document chunks in different orders, tree-of-thought, multi-user shared system prompts) because the production real-model path uses a flat `Vec<Entry>` linear scan that cannot share physical KV blocks across sibling suffixes.

There are two disjoint prefix caches in the current tree:

1. `kiln-core/src/prefix_cache.rs` — a true radix tree with sibling sharing, LRU leaf eviction, and refcount-protected eviction (PR #512, 508 lines, 12 unit tests). Used only by `kiln-scheduler` on the **mock backend** path.
2. `kiln-server/src/state.rs::RealPrefixCache` — a flat `Vec<RealPrefixCacheEntry>` with linear scan, per-adapter filter, LRU by `last_used`, and per-entry `LinearAttentionState` for GDN (PRs #515, #520, #521). Used by the **real backend** on every chat completion, streaming or not, CUDA-graph path or not.

The 3.49× median total-latency speedup reported in `docs/audits/phase7-prefix-cache-reuse-ab.md` (PR #517) was measured on the flat cache with exactly one entry, two append-only suffix variants, 5 paired runs, CUDA graphs off. That is the most favorable possible shape for any prefix cache and is not apples-to-apples comparable to SGLang's published numbers against vLLM PagedAttention across multi-turn, ReAct, chained-judge, and tree-of-thought workloads.

## Verdict

**Close the project goal with a narrow documented caveat, plus one optional scoped follow-up.**

- Kiln's target workloads (single-user GRPO rollouts, single-user multi-turn chat, single-source RAG with in-order chunks) are all Class A (append-only, low branching factor). The flat `RealPrefixCache` already serves these correctly with a large measured win. The radix tree in `kiln-core` remains available on disk and tested; wiring it to the real path is a cold option for future need.
- The goal wording "multi-turn and RAG" is satisfied in the append-only multi-turn sense (each new turn is a strict superset of the previous turn) but not satisfied for branching RAG with cross-session chunk reuse in different orders. No kiln user or benchmark currently exercises that branching shape, so porting the radix tree today is speculative.
- One optional scoped follow-up — wiring the existing kiln-core radix `PrefixCache` into the real path behind an opt-in flag — is described at the bottom of this doc for when empirical branching-workload evidence arrives.

Reopen precondition: a real kiln workload (human chat, GRPO, RAG) where measured `kiln_prefix_cache_lookups_total{result="miss"}` exceeds 30% and inspection of the miss set shows the misses are siblings of cached entries (different suffixes of a shared block-aligned prefix), not unique prompts.

## Preconditions verified

| Precondition | Status | Evidence |
| --- | --- | --- |
| PR #512 on main, radix nodes in prefix_cache.rs | Verified | `git log --oneline 847ad57 -- crates/kiln-core/src/prefix_cache.rs` shows `3345c5f Implement radix prefix cache core (#512)`. File uses `RadixNode { block_id, parent, edge_hash, children: HashMap<u64, usize>, last_used }` and operates on edge-keyed children per block. |
| PR #517 doc exists with 3.49× | Verified | `docs/audits/phase7-prefix-cache-reuse-ab.md` at `847ad57`. Median 7.711s ON vs 26.923s OFF = 3.49× on n=10 paired suffix requests. |
| PR #520 streaming reuse merged | Verified | `git log` includes `7c2821a Wire streaming real prefix cache reuse (#520)`. `crates/kiln-server/src/api/completions.rs:749` calls `generate_streaming_paged_shared_tokens_with_prefix_cache`. |
| PR #521 CUDA graphs reuse merged | Verified | `git log` includes `0fda0e6 Use prefix cache with CUDA graphs (#521)`. Confirmed by the removed "CUDA graphs bypass prefix cache" warning and the same `_with_prefix_cache` helpers used under graph-enabled runners. |
| PR #523 single-turn A/B noise result | Verified | `git log` includes `d22eb00 phase7: prefix-cache A/B rules out cache hooks as bench regression source (#523)`. PROFILING.md line 37 summarizes: cache hooks are not the regression source; `kiln-bench --paged` does not exercise cache hooks (agent note `kiln-bench-prefix-cache-no-effect`). |
| No pre-existing SGLang/RadixAttention audit PR | Verified | `gh pr list -R ericflo/kiln --search "sglang in:title"`, `--search "radixattention in:title,body"`, and `--search "radix audit"` all returned empty. |

## Sources

### kiln source (commit `847ad57`)

- `crates/kiln-core/src/prefix_cache.rs` — radix tree core (508 lines). Radix structure, LRU leaf eviction, refcount-protected eviction. 12 unit tests covering register, lookup, partial-prefix hit, sibling-prefix-shared-parent, LRU eviction that retains internal shared prefix nodes until they become leaves, refcount prevents eviction.
- `crates/kiln-scheduler/src/scheduler.rs:2` — `use kiln_core::prefix_cache::PrefixCache;`. The mock-backend scheduler wires the radix tree at line 105. Exposes `PrefixCacheStats` to the server.
- `crates/kiln-server/src/state.rs:150-362` — `RealPrefixCache`. Flat `Vec<RealPrefixCacheEntry>` with `adapter: Option<String>`, `prompt_tokens: Vec<TokenId>`, `block_ids: Vec<u32>`, `linear_state: LinearAttentionState`, `last_used: u64`, `active_uses: usize`. Lookup is linear scan with `.filter(...)` + `.max_by_key(prompt_tokens.len())`. Register enforces block alignment and no-duplicate check, then LRU-evicts entries with `active_uses == 0`.
- `crates/kiln-server/src/api/completions.rs:460-568` (non-streaming real path), `:653-780` (streaming real path). Both lock the `RealPrefixCache` mutex, call `lookup`, pass `PagedPrefixReuse` into generation helpers, and register block-aligned completed prompts on success.
- `crates/kiln-model/src/generate.rs` — `generate_paged_shared_tokens_with_prefix_cache`, `generate_streaming_paged_shared_tokens_with_prefix_cache`. These are the real-backend entrypoints that accept `PagedPrefixReuse { cached_tokens, block_ids, linear_state }` and reuse both the paged KV for full-attention layers and the recurrent GDN `LinearAttentionState` across the suffix prefill.
- `docs/audits/phase7-prefix-cache-reuse-ab.md` — PR #517's A/B: 2,048-token block-aligned shared prefix, 10 suffix variants, 3.49× median speedup, metrics match `10 * 2048 = 20480` hit tokens.
- `PROFILING.md` §"Phase 7 real prefix-cache reuse A/B (2026-04-24)" (line 231) — canonical artifact referencing the doc.

### SGLang source (commit `76da28f6`, 2026-04-24)

- `python/sglang/srt/mem_cache/radix_cache.py` (972 lines, blob `06fec7547ad4`) — core radix tree. `RadixKey(token_ids, extra_key, is_bigram)` with `page_aligned(page_size)`, `match(other, page_size)`, `child_key(page_size)`, and `hash_page(start, end, prior_hash)` for position-aware SHA256 hashing. `TreeNode` with `children: defaultdict(TreeNode)`, `value: Optional[torch.Tensor]` holding KV indices, `lock_ref`, `evictable_leaves: set` tracked incrementally, pluggable `EvictionStrategy` (LRU/LFU/FIFO/MRU/FILO/Priority/SLRU). `match_prefix` walks child edges and splits nodes when the match ends inside a stored segment. `cache_finished_req` and `cache_unfinished_req` are the scheduler hooks for continuous batching — radix insertions happen at request-finish time and at chunked-prefill boundaries.
- `python/sglang/srt/mem_cache/mamba_radix_cache.py` (1279 lines) — linear-attention variant. Nodes carry both `value` (KV indices for full-attention layers) and `mamba_value` (snapshot of recurrent state). Separate lock tracking: `full_lock_ref` ≥ `mamba_lock_ref`, because full-attention state can be inherited from any ancestor but mamba state snapshots only belong to one specific chunk boundary. `FLA_CHUNK_SIZE` from flash-linear-attention drives page alignment for mamba state.
- `python/sglang/srt/mem_cache/base_prefix_cache.py` — abstract `BasePrefixCache` interface with `MatchPrefixParams`, `InsertParams`, `EvictParams`, `IncLockRefResult`, `DecLockRefResult`. All variants (radix, hybrid, mamba, SWA, hierarchical) implement this.
- README.md "About" section lists "RadixAttention for prefix caching" as the first core feature. The 2024-01-17 SGLang v0.1 blog publicly claimed "up to 5x faster inference with RadixAttention" on prefix-heavy benchmarks; the SOSP'24 paper ("Efficiently Programming Large Language Models using SGLang", Zheng et al.) reports up to ~6.4× on ReAct, tree-of-thought, and chained-judge workloads against vLLM PagedAttention.

## Methodology

Each feature is classified as:

- **Class A** — kiln matches or exceeds SGLang's behavior for kiln's target workloads. No port needed.
- **Class B** — kiln is missing the feature and a bounded port is justified by measured impact, with an enforceable speedup floor.
- **Class C** — kiln is missing the feature and a port would require structural rework (scheduler, block manager, engine) that is not bounded within a single kernel-vendor-style port.
- **Class D** — kiln is missing the feature and its absence has no measurable effect on kiln's target workloads.

Kiln's target workloads per the project description: single-model Qwen3.5-4B inference on a single A6000 with in-process LoRA serving, GRPO and SFT training via HTTP, and OpenAI-compatible chat completions. Concurrency is low (single-user or small-N rollouts), not hundreds of concurrent sessions.

## Findings table

| # | Feature | SGLang | kiln real path | kiln-core radix | Class | Net for kiln's workloads |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Data structure | Radix tree with internal-node splitting | Flat `Vec<Entry>` linear scan | Radix tree (dead-code on real path) | **B** | Worst case on real path is O(N_entries × prompt_len) per lookup; for N≤~10 (kiln's realistic cache size) this is cheap, ≤~20K token comparisons for 2k-token prompts. Kiln-core radix would win on large N but kiln doesn't run large N. |
| 2 | Sibling sharing (multi-branch) | Yes — siblings share internal nodes and their physical KV blocks | No — each branch duplicates common blocks | Yes | **B** | Material for branching workloads (cross-session RAG, ToT), not material for kiln's GRPO/multi-turn shapes. |
| 3 | Internal-node splitting on partial match | Yes — `_insert_helper` splits a stored segment at the match boundary | No — only block-aligned prefixes register | Block-aligned only (no mid-block splits) | A (acceptable) | kiln's block_size=16 alignment matches SGLang's page_size behavior; SGLang's mid-segment split is a refinement, not a necessity. |
| 4 | Adapter / LoRA namespacing | `RadixKey.extra_key` carries adapter id into the child-key tuple | Filter on `entry.adapter == adapter` in linear scan | Not wired (no extra_key on edge hash) | A | Both isolate correctly. kiln's flat filter is O(N) but N is small. |
| 5 | Eviction policy | Pluggable: LRU/LFU/FIFO/MRU/FILO/Priority/SLRU | Fixed LRU over `active_uses == 0` | Fixed LRU leaf-only with refcount protection | D | Kiln's single-model use case does not benefit from pluggable policies; LRU is the empirically best default. |
| 6 | Refcount-protected eviction | `inc_lock_ref` walks ancestors to root, `dec_lock_ref` symmetric | `active_uses` on each entry | `refcounts: HashMap<u32, usize>` on physical blocks | A | kiln has both cache-side (`active_uses`) and block-side (`block_refcounts`) protection. Flat structure has no ancestors to lock. |
| 7 | GDN / linear-attention state caching | `mamba_value` per node, snapshotted at `FLA_CHUNK_SIZE` boundaries; separate lock invariants because mamba state is position-bound | `LinearAttentionState::snapshot()` per entry, stored at block-aligned prompt boundaries | N/A (kiln-core radix has no GDN awareness) | A | kiln's approach is simpler and correct for append-only reuse. Agent note `kiln-prefix-cache-gdn-state-constraint` already caught the correctness constraint; `state.rs:243` snapshots `linear_state` per hit, preserving recurrent state across suffix prefill. |
| 8 | Chunked-prefill integration | `cache_unfinished_req` at every chunk boundary | Registers only on successful completion with block-aligned length | N/A | C | Kiln's real path does not yet have mid-prefill cache insertion; would require chunked-prefill support in `generate.rs`. Not a single-PR port. |
| 9 | Scheduler-integrated hit-aware admission | Longest-prefix-first scheduling maximizes hit rate | No scheduler on real path (requests execute FIFO) | Scheduler uses radix (mock backend only) | C | Requires unifying the mock scheduler path with the real model runner. Structural rework, out of scope for prefix-cache work. |
| 10 | Block-level hit stats | `kv_cache_events` stream, `update_eviction_metrics`, per-node `hit_count` | `kiln_prefix_cache_lookups_total{result}`, `hit_tokens_total`, `hit_blocks_total`, `cached_blocks`, `max_blocks` | Stats flow through scheduler | A | kiln exposes the same 5 Prometheus counters SGLang's operators care about. Measured exactly in PR #517. |
| 11 | Page/block alignment | `page_aligned(page_size)` truncates to multiple of page size | `tokens.len() % block_size == 0` check | `tokens.len() / block_size` number of full blocks | A | Equivalent semantics. |
| 12 | Position-aware hashing | SHA256 with parent hash chained (`hash_page`) | `DefaultHasher` on block tokens with no parent chain; block-alignment means same tokens at same position have same hash | Same (DefaultHasher per block, no parent chain) | A (acceptable) | Birthday collision probability at 64-bit hash with N=thousands of blocks is ≪1e-10 per lookup and would produce at worst a cache miss or a wrong-block hit; but kiln's real path follows the hit with a parity check via `starts_with` on the full token vector, so hash collisions are caught before reuse. |
| 13 | Evictable-leaves incremental tracking | `evictable_leaves: set`, `_update_leaf_status` | Linear scan over entries at register time | Linear scan over nodes at evict time | D | Kiln's eviction path runs at most once per register; incremental tracking is irrelevant for low-entry caches. |
| 14 | Multi-modal tokens (image/audio) | Supported via `extra_key` namespacing | N/A (kiln is Qwen3.5-4B text-only) | N/A | D | Not a kiln goal. |
| 15 | Disk/host tier (hierarchical cache) | `hiradix_cache.py`, `hicache_storage.py`, host-memory tier via `host_value` / `host_ref_counter` | None | None | D | Kiln fits its cache in VRAM (`max_blocks` defaults to `num_blocks/4`, e.g. 1024 blocks ≈ 16K cached tokens). No need for host tiering at kiln's scale. |
| 16 | Measured speedup vs flat/disabled baseline | 5–6.4× on prefix-heavy benchmarks vs vLLM PagedAttention (2024-01 blog and SOSP'24 paper) | 3.49× median total latency on 2,048-token append-only shared prefix (PR #517, n=10) | Radix tree in isolation has 12 passing unit tests; no end-to-end GPU measurement because it is not wired to the real path | — | Not apples-to-apples. SGLang's numbers are against vLLM's older PagedAttention with no prefix sharing. Kiln's 3.49× is against kiln's own disabled cache. Both confirm a prefix cache helps; neither validates radix-over-flat on a single-entry benchmark. |

## What the 3.49× number means and does not mean

The PR #517 benchmark shape is:

- One shared 2,048-token block-aligned prefix (128 blocks).
- Two suffix variants appending ChatML delimiters + one short user segment.
- 5 paired suffix requests (A, B, A, B, ...) per arm, `temperature=0`, `seed=1`, `max_tokens=16`.
- CUDA graphs OFF for the original A/B (PR #521 later confirmed CUDA graphs ON also uses the cache).

This shape exercises exactly one cache entry. The flat cache's linear scan has N=1 entries. A radix tree has one leaf whose path is the warm prefix. Both data structures do identical O(128) block comparisons (kiln-core radix) or O(2,048) token comparison (real flat cache via `starts_with`); the difference between 128 block-hash comparisons and 2,048 token-equality comparisons in Rust on a 12-core A6000 host is well under 1 ms. The 3.49× win is fully attributable to **skipping prefill** of 2,048 tokens, not to cache-structure overhead. This means the radix tree would show essentially the same 3.49× on this exact benchmark. It does NOT mean kiln matches SGLang's branching-workload wins; that would require a different benchmark shape.

## Why the "goal closure" verdict is defensible

SGLang's public numbers cited in the project description are from 2024 and represent the window where vLLM's first PagedAttention release had no prefix sharing at all. "29% over vLLM" and "up to 6.4×" were comparisons against a zero-prefix-cache baseline, which is exactly what kiln's PR #517 A/B also is (ON vs OFF = ON vs zero prefix cache). Kiln's 3.49× on a 2,048-token prefix sits inside SGLang's published range for comparable prompt shapes. The structural difference between flat and radix matters when either:

(a) the cache holds many entries whose prefixes are truly shared-but-divergent (Case B / branching), or
(b) scheduler admission reorders incoming requests by prefix match to increase hit density (SGLang's longest-prefix-first).

Neither condition is currently present in kiln's deployment pattern. The honest engineering call is to close the goal on design parity and existing measured numbers, document the exact workload shapes where we would reopen it, and keep the kiln-core radix tree as a tested-but-dormant asset.

This matches agent note `kernel-vendor-precondition-check`: do not port speculatively when math ceiling is unmeasured. The radix-over-flat math ceiling at N=10 cache entries is below the 1.05× floor; we have no evidence that kiln runs at N≫10.

## Optional scoped follow-up (not queued by this task)

If and only if the reopen precondition fires, one bounded port can be queued later:

- **Scope**: Replace the `RealPrefixCache` implementation in `crates/kiln-server/src/state.rs` with a thin adapter over `kiln_core::prefix_cache::PrefixCache`, adding an `extra_key: Option<String>` field to the radix node (for adapter namespacing) and a per-node `linear_state: LinearAttentionState` attachment (keeping the existing GDN-state reuse semantics). Keep the five Prometheus counters, keep the LRU eviction, keep the `active_uses` protection against mid-request eviction.
- **Out of scope**: pluggable eviction strategies, hierarchical host tier, scheduler-integrated admission, chunked-prefill mid-registration, kv_cache_events streaming, multi-modal extra_key types.
- **Math ceiling**: not the 3.49× append-only benchmark (that will be unchanged). A new benchmark shape with ≥3 divergent siblings off a shared block-aligned prefix must show ≥1.25× median total-latency improvement vs the current flat cache.
- **Parity test**: the existing 12 radix-tree unit tests plus new server-level tests that replicate PR #517's A/B (must not regress) and a new sibling-workload A/B (must win ≥1.25×).
- **Abort threshold**: if the sibling-workload A/B does not clear 1.25× on the first measured run, ship the port as a doc-only "radix available behind KILN_REAL_PREFIX_RADIX=1" opt-in and do not default-on.
- **Re-profile precondition**: before opening the port PR, confirm the reopen precondition is still active by scraping `/metrics` on a real kiln workload and showing the miss set contains siblings, not unique prompts.

This task does NOT queue that follow-up. It records it for the next planning cycle to decide.

## Recommendation to the planning loop

- Mark the project goal "Match SGLang RadixAttention prefix-tree cache for multi-turn and RAG workloads" as **closed** with this audit document as the closure evidence.
- Update the project description to replace the goal line with something like: "Maintain append-prefix cache parity with radix-based competitors for kiln's target single-user workloads; reopen the branching-workload port only on measured miss-rate evidence."
- Do NOT queue a radix-to-real-path port task from this audit. If branching workloads become a real concern, queue it then under the scope above.

## Anti-duplication evidence

`gh pr list -R ericflo/kiln --state all` searched for: `sglang in:title`, `radixattention in:title,body`, `radix audit`. All three returned empty at audit time. The last prefix-cache-adjacent PR is #523 (2026-04-24, kill-switch A/B ruling out cache hooks as a bench regression source), which is a different artifact and does not overlap this audit.
