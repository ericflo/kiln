# PR 1383 production eval speed audit, 2026-05-26

## Context

The first full production-shaped Qwen3.5-4B tool-call eval was statistically useful but operationally too slow for an adapter-training loop:

- Source: `analysis_v12_random_stratified_1000_cap262144.prompt_chosen.jsonl`
- Full audit suite synthesis: 1000 rows, 800 eligible tool-call turns, 775 kept after 25 prompt-length skips.
- Full audit result: 269/775 pass, 34.7% accuracy, Wilson 95% CI 31.4%-38.1%.
- Prompt volume: 55,707,705 prompt tokens.
- Completion volume: 122,540 completion tokens.

The user goal is a closed loop:

1. Train an adapter.
2. Evaluate it against production-shaped traffic.
3. Train the next adapter based on eval failures.
4. Repeat without waiting roughly a day per eval.

The long prefills are part of the production workload and should not be removed from the optimization target.

## Harness changes already on this branch

This branch now has two eval-loop speed tools:

- `trace_api_eval --concurrency N`, so the API runner can submit many saved production prompts concurrently instead of serializing HTTP requests.
- `kiln-eval panel-suite`, which builds a weighted stratified panel from a full suite while preserving the exact original messages and tools. It stratifies by target tool, prompt-size bucket, and split, then reweights selected examples back to source-stratum mass.

These are necessary but not sufficient. They make the eval harness capable of driving a fast backend; they do not make Kiln's current long-prefill serving path fast.

## RunPod real-backend experiments

Pod lease: `pod-8d306100bd5b01863490c6ff` / RunPod pod `po70e5x2r0mjrc`.

Branch/commit: `ce/production-tool-call-eval` at `f01ffe0a`.

Setup:

- Built release CUDA binaries on the pod.
- Downloaded Qwen3.5-4B to `/workspace/Qwen3.5-4B`.
- Materialized the production trace JSONL on the pod.
- Regenerated the full audit-shaped suite with `--max-prompt-chars 524288`, matching the checked-in audit config:
  - 1000 rows
  - 800 eligible tool turns
  - 775 kept
  - 25 prompt-length skips
- Generated a 200-example weighted panel:
  - Source examples: 775
  - Kept examples: 200
  - Strata: 46
  - Dropped strata: 0

### Experiment A: client concurrency 200, default server prefix cache

Command shape:

```bash
trace_api_eval \
  --suite analysis_v12_random_stratified_1000_cap524288.panel200.suite.json \
  --model Qwen3.5-4B \
  --api-base http://127.0.0.1:8420/v1 \
  --temperature 0 \
  --max-tokens 1 \
  --concurrency 200 \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}'
```

Result:

- Wall/metrics elapsed: about 309s.
- Outcomes: 200.
- Errors: 185.
- Non-error outcomes: 15.
- Non-error prompt tokens: 860,328.
- Completion tokens: 15.

Primary failure:

```text
out of memory: no free blocks available
```

The server had 46,543 KV blocks at the default eval server memory split. The prefix cache retained giant, mostly non-reused production prefixes until almost no blocks remained, so later requests failed immediately.

Conclusion: sending hundreds of concurrent long-prefill requests to current Kiln with prefix cache enabled is fast only because it errors. It is not a valid eval loop.

### Experiment B: client concurrency 200, prefix cache disabled, eval-only KV budget

Server overrides:

```bash
KILN_PREFIX_CACHE_ENABLED=false
KILN_INFERENCE_MEMORY_FRACTION=0.9
KILN_MAX_DECODE_BATCH=6
```

Server startup showed:

- KV blocks: 59,841.
- Prefix cache max blocks: 0.
- Server log status 500 count during the partial run: 0.
- Server log `no free blocks available` count during the partial run: 0.

The run was stopped after roughly 9.5 minutes because it was clearly not viable as a fast loop. At that point the eval client log had only reached 4 completed outcomes. Server logs showed real `max_tokens=1` requests with about 54k-62k prompt tokens taking about 494s before response.

Conclusion: disabling the prefix cache and increasing the eval-only KV budget fixes the OOM failure mode, but it exposes the deeper issue: current Kiln admission performs whole-prompt prefill per admitted request rather than true chunked/batched prefill across the waiting queue.

## Why client concurrency alone is not the answer

`trace_api_eval --concurrency 200` is working: it can put 200 exact production prompts in flight. The bottleneck is now server-side.

The current batching actor admits waiting requests by calling `prepare_request`, and `prepare_request` runs the full prompt prefill for that request before the actor can make progress on the next admission. That means the huge production prefixes are not being continuously batched at the prefill-token level.

For short prompts this is acceptable. For 50k-150k token production prompts it turns the server into a long serial prefill machine with some batched decode after the fact. Decode batching is not the limiting part of this workload.

## Recommended solution

### 1. Keep the statistically matched fast panel

Use weighted panels for the inner adapter loop:

- Preserve exact messages/tools.
- Keep the long-prefill distribution.
- Stratify by target tool, split, and prompt-size bucket.
- Reweight examples to the full suite.
- Track Wilson intervals and graduate from 100 -> 200 -> 400 examples only when the uncertainty is too wide.

Run the full 775-example audit periodically, not after every adapter.

### 2. Run eval servers in eval-only memory mode

For random stratified panels, do not let prefix cache consume half the KV budget:

```bash
KILN_PREFIX_CACHE_ENABLED=false
KILN_INFERENCE_MEMORY_FRACTION=0.9
```

Re-enable prefix cache only for prefix-grouped/session-grouped eval modes where reuse is intentionally present and measured.

Admission should be block-budget based, not request-count based. `KILN_MAX_DECODE_BATCH` is too crude for long-context evals because six 10k-token requests and six 120k-token requests are completely different memory loads.

### 3. Implement real chunked/batched prefill in the serving path

This is the main speedup.

Instead of:

```text
admit request -> prefill its entire prompt -> admit next request
```

the batching engine needs:

```text
enqueue many requests
tokenize/render prompts
allocate within a KV block budget
schedule prefill chunks across requests up to max_batch_tokens
prioritize decode steps when any request is ready
continue prefill chunks for long prompts until each is decode-ready
```

There is already a scheduler crate with Sarathi-style chunked-prefill concepts. The missing production path is wiring that scheduling model into the real batching actor so the long prompt workload is batched at the prefill-token level, not merely queued at the HTTP level.

This is an eval-loop sized server change, not a tensor-substrate rewrite.

### 4. Add length-aware and prefix-aware eval ordering

The panel's statistical estimator does not depend on execution order. We can run the same selected examples in an order that improves serving efficiency:

- group by adapter/model/template/tool catalog hash,
- group true session-prefix families together,
- pack active admissions by KV block budget,
- avoid combining only the longest prompts in the same active wave.

This preserves prompt fidelity and weights while reducing avoidable OOM and improving any real prefix-cache reuse.

### 5. Calibrate lower `max_tokens` for inner-loop evals

The full 775-example audit produced:

- Completion p95: 485 tokens.
- Completion p99: 1024 tokens.
- Passing completions p95: 376 tokens.
- Passing completions max: 880 tokens.

A 512-token cap would have truncated at most 7 pass outcomes and 7 fail outcomes in that audit, plus many already-invalid length cases. Before adopting it, run a paired A/B on the same panel at 512 vs 1024 and require score agreement within the panel CI. If it holds, use 512 for inner-loop evals and reserve 1024 for periodic full audits.

This is not the primary orders-of-magnitude win because prompt tokens dominate the corpus, but it cuts avoidable tail latency from invalid generations.

## vLLM hypothesis

vLLM would likely be faster if it can serve this exact Qwen3.5-4B checkpoint and tool-template behavior, because its production scheduler is built around continuous batching, chunked prefill, prefix caching, and paged KV. However, this checkpoint is Kiln's GDN/GQA Qwen3.5-4B target, so support must be verified instead of assumed.

The right vLLM test is:

1. Try to load the exact `/workspace/Qwen3.5-4B` checkpoint.
2. Send the same 200-example panel with `max_tokens=1`.
3. Compare:
   - API errors,
   - prompt tokens/sec,
   - wall time,
   - tool-call parsing compatibility,
   - exact score agreement at `max_tokens=512` or `1024`.

If vLLM cannot load the architecture, the implementation target is to bring vLLM-style chunked prefill into Kiln's serving path.

## Bottom line

The eval can keep the real enormous prefills and still become much faster, but not by only making the client concurrent. The first harness changes were the right first step. The next step is server-side chunked/batched prefill admission plus eval-specific memory policy.
