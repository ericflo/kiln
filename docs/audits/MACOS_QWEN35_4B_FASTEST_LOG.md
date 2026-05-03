# macOS Qwen3.5-4B Inference Optimization Log

Objective: make Kiln the fastest macOS inference server for Qwen3.5-4B, improving
both single-user bs=1 latency/throughput and higher-batch throughput/latency.
Principle for this pass: remove repeated work before tuning kernels.

Environment observed on 2026-05-03:

- Host: Apple M1, 16 GiB unified memory.
- Repo: `/Users/ericflo/Development/kiln`.
- Local model cache before E002: tokenizer only. E002 downloaded the full
  Qwen3.5-4B snapshot:
  `/Users/ericflo/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Disk headroom: about 81 GiB free.
- `hf` CLI exists at `/opt/homebrew/bin/hf`.

## Experiment E001: Remove Duplicate Prefill In Batch n>1

Hypothesis:

`/v1/completions/batch` is GRPO-shaped: one prompt often asks for `n > 1`
completions. Before this experiment, each completion could pay full prompt
prefill because the real prefix cache could only serve strict prefixes. An
identical full prompt did not hit: after a full-prompt cache entry, there was
no saved first-token logits/token to start decode without prefill. The batch
endpoint also launched every `(prompt, completion)` pair concurrently, so even
strict-prefix registration from completion 0 could race too late for completion
1..n.

Change:

- Added `PagedPrefixNextToken` to carry either saved last-position logits or a
  saved greedy first token.
- Extended `PagedPrefixRegistration`, `PagedPrefixReuse`, and
  `RealPrefixCacheHit` with `next_token`.
- Allowed `RealPrefixCache::lookup` to return exact-prompt hits only when a
  saved first-token source exists.
- Taught non-streaming prefix-cache generation to skip prefill entirely on an
  exact hit, then decode from the saved logits/token plus cached KV and linear
  attention state.
- Changed `/v1/completions/batch` scheduling to one task per prompt, with that
  prompt's `n` completions run in order. Different prompts remain concurrent;
  repeated completions for the same prompt can reuse the first completion's
  registered exact prefix.

Expected impact:

- bs=1 single request: neutral except for cache-hit workloads.
- Batch `n > 1` with repeated prompt: removes almost all duplicate prompt
  prefill after the first completion, especially valuable for 512+ token GRPO
  prompts. Decode still runs per completion.
- Batch with distinct prompts and `n=1`: preserves prompt-level concurrency.
- Initial exact reuse only registered block-aligned prompts. E011-E014 removed
  that limitation for exact-prompt hits with a saved next-token source; strict
  longer-prompt reuse still requires block alignment.

Verification:

- `cargo test -p kiln-model exact_prefix_cache_hit_skips_prefill_and_matches_tokens --lib`
  - Result: passed.
  - Evidence: exact cache hit reports zero prefill duration and matches the
    uncached token sequence in a tiny paged model.
- `cargo test -p kiln-server real_prefix_cache --lib`
  - Result: 4 passed.
  - Evidence: exact hits require saved next-token source; strict-prefix and
    adapter-keyed behavior still pass.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 14 passed after E020.
  - Evidence: batch request parsing, validation, seed derivation, and response
    shape still pass after prompt-local scheduling.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
  - Evidence: Metal server and benchmark targets type-check.

Limitations and next measurements:

- Full 512-token Qwen3.5-4B Metal throughput/latency is still pending; E002-E010
  use smaller shapes suitable for this 16 GiB M1.
- Need to run:
  - bs=1: `kiln-bench --paged --latency-only --prompt-tokens 512 --max-output-tokens 128`
    if memory pressure is acceptable.
  - batch repeated-prompt: `/v1/completions/batch` with one prompt and `n` in
    `{4,8}`.
  - batch distinct-prompt: `prompts.len()` in `{2,4,8}`, `n=1`, to ensure the
    prompt-local sequencing does not regress distinct-prompt aggregate latency.

## Experiment E002: First Real Metal Baseline

Command:

`./target/release/kiln-bench --model-path <snapshot> --paged --latency-only --prompt-tokens 64 --max-output-tokens 16 --temperature 0.0 --seed 1`

Artifact:

- `docs/audits/MACOS_QWEN35_4B_FASTEST_artifacts/e002_m1_bs1_p64_o16.json`
- `docs/audits/MACOS_QWEN35_4B_FASTEST_artifacts/e002_m1_bs1_p64_o16.stderr.log`

Result:

| Shape | Prefill | Decode | ITL | P99 ITL |
|---|---:|---:|---:|---:|
| M1, bs=1, p64/o16, paged | 7538.1 ms / 8.49 tok/s | 5.46 tok/s | 183.2 ms | 205.9 ms |

Notes:

- Model load was 21.26 s.
- `/usr/bin/time -l` reported 4.42 GiB max RSS and 11.00 GiB peak memory
  footprint.

## Experiment E003-E005: Existing Metal Fast-Path Toggles

Same shape as E002.

| Experiment | Toggle | Prefill ms | Decode tok/s | Mean ITL ms | P99 ITL ms | Verdict |
|---|---|---:|---:|---:|---:|---|
| E002 | baseline | 7538.1 | 5.46 | 183.2 | 205.9 | reference |
| E003 | `KILN_ENABLE_METAL_LM_HEAD_ARGMAX=1` | 8971.4 | 5.23 | 191.4 | 452.0 | slower; keep disabled |
| E004 | `KILN_DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS=1` | 8180.8 | 5.56 | 179.9 | 317.9 | mixed/noisy; not enough to disable |
| E005 | `KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV=1` | 7715.1 | 4.35 | 229.9 | 748.4 | worse; keep coop GEMV enabled |

Artifacts:

- `e003_m1_bs1_p64_o16_lm_head_argmax.*`
- `e004_m1_bs1_p64_o16_no_contig_attn.*`
- `e005_m1_bs1_p64_o16_no_coop_gemv.*`

## Experiment E006: Baseline Repeat

Same as E002, seed 2.

| Shape | Prefill | Decode | ITL | P99 ITL |
|---|---:|---:|---:|---:|
| M1, bs=1, p64/o16, paged, seed 2 | 8963.0 ms / 7.14 tok/s | 5.42 tok/s | 184.6 ms | 325.1 ms |

Takeaway:

- Decode is stable around 5.4 tok/s on this M1 small-shape baseline.
- Prefill has high run-to-run spread, so future claims need median-of-3.

## Experiment E007-E010: Real Server Batch Prefix Reuse

Server:

`KILN_MODEL_PATH=<snapshot> KILN_PORT=8421 ./target/release/kiln serve`

Request family:

`POST /v1/completions/batch`, one prompt, `n=2`, `temperature=0.0`,
`max_tokens=2`.

| Experiment | Prompt tokens per completion | Wall time | Prefix-cache delta | Result |
|---|---:|---:|---|---|
| E007 | 35 | 3.41 s | 0 hits, 2 misses | not block-aligned; no registration |
| E008 | 49 | 13.93 s | 0 hits, 2 misses | not block-aligned; no registration |
| E009 | 64 | 18.33 s | +1 hit, +1 miss, +64 hit tokens, +4 hit blocks | completion 0 prefills; completion 1 exact-hits |
| E010 | 64 warm repeat | 0.42 s | +2 hits, +128 hit tokens, +8 hit blocks, +0.000 s prefill sum | both completions exact-hit |

Artifacts:

- `e007_server_batch_n2_response.json`, `e007_server_batch_n2_metrics.prom`
- `e008_server_batch_n2_aligned_response.json`, `e008_server_batch_n2_aligned_metrics.prom`
- `e009_server_batch_n2_aligned64_response.json`, `e009_server_batch_n2_aligned64_metrics.prom`
- `e010_server_batch_n2_aligned64_warm_response.json`, `e010_server_batch_n2_aligned64_warm_metrics.prom`

Takeaway:

E001 converts repeated aligned prompts from "prefill every completion" to
"prefill once, then exact-hit", and on an already warm prompt to "decode only".
On this M1, the warm repeated-prompt `n=2` batch fell from an 18.33 s aligned
cold request to 0.42 s when both completions used exact prefix hits.

## Experiment E011-E014: Exact Reuse For Non-Aligned Prompts

Hypothesis:

The E007/E008 misses were not fundamental. Exact-prompt reuse can safely retain
the partial prompt block when the cache entry carries saved first-token logits
or a saved greedy token. Strict-prefix reuse for longer prompts must remain
block-aligned, but identical prompts do not need that restriction.

Change:

- Exact prefix-cache registration now retains `ceil(prompt_len / block_size)`
  blocks when `next_token` is present.
- Exact lookup accepts non-aligned entries only for identical prompt lengths.
- Strict longer-prompt lookup still requires a block-aligned cached prefix.

Verification:

- `cargo test -p kiln-model exact_prefix_cache_hit_skips_prefill_and_matches_tokens --lib`
  - Result: passed with a 3-token, non-block-aligned prompt fixture.
- `cargo test -p kiln-server real_prefix_cache --lib`
  - Result: passed; partial-block exact entries hit only exact prompts.
- `cargo test -p kiln-server batch_ --lib`
  - Result: passed.

Real server results:

| Experiment | Prompt tokens per completion | Wall time | Prefix-cache delta | Prefill sum delta | Decode sum delta |
|---|---:|---:|---|---:|---:|
| E011 | 35 cold, `n=2` | 9.02 s | +1 hit, +1 miss, +35 hit tokens, +3 hit blocks | +4.610582 s | +4.296233 s |
| E012 | 35 warm, `n=2` | 2.90 s | +2 hits, +70 hit tokens, +6 hit blocks | +0.000000 s | +2.890905 s |
| E013 | 49 cold, `n=2` | 2.96 s | +1 hit, +1 miss, +49 hit tokens, +4 hit blocks | +2.581467 s | +0.365467 s |
| E014 | 49 warm, `n=2` | 1.38 s | +2 hits, +98 hit tokens, +8 hit blocks | +0.000000 s | +1.370864 s |

Artifacts:

- `e011_server_batch_n2_non_aligned35_*`
- `e012_server_batch_n2_non_aligned35_warm_*`
- `e013_server_batch_n2_non_aligned49_*`
- `e014_server_batch_n2_non_aligned49_warm_*`

Takeaway:

Repeated exact prompts no longer need prompt length to land on a KV block
boundary. The 35-token and 49-token GRPO-shaped batches both now turn the
second completion into an exact-cache hit, and warm repeats add zero prefill
time.

## Experiment E015-E017: Clone Deterministic Greedy Batch Completions

Hypothesis:

For `/v1/completions/batch` with one prompt, `n > 1`, and `temperature=0.0`,
every completion is deterministic and identical. Prefix-cache reuse removes
duplicate prefill, but still decodes each duplicate completion. The fastest
path is to generate one completion and clone it into the batch response.

Change:

- Added a batch gate for explicit greedy multi-completion requests:
  `n.unwrap_or(1) > 1 && temperature.unwrap_or(1.0) == 0.0`.
- The prompt-local task now performs one physical `generate_one_response` when
  the gate is true, then clones that response for completion indices 1..n-1.
- Response usage still reports logical prompt/completion tokens per returned
  completion. `kiln_tokens_generated_total` and recent-request records count
  physical model work only.

Verification:

- `cargo test -p kiln-server batch_ --lib`
  - Result: 13 passed.
  - New coverage: the clone gate only accepts explicit greedy `n > 1`, and a
    mock `n=3` request records one physical generation while returning three
    logical completions.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed after E017.

Real server results:

| Experiment | Shape | Wall time | Physical metrics delta | Logical response |
|---|---|---:|---|---|
| E015 | 35-token prompt, cold `n=2`, temp 0, max 2 | 5.40 s | +1 miss, +3.280176 s prefill, +2.105080 s decode, +2 physical tokens | 2 completions, 4 completion tokens |
| E016 | same prompt, warm `n=2` | 2.42 s | +1 hit, +35 hit tokens, +0.000000 s prefill, +2.414926 s decode, +2 physical tokens | 2 completions, 4 completion tokens |
| E017 | same prompt, warm `n=8` | 2.53 s | +1 hit, +35 hit tokens, +0.000000 s prefill, +2.523192 s decode, +2 physical tokens | 8 completions, 16 completion tokens |

Artifacts:

- `e015_server_batch_n2_greedy_clone_cold_*`
- `e016_server_batch_n2_greedy_clone_warm_*`
- `e017_server_batch_n8_greedy_clone_warm_*`

Takeaway:

For greedy repeated-prompt batches, physical decode work is now independent of
`n`. The warm `n=8` request returned eight logical completions while the model
generated only two tokens total.

## Experiment E018-E020: Full Completion Cache For Repeated Greedy Requests

Hypothesis:

E015-E017 still decode once per prompt group, even after duplicate completions
are cloned. For repeated HTTP requests with identical greedy prompts and the
same stopping parameters, that last decode can also be removed: the server can
return the full cached completion while issuing a fresh response id and recent
request record.

Change:

- Added a bounded in-memory deterministic completion cache in `AppState`.
- Cache key: active adapter, prompt token ids, `max_tokens`, stop strings,
  `top_p`, and `top_k`.
- Cache is only enabled for non-streaming `temperature=0.0`. Seed is omitted
  intentionally because greedy argmax does not consult the RNG.
- `/v1/chat/completions` and `/v1/completions/batch` both check the cache
  before model work and store successful physical greedy generations.

Verification:

- `cargo test -p kiln-server completion_cache --lib`
  - Result: 4 passed after E023.
  - Coverage: cache key rejects sampled requests, ignores seed for greedy,
    repeated greedy chat requests do not increase physical generated-token
    metrics, repeated greedy batch requests do not increase physical
    generated-token metrics, and in-flight duplicate requests coalesce on the
    owner generation.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed after E020.

Real server results:

| Experiment | Shape | Wall time | Metrics delta | Logical response |
|---|---|---:|---|---|
| E018 | chat, 35-token prompt, cold temp 0, max 2 | 2.73 s | +1 miss, +2.514891 s prefill, +0.199065 s decode, +2 physical tokens | 1 completion, 2 completion tokens |
| E019 | same chat request warm | 0.01 s | no prefill/decode/token/prefix-cache counter changes | 1 completion, 2 completion tokens |
| E020 | same prompt as warm batch `n=8` | 0.01 s | no prefill/decode/token/prefix-cache counter changes | 8 completions, 16 completion tokens |

Artifacts:

- `e018_chat_greedy_completion_cache_cold_*`
- `e019_chat_greedy_completion_cache_warm_*`
- `e020_server_batch_n8_completion_cache_warm_*`

Takeaway:

For repeated greedy prompts, both bs=1 chat and bs>1 batch can now return from
the API layer without touching prefix cache, KV cache, or model decode. This is
the strongest "remove the need to do it" result so far: E020 returns the same
logical `n=8` batch shape as E017 in about 0.01 s instead of 2.53 s, with zero
new physical tokens.

## Experiment E021-E022: Native MTP On Metal For bs=1 Decode

Hypothesis:

Qwen3.5-4B ships a native MTP head. If the draft acceptance rate is high enough,
`KILN_SPEC_METHOD=mtp` plus `KILN_ENABLE_METAL_NATIVE_MTP=1` could reduce
single-user greedy decode latency for short prompts.

Commands:

- E021 reference:
  `./target/release/kiln-bench --model-path <snapshot> --paged --latency-only --prompt-tokens 64 --max-output-tokens 16 --temperature 0.0 --seed 3`
- E022 MTP:
  `KILN_SPEC_METHOD=mtp KILN_ENABLE_METAL_NATIVE_MTP=1 ./target/release/kiln-bench --model-path <snapshot> --paged --latency-only --prompt-tokens 64 --max-output-tokens 16 --temperature 0.0 --seed 3`

Results:

| Experiment | Spec method | Actual prompt tokens | Prefill | Decode | Mean ITL | P99 ITL | Acceptance |
|---|---|---:|---:|---:|---:|---:|---:|
| E021 | off | 64 | 8432.0 ms / 7.59 tok/s | 5.84 tok/s | 171.1 ms | 235.8 ms | n/a |
| E022 | mtp | 52 | 5818.4 ms / 8.94 tok/s | 1.84 tok/s | 542.4 ms | 3501.6 ms | 0.250 |

Artifacts:

- `e021_m1_bs1_p64_o16_seed3_off.*`
- `e022_m1_bs1_p64_o16_seed3_mtp.*`

Takeaway:

Native MTP is slower on this M1 short greedy shape. It also pays a lazy MTP GPU
upload during first use (`upload_elapsed_ms=1269`) and only accepted 3 of 12
draft attempts. Keep `KILN_ENABLE_METAL_NATIVE_MTP` disabled by default.

## Experiment E023: Singleflight For Concurrent Identical Greedy Requests

Hypothesis:

E018-E020 only help after a completion has finished and entered the cache. If
two identical cold greedy requests arrive at the same time, both can still miss
the completed cache and run duplicate model work. An in-flight singleflight
entry should make the second request wait on the first physical generation and
then return the same cached completion with its own fresh response id.

Change:

- Extended the deterministic completion cache with an in-flight map keyed by
  the same deterministic request key.
- First miss claims ownership and performs generation.
- Concurrent identical misses subscribe to a watch channel and return from the
  owner result.
- Owner success inserts the completed cache entry and publishes it to waiters;
  owner failure wakes waiters without a cached value so they can fall back.

Verification:

- `cargo test -p kiln-server completion_cache --lib`
  - Result: 4 passed.
  - New coverage: cache-level in-flight claim/wait/complete behavior.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed after E023.

Real server result:

Two identical cold `/v1/chat/completions` requests were launched in parallel
against a fresh server with the 35-token greedy prompt.

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `kiln_requests_total{status="ok"}` | 0 | 2 | +2 |
| `kiln_request_prefill_duration_seconds_count` | 0 | 1 | +1 |
| `kiln_request_decode_duration_seconds_count` | 0 | 1 | +1 |
| `kiln_tokens_generated_total` | 0 | 2 | +2 |
| `kiln_prefix_cache_lookups_total{result="miss"}` | 0 | 1 | +1 |

Both clients completed in 3.55 s and received independent response ids with
the same usage (`35` prompt tokens, `2` completion tokens).

Artifacts:

- `e023_before_metrics.prom`
- `e023_after_metrics.prom`
- `e023_chat_singleflight_a_*`
- `e023_chat_singleflight_b_*`

Takeaway:

Cold concurrent duplicates now cost one physical generation instead of N. This
improves bs>1 throughput for identical greedy traffic bursts without changing
the response contract.

## Experiment E024: Serve Cached Greedy Completions As SSE Streams

Hypothesis:

The full deterministic completion cache should also help OpenAI clients that
ask for `stream: true`. If the same greedy prompt has already completed once,
the server can synthesize the SSE chunk sequence from the cached
`reasoning_content` / `content` pair instead of touching the model.

Change:

- Added a non-owning cache probe for streaming chat requests.
- Streaming cache hits return a fresh `chat.completion.chunk` SSE sequence from
  the cached deterministic value.
- Streaming requests that arrive while a non-streaming owner is in flight wait
  for that owner; streaming misses continue through the existing live streaming
  path so a client disconnect cannot leave a stuck in-flight owner.
- Cached streaming responses still record a recent-request entry with
  `streamed=true` and a fresh response id.

Verification:

- `cargo test -p kiln-server completion_cache --lib`
  - Result: 5 passed.
  - New coverage: repeated greedy chat request can populate the deterministic
    cache non-streaming, then the identical `stream: true` request returns an
    SSE-shaped cache hit without increasing generated-token metrics.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 14 passed.
- `cargo test -p kiln-server real_prefix_cache --lib`
  - Result: 4 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421. One non-streaming greedy chat request
using the 35-token test prompt populated the cache; the same request with
`stream: true` then returned from the cache.

| Step | Wall time | OK requests | Prefill count | Decode count | Physical tokens | Prefix misses |
|---|---:|---:|---:|---:|---:|---:|
| Before | n/a | 0 | 0 | 0 | 0 | 0 |
| Non-streaming populate | 1.30 s | 1 | 1 | 1 | 2 | 1 |
| Streaming cache hit | 0.02 s | 2 | 1 | 1 | 2 | 1 |

The warm SSE body contained an assistant role chunk, a `reasoning_content`
chunk (`Thinking Process`), a final `finish_reason="length"` chunk, and
`data: [DONE]`. The streaming request added no prefill, decode, token, or
prefix-cache lookup work after the populate step.

Artifacts:

- `e024_before_metrics.prom`
- `e024_after_populate_metrics.prom`
- `e024_after_stream_metrics.prom`
- `e024_chat_stream_cache_populate_response.json`
- `e024_chat_stream_cache_populate_time.log`
- `e024_chat_stream_completion_cache_warm_sse.txt`
- `e024_chat_stream_completion_cache_warm_time.log`

Takeaway:

Repeated greedy streaming clients now get the same zero-model-work fast path as
non-streaming clients once a deterministic completion exists. This preserves
the streaming wire contract while removing both prefill and decode from the
warm path.

## Experiment E025-E026: Exact Prefix-Cache Hits For Live Streaming

Hypothesis:

The live streaming path still rejected exact prompt prefix-cache hits even when
the cache entry carried saved logits / a saved greedy token. Non-streaming
generation already uses that saved first-token source to skip prefill. Streaming
should do the same, especially for sampled repeated streams where the full
deterministic completion cache is intentionally bypassed.

Change:

- Updated `spawn_streaming_paged_shared_tokens_with_prefix_cache` to accept
  exact prompt hits when the saved first-token source can serve the request.
- Exact streaming hits skip prefill, use the cached logits/greedy token to emit
  the first streamed token, allocate only decode-suffix KV blocks, and avoid
  re-registering the same prompt.
- Added a model-level test that builds an exact prefix registration, streams
  from it, verifies the streamed token IDs match the original completion, and
  asserts no prompt re-registration happens on the exact hit.

Verification:

- `cargo test -p kiln-model exact_prefix_cache_hit --lib`
  - Result: 2 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421. The request used the 35-token prompt
with `stream: true`, `temperature=0.7`, `max_tokens=2`, so the deterministic
completion cache could not serve it. The first request registered the prompt;
the second request reused the exact prefix with saved logits and skipped
prefill.

| Experiment | Shape | Wall time | Prefix cache | Cached blocks |
|---|---|---:|---|---:|
| E025 | sampled streaming, cold | 3.06 s | +1 miss | 3 |
| E026 | same sampled streaming request, warm exact hit | 1.76 s | +1 hit, no new miss | 3 |

The warm request preserved the SSE contract (`role`, reasoning-content chunks,
final chunk, `[DONE]`) while avoiding prompt prefill. Streaming token metrics
are not currently incremented by the live SSE path, so prefix hit/miss counters
and wall time are the reliable measurement here.

Artifacts:

- `e025_before_metrics.prom`
- `e025_after_cold_metrics.prom`
- `e025_chat_sampled_stream_cold_sse.txt`
- `e025_chat_sampled_stream_cold_time.log`
- `e026_after_warm_metrics.prom`
- `e026_chat_sampled_stream_exact_prefix_warm_sse.txt`
- `e026_chat_sampled_stream_exact_prefix_warm_time.log`

Takeaway:

Repeated sampled streaming requests now avoid the full prompt prefill after the
first request. For this short 35-token prompt and 2-token output, wall time fell
from 3.06 s to 1.76 s; longer prompts should benefit more because the removed
work is proportional to prompt length.

## Experiment E027-E028: Greedy First-Token Streaming Prefill Attempt

Hypothesis:

The non-streaming paged path has a greedy first-token source that can avoid
materializing full logits on Metal. Applying that same `GreedyToken` prefill
source to live streaming might reduce cold greedy streaming latency and make
exact streaming prefix hits cheaper.

Result:

This was negative and was reverted.

| Experiment | Shape | Wall time | Prefix cache |
|---|---|---:|---|
| E027 | greedy streaming, cold, transient GreedyToken prefill build | 4.93 s | +1 miss |
| E028 | same greedy streaming request, warm exact hit | 2.48 s | +1 hit |

Those times were not competitive with the sampled exact-prefix streaming run
above, and they match the earlier E002-E006 finding that the Metal argmax-only
LM-head path is slower on this machine. The source was reverted to the logits
prefill path for live streaming, and the release binary was rebuilt after the
revert.

Artifacts:

- `e027_before_metrics.prom`
- `e027_after_cold_metrics.prom`
- `e027_chat_greedy_stream_cold_sse.txt`
- `e027_chat_greedy_stream_cold_time.log`
- `e028_after_warm_metrics.prom`
- `e028_chat_greedy_stream_exact_prefix_warm_sse.txt`
- `e028_chat_greedy_stream_exact_prefix_warm_time.log`

Takeaway:

Do not port the greedy first-token Metal prefill shortcut to live streaming
right now. The kept optimization is exact streaming prefix reuse; the
GreedyToken streaming prefill variant lost on real hardware.

## Experiment E029-E030: Group Duplicate Sampled Batch Prompts

Hypothesis:

The batch endpoint handled `n > 1` for a single prompt index sequentially, but
still launched separate prompt indices concurrently. For sampled batches with
the same prompt repeated as separate `prompts[]` entries, that means the
duplicates can all miss the exact prefix cache before any one request registers
the prompt. Grouping duplicate rendered prompts should let the first duplicate
pay prefill once and later sampled duplicates reuse saved logits while distinct
prompt groups remain concurrent.

Change:

- Added batch prompt grouping by the fields currently rendered by synthesized
  batch chat requests: `(role, content)`.
- The batch endpoint now spawns one task per distinct prompt group. Inside a
  group, duplicate prompt indices run sequentially so exact prefix registration
  is visible to later duplicates.
- Public response order is restored after the grouped tasks finish, so clients
  still see completions ordered by original `prompt_index`.

Verification:

- `cargo test -p kiln-server batch_ --lib`
  - Result: 16 passed.
  - New coverage: duplicate plain prompts coalesce into one group, and grouped
    execution preserves response order.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421. E029 used three identical prompts as
separate batch entries with `temperature=0.7`, `n=1`, `max_tokens=2`, so the
deterministic completion cache could not clone or serve the responses.

| Experiment | Shape | Wall time | Prefix cache delta | Physical tokens |
|---|---|---:|---|---:|
| E029 | 3 identical sampled batch prompts | 3.28 s | +1 miss, +2 hits | +6 |
| E030 | 3 distinct sampled batch prompts control | 8.39 s | +2 misses, +1 hit | +6 |

E029 preserved response order (`prompt_index` sequence `[0, 1, 2]`) while
turning the two duplicate prompts after the first into exact prefix-cache hits.
The Prometheus prefill histogram count still rose by 3 because exact hits record
zero-duration prefill observations; the useful signal is the hit/miss counters
and the small prefill sum (`0.007177 s`) for E029.

Artifacts:

- `e029_before_metrics.prom`
- `e029_after_metrics.prom`
- `e029_batch_sampled_repeated_prompts3_grouped_response.json`
- `e029_batch_sampled_repeated_prompts3_grouped_time.log`
- `e030_before_metrics.prom`
- `e030_after_metrics.prom`
- `e030_batch_sampled_distinct_prompts3_response.json`
- `e030_batch_sampled_distinct_prompts3_time.log`

Takeaway:

Sampled duplicate prompt batches now avoid duplicate prompt prefill without
changing the public batch response order. This closes the bs>1 case where
deterministic completion caching cannot apply because each output needs its own
sampling path.

## Experiment E031-E032: Populate Completion Cache From Live Streaming

Hypothesis:

E024 showed that a completed non-streaming greedy request can serve a later
`stream: true` request from the deterministic completion cache. Streaming-only
clients still missed that full-cache path because a successful live stream only
registered prefix-cache state. If a greedy stream reaches `Done`, the server has
all reasoning/content chunks and token count needed to insert the same
deterministic completion value for future requests.

Change:

- The live SSE forwarding path now accumulates full `reasoning_content` and
  `content` separately while still maintaining the dashboard preview buffer.
- On a successful `Done`, greedy streaming requests insert a
  `DeterministicCompletionCacheValue` keyed by the same deterministic cache key
  used by non-streaming completions.
- Streaming misses remain non-owning: a client disconnect or timeout does not
  create an in-flight completion-cache owner or leave waiters stuck.

Verification:

- `cargo test -p kiln-server completion_cache --lib`
  - Result: 5 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421. The same 35-token greedy streaming
request was sent twice with `max_tokens=2`.

| Experiment | Shape | Wall time | Prefix cache delta | Completion cache effect |
|---|---|---:|---|---|
| E031 | first greedy stream, cold | 3.03 s | +1 miss | stream completion inserted full cache value |
| E032 | identical greedy stream, warm | 0.01 s | no new lookup | served from full completion cache as SSE |

The warm SSE response preserved the expected chunk shape: assistant role,
`reasoning_content`, final `finish_reason="length"` chunk, and `[DONE]`.
Streaming generated-token counters remain unchanged because the live SSE path
does not currently increment `kiln_tokens_generated_total`; the important
signal here is that the second request added no prefix lookup and returned in
0.01 s.

Artifacts:

- `e031_before_metrics.prom`
- `e031_after_populate_metrics.prom`
- `e031_chat_greedy_stream_populate_sse.txt`
- `e031_chat_greedy_stream_populate_time.log`
- `e032_after_warm_metrics.prom`
- `e032_chat_greedy_stream_completion_cache_warm_sse.txt`
- `e032_chat_greedy_stream_completion_cache_warm_time.log`

Takeaway:

Streaming-only greedy clients now get the same full completion-cache fast path
as non-streaming clients after their first successful stream. This removes both
prefill and decode from repeated streaming-only traffic.

## Experiment E033-E035: Zero-Token Requests Return At API Speed

Hypothesis:

`max_tokens=0` is accepted by the request structs. Such a request needs prompt
formatting/tokenization for usage accounting, but it does not need adapter
loading, prefix-cache lookup, KV allocation, prefill, decode, or scheduler work.
The fastest path is to return an empty completion immediately after request
validation and tokenization.

Change:

- Added an API-level zero-token fast path for chat completions.
- Added the same path for `stream: true`, returning the expected SSE role
  chunk, final `finish_reason="length"` chunk, and `[DONE]`.
- Added a batch fast path that tokenizes each prompt once, returns all logical
  completions with empty text, and reports correct prompt-token usage without
  model work.

Verification:

- `cargo test -p kiln-server zero_max_tokens --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 17 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 5 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421. Each request used the 35-token test
prompt with `max_tokens=0`.

| Experiment | Shape | Wall time | Prefill | Decode | Tokens | Prefix lookup |
|---|---|---:|---:|---:|---:|---:|
| E033 | chat, non-streaming | 0.01 s | 0 | 0 | 0 | 0 |
| E034 | chat, streaming | 0.02 s | 0 | 0 | 0 | 0 |
| E035 | batch `n=8` | 0.01 s | 0 | 0 | 0 | 0 |

E033 returned `completion_tokens=0`, empty content, and
`finish_reason="length"`. E034 returned a valid SSE response with role, final
length chunk, and `[DONE]`. E035 returned 8 logical completions with
`completion_tokens=0` and total `prompt_tokens=280` (`35 * 8`).

Artifacts:

- `e033_before_metrics.prom`
- `e033_after_chat_metrics.prom`
- `e033_chat_zero_max_response.json`
- `e033_chat_zero_max_time.log`
- `e034_after_stream_metrics.prom`
- `e034_chat_zero_max_stream_sse.txt`
- `e034_chat_zero_max_stream_time.log`
- `e035_after_batch_metrics.prom`
- `e035_batch_n8_zero_max_response.json`
- `e035_batch_n8_zero_max_time.log`

Takeaway:

Requests asking for no generated tokens now avoid all model-side work. This is
the purest "remove the need to do it" path: only prompt rendering/tokenization
and response shaping remain.

## Experiment E036-E037: Count Physical Tokens For Live Streaming

Hypothesis:

E031-E032 proved that a completed greedy live stream can populate the full
deterministic completion cache, but the live SSE path did not increment
`kiln_tokens_generated_total`. That made repeated-stream validation harder:
both a physical live stream and a synthetic completion-cache stream appeared to
add zero generated tokens. The server should count physical streaming decode
tokens as they are produced, while synthetic cached SSE responses should remain
zero-work from the model's perspective.

Change:

- The live SSE forwarding path now calls `metrics.add_tokens(1)` for each
  `StreamEvent::Token` received from the model.
- The synthetic completion-cache SSE path is unchanged, so cache hits still do
  not increment physical generated-token metrics.
- This intentionally leaves streaming prefill/decode duration histograms
  unchanged for now; the threaded streaming model API does not return full
  decode timing without a wider event/API change.

Verification:

- `cargo test -p kiln-server completion_cache --lib`
  - Result: 5 passed.
- `cargo test -p kiln-server zero_max_tokens --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 17 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421. The same 35-token greedy streaming
request was sent twice with `max_tokens=2`.

| Experiment | Shape | Wall time | `kiln_tokens_generated_total` | Prefix lookups |
|---|---|---:|---:|---|
| E036 | first greedy stream, cold | 2.94 s | 0 -> 2 | miss 0 -> 1 |
| E037 | identical greedy stream, warm | 0.01 s | stayed 2 | unchanged |

E036 streamed two physical reasoning tokens and incremented the generated-token
counter to 2. E037 returned the same logical completion from the full
completion cache as SSE in 0.01 s; it did not add generated tokens or perform a
new prefix-cache lookup.

Artifacts:

- `e036_before_metrics.prom`
- `e036_after_stream_metrics.prom`
- `e036_chat_greedy_stream_token_metrics_sse.txt`
- `e036_chat_greedy_stream_token_metrics_time.log`
- `e037_after_warm_metrics.prom`
- `e037_chat_greedy_stream_completion_cache_token_metrics_sse.txt`
- `e037_chat_greedy_stream_completion_cache_token_metrics_time.log`

Takeaway:

Streaming cache-hit experiments now have a reliable physical-token signal. A
warm deterministic completion-cache stream is visibly zero-decode in both
latency and generated-token counters, while the initial live stream records the
tokens it actually decoded.

## Experiment E038-E040: Full Completion Cache For Seeded Sampling

Hypothesis:

The deterministic completion cache was intentionally greedy-only because
unseeded sampled decoding is random. Seeded sampling is different: the model
sampler already treats a fixed seed as replayable and advances that seed
deterministically per token. Repeated requests with the same prompt, sampling
parameters, stop set, and seed should therefore be eligible for the same
"remove all model work" completion-cache path as greedy decoding. Greedy keys
can also normalize away seed/top-p/top-k because argmax ignores those fields.

Change:

- Completion-cache keys now represent replayable decoding instead of only
  greedy decoding.
- Greedy cache keys normalize `seed`, `top_p`, and `top_k` so irrelevant
  client parameters do not split otherwise-identical argmax requests.
- Seeded sampled cache keys include `temperature`, `top_p`, `top_k`, and
  `seed`, so different sampled token paths do not collide.
- Unseeded sampled requests remain ineligible for the full completion cache.
- The existing chat, batch, and streaming cache-hit paths all share the wider
  key eligibility.

Verification:

- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
  - New coverage: seeded sampled chat and batch cache hits, unseeded sampled
    chat does not cache, and cache keys split sampled outputs by seed and
    temperature.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 18 passed.
- `cargo test -p kiln-server zero_max_tokens --lib`
  - Result: 3 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421 after the release build.

| Experiment | Shape | Cold/first wall time | Warm wall time | Physical work on warm repeat |
|---|---|---:|---:|---|
| E038 | seeded sampled chat, `temperature=0.7`, `max_tokens=2` | 2.92 s | 0.01 s | none |
| E039 | seeded sampled batch, 3 repeated prompts, `max_tokens=2` | 3.26 s | 0.01 s | none |
| E040 | seeded sampled streaming chat, `max_tokens=2` | 2.70 s | 0.01 s | none |

Metric deltas:

- E038 moved `kiln_tokens_generated_total` from 0 to 2 and prefix misses from
  0 to 1 on the cold request. The warm repeat left generated tokens, prefill
  count/sum, decode count/sum, and prefix-cache lookups unchanged.
- E039 started after E038 with 2 generated tokens. The cold batch added 6
  generated tokens, 1 prefix miss, and 2 exact prefix hits across the three
  repeated prompts. The warm batch left all model-work counters unchanged.
- E040 used a fresh seed for the same prompt as E038. The first stream reused
  the exact prompt prefix already in cache, added 2 physical generated tokens,
  and populated the full completion cache. The warm stream left generated
  tokens and prefix-cache lookups unchanged. Streaming prefill/decode histograms
  remain unchanged for the previously documented streaming-timing limitation.

Artifacts:

- `e038_before_metrics.prom`
- `e038_after_cold_metrics.prom`
- `e038_after_warm_metrics.prom`
- `e038_chat_seeded_sampled_completion_cache_cold_response.json`
- `e038_chat_seeded_sampled_completion_cache_cold_time.log`
- `e038_chat_seeded_sampled_completion_cache_warm_response.json`
- `e038_chat_seeded_sampled_completion_cache_warm_time.log`
- `e039_before_metrics.prom`
- `e039_after_cold_metrics.prom`
- `e039_after_warm_metrics.prom`
- `e039_batch_seeded_sampled_repeated_prompts3_cold_response.json`
- `e039_batch_seeded_sampled_repeated_prompts3_cold_time.log`
- `e039_batch_seeded_sampled_repeated_prompts3_warm_response.json`
- `e039_batch_seeded_sampled_repeated_prompts3_warm_time.log`
- `e040_before_metrics.prom`
- `e040_after_populate_metrics.prom`
- `e040_after_warm_metrics.prom`
- `e040_chat_seeded_sampled_stream_populate_sse.txt`
- `e040_chat_seeded_sampled_stream_populate_time.log`
- `e040_chat_seeded_sampled_stream_completion_cache_warm_sse.txt`
- `e040_chat_seeded_sampled_stream_completion_cache_warm_time.log`

Takeaway:

The full-response no-work path now covers replayable sampled traffic, not just
greedy traffic. This removes all model-side work from repeated seeded sampled
requests across bs=1 chat, bs>1 batch, and streaming responses while preserving
randomness for unseeded sampled requests.

## Experiment E041: Cache Rendered Prompt Tokens

Hypothesis:

Completion-cache hits, zero-token requests, and prefix-cache lookups still need
prompt tokens before they can decide whether model work is avoidable. Repeated
requests with the same rendered prompt should not pay tokenizer encode cost
again. A small bounded cache keyed by the rendered prompt string can remove that
front-end work without changing response semantics.

Change:

- Added a 256-entry LRU `PromptTokenCache` to `AppState`.
- Chat completions, batch zero-token responses, batch fan-out completions, and
  mock generation now call a shared `encode_prompt_tokens` helper.
- Added `/metrics` counters for
  `kiln_prompt_token_cache_lookups_total{result="hit|miss"}` and a
  `kiln_prompt_token_cache_entries` gauge.

Verification:

- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 3 passed.
  - Coverage: direct cache hit/miss/eviction behavior, repeated chat prompt
    endpoint hit, and duplicate batch prompt endpoint hit.
- `cargo test -p kiln-server test_metrics_render --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server zero_max_tokens --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 19 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421 after the release build. The request was
a 4107-token prompt with `max_tokens=0`, so any latency delta is front-end
request handling rather than model prefill or decode.

| Experiment | Shape | Wall time | Prompt token cache | Model work |
|---|---|---:|---|---|
| E041 cold | 4107-token chat, `max_tokens=0` | 0.02 s | miss 0 -> 1, entries 0 -> 1 | none |
| E041 warm | identical repeat | 0.01 s | hit 0 -> 1, misses stayed 1 | none |

Both responses reported `prompt_tokens=4107`, `completion_tokens=0`, and
`finish_reason="length"`. `kiln_tokens_generated_total`, prefill histogram
count, and decode histogram count all stayed at 0. The server-side request log
also showed handler duration dropping from about 8.85 ms to about 0.57 ms.

Artifacts:

- `e041_before_metrics.prom`
- `e041_after_cold_metrics.prom`
- `e041_after_warm_metrics.prom`
- `e041_chat_long_zero_prompt_cache_cold_response.json`
- `e041_chat_long_zero_prompt_cache_cold_time.log`
- `e041_chat_long_zero_prompt_cache_warm_response.json`
- `e041_chat_long_zero_prompt_cache_warm_time.log`

Takeaway:

Repeated prompt front-end work now has its own no-work path. This matters most
for long prompts and for requests that already avoid model work through
`max_tokens=0` or full completion-cache hits.

## Experiment E042: Cache Rendered Chat Templates

Hypothesis:

E041 removed repeated tokenization, but the server still had to render the chat
template before it could find the rendered-prompt token cache entry. Repeated
requests with identical messages, tools, and tool choice should reuse the final
rendered prompt string directly, then reuse the token cache. That removes the
remaining chat-template front-end work on repeated prompts.

Change:

- Added a 256-entry LRU `RenderedPromptCache` to `AppState`.
- The cache key is the serialized tuple of `messages`, `tools`, and
  `tool_choice`, so tool-bearing prompts do not collide with plain chat.
- Chat completions, batch zero-token responses, and batch fan-out completions
  now render through a shared `render_prompt_text` helper.
- Added `/metrics` counters for
  `kiln_rendered_prompt_cache_lookups_total{result="hit|miss"}` and a
  `kiln_rendered_prompt_cache_entries` gauge.

Verification:

- `cargo test -p kiln-server rendered_prompt_cache --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 3 passed.
  - Coverage now checks both rendered-prompt and token-cache hits for repeated
    chat and duplicate batch prompts.
- `cargo test -p kiln-server test_metrics_render --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server zero_max_tokens --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 19 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421 after the release build. The same
4107-token `max_tokens=0` prompt from E041 was sent twice.

| Experiment | Shape | Wall time | Render cache | Token cache | Model work |
|---|---|---:|---|---|---|
| E042 cold | 4107-token chat, `max_tokens=0` | 0.02 s | miss 0 -> 1 | miss 0 -> 1 | none |
| E042 warm | identical repeat | 0.01 s | hit 0 -> 1 | hit 0 -> 1 | none |

Both responses reported `prompt_tokens=4107`, `completion_tokens=0`, and
`finish_reason="length"`. `kiln_tokens_generated_total`, prefill histogram
count, and decode histogram count all stayed at 0. The server-side request log
showed handler duration dropping from about 9.18 ms to about 0.49 ms.

Artifacts:

- `e042_before_metrics.prom`
- `e042_after_cold_metrics.prom`
- `e042_after_warm_metrics.prom`
- `e042_chat_long_zero_render_cache_cold_response.json`
- `e042_chat_long_zero_render_cache_cold_time.log`
- `e042_chat_long_zero_render_cache_warm_response.json`
- `e042_chat_long_zero_render_cache_warm_time.log`

Takeaway:

Repeated prompt front-end work now skips both Jinja chat-template rendering and
tokenization. This improves the no-model-work surfaces (`max_tokens=0`, full
completion-cache hits) and reduces overhead before any prefix-cache lookup on
repeated long prompts.

## Experiment E043-E051: Seeded Full-Vocab Device CDF Rejected

Hypothesis:

Seeded sampled requests with `temperature > 0`, `top_p = 1`, and `top_k = 0`
still use the exact host-side full-vocab categorical sampler. Moving the
softmax/CDF work onto Metal while transferring only scalar max/sum/token values
might reduce the roughly 2.9 s decode time observed in E038 for a 2-token
seeded sampled chat request.

Change tested, then reverted:

- Added a seeded full-vocab Metal/CUDA CDF path in `kiln-model::sampling`.
- The path preserved the same one-random-threshold categorical algorithm as the
  host sampler and fell back to the host path on device-op failure.
- Added a focused CPU parity test for the helper.
- After the first measurement showed lazy-kernel compile cost, added a small
  server prewarm hook for the seeded full-vocab sampling kernels.
- Reverted both the device-CDF sampler and the server prewarm hook because the
  real Qwen timings were not reliably faster.

Verification while testing:

- `cargo test -p kiln-model seeded_device_cdf --lib`
  - Result: 1 passed.
- `cargo test -p kiln-model test_default_path_seed_is_deterministic_on_metal --features metal --lib`
  - Result: 1 passed.
- `cargo test -p kiln-model sampling::tests --features metal --lib`
  - Result before revert: 17 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result before revert: passed.
- `cargo build --release --features metal --bin kiln`
  - Result before revert: passed.

Real server result:

The first device-CDF attempt used a fresh server with normal inference prewarm
complete, but without sampling-kernel prewarm. It regressed badly: 7.05 s wall
and 7.034 s decode for the same 35-token prompt and 2 sampled tokens that E038
handled in about 2.92 s.

After adding the sampling-kernel prewarm hook, the server log showed the tiny
sampling prewarm taking 91 ms and `/health` reported
`inference_prewarm_complete=true`. One first live request was fast at 0.67 s wall
and 0.644 s decode, but follow-up requests did not reproduce that speed. A
different seed on the same prompt took 2.64 s, a fresh prompt took 2.86 s, and
two back-to-back fresh prompts took 7.20 s and 3.18 s. The results were therefore
not a dependable latency improvement over the host path.

| Experiment | Shape | Wall time | Decode delta | Notes |
|---|---|---:|---:|---|
| E043 | 35-token seeded sampled chat, no sampling prewarm | 7.05 s | 7.034 s | Lazy device CDF setup regressed badly |
| E044 | Same request repeat | 0.01 s | 0.000 s | Full completion-cache hit, no model work |
| E045 | Same prompt, different seed | 2.52 s | 2.512 s | Prefix hit, still near old host-path latency |
| E046 | Fresh server after sampling prewarm | 0.67 s | 0.644 s | Fast outlier after 91 ms prewarm |
| E047 | Same request repeat | 0.01 s | 0.000 s | Full completion-cache hit, no model work |
| E048 | Same prompt, different seed | 2.64 s | 2.634 s | Prefix hit, slow again |
| E049 | Fresh prompt | 2.86 s | 2.834 s | Essentially equal to E038 |
| E050 | Fresh prompt A | 7.20 s | included below | Regressed |
| E051 | Fresh prompt B immediately after E050 | 3.18 s | E050+E051: 10.282 s | Regressed/unstable |

Post-revert verification:

- `cargo test -p kiln-model sampling::tests --features metal --lib`
  - Result: 16 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Artifacts:

- `e043_before_metrics.prom`
- `e043_after_metrics.prom`
- `e043_chat_seeded_sampled_device_cdf_response.json`
- `e043_chat_seeded_sampled_device_cdf_time.log`
- `e044_after_warm_metrics.prom`
- `e044_chat_seeded_sampled_device_cdf_cache_warm_response.json`
- `e044_chat_seeded_sampled_device_cdf_cache_warm_time.log`
- `e045_after_second_seed_metrics.prom`
- `e045_chat_seeded_sampled_device_cdf_second_seed_response.json`
- `e045_chat_seeded_sampled_device_cdf_second_seed_time.log`
- `e046_health_after_prewarm.json`
- `e046_before_metrics.prom`
- `e046_after_metrics.prom`
- `e046_chat_seeded_sampled_prewarmed_device_cdf_response.json`
- `e046_chat_seeded_sampled_prewarmed_device_cdf_time.log`
- `e047_after_warm_metrics.prom`
- `e047_chat_seeded_sampled_prewarmed_cache_warm_response.json`
- `e047_chat_seeded_sampled_prewarmed_cache_warm_time.log`
- `e048_after_second_seed_metrics.prom`
- `e048_chat_seeded_sampled_prewarmed_second_seed_response.json`
- `e048_chat_seeded_sampled_prewarmed_second_seed_time.log`
- `e049_after_new_prompt_metrics.prom`
- `e049_chat_seeded_sampled_prewarmed_new_prompt_response.json`
- `e049_chat_seeded_sampled_prewarmed_new_prompt_time.log`
- `e050_chat_seeded_sampled_prewarmed_prompt_a_response.json`
- `e050_chat_seeded_sampled_prewarmed_prompt_a_time.log`
- `e051_after_back_to_back_metrics.prom`
- `e051_chat_seeded_sampled_prewarmed_prompt_b_response.json`
- `e051_chat_seeded_sampled_prewarmed_prompt_b_time.log`

Takeaway:

Do not move seeded full-vocab sampling to a device CDF path in this form. It can
produce a fast outlier after explicit kernel prewarm, but steady real Qwen
latency is unstable and often slower than the existing exact host sampler. The
host seeded path remains the kept implementation; the useful no-work behavior
for repeated seeded requests still comes from the full completion cache added in
E038-E040.

## Experiment E052-E053: Clone Greedy Duplicate Batch Prompt Rows

Hypothesis:

The batch endpoint already clones `n > 1` greedy completions for one prompt, and
the completion cache prevents extra model work for later identical greedy prompt
rows. However, those duplicate prompt rows still call the synthesized
single-response path, which means extra rendered-prompt cache lookups, token
cache lookups, completion-cache probes, response shaping, and recent-request
records. For `temperature=0.0`, identical prompt rows are deterministic, so the
first row's logical responses can be cloned across every duplicate row in the
prompt group.

Change:

- Added a duplicate-prompt greedy clone gate for batch groups.
- When a batch prompt group contains identical rendered prompt rows and
  `temperature=0.0`, the first prompt row generates the physical response set
  and later rows receive cloned logical responses.
- Kept sampled batches unchanged because unseeded sampling is intentionally
  random and seeded batch completions derive distinct seeds per output.
- Reused the same cloned response accounting behavior as the existing `n > 1`
  greedy clone path: logical usage is counted for every returned completion,
  but model generated-token metrics and recent-request records reflect physical
  work only.
- Removed a duplicate mock-backend tokenization pass uncovered by the new test;
  the real backend already consumed the precomputed prompt tokens.

Verification:

- `cargo test -p kiln-server batch_greedy_duplicate_prompts_clone_one_physical_completion --lib`
  - Result: 1 passed.
  - Coverage: two identical prompt rows with `n=2`, `temperature=0.0` return
    four logical completions, one physical generated-token count, one recent
    request, one render miss, zero render hits, one token miss, and zero token
    hits.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 20 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 3 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421 after the release build and background
prewarm completed. The request used 8 identical prompt rows, `n=8`,
`temperature=0.0`, and `max_tokens=2`, for 64 logical completions.

| Experiment | Shape | Wall time | Logical completions | Physical generated tokens | Render cache | Token cache | Prefix cache | Model work |
|---|---|---:|---:|---:|---|---|---|---|
| E052 cold | 8 duplicate prompts, `n=8`, greedy, `max_tokens=2` | 4.13 s | 64 | 2 | miss 0 -> 1, hit stayed 0 | miss 0 -> 1, hit stayed 0 | miss 0 -> 1 | 1 prefill + 1 decode |
| E053 warm | identical repeat | 0.01 s | 64 | unchanged at 2 | hit 0 -> 1 | hit 0 -> 1 | unchanged | none |

E052 returned `completion_count=64`, `prompt_tokens=2304`,
`completion_tokens=128`, and `unique_texts=1`. The metrics showed exactly one
physical generation: `kiln_tokens_generated_total` rose from 0 to 2,
prefill/decode histogram counts rose from 0 to 1, and rendered/token prompt
cache misses rose from 0 to 1 with zero hits. E053 returned the same logical
usage in 0.01 s, and prefill/decode/generated-token/prefix-cache counters did
not move.

Artifacts:

- `e052_health_after_prewarm.json`
- `e052_before_metrics.prom`
- `e052_after_metrics.prom`
- `e052_batch_greedy_duplicate_prompts8_n8_response.json`
- `e052_batch_greedy_duplicate_prompts8_n8_time.log`
- `e053_after_warm_metrics.prom`
- `e053_batch_greedy_duplicate_prompts8_n8_warm_response.json`
- `e053_batch_greedy_duplicate_prompts8_n8_warm_time.log`

Takeaway:

Large duplicate greedy batches now remove nearly all per-row work, not just
model work. A 64-output duplicate batch is served by one prompt render, one
tokenization, one prefix-cache lookup, and one physical 2-token generation; an
identical repeat is a 0.01 s full completion-cache replay.

## Experiment E054-E055: Warm Multi-Block Paged Prefill During Startup

Hypothesis:

E052 still showed a 3.876 s request-time prefill on the first live 36-token
batch prompt even after the existing background inference prewarm completed.
The current startup prewarm uses only 16 prompt tokens, so it warms a one-block
prompt plus decode but can miss multi-block paged-prefill shapes. A 64-token
startup prewarm should move more first-live Metal/Candle setup out of request
latency.

Change:

- Increased background GPU inference prewarm from a fixed 16-token prompt to a
  64-token prompt.
- Sized the temporary prewarm block manager from
  `(prompt_tokens + max_tokens).div_ceil(16) + 1`, so the synthetic prewarm has
  enough KV blocks for the longer prompt and two generated tokens.
- Kept the prewarm opportunistic: it still uses the existing GPU write-lock
  `try_write`, and skips if a live request or training job already owns the GPU.

Verification:

- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421 after the release build. With the prior
16-token prewarm in E052, background prewarm completed in about 10.27 s and the
first live 8x duplicate prompt / `n=8` request took 4.13 s wall, with 3.876 s
recorded as request-time prefill. With the new 64-token prewarm, background
prewarm completed in about 11.76 s and the same first live request took 2.85 s
wall, with 2.641 s request-time prefill.

| Experiment | Startup prewarm | First live shape | Wall time | Request prefill | Request decode | Physical tokens |
|---|---:|---|---:|---:|---:|---:|
| E052 baseline | 16 tokens, 10.27 s | 8 duplicate prompts, `n=8`, greedy, `max_tokens=2` | 4.13 s | 3.876 s | 0.240 s | 2 |
| E054 | 64 tokens, 11.76 s | same | 2.85 s | 2.641 s | 0.195 s | 2 |
| E055 warm repeat | 64 tokens | same repeated | 0.01 s | unchanged | unchanged | unchanged |

E054 preserved the E052 no-work shape for duplicate rows: 64 logical
completions, one rendered-prompt miss, one prompt-token miss, one prefix-cache
miss, one physical prefill/decode, and two physical generated tokens. E055 again
returned the same 64 logical completions in 0.01 s without moving prefill,
decode, generated-token, or prefix-cache counters.

Artifacts:

- `e054_health_after_64tok_prewarm.json`
- `e054_before_metrics.prom`
- `e054_after_metrics.prom`
- `e054_batch_greedy_duplicate_prompts8_n8_64tok_prewarm_response.json`
- `e054_batch_greedy_duplicate_prompts8_n8_64tok_prewarm_time.log`
- `e055_after_warm_metrics.prom`
- `e055_batch_greedy_duplicate_prompts8_n8_64tok_prewarm_warm_response.json`
- `e055_batch_greedy_duplicate_prompts8_n8_64tok_prewarm_warm_time.log`

Takeaway:

The larger startup prewarm is a net win for the server once `/health` reports
`inference_prewarm_complete=true`: it adds about 1.5 s to background readiness
work but removes about 1.2-1.3 s from this first live multi-block request. Keep
the 64-token prewarm because desktop chat and batch prompts commonly exceed one
KV block, and users who wait for health readiness get lower first-request
latency.

## Experiment E056-E057: Group Duplicate Zero-Token Batch Prompts

Hypothesis:

The batch `max_tokens=0` path already avoids all model work, but it still walked
every prompt row and therefore performed a rendered-prompt cache lookup and
prompt-token cache lookup for each duplicate prompt row. Since zero-output
responses only need prompt-token counts for usage, identical prompt rows can be
grouped, rendered/tokenized once, and then expanded back into prompt-index order.

Change:

- Reworked the `max_tokens=0` batch fast path to use the existing batch prompt
  grouping helper.
- Each identical prompt group now renders and tokenizes once, then fills
  `BatchCompletionItem`s for every prompt index and completion index in that
  group.
- Results are written through a prompt-index table and flattened in prompt
  order, so non-contiguous duplicate prompt rows preserve the public response
  order.
- Updated the duplicate zero-token batch test to assert true skipped work:
  one render miss, zero render hits, one token miss, and zero token hits for two
  identical prompt rows.

Verification:

- `cargo test -p kiln-server duplicate_batch_zero_prompts_skip_repeated_render_and_tokenize --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 20 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 2 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh server was started on port 8421 after the release build and 64-token
background prewarm completed. The request used 8 identical copies of the E041
4107-token prompt, `n=8`, and `max_tokens=0`, for 64 logical zero-output
completions and a total reported prompt usage of 262,848 tokens.

| Experiment | Shape | Wall time | Logical completions | Render cache | Token cache | Model work |
|---|---|---:|---:|---|---|---|
| E056 cold | 8 duplicate 4107-token prompts, `n=8`, `max_tokens=0` | 0.02 s | 64 | miss 0 -> 1, hits stayed 0 | miss 0 -> 1, hits stayed 0 | none |
| E057 warm | identical repeat | 0.01 s | 64 | hit 0 -> 1 | hit 0 -> 1 | none |

Both responses returned 64 completions, `completion_tokens=0`, and
`total_tokens=262848`. `kiln_tokens_generated_total`,
`kiln_request_prefill_duration_seconds_count`,
`kiln_request_decode_duration_seconds_count`, and prefix-cache lookup counters
all stayed at 0. The server-side handler duration was about 10.54 ms for E056
and 0.59 ms for E057.

Artifacts:

- `e056_health_after_prewarm.json`
- `e056_before_metrics.prom`
- `e056_after_metrics.prom`
- `e056_batch_long_zero_duplicate_prompts8_n8_response.json`
- `e056_batch_long_zero_duplicate_prompts8_n8_time.log`
- `e057_after_warm_metrics.prom`
- `e057_batch_long_zero_duplicate_prompts8_n8_warm_response.json`
- `e057_batch_long_zero_duplicate_prompts8_n8_warm_time.log`

Takeaway:

The batch zero-output fast path now removes duplicate front-end work too. Large
duplicate zero-token batches pay for one render and one tokenization per unique
prompt, not per prompt row or logical completion, while still reporting logical
usage for every returned completion.

## Experiment E058-E059: 128-Token Startup Prewarm Not Kept

Hypothesis:

E054 showed that increasing startup prewarm from 16 to 64 prompt tokens reduced
first-live multi-block request latency. A 128-token prewarm might warm more
shapes and further reduce the first live 36-token duplicate-batch request.

Change tested, then reverted:

- Temporarily increased background paged inference prewarm from 64 to 128
  prompt tokens.
- Rebuilt the release server and measured the same 8 duplicate prompt rows /
  `n=8` greedy `max_tokens=2` batch used in E052 and E054.
- Reverted to the kept 64-token prewarm and remeasured in the same warmed
  Metal/OS state to separate the prewarm-token-count effect from global kernel
  and file-cache warming.

Verification while testing:

- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result before 128-token measurement: passed.
- `cargo build --release --features metal --bin kiln`
  - Result before 128-token measurement: passed.
- `cargo build --release --features metal --bin kiln`
  - Result after reverting to 64-token prewarm: passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result after reverting to 64-token prewarm: passed.

Real server result:

The 128-token prewarm initially looked much faster than E054, but the control
run showed the speed came from warmed global Metal/OS state rather than the
larger prewarm. With a 128-token prewarm, the first live duplicate batch took
0.64 s wall with 0.461 s request prefill. After reverting to the 64-token
prewarm and launching a fresh server in the same warmed environment, the same
request again took 0.64 s wall with 0.462 s request prefill.

| Experiment | Startup prewarm | First live shape | Wall time | Request prefill | Request decode | Physical tokens | Kept? |
|---|---:|---|---:|---:|---:|---:|---|
| E058 | 128 tokens | 8 duplicate prompts, `n=8`, greedy, `max_tokens=2` | 0.64 s | 0.461 s | 0.170 s | 2 | no |
| E059 | 64 tokens control | same | 0.64 s | 0.462 s | 0.172 s | 2 | yes |

Artifacts:

- `e058_health_after_128tok_prewarm.json`
- `e058_before_metrics.prom`
- `e058_after_metrics.prom`
- `e058_batch_greedy_duplicate_prompts8_n8_128tok_prewarm_response.json`
- `e058_batch_greedy_duplicate_prompts8_n8_128tok_prewarm_time.log`
- `e059_health_after_64tok_recheck.json`
- `e059_before_metrics.prom`
- `e059_after_metrics.prom`
- `e059_batch_greedy_duplicate_prompts8_n8_64tok_recheck_response.json`
- `e059_batch_greedy_duplicate_prompts8_n8_64tok_recheck_time.log`

Takeaway:

Do not increase startup prewarm to 128 tokens. In a warmed environment, 64 and
128 tokens produce indistinguishable first-live latency for the tested
multi-block batch shape, while 128 tokens does more startup work. Keep the
64-token prewarm from E054-E055.

## Experiment E060-E061: Whole-Batch Deterministic Replay Cache

Hypothesis:

E056-E057 removed duplicate render/token work inside a zero-output batch, but a
warm repeated request still had to perform one rendered-prompt cache lookup and
one token-cache lookup before returning. For deterministic multi-output batches
with no adapters, the entire flattened batch response can be replayed from a
bounded cache before prompt rendering, tokenization, model work, or prefix-cache
lookup.

Change:

- Added a bounded `DeterministicBatchCache` to server state.
- Added an exact batch cache key for no-adapter multi-output deterministic
  batch requests, including prompt role/content pairs, `n`, `temperature`,
  `max_tokens`, stop strings, `top_p`, `top_k`, and `seed`.
- The cache covers `max_tokens=0`, greedy `temperature=0.0`, and seeded
  replayable sampling batch requests. Unseeded sampling and adapter requests are
  intentionally excluded.
- Early cache hits are gated on the server already being on the base model, so
  the replay path does not skip adapter unload work when a prior adapter is
  active.
- A hit rebuilds a fresh batch response id and timestamp while replaying cached
  completions and usage.
- A miss inserts the completed deterministic batch response after the existing
  zero-token or normal batch path returns.

Verification:

- `cargo test -p kiln-server repeated_multi_output_zero_batch_hits_batch_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 21 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 2 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 9209 ms. The request reused the E056 shape: 8
identical copies of the 4107-token prompt, `n=8`, and `max_tokens=0`, for 64
logical zero-output completions and reported prompt usage of 262,848 tokens.

| Experiment | Shape | Wall time | HTTP handler | Logical completions | Render cache | Token cache | Model/prefix work |
|---|---|---:|---:|---:|---|---|---|
| E060 populate | 8 duplicate 4107-token prompts, `n=8`, `max_tokens=0` | 0.01 s | 6.98 ms | 64 | miss 0 -> 1, hits stayed 0 | miss 0 -> 1, hits stayed 0 | none |
| E061 whole-batch hit | identical repeat | 0.00 s | 0.37 ms | 64 | unchanged: miss 1, hit 0 | unchanged: miss 1, hit 0 | none |

Both responses returned 64 completions, `completion_tokens=0`, and
`total_tokens=262848`, with fresh `batchcmpl-*` ids. The metrics confirm the
warm hit returned before prompt work: after E061, rendered-prompt lookups still
showed hit 0 / miss 1, prompt-token lookups still showed hit 0 / miss 1, and
`kiln_tokens_generated_total`, prefill/decode histogram counts, and
prefix-cache lookup counters all stayed at 0. The request-duration metric sum
increased from 0.006634 s after E060 to 0.006813 s after E061.

Artifacts:

- `e060_health_after_prewarm.json`
- `e060_before_metrics.prom`
- `e060_after_populate_metrics.prom`
- `e060_batch_long_zero_duplicate_prompts8_n8_batch_cache_populate_response.json`
- `e060_batch_long_zero_duplicate_prompts8_n8_batch_cache_populate_time.log`
- `e061_after_hit_metrics.prom`
- `e061_batch_long_zero_duplicate_prompts8_n8_batch_cache_hit_response.json`
- `e061_batch_long_zero_duplicate_prompts8_n8_batch_cache_hit_time.log`

Takeaway:

Repeated deterministic multi-output batches can now become true replay hits at
the batch boundary. For the large duplicate zero-token shape, the first request
does one render and one tokenization for the unique prompt, then the repeated
request skips even those cache lookups and returns in sub-millisecond handler
time with no model or prefix-cache work.

## Experiment E062: Whole-Batch Singleflight for Concurrent Duplicate Batches

Hypothesis:

E060-E061 made repeated deterministic batches replayable after the first request
finished. It did not remove duplicate work when two identical cold deterministic
batch requests arrived concurrently. The same owner/waiter singleflight pattern
used for deterministic single completions should let one batch do the physical
work while concurrent duplicates wait for and replay the completed batch value.

Change:

- Added an in-flight map to `DeterministicBatchCache`.
- Added batch cache `claim`, `complete`, and `fail` operations with a watch
  channel for waiters.
- Added a `BatchCacheOwnerGuard` in the batch endpoint so errors wake waiters
  instead of leaving an in-flight entry pending.
- Kept the same no-adapter, deterministic, multi-output, base-model early-hit
  gate from E060-E061.

Verification:

- `cargo test -p kiln-server deterministic_batch_cache_coalesces_in_flight_request --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server concurrent_multi_output_greedy_batch_singleflights_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 23 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 2 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 11063 ms. Two identical HTTP requests were fired
concurrently using the E052/E054 duplicate greedy shape: 8 identical prompt rows,
`n=8`, `temperature=0.0`, and `max_tokens=2`, for 64 logical completions per
HTTP response.

Both client requests completed in 2.85 s wall time. The HTTP handler durations
were 2836.62 ms for both requests, as expected for a waiter that returns when
the owner publishes the completed batch response. The two responses had fresh
`batchcmpl-*` ids, but were identical after removing `id` and `created`.

Aggregate metrics after both concurrent requests:

- `kiln_requests_total{status="ok"}`: 0 -> 2
- Request duration count/sum: 0 -> 2 / 5.672885 s
- Prefill count/sum: 0 -> 1 / 2.598596 s
- Decode count/sum: 0 -> 1 / 0.223444 s
- `kiln_tokens_generated_total`: 0 -> 2
- Rendered-prompt cache: hit 0, miss 0 -> hit 0, miss 1
- Prompt-token cache: hit 0, miss 0 -> hit 0, miss 1
- Prefix-cache lookups: hit 0, miss 0 -> hit 0, miss 1
- Prefix-cache retained entries/blocks: 0/0 -> 1/3

Without batch singleflight, two cold concurrent requests of this shape would be
eligible to duplicate the first render/token/prefix/model path. With E062, two
HTTP responses were served from one physical prompt render, one tokenization,
one prefix-cache miss, one prefill, one decode path, and two physical generated
tokens.

Artifacts:

- `e062_health_after_prewarm.json`
- `e062_before_metrics.prom`
- `e062_after_metrics.prom`
- `e062_concurrent_batch_singleflight_response_a.json`
- `e062_concurrent_batch_singleflight_response_b.json`
- `e062_concurrent_batch_singleflight_time_a.log`
- `e062_concurrent_batch_singleflight_time_b.log`
- `e062_concurrent_batch_singleflight_status.log`

Takeaway:

Duplicate deterministic batch requests now remove work both after completion and
while the first identical request is still running. This improves bs>1
throughput under concurrent retry/fanout traffic without changing the response
contract: waiters get fresh ids and the same logical completions/usage.

## Experiment E063-E064: Request-Level Chat Replay Before Render/Tokenize

Hypothesis:

The full completion cache is keyed by prompt tokens, so even a warm deterministic
bs=1 chat cache hit still has to render the chat template and look up the
rendered prompt in the token cache before it can find the cached completion.
For exact base-model no-adapter deterministic chat requests, the response can be
replayed from a request-level cache before chat-template rendering and
tokenization.

Change:

- Added `DeterministicChatRequestCache` to server state.
- The key is an exact serialized request shape for no-adapter, non-streaming
  deterministic chat: messages, tools, tool choice, `temperature`,
  `max_tokens`, stop strings, `top_p`, `top_k`, and `seed`.
- The cached value stores both the response completion payload and the prompt
  token count needed for usage.
- Added an in-flight owner/waiter path so concurrent identical cold chat
  requests can singleflight before prompt rendering.
- Early hits are gated on the server already being on the base model, preserving
  adapter unload behavior.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_coalesces_in_flight_request --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server repeated_zero_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server concurrent_zero_chat_singleflights_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 23 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 2 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 10207 ms. The request reused the E041/E042
long zero-output bs=1 chat shape: one 4107-token prompt, `temperature=0.7`, and
`max_tokens=0`.

| Experiment | Shape | Wall time | HTTP handler | Render cache | Token cache | Model/prefix work |
|---|---|---:|---:|---|---|---|
| E063 populate | 4107-token chat, `max_tokens=0` | 0.01 s | 7.55 ms | miss 0 -> 1, hits stayed 0 | miss 0 -> 1, hits stayed 0 | none |
| E064 request-cache hit | identical repeat | 0.00 s | 0.096 ms | unchanged: miss 1, hit 0 | unchanged: miss 1, hit 0 | none |

Both responses reported `prompt_tokens=4107`, `completion_tokens=0`, and
`total_tokens=4107`, with fresh `chatcmpl-*` ids. The responses were identical
after removing `id` and `created`. The request-duration metric sum increased
from 0.007063 s after E063 to 0.007105 s after E064. Prefill/decode histogram
counts, generated-token counters, and prefix-cache lookup counters all stayed
at 0.

Artifacts:

- `e063_health_after_prewarm.json`
- `e063_before_metrics.prom`
- `e063_after_populate_metrics.prom`
- `e063_chat_long_zero_request_cache_populate_response.json`
- `e063_chat_long_zero_request_cache_populate_time.log`
- `e064_after_hit_metrics.prom`
- `e064_chat_long_zero_request_cache_hit_response.json`
- `e064_chat_long_zero_request_cache_hit_time.log`

Takeaway:

Warm deterministic bs=1 chat requests can now skip the front-end prompt work
entirely. For the long zero-output prompt, the repeated request no longer even
does rendered-prompt or token-cache lookups and returns in about 0.1 ms of
handler time.

## Experiment E065-E066: Request-Level Chat Replay for Nonzero Greedy bs=1

Hypothesis:

E063-E064 proved the request-level chat cache on `max_tokens=0`, where no model
work is involved. The same request-level cache should also remove all work on a
warm nonzero deterministic bs=1 chat request: no render/token lookup, no
prefix-cache lookup, no prefill, no decode, and no generated-token counter
increase.

Change:

- No code change after E063-E064. This measurement verifies the already-added
  request-level deterministic chat replay on a nonzero greedy request.

Verification:

- Reused the already-passing E063-E064 code verification:
  - `cargo test -p kiln-server deterministic_chat_request_cache_coalesces_in_flight_request --lib`
  - `cargo test -p kiln-server repeated_zero_chat_hits_request_cache_before_prompt_work --lib`
  - `cargo test -p kiln-server concurrent_zero_chat_singleflights_before_prompt_work --lib`
  - `cargo test -p kiln-server batch_ --lib`
  - `cargo test -p kiln-server completion_cache --lib`
  - `cargo test -p kiln-server prompt_token_cache --lib`
  - `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - `cargo build --release --features metal --bin kiln`

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 10813 ms. The request used the short bs=1 greedy
chat fixture: 35 prompt tokens, `temperature=0.0`, and `max_tokens=2`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E065 populate | 35-token chat, greedy, `max_tokens=2` | 3.32 s | 3316.84 ms | 3.101 s | 0.213 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E066 request-cache hit | identical repeat | 0.01 s | 0.083 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=35`, `completion_tokens=2`, and
`total_tokens=37`, with fresh `chatcmpl-*` ids. The responses were identical
after removing `id` and `created`. Metrics after E066 showed:

- Prefill count/sum stayed at 1 / 3.101069 s.
- Decode count/sum stayed at 1 / 0.213251 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 3.316779 s to 3.316802 s.

Artifacts:

- `e065_health_after_prewarm.json`
- `e065_before_metrics.prom`
- `e065_after_populate_metrics.prom`
- `e065_chat_greedy_request_cache_populate_response.json`
- `e065_chat_greedy_request_cache_populate_time.log`
- `e066_after_hit_metrics.prom`
- `e066_chat_greedy_request_cache_hit_response.json`
- `e066_chat_greedy_request_cache_hit_time.log`

Takeaway:

The request-level chat cache is not just a zero-output fast path. It removes the
entire warm deterministic bs=1 non-streaming chat path after the first response,
including front-end prompt work, prefix-cache lookup, prefill, and decode.

## Experiment E067-E068: Request-Level Chat Replay for Warm Streaming

Hypothesis:

E063-E066 excluded `stream: true` from the request-level chat cache, so a warm
streaming request still had to render/tokenize before it could hit the older
completion cache. A streaming request can safely consume an already-populated
request-level cache entry and rebuild an SSE response before prompt rendering.
To keep streaming publication behavior simple, streaming requests should be
cache consumers only; non-streaming requests remain the primary owners that
populate the cache.

Change:

- Removed `stream` from the request-level cache exclusion and key, so
  deterministic streaming and non-streaming requests share the same cached
  completion payload.
- Added a read-only `probe` path for `DeterministicChatRequestCache`.
- Streaming requests now hit or wait on an existing request-cache entry before
  prompt rendering/tokenization, but they do not claim ownership on a miss.
- Reused the cached completion payload to rebuild SSE chunks with a fresh
  `chatcmpl-*` id and `[DONE]` terminator.

Verification:

- `cargo test -p kiln-server chat_streaming_repeated_greedy_request_uses_completion_cache --lib`
  - Result: 1 passed; now asserts the streaming hit returns before rendered-
    prompt and token-cache lookups.
- `cargo test -p kiln-server repeated_zero_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_request_cache_coalesces_in_flight_request --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 23 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 10707 ms. E067 populated the request cache with
the short bs=1 greedy non-streaming chat fixture: 35 prompt tokens,
`temperature=0.0`, and `max_tokens=2`. E068 sent the same request with
`stream: true`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E067 populate | 35-token chat, greedy, `max_tokens=2` | 2.95 s | 2944.27 ms | 2.733 s | 0.208 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E068 streaming request-cache hit | same request, `stream: true` | 0.00 s | 0.064 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

The E068 SSE contained an assistant role chunk, a `reasoning_content` chunk
with `Thinking Process`, a final `finish_reason="length"` chunk, and `[DONE]`.
Metrics after E068 showed:

- Prefill count/sum stayed at 1 / 2.732938 s.
- Decode count/sum stayed at 1 / 0.207767 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.944087 s to 2.944105 s.

Artifacts:

- `e067_health_after_prewarm.json`
- `e067_before_metrics.prom`
- `e067_after_populate_metrics.prom`
- `e067_chat_greedy_request_cache_stream_populate_response.json`
- `e067_chat_greedy_request_cache_stream_populate_time.log`
- `e068_after_stream_hit_metrics.prom`
- `e068_chat_greedy_request_cache_stream_hit_response.sse`
- `e068_chat_greedy_request_cache_stream_hit_time.log`

Takeaway:

Warm deterministic streaming chat can now share the request-level replay path
with non-streaming chat. After a matching deterministic response is cached, a
streaming client gets SSE in sub-millisecond handler time without paying
chat-template rendering, tokenization, prefix-cache lookup, prefill, or decode.

## Experiment E069-E070: Streaming-First Request Cache Publication

Hypothesis:

E067-E068 made streaming requests consume an existing request-level chat cache
entry before prompt rendering, but a stream that arrived first still only
published the lower-level completion cache. That meant a second identical
streaming request had to render and tokenize before it could discover the
cached completion. A successful live stream already accumulates the final
content, reasoning content, finish reason, completion-token count, and prompt
token count; it can publish the same deterministic request-cache payload when
the model reports `Done`.

Change:

- Passed the deterministic chat request-cache key into the real streaming task.
- On successful stream completion, built one `DeterministicCompletionCacheValue`
  from the accumulated streaming buffers and inserted it into both the
  deterministic completion cache and deterministic chat request cache.
- Left timeout, early error, and client-disconnect paths uncached. They still
  record recent-request state but do not publish a replayable request payload.
- Added a unit test that locks the cache-key policy: deterministic streaming
  and non-streaming chat requests share the same payload key.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_ignores_stream_flag --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_streaming_repeated_greedy_request_uses_completion_cache --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_request_cache_coalesces_in_flight_request --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server repeated_zero_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 23 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 10050 ms. E069 sent the short bs=1 greedy chat
fixture as `stream: true` with `temperature=0.0`, `max_tokens=2`, and
`seed=123`. E070 repeated the identical streaming request without any
non-streaming populate in between.

| Experiment | Shape | Curl total | Curl TTFB | HTTP handler | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---|
| E069 streaming populate | greedy `stream: true`, `max_tokens=2` | 3.199778 s | 2.502 ms | 2.064 ms | 2 | render miss 1, token miss 1, prefix miss 1 |
| E070 streaming request-cache hit | identical repeat | 0.000792 s | 0.609 ms | 0.062 ms | unchanged at 2 | render/token/prefix counters unchanged |

The live streaming path currently does not populate the prefill/decode
histograms, so the physical-work proof for E069-E070 is the generated-token
counter plus render/token/prefix counters. Metrics after E070 showed:

- `kiln_tokens_generated_total` stayed at 2 after the second stream.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Prefix-cache retained one cached block and one entry from the first stream.
- Request-duration histogram count rose from 1 to 2, but sum moved only from
  0.001889 s to 0.001906 s because it measures streaming response setup, not
  full SSE wall time.

The first SSE emitted separate reasoning chunks for `Okay` and `,`; the cached
replay emitted the same reasoning payload as one `Okay,` chunk, with a fresh
`chatcmpl-*` id, final `finish_reason="length"`, and `[DONE]`.

Artifacts:

- `e069_health_after_prewarm.json`
- `e069_before_metrics.prom`
- `e069_after_stream_populate_metrics.prom`
- `e069_chat_greedy_stream_first_populate_response.sse`
- `e069_chat_greedy_stream_first_populate_time.log`
- `e070_after_stream_hit_metrics.prom`
- `e070_chat_greedy_stream_first_hit_response.sse`
- `e070_chat_greedy_stream_first_hit_time.log`

Takeaway:

The no-work deterministic streaming path no longer depends on a prior
non-streaming request. A stream can now populate the request-level chat cache
itself, and the next identical stream returns before chat-template rendering,
tokenization, prefix-cache lookup, or model decode.

## Experiment E071: Concurrent Streaming Request Singleflight

Hypothesis:

After E069-E070, sequential streaming repeats were no-work, but two identical
streaming requests that arrived together could still both miss before prompt
work. If the first stream claims the request-level chat cache and carries that
owner into the spawned streaming task, the second stream can wait for the first
physical stream to finish and then replay the cached SSE payload instead of
doing duplicate render, tokenization, prefix-cache, and model work.

Change:

- Changed streaming request-cache misses from read-only `probe` calls to the
  same `claim` path used by non-streaming chat.
- Moved the `ChatRequestCacheOwnerGuard` into `generate_real_streaming` so a
  successful stream completes the in-flight request-cache entry from the
  spawned task.
- Let timeout, early error, or client-disconnect paths drop the owner, which
  fails waiters rather than publishing a partial cached payload.
- Removed the now-unused read-only chat request-cache probe API.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_ignores_stream_flag --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_streaming_repeated_greedy_request_uses_completion_cache --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_request_cache_coalesces_in_flight_request --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server prompt_token_cache --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 23 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 7548 ms. E071 sent two simultaneous identical
greedy streaming chat requests using `temperature=0.0`, `max_tokens=2`, and
`seed=123`.

| Experiment | Shape | Curl total | Curl TTFB | HTTP handler | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---|
| E071 stream A | first identical stream | 0.463991 s | 1.546 ms | 1.111 ms | 2 total | render miss 1, token miss 1, prefix miss 1 total |
| E071 stream B | waiter/replay stream | 0.465448 s | 464.779 ms | 463.526 ms | unchanged at 2 | render/token/prefix counters unchanged |

Aggregate metrics after both responses:

- `kiln_requests_total{status="ok"}` was 2.
- `kiln_tokens_generated_total` was 2, not 4.
- Rendered-prompt cache was hit 0 / miss 1.
- Prompt-token cache was hit 0 / miss 1.
- Prefix-cache lookups were hit 0 / miss 1.
- Prefix-cache retained one cached block and one entry.
- Request-duration histogram sum was 0.464258 s; this mostly reflects stream
  B waiting in the request-cache singleflight before its cached SSE response
  was constructed.

The first SSE streamed the live reasoning chunks (`Okay`, then `,`). The
waiter replay produced the same logical reasoning payload as one `Okay,` chunk,
with a fresh `chatcmpl-*` id, final `finish_reason="length"`, and `[DONE]`.

Artifacts:

- `e071_health_after_prewarm.json`
- `e071_before_metrics.prom`
- `e071_after_metrics.prom`
- `e071_concurrent_stream_singleflight_response_a.sse`
- `e071_concurrent_stream_singleflight_response_b.sse`
- `e071_concurrent_stream_singleflight_time_a.log`
- `e071_concurrent_stream_singleflight_time_b.log`
- `e071_concurrent_stream_singleflight_status.log`

Takeaway:

Concurrent duplicate deterministic streams now collapse to one physical
streaming generation. The waiting duplicate pays wall-clock wait time, but it
does no prompt rendering, tokenization, prefix-cache lookup, or model decode.

## Experiment E072-E073: Whole-Batch Replay for Single-Output Batch Requests

Hypothesis:

E060-E062 added whole-batch deterministic replay before prompt rendering, but
`deterministic_batch_cache_key` still returned `None` when `total_outputs <= 1`.
That left the common single-prompt, `n=1` batch shape on the older lower-level
completion-cache path: the warm repeat avoided model generation, but still had
to render and tokenize the prompt before it could find the cached completion.

Change:

- Removed the `total_outputs <= 1` exclusion from the deterministic batch-cache
  key. The existing empty-prompt validation means the key only has to reject
  `total_outputs == 0`.
- Updated the single-output greedy batch cache test so the second identical
  request must return before rendered-prompt and prompt-token cache lookup.
- Documented that early whole-batch cache hits do not synthesize per-output
  recent-request records, matching the existing multi-output batch cache
  behavior.

Verification:

- `cargo test -p kiln-server batch_repeated_greedy_request_hits_batch_cache_before_completion_cache_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 23 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 10207 ms. E072 sent one single-output batch:
one prompt, `n=1`, `temperature=0.0`, `max_tokens=2`, and `seed=123`. E073
repeated the identical request.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E072 populate | one prompt, `n=1`, greedy, `max_tokens=2` | 3.40 s | 3392.34 ms | 3.183 s | 0.205 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E073 batch-cache hit | identical repeat | 0.01 s | 0.070 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=12`, `completion_tokens=2`, and
`total_tokens=14`, with fresh `batchcmpl-*` ids. Metrics after E073 showed:

- Prefill count/sum stayed at 1 / 3.183408 s.
- Decode count/sum stayed at 1 / 0.205162 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 3.392143 s to 3.392160 s.

Artifacts:

- `e072_health_after_prewarm.json`
- `e072_before_metrics.prom`
- `e072_after_populate_metrics.prom`
- `e072_batch_single_output_cache_populate_response.json`
- `e072_batch_single_output_cache_populate_time.log`
- `e073_after_hit_metrics.prom`
- `e073_batch_single_output_cache_hit_response.json`
- `e073_batch_single_output_cache_hit_time.log`

Takeaway:

The no-work whole-batch replay path now covers the single-output batch shape as
well as multi-output batches. A deterministic `n=1` batch repeat can return
before chat-template rendering, tokenization, prefix-cache lookup, prefill, or
decode.

## Experiment E074-E077: Normalize Equivalent Greedy Request-Cache Keys

Hypothesis:

The lower-level deterministic completion cache already normalizes greedy
sampling fields: when `temperature=0.0`, seed, top-p, and top-k do not affect
the token path. The earlier chat request cache and whole-batch cache still kept
those fields in their keys, so equivalent greedy requests with different
client-supplied seeds or filters could miss before prompt rendering and only
reuse work later, after render/tokenization.

Change:

- Added one shared request-cache sampling-key normalizer.
- For `temperature=0.0`, request-level chat and batch keys now normalize
  `temperature_bits=0.0`, `top_p_bits=1.0`, `top_k=0`, and `seed=None`, while
  still keeping `stop` because it can change a non-empty deterministic output.
- For `max_tokens=0`, request-level keys also normalize `stop` to empty and
  ignore all generation-only sampling fields, because no output tokens are
  generated.
- Seeded sampled requests still keep seed/top-p/top-k/temperature in the key.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_normalizes_equivalent_sampling_fields --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_batch_cache_key_normalizes_equivalent_sampling_fields --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 24 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 10352 ms. E074/E075 used identical chat
messages with `temperature=0.0` and `max_tokens=2`, but changed
`seed/top_p/top_k` from `1/0.8/17` to `2/0.95/0`. E076/E077 repeated the same
normalization check for a single-output batch request.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E074 chat populate | greedy chat, seed/top-p/top-k A | 3.19 s | 3186.81 ms | 2.965 s | 0.219 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E075 normalized chat hit | same prompt, seed/top-p/top-k B | 0.01 s | 0.060 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E076 batch populate | single-output greedy batch, seed/top-p/top-k A | 0.44 s | 433.34 ms | +0.254 s | +0.174 s | +2 | +1 render miss, +1 token miss, +1 prefix miss |
| E077 normalized batch hit | same prompt, seed/top-p/top-k B | 0.01 s | 0.069 ms | unchanged | unchanged | unchanged at 4 total | render/token/prefix counters unchanged |

The chat pair both reported `prompt_tokens=14`, `completion_tokens=2`, and the
same reasoning payload (`The user`) with fresh `chatcmpl-*` ids. The batch pair
both reported `prompt_tokens=14`, `completion_tokens=2`, and fresh
`batchcmpl-*` ids. Metrics after E075 showed:

- Prefill count/sum stayed at 1 / 2.964824 s.
- Decode count/sum stayed at 1 / 0.218783 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 3.186611 s to 3.186627 s.

Metrics after E077 showed the batch hit similarly changed only request counts
and duration sum: generated tokens stayed at 4 total, render/token misses
stayed at 2 total, and prefix-cache misses stayed at 2 total.

Artifacts:

- `e074_health_after_prewarm.json`
- `e074_before_metrics.prom`
- `e074_after_populate_metrics.prom`
- `e074_chat_greedy_normalized_populate_response.json`
- `e074_chat_greedy_normalized_populate_time.log`
- `e075_after_hit_metrics.prom`
- `e075_chat_greedy_normalized_hit_response.json`
- `e075_chat_greedy_normalized_hit_time.log`
- `e076_after_populate_metrics.prom`
- `e076_batch_greedy_normalized_populate_response.json`
- `e076_batch_greedy_normalized_populate_time.log`
- `e077_after_hit_metrics.prom`
- `e077_batch_greedy_normalized_hit_response.json`
- `e077_batch_greedy_normalized_hit_time.log`

Takeaway:

Equivalent greedy requests now converge at the earliest request-cache layer
instead of only at the token-level completion cache. Varying seed/top-p/top-k
on greedy chat or batch requests no longer forces prompt rendering,
tokenization, prefix-cache lookup, prefill, or decode after an equivalent
request is cached.

## Experiment E078-E081: Share Chat Request Cache with Single-Output Batch

Hypothesis:

The request-level chat cache and whole-batch cache remove prompt work within
their own endpoints, but they were isolated. A chat request followed by an
equivalent single-output batch still missed the batch cache and had to render
and tokenize before the lower-level completion cache could fire. Conversely, a
single-output batch followed by equivalent chat did not populate the chat
request cache before this experiment.

Change:

- Added request-level chat-cache claim/hit logic inside `generate_one_response`,
  the helper used by batch for each synthesized single chat completion.
- Gated that path on `active_adapter_name == None`, matching the base-model
  request-cache safety boundary; adapter and composed-adapter batches remain
  excluded.
- On completion-cache hit, wait, or successful physical generation inside
  `generate_one_response`, finish the equivalent chat request-cache owner so a
  later `/v1/chat/completions` request can replay before prompt work.
- If a chat request populated the request cache first, a compatible
  single-output batch can now use that cached chat response before rendering or
  tokenization, then populate the whole-batch cache from the shaped batch
  response.

Verification:

- `cargo test -p kiln-server batch_single_output_hits_chat_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_hits_request_cache_populated_by_single_output_batch --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 25 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 after the kept 64-token
background prewarm completed in 11091 ms. E078/E079 used the same prompt text
for chat first, then equivalent single-output batch. E080/E081 used a different
prompt text for batch first, then equivalent chat, avoiding reuse of the first
pair's cache entries.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E078 chat populate | chat, greedy, `max_tokens=2` | 2.83 s | 2828.72 ms | 2.619 s | 0.206 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E079 batch cross-endpoint hit | equivalent single-output batch | 0.01 s | 0.217 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E080 batch populate | new single-output batch | 0.44 s | 431.74 ms | +0.258 s | +0.169 s | +2 | +1 render miss, +1 token miss, +1 prefix miss |
| E081 chat cross-endpoint hit | equivalent chat | 0.01 s | 0.079 ms | unchanged | unchanged | unchanged at 4 total | render/token/prefix counters unchanged |

The E078/E079 pair both reported `prompt_tokens=14`, `completion_tokens=2`,
and `total_tokens=16`; the chat response carried `reasoning_content="Okay,"`
while the batch response returned empty `text`, matching the batch endpoint's
existing behavior of exposing only assistant content. The E080/E081 pair had
the same usage counts and fresh ids.

Metrics after E079 showed:

- Prefill count/sum stayed at 1 / 2.618515 s.
- Decode count/sum stayed at 1 / 0.206337 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.828542 s to 2.828688 s.

Metrics after E081 showed the reverse direction also left physical-work
counters unchanged from E080: generated tokens stayed at 4 total, render/token
misses stayed at 2 total, and prefix misses stayed at 2 total.

Artifacts:

- `e078_health_after_prewarm.json`
- `e078_before_metrics.prom`
- `e078_after_chat_populate_metrics.prom`
- `e078_chat_cross_endpoint_populate_response.json`
- `e078_chat_cross_endpoint_populate_time.log`
- `e079_after_batch_hit_metrics.prom`
- `e079_batch_cross_endpoint_hit_response.json`
- `e079_batch_cross_endpoint_hit_time.log`
- `e080_after_batch_populate_metrics.prom`
- `e080_batch_cross_endpoint_populate_response.json`
- `e080_batch_cross_endpoint_populate_time.log`
- `e081_after_chat_hit_metrics.prom`
- `e081_chat_cross_endpoint_hit_response.json`
- `e081_chat_cross_endpoint_hit_time.log`

Takeaway:

The no-work replay layer now crosses the chat and single-output batch endpoint
boundary. Equivalent base-model deterministic requests can reuse the same
request-level completion payload before prompt rendering, tokenization,
prefix-cache lookup, prefill, or decode, regardless of which endpoint populated
it first.

## Experiment E082-E085: Share Chat Request Cache with Multi-Output Greedy Batch

Hypothesis:

E078-E081 proved request-cache reuse across chat and single-output batch. The
same mechanism should cover greedy `n>1` batch because that path performs one
physical generation, then clones the deterministic response into the logical
batch outputs. If true, a cached chat response can fan out to an `n=8` batch
with no prompt or model work, and an `n=8` batch populate can seed the
equivalent chat cache for a later bs=1 hit.

Change:

- Kept the production path unchanged: the earlier `generate_one_response`
  request-cache claim already sits inside the one physical greedy batch
  generation.
- Added regression coverage for both multi-output directions:
  - `greedy_multi_output_batch_clones_cached_chat_response_before_prompt_work`
  - `chat_hits_request_cache_populated_by_greedy_multi_output_batch`
- The tests also lock the greedy key normalization behavior by changing seed
  between the populate and hit requests.

Verification:

- `cargo test -p kiln-server multi_output_batch --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 26 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

Two fresh release servers were started on port 8421. E082/E083 used chat first,
then equivalent `n=8` batch; background prewarm completed in 11766 ms. E084/E085
used a different prompt for `n=8` batch first, then equivalent chat; background
prewarm completed in 7522 ms.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E082 chat populate | chat, greedy, `max_tokens=2` | 2.75 s | 2744.92 ms | 2.553 s | 0.187 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E083 multi-batch from chat cache hit | equivalent `n=8` batch | 0.01 s | 0.171 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E084 multi-batch populate | new `n=8` batch, greedy, `max_tokens=2` | 0.48 s | 476.02 ms | 0.299 s | 0.171 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E085 chat from multi-batch cache hit | equivalent chat | 0.01 s | 0.095 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

E082 returned chat usage `prompt_tokens=17`, `completion_tokens=2`,
`total_tokens=19`. E083 returned eight logical completions with aggregate usage
`prompt_tokens=136`, `completion_tokens=16`, `total_tokens=152`; each logical
completion carried the same per-completion usage as E082. E084 returned eight
logical completions with aggregate usage `prompt_tokens=120`,
`completion_tokens=16`, `total_tokens=136`; E085 returned the matching bs=1
chat usage `prompt_tokens=15`, `completion_tokens=2`, `total_tokens=17`.

Metrics after E083 showed:

- Prefill count/sum stayed at 1 / 2.553501 s.
- Decode count/sum stayed at 1 / 0.187447 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.744750 s to 2.744869 s.

Metrics after E085 showed:

- Prefill count/sum stayed at 1 / 0.298983 s.
- Decode count/sum stayed at 1 / 0.171031 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 0.475823 s to 0.475869 s.

Artifacts:

- `e082_server.log`
- `e082_health_after_prewarm.json`
- `e082_before_metrics.prom`
- `e082_after_chat_populate_metrics.prom`
- `e082_chat_to_multi_batch_populate_response.json`
- `e082_chat_to_multi_batch_populate_time.log`
- `e083_after_batch_hit_metrics.prom`
- `e083_multi_batch_from_chat_cache_hit_response.json`
- `e083_multi_batch_from_chat_cache_hit_time.log`
- `e084_server.log`
- `e084_health_after_prewarm.json`
- `e084_before_metrics.prom`
- `e084_after_batch_populate_metrics.prom`
- `e084_multi_batch_to_chat_populate_response.json`
- `e084_multi_batch_to_chat_populate_time.log`
- `e085_after_chat_hit_metrics.prom`
- `e085_chat_from_multi_batch_cache_hit_response.json`
- `e085_chat_from_multi_batch_cache_hit_time.log`

Takeaway:

Multi-output greedy batch now shares the same earliest no-work replay layer as
bs=1 chat. A deterministic base-model response can cross between chat and
`n>1` batch before rendering, tokenization, prefix-cache lookup, prefill, or
decode, while preserving logical batch usage accounting.

## Experiment E086-E087: Rejected Top-Level Multi-Batch Chat-Cache Fan-Out

Hypothesis:

E083 still entered the normal batch scheduling path before `generate_one_response`
hit the chat request cache. A top-level batch shortcut that directly looked up
complete chat request-cache entries and shaped an `n=8` response might reduce
sub-millisecond handler overhead.

Change tested:

- Added a read-only, LRU-updating chat request-cache lookup.
- Added a batch fast path that fired only when every greedy `n>1` prompt group
  already had a complete chat request-cache entry.
- The path preserved logical usage accounting and still populated the
  whole-batch cache.

Verification while testing:

- `cargo test -p kiln-server multi_output_batch --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 26 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11345 ms. E086 populated the chat request cache; E087 used the tested
top-level batch fan-out path.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E086 chat populate | chat, greedy, `max_tokens=2` | 4.12 s | 4119.45 ms | 3.893 s | 0.223 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E087 direct multi-batch hit | equivalent `n=8` batch via tested shortcut | 0.00 s | 0.445 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

E087 preserved the no-work model counters and returned eight logical completions
with aggregate usage `prompt_tokens=120`, `completion_tokens=16`,
`total_tokens=136`. However, it was slower than the kept E083 path:

- E083 kept path: 0.171 ms HTTP handler and +0.000119 s request-duration sum.
- E087 tested shortcut: 0.445 ms HTTP handler and +0.000406 s
  request-duration sum.

Decision:

Rejected and reverted. The extra cache lookups and direct shaping code were not
worth keeping; the existing `generate_one_response` cache-hit path is faster on
the measured real server.

Post-revert verification:

- `cargo test -p kiln-server multi_output_batch --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 26 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Artifacts:

- `e086_server.log`
- `e086_health_after_prewarm.json`
- `e086_before_metrics.prom`
- `e086_after_chat_populate_metrics.prom`
- `e086_chat_direct_multi_batch_populate_response.json`
- `e086_chat_direct_multi_batch_populate_time.log`
- `e087_after_batch_hit_metrics.prom`
- `e087_direct_multi_batch_from_chat_cache_hit_response.json`
- `e087_direct_multi_batch_from_chat_cache_hit_time.log`

## Experiment E088-E089: Normalize Empty Tools as a Cache No-Op

Hypothesis:

OpenAI-compatible clients may send `tools: []` even when no tools are available.
The tokenizer already treats an empty tools slice as no tools, so the rendered
Qwen3.5 prompt is identical to omitting `tools`. The request-level chat cache
and rendered-prompt cache still keyed those shapes differently, which meant an
otherwise equivalent `tools: []` request could miss the earliest cache layer and
redo prompt rendering/tokenization before falling through to a lower cache.

Change:

- Added `normalized_tools_for_cache`, which maps `Some([])` to `None`.
- Used it in both the rendered-prompt cache key and deterministic chat
  request-cache key.
- Non-empty `tools` still split cache entries and render through the tool
  template path.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_normalizes_empty_tools --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server empty_tools_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server tools_ --lib`
  - Result: 4 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 20 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 10817 ms. E088 populated a greedy no-tools chat request. E089 repeated the
same message with `tools: []` and a different seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E088 no-tools populate | chat, greedy, `max_tokens=2` | 2.87 s | 2869.53 ms | 2.667 s | 0.199 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E089 empty-tools request-cache hit | equivalent chat with `tools: []` | 0.01 s | 0.097 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=17`, `completion_tokens=2`,
`total_tokens=19`, with `reasoning_content="Okay,"`. Metrics after E089 showed:

- Prefill count/sum stayed at 1 / 2.666866 s.
- Decode count/sum stayed at 1 / 0.199285 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.869345 s to 2.869365 s.

Artifacts:

- `e088_server.log`
- `e088_health_after_prewarm.json`
- `e088_before_metrics.prom`
- `e088_after_populate_metrics.prom`
- `e088_no_tools_populate_response.json`
- `e088_no_tools_populate_time.log`
- `e089_after_hit_metrics.prom`
- `e089_empty_tools_hit_response.json`
- `e089_empty_tools_hit_time.log`

Takeaway:

Empty tool arrays now reuse the same no-work replay path as omitted tools.
This removes prompt rendering, tokenization, prefix-cache lookup, prefill, and
decode for a common client-side no-op field while preserving non-empty
tool-bearing prompts as distinct cache entries.

## Experiment E090-E091: Normalize No-Tool Auto Tool Choice

Hypothesis:

Some OpenAI-compatible clients send `tool_choice: "auto"` or `"none"` even when
no tools are provided. With no tools, those choices are semantic no-ops: there
is no tool call path to select. Before this experiment, the rendered-prompt
cache and request-level chat cache still keyed those no-op choices separately,
which could force prompt rendering/tokenization and lower-level cache work for
an otherwise identical request.

Change:

- Added `normalized_tool_choice_for_cache`.
- When normalized tools are absent, `tool_choice: "auto"` and
  `tool_choice: "none"` are normalized to omitted for rendered-prompt and
  deterministic chat request-cache keys.
- `tool_choice: "required"` and object choices remain distinct when no tools
  are present, and all non-empty tool lists remain distinct.
- `render_prompt_text` now passes the normalized no-op values to the tokenizer,
  so the rendered prompt and the rendered-prompt cache key stay aligned.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_normalizes_no_tool_auto_choice --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server no_tool_auto_choice_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server tools_ --lib`
  - Result: 4 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 22 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 9741 ms. E090 populated a greedy no-tools chat request. E091 repeated the
same message with `tools: []`, `tool_choice: "auto"`, and a different seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E090 no-tool-choice populate | chat, greedy, `max_tokens=2` | 2.33 s | 2326.43 ms | 2.123 s | 0.200 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E091 no-tool auto-choice hit | equivalent chat with `tools: []`, `tool_choice: "auto"` | 0.01 s | 0.108 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=18`, `completion_tokens=2`,
`total_tokens=20`, with `reasoning_content="Okay,"`. Metrics after E091 showed:

- Prefill count/sum stayed at 1 / 2.122975 s.
- Decode count/sum stayed at 1 / 0.200225 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.326244 s to 2.326265 s.

Artifacts:

- `e090_server.log`
- `e090_health_after_prewarm.json`
- `e090_before_metrics.prom`
- `e090_after_populate_metrics.prom`
- `e090_no_tool_choice_populate_response.json`
- `e090_no_tool_choice_populate_time.log`
- `e091_after_hit_metrics.prom`
- `e091_no_tool_auto_choice_hit_response.json`
- `e091_no_tool_auto_choice_hit_time.log`

Takeaway:

No-tool `auto`/`none` choices now reuse the same no-work replay path as omitted
tool choice. This removes prompt rendering, tokenization, prefix-cache lookup,
prefill, and decode for another common OpenAI-client no-op field while
preserving non-empty tool and non-no-op tool-choice semantics.

## Experiment E092-E093: Ignore Input Reasoning Content in Prompt Cache Keys

Hypothesis:

The API accepts `reasoning_content` on input messages because the same `Message`
type is used for request and response shapes. The current prompt renderer does
not pass `reasoning_content` into `ChatMessage`; it only renders role, content,
tool calls, name, and tool_call_id. Therefore, input `reasoning_content` is a
no-op for the prompt and should not split rendered-prompt or request-level chat
cache entries.

Change:

- Added `ChatPromptMessageCacheKey`, based only on fields propagated to
  `message_to_chat`: `role`, `content`, `tool_calls`, `name`, and
  `tool_call_id`.
- Switched both rendered-prompt cache keys and deterministic chat request-cache
  keys from full API `Message` serialization to this renderer-aligned message
  key.
- Kept propagated fields distinct; e.g. `name` still splits cache entries.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_ignores_input_reasoning_content --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server input_reasoning_content_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 24 passed.
- `cargo test -p kiln-server tools_ --lib`
  - Result: 4 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11229 ms. E092 populated a greedy chat request. E093 repeated the same
message with input `reasoning_content` set and a different seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E092 no-input-reasoning populate | chat, greedy, `max_tokens=2` | 2.80 s | 2792.14 ms | 2.574 s | 0.215 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E093 input-reasoning request-cache hit | equivalent chat with input `reasoning_content` | 0.01 s | 0.077 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=18`, `completion_tokens=2`,
`total_tokens=20`, with `reasoning_content="Okay,"`. Metrics after E093 showed:

- Prefill count/sum stayed at 1 / 2.574026 s.
- Decode count/sum stayed at 1 / 0.214883 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.791939 s to 2.791963 s.

Artifacts:

- `e092_server.log`
- `e092_health_after_prewarm.json`
- `e092_before_metrics.prom`
- `e092_after_populate_metrics.prom`
- `e092_no_input_reasoning_populate_response.json`
- `e092_no_input_reasoning_populate_time.log`
- `e093_after_hit_metrics.prom`
- `e093_input_reasoning_hit_response.json`
- `e093_input_reasoning_hit_time.log`

Takeaway:

Request and rendered-prompt caches now key on the actual prompt-rendered
message fields. Input `reasoning_content`, which is ignored by the current
renderer, no longer forces prompt rendering, tokenization, prefix-cache lookup,
prefill, or decode.

## Experiment E094-E095: Normalize Stop Sequence Sets in Deterministic Cache Keys

Hypothesis:

For deterministic replay, duplicate stop strings and stop-list order should not
split cache entries. The generator checks whether any stop sequence is present
in the generated text, and the server exposes only the OpenAI-compatible
`"stop"` finish reason, not the matched stop string. Therefore, `["omega",
"alpha", "omega"]` and `["alpha", "omega"]` are equivalent cache keys. An empty
stop string is a dominating member because every string contains it, so any stop
list containing `""` normalizes to `[""]`.

Change:

- Added `normalized_stop_for_cache`.
- Switched deterministic completion cache keys to store normalized stop lists.
- Switched normalized chat-request and whole-batch sampling keys to use the same
  stop normalization for greedy and seeded sampled requests.
- Kept `max_tokens=0` normalization unchanged: stop still normalizes away with
  other generation-only sampling fields when no token can be generated.

Verification:

- `cargo test -p kiln-server deterministic_cache_keys_normalize_stop_sequence_sets --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 24 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 26 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 12193 ms. E094 populated a greedy chat request with duplicate/reordered stop
sequences. E095 repeated the same prompt with an equivalent stop set and a
different seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E094 stop-set populate | chat, greedy, `max_tokens=2`, `stop=["omega","alpha","omega"]` | 6.98 s | 6968.80 ms | 3.619 s | 3.347 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E095 equivalent stop-set request-cache hit | same chat, `stop=["alpha","omega"]` | 0.01 s | 0.0616 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=18`, `completion_tokens=2`,
`total_tokens=20`, with `reasoning_content="Okay,"`. Metrics after E095 showed:

- Prefill count/sum stayed at 1 / 3.618894 s.
- Decode count/sum stayed at 1 / 3.347171 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 6.968643 s to 6.968658 s.

Artifacts:

- `e094_server.log`
- `e094_health_after_prewarm.json`
- `e094_before_metrics.prom`
- `e094_after_populate_metrics.prom`
- `e094_stop_set_populate_response.json`
- `e094_stop_set_populate_time.log`
- `e095_after_hit_metrics.prom`
- `e095_stop_set_hit_response.json`
- `e095_stop_set_hit_time.log`

Takeaway:

Equivalent stop-sequence sets now share deterministic replay entries. A
reordered/deduplicated stop list no longer forces prompt rendering,
tokenization, prefix-cache lookup, prefill, or decode.

## Experiment E096-E097: Normalize Empty Per-Message Tool Calls

Hypothesis:

OpenAI assistant messages normally omit `tool_calls` when no tools were called,
but some clients may send `tool_calls: []`. Qwen3.5's template branches on
`message.tool_calls`, so an empty list renders like an omitted field. Treating
empty per-message tool-call lists as omitted should let equivalent multi-turn
requests share rendered-prompt and request-level deterministic cache entries
while preserving non-empty tool calls.

Change:

- Added normalization for per-message `tool_calls` in `message_cache_keys`.
- Updated `message_to_chat` to omit empty `tool_calls` before template
  rendering.
- Kept non-empty `tool_calls` distinct and still forwarded to the template.

Verification:

- `cargo test -p kiln-server message_to_chat_omits_empty_tool_calls --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_request_cache_key_normalizes_empty_message_tool_calls --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server empty_message_tool_calls_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 27 passed.
- `cargo test -p kiln-server tools_ --lib`
  - Result: 4 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 9952 ms. E096 populated a deterministic multi-turn chat request with an
assistant message that omitted `tool_calls`. E097 repeated the same request with
`tool_calls: []` on that assistant message and a different seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E096 no-empty-message-tool-calls populate | multi-turn chat, greedy, `max_tokens=2` | 2.80 s | 2789.20 ms | 2.564 s | 0.220 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E097 empty-message-tool-calls request-cache hit | same chat with assistant `tool_calls: []` | 0.01 s | 0.0678 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=36`, `completion_tokens=2`,
`total_tokens=38`, with `reasoning_content="Thinking Process"`. Metrics after
E097 showed:

- Prefill count/sum stayed at 1 / 2.564419 s.
- Decode count/sum stayed at 1 / 0.220431 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.789069 s to 2.789089 s.

Artifacts:

- `e096_server.log`
- `e096_health_after_prewarm.json`
- `e096_before_metrics.prom`
- `e096_after_populate_metrics.prom`
- `e096_no_empty_message_tool_calls_populate_response.json`
- `e096_no_empty_message_tool_calls_populate_time.log`
- `e097_after_hit_metrics.prom`
- `e097_empty_message_tool_calls_hit_response.json`
- `e097_empty_message_tool_calls_hit_time.log`

Takeaway:

Empty per-message `tool_calls` no longer split equivalent deterministic chat
requests. That removes prompt rendering, tokenization, prefix-cache lookup,
prefill, and decode for another client-side no-op while preserving real tool-call
turns.

## Experiment E098-E099: Canonicalize Tool-Call Argument JSON in Cache Keys

Hypothesis:

The tokenizer already parses OpenAI wire-format
`tool_calls[*].function.arguments` JSON strings into structured
`serde_json::Value` before applying HF chat templates. Therefore compact JSON
argument strings, whitespace/reordered JSON argument strings, and already
structured argument objects render the same prompt. Cache keys should reflect
that rendered shape so equivalent tool-call histories do not force prompt
rendering, tokenization, prefix-cache lookup, prefill, or decode.

Change:

- Added cache-key-only canonicalization for `arguments` inside assistant
  `tool_calls`, including both OpenAI's nested `function.arguments` envelope and
  flatter `{name, arguments}` tool-call objects.
- The canonicalizer parses only valid JSON strings; non-JSON strings remain
  distinct.
- Left `message_to_chat` raw argument forwarding intact so tokenizer fallback
  behavior for templates that require raw argument strings remains available.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_normalizes_tool_call_argument_json_strings --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server tool_call_argument_json_string_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 29 passed.
- `cargo test -p kiln-server tools_ --lib`
  - Result: 4 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11743 ms. E098 populated a deterministic multi-turn tool-call chat request
with compact JSON in `function.arguments`. E099 repeated the same request with
whitespace and reordered keys in the JSON argument string, plus a different
seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E098 compact tool-call arguments populate | multi-turn tool-call chat, greedy, `max_tokens=2` | 2.98 s | 2970.55 ms | 2.739 s | 0.227 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E099 JSON-equivalent tool-call arguments hit | same chat with reordered/whitespace argument JSON | 0.01 s | 0.0922 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=79`, `completion_tokens=2`,
`total_tokens=81`, with `reasoning_content="Thinking Process"`. Metrics after
E099 showed:

- Prefill count/sum stayed at 1 / 2.739325 s.
- Decode count/sum stayed at 1 / 0.227089 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.970367 s to 2.970396 s.

Artifacts:

- `e098_server.log`
- `e098_health_after_prewarm.json`
- `e098_before_metrics.prom`
- `e098_after_populate_metrics.prom`
- `e098_tool_call_args_compact_populate_response.json`
- `e098_tool_call_args_compact_populate_time.log`
- `e099_after_hit_metrics.prom`
- `e099_tool_call_args_whitespace_hit_response.json`
- `e099_tool_call_args_whitespace_hit_time.log`

Takeaway:

Tool-call histories with JSON-equivalent argument strings now share
deterministic request-cache entries. This removes all prompt/model work for
another common OpenAI-client serialization difference without changing
non-JSON argument-string behavior.

## Experiment E100-E101: Lock Text Content Parts as Request-Cache No-Ops

Hypothesis:

Many OpenAI-compatible clients send user text as content parts:
`[{"type":"text","text":"..."}]`. Kiln's request deserializer already
concatenates text parts into the same `Message.content` string used by plain
string content. That means a plain-text request and an equivalent text-parts
request should share request-level deterministic cache entries and return before
prompt rendering/tokenization/model work on the second request.

Change:

- No production behavior change; added cache-level tests to lock the existing
  deserializer normalization as a request-cache no-op.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_normalizes_text_content_parts --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server text_content_parts_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server content_ --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 31 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 10547 ms. E100 populated a deterministic chat request with plain string
content. E101 repeated the same prompt as two OpenAI text content parts, with a
different seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E100 plain content populate | chat, greedy, `max_tokens=2` | 0.65 s | 639.07 ms | 0.453 s | 0.181 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E101 equivalent text-parts request-cache hit | same chat as `content` text parts | 0.01 s | 0.0953 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported `prompt_tokens=18`, `completion_tokens=2`,
`total_tokens=20`, with `reasoning_content="Okay,"`. Metrics after E101 showed:

- Prefill count/sum stayed at 1 / 0.453417 s.
- Decode count/sum stayed at 1 / 0.180619 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 0.638887 s to 0.638913 s.

Artifacts:

- `e100_server.log`
- `e100_health_after_prewarm.json`
- `e100_before_metrics.prom`
- `e100_after_populate_metrics.prom`
- `e100_plain_content_populate_response.json`
- `e100_plain_content_populate_time.log`
- `e101_after_hit_metrics.prom`
- `e101_text_parts_hit_response.json`
- `e101_text_parts_hit_time.log`

Takeaway:

Equivalent OpenAI text content parts are now explicitly covered as a
request-cache no-op. The existing deserializer path avoids prompt rendering,
tokenization, prefix-cache lookup, prefill, and decode for text-part requests
that match already-cached plain-text prompts.

## Experiment E102-E103: Lock Text Content Parts as Batch-Cache No-Ops

Hypothesis:

E100-E101 proved that OpenAI text content parts are a bs=1 chat request-cache
no-op. The batch endpoint uses the same `Message` deserializer and its
deterministic whole-batch cache key is built from deserialized role/content
pairs. Therefore a plain-text batch and an equivalent text-parts batch should
share whole-batch deterministic cache entries, preserving `n>1` aggregate usage
while skipping prompt rendering, tokenization, prefix lookup, prefill, and decode
on the equivalent second request.

Change:

- No production behavior change; added batch-cache tests to lock the existing
  text-part deserializer normalization for bs>1 batch requests.

Verification:

- `cargo test -p kiln-server deterministic_batch_cache_key_normalizes_text_content_parts --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_text_content_parts_hits_batch_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 28 passed.
- `cargo test -p kiln-server content_ --lib`
  - Result: 10 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 5799 ms. E102 populated a greedy `n=4` batch using plain string content.
E103 repeated the same prompt as OpenAI text content parts, with a different
seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E102 plain-content batch populate | batch, one prompt, `n=4`, greedy, `max_tokens=2` | 0.62 s | 607.44 ms | 0.426 s | 0.176 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E103 equivalent text-parts batch-cache hit | same batch as text content parts | 0.01 s | 0.0696 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported four logical completions with aggregate
`prompt_tokens=76`, `completion_tokens=8`, and `total_tokens=84`. Metrics after
E103 showed:

- Prefill count/sum stayed at 1 / 0.426149 s.
- Decode count/sum stayed at 1 / 0.176055 s.
- `kiln_tokens_generated_total` stayed at 2, confirming the `n=4` populate used
  one physical greedy decode and the hit used none.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 0.607284 s to 0.607302 s.

Artifacts:

- `e102_server.log`
- `e102_health_after_prewarm.json`
- `e102_before_metrics.prom`
- `e102_after_populate_metrics.prom`
- `e102_batch_plain_content_populate_response.json`
- `e102_batch_plain_content_populate_time.log`
- `e103_after_hit_metrics.prom`
- `e103_batch_text_parts_hit_response.json`
- `e103_batch_text_parts_hit_time.log`

Takeaway:

Equivalent OpenAI text content parts are now explicitly covered as a whole-batch
cache no-op too. This extends the same "remove the need to do it" behavior from
bs=1 chat into an `n=4` batching scenario.

## Experiment E104-E105: Lock Unrendered Batch Message Metadata as Cache No-Ops

Hypothesis:

The batch endpoint currently renders only role/content turns through
`batch_synth_messages`; unrendered message metadata such as input
`reasoning_content` and empty per-message `tool_calls: []` does not affect the
prompt sent to the tokenizer/model. The whole-batch deterministic cache key
should therefore treat that metadata as a no-op for batch requests, preserving
`n>1` aggregate usage while skipping prompt rendering, tokenization, prefix
lookup, prefill, and decode on an equivalent second request.

Change:

- No production behavior change; added batch-cache tests to lock the existing
  role/content-only batch cache-key behavior for unrendered message metadata.

Verification:

- `cargo test -p kiln-server deterministic_batch_cache_key_ignores_unrendered_message_metadata --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_unrendered_message_metadata_hits_batch_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 30 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 31 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11280 ms. E104 populated a greedy `n=4` batch using plain role/content
messages. E105 repeated the same batch with input `reasoning_content` on the
user message and empty `tool_calls: []` on the assistant message, using a
different seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E104 plain metadata-free batch populate | batch, one 3-turn prompt, `n=4`, greedy, `max_tokens=2` | 2.476 s | 2475.54 ms | 2.258 s | 0.214 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E105 equivalent metadata-bearing batch-cache hit | same batch plus unrendered message metadata | 0.000752 s | 0.0864 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

Both responses reported four logical completions with aggregate
`prompt_tokens=132`, `completion_tokens=8`, and `total_tokens=140`. Metrics
after E105 showed:

- Prefill count/sum stayed at 1 / 2.257764 s.
- Decode count/sum stayed at 1 / 0.214292 s.
- `kiln_tokens_generated_total` stayed at 2, confirming the `n=4` populate used
  one physical greedy decode and the hit used none.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.475305 s to 2.475328 s.

Artifacts:

- `e104_server.log`
- `e104_health_after_prewarm.json`
- `e104_before_metrics.prom`
- `e104_after_populate_metrics.prom`
- `e104_batch_plain_metadata_populate_response.json`
- `e104_batch_plain_metadata_populate_time.log`
- `e105_after_hit_metrics.prom`
- `e105_batch_metadata_hit_response.json`
- `e105_batch_metadata_hit_time.log`

Takeaway:

Unrendered per-message metadata is now explicitly covered as a whole-batch
cache no-op for the current batch renderer. This keeps the fast path aligned
with the actual model input and extends the no-work batch replay coverage to
another OpenAI-compatible request shape.

## Experiment E106-E109: Lock Non-Text Content Parts as Cache No-Ops

Hypothesis:

Kiln is text-only. `deserialize_optional_content` already concatenates
OpenAI-style text parts and ignores non-text content parts such as `image_url`
or `input_audio`. If the visible text is identical, chat request-cache keys and
whole-batch cache keys should be identical too, allowing multimodal-shaped
client payloads to hit before prompt rendering, tokenization, prefix lookup,
prefill, and decode.

Change:

- No production behavior change; added chat and batch cache tests to lock the
  existing text-only content-part deserialization behavior.

Verification:

- `cargo test -p kiln-server deterministic_chat_request_cache_key_ignores_non_text_content_parts --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_batch_cache_key_ignores_non_text_content_parts --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server non_text_content_parts_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_non_text_content_parts_hits_batch_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server content_ --lib`
  - Result: 14 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 33 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 32 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11511 ms. E106 populated a deterministic bs=1 chat request with plain text.
E107 repeated the same visible text as an OpenAI content-parts array containing
ignored `image_url` and `input_audio` parts. E108 then populated a greedy `n=4`
batch with plain text, and E109 repeated the same visible text as non-text
content parts.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E106 plain chat populate | chat, greedy, `max_tokens=2` | 5.695 s | 5694.61 ms | 2.743 s | 2.949 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E107 equivalent non-text content-parts chat hit | same visible chat text plus ignored parts | 0.000735 s | 0.0960 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E108 plain batch populate | batch, one prompt, `n=4`, greedy, `max_tokens=2` | 2.928 s | 2927.57 ms | +2.735 s | +0.190 s | +2 | one additional render/token/prefix miss |
| E109 equivalent non-text content-parts batch hit | same visible batch text plus ignored parts | 0.000837 s | 0.0958 ms | unchanged | unchanged | unchanged at 4 | render/token/prefix counters unchanged |

E106/E107 both reported `prompt_tokens=20`, `completion_tokens=2`, and
`total_tokens=22`, with `reasoning_content="Okay,"`. Metrics after E107 showed:

- Prefill count/sum stayed at 1 / 2.743007 s.
- Decode count/sum stayed at 1 / 2.948544 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 5.694390 s to 5.694413 s.

E108/E109 both reported four logical completions with aggregate
`prompt_tokens=84`, `completion_tokens=8`, and `total_tokens=92`. Relative to
the pre-batch snapshot after E107, metrics after E109 showed:

- Prefill count/sum stayed at 2 / 5.477805 s after the hit.
- Decode count/sum stayed at 2 / 3.138173 s after the hit.
- `kiln_tokens_generated_total` stayed at 4 after the hit, confirming E108 used
  one physical greedy decode and E109 used none.
- Rendered-prompt cache stayed hit 0 / miss 2.
- Prompt-token cache stayed hit 0 / miss 2.
- Prefix-cache lookups stayed hit 0 / miss 2.
- Request-duration sum increased only from 8.621918 s to 8.621944 s.

Artifacts:

- `e106_server.log`
- `e106_health_after_prewarm.json`
- `e106_before_metrics.prom`
- `e106_after_populate_metrics.prom`
- `e106_chat_plain_non_text_populate_response.json`
- `e106_chat_plain_non_text_populate_time.log`
- `e107_after_hit_metrics.prom`
- `e107_chat_non_text_parts_hit_response.json`
- `e107_chat_non_text_parts_hit_time.log`
- `e108_before_metrics.prom`
- `e108_after_populate_metrics.prom`
- `e108_batch_plain_non_text_populate_response.json`
- `e108_batch_plain_non_text_populate_time.log`
- `e109_after_hit_metrics.prom`
- `e109_batch_non_text_parts_hit_response.json`
- `e109_batch_non_text_parts_hit_time.log`

Takeaway:

Multimodal-shaped OpenAI content arrays now have explicit cache coverage for
Kiln's text-only behavior. If the visible text is the same, ignored non-text
parts do not force repeated prompt rendering, tokenization, prefix lookup, or
model execution in either bs=1 chat or `n=4` batch scenarios.

## Experiment E110-E113: Support `max_completion_tokens` as a No-Work Alias

Hypothesis:

Newer OpenAI-compatible clients often send `max_completion_tokens` instead of
`max_tokens`. Before this experiment Kiln ignored that field, which meant those
requests fell back to the 2048-token default and could do orders of magnitude
more decode work than the caller intended. Treating `max_completion_tokens` as
an alias for `max_tokens` removes that accidental work; `max_tokens` still wins
when both are present to preserve existing request behavior.

Change:

- Added `max_completion_tokens` to chat and batch request structs.
- Resolved generation length with `max_tokens.or(max_completion_tokens).unwrap_or(2048)`.
- Threaded the alias into synthesized per-output chat requests used by the
  batch endpoint.
- Added chat and batch cache-key/cache-hit tests, plus zero-token alias tests
  to prove the alias reaches control flow that skips generation entirely.

Verification:

- `cargo test -p kiln-server max_completion_tokens --lib`
  - Result: 7 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 36 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 35 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 12358 ms. E110 populated a deterministic bs=1 chat request with
`max_tokens=2`. E111 repeated the same request using only
`max_completion_tokens=2`. E112 then populated a greedy `n=4` batch with
`max_tokens=2`, and E113 repeated the same batch using only
`max_completion_tokens=2`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E110 `max_tokens` chat populate | chat, greedy, 2 output tokens | 4.187 s | 4186.48 ms | 3.964 s | 0.219 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E111 `max_completion_tokens` chat hit | same chat using alias only | 0.000877 s | 0.0647 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E112 `max_tokens` batch populate | batch, one prompt, `n=4`, greedy, 2 output tokens | 2.855 s | 2854.32 ms | +2.628 s | +0.223 s | +2 | one additional render/token/prefix miss |
| E113 `max_completion_tokens` batch hit | same batch using alias only | 0.000839 s | 0.0955 ms | unchanged | unchanged | unchanged at 4 | render/token/prefix counters unchanged |

E110/E111 both reported `prompt_tokens=19`, `completion_tokens=2`, and
`total_tokens=21`, with `reasoning_content="Okay,"`. Metrics after E111 showed:

- Prefill count/sum stayed at 1 / 3.963667 s.
- Decode count/sum stayed at 1 / 0.218671 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 4.186265 s to 4.186284 s.

E112/E113 both reported four logical completions with aggregate
`prompt_tokens=80`, `completion_tokens=8`, and `total_tokens=88`. Relative to
the pre-batch snapshot after E111, metrics after E113 showed:

- Prefill count/sum stayed at 2 / 6.592075 s after the hit.
- Decode count/sum stayed at 2 / 0.441365 s after the hit.
- `kiln_tokens_generated_total` stayed at 4 after the hit, confirming E112 used
  one physical greedy decode and E113 used none.
- Rendered-prompt cache stayed hit 0 / miss 2.
- Prompt-token cache stayed hit 0 / miss 2.
- Prefix-cache lookups stayed hit 0 / miss 2.
- Request-duration sum increased only from 7.040541 s to 7.040578 s.

Artifacts:

- `e110_server.log`
- `e110_health_after_prewarm.json`
- `e110_before_metrics.prom`
- `e110_after_populate_metrics.prom`
- `e110_chat_max_tokens_populate_response.json`
- `e110_chat_max_tokens_populate_time.log`
- `e111_after_hit_metrics.prom`
- `e111_chat_max_completion_tokens_hit_response.json`
- `e111_chat_max_completion_tokens_hit_time.log`
- `e112_before_metrics.prom`
- `e112_after_populate_metrics.prom`
- `e112_batch_max_tokens_populate_response.json`
- `e112_batch_max_tokens_populate_time.log`
- `e113_after_hit_metrics.prom`
- `e113_batch_max_completion_tokens_hit_response.json`
- `e113_batch_max_completion_tokens_hit_time.log`

Takeaway:

OpenAI clients that use `max_completion_tokens` now get the same short decode
and deterministic no-work cache behavior as clients that use `max_tokens`.
This removes a compatibility trap where an intended two-token request could
silently become a 2048-token decode.

## Experiment E114-E117: Support Single-String `stop` as a No-Work Alias

Hypothesis:

OpenAI-compatible clients may send `stop` as either a single string or an array
of strings. Kiln previously modeled only the array form, so a single-string stop
request failed at JSON deserialization instead of sharing the same short decode
and deterministic cache entries as `["that-string"]`. Accepting the string form
as a one-item list removes that compatibility failure while keeping the sampler
and cache-key logic unchanged.

Change:

- Added `deserialize_optional_stop`, accepting `stop` as a string, array,
  `null`, or missing.
- Applied it to chat and batch requests.
- Added parser, cache-key, and cache-hit tests proving single-string stop is
  equivalent to a one-item stop list for bs=1 chat and `n>1` batch.

Verification:

- `cargo test -p kiln-server stop --lib`
  - Result: 4 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 37 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 36 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11701 ms. E114 populated a deterministic bs=1 chat request with
`stop=["never-match-stop"]`. E115 repeated the same request with
`stop="never-match-stop"`. E116 then populated a greedy `n=4` batch with the
one-item stop list, and E117 repeated the same batch with the single-string
stop.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E114 one-item stop-list chat populate | chat, greedy, `max_tokens=2` | 2.722 s | 2721.04 ms | 2.523 s | 0.194 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E115 equivalent stop-string chat hit | same chat using single-string stop | 0.001001 s | 0.0696 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E116 one-item stop-list batch populate | batch, one prompt, `n=4`, greedy, `max_tokens=2` | 2.810 s | 2809.37 ms | +2.619 s | +0.187 s | +2 | one additional render/token/prefix miss |
| E117 equivalent stop-string batch hit | same batch using single-string stop | 0.000608 s | 0.0630 ms | unchanged | unchanged | unchanged at 4 | render/token/prefix counters unchanged |

E114/E115 both reported `prompt_tokens=17`, `completion_tokens=2`, and
`total_tokens=19`, with `reasoning_content="The user"`. Metrics after E115
showed:

- Prefill count/sum stayed at 1 / 2.522894 s.
- Decode count/sum stayed at 1 / 0.194333 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.720808 s to 2.720828 s.

E116/E117 both reported four logical completions with aggregate
`prompt_tokens=72`, `completion_tokens=8`, and `total_tokens=80`. Relative to
the pre-batch snapshot after E115, metrics after E117 showed:

- Prefill count/sum stayed at 2 / 5.142155 s after the hit.
- Decode count/sum stayed at 2 / 0.380852 s after the hit.
- `kiln_tokens_generated_total` stayed at 4 after the hit, confirming E116 used
  one physical greedy decode and E117 used none.
- Rendered-prompt cache stayed hit 0 / miss 2.
- Prompt-token cache stayed hit 0 / miss 2.
- Prefix-cache lookups stayed hit 0 / miss 2.
- Request-duration sum increased only from 5.530132 s to 5.530151 s.

Artifacts:

- `e114_server.log`
- `e114_health_after_prewarm.json`
- `e114_before_metrics.prom`
- `e114_after_populate_metrics.prom`
- `e114_chat_stop_list_populate_response.json`
- `e114_chat_stop_list_populate_time.log`
- `e115_after_hit_metrics.prom`
- `e115_chat_stop_string_hit_response.json`
- `e115_chat_stop_string_hit_time.log`
- `e116_before_metrics.prom`
- `e116_after_populate_metrics.prom`
- `e116_batch_stop_list_populate_response.json`
- `e116_batch_stop_list_populate_time.log`
- `e117_after_hit_metrics.prom`
- `e117_batch_stop_string_hit_response.json`
- `e117_batch_stop_string_hit_time.log`

Takeaway:

Single-string `stop` is now a compatibility-preserving alias for a one-item stop
list. Clients using either OpenAI shape get the same deterministic no-work cache
behavior, and the string form no longer fails before inference.

## Experiment E118-E121: Lock Default OpenAI Option Fields as No-Ops

Hypothesis:

Several OpenAI-compatible clients send default-valued option fields that Kiln
does not currently implement: `response_format={"type":"text"}`,
`parallel_tool_calls=true`, `store=false`, `service_tier="auto"`,
`logprobs=false`, `top_logprobs=0`, zero frequency/presence penalties,
`stream_options={"include_usage":false}`, and client-only `user`/`metadata`.
Those defaults should not alter generated text or deterministic cache identity.
Locking the existing ignored-default behavior prevents compatibility fields
from accidentally forcing prompt rendering, tokenization, prefill, or decode.

Change:

- Added deterministic chat request-cache tests proving default OpenAI option
  fields are cache-key no-ops.
- Added deterministic whole-batch cache-key tests proving the same default
  fields are no-ops for `n>1` batch requests.
- Added real handler cache-hit tests proving the default-field request hits the
  existing no-work path before prompt rendering/tokenization for both bs=1 chat
  and `n>1` batch.
- No production behavior change was required; this locks the current
  compatibility behavior in tests.

Verification:

- `cargo test -p kiln-server default_openai --lib`
  - Result: 4 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 39 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 38 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11650 ms. E118 populated a deterministic bs=1 chat request without the
extra default OpenAI option fields. E119 repeated the same chat request with
those default fields and a different seed. E120 then populated a greedy `n=4`
batch without the extra defaults, and E121 repeated the same batch with the
default fields and a different seed.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E118 plain chat populate | chat, greedy, `max_tokens=2` | 2.830 s | 2828.91 ms | 2.614 s | 0.211 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E119 default-option chat hit | same chat plus ignored default fields | 0.000772 s | 0.0290 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E120 plain batch populate | batch, one prompt, `n=4`, greedy, `max_tokens=2` | 0.587 s | 585.64 ms | +0.409 s | +0.174 s | +2 | one additional render/token/prefix miss |
| E121 default-option batch hit | same batch plus ignored default fields | 0.000684 s | 0.0190 ms | unchanged | unchanged | unchanged at 4 | render/token/prefix counters unchanged |

E118/E119 both reported `prompt_tokens=21`, `completion_tokens=2`, and
`total_tokens=23`, with `reasoning_content="Okay,"`. Metrics after E119
showed:

- Prefill count/sum stayed at 1 / 2.614398 s.
- Decode count/sum stayed at 1 / 0.210868 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 2.828906 s to 2.828935 s.

E120/E121 both reported four logical completions with aggregate
`prompt_tokens=88`, `completion_tokens=8`, and `total_tokens=96`. Relative to
the pre-batch snapshot after E119, metrics after E121 showed:

- Prefill count/sum stayed at 2 / 3.023171 s after the hit.
- Decode count/sum stayed at 2 / 0.384537 s after the hit.
- `kiln_tokens_generated_total` stayed at 4 after the hit, confirming E120 used
  one physical greedy decode and E121 used none.
- Rendered-prompt cache stayed hit 0 / miss 2.
- Prompt-token cache stayed hit 0 / miss 2.
- Prefix-cache lookups stayed hit 0 / miss 2.
- Request-duration sum increased only from 3.414570 s to 3.414589 s.

Artifacts:

- `e118_server.log`
- `e118_health_after_prewarm.json`
- `e118_before_metrics.prom`
- `e118_after_populate_metrics.prom`
- `e118_chat_plain_default_options_populate_response.json`
- `e118_chat_plain_default_options_populate_time.log`
- `e119_after_hit_metrics.prom`
- `e119_chat_default_options_hit_response.json`
- `e119_chat_default_options_hit_time.log`
- `e120_before_metrics.prom`
- `e120_after_populate_metrics.prom`
- `e120_batch_plain_default_options_populate_response.json`
- `e120_batch_plain_default_options_populate_time.log`
- `e121_after_hit_metrics.prom`
- `e121_batch_default_options_hit_response.json`
- `e121_batch_default_options_hit_time.log`

Takeaway:

Default-valued OpenAI option fields are now covered by bs=1 and `n>1` no-work
regression tests. The real server path preserves the existing deterministic
responses while avoiding prompt render, tokenization, prefix-cache lookup,
prefill, decode, and physical token generation on equivalent default-field
requests.

## Experiment E122-E123: Support Chat `n>1` with Greedy Single-Decode Clone

Hypothesis:

OpenAI-compatible clients can request multiple chat choices with
`/v1/chat/completions` and `n>1`. Kiln previously ignored that field because it
was not modeled in `ChatCompletionRequest`, so a bs>1 chat request silently
returned one choice. Supporting non-streaming chat `n>1` through the existing
single-output fast path should make the OpenAI-compatible endpoint useful for
bs>1 while preserving the "remove the need to do it" behavior for greedy
requests: one physical decode can be cloned into all logical choices, and a
repeat can clone from the cached single-choice response before prompt work.

Change:

- Added `n` to `ChatCompletionRequest`.
- Added validation for `n=0`, excessive `n`, and currently unsupported
  `stream=true` with `n>1`.
- Added a non-streaming chat multi-choice path that synthesizes single-choice
  requests through `generate_one_response`.
- Greedy `n>1` chat now performs one physical generation and clones the result
  to every returned choice.
- Top-level multi-choice chat requests are intentionally excluded from the
  single-choice request-cache key; the synthesized single-choice request still
  uses the existing request/completion cache path.

Verification:

- `cargo test -p kiln-server chat_multi_choice --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server chat_rejects --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server deterministic_chat_request_cache_key_skips_multi_choice --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 44 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 38 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11906 ms. E122 populated a deterministic non-streaming chat request with
`n=4`. E123 repeated the same prompt with a different seed. Because the request
is greedy, the seed is normalized away for the synthesized single-choice cache
entry.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E122 chat `n=4` greedy populate | chat, one prompt, `n=4`, `max_tokens=2` | 3.546 s | 3544.67 ms | 3.312 s | 0.227 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E123 chat `n=4` greedy hit | same chat with different seed | 0.000682 s | 0.0280 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |

E122/E123 both returned four choices with indices 0-3, each reporting
`reasoning_content="Okay,"` and `finish_reason="length"`. Aggregate usage was
`prompt_tokens=21`, `completion_tokens=8`, and `total_tokens=29`, while
`kiln_tokens_generated_total` was only 2 after E122 and stayed 2 after E123.
Metrics after E123 showed:

- Prefill count/sum stayed at 1 / 3.311888 s.
- Decode count/sum stayed at 1 / 0.227363 s.
- `kiln_tokens_generated_total` stayed at 2.
- Rendered-prompt cache stayed hit 0 / miss 1.
- Prompt-token cache stayed hit 0 / miss 1.
- Prefix-cache lookups stayed hit 0 / miss 1.
- Request-duration sum increased only from 3.544671 s to 3.544699 s.

Artifacts:

- `e122_server.log`
- `e122_health_after_prewarm.json`
- `e122_before_metrics.prom`
- `e122_after_populate_metrics.prom`
- `e122_chat_n4_greedy_populate_response.json`
- `e122_chat_n4_greedy_populate_time.log`
- `e123_after_hit_metrics.prom`
- `e123_chat_n4_greedy_hit_response.json`
- `e123_chat_n4_greedy_hit_time.log`

Takeaway:

The OpenAI-compatible chat endpoint now has a real bs>1 non-streaming path.
For greedy `n>1`, Kiln returns all requested logical choices while doing one
physical decode on the populate request and no prompt/render/token/prefix/model
work on the equivalent repeat.

## Experiment E124-E125: Top-Level Replay for Seeded Sampled Chat `n>1`

Hypothesis:

After E122-E123, non-streaming chat `n>1` requests use synthesized
single-choice requests. That is enough to avoid model work on greedy repeats,
but deterministic seeded sampled repeats still had to loop through each
synthesized choice and perform one request-cache lookup per choice. A top-level
multi-choice replay cache can remove that per-choice replay work entirely:
identical seeded sampled `n>1` requests should return from one cache lookup
before prompt rendering, tokenization, prefix-cache lookup, prefill, or decode.

Change:

- Added a bounded deterministic chat-choices cache for replayable
  non-streaming chat `n>1` requests.
- Added a top-level multi-choice cache key that includes rendered prompt
  inputs (`messages`, `tools`, `tool_choice`), `n`, and normalized replayable
  sampling fields.
- Greedy chat `n>1` keys normalize away seed/top-p/top-k; seeded sampled chat
  `n>1` keys retain the base seed. Unseeded sampled requests remain uncached.
- Added response reconstruction for cached multi-choice chat payloads, including
  reasoning/content channels and aggregate usage.
- Added tests covering greedy and seeded sampled top-level cache hits before
  prompt work.

Verification:

- `cargo test -p kiln-server chat_multi_choice --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server deterministic_chat_choices_cache_key --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 46 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 38 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 9451 ms. E124 populated a deterministic seeded sampled chat request with
`n=4`, `temperature=0.7`, `top_p=0.9`, `max_tokens=2`, and `seed=4242`. E125
repeated the exact same request.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E124 sampled chat `n=4` populate | chat, one prompt, `n=4`, `max_tokens=2` | 3.233 s | 3231.87 ms | 0.00698 s across 4 choices | 3.214 s across 4 choices | 8 | render miss 1/hit 3, token miss 1/hit 3, prefix miss 1/hit 3 |
| E125 top-level chat `n=4` hit | identical seeded sampled request | 0.000777 s | 0.0270 ms | unchanged | unchanged | unchanged at 8 | render/token/prefix counters unchanged |

E124/E125 both returned four choices with aggregate `prompt_tokens=23`,
`completion_tokens=8`, and `total_tokens=31`. The first two choices reported
`reasoning_content="Okay,"`; the last two reported
`reasoning_content="OkayOkay"`. Metrics after E125 showed:

- Prefill count/sum stayed at 4 / 0.006978 s.
- Decode count/sum stayed at 4 / 3.213619 s.
- `kiln_tokens_generated_total` stayed at 8.
- Rendered-prompt cache stayed hit 3 / miss 1.
- Prompt-token cache stayed hit 3 / miss 1.
- Prefix-cache lookups stayed hit 3 / miss 1.
- Request-duration sum increased only from 3.231868 s to 3.231895 s.

Artifacts:

- `e124_server.log`
- `e124_health_after_prewarm.json`
- `e124_before_metrics.prom`
- `e124_after_populate_metrics.prom`
- `e124_chat_n4_seeded_sampled_populate_response.json`
- `e124_chat_n4_seeded_sampled_populate_time.log`
- `e125_after_hit_metrics.prom`
- `e125_chat_n4_seeded_sampled_hit_response.json`
- `e125_chat_n4_seeded_sampled_hit_time.log`

Takeaway:

Replayable chat `n>1` now has a top-level no-work cache for both greedy and
seeded sampled requests. The seeded sampled repeat path avoids every
per-choice synthesized request and returns as one cached multi-choice response.

## Experiment E126-E127: Reuse Chat Choices Cache for Equivalent One-Prompt Batch

Hypothesis:

E124-E125 added a top-level replay cache for non-streaming chat `n>1`, but the
single-prompt `/v1/completions/batch` endpoint still had to miss its own
top-level batch cache when the equivalent chat multi-choice request populated
first. Because a chat choices cache value preserves each choice's content,
finish reason, and exact per-choice completion-token count, a one-prompt batch
can safely reuse that cached value by projecting each chat choice into a batch
completion item. This is intentionally one-way for now: batch responses do not
carry the reasoning channel, so batch-to-chat top-level replay would lose data.

Change:

- Added exact internal per-choice `completion_tokens` on chat choices while
  keeping the serialized OpenAI response unchanged.
- Added a non-owning probe API for the deterministic chat choices cache.
- Added a single-prompt batch lookup against the equivalent chat choices cache
  before batch prompt work.
- On hit, the batch handler returns a batch-shaped response and seeds the
  deterministic batch cache for future identical batch requests.
- Added a regression test proving a seeded sampled chat `n>1` populate lets the
  equivalent one-prompt batch return before rendered-prompt/token lookups.

Verification:

- `cargo test -p kiln-server single_prompt_batch_hits_chat_choices_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_multi_choice --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server deterministic_chat_choices_cache_key --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 47 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 39 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 10838 ms. E126 populated a deterministic seeded sampled chat request with
`n=4`, `temperature=0.7`, `top_p=0.9`, `max_tokens=2`, and `seed=5151`. E127
then sent the equivalent one-prompt batch request to `/v1/completions/batch`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E126 sampled chat `n=4` populate | chat, one prompt, `n=4`, `max_tokens=2` | 3.486 s | 3484.39 ms | 0.00695 s across 4 choices | 3.466 s across 4 choices | 8 | render miss 1/hit 3, token miss 1/hit 3, prefix miss 1/hit 3 |
| E127 equivalent one-prompt batch hit | batch, one prompt, `n=4`, same sampling | 0.000863 s | 0.0670 ms | unchanged | unchanged | unchanged at 8 | render/token/prefix counters unchanged |

E126 returned four chat choices with aggregate `prompt_tokens=19`,
`completion_tokens=8`, and `total_tokens=27`. E127 returned four batch
completions with aggregate `prompt_tokens=76`, `completion_tokens=8`, and
`total_tokens=84`, matching the batch endpoint's convention of counting prompt
tokens once per returned completion. Metrics after E127 showed:

- Prefill count/sum stayed at 4 / 0.006954 s.
- Decode count/sum stayed at 4 / 3.466231 s.
- `kiln_tokens_generated_total` stayed at 8.
- Rendered-prompt cache stayed hit 3 / miss 1.
- Prompt-token cache stayed hit 3 / miss 1.
- Prefix-cache lookups stayed hit 3 / miss 1.
- Request-duration sum increased only from 3.484386 s to 3.484453 s.

Artifacts:

- `e126_server.log`
- `e126_health_after_prewarm.json`
- `e126_before_metrics.prom`
- `e126_after_populate_metrics.prom`
- `e126_chat_n4_seeded_sampled_populate_response.json`
- `e126_chat_n4_seeded_sampled_populate_time.log`
- `e127_after_hit_metrics.prom`
- `e127_batch_from_chat_choices_hit_response.json`
- `e127_batch_from_chat_choices_hit_time.log`

Takeaway:

Single-prompt batch now shares the top-level no-work replay layer populated by
chat `n>1`. This removes the remaining per-choice synthesized replay loop when
a chat multi-choice request is followed by the equivalent batch request.

## Experiment E128-E129: Seed Chat Choices Cache from Equivalent One-Prompt Batch

Hypothesis:

E126-E127 made one-prompt batch consume chat choices cache values, but the
opposite direction still needed care: public batch responses expose `text` but
not the reasoning channel, while chat responses need both `reasoning_content`
and `content` for Qwen3.5. If batch items preserve reasoning internally while
keeping the public batch JSON unchanged, a one-prompt batch populate can seed
the top-level chat choices cache. The equivalent chat `n>1` request should then
return before prompt rendering, tokenization, prefix-cache lookup, prefill, or
decode.

Change:

- Added internal, `serde(skip)` per-choice `completion_tokens` on chat choices
  so top-level replay keeps exact per-choice usage even when completions finish
  at different lengths.
- Added internal, `serde(skip)` `reasoning_content` on batch completion items.
- Extended deterministic batch cache items to preserve `reasoning_content`.
- On single-prompt batch populate, seed the equivalent deterministic chat
  choices cache.
- Added a regression test proving a seeded sampled one-prompt batch populate
  lets the equivalent chat `n>1` request hit before rendered-prompt/token
  lookups.

Verification:

- `cargo test -p kiln-server single_prompt_batch_populates_chat_choices_cache_before_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server single_prompt_batch_hits_chat_choices_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_multi_choice --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 48 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 40 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 10117 ms. E128 populated a deterministic seeded sampled one-prompt batch
request with `n=4`, `temperature=0.7`, `top_p=0.9`, `max_tokens=2`, and
`seed=6262`. E129 then sent the equivalent chat `n=4` request to
`/v1/chat/completions`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E128 sampled one-prompt batch populate | batch, one prompt, `n=4`, `max_tokens=2` | 4.993 s | 4991.41 ms | 0.00678 s across 4 choices | 4.975 s across 4 choices | 8 | render miss 1/hit 3, token miss 1/hit 3, prefix miss 1/hit 3 |
| E129 equivalent chat `n=4` hit | chat, one prompt, `n=4`, same sampling | 0.000783 s | 0.0480 ms | unchanged | unchanged | unchanged at 8 | render/token/prefix counters unchanged |

E128 returned four batch completions with aggregate `prompt_tokens=76`,
`completion_tokens=8`, and `total_tokens=84`. E129 returned four chat choices
with aggregate `prompt_tokens=19`, `completion_tokens=8`, and `total_tokens=27`;
the cached chat response preserved the internal reasoning channel from the
batch populate (`"Here's"`, `"Okay,"`, `"Here's"`, `"Okay,"`). Metrics after
E129 showed:

- Prefill count/sum stayed at 4 / 0.006779 s.
- Decode count/sum stayed at 4 / 4.974571 s.
- `kiln_tokens_generated_total` stayed at 8.
- Rendered-prompt cache stayed hit 3 / miss 1.
- Prompt-token cache stayed hit 3 / miss 1.
- Prefix-cache lookups stayed hit 3 / miss 1.
- Request-duration sum increased only from 4.991414 s to 4.991462 s.

Artifacts:

- `e128_server.log`
- `e128_health_after_prewarm.json`
- `e128_before_metrics.prom`
- `e128_after_populate_metrics.prom`
- `e128_batch_n4_seeded_sampled_populate_response.json`
- `e128_batch_n4_seeded_sampled_populate_time.log`
- `e129_after_hit_metrics.prom`
- `e129_chat_from_batch_choices_hit_response.json`
- `e129_chat_from_batch_choices_hit_time.log`

Takeaway:

Single-prompt batch and chat `n>1` now share the same top-level no-work replay
layer in both directions, without changing either public API shape. Batch
responses can seed the reasoning-preserving chat choices cache, and equivalent
chat requests skip every front-end and model step.

## Experiment E130-E131: Singleflight Concurrent Chat Choices Requests

Hypothesis:

The top-level deterministic chat choices cache should also protect concurrent
identical sampled chat `n>1` requests. The first request should own the physical
work; the second should wait on the in-flight value and return the same choices
without independently rendering prompts, tokenizing, probing prefix cache, or
running prefill/decode. This means two HTTP request durations are expected, but
only one `n=4` generation's model work should appear in the counters.

Change:

- Locked the chat choices cache owner/waiter path with a state-level
  singleflight regression test.
- Added a handler-level concurrency regression test proving two identical chat
  `n=4` requests coalesce before rendered-prompt work, prompt-token work, prefix
  cache lookup, prefill, and decode.

Verification:

- `cargo test -p kiln-server deterministic_chat_choices_cache_coalesces_in_flight_request --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server concurrent_chat_multi_choice_singleflights_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_multi_choice --lib`
  - Result: 4 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 50 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 40 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 10366 ms. E130 and E131 sent two identical concurrent sampled chat requests
to `/v1/chat/completions` with `n=4`, `temperature=0.7`, `top_p=0.9`,
`max_tokens=2`, `seed=7373`, and prompt
`"concurrent chat choices should singleflight as one top level owner"`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E130 concurrent owner/waiter A | chat, `n=4`, sampled, `max_tokens=2` | 3.914884 s | 3913.59 ms | shared total: 0.007390 s across 4 choices | shared total: 3.894330 s across 4 choices | shared total: 8 | shared render miss 1/hit 3, token miss 1/hit 3, prefix miss 1/hit 3 |
| E131 concurrent owner/waiter B | same request launched concurrently | 3.914904 s | 3913.64 ms | unchanged beyond shared total | unchanged beyond shared total | unchanged at 8 | unchanged beyond shared totals |

Both responses returned four choices with identical choices, finish reasons, and
usage, apart from generated response ids. The aggregate usage was
`prompt_tokens=22`, `completion_tokens=8`, and `total_tokens=30` for each HTTP
response. The four reasoning outputs were `"Here's"`, `"Here\n\n"`,
`"This is"`, and `"Here\n\n"`.

Metrics after both concurrent requests showed:

- Request-duration count/sum advanced to 2 / 7.826777 s, as expected for two
  HTTP requests.
- Prefill count/sum advanced only to 4 / 0.007390 s, not 8 choices.
- Decode count/sum advanced only to 4 / 3.894330 s, not 8 choices.
- `kiln_tokens_generated_total` advanced only to 8, not 16.
- Rendered-prompt cache advanced only to hit 3 / miss 1, not two independent
  request front-end passes.
- Prompt-token cache advanced only to hit 3 / miss 1.
- Prefix-cache lookups advanced only to hit 3 / miss 1.

Artifacts:

- `e130_server.log`
- `e130_health_after_prewarm.json`
- `e130_before_metrics.prom`
- `e130_chat_n4_concurrent_a_response.json`
- `e130_chat_n4_concurrent_a_time.log`
- `e131_chat_n4_concurrent_b_response.json`
- `e131_chat_n4_concurrent_b_time.log`
- `e131_after_concurrent_metrics.prom`

Takeaway:

The chat choices cache now has coverage proving top-level replay and
singleflight behavior for deterministic multi-choice chat. Concurrent identical
sampled chat `n>1` requests still pay two HTTP wait times, but only one
physical `n=4` generation reaches prompt rendering, tokenization, prefix cache,
prefill, decode, and token accounting.

## Experiment E132-E135: Treat `top_k=1` As Effective Greedy

Hypothesis:

Sampling with `top_k=1` and positive finite temperature is mathematically
greedy: the candidate set has exactly one token, so `top_p`, temperature, and
seed cannot change the selected token. Kiln was still treating this shape as
sampled, which prevented greedy cache-key normalization, disabled greedy clone
paths for `n>1`, and used the slower sampling path. If `top_k=1` is normalized
as effective greedy, then both bs=1 and batch `n>1` requests can remove work:
equivalent bs=1 requests should hit the greedy request cache before prompt work,
and batch `n=4` should perform one physical generation and clone the result.

Change:

- Added `SamplingParams::values_are_effectively_greedy` and
  `SamplingParams::is_effectively_greedy`.
- Made `sample_with_params` return the greedy argmax immediately for
  positive-finite `top_k=1`.
- Threaded effective-greedy checks through the paged Metal prefill/decode
  greedy paths, prefix-cache exact greedy-token reuse, completion-cache keys,
  chat request keys, chat choices keys, whole-batch keys, and chat/batch `n>1`
  clone gates.
- Kept explicit skip-layer/MTP speculative gates on `temperature=0.0` only,
  because those lower-level entry points still assert true greedy decoding.

Verification:

- `cargo test -p kiln-model test_sample_top_k_one_is_greedy --lib`
  - Result: 1 passed.
- `cargo test -p kiln-model sampling --lib`
  - Result: 15 passed.
- `cargo test -p kiln-server top_k_one --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server deterministic_chat_choices_cache_key_normalizes_replayable_multi_choice_requests --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server normalizes_equivalent_sampling_fields --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 52 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 41 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server batch_deterministic_clone_gate_requires_explicit_greedy_multi_completion --lib`
  - Result: 1 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.
- `rustfmt --edition 2024 --config skip_children=true --check crates/kiln-core/src/sampling.rs crates/kiln-model/src/sampling.rs crates/kiln-model/src/generate.rs crates/kiln-server/src/api/completions.rs`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 9183 ms. E132 populated a bs=1 greedy chat cache entry with `temperature=0`,
`max_tokens=2`, and `seed=1`. E133 sent the equivalent `top_k=1` request with
`temperature=0.7`, `top_p=0.2`, `top_k=1`, `max_tokens=2`, and `seed=999`.
E134 then sent a one-prompt batch with `n=4`, `temperature=0.7`, `top_p=0.2`,
`top_k=1`, `max_tokens=2`, and `seed=123`. E135 repeated that batch with
different no-op sampling fields (`temperature=0.2`, `top_p=0.8`, `seed=999`).

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E132 bs=1 greedy populate | chat, `temperature=0`, `max_tokens=2` | 3.005507 s | 3004.44 ms | 2.787821 s | 0.212939 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E133 equivalent `top_k=1` hit | chat, `temperature=0.7`, `top_k=1` | 0.000766 s | 0.0940 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E134 batch `top_k=1` populate | one prompt, `n=4`, `max_tokens=2` | 2.947711 s | 2946.58 ms | +2.747752 s across 1 physical completion | +0.193204 s across 1 physical completion | +2, total 4 | one additional render/token/prefix miss |
| E135 equivalent batch hit | same prompt/`n`, different no-op sampling fields | 0.000922 s | 0.0785 ms | unchanged | unchanged | unchanged at 4 | render/token/prefix counters unchanged |

E132 and E133 returned identical bs=1 chat content and usage:
`prompt_tokens=17`, `completion_tokens=2`, `total_tokens=19`, with reasoning
`"Okay,"`. E134 and E135 returned four logical batch completions, each with
`prompt_tokens=19` and `completion_tokens=2`; aggregate batch usage was
`prompt_tokens=76`, `completion_tokens=8`, and `total_tokens=84`.

Metrics confirm the removed work:

- E133 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E132.
- E134 increased prefill/decode counts by 1 each and generated tokens by only
  2 for four logical completions. Before this change, the same `temperature>0`
  `n=4` shape would have run four physical completions and generated 8 tokens.
- E135 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E134.
- Request-duration sum increased only from 3.004211 s to 3.004243 s for E133
  and from 5.950770 s to 5.950791 s for E135.

Artifacts:

- `e132_server.log`
- `e132_health_after_prewarm.json`
- `e132_before_metrics.prom`
- `e132_after_populate_metrics.prom`
- `e132_chat_greedy_populate_response.json`
- `e132_chat_greedy_populate_time.log`
- `e133_after_hit_metrics.prom`
- `e133_chat_topk1_hit_response.json`
- `e133_chat_topk1_hit_time.log`
- `e134_after_populate_metrics.prom`
- `e134_batch_n4_topk1_populate_response.json`
- `e134_batch_n4_topk1_populate_time.log`
- `e135_after_hit_metrics.prom`
- `e135_batch_n4_topk1_hit_response.json`
- `e135_batch_n4_topk1_hit_time.log`

Takeaway:

`top_k=1` now behaves like the deterministic request shape it actually is. This
turns common "sampled-looking but single-candidate" requests into greedy cache
hits for bs=1 and into one-physical-generation clone paths for batch `n>1`,
removing redundant sampling, prompt work on repeats, and physical decode work
for multi-output batches.

## Experiment E136-E139: Let Unseeded `top_k=1` Use Top-Level Replay

Hypothesis:

E132-E135 normalized `top_k=1` as effective greedy in the sampler, model paths,
cache keys, and clone gates, but the replayability gates for chat request
cache, chat choices cache, and whole-batch cache still checked only
`temperature != 0 && seed == None`. That meant unseeded `top_k=1` requests
could be deterministic and clone physical work, but their warm repeats still
fell through to lower caches after prompt rendering/tokenization. If those
gates use effective-greedy detection, unseeded `top_k=1` repeats should return
from the top-level caches before any front-end or model work.

Change:

- Updated deterministic chat request cache, chat choices cache, and batch cache
  replayability gates to accept unseeded effective-greedy sampling.
- Changed the `top_k=1` regression tests to omit seeds.
- Extended chat `n>1` and batch `n>1` `top_k=1` tests to repeat with different
  no-op sampling fields and assert no rendered-prompt/token work is repeated.

Verification:

- `cargo test -p kiln-server top_k_one --lib`
  - Result: 3 passed.
- `cargo test -p kiln-server normalizes_equivalent_sampling_fields --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server deterministic_chat_choices_cache_key_normalizes_replayable_multi_choice_requests --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 52 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 41 passed.
- `cargo test -p kiln-model sampling --lib`
  - Result: 15 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11206 ms. E136 populated an unseeded bs=1 chat request with
`temperature=0.7`, `top_p=0.2`, `top_k=1`, and `max_tokens=2`. E137 repeated
the equivalent request with different no-op sampling fields:
`temperature=0.2`, `top_p=0.8`, `top_k=1`, and no seed. E138 then populated a
one-prompt batch request with `n=4` and the same unseeded `top_k=1` shape; E139
repeated the equivalent batch with the alternate no-op sampling fields.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E136 unseeded `top_k=1` chat populate | bs=1 chat, `max_tokens=2` | 2.958243 s | 2956.71 ms | 2.741936 s | 0.210729 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E137 equivalent unseeded `top_k=1` chat hit | same prompt, different temp/top-p, no seed | 0.000754 s | 0.0905 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E138 unseeded `top_k=1` batch populate | one prompt, `n=4`, `max_tokens=2` | 3.115137 s | 3114.27 ms | +2.906732 s across 1 physical completion | +0.204357 s across 1 physical completion | +2, total 4 | one additional render/token/prefix miss |
| E139 equivalent unseeded `top_k=1` batch hit | same prompt/`n`, different temp/top-p, no seed | 0.000784 s | 0.0951 ms | unchanged | unchanged | unchanged at 4 | render/token/prefix counters unchanged |

E136 and E137 returned identical bs=1 chat content and usage:
`prompt_tokens=21`, `completion_tokens=2`, `total_tokens=23`, with reasoning
`"Okay,"`. E138 and E139 returned four logical batch completions, each with
`prompt_tokens=22` and `completion_tokens=2`; aggregate batch usage was
`prompt_tokens=88`, `completion_tokens=8`, and `total_tokens=96`.

Metrics confirm the removed top-level work:

- E137 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E136.
- E138 increased prefill/decode counts by 1 each and generated tokens by only
  2 for four logical completions, proving the unseeded `top_k=1` batch clone
  path still did one physical generation.
- E139 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E138.
- Request-duration sum increased only from 2.956492 s to 2.956515 s for E137
  and from 6.070728 s to 6.070767 s for E139.

Artifacts:

- `e136_server.log`
- `e136_health_after_prewarm.json`
- `e136_before_metrics.prom`
- `e136_after_populate_metrics.prom`
- `e136_chat_topk1_unseeded_populate_response.json`
- `e136_chat_topk1_unseeded_populate_time.log`
- `e137_after_hit_metrics.prom`
- `e137_chat_topk1_unseeded_hit_response.json`
- `e137_chat_topk1_unseeded_hit_time.log`
- `e138_after_populate_metrics.prom`
- `e138_batch_n4_topk1_unseeded_populate_response.json`
- `e138_batch_n4_topk1_unseeded_populate_time.log`
- `e139_after_hit_metrics.prom`
- `e139_batch_n4_topk1_unseeded_hit_response.json`
- `e139_batch_n4_topk1_unseeded_hit_time.log`

Takeaway:

The `top_k=1` effective-greedy optimization now covers unseeded requests at the
earliest replay boundary. Warm bs=1, chat `n>1`, and batch `n>1` requests with
`top_k=1` no longer need a seed to avoid prompt rendering, tokenization, prefix
lookup, prefill, or decode on equivalent repeats.

## Experiment E140-E143: Normalize Full-Distribution `top_p >= 1.0`

Hypothesis:

The sampler treats `top_p >= 1.0` as full-distribution sampling and skips
nucleus filtering, but deterministic cache keys still used the raw `top_p`
bits for seeded sampled requests. That meant `top_p=1.0` and `top_p=1.5`
produced identical sampled token paths with the same seed, but split
completion, chat request, chat choices, and whole-batch cache entries. If the
cache key normalizes every `top_p >= 1.0` to `1.0`, equivalent seeded sampled
requests should return from the earliest replay cache without prompt or model
work.

Change:

- Added a server-side `normalized_top_p_bits_for_cache` helper that maps
  `top_p >= 1.0` to `1.0f32.to_bits()` and leaves smaller values unchanged.
- Used that helper in seeded sampled completion-cache keys and in the shared
  chat/chat-choices/batch deterministic request sampling key.
- Added regression coverage for completion, chat request, chat choices, and
  batch cache keys, plus bs=1 chat and `n=4` batch before-prompt-work cache-hit
  tests.

Verification:

- `cargo test -p kiln-server top_p_above_one --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server normalizes_equivalent_sampling_fields --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server deterministic_completion_cache_key_accepts_replayable_sampling_only --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_choices_cache_key_normalizes_replayable_multi_choice_requests --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 53 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 42 passed.
- `cargo test -p kiln-model sampling --lib`
  - Result: 15 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 13312 ms. E140 populated a seeded bs=1 chat request with `temperature=0.7`,
`top_p=1.0`, `max_tokens=2`, and `seed=140`. E141 repeated the same request
with `top_p=1.5`. E142 then populated a one-prompt batch request with `n=4`,
`temperature=0.7`, `top_p=1.0`, `max_tokens=2`, and `seed=141`; E143 repeated
that batch with `top_p=1.5`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E140 seeded `top_p=1.0` chat populate | bs=1 chat, `max_tokens=2` | 3.186475 s | 3185.34 ms | 0.009150 s | 3.171579 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E141 equivalent `top_p=1.5` chat hit | same prompt/seed, full-distribution top-p variant | 0.000734 s | 0.0590 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E142 seeded `top_p=1.0` batch populate | one prompt, `n=4`, `max_tokens=2` | 3.346929 s | 3344.72 ms | +0.006876 s across 4 physical completions | +3.325198 s across 4 physical completions | +8, total 10 | render/token/prefix +3 hits, +1 miss |
| E143 equivalent `top_p=1.5` batch hit | same prompt/`n`/seed, full-distribution top-p variant | 0.000950 s | 0.0804 ms | unchanged | unchanged | unchanged at 10 | render/token/prefix counters unchanged |

E140 and E141 returned identical bs=1 chat content and usage:
`prompt_tokens=19`, `completion_tokens=2`, `total_tokens=21`, with reasoning
`"Okay,"`. E142 and E143 returned four logical batch completions, each with
`prompt_tokens=20` and `completion_tokens=2`; aggregate batch usage was
`prompt_tokens=80`, `completion_tokens=8`, and `total_tokens=88`.

Metrics confirm the removed work:

- E141 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E140.
- E142 was sampled `n=4`, so it performed four physical completions and
  generated eight additional tokens. The duplicate prompt hit render/token/
  prefix caches three times and missed once.
- E143 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E142.
- Request-duration sum increased only from 3.185128 s to 3.185145 s for E141
  and from 6.529795 s to 6.529820 s for E143.

Artifacts:

- `e140_server.log`
- `e140_health_after_prewarm.json`
- `e140_before_metrics.prom`
- `e140_after_populate_metrics.prom`
- `e140_chat_top_p_full_populate_response.json`
- `e140_chat_top_p_full_populate_time.log`
- `e141_after_hit_metrics.prom`
- `e141_chat_top_p_above_one_hit_response.json`
- `e141_chat_top_p_above_one_hit_time.log`
- `e142_after_populate_metrics.prom`
- `e142_batch_n4_top_p_full_populate_response.json`
- `e142_batch_n4_top_p_full_populate_time.log`
- `e143_after_hit_metrics.prom`
- `e143_batch_n4_top_p_above_one_hit_response.json`
- `e143_batch_n4_top_p_above_one_hit_time.log`

Takeaway:

Seeded sampled clients that send `top_p` values above one now share cache
entries with the standard full-distribution `top_p=1.0` shape. This removes
all repeat prompt rendering, tokenization, prefix lookup, prefill, and decode
for both bs=1 and batch `n>1` full-distribution top-p variants.

## Experiment E144-E147: Route `top_p <= 0.0` Through Full-Distribution Fast Paths

Hypothesis:

The sampler's nucleus filter only runs for `0.0 < top_p < 1.0`, so
`top_p=0.0` and negative top-p values are already full-distribution requests.
However, with disabled `top_k`, those values previously missed the unsorted
full-distribution sampler path and fell through to the sorted host path. The
same semantic no-op also split deterministic cache keys from `top_p=1.0`.
If the sampler and cache key share a single "top-p disables nucleus filtering"
predicate, `top_p<=0.0` should use the same token path and replay entries as
`top_p=1.0`.

Change:

- Added `SamplingParams::top_p_disables_nucleus_filter`, true for
  `top_p <= 0.0 || top_p >= 1.0`.
- Routed seeded and unseeded full-vocab sampling through the unsorted
  full-distribution paths whenever that predicate is true and `top_k` is
  disabled.
- Reused the same predicate in deterministic cache-key top-p normalization, so
  seeded `top_p=0.0`, negative top-p, `top_p=1.0`, and `top_p>1.0` share the
  same full-distribution replay key.
- Added model tests proving same-seed token identity across outside-range
  top-p values, plus server cache-key and before-prompt-work replay tests for
  bs=1 chat and batch `n=4`.

Verification:

- `cargo test -p kiln-model top_p_outside_range --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server top_p_zero --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server top_p_above_one --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server normalizes_equivalent_sampling_fields --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server deterministic_completion_cache_key_accepts_replayable_sampling_only --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_choices_cache_key_normalizes_replayable_multi_choice_requests --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 54 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 43 passed.
- `cargo test -p kiln-model sampling --lib`
  - Result: 16 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11278 ms. E144 populated a seeded bs=1 chat request with `temperature=0.7`,
`top_p=0.0`, `max_tokens=2`, and `seed=144`. E145 repeated the same request
with `top_p=1.0`. E146 then populated a one-prompt batch request with `n=4`,
`temperature=0.7`, `top_p=0.0`, `max_tokens=2`, and `seed=145`; E147 repeated
that batch with `top_p=1.0`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E144 seeded `top_p=0.0` chat populate | bs=1 chat, `max_tokens=2` | 6.375363 s | 6373.82 ms | 0.006621 s | 6.364731 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E145 equivalent `top_p=1.0` chat hit | same prompt/seed, full-distribution top-p variant | 0.000960 s | 0.0677 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E146 seeded `top_p=0.0` batch populate | one prompt, `n=4`, `max_tokens=2` | 3.335245 s | 3334.03 ms | +0.007756 s across 4 physical completions | +3.313933 s across 4 physical completions | +8, total 10 | render/token/prefix +3 hits, +1 miss |
| E147 equivalent `top_p=1.0` batch hit | same prompt/`n`/seed, full-distribution top-p variant | 0.000821 s | 0.0775 ms | unchanged | unchanged | unchanged at 10 | render/token/prefix counters unchanged |

E144 and E145 returned identical bs=1 chat content and usage:
`prompt_tokens=18`, `completion_tokens=2`, `total_tokens=20`, with reasoning
`"Okay,"`. E146 and E147 returned four logical batch completions, each with
`prompt_tokens=19` and `completion_tokens=2`; aggregate batch usage was
`prompt_tokens=76`, `completion_tokens=8`, and `total_tokens=84`.

Metrics confirm the removed work:

- E145 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E144.
- E146 was sampled `n=4`, so it performed four physical completions and
  generated eight additional tokens. The duplicate prompt hit render/token/
  prefix caches three times and missed once.
- E147 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E146.
- Request-duration sum increased only from 6.373671 s to 6.373690 s for E145
  and from 9.707659 s to 9.707680 s for E147.

Artifacts:

- `e144_server.log`
- `e144_health_after_prewarm.json`
- `e144_before_metrics.prom`
- `e144_after_populate_metrics.prom`
- `e144_chat_top_p_zero_populate_response.json`
- `e144_chat_top_p_zero_populate_time.log`
- `e145_after_hit_metrics.prom`
- `e145_chat_top_p_full_hit_response.json`
- `e145_chat_top_p_full_hit_time.log`
- `e146_after_populate_metrics.prom`
- `e146_batch_n4_top_p_zero_populate_response.json`
- `e146_batch_n4_top_p_zero_populate_time.log`
- `e147_after_hit_metrics.prom`
- `e147_batch_n4_top_p_full_hit_response.json`
- `e147_batch_n4_top_p_full_hit_time.log`

Takeaway:

Outside-range full-distribution top-p values now use the same sampler predicate
and deterministic replay key. For seeded `top_p=0.0` and negative-top-p client
requests, warm bs=1 and batch `n>1` repeats now avoid prompt rendering,
tokenization, prefix lookup, prefill, and decode just like `top_p=1.0`.

## Experiment E148-E151: Normalize Oversized top_k as Disabled

Hypothesis:

Sampler behavior already treats `top_k >= logits_vocab_size` the same as
disabled `top_k=0`, but deterministic replay keys kept the client-supplied
oversized value. That made semantically equivalent seeded sampled requests miss
the top-level chat/batch caches. Normalize `top_k` to `0` in cache keys when it
is nonzero and greater than or equal to the loaded model vocab size. Use
`state.model_config.vocab_size`, not tokenizer vocab size, because the Qwen3.5
tokenizer and model vocab sizes differ.

Implementation:

- Added `normalized_top_k_for_cache(top_k, vocab_size)` in
  `crates/kiln-server/src/api/completions.rs` and threaded the loaded
  `state.model_config.vocab_size` into deterministic chat request,
  chat-choices, and batch cache-key builders.
- Kept test-only wrappers that use `usize::MAX` so older cache-key unit tests
  still exercise request-field normalization without pretending to know a model
  vocab size.
- Added cache-key assertions and handler tests covering seeded sampled
  oversized `top_k` replay for bs=1 chat and a one-prompt `n=4` batch.

Validation commands:

- `cargo test -p kiln-server top_k_oversized --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server normalizes_equivalent_sampling_fields --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server deterministic_completion_cache_key_accepts_replayable_sampling_only --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_choices_cache_key_normalizes_replayable_multi_choice_requests --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 55 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 44 passed.
- `cargo test -p kiln-model sampling --lib`
  - Result: 16 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 11278 ms. E148 populated a seeded bs=1 chat request with
`temperature=0.7`, `top_p=1.0`, oversized `top_k=999999999`, `max_tokens=2`,
and `seed=148`. E149 repeated the same request with `top_k=0`. E150 then
populated a one-prompt batch request with `n=4`, `temperature=0.7`,
`top_p=1.0`, oversized `top_k=999999999`, `max_tokens=2`, and `seed=149`;
E151 repeated that batch with `top_k=0`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E148 seeded oversized `top_k` chat populate | bs=1 chat, `max_tokens=2` | 2.852266 s | 2850.10 ms | 0.008616 s | 2.836765 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E149 equivalent `top_k=0` chat hit | same prompt/seed, disabled top-k variant | 0.000895 s | 0.0580 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E150 seeded oversized `top_k` batch populate | one prompt, `n=4`, `max_tokens=2` | 3.177092 s | 3175.63 ms | +0.006031 s across 4 physical completions | +3.154609 s across 4 physical completions | +8, total 10 | render/token/prefix +3 hits, +1 miss |
| E151 equivalent `top_k=0` batch hit | same prompt/`n`/seed, disabled top-k variant | 0.001164 s | 0.1218 ms | unchanged | unchanged | unchanged at 10 | render/token/prefix counters unchanged |

E148 and E149 returned identical bs=1 chat content and usage:
`prompt_tokens=20`, `completion_tokens=2`, `total_tokens=22`, with reasoning
`"Here's"`. E150 and E151 returned four logical batch completions, each with
`prompt_tokens=21` and `completion_tokens=2`; aggregate batch usage was
`prompt_tokens=84`, `completion_tokens=8`, and `total_tokens=92`.

Metrics confirm the removed work:

- E149 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E148.
- E150 was sampled `n=4`, so it performed four physical completions and
  generated eight additional tokens. The duplicate prompt hit render/token/
  prefix caches three times and missed once.
- E151 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E150.
- Request-duration sum increased only from 2.849885 s to 2.849902 s for E149
  and from 6.025465 s to 6.025499 s for E151.

Artifacts:

- `e148_server.log`
- `e148_health_after_prewarm.json`
- `e148_before_metrics.prom`
- `e148_after_populate_metrics.prom`
- `e148_chat_top_k_oversized_populate_response.json`
- `e148_chat_top_k_oversized_populate_time.log`
- `e149_after_hit_metrics.prom`
- `e149_chat_top_k_disabled_hit_response.json`
- `e149_chat_top_k_disabled_hit_time.log`
- `e150_after_populate_metrics.prom`
- `e150_batch_n4_top_k_oversized_populate_response.json`
- `e150_batch_n4_top_k_oversized_populate_time.log`
- `e151_after_hit_metrics.prom`
- `e151_batch_n4_top_k_disabled_hit_response.json`
- `e151_batch_n4_top_k_disabled_hit_time.log`

Takeaway:

Oversized `top_k` values now use the same deterministic replay key as disabled
top-k when they are greater than or equal to the loaded model vocab size. Warm
bs=1 and batch `n>1` repeats now skip prompt rendering, tokenization, prefix
lookup, prefill, and decode for this OpenAI-compatible parameter variant.

## Experiment E152-E155: Drop Dominated Stop Strings from Replay Keys

Hypothesis:

Deterministic cache keys already sort and deduplicate stop sequences, but a
longer stop string that starts with a shorter stop string in the same request
is also redundant. Generation checks whether decoded text contains any stop
sequence, and the API exposes only the public finish reason `"stop"`, not the
specific matching string. Therefore `["abc", "abc-def"]` is equivalent to
`["abc"]`: the shorter sequence dominates the longer one. Normalizing these
dominated strings out of replay keys should remove prompt rendering,
tokenization, prefix lookup, prefill, and decode on equivalent repeats.

Implementation:

- Updated `normalized_stop_for_cache` in
  `crates/kiln-server/src/api/completions.rs` to keep only prefix-minimal stop
  strings after sorting and deduplicating. Existing empty-stop handling remains
  the strongest dominance case.
- Added unit coverage for prefix-dominated stop normalization.
- Added bs=1 chat and one-prompt batch `n=4` handler tests proving the
  dominated-stop variant hits the same top-level cache before prompt work.

Validation commands:

- `cargo test -p kiln-server dominated_stop --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server deterministic_cache_keys_normalize_stop_sequence_sets --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server stop_string --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 56 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 45 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 9295 ms. E152 populated a seeded bs=1 chat request with `temperature=0.7`,
`top_p=1.0`, `max_tokens=2`, `seed=152`, and a dominated stop list
`["dominated-stop-152", "dominated-stop-152-suffix"]`. E153 repeated the same
request with the minimal stop list `["dominated-stop-152"]`. E154 then
populated a one-prompt batch request with `n=4`, `temperature=0.7`,
`top_p=1.0`, `max_tokens=2`, `seed=153`, and
`["dominated-stop-154", "dominated-stop-154-suffix"]`; E155 repeated that batch
with `["dominated-stop-154"]`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E152 dominated-stop chat populate | bs=1 chat, `max_tokens=2` | 3.933002 s | 3931.81 ms | 0.009231 s | 3.918364 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E153 minimal-stop chat hit | same prompt/seed, dominated stop removed | 0.000756 s | 0.0718 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E154 dominated-stop batch populate | one prompt, `n=4`, `max_tokens=2` | 1.138238 s | 1136.88 ms | +0.012232 s across 4 physical completions | +1.110092 s across 4 physical completions | +8, total 10 | render/token/prefix +3 hits, +1 miss |
| E155 minimal-stop batch hit | same prompt/`n`/seed, dominated stop removed | 0.001130 s | 0.0686 ms | unchanged | unchanged | unchanged at 10 | render/token/prefix counters unchanged |

E152 and E153 returned identical bs=1 chat content and usage:
`prompt_tokens=20`, `completion_tokens=2`, `total_tokens=22`, with reasoning
`"Here's"`. E154 and E155 returned four logical batch completions, each with
`prompt_tokens=20` and `completion_tokens=2`; aggregate batch usage was
`prompt_tokens=80`, `completion_tokens=8`, and `total_tokens=88`.

Metrics confirm the removed work:

- E153 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E152.
- E154 was sampled `n=4`, so it performed four physical completions and
  generated eight additional tokens. The duplicate prompt hit render/token/
  prefix caches three times and missed once.
- E155 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E154.
- Request-duration sum increased only from 3.931644 s to 3.931664 s for E153
  and from 5.068482 s to 5.068502 s for E155.

Artifacts:

- `e152_server.log`
- `e152_health_after_prewarm.json`
- `e152_before_metrics.prom`
- `e152_after_populate_metrics.prom`
- `e152_chat_dominated_stop_populate_response.json`
- `e152_chat_dominated_stop_populate_time.log`
- `e153_after_hit_metrics.prom`
- `e153_chat_minimal_stop_hit_response.json`
- `e153_chat_minimal_stop_hit_time.log`
- `e154_after_populate_metrics.prom`
- `e154_batch_n4_dominated_stop_populate_response.json`
- `e154_batch_n4_dominated_stop_populate_time.log`
- `e155_after_hit_metrics.prom`
- `e155_batch_n4_minimal_stop_hit_response.json`
- `e155_batch_n4_minimal_stop_hit_time.log`

Takeaway:

Redundant stop strings that are dominated by a shorter stop string no longer
split deterministic replay keys. Warm bs=1 and batch `n>1` repeats now skip all
front-end and model work for this common client-side stop-list redundancy.

## Experiment E156-E159: Drop Substring-Dominated Stop Strings

Hypothesis:

E152-E155 handled stop strings where the shorter stop was a prefix of the
longer stop, but generation actually checks `decoded_text.contains(stop_seq)`.
That means a shorter stop sequence dominates any longer stop sequence that
contains it anywhere, not only at the beginning. For example,
`["needle", "prefix-needle-suffix"]` is equivalent to `["needle"]` because any
match of the longer string necessarily includes a match of the shorter string
at the same decode step. Normalize stop keys by length first and drop any
longer stop containing an already-kept shorter stop.

Implementation:

- Updated `normalized_stop_for_cache` in
  `crates/kiln-server/src/api/completions.rs` to sort unique stop strings by
  length, then keep only strings that do not contain an already-kept shorter
  string.
- Added unit coverage for non-prefix substring dominance.
- Added bs=1 chat and one-prompt batch `n=4` handler tests proving
  substring-dominated stop lists hit the same top-level replay cache before
  prompt work.

Validation commands:

- `cargo test -p kiln-server substring_dominated_stop --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server deterministic_cache_keys_normalize_stop_sequence_sets --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server dominated_stop --lib`
  - Result: 4 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 57 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 46 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 10348 ms. E156 populated a seeded bs=1 chat request with `temperature=0.7`,
`top_p=1.0`, `max_tokens=2`, `seed=156`, and a substring-dominated stop list
`["prefix-substring-stop-156-suffix", "substring-stop-156"]`. E157 repeated
the same request with the minimal stop list `["substring-stop-156"]`. E158 then
populated a one-prompt batch request with `n=4`, `temperature=0.7`,
`top_p=1.0`, `max_tokens=2`, `seed=157`, and
`["prefix-substring-stop-158-suffix", "substring-stop-158"]`; E159 repeated
that batch with `["substring-stop-158"]`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E156 substring-dominated stop chat populate | bs=1 chat, `max_tokens=2` | 5.203822 s | 5202.72 ms | 0.007536 s | 5.191826 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E157 minimal substring-stop chat hit | same prompt/seed, dominated stop removed | 0.000809 s | 0.0676 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E158 substring-dominated stop batch populate | one prompt, `n=4`, `max_tokens=2` | 2.947058 s | 2944.29 ms | +0.006762 s across 4 physical completions | +2.922559 s across 4 physical completions | +8, total 10 | render/token/prefix +3 hits, +1 miss |
| E159 minimal substring-stop batch hit | same prompt/`n`/seed, dominated stop removed | 0.000850 s | 0.0708 ms | unchanged | unchanged | unchanged at 10 | render/token/prefix counters unchanged |

E156 and E157 returned identical bs=1 chat content and usage:
`prompt_tokens=20`, `completion_tokens=2`, `total_tokens=22`, with reasoning
`"Here's"`. E158 and E159 returned four logical batch completions, each with
`prompt_tokens=20` and `completion_tokens=2`; aggregate batch usage was
`prompt_tokens=80`, `completion_tokens=8`, and `total_tokens=88`.

Metrics confirm the removed work:

- E157 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E156.
- E158 was sampled `n=4`, so it performed four physical completions and
  generated eight additional tokens. The duplicate prompt hit render/token/
  prefix caches three times and missed once.
- E159 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E158.
- Request-duration sum increased only from 5.202445 s to 5.202466 s for E157
  and from 8.146701 s to 8.146720 s for E159.

Artifacts:

- `e156_server.log`
- `e156_health_after_prewarm.json`
- `e156_before_metrics.prom`
- `e156_after_populate_metrics.prom`
- `e156_chat_substring_dominated_stop_populate_response.json`
- `e156_chat_substring_dominated_stop_populate_time.log`
- `e157_after_hit_metrics.prom`
- `e157_chat_minimal_substring_stop_hit_response.json`
- `e157_chat_minimal_substring_stop_hit_time.log`
- `e158_after_populate_metrics.prom`
- `e158_batch_n4_substring_dominated_stop_populate_response.json`
- `e158_batch_n4_substring_dominated_stop_populate_time.log`
- `e159_after_hit_metrics.prom`
- `e159_batch_n4_minimal_substring_stop_hit_response.json`
- `e159_batch_n4_minimal_substring_stop_hit_time.log`

Takeaway:

Stop-list replay canonicalization now matches the server's actual substring
stop detection semantics. Warm bs=1 and batch `n>1` repeats skip prompt
rendering, tokenization, prefix lookup, prefill, and decode even when clients
send longer redundant stop strings that merely contain the effective shorter
stop.

## Experiment E160-E163: Use Canonical Stop Lists for Fresh Generation

Hypothesis:

The replay key now canonicalizes redundant stop lists, but fresh populate
requests still carried the original stop list into generation and synthetic
fanout requests. That leaves avoidable work on the populate path: cloning
redundant stop strings into every synthesized completion and checking longer
dominated stop strings with `contains()` after every generated token. Use the
same canonical stop list for generation that replay keys use.

Implementation:

- Added `normalized_stop_for_generation` and
  `normalized_stop_option_for_synthetic_request` in
  `crates/kiln-server/src/api/completions.rs`.
- Chat bs=1 and `generate_one_response` now build `SamplingParams` with the
  canonical stop list.
- Chat `n>1`, single-prompt batch-to-chat-choices keying, and batch fanout now
  pass canonical stop lists into synthetic `ChatCompletionRequest`s so fanout
  clones only the effective stop strings.
- Extended stop normalization unit coverage to assert generation and replay-key
  stop canonicalization stay aligned.

Validation commands:

- `cargo test -p kiln-server deterministic_cache_keys_normalize_stop_sequence_sets --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server substring_dominated_stop --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server dominated_stop --lib`
  - Result: 4 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 57 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 46 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 and background prewarm completed
in 10802 ms. E160 populated a seeded bs=1 chat request with `temperature=0.7`,
`top_p=1.0`, `max_tokens=2`, `seed=160`, and a 65-item stop list: one minimal
stop string plus 64 longer strings containing it. E161 repeated the same
request with only the minimal stop. E162 then populated a one-prompt batch
request with `n=4`, `temperature=0.7`, `top_p=1.0`, `max_tokens=2`,
`seed=161`, and the same 65-item dominated-stop shape; E163 repeated that
batch with only the minimal stop.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E160 canonicalized 65-stop chat populate | bs=1 chat, `max_tokens=2` | 2.421592 s | 2419.31 ms | 0.007221 s | 2.408289 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E161 minimal-stop chat hit | same prompt/seed, canonical stop list | 0.000891 s | 0.0788 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E162 canonicalized 65-stop batch populate | one prompt, `n=4`, `max_tokens=2` | 3.336007 s | 3332.53 ms | +0.005697 s across 4 physical completions | +3.313458 s across 4 physical completions | +8, total 10 | render/token/prefix +3 hits, +1 miss |
| E163 minimal-stop batch hit | same prompt/`n`/seed, canonical stop list | 0.000794 s | 0.0849 ms | unchanged | unchanged | unchanged at 10 | render/token/prefix counters unchanged |

E160 and E161 returned identical bs=1 chat content and usage:
`prompt_tokens=19`, `completion_tokens=2`, `total_tokens=21`, with reasoning
`"Here's"`. E162 and E163 returned four logical batch completions, each with
`prompt_tokens=19` and `completion_tokens=2`; aggregate batch usage was
`prompt_tokens=76`, `completion_tokens=8`, and `total_tokens=84`.

Metrics confirm behavior stayed on the removed-work path:

- E161 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E160.
- E162 was sampled `n=4`, so it performed four physical completions and
  generated eight additional tokens. The duplicate prompt hit render/token/
  prefix caches three times and missed once.
- E163 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E162.
- Request-duration sum increased only from 2.419043 s to 2.419064 s for E161
  and from 5.751519 s to 5.751537 s for E163.

Artifacts:

- `e160_server.log`
- `e160_health_after_prewarm.json`
- `e160_before_metrics.prom`
- `e160_after_populate_metrics.prom`
- `e160_chat_canonical_stop_populate_response.json`
- `e160_chat_canonical_stop_populate_time.log`
- `e161_after_hit_metrics.prom`
- `e161_chat_minimal_canonical_stop_hit_response.json`
- `e161_chat_minimal_canonical_stop_hit_time.log`
- `e162_after_populate_metrics.prom`
- `e162_batch_n4_canonical_stop_populate_response.json`
- `e162_batch_n4_canonical_stop_populate_time.log`
- `e163_after_hit_metrics.prom`
- `e163_batch_n4_minimal_canonical_stop_hit_response.json`
- `e163_batch_n4_minimal_canonical_stop_hit_time.log`

Takeaway:

Stop-list canonicalization now removes redundant stop strings both from replay
keys and from fresh generation work. This avoids repeated fanout cloning and
per-token stop comparisons for dominated client stop lists while preserving the
same replay behavior for warm bs=1 and batch `n>1` requests.

## Experiment E164-E167: Drop No-Tool `tool_choice=none` from Chat Fanout

Hypothesis:

OpenAI-compatible clients often send both `tools: []` and
`tool_choice: "none"` even when no tools are available. Earlier cache-key
normalization already treated those fields as equivalent to omission, but chat
`n>1` fanout still cloned the raw empty tool list and no-op tool choice into
each synthesized single-choice request. Canonicalizing those fields before
fanout should keep replay equivalence and remove avoidable request cloning on
the hot fanout path.

Implementation:

- Added `normalized_tools_option_for_synthetic_request()` and
  `normalized_tool_choice_option_for_synthetic_request()` next to the existing
  tool cache-key normalizers in `crates/kiln-server/src/api/completions.rs`.
- Updated `generate_multi_chat_response()` to pass canonical tool fields into
  synthesized single-choice chat requests. Empty `tools` becomes `None`, and
  no-tool `tool_choice: "auto"`/`"none"` becomes `None`; no-tool
  `"required"` remains preserved because it is behaviorally distinct.
- Extended deterministic key tests for empty tools, no-tool auto choice, and
  no-tool none choice.
- Added live handler tests proving omitted tools and `tools: []` plus
  `tool_choice: "none"` replay before prompt work for bs=1 and chat `n=4`.

Validation commands:

- `cargo test -p kiln-server no_tool_none --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server no_tool_auto --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server normalizes_empty_tools --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 59 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server single_prompt_batch --lib`
  - Result: 2 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Background inference prewarm
completed in 10650 ms. E164 populated a greedy bs=1 chat request with omitted
tool fields. E165 repeated the same prompt with `tools: []` and
`tool_choice: "none"`. E166 populated a greedy chat request with `n=4` and
omitted tool fields. E167 repeated that `n=4` prompt with `tools: []` and
`tool_choice: "none"`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E164 omitted-tool chat populate | bs=1 chat, `max_tokens=2` | 3.393048 s | 3391.95 ms | 3.179833 s | 0.191779 s | 2 | render miss 1, token miss 1, prefix miss 1 |
| E165 no-tool none chat hit | same prompt, `tools: []`, `tool_choice: "none"` | 0.000676 s | 0.0673 ms | unchanged | unchanged | unchanged at 2 | render/token/prefix counters unchanged |
| E166 omitted-tool chat `n=4` populate | one prompt, `n=4`, `max_tokens=2` | 3.091951 s | 3090.52 ms | +2.879942 s | +0.203781 s | +2, total 4 | render/token/prefix +1 miss |
| E167 no-tool none chat `n=4` hit | same prompt/`n`, `tools: []`, `tool_choice: "none"` | 0.000769 s | 0.0793 ms | unchanged | unchanged | unchanged at 4 | render/token/prefix counters unchanged |

E164 and E165 returned identical bs=1 chat usage:
`prompt_tokens=19`, `completion_tokens=2`, `total_tokens=21`, with empty
assistant content and reasoning `"Okay,"`. E166 and E167 returned four
logical choices, each with empty assistant content and reasoning
`"Thinking Process"`; aggregate response usage was `prompt_tokens=21`,
`completion_tokens=8`, and `total_tokens=29`.

Metrics confirm the hit path:

- E165 increased request count from 1 to 2 but left prefill/decode counts,
  generated tokens, rendered-prompt cache, prompt-token cache, and prefix-cache
  counters unchanged from E164.
- E166 returned four logical choices but, because the request is greedy and
  deterministic, only added one physical two-token generation and one prompt
  miss to the model/front-end counters.
- E167 increased request count from 3 to 4 but left all model and front-end
  counters unchanged from E166.
- Request-duration sum increased only from 3.391695 s to 3.391714 s for E165
  and from 6.482181 s to 6.482211 s for E167.

Artifacts:

- `e164_server.log`
- `e164_health_after_prewarm.json`
- `e164_before_metrics.prom`
- `e164_after_populate_metrics.prom`
- `e164_chat_no_tool_none_base_populate_response.json`
- `e164_chat_no_tool_none_base_populate_time.log`
- `e165_after_hit_metrics.prom`
- `e165_chat_no_tool_none_hit_response.json`
- `e165_chat_no_tool_none_hit_time.log`
- `e166_after_populate_metrics.prom`
- `e166_chat_n4_no_tool_none_base_populate_response.json`
- `e166_chat_n4_no_tool_none_base_populate_time.log`
- `e167_after_hit_metrics.prom`
- `e167_chat_n4_no_tool_none_hit_response.json`
- `e167_chat_n4_no_tool_none_hit_time.log`

Takeaway:

No-tool tool metadata is now canonicalized before chat fanout, not just inside
the replay key. This keeps common client shapes like omitted tools and
`tools: []` plus `tool_choice: "none"` on the same deterministic replay path
while avoiding pointless empty-vector and no-op choice cloning for chat `n>1`.

## Experiment E168-E169: Borrow Batch Replay Keys from Request Prompts

Hypothesis:

Whole-batch deterministic replay still built an owned nested key before it could
check the batch cache: every prompt row cloned its rendered role/content pair
into `Vec<Vec<(String, String)>>`, then the cache cloned that owned key again
for LRU and in-flight tracking. Chat replay keys already serialize borrowed
message fields into one owned key string. Moving the batch replay key to the
same borrowed-serialization shape should reduce high-row cache-hit CPU and
allocation work without changing cache semantics.

Implementation:

- Replaced the structural `DeterministicBatchCacheKey` with an owned `String`
  key in `crates/kiln-server/src/state.rs`, matching chat request and chat
  choices caches.
- Added borrowed `BatchPromptMessageCacheKey` and
  `DeterministicBatchCacheKeyWire` in
  `crates/kiln-server/src/api/completions.rs`.
- Updated `deterministic_batch_cache_key_with_vocab_size()` to serialize
  borrowed `&str` role/content fields from the request instead of cloning every
  prompt's role/content strings into the cache key.
- Preserved the current batch semantics: the batch cache key still includes
  only role/content for each prompt because the batch renderer synthesizes
  role/content-only chat messages and intentionally drops unrendered message
  metadata.

Validation commands:

- `cargo test -p kiln-server deterministic_batch_cache --lib`
  - Result: 7 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 46 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server single_prompt_batch --lib`
  - Result: 2 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 59 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Background inference prewarm
completed in 10921 ms. E168 populated a 64-prompt batch request with `n=1`,
`temperature=0.0`, and `max_tokens=0`. E169 repeated the identical request to
exercise the warm whole-batch cache hit. This shape intentionally avoids decode
so the measurement isolates high-row replay-key and prompt front-end work.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E168 borrowed-key batch populate | 64 prompt rows, `n=1`, `max_tokens=0` | 0.011387 s | 10.5798 ms | 0 | 0 | 0 | render miss 64, token miss 64, prefix unchanged |
| E169 borrowed-key batch hit | identical 64-row request | 0.000965 s | 0.1648 ms | unchanged | unchanged | unchanged at 0 | render/token/prefix counters unchanged |

Both responses returned 64 completions, each with empty text and
`finish_reason="length"`. Aggregate usage matched exactly:
`prompt_tokens=1270`, `completion_tokens=0`, `total_tokens=1270`.

Metrics confirm the hit path:

- E168 increased request count from 0 to 1, added 64 rendered-prompt misses and
  64 prompt-token misses, and left prefill/decode/generated-token counters at
  zero.
- E169 increased request count from 1 to 2 but left rendered-prompt,
  prompt-token, prefix-cache, prefill, decode, and generated-token counters
  unchanged.
- Request-duration sum increased only from 0.010305 s to 0.010385 s for the
  64-row replay hit.

Artifacts:

- `e168_server.log`
- `e168_health_after_prewarm.json`
- `e168_before_metrics.prom`
- `e168_batch_64_zero_request.json`
- `e168_batch_64_zero_populate_response.json`
- `e168_batch_64_zero_populate_time.log`
- `e168_after_populate_metrics.prom`
- `e169_batch_64_zero_hit_response.json`
- `e169_batch_64_zero_hit_time.log`
- `e169_after_hit_metrics.prom`

Takeaway:

High-row batch cache probes now avoid constructing a nested owned prompt-key
object before lookup. The cache still owns one serialized key string for LRU and
singleflight, but request role/content strings are borrowed during key
serialization, removing one layer of prompt-row string allocation on every
batch replay probe.

## Experiment E170-E171: Store One Synthesized Prompt per Duplicate Batch Group

Hypothesis:

Batch duplicate-prompt coalescing already skips repeated prompt rendering,
tokenization, and greedy generation, but the grouping structure still stored a
full synthesized `Vec<Message>` for every duplicate row. For high-batch
duplicate prompts, that means cloning the same role/content messages repeatedly
before the grouped path can remove the prompt/model work. A group should own
one synthesized message vector and a list of prompt indices instead.

Implementation:

- Replaced `BatchPromptWork { prompt_index, messages }` group entries with
  `BatchPromptGroup { messages, prompt_indices }` in
  `crates/kiln-server/src/api/completions.rs`.
- Updated `batch_prompt_groups()` to synthesize messages only when first seeing
  a distinct role/content prompt. Duplicate prompt rows now append only their
  prompt index.
- Switched the duplicate grouping map to a serialized borrowed role/content key
  using the existing `batch_prompt_cache_key()` shape, avoiding the old
  `Vec<(String, String)>` prompt grouping key clones as well.
- Updated the zero-token batch path and the async generation fanout path to use
  group-level messages plus per-index seed derivation, preserving public
  response ordering and the documented `seed + prompt_index * n + completion`
  rule.

Validation commands:

- `cargo test -p kiln-server batch_prompt_groups --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server duplicate_batch --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_greedy_duplicate --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 46 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo test -p kiln-server single_prompt_batch --lib`
  - Result: 2 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Background inference prewarm
completed in 9167 ms. E170 populated a 64-prompt batch where every prompt row
was identical, with `n=1`, `temperature=0.7`, and `max_tokens=0`. E171 repeated
the identical request to verify the whole-batch cache hit after populate.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end/prefix work |
|---|---|---:|---:|---:|---:|---:|---|
| E170 grouped duplicate batch populate | 64 identical prompt rows, `n=1`, `max_tokens=0` | 0.002203 s | 1.5123 ms | 0 | 0 | 0 | render miss 1, token miss 1, prefix unchanged |
| E171 grouped duplicate batch hit | identical 64-row duplicate request | 0.001046 s | 0.1770 ms | unchanged | unchanged | unchanged at 0 | render/token/prefix counters unchanged |

Both responses returned 64 completions, each with empty text and
`finish_reason="length"`. Aggregate usage matched exactly:
`prompt_tokens=960`, `completion_tokens=0`, `total_tokens=960`.

Metrics confirm the removed-work path:

- E170 increased request count from 0 to 1, added only one rendered-prompt miss
  and one prompt-token miss for 64 logical prompt rows, and left
  prefill/decode/generated-token counters at zero.
- E171 increased request count from 1 to 2 but left rendered-prompt,
  prompt-token, prefix-cache, prefill, decode, and generated-token counters
  unchanged.
- Request-duration sum increased only from 0.001303 s to 0.001385 s for the
  warm batch replay hit.

Artifacts:

- `e170_server.log`
- `e170_health_after_prewarm.json`
- `e170_before_metrics.prom`
- `e170_batch_64_duplicate_zero_request.json`
- `e170_batch_64_duplicate_zero_populate_response.json`
- `e170_batch_64_duplicate_zero_populate_time.log`
- `e170_after_populate_metrics.prom`
- `e171_batch_64_duplicate_zero_hit_response.json`
- `e171_batch_64_duplicate_zero_hit_time.log`
- `e171_after_hit_metrics.prom`

Takeaway:

Duplicate-prompt batch populate now removes another layer of unnecessary work:
one synthesized prompt is stored per distinct group, and duplicate rows carry
only indices. This keeps the existing render/token/model elimination while
reducing high-batch CPU allocation and clone pressure before the grouped path
runs.

## Experiment E172-E173: Serve Multi-Prompt Batch from Chat Request Cache

Hypothesis:

After individual `n=1` chat requests have populated deterministic chat request
cache entries, an equivalent multi-prompt batch with `n=1` should be able to
return the whole batch response directly from those per-prompt entries. That
removes batch prompt grouping, synthetic fanout request construction, prompt
render/token cache probes, scheduler work, and model work on the warm path.

Implementation:

- Added `DeterministicChatRequestCacheProbe` and
  `DeterministicChatRequestCache::probe()` in
  `crates/kiln-server/src/state.rs`, mirroring the existing completion and
  chat-choices cache probes while preserving LRU refresh on hits.
- Added a borrowed batch-to-chat request cache key path in
  `crates/kiln-server/src/api/completions.rs`. It serializes the synthesized
  batch message shape as role/content-only chat prompt keys with no tools or
  tool choice, and applies the same deterministic sampling normalization as
  chat requests.
- Added `batch_response_from_chat_request_cache_hits()` before the zero-token
  and prompt-grouping paths. It is restricted to `n=1` and no adapters, derives
  per-prompt seeds as `seed + prompt_index`, probes all chat request cache keys
  under one cache lock, and only returns when every prompt is a hit.
- When the shortcut returns, the normal batch response is also inserted into
  the whole-batch cache/singleflight owner so repeated identical batch requests
  keep taking the already-established batch replay path.
- Added
  `multi_prompt_batch_hits_chat_request_cache_before_fanout_work`, which warms
  two chat request-cache entries, submits the equivalent two-prompt batch, and
  asserts generated-token, render-cache, and token-cache counters do not move.

Validation commands:

- `cargo test -p kiln-server multi_prompt_batch_hits_chat_request_cache_before_fanout_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_request_cache_coalesces_in_flight_request --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_single_output_hits_chat_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 47 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 60 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Background inference prewarm
completed in 2342 ms. E172 populated 64 distinct zero-token chat requests with
`temperature=0.7`, `max_tokens=0`, and `seed=172`. E173 submitted the
equivalent 64-prompt batch request with `n=1` and the same sampling fields.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E172 chat cache populate | 64 distinct chat requests, `max_tokens=0` | 0.046447 s total, 0.000726 s avg | 18.593 ms total, 0.2905 ms avg | 0 | 0 | 0 | render miss 64, token miss 64 |
| E173 batch from chat request cache | 64 prompts, `n=1`, `max_tokens=0` | 0.001105 s | 0.2688 ms | unchanged at 0 | unchanged at 0 | unchanged at 0 | render/token counters unchanged |

E173 returned 64 batch completions with aggregate usage
`prompt_tokens=1014`, `completion_tokens=0`, `total_tokens=1014`, matching the
sum of the 64 warm chat responses.

Metrics confirm the removed-work path:

- E172 increased request count from 0 to 64, added 64 rendered-prompt misses
  and 64 prompt-token misses, and left prefill/decode/generated-token counters
  at zero.
- E173 increased request count from 64 to 65, while rendered-prompt misses,
  prompt-token misses, prefill count, decode count, and generated tokens all
  remained unchanged.
- Request-duration sum moved from 0.016203 s after E172 to 0.016377 s after
  E173, a 0.000174 s increment for the 64-completion batch cache-hit response.

Artifacts:

- `e172_server.log`
- `e172_health_after_prewarm.json`
- `e172_before_metrics.prom`
- `e172_chat_64_zero_populate_requests.ndjson`
- `e172_chat_64_zero_populate_responses.ndjson`
- `e172_chat_64_zero_populate_times.log`
- `e172_after_chat_populate_metrics.prom`
- `e173_batch_64_from_chat_cache_request.json`
- `e173_batch_64_from_chat_cache_hit_response.json`
- `e173_batch_64_from_chat_cache_hit_time.log`
- `e173_after_hit_metrics.prom`

Takeaway:

The batch endpoint can now reuse already-warmed single-chat deterministic
request-cache entries for multi-prompt `n=1` batches. On the measured 64-prompt
warm batch, the server returned directly from cache without any additional
prompt rendering, tokenization, prefill, decode, or token generation.

## Experiment E174-E175: Serve Multi-Prompt `n>1` Batch from Chat Choices Cache

Hypothesis:

After per-prompt chat `n>1` requests have populated deterministic chat choices
cache entries, an equivalent multi-prompt batch with the same `n` can assemble
the whole batch response directly from those cached choice groups. For seeded
sampled requests, prompt `i` must probe the chat choices cache with base seed
`batch_seed + i * n`, because each chat choices entry then derives
`+ completion_index` internally.

Implementation:

- Replaced the single-prompt batch-to-chat-choices key builder in
  `crates/kiln-server/src/api/completions.rs` with a borrowed
  per-prompt helper,
  `deterministic_chat_choices_cache_key_from_batch_prompt_with_vocab_size()`.
  This avoids constructing an owned synthetic `ChatCompletionRequest` just to
  probe the cache.
- Added `batch_response_from_cached_chat_choice_groups()`, which maps one
  cached chat choices value per prompt into normal batch completion items,
  counting prompt tokens once per logical completion to preserve batch usage
  semantics.
- Added `batch_response_from_chat_choices_cache_hits()` before the existing
  `n=1` chat request-cache shortcut and before zero-token/generation fanout. It
  is restricted to `n > 1` and no adapters, probes all per-prompt chat choices
  keys under one lock, and only returns when every prompt is a full hit.
- A successful multi-prompt choices hit also populates/completes the whole-batch
  deterministic cache owner, so repeated identical batch requests use the
  batch cache.
- Added
  `multi_prompt_batch_hits_chat_choices_cache_before_fanout_work`, which warms
  two sampled chat `n=3` choices entries with seeds `700` and `703`, then
  submits the equivalent two-prompt batch with base seed `700`. The test asserts
  generated-token, render-cache, and token-cache counters do not move.

Validation commands:

- `cargo test -p kiln-server multi_prompt_batch_hits_chat_choices_cache_before_fanout_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server single_prompt_batch_hits_chat_choices_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server multi_prompt_batch_hits_chat_request_cache_before_fanout_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server deterministic_chat_choices_cache_key_normalizes_replayable_multi_choice_requests --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 48 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 61 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Background inference prewarm
completed in 11078 ms. E174 populated 16 distinct chat requests, each with
`n=4`, `temperature=0.7`, `top_p=0.9`, `max_tokens=0`, and per-prompt base
seeds `174 + i * 4`. E175 submitted the equivalent 16-prompt batch with `n=4`
and base seed `174`, for 64 logical completions.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E174 chat choices populate | 16 distinct chat requests, `n=4`, `max_tokens=0` | 0.014402 s total, 0.000900 s avg | 6.805 ms total, 0.4253 ms avg | 0 | 0 | 0 | render miss 16, token miss 16 |
| E175 batch from chat choices cache | 16 prompts, `n=4`, `max_tokens=0`, 64 outputs | 0.001223 s | 0.1718 ms | unchanged at 0 | unchanged at 0 | unchanged at 0 | render/token counters unchanged |

E174 returned 16 chat responses with 64 total choices and summed chat usage
`prompt_tokens=262`, `completion_tokens=0`, `total_tokens=262`. E175 returned
64 batch completions and aggregate batch usage `prompt_tokens=1048`,
`completion_tokens=0`, `total_tokens=1048`, which is exactly `262 * 4` prompt
tokens under batch's per-completion prompt accounting.

Metrics confirm the removed-work path:

- E174 increased request count from 0 to 16, added 16 rendered-prompt misses
  and 16 prompt-token misses, and left prefill/decode/generated-token counters
  at zero.
- E175 increased request count from 16 to 17, while rendered-prompt misses,
  prompt-token misses, prefill count, decode count, and generated tokens all
  remained unchanged.
- Request-duration sum moved from 0.006004 s after E174 to 0.006104 s after
  E175, a 0.000100 s increment for the 64-completion batch cache-hit response.

Artifacts:

- `e174_server.log`
- `e174_health_after_prewarm.json`
- `e174_before_metrics.prom`
- `e174_chat_16x4_zero_populate_requests.ndjson`
- `e174_chat_16x4_zero_populate_responses.ndjson`
- `e174_chat_16x4_zero_populate_times.log`
- `e174_after_chat_populate_metrics.prom`
- `e175_batch_16x4_from_chat_choices_request.json`
- `e175_batch_16x4_from_chat_choices_hit_response.json`
- `e175_batch_16x4_from_chat_choices_hit_time.log`
- `e175_after_hit_metrics.prom`

Takeaway:

The batch endpoint now removes the remaining cross-endpoint warm-path prompt
work for multi-prompt `n>1` batches when equivalent chat choices are already
cached. The measured 64-output batch hit was served from cached choice groups
with no additional prompt rendering, tokenization, prefill, decode, or token
generation.

## Experiment E176-E177: Multi-Prompt Batch Populates Per-Prompt Chat Choices

Hypothesis:

The previous experiment made a multi-prompt `n>1` batch consume per-prompt chat
choices cache entries. The inverse should also be true: a freshly produced
multi-prompt `n>1` batch can populate one chat choices cache entry per prompt,
so a later equivalent chat `n>1` request returns from the top-level choices
cache instead of looping through synthetic single-choice requests and probing
the lower-level chat request cache repeatedly.

Implementation:

- Replaced the old single-prompt-only batch response cache extractor with
  `chat_choices_cache_value_from_batch_items()` in
  `crates/kiln-server/src/api/completions.rs`.
- Added `store_chat_choices_cache_from_batch_response()`, which groups batch
  completions by prompt index, verifies each prompt has exactly `n` ordered
  completions with consistent per-completion prompt-token accounting, derives
  each per-prompt chat choices key using `batch_seed + prompt_index * n`, and
  inserts one deterministic chat choices value per prompt.
- Called the new store function on fresh zero-token and generated batch
  responses, while leaving hot batch-cache replay untouched so the already-hot
  batch path does not do extra storage work.
- Added
  `multi_prompt_batch_populates_chat_choices_cache_before_chat_work`, which
  generates a two-prompt sampled `n=3` batch, verifies two choices-cache entries
  were populated, then submits the equivalent chat `n=3` request for prompt 1
  and asserts generated-token, render-cache, token-cache, and choices-cache
  entry counts do not change.

Validation commands:

- `cargo test -p kiln-server multi_prompt_batch_populates_chat_choices_cache_before_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server single_prompt_batch_populates_chat_choices_cache_before_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server multi_prompt_batch_hits_chat_choices_cache_before_fanout_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server multi_prompt_batch_hits_chat_request_cache_before_fanout_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 49 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 62 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Background inference prewarm
completed in 16324 ms. E176 populated a 16-prompt batch with `n=4`,
`temperature=0.7`, `top_p=0.9`, `max_tokens=0`, and base seed `176`, for 64
logical completions. E177 submitted an equivalent chat `n=4` request for prompt
9 using seed `212` (`176 + 9 * 4`).

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E176 batch populate + per-prompt choices store | 16 prompts, `n=4`, `max_tokens=0`, 64 outputs | 0.006787 s | 5.7029 ms | 0 | 0 | 0 | render miss 16, token miss 16 |
| E177 chat from batch-populated choices | prompt 9, `n=4`, `max_tokens=0` | 0.005434 s | 0.1093 ms | unchanged at 0 | unchanged at 0 | unchanged at 0 | render/token counters unchanged |

E176 returned 64 batch completions with aggregate usage
`prompt_tokens=1112`, `completion_tokens=0`, `total_tokens=1112`. The prompt-9
per-completion prompt token count was 17. E177 returned 4 chat choices with
usage `prompt_tokens=17`, `completion_tokens=0`, `total_tokens=17`, matching
the per-prompt chat accounting.

Metrics confirm the removed-work path:

- E176 increased request count from 0 to 1, added 16 rendered-prompt misses and
  16 prompt-token misses, and left prefill/decode/generated-token counters at
  zero.
- E177 increased request count from 1 to 2, while rendered-prompt misses,
  prompt-token misses, prefill count, decode count, and generated tokens all
  remained unchanged.
- Request-duration sum moved from 0.005491 s after E176 to 0.005542 s after
  E177, a 0.000051 s increment for the chat choices-cache hit.

Artifacts:

- `e176_server.log`
- `e176_health_after_prewarm.json`
- `e176_before_metrics.prom`
- `e176_batch_16x4_zero_populate_request.json`
- `e176_batch_16x4_zero_populate_response.json`
- `e176_batch_16x4_zero_populate_time.log`
- `e176_after_batch_populate_metrics.prom`
- `e177_chat_prompt9_from_batch_choices_request.json`
- `e177_chat_prompt9_from_batch_choices_hit_response.json`
- `e177_chat_prompt9_from_batch_choices_hit_time.log`
- `e177_after_hit_metrics.prom`

Takeaway:

Fresh multi-prompt `n>1` batch responses now seed the per-prompt chat choices
cache. That removes another cross-endpoint warm-path loop: later equivalent chat
`n>1` requests no longer need to construct and probe one synthetic single
request per choice when the batch already produced the same choice group.

## Experiment E178-E179: Zero-Token Batch Populates Per-Prompt Chat Requests

Hypothesis:

Nonzero `n=1` batch fanout already goes through `generate_one_response()`, which
stores deterministic chat request-cache entries as part of producing each
synthetic response. The fast zero-token batch path builds responses directly and
therefore skipped that cross-endpoint cache population. A fresh zero-token
multi-prompt `n=1` batch should seed one deterministic chat request entry per
prompt, so later equivalent chat requests can return before prompt rendering and
tokenization.

Implementation:

- Added `chat_request_cache_value_from_batch_item()` in
  `crates/kiln-server/src/api/completions.rs` to convert one batch completion
  item into a deterministic chat request-cache value.
- Added `store_chat_request_cache_from_batch_response()`, restricted to
  `n=1` and no adapters. It verifies each prompt has exactly one
  `completion_index=0` item, derives the per-prompt chat request key with
  `batch_seed + prompt_index`, and inserts one request-cache value per prompt.
- Called the new store function only in the zero-token batch path before the
  existing choices-cache store. This avoids duplicating work in the generated
  `n=1` path, where `generate_one_response()` already populates request cache.
- Added
  `multi_prompt_zero_batch_populates_chat_request_cache_before_chat_work`,
  which submits a two-prompt zero-token batch, verifies two chat request-cache
  entries were populated, then submits an equivalent chat request for prompt 1
  and asserts generated-token, render-cache, token-cache, and request-cache
  entry counts do not change.

Validation commands:

- `cargo test -p kiln-server multi_prompt_zero_batch_populates_chat_request_cache_before_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_hits_request_cache_populated_by_single_output_batch --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server multi_prompt_batch_hits_chat_request_cache_before_fanout_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_single_output_hits_chat_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 50 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 63 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Background inference prewarm
completed in 11067 ms. E178 populated a 64-prompt batch with `n=1`,
`temperature=0.7`, `max_tokens=0`, and base seed `178`. E179 submitted an
equivalent chat request for prompt 37 using seed `215` (`178 + 37`).

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E178 zero batch populate + per-prompt request store | 64 prompts, `n=1`, `max_tokens=0` | 0.011937 s | 10.7204 ms | 0 | 0 | 0 | render miss 64, token miss 64 |
| E179 chat from batch-populated request cache | prompt 37, `max_tokens=0` | 0.002842 s | 1.3146 ms | unchanged at 0 | unchanged at 0 | unchanged at 0 | render/token counters unchanged |

E178 returned 64 batch completions with aggregate usage
`prompt_tokens=1142`, `completion_tokens=0`, `total_tokens=1142`. Prompt 37's
per-completion prompt token count was 18. E179 returned one chat choice with
usage `prompt_tokens=18`, `completion_tokens=0`, `total_tokens=18`, matching
the per-prompt chat accounting.

Metrics confirm the removed-work path:

- E178 increased request count from 0 to 1, added 64 rendered-prompt misses and
  64 prompt-token misses, and left prefill/decode/generated-token counters at
  zero.
- E179 increased request count from 1 to 2, while rendered-prompt misses,
  prompt-token misses, prefill count, decode count, and generated tokens all
  remained unchanged.
- Request-duration sum moved from 0.010487 s after E178 to 0.010708 s after
  E179, a 0.000221 s increment for the chat request-cache hit.

Artifacts:

- `e178_server.log`
- `e178_health_after_prewarm.json`
- `e178_before_metrics.prom`
- `e178_batch_64_zero_populate_request.json`
- `e178_batch_64_zero_populate_response.json`
- `e178_batch_64_zero_populate_time.log`
- `e178_after_batch_populate_metrics.prom`
- `e179_chat_prompt37_from_batch_request_cache_request.json`
- `e179_chat_prompt37_from_batch_request_cache_hit_response.json`
- `e179_chat_prompt37_from_batch_request_cache_hit_time.log`
- `e179_after_hit_metrics.prom`

Takeaway:

The zero-token multi-prompt `n=1` batch path now seeds per-prompt chat request
cache entries. Later equivalent chat requests no longer need to repeat prompt
rendering or tokenization after the batch already computed the prompt-token
accounting.

## Experiment E180-E181: Zero-Token `n>1` Batch Populates Chat Request Cache

Hypothesis:

E178-E179 covered zero-token multi-prompt `n=1` batches. The same direct
zero-token path also handles `n>1`, but only populated chat choices entries.
Because `max_tokens=0` deterministic request keys ignore sampling seed, one
request-cache entry per prompt is enough to serve later equivalent `n=1` chat
requests. The store path should therefore derive the normal per-output seed
`batch_seed + prompt_index * n + completion_index`, then de-duplicate normalized
keys so zero-token `n>1` batches do not redundantly insert the same per-prompt
request entry for every logical completion.

Implementation:

- Generalized `store_chat_request_cache_from_batch_response()` in
  `crates/kiln-server/src/api/completions.rs` from `n=1` to any `n >= 1`.
- The helper now groups items by prompt, validates each prompt has exactly `n`
  ordered completions, derives the per-output chat request key with
  `batch_seed + prompt_index * n + completion_index`, and de-duplicates
  normalized keys before inserting request-cache values.
- For zero-token `n>1`, deterministic key normalization collapses each prompt's
  logical completions to one request-cache entry because seed and sampling
  fields are ignored when no tokens are generated.
- Added
  `multi_output_zero_batch_populates_chat_request_cache_before_single_chat_work`,
  which submits a two-prompt zero-token `n=3` batch, verifies one request-cache
  entry and one choices-cache entry per prompt, then submits an equivalent
  single chat request and asserts generated-token, render-cache, token-cache,
  and request-cache counts do not change.

Validation commands:

- `cargo test -p kiln-server multi_output_zero_batch_populates_chat_request_cache_before_single_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server multi_prompt_zero_batch_populates_chat_request_cache_before_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server repeated_multi_output_zero_batch_hits_batch_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server multi_prompt_batch_populates_chat_choices_cache_before_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 51 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 64 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Background inference prewarm
completed in 15877 ms. E180 populated a 16-prompt batch with `n=4`,
`temperature=0.7`, `top_p=0.9`, `max_tokens=0`, and base seed `180`, for 64
logical completions. E181 submitted an equivalent `n=1` chat request for prompt
9 using seed `218` (`180 + 9 * 4 + 2`; seed is normalized away for
`max_tokens=0`).

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E180 zero batch `n=4` populate + request store | 16 prompts, `n=4`, `max_tokens=0`, 64 outputs | 0.018893 s | 12.0640 ms | 0 | 0 | 0 | render miss 16, token miss 16 |
| E181 single chat from batch-populated request cache | prompt 9, `max_tokens=0` | 0.000802 s | 0.0893 ms | unchanged at 0 | unchanged at 0 | unchanged at 0 | render/token counters unchanged |

E180 returned 64 batch completions with aggregate usage
`prompt_tokens=1176`, `completion_tokens=0`, `total_tokens=1176`. Prompt 9's
per-completion prompt token count was 18. E181 returned one chat choice with
usage `prompt_tokens=18`, `completion_tokens=0`, `total_tokens=18`, matching
the per-prompt chat accounting.

Metrics confirm the removed-work path:

- E180 increased request count from 0 to 1, added 16 rendered-prompt misses and
  16 prompt-token misses, and left prefill/decode/generated-token counters at
  zero.
- E181 increased request count from 1 to 2, while rendered-prompt misses,
  prompt-token misses, prefill count, decode count, and generated tokens all
  remained unchanged.
- Request-duration sum moved from 0.011821 s after E180 to 0.011860 s after
  E181, a 0.000039 s increment for the chat request-cache hit.

Artifacts:

- `e180_server.log`
- `e180_health_after_prewarm.json`
- `e180_before_metrics.prom`
- `e180_batch_16x4_zero_populate_request.json`
- `e180_batch_16x4_zero_populate_response.json`
- `e180_batch_16x4_zero_populate_time.log`
- `e180_after_batch_populate_metrics.prom`
- `e181_chat_prompt9_from_zero_n_batch_request_cache_request.json`
- `e181_chat_prompt9_from_zero_n_batch_request_cache_hit_response.json`
- `e181_chat_prompt9_from_zero_n_batch_request_cache_hit_time.log`
- `e181_after_hit_metrics.prom`

Takeaway:

The zero-token multi-output batch path now seeds both per-prompt chat choices
and per-prompt chat request caches. A later single-chat request can reuse the
batch's prompt accounting without rendering or tokenizing again, even when the
batch originally requested multiple logical completions per prompt.

## Experiment E182-E183: Zero-Token Chat n Populates Request Cache

Hypothesis:

The chat-side `n>1` zero-token path already builds one response choice per
logical completion and then stores the aggregate choices response. Because
`max_tokens=0` normalizes sampling away, each of those choices can also seed
the deterministic single-request chat cache. That lets a later equivalent
`n=1` zero-token chat return before rendering or tokenizing the prompt.

Change:

- Added
  `deterministic_chat_request_cache_key_from_chat_choice_with_vocab_size`, a
  chat request-cache key builder that accepts the per-choice derived seed
  instead of rejecting requests where `n > 1`.
- Added `chat_request_cache_value_from_choice` and
  `store_chat_request_cache_from_chat_choices_response`.
  The store is zero-token-only, rejects adapter requests, derives
  `req.seed + choice.index`, de-duplicates normalized keys, and inserts the
  single-choice-compatible response value.
- Updated the chat `n_per > 1` path to store request-cache entries from the
  generated choices response before finishing the chat choices-cache store.
- Added
  `multi_choice_zero_chat_populates_request_cache_before_single_chat_work`,
  which submits a zero-token `n=4` chat request, verifies one request-cache
  entry and one choices-cache entry, then submits an equivalent single chat
  request with different sampling parameters and seed and asserts render,
  token, generated-token, request-cache, and choices-cache counters do not
  change.

Validation commands:

- `cargo test -p kiln-server multi_choice_zero_chat_populates_request_cache_before_single_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server repeated_zero_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_zero_max_tokens_returns_without_generation --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_multi_choice_repeated_seeded_sampled_hits_top_level_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 65 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 51 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 8 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Tokenizer warmup completed in 0
ms. Metal custom kernels were precompiled during background prewarm.
Background inference prewarm completed in 13820 ms. E182 submitted one chat
request with `n=4`, `temperature=0.7`, `top_p=0.9`, `max_tokens=0`, and seed
`182`. E183 submitted an equivalent `n=1` chat request for the same prompt with
`temperature=0.2`, `top_p=0.1`, `max_tokens=0`, and seed `999`; with zero
output tokens those sampling differences normalize to the same request-cache
key.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E182 chat `n=4` populate + request store | 1 prompt, `n=4`, `max_tokens=0`, 4 outputs | 0.003506 s | 2.397 ms | 0 | 0 | 0 | render miss 1, token miss 1 |
| E183 single chat from chat-populated request cache | same prompt, `n=1`, `max_tokens=0` | 0.001416 s | 0.069 ms | unchanged at 0 | unchanged at 0 | unchanged at 0 | render/token counters unchanged |

E182 returned four chat choices with usage `prompt_tokens=16`,
`completion_tokens=0`, `total_tokens=16`. E183 returned one chat choice with
the same usage and `finish_reason="length"`.

Metrics confirm the removed-work path:

- E182 increased request count from 0 to 1, added one rendered-prompt miss and
  one prompt-token miss, and left prefill, decode, and generated-token counters
  at zero.
- E183 increased request count from 1 to 2, while rendered-prompt misses,
  prompt-token misses, prefill count, decode count, and generated tokens all
  remained unchanged.
- Request-duration sum moved from 0.002203 s after E182 to 0.002225 s after
  E183, a 0.000022 s increment for the request-cache hit. The request span log
  reported 0.069 ms for the same hit.

Artifacts:

- `e182_server.log`
- `e182_health_after_prewarm.json`
- `e182_before_metrics.prom`
- `e182_chat_n4_zero_populate_request.json`
- `e182_chat_n4_zero_populate_response.json`
- `e182_chat_n4_zero_populate_time.log`
- `e182_after_chat_populate_metrics.prom`
- `e183_chat_single_from_chat_n_request_cache_request.json`
- `e183_chat_single_from_chat_n_request_cache_hit_response.json`
- `e183_chat_single_from_chat_n_request_cache_hit_time.log`
- `e183_after_hit_metrics.prom`

Takeaway:

The zero-token multi-choice chat path now seeds the deterministic single-chat
request cache directly from its generated choices. A later single-choice
zero-token chat can skip render/token work and model work even when the warmup
request used `n>1` and different sampling parameters.

## Experiment E184-E186: Streaming Completion-Cache Hit Promotes Request Cache

Hypothesis:

The streaming `n=1` path can replay a deterministic lower-level completion
cache entry after rendering and tokenizing the prompt. Before this experiment,
that early return did not finish the already-claimed chat request-cache entry.
So a later equivalent chat could avoid model work but still pay prompt
render/token work again. Promoting the streamed completion-cache hit into the
chat request cache should remove that future prompt work.

Change:

- Added `chat_request_cache_value_from_completion` and
  `finish_chat_request_cache_value`.
- Updated the streaming completion-cache `Hit` and `Wait` branches to build a
  `DeterministicChatRequestCacheValue`, complete/insert the chat request-cache
  entry, and then stream from that promoted value.
- Added
  `chat_streaming_completion_cache_hit_populates_request_cache_before_prompt_work`,
  which seeds only the lower completion cache, submits a streaming chat request
  that must render/token before hitting that lower cache, and then verifies the
  following non-streaming chat returns before render/token lookup.

Validation commands:

- `cargo test -p kiln-server chat_streaming_completion_cache_hit_populates_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server chat_streaming_repeated_greedy_request_uses_completion_cache --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server repeated_zero_chat_hits_request_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 9 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 66 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Tokenizer warmup completed in 0
ms. Metal custom kernels were precompiled during background prewarm.
Background inference prewarm completed in 12050 ms. E184 submitted one greedy
non-streaming chat request for prompt `"stream lower cache promotion live"` with
`max_tokens=2` and seed `184`, populating both completion and chat request
caches. Then 129 distinct zero-token chat requests evicted the 128-entry chat
request cache without adding lower completion-cache entries. E185 submitted the
same prompt as `stream=true`, forcing a chat request-cache miss followed by a
lower completion-cache hit. E186 submitted the same prompt non-streaming to
verify the E185 promotion.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E184 populate lower completion cache | bs=1 greedy chat, `max_tokens=2` | 6.265038 s | 6263.716 ms | 4.469469 s | 1.739725 s | 2 | render miss 1, token miss 1 |
| 129 zero-token eviction fillers | distinct chats, `max_tokens=0` | 0.053323 s aggregate handler | ~0.413 ms avg | unchanged | unchanged | unchanged at 2 | render/token misses to 130 |
| E185 stream from lower completion cache + promote | same prompt, `stream=true` | 0.000615 s | 0.0665 ms | unchanged | unchanged | unchanged at 2 | render hit +1, token hit +1 |
| E186 hit promoted request cache | same prompt, non-streaming | 0.000567 s | 0.0593 ms | unchanged | unchanged | unchanged at 2 | render/token counters unchanged |

E184 returned `reasoning_content="Okay,"`, `content=""`,
`finish_reason="length"`, and usage `prompt_tokens=15`,
`completion_tokens=2`, `total_tokens=17`. E185 streamed the same
`reasoning_content` from cache and `[DONE]`. E186 returned the same
non-streaming response and usage as E184.

Metrics confirm the removed-work path:

- After E184: request count 1, prefill count 1, decode count 1, generated
  tokens 2, rendered/token misses 1 each.
- After the 129 eviction requests: request count 130, generated tokens still 2,
  rendered/token misses 130 each. This evicts the original chat request entry
  while keeping the lower completion entry.
- E185 increased request count to 131, request-duration sum by 0.000027 s,
  rendered/token hits by one each, and left prefill, decode, and generated
  tokens unchanged.
- E186 increased request count to 132, request-duration sum by 0.000017 s, and
  left rendered/token hits, rendered/token misses, prefill, decode, and
  generated tokens unchanged. That proves E185 promoted the lower-cache stream
  replay into a top-level request-cache hit.

Artifacts:

- `e184_server.log`
- `e184_health_after_prewarm.json`
- `e184_before_metrics.prom`
- `e184_chat_populate_request.json`
- `e184_chat_populate_response.json`
- `e184_chat_populate_time.log`
- `e184_after_populate_metrics.prom`
- `e184_evict_request.json`
- `e184_evict_chat_request_cache_times.log`
- `e184_after_eviction_metrics.prom`
- `e185_chat_stream_from_completion_cache_request.json`
- `e185_chat_stream_from_completion_cache_response.sse`
- `e185_chat_stream_from_completion_cache_time.log`
- `e185_after_stream_metrics.prom`
- `e186_chat_after_stream_promotion_request.json`
- `e186_chat_after_stream_promotion_hit_response.json`
- `e186_chat_after_stream_promotion_hit_time.log`
- `e186_after_hit_metrics.prom`

Takeaway:

A streaming replay from the lower deterministic completion cache now becomes a
future top-level chat request-cache hit. The first stream still pays prompt
render/token work if the request cache was evicted, but subsequent equivalent
chat requests skip render, tokenization, prefill, decode, and token generation.

## Experiment E187-E192: Batch Cache Hit Rehydrates Chat Caches

Hypothesis:

The whole-batch deterministic cache can return a response before prompt
rendering/tokenization. Before this experiment, that hot batch-cache path did
not repopulate per-prompt chat request or chat choices caches. If those smaller
chat caches were evicted while the batch cache remained hot, later chat
requests still had to miss the top-level chat cache and do prompt work. A batch
cache hit has enough stored per-item prompt usage and completion payloads to
rehydrate the per-prompt chat caches without rendering or tokenizing.

Change:

- Added `store_chat_caches_from_batch_response`, which calls both existing
  batch-to-chat request-cache and choices-cache stores.
- Updated whole-batch cache `Hit` and `Wait` branches to rehydrate chat caches
  from the cached batch response before returning.
- Added `batch_cache_hit_rehydrates_chat_request_cache_before_chat_work`, which
  clears the request cache after a zero-token `n=1` batch populate, verifies an
  identical batch-cache hit rehydrates request-cache entries without prompt
  work, then verifies a matching chat hits before render/token lookup.
- Added `batch_cache_hit_rehydrates_chat_choices_cache_before_chat_n_work`,
  which performs the same check for zero-token `n=3` batch choices and a later
  chat `n=3` request.

Validation commands:

- `cargo test -p kiln-server batch_cache_hit_rehydrates_chat_request_cache_before_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_cache_hit_rehydrates_chat_choices_cache_before_chat_n_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server repeated_multi_output_zero_batch_hits_batch_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 53 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 68 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 9 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Tokenizer warmup completed in 0
ms. Metal custom kernels were precompiled during background prewarm.
Background inference prewarm completed in 13837 ms.

Request-cache rehydration:

E187 submitted a 64-prompt zero-token batch with `n=1`, `temperature=0.7`,
`top_p=0.9`, and seed `187`. Then 129 distinct zero-token chat requests
evicted the 128-entry chat request cache while leaving the 64-entry batch cache
hot. E188 submitted the identical 64-prompt batch to hit the whole-batch cache
and rehydrate the per-prompt chat request cache. E189 submitted prompt 37 as a
single chat request with different sampling fields that normalize away for
`max_tokens=0`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E187 populate batch/request caches | 64 prompts, `n=1`, `max_tokens=0` | 0.065687 s | 65.040 ms | 0 | 0 | 0 | render miss 64, token miss 64 |
| 129 request-cache eviction fillers | distinct zero-token chats | 0.103639 s aggregate wall | 0.803 ms avg wall | unchanged | unchanged | unchanged at 0 | render/token misses to 193 |
| E188 whole-batch cache hit + request rehydrate | same 64 prompts | 0.001250 s | 0.530 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |
| E189 chat from rehydrated request cache | prompt 37, `max_tokens=0` | 0.000489 s | 0.055 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |

E187 and E188 both returned 64 completions with usage
`prompt_tokens=1142`, `completion_tokens=0`, `total_tokens=1142`. E189
returned one chat choice with usage `prompt_tokens=18`, `completion_tokens=0`,
`total_tokens=18`, matching prompt 37's per-item batch usage.

Choices-cache rehydration:

E190 submitted a 16-prompt zero-token batch with `n=4`, `temperature=0.7`,
`top_p=0.9`, and seed `190`, producing 64 logical completions. Then 65
distinct zero-token chat `n=4` requests evicted the 64-entry chat choices
cache while keeping the batch cache hot. E191 submitted the identical 16-prompt
batch to hit the whole-batch cache and rehydrate per-prompt chat choices.
E192 submitted prompt 9 as chat `n=4` with different sampling fields that
normalize away for `max_tokens=0`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E190 populate batch/choices caches | 16 prompts, `n=4`, `max_tokens=0` | 0.005334 s | 4.827 ms | 0 | 0 | 0 | render/token misses to 209 |
| 65 choices-cache eviction fillers | distinct chat `n=4`, `max_tokens=0` | 0.046248 s aggregate wall | 0.712 ms avg wall | unchanged | unchanged | unchanged at 0 | render/token misses to 274 |
| E191 whole-batch cache hit + choices rehydrate | same 16 prompts, 64 outputs | 0.000726 s | 0.158 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |
| E192 chat from rehydrated choices cache | prompt 9, `n=4`, `max_tokens=0` | 0.000530 s | 0.053 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |

E190 and E191 both returned 64 completions with usage
`prompt_tokens=1112`, `completion_tokens=0`, `total_tokens=1112`. E192
returned four chat choices with usage `prompt_tokens=17`,
`completion_tokens=0`, `total_tokens=17`, matching prompt 9's per-choice batch
usage.

Metrics confirm the removed-work paths:

- E188 increased request count from 130 to 131 and request-duration sum by
  0.000463 s while rendered-prompt misses, prompt-token misses, prefill count,
  decode count, and generated tokens stayed unchanged.
- E189 increased request count from 131 to 132 and request-duration sum by
  0.000019 s while the same render/token/model counters stayed unchanged.
- E191 increased request count from 198 to 199 and request-duration sum by
  0.000109 s while rendered-prompt misses, prompt-token misses, prefill count,
  decode count, and generated tokens stayed unchanged.
- E192 increased request count from 199 to 200 and request-duration sum by
  0.000017 s while the same render/token/model counters stayed unchanged.

Artifacts:

- `e187_server.log`
- `e187_health_after_prewarm.json`
- `e187_before_metrics.prom`
- `e187_batch64_zero_populate_request.json`
- `e187_batch64_zero_populate_response.json`
- `e187_batch64_zero_populate_time.log`
- `e187_after_populate_metrics.prom`
- `e187_evict_request.json`
- `e187_evict_chat_request_cache_times.log`
- `e187_after_request_eviction_metrics.prom`
- `e188_batch64_zero_rehydrate_request.json`
- `e188_batch64_zero_rehydrate_response.json`
- `e188_batch64_zero_rehydrate_time.log`
- `e188_after_rehydrate_metrics.prom`
- `e189_chat_prompt37_from_rehydrated_request_cache_request.json`
- `e189_chat_prompt37_from_rehydrated_request_cache_hit_response.json`
- `e189_chat_prompt37_from_rehydrated_request_cache_hit_time.log`
- `e189_after_hit_metrics.prom`
- `e190_batch16x4_zero_populate_request.json`
- `e190_batch16x4_zero_populate_response.json`
- `e190_batch16x4_zero_populate_time.log`
- `e190_after_populate_choices_metrics.prom`
- `e190_evict_choices_request.json`
- `e190_evict_chat_choices_cache_times.log`
- `e190_after_choices_eviction_metrics.prom`
- `e191_batch16x4_zero_rehydrate_choices_request.json`
- `e191_batch16x4_zero_rehydrate_choices_response.json`
- `e191_batch16x4_zero_rehydrate_choices_time.log`
- `e191_after_rehydrate_choices_metrics.prom`
- `e192_chat_prompt9_from_rehydrated_choices_cache_request.json`
- `e192_chat_prompt9_from_rehydrated_choices_cache_hit_response.json`
- `e192_chat_prompt9_from_rehydrated_choices_cache_hit_time.log`
- `e192_after_hit_metrics.prom`

Takeaway:

Whole-batch cache hits now restore the smaller per-prompt chat caches they can
prove from cached batch items. That keeps endpoint-crossing reuse alive even
after chat cache eviction: a hot batch hit can make later bs=1 or chat `n>1`
requests skip render, tokenization, prefill, decode, and generation again.

## Experiment E193-E198: Batch From Choices Rehydrates Request Cache

Hypothesis:

A batch response synthesized from chat choices-cache hits can return before any
prompt work and already contains the per-choice text, reasoning content,
finish reason, completion-token count, and prompt-token count needed to seed
single-chat request-cache entries. Before this experiment, those batch-from-
choices paths populated the whole-batch cache but did not restore the
single-chat request cache. If the request cache was evicted while the choices
cache remained hot, later `n=1` chats still had to render/tokenize. Rehydrating
request-cache entries from the synthesized batch response removes that future
work.

Change:

- Updated the one-prompt chat choices-cache batch path to call
  `store_chat_request_cache_from_batch_response` before returning.
- Updated the multi-prompt per-prompt chat choices-cache batch path to do the
  same.
- Added
  `single_prompt_batch_from_choices_cache_rehydrates_request_cache_before_single_chat_work`.
- Added
  `multi_prompt_batch_from_choices_cache_rehydrates_request_cache_before_chat_work`.

Validation commands:

- `cargo test -p kiln-server single_prompt_batch_from_choices_cache_rehydrates_request_cache_before_single_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server multi_prompt_batch_from_choices_cache_rehydrates_request_cache_before_chat_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server single_prompt_batch_hits_chat_choices_cache_before_prompt_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server multi_prompt_batch_hits_chat_choices_cache_before_fanout_work --lib`
  - Result: 1 passed.
- `cargo test -p kiln-server batch_ --lib`
  - Result: 55 passed.
- `cargo test -p kiln-server chat_ --lib`
  - Result: 70 passed.
- `cargo test -p kiln-server completion_cache --lib`
  - Result: 9 passed.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Result: passed.
- `cargo build --release --features metal --bin kiln`
  - Result: passed.

Real server result:

A fresh release server was started on port 8421 with
`Qwen/Qwen3.5-4B` snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Tokenizer warmup completed in 0
ms. Metal custom kernels were precompiled during background prewarm.
Background inference prewarm completed in 11521 ms.

Single-prompt choices-to-request rehydration:

E193 submitted one zero-token chat request with `n=4`, `temperature=0.7`,
`top_p=0.9`, and seed `193`, populating both chat choices and request caches.
Then 129 distinct zero-token `n=1` chat requests evicted the 128-entry request
cache without touching the choices cache. E194 submitted the equivalent
one-prompt batch `n=4`, hitting the chat choices cache and rehydrating the
single-chat request cache. E195 submitted the same prompt as `n=1` chat with
different sampling fields that normalize away for `max_tokens=0`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E193 chat choices populate | 1 prompt, chat `n=4`, `max_tokens=0` | 0.002454 s | 1.934 ms | 0 | 0 | 0 | render miss 1, token miss 1 |
| 129 request-cache eviction fillers | distinct chat `n=1`, `max_tokens=0` | 0.095646 s aggregate wall | 0.741 ms avg wall | unchanged | unchanged | unchanged at 0 | render/token misses to 130 |
| E194 batch from choices + request rehydrate | 1 prompt, batch `n=4`, 4 outputs | 0.000547 s | 0.086 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |
| E195 single chat from rehydrated request cache | same prompt, `n=1`, `max_tokens=0` | 0.000482 s | 0.048 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |

E193 returned four chat choices with usage `prompt_tokens=17`,
`completion_tokens=0`, `total_tokens=17`. E194 returned four batch completions
with aggregate usage `prompt_tokens=68`, `completion_tokens=0`,
`total_tokens=68`. E195 returned one chat choice with usage `prompt_tokens=17`,
`completion_tokens=0`, `total_tokens=17`.

Multi-prompt choices-to-request rehydration:

E196 submitted 16 zero-token chat `n=4` requests with derived seeds
`196 + prompt_index * 4`, populating one choices-cache entry and one
request-cache entry per prompt. Then 129 distinct zero-token `n=1` chat
requests evicted the request cache without touching the choices cache. E197
submitted the equivalent 16-prompt batch `n=4`, hitting per-prompt choices
caches and rehydrating per-prompt request-cache entries. E198 submitted prompt
9 as `n=1` chat with sampling fields that normalize away for `max_tokens=0`.

| Experiment | Shape | Wall time | HTTP handler | Request prefill | Request decode | Physical generated tokens | Front-end work |
|---|---|---:|---:|---:|---:|---:|---|
| E196 chat choices populate | 16 chat requests, `n=4`, `max_tokens=0` | 0.013299 s aggregate wall | ~0.831 ms avg wall | 0 | 0 | 0 | render/token misses to 146 |
| 129 request-cache eviction fillers | distinct chat `n=1`, `max_tokens=0` | 0.090151 s aggregate wall | 0.699 ms avg wall | unchanged | unchanged | unchanged at 0 | render/token misses to 275 |
| E197 batch from choices + request rehydrate | 16 prompts, `n=4`, 64 outputs | 0.000633 s | 0.151 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |
| E198 single chat from rehydrated request cache | prompt 9, `max_tokens=0` | 0.000497 s | 0.047 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |

E197 returned 64 batch completions with aggregate usage
`prompt_tokens=1176`, `completion_tokens=0`, `total_tokens=1176`. E198
returned one chat choice with usage `prompt_tokens=18`,
`completion_tokens=0`, `total_tokens=18`, matching prompt 9's per-prompt
accounting.

Metrics confirm the removed-work paths:

- E194 increased request count from 130 to 131 and request-duration sum by
  0.000041 s while rendered-prompt misses, prompt-token misses, prefill count,
  decode count, and generated tokens stayed unchanged.
- E195 increased request count from 131 to 132 and request-duration sum by
  0.000014 s while the same render/token/model counters stayed unchanged.
- E197 increased request count from 277 to 278 and request-duration sum by
  0.000108 s while rendered-prompt misses, prompt-token misses, prefill count,
  decode count, and generated tokens stayed unchanged.
- E198 increased request count from 278 to 279 and request-duration sum by
  0.000014 s while the same render/token/model counters stayed unchanged.

Artifacts:

- `e193_server.log`
- `e193_health_after_prewarm.json`
- `e193_before_metrics.prom`
- `e193_chat_n4_zero_populate_choices_request.json`
- `e193_chat_n4_zero_populate_choices_response.json`
- `e193_chat_n4_zero_populate_choices_time.log`
- `e193_after_populate_metrics.prom`
- `e193_evict_request_cache_request.json`
- `e193_evict_request_cache_times.log`
- `e193_after_request_eviction_metrics.prom`
- `e194_batch_single_from_choices_rehydrate_request.json`
- `e194_batch_single_from_choices_rehydrate_response.json`
- `e194_batch_single_from_choices_rehydrate_time.log`
- `e194_after_rehydrate_metrics.prom`
- `e195_chat_single_from_choices_rehydrated_request_cache_request.json`
- `e195_chat_single_from_choices_rehydrated_request_cache_hit_response.json`
- `e195_chat_single_from_choices_rehydrated_request_cache_hit_time.log`
- `e195_after_hit_metrics.prom`
- `e196_chat_16x4_zero_populate_choices_times.log`
- `e196_chat_16x4_zero_populate_choices_request_*.json`
- `e196_chat_16x4_zero_populate_choices_response_*.json`
- `e196_after_populate_metrics.prom`
- `e196_evict_request_cache_request.json`
- `e196_evict_request_cache_times.log`
- `e196_after_request_eviction_metrics.prom`
- `e197_batch_16x4_from_choices_rehydrate_request.json`
- `e197_batch_16x4_from_choices_rehydrate_response.json`
- `e197_batch_16x4_from_choices_rehydrate_time.log`
- `e197_after_rehydrate_metrics.prom`
- `e198_chat_prompt9_from_choices_rehydrated_request_cache_request.json`
- `e198_chat_prompt9_from_choices_rehydrated_request_cache_hit_response.json`
- `e198_chat_prompt9_from_choices_rehydrated_request_cache_hit_time.log`
- `e198_after_hit_metrics.prom`

Takeaway:

Batch responses synthesized from chat choices-cache hits now restore the
single-chat request cache too. That means a hot `n>1` choices cache can serve a
batch and then make later `n=1` chats skip render, tokenization, prefill,
decode, and generation, even after the request cache had been evicted.

## Experiment E199-E201: Chat Choices Hit Rehydrates Request Cache

Hypothesis:

E182-E183 made fresh chat `n>1` zero-token responses seed the normalized
single-chat request cache. E193-E198 made batch responses synthesized from chat
choices-cache hits do the same. A direct chat `n>1` choices-cache hit still
returned before prompt work without restoring the single-chat request cache. If
the request cache had been evicted while the choices cache was still hot, a
later equivalent `n=1` chat had to render and tokenize again. The direct
choices-cache hit can prove the same per-choice responses, so it should
rehydrate those request-cache entries before returning.

Change:

- Updated the chat `n>1` choices-cache `Hit` and completed `Wait` branches to
  call `store_chat_request_cache_from_chat_choices_response` before returning.
- Added `chat_choices_cache_hit_rehydrates_request_cache_before_single_chat_work`,
  which clears the request cache after a chat choices-cache populate, verifies
  the repeated `n>1` chat returns before render/token work, and verifies the
  following `n=1` chat also returns before render/token work from the
  rehydrated request cache.

Validation:

- `cargo test -p kiln-server chat_choices_cache_hit_rehydrates_request_cache_before_single_chat_work --lib`
  - Passed: 1 test.
- `cargo test -p kiln-server chat_ --lib`
  - Passed: 71 tests.
- `cargo test -p kiln-server batch_ --lib`
  - Passed: 55 tests.
- `cargo test -p kiln-server completion_cache --lib`
  - Passed: 9 tests.
- `cargo check --locked -p kiln-server --features metal --bin kiln --bin kiln-bench`
  - Passed.
- `cargo build --release --features metal --bin kiln`
  - Passed.

Real server result:

A fresh Qwen3.5-4B Metal server was started on port 8421 with the patched
release binary. E199 populated the chat choices cache with a one-prompt
`n=4`, `max_tokens=0` request. 129 distinct zero-token `n=1` chat requests
then evicted the 128-entry request cache without touching the choices cache.
E200 repeated the `n=4` chat request, hit the choices cache, and rehydrated the
normalized request-cache entry. E201 sent the equivalent `n=1` chat with
different zero-token sampling parameters and hit the rehydrated request cache.

| Run | Request | Wall time | Handler duration | Prefill count | Decode count | Generated tokens | Render/token counters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E199 chat choices populate | 1 prompt, chat `n=4`, `max_tokens=0` | 0.039734 s | 38.831 ms | 0 | 0 | 0 | render miss 1, token miss 1 |
| 129 request-cache eviction fillers | distinct chat `n=1`, `max_tokens=0` | 0.115272 s aggregate wall | 0.894 ms avg wall | unchanged | unchanged | unchanged at 0 | render/token misses to 130 |
| E200 choices-cache hit + request rehydrate | same prompt, chat `n=4`, 4 outputs | 0.000671 s | 0.060 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |
| E201 single chat from rehydrated request cache | same prompt, chat `n=1`, `max_tokens=0` | 0.000699 s | 0.051 ms | unchanged | unchanged | unchanged at 0 | render/token counters unchanged |

E199 and E200 both returned four choices with usage `prompt_tokens=18`,
`completion_tokens=0`, `total_tokens=18`. E201 returned one choice with the
same usage and the same first choice payload.

Metrics confirm the removed-work path:

- E200 increased request count from 130 to 131 while rendered-prompt misses,
  prompt-token misses, prefill count, decode count, and generated tokens stayed
  unchanged.
- E201 increased request count from 131 to 132 while the same render/token/model
  counters stayed unchanged.

Artifacts:

- `e199_server.log`
- `e199_health_after_prewarm.json`
- `e199_before_metrics.prom`
- `e199_chat_n4_zero_populate_choices_request.json`
- `e199_chat_n4_zero_populate_choices_response.json`
- `e199_chat_n4_zero_populate_choices_time.log`
- `e199_after_populate_metrics.prom`
- `e200_evict_request_cache_request.json`
- `e200_evict_request_cache_times.log`
- `e200_after_request_eviction_metrics.prom`
- `e200_chat_n4_zero_rehydrate_request.json`
- `e200_chat_n4_zero_rehydrate_response.json`
- `e200_chat_n4_zero_rehydrate_time.log`
- `e200_after_rehydrate_metrics.prom`
- `e201_chat_single_from_choices_hit_rehydrated_request_cache_request.json`
- `e201_chat_single_from_choices_hit_rehydrated_request_cache_hit_response.json`
- `e201_chat_single_from_choices_hit_rehydrated_request_cache_hit_time.log`
- `e201_after_hit_metrics.prom`

Takeaway:

Direct chat `n>1` choices-cache hits now restore the single-chat request cache,
not just return a fast multi-choice response. That keeps the zero-work chain
alive after request-cache eviction: chat `n>1` hit, then chat `n=1` hit, with
no render, tokenization, prefill, decode, or generation in either request.

## Experiment E202: Single-Chat Request Cache Feeds Zero-Token Chat Choices

Hypothesis:

The previous endpoint-cache work closed the direction where hot chat `n>1`
choices entries can restore `n=1` request-cache entries. The reverse zero-token
case was still missing: a hot normalized single-chat request-cache entry proves
the empty completion for a later zero-token chat `n>1` request, but the old path
entered `generate_multi_chat_response`, rendered the prompt, and tokenized
before producing cloned empty choices.

Change:

- Added `zero_chat_choices_response_from_request_cache_hit`, which probes the
  normalized single-chat request cache for zero-token chat `n>1` requests before
  prompt rendering/tokenization.
- On a hit or completed wait, it synthesizes the multi-choice response and
  finishes the chat choices cache owner so repeated `n>1` requests hit the
  top-level choices cache too.
- Added `multi_choice_zero_chat_hits_request_cache_before_prompt_work`, which
  starts with a hot `n=1` zero-token chat request cache, then verifies the
  later `n=4` chat returns before rendered-prompt and prompt-token cache lookup
  and seeds the choices cache.

Validation:

- `cargo test -p kiln-server multi_choice_zero_chat_hits_request_cache_before_prompt_work --lib`
  - Passed: 1 test.
- `cargo test -p kiln-server chat_ --lib`
  - Passed: 72 tests.

Live result:

Skipped for E202. This is the final endpoint-cache symmetry cleanup before
returning to low-level Metal/kernel work; E199-E201 already measured the same
class of zero-token request-cache/choices-cache removed-work path on a real
Qwen3.5-4B Metal server.

Takeaway:

The zero-token endpoint-cache graph is now closed in both chat directions:
`n=1` can feed `n>1`, and `n>1` can feed `n=1`, without render, tokenization,
prefill, decode, or generation on the later request. The next optimization
work should shift back to low-level kernel/profile evidence, since cache reuse
is not the whole fastest-inference objective.

## Experiment E203-E214: Warmed Low-Level Metal Fused-QKV Dispatch

Hypothesis:

The next speed work needs to get back under endpoint caches. The existing
`kiln-bench --paged --latency-only` p64/o16 numbers were not reliable for
kernel decisions because the first timed prefill included first-use Metal and
Candle compilation. The fused QKV decode projection also looked suspect: the
old kernel launched a rectangular 2D grid sized to the largest of Q/K/V, so
Qwen3.5-4B full-attention decode spent half the fused-QKV threadgroups on
immediate K/V returns (`128 * 3 = 384` groups instead of the exact
`128 + 32 + 32 = 192` groups).

Change:

- Added `--latency-warmup-runs <n>` to `kiln-bench`, which runs throwaway
  latency passes before the measured run. This preserves the default benchmark
  behavior while allowing kernel A/B runs that exclude first-use compilation
  from the reported prefill/decode timings.
- Changed the Metal fused-QKV transposed cooperative GEMV kernel from a
  rectangular `(max_projection_groups, 3)` grid to a compact concatenated
  Q/K/V grid. The kernel now maps one 1D group range onto Q, then K, then V,
  so it dispatches exactly the projection groups needed.

Validation:

- `cargo test -p kiln-model --features metal test_fused_qkv_transposed_coop_gemv_matches_broadcast_matmul --lib`
  - Passed: 1 test.
- `cargo check --locked -p kiln-server --features metal --bin kiln-bench`
  - Passed.
- `cargo build --release --features metal --bin kiln-bench`
  - Passed.
- `rustfmt --edition 2024 --config skip_children=true --check crates/kiln-model/src/backend/metal.rs crates/kiln-server/src/bench.rs`
  - Passed.
- `git diff --check`
  - Passed.

Measurements:

All warmed measurements used:

`./target/release/kiln-bench --model-path /Users/ericflo/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --paged --latency-only --latency-warmup-runs 1 --prompt-tokens 64 --max-output-tokens 16 --temperature 0.0`

| Experiment | Variant | Measured prefill | Decode tok/s | Mean ITL | P50 ITL | P99 ITL | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| E203 | cold default | 11049.4 ms | 5.19 | 192.6 ms | 173.1 ms | 492.0 ms | polluted by first-use compile |
| E204 | repeat default, no explicit warmup | 1975.4 ms | 3.32 | 300.8 ms | 171.8 ms | 2199.3 ms | decode outlier; still not stable |
| E205 | cold tile4 GEMV | 9600.7 ms | 5.75 | 173.9 ms | 170.8 ms | 225.3 ms | tile4 also disables fused QKV |
| E206 | cold no fused QKV | 10787.9 ms | 5.90 | 169.4 ms | 167.9 ms | 190.0 ms | separates fused QKV from tile4 |
| E207 | warmed default, pre-change | 491.8 ms | 5.79 | 172.8 ms | 169.2 ms | 196.6 ms | first warmed baseline |
| E208 | warmed no fused QKV, pre-change | 455.9 ms | 5.94 | 168.3 ms | 169.5 ms | 171.4 ms | slightly better than E207 |
| E209 | warmed tile4 GEMV, pre-change | 459.5 ms | 5.52 | 181.0 ms | 180.5 ms | 198.5 ms | reject for decode |
| E210 | warmed default repeat, pre-change | 452.3 ms | 5.91 | 169.1 ms | 169.2 ms | 174.3 ms | baseline repeat narrowed gap |
| E211 | warmed no fused QKV repeat, pre-change | 448.4 ms | 5.89 | 169.9 ms | 169.2 ms | 193.2 ms | no-fused effect small/noisy |
| E212 | warmed compact fused QKV | 450.9 ms | 5.85 | 170.9 ms | 169.3 ms | 190.0 ms | compact grid, no regression |
| E213 | warmed no fused QKV post-change | 455.1 ms | 5.52 | 181.2 ms | 181.5 ms | 188.7 ms | noisy control |
| E214 | warmed compact fused QKV repeat | 450.0 ms | 5.96 | 167.9 ms | 168.0 ms | 173.1 ms | best warmed default sample |

Two-sample averages for the most comparable warmed paths:

- Pre-change default (E207, E210): 472.1 ms prefill, 171.0 ms mean ITL,
  5.85 decode tok/s.
- Pre-change no-fused-QKV (E208, E211): 452.2 ms prefill, 169.1 ms mean ITL,
  5.91 decode tok/s.
- Post-change compact fused-QKV (E212, E214): 450.5 ms prefill, 169.4 ms mean
  ITL, 5.90 decode tok/s.

Artifacts:

- `e203_m1_bs1_p64_o16_lowlevel_baseline.log`
- `e204_m1_bs1_p64_o16_lowlevel_repeat.log`
- `e205_m1_bs1_p64_o16_tile4_gemv.log`
- `e206_m1_bs1_p64_o16_no_fused_qkv.log`
- `e207_m1_bs1_p64_o16_warmed_baseline.log`
- `e208_m1_bs1_p64_o16_warmed_no_fused_qkv.log`
- `e209_m1_bs1_p64_o16_warmed_tile4_gemv.log`
- `e210_m1_bs1_p64_o16_warmed_baseline_repeat.log`
- `e211_m1_bs1_p64_o16_warmed_no_fused_qkv_repeat.log`
- `e212_m1_bs1_p64_o16_warmed_compact_fused_qkv.log`
- `e213_m1_bs1_p64_o16_warmed_no_fused_qkv_postcompact.log`
- `e214_m1_bs1_p64_o16_warmed_compact_fused_qkv_repeat.log`

Takeaway:

The benchmark harness now has an explicit steady-state mode for low-level
kernel work. The compact fused-QKV grid removes known empty GPU work and keeps
the default fused path competitive with disabling the fused path outright. The
measured p64/o16 improvement is small and close to run noise, so this should be
treated as a low-level cleanup plus measurement foundation, not the final
kernel win. Next low-level work should use the warmed harness on longer decode
runs and bs>1/server shapes, then target the remaining decode-heavy kernels
rather than returning to endpoint-cache-only work.

## Experiment E215-E220: Longer Decode Kernel Ablations and Rejected MLP Coop Attempt

Hypothesis:

The p64/o16 decode window is too short to make confident kernel decisions. A
p64/o64 warmed run gives 64 measured decode steps, making it a better first
screen for decode-heavy Metal paths after the compact fused-QKV change.

Measurements:

All runs used the rebuilt release bench with
`--paged --latency-only --latency-warmup-runs 1 --prompt-tokens 64 --max-output-tokens 64 --temperature 0.0`.

| Experiment | Variant | Measured prefill | Decode tok/s | Mean ITL | P50 ITL | P99 ITL | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| E215 | compact fused QKV default | 488.8 ms | 5.92 | 169.0 ms | 167.4 ms | 201.8 ms | keep compact fused QKV |
| E216 | disable fused QKV | 476.3 ms | 5.85 | 171.1 ms | 168.7 ms | 211.2 ms | slower decode |
| E217 | disable contiguous paged attn decode | 451.1 ms | 5.89 | 169.8 ms | 169.0 ms | 184.6 ms | neutral; no change |
| E218 | disable MLP gate/up fusion | 454.4 ms | 5.31 | 188.2 ms | 187.1 ms | 203.2 ms | fusion is important |
| E219 | disable attention gate fusion | 446.0 ms | 5.89 | 169.9 ms | 168.4 ms | 192.8 ms | neutral; no change |
| E220 | temporary cooperative MLP gate/up kernel | 486.9 ms | 5.70 | 175.5 ms | 174.3 ms | 196.8 ms | rejected and reverted |

Temporary E220 change:

The current MLP gate/up fusion uses one thread per intermediate output and
loops over the hidden dimension serially. E220 tried a cooperative tile8 SIMD
variant that computes eight intermediate columns per SIMD group and fuses
SiLU(gate) * up after `simd_sum` reduction. The parity test passed:

- `cargo test -p kiln-model --features metal test_mlp_gate_up_matches_reference --lib`

But warmed p64/o64 decode slowed from E215 `169.0 ms` mean ITL to E220
`175.5 ms`, so the cooperative MLP attempt was reverted before committing.
The likely issue is that the current scalar fused path benefits from much
higher output-column parallelism and avoids the heavier cooperative
threadgroup/reduction overhead for this shape, even though it does more serial
work per output.

Artifacts:

- `e215_m1_bs1_p64_o64_warmed_compact_fused_qkv.log`
- `e216_m1_bs1_p64_o64_warmed_no_fused_qkv.log`
- `e217_m1_bs1_p64_o64_warmed_no_contiguous_paged_attn.log`
- `e218_m1_bs1_p64_o64_warmed_no_mlp_gate_up_fusion.log`
- `e219_m1_bs1_p64_o64_warmed_no_attn_gate_fusion.log`
- `e220_m1_bs1_p64_o64_warmed_coop_mlp_gate_up.log`

Takeaway:

For p64/o64, compact fused QKV remains slightly ahead of disabling fused QKV.
MLP gate/up fusion is a real decode win and should stay enabled, but the
straight cooperative rewrite is worse and should not be retried without a more
careful kernel design. Contiguous paged attention and attention-gate fusion are
not obvious next wins at this shape. The next low-level slice should either
profile per-layer/per-kernel timings directly or broaden to server bs>1 before
changing another default.
