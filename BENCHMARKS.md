# Benchmarks

Inference performance for **kiln** on Qwen3.5-4B against external references on
the same single-GPU class. The acceptance protocol below is current. Numerical
tables later in this file predate that protocol and are historical until they
are replaced by checked-in `kiln.serving-benchmark.v1` receipts.

## Current measured service envelope

As of 2026-07-20, Kiln makes **no high-concurrency performance-parity claim**
against vLLM on any backend. The promotion threshold is at least `0.90x` vLLM
SLO-goodput on the same current driver, model, prompt set, request policy,
hardware, and source revision, with no correctness or tail-
latency regression. No retained comparison satisfies that threshold today.

Kiln is intentionally positioned for correctness-first local Qwen3.5-4B
workflows that need serving, adapter changes, SFT/GRPO, and evals in one artifact.
The measured operating region is external concurrency 1 through 8 with relaxed
latency requirements; that is a product scope, not a claim that Kiln beats a
serving-only engine there. Use vLLM for serving-only capacity, strict latency
SLOs, or external concurrency 8 and above. Vulkan is currently a correctness
and portability route, not a latency-sensitive serving recommendation.

### Current source-bound receipts

| Backend and gate | Exact measured scope | Result | Product meaning |
|---|---|---|---|
| ROCm, Strix Halo development soak | Qwen3.5-4B mixed load; 1,808.128 measured seconds; 688 requests; 22,016 output tokens; maximum eight active requests and decode width four | **Passed safety at 12.176 aggregate output tok/s**; p50/p99 ITL 194.493/907.601 ms; p50/p99 TTFT 6.773/25.496 s; 521 attributed runtime-event outliers and zero request, graph, device, non-finite, unexplained-outlier, or teardown failures | Accepts the 30-minute ROCm development stability gate only. Throughput was 14.66% below the prior accepted soak; a paired old/current short run limited source regression to 2.45%, so variance remains explicit. The final 24-hour endurance gate and vLLM competitiveness remain open. |
| Vulkan, Strix Halo development soak | Qwen3.5-4B mixed load; 1,935.219 measured seconds; 55 requests; 880 output tokens; maximum resident-prefill width two | **Passed stability, poor performance** at 0.455 aggregate output tok/s; p50/p99 ITL 76.721/1,768.377 ms; p50/p99 TTFT 83.194/150.588 s; zero request, allocation-growth, device, unexplained-outlier, or teardown failures | Accepts Vulkan development stability only. Do not use this result to imply acceptable interactive latency or throughput. |

The exact compact receipts are:

- [ROCm development soak](qualification/receipts/rocm/strix-halo/20260720t144932814398z-rocm-strix-halo-serving-rocm-development-053e89eca9-v1.json), source commit `3887b640ce14b79e8396b940c66140f252ebd5e5`, receipt content `sha256:6592587914b46f87f4babc396466702b100324ca682439bdf680050d57d6cd19`.
- Same-machine `autoscale-off` source discriminator: [prior source](qualification/receipts/rocm/strix-halo/20260720t154421710282z-rocm-strix-halo-serving-mixed-rocm-v1-184c082f9e-v1.json) at 12.662 deterministic/8.144 sampled tok/s and [current source](qualification/receipts/rocm/strix-halo/20260720t153545055239z-rocm-strix-halo-serving-mixed-rocm-v1-184c082f9e-v1.json) at 12.432/7.945. The 1.82%/2.45% deltas are below the 3% material-regression threshold.
- [Vulkan development soak](qualification/receipts/vulkan/strix-halo/20260720t105341024462z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json), source commit `fb5cb029d9fefc13796b9ecdf928062d62445f78`, receipt content `sha256:bbf4060c4c6173b30db0469bcecabb8d01c905d8ebb6997be9a36c41d72b70e7`.

These soaks do not share a workload with vLLM and must not be divided by a vLLM
number. They establish sustained behavior for their own source-bound routes.

### What the paired ROCm diagnostic proves

The latest retained same-machine Kiln/vLLM pair used one Qwen3.5-4B
`greedy-short` prompt set, 64 fixed output tokens, and external concurrency
1/8/16/32. All 57 requests per engine completed. Kiln measured
5.914/8.395/7.489/7.118 output tok/s; vLLM measured
3.723/18.036/34.363/51.585. At concurrency 8 and 32, Kiln therefore delivered
0.465x and 0.138x vLLM request-window throughput. This is enough to reject a
high-concurrency competitive claim and recommend vLLM for that use case.

It is **not** an accepted performance comparison. Exact greedy outputs diverged
at every concurrency. Driver v7 localized the c1 divergence to generated token
index three, after three identical tokens, but did not establish an independent
winner. The throughput counterexample guides the conservative backend
recommendation, while the failed output gate prohibits quoting the ratios as
parity evidence.

- [Kiln diagnostic receipt](benchmarks/receipts/rocm/strix-halo/20260718t223203-rocm-strix-halo-greedy-short-c1-32-sourcepair-v1.kiln.json)
- [vLLM failed-comparison receipt](benchmarks/receipts/rocm/strix-halo/20260718t223203-rocm-strix-halo-greedy-short-c1-32-sourcepair-v1.vllm.json)
- [Kiln c1 divergence receipt](benchmarks/receipts/rocm/strix-halo/20260718t232632-rocm-strix-halo-greedy-c1-divergence-v1.kiln.json)
- [vLLM c1 divergence receipt](benchmarks/receipts/rocm/strix-halo/20260718t232632-rocm-strix-halo-greedy-c1-divergence-v1.vllm.json)

The next competitive comparison must use the current driver and current source,
run without stop/resume inside active request windows, satisfy nonzero declared
SLO-goodput, and keep correctness qualification separate from throughput mode.
Until that receipt exists, the public position remains: choose Kiln for its
integrated local improvement loop; choose vLLM for high-concurrency serving.

## Current serving acceptance protocol

[`scripts/bench-concurrent-batch.py`](scripts/bench-concurrent-batch.py) is the
one serving driver for Kiln and vLLM. Measured traffic uses only the shared
streaming `POST /v1/chat/completions` API. Kiln `/health` snapshots are taken
before and after a run when available, outside the timed request path; they do
not alter request bodies or scheduling.

New measurements use driver version 16. The driver:

- fingerprints every weight shard plus the model config, tokenizer, and chat
  template before traffic and rechecks the complete identity after traffic;
  both guarded integrity passes keep the correctness-preserving second content
  read but use one explicit cumulative read-rate limit, 256 MiB/s by default;
- hashes the exact Kiln binary or immutable vLLM runtime manifest before and
  after traffic; for Kiln, the hash must also equal the running process's
  `/health.execution_identity.executable_sha256`;
- uses one of five fixed workload profiles instead of accepting an arbitrary
  prompt/sampling combination;
- requires a unique operational run ID and a separate stable prompt-set ID,
  derives every model-visible prompt only from the latter, and proves the
  recorded prompt hashes still match it;
- records every request's observed prompt-token count and requires the same
  ordered counts from the reference engine, not only the same prompt text;
- pins template thinking mode, output length, neutral penalties, seeds,
  all-at-once arrival, warmup, concurrency, and profile-defined sampling;
- starts every row behind one thread barrier and records dispatch spread;
- requires one `[DONE]`, one positive usage record, one finish reason, and the
  requested number of generated tokens from every request;
- reports client-visible TTFT, SSE-event ITL p50/p99/p99.9, end-to-end latency,
  request and output-token throughput, SLO-goodput, errors, and sampled peak
  device memory;
- records Kiln's effective decode ceiling, observed width, mean rows per
  forward, phase time, and error deltas when its diagnostics are enabled;
- samples one explicit DRM memory counter, fails any row above the declared
  absolute byte limit, and records both absolute peak and baseline delta;
- owns a typed argv-only server launch by default and binds the new process
  group before readiness;
- uses ordinary monotonic wall-clock time without a host-specific temperature
  controller or CPU quota;
- requires a clean process-group shutdown and rejects listener or process
  residue;
- writes an atomic, self-hashing receipt on success or on any recoverable
  post-preflight failure, preserving the exact ordered prefix of completed rows;
- records structured warmup, measurement, sampler-stop, comparison, and
  finalization failures, while independently rechecking repository, model,
  runtime artifact, and engine-specific live-runtime identity after traffic;
- hashes the exclusive server log and retains typed launch, readiness,
  shutdown-status, forced-shutdown, and final process-liveness evidence;
- requires a comparison-compatible v7 through v16 reference receipt with
  identical workload, model, prompts, and outputs for exact-output
  cross-engine runs.

A partial v16 receipt is diagnostic counterevidence, never performance
acceptance. Its top-level verdict is `failed`, `completion.completed_run_count`
names the retained prefix, `completion.expected_run_count` names the complete
declared matrix, and `completion.failures` explains why execution stopped. Kiln
receipts must complete the live execution-identity check and mark the vLLM
manifest check not applicable; vLLM receipts do the inverse. The common source,
model, and runtime-artifact checks are never optional. A process crash or host
loss can still prevent any file from being written, so raw logs remain required
for catastrophic failures.

Driver versions 2 through 15 remain accepted only so historical checked-in
receipts continue to validate. They cannot produce new receipts and do not
satisfy the current acceptance protocol. The complete field and lifecycle
reference is in [Serving Benchmark Protocol](docs/SERVING_BENCHMARK_PROTOCOL.md).

### Fixed workload profiles

| Profile | Prompt shape | Sampling | Cross-engine comparison |
|---|---|---|---|
| `greedy-short` | unique short prompts with the same marker multiset | temperature 0, top-p 1 | prompt text, ordered prompt-token counts, exact output |
| `api-default-sampled` | the same short prompts | temperature 1, top-p 1, fixed per-request seeds | prompt text and ordered prompt-token counts; output hashes are retained but not required to match across different samplers |
| `long-prefill` | a fixed long body plus a unique suffix | temperature 0, top-p 1 | prompt text, ordered prompt-token counts, exact output |
| `prefix-hit` | a byte-identical long shared prefix plus a unique suffix | temperature 0, top-p 1 | prompt text, ordered prompt-token counts, exact output |
| `mixed` | deterministic short, medium, long, and longest rows | temperature 0, top-p 1 | prompt text, ordered non-uniform prompt-token counts, exact output |

The profile owns temperature, top-p, and whether prompt lengths must be
uniform. Conflicting CLI overrides fail before traffic. Exact output is not a
meaningful cross-implementation requirement for sampled decoding because the
seed does not standardize sampler implementation details; the sampled profile
therefore retains each output hash without pretending it is parity evidence.

Current serving accepts only effective speculative method `off`; this driver
therefore measures the ordinary serving path. A configured `skip_layer` or
`mtp` method fails `kiln config` and startup before model loading. The retained
benchmark-only speculative implementation may be exercised only by an isolated
qualification harness with explicit accelerator evidence at the bounded
K=1/K=2/K=4 matrix. It is not a serving or Desktop bypass.

`client_visible_itl_ms_*` is deliberately named: it measures non-empty
semantic SSE-event arrival, which is the only engine-neutral client signal.
An engine may coalesce multiple tokenizer tokens into one visible event. The
driver does not request Kiln-only token-timing events because that would make
the two measured request bodies different.

Run Kiln first from a clean checkout. Use an explicit device-memory telemetry
path on machines with more than one GPU. If the workload declares an absolute
memory limit, choose it before either engine starts and use the same value for
both engines; do not derive it from an observed peak.

Prefer `--server-launch-config`: the driver creates a fresh session-led process
group and arms containment before readiness, then owns graceful shutdown. The
loopback port must be unbound before launch, and after readiness
its listening socket inode must belong to the launched process group. The
closed launch document contains argv, working/log directories,
readiness cadence and deadline, shutdown deadline, and accepted exit codes; it
has no shell string or environment map. `--server-pid` remains available for an
external supervisor. That PID must lead its group, and containment before the
driver attaches is then the supervisor's responsibility. Both modes bind PID,
PGID, Linux boot ID, process start ticks, executable path, and command-line
hash. Machine-specific temperature policies are not part of the current launch
contract.

`output_token_throughput_per_s` is the engine-neutral request-window rate using
ordinary wall-clock time.

Use a checked-in or receipt-bound TOML policy for serving. For CUDA, ROCm, and
Vulkan throughput qualification, start with true batched decode and preserve
the backend-qualified admission cadence:

```toml
[batching]
mode = "enabled"
rowwise_decode = false
prefix_aware_admission = true
prefill_admission_quantum = "auto"
direct_decode_rendezvous_mode = "auto"
direct_decode_rendezvous_max_batch = "auto"
direct_decode_rendezvous_wait_us = "auto"
direct_decode_rendezvous_mixed_seq_lens = "auto"
```

After every restart, capture the exact target before sending measured traffic:

```bash
curl -fsS http://127.0.0.1:8420/v1/config \
  | jq '{batching, decode_runtime}'
```

For an ordinary throughput receipt, require
`batching.actor_active=true`,
`batching.configuration.mode.effective_enabled=true`, and
`batching.configuration.rowwise_decode.enabled=false`. Record the full
`batching.configuration` object, including the admission quantum's configured,
backend-policy, effective, and source values. Do not compare only the TOML
request: deterministic mode, the combined token budget, or a narrower decode
ceiling can lower the width and admission quantum. A deliberate actor-off,
rowwise, prefix-aware, or explicit-quantum A/B must use a distinct run ID and
state that intervention in the receipt notes.

The direct rendezvous settings do not tune an actor-backed throughput run. The
worker can be active in that process, but
`batching.direct_decode_rendezvous.route_available` must be false while
`actor_active=true`. Only an actor-disabled, streaming, effectively-greedy
request can use that route; sampled, non-streaming, and other direct work bypass
it. A fallback-specific A/B must therefore disable the actor, restart, require
`route_available=true`, and record the nested configured/backend/effective
mode, maximum batch, wait, and mixed-sequence values. Do not infer routing from
`worker_active` alone. Backend `auto` tuples are CPU `(8,0,false)`, CUDA
`(1,0,false)`, ROCm `(8,0,false)`, Metal `(8,100,true)`, and Vulkan
`(64,5000,true)`, with maximum batch clamped to effective decode width.

The campaign command defines all five profiles at concurrency 1, 8, 16, 32, 64,
and 128. It stages profile receipts outside the output tree and publishes them
only after execution, so checked-in output paths cannot dirty the source seen by
later profiles. The default policy stops after the first failed profile or
missing receipt, preserves that counterevidence, marks the remaining profiles
`not_run_after_failure`, and publishes the self-hashing v6 summary last:

```bash
python3 scripts/run-serving-benchmark-campaign.py \
  --engine kiln \
  --base-url http://127.0.0.1:8420 \
  --model Qwen3.5-4B \
  --model-path ./Qwen3.5-4B \
  --runtime-identity "kiln-git:$(git rev-parse HEAD)" \
  --runtime-artifact ./target/release/kiln \
  --campaign-id qwen35-4b-rocm-kiln-20260713 \
  --prompt-set-id qwen35-4b-rocm-comparison-v1 \
  --sizes 1,8,16,32,64,128 \
  --repeats 3 \
  --max-tokens 64 \
  --memory-path /sys/class/drm/card1/device/mem_info_vram_used \
  --memory-limit-bytes 30000000000 \
  --server-launch-config .qualification/serving/rocm-kiln-launch.json \
  --out-dir /tmp/kiln-serving-campaign
```

`--continue-after-failure` is an explicit diagnostic override for a known
isolated failure. Never use it after forced shutdown, process residue, OOM, or
device error.

Both direct and campaign runs accept
`--model-fingerprint-read-mib-per-second` in `64..=16384`; the measured default
is 256. Campaign v6 records the selected value and forwards it to every child.
This setting limits only the guarded provenance worker. It does not change
server model loading, request traffic, or timed throughput.

Run vLLM with the same prompt-set ID, model alias, checkpoint, sizes, limits,
and SLOs, but give the campaign attempt a new campaign ID. `--runtime-artifact`
must name the immutable launcher/runtime manifest
that fingerprints the installed vLLM, Torch, Transformers, tokenizer, launch
configuration, and accelerator; a version string alone is not sufficient.
Generate it with the exact model, limits, and vLLM arguments used by the real
launch (see [Immutable vLLM teachers](docs/VLLM_TEACHER_IDENTITY.md)):

```bash
python3 scripts/vllm_teacher.py \
  --manifest-only \
  --model-path ./Qwen3.5-4B \
  --served-model-id Qwen3.5-4B \
  --max-top-k 20 \
  --max-model-len 32768 \
  --max-provenance-read-mib-per-second 256 \
  -- --dtype=bfloat16 \
  > /tmp/vllm-runtime-manifest.json
```

The benchmark rejects an arbitrary file: the manifest's canonical identity,
runtime-content hash, served model, vLLM implementation, and embedded
`kiln-teacher-v1` fingerprint must verify. Start vLLM through the launcher with
the same identity-bearing arguments. The campaign runner then pairs each
profile with the corresponding Kiln receipt:

```bash
python3 scripts/run-serving-benchmark-campaign.py \
  --engine vllm \
  --base-url http://127.0.0.1:8000 \
  --model Qwen3.5-4B \
  --model-path ./Qwen3.5-4B \
  --runtime-identity vllm:VERSION+RUNTIME-CONTENT-SHA256 \
  --runtime-artifact /tmp/vllm-runtime-manifest.json \
  --campaign-id qwen35-4b-rocm-vllm-20260713 \
  --prompt-set-id qwen35-4b-rocm-comparison-v1 \
  --sizes 1,8,16,32,64,128 \
  --repeats 3 \
  --max-tokens 64 \
  --memory-path /sys/class/drm/card1/device/mem_info_vram_used \
  --memory-limit-bytes 30000000000 \
  --server-launch-config .qualification/serving/rocm-vllm-launch.json \
  --reference-dir /tmp/kiln-serving-campaign \
  --out-dir /tmp/vllm-serving-campaign
```

Use `scripts/bench-concurrent-batch.py` directly for a focused profile run.
Select a profile with `--workload-profile` (default `greedy-short`).
`--model-path`, `--runtime-artifact`, `--memory-path`, and
`--memory-limit-bytes` are mandatory; none of the provenance or memory
requirements are optional.

An official receipt must not use `--allow-dirty`, disable fixed output, omit
runtime/model/memory identity, or include failed/zero-token requests in
throughput. Use a unique `--run-id` for every attempt and the same explicit
`--prompt-set-id` for comparable engines or implementation arms. The run ID
owns receipt/log lifecycle identity and never changes a request body. The
prompt-set ID owns prompt ordering and the model-visible benchmark label; do
not put a timestamp, engine, candidate name, or retry number into it. Restart
the engine between implementations so device-memory baselines are independent.
Authenticated endpoints must opt in with `--api-key-env VARIABLE` (preferred)
or `--api-key VALUE`. The driver never inherits `OPENAI_API_KEY` implicitly and
never writes the credential or environment-variable name into a receipt.

Treat exact output comparison as a reproducibility gate, not an unstated
property of the throughput configuration. Start Kiln with
`server.deterministic = true` (or canonical
`KILN_SERVER_DETERMINISTIC=true`) and verify that
`/health` reports `decode_runtime.configuration.deterministic.enabled = true`,
`decode_runtime.configuration.max_decode_batch.effective_source` as
`"deterministic"`,
`decode_runtime.batching_configuration.prefill_admission_quantum.effective = 1`,
its `effective_source` as `"effective_decode_width"`,
and `decode_runtime.batching_engine.max_decode_batch = 1`;
configure the reference
engine for the same single-row deterministic execution. Default multi-row BF16
serving may select different valid GEMM shapes as cohorts form, and a close
greedy-logit boundary can then change the continuation. Preserve any such
failed exact-comparison receipt as evidence; never relabel it as output parity.
Run performance and deterministic correctness as separate receipts so the
reproducibility guarantee does not silently disable the batching being
measured.

When investigating a pause, first separate scheduler time from device or
memory time. Request `include_performance=true` and compare `actor_queue_ms`,
`actor_admission_ms`, and `actor_prefill_wall_ms` with the batching actor's live
phase counters and the memory-governor/allocator diagnostics. A long actor
queue or admission/prefill phase is not evidence of a VRAM rebalance; a reclaim,
resize, or synchronization claim requires the corresponding structured event
or metric. Preserve a counterexample receipt before changing one batching field
at a time and restarting.

Commit detailed serving receipts under
`benchmarks/receipts/<backend>/<host-id>/`. They intentionally do not belong
under `qualification/receipts/`, whose files use the separate compact
`receipt-v1` schema. CI validates both families: standard qualification
receipts with `scripts/qualification/receipt.py`, and detailed serving receipts
with `scripts/bench-concurrent-batch.py --validate-receipt PATH...`. The latter
rejects unknown or missing fields, non-finite metrics, inconsistent gates,
dirty passed sources, workload/run mismatches, and an invalid canonical
self-hash. It also rejects a run if the repository identity changes while
measurement is in progress. Driver v3 added model/runtime identity, closed
profiles, prompt shape, memory, and structured partial-run checks. V4 added
owned lifecycle and post-exit evidence; v5 added post-provenance pre-launch
cooldown; v7 added ordered per-request output evidence; v8 added guarded
initial/final model fingerprints; v9 added route-aware request accounting; and
v10 added closed ROCm graph execution evidence. V11 added typed pre/post
idle-boundary cooldown evidence for every hard-limit-only row and requires both
waits to match the content-hashed policy and fit inside phase wall time.
V12 bounded model-fingerprint I/O; v13 accounted cooperative actor idle; v14
closed multi-row ROCm graph fallback accounting; v15 retained request-local
terminal performance; and v16 separated unique lifecycle identity from stable,
strictly verified prompt identity. Historical v2 through v15 validation is
compatibility, not current performance evidence.

## Historical CUDA setup

| Component | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GB, compute capability 8.6) |
| Driver | 550.127.08 |
| CUDA | 12.4 |
| Host | RunPod on-demand pod (`ghcr.io/ericflo/kiln-runpod:latest` image) |
| Model | Qwen3.5-4B (~4.2B params, 32 layers, hybrid GDN + GQA) |
| Weights | bf16 base, W4A16 Marlin-packed MLP projections at runtime |
| Prompt length | 512 tokens (humaneval subset, ~494 after tokenization) |
| Output length | 128 tokens |
| kiln commit | `c5cf77d` (post-PR #535, captured by PR #536) |
| kiln features | `--features cuda`, `KILN_CUDA_ARCHS=86` |
| kiln env | `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true` |
| kiln binary | `kiln-bench` |

This table describes the historical PR #536 benchmark revision, not a current
serving configuration. Current production loading uses `load_mtp=false`, and
the current K default and hard ceiling are 4. Re-running speculative research
requires an isolated qualification harness rather than `kiln serve`.

The current single-stream protocol is `--paged --prompt-tokens 512
--max-output-tokens 128 --skip-training --prompt-subset humaneval
--chat-template --latency-only --temperature 0.0 --seed N` with three fresh
processes. Median-of-3 governs.

## Historical results

### Single-stream throughput (kiln, post-PR #536, A6000)

512 prompt → 128 generated, MTP-Off, three seeds, median-of-3:

| Metric | Value |
|---|---:|
| Decode (tok/s, median) | **44.75** |
| Decode (tok/s, range) | 44.23 – 45.01 (Δ = 1.8 %) |
| Prefill latency (ms, median) | 355.4 |
| Prefill (tok/s) | ~1390 |
| Mean ITL (ms) | 22.35 |
| P50 ITL (ms) | 22.33 |
| P99 ITL (ms) | 27.33 |
| Peak VRAM | ~10 GB |

Source: `docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.csv` (PR #536), with the
hotspot mix and post-#534 comparison in `docs/archive/phase-c/phase-c65/post-534-profile.md`
(PR #535).

### Multi-engine comparison (Qwen3.5-4B + native MTP, A6000 sm_86)

Phase 7 audited the obvious external references against kiln on the same GPU
class. As of this refresh, no other Qwen3.5-4B + native-MTP serving stack runs
end-to-end on the kiln-runpod stock image, so a side-by-side decode tok/s
table is not currently producible. The audit results are summarized below.

| Engine | Status on A6000 sm_86 / driver 550.x | Decode (tok/s) | TTFT | Peak VRAM | Source |
|---|---|---:|---:|---:|---|
| **kiln (post-#536)** | runs | **44.75** (median) | 355 ms | ~10 GB | this doc |
| **llama.cpp `9d34231` (current main, 512 → 128)¹** | runs | **68.99** (median) | ~85 ms | ~8.8 GB | `bench-results/llama-bench-a6000-post536.json` |
| **llama.cpp `9d34231` (current main, 512 → 256)¹** | runs | **69.23** (median) | ~85 ms | ~8.8 GB | `bench-results/llama-bench-a6000-post536.json` |
| **vLLM v0.19.1** | unsupported² | — | — | — | PR #530 |
| **vLLM v0.20.0** | unsupported³ | — | — | — | PR #533 |
| **SGLang 0.5.10.post1** | unsupported⁴ | — | — | — | PR #532 |
| **HF transformers reference** | runs (α-only)⁵ | — | — | — | PR #534 |

¹ Current-main llama.cpp A6000 re-bench captured on the same kiln-runpod
A6000 / driver 550.127.08 base image as the kiln row above. llama-bench
build commit `9d34231`, Qwen3.5-4B converted to bf16 GGUF via
`convert_hf_to_gguf.py --outtype bf16`. Flags: `n_batch=2048`,
`n_ubatch=512`, `flash_attn=false` (`-fa 0`), `n_gpu_layers=99` (`-ngl 99`),
K/V cache in f16. Three repeats per shape, median-of-3 reported. Decode
tok/s is essentially flat across decode lengths on this stack (Δ = 0.35 %
between tg128 and tg256), so the headline 512 → 128 number is what the
multi-engine apples-to-apples comparison anchors on. Prefill (pp512):
5989 tok/s (median). The pre-existing RTX 6000 Ada (sm_89) row on kiln
commit `f3d5089` is now historical (see [section
below](#historical-llamacpp-head-to-head-on-rtx-6000-ada-kiln-f3d5089)).

² vLLM 0.19.1 loads the model + drafter weights but segfaults inside
spec-decode native code during runtime profiling on Qwen3.5-4B + native MTP.
70-frame native backtrace, no Python file symbol at the terminal frame.
Verdict: `vllm_mtp_unsupported`. See PR #530.

³ vLLM v0.20.0+cu129 transitively requires `torch 2.11.0+cu130`, which itself
requires NVIDIA driver ≥580 (CUDA 13.0). The kiln-runpod stock image ships
driver 550.127.08, so `torch.cuda.is_available()` returns `False` and vLLM's
V1 `EngineCore` worker crashes during CUDA init before any MTP code runs.
Verdict: `vllm_020_mtp_unsupported_dense_4b`. See PR #533.

⁴ SGLang 0.5.10.post1 + Qwen3.5-4B dense BF16 + native MTP segfaults across
three distinct serving configurations (flashinfer + CUDA graphs, flashinfer
graphs-off, triton attention graphs-off). Engine-side dispatch is correct
(`Qwen3_5ForCausalLMMTP` weights load, hybrid GDN dispatcher initializes); the
crash is downstream of model dispatch. Verdict:
`sglang_mtp_unsupported_dense_4b`. See PR #532.

⁵ The hand-rolled HuggingFace `transformers` reference loads and runs
end-to-end but is α-only (acceptance-rate microbench), not a decode tok/s
serving comparison. It is included here because it is the only external stack
that produces a usable Qwen3.5-4B + MTP signal on this GPU; with seed 0
producing a bit-for-bit accept/reject trace match against kiln's c1_attr CSVs
and median α 0.2500 vs kiln 0.3636, the verdict was `kiln_above_hf`. See
PR #534.

The historical takeaway was structural rather than numeric: this kiln revision
was the only audited stack that produced end-to-end Qwen3.5-4B + native-MTP
decode numbers on the stock A6000 / driver 550.x image. That historical result
does not describe the current serving surface or satisfy its qualification
gate.

### Historical native MTP self-spec - α below break-even at bs=1

PR #536 ran a three-seed MTP-On vs MTP-Off A/B against post-#535 main:

| Arm | Decode (tok/s, median) | α (median) | P99 ITL (ms) |
|---|---:|---:|---:|
| MTP-Off | **44.75** | — | 27.33 |
| MTP-On (forced) | 43.09 | 0.6842 | 56.78 (~2.08×) |

Median Δ = −3.7 %, verdict: `mtp_no_decode_win`. α has improved 5.5× since
the last bs=1 measurement (PR #316: α=0.124, MTP-On −25.1 % slower) but is
still below the bs=1 break-even floor of α≈0.72; one seed (α=0.778) cleared
the floor at +8.5 %, the other two (α=0.620, α=0.684) lost 9.2 % and 4.3 %
respectively. P99 ITL roughly doubles when MTP is on because rejected-draft
steps add a heavy tail. This experiment is not an opt-in serving route:
current `kiln config` and startup reject MTP before model loading. See PR #536
and `docs/archive/phase-c/phase-c66/` for the archived evidence and reopen
triggers; any new run must use the isolated K=1/2/4 qualification harness.

For that historical revision, the quantized + GDN-fused base path was already
close to its bs=1 ceiling on A6000 and native MTP did not pay back verifier
cost at the measured α. Current performance claims must come from the ordinary
serving protocol above; speculative promotion requires separate qualification
evidence rather than extrapolation from this experiment.

### Historical batched concurrent-decode throughput (L40S sm_89, May 2026)

Aggregate `/v1/chat/completions` greedy decode tokens/s across N
concurrent HTTP streams, Qwen3.5-4B, `KILN_W4A16=1`,
`KILN_CUDA_GRAPHS=true`, `KILN_MAX_DECODE_BATCH=64`, 128 generated
tokens per stream, using the pre-v2 driver and wall-clock nonce prompts. Those
nonces avoided Kiln's deterministic completion cache, but did not bind the two
engines to one reproducible prompt set or measure streaming TTFT/ITL. These
rows therefore remain historical rather than current acceptance evidence:

| Concurrency | Aggregate tok/s | Scale vs bs=1 |
|---:|---:|---:|
| 1 | 101 | 1.00× |
| 2 | 170 | 1.69× |
| 4 | 322 | 3.19× |
| 8 | 575 | 5.69× |
| 16 | 945 | 9.36× |
| 32 | 1355 | 13.42× |
| 64 | 1776 | 17.59× |

Subsequent cleanup commits (`f6a1e85b` GDN conv layout reshape; `89220f6c`
stable-id cache fingerprint) preserve these numbers without sacrificing
robustness; pointer-fingerprint variant of the cache regressed in
workloads with mixed `max_tokens` (mid-batch `Vec::remove` shifts) and
the stable id closes that gap.

Source: PRs landing the broadcast-matmul / row-loop-default / positions-
uniform-gate / batched-state-cache fixes (commits `2e252f8a`,
`feb316a1`, `ffe0feb7`, `c03e1da0`, `153dc932`, May 2026). Pre-fix
aggregate was a **flat ~100 tok/s ceiling regardless of N**.

Two compounding wins:

1. **`broadcast_matmul` contiguous-copy fix** (`2e252f8a` /
   `feb316a1`): candle's `Tensor::broadcast_matmul` for `[B, T, K] @
   [K, N]` was materializing the RHS via
   `broadcast_as(...).contiguous()` — a multi-hundred-MB BF16 weight
   copy across the batch dim per GDN in-proj matmul. nsys showed
   `ucopy_bf16` at 78 % of GPU time pre-fix; post-fix it falls out of
   the top kernels and actual compute (cutlass / Marlin GEMM)
   dominates.

2. **Persistent batched-state cache** (`153dc932`):
   `decode_next_tokens_paged_contiguous_batch_greedy` was re-running
   `LinearAttentionState::from_batch_rows` (24 GDN layers × 2 state
   kinds = 48 `Tensor::cat` ops) on every decode step. Caching the
   batched state on the `ModelRunner` and keying on per-row pointer
   identity skips this stage on consecutive same-composition batches.
   Adds +10–32 % depending on batch size, biggest win at bs=64.

The bs=1 number is unchanged because the bs=1 codepath is
`graph_runner.decode_step_paged`, which never touched either hot
path; the fix is purely a bs > 1 win.

### Direct head-to-head: kiln vs vLLM 0.21.0 (L40S sm_89, May 2026)

Same Qwen3.5-4B weights and L40S, measured with the pre-v2 version of
`scripts/bench-concurrent-batch.py`, greedy decode, `max_tokens=64`, and
per-call nonce prompts. vLLM serves via
`vllm serve … --gpu-memory-utilization 0.85 --max-model-len 2048
--max-num-seqs 256`, default torch.compile + full CUDA-graph capture
across batch sizes [1..512]. kiln runs `target/release/kiln serve`
with `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true`,
`KILN_MAX_DECODE_BATCH=64`.

Aggregate tokens/s (single warmup pass before each sweep):

| Concurrency | kiln main `f3a5f95e` tok/s | vLLM 0.21.0 tok/s | kiln / vLLM |
|---:|---:|---:|---:|
| 1  | 100.6 | 78.4   | **1.28× kiln** |
| 2  | 163.5 | 140.3  | **1.16× kiln** |
| 4  | 307.5 | 229.9  | **1.34× kiln** |
| 8  | 513.5 | 518.9  | 0.99× (tie)    |
| 16 | 793.2 | 808.4  | 0.98× (tie)    |
| 32 | 998.4 | 1485.5 | 0.67×          |
| 64 | 1181.4| 1906.8 | 0.62×          |

**Picture as of `f3a5f95e`** (today's two commits: GDN `out_proj`
W4A16 opt-in + hoisted per-step paged-decode metadata; both built and
benched here on L40S):

- **bs ≤ 4**: kiln wins by 16–34 %. vLLM pays a steep async-scheduler
  / Python overhead at low concurrency; kiln's batching-engine actor
  + CUDA-graph-captured bs=1 step is cheaper.
- **bs = 8–16**: parity (kiln within 2 % of vLLM either way).
- **bs ≥ 32**: vLLM pulls ahead 1.49× at bs=32 and 1.62× at bs=64,
  driven by their full multi-batch CUDA-graph capture + torch.compile
  inductor fusions. kiln currently only graph-captures bs=1, so per
  bs=64 step every kernel pays full dispatch overhead. Closing this
  is the next major perf lever — see TODO note in `cuda_graph.rs`.

The historical gap reproduced across multiple sweeps, and the old driver's
nonce neutralized the deterministic completion cache that otherwise inflated
Kiln numbers. It did not provide the current workload fingerprint, usage
gates, output parity, device-memory sampling, or streaming ITL contract, so it
cannot serve as a current comparison receipt. Its p50/p99 request latencies follow
the aggregate-throughput ordering (vLLM's bs=64 p99 = 2.13 s, kiln's
bs=64 p99 = 3.46 s).

### Historical batched CUDA-graph capture investigation

> The two environment switches named below were retired on 2026-07-18. The
> unqualified batched route is source-disabled after failing real concurrent
> serving. These measurements preserve investigation history; they are not
> current configuration instructions or qualification evidence.

The historical setup used `KILN_CUDA_GRAPHS_BATCHED=1` and
`KILN_CUDA_GRAPHS_BATCHED_NO_REPLAY=1`. At that source revision, the runner
recorded a
batched CUDA graph for the per-step shape, launches it once, then
evicts the cache so each subsequent step re-captures. The full replay
path on cache-hit is disabled pending a CUDA-side correctness fix
(see `feedback-batched-cuda-graph-debug` memory). Even so, the
capture-and-launch (no replay) path is meaningfully faster than the
eager batched fallback:

| bs | kiln eager (main) | kiln batched-capture | kiln / kiln-eager | kiln / vLLM |
|---:|---:|---:|---:|---:|
| 1  | 100.6 |  97.8  | 0.97× | **1.25× kiln** |
| 2  | 163.5 | 254.3  | **1.55×** | **1.81× kiln** |
| 4  | 307.5 | 407.5  | **1.32×** | **1.77× kiln** |
| 8  | 513.5 | 621.3  | **1.21×** | **1.20× kiln** |
| 16 | 793.2 | 960.5  | **1.21×** | **1.19× kiln** |
| 32 | 998.4 | 1282.1 | **1.28×** | 0.86×          |
| 64 | 1181.4| 1401.2 | **1.19×** | 0.73×          |

At that historical source revision, the `kiln batched-capture` column captured
and launched a fresh graph for every batched decode step, with no replay. At
bs <= 16 kiln led vLLM by 19-81 %. At bs=32 the gap to vLLM narrowed
from 33 % to 14 %; at bs=64 from 38 % to 27 %.

The remaining bs ≥ 32 gap is the missing replay path: capture has a
fixed instantiation cost (~1 ms) that gets paid once per step
without replay, but would be amortized across many steps with replay
working. When the replay correctness bug is fixed (separately
tracked) we expect bs ≥ 32 to close the rest of the gap.

### kiln steady-state — refresh pending

The previous version of this doc included a 1 / 4 / 8 / 16 sequential-runs
table (RTX 6000 Ada, kiln commit `f3d5089`, 512 → 256) showing throughput
flat across runs. Those numbers are pre-Phase-6 and no longer represent
current main; PRs #535 and #536 did not re-run the same configuration on
A6000. A refresh of the sequential-runs steady-state table is on the
to-measure list and will follow a future profiling run.

### Historical: llama.cpp head-to-head on RTX 6000 Ada (kiln `f3d5089`)

Kept for reference only. This was the original end-to-end head-to-head
captured on RTX 6000 Ada (sm_89). The multi-engine row anchors on the
A6000 sm_86 re-bench above; this section exists so the original numbers
remain auditable. Both sides are out of date — kiln-side decode is
~10 tok/s on `f3d5089` versus ~45 tok/s on post-#536, and llama.cpp on
A6000 sm_86 today is 68.99 tok/s at 512 → 128 / 69.23 at 512 → 256
versus the 88.43 quoted here at 512 → 256 on RTX 6000 Ada sm_89.

| Engine | Prefill (tok/s) | Decode (tok/s) | TTFT | Peak VRAM |
|---|---:|---:|---:|---:|
| llama.cpp `408225b` (RTX 6000 Ada, 512 → 256) | **8224.45** | **88.43** | ~62 ms | **8820 MB** |
| kiln `f3d5089` (pre-Phase-6, RTX 6000 Ada, 512 → 256) | 36.31 | 10.89 | 13934 ms | 10278 MB |

Raw JSON: `bench-results/kiln-bench.json` (kiln side, stale) and
`bench-results/llama-bench.json` (llama.cpp side, RTX 6000 Ada — superseded
in the multi-engine table by `bench-results/llama-bench-a6000-post536.json`).

## Interpretation

- **Decode is now ~5× faster on current main than on the original `f3d5089`
  baseline** (10.89 → 44.75 tok/s on the 512 → 128 humaneval shape with
  W4A16 + CUDA graphs). The Phase 6 fused kernels (RMSNorm, GDN gates, GDN
  qk-norm, paged decode) and the W4A16 Marlin MLP wire-in (PR #152) are
  carrying most of the win.
- **The remaining gap to llama.cpp at the same 512 → 128 shape on the same
  A6000 sm_86 / driver 550.x base image is 1.54×** (kiln 44.75 vs llama.cpp
  68.99, both median-of-3, current main on both sides). At 512 → 256 the
  llama.cpp side is essentially flat (69.23 tok/s), so widening the decode
  window does not change the gap — this is not a TTFT-vs-decode artifact.
  The original doc estimated ~2× by comparing the kiln post-#536 A6000
  number against the historical RTX 6000 Ada (sm_89) llama.cpp row at
  512 → 256; the apples-to-apples A6000 sm_86 re-bench in this refresh
  closes that measurement gap and lands the actual delta closer to ~1.5×.
- **vLLM and SGLang comparisons are blocked engine-side**, not because kiln
  is faster. Whenever a vLLM stable wheel ships `+cu124` / `+cu128` (or the
  RunPod kiln-runpod base image upgrades to driver ≥580), the H17b /
  H17 drivers in `scripts/h17b_vllm_020_alpha_dump.py` and
  `scripts/h17_sglang_alpha_dump.py` are re-runnable in 10–30 minutes for
  ~$0.10–$0.20 per engine. Wire those numbers into the multi-engine table at
  that point.
- **VRAM** at ~10 GB on A6000 leaves substantial headroom. KV-cache FP8
  (covered by `KILN_KV_CACHE_FP8`, opt-in) is the most obvious future row
  here.
- **Consistency**: post-#536 decode tok/s spread is 1.8 % across three
  seeds, well inside the 3–5 % run-to-run noise floor for kiln-bench on
  A6000 documented in the agent note `kiln-bench-median-of-3-noise-floor`.

The remaining bs=1 decode lever is verifier-step cost reduction inside the
self-spec path so that α=0.68 actually clears break-even (or, equivalently,
pulling break-even down toward current α). At larger batch / paged-prefill /
multi-turn-prefix workloads the dominant lever shifts to the radix prefix
cache (PR #520 / #521 / #523) and to longer-context KV residency.

## Reproducing

### Provision
Any A6000 / RTX 6000 Ada / L40S / A100 / H100 class GPU with driver ≥570 and
CUDA 12.4+. The `kiln-runpod` image
(`ghcr.io/ericflo/kiln-runpod:latest`) bakes the toolchain, sccache, and
build cache wiring; see `deploy/runpod/`.

### Fetch the model
```bash
hf download Qwen/Qwen3.5-4B --local-dir Qwen3.5-4B
```

### Build kiln
```bash
export KILN_CUDA_ARCHS=86   # 86 on A6000/3090/4090, 89 on RTX 6000 Ada / L40S, 80 on A100, 90 on H100
cargo build --release --features cuda --bin kiln-bench
```

### Build llama.cpp
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j --target llama-bench llama-quantize
python3 convert_hf_to_gguf.py Qwen3.5-4B --outfile Qwen3.5-4B-bf16.gguf --outtype bf16
```

### Run
Current single-stream kiln protocol (matches PR #535 / PR #536):

```bash
KILN_W4A16=1 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench \
  --model-path Qwen3.5-4B \
  --paged \
  --prompt-tokens 512 --max-output-tokens 128 \
  --skip-training \
  --prompt-subset humaneval \
  --chat-template \
  --latency-only \
  --temperature 0.0 --seed 1 > kiln-bench.json
```

llama.cpp side (still on the `f3d5089` head-to-head shape, 512 → 256):

```bash
./llama.cpp/build/bin/llama-bench \
  -m Qwen3.5-4B-bf16.gguf \
  -p 512 -n 256 -r 3 -o json > llama-bench.json
```

Raw JSON for the historical RTX 6000 Ada head-to-head is checked in under
[`bench-results/`](bench-results/). The post-#536 A6000 per-seed CSV is at
[`docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.csv`](docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.csv);
the post-#534 NVTX hotspot tables are in
[`docs/archive/phase-c/phase-c65/`](docs/archive/phase-c/phase-c65/).

## macOS / Apple Silicon (Metal)

Kiln also runs on Apple Silicon via candle-metal. Measured numbers are
not yet in this doc — the methodology below lets a contributor with
M3/M4 Max hardware drop them in.

### Provision
M3 Pro/Max, M4 Pro/Max, or M2 Ultra. Xcode Command Line Tools only
(full Xcode is **not** required — candle-metal-kernels JIT-compiles MSL
at runtime). Rust stable. No x86_64 Macs — Metal perf there is
unusable.

### Build kiln
```bash
cargo build --release --features metal --bin kiln-bench
```

### Run
```bash
./target/release/kiln-bench \
  --model-path Qwen3.5-4B \
  --prompt-tokens 512 --max-output-tokens 256 \
  --paged --skip-training > kiln-bench-metal.json
```

The JSON output includes a top-level `"backend": "metal"` field so
mixed-platform reports can split runs without parsing GPU names.

### Compare against
- **llama.cpp Metal**: `cmake -B build -DGGML_METAL=ON` (no CUDA
  dependency), then the same `llama-bench` invocation.
- **MLX-LM**: Apple's reference inference stack; good baseline for
  Apple Silicon peak perf.

Kiln's Metal backend uses `candle_nn::ops::sdpa` for both prefill and
paged decode (the latter via an `index_select` gather from the paged
pool). GDN linear-attention layers run on the portable candle
composition.
