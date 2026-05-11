# Kiln Server Changelog

## Unreleased — Phase 2 Vulkan training hardening

Active branch since v0.2.15. Targets the Strix Halo / unified-memory APU
training path that previously OOM-killed `kiln serve` and (after the first
Vulkan training-route changes) twice hard-hung the host on the
`/tmp/sft-data.jsonl` SFT repro.

### Added
- vulkan: in-op chunking in `VulkanLinearOp` — oversized BF16-packed
  matmuls split along the output dim (forward) or batch dim (backward)
  via the existing offset/transposed kernels. Each per-chunk submit
  calls `queue_wait_idle()` for compositor preemption. Per-chunk FLOP
  ceiling tunable via `KILN_VULKAN_LINEAR_MAX_GFLOP` (default 20 GFLOP,
  ≈800 ms per submit at 25 TFLOPS — comparable to FLCE's empirically
  safe per-chunk cadence).
- vulkan: `linear_prefill_apply_offset` sub-chunks internally when a
  single chunk_len would exceed the FLOP ceiling, instead of bailing
  to FLCE's CPU fallback. Same kernel, same buffer, finer-grained
  submits.
- vulkan: new `sdpa_prefill_f32.comp` shader replaces the buggy
  `flash_attn.comp` placeholder. F32 in/out, online-softmax, optional
  causal mask, head_dim ≤ 128. Wired into `flash_attn_prefill_vulkan`
  behind opt-in env var `KILN_VULKAN_SDPA=1`.
- vulkan: Phase 4.2 — `sgd_step_f32.comp` shader,
  `dispatch_sgd_step_f32` helper, and `BackendRuntime::dispatch_sgd_step`
  trait method. Vulkan impl looks up both operands in
  `RESIDENT_ACTIVATION_REGISTRY` and falls back to the candle CPU
  path if either is not resident. Drops in for the trainer once
  Phase 4.1 lands and `TrainableLoraParams` are registry-resident.
- vulkan: Phase 4.1 partial (LoRA Vars on device, forward inference
  path).
  - `TrainableLoraParams::register_with_backend` uploads each LoRA
    Var to the resident activation registry at init time. Production
    callers (sft_train, grpo_train) wire it.
  - `BackendRuntime::lora_delta_resident` hook computes
    `(x @ A.T @ B.T) * scale` against registry-resident A and B
    via two dispatches of the existing transposed bf16 kernel.
    Plumbed into `add_lora_delta_to_base` in forward.rs.
  - **Gated to inference-only**: the on-device dispatch returns a
    candle-leaf Tensor with no autograd back-link, so it's only
    invoked when neither x, A, nor B is autograd-tracked. Training
    still uses the candle CPU `compute_lora_delta` path. A future
    CustomOp3 wrapper (with analytic LoRA backward) will make the
    on-device path training-safe.
  - Registry encoding bifurcated by dtype:
    BF16 tensors store packed BF16 bytes (2 bytes/elem) so every
    `load_weight(idx) = data_w[idx >> 1]` kernel reads them
    correctly; F32 tensors store F32 bytes (preserves the original
    boundary-state resolve path). `resolve_resident_activation`
    knows about both encodings.
  - `BackendRuntime::update_resident_activation` keeps the registry
    buffer in sync with candle CPU `var.set(...)` after the SGD
    step. Wired into `sgd_step` and `sgd_step_from_map`. Without
    this, `lora_delta_resident` would forever read the init bytes.
- training: Phase 3.2 partial drop also covers the tiled checkpoint
  paths — boundary_states[seg_idx]'s candle CPU mirror is replaced
  with a 1-element BF16 stub right after the tiled subroutine
  returns (it took `&boundary_states` so couldn't mutate; the
  outer loop does the drop). Frees ~9 MB per boundary on Vulkan
  in tiled mode too.
- inference: Phase 4.1 step 4 — `Model::load_adapter` now registers
  the adapter's A/B tensors in the resident activation registry,
  and `Model::unload_adapter` (and the load-replace path) evicts
  the previous adapter. New `LoraWeights::register_with_backend`
  mirrors the trainer-side TrainableLoraParams API. Effect: an
  inference request that uses an adapter now dispatches the LoRA
  delta on the GPU (via `lora_delta_resident`) instead of going
  through candle CPU `compute_lora_delta`.
- core: tests — `kiln_core::env_flag::TEST_ENV_LOCK` is now `pub`
  (cfg(test)) so sibling test modules (vram, etc.) can hold the
  same mutex when mutating env vars. Deflakes
  `test_unified_memory_apu_corrects_oversized_carveout` which
  used to race with `test_unified_memory_reserve_env_override`.
- vulkan: `BackendRuntime::resolve_resident_activation` read-back
  hook + Vulkan impl using `VulkanBuffer::read_back` +
  `kernels::create_tensor_from_data`. API surface for Phase 3.2's
  actual memory-saving wiring (drop the candle CPU mirror after
  registering, re-materialise from the device buffer when the
  recompute pass needs it).
- core: `kiln_core::env_flag::env_flag(name, default) -> bool` and
  `env_tristate(name) -> Option<bool>` helpers. Centralised the
  truthy/falsy spellings (`1`/`true`/`yes` and `0`/`false`/`no`,
  case-insensitive, whitespace-trimmed) so every `KILN_*` flag
  parses the same way and unrecognised values fall back to the
  default rather than silently flipping behaviour. Refactored
  4 call sites (KILN_VULKAN_LINEAR, KILN_VULKAN_SDPA, KILN_W4A16,
  KILN_VULKAN_FLCE, KILN_VULKAN_RMSNORM_TRAINING).
- vulkan: `register_resident_activation` bails silently on a
  zero-byte tensor (some drivers reject `vkCreateBuffer(size=0)`);
  caller falls through to its CPU path.
- telemetry: one-shot trace for first offset sub-chunked dispatch
  (`linear_prefill_apply_offset` engages internal sub-chunking when
  a FLCE chunk_len exceeds the FLOP ceiling).
- training: Phase 3.2 partial — both `checkpointed_forward_backward`
  and `checkpointed_grpo_forward_backward` now prefer
  `resolve_resident_activation` over `clone()` for the per-segment
  recompute boundary input. Doesn't free the candle CPU mirror yet
  (pending Phase 4.x storage interception) but exercises the
  resolve path inside real training; any registry bytes-divergence
  surfaces as a loss difference vs the non-resident path.
- server: `resident_activation` and `sgd_step_on_device` fields
  added to the "Vulkan training acceleration profile" startup log.
- vulkan: Phase 3.1 hooks — `supports_resident_activation`,
  `register_resident_activation`, `evict_resident_activation`,
  `has_resident_activation` on `BackendRuntime` plus a
  `RESIDENT_ACTIVATION_REGISTRY` impl on `VulkanBackend`. Trainer
  call sites in `checkpointed_forward_backward` and
  `checkpointed_grpo_forward_backward` invoke the lifecycle hooks
  at boundary state push/eviction; gated on
  `supports_resident_activation()` so non-Vulkan backends pay zero
  overhead.
- server: "Vulkan training acceleration profile" startup log surfacing
  the on/off state of every `KILN_VULKAN_*` training-path env var.
- server: HTTP 413 rejection messages from `/v1/training/*` include
  a `vram_source=...` clause when the corrected budget came from the
  unified-memory path (makes `KILN_TRAINING_MEMORY_RESERVE_GB` the
  obvious knob).
- core: `EffectiveBudget` accessor + `vram_source` field on the "GPU
  memory budget" log so the corrected unified-memory budget is honest
  about its provenance (`linux-drm-sysfs-unified` vs the raw DRM
  carveout it overrides).
- telemetry: one-shot tracing for first chunked
  `VulkanLinearOp::cpu_fwd` and `::bwd` dispatch (logs chunk count,
  per-chunk GFLOP) and first
  `VulkanBackend::register_resident_activation` call.
- docs: Phase 2.1 CPU-fallback matmul leak audit, Phase 2 hardware
  validation runbook, and canonical env-var reference in
  `docs/audits/`.
- docs: env-var documentation for `KILN_GPU_MEMORY_GB`,
  `KILN_TRAINING_MEMORY_RESERVE_GB`, and the unified-memory detection
  heuristic in `kiln.example.toml`.

### Changed
- training: FLCE provider auto-engages at `active_count ≥ 16` (was
  `active_count × num_chunks ≥ 50_000`). The 50K threshold was tuned
  against the pre-Phase-2 baseline; post-host-crash the comparison is
  FLCE-chunked-Vulkan vs the unfused lm_head dispatch that hung the
  host. Any non-trivial supervised batch now routes through the
  chunked FLCE path.
- vulkan: `KILN_VULKAN_LINEAR` reverted to default-OFF until the
  in-op chunking has been load-validated end-to-end on real hardware.
  Smaller-shape projections (T≤256) ran ~6% faster with this on at
  bit-exact loss; the runbook in `docs/audits/` walks the operator
  through enabling it safely.

### Fixed
- vulkan: lm_head training-time forward queuing ~4.36M workgroups in
  a single submit on a 40-CU APU (the root cause of two host
  hard-hangs on `/tmp/sft-data.jsonl`). Mitigation lands as a
  combination of the FLCE auto-engagement (so the lm_head matmul
  goes through chunked FLCE for the SFT loss path) and the in-op
  chunking (so any oversized matmul that does reach `VulkanLinearOp`
  is split into TDR-safe per-submit pieces).
- transposed weight cache writer: replace blocking
  `thread::sleep(initial_delay)` with `recv_timeout`-based wait so
  a queued message wakes the writer immediately even mid-deferral.
  Fixes a flaky-under-`cargo test`, fine-under-`cargo nextest`
  unit test (`queue_cache_write_persists_payload_on_background_writer`)
  whose 2-second deadline timed out when an earlier test had
  initialized the OnceLock'd writer with the production-default
  120 s initial delay.

## kiln-v0.2.15 — 2026-05-10

A short follow-up cut covering 15 commits since v0.2.14, focused on CUDA
decode-path fusion and a macOS Metal build fix. No desktop source changes;
the aligned desktop release simply re-pins to the new server payload.

### Changed
- cuda: combine full-attention QKV projections into a single GEMM at decode
  time, mirroring the GDN AB combine in v0.2.14.
- cuda: combine GDN AB projections at prefill (extending the decode-time
  fusion) and use the combined projection on short prompts.
- cuda: enable GDN prefill gates by default and route mixed GDN batches
  through the row loop, with the row-loop forced path generalized for batched
  GDN.
- cuda: fast-path paged KV cache writes and route CUDA decode through paged
  attention end-to-end.
- cuda: fuse decode QKV prep, fuse GDN decode RMSNorm, and fuse the LoRA
  decode add (new `kiln-rmsnorm-kernel/csrc/fused_lora_add.cu`) so single-token
  decode dispatches fewer kernels per layer.
- cuda: use empty CUDA outputs for overwrite kernels and align the default
  ModelRunner CUDA graph behavior.
- training: default CUDA projection drop on for the training fit path so
  training memory residency tracks the inference path.

### Fixed
- macos: pass `backend=None` to `add_lora_delta_to_base` from the
  metal-feature-gated decode path so `cargo build --features metal` compiles
  again. The CUDA LoRA decode add fuse landed in this cut added a `backend`
  parameter and missed the Metal call site (#1019).

## kiln-v0.2.14 — 2026-05-10

This is a large release covering ~325 commits since v0.2.13: a new Vulkan
backend, an embedded server-UI overhaul, replayable LoRA storage, the Phase 12
batched-decode hot path, and continued CUDA / Metal kernel fusion.

### Added
- vulkan: bring up a Vulkan inference backend covering GDN (recurrent, conv1d,
  in-projection, gated norm, QK norm), MLP (gate/up/down), batched gemv, and
  paged decode. Ships with f32 + bf16 paths, rowpair / rowquad batched gemv
  variants, batched chunk transfers, fused-submit conv1d prefill (now on by
  default), and chained MLP dispatches. BF16 GDN state and device-local
  recurrent batch state cut readback overhead and keep state resident on-device.
- training: deterministic replayable LoRA storage with parent lineage so a
  trained adapter records the exact replay set + parent chain it descended from.
  Storage is content-addressed and reproducible across machines (#1015).
- server UI: total visual overhaul of the embedded server dashboard — clearer
  status, request feed, training, and adapter panels; live regions for
  accessibility; mobile-friendly forms; sample training payloads; CLI/REST
  cross-links from every panel (#1009, #1013).
- windows: Windows CUDA release zips now bundle the CUDA 12.4 redist DLLs
  (cudart, cublas, cublasLt, cuRAND, nvrtc) so users without the CUDA Toolkit
  installed no longer hit `cublas64_12.dll was not found` on launch (#726).

### Performance
- phase 12 batched paged decode: per-stream graph replay, per-request
  `KvWriteSlot` to eliminate the prefill mutex contention that bottlenecked
  multi-stream decode, FA-2 varlen paged decode for batched (c>1) decode, and
  batched paged decode API surface (#762, #968, #995, #996, #998, #1000, #1002,
  #1008).
- cuda fusion: fold GDN QK-norm into the recurrent kernel, fuse RoPE with QK,
  fuse MLP SiLU and multiply, fuse attention-gate sigmoid+multiply, BF16 GDN
  state, inference-only RMSNorm fast path, projection-original memory
  compaction, rate-limited graph cache-full warnings.
- metal: full SDPA prefill, GDN in-proj rowpair / rowquad / serial-x2 fast
  paths, MLP gate/up rowquad gemv, batched gemv rowpair / rowquad / row-triple
  variants for batch sizes ≥3 / ≥4 / ≥5, LoRA delta paths covering rank-4 /
  rank-8 / serial-decode and all positive ranks, plus high-rank LoRA bench
  coverage and qkv LoRA-linear bench coverage.
- vulkan perf: enable GDN in-proj rowpair-batch3, GDN rowpair-batch3 chunk
  transfers, fused-submit conv1d prefill (now default), chained MLP dispatches,
  device-local recurrent batch state, BF16 inference state, skip the final GDN
  state readback. f32 GDN recurrent step routing, bf16 hybrid MLP gate/up/down,
  and gdn-recurrent kernel-stage profiling support.
- scheduler / decode batcher: reduce batched linear-state scatter copies, route
  through unexpanded GDN decode QK on both CUDA and Vulkan.

### Fixed
- compatibility: guard GDN fast paths from autograd tensors so training and
  inference share the same kernel surface without aliasing autograd graph
  state.
- compatibility: disable CUDA graphs by default on WSL CUDA and guard the CUDA
  decode batching default (and revert PR #1008's LM-head guard regression).
- ui smoke: `docs/site` + server UI v1 overhaul follow-up — fix CI smoke checks
  that broke against the new dashboard markup (#1013).
- adapters: fix CLI adapter API routes (#774); fix adapter API docs examples
  (#947); align training submission payload shapes (#739).
- quickstart / cli: friendly error when the kiln server is unreachable (#972);
  fix README quickstart timeout anchor (#896); fix quickstart training-status
  command (#765); fix GRPO CLI file help (#738); device init now uses `?`
  instead of a missing `anyhow::Context` import (#734).

### Documentation
- onboarding: cold-reader pass through README, QUICKSTART, and CLI help —
  surface the kiln health probe before the chat curl, surface model-weights
  download before the Docker block, deduplicate Desktop App install tables
  between README and QUICKSTART, surface BENCHMARKS.md from README nav,
  cross-link architecture and GRPO guides, drop legacy prefill notes, and
  collapse the QUICKSTART reader map / prerequisites / KV-cache OOM section
  (#964–#989, plus #969–#977).
- ui: explain decode and scheduler metrics for cold readers via tooltips
  (#971); persistent UI header help links (#798); accessible training tabs
  (#782) and live regions for dashboard panels (#783).
- governance: scrub stale publicity changelog entries and remove launch
  announcement surfaces (#718, #719).

## kiln-v0.2.13 — 2026-05-02

### Observability
- throughput: explain the round-10 workers=2 evidence as request mix and concurrency effects rather than a scheduler regression, and add request, prefill, decode histograms plus an active-request peak metric so future release-gate runs can separate queue shape from decode-path regressions (#712).

### Documentation
- docs: scrub changelog references to removed Phase 11 docs artifacts after the tracked files and page were removed (#717, #718).

## kiln-v0.2.12 — 2026-05-02

### Fixed
- reliability: complete the production-shaped prefill OOM hardening series from #694/#697/#699. KV auto-sizing now uses post-load CUDA residency instead of only the static model estimate, CUDA streaming prefill starts at 8k tokens for production-shaped prompts below the old 32k threshold, and new VRAM metrics expose the static estimate, post-load snapshot, and observed prefill peak (#701).

## kiln-v0.2.11 — 2026-05-02

### Fixed
- memory: bound retained `RealPrefixCache` GDN `LinearAttentionState` snapshots via `prefix_cache.max_entries` / `KILN_PREFIX_CACHE_MAX_ENTRIES`, and expose new prefix-cache state metrics so sustained workers=2 traffic cannot accumulate hidden ~49 MiB state entries into post-v0.2.10 OOM cascades (#697).

## kiln-v0.2.10 — 2026-05-02

### Fixed
- server: route prefix-cache prefill through the tiled/streaming long-prefill dispatcher so first-touch 43k-token prompts no longer bypass the GDN activation-bounded path (#686).

### Observability
- metrics: add `kiln_request_prefill_tokens_completed` to show live long-prefill progress instead of making operators infer whether a request is wedged from latency alone (#686).

### Configuration
- server: raise the default `request_timeout_secs` from 300 to 600 seconds so the production default is honest for long-prefill workloads; explicit TOML/env overrides remain unchanged (#686).

## kiln-v0.2.9 — 2026-05-01

Phase 11 reliability, throughput, and onboarding release. Cancel-drain on
prefill-timeout (#666) closes the v0.1.0 reliability blocker (#664); a
prefix-cache double-counting bug (#674) is root-caused and fixed, allowing the
workers=2 emergency serialize shim from #672 to be reverted (#675) — restoring
+23% throughput / −75% p99 ITL on the default config. Also lands the
tools-bearing chat-template render fix (#653), the Phase 11 demo and onboarding
docs (#669–#671), and the post-v0.2.8 Phase 10 audits (closure, FleCE T=16384
OOM probe, GDN training-streaming reachability) and Phase A FLCE A6000
validation accumulated since the v0.2.8 cut.

### Fixed
- server: cancel-drain `spawn_blocking` generation on prefill timeout — closes the v0.1.0 reliability blocker (#664) where a stuck prefill would orphan the generation thread (#666).
- prefix-cache: never return retained blocks in `evicted_blocks` — root cause of the workers=2 prefill cascade (#673) that motivated the #672 emergency serialize shim (#674).
- server: revert the workers=2 emergency serialize shim from #672 now that #674 fixes the underlying prefix-cache double-decrement (#675).
- template: tools-bearing chat-template render — register the `tojson` minijinja filter and pass `tool_calls.arguments` as a dict so Qwen3.5-4B's chat_template stops 4xx-ing on `unknown filter: tojson` and on string-typed arguments (#653).

### Performance
- Reverting the #672 serialize shim (via #675, enabled by #674's prefix-cache fix) restores +23% throughput and −75% p99 ITL on the default workers=2 config relative to the shim's exclusive `GpuCoordinationLock` path.

### Memory
- memory: relax auto KV cache cap on CUDA — let memory-aware sizing drive instead of clipping at `max_position_embeddings.div_ceil(block_size)`, which had hard-capped Qwen3.5-4B at ~8.6 GiB / 16384 blocks (one full-context window) regardless of available VRAM (#655).

### Documentation
- docs(demo): asciicast script + asciinema player scaffolding for the 60s online-learning demo (#669).
- docs(audits): Phase 11 ops checklist verification (#670).
- docs/demo: record canonical 60s online-learning asciicast and update `SCRIPT.md` to match shipping API (#671).
- docs(quickstart): OpenAI tools/function-calling example (#654).
- docs: workers=1 workaround for tools-bearing long prompts (refs #664, #656) — operator guidance pending the #666 cancel-drain fix (#665).

### Tests
- ci(chat-template): vendor Llama 3.1 + Qwen2.5 fixtures with stress-render tests (closes #657) (#658).
- ci(chat-template): vendor Mistral v0.3 [INST] tools-bearing fixture with stress-render test (#660).
- ci(chat-template): vendor DeepSeek V3 fixture with tool-calls render test (#661).
- ci(chat-template): vendor Hermes-3-Llama-3.1 fixture with tool-calls render test (#662).
- ci(api): integration smoke test for tools-bearing chat-completions render path (closes #659) (#663).

### Audits
- Phase 9 v0.1.0 release-readiness audit. New `docs/audits/PHASE9_V0_1_0_READINESS.md` walks each Phase 9 checklist item from the kiln project description and reports concrete evidence (PR/release/file) per item. **Verdict: Phase 9 is shipped — the "v0.1.0 release with semantic versioning" goal is stale.** kiln-v0.1.0 was tagged on 2026-04-19; the production line is now at kiln-v0.2.8 (Sigstore-signed build provenance, GHCR `kiln-server` Docker image auto-publishing on every `kiln-v*` tag, the `docs/site/` landing page live at https://ericflo.github.io/kiln/, and all 12 findings of `docs/audits/security-audit-v0.1.md` resolved). PR #651's framing ("what remains is an audit pass + the v0.1.0 semver cut") was written without rechecking the release index — the v0.1.0 tag predated that doc by 10 days; this audit is the audit pass and there is no remaining v0.1.0 cut to do.
- Phase 10 (Liger Kernel Integration) closure consolidating §1–§3 verdicts (PR #650 et al). New `docs/audits/PHASE10_CLOSURE.md` is the single state-of-play pointer for the Phase 10 chapter ledger (§1 RMSNorm fusion, §1.5 FLCE Phase B, §2 Mode B trace, §3 next-kernel candidate audit + FleCE T=16384 OOM probe), reproduces #649's math-ceiling table, and ranks three post-Phase-10 pivot directions: Phase 9 v0.1.0 release prep, GDN training-time streaming follow-ups (active in-flight: #635/#636/#637), and a LoRA precision study (FP32→BF16 with FP32 accumulate) targeting the ~42% FP32 SGEMM hotspot surfaced in #649. Records the explicit non-goal that no further Liger kernel ports are planned without a fresh re-profile that surfaces a candidate ≥1.05× ceiling. `PROFILING.md` gains a top banner pointing readers at the closure doc.

### Phase 10 — training-time streaming GDN prefill (impl)
- Implemented the audit-recommended remediation §1 from PR #634: added a streaming branch inside `model_forward_segment` that tiles GDN layers along T while threading `LinearAttentionState` per tile (full-attention layers always run monolithic — training has no KV cache to thread). New helper `gated_deltanet_forward_streaming` slices `[B,T,hidden]` into tiles of `KILN_STREAMING_TILE_TOKENS` (default 8192, must be a multiple of `GDN_CHUNK_SIZE=64`), calls `gated_deltanet_forward` per tile, and `Tensor::cat`s the results back into `[B,T,hidden]`. Bit-exact with the monolithic call by construction (the state hand-off matches `model_forward_paged_streaming_with`). Two new CPU parity tests (`test_gated_deltanet_forward_streaming_matches_monolithic_cpu`, `test_model_forward_segment_streaming_matches_monolithic_cpu`) pass. **A6000 SFT bench result: AMBER — dispatch is reachable now (CPU parity passes; T=8192 STREAMING ON tile=4096 takes 1.3 s of GPU work before OOM vs 0.7 s for monolithic; T=2048 STREAMING-ON loss matches STREAMING-unset bit-for-bit) but T=8192 SFT still OOMs at the 48 GB A6000 ceiling because the autograd recompute saves all per-tile activations simultaneously.** Net: necessary infrastructure for any future training-time GDN streaming work, but on its own this PR doesn't unblock T=8192 SFT — the next slice is the audit's remediation §2 (per-tile forward+backward inside `checkpointed_forward_backward`). See `docs/audits/PHASE10_GDN_TRAINING_STREAMING_IMPL.md` for the full validation, dispatch-reachability evidence, and root-cause analysis. Raw log: `docs/flce_phase_a_streaming_impl_raw_2026-04-29.log`.

### Audits
- Audit-recommended FleCE T=16384 OOM probe on A40 (sm_86, 46 GiB ceiling — A6000 capacity-blocked at task start, A40 is the audit's named arch-equivalent fallback). Two arms: (1) default A40 path (`fused_path=OFF`, A40 below the PR #644 ≥47 GiB gate) → status=OOM, peak=45,447 MiB / 46,068 MiB, delta=29,104 MiB above post-load baseline, step=9.44 s; (2) force-fused (`KILN_FORCE_RMSNORM_KERNEL=1`, simulates A6000's gate-ON dispatch on A40) → status=OOM, peak≥45,095 MiB / 46,068 MiB, delta=28,752 MiB, step=0.89 s (poller likely missed the actual peak in the early-failure case). **§3 closes for FleCE — zero ROI even though T=16384 OOMs.** The ~29 GiB delta is GDN/MLP saved activations across the 8 grad-checkpoint segments, *not* lm_head logits (Phase B's chunked-vocab CustomOp1 already avoids `[T_active, V]` materialization). FleCE Phase C — Liger-style vocab-axis chunking on top of Phase B's existing vocab-axis chunking — addresses the head, which is already <1 GiB at T=16384 with Phase B; it does not move the GDN/MLP-dominated bottleneck. Path forward for long-context SFT is GDN training-time streaming (PR #635/#636/#637 follow-ups), not a head-side fusion. Both audit branches resolve to "no kernel" under the actual mechanism. See the `Addendum 2026-04-29 — FleCE T=16384 OOM probe (closure)` section in `docs/audits/PHASE10_S3_CANDIDATE_PREFLIGHT.md` for full results, mechanism, A6000 implications, and reproduction. Raw logs: `docs/flce_phase_b_t16384_oom_probe_a40_raw_2026-04-29.log` (arm 1) and `docs/flce_phase_b_t16384_oom_probe_a40_fused_raw_2026-04-29.log` (arm 2). Also lands the `kiln-server` cuda → `kiln-train/cuda` Cargo.toml propagation fix that PR #647 and PR #649 each had to apply locally on the pod (see agent note `kiln-server-cuda-doesnt-propagate-to-flce`).

- Audited whether the existing Phase 7 streaming GDN prefill (`KILN_STREAMING_PREFILL`) is reachable from the SFT training path as a remediation for Finding 2 of the Phase A validation (T=8192/T=16384 SFT OOM on A6000). **Result: RED — streaming dispatch lives only in `model_forward_paged_streaming*`, none of which the trainer calls; the env flag is parsed but has no effect on training.** Empirical confirmation on A6000: T=2048 STREAMING ON peak VRAM matches STREAMING unset to 0 MiB and loss is bit-identical; T=8192 STREAMING ON at three tile sizes (default/4096/2048) all OOM identically at the 48 GB ceiling in 0.7 s. Closing Finding 2 requires net-new implementation work (e.g., a streaming branch inside `model_forward_segment`), not configuration. See `docs/audits/PHASE10_GDN_TRAINING_STREAMING.md` for source/empirical evidence and the remediation menu. Raw log: `docs/flce_phase_a_streaming_raw_2026-04-29.log`.

### Validation
- Validated Phase 10 Phase A FLCE on A6000 (48 GB) at T ∈ {2048, 8192, 16384} with rank-8 LoRA and gradient checkpointing ON. **Result: RED — Phase A is insufficient on A6000.** Two findings: (1) `KILN_USE_FLCE=1` fails before completing a step at T=2048 with `matmul is only supported for contiguous tensors` — the V-axis chunk of `embed_tokens_t` is non-contiguous and the chunked-vocab matmul rejects it; (2) even with that bug fixed, T=8192 and T=16384 still pin peak VRAM at the 48 GB A6000 ceiling, indicating GDN-side activations (not the head) dominate at long T. See `docs/audits/PHASE10_FLCE_PREFLIGHT.md` (Phase A validation section, 2026-04-29) for the table, raw log, and required follow-ups. Bench: `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`.

## kiln-v0.2.8 — 2026-04-29

Patch release: post-v0.2.7 dep bumps and CI hygiene. First release that ships
[Sigstore-signed build provenance attestations](SECURITY.md#supply-chain-provenance)
for `kiln-v*` artifacts and the GHCR image.

### Reproducibility / release
- ci: attach build provenance attestations to release artifacts via `actions/attest-build-provenance`. `server-release.yml` and `docker-server-release.yml` now mint Sigstore-signed attestations bound to each artifact's sha256 and the workflow run; verify with `gh attestation verify <artifact> -R ericflo/kiln` (#627).

### CI / release
- ci: gate Apple codesign secrets (`APPLE_*`) to tagged releases so non-tag CI runs (PRs, branches) skip macOS code signing instead of failing on missing secrets (#625).
- ci: also gate Tauri signing secrets (`TAURI_SIGNING_*`) to tagged releases for the same reason as #625 (#626).
- chore(actions): bump `actions/upload-artifact` from 4 to 7 (#614).
- chore(actions): bump `actions/checkout` from 4 to 6 (#615).
- chore(actions): bump `docker/metadata-action` from 5 to 6 (#613).

### Dependencies
- chore(deps): bump `rand` from 0.9.4 to 0.10.1 (#616).
- chore(deps): bump `safetensors` from 0.5.3 to 0.7.0 (#617).
- chore(deps): bump `toml` from 0.8.23 to 1.1.2+spec-1.1.0 (#618).
- chore(deps): bump `tokenizers` from 0.22.2 to 0.23.1 (#619).

### Documentation
- docs(site): scaffold landing page + GitHub Pages workflow under `docs/site/` so https://ericflo.github.io/kiln/ has a published presence (Phase 9) (#624).

### Tests
- fix(test): serialize env-mutating config tests to eliminate flakiness when run in parallel (#623).

## kiln-v0.2.7 — 2026-04-28

Security release: closes all 12 findings in `docs/audits/security-audit-v0.1.md`.
Adds path-traversal hardening on adapter routes, per-route body-size limits,
webhook redirect-disable, training queue + tracked-jobs caps, adapter-dir
disk caps, LRU eviction for the composed-adapter cache, loopback default bind,
and reproducible Docker builds. No new features and no GPU/runtime breaking
changes.

### Security
- Fix path traversal in `DELETE /v1/adapters/:name` and `POST /v1/adapters/load` (Phase 9 audit §2b/§2c, HIGH).
- Validate source adapter names in `POST /v1/adapters/merge` (Phase 9 audit §2d, LOW).
- Per-route body-size limits on training + completions endpoints: 64 MiB for `/v1/train/sft` and `/v1/train/grpo`, 8 MiB for `/v1/chat/completions` and `/v1/completions/batch` (Phase 9 audit §1, LOW).
- Disable HTTP redirects on the training-completion webhook client to prevent server-side redirect chasing into internal infra (Phase 9 audit §7, item 10, LOW).
- Cap training queue depth (audit MEDIUM §4 part 1) — bounded queue with explicit reject when full (#607).
- Cap tracked-jobs map size and TTL Completed/Failed entries (audit MEDIUM §4 part 2) — prevents unbounded memory growth (#608).
- Cap total adapter_dir disk usage on upload (audit LOW §8 / item 9) — `KILN_ADAPTERS_DIR_MAX_BYTES` rejects uploads that would exceed the cap (#620).
- LRU eviction for `.composed/` adapter cache (audit LOW §8 / item 8) — bounded byte-size + entry-count caps with oldest-mtime-first eviction (#621). Closes the last open finding in security-audit-v0.1; all 12 findings now resolved.

### Changed
- Default server listen host changed from `0.0.0.0` to `127.0.0.1` (loopback). Set `server.host = "0.0.0.0"` or `KILN_HOST=0.0.0.0` to accept remote connections; pair with a trusted reverse proxy. Closes security-audit-v0.1 MEDIUM §9.

### Reproducibility / release
- docker: use `--locked` in `deploy/Dockerfile` cargo builds so the published `ghcr.io/ericflo/kiln-server` image matches the exact Cargo.lock dependency set (#601)
- Pin cargo-deny-action to a specific commit SHA and document deny.toml schema (audit LOW §11) (#612).

### Documentation
- docs(security): document training-data trust invariants (security audit §3 / item 12) in README + QUICKSTART. Calls out that `/v1/train/sft` and `/v1/train/grpo` apply a faithful gradient update to anything POSTed — kiln validates structure, not semantics — so the operator's training corpus must be treated as security-sensitive.

## kiln-v0.2.6 — 2026-04-26

Patch release: 3 server bug fixes + first release with bundled
THIRD_PARTY_LICENSES.md asset (cargo-about, MIT/Apache/BSD-only).

### Bug fixes
- metal: disable candle SDPA full path entirely to eliminate intermittent NaN on Apple Silicon (ff84800)
- server: re-emit prefilled `<think>\n` opener in chat-completion responses so streaming clients see it (7548e5a)
- server: split `<think>...</think>` content into llama.cpp-shaped `reasoning_content` field on chat completions (b1ae711)

### CI / release
- First release to ship THIRD_PARTY_LICENSES.md alongside binaries (#598, #599)

## kiln-v0.2.5 — 2026-04-26

Patch release: 4 server bug fixes since v0.2.4.

### Bug fixes
- server: close use-after-free race in prefix-cache streaming path (fad7c6b)
- server: make `stream: true` actually stream tokens in real time (11062f8)
- server: load model chat template so Qwen3.5 gets the `<think>\n` prefix in the rendered chat (e1fcc16)
- metal: bypass candle SDPA full kernel for `8 < q_seq < bq` to avoid a kernel crash (c7cf1ab)

## kiln-v0.2.4 — 2026-04-26

CI / release-prep release. No user-facing API or behavior changes since
v0.2.3; this cut exists to publish the kiln-server Docker image to GHCR
and to validate the auto-publish-on-platforms-green workflow end-to-end.

### CI / release
- Auto-publish the GitHub Release once all 3 platform jobs succeed, instead of leaving each tag in Draft (#592)
- Publish prebuilt server Docker image to `ghcr.io/ericflo/kiln-server` on every `kiln-v*` tag (#593)

## kiln-v0.2.3 — 2026-04-26

Phase 8 advanced features release: batch generation, adapter upload/download,
TIES + concatenation merge modes, per-request adapter composition, and webhook
notifications on training completion. Also lands the Phase 7 `/ui` adapter
controls, refreshed Phase 8 documentation (QUICKSTART, README, ARCHITECTURE,
plus a new docs/GRPO_GUIDE.md), and governance hygiene marking all workspace
crates `publish=false` and tightening cargo-deny wildcards to `deny`.

### Phase 8 advanced features
- POST /v1/completions/batch — efficient multi-prompt batch generation API for GRPO (#583)
- POST /v1/adapters/upload — multipart tar.gz import (#577)
- GET /v1/adapters/{name}/download — streaming tar.gz export (#575)
- TIES merge mode for /v1/adapters/merge (#578)
- Concatenation merge mode for /v1/adapters/merge (#579)
- Per-request adapter composition: stack multiple LoRAs with scaling on /v1/chat/completions (#581)
- Webhook notifications on training completion (#582)

### Phase 7 UI
- Add adapter download / upload / merge controls to `/ui` dashboard (#586)

### Docs
- Document Phase 8 API surface in QUICKSTART.md (#584)
- Refresh README + CHANGELOG for Phase 8 (upload/download/merge modes/composition/batch/webhooks) (#585)
- Refresh ARCHITECTURE.md for Phase 8 (upload/download, TIES/concat merge, composition, webhooks, batch generation) (#587)
- Add docs/GRPO_GUIDE.md with worked verifiable-rewards examples (math, JSON, code) (#588)

### Cleanup
- Move audit/preflight docs into docs/audits/, drop runtime log (#574)

### Governance / hygiene
- Mark workspace crates `publish=false`; tighten cargo-deny wildcards from warn to deny (#589)

### Test fixes
- Rewrite test_upload_rejects_path_escape_in_archive to actually emit a traversal tarball (#580)

## kiln-v0.2.2 — 2026-04-25

Coordinated release aligned with desktop-v0.2.2. Supersedes the unpublished
kiln-v0.2.1 draft; all v0.2.1 changes are included here. Highlights since
v0.2.0 are the radix prefix cache reuse path, more Metal/CUDA decode
fusions, governance docs, and dependency hygiene.

### Phase 7 prefix cache + decode reuse
- Implement radix prefix cache core (#512)
- Wire real append prefix cache (#515) and streaming real prefix cache reuse (#520)
- Use prefix cache with CUDA graphs and warn when bypassed (#518, #521)
- Expose prefix cache metrics (#513)
- Speed up greedy paged prefill defaults (#519)
- Default CUDA streaming prefill for long prompts and lower Metal threshold (#511)
- Refresh post-#521 profiling artifacts (#522)

### Phase 6 / Phase 7 Metal + CUDA fusions
- Fuse Metal attention output gate (#514)
- Fuse Metal GDN gates with recurrent decode
- Fuse Metal contiguous paged decode attention (#501)
- Fuse Metal GDN prefill conv split (#499) and Metal paged KV slot writes (#497)
- Fuse GDN recurrent RMSNorm decode (#496)
- Add CUDA GDN qk norm GQA fast path (#500)
- Add opt-in CUDA GDN decode fuse hook (#498)
- Route shared Metal greedy decode through argmax (#510)
- Defer transposed cache writer (#508); make transposed cache writes reliable (#506)
- Precompile Metal kernels before prewarm lock (#505)
- Batch MTP verifier argmax (#493)

### MTP audits and α-stability work
- H15c stratified C29 v2 reject-row probe (#529)
- H17 SGLang and H15c/H17b/H15a vLLM α microbenches (#530, #532, #533)
- H18 hand-rolled HF transformers MTP α reference (#534)
- H16 external-α reference options audit (#531)
- MTP acceptance-rate state-of-play audit (#527)
- End-to-end native-MTP self-spec decode bench post-#535 (#536)

### Phase 7 CLI / UX
- Recent requests panel on `/ui` dashboard (last 100) (#551)
- Live decode tok/s + p50/p99 ITL on `/ui` dashboard (#550)
- `kiln health` pretty-printed tree output + `--json` escape hatch (#549)
- `kiln train status` CLI subcommand + fix post-submit hint (#548)
- Surface structured server error hints in CLI (#545)
- `KILN_LOG_FORMAT=auto` — TTY-detect pretty vs JSON default (#544)
- GPU name + VRAM in startup banner (#543)
- ProgressBars for model load, SFT, GRPO (#540, #541, #542)

### Server runtime
- Move health adapter scan off runtime
- Document phase 7 prefix cache reuse benchmark
- Audit kiln radix prefix cache vs SGLang RadixAttention (#526)
- Audit vLLM fused_recurrent_gated_delta_rule against kiln-gdn-kernel (#525)
- Kill-switch bisection ruled out a single fused-kernel owner of the post-#166 decode gap (#524)
- Prefix-cache A/B ruled out cache hooks as bench regression source (#523)
- Fast-guard disabled MTP debug taps; reduce safetensors loader map allocations
- RunPod task tasks no longer pin `KILN_CUDA_ARCHS` (#494)

### Governance + CI
- Add Dependabot config for cargo + GitHub Actions (#558)
- Add `cargo-deny` license/source/bans policy and CI check job (#555, #556)
- Add CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md (#552, #553, #554)
- Add GitHub issue + PR templates (#557)

### Dependencies
- Bump tokenizers 0.21.4 → 0.22.2 (#565)
- Bump indicatif 0.17.11 → 0.18.4 (#564)
- Bump console 0.15.11 → 0.16.3 (#563)
- Migrate to rand 0.9 (#567)
- Bump cc in cargo-minor-and-patch group (#562)
- Bump docker/login-action 3 → 4 (#561) and docker/build-push-action 5 → 7 (#560)
- Bump Jimver/cuda-toolkit (#559)

### Docs / repo cleanup
- Refresh ARCHITECTURE.md for post-Phase-6 outcomes (#539)
- Refresh BENCHMARKS.md with post-#536 numbers + add vLLM/SGLang comparison (#537)
- A6000 llama.cpp re-bench at 512 → 128 (#538)
- README + QUICKSTART refresh (`/ui`, banner GPU/VRAM, logging defaults) (#546, #547)
- Archive 71 phase-cXX docs subdirs into `docs/archive/phase-c/` (#570)
- Archive frozen profiling/bench MD reports into `docs/archive/` (#568)
- Purge profiling artifact dirs from working tree (#569)

### CI fixes carried over from the unpublished v0.2.1
- Bump Jimver/cuda-toolkit from v0.2.19 to v0.2.35 to handle NVIDIA's renamed installer URLs (#469)
- Install MSVC dev env on Windows before CUDA build; fixes `M_LOG2E` undefined in `flash_api_c.cu` under MSVC (#472)
- Force static MSVC CRT on Windows CUDA build; fixes CRT mismatch between `esaxx-rs` and `kiln-marlin-gemm` (#477)

## kiln-v0.2.1 — 2026-04-24

(unpublished — superseded by kiln-v0.2.2)


Server re-cut to include CI fixes that were missing from kiln-v0.2.0. No
user-facing behavior changes from v0.2.0 in the core server; this cut also
picks up phase-6 Metal and CUDA kernel work landed on main between v0.2.0 and
v0.2.1.

### CI fixes shipped for the full platform matrix
- Bump Jimver/cuda-toolkit from v0.2.19 to v0.2.35 to handle NVIDIA's renamed
  installer URLs (#469)
- Install MSVC dev env on Windows before CUDA build; fixes M_LOG2E undefined
  in flash_api_c.cu under MSVC (#472)
- Force static MSVC CRT on Windows CUDA build; fixes CRT mismatch between
  esaxx-rs and kiln-marlin-gemm (#477)

### Phase 6 CUDA decode work
- Fuse CUDA GDN gated RMSNorm (#466)
- Add CUDA conv1d prefill fast path and fix conv1d prefill launch bounds
  (#481)
- Document post-466 and post-468 MTP decode profiles and post-476 MTP profile
  failure (#468, #480)
- Audit GDN conv decode hotspot and refresh post-#481 current-main profile
  (#473, #483)

### Phase 6 Metal decode work
- Fuse Metal LM-head argmax for greedy decode and reduce Metal LM-head argmax
  on GPU (#471)
- Speed up Metal decode GEMV and route GDN out-proj through Metal decode GEMV
- Fuse Metal full-attention QKV projections and fuse Metal GDN decode QKV
  conv norm
- Persist transposed weight cache asynchronously

### Infrastructure
- Cap cargo and nvcc parallelism and add OOM postmortem helper for RunPod
  builds (#474)

## kiln-v0.2.0 — 2026-04-24

Coordinated release aligned with desktop-v0.2.0. Headline work is the Metal
decode path for Apple Silicon: a new fused GDN kernel family, MTP speculative
decoding improvements, and a batch of macOS startup and prefill reductions.

### Metal GDN kernel fusion
- Fuse Metal GDN decode input projections
- Fuse Metal GDN chunk prep
- Fuse Metal RoPE for prefill (#418)
- Fuse Metal GQA qk norm expansion (#393)
- Default Metal MLP gate-up fusion (#447)
- Add Metal full-chunk GDN prefill (#394), head-last layout (#395)
- Speed up Metal GDN recurrent prefill (#398) and avoid zeroing recurrent outputs (#419)
- Use direct GDN chunk slices on Metal (#391); read full chunks from strided views (#455)
- Use unexpanded GDN QK for Metal decode (#456)
- Use head-major KV read for Metal decode (#452) and head-major SDPA for paged decode (#342)
- Use uninitialized Metal outputs for full-write kernels (#449)
- Parallelize Metal conv1d prefill

### MTP (multi-token prediction) decode
- Route MTP prefill through Metal streaming (#400)
- Speed up macOS default MTP decode
- Mirror desktop speculative routing in bench (#404)
- Align skip-layer bench draft state (#410)
- Defer MTP upload and trim draft state (#408)
- Avoid native MTP during Metal prewarm (#402)
- Guard non-streaming MTP final window (#454)
- Raise Metal skip-layer crossover to 4096 (#442)
- Route long macOS decode through paged skip-layer

### macOS startup and prefill
- Reduce macOS startup and KV prefill overhead
- Speed up macOS startup and skip-layer prefill
- Improve macOS startup and short-prompt routing (#440) and speculative routing (#437)
- Defer Metal precompile until background prewarm (#453); precompile Metal kernels during startup
- Move tokenizer warmup after listen (#460)
- Prewarm macOS speculative path (#386) and make prewarm opportunistic
- Tune macOS Metal hot paths (#385) and streaming prefill defaults (#377)
- Enable tiled prefill by default on Metal (#367)
- Route Metal prefix attention through head-major SDPA (#366)
- Optimize Metal paged KV prefill reads (#416)
- Speed up Metal LM head decode
- Gate Metal readiness prewarm (#332)
- Harden Metal prewarm and KV auto-sizing
- Drop redundant Metal embedding upload (#335)
- Batch Metal auxiliary weight uploads (#443); stream transposed weight cache reads (#445); mmap transposed weight cache hits (#461)

### Server runtime
- Keep default GPU sampling on device (#336); speed up default sampling and fix speculative KV advance (#328)
- Avoid zero-filling server KV pools (#337)
- Hoist paged decode debug gates (#333)

### Profiling and phase 6 kernel work
- Extensive phase 6 decode profiling work (C35–C50) and MTP α-stability re-benches documented in PROFILING.md and PROFILING-MTP-*.md

## kiln-v0.1.2 — 2026-04-20
- See the GitHub release for details.

## kiln-v0.1.1 — 2026-04-20
- See the GitHub release for details.

## kiln-v0.1.0 — 2026-04-18
- Initial public release.
