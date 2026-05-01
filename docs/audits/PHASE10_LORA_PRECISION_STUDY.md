# Phase 10 — LoRA Precision Study (Design)

**Status:** Closed null-result (PR #681, 2026-05-01). See §10 for empirical evidence and `no_kernel_pivot` verdict.
**Source motivation:** PR #649 (`PHASE10_S3_CANDIDATE_PREFLIGHT.md`) §6 footnote, PR #651 (`PHASE10_CLOSURE.md`) pivot recommendation #3.
**Scope:** A/B experiment design for relaxing the LoRA training precision contract from "FP32 storage + FP32 SGEMM" to "BF16 storage + FP32-accumulate tensor-core BF16 SGEMM" on rank-8 SFT.
**Decision:** Design sections 1–9 below were executed in PR #681. Parity gates passed; the step-time gate failed at +2.94% (well below the 14% floor). The implementation diff landed on `main` because the BF16 storage path is *safe* (parity is tight) — it just is not worth re-running the experiment without new evidence (see §10.4).

---

## 1. Background

### 1.1 The hotspot we are targeting

The PR #649 A100-SXM4-80GB SFT profile (Qwen3.5-4B, rank 8, 7 target modules — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — sequence length 8192, batch 1, gradient-checkpointed backward) measured the following on the FP32 SGEMM hot path:

| Kernel | % step time | Invocations | Avg per call | What it is |
| --- | --- | --- | --- | --- |
| `ampere_sgemm_64x64_nn` | **31.07%** | 720 | 9.1 ms | Rank-8 LoRA-down `(T=8192, H=2560) @ (H, R=8) → (T, R)` and LoRA-up `(T, R) @ (R, H) → (T, H)` |
| `ampere_sgemm_128x64_nn` | **11.02%** | 704 | — | Tail of the same LoRA matmul population |
| **Combined FP32 SGEMM** | **~42.1%** | — | — | The single largest non-flash-attention region in the SFT step |

For reference, BF16 tensor-core kernels in the same profile (the base BF16 GEMM paths that run *outside* LoRA) sit at roughly **4–5% combined** on `ampere_bf16_s16816gemm_*`. So the headroom from collapsing the FP32 SGEMM region into a tensor-core BF16 path is large in principle.

### 1.2 Why we had not already done this (historical context)

PR #649 §6 explicitly carved this out of S3 scope:

> Replacing it would require either (a) running LoRA in BF16 with FP32 accumulate (a numerical-stability change to the LoRA training contract, not a kernel fusion), or (b) fusing LoRA into the base GEMM (a major scheduler/memory-layout change in `kiln-train`'s autograd path). Neither matches the §3 scope. Recording it here for future planning.

PR #651 then named this study as the highest-leverage non-kernel lever to pick up post-v0.1, with a target of "~30–40% step-time reduction on Ampere" if loss parity holds.

**Update post-PR #681 (2026-05-01):** The study has been executed. Parity held within all thresholds, but the predicted step-time win did not materialize on A6000 (+2.94% vs the ≥14% floor). This study is no longer the highest-leverage non-kernel lever from PR #651 — it has been ruled out empirically and is closed null per §10. Future planning that targets the same FP32 SGEMM bucket needs new evidence first (see §10.4).

### 1.3 Where the FP32 promotion actually lives

The relevant code paths in current source (re-confirmed against this branch, not memory):

- **Var storage:** `crates/kiln-train/src/trainer.rs` lines ~137–143 — `TrainableLoraParams::initialize` allocates A as `Var::rand_f64(-bound, bound, (rank, in_features), DType::F32, device)` and B as `Var::zeros((out_features, rank), DType::F32, device)`. Both Vars are stored as F32. There is no separate `lora.rs`; all LoRA training-side code is inlined in `trainer.rs`.
- **Matmul dispatch:** `crates/kiln-model/src/lora_loader.rs::compute_lora_delta` (lines ~209–229) is the single dispatch site for the LoRA delta on every layer:
  ```rust
  let x_f32 = x.to_dtype(DType::F32)?;
  let a_f32 = proj.a.to_dtype(DType::F32)?;
  let b_f32 = proj.b.to_dtype(DType::F32)?;
  let hidden = x_f32.broadcast_matmul(&a_f32.t()?)?; // [..., rank]
  let delta  = hidden.broadcast_matmul(&b_f32.t()?)?; // [..., out_features]
  let delta  = (delta * scale as f64)?;
  let delta  = delta.to_dtype(x.dtype())?;
  ```
  This is the explicit upcast that drives every LoRA-down and LoRA-up matmul into the FP32 SGEMM kernels seen in the profile. The hidden activations `x` arrive as BF16 in the standard SFT path and are upcast here, not at the call site.
- **Optimizer:** `sft_train` in `trainer.rs` calls `sgd_step_from_map` / `sgd_step` (plain SGD, not Adam). There is **no Adam first/second moment buffer** to convert; the optimizer-state precision discussion in standard mixed-precision training (FP32 master weights for momentum/variance) does not apply here. Var precision is the whole story.

---

## 2. Hypothesis

> Storing the rank-8 LoRA A and B Vars in **BF16** and running the LoRA-down and LoRA-up matmuls on **BF16 tensor cores with FP32 accumulate** (`ampere_bf16_s16816gemm_*`) preserves rank-8 LoRA SFT loss-curve parity to the same tolerance PR #649 already accepts for v0.1 training (final loss bit-identical to within `1e-7`, training loss within ±0.5% per step), while collapsing the ~42% FP32 SGEMM region into the existing BF16 tensor-core path.

If the hypothesis holds:

- Step-time reduction is bounded above by the math-ceiling rule. With the FP32 SGEMM region at 42.1% and a target speedup of `s` on that region:
  - At `s = 1.5×` (conservative — reflects roughly the BF16/FP32 ratio without graph-replay hiding savings on a backward-only SFT path), step-time reduction is `0.421 × (1 − 1/1.5) ≈ 14.0%`.
  - At `s = 4×` (upper bound — reflects the raw `ampere_sgemm` vs `ampere_bf16_s16816gemm_*` per-FLOP ratio), step-time reduction is `0.421 × (1 − 1/4) ≈ 31.6%`.
  - The PR #651 target of "~30–40% step-time reduction" implicitly assumes the upper-bound regime. We treat 14% as the floor we must clear to ship and 30% as the celebratory result.
- Both bounds clear the 1.05× math-ceiling floor for SFT backward (no CUDA graphs) by an order of magnitude. The math-ceiling rule is **not** the gating constraint here; numerical parity is.

If the hypothesis fails — i.e. loss-curve drift exceeds the parity tolerance — the experiment terminates and we record a negative result. There is no fallback to "BF16 with FP32 accumulate but FP32 storage" because that is a different experiment with different storage savings (see §6).

---

## 3. Current state

### 3.1 What is already in tree

- LoRA Var storage: F32 (`trainer.rs` ~137–143).
- LoRA delta dispatch: explicit F32 upcast in `compute_lora_delta` (`lora_loader.rs` ~209–229), single site, used by every LoRA-aware projection (`linear_with_lora_t`, `linear_with_lora`, MLP variants in `forward.rs`).
- 7 default target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (`trainer.rs::DEFAULT_TARGET_MODULES`).
- Backward path: candle autograd, with `checkpointed_forward_backward` and `standard_forward_backward` paths gated by `CheckpointConfig`.
- Optimizer: SGD (`sgd_step_from_map` / `sgd_step`).
- LM-head loss: cross-entropy uses `.to_dtype(DType::F32)` for `log_sum_exp` (trainer.rs ~898, 1038–1039) and FLCE Phase B fused dispatch when armed. **The CE/log-sum-exp upcast is intentional for numerical stability and stays F32 in this experiment.** This study is scoped to LoRA matmuls only; nothing in the loss path moves.
- PEFT export: `save_peft` writes `lora_A.weight` / `lora_B.weight` in whatever the Var dtype is, per safetensors round-trip.

### 3.2 What PR #649 accepted as parity

The PR #649 reference SFT run (60 steps, deterministic seed, A100-SXM4-80GB) produced final loss `0.8115726113319397` bit-identical across two consecutive runs and step times `30.03 / 30.98 / 30.98 s`. Peak VRAM `70,053–70,081 MiB`. This is the parity bar this experiment must clear. We will reproduce the exact harness on the candidate branch.

### 3.3 What is *not* in scope

- **Inference-side LoRA precision.** This study only changes training-side Vars and the training-path dispatch. Inference loads through `lora_loader.rs::load_lora_weights` and accepts F16/BF16/F32 already; that path is unchanged.
- **GRPO.** Phase 10 GRPO inner loop calls into the same `compute_lora_delta`, so an in-principle benefit applies, but GRPO has its own sampling/reward variance regime and we do not want to confound the parity signal. GRPO precision parity is a separate study.
- **FLCE / cross-entropy precision.** The `to_dtype(DType::F32)` upcasts in the loss path stay. Touching them is a different experiment with different risk.
- **FP16 LoRA.** BF16 only. FP16 has the wrong dynamic range for our scale factor multiplied through long sequences and is not a candidate.

---

## 4. Experiment design

### 4.1 Variables

- **Independent:** LoRA Var dtype and `compute_lora_delta` dispatch dtype.
  - **Control (A):** F32 storage, F32 dispatch (current main).
  - **Treatment (B):** BF16 storage, BF16 dispatch with FP32 accumulate (Ampere `s16816gemm` default), result cast to input dtype before adding to base output (matches the existing post-cast on line ~227).
- **Dependent (parity gates, all must hold):**
  - Final training loss within `|Δ| ≤ 0.5%` of control across 60 deterministic steps.
  - Training loss curve maximum point-wise deviation `|Δ| ≤ 1.5%` per step.
  - No NaN / Inf in any LoRA Var across all 60 steps.
  - PEFT-exported safetensors load successfully on the inference-side `load_lora_weights` path and round-trip a single forward without dtype panic.
- **Dependent (performance gates, must hold *if* parity gates hold):**
  - Step-time reduction ≥ 14.0% on the same A100-SXM4-80GB harness, single-node, batch 1, seq 8192, rank 8, 7 modules. This is the math-ceiling floor; below it the change is not worth the precision risk.
  - Combined `ampere_sgemm_*` kernel time on the LoRA dispatch reduces to ≤ 5% of step time (i.e. the F32 SGEMM region collapses; if it doesn't, treatment is mis-specified).
  - VRAM peak does not regress by more than +1% (the Var storage *should* drop by ~2× on the LoRA params themselves but those are tiny; activation memory dominates and is unchanged).
- **Held constant:** seed, base model weights, dataset slice, sequence length, batch size, rank, alpha, scale, optimizer (SGD), learning rate, gradient checkpointing config, target modules, GPU model.

### 4.2 Procedure

1. Reproduce the PR #649 harness on the **control** branch as-is. Record: final loss, per-step loss vector, step-time vector, VRAM peak, kernel-time breakdown via Nsight Systems on a 5-step capture window (avoid steady-state bias from a short capture, but cap the capture at 5 steps to keep the trace under 1 GB).
2. Apply the **treatment** patch (see §5 implementation sketch) on a sibling branch off the same SHA.
3. Run the same harness on treatment. Record the same vectors.
4. Compute parity deltas. Either gate fails → record negative result, stop.
5. If parity gates pass, run the performance comparison: median-of-3 step-time on a fresh pod (control vs treatment, alternating runs, 60 steps each, drop the first run to warm sccache and CUDA graph caches in the inference parts of forward).
6. Confirm the kernel-time breakdown on treatment: `ampere_sgemm_*` on LoRA dispatch should be ≤ 5%; `ampere_bf16_s16816gemm_*` should grow to absorb the work.
7. PEFT-export round-trip: load the treatment-trained adapter back through `load_lora_weights`, run one forward step, no dtype panic, output dtype matches input.

### 4.3 Sample size and seeds

- 60 steps per run, single seed, 3 runs per arm. The PR #649 harness already demonstrated bit-identical determinism across two consecutive runs of the control; treatment is expected to be deterministic as well (BF16 SGEMM with FP32 accumulate on Ampere is bitwise reproducible at fixed shape and seed). We are not chasing a stochastic effect — we are chasing a precision-induced bias, which is deterministic.
- 60 steps is the minimum that captures the early-training curvature where rank-8 LoRA is most sensitive to precision. If treatment passes 60-step parity, a follow-up 1000-step run is appropriate before any production cut-over but is **out of scope for this design**.

### 4.4 Decision tree

```
parity gates pass?
├── no  → stop, record negative result, archive trace, do not implement
└── yes → performance gates pass?
         ├── no  → stop, record "parity OK but no win", flag for re-profile
         └── yes → write Phase 11 implementation proposal with this doc as evidence
```

---

## 5. Implementation sketch

### 5.1 Code changes (treatment branch)

**`crates/kiln-train/src/trainer.rs` (~137–143)** — change Var dtype:

```rust
// A: [rank, in_features] — Kaiming uniform, BF16 storage
let a = Var::rand_f64(-bound, bound, (rank, in_features), DType::BF16, device)
    .with_context(|| format!("init LoRA A for layer {layer_idx} {module}"))?;

// B: [out_features, rank] — zeros, BF16 storage
let b = Var::zeros((out_features, rank), DType::BF16, device)
    .with_context(|| format!("init LoRA B for layer {layer_idx} {module}"))?;
```

Note: `rand_f64` generates F64 RNG samples then casts; the cast to BF16 is acceptable for Kaiming-uniform initialization (the bound is small enough that BF16 representable error is dwarfed by the noise itself).

**`crates/kiln-model/src/lora_loader.rs::compute_lora_delta` (~209–229)** — drop the F32 upcast and let the matmul run in the input dtype (BF16 in the SFT path):

```rust
pub fn compute_lora_delta(
    x: &Tensor,
    proj: &LoraProjectionWeights,
    scale: f32,
) -> Result<Tensor> {
    // A: [rank, in_features] -> A^T: [in_features, rank]
    // B: [out_features, rank] -> B^T: [rank, out_features]
    // delta = x @ A^T @ B^T * scale
    // Storage and dispatch dtype now match x (BF16 in SFT, F32 in MTP fp32-head path).
    let hidden = x.broadcast_matmul(&proj.a.to_dtype(x.dtype())?.t()?)?; // [..., rank]
    let delta  = hidden.broadcast_matmul(&proj.b.to_dtype(x.dtype())?.t()?)?; // [..., out_features]
    let delta  = (delta * scale as f64)?;
    let delta  = delta.to_dtype(x.dtype())?;
    Ok(delta)
}
```

The `to_dtype(x.dtype())` calls are no-ops when storage already matches input (the BF16 SFT path). They preserve the existing MTP fp32-head behavior (`linear_with_lora_t` arms `mtp_debug::is_mtp_fp32_head_armed()` and runs base in F32; LoRA delta then naturally promotes to F32 to match the F32 hidden state). This is why we cast *to input dtype* rather than hard-coding BF16.

**Inference loader (`lora_loader.rs::load_lora_weights`)** — no change. The loader already accepts F32, F16, BF16 safetensors and stores whatever it finds. PEFT-exported BF16 adapters will load identically to the F32 adapters that exist today. Inference forward dispatch through `compute_lora_delta` will run BF16-in / BF16-out, which is what the surrounding BF16 base path wants anyway.

### 5.2 What we deliberately do not change

- `linear_with_lora_t` MTP fp32-head branch (lines ~272–276 in `lora_loader.rs`). The base matmul stays F32 there because the MTP inner block runs F32; the LoRA delta in that path will now run F32 too because `x` is F32. This is the correct behavior and is preserved by the `to_dtype(x.dtype())` pattern above.
- The CE / log-sum-exp F32 upcasts in `trainer.rs` (lines ~898, 1038–1039). Loss precision is independent of LoRA storage precision.
- FLCE Phase B dispatch. FLCE handles the LM-head matmul fused with cross-entropy; it does not touch LoRA delta dispatch.
- `save_peft` PEFT export. The function writes whatever dtype the Var has. PEFT consumers (HuggingFace `peft` library, vLLM) accept BF16 LoRA adapters as a first-class format.
- SGD step. SGD has no precision-sensitive accumulator state.
- Gradient checkpointing config. Backward through BF16 storage is a candle-autograd built-in; no rewiring required.

### 5.3 Risk-of-silent-bug surfaces

- **`Var::rand_f64` into BF16:** verify `candle_core::Var::rand_f64` accepts a non-F32 dtype on this candle version. If not, allocate F32 then `to_dtype(DType::BF16)?` and wrap the result in a Var via the appropriate constructor. (Audit step 1 of the implementation task; trivial to confirm.)
- **`Var::zeros` into BF16:** same check.
- **PEFT round-trip dtype:** confirm `save_peft` does not have an implicit `to_dtype(DType::F32)` somewhere in the safetensors writer. If it does, the round-trip is BF16 → F32 → BF16 which is fine (BF16 is a strict subset of F32 representable values), but we want the export to be BF16 directly to match the storage and to match how PEFT exporters in the broader ecosystem produce BF16 LoRAs.
- **MTP fp32-head interaction:** the proposed `to_dtype(x.dtype())` keeps that path bit-identical to today, but we should add a unit test that arms `is_mtp_fp32_head_armed()` and confirms the LoRA delta matmul runs F32. This is a regression guard; if a future refactor breaks the dtype-following pattern, the test catches it.

---

## 6. Risk

### 6.1 Numerical risk

- **BF16 dynamic range vs. rank-8 + scale.** Rank-8 LoRA with the standard `alpha = 16` produces `scale = alpha / rank = 2.0`. Hidden activations passing through `x @ A^T` are bounded by the BF16 storage of A (small, Kaiming-uniform-bounded values), and `hidden @ B^T` is bounded by B (initialized to zero, grows during training). The chain `x @ A^T @ B^T * 2.0` has the same dynamic-range exposure as any BF16 GEMM in the base model — **plus** the fact that B grows from zero, so for the first hundred steps the delta is dominated by activation noise and BF16 rounding is below the gradient signal. After B saturates, the delta magnitude is comparable to base-output scale, where BF16 is the established norm in the base model. We expect parity to hold; we are not in unexplored numerical territory.
- **Long-sequence accumulation.** The matmuls are `(8192, 2560) @ (2560, 8) → (8192, 8)` (LoRA-down) and `(8192, 8) @ (8, 2560) → (8192, 2560)` (LoRA-up). Accumulation depth is 2560 on the down-projection and 8 on the up-projection. FP32 accumulate handles 2560-deep BF16 accumulation cleanly on Ampere; this is the same depth the base BF16 GEMMs already run. No new precision cliff.
- **Backward-pass precision.** candle autograd will run the matmul backward in the storage dtype. BF16 backward of a BF16 matmul on Ampere uses the same `s16816gemm` family with FP32 accumulate. This is the path candle uses for the base model's BF16 backward today; it is well-trodden.
- **Gradient magnitudes.** BF16 has 7 mantissa bits; the smallest gradient magnitude representable above 0 at the typical scale of LoRA gradients (`~1e-4` to `~1e-3`) is well above BF16 underflow (`~1e-38` for normals). Underflow is not a risk.

### 6.2 Engineering risk

- **Unintended call-site dtype change.** `compute_lora_delta` is called from at least three sites in `forward.rs` (attention QKV, MLP gate/up/down, MTP head) and from `linear_with_lora_t` and `linear_with_lora` in `lora_loader.rs`. The proposed `to_dtype(x.dtype())` pattern means every call site automatically picks up the storage dtype of A/B without per-site changes. **This is the design point that lets the change be a one-file edit (plus the trainer init line).** If any call site ends up wanting a *different* storage dtype than its `x`, the design breaks; we should grep for any place that constructs `LoraProjectionWeights` with a deliberately-mismatched dtype today (none exist, per the loader code path).
- **PEFT export consumer compatibility.** External tools that ingest our PEFT adapters need to accept BF16. HuggingFace `peft` and vLLM both do. Any internal tool that hard-codes F32 ingest needs to be updated; we should grep our own repo.
- **Adapter file size.** BF16 storage halves adapter on-disk size. This is a benefit, not a risk, but consumers that pre-allocate F32 buffers based on file size will need a sanity check.

### 6.3 Process risk

- **Conflating step-time wins with parity wins.** The math-ceiling ceiling looks generous but parity is the hard gate. A "negative parity, positive step-time" result is a fail, not a partial win. The decision tree in §4.4 enforces this.
- **A100 vs H100 generalization.** The profile is on A100. H100 has different relative throughput between FP32 SGEMM and BF16 tensor cores (BF16 advantage is even larger). If parity holds on A100, it will hold on H100 — BF16 numerics are identical, only kernel speed differs. We do not need an H100 confirmation run for the parity result; we will need one before any production rollout to confirm step-time on the deployment GPU.
- **Bisecting a parity failure.** If treatment fails parity, the bisection space is small: storage dtype, dispatch dtype, post-cast site. A failure mode log on the treatment run (per-layer, per-step max-abs delta in the LoRA delta tensor relative to the control run) gives us the debug signal in one shot. The implementation task should produce this log.

### 6.4 What this study cannot tell us

- Whether BF16 LoRA generalizes to higher ranks (16, 32, 64) at the same parity tolerance. Rank-8 is the configuration kiln-train ships and the configuration in the profile; higher-rank parity is an extrapolation question.
- Whether BF16 LoRA generalizes to other base models. We are fitting one base model architecture (Qwen3.5-4B). Base-model precision sensitivity varies; this study does not certify the general case.
- Whether the result holds under GRPO sampling-vs-update precision divergence. GRPO is out of scope per §3.3.

---

## 7. GPU budget

This is a doc-only audit. **No GPU pod is required for this PR.**

The downstream implementation task (if Phase 11 sign-off is granted) requires:

| Run | GPU | Wall-clock (median estimate) | Cost (median estimate) |
| --- | --- | --- | --- |
| Control reproduction (3 runs × 60 steps) | A100-SXM4-80GB or A6000 | ~3 × 1.5 min ≈ 4.5 min | ~$0.05 |
| Treatment runs (3 runs × 60 steps) | same | ~3 × 1.0 min ≈ 3.0 min (assumes ~30% step-time win) | ~$0.04 |
| Nsight Systems trace, control, 5-step capture | same | ~10 min including instrumentation overhead | ~$0.10 |
| Nsight Systems trace, treatment, 5-step capture | same | ~10 min | ~$0.10 |
| PEFT round-trip integration test | same or smaller | ~5 min | ~$0.05 |
| Pod overhead, build, sccache cold-start cost | same | ~15 min | ~$0.15 |
| **Total budget for the implementation task** | A100 or A6000 | **~50 min** | **~$0.50** |

This budget assumes the canonical `kiln-pod-acquire` warm-pool path. Cold-pool fallback adds ~15 min and ~$0.20 for the first build.

**Hard caps** (must be in the implementation task description per the kiln SSH-polling-deadlock standard):

> Wall-clock budget (hard cap): If this task exceeds 90 min of wall-clock or $5 of cumulative cost, STOP, post a WIP PR, terminate the pod, and return.

The wide gap between the median estimate and the cap reflects the cost of an SSH wedge or a reproduction-harness drift; the cap is what blocks runaway, not what we expect.

---

## 8. Out-of-band: what would change after a positive result

A positive result would reduce step time by a midpoint estimate of **~22%** (split the difference between the 14% conservative floor and the 30% upper-bound target) on rank-8 SFT, with no scheduler / memory-layout / kernel-vendoring work. The implementation surface is two files and on the order of 10 lines of code change. The downstream Phase 11 task would:

1. Land the two-file change with a feature flag (`KILN_LORA_BF16_STORAGE`, default off) for one release cycle.
2. Run the 1000-step parity check before flipping the default.
3. Extend `save_peft` documentation to note the dtype.
4. Update the kiln-train README to call out BF16 LoRA as the production training contract.

These are notes for the *next* design doc, not deliverables of this one.

---

## 9. References

- `docs/audits/PHASE10_S3_CANDIDATE_PREFLIGHT.md` — PR #649. The source profile and the §6 footnote that named this study.
- `docs/audits/PHASE10_CLOSURE.md` — PR #651. Pivot recommendation #3 and the doc landing path for this file.
- `crates/kiln-train/src/trainer.rs` — `TrainableLoraParams::initialize` (LoRA Var storage); `sft_train` (SFT loop, optimizer, gradient checkpointing dispatch).
- `crates/kiln-model/src/lora_loader.rs` — `compute_lora_delta` (the FP32 upcast site); `linear_with_lora_t` (MTP fp32-head interaction).
- `crates/kiln-model/src/forward.rs` — LoRA delta call sites in attention QKV and MLP paths.

---

## 10. Closure — empirical result (PR #681)

**Date:** 2026-05-01.
**PR:** [#681 — `phase10: LoRA Var BF16 storage + FP32-accumulate (null-result A/B)`](https://github.com/ericflo/kiln/pull/681), squash-merged at `52f8bb0` on 2026-05-01T16:16:37Z.
**Verdict:** Null-result. Parity gates pass; step-time gate fails at +2.94% (below the 14% audit floor). Implementation diff (the two-file change from §5.1) is in `main` because the BF16 storage path is parity-safe; the result of the experiment is that the predicted ~14–32% step-time reduction did not materialize on A6000 at rank-8.

### 10.1 What ran

- **Hardware:** RunPod RTX A6000 (49 GB VRAM), CUDA 12.4, sccache+B2 build cache, `KILN_CUDA_ARCHS=86`.
- **Driver:** `kiln-bench --model-path /workspace/qwen3.5-4b --training-steps 100 --seed 42 --paged` with `KILN_W4A16=0` (W4A16 stubs out `q_proj_t` and is incompatible with SFT). Rank=8, alpha=16, lr=1e-4, SGD, 1 epoch over 100 synthetic examples.
- **Sample size:** 100 SFT steps per arm (vs 60 in §4.3 — extended for tighter loss-curve resolution), single seed (42), one run per arm. Bit-identical determinism on the harness was already established by PR #649; treatment is also deterministic at fixed shape and seed.
- **Arms:**
  - Control: `main` @ `4a3babd`, binary md5 `91d0c53a419cc76a0e42eb398121a40d`.
  - Treatment: `phase10/lora-precision-study-impl` @ `ec66da0`, binary md5 `262b0a9f37f85152ecc6e636835075fb`. Treatment patch is exactly the §5.1 sketch: `Var::rand_f64`/`Var::zeros` `DType::F32` → `DType::BF16` for LoRA A/B init; `compute_lora_delta` drops the `to_dtype(DType::F32)` upcast on `x`, A, B, and casts A/B to `x.dtype()` instead.

### 10.2 Gate results

| Gate | Threshold | Observed | Pass |
| --- | --- | --- | --- |
| Final loss \|Δ\| | ≤ 0.5% | **0.052%** (1.663116 → 1.663987) | ✅ |
| Per-step max \|Δ\| | ≤ 1.5% | **0.622%** (worst single step, step 5) | ✅ |
| Per-step mean \|Δ\| | — | 0.141% | ℹ️ |
| NaN / Inf in losses | none | 0/100 both arms | ✅ |
| PEFT roundtrip | loadable, dtypes valid | 256 BF16 tensors, 0 NaN/Inf, valid HF `adapter_config.json` (r=8, α=16, 7 target modules, CAUSAL_LM) | ✅ |
| Step-time reduction | ≥ 14.0% | **+2.94%** (0.34 s/step → 0.33 s/step, 100-step median) | ❌ |
| Peak training VRAM | ≤ +1% | **−0.64%** (24906 → 24746 MB) | ✅ |

All five parity gates pass tightly. The performance gate fails by a wide margin — the observed +2.94% is at or below the harness's `s/step` rounding precision (0.01 s ≈ ±3% of step time at this dataset shape), so even the sign of the win is not robustly resolvable. The 14% audit floor was chosen specifically so that wins below it are not worth the precision risk; +2.94% does not approach that floor.

### 10.3 Why the predicted speedup did not materialize

Two leading hypotheses, neither falsified by the §10.1 run alone:

1. **The ~42% FP32 SGEMM bucket from PR #649 may not have been LoRA-dominated.** The audit attributed `ampere_sgemm_64x64_nn` (31.07%) + `ampere_sgemm_128x64_nn` (11.02%) to rank-8 LoRA-down/-up matmuls based on shape match. That attribution was indirect — the bucket may also include LM-head, embedding, and elementwise reductions in backward that this change does not affect. A dedicated nsys re-profile of the SFT step on the **treatment** branch would tell us which kernels actually moved (if any), which is the empirical test for this hypothesis.
2. **cuBLAS heuristic may not have flipped to `ampere_bf16_s16816gemm_*` for the candle `broadcast_matmul` shapes used.** Rank-8 with hidden=2560 / intermediate=9216 produces matmuls of shapes `(8192, 2560) @ (2560, 8) → (8192, 8)` and `(8192, 8) @ (8, 2560) → (8192, 2560)`. Candle dispatches matmul through cuBLAS with default heuristics; without `cublasLtMatmulPreferenceSetAttribute` tuning, BF16-input + FP32-accum can still land on a non-tensor-core SGEMM path on Ampere — particularly for the `K=8` accumulation depth on the up-projection, where tensor-core kernels are not always selected by default heuristics. The audit hypothesis (§2) implicitly assumed the s16816gemm path would be selected automatically; that assumption is not load-bearing on the actual cuBLAS dispatch.

The §10.1 evidence is consistent with either hypothesis (or both). Distinguishing them requires the §10.4 work.

### 10.4 Verdict: `no_kernel_pivot`

This study is **closed null**. The implementation diff is on `main` and may stand as long as parity holds (it does), but the design hypothesis — that flipping LoRA Var storage to BF16 buys a step-time win on Ampere — is **not supported** by the empirical evidence at rank-8 on A6000.

**Do not re-queue this study** (or any near-equivalent variant — e.g. "what if we also fuse the post-cast", "what if we use FP16 instead of BF16", "what if we change `alpha`") without **both** of the following:

1. **Re-profile evidence** that the FP32 SGEMM hot bucket is **actually LoRA-dominated**, not LM-head/embedding/reduction-dominated. The right artifact is an nsys trace of the **treatment** branch SFT step that shows which kernels still populate the FP32 SGEMM region. If the bucket is unchanged on the treatment branch, the hypothesis was mis-attributed and the lever does not exist.
2. **Dispatch evidence** that candle's `broadcast_matmul` dispatches the target shapes (rank=8, hidden=2560, intermediate=9216) to the `ampere_bf16_s16816gemm_*` family on the treatment branch. The right artifact is a `cublasLt` log capture (or equivalent profile) showing the actual kernel selected for the LoRA-down and LoRA-up matmuls.

If both pieces of evidence land and still point at a real LoRA-side hotspot, the next move is **not** to re-flip storage — it is to force `cublasLtMatmul` with explicit `s16816gemm` selection (or to fuse LoRA into the base GEMM, a much larger scheduler/autograd rewrite in `kiln-train`).

### 10.5 What this closure does not say

- It does **not** say BF16 LoRA storage is unsafe. The parity evidence is strong: 0.052% final-loss drift at 100 steps, 0.622% worst-step drift, no NaN/Inf, clean PEFT round-trip. The BF16 storage path is *safe*; it is just not *worth* re-running this experiment without new evidence.
- It does **not** say the FP32 SGEMM bucket from PR #649 is wrong. It says the **attribution** from FP32 SGEMM to LoRA matmuls in the audit hypothesis may be mis-specified, and that the right next probe is a treatment-branch re-profile to verify which kernels actually populate the bucket.
- It does **not** generalize to higher LoRA ranks (16, 32, 64), other base models, or GRPO. Those were already out of scope per §6.4 and remain out of scope post-closure.

### 10.6 Cross-references

- PR #680 — design doc landing PR (this file's original commit).
- PR #681 — implementation + A/B run + null-result write-up (canonical empirical evidence).
- PR #649 — `PHASE10_S3_CANDIDATE_PREFLIGHT.md`. Original FP32 SGEMM ~42% attribution that motivated this study.
- PR #651 — `PHASE10_CLOSURE.md`. Pivot recommendation #3 that named this study as the highest-leverage post-v0.1 non-kernel lever. Updated post-#681 to reflect closure.
