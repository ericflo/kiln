# pi-faithful-completion: SFT chain experiments (round-3 follow-up)

Goal: lift no-prompt composite from base 0.656 to ≥0.80 by SFT on filtered
strict-prompt rollouts. Reference ceiling (base + strict prompt at inference):
0.819 ± 0.014 (12σ paired, see capability.jsonl iter 11).

## Method

Training data: 211 filtered strict-prompt rollouts (composite > 0.7 from
292 base + strict-prompt rollouts on `datasets/train.tasks.jsonl`).
SFT input = default system prompt + user prompt;
SFT output = strict-prompted completion (model learns to behave strictly
even with non-strict input).

Trainer: `cuda_sft_file --trainer generic` (the `native` path is currently
~50× slower due to per-step arena allocation and bespoke CUDA without
batched kernels — see "Native trainer slowness" below).

## Results

3-seed paired eval (eval.tasks.jsonl, no-prompt input):

| iter | recipe | no-prompt composite | lift vs base | σ |
|---|---|---:|---:|---:|
| base | — | 0.6563 ± 0.017 | — | — |
| iter5 | r4, lr 1e-4, 1ep, 64 ex | 0.6320 ± 0.026 | -0.012 | 0.5 |
| iter6 | r16, α32, lr 5e-5, 2ep, 211 ex | 0.5366 ± 0.001 | -0.114 | 2.9 (regression) |
| iter7 | r4, α8, lr 1e-5, 1ep, 211 ex | 0.7278 ± 0.026 | **+0.072** | **3.6** |
| iter8 | iter7 + (r4, α8, lr 1e-5, 1ep, 211 ex) | **0.7735 ± 0.017** | **+0.1344** | **7.8** |
| iter9 | iter8 + (r4, α8, lr 1e-5, 1ep, 211 ex) | 0.7507 ± 0.010 | +0.096 | 3.6 |
| ceiling | base + strict prompt at inference | 0.8192 ± 0.010 | +0.169 (3-seed) | 12 |

## Lessons

- **Higher rank / higher lr / more epochs catastrophically forgets** (iter6:
  -0.114 at 2.9σ, format_strict 0.98 → 0.55).
- **Light-touch SFT chained over short epochs is the regime that works.**
  iter7 → iter8 chain added +0.063 lift in one epoch. iter9 regressed (over-
  training).
- **Sub-scores at iter8**: outcome.value_correct 0.7719 (vs 0.6491 base),
  honesty.score 0.8246 (vs 0.7175), format_strict.score 0.8713 (vs 0.9825),
  terseness.score 0.9941. The gains are in outcome and honesty; format
  pays a moderate cost.
- iter8 is **the chain-SFT optimum** at 0.7735, captures 81% of the
  prompted-lift ceiling (0.7735 − 0.6391) / (0.8192 − 0.6391) ≈ 0.74.

## Open

- Goal of 0.80 not yet met (gap +0.027).
- Untried angles that may break past:
  - chain on iter8 with lr 5e-6 (gentler still)
  - OPD with same-model+strict-prompt-template teacher (different signal —
    pulls distribution rather than mimicking outputs)
  - OPD with Qwen 14B/27B teacher (adds knowledge the 4B doesn't have)
  - SFT with anchor / general-reasoning data mixed in to preserve format

## Native trainer slowness

`cuda_sft_file --trainer native` runs at ~180 s/step (vs `generic` at
~3.5 s/step — 50× slower). Root cause in `cuda_train.rs:1325` (function
`cuda_native_sft_train`):

1. `CudaTrainArena::new()` allocated **per step** (line 1469) instead of
   hoisted/reused.
2. `cuda_linear_attention_state_zeros_for_model()` called **per step**
   (line 1471) instead of allocated once and zeroed in place.
3. Step kernel `cuda_lora_model_adamw_step_with_gdn_state_with_arena`
   does many small CUDA launches with CPU sync between them; GPU util
   sits near 0% while CPU dispatches.

Compare `vk_native_sft_train` in `vk_train.rs:4706` which has gradient
checkpointing, recompute paths for hybrid GDN, length-sorted iteration,
and pre-computed GDN state shape. Backporting the same structure to CUDA
(plus routing the inner step through `BackendRuntime` like `trainer.rs`
sft_train does) would close the gap.

Workaround: always pass `--trainer generic` until cuda-native catches up.
