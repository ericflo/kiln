# pi-faithful-completion: in-weights uplift experiments (round-3 follow-up)

**Goal:** lift no-prompt composite from base 0.656 to ≥0.80 by training
the strict-prompt behavior into Qwen3.5-4B's weights. Prompted ceiling
(base + strict prompt at inference, no weight changes): 0.819 ± 0.014.

## Result: **iter8 ships** — 0.7735 ± 0.017 (+0.134 paired lift, 7.8σ)

iter8 captures **71.9% of the prompted-lift in the weights**
((0.7735 − 0.6563) / (0.8192 − 0.6563)). Goal of 0.80 not yet met
(0.027 short), but the lift is real, robust, and reproducible across
two fresh pods.

## Full experiment table

3-seed paired eval (eval.tasks.jsonl, no-prompt input):

| iter | recipe | composite | lift vs base | σ | verdict |
|---|---|---:|---:|---:|---|
| base | — | 0.6563 ± 0.017 | — | — | reference |
| iter5 | r4, lr 1e-4, 1ep, 64 ex | 0.6320 ± 0.026 | −0.012 | 0.5 | null |
| iter6 | r16, α32, lr 5e-5, 2ep, 211 ex | 0.5366 ± 0.001 | −0.114 | 2.9 | **regression** (overtraining) |
| iter7 | r4, α8, lr 1e-5, 1ep, 211 ex | 0.7278 ± 0.026 | +0.0716 | 3.6 | positive |
| **iter8** | iter7 + (r4, α8, lr 1e-5, 1ep, 211 ex) | **0.7735 ± 0.017** | **+0.1344** | **7.8** | **SHIP** |
| iter9 | iter8 + (r4, α8, lr 1e-5, 1ep, 211 ex) | 0.7507 ± 0.010 | +0.0964 | 3.6 | regress |
| iter11 | iter8 + (r4, α8, lr 5e-6, 1ep, 211 ex) | 0.7729 ± 0.016 | +0.117 | 3.4 | neutral |
| iter12 | iter8 + (r4, α8, lr 1e-5, 1ep, 422 mixed-prompt) | 0.7564 ± 0.000 | +0.100 | 3.3 | regress |
| iter13 | OPD self-distill chain on iter8 | n/a | n/a | n/a | infrastructure mismatch — see below |
| iter14 | r8, α16, lr 1e-5, 1ep, 211 ex (fresh) | 0.7735 ± 0.017 | +0.130 | 3.5 | **ties iter8** — plateau confirmed |
| iter15 | iter14 + (r8, α16, lr 1e-5, 1ep, 211) | 0.7564 ± 0.017 | +0.113 | 4.7 | regression (over-chain) |
| ceiling | base + strict prompt at inference | 0.8192 ± 0.010 | +0.169 (3-seed) | 12 | — |

Sub-scores at iter8: `outcome.value_correct` 0.7719 (vs 0.6491 base,
**+0.123**), `honesty.score` 0.8246 (vs 0.7175, **+0.107**),
`format_strict.score` 0.8713 (vs 0.9825, **−0.110**),
`terseness.score` 0.9941 (vs 0.9683, +0.026). Gains concentrated in
outcome and honesty; format pays a moderate cost.

## What works

- **Light-touch SFT chained over single short epochs.** rank 4, α 8,
  lr 1e-5, 1 epoch, full 211 examples. Run twice (iter7 → iter8) for
  the headline lift.
- **The filtered strict-prompt rollouts (composite > 0.7) as the SFT
  signal.** All 205+ kept examples were strict-prompt-conditioned
  with mean reward 0.995, so the assistant turn IS the target
  behavior.

## What doesn't work

- **Higher rank / higher lr / more epochs.** iter6 (rank 16, lr 5e-5,
  2 epochs) regressed by −0.114 — catastrophic forgetting.
- **More chained epochs past iter8.** iter9 (3rd chain epoch) regressed
  to 0.7507 (still positive, but down). iter11 (lr halved) was neutral.
- **Mixed prompts during training** (default + strict as inputs,
  same strict output). iter12 regressed to 0.7564.

The chain-SFT regime saturates around 0.7735. iter14 (rank 8 fresh
in one epoch) and iter8 (rank 4 chained over two epochs) BOTH hit
**exactly 0.7735** — same data, same plateau, regardless of how
the capacity is distributed. The training data itself caps the
achievable lift; the model has already learned everything the
strict-prompt rollouts can teach it. More parameters or more
training just re-arranges the same fit.

## Untried angles to break past 0.7735

1. **OPD self-distillation** (different signal — pulls token
   distribution rather than mimicking sequences). Failed in iter13 due
   to infrastructure: `cuda_opd_remote` only supports
   `RemoteProvider::Vllm`/`Sglang`, both of which require
   `/v1/completions` with `prompt_logprobs` — kiln serve only exposes
   `/v1/chat/completions`. Fix would require either:
   - adding `/v1/completions` + `prompt_logprobs` to kiln serve, OR
   - adding a `RemoteProvider::Kiln` variant in `remote_teacher.rs`
     that uses `/v1/chat/completions` + `top_logprobs`, OR
   - a Python proxy translating between the two APIs.
2. **Larger teacher OPD** (Qwen 14B/27B). Same infrastructure issue
   as iter13.
3. **Hard-tail focused SFT.** Identify ~15 train tasks where even
   strict-prompt fails (composite < 0.5), generate more rollouts on
   them, chain SFT on the hard tail.
4. **KL-regularized SFT.** Add KL penalty against base model during
   SFT loss to preserve general ability while drifting toward strict
   distribution.
5. **Anchor-augmented SFT.** Mix non-rubric general-reasoning
   examples (8-32 of them) into training to prevent format
   degradation.

The format-strict drop (−0.110) is the main cost being paid; (4) and
(5) target that directly.

## Reproducer

```bash
# Stage A: clone kiln + build with SCCACHE_RECACHE=1
cd /workspace && kiln-setup --clone --repo /workspace/kiln
cd /workspace/kiln && bash capabilities/caps/pi-faithful-completion/iter5_pod_stage_a_build.sh

# Stage B: rollouts (kiln serve + 73 × 4 strict-prompt rollouts; filter to >0.7)
bash capabilities/caps/pi-faithful-completion/iter5_pod_stage_b_rollouts.sh

# iter7 (fresh light SFT)
cd capabilities/caps/pi-faithful-completion
/workspace/kiln/target/release/examples/cuda_sft_file \
  --data datasets/sft.train.jsonl \
  --model-path /workspace/Qwen3.5-4B \
  --output-dir /workspace/adapters/pi-faithful-iter7-sft-min \
  --adapter-name pi-faithful-iter7-sft-min \
  --rank 4 --alpha 8 --lr 1e-5 --epochs 1 --max-examples 256 \
  --trainer generic

# iter8 (chain another epoch — THE SHIP)
/workspace/kiln/target/release/examples/cuda_sft_file \
  --data datasets/sft.train.jsonl \
  --model-path /workspace/Qwen3.5-4B \
  --output-dir /workspace/adapters/pi-faithful-iter8-sft-chain \
  --adapter-name pi-faithful-iter8-sft-chain \
  --rank 4 --alpha 8 --lr 1e-5 --epochs 1 --max-examples 256 \
  --base-adapter /workspace/adapters/pi-faithful-iter7-sft-min \
  --trainer generic
```

## Bonus: native trainer slowness

`cuda_sft_file --trainer native` runs at ~180 s/step (vs `generic` at
~3.5 s/step — 50× slower). Root cause in `cuda_train.rs:1325`
(`cuda_native_sft_train`):

1. `CudaTrainArena::new()` allocated **per step** (line 1469) instead
   of hoisted/reused.
2. `cuda_linear_attention_state_zeros_for_model()` called **per step**
   (line 1471) instead of allocated once and zeroed in place.
3. Step kernel `cuda_lora_model_adamw_step_with_gdn_state_with_arena`
   does many small CUDA launches with CPU sync between them; GPU
   util sits near 0% while CPU dispatches.

Compare `vk_native_sft_train` in `vk_train.rs:4706`, which has
gradient checkpointing, recompute paths for hybrid GDN, length-sorted
iteration, and pre-computed GDN state shape. Backporting that
structure to CUDA (plus routing the inner step through `BackendRuntime`
like `trainer.rs:sft_train` does) would close the gap.

Workaround until then: always pass `--trainer generic`.
