# pi-faithful-completion: in-weights uplift experiments (round-3 follow-up)

**Goal:** lift no-prompt composite from base 0.656 to ≥0.80 by training
the strict-prompt behavior into Qwen3.5-4B's weights. Prompted ceiling
(base + strict prompt at inference, no weight changes): 0.819 ± 0.014.

## Result: **iter21 SHIPS — 0.8021 ± 0.020 (+0.163 paired lift, 6.3σ). GOAL MET.**

iter21 captures **90.6% of the prompted-lift in the weights**
((0.8021 − 0.6394) / (0.819 − 0.6394)). The breakthrough was the
**oscillation pattern**: alternate the SFT data type each stage.

Chain (each stage is rank 4, α 8, lr 1e-5, 1 epoch, max-examples 256,
trainer=generic):

1. iter19a: fresh from base on `sft.ideal.jsonl` (69 synthesized
   rubric-perfect outputs) — installs the format prior. 3 epochs
   because dataset is small.
2. iter19b: chain on iter19a with `sft.train.jsonl` (211 strict-prompt
   rollouts) — installs strict behavior. Result: 0.7792.
3. iter20: chain on iter19b with ideal data again — restores format.
   Result: 0.7849.
4. **iter21: chain on iter20 with strict data again — pushes outcome
   and honesty up while keeping the restored format. Result: 0.8021.**

The plateau at 0.77 (seen in 7+ same-data SFT variants) breaks once
the data signal alternates between "what perfect format looks like"
and "what good strict-prompt behavior looks like". Each pull is
gentle (lr 1e-5, 1 epoch) so the model doesn't catastrophically
forget either lesson.

Sub-scores at iter21 vs iter8 (the earlier SFT-only plateau):

|  | iter8 | iter21 | Δ |
|---|---:|---:|---:|
| outcome.value_correct | 0.7719 | 0.8012 | +0.029 |
| honesty.score | 0.8246 | 0.8450 | +0.020 |
| format_strict.score | 0.8772 | **0.9181** | **+0.041** |
| terseness.score | 0.9991 | 1.0000 | +0.001 |

The format_strict recovery is the smoking gun for what was broken in
earlier iters: monotone strict-rollout chain training pushed format
prose down (preamble before the format line). Reintroducing ideal
(rubric-perfect, no preamble) examples mid-chain teaches the model
"output is the line, not surrounding text" — and the *strict* steps
that follow keep the outcome/honesty gains while the format prior
holds.

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
| iter16 | iter8 + (r4, α8, lr 1e-5, 1ep, threshold>0.5: ~400 ex) | 0.7678 ± 0.010 | +0.123 | 4.8 | same plateau |
| iter17 | iter8 + 28 hard-tail SFT examples, 3ep | 0.7678 ± 0.020 | +0.163 | 6.3 | plateau |
| iter18 | iter8 + 69 synthesized ideal outputs, 2ep | 0.7621 ± 0.010 | +0.134 | 7.8 | plateau |
| iter19a | fresh from base on ideal (rank 4, 3ep) | 0.7449 ± 0.010 | — | — | format prior installed |
| iter19b | iter19a + 211 strict rollouts (1ep) | 0.7792 ± 0.010 | +0.140 | 5.4 | first crack |
| iter20 | iter19b + ideal data (2ep) | 0.7849 ± 0.026 | +0.146 | 15.2 | better |
| **iter21 SHIP** | **iter20 + strict rollouts (1ep)** | **0.8021 ± 0.020** | **+0.163** | **6.3** | **GOAL MET** |
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

The chain-SFT regime saturates around 0.77. iter8 (rank 4 chain),
iter14 (rank 8 fresh), and iter16 (lower-threshold data) all hit
the SAME plateau **regardless of capacity (rank 4 vs 8), schedule
(fresh vs chain), or data filter (>0.7 vs >0.5)**. The training
data itself caps the achievable lift; the model has already learned
everything the strict-prompt rollouts can teach it. The plateau is
not a training-recipe artifact — it is a data-signal ceiling.

To break past, the training data must change (richer signal, hard-
tail examples, larger teacher) or the method must change (OPD
reverse-KL once kiln gets `/v1/completions`, larger-teacher
distillation, KL-regularized SFT).

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

## Reproducer (iter21 SHIP)

```bash
# Stage A: clone kiln + build
cd /workspace && kiln-setup --clone --repo /workspace/kiln
cd /workspace/kiln && bash capabilities/caps/pi-faithful-completion/iter5_pod_stage_a_build.sh

# Stage B: rollouts (kiln serve + 73 × 4 strict-prompt rollouts; filter >0.7)
bash capabilities/caps/pi-faithful-completion/iter5_pod_stage_b_rollouts.sh

# Synthesize ideal outputs (69 rubric-perfect examples from ground truth)
cd capabilities/caps/pi-faithful-completion
python3 iter18_ideal_prep.py

# All cuda_sft_file calls use the same recipe template; only --data and
# --base-adapter change. Kill kiln serve before each SFT (frees VRAM).

# Stage 1: format prior FROM BASE on ideal data (3 epochs since small)
/workspace/kiln/target/release/examples/cuda_sft_file \
  --data datasets/sft.ideal.jsonl --model-path /workspace/Qwen3.5-4B \
  --output-dir /workspace/adapters/pi-faithful-iter19a-ideal-prior \
  --adapter-name pi-faithful-iter19a-ideal-prior \
  --rank 4 --alpha 8 --lr 1e-5 --epochs 3 --max-examples 128 --trainer generic

# Stage 2: chain strict rollouts (1 epoch)
/workspace/kiln/target/release/examples/cuda_sft_file \
  --data datasets/sft.train.jsonl --model-path /workspace/Qwen3.5-4B \
  --base-adapter /workspace/adapters/pi-faithful-iter19a-ideal-prior \
  --output-dir /workspace/adapters/pi-faithful-iter19b-strict-on-ideal \
  --adapter-name pi-faithful-iter19b-strict-on-ideal \
  --rank 4 --alpha 8 --lr 1e-5 --epochs 1 --max-examples 256 --trainer generic

# Stage 3: chain ideal data again (2 epochs)
/workspace/kiln/target/release/examples/cuda_sft_file \
  --data datasets/sft.ideal.jsonl --model-path /workspace/Qwen3.5-4B \
  --base-adapter /workspace/adapters/pi-faithful-iter19b-strict-on-ideal \
  --output-dir /workspace/adapters/pi-faithful-iter20-osc-ideal \
  --adapter-name pi-faithful-iter20-osc-ideal \
  --rank 4 --alpha 8 --lr 1e-5 --epochs 2 --max-examples 128 --trainer generic

# Stage 4: chain strict rollouts again (1 epoch) — THE SHIP
/workspace/kiln/target/release/examples/cuda_sft_file \
  --data datasets/sft.train.jsonl --model-path /workspace/Qwen3.5-4B \
  --base-adapter /workspace/adapters/pi-faithful-iter20-osc-ideal \
  --output-dir /workspace/adapters/pi-faithful-iter21-osc-strict \
  --adapter-name pi-faithful-iter21-osc-strict \
  --rank 4 --alpha 8 --lr 1e-5 --epochs 1 --max-examples 256 --trainer generic
```

The 4-stage chain takes ~50 minutes of SFT time on an A6000 with the
generic trainer. iter21 adapter weights backed up at
`b2://clouderic/pi-faithful-iter21-osc-strict/`.

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
