# PB-H2: Train-Rollout Agentic GRPO + Light ECHO

## Hypothesis

PB-H1 showed that idealized SFT slightly lifts edit completion but damages the
final-response contract and efficiency. Training on the model's own train-only
Pi rollouts should preserve the distribution of real tool behavior while the
rubric supplies outcome/format pressure. A small ECHO-enabled GRPO arm should
lift composite by improving real rollout decisions instead of imitating
synthetic traces.

## Recipe

- Collect train-only Pi rollouts from `datasets/train.tasks.jsonl`, bounded to
  16 tasks × 3 generations for the first arm.
- Use the generated `grpo-train.jsonl` groups only if reward variance survives
  `--filter-var-min=0.05`.
- Trainer: `cuda_grpo_ablation`, `rank=4`, `alpha=8`, `lr=5e-6`,
  `echo_lambda=0.02`, `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during rollout collection.

## Falsification

Reject if any of:

- dry-run keeps fewer than 4 reward-varied groups;
- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.90`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Rejected.

- Clean train collection from the base server produced 16 groups / 48
  completions with mean train reward 0.3707. Ten groups survived
  `--filter-var-min=0.05`, so the dry-run gate passed.
- A first collection attempt accidentally used the previously loaded PB-H1
  adapter because the server default adapter was still active; those artifacts
  were moved aside and not used for training.
- Training completed with gradient checkpointing
  (`KILN_GRAD_CHECKPOINT_SEGMENTS=32`), `rank=4`, `alpha=8`, `lr=5e-6`, and
  `echo_lambda=0.02`. Peak observed VRAM from the training log was 19,289 MiB;
  the train receipt reports 2,634,309 ms wall clock, 10 groups trained, and
  30 completions trained.
- Adapter verification passed. The verifier measured a nonzero LoRA effect
  (`lora_update_l2_upper_bound=0.9589008210471477`) with rank 4 / alpha 8.
- Blind 3-seed eval scored 0.2994, delta -0.0001 from the 0.2996 baseline.
  `outcome` improved slightly (0.5000 -> 0.5208), but
  `format_compliance` fell (0.5625 -> 0.5104), leaving composite flat.
  `convention_consistency`, `read_before_edit`, `no_redundant_imports`, and
  `no_style_drift` stayed saturated or near-saturated. Efficiency remained
  within the falsification ceiling at 321.7 thinking chars/tool call.

This falsifies low-rank, low-lambda train-rollout GRPO as a first successful
postbreach stage. The next arm should treat final-response format as a
protected behavior while still applying outcome pressure, likely by adopting
the `pi-faithful-completion` lesson of a light but explicit terminal-state
prompt and stronger GRPO/ECHO hyperparameters.
