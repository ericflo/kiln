# H2: Short Positive Rollout SFT Bootstrap

## Hypothesis

The repaired baseline's largest headroom is still `format_compliance` and
`outcome`, while `convention_consistency` is already near-saturated. A small
SFT bootstrap on short, high-reward train rollouts should teach the final
response contract and the read/edit/verify rhythm more directly than sparse
agentic-GRPO, without increasing thinking cost.

## Recipe

- Source: train rollouts only from H1, never eval tasks or eval transcripts.
- Selection: one shortest completion per group with `reward >= 0.8` and
  completion length <= 2900 chars.
- Normalization: strip rollout sandbox absolute prefixes back to relative
  paths before training.
- Dataset: 14 examples in `datasets/sft.h2-positive-short.jsonl`; actual
  rewards 0.95-1.00 and completion lengths 1341-2005 chars.
- Trainer: `cuda_sft_file`, generic SFT, `rank=4`, `alpha=8`, `lr=1e-5`,
  one epoch, thinking enabled by the serving runtime.

## Falsification

Reject or do not promote if the 3-seed blind eval shows any of:

- composite lift < +0.05 versus `baseline-0`;
- `format_compliance` or `outcome` fails to improve;
- `convention_consistency` drops below 0.90;
- mean thinking chars/tool call increases by more than 25% over the
  baseline value of 302.7;
- adapter verification fails.

## Results

Status: rejected.

- Default SFT checkpointing OOMed in fused linear cross-entropy
  (`flce phase b cuda_fwd: max_keepdim on logits_chunk`).
- Explicit `KILN_GRAD_CHECKPOINT_SEGMENTS=32` trained and verified the
  rank-4/alpha-8 adapter locally. The run used 14 examples, 1182 action
  tokens, 2140 context tokens, and 159s wall clock; the log observed
  peak VRAM 16,481 MiB.
- Adapter verification passed with nonzero LoRA tensors and offline update
  proxy `lora_update_l2_upper_bound=2.660868`.
- Blind 3-seed eval regressed composite from 0.4800 to 0.2208
  (`delta=-0.2592`). `format_compliance` fell to 0.5972 and `outcome`
  collapsed to 0.3333; `convention_consistency` stayed high at 0.9340.
  Thinking chars/tool stayed near baseline (302.7 to 299.4), but nonzero
  rollouts fell from 18 to 8.

Conclusion: the positive-rollout filter was not enough. Flattened Pi action
transcripts preserve brittle execution artifacts even when their train reward
is high, and SFT on that surface hurts actual edit completion. The next
iteration should either test a stricter runtime prompt contract before
training, or use idealized train-only traces rather than copied tool-action
text.

## Rationale

METHODS routes to SFT when format headroom is a large share of total
headroom. This arm used the low-risk SFT lore from `pi-doctest` and
`pi-faithful-completion`: small data, rank-4/alpha-8, `lr=1e-5`, one epoch,
and a full blind 3-seed eval before considering promotion.
