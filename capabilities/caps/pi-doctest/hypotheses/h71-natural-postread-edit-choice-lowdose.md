# H71 Natural Post-Read Edit Choice Low Dose

## Hypothesis

Recent repair-tail and terminal-stop experiments mixed several behaviors into
one contrast: bad first edit, repair behavior, redundant tests, and stopping.
H71 isolated the earliest semantic decision after the model had read
`solution.py`: choose the successful train-only edit action instead of the real
wrong first edit action from the same natural-variance rollout group.

The falsifiable prediction was that this local natural edit-choice contrast
would target implementation quality without teaching terminal-stop or
repair-tail artifacts.

## Data

Source:
`/tmp/pi-doctest-h44-natural-compact-grpo/grpo-train.compact-pair.jsonl`.

Output:
`/tmp/pi-doctest-h71-natural-postread-edit-choice/grpo-train.natural-postread-edit-choice.g2.jsonl`.

For each of the two H44 natural variance pairs, the context was cut after the
successful read action and read observation. The completions were the next
assistant action only:

- preferred: successful edit action, reward 1.0;
- rejected: real wrong first edit action from the near-miss rollout, reward
  0.25.

Dry-run shape was 2 groups, 4 completions, rewards mean 0.625, reward stdev
0.375, 116 action tokens, 0 env tokens, and 1618 context tokens. The dataset
hash was
`sha256:c69d7a8329c0f0101152699c8304a105b16f67371b84c9b019dc9848f2e1fc8a`.

## Training

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4 / lr
`5e-8`, no ECHO, seed `3141592653`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 83.435s observed with peak
observed VRAM 15989 MiB. The receipt reported 79.654s wall-clock.

Output adapter:
`pi-doctest-h71-natural-postread-edit-choice-r4a4lr5e8`.

Safetensors hash:
`sha256:7f65a534441581c3b770ea5f816c18de4073443fc05de0bca2f989e76686ec0c`.

## Verification

Rejected before blind eval. The training adapter smoke test failed all three
canary prompts with no checked-logit delta above threshold. Offline adapter
verify found nonzero tensors and a tiny LoRA update proxy of 0.002750, but the
running server quarantined the adapter because of the failed canary checks.

## Verdict

Rejected at verification. The data shape is trainable and compact, but
`lr=5e-8` is below a useful dose for this GRPO signal. H72 immediately retried
the same data at `lr=2e-7`.

No eval task contents or per-example eval transcripts were inspected.
