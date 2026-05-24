# H72 Natural Post-Read Edit Choice

## Hypothesis

H72 retried H71's natural post-read edit-choice data at a loadable dose. H71
proved that `lr=5e-8` was too small to pass the adapter canary. H72 kept every
other variable fixed and raised the learning rate to `2e-7`.

The falsifiable prediction was that a measurable natural edit-choice adapter
would improve outcome reliability by preferring successful implementation edits
over real wrong first edits without adding terminal-stop or repair-tail drift.

## Data

Dataset:
`/tmp/pi-doctest-h71-natural-postread-edit-choice/grpo-train.natural-postread-edit-choice.g2.jsonl`.

Source:
`/tmp/pi-doctest-h44-natural-compact-grpo/grpo-train.compact-pair.jsonl`.

Shape: 2 groups, 4 completions, rewards 1.0 vs 0.25 in each group, reward
stdev 0.375, 116 action tokens, 0 env tokens, and 1618 context tokens.

## Training

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4 / lr
`2e-7`, no ECHO, seed `3141592653`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 92.377s observed with peak
observed VRAM 15989 MiB. The receipt reported 88.031s wall-clock.

Output adapter:
`pi-doctest-h72-natural-postread-edit-choice-r4a4lr2e7`.

Safetensors hash:
`sha256:a1e4fd4f280b88c44986c9673d1d6b28eb165bb8844684ee41da15ed361104f1`.

## Verification

Adapter verify passed against the running server:

- Rank 4 / alpha 4.
- 400 LoRA tensors, 200 matched projection pairs.
- LoRA update proxy 0.011568.
- Server load succeeded for
  `pi-doctest-h72-natural-postread-edit-choice-r4a4lr2e7`.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.925, no zero rollouts, mean wall-clock 27.44s.
- H72: composite 0.798958, no zero rollouts, mean wall-clock 46.09s.
- Delta: -0.126042 composite, slower by 18.65s.

No wider promotion check was run.

## Verdict

Rejected at smoke. Localizing the natural contrast to the first edit avoided
the zero-rollout failures seen in several wider-gate rejects, but it still hurt
composite and substantially increased latency. The branch closes a useful
question: natural wrong-edit negatives are not safe merely because the contrast
is local and train-only.

Future policy data should not rank wrong implementation edits directly unless
it is paired with a broader confirmed behavior anchor. The repeated pattern is
that narrow policy updates can change the model, but they perturb live
thinking-enabled search enough to lose blind reliability.

No eval task contents or per-example eval transcripts were inspected.
