# H60 Synthetic Ideal SFT

## Hypothesis

H59 showed that SFT on the base model's own successful concise traces still
creates reliability drift. H60 tests a different SFT distribution inspired by
the `pi-faithful-completion` alternating-chain lesson: deterministic
rubric-perfect examples synthesized from the task scaffold, not sampled from
base successes. Use a lower LR than H59 to keep the update small.

## Data

- Output: `/tmp/pi-doctest-h60-synthetic-ideal-sft/sft.synthetic-ideal.g4.jsonl`.
- Examples: 4.
- Tasks: `double`, `is_even`, `first_char`, `clamp`.
- Shape: concise read, `edit`, doctest, concise `DONE`.
- Source: hand-authored synthetic train-style tasks only.
- Character lengths: 1739, 1768, 1845, 1889.
- Tokenized lengths: 507, 515, 525, 592.

## Training

Adapter: `pi-doctest-h60-synthetic-ideal-sft-g4-r4a4lr1e7`.

Training used native `cuda_sft_file`, rank 4 / alpha 4, lr `1e-7`, 1 epoch,
and `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 86.078s observed with peak observed VRAM 15994
MiB. The receipt reported 4 examples trained, 500 action tokens, 1639 context
tokens, rank 4 / alpha 4, and lr `1e-7`. Adapter verify passed with 400
nonzero LoRA tensors and a LoRA update proxy of 0.012513.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.900000, no zero rollouts, mean wall-clock 45.10s.
- H60: composite 0.814583, no zero rollouts, mean wall-clock 44.68s.
- Delta: -0.085417 composite, essentially neutral latency.

No promotion check was run.

## Verdict

Rejected at smoke. Synthetic ideal SFT at this low dose is safer than H59 on
zero rollouts and latency, but it still lowers composite. The useful result is
that the synthetic distribution is fast and not obviously destabilizing; the
negative result is that it lacks enough useful semantic/process signal by
itself.

Next SFT-like work should add explicit base-behavior regularization, use a
stronger/diverse teacher-generated distribution, or wait until there is a
confirmed positive adapter to chain behind. More tiny standalone SFT on simple
ideal traces is unlikely to maximize pi-doctest.
