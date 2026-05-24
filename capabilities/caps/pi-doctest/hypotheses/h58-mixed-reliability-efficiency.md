# H58 Mixed Reliability Plus Efficiency

## Hypothesis

H57 showed that low-dose terse-vs-verbose one-action suffix ranking still
creates reliability failures at the wider gate. H58 keeps the trainable suffix
shape, but mixes in direct reliability pressure: after reading, prefer the
correct edit over premature `DONE`; after editing, prefer running doctest over
premature `DONE`; only after a passing doctest use a softened concise-vs-verbose
`DONE` contrast.

## Data

- Source: `/tmp/pi-doctest-h54-step-local-concise/grpo-train.step-local-concise.g4x3.jsonl`.
- Output: `/tmp/pi-doctest-h58-mixed-reliability-efficiency/grpo-train.mixed-reliability-efficiency.g2x3.jsonl`.
- Selection: first two train-only H54 success anchors, three suffix positions
  each.
- Groups: 6.
- Completions: 12.
- Rewards: 1.0 vs 0.60 for post-read/post-edit premature-stop contrasts; 1.0
  vs 0.85 for post-pass concise-vs-verbose `DONE`.
- Dry-run tokens: 300 action, 0 env, 5422 context.
- Reward stdev: 0.178924.

## Training

Adapter: `pi-doctest-h58-mixed-reliability-efficiency-g6-r4a4lr1e7`.

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4, lr
`1e-7`, seed `3141592653`, no ECHO, reward variance filter min `0.001`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 437.093s observed with peak observed VRAM 15983
MiB. The receipt reported 300 action tokens, 0 env tokens, 5422 context tokens,
reference forward 121116 ms, policy forward 22460 ms, and backward 288556 ms.
Adapter verify passed with 400 nonzero LoRA tensors and a LoRA update proxy of
0.016932.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.896875, no zero rollouts, mean wall-clock 30.84s.
- H58: composite 0.98125, no zero rollouts, mean wall-clock 40.37s.
- Delta: +0.084375 composite, but slower by 9.53s.

Promotion, `LIMIT=8 SEEDS=1`:

- Base: composite 0.86484375, no zero rollouts, mean wall-clock 55.48s.
- H58: composite 0.7265625, two zero rollouts, mean wall-clock 67.08s.
- Delta: -0.13828125 composite, slower by 11.60s.

## Verdict

Rejected at promotion. The mixed reliability signal improved smoke, but it did
not prevent wider-gate zero rollouts and it worsened latency. The useful
systems result is that the mixed six-group shape trained much faster than H57,
but the capability lesson is negative: a tiny anti-premature-DONE suffix
contrast is still too narrow and still shifts the policy into unreliable
terminal behavior.

Next work should avoid another narrow suffix-only GRPO pair. Better candidates:
broader outcome-preserving success anchors, a less policy-drifty method for
terminal/action discipline, or a chain only after a confirmed general behavior
anchor.
