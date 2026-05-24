# H65 Post-Failed-Doctest Repair Local GRPO

## Hypothesis

H64 showed that full-trajectory direct-success-over-repair still penalizes the
repair continuation together with the bad first edit. H65 isolates the local
choice immediately after a failed doctest: prefer a real repair action over a
terminal `DONE`.

The falsifiable prediction was that a tiny one-action local contrast could
reduce hard-tail terminal failures without applying broad pressure to complete
successful workflows.

## Data

- Source: `/tmp/pi-doctest-h44-natural-compact-grpo/grpo-train.compact-pair.jsonl`.
- Output: `/tmp/pi-doctest-h65-postfail-repair-local/grpo-train.postfail-repair-local.g2.jsonl`.
- Groups: 2.
- Completions: 4.
- Context: original task plus the real read, wrong edit, failed doctest
  trajectory prefix, ending immediately after the failed doctest observation.
- Preferred completion: the next real repair action from the H44 near-miss.
- Rejected completion: a synthetic terminal `DONE` after the failed doctest.
- Rewards: 1.0 vs 0.0.

Dry-run:

- action tokens: 116.
- env tokens: 0.
- context tokens: 3132.
- reward stdev: 0.5.

## Training

Adapter: `pi-doctest-h65-postfail-repair-local-g2-r4a4lr1e7`.

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4, lr
`1e-7`, seed `3141592653`, no ECHO, reward variance filter min `0.001`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 642.587s observed with peak observed VRAM 16002
MiB. The receipt reported 2 groups trained, 4 completions, 116 action tokens,
0 env tokens, 3132 context tokens, and reward stdev 0.5. Adapter verify
passed with 400 nonzero LoRA tensors and a small LoRA update proxy of
0.005614.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.925, no zero rollouts, mean wall-clock 34.11s.
- H65: composite 0.75, one zero rollout, mean wall-clock 40.83s.
- Delta: -0.175 composite, slower by 6.72s.

No wider promotion check was run.

## Verdict

Rejected at smoke. H65 confirms that "repair over DONE after failed doctest"
is too narrow as a standalone one-action policy update. Even at a very small
LoRA delta, it introduced a zero rollout and latency regression.

The useful systems lesson is that context tokens dominate local GRPO cost:
only 116 action tokens still took about 643s because the failed-doctest
contexts were large. The next route should avoid additional failed-state
micro-contrasts and instead look for either broader outcome-preserving data or
a non-adapter harness/prompt/decoding control for post-failure behavior.

No eval task contents or per-example eval transcripts were inspected.
