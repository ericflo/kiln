# H28 H26 Hard-Negative Rescue Chain

## Hypothesis

H26 moved the target axis: it improved tool-call efficiency on the larger gate,
but hurt outcome. The `pi-faithful-completion` chain work showed that chaining
can help when the second corpus is broader and complementary, and that rank /
alpha must stay gentle. H28 tests that pattern for pi-doctest: continue from
the H26 adapter with a broader train-only hard-negative contrast.

The contrast avoids H27's bad shape. Instead of showing low-reward traces that
repair and eventually pass, each group has:

- a concise verified pass trace: read, edit, doctest, `DONE`
- a low-reward terminal failure: read, wrong edit, `DONE` without doctest

The goal is to push away from unverified terminal guesses while preserving
H26's efficiency prior.

## Data

- Source: `capabilities/caps/pi-doctest/datasets/train.tasks.jsonl`.
- Dataset: `/tmp/pi-doctest-h28-h26-rescue-hardneg/grpo-train.hardneg.jsonl`.
- Base adapter: `pi-doctest-h26-short-pair-r4a8`.
- Groups: 5 train tasks (`common`, `prime_length`, `largest_prime_factor`,
  `count_distinct_characters`, `sum_to_n`).
- Completions: 10, with rewards `[1.0, 0.0]` in each group.
- Reward stdev: 0.5.

Dry-run token shape:

| metric | value |
| --- | ---: |
| groups | 5 |
| completions | 10 |
| action tokens | 936 |
| env tokens | 1177 |
| context tokens | 2710 |
| max seq length | 554 |
| max action tokens/completion | 150 |

## Training

Command family: `cuda_grpo_ablation --mode baseline`, chained from
`Qwen3.5-4B/adapters/pi-doctest-h26-short-pair-r4a8`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, ECHO lambda 0.05.

Gradient checkpointing was explicitly enabled with
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. Training completed in 270s with observed
peak VRAM about 15981 MiB. The receipt reports 258513 ms in backward, 4859 ms
in reference forward, and 1957 ms in policy forward.

Adapter: `pi-doctest-h28-h26-hardneg-r4a8`.

`kiln adapter verify` passed with 400 nonzero LoRA tensors and delta proxy
0.606598.

## Blind Smoke

`LIMIT=4 SEEDS=1`, paired against `/tmp/pi-doctest-thinking-on-smoke.json`.

| metric | base | H28 |
| --- | ---: | ---: |
| composite | 0.934375 | 0.990625 |
| delta | | +0.056250 |
| outcome | 1.000000 | 1.000000 |
| tested_before_done | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.968750 |
| mean tool calls | 5.25 | 3.50 |
| mean thinking chars | 1966.5 | 1066.0 |
| mean wall-clock s | 35.82 | 26.54 |

## Larger Gate

`LIMIT=8 SEEDS=1`, paired against `/tmp/pi-doctest-h19-promo-base8.json`.

| metric | base | H28 |
| --- | ---: | ---: |
| composite | 0.832812 | 0.659375 |
| delta | | -0.173437 |
| outcome | 0.875000 | 0.687500 |
| tested_before_done | 1.000000 | 0.937500 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.703125 |
| mean tool calls | 5.25 | 6.00 |
| mean thinking chars | 3486.0 | 3518.125 |
| zero rollouts | 1 | 2 |
| mean wall-clock s | 49.62 | 70.71 |

## Verdict

Rejected for promotion. H28 is another strong n=4 false positive: the cheap
smoke looked excellent, but the broader gate regressed every load-bearing
metric except format.

Lessons:

- Chaining from H26 did not rescue outcome; it amplified the fragility seen in
  H26's larger gate.
- Strong hard negatives with `reward=0` train locally and fit the laptop
  profile, but they can destabilize broader task behavior when chained onto a
  rejected efficiency adapter.
- Future chain experiments should only chain from an adapter that passes the
  larger gate, or should first train the broad hard-negative contrast fresh
  from base before stacking any efficiency prior.
- `LIMIT=4` smoke is now definitively a throughput check only; it is not a
  decision gate for this cap.
