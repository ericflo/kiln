# H19: ultrashort write SFT

## Hypothesis

H18's compact write-style SFT data was directionally right but still too slow
when examples exceeded about 3000 supervised chars. H19 keeps the same
thinking-enabled successful workflow anchor and applies a stricter 2900-char
cap.

Dataset: `/tmp/pi-doctest-h19-ultrashort-write-sft-cap2900/sft.train.jsonl`.

- 6 SFT examples.
- Derived only from outcome-perfect, tool-efficient train rollouts.
- Paths normalized to `solution.py`.
- Real tool arguments preserved.
- Code-writing action rendered as `write` with the final file content.
- Char range: 2691 to 2888.

Training used rank 4 / alpha 8 / lr `1e-5` / 1 epoch through the SFT API.

## Result

Kept at the smoke gate, with a promotion-check caveat.

Training job `256391d7-d840-44de-8149-37ec4759a32c` completed in 128004 ms.
The receipt reports 6 examples trained, 1995 supervised action tokens, 2751
context tokens, and adapter-smoke-test passed. Adapter verify passed with rank
4, alpha 8, 400 nonzero LoRA tensors, and LoRA update L2 upper-bound
1.7768756351933703.

Blind aggregate smoke, `LIMIT=4 SEEDS=1`, compared to the normalized
thinking-on base smoke:

- Base composite: 0.90625.
- H19 composite: 0.94375.
- Delta: +0.0375.
- Outcome: 1.0.
- Tested-before-done: 1.0.
- Tool-call efficiency: 0.8125.
- Mean tool calls: 5.0.
- Mean thinking chars: 1717.0.
- Mean thinking chars per tool call: 289.68333333333334.
- Mean wall-clock: 37.70267564058304s.

This is not a full promotion yet. The smoke is only 4 blind rollouts, and
wall-clock is slightly worse than the base smoke even though tool calls and
thinking chars improved. The next step should be a larger paired blind
promotion check before treating H19 as a kept stage.

## Promotion Check

The larger paired blind check rejected H19.

`LIMIT=8 SEEDS=1` base:

- Composite: 0.8328125.
- Outcome: 0.875.
- Tested-before-done: 1.0.
- Tool-call efficiency: 0.78125.
- Mean tool calls: 5.25.
- Mean thinking chars: 3486.0.
- Mean wall-clock: 49.624973833560944s.
- Zero rollouts: 1.

`LIMIT=8 SEEDS=1` H19:

- Composite: 0.6828125.
- Delta: -0.15.
- Outcome: 0.6875.
- Tested-before-done: 0.9375.
- Tool-call efficiency: 0.734375.
- Mean tool calls: 7.0.
- Mean thinking chars: 3587.0.
- Mean wall-clock: 88.93232873082161s.
- Zero rollouts: 2.

The 4-rollout smoke lift was not stable. H19 should not be promoted. The
main lesson is that ultrashort SFT can complete locally, but this particular
workflow anchor overfit and made the adapter less reliable on the broader
blind slice.

No eval task contents or per-example eval transcripts were inspected.
