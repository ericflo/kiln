# H20: bookend micro SFT

## Hypothesis

H19 showed that supervising full code-writing trajectories overfits and hurts
broader blind reliability. H20 tries the opposite extreme: no solution-code
supervision at all.

The dataset trains only workflow bookends:

- first action: briefly think, then read `solution.py`;
- final action after passing doctest: briefly think, then `DONE`.

This should reduce pre-tool thinking and post-success tool loops without
teaching any task-specific implementations.

Dataset: `/tmp/pi-doctest-h20-bookend-micro-sft/sft.train.jsonl`.

- 4 SFT examples.
- 219 supervised assistant tokens.
- 1183 context tokens.
- No old/new code edit targets.
- No task solution code in supervised assistant spans.

Training used rank 4 / alpha 8 / lr `5e-6` / 1 epoch through the SFT API.

## Training

Job `68b13f0c-91f5-4649-bd07-91230d14996e` completed successfully. The CLI
status showed 415s elapsed; the train receipt recorded 316533 ms. Adapter
smoke test passed. Adapter verify passed with 400 nonzero LoRA tensors and
LoRA update L2 upper-bound 0.503938138547667.

The small supervised target was not enough to make training cheap: even this
micro dataset spent several minutes in SFT because context tokens still drive
the forward/backward cost.

## Smoke

Blind smoke, `LIMIT=4 SEEDS=1`, looked excellent:

- Base composite: 0.90625.
- H20 composite: 0.98125.
- Delta: +0.075.
- Outcome: 1.0.
- Tested-before-done: 1.0.
- Tool-call efficiency: 0.9375.
- Mean tool calls: 3.75.
- Mean thinking chars: 1360.25.
- Mean wall-clock: 23.031202018260956s.

## Promotion Check

The larger paired blind check rejected H20.

`LIMIT=8 SEEDS=1` base, reused from the H19 promotion check:

- Composite: 0.8328125.
- Outcome: 0.875.
- Tested-before-done: 1.0.
- Tool-call efficiency: 0.78125.
- Mean tool calls: 5.25.
- Mean thinking chars: 3486.0.
- Mean wall-clock: 49.624973833560944s.
- Zero rollouts: 1.

`LIMIT=8 SEEDS=1` H20:

- Composite: 0.7265625.
- Delta: -0.10625.
- Outcome: 0.75.
- Tested-before-done: 1.0.
- Tool-call efficiency: 0.6875.
- Mean tool calls: 6.125.
- Mean thinking chars: 4370.5.
- Mean wall-clock: 81.99138644337654s.
- Zero rollouts: 2.

The bookend idea produced another false-positive 4-task smoke. On the broader
blind slice it increased thinking and tool use, and reduced outcome. Do not
promote H20.

No eval task contents or per-example eval transcripts were inspected.
