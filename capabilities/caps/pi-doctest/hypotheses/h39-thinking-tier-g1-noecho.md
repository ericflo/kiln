# H39: thinking-tier g1 no-ECHO

## Hypothesis

H37/H38 showed that compressing successful train-only traces is safer than
wrong/no-test hard negatives, but using the original long trace as the low
reward side is too expensive and one compressed trace overfits. H39 tried a
controlled "thinking tier" contrast instead:

- keep the same successful workflow and tool payload;
- synthesize a brief successful thinking version as the preferred completion;
- synthesize a medium successful thinking version as the lower-reward
  completion;
- disable ECHO so the gradient lands on action-token style rather than
  environment-token imitation.

The intent was to teach "spend fewer thinking tokens for the same verified
workflow" without showing failed terminal guesses or verbose repair loops.

## Data

Fresh train-only base rollouts were collected from
`datasets/train.tasks.jsonl` into
`/tmp/pi-doctest-h39-base-train-rollouts/grpo-train.jsonl`.

The initial three-group dataset was:
`/tmp/pi-doctest-h39-thinking-tier-g3/grpo-train.thinking-tier.g3.jsonl`.

Selection summary:

- `task_0024`, source group 0 completion 1: original reward 1.0, 3 tool
  calls, original 1464 chars, brief 599 chars, medium 747 chars.
- `task_0028`, source group 4 completion 0: original reward 1.0, 3 tool
  calls, original 1782 chars, brief 956 chars, medium 1104 chars.
- `task_0031`, source group 7 completion 1: original reward 1.0, 3 tool
  calls, original 1960 chars, brief 746 chars, medium 894 chars.

Dry-run for the three-group shape passed:

- 3 groups, 6 completions.
- 1230 action tokens.
- 1538 env tokens.
- 1668 context tokens.
- reward mean 0.5, stdev 0.5.

## Throughput probes

The three-group and two-group variants were rejected as local training routes
before adapter write.

With `KILN_GRAD_CHECKPOINT_SEGMENTS=24`, the three-group run fit in VRAM but
was stopped after group 2 became clearly compute-bound. A group-2 reference
forward took 186441 ms even though the max sequence length was only 716.

With `KILN_GRAD_CHECKPOINT_SEGMENTS=16`, the two-group run still fit near the
same memory ceiling, but lower/mid checkpoint segments were too slow for the
900s guard. With `KILN_GRAD_CHECKPOINT_SEGMENTS=8`, memory still fit, but
wall-clock got worse: no group-level progress appeared after more than seven
minutes. For this laptop, fewer checkpoint segments are not a reliable speed
fix for this GRPO path.

The bounded trainable variant was the first group only:
`/tmp/pi-doctest-h39-thinking-tier-g1/grpo-train.thinking-tier.g1.jsonl`.

Dry-run passed:

- 1 group, 2 completions.
- 314 action tokens.
- 588 env tokens.
- 556 context tokens.
- reward mean 0.5, stdev 0.5.

## Training

Adapter: `pi-doctest-h39-thinking-tier-g1-noecho-r4a8`.

Training used:

- base model, no base adapter;
- rank 4, alpha 8, lr `5e-6`;
- `--mode baseline`;
- `--no-echo`;
- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

The run completed successfully in 277518 ms according to the receipt
(280.828 seconds observed by the CLI). Token counts were 314 action tokens,
588 env tokens, and 556 context tokens. Timings were 969.771 ms reference
forward, 950.747 ms policy forward, and 275452.690 ms backward. Peak observed
VRAM from the CLI was 15982 MiB.

Adapter verify passed: rank 4, alpha 8, 400 tensors, 400 nonzero tensors,
200 projection pairs, and LoRA update L2 upper-bound 0.22023191624889799.

## Smoke

Blind `LIMIT=4 SEEDS=1` smoke rejected H39 g1:

- Base composite: 0.9343750000000001.
- H39 g1 composite: 0.925.
- Delta: -0.009375000000000022.
- Outcome: 1.0.
- Tested-before-done: 1.0.
- Format compliance: 1.0.
- Tool-call efficiency: 0.75.
- Mean tool calls: 5.5.
- Mean thinking chars: 2703.0.
- Mean wall-clock: 51.08210378885269s.
- Zero rollouts: 0.

Compared to the paired base smoke, this preserved correctness but worsened
tool calls, thinking length, wall-clock, and composite. The larger promotion
gate was skipped.

## Verdict

Rejected at smoke.

The thinking-tier idea remains behaviorally cleaner than wrong/no-test hard
negatives, but this implementation still teaches a brittle style prior. The
one-group version is trainable but not useful; broader versions are currently
throughput-bound on the 16GB laptop. Future attempts should either lower the
target below the current GRPO backward cost (for example shorter synthetic
bookends with no code/tool payload), or switch methodology to a small
alternating SFT chain with complementary ultra-short workflow-only data rather
than more policy-on GRPO preference pairs.

No eval task contents or per-example eval transcripts were inspected.
