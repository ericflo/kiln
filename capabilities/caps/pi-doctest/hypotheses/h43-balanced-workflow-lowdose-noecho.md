# H43: balanced workflow low-dose contrast

## Hypothesis

H41 and H42 trained only terminal post-pass-stop contexts. Both were
smoke-positive and promotion-negative, so H43 adds pre-pass workflow coverage:
read before testing, test after writing, and stop after passing.

The goal is to keep the useful stop-after-pass behavior while preventing the
adapter from treating terminal stopping as the whole task.

Recipe:

- rank 4, alpha 4 (`alpha/rank = 1`);
- lr `1e-6`;
- no ECHO;
- same base model, no base adapter;
- thinking enabled.

## Data

Dataset:
`/tmp/pi-doctest-h43-balanced-workflow-lowdose/grpo-train.workflow-balanced.g5.jsonl`.

The five train-only groups were:

- 1 start-context group: prefer reading `solution.py` over running doctests
  before reading.
- 2 post-write groups: prefer running doctests over finishing immediately
  after a write.
- 2 post-pass groups: prefer `DONE` over rerunning doctests after a passing
  doctest result.

Dry-run passed with:

- 5 groups, 10 completions.
- Rewards `[1.0, 0.0]` per group.
- 324 action tokens.
- 0 env tokens.
- 4500 context tokens.
- Max sequence length 693.
- Max action tokens per completion 45.
- Reward mean 0.5, stdev 0.5.
- `alpha_over_rank=1`.

## Training

Adapter: `pi-doctest-h43-balanced-workflow-lowdose-noecho-r4a4-lr1e6`.

Training used `KILN_GRAD_CHECKPOINT_SEGMENTS=24` and completed successfully.
The receipt/logs report:

- wall-clock: 287880 ms in the receipt, 291.403s observed by the CLI;
- rank 4, alpha 4, lr `1e-6`;
- 5 groups, 10 completions;
- 324 action tokens, 0 env tokens, 4500 context tokens;
- 26870.927 ms reference forward;
- 8593.054 ms policy forward;
- 251915.503 ms backward;
- final loss -0.000460.

The CLI reported peak VRAM 15970 MiB. Adapter verify passed with rank 4,
alpha 4, 400 nonzero tensors, 200 projection pairs, and LoRA update L2
upper-bound 0.11518519680420476.

## Smoke

Blind `LIMIT=4 SEEDS=1` smoke was positive:

- Base composite: 0.9343750000000001.
- H43 composite: 0.953125.
- Delta: +0.018749999999999933.
- Outcome: 1.0.
- Tested-before-done: 1.0.
- Format compliance: 1.0.
- Tool-call efficiency: 0.84375.
- Mean tool calls: 4.5.
- Mean thinking chars: 1821.5.
- Mean wall-clock: 36.972637712955475s.
- Zero rollouts: 0.

The smoke lift was smaller than H41/H42 but still improved tool efficiency and
thinking length while preserving outcome/tested/format.

## Promotion Check

Blind `LIMIT=8 SEEDS=1` rejected H43:

- Base composite: 0.8328125.
- H43 composite: 0.6484375.
- Delta: -0.18437499999999996.
- Outcome: 0.6875.
- Tested-before-done: 0.8125.
- Format compliance: 1.0.
- Tool-call efficiency: 0.8125.
- Mean tool calls: 4.875.
- Mean thinking chars: 3284.125.
- Mean wall-clock: 80.6572782099247s.
- Zero rollouts: 2.

Workflow coverage reduced the worst H42 zero-rollout count but introduced a
tested-before-done regression and still damaged outcome reliability.

## Verdict

Rejected at promotion.

H43 falsifies the small balanced micro-contrast route as currently shaped. The
recent pattern is consistent: these adapter updates find an easy-slice tool
efficiency improvement but do not generalize to the broader gate. The next
attempt should not add more micro-contrast GRPO rows. Better options are to
change the prompt/controller policy outside the adapter, or gather broader
successful train rollouts and train a gentler supervised/OPD-style distribution
that preserves complete task solving instead of isolated tool-choice moments.

No eval task contents or per-example eval transcripts were inspected.
