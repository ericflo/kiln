# H41: post-pass stop contrast

## Hypothesis

Prior thinking-compression attempts tried to shorten entire successful traces,
but they either overfit one workflow or made broader GRPO too expensive. H41
tests a narrower behavior: once a doctest run has already passed in the prompt
context, prefer ending with `DONE` over running the same doctest command again.

This keeps the model's binary thinking setting on and avoids supervising
solution code. The target is efficient stopping after successful verification,
not less capable reasoning.

## Data

Dataset:
`/tmp/pi-doctest-h41-postpass-stop-g4/grpo-train.postpass-stop.g4.jsonl`.

The four train-only groups were built from prior SFT train artifacts:

- `/tmp/pi-doctest-h20-bookend-micro-sft/sft.train.jsonl`, source lines 2, 3,
  and 4.
- `/tmp/pi-doctest-h19-ultrashort-write-sft-cap2900/sft.train.jsonl`, source
  line 1.

Each group uses a context ending after a successful doctest tool result. The
preferred completion is a brief thinking block followed by `DONE`; the rejected
completion thinks about rerunning doctests and emits another doctest tool call.

Dry-run passed with:

- 4 groups, 8 completions.
- Rewards `[1.0, 0.0]` per group.
- 260 action tokens.
- 0 env tokens.
- 2970 context tokens.
- Max sequence length 452.
- Max action tokens per completion 45.
- Reward mean 0.5, stdev 0.5.

The zero env-token count is expected because this is a text-completion GRPO
shape with prompt messages, not trajectory ECHO training.

## Training

Adapter: `pi-doctest-h41-postpass-stop-g4-noecho-r4a8`.

Training used `KILN_GRAD_CHECKPOINT_SEGMENTS=24` and completed successfully.
The receipt/logs report:

- wall-clock: 202119 ms in the receipt, 205.949s observed by the CLI;
- rank 4, alpha 8, lr `5e-6`;
- 4 groups, 8 completions;
- 260 action tokens, 0 env tokens, 2970 context tokens;
- 2853.455 ms reference forward;
- 1097.401 ms policy forward;
- 197786.521 ms backward;
- final loss 0.002202.

The CLI reported peak VRAM 15965 MiB. Adapter verify passed with rank 4,
alpha 8, 400 nonzero tensors, 200 projection pairs, and LoRA update L2
upper-bound 0.7870061197607627.

## Smoke

Blind `LIMIT=4 SEEDS=1` smoke was positive:

- Base composite: 0.9343750000000001.
- H41 composite: 0.971875.
- Delta: +0.03749999999999998.
- Outcome: 1.0.
- Tested-before-done: 1.0.
- Format compliance: 1.0.
- Tool-call efficiency: 0.90625.
- Mean tool calls: 4.0.
- Mean thinking chars: 1664.5.
- Mean wall-clock: 33.50304573774338s.
- Zero rollouts: 0.

This is the first recent smoke result that improved both tool efficiency and
thinking length while preserving outcome/tested/format.

## Promotion Check

Blind `LIMIT=8 SEEDS=1` rejected H41:

- Base composite: 0.8328125.
- H41 composite: 0.7859375.
- Delta: -0.046875.
- Outcome: 0.875.
- Tested-before-done: 1.0.
- Format compliance: 1.0.
- Tool-call efficiency: 0.59375.
- Mean tool calls: 7.25.
- Mean thinking chars: 3783.25.
- Mean wall-clock: 75.03890404105186s.
- Zero rollouts: 1.

The promotion regression is the opposite of the smoke behavior: the adapter
learned the desired stop signal on the easy slice but made the broader gate
less efficient and less reliable.

## Verdict

Rejected at promotion.

H41 shows that the post-pass-stop contrast is the right behavioral target but
too narrow as an adapter update. A next attempt should either combine this stop
contrast with earlier workflow coverage or train it with a lower-impact method
so it does not overwrite broader task-solving behavior. The data should also
include contexts before the first doctest pass, so the model learns both
"verify once" and "stop after verified" instead of only the terminal stop move.

No eval task contents or per-example eval transcripts were inspected.
