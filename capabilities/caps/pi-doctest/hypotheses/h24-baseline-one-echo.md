# H24: Baseline-mode one-completion ECHO

## Hypothesis

H23 showed that one-completion `--no-policy-loss` ECHO is rejected in Phase 1
mode because dynamic GRPO filtering removes the degenerate one-rollout group.
It also showed that the smallest accepted Phase 1 shape, one group with two
completions, is locally too slow because token-level group accumulation keeps
the run expensive.

H24 tests a no-code training shape: use `--mode baseline` with
`--no-policy-loss`. Baseline mode disables dynamic sampling and uses
per-sample loss aggregation, so a one-completion group is accepted and the
optimizer steps immediately after that completion. Because `--no-policy-loss`
is still set, the policy-gradient term is multiplied by zero and only ECHO's
environment-token CE term should drive gradients.

## Data

Dataset:
`/tmp/pi-doctest-h23-echo-one-completion/grpo-train.one.jsonl`.

This is the single highest-reward short H17 completion:

- reward: 0.9871964285714285
- sequence length: 814 tokens
- action tokens: 280
- env tokens: 256
- context tokens: 278

Dry-run command used:

- `--mode baseline`
- `--no-policy-loss`
- `--echo-lambda 0.05`
- rank 4 / alpha 8 / lr `5e-6`
- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`

Dry-run passed:

- 1 valid group.
- 1 valid completion.
- 280 action tokens.
- 256 env tokens after warning filtering.
- 278 context tokens.

The saturated-reward warning is expected and supports the verifier-free ECHO
choice; no policy-gradient loss should be used on this group.

## Training

Training succeeded locally in 129.059 seconds and installed:

`Qwen3.5-4B/adapters/pi-doctest-h24-baseline-one-echo-r4a8`

Training details:

- `no_policy_loss=true`
- ECHO lambda `0.05`
- rank 4 / alpha 8 / lr `5e-6`
- gradient checkpoint segments: 24
- final loss: 0.190592
- ECHO env CE: 3.8116098009049892
- backward time: 124200 ms
- peak observed VRAM: 15975 MiB

`kiln adapter verify` passed offline and through the running server:

- 400 tensors.
- 200 LoRA projection pairs.
- nonzero LoRA tensors found.
- delta proxy L2 upper bound: 0.166696.

## Smoke Eval

Blind smoke: `LIMIT=4 SEEDS=1`, compared to the normalized thinking-on base
smoke `/tmp/pi-doctest-thinking-on-smoke.json`.

Result:

- base composite: 0.934375
- H24 composite: 0.85
- delta: -0.084375
- outcome: 1.0
- tested-before-done: 1.0
- tool-call efficiency: 0.50
- mean tool calls: 9.0
- mean thinking chars: 3585.75
- wall clock: 63.20s mean

H24 preserved outcome on the smoke slice but made the model much less
efficient. It doubled down on tool interaction and thinking instead of
compressing the workflow.

## Verdict

Reject H24 for promotion and do not run the larger paired gate.

This is still an important infrastructure/data finding: baseline-mode
`--no-policy-loss` is a viable local throughput route for one-completion
ECHO. The failed result says the data signal is wrong for the current
headroom. Pure environment modeling on one successful trace does not teach the
model to be concise; the next experiment needs action-side discipline, likely
through concise action SFT, a very small oscillating chain, or an ECHO run
seeded from examples whose action trajectories already have low tool count and
low thinking.

No eval task contents or per-example eval transcripts were inspected.
