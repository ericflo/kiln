# OpenEnv training
OpenEnv is Kiln's native path from an interactive reinforcement-learning
environment to a trained LoRA adapter. The environment owns the task, state
transition, observation, terminal signal, and reward. Kiln owns policy
generation, seed-matched sampling, canonical multi-turn trajectories, grouped
GRPO, ECHO supervision, receipts, and adapter publication.

The shortest complete loop is:

```bash
# Terminal 1: serve the model with training enabled.
KILN_SERVER_SERVING_PROFILE=experimental \
KILN_MODEL_PATH=./Qwen3.5-4B \
./kiln serve

# Terminal 2: serve any OpenEnv environment.
../miniopenenv/build/rel/bin/counter --host 127.0.0.1 --port 8990

# Terminal 3: discover it, then collect episodes and train.
./kiln openenv inspect --environment http://127.0.0.1:8990
./kiln openenv train \
  --environment http://127.0.0.1:8990 \
  --groups 8 \
  --group-size 4 \
  --output-adapter counter-agent
```

`openenv.rollouts.jsonl`, `openenv.replay.json`, and
`openenv.rollout-summary.json` are published before the training request. Keep
all three with the adapter's normal `train_receipt.json`: together they bind
the environment-facing rollout, its exact executable transcript, and the
optimizer-facing training attempt.

## How the loop works

For each group, Kiln:

1. selects one configured environment;
2. derives one deterministic seed;
3. opens one stateful `WS /ws` session per candidate;
4. resets every candidate with the same seed and reset options;
5. asks the selected Kiln policy for exactly one JSON action;
6. sends that action to the environment and records the observation, tagged
   reward, and `done`;
7. feeds recoverable protocol errors back to the policy on the same socket and
   repeats until `done`, `--max-steps`, an invalid model action, a terminal
   protocol error, or the configured recovery budget is exhausted;
8. sums step rewards into the episode return; and
9. writes all candidates as one `AgenticGroup`, then submits the groups to the
   ordinary native GRPO trainer when `train` was selected.

Candidates in a group always share the same initial messages, environment,
reset payload, and seed. That is the comparison unit for group-relative
advantages. Different groups increment the seed and may be assigned
round-robin across multiple `--environment` URLs.

Reset rewards are preserved in the reset observation but do not contribute to
episode return: reset is not a transition. Step rewards retain OpenEnv's wire
type (`null`, boolean, integer, or float) in the trajectory and map to a finite
training scalar (`null = 0`, `false = 0`, `true = 1`).

## Commands

### Inspect

```bash
kiln openenv inspect --environment http://127.0.0.1:8990
kiln openenv inspect --environment http://127.0.0.1:8990 --json
```

Inspection checks `/health`, then reads `/metadata`, `/schema`,
`/list_environments`, and `/openapi.json`. It reports the derived WebSocket
URL, OpenEnv client profile, OpenAPI version when present, and a SHA-256 over
the typed action/observation/state schema.

Run inspection before a long collection. It catches an unavailable server,
unexpected discovery document, or changed environment schema without spending
model tokens.

### Collect without training

```bash
kiln openenv rollout \
  --environment http://127.0.0.1:8990 \
  --groups 16 \
  --group-size 4 \
  --seed-start 1000 \
  --output counter.rollouts.jsonl \
  --replay-output counter.replay.json \
  --summary-output counter.rollout-summary.json
```

Use `rollout` to inspect reward variance, compare policy versions, retain a
batch for audit, or submit the JSONL later with `kiln train grpo`. The output
is the same canonical `AgenticGroup` JSONL accepted by the native GRPO route.

### Verify and replay

```bash
# Offline: rehash and cross-check the JSONL, replay transcript, and receipt.
kiln openenv verify --summary counter.rollout-summary.json

# Live: verify first, then execute the exact reset/action transcript again.
kiln openenv replay --summary counter.rollout-summary.json
```

`verify` is network-free. It checks both byte digests and byte counts, parses
the canonical GRPO groups, fail-closed provenance, replay transcript, returns,
and receipt totals.

`replay` performs that offline verification first. It then inspects each live
target and compares the captured reset, action, result, and final-state
transcript exactly. Move or archive the three files together; use `--dataset`
and `--replay` when their recorded paths have changed. See the
[replay and recovery reference](OPENENV_REPLAY_REFERENCE.md) for the complete
verification, drift, and prefix-only semantics.


### Collect and train

```bash
kiln openenv train \
  --environment http://127.0.0.1:8990 \
  --adapter counter-agent-v1 \
  --output-adapter counter-agent-v2 \
  --groups 32 \
  --group-size 8 \
  --lora-rank 16
```

`--adapter` selects the behavior policy used to act. `base`, `none`, and
`null` explicitly select the base model. `--output-adapter` names the new LoRA.
The trainer receives `behavior_policy: "no_importance_correction"` because the
current multi-turn chat path does not expose exact per-action token
log-probability provenance. Kiln never labels these rollouts as the separate
`recorded` policy contract.

Training admission, memory checks, checkpointing, cancellation, atomic adapter
publication, and serving-profile rules are the same as any other native GRPO
job. Use `kiln train status --job-id …`, the dashboard, and the adapter's
`train_receipt.json` normally.

## Reset tasks and multiple environments

Pass environment-specific reset options in a JSON object such as
`{"difficulty":"hard","split":"train"}`, then add
`--reset-options wordle-reset.json` to `rollout` or `train`.

Kiln always overwrites the object's `seed` with the group seed. The exact
effective reset object is hashed into each rollout; arbitrary task payloads
are not duplicated into every provenance record. It is retained once per
replay group because exact replay is impossible without it. Treat replay files
as potentially sensitive when reset tasks contain private data.

Multiple environments are first-class: repeat
`--environment http://127.0.0.1:PORT` on the same command.

Whole groups—not individual candidates—are distributed round-robin. Relative
advantages therefore never compare rewards from different environments.
Reward scale still matters across optimizer groups; normalize reward semantics
inside environments when mixing tasks with radically different ranges.

## Actions, observations, and ECHO

The system prompt contains the discovered action JSON Schema. At each turn,
the policy must emit one JSON object and no prose. Kiln currently accepts a
bare object or an otherwise exact JSON object inside one Markdown JSON fence.
The environment remains authoritative: its validation and execution errors
become explicit protocol outcomes. Recoverable errors are full feedback turns,
so a policy can correct its next action without losing episode state.

Every sampled action is a `TurnKind::Action`. Every environment observation is
a `TurnKind::Observation`. Native GRPO therefore applies policy-gradient loss
to model actions and ECHO's environment cross-entropy to observation tokens by
default. Harness-generated error observations carry a full warning prefix so
the default warning filter does not teach the model to imitate Kiln's
diagnostic prose.

The prompt for turn N contains the reset prompt plus every prior action and
observation. One WebSocket connection remains open for the whole episode. Do
not substitute OpenEnv's stateless HTTP `/reset` and `/step` routes: they cannot
represent an episode.

## Identity and artifacts

Each scored rollout may carry `kiln.openenv-rollout.v1` provenance: environment
name and URL, schema and reset hashes, seed, steps, return, termination, and an
optional protocol-error code. This identity participates in the scored-rollout
payload hash and fails closed when malformed.

The JSONL is the canonical trainer input. The replay manifest retains exact
environment exchanges, and the summary records configuration, statistics,
content hashes, and any training submission. The receipt cannot prove that
code behind a URL stayed fixed; pin serious environment deployments. The
[replay and recovery reference](OPENENV_REPLAY_REFERENCE.md) defines the
artifact and drift boundary in detail.

## Failure and capacity semantics

Kiln assigns every protocol error `--protocol-error-reward` (default `-1`).
Recoverable errors become observation feedback and the policy may try again on
the same socket up to `--max-recoverable-errors` (default `3`). A terminal
error, or the next recoverable error after that budget is spent, ends the
candidate as `protocol_error`.

On `CAPACITY_REACHED`, Kiln closes that socket and retries a fresh session with
bounded backoff until `--capacity-wait-seconds` expires. Invalid model JSON and
`max_steps` remain distinct outcomes. The
[replay and recovery reference](OPENENV_REPLAY_REFERENCE.md) lists every
recoverable and terminal code, retry rule, and resource bound.

## Security boundary

OpenEnv actions and observations are untrusted external data that enter the
model context and training corpus.

- Prefer loopback or a private network. Kiln does not add an authentication
  extension to the OpenEnv protocol.
- Use HTTPS/WSS for a remote environment and terminate authentication in a
  trusted gateway.
- Inspect the action schema and environment implementation before training.
- Credentials, query strings, and fragments in an environment URL are
  rejected.
- Treat environment prompts and observations as capable of prompt injection.
  The action schema constrains syntax, not intent.
- Retain the rollout summary and environment deployment identity before
  accepting a trained adapter into a higher-trust setting.

## Protocol interoperability oracle

Kiln's reusable `kiln-openenv` crate and native training loop are tested
against the OpenEnv HTTP/1.x protocol. Its byte-real oracle happens to use
miniopenenv for speed, but production Kiln has no implementation-specific type,
setting, command, or branch: every compatible `--environment` follows the same
runtime path. The
[replay and recovery reference](OPENENV_REPLAY_REFERENCE.md) documents the
five-environment gate.

## Troubleshooting

See [OpenEnv troubleshooting](OPENENV_REPLAY_REFERENCE.md#troubleshooting) for
capacity, recovery, replay drift, reset determinism, action, reward, training,
and redirect failures.
