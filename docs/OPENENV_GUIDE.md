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

`openenv.rollouts.jsonl` and `openenv.rollout-summary.json` are published
before the training request. Keep them with the adapter's normal
`train_receipt.json`: together they bind the environment-facing rollout and
the optimizer-facing training attempt.

## How the loop works

For each group, Kiln:

1. selects one configured environment;
2. derives one deterministic seed;
3. opens one stateful `WS /ws` session per candidate;
4. resets every candidate with the same seed and reset options;
5. asks the selected Kiln policy for exactly one JSON action;
6. sends that action to the environment and records the observation, tagged
   reward, and `done`;
7. repeats until `done`, `--max-steps`, an invalid model action, or a protocol
   error;
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
  --summary-output counter.rollout-summary.json
```

Use `rollout` to inspect reward variance, compare policy versions, retain a
batch for audit, or submit the JSONL later with `kiln train grpo`. The output
is the same canonical `AgenticGroup` JSONL accepted by the native GRPO route.

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
are not duplicated into every provenance record.

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
become explicit protocol outcomes.

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
name and URL, OpenAPI version, full-environment/action-schema and reset hashes,
seed, steps, episode return, `done`, termination, and an optional protocol-error
code.

This identity participates in Kiln's scored-rollout payload hash. Deserialization
fails closed on an unsupported schema, malformed hash, non-finite return,
inconsistent `done`, or invalid protocol-error state.

The rollout summary records:

- every discovered environment identity and complete typed schema;
- behavior adapter, seeds, sampling controls, concurrency, and reset hash;
- group, rollout, step, model-token, and latency counts;
- return distribution and termination counts;
- compact JSONL byte count and SHA-256; and
- the training submission response for `openenv train`.

The summary is an audit receipt, not proof that an environment implementation
behind the same URL has not changed. Pin the environment image or binary and
retain its content identity alongside serious training runs.

## Failure and capacity semantics

OpenEnv's recoverable errors are `INVALID_JSON`, `UNKNOWN_TYPE`,
`VALIDATION_ERROR`, and `EXECUTION_ERROR`. Its terminal errors are
`CAPACITY_REACHED`, `FACTORY_ERROR`, and `SESSION_ERROR`.

Kiln records protocol error codes and ends the affected candidate with
`--protocol-error-reward` (default `-1`). Invalid model JSON is recorded
separately as `invalid_model_action`; `max_steps` is not mislabeled as
environment `done`.

OpenEnv servers commonly cap active sessions. Keep `--concurrency` at or below
the environment's capacity. Kiln bounds environments, groups, candidates,
steps, action tokens, active sessions, discovery bodies, WebSocket messages,
and the in-memory/inline training corpus; limit failures ask you to reduce the
corresponding dimension.

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

## miniopenenv interoperability

Kiln's reusable `kiln-openenv` crate is tested against the observed OpenEnv
HTTP/1.x protocol and a pinned miniopenenv counter:

```bash
CARGO_BIN="$(command -v cargo)" scripts/check_miniopenenv_interop.sh
```

The check launches the real C99 server and verifies discovery, schema, reset,
two stateful steps (`2`, then `4`), tagged float rewards, terminal `done`, full
WebSocket state, and close. CI runs the same test against the pinned
miniopenenv revision.

Representative miniopenenv environments include `wordle`, `connect4`, `maze`,
`logic`, `bandit`, `blackjack`, `cartpole`, `g2048`, `snake`, and `pong`.
Start with a short horizon and a small batch, inspect return variance and
termination counts, then scale groups.

## Troubleshooting

**Inspection succeeds but reset fails.** The server may have exhausted session capacity or failed to construct an environment. Reduce concurrency and inspect the exact terminal error code.

**Every candidate in a group gets a different initial prompt.** The environment is not deterministic for the supplied reset seed. Kiln rejects the group rather than compute misleading relative advantages.

**Most outcomes are `invalid_model_action`.** Inspect the action schema, reduce thinking, increase `--max-action-tokens` only if output is truncated, and bootstrap the JSON action format with SFT before GRPO.

**Rewards have no variance.** GRPO has no within-group signal. Use harder tasks,
more stochastic policy sampling, a more informative environment reward, or an
SFT/OPD bootstrap. Do not compensate merely by running more identical groups.

**Training is rejected after collection.** The rollout artifacts remain valid.
Check the Kiln serving profile, training preflight, queue state, adapter name,
and memory diagnostics, then submit the JSONL with `kiln train grpo`.

**A remote environment redirects.** Redirects are rejected at the trust boundary. Pass the canonical base URL directly.
