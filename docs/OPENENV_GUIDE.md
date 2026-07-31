# OpenEnv training

OpenEnv is Kiln's native path from an interactive RL environment to a trained
LoRA. The environment owns state and reward; Kiln owns policy, GRPO, and receipts.

The shortest complete loop is:

```bash
# Terminal 1: serve the model with training enabled.
KILN_SERVER_SERVING_PROFILE=experimental KILN_MODEL_PATH=./Qwen3.5-4B ./kiln serve

# Terminal 2: start any OpenEnv-compatible environment on port 8990.
# Use that environment implementation's normal server command.

# Terminal 3: discover it, then collect episodes and train.
./kiln openenv inspect --environment http://127.0.0.1:8990
./kiln openenv train \
  --environment http://127.0.0.1:8990 \
  --groups 8 \
  --group-size 4 \
  --output-adapter counter-agent
```

Keep the pre-training JSONL, replay, and summary with `train_receipt.json`;
held-out evaluation also retains paired bundles and a decision receipt.

**Training → OpenEnv** in `/ui/` persists artifacts, GRPO progress, evaluations,
gates, failures, and cancellation. Restarts fail interrupted work explicitly.

## How the loop works

For each group, Kiln opens one stateful `WS /ws` session per candidate, resets
identically, samples JSON actions, records outcomes, and submits an `AgenticGroup`.

Candidates share initial messages, environment, reset payload, and seed: the unit
for group-relative advantages. Groups increment seeds and rotate across URLs.

Reset reward is recorded but excluded from return. Step rewards retain their
wire type and map finitely (`null = 0`, `false = 0`, `true = 1`).

## Commands

### Dashboard and server API

Dashboard, API, and CLI share the collector, chat handler, artifacts, and GRPO
admission path. In `/ui/`, choose **Training → OpenEnv**. The equivalent API is:

```bash
curl -sS localhost:8420/v1/openenv/runs \
  -H 'content-type: application/json' \
  -d '{"kind":"train","environment_urls":["http://127.0.0.1:8990"],
       "adapter":"base","output_adapter":"counter-agent",
       "groups":8,"group_size":4,"max_steps":8,
       "environment_eval":{"groups":20,"group_size":1,
         "gate":{"min_mean_improvement":0.05}}}'
```

Use `POST /v1/openenv/inspect` for discovery, `GET /v1/openenv/runs` for status, and
`DELETE /v1/openenv/runs/{run_id}` to cancel. Version 3 continues through `training_running`,
optional `post_evaluating`, `environment_evaluating`, and `completed`. Cancellation reaches
the active collector, trainer, or evaluator. Artifacts persist under `<adapter_dir>/.openenv/runs/<run_id>/`.

The CLI exposes the same lifecycle:

```bash
kiln openenv runs
kiln openenv status 80a26e21-8451-4a64-8666-890c06fd80bd --follow
kiln openenv cancel 80a26e21-8451-4a64-8666-890c06fd80bd
```

`status --follow --json` emits one terminal snapshot; human output follows trainer loss, eval accuracy, held-out returns, exact p-values, and gates.

Server runs accept loopback origins by default. `[openenv]` controls remote origins, capacity,
retention, and TTL; each field has a canonical `KILN_OPENENV_*` override. See the
[recovery reference](OPENENV_REPLAY_REFERENCE.md) and [configuration reference](CONFIGURATION.md).

### Protected environments

Protected deployments use exact-origin bearer credentials without putting a secret in a URL,
request body, run record, metric, or artifact. Server and dashboard requests align opaque
`credential_ids` with URLs; CLI commands align `--credential-env`, using `-` for a public slot.
Kiln authenticates discovery and WebSocket upgrade, requires HTTPS/WSS outside loopback, and
records only `none` or `bearer`. See the [authentication and replay reference](OPENENV_REPLAY_REFERENCE.md#authentication-boundary).

### Inspect

```bash
kiln openenv inspect --environment http://127.0.0.1:8990
kiln openenv inspect --environment http://127.0.0.1:8990 --json
```

Inspection checks `/health`, then reads `/metadata`, `/schema`, `/list_environments`, and
`/openapi.json`. It reports the WebSocket URL, client profile, OpenAPI version, and a SHA-256
over the typed action/observation/state schema.

Run inspection before a long collection. It catches an unavailable server, unexpected
discovery document, or changed environment schema without spending model tokens.

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

Use `rollout` to inspect reward variance, compare policies, retain a batch for audit, or submit
the JSONL later with `kiln train grpo`. It is canonical `AgenticGroup` JSONL.

### Verify and replay

```bash
# Offline: rehash and cross-check the JSONL, replay transcript, and receipt.
kiln openenv verify --summary counter.rollout-summary.json

# Live: verify first, then execute the exact reset/action transcript again.
kiln openenv replay --summary counter.rollout-summary.json
```

`verify` is network-free. It checks byte digests and counts, canonical GRPO groups,
fail-closed provenance, replay transcript, returns, and receipt totals.

`replay` verifies offline first, then inspects each live target and compares the captured reset,
action, result, and final state. Move or archive all three files together; use `--dataset` and
`--replay` when paths change. See the [replay and recovery reference](OPENENV_REPLAY_REFERENCE.md)
for verification, drift, and prefix-only semantics.

Protected replay repeats aligned `--credential-env`; it matches auth method but permits rotation.

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

`--adapter` selects the behavior policy; `base`, `none`, and `null` select the base model.
`--output-adapter` names the new LoRA. The trainer receives
`behavior_policy: "no_importance_correction"` because multi-turn chat does not expose exact
per-action token log-probability provenance. Kiln never labels it `recorded`.

Training admission, memory checks, checkpointing, cancellation, atomic adapter
publication, and serving-profile rules are the same as any other native GRPO
job. Use `kiln train status --job-id …`, the dashboard, and the adapter's
`train_receipt.json` normally.

### Held-out environment returns

`environment_eval` runs the behavior and candidate policies with identical URLs, resets, seeds, candidate indices, generation seeds, and bounds. Its default seed range follows training; overlap is rejected. Both sides get canonical dataset, replay, and summary artifacts, and identity drift fails closed.

Without `gate`, results are diagnostic and normal `auto_load` remains. A gate defers loading and requires 20 seed groups, a two-sided exact sign-test win over per-seed means (`p < 0.05`), and configured thresholds. Same-seed replications do not inflate significance. It cannot coexist with `post_eval.min_accuracy`.

The `kiln.openenv-environment-evaluation.v1` receipt binds policy identities, execution provenance, both summary hashes, evidence, decision, and promotion. Status and dashboard show both phases and returns.

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

The system prompt contains the discovered action JSON Schema. At each turn, the
policy must emit one JSON object and no prose. A non-empty observation
`input_text` is foregrounded as optional environment-provided decision text,
while the complete wire observation remains present; it is not a protocol
field or requirement. The environment remains authoritative: recoverable
validation/execution errors are feedback turns on the same episode.

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
model context and corpus. Prefer loopback or a private network; use HTTPS/WSS
with an origin-scoped server credential remotely. Inspect the schema and
implementation, treat prompts and observations as potentially injected, and
retain the summary plus deployment identity before promoting an adapter.
Environment URL credentials, queries, and fragments are rejected. Bearer
response bodies are still untrusted and bounded; authentication establishes
access, not environment integrity.

## Troubleshooting

See [OpenEnv troubleshooting](OPENENV_REPLAY_REFERENCE.md#troubleshooting) for
capacity, recovery, replay drift, reset determinism, action, reward, training,
and redirect failures.
