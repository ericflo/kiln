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

Use `POST /v1/openenv/inspect` for discovery, `POST /v1/openenv/tasks` for task catalogs, `GET /v1/openenv/runs` for status, and `DELETE /v1/openenv/runs/{run_id}` to cancel.
Version 3 continues through `training_running`, `post_evaluating`, `environment_evaluating`, and `completed`. Cancellation reaches collector, trainer, or evaluator; artifacts persist under `<adapter_dir>/.openenv/runs/<run_id>/`.

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
kiln openenv tasks --environment http://127.0.0.1:8990 --split train
```

Inspection checks `/health`, then reads `/metadata`, `/schema`, `/list_environments`, and
`/openapi.json`. It reports the WebSocket URL, client profile, OpenAPI version, and a SHA-256
over the typed action/observation/state schema.

Run inspection before a long collection. It catches unavailable or changed servers without
spending model tokens. `tasks` reports conforming 501 as unsupported and otherwise pages
arbitrary provider rows; OpenEnv defines no row-to-reset mapping, so Kiln never invents one.

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

Use `--reset-options task.json` when every environment shares a reset object. For a portfolio, repeat an aligned object with each URL:

```bash
kiln openenv rollout \
  --environment http://127.0.0.1:8000 --environment-reset-options arcade.json \
  --environment http://127.0.0.1:8001 --environment-reset-options math.json
```

Use `-` for an empty slot. API and dashboard runs send `environment_reset_options` as one object per `environment_urls` entry. Kiln
removes a caller-supplied `seed`, inserts the group seed, hashes each effective
reset into rollout provenance, and retains it in replay. Summary v3 binds the
ordered seed-free plan; verification reconstructs it. Every configured
endpoint must run, so groups cannot be fewer than environments. Treat private
reset tasks and replay files as sensitive.

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

Actions and observations become their corresponding `TurnKind` segments.
Native GRPO applies policy-gradient loss to actions and ECHO cross-entropy to
observation tokens. Full-warning harness errors stay outside ECHO imitation.
The prompt retains reset and turn history. Kiln pumps Ping/Pong control frames
while the policy thinks; stateless HTTP routes cannot preserve the episode.
One-step exact-verifier environments, including eight math families, need no adapter. Their string actions, `input_text`, integer rewards, and terminal observations flow unchanged through collection, GRPO, verification, replay, and held-out evaluation.

## Identity and artifacts

Each scored rollout may carry `kiln.openenv-rollout.v1` provenance: environment
name and URL, schema and reset hashes, seed, steps, return, termination, and an
optional protocol-error code. This identity participates in the scored-rollout
payload hash and fails closed when malformed.

JSONL trains; replay retains exact exchanges; summary binds configuration,
statistics, hashes, and submission. Collection charges each turn against a
512 MiB aggregate retained-representation budget. Reset files are prebounded;
dataset, replay, and summary each stay under 256 MiB.
Exhaustion publishes no partial bundle. Pin URL deployments. See the
[replay and recovery reference](OPENENV_REPLAY_REFERENCE.md) for artifact and drift boundaries.

## Failure and capacity semantics

Kiln assigns every protocol error `--protocol-error-reward` (default `-1`).
Recoverable errors become observation feedback and the policy may try again on
the same socket up to `--max-recoverable-errors` (default `3`). A terminal
error, or the next recoverable error after that budget is spent, ends the
candidate as `protocol_error`.

On `CAPACITY_REACHED`, Kiln closes that socket and retries a fresh session with
bounded backoff until `--capacity-wait-seconds` expires. Invalid model JSON and
`max_steps` remain distinct outcomes. Timeouts, unsolicited or unreadable
frames, transport failures, and wrong response types poison the socket
permanently; lock-step cannot resynchronize. The
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
