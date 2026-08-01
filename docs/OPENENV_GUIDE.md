# OpenEnv training

OpenEnv is Kiln's native path from an interactive RL environment to a trained
LoRA. The environment owns state and reward; Kiln owns policy, GRPO, and receipts.
The shortest complete loop is:

```bash
# Terminal 1: serve the model. Stable admits training by default.
KILN_MODEL_PATH=./Qwen3.5-4B ./kiln serve

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

**Training → OpenEnv** persists artifacts, GRPO progress, evaluations,
gates, failures, and cancellation. Restarts resume queued work; interrupted executors fail explicitly.

## How the loop works

For each group, Kiln opens one stateful `WS /ws` session per candidate, resets
identically, samples JSON actions, records outcomes, and submits an `AgenticGroup`.

Candidates share initial messages, environment, reset payload, and seed: the unit
for group-relative advantages. Groups increment seeds and rotate across URLs.

Reset reward is recorded but excluded from return. Step rewards retain their
wire type and map finitely (`null = 0`, `false = 0`, `true = 1`).

## Commands

### Persisted dashboard, CLI, and API runs

All three surfaces share one collector, trainer, evaluator, and artifact store. In `/ui/`, choose
**Training → OpenEnv**. Save the `POST /v1/openenv/runs` body as `openenv-run.json`:

```json
{"kind":"train","idempotency_key":"experiment:counter:17","environment_urls":["http://127.0.0.1:8990"],
 "adapter":"base","output_adapter":"counter-agent","groups":8,"group_size":4,
 "environment_eval":{"groups":20,"group_size":1,
   "gate":{"min_mean_improvement":0.05}}}
```

```bash
kiln openenv start --request openenv-run.json --follow
kiln openenv runs
kiln openenv status 80a26e21-8451-4a64-8666-890c06fd80bd --follow
kiln openenv artifact 80a26e21-8451-4a64-8666-890c06fd80bd environment_eval_receipt --output receipt.json
kiln openenv cancel 80a26e21-8451-4a64-8666-890c06fd80bd
```

Bind each attempt to a non-secret idempotency key: exact retries recover the
retained run; changed reuse fails. Run records v5 keep the cancellable FIFO and
sealed `kiln.openenv-training-contract.v1`. Admission precedes discovery, so
rejection spends no episodes. See the [admission reference](OPENENV_REPLAY_REFERENCE.md#training-admission)
for request, artifact-integrity, capacity, and retention details.

### Protected environments

Protected deployments use exact-origin bearer credentials without persisting
secrets. Align server `credential_ids` or CLI `--credential-env` with URLs; use
`-` for a public slot. See the [authentication boundary](OPENENV_REPLAY_REFERENCE.md#authentication-boundary).

### Inspect

```bash
kiln openenv inspect --environment http://127.0.0.1:8990
kiln openenv inspect --environment http://127.0.0.1:8990 --json
kiln openenv tasks --environment http://127.0.0.1:8990 --split train
```

Inspection checks `/health`, then reads `/metadata`, `/schema`, `/list_environments`, and `/openapi.json`.
It reports the WebSocket URL, client profile, OpenAPI version, schema SHA-256, and a
canonical complete-discovery SHA-256 over all four raw JSON values. Object-key order and whitespace do not matter; unknown extension fields do. Every session repeats the status-only health check before upgrade.

Inspect before collection. Task API 501 means unsupported; catalog rows remain
untrusted because OpenEnv defines no portable row-to-reset mapping.

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
Persisted status, `status --follow`, and dashboard cards expose the published
mean/range, terminal outcomes, recoveries, capacity retries, steps, policy
tokens, and latency without requiring an artifact download.

### Verify and replay

```bash
# Offline: rehash and cross-check the JSONL, replay transcript, and receipt.
kiln openenv verify --summary counter.rollout-summary.json

# Live: verify first, then execute the exact reset/action transcript again.
kiln openenv replay --summary counter.rollout-summary.json
```

`verify` is network-free. `replay` verifies first, then compares live reset,
action, result, and final state. Archive all three files together. See the
[replay reference](OPENENV_REPLAY_REFERENCE.md) for drift and prefix semantics.

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

`--adapter` selects behavior (`base`, `none`, or `null` mean the base model);
`--output-adapter` names the LoRA. Every action retains its content-addressed
base model, inference runtime, and optional adapter revision. Multi-turn
episodes cannot supply one append-stable probability trace, so Kiln uses
`no_importance_correction` and proves the corpus is on-policy instead.

`POST /v1/openenv/training/preflight` seals that policy in the contract and
summary v5; `capacity_reserved: false` means final capacity is rechecked. Drift
during collection or before training fails closed; named behavior adapters are
privately snapshotted under the mutation barrier. See the
[admission reference](OPENENV_REPLAY_REFERENCE.md#training-admission).

Subsequent admission, checkpoints, cancellation, publication, and status are ordinary native GRPO; use `kiln train status`, the dashboard, and `train_receipt.json`.

**Prove it after training** attaches an installed static `post_eval` suite. It
may accompany paired evaluation, but only one gate can own promotion.

### Held-out environment returns

`environment_eval` compares behavior and candidate policies on identical URLs, resets, seeds, candidate indices, generation seeds, and bounds. Training-seed overlap and identity drift fail closed; both sides retain canonical artifacts.

Without `gate`, results are diagnostic. A gate defers loading and requires 20
seed groups, an exact sign-test win, and configured thresholds. Its receipt
binds policies, execution, summaries, evidence, and promotion.

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

The system prompt contains the discovered action JSON Schema. The policy emits
one JSON object; Kiln compiles the schema during inspection and validates before
`step`. Internal references work; external HTTP/filesystem references fail as a
protocol error. A mismatch never contacts `step`: it becomes
`invalid_model_action` with `protocol_error_reward` and bounded
`ACTION_SCHEMA_VALIDATION_FAILED` keyword/JSON-Pointer evidence. Optional
observation `input_text` is pasted verbatim as the model-facing prompt, while
the complete wire observation remains in rollout and replay provenance.
Observations without that optional profile use the complete wire JSON.
Recoverable environment errors remain same-episode feedback.

Thinking is the default for CLI, API, and dashboard OpenEnv runs. Kiln keeps
separated `reasoning_content` and the final answer together as the model's
trainable Action segment. Only the independently parsed, schema-valid final
`content` is sent to `step`; hidden reasoning never changes the environment
action. The action mask starts at the first model-generated reasoning token,
after the template-owned `<think>` opener, and includes the closing delimiter
and final answer. Kiln does not infer a reasoning cutoff or reserve final-answer
tokens. Thinking is unlimited unless the run explicitly sets
`--thinking-budget-tokens N` (or API `thinking_budget_tokens`); an omitted run
budget also disables inherited server-wide token and time limits. Set
`--thinking false` only when deliberately collecting a final-action-only
policy.

If generation ends while the policy is still thinking, there is no OpenEnv
action to score or train. Kiln keeps the unfinished reasoning and the stable
`MODEL_ACTION_NO_OUTPUT` diagnosis in the exact artifacts, but discards that
completion from optimizer input without inventing a `</think>` close. Before
submission, Kiln also skips any group with no remaining usable completion or
with identical remaining rewards, because its group-normalized advantages are
uniformly zero. These decisions are warnings in the summary and persisted run
status; if every group is skipped, the run completes without creating a
training job.

Actions and observations become `TurnKind` segments. GRPO trains complete
reasoning-and-answer actions; ECHO trains observation tokens, excluding
full-warning harness errors. Prompts retain reset and turn history. While the
policy thinks, Kiln pumps Ping/Pong control frames and periodic read-only `state` exchanges,
which also maintains servers that renew resource leases
only for application data. One-step exact-verifier environments—including
eight math families—need no adapter; their text actions flow unchanged end to
end.

An unexpected application message can poison the socket; without request IDs, lock-step cannot resynchronize. Kiln does not resend.

## Identity and artifacts

Each rollout binds discovery, reset, seed, outcome, and exact behavior policy. Native GRPO checks that identity against live model and adapter bytes. JSONL trains; replay retains exchanges; summary binds configuration, statistics, and hashes.

After the final episode, `revalidating` repeats complete discovery. A mismatch fails at `identity_verification` with `environment_identity_changed`; nothing is published. This proves boundary equality, not future URL immutability.

One 512 MiB aggregate retained-representation budget covers live and published forms. Only manifest-declared artifacts download; each request rechecks bytes and SHA-256. Exhaustion publishes no partial bundle: every file must fit and pass before publication.

## Failure and capacity semantics

Recoverable errors become bounded same-socket feedback; terminal errors end the
candidate, and capacity saturation retries a fresh session. The
[recovery reference](OPENENV_REPLAY_REFERENCE.md) lists exact outcomes, codes,
rewards, retry rules, and bounds.

## Security boundary

OpenEnv data is untrusted. Prefer loopback or a private network; remotely, use
HTTPS/WSS and an origin-scoped credential. Inspect schemas, assume prompt
injection, and retain deployment identity before promotion. Authentication
establishes access, not environment integrity. See
[troubleshooting](OPENENV_REPLAY_REFERENCE.md#troubleshooting) for failures.
