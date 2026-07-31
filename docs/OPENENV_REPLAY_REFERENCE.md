# OpenEnv replay and recovery reference

This reference defines Kiln's implementation-neutral artifact, verification,
live replay, protocol-error, capacity, and conformance boundaries for OpenEnv
training. The runtime contract is OpenEnv discovery plus a stateful WebSocket
session. It does not depend on any particular environment implementation.

For the normal workflow, start with the [OpenEnv training guide](OPENENV_GUIDE.md).
The machine-readable record types live in
[`contracts/kiln-openenv-v1.schema.json`](../contracts/kiln-openenv-v1.schema.json).

## Artifact bundle

Every successful collection publishes three files:

1. `openenv.rollouts.jsonl` is the canonical `AgenticGroup` corpus consumed by
   native GRPO. Each scored rollout carries fail-closed
   `kiln.openenv-rollout.v1` provenance.
2. `openenv.replay.json` is a `kiln.openenv-replay.v1` manifest. It records each
   group's exact reset payload and reset observation, then every candidate's
   ordered action/result exchanges and final state. It links to the canonical
   JSONL digest instead of creating a second training representation.
3. `openenv.rollout-summary.json` is the collection receipt. Summary v4 binds
   the other files plus the ordered, seed-free per-environment reset plan by
   SHA-256 and records schemas, controls, statistics, any immutable admitted
   training contract, and any submission. Offline verification still accepts
   legacy v2 and v3 receipts.

Move, retain, or archive the three files together. If the receipt's paths no
longer resolve, provide `--dataset` and `--replay` overrides to verification or
live replay.

The summary records every discovered identity and typed schema; behavior
adapter; seed, sampling, concurrency, recovery, capacity, and reset-plan
controls; group, rollout, step, model-token, and latency counts; returns,
outcomes, artifact identities, the exact `kiln.openenv-training-contract.v1`,
and the response from `openenv train`.

Server-owned runs publish the same three representations below
`<adapter_dir>/.openenv/runs/<run-id>/` and expose them as `dataset`, `replay`,
and `summary` artifact links in `GET /v1/openenv/runs/<run-id>`. The adjacent
`run.json` is the durable orchestration record, not a fourth training
representation. It records lifecycle progress, failure or cancellation, the
immutable admitted training contract, and the linked native training job. An interrupted active run is marked failed
after restart because a stateful environment session cannot be assumed
resumable.

The CLI can own this complete persisted lifecycle without reconstructing HTTP
requests or artifact paths:

```bash
kiln openenv start --request openenv-run.json \
  --idempotency-key experiment:counter:17 --follow
kiln openenv artifact <run-id> summary --output openenv.rollout-summary.json
kiln openenv artifact <run-id> environment_eval_receipt \
  --output evidence/environment-evaluation/receipt.json
```

`start` reads exactly one regular, non-symlink JSON object up to 1 MiB and lets
the server validate the authoritative `OpenEnvRunRequest`. `artifact` first
reads current run status and accepts only an exact kind and relative URL from
its manifest. Redirects are disabled, so the Kiln origin cannot be exchanged
for another host. The client requires manifest-matching `Content-Length` and
strong `ETag`, requests identity encoding, requires `private, no-store` and
`nosniff`, then independently caps, counts, and SHA-256 hashes the streamed
body. A same-length mutation still
fails. Bytes are staged beside the destination, synced, and atomically
published only after all checks pass. Existing files are preserved unless
`--force` explicitly requests replacement; failures leave no partial output.
`--json` emits a `kiln.openenv-artifact-download.v1` local receipt without
embedding the configured Kiln base URL or its possible credentials.

### Retry-safe workflow creation

`OpenEnvRunRequest.idempotency_key` is an optional 1..=128-byte opaque token
using ASCII letters, digits, `.`, `_`, `:`, or `-`. A new request returns HTTP
202. While that run remains retained, an exact retry returns its current or
terminal status with HTTP 200 and the original run ID; it creates no second
queue entry, environment session, model request, trainer, or evaluation. The
binding and normalized request survive restart. Concurrent identical POSTs are
atomic, so exactly one creates work.

Normalization occurs through the typed request: input aliases and omitted
defaults therefore compare by effective meaning rather than raw JSON bytes.
Reusing a key with any changed field returns HTTP 409
`openenv_run_idempotency_conflict`. Omitting the key deliberately creates a new
run. Bind one key to one experiment attempt, and never place a credential or
private payload in it because the key persists in `run.json` and public status.

`kiln openenv start --idempotency-key <key>` inserts the key into its bounded
request file and refuses to disagree with an existing body field. The dashboard
generates a UUID and reuses it while an identical submission remains unresolved,
so a lost response can recover the accepted workflow. Idempotency retention is
bounded by the same terminal TTL and `max_tracked_runs` history policy; once a
record is no longer retained, that key can create new work. The fixed-cardinality
`kiln_openenv_runs_total{status="idempotent_replay"}` counter measures recovered
submissions without exposing keys as labels.

A pathname is not publication. The download route accepts only artifact kinds
present in that run's persisted `artifacts` manifest, so staged or partially
written bundles remain unreachable. Publication hashes each regular,
non-symlink file with bounded streaming reads and cross-checks dataset and
replay bytes against the summary. Every download repeats the exact byte-count
and SHA-256 check on one opened descriptor before streaming that same
descriptor. Valid responses carry the manifest byte count as `Content-Length`,
the quoted digest as a strong `ETag`, `private, no-store`, and `nosniff`.
Missing, replaced, oversized, symlinked, truncated, or modified files fail with
`409 openenv_artifact_integrity_failed`; Kiln never serves the drifted bytes.
Restore the original bundle or recollect instead of editing retained artifacts.

The summary is an audit receipt, not proof that an implementation behind the
same URL has remained unchanged. Pin an environment image or binary and retain
its deployment identity for serious experiments.

### Authentication boundary

Environment identity records whether collection used `none` or `bearer`
authentication. It never records the bearer value, API/dashboard credential
handle, or environment-variable name. Server-owned workflows resolve handles
from exact-origin configuration:

```toml
[openenv]
allow_remote_environments = true

[openenv.credentials.production-arcade]
origin = "https://arcade.example.com"
bearer_token_env = "ARCADE_OPENENV_TOKEN"
```

Set `ARCADE_OPENENV_TOKEN` before starting Kiln. API and dashboard requests
align an opaque handle with each environment; use `null` for a public slot:

```json
{
  "environment_urls": ["https://arcade.example.com/openenv"],
  "credential_ids": ["production-arcade"]
}
```

Direct commands resolve an environment-variable name immediately before
client construction:

```bash
ARCADE_OPENENV_TOKEN=... kiln openenv inspect \
  --environment https://arcade.example.com/openenv \
  --credential-env ARCADE_OPENENV_TOKEN

kiln openenv train \
  --environment https://arcade.example.com/openenv \
  --credential-env ARCADE_OPENENV_TOKEN \
  --output-adapter arcade-agent
```

Repeat `--credential-env` in environment order and use `-` for a public slot.
The sensitive header is applied uniformly to `/health`, discovery, Task API, and the
WebSocket upgrade. Remote credentials require HTTPS/WSS; loopback HTTP remains
available for local development. The
`kiln_openenv_authenticated_operations_total{operation="inspect"|"task_catalog"|"run"}` metric
counts protected operations without credential, origin, or token labels.
Authenticated response bodies and episode frames are checked for direct token
reflection before parsing so a peer cannot place the configured credential in
training data or artifacts.

Live replay of an authenticated bundle must provide a credential source in the
same environment position. Replay checks that the resulting authentication
method matches captured identity, but intentionally permits token rotation.
Offline verification needs no credential because secrets are outside the
content-addressed artifact boundary.

```bash
kiln openenv replay --summary protected.rollout-summary.json \
  --credential-env ARCADE_OPENENV_TOKEN
```

### Task API catalogs are discovery, not reset

Dataset-backed OpenEnv servers may expose `GET /{environment}/splits` plus
`POST /{environment}/{tasks|num_tasks|task|task_range}`. HTTP 501 on a
provider-backed route is a conforming declaration that the environment has no
TaskProvider; it does not disable seeded rollout or training.

```bash
# List split names and their provider-authored types.
kiln openenv tasks --environment http://127.0.0.1:8990

# Read a bounded page without fetching the provider's complete task list.
kiln openenv tasks --environment http://127.0.0.1:8990 \
  --split train --start 100 --limit 25 --json
```

The dashboard exposes the same catalog beside protocol inspection. Server
clients use `POST /v1/openenv/tasks` with `environment_urls`, optional aligned
`credential_ids`, optional `environment_name` and `split`, and `start`/`limit`.
The response schema is `kiln.openenv-task-catalog.v1`.

Task rows are arbitrary, untrusted JSON. Kiln preserves open split types and
provider fields, percent-encodes advertised environment names as one URL
segment, limits discovery bodies to 2 MiB, limits raw client collections to
16,384 items, and limits CLI/API/dashboard pages to 200 rows. The server metric
`kiln_openenv_task_catalog_inspections_total{status="started"|"completed"|"failed"}`
has fixed-cardinality labels.

OpenEnv defines no operation that selects a Task API row for a WebSocket
session, and a provider's `reset` need not consult its task catalog. Kiln
therefore never copies a row into reset data, run records, replay manifests, or
training receipts. Portable scheduling remains the explicit reset object plus
Kiln's deterministic group seed. If an environment defines its own row/reset
convention, bind it explicitly through reset options; that resulting reset
payload is then hashed and replayed under the normal artifact contract.

### Paired held-out evaluation bundle

A server train run with `environment_eval` adds:

```text
environment-evaluation/
  baseline/{rollouts.jsonl,replay.json,summary.json}
  candidate/{rollouts.jsonl,replay.json,summary.json}
  receipt.json
```

Both sides use the production collector with identical URL rotation, reset
payloads, held-out seeds, candidate indices, generation seeds, and bounds.
Before comparison, Kiln requires identical discovered environment identities,
reset payloads and observations, group/seed identities, and candidate indices.
Drift fails the run instead of producing an unpaired estimate.

`receipt.json` is `kiln.openenv-environment-evaluation.v1`. It binds exact
baseline and candidate adapter content revisions, execution provenance, both
summary SHA-256 values, mean returns, improved/regressed/tied counts, the
two-sided exact sign-test result, fixed policy version, decision, and promotion
outcome. The run status publishes every file with its own content digest.
Run `kiln openenv verify` against each summary to validate its underlying
dataset/replay bundle independently.

Evaluation seeds cannot overlap the training interval. A promotion gate needs
at least 20 paired seed groups and significant per-seed mean-return improvement
at `p < 0.05`, plus configured point thresholds. Replications within one seed
do not inflate significance. Rejection or inconclusive evidence leaves the
candidate unserved; `auto_load=false` records a passed candidate as kept.

## Offline verification

```bash
kiln openenv verify --summary openenv.rollout-summary.json
```

Verification is network-free and bounded. It:

- checks the byte count and SHA-256 digest of the JSONL and replay manifest;
- parses every canonical GRPO group and fail-closed OpenEnv provenance record;
- validates the replay manifest and its link to the canonical dataset;
- recomputes each episode return from tagged rewards and configured protocol
  error penalties; and
- cross-checks group, candidate, step, environment, seed, termination, and
  artifact totals against the summary.

An unsupported schema, malformed hash, non-finite return, inconsistent `done`,
invalid protocol-error state, mismatched transcript, or inconsistent receipt
causes verification to fail. A successful check emits a
`kiln.openenv-verification.v1` report; it does not contact the environment or
claim that its current behavior is unchanged.

## Exact live replay

```bash
kiln openenv replay --summary openenv.rollout-summary.json
```

Live replay first performs the complete offline verification. It then:

1. discovers every live target and requires the captured environment name and
   schema identity;
2. opens fresh capacity-aware sessions;
3. sends the exact effective reset objects and captured actions; and
4. compares reset observations, observations, tagged rewards, `done`, protocol
   errors, and captured final state exactly.

The result is a `kiln.openenv-replay-run.v1` report. Any mismatch is observable
environment drift. Equal schema identity alone is insufficient because task
data, randomness, or implementation behavior may have changed without a schema
change.

If collection ended because the model produced malformed JSON, only the
environment prefix can be replayed: that malformed policy generation was
never sent on the OpenEnv wire. The replay report counts such prefix-only
candidates explicitly.

Reset options can include arbitrary task payloads. Kiln hashes them into each
rollout and retains the exact effective object once per replay group because
exact replay requires it. Treat replay files as sensitive when reset tasks
contain private data.

## Protocol-error outcomes

The OpenEnv recoverable error codes are:

- `INVALID_JSON`
- `UNKNOWN_TYPE`
- `VALIDATION_ERROR`
- `EXECUTION_ERROR`

Kiln turns a recoverable error into a complete observation feedback turn on the
same WebSocket. The policy can correct its next action without losing episode
state. Each error contributes `--protocol-error-reward` (default `-1`) to the
episode return. A candidate may use at most `--max-recoverable-errors`
(default `3`); the next recoverable error terminates it as `protocol_error`.

The terminal error codes are:

- `CAPACITY_REACHED`
- `FACTORY_ERROR`
- `SESSION_ERROR`

`FACTORY_ERROR` and `SESSION_ERROR` end the candidate. Invalid model JSON is
recorded separately as `invalid_model_action`, while exhaustion of the horizon
is `max_steps`; neither is mislabeled as environment `done`.

OpenEnv has no request IDs or episode resume. Kiln therefore pumps only
Ping/Pong control frames while policy inference is pending and rejects any
unsolicited application message. A timeout, socket failure, binary, malformed
or oversized response, credential reflection, or wrong response type
permanently poisons the session. The action is not resent and a late response
can never be consumed as the answer to a later action.

## Capacity acquisition

Kiln's persisted control plane first applies its own bounded FIFO admission.
At most `openenv.max_active_runs` complete workflows execute; additional valid
runs remain cancellable and position-visible until a slot opens, with total
active, queued, and retained records exported as fixed-cardinality metrics.
The queue consumes no OpenEnv session or model capacity. `max_tracked_runs` is
the hard combined bound for active, queued, and retained terminal records.
Operators can alert on `kiln_openenv_runs_active`,
`kiln_openenv_runs_queued`, and `kiln_openenv_runs_tracked`; cumulative
admission and restart counts use
`kiln_openenv_runs_total{status="admitted"|"resumed"}`, and
`kiln_openenv_run_queue_wait_seconds_total` records aggregate queue delay.
The v5 admission sequence is stable; on restart, entries that never acquired a
slot resume in that exact FIFO order. A pristine queued v4 entry is first
materialized once into a v5 contract; it is never recomputed after that
migration. Any admitted non-terminal workflow fails
explicitly because an external
episode, trainer, or evaluator cannot be assumed resumable.

OpenEnv servers may cap active sessions and can send `CAPACITY_REACHED` as the
first WebSocket application frame. That result is terminal for the attempted
socket, not necessarily for the collection. Kiln closes it, waits with bounded
backoff, and opens a fresh session until capacity becomes available or
`--capacity-wait-seconds` expires.

This permits a rollout group to request more concurrent candidates than a
small server admits without silently dropping candidates. Kiln continues to
bound environment count, group count, candidate count, steps, action tokens,
active sessions, discovery response bodies, WebSocket messages, data and replay
artifacts, the summary, and the inline training corpus.

Those independent limits are not deferred until serialization. One 512 MiB
aggregate retained-representation budget is charged when reset data and each
action, observation, error, and final state arrive. Completed candidates are
moved—not cloned—into their group, replay, and receipt projections, and the
budget is reconciled after each compaction. Reset-option files are rejected
from metadata before a large read, JSONL hashing streams into SHA-256, and the
replay encoder refuses the write that would cross its 256 MiB artifact cap.
Budget exhaustion aborts collection before dataset, replay, or summary is
published; reduce group size, concurrency, steps, or environment payload size.

Live replay uses the same capacity acquisition rule. Capacity retries are
reported separately from environment transitions and never become fabricated
training observations.

## Training admission

For every training request, Kiln materializes the exact native-GRPO
configuration before persistence or direct environment contact.
It overrides rollout-owned `behavior_policy`, `base_adapter`, `output_name`,
and `auto_load`, then validates the environment-token loss and policy contract,
LoRA scale, checkpoint interval, behavior-adapter layout, installed
`post_eval` suite, serving profile, backend GRPO workload, optimizer, and model
rank ceiling. Persisted `kind=train` does this before creating its run
directory and atomically stores the result as
`kiln.openenv-training-contract.v1` in v5 status. Collection, restart, trainer
submission, static evaluation, CLI, and dashboard all consume that exact
contract rather than reapplying current defaults. Direct `kiln openenv train` first calls
`POST /v1/openenv/training/preflight`, validates the v1 receipt, and later
submits its exact `effective_config` and returned optional `post_eval`;
rejection contacts no environment and writes no artifacts. Both paths increment
`kiln_openenv_training_preflights_total{status="accepted"|"rejected"}`;
persisted rejection also increments
`kiln_openenv_runs_total{status="training_preflight_rejected"}`. Rollout-only
requests reject `output_adapter`, `training_config`, `post_eval`, and
`environment_eval` instead of ignoring them.

Summary v4 embeds the same contract before the dataset/replay/summary bundle is
first published. It therefore remains auditable if final native queue or memory
admission rejects after collection. A summary containing a training submission
without its contract fails schema and offline verification; legacy v2/v3
receipts may not claim the new field.

Direct preflight returns the exact current queue/tracked-job snapshot with
`capacity_reserved: false`. The final native GRPO admission repeats immutable
checks after collection and adds time-varying queue and live-memory capacity.
This second gate remains authoritative because capacity can change while
episodes run.

### Retained trainer evidence

A completed persisted train run is self-contained. Its
`training.training_data.openenv` field carries the admitted
`kiln.openenv-training-data.v1` lineage: exact corpus and ordered task-plan
digests, endpoints and schema identities, seed range, steps, rewards, and
termination counts. Clients do not need to join the run to `/v1/train/jobs` to
identify what trained the adapter.

Completion also adds `train_receipt` and `adapter_manifest` to the run's
ordinary manifest-gated `artifacts` array. Kiln validates the native receipt,
requires successful status and the requested adapter name, binds both files to
the admission corpus and semantic OpenEnv lineage, verifies the manifest's
receipt hash, then copies the exact bytes atomically into the run directory.
Each source file has a 4 MiB limit and must be a regular non-symlink file. The
run-owned evidence therefore survives adapter lifecycle operations and every
download keeps the same length, ETag, SHA-256, and same-origin checks:

```bash
kiln openenv artifact <run-id> train_receipt --output evidence/train_receipt.json
kiln openenv artifact <run-id> adapter_manifest --output evidence/adapter_manifest.json
```

The dashboard's **Prove it after training** control emits ordinary
`post_eval`. The named suite must already be installed; Kiln follows adapter
and optional baseline jobs to terminal outcomes. `train-set-eval` is diagnostic
and cannot set `min_accuracy`. Static evaluation can accompany paired
`environment_eval`, but a workflow has one automatic promotion owner:
`post_eval.min_accuracy` and `environment_eval.gate` are mutually exclusive.

## Protocol conformance oracle

Run the byte-real interoperability gate with:

```bash
CARGO_BIN="$(command -v cargo)" scripts/check_miniopenenv_interop.sh
```

The script pins and rebuilds miniopenenv only as a fast OpenEnv protocol oracle.
It launches its C99 counter and all twenty-two text-profiled environment
servers—the original fourteen arcade environments plus eight text-first math
families—then tests:

- discovery, typed schema identity, and close;
- the optional `input_text` profile across every environment in the matrix, without
  making that downstream convention a protocol requirement;
- schema-discovered answer strings, deterministic seeded prompts, recoverable
  wrong-type actions, exact integer rewards, and frozen post-`done`
  observations across all eight one-step math environments;
- integer and floating-point rewards;
- object, integer, and string action shapes;
- dynamic legal actions and procedural seeded state;
- recoverable `EXECUTION_ERROR` followed by correction on the same socket; and
- unsolicited terminal `CAPACITY_REACHED` followed by fresh-session
  reacquisition; and
- paired held-out baseline/candidate collection whose environment-owned bandit
  returns drive the exact sign-test gate.

The end-to-end lane collects two candidates against a bandit limited to one
session, forces a semantic action error and corrective turn, submits the
canonical groups to a fake Kiln training API, verifies all three artifacts, and
replays every environment exchange.

Miniopenenv is deliberately confined to this test-oracle boundary. The gate
rejects miniopenenv-named production environment variables, and production Rust
sources contain no miniopenenv-specific runtime branch. The same client and
training path accepts every implementation that satisfies the OpenEnv protocol.

## Troubleshooting

**Collection times out waiting for capacity.** Another client or a long
candidate held all sessions beyond `--capacity-wait-seconds`. Reduce
concurrency, increase that bounded wait, or raise the environment's session
limit.

**A recoverable error ends the candidate.** The candidate spent
`--max-recoverable-errors`, or the code was terminal. Inspect the trajectory and
receipt counters before changing the budget.

**Replay reports drift.** Confirm that the same environment build, task data,
and reset semantics are deployed at the captured URL. Stable schema identity
does not imply stable behavior.

**A protected environment returns 401 or rejects the WebSocket upgrade.**
Confirm that the configured credential origin exactly matches the URL's
scheme/host/port, the named environment variable is present and non-empty in
the Kiln process, and replay received one aligned `--credential-env`. Remote
bearer credentials require HTTPS/WSS; redirects are never followed.

**Candidates in one group get different initial prompts.** The environment is
not deterministic for the supplied reset seed. Kiln rejects the group instead
of computing misleading relative advantages.

**Most outcomes are `invalid_model_action`.** Inspect the action schema, reduce
thinking, increase `--max-action-tokens` only for truncation, and consider an
SFT bootstrap for the JSON action format.

**Rewards have no variance.** GRPO has no within-group signal. Use harder tasks,
more policy sampling, a more informative reward, or an SFT/OPD bootstrap.

**Training is rejected.** Persisted and direct workflows preflight immutable
failures before collection; correct the returned config, adapter, suite,
backend, or optimizer error. A direct CLI collection can still encounter a
time-varying queue or memory rejection at final native admission because the
preflight snapshot is not a reservation; its rollout artifacts remain valid
and can be resubmitted with `kiln train grpo`.

**A remote environment redirects.** Redirects are rejected at the trust
boundary. Pass the canonical base URL directly.
