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
3. `openenv.rollout-summary.json` is the collection receipt. It binds the other
   two files by path, byte count, and SHA-256 digest and records discovered
   schemas, collection controls, statistics, and any training submission.

Move, retain, or archive the three files together. If the receipt's paths no
longer resolve, provide `--dataset` and `--replay` overrides to verification or
live replay.

The summary records every discovered environment identity and complete typed
schema; behavior adapter; seed, sampling, concurrency, recovery, capacity, and
reset controls; group, rollout, step, model-token, and latency counts; return
distribution and outcome counts; artifact identities; and the response from
`openenv train`.

Server-owned runs publish the same three representations below
`<adapter_dir>/.openenv/runs/<run-id>/` and expose them as `dataset`, `replay`,
and `summary` artifact links in `GET /v1/openenv/runs/<run-id>`. The adjacent
`run.json` is the durable orchestration record, not a fourth training
representation. It records lifecycle progress, failure or cancellation, and
the linked native training job. An interrupted active run is marked failed
after restart because a stateful environment session cannot be assumed
resumable.

The summary is an audit receipt, not proof that an implementation behind the
same URL has remained unchanged. Pin an environment image or binary and retain
its deployment identity for serious experiments.

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

## Capacity acquisition

OpenEnv servers may cap active sessions and can send `CAPACITY_REACHED` as the
first WebSocket application frame. That result is terminal for the attempted
socket, not necessarily for the collection. Kiln closes it, waits with bounded
backoff, and opens a fresh session until capacity becomes available or
`--capacity-wait-seconds` expires.

This permits a rollout group to request more concurrent candidates than a
small server admits without silently dropping candidates. Kiln continues to
bound environment count, group count, candidate count, steps, action tokens,
active sessions, discovery response bodies, WebSocket messages, data and replay
artifacts, and the inline training corpus.

Live replay uses the same capacity acquisition rule. Capacity retries are
reported separately from environment transitions and never become fabricated
training observations.

## Protocol conformance oracle

Run the byte-real interoperability gate with:

```bash
CARGO_BIN="$(command -v cargo)" scripts/check_miniopenenv_interop.sh
```

The script pins and builds miniopenenv only as a fast OpenEnv protocol oracle.
It launches its C99 counter, bandit, Connect Four, maze, and Wordle servers and
tests:

- discovery, typed schema identity, and close;
- integer and floating-point rewards;
- object, integer, and string action shapes;
- dynamic legal actions and procedural seeded state;
- recoverable `EXECUTION_ERROR` followed by correction on the same socket; and
- unsolicited terminal `CAPACITY_REACHED` followed by fresh-session
  reacquisition.

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

**Candidates in one group get different initial prompts.** The environment is
not deterministic for the supplied reset seed. Kiln rejects the group instead
of computing misleading relative advantages.

**Most outcomes are `invalid_model_action`.** Inspect the action schema, reduce
thinking, increase `--max-action-tokens` only for truncation, and consider an
SFT bootstrap for the JSON action format.

**Rewards have no variance.** GRPO has no within-group signal. Use harder tasks,
more policy sampling, a more informative reward, or an SFT/OPD bootstrap.

**Training is rejected after collection.** The rollout artifacts remain valid.
Check the serving profile, training preflight, queue, adapter name, and memory
diagnostics, then submit the JSONL with `kiln train grpo`.

**A remote environment redirects.** Redirects are rejected at the trust
boundary. Pass the canonical base URL directly.
