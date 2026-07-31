# Thinking budgets

A thinking budget limits an open model reasoning block without disabling
thinking or ending the completion. When a finite limit is reached, Kiln writes
the tokenizer's complete closing `</think>` sequence into model history and
then continues generating the visible answer.

Thinking budgets do not bound queue time, prompt prefill, network delivery, or
the entire response. Use request `max_tokens` and ordinary client/server
timeouts for those limits.

## Set a request budget

Chat and batch requests accept independent token and decode-time dimensions:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve this carefully."
    }
  ],
  "max_tokens": 512,
  "thinking_budget_tokens": 128,
  "thinking_budget_ms": 3000
}
```

Each request field has three distinct states:

| Request value | Result for that dimension |
|---|---|
| Property omitted | Inherit the corresponding server default |
| `null` | Explicitly unlimited, even when the server has a finite default |
| Non-negative integer | Use that finite limit; `0` starts closure immediately |

The two dimensions resolve independently. Setting
`thinking_budget_tokens: 0` while omitting `thinking_budget_ms` preserves the
server's time default. When both finite limits are active, the first one
reached wins. If both are reached at the same token boundary, the token trigger
is reported.

## Set server defaults

Omitted defaults are unlimited:

```toml
[server]
default_thinking_budget_tokens = 512
default_thinking_budget_ms = 5000
```

The equivalent environment variables are:

```bash
KILN_SERVER_DEFAULT_THINKING_BUDGET_TOKENS=512
KILN_SERVER_DEFAULT_THINKING_BUDGET_MS=5000
```

Environment values accept a non-negative base-10 integer or case-insensitive
`unlimited`. Malformed values stop startup. Use `GET /v1/config` or
`kiln config --json` to inspect the resolved defaults.

The browser Playground reads the same resolved configuration and previews
which token and time values will come from the server or request.

## Know when a budget applies

A finite budget is `configured` when at least one dimension resolves to a
finite limit. It is `applied` only when:

- the rendered prompt starts generation inside a reasoning block;
- at least one finite limit is configured; and
- the request permits at least one completion token.

A budget does not turn thinking on. If the template or request starts in
non-thinking mode, the configured budget is inert and the response reports
`applied: false`.

`kiln rollout-generate` fails closed on this distinction. Budget flags and
top-level request-template budget fields require `--thinking true`; otherwise
the command rejects them before sending a request or creating output.

## Understand enforcement

The token limit counts generated thinking tokens. The time clock starts when
Kiln evaluates the first decode candidate, after queueing and prompt prefill,
and is checked between token candidates.

When a limit is reached:

1. Kiln compares the sampled candidate with the next expected close-sequence
   token.
2. If the model is already closing naturally, the sampled token wins and the
   outcome remains untriggered.
3. Otherwise Kiln replaces sampled candidates with the complete tokenizer
   close sequence.
4. Those close tokens enter KV history and count toward completion usage and
   request `max_tokens`.
5. Ordinary answer generation resumes after the reasoning block closes.

Kiln also reserves enough remaining completion slots for the close sequence.
If that reservation becomes the first active limit, the outcome trigger is
`max_tokens`. A thinking budget larger than the completion limit is therefore
valid: the completion boundary can close reasoning first.

Before decode, the effective completion limit after context-window clamping
must fit the tokenizer's full close sequence. Kiln rejects a smaller limit.
It also rejects a `stop` string that can match, contain, or overlap any part of
the close sequence, because stopping there could leave model history inside an
unterminated reasoning block. Reserve additional completion tokens when you
want visible answer text after closure.

Time-budgeted requests bypass deterministic completion caches because their
boundary depends on runtime speed. Token-only budgets remain cacheable under a
budget-aware cache key.

## Read configuration and outcome fields

The effective configuration fields are:

| Field | Meaning |
|---|---|
| `configured` | At least one finite token or time limit exists |
| `applied` | The request began in reasoning and the finite budget controller ran |
| `max_tokens` | Effective finite thinking-token limit; absent when unlimited |
| `max_time_ms` | Effective finite decode-time limit; absent when unlimited |
| `tokens_source` | Provenance of the token dimension |
| `time_source` | Provenance of the time dimension |

When runtime outcome exists, the flat record also carries:

| Field | Meaning |
|---|---|
| `triggered` | Whether Kiln forced closure because a limit was reached |
| `trigger` | `tokens`, `time`, or `max_tokens`; absent for natural closure |
| `closed` | Whether the complete closing sequence entered model history |
| `thinking_tokens` | Thinking tokens produced before closure or request end |
| `thinking_time_ms` | Decode time observed from the first candidate to closure or request end |

A natural close reports `triggered: false` and `closed: true`. A triggered
record can report `closed: false` if the request ends before the full sequence
is written. Do not treat `triggered` as proof of successful closure.

Example terminal record:

```json
{
  "configured": true,
  "applied": true,
  "max_tokens": 64,
  "tokens_source": "request",
  "time_source": "request_unlimited",
  "triggered": true,
  "trigger": "tokens",
  "closed": true,
  "thinking_tokens": 64,
  "thinking_time_ms": 800
}
```

## Find the record on each API surface

| Surface | Configuration and outcome location |
|---|---|
| Non-streaming chat | Effective record in `metadata.thinking_budget`; completion outcome in `choices[].thinking_budget` |
| Streaming chat | Final record in `metadata.thinking_budget` on the chunk containing `finish_reason` |
| Batch completion | Request-wide effective configuration in `metadata.thinking_budget`; separate outcome in each `completions[].thinking_budget` |
| Recent requests | Effective record and any known outcome in each row's `thinking_budget` |

In an SSE stream, the finish chunk appears before an optional usage chunk and
the final `[DONE]` sentinel. Batch results remain completion-specific because
different completions can close naturally or trigger different boundaries.
Kiln does not manufacture one aggregate batch outcome.

Recent-request outcome fields remain absent when a configured budget was inert.
`applied` is absent only when a failure occurred before Kiln could determine
applicability.

The shared reader accepts a retired eval-only nested `outcome` object for
compatibility. Current writers emit only the flat version 1 shape.

## Interpret provenance

Token and time sources are recorded independently:

| Source | Meaning |
|---|---|
| `server_default` | The request inherited a finite server default |
| `unlimited` | The request inherited an absent server default |
| `request` / `request_unlimited` | A request supplied a finite limit / explicit unlimited |
| `suite` / `suite_unlimited` | An eval suite supplied a finite limit / explicit unlimited |
| `run_override` / `run_override_unlimited` | An eval run override supplied a finite limit / explicit unlimited |
| `example` / `example_unlimited` | An eval example supplied a finite limit / explicit unlimited |
| `unknown` | A legacy or external source string was not recognized |

## Metrics

`GET /metrics` publishes fixed-cardinality budget telemetry:

- `kiln_thinking_budget_source_total{dimension,source}` counts token and time
  provenance;
- `kiln_thinking_budget_outcomes_total{outcome}` uses exactly `unconfigured`,
  `inert`, `natural_close`, `tokens`, `time`, `max_tokens`, `unclosed`,
  `interrupted`, and `unresolved`; and
- `kiln_thinking_budget_effective_tokens` and
  `kiln_thinking_budget_effective_seconds` histogram finite limits.

Request-supplied numeric limits never become labels.

## Normative contract

The versioned authorities are:

- [`contracts/thinking-budget-v1.schema.json`](../contracts/thinking-budget-v1.schema.json)
  for request, effective, outcome, terminal, and recent-request wire shapes;
- [`contracts/thinking-budget-v1.conformance.json`](../contracts/thinking-budget-v1.conformance.json)
  for executable resolution, server-default, valid-record, legacy, and
  invalid-record vectors.

Rust, browser, desktop, API, eval, and CLI checks consume these vectors instead
of maintaining separate semantics.

<!-- thinking-budget-contract-v1:generated:start -->
- Contract version: `1`
- Request override states: `inherit`, `unlimited`, `limit`
- Source vocabulary: `unlimited`, `server_default`, `request`, `request_unlimited`, `suite`, `suite_unlimited`, `run_override`, `run_override_unlimited`, `example`, `example_unlimited`, `unknown`
- Trigger vocabulary: `tokens`, `time`, `max_tokens`
<!-- thinking-budget-contract-v1:generated:end -->

Run the core vectors with:

```bash
cargo test -p kiln-core versioned_contract_vectors --lib
```

Any new state, source, trigger, or wire field must update the schema,
conformance vectors, documentation, and every consumer in the same change.
