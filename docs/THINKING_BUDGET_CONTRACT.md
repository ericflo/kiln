# Thinking Budget Contract

Kiln's versioned thinking-budget contract lives in:

- [`contracts/thinking-budget-v1.schema.json`](../contracts/thinking-budget-v1.schema.json)
  for request, effective configuration, outcome, result, and recent-request
  wire shapes.
- [`contracts/thinking-budget-v1.conformance.json`](../contracts/thinking-budget-v1.conformance.json)
  for executable resolution and compatibility vectors.

Those files are normative. Rust, browser, desktop, API, eval, and CLI tests
must consume the vectors instead of maintaining independent examples.

## Override Semantics

Token and time dimensions resolve independently:

| Request property | Meaning |
| --- | --- |
| omitted | Inherit that dimension's server default |
| `null` | Explicitly unlimited for that dimension |
| non-negative integer | Finite limit; zero closes immediately |

Editing one dimension must not change the other. For example, a request can
set `thinking_budget_tokens=0` while omitting `thinking_budget_ms`, preserving
the inherited time default.

Finite limits and explicit unlimited values report the surface that supplied
them. Request metadata uses `request`/`request_unlimited`; eval resolution also
uses `suite`, `run_override`, and `example` plus their `_unlimited` forms.
Inherited finite values use `server_default`; an inherited absent default uses
`unlimited`. Unknown legacy or external source text normalizes to `unknown`.

## Result Shape

Effective records contain `configured`, `applied`, optional finite
`max_tokens`/`max_time_ms`, and independent `tokens_source`/`time_source`.
Terminal fields are flat and appear only when a runtime outcome exists:

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

`trigger` is `tokens`, `time`, or `max_tokens`. A natural close has
`triggered=false`, `closed=true`, and measured token/time fields. The shared
reader accepts the earlier eval-only nested `outcome` object, but writers emit
only the flat v1 shape. Recent requests additionally allow `applied` to be
absent when a failure occurred before applicability could be established.

The typed implementation rejects contradictory states, including source/limit
mismatches, an applied unconfigured budget, an outcome on an inert budget, or
`triggered` disagreeing with `trigger` presence.

## Verification

Run the core contract vectors with:

```bash
cargo test -p kiln-core versioned_contract_vectors --lib
```

Cross-runtime smoke checks are expected to execute the same conformance file;
adding a state or source requires updating the schema, vectors, and all
consumers in one checkpoint.
