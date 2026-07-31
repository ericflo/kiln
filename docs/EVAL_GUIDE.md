# Eval guide

Kiln evaluates the base model and LoRA adapters through the same in-process
inference path used for serving. A suite defines prompts, targets, generation
settings, and scoring. A run records raw completions, one reduced outcome per
example, aggregate metrics, and the provenance needed to compare or replay it.

Use the generated [eval and judgment schema](../contracts/kiln-evals-v1.schema.json)
for field types, defaults, bounds, and response shapes. Use the generated
[HTTP API contract](../contracts/kiln-http-api-v1.openapi.json) for methods,
paths, status codes, media types, and errors.

All examples assume a server at `http://localhost:8420`. The browser UI exposes
the same workflow under **Evals → Datasets / Suites / Jobs / Judgments**.

## Serving profile

Normal eval and training use the default profile:

| Task | Profile |
| --- | --- |
| Evaluate the base model or any saved adapter | `stable` |
| Train, load the resulting unmerged adapter, and evaluate it in one process | `stable` |
| Drain inference and train without eval traffic | `maintenance` |

`maintenance` disables inference, so it cannot run evals. After
maintenance-mode training, restart in `stable` to load and evaluate the saved
unmerged adapter. `experimental` is reserved for backend development and is
not required for this workflow. See
[Serving profiles](SERVING_PROFILES.md).

## Choose a workflow

| Starting point | Recommended path |
| --- | --- |
| A small, hand-authored test set | Write a suite JSON file, register it, then run or compare it. |
| An SFT conversation dataset | Upload it, inspect its split, and synthesize a suite from `holdout`. |
| Recorded tool-using traffic | Convert request logs or external traces with `kiln-eval trace-suite`. |
| Human A/B preferences | Record judgments, reserve a holdout, compile SFT data, train a judge adapter, then validate it. |
| A completed run that must reproduce exactly | Use strict replay, not a failure rerun. |

## Five-minute tour

Create `smoke.json`:

```json
{
  "name": "math-smoke",
  "description": "Three arithmetic checks",
  "schema_version": 2,
  "default_scorer": {
    "kind": "numeric_tolerance",
    "atol": 0,
    "rtol": 0,
    "integer_only": true
  },
  "generation": {
    "temperature": 0,
    "top_p": 1,
    "top_k": 0,
    "max_tokens": 32,
    "n": 1
  },
  "aggregation": {"kind": "single"},
  "examples": [
    {
      "id": "add",
      "messages": [{"role": "user", "content": "47 + 138?"}],
      "target": "185"
    },
    {
      "id": "multiply",
      "messages": [{"role": "user", "content": "23 × 17?"}],
      "target": "391"
    },
    {
      "id": "subtract",
      "messages": [{"role": "user", "content": "1024 - 376?"}],
      "target": "648"
    }
  ]
}
```

Register and run it:

```bash
kiln-eval register --file smoke.json
kiln-eval run \
  --suite math-smoke \
  --adapter my-trained-adapter \
  --seed 42 \
  --watch
```

Pass `--adapter ""` to evaluate the base model. Omit `--adapter` to use the
currently active adapter. Add `--json` when automation needs the complete job
rather than the human summary.

## Build a suite from a dataset

Upload an SFT JSONL corpus:

```bash
curl --fail-with-body -sS \
  -F name=my-sft \
  -F format=sft_chat \
  -F file=@my-sft.jsonl \
  http://localhost:8420/v1/eval/datasets/upload
```

Kiln persists deterministic `train`, `validation`, and `holdout` views. Check
their actual counts: normalized duplicates and rows linked by declared
`group_id` or `session_id` stay together, so row percentages need not exactly
match the configured percentages.

Dataset preview and synthesis select `holdout` by default. The available
strategies answer different questions:

| Strategy | Prompt and target |
| --- | --- |
| `final_assistant` | Prompt through the final user turn; target the last assistant turn. |
| `first_assistant_turn` | Initial system/user context; target the first assistant turn. |
| `every_assistant_turn` | Create one example for every assistant turn. |
| `tool_call_predict` | Keep assistant tool-call turns and target canonical tool-call JSON. |
| `tool_response_followup` | End the prompt after tool results; target the next assistant turn. |
| `end_of_trajectory_answer` | Include the agentic trajectory through the last tool result; target the closing answer. |

Preview before saving:

```bash
curl --fail-with-body -sS -X POST \
  http://localhost:8420/v1/eval/datasets/my-sft/preview \
  -H 'content-type: application/json' \
  -d '{
    "suite_name": "my-sft-holdout",
    "source_split": "holdout",
    "strategy": "final_assistant",
    "head_n": 5
  }'
```

Then synthesize:

```bash
curl --fail-with-body -sS -X POST \
  http://localhost:8420/v1/eval/datasets/my-sft/synthesize \
  -H 'content-type: application/json' \
  -d '{
    "suite_name": "my-sft-holdout",
    "source_split": "holdout",
    "strategy": "final_assistant",
    "scorer": {"kind": "auto_detect"},
    "force": true
  }'
```

Synthesis from `train` is available for diagnostics, but it does not make the
result held out. The [dataset split guide](DATASET_SPLITS.md) defines identity,
contamination, migration, and training-provenance behavior.

## Author a suite

A schema-v2 suite contains:

| Field | Purpose |
| --- | --- |
| `name` | Stable registry name. |
| `description` | Human explanation of the suite's purpose and provenance. |
| `default_scorer` | Scorer for examples without an override. |
| `generation` | Suite-wide decode settings. |
| `aggregation` | Reduction from completions to one outcome per example. |
| `system_prompt` | Optional system message added when an example does not already start with one. |
| `tools` | Optional shared tool catalogue. |
| `examples` | Prompts, targets, tags, weights, metadata, and overrides. |

An example may override `scorer`, `generation`, or `tools`. Its `tags` drive
metric slices, `metadata` is copied into results, and `weight` affects only
`weighted_mean_score`. A zero weight keeps the example visible without
contributing to that weighted metric.

Give every case a stable, unique `id`. If omitted, Kiln derives one from the
messages, target, and aliases. Empty or duplicate resolved IDs are rejected
because comparisons, seeds, reruns, and aggregate rows all depend on identity.

Two file layouts are supported:

1. A single JSON document containing the suite and `examples`.
2. A suite header without inline examples plus `examples.jsonl`, with one
   example per non-empty line.

The registry stores suites under `<eval_root>/suites/<name>/`. The default
eval root is `<adapter_dir>/.eval`; `eval.eval_dir` relocates that shared root
for suites, datasets, and judgments.

## Multi-completion evaluation

### The unit of evidence

One example is one independent observation. Asking for five completions from
the same prompt does not create five independent examples. Kiln therefore
keeps:

- `outcomes`: every raw completion, with its completion index, derived seed,
  text, score, timing, tokens, reasoning, and budget evidence;
- `aggregated_outcomes`: exactly one reduced record per example.

Headline accuracy, confidence intervals, tag/tool/scorer slices, comparisons,
failure reruns, regression tests, and promotion gates use
`aggregated_outcomes`. Latency, token totals, reasoning length, and format
diagnostics use raw `outcomes`.

### Reducers

For schema v2, every effective `generation.n` must equal the aggregation's
cardinality. `k` must be from 1 through 128.

| Aggregation | Meaning |
| --- | --- |
| `{"kind":"single"}` | Require `n=1`; use completion 0. |
| `{"kind":"mean_at_k","k":K}` | Average all K scores; pass when the mean is at least `0.5`. |
| `{"kind":"pass_at_k","k":K}` | Pass when any observed completion passes. This is observed any-success, not a population estimator. |
| `{"kind":"majority_at_k","k":K}` | Pass when more than half pass. K must be odd. |

Use `mean_at_k` for continuous or partial-credit scoring, `pass_at_k` for
bounded search success, and `majority_at_k` for sampled consistency. Use a
nonzero temperature when genuinely different samples are intended.

The registry and executor reject mismatched `n`/K, even `majority_at_k`, an
incomplete completion group, duplicate or missing indices, non-finite scores,
and inconsistent group metadata. Schema-v1 suites remain readable only for
the unambiguous `single`/`n=1` case. Do not relabel an old multi-completion
result as v2; rerun it.

## Run, compare, and rerun

Run one target:

```bash
kiln-eval run --suite math-smoke --adapter v1 --seed 42 --watch
```

Compare several targets under the same suite and job seed:

```bash
kiln-eval compare \
  --suite math-smoke \
  --adapter "" \
  --adapter v1 \
  --adapter v2 \
  --seed 42 \
  --watch
```

The compare API accepts at most eight adapters. Pairing uses stable example
IDs, not result order.

`POST /v1/eval/jobs/{job_id}/rerun` creates a diagnostic run from selected
reduced outcomes. By default it includes `fail`, `invalid`, and `error`,
retains the original effective seed, and uses the original target. The body
may set `adapter`, `outcome_kinds`, `include_pass`, or a replacement `seed`.
A rerun does not claim byte reproduction; it reruns the complete selected
examples with their original reducer.

## Interpret a result

Read these fields first:

| Field | Interpretation |
| --- | --- |
| `metrics.num_examples` | Reduced independent examples. |
| `metrics.num_completions` | Raw generations across those examples. |
| `metrics.accuracy` | Passing reduced examples divided by reduced examples. |
| `metrics.accuracy_confidence_interval` | 95% Wilson interval for that pass rate. |
| `metrics.mean_score` | Unweighted mean reduced score. |
| `metrics.weighted_mean_score` | Mean reduced score after example weights. |
| `metrics.tag_breakdown` | Per-tag counts, pass rate, and interval. |
| `metrics.by_scorer` | Per-scorer counts, mean score, and pass rate. |
| `metrics.pass_rate_by_tool` | Per-tool counts, pass rate, and interval for tool-call suites. |
| `metrics.latency` and token totals | Runtime work measured from raw completions. |

The confidence interval describes sampling uncertainty under the suite's
sampling assumptions. It does not correct selection bias, leakage, mislabeled
targets, correlated examples, or an unrepresentative suite.

Each raw `outcomes[]` row includes `example_id`, `completion_index`,
decimal-string `generation_seed`, raw and scoring-form completion text,
`kind`, `score`, `detail`, tags, metadata, tokens, latency, reasoning, and
thinking-budget evidence. Each reduced `aggregated_outcomes[]` row records its
representative completion and the raw pass/fail/invalid/error counts.

## Scorer reference

Every scorer is a JSON object with `kind`.

| Kind | Use |
| --- | --- |
| `exact_match` | Text equality with explicit case and whitespace settings. |
| `contains` | Require any, all, or none of a phrase list. |
| `regex` | Match a pattern or compare one capture group with the target. |
| `json_validity` | Require parseable JSON, optional object shape and JSON Pointer paths, and optional structural target equality. |
| `multiple_choice` | Reduce common answer forms to a declared label. |
| `numeric_tolerance` | Compare the last number in the completion with one target number. |
| `llm_judge` | Ask the base model or a named local judge adapter for a score. |
| `tool_call` | Compare normalized tool names and arguments. |
| `code` | Compare extracted code by presence, exact text, token similarity, or line coverage. |
| `python_doctest` | Execute generated Python against doctests in the user prompt. |
| `all` | Require every nested scorer; use their mean score. |
| `any` | Accept any nested scorer; use their maximum score. |

### Text, JSON, and numbers

`exact_match` defaults to case-sensitive comparison with surrounding
whitespace stripped. State both options when the policy matters:

```json
{"kind":"exact_match","case_sensitive":false,"strip_whitespace":true}
```

`contains` defaults to case-sensitive matching and `mode: "any"`:

```json
{
  "kind": "contains",
  "phrases": ["Paris", "France"],
  "mode": "all",
  "case_sensitive": false
}
```

`numeric_tolerance` requires exactly one number in the target and uses the
last number found in the completion:

```json
{"kind":"numeric_tolerance","atol":0.001,"rtol":0.01,"integer_only":false}
```

It passes when `|prediction - target| <= atol + rtol × |target|`.

`json_validity` can require object output and paths:

```json
{
  "kind": "json_validity",
  "require_object": true,
  "required_paths": ["/tool_call/name", "/tool_call/arguments"]
}
```

If the example also has a target, both sides must parse and compare
structurally; object-key order does not matter.

### Tool calls

`tool_call` normalizes Qwen3.5 XML, canonical `{"tool_calls":[...]}` JSON,
OpenAI `function.arguments`, inline JSON, and fenced tool-call blocks before
scoring:

```json
{
  "kind": "tool_call",
  "name_match": {"kind": "case_insensitive"},
  "args": {"kind": "auto"}
}
```

Argument modes are:

- `auto`: choose structural, numeric, prose, or command-aware comparison from
  each value and key;
- `structural`: require canonical JSON equality;
- `keys_only`: compare only argument keys;
- `per_key`: assign a nested scorer to individual arguments.

Command-like arguments receive additional shell-wrapper and inline-code
analysis. That makes equivalent wrapping less important while retaining
action differences such as `git push` versus `git pull`.

Set `require_xml_format: true` only when native Qwen XML is part of the
contract. Leave it false when semantic tool selection and arguments are what
matter.

### Code and executable scoring

The `code` scorer does not run the completion. It supports:

- `any_block`;
- `exact_block` with optional comment stripping;
- `token_similarity` (default Jaccard threshold `0.6`);
- `line_coverage` (default threshold `0.7`).

Example:

```json
{
  "kind": "code",
  "language": "python",
  "style": {"kind": "token_similarity", "min_jaccard": 0.8}
}
```

`python_doctest` does execute model-produced Python in a
`python3 -I -S` subprocess with a wall-clock timeout:

```json
{"kind":"python_doctest","timeout_seconds":5}
```

Python's isolated flags and a timeout are not a security sandbox. Run
untrusted code only inside an operating-system or container sandbox with
appropriate filesystem, process, and network isolation.

### Local judge

`llm_judge` renders `{question}`, `{target}`, and `{answer}` into a prompt,
then parses a score from the judge response:

```json
{
  "kind": "llm_judge",
  "judge_adapter": "judge-v1",
  "template": "Question: {question}\nReference: {target}\nAnswer: {answer}\nScore from 0 to 1.",
  "score_regex": "(?i)score[:\\s]*([01](?:\\.\\d+)?)"
}
```

Omit `judge_adapter` to use the base model. A judge is another model-based
measurement, not ground truth: validate it on held-out human labels and audit
its disagreement slices.

## Post-training auto-eval

SFT and GRPO requests may attach:

```json
{
  "post_eval": {
    "suite": "math-smoke",
    "data_scope": "held-out",
    "include_baseline": true
  }
}
```

`data_scope` defaults to `held-out`. Before queue publication, Kiln rejects
detected exact, normalized, source-row, group, or session overlap with the
admitted training data. Use `"data_scope":"train-set-eval"` only for an
intentional training-data diagnostic. That diagnostic may overlap but cannot
set `min_accuracy`.

After training, Kiln queues the adapter eval and optionally a standalone
baseline result, then records IDs in `linked_eval_job_ids`.

### Automatic promotion policy

Adding `post_eval.min_accuracy` creates a fail-closed promotion gate. The
versioned `paired_wilson_v1` policy compares the candidate with the adapter
active before training, or with the base model if none was active.

For an ordinary accuracy gate:

1. Both arms must have matching suite and generation hashes, aggregation,
   example count, and unique example IDs.
2. At least 20 reduced, paired examples are required.
3. Kiln runs a two-sided exact binomial sign test on per-example score changes
   at `alpha = 0.05`.
4. The candidate must show significant paired improvement and its 95% Wilson
   lower bound must reach `min_accuracy`.

Twenty examples is only a minimum. Even 20 passes from 20 examples has a
Wilson lower bound of about `0.84`, so it cannot prove a `0.90` floor.

| Outcome | Meaning and action |
| --- | --- |
| `promoted` | Evidence passed and deferred auto-load succeeded. |
| `kept` | Evidence passed, but auto-load was not requested. |
| `regression` | The candidate was significantly worse; it remains for investigation and is not promoted. |
| `demoted` | The Wilson upper bound was below the floor; Kiln removes it from serving/default state and renames it with a `.failed` suffix. |
| `inconclusive` | Evidence could not pass or conclusively fail; the adapter remains on disk and is not promoted. |
| `error` | Evaluation, evidence validation, archival, or adapter transition failed; the adapter is not promoted. |

The gate uses one versioned held-out suite. It does not average several suites
or let one passing suite erase another result. Put required domains in one
suite, version that composition, and use tags for slices. Training status and
history expose `gate_outcome`, `post_eval_verdict`, and
`post_eval_gate_evidence`.

## The judgment flywheel

Kiln can turn human A/B preferences into an SFT dataset for a local judge
adapter:

1. Create a judgment dataset under **Evals → Judgments**.
2. Generate two responses and record A, B, tie, or skip, plus optional notes
   and tags.
3. Compile the dataset. Kiln excludes the most recent holdout rows and emits
   each training judgment in both A/B orientations.
4. Train the compiled `sft_chat` dataset through `/v1/train/sft`.
5. Validate the resulting adapter against the reserved judgment rows.
6. Use the adapter as `judge_adapter` only after reviewing held-out accuracy
   and orientation-bias failures.

If `holdout_n` is omitted during compilation, Kiln reserves
`min(20, floor(rows / 5))`. Small datasets may therefore reserve zero rows and
return a warning. Compilation fails when the chosen holdout leaves no training
rows. Validation also warns when its requested rows overlap the last compiled
training range.

Deleting a bad judgment and retraining removes that row from the new training
corpus. It does not “unlearn” the row from an already trained adapter, and it
does not by itself prove that a replacement adapter is unaffected by related
examples.

This path does not call an external frontier judge. It uses the local base
model or a local adapter through Kiln's evaluator.

## Production trace tool-call evals

### Mine your own request log

Request logging is enabled by default. Kiln records chat, completion, and batch
inference requests under `<adapter_dir>/.requests/`, unless
`request_log.enabled` is false or the logger could not initialize. The active
file is `requests-current.jsonl`; rotated files are
`requests-<timestamp>.jsonl.gz` when compression is enabled.

The log is bounded and intentionally does not block inference:

- bodies beyond `request_log.max_capture_bytes` are truncated in the log;
- a full logging channel drops rows and increments its drop count;
- retention deletes the oldest rotated files after `max_total_bytes`;
- interrupted streams are marked and may lack a final thinking-budget outcome.

Treat the log as sensitive. Prompts, responses, tool arguments, and user-agent
data may contain secrets or personal data. Apply access controls, retention,
redaction, and consent rules appropriate to the deployment.

On systems with GNU `zcat -f`, a simple tool-call extraction is:

```bash
zcat -f /path/to/adapters/.requests/requests-*.jsonl* \
  | jq -c '
      select(.status == 200 and .route == "/v1/chat/completions")
      | {
          messages: (.request.messages + [.response.choices[0].message]),
          tools: .request.tools
        }
    ' > mined-traces.jsonl

kiln-eval trace-suite \
  --input mined-traces.jsonl \
  --format openai_jsonl \
  --output mined-tool-calls.json \
  --stats-output mined-tool-calls.report.json \
  --suite-name mined-tool-calls \
  --max-examples 1000 \
  --seed 42
```

Verify local `zcat` behavior before relying on `-f`; otherwise decompress
rotated files and concatenate the active JSONL explicitly.

### Import external traces

`kiln-eval trace-suite` accepts:

- `prompt_chosen_jsonl`;
- `openai_jsonl`;
- `openai_trajectory_jsonl`;
- `anthropic_jsonl`;
- `anthropic_trajectory_jsonl`;
- `auto`, which honors explicit per-row format labels and otherwise inspects
  each row.

Trajectory formats create one candidate per eligible assistant tool-call
turn. Rows without a current-turn tool call are skipped. Reservoir sampling
is global across repeated `--input` flags. Exact deduplication is off by
default because repeat frequency can be part of the workload distribution;
enable it explicitly with `--dedupe`.

Keep `--stats-output` with the suite. It records sampling settings, the
effective seed, parse and skip counts, format/tool histograms, and source
provenance for retained examples. Review this sidecar before interpreting a
score as representative production evidence.

## Qwen3.5 chat and tool behavior

### Thinking

Kiln separates Qwen reasoning from final content before ordinary scoring.
Raw decoder text and `reasoning_text` remain in outcomes. Aggregate metrics
report reasoning length and unclosed thinking blocks.

Control thinking per suite or example:

```json
{
  "generation": {
    "chat_template_kwargs": {"enable_thinking": true},
    "thinking_budget_tokens": 96,
    "thinking_budget_ms": 2000
  }
}
```

An omitted budget inherits the server setting, `null` means unlimited for that
dimension, and `0` requests immediate closure. If both token and time limits
are active, the first reached wins. See the
[thinking-budget contract](THINKING_BUDGET_CONTRACT.md).

### Tools

Put a shared OpenAI-shaped tool catalogue on the suite:

```json
{
  "name": "weather-agent",
  "schema_version": 2,
  "default_scorer": {"kind": "tool_call"},
  "generation": {"temperature": 0, "max_tokens": 128, "n": 1},
  "aggregation": {"kind": "single"},
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Return weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "examples": [
    {
      "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
      "target": "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]}"
    }
  ]
}
```

The executor renders these tools through the model's chat template.
Per-example `tools` replace the suite catalogue. The scorer can compare
canonical JSON targets with Qwen XML predictions.

### Built-in suite

Kiln installs `qwen3.5-agentic-core` at startup. Its 24 hand-authored examples
cover no-tool answers, tool selection and arguments, refusals, follow-ups,
thinking, code, JSON, multiple choice, and numeric output:

```bash
kiln-eval run \
  --suite qwen3.5-agentic-core \
  --adapter my-trained-adapter \
  --watch
```

It is a smoke/regression suite, not evidence that an application-specific
adapter is production-ready.

## Seed and provenance

### Seed contract

Every job receives one immutable `effective_seed` before queue publication.
Resolution order is:

1. top-level run/compare `seed`;
2. run-level `generation.seed`;
3. suite-level `generation.seed`;
4. a generated random seed.

Kiln derives a per-example/per-completion seed through
`kiln.eval-seed.v1` using the base seed, stable example ID, and completion
index. Compare arms therefore receive the same derived seeds. Seed values are
emitted as decimal strings where the response must preserve the full `u64`
for JavaScript clients.

A shared seed makes sampling inputs paired and auditable. It does not promise
the same bytes across different binaries, drivers, devices, kernels, precision
policies, model revisions, or nondeterministic operations.

### Base-weight provenance

Production admission snapshots `kiln.base-weight-shards.v1`, including every
resident safetensors shard identity. Jobs, terminal archives, results, CLI
JSON, and the dashboard preserve it. See
[base-weight provenance](BASE_WEIGHT_PROVENANCE.md).

### Execution provenance

Production admission also snapshots `kiln.execution-provenance.v1`: bounded
backend/device/runtime evidence, executable and optional source identity,
model/tokenizer/template identities, precision policy, compiled features, and
effective configuration/environment digests. It proves the integrity of the
declared envelope, not driver correctness. See
[execution provenance](EXECUTION_PROVENANCE.md).

## Strict byte replay

A newly completed production run carries a self-validating
`kiln.eval-replay.v1` record. It binds the exact suite, effective generation
state and seed, scorer configurations, base and adapter identities, thinking
budgets, execution provenance, and every raw decoder continuation.

Run:

```bash
kiln-eval replay --job eval_123
kiln-eval replay --job eval_compare_123 --run-index 1 --json
```

Replay admission fails before queue visibility if the source record is
incomplete or invalid, or if current execution, base weights, or required
adapter identities differ. Terminal status is:

| Status | Meaning |
| --- | --- |
| `matched` | The replay record and every raw decoder continuation matched byte for byte. |
| `mismatch` | Execution completed, but at least one bound identity or raw continuation differed. |
| `error` | Kiln could not produce a complete valid replay record. |

The CLI exits nonzero for mismatch, error, failed execution, or a missing
verdict. A match proves reproduction only for the identities declared by that
record. It is not a general claim about other environments.

## CLI

Set `KILN_SERVER_URL` or pass the global `--server` option for another server.

```bash
kiln-eval list
kiln-eval register --file smoke.json --force
kiln-eval run --suite math-smoke --adapter v1 --seed 42 --watch
kiln-eval run --file smoke.json --adapter v1 --watch --json
kiln-eval compare --suite math-smoke --adapter "" --adapter v1 --seed 42 --watch
kiln-eval probe --prompt "1+1?" --target 2 --scorer numeric --adapter v1 --seed 42
kiln-eval trace-suite --help
kiln-eval panel-suite --help
kiln-eval replay --job eval_123 --run-index 0 --json
```

`probe` wraps one example in an inline suite for debugging. `panel-suite`
builds a bounded weighted panel from an existing suite. Inspect each command's
`--help`; the CLI is the normative source for command-line flags.

## HTTP API reference

| Method and path | Purpose |
| --- | --- |
| `POST /v1/eval/suites` | Register a suite; use `?force=true` to replace one. |
| `GET /v1/eval/suites` | List suite summaries. |
| `GET /v1/eval/suites/{name}` | Fetch a complete suite. |
| `DELETE /v1/eval/suites/{name}` | Delete a suite. |
| `POST /v1/eval/run` | Queue a registered or inline suite against one target. |
| `POST /v1/eval/compare` | Queue one registered suite against up to eight targets. |
| `GET /v1/eval/jobs` | List tracked and retained jobs. |
| `GET /v1/eval/jobs/{job_id}` | Read progress, results, provenance, and replay fields. |
| `POST /v1/eval/jobs/{job_id}/rerun` | Rerun selected reduced outcome kinds. |
| `POST /v1/eval/jobs/{job_id}/replay` | Queue strict replay for one completed run. |
| `DELETE /v1/eval/jobs/{job_id}` | Cancel queued/running work; delete a terminal job and its archive. |
| `POST /v1/eval/datasets/upload` | Register dataset content. |
| `GET /v1/eval/datasets/{name}/split` | Inspect persisted row assignments. |
| `PUT /v1/eval/datasets/{name}/split` | Replace split settings and assignments. |
| `POST /v1/eval/datasets/{name}/preview` | Preview suite synthesis. |
| `POST /v1/eval/datasets/{name}/synthesize` | Save a synthesized suite. |
| `POST /v1/judgments/{name}/compile` | Compile judgment rows into an SFT dataset. |
| `POST /v1/judgments/{name}/validate` | Queue held-out validation for a judge adapter. |

For request and response fields, use the generated contracts linked at the top
of this guide.

## What an eval does not prove

An eval result is evidence about a declared suite, scorer, generation policy,
model state, and execution envelope. It is not proof that:

- the suite represents production traffic;
- its labels or scorer are correct;
- examples are independent;
- no semantic or pretraining leakage exists;
- a model is safe outside the measured domains;
- a point estimate will hold after deployment;
- a seed alone makes execution reproducible.

Keep suite source, synthesis/trace sidecars, split identities, job/result
archives, and provenance records together. Review failures and slice counts,
not only headline accuracy. Use an external test corpus and independent
human review for high-stakes release decisions.
