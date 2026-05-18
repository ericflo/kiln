# Eval Guide

Kiln ships a first-class eval system for measuring how well LoRA adapters
perform on tasks that matter to you. Evals run in the same process as
inference, share the same model weights, and slot into the same job-queue
machinery as training — so you can `train → eval → compare` in one HTTP
loop without touching Python.

Three things are unusual about kiln's eval stack:

1. **You don't write a scorer.** Auto-detect picks one per example based on
   the *shape of the target* — numbers go to `numeric_tolerance`, JSON-shaped
   targets to `json_validity`, tool calls to `tool_call`, code blocks to
   `code`, and free-form text to a contains-with-key-phrases scorer.
2. **You don't have to start from scratch.** Upload an SFT JSONL and Kiln
   *synthesizes* an eval suite from it — picking the right decomposition
   strategy for your data (single-turn Q&A, every-step in an agent run,
   tool-call prediction, etc.) and writing the suite to disk for you.
3. **You don't need a frontier LLM as the judge.** The flywheel turns
   user A/B picks into a local SFT dataset; that dataset trains a *local*
   judge LoRA; the judge LoRA scores future evals via `LlmJudge`. Nothing
   leaves your machine.

Everything in this guide assumes a Kiln server running on
`http://localhost:8420`. The `/ui` dashboard has a dedicated Evals tab with
Datasets / Suites / Jobs / Judgments sub-tabs that drive every endpoint
documented here.

## Five-minute tour

1. **Author a suite.** A suite is a JSON document with a name, a default
   scorer, generation params, and a list of examples. Example
   (`smoke.json`):

   ```json
   {
     "name": "math-smoke",
     "description": "Three arithmetic problems",
     "default_scorer": { "kind": "numeric_tolerance", "integer_only": true },
     "generation": { "temperature": 0.0, "max_tokens": 32 },
     "examples": [
       { "messages": [{"role": "user", "content": "47 + 138?"}], "target": "185" },
       { "messages": [{"role": "user", "content": "23 * 17?"}],  "target": "391" },
       { "messages": [{"role": "user", "content": "1024 - 376?"}], "target": "648" }
     ]
   }
   ```

2. **Register it.**

   ```bash
   curl -X POST http://localhost:8420/v1/eval/suites \
        -H 'content-type: application/json' \
        -d @smoke.json
   ```

   or via the CLI:

   ```bash
   kiln-eval register --file smoke.json
   ```

3. **Run it.**

   ```bash
   kiln-eval run --suite math-smoke --adapter my-trained-adapter --watch
   ```

   The CLI prints a per-suite summary when the job lands:

   ```
   Suite: math-smoke | Adapter: my-trained-adapter
     accuracy:  66.7%  |  mean: 0.667  |  weighted: 0.667
     pass:2  fail:1  invalid:0  error:0  (n=3)
     latency ms: p50=412  p90=601  p99=601  mean=478
   ```

## The 60-second onramp: Dataset → Suite → Eval

Most users never hand-author a suite. The canonical flow is:

1. **Upload an SFT JSONL.** Drag your training data (the same `messages: [...]`
   JSONL you'd POST to `/v1/train/sft`) into the **Evals → Datasets → Upload**
   panel, or POST it via:

   ```bash
   curl -F name=my-sft -F format=sft_chat -F file=@my-sft.jsonl \
        http://localhost:8420/v1/eval/datasets/upload
   ```

   Kiln scans the file and reports row counts, role patterns, tool-call
   density — useful signals for picking a synthesis strategy.

2. **Synthesize a suite.** Pick a *strategy* (`final_assistant`,
   `first_assistant_turn`, `every_assistant_turn`, `tool_call_predict`) and
   click **Preview 5 examples** to sanity-check before committing. Then
   **Save & Run vs active** synthesizes the full suite and queues an eval
   against your currently-loaded adapter in one click.

   Strategies in plain English:
   - **`final_assistant`** — Prompt = everything up to the last user turn;
     target = the very last assistant turn. Tests end-to-end correctness on
     Q&A and on the *final answer* of an agent run.
   - **`first_assistant_turn`** — Prompt = system + first user; target =
     first assistant. Cheap, fast, tests immediate response only.
   - **`every_assistant_turn`** — One example per assistant turn. Useful for
     "next-action prediction" evals across long agent trajectories.
   - **`tool_call_predict`** — Only keeps assistant turns that emit tool
     calls. Targets are canonicalized `{"tool_calls":[…]}` JSON; the
     synthesized scorer is `tool_call` with auto args scoring.

3. **Stash + re-run.** Suites live in `<adapter_dir>/.eval/suites/<name>/`
   on disk. Re-run the same suite against any adapter in 1 click.

## The judgment flywheel (training a local judge LoRA, no frontier LLM)

The eval system also captures *user preferences* into a separate kind of
dataset, then compiles them into SFT data for training a local judge LoRA.
The judge LoRA grades future evals via `Scorer::LlmJudge { judge_adapter }`.

End-to-end loop:

1. **Create a judgment dataset.** Evals → Judgments → Create.
2. **Generate pairs.** Enter a prompt, pick two adapters (or two
   temperatures of the same adapter), click **Generate pair**.
3. **Pick a winner.** A / Tie / B / Skip. Optionally add a note + tags.
   Each click POSTs one row to `/v1/judgments/<name>/rows`.
4. **Compile to SFT.** Once you have ~20 judgments, click **Compile to SFT**
   — kiln writes a new SFT dataset whose target is the winner label plus
   your note.
5. **Train a judge LoRA.** Submit the compiled dataset via `/v1/train/sft`
   like any other SFT job.
6. **Validate.** Hold out the most-recent N judgments and click **Validate
   adapter** — kiln runs an inline eval suite that asks the trained LoRA to
   predict the winner on those held-out rows. Accuracy is your judge's
   quality on your own picks.
7. **Use it.** Reference the trained adapter as the `judge_adapter` in any
   future `LlmJudge` scorer. The flywheel closes — your evals get smarter
   the more you use kiln.
8. **Prune and retrain.** When the judge LoRA makes a mistake you disagree
   with, delete that judgment row (`/v1/judgments/<name>/rows/<id>`) and
   retrain. Because the dataset is the asset and training is deterministic,
   removing a row + retraining = removing the LoRA's exposure to that
   data point. The whole system embraces this "data as asset" model.

This loop runs entirely on your kiln server — no Claude/GPT calls
anywhere in the path. The model that grades your evals is one *you* trained
on *your* judgments.

## Scorer reference

Every scorer is a JSON object with a `kind` field. They all live under
`kiln_eval::scorers`. The full set:

### `exact_match`
Strict textual equality after optional normalization.

```json
{ "kind": "exact_match", "case_sensitive": false, "strip_whitespace": true }
```

Defaults: case-insensitive, strip whitespace, NFC-normalize unicode.

### `contains`
Substring check with three modes.

```json
{ "kind": "contains", "phrases": ["thanks", "anytime"], "mode": "any" }
```

`mode` values: `any` (default), `all`, `none`. `none` is useful for safety
evals — "the response must not contain any of these phrases."

### `regex`
Pattern match with an optional capture group.

```json
{ "kind": "regex", "pattern": "answer:\\s*(-?\\d+)", "capture_group": 1 }
```

Without `capture_group`, the example passes when the pattern matches
anywhere. With `capture_group`, the captured slice is compared to the
example's `target`.

### `json_validity`
Did the model emit parseable JSON?

```json
{
  "kind": "json_validity",
  "require_object": true,
  "required_paths": ["/tool_call/name", "/tool_call/arguments"]
}
```

When the example has a `target`, the parsed output is compared
structurally (key order doesn't matter). When `required_paths` is set,
every JSON Pointer must resolve.

### `multiple_choice`
Reduce the completion to a single answer label and compare.

```json
{ "kind": "multiple_choice", "choices": ["A", "B", "C", "D"] }
```

Recognizes `"Answer: X"`, `"The answer is X"`, leading `X)`/`(X)`/`X.`,
and bare standalone letters. `target` is the correct letter.

### `numeric_tolerance`
Extract the last number in the completion and check tolerance.

```json
{ "kind": "numeric_tolerance", "atol": 0.001, "rtol": 0.01, "integer_only": false }
```

Passes iff `|got - target| <= atol + rtol * |target|`. Strips `$`, `%`,
and thousands-separator commas.

### `llm_judge`
Ask another model to grade the completion.

```json
{
  "kind": "llm_judge",
  "judge_adapter": "judge-v1",
  "template": "Score the following answer 0-1...",
  "score_regex": "(?i)score[:\\s]*([01](?:\\.\\d+)?)"
}
```

The judge prompt receives `{question}`, `{target}`, `{answer}`
placeholders. The default template asks for a 0/1 score and the default
regex matches `Score: <0..1>`.

### `tool_call`

The right scorer for agentic trajectories. Compares a predicted tool call
(extracted from the model's completion — supports OpenAI `tool_calls`,
inline JSON, and fenced ```tool_call``` blocks) against a target tool call.
Three things contribute to the final score:

1. **Name match** (`exact` / `case_insensitive` / `one_of`).
2. **Structure** — same argument keys?
3. **Content** — per-argument quality.

```json
{
  "kind": "tool_call",
  "name_match": {"kind": "case_insensitive"},
  "args": {"kind": "auto"}
}
```

`args.kind` values:

- **`auto`** (default) — pick per-arg scoring based on the value's shape.
  Strings of free-form prose get contains-with-key-phrases; numbers go
  through tolerance; JSON args go through structural equality. When the
  arg key is `command` / `cmd` / `script` / `shell`, kiln runs a small
  POSIX-ish lexer + introspector that handles the common case where a
  `bash` tool wraps `python3 -c '…'` / `node -e '…'`: same classification
  (e.g. `python_inline`) gets sub-scored by code token similarity on the
  inner program.
- **`structural`** — full canonicalized equality of the arguments JSON.
  Keys order-independent, but values must match.
- **`keys_only`** — only check the *set* of keys. Forgiving when values
  are non-deterministic (timestamps, IDs).
- **`per_key`** — per-argument scorer map. Lets you score a `query` arg
  with `Contains`, a `code` arg with `Code`, and an `id` arg with
  `ExactMatch` — *in the same tool call*.

```json
{
  "kind": "tool_call",
  "args": {
    "kind": "per_key",
    "scorers": {
      "query": {"kind": "contains", "phrases": ["paris", "france"], "mode": "all", "case_sensitive": false},
      "code":  {"kind": "code", "language": "python", "style": {"kind": "token_similarity", "min_jaccard": 0.6}}
    },
    "extra_key_penalty": 0.1
  }
}
```

### `code`

For code-writing evals. Extracts the first fenced code block (or
fence-less but code-shaped content) from both target and prediction, then
applies one of four styles:

- **`any_block`** — pass if the completion contains any code block of the
  declared language.
- **`exact_block`** — strict equality after normalization (whitespace, and
  optional `strip_comments: true` for `#` / `//` comment removal).
- **`token_similarity`** (default, `min_jaccard: 0.6`) — Jaccard on
  identifier-like tokens. Lets the model rename variables while still
  catching missing logic.
- **`line_coverage`** — fraction of non-trivial target lines present in
  the completion. Useful for "did you import all the right modules" style
  checks where exact-order isn't important.

```json
{"kind": "code", "language": "python",
 "style": {"kind": "token_similarity", "min_jaccard": 0.6}}
```

### `all` / `any`
Compose multiple scorers.

```json
{
  "kind": "all",
  "scorers": [
    { "kind": "json_validity", "require_object": true },
    { "kind": "regex", "pattern": "tool_call" }
  ]
}
```

`all` requires every sub-scorer to pass; `any` accepts the first pass.
Score is the mean (`all`) or max (`any`).

## Per-example overrides

Suite-level defaults can be overridden per example. Common pattern: most
examples use one scorer, but a handful need a custom regex:

```json
{
  "messages": [{"role": "user", "content": "Return JSON: {\"x\": 1}"}],
  "target": "{\"x\": 1}",
  "scorer": { "kind": "json_validity", "require_object": true },
  "generation": { "temperature": 0.0, "max_tokens": 64 },
  "tags": ["json", "easy"],
  "weight": 2.0
}
```

`tags` slice the aggregate report (see `pass_rate_by_tag` in the result).
`weight` contributes to the weighted mean — set it to 0 for "test but
don't count" rows.

## API reference

### `POST /v1/eval/suites`
Register or overwrite (`?force=true`) a suite. Body = `EvalSuite` JSON.

### `GET /v1/eval/suites`
List registered suites with summaries.

### `GET /v1/eval/suites/{name}`
Fetch a single suite's full content.

### `DELETE /v1/eval/suites/{name}`
Remove a registered suite.

### `POST /v1/eval/run`
Submit an eval job. Exactly one of `suite` (registered name) or
`inline_suite` must be set. Optional fields: `adapter` (empty string = base
model), `generation` (override).

### `POST /v1/eval/compare`
Run a suite against multiple adapters in one job. Body:

```json
{ "suite": "math-smoke", "adapters": ["", "v1", "v2"] }
```

The job's `runs` array carries one `SuiteResult` per adapter in the order
you sent them. Cap: 8 adapters per submission.

### `GET /v1/eval/jobs`
List all tracked eval jobs (queued / running / terminal).

### `GET /v1/eval/jobs/{job_id}`
Per-job status. While running, the response includes a `progress` object
with `examples_completed`, `examples_total`, `running_accuracy`, and
`running_mean_score`.

### `DELETE /v1/eval/jobs/{job_id}`
Cancel a queued or running job.

## Post-training auto-eval

Attach `post_eval` to any SFT or GRPO request and the trained adapter is
automatically evaluated when training completes:

```bash
curl -X POST http://localhost:8420/v1/train/sft \
  -H 'content-type: application/json' \
  -d '{
        "examples": [...],
        "config": {"epochs": 3, "output_name": "math-v1"},
        "post_eval": {
          "suite": "math-smoke",
          "include_baseline": true
        }
      }'
```

When the training job finishes the worker enqueues two eval jobs (one
against the trained adapter, one against the base model) and back-links
their IDs onto the training status via `linked_eval_job_id`. Use
`include_baseline: false` to skip the base-model run.

## CLI

The CLI is installed as `kiln-eval` next to `kiln`. Set `KILN_SERVER_URL`
to point at a non-local server.

```bash
kiln-eval list
kiln-eval register --file smoke.json [--force]
kiln-eval run --suite math-smoke --adapter v1 --watch
kiln-eval run --file smoke.json --adapter v1 --watch --json
kiln-eval compare --suite math-smoke --adapter v1 --adapter v2 --watch
kiln-eval probe --prompt "1+1?" --target 2 --scorer numeric --adapter v1
```

`probe` is a one-off helper that wraps a single example as an inline
suite — useful during model debugging.

## Suite-file layouts

Two layouts are supported:

1. **Single-file JSON** — everything inline, as in the example above.
2. **Header + JSONL** — for large suites, write a `suite.json` that
   contains everything *except* the `examples` array, and pair it with
   an `examples.jsonl` (one example per line). The registry loads both
   when you register the directory.

## Result shape

`SuiteResult.metrics` carries:

- `num_examples`, `num_pass`, `num_fail`, `num_invalid`, `num_error`
- `accuracy` — pass rate
- `mean_score`, `weighted_mean_score`
- `latency` — `{p50_ms, p90_ms, p99_ms, mean_ms, max_ms}`
- `total_prompt_tokens`, `total_completion_tokens`
- `elapsed_secs`
- `pass_rate_by_tag` — per-tag pass rate (uses the first completion when
  `n>1`)
- `by_scorer` — per-scorer-kind breakdown when the suite mixes scorers

Every example record (`outcomes[]`) carries:

- `example_id`, `completion_index`, `completion_text`
- `kind` — `pass | fail | invalid | error`
- `score`, `detail`
- `prompt_tokens`, `completion_tokens`, `latency_ms`
- `tags` echoed from the example

## Recipes

**A/B test a fresh adapter against the baseline.**
```bash
kiln-eval compare --suite math-smoke --adapter "" --adapter math-v1 --watch
```

**Gate a promotion behind eval accuracy.** Use `post_eval.min_accuracy`
on your training request. Trained adapters below the threshold are still
saved to disk so you can inspect them, but kiln won't auto-load them.

**JSON-shape gate for tool-call models.** Use a composite scorer:
```json
{
  "kind": "all",
  "scorers": [
    { "kind": "json_validity", "required_paths": ["/tool_call/name"] },
    { "kind": "regex", "pattern": "tool_call" }
  ]
}
```

**Slice a suite by difficulty.** Tag every example (`"tags": ["easy"]` /
`["hard"]`) and read `pass_rate_by_tag` to see where the model breaks.

## Qwen3.5-native chat format

Kiln only targets Qwen3.5-4B, so the eval system is *precise* about the
model's wire format. Three things matter:

### 1. Thinking blocks are stripped before scoring

Qwen3.5's chat template prefills `<think>\n` into every assistant turn,
so the raw completion looks like:

```
<think>
Let me work out the capital of France.
</think>

Paris
```

Every scorer except `tool_call` and `llm_judge` strips the
`<think>…</think>` block before comparing — so an `exact_match` target
of `Paris` passes even when reasoning is verbose. Scorers see the *answer*,
not the chain-of-thought.

The reasoning text is preserved on each `ExampleOutcome` as
`reasoning_text`, so dashboards can render "thought 432 chars before
answering" without re-parsing. The aggregate metrics include a
`reasoning_length` histogram (mean / p50 / p90 / max) across the run, and
`num_unclosed_thinking` flags completions that opened `<think>` but never
closed it (typically max_tokens hit inside reasoning).

To disable thinking on a per-example basis, set
`"generation": {"chat_template_kwargs": {"enable_thinking": false}}` —
Qwen3.5's template will then prefill an empty `<think>\n\n</think>\n\n`
block.

### 2. Tool calls — XML and JSON both score

Qwen3.5's *native* tool-call wire form is XML:

```
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
<parameter=units>
celsius
</parameter>
</function>
</tool_call>
```

`Scorer::ToolCall` parses every format Kiln has seen (Qwen3.5 XML,
canonical JSON `{"tool_calls":[…]}`, OpenAI `function.arguments` strings,
fenced ```` ```tool_call``` ```` blocks) into the same structured
`ParsedToolCall` and compares structurally. A target stored as JSON
canonical scores correctly against a model that emitted Qwen3.5 XML, and
vice versa.

The serving API performs the same normalization for clients: when a
tools-bearing `/v1/chat/completions` request produces Qwen3.5 XML, Kiln
returns OpenAI-shaped `tool_calls` with `finish_reason: "tool_calls"` in
both non-streaming responses and SSE deltas. Raw XML remains accepted in
datasets and eval artifacts, but OpenAI-compatible agents should not see it
as assistant prose.

Numeric and boolean XML params auto-coerce to their JSON shape, so a
target of `{"replace_all": false}` compares correctly against a model
that wrote `<parameter=replace_all>\nfalse\n</parameter>`. Both `False`
(Python `str()`) and `false` (JSON) tokens are recognized.

Multi-call targets pair calls positionally; excess predicted calls
subtract `0.25 / target_count` from the score, and missing calls fail
the corresponding pair. The scorer detail string surfaces the actual
on-the-wire formats (`formats=[xml,json]`) so you can spot a model that
regressed from native XML to JSON.

### 3. Tools belong on the suite

Agentic suites declare their tool catalogue once on the suite itself:

```json
{
  "name": "weather-agent",
  "default_scorer": {"kind": "tool_call"},
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Return current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "examples": [...]
}
```

The executor passes these into the chat template so the Qwen3.5
`<tools>` system block renders into every prompt automatically. Per-example
`tools` override the suite default — useful when one example needs a
broader catalogue than the rest.

For agentic suites, the result includes `pass_rate_by_tool`: a map from
tool name to `(num_examples, num_pass, pass_rate)`. So you can spot a
model that nails `Read` but flubs `Edit` without writing per-tool tags
yourself.

## Built-in: `qwen3.5-agentic-core`

Kiln ships a hand-crafted 24-example agentic eval suite under the name
`qwen3.5-agentic-core`. It auto-registers at server startup, so:

```bash
kiln-eval run --suite qwen3.5-agentic-core --adapter my-trained-adapter --watch
```

…just works without authoring anything. The suite covers:

- Pure no-tool answers (the model must NOT invoke a tool for things it
  already knows).
- Single tool calls with exact args / paraphrase-tolerant args /
  ambiguous tool selection.
- Multi-step trajectory followups (assistant emits a tool call, the
  tool returns a result, what does the model say next?).
- Thinking-then-answer probes.
- Code generation, JSON output validation, MCQ, numeric tolerance.
- Tool-call refusals (the model must NOT invoke tools that aren't in
  the catalogue).

Use it as a regression gate after every training run.

## Synthesis strategies (agentic)

Two strategies on top of `final_assistant` / `first_assistant_turn` /
`every_assistant_turn` / `tool_call_predict` cover the agentic-eval
shapes that matter:

- **`tool_response_followup`** — One example per
  `(assistant tool_call → tool result(s) → next assistant)` triple.
  Prompt ends on the tool response so the model is asked "given this
  tool output, what do you do next?".
- **`end_of_trajectory_answer`** — Prompt = full trajectory up through
  the last tool result, target = the closing assistant turn. Filters
  out non-agentic conversations.

`final_assistant` and `first_assistant_turn` now canonicalize raw
Qwen3.5 XML tool calls in `assistant.content` into the JSON envelope
automatically, so SFT data captured from a base Qwen3.5 model
(no structured `tool_calls`) still produces clean tool-call targets.
