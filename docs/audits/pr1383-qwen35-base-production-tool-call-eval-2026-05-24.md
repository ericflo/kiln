# PR 1383 Qwen3.5-4B Production Tool-Call Eval

Date: 2026-05-24
Branch: `ce/production-tool-call-eval`
PR under audit: #1383, base commit `ae43ffd4` (`Update lockfile for trace API eval example`)
Model: base `Qwen/Qwen3.5-4B`, served as `Qwen3.5-4B`
Environment: RunPod NVIDIA RTX A6000 48GB, CUDA 12.4 image, `rustc 1.95.0`
Status: baseline eval completed; base model is not production-ready for this workload without adaptation.

## Summary

PR #1383 adds the production trace eval path: a `kiln-eval trace-suite`
importer for materialized trajectory turns, the standalone `trace_api_eval`
runner, provenance tags, confidence intervals, and tool breakdowns. I used it
to measure base Qwen3.5-4B on a production-derived tool-call workload from
trajectory-trainer's `analysis-v12-diverse-200` sample.

The base model passed **20/106 eligible tool-call examples**, or **18.9%**
accuracy with Wilson 95% CI **12.6%-27.4%**. It frequently emitted some
parseable tool call, but often selected the wrong tool or wrong arguments.
Invalid completions were concentrated in long rambles that hit the 1024-token
completion cap and contained no parseable tool call.

This is a useful baseline because it is weak in the ways an adapter should
improve: exact tool selection, argument fidelity, and multi-call continuation.
It should not be treated as production-capable.

## Evidence

Checked-in evidence directory:
[`docs/audits/pr1383-qwen35-base-production-tool-call-eval-2026-05-24/`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/)

Key files:

| File | Purpose |
| --- | --- |
| [`analysis_v12_diverse_200.suite_report.json`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/analysis_v12_diverse_200.suite_report.json) | Suite synthesis report and source metadata. |
| [`qwen35_base_trace_api_eval_result.json`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/qwen35_base_trace_api_eval_result.json) | Machine-readable eval result with outcome-level scores. |
| [`trace_suite2.log`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/trace_suite2.log) | Successful suite generation log. |
| [`base_eval.log`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/base_eval.log) | Successful eval log and headline metrics. |
| [`server.log`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/server.log) | Kiln server startup and CUDA residency evidence. |
| [`cuda_build.log`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/cuda_build.log) | RunPod CUDA release build log. |
| [`qwen3_fix_test2.log`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/qwen3_fix_test2.log) | RunPod regression test for the UTF-8 parser fix. |

The 20.6 MB materialized prompt JSONL is intentionally not checked in. The
suite report and eval result are enough to reproduce the aggregate result and
inspect outcome-level behavior without committing the full production prompt
materialization.

## Workload

Source workload: `analysis-v12-diverse-200`, a curated 200-turn production
trajectory sample. Each selected turn was materialized with trajectory-trainer's
`materialize_turn.py` in prompt/chosen JSONL format:

```bash
python3 /data/apps/trajectory-trainer/scripts/materialize_turn.py \
  --turn-id <turn_id> \
  --format prompt-chosen-jsonl
```

Suite generation command on RunPod:

```bash
cargo run -p kiln-eval -- trace-suite \
  --format prompt_chosen_jsonl \
  --input /workspace/pr1383-qwen35-base-eval/analysis_v12_diverse_200.prompt_chosen.jsonl \
  --output /workspace/pr1383-qwen35-base-eval/analysis_v12_diverse_200.suite.json \
  --stats-output /workspace/pr1383-qwen35-base-eval/analysis_v12_diverse_200.suite_report.json \
  --suite-name production-tool-call-analysis-v12-diverse-200 \
  --description "Production trajectory tool-call eval from trajectory-trainer analysis-v12-diverse-200 sample materialized with materialize_turn.py" \
  --max-examples 200 \
  --seed 20260524 \
  --max-tokens 1024
```

Suite synthesis result:

| Field | Value |
| --- | ---: |
| Rows seen / parsed | 200 / 200 |
| Eligible current-turn tool-call examples | 106 |
| Examples kept | 106 |
| Skipped because target had no tool call | 94 |
| Parse errors | 0 |
| Prompt length skips | 0 |
| Target length skips | 0 |
| Suite hash | `sha256:20b644b62342605a8e50f7e719ddb003f7967cb0e34cc31ed96517668be1fb04` |

Target tool histogram across the 106 kept examples counts every target call,
so multi-call examples contribute more than one tool:

| Tool | Target calls |
| --- | ---: |
| Bash | 89 |
| Read | 31 |
| WebFetch | 21 |
| Edit | 10 |
| Glob | 6 |
| Grep | 6 |
| TaskCreate | 6 |
| Agent | 3 |
| ToolSearch | 3 |
| Skill | 2 |
| Write | 2 |
| WebSearch | 1 |
| string | 1 |

## Run Configuration

RunPod pod pool lease:

| Field | Value |
| --- | --- |
| Pod ID | `hl2nuqtjjw2tjd` |
| Lease ID | `pod-3ff344b92f73297be7f61be2` |
| GPU | NVIDIA RTX A6000 48GB |
| CUDA | 12.4 toolkit, driver 570.195.03 |
| Kiln server flags | `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true`, `KILN_REQUEST_TIMEOUT_SECS=1800` |

Release CUDA server build on RunPod:

```bash
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln
```

Server command on RunPod:

```bash
KILN_MODEL_PATH=/workspace/models/Qwen3.5-4B \
KILN_SERVED_MODEL_ID=Qwen3.5-4B \
KILN_W4A16=1 \
KILN_CUDA_GRAPHS=true \
KILN_REQUEST_TIMEOUT_SECS=1800 \
RUST_LOG=info \
./target/release/kiln serve
```

The server log confirms CUDA mode, W4A16 production path, CUDA graphs enabled,
and `4206M` model parameters loaded from `/workspace/models/Qwen3.5-4B`.

Eval command on RunPod:

```bash
cargo run -p kiln-eval --example trace_api_eval -- \
  --suite /workspace/pr1383-qwen35-base-eval/analysis_v12_diverse_200.suite.json \
  --api-base http://127.0.0.1:8420/v1 \
  --model Qwen3.5-4B \
  --temperature 0 \
  --max-tokens 1024 \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --output /workspace/pr1383-qwen35-base-eval/qwen35_base_trace_api_eval_result.json
```

No local Kiln build, test, cargo check, or rustfmt was run. Build, targeted
test, suite generation, server run, and eval were all done on RunPod.

## Headline Metrics

Machine-readable source:
[`qwen35_base_trace_api_eval_result.json`](pr1383-qwen35-base-production-tool-call-eval-2026-05-24/qwen35_base_trace_api_eval_result.json)

| Metric | Value |
| --- | ---: |
| Examples | 106 |
| Pass | 20 |
| Fail | 71 |
| Invalid | 15 |
| Error | 0 |
| Accuracy | 18.867925% |
| Wilson 95% CI | 12.559299%-27.354097% |
| Mean score | 0.3735951 |
| Weighted mean score | 0.3735951 |
| Non-XML tool calls | 91 |
| Unclosed thinking blocks | 0 |
| Missing required schema fields | 0 |
| Extra unknown schema fields | 0 |

Runtime and token totals:

| Metric | Value |
| --- | ---: |
| Eval wall time | 4399.74 seconds |
| Prompt tokens | 3,717,586 |
| Completion tokens | 28,241 |
| Average prompt tokens/example | 35,071.6 |
| Average completion tokens/example | 266.4 |
| Latency p50 | 21,614.9 ms |
| Latency p90 | 131,010.7 ms |
| Latency p99 | 209,755.2 ms |
| Latency mean | 41,505.5 ms |
| Latency max | 210,730.0 ms |

## Breakdowns

By production model tag:

| Production model | Pass / examples | Pass rate |
| --- | ---: | ---: |
| `claude-opus-4-6` | 7 / 41 | 17.1% |
| `claude-opus-4-7` | 6 / 35 | 17.1% |
| `claude-sonnet-4-6` | 7 / 29 | 24.1% |
| `claude-haiku-4-5-20251001` | 0 / 1 | 0.0% |

By split:

| Split | Pass / examples | Pass rate |
| --- | ---: | ---: |
| train | 18 / 91 | 19.8% |
| val | 1 / 10 | 10.0% |
| test | 1 / 5 | 20.0% |

By target-tool tag. These tags count any example containing the tool, so
multi-call examples can appear in more than one row.

| Tool tag | Pass / examples | Pass rate |
| --- | ---: | ---: |
| Bash | 10 / 89 | 11.2% |
| Read | 4 / 31 | 12.9% |
| WebFetch | 1 / 21 | 4.8% |
| Edit | 4 / 10 | 40.0% |
| Glob | 1 / 6 | 16.7% |
| Grep | 0 / 6 | 0.0% |
| TaskCreate | 0 / 6 | 0.0% |
| Agent | 0 / 3 | 0.0% |
| ToolSearch | 0 / 3 | 0.0% |
| Skill | 0 / 2 | 0.0% |
| Write | 0 / 2 | 0.0% |
| WebSearch | 0 / 1 | 0.0% |
| string | 0 / 1 | 0.0% |

By first target tool. This is the `pass_rate_by_tool` view and is most useful
for first-call confusion analysis, not for total workload share.

| First target tool | Pass / examples | Pass rate | Wilson 95% CI |
| --- | ---: | ---: | ---: |
| Bash | 10 / 56 | 17.9% | 10.0%-29.8% |
| Read | 4 / 12 | 33.3% | 13.8%-60.9% |
| Edit | 4 / 10 | 40.0% | 16.8%-68.7% |
| WebFetch | 1 / 9 | 11.1% | 2.0%-43.5% |
| Glob | 1 / 4 | 25.0% | 4.6%-69.9% |
| Grep | 0 / 4 | 0.0% | 0.0%-49.0% |
| ToolSearch | 0 / 3 | 0.0% | 0.0%-56.1% |
| Skill | 0 / 2 | 0.0% | 0.0%-65.8% |
| Write | 0 / 2 | 0.0% | 0.0%-65.8% |
| Agent | 0 / 1 | 0.0% | 0.0%-79.3% |
| TaskCreate | 0 / 1 | 0.0% | 0.0%-79.3% |
| WebSearch | 0 / 1 | 0.0% | 0.0%-79.3% |
| string | 0 / 1 | 0.0% | 0.0%-79.3% |

First-call confusion matrix:

| Target first tool | Predicted first tool counts |
| --- | --- |
| Agent | Read 1 |
| Bash | Bash 37, none 7, Read 4, Write 4, Glob 3, Grep 1 |
| Edit | Edit 4, none 3, Read 3 |
| Glob | Glob 2, none 1, Bash 1 |
| Grep | Read 4 |
| Read | Read 8, Bash 3, TaskUpdate 1 |
| Skill | none 1, Glob 1 |
| TaskCreate | Read 1 |
| ToolSearch | Bash 2, Read 1 |
| WebFetch | WebFetch 8, WebSearch 1 |
| WebSearch | none 1 |
| Write | none 2 |
| string | Read 1 |

## Failure Analysis

Invalid completions:

- 15/106 outcomes were `invalid`.
- All 15 had detail `no tool_call found in completion`.
- All 15 also had `completion_tokens == 1024`, so they exhausted the eval
  completion cap without producing a parseable tool call.

Multi-call turns:

| Target call count | Examples |
| --- | ---: |
| 1 | 76 |
| 2 | 15 |
| 3 | 7 |
| 4 | 3 |
| 6 | 1 |
| 8 | 2 |
| 10 | 2 |

The 30 examples with more than one expected target call all failed. The common
failure shape was a plausible first call followed by one or more "missing
predicted call" scorer details. This is a major caveat for interpreting the
headline result: the standalone runner asks for one generated assistant
message/tool-call sequence, and the base model often emits too few calls for
multi-call production turns.

Tool choice:

- Bash has the most coverage and remains weak: 10/56 first-call pass rate
  (17.9%) and 10/89 target-tag pass rate (11.2%).
- Read and Edit are the strongest first-call slices, but sample sizes are
  small and confidence intervals are wide.
- Grep, ToolSearch, TaskCreate, Write, Agent, Skill, WebSearch, and the single
  `string` pseudo-tool slice had no passes.
- Grep targets were especially poor: all four first-call Grep examples were
  emitted as Read.
- WebFetch tool selection was often close at first-call level
  (8 WebFetch predictions for 9 first-call WebFetch targets), but exact content
  still rarely passed.

Schema behavior:

- `num_schema_missing_required = 0`
- `num_schema_extra_unknown = 0`
- Most failures were not parser schema failures; they were wrong tool names,
  missing calls, or argument/content mismatches.

## PR Issue Exposed During Eval

The first RunPod `trace-suite` attempt found a panic in
`find_first_tool_call_object` in `crates/kiln-eval/src/qwen3.rs`. The JSON
probe sliced `&text[i..lookahead_end]` after a fixed byte lookahead, which can
land inside a multibyte UTF-8 character.

The fix backs `lookahead_end` down to a valid UTF-8 character boundary before
slicing. Added regression test:

```text
qwen3::tests::json_probe_lookahead_is_utf8_boundary_safe
```

RunPod validation:

```bash
cargo test -p kiln-eval json_probe_lookahead_is_utf8_boundary_safe -- --nocapture
```

Result: passed on RunPod. `rustfmt --check` could not be run on the pod because
the rustfmt component was missing; it was not run locally.

## Interpretation

Base Qwen3.5-4B is far below the bar for production trace tool calling on this
workload. The model has some latent ability to emit OpenAI-shaped tool calls,
but exact tool-call reproduction is too unreliable:

- Overall pass rate is only 18.9%.
- 14.2% of eligible examples produced no parseable tool call before the token
  cap.
- Multi-call production turns are essentially unsolved in this setup.
- Tool selection is inconsistent on common coding tools, especially Bash
  versus Read/Glob/Grep confusion.

Use this run as the base-model floor for adapter work. Any production adapter
should be evaluated against at least the same suite, with special attention to
multi-call turns, long-context prompt behavior, and capped no-tool-call invalids.
