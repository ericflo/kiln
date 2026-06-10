# PR 1383 Qwen3.5-4B Production Tool-Call Eval

Date: 2026-05-25
Branch: `ce/production-tool-call-eval`
PR under audit: #1383
Model: base `Qwen/Qwen3.5-4B`, served as `Qwen3.5-4B`
Environment: RunPod NVIDIA RTX A6000 48GB, CUDA 12.4 image
Status: final larger-sample result.

## Summary

This audit evaluates base Qwen3.5-4B on the production trace tool-call
workload added by PR #1383, using a larger random-stratified sample than the
initial 106-example check.

The larger suite was synthesized from
`analysis_v12_random_stratified_1000_cap262144`: 1000 materialized production
turns, 800 eligible current-turn tool-call examples, and 775 kept eval
examples after 25 prompt-length skips.

Across the full 775-example suite, base Qwen3.5-4B passed **269/775** examples,
or **34.7%** exact-pass accuracy. The Wilson 95% confidence interval is
**31.4%-38.1%**. Invalid generations were **35/775** (**4.5%**) and there were
**0 eval/API errors**.

This is a substantially more precise estimate than the initial 106-example
audit, whose result was 20/106 or 18.9% accuracy with a 12.6%-27.4% confidence
interval. The larger random-stratified sample lands higher, but the conclusion
does not change: base Qwen3.5-4B is not close to production-ready for this
tool-call policy without adaptation.

## Evidence

Checked-in evidence directory:
[`docs/audits/pr1383-qwen35-base-production-tool-call-eval-1000-2026-05-25/`](pr1383-qwen35-base-production-tool-call-eval-1000-2026-05-25/)

Included files:

| File | Purpose |
| --- | --- |
| `aggregate_metrics.json` | Machine-readable aggregate metrics over all 775 examples. |
| `analysis_v12_random_stratified_1000_cap262144.suite_report.json` | Suite synthesis report and source metadata for the 775-example suite. |
| `materialize_errors.log` | Materialization error log; empty for this run. |
| `trace_suite2.log` | Suite generation log. |
| `qwen35_base_trace_api_eval_result_shard_01.json` ... `shard_11.json` | Machine-readable eval outputs for all completed shards. |
| `base_eval_shard_01.log` ... `shard_11.log` | Per-shard RunPod eval logs. |

The 239 MB materialized prompt JSONL is intentionally not checked in.

## Suite Synthesis

| Field | Value |
| --- | ---: |
| Source rows seen / parsed | 1000 / 1000 |
| Eligible tool-call turns | 800 |
| Examples kept | 775 |
| Skipped no-tool-call turns | 200 |
| Skipped prompt-too-long turns | 25 |
| Parse errors | 0 |
| Duplicate skips | 0 |
| Target-too-long skips | 0 |

Target tool histogram counts target calls, so multi-call examples can
contribute more than one tool:

| Tool | Target calls |
| --- | ---: |
| Bash | 601 |
| Read | 201 |
| Edit | 45 |
| Grep | 43 |
| TodoWrite | 17 |
| Write | 9 |
| Agent | 6 |
| ToolSearch | 6 |
| WebFetch | 3 |
| ScheduleWakeup | 1 |
| NAME | 1 |

## Run Configuration

| Field | Value |
| --- | --- |
| API base | `http://127.0.0.1:8420/v1` |
| Model | `Qwen3.5-4B` |
| Temperature | `0` |
| Max tokens | `1024` |
| Extra body | `{"chat_template_kwargs":{"enable_thinking":false}}` |
| Shards | 11 total: 10 shards of 75 examples, final shard of 25 examples |
| Hardware | RunPod NVIDIA RTX A6000 48GB |

Shards 01-09 ran via the RunPod release eval command. After the pod pool reset
`/workspace/kiln` to `main`, shards 10-11 ran the already-built
`target/release/examples/trace_api_eval` binary directly with the same suite,
API base, model, temperature, max-token, and output settings. No local Kiln
build, test, cargo check, or rustfmt was run.

## Aggregate Metrics

| Metric | Value |
| --- | ---: |
| Examples | 775 |
| Pass | 269 |
| Fail | 471 |
| Invalid | 35 |
| Error | 0 |
| Accuracy | 34.709677% |
| Wilson 95% CI | 31.440959%-38.129228% |
| Mean score | 0.5593882 |
| Invalid rate | 4.516129% |
| Prompt tokens | 55,707,705 |
| Completion tokens | 122,540 |
| Sum elapsed time | 54,425.30s |
| Latency p50 | 54,149 ms |
| Latency p90 | 128,533 ms |
| Latency p95 | 187,741 ms |
| Latency p99 | 351,034 ms |
| Max latency | 484,766 ms |
| Server eval/API errors | 0 |
| Server timeouts | 0 |

## Per-Shard Metrics

| Shard | Examples | Pass | Fail | Invalid | Error | Accuracy | Wilson 95% CI | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 75 | 29 | 43 | 3 | 0 | 38.7% | 28.5%-50.0% | 0.5781846 |
| 02 | 75 | 29 | 41 | 5 | 0 | 38.7% | 28.5%-50.0% | 0.5431198 |
| 03 | 75 | 30 | 41 | 4 | 0 | 40.0% | 29.7%-51.3% | 0.5711452 |
| 04 | 75 | 23 | 48 | 4 | 0 | 30.7% | 21.4%-41.8% | 0.6190770 |
| 05 | 75 | 24 | 47 | 4 | 0 | 32.0% | 22.5%-43.2% | 0.5011408 |
| 06 | 75 | 27 | 44 | 4 | 0 | 36.0% | 26.1%-47.3% | 0.5675589 |
| 07 | 75 | 24 | 46 | 5 | 0 | 32.0% | 22.5%-43.2% | 0.5098116 |
| 08 | 75 | 30 | 44 | 1 | 0 | 40.0% | 29.7%-51.3% | 0.5958825 |
| 09 | 75 | 22 | 50 | 3 | 0 | 29.3% | 20.2%-40.4% | 0.5252005 |
| 10 | 75 | 23 | 51 | 1 | 0 | 30.7% | 21.4%-41.8% | 0.5966354 |
| 11 | 25 | 8 | 16 | 1 | 0 | 32.0% | 17.2%-51.6% | 0.5177665 |

Per-shard accuracy ranged from **29.3%** to **40.0%**. The final aggregate
confidence interval is much narrower than the per-shard intervals and is the
appropriate estimate to use for this 775-example suite.

## Pass Rate By Tool Tag

This table counts example membership by `tool:*` tag, not raw target calls.
Examples with multiple target tools can contribute to multiple rows.

| Tool tag | Examples | Pass | Invalid | Pass rate |
| --- | ---: | ---: | ---: | ---: |
| `tool:Bash` | 524 | 160 | 21 | 30.5% |
| `tool:Read` | 137 | 67 | 6 | 48.9% |
| `tool:Edit` | 44 | 23 | 5 | 52.3% |
| `tool:Grep` | 39 | 11 | 0 | 28.2% |
| `tool:TodoWrite` | 17 | 4 | 0 | 23.5% |
| `tool:Write` | 9 | 2 | 3 | 22.2% |
| `tool:Agent` | 6 | 1 | 0 | 16.7% |
| `tool:ToolSearch` | 6 | 0 | 0 | 0.0% |
| `tool:WebFetch` | 3 | 1 | 0 | 33.3% |
| `tool:NAME` | 1 | 0 | 0 | 0.0% |
| `tool:ScheduleWakeup` | 1 | 0 | 0 | 0.0% |

## Variance And Precision

The main uncertainty estimate is sampling uncertainty over production
examples. With 775 examples, the Wilson 95% confidence interval is about
6.7 percentage points wide, from 31.4% to 38.1%. That is materially tighter
than the initial 106-example audit interval, which was about 14.8 points wide.

The eval does not estimate repeat-run stochastic variance. Each example was
generated once at `temperature=0`. The per-shard spread, 29.3%-40.0%, is
consistent with normal variation across 75-example shards and should not be
interpreted as separate independent model settings.

## Interpretation

Base Qwen3.5-4B can often identify a plausible tool call shape, but it is not
reliably matching the production trace target. Exact-pass accuracy is only
34.7%, and the upper end of the 95% interval is still just 38.1%. Invalid
outputs are also nontrivial at 4.5%.

The model performs better on `Read` and `Edit` tagged examples than on the
dominant `Bash` cases, but `Bash` is the largest part of this workload and
drives the aggregate result. Rare tools remain too sparse for confident
per-tool conclusions, even in the larger sample.

The larger sample supports a confident baseline conclusion: base Qwen3.5-4B is
not production-capable for this workload as-is. It needs adaptation and
post-adaptation evaluation against this same trace-derived suite before it can
be treated as a viable production tool-call policy.

## Limitations

- This is a trace-derived offline eval, not a live multi-turn rollout.
- The scoring is exact-pass oriented; partial score is reported as mean score,
  but the headline result is exact-pass accuracy.
- Each example was generated once at `temperature=0`; repeat-run stochastic
  variance was not measured.
- Rare tools are still underrepresented.
- The full materialized prompt JSONL is preserved in run artifacts but not
  checked in because it is 239 MB.
- No local Kiln build, test, cargo check, or rustfmt was run.
