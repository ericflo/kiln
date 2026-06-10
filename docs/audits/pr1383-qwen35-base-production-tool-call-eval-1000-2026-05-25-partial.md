# PR 1383 Qwen3.5-4B Production Tool-Call Eval Checkpoint

Date: 2026-05-25
Branch: `ce/production-tool-call-eval`
PR under audit: #1383
Model: base `Qwen/Qwen3.5-4B`, served as `Qwen3.5-4B`
Environment: RunPod NVIDIA RTX A6000 48GB, CUDA 12.4 image
Status: partial larger-sample checkpoint; full sharded eval is still running.

## Summary

This checkpoint preserves the completed portion of the larger production
trace tool-call eval requested after the initial 106-example audit. The larger
suite was synthesized from `analysis_v12_random_stratified_1000_cap262144`:
1000 materialized production turns, 800 eligible current-turn tool-call
examples, and 775 kept eval examples after 25 prompt-length skips.

As of this checkpoint, shards 01-08 have completed and been downloaded:
**600/775 eval examples**. Base Qwen3.5-4B passed **216/600**, or **36.0%**
accuracy with Wilson 95% CI **32.3%-39.9%**. Invalid generations were
**30/600** and there were **0 API/eval errors**.

This is not the final result doc. Shards 09-11 remain to be incorporated in
the final audit. The active RunPod eval loop is continuing shard 09 onward.

## Evidence

Checked-in evidence directory:
[`docs/audits/pr1383-qwen35-base-production-tool-call-eval-1000-2026-05-25-partial/`](pr1383-qwen35-base-production-tool-call-eval-1000-2026-05-25-partial/)

Included files:

| File | Purpose |
| --- | --- |
| `analysis_v12_random_stratified_1000_cap262144.suite_report.json` | Suite synthesis report and source metadata for the 775-example suite. |
| `materialize_errors.log` | Materialization error log; empty for this run. |
| `trace_suite2.log` | Suite generation log. |
| `qwen35_base_trace_api_eval_result_shard_01.json` ... `shard_08.json` | Machine-readable eval outputs for completed shards. |
| `base_eval_shard_01.log` ... `shard_08.log` | Per-shard RunPod eval logs. |

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

Target tool histogram counts every target call, so multi-call examples can
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

## Checkpoint Metrics

| Metric | Value |
| --- | ---: |
| Completed examples | 600 / 775 |
| Pass | 216 |
| Fail | 354 |
| Invalid | 30 |
| Error | 0 |
| Accuracy | 36.000000% |
| Wilson 95% CI | 32.259535%-39.918593% |
| Mean score | 0.5607401 |
| Prompt tokens | 43,492,953 |
| Completion tokens | 92,852 |

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

## Interpretation

At 600 examples, this checkpoint is already much more informative than the
initial 106-example audit: the sampling interval is roughly half as wide, and
the workload includes many more Bash, Read, Edit, and Grep cases. The current
estimate still should not be treated as final because the remaining 175
examples can move the aggregate.

The result remains a base-model baseline, not a production-readiness claim.
The model is substantially better on this random-stratified larger sample than
on the prior diverse-200 subset, but 36% exact-pass accuracy with 5% invalid
outputs is still far from a production-capable tool-call policy without
adaptation.

## Limitations

- This checkpoint covers 600/775 examples; shards 09-11 are not included yet.
- The run is sharded because the long-context workload exceeds the RunPod pod
  pool lease window as a single uninterrupted eval.
- Each example is generated once with `temperature=0`; this measures sampling
  uncertainty over examples, not repeat-run stochastic variance.
- Rare tools remain underrepresented even in the larger suite.
- No local Kiln build, test, cargo check, or rustfmt was run.
