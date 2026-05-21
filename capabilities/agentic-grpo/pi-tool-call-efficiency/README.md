# pi-tool-call-efficiency

**EVAL-ONLY cap (round-2 reshape).** Measures tool-call efficiency
across adapters trained on other caps. Does NOT train an adapter.

## Why repurposed

Round 1's plan was to train a standalone tool-call-efficiency adapter.
That would have either:

- duplicated signal already present in sister caps' rubrics
  (pi-doctest, pi-code-search, pi-code-comprehension, pi-faithful-completion
  all include tool_call_efficiency sub-scores), or
- trained on synthetic prompts that don't transfer to real tasks.

Round 2 makes this a transfer eval: pass any adapter name, get a
report of n_tool_calls distribution.

## Read first

1. [`capability.md`](capability.md) — round-2 reshape rationale.
2. [`../../LAYOUT.md`](../../LAYOUT.md).
3. [`../README.md`](../README.md).

## Status

**Implementation complete. Calibration passes (separation +0.75).**

| File | Status |
|------|--------|
| `capability.md` | Spec + round-2 reshape |
| `capability.config.json` | Marked `mode: eval_only` |
| `build_corpus.py` | Samples tasks from sister caps' evals when available; fallback mini-corpus otherwise |
| `rubric.py` | Single composite = `tool_call_efficiency` (no multiplicative gate; cap is eval-only) |
| `rubric_sanity.py` | Passes |
| `rollout.py` | Pi driver (used for direct eval) |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Eval wrapper — does NOT call cuda_grpo_ablation |
| `calibration/good.jsonl` | 5 examples (0-4 tool calls) |
| `calibration/bad.jsonl` | 5 examples (≥10 tool calls) |

## Quickstart

```bash
# Build / refresh the eval set from sister caps.
python3 build_corpus.py

# Eval base.
./capability.oracle.sh

# Eval any installed adapter.
./capability.oracle.sh pi-doctest-iter5
./capability.oracle.sh pi-faithful-completion-iter50

# Compare adapters by writing a comparison shell script that loops over
# adapters and runs capability.oracle.sh.
```

## Output shape

The oracle reports the per-adapter tool-call-efficiency distribution:

  - mean n_tool_calls
  - distribution: efficient (<=4) / moderate (5-9) / wasteful (>=10)
  - delta vs base mean

## Composition

- This cap composes WITH every other agentic cap via the integration
  track. Use it after training a per-cap adapter to verify the trained
  adapter didn't accidentally INCREASE tool-call counts.

## History

Round-1 scaffold; reshaped in round 2 to eval-only.
