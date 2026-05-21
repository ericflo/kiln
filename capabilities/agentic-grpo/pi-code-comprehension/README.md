# pi-code-comprehension

Given a target symbol in a Python snapshot, the agent reads + greps +
emits a structured JSON summary. **Round 1 BIG winner (+12.93pp,
0.6112 → 0.7405)**, ECHO λ=0.075 was the productive ceiling.

## Read first

1. [`capability.md`](capability.md) — 5-component rubric.
2. [`../../LAYOUT.md`](../../LAYOUT.md), [`../README.md`](../README.md).

## Status (round 2)

| File | Status |
|------|--------|
| `capability.md` | Full spec + round-2 improvement plan |
| `capability.config.json` | Tuned |
| `build_corpus.py` | Real corpus (Python repos) |
| `rubric.py` | 5 sub-scores: outcome F1 × (grounding·0.20 + cross_file·0.15 + inv_cov·0.10 + format·0.05 + 0.50) |
| `rubric_sanity.py` | Workdir-dependent; bypassed (see calibration/README.md) |
| `rollout.py` | Pi driver |
| `archive/` | Round-1 WRITEUP, 50-iter loop attempts |

## Round-2 improvements

1. **Cross-file generalization eval** — round 1 saturated cross-file
   recall at 1.00; held-out repo with different layout tests transfer.
2. **OPD from 27B for structured JSON** — format polish on top of GRPO win.
3. **Anchor regression suite** — does adapter hurt code-search?

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh           # baseline ~0.61
./run_iter.sh h1-default-recipe
ECHO_LAMBDA=0.075 ./run_iter.sh h4-echo-0075   # round-1 winning recipe
```

## Headroom

- **Round-1 baseline**: 0.611.
- **Round-1 best (iter 4)**: 0.741.
- **Round-2 target**: cross-file gen + OPD format → ~0.80.
