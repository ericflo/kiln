# math-broad (SFT)

Word-problem math via non-mathy supervision. **Round 1 saturated at
1.000 single-seed (order-noise tail); 3-seed mean ~0.95.**

## Status

| File | Status |
|------|--------|
| `capability.md` | Spec + ideas + round-2 plan |
| `rubric.py` | Implemented (exact + substring match, normalize) |
| `calibration/` | 5 good + 5 bad fixtures, separation +1.00 PASS |
| `capability.anchor.sh` | Regression watch on non-math suite |
| `archive/` | Round-1 ledger (20 iters + baseline) |

## Round-2 plan

1. Replicate iter-13 recipe across 5 seeds.
2. Hard-eval pool of harder algebra/calculus.
3. SFT vs OPD comparison (if math teacher available).

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh        # baseline
./run_iter.sh h1-iter13-replicate
```
