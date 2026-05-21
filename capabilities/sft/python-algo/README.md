# python-algo (SFT)

Algorithmic Python coding via SFT.

## Status

| File | Status |
|------|--------|
| `capability.md` | Spec |
| `rubric.py` | compile + define_fn + test_pass_rate |
| `calibration/` | 5 good + 5 bad, separation +0.50 PASS |
| `capability.anchor.sh` | Regression watch |

## Round-2 coordination

Coordinate corpus with `pi-doctest` to avoid HumanEval leak. Candidates:
LeetCode-style, competitive programming, Project Euler.

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh
./run_iter.sh h1-default
```
