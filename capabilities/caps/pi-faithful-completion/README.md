# pi-faithful-completion

Honest termination + OUTPUT FORMAT discipline. **Round 1 BIG winner
(+8.28pp)**, combination recipe (temp 0.6 × light prompt × lr 3e-5)
broke through.

## Status (round 2)

| File | Status |
|------|--------|
| `capability.md` | Full spec + round-2 improvement plan |
| `rubric.py` | Multi-component (response, task) signature; no workdir |
| `rubric_sanity.py` | Custom (passes; sep +0.93) |
| `calibration/` | **5 good + 5 bad fixtures, separation +0.93 PASS** |
| `archive/` | 50-iter closeout |

## Round-2 improvements

1. **Chain training from iter-50 best** — round 1 stopped at 50.
2. **Cross-cap anchor regression** — did this hurt code-comprehension?
3. **OPD for format polish** on top of GRPO behavior win.
4. Combinations compound; sweep crosses not axes.

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh           # baseline ~0.72
./run_iter.sh h1-default-recipe
BASE_ADAPTER=pi-faithful-iter50-best ./run_iter.sh h-chain
```
