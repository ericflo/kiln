# pi-source-mod-workflow

`pi-source-mod-workflow` capability.

## Read first

1. [`capability.md`](capability.md) — contract.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.
3. [`../README.md`](../README.md) — paradigm defaults.

## Status

**Implementation complete. Calibration passes.**

| File | Status |
|------|--------|
| `capability.md` | Spec |
| `capability.config.json` | Tuned |
| `build_corpus.py` | Generates train + eval |
| `rubric.py` | Score_one implemented |
| `rubric_sanity.py` | Mandatory gate |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline |
| `calibration/{good,bad}.jsonl` | 5 + 5 fixtures |

## Quickstart

```bash
python3 build_corpus.py
python3 rubric_sanity.py
./capability.oracle.sh
./run_iter.sh h1-default-recipe
```

See `capability.md` for hypotheses and headroom.
