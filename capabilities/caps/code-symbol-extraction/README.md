# code-symbol-extraction (OPD canary)

Round-2 canary cap for the kiln OPD trainer fix. **Round 1 blocked by
97% EOS-skip bug; now fixed.**

## Status (round 2)

| File | Status |
|------|--------|
| `capability.md` | Spec + round-2 canary plan |
| `rubric.py` | 4-component (parses, format, recall, precision) |
| `calibration/` | 5 good + 5 bad; separation +0.24 PASS |
| `archive/` | Round-1 closeout (97% skip diagnosis) |

## Canary plan

1. Re-run round-1 H1-r16-6ep recipe on patched kiln.
2. Expect effective steps to jump from 7 → 30+ (97% skip drops to <50%).
3. Expect composite 0.937 → 0.96+ (round-1 closeout predicted this).
4. **If canary passes**: unblock the other 5 OPD caps.
5. **If canary fails**: file against kiln; pause all OPD work.

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh
./run_iter.sh h1-r16-6ep         # the round-1 recipe
```
