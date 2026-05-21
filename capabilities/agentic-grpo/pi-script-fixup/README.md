# pi-script-fixup

Verifier-free §5.5 paper recipe. **Round-1 scaffold.**

## Status (round 2)

| File | Status |
|------|--------|
| `capability.md` | Spec + base-adapter dependency doc |
| `rubric.py` | Implemented (mult-gate, read_error_before_edit) |
| `calibration/` | 5 good + 5 bad; separation +0.50 PASS |
| `archive/` | Round-1 verifier-free script |

## Dependency

**REQUIRES base adapter from pi-terminal-bench-lite Phase 2.** This cap
chain-trains via `--no-policy-loss` ECHO-only on the Phase-2 best
checkpoint. Don't run standalone.

## Quickstart

```bash
python3 build_corpus.py
BASE_ADAPTER=echo-tblite-phase2-iter-best ./run_iter.sh h-vf-adaptation
```

## Round-2 plan

1. Land pi-terminal-bench-lite Phase 2 first.
2. Use that as `--base-adapter`.
3. Run paper's §5.5 100-step recipe.
4. Eval on val100/ITD/PyTerm/TBLite — expect paper's deltas
   (+3.8/+5.2/+10.0/-3.9 pp).
