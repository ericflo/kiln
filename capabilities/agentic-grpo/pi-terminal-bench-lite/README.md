# pi-terminal-bench-lite

Paper-track integration cap. **Round 1: mechanical validation complete;
paper-scale experiment NOT run.** Round-2 priority Tier 1: actually run it.

## Status (round 2)

| File | Status |
|------|--------|
| `capability.md` | Spec + round-2 plan (run paper-scale) |
| `rubric.py` | Multi-turn paper rubric |
| `calibration/` | Workdir-dependent; bypassed |
| `archive/` | Round-1 mechanical validation: ECHO firing, --no-policy-loss verified |

## Round-2 plan

1. Build paper §4 task corpus.
2. Phase 1 (GRPO + ECHO) → Phase 2 (`--no-policy-loss` verifier-free).
3. Eval on val100, ITD, PyTerm, TBLite (with TBLite as negative control).
4. Triple-seed the headline (round-1 lesson).

This cap should produce the **round-2 publishable headline**. Round-1
pi-doctest gives a small reproducible win on narrow task; this should
give paper-quality multi-eval reproduction.

## Quickstart

```bash
python3 build_corpus.py        # builds val100, ITD, PyTerm, TBLite
./capability.oracle.sh
./run_iter.sh phase1-grpo-echo
BASE_ADAPTER=tblite-phase1-best ./run_iter.sh phase2-no-policy-loss
```
