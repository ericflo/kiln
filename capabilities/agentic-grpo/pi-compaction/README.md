# pi-compaction

Long-context conversation/transcript compaction. **Round-1 result:
byte-identical adapter outputs (no-op).** Was blocked on kiln
long-context training. Round 2 unblocked via kiln #25-28.

## Status (round 2)

**DEPENDENCY-GATED.** Wait for:
1. kiln #25 long-context bench to confirm weights move at 32K.
2. Then: switch this cap to **OPD from 27B** (GRPO can't drive
   summarization quality with the noise levels at long context).

## Read first

1. [`capability.md`](capability.md).
2. [`../../LAYOUT.md`](../../LAYOUT.md).

## Files

| File | Status |
|------|--------|
| `capability.md` | Spec + round-2 plan (dependency-gated) |
| `rubric.py` | Long-context summarization rubric |
| `calibration/` | 5 good + 25 bad (existing); separation TBD post-rubric-fix |
| `archive/` | Round-1 iter-artifacts, kiln-polish issues |

## Round-2 plan

1. Run kiln #25 long-context bench at 32K. Confirm `lora_delta_norm_summary`
   non-zero.
2. Use kiln #27 byte-identical-adapter diagnostic if weights *don't* move.
3. Once weights move: switch to `cuda_opd_remote` (verifier-free fits poorly
   for noisy long-context).

## Quickstart

```bash
# Don't run training yet — first verify weights move:
KILN_CUDA_ARCHS=86 cargo run -p kiln-train --features cuda --example long_context_bench -- --tokens 32768
# Then:
./run_iter.sh h1-opd-long-context
```
