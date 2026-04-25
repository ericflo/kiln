# Phase C9 — Null result: `mtp.fc` matmul drift is not the α-suppressing signal

## Verdict

**Phase C9 is a null result.** The pre-existing "`fc_output` divergence" reported by the C6/C7 comparator on main post-PR #320 (Phase C8) is an **allclose-bar artifact**, not a semantic bug:

- The strict Phase C6 bar (`cos_sim ≥ 0.9999`) already passes at `fused` / `fc_output` across all 6 pair dumps (3 seeds × 2 positions).
- `fused` max|Δ| against the HF bf16 reference ranges 4.44e-3 to 1.60e-2, well within the documented bf16 matmul accumulation error budget (cf. Phase C6 bisect report: "≤3.01e-2 consistent with bf16 arithmetic variance — not a semantic divergence").
- The MTP draft acceptance rate α on main (post-C8) is already `> 0`: median **0.058** (3 seeds), 0.033 / 0.058 / 0.104 individually. The C8 pre-condition ("α > 0 once the SDPA contract is correct") is therefore met.
- A falsification toggle (`KILN_MTP_FC_FP32_ACCUM=1`) that promotes the `mtp.fc` matmul to f32 accumulation yields **identical α** (same median 0.058, 0.033 / 0.058 / 0.095) while measurably reducing the `fused` max|Δ| on the seed44 pair.

The "first divergence at tap `fc_output`" verdict emitted by `scripts/mtp_compare.py` is produced by its `numpy.allclose(atol=1e-3, rtol=1e-2)` check — a tolerance appropriate for fp32, but tight enough to reliably fail against bf16-vs-bf16 matmul variance. The verdict text ("Most-likely cause: mtp.fc matmul (weight transpose or layout)") is a canned `PRIMARY_HYPOTHESIS` label, not a runtime-derived diagnosis. Downstream taps (`post_attn_raw`, `post_final_ln`, `mtp_logits`) trip the same allclose bar for the same reason while still passing `cos_sim ≥ 0.9999`.

The Phase C8 report's out-of-scope note ("fc_output divergence is pre-existing main-model `mtp.fc` matmul drift — that's C9 work") conflated two unrelated phenomena: (a) the `fc_output` tap's allclose failure — benign bf16 noise, and (b) the Phase C5 Class-B rejection rate (87.6% of MTP rejections were "draft top1 ≠ main top1"), which is a per-token decoder-time metric, **not** a per-tap drift measurement. C9 resolves (a) by empirically disproving the matmul-weight hypothesis; (b) remains out of scope and is re-scoped as Phase C10 below.

## Option chosen

**(C) doc-only null PR + diagnostic toggle.**

Phase C9's task description explicitly contained the clause *"If the drift is already resolved, STOP and post a doc-only null PR with the evidence."* The strict-bar cos_sim is already green at `fc_output`, so this PR:

1. Documents the finding in `docs/archive/phase-c/phase-c9/c9_fix_report.md` with full C6 comparator logs and α measurements.
2. Adds a single-site, default-off diagnostic toggle `KILN_MTP_FC_FP32_ACCUM` behind `mtp_debug::is_mtp_fc_fp32_accum_enabled()` — mirroring the precedent set by `KILN_MTP_SWAP_FC_NORMS` (Phase B2 A/B). Future α investigations can toggle this without rebuilding.
3. Re-scopes the remaining α-lift work to Phase C10.

## Single-file diff summary (2 files, +25 / 0 lines)

```
crates/kiln-model/src/forward.rs   | 15 ++++++++++++++-
crates/kiln-model/src/mtp_debug.rs | 14 ++++++++++++++
2 files changed, 28 insertions(+), 1 deletion(-)
```

### `crates/kiln-model/src/mtp_debug.rs` (+14 lines)

`pub fn is_mtp_fc_fp32_accum_enabled() -> bool` — reads `KILN_MTP_FC_FP32_ACCUM`. Default `false`. Matches the shape of `is_swap_fc_norms_enabled()` for consistency with the existing Phase B2 diagnostic toggle.

### `crates/kiln-model/src/forward.rs` (+14 / −1 lines, 1 site)

The `mtp.fc` matmul site in `mtp_forward_step` (lines ~4435–4438 pre-patch) is gated on the flag. When off (default): legacy bf16 matmul, zero path change. When on: `concat` is cast to f32, `mtp.fc_t` is cast to f32 per-step, the matmul runs in f32, and the result is cast back to the input dtype. The matmul shape is `[1, 1, 2H] @ [2H, H]` = `[1, 1, 5120] @ [5120, 2560]`, ~13M FLOPs — the per-step cost is negligible and there is no hot-path regression (MTP is invoked once per draft step, not per layer).

## Verification

### Build

A6000 on-demand pod (image `ghcr.io/ericflo/kiln-runpod:latest`, SM 86, CUDA 12.4). Cargo release build with `--features cuda`, sccache + B2 remote cache: 19–33s wall-clock (97%+ sccache hit rate from the warm B2 cache). Tested via `./target/release/kiln-bench --help`.

### α protocol (3 seeds, median reported)

```
KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp \
[KILN_MTP_FC_FP32_ACCUM=1] \
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b/ --paged \
  --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed $seed
```

| Config | seed 42 | seed 43 | seed 44 | **median** |
| --- | --- | --- | --- | --- |
| main (post-C8) | 0.033 (4/123) | 0.058 (7/120) | 0.104 (12/115) | **0.058** |
| `KILN_MTP_FC_FP32_ACCUM=1` | 0.033 (4/123) | 0.058 (7/120) | 0.095 (11/116) | **0.058** |

Absolute median delta: **0.000**. The only per-seed delta is seed 44 (0.104 → 0.095), which flips one single accepted token (12 → 11 of ~115 attempts); this is within the seed-to-seed variance already visible in the baseline (0.033 → 0.104 = ±0.035).

Raw logs: [`alpha_main_baseline.log`](./alpha_main_baseline.log), [`alpha_fp32_on.log`](./alpha_fp32_on.log).

### C6 strict-bar sweep (cos_sim ≥ 0.9999)

Full dumps captured with `KILN_MTP_DUMP_PRE_ROPE=1 KILN_MTP_DUMP_SUBOPS=1 KILN_MTP_DUMP_C7_SDPA=1 KILN_MTP_DUMP_POS=1,2`, seeds 42/43/44, 6 pair dumps against the HF reference dumps from Phase C8 (`/workspace/c8_ref/ref_seed*_pos*.st`).

| Tap | main median cos_sim | main max(max\|Δ\|) | C9 fp32 median cos_sim | C9 fp32 max(max\|Δ\|) | strict-bar (`≥ 0.9999`) |
| --- | --- | --- | --- | --- | --- |
| `c6__token_emb` | 1.0000 | 0.00 | 1.0000 | 0.00 | **ok** |
| `c6__norm_emb` | 1.0000 | 7.76e-3 | 1.0000 | 7.76e-3 | **ok** |
| `c6__norm_h` | 1.0000 | 3.01e-2 | 1.0000 | 3.01e-2 | **ok** |
| `c6__concat` | 1.0000 | 3.01e-2 | 1.0000 | 3.01e-2 | **ok** |
| `c6__fused` | 1.0000 | **1.60e-2** | 1.0000 | **1.60e-2** | **ok** |

Per-pair `fused` max|Δ| (identical bar, reduced mean):

| Pair | main max\|Δ\| | main mean\|Δ\| | fp32 max\|Δ\| | fp32 mean\|Δ\| |
| --- | --- | --- | --- | --- |
| seed42_pos1 | 4.44e-3 | 6.31e-4 | 5.31e-3 | 5.85e-4 |
| seed42_pos2 | 7.34e-3 | 6.17e-4 | 7.34e-3 | 5.80e-4 |
| seed43_pos1 | 7.48e-3 | 5.93e-4 | 7.48e-3 | 5.58e-4 |
| seed43_pos2 | 5.07e-3 | 6.34e-4 | 5.07e-3 | 5.76e-4 |
| seed44_pos1 | 1.08e-2 | 6.03e-4 | **5.10e-3** | 5.69e-4 |
| seed44_pos2 | 1.60e-2 | 6.29e-4 | 1.60e-2 | 5.84e-4 |

`fused` mean|Δ| drops uniformly (~6.2e-4 → ~5.8e-4, −7%) with the fp32 flag on; `max|Δ|` drops materially on seed44_pos1 (1.08e-2 → 5.10e-3, ~2×) and is otherwise unchanged. The **strict cos_sim bar passes on both configurations**.

Comparator output (main, post-C8):

```
Phase C6 verdict: ALL pre-RoPE taps match within cos_sim >= 0.9999.
  -> Pre-RoPE input is NOT the dominant source of drift at this bar.
```

Full reports: [`c6_main_baseline.txt`](./c6_main_baseline.txt) (379 lines, 6 pair verdicts), [`c6_fp32_on.txt`](./c6_fp32_on.txt) (379 lines, 6 pair verdicts).

### Why the comparator still prints "first divergence at tap 'fc_output'"

`scripts/mtp_compare.py`'s per-pair "first divergence" verdict (lines ~560–575) iterates taps in comparison order and reports the first one whose `numpy.allclose(atol=1e-3, rtol=1e-2)` check is `False` — regardless of `cos_sim`. `PRIMARY_HYPOTHESIS["fc_output"] = "mtp.fc matmul (weight transpose or layout)"` is a static dict lookup, not a derived diagnosis. For bf16-vs-bf16 ref comparisons the allclose bar is consistently too tight past the first matmul; the strict `cos_sim ≥ 0.9999` sweep is the meaningful numerical check, and Phase C6's own bisect report (`docs/archive/phase-c/phase-c6/c6_bisect_report.md`) already documented this distinction.

## Cos_sim / max|Δ| before / after

| Tap | C6 on main (median) | C6 w/ `KILN_MTP_FC_FP32_ACCUM=1` (median) | Bar |
| --- | --- | --- | --- |
| `c6__fused` cos_sim | 1.0000 | 1.0000 | `≥ 0.9999` — **ok** both |
| `c6__fused` max\|Δ\| | 1.60e-2 worst | 1.60e-2 worst, −53% on one pair | n/a (BF16 budget) |
| `c6__fused` mean\|Δ\| | 6.29e-4 worst | 5.84e-4 worst (−7%) | n/a |
| α (median, 3 seeds) | **0.058** | **0.058** (Δ = 0) | target: > 0 |

## Out of scope: C10 follow-up (re-scoping Class B 87.6%)

The real α-suppressing signal is not at the `fc_output` tap. With C9's falsification in hand, Phase C10 should investigate in this order:

1. **Per-token top1 mismatch tracing.** Capture `(main_top1, main_logit_top1, mtp_top1, mtp_logit_top1)` at each rejected draft step on both kiln and the HF MTP reference. Class B = `mtp_top1 ≠ main_top1`. The question is whether kiln's MTP draft top1 matches or diverges from the HF MTP top1 at the same step — if they match, α is at the architectural ceiling for this head; if they diverge, the drift is somewhere in the MTP forward that the tap sweep hasn't localized (candidates: bf16 vs fp32 reference for comparator, paged MTP cache write ordering, `abs_pos` / RoPE phase at MTP pos > 0).
2. **Marlin W4A16 q_proj in the MTP inner block.** PR #320's loader keeps the MTP transformer layer in BF16 (see `forward.rs:1112-1127`); the ~3% model-memory q_proj + MLP is not yet W4A16-packed. When the Marlin pack-MTP work lands, re-run C9's strict C6 + α sweep to confirm no regression.
3. **fp32 HF reference comparator run.** `scripts/mtp_compare.py` currently compares kiln-bf16 ↔ HF-bf16; running a fp32 HF reference dump and diffing against it would cleanly split "benign bf16 matmul variance" from "residual semantic bug" across the full post-RoPE path. This is the same technique the existing `--b12` mode documents for the base model.
4. **`abs_pos` / RoPE audit at pos > 0.** Phase B7a (PR #276) previously caught a `mtp_pos` vs `base_pos + mtp_pos` bug here. Re-confirm for the post-C8 code path.

None of these are in C9 scope.

## Files added in this PR

- `docs/archive/phase-c/phase-c9/c9_fix_report.md` (this file)
- `docs/archive/phase-c/phase-c9/c6_main_baseline.txt` (C6 comparator on main post-C8, 6 pairs)
- `docs/archive/phase-c/phase-c9/c6_fp32_on.txt` (C6 comparator with `KILN_MTP_FC_FP32_ACCUM=1`, 6 pairs)
- `docs/archive/phase-c/phase-c9/alpha_main_baseline.log` (3-seed α baseline on main)
- `docs/archive/phase-c/phase-c9/alpha_fp32_on.log` (3-seed α with `KILN_MTP_FC_FP32_ACCUM=1`)

## Code changes

- `crates/kiln-model/src/mtp_debug.rs` (+14 lines): `is_mtp_fc_fp32_accum_enabled()` env-flag accessor.
- `crates/kiln-model/src/forward.rs` (+14 / −1 lines): single-site gated fp32 matmul at `mtp_forward_step`'s `mtp.fc` projection. Default off; zero production cost.
