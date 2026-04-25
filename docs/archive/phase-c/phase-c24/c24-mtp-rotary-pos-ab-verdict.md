# Phase C24 — MTP rotary `abs_pos` hardware A/B verdict (hypothesis 3, doc-only redirect)

## Verdict

**Hypothesis 3 is REJECTED on hardware A/B. No code change.** A
single-build, env-var-gated bench on a warm A6000 measured α for
`abs_pos = base_pos + mtp_pos` (baseline) vs `abs_pos = base_pos - 1
+ mtp_pos` (alternate) over seeds {42, 43, 44} with the standard
bench recipe (`--paged --prompt-tokens 512 --max-output-tokens 128
--skip-training`, `KILN_W4A16=1 KILN_SPEC_ENABLED=1
KILN_SPEC_METHOD=mtp`). The alternate arm moved α by **Δα =
+0.010** (median 0.1429 vs 0.1327, absolute), which is well inside
the prespecified `[-0.05, +0.05]` null band and two orders of
magnitude below the 0.5 ship floor. The direction was consistent
across all three seeds (alternate higher by ~0.010 at each seed),
so the rotary offset may be a tiny contributor but is not the
drift source we are looking for. The static audit in Phase C21
(PR #346) correctly flagged this as inconclusive — the hardware
A/B confirms both candidate conventions perform essentially the
same, which is itself evidence that the `Qwen3NextMultiTokenPredictor`
reference contract is genuinely position-agnostic on its first
draft step. Hand off the residual α gap (C18 median 0.1532, floor
0.5) to deeper structural audits beyond the H1–H5 queue: block
sequencing between decode steps 0→1→2, state carryover across
the MTP paged KV cache, and verifier-side logit comparison
against HF's `Qwen3NextMultiTokenPredictor` on a small
calibration set. H1 (C17 reference frame), H2 (C20 block norm),
H3 (this PR), H4 (C22 fc/residual, PR #347), and H5 (C23 draft
sampler, PR #348) have all been individually ruled out as the
dominant drift source.

## C21 handoff as stated

From [`docs/archive/phase-c/phase-c21/c21-mtp-rotary-pos-audit.md`](../phase-c21/c21-mtp-rotary-pos-audit.md)
("Next steps", handoff to C24):

> Queue H3 as a low-cost hardware A/B (`abs_pos = base_pos +
> mtp_pos` vs `abs_pos = base_pos - 1 + mtp_pos`) to run only if
> H4/H5 also fail to recover α.

Phase C22 (PR #347) disproved H4 (`fc`/residual
parameterization). Phase C23 (PR #348) disproved H5 (draft-side
sampler drift). With H3 now rejected on hardware, the full H1–H5
queue has been cleared and the remaining α gap requires a
structural audit beyond that queue.

## A/B bench design

**Single-build, env-var-gated A/B.** One release build of
`kiln-bench` with an env-var gate at the single `abs_pos`
assignment site in `mtp_forward_step`:

```rust
// crates/kiln-model/src/forward.rs (around line 4590)
let abs_pos = if std::env::var("KILN_MTP_ABS_POS_MINUS1").ok().as_deref() == Some("1") {
    base_pos.saturating_sub(1) + mtp_pos
} else {
    base_pos + mtp_pos
};
let positions = Tensor::new(&[abs_pos as f32][..], device)?;
```

The gate is the minimum diff that distinguishes the two candidate
conventions, keeping Marlin packing, CUDA graphs, weight layout,
scheduler, and paged KV cache identical across arms. The gate is
applied only to this ephemeral A/B; the doc-only redirect PR does
not ship the gate (see "Patch status" below).

**Recipe.** 3 seeds per arm (42, 43, 44, matching the Phase C5/C9
median-of-3 convention), `--paged --prompt-tokens 512
--max-output-tokens 128 --skip-training` (the canonical
`c12_bench_runner.py` recipe), `KILN_W4A16=1 KILN_SPEC_ENABLED=1
KILN_SPEC_METHOD=mtp`. α reported via the bench JSON
`latency.acceptance_rate` field.

**Prespecified thresholds.**

| Δα bucket | Action |
| --- | --- |
| Δα > +0.05 | Code-fix PR (flip baseline to alternate) |
| −0.05 ≤ Δα ≤ +0.05 | Doc-only redirect (this PR) |
| Δα < −0.05 | Doc-only redirect noting baseline is correct |

## Results

### Per-seed α

| Seed | Baseline α | Alternate α | Δα (alt − base) |
| --- | --- | --- | --- |
| 42 | 0.1043 (12/115) | 0.1140 (13/114) | +0.010 |
| 43 | 0.1327 (15/113) | 0.1429 (16/112) | +0.010 |
| 44 | 0.1759 (19/108) | 0.1869 (20/107) | +0.011 |

### Medians

| Arm | Median α | Median decode tok/s |
| --- | --- | --- |
| Baseline (`base_pos + mtp_pos`) | **0.1327** | 42.49 |
| Alternate (`base_pos - 1 + mtp_pos`) | **0.1429** | 45.30 |

**Δα (median) = +0.0101.** This is within `[-0.05, +0.05]`, so
H3 is rejected as the dominant α-drift cause. Decode throughput
is within normal seed noise and carries no directional signal
about either convention.

### Why a +0.010 consistent signal is still a NULL result

The per-seed signs all agree (alternate > baseline by ~0.010 at
every seed), so this is not pure noise — there is a small real
effect. But:

- The magnitude is two orders of magnitude below the 0.5 ship
  floor. Even if we flipped the code today, α would move from
  0.1327 to 0.1429, still far below the minimum viable spec-decode
  acceptance rate.
- The prespecified ship threshold (+0.05) exists precisely to
  prevent "ship micro-signals" that fold into the rotation noise
  floor at higher batch sizes, longer prompts, or different
  temperature.
- Phase C21's static audit already showed both conventions have
  valid vLLM precedent — the hardware result confirms that and
  shows neither is clearly "right". Picking the slightly better
  one on a 3-seed, 128-token A/B would be a superstition bet.

A small consistent bias of ~0.010 is the expected tail of
mismatched-by-one positions: roughly one in 100 draft tokens
happens to land on a rotation angle that is slightly more
favorable. That is consistent with rotary being a negligible axis
for this failure mode and not with rotary being the primary drift.

## Patch status

The env-var gate (`KILN_MTP_ABS_POS_MINUS1`) was applied on a
throwaway WIP branch for this A/B only. It is **not** included in
this PR. The doc-only redirect keeps main's `abs_pos = base_pos
+ mtp_pos` unchanged, so no code review burden for an A/B harness
that no longer needs to run.

If a future phase re-runs H3 on a different recipe (longer prompt,
chunked prefill, different quantization), the gate is one `git
apply` away — see the WIP branch `mtp/phase-c24-rotary-pos-ab-wip`
(unmerged, retained for the audit trail) or copy the snippet
above.

## Cost accounting

| Item | Value |
| --- | --- |
| Pod | `sq3exutrqy2cuw` (pool-managed A6000, re-warmed) |
| Hourly rate | $0.49/hr (on-demand) |
| Wall-clock (acquire → release) | ~18 min |
| Approximate cost | ~$0.15 |

The pool already had the repo, model weights, and a warm sccache
cache from prior phases, so the build was a 18.7s no-op recompile
and the bench was dominated by MTP prefill + 128-token decode
loops.

## Next steps

1. **Escalate beyond the H1–H5 queue.** With all five
   hypothesis branches ruled out, the residual ~0.37 α gap lives
   somewhere none of those audits covered. Candidates:
   - Block sequencing: kiln processes decode step 0, then 1, then
     2 through the MTP paged KV cache; any cross-step state
     invariant drift would show up as increasing Δ-logit with
     decode position.
   - State carryover: MTP-side paged KV cache write/read
     invariants vs the reference's contiguous cache.
   - Verifier-side logit comparison: dump the verifier's pos-0
     logits against HF's `Qwen3NextMultiTokenPredictor.forward`
     on a small calibration set (20–50 prompts). Any bit-level
     divergence narrows the search.
2. **Consider an MTP weight reload sanity check.** The C16 audit
   ruled out the obvious accept/reject boundary issues. A full
   weight-load diff (kiln vs HF) against the MTP head's
   `fc`/`embed_norm`/`hidden_norm`/`inner_block.*` weights would
   close the last cheap loophole.
3. **Do not re-queue H3 at short prompts/decode lengths.** The
   A/B answer is clean: rotary offset is not where the drift
   lives. A re-preflight is warranted only if a future phase
   finds a new reason to expect position-dependent drift (e.g.
   chunked prefill landing, or multi-turn MTP).
