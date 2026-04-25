# Phase C18 — Fix `h_prev` reference frame (post-final-norm)

## Verdict

**Partial recovery. C17's reference-frame hypothesis is directionally confirmed — α
improved 4.7× (median 0.033 → 0.153) — but α is still below the 0.5 minimum floor,
so additional bugs remain downstream. Ship C18 and hand off the residual α gap to
Phase C19.**

## Context

Phase C17 ([PR #340](https://github.com/ericflo/kiln/pull/340)) established that
`h_prev` was being returned to the MTP head in the **pre-final-norm** frame,
whereas the vLLM / SGLang `last_hidden_state` contract requires the
**post-final-norm** frame. C15 had already measured a 2.0–2.4× magnitude
ratio between kiln's `h_main` and HF's `hs[-1]`, exactly the signature of one
RMSNorm being missing.

This task implements the fix and measures its α impact in isolation.

## Change summary

Branch: `ce/mtp-phase-c18-h-prev-post-norm` — commit `dd1cdef`.

### `crates/kiln-model/src/forward.rs`

Both `LmHeadMode::FullWithLastHidden` and `LmHeadMode::LastRowWithLastHidden`
arms now compute the final RMSNorm once and return the post-norm last-row as
`h_prev`. The math-cheaper variant is used: one `rms_norm(hidden)` call whose
output both feeds `lm_head` and (sliced) becomes `last_hidden`.

```rust
LmHeadMode::FullWithLastHidden => {
    let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
    let last_hidden = normed.narrow(1, seq_len - 1, 1)?.contiguous()?;
    // capture tap: "h_post_final_norm"
    let logits = normed.broadcast_matmul(&weights.embed_tokens_t)?;
    Ok((Some(logits), Some(last_hidden)))
}
LmHeadMode::LastRowWithLastHidden => {
    let last_pre_norm = hidden.narrow(1, seq_len - 1, 1)?.contiguous()?;
    let last_hidden = rms_norm(&last_pre_norm, &weights.final_norm, config.rms_norm_eps)?;
    // capture tap: "h_post_final_norm"
    let logits = last_hidden.broadcast_matmul(&weights.embed_tokens_t)?;
    Ok((Some(logits), Some(last_hidden)))
}
```

### `crates/kiln-model/src/mtp_debug.rs`

Renamed the h_main tap from `h_pre_final_norm` → `h_post_final_norm` (docstrings
+ the `h_main_capture_records_then_drains` test).

### `scripts/mtp_reference_dump.py`, `scripts/c15_h_main_drift_audit.py`

Updated docstrings / inline comments to reflect the new contract. HF's
`output_hidden_states[-1]` is already post-final-norm, so no reference-side
conversion is required.

## Call-site audit

The two callers of `model_forward_paged_with_last_hidden` both consume `h_prev`
as MTP context:

- `crates/kiln-server/src/bench.rs:1034` — MTP bench path.
- `crates/kiln-model/src/speculative.rs:585, 680` — production MTP decode.

Neither depends on the pre-norm frame. No non-MTP callers exist.

## α recovery benchmark

A6000, `KILN_SPEC_METHOD=mtp`, `KILN_W4A16=1`, CUDA graphs ON, prompt 512 tok,
decode 128 tok, 3 seeds. See `alpha-pre.log` and `alpha-post.log` in this
directory for raw output.

### Pre-fix (main merge `6b5aa07`)

| seed | α (accepted / drafted) | decode tok/s |
| --- | --- | --- |
| 42 | 0.0242 (3/124) | 42.2 |
| 43 | 0.0325 (4/123) | 42.2 |
| 44 | 0.1043 (12/115) | 42.2 |
| **median** | **0.0325** | **42.2** |

### Post-fix (C18 branch `dd1cdef`)

| seed | α (accepted / drafted) | decode tok/s |
| --- | --- | --- |
| 42 | 0.1140 (13/114) | 42.4 |
| 43 | 0.1532 (17/111) | 42.8 |
| 44 | 0.1869 (20/107) | 43.4 |
| **median** | **0.1532** | **42.85** |

### Recovery

- **α: 0.0325 → 0.1532 (median)** — **4.7× multiplicative**, +0.12 absolute.
- **Decode tok/s: 42.2 → 42.85** — essentially unchanged. The single extra
  `rms_norm` call in the `LastRowWithLastHidden` arm is fully amortized under
  CUDA graph replay. No measurable regression.
- Every seed's post-fix α exceeds every seed's pre-fix α.

## Interpretation vs. ship floors

| Floor | Threshold | C18 median α | Status |
| --- | --- | --- | --- |
| Minimum (must-pass) | α ≥ 0.5 | 0.1532 | **below floor** |
| Clean-ship | α ≥ 0.72 | 0.1532 | **below floor** |

The 4.7× recovery confirms C17's directional analysis: `h_prev` reference-frame
was the single largest pre-decoding bug. The remaining gap to 0.5+ means **at
least one more bug still sits between `h_prev` and accepted draft tokens**.
Candidates are tracked in the Phase C19 handoff below.

## Phase C19 handoff — remaining hypotheses

Per C17's first-divergence audit, the following were not disproven as
secondary contributors. With `h_prev` now in the correct frame, these become
the new head of the queue and should be tested one at a time, in the order
that each has the highest math-ceiling × cheapest test cost.

1. **`mtp.fc_norm` vs. `model.norm` reuse.** HF Transformers instantiates a
   *separate* RMSNorm (`fc_norm`) inside the MTP head and applies it to
   `h_main` before the `fc` projection. Kiln currently reuses the main
   final-norm weights for both. Swap to the dedicated `fc_norm` tensors from
   the checkpoint and re-bench α.
2. **Dual-norm inversion inside the MTP block.** The MTP decoder block has its
   own pre-attn / pre-MLP RMSNorms. If kiln's `MtpBlock` currently applies
   these in the wrong order (or substitutes the main-model norms), token
   logits will be subtly miscalibrated but still on-manifold — exactly the
   α ≈ 0.15 shape.
3. **Rotary `mtp_pos` offset.** MTP inner-block rotary should receive
   `pos + 1` (the next position) relative to the main block's rotary at `pos`.
   If kiln feeds the same position index to both, the rotary frequency is
   off-by-one and draft tokens will consistently miss.
4. **Gate / residual parameterization.** MTP splices
   `h_prev` ⊕ `e(token_prev)` through an `fc` linear + residual gate. Check
   the gate parameterization and the `fc` weight slicing against HF's reference.
5. **Draft token sampling.** Confirm the draft-side sampler uses temperature
   and top-p that match main-side semantics. A sampler mismatch (e.g. greedy
   on draft + sampled on main) manifests as α < 0.2 with the full prefix
   pipeline otherwise correct.

The B6 inner-MTP splice parity harness already exists (`scripts/mtp_compare.py`
+ `scripts/mtp_h_main_reference_dump.py`) and can be retargeted at each
hypothesis in turn without additional GPU time for the first two (CPU-only
tensor comparisons against HF).

## Cost / budget

- Wall-clock on pod: ~35 min (build 12 min sccache-warm, bench 3×2 min pre-fix,
  rebuild 5 min incremental, bench 3×2 min post-fix, wait-file polls).
- Pod cost envelope: well under the 120 min / $50 cap.
- No SSH polling loops used — all long-running commands supervised via
  `runpod_api.py wait-file` per the kiln RunPod workflow.
