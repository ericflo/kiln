# Phase C16 — MTP Accept/Reject Plumbing Audit

**Scope (from C15 handoff, `docs/archive/phase-c/phase-c15/c15-h-main-drift-verdict.md` §"Handoff to C16")**: audit the path from validated-stable `h_main` through MTP head-0 output to the accept/reject decision, covering four candidate α-collapse causes downstream of `h_main`:

- **H1**  MTP head-0 `mtp_logits` indexing — is the correct logit being sampled?
- **H2**  `mtp_advance` accounting in `bench_latency_paged_mtp` — does `mtp_pos` advance on the correct side of the accept/reject branch?
- **H3**  Draft-token positional KV wiring — is the accepted-token's KV entry written to the slot the next forward reads?
- **H4**  KV-rollback semantics on REJECT — does the rollback leave a consistent cache, or is it inserting an off-by-one that permanently biases `pos=0`?

**Method**: static code audit of the current accept/reject code path under `KILN_SPEC_METHOD=mtp` (Qwen3.5-4B native k=1 MTP), cross-referenced with the Phase C1 attribution CSV schema that already captures the greedy invariant. A Python verifier (`scripts/c16_plumbing_analyze.py`, new in this PR) replays any existing C1 CSV and mechanically asserts the H1/H2/H3 invariants per row and per consecutive-row transition.

**No runtime behaviour change** — this is a doc + script PR. The existing `KILN_C1_ATTR_PATH` sink (`crates/kiln-model/src/c1_attr.rs`, shipped in Phase C1) already emits every field the four hypotheses need; adding a second runtime diagnostic path was the wrong tool for this question.

## Verdict

**All four hypotheses REJECTED.** The α-collapse root cause is not in the accept/reject plumbing.

| Hypothesis | Verdict | Basis |
| --- | --- | --- |
| H1 | REJECTED | Greedy top-1 compare: `accepted` is set by `draft_accepted = target_at_0 == draft_token`. `topk_match` is computed as `draft_token == target_at_0` from the same two tokens. The two are identical expressions on the same `u32` values; they cannot disagree without a mutation between lines 609 and 641 of `speculative.rs`. None exists. |
| H2 | REJECTED | `bench.rs:1131–1132` threads `base_pos += step.base_advance; mtp_pos += step.mtp_advance`. `speculative.rs:670–716` sets `(base_advance, mtp_advance) = (2, 1)` on ACCEPT and `(1, 0)` on REJECT. C15's observation that `mtp_pos` never advanced is a **consequence** of α=0.000, not its cause. |
| H3 | REJECTED | Split-verify writes base KV at `base_pos` on every step and at `base_pos + 1` **only** on ACCEPT (conditional second `model_forward_paged_with_last_hidden` call at `speculative.rs:679–691`). On REJECT, no stale `base_pos + 1` is written — `base_advance = 1` is the correct slot delta. The MTP cache write happens inside `mtp_forward_step` at slot `mtp_pos`; on REJECT the slot is re-overwritten next iteration because `mtp_advance = 0`. |
| H4 | N/A (not a bug surface) | Split-verify k=1 greedy has **no rollback path**. REJECT never runs the draft through the base model (that forward is gated behind `if draft_accepted`, `speculative.rs:670`), so there is no base-cache state to undo. The MTP cache entry from the draft forward stays in place intentionally — it is the next step's write target since `mtp_pos` is pinned. There is no off-by-one window where the base cache holds a stale post-draft slot. |

## Evidence — static audit

### Accept/reject core (`crates/kiln-model/src/speculative.rs`, lines 538–725)

The function is `speculative_mtp_decode_step`. The two flow-controlling branches that matter for C16:

**Draft + verify (both paths take this)**, lines 565–604:

```rust
// 1. Draft
let (mtp_logits, _mtp_hidden) = mtp_forward_step(
    backend, last_token, h_prev, weights, config,
    mtp_cache, mtp_block_table, base_pos, mtp_pos,
)?;
let draft_token = greedy_sample(&mtp_logits)?;

// 2. Verify pos 0 only — first forward of split verify
let (verify_logits0, hidden_after_last) = model_forward_paged_with_last_hidden(
    backend, &[last_token], weights, config,
    base_cache, base_block_table, base_pos,
    Some(linear_state), None, None,
)?;
let verify_pos0 = verify_logits0.squeeze(1)?;
let target_at_0 = greedy_sample(&verify_pos0)?;
```

**Accept decision**, line 609:

```rust
let draft_accepted = target_at_0 == draft_token;
```

This is the sole assignment that drives the accept/reject branch. Under greedy `top_k=1`, `target_at_0` and `draft_token` are both `u32` tokens returned by `greedy_sample` on the respective logit tensors.

**C1 attribution capture**, lines 630–642 (unchanged in this PR):

```rust
c1_attr::push_row(c1_attr::C1Row {
    step_idx: c1_attr::next_step_idx(),
    pos_in_k: 0,
    base_pos,
    mtp_pos,
    last_token,
    mtp_top1: draft_token,
    mtp_top1_logit,
    main_top1: target_at_0,
    main_top1_logit,
    accepted: draft_accepted,
    topk_match: draft_token == target_at_0,   // same two u32 values
});
```

`accepted` and `topk_match` are computed from the same two `u32` values with no intervening mutation. Under greedy they are identical by construction. **H1 is refuted by the source itself.**

**Advance counters**, lines 670–716:

```rust
let mut new_h_prev = hidden_after_last;
let (base_advance, mtp_advance) = if draft_accepted {
    // ACCEPT: draft_token matches target. Emit [draft, bonus].
    if eos_token_ids.contains(&draft_token) {
        hit_eos = true;
        (1, 1)
    } else {
        accepted_tokens.push(draft_token);
        let verify_input1 = [draft_token];
        let (verify_logits1, hidden_after_draft) = model_forward_paged_with_last_hidden(
            backend, &verify_input1, weights, config,
            base_cache, base_block_table, base_pos + 1,
            Some(linear_state), None, None,
        )?;
        new_h_prev = hidden_after_draft;
        let bonus = greedy_sample(&verify_logits1.squeeze(1)?)?;
        if eos_token_ids.contains(&bonus) { hit_eos = true; }
        else { accepted_tokens.push(bonus); }
        (2, 1)
    }
} else {
    // REJECT: emit the target's token at position 0.
    if eos_token_ids.contains(&target_at_0) { hit_eos = true; }
    else { accepted_tokens.push(target_at_0); }
    (1, 0)
};
```

Three things to call out:

- `(base_advance, mtp_advance) = (2, 1)` on ACCEPT and `(1, 0)` on REJECT. **H2 is correct.**
- The second base forward at `base_pos + 1` runs only inside the `if draft_accepted` block. On REJECT, the base cache has only written `base_pos`. **H3 is correct** — there is no stale `base_pos + 1` write that needs to be rolled back.
- The MTP cache write happens once per call, inside `mtp_forward_step(..., mtp_cache, mtp_block_table, base_pos, mtp_pos)` at line 565. On REJECT, the next call arrives with the same `mtp_pos` (because `mtp_advance = 0`), so the MTP cache slot is overwritten rather than rolled back. This is the intended k=1 MTP semantics — not a bug.

### Bench decode loop (`crates/kiln-server/src/bench.rs`, lines 1086–1138)

The caller correctly threads both advance values:

```rust
let step = speculative_mtp_decode_step(...)?;
// ...
base_pos += step.base_advance;
mtp_pos  += step.mtp_advance;
h_prev    = step.new_h_prev;
```

`base_cache` and `mtp_cache` are independently allocated `PagedKvCache` instances (lines 987–1013), each with its own `BlockTable`. They never share address space. **H3's "wrong cache" failure mode is architecturally impossible here** — you'd need to mis-wire the constructor arguments, which are grep-checkable and match type by type.

### C1 attribution schema (`crates/kiln-model/src/c1_attr.rs`, lines 32–58)

Every field the plumbing audit needs is already in the CSV — no new runtime instrumentation required:

| Field | Purpose for C16 |
| --- | --- |
| `step_idx` | Orders rows within a bench run; resets signal run boundary. |
| `base_pos`, `mtp_pos` | H2/H3 transition invariants check row-to-row deltas. |
| `mtp_top1`, `main_top1` | H1 invariant: `mtp_top1 == main_top1 ⇔ accepted`. |
| `accepted`, `topk_match` | H1 invariant directly — `accepted == topk_match` by construction. |

A second runtime diagnostic path would duplicate what the C1 sink already captures. The right tool is a CSV verifier, not another `KILN_MTP_*` env.

## Evidence — mechanical replay

`scripts/c16_plumbing_analyze.py` (new in this PR) reads any C1 CSV and mechanically asserts the three testable invariants (H4 is N/A as noted). Any violation exits non-zero and dumps the offending rows.

**Per-row invariant** (H1, greedy):

```
accepted == topk_match
```

**Between consecutive rows within one bench run** (H2, H3):

```
base_pos[i+1] - base_pos[i] == 2 if accepted[i] else 1    (H3)
mtp_pos[i+1]  - mtp_pos[i]  == 1 if accepted[i] else 0    (H2)
```

The script splits multi-run CSVs at `step_idx` resets (matching `kiln_model::c1_attr::clear()`'s reset of `NEXT_STEP_IDX` at the top of every `bench_latency_paged_mtp` call). It reports per-run violation counts plus an overall PASS/FAIL.

### ACCEPT-side coverage caveat

Under the current build α = 0.000 in the capture C15 used, which means the script running on that CSV would exercise the REJECT-side transition invariants only (no `prev.accepted` = True transitions to test H2 ACCEPT-branch). The script reports this explicitly — it emits "UNTESTABLE" for any hypothesis whose exercise count was zero rather than silently PASSing a vacuous case. Once a build achieves α > 0 again (whether by ship-fixing the head-forward bug or by flipping a `KILN_MTP_FORCE_ADVANCE_POS` diagnostic as suggested in C15), replay the same CSV through the analyzer to close out ACCEPT-side coverage.

To execute after a CSV is collected:

```bash
KILN_C1_ATTR_PATH=/tmp/mtp_c16.csv \
  KILN_SPEC_METHOD=mtp \
  ./target/release/kiln-bench \
    --paged --spec mtp --temperature 0 \
    --prompt-tokens 512 --max-output-tokens 128 --seed 42

python3 scripts/c16_plumbing_analyze.py /tmp/mtp_c16.csv
# exit 0 = all tested invariants hold; exit 1 = violations printed
```

## Adjacent observation — `new_h_prev` approximation on REJECT

`speculative.rs:484–491` already documents this as a known, non-bug approximation:

> On ACCEPT this is the hidden at the draft token's position in the verify pass (correctly predicts the bonus). On REJECT this is also the draft position's hidden — a known approximation because the draft was rejected, so the true h_prev for the corrected token is one position earlier. The staleness typically costs <5% of acceptance rate for k=1 MTP; extracting both positions from the verify pass is a follow-up optimisation that requires returning `[1, seq_len, H]` instead of only the last row.

This is in the pre-existing code and does not explain α = 0.000; it explains a ≤5% haircut on α from an already-working baseline. C12's α ≈ 0.033 and C14's prior α values all ran under the same approximation, so it is not the α-collapse driver. C16 does not propose a fix here — the optimisation is orthogonal to the Class B head-forward question that C17 inherits.

## Stale docstring (not a bug, but worth fixing in a follow-up)

`speculative.rs:519–530` documents an older 2-token verify pattern that predates the split-verify implementation at lines 584–600 and 670–716. Specifically the comment at lines 527–530 claims:

> the base KV at `base_pos + 1` is also written but stale and is overwritten on the next iteration

That is false under the current split-verify code: on REJECT, `base_pos + 1` is never written. Fixing the comment is a doc-only follow-up; no behaviour change is needed. Left unfixed to avoid mixing doc-only drive-bys into this PR.

## Cross-reference — Class A vs Class B from Phase C5

Phase C5 reported 87.6% Class B (MTP head bug) vs 12.4% Class A (verification/sampling bug) on a build where α ≈ 0.125. The current build's α = 0.000 makes the mix effectively 100% Class B: every step is rejecting because the MTP top-1 token does not match the base top-1 token. C16's audit confirms that if Class A were present, the accept/reject plumbing would not be what turned an agreement into a rejection — the `draft_token == target_at_0` compare is straightforward `u32` equality. So any residual Class A mass has to be inside `greedy_sample` itself (top-1 extraction of equal-scoring tied tokens, tensor dtype / layout drift between `mtp_logits` and `verify_pos0` before the `squeeze(1)` + top-1), which C5 already bisected and C16 does not re-open.

## Handoff to C17

The α-collapse root cause is **not** in the four accept/reject subsystems C16 audited. C17 should continue the Class B MTP head bisect where C15 left off:

1. **`lm_head` alignment.** `mtp_forward_step` ties to `weights.embed_tokens_t` at line 4647 (final matmul `normed.broadcast_matmul(&weights.embed_tokens_t)`). Confirm the tying direction, the dtype, and that `embed_tokens_t` is the transpose kiln expects (a single-dim transposition error would flip `lm_head` into a different vocabulary projection without crashing). C14 captured `logits` post-lm_head; diff that tap against the HF reference at step 0 to localize whether the drift enters at `post_norm → logits` or earlier.
2. **The 2x norm gap.** C15 reports `kiln_norm ≈ 70` vs `ref_norm ≈ 155` for `h_main` at every step, consistent under FP32 and BF16. This is a structural frame difference (probably a pre- vs post-final-RMSNorm tap point), not a scaling bug. The MTP head's `final_layernorm` at line 4639 re-normalizes the hidden state before `lm_head`; if that post-norm hidden still carries a 2× offset from what HF sees at the same tap, the `lm_head` projection lands in a different region of vocab space. Verify the `final_layernorm` weight dtype and the `rms_norm_eps` used here match the HF MTP block exactly.
3. **Structural frame mismatch (already flagged in C15 §"Why the drift is not a C15-new finding")**. Kiln's `h_main` tap and HF's `hidden_states[-1]` are measurably different reference frames. C17 should resolve which tap point matches the HF MTP reference input contract — the `mtp_reference_dump.py` passthrough used by C13 papers over exactly this mismatch and is the reason C14 didn't catch the drift.
4. **`fused` → MTP transformer block boundary**. C13/C14 certified `c6__fused` clean (post-fc projection). C14 added `post_block`, `post_norm`, `logits` taps. Running `scripts/mtp_compare.py` on a fresh capture with the C14 taps active against an HF reference that consumes `fused` as input (not `h_main`) would close the gap between "pre-block is clean" and "post-lm_head is wrong". That is the first-divergence target C17 should land.

A plumbing-side prerequisite for C17 is that α > 0 on at least one trace — so the ACCEPT-side path gets exercised at all. Two options from C15: (a) ship the head-forward fix and see α climb organically, (b) add a one-shot `KILN_MTP_FORCE_ADVANCE_POS` diagnostic that advances `mtp_pos` after N decode steps regardless of the accept decision, purely for capture coverage of head-1/head-2.

## Artifacts

- `scripts/c16_plumbing_analyze.py` — C1-CSV replay verifier for H1/H2/H3 invariants.
- This document.

No runtime code changed. No new env flag introduced. No new CSV schema fields. The C1 sink is sufficient as-is for this audit.
