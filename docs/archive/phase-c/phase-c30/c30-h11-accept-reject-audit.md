# Phase C30 — Accept/Reject + KV-Rollback Static Audit (H11)

**Date**: 2026-04-22
**Phase**: C30 (MTP α-regression hunt, post-C16/C17/C18)
**Hypothesis (H11)**: The MTP α=0.411 regression originates in the
speculative-decoding accept/reject pipeline *downstream* of `mtp_logits`
(acceptance math, KV-cache rollback on rejection, or `mtp_pos` / `h_prev`
index resync) — **not** in the MTP head itself.
**Method**: CPU-only static audit of the current accept/reject + KV-rollback
code path. No pod spend.
**Scope**: Three sub-hypotheses only — H11.1 acceptance math, H11.2 KV
rollback, H11.3 position/index resync.

## TL;DR

**Verdict: H11 REFUTED.**

The accept/reject + GDN-snapshot + position-advance pipeline is structurally
correct. None of the three sub-hypotheses survive a direct reading of the
current source:

| Sub-hypothesis | Verdict | Key evidence |
| --- | --- | --- |
| H11.1 acceptance math | Refuted | Greedy-only `u32 == u32` compare at `speculative.rs:598`. The stochastic `rejection_sample` at `speculative.rs:732+` is called only by unit tests — never by `speculative_mtp_decode_step`. C16 already rejected this as H1. |
| H11.2 KV rollback | Refuted | `LinearAttentionState::snapshot` / `restore_from` (`forward.rs:375-424`) deep-copy both recurrent and conv1d states. The stale base KV at `base_pos+1` after REJECT is overwritten by the next verify pass *before* any attention query reads it. |
| H11.3 position/index resync | Refuted | `(base_advance, mtp_advance) = (2, 1)` on ACCEPT, `(1, 0)` on REJECT is consistent with KV-slot allocation and MTP draft position reuse. Caller threads counters correctly in `bench.rs:1063-1131`. |

No single-line bug explains α=0.411. The regression must live in the logit
pipeline upstream of or at the MTP head itself — continue the Class B head
bisect narrowed by C17/C18.

**Recommended next phase (C31)**: pick up the lm_head / MTP-head /
final_layernorm bisect from C17's handoff. Do **not** re-audit accept/reject.

**Cost**: $0 (static audit, doc-only PR, no pod). ~45 min wall-clock.

---

## 1. Context and preconditions

C16, C17, and C18 collectively established:

- **C16** (`docs/archive/phase-c/phase-c16/c16-accept-reject-verdict.md`) audited an *earlier*
  accept/reject path and rejected the equivalent of H11.1–H11.3. At the time
  of C16 the code had **no rollback path at all** — the C16 note explicitly
  records: "Split-verify k=1 greedy has no rollback path. REJECT never runs
  the draft through the base model." That statement is now stale: the
  current `speculative_mtp_decode_step` has a `linear_state_snapshot` /
  `restore_from` mechanism for GDN state rollback, plus an explicit replay
  of `[last_token]` through the base model on REJECT. The rollback layer is
  new territory that C16 did not cover.

- **C17/C18** shifted the hunt to the MTP head itself (Class B bisect:
  `lm_head` weight tying, `norm_gap`, `final_layernorm`, MTP-internal
  projection vs base projection).

C30 is the "close the loop" audit before continuing the Class B hunt: prove
the **new** rollback layer isn't the regression source so C31 doesn't have
to re-open this ground later.

---

## 2. H11.1 — Acceptance math

### Code path

`crates/kiln-model/src/speculative.rs:596-598`:

```rust
let mut accepted_tokens: Vec<TokenId> = Vec::new();
let mut hit_eos = false;
let draft_accepted = target_at_0 == draft_token;
```

- `draft_token` is `greedy_sample(&mtp_logits)` — the argmax of the MTP
  head's logits over the vocabulary (line 565).
- `target_at_0` is `greedy_sample(&verify_pos0)` — the argmax of the base
  model's verify-pass logits at position 0 (line 593).
- `draft_accepted` is a pure `u32 == u32` integer compare.

### Dead code: stochastic rejection sampling

`speculative.rs:732+` contains a `rejection_sample` function with proper
uniform-random thresholding against `target_prob / draft_prob`. It is
exercised by unit tests (`test_rejection_sample_*`) but never called by
`speculative_mtp_decode_step`. The hot path is 100% greedy-deterministic.

### Why this cannot explain α=0.411

Under greedy decoding, `accepted == (mtp_top1 == main_top1)` holds by
construction. If `mtp_top1` agrees with `main_top1` on the same token, the
compare returns true — there is no scaling, probability-ratio arithmetic,
or randomness to break. The Phase C1 attribution CSV explicitly records
`topk_match = (draft_token == target_at_0)` alongside `accepted`; if those
ever disagreed that would be evidence of an acceptance-math bug, but under
the current `==` comparison they cannot disagree.

**Refuted.** The acceptance decision is a 1-line integer compare. α=0.411
means `mtp_top1 != main_top1` roughly 58.9% of the time — a head-logit
divergence, not an acceptance-math bug.

---

## 3. H11.2 — KV rollback

This is the sub-hypothesis C16 could not audit (no rollback existed at the
time). It has three moving parts: (a) the GDN recurrent + conv1d state
snapshot, (b) the base-model KV at `base_pos + 1` after REJECT, (c) the
MTP-cache KV at the rejected `mtp_pos`.

### 3.1 GDN snapshot / restore

`crates/kiln-model/src/speculative.rs:572-588` (snapshot before verify):

```rust
let linear_state_snapshot = linear_state
    .snapshot()
    .context("snapshot linear attention state before MTP verify")?;
let verify_input = [last_token, draft_token];
let (verify_logits, hidden_after_draft) = model_forward_paged_with_last_hidden(
    backend,
    &verify_input,
    weights,
    config,
    base_cache,
    base_block_table,
    base_pos,
    Some(linear_state),
    None, None,
)
```

`crates/kiln-model/src/speculative.rs:683-699` (restore on REJECT):

```rust
linear_state
    .restore_from(&linear_state_snapshot)
    .context("restore linear attention state after MTP rejection")?;
let verify_input0 = [last_token];
let (_verify_logits0, hidden_after_last) = model_forward_paged_with_last_hidden(
    backend, &verify_input0, weights, config,
    base_cache, base_block_table, base_pos,
    Some(linear_state), None, None,
)
.context("mtp rejection replay forward failed")?;
new_h_prev = hidden_after_last;
```

The snapshot is correctly taken **after** the MTP draft (which does not
touch `linear_state` — it operates on `mtp_cache` only, see
`speculative.rs:553-564`) and **before** the two-token verify (which does
advance `linear_state`). On REJECT we restore pre-verify state and replay
exactly `[last_token]`, so `linear_state` ends the step advanced by a net
one token (`last_token`), matching `base_advance = 1`.

### 3.2 Deep-copy semantics of snapshot / restore

`crates/kiln-model/src/forward.rs:375-388` (snapshot):

```rust
pub fn snapshot(&self) -> Result<Self> {
    let mut recurrent_states = Vec::with_capacity(self.recurrent_states.len());
    for t in &self.recurrent_states {
        recurrent_states.push(t.copy().context("snapshot recurrent state")?);
    }
    let mut conv_states = Vec::with_capacity(self.conv_states.len());
    for t in &self.conv_states {
        conv_states.push(t.copy().context("snapshot conv state")?);
    }
    Ok(Self { recurrent_states, conv_states })
}
```

`crates/kiln-model/src/forward.rs:398-424` (restore):

```rust
pub fn restore_from(&mut self, snapshot: &Self) -> Result<()> {
    // ... length-mismatch guards ...
    for (dst, src) in self.recurrent_states.iter_mut()
        .zip(snapshot.recurrent_states.iter()) {
        *dst = src.copy().context("restore recurrent state")?;
    }
    for (dst, src) in self.conv_states.iter_mut()
        .zip(snapshot.conv_states.iter()) {
        *dst = src.copy().context("restore conv state")?;
    }
    Ok(())
}
```

`Tensor::copy()` in candle issues a device-to-device memcpy per tensor. The
file-level doc comment at `forward.rs:367-374` explicitly states this:

> "This snapshot allocates new device tensors and issues a
> `cudaMemcpyDeviceToDevice` per layer. For Qwen3.5-4B that is
> 24 × (recurrent ≈ 2 MiB + conv ≈ 24 KiB) ≈ 49 MiB per snapshot."

Both the recurrent state (shape `[1, nv, dk, dv]`) and the conv1d window
(shape `[1, conv_dim, kernel_size - 1]`) are snapshotted — there is no
"forgotten half" of the GDN state that could leak draft-side history
through a REJECT boundary.

### 3.3 Base KV stale-slot at `base_pos + 1`

The two-token verify writes base KV at slots `[base_pos, base_pos + 1]`
(the base cache is paged; the verify pass indexes into
`base_block_table[base_pos]` and `base_block_table[base_pos + 1]`). On
REJECT:

- The replay of `[last_token]` at `base_pos` overwrites slot `base_pos`
  with KV that is bit-identical to the verify's write at the same slot
  (same token, same RoPE position, restored linear_state) — confirmed by
  reasoning about the per-layer computation, which depends only on
  `(token_id, position, linear_state_snapshot, weights)`.
- Slot `base_pos + 1` retains the stale KV for `draft_token` at position
  `base_pos + 1`.
- `base_advance = 1`, so the next iteration runs verify with
  `[new_last_token = target_at_0, new_draft]` at `new_base_pos =
  base_pos + 1`. That call writes slots `[base_pos + 1, base_pos + 2]`,
  overwriting the stale slot with the correct KV for `target_at_0` (which
  IS the token that belongs at position `base_pos + 1`) before any
  attention query at position `base_pos + 1` reads it.

Standard "write before attend" order in paged attention kernels: a token's
KV is written into its slot at the start of its forward step, and reads
for attention queries at later positions happen after the writer. There is
no window where an attention query at position ≥ `base_pos + 1` reads the
stale draft KV.

### 3.4 MTP-cache KV at rejected `mtp_pos`

On REJECT, `mtp_advance = 0` (`speculative.rs:711`). The next iteration
calls `mtp_forward_step` with the same `mtp_pos`, which writes MTP KV at
the same slot. The rejected draft's MTP KV is overwritten in place before
the next MTP draft reads it — same write-before-attend invariant.

### 3.5 Secondary observations (not correctness bugs)

Two documentation inconsistencies surfaced but do not affect correctness:

1. **Stale inline comment at `speculative.rs:708-710`**:

   ```rust
   // Base consumed exactly last_token. The rejected draft was never fed
   // through the base model, so no recurrent-state snapshot/restore is
   // needed and the KV cache has no stale base_pos+1 slot.
   (1, 0)
   ```

   This comment contradicts the enclosing branch: the draft *was* fed
   through the base model (via the two-token verify above), which is
   precisely why the snapshot/restore + stale-slot reasoning matters. The
   comment is a relic from an earlier iteration that predates the
   snapshot/restore rewrite. It should be replaced with an accurate
   description. **Not a correctness bug** — the code does the right
   thing; only the comment is wrong.

2. **Misleading doc at `forward.rs:394-397`**: the `restore_from` doc
   claims "Overwrites the current tensors in place so downstream GPU
   pointers (e.g. those captured inside a CUDA graph) stay valid." The
   actual implementation reassigns the Vec slot via `*dst = src.copy()`,
   which allocates a fresh tensor and moves it in — the old tensor's GPU
   storage is dropped. State *content* is correct, but any consumer that
   captured a raw GPU pointer through `linear_state.recurrent_states[i]`
   before the restore would see a dangling address. A grep of
   `crates/kiln-model/src/` for `cuda_graph` / `CudaGraph` confirms no
   such capture exists in `forward.rs`; graph machinery lives in
   `generate.rs` and `cuda_graph.rs` but does not close over
   `linear_state` tensors, so this is latent only. **Not a correctness
   bug today** — the doc claim should be updated or the impl should move
   to in-place slicing. Orthogonal to H11.

### Why this cannot explain α=0.411

α is computed as `draft_accepted_count / total_draft_attempts` over the
whole decode run. A KV-rollback bug would corrupt the **next** verify
pass's `target_at_0`, not the **current** `draft_accepted` decision. To
move α from a healthy ~0.72+ down to 0.411 the bug would need to bias
`target_at_0` away from the MTP head's `mtp_top1` consistently. The
snapshot/restore + stale-slot-overwrite analysis shows the rollback
layer does not corrupt state that the next verify reads from — the
linear_state is restored exactly, the MTP KV is overwritten, and the
stale base KV at `base_pos + 1` is overwritten before it becomes live.

**Refuted.**

---

## 4. H11.3 — Position/index resync

### Advance counters

`crates/kiln-model/src/speculative.rs:659-712`:

- ACCEPT (non-EOS): `(base_advance, mtp_advance) = (2, 1)` and
  `accepted_tokens = [draft, bonus]`.
- ACCEPT + EOS on draft: `(1, 1)` and `accepted_tokens = []` (EOS cuts
  the step).
- REJECT: `(1, 0)` and `accepted_tokens = [target_at_0]`.

### Caller threading

`crates/kiln-server/src/bench.rs:1063-1131` (the only production caller of
`speculative_mtp_decode_step`):

```rust
let mut base_pos = actual_prompt_tokens;
let mut mtp_pos = 0usize;
// ...
last_token = *step.accepted_tokens.last().unwrap();
base_pos += step.base_advance;
mtp_pos += step.mtp_advance;
h_prev = step.new_h_prev;
```

Per-iter check:

- After ACCEPT: base advanced by 2 (to account for `last_token` + `draft`
  both written in verify), MTP advanced by 1 (one draft slot consumed).
  `new_h_prev = hidden_after_draft` — the verify-pass hidden at position 1
  (the draft's position), which is exactly the hidden whose logits produced
  `bonus`. Per the MTP head contract documented at `speculative.rs:474-479`,
  that is the correct `h_prev` to feed the next MTP draft whose input token
  is `bonus`.
- After REJECT: base advanced by 1 (only `last_token` committed), MTP
  unchanged (the draft slot is re-used next iter). `new_h_prev =
  hidden_after_last` from the replay of `[last_token]` — the base model's
  hidden at `base_pos`, the canonical feed for the next MTP draft whose
  input is `target_at_0` at position `base_pos + 1`.

### RoPE position for MTP draft

`speculative.rs:545-564` documents and implements `base_pos + mtp_pos` as
the RoPE position for the MTP draft:

> "The inner block uses `base_pos + mtp_pos` for RoPE — so the MTP head
> sees the same rotation angles the main GQA block would have applied at
> that absolute position — and keeps `mtp_pos` as the write slot in the
> isolated MTP paged cache. See Phase B7a evidence in PR #276: without
> this, `post_layer` drifts monotonically with `mtp_pos` (cos_sim
> 0.999977 → 0.999531 → 0.997527 at pos 0/1/2)."

This is a **fixed** bug from earlier phases (PR #276). The current audit
confirms the fix is still in place and honored by `mtp_forward_step`.

### Why this cannot explain α=0.411

A position/index resync bug would manifest as either (a) MTP draft
producing tokens for the wrong absolute position (wrong RoPE angles → the
Phase B7a symptom, which was fixed), or (b) `target_at_0` being computed
from the wrong base-model position (the verify pass would have to read
stale KV or misplaced positions). The bench caller initializes counters
correctly, threads the returned advances correctly, and the
`speculative_mtp_decode_step` inner logic uses `base_pos + mtp_pos` for
RoPE and the correct cache indices. No resync bug is visible.

**Refuted.**

---

## 5. Cross-checks against C16

C16 left three notes that should be re-examined in light of the current
code:

1. **"Split-verify k=1 greedy has no rollback path. REJECT never runs the
   draft through the base model."** — Stale. The current code explicitly
   snapshots GDN state before verify and restores + replays on REJECT.
   This audit covers that new layer.
2. **"Class B divergence upstream of lm_head."** — Still the live
   hypothesis. H11 refuted => C17/C18 handoff to head bisect is still
   correct.
3. **"No acceptance-math bug."** — C30 confirms: greedy compare
   `u32 == u32`, no rejection sampling, no probability arithmetic.

---

## 6. Handoff to C31

**Narrowed search space** after C30:

| Sub-system | Status |
| --- | --- |
| MTP head draft logits | Suspect (Class B bisect continues) |
| MTP head weight tying (`lm_head` vs MTP projection) | Suspect |
| `final_layernorm` / `norm_gap` | Suspect |
| Accept/reject decision | Cleared (H11.1, C16 H1) |
| GDN state rollback on REJECT | Cleared (H11.2, new) |
| Base KV stale-slot correctness | Cleared (H11.2, new) |
| MTP KV slot re-use on REJECT | Cleared (H11.2, new) |
| `base_pos` / `mtp_pos` / `h_prev` threading | Cleared (H11.3, C16 H3) |
| RoPE position for MTP draft | Cleared (PR #276 fix still in place) |

**C31 should continue**:

- The Class B lm_head / MTP-head / final_layernorm bisect that C17/C18
  opened.
- Compare MTP-head logit distribution vs base-head logit distribution at
  the same absolute positions on the same prompt to find the head-level
  divergence source.
- A suitable starting point is the Phase C1 attribution CSV: the
  `mtp_top1_logit` and `main_top1_logit` columns (populated when
  `KILN_C1_ATTR_PATH` is set) give per-step head-logit pairs that can be
  diffed directly.

**C31 should NOT re-audit**:

- Accept/reject greedy compare.
- GDN snapshot / restore.
- Base KV stale-slot behavior on REJECT.
- MTP KV re-use on REJECT.
- Position / `mtp_pos` / `h_prev` threading.

All five are cleared as of this audit.

---

## 7. Secondary cleanup candidates (optional, non-blocking)

These are not required to make progress on α but should be picked up in a
low-priority cleanup PR when convenient:

1. **Fix the stale comment at `speculative.rs:708-710`** — replace with
   an accurate description of the snapshot/restore + stale-slot-overwrite
   reasoning from §3.3 above.
2. **Fix the misleading `restore_from` doc at `forward.rs:394-397`** —
   either update the doc to say the tensor slot is reassigned (not
   in-place) or change the impl to use in-place slicing. Only matters if
   CUDA-graph capture ever closes over `linear_state` tensors; not the
   case today.

Neither affects α. Filing them here so they aren't lost.

---

## 8. Cost accounting

- Pod time: **$0** (CPU-only static audit).
- Wall-clock: ~45 min (clone + read + write).
- Under the C30 task's declared cap ($15 / 60 min) with margin.
