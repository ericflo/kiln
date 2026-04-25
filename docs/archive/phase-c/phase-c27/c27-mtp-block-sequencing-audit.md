# Phase C27 — MTP block sequencing + paged-KV carryover audit (H8)

**Verdict: H8 RULED OUT. No asymmetry found across any of the 5 sub-properties.**
All multi-step verify scaffolding (block tables, seq_len flow, RoPE frame,
post-reject slot state, prefill-vs-decode mode) is consistent with the
Qwen3-Next-MTP reference contract once Phase C8's single-token self-attn
arming is accounted for. One factually-wrong doc comment in
`speculative.rs` is flagged for inline cleanup; it does not affect behavior.

Phase C27 is therefore a **doc-only redirect** in the C19 / C20 / C22 / C23 /
C25 / C26 family — $0 pod spend, handoff to H9.

## Task context

Per the C26 handoff (`docs/archive/phase-c/phase-c26/c26-mtp-weight-reload-sanity.md`), the
ruled-out hypothesis space after Phase C26 is **H1–H7**. α median remains
stuck at **0.1532**, far below the 0.5 ship floor. H8 was scoped as a
static-code audit of "MTP block sequencing + paged-KV carryover across
multi-step verify" — cheap to settle with eyeball + source grep, no pod
required unless a candidate asymmetry surfaced.

## Sub-property matrix

| # | Property | Verdict | Evidence file (line) |
|---|----------|---------|----------------------|
| 1 | MTP block-table allocation stable across multi-step verify | **Clean** | `generate.rs:1367-1390` |
| 2 | MTP seq_len / write-slot accounting per spec-decode step | **Clean** | `speculative.rs:553-711`, `forward.rs:4616-4633` |
| 3 | Position stride / RoPE frame on step k > 0 | **Clean (inert under Phase C8)** | `forward.rs:4590-4633`, `gqa_attention_paged:3409-3416` |
| 4 | Block deallocation / reuse across rejected drafts | **Clean (behaviorally); one stale comment** | `speculative.rs:678-711`, `paged_kv_cache.rs::PagedKvCache::write` |
| 5 | Multi-step prefill-vs-decode mode mismatch in MTP layer | **Clean** | `forward.rs:4408-4815`, `mtp_debug.rs:1089-1107` |

## Sub-property 1: block-table allocation

Both paged caches are pre-allocated once at prompt time and their block
tables are filled with the full physical block set
(`0..num_blocks`). Neither table is mutated for the rest of the decode:

```rust
// generate.rs:1367-1390
let base_cache = PagedKvCache::new(config.num_full_attention_layers, ...)?;
let mtp_cache  = PagedKvCache::new(1, ...)?;              // 1 layer for MTP
let mut base_block_table = BlockTable::new(block_size);
let mut mtp_block_table  = BlockTable::new(block_size);
for i in 0..num_blocks as u32 {
    base_block_table.push(i);
    mtp_block_table.push(i);
}
```

- Identical layout across base and MTP (block `i` maps to physical slot `i`
  in each isolated cache).
- No `push_empty` / `reclaim` / dealloc calls anywhere in the decode loop
  (`speculative.rs`, `generate.rs`, `forward.rs`).
- No multi-request or prefix-cache path runs during speculative decode, so
  the `BlockManager` free list never reshuffles mid-sequence.

**Result: no block-table asymmetry between base and MTP across multi-step
verify.**

## Sub-property 2: seq_len / write-slot accounting

Per-call seq_len values, traced through each branch of the spec-decode step:

| Call | Path | start_pos | seq_len | Cache writes |
|------|------|-----------|---------|--------------|
| Draft | `mtp_forward_step` | `base_pos + mtp_pos` | 1 | **None** — single-token self-attn armed (Phase C8) |
| Verify | `model_forward_paged_with_last_hidden` | `base_pos` | 2 | base_cache @ `[base_pos, base_pos+1]` |
| Accept replay | (none needed) | — | — | — |
| Reject replay | `model_forward_paged_with_last_hidden` | `base_pos` | 1 | base_cache @ `[base_pos]` (overwrites verify's last_token slot with same K,V) |

`base_pos` and `mtp_pos` update per `MtpSpeculativeStepResult`:
- Accept: `(base_advance, mtp_advance) = (2, 1)` — base jumps past
  `[last_token, draft_token]`; MTP jumps past its single slot.
- Reject: `(base_advance, mtp_advance) = (1, 0)` — base only commits
  `last_token`; MTP does not advance (its slot was never written, so there's
  nothing to commit).

`mtp_cache` is **never written** post-Phase-C8: `arm_mtp_single_token_self_attn`
(unconditional at `forward.rs:4602`) gates `gqa_attention_paged` so both the
write and the read skip the paged path:

```rust
// forward.rs:3334-3416 (gqa_attention_paged)
let single_token_self_attn = crate::mtp_debug::is_mtp_single_token_self_attn_armed();
// ... write skipped when armed ...
let (k, v, kv_len) = if single_token_self_attn { (k, v, 1) } else { paged_cache.read(...) };
```

So `mtp_pos` is effectively a logical counter with no physical manifestation.
Its value at each call determines only the `positions` tensor value passed
into the MTP inner block (which is also inert — see sub-property 3).

**Result: seq_len flow is linear, monotonic, and consistent across accepts
and rejects; no per-step accumulator goes out of sync.**

## Sub-property 3: position / RoPE frame

The MTP inner block call uses:

```rust
// forward.rs:4590
let abs_pos = base_pos + mtp_pos;
let positions = Tensor::new(&[abs_pos as f32][..], device)?;
```

Phase B7a (PR #276) validated switching from bare `mtp_pos` to
`base_pos + mtp_pos` — the larger "absolute position 0 vs P" mismatch was
visible in attention outputs. In-source comment at `forward.rs:4580-4581`
already documents that prior bare `mtp_pos` caused "monotonic `post_layer`
drift at pos=1,2".

At first pass this looks like a **residual +1-per-accept drift**:

```
call 1 (initial):       base_pos = P,       mtp_pos = 0   → abs_pos = P
call 2 after 1 accept:  base_pos = P + 2,   mtp_pos = 1   → abs_pos = P + 3
call 2 after 1 reject:  base_pos = P + 1,   mtp_pos = 0   → abs_pos = P + 1
```

After `k` consecutive accepts, `abs_pos = last_token_pos + k` (rather than
tracking `last_token_pos` exactly). That is a real drift in the numeric
value of `abs_pos` — but it is **unobservable** through the MTP inner block
output once Phase C8 armed single-token self-attn, because:

1. **RoPE rotations cancel in single-token self-attention.** Both Q and K
   get the same rotation `R_{abs_pos}` applied before the Q·Kᵀ inner
   product. Because RoPE is a block-diagonal orthogonal rotation,
   `(R_p·q) · (R_p·k)ᵀ = q · Rₚᵀ·Rₚ · kᵀ = q · kᵀ`, independent of `p`.
2. **Softmax over kv_len = 1 is always 1.0.** The attention output equals
   `V` regardless of the score.
3. **V itself is computed from the fused hidden state without any RoPE
   dependency** (RoPE only hits Q and K).
4. **The MLP is position-independent** (no positional embedding inside
   `swiglu_ffn`).

So the MTP layer's output as a whole is invariant to `abs_pos` under the
post-C8 arming. The `base_pos + mtp_pos` formula is effectively dead
parameter — it produces a tensor that is consumed by a code path whose
output does not observe it.

**This is only safe because of the C8 single-token arming.** If a future PR
lifts the C8 arming (e.g. to restore paged MTP cache reads with multi-token
context), the residual +1-per-accept drift becomes immediately observable
and would need to be replaced with either:
- A per-call `last_token_pos` tracked independently of `base_pos / mtp_pos`, or
- `abs_pos = base_pos - 1` on calls where `mtp_pos == 0` and the token being
  modelled is `last_token` (its stream position is `base_pos - 1`).

A narrow cleanup follow-up could replace the formula with
`abs_pos = base_pos - 1 + mtp_pos` (which tracks `last_token_pos` exactly
assuming `last_token` is at position `base_pos - 1` at each call) and drop
the dead formula entirely. That is deferred — not on the α-collapse
critical path.

**Result: no behaviorally-observable RoPE-frame asymmetry under the current
C8 arming.** Flag for future lifts of C8, and flag for narrow doc /
formula cleanup when convenient.

## Sub-property 4: block deallocation / reuse across rejected drafts

`PagedKvCache::write(layer, block_table, start_pos, k, v)` writes directly
into block-mapped slots with no dirty tracking, no per-position
"invalidate", no deallocation. On reject, the sequence is:

1. Verify writes `[base_pos, base_pos+1]` in base_cache (both full-attn
   layers and linear-attn state).
2. Snapshot of `linear_state` is captured **before** verify.
3. On reject, `linear_state.restore_from(&snapshot)` reverts the
   linear-attn recurrence to its pre-verify value.
4. Replay `[last_token]` via `model_forward_paged_with_last_hidden` at
   `start_pos = base_pos` — this writes K,V at `[base_pos]`, overwriting
   the verify's K,V at `base_pos` (which happen to be the same correct K,V
   for `last_token`, since verify_input = `[last_token, draft_token]`).
5. K,V at `[base_pos+1]` are left as the draft token's state.

Step 5's stale slot is **not read** by any subsequent call because:
- Next iteration's `base_pos` is `old_base_pos + 1` (we advanced by 1).
- Verify at `start_pos = new_base_pos` writes `[new_base_pos, new_base_pos+1]`
  = `[old_base_pos+1, old_base_pos+2]`, starting with a **write** to the
  stale slot.
- Paged-attn reads happen per forward pass and only cover
  `[0..current_pos]` worth of slots — at the time of the write, nothing is
  attending over `base_pos+1` yet.

This is correct behavior. But there are **two comments in `speculative.rs`
about this same state that contradict each other**:

- `speculative.rs:678-682` (correct): "The stale base KV written at
  `base_pos + 1` is outside the next attention window and will be
  overwritten if that slot becomes live later."
- `speculative.rs:708-710` (**factually wrong**): "Base consumed exactly
  last_token. The rejected draft was never fed through the base model, so
  no recurrent-state snapshot/restore is needed and the KV cache has no
  stale base_pos+1 slot."

The second comment is wrong on both claims:
- The rejected draft **was** fed through the base model (as the second
  position of `verify_input = [last_token, draft_token]` at
  `speculative.rs:575-588`).
- The KV cache **does** have a stale `base_pos+1` slot (written during
  verify, left after snapshot/restore + replay of only `last_token`).

The comment is misleading future readers — it reads as if reject takes an
entirely different code path that skips verify, but verify runs before the
accept/reject decision (reject is discovered by comparing verify's output
to the draft's prediction). Proposed inline fix:

```rust
// Base consumed exactly last_token. The rejected draft's K,V at
// base_pos+1 is stale but outside the next attention window and will be
// overwritten before any read (see comment above at line ~681). Linear
// state was snapshot-restored pre-replay, so no recurrent drift
// remains.
(1, 0)
```

This cleanup is safe to ship alongside the H8 redirect PR; it is
documentation-only with no behavioral change. (C27 ships the audit doc
only — the inline comment fix is proposed for a follow-up PR or can be
tacked into the first H9 source-edit PR.)

**Result: behaviorally correct, with one local documentation fix
recommended.**

## Sub-property 5: prefill-vs-decode mode mismatch in the MTP layer

There is no prefill-vs-decode branching *inside* `mtp_forward_step`:

- Input is always a single-token fused hidden state: `fused ∈ [1, 1, H]`
  (line 4527).
- `seq_len` passed to `transformer_block_paged` is always 1 (derived from
  tensor shape).
- `full_attn_layer_idx = 0` is hard-coded at `forward.rs:4631`.
- `single_token_self_attn` is armed for the duration of the inner block
  and disarmed immediately on return (line 4636), so no non-MTP code path
  can observe or misuse it.

The only variable value entering the MTP inner block is `(fused, positions,
mtp_pos, mtp_cache, mtp_block_table)`. Per sub-properties 1–3, each of
those is either stable, consistent across calls, or inert under the C8
arming.

The base model's forward path does branch by `seq_len`:
- Verify (`model_forward_paged_with_last_hidden`, seq_len = 2): prefill
  path — multi-position tensor, positions spanning `[base_pos, base_pos+1]`.
- Replay (seq_len = 1): decode path — single position, `positions =
  [base_pos]`.

But this branching is wholly inside the **base** model and well-tested
pre-Phase-C1. The MTP layer never observes it — MTP always runs with
seq_len = 1 regardless of verify vs replay.

**Result: no mode mismatch across the MTP layer. Base-side mode dispatch
is unchanged from pre-C8 and is not a plausible α-collapse locus.**

## What this means for the α-collapse root cause

H8 was the most plausible "mechanical sequencing bug" hypothesis left
after C26. Ruling it out pushes the investigation further toward model
correctness territory — specifically, toward hypotheses that deal with
what the MTP layer's **output** contains (not how it's called or stored):

### H9 (proposed): MTP output distribution divergence from the reference

- Does the MTP head's logits distribution (post-softmax, per-position)
  match `scripts/mtp_reference_dump.py`'s reference sample-by-sample at
  k ≥ 1 positions?
- We know from Phase C12 that projection matmul output magnitudes match
  after fp32-head arming. We know from Phase C5 / C6 that block structure
  and weight names line up. But we have **not** compared the actual
  probability distribution of the MTP head's top-K tokens against the
  reference on the same prompt.
- Next experiment: dump `(prompt, last_token, draft_logits_top_K)` from
  kiln and from HF/vLLM on 10 prompts where kiln's measured α is <0.2.
  Compute rank-correlation, top-1 overlap, and KL divergence between the
  two distributions.
- If distributions diverge systematically on the SAME `(prompt,
  last_token, h_prev)`, the bug is in the MTP head compute itself — not
  in how it's invoked, scheduled, or cached. This rules in or out the
  entire "wiring / scaffolding" class of hypotheses and forces the
  focus onto norms, fused hidden state construction, or the `fc` (concat
  → column-parallel projection) layer.

### H10 (proposed): `h_prev` contract mismatch

- Phase C17 / C18 landed the post-final-norm `h_prev` contract. Did
  post-C18 α actually improve relative to pre-C17 baseline? If the
  answer is "yes, by some amount, but still below 0.2", then we know
  `h_prev` normalization is *partially* correct but potentially not
  fully — e.g. is the base model's final-norm identical to what MTP
  expects (`pre_fc_norm_hidden`)?
- vLLM's `Qwen3NextMultiTokenPredictor.forward` (line 98-134 of the
  reference) applies `pre_fc_norm_hidden` to the incoming
  `hidden_states` — **after** the base model's own final norm. So the
  chain is:
  1. Base model layers → `RMSNorm(base.norm)` → `h_prev`
  2. MTP receives `h_prev`, applies `RMSNorm(pre_fc_norm_hidden)`
  3. concat with `RMSNorm(pre_fc_norm_embedding)(embed(last_token))`
  4. `fc` → decoder block
- Does kiln follow all four steps, or is one of the norms missing /
  using a different eps / using the wrong weight tensor? This is a
  fast static-code audit (similar in shape to C27) and belongs in the
  next cycle.

H9 is the natural next hypothesis to investigate — it's empirical, cheap
to bench, and differentiates "compute bug" from "wiring bug".

## Files inspected

- `crates/kiln-model/src/speculative.rs` (909 lines; full read)
- `crates/kiln-model/src/forward.rs` (lines 3107-3416, 4272-4815,
  4900-5110)
- `crates/kiln-model/src/mtp_debug.rs` (lines 1089-1107)
- `crates/kiln-model/src/generate.rs` (lines 1320-1558)
- `crates/kiln-model/src/paged_kv_cache.rs` (full read)
- `crates/kiln-core/src/block.rs` (full read)
- `vllm/model_executor/models/qwen3_next_mtp.py` (cross-reference, full
  read)

## Cross-references

- Phase B7a `base_pos + mtp_pos` rationale: PR #276
- Phase C8 single-token self-attn arming: PR #319
- Phase C12 fp32-head arming: earlier Phase C12 PR
- Phase C17 / C18 `h_prev` post-norm contract: C17 / C18 PRs
- Ruled-out hypothesis chain: H1 (#340), H2 (#344), H3 (#346 + #350),
  H4 (#347), H5 (#348), H6 (#351), H7 (#352 / C26)

## Followups (not in this PR)

- Inline comment fix at `speculative.rs:708-710` (contradicts `:678-682`).
- Narrow cleanup of `abs_pos = base_pos + mtp_pos` → `base_pos - 1 +
  mtp_pos` with comment update, OR drop the formula entirely in favour of
  a tracked `last_token_pos` field on `MtpSpeculativeStepResult`.
- H9 and H10 planning tasks.

## Cost

$0 pod spend (static audit, matches C19 / C20 / C22 / C23 / C25 / C26
doc-only redirect pattern).
