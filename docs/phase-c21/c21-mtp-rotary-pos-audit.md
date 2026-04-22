# Phase C21 — MTP rotary `mtp_pos` offset audit (hypothesis 3, doc-only redirect)

## Verdict

**Hypothesis 3 is INCONCLUSIVE on static audit alone, with no positive
evidence of a bug. No code change. No pod spend.** Kiln's MTP rotary
formula `abs_pos = base_pos + mtp_pos` (i.e. position `N` for the first
draft step at the end of an `N`-token prompt) **matches one of the two
position conventions used inside vLLM** (the DFlash first-pass kernel)
and differs from the other (the EAGLE first-pass `set_inputs_first_pass`
default pathway, which holds positions at `N-1` while shifting input
ids). vLLM's `Qwen3NextMultiTokenPredictor.forward` itself is
position-agnostic — it forwards `positions` straight through to the
inner block — so the canonical reference does not pin a single value.
Kiln's `scripts/mtp_reference_dump.py` is **not** an independent ground
truth: it was authored to mirror kiln's own absolute-position
convention. Without an external pin (the original training-time MTP
driver, or a hardware A/B), the static audit cannot disprove H3 and
cannot confirm a bug either. Hand off the residual α gap (C18 median
0.1532, floor 0.5) to **H4 (MTP-head gate / residual / `fc`
parameterization)** and **H5 (draft-side sampler mismatch)**, and queue
H3 as a low-cost hardware A/B (`abs_pos = base_pos + mtp_pos` vs
`abs_pos = base_pos - 1 + mtp_pos`) to run only if H4/H5 also fail to
recover α.

## C18/C20 hypothesis as stated

From [`docs/phase-c20/c20-mtp-block-norm-audit.md`](../phase-c20/c20-mtp-block-norm-audit.md)
(handoff to C21, item 3 of the C18 queue):

> **Rotary `mtp_pos` offset.** MTP inner-block rotary should receive
> `pos + 1` (the next position) relative to the main block's rotary at
> `pos`. If kiln feeds the same position index to both, the rotary
> frequency is off-by-one and draft tokens will consistently miss.

Two factual claims to test on static audit:

1. **Kiln-position claim:** Kiln passes the same position index to the
   MTP inner-block rotary that the main block just consumed (i.e.,
   uses `N-1` when the main block last touched position `N-1`).
2. **HF-canonical claim:** The HF / vLLM `Qwen3NextMTP` reference
   feeds the inner block `position_ids = main_position_ids + 1`.

Both claims are sharper than "kiln has a +1 bug" and are individually
testable from source. The audit below disproves claim 1, partially
disproves claim 2, and surfaces a structural ambiguity in vLLM's own
two-driver design that prevents a clean pass/fail at the static layer.

## Tensors and counters in scope

For an `N`-token prompt followed by speculative MTP decoding:

| Symbol | Meaning | First MTP step |
| --- | --- | --- |
| `actual_prompt_tokens` | The prompt length consumed by prefill | `N` |
| `base_pos` | Absolute position of `last_token` in the base sequence | `N` (post-prefill) |
| `mtp_pos` | Local slot index into the isolated MTP paged KV cache | `0` |
| `abs_pos` | Absolute position fed to MTP inner-block RoPE | kiln: `N + 0 = N` |

Two reference conventions appear in vLLM:

| vLLM driver | First-MTP-step position for the new draft | Code path |
| --- | --- | --- |
| EAGLE `set_inputs_first_pass` (default) | `N-1` (positions held; input shifted by 1) | `vllm/v1/spec_decode/eagle.py:646-678` |
| DFlash kernel (`dflash_step_*`) | `N` (= `last_target_pos + 1`) | `vllm/v1/spec_decode/utils.py:491,526-527` |

The third reference, `Qwen3NextMultiTokenPredictor.forward`
([`qwen3_next_mtp.py:98-126`](#references-vllm)), is position-agnostic
— it accepts `positions` as a tensor argument and passes it straight
into `self.layers[current_step_idx](positions=positions, ...)` with no
`+1` rewriting and no shift. The choice of `N-1` vs `N` is made
*entirely* in the caller (the EAGLE drafter) before the model sees the
tensor.

## Evidence

### 1. Kiln's caller initializes `base_pos = N`, `mtp_pos = 0`

`kiln-server/src/bench.rs` ([lines 1063-1066](#references-kiln)) opens
the speculative decode loop after a single prefill on the prompt:

```rust
let mut num_tokens = 1usize; // counting the first token from prefill
let mut base_pos = actual_prompt_tokens;
let mut mtp_pos = 0usize;
```

`base_pos = actual_prompt_tokens = N` here means "the absolute position
where `last_token` sits in the base sequence" — i.e., the position of
the prefill's final sampled token, which is the `last_token` input to
the very first speculative step. There is no off-by-one against
`actual_prompt_tokens`: the prefill writes positions `[0, N-1]` to the
base KV cache, samples the first token at slot `N-1`, and that token's
absolute position is treated as `N` for the post-prefill speculative
loop, which is consistent with the C13/C18 base-stream accounting.

### 2. Speculative decoder forwards `(base_pos, mtp_pos)` unchanged

`speculative_mtp_decode_step` ([`speculative.rs:565-575`](#references-kiln))
forwards both counters into the draft step verbatim:

```rust
let (mtp_logits, _mtp_hidden) = mtp_forward_step(
    backend,
    last_token,
    h_prev,
    weights,
    config,
    mtp_cache,
    mtp_block_table,
    base_pos,
    mtp_pos,
)
```

The accompanying doc-comment ([lines 555-564](#references-kiln))
already names the convention explicitly: "`base_pos` (the absolute
sequence index the draft token will occupy in the base stream) and
`mtp_pos` (the MTP-cache slot index). The inner block uses
`base_pos + mtp_pos` for RoPE …" The comment also recalls the Phase
B7a evidence — under bare `mtp_pos`, `post_layer` cosine drifted from
0.999977 to 0.997527 across positions 0/1/2 — that motivated the
current `abs_pos = base_pos + mtp_pos` formula introduced in PR #284
(B8) and PR #314 (C3).

### 3. `mtp_forward_step` materializes RoPE at `abs_pos = base_pos + mtp_pos`

`mtp_forward_step` ([`forward.rs:4559-4622`](#references-kiln)):

```rust
let abs_pos = base_pos + mtp_pos;
let positions = Tensor::new(&[abs_pos as f32][..], device)?;
…
let mtp_hidden_result = transformer_block_paged(
    backend, &fused, &mtp.layer, config,
    &positions, mtp_pos, …
);
```

The two arguments to `transformer_block_paged` are intentionally
distinct: `&positions` (a single-element tensor holding `abs_pos`)
drives RoPE inside `gqa_attention_paged`, while the bare `mtp_pos`
serves as the write-slot index into the isolated MTP paged KV cache.
The MTP cache is a separate address space; slot `mtp_pos` is the
correct write target regardless of where the token sits in absolute
stream order.

For the very first MTP step (`base_pos = N`, `mtp_pos = 0`), this gives
`abs_pos = N`. **Claim 1 ("kiln uses `N-1`") is therefore false**:
kiln does *not* pass the same position the main block just consumed.
The main block consumed position `N-1` for the prefill's last input;
kiln's MTP rotary uses position `N`.

### 4. vLLM `Qwen3NextMultiTokenPredictor.forward` is position-agnostic

The canonical vLLM implementation
([`vllm/model_executor/models/qwen3_next_mtp.py:98-126`](#references-vllm),
also identical in `qwen3_5_mtp.py:121-149`):

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    if get_pp_group().is_first_rank:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None
    …
    current_step_idx = spec_step_idx % self.num_mtp_layers
    hidden_states, residual = self.layers[current_step_idx](
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )
```

`positions` is passed straight through. There is no `positions + 1`,
no shift, and no slot-relative arithmetic anywhere in the MTP module.
**Claim 2 ("HF/vLLM rewrites positions to `main_position_ids + 1`
inside the MTP head") is therefore false at the model-module layer.**
The `+1` (or non-`+1`) decision is entirely the caller's.

### 5. vLLM EAGLE first-pass holds `positions = target_positions`

`Eagle3Proposer.set_inputs_first_pass`
([`vllm/v1/spec_decode/eagle.py:646-678`](#references-vllm)) is the
default first-step driver. Its body for the EAGLE/EAGLE3 case
(`needs_extra_input_slots == False`):

```python
if not self.needs_extra_input_slots:
    if token_indices_to_sample is None:
        token_indices_to_sample = cad.query_start_loc[1:] - 1
    num_tokens = target_token_ids.shape[0]
    # Shift the input ids by one token.
    # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
    self.input_ids[: num_tokens - 1] = target_token_ids[1:]
    # Replace the last token with the next token.
    # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
    self.input_ids[token_indices_to_sample] = next_token_ids
    …
    self._set_positions(num_tokens, target_positions)
```

For the single-prompt, single-draft-step case (`N`-token prompt, one
new sampled token, one MTP step):

- `target_token_ids[0:N]` are the prompt tokens at base positions
  `[0, N-1]`.
- `target_positions[0:N]` are exactly `[0, N-1]`.
- `next_token_ids[0]` is the post-prefill sampled token, whose true
  absolute position would be `N`.
- The shift writes `input_ids[0:N-1] = target_token_ids[1:N]` — i.e.,
  every slot now holds the token that originally lived one position
  later.
- The last write `input_ids[N-1] = next_token_ids[0]` replaces slot
  `N-1` with the new sampled token, whose true position is `N`.
- `self._set_positions(num_tokens, target_positions)` leaves positions
  as `target_positions = [0, N-1]`.

**Net result: the slot the MTP head will sample from holds `input =
next_token@N` paired with `position = N-1`.** The rotary the MTP head
receives at the predict slot is therefore one less than the input
token's "natural" absolute position. This is the source of the
"`pos + 1`" framing in the C18 hypothesis: relative to the
*main-block* rotation that handled the same token, the MTP head's
rotation is **one less**, i.e. equivalently, kiln's `abs_pos = N`
is one *more* than vLLM EAGLE's `position = N-1`.

The subsequent autoregressive draft steps in vLLM EAGLE
(`utils.py:49-62`, `eagle_step_slot_mapping_metadata_kernel`) then do
`new_position = position + 1` per draft step, which mirrors kiln's
`abs_pos = base_pos + mtp_pos` increment-by-1 in the second-and-later
MTP slot — the disagreement is a constant offset of 1, not a stride.

### 6. vLLM DFlash first-pass uses `query_pos = N` (matches kiln)

The DFlash drafter
([`vllm/v1/spec_decode/utils.py:491,526-527`](#references-vllm)) is the
alternate first-pass kernel (`needs_extra_input_slots == True` /
`parallel_drafting == True`). Its query-position computation:

```python
# 2. Computes query positions (last_target_pos + 1 + offset)
last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
query_pos = last_pos + 1 + query_off
```

For the first draft step (`query_off = 0`), `query_pos = last_pos + 1
= (N-1) + 1 = N`. **DFlash's first-MTP-step position is `N`, exactly
what kiln uses.** The dflash + `mtp` + `draft_model` methods follow
this driver; the EAGLE / EAGLE3 methods follow §5.

`eagle.py:826` confirms the dispatch split:

```python
return self.method not in ("mtp", "draft_model", "dflash")
```

i.e., the methods named `mtp`, `draft_model`, and `dflash` go through
the DFlash-style path (which produces `position = N`); the methods
named `eagle` / `eagle3` go through the EAGLE first-pass shift path
(which produces `position = N-1`). **vLLM contains both conventions
and the choice is method-specific.**

### 7. Kiln's `scripts/mtp_reference_dump.py` is not independent ground truth

`scripts/mtp_reference_dump.py` ([lines 380-470](#references-kiln))
hand-rolls a PyTorch reference for the MTP step. The RoPE block reads:

```python
# Phase C3: rotate by `abs_pos = base_pos + mtp_pos`, not bare `mtp_pos`.
abs_pos = base_pos + mtp_pos
q = apply_rope_partial(q, abs_pos, head_dim, rotary_dim, rope_theta)
k = apply_rope_partial(k, abs_pos, head_dim, rotary_dim, rope_theta)
```

This script was authored in PR #276 (B7a) and tightened in PR #314
(C3) **explicitly to mirror kiln's `abs_pos = base_pos + mtp_pos`
formula**, then used by `scripts/mtp_compare.py` to bound kiln's
divergence from "the reference" at <1e-5 cos similarity. Any
comparison run that uses this script will agree with kiln by
construction whether or not kiln's convention is correct. The Phase
B7a parity result therefore validates the *implementation* of the
chosen convention but does **not** independently validate the
*choice* of convention. C21 closes the loop: there is no daylight
between kiln and its own reference dump because they share the same
arithmetic.

To get an independent pin, one of the following would be needed:

1. The Qwen3-Next MTP **training** code, where the rotary positions
   for the MTP target slot are determined by the loss target rather
   than by inference-time scheduling. If the training driver
   computes `position = N` (DFlash convention), kiln is correct; if
   it computes `position = N-1` (EAGLE convention), kiln is one off.
2. A non-kiln, non-vLLM MTP driver (e.g. SGLang's Qwen3.5 MTP
   reference if/when it lands; HF's planned MTP support — currently
   blocked by `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`,
   per the C19 audit).
3. A hardware A/B that toggles `abs_pos` between `base_pos + mtp_pos`
   and `base_pos - 1 + mtp_pos` and measures α directly. This is
   cheap (~25 min of A6000 time) but is not a static-audit deliverable
   and was deliberately scoped out of the C21 budget per the task's
   "ship doc-only PR if audit disproves H3" branch and the "do not
   acquire a pod for inconclusive evidence" preflight.

## Why this hypothesis was tempting

The C18 doc framed H3 as the most mechanically suspect of the three
remaining hypotheses, on the strength of two cues:

- **Phase B7a (PR #276) precedent.** Kiln's prior bug — passing bare
  `mtp_pos` instead of `base_pos + mtp_pos` — produced the textbook
  RoPE-wrong-position signature: monotonic `post_layer` cosine drift
  with the position index. The fix moved kiln from "position
  arithmetic visibly broken" to "position arithmetic plausibly
  correct," which made "still off by one" the natural next sub-bug
  to suspect.
- **vLLM EAGLE convention.** A surface-level read of vLLM's EAGLE
  driver does suggest "MTP runs at a different position than the
  main block did" — which is true for the EAGLE first-pass (positions
  held at `N-1` while inputs shift) and for the autoregressive draft
  loop (per-step `+1`). Reading only that path, "kiln must be using
  the wrong index" is the natural inference.

The audit shows both cues were over-interpreted:

- The B7a fix already absorbed the "absolute vs cache-local" axis;
  what remains is a 1-step phase choice (`N` vs `N-1`) that the B7a
  evidence cannot discriminate (cosine to a kiln-shaped reference
  stays high either way, because both options sit on the same
  rotation manifold).
- vLLM has *two* drivers (EAGLE first-pass and DFlash) that resolve
  the phase choice differently, and `Qwen3NextMTP` itself is
  position-agnostic. The trained Qwen3.5-4B MTP head was trained
  under a single convention, but neither vLLM driver tells us which.

## Hypothesis 4 — the next target

Carrying the C18 handoff item 4 forward as the new head of the queue,
**advanced over H3 because H4 has stronger pre-bench evidence**:

> **MTP-head gate / residual / `fc` parameterization.** The
> `mtp.fc` projection collapses `[h_main || embed]` → `H` *before*
> the inner block. If kiln's gating, residual addition, or
> normalization around `fc` differs from HF's by even a sign or a
> missed scale, the MTP block sees a uniformly mis-scaled input and
> α pegs near a constant low.

Concrete artifacts in scope:

- `mtp.fc` projection ([`forward.rs:4527-4543`](../../crates/kiln-model/src/forward.rs)):
  the `concat([pre_fc_norm_embedding(embed), pre_fc_norm_hidden(h_prev)])`
  → `[2H → H]` projection. Cross-check that residual / bias / dtype
  exactly mirrors `Qwen3NextMultiTokenPredictor.forward`'s `fc`
  application.
- `mtp.layer` first-residual application
  ([`forward.rs:4605-4622`](../../crates/kiln-model/src/forward.rs)):
  the inner `transformer_block_paged` call expects `residual = None`
  on entry per HF
  ([`qwen3_next_mtp.py:115`](#references-vllm) sets `residual = None`).
  Verify kiln's call honours this and does not accidentally feed in
  a stale residual from a prior MTP step.
- HF reference parity: `Qwen3NextMtpDecoderLayer.forward` (vendored
  inline in vLLM via the shared `Qwen3NextDecoderLayer`) for the
  exact pre-residual / post-residual sequence around the inner
  block.

H5 (draft-side sampler mismatch / temperature / penalty drift) remains
on deck as the fallback if H4 also disproves cleanly.

### Suggested next-step (CPU-only, no pod)

1. **Targeted forward read.** Trace `mtp.fc` and the inner-block
   residual hand-off in `mtp_forward_step`
   ([`forward.rs:4500-4622`](../../crates/kiln-model/src/forward.rs)).
   Compare the dtype, bias, and residual-init sequence to vLLM's
   `Qwen3NextMultiTokenPredictor.forward` *and* to
   `Qwen3NextDecoderLayer.forward` (the inner block) one line at a
   time.
2. **Reference-dump comparison.** Update
   `scripts/mtp_reference_dump.py` to dump the post-`fc` and
   post-first-residual tensors and compare against kiln. (Note: the
   reference-dump self-consistency caveat from §7 applies here too —
   any tap that was authored to mirror kiln will agree by
   construction. The cleanest discriminator is the residual-init
   path: confirm kiln passes `residual = None` to the inner block,
   not a borrowed value.)
3. **If parity diverges:** ship the `fc`/residual fix and re-run the
   C18 α-recovery bench (3 seeds, 512 prompt × 128 decode, A6000,
   `KILN_SPEC_METHOD=mtp`, `KILN_W4A16=1`).
4. **If parity matches:** disprove H4 with a Phase C22 doc-only PR
   and advance to **H5 (draft-side sampler mismatch)** or, if H5
   also disproves, return to H3 and run the cheap hardware A/B
   (`abs_pos = base_pos + mtp_pos` vs `abs_pos = base_pos - 1 +
   mtp_pos`) as a last discriminator before opening a deeper
   investigation into the trained MTP head's expected calibration.

The CPU-only steps cost no pod time. Only step 3 — once an actual
fix is in hand — justifies acquiring an A6000.

### Why H3 is parked, not closed

H3 is the only one of H1/H2/H3 that survives static audit without a
clean disproof. The reason it is parked rather than tested now:

- **No positive evidence kiln is wrong.** Kiln's convention matches
  vLLM DFlash exactly. The EAGLE first-pass disagreement could be
  EAGLE's choice, not the trained head's expectation.
- **Hardware A/B is cheap but not free.** ~25 min A6000 + 3 seeds
  to discriminate; ~$5 of pod time. Worth running, but only after
  H4/H5 (which have stronger pre-bench evidence and may not require
  a hardware bench at all if a code-side parity bug surfaces).
- **The C18 handoff explicitly enumerates H4 and H5.** The queue
  ordering was set by C18 as 1 → 2 → 3 → 4 → 5; C19 disproved 1,
  C20 disproved 2, C21 leaves 3 ambiguous. Continuing 4 → 5 → A/B
  for 3 keeps the cheap-first ordering intact.

## Cost / budget

- Wall-clock: ~75 min (no pod, source audit + vLLM cross-reference of
  three driver paths + reference-dump self-consistency analysis +
  doc).
- Pod cost: $0. Hypothesis was reduced to "structural ambiguity in
  the canonical reference" from the existing kiln source +
  `qwen3_next_mtp.py` + `qwen3_5_mtp.py` + `eagle.py` +
  `utils.py` + `mtp_reference_dump.py`.
- Well under the 120 min / $50 task cap.

## References (kiln)

- `crates/kiln-server/src/bench.rs:1063-1066` — initial counters for
  the speculative MTP loop (`base_pos = actual_prompt_tokens`,
  `mtp_pos = 0`).
- `crates/kiln-server/src/bench.rs:1086-1138` — the speculative MTP
  decode loop body and counter advancement
  (`base_pos += step.base_advance`, `mtp_pos += step.mtp_advance`).
- `crates/kiln-model/src/speculative.rs:510-577` — flow doc and
  forwarding of `(base_pos, mtp_pos)` into `mtp_forward_step`.
- `crates/kiln-model/src/forward.rs:4559-4622` — `abs_pos =
  base_pos + mtp_pos` and call into `transformer_block_paged`.
- `scripts/mtp_reference_dump.py:380-470` — reference dump hand-rolled
  to mirror kiln's `abs_pos` convention (see §7 for the
  self-consistency caveat).
- `docs/phase-c18/c18-h-prev-post-norm-fix.md` — origin of the H1/H2/H3
  hypothesis queue and the residual α-gap framing.
- `docs/phase-c19/c19-fc-norm-audit.md` — H1 disproof (no
  `mtp.fc_norm` tensor in checkpoint).
- `docs/phase-c20/c20-mtp-block-norm-audit.md` — H2 disproof (block
  dual-norm wiring is correct) and direct C21 handoff.

## References (vLLM)

Sparse-cloned at `/tmp/vllm-mtp-check/` (read-only, CPU-only):

- `vllm/model_executor/models/qwen3_next_mtp.py:98-126,262-274` —
  `Qwen3NextMultiTokenPredictor.forward` and `Qwen3NextMTP.forward`.
  Position-agnostic; passes `positions` straight through to the
  inner layer with no `+1` rewriting.
- `vllm/model_executor/models/qwen3_5_mtp.py:121-149` — Identical
  position handling for the Qwen3.5 MTP variant.
- `vllm/v1/spec_decode/eagle.py:85-194` — Drafter `__init__` defining
  `needs_extra_input_slots` and `parallel_drafting`.
- `vllm/v1/spec_decode/eagle.py:646-678` — `set_inputs_first_pass`
  default pathway: input_ids shifted by 1, positions held at
  `target_positions` (= `N-1` for the predict slot).
- `vllm/v1/spec_decode/eagle.py:826` — Method dispatch:
  `self.method not in ("mtp", "draft_model", "dflash")` — splits
  EAGLE-style first-pass vs DFlash first-pass.
- `vllm/v1/spec_decode/utils.py:49-66` —
  `eagle_step_slot_mapping_metadata_kernel`: per-step
  `new_position = position + 1`.
- `vllm/v1/spec_decode/utils.py:491,526-527` — DFlash kernel:
  `query_pos = last_target_pos + 1 + offset` (= `N` for the first
  query slot).
