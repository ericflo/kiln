# Phase C2 — MTP Head Forward Bisect (8-tap dump + diff vs HF reference)

**Date**: 2026-04-21
**Branch**: `mtp-c2-head-forward-bisect`
**Kiln commit under test**: `c44c944` (post-PR #311, Phase C1 acceptance-rate attribution merged)
**HF reference checkpoint**: `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
**GPU**: NVIDIA RTX A6000 (RunPod, kiln-pod pool)
**Input**: `--prompt-tokens 32 --max-output-tokens 32 --seed 0 --paged`, `KILN_SPEC_METHOD=mtp`,
`KILN_MTP_DUMP_POS=0,1,2`, `KILN_MTP_DUMP_SUBOPS=1`

## C1 setup (what C2 is answering)

Phase C1 (#311) verdict'd `C2_BISECT_MTP_HEAD`: acceptance α=15.11%, topk_match_rate=15.11%,
Class A (accept-path mask bug)=0, Class B (MTP head draft top-1 ≠ main-head top-1)=281 (of 281
attempts). The accept path itself is correct; the draft logits are wrong. C2 localizes *where*
inside the MTP head forward pass the kiln path diverges from a pure-PyTorch HF reference that
takes the same `h_main` as input.

## Pipeline

1. `kiln-bench` run with MTP dumps written to `/tmp/kiln_pos{pos}.st` for pos=0,1,2 in one
   process. Each dump carries the 8 canonical taps plus 24 sub-op taps and the step's
   `meta__draft_token_id` (matches HF ref on all three positions — see kiln/ref metadata
   comparison below).
2. `scripts/mtp_reference_dump.py --checkpoint <HF> --kiln-dump /tmp/kiln_pos{p}.st
   --out /tmp/ref_pos{p}.st --capture-subops` for p in {0,1,2}. The reference takes kiln's
   `h_main` as its entry point and runs the dual-RMSNorm + fc + single-layer inner block + final
   norm + lm_head path in pure fp32 PyTorch, dumping the same 8 + 24 taps.
3. `scripts/mtp_compare.py --pair pos0:... --pair pos1:... --pair pos2:...` produces the per-
   position table, B7a multi-position summary, and B9 sub-op bisect zones.

Raw artifacts live under `scripts/c2_artifacts/` (6 safetensors + comparator stdout, 7.9 MB).

## Metadata agreement (kiln vs ref)

| pos | meta__mtp_pos | draft_token_id (kiln/ref) | swap_fc_norms |
| --- | ------------- | ------------------------- | ------------- |
| 0   | 0 / 0 ✅      | 561 / 561 ✅              | 0 / 0 ✅      |
| 1   | 1 / 1 ✅      | 37550 / 37550 ✅          | 0 / 0 ✅      |
| 2   | 2 / 2 ✅      | 15217 / 15217 ✅          | 0 / 0 ✅      |

Draft-token agreement means kiln and ref picked the same argmax token from `mtp_logits`, so the
divergence below is not a downstream bug in logit → token selection; it is inside the head.

## 8 canonical MTP-head taps — cos_sim per position

Input side (all clean, bf16-noise level):

| tap              | pos=0 cos_sim | pos=1 cos_sim | pos=2 cos_sim |
| ---------------- | ------------- | ------------- | ------------- |
| `h_main`         | 1.000         | 1.000         | 1.000         |
| `tok_embed`      | 1.000         | 1.000         | 1.000         |
| `fc_input`       | 1.000         | 1.000         | 1.000         |
| `fc_output`      | 1.000         | 1.000         | 1.000         |
| `pre_layer`      | 1.000         | 1.000         | 1.000         |

Output side (monotonic divergence vs `mtp_pos`):

| tap              | pos=0 cos_sim | pos=1 cos_sim | pos=2 cos_sim | max&#124;Δ&#124; pos=2 |
| ---------------- | ------------- | ------------- | ------------- | ---------------------- |
| `post_layer`     | 1.000         | 0.996         | **0.984**     | 8.55e-01               |
| `post_final_ln`  | 1.000         | 0.995         | **0.983**     | 2.31e+00               |
| `mtp_logits`     | 1.000         | 0.996         | **0.988**     | 2.26e+00               |

`post_layer` cos_sim spread across positions: 1.56e-02, monotonically decreasing. All inputs
to the inner transformer block match; all outputs diverge and the divergence grows with
`mtp_pos`. The failure is inside `mtp_inner_block`.

## Sub-op bisect inside `mtp_inner_block`

Pre-RoPE sub-ops (pre-attention norm → q/k/v projections → qk norms) are noise-level across
all three positions — cos_sim = 1.00 for every `post_pre_attn_norm`, `post_q_proj_raw`,
`post_k_proj`, `post_v_proj`, `post_q_split`, `post_qk_norm_q`, `post_qk_norm_k`. Inputs to
RoPE are correct.

Post-RoPE sub-ops (the RoPE call itself):

| sub-op            | pos=0 cos_sim | pos=1 cos_sim | pos=2 cos_sim | pos=2 max&#124;Δ&#124; |
| ----------------- | ------------- | ------------- | ------------- | ---------------------- |
| `post_q_rope`     | 0.922         | 0.935         | **0.834**     | **2.14e+01**           |
| `post_k_rope`     | 0.976         | 0.962         | 0.959         | 8.07e+00               |

`post_q_rope` is the first tap on the kiln/ref comparison where cos_sim drops materially below
1.0 at pos=0, and it drops further at pos=2 where the pre-RoPE inputs were exact matches. The
inner block's RoPE call is being driven with a position value that differs from the reference.

Everything downstream (post_attn_block, post_attn_residual, post_pre_mlp_norm, post_mlp, and
therefore `post_layer` → `post_final_ln` → `mtp_logits`) inherits this error. The
`post_attn_raw` / `post_attn_gated` / `post_o_proj` entries marked MISSING IN KILN DUMP in the
per-pair tables are a dump-format gap in kiln's sub-op set, not a real divergence.

## C2 verdict

**First-divergence tap: `post_q_rope` (matched by `post_k_rope` a hair later).**

Root cause is RoPE position threading inside the MTP head's single-layer inner block. The
B7a-style hypothesis H1 stated in the comparator (`The MTP path's RoPE position threading is
the divergence source`) is reproduced under the current main. cos_sim on `post_q_rope` is 0.92
at pos=0 and degrades to 0.83 at pos=2; pre-RoPE inputs are exact (cos_sim=1.00 across all
three positions and all pre-RoPE sub-ops), so the error is purely in the RoPE computation or
its position argument.

Mapped to canonical 8-tap order, the first-divergence canonical tap is `post_layer`, since
kiln does not currently dump `post_attn_raw`. The sub-op break-in confirms that `post_layer`
divergence is entirely accounted for by the RoPE divergence on its way through the attention
block.

## C3 follow-up recommendation

Open a Phase C3 fix task: thread `mtp_pos` through `mtp_inner_block` to the RoPE call site so
that kiln's RoPE is evaluated at the same position index the HF reference uses. Specifically:

- Inspect the RoPE invocation inside the inner block (call site of
  `crates/kiln-model/src/forward.rs` around `mtp_forward_step` / `mtp_inner_block`) and verify
  it receives `mtp_pos` (not a constant 0, not the outer-sequence position). The HF reference
  uses `position_ids` shifted by `mtp_pos` for each draft.
- Extend kiln's sub-op capture set to include `post_attn_raw`, `post_attn_gated`, and
  `post_o_proj` so the next bisect can close the dump-format gap.
- After the fix, re-run the C2 pipeline on the same prompt+seed; `post_q_rope` and
  `post_k_rope` should return to cos_sim≈1.0 and `post_layer`/`post_final_ln`/`mtp_logits`
  spread should collapse toward bf16 noise. Then re-measure acceptance-rate (C1 harness) and
  expect α to move off the 0.15 floor toward the paper's 0.72 target.

## Artifacts

Under `scripts/c2_artifacts/`:

- `kiln_pos{0,1,2}.st` — kiln dump (8 canonical taps + 24 sub-op taps, bf16)
- `ref_pos{0,1,2}.st` — HF reference dump (same 8 + 24 + `post_attn_raw` / `post_attn_gated`
  / `post_o_proj` that kiln does not emit yet, fp32)
- `c2_compare.txt` — full `mtp_compare.py` stdout including the B7a multi-position summary
  and B9 H2/H3 zone bisect

Total 7.9 MB (well under the 20 MB artifact cap for the PR).
