#!/usr/bin/env python3
"""
Phase B6/B7 — Per-tap numerical comparison of MTP intermediates.

Single-pair mode (Phase B6 — backward compatible):
    python3 scripts/mtp_compare.py \\
        --kiln /tmp/mtp-kiln.safetensors \\
        --ref  /tmp/mtp-ref.safetensors \\
        [--atol 1e-3] [--rtol 1e-2] \\
        [--out /tmp/mtp-compare.txt]

Multi-position mode (Phase B7a — H1 RoPE bisect):
    python3 scripts/mtp_compare.py \\
        --pair 0:/tmp/dump_pos0.st,/tmp/ref_pos0.st \\
        --pair 1:/tmp/dump_pos1.st,/tmp/ref_pos1.st \\
        --pair 2:/tmp/dump_pos2.st,/tmp/ref_pos2.st \\
        [--atol 1e-3] [--rtol 1e-2] \\
        [--out /tmp/mtp-compare-pos.txt]

In multi-position mode, prints a per-position cos_sim table for each tap, then
a B7a verdict line: monotonic degradation in `post_layer` across mtp_pos
implicates H1 (RoPE position threading); flat (invariant) cos_sim across
positions REJECTS H1 and the next step is per-sub-op tap bisect (B7b).

Sub-op tap mode (Phase B7b — fine-grained bisect):
    Sub-op taps are auto-detected. Any non-meta tensor present in both
    dumps that isn't a primary tap is compared at the end of the table
    in the canonical capture order from mtp_inner_block.

Base-model h_main bisect (Phase B10 — per-layer divergence):
    python3 scripts/mtp_compare.py --b10 \\
        --pair main:/tmp/h-kiln.safetensors,/tmp/h-ref.safetensors

    --b10 defaults atol=1e-2, rtol=1e-1 (bf16-appropriate) and emits a
    per-layer cos_sim + max|Δ| table for taps h_layer_{0,8,16,23,31} and
    h_pre_final_norm, plus a verdict identifying the first layer at which
    kiln's base-model hidden state diverges from the HF reference.

Layer-0 GDN sub-op bisect (Phase B11b — per-sub-op divergence):
    python3 scripts/mtp_compare.py --b11 \\
        --pair main:/tmp/h-kiln.safetensors,/tmp/h-ref.safetensors

    --b11 defaults atol=1e-2, rtol=1e-1 (bf16-appropriate) and emits a
    per-sub-op cos_sim + max|Δ| + mean|Δ| + relative_l2 table for the 11
    `b11__<name>` taps written by kiln (KILN_MTP_DUMP_B11_TAPS=1) and by
    mtp_h_main_reference_dump.py --b11-taps. The verdict identifies the
    first sub-op whose median cos_sim across positions falls below 0.95,
    which becomes the B12 recommendation for targeted investigation.

Layer-31 GQA sub-op bisect (Phase B12 — per-layer + per-sub-op drift):
    python3 scripts/mtp_compare.py --b12 \\
        --pair 0:/tmp/h-kiln-p0.safetensors,/tmp/h-ref-p0.safetensors \\
        --pair 1:/tmp/h-kiln-p1.safetensors,/tmp/h-ref-p1.safetensors \\
        --pair 2:/tmp/h-kiln-p2.safetensors,/tmp/h-ref-p2.safetensors

    --b12 defaults atol=1e-2, rtol=1e-1 (bf16-appropriate) and emits TWO
    tables per run: (1) per-layer cos_sim + max|Δ| for h_layer_24..31 and
    h_pre_final_norm, and (2) per-sub-op cos_sim + max|Δ| + mean|Δ| + rel_l2
    for the 14 `b12__<name>` GQA taps captured inside layer 31 by kiln
    (KILN_MTP_DUMP_B12_GQA_TAPS=1) and mtp_h_main_reference_dump.py
    --b12-taps. The verdict identifies the first layer where median cos_sim
    across positions drops below 0.995 (a tight bar chosen so bf16
    accumulation noise does not mask a real bug) and, if that first layer is
    31, the first sub-op below the same bar. Run twice — once with bf16 HF
    refs and once with fp32 HF refs (--b12-taps --fp32) — to tell whether
    residual drift at layer 31 exceeds bf16 accumulation noise or is benign.

Exit code:
    0 — all taps match within tolerance (all positions, in multi mode)
    1 — at least one tap diverges (normal outcome of a bisect)
    2 — structural error (missing file, missing tap, shape mismatch, bad CLI)
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors.numpy import load_file
except ImportError:
    print("ERROR: safetensors not installed. pip install safetensors", file=sys.stderr)
    sys.exit(2)


# Canonical primary tap order — must match the order written by write_mtp_dump
# in mtp_debug.rs and by mtp_reference_dump.py. First divergence is reported by
# this sequence.
TAP_ORDER: List[str] = [
    "h_main",
    "tok_embed",
    "fc_input",
    "fc_output",
    "pre_layer",
    "post_layer",
    "post_final_ln",
    "mtp_logits",
]

# Phase B7b sub-op taps — match capture order in mtp_inner_block (Python ref)
# and gqa_attention_paged + transformer_block_paged (Rust). Any sub-op present
# in both dumps but not listed here gets appended at the end in name-sort order.
SUBOP_ORDER: List[str] = [
    "post_pre_attn_norm",
    "post_q_proj_raw",
    "post_k_proj",
    "post_v_proj",
    # Phase B9 H3 zone: pre/post gated-attn split (mostly aliases of the
    # neighbouring B7b taps, kept distinct so the comparator can localize
    # the H3 hypothesis by name).
    "pre_gated_attn_split",
    "post_q_split",
    "post_gated_attn_split_value",
    "post_gate_split",
    "post_gated_attn_split_gate",
    # Phase B9 H2 zone: pre/post per-head qk-norm.
    "pre_qk_norm_q",
    "pre_qk_norm_k",
    "post_q_norm",
    "post_k_norm",
    "post_qk_norm_q",
    "post_qk_norm_k",
    "post_q_rope",
    "post_k_rope",
    "post_attn_raw",
    "post_attn_gated",
    "post_o_proj",
    "post_attn_block",
    "post_attn_residual",
    "post_pre_mlp_norm",
    "post_mlp",
]

# Phase B9 hypothesis zones — the comparator classifies the first sub-op
# divergence into one of these zones to emit an H2/H3/both/neither verdict.
B9_H2_ZONE = ("pre_qk_norm_q", "pre_qk_norm_k", "post_qk_norm_q", "post_qk_norm_k")
B9_H3_ZONE = (
    "pre_gated_attn_split",
    "post_gated_attn_split_value",
    "post_gated_attn_split_gate",
)

# Phase B10 — per-layer base-model hidden-state taps, in ascending layer order
# so the first entry that diverges identifies the first divergent transformer
# block. `h_pre_final_norm` is the last-row hidden fed into `model.norm`, i.e.
# the same tensor the MTP head consumes as `h_prev`, kept as a distinct tap so
# the comparator can verify the final-layer → h_main handoff independently.
B10_LAYER_TAPS: List[str] = [
    "h_layer_0",
    "h_layer_8",
    "h_layer_16",
    "h_layer_23",
    "h_layer_31",
    "h_pre_final_norm",
]

# Phase B11b — layer-0 GDN sub-op taps, in forward-graph order. Kiln and the
# HF reference both write these under a `b11__` prefix so they slot alongside
# the B10 layer taps without colliding with primary taps. The comparator
# walks these in order and reports the first sub-op whose median cos_sim
# across positions falls below 0.95; that sub-op becomes the B12 target.
B11_LAYER0_TAP_NAMES: Tuple[str, ...] = (
    "tok_embed",
    "layer_0_post_input_norm",
    "gdn_in_proj",
    "gdn_conv",
    "gdn_qk_norm_q",
    "gdn_qk_norm_k",
    "gdn_gate_beta",
    "gdn_gate_g",
    "gdn_recur_out",
    "gdn_gated_norm",
    "gdn_out_proj",
)

# One-line hypotheses for each B11b sub-op, used in verdict output.
B11_HYPOTHESIS: Dict[str, str] = {
    "tok_embed": "embedding lookup: token-id routing, tied-embed weight, or dtype",
    "layer_0_post_input_norm": "layer 0 input_layernorm weight/eps or pre-GDN norm site",
    "gdn_in_proj": "in_proj_qkvz / in_proj_ba weight layout, split order, or dtype",
    "gdn_conv": "causal_conv1d: kernel packing, stride/pad, SiLU activation, or bias",
    "gdn_qk_norm_q": "Qwen3Next in-kernel L2 qk_norm on Q (use_qk_l2norm_in_kernel path)",
    "gdn_qk_norm_k": "Qwen3Next in-kernel L2 qk_norm on K (use_qk_l2norm_in_kernel path)",
    "gdn_gate_beta": "beta gate: sigmoid(b) from in_proj_ba split order / head layout",
    "gdn_gate_g": "g gate: -A_log.exp() * softplus(a + dt_bias), log-gate parameterization",
    "gdn_recur_out": "chunk_gated_delta_rule / fused_recurrent_gated_delta_rule math",
    "gdn_gated_norm": "GatedRMSNorm(core_attn_out, z): gate-modulated RMSNorm weight/eps",
    "gdn_out_proj": "out_proj weight layout, transpose, or residual dtype",
}

# Phase B12 — per-layer taps covering the GQA tail (24..31) + the final-norm
# handoff. B10 already captures layer 0/8/16/23/31; B12 fills in 24..30 so the
# comparator can pinpoint whether the 0.97-0.98 cos_sim drift visible at
# h_layer_31 is introduced by a specific GQA layer or accumulates gradually
# across the 8-layer tail. `h_pre_final_norm` mirrors the B10 meaning (last-row
# hidden feeding model.norm).
B12_LAYER_TAPS: List[str] = [
    "h_layer_24",
    "h_layer_25",
    "h_layer_26",
    "h_layer_27",
    "h_layer_28",
    "h_layer_29",
    "h_layer_30",
    "h_layer_31",
    "h_pre_final_norm",
]

# Phase B12 — layer-31 GQA sub-op taps, in forward-graph order. Must stay in
# lock-step with `B12_GQA_TAP_NAMES` in `crates/kiln-model/src/mtp_debug.rs`
# and `B12_LAYER31_GQA_TAP_NAMES` in `scripts/mtp_h_main_reference_dump.py`.
# Both dumps write these under a `b12__` prefix.
B12_LAYER31_GQA_TAP_NAMES: Tuple[str, ...] = (
    "post_input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_norm_q",
    "qk_norm_k",
    "rope_q",
    "rope_k",
    "attn_out",
    "o_proj",
    "post_attn_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
)

# One-line hypotheses for each B12 layer-31 GQA sub-op, used in verdict output.
B12_HYPOTHESIS: Dict[str, str] = {
    "post_input_norm": "layer 31 input_layernorm weight/eps mismatch",
    "q_proj": "q_proj weight layout, transpose, or Marlin packing",
    "k_proj": "k_proj weight layout, transpose, or Marlin packing",
    "v_proj": "v_proj weight layout, transpose, or Marlin packing",
    "qk_norm_q": "q_norm per-head RMSNorm weight or eps",
    "qk_norm_k": "k_norm per-head RMSNorm weight or eps",
    "rope_q": "RoPE on Q: partial_rotary_factor, rotary theta 10M, half-rotate vs interleaved, position threading",
    "rope_k": "RoPE on K: partial_rotary_factor, rotary theta 10M, half-rotate vs interleaved, position threading",
    "attn_out": "softmax(QK^T)V GQA attention mechanics or FA kernel divergence vs reference",
    "o_proj": "o_proj weight layout, transpose, or residual dtype",
    "post_attn_norm": "layer 31 post_attention_layernorm weight or eps",
    "mlp_gate": "MLP gate_proj weight/transpose or pre-SiLU bf16 accumulation",
    "mlp_up": "MLP up_proj weight layout or elementwise SiLU(gate)*up path",
    "mlp_down": "MLP down_proj weight layout or bf16 GEMM epilogue cast",
}

# Phase B12 — the cos_sim bar is tighter than B10/B11 because the expected
# residual drift is small (0.97-0.98) and the goal is to localize it. A
# per-layer median below this threshold marks the first layer of interest; if
# that layer is 31, the sub-op table identifies the specific op.
B12_COS_SIM_BAR = 0.995

# Phase C6 — pre-RoPE MTP input bisect. Five taps in strict forward-graph
# order. Kiln and the HF reference both write these under a `c6__` prefix so
# they slot alongside the existing B-phase taps without colliding with the
# pre-existing Phase B6 taps (which share 3 of 5 activations under different
# names; C6 keeps them under a self-contained prefix so the bisect table is
# self-contained). Must stay in lock-step with `C6_TAP_NAMES` in
# `crates/kiln-model/src/mtp_debug.rs` and the `out_dict["c6__*"]` writes in
# `scripts/mtp_reference_dump.py`.
C6_PRE_ROPE_TAP_NAMES: Tuple[str, ...] = (
    "token_emb",
    "norm_emb",
    "norm_h",
    "concat",
    "fused",
)

# One-line hypotheses for each C6 pre-RoPE tap, used in verdict output.
C6_HYPOTHESIS: Dict[str, str] = {
    "token_emb": "embedding lookup for draft_token_id: tied embed weight, dtype, or token-id routing",
    "norm_emb": "pre_fc_norm_embedding: RMSNorm weight, eps, or the dual-norm A/B swap",
    "norm_h": "pre_fc_norm_hidden: RMSNorm weight, eps, or the upstream h_prev path",
    "concat": "Tensor::cat([norm_emb, norm_h], dim=2): half ordering / contiguity",
    "fused": "mtp.fc matmul: weight layout, transpose, or bf16 GEMM epilogue",
}

# Phase C6 — same cos_sim bar as C5 bench alpha floor: any tap below 0.9999
# is a first-class divergence candidate. The bisect walks top-down and names
# the first tap to dip below this bar as the dominant source of drift.
C6_COS_SIM_BAR = 0.9999

# Phase C7 — SDPA-internal bisect inside the MTP inner transformer block.
# Seven taps in strict forward-graph order inside scaled dot-product attention.
# Kiln and the HF reference both write these under a `c7__` prefix so they
# slot alongside the B- and C6 taps without collision. Must stay in lock-step
# with `C7_SDPA_TAP_NAMES` in `crates/kiln-model/src/mtp_debug.rs` and the
# `out_dict["c7__*"]` writes in `scripts/mtp_reference_dump.py`.
#
# Shape conventions (decode step, batch=1):
#   pre_sdpa_q              [1, 16, 1, 256]   canonical GQA Q (num_heads=16)
#   pre_sdpa_k              [1, 4, kv_len, 256]  pre-repeat_kv (num_kv_heads=4)
#   pre_sdpa_v              [1, 4, kv_len, 256]  pre-repeat_kv
#   causal_mask             scalar 0 placeholder (decode q_len=1 — no mask)
#   attn_scores_pre_softmax [1, 16, 1, kv_len]
#   attn_probs              [1, 16, 1, kv_len]
#   attn_out                [1, 1, 4096]       same tensor as existing post_attn_raw
C7_SDPA_TAP_NAMES: Tuple[str, ...] = (
    "pre_sdpa_q",
    "pre_sdpa_k",
    "pre_sdpa_v",
    "causal_mask",
    "attn_scores_pre_softmax",
    "attn_probs",
    "attn_out",
)

# One-line hypotheses for each C7 SDPA tap, used in verdict output. Targets
# the most likely root cause at each bisect point inside the inner transformer
# block's scaled-dot-product attention.
C7_HYPOTHESIS: Dict[str, str] = {
    "pre_sdpa_q": "Q projection output / RoPE mtp_pos threading / Q-head layout or transpose",
    "pre_sdpa_k": "K projection output / RoPE mtp_pos threading / KV-head layout or cache write",
    "pre_sdpa_v": "V projection output / KV-head layout or paged-cache write path",
    "causal_mask": "decode should have no causal mask; divergence here = mask policy mismatch",
    "attn_scores_pre_softmax": "scaled QK^T matmul: scale factor, head-grouping reshape, or bf16 GEMM epilogue",
    "attn_probs": "softmax numerical path (cuda_softmax_last_dim vs HF F.softmax)",
    "attn_out": "probs · V matmul or final head-merge reshape (matches existing post_attn_raw)",
}

# Phase C7 — same fp32-friendly cos_sim bar as C6 because most SDPA-internal
# taps are in fp32 on the HF side; the KV cache path materializes bf16 in kiln
# but the tap bar stays strict so the first genuine structural drop (not
# bf16 noise) shows up clearly. If every tap rides above this bar, C7 rules
# SDPA out as a root cause and the bisect advances past attention.
C7_COS_SIM_BAR = 0.9999

META_KEYS = (
    "meta__draft_token_id",
    "meta__mtp_pos",
    "meta__swap_fc_norms",
    "meta__kiln_mtp_pos",
    "meta__capture_subops",
)

# Hypothesis map for the 8 primary taps. Keep in sync with the planning notes
# in mtp_debug.rs / forward.rs.
PRIMARY_HYPOTHESIS = {
    "h_main": "upstream (base model) — this ref takes h_main FROM kiln, so shouldn't differ",
    "tok_embed": "tied embed lookup or token-id path",
    "fc_input": "RMSNorm inputs, concat ordering, or the dual-norm A/B swap",
    "fc_output": "mtp.fc matmul (weight transpose or layout)",
    "pre_layer": "(same tensor as fc_output at this site)",
    "post_layer": "single-layer MTP block: RoPE mtp_pos advancement, Q/K/V proj, or gated-attn",
    "post_final_ln": "mtp.final_layernorm application site / weight",
    "mtp_logits": "tied embed_tokens_t transpose vs alias for LM head",
}

# Hypothesis map for sub-op taps (B7b bisect targets).
SUBOP_HYPOTHESIS = {
    "post_pre_attn_norm": "input_layernorm weight or eps mismatch",
    "post_q_proj_raw": "q_proj weight layout or transpose",
    "post_k_proj": "k_proj weight layout or transpose",
    "post_v_proj": "v_proj weight layout or transpose",
    "post_q_split": "Q/gate split: per-head narrow vs flat-half chunk semantic mismatch",
    "post_gate_split": "Q/gate split: gate half order or layout",
    "pre_gated_attn_split": "B9 H3: q_raw input to gated-attn split (alias of post_q_proj_raw)",
    "post_gated_attn_split_value": "B9 H3: value half after split (alias of post_q_split)",
    "post_gated_attn_split_gate": "B9 H3: gate half after split (alias of post_gate_split)",
    "pre_qk_norm_q": "B9 H2: per-head Q before RMSNorm (alias of post_q_split)",
    "pre_qk_norm_k": "B9 H2: per-head K before RMSNorm (post_k_proj reshaped)",
    "post_qk_norm_q": "B9 H2: per-head Q after RMSNorm (alias of post_q_norm)",
    "post_qk_norm_k": "B9 H2: per-head K after RMSNorm (alias of post_k_norm)",
    "post_q_norm": "q_norm weight or per-head broadcast",
    "post_k_norm": "k_norm weight or per-head broadcast",
    "post_q_rope": "RoPE on Q: mtp_pos value, rotary_dim, half-rotate vs interleaved, theta",
    "post_k_rope": "RoPE on K: mtp_pos value, rotary_dim, half-rotate vs interleaved, theta",
    "post_attn_raw": "softmax(QK^T)V mechanics (single-token reduces to V — divergence here is unusual)",
    "post_attn_gated": "gated attention: sigmoid/silu choice, gate broadcast",
    "post_o_proj": "o_proj weight layout or transpose",
    "post_attn_block": "post-attn output before residual",
    "post_attn_residual": "residual addition site / dtype",
    "post_pre_mlp_norm": "post_attention_layernorm weight or eps",
    "post_mlp": "MLP path: gate/up/down matmul + activation",
}


def _fmt_sci(x: float) -> str:
    if not np.isfinite(x):
        return f"{x}"
    if x == 0.0:
        return "0.00e+00"
    return f"{x:.2e}"


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64).reshape(-1)
    b_f = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a_f)
    nb = np.linalg.norm(b_f)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a_f, b_f) / (na * nb))


def _compare_tap(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    atol: float,
    rtol: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "name": name,
        "shape_kiln": tuple(a.shape),
        "shape_ref": tuple(b.shape),
        "shape_match": tuple(a.shape) == tuple(b.shape),
    }
    if not row["shape_match"]:
        row["allclose"] = False
        row["cos_sim"] = float("nan")
        row["max_abs_diff"] = float("nan")
        row["mean_abs_diff"] = float("nan")
        row["rel_l2"] = float("nan")
        return row

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    diff = np.abs(a64 - b64)
    row["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
    row["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
    row["allclose"] = bool(np.allclose(a64, b64, atol=atol, rtol=rtol))
    row["cos_sim"] = _cos_sim(a64, b64)
    ref_norm = float(np.linalg.norm(b64.reshape(-1)))
    if ref_norm == 0.0 or not diff.size:
        row["rel_l2"] = float("nan")
    else:
        row["rel_l2"] = float(np.linalg.norm(diff.reshape(-1)) / ref_norm)
    return row


def _load(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    try:
        tensors = load_file(path)
    except Exception as e:
        print(f"ERROR: failed to load {path}: {e}", file=sys.stderr)
        sys.exit(2)
    meta: Dict[str, int] = {}
    arrays: Dict[str, np.ndarray] = {}
    for key, val in tensors.items():
        if key in META_KEYS or key.startswith("meta__"):
            meta[key] = int(val.item()) if val.size == 1 else int(val.flatten()[0])
        else:
            arrays[key] = val
    return arrays, meta


def _ordered_taps(
    kiln_arr: Dict[str, np.ndarray],
    ref_arr: Dict[str, np.ndarray],
) -> List[str]:
    """Return tap names in canonical order: primary taps first, then sub-op
    taps in SUBOP_ORDER, then any extras alpha-sorted. Only taps present in
    BOTH dumps are emitted here; missing-on-one-side is reported separately."""
    out: List[str] = []
    common = set(kiln_arr.keys()) & set(ref_arr.keys())
    for name in TAP_ORDER:
        if name in common:
            out.append(name)
    for name in SUBOP_ORDER:
        if name in common:
            out.append(name)
    for name in B10_LAYER_TAPS:
        if name in common and name not in out:
            out.append(name)
    for name in B11_LAYER0_TAP_NAMES:
        key = f"b11__{name}"
        if key in common and key not in out:
            out.append(key)
    for name in B12_LAYER_TAPS:
        if name in common and name not in out:
            out.append(name)
    for name in B12_LAYER31_GQA_TAP_NAMES:
        key = f"b12__{name}"
        if key in common and key not in out:
            out.append(key)
    for name in C6_PRE_ROPE_TAP_NAMES:
        key = f"c6__{name}"
        if key in common and key not in out:
            out.append(key)
    for name in C7_SDPA_TAP_NAMES:
        key = f"c7__{name}"
        if key in common and key not in out:
            out.append(key)
    extras = sorted(common - set(out))
    out.extend(extras)
    return out


def _compare_pair(
    label: str,
    kiln_path: str,
    ref_path: str,
    atol: float,
    rtol: float,
    emit,
) -> Tuple[bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]:
    """Compare a single (kiln, ref) pair. Returns (all_ok, first_divergence,
    rows, kiln_meta, ref_meta). `label` is shown in headers for multi-pair
    runs."""
    kiln_arr, kiln_meta = _load(kiln_path)
    ref_arr, ref_meta = _load(ref_path)

    if label:
        emit(f"\n=== {label} ===")
    emit(f"  kiln dump : {kiln_path}")
    emit(f"  ref  dump : {ref_path}")
    emit("Metadata:")
    for k in META_KEYS:
        kv = kiln_meta.get(k, "<missing>")
        rv = ref_meta.get(k, "<missing>")
        match = "OK" if kv == rv else "MISMATCH"
        emit(f"  {k}: kiln={kv} ref={rv} [{match}]")
    emit("")

    header = (
        f"{'tap':<22} {'shape':<22} {'shape_ok':<8} "
        f"{'allclose':<8} {'cos_sim':<10} {'max|Δ|':<12} {'mean|Δ|':<12}"
    )
    emit(header)
    emit("-" * len(header))

    first_div: str = ""
    all_ok = True
    rows: List[Dict[str, object]] = []
    canonical_order = _ordered_taps(kiln_arr, ref_arr)

    # Report missing-on-one-side first so the table itself is uniform width.
    missing_in_kiln = [n for n in (TAP_ORDER + SUBOP_ORDER) if n in ref_arr and n not in kiln_arr]
    missing_in_ref = [n for n in (TAP_ORDER + SUBOP_ORDER) if n in kiln_arr and n not in ref_arr]
    for n in missing_in_kiln:
        emit(f"{n:<22} MISSING IN KILN DUMP")
        all_ok = False
        if not first_div:
            first_div = n
    for n in missing_in_ref:
        emit(f"{n:<22} MISSING IN REF DUMP")
        all_ok = False
        if not first_div:
            first_div = n

    for name in canonical_order:
        row = _compare_tap(name, kiln_arr[name], ref_arr[name], atol, rtol)
        rows.append(row)
        shape_str = "x".join(str(s) for s in row["shape_kiln"])
        emit(
            f"{row['name']:<22} {shape_str:<22} {str(row['shape_match']):<8} "
            f"{str(row['allclose']):<8} {_fmt_sci(row['cos_sim']):<10} "
            f"{_fmt_sci(row['max_abs_diff']):<12} {_fmt_sci(row['mean_abs_diff']):<12}"
        )
        if not row["allclose"] or not row["shape_match"]:
            all_ok = False
            if not first_div:
                first_div = row["name"]

    emit("")
    if all_ok:
        emit("  pair verdict: all taps match within tolerance.")
    else:
        cause = PRIMARY_HYPOTHESIS.get(first_div) or SUBOP_HYPOTHESIS.get(
            first_div, "<unknown tap>"
        )
        emit(f"  pair verdict: first divergence at tap '{first_div}'.")
        emit(f"    Most-likely cause: {cause}")

    return all_ok, first_div, rows, kiln_meta, ref_meta


def _parse_pair_arg(s: str) -> Tuple[str, str, str]:
    """Parse `LABEL:KILN_PATH,REF_PATH` (LABEL is typically the mtp_pos)."""
    if ":" not in s:
        print(f"ERROR: --pair argument '{s}' missing 'LABEL:' prefix", file=sys.stderr)
        sys.exit(2)
    label, paths = s.split(":", 1)
    if "," not in paths:
        print(f"ERROR: --pair argument '{s}' must be LABEL:KILN,REF", file=sys.stderr)
        sys.exit(2)
    kiln_path, ref_path = paths.split(",", 1)
    return label.strip(), kiln_path.strip(), ref_path.strip()


def _emit_b7a_summary(
    pair_results: List[Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]],
    emit,
) -> None:
    """Print per-position cos_sim summary for the B7a decision: H1 confirmed
    (monotonic degradation in post_layer across mtp_pos) vs rejected
    (cos_sim invariant across positions)."""
    emit("")
    emit("=" * 78)
    emit("Phase B7a multi-position summary — cos_sim per primary tap, per position")
    emit("=" * 78)

    # Build label -> {tap_name: cos_sim} map.
    labels = [pr[0] for pr in pair_results]
    cos_table: Dict[str, Dict[str, float]] = {name: {} for name in TAP_ORDER}
    for label, _ok, _first, rows, _km, _rm in pair_results:
        for r in rows:
            n = r["name"]
            if n in cos_table:
                cos_table[n][label] = float(r["cos_sim"])

    header = "  " + "tap".ljust(20) + " ".join(f"pos={lab:<8}" for lab in labels)
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    for tap in TAP_ORDER:
        cells = " ".join(
            f"{_fmt_sci(cos_table[tap].get(lab, float('nan'))):<12}" for lab in labels
        )
        emit(f"  {tap:<20}{cells}")

    # H1 decision on `post_layer` specifically.
    emit("")
    pl_vals = [cos_table["post_layer"].get(lab, float("nan")) for lab in labels]
    finite = [v for v in pl_vals if np.isfinite(v)]
    if len(finite) < 2:
        emit("Phase B7a verdict: insufficient finite post_layer cos_sim values to decide.")
        return
    spread = max(finite) - min(finite)
    monotonic_down = all(pl_vals[i] >= pl_vals[i + 1] - 1e-6 for i in range(len(pl_vals) - 1)) and (
        pl_vals[0] - pl_vals[-1] > 1e-3
    )
    invariant = spread < 1e-3
    emit(f"  post_layer cos_sim across positions: {[round(v, 6) for v in pl_vals]}")
    emit(f"  spread (max-min): {_fmt_sci(spread)}")
    if invariant:
        emit("Phase B7a verdict: H1 REJECTED — post_layer cos_sim is essentially invariant")
        emit("  across mtp_pos. RoPE-position threading is NOT the divergence source.")
        emit("  -> Proceed to Phase B7b (per-sub-op tap bisect).")
    elif monotonic_down:
        emit("Phase B7a verdict: H1 CONFIRMED — post_layer cos_sim degrades monotonically")
        emit("  with mtp_pos. The MTP path's RoPE position threading is the divergence")
        emit("  source. -> Land Phase B7 PR with this finding; queue B8 fix task to")
        emit("  thread the correct mtp_pos through the inner block's RoPE call site.")
    else:
        emit("Phase B7a verdict: INCONCLUSIVE — post_layer cos_sim varies across positions")
        emit("  but not monotonically. May indicate a position-dependent issue mixed with")
        emit("  another bug. Recommend running B7b anyway and re-evaluating.")


def _emit_b9_summary(
    pair_results: List[Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]],
    atol: float,
    rtol: float,
    emit,
) -> None:
    """Phase B9 — H2 (qk-norm) vs H3 (gated-attn split) bisect.

    Walks the canonical sub-op order across every position and finds the first
    diverging sub-op tap. If that tap lands in the H2 zone, prints
    'H2 dominant'; if in the H3 zone, 'H3 dominant'; if BOTH zones diverge in
    one or more positions, prints 'BOTH'; if neither zone diverges, prints
    'NEITHER (look upstream)'. Per-sub-op cos_sim + max|Δ| table per position
    is also emitted so the verdict is auditable.
    """
    emit("")
    emit("=" * 78)
    emit("Phase B9 H2/H3 sub-op bisect — per-position cos_sim + max|Δ|")
    emit("=" * 78)

    labels = [pr[0] for pr in pair_results]

    # B9 zone tap names in canonical capture order. We restrict the table to
    # zone taps so the verdict is unambiguous; the full sub-op table for
    # context already prints in the per-pair section above.
    zone_taps: List[str] = []
    for n in SUBOP_ORDER:
        if n in B9_H2_ZONE or n in B9_H3_ZONE:
            zone_taps.append(n)

    # Build {tap: {label: (cos_sim, max_abs_diff, allclose)}} from rows.
    cell: Dict[str, Dict[str, Tuple[float, float, bool]]] = {n: {} for n in zone_taps}
    for label, _ok, _fd, rows, _km, _rm in pair_results:
        for r in rows:
            n = r["name"]
            if n in cell:
                cell[n][label] = (
                    float(r["cos_sim"]),
                    float(r["max_abs_diff"]),
                    bool(r["allclose"]),
                )

    # Header.
    header = "  " + "tap".ljust(30) + " ".join(
        f"pos={lab:<22}" for lab in labels
    )
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    for tap in zone_taps:
        zone_marker = "[H2]" if tap in B9_H2_ZONE else "[H3]"
        cells = []
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                cells.append(f"{'<missing>':<26}")
            else:
                cs, mxd, ok = entry
                tag = "ok" if ok else "DIV"
                cells.append(
                    f"cos={_fmt_sci(cs):<10} max|Δ|={_fmt_sci(mxd):<8} {tag:<3} "
                )
        emit(f"  {zone_marker} {tap:<26}{' '.join(cells)}")

    # Decide H2 / H3 / both / neither based on whether ANY pair has a divergent
    # tap in each zone, keyed off the row-level allclose flag the comparator
    # already computed at the requested atol/rtol.
    h2_div = any(
        not entry[2]
        for tap in B9_H2_ZONE
        for entry in cell.get(tap, {}).values()
    )
    h3_div = any(
        not entry[2]
        for tap in B9_H3_ZONE
        for entry in cell.get(tap, {}).values()
    )

    # First-diverging zone tap in canonical order, taken across all positions.
    first_zone_div: Optional[str] = None
    first_zone_pos: Optional[str] = None
    for tap in zone_taps:
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is not None and not entry[2]:
                first_zone_div = tap
                first_zone_pos = lab
                break
        if first_zone_div is not None:
            break

    emit("")
    emit(f"  H2 zone (qk-norm) divergence present:   {h2_div}")
    emit(f"  H3 zone (gated-attn split) divergence present: {h3_div}")
    if first_zone_div is not None:
        emit(f"  First zone-tap divergence in canonical order: '{first_zone_div}' (pos={first_zone_pos})")

    if h2_div and h3_div:
        # Tie-break by which zone the canonical-first-diverging tap lands in.
        if first_zone_div in B9_H2_ZONE:
            emit("Phase B9 verdict: BOTH zones diverge; canonical-first divergence is in H2 zone")
            emit("  -> H2 (qk-norm semantics) is the upstream cause; H3 likely inherits.")
            emit("  -> Land Phase B9 PR with this finding; queue B10 fix targeting qk-norm.")
        elif first_zone_div in B9_H3_ZONE:
            emit("Phase B9 verdict: BOTH zones diverge; canonical-first divergence is in H3 zone")
            emit("  -> H3 (gated-attn split semantics) is the upstream cause; H2 likely inherits.")
            emit("  -> Land Phase B9 PR with this finding; queue B10 fix targeting gated-attn split.")
        else:
            emit("Phase B9 verdict: BOTH zones diverge; cannot pick canonical-first cleanly.")
            emit("  -> Land Phase B9 PR with this finding; queue B10 to investigate both zones.")
    elif h2_div:
        emit("Phase B9 verdict: H2 DOMINANT — qk-norm zone diverges, gated-attn split zone matches.")
        emit("  -> H2 (qk-norm semantics) drives the post_layer divergence on this pair.")
        emit("  -> Land Phase B9 PR with this finding; queue B10 fix targeting qk-norm.")
    elif h3_div:
        emit("Phase B9 verdict: H3 DOMINANT — gated-attn split zone diverges, qk-norm zone matches.")
        emit("  -> H3 (gated-attn split semantics) drives the post_layer divergence on this pair.")
        emit("  -> Land Phase B9 PR with this finding; queue B10 fix targeting gated-attn split.")
    else:
        emit("Phase B9 verdict: NEITHER zone diverges in the H2/H3 bisect.")
        emit("  -> The post_layer divergence is upstream of qk-norm and the gated-attn split,")
        emit("     OR the divergence emerges only at a downstream tap (RoPE/attn/o_proj/MLP).")
        emit("  -> Re-check the full sub-op table above for the first non-zone divergence.")


def _emit_b10_summary(
    pair_results: List[Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]],
    atol: float,
    rtol: float,
    emit,
) -> None:
    """Phase B10 — per-layer base-model hidden-state bisect.

    Walks `B10_LAYER_TAPS` in ascending order across every pair and reports
    the first layer at which kiln's hidden state diverges from the HF
    reference. Because the full 32-layer main stack is being exercised, the
    first divergence pinpoints either the first bad transformer block or the
    final-layer → h_main handoff (when `h_pre_final_norm` is the first
    diverging tap but `h_layer_31` matched).
    """
    emit("")
    emit("=" * 78)
    emit("Phase B10 per-layer base-model h_main bisect — cos_sim + max|Δ|")
    emit("=" * 78)

    labels = [pr[0] for pr in pair_results]

    cell: Dict[str, Dict[str, Tuple[float, float, bool, Tuple[int, ...]]]] = {
        n: {} for n in B10_LAYER_TAPS
    }
    for label, _ok, _fd, rows, _km, _rm in pair_results:
        for r in rows:
            n = r["name"]
            if n in cell:
                cell[n][label] = (
                    float(r["cos_sim"]),
                    float(r["max_abs_diff"]),
                    bool(r["allclose"]),
                    tuple(r["shape_kiln"]),
                )

    header = "  " + "tap".ljust(22) + " ".join(
        f"pos={lab:<28}" for lab in labels
    )
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    captured_any = False
    for tap in B10_LAYER_TAPS:
        cells = []
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                cells.append(f"{'<missing>':<32}")
            else:
                cs, mxd, ok, _shape = entry
                captured_any = True
                tag = "ok" if ok else "DIV"
                cells.append(
                    f"cos={_fmt_sci(cs):<10} max|Δ|={_fmt_sci(mxd):<10} {tag:<3}  "
                )
        emit(f"  {tap:<22}{' '.join(cells)}")

    emit("")
    if not captured_any:
        emit("Phase B10 verdict: no B10 layer taps present in dumps.")
        emit("  -> Re-run kiln-bench with KILN_MTP_DUMP_HIDDEN_STATES=1, and re-run")
        emit("     mtp_h_main_reference_dump.py against the same kiln dump.")
        return

    first_div_tap: Optional[str] = None
    first_div_pos: Optional[str] = None
    last_match_tap: Optional[str] = None
    for tap in B10_LAYER_TAPS:
        any_div_here = False
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                continue
            if not entry[2]:
                any_div_here = True
                if first_div_tap is None:
                    first_div_tap = tap
                    first_div_pos = lab
        if not any_div_here and first_div_tap is None:
            last_match_tap = tap

    if first_div_tap is None:
        emit("Phase B10 verdict: all layer taps match within tolerance.")
        emit(f"  (atol={atol}, rtol={rtol} — bf16-appropriate)")
        emit("  -> Base-model h_main is NUMERICALLY CLEAN against the HF reference.")
        emit("  -> Divergence must be introduced downstream of model.norm, i.e. in")
        emit("     the MTP head itself. Re-inspect B6–B9 outcomes and the final")
        emit("     tokenizer/sampling path; the base stack is exonerated.")
        return

    emit(f"  first diverging tap: '{first_div_tap}' (pos={first_div_pos})")
    if last_match_tap is not None:
        emit(f"  last matching tap before divergence: '{last_match_tap}'")

    if first_div_tap == "h_layer_0":
        emit("Phase B10 verdict: DIVERGENCE AT LAYER 0")
        emit("  -> Input-path bug: embedding lookup, rotary_inv_freq build, first")
        emit("     block's Q/K/V proj or GDN sub-op, OR prompt-token construction")
        emit("     mismatch between kiln and reference. Verify `prompt_tokens` meta")
        emit("     matches exactly, then narrow to layer 0's sub-ops.")
    elif first_div_tap in ("h_layer_8", "h_layer_16"):
        emit("Phase B10 verdict: DIVERGENCE IN EARLY GDN STACK")
        emit(f"  -> The first bad transformer block lies between {last_match_tap or 'start'}")
        emit(f"     and {first_div_tap}. Narrow by capturing intermediate layers")
        emit("     (e.g. layers 4 and 12) and re-running B10 on the new dump.")
    elif first_div_tap == "h_layer_23":
        emit("Phase B10 verdict: DIVERGENCE IN MID-STACK (GDN→GQA transition zone)")
        emit(f"  -> The first bad block lies between {last_match_tap or 'start'} and 23.")
        emit("     Layer 23 is a GQA block (full_attention_interval=4). Narrow by")
        emit("     capturing layers 20, 21, 22 on the next dump.")
    elif first_div_tap == "h_layer_31":
        emit("Phase B10 verdict: DIVERGENCE IN LATE STACK")
        emit(f"  -> The first bad block lies between {last_match_tap or 'start'} and 31.")
        emit("     Narrow by capturing layers 24, 27, 29 on the next dump.")
    elif first_div_tap == "h_pre_final_norm":
        emit("Phase B10 verdict: DIVERGENCE AT FINAL-LAYER → h_main HANDOFF")
        emit("  -> All 32 layers match, but `h_pre_final_norm` (what MTP consumes")
        emit("     as `h_prev`) diverges. The bug is in the last-row extraction")
        emit("     (`narrow(1, seq_len-1, 1)`), the residual add at layer 31, or")
        emit("     how the base-model main path routes hidden into the MTP head.")


def _emit_b11_summary(
    pair_results: List[Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]],
    atol: float,
    rtol: float,
    emit,
) -> None:
    """Phase B11b — layer-0 GDN sub-op bisect.

    Walks `B11_LAYER0_TAP_NAMES` (prefixed with `b11__` in both dumps) across
    every pair, prints a per-position cos_sim / max|Δ| / mean|Δ| / rel_l2
    table, and recommends the first sub-op whose MEDIAN cos_sim across
    positions falls below 0.95 as the B12 target. Falls back to the first
    tap with ANY position diverging when no tap crosses the 0.95 median bar
    so the recommendation stays actionable when every sub-op is noisy.
    """
    emit("")
    emit("=" * 78)
    emit("Phase B11b layer-0 GDN sub-op bisect — cos_sim / max|Δ| / mean|Δ| / rel_l2")
    emit("=" * 78)

    labels = [pr[0] for pr in pair_results]

    # tap -> {label: (cos_sim, max|Δ|, mean|Δ|, rel_l2)}
    cell: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
        n: {} for n in B11_LAYER0_TAP_NAMES
    }
    for label, _ok, _fd, rows, _km, _rm in pair_results:
        for r in rows:
            n = r["name"]
            if not isinstance(n, str) or not n.startswith("b11__"):
                continue
            short = n[len("b11__"):]
            if short not in cell:
                continue
            cell[short][label] = (
                float(r["cos_sim"]),
                float(r["max_abs_diff"]),
                float(r["mean_abs_diff"]),
                float(r.get("rel_l2", float("nan"))),
            )

    header = "  " + "tap".ljust(26) + " ".join(
        f"pos={lab:<48}" for lab in labels
    )
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    captured_any = False
    for tap in B11_LAYER0_TAP_NAMES:
        cells = []
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                cells.append(f"{'<missing>':<52}")
            else:
                cs, mxd, mnd, rl2 = entry
                captured_any = True
                tag = "ok" if cs >= 0.95 else "DIV"
                cells.append(
                    f"cos={_fmt_sci(cs):<10} max|Δ|={_fmt_sci(mxd):<10} "
                    f"mean|Δ|={_fmt_sci(mnd):<10} rel_l2={_fmt_sci(rl2):<10} {tag:<3}"
                )
        emit(f"  {tap:<26}{' '.join(cells)}")

    emit("")
    if not captured_any:
        emit("Phase B11b verdict: no B11 layer-0 taps present in dumps.")
        emit("  -> Re-run kiln-bench with KILN_MTP_DUMP_HIDDEN_STATES=1 AND")
        emit("     KILN_MTP_DUMP_B11_TAPS=1, then re-run mtp_h_main_reference_dump.py")
        emit("     with --b11-taps against the same kiln dump.")
        return

    # B12 recommendation: first tap whose MEDIAN cos_sim across finite
    # positions dips below 0.95. Falls back to first tap with ANY position
    # diverging under the atol/rtol bar when no tap crosses the 0.95 line.
    def _median(xs: List[float]) -> float:
        finite = sorted(v for v in xs if np.isfinite(v))
        if not finite:
            return float("nan")
        n = len(finite)
        mid = n // 2
        if n % 2 == 1:
            return finite[mid]
        return 0.5 * (finite[mid - 1] + finite[mid])

    first_median_div: Optional[str] = None
    first_median_val: float = float("nan")
    first_any_div: Optional[str] = None
    first_any_pos: Optional[str] = None
    for tap in B11_LAYER0_TAP_NAMES:
        per_pos = [cell[tap][lab][0] for lab in labels if lab in cell[tap]]
        if not per_pos:
            continue
        med = _median(per_pos)
        if first_median_div is None and np.isfinite(med) and med < 0.95:
            first_median_div = tap
            first_median_val = med
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                continue
            cs = entry[0]
            if np.isfinite(cs) and cs < 0.95 and first_any_div is None:
                first_any_div = tap
                first_any_pos = lab

    if first_median_div is not None:
        hypo = B11_HYPOTHESIS.get(first_median_div, "<unknown sub-op>")
        emit(
            f"  first sub-op with MEDIAN cos_sim < 0.95: '{first_median_div}' "
            f"(median cos_sim = {first_median_val:.6f})"
        )
        emit(f"Phase B11b verdict: B12 TARGET = '{first_median_div}'.")
        emit(f"    Most-likely cause: {hypo}")
        emit("  -> Queue B12 task to investigate this sub-op in isolation.")
    elif first_any_div is not None:
        hypo = B11_HYPOTHESIS.get(first_any_div, "<unknown sub-op>")
        emit(
            f"  no sub-op has MEDIAN cos_sim < 0.95, but '{first_any_div}' "
            f"(pos={first_any_pos}) dips below 0.95 on at least one position."
        )
        emit(f"Phase B11b verdict: B12 TARGET (noisy) = '{first_any_div}'.")
        emit(f"    Most-likely cause: {hypo}")
        emit("  -> Recommend capturing more positions / longer prompts to confirm")
        emit("     before committing to a B12 fix direction.")
    else:
        emit("Phase B11b verdict: ALL layer-0 GDN sub-ops match within cos_sim >= 0.95.")
        emit("  -> Divergence is NOT localized inside layer 0's GDN block at this bar.")
        emit("  -> Re-check B10 atol/rtol and consider extending taps to layers 1-7")
        emit("     (GDN stack) before committing to a B12 target.")


def _emit_b12_summary(
    pair_results: List[Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]],
    atol: float,
    rtol: float,
    emit,
) -> None:
    """Phase B12 — layer-31 drift bisect.

    Walks `B12_LAYER_TAPS` (h_layer_24..31 + h_pre_final_norm) across every
    pair and prints a per-layer cos_sim + max|Δ| table. Then, if any b12__
    sub-op taps are present, walks `B12_LAYER31_GQA_TAP_NAMES` and prints a
    per-sub-op cos_sim / max|Δ| / mean|Δ| / rel_l2 table.

    Verdict:
      - If every layer's MEDIAN cos_sim across positions is >= `B12_COS_SIM_BAR`
        (0.995), the drift is within the tight bar and no specific layer is
        called out — compare against fp32 HF to decide if bf16 accumulation is
        the sole explanation.
      - Otherwise, the first layer whose median falls below the bar is
        reported as the first divergent layer. If that layer is 31 AND b12__
        sub-op taps are present, the sub-op table is used to recommend the
        first sub-op (in forward-graph order) with median cos_sim below the
        bar as the localized B12 target.
    """
    emit("")
    emit("=" * 78)
    emit("Phase B12 layer-31 drift bisect — per-layer cos_sim + max|Δ|")
    emit("=" * 78)

    labels = [pr[0] for pr in pair_results]

    # Layer table: tap -> {label: (cos_sim, max|Δ|, allclose)}
    layer_cell: Dict[str, Dict[str, Tuple[float, float, bool]]] = {
        n: {} for n in B12_LAYER_TAPS
    }
    for label, _ok, _fd, rows, _km, _rm in pair_results:
        for r in rows:
            n = r["name"]
            if isinstance(n, str) and n in layer_cell:
                layer_cell[n][label] = (
                    float(r["cos_sim"]),
                    float(r["max_abs_diff"]),
                    bool(r["allclose"]),
                )

    header = "  " + "tap".ljust(22) + " ".join(
        f"pos={lab:<28}" for lab in labels
    )
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    captured_any_layer = False
    for tap in B12_LAYER_TAPS:
        cells = []
        for lab in labels:
            entry = layer_cell[tap].get(lab)
            if entry is None:
                cells.append(f"{'<missing>':<32}")
            else:
                cs, mxd, ok = entry
                captured_any_layer = True
                tag = "ok" if cs >= B12_COS_SIM_BAR else "DIV"
                cells.append(
                    f"cos={_fmt_sci(cs):<10} max|Δ|={_fmt_sci(mxd):<10} {tag:<3}  "
                )
        emit(f"  {tap:<22}{' '.join(cells)}")

    emit("")
    if not captured_any_layer:
        emit("Phase B12 verdict: no B12 layer taps present in dumps.")
        emit("  -> Re-run kiln-bench with KILN_MTP_DUMP_HIDDEN_STATES=1 AND")
        emit("     KILN_MTP_DUMP_B12_GQA_TAPS=1, then re-run mtp_h_main_reference_dump.py")
        emit("     --b12-taps against the same kiln dump.")
        return

    def _median(xs: List[float]) -> float:
        finite = sorted(v for v in xs if np.isfinite(v))
        if not finite:
            return float("nan")
        n = len(finite)
        mid = n // 2
        if n % 2 == 1:
            return finite[mid]
        return 0.5 * (finite[mid - 1] + finite[mid])

    # Per-layer medians — find the first layer whose median falls below the bar.
    first_layer_below: Optional[str] = None
    first_layer_median: float = float("nan")
    layer_medians: List[Tuple[str, float]] = []
    for tap in B12_LAYER_TAPS:
        per_pos = [layer_cell[tap][lab][0] for lab in labels if lab in layer_cell[tap]]
        if not per_pos:
            continue
        med = _median(per_pos)
        layer_medians.append((tap, med))
        if first_layer_below is None and np.isfinite(med) and med < B12_COS_SIM_BAR:
            first_layer_below = tap
            first_layer_median = med

    emit("  per-layer median cos_sim across positions:")
    for tap, med in layer_medians:
        bar = "ok" if np.isfinite(med) and med >= B12_COS_SIM_BAR else "DIV"
        emit(f"    {tap:<22} median={_fmt_sci(med):<12} {bar}")

    # Sub-op table: only if b12__ taps are present.
    subop_cell: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
        n: {} for n in B12_LAYER31_GQA_TAP_NAMES
    }
    captured_any_subop = False
    for label, _ok, _fd, rows, _km, _rm in pair_results:
        for r in rows:
            n = r["name"]
            if not isinstance(n, str) or not n.startswith("b12__"):
                continue
            short = n[len("b12__"):]
            if short not in subop_cell:
                continue
            captured_any_subop = True
            subop_cell[short][label] = (
                float(r["cos_sim"]),
                float(r["max_abs_diff"]),
                float(r["mean_abs_diff"]),
                float(r.get("rel_l2", float("nan"))),
            )

    first_subop_below: Optional[str] = None
    first_subop_median: float = float("nan")
    if captured_any_subop:
        emit("")
        emit("=" * 78)
        emit("Phase B12 layer-31 GQA sub-op bisect — cos_sim / max|Δ| / mean|Δ| / rel_l2")
        emit("=" * 78)
        subop_header = "  " + "tap".ljust(20) + " ".join(
            f"pos={lab:<48}" for lab in labels
        )
        emit(subop_header)
        emit("  " + "-" * (len(subop_header) - 2))
        for tap in B12_LAYER31_GQA_TAP_NAMES:
            cells = []
            for lab in labels:
                entry = subop_cell[tap].get(lab)
                if entry is None:
                    cells.append(f"{'<missing>':<52}")
                else:
                    cs, mxd, mnd, rl2 = entry
                    tag = "ok" if cs >= B12_COS_SIM_BAR else "DIV"
                    cells.append(
                        f"cos={_fmt_sci(cs):<10} max|Δ|={_fmt_sci(mxd):<10} "
                        f"mean|Δ|={_fmt_sci(mnd):<10} rel_l2={_fmt_sci(rl2):<10} {tag:<3}"
                    )
            emit(f"  {tap:<20}{' '.join(cells)}")

        emit("")
        emit("  per-sub-op median cos_sim across positions:")
        for tap in B12_LAYER31_GQA_TAP_NAMES:
            per_pos = [subop_cell[tap][lab][0] for lab in labels if lab in subop_cell[tap]]
            if not per_pos:
                continue
            med = _median(per_pos)
            bar = "ok" if np.isfinite(med) and med >= B12_COS_SIM_BAR else "DIV"
            emit(f"    {tap:<20} median={_fmt_sci(med):<12} {bar}")
            if first_subop_below is None and np.isfinite(med) and med < B12_COS_SIM_BAR:
                first_subop_below = tap
                first_subop_median = med

    emit("")
    if first_layer_below is None:
        emit(
            f"Phase B12 verdict: all layers in 24..31 + h_pre_final_norm have median "
            f"cos_sim >= {B12_COS_SIM_BAR}."
        )
        emit("  -> No layer-localized drift at this bar. Compare this run against a")
        emit("     fp32 HF reference too (re-run the Python dumper with --fp32).")
        emit("     If fp32 HF also clears this bar, the drift was benign bf16")
        emit("     accumulation. If fp32 HF drops below it, a real bug remains.")
        return

    emit(
        f"  first layer with MEDIAN cos_sim < {B12_COS_SIM_BAR}: "
        f"'{first_layer_below}' (median = {first_layer_median:.6f})"
    )

    if first_layer_below != "h_layer_31":
        emit(
            f"Phase B12 verdict: DRIFT FIRST APPEARS AT '{first_layer_below}' "
            f"(not layer 31)."
        )
        emit("  -> The sub-op table for layer 31 localizes only layer-31 drift, so")
        emit(f"     instrument sub-ops inside '{first_layer_below}' on the next pass.")
        emit("     Compare against fp32 HF to rule out bf16 accumulation before")
        emit("     committing more capture budget.")
        return

    if not captured_any_subop:
        emit("Phase B12 verdict: LAYER 31 IS THE FIRST DIVERGENT LAYER, but no b12__")
        emit("  sub-op taps were captured, so no localized target is available.")
        emit("  -> Re-run both dumps with the b12__ sub-op tap env flags set.")
        return

    if first_subop_below is not None:
        hypo = B12_HYPOTHESIS.get(first_subop_below, "<unknown sub-op>")
        emit(
            f"  first sub-op with MEDIAN cos_sim < {B12_COS_SIM_BAR}: "
            f"'{first_subop_below}' (median = {first_subop_median:.6f})"
        )
        emit(f"Phase B12 verdict: B12 TARGET = layer 31 / '{first_subop_below}'.")
        emit(f"    Most-likely cause: {hypo}")
        emit("  -> Compare this run against fp32 HF next. If the same sub-op is")
        emit("     still the first below-bar entry, queue a targeted B13 fix task.")
        emit("     If fp32 HF clears the bar, the drift is benign bf16 accumulation.")
    else:
        emit("Phase B12 verdict: LAYER 31 median is below-bar, but NO individual sub-op")
        emit(f"  at layer 31 has median cos_sim < {B12_COS_SIM_BAR}.")
        emit("  -> Drift is spread across multiple sub-ops (accumulation pattern).")
        emit("     Strong signal for benign bf16 accumulation. Confirm against fp32 HF.")


def _emit_c6_summary(
    pair_results: List[Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]],
    atol: float,
    rtol: float,
    emit,
) -> None:
    """Phase C6 — pre-RoPE MTP input bisect.

    Walks `C6_PRE_ROPE_TAP_NAMES` (prefixed with `c6__` in both dumps) in
    strict forward-graph order across every pair, prints a per-position
    cos_sim / max|Δ| / mean|Δ| / rel_l2 table, and names the FIRST tap
    whose MEDIAN cos_sim across positions falls below C6_COS_SIM_BAR
    (0.9999) as the dominant source of pre-RoPE drift. Falls back to the
    first tap with ANY position diverging when no tap's median crosses
    the bar.
    """
    emit("")
    emit("=" * 78)
    emit("Phase C6 pre-RoPE MTP input bisect — cos_sim / max|Δ| / mean|Δ| / rel_l2")
    emit("=" * 78)

    labels = [pr[0] for pr in pair_results]

    cell: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
        n: {} for n in C6_PRE_ROPE_TAP_NAMES
    }
    for label, _ok, _fd, rows, _km, _rm in pair_results:
        for r in rows:
            n = r["name"]
            if not isinstance(n, str) or not n.startswith("c6__"):
                continue
            short = n[len("c6__"):]
            if short not in cell:
                continue
            cell[short][label] = (
                float(r["cos_sim"]),
                float(r["max_abs_diff"]),
                float(r["mean_abs_diff"]),
                float(r.get("rel_l2", float("nan"))),
            )

    header = "  " + "tap".ljust(14) + " ".join(
        f"pos={lab:<48}" for lab in labels
    )
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    captured_any = False
    for tap in C6_PRE_ROPE_TAP_NAMES:
        cells = []
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                cells.append(f"{'<missing>':<52}")
            else:
                cs, mxd, mnd, rl2 = entry
                captured_any = True
                tag = "ok" if cs >= C6_COS_SIM_BAR else "DIV"
                cells.append(
                    f"cos={_fmt_sci(cs):<10} max|Δ|={_fmt_sci(mxd):<10} "
                    f"mean|Δ|={_fmt_sci(mnd):<10} rel_l2={_fmt_sci(rl2):<10} {tag:<3}"
                )
        emit(f"  {tap:<14}{' '.join(cells)}")

    emit("")
    if not captured_any:
        emit("Phase C6 verdict: no C6 pre-RoPE taps present in dumps.")
        emit("  -> Re-run kiln-bench with KILN_MTP_DUMP_POS=0,2 AND")
        emit("     KILN_MTP_DUMP_PRE_ROPE=1 set, and re-run scripts/mtp_reference_dump.py")
        emit("     against the same kiln dump (the reference always emits c6__ taps).")
        return

    def _median(xs: List[float]) -> float:
        finite = sorted(v for v in xs if np.isfinite(v))
        if not finite:
            return float("nan")
        n = len(finite)
        mid = n // 2
        if n % 2 == 1:
            return finite[mid]
        return 0.5 * (finite[mid - 1] + finite[mid])

    first_median_div: Optional[str] = None
    first_median_val: float = float("nan")
    first_any_div: Optional[str] = None
    first_any_pos: Optional[str] = None
    for tap in C6_PRE_ROPE_TAP_NAMES:
        per_pos = [cell[tap][lab][0] for lab in labels if lab in cell[tap]]
        if not per_pos:
            continue
        med = _median(per_pos)
        if first_median_div is None and np.isfinite(med) and med < C6_COS_SIM_BAR:
            first_median_div = tap
            first_median_val = med
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                continue
            cs = entry[0]
            if np.isfinite(cs) and cs < C6_COS_SIM_BAR and first_any_div is None:
                first_any_div = tap
                first_any_pos = lab

    if first_median_div is not None:
        hypo = C6_HYPOTHESIS.get(first_median_div, "<unknown tap>")
        emit(
            f"  first tap with MEDIAN cos_sim < {C6_COS_SIM_BAR}: '{first_median_div}' "
            f"(median cos_sim = {first_median_val:.6f})"
        )
        emit(f"Phase C6 verdict: PRE-ROPE TARGET = '{first_median_div}'.")
        emit(f"    Most-likely cause: {hypo}")
        emit("  -> Queue the follow-up phase to investigate this tap in isolation.")
    elif first_any_div is not None:
        hypo = C6_HYPOTHESIS.get(first_any_div, "<unknown tap>")
        emit(
            f"  no tap has MEDIAN cos_sim < {C6_COS_SIM_BAR}, but '{first_any_div}' "
            f"(pos={first_any_pos}) dips below {C6_COS_SIM_BAR} on at least one position."
        )
        emit(f"Phase C6 verdict: PRE-ROPE TARGET (noisy) = '{first_any_div}'.")
        emit(f"    Most-likely cause: {hypo}")
        emit("  -> Recommend capturing more positions / seeds before committing")
        emit("     to a C7 fix direction.")
    else:
        emit(f"Phase C6 verdict: ALL pre-RoPE taps match within cos_sim >= {C6_COS_SIM_BAR}.")
        emit("  -> Pre-RoPE input is NOT the dominant source of drift at this bar.")
        emit("  -> Look downstream of `fused` (RoPE / attention / final_layernorm)")
        emit("     and/or re-check the abs_pos threading in the MTP block.")


def _emit_c7_summary(
    pair_results: List[Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]],
    atol: float,
    rtol: float,
    emit,
) -> None:
    """Phase C7 — SDPA-internal bisect inside MTP inner transformer block.

    Walks `C7_SDPA_TAP_NAMES` (prefixed with `c7__` in both dumps) in strict
    forward-graph order across every pair, prints a per-position cos_sim /
    max|Δ| / mean|Δ| / rel_l2 table, and names the FIRST tap whose MEDIAN
    cos_sim across positions falls below C7_COS_SIM_BAR (0.9999) as the
    dominant source of SDPA drift. Falls back to the first tap with ANY
    position diverging when no tap's median crosses the bar.
    """
    emit("")
    emit("=" * 78)
    emit("Phase C7 SDPA-internal bisect — cos_sim / max|Δ| / mean|Δ| / rel_l2")
    emit("=" * 78)

    labels = [pr[0] for pr in pair_results]

    cell: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
        n: {} for n in C7_SDPA_TAP_NAMES
    }
    for label, _ok, _fd, rows, _km, _rm in pair_results:
        for r in rows:
            n = r["name"]
            if not isinstance(n, str) or not n.startswith("c7__"):
                continue
            short = n[len("c7__"):]
            if short not in cell:
                continue
            cell[short][label] = (
                float(r["cos_sim"]),
                float(r["max_abs_diff"]),
                float(r["mean_abs_diff"]),
                float(r.get("rel_l2", float("nan"))),
            )

    header = "  " + "tap".ljust(26) + " ".join(
        f"pos={lab:<48}" for lab in labels
    )
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    captured_any = False
    for tap in C7_SDPA_TAP_NAMES:
        cells = []
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                cells.append(f"{'<missing>':<52}")
            else:
                cs, mxd, mnd, rl2 = entry
                captured_any = True
                tag = "ok" if cs >= C7_COS_SIM_BAR else "DIV"
                cells.append(
                    f"cos={_fmt_sci(cs):<10} max|Δ|={_fmt_sci(mxd):<10} "
                    f"mean|Δ|={_fmt_sci(mnd):<10} rel_l2={_fmt_sci(rl2):<10} {tag:<3}"
                )
        emit(f"  {tap:<26}{' '.join(cells)}")

    emit("")
    if not captured_any:
        emit("Phase C7 verdict: no C7 SDPA taps present in dumps.")
        emit("  -> Re-run kiln-bench with KILN_MTP_DUMP_POS=1,2 AND")
        emit("     KILN_MTP_DUMP_C7_SDPA=1 set, and re-run scripts/mtp_reference_dump.py")
        emit("     against the same kiln dump (the reference always emits c7__ taps).")
        return

    def _median(xs: List[float]) -> float:
        finite = sorted(v for v in xs if np.isfinite(v))
        if not finite:
            return float("nan")
        n = len(finite)
        mid = n // 2
        if n % 2 == 1:
            return finite[mid]
        return 0.5 * (finite[mid - 1] + finite[mid])

    first_median_div: Optional[str] = None
    first_median_val: float = float("nan")
    first_any_div: Optional[str] = None
    first_any_pos: Optional[str] = None
    for tap in C7_SDPA_TAP_NAMES:
        per_pos = [cell[tap][lab][0] for lab in labels if lab in cell[tap]]
        if not per_pos:
            continue
        med = _median(per_pos)
        if first_median_div is None and np.isfinite(med) and med < C7_COS_SIM_BAR:
            first_median_div = tap
            first_median_val = med
        for lab in labels:
            entry = cell[tap].get(lab)
            if entry is None:
                continue
            cs = entry[0]
            if np.isfinite(cs) and cs < C7_COS_SIM_BAR and first_any_div is None:
                first_any_div = tap
                first_any_pos = lab

    if first_median_div is not None:
        hypo = C7_HYPOTHESIS.get(first_median_div, "<unknown tap>")
        emit(
            f"  first tap with MEDIAN cos_sim < {C7_COS_SIM_BAR}: '{first_median_div}' "
            f"(median cos_sim = {first_median_val:.6f})"
        )
        emit(f"Phase C7 verdict: SDPA TARGET = '{first_median_div}'.")
        emit(f"    Most-likely cause: {hypo}")
        emit("  -> Queue the follow-up phase to investigate this tap in isolation.")
    elif first_any_div is not None:
        hypo = C7_HYPOTHESIS.get(first_any_div, "<unknown tap>")
        emit(
            f"  no tap has MEDIAN cos_sim < {C7_COS_SIM_BAR}, but '{first_any_div}' "
            f"(pos={first_any_pos}) dips below {C7_COS_SIM_BAR} on at least one position."
        )
        emit(f"Phase C7 verdict: SDPA TARGET (noisy) = '{first_any_div}'.")
        emit(f"    Most-likely cause: {hypo}")
        emit("  -> Recommend capturing more positions / seeds before committing")
        emit("     to a C8 fix direction.")
    else:
        emit(f"Phase C7 verdict: ALL SDPA taps match within cos_sim >= {C7_COS_SIM_BAR}.")
        emit("  -> SDPA is NOT the dominant source of post_attn_raw drift at this bar.")
        emit("  -> Look downstream of `attn_out` (o_proj, residual, post-attn RMSNorm,")
        emit("     MLP) and/or upstream of SDPA (Q/K/V projections, RoPE, dual-norm).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kiln", help="Path to kiln dump (.safetensors) — single-pair mode")
    ap.add_argument("--ref", help="Path to reference dump (.safetensors) — single-pair mode")
    ap.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Multi-position mode: LABEL:KILN_PATH,REF_PATH. Repeat for each position.",
    )
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument(
        "--b10",
        action="store_true",
        help=(
            "Phase B10 mode: emit per-layer base-model hidden-state summary "
            "(taps h_layer_{0,8,16,23,31} + h_pre_final_norm). Defaults atol=1e-2, "
            "rtol=1e-1 (bf16-appropriate) unless explicitly overridden."
        ),
    )
    ap.add_argument(
        "--b11",
        action="store_true",
        help=(
            "Phase B11b mode: emit layer-0 GDN sub-op bisect summary (11 taps "
            "prefixed with `b11__`). Defaults atol=1e-2, rtol=1e-1 "
            "(bf16-appropriate) unless explicitly overridden. Recommends the "
            "first sub-op whose median cos_sim across positions falls below "
            "0.95 as the B12 target."
        ),
    )
    ap.add_argument(
        "--b12",
        action="store_true",
        help=(
            "Phase B12 mode: emit layer-31 drift bisect. Prints a per-layer "
            "cos_sim table for h_layer_24..31 + h_pre_final_norm and, if "
            "b12__<name> sub-op taps are present, a per-sub-op table over the "
            "14 layer-31 GQA taps. Defaults atol=1e-2, rtol=1e-1 "
            "(bf16-appropriate). Verdict identifies the first layer and (if "
            "that layer is 31) the first sub-op whose median cos_sim falls "
            "below 0.995. Run once against a bf16 HF ref and once against a "
            "fp32 HF ref to separate real bugs from benign bf16 accumulation."
        ),
    )
    ap.add_argument(
        "--c6",
        action="store_true",
        help=(
            "Phase C6 mode: emit pre-RoPE MTP input bisect (5 taps prefixed "
            "with `c6__`: token_emb, norm_emb, norm_h, concat, fused). "
            "Defaults atol=1e-3, rtol=1e-2 (fp32-appropriate; these taps are "
            "computed before any bf16 transformer block). Verdict names the "
            "first tap (top-down) whose median cos_sim across positions dips "
            "below 0.9999 as the dominant source of pre-RoPE drift."
        ),
    )
    ap.add_argument(
        "--c7",
        action="store_true",
        help=(
            "Phase C7 mode: emit SDPA-internal bisect inside the MTP inner "
            "transformer block (7 taps prefixed with `c7__`: pre_sdpa_q, "
            "pre_sdpa_k, pre_sdpa_v, causal_mask, attn_scores_pre_softmax, "
            "attn_probs, attn_out). Defaults atol=1e-3, rtol=1e-2 "
            "(fp32-appropriate; bf16 KV-cache noise is confined to a couple "
            "of taps, the rest are fp32-clean). Verdict names the first tap "
            "(top-down) whose median cos_sim across positions dips below "
            "0.9999 as the dominant source of SDPA drift. Run after --c6 "
            "rules pre-RoPE out as the cause."
        ),
    )
    ap.add_argument("--out", default=None, help="Optional path to write report text")
    args = ap.parse_args()

    # Default tolerances depend on mode. B10/B11/B12 compare bf16 base-model
    # hidden states end-to-end through 32 transformer blocks; the accumulated
    # arithmetic noise across that many ops means a strict 1e-3/1e-2 bar
    # flags semantically-identical tensors as divergent. C6 taps are all
    # pre-RoPE (embed / dual RMSNorms / concat / single fc matmul), so keep
    # the strict fp32 bar for C6 even though B-phase modes relax it. C7 taps
    # sit inside SDPA — Q/K/V projections, scores, softmax, and attn_out are
    # fp32-clean on the HF side with only the bf16 KV cache read introducing
    # noise; the strict bar stays on to catch real structural drops and the
    # comparator report makes bf16-accumulation drift visible via per-position
    # max|Δ| / rel_l2 columns.
    bf16_mode = args.b10 or args.b11 or args.b12
    if args.atol is None:
        args.atol = 1e-2 if bf16_mode else 1e-3
    if args.rtol is None:
        args.rtol = 1e-1 if bf16_mode else 1e-2

    pairs: List[Tuple[str, str, str]] = []
    if args.pair:
        if args.kiln or args.ref:
            print("ERROR: cannot mix --pair with --kiln/--ref", file=sys.stderr)
            return 2
        for s in args.pair:
            pairs.append(_parse_pair_arg(s))
    else:
        if not (args.kiln and args.ref):
            print("ERROR: must specify --kiln and --ref, or one or more --pair args", file=sys.stderr)
            return 2
        pairs.append(("", args.kiln, args.ref))

    lines: List[str] = []

    def emit(s: str) -> None:
        print(s)
        lines.append(s)

    multi = len(pairs) > 1
    if args.c7:
        mode = "SDPA-internal bisect (C7)"
    elif args.c6:
        mode = "pre-RoPE MTP input bisect (C6)"
    elif args.b12:
        mode = "layer-31 drift bisect (B12)"
    elif args.b11:
        mode = "layer-0 GDN sub-op bisect (B11b)"
    elif args.b10:
        mode = "base-model h_main bisect (B10)"
    elif multi:
        mode = "multi-position (B7a)"
    else:
        mode = "single-pair (B6/B7)"
    emit(f"MTP numerical bisect — mode: {mode}, atol={args.atol}, rtol={args.rtol}")

    pair_results: List[
        Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]
    ] = []
    overall_ok = True
    for label, kpath, rpath in pairs:
        ok, first_div, rows, km, rm = _compare_pair(
            label, kpath, rpath, args.atol, args.rtol, emit
        )
        pair_results.append((label, ok, first_div, rows, km, rm))
        if not ok:
            overall_ok = False

    if args.c7:
        _emit_c7_summary(pair_results, args.atol, args.rtol, emit)
    elif args.c6:
        _emit_c6_summary(pair_results, args.atol, args.rtol, emit)
    elif args.b12:
        _emit_b12_summary(pair_results, args.atol, args.rtol, emit)
    elif args.b11:
        _emit_b11_summary(pair_results, args.atol, args.rtol, emit)
    elif args.b10:
        _emit_b10_summary(pair_results, args.atol, args.rtol, emit)
    elif multi:
        _emit_b7a_summary(pair_results, emit)

    # B9 sub-op zone bisect — emitted whenever any pair captured zone taps,
    # whether single- or multi-pair. Exits cleanly with a "no zone taps
    # captured" note when the dump didn't include the B9 sub-ops (e.g.
    # legacy dumps from before this PR). Skipped in explicit --b10/--b11/--b12/--c6
    # mode so the per-phase report isn't cluttered by unrelated sub-op tables.
    if not args.b10 and not args.b11 and not args.b12 and not args.c6 and not args.c7:
        have_b9_taps = any(
            any(r["name"] in B9_H2_ZONE or r["name"] in B9_H3_ZONE for r in pr[3])
            for pr in pair_results
        )
        if have_b9_taps:
            _emit_b9_summary(pair_results, args.atol, args.rtol, emit)
        elif multi:
            emit("")
            emit("Phase B9 zone bisect: skipped (no H2/H3 zone sub-op taps in dumps).")

    emit("")
    if overall_ok:
        emit("Overall verdict: all taps match within tolerance.")
    else:
        first_divs = [pr[2] for pr in pair_results if pr[2]]
        emit(f"Overall verdict: divergence(s) at: {first_divs}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote report to {args.out}", file=sys.stderr)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
