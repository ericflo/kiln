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

Phase B10 — independent h_main audit:
    python3 scripts/mtp_compare.py \\
        --pair 0:/tmp/dump_pos0.st,/tmp/ref_pos0.st \\
        --pair 1:/tmp/dump_pos1.st,/tmp/ref_pos1.st \\
        --pair 2:/tmp/dump_pos2.st,/tmp/ref_pos2.st \\
        --independent-ref 0:/tmp/ref_independent_pos0.safetensors \\
        [--atol 1e-3] [--rtol 1e-2]

    The `--independent-ref` flag accepts one or more `LABEL:PATH` pairs. For
    each matching LABEL, the comparator loads `h_main_independent` from the
    referenced safetensors and compares against the kiln dump's `h_main`
    tap at that same position label. Emits a B10 verdict line:

      - "MATCHES independent reference (kiln main-model stack OK)" when
        cos_sim >= 0.999 AND max|Δ| is within BF16 noise (<= 0.05 absolute).
        In that case the α collapse is NOT driven by the 32-layer forward.

      - "DIVERGES from independent reference (kiln main-model stack is
        upstream cause)" otherwise. Next step is to bisect the 24×GDN +
        8×GQA stack (Phase B11).

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
        return row

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    diff = np.abs(a64 - b64)
    row["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
    row["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
    row["allclose"] = bool(np.allclose(a64, b64, atol=atol, rtol=rtol))
    row["cos_sim"] = _cos_sim(a64, b64)
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


# Phase B10 thresholds: a clean "MATCHES" verdict requires cos_sim near 1 and
# max|Δ| within BF16 noise. Kiln runs main-model forward in BF16 on GPU while
# the HF independent reference can run BF16 on CUDA or F32 on CPU — either way
# we expect BF16-scale noise on the order of 1e-2 absolute across a [1, 1, H]
# hidden vector. 0.05 is a generous absolute ceiling for "noise-level match";
# divergences from a true 32-layer-stack bug will blow past this easily.
B10_COS_SIM_THRESHOLD = 0.999
B10_MAX_ABS_DIFF_THRESHOLD = 0.05


def _parse_independent_ref_arg(s: str) -> Tuple[str, str]:
    """Parse `LABEL:PATH` used for --independent-ref."""
    if ":" not in s:
        print(
            f"ERROR: --independent-ref argument '{s}' missing 'LABEL:' prefix",
            file=sys.stderr,
        )
        sys.exit(2)
    label, path = s.split(":", 1)
    return label.strip(), path.strip()


def _emit_b10_summary(
    pair_results: List[
        Tuple[str, bool, str, List[Dict[str, object]], Dict[str, int], Dict[str, int]]
    ],
    independent_refs: Dict[str, str],
    atol: float,
    rtol: float,
    emit,
) -> None:
    """Phase B10 — audit the kiln main-model stack by comparing kiln's `h_main`
    tap against an independent HF-derived `h_main_independent` reference.

    For each matching position label:
      1. Re-load the kiln dump to get `h_main`.
      2. Load the independent ref safetensors to get `h_main_independent`.
      3. Compare with _compare_tap (same atol/rtol as elsewhere); also apply
         B10-specific cos_sim + max|Δ| thresholds for the verdict.

    Verdict:
      - MATCHES: every requested label passes (cos_sim >= B10_COS_SIM_THRESHOLD
        AND max|Δ| <= B10_MAX_ABS_DIFF_THRESHOLD). -> The 32-layer main-model
        forward stack is NOT driving the α collapse; look inside MTP head.
      - DIVERGES: at least one requested label fails. -> The 32-layer stack is
        the upstream cause; next step is a per-layer bisect (Phase B11).
    """
    emit("")
    emit("=" * 78)
    emit("Phase B10 independent h_main audit — kiln main-model stack verdict")
    emit("=" * 78)

    # Build label -> kiln dump path map from pair_results.
    # pair_results stores (label, ok, first_div, rows, kiln_meta, ref_meta).
    # We need the kiln dump path — re-load via the row data isn't possible
    # since we only stored metrics. Instead, cross-reference by re-reading
    # the kiln `h_main` tensor directly from the kiln dump path stored in
    # the global label->kiln_path table below. We pass that through from
    # main() by looking it up in the `pairs` list there; the pair_results
    # tuple already carries kiln_meta which lets us cross-check seed/pos,
    # but we need the raw tensor so we re-open the file here.
    #
    # To avoid plumbing dump paths through pair_results (which would be a
    # wider refactor), we require the caller to also pass the label->kiln
    # path map via `_b10_label_to_kiln_path` set on the function at call
    # time. Keeping this localized below.
    label_to_kiln_path: Dict[str, str] = getattr(
        _emit_b10_summary, "_label_to_kiln_path", {}
    )

    rows: List[Dict[str, object]] = []
    all_labels: List[str] = []
    missing_labels: List[str] = []
    for label, ref_path in independent_refs.items():
        all_labels.append(label)
        kiln_path = label_to_kiln_path.get(label)
        if kiln_path is None:
            emit(f"  pos={label}: ERROR — no matching kiln pair for this label")
            missing_labels.append(label)
            continue

        kiln_arrays, _ = _load(kiln_path)
        ref_arrays, ref_meta = _load(ref_path)

        if "h_main" not in kiln_arrays:
            emit(f"  pos={label}: ERROR — kiln dump at {kiln_path} has no `h_main` tap")
            missing_labels.append(label)
            continue
        if "h_main_independent" not in ref_arrays:
            emit(
                f"  pos={label}: ERROR — independent ref at {ref_path} has no "
                "`h_main_independent` tensor"
            )
            missing_labels.append(label)
            continue
        bench_mtp_pos = ref_meta.get("meta__bench_mtp_pos", None)
        if bench_mtp_pos is not None and str(bench_mtp_pos) != str(label):
            emit(
                f"  pos={label}: WARN — independent ref meta__bench_mtp_pos="
                f"{bench_mtp_pos} does not match pair label {label}"
            )

        row = _compare_tap(
            f"h_main@pos={label}",
            kiln_arrays["h_main"],
            ref_arrays["h_main_independent"],
            atol,
            rtol,
        )
        rows.append(row)
        shape_str = "x".join(str(s) for s in row["shape_kiln"])
        emit(
            f"  pos={label}: shape={shape_str} shape_ok={row['shape_match']} "
            f"cos_sim={_fmt_sci(row['cos_sim'])} max|Δ|={_fmt_sci(row['max_abs_diff'])} "
            f"mean|Δ|={_fmt_sci(row['mean_abs_diff'])} allclose={row['allclose']}"
        )

    if missing_labels:
        emit("")
        emit(
            f"Phase B10 verdict: INCONCLUSIVE — could not evaluate labels "
            f"{missing_labels}."
        )
        return

    if not rows:
        emit("")
        emit("Phase B10 verdict: INCONCLUSIVE — no --independent-ref pairs resolved.")
        return

    # B10 threshold decision.
    all_pass = True
    worst: Dict[str, object] = rows[0]
    for r in rows:
        cs = float(r["cos_sim"])
        mxd = float(r["max_abs_diff"])
        if not r["shape_match"] or not np.isfinite(cs) or cs < B10_COS_SIM_THRESHOLD:
            all_pass = False
        if mxd > B10_MAX_ABS_DIFF_THRESHOLD:
            all_pass = False
        # Track the position with the lowest cos_sim (worst match) for reporting.
        if (
            not np.isfinite(float(worst["cos_sim"]))
            or cs < float(worst["cos_sim"])
        ):
            worst = r

    emit("")
    emit(
        f"  Thresholds: cos_sim >= {B10_COS_SIM_THRESHOLD}, "
        f"max|Δ| <= {B10_MAX_ABS_DIFF_THRESHOLD}"
    )
    if all_pass:
        emit(
            "Phase B10 verdict: h_main MATCHES independent reference "
            "(kiln main-model stack OK; collapse is in MTP-head-internal path)"
        )
        emit(
            f"  Worst pos across {len(rows)} label(s): {worst['name']} "
            f"cos_sim={_fmt_sci(worst['cos_sim'])} max|Δ|={_fmt_sci(worst['max_abs_diff'])}"
        )
        emit(
            "  -> Next step: continue investigating MTP-head-internal path "
            "(dual-norm order, fc matmul, inner-block sub-ops)."
        )
    else:
        emit(
            "Phase B10 verdict: h_main DIVERGES from independent reference "
            "(kiln main-model stack is upstream cause; bisect 32 layers next)"
        )
        emit(
            f"  Worst pos across {len(rows)} label(s): {worst['name']} "
            f"cos_sim={_fmt_sci(worst['cos_sim'])} max|Δ|={_fmt_sci(worst['max_abs_diff'])}"
        )
        emit(
            "  -> Next step: Phase B11 — bisect the 24×GDN + 8×GQA main-model "
            "stack to localize which layer's forward drifts from HF."
        )


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
    ap.add_argument(
        "--independent-ref",
        action="append",
        default=[],
        help="Phase B10: LABEL:INDEPENDENT_REF_PATH. Compare kiln's `h_main` tap "
        "at this pair label against `h_main_independent` in the referenced "
        "safetensors (produced by `mtp_reference_dump.py --independent-h-main`). "
        "Repeat per position.",
    )
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--out", default=None, help="Optional path to write report text")
    args = ap.parse_args()

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

    independent_refs: Dict[str, str] = {}
    for s in args.independent_ref:
        lab, path = _parse_independent_ref_arg(s)
        independent_refs[lab] = path

    lines: List[str] = []

    def emit(s: str) -> None:
        print(s)
        lines.append(s)

    multi = len(pairs) > 1
    mode = "multi-position (B7a)" if multi else "single-pair (B6/B7)"
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

    if multi:
        _emit_b7a_summary(pair_results, emit)

    # B9 sub-op zone bisect — emitted whenever any pair captured zone taps,
    # whether single- or multi-pair. Exits cleanly with a "no zone taps
    # captured" note when the dump didn't include the B9 sub-ops (e.g.
    # legacy dumps from before this PR).
    have_b9_taps = any(
        any(r["name"] in B9_H2_ZONE or r["name"] in B9_H3_ZONE for r in pr[3])
        for pr in pair_results
    )
    if have_b9_taps:
        _emit_b9_summary(pair_results, args.atol, args.rtol, emit)
    elif multi:
        emit("")
        emit("Phase B9 zone bisect: skipped (no H2/H3 zone sub-op taps in dumps).")

    # Phase B10: independent h_main audit. Only runs when --independent-ref is
    # provided. Passes the label->kiln_path map through an attribute on the
    # summary function so we don't need to refactor pair_results.
    if independent_refs:
        label_to_kiln_path = {label: kpath for (label, kpath, _rpath) in pairs}
        _emit_b10_summary._label_to_kiln_path = label_to_kiln_path  # type: ignore[attr-defined]
        _emit_b10_summary(pair_results, independent_refs, args.atol, args.rtol, emit)

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
