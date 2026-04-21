#!/usr/bin/env python3
"""
Phase B6 — Per-tap numerical comparison of MTP intermediates.

Consumes two safetensors dumps (kiln native + PyTorch reference), prints a
divergence table, and identifies the first tap where the two paths disagree
beyond (atol=1e-3, rtol=1e-2).

Usage:
    python3 scripts/mtp_compare.py \\
        --kiln /tmp/mtp-kiln.safetensors \\
        --ref  /tmp/mtp-ref.safetensors \\
        [--atol 1e-3] [--rtol 1e-2] \\
        [--out /tmp/mtp-compare.txt]

Exit code:
    0 — all taps match within tolerance
    1 — at least one tap diverges (normal outcome of a bisect)
    2 — structural error (missing file, missing tap, shape mismatch)
"""

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    from safetensors.numpy import load_file
except ImportError:
    print("ERROR: safetensors not installed. pip install safetensors", file=sys.stderr)
    sys.exit(2)


# Canonical tap order — must match the order written by write_mtp_dump in
# mtp_debug.rs and by mtp_reference_dump.py. First divergence is reported by
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

META_KEYS = ("meta__draft_token_id", "meta__mtp_pos", "meta__swap_fc_norms")


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


def _print_meta(title: str, meta: Dict[str, int]) -> None:
    print(f"  {title}:")
    for k in META_KEYS:
        print(f"    {k}: {meta.get(k, '<missing>')}")


def _load(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    try:
        tensors = load_file(path)
    except Exception as e:
        print(f"ERROR: failed to load {path}: {e}", file=sys.stderr)
        sys.exit(2)
    meta: Dict[str, int] = {}
    arrays: Dict[str, np.ndarray] = {}
    for key, val in tensors.items():
        if key in META_KEYS:
            meta[key] = int(val.item()) if val.size == 1 else int(val.flatten()[0])
        else:
            arrays[key] = val
    return arrays, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kiln", required=True, help="Path to kiln dump (.safetensors)")
    ap.add_argument("--ref", required=True, help="Path to reference dump (.safetensors)")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--out", default=None, help="Optional path to write report text")
    args = ap.parse_args()

    kiln_arr, kiln_meta = _load(args.kiln)
    ref_arr, ref_meta = _load(args.ref)

    lines: List[str] = []

    def emit(s: str) -> None:
        print(s)
        lines.append(s)

    emit(f"MTP Phase B6 numerical bisect — atol={args.atol}, rtol={args.rtol}")
    emit(f"  kiln dump : {args.kiln}")
    emit(f"  ref  dump : {args.ref}")
    emit("")
    emit("Metadata:")
    for k in META_KEYS:
        kv = kiln_meta.get(k, "<missing>")
        rv = ref_meta.get(k, "<missing>")
        match = "OK" if kv == rv else "MISMATCH"
        emit(f"  {k}: kiln={kv} ref={rv} [{match}]")
    emit("")

    # Header
    header = (
        f"{'tap':<18} {'shape':<18} {'shape_ok':<8} "
        f"{'allclose':<8} {'cos_sim':<10} {'max|Δ|':<12} {'mean|Δ|':<12}"
    )
    emit(header)
    emit("-" * len(header))

    first_div: str = ""
    all_ok = True
    rows: List[Dict[str, object]] = []

    for name in TAP_ORDER:
        if name not in kiln_arr:
            emit(f"{name:<18} MISSING IN KILN DUMP")
            all_ok = False
            if not first_div:
                first_div = name
            continue
        if name not in ref_arr:
            emit(f"{name:<18} MISSING IN REF DUMP")
            all_ok = False
            if not first_div:
                first_div = name
            continue
        row = _compare_tap(name, kiln_arr[name], ref_arr[name], args.atol, args.rtol)
        rows.append(row)
        shape_str = "x".join(str(s) for s in row["shape_kiln"])
        emit(
            f"{row['name']:<18} {shape_str:<18} {str(row['shape_match']):<8} "
            f"{str(row['allclose']):<8} {_fmt_sci(row['cos_sim']):<10} "
            f"{_fmt_sci(row['max_abs_diff']):<12} {_fmt_sci(row['mean_abs_diff']):<12}"
        )
        if not row["allclose"] or not row["shape_match"]:
            all_ok = False
            if not first_div:
                first_div = row["name"]

    emit("")
    if all_ok:
        emit("VERDICT: all taps match within tolerance — no divergence found.")
    else:
        emit(f"VERDICT: first divergence at tap '{first_div}'.")
        # Map first-divergence tap back to the hypothesis it most directly implicates.
        hypothesis_map = {
            "h_main": "upstream (base model) — this ref takes h_main FROM kiln, so shouldn't differ",
            "tok_embed": "tied embed lookup or token-id path",
            "fc_input": "RMSNorm inputs, concat ordering, or the dual-norm A/B swap",
            "fc_output": "mtp.fc matmul (weight transpose or layout)",
            "pre_layer": "(same tensor as fc_output at this site)",
            "post_layer": "single-layer MTP block: RoPE mtp_pos advancement, Q/K/V proj, or gated-attn",
            "post_final_ln": "mtp.final_layernorm application site / weight",
            "mtp_logits": "tied embed_tokens_t transpose vs alias for LM head",
        }
        emit(f"  Most-likely cause: {hypothesis_map.get(first_div, '<unknown tap>')}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote report to {args.out}", file=sys.stderr)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
