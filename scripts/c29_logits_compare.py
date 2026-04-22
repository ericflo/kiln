#!/usr/bin/env python3
"""Phase C29 empirical MTP-logits comparator (H9).

Walks paired (kiln, ref) safetensors trees produced by c29_kiln_logits_dump.sh
and c29_hf_reference_dump.py and computes the H9-critical metrics that the
existing C14 cos_sim/max|Δ| comparison cannot resolve:

  - top-1 agreement rate (does the argmax token differ?)
  - top-K Jaccard overlap for K ∈ {1, 5, 10, 20}
  - KL(softmax(kiln) || softmax(ref)) at temperature 1.0
  - probability mass on kiln's top-1 under ref's distribution

Why these metrics matter:

  C14 reported median cos_sim 0.9999736 on `c14__logits` ([1, 1, 248320]).
  At a 248K-dim head with bf16 noise, cos_sim ≈ 0.99997 is consistent with
  bf16 forward noise — and yet the *speculative-decoding decision boundary*
  is exquisitely sensitive to top-K agreement. A wrong top-1 even 5% of the
  time crashes acceptance to ≈0.95×base_alpha, which compounds across draft
  positions. H9 asks: is the bench α ≈ 0.0328 explained by enough top-K
  rotation that a per-prompt rejection probe would catch what cos_sim hides?

Inputs
------

    <kiln_root>/seed-{S}/mtp_pos-{N}/step-{K}.safetensors   (kiln dumps)
    <ref_root>/seed-{S}/mtp_pos-{N}/step-{K}.safetensors    (HF ref dumps)

Outputs
-------

    <out_dir>/c29-logits-compare.json   (full per-(seed, pos, step) breakdown)
    <out_dir>/c29-logits-compare.md     (human-readable verdict tables)
    stderr                              (digest aligned with C14 format)

Comparison taps
---------------

The primary tap is `c14__logits`. The script also recomputes cos/max|Δ| on
the four other major taps (`h_main`, `c14__post_block`, `c14__post_norm`,
plus `pre_rope__fused` if present) so the C29 verdict can show side-by-side
that the cosine metric is clean while the top-K metric is (or isn't) noisy.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print(
        "[c29_compare] safetensors not installed — run `pip install safetensors numpy` first",
        file=sys.stderr,
    )
    raise


LOGITS_TAP = "c14__logits"
SECONDARY_TAPS = ["h_main", "c14__post_block", "c14__post_norm", "pre_rope__fused"]
TOP_K_VALUES = [1, 5, 10, 20]
JACCARD_FLOOR = 0.90  # threshold below which we flag a site
TOP1_AGREEMENT_FLOOR = 0.95


@dataclass
class LogitsStats:
    cos_sim: float
    max_abs_delta: float
    top1_match: bool
    top_k_jaccard: Dict[int, float]
    kl_kiln_to_ref: float
    kl_ref_to_kiln: float
    ref_prob_at_kiln_top1: float
    kiln_top1_id: int
    ref_top1_id: int


@dataclass
class SecondaryStats:
    tap: str
    cos_sim: float
    max_abs_delta: float


@dataclass
class FileRecord:
    seed: int
    mtp_pos: int
    step: int
    kiln_path: str
    ref_path: str
    logits: Optional[LogitsStats] = None
    secondary: List[SecondaryStats] = field(default_factory=list)
    error: Optional[str] = None


def parse_seed_pos_step(kiln_root: Path, path: Path) -> Optional[Tuple[int, int, int]]:
    rel = str(path.relative_to(kiln_root))
    m = re.search(r"seed-(\d+)[/\\]mtp_pos-(\d+)[/\\]step-(\d+)\.safetensors$", rel)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_tensor(path: Path, name: str) -> Optional[np.ndarray]:
    with safe_open(str(path), framework="np") as f:
        if name not in f.keys():
            return None
        return np.asarray(f.get_tensor(name), dtype=np.float32)


def cos_sim_flat(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(af, bf) / (na * nb))


def softmax_log(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    x = logits.astype(np.float64)
    m = np.max(x)
    y = x - m
    log_z = math.log(float(np.sum(np.exp(y))))
    return y - log_z


def kl_divergence(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """KL(P || Q) = sum P (log P - log Q)."""
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


def compare_logits(kiln: np.ndarray, ref: np.ndarray) -> LogitsStats:
    assert kiln.shape == ref.shape, f"logits shape mismatch: {kiln.shape} vs {ref.shape}"
    flat_k = kiln.reshape(-1).astype(np.float32)
    flat_r = ref.reshape(-1).astype(np.float32)

    delta = flat_k - flat_r
    max_abs = float(np.abs(delta).max())
    cos = cos_sim_flat(flat_k, flat_r)

    kiln_top1 = int(np.argmax(flat_k))
    ref_top1 = int(np.argmax(flat_r))

    jaccards: Dict[int, float] = {}
    for k in TOP_K_VALUES:
        # Use partial sort for speed on the 248K-dim vocab; argpartition is O(n).
        kiln_topk = set(np.argpartition(-flat_k, k)[:k].tolist())
        ref_topk = set(np.argpartition(-flat_r, k)[:k].tolist())
        union = kiln_topk | ref_topk
        inter = kiln_topk & ref_topk
        jaccards[k] = float(len(inter)) / float(len(union)) if union else 1.0

    log_pk = softmax_log(flat_k)
    log_pr = softmax_log(flat_r)
    kl_kr = kl_divergence(log_pk, log_pr)
    kl_rk = kl_divergence(log_pr, log_pk)
    ref_prob_at_kiln_top1 = float(np.exp(log_pr[kiln_top1]))

    return LogitsStats(
        cos_sim=cos,
        max_abs_delta=max_abs,
        top1_match=(kiln_top1 == ref_top1),
        top_k_jaccard=jaccards,
        kl_kiln_to_ref=kl_kr,
        kl_ref_to_kiln=kl_rk,
        ref_prob_at_kiln_top1=ref_prob_at_kiln_top1,
        kiln_top1_id=kiln_top1,
        ref_top1_id=ref_top1,
    )


def compare_secondary(name: str, kiln: np.ndarray, ref: np.ndarray) -> SecondaryStats:
    assert kiln.shape == ref.shape, f"{name} shape mismatch: {kiln.shape} vs {ref.shape}"
    delta = kiln.reshape(-1).astype(np.float32) - ref.reshape(-1).astype(np.float32)
    return SecondaryStats(
        tap=name,
        cos_sim=cos_sim_flat(kiln, ref),
        max_abs_delta=float(np.abs(delta).max()),
    )


def compare_file(
    kiln_path: Path, ref_path: Path, seed: int, mtp_pos: int, step: int
) -> FileRecord:
    rec = FileRecord(
        seed=seed,
        mtp_pos=mtp_pos,
        step=step,
        kiln_path=str(kiln_path),
        ref_path=str(ref_path),
    )
    try:
        k_logits = load_tensor(kiln_path, LOGITS_TAP)
        r_logits = load_tensor(ref_path, LOGITS_TAP)
        if k_logits is None or r_logits is None:
            rec.error = f"missing {LOGITS_TAP} (kiln={k_logits is not None}, ref={r_logits is not None})"
            return rec
        rec.logits = compare_logits(k_logits, r_logits)
        for tap in SECONDARY_TAPS:
            kt = load_tensor(kiln_path, tap)
            rt = load_tensor(ref_path, tap)
            if kt is None or rt is None:
                continue
            rec.secondary.append(compare_secondary(tap, kt, rt))
    except Exception as exc:  # noqa: BLE001
        rec.error = str(exc)
    return rec


def write_markdown(
    summary: Dict[str, object], out_path: Path, title: str = "Phase C29 — Empirical MTP Logits Compare (H9)"
) -> None:
    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append("**Primary tap**: `c14__logits` — `[1, 1, 248320]` (full base-vocab head over the MTP-block output).\n")
    lines.append(f"**Floors**: top-1 agreement ≥ {TOP1_AGREEMENT_FLOOR}, top-K Jaccard ≥ {JACCARD_FLOOR}.\n")
    lines.append(f"**Total dump pairs**: {summary['total_files']} ({summary['files_with_errors']} errors).\n")

    lines.append("\n## Aggregate (across all dumps)\n")
    agg = summary["aggregate"]
    if agg:
        lines.append("| metric | median | min | max | n |")
        lines.append("|--------|-------:|----:|----:|--:|")
        for key in [
            "cos_sim",
            "max_abs_delta",
            "top1_match_rate",
            "jaccard_k1",
            "jaccard_k5",
            "jaccard_k10",
            "jaccard_k20",
            "kl_kiln_to_ref",
            "kl_ref_to_kiln",
            "ref_prob_at_kiln_top1",
        ]:
            entry = agg.get(key)
            if not entry:
                continue
            lines.append(
                f"| `{key}` | {entry['median']:.6g} | {entry['min']:.6g} | {entry['max']:.6g} | {entry['n']} |"
            )

    lines.append("\n## Per-position medians\n")
    by_pos = summary.get("per_pos", {})
    if by_pos:
        lines.append("| mtp_pos | n | median cos | median top-1 match | median J@5 | median J@10 | median KL(k‖r) |")
        lines.append("|--------:|--:|-----------:|-------------------:|-----------:|------------:|----------------:|")
        for pos in sorted(by_pos.keys(), key=int):
            row = by_pos[pos]
            lines.append(
                f"| {pos} | {row['n']} | {row['median_cos_sim']:.6f} | "
                f"{row['median_top1_match']:.4f} | {row['median_jaccard_k5']:.4f} | "
                f"{row['median_jaccard_k10']:.4f} | {row['median_kl_kiln_to_ref']:.4e} |"
            )

    lines.append("\n## Per-seed medians (prompt-level breakdown)\n")
    by_seed = summary.get("per_seed", {})
    if by_seed:
        lines.append("| seed | n | median cos | median top-1 match | median J@5 | median J@10 |")
        lines.append("|-----:|--:|-----------:|-------------------:|-----------:|------------:|")
        for seed in sorted(by_seed.keys(), key=int):
            row = by_seed[seed]
            lines.append(
                f"| {seed} | {row['n']} | {row['median_cos_sim']:.6f} | "
                f"{row['median_top1_match']:.4f} | {row['median_jaccard_k5']:.4f} | "
                f"{row['median_jaccard_k10']:.4f} |"
            )

    flagged = summary.get("flagged_sites", [])
    lines.append(f"\n## Flagged sites: {len(flagged)}\n")
    if flagged:
        lines.append("Sites with top-1 mismatch OR jaccard@10 < floor:\n")
        lines.append("| seed | pos | step | top-1 | J@1 | J@5 | J@10 | cos | KL(k‖r) |")
        lines.append("|-----:|----:|----:|------:|----:|----:|-----:|----:|--------:|")
        for f in flagged:
            lines.append(
                f"| {f['seed']} | {f['mtp_pos']} | {f['step']} | "
                f"{'✓' if f['top1_match'] else '✗'} | "
                f"{f['jaccard_k1']:.3f} | {f['jaccard_k5']:.3f} | {f['jaccard_k10']:.3f} | "
                f"{f['cos_sim']:.6f} | {f['kl_kiln_to_ref']:.3e} |"
            )

    lines.append("\n## Secondary taps (sanity)\n")
    sec_agg = summary.get("secondary_aggregate", {})
    if sec_agg:
        lines.append("| tap | n | median cos | min cos | median max\\|Δ\\| |")
        lines.append("|-----|--:|-----------:|--------:|----------------:|")
        for tap, row in sec_agg.items():
            lines.append(
                f"| `{tap}` | {row['n']} | {row['median_cos_sim']:.6f} | "
                f"{row['min_cos_sim']:.6f} | {row['median_max_abs_delta']:.3e} |"
            )

    out_path.write_text("\n".join(lines) + "\n")


def summarize(records: List[FileRecord]) -> Dict[str, object]:
    valid = [r for r in records if r.logits is not None]

    def collect(field_extractor) -> Dict[str, float]:
        vals = [field_extractor(r) for r in valid]
        if not vals:
            return {}
        return {
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals),
        }

    aggregate = {
        "cos_sim": collect(lambda r: r.logits.cos_sim),
        "max_abs_delta": collect(lambda r: r.logits.max_abs_delta),
        "top1_match_rate": collect(lambda r: 1.0 if r.logits.top1_match else 0.0),
        "kl_kiln_to_ref": collect(lambda r: r.logits.kl_kiln_to_ref),
        "kl_ref_to_kiln": collect(lambda r: r.logits.kl_ref_to_kiln),
        "ref_prob_at_kiln_top1": collect(lambda r: r.logits.ref_prob_at_kiln_top1),
    }
    for k in TOP_K_VALUES:
        aggregate[f"jaccard_k{k}"] = collect(lambda r, k=k: r.logits.top_k_jaccard[k])

    def group_medians(group_key) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        groups: Dict[str, List[FileRecord]] = {}
        for r in valid:
            groups.setdefault(str(group_key(r)), []).append(r)
        for key, rs in groups.items():
            out[key] = {
                "n": len(rs),
                "median_cos_sim": float(np.median([r.logits.cos_sim for r in rs])),
                "median_top1_match": float(
                    np.median([1.0 if r.logits.top1_match else 0.0 for r in rs])
                ),
                "median_jaccard_k5": float(
                    np.median([r.logits.top_k_jaccard[5] for r in rs])
                ),
                "median_jaccard_k10": float(
                    np.median([r.logits.top_k_jaccard[10] for r in rs])
                ),
                "median_kl_kiln_to_ref": float(
                    np.median([r.logits.kl_kiln_to_ref for r in rs])
                ),
            }
        return out

    per_pos = group_medians(lambda r: r.mtp_pos)
    per_seed = group_medians(lambda r: r.seed)

    flagged: List[Dict[str, object]] = []
    for r in valid:
        if (not r.logits.top1_match) or r.logits.top_k_jaccard[10] < JACCARD_FLOOR:
            flagged.append(
                {
                    "seed": r.seed,
                    "mtp_pos": r.mtp_pos,
                    "step": r.step,
                    "top1_match": r.logits.top1_match,
                    "jaccard_k1": r.logits.top_k_jaccard[1],
                    "jaccard_k5": r.logits.top_k_jaccard[5],
                    "jaccard_k10": r.logits.top_k_jaccard[10],
                    "cos_sim": r.logits.cos_sim,
                    "kl_kiln_to_ref": r.logits.kl_kiln_to_ref,
                }
            )

    secondary_aggregate: Dict[str, Dict[str, float]] = {}
    for tap in SECONDARY_TAPS:
        rows = []
        for r in valid:
            for s in r.secondary:
                if s.tap == tap:
                    rows.append(s)
        if not rows:
            continue
        secondary_aggregate[tap] = {
            "n": len(rows),
            "median_cos_sim": float(np.median([s.cos_sim for s in rows])),
            "min_cos_sim": float(np.min([s.cos_sim for s in rows])),
            "median_max_abs_delta": float(np.median([s.max_abs_delta for s in rows])),
        }

    return {
        "primary_tap": LOGITS_TAP,
        "secondary_taps": SECONDARY_TAPS,
        "top_k_values": TOP_K_VALUES,
        "jaccard_floor": JACCARD_FLOOR,
        "top1_agreement_floor": TOP1_AGREEMENT_FLOOR,
        "total_files": len(records),
        "files_with_errors": sum(1 for r in records if r.error),
        "aggregate": aggregate,
        "per_pos": per_pos,
        "per_seed": per_seed,
        "flagged_sites": flagged,
        "secondary_aggregate": secondary_aggregate,
        "files": [
            {
                "seed": r.seed,
                "mtp_pos": r.mtp_pos,
                "step": r.step,
                "kiln_path": r.kiln_path,
                "ref_path": r.ref_path,
                "error": r.error,
                "logits": asdict(r.logits) if r.logits else None,
                "secondary": [asdict(s) for s in r.secondary],
            }
            for r in records
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase C29 empirical MTP logits comparator (H9 — top-K Jaccard probe)"
    )
    ap.add_argument(
        "--kiln-root",
        required=True,
        type=Path,
        help="Root containing seed-{S}/mtp_pos-{N}/step-{K}.safetensors (kiln dumps)",
    )
    ap.add_argument(
        "--ref-root",
        required=True,
        type=Path,
        help="Root containing seed-{S}/mtp_pos-{N}/step-{K}.safetensors (HF reference dumps)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for c29-logits-compare.{json,md}",
    )
    args = ap.parse_args()

    kiln_files = sorted(args.kiln_root.glob("seed-*/mtp_pos-*/step-*.safetensors"))
    if not kiln_files:
        print(f"[c29_compare] no kiln dumps under {args.kiln_root}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: List[FileRecord] = []
    for kiln_path in kiln_files:
        parsed = parse_seed_pos_step(args.kiln_root, kiln_path)
        if parsed is None:
            print(f"[c29_compare] unparseable path: {kiln_path}", file=sys.stderr)
            continue
        seed, pos, step = parsed
        rel = kiln_path.relative_to(args.kiln_root)
        ref_path = args.ref_root / rel
        if not ref_path.exists():
            records.append(
                FileRecord(
                    seed=seed,
                    mtp_pos=pos,
                    step=step,
                    kiln_path=str(kiln_path),
                    ref_path=str(ref_path),
                    error="reference file missing",
                )
            )
            print(f"[c29_compare] ref missing for {rel}", file=sys.stderr)
            continue
        records.append(compare_file(kiln_path, ref_path, seed, pos, step))

    summary = summarize(records)

    json_path = args.out_dir / "c29-logits-compare.json"
    md_path = args.out_dir / "c29-logits-compare.md"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_markdown(summary, md_path)

    valid = [r for r in records if r.logits is not None]
    print(
        f"[c29_compare] processed {len(records)} dumps ({len(valid)} valid, "
        f"{sum(1 for r in records if r.error)} errors)",
        file=sys.stderr,
    )
    if valid:
        agg = summary["aggregate"]
        print(
            f"[c29_compare] median cos={agg['cos_sim']['median']:.6f}  "
            f"top-1 match rate={agg['top1_match_rate']['median']:.4f}  "
            f"median J@5={agg['jaccard_k5']['median']:.4f}  "
            f"median J@10={agg['jaccard_k10']['median']:.4f}  "
            f"median KL(k‖r)={agg['kl_kiln_to_ref']['median']:.3e}",
            file=sys.stderr,
        )
    print(
        f"[c29_compare] flagged {len(summary['flagged_sites'])} sites "
        f"(top-1 mismatch OR J@10 < {JACCARD_FLOOR})",
        file=sys.stderr,
    )
    print(f"[c29_compare] wrote {json_path} and {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
