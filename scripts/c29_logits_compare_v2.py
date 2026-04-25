#!/usr/bin/env python3
"""Phase C29 v2 empirical MTP-logits comparator — **stratified by accept vs reject**.

This is the H15b probe queued by PR #527 (phase7-mtp-acceptance-state-of-play)
and re-confirmed by PR #528 (H15a, Marlin pack determinism RULED OUT). H15b
asks the concrete next question:

    Does kiln's MTP head, on the **rejected** sub-population of draft rows,
    stay within the BF16-noise cosine ceiling (≥ 0.999), or does it diverge
    materially (< 0.99) from an fp32 HF reference?

The PR #355 C29 run (merged 2026-04-22) reported median cos_sim 0.99998 and
100% top-1 agreement across ALL 49 dumps. That result aggregated across
accept AND reject rows — if the reject sub-population hides cos_sim < 0.99
behind a 90%+ accept floor, the aggregate metric would miss it.

This v2 comparator re-uses the C29 primary tap (`c14__logits`, shape
`[1, 1, 248320]`) and the C29 metrics (cos_sim, max|Δ|, top-K Jaccard for
K∈{1,5,10,20}, KL(kiln‖ref), top-1 agreement), but emits **parallel stat
blocks** for the accept and reject sub-populations plus a machine-readable
`verdict.json` implementing the PR #527 decision rule:

    reject_cos_sim_median ≥ 0.999  → "kiln_native_ceiling"
                                     next: queue vLLM α microbench
    reject_cos_sim_median < 0.99   → "verifier_numerical_drift"
                                     next: queue per-layer bisect on reject rows
    in between                     → "ambiguous"
                                     next: expand seeds to 0..5 in follow-up

Input layout (same as c29_logits_compare.py, plus accept-labels CSV):

    <kiln_root>/seed-{S}/mtp_pos-{N}/step-{K}.safetensors   (kiln dumps)
    <ref_root>/seed-{S}/mtp_pos-{N}/step-{K}.safetensors    (HF ref dumps)
    --accept-labels-csv CSV1[,CSV2,...]                     (C1-attr CSVs)

Accept-label alignment
----------------------

Each C29 v2 splice dump at `(mtp_pos=N, step=K)` is produced by the K-th
call to `speculative_mtp_decode_step` with `mtp_pos=N` in that bench run.
The C1 attribution sink (`c1_attr::push_row`) fires from the same control
flow and therefore produces one CSV row per call to
`speculative_mtp_decode_step`. So the K-th CSV row with `mtp_pos=N`
corresponds to the K-th splice dump at `mtp_pos=N`. We key on this index.

This alignment is only valid when the accept-labels CSV was produced by
the **same bench run** that produced the C29 v2 dumps (i.e., C1 attribution
and splice dump armed simultaneously in one kiln-bench invocation). Cross-
run alignment (e.g., using a separate `docs/phase-c36/c1_seed*.csv` that
was produced with different `--chat-template` / `--max-output-tokens` /
sampling settings) would match wrong rows — the bench trajectory and
accept pattern differ across runs. The driver script
`scripts/c29_kiln_logits_dump.sh` is updated in PR #XXX (H15b) to emit the
CSV alongside the dumps for per-seed alignment.

Decision-rule fields in `verdict.json`
--------------------------------------

    {
      "verdict": "kiln_native_ceiling" | "verifier_numerical_drift" | "ambiguous",
      "next_action": "<one-sentence queued next step>",
      "reject_cos_sim_median": float,
      "reject_cos_sim_p10": float,
      "accept_cos_sim_median": float,
      "accept_cos_sim_p10": float,
      "n_reject": int,
      "n_accept": int,
      "n_unlabeled": int
    }
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print(
        "[c29_compare_v2] safetensors not installed — run `pip install safetensors numpy` first",
        file=sys.stderr,
    )
    raise


LOGITS_TAP = "c14__logits"
SECONDARY_TAPS = ["h_main", "c14__post_block", "c14__post_norm", "pre_rope__fused"]
TOP_K_VALUES = [1, 5, 10, 20]
JACCARD_FLOOR = 0.90
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
    accepted: Optional[int] = None  # 0=reject, 1=accept, None=unlabeled
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
    x = logits.astype(np.float64)
    m = np.max(x)
    y = x - m
    log_z = math.log(float(np.sum(np.exp(y))))
    return y - log_z


def kl_divergence(log_p: np.ndarray, log_q: np.ndarray) -> float:
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
    kiln_path: Path, ref_path: Path, seed: int, mtp_pos: int, step: int, accepted: Optional[int]
) -> FileRecord:
    rec = FileRecord(
        seed=seed,
        mtp_pos=mtp_pos,
        step=step,
        kiln_path=str(kiln_path),
        ref_path=str(ref_path),
        accepted=accepted,
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


# -----------------------------------------------------------------------------
# Accept-label loading and alignment
# -----------------------------------------------------------------------------


def load_accept_labels(csv_paths: List[Path]) -> Dict[Tuple[int, int, int], int]:
    """Build `(seed, mtp_pos, step_within_pos) → accepted` map.

    Expects filenames of the form `c1_seed{SEED}*.csv`. Within each file, the
    K-th row with `mtp_pos=N` is assumed to correspond to the K-th splice
    dump at `mtp_pos=N` in the same run (step index 0-based).
    """
    labels: Dict[Tuple[int, int, int], int] = {}
    for p in csv_paths:
        # Preferred: seed lives in the parent directory name as `seed-{N}`
        # (matches the layout written by c29_kiln_logits_dump.sh, one CSV per
        # seed dir at `<kiln_root>/seed-{N}/c1_attr.csv`).
        parent = p.parent.name
        m = re.match(r"seed-(\d+)$", parent)
        if not m:
            # Fallback: seed encoded in the filename, e.g. `c1_seed0.csv`
            m = re.search(r"seed[_-]?(\d+)", p.stem)
        if not m:
            print(f"[c29_compare_v2] cannot parse seed from {p}; skipping", file=sys.stderr)
            continue
        seed = int(m.group(1))
        # Count occurrences per mtp_pos to derive step_within_pos
        pos_counter: Dict[int, int] = defaultdict(int)
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mtp_pos = int(row["mtp_pos"])
                    accepted = int(row["accepted"])
                except (KeyError, ValueError):
                    continue
                step_within_pos = pos_counter[mtp_pos]
                pos_counter[mtp_pos] += 1
                labels[(seed, mtp_pos, step_within_pos)] = accepted
    return labels


# -----------------------------------------------------------------------------
# Stratified summary
# -----------------------------------------------------------------------------


def _stats_block(valid: List[FileRecord]) -> Dict[str, object]:
    """Compute the aggregate stat block for a list of valid records."""
    if not valid:
        return {"n": 0}

    def collect(field_extractor) -> Dict[str, float]:
        vals = [field_extractor(r) for r in valid]
        if not vals:
            return {}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "n": len(vals),
        }

    block: Dict[str, object] = {
        "n": len(valid),
        "cos_sim": collect(lambda r: r.logits.cos_sim),
        "max_abs_delta": collect(lambda r: r.logits.max_abs_delta),
        "top1_match_rate": collect(lambda r: 1.0 if r.logits.top1_match else 0.0),
        "kl_kiln_to_ref": collect(lambda r: r.logits.kl_kiln_to_ref),
        "kl_ref_to_kiln": collect(lambda r: r.logits.kl_ref_to_kiln),
        "ref_prob_at_kiln_top1": collect(lambda r: r.logits.ref_prob_at_kiln_top1),
    }
    for k in TOP_K_VALUES:
        block[f"jaccard_k{k}"] = collect(lambda r, k=k: r.logits.top_k_jaccard[k])
    return block


def summarize(records: List[FileRecord]) -> Dict[str, object]:
    valid = [r for r in records if r.logits is not None]
    accept_only = [r for r in valid if r.accepted == 1]
    reject_only = [r for r in valid if r.accepted == 0]
    unlabeled = [r for r in valid if r.accepted is None]

    per_pos_strat: Dict[str, Dict[str, Dict[str, object]]] = {}
    # Group by mtp_pos, then split accept/reject
    pos_groups: Dict[int, Dict[str, List[FileRecord]]] = defaultdict(lambda: {"accept": [], "reject": []})
    for r in valid:
        if r.accepted == 1:
            pos_groups[r.mtp_pos]["accept"].append(r)
        elif r.accepted == 0:
            pos_groups[r.mtp_pos]["reject"].append(r)
    for pos, buckets in pos_groups.items():
        per_pos_strat[str(pos)] = {
            "accept": _stats_block(buckets["accept"]),
            "reject": _stats_block(buckets["reject"]),
        }

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
        "n_accept": len(accept_only),
        "n_reject": len(reject_only),
        "n_unlabeled": len(unlabeled),
        "aggregate_all": _stats_block(valid),
        "accept_only": _stats_block(accept_only),
        "reject_only": _stats_block(reject_only),
        "per_pos_stratified": per_pos_strat,
        "secondary_aggregate": secondary_aggregate,
        "files": [
            {
                "seed": r.seed,
                "mtp_pos": r.mtp_pos,
                "step": r.step,
                "accepted": r.accepted,
                "kiln_path": r.kiln_path,
                "ref_path": r.ref_path,
                "error": r.error,
                "logits": asdict(r.logits) if r.logits else None,
                "secondary": [asdict(s) for s in r.secondary],
            }
            for r in records
        ],
    }


def compute_verdict(summary: Dict[str, object]) -> Dict[str, object]:
    reject = summary.get("reject_only", {}) or {}
    accept = summary.get("accept_only", {}) or {}
    reject_cos = reject.get("cos_sim", {}) or {}
    accept_cos = accept.get("cos_sim", {}) or {}

    reject_med = reject_cos.get("median")
    reject_p10 = reject_cos.get("p10")
    accept_med = accept_cos.get("median")
    accept_p10 = accept_cos.get("p10")
    n_reject = int(summary.get("n_reject", 0) or 0)
    n_accept = int(summary.get("n_accept", 0) or 0)
    n_unlabeled = int(summary.get("n_unlabeled", 0) or 0)

    if reject_med is None or n_reject == 0:
        verdict = "no_reject_rows"
        next_action = (
            "Cannot form verdict — 0 reject rows in the labeled sub-population. "
            "Confirm C1 attribution CSV alignment, then re-run with longer "
            "--max-output-tokens to capture more reject events."
        )
    elif reject_med >= 0.999:
        verdict = "kiln_native_ceiling"
        next_action = (
            "Queue a vLLM α microbench to establish the external-reference "
            "upper bound on Qwen3.5-4B A6000 bs=1 at this workload. "
            "Kiln's MTP head logits are at the BF16-noise cosine ceiling "
            "even on rejected drafts."
        )
    elif reject_med < 0.99:
        verdict = "verifier_numerical_drift"
        next_action = (
            "Queue a per-layer cos_sim bisect restricted to reject rows. "
            "Kiln's MTP head diverges materially from fp32 reference on "
            "the rejected sub-population — locate the layer where the "
            "divergence first exceeds BF16 noise."
        )
    else:
        verdict = "ambiguous"
        next_action = (
            "Expand seeds to 0..5 (or wider) in a follow-up H15b v2 run and "
            "re-stratify. Current reject_cos_sim_median sits in the "
            "[0.99, 0.999) band where neither the kiln-native-ceiling nor "
            "the verifier-numerical-drift verdict is supported."
        )

    return {
        "verdict": verdict,
        "next_action": next_action,
        "reject_cos_sim_median": reject_med,
        "reject_cos_sim_p10": reject_p10,
        "accept_cos_sim_median": accept_med,
        "accept_cos_sim_p10": accept_p10,
        "n_reject": n_reject,
        "n_accept": n_accept,
        "n_unlabeled": n_unlabeled,
        "decision_rule": {
            "kiln_native_ceiling": "reject_cos_sim_median >= 0.999",
            "verifier_numerical_drift": "reject_cos_sim_median < 0.99",
            "ambiguous": "0.99 <= reject_cos_sim_median < 0.999",
        },
    }


# -----------------------------------------------------------------------------
# Markdown rendering
# -----------------------------------------------------------------------------


def _fmt_block(block: Dict[str, object], label: str) -> List[str]:
    if not block or block.get("n", 0) == 0:
        return [f"\n### {label} — n=0 (no rows in this stratum)\n"]
    lines = [f"\n### {label} — n={block['n']}\n"]
    lines.append("| metric | median | mean | p10 | p90 | min | max |")
    lines.append("|--------|-------:|-----:|----:|----:|----:|----:|")
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
        e = block.get(key) or {}
        if not e:
            continue
        lines.append(
            f"| `{key}` | {e['median']:.6g} | {e['mean']:.6g} | {e['p10']:.6g} | "
            f"{e['p90']:.6g} | {e['min']:.6g} | {e['max']:.6g} |"
        )
    return lines


def write_markdown(summary: Dict[str, object], verdict: Dict[str, object], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Phase C29 v2 — Stratified MTP-Logits Reject-Row Probe (H15b)\n")
    lines.append("**Primary tap**: `c14__logits` — `[1, 1, 248320]` (full base-vocab head over the MTP-block output).")
    lines.append(f"**Total dump pairs**: {summary['total_files']} ({summary['files_with_errors']} errors).")
    lines.append(
        f"**Labeled coverage**: accept n={summary['n_accept']}, reject n={summary['n_reject']}, unlabeled n={summary['n_unlabeled']}."
    )

    lines.append("\n## Verdict\n")
    lines.append(f"**{verdict['verdict']}**")
    lines.append("")
    lines.append(f"- reject_cos_sim_median = {verdict['reject_cos_sim_median']}")
    lines.append(f"- reject_cos_sim_p10    = {verdict['reject_cos_sim_p10']}")
    lines.append(f"- accept_cos_sim_median = {verdict['accept_cos_sim_median']}")
    lines.append(f"- accept_cos_sim_p10    = {verdict['accept_cos_sim_p10']}")
    lines.append("")
    lines.append(f"**Next action**: {verdict['next_action']}")
    lines.append("")
    lines.append("Decision rule (from PR #527 §\"Recommended next H\"):")
    lines.append("")
    lines.append("- `reject_cos_sim_median >= 0.999` → **kiln_native_ceiling** → queue vLLM α microbench")
    lines.append("- `reject_cos_sim_median <  0.99 ` → **verifier_numerical_drift** → queue per-layer bisect on reject rows")
    lines.append("- `0.99 <= reject_cos_sim_median < 0.999` → **ambiguous** → expand seeds and re-run")

    lines.append("\n## Accept vs Reject stratified aggregate")
    lines.extend(_fmt_block(summary.get("accept_only", {}), "Accept-only"))
    lines.extend(_fmt_block(summary.get("reject_only", {}), "Reject-only"))
    lines.extend(_fmt_block(summary.get("aggregate_all", {}), "All labeled + unlabeled (reference)"))

    lines.append("\n## Per-position stratified cos_sim medians\n")
    per_pos = summary.get("per_pos_stratified", {})
    if per_pos:
        lines.append("| mtp_pos | accept n | accept median cos | accept p10 | reject n | reject median cos | reject p10 |")
        lines.append("|--------:|---------:|------------------:|-----------:|---------:|------------------:|-----------:|")
        for pos in sorted(per_pos.keys(), key=int):
            buckets = per_pos[pos]
            a = (buckets.get("accept") or {}).get("cos_sim") or {}
            r = (buckets.get("reject") or {}).get("cos_sim") or {}
            a_n = (buckets.get("accept") or {}).get("n", 0)
            r_n = (buckets.get("reject") or {}).get("n", 0)
            a_med = f"{a['median']:.6f}" if a else "-"
            a_p10 = f"{a['p10']:.6f}" if a else "-"
            r_med = f"{r['median']:.6f}" if r else "-"
            r_p10 = f"{r['p10']:.6f}" if r else "-"
            lines.append(f"| {pos} | {a_n} | {a_med} | {a_p10} | {r_n} | {r_med} | {r_p10} |")

    sec = summary.get("secondary_aggregate", {})
    if sec:
        lines.append("\n## Secondary taps (sanity, not stratified)\n")
        lines.append("| tap | n | median cos | min cos | median max\\|Δ\\| |")
        lines.append("|-----|--:|-----------:|--------:|----------------:|")
        for tap, row in sec.items():
            lines.append(
                f"| `{tap}` | {row['n']} | {row['median_cos_sim']:.6f} | "
                f"{row['min_cos_sim']:.6f} | {row['median_max_abs_delta']:.3e} |"
            )

    out_path.write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase C29 v2 stratified MTP logits comparator (H15b reject-row probe)"
    )
    ap.add_argument("--kiln-root", required=True, type=Path,
                    help="Root containing seed-{S}/mtp_pos-{N}/step-{K}.safetensors (kiln dumps)")
    ap.add_argument("--ref-root", required=True, type=Path,
                    help="Root containing seed-{S}/mtp_pos-{N}/step-{K}.safetensors (HF reference dumps)")
    ap.add_argument("--accept-labels-csv", required=True, type=str,
                    help="Comma-separated list of C1 attribution CSVs (filenames must encode seed as c1_seed{N}*.csv)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory for c29-v2-stratified-compare.{json,md} and verdict.json")
    args = ap.parse_args()

    csv_paths = [Path(p.strip()) for p in args.accept_labels_csv.split(",") if p.strip()]
    for cp in csv_paths:
        if not cp.exists():
            print(f"[c29_compare_v2] accept-labels CSV not found: {cp}", file=sys.stderr)
            return 2

    labels = load_accept_labels(csv_paths)
    print(f"[c29_compare_v2] loaded {len(labels)} (seed, mtp_pos, step_within_pos) labels", file=sys.stderr)

    kiln_files = sorted(args.kiln_root.glob("seed-*/mtp_pos-*/step-*.safetensors"))
    if not kiln_files:
        print(f"[c29_compare_v2] no kiln dumps under {args.kiln_root}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: List[FileRecord] = []
    for kiln_path in kiln_files:
        parsed = parse_seed_pos_step(args.kiln_root, kiln_path)
        if parsed is None:
            print(f"[c29_compare_v2] unparseable path: {kiln_path}", file=sys.stderr)
            continue
        seed, pos, step = parsed
        rel = kiln_path.relative_to(args.kiln_root)
        ref_path = args.ref_root / rel
        accepted = labels.get((seed, pos, step))
        if not ref_path.exists():
            records.append(
                FileRecord(
                    seed=seed, mtp_pos=pos, step=step,
                    kiln_path=str(kiln_path), ref_path=str(ref_path),
                    accepted=accepted, error="reference file missing",
                )
            )
            print(f"[c29_compare_v2] ref missing for {rel}", file=sys.stderr)
            continue
        records.append(compare_file(kiln_path, ref_path, seed, pos, step, accepted))

    summary = summarize(records)
    verdict = compute_verdict(summary)

    json_path = args.out_dir / "c29-v2-stratified-compare.json"
    md_path = args.out_dir / "c29-v2-stratified-compare.md"
    verdict_path = args.out_dir / "verdict.json"

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with verdict_path.open("w") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)
    write_markdown(summary, verdict, md_path)

    valid = [r for r in records if r.logits is not None]
    print(
        f"[c29_compare_v2] processed {len(records)} dumps "
        f"({len(valid)} valid, accept={summary['n_accept']}, reject={summary['n_reject']}, "
        f"unlabeled={summary['n_unlabeled']}, errors={sum(1 for r in records if r.error)})",
        file=sys.stderr,
    )
    print(f"[c29_compare_v2] verdict: {verdict['verdict']}", file=sys.stderr)
    print(f"[c29_compare_v2]   reject_cos_sim_median = {verdict['reject_cos_sim_median']}", file=sys.stderr)
    print(f"[c29_compare_v2]   accept_cos_sim_median = {verdict['accept_cos_sim_median']}", file=sys.stderr)
    print(f"[c29_compare_v2] wrote {json_path}, {md_path}, {verdict_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
