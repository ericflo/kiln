#!/usr/bin/env python3
"""Summarize Phase C1 MTP acceptance-rate attribution CSVs.

Reads one or more CSV files produced by `KILN_C1_ATTR_PATH=...` on
`kiln-bench --spec mtp --temperature 0` runs, and prints:

  - overall acceptance rate α = accepted / total
  - overall top-k match rate (mtp_top1 == main_top1)
  - per-pos-in-K acceptance + topk-match rates
  - per-mtp-pos bucketed acceptance (decode-wide position trend)
  - row counts for Class A (topk_match=True, accepted=False) — verification
    bug under greedy
  - row counts for Class B (topk_match=False) — MTP head disagrees with
    main under greedy
  - verdict recommendation for the C2 pivot, per the Phase C1 task brief

Under greedy decoding the invariant `accepted == (mtp_top1 == main_top1)`
must hold; any divergence is itself the bug signal.

CSV format (per row, produced by `kiln_model::c1_attr::row_to_csv_line`):

    step_idx,pos_in_k,base_pos,mtp_pos,last_token,
    mtp_top1,mtp_top1_logit,main_top1,main_top1_logit,
    accepted,topk_match

Bools are encoded as 0/1.

Usage:

    python3 scripts/mtp_c1_summarize.py kiln_c1_seed0.csv [seed1.csv ...]
    python3 scripts/mtp_c1_summarize.py --json out.json *.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from statistics import median


@dataclass
class Row:
    step_idx: int
    pos_in_k: int
    base_pos: int
    mtp_pos: int
    last_token: int
    mtp_top1: int
    mtp_top1_logit: float
    main_top1: int
    main_top1_logit: float
    accepted: bool
    topk_match: bool


def read_csv(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "step_idx",
            "pos_in_k",
            "base_pos",
            "mtp_pos",
            "last_token",
            "mtp_top1",
            "mtp_top1_logit",
            "main_top1",
            "main_top1_logit",
            "accepted",
            "topk_match",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{path}: CSV missing columns {sorted(missing)} "
                f"(got {reader.fieldnames})"
            )
        for rec in reader:
            rows.append(
                Row(
                    step_idx=int(rec["step_idx"]),
                    pos_in_k=int(rec["pos_in_k"]),
                    base_pos=int(rec["base_pos"]),
                    mtp_pos=int(rec["mtp_pos"]),
                    last_token=int(rec["last_token"]),
                    mtp_top1=int(rec["mtp_top1"]),
                    mtp_top1_logit=float(rec["mtp_top1_logit"]),
                    main_top1=int(rec["main_top1"]),
                    main_top1_logit=float(rec["main_top1_logit"]),
                    accepted=rec["accepted"] == "1",
                    topk_match=rec["topk_match"] == "1",
                )
            )
    return rows


def rate(num: int, denom: int) -> float:
    return num / denom if denom > 0 else 0.0


def fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def bucket_by_mtp_pos(rows: list[Row], n_buckets: int = 8) -> list[tuple[int, int, int, int]]:
    """Returns list of (bucket_lo, bucket_hi, n_rows, n_accepted) tuples."""
    if not rows:
        return []
    max_pos = max(r.mtp_pos for r in rows)
    width = max(1, (max_pos + 1 + n_buckets - 1) // n_buckets)
    buckets: list[list[Row]] = [[] for _ in range(n_buckets)]
    for r in rows:
        b = min(r.mtp_pos // width, n_buckets - 1)
        buckets[b].append(r)
    out = []
    for i, bucket in enumerate(buckets):
        lo = i * width
        hi = (i + 1) * width - 1
        n = len(bucket)
        a = sum(1 for r in bucket if r.accepted)
        out.append((lo, hi, n, a))
    return out


def identity_stats(rows: list[Row]) -> dict[str, float | int]:
    total = len(rows)
    identity = sum(1 for r in rows if r.mtp_top1 == r.last_token)
    accepted_rows = [r for r in rows if r.accepted]
    rejected_rows = [r for r in rows if not r.accepted]
    accepted_identity = sum(1 for r in accepted_rows if r.mtp_top1 == r.last_token)
    rejected_identity = sum(1 for r in rejected_rows if r.mtp_top1 == r.last_token)
    return {
        "draft_equals_last_token_count": identity,
        "draft_equals_last_token_rate": rate(identity, total),
        "accept_conditioned_count": accepted_identity,
        "accept_conditioned_rate": rate(accepted_identity, len(accepted_rows)),
        "reject_conditioned_count": rejected_identity,
        "reject_conditioned_rate": rate(rejected_identity, len(rejected_rows)),
    }


def summarize_file(path: str, rows: list[Row]) -> dict:
    n = len(rows)
    accepted = sum(1 for r in rows if r.accepted)
    topk_match = sum(1 for r in rows if r.topk_match)
    class_a = sum(1 for r in rows if r.topk_match and not r.accepted)
    class_b = sum(1 for r in rows if not r.topk_match)
    # Class B rows where we still accepted (impossible under greedy — reported
    # if non-zero as a sanity check on the compare logic).
    class_b_accepted = sum(1 for r in rows if not r.topk_match and r.accepted)

    # Per pos_in_k breakdown
    by_pos_in_k: dict[int, list[Row]] = {}
    for r in rows:
        by_pos_in_k.setdefault(r.pos_in_k, []).append(r)

    mtp_logits = [r.mtp_top1_logit for r in rows]
    main_logits = [r.main_top1_logit for r in rows]
    id_stats = identity_stats(rows)

    return {
        "file": os.path.basename(path),
        "total_rows": n,
        "accepted": accepted,
        "topk_match": topk_match,
        "alpha": rate(accepted, n),
        "topk_match_rate": rate(topk_match, n),
        "class_a_rows": class_a,
        "class_b_rows": class_b,
        "class_b_accepted_rows": class_b_accepted,
        "per_pos_in_k": {
            str(k): {
                "n": len(v),
                "alpha": rate(sum(1 for r in v if r.accepted), len(v)),
                "topk_match_rate": rate(sum(1 for r in v if r.topk_match), len(v)),
            }
            for k, v in sorted(by_pos_in_k.items())
        },
        "mtp_pos_buckets": [
            {
                "lo": lo,
                "hi": hi,
                "n": len(bucket),
                "accepted": sum(1 for r in bucket if r.accepted),
                "alpha": rate(sum(1 for r in bucket if r.accepted), len(bucket)),
                "draft_equals_last_token_count": sum(
                    1 for r in bucket if r.mtp_top1 == r.last_token
                ),
                "draft_equals_last_token_rate": rate(
                    sum(1 for r in bucket if r.mtp_top1 == r.last_token), len(bucket)
                ),
            }
            for (lo, hi, _n_b, _a_b) in bucket_by_mtp_pos(rows)
            for bucket in [[r for r in rows if lo <= r.mtp_pos <= hi]]
        ],
        "identity_bias": id_stats,
        "logit_stats": {
            "mtp_top1_logit_median": median(mtp_logits) if mtp_logits else float("nan"),
            "main_top1_logit_median": median(main_logits) if main_logits else float("nan"),
            "mtp_top1_logit_min": min(mtp_logits) if mtp_logits else float("nan"),
            "mtp_top1_logit_max": max(mtp_logits) if mtp_logits else float("nan"),
        },
    }


def aggregate_summaries(summaries: list[dict]) -> dict:
    total = sum(s["total_rows"] for s in summaries)
    accepted = sum(s["accepted"] for s in summaries)
    topk_match = sum(s["topk_match"] for s in summaries)
    class_a = sum(s["class_a_rows"] for s in summaries)
    class_b = sum(s["class_b_rows"] for s in summaries)
    identity_count = sum(
        s["identity_bias"]["draft_equals_last_token_count"] for s in summaries
    )
    accept_identity_count = sum(
        s["identity_bias"]["accept_conditioned_count"] for s in summaries
    )
    reject_identity_count = sum(
        s["identity_bias"]["reject_conditioned_count"] for s in summaries
    )
    rejected = total - accepted
    return {
        "total_rows": total,
        "accepted": accepted,
        "topk_match": topk_match,
        "alpha": rate(accepted, total),
        "topk_match_rate": rate(topk_match, total),
        "alpha_given_match": rate(accepted, topk_match) if topk_match else 0.0,
        "class_a_rows": class_a,
        "class_b_rows": class_b,
        "identity_bias": {
            "draft_equals_last_token_count": identity_count,
            "draft_equals_last_token_rate": rate(identity_count, total),
            "accept_conditioned_count": accept_identity_count,
            "accept_conditioned_rate": rate(accept_identity_count, accepted),
            "reject_conditioned_count": reject_identity_count,
            "reject_conditioned_rate": rate(reject_identity_count, rejected),
        },
    }


def verdict(summaries: list[dict]) -> tuple[str, str]:
    """Return (verdict_code, recommendation_text) per the task brief.

    Thresholds (from Phase C1 task brief):
      - topk_match_rate >= 0.95 AND alpha / topk_match_rate >= 0.95
            → MTP path is correct; C2 should pivot to non-MTP decode gains.
      - topk_match_rate >= 0.95 AND alpha / topk_match_rate < 0.95
            → verification / sampling bug (Class A). C2 bisects accept check.
      - topk_match_rate < 0.95 (i.e. Class B rate > 5%)
            → MTP head bug. C2 bisects the MTP forward vs an independent ref.
    """
    # Aggregate across files (sum counts, not mean rates — seeds have unequal
    # token counts).
    aggregate = aggregate_summaries(summaries)
    total = aggregate["total_rows"]
    accepted = aggregate["accepted"]
    topk_match = aggregate["topk_match"]
    alpha = aggregate["alpha"]
    topk_rate = aggregate["topk_match_rate"]
    # Alpha conditioned on topk_match — the "verification consistency" ratio.
    alpha_given_match = rate(accepted, topk_match) if topk_match else 0.0

    if topk_rate >= 0.95 and alpha_given_match >= 0.95:
        return (
            "C2_PIVOT_NON_MTP",
            (
                "VERDICT: MTP path is CORRECT. Aggregate topk_match_rate = "
                f"{fmt_pct(topk_rate)} (>=95% target) and alpha|match = "
                f"{fmt_pct(alpha_given_match)} (>=95% target). The α ceiling "
                f"{fmt_pct(alpha)} reflects pretrained MTP head quality on "
                "this workload, not a kiln bug.\n"
                "RECOMMENDATION: Pivot C2 to non-MTP decode optimizations — "
                "FlashInfer paged GQA decode, FP8 KV, prefix cache. Re-profile "
                "current main first to pick the highest-leverage target."
            ),
        )
    if topk_rate >= 0.95 and alpha_given_match < 0.95:
        return (
            "C2_BISECT_VERIFICATION",
            (
                "VERDICT: VERIFICATION / SAMPLING BUG (Class A). Aggregate "
                f"topk_match_rate = {fmt_pct(topk_rate)} (>=95% — tokens "
                f"agree) but alpha|match = {fmt_pct(alpha_given_match)} "
                "(<95% — accept check flips false even when they agree). "
                "Under greedy this invariant cannot fail without a bug in "
                "the accept path.\n"
                "RECOMMENDATION: C2 bisects the draft → verify → accept "
                "pipeline in crates/kiln-model/src/speculative.rs. Focus on: "
                "(a) greedy_sample determinism across the two tensors, (b) "
                "the target_at_0 == draft_token compare, (c) any tensor "
                "dtype / layout drift between mtp_logits and verify_pos0 "
                "before top-1 extraction."
            ),
        )
    return (
        "C2_BISECT_MTP_HEAD",
        (
            "VERDICT: MTP HEAD BUG (Class B). Aggregate topk_match_rate = "
            f"{fmt_pct(topk_rate)} (<95% — MTP top-1 disagrees with main "
            "top-1 on more than 5% of greedy positions). Either the MTP "
            "head forward is wrong, or the pretrained head is genuinely "
            "this noisy — the former must be ruled out first.\n"
            "RECOMMENDATION: C2 bisects the MTP forward pass. Dump the "
            "mtp_logits top-5 per step alongside the HF remote-code MTP "
            "head output on the same prompt (see scripts/ for existing B-"
            "phase reference-dump tooling) and localize the first divergent "
            "tap."
        ),
    )


def print_summary(summaries: list[dict], v_code: str, v_text: str) -> None:
    print("=" * 78)
    print("Phase C1 — MTP Acceptance-Rate Attribution Summary")
    print("=" * 78)
    for s in summaries:
        print()
        print(f"File: {s['file']}   rows={s['total_rows']}")
        print(
            f"  α = {fmt_pct(s['alpha'])}   "
            f"topk_match_rate = {fmt_pct(s['topk_match_rate'])}   "
            f"Class A (match, !accept) = {s['class_a_rows']}   "
            f"Class B (!match) = {s['class_b_rows']}"
        )
        print(
            "  identity bias: draft==last_token "
            f"{s['identity_bias']['draft_equals_last_token_count']}/{s['total_rows']} "
            f"({fmt_pct(s['identity_bias']['draft_equals_last_token_rate'])})   "
            f"reject-conditioned = {fmt_pct(s['identity_bias']['reject_conditioned_rate'])}   "
            f"accept-conditioned = {fmt_pct(s['identity_bias']['accept_conditioned_rate'])}"
        )
        if s["class_b_accepted_rows"] > 0:
            print(
                f"  ⚠ {s['class_b_accepted_rows']} rows accepted despite "
                f"topk_match=False (impossible under greedy)"
            )
        if len(s["per_pos_in_k"]) > 1:
            print("  Per pos_in_k:")
            for k, v in s["per_pos_in_k"].items():
                print(
                    f"    pos_in_k={k}:  n={v['n']:5d}  "
                    f"α={fmt_pct(v['alpha'])}  "
                    f"match={fmt_pct(v['topk_match_rate'])}"
                )
        print("  mtp_pos buckets (decode-wide trend):")
        for b in s["mtp_pos_buckets"]:
            if b["n"] == 0:
                continue
            print(
                f"    mtp_pos∈[{b['lo']:4d}..{b['hi']:4d}]  "
                f"n={b['n']:4d}  α={fmt_pct(b['alpha'])}  "
                f"identity={b['draft_equals_last_token_count']:4d} "
                f"({fmt_pct(b['draft_equals_last_token_rate'])})"
            )
        print(
            "  logit medians: mtp={:.3f}  main={:.3f}".format(
                s["logit_stats"]["mtp_top1_logit_median"],
                s["logit_stats"]["main_top1_logit_median"],
            )
        )

    # Aggregate
    aggregate = aggregate_summaries(summaries)
    print()
    print("-" * 78)
    print("Aggregate across files:")
    print(f"  rows            = {aggregate['total_rows']}")
    print(f"  α               = {fmt_pct(aggregate['alpha'])}")
    print(f"  topk_match_rate = {fmt_pct(aggregate['topk_match_rate'])}")
    print(f"  α | match       = {fmt_pct(aggregate['alpha_given_match']) if aggregate['topk_match'] else '  n/a '}")
    print(f"  Class A rows    = {aggregate['class_a_rows']}  (topk_match=True, accepted=False)")
    print(f"  Class B rows    = {aggregate['class_b_rows']}  (topk_match=False)")
    print(
        "  identity bias   = "
        f"{aggregate['identity_bias']['draft_equals_last_token_count']}/"
        f"{aggregate['total_rows']} "
        f"({fmt_pct(aggregate['identity_bias']['draft_equals_last_token_rate'])})"
    )
    print(
        "  reject | identity = "
        f"{fmt_pct(aggregate['identity_bias']['reject_conditioned_rate'])}   "
        "accept | identity = "
        f"{fmt_pct(aggregate['identity_bias']['accept_conditioned_rate'])}"
    )
    print()
    print(f"VERDICT CODE: {v_code}")
    print(v_text)
    print("-" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="CSV files from KILN_C1_ATTR_PATH runs")
    ap.add_argument("--json", metavar="PATH", help="Write full summary + verdict as JSON to PATH")
    args = ap.parse_args()

    summaries = []
    for path in args.csvs:
        rows = read_csv(path)
        summaries.append(summarize_file(path, rows))

    v_code, v_text = verdict(summaries)
    print_summary(summaries, v_code, v_text)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "files": summaries,
                    "aggregate": aggregate_summaries(summaries),
                    "verdict_code": v_code,
                    "verdict_text": v_text,
                },
                f,
                indent=2,
            )
        print(f"\nJSON summary → {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
