#!/usr/bin/env python3
"""Phase C16 — MTP accept/reject plumbing verifier.

Reads one or more Phase C1 attribution CSVs (produced by
`KILN_C1_ATTR_PATH=...` on `kiln-bench --spec mtp --temperature 0` runs) and
audits the four plumbing hypotheses the C15 verdict handed off to C16:

    H1  mtp_logits indexing — is greedy `accepted` the same as `topk_match`?
    H2  mtp_advance accounting — does `mtp_pos` advance on ACCEPT only?
    H3  draft-token KV wiring — does `base_pos` advance by 2 on ACCEPT / 1 on
        REJECT? (This is how `base_advance` is threaded into `base_pos` in
        `bench_latency_paged_mtp`; a wrong delta would imply either the
        accept-branch double-forward write or the reject-branch slot
        accounting drifted.)
    H4  KV-rollback semantics — N/A for split-verify greedy k=1. On REJECT
        the draft was never forwarded through the base model, so there is
        nothing to roll back in the base cache; the MTP cache write from
        step 1 is overwritten on the next iteration because `mtp_pos` stays
        pinned. The script reports H4 as N/A with supporting evidence.

Under greedy decoding the following invariants must hold by construction,
any violation is a plumbing bug worth chasing:

    Row-level:
        accepted == topk_match                          (H1)

    Between consecutive rows within one bench run:
        base_pos[i+1] - base_pos[i] == 2 if accepted[i] else 1    (H3)
        mtp_pos[i+1]  - mtp_pos[i]  == 1 if accepted[i] else 0    (H2)

A "bench run" boundary is detected by `step_idx` resetting to 0 — the sink
clears and resets `NEXT_STEP_IDX` at the top of each run
(`kiln_model::c1_attr::clear`), so multiple runs concatenated into one CSV
(or concatenated across CSVs) can be split cleanly.

Usage:

    python3 scripts/c16_plumbing_analyze.py kiln_c1_seed0.csv [seed1.csv ...]
    python3 scripts/c16_plumbing_analyze.py --json out.json *.csv

Exit status is 0 when every invariant holds across every input (or the
input was all-REJECT so the transition invariants had the trivial path),
and 1 when any invariant was violated.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable


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


def split_runs(rows: list[Row]) -> list[list[Row]]:
    """Split a flat row list at `step_idx` resets.

    `kiln_model::c1_attr::clear()` resets NEXT_STEP_IDX to 0 at the top of
    every bench run, so a monotonic `step_idx` stays within one run and a
    reset-to-zero marks a new run. This lets us chain multi-seed CSVs
    together without crossing the run boundary when checking transition
    invariants.
    """
    if not rows:
        return []
    runs: list[list[Row]] = []
    cur: list[Row] = [rows[0]]
    for prev, nxt in zip(rows, rows[1:]):
        if nxt.step_idx <= prev.step_idx:
            runs.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    runs.append(cur)
    return runs


@dataclass
class RunVerdict:
    file: str
    run_idx: int
    n_rows: int
    n_accepted: int
    alpha: float
    # H1 — greedy invariant violations.
    h1_violations: list[dict] = field(default_factory=list)
    # H2 — mtp_advance invariant violations.
    h2_violations: list[dict] = field(default_factory=list)
    # H3 — base_advance invariant violations.
    h3_violations: list[dict] = field(default_factory=list)
    # Diagnostic: how many ACCEPT transitions did the run actually contain?
    # If 0 the H2 ACCEPT branch is untestable by this data.
    n_accept_transitions: int = 0
    n_reject_transitions: int = 0


def audit_run(file: str, run_idx: int, rows: list[Row]) -> RunVerdict:
    n = len(rows)
    n_accepted = sum(1 for r in rows if r.accepted)
    v = RunVerdict(
        file=file,
        run_idx=run_idx,
        n_rows=n,
        n_accepted=n_accepted,
        alpha=n_accepted / n if n else 0.0,
    )

    # H1 — per-row greedy invariant.
    for r in rows:
        if r.accepted != r.topk_match:
            v.h1_violations.append(
                {
                    "step_idx": r.step_idx,
                    "base_pos": r.base_pos,
                    "mtp_pos": r.mtp_pos,
                    "mtp_top1": r.mtp_top1,
                    "main_top1": r.main_top1,
                    "accepted": r.accepted,
                    "topk_match": r.topk_match,
                }
            )

    # H2 / H3 — between-row transition invariants.
    for prev, nxt in zip(rows, rows[1:]):
        expected_base = 2 if prev.accepted else 1
        expected_mtp = 1 if prev.accepted else 0
        actual_base = nxt.base_pos - prev.base_pos
        actual_mtp = nxt.mtp_pos - prev.mtp_pos
        if prev.accepted:
            v.n_accept_transitions += 1
        else:
            v.n_reject_transitions += 1
        if actual_base != expected_base:
            v.h3_violations.append(
                {
                    "prev_step_idx": prev.step_idx,
                    "next_step_idx": nxt.step_idx,
                    "prev_base_pos": prev.base_pos,
                    "next_base_pos": nxt.base_pos,
                    "expected_delta": expected_base,
                    "actual_delta": actual_base,
                    "prev_accepted": prev.accepted,
                }
            )
        if actual_mtp != expected_mtp:
            v.h2_violations.append(
                {
                    "prev_step_idx": prev.step_idx,
                    "next_step_idx": nxt.step_idx,
                    "prev_mtp_pos": prev.mtp_pos,
                    "next_mtp_pos": nxt.mtp_pos,
                    "expected_delta": expected_mtp,
                    "actual_delta": actual_mtp,
                    "prev_accepted": prev.accepted,
                }
            )
    return v


def render_verdict_table(verdicts: list[RunVerdict]) -> str:
    header = (
        f"{'file':<40} {'run':>3} {'rows':>5} {'acc':>5} {'α':>7}  "
        f"{'H1':>4} {'H2':>4} {'H3':>4}  {'accT':>4} {'rejT':>4}"
    )
    lines = [header, "-" * len(header)]
    for v in verdicts:
        lines.append(
            f"{v.file[-40:]:<40} {v.run_idx:>3} {v.n_rows:>5} "
            f"{v.n_accepted:>5} {v.alpha*100:>6.2f}%  "
            f"{len(v.h1_violations):>4} {len(v.h2_violations):>4} "
            f"{len(v.h3_violations):>4}  "
            f"{v.n_accept_transitions:>4} {v.n_reject_transitions:>4}"
        )
    return "\n".join(lines)


def render_summary(verdicts: list[RunVerdict]) -> tuple[str, bool]:
    total_rows = sum(v.n_rows for v in verdicts)
    total_acc = sum(v.n_accepted for v in verdicts)
    total_h1 = sum(len(v.h1_violations) for v in verdicts)
    total_h2 = sum(len(v.h2_violations) for v in verdicts)
    total_h3 = sum(len(v.h3_violations) for v in verdicts)
    total_accT = sum(v.n_accept_transitions for v in verdicts)
    total_rejT = sum(v.n_reject_transitions for v in verdicts)

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("Phase C16 — MTP plumbing audit (derived from Phase C1 CSV)")
    lines.append("=" * 78)
    lines.append(
        f"Aggregate: rows={total_rows}  accepted={total_acc} "
        f"(α={total_acc/total_rows*100:.2f}%)  "
        f"accept_transitions={total_accT}  reject_transitions={total_rejT}"
    )
    lines.append("")

    def verdict_line(name: str, desc: str, n_viol: int, gated_on: int | None = None) -> str:
        if gated_on is not None and gated_on == 0:
            return f"  {name}: UNTESTABLE (no {desc} transitions in this data)"
        if n_viol == 0:
            return f"  {name}: PASS — {desc}"
        return f"  {name}: FAIL — {n_viol} violations ({desc})"

    lines.append(verdict_line("H1", "greedy invariant accepted == topk_match", total_h1))
    lines.append(
        verdict_line(
            "H2",
            "mtp_pos advances +1 on ACCEPT, +0 on REJECT",
            total_h2,
            gated_on=total_accT if total_accT == 0 else None,
        )
    )
    lines.append(
        verdict_line(
            "H3",
            "base_pos advances +2 on ACCEPT, +1 on REJECT",
            total_h3,
            gated_on=total_accT if total_accT == 0 else None,
        )
    )
    lines.append(
        "  H4: N/A — split-verify k=1 greedy has no rollback path (REJECT "
        "never runs the draft through the base model; MTP cache slot is "
        "reused because mtp_pos stays pinned)."
    )

    if total_accT == 0:
        lines.append("")
        lines.append(
            "⚠  No ACCEPT transitions in this data (α likely collapsed). "
            "H2/H3 ACCEPT-side branches are not exercised by these CSVs; "
            "only the REJECT-side transitions were audited. Collect a CSV "
            "from a build where α > 0 to cover ACCEPT-side plumbing."
        )

    all_pass = total_h1 == 0 and total_h2 == 0 and total_h3 == 0
    lines.append("")
    lines.append(
        "OVERALL: PASS (all tested invariants hold)"
        if all_pass
        else "OVERALL: FAIL (see violations above)"
    )
    return "\n".join(lines), all_pass


def violations_to_records(v: RunVerdict) -> list[dict]:
    out: list[dict] = []
    for h, items in (
        ("H1", v.h1_violations),
        ("H2", v.h2_violations),
        ("H3", v.h3_violations),
    ):
        for item in items:
            rec = {"hypothesis": h, "file": v.file, "run_idx": v.run_idx}
            rec.update(item)
            out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csvs", nargs="+", help="CSV files from KILN_C1_ATTR_PATH runs")
    ap.add_argument(
        "--json",
        metavar="PATH",
        help="Write full per-run verdicts + violations as JSON to PATH",
    )
    ap.add_argument(
        "--show-violations",
        type=int,
        default=5,
        help="How many violation rows to print per hypothesis (default 5)",
    )
    args = ap.parse_args(argv)

    verdicts: list[RunVerdict] = []
    for path in args.csvs:
        rows = read_csv(path)
        runs = split_runs(rows)
        for i, run_rows in enumerate(runs):
            verdicts.append(audit_run(os.path.basename(path), i, run_rows))

    print(render_verdict_table(verdicts))
    summary, all_pass = render_summary(verdicts)
    print(summary)

    # Print a handful of violation examples per hypothesis so the operator
    # can see concrete failing rows without having to load the JSON.
    def print_examples(name: str, acc: Iterable[dict]) -> None:
        items = list(acc)
        if not items:
            return
        print()
        print(f"{name} violation examples (first {args.show_violations}):")
        for ex in items[: args.show_violations]:
            print(f"  {ex}")

    print_examples(
        "H1", (e for v in verdicts for e in ({"run": v.run_idx, "file": v.file, **x} for x in v.h1_violations))
    )
    print_examples(
        "H2", (e for v in verdicts for e in ({"run": v.run_idx, "file": v.file, **x} for x in v.h2_violations))
    )
    print_examples(
        "H3", (e for v in verdicts for e in ({"run": v.run_idx, "file": v.file, **x} for x in v.h3_violations))
    )

    if args.json:
        payload = {
            "runs": [
                {
                    "file": v.file,
                    "run_idx": v.run_idx,
                    "n_rows": v.n_rows,
                    "n_accepted": v.n_accepted,
                    "alpha": v.alpha,
                    "n_accept_transitions": v.n_accept_transitions,
                    "n_reject_transitions": v.n_reject_transitions,
                    "h1_violations": v.h1_violations,
                    "h2_violations": v.h2_violations,
                    "h3_violations": v.h3_violations,
                }
                for v in verdicts
            ],
            "overall_pass": all_pass,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON verdict → {args.json}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
