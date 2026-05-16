#!/usr/bin/env python3
"""§9.9 OPD bench-as-CI gate.

Compares the stdout of `cargo run --release --example bench_opd_topk_kl
--features cuda -p kiln-opd-loss-kernel` against the canonical
`opd-a6000-baseline.json` and exits non-zero if any row's
`kernel_tok_s` regresses by more than the baseline's
`regression_threshold_pct` (default 5%).

Usage:
    python3 bench-results/check_opd_regression.py \
        --bench-stdout <path-to-bench-output.txt> \
        [--baseline bench-results/opd-a6000-baseline.json] \
        [--threshold-pct 5.0]

The bench output is the literal stdout the example prints — one line
per shape, in the format:

    T=  256  H=2560  V=32000  K=32  F32  iters= 20  \
    kernel=  0.563ms  candle=  1.334ms   2.37x     455014 tok/s

Exit codes:
  0 — every row meets or beats the baseline within the threshold.
  1 — at least one row regresses past the threshold.
  2 — input file / baseline parse failure.
"""

import argparse
import json
import re
import sys
from pathlib import Path


FWD_ROW_RE = re.compile(
    # The bench prints two lines per shape: `FWD T=...` (forward only)
    # and `FWD+BWD T=...` (forward + backward). The committed baseline
    # was captured against forward-only throughput; we match the FWD
    # rows here and ignore FWD+BWD. The pre-FWD-prefix format (rows
    # starting with `T=`) is also accepted for backwards-compat with
    # the original 60db09ff capture.
    r"(?:^|\b)(?:FWD\s+)?T=\s*(?P<T>\d+)\s+"
    r"H=\s*(?P<H>\d+)\s+"
    r"V=\s*(?P<V>\d+)\s+"
    r"K=\s*(?P<K>\d+)\s+"
    r"(?P<dtype>F32|BF16|F16)\s+"
    r"iters=\s*(?P<iters>\d+)\s+"
    r"kernel=\s*(?P<kernel_ms>[\d.]+)ms\s+"
    r"candle=\s*(?P<candle_ms>[\d.]+)ms\s+"
    r"(?P<speedup>[\d.]+)x\s+"
    r"(?P<kernel_tok_s>\d+)\s+tok/s",
)


def parse_bench_stdout(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        # Skip FWD+BWD rows — only forward-only throughput is gated.
        if "FWD+BWD" in line:
            continue
        m = FWD_ROW_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "T": int(m["T"]),
                "H": int(m["H"]),
                "V": int(m["V"]),
                "K": int(m["K"]),
                "dtype": m["dtype"],
                "iters": int(m["iters"]),
                "kernel_ms": float(m["kernel_ms"]),
                "candle_ms": float(m["candle_ms"]),
                "speedup_x": float(m["speedup"]),
                "kernel_tok_s": int(m["kernel_tok_s"]),
            }
        )
    return rows


def row_key(row: dict) -> tuple:
    return (row["T"], row["H"], row["V"], row["K"], row["dtype"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench-stdout",
        required=True,
        type=Path,
        help="Path to the captured bench stdout (one row per shape).",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to the baseline JSON. When omitted, auto-picks "
             "opd-a100-baseline.json if 'A100' appears in the bench stdout, "
             "opd-a6000-baseline.json otherwise.",
    )
    p.add_argument(
        "--threshold-pct",
        type=float,
        default=None,
        help="Override the baseline's regression_threshold_pct.",
    )
    args = p.parse_args()

    if not args.bench_stdout.exists():
        print(f"ERROR: bench stdout not found: {args.bench_stdout}", file=sys.stderr)
        return 2

    bench_text = args.bench_stdout.read_text()
    if args.baseline is None:
        repo_root = Path(__file__).resolve().parent
        baseline = (
            repo_root / "opd-a100-baseline.json"
            if "A100" in bench_text
            else repo_root / "opd-a6000-baseline.json"
        )
    else:
        baseline = args.baseline
    if not baseline.exists():
        print(f"ERROR: baseline not found: {baseline}", file=sys.stderr)
        return 2
    args.baseline = baseline

    baseline = json.loads(args.baseline.read_text())
    threshold = args.threshold_pct or baseline.get("regression_threshold_pct", 5.0)

    new_rows = parse_bench_stdout(bench_text)
    if not new_rows:
        print(
            f"ERROR: parsed zero rows from {args.bench_stdout} — check format",
            file=sys.stderr,
        )
        return 2

    new_by_key = {row_key(r): r for r in new_rows}
    failures: list[str] = []
    deltas: list[tuple[tuple, float]] = []

    for base_row in baseline["rows"]:
        key = row_key(base_row)
        new_row = new_by_key.get(key)
        if not new_row:
            failures.append(
                f"missing row in new bench output: T={key[0]} K={key[3]} dtype={key[4]}"
            )
            continue
        base_tps = base_row["kernel_tok_s"]
        new_tps = new_row["kernel_tok_s"]
        delta_pct = (new_tps - base_tps) / base_tps * 100.0
        deltas.append((key, delta_pct))
        if delta_pct < -threshold:
            failures.append(
                f"REGRESSION T={key[0]} K={key[3]} dtype={key[4]}: "
                f"{base_tps} → {new_tps} tok/s ({delta_pct:+.2f}%)"
            )

    print("Per-shape Δ% vs baseline (negative = slower than baseline):")
    print("  T      K   dtype   baseline_tok/s   new_tok/s    Δ%")
    for (t, _h, _v, k, dtype), pct in deltas:
        nrow = new_by_key[(t, _h, _v, k, dtype)]
        brow = next(r for r in baseline["rows"] if row_key(r) == (t, _h, _v, k, dtype))
        print(
            f"  {t:5d}  {k:3d}  {dtype:5s}  {brow['kernel_tok_s']:14d}   "
            f"{nrow['kernel_tok_s']:10d}   {pct:+6.2f}"
        )

    if failures:
        print()
        print(f"FAIL — {len(failures)} regression(s) past {threshold:.1f}% threshold:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nOK — all shapes within ±{threshold:.1f}% of baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
