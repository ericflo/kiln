#!/usr/bin/env python3
"""
Perf-regression check for SFT/GRPO/OPD step-time on A6000 (#1077 Tier 2).

Mirrors `bench-results/check_opd_regression.py`'s shape — parses the JSON
that `kiln-bench` prints to stdout, compares the `training.secs_per_step`
and `training.peak_vram_mb` fields against a pinned baseline JSON, and
exits non-zero on regression.

Usage
-----

    python3 bench-results/check_sft_train_regression.py \
        --bench-stdout /tmp/kiln_bench_stdout.txt \
        --baseline bench-results/regression/sft_train_a6000_baseline.json \
        [--secs-per-step-tolerance 0.10] \
        [--peak-vram-tolerance 0.15]

Exit codes:

    0  — within tolerance (or first-run case where the baseline is the
         placeholder default of `null` — write the observed numbers to the
         baseline file via --write-baseline-if-null).
    1  — regression detected (or stdout JSON is malformed / missing the
         `training` field).
    2  — invalid arguments / baseline file missing.

Baseline shape
--------------

    {
      "schema_version": 1,
      "workload": "sft_short",        # any string identifier
      "trainer": "native|generic",
      "gpu": "NVIDIA RTX A6000",
      "secs_per_step": 0.85,          # null on the placeholder
      "peak_vram_mb": 12_400,         # null on the placeholder
      "comment": "set by first nightly run on 2026-MM-DD",
      "pinned_at_commit": "abcdef..."  # optional
    }

A `null` secs_per_step or peak_vram_mb means "no baseline pinned yet" —
the first nightly run will write its observed numbers via
`--write-baseline-if-null` and from then on the script gates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def find_json_blob(text: str) -> dict:
    """
    kiln-bench prints a JSON blob to stdout at the end of its run. The
    blob is pretty-printed across many lines. Find the outermost
    {...} that includes the `"training"` field.

    Returns the parsed JSON dict, or raises ValueError if no valid
    blob is found.
    """
    # Greedy match — kiln-bench emits one blob. Strip ANSI escape
    # codes that may have bled in from the human-readable summary,
    # though stderr is the usual sink for those.
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    decoder = json.JSONDecoder()
    # Scan for `{` and try to decode forward — first successful decode
    # that contains a `training` field wins.
    idx = 0
    while True:
        next_brace = text.find("{", idx)
        if next_brace < 0:
            raise ValueError(
                "could not find a JSON object in bench stdout — did kiln-bench fail before "
                "emitting the JSON summary?"
            )
        try:
            obj, end = decoder.raw_decode(text[next_brace:])
            if isinstance(obj, dict) and "training" in obj:
                return obj
            # Decoded but not the right blob — keep scanning.
            idx = next_brace + end
        except json.JSONDecodeError:
            idx = next_brace + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--bench-stdout",
        required=True,
        help="Path to a file containing kiln-bench's stdout (e.g. /tmp/kiln_bench_stdout.txt).",
    )
    p.add_argument(
        "--baseline",
        required=True,
        help="Path to the pinned baseline JSON.",
    )
    p.add_argument(
        "--secs-per-step-tolerance",
        type=float,
        default=0.10,
        help="Fractional regression tolerance for secs_per_step (default 0.10 = 10%%).",
    )
    p.add_argument(
        "--peak-vram-tolerance",
        type=float,
        default=0.15,
        help="Fractional regression tolerance for peak_vram_mb (default 0.15 = 15%%).",
    )
    p.add_argument(
        "--write-baseline-if-null",
        action="store_true",
        help="If the baseline's secs_per_step or peak_vram_mb is null, write the observed "
             "values to the baseline file and exit 0. Used by the very first nightly run on a "
             "new workload row to seed the baseline.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a one-line summary and exit, without failing on regressions. Useful for "
             "dry-run validation of the harness.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    bench_path = Path(args.bench_stdout)
    if not bench_path.is_file():
        print(f"error: bench stdout file not found at {bench_path}", file=sys.stderr)
        return 2

    baseline_path = Path(args.baseline)
    if not baseline_path.is_file():
        print(f"error: baseline JSON not found at {baseline_path}", file=sys.stderr)
        return 2

    try:
        results = find_json_blob(bench_path.read_text())
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    training = results.get("training")
    if not isinstance(training, dict):
        print("error: bench JSON has no `training` field — was --skip-training set?", file=sys.stderr)
        return 1

    observed_secs = training.get("secs_per_step")
    observed_vram = training.get("peak_vram_mb")
    if not isinstance(observed_secs, (int, float)) or observed_secs <= 0:
        print(f"error: bench `secs_per_step` is invalid: {observed_secs!r}", file=sys.stderr)
        return 1
    if not isinstance(observed_vram, (int, float)) or observed_vram <= 0:
        print(f"error: bench `peak_vram_mb` is invalid: {observed_vram!r}", file=sys.stderr)
        return 1

    baseline = json.loads(baseline_path.read_text())
    baseline_secs = baseline.get("secs_per_step")
    baseline_vram = baseline.get("peak_vram_mb")

    workload = baseline.get("workload", "<unknown>")
    trainer = baseline.get("trainer", "<unknown>")
    gpu = baseline.get("gpu", "<unknown>")

    print(
        f"workload={workload} trainer={trainer} gpu={gpu}",
        file=sys.stderr,
    )
    print(
        f"observed: secs_per_step={observed_secs:.4f} peak_vram_mb={observed_vram}",
        file=sys.stderr,
    )
    print(
        f"baseline: secs_per_step={baseline_secs} peak_vram_mb={baseline_vram}",
        file=sys.stderr,
    )

    if (baseline_secs is None or baseline_vram is None) and args.write_baseline_if_null:
        baseline["secs_per_step"] = float(observed_secs)
        baseline["peak_vram_mb"] = int(observed_vram)
        baseline.setdefault("comment", "auto-seeded by first nightly run")
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
        print(
            f"OK — wrote first-run baseline to {baseline_path} (secs={observed_secs:.4f}, "
            f"vram={observed_vram} MB). The next run will gate against these values.",
            file=sys.stderr,
        )
        return 0

    if baseline_secs is None or baseline_vram is None:
        print(
            "error: baseline secs_per_step or peak_vram_mb is null and "
            "--write-baseline-if-null was not set",
            file=sys.stderr,
        )
        return 1

    if args.summary_only:
        delta_secs = (observed_secs - baseline_secs) / baseline_secs
        delta_vram = (observed_vram - baseline_vram) / baseline_vram
        print(
            f"summary: secs_delta={delta_secs:+.2%} vram_delta={delta_vram:+.2%}",
            file=sys.stderr,
        )
        return 0

    failures = []
    delta_secs = (observed_secs - baseline_secs) / baseline_secs
    delta_vram = (observed_vram - baseline_vram) / baseline_vram

    if delta_secs > args.secs_per_step_tolerance:
        failures.append(
            f"secs_per_step regressed {delta_secs:+.2%} "
            f"(observed {observed_secs:.4f} vs baseline {baseline_secs:.4f}, "
            f"tolerance ±{args.secs_per_step_tolerance:.0%})"
        )

    if delta_vram > args.peak_vram_tolerance:
        failures.append(
            f"peak_vram_mb regressed {delta_vram:+.2%} "
            f"(observed {observed_vram} vs baseline {baseline_vram} MB, "
            f"tolerance ±{args.peak_vram_tolerance:.0%})"
        )

    if failures:
        print("REGRESSION:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "To accept these as the new baseline, re-run with --write-baseline-if-null after "
            "first nulling out the corresponding fields in the baseline JSON.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK — secs_delta={delta_secs:+.2%} vram_delta={delta_vram:+.2%} (both within tolerance)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
