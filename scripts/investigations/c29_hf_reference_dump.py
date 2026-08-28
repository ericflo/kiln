#!/usr/bin/env python3
"""Phase C29 HF reference sidecar — loops c14_hf_reference_dump.py over a
multi-prompt kiln dump tree.

Layout consumed (matches what c29_kiln_logits_dump.sh writes):

    <kiln_root>/
        seed-{S}/
            mtp_pos-{N}/step-{K}.safetensors

Layout produced:

    <out_root>/
        seed-{S}/
            mtp_pos-{N}/step-{K}.safetensors
        seed-{S}/c29-c14-summary.json   (per-seed C14 reference summary)

This script is a thin scheduler. The heavy lifting (subprocess to
`mtp_reference_dump.py --capture-subops` per kiln dump file) lives in
`c14_hf_reference_dump.py`; we just dispatch one invocation per `seed-*`
subtree so the C29 comparator has matching reference safetensors for every
kiln dump in every prompt.

Cost envelope: ~20-30 s cold-start CPU per kiln dump × N seeds × M positions
× K steps. A 4×4×2 = 32-dump matrix runs in ~10-15 min on CPU, well under
the 90-min wall-clock cap.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase C29 HF reference sidecar for multi-prompt kiln logits dump"
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Qwen3.5-4B HF checkpoint directory",
    )
    ap.add_argument(
        "--kiln-root",
        required=True,
        type=Path,
        help="Root directory containing seed-{S}/mtp_pos-{N}/step-{K}.safetensors",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Output directory (mirrors --kiln-root tree under seed-{S}/...)",
    )
    ap.add_argument(
        "--c14-script",
        type=Path,
        default=Path(__file__).resolve().parent / "c14_hf_reference_dump.py",
        help="Path to c14_hf_reference_dump.py (default: co-located in scripts/)",
    )
    ap.add_argument(
        "--reference-script",
        type=Path,
        default=Path(__file__).resolve().parent / "mtp_reference_dump.py",
        help="Path to mtp_reference_dump.py (passed through to c14 wrapper)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip per-file HF re-runs when the output safetensors already exists",
    )
    args = ap.parse_args()

    if not args.c14_script.exists():
        print(
            f"[c29_hf_ref] c14 wrapper script not found: {args.c14_script}",
            file=sys.stderr,
        )
        return 2
    if not args.reference_script.exists():
        print(
            f"[c29_hf_ref] reference script not found: {args.reference_script}",
            file=sys.stderr,
        )
        return 2

    seed_dirs = sorted(d for d in args.kiln_root.glob("seed-*") if d.is_dir())
    if not seed_dirs:
        print(
            f"[c29_hf_ref] no seed-* subdirectories under {args.kiln_root}",
            file=sys.stderr,
        )
        return 2

    print(
        f"[c29_hf_ref] dispatching HF reference for {len(seed_dirs)} prompt seeds",
        file=sys.stderr,
    )

    any_error = False
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        out_seed_dir = args.out_root / seed_name
        summary_path = out_seed_dir / "c29-c14-summary.json"
        out_seed_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(args.c14_script),
            "--checkpoint",
            str(args.checkpoint),
            "--kiln-root",
            str(seed_dir),
            "--out-root",
            str(out_seed_dir),
            "--summary",
            str(summary_path),
            "--reference-script",
            str(args.reference_script),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        print(
            f"[c29_hf_ref] === {seed_name}: invoking c14 wrapper ===",
            file=sys.stderr,
        )
        result = subprocess.run(cmd)
        if result.returncode != 0:
            any_error = True
            print(
                f"[c29_hf_ref] {seed_name}: c14 wrapper exited {result.returncode}",
                file=sys.stderr,
            )

    print(
        f"[c29_hf_ref] DONE — produced reference dumps for {len(seed_dirs)} seeds under {args.out_root}",
        file=sys.stderr,
    )
    return 2 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
