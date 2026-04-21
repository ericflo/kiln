#!/usr/bin/env python3
"""Phase C14 HF reference sidecar for the post-MTP-transformer-block splice dump.

Walks a directory tree written by `KILN_MTP_DUMP_SPLICE=1` (expected layout
`<root>/mtp_pos-{N}/step-{K}.safetensors`) and for each kiln dump:

  1. Runs `scripts/mtp_reference_dump.py` in a subprocess to produce a
     same-named reference file under `<root>-ref/mtp_pos-{N}/step-{K}.safetensors`.
  2. Compares the 3 post-block taps (c14__post_block, c14__post_norm,
     c14__logits) and the splice-source tap (h_main) between kiln and
     reference. Emits a JSON summary with per-tap cos_sim and max|Δ|.
  3. Flags any post-block site where cos_sim < 0.999 or max|Δ| > 1e-2.

The kiln dump MUST have been written with `KILN_MTP_DUMP_SPLICE=1`, which also
forces post-block capture (otherwise the `c14__*` taps will be missing and the
comparison is skipped for that file).

C14 extends the C13 bisect window past the `fc` projection: C13 proved the
*inputs* to the MTP head are clean (cos ≥ 0.9999928 for all 5 pre-projection
taps). C14 asks whether the divergence appears *inside* the MTP transformer
block (post_block), at the final norm (post_norm), or at the tied lm_head
(logits). Any tap that fails the cos ≥ 0.999 / max|Δ| ≤ 1e-2 thresholds tells
us exactly where to look next.

Usage
-----

    python3 scripts/c14_hf_reference_dump.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --kiln-root /workspace/kiln/c14-out/splice-dump \\
        --out-root  /workspace/kiln/c14-out/splice-dump-ref \\
        --summary   /workspace/kiln/c14-out/splice-summary.json

The script shells out to `mtp_reference_dump.py` one file at a time (cold
torch start per invocation is roughly 20–30 s; for 2 positions × 8 steps = 16
dumps the total HF side runs in ~5–10 minutes on CPU, which is acceptable for
a verdict pass).

Exit code is 0 on success (even when divergences are flagged — the point of
C14 is to *find* them). Exit code is 2 when any kiln dump fails to compare
(missing taps, subprocess error, corrupted safetensors).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover
    print(
        "[c14_hf_ref] safetensors not installed — run `pip install safetensors numpy torch` first",
        file=sys.stderr,
    )
    raise

# Post-block tap names captured by `c14_taps` when post-block capture is armed.
# Must match `C14_TAP_NAMES` in `crates/kiln-model/src/mtp_debug.rs`.
C14_TAP_NAMES = ["post_block", "post_norm", "logits"]
# Splice-source tap: the h_main feeding the MTP head. Lives in the legacy
# outer-tap namespace (no `c14__` prefix) under key `h_main`. Included so the
# C14 summary can cross-check the same sanity-identity row that C13 uses.
SPLICE_SOURCE_TAP = "h_main"

COS_SIM_FLOOR = 0.999
MAX_ABS_CEIL = 1e-2


@dataclass
class TapStats:
    tap: str
    shape: List[int]
    cos_sim: float
    max_abs_delta: float
    l2_delta: float
    kiln_l2: float
    ref_l2: float
    flagged: bool


@dataclass
class FileResult:
    mtp_pos: int
    step: int
    kiln_path: str
    ref_path: str
    taps: List[TapStats] = field(default_factory=list)
    missing_taps: List[str] = field(default_factory=list)
    error: Optional[str] = None


def parse_pos_step(path: Path) -> Optional[tuple[int, int]]:
    """Parse `mtp_pos-{N}/step-{K}.safetensors` -> (N, K)."""
    m = re.search(r"mtp_pos-(\d+)[/\\]step-(\d+)\.safetensors$", str(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_taps(path: Path, tap_names: list[str]) -> Dict[str, np.ndarray]:
    """Load a subset of taps from a safetensors file as float32 numpy."""
    out: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="np") as f:
        keys = set(f.keys())
        for name in tap_names:
            if name in keys:
                out[name] = np.asarray(f.get_tensor(name), dtype=np.float32)
    return out


def cos_sim_flat(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(af, bf) / (na * nb))


def compare_tap(name: str, kiln_t: np.ndarray, ref_t: np.ndarray) -> TapStats:
    assert kiln_t.shape == ref_t.shape, (
        f"shape mismatch for `{name}`: kiln {kiln_t.shape} vs ref {ref_t.shape}"
    )
    delta = kiln_t.astype(np.float32) - ref_t.astype(np.float32)
    max_abs = float(np.abs(delta).max())
    l2_delta = float(np.linalg.norm(delta))
    cos = cos_sim_flat(kiln_t, ref_t)
    k_l2 = float(np.linalg.norm(kiln_t.astype(np.float32)))
    r_l2 = float(np.linalg.norm(ref_t.astype(np.float32)))
    flagged = (cos < COS_SIM_FLOOR) or (max_abs > MAX_ABS_CEIL)
    return TapStats(
        tap=name,
        shape=list(kiln_t.shape),
        cos_sim=cos,
        max_abs_delta=max_abs,
        l2_delta=l2_delta,
        kiln_l2=k_l2,
        ref_l2=r_l2,
        flagged=flagged,
    )


def run_hf_reference(
    checkpoint: Path,
    kiln_dump: Path,
    out_path: Path,
    *,
    script: Path,
    extra_args: list[str],
) -> None:
    """Run `mtp_reference_dump.py --capture-subops` to emit a reference file.

    `--capture-subops` is set so the resulting file includes the full sub-op
    tap set; C14 only reads `c14__*` + `h_main` but sharing a reference file
    across C6/C7/C14 comparators keeps HF recompute cost to 1× per kiln dump.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--kiln-dump",
        str(kiln_dump),
        "--out",
        str(out_path),
        "--capture-subops",
    ] + extra_args
    print(f"[c14_hf_ref] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"mtp_reference_dump.py failed ({result.returncode}) for {kiln_dump}"
        )


def compare_file(kiln_path: Path, ref_path: Path, mtp_pos: int, step: int) -> FileResult:
    fr = FileResult(
        mtp_pos=mtp_pos,
        step=step,
        kiln_path=str(kiln_path),
        ref_path=str(ref_path),
    )
    wanted = [SPLICE_SOURCE_TAP] + [f"c14__{n}" for n in C14_TAP_NAMES]
    kiln_taps = load_taps(kiln_path, wanted)
    ref_taps = load_taps(ref_path, wanted)
    for name in wanted:
        if name not in kiln_taps or name not in ref_taps:
            fr.missing_taps.append(name)
            continue
        fr.taps.append(compare_tap(name, kiln_taps[name], ref_taps[name]))
    return fr


def summarize(results: List[FileResult]) -> Dict[str, object]:
    flagged_sites: List[Dict[str, object]] = []
    for r in results:
        for ts in r.taps:
            if ts.flagged:
                flagged_sites.append(
                    {
                        "mtp_pos": r.mtp_pos,
                        "step": r.step,
                        "tap": ts.tap,
                        "cos_sim": ts.cos_sim,
                        "max_abs_delta": ts.max_abs_delta,
                    }
                )
    # Median cos_sim per tap across all comparisons.
    by_tap: Dict[str, List[float]] = {}
    by_tap_max: Dict[str, List[float]] = {}
    for r in results:
        for ts in r.taps:
            by_tap.setdefault(ts.tap, []).append(ts.cos_sim)
            by_tap_max.setdefault(ts.tap, []).append(ts.max_abs_delta)
    tap_medians = {
        tap: {
            "median_cos_sim": float(np.median(vals)) if vals else float("nan"),
            "min_cos_sim": float(np.min(vals)) if vals else float("nan"),
            "median_max_abs": float(np.median(by_tap_max.get(tap, []))) if by_tap_max.get(tap) else float("nan"),
            "max_max_abs": float(np.max(by_tap_max.get(tap, []))) if by_tap_max.get(tap) else float("nan"),
            "n": len(vals),
        }
        for tap, vals in by_tap.items()
    }
    return {
        "cos_sim_floor": COS_SIM_FLOOR,
        "max_abs_ceil": MAX_ABS_CEIL,
        "total_files": len(results),
        "files_with_errors": sum(1 for r in results if r.error),
        "files_with_missing_taps": sum(1 for r in results if r.missing_taps),
        "flagged_sites": flagged_sites,
        "tap_medians": tap_medians,
        "files": [
            {
                "mtp_pos": r.mtp_pos,
                "step": r.step,
                "kiln_path": r.kiln_path,
                "ref_path": r.ref_path,
                "error": r.error,
                "missing_taps": r.missing_taps,
                "taps": [asdict(ts) for ts in r.taps],
            }
            for r in results
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase C14 HF reference sidecar for the post-MTP-transformer-block splice dump"
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Qwen3.5-4B HF checkpoint directory (passed through to mtp_reference_dump.py)",
    )
    ap.add_argument(
        "--kiln-root",
        required=True,
        type=Path,
        help="Root directory containing mtp_pos-{N}/step-{K}.safetensors files written by kiln",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Output directory for HF reference files (mirrors --kiln-root tree)",
    )
    ap.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="Output path for the JSON summary (includes flagged post-block sites)",
    )
    ap.add_argument(
        "--reference-script",
        type=Path,
        default=Path(__file__).resolve().parent / "mtp_reference_dump.py",
        help="Path to mtp_reference_dump.py (default: co-located in scripts/)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-running mtp_reference_dump.py for output files that already exist",
    )
    ap.add_argument(
        "--only-pos",
        type=int,
        action="append",
        default=None,
        help="Restrict comparison to specific mtp_pos values (repeatable). Default: all.",
    )
    ap.add_argument(
        "--reference-arg",
        action="append",
        default=[],
        help="Extra argument to pass through to mtp_reference_dump.py (repeatable)",
    )
    args = ap.parse_args()

    if not args.reference_script.exists():
        print(
            f"[c14_hf_ref] reference script not found: {args.reference_script}",
            file=sys.stderr,
        )
        return 2

    kiln_files = sorted(args.kiln_root.glob("mtp_pos-*/step-*.safetensors"))
    if not kiln_files:
        print(
            f"[c14_hf_ref] no kiln dumps found under {args.kiln_root}",
            file=sys.stderr,
        )
        return 2

    results: List[FileResult] = []
    any_error = False
    for kiln_path in kiln_files:
        parsed = parse_pos_step(kiln_path)
        if parsed is None:
            print(
                f"[c14_hf_ref] skipping unparseable path: {kiln_path}",
                file=sys.stderr,
            )
            continue
        mtp_pos, step = parsed
        if args.only_pos is not None and mtp_pos not in args.only_pos:
            continue
        rel = kiln_path.relative_to(args.kiln_root)
        ref_path = args.out_root / rel
        try:
            if args.skip_existing and ref_path.exists():
                print(
                    f"[c14_hf_ref] reusing existing reference {ref_path}",
                    file=sys.stderr,
                )
            else:
                run_hf_reference(
                    args.checkpoint,
                    kiln_path,
                    ref_path,
                    script=args.reference_script,
                    extra_args=list(args.reference_arg),
                )
            fr = compare_file(kiln_path, ref_path, mtp_pos, step)
        except Exception as exc:  # noqa: BLE001
            any_error = True
            fr = FileResult(
                mtp_pos=mtp_pos,
                step=step,
                kiln_path=str(kiln_path),
                ref_path=str(ref_path),
                error=str(exc),
            )
            print(f"[c14_hf_ref] ERROR for {kiln_path}: {exc}", file=sys.stderr)
        results.append(fr)

    summary = summarize(results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Human-readable digest to stderr.
    print(f"[c14_hf_ref] wrote summary to {args.summary}", file=sys.stderr)
    print(
        f"[c14_hf_ref] total files: {summary['total_files']}  "
        f"errors: {summary['files_with_errors']}  "
        f"missing-taps: {summary['files_with_missing_taps']}  "
        f"flagged sites: {len(summary['flagged_sites'])}",
        file=sys.stderr,
    )
    for tap, agg in summary["tap_medians"].items():
        print(
            f"  {tap:>20s}: median cos={agg['median_cos_sim']:.6f}  "
            f"min cos={agg['min_cos_sim']:.6f}  "
            f"median |Δ|_max={agg['median_max_abs']:.3e}  "
            f"max |Δ|_max={agg['max_max_abs']:.3e}  n={agg['n']}",
            file=sys.stderr,
        )
    if any_error:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
