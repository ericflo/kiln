#!/usr/bin/env python3
"""
Phase C12 — bench runner for α + top-1 agreement under 3 configs × 3 trials.

Per Phase C5/C9 convention, seeds 42/43/44 are used for median-of-3 α. Under
greedy decode (kiln-bench default temperature=0.0), Class C (`mtp_top1 ==
main_top1` but not accepted) was empirically 0/339 in Phase C5, so α ==
top-1-agreement up to sampling variance. The bench reports α directly via the
JSON `acceptance_rate` field; top-1 agreement is recorded as `alpha` per seed.

Configs:
  - baseline       : KILN_W4A16=1
  - primary        : KILN_W4A16=1  KILN_MTP_FP32_HEAD=1
  - sanity_nomarlin: KILN_W4A16=0  (establishes ceiling without Marlin drift)

All three carry KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp and the standard
bench flags (`--paged --prompt-tokens 512 --max-output-tokens 128
--skip-training`).

Emits:
  - stdout: human-readable table
  - JSON:   --json-out <path> (for the verdict doc)
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


SEEDS = [42, 43, 44]

CONFIGS = [
    {
        "name": "baseline",
        "env": {
            "KILN_W4A16": "1",
        },
    },
    {
        "name": "primary",
        "env": {
            "KILN_W4A16": "1",
            "KILN_MTP_FP32_HEAD": "1",
        },
    },
    {
        "name": "sanity_nomarlin",
        "env": {
            "KILN_W4A16": "0",
        },
    },
]

MTP_BASE_ENV = {
    "KILN_SPEC_ENABLED": "1",
    "KILN_SPEC_METHOD": "mtp",
}


def run_one(
    bench_bin: Path,
    model_path: Path,
    seed: int,
    extra_env: Dict[str, str],
) -> Dict:
    """Run kiln-bench once, return parsed JSON dict + raw α string."""
    env = os.environ.copy()
    env.update(MTP_BASE_ENV)
    env.update(extra_env)

    cmd = [
        str(bench_bin),
        "--model-path",
        str(model_path),
        "--paged",
        "--prompt-tokens",
        "512",
        "--max-output-tokens",
        "128",
        "--skip-training",
        "--seed",
        str(seed),
    ]

    print(f"    → seed={seed} env=" + " ".join(f"{k}={v}" for k, v in extra_env.items()), flush=True)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        sys.stderr.write(f"[bench_runner] seed={seed} exit={proc.returncode}\n")
        sys.stderr.write(f"--- stderr ---\n{proc.stderr[-2000:]}\n")
        raise RuntimeError(f"kiln-bench failed (seed={seed}, rc={proc.returncode})")

    # kiln-bench prints JSON to stdout; logs + alpha line go to stderr
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(f"[bench_runner] non-JSON stdout:\n{proc.stdout[-2000:]}\n")
        raise

    alpha = result.get("latency", {}).get("acceptance_rate")
    # Extract the "(k/N)" accept count from stderr for transparency
    accept_line = ""
    for line in proc.stderr.splitlines():
        if "α =" in line or "α=" in line:
            accept_line = line.strip()
            break

    return {
        "seed": seed,
        "alpha": alpha,
        "top1_agreement": alpha,  # greedy decode: Class C == 0 (C5)
        "decode_tok_per_sec": result.get("latency", {}).get("decode_tokens_per_sec"),
        "mean_itl_ms": result.get("latency", {}).get("mean_inter_token_ms"),
        "accept_line": accept_line,
        "elapsed_s": round(elapsed, 1),
    }


def run_config(
    bench_bin: Path,
    model_path: Path,
    name: str,
    extra_env: Dict[str, str],
) -> Dict:
    print(f"\n[config={name}] env={extra_env}", flush=True)
    trials = []
    for seed in SEEDS:
        trials.append(run_one(bench_bin, model_path, seed, extra_env))

    alphas = [t["alpha"] for t in trials if t["alpha"] is not None]
    decodes = [t["decode_tok_per_sec"] for t in trials if t["decode_tok_per_sec"]]
    median_alpha = statistics.median(alphas) if alphas else None
    median_decode = statistics.median(decodes) if decodes else None

    print(
        f"  → median α = {median_alpha:.4f}" if median_alpha is not None else "  → median α = N/A"
    )
    print(
        f"  → median decode = {median_decode:.2f} tok/s" if median_decode else ""
    )
    return {
        "name": name,
        "env": extra_env,
        "trials": trials,
        "median_alpha": median_alpha,
        "median_top1_agreement": median_alpha,
        "median_decode_tok_per_sec": median_decode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-bin", required=True, help="Path to kiln-bench binary")
    ap.add_argument("--model-path", required=True, help="Path to Qwen3.5-4B weights")
    ap.add_argument("--json-out", required=True, help="Write summary JSON here")
    ap.add_argument(
        "--config",
        choices=["all"] + [c["name"] for c in CONFIGS],
        default="all",
        help="Run only one config (for debugging). Default: all three.",
    )
    args = ap.parse_args()

    bench_bin = Path(args.bench_bin).resolve()
    model_path = Path(args.model_path).resolve()
    if not bench_bin.exists():
        sys.exit(f"bench binary not found: {bench_bin}")
    if not model_path.exists():
        sys.exit(f"model path not found: {model_path}")

    configs_to_run = CONFIGS if args.config == "all" else [c for c in CONFIGS if c["name"] == args.config]

    summary: Dict = {
        "seeds": SEEDS,
        "bench_flags": [
            "--paged",
            "--prompt-tokens",
            "512",
            "--max-output-tokens",
            "128",
            "--skip-training",
        ],
        "configs": [],
    }

    t0 = time.time()
    for cfg in configs_to_run:
        result = run_config(bench_bin, model_path, cfg["name"], cfg["env"])
        summary["configs"].append(result)

    summary["total_elapsed_s"] = round(time.time() - t0, 1)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[bench_runner] wrote {out_path}")

    # Pretty table
    print("\n=== Median-of-3 Summary ===")
    header = f"{'config':<22} {'median α':>10} {'median top1':>12} {'median tok/s':>14}"
    print(header)
    print("-" * len(header))
    for c in summary["configs"]:
        a = c["median_alpha"]
        t1 = c["median_top1_agreement"]
        d = c["median_decode_tok_per_sec"]
        a_s = f"{a:.4f}" if a is not None else "N/A"
        t1_s = f"{t1:.4f}" if t1 is not None else "N/A"
        d_s = f"{d:.2f}" if d is not None else "N/A"
        print(f"{c['name']:<22} {a_s:>10} {t1_s:>12} {d_s:>14}")


if __name__ == "__main__":
    main()
