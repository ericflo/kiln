#!/usr/bin/env python3
import argparse
import json
import random
import re
import statistics
from pathlib import Path


def seed_key(path: Path) -> int:
    match = re.search(r"seed-(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"unexpected seed filename: {path}")
    return int(match.group(1))


def load_rows(artifacts_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(artifacts_dir.glob("seed-*.json"), key=seed_key):
        data = json.loads(path.read_text())
        latency = data["latency"]
        rows.append(
            {
                "seed": seed_key(path),
                "acceptance_rate": latency["acceptance_rate"],
                "decode_tokens_per_sec": latency["decode_tokens_per_sec"],
                "mean_inter_token_ms": latency["mean_inter_token_ms"],
                "prefill_tokens_per_sec": latency["prefill_tokens_per_sec"],
                "prefill_time_ms": latency["prefill_time_ms"],
                "model_load_secs": data["model_load"]["load_time_secs"],
                "model_vram_mb": data["model_load"]["model_vram_mb"],
            }
        )
    return rows


def bootstrap_ci(values: list[float], resamples: int, rng_seed: int) -> tuple[float, float]:
    rng = random.Random(rng_seed)
    medians = []
    n = len(values)
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo = medians[int(0.025 * resamples)]
    hi = medians[int(0.975 * resamples)]
    return lo, hi


def summarize(rows: list[dict], resamples: int, rng_seed: int) -> dict:
    alphas = [row["acceptance_rate"] for row in rows]
    decode_tps = [row["decode_tokens_per_sec"] for row in rows]
    mean_itl = [row["mean_inter_token_ms"] for row in rows]
    ci_lo, ci_hi = bootstrap_ci(alphas, resamples=resamples, rng_seed=rng_seed)
    return {
        "n": len(rows),
        "median_alpha": statistics.median(alphas),
        "mean_alpha": statistics.mean(alphas),
        "stdev_alpha": statistics.stdev(alphas),
        "min_alpha": min(alphas),
        "max_alpha": max(alphas),
        "bootstrap_median_ci_95": {
            "resamples": resamples,
            "rng_seed": rng_seed,
            "lo": ci_lo,
            "hi": ci_hi,
        },
        "num_alpha_ge_0_72": sum(alpha >= 0.72 for alpha in alphas),
        "median_decode_tokens_per_sec": statistics.median(decode_tps),
        "median_mean_inter_token_ms": statistics.median(mean_itl),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing seed-*.json artifacts",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=12345)
    parser.add_argument("--out", help="Optional path to write summary JSON")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    rows = load_rows(artifacts_dir)
    summary = summarize(rows, resamples=args.resamples, rng_seed=args.rng_seed)
    payload = {"rows": rows, "summary": summary}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
