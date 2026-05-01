#!/usr/bin/env python3
"""
Aggregate Phase B3 MTP A/B sweep logs.

Reads mtp-b3-seed{0..7}-{off,on}.log, extracts per-step mtp_draft + mtp_verify
events from the Latency Benchmark section only (stops at the Inference
Throughput section), computes per-run metrics:
  - alpha: mean accept rate on mtp_verify events
  - identity_bias: fraction of mtp_verify events where draft_token == last_token
  - mean_halves_ratio, mean_norm_emb_l2, mean_norm_h_l2 (from mtp_draft)
  - oov_draft_count: drafts where draft_token >= 151936 (Qwen3.5 vocab)
  - n_drafts, n_verifies

Outputs an 8x2 matrix and paired-difference stats (ON - OFF).
"""
import os
import re
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

BASE = Path(os.environ.get("B3_DIR", "/tmp/b3"))
VOCAB = 151936

ANSI_RE = re.compile(r"\x1b\[[\d;]*m")
KV_RE = re.compile(r"(\w+)=([-\w.]+)")
DRAFT_RE = re.compile(r"mtp_draft\s+(.*)$")
VERIFY_RE = re.compile(r"mtp_verify\s+(.*)$")
STOP_RE = re.compile(r"Inference Throughput Benchmarks")


def parse_kv(s):
    d = {}
    for k, v in KV_RE.findall(s):
        if v in ("true", "false"):
            d[k] = v == "true"
            continue
        try:
            if "." in v or ("e" in v.lower() and v.lower() != "false"):
                d[k] = float(v)
            else:
                d[k] = int(v)
        except ValueError:
            d[k] = v
    return d


def analyze_log(path: Path):
    drafts = []
    verifies = []
    identity_hits = 0
    oov_count = 0
    stopped = False
    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            if STOP_RE.search(line):
                stopped = True
                break
            m = DRAFT_RE.search(line)
            if m:
                drafts.append(parse_kv(m.group(1)))
                continue
            m = VERIFY_RE.search(line)
            if m:
                kv = parse_kv(m.group(1))
                verifies.append(kv)
                dt = kv.get("draft_token")
                lt = kv.get("last_token")
                if isinstance(dt, int) and isinstance(lt, int) and dt == lt:
                    identity_hits += 1
                if isinstance(dt, int) and dt >= VOCAB:
                    oov_count += 1

    n_v = len(verifies)
    accepted_count = sum(1 for v in verifies if v.get("accepted") is True)
    alpha = accepted_count / n_v if n_v else 0.0
    n_d = len(drafts)
    id_bias = identity_hits / n_v if n_v else 0.0

    def safe_mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return mean(vals) if vals else 0.0

    return {
        "alpha": alpha,
        "identity_bias": id_bias,
        "mean_halves_ratio": safe_mean([d.get("halves_ratio", 0.0) for d in drafts]),
        "mean_norm_emb_l2": safe_mean([d.get("norm_emb_l2", 0.0) for d in drafts]),
        "mean_norm_h_l2": safe_mean([d.get("norm_h_l2", 0.0) for d in drafts]),
        "oov_draft_count": oov_count,
        "n_drafts": n_d,
        "n_verifies": n_v,
        "stopped_at_throughput": stopped,
    }


def main():
    seeds = list(range(8))
    arms = ["off", "on"]
    rows = {}
    for seed in seeds:
        rows[seed] = {}
        for arm in arms:
            path = BASE / f"mtp-b3-seed{seed}-{arm}.log"
            if not path.exists():
                print(f"WARN missing {path}", file=sys.stderr)
                rows[seed][arm] = None
                continue
            rows[seed][arm] = analyze_log(path)

    hdr = f"{'seed':>4} {'arm':>3} {'alpha':>7} {'id_bias':>8} {'halves':>8} {'norm_e':>8} {'norm_h':>8} {'oov':>4} {'n_d':>4} {'n_v':>4}"
    print(hdr)
    print("-" * len(hdr))
    for seed in seeds:
        for arm in arms:
            r = rows[seed][arm]
            if r is None:
                print(f"{seed:>4} {arm:>3} missing")
                continue
            print(
                f"{seed:>4} {arm:>3} {r['alpha']:>7.3f} {r['identity_bias']:>8.3f} "
                f"{r['mean_halves_ratio']:>8.3f} {r['mean_norm_emb_l2']:>8.3f} "
                f"{r['mean_norm_h_l2']:>8.3f} {r['oov_draft_count']:>4d} "
                f"{r['n_drafts']:>4d} {r['n_verifies']:>4d}"
            )

    print()
    print("Paired ON - OFF (positive = swap helps):")
    deltas = {"alpha": [], "identity_bias": [], "halves_ratio": [], "oov_draft_count": []}
    on_wins = 0
    valid = 0
    for seed in seeds:
        off = rows[seed]["off"]
        on = rows[seed]["on"]
        if off is None or on is None:
            continue
        valid += 1
        d_alpha = on["alpha"] - off["alpha"]
        d_id = on["identity_bias"] - off["identity_bias"]
        d_halves = on["mean_halves_ratio"] - off["mean_halves_ratio"]
        d_oov = on["oov_draft_count"] - off["oov_draft_count"]
        deltas["alpha"].append(d_alpha)
        deltas["identity_bias"].append(d_id)
        deltas["halves_ratio"].append(d_halves)
        deltas["oov_draft_count"].append(d_oov)
        if d_alpha > 0:
            on_wins += 1
        print(
            f"  seed={seed} d_alpha={d_alpha:+.3f} d_id_bias={d_id:+.3f} "
            f"d_halves={d_halves:+.3f} d_oov={d_oov:+d}"
        )

    def summarize(vals, name):
        if not vals:
            return
        print(
            f"  {name:>14}: mean={mean(vals):+.3f} std={pstdev(vals):.3f} "
            f"min={min(vals):+.3f} max={max(vals):+.3f} n={len(vals)}"
        )

    print()
    summarize(deltas["alpha"], "d_alpha")
    summarize(deltas["identity_bias"], "d_id_bias")
    summarize(deltas["halves_ratio"], "d_halves")
    summarize(deltas["oov_draft_count"], "d_oov")
    print()
    print(f"ON > OFF on d_alpha: {on_wins}/{valid} seeds")

    with open(BASE / "b3-aggregate.json", "w") as f:
        json.dump(
            {"per_seed": rows, "deltas": deltas, "on_wins": on_wins, "valid": valid},
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {BASE}/b3-aggregate.json")


if __name__ == "__main__":
    main()
