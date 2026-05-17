#!/usr/bin/env python3
"""GRPO Phase 1+2 ablation analyzer.

Reads cuda_grpo_ablation progress logs (one per mode), extracts per-step loss
curves and final summaries, and prints a side-by-side comparison table.

Usage:
    python3 analyze.py /path/to/logs_dir
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from statistics import mean

PROGRESS_RE = re.compile(r"^progress step=(\d+)/(\d+)\s+loss=([-+\d\.eE]+)\s+vram_mib=(\d+)")
ELAPSED_RE = re.compile(r"^elapsed_secs=([-+\d\.eE]+)")
PEAK_RE = re.compile(r"^peak_vram_mib=(\d+)")
ADAPTER_RE = re.compile(r"^adapter=(.+)$")
MODE_RE = re.compile(r"^mode_done=(\S+)")
CONFIG_RE = re.compile(r"^config mode=\S+ (.+)$")


def parse_log(path: Path) -> dict:
    losses: list[float] = []
    vrams: list[int] = []
    total_steps = 0
    elapsed = None
    peak_vram = None
    adapter = None
    mode = None
    config = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := PROGRESS_RE.match(line):
            total_steps = max(total_steps, int(m.group(2)))
            losses.append(float(m.group(3)))
            vrams.append(int(m.group(4)))
        elif m := ELAPSED_RE.match(line):
            elapsed = float(m.group(1))
        elif m := PEAK_RE.match(line):
            peak_vram = int(m.group(1))
        elif m := ADAPTER_RE.match(line):
            adapter = m.group(1)
        elif m := MODE_RE.match(line):
            mode = m.group(1)
        elif m := CONFIG_RE.match(line):
            config = m.group(1)
    return {
        "mode": mode or path.stem,
        "config": config,
        "steps": len(losses),
        "total_steps": total_steps,
        "losses": losses,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_loss": mean(losses) if losses else None,
        "peak_vram_mib": peak_vram,
        "elapsed_secs": elapsed,
        "adapter": adapter,
    }


def main(logs_dir: str) -> int:
    root = Path(logs_dir)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1
    summaries = [parse_log(p) for p in sorted(root.glob("*.log"))]
    if not summaries:
        print(f"no .log files in {root}", file=sys.stderr)
        return 1

    # Compact table.
    header = ["mode", "steps", "first_loss", "last_loss", "mean_loss", "peak_vram_mib", "elapsed_secs"]
    widths = {h: max(len(h), max(len(str(s.get(h) if h != "mode" else s["mode"])) for s in summaries)) for h in header}
    fmt = "  ".join(f"{{:<{widths[h]}}}" for h in header)
    print(fmt.format(*header))
    print(fmt.format(*["-" * widths[h] for h in header]))
    for s in summaries:
        row = [
            s["mode"],
            s["steps"],
            f"{s['first_loss']:.6f}" if s["first_loss"] is not None else "-",
            f"{s['last_loss']:.6f}" if s["last_loss"] is not None else "-",
            f"{s['mean_loss']:.6f}" if s["mean_loss"] is not None else "-",
            s["peak_vram_mib"] if s["peak_vram_mib"] is not None else "-",
            f"{s['elapsed_secs']:.1f}" if s["elapsed_secs"] is not None else "-",
        ]
        print(fmt.format(*[str(x) for x in row]))

    # Per-mode loss trajectory.
    print("")
    print("loss trajectories (every other step):")
    for s in summaries:
        sample = s["losses"][::2]
        line = " ".join(f"{x:+.3f}" for x in sample[:30])
        print(f"  {s['mode']:<20} [{s['steps']} steps] {line}{' ...' if len(s['losses']) > 60 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
