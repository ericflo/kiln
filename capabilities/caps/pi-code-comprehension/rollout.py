"""Rollout runner for pi-code-comprehension.

Identical in shape to capabilities/agentic-grpo/pi-doctest/rollout.py:
spawns one pi session per (task, gen) in an isolated workdir, captures the
session JSONL, scores it via `rubric.score_rollout`, and emits either a
GRPO training group JSONL or an eval summary.

Usage:
  rollout.py --tasks <tasks.jsonl> --out-dir <dir>
             [--adapter NAME|current]
             [--num-generations 4]
             [--mode train|eval]
             [--kiln-url http://localhost:8420]
             [--max-wall-clock-s 180]
             [--task-limit N]
             [--concurrency 1]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lib"))
import rubric  # noqa: E402
import task_scaffold  # noqa: E402
import pi_trajectory  # noqa: E402


PI_BIN = os.environ.get("PI_BIN", "pi")


def kiln_set_adapter(url: str, adapter: str | None) -> None:
    if adapter is None or adapter == "":
        req = urllib.request.Request(
            f"{url}/v1/adapters/unload", data=b"", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return
            raise
        return
    body = json.dumps({"name": adapter}).encode()
    req = urllib.request.Request(
        f"{url}/v1/adapters/load", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def latest_session_jsonl(session_dir: Path) -> Path | None:
    candidates = list(session_dir.rglob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_transcript(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def run_one(task: dict, gen_idx: int, out_root: Path, kiln_url: str,
            max_wall_clock_s: int, verbose: bool) -> dict:
    tdir = out_root / task["task_id"] / f"gen{gen_idx:02d}"
    workdir = tdir / "workdir"
    session_dir = tdir / "sessions"
    workdir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    task_scaffold.init_workdir(task, str(workdir))
    prompt = task_scaffold.pi_prompt(task)

    cmd = [
        "timeout", f"{max_wall_clock_s}s",
        PI_BIN, "-p", prompt,
        "--session-dir", str(session_dir),
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-themes",
        "--offline",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)
    elapsed = time.time() - t0
    transcript_path = latest_session_jsonl(session_dir)
    transcript = parse_transcript(transcript_path)
    score = rubric.score_rollout(transcript, str(workdir), task)
    record = {
        "task_id": task["task_id"],
        "gen": gen_idx,
        "reward": score["composite"],
        "sub_scores": {k: v for k, v in score.items() if not k.startswith("_")},
        "diagnostics": {k: v for k, v in score.items() if k.startswith("_") and not k.startswith("_diag")},
        "wall_clock_s": elapsed,
        "exit_code": proc.returncode,
        "transcript_path": str(transcript_path) if transcript_path else None,
        "workdir": str(workdir),
        "stdout_tail": (proc.stdout or "")[-400:],
        "stderr_tail": (proc.stderr or "")[-400:],
    }
    if verbose:
        print(
            f"  task={task['task_id']} gen={gen_idx} "
            f"reward={score['composite']:.3f} outcome={score['outcome']:.2f} "
            f"wall={elapsed:.1f}s exit={proc.returncode}",
            flush=True,
        )
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter", default="",
                    help="Adapter name, '' for base, 'current' to skip switch.")
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--kiln-url", default=os.environ.get("KILN_URL", "http://localhost:8420"))
    ap.add_argument("--max-wall-clock-s", type=int, default=180)
    ap.add_argument("--task-limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if args.task_limit > 0:
        tasks = tasks[: args.task_limit]

    print(f"loaded {len(tasks)} tasks; num_gens={args.num_generations} "
          f"adapter={args.adapter or '(base)'} mode={args.mode}", flush=True)

    if args.adapter == "current":
        print("using currently-loaded kiln adapter (no switch)", flush=True)
    else:
        try:
            kiln_set_adapter(args.kiln_url, args.adapter or None)
            print(f"set kiln active adapter to {args.adapter or '(base)'}", flush=True)
        except Exception as e:
            print(f"WARN: adapter switch failed ({e}); proceeding with current", flush=True)

    started = time.time()
    work = [(t, g) for t in tasks for g in range(args.num_generations)]
    records: list[dict] = []
    if args.concurrency <= 1:
        for t, g in work:
            records.append(run_one(t, g, out_root, args.kiln_url,
                                   args.max_wall_clock_s, args.verbose))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(run_one, t, g, out_root, args.kiln_url,
                              args.max_wall_clock_s, args.verbose)
                    for t, g in work]
            for f in futs:
                records.append(f.result())
    elapsed = time.time() - started

    rec_path = out_root / "rollouts.jsonl"
    with rec_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    if args.mode == "train":
        by_task: dict[str, list[dict]] = {}
        for r in records:
            by_task.setdefault(r["task_id"], []).append(r)
        groups: list[dict] = []
        for t in tasks:
            rolls = by_task.get(t["task_id"], [])
            if len(rolls) < 2:
                continue
            messages = [
                {"role": "system",
                 "content": "You are a Python code-comprehension assistant. "
                            "You have access to read, edit, write, and bash "
                            "tools. Investigate the codebase, then emit your "
                            "final answer as a single <answer>{...}</answer> "
                            "JSON block."},
                {"role": "user", "content": task_scaffold.pi_prompt(t)},
            ]
            completions: list[dict] = []
            for r in rolls:
                tp = r.get("transcript_path")
                if tp:
                    rollout_dict = pi_trajectory.build_scored_rollout(
                        Path(tp), reward=r["reward"])
                    if not rollout_dict.get("trajectory"):
                        rollout_dict["text"] = "(empty)"
                else:
                    rollout_dict = {"text": "(empty)", "reward": r["reward"],
                                    "trajectory": []}
                completions.append(rollout_dict)
            groups.append({"messages": messages, "completions": completions})
        grp_path = out_root / "grpo-train.jsonl"
        with grp_path.open("w") as f:
            for g in groups:
                f.write(json.dumps(g) + "\n")
        print(f"wrote {len(groups)} GRPO groups → {grp_path}", flush=True)

    if records:
        composites = [r["reward"] for r in records]
        outcomes = [r["sub_scores"].get("outcome", 0) for r in records]
        groundings = [r["sub_scores"].get("grounding", 0) for r in records]
        cross_file = [r["sub_scores"].get("cross_file_caller_recall", 0) for r in records]
        invariants = [r["sub_scores"].get("invariant_coverage", 0) for r in records]
        fmts = [r["sub_scores"].get("format_compliance", 0) for r in records]
        summary = {
            "mode": args.mode,
            "adapter": args.adapter or "",
            "n_tasks": len(tasks),
            "n_rollouts": len(records),
            "n_generations": args.num_generations,
            "mean_composite": sum(composites) / len(composites),
            "p50_composite": sorted(composites)[len(composites) // 2],
            "p05_composite": sorted(composites)[max(0, len(composites) // 20 - 1)],
            "p95_composite": sorted(composites)[min(len(composites) - 1,
                                                    len(composites) * 19 // 20)],
            "mean_outcome": sum(outcomes) / len(outcomes),
            "mean_grounding": sum(groundings) / len(groundings),
            "mean_cross_file_caller_recall": sum(cross_file) / len(cross_file),
            "mean_invariant_coverage": sum(invariants) / len(invariants),
            "mean_format_compliance": sum(fmts) / len(fmts),
            "mean_wall_clock_s": sum(r["wall_clock_s"] for r in records) / len(records),
            "rollouts_nonzero": sum(1 for r in records if r["reward"] > 0),
            "rollouts_zero": sum(1 for r in records if r["reward"] <= 0),
            "wall_clock_s_total": elapsed,
        }
        # Group-variance (per task)
        gvar = []
        for t in tasks:
            rolls = [r for r in records if r["task_id"] == t["task_id"]]
            if len(rolls) < 2:
                continue
            mean = sum(r["reward"] for r in rolls) / len(rolls)
            var = sum((r["reward"] - mean) ** 2 for r in rolls) / len(rolls)
            gvar.append(var)
        if gvar:
            summary["mean_within_group_variance"] = sum(gvar) / len(gvar)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
