"""Rollout runner for pi-terminal-bench-lite. Spawns pi sessions per
task, scores them, emits AgenticGroup-shaped JSONL or eval summary.

Usage:
  rollout.py --tasks <tasks.jsonl> --config <capability.config.json>
             --out-dir <dir> [--adapter NAME] [--mode train|eval]
             [--num-generations 4]

Train mode writes <out_dir>/grpo-train.jsonl with rollouts in the
canonical `{messages, completions: [{text, reward, trajectory}, ...]}`
shape (matching kiln-train::ScoredRollout). Eval mode writes
<out_dir>/eval.json with the mean composite + sub-scores.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import pi_trajectory  # noqa: E402
import rubric  # noqa: E402
import task_scaffold  # noqa: E402


def kiln_active_adapter(url: str, adapter: str | None) -> None:
    if not adapter:
        req = urllib.request.Request(
            f"{url}/v1/adapters/unload",
            data=b"",
            method="POST",
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
        f"{url}/v1/adapters/load",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def run_one_rollout(
    task: dict,
    gen_idx: int,
    out_root: Path,
    config: dict,
    seed: int | None,
) -> dict:
    """One pi session for one (task, gen). Returns per-rollout record."""
    task_id = task["task_id"]
    task_dir = out_root / task_id / f"gen{gen_idx:02d}"
    workdir = task_dir / "workdir"
    sessions_dir = task_dir / "sessions"
    task_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    task_scaffold.init_workdir(task, workdir)
    prompt = task_scaffold.pi_prompt(task)

    pi_bin = os.environ.get("PI_BIN", config["pi_bin"])
    max_wall_clock_s = config["rollout"]["max_wall_clock_s"]
    pi_model_id = config["pi_model_id"]

    pi_args = [
        pi_bin,
        "-p",
        prompt,
        "--session-dir",
        str(sessions_dir),
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-themes",
        "--offline",
        "--model",
        pi_model_id,
    ]
    if seed is not None:
        pi_args.extend(["--seed", str(seed + gen_idx)])

    started = time.time()
    try:
        result = subprocess.run(
            ["timeout", str(max_wall_clock_s), *pi_args],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=max_wall_clock_s + 30,
        )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    elapsed = time.time() - started

    # Find the most-recent session JSONL pi wrote.
    candidates = list(sessions_dir.rglob("*.jsonl"))
    transcript_path = (
        str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else ""
    )

    # Materialize hidden verifier-only files now (before scoring).
    task_scaffold.post_rollout_setup(task, workdir)

    scores = rubric.score_rollout(transcript_path, str(workdir), task)

    return {
        "task_id": task_id,
        "gen": gen_idx,
        "reward": scores["composite"],
        "sub_scores": scores["sub_scores"],
        "wall_clock_s": elapsed,
        "exit_code": exit_code,
        "transcript_path": transcript_path,
        "workdir": str(workdir),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--adapter", default="")
    p.add_argument("--mode", choices=["train", "eval"], default="eval")
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    with Path(args.tasks).open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    if args.mode == "eval":
        num_gens = args.num_generations or config["rollout"]["num_generations_eval"]
    else:
        num_gens = args.num_generations or config["rollout"]["num_generations_train"]

    if args.adapter is not None:
        kiln_active_adapter(config["kiln_url"], args.adapter or None)

    records: list[dict] = []
    parallel = config["rollout"].get("parallel", 1)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = []
        for task in tasks:
            for gen_idx in range(num_gens):
                futures.append(
                    pool.submit(run_one_rollout, task, gen_idx, out_root, config, args.seed)
                )
        for fut in futures:
            r = fut.result()
            records.append(r)
            if args.verbose:
                print(
                    f"  {r['task_id']} gen{r['gen']:02d} reward={r['reward']:.3f}"
                    f" wall={r['wall_clock_s']:.1f}s exit={r['exit_code']}",
                    flush=True,
                )

    # Per-rollout artifact.
    (out_root / "rollouts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    if args.mode == "train":
        # Emit one AgenticGroup per task: messages + rollouts (with trajectory).
        by_task: dict[str, list[dict]] = {}
        for r in records:
            by_task.setdefault(r["task_id"], []).append(r)
        groups = []
        for task in tasks:
            task_id = task["task_id"]
            rolls = by_task.get(task_id, [])
            if len(rolls) < 2:
                continue
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Python coding assistant. You have access to "
                        "bash, write, read, and edit tools. Solve the user's "
                        "task. When complete, emit a final assistant message "
                        "with no tool calls."
                    ),
                },
                {"role": "user", "content": task_scaffold.pi_prompt(task)},
            ]
            completions = []
            for r in rolls:
                if r.get("transcript_path"):
                    rollout = pi_trajectory.build_scored_rollout(
                        Path(r["transcript_path"]),
                        reward=r["reward"],
                    )
                    if not rollout.get("trajectory"):
                        rollout["text"] = "(empty)"
                else:
                    rollout = {"text": "(empty)", "reward": r["reward"], "trajectory": []}
                completions.append(rollout)
            groups.append({"messages": messages, "completions": completions})
        (out_root / "grpo-train.jsonl").write_text(
            "\n".join(json.dumps(g) for g in groups) + "\n"
        )
        print(f"wrote {len(groups)} AgenticGroups → {out_root}/grpo-train.jsonl", flush=True)

    # Always emit eval summary.
    if records:
        composites = [r["reward"] for r in records]
        sub_scores = {
            "outcome": sum(r["sub_scores"]["outcome"] for r in records) / len(records),
            "tool_call_efficiency": sum(r["sub_scores"]["tool_call_efficiency"] for r in records) / len(records),
            "format_compliance": sum(r["sub_scores"]["format_compliance"] for r in records) / len(records),
            "no_loop": sum(r["sub_scores"]["no_loop"] for r in records) / len(records),
        }
        summary = {
            "mode": args.mode,
            "adapter": args.adapter,
            "n_tasks": len(tasks),
            "n_rollouts": len(records),
            "n_generations": num_gens,
            "mean_composite": sum(composites) / len(composites),
            "sub_scores": sub_scores,
            "p50_composite": sorted(composites)[len(composites) // 2],
            "p05_composite": sorted(composites)[max(0, len(composites) // 20 - 1)],
            "p95_composite": sorted(composites)[min(len(composites) - 1, len(composites) * 19 // 20)],
        }
        (out_root / "eval.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote eval summary → {out_root}/eval.json", flush=True)
        print(f"mean_composite={summary['mean_composite']:.4f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
