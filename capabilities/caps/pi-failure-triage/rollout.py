"""Rollout runner for pi-failure-triage.

For each (task, generation_index) pair:
  1. Initialize a fresh sandbox dir.
  2. task_scaffold.init_workdir(task, dir).
  3. Invoke pi headless against kiln, capturing the session.
  4. Score with rubric.score_rollout(transcript, dir, task).
  5. Emit a per-rollout record.

Modes:
  --mode train: emit grpo-train.jsonl (one group per task).
  --mode eval:  emit summary.json (eval stats).

Always writes rollouts.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))

import rubric  # noqa: E402
import task_scaffold  # noqa: E402
import pi_trajectory  # noqa: E402


PI_BIN = os.environ.get("PI_BIN", "pi")


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
        except Exception:
            return
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


def latest_session_jsonl(session_dir: Path) -> Path | None:
    candidates = list(session_dir.rglob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_transcript(path: Path) -> list[dict]:
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


def run_one_rollout(
    task: dict,
    gen_idx: int,
    out_root: Path,
    kiln_url: str,
    max_wall_clock_s: int,
    verbose: bool = False,
) -> dict:
    task_dir = out_root / task["task_id"] / f"gen{gen_idx:02d}"
    workdir = task_dir / "workdir"
    session_dir = task_dir / "sessions"
    workdir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    task_scaffold.init_workdir(task, str(workdir))

    prompt = task_scaffold.pi_prompt(task)

    cmd = [
        "timeout", f"{max_wall_clock_s}s",
        PI_BIN,
        "-p", prompt,
        "--session-dir", str(session_dir),
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-themes",
        "--offline",
    ]

    started = time.time()
    proc = subprocess.run(
        cmd, cwd=str(workdir), capture_output=True, text=True,
    )
    elapsed = time.time() - started

    transcript_path = latest_session_jsonl(session_dir)
    transcript = parse_transcript(transcript_path) if transcript_path else []
    score = rubric.score_rollout(transcript, str(workdir), task)

    record = {
        "task_id": task["task_id"],
        "gen": gen_idx,
        "reward": score["composite"],
        "sub_scores": {k: v for k, v in score.items() if not k.startswith("_")},
        "diagnostics": {k: v for k, v in score.items() if k.startswith("_")},
        "wall_clock_s": elapsed,
        "exit_code": proc.returncode,
        "transcript_path": str(transcript_path) if transcript_path else None,
        "workdir": str(workdir),
        "stdout_tail": proc.stdout[-400:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
    }
    if verbose:
        sub = score
        print(
            f"  task={task['task_id']} gen={gen_idx} reward={score['composite']:.2f} "
            f"outcome={sub['outcome']:.1f} held_out={sub['held_out_passes']:.1f} "
            f"wall={elapsed:.1f}s exit={proc.returncode}",
            flush=True,
        )
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter", default="",
                    help="empty=base, 'current'=skip switch, else load named adapter")
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--kiln-url", default=os.environ.get("KILN_URL", "http://localhost:8420"))
    ap.add_argument("--max-wall-clock-s", type=int, default=120)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--task-limit", type=int, default=0, help="alias for --limit")
    args = ap.parse_args()

    if args.task_limit and not args.limit:
        args.limit = args.task_limit

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"loaded {len(tasks)} tasks; num_generations={args.num_generations} "
          f"adapter={args.adapter or '(base)'} mode={args.mode}", flush=True)

    if args.adapter == "current":
        print("using currently-loaded kiln adapter (no switch)", flush=True)
    else:
        try:
            kiln_active_adapter(args.kiln_url, args.adapter or None)
            print(f"set kiln active adapter to {args.adapter or '(base)'}", flush=True)
        except Exception as e:
            print(f"WARN: adapter switch failed ({e}); proceeding", flush=True)

    started = time.time()
    records: list[dict] = []
    work_items = [(task, gen) for task in tasks for gen in range(args.num_generations)]
    if args.parallel <= 1:
        for task, gen in work_items:
            records.append(run_one_rollout(
                task, gen, out_root, args.kiln_url, args.max_wall_clock_s,
                verbose=args.verbose,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [
                ex.submit(
                    run_one_rollout, task, gen, out_root, args.kiln_url,
                    args.max_wall_clock_s, args.verbose,
                )
                for task, gen in work_items
            ]
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
        groups = []
        for task in tasks:
            task_id = task["task_id"]
            rolls = by_task.get(task_id, [])
            if len(rolls) < 2:
                continue
            messages = [
                {"role": "system",
                 "content": "You are a Python debugging assistant. You have "
                            "access to bash, write, read, and edit tools. "
                            "Diagnose the failing test, find the root cause, "
                            "and fix it. End with `Fix: <file>::<func>: <line>`."},
                {"role": "user", "content": task_scaffold.pi_prompt(task)},
            ]
            completions = []
            for r in rolls:
                tp = r.get("transcript_path")
                if tp:
                    rollout_dict = pi_trajectory.build_scored_rollout(
                        Path(tp), reward=r["reward"],
                    )
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
        sub_keys = list(records[0]["sub_scores"].keys())
        summary: dict = {
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
            "mean_wall_clock_s": sum(r["wall_clock_s"] for r in records) / len(records),
            "rollouts_nonzero": sum(1 for r in records if r["reward"] > 0),
            "rollouts_zero": sum(1 for r in records if r["reward"] <= 0),
            "wall_clock_s_total": elapsed,
        }
        for k in sub_keys:
            vals = [r["sub_scores"][k] for r in records]
            summary[f"mean_{k}"] = sum(vals) / len(vals)
        gvar = []
        for task in tasks:
            rolls = [r for r in records if r["task_id"] == task["task_id"]]
            if len(rolls) < 2:
                continue
            mean = sum(r["reward"] for r in rolls) / len(rolls)
            var = sum((r["reward"] - mean) ** 2 for r in rolls) / len(rolls)
            gvar.append(var)
        if gvar:
            summary["mean_within_group_variance"] = sum(gvar) / len(gvar)
        sum_path = out_root / "summary.json"
        sum_path.write_text(json.dumps(summary, indent=2))
        print(f"wrote summary → {sum_path}", flush=True)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
