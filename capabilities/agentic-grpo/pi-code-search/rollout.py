"""Rollout runner for pi-code-search.

Spawns pi sessions in isolated workdirs, scores each rollout via
rubric.score_rollout, and emits:

- per-rollout records → <out_dir>/rollouts.jsonl
- summary stats → <out_dir>/summary.json
- (if --mode train) GRPO groups → <out_dir>/grpo-train.jsonl

Heavily reuses the pi-doctest rollout shape; the only meaningful diffs
are (a) the rubric, (b) the workdir scaffold, (c) the prompt source.
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

# Local + shared imports.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import rubric  # noqa: E402
import task_scaffold  # noqa: E402
import pi_trajectory  # noqa: E402


PI_BIN = os.environ.get("PI_BIN", "/usr/bin/pi")


def kiln_active_adapter(url: str, adapter: str | None) -> None:
    """Set kiln's active adapter via POST /v1/adapters/{load,unload}."""
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
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()


def latest_session_jsonl(session_dir: Path) -> Path | None:
    candidates = list(session_dir.rglob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_transcript(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    out: list[dict] = []
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
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True,
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
        "stdout_tail": (proc.stdout or "")[-300:],
        "stderr_tail": (proc.stderr or "")[-300:],
    }
    if verbose:
        print(
            f"  task={task['task_id']} gen={gen_idx} "
            f"r={score['composite']:.2f} "
            f"out={score['outcome']:.2f} "
            f"eff={score['efficiency']:.2f} "
            f"tc={score['tool_choice']:.2f} "
            f"gr={score['grounding']:.2f} "
            f"calls={score['_n_tool_calls']} "
            f"bytes={score['_bytes_consumed']} "
            f"wall={elapsed:.1f}s",
            flush=True,
        )
    return record


def build_grpo_groups(tasks, records):
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
             "content": "You are a code-search assistant. You have access "
                        "to bash, read, glob, and edit tools. Search the "
                        "repository under `repo/`. Prefer grep, rg, glob, "
                        "and find over Read. When confident, emit a final "
                        "assistant message with the answer in `file:line` "
                        "shape (one per line, no prose)."},
            {"role": "user", "content": task_scaffold.pi_prompt(task)},
        ]
        completions = []
        for r in rolls:
            transcript_path = r.get("transcript_path")
            if transcript_path:
                rollout_dict = pi_trajectory.build_scored_rollout(
                    Path(transcript_path),
                    reward=r["reward"],
                )
                if not rollout_dict.get("trajectory"):
                    rollout_dict["text"] = "(empty)"
            else:
                rollout_dict = {"text": "(empty)", "reward": r["reward"],
                                "trajectory": []}
            completions.append(rollout_dict)
        groups.append({"messages": messages, "completions": completions})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter", default="",
                    help="If 'current', leave whatever adapter is active. "
                         "Otherwise switch to this name (or unload if '').")
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--kiln-url",
                    default=os.environ.get("KILN_URL", "http://localhost:8420"))
    ap.add_argument("--max-wall-clock-s", type=int, default=120)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="If set, shuffle the task list with this seed.")
    ap.add_argument("--verbose", action="store_true")
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

    if args.shuffle_seed is not None:
        import random as _r
        _r.Random(args.shuffle_seed).shuffle(tasks)

    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"loaded {len(tasks)} tasks; num_generations={args.num_generations} "
          f"adapter={args.adapter or '(base)'} mode={args.mode}", flush=True)

    if args.adapter == "current":
        print("using currently-loaded kiln adapter (no switch)", flush=True)
    else:
        try:
            kiln_active_adapter(args.kiln_url, args.adapter or None)
            print(f"set kiln active adapter to {args.adapter or '(base)'}",
                  flush=True)
        except Exception as e:
            print(f"WARN: failed to switch adapter ({e}); proceeding",
                  flush=True)

    started = time.time()
    records: list[dict] = []
    work_items = [
        (task, gen) for task in tasks for gen in range(args.num_generations)
    ]
    if args.parallel <= 1:
        for task, gen in work_items:
            records.append(run_one_rollout(
                task, gen, out_root, args.kiln_url,
                args.max_wall_clock_s, verbose=args.verbose,
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
            f.write(json.dumps(r, default=str) + "\n")

    if args.mode == "train":
        groups = build_grpo_groups(tasks, records)
        grp_path = out_root / "grpo-train.jsonl"
        with grp_path.open("w") as f:
            for g in groups:
                f.write(json.dumps(g) + "\n")
        print(f"wrote {len(groups)} GRPO groups → {grp_path}", flush=True)

    if records:
        composites = [r["reward"] for r in records]
        outcomes = [r["sub_scores"]["outcome"] for r in records]
        efficiencies = [r["sub_scores"]["efficiency"] for r in records]
        tcs = [r["sub_scores"]["tool_choice"] for r in records]
        grds = [r["sub_scores"]["grounding"] for r in records]
        fmts = [r["sub_scores"]["format_compliance"] for r in records]
        n_calls = [r["diagnostics"].get("_n_tool_calls", 0) for r in records]
        bytes_consumed = [r["diagnostics"].get("_bytes_consumed", 0)
                          for r in records]
        n_large_reads = [r["diagnostics"].get("_n_large_reads", 0)
                         for r in records]
        n_reads = [r["diagnostics"].get("_n_reads", 0) for r in records]
        summary = {
            "mode": args.mode,
            "adapter": args.adapter or "",
            "n_tasks": len(tasks),
            "n_rollouts": len(records),
            "n_generations": args.num_generations,
            "mean_composite": sum(composites) / len(composites),
            "mean_outcome": sum(outcomes) / len(outcomes),
            "mean_efficiency": sum(efficiencies) / len(efficiencies),
            "mean_tool_choice": sum(tcs) / len(tcs),
            "mean_grounding": sum(grds) / len(grds),
            "mean_format": sum(fmts) / len(fmts),
            "mean_wall_clock_s": sum(r["wall_clock_s"] for r in records) / len(records),
            "mean_n_tool_calls": sum(n_calls) / len(n_calls),
            "mean_bytes_consumed": sum(bytes_consumed) / len(bytes_consumed),
            "mean_n_large_reads": sum(n_large_reads) / len(n_large_reads),
            "mean_n_reads": sum(n_reads) / len(n_reads),
            "rollouts_outcome_pass": sum(1 for o in outcomes if o > 0.5),
            "rollouts_zero": sum(1 for r in composites if r <= 0),
            "wall_clock_s_total": elapsed,
        }
        # Group-variance per task.
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
