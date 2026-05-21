"""Pi-diff-patch-apply rollout harness.

Spawns one pi session per (task, generation) in an isolated workdir,
scores it via the multi-component rubric, emits a GRPO group JSONL (train
mode) or eval summary (eval mode).

Usage
-----
    python3 rollout.py \\
        --tasks datasets/train.tasks.jsonl \\
        --out-dir /tmp/iter0-rollouts \\
        --mode train \\
        --num-generations 4 \\
        --kiln-url http://localhost:8420 \\
        --parallel 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "lib"))

import rubric  # noqa: E402
import task_scaffold  # noqa: E402
import pi_trajectory  # noqa: E402


PI_BIN = os.environ.get("PI_BIN", "/usr/bin/pi")


def kiln_active_adapter(url: str, adapter: str | None) -> None:
    """POST /v1/adapters/(load|unload). Empty/None = unload (base model)."""
    if not adapter:
        req = urllib.request.Request(
            f"{url}/v1/adapters/unload",
            data=b"", method="POST",
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
        data=body, method="POST",
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


def run_one_rollout(
    task: dict,
    gen_idx: int,
    out_root: Path,
    kiln_url: str,
    max_wall_clock_s: int,
    max_turns: int,
    temperature: float,
    seed: int | None,
    verbose: bool = False,
) -> dict:
    """Run a single pi session for one (task, gen)."""
    task_dir = out_root / task["task_id"] / f"gen{gen_idx:02d}"
    workdir = task_dir / "workdir"
    session_dir = task_dir / "sessions"
    task_dir.mkdir(parents=True, exist_ok=True)
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
    # NB: pi 0.75.x does not expose a `--max-turns` flag. Turn budget is
    # enforced indirectly via the wall-clock timeout. We carry max_turns for
    # diagnostics only.
    _ = max_turns
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started

    transcript_path = latest_session_jsonl(session_dir)
    transcript = parse_transcript(transcript_path)
    score = rubric.score_rollout(transcript, str(workdir), task)

    record = {
        "task_id": task["task_id"],
        "patch_class": task.get("patch_class"),
        "gen": gen_idx,
        "reward": score["composite"],
        "sub_scores": {k: v for k, v in score.items() if not k.startswith("_")},
        "diagnostics": {k: v for k, v in score.items() if k.startswith("_")},
        "wall_clock_s": elapsed,
        "exit_code": proc.returncode,
        "transcript_path": str(transcript_path) if transcript_path else None,
        "workdir": str(workdir),
        "stdout_tail": (proc.stdout or "")[-400:],
        "stderr_tail": (proc.stderr or "")[-400:],
    }
    if verbose:
        print(
            f"  task={task['task_id']} gen={gen_idx} class={task.get('patch_class')} "
            f"reward={score['composite']:.3f} "
            f"outcome={score['outcome']:.0f} "
            f"applied={score.get('applied_fraction', 0):.2f} "
            f"min={score['minimality']:.2f} "
            f"nourel={score['no_unrelated_edits']:.2f} "
            f"repair={score['repair_efficiency']:.2f} "
            f"tcalls={score.get('_n_tool_calls', 0)} "
            f"wall={elapsed:.1f}s",
            flush=True,
        )
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter", default="",
                    help="Adapter to switch to before rolling out. "
                         "'current' = leave whatever kiln has loaded.")
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--kiln-url", default=os.environ.get("KILN_URL", "http://localhost:8420"))
    ap.add_argument("--max-wall-clock-s", type=int, default=240)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--task-limit", type=int, default=0)
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if args.task_offset:
        tasks = tasks[args.task_offset:]
    if args.task_limit:
        tasks = tasks[:args.task_limit]

    print(f"loaded {len(tasks)} tasks; num_generations={args.num_generations} "
          f"adapter={args.adapter or '(base)'} mode={args.mode}", flush=True)

    # Set active adapter.
    if args.adapter == "current":
        print("using currently-loaded kiln adapter (no switch)", flush=True)
    else:
        try:
            kiln_active_adapter(args.kiln_url, args.adapter or None)
            print(f"set kiln adapter -> {args.adapter or '(base)'}", flush=True)
        except Exception as e:
            print(f"WARN: failed to switch adapter ({e}); proceeding with whatever "
                  f"kiln-server currently has active", flush=True)

    started = time.time()
    records: list[dict] = []
    work = [(t, g) for t in tasks for g in range(args.num_generations)]
    if args.parallel <= 1:
        for t, g in work:
            records.append(run_one_rollout(
                t, g, out_root, args.kiln_url,
                args.max_wall_clock_s, args.max_turns, args.temperature,
                (args.seed_base + g) if args.seed_base else None,
                verbose=args.verbose,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [
                ex.submit(
                    run_one_rollout, t, g, out_root, args.kiln_url,
                    args.max_wall_clock_s, args.max_turns, args.temperature,
                    (args.seed_base + g) if args.seed_base else None,
                    args.verbose,
                )
                for t, g in work
            ]
            for f in as_completed(futs):
                records.append(f.result())
    elapsed = time.time() - started

    # Per-rollout JSONL.
    rec_path = out_root / "rollouts.jsonl"
    with rec_path.open("w") as f:
        for r in sorted(records, key=lambda r: (r["task_id"], r["gen"])):
            f.write(json.dumps(r) + "\n")

    # GRPO groups (training mode).
    if args.mode == "train":
        by_task: dict[str, list[dict]] = {}
        for r in records:
            by_task.setdefault(r["task_id"], []).append(r)
        groups = []
        for task in tasks:
            rolls = by_task.get(task["task_id"], [])
            if len(rolls) < 2:
                continue
            rolls = sorted(rolls, key=lambda r: r["gen"])
            messages = task_scaffold.build_messages(task)
            completions = []
            for r in rolls:
                trans_path = r.get("transcript_path")
                if trans_path:
                    sr = pi_trajectory.build_scored_rollout(
                        Path(trans_path), reward=r["reward"],
                    )
                    if not sr.get("trajectory"):
                        sr["text"] = "(empty)"
                else:
                    sr = {"text": "(empty)", "reward": r["reward"], "trajectory": []}
                completions.append(sr)
            groups.append({"messages": messages, "completions": completions, "task_id": task["task_id"]})

        grp_path = out_root / "grpo-train.jsonl"
        with grp_path.open("w") as f:
            for g in groups:
                f.write(json.dumps(g) + "\n")
        print(f"wrote {len(groups)} GRPO groups -> {grp_path}", flush=True)

    # Summary.
    composites = [r["reward"] for r in records]
    if not composites:
        print("no rollouts produced", file=sys.stderr)
        sys.exit(2)

    def pct(xs, q):
        s = sorted(xs)
        idx = min(len(s) - 1, int(q / 100 * len(s)))
        return s[idx]

    by_class: dict[str, list[float]] = {}
    for r in records:
        by_class.setdefault(r.get("patch_class") or "?", []).append(r["reward"])
    class_means = {k: sum(v) / len(v) for k, v in by_class.items()}

    # Group-variance (within-task std across generations).
    gvar_by_task: dict[str, float] = {}
    for tid, rolls in {}.items():
        pass
    gvar = []
    for t in tasks:
        rolls = [r for r in records if r["task_id"] == t["task_id"]]
        if len(rolls) < 2:
            continue
        mean = sum(r["reward"] for r in rolls) / len(rolls)
        var = sum((r["reward"] - mean) ** 2 for r in rolls) / len(rolls)
        gvar.append(var)

    # Sub-score means.
    sub_means: dict[str, float] = {}
    sub_keys = set()
    for r in records:
        for k in (r.get("sub_scores") or {}):
            sub_keys.add(k)
    for k in sub_keys:
        vals = [r["sub_scores"].get(k, 0.0) for r in records]
        sub_means[k] = sum(vals) / len(vals)

    summary = {
        "mode": args.mode,
        "adapter": args.adapter,
        "n_tasks": len(tasks),
        "n_rollouts": len(records),
        "n_generations": args.num_generations,
        "mean_composite": sum(composites) / len(composites),
        "p50_composite": pct(composites, 50),
        "p25_composite": pct(composites, 25),
        "p75_composite": pct(composites, 75),
        "rollouts_nonzero": sum(1 for r in records if r["reward"] > 0),
        "rollouts_zero": sum(1 for r in records if r["reward"] <= 0),
        "rollouts_passed": sum(1 for r in records if r["sub_scores"].get("outcome", 0) >= 1.0),
        "mean_wall_clock_s": sum(r["wall_clock_s"] for r in records) / len(records),
        "wall_clock_s_total": elapsed,
        "mean_within_group_variance": (sum(gvar) / len(gvar)) if gvar else 0.0,
        "class_means": class_means,
        "sub_score_means": sub_means,
    }
    sum_path = out_root / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote summary -> {sum_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
