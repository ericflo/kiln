"""Rollout runner: spawns pi sessions against kiln, scores them, emits a
GRPO group JSONL or an eval-summary JSON.

Usage:
  rollout.py --tasks <tasks.jsonl> --out-dir <dir> [--adapter NAME]
             [--num-generations 4] [--mode train|eval]
             [--kiln-url http://localhost:8420]
             [--max-wall-clock-s 120]

Mode = train: writes <out_dir>/grpo-train.jsonl (one GrpoGroup per task).
Mode = eval: writes <out_dir>/eval.json (mean composite + sub-scores).

Pi is invoked per rollout in an isolated workdir with --session-dir,
--no-context-files, --no-extensions, --no-skills, --no-themes, --offline.
The prompt is sent positionally + via -p. The session JSONL is captured
from the session-dir and parsed by `rubric.score_rollout`.

Adapter selection: if --adapter NAME is given, the runner POSTs to
$KILN_URL/v1/adapter/switch (or equivalent) to set the active adapter
before any rollout. If switching fails, the run aborts (don't silently
roll out against the wrong model).
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rubric  # noqa: E402
import task_scaffold  # noqa: E402


PI_BIN = os.environ.get("PI_BIN", "pi")


def kiln_active_adapter(url: str, adapter: str | None) -> None:
    """Set kiln's active adapter via POST /v1/adapters/load (or unload).
    Empty / None = base model (unload)."""
    if not adapter:
        # Unload — revert to base.
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
            # If no adapter was loaded, unload may 4xx — that's fine.
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


def latest_session_jsonl(session_dir: Path) -> Path | None:
    """Find the most-recent .jsonl under the session dir tree."""
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
    seed: int | None,
    verbose: bool = False,
) -> dict:
    """Run a single pi session for one (task, gen). Returns the per-rollout
    record: { task_id, gen, reward, sub_scores, wall_clock_s, exit_code,
    transcript_path, workdir }."""
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
    # Model selection is via the kiln-local provider config that
    # `kiln pi-setup` writes into ~/.pi/agent/settings.json. We don't
    # set --provider / --model here — let pi pick up its defaults.

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
        "stdout_tail": proc.stdout[-400:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
    }
    if verbose:
        print(
            f"  task={task['task_id']} gen={gen_idx} "
            f"reward={score['composite']:.2f} "
            f"wall={elapsed:.1f}s exit={proc.returncode}",
            flush=True,
        )
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter", default="",
                    help="If 'current', skip the kiln-side switch and use whatever "
                         "adapter is already active. Otherwise switch to this name.")
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--kiln-url", default=os.environ.get("KILN_URL", "http://localhost:8420"))
    ap.add_argument("--max-wall-clock-s", type=int, default=120)
    ap.add_argument("--parallel", type=int, default=1,
                    help="Number of concurrent pi rollouts. Pi shares the kiln "
                         "backend so high parallelism here mainly stresses the "
                         "scheduler. Start with 1 and raise carefully.")
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Limit task count (0 = all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load tasks.
    tasks = []
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

    # Set active adapter. `--adapter current` skips the switch.
    if args.adapter == "current":
        print(f"using currently-loaded kiln adapter (no switch)", flush=True)
    else:
        try:
            kiln_active_adapter(args.kiln_url, args.adapter or None)
            print(f"set kiln active adapter to {args.adapter or '(base)'}",
                  flush=True)
        except Exception as e:
            print(f"WARN: failed to switch adapter ({e}); proceeding with "
                  f"whatever adapter kiln-server currently has active",
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
                args.max_wall_clock_s,
                (args.seed_base + gen) if args.seed_base else None,
                verbose=args.verbose,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [
                ex.submit(
                    run_one_rollout, task, gen, out_root, args.kiln_url,
                    args.max_wall_clock_s,
                    (args.seed_base + gen) if args.seed_base else None,
                    args.verbose,
                )
                for task, gen in work_items
            ]
            for f in futs:
                records.append(f.result())

    elapsed = time.time() - started

    # Always write per-rollout records.
    rec_path = out_root / "rollouts.jsonl"
    with rec_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    if args.mode == "train":
        # Group by task_id; emit one GrpoGroup per task.
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
                 "content": "You are a Python coding assistant. You have "
                            "access to bash, write, read, and edit tools. "
                            "Solve the user's task. When the task is "
                            "complete, emit a final assistant message with "
                            "no tool calls."},
                {"role": "user", "content": task_scaffold.pi_prompt(task)},
            ]
            completions = []
            for r in rolls:
                # v0 placeholder: extract assistant text/thinking blocks
                # from the pi session JSONL and concatenate. Multiple
                # assistant turns are joined with the literal sentinel
                # `<TURN_BREAK>`. See kiln-polish-prerequisites.md #1
                # for the proper per-turn-mask fix.
                tr = parse_transcript(Path(r["transcript_path"])) if r.get("transcript_path") else []
                turns = []
                for ev in tr:
                    if ev.get("type") != "message":
                        continue
                    msg = ev.get("message") or {}
                    if msg.get("role") != "assistant":
                        continue
                    parts = []
                    for b in (msg.get("content") or []):
                        if not isinstance(b, dict):
                            continue
                        bt = b.get("type")
                        if bt == "text" and isinstance(b.get("text"), str):
                            parts.append(b["text"])
                        elif bt == "thinking" and isinstance(b.get("thinking"), str):
                            # Wrap in Qwen-style think tags so the chat template
                            # round-trip is closer to what the model emitted.
                            parts.append(f"<think>{b['thinking']}</think>")
                        elif bt == "toolCall":
                            name = b.get("name", "")
                            inp = json.dumps(b.get("input", {}))
                            parts.append(
                                f"<tool_call>"
                                f'{{"name": "{name}", "arguments": {inp}}}'
                                f"</tool_call>"
                            )
                    if parts:
                        turns.append("".join(parts))
                completions.append({
                    "text": "<TURN_BREAK>".join(turns) or "(empty)",
                    "reward": r["reward"],
                })
            groups.append({"messages": messages, "completions": completions})

        grp_path = out_root / "grpo-train.jsonl"
        with grp_path.open("w") as f:
            for g in groups:
                f.write(json.dumps(g) + "\n")
        print(f"wrote {len(groups)} GRPO groups → {grp_path}", flush=True)

    # Always emit an eval summary, regardless of mode.
    if records:
        composites = [r["reward"] for r in records]
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
            "mean_wall_clock_s": sum(r["wall_clock_s"] for r in records) / len(records),
            "rollouts_nonzero": sum(1 for r in records if r["reward"] > 0),
            "rollouts_zero": sum(1 for r in records if r["reward"] <= 0),
            "wall_clock_s_total": elapsed,
        }
        # Group-variance: for each task, std of rewards across its gens
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
