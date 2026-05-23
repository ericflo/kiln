"""Pi rollout driver for pi-error-recovery.

For each task in `--tasks`, materializes the task's `init_files` into a
fresh sandbox, sets `init_chmod` if present, and runs `pi -p <prompt>`
with the kiln-served base model (or named adapter). On completion,
reads the pi session JSONL, derives `outcome_passed` from the task's
`gold_state` (file equality) or `gold_state_predicate` (e.g. tests
passing via `verify_cmd`), and emits:

  - rollout.jsonl : per-rollout dict {task, transcript, workdir, outcome_passed, format_text, score}
  - grpo-train.jsonl : trainer-shaped ScoredRollout JSONL (kiln canonical schema)
  - summary.json : aggregate stats

Reference: ../pi-doctest/rollout.py (the most-mature pi driver). This
driver follows the same shape with this cap's task spec.
"""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Make the shared lib importable for canonical pi → trajectory rendering.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lib"))
try:
    from pi_trajectory import session_to_trajectory  # type: ignore
except ImportError:
    session_to_trajectory = None  # graceful fallback; trainer-shape only

# Import this cap's rubric.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rubric  # noqa: E402


def _materialize_sandbox(task: dict, sandbox: Path) -> None:
    """Write each init_file into sandbox; apply init_chmod permissions."""
    for rel, content in (task.get("init_files") or {}).items():
        p = sandbox / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    for rel, mode_str in (task.get("init_chmod") or {}).items():
        p = sandbox / rel
        if p.exists():
            mode = int(mode_str, 8) if mode_str.startswith("0o") else int(mode_str)
            p.chmod(mode)


def _gold_state_matches(task: dict, sandbox: Path) -> bool:
    gs = task.get("gold_state")
    if isinstance(gs, dict):
        for rel, expected in gs.items():
            p = sandbox / rel
            if not p.exists():
                return False
            try:
                if p.read_text() != expected:
                    return False
            except (UnicodeDecodeError, OSError):
                return False
        return True

    verify_cmd = task.get("verify_cmd")
    if verify_cmd:
        try:
            r = subprocess.run(
                ["bash", "-c", verify_cmd],
                cwd=str(sandbox),
                capture_output=True,
                text=True,
                timeout=60,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    pred = task.get("gold_state_predicate")
    if pred == "tests pass":
        try:
            r = subprocess.run(
                ["bash", "-c", "python3 -m pytest -q"],
                cwd=str(sandbox),
                capture_output=True,
                text=True,
                timeout=60,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    return False


def _load_session_jsonl(session_dir: Path) -> list[dict]:
    """Locate and load the pi session JSONL file pi just wrote."""
    # Pi v0.75.x writes ~/.pi/agent/sessions/<workdir-encoded>/<uuid>.jsonl
    # If --session-dir is passed, pi writes there directly.
    candidates = sorted(session_dir.glob("*.jsonl"))
    if not candidates:
        # Fall back to a recursive scan.
        candidates = sorted(session_dir.rglob("*.jsonl"))
    if not candidates:
        return []
    # Take the newest.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [json.loads(line) for line in candidates[0].read_text().splitlines() if line.strip()]


def _final_assistant_text(transcript: list) -> str:
    final = ""
    for ev in transcript:
        if not isinstance(ev, dict) or ev.get("type") != "message":
            continue
        msg = ev.get("message")
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content") or []
            text = ""
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text")
                    if isinstance(t, str):
                        text += t
            if text:
                final = text
    return final


def _trajectory_diagnostics(transcript: list) -> dict:
    n_tool_calls = 0
    thinking_blocks = 0
    thinking_chars = 0
    for ev in transcript or []:
        if not isinstance(ev, dict) or ev.get("type") != "message":
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "toolCall":
                n_tool_calls += 1
            elif block.get("type") == "thinking":
                thinking_blocks += 1
                thinking_chars += len(block.get("thinking") or "")
    return {
        "_n_tool_calls": n_tool_calls,
        "_thinking_blocks": thinking_blocks,
        "_thinking_chars": thinking_chars,
        "_thinking_chars_per_tool_call": (
            thinking_chars / n_tool_calls if n_tool_calls else thinking_chars
        ),
    }


def _run_pi_one(task: dict, cfg: dict, adapter: str, sandbox_root: Path, mode: str) -> dict:
    sb = sandbox_root / task["task_id"]
    if sb.exists():
        import shutil
        shutil.rmtree(sb)
    sb.mkdir(parents=True)
    _materialize_sandbox(task, sb)

    session_dir = sb / ".pi_session"
    session_dir.mkdir()

    pi_bin = cfg.get("pi_bin", "/usr/bin/pi")
    pi_model_id = cfg.get("pi_model_id", "Qwen3.5-4B")
    rollout_cfg = cfg.get("rollout", {})
    max_wall = int(rollout_cfg.get("max_wall_clock_s", 120))

    prompt = task["prompt"]

    env = os.environ.copy()
    if adapter:
        env["KILN_ADAPTER"] = adapter  # pi-kiln proxy honors this header

    cmd = [
        pi_bin,
        "-p", prompt,
        "--session-dir", str(session_dir),
        "--model", pi_model_id,
    ]
    t0 = time.time()
    try:
        subprocess.run(
            cmd,
            cwd=str(sb),
            env=env,
            timeout=max_wall,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pass
    wall = time.time() - t0

    transcript = _load_session_jsonl(session_dir)
    outcome_passed = _gold_state_matches(task, sb)
    fmt_text = _final_assistant_text(transcript)

    rollout = {
        "task": task,
        "transcript": transcript,
        "workdir": str(sb),
        "outcome_passed": outcome_passed,
        "format_text": fmt_text,
        "wall_clock_s": wall,
    }
    score = rubric.score_one(rollout)
    rollout["score"] = score
    rollout["diagnostics"] = _trajectory_diagnostics(transcript)
    return rollout


def _to_scored_rollout(rollout: dict) -> dict | None:
    """Convert to kiln ScoredRollout JSONL shape (one row of grpo-train.jsonl)."""
    if session_to_trajectory is None:
        return None
    try:
        traj = session_to_trajectory(rollout["transcript"])
    except Exception:
        return None
    return {
        "task_id": rollout["task"]["task_id"],
        "trajectory": traj,
        "reward": rollout["score"]["composite"],
        "sub_scores": {k: v for k, v in rollout["score"].items() if k != "composite"},
        "diagnostics": rollout.get("diagnostics") or {},
        "rubric_version": rubric.RUBRIC_VERSION,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="capability.config.json")
    ap.add_argument("--num-generations", type=int, default=1)
    ap.add_argument("--mode", choices=["train", "eval"], default="eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--max-wall-clock-s", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    if args.max_wall_clock_s is not None:
        cfg.setdefault("rollout", {})["max_wall_clock_s"] = args.max_wall_clock_s
    sandbox_root = Path(cfg.get("sandbox_root", f"/tmp/{cfg['slug']}-rollouts"))
    sandbox_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    with open(args.tasks) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    rollouts = []
    grpo_rows = []
    n_gen = max(1, args.num_generations if args.mode == "train" else 1)
    for task in tasks:
        for g in range(n_gen):
            r = _run_pi_one(task, cfg, args.adapter, sandbox_root, args.mode)
            rollouts.append(r)
            sr = _to_scored_rollout(r)
            if sr is not None:
                grpo_rows.append(sr)

    # Write rollout.jsonl (raw, debugging).
    with open(out_dir / "rollout.jsonl", "w") as f:
        for r in rollouts:
            # Compact: drop full transcript from the raw dump (kept in pi session dir).
            row = {**r}
            row["transcript_len"] = len(row.pop("transcript", []))
            f.write(json.dumps(row, default=str) + "\n")

    # Write trainer-shaped grpo-train.jsonl.
    with open(out_dir / "grpo-train.jsonl", "w") as f:
        for sr in grpo_rows:
            f.write(json.dumps(sr) + "\n")

    # Summary.
    composites = [r["score"]["composite"] for r in rollouts]
    mean = sum(composites) / max(1, len(composites))
    tool_calls = [
        float((r.get("diagnostics") or {}).get("_n_tool_calls"))
        for r in rollouts
        if isinstance((r.get("diagnostics") or {}).get("_n_tool_calls"), (int, float))
    ]
    thinking_chars = [
        float((r.get("diagnostics") or {}).get("_thinking_chars"))
        for r in rollouts
        if isinstance((r.get("diagnostics") or {}).get("_thinking_chars"), (int, float))
    ]
    thinking_blocks = [
        float((r.get("diagnostics") or {}).get("_thinking_blocks"))
        for r in rollouts
        if isinstance((r.get("diagnostics") or {}).get("_thinking_blocks"), (int, float))
    ]
    thinking_chars_per_tool_call = [
        float((r.get("diagnostics") or {}).get("_thinking_chars_per_tool_call"))
        for r in rollouts
        if isinstance(
            (r.get("diagnostics") or {}).get("_thinking_chars_per_tool_call"),
            (int, float),
        )
    ]
    summary = {
        "n_rollouts": len(rollouts),
        "n_tasks": len(tasks),
        "n_generations": n_gen,
        "adapter": args.adapter,
        "mean_composite": mean,
        "sub_scores_mean": {
            k: sum(r["score"].get(k, 0.0) for r in rollouts) / max(1, len(rollouts))
            for k in (rollouts[0]["score"] if rollouts else {})
            if k != "composite"
        },
        "mean_wall_clock_s": sum(r.get("wall_clock_s", 0.0) for r in rollouts)
        / max(1, len(rollouts)),
        "rollouts_nonzero": sum(1 for x in composites if x > 0),
        "rollouts_zero": sum(1 for x in composites if x == 0),
        "mean_tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else None,
        "mean_thinking_chars": (
            sum(thinking_chars) / len(thinking_chars) if thinking_chars else None
        ),
        "mean_thinking_blocks": (
            sum(thinking_blocks) / len(thinking_blocks) if thinking_blocks else None
        ),
        "mean_thinking_chars_per_tool_call": (
            sum(thinking_chars_per_tool_call) / len(thinking_chars_per_tool_call)
            if thinking_chars_per_tool_call
            else None
        ),
        "rubric_version": rubric.RUBRIC_VERSION,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"mean_composite={mean:.4f}  n_rollouts={len(rollouts)}")


if __name__ == "__main__":
    main()
