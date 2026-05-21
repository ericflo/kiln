"""Pi-compaction GRPO rollout harness.

Single-turn text completion: send pi's compaction prompt to kiln, score
the response with the multi-component rubric, write JSONL ready for
`cuda_grpo_ablation`.

Modes
-----
- `--mode train`: emit one GrpoGroup line per task, with N completions
  each, ready to feed kiln-train.
- `--mode eval`: single best-of-N pass (N=1 by default), emit rollouts
  + a summary.json with mean composite.

Usage
-----
    python3 rollout.py \\
        --tasks datasets/train.tasks.jsonl \\
        --out-dir /tmp/iter0-rollouts \\
        --mode train \\
        --num-generations 4 \\
        --kiln-base http://localhost:8420

The rollout uses `--adapter current` to mean "whatever the kiln server
has loaded right now." Set the adapter before invoking via
`POST /v1/adapters/load` (or `unload` for base).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rubric
import task_scaffold


def kiln_chat_completion(
    base_url: str,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int | None = None,
    adapter: str | None = None,
    timeout: int = 600,
) -> dict:
    body = {
        "model": "Qwen3.5-4B",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        body["seed"] = seed
    if adapter:
        body["adapter"] = adapter
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_response_text(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def run_rollout(
    task: dict,
    *,
    base_url: str,
    adapter: str | None,
    temperature: float,
    seed: int,
    max_tokens: int,
    gen_idx: int,
) -> dict:
    pi_messages = task_scaffold.build_rollout_messages(task["source_messages"])
    t0 = time.time()
    try:
        resp = kiln_chat_completion(
            base_url,
            pi_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            adapter=adapter,
        )
        wall = time.time() - t0
        response_text = parse_response_text(resp)
        usage = resp.get("usage", {}) or {}
        err = None
    except Exception as e:  # noqa: BLE001
        wall = time.time() - t0
        response_text = ""
        usage = {}
        err = f"{type(e).__name__}: {e}"

    score = rubric.score_rollout(response_text, task["source_text"], task["ground_truth"])
    return {
        "task_id": task["task_id"],
        "gen": gen_idx,
        "response": response_text,
        "wall_clock_s": wall,
        "error": err,
        "usage": usage,
        "reward": score["composite"],
        "sub_scores": {
            k: v for k, v in score.items()
            if (k.startswith("format.") or k.startswith("content.") or
                k.startswith("faithfulness.") or k.startswith("compression.") or
                k.startswith("continuability.") or k == "outcome")
            and isinstance(v, float)
        },
        "diagnostics": {
            "response_chars": len(response_text),
            "source_chars": len(task["source_text"]),
        },
    }


def build_grpo_group(task: dict, gens: list[dict]) -> dict:
    """One GRPO training group ready for cuda_grpo_ablation."""
    pi_messages = task_scaffold.build_rollout_messages(task["source_messages"])
    completions = [
        {"text": g["response"], "reward": g["reward"]}
        for g in gens
    ]
    return {"messages": pi_messages, "completions": completions, "task_id": task["task_id"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--kiln-base", default="http://localhost:8420")
    ap.add_argument("--adapter", default=None,
                    help="Adapter name to use (omit/empty = use currently-loaded). "
                         "If 'current', leaves whatever kiln has active.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=3141592653)
    ap.add_argument("--task-limit", type=int, default=None,
                    help="Only run first N tasks (for smoke tests).")
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Parallel rollouts (kiln serves ~2 cleanly).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    with open(args.tasks) as f:
        for line in f:
            tasks.append(json.loads(line))
    if args.task_offset:
        tasks = tasks[args.task_offset:]
    if args.task_limit:
        tasks = tasks[:args.task_limit]

    print(f"loaded {len(tasks)} tasks; num_generations={args.num_generations} adapter={args.adapter or '(current)'} mode={args.mode}", flush=True)

    adapter_arg = None if args.adapter in (None, "", "current") else args.adapter

    grpo_groups: list[dict] = []
    rollouts_records: list[dict] = []

    # Build a list of (task, gen_idx) work items so the thread pool can
    # parallelise across both axes (helps if some tasks are slow).
    items: list[tuple[dict, int]] = []
    for t in tasks:
        for g in range(args.num_generations):
            items.append((t, g))

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = []
        for t, g in items:
            futures.append(pool.submit(
                run_rollout,
                t,
                base_url=args.kiln_base,
                adapter=adapter_arg,
                temperature=args.temperature,
                seed=args.seed + g,
                max_tokens=args.max_tokens,
                gen_idx=g,
            ))
        rollouts_by_task: dict[str, list[dict]] = {}
        for fut in as_completed(futures):
            r = fut.result()
            tid = r["task_id"]
            rollouts_by_task.setdefault(tid, []).append(r)
            rollouts_records.append(r)
            if args.verbose:
                print(
                    f"  task={tid} gen={r['gen']} reward={r['reward']:.2f} wall={r['wall_clock_s']:.1f}s err={r['error'] or 'ok'}",
                    flush=True,
                )

    # Order rollouts deterministically (task order + gen order) for downstream stability.
    rollouts_records.sort(key=lambda r: (r["task_id"], r["gen"]))

    rollouts_path = out_dir / "rollouts.jsonl"
    with rollouts_path.open("w") as f:
        for r in rollouts_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # GRPO group jsonl (training mode only)
    if args.mode == "train":
        for t in tasks:
            gens = sorted(rollouts_by_task.get(t["task_id"], []), key=lambda r: r["gen"])
            if len(gens) >= 2:
                grpo_groups.append(build_grpo_group(t, gens))
        grpo_path = out_dir / "grpo-train.jsonl"
        with grpo_path.open("w") as f:
            for g in grpo_groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

    composites = [r["reward"] for r in rollouts_records]
    walls = [r["wall_clock_s"] for r in rollouts_records]
    nonzero = sum(1 for r in composites if r > 0)
    zero = sum(1 for r in composites if r == 0)

    summary = {
        "mode": args.mode,
        "adapter": args.adapter,
        "n_tasks": len(tasks),
        "n_rollouts": len(rollouts_records),
        "n_generations": args.num_generations,
        "mean_composite": sum(composites) / max(1, len(composites)),
        "p50_composite": sorted(composites)[len(composites) // 2] if composites else 0,
        "rollouts_nonzero": nonzero,
        "rollouts_zero": zero,
        "mean_wall_clock_s": sum(walls) / max(1, len(walls)),
        "wall_clock_s_total": sum(walls),
        "grpo_groups_written": len(grpo_groups),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote rollouts -> {rollouts_path}", flush=True)
    if args.mode == "train":
        print(f"wrote grpo groups -> {out_dir / 'grpo-train.jsonl'}", flush=True)
    print(f"wrote summary -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
