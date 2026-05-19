"""Pi-faithful-completion rollout harness — single-turn text completion.

Each rollout is one chat-completion call to kiln. Score the response with
the multi-component rubric, emit JSONL ready for `cuda_grpo_ablation`.

Modes
-----
- `--mode train`: emit one GrpoGroup line per task, with N completions each
- `--mode eval`:  single best-of-N pass (N=1 by default for determinism),
                  emit rollouts + summary.json with mean composite

Usage
-----
    python3 rollout.py --tasks datasets/train.tasks.jsonl \\
                       --out-dir /tmp/iter0-rollouts \\
                       --mode train --num-generations 4

The rollout uses `--adapter current` to mean "whatever kiln has loaded".
Set the adapter before invoking via POST /v1/adapters/load (or unload for base).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rubric  # noqa: E402
import task_scaffold  # noqa: E402


def kiln_chat_completion(
    base_url: str,
    messages: list[dict],
    *,
    max_tokens: int = 768,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int | None = None,
    adapter: str | None = None,
    enable_thinking: bool = False,
    timeout: int = 180,
) -> dict:
    body = {
        "model": "qwen3.5-4b-kiln",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
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
    top_p: float,
    seed: int,
    max_tokens: int,
    gen_idx: int,
    retries: int = 2,
) -> dict:
    messages = task_scaffold.build_messages(task)
    last_err = None
    response_text = ""
    usage = {}
    wall = 0.0
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            resp = kiln_chat_completion(
                base_url, messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                adapter=adapter,
            )
            wall = time.time() - t0
            response_text = parse_response_text(resp)
            usage = resp.get("usage", {}) or {}
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            wall = time.time() - t0
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2.0 + attempt, 5.0))

    score = rubric.score_rollout(response_text, task)
    return {
        "task_id": task["task_id"],
        "task_kind": task.get("task_kind", ""),
        "is_failure_task": task.get("is_failure", False),
        "gen": gen_idx,
        "response": response_text,
        "wall_clock_s": wall,
        "error": last_err,
        "usage": usage,
        "reward": score["composite"],
        "composite_ungated": score.get("composite_ungated", 0.0),
        "sub_scores": {
            k: v for k, v in score.items()
            if (not k.startswith("_")) and isinstance(v, (int, float))
        },
        "diagnostics": {
            "response_chars": len(response_text),
            "response_words": len(response_text.split()),
        },
    }


def build_grpo_group(task: dict, gens: list[dict]) -> dict:
    """One GRPO training group for cuda_grpo_ablation."""
    messages = task_scaffold.build_messages(task)
    completions = [
        {"text": g["response"], "reward": g["reward"]}
        for g in gens
    ]
    return {
        "messages": messages,
        "completions": completions,
        "task_id": task["task_id"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--kiln-base", default="http://localhost:8420")
    ap.add_argument("--adapter", default=None,
                    help="Adapter name to use (omit/empty/`current` = use whatever is loaded).")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=3141592653)
    ap.add_argument("--task-limit", type=int, default=None)
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--system-prompt-file", default=None,
                    help="If set, override each task's system_prompt with the file contents.")
    args = ap.parse_args()

    system_prompt_override = None
    if args.system_prompt_file:
        system_prompt_override = Path(args.system_prompt_file).read_text()

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

    # Apply system-prompt override if requested
    if system_prompt_override is not None:
        for t in tasks:
            t["system_prompt"] = system_prompt_override

    print(f"loaded {len(tasks)} tasks; num_generations={args.num_generations} "
          f"adapter={args.adapter or '(current)'} mode={args.mode}", flush=True)

    adapter_arg = None if args.adapter in (None, "", "current") else args.adapter

    rollouts_records: list[dict] = []
    rollouts_by_task: dict[str, list[dict]] = {}

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
                top_p=args.top_p,
                seed=args.seed + g,
                max_tokens=args.max_tokens,
                gen_idx=g,
            ))
        for fut in as_completed(futures):
            r = fut.result()
            tid = r["task_id"]
            rollouts_by_task.setdefault(tid, []).append(r)
            rollouts_records.append(r)
            if args.verbose:
                print(
                    f"  task={tid} gen={r['gen']} reward={r['reward']:.3f} "
                    f"wall={r['wall_clock_s']:.1f}s err={r['error'] or 'ok'}",
                    flush=True,
                )

    rollouts_records.sort(key=lambda r: (r["task_id"], r["gen"]))
    rollouts_path = out_dir / "rollouts.jsonl"
    with rollouts_path.open("w") as f:
        for r in rollouts_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    grpo_groups: list[dict] = []
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
    # group variance
    group_vars = []
    for tid, gens in rollouts_by_task.items():
        rs = [g["reward"] for g in gens]
        if len(rs) >= 2:
            group_vars.append(statistics.variance(rs))
    mean_group_var = sum(group_vars) / max(1, len(group_vars)) if group_vars else 0.0

    # Sub-score means
    subscore_keys = set()
    for r in rollouts_records:
        subscore_keys.update(r.get("sub_scores", {}).keys())
    subscore_means: dict[str, float] = {}
    for k in subscore_keys:
        vals = [r["sub_scores"].get(k, 0.0) for r in rollouts_records if k in r.get("sub_scores", {})]
        if vals:
            subscore_means[k] = sum(vals) / len(vals)

    summary = {
        "mode": args.mode,
        "adapter": args.adapter,
        "n_tasks": len(tasks),
        "n_rollouts": len(rollouts_records),
        "n_generations": args.num_generations,
        "mean_composite": sum(composites) / max(1, len(composites)),
        "p50_composite": sorted(composites)[len(composites) // 2] if composites else 0.0,
        "rollouts_nonzero": nonzero,
        "rollouts_zero": zero,
        "mean_wall_clock_s": sum(walls) / max(1, len(walls)),
        "wall_clock_s_total": sum(walls),
        "grpo_groups_written": len(grpo_groups),
        "mean_group_variance": mean_group_var,
        "subscore_means": subscore_means,
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
