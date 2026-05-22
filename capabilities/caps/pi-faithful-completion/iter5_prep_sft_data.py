"""iter5 data prep: convert strict-prompt rollouts into SFT pairs.

Strategy (§4.3 high-baseline rescue adapted as forward strategy):
  1. Generate rollouts on train.tasks.jsonl using the STRICT system prompt
     (h15-strict-system-prompt-system.txt) — the "teacher" is self+strict.
  2. Filter to composite > THRESHOLD (default 0.7) — keep only good ones.
  3. Build sft.train.jsonl with the DEFAULT system prompt in the input but
     the strict completion in the output. This teaches the base model to
     produce strict-style outputs even when not given the strict prompt.

Input: a rollouts.jsonl produced by rollout.py with --system-prompt-file
       set to h15-strict-system-prompt-system.txt.
Output: datasets/sft.train.jsonl in standard chat-completion format.

Why this should work and iter4 didn't:
  - iter4 used GRPO on strict-prompt rollouts. GRPO needs reward variance
    to learn — but strict-prompt rollouts saturate at mean 0.82 with low
    variance, so GRPO has no signal.
  - SFT just does maximum-likelihood on the filtered completions, which
    works fine on saturated reward distributions. It also doesn't care
    about variance — it cares about which completions to imitate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import task_scaffold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="Path to rollouts.jsonl from a strict-prompt run.")
    ap.add_argument("--tasks", required=True,
                    help="Path to train.tasks.jsonl (for the original system_prompt).")
    ap.add_argument("--out", required=True,
                    help="Path to write sft.train.jsonl.")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Keep rollouts with composite > this value.")
    ap.add_argument("--input-system-prompt", default="default",
                    choices=["default", "none", "strict"],
                    help="Which system prompt to use in SFT INPUT. 'default' "
                         "uses the task's default; 'none' uses no system; "
                         "'strict' uses the strict prompt (no transfer).")
    ap.add_argument("--strict-prompt-file", default=str(ROOT / "prompts/h15-strict-system-prompt-system.txt"))
    args = ap.parse_args()

    # Load tasks for default system prompts
    tasks_by_id: dict[str, dict] = {}
    with open(args.tasks) as f:
        for line in f:
            t = json.loads(line)
            tasks_by_id[t["task_id"]] = t

    strict_text = Path(args.strict_prompt_file).read_text() if args.input_system_prompt == "strict" else None

    # Load rollouts, filter, sort by score, dedupe per task to highest
    rollouts: list[dict] = []
    with open(args.rollouts) as f:
        for line in f:
            r = json.loads(line)
            rollouts.append(r)

    kept_total = 0
    filtered_total = 0
    no_task_total = 0
    by_task: dict[str, list[dict]] = {}

    for r in rollouts:
        if r.get("error"):
            continue
        if not r.get("response", "").strip():
            continue
        reward = r.get("reward", 0.0)
        if reward <= args.threshold:
            filtered_total += 1
            continue
        tid = r["task_id"]
        if tid not in tasks_by_id:
            no_task_total += 1
            continue
        by_task.setdefault(tid, []).append(r)
        kept_total += 1

    # Write SFT pairs — keep ALL kept rollouts as separate examples
    # (multiple high-scoring completions per task = more diverse training signal)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    avg_reward = 0.0
    score_histogram = {"0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0}

    with out_path.open("w") as f:
        for tid, rs in by_task.items():
            task = tasks_by_id[tid]
            for r in rs:
                # Build SFT input messages
                if args.input_system_prompt == "default":
                    sys_prompt = task.get("system_prompt", task_scaffold.DEFAULT_SYSTEM_PROMPT)
                elif args.input_system_prompt == "strict":
                    sys_prompt = strict_text
                else:  # none
                    sys_prompt = None

                messages: list[dict] = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": task["user_prompt"]})
                messages.append({"role": "assistant", "content": r["response"]})

                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                n_written += 1
                avg_reward += r["reward"]
                if r["reward"] >= 0.9:
                    score_histogram["0.9-1.0"] += 1
                elif r["reward"] >= 0.8:
                    score_histogram["0.8-0.9"] += 1
                else:
                    score_histogram["0.7-0.8"] += 1

    if n_written:
        avg_reward /= n_written

    summary = {
        "rollouts_total": len(rollouts),
        "kept": kept_total,
        "filtered_below_threshold": filtered_total,
        "missing_task": no_task_total,
        "threshold": args.threshold,
        "input_system_prompt": args.input_system_prompt,
        "sft_examples_written": n_written,
        "tasks_with_at_least_one_kept": len(by_task),
        "avg_kept_reward": avg_reward,
        "score_histogram": score_histogram,
        "out_path": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
