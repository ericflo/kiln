"""Synthesize "ideal" SFT data from each train task's gold structured answer.

Mirrors pi-faithful-completion's iter18 pattern: every train task has a
canonical gold JSON answer. The "ideal" assistant output is

    <answer>
    {json.dumps(gold)}
    </answer>

This is GROUND TRUTH, not a sampled rollout — every example scores ~1.0 on
the rubric by construction (modulo type/identifier normalisation noise).

For SFT, the training pair is:
  - prompt:     the user message that pi sees (from task_scaffold.pi_prompt)
  - completion: the ideal <answer>{...}</answer> block

The SFT trainer treats this as single-turn chat completion. At inference,
pi multi-turn still works — the model has just learned the OUTPUT shape.
The tool-use trajectory (read, grep) emerges from pi's session shape and
the model's pre-trained tool-use behaviour; SFT here teaches "what shape
the FINAL turn must take," not "when to call read/grep."

Usage:
    python3 prep_sft_ideal.py [--out datasets/sft.ideal.jsonl]
                              [--max-tasks N]
                              [--include-system-prompt]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import task_scaffold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "datasets/train.tasks.jsonl"))
    ap.add_argument("--out",   default=str(ROOT / "datasets/sft.ideal.jsonl"))
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--include-system-prompt", action="store_true",
                    help="Prepend the GRPO-trained system message so SFT shares the same anchor")
    args = ap.parse_args()

    system_msg = (
        "You are a Python code-comprehension assistant. "
        "You have access to read, edit, write, and bash "
        "tools. Investigate the codebase, then emit your "
        "final answer as a single <answer>{...}</answer> "
        "JSON block."
    )

    n_in = n_out = n_skip = 0
    with open(args.tasks) as f, open(args.out, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.max_tasks > 0 and n_out >= args.max_tasks:
                break
            t = json.loads(line)
            gold = t.get("gold") or {}
            # Normalize gold so json.dumps doesn't crash on tuple/sets — should
            # be plain dict/list already, but defensive.
            required = ["inputs", "returns", "mutates", "calls",
                        "called_by", "invariants", "side_effects"]
            answer = {k: gold.get(k, []) for k in required}
            # If gold has invariants/side_effects as {primary, paraphrases},
            # pick the primary text only (model emits flat strings).
            for key in ("invariants", "side_effects"):
                fixed = []
                for item in answer.get(key, []):
                    if isinstance(item, dict) and "primary" in item:
                        fixed.append(item["primary"])
                    elif isinstance(item, str):
                        fixed.append(item)
                    else:
                        fixed.append(str(item))
                answer[key] = fixed

            user_prompt = task_scaffold.pi_prompt(t)
            assistant_text = "<answer>\n" + json.dumps(answer) + "\n</answer>"

            messages = []
            if args.include_system_prompt:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": user_prompt})
            # SftExample schema (kiln-train::lib.rs) = {messages: [..., assistant]}.
            # Assistant turn must be IN messages as the last element. No
            # separate "completion" field.
            messages.append({"role": "assistant", "content": assistant_text})
            row = {
                "task_id": t.get("task_id"),
                "messages": messages,
            }
            out.write(json.dumps(row) + "\n")
            n_out += 1

    print(f"wrote {n_out}/{n_in} SFT rows to {args.out}  (skipped {n_skip})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
