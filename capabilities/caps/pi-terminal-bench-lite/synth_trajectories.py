"""Generate synthetic AgenticGroup trajectory JSONL for end-to-end
validation of the ECHO training pipeline.

Each group represents one task. Each rollout in the group has a
multi-turn trajectory with action+observation segments and a
deterministic reward (so GRPO advantage is non-degenerate).

The trajectories model a realistic file-manipulation agent: read a
file → edit it → verify. The Observation segments include
shell-like output that the env-CE term can learn the structure of.

Usage:
    python synth_trajectories.py --out trajectories.jsonl --n-groups 3 --n-per-group 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


TASK_TEMPLATES = [
    {
        "task": "Write 'hello' to greet.txt and then print its content.",
        "obs_good": "$ cat greet.txt\nhello\n",
        "obs_bad_a": "cat: greet.txt: No such file or directory\n",
        "obs_bad_b": "$ cat greet.txt\nhelo\n",  # typo
    },
    {
        "task": "Append the number 42 to nums.txt as a new line.",
        "obs_good": "$ cat nums.txt\n1\n2\n3\n42\n",
        "obs_bad_a": "$ cat nums.txt\n1\n2\n3\n",
        "obs_bad_b": "Permission denied: nums.txt\n",
    },
    {
        "task": "Create config.json with {'enabled': true}.",
        "obs_good": "$ cat config.json\n{\"enabled\": true}\n",
        "obs_bad_a": "$ cat config.json\n{}\n",
        "obs_bad_b": "$ cat config.json\nJSON parse error\n",
    },
]


def make_action(text: str) -> dict:
    return {"role": "assistant", "content": text, "kind": "action"}


def make_obs(text: str) -> dict:
    return {"role": "tool", "content": text, "kind": "observation"}


def build_good_trajectory(task: dict) -> tuple[list[dict], float]:
    """Successful three-turn solve. Reward = 1.0. Matches the pi_trajectory
    convention: one Action segment per assistant turn (reasoning +
    tool_call combined into one content block)."""
    traj = [
        make_action(
            f"I will solve: {task['task']}\n"
            f"Let me plan: first I'll do the action, then verify.\n"
            f"<tool_call>{{\"name\":\"bash\",\"input\":{{\"command\":\"# do the action\"}}}}</tool_call>"
        ),
        make_obs("Action completed.\n"),
        make_action(
            f"Now I'll verify by reading the file back.\n"
            f"<tool_call>{{\"name\":\"bash\",\"input\":{{\"command\":\"cat file\"}}}}</tool_call>"
        ),
        make_obs(task["obs_good"]),
        make_action("Verified. The task is complete."),
    ]
    return traj, 1.0


def build_partial_trajectory(task: dict, kind: str) -> tuple[list[dict], float]:
    """One mistake then partial recovery. Reward = 0.5."""
    bad = task[f"obs_bad_{kind}"]
    traj = [
        make_action(
            f"I will solve: {task['task']}\n"
            f"<tool_call>{{\"name\":\"bash\",\"input\":{{\"command\":\"# attempt\"}}}}</tool_call>"
        ),
        make_obs(bad),
        make_action(
            f"That output is wrong. Let me retry.\n"
            f"<tool_call>{{\"name\":\"bash\",\"input\":{{\"command\":\"# retry\"}}}}</tool_call>"
        ),
        make_obs(task["obs_good"]),
        make_action("Now verified."),
    ]
    return traj, 0.5


def build_failed_trajectory(task: dict, kind: str) -> tuple[list[dict], float]:
    """Loops on the same error. Reward = 0.0."""
    bad = task[f"obs_bad_{kind}"]
    traj = [
        make_action(
            f"I will solve: {task['task']}\n"
            f"<tool_call>{{\"name\":\"bash\",\"input\":{{\"command\":\"# attempt\"}}}}</tool_call>"
        ),
        make_obs(bad),
        make_action(
            f"Let me try again.\n"
            f"<tool_call>{{\"name\":\"bash\",\"input\":{{\"command\":\"# attempt 2\"}}}}</tool_call>"
        ),
        make_obs(bad),
        make_action("Still broken; giving up."),
    ]
    return traj, 0.0


def flatten_actions(trajectory: list[dict]) -> str:
    """Concat all Action segment content with <TURN_BREAK> separators —
    matches what kiln-train::ScoredRollout.text expects when callers
    don't have a richer text representation."""
    return "<TURN_BREAK>".join(seg["content"] for seg in trajectory if seg["kind"] == "action")


def build_group(task: dict, rng: random.Random) -> dict:
    """One AgenticGroup with 4 rollouts of varying quality."""
    sys_prompt = (
        "You are a careful Python agent. You have access to a bash tool. "
        "Solve the user's task, verify your work by reading file output, "
        "and emit a final assistant message with no tool calls."
    )
    user_msg = task["task"]

    rollouts = []
    # 1 good, 2 partial (different mistakes), 1 failed
    traj1, r1 = build_good_trajectory(task)
    traj2, r2 = build_partial_trajectory(task, "a")
    traj3, r3 = build_partial_trajectory(task, "b")
    traj4, r4 = build_failed_trajectory(task, "a")
    for traj, reward in [(traj1, r1), (traj2, r2), (traj3, r3), (traj4, r4)]:
        rollouts.append({
            "text": flatten_actions(traj),
            "reward": reward,
            "trajectory": traj,
        })
    rng.shuffle(rollouts)  # don't pin best-first

    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ],
        "completions": rollouts,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-groups", type=int, default=3)
    p.add_argument("--seed", type=int, default=3141592653)
    args = p.parse_args()

    rng = random.Random(args.seed)
    tasks = TASK_TEMPLATES[: args.n_groups]
    while len(tasks) < args.n_groups:
        tasks.append(TASK_TEMPLATES[len(tasks) % len(TASK_TEMPLATES)])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for task in tasks:
            f.write(json.dumps(build_group(task, rng)) + "\n")

    n_rollouts = sum(1 for _ in tasks) * 4
    n_obs_tokens_est = n_rollouts * 2 * 40  # ~2 obs per trajectory, ~40 chars each
    print(f"wrote {args.n_groups} groups, {n_rollouts} rollouts → {out}")
    print(f"approx {n_obs_tokens_est} observation chars across all groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
