"""Task-corpus builder for pi-shell-hygiene.

Each task asks the agent to run a long-running shell command (or
manage a background process). The rubric inspects the issued commands
for known good/bad patterns documented in the clouderic kiln-skill.

Output: 24 train + 12 eval.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31420))


SCENARIOS = [
    {
        "id": "long_build",
        "prompt": (
            "You need to run `make build` which takes 10-20 minutes. "
            "Launch it in the background and return when it's done. "
            "Don't poll with sleep+ssh."
        ),
        "good_patterns": ["nohup", "bg ", "wait-file --timeout", "&\n"],
        "bad_patterns": ["until ssh", "while ssh", "sleep 5;", "sleep 10;"],
    },
    {
        "id": "ci_watch",
        "prompt": (
            "Watch a CI run that takes about 8 minutes. Don't burn cache "
            "by polling every second; use a single check with appropriate "
            "timeout."
        ),
        "good_patterns": ["sleep 270", "sleep 600", "wait-file"],
        "bad_patterns": ["sleep 5", "sleep 10", "until "],
    },
    {
        "id": "process_cleanup",
        "prompt": (
            "Spawn a background process and ensure cleanup on failure. "
            "Do NOT use bare `trap ... EXIT` because each bash tool-call "
            "is a fresh shell and EXIT would kill the pod immediately."
        ),
        "good_patterns": ["trap ", "ERR INT TERM"],
        "bad_patterns": ["trap 'cleanup' EXIT", "trap cleanup EXIT"],
    },
    {
        "id": "wait_for_file",
        "prompt": (
            "Start a build and wait for /tmp/build.done to appear. "
            "Use `wait-file --timeout` rather than a sleep+test loop."
        ),
        "good_patterns": ["wait-file", "--timeout"],
        "bad_patterns": ["while [ ! -f", "until [ -f", "while ! test -f"],
    },
]


def _make_task(scenario: dict, idx: int, label: str) -> dict:
    return {
        "task_id": f"{scenario['id']}_{idx:03d}",
        "scenario_id": scenario["id"],
        "prompt": scenario["prompt"],
        "good_patterns": scenario["good_patterns"],
        "bad_patterns": scenario["bad_patterns"],
        "init_files": {
            "README.md": scenario["prompt"] + "\n",
        },
        "gold_state_predicate": "command_uses_good_pattern_not_bad",
    }


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for scenario in SCENARIOS:
        for i in range(6):
            train.append(_make_task(scenario, i, "train"))
        for i in range(3):
            eval_.append(_make_task(scenario, 100 + i, "eval"))
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train tasks across 4 scenarios")
    print(f"wrote {len(eval_)} eval tasks")


if __name__ == "__main__":
    main()
