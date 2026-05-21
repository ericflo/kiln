"""Task-corpus builder for pi-test-interpretation.

Each task asks the agent to run a flaky test suite or benchmark
multiple times and report median/classifications correctly.

Output: 24 train + 12 eval.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31421))


SCENARIOS = [
    {
        "id": "flake_classification",
        "prompt": "Run the test suite 3 times. Classify each test as real-fail, flake, or pass.",
        "init_files": {
            "tests/test_a.py": "def test_a(): assert 1 == 1\n",
            "tests/test_flaky.py": "import random\n\ndef test_flaky():\n    assert random.random() > 0.3\n",
        },
        "gold_predicate": "agent_emits_per_test_classification",
    },
    {
        "id": "warmup_recognition",
        "prompt": "Run the bench at least 3 times. Recognize the first run is warmup; report median.",
        "init_files": {
            "bench.sh": "#!/bin/bash\necho '15.2ms'\n",
            "README.md": "First run warm; report median.\n",
        },
        "gold_predicate": "agent_reports_median",
    },
    {
        "id": "median_of_3",
        "prompt": "Run the perf test 3 times; report the MEDIAN.",
        "init_files": {
            "tests/test_perf.py": "def test_speed(): pass\n",
        },
        "gold_predicate": "agent_reports_median",
    },
    {
        "id": "real_vs_flake",
        "prompt": "3 runs. Tests failing all 3 are real; flaky tests pass sometimes.",
        "init_files": {
            "tests/test_real.py": "def test_broken(): assert False\n",
            "tests/test_flaky.py": "import random\n\ndef test_flaky():\n    assert random.random() > 0.5\n",
        },
        "gold_predicate": "agent_classifies_correctly",
    },
]


def _make_task(scenario: dict, idx: int) -> dict:
    return {
        "task_id": f"{scenario['id']}_{idx:03d}",
        "scenario_id": scenario["id"],
        "prompt": scenario["prompt"],
        "init_files": scenario["init_files"],
        "gold_predicate": scenario["gold_predicate"],
    }


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for scenario in SCENARIOS:
        for i in range(6):
            train.append(_make_task(scenario, i))
        for i in range(3):
            eval_.append(_make_task(scenario, 100 + i))
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train + {len(eval_)} eval tasks")


if __name__ == "__main__":
    main()
