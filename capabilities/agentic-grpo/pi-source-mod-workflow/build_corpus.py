"""Task-corpus builder for pi-source-mod-workflow (eval-only integration).

Round 2 reshape: this cap is an INTEGRATION TEST that runs the full
clone -> branch -> edit -> test -> push -> PR workflow end-to-end. It
does NOT train its own adapter — it evaluates an existing adapter
(typically a composite adapter or the latest cross-cap-coherence adapter).

The tasks here are full workflow scenarios with sub-step gold checks.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31423))


WORKFLOWS = [
    {
        "id": "clone_branch_edit_test",
        "prompt": (
            "Clone a fresh repo, create a branch, add a Python function "
            "with a passing test, then commit. Sub-steps:\n"
            "  1. git clone (simulated via init_files)\n"
            "  2. git checkout -b feature\n"
            "  3. write lib/foo.py with the function\n"
            "  4. write tests/test_foo.py\n"
            "  5. run pytest\n"
            "  6. git add + git commit\n"
        ),
        "init_files": {
            ".git/HEAD": "ref: refs/heads/main\n",
            "README.md": "Test repo.\n",
        },
        "gold_sub_steps": [
            "git checkout -b",
            "lib/foo.py",
            "tests/test_foo.py",
            "pytest",
            "git commit",
        ],
    },
    {
        "id": "fix_and_pr",
        "prompt": (
            "There's a bug in lib/util.py (off-by-one). Branch, fix, "
            "verify with pytest, commit, then prepare a PR description."
        ),
        "init_files": {
            ".git/HEAD": "ref: refs/heads/main\n",
            "lib/util.py": "def last(items): return items[len(items)]\n",
            "tests/test_util.py": "from lib.util import last\n\ndef test_last():\n    assert last([1,2,3]) == 3\n",
        },
        "gold_sub_steps": [
            "git checkout -b",
            "lib/util.py",
            "pytest",
            "git commit",
            "## Summary",
        ],
    },
]


def _make_task(workflow: dict, idx: int) -> dict:
    return {
        "task_id": f"smw_{workflow['id']}_{idx:03d}",
        "workflow_id": workflow["id"],
        "prompt": workflow["prompt"],
        "init_files": workflow["init_files"],
        "gold_sub_steps": workflow["gold_sub_steps"],
    }


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for w in WORKFLOWS:
        for i in range(6):
            train.append(_make_task(w, i))
        for i in range(3):
            eval_.append(_make_task(w, 100 + i))
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train + {len(eval_)} eval")
    print("NOTE: this cap is an INTEGRATION TEST. run_iter.sh does NOT train.")


if __name__ == "__main__":
    main()
