"""Task-corpus builder for pi-error-recovery.

Generates a balanced corpus across the 6 error classes:
  file_not_found, permission_denied, syntax_error, command_not_found,
  dependency_missing, timeout.

Each task is a self-contained sandbox spec with seeded files that
deliberately trigger the intended failure on the first attempt, plus
a gold-state spec describing the workspace after a correct recovery.

Output:
  datasets/train.tasks.jsonl  (committed; 30 tasks)
  datasets/eval.tasks.jsonl   (GITIGNORED; 18 tasks, balanced per class)

Determinism: seeded by SEED env var (default 31415).
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31415))


ERROR_CLASSES = [
    "file_not_found",
    "permission_denied",
    "syntax_error",
    "command_not_found",
    "dependency_missing",
    "timeout",
]


def _t_file_not_found(rng: random.Random, idx: int) -> dict:
    target = f"data/file_{rng.randrange(100, 999)}.txt"
    actual = f"data/file_{rng.randrange(100, 999)}.txt"
    while actual == target:
        actual = f"data/file_{rng.randrange(100, 999)}.txt"
    return {
        "task_id": f"file_not_found_{idx:03d}",
        "error_class": "file_not_found",
        "init_files": {
            actual: "Important: the data you want is in this file.\nValue: 42\n",
            "README.md": f"Look up the value in {target}.\n",
        },
        "prompt": (
            f"Read the value from `{target}` and write it to `result.txt`. "
            f"If the file isn't where you expect, find it and use the actual path."
        ),
        "gold_state": {"result.txt": "42\n"},
        "valid_recovery_actions": ["ls", "find", "rg -l", "tree"],
        "invalid_recovery_actions": ["cat (already failed)", "git apply"],
    }


def _t_permission_denied(rng: random.Random, idx: int) -> dict:
    return {
        "task_id": f"permission_denied_{idx:03d}",
        "error_class": "permission_denied",
        "init_files": {
            "config.yaml": "version: 1\nname: alpha\n",
            "README.md": "Update config.yaml to set version to 2.\n",
        },
        "init_chmod": {"config.yaml": "0o444"},
        "prompt": "Update `config.yaml` to set `version: 2`. Preserve the other fields.",
        "gold_state": {"config.yaml": "version: 2\nname: alpha\n"},
        "valid_recovery_actions": ["chmod +w then write", "rm + write fresh"],
        "invalid_recovery_actions": ["sudo (no sudo in sandbox)"],
    }


def _t_syntax_error(rng: random.Random, idx: int) -> dict:
    return {
        "task_id": f"syntax_error_{idx:03d}",
        "error_class": "syntax_error",
        "init_files": {
            "lib/util.py": "def double(x):\n    return x * 2\n",
            "tests/test_util.py": (
                "from lib.util import triple\n\n"
                "def test_triple():\n"
                "    assert triple(3) == 9\n"
                "    assert triple(0) == 0\n"
            ),
        },
        "prompt": (
            "Add a `triple(x)` function to `lib/util.py` that returns `3*x`. "
            "Run `pytest -q tests/test_util.py` to verify. The first attempt "
            "you make may have a syntax issue; fix it and re-run."
        ),
        "verify_cmd": "python3 -m pytest -q tests/test_util.py",
        "gold_state_predicate": "tests pass",
        "valid_recovery_actions": ["re-edit lib/util.py with fixed syntax"],
        "invalid_recovery_actions": ["delete tests"],
    }


def _t_command_not_found(rng: random.Random, idx: int) -> dict:
    return {
        "task_id": f"command_not_found_{idx:03d}",
        "error_class": "command_not_found",
        "init_files": {
            "tests/test_one.py": (
                "import json\n\n"
                "def test_basic():\n"
                "    assert json.loads('{\"a\":1}')['a'] == 1\n"
            ),
            "README.md": "Run the tests under tests/.\n",
        },
        "prompt": (
            "Run the test suite using `runtests`. If `runtests` isn't "
            "available, use an alternative test runner that IS available."
        ),
        "verify_cmd": "python3 -m pytest -q tests/",
        "gold_state_predicate": "tests pass",
        "valid_recovery_actions": [
            "python -m pytest", "python -m unittest", "pytest"
        ],
        "invalid_recovery_actions": ["sudo apt-get install"],
    }


def _t_dependency_missing(rng: random.Random, idx: int) -> dict:
    return {
        "task_id": f"dependency_missing_{idx:03d}",
        "error_class": "dependency_missing",
        "init_files": {
            "lib/compute.py": "# TODO: implement using a string-distance metric\n",
            "tests/test_compute.py": (
                "from lib.compute import similar\n\n"
                "def test_simple():\n"
                "    assert similar('cat', 'cat') == 1.0\n"
                "    assert similar('cat', 'bat') < 1.0\n"
                "    assert similar('cat', 'xyz') < similar('cat', 'cap')\n"
            ),
        },
        "prompt": (
            "Implement `similar(a, b)` in `lib/compute.py`. Returns a "
            "string-similarity score in [0,1]. The task description hints at "
            "using `Levenshtein` (a third-party library) but you may use any "
            "stdlib alternative if Levenshtein isn't installed."
        ),
        "verify_cmd": "python3 -m pytest -q tests/test_compute.py",
        "gold_state_predicate": "tests pass",
        "valid_recovery_actions": [
            "use difflib.SequenceMatcher", "custom edit distance",
        ],
        "invalid_recovery_actions": ["pip install (sandbox blocks)"],
    }


def _t_timeout(rng: random.Random, idx: int) -> dict:
    return {
        "task_id": f"timeout_{idx:03d}",
        "error_class": "timeout",
        "init_files": {
            "scripts/scan.sh": (
                "#!/bin/bash\n# Pretend deep scan — sleeps indefinitely.\n"
                "while true; do sleep 1; done\n"
            ),
            "README.md": (
                "Use scripts/scan.sh to count lines in *.txt files.\n"
                "The script may hang; if it does, use a faster alternative.\n"
            ),
            "data/a.txt": "alpha\nbeta\ngamma\n",
            "data/b.txt": "one\ntwo\n",
        },
        "init_chmod": {"scripts/scan.sh": "0o755"},
        "prompt": (
            "Count the lines across all *.txt files under data/ and write the "
            "total to `count.txt`. README suggests scripts/scan.sh; if that "
            "hangs, use a faster alternative."
        ),
        "gold_state": {"count.txt": "5\n"},
        "valid_recovery_actions": [
            "wc -l data/*.txt | tail",
            "find data -name '*.txt' -exec cat {} \\; | wc -l",
        ],
        "invalid_recovery_actions": ["nohup scripts/scan.sh &"],
    }


GENERATORS = {
    "file_not_found": _t_file_not_found,
    "permission_denied": _t_permission_denied,
    "syntax_error": _t_syntax_error,
    "command_not_found": _t_command_not_found,
    "dependency_missing": _t_dependency_missing,
    "timeout": _t_timeout,
}


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    train_per_class = 5
    eval_per_class = 3

    train, eval_ = [], []
    for cls in ERROR_CLASSES:
        gen = GENERATORS[cls]
        for i in range(train_per_class):
            train.append(gen(rng, i))
        for i in range(eval_per_class):
            eval_.append(gen(rng, 100 + i))

    rng.shuffle(train)
    rng.shuffle(eval_)

    train_path = DATASETS / "train.tasks.jsonl"
    eval_path = DATASETS / "eval.tasks.jsonl"
    with open(train_path, "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(eval_path, "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train tasks → {train_path}")
    print(f"wrote {len(eval_)} eval tasks  → {eval_path}")
    print(f"  per class: train={train_per_class}, eval={eval_per_class}")
    print(f"  classes:   {', '.join(ERROR_CLASSES)}")


if __name__ == "__main__":
    main()
