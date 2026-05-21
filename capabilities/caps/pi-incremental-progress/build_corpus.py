"""Task-corpus builder for pi-incremental-progress.

Generates a corpus of multi-step refactor tasks across 3 difficulty
levels (each requiring 2-6 verifiable sub-steps). Each task includes a
gold decomposition the rubric uses to measure alignment.

Output:
  datasets/train.tasks.jsonl  (committed; 24 tasks)
  datasets/eval.tasks.jsonl   (GITIGNORED; 12 tasks)

Determinism: seeded by SEED env var (default 31416).
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31416))


def _t_extract_module(rng: random.Random, idx: int) -> dict:
    return {
        "task_id": f"extract_module_{idx:03d}",
        "difficulty": "medium",
        "init_files": {
            "lib/cache.py": (
                "import time\n\n"
                "class MemCache:\n"
                "    def __init__(self): self._d = {}\n"
                "    def get(self, k): return self._d.get(k)\n"
                "    def set(self, k, v): self._d[k] = (time.time(), v)\n\n"
                "class DiskCache:\n"
                "    def __init__(self, path): self.path = path\n"
                "    def get(self, k): return None\n"
                "    def set(self, k, v): pass\n"
            ),
            "tests/test_cache.py": (
                "from lib.memory_cache import MemCache\n"
                "from lib.disk_cache import DiskCache\n\n"
                "def test_mem():\n"
                "    c = MemCache()\n"
                "    c.set('a', 1)\n"
                "    assert c.get('a')[1] == 1\n\n"
                "def test_disk(tmp_path):\n"
                "    c = DiskCache(str(tmp_path))\n"
                "    assert c.get('x') is None\n"
            ),
        },
        "prompt": (
            "Refactor lib/cache.py: extract MemCache into lib/memory_cache.py "
            "and DiskCache into lib/disk_cache.py. Each step should be "
            "verifiable on its own. Remove the original lib/cache.py last."
        ),
        "gold_decomposition": [
            "create lib/memory_cache.py with MemCache",
            "verify memory_cache import works",
            "create lib/disk_cache.py with DiskCache",
            "verify disk_cache import works",
            "remove lib/cache.py",
            "run pytest -q",
        ],
        "verify_cmd": "python3 -m pytest -q tests/test_cache.py",
        "gold_state_predicate": "tests pass",
        "gold_touched_paths": ["lib/memory_cache.py", "lib/disk_cache.py", "lib/cache.py"],
    }


def _t_rename_symbol(rng: random.Random, idx: int) -> dict:
    return {
        "task_id": f"rename_symbol_{idx:03d}",
        "difficulty": "easy",
        "init_files": {
            "lib/api.py": (
                "def fetch_data(url):\n    return {'url': url}\n\n"
                "def process_data(d):\n    return d.get('url', '')\n"
            ),
            "tests/test_api.py": (
                "from lib.api import retrieve_payload, transform_payload\n\n"
                "def test_pipeline():\n"
                "    p = retrieve_payload('https://example.com')\n"
                "    assert transform_payload(p) == 'https://example.com'\n"
            ),
        },
        "prompt": (
            "Rename `fetch_data` → `retrieve_payload` and "
            "`process_data` → `transform_payload` in lib/api.py. The tests "
            "in tests/test_api.py expect the new names. Do this in TWO "
            "verifiable steps (rename one symbol, verify, rename the "
            "second, verify) rather than one big edit."
        ),
        "gold_decomposition": [
            "rename fetch_data to retrieve_payload",
            "verify partial test (test_pipeline fails on transform_payload only)",
            "rename process_data to transform_payload",
            "verify full test passes",
        ],
        "verify_cmd": "python3 -m pytest -q tests/test_api.py",
        "gold_state_predicate": "tests pass",
        "gold_touched_paths": ["lib/api.py"],
    }


def _t_add_validation(rng: random.Random, idx: int) -> dict:
    return {
        "task_id": f"add_validation_{idx:03d}",
        "difficulty": "medium",
        "init_files": {
            "lib/user.py": (
                "def create_user(name, age, email):\n"
                "    return {'name': name, 'age': age, 'email': email}\n"
            ),
            "tests/test_user.py": (
                "import pytest\n"
                "from lib.user import create_user\n\n"
                "def test_happy():\n"
                "    u = create_user('Alice', 30, 'a@b.c')\n"
                "    assert u['name'] == 'Alice'\n\n"
                "def test_empty_name():\n"
                "    with pytest.raises(ValueError): create_user('', 30, 'a@b.c')\n\n"
                "def test_negative_age():\n"
                "    with pytest.raises(ValueError): create_user('Alice', -1, 'a@b.c')\n\n"
                "def test_bad_email():\n"
                "    with pytest.raises(ValueError): create_user('Alice', 30, 'noatsign')\n"
            ),
        },
        "prompt": (
            "Add validation to `create_user` for: empty name, negative age, "
            "and missing '@' in email. There are 4 tests; add validation in "
            "verifiable steps (one rule per step is ideal)."
        ),
        "gold_decomposition": [
            "add empty-name check; verify test_empty_name passes",
            "add negative-age check; verify test_negative_age passes",
            "add email '@' check; verify all 4 tests pass",
        ],
        "verify_cmd": "python3 -m pytest -q tests/test_user.py",
        "gold_state_predicate": "tests pass",
        "gold_touched_paths": ["lib/user.py"],
    }


GENERATORS = [_t_extract_module, _t_rename_symbol, _t_add_validation]


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for gen in GENERATORS:
        for i in range(8):
            train.append(gen(rng, i))
        for i in range(4):
            eval_.append(gen(rng, 100 + i))
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train tasks")
    print(f"wrote {len(eval_)} eval tasks")


if __name__ == "__main__":
    main()
