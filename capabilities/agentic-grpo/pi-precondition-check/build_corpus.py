"""Task-corpus builder for pi-precondition-check (rank-1 cap, round 2).

Each task is a workspace + a claim about it + a ground-truth label
(holds_true OR stale). Balanced 50/50.

Output: 32 train + 16 eval, balanced across 4 claim templates.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31419))


CLAIM_TEMPLATES = [
    {
        "id": "md5_to_sha256",
        "prompt_template": "The `compute_hash` helper in `lib/util.py` uses MD5. Migrate it to SHA-256 by changing the import and the `hashlib.md5(...)` call.",
        "init_holds_true": {"lib/util.py": "import hashlib\n\ndef compute_hash(s: str) -> str:\n    return hashlib.md5(s.encode()).hexdigest()\n"},
        "init_stale": {"lib/util.py": "import hashlib\n\ndef compute_hash(s: str) -> str:\n    return hashlib.sha256(s.encode()).hexdigest()\n"},
        "claim_id": "claim_md5",
        "claim_path": "lib/util.py",
        "claim_symbol": "compute_hash",
        "claim_substring": "hashlib.md5",
        "gold_after_holds_substring": "hashlib.sha256",
    },
    {
        "id": "log_level_info_to_debug",
        "prompt_template": "Lower the log level in `lib/server.py` from INFO to DEBUG.",
        "init_holds_true": {"lib/server.py": "import logging\nlogger = logging.getLogger(__name__)\nlogger.setLevel(logging.INFO)\n"},
        "init_stale": {"lib/server.py": "import logging\nlogger = logging.getLogger(__name__)\nlogger.setLevel(logging.DEBUG)\n"},
        "claim_id": "claim_loglevel",
        "claim_path": "lib/server.py",
        "claim_symbol": "logger.setLevel",
        "claim_substring": "logging.INFO",
        "gold_after_holds_substring": "logging.DEBUG",
    },
    {
        "id": "rename_handler",
        "prompt_template": "Rename `handle_request` in `lib/api.py` to `process_request`.",
        "init_holds_true": {"lib/api.py": "def handle_request(req):\n    return {'ok': True}\n"},
        "init_stale": {"lib/api.py": "def process_request(req):\n    return {'ok': True}\n"},
        "claim_id": "claim_rename",
        "claim_path": "lib/api.py",
        "claim_symbol": "handle_request",
        "claim_substring": "def handle_request",
        "gold_after_holds_substring": "def process_request",
    },
    {
        "id": "fix_off_by_one",
        "prompt_template": "Fix the off-by-one bug in `range(len(items)-1)` in `lib/iter.py` — it should be `range(len(items))`.",
        "init_holds_true": {"lib/iter.py": "def iterate(items):\n    return [items[i] for i in range(len(items)-1)]\n"},
        "init_stale": {"lib/iter.py": "def iterate(items):\n    return [items[i] for i in range(len(items))]\n"},
        "claim_id": "claim_offbyone",
        "claim_path": "lib/iter.py",
        "claim_symbol": "iterate",
        "claim_substring": "range(len(items)-1)",
        "gold_after_holds_substring": "range(len(items))",
    },
]


def _task_from_template(template: dict, label: str, idx: int) -> dict:
    init_files = template[f"init_{label}"]
    return {
        "task_id": f"{template['id']}_{label}_{idx:03d}",
        "claim_id": template["claim_id"],
        "claim_label": label,
        "claim_path": template["claim_path"],
        "claim_symbol": template["claim_symbol"],
        "claim_substring": template["claim_substring"],
        "init_files": init_files,
        "prompt": template["prompt_template"],
        "gold_state_predicate": (
            "post-edit-substring-present" if label == "holds_true"
            else "no-mutation-and-sentinel"
        ),
        "gold_after_substring": template.get("gold_after_holds_substring"),
        "sentinel": f"precondition_failed:{template['claim_id']}",
    }


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for template in CLAIM_TEMPLATES:
        for i in range(4):
            train.append(_task_from_template(template, "holds_true", i))
            train.append(_task_from_template(template, "stale", i))
        for i in range(2):
            eval_.append(_task_from_template(template, "holds_true", 100 + i))
            eval_.append(_task_from_template(template, "stale", 100 + i))
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train tasks (50/50 holds_true vs stale)")
    print(f"wrote {len(eval_)} eval tasks (50/50)")


if __name__ == "__main__":
    main()
