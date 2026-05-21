"""Task-corpus builder for pi-script-fixup (§5.5 verifier-free).

Each task gives the agent a script that fails to run (parse error,
exception at startup, wrong shebang). The agent must read the error
and fix it. There's no oracle verifier — the rubric uses ECHO-only
adaptation, training on env tokens (the error output).

Output: 24 train + 12 eval.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31424))


SCRIPTS = [
    {
        "id": "syntax_fix",
        "init_files": {
            "broken.py": "def main()\n    print('hello')\n\nmain()\n",
        },
        "prompt": "Run `python3 broken.py` and fix whatever errors show up.",
        "gold_fix": "missing colon after def main()",
    },
    {
        "id": "import_fix",
        "init_files": {
            "broken.py": "import non_existent_module\nprint('hi')\n",
        },
        "prompt": "Fix broken.py so it runs.",
        "gold_fix": "remove or replace the bad import",
    },
    {
        "id": "indent_fix",
        "init_files": {
            "broken.py": "def f(x):\n    if x > 0:\n    return 1\n    return 0\n",
        },
        "prompt": "Fix the indentation in broken.py.",
        "gold_fix": "indent return 1 under the if",
    },
    {
        "id": "shebang_fix",
        "init_files": {
            "broken.sh": "#!/usr/bin/wrong_shell\necho hello\n",
        },
        "prompt": "Make broken.sh executable and runnable.",
        "gold_fix": "change shebang to /bin/bash",
    },
]


def _make_task(s: dict, idx: int) -> dict:
    return {
        "task_id": f"{s['id']}_{idx:03d}",
        "fix_kind": s["id"],
        "init_files": s["init_files"],
        "prompt": s["prompt"],
        "gold_fix": s["gold_fix"],
    }


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for s in SCRIPTS:
        for i in range(6):
            train.append(_make_task(s, i))
        for i in range(3):
            eval_.append(_make_task(s, 100 + i))
    rng.shuffle(train)
    rng.shuffle(eval_)
    with open(DATASETS / "train.tasks.jsonl", "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
    with open(DATASETS / "eval.tasks.jsonl", "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(train)} train + {len(eval_)} eval")


if __name__ == "__main__":
    main()
