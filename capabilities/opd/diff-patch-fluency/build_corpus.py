"""Synthesize diff-patch-fluency prompts.

Each prompt shows the model a source file and asks for a specific edit
as a unified diff. The eval scorer applies the diff and checks intent
keywords + minimal-change ratio.

Output:
  prompts/h1-r16-6ep.jsonl  — training prompts (with dummy assistant
                              turn for kiln tokenizer; student samples)
  datasets/train.opd.jsonl  — copy of training prompts (kiln OPD reads)
  datasets/eval.jsonl       — held-out eval set (firewall-protected)

Schema per line:
  {
    "id": "<tool>-<hash>",
    "source": "<file content>",
    "source_path": "<path/in/diff/header>",
    "intent_keywords": ["..."],
    "intent_anti_keywords": ["..."],
    "expected_line_changes": <int>,
    "messages": [system, user, dummy-assistant],
  }
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(20260518)

# 12 distinct edit shapes. Each shape declares: file content template,
# the asked-for change, and the rubric inputs that verify it.
EDITS = [
    {
        "shape": "rename_var",
        "source": "x = 10\ny = 20\nresult = x + y\nprint(result)\n",
        "source_path": "calc.py",
        "ask": "Rename the variable `result` to `total` throughout calc.py",
        "intent_keywords": ["total = x + y", "print(total)"],
        "intent_anti_keywords": ["result = x + y", "print(result)"],
        "expected_line_changes": 2,
    },
    {
        "shape": "add_import",
        "source": "import os\nimport sys\n\ndef main():\n    pass\n",
        "source_path": "tool.py",
        "ask": "Add `import json` after the existing imports",
        "intent_keywords": ["import json"],
        "intent_anti_keywords": [],
        "expected_line_changes": 1,
    },
    {
        "shape": "change_literal",
        "source": "PORT = 8080\nHOST = 'localhost'\nDEBUG = False\n",
        "source_path": "config.py",
        "ask": "Change PORT from 8080 to 9090",
        "intent_keywords": ["PORT = 9090"],
        "intent_anti_keywords": ["PORT = 8080"],
        "expected_line_changes": 1,
    },
    {
        "shape": "delete_line",
        "source": "step 1: prepare\nstep 2: deprecated do not use\nstep 3: execute\nstep 4: cleanup\n",
        "source_path": "plan.txt",
        "ask": "Delete the line that says 'deprecated do not use'",
        "intent_keywords": ["step 1", "step 3", "step 4"],
        "intent_anti_keywords": ["deprecated do not use"],
        "expected_line_changes": 1,
    },
    {
        "shape": "fix_typo",
        "source": "def calculate_avarage(values):\n    return sum(values) / len(values)\n\nprint(calculate_avarage([1,2,3]))\n",
        "source_path": "stats.py",
        "ask": "Fix the typo: rename `calculate_avarage` to `calculate_average`",
        "intent_keywords": ["def calculate_average", "print(calculate_average"],
        "intent_anti_keywords": ["calculate_avarage"],
        "expected_line_changes": 2,
    },
    {
        "shape": "add_function",
        "source": "def greet(name):\n    return f'Hello, {name}'\n\nprint(greet('world'))\n",
        "source_path": "greet.py",
        "ask": "Add a function `farewell(name)` that returns `f'Goodbye, {name}'` after the existing greet function",
        "intent_keywords": ["def farewell(name)", "return f'Goodbye, {name}'"],
        "intent_anti_keywords": [],
        "expected_line_changes": 3,
    },
    {
        "shape": "update_string",
        "source": "// version 1.2.3\nconst VERSION = '1.2.3';\nexport default VERSION;\n",
        "source_path": "version.js",
        "ask": "Bump VERSION from 1.2.3 to 1.3.0 (update both the comment and the const)",
        "intent_keywords": ["// version 1.3.0", "VERSION = '1.3.0'"],
        "intent_anti_keywords": ["1.2.3"],
        "expected_line_changes": 2,
    },
    {
        "shape": "change_default",
        "source": "fn connect(host: &str, port: u16) -> Result<()> {\n    let timeout = 30;\n    do_connect(host, port, timeout)\n}\n",
        "source_path": "net.rs",
        "ask": "Change the default timeout from 30 to 60 seconds",
        "intent_keywords": ["let timeout = 60"],
        "intent_anti_keywords": ["let timeout = 30"],
        "expected_line_changes": 1,
    },
    {
        "shape": "add_field",
        "source": "{\n  \"name\": \"alice\",\n  \"age\": 30\n}\n",
        "source_path": "user.json",
        "ask": "Add an `email` field with value `alice@example.com`",
        "intent_keywords": ["alice@example.com"],
        "intent_anti_keywords": [],
        "expected_line_changes": 2,  # often involves adding a comma too
    },
    {
        "shape": "swap_lines",
        "source": "first()\nsecond()\nthird()\nfourth()\n",
        "source_path": "ops.py",
        "ask": "Swap the order of `second()` and `third()` so third runs before second",
        "intent_keywords": ["third()\nsecond()"],
        "intent_anti_keywords": [],
        "expected_line_changes": 2,
    },
    {
        "shape": "comment_out",
        "source": "import os\nimport sys\nimport requests\nimport json\n",
        "source_path": "imports.py",
        "ask": "Comment out `import requests` (prefix with `# `) — keep the other imports",
        "intent_keywords": ["# import requests"],
        "intent_anti_keywords": [],
        "expected_line_changes": 1,
    },
    {
        "shape": "modify_loop",
        "source": "for i in range(10):\n    print(i)\n",
        "source_path": "loop.py",
        "ask": "Change the loop bound from 10 to 100",
        "intent_keywords": ["range(100)"],
        "intent_anti_keywords": ["range(10)"],
        "expected_line_changes": 1,
    },
]

SYSTEM_PROMPT_TMPL = """You are a coding agent. The user will describe an edit to a file. Respond with a unified diff that, when applied with `patch -p1`, makes the requested change.

Strict format requirements (responses violating these are rejected):
- Output ONLY the unified diff — no prose preamble, no trailing commentary.
- Start with `--- a/<path>` and `+++ b/<path>`.
- Include `@@ -A,B +A,B @@` hunk markers.
- Include 1–3 lines of context above and below each change."""


def make_prompts(edit: dict, n_variants: int) -> list[dict]:
    out = []
    for i in range(n_variants):
        # Slight ask-rewordings to avoid eval-set mimicry.
        ask = edit["ask"]
        if i == 1:
            ask = ask.replace("Change", "Update").replace("Add", "Insert").replace("Delete", "Remove")
        if i == 2:
            ask = "Please " + ask[0].lower() + ask[1:]
        prompt = {
            "id": f"{edit['shape']}-v{i}",
            "source": edit["source"],
            "source_path": edit["source_path"],
            "intent_keywords": edit["intent_keywords"],
            "intent_anti_keywords": edit["intent_anti_keywords"],
            "expected_line_changes": edit["expected_line_changes"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_TMPL},
                {"role": "user", "content": f"File `{edit['source_path']}`:\n```\n{edit['source']}```\n\nTask: {ask}"},
                {"role": "assistant", "content": "<dummy>"},
            ],
        }
        out.append(prompt)
    return out


def main() -> None:
    workdir = Path(__file__).resolve().parent
    train_prompts: list[dict] = []
    eval_prompts: list[dict] = []
    for edit in EDITS:
        variants = make_prompts(edit, n_variants=3)
        random.shuffle(variants)
        train_prompts.extend(variants[:2])
        eval_prompts.extend(variants[2:])
    # Pad eval to 30
    while len(eval_prompts) < 30:
        eval_prompts.append(make_prompts(random.choice(EDITS), n_variants=1)[0])
    random.shuffle(train_prompts)
    random.shuffle(eval_prompts)

    (workdir / "prompts").mkdir(exist_ok=True)
    (workdir / "datasets").mkdir(exist_ok=True)
    with (workdir / "prompts/h1-r16-6ep.jsonl").open("w") as f:
        for p in train_prompts:
            f.write(json.dumps(p) + "\n")
    with (workdir / "datasets/train.opd.jsonl").open("w") as f:
        for p in train_prompts:
            f.write(json.dumps(p) + "\n")
    with (workdir / "datasets/eval.jsonl").open("w") as f:
        for p in eval_prompts:
            f.write(json.dumps(p) + "\n")
    print(f"train: {len(train_prompts)} prompts")
    print(f"eval:  {len(eval_prompts)} prompts")
    print(f"edits: {len(EDITS)} distinct shapes")


if __name__ == "__main__":
    main()
