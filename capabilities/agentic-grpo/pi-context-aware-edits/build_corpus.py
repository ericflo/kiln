"""Task-corpus builder for pi-context-aware-edits.

Each task is a small workspace with a *consistent style* across all
files, plus an edit request. The rubric measures whether the agent's
edit preserves the style.

Convention categories tracked:
  1. import_style    — top alphabetical vs grouped
  2. naming_case     — snake_case / camelCase / PascalCase
  3. error_handling  — try/except, Result, panic, unwrap
  4. logging_style   — print, logging.X, structured
  5. comment_style   — docstrings, inline, minimal
  6. type_annotations — present / absent / partial / strict

Output:
  datasets/train.tasks.jsonl  (30 tasks across mixed conventions)
  datasets/eval.tasks.jsonl   (18 held-out tasks)
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SEED = int(os.environ.get("SEED", 31417))


# Style profiles per workspace. Each profile names a consistent convention
# across all files.
PROFILES = [
    {
        "name": "py_strict_typed_snake",
        "language": "python",
        "naming_case": "snake_case",
        "type_annotations": "strict",
        "logging_style": "logging",
        "comment_style": "docstrings",
        "import_style": "top_alphabetical",
        "error_handling": "try_except",
    },
    {
        "name": "py_camel_loose",
        "language": "python",
        "naming_case": "camelCase",
        "type_annotations": "absent",
        "logging_style": "print",
        "comment_style": "inline",
        "import_style": "top_grouped",
        "error_handling": "raise_only",
    },
    {
        "name": "rust_snake_result",
        "language": "rust",
        "naming_case": "snake_case",
        "type_annotations": "strict",
        "logging_style": "log_crate",
        "comment_style": "docstrings",
        "import_style": "top_alphabetical",
        "error_handling": "result",
    },
    {
        "name": "go_camel_pascal",
        "language": "go",
        "naming_case": "PascalCase_pub_camel_priv",
        "type_annotations": "strict",
        "logging_style": "log_package",
        "comment_style": "godoc",
        "import_style": "grouped",
        "error_handling": "explicit_err",
    },
]


def _t_add_function_python_strict(profile, idx) -> dict:
    return {
        "task_id": f"{profile['name']}_add_fn_{idx:03d}",
        "profile": profile,
        "init_files": {
            "lib/parse.py": (
                "\"\"\"Date parsing helpers.\"\"\"\n"
                "from __future__ import annotations\n"
                "import datetime as dt\n"
                "import logging\n\n"
                "logger = logging.getLogger(__name__)\n\n\n"
                "def parse_iso(s: str) -> dt.datetime:\n"
                "    \"\"\"Parse an ISO 8601 timestamp string.\"\"\"\n"
                "    try:\n"
                "        return dt.datetime.fromisoformat(s)\n"
                "    except ValueError:\n"
                "        logger.exception(\"failed to parse %r\", s)\n"
                "        raise\n"
            ),
        },
        "prompt": (
            "Add a `parse_unix(timestamp: int) -> datetime.datetime` function "
            "to `lib/parse.py`. It converts a Unix timestamp (seconds since "
            "epoch) to a UTC datetime. Match the existing style of this file."
        ),
        "expected_conventions_in_edit": {
            "naming_case": "snake_case",
            "type_annotations": "strict",
            "logging_style": "logging",
            "error_handling": "try_except",
        },
        "gold_state_predicate": "import + function callable",
        "verify_cmd": "python3 -c 'from lib.parse import parse_unix; import datetime; assert parse_unix(0).year == 1970'",
    }


def _t_add_function_python_loose(profile, idx) -> dict:
    return {
        "task_id": f"{profile['name']}_add_fn_{idx:03d}",
        "profile": profile,
        "init_files": {
            "lib/parse.py": (
                "import datetime\n\n"
                "def parseIso(s):\n"
                "    # parse iso string\n"
                "    try:\n"
                "        return datetime.datetime.fromisoformat(s)\n"
                "    except:\n"
                "        print('failed to parse', s)\n"
                "        raise\n"
            ),
        },
        "prompt": (
            "Add a `parseUnix(ts)` function to `lib/parse.py`. It should "
            "convert a Unix timestamp (seconds since epoch) to a datetime. "
            "Match the existing style."
        ),
        "expected_conventions_in_edit": {
            "naming_case": "camelCase",
            "type_annotations": "absent",
            "logging_style": "print",
            "error_handling": "raise_only",
        },
        "gold_state_predicate": "function callable",
        "verify_cmd": "python3 -c 'from lib.parse import parseUnix; import datetime; assert parseUnix(0).year == 1970'",
    }


def _t_add_function_rust(profile, idx) -> dict:
    return {
        "task_id": f"{profile['name']}_add_fn_{idx:03d}",
        "profile": profile,
        "init_files": {
            "src/parse.rs": (
                "use chrono::DateTime;\n"
                "use chrono::Utc;\n\n"
                "/// Parse an ISO 8601 timestamp.\n"
                "pub fn parse_iso(s: &str) -> Result<DateTime<Utc>, chrono::ParseError> {\n"
                "    let dt: DateTime<Utc> = s.parse()?;\n"
                "    Ok(dt)\n"
                "}\n"
            ),
        },
        "prompt": (
            "Add a `parse_unix(ts: i64) -> Result<DateTime<Utc>, ...>` "
            "function to `src/parse.rs`. Match the existing style "
            "(snake_case, Result return, doc comment)."
        ),
        "expected_conventions_in_edit": {
            "naming_case": "snake_case",
            "type_annotations": "strict",
            "error_handling": "result",
            "comment_style": "docstrings",
        },
        "gold_state_predicate": "compiles",
        "verify_cmd": "rustc --crate-type lib src/parse.rs -o /dev/null 2>&1 | grep -q . && exit 1 || exit 0",
    }


def _t_add_function_go(profile, idx) -> dict:
    return {
        "task_id": f"{profile['name']}_add_fn_{idx:03d}",
        "profile": profile,
        "init_files": {
            "parse.go": (
                "package parse\n\n"
                "import (\n\t\"time\"\n)\n\n"
                "// ParseISO parses an ISO 8601 timestamp string.\n"
                "func ParseISO(s string) (time.Time, error) {\n"
                "\tt, err := time.Parse(time.RFC3339, s)\n"
                "\tif err != nil {\n"
                "\t\treturn time.Time{}, err\n"
                "\t}\n"
                "\treturn t, nil\n"
                "}\n"
            ),
        },
        "prompt": (
            "Add a `ParseUnix(ts int64) (time.Time, error)` function to "
            "`parse.go`. Match the existing Go style (PascalCase exported, "
            "explicit error returns, godoc comment)."
        ),
        "expected_conventions_in_edit": {
            "naming_case": "PascalCase_pub_camel_priv",
            "error_handling": "explicit_err",
            "comment_style": "godoc",
        },
        "gold_state_predicate": "compiles",
        "verify_cmd": "go build ./...",
    }


GENERATORS = {
    "py_strict_typed_snake": _t_add_function_python_strict,
    "py_camel_loose": _t_add_function_python_loose,
    "rust_snake_result": _t_add_function_rust,
    "go_camel_pascal": _t_add_function_go,
}


def main():
    DATASETS.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    train, eval_ = [], []
    for profile in PROFILES:
        gen = GENERATORS[profile["name"]]
        for i in range(8):
            train.append(gen(profile, i))
        for i in range(4):
            eval_.append(gen(profile, 100 + i))
    # Skip Go tasks when go isn't available — but commit them so the
    # corpus is reproducible; the oracle just won't run them.
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
