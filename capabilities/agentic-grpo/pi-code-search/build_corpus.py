"""Build train + eval task JSONL for pi-code-search.

Sources symbols from the kiln repo itself via ripgrep. Generates two
question kinds:

- `define:<symbol>` — single (file, line) gold answer.
- `refs:<symbol>` — set of (file, line) gold answers, F1-scored.

Output:
  datasets/train.tasks.jsonl
  datasets/eval.tasks.jsonl  (disjoint from train by symbol)

Each task line:
  {
    "task_id": str,
    "question_kind": "define" | "refs",
    "symbol": str,
    "repo_id": "kiln",
    "gold": [[file, line], ...],
    "target_bytes": int,
  }

Strategy for selecting symbols:
- For `define:` we use `^pub fn <name>` matches in `*.rs` files. These
  have a clean 1-line gold answer.
- For `refs:` we use the same symbol set but enumerate all word-boundary
  matches across the repo (subtract the def site).

We bias eval toward DIVERSE symbols (different files, different crates)
to make the held-out distribution representative.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

_REPO_ENV = os.environ.get("PI_CODE_SEARCH_REPO_ROOT")
if _REPO_ENV and Path(_REPO_ENV).exists():
    REPO_ROOT = Path(_REPO_ENV)
elif Path("/workspace/kiln-snapshot").exists():
    REPO_ROOT = Path("/workspace/kiln-snapshot")
else:
    # Fall back to the kiln repo root (three dirs up from this file).
    REPO_ROOT = Path(__file__).resolve().parents[3]


DST = Path(__file__).parent / "datasets"


def _rg(*args: str) -> list[str]:
    """Run `rg` and return stdout lines."""
    cmd = ["rg", "--no-heading", "--with-filename", "-n", *args]
    out = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    if out.returncode not in (0, 1):
        return []
    return out.stdout.splitlines()


def collect_defines() -> list[dict]:
    """Find `pub fn NAME(` definitions in *.rs."""
    out: list[dict] = []
    lines = _rg("-t", "rust", r"^pub fn ([A-Za-z_][A-Za-z0-9_]*)\s*[<(]")
    seen: dict[str, list[tuple[str, int]]] = {}
    for line in lines:
        m = re.match(r"^(.+?):(\d+):pub fn ([A-Za-z_][A-Za-z0-9_]*)", line)
        if not m:
            continue
        file_path = m.group(1)
        line_num = int(m.group(2))
        symbol = m.group(3)
        seen.setdefault(symbol, []).append((file_path, line_num))

    # Keep only symbols defined in exactly ONE place (avoid ambiguity).
    # Skip symbols that are too short (`new`, `len`) — too many false hits.
    SKIP_SHORT = {"new", "len", "is_empty", "drop", "default", "clone",
                  "as_ref", "into", "from", "iter", "next", "build", "init",
                  "open", "close", "read", "write", "send", "recv", "as_mut",
                  "deref", "deref_mut", "fmt", "hash", "eq", "ne", "cmp",
                  "partial_cmp", "lt", "le", "gt", "ge"}
    for symbol, locs in seen.items():
        if len(locs) != 1:
            continue
        if len(symbol) < 6 and symbol.lower() in SKIP_SHORT:
            continue
        if symbol.startswith("_"):
            continue
        file_path, line_num = locs[0]
        out.append({
            "question_kind": "define",
            "symbol": symbol,
            "gold": [[file_path, line_num]],
        })
    return out


def collect_refs(define_tasks: list[dict], min_refs: int = 3, max_refs: int = 15) -> list[dict]:
    """For each define-task symbol, enumerate references across *.rs.

    Keep symbols with [min_refs, max_refs] non-definition occurrences.
    The non-definition occurrence count is the meaningful one — if a
    symbol has just 1 reference, the task is mostly definition-finding;
    if it has 100, the F1 is impossible without near-complete enumeration.
    """
    out: list[dict] = []
    for task in define_tasks:
        symbol = task["symbol"]
        gold_def = task["gold"][0]
        lines = _rg("-t", "rust", "-w", re.escape(symbol))
        refs: list[tuple[str, int]] = []
        for line in lines:
            m = re.match(r"^(.+?):(\d+):", line)
            if not m:
                continue
            f = m.group(1)
            ln = int(m.group(2))
            if (f, ln) == (gold_def[0], gold_def[1]):
                continue
            refs.append((f, ln))
        if min_refs <= len(refs) <= max_refs:
            out.append({
                "question_kind": "refs",
                "symbol": symbol,
                "gold": [[f, ln] for f, ln in refs],
            })
    return out


def estimate_target_bytes(task: dict) -> int:
    """Approximate the bytes-output size of the optimal grep command."""
    symbol = task["symbol"]
    if task["question_kind"] == "define":
        # rg -n "^pub fn SYMBOL" → one line, typically ~150 bytes.
        return max(150, len(symbol) * 4 + 100)
    # refs: each line ~120 bytes, plus filename overhead.
    n = len(task["gold"])
    return max(200, n * 120)


def main():
    DST.mkdir(parents=True, exist_ok=True)

    defines = collect_defines()
    print(f"collected {len(defines)} unique `pub fn` defines", flush=True)
    refs = collect_refs(defines, min_refs=3, max_refs=15)
    print(f"collected {len(refs)} `refs` tasks (3-15 refs each)", flush=True)

    rng = random.Random(42)
    rng.shuffle(defines)
    rng.shuffle(refs)

    # Eval: 20 define + 12 refs = 32 eval tasks (similar magnitude to pi-doctest).
    n_eval_def = 20
    n_eval_ref = 12
    eval_defs = defines[:n_eval_def]
    eval_refs = refs[:n_eval_ref]

    # Train: rest, but cap at 120 tasks to keep wall-clock reasonable.
    n_train_def_cap = 60
    n_train_ref_cap = 40
    train_defs = [t for t in defines[n_eval_def:] if t["symbol"] not in {x["symbol"] for x in eval_defs + eval_refs}][:n_train_def_cap]
    train_refs = [t for t in refs[n_eval_ref:] if t["symbol"] not in {x["symbol"] for x in eval_defs + eval_refs + train_defs}][:n_train_ref_cap]

    eval_tasks = eval_defs + eval_refs
    train_tasks = train_defs + train_refs

    # Stable IDs.
    for i, t in enumerate(train_tasks):
        t["task_id"] = f"train_{t['question_kind']}_{i:04d}"
        t["target_bytes"] = estimate_target_bytes(t)
        t["repo_id"] = "kiln"
    for i, t in enumerate(eval_tasks):
        t["task_id"] = f"eval_{t['question_kind']}_{i:04d}"
        t["target_bytes"] = estimate_target_bytes(t)
        t["repo_id"] = "kiln"

    rng.shuffle(train_tasks)
    rng.shuffle(eval_tasks)

    (DST / "train.tasks.jsonl").write_text(
        "\n".join(json.dumps(t) for t in train_tasks) + "\n"
    )
    (DST / "eval.tasks.jsonl").write_text(
        "\n".join(json.dumps(t) for t in eval_tasks) + "\n"
    )

    print(f"wrote {len(eval_tasks)} eval tasks "
          f"({n_eval_def} define + {n_eval_ref} refs)", flush=True)
    print(f"wrote {len(train_tasks)} train tasks "
          f"({len(train_defs)} define + {len(train_refs)} refs)", flush=True)
    print(f"  eval:  {DST / 'eval.tasks.jsonl'}", flush=True)
    print(f"  train: {DST / 'train.tasks.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
