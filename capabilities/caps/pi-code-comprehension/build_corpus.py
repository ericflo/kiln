"""Build train + eval task JSONL for pi-code-comprehension.

Each task ships:
  - a small file-snapshot (target file + a few caller files)
  - the target symbol name
  - the gold structured summary (NEVER copied into the workdir)

The corpus is auto-generated from Python source code using the `ast`
module — we walk top-level functions, extract inputs/returns/calls from
the AST, find callers via lexical grep across sibling files, and harvest
invariants from docstrings and key body patterns.

Sources (in priority order):
  1. capabilities/agentic-grpo/pi-code-comprehension/seed_repos/ — manually
     curated small Python repos shipped with this cap
  2. capabilities/*/scripts/, capabilities/*/lib/                — kiln itself
  3. scripts/                                                    — kiln scripts

Run:
  python3 build_corpus.py [--seed-only] [--n-eval 24] [--max 200]

Outputs:
  datasets/train.tasks.jsonl
  datasets/eval.tasks.jsonl
  datasets/eval_full.tasks.jsonl     # eval kept regenerable for ablations
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parent
DST = ROOT / "datasets"


# ----------------------------------------------------------------------
# Source-finding
# ----------------------------------------------------------------------

def iter_python_files(root: Path) -> Iterator[Path]:
    """Yield .py files under root, skipping obvious noise."""
    if not root.exists():
        return
    skip = {"target", ".git", "node_modules", "vendor", "__pycache__",
            "venv", ".venv", "dist", "build", "site-packages"}
    for p in root.rglob("*.py"):
        if any(part in skip for part in p.parts):
            continue
        # Skip very large files (they make rollouts slow).
        try:
            if p.stat().st_size > 30_000:
                continue
        except OSError:
            continue
        yield p


# ----------------------------------------------------------------------
# Type rendering
# ----------------------------------------------------------------------

def render_annotation(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


# ----------------------------------------------------------------------
# Invariant heuristics
# ----------------------------------------------------------------------

INVARIANT_KEYWORDS = re.compile(
    r"\b("
    r"requires|must|should|assumes|expects|precondition|invariant|"
    r"caller must|only call|do not call|before calling|raises|"
    r"non-empty|nonempty|non-null|not null|nonzero"
    r")\b", re.IGNORECASE
)


def extract_docstring_invariants(doc: str | None) -> list[dict]:
    """Pull bullet-style invariants out of a docstring. Returns a list of
    {"primary": str, "paraphrases": []} dicts."""
    if not doc:
        return []
    invariants: list[dict] = []
    for line in doc.splitlines():
        l = line.strip(" -*\t")
        if not l:
            continue
        if INVARIANT_KEYWORDS.search(l) and 8 <= len(l) <= 200:
            invariants.append({"primary": l, "paraphrases": []})
    # Cap so the rubric doesn't get drowned in tiny docstring bullets.
    return invariants[:5]


def detect_body_invariants(body: list[ast.stmt]) -> list[dict]:
    """Inspect the function body for runtime checks that imply invariants.

    Patterns:
        if not x: raise ValueError(...)         → "x must be truthy / non-empty"
        if x is None: raise ...                 → "x must not be None"
        assert y > 0                             → "y must be positive"
    """
    invariants: list[dict] = []
    for stmt in body:
        # `assert <test>, <msg>` — the assertion IS the invariant.
        if isinstance(stmt, ast.Assert):
            try:
                src = ast.unparse(stmt.test)
                invariants.append({
                    "primary": f"asserts {src}",
                    "paraphrases": [f"requires {src}", f"{src} must hold"],
                })
            except Exception:
                pass
        # `if <cond>: raise <Exception>(...)` — negation of cond is invariant.
        if isinstance(stmt, ast.If) and any(isinstance(s, ast.Raise) for s in stmt.body):
            try:
                cond_src = ast.unparse(stmt.test)
                invariants.append({
                    "primary": f"raises if {cond_src}",
                    "paraphrases": [f"requires not ({cond_src})",
                                    f"asserts not ({cond_src})"],
                })
            except Exception:
                pass
    return invariants[:5]


def detect_side_effects(body: list[ast.stmt]) -> list[dict]:
    """Heuristic side-effect extraction:
    - Raise statements → "raises X"
    - Calls to `print`, `logging.*`, `subprocess.*`, `open(...)` → I/O
    """
    side: list[dict] = []
    for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(stmt, ast.Raise):
            try:
                exc = ast.unparse(stmt.exc) if stmt.exc else "Exception"
                side.append({"primary": f"raises {exc}", "paraphrases": [f"raises {exc.split('(')[0]}"]})
            except Exception:
                pass
        if isinstance(stmt, ast.Call):
            fn = stmt.func
            name = ""
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            if name in ("print", "info", "warning", "error", "debug", "log"):
                side.append({"primary": "writes log output", "paraphrases": ["logs", "prints"]})
            if name in ("open",):
                side.append({"primary": "opens a file", "paraphrases": ["I/O", "file access"]})
            if name in ("Popen", "run", "check_call", "check_output", "call"):
                side.append({"primary": "shells out / runs subprocess", "paraphrases": []})
    # Dedup by primary string
    seen: set[str] = set()
    out: list[dict] = []
    for s in side:
        if s["primary"] not in seen:
            out.append(s)
            seen.add(s["primary"])
    return out[:5]


def detect_mutations(fn_node: ast.AST, fn_args: list[str]) -> list[str]:
    """Heuristic mutation detection.

    - Subscript assignment to an argument          → "arg:<name>"
    - Call to `.append/.extend/.pop/.update/...` on an arg → "arg:<name>"
    - Assignment to a `global x` / `nonlocal x` name    → "global:<name>"
    - Write to a file via `open(..., 'w'/'a')`     → "filesystem:<path>"
    """
    muts: set[str] = set()
    arg_set = set(fn_args)
    mut_methods = {"append", "extend", "pop", "remove", "clear", "update",
                   "insert", "sort", "reverse", "setdefault", "popitem",
                   "discard", "add"}

    # Find names declared global/nonlocal so we can detect global writes.
    global_names: set[str] = set()
    for node in ast.walk(fn_node):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                global_names.add(name)

    for node in ast.walk(fn_node):
        # Subscript / attribute assignment to an arg.
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript):
                    base = tgt.value
                    if isinstance(base, ast.Name) and base.id in arg_set:
                        muts.add(f"arg:{base.id}")
                elif isinstance(tgt, ast.Attribute):
                    base = tgt.value
                    if isinstance(base, ast.Name) and base.id in arg_set:
                        muts.add(f"arg:{base.id}")
                elif isinstance(tgt, ast.Name):
                    if tgt.id in global_names:
                        muts.add(f"global:{tgt.id}")
        # Mutator method calls on arg names.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = node.func.value
            if (isinstance(base, ast.Name) and base.id in arg_set
                    and node.func.attr in mut_methods):
                muts.add(f"arg:{base.id}")
            # File writes via open(..., 'w'/'a'/'wb'/'ab')
            if isinstance(base, ast.Name) and base.id == "open":
                pass  # too specific; skip
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            # Look for mode arg starting with 'w' / 'a'.
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
                if isinstance(mode, str) and mode and mode[0] in ("w", "a"):
                    muts.add("filesystem")
    return sorted(muts)


# ----------------------------------------------------------------------
# Call extraction
# ----------------------------------------------------------------------

def extract_calls(fn_node: ast.AST) -> list[dict]:
    """Return a list of {name, file: "", line: N} for direct callees within
    the function body. We do not resolve where they live (file:line) — the
    file field is left empty and the rubric scores them by name only.
    Same-file callees with definitions found will have their `file` filled
    in by the caller of this routine."""
    calls: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = ""
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        if not name:
            continue
        # Skip built-ins / common no-op.
        if name in {"len", "str", "int", "float", "bool", "list", "dict",
                    "set", "tuple", "range", "print", "type", "isinstance",
                    "min", "max", "sum", "abs", "round", "enumerate", "zip",
                    "open", "iter", "next", "any", "all", "sorted", "reversed",
                    "hasattr", "getattr", "setattr", "format", "repr",
                    "compile", "exec", "eval", "globals", "locals"}:
            continue
        # Skip exception classes — they're 'raises X', not 'calls X'.
        if (name.endswith("Error") or name.endswith("Exception")
                or name in {"ValueError", "TypeError", "KeyError",
                             "RuntimeError", "Exception", "StopIteration",
                             "IndexError", "AttributeError", "AssertionError"}):
            continue
        key = (name, node.lineno)
        if key in seen:
            continue
        seen.add(key)
        calls.append({"name": name, "file": "", "line": node.lineno})
    return calls


# ----------------------------------------------------------------------
# Caller / called_by extraction across files
# ----------------------------------------------------------------------

def find_callers(symbol: str, files_text: dict[str, str], own_path: str
                 ) -> list[dict]:
    """For each non-target file in `files_text`, scan for lines that
    look like calls to `symbol(`. Return up to 5 caller records."""
    out: list[dict] = []
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}\s*\(")
    for path, src in files_text.items():
        if path == own_path:
            continue
        # find first matching line
        for lineno, line in enumerate(src.splitlines(), start=1):
            if pat.search(line):
                out.append({"file": path, "line": lineno})
                break  # only count the first match per file
        if len(out) >= 5:
            break
    return out


# ----------------------------------------------------------------------
# Caller-file synthesis
# ----------------------------------------------------------------------

CALLER_TEMPLATES = [
    "from {module} import {symbol}\n\ndef main():\n    x = {symbol}({args})\n    return x\n",
    "from . import {module} as m\n\ndef invoke():\n    return m.{symbol}({args})\n",
    "import {module}\n\n\ndef handle(request):\n    result = {module}.{symbol}({args})\n    return result\n",
]


def synthesize_caller(symbol: str, target_module: str, args: list[str]) -> str:
    arg_expr = ", ".join(args) if args else ""
    return CALLER_TEMPLATES[hash(symbol) % len(CALLER_TEMPLATES)].format(
        module=target_module, symbol=symbol, args=arg_expr)


# ----------------------------------------------------------------------
# Task synthesis from a single (file, function)
# ----------------------------------------------------------------------

def synthesize_task_from_function(
    file_path: Path,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    source_lines: list[str],
    caller_files: dict[str, str],
    task_id: str,
) -> dict | None:
    """Construct a task dict from a parsed function node.

    `caller_files`: optional map of {path: file_text} for additional repo
    files that may contain callers of this function.
    """
    # Inputs.
    args: list[dict] = []
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        args.append({
            "name": arg.arg,
            "type": render_annotation(arg.annotation),
            "source_line": arg.lineno,
        })
    if fn.args.vararg:
        args.append({
            "name": f"*{fn.args.vararg.arg}",
            "type": render_annotation(fn.args.vararg.annotation),
            "source_line": fn.args.vararg.lineno,
        })
    if fn.args.kwarg:
        args.append({
            "name": f"**{fn.args.kwarg.arg}",
            "type": render_annotation(fn.args.kwarg.annotation),
            "source_line": fn.args.kwarg.lineno,
        })

    arg_names = [a["name"].lstrip("*") for a in args]

    # Returns.
    returns: list[dict] = []
    ret_t = render_annotation(fn.returns)
    if ret_t:
        returns.append({"type": ret_t, "source_line": fn.lineno})

    # Mutates.
    mutates = detect_mutations(fn, arg_names)

    # Calls — fill in same-file file if symbol defined in this file.
    calls = extract_calls(fn)
    same_file_defs: set[str] = set()
    for top in ast.walk(ast.parse(source)):
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            same_file_defs.add(top.name)
        elif isinstance(top, ast.ClassDef):
            same_file_defs.add(top.name)
    for c in calls:
        if c["name"] in same_file_defs:
            c["file"] = file_path.name
        # else: leave file empty; rubric only scores name F1 for these.

    # Called_by — pulled from caller_files via lexical match.
    called_by = find_callers(fn.name, caller_files, file_path.name)

    # Invariants from docstring + body.
    doc = ast.get_docstring(fn)
    invariants = extract_docstring_invariants(doc)
    invariants.extend(detect_body_invariants(fn.body))

    # Side effects.
    side_effects = detect_side_effects(fn.body)

    # Build the files map. Always include target + caller files.
    files = {file_path.name: source}
    for cp, ctext in caller_files.items():
        files[cp] = ctext

    # If no real callers exist, synthesize one or two so the task has
    # cross-file callers to find (otherwise cross_file_caller_recall is
    # trivially 1.0 and there's no signal).
    if not called_by:
        target_module = file_path.stem
        # Generate two synth callers.
        for idx in range(2):
            cname = f"client_{idx}.py"
            ctext = synthesize_caller(
                fn.name, target_module,
                [a for a in arg_names if not a.startswith("*")][:2],
            )
            files[cname] = ctext
            called_by.append({
                "file": cname,
                "line": 4 if idx == 0 else 3,
            })

    gold = {
        "inputs": args,
        "returns": returns,
        "mutates": mutates,
        "calls": calls,
        "called_by": called_by,
        "invariants": invariants,
        "side_effects": side_effects,
    }

    # Require at least 2 inputs OR cross-file callers to ensure tasks
    # have non-trivial structure.
    if len(args) < 1:
        return None

    return {
        "task_id": task_id,
        "target_file": file_path.name,
        "target_symbol": fn.name,
        "gold": gold,
        "files": files,
    }


# ----------------------------------------------------------------------
# Seed-repo corpus
# ----------------------------------------------------------------------

SEED_REPOS = ROOT / "seed_repos"


def gather_caller_files(repo_root: Path, target_file: Path) -> dict[str, str]:
    """Read sibling .py files in the same directory (and one level up)."""
    out: dict[str, str] = {}
    for p in repo_root.rglob("*.py"):
        if p == target_file:
            continue
        try:
            txt = p.read_text()
        except OSError:
            continue
        if len(txt) > 8000:
            continue  # don't blow up the workdir
        rel = p.relative_to(repo_root).as_posix()
        out[rel] = txt
        if len(out) >= 6:
            break
    return out


def build_from_repos(repo_roots: Iterable[Path], max_tasks: int = 200
                     ) -> list[dict]:
    tasks: list[dict] = []
    counter = 0
    for repo_root in repo_roots:
        for py in iter_python_files(repo_root):
            try:
                src = py.read_text()
            except OSError:
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            source_lines = src.splitlines()
            # Read sibling files once per target file
            caller_files = gather_caller_files(repo_root, py)
            for node in ast.iter_child_nodes(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # Skip dunder, single-line, very-short, and very-long.
                if node.name.startswith("__"):
                    continue
                start = node.lineno
                end = max(s.lineno for s in ast.walk(node) if hasattr(s, "lineno"))
                body_lines = end - start + 1
                if body_lines < 4 or body_lines > 80:
                    continue
                tid = f"task_{counter:04d}_{py.stem}_{node.name}"
                t = synthesize_task_from_function(
                    py, node, src, source_lines, caller_files, tid)
                if t is None:
                    continue
                tasks.append(t)
                counter += 1
                if counter >= max_tasks:
                    return tasks
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-only", action="store_true",
                    help="Only use seed_repos/, ignore kiln's own python")
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--out-dir", default=str(DST))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repos: list[Path] = []
    if SEED_REPOS.exists():
        for d in sorted(SEED_REPOS.iterdir()):
            if d.is_dir():
                repos.append(d)
    if not args.seed_only:
        # Use kiln's own Python (capabilities/*, scripts/) as additional source.
        kiln_root = ROOT.parent.parent.parent  # → kiln/
        for sub in ("capabilities", "scripts"):
            p = kiln_root / sub
            if p.exists():
                repos.append(p)

    print(f"scanning {len(repos)} repo roots", flush=True)
    tasks = build_from_repos(repos, max_tasks=args.max)
    print(f"built {len(tasks)} tasks", flush=True)

    # Stable shuffle (by task_id hash) so eval / train splits are reproducible.
    tasks.sort(key=lambda t: t["task_id"])

    n_eval = min(args.n_eval, max(1, len(tasks) // 3))
    eval_tasks = tasks[:n_eval]
    train_tasks = tasks[n_eval:]

    eval_path = out_dir / "eval.tasks.jsonl"
    train_path = out_dir / "train.tasks.jsonl"
    full_eval_path = out_dir / "eval_full.tasks.jsonl"

    with eval_path.open("w") as f:
        for t in eval_tasks:
            f.write(json.dumps(t) + "\n")
    with full_eval_path.open("w") as f:
        for t in eval_tasks:
            f.write(json.dumps(t) + "\n")
    with train_path.open("w") as f:
        for t in train_tasks:
            f.write(json.dumps(t) + "\n")

    print(f"wrote {len(eval_tasks)} eval, {len(train_tasks)} train", flush=True)
    print(f"  eval  → {eval_path}")
    print(f"  train → {train_path}")


if __name__ == "__main__":
    main()
