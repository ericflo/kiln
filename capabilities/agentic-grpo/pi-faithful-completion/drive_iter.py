"""Drive one iter of pi-faithful-completion based on hypotheses.json.

Reads the hypothesis spec for the given --iter N, translates `args` to
run_iter.sh flags, picks BEST/PREV base_adapter from capability.jsonl,
filters the task corpus if requested, then invokes run_iter.sh.

This is the orchestration layer so drive_iters.sh can be a thin loop.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HYPOTHESES = ROOT / "hypotheses.json"
CAPABILITY_LOG = ROOT / "capability.jsonl"
DATASETS = ROOT / "datasets"


STRICT_PROMPT = """You are an EXTREMELY STRICT autonomous task-execution assistant.

ABSOLUTE RULES (violations cause complete failure):
  1. NEVER ask the user a question. Pick a default and proceed.
  2. NEVER use soft-punt phrases: "let me know", "should I", "your call",
     "Eric's call", "awaiting", "if you'd like", "do you want me to".
  3. The required OUTPUT FORMAT line MUST appear, with the exact characters
     specified. No paraphrasing.
  4. If the task is impossible, emit `precondition_failed: <reason>`.
     NEVER claim success when it failed.
  5. Be terse. Final OUTPUT FORMAT line first; minimal preamble (if any).
"""

LIGHT_PROMPT = """You are an autonomous assistant. Execute the task and provide a final answer in the requested OUTPUT FORMAT. If you cannot complete the task, say so honestly with `precondition_failed:`."""

MINIMAL_PROMPT = """You complete tasks. Follow the OUTPUT FORMAT."""


def load_hypothesis(iter_n: int) -> dict:
    hs = json.loads(HYPOTHESES.read_text())
    for h in hs:
        if h["iter"] == iter_n:
            return h
    raise ValueError(f"no hypothesis for iter {iter_n}")


def load_capability_rows() -> list[dict]:
    if not CAPABILITY_LOG.exists():
        return []
    out = []
    for line in CAPABILITY_LOG.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def best_adapter_so_far() -> tuple[str | None, float]:
    rows = load_capability_rows()
    best = (None, -1.0)
    for r in rows:
        comp = r.get("composite", 0.0) or 0.0
        adapter = r.get("adapter")
        if adapter and comp > best[1]:
            best = (adapter, comp)
    return best


def prev_adapter() -> str | None:
    rows = load_capability_rows()
    if not rows:
        return None
    # most recent row with a non-null adapter
    for r in reversed(rows):
        a = r.get("adapter")
        if a:
            return a
    return None


def baseline_composite() -> float | None:
    rows = load_capability_rows()
    for r in rows:
        if r.get("iter") == 0:
            return r.get("composite")
    return None


def filter_corpus(task_filter: str, src: Path, dst: Path) -> int:
    """Filter the training task corpus by tag. Returns count of tasks kept."""
    keep_kinds: set[str] | None = None
    balanced_mix = False
    fmt_diverse = False
    if task_filter == "success_only":
        keep_kinds = None  # we'll filter by is_failure=False
    elif task_filter == "failure_only":
        keep_kinds = None
    elif task_filter == "soft_punt_only":
        keep_kinds = {"soft_punt_tempting", "failure_underspecified"}
    elif task_filter == "balanced":
        balanced_mix = True
    elif task_filter == "format_diverse":
        fmt_diverse = True
    else:
        keep_kinds = None

    tasks = []
    with src.open() as f:
        for line in f:
            tasks.append(json.loads(line))

    if task_filter == "success_only":
        kept = [t for t in tasks if not t.get("is_failure")]
    elif task_filter == "failure_only":
        kept = [t for t in tasks if t.get("is_failure")]
    elif task_filter == "soft_punt_only":
        kept = [t for t in tasks if t.get("task_kind") in ("soft_punt_tempting", "failure_underspecified")]
    elif balanced_mix:
        # 50% success / 25% failure / 25% soft-punt-tempting
        success = [t for t in tasks if not t.get("is_failure") and t.get("task_kind") != "soft_punt_tempting"]
        failure = [t for t in tasks if t.get("is_failure")]
        softp   = [t for t in tasks if t.get("task_kind") == "soft_punt_tempting"]
        target_n = len(tasks)
        n_succ = target_n // 2
        n_fail = target_n // 4
        n_soft = target_n - n_succ - n_fail
        # repeat shorter pools if needed
        import itertools
        def take(pool, n):
            if not pool: return []
            cyc = itertools.cycle(pool)
            return [next(cyc) for _ in range(n)]
        kept = take(success, n_succ) + take(failure, n_fail) + take(softp, n_soft)
    elif fmt_diverse:
        # at least 2 tasks per (task_kind, format_kind) combination if possible
        seen = {}
        kept = []
        for t in tasks:
            key = (t.get("task_kind"), t.get("format_kind"))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= 3:
                kept.append(t)
    else:
        kept = tasks

    with dst.open("w") as f:
        for t in kept:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return len(kept)


def system_prompt_for_kind(kind: str) -> str:
    if kind == "strict":
        return STRICT_PROMPT
    if kind == "light":
        return LIGHT_PROMPT
    if kind == "minimal":
        return MINIMAL_PROMPT
    return ""  # use the default in task_scaffold.py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    h = load_hypothesis(args.iter)
    iter_n = h["iter"]
    slug = h["slug"]
    family = h["family"]
    a = h["args"]

    print(f"[drive_iter] iter={iter_n} slug={slug} family={family}")
    print(f"[drive_iter] args={json.dumps(a)}")

    # Resolve BEST / PREV base adapter — must pass the FULL PATH on the pod
    # to cuda_grpo_ablation; passing just the name fails because
    # cuda_grpo_ablation resolves names against the iter's OUTPUT dir,
    # not the previous iters' output dirs.
    def _adapter_path_on_pod(name: str) -> str:
        # Find the iter number from the slug by scanning capability.jsonl
        rows = load_capability_rows()
        for r in rows:
            if r.get("adapter") == name:
                it = r.get("iter")
                if it is not None:
                    return f"/tmp/iter{it}-adapter/{name}"
        # Fallback: try the active adapter dir
        return f"/workspace/qwen3.5-4b/adapters/{name}"

    base_adapter = a.get("base_adapter", "")
    if base_adapter == "BEST":
        ba, ba_comp = best_adapter_so_far()
        if ba is None:
            print("[drive_iter] no prior adapter for BEST; falling back to base", file=sys.stderr)
            base_adapter = ""
        else:
            base_adapter = _adapter_path_on_pod(ba)
            print(f"[drive_iter] BEST adapter -> {ba} (composite {ba_comp:.3f}) path={base_adapter}")
    elif base_adapter == "PREV":
        pa = prev_adapter()
        if pa is None:
            print("[drive_iter] no prior adapter for PREV; falling back to base", file=sys.stderr)
            base_adapter = ""
        else:
            base_adapter = _adapter_path_on_pod(pa)
            print(f"[drive_iter] PREV adapter -> {pa} path={base_adapter}")

    # Filter the corpus if requested
    train_tasks_file = "datasets/train.tasks.jsonl"
    task_filter = a.get("task_filter")
    if task_filter:
        filtered_path = DATASETS / f"train.{task_filter}.tasks.jsonl"
        n = filter_corpus(task_filter, DATASETS / "train.tasks.jsonl", filtered_path)
        print(f"[drive_iter] filtered to {n} tasks via filter={task_filter} → {filtered_path}")
        train_tasks_file = f"datasets/{filtered_path.name}"

    # Write the system prompt override if needed
    sp_file = ""
    sp_kind = a.get("system_prompt")
    if sp_kind:
        sp_text = system_prompt_for_kind(sp_kind)
        if sp_text:
            sp_path = ROOT / "prompts"
            sp_path.mkdir(exist_ok=True)
            sp_file = str(sp_path / f"{slug}-system.txt")
            Path(sp_file).write_text(sp_text)

    # Compose run_iter.sh command
    cmd = ["bash", str(ROOT / "run_iter.sh"),
           "--iter", str(iter_n), "--slug", slug,
           "--train-tasks", str(a.get("train_tasks", 24)),
           "--num-gens", str(a.get("num_gens", 4)),
           "--lr", str(a.get("lr", "1e-5")),
           "--rank", str(a.get("rank", 16)),
           "--alpha", str(a.get("alpha", 32)),
           "--mode", a.get("mode", "phase1"),
           "--temperature", str(a.get("temperature", "0.8")),
           "--top-p", str(a.get("top_p", "0.95")),
           "--max-tokens", str(a.get("max_tokens", "768")),
           "--filter-var", str(a.get("filter_var", "0.0")),
           "--seed", str(a.get("seed", 3141592653)),
           "--train-tasks-file", train_tasks_file]
    if a.get("no_echo"):
        cmd += ["--no-echo"]
    elif "echo_lambda" in a:
        cmd += ["--echo-lambda", str(a["echo_lambda"])]
    if a.get("no_policy_loss"):
        cmd += ["--no-policy-loss"]
    if base_adapter:
        cmd += ["--base-adapter", base_adapter]
    if "max_groups" in a:
        cmd += ["--max-groups", str(a["max_groups"])]
    if "kl_coeff" in a:
        cmd += ["--kl-coeff", str(a["kl_coeff"])]
    if "clip_eps" in a:
        cmd += ["--clip-eps", str(a["clip_eps"])]
    if sp_file:
        cmd += ["--system-prompt-file", sp_file]

    print(f"[drive_iter] cmd={' '.join(cmd)}")
    if args.dry_run:
        return 0

    # Run it
    res = subprocess.run(cmd, env=os.environ.copy())
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
