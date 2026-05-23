#!/usr/bin/env bash
# stage_2_strict_prompt.sh — prompted-ceiling diagnostic for pi-code-comprehension.
#
# Per pi-faithful-completion pattern (stage 1: prompting ceiling), this runs
# 3-seed eval with a strict pi user prompt that explicitly invokes every
# rubric component (grounding, cross_file, invariant_coverage, format).
# If the strict-prompt lift is ≥ +0.10 vs no-prompt base, the recipe for
# stage 2+ is "bake the strict behavior into weights via SFT chain."
#
# How: temporarily replaces task_scaffold.PI_PROMPT_TEMPLATE with a strict
# version, runs rollouts via pi (multi-turn), restores the original.
set -euo pipefail
cd /workspace/kiln/capabilities/caps/pi-code-comprehension

ROOT="/workspace/iter0"
mkdir -p "$ROOT/strict_prompt"

# 1. Backup current task_scaffold.py
cp task_scaffold.py task_scaffold.py.original

# 2. Install strict-prompt variant
cat > task_scaffold.py <<'STRICTSCAFFOLD'
"""STRICT variant of task_scaffold.py — emits a rubric-aware prompt that
explicitly invokes every scoring component to find the prompted ceiling.

Per pi-faithful-completion §"Strict prompt" experiment: an inference-time
prompt that names the scoring rubric inline often produces large lifts on
4B agentic capabilities, because the model can't otherwise infer what
'good' looks like from in-context signals alone.
"""

from __future__ import annotations
import json, os, sys
from pathlib import Path


def init_workdir(task: dict, dir: str) -> None:
    workdir = Path(dir)
    workdir.mkdir(parents=True, exist_ok=True)
    files = task.get("files") or {}
    for rel_path, content in files.items():
        if not isinstance(rel_path, str) or not isinstance(content, str):
            continue
        rel = rel_path.lstrip("/").replace("\\", "/")
        if ".." in rel.split("/"):
            continue
        dest = workdir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    target_file = task.get("target_file", "")
    target_symbol = task.get("target_symbol", "")
    readme = (
        "# Code comprehension task\n\n"
        f"Target file:   `{target_file}`\n"
        f"Target symbol: `{target_symbol}`\n\n"
        "Goal: produce a structured JSON summary of the target symbol's\n"
        "inputs, returns, mutations, calls, callers, invariants and side\n"
        "effects, with line-number citations from the source.\n\n"
        "See the user prompt for the exact JSON schema.\n"
    )
    (workdir / "README.md").write_text(readme)


PI_PROMPT_TEMPLATE = """You are an EXTREMELY METHODICAL code-comprehension assistant.
Your sole goal is to produce a RUBRIC-PERFECT structured JSON summary of
the target symbol.

Target file:   {target_file}
Target symbol: {target_symbol}

MANDATORY PROCEDURE — do not skip steps:
1. `read` the target file IN FULL. Note exact line numbers of the symbol
   definition, every parameter, every return, every call site.
2. Run `bash` with `grep -rn '{target_symbol}' .` — find EVERY cross-file
   caller. Do NOT answer until you have grepped. Empty caller list MUST
   be empirically confirmed by grep, not assumed.
3. `read` 1-2 caller files to confirm caller name and line.
4. Re-read the target function and enumerate IMPLICIT invariants
   (lock held, init called first, sorted input, etc.) — not just
   docstring text.

OUTPUT RULES (the rubric scores these — violate ANY and the score drops):
- Cite ALL line numbers from real source. Tolerance is ±2. NEVER cite
  line 1 unless the answer is genuinely on line 1.
- `inputs`: every parameter with (name, type, source_line). Types like
  `list[dict]` / `Tokenizer` / `str` — match source casing.
- `returns`: every return type with source_line.
- `mutates`: list of `arg:NAME` / `filesystem:PATH` / `global:NAME` tags.
  Use exact tag prefix. Empty list `[]` if pure.
- `calls`: every helper called with (name, file, line). Same-file calls
  use the target_file basename.
- `called_by`: cross-file callers ONLY. `(file, line)` for each.
  EMPTY LIST `[]` if grep showed zero — that is honest abstention and
  scores BETTER than fake callers.
- `invariants`: IMPLICIT preconditions (e.g. "messages must be non-empty",
  "tokenizer must be initialized"). Paraphrases score on semantic match.
- `side_effects`: raises / log / I/O / network. Empty `[]` if pure.

EMIT ALL 7 FIELDS. Empty fields use `[]`. Format must be parseable JSON.

When the investigation is complete, emit a PLAIN ASSISTANT TEXT MESSAGE
(no tool calls) containing this exact form:

<answer>
{{"inputs": [...], "returns": [...], "mutates": [...], "calls": [...], "called_by": [...], "invariants": [...], "side_effects": [...]}}
</answer>

Rules:
- DO NOT call any tool named `answer` — there is no such tool. Use a
  plain assistant text turn instead.
- DO NOT pass the JSON via `write` or `bash` — emit it as ordinary text.
- ONE `<answer>` block. End the session after it.
"""


def pi_prompt(task: dict) -> str:
    return PI_PROMPT_TEMPLATE.format(
        target_file=task.get("target_file", "<unknown>"),
        target_symbol=task.get("target_symbol", "<unknown>"),
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: task_scaffold.py <task.json> <dir>", file=sys.stderr)
        sys.exit(2)
    task = json.loads(Path(sys.argv[1]).read_text())
    init_workdir(task, sys.argv[2])
    print(pi_prompt(task))
STRICTSCAFFOLD

# 3. Run 3-seed eval with strict prompt (no adapter — measuring base ceiling).
echo "=== BASE+STRICT-PROMPT 3-seed eval ==="
for s in 1 2 3; do
  outdir="$ROOT/strict_prompt/seed-$s"
  mkdir -p "$outdir"
  echo "--- seed $s -> $outdir ---"
  KILN_URL=http://localhost:8420 PI_BIN=/usr/bin/pi \
  python3 rollout.py \
    --tasks datasets/eval.tasks.jsonl \
    --out-dir "$outdir" \
    --mode eval \
    --num-generations 1 \
    --kiln-url http://localhost:8420 \
    --max-wall-clock-s 180 \
    --concurrency 4 \
    --seed $((100 + s)) \
    --adapter "" \
    --verbose 2>&1 | tail -25 | tee "$outdir/rollout-tail.log"
  jq '.mean_composite, .mean_outcome, .mean_grounding, .mean_cross_file_caller_recall' "$outdir/summary.json" 2>/dev/null || true
done

# 4. Restore original
mv task_scaffold.py.original task_scaffold.py

# 5. Aggregate + paired lift
python3 - "$ROOT" <<'PY'
import json, sys, statistics
from pathlib import Path
root = Path(sys.argv[1])
seeds = [1, 2, 3]

def load_arm(arm):
    out = []
    for s in seeds:
        p = root / arm / f"seed-{s}" / "summary.json"
        if p.exists():
            out.append(json.load(open(p)))
    return out

base = load_arm("base")
strict = load_arm("strict_prompt")
if not base or not strict:
    print(f"missing arm; base={len(base)} strict={len(strict)}")
    sys.exit(0)
base_comps = [s["mean_composite"] for s in base]
strict_comps = [s["mean_composite"] for s in strict]
paired = [a - b for a, b in zip(strict_comps, base_comps)]
out = {
    "arm": "strict_prompt vs base",
    "base_3seed_mean": statistics.mean(base_comps),
    "strict_3seed_mean": statistics.mean(strict_comps),
    "paired_lifts": paired,
    "paired_lift_mean": statistics.mean(paired),
    "paired_lift_stdev": statistics.stdev(paired) if len(paired) > 1 else 0,
    "sigma_above_zero": statistics.mean(paired) / max(statistics.stdev(paired) if len(paired) > 1 else 1e-9, 1e-9),
    "ceiling_recipe_unlocked": statistics.mean(paired) >= 0.10,
}
out_path = root / "strict-vs-base-paired.json"
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

echo DONE > "$ROOT/stage_2_strict_prompt.done"
