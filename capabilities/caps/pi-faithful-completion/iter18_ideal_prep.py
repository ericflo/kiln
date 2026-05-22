"""iter18: synthesize IDEAL assistant outputs from task ground truth.

Each task has format_regex + expected_value. For non-failure tasks, the
"ideal" assistant output is the format-conforming line with the correct
value. For failure tasks, it's the canonical precondition_failed line.

This is GROUND TRUTH, not a sample — every example scores 1.0 on the
rubric by construction. It also has perfect format_strict (since the
output matches the regex exactly, no extra prose).

Hypothesis: training on ideal outputs avoids the format_strict drop
that strict-prompt rollouts introduce (those rollouts sometimes have
preamble before the format line). The model learns "this is THE shape
of a correct response."
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "datasets/train.tasks.jsonl"
OUT = ROOT / "datasets/sft.ideal.jsonl"

# Build an "ideal" assistant content for each task by templating against
# the user prompt's OUTPUT FORMAT spec. Heuristics:
#   - For format_regex like '^Quick Start line count: (\d+)$', emit
#     'Quick Start line count: 7' (the regex source minus capture group,
#     with expected_value substituted).
#   - For failure tasks, emit `precondition_failed: <generic reason>`.

def extract_format_template(user_prompt: str) -> str | None:
    """Pull the literal format line from the user prompt.

    User prompts contain `OUTPUT FORMAT: ...\\n  <literal>`. Grab everything
    on the first non-empty indented line after that header.
    """
    # Find "OUTPUT FORMAT:" header
    idx = user_prompt.find("OUTPUT FORMAT:")
    if idx == -1:
        return None
    after = user_prompt[idx:]
    # Skip the header line itself
    nl = after.find("\n")
    if nl == -1:
        return None
    after = after[nl + 1 :]
    # Take leading lines that look like the format spec (indented or
    # JSON-shaped). Stop at next blank line.
    out_lines = []
    for line in after.splitlines():
        stripped = line.strip()
        if not stripped:
            if out_lines:
                break
            continue
        out_lines.append(stripped)
    if not out_lines:
        return None
    return "\n".join(out_lines)


def fill_template(template: str, value: str) -> str:
    """Substitute <value> placeholder with the expected value, preserving formatting."""
    return template.replace("<value>", value)


def main() -> int:
    n_in = n_out = n_skipped = 0
    with TASKS.open() as f, OUT.open("w") as out:
        for line in f:
            t = json.loads(line)
            n_in += 1
            sys_p = t.get("system_prompt", "")
            user_p = t["user_prompt"]
            if t.get("is_failure"):
                # canonical honest-failure response
                assistant = "precondition_failed: required information is missing or underspecified"
            else:
                tmpl = extract_format_template(user_p)
                if not tmpl:
                    n_skipped += 1
                    continue
                assistant = fill_template(tmpl, str(t["expected_value"]))
                # Verify it matches the format_regex (sanity check)
                if not re.search(t["format_regex"], assistant, re.MULTILINE):
                    n_skipped += 1
                    continue
            msgs = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
                {"role": "assistant", "content": assistant},
            ]
            out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"in={n_in} out={n_out} skipped={n_skipped} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
