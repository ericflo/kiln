#!/usr/bin/env bash
# log_iter.sh — append one entry to capability.jsonl with anti-laziness gates.
#
# Central enforcement point. REJECTS entries that don't satisfy the gates
# listed in SKILL.md §5:
#
#   - verdict: non-empty, starts with ✓/✗/?, references numbers, ≥30 chars
#   - target_sub_score: non-empty
#   - asi.what_worked OR asi.what_failed: at least one non-empty, non-boilerplate
#   - asi.next_focus: non-empty, ≥12 chars, non-boilerplate
#   - regressions: explicit array (possibly empty)
#   - kiln_polish_noted: explicit bool
#   - status=broken: requires failure_mode.md in cwd
#
# Usage:
#   $SKILL/templates/log_iter.sh <entry.json>
#
# Exit codes:
#   0  appended
#   2  validation failed (gate rejection)
#   3  capability.jsonl missing
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "usage: log_iter.sh <entry.json>" >&2
  exit 2
fi
ENTRY="$1"

if [ ! -f "$ENTRY" ]; then
  echo "log_iter: entry file not found: $ENTRY" >&2
  exit 2
fi
if [ ! -f capability.jsonl ]; then
  echo "log_iter: capability.jsonl not found in cwd (run scaffold first)" >&2
  exit 3
fi

ENTRY_PATH="$ENTRY" exec python3 - <<'PY'
import json
import os
import re
import sys
from pathlib import Path

data = json.loads(Path(os.environ["ENTRY_PATH"]).read_text())
errs: list[str] = []

# Gate 1: verdict
verdict = (data.get("verdict") or "").strip()
if not verdict:
    errs.append("verdict is empty (Gate 1: hypothesis loop not closed)")
elif not re.match(r"^[✓✗?]", verdict):
    errs.append("verdict must start with ✓ (confirmed), ✗ (falsified), or ? (inconclusive)")
elif not re.search(r"[+\-]?\d+\.?\d*\s*(pp|%|×)?", verdict):
    errs.append("verdict must reference actual numbers (e.g. '+3.1pp', '0.05')")
elif len(verdict) < 30:
    errs.append(f"verdict too short ({len(verdict)} chars; expected ≥30) — write a real justification")

# Gate 2: target_sub_score
target = (data.get("target_sub_score") or "").strip()
if not target:
    errs.append("target_sub_score is empty (which sub-score was this iter trying to lift?)")

# Gate 3: asi block
asi = data.get("asi") or {}
ww = (asi.get("what_worked") or "").strip()
wf = (asi.get("what_failed") or "").strip()
nf = (asi.get("next_focus") or "").strip()

if not ww and not wf:
    errs.append("asi.what_worked and asi.what_failed are both empty — at least one is required")

boilerplate = {
    "looks good", "works well", "did fine", "as expected", "see results",
    "see notes", "see above", "see entry", "ok", "good", "fine", "nothing",
    "n/a", "na", "tbd", "todo",
}
for fname, fval in [("what_worked", ww), ("what_failed", wf), ("next_focus", nf)]:
    if fval and fval.lower().strip().rstrip(".") in boilerplate:
        errs.append(f"asi.{fname} is boilerplate ({fval!r}); write actual mechanism")
    elif fval and len(fval) < 12:
        errs.append(f"asi.{fname} too short ({len(fval)} chars; expected ≥12)")

if not nf:
    errs.append("asi.next_focus is empty — name the next hypothesis or family")

# Gate 4: regressions array
if "regressions" not in data:
    errs.append("regressions missing — must be an explicit array (possibly empty)")
elif not isinstance(data["regressions"], list):
    errs.append("regressions must be a JSON array")

# Gate 5: kiln_polish_noted
if "kiln_polish_noted" not in data:
    errs.append("kiln_polish_noted missing — must be true/false explicitly")
elif not isinstance(data["kiln_polish_noted"], bool):
    errs.append("kiln_polish_noted must be a JSON bool (true/false)")

# Gate 6: status=broken requires failure_mode.md
if data.get("status") == "broken" and not Path("failure_mode.md").exists():
    errs.append(
        "status=broken requires failure_mode.md in cwd before the next iter "
        "(inspect responses, diagnose, propose a fix). See SKILL.md §10."
    )

# Gate 7: structural fields
for f in ["iter", "slug", "ts", "status", "family", "composite", "sub_scores", "training"]:
    if f not in data:
        errs.append(f"missing required field: {f}")

if errs:
    print("log_iter: VALIDATION FAILED — entry rejected.", file=sys.stderr)
    print("", file=sys.stderr)
    for e in errs:
        print(f"  ✗ {e}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Fix the entry and re-run. Gates exist to prevent the lazy habits "
        "the SFT skill suffered from. See SKILL.md §5.",
        file=sys.stderr,
    )
    sys.exit(2)

with open("capability.jsonl", "a") as f:
    f.write(json.dumps(data) + "\n")

print(
    f"log_iter: appended iter {data.get('iter')} slug={data.get('slug')} "
    f"composite={data.get('composite')} status={data.get('status')}"
)
PY
