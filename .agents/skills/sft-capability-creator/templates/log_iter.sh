#!/usr/bin/env bash
# log_iter.sh — append one entry to capability.jsonl with anti-laziness gates.
#
# Central enforcement point. REJECTS entries that don't satisfy the gates
# adopted from the OPD sister-skill, which were forged in the python-algo
# session's failure mode (208 iters, 6 hypothesis files — verdicts were
# skipped because nothing was forcing them).
#
# Two invocation modes:
#
#   1. JSON-file mode (preferred): pass a path to a pre-built JSON entry.
#        $SKILL/templates/log_iter.sh <entry.json>
#      The entry should be a complete JSON object with all fields below.
#
#   2. Legacy positional mode (back-compat with older agent code):
#        log_iter.sh <slug> <status> <score> <n> <hypothesis_file> \
#                    <dataset_file> <adapter_name> <final_loss> <elapsed_s>
#      In this mode the script builds the entry from the args + cwd
#      state but applies the SAME gates. The agent must additionally
#      run `templates/annotate.sh` to fill in verdict + asi before the
#      next iter — see Phase 5.
#
# Gates (both modes):
#   - verdict: non-empty, starts with ✓/✗/?, references numbers, ≥30 chars
#   - asi.what_worked OR asi.what_failed: at least one non-empty, non-boilerplate
#   - asi.next_focus: non-empty, ≥12 chars, non-boilerplate
#   - status=broken: requires failure_mode.md in cwd
#   - structural fields: iter, slug, ts, status, score (or null), dataset, training
#
# Exit codes:
#   0  appended
#   2  validation failed (gate rejection) or bad args
#   3  capability.jsonl missing
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "usage: log_iter.sh <entry.json>" >&2
  echo "   or: log_iter.sh <slug> <status> <score> <n> <hyp> <ds> <adapter> <loss> <elapsed>" >&2
  exit 2
fi

if [ ! -f capability.jsonl ]; then
  echo "log_iter: capability.jsonl not found in cwd (run scaffold first)" >&2
  exit 3
fi

# Detect mode
ENTRY_JSON=""
if [ "$#" -eq 1 ] && [ -f "$1" ]; then
  ENTRY_JSON="$(cat "$1")"
elif [ "$#" -ge 2 ]; then
  # Legacy positional mode — build a partial JSON entry.
  SLUG="$1"; STATUS="$2"; SCORE="${3:-}"; N="${4:-}"
  HYP_FILE="${5-}"; DS_FILE="${6-}"; ADAPTER="${7-}"; LOSS="${8-}"; ELAPSED="${9-}"

  NO_SCORE_STATES="crash|oracle_error|firewall_breach"
  if echo "$STATUS" | grep -qE "^($NO_SCORE_STATES)$" || [ -z "$SCORE" ]; then
    SCORE_JSON=null; SCORE=0
  else
    SCORE_JSON="$SCORE"
  fi

  CFG=capability.config.json
  DIR=$(jq -r '.direction // "higher"' "$CFG" 2>/dev/null || echo "higher")
  PREV_BEST=$(jq -rs --arg dir "$DIR" '
    map(select(.score != null and (.status == "kept" or .slug == "baseline" or .iter == 0)))
    | if length == 0 then "null"
      elif $dir == "lower" then (min_by(.score).score|tostring)
      else (max_by(.score).score|tostring) end
  ' capability.jsonl)
  ITER=$(($(wc -l < capability.jsonl)))

  if [ "$PREV_BEST" = "null" ] || [ "$SCORE_JSON" = "null" ]; then
    DELTA=0
  else
    DELTA=$(awk -v a="$SCORE" -v b="$PREV_BEST" -v d="$DIR" 'BEGIN {
      if (d == "lower") print b - a; else print a - b
    }')
  fi

  CLAIM=""
  VERDICT=""
  if [ -n "$HYP_FILE" ] && [ -f "$HYP_FILE" ]; then
    CLAIM=$(awk '/^## Claim/{flag=1; next} /^## /{flag=0} flag' "$HYP_FILE" | sed '/^$/d' | head -1)
    # Extract Verdict block (after "## Verdict") — required for gate.
    VERDICT=$(awk '/^## Verdict/{flag=1; next} /^## /{flag=0} flag' "$HYP_FILE" | sed '/^$/d;/^(filled after eval)$/d' | tr '\n' ' ' | sed 's/[[:space:]]\+$//')
  fi

  DS_SIZE=0
  [ -n "$DS_FILE" ] && [ -f "$DS_FILE" ] && DS_SIZE=$(wc -l < "$DS_FILE")
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  ENTRY_JSON=$(jq -nc \
    --argjson iter "$ITER" \
    --arg slug "$SLUG" \
    --arg ts "$TS" \
    --arg status "$STATUS" \
    --argjson score "$SCORE_JSON" \
    --argjson n "${N:-0}" \
    --argjson delta "$DELTA" \
    --arg hypothesis "$CLAIM" \
    --arg verdict "$VERDICT" \
    --arg ds_path "$DS_FILE" \
    --argjson ds_size "$DS_SIZE" \
    --arg adapter "$ADAPTER" \
    --argjson loss "${LOSS:-0}" \
    --argjson elapsed "${ELAPSED:-0}" \
    '{
      iter: $iter, slug: $slug, ts: $ts, status: $status,
      score: $score, n: $n, delta: $delta,
      hypothesis: $hypothesis, verdict: $verdict,
      dataset: { path: $ds_path, size: $ds_size },
      training: { adapter: $adapter, final_loss: $loss, elapsed_s: $elapsed },
      asi: {}, notes: ""
    }')
else
  echo "log_iter: bad args" >&2
  exit 2
fi

# Validate via Python so gate logic is readable.
ENTRY_JSON="$ENTRY_JSON" exec python3 - <<'PY'
import json, os, re, sys
from pathlib import Path

data = json.loads(os.environ["ENTRY_JSON"])
errs: list[str] = []

# Gate 1: verdict
verdict = (data.get("verdict") or "").strip()
if not verdict:
    errs.append("verdict is empty (Gate 1: hypothesis loop not closed)")
elif not re.match(r"^[✓✗?]", verdict):
    errs.append("verdict must start with ✓ (confirmed), ✗ (falsified), or ? (inconclusive)")
elif not re.search(r"[+\-]?\d+\.?\d*\s*(pp|%|×)?", verdict):
    errs.append("verdict must reference actual numbers (e.g. '+0.06', '0.95')")
elif len(verdict) < 30:
    errs.append(f"verdict too short ({len(verdict)} chars; expected ≥30) — write a real justification")

# Gate 2: asi block (only required for normal training entries — crashes/errors get a pass)
NO_ASI_STATES = {"crash", "oracle_error", "firewall_breach"}
if data.get("status") not in NO_ASI_STATES:
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
        errs.append("asi.next_focus is empty — name the next hypothesis or direction")

# Gate 3: status=broken requires failure_mode.md
if data.get("status") == "broken" and not Path("failure_mode.md").exists():
    errs.append(
        "status=broken requires failure_mode.md in cwd before the next iter "
        "(inspect responses, diagnose, propose a fix). See SKILL.md anti-pattern §16."
    )

# Gate 4: structural fields
for f in ["iter", "slug", "ts", "status", "dataset", "training"]:
    if f not in data:
        errs.append(f"missing required field: {f}")

if errs:
    print("log_iter: VALIDATION FAILED — entry rejected.", file=sys.stderr)
    print("", file=sys.stderr)
    for e in errs:
        print(f"  ✗ {e}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Fix the entry and re-run. Gates exist to prevent the lazy habits the "
        "python-algo 208-iter / 6-hypothesis session fell into. See SKILL.md §3 Phase 5.",
        file=sys.stderr,
    )
    sys.exit(2)

with open("capability.jsonl", "a") as f:
    f.write(json.dumps(data) + "\n")

iter_ = data.get("iter")
score = data.get("score")
status = data.get("status")
slug = data.get("slug")
print(f"log_iter: appended iter={iter_} slug={slug} score={score} status={status}")
PY
