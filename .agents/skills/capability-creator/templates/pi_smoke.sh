#!/usr/bin/env bash
# pi_smoke.sh — mandatory before iter 1 of any agentic-GRPO stage.
#
# Verifies pi binary + kiln serve + tool-call session + trajectory parsing
# in 7 steps. If any step fails, fix BEFORE burning GPU time on the cap.
#
# Run from inside a cap dir:
#   bash $SKILL/templates/pi_smoke.sh

set -euo pipefail

PI_BIN="${PI_BIN:-/usr/bin/pi}"
KILN_URL="${KILN_URL:-http://localhost:8420}"
MODEL_ID="${MODEL_ID:-Qwen3.5-4B}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/pi-smoke-$$}"
mkdir -p "$SMOKE_DIR"
trap 'rm -rf "$SMOKE_DIR"' EXIT

echo "1. pi binary on PATH..."
if ! command -v "$PI_BIN" > /dev/null 2>&1; then
  echo "FAIL: $PI_BIN not found" >&2
  exit 1
fi
echo "   OK: $($PI_BIN --version 2>/dev/null || echo 'pi --version unavailable')"

echo "2. kiln serving the base model on $KILN_URL..."
if ! curl -sf "$KILN_URL/v1/health" > /dev/null; then
  echo "FAIL: kiln not reachable at $KILN_URL" >&2
  exit 1
fi
echo "   OK"

echo "3. pi configured against kiln..."
if [[ ! -f "$HOME/.pi/config.json" ]]; then
  echo "WARN: ~/.pi/config.json missing; pi may use defaults"
fi
echo "   OK (best effort)"

echo "4. headless pi session: print HELLO and exit..."
(
  cd "$SMOKE_DIR"
  echo "print HELLO and exit" | "$PI_BIN" --no-interactive --model "$MODEL_ID" 2>/dev/null
) || true

echo "5. session JSONL appears under ~/.pi/agent/sessions/..."
NEWEST_SESSION=$(find "$HOME/.pi/agent/sessions" -name "*.jsonl" -mmin -5 2>/dev/null | head -1)
if [[ -z "$NEWEST_SESSION" ]]; then
  echo "FAIL: no recent pi session JSONL found" >&2
  exit 1
fi
echo "   OK: $NEWEST_SESSION"

echo "6. session JSONL has assistant turn..."
if ! grep -q '"role":"assistant"' "$NEWEST_SESSION"; then
  echo "FAIL: no assistant turn in $NEWEST_SESSION" >&2
  exit 1
fi
echo "   OK"

echo "7. kiln trajectory inspect parses with nonzero action_mask..."
INSPECT=$(kiln trajectory inspect "$NEWEST_SESSION" --json 2>/dev/null) || {
  echo "FAIL: kiln trajectory inspect failed" >&2
  exit 1
}
ACTION_COUNT=$(echo "$INSPECT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('action_token_count', 0))")
if [[ "$ACTION_COUNT" -eq 0 ]]; then
  echo "FAIL: trajectory has zero action tokens" >&2
  exit 1
fi
echo "   OK: action_token_count=$ACTION_COUNT"

echo ""
echo "pi_smoke PASSED. Safe to run agentic-GRPO stages."
