#!/bin/bash
# Install the sft-capability-creator skill into .claude/skills/ via a
# symlink to the .agents canonical copy. Idempotent.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT/.agents/skills/sft-capability-creator"
DEST_DIR="$ROOT/.claude/skills"
DEST="$DEST_DIR/sft-capability-creator"

mkdir -p "$DEST_DIR"
ln -sfn "../../.agents/skills/sft-capability-creator" "$DEST"
echo "installed: $DEST -> .agents/skills/sft-capability-creator"
