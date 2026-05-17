#!/bin/bash
# Install the opd-capability-creator skill into .claude/skills/ via a
# symlink to the .agents canonical copy. Idempotent.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$ROOT/.agents/skills/opd-capability-creator"
DEST_DIR="$ROOT/.claude/skills"
DEST="$DEST_DIR/opd-capability-creator"

mkdir -p "$DEST_DIR"
ln -sfn "../../.agents/skills/opd-capability-creator" "$DEST"
echo "installed: $DEST -> .agents/skills/opd-capability-creator"
