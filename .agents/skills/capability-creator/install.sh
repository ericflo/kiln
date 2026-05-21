#!/usr/bin/env bash
# install.sh — symlink the capability-creator skill into .claude/skills/
#
# Run once per worktree to make the skill discoverable by Claude Code.

set -euo pipefail

SKILL_SRC="$(cd "$(dirname "$0")" && pwd)"
SKILL_NAME="capability-creator"

REPO_ROOT="$(git rev-parse --show-toplevel)"
TARGET_DIR="$REPO_ROOT/.claude/skills"
TARGET="$TARGET_DIR/$SKILL_NAME"

mkdir -p "$TARGET_DIR"

if [[ -e "$TARGET" ]]; then
  if [[ -L "$TARGET" ]]; then
    echo "skill already linked at $TARGET; replacing"
    rm "$TARGET"
  else
    echo "error: $TARGET exists and is not a symlink" >&2
    exit 1
  fi
fi

ln -s "$SKILL_SRC" "$TARGET"
echo "linked $SKILL_NAME → $TARGET"

# Make scripts executable
chmod +x "$SKILL_SRC"/templates/*.sh 2>/dev/null || true

echo ""
echo "skill installed. Test with:"
echo "  ls $TARGET/SKILL.md"
echo "  bash $SKILL_SRC/templates/scaffold.sh test-cap  # then rm -rf the test cap"
