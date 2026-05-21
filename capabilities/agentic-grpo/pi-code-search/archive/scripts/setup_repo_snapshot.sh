#!/usr/bin/env bash
# Build the read-only repo snapshot pi-code-search rollouts will search.
#
# Run ONCE per pod: clones kiln into /workspace/kiln-snapshot at the
# current HEAD. After this completes, every rollout's workdir gets a
# symlink to /workspace/kiln-snapshot under `repo/`.
set -euo pipefail

DST="${PI_CODE_SEARCH_REPO:-/workspace/kiln-snapshot}"
SRC="${1:-/workspace/kiln}"

if [ -d "$DST" ]; then
  echo "[setup_repo_snapshot] $DST already exists — refreshing"
  cd "$DST"
  git fetch origin --depth=1 2>/dev/null || true
  git reset --hard "$(git ls-remote origin HEAD | awk '{print $1}')" 2>/dev/null || true
  exit 0
fi

if [ -d "$SRC" ]; then
  echo "[setup_repo_snapshot] copying $SRC → $DST"
  cp -r "$SRC" "$DST"
else
  echo "[setup_repo_snapshot] cloning kiln → $DST"
  git clone --depth=1 https://github.com/ericflo/kiln.git "$DST"
fi

# Strip the heavy build outputs so the snapshot is search-only.
rm -rf "$DST/target" "$DST/.git/lfs"
echo "[setup_repo_snapshot] done. $DST contains $(find "$DST" -type f | wc -l) files."
