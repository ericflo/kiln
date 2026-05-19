#!/usr/bin/env bash
# Install pi (https://github.com/earendil-works/pi) onto a kiln runpod.
#
# Strategy:
#  1. Try the npm tarball via `npm install -g`.
#  2. Fall back to cloning the github repo and running `npm link`.
#  3. Verify `pi --version` works.
set -euo pipefail

if command -v pi >/dev/null 2>&1; then
  echo "[install_pi] pi already on PATH: $(which pi); skipping"
  pi --version || true
  exit 0
fi

# Ensure node 20+ is present.
if ! command -v node >/dev/null 2>&1 || [ "$(node --version | sed 's/v//' | cut -d. -f1)" -lt 20 ]; then
  echo "[install_pi] installing node 20"
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
node --version
npm --version

# Try the simplest: clone + npm link (verified path).
WORK=/workspace/pi-src
if [ ! -d "$WORK" ]; then
  git clone --depth=1 https://github.com/earendil-works/pi.git "$WORK"
fi
cd "$WORK"
# pi is a multi-package repo. The CLI lives under packages/agent or similar.
ls -la
# Find which workspace owns the `pi` binary.
PKG_DIR=$(grep -lr '"name": "pi"' packages */package.json 2>/dev/null | head -1 | xargs -r dirname)
PKG_DIR="${PKG_DIR:-packages/agent}"
echo "[install_pi] PKG_DIR=$PKG_DIR"
cd "$WORK"
npm install 2>&1 | tail -5
# Some repos use pnpm or yarn; try npm first.
npm run build 2>&1 | tail -5 || true
cd "$WORK/$PKG_DIR" 2>/dev/null || cd "$WORK"
npm link 2>&1 | tail -5 || true

# Sanity.
if command -v pi >/dev/null 2>&1; then
  echo "[install_pi] OK: $(which pi)"
  pi --version || true
else
  echo "[install_pi] FAILED — see logs above" >&2
  exit 1
fi
