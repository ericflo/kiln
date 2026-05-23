#!/usr/bin/env bash
# Phase 0.6 — Multi-GPU seam audit.
#
# Multi-GPU stays out of scope for #1082 (anti-pattern 12), but we cannot
# hardcode `Device::Cuda(0)` anywhere — device selection must route through
# a centralized accessor so:
#
#   (a) two kiln processes on one box do not both grab GPU 0,
#   (b) a future TP rewrite does not have to revisit ~100+ call sites.
#
# This script greps for hardcoded device-0 literals and records them to
# `bench-results/multi-gpu-seam.csv` with the recommended replacement.
#
# Production sites (non-test) get the centralized accessor replacement
# (`kiln_core::device::primary_cuda()`-style). Test sites get a test helper
# replacement (`test_helpers::cuda_device()`-style) so a future TP runner
# can override.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_CSV="$REPO_ROOT/bench-results/multi-gpu-seam.csv"

mkdir -p "$(dirname "$OUT_CSV")"

# Patterns we treat as a hardcoded device-0 site. We do NOT match
# `set_device(0)` calls in vendor/, examples that take a CLI flag, or
# comments — only literal source.
#
# What we match:
#   Device::new_cuda(0)
#   Device::new_cuda_with_stream(0)
#   Device::Cuda(0)
#   candle_core::Device::new_cuda(0)
#   candle_core::Device::Cuda(0)
PATTERN='Device::new_cuda\(0\)|Device::new_cuda_with_stream\(0\)|Device::Cuda\(0\)'

# Collect all hits as path:line:text.
HITS_FILE="$(mktemp)"
trap "rm -f $HITS_FILE" EXIT

if command -v rg >/dev/null 2>&1; then
    rg --no-heading -n --type rust -e "$PATTERN" "$REPO_ROOT/crates" \
        > "$HITS_FILE" || true
else
    grep -RIn --include='*.rs' -E "$PATTERN" "$REPO_ROOT/crates" \
        > "$HITS_FILE" || true
fi

python3 - "$HITS_FILE" "$OUT_CSV" "$REPO_ROOT" <<'PY'
import csv, os, re, sys

hits_in, csv_out, repo_root = sys.argv[1:4]

# Categorize:
#   - is_test: path contains /tests/ or src ends in _test.rs or is examples/
#               or under tests/ directory
#   - is_example: path under crates/.../examples/
#   - is_production: everything else
def categorize(rel, abs_path, lineno):
    """Categorize a hit as 'test', 'example', or 'production'.

    A `#[test]` block inside a `src/` file is still test code — promoted from
    the path-based heuristic by reading 200 lines of context backward and
    looking for `#[test]` / `#[cfg(test)]` / `mod tests {` markers since the
    last top-level item.
    """
    parts = rel.split(os.sep)
    if "tests" in parts:
        return "test"
    if "examples" in parts:
        return "example"
    if rel.endswith("_test.rs") or rel.endswith("_tests.rs"):
        return "test"
    if os.path.basename(rel).startswith("test_"):
        return "test"

    # Look upward 200 lines (or to start of file) for a `#[test]` /
    # `#[cfg(test)]` attribute attached to an enclosing item.
    try:
        with open(abs_path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return "production"

    upto = int(lineno) - 1
    start = max(0, upto - 200)
    window = lines[start:upto]
    # Walk backwards from the hit. Track the nearest `fn` / `mod` boundary;
    # if a `#[test]` or `#[cfg(test)]` attribute precedes it, this is test.
    found_test = False
    for i in range(len(window) - 1, -1, -1):
        s = window[i].strip()
        if s.startswith("#[test]") or s.startswith("#[cfg(test)]"):
            found_test = True
            break
        # Stop on a top-level item that is clearly NOT inside a test mod —
        # bare `fn` at column 0, `pub fn`, `impl`, or a non-test `mod ...`.
        if s.startswith("pub fn ") or s.startswith("fn "):
            # Could be a test fn whose attribute lives further up; keep going
            # only if there's no `mod tests` separator in between.
            continue
        if s.startswith("mod tests") or s.startswith("mod test_"):
            found_test = True
            break

    return "test" if found_test else "production"

# Recommend replacement based on bucket.
REPLACEMENTS = {
    "production": "kiln_core::device::primary_cuda()",
    "test":       "kiln_core::test_support::cuda_device()",
    "example":    "kiln_core::device::cuda_from_args_or_primary()",
}

rows = []
with open(hits_in, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        m = re.match(r"^([^:]+):(\d+):(.*)$", line)
        if not m:
            continue
        path, lineno, text = m.group(1), m.group(2), m.group(3)
        rel = os.path.relpath(path, repo_root)
        parts = rel.split(os.sep)
        crate = parts[1] if len(parts) > 2 and parts[0] == "crates" else "(other)"

        # Extract the literal device-0 token actually used on this line.
        lit_match = re.search(
            r"(Device::new_cuda_with_stream\(0\)|"
            r"Device::new_cuda\(0\)|"
            r"Device::Cuda\(0\))", text
        )
        if not lit_match:
            continue
        current_literal = lit_match.group(1)

        bucket = categorize(rel, path, lineno)
        replacement = REPLACEMENTS[bucket]

        rows.append({
            "file": rel,
            "line": lineno,
            "crate": crate,
            "bucket": bucket,
            "current_literal": current_literal,
            "replacement": replacement,
            "snippet": text.strip()[:240],
        })

# Sort by crate, file, line.
rows.sort(key=lambda r: (r["crate"], r["file"], int(r["line"])))

with open(csv_out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "line", "crate", "bucket", "current_literal",
                "replacement", "snippet"])
    for r in rows:
        w.writerow([r["file"], r["line"], r["crate"], r["bucket"],
                    r["current_literal"], r["replacement"], r["snippet"]])

# Print a per-crate summary to stderr.
from collections import Counter
print(f"multi-gpu-seam: {len(rows)} hardcoded device-0 sites", file=sys.stderr)
crate_counts = Counter(r["crate"] for r in rows)
for crate, n in crate_counts.most_common():
    bs = Counter(r["bucket"] for r in rows if r["crate"] == crate)
    bs_str = ", ".join(f"{k}={v}" for k, v in bs.most_common())
    print(f"  {crate}: {n} ({bs_str})", file=sys.stderr)
PY

echo "wrote $OUT_CSV"
