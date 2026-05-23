#!/usr/bin/env bash
# Phase 0.1 — Candle API surface audit.
#
# Greps every `candle_core::`, `candle_nn::`, and bare `candle::` call site
# across `crates/`, buckets matches by module path + symbol, and writes
# `bench-results/candle-api-surface.csv`:
#
#     api_path,call_site_count,crate,first_seen_path:line
#
# The full row-per-call-site detail is also written to
# `bench-results/candle-api-surface.raw.tsv` for sanity checks.
#
# The vendored `vendor/candle-core/` tree is excluded — it is the upstream
# we are removing, not surface we have to migrate. (Phase 7 deletes it.)
#
# Usage:
#     scripts/audit-candle-usage.sh                 # write CSV + raw TSV
#     scripts/audit-candle-usage.sh --summary       # also print a top-20 to stdout
#
# Reproducible: the script is deterministic given a fixed working tree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/bench-results"
RAW="$OUT_DIR/candle-api-surface.raw.tsv"
CSV="$OUT_DIR/candle-api-surface.csv"

PRINT_SUMMARY=0
if [[ "${1:-}" == "--summary" ]]; then
    PRINT_SUMMARY=1
fi

mkdir -p "$OUT_DIR"

# 1. Capture every candle call site under crates/.
#    Excludes vendor/, target/, comments-only lines? — we keep comments because
#    a doc comment that names `candle_core::Tensor` is itself surface to retire.
#    We rely on python below to bucket; ripgrep gives us file:line:match for free.
{
    if command -v rg >/dev/null 2>&1; then
        rg --no-heading -n --type rust \
           '\bcandle_core::|\bcandle_nn::|(^|[^A-Za-z_])candle::' \
           "$REPO_ROOT/crates" || true
    else
        # Fallback: grep -RIn against *.rs only.
        grep -RIn --include='*.rs' -E \
             '\bcandle_core::|\bcandle_nn::|(^|[^A-Za-z_])candle::' \
             "$REPO_ROOT/crates" || true
    fi
} > "$RAW.tmp"

# 2. Bucket by API path (the candle_core::FOO::BAR token, longest dotted form
#    found on the line).
python3 - "$RAW.tmp" "$RAW" "$CSV" "$REPO_ROOT" <<'PY'
import csv, os, re, sys
from collections import defaultdict

raw_in, raw_out, csv_out, repo_root = sys.argv[1:5]

# Match candle_core::A::B or candle_nn::A or candle::A — capture the longest
# `::`-joined identifier chain rooted at one of the three roots.
PAT = re.compile(
    r"\b(candle_core|candle_nn|candle)"           # root
    r"((?:::[A-Za-z_][A-Za-z0-9_]*)+)"            # at least one ::ident
)

buckets = defaultdict(lambda: {"count": 0, "crate": "", "first": ""})

with open(raw_in, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        # ripgrep / grep -n format: <path>:<lineno>:<text>
        m = re.match(r"^([^:]+):(\d+):(.*)$", line)
        if not m:
            continue
        path, lineno, text = m.group(1), m.group(2), m.group(3)
        rel = os.path.relpath(path, repo_root)
        # First crate is the second path segment under crates/ (crates/<crate>/...).
        parts = rel.split(os.sep)
        crate = parts[1] if len(parts) > 2 and parts[0] == "crates" else "(other)"

        for sym in PAT.finditer(text):
            api = sym.group(1) + sym.group(2)
            b = buckets[api]
            b["count"] += 1
            if not b["first"]:
                b["first"] = f"{rel}:{lineno}"
                b["crate"] = crate

# Raw TSV: one row per call site (api\tcrate\tpath:line\tsnippet).
with open(raw_out, "w", encoding="utf-8") as f:
    f.write("api\tcrate\tpath_line\tsnippet\n")
    with open(raw_in, encoding="utf-8") as src:
        for line in src:
            line = line.rstrip("\n")
            m = re.match(r"^([^:]+):(\d+):(.*)$", line)
            if not m:
                continue
            path, lineno, text = m.group(1), m.group(2), m.group(3)
            rel = os.path.relpath(path, repo_root)
            parts = rel.split(os.sep)
            crate = parts[1] if len(parts) > 2 and parts[0] == "crates" else "(other)"
            for sym in PAT.finditer(text):
                api = sym.group(1) + sym.group(2)
                snippet = text.strip()[:240]
                f.write(f"{api}\t{crate}\t{rel}:{lineno}\t{snippet}\n")

with open(csv_out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["api_path", "call_site_count", "first_crate", "first_seen"])
    for api in sorted(buckets, key=lambda a: (-buckets[a]["count"], a)):
        b = buckets[api]
        w.writerow([api, b["count"], b["crate"], b["first"]])

total = sum(b["count"] for b in buckets.values())
print(f"audit-candle-usage: {len(buckets)} distinct APIs, {total} call sites", file=sys.stderr)
PY

rm -f "$RAW.tmp"

if [[ "$PRINT_SUMMARY" == "1" ]]; then
    echo "Top 20 candle APIs by call-site count:"
    # Skip header; first 20 rows after header.
    awk -F, 'NR==1 {next} {print $2"\t"$1}' "$CSV" | sort -k1 -n -r | head -20
fi

echo "wrote $CSV"
echo "wrote $RAW"
