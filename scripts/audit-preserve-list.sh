#!/usr/bin/env bash
# Phase 0.7 — Preserve-list audit.
#
# Records three contract surfaces that must NOT change names during the
# candle-removal migration, per the issue:
#
#   (a) Every NVTX range name in `forward.rs` (kiln_nvtx::range!(c"kiln/…"))
#       — these are how PROFILING.md hot-region percentages stay comparable
#       across the migration.
#   (b) Every KILN_* env var that gates a code path
#       — these are user-visible; renaming breaks deployments.
#   (c) Every BackendRuntime method that takes a `candle_core::Tensor` arg
#       — these are the seam Phase 1's `kiln_tensor::Tensor` slots into.
#
# Migration PRs that touch any of these surfaces must include an explicit
# "preserved" checkbox in the PR description.
#
# Outputs:
#   bench-results/preserve-list-nvtx.csv
#   bench-results/preserve-list-env.csv
#   bench-results/preserve-list-backend-runtime.csv
#   bench-results/preserve-list.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/bench-results"
NVTX_CSV="$OUT_DIR/preserve-list-nvtx.csv"
ENV_CSV="$OUT_DIR/preserve-list-env.csv"
BR_CSV="$OUT_DIR/preserve-list-backend-runtime.csv"
SUMMARY_MD="$OUT_DIR/preserve-list.md"

mkdir -p "$OUT_DIR"

###############################################################################
# (a) NVTX ranges in forward.rs (and any other kernel-instrumented files)
###############################################################################

python3 - "$REPO_ROOT" "$NVTX_CSV" <<'PY'
import csv, os, re, sys
from collections import defaultdict

repo_root, csv_out = sys.argv[1:3]

# Match `kiln_nvtx::range!(c"<name>")` and `kiln_nvtx::range!("<name>")` and
# their `range_arg!` variant.
PAT = re.compile(
    r"kiln_nvtx::range(?:_arg)?!\s*\(\s*c?\"(?P<name>[^\"]+)\""
)

seen = defaultdict(lambda: {"count": 0, "first": ""})

# Walk crates/ and grep every .rs file.
for dirpath, dirs, files in os.walk(os.path.join(repo_root, "crates")):
    # Skip target/ if it somehow appears.
    dirs[:] = [d for d in dirs if d != "target"]
    for name in files:
        if not name.endswith(".rs"):
            continue
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, repo_root)
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                for m in PAT.finditer(line):
                    nm = m.group("name")
                    e = seen[nm]
                    e["count"] += 1
                    if not e["first"]:
                        e["first"] = f"{rel}:{lineno}"

with open(csv_out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nvtx_range_name", "call_site_count", "first_seen"])
    for nm in sorted(seen, key=lambda n: (-seen[n]["count"], n)):
        e = seen[nm]
        w.writerow([nm, e["count"], e["first"]])

print(f"nvtx: {len(seen)} distinct range names, "
      f"{sum(e['count'] for e in seen.values())} call sites",
      file=sys.stderr)
PY

###############################################################################
# (b) KILN_* env vars
###############################################################################

python3 - "$REPO_ROOT" "$ENV_CSV" <<'PY'
import csv, os, re, sys
from collections import defaultdict

repo_root, csv_out = sys.argv[1:3]

# Match either:
#   "KILN_FOO"  (string literal — most call sites)
#   env::var("KILN_FOO") / std::env::var("KILN_FOO")
#   env_flag("KILN_FOO", ...) / env_tristate("KILN_FOO")
#
# Capture every string literal starting with `KILN_` followed by an
# uppercase / underscore identifier.
LIT_PAT = re.compile(r'"KILN_([A-Z][A-Z0-9_]*)"')

seen = defaultdict(lambda: {"count": 0, "crates": set(), "first": "",
                            "via_env_flag": False, "via_env_var": False})

ENV_FLAG_FN = re.compile(r"\benv_flag\s*\(")
ENV_TRI_FN = re.compile(r"\benv_tristate\s*\(")
ENV_VAR_FN = re.compile(r"std::env::var\s*\(|^\s*env::var\s*\(|[^a-zA-Z_]env::var\s*\(")

for dirpath, dirs, files in os.walk(os.path.join(repo_root, "crates")):
    dirs[:] = [d for d in dirs if d != "target"]
    for name in files:
        if not name.endswith(".rs"):
            continue
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, repo_root)
        parts = rel.split(os.sep)
        crate = parts[1] if len(parts) > 2 and parts[0] == "crates" else "(other)"
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                via_env_flag = bool(ENV_FLAG_FN.search(line) or ENV_TRI_FN.search(line))
                via_env_var = bool(ENV_VAR_FN.search(line))
                for m in LIT_PAT.finditer(line):
                    nm = "KILN_" + m.group(1)
                    e = seen[nm]
                    e["count"] += 1
                    e["crates"].add(crate)
                    if via_env_flag:
                        e["via_env_flag"] = True
                    if via_env_var:
                        e["via_env_var"] = True
                    if not e["first"]:
                        e["first"] = f"{rel}:{lineno}"

with open(csv_out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["env_var", "call_site_count", "first_seen",
                "via_env_flag_or_tristate", "via_env_var", "crates_touched"])
    for nm in sorted(seen, key=lambda n: (-seen[n]["count"], n)):
        e = seen[nm]
        w.writerow([nm, e["count"], e["first"],
                    "yes" if e["via_env_flag"] else "no",
                    "yes" if e["via_env_var"] else "no",
                    ";".join(sorted(e["crates"]))])

centralized = sum(1 for e in seen.values() if e["via_env_flag"])
print(f"env: {len(seen)} distinct KILN_* vars, "
      f"{sum(e['count'] for e in seen.values())} call sites; "
      f"{centralized} go through env_flag/env_tristate",
      file=sys.stderr)
PY

###############################################################################
# (c) BackendRuntime methods taking candle_core::Tensor / candle_core::Var
###############################################################################

python3 - "$REPO_ROOT" "$BR_CSV" <<'PY'
import csv, os, re, sys

repo_root, csv_out = sys.argv[1:3]

backend_mod = os.path.join(repo_root, "crates/kiln-model/src/backend/mod.rs")
if not os.path.exists(backend_mod):
    print(f"warning: {backend_mod} not found; skipping BackendRuntime audit",
          file=sys.stderr)
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method_name", "first_line", "signature_snippet",
                    "uses_candle_tensor", "uses_candle_var"])
    sys.exit(0)

# Parse the file looking for `pub fn <name>` lines inside the BackendRuntime
# trait. We don't care about non-trait fns, so we scope to the
# `pub trait BackendRuntime` block.
with open(backend_mod, encoding="utf-8") as f:
    text = f.read()
    lines = text.splitlines()

# Find the start and end of the trait. We treat the brace count as the
# delimiter (kiln-model has the trait in one mod file).
trait_idx = None
for i, line in enumerate(lines):
    if re.search(r"\bpub\s+trait\s+BackendRuntime\b", line):
        trait_idx = i
        break

rows = []
if trait_idx is not None:
    depth = 0
    started = False
    method_buf = []
    method_line = None
    for i in range(trait_idx, len(lines)):
        line = lines[i]
        opens = line.count("{")
        closes = line.count("}")
        if not started:
            if opens > 0:
                started = True
                depth += opens - closes
                continue
        else:
            depth += opens - closes
            if depth <= 0:
                break

            # Inside trait: detect "fn name" signatures (handles multi-line
            # signatures by collecting until terminator ';' or '{').
            if not method_buf:
                m = re.search(
                    r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<\(]",
                    line.strip())
                if m:
                    method_buf = [line]
                    method_line = i + 1
                    method_name = m.group(1)
            else:
                method_buf.append(line)
                method_name = (re.search(
                    r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)",
                    method_buf[0]) or [None, "?"])[1]

            if method_buf and (";" in line or " {" in line.rstrip()):
                sig = " ".join(s.strip() for s in method_buf)
                # Truncate to opening brace or terminator.
                m_end = re.search(r"[;{]", sig)
                if m_end:
                    sig = sig[: m_end.start() + 1]
                uses_tensor = "candle_core::Tensor" in sig or "Tensor)" in sig or " Tensor " in sig
                uses_var = "candle_core::Var" in sig or " Var " in sig or " Var)" in sig
                # Only record methods that actually mention a candle type
                # OR mention `Tensor` ambiguously (it could be a kiln-tensor
                # type already; the audit's purpose is to flag the seam).
                if "candle_core::" in sig or " Tensor" in sig or " Var" in sig:
                    rows.append({
                        "method_name": method_name,
                        "first_line": method_line,
                        "signature_snippet": sig[:380],
                        "uses_candle_tensor": "yes" if uses_tensor else "no",
                        "uses_candle_var": "yes" if uses_var else "no",
                    })
                method_buf = []
                method_line = None

with open(csv_out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method_name", "first_line", "uses_candle_tensor",
                "uses_candle_var", "signature_snippet"])
    for r in rows:
        w.writerow([r["method_name"], r["first_line"],
                    r["uses_candle_tensor"], r["uses_candle_var"],
                    r["signature_snippet"]])

print(f"backend_runtime: {len(rows)} methods with candle-typed signatures",
      file=sys.stderr)
PY

###############################################################################
# Build a markdown summary.
###############################################################################

python3 - "$REPO_ROOT" "$SUMMARY_MD" "$NVTX_CSV" "$ENV_CSV" "$BR_CSV" <<'PY'
import csv, sys
from collections import Counter

repo_root, out_md, nvtx_csv, env_csv, br_csv = sys.argv[1:6]

# Load.
with open(nvtx_csv, encoding="utf-8") as f:
    nvtx = list(csv.DictReader(f))
with open(env_csv, encoding="utf-8") as f:
    env = list(csv.DictReader(f))
with open(br_csv, encoding="utf-8") as f:
    br = list(csv.DictReader(f))

nvtx_total = sum(int(r["call_site_count"]) for r in nvtx)
env_total = sum(int(r["call_site_count"]) for r in env)
env_central = sum(1 for r in env if r["via_env_flag_or_tristate"] == "yes")

# Cluster NVTX names by their `:kiln/<prefix>/` root so PROFILING.md's
# hot-region buckets stay legible.
def prefix(name):
    # name looks like ":kiln/gdn/in_proj" or "kiln/mlp/gate"; normalize.
    n = name.lstrip(":")
    if n.startswith("kiln/"):
        parts = n.split("/", 2)
        if len(parts) >= 2:
            return parts[1]
    return "(other)"

clusters = Counter()
for r in nvtx:
    clusters[prefix(r["nvtx_range_name"])] += int(r["call_site_count"])

with open(out_md, "w", encoding="utf-8") as f:
    f.write("# Phase 0.7 — Preserve-list audit\n\n")
    f.write(
        "Sources of truth:\n\n"
        f"- `bench-results/preserve-list-nvtx.csv` ({nvtx_total} call sites, "
        f"{len(nvtx)} distinct range names)\n"
        f"- `bench-results/preserve-list-env.csv` ({env_total} call sites, "
        f"{len(env)} distinct `KILN_*` vars; {env_central} go through "
        "`env_flag` / `env_tristate`)\n"
        f"- `bench-results/preserve-list-backend-runtime.csv` ({len(br)} "
        "trait methods whose signature still mentions a candle type)\n\n"
        "Regenerate: `scripts/audit-preserve-list.sh`.\n\n"
        "---\n\n"
    )

    f.write("## Contract\n\n")
    f.write(
        "Every migration PR that touches one of these three surfaces must "
        "include an explicit 'preserved' checkbox in the PR description. "
        "Specifically:\n\n"
        "- **NVTX range names** keep their exact string spelling so "
        "`PROFILING.md`'s hot-region percentages stay comparable across the "
        "migration.\n"
        "- **`KILN_*` env vars** keep their exact name so user deployments "
        "do not break.\n"
        "- **`BackendRuntime` trait methods** keep their method names; only "
        "the argument types swap from candle to kiln-tensor.\n\n"
    )

    f.write("## NVTX range clusters\n\n")
    f.write("Grouped by `kiln/<prefix>/...`; counts are total call sites.\n\n")
    f.write("| cluster | call sites |\n|---|---:|\n")
    for cluster, n in clusters.most_common():
        f.write(f"| `{cluster}` | {n} |\n")

    f.write("\n## Top 20 `KILN_*` env vars by call-site count\n\n")
    f.write(
        "| env var | sites | via `env_flag`? | crates touched |\n"
        "|---|---:|---:|---|\n"
    )
    for r in env[:20]:
        f.write(
            f"| `{r['env_var']}` | {r['call_site_count']} | "
            f"{r['via_env_flag_or_tristate']} | "
            f"{r['crates_touched']} |\n"
        )

    f.write("\n## `BackendRuntime` candle-typed methods\n\n")
    if not br:
        f.write("_No methods detected — the audit may need tuning if you "
                "expected hits here._\n")
    else:
        f.write(
            "| method | first line | candle Tensor? | candle Var? |\n"
            "|---|---:|:-:|:-:|\n"
        )
        for r in br[:40]:
            f.write(
                f"| `{r['method_name']}` | {r['first_line']} | "
                f"{r['uses_candle_tensor']} | "
                f"{r['uses_candle_var']} |\n"
            )

    f.write("\n## Causal links forward\n\n")
    f.write(
        "- **Phase 1 contract**: `BackendRuntime` is the seam (per the "
        "issue's starting points). The above method list is the call surface "
        "Phase 1 must shim — every method on it gets a kiln-tensor variant "
        "that the trait dispatches to under `KILN_USE_KILN_TENSOR_*`.\n"
        "- **Phase 9 enforcement**: re-run this audit as a CI step; "
        "renaming a row in any of the three CSVs without a deliberate, "
        "documented decision fails the gate.\n"
        "- **Anti-pattern 13 ('NVTX range names are part of the trace "
        "contract')**: the NVTX CSV is the verifiable form of that anti-"
        "pattern.\n"
    )

print(f"wrote {out_md}", file=sys.stderr)
PY

echo "wrote $NVTX_CSV"
echo "wrote $ENV_CSV"
echo "wrote $BR_CSV"
echo "wrote $SUMMARY_MD"
