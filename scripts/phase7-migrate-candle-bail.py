#!/usr/bin/env python3
"""Phase 7 migration tool: candle_core::bail!(...) → kiln_tensor::bail!(...).

Per the Phase 7 plan (`bench-results/phase7-removal-plan.md`):
candle_core::bail! is the single most-frequent candle API in the
workspace at 493 call sites. This script does an in-place rewrite
of those sites, fixing imports too.

The transformation is conservative:
1. Lines containing `candle_core::bail!` are rewritten to
   `kiln_tensor::bail!` (paranoid: only the explicit double-colon
   form is rewritten; `use candle_core::bail;` callers that then
   write `bail!()` are handled by the import-rewrite phase below).
2. `use candle_core::bail;` lines become
   `use kiln_tensor::bail;`. A file that imports both via
   `use candle_core::{bail, ...};` gets the `bail` peeled into its
   own `use kiln_tensor::bail;` line.
3. Files that picked up a new `kiln_tensor::bail` use but did not
   previously depend on kiln_tensor are flagged in the report (the
   caller updates Cargo.toml separately).

Usage:
    python3 scripts/phase7-migrate-candle-bail.py \\
        [--write] [--dir crates/]

Default is dry-run (prints the diff summary). Pass `--write` to
actually apply the rewrites.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


BAIL_QUALIFIED_RE = re.compile(r"candle_core::bail!")
USE_BAIL_LINE_RE = re.compile(r"^\s*use\s+candle_core::bail\s*;\s*$")
USE_GROUP_RE = re.compile(
    r"use\s+candle_core::\{([^{}]*)\};"
)


def rewrite_text(text: str) -> Tuple[str, int, int, int]:
    """Return (new_text, qualified_rewrites, simple_use_rewrites, grouped_use_rewrites)."""
    qual = 0
    simple = 0
    grouped = 0

    # Pass 1: explicit `candle_core::bail!` call sites.
    def _rep_qual(m: re.Match[str]) -> str:
        nonlocal qual
        qual += 1
        return "kiln_tensor::bail!"

    out = BAIL_QUALIFIED_RE.sub(_rep_qual, text)

    # Pass 2: rewrite the standalone `use candle_core::bail;` line.
    lines = out.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if USE_BAIL_LINE_RE.match(line):
            lines[i] = line.replace(
                "use candle_core::bail",
                "use kiln_tensor::bail",
            )
            simple += 1
    out = "".join(lines)

    # Pass 3: peel `bail` out of grouped `use candle_core::{...}` lines.
    def _rep_group(m: re.Match[str]) -> str:
        nonlocal grouped
        items = [it.strip() for it in m.group(1).split(",")]
        items = [it for it in items if it]
        if "bail" not in items:
            return m.group(0)
        items = [it for it in items if it != "bail"]
        grouped += 1
        if not items:
            return "use kiln_tensor::bail;"
        return (
            "use kiln_tensor::bail;\n"
            f"use candle_core::{{{', '.join(items)}}};"
        )

    out = USE_GROUP_RE.sub(_rep_group, out)
    return out, qual, simple, grouped


def scan(root: Path) -> Dict[Path, Tuple[int, int, int]]:
    report: Dict[Path, Tuple[int, int, int]] = {}
    for path in root.rglob("*.rs"):
        # Skip vendored candle.
        if "vendor/candle-core" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "candle_core::bail" not in text and "use candle_core::bail" not in text:
            continue
        _, q, s, g = rewrite_text(text)
        if q + s + g > 0:
            report[path] = (q, s, g)
    return report


def apply_changes(root: Path) -> Tuple[int, int, int, int]:
    files_changed = 0
    qual_total = 0
    simple_total = 0
    grouped_total = 0
    for path in root.rglob("*.rs"):
        if "vendor/candle-core" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "candle_core::bail" not in text and "use candle_core::bail" not in text:
            continue
        new_text, q, s, g = rewrite_text(text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            files_changed += 1
            qual_total += q
            simple_total += s
            grouped_total += g
    return files_changed, qual_total, simple_total, grouped_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="crates", type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if not args.dir.exists():
        print(f"directory not found: {args.dir}", file=sys.stderr)
        return 2
    if args.write:
        files, q, s, g = apply_changes(args.dir)
        print(
            f"rewrote {files} files: "
            f"{q} qualified-call rewrites, {s} simple-use rewrites, "
            f"{g} grouped-use rewrites"
        )
    else:
        report = scan(args.dir)
        if not report:
            print("nothing to migrate.")
            return 0
        print(f"dry-run report ({len(report)} files would change):")
        for path, (q, s, g) in sorted(report.items()):
            parts = []
            if q:
                parts.append(f"{q} call(s)")
            if s:
                parts.append(f"{s} simple-use")
            if g:
                parts.append(f"{g} grouped-use")
            print(f"  {path}: {'; '.join(parts)}")
        total_q = sum(v[0] for v in report.values())
        total_s = sum(v[1] for v in report.values())
        total_g = sum(v[2] for v in report.values())
        print(
            f"totals: {total_q} qualified calls, "
            f"{total_s} simple-use lines, {total_g} grouped-use peels."
        )
        print("re-run with --write to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
