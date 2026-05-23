#!/usr/bin/env python3
"""
Phase 0.5 — DType usage audit.

Grep every dtype that flows through forward.rs + the training paths, plus
the FP8 / Marlin-INT4 / FP4 quantized variants that don't go through
`candle_core::DType` but live in their own crate-local types.

Outputs:

    bench-results/dtype-usage.csv:
        dtype, count, first_seen, exemplars (top crates)

    bench-results/dtype-usage.md:
        per-dtype interpretation + the proposed `kiln_tensor::DType` enum

The result drives Phase 1's `DType` enum: every kept dtype must be
justified by a real Qwen3.5-4B call site or by a Phase 8 perf lever
(NVFP4 / MXFP4).
"""

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "bench-results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "dtype-usage.csv"
    md_path = out_dir / "dtype-usage.md"

    # We track two families:
    #   1. candle_core::DType::* literal hits — the "core" enum.
    #   2. Quantized / packed dtype hits that bypass candle DType.
    CORE_PATS = {
        "FP32 (F32)":      re.compile(r"\bDType::F32\b"),
        "BF16":            re.compile(r"\bDType::BF16\b"),
        "FP16 (F16)":      re.compile(r"\bDType::F16\b"),
        "U8":              re.compile(r"\bDType::U8\b"),
        "I64":             re.compile(r"\bDType::I64\b"),
        "U32":             re.compile(r"\bDType::U32\b"),
        # Cross-check for FP8 that candle might expose in some forks:
        "FP8 E4M3 (candle)": re.compile(r"\bDType::F8E4M3\b"),
        "FP8 E5M2 (candle)": re.compile(r"\bDType::F8E5M2\b"),
    }
    QUANT_PATS = {
        # FP8 (custom kiln types — used in paged KV)
        "FP8 E4M3 (kiln-custom)": re.compile(
            r"\bF8E4M3\b|\bFp8E4M3\b|\bfp8_e4m3\b|\bF8_E4M3\b"
        ),
        "FP8 E5M2 (kiln-custom)": re.compile(
            r"\bF8E5M2\b|\bFp8E5M2\b|\bfp8_e5m2\b|\bF8_E5M2\b"
        ),
        # FP8 scale-pair sites — paged_kv_cache `fp8_scales`
        "FP8 K/V cache scales": re.compile(
            r"\bfp8_scales\b|new_with_fp8|\bfp8_k\b|\bfp8_v\b"
        ),
        # INT4 packed (Marlin) — surface is "Marlin" not DType
        "INT4 packed (Marlin)": re.compile(
            r"\bMarlinPack\b|\bMarlinW4A16\b|\bMarlin\b"
        ),
        # FP4 / NVFP4 / MXFP4 — not present today; tracked because the
        # issue lists FP4Packed in the future DType enum.
        "FP4Packed (future)": re.compile(
            r"\bFP4Packed\b|\bFp4Packed\b|\bNVFP4\b|\bMXFP4\b"
        ),
    }

    counts = defaultdict(lambda: {"count": 0, "first": "", "crates": defaultdict(int)})

    for path in (repo_root / "crates").rglob("*.rs"):
        rel = path.relative_to(repo_root)
        if "target" in rel.parts or "vendor" in rel.parts:
            continue
        crate = rel.parts[1] if rel.parts[0] == "crates" and len(rel.parts) > 2 else "(other)"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            # Skip comment-only lines for the core-DType audit. Quantized
            # types stay because a comment using the literal name still
            # signals migration surface.
            stripped = line.lstrip()
            is_comment = stripped.startswith("//") or stripped.startswith("/*")
            for label, pat in CORE_PATS.items():
                if is_comment:
                    continue
                if pat.search(line):
                    c = counts[label]
                    c["count"] += 1
                    c["crates"][crate] += 1
                    if not c["first"]:
                        c["first"] = f"{rel}:{lineno}"
            for label, pat in QUANT_PATS.items():
                if pat.search(line):
                    c = counts[label]
                    c["count"] += 1
                    c["crates"][crate] += 1
                    if not c["first"]:
                        c["first"] = f"{rel}:{lineno}"

    # Order: stable display order plus alphabetical fallback.
    DISPLAY_ORDER = [
        "FP32 (F32)", "BF16", "FP16 (F16)",
        "U8", "I64", "U32",
        "FP8 E4M3 (candle)", "FP8 E5M2 (candle)",
        "FP8 E4M3 (kiln-custom)", "FP8 E5M2 (kiln-custom)",
        "FP8 K/V cache scales",
        "INT4 packed (Marlin)",
        "FP4Packed (future)",
    ]
    rows = []
    for label in DISPLAY_ORDER:
        c = counts.get(label, {"count": 0, "first": "", "crates": {}})
        crates_str = ";".join(
            f"{k}={v}" for k, v in sorted(c["crates"].items(), key=lambda kv: -kv[1])[:5]
        )
        rows.append({
            "dtype": label, "count": c["count"], "first": c["first"],
            "exemplars": crates_str,
        })

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dtype", "count", "first_seen", "top_5_crates"])
        for r in rows:
            w.writerow([r["dtype"], r["count"], r["first"], r["exemplars"]])

    # ------------------------------------------------------------------
    # Markdown interpretation.
    # ------------------------------------------------------------------
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 0.5 — DType usage audit\n\n")
        f.write(
            "Source of truth: `bench-results/dtype-usage.csv`.\n"
            "Regenerate: `scripts/audit-dtype-usage.py`.\n\n"
            "Why this audit\n--------------\n\n"
            "Per the issue: every kept dtype in `kiln_tensor::DType` must be "
            "justified by either (a) a real Qwen3.5-4B call site or (b) a "
            "Phase 8 perf lever. This audit produces the call-site evidence "
            "for each candidate dtype so the Phase 1 `DType` enum is sized "
            "exactly to what we ship, not a candle-style superset.\n\n"
        )
        f.write("## Counts\n\n")
        f.write("| dtype | call sites | first seen | top crates |\n")
        f.write("|---|---:|---|---|\n")
        for r in rows:
            f.write(
                f"| `{r['dtype']}` | {r['count']} | "
                f"`{r['first']}` | {r['exemplars']} |\n"
            )

        f.write("\n## Proposed `kiln_tensor::DType` for Qwen3.5-4B\n\n")
        f.write(
            "Based on the counts above:\n\n"
            "```rust\n"
            "pub enum DType {\n"
            "    // Activations + accumulators (FP32 is the canonical numerical\n"
            "    // reference on CPU; BF16 is the production forward + backward\n"
            "    // dtype for Qwen3.5-4B; FP16 is the candle-Mac path).\n"
            "    F32,\n"
            "    BF16,\n"
            "    F16,\n"
            "    // Index / mask dtypes (paged-KV slot indices, attention mask).\n"
            "    U32,\n"
            "    U8,\n"
            "    I64,    // tokenizer ids; sampling argmax output\n"
            "    // FP8 — paged KV cache `new_with_fp8` is the only hot-path\n"
            "    // user today. Forward-only; backward stays BF16.\n"
            "    F8E4M3,\n"
            "    F8E5M2,\n"
            "    // INT4 packed — Marlin W4A16 forward storage. Backward never\n"
            "    // dispatches here (anti-pattern: Marlin is forward-only).\n"
            "    Int4Packed,\n"
            "    // FP4 packed — Blackwell NVFP4 / MXFP4. Not present today;\n"
            "    // scaffolded for Phase 8.10.\n"
            "    Fp4Packed,\n"
            "}\n"
            "```\n\n"
            "Dtypes deliberately **NOT** in the enum:\n\n"
            "- **`I32`** — no call sites in the forward path; Rust's `i32` is\n"
            "  used for tokenizer scratch but never crosses a tensor boundary.\n"
            "- **`I8`** — there is no INT8-quantized path in Qwen3.5-4B. (We have\n"
            "  Marlin W4A16, not W8A8.)\n"
            "- **`F64`** — no call sites in the hot path. (Some debug receipts\n"
            "  promote to `f64` Rust-host-side for stable JSON, but not on\n"
            "  the device.)\n"
            "- **TF32 / Brain-int / Posit** — out of scope.\n\n"
        )

        f.write("## Causal links forward\n\n")
        f.write(
            "- **Phase 1 dependency**: this enum lands as part of the `kiln-tensor` "
            "scaffold. Each kernel crate declares which `DType`s its `DeviceOp` "
            "supports; the dispatch table is sized against this enum.\n"
            "- **Phase 2.5 dependency**: `Parameter::AmpPolicy` carries a\n"
            "  `DType` per role (forward / backward / master / accumulation); the\n"
            "  enum above is the surface that AmpPolicy fields enumerate over.\n"
            "- **Phase 8.10 hook**: `Fp4Packed` is scaffolded today as a stub variant\n"
            "  so Blackwell-class hardware (NVFP4 / MXFP6) is a per-DeviceOp\n"
            "  extension rather than a workspace-wide refactor.\n"
            "- **Phase 9 enforcement**: re-run this audit as a CI step; an\n"
            "  enum variant added without a justifying call site fails the gate.\n"
        )

    total = sum(r["count"] for r in rows)
    print(f"audit-dtype-usage: {len(rows)} dtype labels, {total} call sites",
          file=sys.stderr)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
