#!/usr/bin/env python3
"""
Phase 0.2 — CustomOpN audit.

The Vulkan-removed-from-this-issue's fifth candle subsystem: every fused
kernel crate plugs into candle's `CustomOp1` / `CustomOp2` / `CustomOp3`
trait, providing per-backend `cpu_fwd` / `cuda_fwd` / `metal_fwd`
implementations plus an optional `bwd`. The migration design must define
the kiln-tensor equivalent (forward closure + backward closure with
explicit `BackwardOp` registration) BEFORE Phase 1 lands or the kernels
port twice.

This script walks the `impl Custom{Op1,Op2,Op3}` blocks under `crates/`
and writes a structural table:

    bench-results/customop-audit.csv:
        impl_name, op_arity, file, line, crate, has_cpu_fwd, has_cuda_fwd,
        has_metal_fwd, has_bwd, proposed_kiln_tensor_shape

    bench-results/customop-audit.md:
        per-impl interpretation + the proposed kiln-tensor design

The "proposed_kiln_tensor_shape" column is the migration-shape proposal:

- `closure-only` — forward-only ops (no `bwd`); become a plain device-method.
- `fwd+bwd-closure` — has `bwd`; becomes a `kiln_tensor::Op { fwd, bwd }`
  pair registered with the `BackwardOp` trait.
- `static-tape-op` — the bwd already names an explicit op handle on the
  backprop graph; lifts directly to `kiln_autograd::BackwardOp` enum
  variant.
"""

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def audit(repo_root: Path):
    impls = []
    for path in (repo_root / "crates").rglob("*.rs"):
        rel = path.relative_to(repo_root)
        if "target" in rel.parts:
            continue
        if "vendor" in rel.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            m = re.search(
                r"\bimpl\s+(?:candle_core::)?(?:CustomOp([123]))\s+for\s+([A-Za-z_][A-Za-z0-9_<>'\,\s]*?)\s*\{",
                line,
            )
            if not m:
                continue
            arity = int(m.group(1))
            impl_name = m.group(2).strip().rstrip("<>'").split("<", 1)[0]
            # Walk forward to the matching closing brace at depth 0.
            depth = 1
            start = i
            j = i
            body = [line]
            while j < len(lines) and depth > 0:
                nxt = lines[j]
                j += 1
                depth += nxt.count("{") - nxt.count("}")
                if depth == 0:
                    body.append(nxt)
                    break
                body.append(nxt)
            block = "\n".join(body)
            crate = rel.parts[1] if rel.parts[0] == "crates" and len(rel.parts) > 2 else "(other)"
            impls.append({
                "impl_name": impl_name,
                "op_arity": arity,
                "file": str(rel),
                "line": start,
                "crate": crate,
                "has_cpu_fwd":   "fn cpu_fwd" in block,
                "has_cuda_fwd":  "fn cuda_fwd" in block,
                "has_metal_fwd": "fn metal_fwd" in block,
                "has_bwd":       "fn bwd" in block,
                "body_lines":    len(body),
            })
    return impls


def proposed_shape(impl):
    if not impl["has_bwd"]:
        return "closure-only"
    if "BackpropOp" in impl["impl_name"]:
        return "static-tape-op"
    return "fwd+bwd-closure"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "bench-results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "customop-audit.csv"
    md_path = out_dir / "customop-audit.md"

    impls = audit(repo_root)
    impls.sort(key=lambda r: (r["crate"], r["file"], r["line"]))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "impl_name", "op_arity", "file", "line", "crate",
            "has_cpu_fwd", "has_cuda_fwd", "has_metal_fwd", "has_bwd",
            "body_lines", "proposed_kiln_tensor_shape",
        ])
        for r in impls:
            w.writerow([
                r["impl_name"], r["op_arity"], r["file"], r["line"], r["crate"],
                "yes" if r["has_cpu_fwd"] else "no",
                "yes" if r["has_cuda_fwd"] else "no",
                "yes" if r["has_metal_fwd"] else "no",
                "yes" if r["has_bwd"] else "no",
                r["body_lines"], proposed_shape(r),
            ])

    by_crate = defaultdict(list)
    for r in impls:
        by_crate[r["crate"]].append(r)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 0.2 — CustomOpN audit\n\n")
        f.write(
            "Sources of truth:\n\n"
            f"- `bench-results/customop-audit.csv` — {len(impls)} `impl CustomOpN` blocks\n\n"
            "Regenerate: `scripts/audit-customop.py`.\n\n"
            "Why this audit\n--------------\n\n"
            "candle's `CustomOp1` / `CustomOp2` / `CustomOp3` trait is the fifth "
            "(and largest unlisted) candle subsystem the migration must replace. "
            "Every fused kernel crate — `kiln-flce-kernel`, `kiln-opd-loss-kernel`, "
            "`kiln-gdn-kernel`, `kiln-rmsnorm-kernel`, `kiln-conv1d-kernel` — "
            "plugs into it for per-backend dispatch plus optional `bwd`. "
            "If the kiln-tensor `Op` shape is not defined before Phase 1 lands, "
            "the kernels port twice: once for the forward path and again for "
            "the backward.\n\n"
        )

        f.write("## Per-crate breakdown\n\n")
        f.write("| crate | impls | has bwd | only cuda fwd | only metal fwd | all-backend |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for crate, rs in sorted(by_crate.items()):
            has_bwd = sum(1 for r in rs if r["has_bwd"])
            only_cuda = sum(
                1 for r in rs
                if r["has_cuda_fwd"] and not r["has_metal_fwd"] and not r["has_cpu_fwd"]
            )
            only_metal = sum(
                1 for r in rs
                if r["has_metal_fwd"] and not r["has_cuda_fwd"] and not r["has_cpu_fwd"]
            )
            all_back = sum(
                1 for r in rs
                if r["has_cpu_fwd"] and r["has_cuda_fwd"] and r["has_metal_fwd"]
            )
            f.write(
                f"| {crate} | {len(rs)} | {has_bwd} | {only_cuda} | "
                f"{only_metal} | {all_back} |\n"
            )

        f.write("\n## All impls\n\n")
        f.write(
            "| impl | arity | crate | file:line | bwd? | cpu/cuda/metal fwd | proposed shape |\n"
            "|---|---:|---|---|:-:|:-:|---|\n"
        )
        for r in impls:
            fwd = (
                ("c" if r["has_cpu_fwd"] else "-")
                + ("u" if r["has_cuda_fwd"] else "-")
                + ("m" if r["has_metal_fwd"] else "-")
            )
            f.write(
                f"| `{r['impl_name']}` | {r['op_arity']} | {r['crate']} | "
                f"{r['file']}:{r['line']} | "
                f"{'yes' if r['has_bwd'] else 'no'} | "
                f"`{fwd}` | "
                f"{proposed_shape(r)} |\n"
            )

        f.write("\n## Proposed kiln-tensor shape\n\n")
        f.write(
            "The kiln-tensor equivalent the kernels can port onto **once**:\n\n"
            "```rust\n"
            "// kiln-tensor crate.\n"
            "pub trait DeviceOp: Send + Sync {\n"
            "    /// Arity is fixed at the impl: `DeviceOp1` / `DeviceOp2` / `DeviceOp3`.\n"
            "    /// Each per-backend method returns Option<...>; None means the op falls\n"
            "    /// back to the next backend in the device's preference order, matching\n"
            "    /// today's BackendRuntime contract.\n"
            "    fn name(&self) -> &'static str;\n"
            "    fn cpu_fwd  (&self, ...) -> Result<Option<Tensor>>;\n"
            "    fn cuda_fwd (&self, ...) -> Result<Option<Tensor>>;\n"
            "    fn metal_fwd(&self, ...) -> Result<Option<Tensor>>;\n"
            "    fn vulkan_fwd(&self, ...) -> Result<Option<Tensor>>;\n"
            "    /// Optional backward closure; absence == forward-only kernel.\n"
            "    /// The boxed closure carries its own captured tensors;\n"
            "    /// `kiln_autograd::BackwardOp::register` records the closure on\n"
            "    /// the tape with the source tensor's TensorId.\n"
            "    fn bwd(&self) -> Option<Box<dyn BackwardOp>>;\n"
            "}\n"
            "```\n\n"
            "Three migration shapes (column `proposed_kiln_tensor_shape` in the CSV):\n\n"
            "- `closure-only` — forward-only ops without `bwd`. Becomes a plain device-method "
            "on `kiln_tensor::Tensor`. Migration is mechanical.\n"
            "- `fwd+bwd-closure` — has `bwd`. Becomes a `kiln_tensor::DeviceOp` plus a "
            "`kiln_autograd::BackwardOp` impl. The Vulkan path's `VkBackwardOp` (`vk_ops/`) "
            "is the lift template — 34 `impl VkBackwardOp for ...` blocks already follow "
            "this shape.\n"
            "- `static-tape-op` — the existing bwd already routes through "
            "`candle_core::backprop::BackpropOp`. The tape op becomes a `kiln_autograd::Op` "
            "enum variant; the `bwd` method is its `apply` impl.\n\n"
            "The audit is the input to Phase 1's `DeviceOp` API design and to Phase 6a's "
            "`kiln-autograd` crate skeleton.\n"
        )

    print(f"audit-customop: {len(impls)} impls", file=sys.stderr)
    for crate, rs in sorted(by_crate.items()):
        print(f"  {crate}: {len(rs)}", file=sys.stderr)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
