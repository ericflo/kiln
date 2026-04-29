#!/usr/bin/env python3
"""Idempotent patch: replace the 6-cell main() body in
phase10_rmsnorm_bench.rs with a single cell that runs T=8192 RMSNorm-on.

This is the FLCE Phase B closure check — we don't need the full A/B sweep,
we just need to confirm Phase B unblocks T=8192 SFT on A6000-class GPUs.

After running, build and execute the patched bench. Restore the file via
git checkout when done.
"""

import re
import sys
from pathlib import Path

PATCHED_BLOCK = """    // Phase B closure validation: only T=8192 RMSNorm-on. The full 6-cell
    // sweep is preserved on disk; this patch is applied via
    // scripts/phase10_flce_phase_b_t8192_only.py during the closure
    // bench and reverted on PR cleanup.
    rows.push(run_one(
        8192, true, &tokenizer, &model_config, &gpu_weights, baseline_mib,
    )?);
"""

ORIGINAL_BLOCK_RE = re.compile(
    r"    // Cells 1-2: T=2048 parity check.*?"
    r"    rows\.push\(run_one\(\s*8192, false.*?\)\?\);\n",
    re.DOTALL,
)

def main() -> int:
    src = Path("crates/kiln-server/examples/phase10_rmsnorm_bench.rs")
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1
    text = src.read_text()
    if "Phase B closure validation: only T=8192" in text:
        print("Already patched — no-op")
        return 0
    new_text, n = ORIGINAL_BLOCK_RE.subn(PATCHED_BLOCK, text)
    if n != 1:
        print(f"ERROR: regex matched {n} times, expected 1", file=sys.stderr)
        return 1
    src.write_text(new_text)
    print(f"Patched {src} (replaced 6 cells with T=8192 rmsnorm-on)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
