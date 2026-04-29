#!/usr/bin/env python3
"""Idempotent patch: replace the 6-cell main() body in
phase10_rmsnorm_bench.rs with a single cell that runs T=16384 RMSNorm-on.

This is the §3 conditional-reopen FleCE T=16384 OOM probe — we don't need
the full A/B sweep, we just need to observe whether T=16384 closes (peak
< 48 GiB) or OOMs at the 48 GiB ceiling on A6000/A40 with Phase B alone.

The bench's run_one() already sets KILN_USE_FLCE=1 (Phase B = chunked-vocab
CustomOp1 forward), so this probe matches the audit's "Phase B alone, no
future FleCE Phase C" configuration. PR #649's audit recommends this exact
probe.

After running, build and execute the patched bench. Restore the file via
git checkout when done.
"""

import re
import sys
from pathlib import Path

PATCHED_BLOCK = """    // §3 conditional-reopen FleCE T=16384 OOM probe: only T=16384 RMSNorm-on.
    // The full 6-cell sweep is preserved on disk; this patch is applied via
    // scripts/phase10_flce_phase_b_t16384_only.py during the probe and
    // reverted on PR cleanup.
    rows.push(run_one(
        16384, true, &tokenizer, &model_config, &gpu_weights, baseline_mib,
    )?);
"""

ORIGINAL_BLOCK_RE = re.compile(
    r"    // Cells 1-2: T=2048 parity check.*?"
    r"    rows\.push\(run_one\(\s*8192, false.*?\)\?\);\n",
    re.DOTALL,
)

# Also handle the case where PR #647's T=8192-only patch is already applied.
T8192_PATCHED_RE = re.compile(
    r"    // Phase B closure validation: only T=8192.*?"
    r"    rows\.push\(run_one\(\s*8192, true.*?\)\?\);\n",
    re.DOTALL,
)

def main() -> int:
    src = Path("crates/kiln-server/examples/phase10_rmsnorm_bench.rs")
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1
    text = src.read_text()
    if "§3 conditional-reopen FleCE T=16384 OOM probe" in text:
        print("Already patched for T=16384 — no-op")
        return 0

    # Try patching from the original 6-cell layout first.
    new_text, n = ORIGINAL_BLOCK_RE.subn(PATCHED_BLOCK, text)
    if n == 1:
        src.write_text(new_text)
        print(f"Patched {src} (replaced 6 cells with T=16384 rmsnorm-on)")
        return 0

    # Fallback: patching from the T=8192-only layout (PR #647 helper applied).
    new_text, n = T8192_PATCHED_RE.subn(PATCHED_BLOCK, text)
    if n == 1:
        src.write_text(new_text)
        print(f"Patched {src} (replaced T=8192-only cell with T=16384 rmsnorm-on)")
        return 0

    print(
        f"ERROR: regex matched 0 times in either layout — bench may have changed",
        file=sys.stderr,
    )
    return 1

if __name__ == "__main__":
    sys.exit(main())
