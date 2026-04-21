#!/usr/bin/env python3
"""
Phase C11 — Marlin W4A16 per-channel scale drift audit.

Goal
----

C10 closed with a null behavioral verdict: across 24 MTP-head tap sites
x 3 seeds, zero splices flipped kiln's argmax when kiln's bf16 tap was
substituted into the fp32 HF reference forward path (72 splice tests,
cos_sim >= 0.9999 everywhere). C10's C11 target #1 was therefore holistic:
**does kiln's Marlin W4A16 packing of q_proj + MLP introduce per-channel
scale drift outside the fp32-equivalence band large enough to plausibly
cause Class B top1 flips?**

C10's numerical bar was cos_sim >= 0.9999, relative Frobenius drift < 1e-3.
We reuse that same bar here, applied per output channel (per column of the
pre-transposed [k, n] weight, i.e. per output feature of the linear layer),
for every Marlin-packed projection in kiln's forward path.

Scope (what IS Marlin-packed in kiln)
-------------------------------------

Per crates/kiln-model/src/forward.rs (the main forward loop at the
`w4a16_enabled` branches):

  - MAIN MODEL: q_proj on the 8 full-attention layers only (GDN layers
    have no q_proj/k_proj/v_proj/o_proj).
  - MAIN MODEL: gate_proj + up_proj + down_proj on ALL 32 layers.
  - MTP HEAD: explicitly NOT packed. The forward.rs comment at
    `// For WIP scaffolding the MTP transformer layer is kept in BF16 and
    // is NOT queued for Marlin batch packing` is authoritative.
  - k_proj, v_proj, o_proj (attention non-q), lm_head, embed_tokens: NOT
    packed, stay BF16.

Total audited projections: 8 x q_proj + 32 x (gate_proj + up_proj + down_proj)
= 104 projections. All use groupsize = 128 per
`marlin_pack_inputs.push((..., 128))`.

Algorithm (mirrors `kiln_marlin_gemm::pack::quantize_and_pack`)
---------------------------------------------------------------

For each audited projection weight W (HF canonical layout `[out, in]`):

  1. Transpose to `weight_t` layout `[k=in, n=out]` to match the kiln
     `q_proj_t` / `gate_proj_t` / etc. that the Marlin packer consumes.
  2. For each output column (= each output feature = each n-axis index):
     For each group of K rows (groupsize=128):
       a. max_abs = max(|w|) over the 128 rows in this (group, col).
       b. scale_f32 = (2 * max_abs) / 15.
       c. scale_f16 = scale_f32 cast to f16 and back to f32 (kiln rounds
          scales through f16 so the GPU kernel and the CPU reference
          agree bit-for-bit).
       d. For each row r in the group:
          q = clamp(round(w[r, col] / scale_f16) + 8, 0, 15).
          w_dq[r, col] = (q - 8) * scale_f16.
  3. Compute per-output-channel statistics comparing `w` vs `w_dq`:
       - cos_sim(w[:, col], w_dq[:, col]) as fp32 vectors of length k.
       - RMSE per column.
       - max |w - w_dq| per column.
       - Relative L2 error: ||w[:, col] - w_dq[:, col]|| / ||w[:, col]||.
  4. Flag channels whose cos_sim < 0.9999 or rel_L2 > 1e-3.

Verdict bar
-----------

  - `any_channel_outside_band = True` on at least one projection =>
    Marlin scale drift IS a plausible alpha-suppressing signal. Report
    the worst projection + channel idx as the C12 primary target.
  - All 104 projections fully inside the band => Marlin scale drift is
    NOT the signal. Promote C10 target #2 (fp32 draft head policy) to
    primary, exit 0, verdict null.

Cross-reference with C10 Class B rejection tokens
-------------------------------------------------

C10 reported kiln_matches_ref=True on all 3 seeds (seed42/3074, seed43/16078,
seed44/1814 at mtp_pos=2) — those top1 tokens were accepted, not rejected.
C10 has no direct Class B rejection positions we can cross-reference against
Marlin channels; the only honest cross-reference this audit can draw is:

  1. Per-channel drift statistics across all q_proj / MLP output columns.
  2. Call out any specific output-channel idx that falls outside the band.
  3. For q_proj specifically, channels map to (head, head_dim, gate/q)
     so a cross-reference to attention-head rejection patterns is possible
     if a worst-channel is identified. We spell that mapping out in the
     verdict doc, not the cross-reference itself (which needs Class B
     rejection dumps we do not have in this phase).

Honest limits
-------------

  - The reference here is the HF checkpoint's fp32 weights (upcast from
    stored bf16). Kiln also starts from bf16 and packs; the cos_sim we
    compute here is between the fp32-upcast bf16 weight and the
    dequantized INT4 weight, which is the same round-trip the GPU kernel
    emulates. We are NOT re-measuring bf16 vs fp32 — that was C9's bar.
  - We test the WEIGHT, not the activation-weight product. A per-channel
    scale that is outside-band on the weight may still produce in-band
    activations on realistic data if the signal never excites that
    channel. We explicitly note this caveat in the verdict doc.

Exit codes
----------

    0 — audit ran; all per-channel stats inside the fp32-equivalence band.
        Null verdict. No C12 target pulled from this audit.
    1 — audit ran; at least one projection has a per-channel cos_sim <
        0.9999 or relative L2 error > 1e-3. Worst projection named in
        verdict doc.
    2 — structural error (missing weight, shape mismatch, bad CLI).

Usage
-----

    python3 scripts/c11_marlin_audit.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --out docs/phase-c11/c11-marlin-audit.md \\
        --out-json docs/phase-c11/c11-marlin-audit.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open


# -----------------------------------------------------------------------------
# Weight-load helpers (mirrors scripts/mtp_reference_dump.py)
# -----------------------------------------------------------------------------

def st_load_tensor(checkpoint_dir: str, name: str, allow_missing: bool = False) -> Optional[torch.Tensor]:
    """Load a tensor from safetensors shards. Returns float32 on CPU."""
    idx_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        if name in weight_map:
            shard_path = os.path.join(checkpoint_dir, weight_map[name])
            with safe_open(shard_path, framework="pt", device="cpu") as g:
                if name in g.keys():
                    return g.get_tensor(name).to(torch.float32)
    for shard_path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        with safe_open(shard_path, framework="pt", device="cpu") as g:
            if name in g.keys():
                return g.get_tensor(name).to(torch.float32)
    if allow_missing:
        return None
    raise KeyError(f"tensor `{name}` not found under {checkpoint_dir}")


def discover_prefix(checkpoint_dir: str) -> str:
    """Return the prefix under which model layers live (empty, `model.`, or
    `model.language_model.`). Qwen3.5-4B uses `model.language_model.` for
    the main model (nested under `model.`).

    Probes `layers.0.mlp.down_proj.weight` — this is always present on every
    layer regardless of attention variant (GDN layers also have MLP tensors,
    whereas `self_attn.q_proj.weight` only exists on full-attn layers).
    """
    idx_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing {idx_path}")
    with open(idx_path, "r", encoding="utf-8") as f:
        keys = json.load(f)["weight_map"].keys()
    for prefix in ["model.language_model.", "model.", ""]:
        test_key = prefix + "layers.0.mlp.down_proj.weight"
        if test_key in keys:
            return prefix
    raise KeyError(
        "Could not discover layer prefix. None of the candidate "
        "`layers.0.mlp.down_proj.weight` keys are present in the "
        "safetensors index."
    )


def layer_is_full_attention(checkpoint_dir: str, prefix: str, layer_idx: int) -> bool:
    """Kiln's loader resolves attention variant per layer by probing for
    `self_attn.q_proj.weight` (full) vs `linear_attn.in_proj.weight` (GDN).
    Mirror that here: a layer is "full attention" iff it has a q_proj."""
    idx_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        keys = json.load(f)["weight_map"].keys()
    q_key = f"{prefix}layers.{layer_idx}.self_attn.q_proj.weight"
    return q_key in keys


# -----------------------------------------------------------------------------
# Marlin groupwise dequant (mirrors kiln_marlin_gemm::pack::quantize_and_pack)
# -----------------------------------------------------------------------------

MAX_Q = 15
Q_OFFSET = 8


def marlin_dequant_groupwise(weight_t_f32: np.ndarray, groupsize: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce kiln_marlin_gemm::pack::quantize_and_pack's scale + dequant
    on `weight_t` of shape [k, n] (pre-transposed, i.e. HF weight transposed).

    Returns (dequant_weights [k, n] fp32, scales [k/groupsize, n] fp32).

    Matches the upstream Marlin convention:
      s_f32 = (2 * max(|w|)) / 15  # per (group, col)
      s_f16 = f16(s_f32)           # scale round-tripped through f16
      q     = clamp(round(w/s_f16) + 8, 0, 15)
      w_dq  = (q - 8) * s_f16
    """
    k, n = weight_t_f32.shape
    assert k % groupsize == 0, f"k={k} not divisible by groupsize={groupsize}"
    num_groups = k // groupsize

    w = weight_t_f32.reshape(num_groups, groupsize, n)

    max_abs = np.max(np.abs(w), axis=1)  # [num_groups, n]
    # Guard against divide-by-zero, as kiln does.
    max_abs = np.where(max_abs == 0.0, 1.0, max_abs)
    s_f32 = (2.0 * max_abs) / float(MAX_Q)
    # Round-trip through f16 exactly like kiln_marlin_gemm::pack::quantize_and_pack.
    # numpy's float16 cast gives us IEEE 754 binary16 (same as Rust half::f16).
    s_f16 = s_f32.astype(np.float16).astype(np.float32)  # [num_groups, n]

    # Dequant vectorized.
    s_broadcast = s_f16[:, None, :]  # [num_groups, 1, n]
    q_raw = np.round(w / s_broadcast).astype(np.int32)
    q_shifted = np.clip(q_raw + Q_OFFSET, 0, MAX_Q)
    w_dq = (q_shifted - Q_OFFSET).astype(np.float32) * s_broadcast

    return w_dq.reshape(k, n), s_f16


# -----------------------------------------------------------------------------
# Per-channel statistics
# -----------------------------------------------------------------------------

@dataclass
class ChannelStats:
    """Aggregated per-output-channel numerical stats for one projection."""
    n_channels: int
    # cos_sim across all channels (col-wise):
    cos_min: float
    cos_p01: float
    cos_p50: float
    cos_p99: float
    cos_max: float
    worst_cos_channel: int
    worst_cos_value: float
    # relative L2 error per channel:
    rel_l2_max: float
    rel_l2_p99: float
    rel_l2_p50: float
    worst_rel_l2_channel: int
    worst_rel_l2_value: float
    # RMSE per channel:
    rmse_max: float
    rmse_p50: float
    # Absolute error:
    abs_err_max: float
    # Counts of channels outside the fp32-equivalence band:
    n_cos_below_0_9999: int
    n_rel_l2_above_1e3: int
    # Scale sanity:
    scale_abs_min: float  # smallest |scale| across (group, col)
    scale_abs_max: float


@dataclass
class ProjectionAuditResult:
    layer_idx: int
    kind: str                # q_proj, gate_proj, up_proj, down_proj
    k: int                   # rows of weight_t
    n: int                   # cols of weight_t
    groupsize: int
    stats: ChannelStats


def per_channel_stats(w_orig: np.ndarray, w_dq: np.ndarray, scales: np.ndarray) -> ChannelStats:
    """Compute per-column cos_sim, RMSE, relative L2 error between the
    pre-transposed [k, n] original and dequantized weights."""
    k, n = w_orig.shape
    diff = w_orig - w_dq  # [k, n]

    # cos_sim per column.
    dot = np.sum(w_orig * w_dq, axis=0)  # [n]
    orig_norm = np.linalg.norm(w_orig, axis=0)  # [n]
    dq_norm = np.linalg.norm(w_dq, axis=0)  # [n]
    denom = np.maximum(orig_norm * dq_norm, 1e-30)
    cos = np.clip(dot / denom, -1.0, 1.0)

    # rel L2 per column. Columns whose original norm is zero are kept at 0
    # (they can't drift).
    rel_l2 = np.where(orig_norm > 1e-30, np.linalg.norm(diff, axis=0) / orig_norm, 0.0)

    # RMSE per column.
    rmse = np.sqrt(np.mean(diff * diff, axis=0))

    abs_err = np.max(np.abs(diff), axis=0)

    cos_worst_idx = int(np.argmin(cos))
    rel_l2_worst_idx = int(np.argmax(rel_l2))

    return ChannelStats(
        n_channels=n,
        cos_min=float(np.min(cos)),
        cos_p01=float(np.quantile(cos, 0.01)),
        cos_p50=float(np.quantile(cos, 0.50)),
        cos_p99=float(np.quantile(cos, 0.99)),
        cos_max=float(np.max(cos)),
        worst_cos_channel=cos_worst_idx,
        worst_cos_value=float(cos[cos_worst_idx]),
        rel_l2_max=float(np.max(rel_l2)),
        rel_l2_p99=float(np.quantile(rel_l2, 0.99)),
        rel_l2_p50=float(np.quantile(rel_l2, 0.50)),
        worst_rel_l2_channel=rel_l2_worst_idx,
        worst_rel_l2_value=float(rel_l2[rel_l2_worst_idx]),
        rmse_max=float(np.max(rmse)),
        rmse_p50=float(np.quantile(rmse, 0.50)),
        abs_err_max=float(np.max(abs_err)),
        n_cos_below_0_9999=int(np.sum(cos < 0.9999)),
        n_rel_l2_above_1e3=int(np.sum(rel_l2 > 1e-3)),
        scale_abs_min=float(np.min(np.abs(scales))),
        scale_abs_max=float(np.max(np.abs(scales))),
    )


# -----------------------------------------------------------------------------
# Projection iteration
# -----------------------------------------------------------------------------

# (kind, hf_name_template) — template uses {prefix} and {i} placeholders.
# HF canonical layout: weight is [out, in]. Kiln transposes to [in, out]
# as `weight_t` before packing. We replicate that transpose here so k, n
# match kiln's packer contract exactly.
PROJECTION_TEMPLATES = [
    ("q_proj",    "{prefix}layers.{i}.self_attn.q_proj.weight"),
    ("gate_proj", "{prefix}layers.{i}.mlp.gate_proj.weight"),
    ("up_proj",   "{prefix}layers.{i}.mlp.up_proj.weight"),
    ("down_proj", "{prefix}layers.{i}.mlp.down_proj.weight"),
]


def audit_projection(
    checkpoint_dir: str,
    prefix: str,
    layer_idx: int,
    kind: str,
    name_template: str,
    groupsize: int = 128,
) -> Optional[ProjectionAuditResult]:
    name = name_template.format(prefix=prefix, i=layer_idx)
    w_hf = st_load_tensor(checkpoint_dir, name, allow_missing=True)
    if w_hf is None:
        return None

    # HF canonical layout: [out, in]. Kiln's packer takes [k=in, n=out].
    w_t = w_hf.t().contiguous().numpy().astype(np.float32, copy=False)
    k, n = w_t.shape

    # Marlin shape constraints: k % 128 == 0 and n % 256 == 0.
    if (k % 128) != 0 or (n % 256) != 0:
        # Falls back to BF16 in kiln; not actually Marlin-packed. Skip.
        return None

    # Kiln uses groupsize=128 for all packed projections.
    w_dq, scales = marlin_dequant_groupwise(w_t, groupsize=groupsize)
    stats = per_channel_stats(w_t, w_dq, scales)
    return ProjectionAuditResult(
        layer_idx=layer_idx,
        kind=kind,
        k=k,
        n=n,
        groupsize=groupsize,
        stats=stats,
    )


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_markdown(results: List[ProjectionAuditResult], checkpoint: str, verdict: str, any_outside_band: bool) -> str:
    lines = []
    lines.append("# Phase C11 — Marlin W4A16 per-channel scale drift audit")
    lines.append("")
    lines.append(f"**Status:** {verdict}")
    lines.append("")
    lines.append(f"**Checkpoint:** `{checkpoint}` (HF canonical bf16; cast to fp32 for the audit)")
    lines.append("")
    lines.append("**Bar:** per-output-channel `cos_sim >= 0.9999` AND `rel_l2_error <= 1e-3` — the same fp32-equivalence band C10 used behaviorally, now applied numerically to Marlin's weight round-trip.")
    lines.append("")
    lines.append("**Scope:** all 104 Marlin-packed main-model projections (8 x q_proj on full-attention layers + 32 x gate_proj + 32 x up_proj + 32 x down_proj), groupsize=128. The MTP head itself stays BF16 per `forward.rs` (`// For WIP scaffolding the MTP transformer layer is kept in BF16 and is NOT queued for Marlin batch packing`).")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if any_outside_band:
        lines.append("❌ **Per-channel scale drift found outside the fp32-equivalence band.** Marlin packing IS a plausible alpha-suppressing signal; promote to C12 primary target.")
    else:
        lines.append("✅ **Null result.** Every one of the 104 Marlin-packed projections has all output channels fully inside the fp32-equivalence band (cos_sim >= 0.9999, rel_l2 <= 1e-3). Marlin W4A16 packing does NOT produce the per-channel scale drift that could drive the Class B rejection rate.")
        lines.append("")
        lines.append("**Implication for C12:** Promote C10 target #2 — **fp32 draft head policy** — to the primary next target. The Class B rejections and α ≈ 0.058 floor cannot be pinned on Marlin weight numerics under the fp32-equivalence bar C9/C10 established.")
    lines.append("")

    # Aggregate the cross-projection worst-case.
    worst_proj = min(results, key=lambda r: r.stats.cos_min)
    worst_l2_proj = max(results, key=lambda r: r.stats.rel_l2_max)
    n_total_channels = sum(r.stats.n_channels for r in results)
    n_total_outside = sum(r.stats.n_cos_below_0_9999 for r in results)
    n_total_l2_outside = sum(r.stats.n_rel_l2_above_1e3 for r in results)

    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- **Projections audited:** {len(results)}")
    lines.append(f"- **Output channels total:** {n_total_channels}")
    lines.append(f"- **Channels with cos_sim < 0.9999:** {n_total_outside} ({100.0 * n_total_outside / max(n_total_channels, 1):.3f} %)")
    lines.append(f"- **Channels with rel_l2 > 1e-3:** {n_total_l2_outside} ({100.0 * n_total_l2_outside / max(n_total_channels, 1):.3f} %)")
    lines.append("")
    lines.append(f"- **Worst cos_sim across all projections:** `{worst_proj.stats.cos_min:.6f}` at `{worst_proj.kind}[layer={worst_proj.layer_idx}]` channel `{worst_proj.stats.worst_cos_channel}`")
    lines.append(f"- **Worst rel_l2 across all projections:** `{worst_l2_proj.stats.rel_l2_max:.6f}` at `{worst_l2_proj.kind}[layer={worst_l2_proj.layer_idx}]` channel `{worst_l2_proj.stats.worst_rel_l2_channel}`")
    lines.append("")

    # Per-projection table.
    lines.append("## Per-projection summary")
    lines.append("")
    lines.append("| Layer | Kind | k x n | cos_min | cos_p01 | cos_p50 | rel_l2_max | rel_l2_p99 | rmse_max | n_cos<0.9999 | n_rel_l2>1e-3 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        s = r.stats
        lines.append(
            f"| {r.layer_idx} | {r.kind} | {r.k}x{r.n} | "
            f"{s.cos_min:.6f} | {s.cos_p01:.6f} | {s.cos_p50:.6f} | "
            f"{s.rel_l2_max:.6f} | {s.rel_l2_p99:.6f} | {s.rmse_max:.6e} | "
            f"{s.n_cos_below_0_9999} | {s.n_rel_l2_above_1e3} |"
        )
    lines.append("")
    lines.append("## Cross-reference with C10 Class B rejection positions")
    lines.append("")
    lines.append("C10 reported `kiln_matches_ref=True` on all 3 audited positions (seeds 42/43/44 at `mtp_pos=2`, draft tokens 3074 / 16078 / 1814). Those positions were accepted, not Class B rejections, so C10 does not expose direct rejection positions we can cross-reference against specific Marlin output channels.")
    lines.append("")
    lines.append("The per-channel scale drift audit above is therefore a **necessary-condition** check, not a sufficient-condition one:")
    lines.append("")
    lines.append("- If **any** channel had drifted outside the band, we would have had a strong prior for channel-level attribution of Class B flips, and C12 would have been a targeted quantization fix.")
    lines.append("- The null result means the fp32-equivalence band is preserved at the WEIGHT level for every packed projection. Residual alpha suppression must therefore come from **activation-weight interaction** (which this audit does not measure) OR a policy-level change (fp32 draft head, C10 target #2).")
    lines.append("")
    lines.append("## Honest limits")
    lines.append("")
    lines.append("1. **Weight-only, not activation-weighted.** A channel that is in-band on weights alone may still amplify drift when multiplied by a large-magnitude activation. We do NOT measure this here. A follow-up audit under the rejected Class B activation distribution would close this gap.")
    lines.append("2. **No KV cache FP8 coupling.** Kiln optionally runs `KILN_KV_CACHE_FP8=true`; this audit is orthogonal to that path.")
    lines.append("3. **MTP layer untouched.** The MTP head is BF16-only in kiln today; this audit has nothing to say about MTP-layer numerics (C10 closed that cleanly with the tap bisect).")
    lines.append("4. **Scale round-trip is deterministic.** We reproduce kiln's f16 round-trip of the scale. The kernel's FP16-only mma (`mma.m16n8k16.f32.f16.f16.f32`) adds an additional bf16->fp16 cast on the activation side, not audited here.")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_json(results: List[ProjectionAuditResult], any_outside_band: bool, checkpoint: str) -> str:
    payload = {
        "checkpoint": checkpoint,
        "bar": {"cos_sim_min": 0.9999, "rel_l2_max": 1e-3},
        "any_channel_outside_band": any_outside_band,
        "projections": [
            {
                "layer_idx": r.layer_idx,
                "kind": r.kind,
                "k": r.k,
                "n": r.n,
                "groupsize": r.groupsize,
                "stats": dataclasses.asdict(r.stats),
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF checkpoint dir containing model.safetensors")
    ap.add_argument("--out", required=True, help="Path to write markdown audit report")
    ap.add_argument("--out-json", required=True, help="Path to write structured JSON report")
    ap.add_argument("--num-layers", type=int, default=32, help="Number of main-model layers (default 32 for Qwen3.5-4B)")
    args = ap.parse_args()

    ckpt = args.checkpoint
    prefix = discover_prefix(ckpt)
    print(f"[c11] checkpoint prefix = `{prefix}`", file=sys.stderr)

    t0 = time.time()
    results: List[ProjectionAuditResult] = []
    for layer_idx in range(args.num_layers):
        for kind, template in PROJECTION_TEMPLATES:
            # q_proj only exists on full-attention layers. The loader probes
            # for it; we do the same.
            if kind == "q_proj" and not layer_is_full_attention(ckpt, prefix, layer_idx):
                continue
            res = audit_projection(ckpt, prefix, layer_idx, kind, template)
            if res is None:
                # Either missing from checkpoint or shape not Marlin-packable.
                continue
            results.append(res)
            s = res.stats
            print(
                f"[c11] layer={layer_idx:>2} {kind:<9} {res.k}x{res.n} "
                f"cos_min={s.cos_min:.6f} rel_l2_max={s.rel_l2_max:.6f} "
                f"n_out_band={s.n_cos_below_0_9999}",
                file=sys.stderr,
            )

    elapsed = time.time() - t0
    print(f"[c11] audited {len(results)} projections in {elapsed:.1f}s", file=sys.stderr)

    any_outside_band = any(
        r.stats.n_cos_below_0_9999 > 0 or r.stats.n_rel_l2_above_1e3 > 0
        for r in results
    )
    verdict = (
        "**Result:** per-channel scale drift FOUND outside the fp32-equivalence band — Marlin is a plausible α-suppressing signal."
        if any_outside_band
        else "**Null result** — every one of the audited Marlin-packed projections has all output channels fully inside the fp32-equivalence band (cos_sim >= 0.9999, rel_l2 <= 1e-3)."
    )

    md = render_markdown(results, ckpt, verdict, any_outside_band)
    js = render_json(results, any_outside_band, ckpt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"[c11] wrote {args.out}", file=sys.stderr)
    print(f"[c11] wrote {args.out_json}", file=sys.stderr)

    return 1 if any_outside_band else 0


if __name__ == "__main__":
    sys.exit(main())
