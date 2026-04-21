#!/usr/bin/env python3
"""
Phase C10 — fp32 HF reference comparator + top1-flip per-site bisect.

Goal
----

The Class B rejection audit after C8/C9 left MTP α at ~0.058 (median across
seeds 42/43/44), with ~87.6% of rejections being Class B (draft top1 ≠ main
top1). Per-tap cos_sim bisects (Phases B6-C7) show every MTP-head tap above
the cos_sim ≥ 0.9999 bar. C9's verdict: the remaining drift looks like bf16
GEMM/accumulation noise, not a semantic bug.

C10 sharpens that claim using a *behavioral* metric instead of a purely
numerical one. For each tap S in the MTP forward path:

    1. Compute pure-fp32 HF reference forward -> ref_logits -> ref_top1.
    2. Splice kiln's bf16 activation at S (upcast to fp32) into the reference
       path and continue the rest of the forward in fp32 -> spliced_logits_S
       -> spliced_top1_S.
    3. Count flips: did this single bf16-vs-fp32 substitution at tap S flip
       the final argmax?

If every tap has flip_vs_ref = False across multiple (seed, position) pairs,
the argmax path is robust to bf16 noise at every site and the C9 "benign
drift" verdict survives. If a specific tap consistently flips, that tap is
the operator whose bf16 materialization is responsible for Class B flips —
i.e. the C11 target.

Design notes
------------

* Reuses `mtp_reference_dump.py` helpers: `rms_norm`, `apply_rope_partial`,
  `load_mtp_weights`, `MtpRefWeights`. Those are verified against HF.
* The reference path here is rebuilt inline (not called out to
  `mtp_inner_block`) because splicing requires replacing *local variables*
  mid-forward, which the existing closed-over function doesn't expose.
* h_main is still taken from the kiln dump (same honest limit as B6/C6/C7).
  C10 bisects the MTP head in isolation; the main-model hidden state
  hand-off is audited separately in B10/B11b/B12.
* draft_token_id and positions come from the kiln dump metadata.
* Weights are fp32 (upcast in st_load_tensor) throughout.

Tap sites
---------

Ordered top-down through the MTP head. Each is a distinct variable in the
forward path where kiln's bf16 dump writes a tensor we can splice in.

    A-group (pre-layer)
        tok_embed, norm_emb, norm_h, fc_input (== concat), fc_output (== fused)
    B-group (inside mtp_inner_block — SDPA path)
        post_pre_attn_norm
        post_q_proj_raw, post_k_proj, post_v_proj
        post_q_split, post_gate_split   (q_raw is reshaped here; gate feeds post-attn sigmoid)
        post_q_norm, post_k_norm
        post_q_rope, post_k_rope
        attn_out (== post_attn_raw)
        post_attn_gated
        post_o_proj
        post_attn_residual
        post_pre_mlp_norm
        post_mlp
    C-group (post-layer)
        post_layer, post_final_ln, mtp_logits

`mtp_logits` is the terminal site: splicing there just takes kiln's argmax
and tells us "does kiln's bf16 argmax match fp32 ref?".

Exit codes
----------

    0 — comparator ran; per-tap table + verdict written successfully
    1 — comparator detected >=1 tap whose splice flipped the argmax at
         least once across the requested pairs (legitimate signal)
    2 — structural error (missing weight, shape mismatch, bad CLI)

Usage
-----

    python3 scripts/mtp_c10_splice_bisect.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --pair seed42-pos1:/tmp/kiln-seed42-pos1.safetensors \\
        --pair seed42-pos2:/tmp/kiln-seed42-pos2.safetensors \\
        [--pair ...]
        --out docs/phase-c10/c10_splice_bisect_table.md
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open


# -----------------------------------------------------------------------------
# Reuse reference-dump helpers (vendored inline to keep this script
# self-contained; identical to the canonical definitions in
# mtp_reference_dump.py).
# -----------------------------------------------------------------------------


def st_load_tensor(checkpoint_dir: str, name: str, allow_missing: bool = False) -> Optional[torch.Tensor]:
    import glob
    import json
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


@dataclass
class MtpRefWeights:
    embed_tokens: torch.Tensor
    fc_weight: torch.Tensor
    pre_fc_norm_embedding: torch.Tensor
    pre_fc_norm_hidden: torch.Tensor
    final_layernorm: torch.Tensor
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    q_proj_weight: torch.Tensor
    k_proj_weight: torch.Tensor
    v_proj_weight: torch.Tensor
    o_proj_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    gate_proj_weight: torch.Tensor
    up_proj_weight: torch.Tensor
    down_proj_weight: torch.Tensor
    config: dict


MTP_PREFIXES = ["mtp.", "model.mtp.", "model.language_model.mtp."]
EMBED_PREFIXES = [
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "embed_tokens.weight",
]


def load_mtp_weights(checkpoint_dir: str) -> MtpRefWeights:
    import json
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        raw = json.load(f)
    cfg = raw.get("text_config") or raw.get("language_config") or raw

    embed = None
    for ep in EMBED_PREFIXES:
        try:
            embed = st_load_tensor(checkpoint_dir, ep)
            break
        except KeyError:
            continue
    if embed is None:
        raise KeyError("No embed_tokens.weight under any known prefix")

    mtp_prefix = None
    mtp_fc = None
    for p in MTP_PREFIXES:
        try:
            mtp_fc = st_load_tensor(checkpoint_dir, p + "fc.weight")
            mtp_prefix = p
            break
        except KeyError:
            continue
    if mtp_fc is None:
        raise KeyError(f"No `mtp.fc.weight` under any of {MTP_PREFIXES}")

    def get(suf): return st_load_tensor(checkpoint_dir, mtp_prefix + suf)

    def first_match(*names):
        for n in names:
            t = st_load_tensor(checkpoint_dir, n, allow_missing=True)
            if t is not None:
                return t
        raise KeyError(f"None of {names} present")

    return MtpRefWeights(
        embed_tokens=embed,
        fc_weight=mtp_fc,
        pre_fc_norm_embedding=get("pre_fc_norm_embedding.weight"),
        pre_fc_norm_hidden=get("pre_fc_norm_hidden.weight"),
        final_layernorm=first_match(
            mtp_prefix + "norm.weight",
            mtp_prefix + "final_layernorm.weight",
        ),
        input_layernorm=first_match(
            mtp_prefix + "layers.0.input_layernorm.weight",
            mtp_prefix + "layer.input_layernorm.weight",
        ),
        post_attention_layernorm=first_match(
            mtp_prefix + "layers.0.post_attention_layernorm.weight",
            mtp_prefix + "layer.post_attention_layernorm.weight",
        ),
        q_proj_weight=first_match(
            mtp_prefix + "layers.0.self_attn.q_proj.weight",
            mtp_prefix + "layer.self_attn.q_proj.weight",
        ),
        k_proj_weight=first_match(
            mtp_prefix + "layers.0.self_attn.k_proj.weight",
            mtp_prefix + "layer.self_attn.k_proj.weight",
        ),
        v_proj_weight=first_match(
            mtp_prefix + "layers.0.self_attn.v_proj.weight",
            mtp_prefix + "layer.self_attn.v_proj.weight",
        ),
        o_proj_weight=first_match(
            mtp_prefix + "layers.0.self_attn.o_proj.weight",
            mtp_prefix + "layer.self_attn.o_proj.weight",
        ),
        q_norm_weight=first_match(
            mtp_prefix + "layers.0.self_attn.q_norm.weight",
            mtp_prefix + "layer.self_attn.q_norm.weight",
        ),
        k_norm_weight=first_match(
            mtp_prefix + "layers.0.self_attn.k_norm.weight",
            mtp_prefix + "layer.self_attn.k_norm.weight",
        ),
        gate_proj_weight=first_match(
            mtp_prefix + "layers.0.mlp.gate_proj.weight",
            mtp_prefix + "layer.mlp.gate_proj.weight",
        ),
        up_proj_weight=first_match(
            mtp_prefix + "layers.0.mlp.up_proj.weight",
            mtp_prefix + "layer.mlp.up_proj.weight",
        ),
        down_proj_weight=first_match(
            mtp_prefix + "layers.0.mlp.down_proj.weight",
            mtp_prefix + "layer.mlp.down_proj.weight",
        ),
        config=cfg,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    w = weight.to(torch.float32)
    return (x * (1.0 + w)).to(orig)


def apply_rope_partial(
    x: torch.Tensor, position: int, head_dim: int, rotary_dim: int, rope_theta: float
) -> torch.Tensor:
    if rotary_dim == 0:
        return x
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.tensor([position], dtype=torch.float32)[:, None] * inv_freq[None, :]
    cos = freqs.cos()
    sin = freqs.sin()
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    x_rot_new = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([x_rot_new, x_pass], dim=-1)


# -----------------------------------------------------------------------------
# Splice-aware MTP forward (fp32)
# -----------------------------------------------------------------------------

# Canonical tap order — top-down. Each name corresponds to a local variable
# in `full_forward_fp32` whose value the override dict can replace AFTER
# natural computation but BEFORE that variable is used downstream.
TAP_ORDER: Tuple[str, ...] = (
    # A-group: pre-layer
    "tok_embed",
    "norm_emb",
    "norm_h",
    "fc_input",
    "fc_output",
    # B-group: inside inner block
    "post_pre_attn_norm",
    "post_q_proj_raw",
    "post_k_proj",
    "post_v_proj",
    "post_q_split",
    "post_gate_split",
    "post_q_norm",
    "post_k_norm",
    "post_q_rope",
    "post_k_rope",
    "attn_out",
    "post_attn_gated",
    "post_o_proj",
    "post_attn_residual",
    "post_pre_mlp_norm",
    "post_mlp",
    # C-group: post-layer
    "post_layer",
    "post_final_ln",
    "mtp_logits",
)

# Kiln dump naming -> our local tap name. Some tap names differ between kiln
# and the reference; map kiln's dump key to our variable name here so we can
# look up the bf16 override to splice in.
KILN_TAP_ALIAS: Dict[str, str] = {
    # A-group: kiln writes these at the top level (same names as ours).
    "tok_embed": "tok_embed",
    "fc_input": "fc_input",
    "fc_output": "fc_output",
    "pre_layer": "fc_output",        # kiln aliases fc_output -> pre_layer
    # B-group: kiln writes sub-op taps under bare names when KILN_MTP_DUMP_SUBOPS=1.
    "post_pre_attn_norm": "post_pre_attn_norm",
    "post_q_proj_raw": "post_q_proj_raw",
    "post_k_proj": "post_k_proj",
    "post_v_proj": "post_v_proj",
    "post_q_split": "post_q_split",
    "post_gate_split": "post_gate_split",
    "post_q_norm": "post_q_norm",
    "post_k_norm": "post_k_norm",
    "post_q_rope": "post_q_rope",
    "post_k_rope": "post_k_rope",
    "post_attn_raw": "attn_out",
    "post_attn_gated": "post_attn_gated",
    "post_o_proj": "post_o_proj",
    "post_attn_residual": "post_attn_residual",
    "post_pre_mlp_norm": "post_pre_mlp_norm",
    "post_mlp": "post_mlp",
    # C6 pre-RoPE taps
    "c6__token_emb": "tok_embed",
    "c6__norm_emb": "norm_emb",
    "c6__norm_h": "norm_h",
    "c6__concat": "fc_input",
    "c6__fused": "fc_output",
    # C-group
    "post_layer": "post_layer",
    "post_final_ln": "post_final_ln",
    "mtp_logits": "mtp_logits",
}


def _apply_override(
    name: str,
    val: torch.Tensor,
    override_tap: Optional[str],
    override_val: Optional[torch.Tensor],
) -> torch.Tensor:
    if override_tap is None or name != override_tap:
        return val
    # Cast kiln bf16 -> fp32 and reshape to match.
    out = override_val.to(torch.float32)
    if out.shape != val.shape:
        # Allow broadcast-compatible reshape (kiln sometimes writes flat
        # [1,1,H*D]; our local may be [1,1,H,D]).
        try:
            out = out.reshape(val.shape)
        except Exception as e:
            raise RuntimeError(
                f"splice shape mismatch for `{name}`: kiln={tuple(override_val.shape)} ref={tuple(val.shape)}"
            ) from e
    return out.contiguous()


def full_forward_fp32(
    w: MtpRefWeights,
    h_main: torch.Tensor,
    draft_token_id: int,
    mtp_pos: int,
    base_pos: int,
    *,
    swap_fc_norms: int = 0,
    override_tap: Optional[str] = None,
    override_val: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run MTP head forward in fp32. If (override_tap, override_val) given,
    replace the local variable at `override_tap` with `override_val` AFTER
    its natural computation, then continue forward. Returns `mtp_logits`
    (fp32, shape [1, 1, V]).
    """
    cfg = w.config
    H = h_main.shape[-1]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = cfg.get("head_dim", H // num_heads)
    rotary_theta = float(os.environ.get("KILN_REF_ROPE_THETA", cfg.get("rope_theta", 10_000_000.0)) or 10_000_000.0)
    rotary_frac = float(os.environ.get("KILN_REF_ROTARY_FRAC", cfg.get("partial_rotary_factor", 0.25)) or 0.25)
    rotary_dim = int(head_dim * rotary_frac)
    eps = 1e-6

    # All state in fp32.
    h_main = h_main.to(torch.float32)
    ov = lambda name, v: _apply_override(name, v, override_tap, override_val)

    # 1. tok_embed.
    tok_embed = w.embed_tokens[draft_token_id : draft_token_id + 1].unsqueeze(0).to(torch.float32)  # [1,1,H]
    tok_embed = ov("tok_embed", tok_embed)

    # 2-3. Dual RMSNorms with swap flag for parity.
    if swap_fc_norms:
        norm_emb_w = w.pre_fc_norm_hidden
        norm_h_w = w.pre_fc_norm_embedding
    else:
        norm_emb_w = w.pre_fc_norm_embedding
        norm_h_w = w.pre_fc_norm_hidden
    norm_emb = rms_norm(tok_embed, norm_emb_w, eps)
    norm_emb = ov("norm_emb", norm_emb)
    norm_h = rms_norm(h_main, norm_h_w, eps)
    norm_h = ov("norm_h", norm_h)

    # 4. Concat + fc.
    fc_input = torch.cat([norm_emb, norm_h], dim=-1)  # [1,1,2H]
    fc_input = ov("fc_input", fc_input)
    fc_output = torch.matmul(fc_input, w.fc_weight.to(torch.float32).t())  # [1,1,H]
    fc_output = ov("fc_output", fc_output)

    # 5. Inner block.
    x = fc_output
    residual = x
    h = rms_norm(x, w.input_layernorm, eps)
    h = ov("post_pre_attn_norm", h)

    q_raw = torch.matmul(h, w.q_proj_weight.to(torch.float32).t())
    q_raw = ov("post_q_proj_raw", q_raw)
    k_flat = torch.matmul(h, w.k_proj_weight.to(torch.float32).t())
    k_flat = ov("post_k_proj", k_flat)
    v_flat = torch.matmul(h, w.v_proj_weight.to(torch.float32).t())
    v_flat = ov("post_v_proj", v_flat)

    q_gated = (q_raw.shape[-1] == 2 * num_heads * head_dim)
    if q_gated:
        q_pair = q_raw.view(1, 1, num_heads, head_dim * 2)
        q = q_pair[..., :head_dim].contiguous()  # [1,1,num_heads,head_dim]
        gate = q_pair[..., head_dim:].contiguous().view(1, 1, num_heads * head_dim)
    else:
        q = q_raw.view(1, 1, num_heads, head_dim)
        gate = None
    q = ov("post_q_split", q)
    if gate is not None:
        gate = ov("post_gate_split", gate)

    k = k_flat.view(1, 1, num_kv_heads, head_dim)
    v = v_flat.view(1, 1, num_kv_heads, head_dim)

    # Per-head RMSNorm
    q = rms_norm(q, w.q_norm_weight, eps)
    q = ov("post_q_norm", q)
    k = rms_norm(k, w.k_norm_weight, eps)
    k = ov("post_k_norm", k)

    # RoPE at abs_pos = base_pos + mtp_pos
    abs_pos = base_pos + mtp_pos
    q = apply_rope_partial(q, abs_pos, head_dim, rotary_dim, rotary_theta)
    q = ov("post_q_rope", q)
    k = apply_rope_partial(k, abs_pos, head_dim, rotary_dim, rotary_theta)
    k = ov("post_k_rope", k)

    # Attention.
    q_t = q.transpose(1, 2).contiguous()   # [1, num_heads, 1, head_dim]
    k_t = k.transpose(1, 2).contiguous()   # [1, num_kv_heads, 1, head_dim]
    v_t = v.transpose(1, 2).contiguous()
    repeat = num_heads // num_kv_heads
    if repeat > 1:
        k_full = k_t.repeat_interleave(repeat, dim=1)
        v_full = v_t.repeat_interleave(repeat, dim=1)
    else:
        k_full = k_t
        v_full = v_t
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_t, k_full.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    attn_out_t = torch.matmul(attn, v_full)  # [1, num_heads, 1, head_dim]
    attn_out = attn_out_t.transpose(1, 2).contiguous().view(1, 1, num_heads * head_dim)
    attn_out = ov("attn_out", attn_out)

    # Gated-attn sigmoid.
    if gate is not None:
        attn_out = attn_out * torch.sigmoid(gate)
    attn_out = ov("post_attn_gated", attn_out)

    # o_proj
    o = torch.matmul(attn_out, w.o_proj_weight.to(torch.float32).t())
    o = ov("post_o_proj", o)

    # Residual
    h = residual + o
    h = ov("post_attn_residual", h)

    # MLP path
    residual2 = h
    h2 = rms_norm(h, w.post_attention_layernorm, eps)
    h2 = ov("post_pre_mlp_norm", h2)
    gate_p = torch.matmul(h2, w.gate_proj_weight.to(torch.float32).t())
    up_p = torch.matmul(h2, w.up_proj_weight.to(torch.float32).t())
    act = torch.nn.functional.silu(gate_p) * up_p
    down = torch.matmul(act, w.down_proj_weight.to(torch.float32).t())
    down = ov("post_mlp", down)
    post_layer = residual2 + down
    post_layer = ov("post_layer", post_layer)

    # Final norm + LM head
    post_final_ln = rms_norm(post_layer, w.final_layernorm, eps)
    post_final_ln = ov("post_final_ln", post_final_ln)
    mtp_logits = torch.matmul(post_final_ln, w.embed_tokens.to(torch.float32).t())
    mtp_logits = ov("mtp_logits", mtp_logits)

    return mtp_logits


# -----------------------------------------------------------------------------
# Kiln dump loader
# -----------------------------------------------------------------------------


def load_kiln_dump(path: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name.startswith("meta__"):
                out[name] = int(t.flatten()[0].item())
            else:
                out[name] = t
    return out


def kiln_override_for_tap(kiln: Dict[str, object], local_tap: str) -> Optional[torch.Tensor]:
    """Find the kiln dump tensor corresponding to `local_tap`. Tries the
    canonical tap name first, then any alias from KILN_TAP_ALIAS whose
    value equals `local_tap`."""
    if local_tap in kiln and not local_tap.startswith("meta__"):
        return kiln[local_tap]  # type: ignore[return-value]
    for kname, alias_local in KILN_TAP_ALIAS.items():
        if alias_local == local_tap and kname in kiln:
            return kiln[kname]  # type: ignore[return-value]
    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_pair(
    label: str,
    kiln_path: str,
    w: MtpRefWeights,
) -> List[Dict[str, object]]:
    """For a single kiln dump, compute (ref_top1, kiln_top1) and per-tap
    splice results. Returns a list of row dicts."""
    kiln = load_kiln_dump(kiln_path)
    if "h_main" not in kiln:
        print(f"[c10] {label}: kiln dump missing h_main; skipping", file=sys.stderr)
        return []
    h_main = kiln["h_main"].to(torch.float32)  # type: ignore[attr-defined]
    draft_token_id = int(kiln["meta__draft_token_id"])
    mtp_pos = int(kiln["meta__mtp_pos"])
    base_pos = int(kiln.get("meta__base_pos", 0))
    swap = int(kiln.get("meta__swap_fc_norms", 0))

    # Pure-fp32 reference.
    ref_logits = full_forward_fp32(
        w, h_main, draft_token_id, mtp_pos, base_pos, swap_fc_norms=swap,
    )
    ref_logits_np = ref_logits.view(-1).cpu().numpy().astype(np.float32)
    ref_top1 = int(np.argmax(ref_logits_np))
    ref_top5 = np.argsort(-ref_logits_np)[:5].tolist()

    # Kiln bf16 argmax (from kiln's own mtp_logits tap, cast to fp32 for argmax).
    kiln_logits = kiln["mtp_logits"].to(torch.float32).view(-1).cpu().numpy().astype(np.float32)  # type: ignore[attr-defined]
    kiln_top1 = int(np.argmax(kiln_logits))

    rows: List[Dict[str, object]] = [
        {
            "label": label,
            "tap": "__REFERENCE__",
            "has_kiln": True,
            "ref_top1": ref_top1,
            "kiln_top1": kiln_top1,
            "kiln_matches_ref": (kiln_top1 == ref_top1),
            "spliced_top1": ref_top1,  # no splice
            "spliced_flips_vs_ref": False,
            "ref_top5": ",".join(str(x) for x in ref_top5),
            "notes": f"draft_token_id={draft_token_id} mtp_pos={mtp_pos} base_pos={base_pos}",
        }
    ]

    # Per-tap splice.
    for tap in TAP_ORDER:
        kiln_val = kiln_override_for_tap(kiln, tap)
        if kiln_val is None:
            rows.append({
                "label": label,
                "tap": tap,
                "has_kiln": False,
                "ref_top1": ref_top1,
                "kiln_top1": kiln_top1,
                "spliced_top1": None,
                "spliced_flips_vs_ref": None,
                "notes": "kiln dump missing this tap",
            })
            continue
        try:
            spliced_logits = full_forward_fp32(
                w, h_main, draft_token_id, mtp_pos, base_pos,
                swap_fc_norms=swap,
                override_tap=tap,
                override_val=kiln_val if isinstance(kiln_val, torch.Tensor) else torch.as_tensor(kiln_val),
            )
        except RuntimeError as e:
            rows.append({
                "label": label,
                "tap": tap,
                "has_kiln": True,
                "ref_top1": ref_top1,
                "kiln_top1": kiln_top1,
                "spliced_top1": None,
                "spliced_flips_vs_ref": None,
                "notes": f"splice failed: {e}",
            })
            continue
        spliced_np = spliced_logits.view(-1).cpu().numpy().astype(np.float32)
        spliced_top1 = int(np.argmax(spliced_np))
        flipped = (spliced_top1 != ref_top1)
        # Margin between ref_top1 and its closest competitor in spliced logits
        ref_logit = float(spliced_np[ref_top1])
        mask = np.ones_like(spliced_np, dtype=bool)
        mask[ref_top1] = False
        runner_up = float(spliced_np[mask].max())
        margin = ref_logit - runner_up
        rows.append({
            "label": label,
            "tap": tap,
            "has_kiln": True,
            "ref_top1": ref_top1,
            "kiln_top1": kiln_top1,
            "spliced_top1": spliced_top1,
            "spliced_flips_vs_ref": flipped,
            "ref_margin_after_splice": margin,
            "notes": "",
        })

    return rows


def render_markdown(all_rows: List[Dict[str, object]]) -> str:
    # Group by tap across labels.
    labels = []
    for r in all_rows:
        if r["label"] not in labels:
            labels.append(r["label"])  # preserve insertion order

    # First: per-pair summary.
    lines: List[str] = []
    lines.append("## Per-pair summary")
    lines.append("")
    lines.append("| pair | ref_top1 | kiln_top1 | kiln_matches_ref | notes |")
    lines.append("|------|----------|-----------|------------------|-------|")
    for lbl in labels:
        meta = [r for r in all_rows if r["label"] == lbl and r["tap"] == "__REFERENCE__"]
        if not meta:
            continue
        m = meta[0]
        lines.append(
            f"| {lbl} | {m['ref_top1']} | {m['kiln_top1']} | {m['kiln_matches_ref']} | {m.get('notes','')} |"
        )
    lines.append("")

    # Second: per-tap flip table, one column per pair.
    lines.append("## Per-tap splice flips (kiln bf16 → fp32 reference)")
    lines.append("")
    header = "| tap | " + " | ".join(f"flip@{lbl}" for lbl in labels) + " | total_flips |"
    sep = "|-----|" + "|".join(["-" * max(4, len(f"flip@{lbl}")) for lbl in labels]) + "|-------------|"
    lines.append(header)
    lines.append(sep)

    # Build lookup.
    tap_pair_flip: Dict[str, Dict[str, Optional[bool]]] = {}
    for r in all_rows:
        if r["tap"] == "__REFERENCE__":
            continue
        tap = str(r["tap"])
        tap_pair_flip.setdefault(tap, {})
        tap_pair_flip[tap][str(r["label"])] = r.get("spliced_flips_vs_ref")  # type: ignore[assignment]

    for tap in TAP_ORDER:
        row = tap_pair_flip.get(tap, {})
        cells = []
        flips = 0
        for lbl in labels:
            v = row.get(lbl)
            if v is None:
                cells.append("—")
            elif v:
                cells.append("FLIP")
                flips += 1
            else:
                cells.append(".")
        lines.append(f"| {tap} | " + " | ".join(cells) + f" | {flips} |")
    lines.append("")

    # Third: per-tap margin shrinkage (ref_margin_after_splice). Useful to see
    # *how close* each splice is to flipping, even when it doesn't.
    lines.append("## Per-tap ref_top1 margin after splice (fp32-logit units)")
    lines.append("")
    header = "| tap | " + " | ".join(f"margin@{lbl}" for lbl in labels) + " |"
    sep = "|-----|" + "|".join(["-" * max(6, len(f"margin@{lbl}")) for lbl in labels]) + "|"
    lines.append(header)
    lines.append(sep)
    for tap in TAP_ORDER:
        cells = []
        for lbl in labels:
            matching = [r for r in all_rows if r["tap"] == tap and r["label"] == lbl]
            if not matching or "ref_margin_after_splice" not in matching[0]:
                cells.append("—")
                continue
            m = matching[0].get("ref_margin_after_splice")
            if m is None:
                cells.append("—")
            else:
                cells.append(f"{float(m):+.4f}")
        lines.append(f"| {tap} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase C10 — fp32 HF reference + top1-flip per-site bisect")
    ap.add_argument("--checkpoint", required=True, help="Qwen3.5-4B checkpoint directory")
    ap.add_argument(
        "--pair",
        action="append",
        required=True,
        help="LABEL:KILN_DUMP path (may be given multiple times, one per seed/pos pair)",
    )
    ap.add_argument("--out", required=True, help="Output markdown report path")
    args = ap.parse_args()

    pairs: List[Tuple[str, str]] = []
    for p in args.pair:
        if ":" not in p:
            print(f"[c10] bad --pair `{p}`; expected LABEL:PATH", file=sys.stderr)
            return 2
        lbl, path = p.split(":", 1)
        if not os.path.exists(path):
            print(f"[c10] missing kiln dump at `{path}`", file=sys.stderr)
            return 2
        pairs.append((lbl, path))

    print(f"[c10] loading weights from {args.checkpoint}", file=sys.stderr)
    w = load_mtp_weights(args.checkpoint)

    all_rows: List[Dict[str, object]] = []
    any_flip = False
    for lbl, path in pairs:
        print(f"[c10] running pair `{lbl}` <- {path}", file=sys.stderr)
        rows = run_pair(lbl, path, w)
        for r in rows:
            if r["tap"] != "__REFERENCE__" and r.get("spliced_flips_vs_ref"):
                any_flip = True
        all_rows.extend(rows)

    md = render_markdown(all_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[c10] wrote {args.out} (any_flip={any_flip})", file=sys.stderr)

    return 1 if any_flip else 0


if __name__ == "__main__":
    sys.exit(main())
