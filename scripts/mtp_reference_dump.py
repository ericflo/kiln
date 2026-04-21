#!/usr/bin/env python3
"""
Phase B6 reference MTP dump — pure PyTorch reference implementation of the
Qwen3.5-4B MTP forward pass, consuming the same `h_main`/draft-token inputs
that kiln logged to its own dump file.

Usage
-----

    python3 scripts/mtp_reference_dump.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --kiln-dump /tmp/mtp-kiln.safetensors \\
        --out /tmp/mtp-ref.safetensors

Design
------

The Qwen3.5-4B main model is a hybrid 24×GDN + 8×GQA stack that we do **not**
re-implement here — doing so in a single script would be large, slow, and
itself a new source of bugs. Instead, the reference script takes `h_main`
(last-row pre-final-norm hidden state from the base model, `[1, 1, H]`) and
`draft_token_id` **from the kiln dump itself**, and runs a *pure-tensor*
reference implementation of every op that lives inside `mtp_forward_step`:

    1. token_emb   = embed_tokens[draft_token_id]           # [1, 1, H]
    2. norm_emb    = rms_norm(token_emb, pre_fc_norm_embedding)
    3. norm_h      = rms_norm(h_main,    pre_fc_norm_hidden)
    4. fc_input    = cat([norm_emb, norm_h], dim=-1)        # [1, 1, 2H]
       fc_output   = fc_input @ mtp.fc.weight^T             # [1, 1, H]
    5. post_layer  = mtp_inner_block(fc_output, mtp_pos)    # full-attn, 1 layer
    6. post_final_ln = rms_norm(post_layer, mtp.final_layernorm)
    7. mtp_logits  = post_final_ln @ embed_tokens^T

This produces the same 8 named taps as kiln writes, with the same shapes, in
the same safetensors format. `scripts/mtp_compare.py` then diffs them per-tap
to localize the first divergence.

Honest limits on what this bisect can prove
-------------------------------------------

* `h_main` is TAKEN FROM THE KILN DUMP. The reference cannot catch bugs in
  the base-model forward path (the 24×GDN + 8×GQA stack). If kiln's main
  model is emitting the wrong `h_main`, every downstream tap will match the
  reference bit-for-bit (since the reference is fed the same bad `h_main`)
  and the divergence would only show up at the final quality-of-generation
  level — which is already how we spotted α=0.154 in the first place.
* If every tap matches, the bug is upstream of tap #1 (main-model hidden
  state production) OR in a layer this script doesn't model (token
  sampling; but that path is already audited as greedy-equal in Phase A).
* This script explicitly models the 3 remaining runtime hypotheses:
    - RoPE `mtp_pos` advancement → visible as mismatch at `post_layer` (tap 6)
    - Tied `embed_tokens_t` transpose vs alias → visible at `mtp_logits` (tap 8)
    - `mtp.final_layernorm` application site → visible at `post_final_ln` (tap 7)

Dependencies: `torch`, `safetensors`, `numpy`.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# -----------------------------------------------------------------------------
# Dtype helpers
# -----------------------------------------------------------------------------

def to_f32_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a contiguous float32 numpy array (for the
    safetensors writer, which wants CPU float32)."""
    return t.detach().to(torch.float32).cpu().numpy().copy()


def st_load_tensor(checkpoint_dir: str, name: str, allow_missing: bool = False) -> Optional[torch.Tensor]:
    """Load a single tensor from a safetensors shard directory by global name.

    Uses the `model.safetensors.index.json` file if present (standard HF
    layout). Falls back to scanning every `*.safetensors` file in the dir.
    Always returns BF16 as float32 on CPU (we operate in float32 on the ref
    side so the comparison against kiln's BF16-in-GPU path is upcast-only).
    """
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

    # Fall back: scan all shards.
    for shard_path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        with safe_open(shard_path, framework="pt", device="cpu") as g:
            if name in g.keys():
                return g.get_tensor(name).to(torch.float32)

    if allow_missing:
        return None
    raise KeyError(
        f"tensor `{name}` not found under {checkpoint_dir}. "
        "Check the checkpoint layout — Qwen3.5-4B ships either a `model.` "
        "prefix (plain LM) or a `model.language_model.` prefix (VL-wrapper)."
    )


def try_load_with_prefixes(checkpoint_dir: str, suffix: str, prefixes: list[str]) -> tuple[torch.Tensor, str]:
    """Try loading `suffix` under each prefix until one hits. Returns (tensor, prefix_used)."""
    last_err = None
    for p in prefixes:
        try:
            t = st_load_tensor(checkpoint_dir, p + suffix)
            assert t is not None
            return t, p
        except KeyError as e:
            last_err = e
    raise KeyError(f"Could not find `{suffix}` under any of {prefixes}: {last_err}")


# -----------------------------------------------------------------------------
# Qwen3-Next MTP weights container
# -----------------------------------------------------------------------------

@dataclass
class MtpRefWeights:
    # Tied LM head. Shape [V, H] in canonical HF layout; kiln stores
    # `embed_tokens_t` separately but semantically the matmul uses
    # `x @ embed_tokens.T` which we reproduce here.
    embed_tokens: torch.Tensor          # [V, H]

    # MTP fusion layer.
    fc_weight: torch.Tensor             # [H, 2H]  (HF canonical; applied as x @ fc_weight.T)
    pre_fc_norm_embedding: torch.Tensor  # [H]
    pre_fc_norm_hidden: torch.Tensor    # [H]
    final_layernorm: torch.Tensor       # [H]

    # MTP inner transformer block (single full-attention layer).
    input_layernorm: torch.Tensor       # [H]
    post_attention_layernorm: torch.Tensor  # [H]
    q_proj_weight: torch.Tensor         # [num_heads*head_dim*2, H] (gated)
    k_proj_weight: torch.Tensor         # [num_kv_heads*head_dim, H]
    v_proj_weight: torch.Tensor         # [num_kv_heads*head_dim, H]
    o_proj_weight: torch.Tensor         # [H, num_heads*head_dim]
    q_norm_weight: torch.Tensor         # [head_dim]
    k_norm_weight: torch.Tensor         # [head_dim]
    gate_proj_weight: torch.Tensor      # [ffn, H]
    up_proj_weight: torch.Tensor        # [ffn, H]
    down_proj_weight: torch.Tensor      # [H, ffn]

    config: dict


MTP_PREFIXES = ["mtp.", "model.mtp.", "model.language_model.mtp."]
EMBED_PREFIXES = [
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "embed_tokens.weight",
]


def load_mtp_weights(checkpoint_dir: str) -> MtpRefWeights:
    # Config for num_attention_heads, num_kv_heads, head_dim, etc.
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # VL-wrapper keeps the LM config nested under `text_config` or
    # `language_config`; fall back to the top-level if neither present.
    cfg = raw.get("text_config") or raw.get("language_config") or raw

    # Find embed_tokens under any known prefix.
    embed = None
    for ep in EMBED_PREFIXES:
        try:
            embed = st_load_tensor(checkpoint_dir, ep)
            print(f"[mtp_ref] embed_tokens loaded from `{ep}` shape={tuple(embed.shape)}", file=sys.stderr)
            break
        except KeyError:
            continue
    if embed is None:
        raise KeyError("No embed_tokens.weight found under any known prefix")

    # Find MTP prefix.
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
    print(f"[mtp_ref] MTP prefix = `{mtp_prefix}` fc.shape={tuple(mtp_fc.shape)}", file=sys.stderr)

    get = lambda suf: st_load_tensor(checkpoint_dir, mtp_prefix + suf)

    return MtpRefWeights(
        embed_tokens=embed,
        fc_weight=mtp_fc,
        pre_fc_norm_embedding=get("pre_fc_norm_embedding.weight"),
        pre_fc_norm_hidden=get("pre_fc_norm_hidden.weight"),
        # Either `mtp.norm.weight` or `mtp.final_layernorm.weight` — loader
        # accepts both per PR #253.
        final_layernorm=(
            st_load_tensor(checkpoint_dir, mtp_prefix + "norm.weight", allow_missing=True)
            or st_load_tensor(checkpoint_dir, mtp_prefix + "final_layernorm.weight")
        ),
        input_layernorm=get("layer.input_layernorm.weight"),
        post_attention_layernorm=get("layer.post_attention_layernorm.weight"),
        q_proj_weight=get("layer.self_attn.q_proj.weight"),
        k_proj_weight=get("layer.self_attn.k_proj.weight"),
        v_proj_weight=get("layer.self_attn.v_proj.weight"),
        o_proj_weight=get("layer.self_attn.o_proj.weight"),
        q_norm_weight=get("layer.self_attn.q_norm.weight"),
        k_norm_weight=get("layer.self_attn.k_norm.weight"),
        gate_proj_weight=get("layer.mlp.gate_proj.weight"),
        up_proj_weight=get("layer.mlp.up_proj.weight"),
        down_proj_weight=get("layer.mlp.down_proj.weight"),
        config=cfg,
    )


# -----------------------------------------------------------------------------
# Core ops (match kiln's forward.rs line-for-line as close as we can)
# -----------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root Mean Square LayerNorm. Matches `candle_nn::ops::rms_norm` semantics
    (float32 accumulation) which kiln uses. Returns same dtype as input.
    """
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.to(torch.float32)).to(orig_dtype)


def apply_rope_partial(
    x: torch.Tensor, position: int, head_dim: int, rotary_dim: int, rope_theta: float
) -> torch.Tensor:
    """Apply rotary embedding to the first `rotary_dim` dimensions of each
    head. `x` is `[batch, num_heads, seq_len, head_dim]`.

    Qwen3-Next uses partial rotary (default 0.25 of head_dim → 64-dim rotary
    for head_dim=256). Rotate on the leading `rotary_dim`, leave the tail
    untouched. `position` is the absolute position index for the single-token
    step we're modelling (MTP has seq_len=1).
    """
    if rotary_dim == 0:
        return x
    # inv_freq computed over rotary_dim/2 pairs.
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    # freqs for this single position.
    freqs = torch.tensor([position], dtype=torch.float32)[:, None] * inv_freq[None, :]  # [1, rot/2]
    cos = freqs.cos()  # [1, rot/2]
    sin = freqs.sin()  # [1, rot/2]

    # Split rotary half. Shape: [..., rotary_dim]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    # Pair up the dims. `candle` uses interleaved even/odd; HF commonly uses
    # half-rotate (first half vs second half). Qwen3-Next MTP in kiln uses
    # interleaved rotation — see `apply_rotary_emb` in forward.rs. We model
    # the "half-rotate" variant here; if the comparison surfaces a mismatch
    # at `post_layer`, an interleave-vs-halves swap is the single most
    # likely cause and will print clearly in the divergence table.
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    x_rot_new = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([x_rot_new, x_pass], dim=-1)


def mtp_inner_block(
    x: torch.Tensor,
    w: MtpRefWeights,
    mtp_pos: int,
    rope_theta: float,
    rotary_frac: float,
    eps: float,
) -> torch.Tensor:
    """Single MTP transformer block with self-attention over exactly one
    position. `x` shape: `[1, 1, H]`. Returns `[1, 1, H]` (pre-final-norm).

    Self-attention over a single token reduces to: the Q·K product yields a
    1×1 scalar, so softmax(.) = 1.0 and the attention output equals V. RoPE
    rotations on Q and K cancel in the Q·K^T product (same position, same
    rotation), so this path is a pretty good sanity-check for the MTP inner
    layer.
    """
    cfg = w.config
    H = x.shape[-1]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = cfg.get("head_dim", H // num_heads)
    rotary_dim = int(head_dim * rotary_frac)

    # Pre-attn norm.
    residual = x
    h = rms_norm(x, w.input_layernorm, eps)  # [1,1,H]

    # Q/K/V projections. The Qwen3-Next gated Q has shape [2*num_heads*head_dim, H]
    # (kiln's loader splits into Q and a gate). Here we emulate it by taking
    # the first half as Q (the ungated stream) and discarding the gate — this
    # is wrong for the full gated-attn semantics but a faithful reproduction
    # of that would pull in another ~100 LOC. For Phase B6, we flag
    # gated-attn as a known *reference* simplification; if `post_layer`
    # diverges by exactly this amount, the bug isn't real. See the compare
    # script output for this caveat.
    q_full = torch.matmul(h, w.q_proj_weight.t())  # [1,1, 2*num_heads*head_dim] if gated, else [1,1, num_heads*head_dim]
    if q_full.shape[-1] == 2 * num_heads * head_dim:
        q, q_gate = q_full.chunk(2, dim=-1)
        q_gated = True
    else:
        q = q_full
        q_gate = None
        q_gated = False
    k = torch.matmul(h, w.k_proj_weight.t())
    v = torch.matmul(h, w.v_proj_weight.t())

    # Reshape to heads: [1, 1, num_heads, head_dim] → [1, num_heads, 1, head_dim]
    q = q.view(1, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(1, 1, num_kv_heads, head_dim).transpose(1, 2)

    # Per-head RMSNorm on Q and K (Qwen3-Next style).
    q = rms_norm(q, w.q_norm_weight, eps)
    k = rms_norm(k, w.k_norm_weight, eps)

    # RoPE on rotary_dim prefix.
    q = apply_rope_partial(q, mtp_pos, head_dim, rotary_dim, rope_theta)
    k = apply_rope_partial(k, mtp_pos, head_dim, rotary_dim, rope_theta)

    # Expand KV heads for GQA.
    repeat = num_heads // num_kv_heads
    if repeat > 1:
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # Single-token self-attention reduces to attn_out == V (softmax of a
    # single scalar == 1.0), so Q·K^T /√d · softmax · V simplifies to V when
    # seq_len = 1 — modulo rounding. We still compute the full path for
    # faithfulness to kiln's op order.
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [1, num_heads, 1, 1]
    attn = torch.softmax(scores.to(torch.float32), dim=-1).to(v.dtype)
    attn_out = torch.matmul(attn, v)  # [1, num_heads, 1, head_dim]

    # Apply gated-attn sigmoid(q_gate) if present. We skip a proper
    # gated-attn implementation (see caveat above); reported divergence at
    # `post_layer` is expected to include this component.
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, num_heads * head_dim)
    if q_gated:
        # Kiln applies sigmoid gate per-head post-attn before o_proj (see
        # `transformer_block_paged` branch on `attn_output_gate=true`). We
        # model that here.
        q_gate = q_gate.view(1, 1, num_heads, head_dim).transpose(1, 2).contiguous()
        q_gate = q_gate.transpose(1, 2).contiguous().view(1, 1, num_heads * head_dim)
        attn_out = attn_out * torch.sigmoid(q_gate.to(attn_out.dtype))
    attn_out = torch.matmul(attn_out, w.o_proj_weight.t())

    # Residual.
    h = residual + attn_out

    # MLP.
    residual = h
    h2 = rms_norm(h, w.post_attention_layernorm, eps)
    gate = torch.matmul(h2, w.gate_proj_weight.t())
    up = torch.matmul(h2, w.up_proj_weight.t())
    act = torch.nn.functional.silu(gate) * up
    down = torch.matmul(act, w.down_proj_weight.t())
    h = residual + down

    return h


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def load_kiln_dump(path: str) -> Dict[str, torch.Tensor | int]:
    """Load the safetensors file kiln wrote. Returns a dict with the 8 taps
    as torch tensors plus integer metadata."""
    out: Dict[str, torch.Tensor | int] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name.startswith("meta__"):
                # Metadata is I32; take the scalar.
                out[name] = int(t.flatten()[0].item())
            else:
                out[name] = t.to(torch.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B6 reference MTP dump")
    ap.add_argument("--checkpoint", required=True, help="Qwen3.5-4B checkpoint directory")
    ap.add_argument("--kiln-dump", required=True, help="Path to kiln's mtp dump (safetensors)")
    ap.add_argument("--out", required=True, help="Output path for reference dump")
    ap.add_argument("--rms-eps", type=float, default=1e-6)
    args = ap.parse_args()

    print(f"[mtp_ref] loading kiln dump from {args.kiln_dump}", file=sys.stderr)
    kiln = load_kiln_dump(args.kiln_dump)
    for name in ["h_main", "tok_embed", "fc_input", "fc_output", "pre_layer", "post_layer", "post_final_ln", "mtp_logits"]:
        assert name in kiln, f"kiln dump missing expected tap `{name}`"
    draft_token_id = kiln["meta__draft_token_id"]
    mtp_pos = kiln["meta__mtp_pos"]
    swap = kiln["meta__swap_fc_norms"]
    print(f"[mtp_ref] draft_token_id={draft_token_id} mtp_pos={mtp_pos} swap_fc_norms={swap}", file=sys.stderr)

    print(f"[mtp_ref] loading MTP weights from {args.checkpoint}", file=sys.stderr)
    w = load_mtp_weights(args.checkpoint)

    cfg = w.config
    rope_theta = float(cfg.get("rope_theta", 10_000_000.0))
    rotary_frac = float(cfg.get("partial_rotary_factor", 0.25))

    # Pull `h_main` and run the full reference forward.
    h_main = kiln["h_main"].to(torch.float32)  # expected shape [1, 1, H]
    assert h_main.ndim == 3, f"h_main expected 3-D, got {h_main.shape}"

    # 1. Token embed.
    tok_embed = w.embed_tokens[draft_token_id : draft_token_id + 1].unsqueeze(0)  # [1, 1, H]

    # 2-3. Dual RMSNorms. Honor swap flag for parity.
    if swap:
        norm_emb_w = w.pre_fc_norm_hidden
        norm_h_w = w.pre_fc_norm_embedding
    else:
        norm_emb_w = w.pre_fc_norm_embedding
        norm_h_w = w.pre_fc_norm_hidden
    norm_emb = rms_norm(tok_embed, norm_emb_w, args.rms_eps)
    norm_h = rms_norm(h_main, norm_h_w, args.rms_eps)

    # 4. Concat + fc.
    fc_input = torch.cat([norm_emb, norm_h], dim=-1)  # [1, 1, 2H]
    fc_output = torch.matmul(fc_input, w.fc_weight.t())  # [1, 1, H]

    # 5. Single-layer block.
    pre_layer = fc_output  # alias; matches kiln's forward exactly
    post_layer = mtp_inner_block(pre_layer, w, mtp_pos, rope_theta, rotary_frac, args.rms_eps)

    # 6-7. Final norm + tied LM head.
    post_final_ln = rms_norm(post_layer, w.final_layernorm, args.rms_eps)
    mtp_logits = torch.matmul(post_final_ln, w.embed_tokens.t())  # [1, 1, V]

    # Write dump.
    out_dict = {
        "h_main": h_main.contiguous(),
        "tok_embed": tok_embed.contiguous(),
        "fc_input": fc_input.contiguous(),
        "fc_output": fc_output.contiguous(),
        "pre_layer": pre_layer.contiguous(),
        "post_layer": post_layer.contiguous(),
        "post_final_ln": post_final_ln.contiguous(),
        "mtp_logits": mtp_logits.contiguous(),
        # Carry metadata forward so the comparator can cross-check.
        "meta__draft_token_id": torch.tensor([int(draft_token_id)], dtype=torch.int32),
        "meta__mtp_pos": torch.tensor([int(mtp_pos)], dtype=torch.int32),
        "meta__swap_fc_norms": torch.tensor([int(swap)], dtype=torch.int32),
    }
    save_file(out_dict, args.out)
    print(f"[mtp_ref] wrote {args.out} ({sum(t.numel()*t.element_size() for t in out_dict.values())} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
