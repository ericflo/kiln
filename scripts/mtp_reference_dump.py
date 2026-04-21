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

Phase B10 — independent `h_main` reference
------------------------------------------

    python3 scripts/mtp_reference_dump.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --independent-h-main \\
        --seed 42 --prompt-tokens 512 --bench-mtp-pos 0 \\
        --out /tmp/mtp-ref-independent-pos0.safetensors

When `--independent-h-main` is set, the script loads the full Qwen3.5-4B base
model via `transformers.AutoModelForCausalLM` and runs forward on the same
prompt + seed the kiln-bench harness would produce. It captures the
pre-final-norm hidden at the position kiln uses as `h_main` (see below) and
saves it as `h_main_independent`. The comparator (`mtp_compare.py`) then
compares kiln's `h_main` tap against `h_main_independent` to decide whether
the 32-layer main stack is the upstream cause of the α collapse.

Scope: only `--bench-mtp-pos=0` is supported. For `mtp_pos=0` the value of
`h_main` is the last-row pre-final-norm hidden from the prefill pass — fully
reproducible from `(seed, prompt-tokens)` because the kiln-bench prompt is
deterministic. For `mtp_pos>=1`, `h_main` depends on the accepted-token
sequence which the kiln dump does not include; reproducing that would need a
Rust-side extension to the dump format (future work).

Gotcha: `h_main` is PRE-final-norm. HF's `outputs.hidden_states[-1]` and
`outputs.last_hidden_state` are BOTH post-norm, so we install a forward hook
on `model.model.layers[-1]` and grab its residual output before the final
`model.model.norm` is applied. See `forward.rs:3991` comment: `h_prev is
[1, 1, H] PRE-final-norm`.

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

    # Avoid `or` on torch.Tensor (it calls __bool__ which errors for >1 element).
    # Use a helper that tries candidates in order and returns the first hit.
    def first_match(*names: str) -> torch.Tensor:
        last_err: Optional[KeyError] = None
        for n in names:
            t = st_load_tensor(checkpoint_dir, n, allow_missing=True)
            if t is not None:
                return t
        raise KeyError(f"None of {names} present in checkpoint")

    return MtpRefWeights(
        embed_tokens=embed,
        fc_weight=mtp_fc,
        pre_fc_norm_embedding=get("pre_fc_norm_embedding.weight"),
        pre_fc_norm_hidden=get("pre_fc_norm_hidden.weight"),
        # Either `mtp.norm.weight` or `mtp.final_layernorm.weight` — loader
        # accepts both per PR #253.
        final_layernorm=first_match(
            mtp_prefix + "norm.weight",
            mtp_prefix + "final_layernorm.weight",
        ),
        # Actual Qwen3.5-4B checkpoint uses `mtp.layers.0.<...>` naming. Some
        # alternate checkpoints (and some of the older audit docs) referred to
        # `mtp.layer.<...>` — we try both to be safe.
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


# -----------------------------------------------------------------------------
# Core ops (match kiln's forward.rs line-for-line as close as we can)
# -----------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5-style RMSNorm: `out = (1 + w) * x * rsqrt(mean(x^2) + eps)`.

    This matches kiln's `rms_norm_fallback` in forward.rs (line 936), which
    the production kernel is validated against. The crucial detail is that
    Qwen3.5 stores RMSNorm weights **centered around 0** and applies them
    as `(1 + weight)`, not as bare `weight`. See forward.rs:955–957 and HF
    Qwen3NextRMSNorm.forward.

    Float32 accumulation throughout; output returned in the input dtype.
    """
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    w = weight.to(torch.float32)
    return (x * (1.0 + w)).to(orig_dtype)


def apply_rope_partial(
    x: torch.Tensor, position: int, head_dim: int, rotary_dim: int, rope_theta: float
) -> torch.Tensor:
    """Apply rotary embedding to the first `rotary_dim` dimensions of each
    head. `x` is `[batch, num_heads, seq_len, head_dim]` (legacy callers) or
    `[batch, seq_len, num_heads, head_dim]` (Phase B7b kiln-aligned form —
    same per-element semantics regardless of the head/seq order).

    Qwen3-Next uses partial rotary (default 0.25 of head_dim → 64-dim rotary
    for head_dim=256). Rotate on the leading `rotary_dim`, leave the tail
    untouched. `position` is the absolute position index for the single-token
    step we're modelling (MTP has seq_len=1).

    Verified against kiln in forward.rs:1108-1148 — both use **half-rotate**
    (split into first half + second half, recombine `[r1, r2, pass]`). Earlier
    versions of this docstring claimed kiln used interleaved rotation; that
    was wrong and has been corrected.
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
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    x_rot_new = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([x_rot_new, x_pass], dim=-1)


def _capture(buf: Optional[Dict[str, torch.Tensor]], name: str, t: torch.Tensor) -> None:
    """Best-effort sub-op capture. No-op when `buf` is None (production path
    when --capture-subops is not set). Always stores a contiguous F32 clone so
    the writer can serialize without later in-place ops corrupting the buffer.
    """
    if buf is None:
        return
    buf[name] = t.detach().to(torch.float32).contiguous().clone()


def mtp_inner_block(
    x: torch.Tensor,
    w: MtpRefWeights,
    mtp_pos: int,
    rope_theta: float,
    rotary_frac: float,
    eps: float,
    capture_subops: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Single MTP transformer block with self-attention over exactly one
    position. `x` shape: `[1, 1, H]`. Returns `[1, 1, H]` (pre-final-norm).

    Self-attention over a single token reduces to: the Q·K product yields a
    1×1 scalar, so softmax(.) = 1.0 and the attention output equals V. RoPE
    rotations on Q and K cancel in the Q·K^T product (same position, same
    rotation), so this path is a pretty good sanity-check for the MTP inner
    layer.

    Phase B7b: when `capture_subops` is a dict, every named tap below is
    written into it under the same name kiln captures (see
    `gqa_attention_paged` and `transformer_block_paged` in forward.rs). The
    tensor layouts at each tap match kiln's exactly so per-element comparison
    is meaningful — see in particular the per-head Q/gate narrow that
    replaces the older flat-half chunk.
    """
    cfg = w.config
    H = x.shape[-1]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = cfg.get("head_dim", H // num_heads)
    rotary_dim = int(head_dim * rotary_frac)

    # Pre-attn norm. Tap matches transformer_block_paged in forward.rs:3122-3126.
    residual = x
    h = rms_norm(x, w.input_layernorm, eps)  # [1,1,H]
    _capture(capture_subops, "post_pre_attn_norm", h)

    # Q/K/V projections. q_proj is gated for Qwen3-Next when q_proj.weight has
    # shape [num_heads*head_dim*2, H]; the trailing 2H output is split into
    # the rotation target (Q) and a sigmoid gate applied post-attention. Tap
    # name matches gqa_attention_paged in forward.rs:2620-2622.
    q_raw = torch.matmul(h, w.q_proj_weight.t())  # [1, 1, num_heads*head_dim*2 if gated else num_heads*head_dim]
    k = torch.matmul(h, w.k_proj_weight.t())       # [1, 1, num_kv_heads*head_dim]
    v = torch.matmul(h, w.v_proj_weight.t())       # [1, 1, num_kv_heads*head_dim]
    _capture(capture_subops, "post_q_proj_raw", q_raw)
    _capture(capture_subops, "post_k_proj", k)
    _capture(capture_subops, "post_v_proj", v)
    # Phase B9 H3 alias: pre_gated_attn_split = q_raw before per-head narrow.
    _capture(capture_subops, "pre_gated_attn_split", q_raw)

    # Per-head Q/gate split — must mirror kiln's narrow exactly. Kiln does:
    #   q_pair = q_raw.reshape((batch, seq_len, num_heads, head_dim*2))
    #   q      = q_pair.narrow(3, 0,        head_dim)         # [B, S, H, D]
    #   gate   = q_pair.narrow(3, head_dim, head_dim)         # [B, S, H, D]
    #   gate   = gate.reshape((batch, seq_len, num_heads*head_dim))
    # An earlier reference used `q_full.chunk(2, dim=-1)` which splits the
    # *flat* trailing dim in half — that produces a DIFFERENT q tensor for
    # any weight layout that pairs (q_dim, gate_dim) per head. The flat
    # chunk was a known-wrong simplification noted in PR #271; this is the
    # corrected per-head split.
    q_gated = (q_raw.shape[-1] == 2 * num_heads * head_dim)
    if q_gated:
        q_pair = q_raw.view(1, 1, num_heads, head_dim * 2)
        q = q_pair[..., :head_dim].contiguous()                             # [1, 1, num_heads, head_dim]
        gate = q_pair[..., head_dim:].contiguous().view(1, 1, num_heads * head_dim)  # [1, 1, num_heads*head_dim]
    else:
        q = q_raw.view(1, 1, num_heads, head_dim)                           # [1, 1, num_heads, head_dim]
        gate = None
    _capture(capture_subops, "post_q_split", q)
    # Phase B9 H3 alias: post_gated_attn_split_value mirrors post_q_split.
    _capture(capture_subops, "post_gated_attn_split_value", q)
    if gate is not None:
        _capture(capture_subops, "post_gate_split", gate)
        # Phase B9 H3 alias: post_gated_attn_split_gate mirrors post_gate_split.
        _capture(capture_subops, "post_gated_attn_split_gate", gate)

    k = k.view(1, 1, num_kv_heads, head_dim)  # [1, 1, num_kv_heads, head_dim]
    v = v.view(1, 1, num_kv_heads, head_dim)  # [1, 1, num_kv_heads, head_dim]

    # Phase B9 H2 taps: pre_qk_norm_{q,k} are per-head reshaped tensors
    # immediately before per-head RMSNorm. pre_qk_norm_q is alias of
    # post_q_split; pre_qk_norm_k is genuinely new (post_k_proj is pre-reshape).
    _capture(capture_subops, "pre_qk_norm_q", q)
    _capture(capture_subops, "pre_qk_norm_k", k)

    # Per-head RMSNorm on Q and K (Qwen3-Next style). Same shape in/out.
    q = rms_norm(q, w.q_norm_weight, eps)
    k = rms_norm(k, w.k_norm_weight, eps)
    _capture(capture_subops, "post_q_norm", q)
    _capture(capture_subops, "post_k_norm", k)
    # Phase B9 H2 aliases: post_qk_norm_{q,k} mirror post_{q,k}_norm.
    _capture(capture_subops, "post_qk_norm_q", q)
    _capture(capture_subops, "post_qk_norm_k", k)

    # RoPE on rotary_dim prefix in the [B, S, H, D] layout — matches kiln's
    # `rotary_embedding_from_tensor` call site in forward.rs:2661-2666 (both
    # use half-rotate; see `apply_rope_partial` docstring above).
    q = apply_rope_partial(q, mtp_pos, head_dim, rotary_dim, rope_theta)
    k = apply_rope_partial(k, mtp_pos, head_dim, rotary_dim, rope_theta)
    _capture(capture_subops, "post_q_rope", q)
    _capture(capture_subops, "post_k_rope", k)

    # Transpose to [batch, num_heads, seq_len, head_dim] for the attention
    # matmuls. This mirrors forward.rs:2669-2675 (the qkv_transpose block).
    q_t = q.transpose(1, 2).contiguous()  # [1, num_heads,    1, head_dim]
    k_t = k.transpose(1, 2).contiguous()  # [1, num_kv_heads, 1, head_dim]
    v_t = v.transpose(1, 2).contiguous()  # [1, num_kv_heads, 1, head_dim]

    # Expand KV for GQA. seq_len=1 so this is a no-op cost-wise; included
    # only so the math matches the gqa-grouped path in forward.rs:2837-2879
    # without having to reproduce the per-group reshape.
    repeat = num_heads // num_kv_heads
    if repeat > 1:
        k_full = k_t.repeat_interleave(repeat, dim=1)
        v_full = v_t.repeat_interleave(repeat, dim=1)
    else:
        k_full = k_t
        v_full = v_t

    # Single-token self-attention.
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_t, k_full.transpose(-2, -1)) * scale  # [1, num_heads, 1, 1]
    attn = torch.softmax(scores.to(torch.float32), dim=-1).to(v_full.dtype)
    attn_out = torch.matmul(attn, v_full)  # [1, num_heads, 1, head_dim]

    # Reshape into kiln's `post_attn_raw` form: [1, 1, num_heads * head_dim].
    # Mirrors forward.rs:2875-2879 (the attn_output reshape after softmax/V).
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, num_heads * head_dim)
    _capture(capture_subops, "post_attn_raw", attn_out)

    # Gated-attn: per-element sigmoid(gate) * attn_out, both flat last dim.
    if gate is not None:
        attn_out = attn_out * torch.sigmoid(gate.to(attn_out.dtype))
    _capture(capture_subops, "post_attn_gated", attn_out)

    # Output projection. Tap matches forward.rs:2898 / 2955.
    o = torch.matmul(attn_out, w.o_proj_weight.t())  # [1, 1, H]
    _capture(capture_subops, "post_o_proj", o)

    # Capture the inner-attn-block output BEFORE the residual add.
    # Matches forward.rs:3147 (`post_attn_block`).
    _capture(capture_subops, "post_attn_block", o)

    # Residual. Tap matches forward.rs:3154.
    h = residual + o
    _capture(capture_subops, "post_attn_residual", h)

    # MLP path.
    residual = h
    h2 = rms_norm(h, w.post_attention_layernorm, eps)
    _capture(capture_subops, "post_pre_mlp_norm", h2)

    gate_p = torch.matmul(h2, w.gate_proj_weight.t())
    up_p = torch.matmul(h2, w.up_proj_weight.t())
    act = torch.nn.functional.silu(gate_p) * up_p
    down = torch.matmul(act, w.down_proj_weight.t())
    _capture(capture_subops, "post_mlp", down)

    # Final residual; outer caller dumps this as `post_layer`.
    h = residual + down
    return h


# -----------------------------------------------------------------------------
# Phase B10 — independent `h_main` reference via HF transformers
# -----------------------------------------------------------------------------

# Verbatim copy of PROMPT_POOL from crates/kiln-server/src/bench.rs:229-270.
# Must stay bit-for-bit identical to the Rust source or the tokenized prompt
# will differ and `h_main_independent` will not be comparable to kiln's
# `h_main`. The whitespace (including the line-continuation indentation) is
# preserved exactly as Rust's raw `\` line-continuation would render it.
PROMPT_POOL: list[str] = [
    # 0: canonical B2 baseline — do not edit
    "The quick brown fox jumps over the lazy dog near the river bank. \
     Scientists discovered a new species of deep-sea fish in the Pacific Ocean. \
     The quantum computer solved the optimization problem in record time. \
     She wrote a comprehensive analysis of market trends for the quarterly report. ",
    # 1: software / algorithms
    "A red-black tree balances itself on every insertion and deletion to keep operations logarithmic. \
     The compiler lowered the expression into three-address code before register allocation. \
     Pagination in a distributed system needs a stable sort key or the client will see duplicates. \
     An idempotent retry policy is the simplest defense against transient network failures. ",
    # 2: history / biography
    "The library at Alexandria housed an estimated half million scrolls before the great fire. \
     Marie Curie remains the only person to win Nobel prizes in two distinct scientific fields. \
     The printing press transformed European literacy faster than any single invention before it. \
     Ancient Sumerian accountants pressed wedge-shaped marks into damp clay to track grain shipments. ",
    # 3: nature / geography
    "The Mariana Trench descends nearly eleven kilometers below the surface of the western Pacific. \
     Alpine glaciers carve U-shaped valleys whose walls record the slow movement of compacted ice. \
     Monarch butterflies navigate thousands of miles to the same Mexican forests their ancestors left. \
     The Amazon basin exchanges more moisture with the atmosphere than any other river system on Earth. ",
    # 4: philosophy
    "Stoic ethics ground virtue in accepting what is outside the rational agent's direct control. \
     Hume argued that no description of the world by itself entails any obligation about what ought to be. \
     Phenomenology studies the structures of conscious experience from the first-person point of view. \
     A thought experiment about swapped qualia probes whether subjective color really supervenes on function. ",
    # 5: cooking / food science
    "A pinch of salt in caramel suppresses perceived sweetness and sharpens the roasted sugar flavor. \
     Bread dough becomes elastic because kneading aligns the gluten strands formed by wheat proteins. \
     Emulsifying lemon juice into olive oil yields a stable vinaigrette only when whisked vigorously. \
     Slow braising converts tough connective tissue into gelatin, tenderizing an otherwise unpromising cut. ",
    # 6: space / astronomy
    "A binary pulsar's timing drift confirmed general relativity's prediction of gravitational waves. \
     The Parker Solar Probe samples the corona from closer than any spacecraft has previously flown. \
     Tidally locked exoplanets have a permanent dayside whose climate differs from the eternal night side. \
     Supernova remnants seed the interstellar medium with the heavier elements later rocky worlds inherit. ",
    # 7: music / craft
    "Equal temperament tuning slightly detunes every interval except the octave to enable free modulation. \
     A violin luthier planes the spruce top until tap tones resonate in the correct pitch relationship. \
     Polyrhythms layer pulses that only align at measure boundaries, driving much West African drumming. \
     The overtone series produced by a bowed string determines the characteristic timbre of each instrument. ",
]


def build_bench_prompt(tokenizer, target_tokens: int, seed: int) -> str:
    """Reproduce `build_prompt()` from crates/kiln-server/src/bench.rs:276-295.

    - Pick `base = PROMPT_POOL[seed % 8]`.
    - Append `base` to a growing string until tokenized length >= target.
    - Truncate back via repeated `rfind(". ") + 1` until tokenized length
      <= target, or no `. ` separator remains.

    The kiln-bench harness is fully deterministic from `(seed, target_tokens)`,
    which is what makes mtp_pos=0 reproducible in HF without needing the kiln
    runtime.
    """
    base = PROMPT_POOL[seed % len(PROMPT_POOL)]
    prompt = ""
    # Tight bound: kiln's Rust version has no step cap but we add one for
    # defensive purposes (Python tokenizer quirks shouldn't spin forever).
    for _ in range(10_000):
        prompt += base
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) >= target_tokens:
            while True:
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids) <= target_tokens:
                    break
                idx = prompt.rfind(". ")
                if idx < 0:
                    break
                prompt = prompt[: idx + 1]
            return prompt
    raise RuntimeError("build_bench_prompt: did not reach target tokens after 10k iterations")


def run_independent_h_main(
    checkpoint: str,
    out_path: str,
    seed: int,
    prompt_tokens: int,
    bench_mtp_pos: int,
) -> int:
    """Phase B10 — load Qwen3.5-4B via HF transformers, reproduce the
    kiln-bench prompt, capture PRE-final-norm last-position hidden as
    `h_main_independent`, save to safetensors.

    Scope: only `bench_mtp_pos == 0` is supported. For pos >= 1 the kiln
    `h_main` input depends on the accepted-token stream which is not dumped
    by kiln today (future work: extend `write_mtp_dump`).
    """
    if bench_mtp_pos != 0:
        print(
            f"ERROR: --bench-mtp-pos={bench_mtp_pos} not supported. Phase B10 supports "
            "only bench_mtp_pos=0 (the prefill-derived h_main). See docstring.",
            file=sys.stderr,
        )
        return 2

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as e:
        print(
            f"ERROR: transformers not installed: {e}. "
            "Install with `uv pip install --system transformers` on the build pod.",
            file=sys.stderr,
        )
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(
        f"[mtp_ref/b10] loading HF model from {checkpoint} on {device} dtype={dtype}",
        file=sys.stderr,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    # Build the deterministic bench prompt.
    prompt = build_bench_prompt(tokenizer, prompt_tokens, seed)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    actual_tokens = len(ids)
    print(
        f"[mtp_ref/b10] seed={seed} target_prompt_tokens={prompt_tokens} "
        f"actual_tokens={actual_tokens}",
        file=sys.stderr,
    )

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Find the main backbone module that owns `.layers` and `.norm`.
    # HF Qwen3Next layouts: `model.model` (causal-LM wrapper) OR
    # `model.model.language_model` (VL-wrapper). Try in order.
    candidates = []
    root = getattr(model, "model", None)
    if root is not None:
        candidates.append(("model.model", root))
        lm_inner = getattr(root, "language_model", None)
        if lm_inner is not None:
            candidates.append(("model.model.language_model", lm_inner))

    backbone = None
    backbone_name = ""
    for name, cand in candidates:
        if hasattr(cand, "layers") and hasattr(cand, "norm"):
            backbone = cand
            backbone_name = name
            break
    if backbone is None:
        print(
            "ERROR: could not locate the base-model backbone (a module with "
            ".layers[-1] and .norm). Tried: "
            + ", ".join(n for n, _ in candidates),
            file=sys.stderr,
        )
        return 2
    print(
        f"[mtp_ref/b10] backbone={backbone_name} num_layers={len(backbone.layers)}",
        file=sys.stderr,
    )

    # Install a forward hook on the LAST layer of the backbone to capture the
    # pre-final-norm residual. HF Llama-style layer forwards return a tuple
    # whose [0] is the residual hidden state of shape [B, S, H]. That is
    # exactly the input to `backbone.norm`, i.e. `h_main` in kiln parlance.
    captured: Dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        # `output` is either a Tensor or a tuple with Tensor at index 0.
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        captured["pre_norm"] = hidden.detach()

    handle = backbone.layers[-1].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            # `output_hidden_states=False` — we don't need HF's hidden_states
            # list (which is post-norm on the final entry anyway).
            _ = model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()

    if "pre_norm" not in captured:
        print("ERROR: forward hook on layers[-1] did not fire.", file=sys.stderr)
        return 2

    pre_norm = captured["pre_norm"]  # [1, S, H]
    if pre_norm.ndim != 3 or pre_norm.shape[0] != 1 or pre_norm.shape[1] != actual_tokens:
        print(
            f"WARN: pre_norm hidden shape is {tuple(pre_norm.shape)}, expected "
            f"[1, {actual_tokens}, H]. Using last-position slice anyway.",
            file=sys.stderr,
        )

    # kiln's h_main for mtp_pos=0 is the last-row pre-final-norm hidden from
    # the prefill pass. See bench.rs ~line 1030 / speculative.rs.
    h_main_independent = pre_norm[:, -1:, :].contiguous()
    print(
        f"[mtp_ref/b10] h_main_independent shape={tuple(h_main_independent.shape)} "
        f"dtype={h_main_independent.dtype}",
        file=sys.stderr,
    )

    out_dict: Dict[str, torch.Tensor] = {
        "h_main_independent": to_f32_tensor(h_main_independent),
        "meta__seed": torch.tensor([int(seed)], dtype=torch.int32),
        "meta__prompt_tokens": torch.tensor([int(actual_tokens)], dtype=torch.int32),
        "meta__target_prompt_tokens": torch.tensor([int(prompt_tokens)], dtype=torch.int32),
        "meta__bench_mtp_pos": torch.tensor([int(bench_mtp_pos)], dtype=torch.int32),
        "meta__b10_mode": torch.tensor([1], dtype=torch.int32),
    }
    save_file(out_dict, out_path)
    print(
        f"[mtp_ref/b10] wrote {out_path} "
        f"({sum(t.numel()*t.element_size() for t in out_dict.values())} bytes)",
        file=sys.stderr,
    )
    return 0


def to_f32_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return a CPU float32 contiguous copy of `t` (safetensors-ready)."""
    return t.detach().to(torch.float32).cpu().contiguous()


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
    ap = argparse.ArgumentParser(description="Phase B6/B7/B10 reference MTP dump")
    ap.add_argument("--checkpoint", required=True, help="Qwen3.5-4B checkpoint directory")
    ap.add_argument(
        "--kiln-dump",
        required=False,
        default=None,
        help="Path to kiln's mtp dump (safetensors). Required except in --independent-h-main mode.",
    )
    ap.add_argument("--out", required=True, help="Output path for reference dump")
    ap.add_argument("--rms-eps", type=float, default=1e-6)
    ap.add_argument(
        "--capture-subops",
        action="store_true",
        help="Phase B7b: capture per-sub-op taps inside mtp_inner_block and write them into the output safetensors alongside the 8 standard taps.",
    )
    ap.add_argument(
        "--mtp-pos-override",
        type=int,
        default=None,
        help="Phase B7a sweep: override the mtp_pos used for RoPE without re-running kiln. Lets one kiln dump be replayed against the reference at different positions.",
    )
    ap.add_argument(
        "--independent-h-main",
        action="store_true",
        help="Phase B10: load full Qwen3.5-4B via HF transformers, reproduce the kiln-bench prompt from (seed, prompt-tokens), capture PRE-final-norm last-position hidden as `h_main_independent`. Supports --bench-mtp-pos=0 only.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Phase B10: kiln-bench seed to reproduce (default 42; must match kiln run).",
    )
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="Phase B10: target prompt-token count passed to kiln-bench --prompt-tokens (default 512).",
    )
    ap.add_argument(
        "--bench-mtp-pos",
        type=int,
        default=0,
        help="Phase B10: which mtp_pos to reproduce h_main for (ONLY 0 supported).",
    )
    args = ap.parse_args()

    # Phase B10 branch: independent h_main reference from HF.
    if args.independent_h_main:
        return run_independent_h_main(
            checkpoint=args.checkpoint,
            out_path=args.out,
            seed=int(args.seed),
            prompt_tokens=int(args.prompt_tokens),
            bench_mtp_pos=int(args.bench_mtp_pos),
        )

    # Existing B6/B7 path: requires --kiln-dump.
    if not args.kiln_dump:
        print(
            "ERROR: --kiln-dump is required unless --independent-h-main is set.",
            file=sys.stderr,
        )
        return 2

    print(f"[mtp_ref] loading kiln dump from {args.kiln_dump}", file=sys.stderr)
    kiln = load_kiln_dump(args.kiln_dump)
    for name in ["h_main", "tok_embed", "fc_input", "fc_output", "pre_layer", "post_layer", "post_final_ln", "mtp_logits"]:
        assert name in kiln, f"kiln dump missing expected tap `{name}`"
    draft_token_id = kiln["meta__draft_token_id"]
    kiln_mtp_pos = int(kiln["meta__mtp_pos"])
    swap = kiln["meta__swap_fc_norms"]
    if args.mtp_pos_override is not None:
        mtp_pos = int(args.mtp_pos_override)
        print(
            f"[mtp_ref] draft_token_id={draft_token_id} kiln_mtp_pos={kiln_mtp_pos} -> overridden mtp_pos={mtp_pos} swap_fc_norms={swap}",
            file=sys.stderr,
        )
    else:
        mtp_pos = kiln_mtp_pos
        print(
            f"[mtp_ref] draft_token_id={draft_token_id} mtp_pos={mtp_pos} swap_fc_norms={swap}",
            file=sys.stderr,
        )

    print(f"[mtp_ref] loading MTP weights from {args.checkpoint}", file=sys.stderr)
    w = load_mtp_weights(args.checkpoint)

    cfg = w.config
    # Qwen3.5-4B defaults (hardcoded in kiln kiln-core/src/config.rs:80-101 since
    # the shipped config.json text_config doesn't list them). Let env vars or
    # explicit CLI override if needed for sensitivity sweeps.
    rope_theta = float(os.environ.get("KILN_REF_ROPE_THETA", cfg.get("rope_theta", 10_000_000.0)) or 10_000_000.0)
    rotary_frac = float(os.environ.get("KILN_REF_ROTARY_FRAC", cfg.get("partial_rotary_factor", 0.25)) or 0.25)
    print(f"[mtp_ref] rope_theta={rope_theta} rotary_frac={rotary_frac}", file=sys.stderr)

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
    # Alias-clone: kiln's forward.rs binds the fused result to BOTH `fused`
    # (called `fc_output`) and `pre_layer`. For safetensors we need them as
    # distinct tensors.
    pre_layer = fc_output.clone()
    capture_subops: Optional[Dict[str, torch.Tensor]] = {} if args.capture_subops else None
    post_layer = mtp_inner_block(
        pre_layer,
        w,
        mtp_pos,
        rope_theta,
        rotary_frac,
        args.rms_eps,
        capture_subops=capture_subops,
    )
    if capture_subops is not None:
        print(
            f"[mtp_ref] captured {len(capture_subops)} sub-op taps: {sorted(capture_subops.keys())}",
            file=sys.stderr,
        )

    # 6-7. Final norm + tied LM head.
    post_final_ln = rms_norm(post_layer, w.final_layernorm, args.rms_eps)
    mtp_logits = torch.matmul(post_final_ln, w.embed_tokens.t())  # [1, 1, V]

    # Write dump.
    out_dict: Dict[str, torch.Tensor] = {
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
        "meta__kiln_mtp_pos": torch.tensor([int(kiln_mtp_pos)], dtype=torch.int32),
        "meta__swap_fc_norms": torch.tensor([int(swap)], dtype=torch.int32),
        "meta__capture_subops": torch.tensor([int(args.capture_subops)], dtype=torch.int32),
    }
    if capture_subops is not None:
        for name, t in capture_subops.items():
            assert name not in out_dict, f"sub-op tap name `{name}` collides with existing tap"
            out_dict[name] = t.contiguous()
    save_file(out_dict, args.out)
    print(f"[mtp_ref] wrote {args.out} ({sum(t.numel()*t.element_size() for t in out_dict.values())} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
