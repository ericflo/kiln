#!/usr/bin/env python3
"""
Phase B10 — `h_main` base-model reference dump.
Phase B11b — optional layer-0 GDN sub-op taps (`--b11-taps`).
Phase B12  — optional per-layer taps for layers 24..31 + GQA sub-op taps at
             layer 31 (`--b12-taps`), plus optional fp32 reference (`--fp32`).
Phase C41  — optional transformer-block-1 sub-op taps (`--c41-taps`).
Phase C42  — optional layer-1 pre-norm / input-layernorm taps (`--c42-taps`).

This script produces the independent pure-Python reference counterpart to the
kiln-side main-model forward (the 24×GDN + 8×GQA stack) so that Phase B10 can
bisect which layer first diverges. Prior phases (B6–B9) ruled out every sub-op
inside the MTP head, so the remaining candidate is the base-model `h_prev`
(`h_main`) that kiln feeds into MTP.

Phase B11b extends this with 11 intra-layer taps at layer 0's GDN block so we
can bisect the first diverging sub-op (B10 showed layer 0 as the first
boundary where cos ≈ 0.827 between kiln and HF).

Phase B12 adds per-layer `h_layer_{24..31}` taps (the 8 GQA layers that carry
the accumulated-but-benign-looking drift) and, at layer 31, a full suite of
sub-op taps inside `gqa_attention` + the MLP. B12 also supports running the
HF reference in fp32 (`--fp32`) so the comparator can separate numerical
bf16-accumulation noise from real kernel divergence.

GDN class handling
------------------

Qwen3.5-4B instantiates `Qwen3_5GatedDeltaNet` with a 4-way in_proj:

    mixed_qkv = in_proj_qkv(h)   # [B, T, key_dim*2 + value_dim]
    z         = in_proj_z(h)     # [B, T, value_dim]
    b         = in_proj_b(h)     # [B, T, num_v_heads]
    a         = in_proj_a(h)     # [B, T, num_v_heads]

Earlier Qwen3-Next uses `Qwen3NextGatedDeltaNet` with a packed 2-way in_proj:

    projected_qkvz = in_proj_qkvz(h)   # [..., [q, k, v, z]]
    projected_ba   = in_proj_ba(h)     # [..., [b, a]]

This script detects which layout the loaded checkpoint uses (by inspecting
layer 0's submodules for `in_proj_qkv` vs `in_proj_qkvz`) and monkey-patches
the correct class's `forward`. Older revisions of this script only patched
`Qwen3NextGatedDeltaNet`, silently no-op'ing on Qwen3.5-4B.

Usage
-----

    # B10-only boundary taps (original flow)
    python3 scripts/mtp_h_main_reference_dump.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --kiln-dump /tmp/h-kiln.safetensors \\
        --out /tmp/h-ref.safetensors

    # B10 boundary taps + B11b layer-0 GDN sub-op taps
    python3 scripts/mtp_h_main_reference_dump.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --kiln-dump /tmp/h-kiln.safetensors \\
        --out /tmp/h-ref.safetensors \\
        --b11-taps

Design
------

Unlike `mtp_reference_dump.py`, this script does NOT take any hidden state
from kiln. It loads the Qwen3.5-4B checkpoint via HuggingFace transformers
(which natively supports Qwen3-Next's hybrid 24×GDN + 8×GQA architecture via
`trust_remote_code=True`), tokenizes the exact same prompt the kiln dump saw,
and runs `model(..., output_hidden_states=True)` in BF16. We then extract
the last-row slice of the hidden state at five boundary layers (0, 8, 16, 23,
31) plus the pre-final-norm hidden state (= output of layer 31, before
`model.norm`). Those are the same six taps kiln captures when
`KILN_MTP_DUMP_HIDDEN_STATES=1`.

When `--b11-taps` is passed, we additionally monkey-patch
`Qwen3NextGatedDeltaNet.forward` with an instrumented copy that captures
11 intra-layer sub-op outputs at layer 0 (matching `B11_TAP_NAMES` in
`crates/kiln-model/src/mtp_debug.rs`). The full-tensor layer-0 taps are saved
alongside the boundary taps with a `b11__` prefix.

The safetensors file written here uses the same layout and naming convention
as kiln's dump so `scripts/mtp_compare.py --b10` / `--b11` can diff them
tap-for-tap.

Prompt provenance
-----------------

* If the kiln dump contains a `meta__replay_tokens_len` scalar and a
  `replay_tokens` I32 tensor, we use those directly.
* Otherwise if the kiln dump contains a `meta__prompt_tokens_len` scalar and a
  `prompt_tokens` I32 tensor, we use those directly.
* Otherwise we fall back to a canonical 512-token greeting prompt with
  torch seed=42, matching what the kiln bench harness emits by default.

Numerics
--------

Everything is run in BF16 on GPU if CUDA is available (else CPU), then
up-cast to F32 for the safetensors dump. This matches kiln's compute dtype.
Expected tolerances at the per-layer comparator level are atol=1e-2,
rtol=1e-1 — looser than B9's 1e-3 because BF16 accumulates noise across
24 GDN + 8 GQA layers before the last-layer tap.

Dependencies: `torch`, `transformers`, `safetensors`, `numpy`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Boundary layer indices captured by both kiln and the reference. Chosen to
# span the 32-layer stack so the first-diverging layer can be localized with
# a single additional mid-bisect on the next run if needed.
#
# Layer indices under Qwen3-Next's `full_attention_interval=4` layout:
#   - layer 0:  GDN (first linear-attention layer)
#   - layer 8:  GDN
#   - layer 16: GDN
#   - layer 23: GQA (indices 3, 7, 11, 15, 19, 23, 27, 31 are full-attn)
#   - layer 31: GQA (last layer; its output IS `h_pre_final_norm`)
B10_BOUNDARY_LAYERS: Tuple[int, ...] = (0, 8, 16, 23, 31)

# Phase C40 — optional dense early-stack sweep used to localize the first
# shared upstream drift inside C39's coarse `h_layer_0 -> h_layer_8` span.
# The kiln dump now serializes the actual boundary set in
# `meta__boundary_layers`; this tuple is the fallback when that metadata is
# absent.
C40_EARLY_BOUNDARY_LAYERS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)


# Phase B11b — ordered layer-0 GDN sub-op tap names. Must stay in lock-step
# with `B11_TAP_NAMES` in `crates/kiln-model/src/mtp_debug.rs`. The comparator
# (`scripts/mtp_compare.py --b11`) prints rows in this order.
B11_LAYER0_TAP_NAMES: Tuple[str, ...] = (
    "tok_embed",
    "layer_0_post_input_norm",
    "gdn_in_proj",
    "gdn_conv",
    "gdn_qk_norm_q",
    "gdn_qk_norm_k",
    "gdn_gate_beta",
    "gdn_gate_g",
    "gdn_recur_out",
    "gdn_gated_norm",
    "gdn_out_proj",
)


# Phase B12 — additional per-layer boundary taps for layers 24..30. Layer 31
# is already captured by B10. Together with the existing B10 taps (which
# include 23 and 31), this gives dense coverage of the 8 GQA layers where
# the residual drift (cos_sim 0.97-0.98) currently lives.
B12_GQA_LAYERS: Tuple[int, ...] = (24, 25, 26, 27, 28, 29, 30)


# Phase B12 — ordered GQA sub-op tap names for layer 31. Must stay in
# lock-step with `B12_GQA_TAP_NAMES` in `crates/kiln-model/src/mtp_debug.rs`.
# Captured at the last GQA layer because B10 shows that's where drift is
# already visible; instrumenting earlier GQA layers would cost more RAM
# without new information.
B12_LAYER31_GQA_TAP_NAMES: Tuple[str, ...] = (
    "post_input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_norm_q",
    "qk_norm_k",
    "rope_q",
    "rope_k",
    "attn_out",
    "o_proj",
    "post_attn_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
)


# Phase C41 — ordered transformer-block-1 tap names. Must stay in lock-step
# with `C41_LAYER1_TAP_NAMES` in `crates/kiln-model/src/mtp_debug.rs`.
C41_LAYER1_TAP_NAMES: Tuple[str, ...] = (
    "layer_1_post_input_norm",
    "gdn_in_proj",
    "gdn_conv",
    "gdn_qk_norm_q",
    "gdn_qk_norm_k",
    "gdn_gate_beta",
    "gdn_gate_g",
    "gdn_recur_out",
    "gdn_gated_norm",
    "gdn_out_proj",
    "layer_1_post_attn_residual",
    "layer_1_output",
)


# Phase C42 — ordered layer-1 pre-norm / input-layernorm tap names. Must stay
# in lock-step with `C42_LAYER1_NORM_TAP_NAMES` in
# `crates/kiln-model/src/mtp_debug.rs`.
C42_LAYER1_NORM_TAP_NAMES: Tuple[str, ...] = (
    "layer_1_residual_input",
    "layer_1_input_norm_rms_inv",
    "layer_1_input_norm_pre_weight",
    "layer_1_post_input_norm",
)


# -----------------------------------------------------------------------------
# Kiln-dump loader (mirrors mtp_reference_dump.py's loader)
# -----------------------------------------------------------------------------


def load_kiln_dump(path: str) -> Dict[str, object]:
    """Load a kiln-side MTP dump. Int tensors with a `meta__` prefix are
    returned as plain Python ints; other tensors as float32 (or int64 for
    `prompt_tokens`) on CPU."""
    out: Dict[str, object] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name in ("prompt_tokens", "replay_tokens"):
                # Token IDs — keep integral.
                out[name] = t.to(torch.int64).contiguous()
            elif name.startswith("meta__"):
                if t.numel() == 1:
                    out[name] = int(t.flatten()[0].item())
                else:
                    out[name] = [int(v) for v in t.view(-1).tolist()]
            else:
                out[name] = t.to(torch.float32)
    return out


def resolve_boundary_layers(kiln_dump: Dict[str, object], capture_b12: bool) -> List[int]:
    raw = kiln_dump.get("meta__boundary_layers")
    if isinstance(raw, list) and raw:
        return sorted({int(v) for v in raw})

    layers = set(B10_BOUNDARY_LAYERS)
    if capture_b12:
        layers.update(B12_GQA_LAYERS)
    return sorted(layers)


def resolve_c41_tap_names(kiln_dump: Dict[str, object]) -> List[str]:
    raw = kiln_dump.get("meta__c41_tap_ids")
    if isinstance(raw, list) and raw:
        names: List[str] = []
        for idx in raw:
            idx_i = int(idx)
            if 0 <= idx_i < len(C41_LAYER1_TAP_NAMES):
                names.append(C41_LAYER1_TAP_NAMES[idx_i])
        if names:
            return names

    from_keys = sorted(
        {
            key[len("c41__") :]
            for key in kiln_dump.keys()
            if key.startswith("c41__")
        },
        key=lambda name: (
            C41_LAYER1_TAP_NAMES.index(name)
            if name in C41_LAYER1_TAP_NAMES
            else len(C41_LAYER1_TAP_NAMES),
            name,
        ),
    )
    if from_keys:
        return from_keys
    return list(C41_LAYER1_TAP_NAMES)


def resolve_c42_tap_names(kiln_dump: Dict[str, object]) -> List[str]:
    raw = kiln_dump.get("meta__c42_tap_ids")
    if isinstance(raw, list) and raw:
        names: List[str] = []
        for idx in raw:
            idx_i = int(idx)
            if 0 <= idx_i < len(C42_LAYER1_NORM_TAP_NAMES):
                names.append(C42_LAYER1_NORM_TAP_NAMES[idx_i])
        if names:
            return names

    from_keys = sorted(
        {
            key[len("c42__") :]
            for key in kiln_dump.keys()
            if key.startswith("c42__")
        },
        key=lambda name: (
            C42_LAYER1_NORM_TAP_NAMES.index(name)
            if name in C42_LAYER1_NORM_TAP_NAMES
            else len(C42_LAYER1_NORM_TAP_NAMES),
            name,
        ),
    )
    if from_keys:
        return from_keys

    return list(C42_LAYER1_NORM_TAP_NAMES)


# -----------------------------------------------------------------------------
# Canonical fallback prompt
# -----------------------------------------------------------------------------


CANONICAL_FALLBACK_PROMPT = (
    "Hello, world. This is a deterministic greeting used by the kiln MTP "
    "bench harness when no explicit prompt is provided. It exists so that "
    "the reference dump can reconstruct the exact sequence kiln saw even "
    "when the dump does not embed its own prompt_tokens array. Please reply "
    "with a simple acknowledgement that describes what you just read and why "
    "deterministic prompts matter for reproducibility. "
)


def _pad_or_trim_tokens_to(
    tokens: torch.Tensor, target_len: int, pad_token_id: int
) -> torch.Tensor:
    if tokens.shape[-1] == target_len:
        return tokens
    if tokens.shape[-1] > target_len:
        return tokens[..., :target_len].contiguous()
    # Pad on the right with pad_token_id.
    pad = torch.full(
        (*tokens.shape[:-1], target_len - tokens.shape[-1]),
        pad_token_id,
        dtype=tokens.dtype,
    )
    return torch.cat([tokens, pad], dim=-1).contiguous()


def build_fallback_prompt_tokens(
    tokenizer, target_len: int = 512, seed: int = 42
) -> torch.Tensor:
    """Build the canonical 512-token prompt. Uses a deterministic text
    seed; pads or trims to exactly target_len tokens."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    text = CANONICAL_FALLBACK_PROMPT
    # Repeat until we have enough raw tokens to exceed target_len, then trim.
    tokens = tokenizer(text, return_tensors="pt").input_ids
    while tokens.shape[-1] < target_len:
        text = text + " " + CANONICAL_FALLBACK_PROMPT
        tokens = tokenizer(text, return_tensors="pt").input_ids
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return _pad_or_trim_tokens_to(tokens, target_len, pad_id)


# -----------------------------------------------------------------------------
# Phase B11b — instrumented layer-0 GDN forward
# -----------------------------------------------------------------------------


def _l2_norm_last_dim(x: "torch.Tensor", eps: float = 1e-6) -> "torch.Tensor":
    """L2-normalize along the final dim. Matches the
    `use_qk_l2norm_in_kernel=True` behavior applied by the fla-org
    `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` kernels
    to query and key tensors right before recurrence. We compute this in
    FP32 for stability; callers cast back if needed."""
    norm = x.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()
    return x / norm


def _get_base_model(model):
    """Unwrap multimodal / causal-LM wrappers to return the transformer stack
    whose `.embed_tokens` and `.layers` we want to hook.

    * Qwen3-VL-style multimodal: `model.language_model.model` is the stack.
    * Plain causal LMs: `model.model` is the stack.
    * Already-unwrapped stacks (has `embed_tokens` + `layers`): returned as-is.
    """
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return model.language_model.model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "embed_tokens") and hasattr(model, "layers"):
        return model
    raise RuntimeError(
        "Could not locate base transformer stack: expected model.model, "
        "model.language_model.model, or top-level embed_tokens+layers."
    )


def _find_gdn_class_and_attr(model) -> Tuple[type, str, str]:
    """Inspect layer 0 to determine which GatedDeltaNet class the checkpoint
    uses and what attribute name the GDN submodule is exposed under. Returns
    `(klass, attr_name, layout)` where `layout` is either `"qwen3_5"`
    (4-way in_proj: in_proj_qkv/in_proj_z/in_proj_b/in_proj_a) or
    `"qwen3_next"` (2-way in_proj: in_proj_qkvz/in_proj_ba).
    """
    base = _get_base_model(model)
    layer_0 = base.layers[0]
    for attr_name, child in layer_0.named_children():
        if hasattr(child, "in_proj_qkv") and hasattr(child, "in_proj_z"):
            return type(child), attr_name, "qwen3_5"
        if hasattr(child, "in_proj_qkvz") and hasattr(child, "in_proj_ba"):
            return type(child), attr_name, "qwen3_next"
    raise RuntimeError(
        "Could not find GatedDeltaNet module in layer 0. Expected a submodule "
        "with either in_proj_qkv (Qwen3_5) or in_proj_qkvz (Qwen3-Next)."
    )


def _import_apply_mask_to_padding_states():
    """Locate `apply_mask_to_padding_states`. Qwen3-Next exports it directly;
    Qwen3_5 may re-export the same helper under a different module path.
    Fall back to a no-op (identity) if the helper isn't found — the function
    is only a safety trim when attention_mask is present, and both classes
    work correctly when attention_mask is None (our case)."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (  # type: ignore
            apply_mask_to_padding_states,
        )

        return apply_mask_to_padding_states
    except Exception:
        pass
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (  # type: ignore
            apply_mask_to_padding_states,
        )

        return apply_mask_to_padding_states
    except Exception:
        pass

    def _identity(hidden_states, attention_mask):
        return hidden_states

    return _identity


def make_qwen3_next_instrumented_gdn_forward(
    taps_out: Dict[str, "torch.Tensor"], target_layer_idx: int = 0
):
    """Return a replacement for `Qwen3NextGatedDeltaNet.forward` (2-way in_proj:
    in_proj_qkvz + in_proj_ba). When `self.layer_idx == target_layer_idx`,
    captures each of the 11 B11b sub-op outputs into `taps_out`."""

    import torch.nn.functional as F  # noqa: F401 — needed inside closure

    apply_mask_to_padding_states = _import_apply_mask_to_padding_states()

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        capture = int(self.layer_idx) == int(target_layer_idx)

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state(self.layer_idx)
            and seq_len == 1
        )

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)

        if capture:
            # gdn_in_proj mirrors kiln's `Tensor::cat(&[&mixed_qkv, &z, &b, &a])`
            # which equals `cat([in_proj_qkvz, in_proj_ba], dim=-1)` because
            # `in_proj_qkvz` packs [q, k, v, z] and `in_proj_ba` packs [b, a].
            taps_out["gdn_in_proj"] = (
                torch.cat([projected_states_qkvz, projected_states_ba], dim=-1)
                .detach()
                .clone()
            )

        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = (
            x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value)
        )

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(
                    mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)

        if capture:
            # Post-silu, post-transpose, shape [B, T, conv_dim]. Matches
            # kiln's `gdn_conv` tap captured at the same semantic point.
            taps_out["gdn_conv"] = mixed_qkv.detach().clone()

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if capture:
            taps_out["gdn_gate_beta"] = beta.detach().clone()  # [B, T, nv]
            taps_out["gdn_gate_g"] = g.detach().clone()  # [B, T, nv]

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2
            )
            key = key.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2
            )

        if capture:
            # HF kernel applies L2 norm internally (`use_qk_l2norm_in_kernel=True`),
            # so we mirror it explicitly here to match kiln's captured
            # post-L2-norm q/k tensors.
            q_normed = _l2_norm_last_dim(query.float())
            k_normed = _l2_norm_last_dim(key.float())
            taps_out["gdn_qk_norm_q"] = q_normed.to(query.dtype).detach().clone()
            taps_out["gdn_qk_norm_k"] = k_normed.to(key.dtype).detach().clone()

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if capture:
            # Recurrence output, shape [B, T, nv, dv]. This is the tensor
            # kiln captures at post-transpose time (matches the input to
            # GatedRMSNorm).
            taps_out["gdn_recur_out"] = core_attn_out.detach().clone()

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        if capture:
            # After GatedRMSNorm + flatten-last-two-dims, shape [B, T, value_dim].
            taps_out["gdn_gated_norm"] = core_attn_out.detach().clone()

        output = self.out_proj(core_attn_out)

        if capture:
            # Final out_proj output, shape [B, T, hidden].
            taps_out["gdn_out_proj"] = output.detach().clone()

        return output

    return forward


def make_qwen3_5_instrumented_gdn_forward(
    taps_out: Dict[str, "torch.Tensor"], target_layer_idx: int = 0
):
    """Return a replacement for `Qwen3_5GatedDeltaNet.forward` (4-way in_proj:
    in_proj_qkv / in_proj_z / in_proj_b / in_proj_a). Structurally mirrors
    the Qwen3-Next variant above, but consumes the four separate projections
    the way kiln's `gated_deltanet_forward()` does.

    Layout (matches kiln's forward.rs `gated_deltanet_forward`):
        mixed_qkv = in_proj_qkv(h)   # [B, T, key_dim*2 + value_dim]
        z         = in_proj_z(h)     # [B, T, value_dim]
        b         = in_proj_b(h)     # [B, T, num_v_heads]
        a         = in_proj_a(h)     # [B, T, num_v_heads]

    The `gdn_in_proj` tap matches kiln's tap: `cat([mixed_qkv, z, b, a], -1)`.
    """

    import torch.nn.functional as F  # noqa: F401 — needed inside closure

    apply_mask_to_padding_states = _import_apply_mask_to_padding_states()

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        capture = int(self.layer_idx) == int(target_layer_idx)

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state(self.layer_idx)
            and seq_len == 1
        )

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if capture:
            # Matches kiln's `Tensor::cat(&[&mixed_qkv, &z, &b, &a])`.
            taps_out["gdn_in_proj"] = (
                torch.cat([mixed_qkv, z, b, a], dim=-1).detach().clone()
            )

        # conv1d expects [B, C, T] — transpose, run, transpose back.
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(
                    mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, T, key_dim*2 + value_dim]

        if capture:
            taps_out["gdn_conv"] = mixed_qkv.detach().clone()

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()  # [B, T, num_v_heads]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if capture:
            taps_out["gdn_gate_beta"] = beta.detach().clone()
            taps_out["gdn_gate_g"] = g.detach().clone()

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2
            )
            key = key.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2
            )

        if capture:
            q_normed = _l2_norm_last_dim(query.float())
            k_normed = _l2_norm_last_dim(key.float())
            taps_out["gdn_qk_norm_q"] = q_normed.to(query.dtype).detach().clone()
            taps_out["gdn_qk_norm_k"] = k_normed.to(key.dtype).detach().clone()

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if capture:
            taps_out["gdn_recur_out"] = core_attn_out.detach().clone()

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # GatedRMSNorm. For Qwen3_5 z starts as [B, T, value_dim]; reshape to
        # [B, T, num_v_heads, head_v_dim] so the flatten-then-norm pattern
        # matches the Qwen3-Next path (and kiln's GatedRMSNorm application).
        z_4d = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        z_shape_og = z_4d.shape
        core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z_flat = z_4d.reshape(-1, z_4d.shape[-1])
        core_attn_out_flat = self.norm(core_attn_out_flat, z_flat)
        core_attn_out = core_attn_out_flat.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )

        if capture:
            taps_out["gdn_gated_norm"] = core_attn_out.detach().clone()

        output = self.out_proj(core_attn_out)

        if capture:
            taps_out["gdn_out_proj"] = output.detach().clone()

        return output

    return forward


# Back-compat alias for any external caller / old invocation sites.
make_instrumented_gdn_forward = make_qwen3_next_instrumented_gdn_forward


# -----------------------------------------------------------------------------
# Phase B12 — layer-31 GQA sub-op instrumentation via forward hooks
# -----------------------------------------------------------------------------


def _arm_b12_layer31_hooks(base, taps_out: Dict[str, "torch.Tensor"]) -> List[object]:
    """Attach forward hooks to layer-31 submodules to capture the GQA sub-op
    taps listed in `B12_LAYER31_GQA_TAP_NAMES`. Not every tap is observable
    via hooks — the intermediate rope_q / rope_k / attn_out tensors live
    inside the attention forward and can't be read via module hooks alone.
    We capture what hooks can reach; missing taps are flagged on the
    comparator side rather than blocking the run.

    Returns the list of hook handles (caller is responsible for removing).
    """
    handles: List[object] = []
    layer = base.layers[31]

    def _hook(name: str):
        def _fn(_mod, _inputs, output):
            # Some HF modules return tuples; prefer the first tensor.
            tensor = output[0] if isinstance(output, tuple) else output
            taps_out[name] = tensor.detach().clone()

        return _fn

    # Module-level taps we can cleanly address by name.
    # input_layernorm / post_attention_layernorm are stable across Qwen3
    # variants; q_proj/k_proj/v_proj/o_proj live under `self_attn`; gate/up/
    # down projections live under `mlp`.
    if hasattr(layer, "input_layernorm"):
        handles.append(
            layer.input_layernorm.register_forward_hook(_hook("post_input_norm"))
        )
    if hasattr(layer, "post_attention_layernorm"):
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                _hook("post_attn_norm")
            )
        )

    attn = getattr(layer, "self_attn", None)
    if attn is not None:
        for attr, tap in (
            ("q_proj", "q_proj"),
            ("k_proj", "k_proj"),
            ("v_proj", "v_proj"),
            ("o_proj", "o_proj"),
        ):
            if hasattr(attn, attr):
                handles.append(getattr(attn, attr).register_forward_hook(_hook(tap)))
        # HF Qwen3 exposes per-head RMSNorm on q and k as q_norm / k_norm.
        for attr, tap in (("q_norm", "qk_norm_q"), ("k_norm", "qk_norm_k")):
            if hasattr(attn, attr):
                handles.append(getattr(attn, attr).register_forward_hook(_hook(tap)))

    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        for attr, tap in (
            ("gate_proj", "mlp_gate"),
            ("up_proj", "mlp_up"),
            ("down_proj", "mlp_down"),
        ):
            if hasattr(mlp, attr):
                handles.append(getattr(mlp, attr).register_forward_hook(_hook(tap)))

    return handles


def _arm_c41_layer1_hooks(base, taps_out: Dict[str, "torch.Tensor"]) -> List[object]:
    """Attach hooks for the explicit non-GDN boundaries in transformer block 1.

    The internal GDN taps are captured by monkey-patching the layer-1
    GatedDeltaNet forward; these hooks fill in the surrounding block-level
    boundaries that are naturally visible outside the GDN helper:

    - `layer_1_post_input_norm`: output of layer 1's input_layernorm
    - `layer_1_post_attn_residual`: hidden state after `hidden + attn_out`,
      observed as the input to `post_attention_layernorm`
    - `layer_1_output`: final transformer-block output after the MLP residual
    """
    handles: List[object] = []
    layer = base.layers[1]

    def _forward_hook(name: str):
        def _fn(_mod, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            taps_out[name] = tensor.detach().clone()

        return _fn

    def _pre_hook(name: str):
        def _fn(_mod, inputs):
            tensor = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
            taps_out[name] = tensor.detach().clone()

        return _fn

    if hasattr(layer, "input_layernorm"):
        handles.append(
            layer.input_layernorm.register_forward_hook(
                _forward_hook("layer_1_post_input_norm")
            )
        )
    if hasattr(layer, "post_attention_layernorm"):
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                _pre_hook("layer_1_post_attn_residual")
            )
        )
    handles.append(layer.register_forward_hook(_forward_hook("layer_1_output")))
    return handles


def _arm_c42_layer1_norm_hooks(base, taps_out: Dict[str, "torch.Tensor"]) -> List[object]:
    """Attach hooks for the narrowed Phase C42 input-layernorm bisect."""
    handles: List[object] = []
    layer = base.layers[1]
    norm = getattr(layer, "input_layernorm", None)
    if norm is None:
        return handles

    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))

    def _pre_hook(_mod, inputs):
        tensor = inputs[0][0] if isinstance(inputs[0], tuple) else inputs[0]
        residual = tensor.detach().clone()
        taps_out["layer_1_residual_input"] = residual
        residual_f32 = residual.to(torch.float32)
        rms_inv = torch.rsqrt(residual_f32.square().mean(dim=-1, keepdim=True) + eps)
        taps_out["layer_1_input_norm_rms_inv"] = rms_inv
        taps_out["layer_1_input_norm_pre_weight"] = residual_f32 * rms_inv

    def _forward_hook(_mod, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        taps_out["layer_1_post_input_norm"] = tensor.detach().clone()

    handles.append(norm.register_forward_pre_hook(_pre_hook))
    handles.append(norm.register_forward_hook(_forward_hook))
    return handles


# -----------------------------------------------------------------------------
# Reference forward
# -----------------------------------------------------------------------------


def run_reference_forward(
    checkpoint: str,
    input_ids: torch.Tensor,
    device: torch.device,
    boundary_layers: Optional[List[int]] = None,
    capture_b11_layer0: bool = False,
    capture_b12: bool = False,
    capture_c41: bool = False,
    c41_tap_names: Optional[List[str]] = None,
    capture_c42: bool = False,
    c42_tap_names: Optional[List[str]] = None,
    fp32: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run HF Qwen3-Next / Qwen3.5 forward with `output_hidden_states=True`.
    Returns a dict mapping tap name -> F32 tensor.

    - Boundary taps (`h_layer_*`, `h_pre_final_norm`) are last-row slices at
      shape [1, 1, H].
    - `capture_b11_layer0=True`: 11 full-tensor layer-0 GDN sub-op taps
      under keys in `B11_LAYER0_TAP_NAMES`.
    - `boundary_layers`: explicit ordered list of `h_layer_<idx>` taps to
      emit. When omitted, falls back to the legacy B10 set (and the B12 tail
      extension when `capture_b12=True`).
    - `capture_b12=True`: per-layer `h_layer_{24..30}` taps plus layer-31
      GQA sub-op taps under `b12__<name>` keys.
    - `capture_c42=True`: layer-1 residual input + explicit input-layernorm
      intermediates under `c42__<name>` keys.
    - `fp32=True`: load the HF model in float32 (instead of bfloat16) so the
      comparator can distinguish bf16-accumulation noise from real divergence.
    """
    from transformers import AutoModelForCausalLM  # type: ignore

    dtype = torch.float32 if fp32 else torch.bfloat16
    print(
        f"[mtp_h_main_ref] loading HF model from {checkpoint} "
        f"(dtype={dtype}, trust_remote_code=True)",
        file=sys.stderr,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    # Force dtype post-load: Qwen3.5 with trust_remote_code + low_cpu_mem_usage
    # can silently honour `config.torch_dtype` (bf16) instead of the requested
    # `torch_dtype` kwarg, leaving the `--fp32` comparator path running in
    # bf16. Cast explicitly so the dumped activations really are fp32 when
    # requested. No-op on the bf16 default path.
    if fp32:
        model.to(torch.float32)

    base = _get_base_model(model)

    # Detect which GatedDeltaNet class the checkpoint instantiates so B11b
    # monkey-patches the right one. Silently no-op'ing on Qwen3.5 is the bug
    # this patch fixes.
    gdn_klass, _gdn_attr, gdn_layout = _find_gdn_class_and_attr(model)
    print(
        f"[mtp_h_main_ref] detected GDN layout={gdn_layout!r} class={gdn_klass.__name__}",
        file=sys.stderr,
    )

    if sum(bool(v) for v in (capture_b11_layer0, capture_c41, capture_c42)) > 1:
        raise ValueError("capture_b11_layer0, capture_c41, and capture_c42 are mutually exclusive")

    # Set up B11b/C41 instrumentation + forward hooks BEFORE the forward.
    b11_captures: Dict[str, torch.Tensor] = {}
    b12_captures: Dict[str, torch.Tensor] = {}
    c41_captures: Dict[str, torch.Tensor] = {}
    c42_captures: Dict[str, torch.Tensor] = {}
    hook_handles: List[object] = []
    orig_gdn_forward = None
    if capture_b11_layer0:
        orig_gdn_forward = gdn_klass.forward  # type: ignore[attr-defined]
        if gdn_layout == "qwen3_5":
            instrumented = make_qwen3_5_instrumented_gdn_forward(
                b11_captures, target_layer_idx=0
            )
        else:
            instrumented = make_qwen3_next_instrumented_gdn_forward(
                b11_captures, target_layer_idx=0
            )
        gdn_klass.forward = instrumented  # type: ignore[assignment]

        def _save_tok_embed(_mod, _inputs, output):
            b11_captures["tok_embed"] = output.detach().clone()

        def _save_post_input_norm(_mod, _inputs, output):
            b11_captures["layer_0_post_input_norm"] = output.detach().clone()

        hook_handles.append(
            base.embed_tokens.register_forward_hook(_save_tok_embed)
        )
        hook_handles.append(
            base.layers[0].input_layernorm.register_forward_hook(
                _save_post_input_norm
            )
        )

        print(
            f"[mtp_h_main_ref] B11b instrumentation armed: "
            f"monkey-patched {gdn_klass.__name__}.forward + 2 embed/norm hooks",
            file=sys.stderr,
        )
    elif capture_c41:
        orig_gdn_forward = gdn_klass.forward  # type: ignore[attr-defined]
        if gdn_layout == "qwen3_5":
            instrumented = make_qwen3_5_instrumented_gdn_forward(
                c41_captures, target_layer_idx=1
            )
        else:
            instrumented = make_qwen3_next_instrumented_gdn_forward(
                c41_captures, target_layer_idx=1
            )
        gdn_klass.forward = instrumented  # type: ignore[assignment]
        hook_handles.extend(_arm_c41_layer1_hooks(base, c41_captures))
        print(
            f"[mtp_h_main_ref] C41 instrumentation armed: "
            f"monkey-patched {gdn_klass.__name__}.forward at layer 1 + 3 block hooks",
            file=sys.stderr,
        )
    elif capture_c42:
        hook_handles.extend(_arm_c42_layer1_norm_hooks(base, c42_captures))
        print(
            "[mtp_h_main_ref] C42 instrumentation armed: layer 1 input_layernorm "
            "pre/post hooks + explicit norm-local intermediates",
            file=sys.stderr,
        )

    if capture_b12:
        hook_handles.extend(_arm_b12_layer31_hooks(base, b12_captures))
        print(
            f"[mtp_h_main_ref] B12 instrumentation armed: "
            f"{len(b12_captures)} placeholder slots + layer-31 submodule hooks",
            file=sys.stderr,
        )

    input_ids = input_ids.to(device)
    print(
        f"[mtp_h_main_ref] forward: input_ids shape={tuple(input_ids.shape)} device={device}",
        file=sys.stderr,
    )
    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        # Always tear down instrumentation, even if the forward blows up.
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        if orig_gdn_forward is not None:
            gdn_klass.forward = orig_gdn_forward  # type: ignore[assignment]
    # HF convention: `hidden_states` has `num_hidden_layers + 1` entries.
    #   hidden_states[0]   = embedding input
    #   hidden_states[i+1] = output after layer i (for i in 0..num_layers-1)
    # So hidden_states[32] = output after layer 31 = pre-final-norm hidden.
    hs: List[torch.Tensor] = list(out.hidden_states)  # type: ignore[arg-type]
    num_layers = len(hs) - 1
    if num_layers != 32:
        print(
            f"[mtp_h_main_ref] WARNING: expected 32 layers, got {num_layers}; "
            "B10 boundary taps assume Qwen3.5-4B layout",
            file=sys.stderr,
        )

    seq_len = input_ids.shape[-1]
    assert hs[0].shape[1] == seq_len, (
        f"hidden_states shape mismatch: {hs[0].shape} vs seq_len={seq_len}"
    )

    taps: Dict[str, torch.Tensor] = {}
    layers_to_emit: List[int] = (
        sorted(set(boundary_layers)) if boundary_layers is not None else list(B10_BOUNDARY_LAYERS)
    )
    if capture_b12:
        layers_to_emit = sorted(set(layers_to_emit) | set(B12_GQA_LAYERS))
    for li in layers_to_emit:
        assert li < num_layers, f"layer {li} out of range (num_layers={num_layers})"
        # Output AFTER layer `li` sits at index li+1.
        h_after = hs[li + 1]  # [1, seq_len, H]
        last = h_after[:, -1:, :].contiguous()  # [1, 1, H]
        taps[f"h_layer_{li}"] = last

    # Pre-final-norm = output of the last transformer block, i.e. the input
    # to `model.norm`. In HF this is `hidden_states[-1]` (= hidden_states[num_layers]).
    # For the reference checkpoint we double-bind this to a dedicated name so
    # the comparator can track the channel independently from `h_layer_31`.
    h_pre_final = hs[num_layers][:, -1:, :].contiguous()  # [1, 1, H]
    taps["h_pre_final_norm"] = h_pre_final

    # Emit B11b layer-0 sub-op taps if captured. Each tap goes in under the
    # `b11__<name>` key so the comparator can identify them distinct from
    # the boundary taps.
    if capture_b11_layer0:
        missing = [n for n in B11_LAYER0_TAP_NAMES if n not in b11_captures]
        if missing:
            print(
                f"[mtp_h_main_ref] WARNING: B11b instrumentation missed taps: {missing}",
                file=sys.stderr,
            )
        for name in B11_LAYER0_TAP_NAMES:
            if name in b11_captures:
                taps[f"b11__{name}"] = b11_captures[name]

    # Emit B12 layer-31 GQA sub-op taps under `b12__<name>`. We only warn on
    # missing taps — hook-unreachable taps (rope_q, rope_k, attn_out) are
    # expected to be absent until a later patch monkey-patches attention
    # forward.
    if capture_b12:
        missing = [n for n in B12_LAYER31_GQA_TAP_NAMES if n not in b12_captures]
        if missing:
            print(
                f"[mtp_h_main_ref] NOTE: B12 hooks did not capture {missing} "
                "(hook-unreachable taps require attention forward monkey-patch)",
                file=sys.stderr,
            )
        for name in B12_LAYER31_GQA_TAP_NAMES:
            if name in b12_captures:
                taps[f"b12__{name}"] = b12_captures[name]

    if capture_c41:
        wanted = c41_tap_names or list(C41_LAYER1_TAP_NAMES)
        missing = [n for n in wanted if n not in c41_captures]
        if missing:
            print(
                f"[mtp_h_main_ref] WARNING: C41 instrumentation missed taps: {missing}",
                file=sys.stderr,
            )
        for name in wanted:
            if name in c41_captures:
                tensor = c41_captures[name]
                if tensor.ndim >= 2:
                    tensor = tensor[:, -1:, ...].contiguous()
                taps[f"c41__{name}"] = tensor

    if capture_c42:
        wanted = c42_tap_names or list(C42_LAYER1_NORM_TAP_NAMES)
        missing = [n for n in wanted if n not in c42_captures]
        if missing:
            print(
                f"[mtp_h_main_ref] WARNING: C42 instrumentation missed taps: {missing}",
                file=sys.stderr,
            )
        for name in wanted:
            if name in c42_captures:
                tensor = c42_captures[name]
                if tensor.ndim >= 2:
                    tensor = tensor[:, -1:, ...].contiguous()
                taps[f"c42__{name}"] = tensor

    # Free the model and all retained activations before returning.
    del model, out, hs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return taps


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B10 h_main base-model reference dump")
    ap.add_argument("--checkpoint", required=True, help="Qwen3.5-4B HF checkpoint directory")
    ap.add_argument(
        "--kiln-dump",
        required=True,
        help="Kiln MTP dump with hidden-state taps captured via KILN_MTP_DUMP_HIDDEN_STATES=1",
    )
    ap.add_argument("--out", required=True, help="Output path for reference dump")
    ap.add_argument(
        "--device",
        default=None,
        help="Device override (cuda / cpu). Default: cuda if available.",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Fallback prompt length if kiln dump lacks prompt_tokens (default 512)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch/NumPy seed for fallback prompt generation (default 42)",
    )
    ap.add_argument(
        "--b11-taps",
        action="store_true",
        help=(
            "Also capture 11 layer-0 GDN sub-op taps (Phase B11b) by "
            "monkey-patching the checkpoint's GatedDeltaNet.forward "
            "(Qwen3NextGatedDeltaNet or Qwen3_5GatedDeltaNet, auto-detected). "
            "Output tensors are emitted under `b11__<name>` keys matching "
            "B11_TAP_NAMES in crates/kiln-model/src/mtp_debug.rs."
        ),
    )
    ap.add_argument(
        "--b12-taps",
        action="store_true",
        help=(
            "Also capture per-layer h_layer_{24..30} taps plus layer-31 GQA "
            "sub-op taps (Phase B12). Sub-op taps go under `b12__<name>` and "
            "match B12_GQA_TAP_NAMES in crates/kiln-model/src/mtp_debug.rs."
        ),
    )
    ap.add_argument(
        "--c41-taps",
        action="store_true",
        help=(
            "Also capture the explicit Phase C41 transformer-block-1 tap set "
            "(input norm output, layer-1 GDN internals, post-attn residual, "
            "and final layer output). Output tensors are emitted under "
            "`c41__<name>` keys matching C41_LAYER1_TAP_NAMES in "
            "crates/kiln-model/src/mtp_debug.rs. The exact tap order is "
            "resolved from the kiln dump's `meta__c41_tap_ids` when present."
        ),
    )
    ap.add_argument(
        "--c42-taps",
        action="store_true",
        help=(
            "Also capture the explicit Phase C42 layer-1 pre-norm / "
            "input-layernorm tap set (residual input, RMS inverse, pre-weight "
            "normalized hidden, post-input-norm output). Output tensors are "
            "emitted under `c42__<name>` keys matching C42_LAYER1_NORM_TAP_NAMES "
            "in crates/kiln-model/src/mtp_debug.rs. The exact tap order is "
            "resolved from the kiln dump's `meta__c42_tap_ids` when present."
        ),
    )
    ap.add_argument(
        "--fp32",
        action="store_true",
        help=(
            "Load the HF model in float32 instead of bfloat16. Use this to "
            "distinguish numerical bf16-accumulation drift from real kernel "
            "divergence in the B12 comparator."
        ),
    )
    args = ap.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[mtp_h_main_ref] loading kiln dump from {args.kiln_dump}", file=sys.stderr)
    kiln = load_kiln_dump(args.kiln_dump)
    boundary_layers = resolve_boundary_layers(kiln, capture_b12=args.b12_taps)
    c41_tap_names = resolve_c41_tap_names(kiln) if args.c41_taps else None
    c42_tap_names = resolve_c42_tap_names(kiln) if args.c42_taps else None
    draft_token_id = int(kiln.get("meta__draft_token_id", -1))
    mtp_pos = int(kiln.get("meta__mtp_pos", -1))
    print(
        f"[mtp_h_main_ref] draft_token_id={draft_token_id} mtp_pos={mtp_pos}",
        file=sys.stderr,
    )

    # Resolve the replay tokens. Prefer the fully conditioned `replay_tokens`
    # contract added in C39; fall back to legacy `prompt_tokens`, then to the
    # canonical 512-token greeting.
    prompt_tokens: Optional[torch.Tensor] = None
    if "replay_tokens" in kiln:
        raw = kiln["replay_tokens"]  # type: ignore[assignment]
        assert isinstance(raw, torch.Tensor), f"replay_tokens must be a tensor, got {type(raw)}"
        # Kiln writes it as a flat I32 vector; reshape to [1, N].
        if raw.ndim == 1:
            prompt_tokens = raw.view(1, -1).to(torch.int64).contiguous()
        else:
            prompt_tokens = raw.to(torch.int64).contiguous()
        print(
            f"[mtp_h_main_ref] replay_tokens from kiln dump: shape={tuple(prompt_tokens.shape)}",
            file=sys.stderr,
        )
    elif "prompt_tokens" in kiln:
        raw = kiln["prompt_tokens"]  # type: ignore[assignment]
        assert isinstance(raw, torch.Tensor), f"prompt_tokens must be a tensor, got {type(raw)}"
        if raw.ndim == 1:
            prompt_tokens = raw.view(1, -1).to(torch.int64).contiguous()
        else:
            prompt_tokens = raw.to(torch.int64).contiguous()
        print(
            f"[mtp_h_main_ref] prompt_tokens from kiln dump: shape={tuple(prompt_tokens.shape)}",
            file=sys.stderr,
        )
    else:
        print(
            f"[mtp_h_main_ref] kiln dump has no prompt_tokens; falling back to canonical "
            f"{args.seq_len}-token greeting (seed={args.seed})",
            file=sys.stderr,
        )
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        prompt_tokens = build_fallback_prompt_tokens(
            tokenizer, target_len=args.seq_len, seed=args.seed
        )
        print(
            f"[mtp_h_main_ref] fallback prompt_tokens: shape={tuple(prompt_tokens.shape)}",
            file=sys.stderr,
        )

    # Set seed before model load / forward for determinism.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    taps_bf16 = run_reference_forward(
        args.checkpoint,
        prompt_tokens,
        device,
        boundary_layers=boundary_layers,
        capture_b11_layer0=args.b11_taps,
        capture_b12=args.b12_taps,
        capture_c41=args.c41_taps,
        c41_tap_names=c41_tap_names,
        capture_c42=args.c42_taps,
        c42_tap_names=c42_tap_names,
        fp32=args.fp32,
    )

    # Up-cast to F32 and move to CPU for the dump.
    def _to_f32_cpu(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(torch.float32).cpu().contiguous()

    out_dict: Dict[str, torch.Tensor] = {}
    for name, t in taps_bf16.items():
        out_dict[name] = _to_f32_cpu(t)

    # Carry metadata forward for cross-checking in the comparator.
    out_dict["meta__draft_token_id"] = torch.tensor([draft_token_id], dtype=torch.int32)
    out_dict["meta__mtp_pos"] = torch.tensor([mtp_pos], dtype=torch.int32)
    out_dict["meta__prompt_tokens_len"] = torch.tensor(
        [int(prompt_tokens.shape[-1])], dtype=torch.int32
    )
    out_dict["meta__replay_tokens_len"] = torch.tensor(
        [int(prompt_tokens.shape[-1])], dtype=torch.int32
    )
    out_dict["meta__boundary_layers"] = torch.tensor(
        boundary_layers, dtype=torch.int32
    )
    if c41_tap_names:
        out_dict["meta__c41_tap_ids"] = torch.tensor(
            [C41_LAYER1_TAP_NAMES.index(name) for name in c41_tap_names],
            dtype=torch.int32,
        )
    if c42_tap_names:
        out_dict["meta__c42_tap_ids"] = torch.tensor(
            [C42_LAYER1_NORM_TAP_NAMES.index(name) for name in c42_tap_names],
            dtype=torch.int32,
        )
    # Also write the resolved prompt tokens so later reruns can reproduce
    # bit-exactly even if the kiln dump gets rotated.
    out_dict["prompt_tokens"] = prompt_tokens.view(-1).to(torch.int32).contiguous()
    out_dict["replay_tokens"] = prompt_tokens.view(-1).to(torch.int32).contiguous()

    save_file(out_dict, args.out)
    total_bytes = sum(t.numel() * t.element_size() for t in out_dict.values())
    print(
        f"[mtp_h_main_ref] wrote {args.out} ({total_bytes} bytes, {len(out_dict)} tensors)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
