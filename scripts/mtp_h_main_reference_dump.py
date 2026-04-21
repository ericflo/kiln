#!/usr/bin/env python3
"""
Phase B10 — `h_main` base-model reference dump.
Phase B11b — optional layer-0 GDN sub-op taps (`--b11-taps`).

This script produces the independent pure-Python reference counterpart to the
kiln-side main-model forward (the 24×GDN + 8×GQA stack) so that Phase B10 can
bisect which layer first diverges. Prior phases (B6–B9) ruled out every sub-op
inside the MTP head, so the remaining candidate is the base-model `h_prev`
(`h_main`) that kiln feeds into MTP.

Phase B11b extends this with 11 intra-layer taps at layer 0's GDN block so we
can bisect the first diverging sub-op (B10 showed layer 0 as the first
boundary where cos ≈ 0.827 between kiln and HF).

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

* If the kiln dump contains a `meta__prompt_tokens_len` scalar and a
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
            if name == "prompt_tokens":
                # Token IDs — keep integral.
                out[name] = t.to(torch.int64).contiguous()
            elif name.startswith("meta__"):
                out[name] = int(t.flatten()[0].item())
            else:
                out[name] = t.to(torch.float32)
    return out


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


def make_instrumented_gdn_forward(
    taps_out: Dict[str, "torch.Tensor"], target_layer_idx: int = 0
):
    """Return a replacement for `Qwen3NextGatedDeltaNet.forward` that runs the
    module's math unmodified but, when `self.layer_idx == target_layer_idx`,
    captures each of the 11 B11b sub-op outputs into `taps_out`. The returned
    function binds `target_layer_idx` and `taps_out` via closure; assign it
    with `Qwen3NextGatedDeltaNet.forward = ...` to instrument every GDN layer
    in a model (only the target layer will write taps)."""

    import torch.nn.functional as F  # noqa: F401 — needed inside closure
    from transformers.models.qwen3_next.modeling_qwen3_next import (  # type: ignore
        apply_mask_to_padding_states,
    )

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


# -----------------------------------------------------------------------------
# Reference forward
# -----------------------------------------------------------------------------


def run_reference_forward(
    checkpoint: str,
    input_ids: torch.Tensor,
    device: torch.device,
    capture_b11_layer0: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run HF Qwen3-Next forward with `output_hidden_states=True`. Returns a
    dict mapping tap name -> F32 tensor. Boundary taps (`h_layer_*`,
    `h_pre_final_norm`) are last-row slices at shape [1, 1, H]. When
    `capture_b11_layer0=True`, also returns 11 full-tensor layer-0 GDN sub-op
    taps under keys in `B11_LAYER0_TAP_NAMES` (shapes match kiln's)."""
    from transformers import AutoModelForCausalLM  # type: ignore

    print(
        f"[mtp_h_main_ref] loading HF model from {checkpoint} (bf16, trust_remote_code=True)",
        file=sys.stderr,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    # Set up B11b instrumentation + embed/norm forward_hooks BEFORE the forward.
    b11_captures: Dict[str, torch.Tensor] = {}
    hook_handles: List[object] = []
    orig_gdn_forward = None
    if capture_b11_layer0:
        from transformers.models.qwen3_next.modeling_qwen3_next import (  # type: ignore
            Qwen3NextGatedDeltaNet,
        )

        orig_gdn_forward = Qwen3NextGatedDeltaNet.forward  # type: ignore[attr-defined]
        Qwen3NextGatedDeltaNet.forward = make_instrumented_gdn_forward(  # type: ignore[assignment]
            b11_captures, target_layer_idx=0
        )

        base = model.model if hasattr(model, "model") else model

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
            "[mtp_h_main_ref] B11b instrumentation armed: "
            "monkey-patched Qwen3NextGatedDeltaNet.forward + 2 embed/norm hooks",
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
            from transformers.models.qwen3_next.modeling_qwen3_next import (  # type: ignore
                Qwen3NextGatedDeltaNet,
            )

            Qwen3NextGatedDeltaNet.forward = orig_gdn_forward  # type: ignore[assignment]
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
    for li in B10_BOUNDARY_LAYERS:
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
            "monkey-patching Qwen3NextGatedDeltaNet.forward. Output tensors "
            "are emitted under `b11__<name>` keys matching B11_TAP_NAMES in "
            "crates/kiln-model/src/mtp_debug.rs."
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
    draft_token_id = int(kiln.get("meta__draft_token_id", -1))
    mtp_pos = int(kiln.get("meta__mtp_pos", -1))
    print(
        f"[mtp_h_main_ref] draft_token_id={draft_token_id} mtp_pos={mtp_pos}",
        file=sys.stderr,
    )

    # Resolve the prompt tokens. Prefer kiln-provided `prompt_tokens`, fall
    # back to the canonical 512-token greeting.
    prompt_tokens: Optional[torch.Tensor] = None
    if "prompt_tokens" in kiln:
        raw = kiln["prompt_tokens"]  # type: ignore[assignment]
        assert isinstance(raw, torch.Tensor), f"prompt_tokens must be a tensor, got {type(raw)}"
        # Kiln writes it as a flat I32 vector; reshape to [1, N].
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
        args.checkpoint, prompt_tokens, device, capture_b11_layer0=args.b11_taps
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
    out_dict["meta__boundary_layers"] = torch.tensor(
        list(B10_BOUNDARY_LAYERS), dtype=torch.int32
    )
    # Also write the resolved prompt tokens so later reruns can reproduce
    # bit-exactly even if the kiln dump gets rotated.
    out_dict["prompt_tokens"] = prompt_tokens.view(-1).to(torch.int32).contiguous()

    save_file(out_dict, args.out)
    total_bytes = sum(t.numel() * t.element_size() for t in out_dict.values())
    print(
        f"[mtp_h_main_ref] wrote {args.out} ({total_bytes} bytes, {len(out_dict)} tensors)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
