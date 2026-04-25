#!/usr/bin/env python3
"""H18 — hand-rolled HF transformers reference α probe for Qwen3.5-4B native MTP.

Lineage: PR #530 (vLLM 0.19.1 unsupported), PR #532 (SGLang unsupported), and
PR #533 (vLLM v0.20.0 unsupported on driver 550.x) all blocked the external-
OSS-server α-reference path. PR #531 candidate 8 designated this hand-rolled
HF transformers driver as the only remaining viable external α reference for
Qwen3.5-4B + native MTP on A6000 sm_86.

This is NOT a serving system. It is a small reference probe that:

  1. Loads Qwen3.5-4B as the BASE/VERIFIER via HuggingFace transformers
     (trust_remote_code=True so the hybrid 24×GDN + 8×GQA stack dispatches).
  2. Loads the pretrained `mtp.*` head tensors directly from the canonical
     safetensors checkpoint (HF transformers has NO native loader for MTP
     candidate-generation; this script reuses the proven loader from
     `scripts/mtp_reference_dump.py`).
  3. Constructs the MTP head as a pure-Python module matching the Qwen3-Next
     MTP spec (single transformer block + tied LM head).
  4. Drives the k=1 native MTP greedy decode loop with the SAME accept/reject
     semantics as kiln's `speculative_mtp_decode_step`:
        - Draft: mtp_forward(last_token, h_prev, base_pos, mtp_pos) → draft.
        - Verify: model([last_token, draft]) → logits at 2 positions.
        - target_at_0 = argmax(verify_logits[:, -2]).
        - target_at_1 = argmax(verify_logits[:, -1]).
        - accepted = (target_at_0 == draft).
        - On ACCEPT: emit [draft, target_at_1]; base_pos += 2; mtp_pos += 1;
          new last_token = target_at_1; new h_prev = h at position base_pos+1.
        - On REJECT: emit [target_at_0]; base_pos += 1; mtp_pos unchanged;
          new last_token = target_at_0; new h_prev = h at old base_pos.
  5. Computes α per seed = n_accepts / n_steps, then the cross-seed median.

Workload (must match PR #530 / PR #532 / PR #533 / PR #529 byte-for-byte):

  - Seeds: {0, 1, 2}  (matches PR #529 c1_attr CSVs that pinned kiln α)
  - Prompts: PROMPT_POOL[seed % 30] for seeds {0,1,2} → GSM8K prose 0/1/2
  - Prefill 512 tokens, decode max 16 tokens, greedy (T=0)
  - Spec: k=1 native MTP, no chat template, raw prose
  - Verifier dtype: BF16 on GPU if CUDA is available else CPU

Note on attention semantics: the reference MTP path treats the inner
transformer block as single-position self-attention (Q-len = K-len = 1) per
`scripts/mtp_reference_dump.py` line 366. Phase C7 documented the structural
divergence with kiln (kiln's kv_len = mtp_pos + 1). For an *upper-bound*
reference α this is the canonical PyTorch path — it matches the public
mtp_reference_dump.py contract that has been used as the source of truth for
B/C-phase audits.

Usage on a RunPod A6000 with HF transformers + safetensors installed:

    python3 scripts/h18_hf_alpha_dump.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --prompt-tokens 512 --max-output-tokens 16 \\
        --seeds 0 1 2 \\
        --out docs/phase-c29-v3-hf/hf_alpha_per_seed.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# Re-use the proven MTP-head loader and ops from the existing reference
# (mtp_reference_dump.py — used as the canonical PyTorch source of truth in
# every Phase B/C MTP audit since PR #253).
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import mtp_reference_dump  # noqa: E402  ← script-local import
from mtp_reference_dump import (  # noqa: E402  ← script-local import
    MtpRefWeights,
    load_mtp_weights,
    mtp_inner_block,
    rms_norm,
)
from h15c_vllm_alpha_dump import (  # noqa: E402  ← script-local import
    BASE_PROMPTS,
    build_prompt,
)


def _apply_rope_partial_device_aware(
    x, position: int, head_dim: int, rotary_dim: int, rope_theta: float
):
    """Device-aware RoPE — drop-in replacement for `mtp_reference_dump.apply_rope_partial`.

    The canonical reference builds inv_freq/cos/sin via `torch.arange()` and
    `torch.tensor([position])` with no device argument, defaulting to CPU. That
    matches every Phase B/C audit that loads kiln dumps as CPU float32. For H18
    we drive the MTP head on GPU (BF16) so the rotation tensors must live on
    `x.device`. Monkey-patched into `mtp_reference_dump` at import time below.
    Numerics are bit-identical to the original (same float32 inv_freq math, same
    half-rotate ordering, same final upcast back to `x.dtype`).
    """
    import torch

    if rotary_dim == 0:
        return x
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device)
            / rotary_dim
        )
    )
    freqs = torch.tensor([position], dtype=torch.float32, device=x.device)[:, None] * inv_freq[None, :]
    cos = freqs.cos()
    sin = freqs.sin()

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    cos_a = cos.to(x.dtype)
    sin_a = sin.to(x.dtype)
    x_rot_new = torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1)
    return torch.cat([x_rot_new, x_pass], dim=-1)


# Inject the device-aware RoPE into the imported module so `mtp_inner_block`
# (which calls the module-level `apply_rope_partial` directly) picks it up.
mtp_reference_dump.apply_rope_partial = _apply_rope_partial_device_aware
apply_rope_partial = _apply_rope_partial_device_aware  # noqa: F841 — re-export


def device_dtype():
    import torch  # delayed so the failure mode below catches a missing torch

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def mtp_forward(
    last_token: int,
    h_prev,  # torch.Tensor [1, 1, H], post-final-norm
    weights: MtpRefWeights,
    base_pos: int,
    mtp_pos: int,
    rope_theta: float,
    rotary_frac: float,
    rms_eps: float,
):
    """Run one MTP draft step. Returns (mtp_logits [1, V], hidden_post_layer [1, 1, H]).

    Mirrors `mtp_reference_dump.py` lines 655-700 exactly, packaged for repeated
    invocation in a decode loop.
    """
    import torch

    # 1. Token embed for last_token (the base-model-emitted token whose
    # continuation the MTP head is drafting).
    tok_embed = weights.embed_tokens[last_token : last_token + 1].unsqueeze(0)  # [1,1,H]

    # 2-3. Dual RMSNorms (Qwen3-Next: pre_fc_norm_embedding for token embed,
    # pre_fc_norm_hidden for h_main). swap_fc_norms=False is the canonical
    # production setting per PR #253; we do not test the swap here.
    norm_emb = rms_norm(tok_embed, weights.pre_fc_norm_embedding, rms_eps)
    norm_h = rms_norm(h_prev, weights.pre_fc_norm_hidden, rms_eps)

    # 4. Concat + fc.
    fc_input = torch.cat([norm_emb, norm_h], dim=-1)  # [1,1,2H]
    fc_output = torch.matmul(fc_input, weights.fc_weight.t())  # [1,1,H]

    # 5. Single inner transformer block. Single-position self-attention; the
    # inner block reads `base_pos + mtp_pos` for RoPE but Q-len=K-len=1 means
    # the rotations cancel in Q·K^T and the attention output reduces to V.
    post_layer = mtp_inner_block(
        fc_output,
        weights,
        mtp_pos,
        rope_theta,
        rotary_frac,
        rms_eps,
        capture_subops=None,
        base_pos=base_pos,
        capture_c7=None,
    )

    # 6-7. Final norm + tied LM head.
    post_final_ln = rms_norm(post_layer, weights.final_layernorm, rms_eps)
    mtp_logits = torch.matmul(post_final_ln, weights.embed_tokens.t())  # [1,1,V]
    return mtp_logits[0, 0], post_layer  # logits [V], post_layer [1,1,H]


def _move_weights_to_device(weights: MtpRefWeights, device, dtype):
    """Push the MtpRefWeights tensors to the given device + dtype.

    `mtp_reference_dump.py` loads everything as float32 on CPU. For decode
    parity with the HF base model (BF16 on GPU) we cast and move once up-front.
    """
    import torch

    fields = [
        "embed_tokens",
        "fc_weight",
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "final_layernorm",
        "input_layernorm",
        "post_attention_layernorm",
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "o_proj_weight",
        "q_norm_weight",
        "k_norm_weight",
        "gate_proj_weight",
        "up_proj_weight",
        "down_proj_weight",
    ]
    for name in fields:
        t = getattr(weights, name)
        # Norm weights stay in their original dtype (float32) so rms_norm's
        # internal F32 promotion does not lose precision; matmul weights go to
        # the model dtype to match the base model.
        if "norm" in name or name in {"q_norm_weight", "k_norm_weight"}:
            setattr(weights, name, t.to(device=device, dtype=torch.float32))
        else:
            setattr(weights, name, t.to(device=device, dtype=dtype))
    return weights


def load_base_model(checkpoint: str, device, dtype):
    """Load the Qwen3.5-4B base model via HF transformers.

    Qwen3.5-4B ships as `Qwen3_5ForConditionalGeneration` (multimodal text +
    vision). For pure-text reference α we want the language-model forward
    path. AutoModelForCausalLM dispatches via trust_remote_code; this matches
    `scripts/mtp_h_main_reference_dump.py` line 67-74.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[h18_hf] loading tokenizer from {checkpoint}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(checkpoint)

    print(
        f"[h18_hf] loading base model from {checkpoint} (dtype={dtype}, "
        f"device={device}, trust_remote_code=True)",
        file=sys.stderr,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(
        f"[h18_hf] base model loaded: class={type(model).__name__}",
        file=sys.stderr,
    )
    return model, tok


def base_forward(model, input_ids):
    """Run a single base-model forward and return (logits, hidden_states_last_layer).

    No KV cache (use_cache=False) — H18 recomputes from scratch each call. With
    16 output tokens and ~512-prompt context, ~22 forwards per seed × 3 seeds
    is well within budget on A6000.
    """
    import torch

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    # Last layer's post-final-norm hidden states. HF returns hidden_states
    # as a tuple (embed_out, layer1_out, ..., final_layernorm_out); index -1
    # is post-final-norm matching the vLLM/SGLang `last_hidden_state` contract
    # that kiln's PR #285 (Phase C18) aligned to.
    hidden_last = out.hidden_states[-1]
    return out.logits, hidden_last


def run_single_seed(
    model,
    tokenizer,
    weights: MtpRefWeights,
    seed: int,
    prompt_tokens: int,
    max_output_tokens: int,
    device,
    dtype,
    rope_theta: float,
    rotary_frac: float,
    rms_eps: float,
    eos_token_ids: list[int],
):
    """Run the k=1 native MTP greedy decode loop for one prompt seed."""
    import torch

    # Build the same byte-for-byte prompt PR #529 captured kiln α on.
    prompt_text = build_prompt(tokenizer, prompt_tokens, seed)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    print(
        f"[h18_hf] seed={seed} prompt_tokens={n_prompt} prompt_prefix={prompt_text[:80]!r}",
        file=sys.stderr,
    )

    # Initial prefill — sample first emitted token + capture h_prev for it.
    t_seed_start = time.perf_counter()
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    logits, hiddens = base_forward(model, input_ids)
    # First emitted token = greedy from logits at last prompt position.
    first_logits = logits[0, -1].to(torch.float32)
    last_token = int(first_logits.argmax().item())
    # h_prev for the first MTP step = hidden at last prompt position (post-
    # final-norm). Per Phase C18 this matches the vLLM `last_hidden_state`
    # contract.
    h_prev = hiddens[0, -1:, :].unsqueeze(0).to(dtype)  # [1,1,H]
    # Free the full hidden_states tensor early (saves ~prompt_tokens × H × 2 bytes
    # of VRAM that would otherwise be held until the next forward).
    del logits, hiddens, input_ids
    torch.cuda.empty_cache() if device.type == "cuda" else None

    generated = [last_token]
    base_pos = n_prompt  # position where last_token will be written
    mtp_pos = 0
    n_steps = 0
    n_accepts = 0
    trace = []
    hit_eos = False
    eos_token_set = set(eos_token_ids)

    while len(generated) < max_output_tokens and not hit_eos:
        n_steps += 1

        # 1. MTP draft.
        with torch.inference_mode():
            mtp_logits, _ = mtp_forward(
                last_token,
                h_prev,
                weights,
                base_pos,
                mtp_pos,
                rope_theta,
                rotary_frac,
                rms_eps,
            )
            draft_token = int(mtp_logits.to(torch.float32).argmax().item())

        # 2. Verify forward — recompute from prompt + generated + [draft].
        # generated already includes last_token (= generated[-1]); draft is
        # appended for the verifier to score the next-step prediction at the
        # last_token position AND at the draft position.
        verify_input = list(prompt_ids) + list(generated) + [draft_token]
        v_input_ids = torch.tensor([verify_input], device=device, dtype=torch.long)
        v_logits, v_hiddens = base_forward(model, v_input_ids)

        # logits at last 2 positions:
        # - logits[0, -2] : prediction at last_token's position (= what should
        #   come AFTER last_token). target_at_0 in kiln nomenclature.
        # - logits[0, -1] : prediction at draft's position (= what should come
        #   AFTER draft). target_at_1 / "bonus" on accept.
        target_at_0 = int(v_logits[0, -2].to(torch.float32).argmax().item())
        target_at_1 = int(v_logits[0, -1].to(torch.float32).argmax().item())
        # Hidden states at the same 2 positions:
        # - v_hiddens[0, -2] : h at last_token's position (used as new h_prev
        #   on REJECT, where target_at_0 was sampled from logits at this pos).
        # - v_hiddens[0, -1] : h at draft's position (used as new h_prev on
        #   ACCEPT, where target_at_1 was sampled from logits at this pos).
        h_at_last = v_hiddens[0, -2:-1, :].unsqueeze(0).to(dtype)  # [1,1,H]
        h_at_draft = v_hiddens[0, -1:, :].unsqueeze(0).to(dtype)  # [1,1,H]
        del v_logits, v_hiddens, v_input_ids
        torch.cuda.empty_cache() if device.type == "cuda" else None

        accepted = target_at_0 == draft_token
        if accepted:
            n_accepts += 1

        trace.append(
            {
                "step_idx": n_steps - 1,
                "base_pos": base_pos,
                "mtp_pos": mtp_pos,
                "last_token": last_token,
                "mtp_top1": draft_token,
                "main_top1": target_at_0,
                "accepted": int(accepted),
            }
        )

        if accepted:
            # ACCEPT: emit [draft, bonus] (target_at_1).
            bonus = target_at_1
            if draft_token in eos_token_set:
                generated.append(draft_token)
                hit_eos = True
            else:
                generated.append(draft_token)
                if bonus in eos_token_set:
                    generated.append(bonus)
                    hit_eos = True
                else:
                    generated.append(bonus)
                    last_token = bonus
                    h_prev = h_at_draft
                    base_pos += 2
                    mtp_pos += 1
        else:
            # REJECT: emit [target_at_0].
            if target_at_0 in eos_token_set:
                generated.append(target_at_0)
                hit_eos = True
            else:
                generated.append(target_at_0)
                last_token = target_at_0
                h_prev = h_at_last
                base_pos += 1
                # mtp_pos unchanged

    elapsed = time.perf_counter() - t_seed_start
    alpha = (n_accepts / n_steps) if n_steps > 0 else 0.0
    # Trim generated to max_output_tokens for parity with the kiln bench cap.
    generated = generated[:max_output_tokens]

    # Decode head for sanity-check inclusion in the per-seed record.
    try:
        generated_head = tokenizer.decode(generated[: max(8, max_output_tokens // 2)])
    except Exception:
        generated_head = ""

    print(
        f"[h18_hf] seed={seed} α={alpha:.4f} "
        f"n_accept={n_accepts} n_steps={n_steps} "
        f"hit_eos={hit_eos} elapsed={elapsed:.2f}s "
        f"generated_head={generated_head[:80]!r}",
        file=sys.stderr,
    )

    return {
        "seed": seed,
        "alpha": alpha,
        "n_accept": n_accepts,
        "n_steps": n_steps,
        "n_generated": len(generated),
        "hit_eos": hit_eos,
        "elapsed_s": elapsed,
        "generated_head": generated_head,
        "trace": trace,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/workspace/qwen3.5-4b")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-output-tokens", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rms-eps", type=float, default=1e-6)
    ap.add_argument(
        "--out",
        default="docs/phase-c29-v3-hf/hf_alpha_per_seed.json",
    )
    ap.add_argument(
        "--trace-dir",
        default="docs/phase-c29-v3-hf/artifacts",
        help="Directory for per-seed trace dumps (one JSON per seed)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Module imports are deferred so a missing torch/transformers gives a clean
    # JSON failure record (matches PR #530's failure-mode pattern).
    try:
        import torch  # noqa: F401
        import transformers
    except ImportError as e:
        failure = {
            "mtp_supported": False,
            "reason": f"missing python deps: {type(e).__name__}: {e}",
            "transformers_version": "unavailable",
            "torch_version": "unavailable",
        }
        out_path.write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2))
        return 2

    transformers_version = transformers.__version__
    import torch

    torch_version = torch.__version__
    print(
        f"[h18_hf] transformers={transformers_version} torch={torch_version}",
        file=sys.stderr,
    )

    device, dtype = device_dtype()
    print(f"[h18_hf] device={device} dtype={dtype}", file=sys.stderr)

    # Load MTP head weights from safetensors. Reuses load_mtp_weights from
    # mtp_reference_dump.py — proven across all Phase B/C audits since PR #253.
    print(f"[h18_hf] loading MTP weights from {args.checkpoint}", file=sys.stderr)
    try:
        weights = load_mtp_weights(args.checkpoint)
    except (KeyError, FileNotFoundError) as e:
        failure = {
            "mtp_supported": False,
            "reason": (
                f"hf transformers cannot find pretrained mtp.* head tensors: "
                f"{type(e).__name__}: {e}"
            ),
            "traceback": traceback.format_exc(limit=3),
            "transformers_version": transformers_version,
            "torch_version": torch_version,
        }
        out_path.write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 4

    weights = _move_weights_to_device(weights, device, dtype)
    print(
        f"[h18_hf] MTP head loaded: V={weights.embed_tokens.shape[0]} "
        f"H={weights.embed_tokens.shape[1]} "
        f"fc.shape={tuple(weights.fc_weight.shape)}",
        file=sys.stderr,
    )

    cfg = weights.config
    rope_theta = float(cfg.get("rope_theta", 10_000_000.0) or 10_000_000.0)
    rotary_frac = float(cfg.get("partial_rotary_factor", 0.25) or 0.25)
    print(
        f"[h18_hf] rope_theta={rope_theta} rotary_frac={rotary_frac}",
        file=sys.stderr,
    )

    # Load base model.
    try:
        model, tokenizer = load_base_model(args.checkpoint, device, dtype)
    except Exception as e:
        failure = {
            "mtp_supported": False,
            "reason": (
                f"hf transformers failed to load Qwen3.5-4B base model: "
                f"{type(e).__name__}: {e}"
            ),
            "traceback": traceback.format_exc(limit=5),
            "transformers_version": transformers_version,
            "torch_version": torch_version,
        }
        out_path.write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 5

    # EOS tokens from tokenizer / generation_config.
    eos_token_ids = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_token_ids.append(int(tokenizer.eos_token_id))
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "eos_token_id", None) is not None:
        eos = gen_cfg.eos_token_id
        if isinstance(eos, list):
            eos_token_ids.extend(int(t) for t in eos)
        else:
            eos_token_ids.append(int(eos))
    eos_token_ids = sorted(set(eos_token_ids))
    print(f"[h18_hf] eos_token_ids={eos_token_ids}", file=sys.stderr)

    per_seed = []
    alphas = []
    total_start = time.perf_counter()

    for seed in args.seeds:
        try:
            result = run_single_seed(
                model,
                tokenizer,
                weights,
                seed=seed,
                prompt_tokens=args.prompt_tokens,
                max_output_tokens=args.max_output_tokens,
                device=device,
                dtype=dtype,
                rope_theta=rope_theta,
                rotary_frac=rotary_frac,
                rms_eps=args.rms_eps,
                eos_token_ids=eos_token_ids,
            )
        except Exception as e:
            print(
                f"[h18_hf] seed={seed} FAILED: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
            result = {
                "seed": seed,
                "alpha": 0.0,
                "n_accept": 0,
                "n_steps": 0,
                "n_generated": 0,
                "hit_eos": False,
                "elapsed_s": 0.0,
                "generated_head": "",
                "trace": [],
                "error": f"{type(e).__name__}: {e}",
            }
        per_seed.append(
            {k: v for k, v in result.items() if k != "trace"}
        )
        alphas.append(result["alpha"])
        # Trace per seed → its own JSON for diff against c1_attr CSVs later.
        trace_path = trace_dir / f"hf_trace_seed{seed}.json"
        trace_path.write_text(json.dumps(result, indent=2))
        # Free GPU memory between seeds (forward stack accumulates otherwise).
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_dt = time.perf_counter() - total_start

    # mtp_supported = the MTP head loaded AND every seed produced ≥1 step.
    mtp_supported = all(s["n_steps"] > 0 for s in per_seed)

    result = {
        "mtp_supported": mtp_supported,
        "per_seed": per_seed,
        "median_alpha": statistics.median(alphas) if alphas else 0.0,
        "mean_alpha": statistics.mean(alphas) if alphas else 0.0,
        "min_alpha": min(alphas) if alphas else 0.0,
        "max_alpha": max(alphas) if alphas else 0.0,
        "n_seeds": len(alphas),
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "config": {
            "checkpoint": args.checkpoint,
            "dtype": str(dtype),
            "device": str(device),
            "prompt_tokens": args.prompt_tokens,
            "max_output_tokens": args.max_output_tokens,
            "seeds": args.seeds,
            "rms_eps": args.rms_eps,
            "rope_theta": rope_theta,
            "rotary_frac": rotary_frac,
            "eos_token_ids": eos_token_ids,
            "chat_template": False,
            "spec_method": "qwen3_5_mtp_handrolled_hf",
            "num_speculative_tokens": 1,
            "sampling": {"temperature": 0.0, "greedy": True},
        },
        "total_elapsed_s": total_dt,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))
    for s in per_seed:
        print(
            f"  seed={s['seed']} α={s['alpha']:.4f} "
            f"({s['n_accept']}/{s['n_steps']})",
            file=sys.stderr,
        )
    return 0 if mtp_supported else 6


if __name__ == "__main__":
    raise SystemExit(main())
