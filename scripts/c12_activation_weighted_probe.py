#!/usr/bin/env python3
"""
Phase C12 — activation-weighted Marlin drift probe.

Goal
----

C11 (PR #326) measured per-channel Marlin W4A16 scale drift on all 104
Marlin-packed projections and found some output channels outside the
fp32-equivalence band (cos_sim < 0.9999). C11 also noted — in the "Honest
limits" section — that a per-channel scale outside-band on the *weight*
may still produce in-band *activations* on realistic data if the signal
never excites that channel.

This probe closes that gap by re-weighting C11's pure-weight drift by an
empirical activation distribution. For every Marlin-packed projection
with weight `W` (HF layout [out, in]) and Marlin-dequant weight `W_dq`:

    delta = W - W_dq                        # [out, in]
    X = stack of activations feeding W      # [T, in] over calibration set
    numerator   = || X @ delta.T ||_F       # Frobenius norm of output error
    denominator = || X @ W.T    ||_F
    activation_weighted_drift_ratio = numerator / denominator

This is the empirical output-level SNR contribution of the dequant drift,
observed on realistic inputs. If it is >= 1e-3 on the worst projection
that C11 flagged, the C11 drift is not a benign artifact — it projects
onto high-activation channels and is a plausible α-suppressing signal.
If it is well below 1e-3, C11 drift is safely hidden under the activation
distribution and the alpha floor comes from elsewhere.

This script is support evidence for the C12 fp32-head kill-switch bench.
A PRIMARY-POSITIVE bench (α recovers with `KILN_MTP_FP32_HEAD=1`) would
be corroborated by a non-trivial weighted drift ratio here; a
PRIMARY-NEGATIVE bench would be corroborated by a benign weighted drift
ratio here.

Scope
-----

C11 confirmed the MTP head is NOT Marlin-packed. The projections this
probe audits live on the *main model* (8 full-attn q_proj + 32*3 MLP).
Those are the weights the base model uses when producing the `h_main`
hidden state that feeds `mtp_forward_step`. Even if none of the MTP
*head* matmuls are Marlin-packed, the base-model Marlin projections are
upstream of the MTP head and could still be a distal α-suppressing
signal via the `h_main` tap. Auditing them is therefore on-scope for C12.

Calibration set
---------------

Reuses the 5 fixed prompts from `scripts/mtp_reference_dump.py` plus 27
additional Wikipedia-style prompts shipped below (= 32 prompts total,
32-96 tokens each). Short enough to be tractable on an A6000 in a few
minutes; long enough to produce representative activation statistics
across the hidden dimension.

Runs on GPU when available; falls back to CPU. CPU run is feasible but
takes ~15 minutes on a modest host. A6000 completes in ~60 seconds.

Usage
-----

    python3 scripts/c12_activation_weighted_probe.py \\
        --checkpoint /workspace/qwen3.5-4b \\
        --c11-json docs/archive/phase-c/phase-c11/c11-marlin-audit.json \\
        --out docs/archive/phase-c/phase-c12/c12-weighted-drift.md \\
        --out-json docs/archive/phase-c/phase-c12/c12-weighted-drift.json \\
        [--device cuda] [--dtype bf16] [--max-prompts 32] [--max-tokens 96]

Exit codes
----------

    0 — probe ran; weighted drift ratio is BENIGN on all projections
        (< 1e-3 on the worst projection C11 flagged).  Corroborates
        PRIMARY-NEGATIVE.
    1 — probe ran; weighted drift ratio is NON-TRIVIAL (>= 1e-3) on at
        least one projection.  Corroborates PRIMARY-POSITIVE.
    2 — structural error (missing weight, bad CLI, CUDA OOM).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open


# Calibration prompts. 5 from mtp_reference_dump.py style + 27 general.
CALIBRATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the Word, and the Word was with God.",
    "To be or not to be, that is the question. Whether tis nobler in the mind to suffer",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation",
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "The Roman Empire reached its greatest territorial extent under the emperor Trajan in 117 AD.",
    "Quantum mechanics describes the behavior of matter and light at atomic and subatomic scales.",
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells.",
    "Shakespeare wrote approximately 39 plays, 154 sonnets, and two long narrative poems.",
    "The Pacific Ocean is the largest and deepest of the Earth's five oceanic divisions.",
    "Albert Einstein published the theory of special relativity in 1905 while working as a patent clerk.",
    "The internet originated as ARPANET, a project funded by the US Department of Defense.",
    "DNA is a double-helix molecule that encodes genetic instructions for living organisms.",
    "The French Revolution began in 1789 with the storming of the Bastille in Paris.",
    "Photography was invented in the early nineteenth century by Niepce and Daguerre.",
    "The Great Wall of China was built over centuries by successive imperial dynasties.",
    "Computer science is the study of algorithms, computation, and information processing.",
    "The Industrial Revolution began in Britain in the late eighteenth century.",
    "Vincent van Gogh painted over 2,100 artworks in a career that lasted just over a decade.",
    "Gravity is described by Einstein's general theory of relativity as the curvature of spacetime.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "The Renaissance was a period of cultural rebirth that began in Italy in the fourteenth century.",
    "Python is a high-level programming language known for readable syntax and broad applicability.",
    "The Amazon rainforest produces roughly 20% of the oxygen in Earth's atmosphere.",
    "The Beatles formed in Liverpool in 1960 and disbanded in 1970 after ten studio albums.",
    "Black holes are regions of spacetime where gravitational attraction is so strong",
    "Mount Everest is the highest mountain on Earth, with a peak elevation of 8,849 meters.",
    "The Turing test, proposed in 1950, evaluates a machine's ability to exhibit human-like intelligence.",
    "Climate change is the long-term alteration of temperature and typical weather patterns.",
    "The Milky Way galaxy contains between 100 and 400 billion stars and has a spiral shape.",
    "Abraham Lincoln delivered the Gettysburg Address on November 19, 1863, during the Civil War.",
]


MAX_Q = 15
Q_OFFSET = 8


# -----------------------------------------------------------------------------
# Safetensors weight loading — mirrors scripts/c11_marlin_audit.py
# -----------------------------------------------------------------------------

def st_load_tensor(checkpoint_dir: str, name: str, allow_missing: bool = False) -> Optional[torch.Tensor]:
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
    idx_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        keys = json.load(f)["weight_map"].keys()
    for prefix in ["model.language_model.", "model.", ""]:
        if prefix + "layers.0.mlp.down_proj.weight" in keys:
            return prefix
    raise KeyError("Could not discover layer prefix")


def marlin_dequant_groupwise(weight_t_f32: np.ndarray, groupsize: int = 128) -> np.ndarray:
    """Reproduce kiln_marlin_gemm::pack::quantize_and_pack's dequant on
    weight_t layout [k=in, n=out]. Returns [k, n] dequantized weights."""
    k, n = weight_t_f32.shape
    assert k % groupsize == 0, f"k={k} not divisible by groupsize={groupsize}"
    num_groups = k // groupsize
    w = weight_t_f32.reshape(num_groups, groupsize, n)

    max_abs = np.max(np.abs(w), axis=1)
    max_abs = np.where(max_abs == 0.0, 1.0, max_abs)
    s_f32 = (2.0 * max_abs) / float(MAX_Q)
    s_f16 = s_f32.astype(np.float16).astype(np.float32)

    s_broadcast = s_f16[:, None, :]
    q_raw = np.round(w / s_broadcast).astype(np.int32)
    q_shifted = np.clip(q_raw + Q_OFFSET, 0, MAX_Q)
    w_dq = (q_shifted - Q_OFFSET).astype(np.float32) * s_broadcast
    return w_dq.reshape(k, n)


# -----------------------------------------------------------------------------
# Activation capture via transformers hooks
# -----------------------------------------------------------------------------

@dataclass
class ProjectionResult:
    layer_idx: int
    kind: str
    n_in: int
    n_out: int
    weighted_drift_ratio: float    # || X @ delta.T || / || X @ W.T ||
    abs_err_frobenius: float
    weight_drift_frobenius: float  # || delta || (unweighted, for cross-check vs C11)
    n_tokens: int


def compute_projection_drift(
    weight_hf: torch.Tensor,      # [out, in] fp32
    activations: torch.Tensor,    # [T, in] fp32
) -> Tuple[float, float, float]:
    """Return (weighted_drift_ratio, abs_err_frob, weight_drift_frob).

    `weight_drift_ratio` is || X @ delta.T || / || X @ W.T || where delta
    is the fp32 difference between HF weights and Marlin-dequantized
    weights (groupsize=128).
    """
    weight_t = weight_hf.t().contiguous()  # [in, out]
    w_t_np = weight_t.numpy().astype(np.float32, copy=False)
    w_dq_t_np = marlin_dequant_groupwise(w_t_np, groupsize=128)
    delta_t = torch.from_numpy(w_t_np - w_dq_t_np)  # [in, out]

    # X @ W.T_hf gives base output; X @ delta_t gives error.
    out = activations @ weight_t            # [T, out]
    err = activations @ delta_t             # [T, out]

    out_norm = float(torch.linalg.norm(out).item())
    err_norm = float(torch.linalg.norm(err).item())
    weight_drift_frob = float(torch.linalg.norm(delta_t).item())
    ratio = err_norm / max(out_norm, 1e-30)
    return ratio, err_norm, weight_drift_frob


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="markdown summary")
    parser.add_argument("--out-json", required=True, help="full JSON dump")
    parser.add_argument("--c11-json", default=None, help="optional — cross-ref C11 audit")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--threshold-benign", type=float, default=1e-3,
        help="below this weighted drift ratio = BENIGN (exit 0). Default 1e-3.",
    )
    args = parser.parse_args()

    t_start = time.time()

    # Dynamic import to avoid demanding transformers in CI / linter.
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run `pip install transformers`.", file=sys.stderr)
        return 2

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[c12] device={device} dtype={args.dtype}", file=sys.stderr)

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[c12] loading model from {args.checkpoint}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()

    prefix = discover_prefix(args.checkpoint)
    print(f"[c12] layer prefix = `{prefix}`", file=sys.stderr)

    # Hook each linear we care about (q_proj on full-attn layers; gate/up/down on all).
    activation_buffers: Dict[str, List[torch.Tensor]] = {}

    def hook_for(key: str):
        def _fn(_mod, inputs, _out):
            x = inputs[0].detach()  # [..., in]
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu()
            activation_buffers.setdefault(key, []).append(x_flat)
        return _fn

    # Walk the underlying transformer blocks to install hooks.
    # For Qwen3-Next the layers live under `model.model.layers` or
    # `model.model.language_model.layers` (prefix we discovered tells us).
    root = model
    if hasattr(root, "model"):
        root = root.model
    # Qwen3.5-4B nests as model.language_model.layers
    if hasattr(root, "language_model"):
        root = root.language_model
    layers = root.layers
    hooks = []
    n_full_attn = 0
    for i, layer in enumerate(layers):
        # MLP gate/up/down always present.
        if hasattr(layer, "mlp"):
            for sub in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(layer.mlp, sub, None)
                if lin is not None:
                    hooks.append(lin.register_forward_hook(hook_for(f"L{i}.{sub}")))
        # q_proj only on full-attention layers.
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "q_proj"):
            hooks.append(sa.q_proj.register_forward_hook(hook_for(f"L{i}.q_proj")))
            n_full_attn += 1
    print(f"[c12] installed {len(hooks)} hooks across {len(layers)} layers "
          f"({n_full_attn} full-attn q_proj hooks)", file=sys.stderr)

    # Run calibration forward.
    prompts = CALIBRATION_PROMPTS[:args.max_prompts]
    tok = 0
    with torch.inference_mode():
        for pi, p in enumerate(prompts):
            ids = tokenizer(p, return_tensors="pt", truncation=True,
                            max_length=args.max_tokens).to(device)
            _ = model(**ids)
            tok += int(ids["input_ids"].shape[1])
            if pi < 3:
                print(f"[c12] prompt {pi}: {ids['input_ids'].shape[1]} tokens", file=sys.stderr)
    print(f"[c12] captured activations over {tok} total tokens across {len(prompts)} prompts",
          file=sys.stderr)

    for h in hooks:
        h.remove()

    # Free the HF model to make room for NumPy weight-dequant work.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # For each hook key, concat activations to a single [T, in] fp32 tensor
    # and compute the drift against the HF weight.
    results: List[ProjectionResult] = []
    worst_ratio = 0.0
    worst_name = ""
    for key, chunks in sorted(activation_buffers.items(),
                              key=lambda kv: (int(kv[0].split('.')[0][1:]), kv[0])):
        layer_tag, kind = key.split(".", 1)
        layer_idx = int(layer_tag[1:])
        X = torch.cat(chunks, dim=0)  # [T, in]
        weight_name = (
            f"{prefix}layers.{layer_idx}.self_attn.q_proj.weight"
            if kind == "q_proj"
            else f"{prefix}layers.{layer_idx}.mlp.{kind}.weight"
        )
        w_hf = st_load_tensor(args.checkpoint, weight_name, allow_missing=True)
        if w_hf is None:
            print(f"[c12] WARN: missing {weight_name}; skipping", file=sys.stderr)
            continue
        w_hf = w_hf.to(torch.float32)
        ratio, err_norm, drift_frob = compute_projection_drift(w_hf, X)
        results.append(ProjectionResult(
            layer_idx=layer_idx,
            kind=kind,
            n_in=w_hf.shape[1],
            n_out=w_hf.shape[0],
            weighted_drift_ratio=ratio,
            abs_err_frobenius=err_norm,
            weight_drift_frobenius=drift_frob,
            n_tokens=X.shape[0],
        ))
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_name = key
        if len(results) % 20 == 0:
            print(f"[c12] processed {len(results)} projections "
                  f"(current worst {worst_name} ratio={worst_ratio:.3e})", file=sys.stderr)
    # Sort for reporting: worst ratio first.
    results.sort(key=lambda r: -r.weighted_drift_ratio)

    # C11 cross-reference.
    c11_summary = None
    if args.c11_json and os.path.exists(args.c11_json):
        with open(args.c11_json, "r") as f:
            c11_summary = json.load(f)

    elapsed_s = time.time() - t_start

    # Markdown summary.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Phase C12 — activation-weighted Marlin drift probe\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n")
        f.write(f"- Device: `{device}`, dtype: `{args.dtype}`\n")
        f.write(f"- Calibration prompts: {len(prompts)} ({tok} tokens total)\n")
        f.write(f"- Projections audited: {len(results)}\n")
        f.write(f"- Elapsed: {elapsed_s:.1f}s\n")
        f.write(f"- Benign threshold: weighted_drift_ratio < {args.threshold_benign}\n\n")
        f.write("## Top 10 projections by weighted drift ratio\n\n")
        f.write("| layer | kind | k (in) | n (out) | weighted_ratio | err_frob | drift_frob |\n")
        f.write("|------:|:-----|------:|--------:|----------------:|---------:|-----------:|\n")
        for r in results[:10]:
            f.write(
                f"| {r.layer_idx} | {r.kind} | {r.n_in} | {r.n_out} | "
                f"{r.weighted_drift_ratio:.3e} | {r.abs_err_frobenius:.3e} | "
                f"{r.weight_drift_frobenius:.3e} |\n"
            )
        f.write("\n")
        f.write(f"**Worst projection: `{worst_name}`, weighted_drift_ratio = "
                f"{worst_ratio:.3e}**\n\n")
        if worst_ratio < args.threshold_benign:
            f.write("**Verdict: BENIGN.** Activation-weighted drift is below the "
                    f"{args.threshold_benign:.0e} threshold on every projection. "
                    "Marlin drift does not project onto high-activation channels on "
                    "this calibration set; corroborates PRIMARY-NEGATIVE on the C12 "
                    "fp32-head bench.\n")
        else:
            f.write("**Verdict: NON-TRIVIAL.** At least one projection has an "
                    f"activation-weighted drift ratio >= {args.threshold_benign:.0e}; "
                    "Marlin drift plausibly lands on high-activation channels. "
                    "Corroborates PRIMARY-POSITIVE on the C12 fp32-head bench.\n")
        if c11_summary is not None:
            f.write("\n## C11 cross-reference\n\n")
            f.write("See `docs/archive/phase-c/phase-c11/c11-marlin-audit.md` for the unweighted "
                    "per-channel drift audit. This probe re-weights that drift by "
                    "empirical activation mass on a 32-prompt calibration set, which "
                    "is the realistic input distribution the model actually sees.\n")

    # JSON dump.
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "device": device,
            "dtype": args.dtype,
            "n_prompts": len(prompts),
            "n_tokens": tok,
            "threshold_benign": args.threshold_benign,
            "worst_name": worst_name,
            "worst_ratio": worst_ratio,
            "elapsed_s": elapsed_s,
            "results": [asdict(r) for r in results],
        }, f, indent=2)

    print(f"[c12] wrote {args.out} and {args.out_json}", file=sys.stderr)
    print(f"[c12] worst={worst_name} ratio={worst_ratio:.3e}", file=sys.stderr)

    return 0 if worst_ratio < args.threshold_benign else 1


if __name__ == "__main__":
    sys.exit(main())
