#!/usr/bin/env python3
"""H17b — vLLM v0.20.0 MTP α microbench retest on Qwen3.5-4B A6000 bs=1.

Direct lineage: scripts/h15c_vllm_alpha_dump.py (PR #530, vLLM 0.19.1).

This driver is intentionally a thin wrapper that re-exports the H15c
driver against vLLM v0.20.0 (PyTorch 2.11 + CUDA 13.0 default native
stack vs PR #530's torch 2.10 + cu128). Same workload, same prompts,
same seeds, same decision rule, just a different installed vLLM version.

Decision rule (re-stated, unchanged from PR #530/#531):

    delta = vllm_020_median_alpha - kiln_median_alpha

    delta >= +0.05          → external_ceiling_exists
    -0.02 <= delta < +0.05  → mtp_head_quality_ceiling
    delta <  -0.02          → kiln_above_vllm
    mtp_supported == false  → vllm_020_mtp_unsupported_dense_4b
                              (escalate to hand-rolled HF transformers
                               H18 reference, PR #531 candidate 8)

Usage on a RunPod A6000 pod with vllm==0.20.x installed:

    python3 scripts/h17b_vllm_020_alpha_dump.py \
        --model-path /workspace/qwen3.5-4b \
        --prompt-tokens 512 --max-tokens 16 \
        --seeds 1 2 3 --num-spec-tokens 1 \
        --out docs/archive/phase-c/phase-c29-v3-vllm-020/vllm_020_alpha_per_seed.json

The driver is intentionally version-agnostic — see PR #530 §
"Reproduction" for why we re-run with the same script under a new
vLLM version rather than forking it.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

# Re-use the PR #530 driver verbatim — same prompts, same prompt builder,
# same try_load_llm, same extract_alpha. We only override the default
# --out path and add a vLLM version stamp to the result JSON.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from h15c_vllm_alpha_dump import (  # noqa: E402  ← script-local import
    BASE_PROMPTS,
    build_prompt,
    extract_alpha,
    try_load_llm,
)


def vllm_version() -> str:
    try:
        import vllm  # type: ignore[import-not-found]
        return getattr(vllm, "__version__", "unknown")
    except Exception as e:
        return f"unavailable: {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/qwen3.5-4b")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help=(
            "Seed values matching PR #530's matched workload. PR #530 used "
            "seeds 0 1 2; we mirror seeds 1 2 3 per the H17b task spec to "
            "avoid trivial cache-hit confounders if the same vLLM build "
            "happens to run on the same pod state."
        ),
    )
    ap.add_argument("--num-spec-tokens", type=int, default=1)
    ap.add_argument(
        "--out",
        default="docs/archive/phase-c/phase-c29-v3-vllm-020/vllm_020_alpha_per_seed.json",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build prompts first — needs tokenizer only, no vLLM GPU.
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        failure = {
            "mtp_supported": False,
            "reason": f"transformers not installed: {e}",
            "vllm_version": vllm_version(),
        }
        out_path.write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2))
        return 2

    print(f"[h17b_vllm_020] vllm_version={vllm_version()}", file=sys.stderr)
    print(f"[h17b_vllm_020] loading tokenizer from {args.model_path}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    # Re-use PR #530's prompt builder verbatim. seed_index = seed % len(BASE_PROMPTS).
    prompts = [build_prompt(tok, args.prompt_tokens, s) for s in args.seeds]
    for i, p in enumerate(prompts):
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        print(
            f"[h17b_vllm_020] seed={args.seeds[i]} prompt_tokens={n} "
            f"prompt_prefix={p[:80]!r}",
            file=sys.stderr,
        )

    # Load vLLM with MTP spec decoding (re-uses PR #530's try_load_llm)
    try:
        llm, used_kwargs = try_load_llm(args.model_path, args.num_spec_tokens)
    except (ImportError, RuntimeError) as e:
        failure = {
            "mtp_supported": False,
            "reason": (
                f"vLLM v0.20 failed to load Qwen3.5-4B with MTP "
                f"speculative config: {e}"
            ),
            "traceback": traceback.format_exc(limit=3),
            "vllm_version": vllm_version(),
        }
        out_path.write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 3

    print(
        f"[h17b_vllm_020] vLLM loaded with kwargs={used_kwargs!r}",
        file=sys.stderr,
    )

    # Run identical sampling / generation loop as PR #530 (cannot import the
    # main() function because it builds its own out_path; just inline the
    # generation block here).
    import statistics
    import time

    from vllm import SamplingParams

    per_seed: list[dict] = []
    alphas: list[float] = []
    total_start = time.perf_counter()

    for seed, prompt in zip(args.seeds, prompts):
        sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            seed=seed,
        )
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp)
        dt = time.perf_counter() - t0
        try:
            stats = extract_alpha(outs, llm)
        except RuntimeError as e:
            stats = {
                "accepted": 0,
                "proposed": 0,
                "source": "UNAVAILABLE",
                "error": str(e),
            }
        alpha = stats["accepted"] / stats["proposed"] if stats["proposed"] > 0 else 0.0
        generated_text = outs[0].outputs[0].text[:120] if outs else ""
        per_seed.append(
            {
                "seed": seed,
                "alpha": alpha,
                "n_accepted": stats["accepted"],
                "n_proposed": stats["proposed"],
                "elapsed_s": dt,
                "stats_source": stats.get("source", "UNAVAILABLE"),
                "generated_head": generated_text,
            }
        )
        alphas.append(alpha)
        print(
            f"[h17b_vllm_020] seed={seed} α={alpha:.4f} "
            f"accepted={stats['accepted']} proposed={stats['proposed']} "
            f"source={stats.get('source')} dt={dt:.2f}s",
            file=sys.stderr,
        )

    total_dt = time.perf_counter() - total_start

    result = {
        "mtp_supported": any(s["n_proposed"] > 0 for s in per_seed),
        "per_seed": per_seed,
        "median_alpha": statistics.median(alphas) if alphas else 0.0,
        "mean_alpha": statistics.mean(alphas) if alphas else 0.0,
        "min_alpha": min(alphas) if alphas else 0.0,
        "max_alpha": max(alphas) if alphas else 0.0,
        "n_seeds": len(alphas),
        "vllm_version": vllm_version(),
        "vllm_config": {
            "model_path": args.model_path,
            "dtype": "bfloat16",
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "num_spec_tokens": args.num_spec_tokens,
            "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
            "used_kwargs": {k: repr(v) for k, v in used_kwargs.items()},
        },
        "total_elapsed_s": total_dt,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["mtp_supported"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
