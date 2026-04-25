#!/usr/bin/env python3
"""H15c — vLLM MTP α microbench on Qwen3.5-4B A6000 bs=1.

Runs the *same* workload as PR #529's kiln c1_attr capture:

    - 3 prose GSM8K prompts (PROMPT_POOL indices 0,1,2) re-built from the
      hand-rolled kiln-bench strings and expanded to ~512 prefill tokens
      by sentence repetition (matching kiln-bench's `build_prompt`).
    - Greedy decode (temperature=0, top_p=1, top_k=-1).
    - max_tokens=16 (matches PR #529 --max-output-tokens 16).
    - Seeds 0, 1, 2.
    - Speculative decoding: Qwen3-Next native MTP, k=1.

Measures α = sum(accepted_tokens) / sum(proposed_tokens) per request, then
aggregates per seed and reports the cross-seed median for the H15c compare.

The vLLM speculative-decoding stats API has varied across versions, so we
collect α with three parallel strategies and take the first one that yields
a non-empty signal:

    1. RequestOutput.metrics / spec fields on the `outputs[i].metrics`.
    2. The v1 SpecDecodingStats counters on `llm.llm_engine.get_stats()`.
    3. Re-running with `disable_log_stats=False` + the `--dump-stats` side
       channel (vLLM writes `num_accepted_tokens` / `num_draft_tokens` to
       the logging stat loggers).

If vLLM refuses to load Qwen3.5-4B with a Qwen3-Next MTP speculative config
(e.g. arch dispatch mismatch, missing weights, or unsupported combo), the
driver emits `{"mtp_supported": false, "reason": ...}` so the compare step
can post a doc-only redirect PR without burning more pod time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


# Exact prose prompts from crates/kiln-server/src/bench.rs PROMPT_POOL[0,1,2]
# (GSM8K-style grade-school math word problems). Matching these character-for-
# character is required for valid kiln-vs-vLLM α comparison — seed N picks
# PROMPT_POOL[seed % 30] in kiln, so seeds 0/1/2 pick prompts 0/1/2.
BASE_PROMPTS = [
    # 0: eggs-per-day revenue
    "Janet's ducks lay sixteen eggs per day. She eats three for breakfast every morning and bakes muffins for her friends with four more. She sells the remainder at the local farmers' market daily for two dollars per fresh duck egg. We want to know how much she makes every day at the farmers' market. First subtract eaten and baked eggs from the daily lay, then multiply the leftover count by the per-egg price. ",
    # 1: robe bolts
    "A robe takes two bolts of blue fiber and half that much white fiber. The bolts are purchased separately from two different mills, each with its own shipping schedule. We need the total number of bolts required to make a single robe for one customer at the shop. Half of two bolts is one bolt of white fiber, and that amount is added to the original two bolts of blue. ",
    # 2: house flip profit
    "Josh buys a run-down property for eighty thousand dollars and then spends fifty thousand more on repairs. The renovation increases the value of the house by one hundred and fifty percent over the original purchase price. We want the profit after selling at the appreciated market price. Compute the new value using the percentage increase, then subtract the purchase price and the repair cost to find the net profit. ",
]


def build_prompt(tokenizer, target_tokens: int, seed: int) -> str:
    """Replicates kiln-bench's build_prompt for PROMPT_POOL[seed%30]."""
    base = BASE_PROMPTS[seed % len(BASE_PROMPTS)]
    prompt = ""
    while True:
        prompt += base
        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(tokens) >= target_tokens:
            # Trim by cutting at ". " boundaries, matching kiln-bench
            while len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > target_tokens:
                pos = prompt.rfind(". ")
                if pos == -1:
                    break
                prompt = prompt[: pos + 1]
            return prompt


def try_load_llm(model_path: str, num_spec_tokens: int):
    """Load vLLM with Qwen3-Next MTP speculative config.

    Tries multiple config shapes across vLLM versions (the API moved from
    SpeculativeConfig-dict to speculative_config-dict between 0.6 and 0.7).
    Returns the `LLM` or raises RuntimeError.
    """
    from vllm import LLM  # noqa: WPS433  import here so try_load_llm's caller can catch ImportError

    errors = []
    attempts = [
        # vLLM >= 0.18 native Qwen3.5 MTP head (Qwen3_5MTP arch in registry)
        dict(
            speculative_config={
                "method": "qwen3_5_mtp",
                "num_speculative_tokens": num_spec_tokens,
            },
        ),
        # Auto-dispatch: vllm.config.speculative rewrites model_type "qwen3_5"
        # → "qwen3_5_mtp" when method is "mtp" (lines 323-330 in 0.19.1)
        dict(
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": num_spec_tokens,
            },
        ),
        # Fallback: Qwen3-Next MTP method name (older API alias)
        dict(
            speculative_config={
                "method": "qwen3_next_mtp",
                "num_speculative_tokens": num_spec_tokens,
            },
        ),
    ]
    # Qwen3.5-4B is `Qwen3_5ForConditionalGeneration` (multimodal: text + vision).
    # vLLM 0.19.1 segfaults during encoder cache profile_run when MTP speculative
    # decoding is combined with the default image/video MM profiling step (the
    # crash signature is "!!!!!!! Segfault encountered !!!!!!!" right after
    # "Encoder cache will be initialized with a budget of 16384 tokens..."). We
    # work around it by:
    #   (1) limit_mm_per_prompt={"image": 0, "video": 0} to skip MM dummy profile
    #   (2) enforce_eager=True to bypass CUDA graph capture path that is the
    #       most common segfault source for this MTP-on-multimodal combo
    base = dict(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        disable_log_stats=False,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        skip_mm_profiling=True,
    )

    for kwargs in attempts:
        try:
            llm = LLM(**base, **kwargs)
            return llm, {**{"_base": "enforce_eager+no-mm"}, **kwargs}
        except (TypeError, ValueError) as e:
            # Older vLLM may not accept skip_mm_profiling — drop it and retry
            if "skip_mm_profiling" in str(e):
                base2 = {k: v for k, v in base.items() if k != "skip_mm_profiling"}
                try:
                    llm = LLM(**base2, **kwargs)
                    return llm, {**{"_base": "enforce_eager+no-mm-no-skipprofile"}, **kwargs}
                except Exception as e2:
                    errors.append(f"kwargs={kwargs!r}: {type(e2).__name__}: {e2}")
                    continue
            errors.append(f"kwargs={kwargs!r}: {type(e).__name__}: {e}")
        except Exception as e:
            errors.append(f"kwargs={kwargs!r}: {type(e).__name__}: {e}")
    raise RuntimeError("vLLM failed to load with any MTP speculative config; tried:\n  " + "\n  ".join(errors))


def extract_alpha(outputs, llm) -> dict:
    """Pull accepted/proposed counts from whatever vLLM version is installed.

    Returns {accepted: int, proposed: int, source: str, detail: dict} or
    raises RuntimeError if no stats source is available.
    """
    # Strategy 1: RequestOutput.metrics → spec_decode_* fields
    accepted = 0
    proposed = 0
    detail = {}
    for out in outputs:
        metrics = getattr(out, "metrics", None)
        if metrics is None:
            continue
        a = (
            getattr(metrics, "num_accepted_tokens", None)
            or getattr(metrics, "spec_accepted_tokens", None)
            or getattr(metrics, "accepted_tokens", None)
        )
        p = (
            getattr(metrics, "num_proposal_tokens", None)
            or getattr(metrics, "num_draft_tokens", None)
            or getattr(metrics, "spec_draft_tokens", None)
            or getattr(metrics, "proposal_tokens", None)
        )
        if a is not None and p is not None:
            accepted += int(a)
            proposed += int(p)
    if proposed > 0:
        return {
            "accepted": accepted,
            "proposed": proposed,
            "source": "RequestOutput.metrics",
            "detail": detail,
        }

    # Strategy 2: engine-level stat loggers
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    if engine is not None:
        try:
            stats = engine.get_stats() if hasattr(engine, "get_stats") else None
            if stats is not None:
                a = (
                    getattr(stats, "num_accepted_tokens", None)
                    or getattr(stats, "spec_decoding_stats", {}).get("num_accepted_tokens", None)
                )
                p = (
                    getattr(stats, "num_draft_tokens", None)
                    or getattr(stats, "spec_decoding_stats", {}).get("num_draft_tokens", None)
                )
                if a is not None and p is not None and int(p) > 0:
                    return {
                        "accepted": int(a),
                        "proposed": int(p),
                        "source": "engine.get_stats()",
                        "detail": {"stats_repr": repr(stats)[:200]},
                    }
        except Exception as e:
            detail["get_stats_err"] = f"{type(e).__name__}: {e}"

    # Strategy 3: stat_loggers attribute
    engine = engine or getattr(llm, "llm_engine", None)
    if engine is not None:
        stat_loggers = (
            getattr(engine, "stat_loggers", None)
            or getattr(engine, "_stat_loggers", None)
            or {}
        )
        # Prometheus logger usually has a `spec_decode_metrics` attribute
        for key, logger in (stat_loggers.items() if hasattr(stat_loggers, "items") else []):
            for attr in ("spec_decoding_metrics", "spec_decode_metrics"):
                metrics = getattr(logger, attr, None)
                if metrics is None:
                    continue
                a = getattr(metrics, "num_accepted_tokens", None) or getattr(metrics, "accepted_tokens", None)
                p = getattr(metrics, "num_draft_tokens", None) or getattr(metrics, "proposed_tokens", None)
                if a is not None and p is not None and int(p) > 0:
                    return {
                        "accepted": int(a),
                        "proposed": int(p),
                        "source": f"stat_loggers[{key}].{attr}",
                        "detail": {},
                    }

    raise RuntimeError(
        "could not extract MTP α from vLLM outputs — tried RequestOutput.metrics, "
        "engine.get_stats(), and stat_loggers. Inspect the installed vLLM version "
        "and add a new extractor."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/qwen3.5-4b")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--num-spec-tokens", type=int, default=1)
    ap.add_argument(
        "--out",
        default="docs/phase-c29-v3-vllm/vllm_alpha_per_seed.json",
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
        }
        out_path.write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2))
        return 2

    print(f"[h15c_vllm] loading tokenizer from {args.model_path}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [build_prompt(tok, args.prompt_tokens, s) for s in args.seeds]
    for i, p in enumerate(prompts):
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        print(f"[h15c_vllm] seed={args.seeds[i]} prompt_tokens={n} prompt_prefix={p[:80]!r}", file=sys.stderr)

    # Load vLLM with MTP spec decoding
    try:
        llm, used_kwargs = try_load_llm(args.model_path, args.num_spec_tokens)
    except (ImportError, RuntimeError) as e:
        failure = {
            "mtp_supported": False,
            "reason": f"vLLM failed to load Qwen3.5-4B with MTP speculative config: {e}",
            "traceback": traceback.format_exc(limit=3),
        }
        out_path.write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 3

    print(f"[h15c_vllm] vLLM loaded with kwargs={used_kwargs!r}", file=sys.stderr)

    # Also import SamplingParams, which lives at top-level
    from vllm import SamplingParams

    per_seed = []
    alphas = []
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
            stats = {"accepted": 0, "proposed": 0, "source": "UNAVAILABLE", "error": str(e)}
        alpha = stats["accepted"] / stats["proposed"] if stats["proposed"] > 0 else 0.0
        generated_text = outs[0].outputs[0].text[:120] if outs else ""
        per_seed.append({
            "seed": seed,
            "alpha": alpha,
            "n_accepted": stats["accepted"],
            "n_proposed": stats["proposed"],
            "elapsed_s": dt,
            "stats_source": stats.get("source", "UNAVAILABLE"),
            "generated_head": generated_text,
        })
        alphas.append(alpha)
        print(f"[h15c_vllm] seed={seed} α={alpha:.4f} accepted={stats['accepted']} proposed={stats['proposed']} source={stats.get('source')} dt={dt:.2f}s", file=sys.stderr)

    total_dt = time.perf_counter() - total_start

    import statistics
    result = {
        "mtp_supported": any(s["n_proposed"] > 0 for s in per_seed),
        "per_seed": per_seed,
        "median_alpha": statistics.median(alphas) if alphas else 0.0,
        "mean_alpha": statistics.mean(alphas) if alphas else 0.0,
        "min_alpha": min(alphas) if alphas else 0.0,
        "max_alpha": max(alphas) if alphas else 0.0,
        "n_seeds": len(alphas),
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
