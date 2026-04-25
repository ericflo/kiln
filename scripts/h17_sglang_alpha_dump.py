#!/usr/bin/env python3
"""H17 — SGLang MTP α microbench on Qwen3.5-4B A6000 bs=1.

Runs the *same* workload as PR #529's kiln c1_attr capture and PR #530's
vLLM driver:

    - 3 prose GSM8K prompts (PROMPT_POOL indices 0,1,2) re-built from the
      hand-rolled kiln-bench strings and expanded to ~512 prefill tokens
      by sentence repetition (matching kiln-bench's `build_prompt`).
    - Greedy decode (temperature=0, top_p=1, top_k=-1).
    - max_tokens=16 (matches PR #529 --max-output-tokens 16).
    - Seeds 0, 1, 2.
    - Speculative decoding: Qwen3_5ForCausalLMMTP native MTP, k=1.

Measures α = sum(accepted_tokens) / sum(proposed_tokens) per request, then
aggregates per seed and reports the cross-seed median for the H17 compare.

SGLang's speculative-decoding stats API surfaces accepted / proposed via the
engine's server metrics or per-request stats depending on version. We collect
α with parallel strategies and take the first one that yields a non-empty
signal:

    1. The GenerateReqOutput.meta_info spec_verify_ct / completion_tokens
       pair (MTP accept bookkeeping).
    2. The /metrics Prometheus text endpoint
       (sglang:spec_verify_ct / sglang:num_spec_tokens_accepted counters).
    3. Engine state after `flush_cache()` via get_server_info.

If SGLang refuses to load Qwen3.5-4B dense + MTP (e.g. arch dispatch
mismatch, missing weights, or SGLang's dense 4B + MTP path end-to-end
failure noted in H16 audit caveat 2), the driver emits
`{"mtp_supported": false, "reason": ...}` so the compare step can post a
doc-only redirect PR without burning more pod time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path


# Exact prose prompts from crates/kiln-server/src/bench.rs PROMPT_POOL[0,1,2]
# (GSM8K-style grade-school math word problems). Matching these character-for-
# character is required for valid kiln-vs-SGLang α comparison — seed N picks
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


def try_load_engine(model_path: str, num_spec_tokens: int):
    """Load SGLang Engine with Qwen3.5 MTP speculative config.

    Tries multiple config shapes across SGLang versions (the API moved between
    0.4 and 0.5 for speculative-decoding kwargs). Returns (engine, used_kwargs)
    or raises RuntimeError.
    """
    import sglang as sgl  # noqa: WPS433  import here so caller can catch ImportError

    errors = []
    # SGLang 0.5.10's SpeculativeAlgorithm enum at
    # python/sglang/srt/speculative/spec_info.py accepts exactly:
    #   EAGLE, EAGLE3, STANDALONE, NGRAM, NONE
    # with NEXTN -> EAGLE rewrite in server_args.py:2983. For Qwen3.5 native
    # MTP the model's pretrained mtp.* head IS the drafter, loaded via the
    # Qwen3_5ForCausalLMMTP arch — no separate draft_model_path. Pass
    # speculative_eagle_topk=num_spec_tokens to satisfy the int>None compare.
    attempts = [
        # EAGLE is the modern rename of MTP / NEXTN in SGLang
        dict(
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=num_spec_tokens,
            speculative_num_steps=num_spec_tokens,
            speculative_eagle_topk=num_spec_tokens,
        ),
        # Legacy alias: NEXTN auto-rewrites to EAGLE per server_args.py:2983
        dict(
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=num_spec_tokens,
            speculative_num_steps=num_spec_tokens,
            speculative_eagle_topk=num_spec_tokens,
        ),
    ]
    # Dense 4B + MTP: Qwen3_5ForCausalLMMTP arch (per H16 audit). We pass
    # `dtype="bfloat16"` matching vLLM confounder acknowledgment. Spec V2 is
    # required per SGLang 0.5.10 for Qwen3.5 + MTP + radix cache — enabled via
    # SGLANG_ENABLE_SPEC_V2=1 env var AND mamba_scheduler_strategy='extra_buffer'.
    os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
    base = dict(
        model_path=model_path,
        dtype="bfloat16",
        mem_fraction_static=0.85,
        context_length=1024,
        log_level="info",
        disable_cuda_graph=True,
        trust_remote_code=True,
        mamba_scheduler_strategy="extra_buffer",
        attention_backend="triton",  # Flashinfer crashed in eagle_info.generate_attn_arg_prefill — route around it
    )

    for kwargs in attempts:
        try:
            engine = sgl.Engine(**base, **kwargs)
            return engine, {**{"_base": "trust_remote_code+bf16"}, **kwargs}
        except (TypeError, ValueError) as e:
            msg = str(e)
            # Older SGLang may not accept one of our base kwargs — strip and retry
            drop_candidates = (
                "disable_cuda_graph",
                "trust_remote_code",
                "speculative_num_steps",
                "speculative_draft_model_path",
            )
            dropped = False
            for cand in drop_candidates:
                if cand in msg:
                    if cand in base:
                        base = {k: v for k, v in base.items() if k != cand}
                    if cand in kwargs:
                        kwargs = {k: v for k, v in kwargs.items() if k != cand}
                    dropped = True
                    break
            if dropped:
                try:
                    engine = sgl.Engine(**base, **kwargs)
                    return engine, {**{"_base": "bf16-kwargs-trimmed"}, **kwargs}
                except Exception as e2:
                    errors.append(
                        f"kwargs={kwargs!r} (after dropping): {type(e2).__name__}: {e2}"
                    )
                    continue
            errors.append(f"kwargs={kwargs!r}: {type(e).__name__}: {e}")
        except Exception as e:
            errors.append(f"kwargs={kwargs!r}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "SGLang failed to load Qwen3.5-4B with any MTP speculative config; tried:\n  "
        + "\n  ".join(errors)
    )


def extract_alpha_from_meta(meta_info: dict, num_spec_tokens: int) -> dict | None:
    """Pull accepted/proposed from SGLang GenerateReqOutput.meta_info.

    SGLang's meta_info for a speculative-decode request typically contains:
      spec_verify_ct : int  — number of draft-token verify rounds
      completion_tokens : int — total output tokens (target + accepted drafts)
    From these we back out:
      proposed = spec_verify_ct * num_spec_tokens
      accepted = (completion_tokens - spec_verify_ct)
      ... BUT the exact bookkeeping varies; record both raw fields and a
      best-effort α so the compare step has something to work with.
    """
    if not isinstance(meta_info, dict):
        return None
    spec_verify_ct = meta_info.get("spec_verify_ct")
    completion_tokens = meta_info.get("completion_tokens")
    if spec_verify_ct is None or completion_tokens is None:
        return None
    # Alternative SGLang field name: num_spec_accepted_tokens
    accepted_field = meta_info.get("num_spec_accepted_tokens")
    if accepted_field is not None:
        # SGLang exposes accept count directly — use it.
        accepted = int(accepted_field)
        proposed = int(spec_verify_ct) * int(num_spec_tokens)
    else:
        # Back out accepted from completion_tokens - spec_verify_ct:
        # per SGLang V1 scheduler: each verify round produces 1 target token +
        # 0..k accepted drafts. So completion = verify_ct * 1 + sum_accepted.
        accepted = max(0, int(completion_tokens) - int(spec_verify_ct))
        proposed = int(spec_verify_ct) * int(num_spec_tokens)
    if proposed == 0:
        return None
    return {
        "accepted": accepted,
        "proposed": proposed,
        "source": "meta_info.spec_verify_ct + completion_tokens",
        "detail": {
            "spec_verify_ct": int(spec_verify_ct),
            "completion_tokens": int(completion_tokens),
            "num_spec_accepted_tokens": accepted_field,
        },
    }


def extract_alpha_from_metrics(engine) -> dict | None:
    """Scrape the engine's /metrics endpoint for spec-decode counters.

    Returns {accepted, proposed, source, detail} or None if not available.
    """
    # Try a few known SGLang metric field paths — the exact attribute varies
    # across versions. Start with sglang.Engine.get_server_info().
    try:
        info = engine.get_server_info() if hasattr(engine, "get_server_info") else None
    except Exception:
        info = None
    if isinstance(info, dict):
        a = info.get("num_spec_tokens_accepted") or info.get("spec_decode_accepted_tokens")
        p = info.get("num_spec_tokens_total") or info.get("spec_decode_proposed_tokens")
        if a is not None and p is not None and int(p) > 0:
            return {
                "accepted": int(a),
                "proposed": int(p),
                "source": "engine.get_server_info()",
                "detail": {"keys": sorted(info.keys())[:20]},
            }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/qwen3.5-4b")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--num-spec-tokens", type=int, default=1)
    ap.add_argument(
        "--out",
        default="docs/phase-c29-v3-sglang/sglang_alpha_per_seed.json",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build prompts first — needs tokenizer only, no GPU.
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

    print(f"[h17_sglang] loading tokenizer from {args.model_path}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [build_prompt(tok, args.prompt_tokens, s) for s in args.seeds]
    for i, p in enumerate(prompts):
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        print(
            f"[h17_sglang] seed={args.seeds[i]} prompt_tokens={n} "
            f"prompt_prefix={p[:80]!r}",
            file=sys.stderr,
        )

    # Load SGLang Engine with MTP spec decoding
    engine = None
    used_kwargs = {}
    try:
        engine, used_kwargs = try_load_engine(args.model_path, args.num_spec_tokens)
    except (ImportError, RuntimeError) as e:
        failure = {
            "mtp_supported": False,
            "reason": (
                f"SGLang failed to load Qwen3.5-4B with MTP speculative config: {e}"
            ),
            "traceback": traceback.format_exc(limit=5),
        }
        out_path.write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 3

    print(f"[h17_sglang] SGLang loaded with kwargs={used_kwargs!r}", file=sys.stderr)

    per_seed = []
    alphas = []
    total_start = time.perf_counter()

    for seed, prompt in zip(args.seeds, prompts):
        sampling = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": args.max_tokens,
        }
        t0 = time.perf_counter()
        # SGLang Engine.generate is sync by default; returns dict with text+meta_info
        try:
            result = engine.generate(prompt=prompt, sampling_params=sampling)
        except Exception as e:
            per_seed.append({
                "seed": seed,
                "alpha": 0.0,
                "n_accepted": 0,
                "n_proposed": 0,
                "elapsed_s": time.perf_counter() - t0,
                "stats_source": "GENERATE_FAILED",
                "error": f"{type(e).__name__}: {e}",
            })
            alphas.append(0.0)
            print(
                f"[h17_sglang] seed={seed} GENERATE FAILED: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue
        dt = time.perf_counter() - t0

        meta = result.get("meta_info") if isinstance(result, dict) else None
        text = result.get("text", "") if isinstance(result, dict) else str(result)

        stats = extract_alpha_from_meta(meta, args.num_spec_tokens)
        if stats is None:
            stats = extract_alpha_from_metrics(engine)
        if stats is None:
            stats = {
                "accepted": 0,
                "proposed": 0,
                "source": "UNAVAILABLE",
                "detail": {
                    "meta_info_keys": sorted(meta.keys())[:30]
                    if isinstance(meta, dict) else None,
                    "meta_info_repr": repr(meta)[:300],
                },
            }

        alpha = (
            stats["accepted"] / stats["proposed"]
            if stats["proposed"] > 0
            else 0.0
        )
        per_seed.append({
            "seed": seed,
            "alpha": alpha,
            "n_accepted": stats["accepted"],
            "n_proposed": stats["proposed"],
            "elapsed_s": dt,
            "stats_source": stats.get("source", "UNAVAILABLE"),
            "stats_detail": stats.get("detail"),
            "generated_head": text[:120] if isinstance(text, str) else repr(text)[:120],
        })
        alphas.append(alpha)
        print(
            f"[h17_sglang] seed={seed} α={alpha:.4f} "
            f"accepted={stats['accepted']} proposed={stats['proposed']} "
            f"source={stats.get('source')} dt={dt:.2f}s",
            file=sys.stderr,
        )

    total_dt = time.perf_counter() - total_start

    try:
        engine.shutdown() if hasattr(engine, "shutdown") else None
    except Exception:
        pass

    import statistics
    result = {
        "mtp_supported": any(s["n_proposed"] > 0 for s in per_seed),
        "per_seed": per_seed,
        "median_alpha": statistics.median(alphas) if alphas else 0.0,
        "mean_alpha": statistics.mean(alphas) if alphas else 0.0,
        "min_alpha": min(alphas) if alphas else 0.0,
        "max_alpha": max(alphas) if alphas else 0.0,
        "n_seeds": len(alphas),
        "sglang_config": {
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
    try:
        import sglang as sgl_mod  # type: ignore
        result["sglang_version"] = getattr(sgl_mod, "__version__", "unknown")
    except Exception:
        result["sglang_version"] = "unknown"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["mtp_supported"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
