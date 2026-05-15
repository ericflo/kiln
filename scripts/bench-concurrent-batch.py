#!/usr/bin/env python3
"""Batched throughput sweep for kiln /v1/completions/batch and concurrent HTTP."""
import argparse
import json
import statistics
import time
import urllib.request
import urllib.error
import threading
import sys


PROMPT = (
    "Write a short paragraph that describes the city of San Francisco, "
    "California, including its history, geography, climate, and culture. "
    "Be detailed and informative. Begin your answer with 'San Francisco'."
)


def post(url, body, timeout=600):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def bench_batch_endpoint(host, batch_size, n_per_prompt, max_tokens):
    """Use /v1/completions/batch with `batch_size` prompts × n_per_prompt completions."""
    url = f"{host}/v1/completions/batch"
    body = {
        "prompts": [
            [{"role": "user", "content": PROMPT + f" Variant {i:03d}."}]
            for i in range(batch_size)
        ],
        "n": n_per_prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }
    t0 = time.perf_counter()
    resp = post(url, body)
    elapsed = time.perf_counter() - t0
    total_completion_tokens = 0
    total_prompt_tokens = 0
    for choice in resp.get("choices", []):
        usage = choice.get("usage") or resp.get("usage") or {}
        total_completion_tokens += usage.get("completion_tokens", 0)
        total_prompt_tokens += usage.get("prompt_tokens", 0)
    if total_completion_tokens == 0:
        usage = resp.get("usage") or {}
        total_completion_tokens = usage.get("completion_tokens", 0)
        total_prompt_tokens = usage.get("prompt_tokens", 0)
    return {
        "batch_size": batch_size,
        "n_per_prompt": n_per_prompt,
        "elapsed_s": elapsed,
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "tokens_per_s": total_completion_tokens / elapsed if elapsed > 0 else 0,
    }


def bench_concurrent(host, num_concurrent, max_tokens, prompt_idx_seed=0):
    """Fire N parallel /v1/chat/completions requests."""
    url = f"{host}/v1/chat/completions"
    results = [None] * num_concurrent
    errors = [None] * num_concurrent

    def worker(i):
        # Same-length distinguishing suffix so positions_uniform=True but the
        # deterministic cache does not collapse identical greedy requests.
        body = {
            "messages": [
                {"role": "user", "content": PROMPT + f" Variant {i:03d}."}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": prompt_idx_seed + i,
            "stream": False,
        }
        try:
            t0 = time.perf_counter()
            resp = post(url, body)
            results[i] = {
                "elapsed_s": time.perf_counter() - t0,
                "completion_tokens": resp["usage"]["completion_tokens"],
                "prompt_tokens": resp["usage"]["prompt_tokens"],
            }
        except Exception as e:
            errors[i] = str(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_concurrent)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    successes = [r for r in results if r is not None]
    total_completion_tokens = sum(r["completion_tokens"] for r in successes)
    return {
        "num_concurrent": num_concurrent,
        "elapsed_s": elapsed,
        "successes": len(successes),
        "errors": [e for e in errors if e],
        "total_completion_tokens": total_completion_tokens,
        "tokens_per_s": total_completion_tokens / elapsed if elapsed > 0 else 0,
        "per_req_tokens_per_s": [
            r["completion_tokens"] / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
            for r in successes
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8420")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--mode",
        choices=["batch", "concurrent", "both"],
        default="concurrent",
        help="batch=/v1/completions/batch, concurrent=N parallel /v1/chat/completions",
    )
    ap.add_argument(
        "--sizes",
        default="1,2,4,8,16",
        help="Comma-separated batch sizes to sweep",
    )
    ap.add_argument("--warmup", action="store_true", help="Do a warmup pass first")
    ap.add_argument("--out", default=None, help="Optional JSON output")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    if args.warmup:
        print("warmup ...", flush=True)
        bench_concurrent(args.host, 1, max(8, args.max_tokens // 4))

    all_results = []
    for sz in sizes:
        if args.mode in ("concurrent", "both"):
            r = bench_concurrent(args.host, sz, args.max_tokens)
            r["mode"] = "concurrent"
            print(
                f"[concurrent] n={sz:>2} elapsed={r['elapsed_s']:.3f}s "
                f"tokens={r['total_completion_tokens']:>5} "
                f"tok/s={r['tokens_per_s']:.2f} "
                f"successes={r['successes']}/{sz} "
                f"errors={len(r['errors'])}",
                flush=True,
            )
            if r["errors"]:
                for e in r["errors"][:3]:
                    print(f"  error: {e}", flush=True)
            all_results.append(r)
        if args.mode in ("batch", "both"):
            r = bench_batch_endpoint(args.host, sz, 1, args.max_tokens)
            r["mode"] = "batch"
            print(
                f"[batch_ep ] n={sz:>2} elapsed={r['elapsed_s']:.3f}s "
                f"tokens={r['completion_tokens']:>5} "
                f"tok/s={r['tokens_per_s']:.2f}",
                flush=True,
            )
            all_results.append(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
