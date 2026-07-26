#!/usr/bin/env python3
"""Tokenize every fixed serving prompt and reject context-window overflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "kiln.serving-prompt-context-check.v1"


class PromptContextError(RuntimeError):
    pass


def _load_benchmark_module() -> Any:
    path = ROOT / "scripts/bench-concurrent-batch.py"
    spec = importlib.util.spec_from_file_location("kiln_prompt_context_benchmark", path)
    if spec is None or spec.loader is None:
        raise PromptContextError(f"cannot load benchmark module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def parse_csv(raw: str, *, label: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise PromptContextError(f"{label} must contain unique comma-separated values")
    return values


def parse_sizes(raw: str) -> list[int]:
    try:
        values = [int(value) for value in parse_csv(raw, label="sizes")]
    except ValueError as exc:
        raise PromptContextError("sizes must contain decimal integers") from exc
    if values != sorted(values) or any(value <= 0 or value > 4096 for value in values):
        raise PromptContextError("sizes must be increasing integers in 1..=4096")
    return values


def token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if (
        not isinstance(encoded, list)
        or not encoded
        or any(isinstance(value, bool) or not isinstance(value, int) for value in encoded)
    ):
        raise PromptContextError("tokenizer returned malformed input_ids")
    return encoded


def check_prompts(
    *,
    tokenizer: Any,
    benchmark: Any,
    prompt_set_id: str,
    profiles: list[str],
    sizes: list[int],
    repeats: int,
    warmup_requests: int,
    max_tokens: int,
    context_ceiling: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        contract = benchmark.PROFILE_CONTRACTS.get(profile)
        if contract is None:
            raise PromptContextError(f"unsupported workload profile {profile!r}")
        profile_prompt_set_id = f"{prompt_set_id}-{profile}"
        phases: list[tuple[str, int, int]] = []
        if warmup_requests:
            phases.append(
                (
                    f"warmup-c{warmup_requests:03d}",
                    warmup_requests,
                    min(16, max_tokens),
                )
            )
        for size in sizes:
            for repeat in range(repeats):
                phases.append((f"measure-c{size:03d}-r{repeat:03d}", size, max_tokens))
        for phase, concurrency, output_tokens in phases:
            for request_index in range(concurrency):
                prompt = benchmark.deterministic_prompt(
                    profile_prompt_set_id,
                    phase,
                    request_index,
                    contract["prompt_profile"],
                )
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_tokens = len(token_ids(encoded))
                total_tokens = prompt_tokens + output_tokens
                if total_tokens > context_ceiling:
                    raise PromptContextError(
                        f"{profile} {phase} request {request_index} needs "
                        f"{prompt_tokens}+{output_tokens}={total_tokens} tokens, "
                        f"above context ceiling {context_ceiling}"
                    )
                rows.append(
                    {
                        "profile": profile,
                        "phase": phase,
                        "request_index": request_index,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "prompt_sha256": "sha256:"
                        + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    }
                )
    if not rows:
        raise PromptContextError("prompt context check produced no rows")
    max_prompt = max(row["prompt_tokens"] for row in rows)
    max_total = max(row["total_tokens"] for row in rows)
    return {
        "checked_prompt_count": len(rows),
        "max_prompt_tokens": max_prompt,
        "max_total_tokens": max_total,
        "minimum_headroom_tokens": context_ceiling - max_total,
        "profiles": profiles,
        "sizes": sizes,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt-set-id", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--sizes", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--context-ceiling", type=int, required=True)
    parser.add_argument("--expected-max-prompt-tokens", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", args.prompt_set_id) is None:
            raise PromptContextError("prompt-set-id is not a portable identifier")
        if args.repeats <= 0 or args.warmup_requests < 0:
            raise PromptContextError("repeats must be positive and warmup-requests non-negative")
        if args.max_tokens <= 0 or args.context_ceiling <= args.max_tokens:
            raise PromptContextError("context ceiling must exceed positive max-tokens")
        model_path = args.model_path.resolve(strict=True)
        tokenizer_path = model_path / "tokenizer.json"
        template_path = model_path / "chat_template.jinja"
        if any(path.is_symlink() or not path.is_file() for path in (tokenizer_path, template_path)):
            raise PromptContextError("model tokenizer and chat template must be regular files")

        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        benchmark = _load_benchmark_module()
        result = check_prompts(
            tokenizer=tokenizer,
            benchmark=benchmark,
            prompt_set_id=args.prompt_set_id,
            profiles=parse_csv(args.profiles, label="profiles"),
            sizes=parse_sizes(args.sizes),
            repeats=args.repeats,
            warmup_requests=args.warmup_requests,
            max_tokens=args.max_tokens,
            context_ceiling=args.context_ceiling,
        )
        if result["max_prompt_tokens"] != args.expected_max_prompt_tokens:
            raise PromptContextError(
                "maximum prompt token count drifted: "
                f"{result['max_prompt_tokens']}, expected {args.expected_max_prompt_tokens}"
            )
        record = {
            "schema": SCHEMA,
            "verdict": "passed",
            "driver_version": benchmark.DRIVER_VERSION,
            "prompt_template_version": benchmark.PROMPT_TEMPLATE_VERSION,
            "context_ceiling_tokens": args.context_ceiling,
            "max_tokens": args.max_tokens,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "chat_template_sha256": sha256_file(template_path),
            "transformers_version": transformers.__version__,
            **result,
        }
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, PromptContextError, ValueError) as exc:
        print(f"prompt context check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
