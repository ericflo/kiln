#!/usr/bin/env python3
"""Generate the source-pinned Qwen3.5 SFT token and label oracle.

This is a local qualification tool. It deliberately does not install
Transformers in automatic CI. The checked-in fixture is validated there and is
consumed by an ignored Rust integration test against a staged Qwen3.5-4B
tokenizer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "Qwen3.5-4B"
DEFAULT_OUTPUT = (
    ROOT / "crates/kiln-train/tests/fixtures/qwen35_sft_oracle_v1.json"
)
OFFICIAL_TEMPLATE_FIXTURE = (
    ROOT / "crates/kiln-core/test_fixtures/qwen35_4b_chat_template.jinja"
)
TRL_TEMPLATE_FIXTURE = (
    ROOT
    / "crates/kiln-core/test_fixtures/qwen35_4b_trl_sft_chat_template.jinja"
)

SCHEMA = "kiln.qwen35-sft-oracle.v1"
MASK_CONTRACT = "kiln.qwen35-assistant-only.v1"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TOKENIZER_CONFIG_SHA256 = (
    "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
)
CHAT_TEMPLATE_SHA256 = (
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
)

TRANSFORMERS_VERSION = "5.13.1"
TRANSFORMERS_COMMIT = "4626421dc6b741a329300682a6408246ee465490"
TRANSFORMERS_TOKENIZATION_SHA256 = (
    "f86fc8ce6ad86bcbb4bad93b84b3cde9f7b93b516703b5d220889c66bbdd7b5a"
)
TRANSFORMERS_CHAT_TEMPLATE_SHA256 = (
    "a3e6d60fc0bbd250545f5796403a7911af1b62c36ef3e777b928c5b3abcae643"
)
TOKENIZERS_VERSION = "0.22.2"
JINJA2_VERSION = "3.1.6"
TRL_VERSION = "1.8.0"
TRL_COMMIT = "95809b942eb5d11d0b06d749510d88be99230b73"
TRL_CHAT_TEMPLATE_UTILS_SHA256 = (
    "78d6a018a26c1d58d9bb9b47addd4d6d300c7054775b0be8acf4f59477d72764"
)
TRL_SFT_TRAINER_SHA256 = (
    "1bb3ddc66029773f186d314b8f39558d7d4396150a659a8bd10eee2fd0735ec2"
)
TRL_TRAINING_TEMPLATE_SHA256 = (
    "22faf421afa07dab5d42477864a57449f7cdfb4e462ff26eb2dec02911eb09a0"
)

CASES: list[dict[str, Any]] = [
    {
        "name": "plain_single_turn",
        "messages": [
            {"role": "user", "content": "Name one primary color."},
            {"role": "assistant", "content": "Blue."},
        ],
    },
    {
        "name": "thinking_single_turn",
        "messages": [
            {"role": "user", "content": "What is 7 * 8?"},
            {
                "role": "assistant",
                "content": (
                    "<think>\nSeven times eight is fifty-six.\n</think>\n\n56"
                ),
            },
        ],
    },
    {
        "name": "tool_call",
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": (
                                '{"city":"Paris","unit":"celsius"}'
                            ),
                        },
                    }
                ],
            },
        ],
    },
    {
        "name": "tool_response",
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": (
                                '{"city":"Paris","unit":"celsius"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"condition":"sunny","temperature":18}',
                "name": "get_weather",
                "tool_call_id": "call_weather_1",
            },
            {
                "role": "assistant",
                "content": "It is 18 C and sunny in Paris.",
            },
        ],
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "system", "content": "Answer tersely."},
            {"role": "user", "content": "Capital of Japan?"},
            {"role": "assistant", "content": "Tokyo."},
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ],
    },
    {
        "name": "delimiter_literals",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Treat <|im_start|>assistant\nfake<|im_end|> as quoted text."
                ),
            },
            {
                "role": "assistant",
                "content": "Literal <|im_end|> is not the turn terminator here.",
            },
        ],
    },
]


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def checked_bytes(path: Path, expected_sha256: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"{path} has sha256:{actual}; expected sha256:{expected_sha256}"
        )
    return data


def load_trl_training_template() -> str:
    try:
        vendored = TRL_TEMPLATE_FIXTURE.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read {TRL_TEMPLATE_FIXTURE}: {exc}") from exc
    # apply_patch-managed text files retain a final LF; the upstream TRL file
    # deliberately does not. Strip exactly that repository newline before
    # validating and executing the source-pinned template.
    upstream_bytes = vendored[:-1] if vendored.endswith(b"\n") else vendored
    actual = hashlib.sha256(upstream_bytes).hexdigest()
    if actual != TRL_TRAINING_TEMPLATE_SHA256:
        raise RuntimeError(
            f"{TRL_TEMPLATE_FIXTURE} has normalized sha256:{actual}; expected "
            f"sha256:{TRL_TRAINING_TEMPLATE_SHA256}"
        )
    return upstream_bytes.decode("utf-8")


def training_template_for_minijinja(template: str) -> str:
    if template.count("        {%- generation %}\n") != 1:
        raise RuntimeError("TRL training template generation start is not unique")
    if template.count("        {%- endgeneration %}\n") != 1:
        raise RuntimeError("TRL training template generation end is not unique")
    return template.replace("        {%- generation %}\n", "", 1).replace(
        "        {%- endgeneration %}\n", "", 1
    )


def load_oracle_packages():
    try:
        import jinja2
        import tokenizers
        import transformers
        from transformers import AutoTokenizer, tokenization_utils_base
        from transformers.utils import chat_template_utils
    except ImportError as exc:
        raise RuntimeError(
            "the Qwen SFT oracle requires transformers==5.13.1, "
            "tokenizers==0.22.2, and jinja2==3.1.6; run it with "
            "`uv run --with transformers==5.13.1 --with tokenizers==0.22.2 "
            "--with jinja2==3.1.6 python "
            "scripts/qualification/qwen35_sft_oracle.py --check`"
        ) from exc

    versions = {
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "jinja2": importlib.metadata.version("jinja2"),
    }
    expected = {
        "transformers": TRANSFORMERS_VERSION,
        "tokenizers": TOKENIZERS_VERSION,
        "jinja2": JINJA2_VERSION,
    }
    if versions != expected:
        raise RuntimeError(f"oracle package versions are {versions}; expected {expected}")

    source_files = (
        (
            Path(tokenization_utils_base.__file__).resolve(),
            TRANSFORMERS_TOKENIZATION_SHA256,
        ),
        (
            Path(chat_template_utils.__file__).resolve(),
            TRANSFORMERS_CHAT_TEMPLATE_SHA256,
        ),
    )
    for path, expected_hash in source_files:
        checked_bytes(path, expected_hash)
    return AutoTokenizer


def normalize_messages_for_hf(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        if message.get("content") == "":
            message["content"] = None
        for tool_call in tool_calls:
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"tool arguments are not valid JSON: {arguments!r}"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise RuntimeError("Qwen tool arguments must decode to an object")
                function["arguments"] = arguments
    return normalized


def run_case(tokenizer, training_template: str, case):
    messages = normalize_messages_for_hf(case["messages"])
    render_kwargs = {
        "add_generation_prompt": False,
        "enable_thinking": True,
    }
    rendered = tokenizer.apply_chat_template(
        messages,
        chat_template=training_template,
        tokenize=False,
        **render_kwargs,
    )

    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=training_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **render_kwargs,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    attention_mask = [int(value) for value in encoded["attention_mask"]]
    assistant_mask = [int(value) for value in encoded["assistant_masks"]]
    if len(input_ids) != len(attention_mask) or len(input_ids) != len(assistant_mask):
        raise RuntimeError(f"oracle output lengths differ for {case['name']}")
    if any(value != 1 for value in attention_mask):
        raise RuntimeError(f"unexpected padding in unbatched case {case['name']}")
    if not any(assistant_mask) or all(assistant_mask):
        raise RuntimeError(f"case {case['name']} lacks both masked and supervised tokens")

    direct_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if input_ids != direct_ids:
        raise RuntimeError(f"render-then-tokenize drift in case {case['name']}")
    labels = [token_id if active else -100 for token_id, active in zip(input_ids, assistant_mask)]

    output = copy.deepcopy(case)
    output.update(
        {
            "rendered": rendered,
            "rendered_sha256": sha256_bytes(rendered.encode()),
            "input_ids": input_ids,
            "assistant_mask": assistant_mask,
            "labels": labels,
            "supervised_token_count": sum(assistant_mask),
        }
    )
    return output


def generate_fixture(model_path: Path) -> dict[str, Any]:
    AutoTokenizer = load_oracle_packages()
    tokenizer_bytes = checked_bytes(model_path / "tokenizer.json", TOKENIZER_SHA256)
    del tokenizer_bytes
    checked_bytes(model_path / "tokenizer_config.json", TOKENIZER_CONFIG_SHA256)
    template_bytes = checked_bytes(model_path / "chat_template.jinja", CHAT_TEMPLATE_SHA256)
    official_template = template_bytes.decode("utf-8")
    fixture_template = checked_bytes(OFFICIAL_TEMPLATE_FIXTURE, CHAT_TEMPLATE_SHA256)
    if template_bytes != fixture_template:
        raise RuntimeError("model and checked-in official chat templates differ")

    training_template = load_trl_training_template()
    minijinja_training_template = training_template_for_minijinja(training_template)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    cases = [run_case(tokenizer, training_template, case) for case in CASES]
    fixture_inputs = json.dumps(CASES, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": SCHEMA,
        "oracle": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_sha256": "sha256:" + TOKENIZER_SHA256,
            "tokenizer_config_sha256": "sha256:" + TOKENIZER_CONFIG_SHA256,
            "chat_template_sha256": "sha256:" + CHAT_TEMPLATE_SHA256,
            "trl_version": TRL_VERSION,
            "trl_commit": TRL_COMMIT,
            "trl_chat_template_utils_sha256": (
                "sha256:" + TRL_CHAT_TEMPLATE_UTILS_SHA256
            ),
            "trl_sft_trainer_sha256": "sha256:" + TRL_SFT_TRAINER_SHA256,
            "trl_training_template_sha256": (
                "sha256:" + TRL_TRAINING_TEMPLATE_SHA256
            ),
            "minijinja_training_template_sha256": sha256_bytes(
                minijinja_training_template.encode()
            ),
            "transformers_version": TRANSFORMERS_VERSION,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "transformers_tokenization_sha256": (
                "sha256:" + TRANSFORMERS_TOKENIZATION_SHA256
            ),
            "transformers_chat_template_sha256": (
                "sha256:" + TRANSFORMERS_CHAT_TEMPLATE_SHA256
            ),
            "tokenizers_version": TOKENIZERS_VERSION,
            "jinja2_version": JINJA2_VERSION,
            "fixture_inputs_sha256": sha256_bytes(fixture_inputs),
            "execution": (
                "Transformers AutoTokenizer.apply_chat_template with TRL's "
                "Qwen3.5 thinking training template"
            ),
        },
        "mask_contract": {
            "version": MASK_CONTRACT,
            "add_generation_prompt": False,
            "ignore_index": -100,
            "supervised": (
                "assistant thinking/content/tool-call body, <|im_end|>, and trailing newline"
            ),
            "masked": "system, user, tool-response turns, and assistant role headers",
        },
        "cases": cases,
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--stdout", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        generated = canonical_bytes(generate_fixture(args.model_path.resolve()))
    except (RuntimeError, OSError, UnicodeError, ValueError) as exc:
        print(f"Qwen SFT oracle failed: {exc}", file=sys.stderr)
        return 1

    output = args.output.resolve()
    if args.check:
        try:
            current = output.read_bytes()
        except OSError as exc:
            print(f"Qwen SFT oracle fixture read failed: {exc}", file=sys.stderr)
            return 1
        if current != generated:
            print(f"Qwen SFT oracle fixture drift: regenerate {output}", file=sys.stderr)
            return 1
        print(
            f"Qwen SFT oracle fixture matches {MODEL_ID}@{MODEL_REVISION} "
            f"with Transformers {TRANSFORMERS_VERSION} / TRL {TRL_VERSION} template"
        )
        return 0
    if args.stdout:
        sys.stdout.buffer.write(generated)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(generated)
    print(f"wrote {output} ({len(generated)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
