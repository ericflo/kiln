#!/usr/bin/env python3
"""Produce pinned Qwen3.5-4B Hugging Face logits and optional layer rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from hf_next_token_contract import (
    PASS_PREFIX as NEXT_TOKEN_PASS_PREFIX,
    canonical_sha256,
    load_request,
)


SCHEMA = "kiln.qwen35-hf-full-logits.v1"
LAYER_SCHEMA = "kiln.qwen35-hf-layer-last-rows.v1"
LAYER_PASS_PREFIX = "KILN_HF_LAYER_LAST_ROWS_REFERENCE_PASS "
INPUT_TOKEN_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 100]
TORCH_VERSION = "2.13.0"
TORCH_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
TRANSFORMERS_VERSION = "5.13.1"
SAFETENSORS_VERSION = "0.8.0"
MODELING_SHA256 = "cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba"
CONFIGURATION_SHA256 = "3c01b3cdcff8d77cbafac9841bc48c41e5a5b38637231f1bde3d843cd198dbaf"


class OracleError(RuntimeError):
    """The independent reference cannot be produced exactly as declared."""


def _current_cgroup_memory() -> dict[str, int]:
    try:
        cgroup_lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise OracleError(f"cannot read the HF service cgroup: {exc}") from exc
    unified = []
    for line in cgroup_lines:
        hierarchy, separator, remainder = line.partition(":")
        controllers, second_separator, path = remainder.partition(":")
        if separator and second_separator and hierarchy == "0" and controllers == "":
            unified.append(path)
    if len(unified) != 1 or not unified[0].startswith("/"):
        raise OracleError(f"expected one cgroup-v2 path, got {unified!r}")
    root = Path("/sys/fs/cgroup") / unified[0].lstrip("/")

    def read_integer(name: str) -> int:
        try:
            value = (root / name).read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise OracleError(f"cannot read HF cgroup {name}: {exc}") from exc
        if not value.isdigit():
            raise OracleError(f"HF cgroup {name} is not numeric: {value!r}")
        return int(value)

    events: dict[str, int] = {}
    try:
        for line in (root / "memory.events").read_text(encoding="ascii").splitlines():
            name, value = line.split()
            events[name] = int(value)
    except (OSError, UnicodeError, ValueError) as exc:
        raise OracleError(f"cannot read HF cgroup memory.events: {exc}") from exc
    return {
        "memory_high_events": events.get("high", 0),
        "memory_max_events": events.get("max", 0),
        "memory_oom_events": events.get("oom", 0),
        "memory_oom_kill_events": events.get("oom_kill", 0),
        "memory_peak_bytes": read_integer("memory.peak"),
        "memory_swap_bytes": read_integer("memory.swap.current"),
    }


def _source_sha256(module: object) -> str:
    path = Path(getattr(module, "__file__", "")).resolve()
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise OracleError(f"cannot hash installed oracle source {path}: {exc}") from exc


def _load_packages():
    try:
        import safetensors
        import torch
        import transformers
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3_5 import configuration_qwen3_5, modeling_qwen3_5
    except ImportError as exc:
        raise OracleError(
            "the HF full-logit oracle requires the pinned requirements-sft.lock environment"
        ) from exc

    versions = {
        "safetensors": safetensors.__version__,
        "torch": torch.__version__.split("+", 1)[0],
        "transformers": transformers.__version__,
    }
    expected = {
        "safetensors": SAFETENSORS_VERSION,
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
    }
    if versions != expected:
        raise OracleError(f"oracle package versions are {versions}; expected {expected}")
    if torch.version.git_version != TORCH_COMMIT:
        raise OracleError(
            f"expected torch commit {TORCH_COMMIT}, got {torch.version.git_version}"
        )
    for module, expected_hash in (
        (modeling_qwen3_5, MODELING_SHA256),
        (configuration_qwen3_5, CONFIGURATION_SHA256),
    ):
        actual = _source_sha256(module)
        if actual != expected_hash:
            raise OracleError(
                f"installed {Path(module.__file__).name} has sha256:{actual}; "
                f"expected sha256:{expected_hash}"
            )
    if modeling_qwen3_5.is_fast_path_available:
        raise OracleError(
            "the oracle requires Transformers' pinned independent torch fallback, "
            "but optional fused linear-attention packages are active"
        )
    return torch, AutoModelForCausalLM, AutoTokenizer, save_file


def _wait_for_start_gate(path: Path | None, timeout_seconds: float = 60.0) -> None:
    if path is None:
        return
    if not path.is_absolute():
        raise OracleError("--start-gate must be absolute")
    deadline = time.monotonic() + timeout_seconds
    while True:
        if path.is_symlink():
            raise OracleError("start gate must not be a symlink")
        if path.is_file():
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise OracleError(f"cannot read start gate: {exc}") from exc
            if payload != b"go\n":
                raise OracleError("start gate payload must equal 'go\\n'")
            return
        if path.exists():
            raise OracleError("start gate must be a regular file")
        if time.monotonic() >= deadline:
            raise OracleError(
                f"start gate was not released within {timeout_seconds:.3f} seconds"
            )
        time.sleep(0.01)


def _validate_paths(model_path: Path, output_path: Path) -> tuple[Path, Path]:
    model = model_path.absolute()
    if model.is_symlink():
        raise OracleError("--model must be a non-symlink directory")
    model = model.resolve(strict=True)
    if not model.is_dir():
        raise OracleError("--model must be a non-symlink directory")
    output = output_path.absolute()
    if output.exists() or output.is_symlink():
        raise OracleError(f"refusing to replace oracle output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise OracleError("oracle output parent must not be a symlink")
    return model, output


def _layer_capture_modules(model: object) -> tuple[list[str], list[object]]:
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise OracleError(
            f"layer capture requires Qwen3_5ForCausalLM, got {type(model).__name__}"
        )
    try:
        text_model = model.model
    except AttributeError as exc:
        raise OracleError("pinned Qwen3.5 causal LM omits its text model") from exc
    if type(text_model).__name__ != "Qwen3_5TextModel":
        raise OracleError(
            f"layer capture requires Qwen3_5TextModel, got {type(text_model).__name__}"
        )
    try:
        layer_types = list(text_model.config.layer_types)
        layers = list(text_model.layers)
        modules = [("embedding", text_model.embed_tokens)]
        modules.extend(
            (f"layer_{index:02}_{layer_types[index]}", layer)
            for index, layer in enumerate(layers)
        )
        modules.append(("final_norm", text_model.norm))
    except (AttributeError, IndexError, TypeError) as exc:
        raise OracleError("pinned Qwen3.5 text model structure changed") from exc
    if len(layers) != 32 or len(layer_types) != len(layers):
        raise OracleError("pinned Qwen3.5 layer inventory changed")
    return [name for name, _ in modules], [module for _, module in modules]


def generate(
    model_path: Path,
    output_path: Path,
    *,
    request_path: Path | None = None,
    start_gate: Path | None = None,
    capture_layer_last_rows: bool = False,
) -> tuple[dict[str, object], bool]:
    model_path, output_path = _validate_paths(model_path, output_path)
    request: dict[str, object] | None = None
    request_sha256: str | None = None
    if request_path is not None:
        request, request_sha256 = load_request(request_path)
    _wait_for_start_gate(start_gate)
    torch, AutoModelForCausalLM, AutoTokenizer, save_file = _load_packages()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise OracleError("the bounded HF oracle requires exactly one Torch accelerator")
    if torch.version.hip is None:
        raise OracleError("the Strix Halo HF oracle requires the pinned ROCm Torch build")

    torch.manual_seed(20260715)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    input_token_ids = INPUT_TOKEN_IDS
    if request is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        prompt = request["prompt"]
        encoded = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=True,
            add_generation_prompt=prompt["add_generation_prompt"],
            **prompt["template_kwargs"],
        )
        token_ids = encoded["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if token_ids != prompt["token_ids"]:
            raise OracleError(
                "pinned tokenizer/chat template does not reproduce request prompt token IDs"
            )
        for item in [*request["continuation_prefix"], *request["candidates"]]:
            decoded = tokenizer.decode(
                [item["token_id"]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded != item["text"]:
                raise OracleError(
                    f"pinned tokenizer decodes token {item['token_id']} as {decoded!r}; "
                    f"request records {item['text']!r}"
                )
        input_token_ids = request["input_token_ids"]
    if capture_layer_last_rows and request is None:
        raise OracleError("--capture-layer-last-rows requires --request")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to("cuda")
    input_ids = torch.tensor([input_token_ids], dtype=torch.long, device="cuda")
    captured_rows = []
    boundary_names: list[str] = []
    handles = []
    if capture_layer_last_rows:
        boundary_names, modules = _layer_capture_modules(model)

        def capture_last_row(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise OracleError("HF layer boundary did not return a rank-three tensor")
            if output.shape[0] != 1 or output.shape[2] != 2560:
                raise OracleError(
                    f"unexpected HF layer boundary shape {tuple(output.shape)}"
                )
            captured_rows.append(output[:, -1, :].detach().clone())

        handles = [module.register_forward_hook(capture_last_row) for module in modules]
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if logits.ndim != 1 or logits.numel() != 248_320:
        raise OracleError(f"unexpected HF logit shape {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise OracleError("HF full-vocabulary logits contain non-finite values")
    logits_sha256 = "sha256:" + hashlib.sha256(
        logits.contiguous().numpy().tobytes(order="C")
    ).hexdigest()

    layer_last_rows = None
    if capture_layer_last_rows:
        if len(captured_rows) != len(boundary_names):
            raise OracleError(
                f"captured {len(captured_rows)} HF boundaries; expected {len(boundary_names)}"
            )
        layer_last_rows = torch.cat(captured_rows, dim=0).float().cpu().contiguous()
        if tuple(layer_last_rows.shape) != (34, 2560):
            raise OracleError(
                f"unexpected HF layer-last-row shape {tuple(layer_last_rows.shape)}"
            )
        if not bool(torch.isfinite(layer_last_rows).all()):
            raise OracleError("HF layer-last-row reference contains non-finite values")

    metadata = {
        "attention_implementation": "eager",
        "device_name": torch.cuda.get_device_name(0),
        "input_token_ids": json.dumps(input_token_ids, separators=(",", ":")),
        "linear_attention_implementation": "transformers_torch_fallback",
        "schema": LAYER_SCHEMA if capture_layer_last_rows else SCHEMA,
        "torch_commit": TORCH_COMMIT,
        "torch_hip_version": str(torch.version.hip),
        "torch_version": torch.__version__,
        "transformers_version": TRANSFORMERS_VERSION,
    }
    if capture_layer_last_rows:
        metadata["boundary_names"] = json.dumps(
            boundary_names, ensure_ascii=True, separators=(",", ":")
        )
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise OracleError(f"refusing stale temporary oracle output {temporary}")
    tensors = {"input_ids": input_ids.cpu(), "logits": logits.contiguous()}
    if layer_last_rows is not None:
        tensors["layer_last_rows"] = layer_last_rows
    save_file(tensors, temporary, metadata=metadata)
    os.replace(temporary, output_path)
    output_path.chmod(0o600)
    evidence = {
        "argmax": int(logits.argmax().item()),
        "device": metadata["device_name"],
        "duration_seconds": time.monotonic() - started,
        "logits_sha256": logits_sha256,
        "output_bytes": output_path.stat().st_size,
        "torch_hip_version": metadata["torch_hip_version"],
        "torch_version": metadata["torch_version"],
        "transformers_version": TRANSFORMERS_VERSION,
        "vocab": int(logits.numel()),
    }
    evidence.update(_current_cgroup_memory())
    if layer_last_rows is not None:
        evidence.update(
            {
                "boundary_count": len(boundary_names),
                "boundary_names": boundary_names,
                "hidden_size": int(layer_last_rows.shape[1]),
                "layer_last_rows_sha256": "sha256:"
                + hashlib.sha256(
                    layer_last_rows.numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    if request is not None:
        top_values, top_indices = torch.topk(logits, k=10, largest=True, sorted=True)
        top_tokens = [
            {
                "logit": float(value),
                "text": tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "token_id": int(token_id),
            }
            for value, token_id in zip(top_values.tolist(), top_indices.tolist())
        ]
        candidates = []
        for item in request["candidates"]:
            token_id = item["token_id"]
            logit = float(logits[token_id].item())
            candidates.append(
                {
                    "engine": item["engine"],
                    "logit": logit,
                    "rank": int((logits > logit).sum().item()) + 1,
                    "text": item["text"],
                    "token_id": token_id,
                }
            )
        evidence.update(
            {
                "attention_implementation": "eager",
                "argmax_text": top_tokens[0]["text"],
                "candidate_tokens": candidates,
                "configuration_sha256": f"sha256:{CONFIGURATION_SHA256}",
                "deterministic_algorithms": True,
                "dtype": "bfloat16",
                "input_token_count": len(input_token_ids),
                "input_token_ids_sha256": canonical_sha256(input_token_ids),
                "linear_attention_implementation": "transformers_torch_fallback",
                "modeling_sha256": f"sha256:{MODELING_SHA256}",
                "request_id": request["id"],
                "request_sha256": request_sha256,
                "tf32_allowed": False,
                "torch_commit": TORCH_COMMIT,
                "top_logit_margin": float(top_values[0].item() - top_values[1].item()),
                "top_tokens": top_tokens,
            }
        )
    return evidence, request is not None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--start-gate", type=Path)
    parser.add_argument("--capture-layer-last-rows", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        evidence, is_next_token = generate(
            args.model,
            args.output,
            request_path=args.request,
            start_gate=args.start_gate,
            capture_layer_last_rows=args.capture_layer_last_rows,
        )
    except BaseException as exc:
        print(f"HF full-logit oracle failed: {exc}", file=sys.stderr)
        return 1
    print(
        (
            LAYER_PASS_PREFIX
            if args.capture_layer_last_rows
            else NEXT_TOKEN_PASS_PREFIX
            if is_next_token
            else "KILN_HF_FULL_LOGIT_REFERENCE_PASS "
        )
        + json.dumps(evidence, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
