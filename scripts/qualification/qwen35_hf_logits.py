#!/usr/bin/env python3
"""Produce a pinned full-vocabulary Qwen3.5-4B Hugging Face reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


SCHEMA = "kiln.qwen35-hf-full-logits.v1"
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
        from transformers import AutoModelForCausalLM
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
    return torch, AutoModelForCausalLM, save_file


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


def generate(model_path: Path, output_path: Path) -> dict[str, object]:
    model_path, output_path = _validate_paths(model_path, output_path)
    torch, AutoModelForCausalLM, save_file = _load_packages()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise OracleError("the bounded HF oracle requires exactly one Torch accelerator")
    if torch.version.hip is None:
        raise OracleError("the Strix Halo HF oracle requires the pinned ROCm Torch build")

    torch.manual_seed(20260715)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to("cuda")
    input_ids = torch.tensor([INPUT_TOKEN_IDS], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float().cpu()
    if logits.ndim != 1 or logits.numel() != 248_320:
        raise OracleError(f"unexpected HF logit shape {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise OracleError("HF full-vocabulary logits contain non-finite values")
    logits_sha256 = "sha256:" + hashlib.sha256(
        logits.contiguous().numpy().tobytes(order="C")
    ).hexdigest()

    metadata = {
        "attention_implementation": "eager",
        "device_name": torch.cuda.get_device_name(0),
        "input_token_ids": json.dumps(INPUT_TOKEN_IDS, separators=(",", ":")),
        "linear_attention_implementation": "transformers_torch_fallback",
        "schema": SCHEMA,
        "torch_commit": TORCH_COMMIT,
        "torch_hip_version": str(torch.version.hip),
        "torch_version": torch.__version__,
        "transformers_version": TRANSFORMERS_VERSION,
    }
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise OracleError(f"refusing stale temporary oracle output {temporary}")
    save_file(
        {"input_ids": input_ids.cpu(), "logits": logits.contiguous()},
        temporary,
        metadata=metadata,
    )
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
    return evidence


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        evidence = generate(args.model, args.output)
    except BaseException as exc:
        print(f"HF full-logit oracle failed: {exc}", file=sys.stderr)
        return 1
    print(
        "KILN_HF_FULL_LOGIT_REFERENCE_PASS "
        + json.dumps(evidence, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
