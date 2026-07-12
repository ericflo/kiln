#!/usr/bin/env python3
"""Generate the source-pinned PyTorch AdamW trajectory oracle.

The fixture covers ordinary and epsilon-dominated low gradients for ten steps
under both of Kiln's declared native storage contracts: F32 parameters/moments
and BF16 parameters/moments without a separate F32 master. Automatic CI checks
the committed fixture structurally; regeneration uses the pinned local Torch
environment because installing accelerator packages in hosted CI is wasteful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "crates/kiln-optim/tests/fixtures/adamw_pytorch_oracle_v1.json"
SCHEMA = "kiln.adamw-pytorch-oracle.v1"
TORCH_VERSION = "2.13.0"
TORCH_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
TORCH_ADAMW_SHA256 = "d0f3a2d889a43b65ed48175900f634bb5a244c8cd79a3f1117dc852ddad4b1e9"
TORCH_ADAM_SHA256 = "bde360b0bb9b7869f1cec04a3b41a90b8eabb84a613787d97b88d87f2f3ae1ec"
STEP_COUNT = 10
SEED = 0


def _ordinary_gradients() -> list[list[float]]:
    rows = []
    for step in range(1, STEP_COUNT + 1):
        row = []
        for lane in range(8):
            sign = -1.0 if (step + lane) % 3 == 0 else 1.0
            row.append(sign * 0.004 * (lane + 1) * (1.0 + step * 0.07))
        rows.append(row)
    return rows


def _low_gradients() -> list[list[float]]:
    base = [1e-12, -3e-11, 5e-10, -7e-9, 9e-8, -2e-7, 1e-6, -5e-6]
    multipliers = [1.0, 0.5, -1.0, 1.5, -0.25, 2.0, -0.75, 1.25, -1.5, 0.8]
    return [[value * multipliers[step] for value in base] for step in range(STEP_COUNT)]


INITIAL_PARAMETER = [1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 0.125, -0.0625]
ORDINARY_HP = {
    "lr": 0.003,
    "beta1": 0.85,
    "beta2": 0.97,
    "eps": 1e-7,
    "weight_decay": 0.04,
}
LOW_GRADIENT_HP = {
    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.0,
}
CASES: list[dict[str, Any]] = [
    {
        "id": f"ordinary_{dtype}",
        "class": "ordinary",
        "dtype": dtype,
        "initial_parameter": INITIAL_PARAMETER,
        "gradients": _ordinary_gradients(),
        "hyperparameters": ORDINARY_HP,
    }
    for dtype in ("float32", "bfloat16")
] + [
    {
        "id": f"low_gradient_{dtype}",
        "class": "low_gradient",
        "dtype": dtype,
        "initial_parameter": INITIAL_PARAMETER,
        "gradients": _low_gradients(),
        "hyperparameters": LOW_GRADIENT_HP,
    }
    for dtype in ("float32", "bfloat16")
]


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _package_version_base(version: str) -> str:
    return version.split("+", 1)[0]


def _load_torch():
    try:
        import torch
        from torch.optim import adam as adam_module
        from torch.optim import adamw as adamw_module
    except ImportError as exc:
        raise RuntimeError(
            "the AdamW oracle requires torch==2.13.0; run it with the "
            "requirements-sft.lock environment"
        ) from exc
    if _package_version_base(torch.__version__) != TORCH_VERSION:
        raise RuntimeError(f"expected torch {TORCH_VERSION}, got {torch.__version__}")
    if torch.version.git_version != TORCH_COMMIT:
        raise RuntimeError(
            f"expected torch commit {TORCH_COMMIT}, got {torch.version.git_version}"
        )
    for module, expected in (
        (adamw_module, TORCH_ADAMW_SHA256),
        (adam_module, TORCH_ADAM_SHA256),
    ):
        source_path = Path(module.__file__).resolve()
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_hash != expected:
            raise RuntimeError(
                "installed torch optimizer source does not match the pinned build: "
                f"{source_path} has sha256:{source_hash}"
            )
    return torch


def _tensor_values(tensor) -> list[float]:
    return [float(value) for value in tensor.detach().reshape(-1).cpu().tolist()]


def _run_case(torch, case: dict[str, Any]) -> dict[str, Any]:
    dtype = torch.float32 if case["dtype"] == "float32" else torch.bfloat16
    parameter = torch.nn.Parameter(
        torch.tensor(case["initial_parameter"], dtype=dtype, device="cpu")
    )
    hp = case["hyperparameters"]
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=hp["lr"],
        betas=(hp["beta1"], hp["beta2"]),
        eps=hp["eps"],
        weight_decay=hp["weight_decay"],
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    stored_gradients = []
    trajectory = []
    for expected_step, raw_gradient in enumerate(case["gradients"], start=1):
        gradient = torch.tensor(raw_gradient, dtype=dtype, device="cpu")
        stored_gradients.append(_tensor_values(gradient))
        parameter.grad = gradient
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = optimizer.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"unexpected AdamW state keys: {sorted(state)}")
        if state["step"].item() != expected_step:
            raise RuntimeError("PyTorch AdamW step counter drifted")
        if state["exp_avg"].dtype != dtype or state["exp_avg_sq"].dtype != dtype:
            raise RuntimeError("PyTorch AdamW moments do not use parameter dtype")
        trajectory.append(
            {
                "step": expected_step,
                "parameter": _tensor_values(parameter),
                "exp_avg": _tensor_values(state["exp_avg"]),
                "exp_avg_sq": _tensor_values(state["exp_avg_sq"]),
            }
        )

    return {
        "id": case["id"],
        "class": case["class"],
        "dtype": case["dtype"],
        "hyperparameters": hp,
        "stored_initial_parameter": _tensor_values(
            torch.tensor(case["initial_parameter"], dtype=dtype)
        ),
        "stored_gradients": stored_gradients,
        "state_contract": {
            "parameter_dtype": case["dtype"],
            "first_moment_dtype": case["dtype"],
            "second_moment_dtype": case["dtype"],
            "step_dtype": "float32",
            "separate_master_parameter": False,
        },
        "trajectory": trajectory,
    }


def generate_fixture() -> dict[str, Any]:
    torch = _load_torch()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inputs = json.dumps(CASES, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": SCHEMA,
        "oracle": {
            "seed": SEED,
            "torch_version": TORCH_VERSION,
            "torch_commit": TORCH_COMMIT,
            "torch_adamw_sha256": "sha256:" + TORCH_ADAMW_SHA256,
            "torch_adam_sha256": "sha256:" + TORCH_ADAM_SHA256,
            "device": "cpu",
            "implementation": "torch.optim.AdamW",
            "foreach": False,
            "fused": False,
            "capturable": False,
            "differentiable": False,
            "fixture_inputs_sha256": _sha256_bytes(inputs),
        },
        "tolerances": {
            "float32_absolute": 2e-12,
            "float32_relative": 5e-6,
            "bfloat16_parameter_max_ulp": 1,
            "bfloat16_first_moment_max_ulp": 4,
            "bfloat16_second_moment_max_ulp": 3,
            "bfloat16_reason": (
                "Kiln's fused kernel computes each lane in F32 and rounds each output once; "
                "PyTorch eager BF16 AdamW rounds across separate lerp, mul, addcmul, sqrt, "
                "division, and addcdiv tensor operations."
            ),
        },
        "cases": [_run_case(torch, case) for case in CASES],
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--stdout", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        generated = canonical_bytes(generate_fixture())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"PyTorch AdamW oracle failed: {exc}", file=sys.stderr)
        return 1
    output = args.output.resolve()
    if args.check:
        try:
            current = output.read_bytes()
        except OSError as exc:
            print(f"PyTorch AdamW fixture read failed: {exc}", file=sys.stderr)
            return 1
        if current != generated:
            print(f"PyTorch AdamW fixture drift: regenerate {output}", file=sys.stderr)
            return 1
        print(f"PyTorch AdamW fixture matches torch {TORCH_VERSION}")
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
