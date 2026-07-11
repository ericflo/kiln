#!/usr/bin/env python3
"""Generate the source-pinned TRL/PyTorch GRPO oracle fixture.

This is a local qualification tool. It deliberately does not run in automatic
CI because installing PyTorch and TRL there would be expensive. The checked-in
fixture is consumed by Rust tests on every backend; rerun this script only when
the pinned oracle or fixture contract changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "crates/kiln-train/tests/fixtures/grpo_trl_oracle_v1.json"

SCHEMA = "kiln.grpo-trl-oracle.v1"
TRL_VERSION = "1.8.0"
TRL_COMMIT = "95809b942eb5d11d0b06d749510d88be99230b73"
TRL_GRPO_TRAINER_SHA256 = "52d9a6c1e298df35d0da4a6fa17874d750ee627f6ac15393c8860d74d1ba4917"
TORCH_VERSION = "2.13.0"
TORCH_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"

ADAMW = {
    "lr": 0.003,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.01,
}

CASES: list[dict[str, Any]] = [
    {
        "name": "token_positive_asymmetric_k3",
        "policy_log_probs": [-0.7, -1.5, -0.8, -1.4],
        "behavior_log_probs": [-1.0, -1.0, -1.1, -1.2],
        "kl_reference_log_probs": [-0.6, -1.7, -1.0, -1.0],
        "advantage": 0.6,
        "clip_low": 0.2,
        "clip_high": 0.28,
        "kl_coeff": 0.07,
        "is_level": "token",
        "loss_type": "grpo",
        "reinforce": False,
    },
    {
        "name": "token_negative_lower_clip_k3",
        "policy_log_probs": [-1.5, -0.7, -1.3, -0.9],
        "behavior_log_probs": [-1.0, -1.0, -1.0, -1.0],
        "kl_reference_log_probs": [-1.2, -0.9, -1.6, -0.5],
        "advantage": -0.45,
        "clip_low": 0.2,
        "clip_high": 0.28,
        "kl_coeff": 0.05,
        "is_level": "token",
        "loss_type": "grpo",
        "reinforce": False,
    },
    {
        "name": "sequence_gspo_k3",
        "policy_log_probs": [-0.9, -1.2, -0.9, -0.75],
        "behavior_log_probs": [-1.2, -0.9, -1.3, -0.8],
        "kl_reference_log_probs": [-0.7, -1.4, -1.0, -1.1],
        "advantage": 0.55,
        "clip_low": 0.2,
        "clip_high": 0.2,
        "kl_coeff": 0.08,
        "is_level": "sequence",
        "loss_type": "grpo",
        "reinforce": False,
    },
    {
        "name": "cispo_upper_weight_cap_k3",
        "policy_log_probs": [-0.2, -2.2, -0.6, -1.4],
        "behavior_log_probs": [-1.8, -0.5, -1.0, -1.0],
        "kl_reference_log_probs": [-0.5, -1.8, -0.9, -1.1],
        "advantage": 0.5,
        "clip_low": 0.2,
        "clip_high": 0.28,
        "cispo_max_weight": 2.0,
        "kl_coeff": 0.04,
        "is_level": "cispo",
        "loss_type": "cispo",
        "reinforce": False,
    },
    {
        "name": "no_importance_correction_keeps_k3",
        "policy_log_probs": [-1.0, -1.2, -0.8, -1.5],
        "behavior_log_probs": [-7.0, -6.0, -8.0, -9.0],
        "kl_reference_log_probs": [-0.7, -1.8, -1.0, -1.1],
        "advantage": 0.4,
        "clip_low": 0.2,
        "clip_high": 0.2,
        "kl_coeff": 0.06,
        "is_level": "token",
        "loss_type": "grpo",
        "reinforce": True,
    },
]


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def package_version_base(version: str) -> str:
    return version.split("+", 1)[0]


def load_oracle_packages():
    try:
        import torch
        import trl
        from trl.trainer import grpo_trainer
    except ImportError as exc:
        raise RuntimeError(
            "the GRPO oracle requires torch==2.13.0 and trl==1.8.0; "
            "run it with `uv run --index https://download.pytorch.org/whl/cpu "
            "--index-strategy unsafe-best-match --with torch==2.13.0+cpu "
            "--with trl==1.8.0 python "
            "scripts/qualification/grpo_trl_oracle.py --check`"
        ) from exc

    if package_version_base(torch.__version__) != TORCH_VERSION:
        raise RuntimeError(
            f"expected torch {TORCH_VERSION}, got {torch.__version__}"
        )
    if package_version_base(trl.__version__) != TRL_VERSION:
        raise RuntimeError(f"expected trl {TRL_VERSION}, got {trl.__version__}")

    trainer_path = Path(grpo_trainer.__file__).resolve()
    trainer_hash = hashlib.sha256(trainer_path.read_bytes()).hexdigest()
    if trainer_hash != TRL_GRPO_TRAINER_SHA256:
        raise RuntimeError(
            "installed TRL GRPO source does not match the pinned v1.8.0 tag: "
            f"{trainer_path} has sha256:{trainer_hash}"
        )
    return torch, grpo_trainer


class OracleAccelerator:
    num_processes = 1
    sync_gradients = True

    @staticmethod
    def gather(value):
        return value


def tensor_values(tensor) -> list[float]:
    return [float(value) for value in tensor.detach().reshape(-1).cpu().tolist()]


def run_case(torch, grpo_trainer, case: dict[str, Any]) -> dict[str, Any]:
    policy = torch.nn.Parameter(
        torch.tensor([case["policy_log_probs"]], dtype=torch.float32)
    )
    behavior = torch.tensor(
        [case["behavior_log_probs"]], dtype=torch.float32
    )
    reference = torch.tensor(
        [case["kl_reference_log_probs"]], dtype=torch.float32
    )
    num_tokens = policy.shape[1]
    mask = torch.ones((1, num_tokens), dtype=torch.float32)

    trainer = object.__new__(grpo_trainer.GRPOTrainer)
    trainer.model = types.SimpleNamespace(training=False)
    trainer.accelerator = OracleAccelerator()
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.top_entropy_quantile = 1.0
    trainer.off_policy_mask_threshold = None
    trainer.importance_sampling_level = (
        "sequence" if case["is_level"] == "sequence" else "token"
    )
    trainer.beta = case["kl_coeff"]
    trainer.epsilon_low = case["clip_low"]
    trainer.epsilon_high = (
        case["cispo_max_weight"]
        if case["loss_type"] == "cispo"
        else case["clip_high"]
    )
    trainer.loss_type = case["loss_type"]
    trainer.use_vllm = False
    trainer._entropy_bonus_enabled = False
    trainer.aux_loss_enabled = False
    trainer.current_gradient_accumulation_steps = 1
    trainer.args = types.SimpleNamespace(
        use_bias_correction_kl=False,
        delta=None,
    )

    def precomputed_log_probs(_self, *_args, **_kwargs):
        return policy, torch.zeros_like(policy), None

    trainer._get_per_token_logps_and_entropies = types.MethodType(
        precomputed_log_probs, trainer
    )
    old = policy.detach() if case["reinforce"] else behavior
    inputs = {
        "prompt_ids": torch.zeros((1, 1), dtype=torch.long),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.zeros((1, num_tokens), dtype=torch.long),
        "completion_mask": mask,
        "advantages": torch.tensor([case["advantage"]], dtype=torch.float32),
        "old_per_token_logps": old,
        "ref_per_token_logps": reference,
        "num_items_in_batch": num_tokens,
    }

    loss = grpo_trainer.GRPOTrainer._compute_loss(trainer, trainer.model, inputs)
    loss.backward()
    gradient = tensor_values(policy.grad)

    importance_log_ratios = (
        torch.zeros_like(policy)
        if case["reinforce"]
        else policy.detach() - behavior
    )
    token_ratios = torch.exp(importance_log_ratios)
    if case["is_level"] == "sequence":
        observed_ratios = torch.exp(importance_log_ratios.mean(dim=1, keepdim=True))
    else:
        observed_ratios = token_ratios
    k3 = torch.exp(reference - policy.detach()) - (reference - policy.detach()) - 1.0

    if case["loss_type"] == "cispo":
        below_clip = torch.zeros_like(observed_ratios, dtype=torch.bool)
        above_clip = observed_ratios > case["cispo_max_weight"]
        trl_clip_metric = trainer._metrics["eval"]["cispo_clip_ratio"][-1]
    else:
        below_clip = observed_ratios < 1.0 - case["clip_low"]
        above_clip = observed_ratios > 1.0 + case["clip_high"]
        trl_clip_metric = trainer._metrics["eval"]["clip_ratio/region_mean"][-1]

    optimizer = torch.optim.AdamW(
        [policy],
        lr=ADAMW["lr"],
        betas=(ADAMW["beta1"], ADAMW["beta2"]),
        eps=ADAMW["eps"],
        weight_decay=ADAMW["weight_decay"],
    )
    optimizer.step()
    state = optimizer.state[policy]

    output = dict(case)
    output["loss_normalizer"] = 1.0 / num_tokens
    output["kl_estimator"] = "k3"
    output["expected"] = {
        "loss": float(loss.detach().cpu().item()),
        "policy_log_prob_grad": gradient,
        "token_importance_ratios": tensor_values(token_ratios),
        "observed_importance_ratios": tensor_values(observed_ratios),
        "below_clip_count": int(below_clip.sum().item()),
        "above_clip_count": int(above_clip.sum().item()),
        "trl_clip_fraction": float(trl_clip_metric),
        "k3_per_token": tensor_values(k3),
        "mean_k3": float(k3.mean().cpu().item()),
        "adamw_parameter": tensor_values(policy),
        "adamw_exp_avg": tensor_values(state["exp_avg"]),
        "adamw_exp_avg_sq": tensor_values(state["exp_avg_sq"]),
    }
    return output


def generate_fixture() -> dict[str, Any]:
    torch, grpo_trainer = load_oracle_packages()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    cases = [run_case(torch, grpo_trainer, case) for case in CASES]
    fixture_inputs = json.dumps(CASES, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": SCHEMA,
        "oracle": {
            "trl_version": TRL_VERSION,
            "trl_commit": TRL_COMMIT,
            "trl_grpo_trainer_sha256": "sha256:" + TRL_GRPO_TRAINER_SHA256,
            "torch_version": TORCH_VERSION,
            "torch_commit": TORCH_COMMIT,
            "device": "cpu",
            "dtype": "float32",
            "fixture_inputs_sha256": sha256_bytes(fixture_inputs),
            "execution": "TRL GRPOTrainer._compute_loss + PyTorch autograd/AdamW",
        },
        "adamw": ADAMW,
        "tolerances": {
            "loss_abs": 2e-6,
            "metric_abs": 2e-6,
            "gradient_abs": 3e-6,
            "adamw_abs": 3e-6,
        },
        "cases": cases,
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="regenerate and compare the checked-in fixture")
    mode.add_argument("--stdout", action="store_true", help="write the generated fixture to stdout")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        generated = canonical_bytes(generate_fixture())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"GRPO TRL oracle failed: {exc}", file=sys.stderr)
        return 1

    output = args.output.resolve()
    if args.check:
        try:
            current = output.read_bytes()
        except OSError as exc:
            print(f"GRPO TRL oracle fixture read failed: {exc}", file=sys.stderr)
            return 1
        if current != generated:
            print(
                f"GRPO TRL oracle fixture drift: regenerate {output}",
                file=sys.stderr,
            )
            return 1
        print(f"GRPO TRL oracle fixture matches {TRL_VERSION} / torch {TORCH_VERSION}")
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
