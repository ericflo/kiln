#!/usr/bin/env python3
"""Generate backend capability reports from the live source tree."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "docs" / "backend-capability-report.md"
REPORT_JSON = ROOT / "docs" / "backend-capability-report.json"

BACKENDS = {
    "cuda": ROOT / "crates" / "kiln-model" / "src" / "backend" / "cuda.rs",
    "rocm": ROOT / "crates" / "kiln-model" / "src" / "backend" / "rocm.rs",
    "metal": ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal.rs",
    "vulkan": ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan.rs",
}

FEATURE_CRATES = [
    ROOT / "crates" / "kiln-server" / "Cargo.toml",
    ROOT / "crates" / "kiln-model" / "Cargo.toml",
    ROOT / "crates" / "kiln-tensor" / "Cargo.toml",
    ROOT / "crates" / "kiln-train" / "Cargo.toml",
]

FEATURE_FAMILIES = ["cuda", "rocm", "metal", "vulkan"]

SUPPORT_PAIRS = {
    "supports_flash_attn_prefill": "flash_attn_prefill",
    "supports_flash_attn_prefill_head_major": "flash_attn_prefill_head_major",
    "supports_flash_attn_paged_decode": "flash_attn_paged_decode",
    "supports_strict_paged_decode_contiguous_batch": "flash_attn_paged_decode_contiguous_batch",
    "supports_resident_decode": "decode_resident_pool_ready",
    "supports_gdn_forward_substitution": "gdn_forward_substitution",
    "supports_gdn_recurrent_step": "gdn_recurrent_step",
    "supports_gdn_chunk_prep": "gdn_chunk_prep",
    "supports_gdn_chunk_scan": "gdn_chunk_scan",
    "supports_gdn_full_chunk_forward": "gdn_full_chunk_forward",
    "supports_gdn_gates": "gdn_gates",
    "supports_gdn_gated_rms_norm": "gdn_gated_rms_norm",
    "supports_linear_decode_argmax": "linear_decode_argmax",
    "supports_linear_decode_argmax_batch": "linear_decode_argmax_batch",
}


@dataclass
class FunctionDef:
    name: str
    body: str
    line: int


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    escape = False
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif in_char:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_char = False
        elif ch == "/" and nxt == "/":
            in_line_comment = True
            i += 1
        elif ch == "/" and nxt == "*":
            in_block_comment = True
            i += 1
        elif ch == '"':
            in_string = True
        elif ch == "'":
            in_char = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unclosed brace at byte {open_idx}")


def parse_functions(path: Path) -> dict[str, FunctionDef]:
    text = path.read_text()
    functions: dict[str, FunctionDef] = {}
    pattern = re.compile(r"(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]")
    for match in pattern.finditer(text):
        name = match.group(1)
        brace = text.find("{", match.end())
        if brace == -1:
            continue
        try:
            end = find_matching_brace(text, brace)
        except ValueError:
            continue
        body = text[brace + 1 : end]
        functions[name] = FunctionDef(name=name, body=body, line=text.count("\n", 0, match.start()) + 1)
    return functions


def parse_trait_method_names(path: Path, trait_name: str) -> set[str]:
    text = path.read_text()
    match = re.search(rf"\bpub\s+trait\s+{re.escape(trait_name)}\b", text)
    if not match:
        raise ValueError(f"{trait_name} trait not found in {path}")
    brace = text.find("{", match.end())
    if brace == -1:
        raise ValueError(f"{trait_name} trait body not found in {path}")
    end = find_matching_brace(text, brace)
    body = text[brace + 1 : end]
    return set(re.findall(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]", body))


def body_without_comments(body: str) -> str:
    body = re.sub(r"//.*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return body.strip()


def support_status(body: str) -> str:
    stripped = body_without_comments(body)
    compact = re.sub(r"\s+", "", stripped)
    if compact == "true":
        return "literal_true"
    if compact == "false":
        return "literal_false"
    if "std::env::var" in stripped or "env_flag" in stripped:
        return "env_gated"
    return "dynamic"


def typed_support_state(status: str, pair_declines: bool) -> str:
    if pair_declines:
        return "Declined"
    if status == "literal_false":
        return "Declined"
    if status == "env_gated":
        return "NativeWithConstraints"
    if status in {"literal_true", "dynamic"}:
        return "NativeWithConstraints"
    return "Unsupported"


def always_declines(body: str) -> bool:
    stripped = body_without_comments(body)
    compact = re.sub(r"\s+", "", stripped)
    return compact in {"Ok(None)", "returnOk(None);"}


def env_gates(source: str) -> list[str]:
    names = set(re.findall(r'"(KILN_[A-Z0-9_]+)"', source))
    return sorted(name for name in names if any(key in name for key in FEATURE_FAMILIES + ["CUDA", "ROCM", "METAL", "VULKAN"]))


def gate_hints(body: str) -> dict[str, bool]:
    return {
        "dtype": ".dtype()" in body or "DType::" in body,
        "shape": ".dims()" in body or ".shape()" in body,
        "layout": "contiguous" in body or "stride" in body,
        "env": bool(env_gates(body)),
    }


def feature_report() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for path in FEATURE_CRATES:
        data = load_toml(path)
        package = data.get("package", {}).get("name", path.parent.name)
        features = data.get("features", {})
        report[package] = {
            family: features.get(family, [])
            for family in FEATURE_FAMILIES
            if family in features
        }
    return report


def backend_report(trait_methods: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for backend, path in BACKENDS.items():
        functions = parse_functions(path)
        overrides = sorted(name for name in functions if name in trait_methods)
        support_methods: dict[str, Any] = {}
        for name, fun in sorted(functions.items()):
            if not name.startswith("supports_"):
                continue
            pair = SUPPORT_PAIRS.get(name, name.removeprefix("supports_"))
            paired_fun = functions.get(pair)
            status = support_status(fun.body)
            pair_declines = bool(paired_fun and always_declines(paired_fun.body))
            entry = {
                "line": fun.line,
                "status": status,
                "support_state": typed_support_state(status, pair_declines),
                "paired_method": pair if paired_fun else None,
                "paired_method_line": paired_fun.line if paired_fun else None,
                "paired_method_always_declines": pair_declines,
                "env_gates": env_gates(fun.body),
                "gate_hints": gate_hints(fun.body),
            }
            support_methods[name] = entry
            if status == "literal_true" and pair_declines:
                mismatches.append(
                    {
                        "backend": backend,
                        "support_method": name,
                        "support_line": fun.line,
                        "paired_method": pair,
                        "paired_line": paired_fun.line if paired_fun else None,
                    }
                )
        source = path.read_text()
        report[backend] = {
            "source": str(path.relative_to(ROOT)),
            "override_count": len(overrides),
            "overrides": overrides,
            "support_methods": support_methods,
            "env_gates": env_gates(source),
        }
    return report, mismatches


def fallback_policy_report() -> dict[str, Any]:
    return {
        "cuda": {
            "generic_device_op_fallback": "strict_native_miss_errors",
            "evidence": "crates/kiln-tensor/src/device_op.rs CUDA native miss falls through on CUDA storage and fails loudly",
            "counter": "none",
        },
        "rocm": {
            "generic_device_op_fallback": "host_round_trip_correctness_fallback",
            "evidence": "crates/kiln-tensor/src/device_op.rs ROCm missing native forward stages through CPU",
            "counter": "kiln_tensor::profile::device_op_host_fallback_counts().rocm_op{1,2,3}",
        },
        "metal": {
            "generic_device_op_fallback": "host_round_trip_correctness_fallback",
            "evidence": "crates/kiln-tensor/src/device_op.rs Metal missing native forward stages through CPU",
            "counter": "kiln_tensor::profile::device_op_host_fallback_counts().metal_op{1,2,3}",
        },
        "vulkan": {
            "generic_device_op_fallback": "host_round_trip_correctness_fallback",
            "evidence": "crates/kiln-tensor/src/device_op.rs Vulkan missing native forward stages through CPU",
            "counter": "kiln_tensor::profile::device_op_host_fallback_counts().vulkan_op{1,2,3}",
        },
    }


def decode_hot_path_policy_report() -> dict[str, Any]:
    return {
        "cpu": {
            "default_policy": "CorrectnessAllowed",
            "debug_opt_in": "not required",
            "enforcement": "CPU is the reference path",
        },
        "cuda": {
            "default_policy": "CorrectnessAllowed",
            "debug_opt_in": "not required",
            "enforcement": "CUDA native misses remain device-visible/errors rather than silent host staging",
        },
        "rocm": {
            "default_policy": "NativeRequired",
            "debug_opt_in": "KILN_DECODE_HOT_PATH_DEBUG_FALLBACK=1 or KILN_ROCM_DECODE_BATCH_GENERIC_FALLBACK=1",
            "enforcement": "batched decode errors before generic fallback when no ROCm native path produced tokens",
        },
        "metal": {
            "default_policy": "NativeRequired",
            "debug_opt_in": "KILN_DECODE_HOT_PATH_DEBUG_FALLBACK=1 or KILN_METAL_DECODE_BATCH_GENERIC_FALLBACK=1",
            "enforcement": "batched/sample decode errors before generic fallback when no Metal native path produced tokens",
        },
        "vulkan": {
            "default_policy": "NativeRequired",
            "debug_opt_in": "KILN_DECODE_HOT_PATH_DEBUG_FALLBACK=1 or KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK=1",
            "enforcement": "keeps the existing Vulkan no-generic-fallback default and routes it through FallbackPolicy",
        },
    }


def training_optimizer_fallback_policy_report() -> dict[str, Any]:
    return {
        "cpu": {
            "default_policy": "CorrectnessAllowed",
            "debug_opt_in": "not required",
            "enforcement": "CPU is the reference optimizer path",
        },
        "cuda": {
            "default_policy": "NativeRequired",
            "debug_opt_in": "KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK=1 or KILN_CUDA_TRAINING_OPTIMIZER_FALLBACK=1",
            "enforcement": "SGD/AdamW GPU training errors before kt/host fallback when native dispatch declines",
        },
        "rocm": {
            "default_policy": "NativeRequired",
            "debug_opt_in": "KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK=1 or KILN_ROCM_TRAINING_OPTIMIZER_FALLBACK=1",
            "enforcement": "SGD/AdamW GPU training errors before kt/host fallback when native dispatch declines",
        },
        "metal": {
            "default_policy": "NativeRequired",
            "debug_opt_in": "KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK=1 or KILN_METAL_TRAINING_OPTIMIZER_FALLBACK=1",
            "enforcement": "SGD/AdamW GPU training errors before kt/host fallback when native dispatch declines",
        },
        "vulkan": {
            "default_policy": "NativeRequired",
            "debug_opt_in": "KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK=1 or KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK=1",
            "enforcement": "SGD/AdamW GPU training errors before kt/host fallback when native dispatch declines",
        },
    }


def optimizer_dispatch_report(backends: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for backend, info in backends.items():
        overrides = set(info["overrides"])
        report[backend] = {
            "sgd_step": "overridden" if "dispatch_sgd_step" in overrides else "default_decline",
            "adamw_step": "overridden" if "dispatch_adamw_step" in overrides else "default_decline",
        }
    return report


def markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Backend Capability Report")
    lines.append("")
    lines.append("Generated from the live source tree by `scripts/generate_backend_capability_report.py`.")
    lines.append("")
    lines.append(f"- Branch: `{data['source']['branch']}`")
    lines.append("")
    lines.append("## Feature Fanout")
    lines.append("")
    lines.append("| Crate | CUDA | ROCm | Metal | Vulkan |")
    lines.append("|---|---|---|---|---|")
    for crate, features in data["features"].items():
        row = [crate]
        for family in FEATURE_FAMILIES:
            deps = features.get(family)
            row.append("yes" if deps is not None else "no")
        lines.append("| " + " | ".join(f"`{cell}`" if cell not in {"yes", "no"} else cell for cell in row) + " |")
    lines.append("")
    lines.append("## BackendRuntime Overrides")
    lines.append("")
    lines.append("| Backend | Source | Override Count | Support Methods | Env Gates |")
    lines.append("|---|---|---:|---:|---:|")
    for backend, info in data["backends"].items():
        lines.append(
            f"| `{backend}` | `{info['source']}` | {info['override_count']} | "
            f"{len(info['support_methods'])} | {len(info['env_gates'])} |"
        )
    lines.append("")
    lines.append("## Support Predicates")
    lines.append("")
    lines.append("| Backend | Method | Predicate Status | Support State | Paired Method | Pair Always Declines | Gates |")
    lines.append("|---|---|---|---|---|---|---|")
    for backend, info in data["backends"].items():
        for method, entry in info["support_methods"].items():
            gates = ",".join(key for key, enabled in entry["gate_hints"].items() if enabled) or "none"
            pair = entry["paired_method"] or ""
            declines = "yes" if entry["paired_method_always_declines"] else "no"
            lines.append(
                f"| `{backend}` | `{method}` | `{entry['status']}` | "
                f"`{entry['support_state']}` | `{pair}` | {declines} | {gates} |"
            )
    lines.append("")
    lines.append("## Generic DeviceOp Fallback")
    lines.append("")
    lines.append("| Backend | Policy | Counter | Evidence |")
    lines.append("|---|---|---|---|")
    for backend, info in data["fallback_policy"].items():
        lines.append(
            f"| `{backend}` | `{info['generic_device_op_fallback']}` | "
            f"`{info['counter']}` | {info['evidence']} |"
        )
    lines.append("")
    lines.append("## Decode Hot-Path Fallback")
    lines.append("")
    lines.append("| Backend | Default Policy | Debug Opt-In | Enforcement |")
    lines.append("|---|---|---|---|")
    for backend, info in data["decode_hot_path_policy"].items():
        lines.append(
            f"| `{backend}` | `{info['default_policy']}` | `{info['debug_opt_in']}` | "
            f"{info['enforcement']} |"
        )
    lines.append("")
    lines.append("## Training Optimizer Fallback")
    lines.append("")
    lines.append("| Backend | Default Policy | Debug Opt-In | Enforcement |")
    lines.append("|---|---|---|---|")
    for backend, info in data["training_optimizer_fallback_policy"].items():
        lines.append(
            f"| `{backend}` | `{info['default_policy']}` | `{info['debug_opt_in']}` | "
            f"{info['enforcement']} |"
        )
    lines.append("")
    lines.append("## Optimizer Dispatch")
    lines.append("")
    lines.append("| Backend | SGD Step | AdamW Step |")
    lines.append("|---|---|---|")
    for backend, info in data["optimizer_dispatch"].items():
        lines.append(f"| `{backend}` | `{info['sgd_step']}` | `{info['adamw_step']}` |")
    lines.append("")
    lines.append("## Mismatch Audit")
    lines.append("")
    if data["mismatches"]:
        lines.append("| Backend | Support Method | Paired Method | Lines |")
        lines.append("|---|---|---|---|")
        for item in data["mismatches"]:
            lines.append(
                f"| `{item['backend']}` | `{item['support_method']}` | `{item['paired_method']}` | "
                f"{item['support_line']} / {item['paired_line']} |"
            )
    else:
        lines.append("No literal-true support predicate currently pairs with an always-declining method body.")
    lines.append("")
    lines.append("## Backend Env Gates")
    lines.append("")
    for backend, info in data["backends"].items():
        lines.append(f"### {backend.upper()}")
        if info["env_gates"]:
            for gate in info["env_gates"]:
                lines.append(f"- `{gate}`")
        else:
            lines.append("- none detected")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    trait_methods = parse_trait_method_names(
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "mod.rs",
        "BackendRuntime",
    )
    backends, mismatches = backend_report(trait_methods)
    data = {
        "source": {
            "branch": run_git(["branch", "--show-current"]),
            "script": str(Path(__file__).relative_to(ROOT)),
        },
        "features": feature_report(),
        "trait_method_count": len(trait_methods),
        "backends": backends,
        "fallback_policy": fallback_policy_report(),
        "decode_hot_path_policy": decode_hot_path_policy_report(),
        "training_optimizer_fallback_policy": training_optimizer_fallback_policy_report(),
        "optimizer_dispatch": optimizer_dispatch_report(backends),
        "mismatches": mismatches,
    }
    REPORT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    REPORT_MD.write_text(markdown(data) + "\n")
    if "--check" in sys.argv and mismatches:
        print(json.dumps(mismatches, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
