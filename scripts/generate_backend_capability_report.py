#!/usr/bin/env python3
"""Generate backend capability reports from the live source tree."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "docs" / "backend-capability-report.md"
REPORT_JSON = ROOT / "docs" / "backend-capability-report.json"

BACKENDS = {
    "cuda": ROOT / "crates" / "kiln-model" / "src" / "backend" / "cuda.rs",
    "rocm": ROOT / "crates" / "kiln-model" / "src" / "backend" / "rocm.rs",
    "metal": ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal.rs",
    "vulkan": ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan.rs",
}

BACKEND_EXTRA_SOURCES = {
    "cuda": [
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "cuda_rocm_common.rs",
    ],
    "rocm": [
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "cuda_rocm_common.rs",
    ],
    "metal": [
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_attention.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_config.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_conv1d.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_core.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_dense.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_gdn.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_icb.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_lm_head.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_msl.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_norm.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_paged.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_pipeline.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_precompile.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_residency.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_runtime.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "metal_training.rs",
    ],
    "vulkan": [
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_attention.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_config.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_conv1d.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_decode_state.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_dense.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_device.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_gdn.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_linear.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_residency.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_resources.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_tensor_bridge.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_training.rs",
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "vulkan_weights.rs",
    ],
}

CAPABILITY_RS = ROOT / "crates" / "kiln-model" / "src" / "backend" / "capability.rs"
RESIDENCY_RS = ROOT / "crates" / "kiln-model" / "src" / "backend" / "residency.rs"

REQUEST_DESCRIPTOR_STRUCTS = [
    "AttentionRequest",
    "MatmulRequest",
    "MatmulBlasRequest",
    "LinearRequest",
    "ReplayRequest",
]

CAPABILITY_DESCRIPTOR_STRUCTS = [
    "BackendCapabilities",
    "StorageCapabilities",
    "MatmulCapabilities",
    "AttentionCapabilities",
    "GdnCapabilities",
    "DecodeCapabilities",
    "DecodeBatcherPolicy",
    "BackendTrainingCapabilities",
    "ReplayCapabilities",
    "ReplayAuthority",
    "BackendFallbackCapabilities",
]

RESIDENT_RESOURCE_DESCRIPTOR_STRUCTS = [
    "ResidentResource",
    "ResidentResourceLayout",
]

FOCUSED_BACKEND_TRAITS = [
    "BackendIdentity",
    "AttentionBackend",
    "PagedKvBackend",
    "GdnBackend",
    "ConvBackend",
    "LinearBackend",
    "SamplingBackend",
    "ResidencyBackend",
    "OptimizerBackend",
    "TrainingLossBackend",
    "ReplayBackend",
]

REPLAY_AUTHORITIES = {
    "cuda": {
        "production_authority": "model_level_runner",
        "native_primitive": "CUDA graph",
        "runner_paths": ["crates/kiln-model/src/cuda_graph.rs"],
        "graph_crate_paths": ["crates/kiln-graph-cuda/src/lib.rs"],
        "contract_paths": ["crates/kiln-graph/src/replay_plan.rs"],
        "parity_sources": ["crates/kiln-model/src/forward.rs"],
        "parity_tests": ["test_cuda_graph_bs1_decode_matches_eager"],
    },
    "rocm": {
        "production_authority": "model_level_runner",
        "native_primitive": "HIP graph",
        "runner_paths": ["crates/kiln-model/src/rocm_graph.rs"],
        "graph_crate_paths": [],
        "contract_paths": [
            "crates/kiln-graph/src/replay_plan.rs",
            "crates/kiln-tensor/tests/rocm_capture_arena.rs",
        ],
        "parity_sources": ["crates/kiln-model/src/rocm_graph.rs"],
        "parity_tests": ["ROCm graph runner byte-identical eager/replay source contract"],
    },
    "metal": {
        "production_authority": "model_level_runner_with_graph_crate_replay_object",
        "native_primitive": "Metal ICB",
        "runner_paths": [
            "crates/kiln-model/src/metal_graph.rs",
            "crates/kiln-model/src/backend/metal_paged.rs",
        ],
        "graph_crate_paths": ["crates/kiln-graph-metal/src/lib.rs"],
        "contract_paths": ["crates/kiln-graph/src/replay_plan.rs"],
        "parity_sources": [
            "crates/kiln-model/src/forward.rs",
            "crates/kiln-model/src/backend/metal_paged.rs",
        ],
        "parity_tests": [
            "test_metal_graph_bs1_decode_matches_eager_across_boundaries_and_buckets",
            "test_metal_graph_batched_decode_matches_eager_and_replays_bucket",
            "single_token_paged_decode_icb_matches_eager_and_updates_slot",
            "batched_paged_decode_icb_matches_eager_and_updates_slots",
        ],
    },
    "vulkan": {
        "production_authority": "resident_decode_command_batch",
        "native_primitive": "Vulkan CommandBatch",
        "runner_paths": [
            "crates/kiln-model/src/vk_decode_resident.rs",
            "crates/kiln-vulkan-kernel/src/cmd_batch.rs",
        ],
        "graph_crate_paths": ["crates/kiln-graph-vulkan/src/lib.rs"],
        "contract_paths": ["crates/kiln-graph/src/replay_plan.rs"],
        "parity_sources": ["crates/kiln-model/tests/vk_resident_decode_parity.rs"],
        "parity_tests": ["vk_resident_decode_matches_nonresident_on_qwen35_4b"],
    },
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


def source_branch() -> str:
    return (
        os.environ.get("GITHUB_HEAD_REF")
        or os.environ.get("GITHUB_REF_NAME")
        or run_git(["branch", "--show-current"])
        or "unknown"
    )


def load_toml(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)
    return load_toml_feature_subset(path)


def strip_toml_comment(line: str) -> str:
    in_string = False
    escape = False
    for i, ch in enumerate(line):
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif ch == "#" and not in_string:
            return line[:i]
    return line


def parse_toml_string_array(value: str) -> list[str]:
    return [
        bytes(match.group(1), "utf-8").decode("unicode_escape")
        for match in re.finditer(r'"((?:[^"\\]|\\.)*)"', value, flags=re.S)
    ]


def load_toml_feature_subset(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"package": {}, "features": {}}
    section = ""
    pending_feature: str | None = None
    pending_value = ""
    bracket_depth = 0

    for raw_line in path.read_text().splitlines():
        line = strip_toml_comment(raw_line).strip()
        if not line:
            continue

        if pending_feature is not None:
            pending_value += "\n" + line
            bracket_depth += line.count("[") - line.count("]")
            if bracket_depth <= 0:
                data["features"][pending_feature] = parse_toml_string_array(pending_value)
                pending_feature = None
                pending_value = ""
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            continue

        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        key = key.strip('"')

        if section == "package" and key == "name":
            names = parse_toml_string_array(value)
            if names:
                data["package"]["name"] = names[0]
        elif section == "features":
            bracket_depth = value.count("[") - value.count("]")
            if bracket_depth > 0:
                pending_feature = key
                pending_value = value
            else:
                data["features"][key] = parse_toml_string_array(value)

    return data


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
        elif ch == "'" and not (nxt.isalpha() or nxt == "_"):
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


def parse_backend_functions(source_paths: list[Path]) -> dict[str, FunctionDef]:
    functions: dict[str, FunctionDef] = {}
    for path in source_paths:
        functions.update(parse_functions(path))
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


def parse_pub_struct_fields(path: Path, struct_names: list[str]) -> dict[str, list[dict[str, str]]]:
    text = path.read_text()
    report: dict[str, list[dict[str, str]]] = {}
    for struct_name in struct_names:
        match = re.search(rf"\bpub\s+struct\s+{re.escape(struct_name)}\b", text)
        if not match:
            raise ValueError(f"{struct_name} struct not found in {path}")
        brace = text.find("{", match.end())
        if brace == -1:
            raise ValueError(f"{struct_name} body not found in {path}")
        end = find_matching_brace(text, brace)
        body = text[brace + 1 : end]
        fields = []
        for field_match in re.finditer(
            r"\bpub\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,\n]+)", body
        ):
            fields.append(
                {
                    "name": field_match.group(1),
                    "type": field_match.group(2).strip(),
                }
            )
        report[struct_name] = fields
    return report


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


def backend_source_paths(backend: str, main_path: Path) -> list[Path]:
    return [main_path, *BACKEND_EXTRA_SOURCES.get(backend, [])]


def read_backend_sources(backend: str, main_path: Path) -> str:
    return "\n".join(path.read_text() for path in backend_source_paths(backend, main_path))


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


def request_descriptor_report() -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    for name, fields in parse_pub_struct_fields(CAPABILITY_RS, REQUEST_DESCRIPTOR_STRUCTS).items():
        field_names = [field["name"] for field in fields]
        descriptors[name] = {
            "source": str(CAPABILITY_RS.relative_to(ROOT)),
            "field_count": len(fields),
            "fields": fields,
            "has_dtype": any("dtype" in field_name for field_name in field_names),
            "has_shape": any("shape" in field_name for field_name in field_names)
            or all(dim in field_names for dim in ["m", "n", "k"]),
            "has_layout": any("layout" in field_name for field_name in field_names),
            "has_batch": any("batch" in field_name for field_name in field_names),
            "has_replay_safe": "replay_safe" in field_names,
        }
    return descriptors


def capability_descriptor_report() -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    for name, fields in parse_pub_struct_fields(
        CAPABILITY_RS, CAPABILITY_DESCRIPTOR_STRUCTS
    ).items():
        descriptors[name] = {
            "source": str(CAPABILITY_RS.relative_to(ROOT)),
            "field_count": len(fields),
            "fields": fields,
        }
    return descriptors


def focused_backend_facet_report() -> dict[str, Any]:
    backend_mod = ROOT / "crates" / "kiln-model" / "src" / "backend" / "mod.rs"
    source = backend_mod.read_text()
    report: dict[str, Any] = {}
    for trait_name in FOCUSED_BACKEND_TRAITS:
        methods = sorted(parse_trait_method_names(backend_mod, trait_name))
        blanket_impl = (
            f"impl<T: BackendRuntime + ?Sized> {trait_name} for T" in source
            or f"impl<T> {trait_name} for T" in source
        )
        report[trait_name] = {
            "source": str(backend_mod.relative_to(ROOT)),
            "method_count": len(methods),
            "methods": methods,
            "forwarding_impl": "blanket_backend_runtime" if blanket_impl else "missing",
        }
    return report


def existing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if path_exists(path)]


def missing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not path_exists(path)]


def replay_authority_report() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for backend, info in REPLAY_AUTHORITIES.items():
        evidence = list(dict.fromkeys([
            *info["runner_paths"],
            *info["graph_crate_paths"],
            *info["contract_paths"],
            *info["parity_sources"],
        ]))
        report[backend] = {
            **info,
            "evidence_present": existing_paths(evidence),
            "evidence_missing": missing_paths(evidence),
        }
    return report


def resident_resource_descriptor_report() -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    for name, fields in parse_pub_struct_fields(
        RESIDENCY_RS, RESIDENT_RESOURCE_DESCRIPTOR_STRUCTS
    ).items():
        descriptors[name] = {
            "source": str(RESIDENCY_RS.relative_to(ROOT)),
            "field_count": len(fields),
            "fields": fields,
        }
    return descriptors


def backend_report(trait_methods: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for backend, path in BACKENDS.items():
        source_paths = backend_source_paths(backend, path)
        functions = parse_backend_functions(source_paths)
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
        source = read_backend_sources(backend, path)
        report[backend] = {
            "source": str(path.relative_to(ROOT)),
            "source_modules": [str(source_path.relative_to(ROOT)) for source_path in source_paths],
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


def training_precision_policy_report() -> dict[str, Any]:
    return {
        "cpu": {
            "name": "cpu_f32_reference",
            "activation_dtypes": ["F32"],
            "base_weight_dtypes": ["F32"],
            "lora_parameter_dtypes": ["F32"],
            "loss_accumulation_dtype": "F32",
            "optimizer_parameter_dtypes": ["F32"],
            "mixed_precision": False,
            "notes": "CPU reference training uses F32 tensors and portable optimizer math.",
        },
        "cuda": {
            "name": "cuda_native_float",
            "activation_dtypes": ["F32", "BF16", "F16"],
            "base_weight_dtypes": ["F32", "BF16", "F16"],
            "lora_parameter_dtypes": ["F32", "BF16"],
            "loss_accumulation_dtype": "F32",
            "optimizer_parameter_dtypes": ["F32", "BF16"],
            "mixed_precision": True,
            "notes": "CUDA keeps kt tape authoritative and routes BF16/F16/F32 leaves through CUDA-native kernels where available.",
        },
        "rocm": {
            "name": "rocm_native_float",
            "activation_dtypes": ["F32", "BF16", "F16"],
            "base_weight_dtypes": ["F32", "BF16", "F16"],
            "lora_parameter_dtypes": ["F32", "BF16"],
            "loss_accumulation_dtype": "F32",
            "optimizer_parameter_dtypes": ["F32", "BF16"],
            "mixed_precision": True,
            "notes": "ROCm mirrors CUDA's kt-tape dtype envelope while dispatching through HIP/hipBLASLt-native leaves where available.",
        },
        "metal": {
            "name": "metal_bf16_uma",
            "activation_dtypes": ["BF16"],
            "base_weight_dtypes": ["BF16"],
            "lora_parameter_dtypes": ["F32", "BF16"],
            "loss_accumulation_dtype": "F32",
            "optimizer_parameter_dtypes": ["F32", "BF16"],
            "mixed_precision": True,
            "notes": "Metal training is BF16-focused on UMA buffers, with F32 loss accumulation and F32/BF16 AdamW residency.",
        },
        "vulkan": {
            "name": "vulkan_mixed_f32_bf16",
            "activation_dtypes": ["F32"],
            "base_weight_dtypes": ["F32", "BF16"],
            "lora_parameter_dtypes": ["F32"],
            "loss_accumulation_dtype": "F32",
            "optimizer_parameter_dtypes": ["F32", "BF16"],
            "mixed_precision": True,
            "notes": "Vulkan keeps training activations and LoRA parameters F32 while allowing BF16 base weights through explicit VkTensor buffer bridges.",
        },
    }


def path_exists(path: str) -> bool:
    return (ROOT / path).exists()


def conformance_gate_report() -> list[dict[str, Any]]:
    gates = [
        {
            "gate": "storage_round_trip",
            "phase8_requirement": "storage round trip",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-tensor rocm_storage_smoke",
            "evidence": [
                "crates/kiln-tensor/tests/rocm_storage_smoke.rs",
                "crates/kiln-vulkan-kernel/tests/vk_tensor_parity.rs",
            ],
        },
        {
            "gate": "host_transfer_to_device_parity",
            "phase8_requirement": "host transfer / to_device parity with explicit unsupported errors",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-tensor device_transfer_support_classifies_explicit_transitions && /home/ericflo/.cargo/bin/cargo test -p kiln-tensor to_device_without_gpu_features_reports_explicit_unsupported_transition && /home/ericflo/.cargo/bin/cargo test -p kiln-tensor cuda_resize_copy_primitives",
            "evidence": [
                "crates/kiln-tensor/src/tensor.rs",
                "crates/kiln-tensor/tests/cuda_resize_copy_primitives.rs",
                "crates/kiln-tensor/tests/metal_ops_parity.rs",
                "crates/kiln-tensor/tests/rocm_compare_parity.rs",
                "crates/kiln-vulkan-kernel/tests/vk_tensor_parity.rs",
            ],
        },
        {
            "gate": "device_op_parity",
            "phase8_requirement": "DeviceOp parity",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-tensor device_op::tests",
            "evidence": [
                "crates/kiln-tensor/src/device_op.rs",
                "crates/kiln-tensor/tests/rocm_scalar_op_parity.rs",
                "crates/kiln-tensor/tests/metal_ops_parity.rs",
            ],
        },
        {
            "gate": "matmul_linear_parity",
            "phase8_requirement": "matmul/linear parity",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-model matmul_request_projects_to_blas_shape_contract && /home/ericflo/.cargo/bin/cargo test -p kiln-tensor rocm_matmul_parity && /home/ericflo/.cargo/bin/cargo test -p kiln-tensor matmul_matrix_core && /home/ericflo/.cargo/bin/cargo test -p kiln-vulkan-kernel vk_matmul && /home/ericflo/.cargo/bin/cargo test -p kiln-vulkan-kernel linear_decode && /home/ericflo/.cargo/bin/cargo test -p kiln-model tape_forward_matmul_bit_exact_parity_with_baseline && /home/ericflo/.cargo/bin/cargo test -p kiln-blas cublaslt_handle_smoke && /home/ericflo/.cargo/bin/cargo test -p kiln-model --test backend_capability_contract",
            "evidence": [
                "crates/kiln-model/src/backend/capability.rs",
                "crates/kiln-model/src/backend/mod.rs",
                "crates/kiln-blas/tests/cublaslt_handle_smoke.rs",
                "crates/kiln-tensor/tests/rocm_matmul_parity.rs",
                "crates/kiln-tensor/tests/metal_ops_parity.rs",
                "crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs",
                "crates/kiln-vulkan-kernel/tests/linear_decode_argmax.rs",
                "crates/kiln-vulkan-kernel/tests/linear_decode_sample.rs",
                "crates/kiln-model/tests/tape_forward_parity.rs",
                "crates/kiln-model/tests/marlin_qproj_parity.rs",
            ],
        },
        {
            "gate": "attention_gdn_conv_parity",
            "phase8_requirement": "attention/GDN/conv parity",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-model rocm_flash_attn_bwd_gradcheck",
            "evidence": [
                "crates/kiln-flash-attn/tests/rocm_flash_attn_parity.rs",
                "crates/kiln-gdn-kernel/tests/rocm_gdn_parity.rs",
                "crates/kiln-conv1d-kernel/tests/rocm_conv1d_parity.rs",
                "crates/kiln-vulkan-kernel/tests/vk_attention_parity.rs",
                "crates/kiln-vulkan-kernel/tests/vk_gdn_foundation_parity.rs",
            ],
        },
        {
            "gate": "optimizer_parity",
            "phase8_requirement": "optimizer parity",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-optim --test integration && /home/ericflo/.cargo/bin/cargo test -p kiln-train training_optimizer && /home/ericflo/.cargo/bin/cargo test -p kiln-model --test backend_capability_contract",
            "evidence": [
                "crates/kiln-optim/tests/integration.rs",
                "crates/kiln-model/src/backend/cuda.rs",
                "crates/kiln-model/src/backend/rocm.rs",
                "crates/kiln-model/src/backend/metal_training.rs",
                "crates/kiln-model/src/backend/vulkan.rs",
                "crates/kiln-model/src/backend/vulkan_training.rs",
                "crates/kiln-model/src/backend/mod.rs",
                "crates/kiln-train/src/trainer.rs",
                "crates/kiln-train/tests/vk_cuda_opd_parity.rs",
            ],
        },
        {
            "gate": "replay_parity",
            "phase8_requirement": "replay parity",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-graph replay && /home/ericflo/.cargo/bin/cargo test -p kiln-graph-cuda replay && /home/ericflo/.cargo/bin/cargo test -p kiln-graph-metal replay && /home/ericflo/.cargo/bin/cargo test -p kiln-graph-vulkan replay && /home/ericflo/.cargo/bin/cargo test -p kiln-model --test backend_capability_contract",
            "evidence": [
                "crates/kiln-graph/src/replay_plan.rs",
                "crates/kiln-graph/src/captured_graph.rs",
                "crates/kiln-graph/tests/capture_lifetime.rs",
                "crates/kiln-graph-cuda/src/lib.rs",
                "crates/kiln-graph-metal/src/lib.rs",
                "crates/kiln-graph-vulkan/src/lib.rs",
                "crates/kiln-model/src/backend/capability.rs",
                "crates/kiln-model/src/backend/residency.rs",
                "crates/kiln-model/tests/vk_resident_decode_parity.rs",
                "crates/kiln-tensor/tests/rocm_capture_arena.rs",
            ],
        },
        {
            "gate": "one_step_training_proof",
            "phase8_requirement": "one-step training proof",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-model cuda_sft_step_proof && /home/ericflo/.cargo/bin/cargo test -p kiln-model metal_sft_step_proof && /home/ericflo/.cargo/bin/cargo test -p kiln-model vk_sft_step_proof && /home/ericflo/.cargo/bin/cargo test -p kiln-model rocm_sft_step_proof && /home/ericflo/.cargo/bin/cargo test -p kiln-optim --test end_to_end_training && /home/ericflo/.cargo/bin/cargo test -p kiln-model --test backend_capability_contract",
            "evidence": [
                "crates/kiln-model/tests/cuda_sft_step_proof.rs",
                "crates/kiln-model/tests/metal_sft_step_proof.rs",
                "crates/kiln-model/tests/vk_sft_step_proof.rs",
                "crates/kiln-model/tests/rocm_sft_step_proof.rs",
                "crates/kiln-optim/tests/end_to_end_training.rs",
            ],
        },
        {
            "gate": "no_unexpected_host_fallback",
            "phase8_requirement": "no unexpected host fallback in decode/training hot paths",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-tensor device_op_host_fallback_counts_are_backend_and_arity_specific",
            "evidence": [
                "crates/kiln-tensor/src/device_op.rs",
                "crates/kiln-model/src/generate.rs",
                "crates/kiln-train/src/trainer.rs",
            ],
        },
        {
            "gate": "decode_submit_or_replay_count",
            "phase8_requirement": "max submit count or replay count per decode token",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-model decode_batcher_stats_report_runner_calls_per_token && /home/ericflo/.cargo/bin/cargo test -p kiln-server test_metrics_render && /home/ericflo/.cargo/bin/cargo test -p kiln-graph replay",
            "evidence": [
                "crates/kiln-model/src/generate.rs",
                "crates/kiln-server/src/metrics.rs",
                "crates/kiln-server/src/api/health.rs",
                "crates/kiln-server/src/api/debug_model_state.rs",
                "crates/kiln-graph/src/captured_graph.rs",
                "crates/kiln-graph/src/replay_plan.rs",
            ],
        },
        {
            "gate": "matmul_algorithm_cache_reporting",
            "phase8_requirement": "matmul algorithm/cache hit reporting",
            "status": "covered",
            "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-blas cache_stats_reports_entries_and_hit_rate && /home/ericflo/.cargo/bin/cargo test -p kiln-rocblas cache_stats_reports_entries_and_hit_rate && CUDARC_CUDA_VERSION=12080 /home/ericflo/.cargo/bin/cargo check -p kiln-blas --features cublaslt --tests && /home/ericflo/.cargo/bin/cargo check -p kiln-rocblas --features hipblaslt --tests",
            "evidence": [
                "crates/kiln-blas/src/algo_cache.rs",
                "crates/kiln-blas/src/cublaslt_handle.rs",
                "crates/kiln-blas/tests/cublaslt_handle_smoke.rs",
                "crates/kiln-rocblas/src/algo_cache.rs",
                "crates/kiln-rocblas/src/hipblaslt_handle.rs",
            ],
        },
        {
            "gate": "hardware_latency_thresholds",
            "phase8_requirement": "backend-specific latency thresholds on known hardware fixtures",
            "status": "fixture_required",
            "command": "python3 scripts/run_backend_latency_fixture.py --self-test && python3 scripts/write_backend_latency_result_artifact.py --self-test && python3 scripts/lock_backend_latency_thresholds.py --self-test && python3 scripts/check_backend_latency_fixtures.py --self-test && hardware runner required; python3 scripts/check_backend_latency_fixtures.py docs/backend-latency-fixtures.json --require-covered",
            "evidence": [
                "docs/backend-latency-fixtures.json",
                "docs/backend-latency-result-schema.md",
                "scripts/run_backend_latency_fixture.py",
                "scripts/write_backend_latency_result_artifact.py",
                "scripts/lock_backend_latency_thresholds.py",
                "scripts/check_backend_latency_fixtures.py",
                "crates/kiln-server/examples/flce_preflight_bench.rs",
                "crates/kiln-server/examples/flce_phase_a_validation_bench.rs",
                "crates/kiln-tensor/tests/metal_matmul_bench.rs",
                "crates/kiln-tensor/tests/metal_sdpa_bench.rs",
                "crates/kiln-vulkan-kernel/src/bin/vulkan_decode_microbench.rs",
                "crates/kiln-tensor/tests/rocm_latency_bench.rs",
            ],
        },
        {
            "gate": "generated_capability_dashboard",
            "phase8_requirement": "generated capability dashboard checked into docs or build artifacts",
            "status": "covered",
            "command": "python3 scripts/generate_backend_capability_report.py --self-test && python3 scripts/generate_backend_capability_report.py --check",
            "evidence": [
                "docs/backend-capability-report.md",
                "docs/backend-capability-report.json",
                "scripts/generate_backend_capability_report.py",
            ],
        },
    ]

    for gate in gates:
        gate["evidence_present"] = [
            evidence for evidence in gate["evidence"] if path_exists(evidence)
        ]
        gate["evidence_missing"] = [
            evidence for evidence in gate["evidence"] if not path_exists(evidence)
        ]
    return gates


def migration_phase_status_report(conformance_gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gate_statuses = {gate["gate"]: gate["status"] for gate in conformance_gates}
    phase8_status = (
        "covered"
        if all(status == "covered" for status in gate_statuses.values())
        else "fixture_required"
        if gate_statuses.get("hardware_latency_thresholds") == "fixture_required"
        and all(
            status == "covered"
            for gate, status in gate_statuses.items()
            if gate != "hardware_latency_thresholds"
        )
        else "partial"
    )
    phases = [
        {
            "phase": 0,
            "title": "Audit and stabilize capability reporting",
            "status": "covered",
            "deliverables": [
                "generated Markdown and JSON capability report",
                "feature fanout, override, support predicate, env gate, and fallback audit",
                "literal-true support predicate mismatch guard",
                "stale naming and graph authority clarification evidence",
            ],
            "evidence": [
                "docs/backend-engine-unification-plan.md",
                "scripts/generate_backend_capability_report.py",
                "docs/backend-capability-report.md",
                "docs/backend-capability-report.json",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Feature Fanout",
                "BackendRuntime Overrides",
                "Support Predicates",
                "Replay Authority",
                "Mismatch Audit",
            ],
            "remaining": [],
        },
        {
            "phase": 1,
            "title": "Introduce focused backend traits",
            "status": "covered",
            "deliverables": [
                "focused backend trait family",
                "BackendRuntime compatibility facade",
                "focused facet forwarding evidence",
                "call-site contracts against broad facade regressions",
            ],
            "evidence": [
                "crates/kiln-model/src/backend/mod.rs",
                "scripts/generate_backend_capability_report.py",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Focused Backend Facets",
                "Request Capability Queries",
            ],
            "remaining": [],
        },
        {
            "phase": 2,
            "title": "Normalize fallback policy",
            "status": "covered",
            "deliverables": [
                "typed fallback policy per backend and mode",
                "host fallback counters for non-CUDA bring-up paths",
                "decode and training hot-path native-required guards",
                "CPU/correctness fallback observability",
            ],
            "evidence": [
                "crates/kiln-tensor/src/device_op.rs",
                "crates/kiln-model/src/generate.rs",
                "crates/kiln-train/src/trainer.rs",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Generic DeviceOp Fallback",
                "Decode Hot-Path Fallback",
                "Training Optimizer Fallback",
            ],
            "remaining": [],
        },
        {
            "phase": 3,
            "title": "Unify resident resource semantics",
            "status": "covered",
            "deliverables": [
                "ResidentResource and ResidentRegistry descriptors",
                "backend-specific resident resource wrappers",
                "shared lifecycle state and replay-stability metadata",
                "focused residency call-site contracts",
            ],
            "evidence": [
                "crates/kiln-model/src/backend/residency.rs",
                "crates/kiln-model/src/backend/metal_residency.rs",
                "crates/kiln-model/src/backend/vulkan_residency.rs",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Resident Resource Descriptors",
                "Focused Backend Facets",
            ],
            "remaining": [],
        },
        {
            "phase": 4,
            "title": "Unify matmul and linear dispatch",
            "status": "covered",
            "deliverables": [
                "MatmulRequest and LinearRequest descriptors",
                "BLASLt request projection shared by CUDA and ROCm",
                "Metal and Vulkan request/capability evidence",
                "matmul/linear parity and algorithm-cache gates",
            ],
            "evidence": [
                "crates/kiln-model/src/backend/capability.rs",
                "crates/kiln-model/src/backend/mod.rs",
                "crates/kiln-blas/src/cublaslt_handle.rs",
                "crates/kiln-rocblas/src/hipblaslt_handle.rs",
                "crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Typed Request Descriptors",
                "Request Capability Queries",
                "Conformance And Performance Gates",
            ],
            "remaining": [],
        },
        {
            "phase": 5,
            "title": "Move replay into the authoritative graph layer",
            "status": "covered",
            "deliverables": [
                "ReplayBackend focused facet",
                "shared replay key and replay authority descriptor",
                "CUDA/HIP graph, Metal ICB, and Vulkan CommandBatch authority evidence",
                "eager-vs-replay parity gates",
            ],
            "evidence": [
                "crates/kiln-graph/src/replay_plan.rs",
                "crates/kiln-graph-cuda/src/lib.rs",
                "crates/kiln-graph-metal/src/lib.rs",
                "crates/kiln-graph-vulkan/src/lib.rs",
                "crates/kiln-model/src/cuda_graph.rs",
                "crates/kiln-model/src/rocm_graph.rs",
                "crates/kiln-model/src/metal_graph.rs",
                "crates/kiln-vulkan-kernel/src/cmd_batch.rs",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Focused Backend Facets",
                "Replay Authority",
                "Conformance And Performance Gates",
            ],
            "remaining": [],
        },
        {
            "phase": 6,
            "title": "Finish shared training integration",
            "status": "covered",
            "deliverables": [
                "SFT/GRPO/OPD policy routed through focused capability surfaces",
                "TrainingLossBackend and OptimizerBackend evidence",
                "explicit backend training precision policy",
                "per-backend one-step training proof gates",
            ],
            "evidence": [
                "crates/kiln-train/src/trainer.rs",
                "crates/kiln-train/src/sft_tape_shim.rs",
                "crates/kiln-train/src/grpo_tape_shim.rs",
                "crates/kiln-train/src/opd_tape_shim.rs",
                "crates/kiln-model/src/backend/metal_training.rs",
                "crates/kiln-model/src/backend/vulkan_training.rs",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Training Optimizer Fallback",
                "Training Precision Policy",
                "Optimizer Dispatch",
                "Conformance And Performance Gates",
            ],
            "remaining": [],
        },
        {
            "phase": 7,
            "title": "Decompose backend modules",
            "status": "covered",
            "deliverables": [
                "Metal split by operation family and runtime concern",
                "Vulkan split around explicit-resource boundaries",
                "CUDA/ROCm common helper factoring",
                "backend-native platform differences preserved",
            ],
            "evidence": [
                "crates/kiln-model/src/backend/metal.rs",
                "crates/kiln-model/src/backend/metal_attention.rs",
                "crates/kiln-model/src/backend/metal_gdn.rs",
                "crates/kiln-model/src/backend/metal_residency.rs",
                "crates/kiln-model/src/backend/metal_training.rs",
                "crates/kiln-model/src/backend/vulkan.rs",
                "crates/kiln-model/src/backend/vulkan_residency.rs",
                "crates/kiln-model/src/backend/vulkan_tensor_bridge.rs",
                "crates/kiln-model/src/backend/cuda_rocm_common.rs",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "BackendRuntime Overrides",
                "Focused Backend Facets",
            ],
            "remaining": [],
        },
        {
            "phase": 8,
            "title": "Conformance and performance gates",
            "status": phase8_status,
            "deliverables": [
                "backend conformance suite",
                "performance sentinel suite",
                "checked-in generated capability dashboard",
                "hardware latency fixture manifest and result schema",
            ],
            "evidence": [
                "docs/backend-capability-report.md",
                "docs/backend-capability-report.json",
                "docs/backend-latency-fixtures.json",
                "docs/backend-latency-result-schema.md",
                "scripts/run_backend_latency_fixture.py",
                "scripts/write_backend_latency_result_artifact.py",
                "scripts/lock_backend_latency_thresholds.py",
                "scripts/check_backend_latency_fixtures.py",
                "scripts/generate_backend_capability_report.py",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Conformance And Performance Gates",
            ],
            "remaining": []
            if phase8_status == "covered"
            else [
                "hardware_latency_thresholds remains fixture_required until real known-hardware result artifacts satisfy --require-covered",
            ],
        },
    ]
    for phase in phases:
        phase["evidence_present"] = existing_paths(phase["evidence"])
        phase["evidence_missing"] = missing_paths(phase["evidence"])
        if phase["evidence_missing"] and phase["status"] == "covered":
            phase["status"] = "partial"
            phase["remaining"] = [
                *phase["remaining"],
                "missing source evidence listed in evidence_missing",
            ]
    return phases


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
    lines.append("## Migration Phase Status")
    lines.append("")
    lines.append("| Phase | Title | Status | Evidence | Remaining |")
    lines.append("|---|---|---|---|---|")
    for phase in data["migration_phase_status"]:
        evidence = ", ".join(f"`{path}`" for path in phase["evidence_present"]) or "none"
        remaining = "; ".join(phase["remaining"]) or "none"
        lines.append(
            f"| Phase {phase['phase']} | {phase['title']} | `{phase['status']}` | "
            f"{evidence} | {remaining} |"
        )
    lines.append("")
    lines.append("## BackendRuntime Overrides")
    lines.append("")
    lines.append("| Backend | Source Modules | Override Count | Support Methods | Env Gates |")
    lines.append("|---|---|---:|---:|---:|")
    for backend, info in data["backends"].items():
        sources = ", ".join(f"`{source}`" for source in info.get("source_modules", [info["source"]]))
        lines.append(
            f"| `{backend}` | {sources} | {info['override_count']} | "
            f"{len(info['support_methods'])} | {len(info['env_gates'])} |"
        )
    lines.append("")
    lines.append("## Focused Backend Facets")
    lines.append("")
    lines.append("| Facet | Method Count | Forwarding Impl | Methods |")
    lines.append("|---|---:|---|---|")
    for name, info in data["focused_backend_facets"].items():
        methods = ", ".join(f"`{method}`" for method in info["methods"])
        lines.append(
            f"| `{name}` | {info['method_count']} | `{info['forwarding_impl']}` | {methods} |"
        )
    lines.append("")
    lines.append("## Replay Authority")
    lines.append("")
    lines.append(
        "| Backend | Production Authority | Native Primitive | Runners | Graph Crates | Parity Tests | Missing Evidence |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for backend, info in data["replay_authority"].items():
        runners = ", ".join(f"`{path}`" for path in info["runner_paths"]) or "none"
        graph_crates = ", ".join(f"`{path}`" for path in info["graph_crate_paths"]) or "none"
        tests = ", ".join(f"`{test}`" for test in info["parity_tests"]) or "none"
        missing = ", ".join(f"`{path}`" for path in info["evidence_missing"]) or "none"
        lines.append(
            f"| `{backend}` | `{info['production_authority']}` | `{info['native_primitive']}` | "
            f"{runners} | {graph_crates} | {tests} | {missing} |"
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
    lines.append("## Typed Request Descriptors")
    lines.append("")
    lines.append(
        "| Descriptor | Field Count | DType | Shape | Layout | Batch | Replay Safe | Fields |"
    )
    lines.append("|---|---:|---|---|---|---|---|---|")
    for name, info in data["request_descriptors"].items():
        fields = ", ".join(f"`{field['name']}`" for field in info["fields"])
        lines.append(
            f"| `{name}` | {info['field_count']} | "
            f"{'yes' if info['has_dtype'] else 'no'} | "
            f"{'yes' if info['has_shape'] else 'no'} | "
            f"{'yes' if info['has_layout'] else 'no'} | "
            f"{'yes' if info['has_batch'] else 'no'} | "
            f"{'yes' if info['has_replay_safe'] else 'no'} | {fields} |"
        )
    lines.append("")
    lines.append("## Request Capability Queries")
    lines.append("")
    for method in data["request_capability_queries"]:
        lines.append(f"- `{method}`")
    lines.append("")
    lines.append("## Typed Capability Descriptors")
    lines.append("")
    lines.append("| Descriptor | Field Count | Fields |")
    lines.append("|---|---:|---|")
    for name, info in data["capability_descriptors"].items():
        fields = ", ".join(f"`{field['name']}`" for field in info["fields"])
        lines.append(f"| `{name}` | {info['field_count']} | {fields} |")
    lines.append("")
    lines.append("## Resident Resource Descriptors")
    lines.append("")
    lines.append("| Descriptor | Field Count | Fields |")
    lines.append("|---|---:|---|")
    for name, info in data["resident_resource_descriptors"].items():
        fields = ", ".join(f"`{field['name']}`" for field in info["fields"])
        lines.append(f"| `{name}` | {info['field_count']} | {fields} |")
    lines.append("")
    lines.append("## Conformance And Performance Gates")
    lines.append("")
    lines.append("| Gate | Phase 8 Requirement | Status | Command | Evidence | Missing Evidence |")
    lines.append("|---|---|---|---|---|---|")
    for gate in data["conformance_gates"]:
        evidence = ", ".join(f"`{path}`" for path in gate["evidence_present"]) or "none"
        missing = ", ".join(f"`{path}`" for path in gate["evidence_missing"]) or "none"
        lines.append(
            f"| `{gate['gate']}` | {gate['phase8_requirement']} | "
            f"`{gate['status']}` | `{gate['command']}` | {evidence} | {missing} |"
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
    lines.append("## Training Precision Policy")
    lines.append("")
    lines.append("| Backend | Policy | Activations | Base Weights | LoRA | Loss Accum | Optimizer Params | Mixed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for backend, info in data["training_precision_policy"].items():
        lines.append(
            f"| `{backend}` | `{info['name']}` | "
            f"`{','.join(info['activation_dtypes'])}` | "
            f"`{','.join(info['base_weight_dtypes'])}` | "
            f"`{','.join(info['lora_parameter_dtypes'])}` | "
            f"`{info['loss_accumulation_dtype']}` | "
            f"`{','.join(info['optimizer_parameter_dtypes'])}` | "
            f"{'yes' if info['mixed_precision'] else 'no'} |"
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


def report_outputs(data: dict[str, Any]) -> tuple[str, str]:
    return (
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        markdown(data) + "\n",
    )


def write_report_files(json_text: str, markdown_text: str) -> None:
    REPORT_JSON.write_text(json_text)
    REPORT_MD.write_text(markdown_text)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_report_files(
    expected_json: str,
    expected_markdown: str,
    json_path: Path = REPORT_JSON,
    markdown_path: Path = REPORT_MD,
) -> list[str]:
    failures: list[str] = []
    for label, path, expected in [
        ("JSON", json_path, expected_json),
        ("Markdown", markdown_path, expected_markdown),
    ]:
        if not path.exists():
            failures.append(f"{label} report is missing: {display_path(path)}")
            continue
        actual = path.read_text()
        if actual != expected:
            failures.append(
                f"{label} report is stale: run scripts/generate_backend_capability_report.py"
            )
    return failures


def run_self_test() -> int:
    expected_json = '{"status": "fresh"}\n'
    expected_markdown = "# Fresh\n"
    with TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        json_path = tmp_root / "backend-capability-report.json"
        markdown_path = tmp_root / "backend-capability-report.md"

        missing = check_report_files(
            expected_json,
            expected_markdown,
            json_path=json_path,
            markdown_path=markdown_path,
        )
        if len(missing) != 2 or not all("missing" in failure for failure in missing):
            print(f"missing-report self-test failed: {missing}", file=sys.stderr)
            return 1

        json_path.write_text(expected_json)
        markdown_path.write_text(expected_markdown)
        fresh = check_report_files(
            expected_json,
            expected_markdown,
            json_path=json_path,
            markdown_path=markdown_path,
        )
        if fresh:
            print(f"fresh-report self-test failed: {fresh}", file=sys.stderr)
            return 1

        markdown_path.write_text("# Stale\n")
        before = markdown_path.read_text()
        stale = check_report_files(
            expected_json,
            expected_markdown,
            json_path=json_path,
            markdown_path=markdown_path,
        )
        after = markdown_path.read_text()
        if len(stale) != 1 or "Markdown report is stale" not in stale[0]:
            print(f"stale-report self-test failed: {stale}", file=sys.stderr)
            return 1
        if before != after:
            print("--check helper rewrote report files during self-test", file=sys.stderr)
            return 1

        manifest_path = tmp_root / "Cargo.toml"
        manifest_path.write_text(
            '\n'.join(
                [
                    "[package]",
                    'name = "fallback-sample"',
                    "",
                    "[features]",
                    'cuda = ["dep:cudarc", "kiln-tensor/cuda"]',
                    "rocm = [",
                    '  "dep:kiln-hip", # comment outside the string',
                    '  "kiln-tensor/rocm"',
                    "]",
                    'metal = []',
                    'vulkan = ["kiln-vulkan-kernel"]',
                    "",
                ]
            )
        )
        fallback_toml = load_toml_feature_subset(manifest_path)
        expected_features = {
            "cuda": ["dep:cudarc", "kiln-tensor/cuda"],
            "rocm": ["dep:kiln-hip", "kiln-tensor/rocm"],
            "metal": [],
            "vulkan": ["kiln-vulkan-kernel"],
        }
        if fallback_toml.get("package", {}).get("name") != "fallback-sample":
            print(f"fallback TOML package self-test failed: {fallback_toml}", file=sys.stderr)
            return 1
        if fallback_toml.get("features") != expected_features:
            print(f"fallback TOML feature self-test failed: {fallback_toml}", file=sys.stderr)
            return 1
    return 0


def build_report_data() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trait_methods = parse_trait_method_names(
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "mod.rs",
        "BackendRuntime",
    )
    backends, mismatches = backend_report(trait_methods)
    conformance_gates = conformance_gate_report()
    data = {
        "source": {
            "branch": source_branch(),
            "script": str(Path(__file__).relative_to(ROOT)),
        },
        "features": feature_report(),
        "trait_method_count": len(trait_methods),
        "request_descriptors": request_descriptor_report(),
        "capability_descriptors": capability_descriptor_report(),
        "focused_backend_facets": focused_backend_facet_report(),
        "replay_authority": replay_authority_report(),
        "resident_resource_descriptors": resident_resource_descriptor_report(),
        "migration_phase_status": migration_phase_status_report(conformance_gates),
        "conformance_gates": conformance_gates,
        "request_capability_queries": sorted(
            parse_trait_method_names(CAPABILITY_RS, "BackendCapabilityQueries")
        ),
        "backends": backends,
        "fallback_policy": fallback_policy_report(),
        "decode_hot_path_policy": decode_hot_path_policy_report(),
        "training_optimizer_fallback_policy": training_optimizer_fallback_policy_report(),
        "training_precision_policy": training_precision_policy_report(),
        "optimizer_dispatch": optimizer_dispatch_report(backends),
        "mismatches": mismatches,
    }
    return data, mismatches


def main() -> int:
    if "--self-test" in sys.argv:
        return run_self_test()

    data, mismatches = build_report_data()
    json_text, markdown_text = report_outputs(data)

    if "--check" in sys.argv:
        failures = check_report_files(json_text, markdown_text)
        if mismatches:
            print(json.dumps(mismatches, indent=2), file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1 if mismatches or failures else 0

    write_report_files(json_text, markdown_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
