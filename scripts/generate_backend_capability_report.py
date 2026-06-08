#!/usr/bin/env python3
"""Generate backend capability reports from the live source tree."""

from __future__ import annotations

import json
import math
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
    "ProjectionLoadPolicy",
    "GpuMemoryDetectionPolicy",
    "GpuMemoryBudgetPolicy",
    "GpuAllocatorMemoryProbePolicy",
    "GpuMemoryReclaimPolicy",
    "KvCacheAutoBlockPolicy",
    "KvCacheMemoryTierBlockCap",
    "KvCacheFp8Policy",
    "StartupCapabilities",
    "MatmulCapabilities",
    "AttentionCapabilities",
    "GdnCapabilities",
    "InferenceRecurrentStatePolicy",
    "DecodeCapabilities",
    "SpeculativeDecodePolicy",
    "DecodeBatcherPolicy",
    "BackendTrainingCapabilities",
    "ServerTrainingDispatchPolicy",
    "TrainingAccelerationProfilePolicy",
    "TrainingAccelerationEnvFlagPolicy",
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
    "StartupBackend",
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

PRODUCTION_BACKEND_TYPES = {
    "CpuBackend",
    "CudaBackend",
    "RocmBackend",
    "MetalBackend",
    "VulkanBackend",
}

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

LEGACY_ENV_ALIAS_PREFIXES = {
    "rocm": ("KILN_DISABLE_CUDA_",),
}

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


def production_source_text(path: str) -> str:
    """Return source excluding cfg(test) modules while preserving later production code."""
    source = file_text(path)
    pattern = re.compile(
        r"\n?\s*#\[cfg\(test\)\]\s*(?:\n\s*#\[[^\n]+\]\s*)*\n\s*mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{"
    )
    pieces: list[str] = []
    cursor = 0
    while True:
        match = pattern.search(source, cursor)
        if not match:
            pieces.append(source[cursor:])
            break
        pieces.append(source[cursor : match.start()])
        open_idx = source.rfind("{", match.start(), match.end())
        end_idx = find_matching_brace(source, open_idx)
        cursor = end_idx + 1
    return "".join(pieces)


def source_between(source: str, start_marker: str, end_marker: str) -> str:
    start = source.find(start_marker)
    if start < 0:
        return ""
    end = source.find(end_marker, start)
    if end < 0:
        return source[start:]
    return source[start:end]


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


def is_support_method_name(name: str) -> bool:
    return name.startswith("supports_") or name.startswith("runtime_supports_")


def normalized_support_method_name(name: str) -> str:
    if name.startswith("runtime_supports_"):
        return name.removeprefix("runtime_")
    return name


def support_pair_candidates(name: str) -> list[str]:
    support_name = normalized_support_method_name(name)
    pair = SUPPORT_PAIRS.get(support_name, support_name.removeprefix("supports_"))
    candidates = []
    if name.startswith("runtime_supports_"):
        candidates.append(f"runtime_{pair}")
    candidates.extend([pair, f"runtime_{pair}"])
    return list(dict.fromkeys(candidates))


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


def legacy_env_aliases(backend: str, gates: list[str]) -> list[str]:
    prefixes = LEGACY_ENV_ALIAS_PREFIXES.get(backend, ())
    return sorted(gate for gate in gates if gate.startswith(prefixes))


def native_env_gates(backend: str, gates: list[str]) -> list[str]:
    legacy = set(legacy_env_aliases(backend, gates))
    return sorted(gate for gate in gates if gate not in legacy)


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
    backend_sources = [backend_mod] + sorted(
        (ROOT / "crates" / "kiln-model" / "src" / "backend").glob("*.rs")
    )
    source = "\n".join(path.read_text() for path in backend_sources)
    report: dict[str, Any] = {}
    for trait_name in FOCUSED_BACKEND_TRAITS:
        methods = sorted(parse_trait_method_names(backend_mod, trait_name))
        blanket_impl = (
            f"impl<T: BackendRuntime + ?Sized> {trait_name} for T" in source
            or f"impl<T> {trait_name} for T" in source
        )
        concrete_impls = sorted(
            impl
            for impl in set(
                re.findall(rf"impl\s+{trait_name}\s+for\s+([A-Za-z0-9_]+Backend)\b", source)
            )
            if impl in PRODUCTION_BACKEND_TYPES
        )
        forwarding_impl = (
            "blanket_backend_runtime"
            if blanket_impl
            else "concrete_authoritative"
            if concrete_impls
            else "missing"
        )
        report[trait_name] = {
            "source": str(backend_mod.relative_to(ROOT)),
            "method_count": len(methods),
            "methods": methods,
            "forwarding_impl": forwarding_impl,
            "concrete_impl_count": len(concrete_impls),
            "concrete_impls": concrete_impls,
        }
    return report


def existing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if path_exists(path)]


def missing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not path_exists(path)]


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def hardware_latency_coverage_blockers() -> list[str]:
    manifest_path = ROOT / "docs" / "backend-latency-fixtures.json"
    if not manifest_path.is_file():
        return ["docs/backend-latency-fixtures.json is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"docs/backend-latency-fixtures.json is unreadable: {exc}"]

    blockers: list[str] = []
    status = manifest.get("status")
    if status != "covered":
        blockers.append(f"manifest status is {status!r}, expected 'covered'")

    fixtures = manifest.get("fixtures", [])
    if not isinstance(fixtures, list):
        return [*blockers, "fixtures must be an array"]

    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict):
            blockers.append(f"fixtures[{index}] must be an object")
            continue
        fixture_id = fixture.get("id") if isinstance(fixture.get("id"), str) else f"fixtures[{index}]"
        result_artifact = fixture.get("result_artifact")
        if isinstance(result_artifact, str) and result_artifact:
            if not path_exists(result_artifact):
                blockers.append(f"{fixture_id}: missing result artifact {result_artifact}")
        else:
            blockers.append(f"{fixture_id}: result_artifact is missing")

        threshold_state = fixture.get("threshold_state")
        if threshold_state != "locked_threshold":
            blockers.append(
                f"{fixture_id}: threshold_state is {threshold_state!r}, expected 'locked_threshold'"
            )

        metrics = fixture.get("metrics", [])
        if not isinstance(metrics, list):
            blockers.append(f"{fixture_id}: metrics must be an array")
            continue
        for metric_index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                blockers.append(f"{fixture_id}.metrics[{metric_index}] must be an object")
                continue
            metric_name = metric.get("name")
            label = metric_name if isinstance(metric_name, str) and metric_name else f"metrics[{metric_index}]"
            if not finite_number(metric.get("max")):
                blockers.append(f"{fixture_id}.{label}: max threshold is not finite numeric")

    return blockers


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
        runtime_supports = {
            normalized_support_method_name(name)
            for name in functions
            if name.startswith("runtime_supports_")
        }
        support_methods: dict[str, Any] = {}
        for name, fun in sorted(functions.items()):
            if not is_support_method_name(name):
                continue
            if name.startswith("supports_") and name in runtime_supports:
                continue
            pair = support_pair_candidates(name)[0]
            paired_fun = None
            for candidate in support_pair_candidates(name):
                if candidate in functions:
                    pair = candidate
                    paired_fun = functions[candidate]
                    break
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
        gates = env_gates(source)
        report[backend] = {
            "source": str(path.relative_to(ROOT)),
            "source_modules": [str(source_path.relative_to(ROOT)) for source_path in source_paths],
            "override_count": len(overrides),
            "overrides": overrides,
            "support_methods": support_methods,
            "env_gates": gates,
            "native_env_gates": native_env_gates(backend, gates),
            "legacy_env_aliases": legacy_env_aliases(backend, gates),
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


def training_loss_policy_report() -> dict[str, Any]:
    return {
        "cpu": {
            "sft_flce_loss_route": "full_logits",
            "tape_forward_backward_route": "unsupported",
            "grpo_loss_route": "kt_composite",
            "grpo_kl_auxiliary_route": "host_composite",
            "opd_loss_route": "unsupported",
            "opd_phase_b_backward_route": "unsupported",
            "final_rmsnorm_backward_route": "kt_composite",
            "evidence": "TrainingCapabilities::portable keeps tape forward/backward unsupported, SFT on the portable full-logits loss path, GRPO on the shared kt composite loss root, GRPO KL auxiliaries on the host-composite route, OPD unsupported on the portable backend surface, and final RMSNorm backward on the kt-composite route",
        },
        "cuda": {
            "sft_flce_loss_route": "kt_tape_flce",
            "tape_forward_backward_route": "kt_tape_authoritative",
            "grpo_loss_route": "kt_composite",
            "grpo_kl_auxiliary_route": "cuda_rocm_device_fast_path",
            "opd_loss_route": "kt_tape_phase_b",
            "opd_phase_b_backward_route": "cuda_rocm_fused_unit_grad",
            "final_rmsnorm_backward_route": "cuda_rocm_fused_tail",
            "evidence": "CudaBackend::training_capabilities_static advertises kt tape-authoritative forward/backward, kt-tape FLCE over CUDA tensors, the shared kt GRPO composite route, CUDA/ROCm device fast paths for GRPO KL auxiliaries, the shared kt-tape OPD Phase-B route, the fused CUDA/ROCm Phase-B hidden-gradient leaf, and the fused final-RMSNorm tail route",
        },
        "rocm": {
            "sft_flce_loss_route": "kt_tape_flce",
            "tape_forward_backward_route": "kt_tape_authoritative",
            "grpo_loss_route": "kt_composite",
            "grpo_kl_auxiliary_route": "cuda_rocm_device_fast_path",
            "opd_loss_route": "kt_tape_phase_b",
            "opd_phase_b_backward_route": "cuda_rocm_fused_unit_grad",
            "final_rmsnorm_backward_route": "cuda_rocm_fused_tail",
            "evidence": "RocmBackend::training_capabilities_static advertises kt tape-authoritative forward/backward, the shared kt-tape FLCE route over ROCm tensors, the shared kt GRPO composite route, CUDA/ROCm device fast paths for GRPO KL auxiliaries, the shared kt-tape OPD Phase-B route, the fused CUDA/ROCm Phase-B hidden-gradient leaf, and the fused final-RMSNorm tail route",
        },
        "metal": {
            "sft_flce_loss_route": "full_logits",
            "tape_forward_backward_route": "kt_tape_authoritative",
            "grpo_loss_route": "kt_composite",
            "grpo_kl_auxiliary_route": "host_composite",
            "opd_loss_route": "kt_tape_phase_b",
            "opd_phase_b_backward_route": "kt_composite",
            "final_rmsnorm_backward_route": "kt_composite",
            "evidence": "Metal training capabilities advertise kt tape-authoritative forward/backward, inherit the portable full-logits SFT loss route, shared kt GRPO composite route, host-composite GRPO KL auxiliaries, shared kt-tape OPD Phase-B route, device-agnostic kt composite Phase-B backward, and kt-composite final RMSNorm backward",
        },
        "vulkan": {
            "sft_flce_loss_route": "vulkan_active_rows",
            "tape_forward_backward_route": "kt_tape_authoritative",
            "grpo_loss_route": "vulkan_active_rows",
            "grpo_kl_auxiliary_route": "host_composite",
            "opd_loss_route": "vulkan_active_hidden",
            "opd_phase_b_backward_route": "vulkan_active_hidden",
            "final_rmsnorm_backward_route": "kt_composite",
            "evidence": "Vulkan training capabilities advertise kt tape-authoritative forward/backward, active-row fused SFT/GRPO shader routes, host-composite GRPO KL auxiliaries, the active-hidden fused OPD loss/backward shader route, and kt-composite final RMSNorm backward",
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
            "mixed_rms_norm_weight_dtype": None,
            "streaming_prefill_tile_tokens": 8192,
            "tape_streaming_tile_tokens": 8192,
            "paged_prefill_medium_tile_tokens": None,
            "paged_prefill_medium_tile_max_tokens": None,
            "exact_gdn_backward_tile_tokens": None,
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
            "mixed_rms_norm_weight_dtype": None,
            "streaming_prefill_tile_tokens": 1024,
            "tape_streaming_tile_tokens": 1024,
            "paged_prefill_medium_tile_tokens": None,
            "paged_prefill_medium_tile_max_tokens": None,
            "exact_gdn_backward_tile_tokens": 1024,
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
            "mixed_rms_norm_weight_dtype": None,
            "streaming_prefill_tile_tokens": 1024,
            "tape_streaming_tile_tokens": 1024,
            "paged_prefill_medium_tile_tokens": 1024,
            "paged_prefill_medium_tile_max_tokens": 20000,
            "exact_gdn_backward_tile_tokens": None,
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
            "mixed_rms_norm_weight_dtype": None,
            "streaming_prefill_tile_tokens": 2048,
            "tape_streaming_tile_tokens": 2048,
            "paged_prefill_medium_tile_tokens": None,
            "paged_prefill_medium_tile_max_tokens": None,
            "exact_gdn_backward_tile_tokens": None,
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
            "mixed_rms_norm_weight_dtype": "BF16",
            "streaming_prefill_tile_tokens": 2048,
            "tape_streaming_tile_tokens": 2048,
            "paged_prefill_medium_tile_tokens": None,
            "paged_prefill_medium_tile_max_tokens": None,
            "exact_gdn_backward_tile_tokens": None,
            "mixed_precision": True,
            "notes": "Vulkan keeps training activations and LoRA parameters F32 while allowing BF16 base weights through explicit VkTensor buffer bridges.",
        },
    }


def path_exists(path: str) -> bool:
    return (ROOT / path).exists()


def conformance_gate_report() -> list[dict[str, Any]]:
    hardware_latency_blockers = hardware_latency_coverage_blockers()

    gates = [
        {
            "gate": "storage_round_trip",
            "phase8_requirement": "storage round trip",
            "status": None,
            "command": "cargo test -p kiln-tensor --features rocm --test rocm_storage_smoke && cargo test -p kiln-vulkan-kernel --test vk_tensor_parity",
            "supplemental_commands": [
                {
                    "scope": "ROCm feature lane",
                    "command": "cargo test -p kiln-tensor --features rocm --test rocm_storage_smoke",
                },
                {
                    "scope": "Vulkan kernel lane",
                    "command": "cargo test -p kiln-vulkan-kernel --test vk_tensor_parity",
                },
            ],
            "evidence": [
                "crates/kiln-tensor/tests/rocm_storage_smoke.rs",
                "crates/kiln-vulkan-kernel/tests/vk_tensor_parity.rs",
            ],
        },
        {
            "gate": "host_transfer_to_device_parity",
            "phase8_requirement": "host transfer / to_device parity with explicit unsupported errors",
            "status": None,
            "command": "cargo test -p kiln-tensor device_transfer_support_classifies_explicit_transitions && cargo test -p kiln-tensor to_device_without_gpu_features_reports_explicit_unsupported_transition",
            "supplemental_commands": [
                {
                    "scope": "CUDA hardware lane",
                    "command": "CUDARC_CUDA_VERSION=12080 cargo test -p kiln-tensor --no-default-features --features cuda --test cuda_resize_copy_primitives",
                },
                {
                    "scope": "ROCm feature lane",
                    "command": "cargo test -p kiln-tensor --features rocm --test rocm_compare_parity",
                },
                {
                    "scope": "macOS Metal feature lane",
                    "command": "cargo test -p kiln-tensor --features metal --test metal_ops_parity",
                },
                {
                    "scope": "Vulkan kernel lane",
                    "command": "cargo test -p kiln-vulkan-kernel --test vk_tensor_parity",
                },
            ],
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
            "status": None,
            "command": "cargo test -p kiln-tensor device_op::tests",
            "supplemental_commands": [
                {
                    "scope": "ROCm feature lane",
                    "command": "cargo test -p kiln-tensor --features rocm --test rocm_scalar_op_parity",
                },
                {
                    "scope": "macOS Metal feature lane",
                    "command": "cargo test -p kiln-tensor --features metal --test metal_ops_parity",
                },
            ],
            "evidence": [
                "crates/kiln-tensor/src/device_op.rs",
                "crates/kiln-tensor/tests/rocm_scalar_op_parity.rs",
                "crates/kiln-tensor/tests/metal_ops_parity.rs",
            ],
        },
        {
            "gate": "matmul_linear_parity",
            "phase8_requirement": "matmul/linear parity",
            "status": None,
            "command": "cargo test -p kiln-model matmul_request_projects_to_blas_shape_contract && cargo test -p kiln-tensor --features rocm --test rocm_matmul_parity && cargo test -p kiln-tensor matmul_matrix_core && cargo test -p kiln-vulkan-kernel --test vk_matmul_parity && cargo test -p kiln-vulkan-kernel --test linear_decode_argmax && cargo test -p kiln-vulkan-kernel --test linear_decode_sample && cargo test -p kiln-model tape_forward_matmul_bit_exact_parity_with_baseline && CUDARC_CUDA_VERSION=12080 cargo check -p kiln-blas --features cublaslt --tests && cargo test -p kiln-model --test backend_capability_contract",
            "supplemental_commands": [
                {
                    "scope": "CUDA cublasLt hardware lane",
                    "command": "CUDARC_CUDA_VERSION=12080 cargo test -p kiln-blas --features cublaslt --test cublaslt_handle_smoke",
                },
                {
                    "scope": "macOS Metal feature lane",
                    "command": "cargo test -p kiln-tensor --features metal --test metal_ops_parity",
                },
            ],
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
            "status": None,
            "command": "cargo test -p kiln-model --no-default-features --features rocm --test rocm_flash_attn_bwd_gradcheck && cargo test -p kiln-flash-attn --no-default-features --features rocm --test rocm_flash_attn_parity && cargo test -p kiln-gdn-kernel --no-default-features --features rocm --test rocm_gdn_parity && cargo test -p kiln-conv1d-kernel --no-default-features --features rocm --test rocm_conv1d_parity && cargo test -p kiln-vulkan-kernel --test vk_attention_parity && cargo test -p kiln-vulkan-kernel --test vk_sdpa_prefill_kernel_parity && cargo test -p kiln-vulkan-kernel --test vk_gdn_foundation_parity && cargo test -p kiln-vulkan-kernel --test vk_gdn_backward_parity && cargo test -p kiln-vulkan-kernel --test gdn_parity",
            "evidence": [
                "crates/kiln-model/tests/rocm_flash_attn_bwd_gradcheck.rs",
                "crates/kiln-flash-attn/tests/rocm_flash_attn_parity.rs",
                "crates/kiln-gdn-kernel/tests/rocm_gdn_parity.rs",
                "crates/kiln-conv1d-kernel/tests/rocm_conv1d_parity.rs",
                "crates/kiln-vulkan-kernel/tests/vk_attention_parity.rs",
                "crates/kiln-vulkan-kernel/tests/vk_sdpa_prefill_kernel_parity.rs",
                "crates/kiln-vulkan-kernel/tests/vk_gdn_foundation_parity.rs",
                "crates/kiln-vulkan-kernel/tests/vk_gdn_backward_parity.rs",
                "crates/kiln-vulkan-kernel/tests/gdn_parity.rs",
            ],
        },
        {
            "gate": "optimizer_parity",
            "phase8_requirement": "optimizer parity",
            "status": None,
            "command": "cargo test -p kiln-optim --test integration && cargo test -p kiln-train training_optimizer && cargo test -p kiln-model --test backend_capability_contract",
            "supplemental_commands": [
                {
                    "scope": "CUDA plus Vulkan OPD hardware lane",
                    "command": "CUDARC_CUDA_VERSION=12080 cargo test -p kiln-train --features cuda,vulkan --test vk_cuda_opd_parity",
                },
            ],
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
            "status": None,
            "command": "cargo test -p kiln-graph replay && cargo test -p kiln-graph --test capture_lifetime && cargo test -p kiln-graph-cuda replay && cargo test -p kiln-graph-metal replay && cargo test -p kiln-graph-vulkan replay && cargo test -p kiln-model --features vulkan --test vk_resident_decode_parity && cargo test -p kiln-tensor --features rocm --test rocm_capture_arena && cargo test -p kiln-model --test backend_capability_contract",
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
            "status": None,
            "command": "cargo test -p kiln-optim --test end_to_end_training && cargo test -p kiln-model --test backend_capability_contract",
            "supplemental_commands": [
                {
                    "scope": "CUDA hardware lane",
                    "command": "CUDARC_CUDA_VERSION=12080 cargo test -p kiln-model --features cuda --test cuda_sft_step_proof",
                },
                {
                    "scope": "ROCm feature lane",
                    "command": "cargo test -p kiln-model --features rocm --test rocm_sft_step_proof",
                },
                {
                    "scope": "macOS Metal feature lane",
                    "command": "cargo test -p kiln-model --features metal --test metal_sft_step_proof",
                },
                {
                    "scope": "Vulkan hardware opt-in lane",
                    "command": "KILN_TENSOR_VULKAN_TEST=1 KILN_USE_TAPE_FORWARD=1 KILN_USE_TAPE_LORA_ADD=1 cargo test -p kiln-model --features vulkan --test vk_sft_step_proof",
                },
            ],
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
            "status": None,
            "command": "cargo test -p kiln-tensor device_op_host_fallback_counts_are_backend_and_arity_specific",
            "evidence": [
                "crates/kiln-tensor/src/device_op.rs",
                "crates/kiln-model/src/generate.rs",
                "crates/kiln-train/src/trainer.rs",
            ],
        },
        {
            "gate": "decode_submit_or_replay_count",
            "phase8_requirement": "max submit count or replay count per decode token",
            "status": None,
            "command": "cargo test -p kiln-model decode_batcher_stats_report_runner_calls_per_token && cargo test -p kiln-server test_metrics_render && cargo test -p kiln-graph replay",
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
            "status": None,
            "command": "cargo test -p kiln-blas cache_stats_reports_entries_and_hit_rate && cargo test -p kiln-rocblas cache_stats_reports_entries_and_hit_rate && CUDARC_CUDA_VERSION=12080 cargo check -p kiln-blas --features cublaslt --tests && cargo check -p kiln-rocblas --features hipblaslt --tests",
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
            "status": "fixture_required" if hardware_latency_blockers else None,
            "command": "python3 scripts/run_backend_latency_fixture.py --self-test && python3 scripts/write_backend_latency_result_artifact.py --self-test && python3 scripts/import_backend_latency_artifact.py --self-test && python3 scripts/lock_backend_latency_thresholds.py --self-test && python3 scripts/check_backend_latency_fixtures.py --self-test && python3 scripts/plan_backend_latency_fixture_dispatch.py --self-test && python3 scripts/check_backend_latency_fixtures.py docs/backend-latency-fixtures.json --require-covered",
            "coverage_blockers": hardware_latency_blockers,
            "evidence": [
                "docs/backend-latency-fixtures.json",
                "docs/backend-latency-result-schema.md",
                "scripts/check_unification_gates.sh",
                "scripts/run_backend_latency_fixture.py",
                "scripts/write_backend_latency_result_artifact.py",
                "scripts/import_backend_latency_artifact.py",
                "scripts/lock_backend_latency_thresholds.py",
                "scripts/check_backend_latency_fixtures.py",
                "scripts/plan_backend_latency_fixture_dispatch.py",
                "crates/kiln-tensor/tests/cuda_latency_bench.rs",
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
            "status": None,
            "command": "python3 scripts/generate_backend_capability_report.py --self-test && python3 scripts/generate_backend_capability_report.py --check",
            "evidence": [
                "docs/backend-capability-report.md",
                "docs/backend-capability-report.json",
                "scripts/generate_backend_capability_report.py",
                "scripts/check_unification_gates.sh",
            ],
        },
    ]

    for gate in gates:
        gate["supplemental_commands"] = gate.get("supplemental_commands", [])
        gate["coverage_blockers"] = gate.get("coverage_blockers", [])
        gate["evidence_present"] = [
            evidence for evidence in gate["evidence"] if path_exists(evidence)
        ]
        gate["evidence_missing"] = [
            evidence for evidence in gate["evidence"] if not path_exists(evidence)
        ]
        if gate["status"] is None:
            gate["status"] = "covered" if not gate["evidence_missing"] else "gap"
    return gates


def phase_status(contract: str, migration: str) -> str:
    if contract == "absent":
        return "gap"
    if migration == "complete":
        return "covered"
    if migration == "partial":
        return "partial"
    return "gap"


def phase_signal(name: str, passed: bool, observed: Any, expected: Any, evidence: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "expected": expected,
        "evidence": evidence,
    }


def migration_from_signals(signals: list[dict[str, Any]]) -> str:
    if signals and all(signal["passed"] for signal in signals):
        return "complete"
    if any(signal["passed"] for signal in signals):
        return "partial"
    return "none"


def phase1_remaining_from_signals(signals: list[dict[str, Any]]) -> list[str]:
    remaining: list[str] = []
    for signal in signals:
        if signal["passed"]:
            continue
        name = signal["name"]
        if name == "focused_trait_forwarding_shims_removed":
            remaining.append(
                "focused trait blanket forwarding shims remain in BackendRuntime"
            )
        elif name == "backend_runtime_method_count_below_gate":
            remaining.append("BackendRuntime remains above the method-count gate")
        elif name.endswith("_facet_authoritative"):
            trait_slug = name.removesuffix("_facet_authoritative")
            trait_name = "".join(part.capitalize() for part in trait_slug.split("_"))
            remaining.append(
                f"{trait_name} is not authoritative on all concrete backends"
            )
        else:
            remaining.append(f"{name} migration signal is not yet satisfied")
    return remaining


def phase3_remaining_from_signals(signals: list[dict[str, Any]]) -> list[str]:
    remaining: list[str] = []
    for signal in signals:
        if signal["passed"]:
            continue
        name = signal["name"]
        if name == "resident_registry_blanket_adapter_removed":
            remaining.append("ResidentRegistry blanket adapter remains in backend/mod.rs")
        elif name == "production_backends_implement_resident_registry":
            remaining.append(
                "not all production backends implement ResidentRegistry directly"
            )
        elif name == "residency_backend_facade_delegates_to_registry":
            remaining.append(
                "ResidencyBackend activation facade does not delegate through ResidentRegistry"
            )
        elif name == "resident_registry_process_global_statics_removed":
            remaining.append(
                "resident activation registries still use process-global statics"
            )
        elif name == "resident_registry_drop_drains_test_present":
            remaining.append("drop-drains-registry behavioral test is not present")
        elif name == "resident_registry_lifecycle_metadata_persisted":
            remaining.append(
                "production resident registries do not persist lifecycle metadata"
            )
        else:
            remaining.append(f"{name} migration signal is not yet satisfied")
    return remaining


def phase5_remaining_from_signals(signals: list[dict[str, Any]]) -> list[str]:
    remaining: list[str] = []
    for signal in signals:
        if signal["passed"]:
            continue
        name = signal["name"]
        if name == "replay_contract_w5_0_fixed":
            remaining.append(
                "ReplayPlan contract still lacks W5.0 byte-length, layout, stability, or key-change guards"
            )
        elif name == "production_replay_paths_use_replay_plan":
            remaining.append("production replay runners are not wired through ReplayPlan")
        elif name == "replay_parity_w5_3_live_gate":
            remaining.append(
                "eager-vs-replay parity gate is not live in both local ReplayPlan contract and hardware graph tests"
            )
        else:
            remaining.append(f"{name} migration signal is not yet satisfied")
    return remaining


def phase2_remaining_from_signals(signals: list[dict[str, Any]]) -> list[str]:
    remaining: list[str] = []
    for signal in signals:
        if signal["passed"]:
            continue
        name = signal["name"]
        if name == "decode_hot_path_duplicate_helpers_removed":
            remaining.append(
                "forward/generate still keep duplicate decode hot-path fallback helpers"
            )
        elif name == "decode_hot_path_fallback_delegates_to_backend_capability":
            remaining.append(
                "decode hot-path fallback decisions are not centralized in BackendFallbackCapabilities"
            )
        else:
            remaining.append(f"{name} migration signal is not yet satisfied")
    return remaining


def phase6_remaining_from_signals(signals: list[dict[str, Any]]) -> list[str]:
    remaining: list[str] = []
    for signal in signals:
        if signal["passed"]:
            continue
        name = signal["name"]
        if name == "training_precision_for_device_family_removed_from_production":
            remaining.append(
                "production training paths still call TrainingPrecisionPolicy::for_device_family"
            )
        elif name == "training_precision_policy_delegates_to_backend_trait":
            remaining.append(
                "training precision policy is not selected through TrainingLossBackend"
            )
        elif name == "tape_forward_gpu_family_guards_removed":
            remaining.append(
                "tape-forward production adapters still gate support with a hard-coded GPU family allowlist"
            )
        elif name == "tape_forward_route_delegates_to_backend_trait":
            remaining.append(
                "tape-forward device support is not selected through TrainingLossBackend"
            )
        elif name == "sft_step_proofs_route_optimizer_backend":
            remaining.append(
                "CUDA/ROCm SFT step proofs still bypass OptimizerBackend for AdamW"
            )
        else:
            remaining.append(f"{name} migration signal is not yet satisfied")
    return remaining


def file_text(path: str) -> str:
    try:
        return (ROOT / path).read_text()
    except OSError:
        return ""


def regex_count(path: str, pattern: str) -> int:
    return len(re.findall(pattern, file_text(path), flags=re.MULTILINE | re.DOTALL))


def focused_trait_forwarding_shim_count() -> int:
    return regex_count(
        "crates/kiln-model/src/backend/mod.rs",
        r"impl<T(?::|>).*?BackendRuntime.*?>\s+\w+Backend\s+for\s+T",
    )


def focused_trait_blanket_shim_count(trait_name: str) -> int:
    return regex_count(
        "crates/kiln-model/src/backend/mod.rs",
        rf"impl<T(?::|>).*?BackendRuntime.*?>\s+{trait_name}\s+for\s+T",
    )


def focused_trait_concrete_impls(trait_name: str) -> list[str]:
    backend_sources = [
        ROOT / "crates" / "kiln-model" / "src" / "backend" / "mod.rs"
    ] + sorted((ROOT / "crates" / "kiln-model" / "src" / "backend").glob("*.rs"))
    source = "\n".join(path.read_text() for path in backend_sources)
    return sorted(
        impl
        for impl in set(
            re.findall(
                rf"impl\s+{trait_name}\s+for\s+([A-Za-z0-9_]+Backend)\b",
                source,
            )
        )
        if impl in PRODUCTION_BACKEND_TYPES
    )


def backend_runtime_supertraits() -> set[str]:
    match = re.search(
        r"pub\s+trait\s+BackendRuntime\s*:\s*([^{]+)\{",
        file_text("crates/kiln-model/src/backend/mod.rs"),
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        return set()
    return set(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", match.group(1)))


def focused_trait_authoritative_signal(trait_name: str, signal_name: str) -> dict[str, Any]:
    concrete_impls = focused_trait_concrete_impls(trait_name)
    authoritative = (
        focused_trait_blanket_shim_count(trait_name) == 0
        and set(concrete_impls) == PRODUCTION_BACKEND_TYPES
        and trait_name in backend_runtime_supertraits()
    )
    return phase_signal(
        signal_name,
        authoritative,
        concrete_impls,
        sorted(PRODUCTION_BACKEND_TYPES),
        [
            "crates/kiln-model/src/backend/mod.rs",
            "crates/kiln-model/src/backend/cpu.rs",
            "crates/kiln-model/src/backend/cuda.rs",
            "crates/kiln-model/src/backend/rocm.rs",
            "crates/kiln-model/src/backend/metal_runtime.rs",
            "crates/kiln-model/src/backend/vulkan.rs",
        ],
    )


def resident_registry_forwarding_shim_count() -> int:
    return regex_count(
        "crates/kiln-model/src/backend/mod.rs",
        r"impl<T>\s+residency::ResidentRegistry\s+for\s+T",
    )


def concrete_resident_registry_impl_count() -> int:
    impl_specs = [
        (
            "crates/kiln-model/src/backend/cpu.rs",
            r"impl\s+super::residency::ResidentRegistry\s+for\s+CpuBackend",
        ),
        (
            "crates/kiln-model/src/backend/cuda.rs",
            r"impl\s+super::residency::ResidentRegistry\s+for\s+CudaBackend",
        ),
        (
            "crates/kiln-model/src/backend/rocm.rs",
            r"impl\s+super::residency::ResidentRegistry\s+for\s+RocmBackend",
        ),
        (
            "crates/kiln-model/src/backend/metal_runtime.rs",
            r"impl\s+super::residency::ResidentRegistry\s+for\s+MetalBackend",
        ),
        (
            "crates/kiln-model/src/backend/vulkan.rs",
            r"impl\s+super::residency::ResidentRegistry\s+for\s+VulkanBackend",
        ),
    ]
    return sum(1 for path, pattern in impl_specs if regex_count(path, pattern) > 0)


def residency_backend_facade_registry_delegate_count() -> int:
    source = file_text("crates/kiln-model/src/backend/mod.rs")
    trait_start = source.find("pub trait ResidencyBackend:")
    if trait_start < 0:
        return 0
    trait_end = source.find("/// Focused `OptimizerBackend`", trait_start)
    if trait_end < 0:
        return 0
    trait_source = source[trait_start:trait_end]
    required = [
        "BackendIdentity + residency::ResidentRegistry",
        "residency::ResidentRegistry::register_resource",
        "residency::ResidentRegistry::evict_resource",
        "residency::ResidentRegistry::update_resource",
        "residency::ResidentRegistry::has_resident_resource",
        "residency::ResidentRegistry::resident_resource",
        "residency::ResidentRegistry::resolve_resource",
    ]
    return sum(1 for needle in required if needle in trait_source)


def resident_registry_process_global_static_count() -> int:
    patterns = [
        (
            "crates/kiln-model/src/backend/cuda.rs",
            r"static\s+CUDA_RESIDENT_TENSOR_IDS",
        ),
        (
            "crates/kiln-model/src/backend/rocm.rs",
            r"static\s+ROCM_RESIDENT_TENSOR_IDS",
        ),
        (
            "crates/kiln-model/src/backend/metal_residency.rs",
            r"static\s+METAL_RESIDENT_ACTIVATION_REGISTRY",
        ),
        (
            "crates/kiln-model/src/backend/vulkan_residency.rs",
            r"static\s+RESIDENT_ACTIVATION_REGISTRY",
        ),
    ]
    return sum(regex_count(path, pattern) for path, pattern in patterns)


def resident_registry_drop_drains_test_count() -> int:
    return regex_count(
        "crates/kiln-model/src/backend/mod.rs",
        r"drop.*drains.*resident|resident.*drop.*drains",
    ) + regex_count(
        "crates/kiln-model/src/backend/residency.rs",
        r"drop.*drains.*resident|resident.*drop.*drains",
    )


def resident_registry_lifecycle_metadata_store_count() -> int:
    specs = [
        (
            "crates/kiln-model/src/backend/cuda_rocm_common.rs",
            ["ResidentResource", "HashMap<TensorId"],
        ),
        (
            "crates/kiln-model/src/backend/metal_residency.rs",
            ["ResidentResource", "ResidentResourceState", "ReplayStability"],
        ),
        (
            "crates/kiln-model/src/backend/vulkan_residency.rs",
            ["ResidentActivationEntry", "ResidentResource", "resource:"],
        ),
    ]
    return sum(
        1
        for path, needles in specs
        if all(needle in file_text(path) for needle in needles)
    )


def matmul_identity_dispatch_count() -> int:
    pattern = r"Device::(?:Cuda|Rocm|Metal|Vulkan)|match\s+self\.name\(\)"
    forward = production_source_text("crates/kiln-model/src/forward.rs")
    capability = production_source_text("crates/kiln-model/src/backend/capability.rs")
    sections = [
        production_source_text("crates/kiln-tensor/src/ops/matmul.rs"),
        source_between(
            capability,
            "fn supports_matmul_request(&self, req: &MatmulRequest)",
            "fn supports_linear_request",
        ),
        source_between(
            forward,
            "fn matmul_no_broadcast_copy(",
            "fn runtime_matmul_no_broadcast_copy(",
        ),
        source_between(
            forward,
            "fn runtime_matmul_no_broadcast_copy(",
            "fn runtime_matmul_or_broadcast(",
        ),
        source_between(
            forward,
            "fn runtime_matmul_or_broadcast(",
            "/// Phase 7 — kt-API matmul migration helper",
        ),
        source_between(
            forward,
            "fn gdn_in_proj_matmul(",
            "fn promote_cpu_activation(",
        ),
        source_between(
            forward,
            "fn kt_lm_head_native(",
            "/// Phase 7 (#1082) — kt-API LM head migration helper.",
        ),
        source_between(
            forward,
            "fn try_kt_lm_head(",
            "fn lm_head_forward_backend_decode_if(",
        ),
        source_between(
            forward,
            "fn lm_head_forward_backend_decode_if(",
            "fn lm_head_argmax_with_backend(",
        ),
        source_between(
            forward,
            "fn try_kt_lm_head_argmax(",
            "/// Phase 7 (#1082) — kt-API argmax migration helper.",
        ),
        source_between(
            forward,
            "fn lm_head_argmax_backend_decode_if(",
            "/// Phase 7 (#1082) — kt-API sampler argmax migration helper",
        ),
        source_between(
            forward,
            "fn lm_head_argmax_rows_backend_decode_if(",
            "fn lm_head_weighted_prep_argmax(",
        ),
        source_between(
            forward,
            "fn full_attn_qkv_proj_decode_if(",
            "/// CUDA-compatible softmax",
        ),
        source_between(
            forward,
            "fn gated_deltanet_forward_decode_if_inner(",
            "// Phase B11b tap",
        ),
        source_between(
            forward,
            "pub fn mtp_forward_step(",
            "fn model_forward_paged_inner(",
        ),
    ]
    return sum(len(re.findall(pattern, section)) for section in sections)


def matmul_request_descriptor_w4_1_signal_count() -> tuple[int, int]:
    required = [
        ("crates/kiln-model/src/backend/capability.rs", "pub enum MatmulOperandLayout"),
        ("crates/kiln-model/src/backend/capability.rs", "ColMajor"),
        ("crates/kiln-model/src/backend/capability.rs", "pub enum MatmulBatchPolicy"),
        ("crates/kiln-model/src/backend/capability.rs", "Batched { batches: usize }"),
        ("crates/kiln-model/src/backend/capability.rs", "pub lhs_dtype: kiln_tensor::DType"),
        ("crates/kiln-model/src/backend/capability.rs", "pub rhs_dtype: kiln_tensor::DType"),
        ("crates/kiln-model/src/backend/capability.rs", "pub out_dtype: kiln_tensor::DType"),
        ("crates/kiln-model/src/backend/capability.rs", "pub fn with_dtypes"),
        ("crates/kiln-model/src/backend/capability.rs", "fn logical_operand_dims"),
        ("crates/kiln-model/src/backend/capability.rs", "pub fn to_blas_request"),
        ("crates/kiln-model/src/backend/mod.rs", "transposed rhs request projects losslessly"),
        ("crates/kiln-model/src/backend/mod.rs", "mixed dtype request should project without dropping dtype metadata"),
        ("crates/kiln-model/src/backend/mod.rs", "batched request projects"),
    ]
    observed = sum(1 for path, needle in required if needle in file_text(path))
    return observed, len(required)


def matmul_support_query_authority_signal_count() -> tuple[int, int]:
    required = [
        (
            "crates/kiln-model/src/backend/capability.rs",
            "LinearBackend::runtime_supports_matmul_request(self, req)",
        ),
        (
            "crates/kiln-model/src/backend/mod.rs",
            "fn runtime_supports_matmul_request(",
        ),
        (
            "crates/kiln-model/src/backend/cpu.rs",
            "impl LinearBackend for CpuBackend",
        ),
        (
            "crates/kiln-model/src/backend/cuda.rs",
            "impl LinearBackend for CudaBackend",
        ),
        (
            "crates/kiln-model/src/backend/rocm.rs",
            "impl LinearBackend for RocmBackend",
        ),
        (
            "crates/kiln-model/src/backend/metal_runtime.rs",
            "impl LinearBackend for MetalBackend",
        ),
        (
            "crates/kiln-model/src/backend/vulkan.rs",
            "impl LinearBackend for VulkanBackend",
        ),
    ]
    observed = sum(1 for path, needle in required if needle in file_text(path))
    return observed, len(required)


def matmul_transposed_request_contract_signal_count() -> tuple[int, int]:
    required = [
        (
            "crates/kiln-model/src/backend/mod.rs",
            "let cuda_transposed = capability::MatmulRequest::plain",
        ),
        (
            "crates/kiln-model/src/backend/mod.rs",
            "transposed_bias",
        ),
        (
            "crates/kiln-model/src/backend/mod.rs",
            "cpu backend should route rhs-transposed matmul request",
        ),
        (
            "crates/kiln-model/src/backend/mod.rs",
            "batched CPU backend should route rank-3 matmul request",
        ),
        (
            "crates/kiln-model/src/backend/mod.rs",
            "mixed dtype CPU request should route through the F32 oracle",
        ),
    ]
    observed = sum(1 for path, needle in required if needle in file_text(path))
    return observed, len(required)


def replay_production_replay_plan_signal_count() -> tuple[int, int]:
    paths = [
        "crates/kiln-model/src/cuda_graph.rs",
        "crates/kiln-model/src/rocm_graph.rs",
        "crates/kiln-model/src/metal_graph.rs",
        "crates/kiln-model/src/vk_decode_resident.rs",
    ]
    observed = 0
    for path in paths:
        source = production_source_text(path)
        if "ReplayPlan" in source and "ReplayPlan::replay(" in source:
            observed += 1
    return observed, len(paths)


def replay_contract_w5_0_signal_count() -> int:
    source = file_text("crates/kiln-graph/src/replay_plan.rs")
    required = [
        "packed_buffer_bytes(element_count)",
        "pub strides: Vec<usize>",
        "pub start_offset: usize",
        "pub contiguous: bool",
        "fn validate_inputs(&self, inputs: ReplayInputs<'_>)",
        "replay_state_accepts_stable_within_step_inputs",
        "replay_resource_ref_tracks_packed_byte_len_and_layout",
        "replay_plan_validate_inputs_rejects_key_changes",
    ]
    return sum(1 for needle in required if needle in source)


def replay_parity_w5_3_signal_count() -> tuple[int, int]:
    contract = file_text("crates/kiln-model/tests/backend_capability_contract.rs")
    forward = file_text("crates/kiln-model/src/forward.rs")
    local_contract = source_between(
        contract,
        "fn replay_plan_cpu_mock_parity_gate_runs_in_unification_contract",
        "#[test]\nfn backend_engine_unification_plan_matches_current_training_status",
    )
    metal_parity = source_between(
        forward,
        "fn test_metal_graph_batched_decode_matches_eager_and_replays_bucket",
        "/// bs=1 CUDA-graph-capture+replay vs. eager decode parity.",
    )
    cuda_parity = source_between(
        forward,
        "fn test_cuda_graph_bs1_decode_matches_eager",
        "#[cfg(feature = \"metal\")]\n    #[test]\n    fn test_model_forward_paged_decode_contiguous_batch_hybrid_matches_rowwise_metal",
    )
    checks = [
        all(
            needle in local_contract
            for needle in [
                "MockCpuDecodeReplayPlan",
                "ReplayPlan::replay(",
                "assert_eq!(\n        replayed, &eager",
                "CPU/mock ReplayPlan parity gate should compare replayed output to eager output",
            ]
        ),
        all(
            needle in metal_parity
            for needle in [
                "assert_eq!(\n                    graph, eager",
                "captured_graph_replay_count_sum()",
                "same-bucket batched step should replay the captured Metal ICB graph",
            ]
        ),
        all(
            needle in cuda_parity
            for needle in [
                "let replay = step(&mut runner",
                "assert_eq!(\n                ea, ra",
                "CUDA-graph replay and eager decode picked DIFFERENT tokens",
            ]
        )
        and "SKIP: bs=1 graph capture did not succeed" not in cuda_parity
        and "it becomes a live graph-replay-vs-eager decode-parity gate" not in cuda_parity,
    ]
    return sum(1 for check in checks if check), len(checks)


def training_precision_for_device_family_production_count() -> int:
    paths = [
        "crates/kiln-model/src/forward.rs",
        "crates/kiln-model/src/tape_forward.rs",
        "crates/kiln-train/src/trainer.rs",
    ]
    return sum(
        production_source_text(path).count("TrainingPrecisionPolicy::for_device_family")
        for path in paths
    )


def training_precision_backend_trait_policy_signal_count() -> tuple[int, int]:
    backend = production_source_text("crates/kiln-model/src/backend/mod.rs")
    required = [
        "pub fn training_precision_policy_for_device_kt",
        "TrainingLossBackend::runtime_training_precision_policy(&backend)",
    ]
    observed = sum(1 for needle in required if needle in backend)
    return observed, len(required)


def decode_hot_path_duplicate_helper_count() -> int:
    paths = [
        "crates/kiln-model/src/forward.rs",
        "crates/kiln-model/src/generate.rs",
    ]
    forbidden = [
        "fn decode_batch_generic_fallback_enabled(",
        "fn decode_hot_path_fallback_policy(",
        "fn decode_hot_path_debug_fallback_enabled(",
    ]
    return sum(
        production_source_text(path).count(needle)
        for path in paths
        for needle in forbidden
    )


def decode_hot_path_shared_fallback_signal_count() -> tuple[int, int]:
    capability = production_source_text("crates/kiln-model/src/backend/capability.rs")
    forward = production_source_text("crates/kiln-model/src/forward.rs")
    generate = production_source_text("crates/kiln-model/src/generate.rs")
    required = [
        "pub(crate) fn decode_hot_path_fallback_policy_for_backend",
        "pub(crate) fn decode_hot_path_debug_fallback_enabled_for_backend",
        "pub(crate) fn decode_hot_path_generic_fallback_enabled_for_backend",
        "decode_hot_path_debug_fallback_enabled_for_backend(backend)",
        "decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend)",
        "decode_hot_path_fallback_policy_for_backend(backend)",
    ]
    sources = [capability, capability, capability, forward, generate, generate]
    observed = sum(1 for needle, source in zip(required, sources) if needle in source)
    return observed, len(required)


def tape_forward_gpu_family_guard_count() -> int:
    tape_forward = production_source_text("crates/kiln-model/src/tape_forward.rs")
    return len(
        re.findall(
            r"matches!\s*\([^;]*Device::Cuda\(_\)[^;]*Device::Metal\(_\)"
            r"[^;]*Device::Vulkan\(_\)[^;]*Device::Rocm\(_\)",
            tape_forward,
            re.DOTALL,
        )
    )


def tape_forward_backend_trait_route_signal_count() -> tuple[int, int]:
    backend = production_source_text("crates/kiln-model/src/backend/mod.rs")
    tape_forward = production_source_text("crates/kiln-model/src/tape_forward.rs")
    required = [
        "pub fn training_tape_route_for_device_kt",
        "TrainingLossBackend::runtime_tape_forward_backward_route(&backend)",
        "fn tape_forward_device_supported",
        "training_tape_route_for_device_kt(device)",
    ]
    sources = [backend, backend, tape_forward, tape_forward]
    observed = sum(1 for needle, source in zip(required, sources) if needle in source)
    return observed, len(required)


def sft_step_proof_optimizer_backend_signal_count() -> tuple[int, int]:
    checks: list[bool] = []
    for path in [
        "crates/kiln-model/tests/cuda_sft_step_proof.rs",
        "crates/kiln-model/tests/rocm_sft_step_proof.rs",
    ]:
        source = production_source_text(path)
        checks.extend(
            [
                "OptimizerBackend::runtime_dispatch_adamw_step" in source,
                "ResidencyBackend::runtime_register_resident_activation" in source,
                "kiln_rmsnorm_kernel::adamw_step_f32_kt" not in source,
            ]
        )
    return sum(1 for check in checks if check), len(checks)


def phase_migration_signals(phase: int) -> list[dict[str, Any]]:
    if phase == 1:
        shim_count = focused_trait_forwarding_shim_count()
        method_count = len(
            parse_trait_method_names(
                ROOT / "crates" / "kiln-model" / "src" / "backend" / "mod.rs",
                "BackendRuntime",
            )
        )
        return [
            focused_trait_authoritative_signal(
                "BackendIdentity",
                "backend_identity_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "StartupBackend",
                "startup_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "AttentionBackend",
                "attention_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "GdnBackend",
                "gdn_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "ConvBackend",
                "conv_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "LinearBackend",
                "linear_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "ResidencyBackend",
                "residency_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "SamplingBackend",
                "sampling_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "OptimizerBackend",
                "optimizer_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "PagedKvBackend",
                "paged_kv_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "ReplayBackend",
                "replay_backend_facet_authoritative",
            ),
            focused_trait_authoritative_signal(
                "TrainingLossBackend",
                "training_loss_backend_facet_authoritative",
            ),
            phase_signal(
                "focused_trait_forwarding_shims_removed",
                shim_count == 0,
                shim_count,
                0,
                ["crates/kiln-model/src/backend/mod.rs"],
            ),
            phase_signal(
                "backend_runtime_method_count_below_gate",
                method_count <= 8,
                method_count,
                "<= 8",
                ["crates/kiln-model/src/backend/mod.rs"],
            ),
        ]
    if phase == 2:
        duplicate_count = decode_hot_path_duplicate_helper_count()
        shared_count, shared_expected = decode_hot_path_shared_fallback_signal_count()
        return [
            phase_signal(
                "decode_hot_path_duplicate_helpers_removed",
                duplicate_count == 0,
                duplicate_count,
                0,
                [
                    "crates/kiln-model/src/forward.rs",
                    "crates/kiln-model/src/generate.rs",
                ],
            ),
            phase_signal(
                "decode_hot_path_fallback_delegates_to_backend_capability",
                shared_count == shared_expected,
                shared_count,
                shared_expected,
                [
                    "crates/kiln-model/src/backend/capability.rs",
                    "crates/kiln-model/src/forward.rs",
                    "crates/kiln-model/src/generate.rs",
                ],
            ),
        ]
    if phase == 3:
        shim_count = resident_registry_forwarding_shim_count()
        concrete_impl_count = concrete_resident_registry_impl_count()
        delegate_count = residency_backend_facade_registry_delegate_count()
        process_global_count = resident_registry_process_global_static_count()
        drop_drains_test_count = resident_registry_drop_drains_test_count()
        lifecycle_store_count = resident_registry_lifecycle_metadata_store_count()
        return [
            phase_signal(
                "resident_registry_blanket_adapter_removed",
                shim_count == 0,
                shim_count,
                0,
                ["crates/kiln-model/src/backend/mod.rs"],
            ),
            phase_signal(
                "production_backends_implement_resident_registry",
                concrete_impl_count >= 5,
                concrete_impl_count,
                ">= 5",
                [
                    "crates/kiln-model/src/backend/cpu.rs",
                    "crates/kiln-model/src/backend/cuda.rs",
                    "crates/kiln-model/src/backend/rocm.rs",
                    "crates/kiln-model/src/backend/metal_runtime.rs",
                    "crates/kiln-model/src/backend/vulkan.rs",
                ],
            ),
            phase_signal(
                "residency_backend_facade_delegates_to_registry",
                delegate_count == 7,
                delegate_count,
                7,
                ["crates/kiln-model/src/backend/mod.rs"],
            ),
            phase_signal(
                "resident_registry_process_global_statics_removed",
                process_global_count == 0,
                process_global_count,
                0,
                [
                    "crates/kiln-model/src/backend/cuda.rs",
                    "crates/kiln-model/src/backend/rocm.rs",
                    "crates/kiln-model/src/backend/metal_residency.rs",
                    "crates/kiln-model/src/backend/vulkan_residency.rs",
                ],
            ),
            phase_signal(
                "resident_registry_drop_drains_test_present",
                drop_drains_test_count > 0,
                drop_drains_test_count,
                "> 0",
                [
                    "crates/kiln-model/src/backend/mod.rs",
                    "crates/kiln-model/src/backend/residency.rs",
                ],
            ),
            phase_signal(
                "resident_registry_lifecycle_metadata_persisted",
                lifecycle_store_count >= 3,
                lifecycle_store_count,
                ">= 3",
                [
                    "crates/kiln-model/src/backend/cuda_rocm_common.rs",
                    "crates/kiln-model/src/backend/metal_residency.rs",
                    "crates/kiln-model/src/backend/vulkan_residency.rs",
                ],
            ),
        ]
    if phase == 4:
        descriptor_count, descriptor_expected = matmul_request_descriptor_w4_1_signal_count()
        support_count, support_expected = matmul_support_query_authority_signal_count()
        transposed_count, transposed_expected = matmul_transposed_request_contract_signal_count()
        dispatch_count = matmul_identity_dispatch_count()
        return [
            phase_signal(
                "matmul_request_descriptor_w4_1_lossless",
                descriptor_count == descriptor_expected,
                descriptor_count,
                descriptor_expected,
                [
                    "crates/kiln-model/src/backend/capability.rs",
                    "crates/kiln-model/src/backend/mod.rs",
                ],
            ),
            phase_signal(
                "matmul_support_query_delegates_to_linear_backend",
                support_count == support_expected,
                support_count,
                support_expected,
                [
                    "crates/kiln-model/src/backend/capability.rs",
                    "crates/kiln-model/src/backend/mod.rs",
                    "crates/kiln-model/src/backend/cpu.rs",
                    "crates/kiln-model/src/backend/cuda.rs",
                    "crates/kiln-model/src/backend/rocm.rs",
                    "crates/kiln-model/src/backend/metal_runtime.rs",
                    "crates/kiln-model/src/backend/vulkan.rs",
                ],
            ),
            phase_signal(
                "matmul_transposed_request_contract_present",
                transposed_count == transposed_expected,
                transposed_count,
                transposed_expected,
                ["crates/kiln-model/src/backend/mod.rs"],
            ),
            phase_signal(
                "matmul_linear_identity_dispatch_removed",
                dispatch_count == 0,
                dispatch_count,
                0,
                [
                    "crates/kiln-tensor/src/ops/matmul.rs",
                    "crates/kiln-model/src/forward.rs",
                    "crates/kiln-model/src/backend/capability.rs",
                ],
            ),
        ]
    if phase == 5:
        replay_plan_count, replay_plan_expected = replay_production_replay_plan_signal_count()
        replay_contract_count = replay_contract_w5_0_signal_count()
        replay_parity_count, replay_parity_expected = replay_parity_w5_3_signal_count()
        return [
            phase_signal(
                "replay_contract_w5_0_fixed",
                replay_contract_count == 8,
                replay_contract_count,
                8,
                ["crates/kiln-graph/src/replay_plan.rs"],
            ),
            phase_signal(
                "production_replay_paths_use_replay_plan",
                replay_plan_count == replay_plan_expected,
                replay_plan_count,
                replay_plan_expected,
                [
                    "crates/kiln-model/src/cuda_graph.rs",
                    "crates/kiln-model/src/rocm_graph.rs",
                    "crates/kiln-model/src/metal_graph.rs",
                    "crates/kiln-model/src/vk_decode_resident.rs",
                ],
            ),
            phase_signal(
                "replay_parity_w5_3_live_gate",
                replay_parity_count == replay_parity_expected,
                replay_parity_count,
                replay_parity_expected,
                [
                    "crates/kiln-model/tests/backend_capability_contract.rs",
                    "crates/kiln-model/src/forward.rs",
                ],
            ),
        ]
    if phase == 6:
        device_family_count = training_precision_for_device_family_production_count()
        trait_policy_count, trait_policy_expected = (
            training_precision_backend_trait_policy_signal_count()
        )
        tape_guard_count = tape_forward_gpu_family_guard_count()
        tape_route_count, tape_route_expected = (
            tape_forward_backend_trait_route_signal_count()
        )
        sft_proof_count, sft_proof_expected = (
            sft_step_proof_optimizer_backend_signal_count()
        )
        return [
            phase_signal(
                "training_precision_for_device_family_removed_from_production",
                device_family_count == 0,
                device_family_count,
                0,
                [
                    "crates/kiln-model/src/forward.rs",
                    "crates/kiln-model/src/tape_forward.rs",
                    "crates/kiln-train/src/trainer.rs",
                ],
            ),
            phase_signal(
                "training_precision_policy_delegates_to_backend_trait",
                trait_policy_count == trait_policy_expected,
                trait_policy_count,
                trait_policy_expected,
                ["crates/kiln-model/src/backend/mod.rs"],
            ),
            phase_signal(
                "tape_forward_gpu_family_guards_removed",
                tape_guard_count == 0,
                tape_guard_count,
                0,
                ["crates/kiln-model/src/tape_forward.rs"],
            ),
            phase_signal(
                "tape_forward_route_delegates_to_backend_trait",
                tape_route_count == tape_route_expected,
                tape_route_count,
                tape_route_expected,
                [
                    "crates/kiln-model/src/backend/mod.rs",
                    "crates/kiln-model/src/tape_forward.rs",
                ],
            ),
            phase_signal(
                "sft_step_proofs_route_optimizer_backend",
                sft_proof_count == sft_proof_expected,
                sft_proof_count,
                sft_proof_expected,
                [
                    "crates/kiln-model/tests/cuda_sft_step_proof.rs",
                    "crates/kiln-model/tests/rocm_sft_step_proof.rs",
                ],
            ),
        ]
    return []


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
    phase8_migration = "complete" if phase8_status == "covered" else "partial"
    migration_by_phase = {
        0: "complete",
        7: "complete",
        8: phase8_migration,
    }
    migration_signals_by_phase = {
        phase: phase_migration_signals(phase) for phase in [1, 2, 3, 4, 5, 6]
    }
    for phase in [1, 2, 3, 4, 5, 6]:
        migration_by_phase[phase] = migration_from_signals(
            migration_signals_by_phase[phase]
        )
    phases = [
        {
            "phase": 0,
            "title": "Audit and stabilize capability reporting",
            "contract": "landed",
            "migration": migration_by_phase[0],
            "genuine": migration_by_phase[0] == "complete",
            "status": phase_status("landed", migration_by_phase[0]),
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
            "migration_signals": [],
            "remaining": [],
        },
        {
            "phase": 1,
            "title": "Introduce focused backend traits",
            "contract": "landed",
            "migration": migration_by_phase[1],
            "genuine": migration_by_phase[1] == "complete",
            "status": phase_status("landed", migration_by_phase[1]),
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
            "migration_signals": migration_signals_by_phase[1],
            "remaining": phase1_remaining_from_signals(
                migration_signals_by_phase[1]
            ),
        },
        {
            "phase": 2,
            "title": "Normalize fallback policy",
            "contract": "landed",
            "migration": migration_by_phase[2],
            "genuine": migration_by_phase[2] == "complete",
            "status": phase_status("landed", migration_by_phase[2]),
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
            "migration_signals": migration_signals_by_phase[2],
            "remaining": phase2_remaining_from_signals(
                migration_signals_by_phase[2]
            ),
        },
        {
            "phase": 3,
            "title": "Unify resident resource semantics",
            "contract": "landed",
            "migration": migration_by_phase[3],
            "genuine": migration_by_phase[3] == "complete",
            "status": phase_status("landed", migration_by_phase[3]),
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
            "migration_signals": migration_signals_by_phase[3],
            "remaining": []
            if migration_by_phase[3] == "complete"
            else phase3_remaining_from_signals(migration_signals_by_phase[3]),
        },
        {
            "phase": 4,
            "title": "Unify matmul and linear dispatch",
            "contract": "landed",
            "migration": migration_by_phase[4],
            "genuine": migration_by_phase[4] == "complete",
            "status": phase_status("landed", migration_by_phase[4]),
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
            "migration_signals": migration_signals_by_phase[4],
            "remaining": []
            if migration_by_phase[4] == "complete"
            else [
                "matmul/linear dispatch still contains backend identity branches and request routing is not authoritative",
            ],
        },
        {
            "phase": 5,
            "title": "Move replay into the authoritative graph layer",
            "contract": "landed",
            "migration": migration_by_phase[5],
            "genuine": migration_by_phase[5] == "complete",
            "status": phase_status("landed", migration_by_phase[5]),
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
                "crates/kiln-model/src/vk_decode_resident.rs",
                "crates/kiln-vulkan-kernel/src/cmd_batch.rs",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Focused Backend Facets",
                "Replay Authority",
                "Conformance And Performance Gates",
            ],
            "migration_signals": migration_signals_by_phase[5],
            "remaining": []
            if migration_by_phase[5] == "complete"
            else phase5_remaining_from_signals(migration_signals_by_phase[5]),
        },
        {
            "phase": 6,
            "title": "Finish shared training integration",
            "contract": "landed",
            "migration": migration_by_phase[6],
            "genuine": migration_by_phase[6] == "complete",
            "status": phase_status("landed", migration_by_phase[6]),
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
                "Training Loss Routing",
                "Training Precision Policy",
                "Optimizer Dispatch",
                "Conformance And Performance Gates",
            ],
            "migration_signals": migration_signals_by_phase[6],
            "remaining": []
            if migration_by_phase[6] == "complete"
            else phase6_remaining_from_signals(migration_signals_by_phase[6]),
        },
        {
            "phase": 7,
            "title": "Decompose backend modules",
            "contract": "landed",
            "migration": migration_by_phase[7],
            "genuine": migration_by_phase[7] == "complete",
            "status": phase_status("landed", migration_by_phase[7]),
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
            "migration_signals": [],
            "remaining": [],
        },
        {
            "phase": 8,
            "title": "Conformance and performance gates",
            "contract": "landed",
            "migration": migration_by_phase[8],
            "genuine": migration_by_phase[8] == "complete",
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
                "scripts/check_unification_gates.sh",
                "scripts/run_backend_latency_fixture.py",
                "scripts/write_backend_latency_result_artifact.py",
                "scripts/import_backend_latency_artifact.py",
                "scripts/lock_backend_latency_thresholds.py",
                "scripts/check_backend_latency_fixtures.py",
                "scripts/plan_backend_latency_fixture_dispatch.py",
                "scripts/generate_backend_capability_report.py",
                "crates/kiln-model/tests/backend_capability_contract.rs",
            ],
            "report_sections": [
                "Conformance And Performance Gates",
            ],
            "migration_signals": [],
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
        if phase["evidence_missing"]:
            phase["contract"] = "absent"
            phase["status"] = "partial"
            phase["genuine"] = False
            phase["remaining"] = [
                *phase["remaining"],
                "missing source evidence listed in evidence_missing",
            ]
    return phases


def optimizer_dispatch_report(backends: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for backend, info in backends.items():
        overrides = set(info["overrides"])
        source_paths = [ROOT / source for source in info.get("source_modules", [info["source"]])]
        functions = parse_backend_functions(source_paths)
        report[backend] = {
            "sgd_step": "overridden"
            if "runtime_dispatch_sgd_step" in functions or "dispatch_sgd_step" in overrides
            else "default_decline",
            "adamw_step": "overridden"
            if "runtime_dispatch_adamw_step" in functions or "dispatch_adamw_step" in overrides
            else "default_decline",
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
    lines.append("| Phase | Title | Status | Contract | Migration | Genuine | Evidence | Remaining |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for phase in data["migration_phase_status"]:
        evidence = ", ".join(f"`{path}`" for path in phase["evidence_present"]) or "none"
        remaining = "; ".join(phase["remaining"]) or "none"
        genuine = "yes" if phase["genuine"] else "no"
        lines.append(
            f"| Phase {phase['phase']} | {phase['title']} | `{phase['status']}` | "
            f"`{phase['contract']}` | `{phase['migration']}` | {genuine} | "
            f"{evidence} | {remaining} |"
        )
    lines.append("")
    lines.append("## BackendRuntime Overrides")
    lines.append("")
    lines.append("| Backend | Source Modules | Override Count | Support Methods | Native Env Gates | Legacy Env Aliases |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for backend, info in data["backends"].items():
        sources = ", ".join(f"`{source}`" for source in info.get("source_modules", [info["source"]]))
        lines.append(
            f"| `{backend}` | {sources} | {info['override_count']} | "
            f"{len(info['support_methods'])} | {len(info['native_env_gates'])} | "
            f"{len(info['legacy_env_aliases'])} |"
        )
    lines.append("")
    lines.append("## Focused Backend Facets")
    lines.append("")
    lines.append("| Facet | Method Count | Forwarding Impl | Concrete Impl Count | Concrete Impls | Methods |")
    lines.append("|---|---:|---|---:|---|---|")
    for name, info in data["focused_backend_facets"].items():
        methods = ", ".join(f"`{method}`" for method in info["methods"])
        concrete_impls = ", ".join(f"`{impl}`" for impl in info["concrete_impls"]) or "none"
        lines.append(
            f"| `{name}` | {info['method_count']} | `{info['forwarding_impl']}` | "
            f"{info['concrete_impl_count']} | {concrete_impls} | {methods} |"
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
    lines.append(
        "| Gate | Phase 8 Requirement | Status | Command | Supplemental Commands | Evidence | Missing Evidence | Coverage Blockers |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for gate in data["conformance_gates"]:
        supplemental = (
            "; ".join(
                f"`{entry['scope']}: {entry['command']}`"
                for entry in gate["supplemental_commands"]
            )
            or "none"
        )
        evidence = ", ".join(f"`{path}`" for path in gate["evidence_present"]) or "none"
        missing = ", ".join(f"`{path}`" for path in gate["evidence_missing"]) or "none"
        blockers = (
            "; ".join(f"`{blocker}`" for blocker in gate["coverage_blockers"]) or "none"
        )
        lines.append(
            f"| `{gate['gate']}` | {gate['phase8_requirement']} | "
            f"`{gate['status']}` | `{gate['command']}` | {supplemental} | "
            f"{evidence} | {missing} | {blockers} |"
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
    lines.append("| Backend | Policy | Activations | Base Weights | LoRA | Loss Accum | Optimizer Params | Mixed RMSNorm Weight | Streaming Tile | Tape Tile | Paged Medium Tile | Exact GDN Backward Tile | Mixed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for backend, info in data["training_precision_policy"].items():
        exact_gdn_tile = info["exact_gdn_backward_tile_tokens"]
        exact_gdn_tile_display = (
            str(exact_gdn_tile)
            if exact_gdn_tile is not None
            else "streaming_tile_tokens_for(device)"
        )
        paged_medium_tile = info["paged_prefill_medium_tile_tokens"]
        paged_medium_max_tokens = info["paged_prefill_medium_tile_max_tokens"]
        paged_medium_tile_display = (
            f"{paged_medium_tile} <= {paged_medium_max_tokens}"
            if paged_medium_tile is not None and paged_medium_max_tokens is not None
            else "none"
        )
        mixed_rms_norm_weight_dtype = info["mixed_rms_norm_weight_dtype"] or "none"
        lines.append(
            f"| `{backend}` | `{info['name']}` | "
            f"`{','.join(info['activation_dtypes'])}` | "
            f"`{','.join(info['base_weight_dtypes'])}` | "
            f"`{','.join(info['lora_parameter_dtypes'])}` | "
            f"`{info['loss_accumulation_dtype']}` | "
            f"`{','.join(info['optimizer_parameter_dtypes'])}` | "
            f"`{mixed_rms_norm_weight_dtype}` | "
            f"`{info['streaming_prefill_tile_tokens']}` | "
            f"`{info['tape_streaming_tile_tokens']}` | "
            f"`{paged_medium_tile_display}` | "
            f"`{exact_gdn_tile_display}` | "
            f"{'yes' if info['mixed_precision'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Training Loss Routing")
    lines.append("")
    lines.append(
        "| Backend | Tape Forward/Backward Route | SFT FLCE Route | GRPO Route | GRPO KL Auxiliary Route | OPD Route | OPD Phase-B Backward Route | Final RMSNorm Backward Route | Evidence |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for backend, info in data["training_loss_policy"].items():
        lines.append(
            f"| `{backend}` | `{info['tape_forward_backward_route']}` | "
            f"`{info['sft_flce_loss_route']}` | "
            f"`{info['grpo_loss_route']}` | "
            f"`{info['grpo_kl_auxiliary_route']}` | `{info['opd_loss_route']}` | "
            f"`{info['opd_phase_b_backward_route']}` | "
            f"`{info['final_rmsnorm_backward_route']}` | {info['evidence']} |"
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
        if info["native_env_gates"]:
            for gate in info["native_env_gates"]:
                lines.append(f"- `{gate}`")
        else:
            lines.append("- none detected")
        if info["legacy_env_aliases"]:
            lines.append("")
            lines.append("Legacy aliases honored for compatibility:")
            for gate in info["legacy_env_aliases"]:
                lines.append(f"- `{gate}`")
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
        "training_loss_policy": training_loss_policy_report(),
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
