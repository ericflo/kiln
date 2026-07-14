#!/usr/bin/env python3
"""Generate Kiln's adapter, HF/TRL, receipt, and teacher API schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "contracts" / "kiln-artifacts-v1.schema.json"
STATUS = {"x-kiln-field-schema-status": "complete"}
OBSERVABILITY_SCHEMA = "kiln-observability-v1.schema.json"
INFERENCE_SCHEMA = "kiln-inference-v1.schema.json"
ENTRYPOINTS = (
    "AdapterDetail",
    "AdapterUploadMultipart",
    "AdaptersResponse",
    "DeleteAdapterResponse",
    "DeleteExportResponse",
    "DeleteTeacherResponse",
    "ExportDetail",
    "ExportList",
    "ExportSummary",
    "GrpoExportRequest",
    "ImportPeftResponse",
    "LoadAdapterRequest",
    "LoadAdapterResponse",
    "MergeAdapterRequest",
    "MergeAdapterResponse",
    "RegisterTeacherRequest",
    "SftExportRequest",
    "TeacherEntry",
    "TeachersListResponse",
    "UnloadAdapterResponse",
    "UploadAdapterResponse",
    "kiln_train_AdapterReceipt",
)
DEFS: dict[str, dict[str, Any]] = {}


def ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


def external_ref(document: str, name: str) -> dict[str, str]:
    return {"$ref": f"{document}#/$defs/{name}"}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def array(
    schema: dict[str, Any],
    *,
    min_items: int | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "array", "items": schema}
    if min_items is not None:
        value["minItems"] = min_items
    if max_items is not None:
        value["maxItems"] = max_items
    return value


def mapping(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": schema}


def described(schema: dict[str, Any], description: str) -> dict[str, Any]:
    return {**schema, "description": description}


def exact_file(path: str) -> dict[str, Any]:
    return {
        "allOf": [
            ref("HfTrlFileIdentity"),
            {"properties": {"relative_path": {"const": path}}},
        ]
    }


def add_definition(name: str, rust_type: str, schema: dict[str, Any], description: str) -> None:
    DEFS[name] = {
        **schema,
        "description": description,
        "x-kiln-rust-type": rust_type,
        **STATUS,
    }


def add_enum(name: str, rust_type: str, values: list[str], description: str) -> None:
    add_definition(name, rust_type, {"type": "string", "enum": values}, description)


def add_object(
    name: str,
    rust_type: str,
    fields: dict[str, dict[str, Any]],
    description: str,
    *,
    optional: tuple[str, ...] = (),
    open_input: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    optional_set = set(optional)
    unknown = optional_set - set(fields)
    if unknown:
        raise ValueError(f"{name}: optional fields are not declared: {sorted(unknown)}")
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": open_input,
        "required": [field for field in fields if field not in optional_set],
        "properties": fields,
    }
    if open_input:
        schema["x-kiln-unknown-field-policy"] = "accepted_and_ignored"
    if extra:
        schema.update(extra)
    add_definition(name, rust_type, schema, description)


def build_primitives_and_enums() -> None:
    add_definition("AnyJson", "serde_json::Value", {}, "Any JSON value carried without field-level interpretation.")
    add_definition("Boolean", "bool", {"type": "boolean"}, "A serialized Rust boolean.")
    add_definition("String", "String", {"type": "string"}, "A serialized UTF-8 Rust string.")
    add_definition("NonEmptyString", "String", {"type": "string", "minLength": 1}, "A non-empty UTF-8 string.")
    add_definition(
        "ExportName",
        "String",
        {"type": "string", "minLength": 1, "maxLength": 128, "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]*$"},
        "A server-owned HF/TRL export name.",
    )
    add_definition(
        "AdapterName",
        "String",
        {
            "type": "string",
            "minLength": 1,
            "pattern": r"^(?!\.)(?![\s\S]*\.\.)(?![\s\S]*[\\/])[\s\S]+$",
            "x-kiln-runtime-footgun": "adapter names currently have no byte limit or control-character restriction",
        },
        "An adapter registry name: non-empty, relative, not dot-prefixed, and without separators or '..'.",
    )
    add_definition(
        "ImportAdapterName",
        "String",
        {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
            "pattern": r"^[A-Za-z0-9](?:[A-Za-z0-9_-]|\.(?!\.))*$",
        },
        "A path-safe adapter name accepted by verified HF/TRL PEFT import.",
    )
    add_definition("NonNegativeInteger", "u64 | u32 | usize", {"type": "integer", "minimum": 0}, "A non-negative integer.")
    add_definition("PositiveInteger", "u64 | u32 | usize", {"type": "integer", "minimum": 1}, "A positive integer.")
    add_definition("FiniteNumber", "f32 | f64", {"type": "number"}, "A finite JSON number.")
    add_definition("UnitInterval", "f32", {"type": "number", "exclusiveMinimum": 0, "maximum": 1}, "A merge density in the interval (0, 1].")
    add_definition("Sha256", "String", {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}, "A lowercase SHA-256 digest with the sha256: prefix.")
    add_definition("RawSha256", "String", {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "A raw lowercase 64-character SHA-256 digest.")
    add_definition("Rfc3339Timestamp", "String", {"type": "string", "format": "date-time"}, "An RFC 3339 timestamp.")

    add_enum("AdapterRegistryStatus", "String", ["loaded", "available", "quarantined", "invalid"], "The registry state of an adapter directory.")
    add_enum("AdapterCanaryStatus", "String", ["unknown", "passed", "quarantined"], "The current adapter canary result.")
    add_enum("TrainingJobType", "TrainingJobType", ["sft", "grpo", "opd"], "The native training workflow that produced a linked job.")
    add_enum("TrainingState", "kiln_train::TrainingState", ["queued", "running", "completed", "failed"], "A training job lifecycle state.")
    add_enum("EvalJobState", "kiln_eval::EvalJobState", ["queued", "running", "completed", "failed", "cancelled"], "An evaluation job lifecycle state.")
    add_enum("MergeMode", "String", ["weighted_average", "ties", "concat"], "The adapter merge algorithm.")
    add_enum("HfTrlTask", "kiln_train::HfTrlTask", ["sft", "grpo"], "The external HF/TRL training task.")
    add_enum("HfTrlDatasetFormat", "HfTrlDatasetFormat", ["sft_messages_jsonl", "grpo_groups_jsonl"], "The exported JSONL row format.")
    add_enum("HfTrlSftLabelPolicy", "HfTrlSftLabelPolicy", ["assistant_only_generation_spans"], "How the exported SFT template selects labels.")
    add_enum("SftInvalidRowPolicy", "SftInvalidRowPolicy", ["fail", "skip"], "How invalid SFT rows are admitted.")
    add_enum("TurnKind", "kiln_train::TurnKind", ["context", "action", "observation"], "The supervision role of one trajectory segment.")
    add_enum("TeacherKind", "TeacherKind", ["fixture", "local", "remote"], "The teacher implementation family.")
    add_enum(
        "RemoteProvider",
        "kiln_train::RemoteProvider",
        ["vllm", "sglang", "llama_cpp", "open_router", "together", "fireworks", "deep_infra", "tgi"],
        "A named remote scoring protocol. Registration currently admits only vllm.",
    )
    add_enum("TeacherStatus", "TeacherStatus", ["verified", "configured", "legacy_unverified", "unavailable"], "The registry's stable usability state.")


def build_adapter_definitions() -> None:
    add_object(
        "LoadedAdapterIdentity",
        "LoadedAdapterIdentity",
        {"name": ref("AdapterName"), "content_revision": ref("RawSha256")},
        "The exact name and immutable content revision published by the live runner.",
    )
    add_object(
        "AdapterCanaryCheckReceipt",
        "kiln_train::AdapterCanaryCheckReceipt",
        {"name": ref("NonEmptyString"), "passed": ref("Boolean"), "failure_reason": ref("NonEmptyString")},
        "One adapter smoke-check outcome.",
        optional=("failure_reason",),
    )
    add_object(
        "TrainingCheckpointPrecision",
        "kiln_train::TrainingCheckpointPrecision",
        {
            "parameter_dtype": ref("NonEmptyString"),
            "optimizer_state_dtype": ref("NonEmptyString"),
            "activation_dtype": ref("NonEmptyString"),
            "gradient_dtype": ref("NonEmptyString"),
            "stochastic_rounding": described(mapping(ref("AnyJson")), "The validated stochastic-rounding policy object."),
        },
        "The exact precision policy recorded by a native training checkpoint.",
    )
    add_object(
        "AdapterManifestFiles",
        "kiln_train::AdapterManifestFiles",
        {
            "adapter_model": ref("NonEmptyString"),
            "adapter_config": ref("NonEmptyString"),
            "train_receipt": nullable(ref("NonEmptyString")),
        },
        "Canonical filenames bound by an adapter manifest.",
    )
    add_object(
        "AdapterManifest",
        "kiln_train::AdapterManifest",
        {
            "schema_version": {"const": 1},
            "manifest_type": {"const": "kiln_adapter_manifest"},
            "adapter_name": ref("AdapterName"),
            "safetensors_hash": ref("Sha256"),
            "config_hash": ref("Sha256"),
            "receipt_hash": nullable(ref("Sha256")),
            "parent_adapter": nullable(ref("NonEmptyString")),
            "model_config_hash": nullable(ref("Sha256")),
            "training_chat_template_hash": ref("Sha256"),
            "base_weight_shard_manifest": external_ref(OBSERVABILITY_SCHEMA, "BaseWeightShardManifest"),
            "execution_provenance": external_ref(OBSERVABILITY_SCHEMA, "ExecutionProvenanceV1"),
            "training_precision": ref("TrainingCheckpointPrecision"),
            "kiln_commit": nullable(ref("NonEmptyString")),
            "training_data_hash": nullable(ref("Sha256")),
            "training_data_source": nullable(ref("NonEmptyString")),
            "training_data_path": nullable(ref("NonEmptyString")),
            "files": ref("AdapterManifestFiles"),
        },
        "Audit and content identity metadata for an installed adapter.",
        optional=("training_chat_template_hash", "base_weight_shard_manifest", "execution_provenance", "training_precision"),
    )
    add_object(
        "AdapterDiskEntry",
        "AdapterDiskEntry",
        {
            "name": ref("AdapterName"),
            "has_config": ref("Boolean"),
            "has_weights": ref("Boolean"),
            "size_bytes": ref("NonNegativeInteger"),
            "modified_at": nullable(ref("Rfc3339Timestamp")),
            "files": array(ref("String")),
            "path": nullable(ref("NonEmptyString")),
            "status": ref("AdapterRegistryStatus"),
            "adapter_model_sha256": nullable(ref("RawSha256")),
            "adapter_model_size_bytes": nullable(ref("NonNegativeInteger")),
            "rank": nullable(ref("NonNegativeInteger")),
            "alpha": nullable(ref("FiniteNumber")),
            "alpha_over_rank": nullable(ref("FiniteNumber")),
            "target_modules": array(ref("String")),
            "base_model_name_or_path": nullable(ref("NonEmptyString")),
            "parent_adapter_metadata": nullable(ref("AnyJson")),
            "canary_status": ref("AdapterCanaryStatus"),
            "canary_failure_reason": nullable(ref("NonEmptyString")),
            "canary_warnings": array(ref("String")),
            "canary_checks": array(ref("AdapterCanaryCheckReceipt")),
            "canary_status_path": nullable(ref("NonEmptyString")),
            "adapter_manifest": nullable(ref("AdapterManifest")),
            "adapter_manifest_path": nullable(ref("NonEmptyString")),
            "adapter_manifest_error": nullable(ref("NonEmptyString")),
            "last_load_error": nullable(ref("NonEmptyString")),
            "error": nullable(ref("NonEmptyString")),
        },
        "One adapter directory discovered by the server, including invalid and quarantined entries.",
    )
    add_object(
        "AdaptersResponse",
        "AdaptersResponse",
        {
            "active_adapter": nullable(ref("NonEmptyString")),
            "active": nullable(ref("NonEmptyString")),
            "loaded_adapter": nullable(ref("NonEmptyString")),
            "loaded_adapter_identity": nullable(ref("LoadedAdapterIdentity")),
            "loaded_adapters": array(ref("NonEmptyString")),
            "adapter_dir": ref("NonEmptyString"),
            "available": array(ref("AdapterDiskEntry")),
            "available_adapters": array(ref("AdapterDiskEntry")),
        },
        "The complete resident and on-disk adapter registry view.",
    )
    add_object(
        "LoadAdapterRequest",
        "LoadAdapterRequest",
        {"name": ref("AdapterName"), "allow_quarantined": {"type": "boolean", "default": False}},
        "Load an installed adapter, optionally overriding quarantine admission.",
        optional=("allow_quarantined",),
        open_input=True,
    )
    add_object(
        "LoadAdapterResponse",
        "LoadAdapterResponse",
        {"status": {"const": "loaded"}, "name": ref("AdapterName"), "content_revision": ref("RawSha256")},
        "Confirmation that an exact adapter revision is live.",
    )
    add_object("UnloadAdapterResponse", "UnloadAdapterResponse", {"status": {"const": "unloaded"}}, "Confirmation that the runtime adapter is unloaded.")
    add_object(
        "DeleteAdapterResponse",
        "DeleteAdapterResponse",
        {"status": {"const": "deleted"}, "name": ref("AdapterName")},
        "Confirmation that an adapter directory was deleted.",
    )
    add_object("MergeSource", "MergeSource", {"name": ref("AdapterName"), "weight": ref("FiniteNumber")}, "One weighted source adapter.", open_input=True)
    add_object(
        "MergeAdapterRequest",
        "MergeAdapterRequest",
        {
            "sources": array(ref("MergeSource"), min_items=1),
            "output_name": ref("AdapterName"),
            "mode": nullable(ref("MergeMode")),
            "density": nullable(ref("UnitInterval")),
        },
        "Create an immutable merged adapter. Density is accepted only for TIES and defaults to 0.2 there.",
        optional=("mode", "density"),
        open_input=True,
        extra={
            "allOf": [
                {
                    "if": {"required": ["density"], "properties": {"density": {"type": "number"}}},
                    "then": {"required": ["mode"], "properties": {"mode": {"const": "ties"}}},
                }
            ],
            "x-kiln-semantic-constraints": ["source names must be distinct", "output_name must not already exist"],
        },
    )
    add_object("MergeSourceInfo", "MergeSourceInfo", {"name": ref("AdapterName"), "weight": ref("FiniteNumber")}, "One source echoed after a merge.")
    add_object(
        "MergeAdapterResponse",
        "MergeAdapterResponse",
        {
            "status": {"const": "merged"},
            "output_name": ref("AdapterName"),
            "mode": ref("MergeMode"),
            "sources": array(ref("MergeSourceInfo"), min_items=1),
            "num_tensors": ref("PositiveInteger"),
        },
        "The published adapter merge summary.",
    )
    add_object(
        "AdapterUploadMultipart",
        "AdapterUploadMultipart",
        {
            "name": ref("AdapterName"),
            "archive": {"type": "string", "format": "binary", "contentMediaType": "application/gzip"},
        },
        "Multipart adapter installation body. Unknown parts are drained and ignored.",
        open_input=True,
        extra={"x-kiln-max-body-bytes": 2147483648, "x-kiln-max-extracted-bytes": 4294967296, "x-kiln-max-extracted-entries": 100000},
    )
    add_object(
        "UploadAdapterResponse",
        "UploadAdapterResponse",
        {"name": ref("AdapterName"), "size_bytes": ref("NonNegativeInteger"), "files": ref("PositiveInteger")},
        "The atomically installed adapter archive summary.",
    )
    add_object("AdapterFileEntry", "AdapterFileEntry", {"name": ref("NonEmptyString"), "size_bytes": ref("NonNegativeInteger")}, "One regular file in an adapter directory.")
    add_object(
        "AdapterLinkedJob",
        "AdapterLinkedJob",
        {
            "job_id": ref("NonEmptyString"),
            "job_type": ref("TrainingJobType"),
            "state": ref("TrainingState"),
            "elapsed_secs": {"type": "number", "minimum": 0},
            "final_loss": nullable(ref("FiniteNumber")),
        },
        "A retained training job linked to an adapter.",
    )
    add_object(
        "AdapterLinkedEval",
        "AdapterLinkedEval",
        {
            "job_id": ref("NonEmptyString"),
            "suite_name": ref("NonEmptyString"),
            "accuracy": nullable(ref("FiniteNumber")),
            "state": ref("EvalJobState"),
        },
        "A retained evaluation linked to an adapter.",
    )
    add_object(
        "AdapterDetail",
        "AdapterDetail",
        {
            "name": ref("AdapterName"),
            "is_active": ref("Boolean"),
            "has_config": ref("Boolean"),
            "has_weights": ref("Boolean"),
            "size_bytes": ref("NonNegativeInteger"),
            "files": array(ref("AdapterFileEntry")),
            "training_jobs": array(ref("AdapterLinkedJob")),
            "eval_jobs": array(ref("AdapterLinkedEval")),
            "lineage": ref("AnyJson"),
        },
        "Adapter file inventory plus retained training, evaluation, and lineage evidence.",
        optional=("lineage",),
    )


def build_receipt_definitions() -> None:
    add_object(
        "TeacherAdapterIdentityV1",
        "kiln_train::TeacherAdapterIdentityV1",
        {"name": ref("NonEmptyString"), "weights_sha256": ref("RawSha256"), "config_sha256": ref("RawSha256")},
        "Exact static adapter identity served by a teacher.",
    )
    add_object(
        "TeacherIdentityV1",
        "kiln_train::TeacherIdentityV1",
        {
            "schema": {"const": "kiln.teacher-identity.v1"},
            "protocol": {"const": "vllm.prompt-logprobs.numeric-token-ids.causal.v1"},
            "served_model_id": ref("NonEmptyString"),
            "base_model_sha256": ref("RawSha256"),
            "tokenizer_vocab_sha256": ref("RawSha256"),
            "tokenizer_config_sha256": ref("RawSha256"),
            "adapter": nullable(ref("TeacherAdapterIdentityV1")),
            "vocab_size": {"type": "integer", "minimum": 1, "maximum": 16777216},
            "max_top_k": {"type": "integer", "minimum": 1, "maximum": 65536},
            "max_model_len": {"type": "integer", "minimum": 1, "maximum": 16777216},
            "max_prompt_logprob_candidates": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "logprobs_mode": {"const": "raw_logprobs"},
            "implementation": ref("NonEmptyString"),
            "inference_config_sha256": ref("RawSha256"),
        },
        "Canonical content and protocol identity pinned by teacher registration.",
    )
    add_object(
        "TeacherDescriptor",
        "kiln_train::TeacherDescriptor",
        {
            "alias": ref("NonEmptyString"),
            "model_id": ref("NonEmptyString"),
            "model_version_hash": ref("NonEmptyString"),
            "identity": ref("TeacherIdentityV1"),
            "snapshot_url": ref("NonEmptyString"),
        },
        "Teacher audit identity retained by a legacy adapter receipt.",
        optional=("model_version_hash", "identity", "snapshot_url"),
    )
    add_object(
        "PromptSourceDescriptor",
        "kiln_train::PromptSourceDescriptor",
        {"source": ref("NonEmptyString"), "manifest_hash": ref("Sha256"), "count": ref("NonNegativeInteger")},
        "Seed prompt-source audit descriptor.",
    )
    add_object(
        "EchoDiagnosticSummary",
        "kiln_train::EchoDiagnosticSummary",
        {
            "lambda": ref("FiniteNumber"),
            "env_ce_initial": ref("FiniteNumber"),
            "env_ce_final": ref("FiniteNumber"),
            "env_ce_drop_pct": ref("FiniteNumber"),
            "lambda_effective_final": ref("FiniteNumber"),
            "env_tokens_supervised": ref("NonNegativeInteger"),
            "dynamics_holdout_ce_initial": ref("FiniteNumber"),
            "dynamics_holdout_ce_final": ref("FiniteNumber"),
        },
        "Receipt-grade ECHO training diagnostics.",
        optional=("env_ce_initial", "env_ce_final", "env_ce_drop_pct", "lambda_effective_final", "dynamics_holdout_ce_initial", "dynamics_holdout_ce_final"),
    )
    add_object(
        "DiagnosticSummary",
        "kiln_train::DiagnosticSummary",
        {
            "overlap_ratio_final": ref("FiniteNumber"),
            "rep_rate_max": ref("FiniteNumber"),
            "guardrail_triggers": array(ref("String")),
            "final_loss": ref("FiniteNumber"),
            "echo": ref("EchoDiagnosticSummary"),
        },
        "Coarse post-mortem diagnostics from an adapter training run.",
        optional=("overlap_ratio_final", "rep_rate_max", "guardrail_triggers", "final_loss", "echo"),
    )
    add_object(
        "kiln_train_AdapterReceipt",
        "kiln_train::AdapterReceipt",
        {
            "schema_version": {"const": 1},
            "adapter": ref("AdapterName"),
            "produced_at": ref("Rfc3339Timestamp"),
            "kiln_version": ref("NonEmptyString"),
            "kernel_versions": mapping(ref("String")),
            "seed": ref("NonNegativeInteger"),
            "source_kind": ref("NonEmptyString"),
            "teacher": ref("TeacherDescriptor"),
            "prompts": ref("PromptSourceDescriptor"),
            "hyperparameters": ref("AnyJson"),
            "diagnostic_summary": ref("DiagnosticSummary"),
            "post_eval": mapping(ref("FiniteNumber")),
        },
        "Legacy adapter audit receipt. It supports drift analysis, not exact replay.",
        optional=("teacher", "prompts"),
    )


def source_selection_rules(source_fields: tuple[str, ...], list_field: str) -> dict[str, Any]:
    branches = []
    for selected in source_fields:
        properties: dict[str, Any] = {}
        for field in source_fields:
            if field == list_field:
                properties[field] = {"minItems": 1} if field == selected else {"maxItems": 0}
            elif field == selected:
                properties[field] = {"type": "string", "minLength": 1}
            else:
                properties[field] = {"type": "null"}
        branches.append({"required": [selected], "properties": properties})
    return {"oneOf": branches}


def build_hf_trl_definitions() -> None:
    add_object(
        "TrainingChatMessage",
        "kiln_core::tokenizer::ChatMessage",
        {
            "role": ref("NonEmptyString"),
            "content": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "null"},
                    {"type": "array", "items": ref("AnyJson")},
                ],
                "default": "",
            },
            "tool_calls": nullable(array(ref("AnyJson"))),
            "name": nullable(ref("String")),
            "tool_call_id": nullable(ref("String")),
        },
        "A training chat message. Text content parts are concatenated and non-text parts ignored.",
        optional=("content", "tool_calls", "name", "tool_call_id"),
        open_input=True,
    )
    add_object(
        "SftExample",
        "kiln_train::SftExample",
        {"messages": array(ref("TrainingChatMessage"))},
        "One supervised conversation.",
        optional=("messages",),
        open_input=True,
    )
    add_object(
        "TurnSegment",
        "kiln_train::TurnSegment",
        {
            "role": ref("NonEmptyString"),
            "content": ref("String"),
            "kind": ref("TurnKind"),
            "tool_call_id": nullable(ref("String")),
            "warning_prefix_len": nullable(ref("NonNegativeInteger")),
        },
        "One context, action, or observation segment in an agentic rollout.",
        optional=("kind", "tool_call_id", "warning_prefix_len"),
        open_input=True,
    )
    add_object(
        "ScoredRollout",
        "kiln_train::ScoredRollout",
        {
            "text": ref("String"),
            "reward": ref("FiniteNumber"),
            "trajectory": array(ref("TurnSegment")),
            "provenance": nullable(external_ref(INFERENCE_SCHEMA, "RolloutProvenanceV1")),
        },
        "One reward-scored rollout with optional multi-turn and generation provenance.",
        optional=("trajectory", "provenance"),
        open_input=True,
    )
    add_object(
        "AgenticGroup",
        "kiln_train::GrpoGroup",
        {
            "messages": array(ref("TrainingChatMessage"), min_items=1),
            "completions": array(
                {"allOf": [ref("ScoredRollout"), {"required": ["provenance"]}]},
                min_items=2,
                max_items=1024,
            ),
            "rollouts": array(
                {"allOf": [ref("ScoredRollout"), {"required": ["provenance"]}]},
                min_items=2,
                max_items=1024,
            ),
        },
        "A prompt and its group-relative scored rollouts. rollouts is a request alias for completions.",
        optional=("completions", "rollouts"),
        open_input=True,
        extra={"oneOf": [{"required": ["completions"]}, {"required": ["rollouts"]}]},
    )
    add_object(
        "SftExportRequest",
        "SftExportRequest",
        {
            "name": ref("ExportName"),
            "examples": array(ref("SftExample")),
            "dataset_path": nullable(ref("NonEmptyString")),
            "dataset": nullable(ref("NonEmptyString")),
            "invalid_row_policy": {**ref("SftInvalidRowPolicy"), "default": "fail"},
            "input_adapter": nullable(ref("AdapterName")),
            "split_manifest": nullable(ref("AnyJson")),
        },
        "Create an immutable SFT handoff from exactly one inline, filesystem, or named-dataset source.",
        optional=("examples", "dataset_path", "dataset", "invalid_row_policy", "input_adapter", "split_manifest"),
        extra=source_selection_rules(("examples", "dataset_path", "dataset"), "examples"),
    )
    add_object(
        "GrpoExportRequest",
        "GrpoExportRequest",
        {
            "name": ref("ExportName"),
            "groups": array(ref("AgenticGroup"), max_items=10000000),
            "dataset_path": nullable(ref("NonEmptyString")),
            "input_adapter": nullable(ref("AdapterName")),
            "split_manifest": nullable(ref("AnyJson")),
        },
        "Create an immutable GRPO handoff from exactly one inline or filesystem source.",
        optional=("groups", "dataset_path", "input_adapter", "split_manifest"),
        extra=source_selection_rules(("groups", "dataset_path"), "groups"),
    )
    add_object(
        "HfTrlFileIdentity",
        "kiln_train::HfTrlFileIdentity",
        {"relative_path": ref("NonEmptyString"), "size_bytes": ref("PositiveInteger"), "sha256": ref("Sha256")},
        "Exact identity of one regular file in an interoperability bundle.",
    )
    add_object(
        "HfTrlModelIdentity",
        "kiln_train::HfTrlModelIdentity",
        {
            "served_model_id": ref("NonEmptyString"),
            "base_weight_shard_manifest": external_ref(OBSERVABILITY_SCHEMA, "BaseWeightShardManifest"),
            "tokenizer_vocab_sha256": ref("Sha256"),
            "model_config": exact_file("kiln_model_config.json"),
            "tokenizer": exact_file("tokenizer.json"),
            "chat_template": exact_file("chat_template.jinja"),
            "native_training_chat_template": exact_file("kiln_training_chat_template.jinja"),
            "trl_training_chat_template": exact_file("training_chat_template.jinja"),
        },
        "Model bytes and template identities an external trainer must use.",
    )
    add_object(
        "HfTrlSftSelection",
        "kiln_train::HfTrlSftSelection",
        {
            "invalid_row_policy": ref("SftInvalidRowPolicy"),
            "label_policy": ref("HfTrlSftLabelPolicy"),
            "rows_read": ref("PositiveInteger"),
            "rows_kept": ref("PositiveInteger"),
            "rows_rejected": ref("NonNegativeInteger"),
            "kept_corpus_sha256": ref("Sha256"),
            "ingestion_receipt": exact_file("sft_ingestion.json"),
        },
        "Compact SFT row-selection evidence.",
        extra={"x-kiln-semantic-constraints": ["rows_read = rows_kept + rows_rejected"]},
    )
    add_object(
        "HfTrlDataExport",
        "kiln_train::HfTrlDataExport",
        {
            "source_name": ref("NonEmptyString"),
            "format": ref("HfTrlDatasetFormat"),
            "row_count": ref("PositiveInteger"),
            "ordered_corpus_sha256": ref("Sha256"),
            "dataset": exact_file("train.jsonl"),
            "sft_selection": ref("HfTrlSftSelection"),
            "rollout_provenance_schema": ref("NonEmptyString"),
            "split_manifest": exact_file("split_manifest.json"),
        },
        "Dataset bytes and selection/provenance policy exported to HF/TRL.",
        optional=("sft_selection", "rollout_provenance_schema", "split_manifest"),
        extra={
            "allOf": [
                {
                    "if": {"properties": {"format": {"const": "sft_messages_jsonl"}}},
                    "then": {"required": ["sft_selection"], "not": {"required": ["rollout_provenance_schema"]}},
                    "else": {"required": ["rollout_provenance_schema"], "properties": {"rollout_provenance_schema": {"const": "kiln.rollout-provenance.v1"}}, "not": {"required": ["sft_selection"]}},
                }
            ]
        },
    )
    add_object(
        "HfTrlInputAdapter",
        "kiln_train::HfTrlInputAdapter",
        {
            "name": ref("AdapterName"),
            "config": exact_file("input_adapter/adapter_config.json"),
            "model": exact_file("input_adapter/adapter_model.safetensors"),
            "kiln_manifest": exact_file("input_adapter/adapter_manifest.json"),
        },
        "Exact optional Kiln adapter copied into an interoperability bundle.",
        optional=("kiln_manifest",),
    )
    add_object(
        "HfTrlExportManifestV1",
        "kiln_train::HfTrlExportManifestV1",
        {
            "schema_version": {"const": 1},
            "manifest_type": {"const": "kiln.hf-trl-export.v1"},
            "task": ref("HfTrlTask"),
            "source_execution_provenance": external_ref(OBSERVABILITY_SCHEMA, "ExecutionProvenanceV1"),
            "model": ref("HfTrlModelIdentity"),
            "data": ref("HfTrlDataExport"),
            "reference_script": exact_file("train.py"),
            "environment_lock": exact_file("requirements.lock"),
            "input_adapter": ref("HfTrlInputAdapter"),
            "export_sha256": ref("Sha256"),
        },
        "Immutable identity of one server-owned Kiln-to-HF/TRL export directory.",
        optional=("input_adapter",),
        extra={
            "allOf": [
                {
                    "if": {"properties": {"task": {"const": "sft"}}},
                    "then": {"properties": {"data": {"properties": {"format": {"const": "sft_messages_jsonl"}}}}},
                    "else": {"properties": {"data": {"properties": {"format": {"const": "grpo_groups_jsonl"}}}}},
                }
            ],
            "x-kiln-semantic-constraints": [
                "model file identities match source_execution_provenance.model",
                "base_weight_shard_manifest matches source_execution_provenance model content",
                "export_sha256 matches the canonical manifest digest",
            ],
        },
    )
    add_object(
        "ExportSummary",
        "ExportSummary",
        {
            "name": ref("ExportName"),
            "task": ref("HfTrlTask"),
            "export_sha256": ref("Sha256"),
            "source_name": ref("NonEmptyString"),
            "row_count": ref("PositiveInteger"),
            "ordered_corpus_sha256": ref("Sha256"),
            "input_adapter": nullable(ref("AdapterName")),
            "bundle_filename": ref("NonEmptyString"),
            "download_url": ref("NonEmptyString"),
        },
        "Compact immutable HF/TRL export identity.",
    )
    add_object("ExportList", "ExportList", {"data": array(ref("ExportSummary"))}, "All immutable HF/TRL exports currently retained by the server.")
    add_object("ExportDetail", "ExportDetail", {"summary": ref("ExportSummary"), "manifest": ref("HfTrlExportManifestV1")}, "An export summary plus its full self-verifying handoff manifest.")
    add_object("DeleteExportResponse", "DeleteExportResponse", {"status": {"const": "deleted"}, "name": ref("ExportName")}, "Confirmation that an immutable export bundle was deleted.")
    add_object(
        "ImportPeftResponse",
        "ImportPeftResponse",
        {
            "status": {"const": "imported"},
            "name": ref("ImportAdapterName"),
            "task": ref("HfTrlTask"),
            "export_sha256": ref("Sha256"),
            "result_sha256": ref("Sha256"),
            "import_sha256": ref("Sha256"),
            "content_revision": ref("RawSha256"),
            "used_exported_reference_script": ref("Boolean"),
            "size_bytes": ref("PositiveInteger"),
            "files": ref("PositiveInteger"),
        },
        "The verified external PEFT result installed into Kiln's adapter registry.",
    )


def build_teacher_definitions() -> None:
    add_object(
        "LogitSourceCaps",
        "kiln_train::LogitSourceCaps",
        {
            "teacher_id": ref("NonEmptyString"),
            "vocab_size": ref("PositiveInteger"),
            "max_top_k": ref("PositiveInteger"),
            "supports_full_vocab": ref("Boolean"),
            "supports_batched": ref("Boolean"),
            "tokenizer_hash": nullable(ref("NonEmptyString")),
        },
        "Resolved scoring capabilities for a teacher source.",
    )
    add_object(
        "TeacherSpec",
        "TeacherSpec",
        {
            "alias": ref("NonEmptyString"),
            "kind": ref("TeacherKind"),
            "provider": nullable(ref("RemoteProvider")),
            "model_id": ref("String"),
            "max_top_k": nullable(ref("NonNegativeInteger")),
            "vocab_size": nullable(ref("PositiveInteger")),
            "supports_full_vocab": nullable(ref("Boolean")),
            "tokenizer_hash": nullable(ref("NonEmptyString")),
            "identity": ref("TeacherIdentityV1"),
            "url": nullable(ref("NonEmptyString")),
            "credential_id": ref("NonEmptyString"),
            "notes": nullable(ref("String")),
            "adapter": nullable(ref("AdapterName")),
        },
        "Persisted teacher registry description. Identity and capability claims are server-controlled.",
        optional=("identity", "credential_id"),
    )
    add_object(
        "TeacherEntry",
        "TeacherEntry",
        {
            "spec": ref("TeacherSpec"),
            "capabilities": nullable(ref("LogitSourceCaps")),
            "status": ref("TeacherStatus"),
            "usable": ref("Boolean"),
            "identity_revision": ref("Sha256"),
            "off_policy_manifest": ref("NonEmptyString"),
            "status_message": ref("NonEmptyString"),
        },
        "A persisted teacher plus its resolved identity, capabilities, and usability state.",
        optional=("identity_revision", "off_policy_manifest", "status_message"),
    )
    add_object("TeachersListResponse", "TeachersListResponse", {"teachers": array(ref("TeacherEntry"))}, "The complete teacher registry view.")
    add_object(
        "RegisterTeacherRequest",
        "RegisterTeacherRequest",
        {
            "alias": ref("NonEmptyString"),
            "kind": ref("TeacherKind"),
            "provider": nullable(ref("RemoteProvider")),
            "model_id": nullable(ref("NonEmptyString")),
            "url": nullable(ref("NonEmptyString")),
            "credential_id": nullable(ref("NonEmptyString")),
            "notes": nullable(ref("String")),
            "adapter": nullable(ref("AdapterName")),
        },
        "Caller-controlled teacher registration fields. Identity, capabilities, and secret environment names are rejected.",
        optional=("provider", "model_id", "url", "credential_id", "notes", "adapter"),
        extra={
            "allOf": [
                {
                    "if": {"properties": {"kind": {"const": "remote"}}},
                    "then": {
                        "required": ["provider", "model_id", "url"],
                        "properties": {
                            "provider": {"const": "vllm"},
                            "model_id": {"type": "string", "minLength": 1},
                            "url": {"type": "string", "minLength": 1},
                            "adapter": {"type": "null"},
                        },
                    },
                    "else": {
                        "properties": {"provider": {"type": "null"}, "url": {"type": "null"}, "credential_id": {"type": "null"}},
                    },
                },
                {
                    "if": {"properties": {"kind": {"const": "fixture"}}},
                    "then": {
                        "required": ["model_id"],
                        "properties": {
                            "model_id": {"type": "string", "minLength": 1},
                            "adapter": {"type": "null"},
                        },
                    },
                },
            ]
        },
    )
    add_object(
        "DeleteTeacherResponse",
        "DeleteTeacherResponse",
        {"status": {"const": "deleted"}, "alias": ref("NonEmptyString")},
        "Confirmation that a teacher alias was removed.",
    )


def hashes(char: str, *, prefixed: bool = True) -> str:
    value = char * 64
    return f"sha256:{value}" if prefixed else value


def file_identity(path: str, char: str, size: int = 128) -> dict[str, Any]:
    return {"relative_path": path, "size_bytes": size, "sha256": hashes(char)}


def base_weight_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_type": "kiln.base-weight-shards.v1",
        "aggregate_algorithm": "kiln.base-model-content.v1",
        "aggregate_sha256": hashes("a"),
        "total_size_bytes": 1024,
        "shards": [{"filename": "model.safetensors", "size_bytes": 1024, "sha256": hashes("b")}],
    }


def execution_provenance() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provenance_type": "kiln.execution-provenance.v1",
        "backend": {"name": "rocm", "device": "gfx1151", "numerical_runtime_sha256": hashes("c")},
        "build": {"package_version": "0.1.0", "target": "x86_64-unknown-linux-gnu", "executable_sha256": hashes("d")},
        "model": {
            "model_config_sha256": hashes("e"),
            "tokenizer_vocab_sha256": hashes("f"),
            "tokenizer_config_sha256": hashes("1"),
            "chat_template_sha256": hashes("2"),
            "training_chat_template_sha256": hashes("3"),
        },
        "precision": {"inference_dtype": "bf16", "training_policy": "bf16"},
        "kernels": {
            "contract_type": "kiln.kernel-contract.v1",
            "versions": {"kiln": "0.1.0"},
            "compiled_features": ["rocm"],
            "contract_sha256": hashes("4"),
        },
        "configuration": {"effective_server_config_sha256": hashes("5"), "effective_environment_sha256": hashes("6")},
        "provenance_sha256": hashes("7"),
    }


def rollout_provenance(seed: int, token_id: int) -> dict[str, Any]:
    return {
        "schema": "kiln.rollout-provenance.v1",
        "input_token_ids": [1, token_id],
        "prompt_token_count": 1,
        "prompt_messages_sha256": hashes("8"),
        "scored_payload_sha256": hashes("9"),
        "action_tokens": [
            {
                "sequence_index": 1,
                "token_id": token_id,
                "source": "sampled",
                "behavior_logprob": -0.25,
            }
        ],
        "behavior_policy": {
            "served_model_id": "Qwen/Qwen3.5-4B",
            "base_model_sha256": hashes("a"),
            "inference_config_sha256": hashes("b"),
            "implementation": "kiln/rocm",
        },
        "tokenizer": {
            "vocab_sha256": hashes("c"),
            "config_sha256": hashes("d"),
            "chat_template_sha256": hashes("e"),
        },
        "sampling": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 64,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": [],
        },
        "seed": seed,
        "generation_backend": "rocm",
    }


def export_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_type": "kiln.hf-trl-export.v1",
        "task": "sft",
        "source_execution_provenance": execution_provenance(),
        "model": {
            "served_model_id": "Qwen/Qwen3.5-4B",
            "base_weight_shard_manifest": base_weight_manifest(),
            "tokenizer_vocab_sha256": hashes("f"),
            "model_config": file_identity("kiln_model_config.json", "8"),
            "tokenizer": file_identity("tokenizer.json", "9"),
            "chat_template": file_identity("chat_template.jinja", "a"),
            "native_training_chat_template": file_identity("kiln_training_chat_template.jinja", "b"),
            "trl_training_chat_template": file_identity("training_chat_template.jinja", "c"),
        },
        "data": {
            "source_name": "inline",
            "format": "sft_messages_jsonl",
            "row_count": 1,
            "ordered_corpus_sha256": hashes("d"),
            "dataset": file_identity("train.jsonl", "e"),
            "sft_selection": {
                "invalid_row_policy": "fail",
                "label_policy": "assistant_only_generation_spans",
                "rows_read": 1,
                "rows_kept": 1,
                "rows_rejected": 0,
                "kept_corpus_sha256": hashes("d"),
                "ingestion_receipt": file_identity("sft_ingestion.json", "f"),
            },
        },
        "reference_script": file_identity("train.py", "1"),
        "environment_lock": file_identity("requirements.lock", "2"),
        "export_sha256": summary["export_sha256"],
    }


def build_examples() -> dict[str, list[Any]]:
    disk_entry = {
        "name": "reasoning-v1",
        "has_config": True,
        "has_weights": True,
        "size_bytes": 4096,
        "modified_at": "2026-07-14T12:00:00Z",
        "files": ["adapter_config.json", "adapter_model.safetensors"],
        "path": "/srv/kiln/adapters/reasoning-v1",
        "status": "loaded",
        "adapter_model_sha256": hashes("a", prefixed=False),
        "adapter_model_size_bytes": 3072,
        "rank": 16,
        "alpha": 32.0,
        "alpha_over_rank": 2.0,
        "target_modules": ["q_proj", "v_proj"],
        "base_model_name_or_path": "Qwen/Qwen3.5-4B",
        "parent_adapter_metadata": None,
        "canary_status": "passed",
        "canary_failure_reason": None,
        "canary_warnings": [],
        "canary_checks": [{"name": "finite_logits", "passed": True}],
        "canary_status_path": "/srv/kiln/adapters/reasoning-v1/adapter_canary_status.json",
        "adapter_manifest": None,
        "adapter_manifest_path": None,
        "adapter_manifest_error": None,
        "last_load_error": None,
        "error": None,
    }
    summary = {
        "name": "math-sft-v1",
        "task": "sft",
        "export_sha256": hashes("3"),
        "source_name": "inline",
        "row_count": 1,
        "ordered_corpus_sha256": hashes("d"),
        "input_adapter": None,
        "bundle_filename": "math-sft-v1.kiln-hf",
        "download_url": "/v1/train/hf/exports/math-sft-v1/download",
    }
    teacher_entry = {
        "spec": {
            "alias": "fixture-teacher",
            "kind": "fixture",
            "provider": None,
            "model_id": "fixture/model",
            "max_top_k": 20,
            "vocab_size": 248320,
            "supports_full_vocab": False,
            "tokenizer_hash": None,
            "url": None,
            "notes": "deterministic local fixture",
            "adapter": None,
        },
        "capabilities": {
            "teacher_id": "fixture-teacher",
            "vocab_size": 248320,
            "max_top_k": 20,
            "supports_full_vocab": False,
            "supports_batched": True,
            "tokenizer_hash": None,
        },
        "status": "configured",
        "usable": True,
    }
    return {
        "AdapterDetail": [{
            "name": "reasoning-v1", "is_active": True, "has_config": True, "has_weights": True,
            "size_bytes": 4096, "files": [{"name": "adapter_model.safetensors", "size_bytes": 3072}],
            "training_jobs": [{"job_id": "train-1", "job_type": "sft", "state": "completed", "elapsed_secs": 12.5, "final_loss": 0.12}],
            "eval_jobs": [{"job_id": "eval-1", "suite_name": "math", "accuracy": 0.75, "state": "completed"}],
        }],
        "AdapterUploadMultipart": [{"name": "reasoning-v1", "archive": "<binary gzip body>"}],
        "AdaptersResponse": [{
            "active_adapter": "reasoning-v1", "active": "reasoning-v1", "loaded_adapter": "reasoning-v1",
            "loaded_adapter_identity": {"name": "reasoning-v1", "content_revision": hashes("b", prefixed=False)},
            "loaded_adapters": ["reasoning-v1"], "adapter_dir": "/srv/kiln/adapters",
            "available": [disk_entry], "available_adapters": [disk_entry],
        }],
        "DeleteAdapterResponse": [{"status": "deleted", "name": "reasoning-v1"}],
        "DeleteExportResponse": [{"status": "deleted", "name": "math-sft-v1"}],
        "DeleteTeacherResponse": [{"status": "deleted", "alias": "fixture-teacher"}],
        "ExportDetail": [{"summary": summary, "manifest": export_manifest(summary)}],
        "ExportList": [{"data": [summary]}],
        "ExportSummary": [summary],
        "GrpoExportRequest": [{
            "name": "math-grpo-v1",
            "groups": [{
                "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                "completions": [
                    {"text": "4", "reward": 1.0, "provenance": rollout_provenance(42, 4)},
                    {"text": "5", "reward": 0.0, "provenance": rollout_provenance(43, 5)},
                ],
            }],
        }],
        "ImportPeftResponse": [{
            "status": "imported", "name": "math-sft-result", "task": "sft",
            "export_sha256": hashes("3"), "result_sha256": hashes("4"), "import_sha256": hashes("5"),
            "content_revision": hashes("6", prefixed=False), "used_exported_reference_script": True,
            "size_bytes": 4096, "files": 3,
        }],
        "LoadAdapterRequest": [{"name": "reasoning-v1"}],
        "LoadAdapterResponse": [{"status": "loaded", "name": "reasoning-v1", "content_revision": hashes("b", prefixed=False)}],
        "MergeAdapterRequest": [{
            "sources": [{"name": "math-v1", "weight": 0.6}, {"name": "code-v1", "weight": 0.4}],
            "output_name": "blended-v1", "mode": "weighted_average",
        }],
        "MergeAdapterResponse": [{
            "status": "merged", "output_name": "blended-v1", "mode": "weighted_average",
            "sources": [{"name": "math-v1", "weight": 0.6}, {"name": "code-v1", "weight": 0.4}], "num_tensors": 64,
        }],
        "RegisterTeacherRequest": [{"alias": "fixture-teacher", "kind": "fixture", "model_id": "fixture/model", "notes": "deterministic local fixture"}],
        "SftExportRequest": [{
            "name": "math-sft-v1", "examples": [{"messages": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4"}]}],
            "invalid_row_policy": "fail",
        }],
        "TeacherEntry": [teacher_entry],
        "TeachersListResponse": [{"teachers": [teacher_entry]}],
        "UnloadAdapterResponse": [{"status": "unloaded"}],
        "UploadAdapterResponse": [{"name": "reasoning-v1", "size_bytes": 4096, "files": 2}],
        "kiln_train_AdapterReceipt": [{
            "schema_version": 1, "adapter": "reasoning-v1", "produced_at": "2026-07-14T12:00:00Z",
            "kiln_version": "0.1.0", "kernel_versions": {"kiln-opd-loss-kernel": "0.1.0"},
            "seed": 42, "source_kind": "sft", "hyperparameters": {"rank": 16},
            "diagnostic_summary": {"final_loss": 0.12}, "post_eval": {"math": 0.75},
        }],
    }


def build_schema() -> dict[str, Any]:
    build_primitives_and_enums()
    build_adapter_definitions()
    build_receipt_definitions()
    build_hf_trl_definitions()
    build_teacher_definitions()
    examples = build_examples()
    if set(examples) != set(ENTRYPOINTS):
        raise ValueError("artifact example coverage does not match public entrypoints")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ericflo.github.io/kiln/contracts/kiln-artifacts-v1.schema.json",
        "title": "Kiln Artifact Lifecycle API",
        "description": (
            "Complete field-level wire contract for installed adapters, live adapter transitions, "
            "immutable HF/TRL handoffs and PEFT imports, adapter audit receipts, and teacher registry identities. "
            "Open input objects explicitly identify serde-compatible ignored unknown fields; emitted objects are closed."
        ),
        "x-kiln-field-schema-status": "complete",
        "x-kiln-entrypoints": list(ENTRYPOINTS),
        "x-kiln-external-contracts": [OBSERVABILITY_SCHEMA, INFERENCE_SCHEMA],
        "oneOf": [ref(name) for name in ENTRYPOINTS],
        "$defs": dict(sorted(DEFS.items())),
        "x-kiln-examples": examples,
    }


def rendered_schema() -> str:
    return json.dumps(build_schema(), indent=2, ensure_ascii=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated schema is stale")
    args = parser.parse_args()
    rendered = rendered_schema()
    if args.check:
        try:
            existing = OUTPUT.read_text()
        except OSError as error:
            parser.error(f"cannot read {OUTPUT.relative_to(ROOT)}: {error}")
        if existing != rendered:
            parser.error(f"{OUTPUT.relative_to(ROOT)} is stale; run {Path(__file__).name}")
        print(f"Artifact schema is current: {len(DEFS)} reachable definitions, {len(ENTRYPOINTS)} entrypoints")
        return 0
    OUTPUT.write_text(rendered)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(DEFS)} definitions, {len(ENTRYPOINTS)} entrypoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
