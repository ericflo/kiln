#!/usr/bin/env python3
"""Generate Kiln's eval, dataset-synthesis, and judgment API schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "contracts" / "kiln-evals-v1.schema.json"
OBSERVABILITY_SCHEMA = "kiln-observability-v1.schema.json"
THINKING_SCHEMA = "thinking-budget-v1.schema.json"
STATUS = {"x-kiln-field-schema-status": "complete"}
ENTRYPOINTS = (
    "AppendJudgmentBody",
    "AppendJudgmentResponse",
    "CancelEvalJobResponse",
    "CompileJudgmentBody",
    "CompileJudgmentResponse",
    "CreateJudgmentBody",
    "DatasetListResponse",
    "DatasetManifest",
    "DatasetUploadMultipart",
    "DeleteDatasetResponse",
    "DeleteJudgmentResponse",
    "DeleteSuiteResponse",
    "EvalCompareSpec",
    "EvalJobListResponse",
    "EvalResult",
    "EvalRunRequest",
    "EvalRunResponse",
    "EvalSuite",
    "JudgmentListResponse",
    "JudgmentManifest",
    "PromoteJudgmentBody",
    "RenderJudgmentPromptResponse",
    "RerunBody",
    "SuiteListResponse",
    "SuiteSaveResponse",
    "SynthesisPreview",
    "SynthesisPreviewBody",
    "SynthesizeBody",
    "SynthesizeDatasetResponse",
    "ValidateJudgmentResponse",
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
    result: dict[str, Any] = {"type": "array", "items": schema}
    if min_items is not None:
        result["minItems"] = min_items
    if max_items is not None:
        result["maxItems"] = max_items
    return result


def mapping(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": schema}


def add_definition(name: str, rust_type: str, schema: dict[str, Any], description: str) -> None:
    DEFS[name] = {
        **schema,
        "description": description,
        "x-kiln-rust-type": rust_type,
        **STATUS,
    }


def add_enum(name: str, rust_type: str, values: list[str], description: str) -> None:
    add_definition(name, rust_type, {"type": "string", "enum": values}, description)


def object_schema(
    fields: dict[str, dict[str, Any]],
    *,
    optional: tuple[str, ...] = (),
    open_input: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    optional_set = set(optional)
    unknown = optional_set - set(fields)
    if unknown:
        raise ValueError(f"optional fields are not declared: {sorted(unknown)}")
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
    return schema


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
    add_definition(
        name,
        rust_type,
        object_schema(fields, optional=optional, open_input=open_input, extra=extra),
        description,
    )


def tagged_variant(
    kind: str,
    fields: dict[str, dict[str, Any]] | None = None,
    *,
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    return object_schema(
        {"kind": {"const": kind}, **(fields or {})},
        optional=optional,
        open_input=True,
    )


def build_primitives() -> None:
    add_definition("AnyJson", "serde_json::Value", {}, "Any JSON value carried without field-level interpretation.")
    add_definition("Boolean", "bool", {"type": "boolean"}, "A serialized Rust boolean.")
    add_definition("String", "String", {"type": "string"}, "A serialized UTF-8 Rust string.")
    add_definition("NonEmptyString", "String", {"type": "string", "minLength": 1}, "A non-empty UTF-8 string.")
    add_definition("NonNegativeInteger", "u64 | u32 | usize", {"type": "integer", "minimum": 0}, "A non-negative integer.")
    add_definition("PositiveInteger", "u64 | u32 | usize", {"type": "integer", "minimum": 1}, "A positive integer.")
    add_definition("FiniteNumber", "f32 | f64", {"type": "number"}, "A finite JSON number.")
    add_definition("UnitInterval", "f32", {"type": "number", "minimum": 0, "maximum": 1}, "A score or rate in the closed interval [0, 1].")
    add_definition(
        "DecimalU64",
        "u64",
        {"type": "string", "pattern": "^(0|[1-9][0-9]*)$"},
        "An exact unsigned 64-bit value serialized as a base-10 string for JavaScript safety.",
    )
    add_definition("Rfc3339Timestamp", "String", {"type": "string", "format": "date-time"}, "An RFC 3339 timestamp.")
    add_definition(
        "EvalResourceName",
        "String",
        {"type": "string", "minLength": 1, "pattern": r"^(?![\s\S]*(?:/|\\|\.\.))(?=[\s\S]*\S)[\s\S]+$"},
        "A non-blank suite, dataset, or judgment name without path separators or '..'.",
    )


def build_enums() -> None:
    add_enum("EvalJobState", "kiln_eval::EvalJobState", ["queued", "running", "completed", "failed", "cancelled"], "Evaluation job lifecycle state.")
    add_enum("EvalOutcomeKind", "kiln_eval::EvalOutcomeKind", ["pass", "fail", "invalid", "error"], "Per-completion scoring outcome.")
    add_enum("EvalSubmissionKind", "EvalSubmissionKind", ["on_demand", "post_training", "compare"], "Admission path that created an eval job.")
    add_enum("DatasetFormat", "DatasetFormat", ["sft_chat", "grpo_groups", "raw"], "Canonical persisted dataset format.")
    add_enum("DatasetUploadFormat", "DatasetUploadMultipart::format", ["sft_chat", "sft", "grpo_groups", "grpo", "raw"], "Accepted multipart dataset format including request aliases.")
    add_enum("JudgmentWinner", "JudgmentWinner", ["a", "b", "tie", "skip"], "Human pairwise preference label.")
    add_enum("SynthesisStrategy", "SynthesisStrategy", ["final_assistant", "first_assistant_turn", "every_assistant_turn", "tool_call_predict", "tool_response_followup", "end_of_trajectory_answer"], "How a conversation is decomposed into eval examples.")
    add_enum("ContainsMode", "ContainsMode", ["any", "all", "none"], "How phrase containment matches combine.")


def build_scorers() -> None:
    add_definition(
        "CodeStyle",
        "kiln_eval::scorers::CodeStyle",
        {
            "oneOf": [
                tagged_variant("any_block"),
                tagged_variant("exact_block", {"strip_comments": ref("Boolean")}, optional=("strip_comments",)),
                tagged_variant("token_similarity", {"min_jaccard": ref("FiniteNumber")}, optional=("min_jaccard",)),
                tagged_variant("line_coverage", {"min_coverage": ref("FiniteNumber")}, optional=("min_coverage",)),
            ]
        },
        "Code comparison policy. Variant-specific defaulted fields may be omitted on input.",
    )
    add_definition(
        "NameMatch",
        "kiln_eval::scorers::NameMatch",
        {
            "oneOf": [
                tagged_variant("exact"),
                tagged_variant("case_insensitive"),
                tagged_variant("one_of", {"allowed": array(ref("String"))}),
            ]
        },
        "Tool-name matching policy.",
    )
    add_object(
        "ToolCallWeights",
        "kiln_eval::scorers::ToolCallWeights",
        {"name": ref("FiniteNumber"), "structure": ref("FiniteNumber"), "content": ref("FiniteNumber")},
        "Relative tool-call name, structure, and content weights.",
        open_input=True,
    )
    add_definition(
        "ArgsScoring",
        "kiln_eval::scorers::ArgsScoring",
        {
            "oneOf": [
                tagged_variant("keys_only"),
                tagged_variant("structural"),
                tagged_variant("auto"),
                tagged_variant(
                    "per_key",
                    {"scorers": mapping(ref("Scorer")), "extra_key_penalty": ref("FiniteNumber")},
                    optional=("extra_key_penalty",),
                ),
            ]
        },
        "Tool-argument scoring policy, including recursive per-key scorers.",
    )
    scorer_variants = [
        tagged_variant("exact_match", {"case_sensitive": ref("Boolean"), "strip_whitespace": ref("Boolean")}, optional=("case_sensitive", "strip_whitespace")),
        tagged_variant("contains", {"phrases": array(ref("String")), "mode": ref("ContainsMode"), "case_sensitive": ref("Boolean")}, optional=("mode", "case_sensitive")),
        tagged_variant("regex", {"pattern": ref("String"), "capture_group": nullable(ref("NonNegativeInteger")), "case_sensitive": ref("Boolean")}, optional=("capture_group", "case_sensitive")),
        tagged_variant("json_validity", {"require_object": ref("Boolean"), "required_paths": array(ref("String"))}, optional=("require_object", "required_paths")),
        tagged_variant("multiple_choice", {"choices": array(ref("String"))}, optional=("choices",)),
        tagged_variant("numeric_tolerance", {"atol": ref("FiniteNumber"), "rtol": ref("FiniteNumber"), "integer_only": ref("Boolean")}, optional=("atol", "rtol", "integer_only")),
        tagged_variant("llm_judge", {"judge_adapter": nullable(ref("String")), "template": ref("String"), "score_regex": ref("String")}, optional=("judge_adapter", "template", "score_regex")),
        tagged_variant("tool_call", {"name_match": ref("NameMatch"), "args": ref("ArgsScoring"), "weights": ref("ToolCallWeights"), "require_xml_format": ref("Boolean")}, optional=("name_match", "args", "weights", "require_xml_format")),
        tagged_variant("code", {"language": nullable(ref("String")), "style": ref("CodeStyle")}, optional=("language", "style")),
        tagged_variant("python_doctest", {"timeout_seconds": ref("FiniteNumber"), "python_bin": nullable(ref("String"))}, optional=("timeout_seconds", "python_bin")),
        tagged_variant("all", {"scorers": array(ref("Scorer"))}),
        tagged_variant("any", {"scorers": array(ref("Scorer"))}),
    ]
    add_definition(
        "Scorer",
        "kiln_eval::scorers::Scorer",
        {"oneOf": scorer_variants},
        "Complete built-in scorer union. Unknown fields are accepted and ignored within each input variant.",
    )


def build_suite_types() -> None:
    aggregation_k = {"type": "integer", "minimum": 1, "maximum": 128}
    add_definition(
        "EvalAggregation",
        "kiln_eval::EvalAggregation",
        {
            "oneOf": [
                object_schema({"kind": {"const": "single"}}),
                object_schema({"kind": {"const": "mean_at_k"}, "k": aggregation_k}),
                object_schema({"kind": {"const": "pass_at_k"}, "k": aggregation_k}),
                object_schema({"kind": {"const": "majority_at_k"}, "k": aggregation_k}),
            ],
            "x-kiln-semantic-constraints": ["majority_at_k requires odd k"],
        },
        "Explicit reduction from k raw completions to one independent example statistic.",
    )
    add_object(
        "EvalChatMessage",
        "kiln_core::tokenizer::ChatMessage",
        {
            "role": ref("String"),
            "content": {"oneOf": [ref("String"), {"type": "null"}, array(ref("AnyJson"))]},
            "tool_calls": nullable(array(ref("AnyJson"))),
            "name": nullable(ref("String")),
            "tool_call_id": nullable(ref("String")),
        },
        "Canonical chat message. Non-string input content is normalized by the runtime's compatibility deserializer.",
        optional=("content", "tool_calls", "name", "tool_call_id"),
        open_input=True,
    )
    add_object(
        "EvalGenerationParams",
        "kiln_eval::EvalGenerationParams",
        {
            "temperature": ref("FiniteNumber"),
            "top_p": ref("FiniteNumber"),
            "top_k": ref("NonNegativeInteger"),
            "max_tokens": ref("NonNegativeInteger"),
            "n": {"type": "integer", "minimum": 1, "maximum": 128},
            "stop": array(ref("String")),
            "seed": nullable(ref("NonNegativeInteger")),
            "thinking_budget_tokens": nullable(ref("NonNegativeInteger")),
            "thinking_budget_ms": nullable(ref("NonNegativeInteger")),
            "chat_template_kwargs": mapping(ref("AnyJson")),
        },
        "Eval decode parameters. Omitted thinking budgets inherit; explicit null is unlimited; finite values include zero.",
        optional=("temperature", "top_p", "top_k", "max_tokens", "n", "stop", "seed", "thinking_budget_tokens", "thinking_budget_ms", "chat_template_kwargs"),
        open_input=True,
        extra={"x-kiln-thinking-budget-contract": THINKING_SCHEMA},
    )
    add_object(
        "EvalExample",
        "kiln_eval::EvalExample",
        {
            "id": ref("String"),
            "messages": array(ref("EvalChatMessage"), min_items=1),
            "target": ref("String"),
            "aliases": array(ref("String")),
            "tags": array(ref("String")),
            "metadata": ref("AnyJson"),
            "scorer": ref("Scorer"),
            "generation": ref("EvalGenerationParams"),
            "weight": {"type": "number", "minimum": 0},
            "tools": array(ref("AnyJson")),
        },
        "One prompt/target pair with optional scoring and generation overrides.",
        optional=("id", "target", "aliases", "tags", "metadata", "scorer", "generation", "weight", "tools"),
        open_input=True,
    )
    add_object(
        "EvalSuite",
        "EvalSuite",
        {
            "name": ref("EvalResourceName"),
            "description": nullable(ref("String")),
            "default_scorer": ref("Scorer"),
            "generation": ref("EvalGenerationParams"),
            "aggregation": ref("EvalAggregation"),
            "system_prompt": nullable(ref("String")),
            "examples": array(ref("EvalExample"), min_items=1),
            "schema_version": {"type": "integer", "enum": [1, 2]},
            "tools": array(ref("AnyJson")),
        },
        "A validated evaluation suite. Defaulted fields may be omitted on input and are materialized when serialized.",
        optional=("description", "generation", "aggregation", "system_prompt", "schema_version", "tools"),
        open_input=True,
        extra={"x-kiln-semantic-constraints": ["resolved example IDs are unique", "weights are finite and non-negative", "tool entries have a non-empty function.name or name", "every effective generation.n equals aggregation.k", "schema version 1 permits only single aggregation and n=1"]},
    )
    add_object(
        "EvalSuiteSummary",
        "kiln_eval::EvalSuiteSummary",
        {"name": ref("EvalResourceName"), "description": nullable(ref("String")), "num_examples": ref("NonNegativeInteger"), "completions_per_example": {"type": "integer", "minimum": 1, "maximum": 128}, "aggregation": ref("EvalAggregation"), "schema_version": {"type": "integer", "enum": [1, 2]}, "default_scorer_kind": ref("NonEmptyString"), "tags": mapping(ref("NonNegativeInteger"))},
        "List-view projection of a registered suite.",
    )
    add_object("SuiteListResponse", "SuiteListResponse", {"suites": array(ref("EvalSuiteSummary"))}, "All registered evaluation suites.")
    add_object(
        "SuiteSaveResponse",
        "SuiteSaveResponse",
        {"name": ref("EvalResourceName"), "path": ref("NonEmptyString"), "status": {"enum": ["created", "overwritten"]}},
        "Suite persistence confirmation.",
    )
    add_object("DeleteSuiteResponse", "DeleteSuiteResponse", {"status": {"const": "deleted"}, "name": ref("EvalResourceName")}, "Confirmation that a suite was deleted.")
    add_object(
        "EvalCompareSpec",
        "EvalCompareSpec",
        {"suite": ref("EvalResourceName"), "adapters": array(ref("String"), min_items=1, max_items=8), "seed": ref("NonNegativeInteger"), "generation": ref("EvalGenerationParams")},
        "Run one registered suite against one to eight ordered adapters. Empty adapter names select the base model.",
        optional=("seed", "generation"),
        open_input=True,
    )
    add_object(
        "EvalRunRequest",
        "EvalRunRequest",
        {"suite": ref("EvalResourceName"), "inline_suite": ref("EvalSuite"), "adapter": nullable(ref("String")), "seed": ref("NonNegativeInteger"), "generation": ref("EvalGenerationParams")},
        "Submit exactly one registered or inline suite for evaluation.",
        optional=("suite", "inline_suite", "adapter", "seed", "generation"),
        open_input=True,
        extra={"oneOf": [{"required": ["suite"], "not": {"required": ["inline_suite"]}}, {"required": ["inline_suite"], "not": {"required": ["suite"]}}]},
    )
    add_object(
        "EvalRunResponse",
        "EvalRunResponse",
        {"job_id": ref("NonEmptyString"), "state": {"const": "queued"}, "effective_seed": ref("DecimalU64"), "message": ref("NonEmptyString")},
        "Immutable admission receipt for a queued eval job.",
    )
    add_object(
        "RerunBody",
        "RerunBody",
        {"adapter": nullable(ref("String")), "outcome_kinds": array(ref("EvalOutcomeKind")), "include_pass": ref("Boolean"), "seed": ref("NonNegativeInteger")},
        "Select outcomes and adapter identity for a failure-focused re-run.",
        optional=("adapter", "outcome_kinds", "include_pass", "seed"),
        open_input=True,
    )


def build_result_types() -> None:
    add_object(
        "AggregatedExampleOutcome",
        "kiln_eval::AggregatedExampleOutcome",
        {
            "example_id": ref("NonEmptyString"), "kind": ref("EvalOutcomeKind"), "score": ref("UnitInterval"),
            "completion_indices": array(ref("NonNegativeInteger"), min_items=1, max_items=128),
            "representative_completion_index": ref("NonNegativeInteger"),
            "num_pass": ref("NonNegativeInteger"), "num_fail": ref("NonNegativeInteger"),
            "num_invalid": ref("NonNegativeInteger"), "num_error": ref("NonNegativeInteger"),
            "tags": array(ref("String")), "metadata": ref("AnyJson"),
        },
        "One independent example statistic reduced from its complete raw completion group.",
        optional=("tags", "metadata"),
        extra={"x-kiln-semantic-constraints": ["completion_indices are exactly 0..k", "raw kind counts sum to k", "representative_completion_index is a member of completion_indices"]},
    )
    add_object(
        "ExampleOutcome",
        "kiln_eval::ExampleOutcome",
        {
            "example_id": ref("NonEmptyString"), "completion_index": ref("NonNegativeInteger"),
            "generation_seed": ref("DecimalU64"), "completion_text": ref("String"), "raw_completion_text": ref("String"),
            "thinking_budget": external_ref(THINKING_SCHEMA, "record"), "kind": ref("EvalOutcomeKind"), "score": ref("UnitInterval"),
            "detail": ref("String"), "prompt_tokens": ref("NonNegativeInteger"), "completion_tokens": ref("NonNegativeInteger"),
            "latency_ms": ref("FiniteNumber"), "tags": array(ref("String")), "metadata": ref("AnyJson"),
            "reasoning_text": ref("String"), "unclosed_thinking": ref("Boolean"),
        },
        "One scored completion with optional raw reasoning, latency, seed, and thinking-budget evidence.",
        optional=("generation_seed", "raw_completion_text", "thinking_budget", "detail", "prompt_tokens", "completion_tokens", "latency_ms", "tags", "metadata", "reasoning_text", "unclosed_thinking"),
    )
    add_object("LatencyStats", "kiln_eval::LatencyStats", {"p50_ms": ref("FiniteNumber"), "p90_ms": ref("FiniteNumber"), "p99_ms": ref("FiniteNumber"), "mean_ms": ref("FiniteNumber"), "max_ms": ref("FiniteNumber")}, "Latency distribution summary in milliseconds.")
    add_object("PassRateConfidenceInterval", "kiln_eval::PassRateConfidenceInterval", {"confidence_level": ref("UnitInterval"), "lower": ref("UnitInterval"), "upper": ref("UnitInterval")}, "Wilson confidence interval for a pass rate.")
    add_object("ScorerBreakdown", "kiln_eval::ScorerBreakdown", {"scorer_kind": ref("NonEmptyString"), "num_examples": ref("NonNegativeInteger"), "mean_score": ref("UnitInterval"), "pass_rate": ref("UnitInterval")}, "Aggregate metrics for one scorer kind.")
    add_object("ToolBreakdown", "kiln_eval::ToolBreakdown", {"num_examples": ref("NonNegativeInteger"), "num_pass": ref("NonNegativeInteger"), "pass_rate": ref("UnitInterval"), "confidence_interval": ref("PassRateConfidenceInterval")}, "Per-tool pass counts and interval.")
    add_object("TagBreakdown", "kiln_eval::TagBreakdown", {"num_examples": ref("NonNegativeInteger"), "num_pass": ref("NonNegativeInteger"), "pass_rate": ref("UnitInterval"), "confidence_interval": ref("PassRateConfidenceInterval")}, "Per-tag pass counts and interval.")
    add_object("ReasoningLengthStats", "kiln_eval::ReasoningLengthStats", {"num_with_thinking": ref("NonNegativeInteger"), "mean_chars": ref("FiniteNumber"), "p50_chars": ref("NonNegativeInteger"), "p90_chars": ref("NonNegativeInteger"), "max_chars": ref("NonNegativeInteger")}, "Reasoning-block character-length summary.")
    add_object(
        "AggregateMetrics",
        "kiln_eval::AggregateMetrics",
        {
            "num_examples": ref("NonNegativeInteger"), "num_completions": ref("NonNegativeInteger"), "num_pass": ref("NonNegativeInteger"), "num_fail": ref("NonNegativeInteger"),
            "num_invalid": ref("NonNegativeInteger"), "num_error": ref("NonNegativeInteger"), "accuracy": ref("UnitInterval"),
            "accuracy_confidence_interval": ref("PassRateConfidenceInterval"), "mean_score": ref("UnitInterval"),
            "weighted_mean_score": ref("UnitInterval"), "latency": ref("LatencyStats"), "total_prompt_tokens": ref("NonNegativeInteger"),
            "total_completion_tokens": ref("NonNegativeInteger"), "elapsed_secs": ref("FiniteNumber"), "pass_rate_by_tag": mapping(ref("UnitInterval")),
            "tag_breakdown": mapping(ref("TagBreakdown")), "by_scorer": array(ref("ScorerBreakdown")),
            "pass_rate_by_tool": mapping(ref("ToolBreakdown")), "reasoning_length": ref("ReasoningLengthStats"),
            "num_unclosed_thinking": ref("NonNegativeInteger"), "num_non_xml_tool_calls": ref("NonNegativeInteger"),
            "confusion_by_tool": mapping(mapping(ref("NonNegativeInteger"))), "num_schema_missing_required": ref("NonNegativeInteger"),
            "num_schema_extra_unknown": ref("NonNegativeInteger"),
        },
        "Complete aggregate metrics for one suite/adapter run.",
        optional=("tag_breakdown", "pass_rate_by_tool", "confusion_by_tool"),
    )
    add_object("EvalProgress", "kiln_eval::EvalProgress", {"examples_completed": ref("NonNegativeInteger"), "examples_total": ref("NonNegativeInteger"), "running_accuracy": ref("UnitInterval"), "running_mean_score": ref("UnitInterval")}, "Live progress for the active adapter run.")
    add_object(
        "SuiteResult", "kiln_eval::SuiteResult",
        {"suite_name": ref("EvalResourceName"), "adapter": nullable(ref("String")), "aggregation": ref("EvalAggregation"), "metrics": ref("AggregateMetrics"), "aggregated_outcomes": array(ref("AggregatedExampleOutcome")), "outcomes": array(ref("ExampleOutcome")), "started_at": ref("Rfc3339Timestamp"), "finished_at": ref("Rfc3339Timestamp"), "suite_hash": ref("NonEmptyString"), "effective_generation_hash": ref("NonEmptyString")},
        "Complete result for one suite and adapter, retaining both independent example reductions and raw completions.", optional=("effective_generation_hash",),
    )
    add_object(
        "EvalResult", "EvalResult",
        {"schema_version": {"const": 2}, "job_id": ref("NonEmptyString"), "state": ref("EvalJobState"), "base_weight_shard_manifest": external_ref(OBSERVABILITY_SCHEMA, "BaseWeightShardManifest"), "execution_provenance": external_ref(OBSERVABILITY_SCHEMA, "ExecutionProvenanceV1"), "effective_seed": ref("DecimalU64"), "seed_derivation": ref("String"), "runs": array(ref("SuiteResult")), "progress": ref("EvalProgress"), "error": ref("String")},
        "Top-level retained result for a single- or multi-adapter eval job.",
        optional=("base_weight_shard_manifest", "execution_provenance", "effective_seed", "seed_derivation", "progress", "error"),
    )
    add_object(
        "PostEvalGate", "PostEvalGate",
        {"min_accuracy": ref("FiniteNumber"), "relative_recovery": ref("FiniteNumber"), "absolute_gain": ref("FiniteNumber"), "adapter_name": ref("String"), "training_job_id": ref("NonEmptyString"), "auto_load_on_pass": ref("Boolean")},
        "Post-training promotion thresholds retained on an eval job.", optional=("relative_recovery", "absolute_gain"),
    )
    add_object(
        "EvalJobInfo", "EvalJobInfo",
        {"schema_version": {"const": 2}, "job_id": ref("NonEmptyString"), "suite_name": ref("EvalResourceName"), "adapters": array(nullable(ref("String"))), "submission_kind": ref("EvalSubmissionKind"), "base_weight_shard_manifest": external_ref(OBSERVABILITY_SCHEMA, "BaseWeightShardManifest"), "execution_provenance": external_ref(OBSERVABILITY_SCHEMA, "ExecutionProvenanceV1"), "effective_seed": ref("DecimalU64"), "state": ref("EvalJobState"), "progress": ref("EvalProgress"), "finished_runs": array(ref("SuiteResult")), "headline_accuracy": nullable(ref("FiniteNumber")), "error": nullable(ref("String")), "source_training_job_id": nullable(ref("String")), "submitted_at_iso": ref("Rfc3339Timestamp"), "started_at_iso": nullable(ref("Rfc3339Timestamp")), "finished_at_iso": nullable(ref("Rfc3339Timestamp")), "post_eval_gate": ref("PostEvalGate")},
        "Tracked eval-job list record; runtime-only Instants and cancellation handles are never serialized.",
        optional=("base_weight_shard_manifest", "execution_provenance", "effective_seed", "post_eval_gate"),
    )
    add_object("EvalJobListResponse", "EvalJobListResponse", {"jobs": array(ref("EvalJobInfo"))}, "All retained eval jobs in descending submission order.")
    cancel_variants = [
        object_schema({"status": {"const": "cancelled"}, "job_id": ref("NonEmptyString"), "was_in_queue": ref("Boolean")}),
        object_schema({"status": {"const": "cancelling"}, "job_id": ref("NonEmptyString"), "note": {"const": "running job will exit at the next example boundary"}}),
        object_schema({"status": {"const": "deleted"}, "job_id": ref("NonEmptyString"), "removed_archive_file": ref("Boolean")}),
    ]
    add_definition("CancelEvalJobResponse", "CancelEvalJobResponse", {"oneOf": cancel_variants}, "Queued cancellation, cooperative running cancellation, or terminal-record deletion result.")


def synthesis_fields() -> dict[str, dict[str, Any]]:
    return {
        "suite_name": ref("EvalResourceName"), "description": nullable(ref("String")), "strategy": ref("SynthesisStrategy"),
        "scorer": ref("ScorerChoice"), "generation": ref("EvalGenerationParams"), "aggregation": ref("EvalAggregation"), "sampling": ref("Sampling"),
        "system_prompt": nullable(ref("String")), "strip_system_prompt": ref("Boolean"), "suite_tools": array(ref("AnyJson")),
    }


def build_dataset_and_synthesis_types() -> None:
    add_object("DatasetStats", "DatasetStats", {"num_assistant_turns": ref("NonNegativeInteger"), "num_tool_messages": ref("NonNegativeInteger"), "num_with_tool_calls": ref("NonNegativeInteger"), "max_messages_per_conv": ref("NonNegativeInteger"), "max_content_chars": ref("NonNegativeInteger"), "avg_messages_per_conv": ref("FiniteNumber"), "sample_role_patterns": array(ref("String"))}, "Bounded structural statistics computed from an uploaded dataset.")
    add_object("DatasetManifest", "DatasetManifest", {"name": ref("EvalResourceName"), "format": ref("DatasetFormat"), "description": nullable(ref("String")), "num_rows": ref("NonNegativeInteger"), "size_bytes": ref("NonNegativeInteger"), "created_at": ref("Rfc3339Timestamp"), "updated_at": ref("Rfc3339Timestamp"), "stats": ref("DatasetStats")}, "Persisted dataset identity and structural summary.")
    add_object("DatasetListResponse", "DatasetListResponse", {"datasets": array(ref("DatasetManifest"))}, "All uploaded eval/training datasets.")
    add_object("DatasetUploadMultipart", "DatasetUploadMultipart", {"name": ref("EvalResourceName"), "format": ref("DatasetUploadFormat"), "description": ref("String"), "file": {"type": "string", "format": "binary", "contentMediaType": "application/jsonl"}}, "Multipart dataset upload. Unknown parts are drained and ignored.", optional=("format", "description"), open_input=True, extra={"x-kiln-default-format": "sft_chat"})
    add_object("DeleteDatasetResponse", "DeleteDatasetResponse", {"status": {"const": "deleted"}, "name": ref("EvalResourceName")}, "Confirmation that a dataset was deleted.")
    add_object("Sampling", "kiln_eval::synthesis::Sampling", {"max_examples": nullable(ref("NonNegativeInteger")), "max_prompt_chars": ref("NonNegativeInteger"), "max_target_chars": ref("NonNegativeInteger"), "seed": nullable(ref("NonNegativeInteger")), "dedupe": ref("Boolean")}, "Synthesis sampling and filtering controls.", optional=("max_examples", "max_prompt_chars", "max_target_chars", "seed", "dedupe"), open_input=True)
    add_definition(
        "ScorerChoice", "kiln_eval::synthesis::ScorerChoice",
        {"oneOf": [tagged_variant("auto_detect"), tagged_variant("fixed", {"scorer": ref("Scorer")}), tagged_variant("judge", {"judge_adapter": nullable(ref("String"))}, optional=("judge_adapter",))]},
        "Synthesis scorer policy. Fixed scorers use a nested scorer field so both tagged unions remain unambiguous.",
    )
    preview_optional = tuple(field for field in synthesis_fields() if field != "suite_name") + ("head_n",)
    add_object("SynthesisPreviewBody", "SynthesisPreviewBody", {**synthesis_fields(), "head_n": ref("NonNegativeInteger")}, "Preview a synthesized suite without persisting it.", optional=preview_optional, open_input=True)
    synth_optional = tuple(field for field in synthesis_fields() if field != "suite_name") + ("force", "run_against")
    add_object("SynthesizeBody", "SynthesizeBody", {**synthesis_fields(), "force": ref("Boolean"), "run_against": nullable(array(ref("String")))}, "Persist a synthesized suite and optionally queue it against adapters.", optional=synth_optional, open_input=True)
    add_object("SynthesisStats", "kiln_eval::synthesis::SynthesisStats", {"trajectories_seen": ref("NonNegativeInteger"), "trajectories_used": ref("NonNegativeInteger"), "examples_generated": ref("NonNegativeInteger"), "skipped_no_target": ref("NonNegativeInteger"), "skipped_prompt_too_long": ref("NonNegativeInteger"), "skipped_target_too_long": ref("NonNegativeInteger"), "skipped_duplicate": ref("NonNegativeInteger"), "skipped_strategy_match": ref("NonNegativeInteger"), "sample_kept": ref("NonNegativeInteger"), "effective_seed": ref("DecimalU64"), "auto_scorer_histogram": mapping(ref("NonNegativeInteger"))}, "Complete synthesis filtering and deterministic seed statistics. The u64 seed is emitted as an exact decimal string.")
    add_object("SynthesisPreview", "SynthesisPreview", {"examples": array(ref("EvalExample")), "stats": ref("SynthesisStats"), "suite_name": ref("EvalResourceName"), "default_scorer_kind": ref("NonEmptyString"), "aggregation": ref("EvalAggregation"), "completions_per_example": {"type": "integer", "minimum": 1, "maximum": 128}}, "Non-persisted synthesis preview including the completion reduction that would be persisted.")
    add_object("SynthesizeDatasetResponse", "SynthesizeDatasetResponse", {"suite": ref("EvalSuiteSummary"), "stats": ref("SynthesisStats"), "queued_eval_job_ids": array(ref("String"))}, "Persisted suite summary plus any auto-queued eval job IDs.")


def build_judgment_types() -> None:
    manifest_fields = {"name": ref("EvalResourceName"), "description": nullable(ref("String")), "num_rows": ref("NonNegativeInteger"), "created_at": ref("Rfc3339Timestamp"), "updated_at": ref("Rfc3339Timestamp"), "winner_histogram": mapping(ref("NonNegativeInteger")), "last_compiled_split": ref("NonNegativeInteger"), "last_compiled_at": ref("Rfc3339Timestamp")}
    manifest_optional = ("last_compiled_split", "last_compiled_at")
    add_object("JudgmentManifest", "JudgmentManifest", manifest_fields, "Persisted pairwise-judgment dataset identity and compile split.", optional=manifest_optional)
    add_object("JudgmentListResponse", "JudgmentListResponse", {"judgments": array(ref("JudgmentManifest"))}, "All retained pairwise-judgment datasets.")
    add_object("CreateJudgmentBody", "CreateJudgmentBody", {"name": ref("EvalResourceName"), "description": nullable(ref("String"))}, "Create an append-only judgment dataset.", optional=("description",), open_input=True)
    append_fields = {"id": nullable(ref("String")), "prompt": array(ref("EvalChatMessage")), "adapter_a": nullable(ref("String")), "adapter_b": nullable(ref("String")), "response_a": ref("String"), "response_b": ref("String"), "winner": ref("JudgmentWinner"), "note": nullable(ref("String")), "tags": array(ref("String"))}
    add_object("AppendJudgmentBody", "AppendJudgmentBody", append_fields, "Append one exact A/B preference row.", optional=("id", "adapter_a", "adapter_b", "note", "tags"), open_input=True)
    add_object("AppendJudgmentResponse", "AppendJudgmentResponse", {"judgment_id": ref("NonEmptyString"), **manifest_fields}, "Assigned row ID flattened with the updated judgment manifest.", optional=manifest_optional)
    add_object("DeleteJudgmentResponse", "DeleteJudgmentResponse", {"status": {"const": "deleted"}, "name": ref("EvalResourceName")}, "Confirmation that a judgment dataset was deleted.")
    add_object("CompileJudgmentBody", "CompileJudgmentBody", {"output_dataset": ref("EvalResourceName"), "include_skips": ref("Boolean"), "holdout_n": nullable(ref("NonNegativeInteger"))}, "Compile judgment rows to a swap-augmented SFT dataset with a holdout.", optional=("include_skips", "holdout_n"), open_input=True)
    add_object("CompileJudgmentResponse", "CompileJudgmentResponse", {"status": {"const": "compiled"}, "rows": ref("PositiveInteger"), "holdout_n": ref("NonNegativeInteger"), "train_validation_split": ref("PositiveInteger"), "dataset": ref("DatasetManifest"), "warnings": array(ref("String"))}, "Compiled SFT dataset and exact train/validation boundary.")
    add_object("PromoteJudgmentBody", "PromoteJudgmentBody", {"adapter": ref("String"), "holdout_n": ref("NonNegativeInteger")}, "Queue held-out validation for a trained judge adapter.", optional=("holdout_n",), open_input=True)
    add_object("ValidateJudgmentResponse", "ValidateJudgmentResponse", {"status": {"const": "queued"}, "eval_job_id": ref("NonEmptyString"), "effective_seed": ref("DecimalU64"), "validation_suite": ref("EvalResourceName"), "warnings": array(ref("String"))}, "Held-out judge-validation admission receipt.")
    add_object("RenderJudgmentPromptResponse", "RenderJudgmentPromptResponse", {"prompt": ref("String")}, "Exact pairwise prompt used by both judge training and inference.")


def metrics_example() -> dict[str, Any]:
    interval = {"confidence_level": 0.95, "lower": 0.2, "upper": 1.0}
    return {
        "num_examples": 1, "num_completions": 1, "num_pass": 1, "num_fail": 0, "num_invalid": 0, "num_error": 0,
        "accuracy": 1.0, "accuracy_confidence_interval": interval, "mean_score": 1.0,
        "weighted_mean_score": 1.0, "latency": {"p50_ms": 10.0, "p90_ms": 10.0, "p99_ms": 10.0, "mean_ms": 10.0, "max_ms": 10.0},
        "total_prompt_tokens": 8, "total_completion_tokens": 1, "elapsed_secs": 0.01,
        "pass_rate_by_tag": {"math": 1.0}, "tag_breakdown": {"math": {"num_examples": 1, "num_pass": 1, "pass_rate": 1.0, "confidence_interval": interval}},
        "by_scorer": [{"scorer_kind": "exact_match", "num_examples": 1, "mean_score": 1.0, "pass_rate": 1.0}],
        "reasoning_length": {"num_with_thinking": 0, "mean_chars": 0.0, "p50_chars": 0, "p90_chars": 0, "max_chars": 0},
        "num_unclosed_thinking": 0, "num_non_xml_tool_calls": 0, "num_schema_missing_required": 0, "num_schema_extra_unknown": 0,
    }


def suite_example() -> dict[str, Any]:
    return {
        "name": "math-smoke", "description": "One deterministic arithmetic case",
        "default_scorer": {"kind": "exact_match", "case_sensitive": False, "strip_whitespace": True},
        "generation": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_tokens": 32, "n": 1, "stop": [], "seed": None},
        "aggregation": {"kind": "single"},
        "examples": [{"id": "two-plus-two", "messages": [{"role": "user", "content": "2 + 2?"}], "target": "4", "weight": 1.0}],
        "schema_version": 2,
    }


def dataset_example() -> dict[str, Any]:
    return {"name": "math-sft", "format": "sft_chat", "description": "Arithmetic conversations", "num_rows": 1, "size_bytes": 64, "created_at": "2026-07-14T12:00:00Z", "updated_at": "2026-07-14T12:00:00Z", "stats": {"num_assistant_turns": 1, "num_tool_messages": 0, "num_with_tool_calls": 0, "max_messages_per_conv": 2, "max_content_chars": 8, "avg_messages_per_conv": 2.0, "sample_role_patterns": ["user>assistant"]}}


def judgment_example() -> dict[str, Any]:
    return {"name": "style-prefs", "description": "Concise-answer preferences", "num_rows": 1, "created_at": "2026-07-14T12:00:00Z", "updated_at": "2026-07-14T12:01:00Z", "winner_histogram": {"a": 1}}


def stats_example() -> dict[str, Any]:
    return {"trajectories_seen": 1, "trajectories_used": 1, "examples_generated": 1, "skipped_no_target": 0, "skipped_prompt_too_long": 0, "skipped_target_too_long": 0, "skipped_duplicate": 0, "skipped_strategy_match": 0, "sample_kept": 1, "effective_seed": "42", "auto_scorer_histogram": {"exact_match": 1}}


def build_examples() -> dict[str, list[Any]]:
    suite = suite_example()
    summary = {"name": "math-smoke", "description": "One deterministic arithmetic case", "num_examples": 1, "completions_per_example": 1, "aggregation": {"kind": "single"}, "schema_version": 2, "default_scorer_kind": "exact_match", "tags": {"math": 1}}
    dataset = dataset_example()
    judgment = judgment_example()
    outcome = {"example_id": "two-plus-two", "completion_index": 0, "generation_seed": "43", "completion_text": "4", "kind": "pass", "score": 1.0, "tags": ["math"]}
    aggregated = {"example_id": "two-plus-two", "kind": "pass", "score": 1.0, "completion_indices": [0], "representative_completion_index": 0, "num_pass": 1, "num_fail": 0, "num_invalid": 0, "num_error": 0, "tags": ["math"]}
    run = {"suite_name": "math-smoke", "adapter": None, "aggregation": {"kind": "single"}, "metrics": metrics_example(), "aggregated_outcomes": [aggregated], "outcomes": [outcome], "started_at": "2026-07-14T12:00:00Z", "finished_at": "2026-07-14T12:00:01Z", "suite_hash": "suite-sha256"}
    job = {"schema_version": 2, "job_id": "eval-1", "suite_name": "math-smoke", "adapters": [None], "submission_kind": "on_demand", "effective_seed": "42", "state": "completed", "progress": {"examples_completed": 1, "examples_total": 1, "running_accuracy": 1.0, "running_mean_score": 1.0}, "finished_runs": [run], "headline_accuracy": 1.0, "error": None, "source_training_job_id": None, "submitted_at_iso": "2026-07-14T12:00:00Z", "started_at_iso": "2026-07-14T12:00:00Z", "finished_at_iso": "2026-07-14T12:00:01Z"}
    append = {"prompt": [{"role": "user", "content": "Explain the answer."}], "adapter_a": None, "adapter_b": "concise-v1", "response_a": "A long answer", "response_b": "A concise answer", "winner": "b", "tags": ["style"]}
    synth_body = {"suite_name": "math-smoke", "strategy": "final_assistant", "scorer": {"kind": "fixed", "scorer": {"kind": "exact_match"}}, "aggregation": {"kind": "single"}, "sampling": {"max_examples": 100, "seed": 42}, "force": True, "run_against": [""]}
    examples: dict[str, list[Any]] = {
        "AppendJudgmentBody": [append],
        "AppendJudgmentResponse": [{"judgment_id": "judgment-1", **judgment}],
        "CancelEvalJobResponse": [{"status": "cancelled", "job_id": "eval-1", "was_in_queue": True}, {"status": "cancelling", "job_id": "eval-2", "note": "running job will exit at the next example boundary"}, {"status": "deleted", "job_id": "eval-3", "removed_archive_file": True}],
        "CompileJudgmentBody": [{"output_dataset": "style-prefs-sft", "include_skips": False, "holdout_n": 1}],
        "CompileJudgmentResponse": [{"status": "compiled", "rows": 2, "holdout_n": 1, "train_validation_split": 1, "dataset": dataset, "warnings": []}],
        "CreateJudgmentBody": [{"name": "style-prefs", "description": "Concise-answer preferences"}],
        "DatasetListResponse": [{"datasets": [dataset]}],
        "DatasetManifest": [dataset],
        "DatasetUploadMultipart": [{"name": "math-sft", "format": "sft_chat", "description": "Arithmetic", "file": "<binary JSONL body>"}],
        "DeleteDatasetResponse": [{"status": "deleted", "name": "math-sft"}],
        "DeleteJudgmentResponse": [{"status": "deleted", "name": "style-prefs"}],
        "DeleteSuiteResponse": [{"status": "deleted", "name": "math-smoke"}],
        "EvalCompareSpec": [{"suite": "math-smoke", "adapters": ["", "math-v1"], "seed": 42}],
        "EvalJobListResponse": [{"jobs": [job]}],
        "EvalResult": [{"schema_version": 2, "job_id": "eval-1", "state": "completed", "effective_seed": "42", "seed_derivation": "kiln.eval-seed.v1", "runs": [run]}],
        "EvalRunRequest": [{"suite": "math-smoke", "adapter": "math-v1", "seed": 42}],
        "EvalRunResponse": [{"job_id": "eval-1", "state": "queued", "effective_seed": "42", "message": "Queued eval against suite `math-smoke`"}],
        "EvalSuite": [suite],
        "JudgmentListResponse": [{"judgments": [judgment]}],
        "JudgmentManifest": [judgment],
        "PromoteJudgmentBody": [{"adapter": "judge-v1", "holdout_n": 20}],
        "RenderJudgmentPromptResponse": [{"prompt": "Compare the following two assistant replies..."}],
        "RerunBody": [{"adapter": "math-v2", "outcome_kinds": ["fail", "invalid", "error"], "include_pass": False, "seed": 42}],
        "SuiteListResponse": [{"suites": [summary]}],
        "SuiteSaveResponse": [{"name": "math-smoke", "path": "/srv/kiln/.eval/suites/math-smoke.json", "status": "created"}],
        "SynthesisPreview": [{"examples": suite["examples"], "stats": stats_example(), "suite_name": "math-smoke", "default_scorer_kind": "exact_match", "aggregation": {"kind": "single"}, "completions_per_example": 1}],
        "SynthesisPreviewBody": [{key: value for key, value in synth_body.items() if key not in {"force", "run_against"}} | {"head_n": 5}],
        "SynthesizeBody": [synth_body],
        "SynthesizeDatasetResponse": [{"suite": summary, "stats": stats_example(), "queued_eval_job_ids": ["eval-1"]}],
        "ValidateJudgmentResponse": [{"status": "queued", "eval_job_id": "eval-judge-1", "effective_seed": "42", "validation_suite": "judge-validate-style-prefs", "warnings": []}],
    }
    if set(examples) != set(ENTRYPOINTS):
        raise ValueError("eval example coverage does not match public entrypoints")
    return examples


def build_schema() -> dict[str, Any]:
    DEFS.clear()
    build_primitives()
    build_enums()
    build_scorers()
    build_suite_types()
    build_result_types()
    build_dataset_and_synthesis_types()
    build_judgment_types()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ericflo.github.io/kiln/contracts/kiln-evals-v1.schema.json",
        "title": "Kiln Eval, Dataset Synthesis, and Judgment API",
        "description": "Complete field-level wire contract for eval suites and jobs, uploaded datasets, deterministic suite synthesis, and the pairwise-judgment flywheel. Open input objects explicitly preserve serde's accepted-and-ignored unknown-field behavior; emitted objects are closed.",
        "x-kiln-field-schema-status": "complete",
        "x-kiln-entrypoints": list(ENTRYPOINTS),
        "x-kiln-external-contracts": [OBSERVABILITY_SCHEMA, THINKING_SCHEMA],
        "oneOf": [ref(name) for name in ENTRYPOINTS],
        "$defs": dict(sorted(DEFS.items())),
        "x-kiln-examples": build_examples(),
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
        print(f"Eval schema is current: {len(DEFS)} reachable definitions, {len(ENTRYPOINTS)} entrypoints")
        return 0
    OUTPUT.write_text(rendered)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(DEFS)} definitions, {len(ENTRYPOINTS)} entrypoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
