#!/usr/bin/env python3
"""Generate Kiln's training, agent, and product control-plane API schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "contracts" / "kiln-control-plane-v1.schema.json"
EVAL_SCHEMA = "kiln-evals-v1.schema.json"
INFERENCE_SCHEMA = "kiln-inference-v1.schema.json"
STATUS = {"x-kiln-field-schema-status": "complete"}
ENTRYPOINTS = (
    "AgentRunAbortResponse",
    "AgentRunEventsResponse",
    "AgentRunListResponse",
    "AgentRunQueuedResponse",
    "AgentRunRecord",
    "AgentRunsStatusResponse",
    "AgentTrace",
    "AgentTracesListResponse",
    "CancelTrainingJobResponse",
    "CapacityRequest",
    "CapacityResponse",
    "ClearCorrectionsResponse",
    "CompatibilityResponse",
    "CorrectionRow",
    "CorrectionRowInput",
    "CreateRunRequest",
    "DeleteCorrectionResponse",
    "DeleteTrainingJobResponse",
    "DiscoverRequest",
    "DiscoverResponse",
    "DistillMergeRequest",
    "DistillPumpRequest",
    "DistillRefreshRequest",
    "DistillSelfRequest",
    "FrontDoorRequest",
    "FrontDoorResponse",
    "GrpoRequest",
    "JudgeDistillRequest",
    "JudgeDistillResponse",
    "JudgeDriftCheckRequest",
    "LibraryListResponse",
    "ListResponse",
    "MarkTrainedRequest",
    "MarkTrainedResponse",
    "MessageRequest",
    "OpdRequest",
    "PublishPayload",
    "PublishToLibraryResponse",
    "QueueResponse",
    "RecipeRunRequest",
    "RecipeRunResponse",
    "RecipesListResponse",
    "SelfImproveRequest",
    "SelfImproveResponse",
    "SftRequest",
    "TerminalStatusResponse",
    "TierDefaultsListResponse",
    "TierDefaultsResponse",
    "TrainingJobDetail",
    "TrainingResponse",
    "TrainingStatus",
    "Vec_TrainingStatus",
)
DEFS: dict[str, dict[str, Any]] = {}


def ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


def external_ref(document: str, name: str) -> dict[str, str]:
    return {"$ref": f"{document}#/$defs/{name}"}


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def array(schema: dict[str, Any], *, min_items: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "array", "items": schema}
    if min_items is not None:
        result["minItems"] = min_items
    return result


def mapping(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": schema}


def active_optional(name: str) -> dict[str, Any]:
    """Match an Option field only when it is present and non-null."""
    return {"required": [name], "properties": {name: {"not": {"type": "null"}}}}


def object_schema(
    fields: dict[str, dict[str, Any]],
    *,
    optional: tuple[str, ...] = (),
    open_input: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unknown = set(optional) - set(fields)
    if unknown:
        raise ValueError(f"optional fields are not declared: {sorted(unknown)}")
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": open_input,
        "required": [field for field in fields if field not in set(optional)],
        "properties": fields,
    }
    if open_input:
        schema["x-kiln-unknown-field-policy"] = "accepted_and_ignored"
    if extra:
        schema.update(extra)
    return schema


def add_definition(name: str, rust_type: str, schema: dict[str, Any], description: str) -> None:
    DEFS[name] = {
        **schema,
        "description": description,
        "x-kiln-rust-type": rust_type,
        **STATUS,
    }


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


def add_enum(name: str, rust_type: str, values: list[str], description: str) -> None:
    add_definition(name, rust_type, {"type": "string", "enum": values}, description)


def tagged(
    kind: str,
    fields: dict[str, dict[str, Any]] | None = None,
    *,
    optional: tuple[str, ...] = (),
    open_input: bool = True,
) -> dict[str, Any]:
    return object_schema(
        {"kind": {"const": kind}, **(fields or {})},
        optional=optional,
        open_input=open_input,
    )


def build_primitives() -> None:
    add_definition("AnyJson", "serde_json::Value", {}, "Any JSON value.")
    add_definition("Boolean", "bool", {"type": "boolean"}, "A serialized Rust boolean.")
    add_definition("String", "String", {"type": "string"}, "A UTF-8 string.")
    add_definition("NonEmptyString", "String", {"type": "string", "minLength": 1}, "A non-empty UTF-8 string.")
    add_definition("NonNegativeInteger", "u64 | u32 | usize", {"type": "integer", "minimum": 0}, "A non-negative integer.")
    add_definition("PositiveInteger", "u64 | u32 | usize", {"type": "integer", "minimum": 1}, "A positive integer.")
    add_definition("FiniteNumber", "f32 | f64", {"type": "number"}, "A finite JSON number.")
    add_definition("UnitInterval", "f32 | f64", {"type": "number", "minimum": 0, "maximum": 1}, "A number in the closed interval [0, 1].")
    add_definition("DecimalU64", "u64", {"type": "string", "pattern": "^(0|[1-9][0-9]*)$"}, "An exact unsigned 64-bit value serialized as a decimal string.")
    add_definition("Rfc3339Timestamp", "String", {"type": "string", "format": "date-time"}, "An RFC 3339 timestamp.")
    add_definition("Sha256", "String", {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}, "A prefixed SHA-256 digest.")


def build_shared_training_types() -> None:
    add_object(
        "TrainingChatMessageInput",
        "kiln_core::tokenizer::ChatMessage",
        {
            "role": ref("String"),
            "content": {"oneOf": [ref("String"), {"type": "null"}, array(ref("AnyJson"))]},
            "tool_calls": nullable(array(ref("AnyJson"))),
            "name": nullable(ref("String")),
            "tool_call_id": nullable(ref("String")),
        },
        "Canonical training chat input. Content-part arrays are reduced to text by the compatibility decoder.",
        optional=("content", "tool_calls", "name", "tool_call_id"),
        open_input=True,
    )
    add_object(
        "TrainingChatMessageOutput",
        "kiln_core::tokenizer::ChatMessage",
        {
            "role": ref("String"),
            "content": ref("String"),
            "tool_calls": array(ref("AnyJson")),
            "name": ref("String"),
            "tool_call_id": ref("String"),
        },
        "Closed emitted training chat message.",
        optional=("tool_calls", "name", "tool_call_id"),
    )
    add_object("SftExample", "kiln_train::SftExample", {"messages": array(ref("TrainingChatMessageInput"))}, "One SFT conversation.", optional=("messages",), open_input=True)
    add_enum("SftTrainingProfile", "kiln_train::SftTrainingProfile", ["native_online_lora_v1"], "Versioned native SFT training shape.")
    add_enum("SftInvalidRowPolicy", "kiln_train::SftInvalidRowPolicy", ["fail", "skip"], "Malformed-row admission policy.")
    add_definition(
        "Optimizer",
        "kiln_train::Optimizer",
        {
            "oneOf": [
                tagged("sgd", open_input=False),
                tagged(
                    "adam_w",
                    {"beta1": ref("FiniteNumber"), "beta2": ref("FiniteNumber"), "eps": ref("FiniteNumber"), "weight_decay": ref("FiniteNumber")},
                    optional=("beta1", "beta2", "eps", "weight_decay"),
                    open_input=False,
                ),
                tagged(
                    "muon",
                    {"momentum": ref("FiniteNumber"), "nesterov": ref("Boolean"), "ns_iters": ref("NonNegativeInteger"), "weight_decay": ref("FiniteNumber")},
                    optional=("momentum", "nesterov", "ns_iters", "weight_decay"),
                    open_input=False,
                ),
            ]
        },
        "Fail-closed optimizer selection and variant-specific controls.",
    )
    sft_config_fields = {
        "training_profile": ref("SftTrainingProfile"),
        "invalid_row_policy": ref("SftInvalidRowPolicy"),
        "epochs": ref("PositiveInteger"),
        "learning_rate": nullable(ref("FiniteNumber")),
        "lora_rank": ref("PositiveInteger"),
        "lora_alpha": ref("FiniteNumber"),
        "train_mtp": nullable(ref("Boolean")),
        "base_adapter": nullable(ref("String")),
        "allow_adapter_shape_conversion": ref("Boolean"),
        "allow_high_lora_scale": ref("Boolean"),
        "output_name": nullable(ref("String")),
        "auto_load": ref("Boolean"),
        "checkpoint_interval": nullable(ref("PositiveInteger")),
        "resume_checkpoint": nullable(ref("String")),
        "grad_checkpoint_segments": nullable(ref("PositiveInteger")),
        "seed": nullable(ref("NonNegativeInteger")),
        "optimizer": ref("Optimizer"),
        "adapter_smoke_test": ref("Boolean"),
    }
    add_object("SftConfig", "kiln_train::SftConfig", sft_config_fields, "Complete native SFT configuration. Unknown fields fail admission.", optional=tuple(sft_config_fields), open_input=False)
    add_object(
        "PostEvalConfig",
        "kiln_eval::PostEvalConfig",
        {
            "suite": ref("NonEmptyString"),
            "generation": external_ref(EVAL_SCHEMA, "EvalGenerationParams"),
            "min_accuracy": ref("UnitInterval"),
            "include_baseline": ref("Boolean"),
        },
        "Optional post-training eval and promotion gate.",
        optional=("generation", "min_accuracy", "include_baseline"),
        open_input=True,
    )
    add_object(
        "SftRequest",
        "SftRequest",
        {
            "examples": array(ref("SftExample"), min_items=1),
            "dataset_path": nullable(ref("String")),
            "dataset": nullable(ref("String")),
            "config": ref("SftConfig"),
            "post_eval": nullable(ref("PostEvalConfig")),
        },
        "SFT submission using exactly one inline, local-path, or registered-dataset source.",
        optional=("examples", "dataset_path", "dataset", "config", "post_eval"),
        extra={
            "oneOf": [
                {"required": ["examples"], "not": {"anyOf": [active_optional("dataset_path"), active_optional("dataset")]}},
                {"required": ["dataset_path"], "properties": {"dataset_path": ref("NonEmptyString")}, "not": {"anyOf": [{"required": ["examples"]}, active_optional("dataset")]}},
                {"required": ["dataset"], "properties": {"dataset": ref("NonEmptyString")}, "not": {"anyOf": [{"required": ["examples"]}, active_optional("dataset_path")]}},
            ],
            "x-kiln-semantic-constraints": ["server SFT rejects train_mtp=true"],
        },
    )


def build_grpo_types() -> None:
    add_enum("AdvantageMode", "kiln_train::AdvantageMode", ["vanilla", "dr_grpo"], "Group-relative advantage normalization.")
    add_enum("LossAggregation", "kiln_train::LossAggregation", ["per_sample", "token_level"], "GRPO surrogate aggregation.")
    add_enum("IsLevel", "kiln_train::IsLevel", ["token", "sequence", "cispo"], "Importance-sampling granularity.")
    add_enum("BehaviorPolicy", "kiln_train::BehaviorPolicy", ["no_importance_correction", "recorded"], "Behavior-probability source.")
    add_enum("KlEstimator", "kiln_train::KlEstimator", ["k1", "k3", "none"], "Per-token KL estimator.")
    add_enum("RewardFilterOnEmpty", "kiln_train::RewardFilterOnEmpty", ["fail", "train-all", "skip"], "Behavior when variance filtering keeps too few groups.")
    add_enum("EnvMaskMode", "kiln_train::EnvMaskMode", ["env_only", "full_obs"], "Observation-token mask policy.")
    add_definition(
        "KlReferencePolicy",
        "kiln_train::KlReferencePolicy",
        {
            "oneOf": [
                tagged("base_per_step"),
                tagged("none"),
                tagged("ema", {"decay": ref("FiniteNumber"), "refresh_every": ref("PositiveInteger")}, optional=("decay", "refresh_every")),
            ]
        },
        "Frozen policy used only for the KL penalty.",
    )
    add_object("EchoConfig", "kiln_train::EchoConfig", {"lambda": ref("FiniteNumber"), "env_mask_mode": ref("EnvMaskMode"), "warning_filter": ref("Boolean")}, "ECHO observation-token auxiliary loss.", optional=("lambda", "env_mask_mode", "warning_filter"), open_input=True)
    add_object("OpdAuxConfig", "kiln_train::OpdAuxConfig", {"lambda": ref("FiniteNumber")}, "Reserved OPD auxiliary-loss slot.", optional=("lambda",), open_input=True)
    add_object("LossConfig", "kiln_train::LossConfig", {"echo": nullable(ref("EchoConfig")), "opd": nullable(ref("OpdAuxConfig")), "no_policy_loss": ref("Boolean")}, "Composition of policy, ECHO, and reserved OPD objectives.", optional=("echo", "opd", "no_policy_loss"), open_input=True)
    add_object(
        "TurnSegmentInput",
        "kiln_train::TurnSegment",
        {"role": ref("String"), "content": ref("String"), "kind": {"enum": ["context", "action", "observation"]}, "tool_call_id": ref("String"), "warning_prefix_len": ref("NonNegativeInteger")},
        "One trajectory segment accepted by training.",
        optional=("kind", "tool_call_id", "warning_prefix_len"),
        open_input=True,
    )
    add_object(
        "TurnSegmentOutput",
        "kiln_train::TurnSegment",
        {"role": ref("String"), "content": ref("String"), "kind": {"enum": ["context", "action", "observation"]}, "tool_call_id": ref("String"), "warning_prefix_len": ref("NonNegativeInteger")},
        "Closed emitted trajectory segment.",
        optional=("tool_call_id", "warning_prefix_len"),
    )
    add_object(
        "ScoredRollout",
        "kiln_train::ScoredRollout",
        {
            "text": ref("String"),
            "reward": ref("FiniteNumber"),
            "trajectory": array(ref("TurnSegmentInput")),
            "provenance": external_ref(INFERENCE_SCHEMA, "RolloutProvenanceV1"),
        },
        "One rewarded completion with optional trajectory and exact generation provenance.",
        optional=("trajectory", "provenance"),
        open_input=True,
    )
    add_object(
        "AgenticGroup",
        "kiln_train::AgenticGroup",
        {"messages": array(ref("TrainingChatMessageInput")), "completions": array(ref("ScoredRollout")), "rollouts": array(ref("ScoredRollout"))},
        "One prompt and its scored rollouts; rollouts is an input alias for completions.",
        optional=("completions", "rollouts"),
        open_input=True,
        extra={"oneOf": [{"required": ["completions"], "not": {"required": ["rollouts"]}}, {"required": ["rollouts"], "not": {"required": ["completions"]}}]},
    )
    grpo_fields = {
        "learning_rate": nullable(ref("FiniteNumber")), "kl_coeff": ref("FiniteNumber"), "clip_epsilon": ref("FiniteNumber"),
        "clip_eps_high": nullable(ref("FiniteNumber")), "cispo_max_weight": ref("FiniteNumber"), "advantage_mode": ref("AdvantageMode"),
        "loss_aggregation": ref("LossAggregation"), "kl_estimator": ref("KlEstimator"), "dynamic_sampling": ref("Boolean"),
        "is_level": ref("IsLevel"), "behavior_policy": ref("BehaviorPolicy"), "kl_reference_policy": ref("KlReferencePolicy"),
        "reference_policy": ref("KlReferencePolicy"), "entropy_aware_kl_quantile": nullable(ref("UnitInterval")),
        "reward_saturation_threshold": ref("FiniteNumber"), "reward_low_variance_threshold": ref("FiniteNumber"),
        "reward_filter_var_min": nullable(ref("FiniteNumber")), "reward_filter_var_max": nullable(ref("FiniteNumber")),
        "reward_filter_min_groups": ref("PositiveInteger"), "reward_filter_on_empty": ref("RewardFilterOnEmpty"),
        "lora_rank": ref("PositiveInteger"), "lora_alpha": ref("FiniteNumber"), "base_adapter": nullable(ref("String")),
        "allow_adapter_shape_conversion": ref("Boolean"), "allow_high_lora_scale": ref("Boolean"), "output_name": nullable(ref("String")),
        "auto_load": ref("Boolean"), "checkpoint_interval": nullable(ref("PositiveInteger")), "resume_checkpoint": nullable(ref("String")),
        "grad_checkpoint_segments": nullable(ref("PositiveInteger")), "seed": nullable(ref("NonNegativeInteger")), "optimizer": ref("Optimizer"),
        "adapter_smoke_test": ref("Boolean"), "loss": ref("LossConfig"),
    }
    add_object("GrpoConfig", "kiln_train::GrpoConfig", grpo_fields, "Complete GRPO optimizer, policy, filtering, checkpoint, and composite-loss configuration.", optional=tuple(grpo_fields), open_input=True, extra={"x-kiln-input-aliases": {"reference_policy": "kl_reference_policy"}})
    add_object(
        "GrpoRequest",
        "GrpoRequest",
        {"groups": array(ref("AgenticGroup"), min_items=1), "agentic_groups": array(ref("AgenticGroup"), min_items=1), "dataset_path": nullable(ref("String")), "dataset": nullable(ref("String")), "config": ref("GrpoConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "GRPO submission using exactly one inline, local-path, or registered-dataset source.",
        optional=("groups", "agentic_groups", "dataset_path", "dataset", "config", "post_eval"),
        open_input=True,
        extra={
            "oneOf": [
                {"required": ["groups"], "not": {"anyOf": [{"required": ["agentic_groups"]}, active_optional("dataset_path"), active_optional("dataset")]}},
                {"required": ["agentic_groups"], "not": {"anyOf": [{"required": ["groups"]}, active_optional("dataset_path"), active_optional("dataset")]}},
                {"required": ["dataset_path"], "properties": {"dataset_path": ref("NonEmptyString")}, "not": {"anyOf": [{"required": ["groups"]}, {"required": ["agentic_groups"]}, active_optional("dataset")]}},
                {"required": ["dataset"], "properties": {"dataset": ref("NonEmptyString")}, "not": {"anyOf": [{"required": ["groups"]}, {"required": ["agentic_groups"]}, active_optional("dataset_path")]}},
            ],
            "x-kiln-input-aliases": {"agentic_groups": "groups"},
            "x-kiln-semantic-constraints": ["recorded behavior policy requires provenance on every rollout"],
        },
    )


def build_opd_types() -> None:
    add_enum("OpdLossGranularity", "kiln_train::OpdLossGranularity", ["sampled_token", "teacher_top_k", "full_vocab"], "OPD teacher-support granularity; only teacher_top_k is server executable.")
    add_enum("OpdTrainingMode", "kiln_train::OpdTrainingMode", ["on_policy", "off_policy"], "Fresh-rollout versus teacher-authored replay mode.")
    add_enum("OpdObjective", "kiln_train::OpdObjective", ["reverse_kl", "cross_entropy"], "OPD objective; cross_entropy is a rejected compatibility value.")
    add_definition(
        "StableOpdMode",
        "kiln_train::StableOpdMode",
        {
            "oneOf": [
                object_schema({"mode": {"const": "off"}}, open_input=True),
                object_schema({"mode": {"const": "auto"}}, open_input=True),
                object_schema({"mode": {"const": "manual"}, "kl_beta": ref("FiniteNumber"), "sft_lambda": ref("FiniteNumber")}, open_input=True),
            ]
        },
        "Stable-OPD mode; only off is currently executable.",
    )
    add_object(
        "OpdPrompt",
        "kiln_train::OpdPrompt",
        {"messages": array(ref("TrainingChatMessageInput")), "teacher_extra_messages": array(ref("TrainingChatMessageInput")), "trajectory": array(ref("TurnSegmentInput"))},
        "Student-visible prompt with optional asymmetric teacher context and agentic replay.",
        optional=("teacher_extra_messages", "trajectory"),
        open_input=True,
    )
    opd_fields = {
        "training_mode": ref("OpdTrainingMode"), "objective": ref("OpdObjective"), "loss": ref("OpdLossGranularity"),
        "top_k": ref("PositiveInteger"), "samples_per_prompt": ref("PositiveInteger"), "temperature": ref("FiniteNumber"),
        "top_p": ref("FiniteNumber"), "max_tokens": ref("PositiveInteger"), "stable_opd": ref("StableOpdMode"),
        "discount": ref("FiniteNumber"), "clip_epsilon": ref("FiniteNumber"), "learning_rate": nullable(ref("FiniteNumber")),
        "lora_rank": ref("PositiveInteger"), "lora_alpha": ref("FiniteNumber"), "allow_high_lora_scale": ref("Boolean"),
        "base_adapter": nullable(ref("String")), "output_name": nullable(ref("String")), "auto_load": ref("Boolean"),
        "checkpoint_interval": nullable(ref("PositiveInteger")), "resume_checkpoint": nullable(ref("String")),
        "grad_checkpoint_segments": ref("PositiveInteger"), "seed": nullable(ref("NonNegativeInteger")), "optimizer": ref("Optimizer"),
        "echo": nullable(ref("EchoConfig")), "epochs": ref("PositiveInteger"), "max_cost_usd": nullable(ref("FiniteNumber")),
    }
    add_object(
        "OpdConfig",
        "kiln_train::OpdConfig",
        opd_fields,
        "Complete OPD rollout, objective, optimizer, checkpoint, and optional ECHO configuration.",
        optional=tuple(opd_fields),
        open_input=True,
        extra={"x-kiln-semantic-constraints": ["server loss is teacher_top_k", "stable_opd.mode is off", "discount and clip_epsilon are zero", "max_cost_usd is null"]},
    )
    add_object(
        "OpdRequest",
        "OpdRequest",
        {"prompts": array(ref("OpdPrompt"), min_items=1), "dataset_path": nullable(ref("String")), "teacher": ref("NonEmptyString"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Direct OPD submission using inline prompts or a server-local replay dataset.",
        optional=("prompts", "dataset_path", "config", "post_eval"),
        open_input=True,
        extra={
            "oneOf": [
                {"required": ["prompts"], "not": active_optional("dataset_path")},
                {"required": ["dataset_path"], "properties": {"dataset_path": ref("NonEmptyString")}, "not": {"required": ["prompts"]}},
            ],
            "x-kiln-semantic-constraints": ["teacher alias is registered"],
        },
    )
    add_definition(
        "NewKnowledgeSource",
        "kiln_train::NewKnowledgeSource",
        {
            "oneOf": [
                object_schema({"dataset": ref("NonEmptyString")}, open_input=True),
                object_schema({"examples": array(ref("OpdPrompt"), min_items=1)}, open_input=True),
            ]
        },
        "Registered dataset or inline prompts for a refresh recipe.",
    )
    add_object(
        "DistillRefreshRequest",
        "DistillRefreshRequest",
        {
            "name": ref("NonEmptyString"), "new_data": ref("NewKnowledgeSource"), "behavioural_teacher": ref("NonEmptyString"),
            "background_chat": ref("String"), "require_if_eval_recovery": ref("FiniteNumber"), "require_internal_qa_gain": ref("FiniteNumber"),
            "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig")), "if_eval_suite": nullable(ref("String")), "new_knowledge_eval_suite": nullable(ref("String")),
        },
        "Two-phase new-knowledge training and instruction-following recovery request.",
        optional=("background_chat", "require_if_eval_recovery", "require_internal_qa_gain", "config", "post_eval", "if_eval_suite", "new_knowledge_eval_suite"),
        open_input=True,
    )
    add_object("DistillMergeSource", "kiln_train::DistillMergeSource", {"adapter": ref("NonEmptyString"), "weight": ref("FiniteNumber")}, "One weighted adapter source.", optional=("weight",), open_input=True)
    add_object(
        "DistillMergeRequest",
        "DistillMergeRequest",
        {"name": ref("NonEmptyString"), "sources": array(ref("DistillMergeSource"), min_items=1), "student": ref("String"), "rollout_budget": ref("PositiveInteger"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Behavior-space adapter merge request.",
        optional=("student", "rollout_budget", "config", "post_eval"),
        open_input=True,
    )
    add_definition(
        "DistillPumpMode",
        "kiln_train::DistillPumpMode",
        {
            "oneOf": [
                object_schema({"domain": ref("NonEmptyString")}, open_input=True),
                object_schema({"examples": array(ref("OpdPrompt"), min_items=1)}, open_input=True),
                object_schema({"wide": ref("Boolean")}, open_input=True),
            ]
        },
        "Targeted-domain, inline-example, or wide-corpus knowledge-pump mode.",
    )
    add_object(
        "DistillPumpRequest",
        "DistillPumpRequest",
        {"name": ref("NonEmptyString"), "teacher": ref("NonEmptyString"), "mode": ref("DistillPumpMode"), "rank": nullable(ref("PositiveInteger")), "rollout_budget": ref("PositiveInteger"), "use_cache": ref("Boolean"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Knowledge-pump request with an explicit teacher and rollout budget.",
        optional=("rank", "rollout_budget", "use_cache", "config", "post_eval"),
        open_input=True,
    )
    add_enum("SelfDistillMode", "kiln_train::SelfDistillMode", ["ground_truth_conditioning", "conciseness", "document_as_pi", "reverse_teacher"], "Privileged-information self-distillation mode.")
    add_object(
        "DistillSelfRequest",
        "DistillSelfRequest",
        {"name": ref("NonEmptyString"), "mode": ref("SelfDistillMode"), "prompts": nullable(array(ref("OpdPrompt"))), "ground_truth": nullable(array(ref("String"))), "documents": nullable(array(ref("String"))), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Privileged-information self-distillation request.",
        optional=("prompts", "ground_truth", "documents", "config", "post_eval"),
        open_input=True,
    )


def build_training_responses() -> None:
    add_enum("TrainingState", "kiln_train::TrainingState", ["queued", "running", "completed", "failed"], "Training-job lifecycle state.")
    add_enum("TrainingJobType", "crate::state::TrainingJobType", ["sft", "grpo", "opd"], "Native training pipeline identity.")
    status_fields = {
        "job_id": ref("NonEmptyString"), "state": ref("TrainingState"), "progress": ref("FiniteNumber"),
        "current_loss": nullable(ref("FiniteNumber")), "adapter_name": nullable(ref("String")), "effective_seed": ref("DecimalU64"),
        "started_at": ref("String"), "elapsed_secs": ref("FiniteNumber"), "submitted_unix_ms": ref("NonNegativeInteger"),
        "finished_unix_ms": ref("NonNegativeInteger"), "job_type": ref("TrainingJobType"), "error": ref("String"),
        "post_eval_verdict": ref("String"), "gate_outcome": {"enum": ["promoted", "kept", "regression", "demoted", "error"]},
    }
    status_optional = ("effective_seed", "submitted_unix_ms", "finished_unix_ms", "job_type", "error", "post_eval_verdict", "gate_outcome")
    add_object("TrainingStatus", "TrainingStatus", status_fields, "Live or archived training-job status.", optional=status_optional)
    add_definition("Vec_TrainingStatus", "Vec<TrainingStatus>", array(ref("TrainingStatus")), "All retained training statuses.")
    add_object("TrainingResponse", "TrainingResponse", {"job_id": ref("NonEmptyString"), "state": {"const": "queued"}, "effective_seed": ref("DecimalU64"), "message": ref("String")}, "Immutable training admission receipt.")
    add_object("QueueStatusEntry", "QueueStatusEntry", {"job_id": ref("NonEmptyString"), "job_type": ref("TrainingJobType"), "adapter_name": ref("String"), "position": ref("PositiveInteger")}, "One queued training job and its FIFO position.")
    add_object("QueueResponse", "QueueResponse", {"running": nullable(ref("TrainingStatus")), "queued": array(ref("QueueStatusEntry")), "completed": array(ref("TrainingStatus"))}, "Training queue grouped into running, queued, and terminal jobs.")
    add_object("TrainingLossSample", "crate::state::TrainingLossSample", {"epoch": ref("NonNegativeInteger"), "progress": ref("FiniteNumber"), "loss": ref("FiniteNumber"), "elapsed_secs": ref("FiniteNumber")}, "One bounded live loss-curve sample.")
    add_enum("TrainingKind", "kiln_train::checkpoint::TrainingKind", ["sft", "grpo", "opd", "capability-distillation"], "Immutable checkpoint training kind.")
    add_object(
        "TrainingCheckpointSummary",
        "TrainingCheckpointSummary",
        {
            "resume_checkpoint": ref("NonEmptyString"), "checkpoint_id": ref("NonEmptyString"), "training_kind": ref("TrainingKind"),
            "data_source_kind": ref("String"), "global_step": ref("NonNegativeInteger"), "total_steps": ref("NonNegativeInteger"),
            "next_epoch_index": ref("NonNegativeInteger"), "next_cursor_in_epoch": ref("NonNegativeInteger"), "complete": ref("Boolean"),
            "created_at": ref("Rfc3339Timestamp"), "effective_config": ref("AnyJson"), "data_content_sha256": ref("Sha256"),
            "data_item_count": ref("NonNegativeInteger"), "teacher_id": ref("String"), "teacher_identity_revision": ref("String"), "teacher_content_revision": ref("String"),
        },
        "Newest manifest-valid resumable checkpoint summary.",
        optional=("teacher_id", "teacher_identity_revision", "teacher_content_revision"),
    )
    detail_fields = {
        **status_fields,
        "epoch": nullable(ref("NonNegativeInteger")), "adapter_path": nullable(ref("String")), "auto_load": ref("Boolean"),
        "linked_eval_job_ids": array(ref("String")), "loss_history": array(ref("TrainingLossSample")), "train_receipt": nullable(ref("AnyJson")),
        "replay_request": nullable(ref("AnyJson")), "latest_checkpoint": nullable(ref("TrainingCheckpointSummary")),
        "checkpoint_error": nullable(ref("String")), "metadata_error": nullable(ref("String")),
    }
    add_object("TrainingJobDetail", "TrainingJobDetail", detail_fields, "Training status plus curves, receipts, replay summary, and newest checkpoint.", optional=status_optional)
    add_definition(
        "CancelTrainingJobResponse",
        "CancelTrainingJobResponse",
        {
            "oneOf": [
                object_schema({"status": {"const": "cancelling"}, "job_id": ref("NonEmptyString"), "message": {"const": "stop requested — the trainer aborts at the next step boundary"}}),
                object_schema({"status": {"const": "cancelled"}, "job_id": ref("NonEmptyString")}),
            ]
        },
        "Cooperative running cancellation or immediate queued cancellation.",
    )
    add_object("DeleteTrainingJobResponse", "DeleteTrainingJobResponse", {"job_id": ref("NonEmptyString"), "status": {"const": "deleted"}, "removed_archive_file": ref("Boolean")}, "Terminal training-record deletion result.")


def build_preflight_types() -> None:
    def front_door_variant(kind: str, request: str, *, open_input: bool = True) -> dict[str, Any]:
        definition = DEFS[request]
        variant = tagged(
            kind,
            dict(definition["properties"]),
            optional=tuple(
                field for field in definition["properties"] if field not in definition["required"]
            ),
            open_input=open_input,
        )
        if "oneOf" in definition:
            variant["oneOf"] = definition["oneOf"]
        return variant

    front_door_variants = [
        front_door_variant("distill_refresh", "DistillRefreshRequest"),
        front_door_variant("distill_merge", "DistillMergeRequest"),
        front_door_variant("distill_pump", "DistillPumpRequest"),
        front_door_variant("opd", "OpdRequest"),
        front_door_variant("grpo", "GrpoRequest"),
        front_door_variant("sft", "SftRequest", open_input=False),
    ]
    add_definition("FrontDoorRequest", "FrontDoorRequest", {"oneOf": front_door_variants}, "Single tagged training front door across all native pipelines.")
    add_object("FrontDoorResponse", "FrontDoorResponse", {"picked": {"enum": ["distill_refresh", "distill_merge", "distill_pump", "opd", "grpo", "sft"]}, "training": ref("TrainingResponse")}, "Selected pipeline plus its admission receipt.")
    add_object(
        "CapacityRequest",
        "CapacityRequest",
        {"rollouts": ref("PositiveInteger"), "tokens_per_rollout": ref("PositiveInteger"), "top_k": ref("PositiveInteger"), "rank": ref("PositiveInteger"), "num_layers": ref("PositiveInteger"), "hidden_size": ref("PositiveInteger"), "initial_overlap_probe": nullable(ref("UnitInterval"))},
        "Preflight information-capacity estimate inputs.",
        optional=("num_layers", "hidden_size", "initial_overlap_probe"),
        open_input=True,
    )
    add_object("CapacityResponse", "CapacityResponse", {"bits_needed": ref("FiniteNumber"), "bits_storable_in_lora": ref("FiniteNumber"), "capacity_ratio": ref("FiniteNumber"), "expected_overlap_at_step_50": ref("UnitInterval"), "warnings": array(ref("String"))}, "Capacity estimate and admission warnings.")
    add_object(
        "CompatibilityRow",
        "CompatibilityRow",
        {"teacher": ref("String"), "student": ref("String"), "domain": ref("String"), "predicted_initial_overlap": ref("UnitInterval"), "recommended_rank": ref("PositiveInteger"), "cold_start_epochs": nullable(ref("PositiveInteger")), "expected_gpu_hours": ref("FiniteNumber"), "expected_cost_usd": nullable(ref("FiniteNumber")), "validation_eval": ref("String"), "expected_eval_delta_points": ref("FiniteNumber")},
        "One teacher/student/domain compatibility estimate.",
    )
    add_object("CompatibilityResponse", "CompatibilityResponse", {"matches": array(ref("CompatibilityRow")), "note": nullable(ref("String"))}, "Filtered compatibility-table results.")
    add_object(
        "TierDefaults",
        "TierDefaults",
        {"tier": {"enum": ["laptop", "prosumer", "corporate"]}, "default_logit_source": ref("String"), "default_loss": ref("String"), "default_top_k": ref("PositiveInteger"), "lora_rank": ref("PositiveInteger"), "batch_size": ref("PositiveInteger"), "samples_per_prompt_default": ref("PositiveInteger"), "samples_per_prompt_data_multiplier": ref("PositiveInteger"), "max_rollout_tokens": ref("PositiveInteger"), "auto_checkpoint_cadence_steps": ref("PositiveInteger"), "cost_cap_default_usd": nullable(ref("FiniteNumber")), "cold_start_overlap_threshold": ref("UnitInterval"), "mixture_distillation_golden_fraction": ref("UnitInterval"), "eval_gate_required": ref("Boolean"), "notifications_channels": array(ref("String"))},
        "Paper-cited defaults for one hardware/deployment tier.",
    )
    add_object("TierDefaultsResponse", "TierDefaultsResponse", {"tier": {"enum": ["laptop", "prosumer", "corporate"]}, "defaults": ref("TierDefaults")}, "One named tier and its defaults.")
    add_object("TierDefaultsListResponse", "TierDefaultsListResponse", {"tiers": array(ref("TierDefaults"), min_items=3)}, "All built-in tier defaults.")


def build_agent_types() -> None:
    add_enum("RunStatus", "crate::agent_runs::RunStatus", ["queued", "running", "completed", "failed", "aborted", "timed_out", "interrupted"], "Embedded agent-run lifecycle state.")
    run_fields = {
        "id": ref("NonEmptyString"), "task": ref("String"), "cwd": ref("String"), "label": ref("String"), "status": ref("RunStatus"),
        "created_unix_ms": ref("NonNegativeInteger"), "queue_seq": ref("NonNegativeInteger"), "started_unix_ms": ref("NonNegativeInteger"),
        "finished_unix_ms": ref("NonNegativeInteger"), "num_turns": ref("NonNegativeInteger"), "num_tool_calls": ref("NonNegativeInteger"),
        "session_id": ref("String"), "session_path": ref("String"), "trace_indexed": ref("Boolean"), "last_assistant_text": ref("String"), "error": ref("String"),
    }
    add_object("AgentRunRecord", "AgentRunRecord", run_fields, "One persisted embedded-agent run.", optional=("label", "started_unix_ms", "finished_unix_ms", "session_id", "session_path", "last_assistant_text", "error"))
    add_object(
        "CreateRunRequest",
        "CreateRunRequest",
        {"task": ref("NonEmptyString"), "cwd": nullable(ref("String")), "label": nullable(ref("String")), "tools": nullable(array(ref("String"))), "thinking_level": nullable({"enum": ["off", "minimal", "low", "medium", "high", "xhigh"]}), "timeout_secs": nullable({"type": "integer", "minimum": 10})},
        "Start one embedded agent run.",
        optional=("cwd", "label", "tools", "thinking_level", "timeout_secs"),
        open_input=True,
    )
    add_object("MessageRequest", "MessageRequest", {"message": ref("NonEmptyString")}, "Steering or follow-up message for an active run.", open_input=True)
    add_object("AgentRunsStatusResponse", "AgentRunsStatusResponse", {"enabled": ref("Boolean"), "disabled_reason": nullable(ref("String")), "pi_available": ref("Boolean"), "pi_path": nullable(ref("String")), "max_concurrent_runs": ref("PositiveInteger"), "active_runs": ref("NonNegativeInteger"), "sessions_dir": ref("String")}, "Embedded-run security gate and local pi availability.")
    add_object("AgentRunListResponse", "AgentRunListResponse", {"runs": array(ref("AgentRunRecord"))}, "Retained embedded-agent runs.")
    add_object("AgentRunEvent", "AgentRunEvent", {"seq": ref("NonNegativeInteger"), "event": ref("AnyJson")}, "One sequenced raw pi event.")
    add_object("AgentRunEventsResponse", "AgentRunEventsResponse", {"events": array(ref("AgentRunEvent")), "next_after": ref("NonNegativeInteger"), "status": ref("RunStatus"), "first_available_seq": nullable(ref("NonNegativeInteger")), "truncated": ref("Boolean")}, "Incremental event page with replay-gap detection.")
    add_object("AgentRunQueuedResponse", "AgentRunQueuedResponse", {"queued": {"const": True}}, "Control message accepted for delivery.")
    add_object("AgentRunAbortResponse", "AgentRunAbortResponse", {"aborting": {"const": True}}, "Cooperative abort accepted.")
    add_object("TerminalStatusResponse", "TerminalStatusResponse", {"enabled": ref("Boolean"), "disabled_reason": nullable(ref("String")), "pi_available": ref("Boolean"), "pi_path": nullable(ref("String")), "cwd": ref("String"), "session_active": ref("Boolean")}, "Embedded terminal security gate, binary, cwd, and session status.")
    add_object(
        "TraceOutcome",
        "TraceOutcome",
        {"ended_with_exit_0": nullable(ref("Boolean")), "user_edited_agent_files": array(ref("String")), "has_followup_attempt": nullable(ref("Boolean"))},
        "Best-effort outcome signals inferred from a pi session.",
    )
    add_object(
        "AgentTrace",
        "AgentTrace",
        {"id": ref("NonEmptyString"), "working_dir": ref("String"), "num_turns": ref("NonNegativeInteger"), "num_tool_calls": ref("NonNegativeInteger"), "outcome": ref("TraceOutcome"), "first_event_at": nullable(ref("Rfc3339Timestamp")), "last_event_at": nullable(ref("Rfc3339Timestamp")), "forked": ref("Boolean"), "parent_id": nullable(ref("String")), "tool_manifest_sha": nullable(ref("String")), "prompt_messages": array(ref("TrainingChatMessageOutput")), "trajectory": array(ref("TurnSegmentOutput"))},
        "Indexed pi session and canonical training trajectory.",
        optional=("prompt_messages", "trajectory"),
    )
    add_object("AgentTracesListResponse", "AgentTracesListResponse", {"traces": array(ref("AgentTrace"))}, "All indexed agent traces.")
    add_object("DiscoverRequest", "DiscoverRequest", {"path": nullable(ref("String"))}, "Optional server-local pi sessions directory.", optional=("path",), open_input=True)
    add_object("DiscoverResponse", "DiscoverResponse", {"indexed": ref("NonNegativeInteger"), "path": ref("String")}, "Trace discovery result.")
    add_object(
        "JudgeDistillRequest",
        "JudgeDistillRequest",
        {"name": ref("NonEmptyString"), "teacher": ref("NonEmptyString"), "include_pi_share": ref("Boolean"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Distill a local judge from indexed agent traces.",
        optional=("name", "teacher", "include_pi_share", "config", "post_eval"),
        open_input=True,
    )
    add_object("JudgeDistillResponse", "JudgeDistillResponse", {"job_id": ref("NonEmptyString"), "state": {"const": "queued"}, "effective_seed": ref("DecimalU64"), "message": ref("String")}, "Judge-distillation admission receipt.")
    add_object(
        "SelfImproveRequest",
        "SelfImproveRequest",
        {"agent": ref("NonEmptyString"), "judge": ref("NonEmptyString"), "crisp": ref("Boolean"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Queue the agentic self-improvement loop.",
        optional=("agent", "judge", "crisp", "config", "post_eval"),
        open_input=True,
    )
    add_object("SelfImproveResponse", "SelfImproveResponse", {"job_ids": array(ref("NonEmptyString"), min_items=1), "effective_seeds": mapping(ref("DecimalU64")), "state": {"const": "queued"}, "message": ref("String")}, "All phases admitted for one self-improvement round.")
    add_object(
        "JudgeDriftCheckRequest",
        "JudgeDriftCheckRequest",
        {"judge": ref("NonEmptyString"), "teacher": ref("NonEmptyString"), "sample_size": ref("PositiveInteger"), "agreement_threshold": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}},
        "Validated drift-check inputs. The endpoint currently returns only HTTP 501 after validation.",
        optional=("judge", "teacher", "sample_size", "agreement_threshold"),
        open_input=True,
        extra={"x-kiln-current-runtime-result": "http_501_not_implemented"},
    )


def build_recipe_types() -> None:
    add_definition(
        "PromptsSource",
        "PromptsSource",
        {"oneOf": [object_schema({"dataset": ref("NonEmptyString")}, open_input=True), object_schema({"prompts": array(ref("OpdPrompt"), min_items=1)}, open_input=True)]},
        "Registered dataset or inline OPD prompts.",
    )
    add_definition(
        "ExamplesSource",
        "ExamplesSource",
        {"oneOf": [object_schema({"dataset": ref("NonEmptyString")}, open_input=True), object_schema({"examples": array(ref("SftExample"), min_items=1)}, open_input=True)]},
        "Registered dataset or inline SFT examples.",
    )
    recipe_steps = [
        tagged("sft", {"name": ref("NonEmptyString"), "base_adapter": nullable(ref("String")), "examples_from": ref("ExamplesSource"), "config": ref("SftConfig")}, optional=("base_adapter", "config")),
        tagged("opd", {"name": ref("NonEmptyString"), "teacher": ref("NonEmptyString"), "prompts_from": ref("PromptsSource"), "config": ref("OpdConfig")}, optional=("config",)),
        tagged("distill_merge", {"name": ref("NonEmptyString"), "sources": array(ref("DistillMergeSource"), min_items=1), "student": ref("String"), "rollout_budget": ref("PositiveInteger"), "config": ref("OpdConfig")}, optional=("student", "rollout_budget", "config")),
        tagged("distill_pump", {"name": ref("NonEmptyString"), "teacher": ref("NonEmptyString"), "mode": ref("DistillPumpMode"), "config": ref("OpdConfig")}, optional=("config",)),
        tagged("distill_refresh", {"name": ref("NonEmptyString"), "new_data": ref("NewKnowledgeSource"), "behavioural_teacher": ref("NonEmptyString"), "background_chat": ref("String"), "config": ref("OpdConfig")}, optional=("background_chat", "config")),
        tagged("distill_self", {"name": ref("NonEmptyString"), "mode": ref("SelfDistillMode"), "config": ref("OpdConfig")}, optional=("config",)),
        tagged("post_eval", {"suite": ref("NonEmptyString"), "adapter": ref("String"), "require_min_score": ref("UnitInterval")}),
    ]
    add_definition("RecipeStep", "RecipeStep", {"oneOf": recipe_steps}, "One typed recipe step.")
    add_object("Recipe", "Recipe", {"name": ref("NonEmptyString"), "description": nullable(ref("String")), "steps": array(ref("RecipeStep"), min_items=1)}, "Inline multi-step training recipe.", optional=("description",), open_input=True)
    add_definition(
        "RecipeRunRequest",
        "RecipeRunRequest",
        {"oneOf": [object_schema({"recipe": ref("NonEmptyString"), "inputs": mapping(ref("AnyJson"))}, optional=("inputs",), open_input=True), object_schema({"body": ref("Recipe"), "inputs": mapping(ref("AnyJson"))}, optional=("inputs",), open_input=True)]},
        "Named or inline recipe invocation with optional placeholder inputs.",
    )
    add_object("RecipeRunResponse", "RecipeRunResponse", {"recipe": ref("NonEmptyString"), "job_ids": array(ref("NonEmptyString")), "effective_seeds": mapping(ref("DecimalU64")), "message": ref("String")}, "Queued jobs for one recipe execution.")
    add_object("RecipeAdmissionDescriptor", "RecipeAdmissionDescriptor", {"supported": ref("Boolean"), "unavailable_reason": nullable(ref("String"))}, "Current substrate admission for a built-in recipe.")
    add_object("RecipeDescriptor", "RecipeDescriptor", {"name": ref("NonEmptyString"), "description": nullable(ref("String")), "num_steps": ref("NonNegativeInteger"), "admission": ref("RecipeAdmissionDescriptor")}, "List projection for a built-in recipe.")
    add_object("RecipesListResponse", "RecipesListResponse", {"recipes": array(ref("RecipeDescriptor"))}, "All built-in recipes and their current admission state.")


def correction_fields() -> dict[str, dict[str, Any]]:
    return {
        "request_id": ref("NonEmptyString"), "agent": ref("String"), "adapter": nullable(ref("String")), "user": ref("NonEmptyString"),
        "original": ref("String"), "ideal": ref("String"), "truncated": ref("Boolean"), "created_at": ref("String"),
        "trained_into": ref("String"), "trained_at": ref("Rfc3339Timestamp"),
    }


def build_correction_types() -> None:
    fields = correction_fields()
    input_fields = {
        **fields,
        "trained_into": nullable(ref("String")),
        "trained_at": nullable(ref("Rfc3339Timestamp")),
    }
    add_object("CorrectionRowInput", "CorrectionRow", input_fields, "Correction upsert input. Server-owned timestamps and training markers may be omitted.", optional=("agent", "adapter", "original", "ideal", "truncated", "created_at", "trained_into", "trained_at"), open_input=True)
    add_object("CorrectionRow", "CorrectionRow", fields, "Closed emitted correction row.", optional=("trained_into", "trained_at"))
    add_object("ListResponse", "ListResponse", {"corrections": array(ref("CorrectionRow"))}, "Active or historical corrections.")
    add_object("MarkTrainedRequest", "MarkTrainedRequest", {"request_ids": array(ref("NonEmptyString"), min_items=1), "adapter": ref("NonEmptyString")}, "Mark selected correction rows as consumed by an adapter.", open_input=True)
    add_object("MarkTrainedResponse", "MarkTrainedResponse", {"status": {"const": "marked"}, "marked": ref("NonNegativeInteger")}, "Number of correction rows marked.")
    add_object("DeleteCorrectionResponse", "DeleteCorrectionResponse", {"status": {"const": "deleted"}, "request_id": ref("NonEmptyString")}, "Single correction deletion result.")
    add_object("ClearCorrectionsResponse", "ClearCorrectionsResponse", {"status": {"const": "cleared"}, "removed": ref("NonNegativeInteger")}, "Active-correction clear result; trained history remains.")


def build_library_types() -> None:
    add_object(
        "LibraryAdapterEntry",
        "LibraryAdapterEntry",
        {"id": ref("NonEmptyString"), "name": ref("NonEmptyString"), "source_kind": ref("String"), "description": nullable(ref("String")), "post_eval": mapping(ref("FiniteNumber")), "uploader": nullable(ref("String")), "size_bytes": nullable(ref("NonNegativeInteger"))},
        "One adapter-library catalog entry.",
    )
    add_object("LibraryListResponse", "LibraryListResponse", {"backend": ref("String"), "adapters": array(ref("LibraryAdapterEntry")), "note": nullable(ref("String"))}, "Configured library backend and visible catalog entries.")
    add_object("PublishPayload", "PublishPayload", {"description": nullable(ref("String")), "uploader": nullable(ref("String"))}, "Optional publication metadata.", optional=("description", "uploader"), open_input=True)
    add_object(
        "PublishToLibraryResponse",
        "PublishToLibraryResponse",
        {"status": {"const": "ready_to_publish"}, "backend": ref("String"), "intended_id": ref("NonEmptyString"), "uploader": nullable(ref("String")), "description": nullable(ref("String")), "receipt_schema_version": ref("PositiveInteger"), "note": ref("String")},
        "Validated local publication bundle. Remote upload is not yet operational.",
        extra={"x-kiln-current-runtime-result": "contract_only_no_remote_upload"},
    )


def training_response_example(job_id: str = "train-1") -> dict[str, Any]:
    return {"job_id": job_id, "state": "queued", "effective_seed": "42", "message": "Queued training job"}


def opd_config_example() -> dict[str, Any]:
    return {"training_mode": "on_policy", "objective": "reverse_kl", "loss": "teacher_top_k", "top_k": 32, "stable_opd": {"mode": "off"}, "optimizer": {"kind": "muon"}}


def run_example() -> dict[str, Any]:
    return {"id": "run-1", "task": "Fix the failing test", "cwd": "/srv/project", "status": "completed", "created_unix_ms": 1_700_000_000_000, "queue_seq": 1, "started_unix_ms": 1_700_000_000_010, "finished_unix_ms": 1_700_000_001_000, "num_turns": 2, "num_tool_calls": 1, "session_id": "session-1", "session_path": "/srv/adapters/agent_runs/sessions/session-1.jsonl", "trace_indexed": True, "last_assistant_text": "Fixed."}


def status_example() -> dict[str, Any]:
    return {"job_id": "train-1", "state": "completed", "progress": 1.0, "current_loss": 0.2, "adapter_name": "math-v1", "effective_seed": "42", "started_at": "12s ago", "elapsed_secs": 12.0, "submitted_unix_ms": 1_700_000_000_000, "finished_unix_ms": 1_700_000_012_000, "job_type": "sft"}


def build_examples() -> dict[str, list[Any]]:
    message = {"role": "user", "content": "Explain 2 + 2."}
    sft = {"examples": [{"messages": [message, {"role": "assistant", "content": "4"}]}], "config": {"output_name": "math-v1", "optimizer": {"kind": "muon"}}}
    opd = {"prompts": [{"messages": [message]}], "teacher": "qwen-27b", "config": opd_config_example()}
    grpo = {"groups": [{"messages": [message], "completions": [{"text": "4", "reward": 1.0}, {"text": "5", "reward": 0.0}]}], "config": {"optimizer": {"kind": "muon"}}}
    status = status_example()
    run = run_example()
    trace = {"id": "session-1", "working_dir": "/srv/project", "num_turns": 2, "num_tool_calls": 1, "outcome": {"ended_with_exit_0": True, "user_edited_agent_files": [], "has_followup_attempt": None}, "first_event_at": "2026-07-14T12:00:00Z", "last_event_at": "2026-07-14T12:01:00Z", "forked": False, "parent_id": None, "tool_manifest_sha": None, "prompt_messages": [message], "trajectory": [{"role": "assistant", "content": "Fixed.", "kind": "action"}]}
    correction_input = {"request_id": "chat-1", "agent": "pi", "adapter": None, "user": "2 + 2?", "original": "5", "ideal": "4", "truncated": False, "created_at": ""}
    correction = {**correction_input, "created_at": "2026-07-14T12:00:00Z"}
    tier = {"tier": "prosumer", "default_logit_source": "LocalTeacher", "default_loss": "teacher_top_k (K=32)", "default_top_k": 32, "lora_rank": 32, "batch_size": 16, "samples_per_prompt_default": 4, "samples_per_prompt_data_multiplier": 32, "max_rollout_tokens": 7168, "auto_checkpoint_cadence_steps": 10, "cost_cap_default_usd": 25.0, "cold_start_overlap_threshold": 0.5, "mixture_distillation_golden_fraction": 0.25, "eval_gate_required": True, "notifications_channels": ["desktop_tray", "email", "webhook"]}
    recipe = {"name": "quick-sft", "description": "Train one adapter", "steps": [{"kind": "sft", "name": "math-v1", "examples_from": {"examples": sft["examples"]}}]}
    examples: dict[str, list[Any]] = {
        "AgentRunAbortResponse": [{"aborting": True}],
        "AgentRunEventsResponse": [{"events": [{"seq": 0, "event": {"type": "message_end"}}], "next_after": 1, "status": "running", "first_available_seq": 0, "truncated": False}],
        "AgentRunListResponse": [{"runs": [run]}],
        "AgentRunQueuedResponse": [{"queued": True}],
        "AgentRunRecord": [run],
        "AgentRunsStatusResponse": [{"enabled": True, "disabled_reason": None, "pi_available": True, "pi_path": "/usr/bin/pi", "max_concurrent_runs": 2, "active_runs": 0, "sessions_dir": "/srv/adapters/agent_runs/sessions"}],
        "AgentTrace": [trace],
        "AgentTracesListResponse": [{"traces": [trace]}],
        "CancelTrainingJobResponse": [{"status": "cancelled", "job_id": "train-1"}, {"status": "cancelling", "job_id": "train-2", "message": "stop requested — the trainer aborts at the next step boundary"}],
        "CapacityRequest": [{"rollouts": 100, "tokens_per_rollout": 512, "top_k": 32, "rank": 32}],
        "CapacityResponse": [{"bits_needed": 256000.0, "bits_storable_in_lora": 41943040.0, "capacity_ratio": 163.84, "expected_overlap_at_step_50": 0.9, "warnings": []}],
        "ClearCorrectionsResponse": [{"status": "cleared", "removed": 1}],
        "CompatibilityResponse": [{"matches": [{"teacher": "qwen3.6-27b@vllm", "student": "Qwen3.5-4B@kiln", "domain": "math_reasoning", "predicted_initial_overlap": 0.78, "recommended_rank": 64, "cold_start_epochs": None, "expected_gpu_hours": 4.0, "expected_cost_usd": None, "validation_eval": "math-frontier-eval", "expected_eval_delta_points": 12.0}], "note": None}],
        "CorrectionRow": [correction],
        "CorrectionRowInput": [correction_input],
        "CreateRunRequest": [{"task": "Fix the failing test", "cwd": "/srv/project", "thinking_level": "high", "timeout_secs": 900}],
        "DeleteCorrectionResponse": [{"status": "deleted", "request_id": "chat-1"}],
        "DeleteTrainingJobResponse": [{"job_id": "train-1", "status": "deleted", "removed_archive_file": True}],
        "DiscoverRequest": [{"path": "/home/user/.pi/agent/sessions"}],
        "DiscoverResponse": [{"indexed": 4, "path": "/home/user/.pi/agent/sessions"}],
        "DistillMergeRequest": [{"name": "merged-v1", "sources": [{"adapter": "math-v1", "weight": 1.0}], "config": opd_config_example()}],
        "DistillPumpRequest": [{"name": "rust-v1", "teacher": "qwen-27b", "mode": {"domain": "rust"}, "rollout_budget": 1000, "config": opd_config_example()}],
        "DistillRefreshRequest": [{"name": "assistant-v1", "new_data": {"dataset": "new-docs"}, "behavioural_teacher": "prior-self", "config": opd_config_example()}],
        "DistillSelfRequest": [{"name": "concise-v1", "mode": "conciseness", "config": opd_config_example()}],
        "FrontDoorRequest": [{"kind": "sft", **sft}],
        "FrontDoorResponse": [{"picked": "sft", "training": training_response_example()}],
        "GrpoRequest": [grpo],
        "JudgeDistillRequest": [{"name": "judge-pi-v1", "teacher": "qwen-27b", "include_pi_share": False, "config": opd_config_example()}],
        "JudgeDistillResponse": [training_response_example("judge-1")],
        "JudgeDriftCheckRequest": [{"judge": "judge-pi-v1", "teacher": "qwen-27b", "sample_size": 50, "agreement_threshold": 0.8}],
        "LibraryListResponse": [{"backend": "https://library.kiln.run", "adapters": [], "note": "Library backend not yet operational"}],
        "ListResponse": [{"corrections": [correction]}],
        "MarkTrainedRequest": [{"request_ids": ["chat-1"], "adapter": "math-v1"}],
        "MarkTrainedResponse": [{"status": "marked", "marked": 1}],
        "MessageRequest": [{"message": "Also run the focused test."}],
        "OpdRequest": [opd],
        "PublishPayload": [{"description": "Math adapter", "uploader": "local-user"}],
        "PublishToLibraryResponse": [{"status": "ready_to_publish", "backend": "https://library.kiln.run", "intended_id": "math-v1@2026-07-14", "uploader": "local-user", "description": "Math adapter", "receipt_schema_version": 1, "note": "Library publish endpoint is contract-only until launch."}],
        "QueueResponse": [{"running": None, "queued": [{"job_id": "train-1", "job_type": "sft", "adapter_name": "math-v1", "position": 1}], "completed": []}],
        "RecipeRunRequest": [{"body": recipe, "inputs": {}}],
        "RecipeRunResponse": [{"recipe": "quick-sft", "job_ids": ["train-1"], "effective_seeds": {"train-1": "42"}, "message": "Queued recipe"}],
        "RecipesListResponse": [{"recipes": [{"name": "quick-sft", "description": "Train one adapter", "num_steps": 1, "admission": {"supported": True, "unavailable_reason": None}}]}],
        "SelfImproveRequest": [{"agent": "pi-coder-current", "judge": "judge-pi-v1", "crisp": True, "config": opd_config_example()}],
        "SelfImproveResponse": [{"job_ids": ["opd-1", "crisp-1"], "effective_seeds": {"opd-1": "42", "crisp-1": "43"}, "state": "queued", "message": "Self-improvement queued"}],
        "SftRequest": [sft],
        "TerminalStatusResponse": [{"enabled": True, "disabled_reason": None, "pi_available": True, "pi_path": "/usr/bin/pi", "cwd": "/srv/project", "session_active": False}],
        "TierDefaultsListResponse": [{"tiers": [tier, {**tier, "tier": "laptop"}, {**tier, "tier": "corporate"}]}],
        "TierDefaultsResponse": [{"tier": "prosumer", "defaults": tier}],
        "TrainingJobDetail": [{**status, "epoch": 3, "adapter_path": "/srv/adapters/math-v1", "auto_load": True, "linked_eval_job_ids": [], "loss_history": [{"epoch": 1, "progress": 0.5, "loss": 0.4, "elapsed_secs": 5.0}], "train_receipt": None, "replay_request": None, "latest_checkpoint": None, "checkpoint_error": None, "metadata_error": None}],
        "TrainingResponse": [training_response_example()],
        "TrainingStatus": [status],
        "Vec_TrainingStatus": [[status]],
    }
    if set(examples) != set(ENTRYPOINTS):
        missing = sorted(set(ENTRYPOINTS) - set(examples))
        extra = sorted(set(examples) - set(ENTRYPOINTS))
        raise ValueError(f"control-plane example coverage mismatch: missing={missing}, extra={extra}")
    return examples


def build_schema() -> dict[str, Any]:
    DEFS.clear()
    build_primitives()
    build_shared_training_types()
    build_grpo_types()
    build_opd_types()
    build_training_responses()
    build_preflight_types()
    build_agent_types()
    build_recipe_types()
    build_correction_types()
    build_library_types()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ericflo.github.io/kiln/contracts/kiln-control-plane-v1.schema.json",
        "title": "Kiln Training and Agent Control Plane API",
        "description": "Complete field-level wire contract for native training, distillation, preflight, embedded agents, recipes, corrections, and the adapter-library surface. Open inputs explicitly preserve accepted-and-ignored unknown fields; emitted objects are closed.",
        "x-kiln-field-schema-status": "complete",
        "x-kiln-entrypoints": list(ENTRYPOINTS),
        "x-kiln-external-contracts": [EVAL_SCHEMA, INFERENCE_SCHEMA],
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
        print(f"Control-plane schema is current: {len(DEFS)} reachable definitions, {len(ENTRYPOINTS)} entrypoints")
        return 0
    OUTPUT.write_text(rendered)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(DEFS)} definitions, {len(ENTRYPOINTS)} entrypoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
