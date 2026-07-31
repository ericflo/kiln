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
    "OpenEnvInspectRequest",
    "OpenEnvInspectResponse",
    "OpenEnvRunList",
    "OpenEnvRunRequest",
    "OpenEnvRunStatus",
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
    add_definition("AnyJson", "serde_json::Value", {}, "Any valid JSON value.")
    add_definition("Boolean", "bool", {"type": "boolean"}, "Either true or false.")
    add_definition("String", "String", {"type": "string"}, "A text string.")
    add_definition("NonEmptyString", "String", {"type": "string", "minLength": 1}, "A text string containing at least one character.")
    add_definition("NonNegativeInteger", "u64 | u32 | usize", {"type": "integer", "minimum": 0}, "A whole number greater than or equal to zero.")
    add_definition("PositiveInteger", "u64 | u32 | usize", {"type": "integer", "minimum": 1}, "A whole number greater than or equal to one.")
    add_definition("FiniteNumber", "f32 | f64", {"type": "number"}, "A finite JSON number.")
    add_definition("UnitInterval", "f32 | f64", {"type": "number", "minimum": 0, "maximum": 1}, "A number from 0 through 1, inclusive.")
    add_definition("DecimalU64", "u64", {"type": "string", "pattern": "^(0|[1-9][0-9]*)$"}, "An unsigned 64-bit integer encoded as decimal text to preserve exact values.")
    add_definition("Rfc3339Timestamp", "String", {"type": "string", "format": "date-time"}, "An RFC 3339 timestamp.")
    add_definition("Sha256", "String", {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}, "A lowercase SHA-256 digest prefixed with `sha256:`.")


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
        "A chat message accepted by training endpoints. `content` may be text, null, or a structured content-parts array; structured arrays are converted to text before training.",
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
        "A normalized chat message returned by Kiln.",
        optional=("tool_calls", "name", "tool_call_id"),
    )
    add_object("SftExample", "kiln_train::SftExample", {"messages": array(ref("TrainingChatMessageInput"))}, "One conversation submitted for supervised fine-tuning.", optional=("messages",), open_input=True)
    add_enum("SftTrainingProfile", "kiln_train::SftTrainingProfile", ["native_online_lora_v1"], "Versioned identifier for the native SFT workflow.")
    add_enum("SftInvalidRowPolicy", "kiln_train::SftInvalidRowPolicy", ["fail", "skip"], "Whether a malformed dataset row rejects the request or is skipped.")
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
        "Optimizer selection and the settings accepted by each optimizer. Unknown optimizer settings are rejected.",
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
        "detect_anomaly": ref("Boolean"),
        "seed": nullable(ref("NonNegativeInteger")),
        "optimizer": ref("Optimizer"),
        "adapter_smoke_test": ref("Boolean"),
        "adapter_smoke_prompts": nullable(array(ref("NonEmptyString"), min_items=1)),
    }
    add_object("SftConfig", "kiln_train::SftConfig", sft_config_fields, "Native SFT settings. Omitted fields use server defaults; unknown fields reject the request.", optional=tuple(sft_config_fields), open_input=False)
    add_enum(
        "PostEvalDataScope",
        "kiln_eval::PostEvalDataScope",
        ["held-out", "train-set-eval"],
        "Whether post-training evaluation is held out or an explicit train-set diagnostic.",
    )
    add_object(
        "PostEvalConfig",
        "kiln_eval::PostEvalConfig",
        {
            "suite": ref("NonEmptyString"),
            "data_scope": ref("PostEvalDataScope"),
            "generation": external_ref(EVAL_SCHEMA, "EvalGenerationParams"),
            "min_accuracy": ref("UnitInterval"),
            "include_baseline": ref("Boolean"),
        },
        "Optional evaluation after training. A minimum accuracy can prevent automatic adapter loading when the candidate misses the gate.",
        optional=("data_scope", "generation", "min_accuracy", "include_baseline"),
        open_input=True,
        extra={
            "allOf": [{
                "if": {"required": ["data_scope"], "properties": {"data_scope": {"const": "train-set-eval"}}},
                "then": {"not": {"required": ["min_accuracy"]}},
            }],
            "x-kiln-semantic-constraints": ["train-set-eval is diagnostic only and cannot set min_accuracy"],
        },
    )
    add_object(
        "SftRequest",
        "SftRequest",
        {
            "examples": array(ref("SftExample"), min_items=1),
            "dataset_path": nullable(ref("String")),
            "dataset": nullable(ref("String")),
            "dataset_split": external_ref(EVAL_SCHEMA, "DatasetSplit"),
            "config": ref("SftConfig"),
            "post_eval": nullable(ref("PostEvalConfig")),
        },
        "SFT submission using exactly one inline, local-path, or registered-dataset source.",
        optional=("examples", "dataset_path", "dataset", "dataset_split", "config", "post_eval"),
        extra={
            "oneOf": [
                {"required": ["examples"], "not": {"anyOf": [active_optional("dataset_path"), active_optional("dataset")]}},
                {"required": ["dataset_path"], "properties": {"dataset_path": ref("NonEmptyString")}, "not": {"anyOf": [{"required": ["examples"]}, active_optional("dataset")]}},
                {"required": ["dataset"], "properties": {"dataset": ref("NonEmptyString")}, "not": {"anyOf": [{"required": ["examples"]}, active_optional("dataset_path")]}},
            ],
            "allOf": [{
                "if": {"required": ["dataset_split"]},
                "then": {"required": ["dataset"], "properties": {"dataset": ref("NonEmptyString")}},
            }],
            "x-kiln-semantic-constraints": ["server SFT rejects train_mtp=true", "dataset_split requires a registered named dataset and defaults to train"],
        },
    )


def build_grpo_types() -> None:
    add_enum("AdvantageMode", "kiln_train::AdvantageMode", ["vanilla", "dr_grpo"], "Method used to normalize rewards within each rollout group.")
    add_enum("LossAggregation", "kiln_train::LossAggregation", ["per_sample", "token_level"], "Whether Kiln aggregates the GRPO objective by completion or by token.")
    add_enum("IsLevel", "kiln_train::IsLevel", ["token", "sequence", "cispo"], "Granularity used for importance-sampling correction.")
    add_enum("BehaviorPolicy", "kiln_train::BehaviorPolicy", ["no_importance_correction", "recorded"], "Whether training uses no importance correction or the probabilities recorded during rollout generation.")
    add_enum("KlEstimator", "kiln_train::KlEstimator", ["k1", "k3", "none"], "Estimator used for the per-token KL penalty.")
    add_enum("RewardFilterOnEmpty", "kiln_train::RewardFilterOnEmpty", ["fail", "train-all", "skip"], "What to do when reward-variance filtering retains too few groups.")
    add_enum("EnvMaskMode", "kiln_train::EnvMaskMode", ["env_only", "full_obs"], "Which observation tokens contribute to the ECHO auxiliary objective.")
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
        "Reference-policy strategy used only to calculate the KL penalty.",
    )
    add_object("EchoConfig", "kiln_train::EchoConfig", {"lambda": ref("FiniteNumber"), "env_mask_mode": ref("EnvMaskMode"), "warning_filter": ref("Boolean")}, "Settings for ECHO's observation-token auxiliary objective.", optional=("lambda", "env_mask_mode", "warning_filter"), open_input=True)
    add_object("OpdAuxConfig", "kiln_train::OpdAuxConfig", {"lambda": ref("FiniteNumber")}, "Reserved GRPO auxiliary-objective shape. Supplying `loss.opd` currently rejects the request; use `/v1/train/opd` instead.", optional=("lambda",), open_input=True)
    add_object("LossConfig", "kiln_train::LossConfig", {"echo": nullable(ref("EchoConfig")), "opd": nullable(ref("OpdAuxConfig")), "no_policy_loss": ref("Boolean")}, "GRPO objective settings. ECHO is supported; `loss.opd` is reserved and currently rejects the request.", optional=("echo", "opd", "no_policy_loss"), open_input=True)
    add_object(
        "TurnSegmentInput",
        "kiln_train::TurnSegment",
        {"role": ref("String"), "content": ref("String"), "kind": {"enum": ["context", "action", "observation"]}, "tool_call_id": ref("String"), "warning_prefix_len": ref("NonNegativeInteger")},
        "One context, model-action, or environment-observation segment accepted by training.",
        optional=("kind", "tool_call_id", "warning_prefix_len"),
        open_input=True,
    )
    add_object(
        "TurnSegmentOutput",
        "kiln_train::TurnSegment",
        {"role": ref("String"), "content": ref("String"), "kind": {"enum": ["context", "action", "observation"]}, "tool_call_id": ref("String"), "warning_prefix_len": ref("NonNegativeInteger")},
        "One normalized trajectory segment returned by Kiln.",
        optional=("tool_call_id", "warning_prefix_len"),
    )
    add_enum(
        "OpenEnvEpisodeTerminationV1",
        "kiln_train::OpenEnvEpisodeTerminationV1",
        ["done", "max_steps", "invalid_model_action", "protocol_error"],
        "Why an OpenEnv episode ended from Kiln's point of view.",
    )
    add_object(
        "OpenEnvRolloutProvenanceV1",
        "kiln_train::OpenEnvRolloutProvenanceV1",
        {
            "schema": {"const": "kiln.openenv-rollout.v1"},
            "environment_name": ref("NonEmptyString"),
            "environment_base_url": ref("NonEmptyString"),
            "openapi_version": ref("NonEmptyString"),
            "environment_schema_sha256": ref("Sha256"),
            "action_schema_sha256": ref("Sha256"),
            "reset_sha256": ref("Sha256"),
            "seed": ref("NonNegativeInteger"),
            "steps": ref("NonNegativeInteger"),
            "episode_return": ref("FiniteNumber"),
            "terminal_done": ref("Boolean"),
            "termination": ref("OpenEnvEpisodeTerminationV1"),
            "protocol_error_code": ref("NonEmptyString"),
        },
        "Fail-closed environment, task, and outcome identity attached to a native OpenEnv rollout.",
        optional=("openapi_version", "protocol_error_code"),
        extra={
            "x-kiln-semantic-constraints": [
                "terminal_done is true exactly when termination is done",
                "protocol_error_code is present exactly when termination is protocol_error",
            ]
        },
    )
    add_object(
        "ScoredRollout",
        "kiln_train::ScoredRollout",
        {
            "text": ref("String"),
            "reward": ref("FiniteNumber"),
            "trajectory": array(ref("TurnSegmentInput")),
            "provenance": external_ref(INFERENCE_SCHEMA, "RolloutProvenanceV1"),
            "openenv": ref("OpenEnvRolloutProvenanceV1"),
        },
        "One rewarded completion with optional trajectory, exact generation provenance, and OpenEnv episode identity.",
        optional=("trajectory", "provenance", "openenv"),
        open_input=True,
    )
    add_object(
        "AgenticGroup",
        "kiln_train::AgenticGroup",
        {"messages": array(ref("TrainingChatMessageInput")), "completions": array(ref("ScoredRollout")), "rollouts": array(ref("ScoredRollout"))},
        "One prompt and its scored rollouts. `rollouts` is an input alias for `completions`; send exactly one of them.",
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
        "grad_checkpoint_segments": nullable(ref("PositiveInteger")), "detect_anomaly": ref("Boolean"),
        "seed": nullable(ref("NonNegativeInteger")), "optimizer": ref("Optimizer"),
        "adapter_smoke_test": ref("Boolean"),
        "adapter_smoke_prompts": nullable(array(ref("NonEmptyString"), min_items=1)),
        "shared_prefix_reference": ref("Boolean"), "loss": ref("LossConfig"),
    }
    add_object("GrpoConfig", "kiln_train::GrpoConfig", grpo_fields, "Complete GRPO optimizer, policy, filtering, checkpoint, and composite-loss configuration.", optional=tuple(grpo_fields), open_input=True, extra={"x-kiln-input-aliases": {"reference_policy": "kl_reference_policy"}})
    add_object(
        "GrpoRequest",
        "GrpoRequest",
        {"groups": array(ref("AgenticGroup"), min_items=1), "agentic_groups": array(ref("AgenticGroup"), min_items=1), "dataset_path": nullable(ref("String")), "dataset": nullable(ref("String")), "dataset_split": external_ref(EVAL_SCHEMA, "DatasetSplit"), "config": ref("GrpoConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "GRPO submission using exactly one inline, local-path, or registered-dataset source.",
        optional=("groups", "agentic_groups", "dataset_path", "dataset", "dataset_split", "config", "post_eval"),
        open_input=True,
        extra={
            "oneOf": [
                {"required": ["groups"], "not": {"anyOf": [{"required": ["agentic_groups"]}, active_optional("dataset_path"), active_optional("dataset")]}},
                {"required": ["agentic_groups"], "not": {"anyOf": [{"required": ["groups"]}, active_optional("dataset_path"), active_optional("dataset")]}},
                {"required": ["dataset_path"], "properties": {"dataset_path": ref("NonEmptyString")}, "not": {"anyOf": [{"required": ["groups"]}, {"required": ["agentic_groups"]}, active_optional("dataset")]}},
                {"required": ["dataset"], "properties": {"dataset": ref("NonEmptyString")}, "not": {"anyOf": [{"required": ["groups"]}, {"required": ["agentic_groups"]}, active_optional("dataset_path")]}},
            ],
            "x-kiln-input-aliases": {"agentic_groups": "groups"},
            "allOf": [{
                "if": {"required": ["dataset_split"]},
                "then": {"required": ["dataset"], "properties": {"dataset": ref("NonEmptyString")}},
            }],
            "x-kiln-semantic-constraints": ["recorded behavior policy requires provenance on every rollout", "dataset_split requires a registered named dataset and defaults to train"],
        },
    )


def build_openenv_types() -> None:
    add_object(
        "OpenEnvMetadata",
        "kiln_openenv::OpenEnvMetadata",
        {
            "name": ref("String"),
            "description": ref("String"),
            "readme_content": nullable(ref("String")),
            "version": nullable(ref("String")),
            "author": nullable(ref("String")),
            "documentation_url": nullable(ref("String")),
        },
        "Environment-authored metadata discovered from an OpenEnv server.",
        optional=("readme_content", "version", "author", "documentation_url"),
    )
    add_object(
        "OpenEnvSchema",
        "kiln_openenv::OpenEnvSchema",
        {
            "action": ref("AnyJson"),
            "observation": ref("AnyJson"),
            "state": ref("AnyJson"),
        },
        "JSON Schemas for OpenEnv actions, observations, and state.",
    )
    add_object(
        "OpenEnvIdentity",
        "kiln_openenv::OpenEnvIdentity",
        {
            "schema": ref("String"),
            "client_profile": ref("String"),
            "base_url": ref("String"),
            "websocket_url": ref("String"),
            "openapi_version": nullable(ref("String")),
            "environments": array(ref("String")),
            "metadata": ref("OpenEnvMetadata"),
            "schema_sha256": ref("Sha256"),
        },
        "Content-addressed identity established by OpenEnv discovery.",
        optional=("openapi_version",),
    )
    add_object(
        "OpenEnvInspection",
        "kiln_openenv::OpenEnvInspection",
        {
            "identity": ref("OpenEnvIdentity"),
            "schema": ref("OpenEnvSchema"),
        },
        "Complete protocol discovery result for one OpenEnv server.",
    )
    add_enum(
        "OpenEnvRunKind",
        "OpenEnvRunKind",
        ["rollout", "train"],
        "Whether a server-owned run stops after artifacts or queues native GRPO.",
    )
    add_enum(
        "OpenEnvRunState",
        "OpenEnvRunState",
        [
            "queued",
            "discovering",
            "collecting",
            "submitting",
            "rollout_ready",
            "training_queued",
            "training_running",
            "post_evaluating",
            "completed",
            "failed",
            "cancelled",
        ],
        "Persisted OpenEnv orchestration lifecycle state.",
    )
    add_object(
        "OpenEnvRunProgress",
        "OpenEnvRunProgress",
        {
            "groups_completed": ref("NonNegativeInteger"),
            "groups_total": ref("PositiveInteger"),
            "rollouts_completed": ref("NonNegativeInteger"),
            "rollouts_total": ref("PositiveInteger"),
        },
        "Bounded group and episode progress for one OpenEnv run.",
    )
    add_object(
        "OpenEnvArtifact",
        "OpenEnvArtifact",
        {
            "kind": {"enum": ["dataset", "replay", "summary"]},
            "url": ref("String"),
            "sha256": ref("Sha256"),
            "bytes": ref("NonNegativeInteger"),
        },
        "Content-addressed artifact downloadable from a retained OpenEnv run.",
    )
    add_object(
        "OpenEnvTrainingStatus",
        "OpenEnvTrainingStatus",
        {
            "job_id": ref("NonEmptyString"),
            "state": ref("TrainingState"),
            "progress": ref("FiniteNumber"),
            "current_loss": ref("FiniteNumber"),
            "epoch": ref("NonNegativeInteger"),
            "adapter_path": ref("String"),
            "linked_eval_job_ids": array(ref("String")),
            "post_eval_verdict": ref("String"),
            "gate_outcome": {
                "enum": [
                    "promoted",
                    "kept",
                    "regression",
                    "demoted",
                    "inconclusive",
                    "error",
                ]
            },
            "error": ref("String"),
        },
        "Authoritative bounded projection of the native trainer owned by an OpenEnv run.",
        optional=(
            "current_loss",
            "epoch",
            "adapter_path",
            "linked_eval_job_ids",
            "post_eval_verdict",
            "gate_outcome",
            "error",
        ),
    )
    add_object(
        "OpenEnvPostEvalStatus",
        "OpenEnvPostEvalStatus",
        {
            "job_id": ref("NonEmptyString"),
            "suite_name": ref("NonEmptyString"),
            "state": external_ref(EVAL_SCHEMA, "EvalJobState"),
            "examples_completed": ref("NonNegativeInteger"),
            "examples_total": ref("NonNegativeInteger"),
            "headline_accuracy": ref("FiniteNumber"),
            "error": ref("String"),
        },
        "Bounded projection of one post-training evaluation linked to an OpenEnv run.",
        optional=("headline_accuracy", "error"),
    )
    add_object(
        "OpenEnvRunRequest",
        "OpenEnvRunRequest",
        {
            "kind": ref("OpenEnvRunKind"),
            "environment_urls": array(ref("NonEmptyString"), min_items=1),
            "adapter": ref("NonEmptyString"),
            "groups": ref("PositiveInteger"),
            "group_size": ref("PositiveInteger"),
            "seed_start": ref("NonNegativeInteger"),
            "reset_options": {"type": "object"},
            "max_steps": ref("PositiveInteger"),
            "concurrency": ref("PositiveInteger"),
            "max_action_tokens": ref("PositiveInteger"),
            "temperature": ref("FiniteNumber"),
            "thinking": ref("Boolean"),
            "protocol_error_reward": ref("FiniteNumber"),
            "max_recoverable_errors": ref("NonNegativeInteger"),
            "capacity_wait_seconds": ref("PositiveInteger"),
            "output_adapter": ref("NonEmptyString"),
            "training_config": ref("GrpoConfig"),
            "auto_load": ref("Boolean"),
            "post_eval": ref("PostEvalConfig"),
        },
        "A persisted OpenEnv rollout or rollout-and-train request. `environments` is accepted as an input alias for `environment_urls`.",
        optional=(
            "kind",
            "adapter",
            "groups",
            "group_size",
            "seed_start",
            "reset_options",
            "max_steps",
            "concurrency",
            "max_action_tokens",
            "temperature",
            "thinking",
            "protocol_error_reward",
            "max_recoverable_errors",
            "capacity_wait_seconds",
            "output_adapter",
            "training_config",
            "auto_load",
            "post_eval",
        ),
        extra={
            "x-kiln-input-aliases": {"environments": "environment_urls"},
            "x-kiln-semantic-constraints": [
                "kind=train requires output_adapter",
                "kind=rollout rejects output_adapter",
                "Kiln overrides behavior_policy, base_adapter, output_name, and auto_load in training_config",
            ],
        },
    )
    add_object(
        "OpenEnvRunStatus",
        "OpenEnvRunStatus",
        {
            "schema": {
                "enum": ["kiln.openenv-run.v1", "kiln.openenv-run.v2"]
            },
            "run_id": ref("String"),
            "kind": ref("OpenEnvRunKind"),
            "state": ref("OpenEnvRunState"),
            "request": ref("OpenEnvRunRequest"),
            "submitted_unix_ms": ref("NonNegativeInteger"),
            "finished_unix_ms": ref("NonNegativeInteger"),
            "progress": ref("OpenEnvRunProgress"),
            "environments": array(ref("OpenEnvIdentity")),
            "artifacts": array(ref("OpenEnvArtifact")),
            "training_job_id": ref("String"),
            "training_submission": ref("TrainingResponse"),
            "training": ref("OpenEnvTrainingStatus"),
            "post_evaluations": array(ref("OpenEnvPostEvalStatus")),
            "error": ref("String"),
        },
        "Persisted status, collection artifacts, and authoritative training and post-evaluation lifecycle for one OpenEnv run. Version 1 records may retain the historical terminal training_queued handoff; version 2 follows learning to completion.",
        optional=(
            "finished_unix_ms",
            "environments",
            "artifacts",
            "training_job_id",
            "training_submission",
            "training",
            "post_evaluations",
            "error",
        ),
    )
    add_object(
        "OpenEnvRunList",
        "OpenEnvRunList",
        {
            "schema": {"const": "kiln.openenv-run-list.v2"},
            "runs": array(ref("OpenEnvRunStatus")),
        },
        "Newest-first retained OpenEnv run records.",
    )
    add_object(
        "OpenEnvInspectRequest",
        "OpenEnvInspectRequest",
        {"environment_urls": array(ref("NonEmptyString"), min_items=1)},
        "One or more OpenEnv HTTP base URLs to discover. `environments` is accepted as an input alias.",
        extra={"x-kiln-input-aliases": {"environments": "environment_urls"}},
    )
    add_object(
        "OpenEnvInspectResponse",
        "OpenEnvInspectResponse",
        {
            "schema": {"const": "kiln.openenv-inspection.v1"},
            "environments": array(ref("OpenEnvInspection"), min_items=1),
        },
        "Complete discovery results for the requested OpenEnv servers.",
    )


def build_opd_types() -> None:
    add_enum("OpdLossGranularity", "kiln_train::OpdLossGranularity", ["sampled_token", "teacher_top_k", "full_vocab"], "Amount of teacher-distribution data used by OPD. The server currently executes only `teacher_top_k`.")
    add_enum("OpdTrainingMode", "kiln_train::OpdTrainingMode", ["on_policy", "off_policy"], "Whether OPD generates fresh student rollouts or replays teacher-authored data.")
    add_enum("OpdObjective", "kiln_train::OpdObjective", ["reverse_kl", "cross_entropy"], "OPD objective. `cross_entropy` is parsed for compatibility but rejected by the server.")
    add_enum("OpdRolloutPromptRendering", "kiln_train::OpdRolloutPromptRendering", ["legacy_action_boundary", "chat_template"], "How Kiln constructs the student rollout prefix. `chat_template` is experimental.")
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
        "Stable-OPD setting. Only `off` is currently executable.",
    )
    add_object(
        "OpdPrompt",
        "kiln_train::OpdPrompt",
        {"messages": array(ref("TrainingChatMessageInput")), "teacher_extra_messages": array(ref("TrainingChatMessageInput")), "trajectory": array(ref("TurnSegmentInput"))},
        "Student-visible prompt with optional teacher-only context and an optional agent trajectory to replay.",
        optional=("teacher_extra_messages", "trajectory"),
        open_input=True,
    )
    opd_fields = {
        "training_mode": ref("OpdTrainingMode"), "objective": ref("OpdObjective"), "loss": ref("OpdLossGranularity"),
        "top_k": ref("PositiveInteger"), "samples_per_prompt": ref("PositiveInteger"), "temperature": ref("FiniteNumber"),
        "top_p": ref("FiniteNumber"), "max_tokens": ref("PositiveInteger"), "sampler_segments": nullable(ref("PositiveInteger")),
        "rollout_prompt_rendering": ref("OpdRolloutPromptRendering"), "stable_opd": ref("StableOpdMode"),
        "discount": ref("FiniteNumber"), "clip_epsilon": ref("FiniteNumber"), "learning_rate": nullable(ref("FiniteNumber")),
        "lora_rank": ref("PositiveInteger"), "lora_alpha": ref("FiniteNumber"), "allow_high_lora_scale": ref("Boolean"),
        "base_adapter": nullable(ref("String")), "output_name": nullable(ref("String")), "auto_load": ref("Boolean"),
        "checkpoint_interval": nullable(ref("PositiveInteger")), "resume_checkpoint": nullable(ref("String")),
        "grad_checkpoint_segments": ref("PositiveInteger"), "detect_anomaly": ref("Boolean"),
        "seed": nullable(ref("NonNegativeInteger")), "optimizer": ref("Optimizer"),
        "echo": nullable(ref("EchoConfig")), "epochs": ref("PositiveInteger"), "max_cost_usd": nullable(ref("FiniteNumber")),
    }
    add_object(
        "OpdConfig",
        "kiln_train::OpdConfig",
        opd_fields,
        "OPD rollout, objective, optimizer, checkpoint, and optional ECHO settings. The semantic constraints list values the server currently accepts.",
        optional=tuple(opd_fields),
        open_input=True,
        extra={"x-kiln-semantic-constraints": ["server loss is teacher_top_k", "stable_opd.mode is off", "discount and clip_epsilon are zero", "max_cost_usd is null"]},
    )
    add_object(
        "OpdRequest",
        "OpdRequest",
        {"prompts": array(ref("OpdPrompt"), min_items=1), "dataset_path": nullable(ref("String")), "teacher": ref("NonEmptyString"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Direct OPD submission using either inline prompts or a replay dataset path on the server.",
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
        "Either a registered dataset name or inline prompts for a refresh workflow.",
    )
    add_object(
        "DistillRefreshRequest",
        "DistillRefreshRequest",
        {
            "name": ref("NonEmptyString"), "new_data": ref("NewKnowledgeSource"), "behavioural_teacher": ref("NonEmptyString"),
            "background_chat": ref("String"), "require_if_eval_recovery": ref("FiniteNumber"), "require_internal_qa_gain": ref("FiniteNumber"),
            "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig")), "if_eval_suite": nullable(ref("String")), "new_knowledge_eval_suite": nullable(ref("String")),
        },
        "Two-phase request for learning new material and recovering instruction-following behavior. Automatic loading supports at most one gate across `post_eval`, `if_eval_suite`, and `new_knowledge_eval_suite`; disable `config.auto_load` for independent multi-suite diagnostics.",
        optional=("background_chat", "require_if_eval_recovery", "require_internal_qa_gain", "config", "post_eval", "if_eval_suite", "new_knowledge_eval_suite"),
        open_input=True,
    )
    add_object("DistillMergeSource", "kiln_train::DistillMergeSource", {"adapter": ref("NonEmptyString"), "weight": ref("FiniteNumber")}, "One source adapter and its optional merge weight.", optional=("weight",), open_input=True)
    add_object(
        "DistillMergeRequest",
        "DistillMergeRequest",
        {"name": ref("NonEmptyString"), "sources": array(ref("DistillMergeSource"), min_items=1), "student": ref("String"), "rollout_budget": ref("PositiveInteger"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Request to distill behavior from one or more source adapters into a new adapter.",
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
        "Choose a named domain, inline examples, or the built-in wide-corpus mode for knowledge pumping.",
    )
    add_object(
        "DistillPumpRequest",
        "DistillPumpRequest",
        {"name": ref("NonEmptyString"), "teacher": ref("NonEmptyString"), "mode": ref("DistillPumpMode"), "rank": nullable(ref("PositiveInteger")), "rollout_budget": ref("PositiveInteger"), "use_cache": ref("Boolean"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Knowledge-pump request with a registered teacher alias and a rollout budget.",
        optional=("rank", "rollout_budget", "use_cache", "config", "post_eval"),
        open_input=True,
    )
    add_enum("SelfDistillMode", "kiln_train::SelfDistillMode", ["ground_truth_conditioning", "conciseness", "document_as_pi", "reverse_teacher"], "Self-distillation strategy for using information available only during training.")
    add_object(
        "DistillSelfRequest",
        "DistillSelfRequest",
        {"name": ref("NonEmptyString"), "mode": ref("SelfDistillMode"), "prompts": nullable(array(ref("OpdPrompt"))), "ground_truth": nullable(array(ref("String"))), "documents": nullable(array(ref("String"))), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Self-distillation request that uses information available only during training.",
        optional=("prompts", "ground_truth", "documents", "config", "post_eval"),
        open_input=True,
    )


def build_training_responses() -> None:
    add_enum("TrainingState", "kiln_train::TrainingState", ["queued", "running", "completed", "failed"], "Training-job lifecycle state. Cancelled jobs are represented as `failed` with a cancellation error.")
    add_enum("TrainingJobType", "crate::state::TrainingJobType", ["sft", "grpo", "opd"], "Training pipeline used by the job.")
    add_object(
        "PostEvalGateEvidence",
        "kiln_train::PostEvalGateEvidence",
        {
            "policy_version": ref("NonEmptyString"),
            "suite_policy": ref("NonEmptyString"),
            "eval_job_id": ref("NonEmptyString"),
            "suite_name": ref("NonEmptyString"),
            "suite_hash": ref("NonEmptyString"),
            "effective_generation_hash": ref("NonEmptyString"),
            "baseline_adapter": ref("String"),
            "candidate_adapter": ref("NonEmptyString"),
            "aggregation": ref("NonEmptyString"),
            "minimum_paired_examples": ref("PositiveInteger"),
            "paired_examples": ref("PositiveInteger"),
            "improved": ref("NonNegativeInteger"),
            "regressed": ref("NonNegativeInteger"),
            "tied": ref("NonNegativeInteger"),
            "exact_sign_test_p_value": ref("FiniteNumber"),
            "exact_sign_test_alpha": ref("FiniteNumber"),
            "baseline_accuracy": ref("FiniteNumber"),
            "baseline_accuracy_lower_bound": ref("FiniteNumber"),
            "baseline_accuracy_upper_bound": ref("FiniteNumber"),
            "candidate_accuracy": ref("FiniteNumber"),
            "candidate_accuracy_lower_bound": ref("FiniteNumber"),
            "candidate_accuracy_upper_bound": ref("FiniteNumber"),
            "accuracy_confidence_level": ref("FiniteNumber"),
            "minimum_accuracy": ref("FiniteNumber"),
            "required_relative_recovery": ref("FiniteNumber"),
            "relative_recovery_lower_bound": ref("FiniteNumber"),
            "required_absolute_gain": ref("FiniteNumber"),
            "absolute_gain_lower_bound": ref("FiniteNumber"),
            "outcome": {"enum": ["promoted", "kept", "regression", "demoted", "inconclusive", "error"]},
        },
        "Stored paired-test and confidence-interval evidence for one post-training promotion decision.",
        optional=(
            "baseline_adapter",
            "required_relative_recovery",
            "relative_recovery_lower_bound",
            "required_absolute_gain",
            "absolute_gain_lower_bound",
        ),
    )
    add_object(
        "TrainingDataProvenance",
        "kiln_train::TrainingDataProvenance",
        {
            "source": ref("NonEmptyString"),
            "dataset": ref("NonEmptyString"),
            "split": external_ref(EVAL_SCHEMA, "DatasetSplit"),
            "dataset_corpus_sha256": external_ref(EVAL_SCHEMA, "Sha256Digest"),
            "split_manifest_sha256": external_ref(EVAL_SCHEMA, "Sha256Digest"),
            "admitted_corpus_sha256": external_ref(EVAL_SCHEMA, "Sha256Digest"),
            "rows": ref("NonNegativeInteger"),
        },
        "Dataset and split identity recorded when Kiln accepts the training request.",
        optional=("dataset", "split", "dataset_corpus_sha256", "split_manifest_sha256"),
    )
    status_fields = {
        "job_id": ref("NonEmptyString"), "state": ref("TrainingState"), "progress": ref("FiniteNumber"),
        "current_loss": nullable(ref("FiniteNumber")), "adapter_name": nullable(ref("String")), "effective_seed": ref("DecimalU64"),
        "started_at": ref("String"), "elapsed_secs": ref("FiniteNumber"), "submitted_unix_ms": ref("NonNegativeInteger"),
        "finished_unix_ms": ref("NonNegativeInteger"), "job_type": ref("TrainingJobType"), "training_data": ref("TrainingDataProvenance"), "error": ref("String"),
        "post_eval_verdict": ref("String"), "gate_outcome": {"enum": ["promoted", "kept", "regression", "demoted", "inconclusive", "error"]},
        "post_eval_gate_evidence": array(ref("PostEvalGateEvidence")),
    }
    status_optional = ("effective_seed", "submitted_unix_ms", "finished_unix_ms", "job_type", "training_data", "error", "post_eval_verdict", "gate_outcome", "post_eval_gate_evidence")
    add_object("TrainingStatus", "TrainingStatus", status_fields, "Current or archived status for one training job.", optional=status_optional)
    add_definition("Vec_TrainingStatus", "Vec<TrainingStatus>", array(ref("TrainingStatus")), "All training statuses Kiln currently retains.")
    add_object("TrainingResponse", "TrainingResponse", {"job_id": ref("NonEmptyString"), "state": {"const": "queued"}, "effective_seed": ref("DecimalU64"), "message": ref("String")}, "Receipt returned after Kiln validates and queues a training request.")
    add_object("QueueStatusEntry", "QueueStatusEntry", {"job_id": ref("NonEmptyString"), "job_type": ref("TrainingJobType"), "adapter_name": ref("String"), "position": ref("PositiveInteger")}, "One queued training job and its first-in, first-out position.")
    add_object("QueueResponse", "QueueResponse", {"running": nullable(ref("TrainingStatus")), "queued": array(ref("QueueStatusEntry")), "completed": array(ref("TrainingStatus"))}, "Training jobs grouped into running, queued, and terminal collections.")
    add_object("TrainingLossSample", "crate::state::TrainingLossSample", {"epoch": ref("NonNegativeInteger"), "progress": ref("FiniteNumber"), "loss": ref("FiniteNumber"), "elapsed_secs": ref("FiniteNumber")}, "One retained sample from the live training-loss curve.")
    add_enum("TrainingKind", "kiln_train::checkpoint::TrainingKind", ["sft", "grpo", "opd", "capability-distillation"], "Training workflow recorded in a checkpoint.")
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
        "Summary of the newest checkpoint whose manifest is valid and can be resumed.",
        optional=("teacher_id", "teacher_identity_revision", "teacher_content_revision"),
    )
    detail_fields = {
        **status_fields,
        "epoch": nullable(ref("NonNegativeInteger")), "adapter_path": nullable(ref("String")), "auto_load": ref("Boolean"),
        "linked_eval_job_ids": array(ref("String")), "loss_history": array(ref("TrainingLossSample")), "train_receipt": nullable(ref("AnyJson")),
        "replay_request": nullable(ref("AnyJson")), "latest_checkpoint": nullable(ref("TrainingCheckpointSummary")),
        "checkpoint_error": nullable(ref("String")), "metadata_error": nullable(ref("String")),
    }
    add_object("TrainingJobDetail", "TrainingJobDetail", detail_fields, "Detailed job status with its loss curve, receipts, replay request, linked evals, and newest checkpoint.", optional=status_optional)
    add_definition(
        "CancelTrainingJobResponse",
        "CancelTrainingJobResponse",
        {
            "oneOf": [
                object_schema({"status": {"const": "cancelling"}, "job_id": ref("NonEmptyString"), "message": {"const": "stop requested — the trainer aborts at the next step boundary"}}),
                object_schema({"status": {"const": "cancelled"}, "job_id": ref("NonEmptyString")}),
            ]
        },
        "Cancellation result. Queued jobs stop immediately; running jobs stop cooperatively at the next training-step boundary and later appear as failed with a cancellation error.",
    )
    add_object("DeleteTrainingJobResponse", "DeleteTrainingJobResponse", {"job_id": ref("NonEmptyString"), "status": {"const": "deleted"}, "removed_archive_file": ref("Boolean")}, "Result of permanently deleting a terminal job from memory and, when present, its archive file. Active jobs must be cancelled first.")


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
    add_definition("FrontDoorRequest", "FrontDoorRequest", {"oneOf": front_door_variants}, "Tagged request accepted by the shared `/v1/train` endpoint for all supported training workflows.")
    add_object("FrontDoorResponse", "FrontDoorResponse", {"picked": {"enum": ["distill_refresh", "distill_merge", "distill_pump", "opd", "grpo", "sft"]}, "training": ref("TrainingResponse")}, "Training workflow selected by Kiln and the resulting queue receipt.")
    add_object(
        "CapacityRequest",
        "CapacityRequest",
        {"rollouts": ref("PositiveInteger"), "tokens_per_rollout": ref("PositiveInteger"), "top_k": ref("PositiveInteger"), "rank": ref("PositiveInteger"), "num_layers": ref("PositiveInteger"), "hidden_size": ref("PositiveInteger"), "initial_overlap_probe": nullable(ref("UnitInterval"))},
        "Inputs to the preflight capacity model. `num_layers` and `hidden_size` default to Qwen3.5-4B values when omitted, so send both for any other model.",
        optional=("num_layers", "hidden_size", "initial_overlap_probe"),
        open_input=True,
    )
    add_object("CapacityResponse", "CapacityResponse", {"bits_needed": ref("FiniteNumber"), "bits_storable_in_lora": ref("FiniteNumber"), "capacity_ratio": ref("FiniteNumber"), "expected_overlap_at_step_50": ref("UnitInterval"), "warnings": array(ref("String"))}, "Modeled capacity estimate and any preflight warnings. This is not a measured performance result.")
    add_object(
        "CompatibilityRow",
        "CompatibilityRow",
        {"teacher": ref("String"), "student": ref("String"), "domain": ref("String"), "predicted_initial_overlap": ref("UnitInterval"), "recommended_rank": ref("PositiveInteger"), "cold_start_epochs": nullable(ref("PositiveInteger")), "expected_gpu_hours": ref("FiniteNumber"), "expected_cost_usd": nullable(ref("FiniteNumber")), "validation_eval": ref("String"), "expected_eval_delta_points": ref("FiniteNumber")},
        "One built-in teacher, student, and domain estimate. Values are modeled planning inputs, not benchmark measurements or compatibility guarantees.",
    )
    add_object("CompatibilityResponse", "CompatibilityResponse", {"matches": array(ref("CompatibilityRow")), "note": nullable(ref("String"))}, "Filtered rows from Kiln's small, model-specific built-in estimate table.")
    add_object(
        "TierDefaults",
        "TierDefaults",
        {"tier": {"enum": ["laptop", "prosumer", "corporate"]}, "default_logit_source": ref("String"), "default_loss": ref("String"), "default_top_k": ref("PositiveInteger"), "lora_rank": ref("PositiveInteger"), "batch_size": ref("PositiveInteger"), "samples_per_prompt_default": ref("PositiveInteger"), "samples_per_prompt_data_multiplier": ref("PositiveInteger"), "max_rollout_tokens": ref("PositiveInteger"), "auto_checkpoint_cadence_steps": ref("PositiveInteger"), "cost_cap_default_usd": nullable(ref("FiniteNumber")), "cold_start_overlap_threshold": ref("UnitInterval"), "mixture_distillation_golden_fraction": ref("UnitInterval"), "eval_gate_required": ref("Boolean"), "notifications_channels": array(ref("String"))},
        "Built-in planning defaults for a coarse deployment tier. Tier names are presets, not hardware detection, measured performance classes, or device-specific tuning.",
    )
    add_object("TierDefaultsResponse", "TierDefaultsResponse", {"tier": {"enum": ["laptop", "prosumer", "corporate"]}, "defaults": ref("TierDefaults")}, "One requested preset and its built-in planning defaults.")
    add_object("TierDefaultsListResponse", "TierDefaultsListResponse", {"tiers": array(ref("TierDefaults"), min_items=3)}, "All built-in deployment-tier presets.")


def build_agent_types() -> None:
    add_enum("RunStatus", "crate::agent_runs::RunStatus", ["queued", "running", "completed", "failed", "aborted", "timed_out", "interrupted"], "Lifecycle state for an embedded agent run.")
    run_fields = {
        "id": ref("NonEmptyString"), "task": ref("String"), "cwd": ref("String"), "label": ref("String"), "status": ref("RunStatus"),
        "created_unix_ms": ref("NonNegativeInteger"), "queue_seq": ref("NonNegativeInteger"), "started_unix_ms": ref("NonNegativeInteger"),
        "finished_unix_ms": ref("NonNegativeInteger"), "num_turns": ref("NonNegativeInteger"), "num_tool_calls": ref("NonNegativeInteger"),
        "session_id": ref("String"), "session_path": ref("String"), "trace_indexed": ref("Boolean"), "last_assistant_text": ref("String"), "error": ref("String"),
    }
    add_object("AgentRunRecord", "AgentRunRecord", run_fields, "Stored metadata and outcome for one embedded agent run.", optional=("label", "started_unix_ms", "finished_unix_ms", "session_id", "session_path", "last_assistant_text", "error"))
    add_object(
        "CreateRunRequest",
        "CreateRunRequest",
        {"task": ref("NonEmptyString"), "cwd": nullable(ref("String")), "label": nullable(ref("String")), "tools": nullable(array(ref("String"))), "thinking_level": nullable({"enum": ["off", "minimal", "low", "medium", "high", "xhigh"]}), "timeout_secs": nullable({"type": "integer", "minimum": 10})},
        "Request to queue one embedded agent run.",
        optional=("cwd", "label", "tools", "thinking_level", "timeout_secs"),
        open_input=True,
    )
    add_object("MessageRequest", "MessageRequest", {"message": ref("NonEmptyString")}, "Steering or follow-up message for an active run.", open_input=True)
    add_object("AgentRunsStatusResponse", "AgentRunsStatusResponse", {"enabled": ref("Boolean"), "disabled_reason": nullable(ref("String")), "pi_available": ref("Boolean"), "pi_path": nullable(ref("String")), "max_concurrent_runs": ref("PositiveInteger"), "active_runs": ref("NonNegativeInteger"), "sessions_dir": ref("String")}, "Whether embedded runs are enabled, whether the local `pi` executable is available, and current concurrency.")
    add_object("AgentRunListResponse", "AgentRunListResponse", {"runs": array(ref("AgentRunRecord"))}, "Embedded agent runs Kiln currently retains.")
    add_object("AgentRunEvent", "AgentRunEvent", {"seq": ref("NonNegativeInteger"), "event": ref("AnyJson")}, "One ordered event emitted by `pi`.")
    add_object("AgentRunEventsResponse", "AgentRunEventsResponse", {"events": array(ref("AgentRunEvent")), "next_after": ref("NonNegativeInteger"), "status": ref("RunStatus"), "first_available_seq": nullable(ref("NonNegativeInteger")), "truncated": ref("Boolean")}, "Incremental event page with sequence information that lets clients detect truncated history.")
    add_object("AgentRunQueuedResponse", "AgentRunQueuedResponse", {"queued": {"const": True}}, "Acknowledgment that a steering or follow-up message was queued for delivery.")
    add_object("AgentRunAbortResponse", "AgentRunAbortResponse", {"aborting": {"const": True}}, "Acknowledgment that a cooperative abort was requested.")
    add_object("TerminalStatusResponse", "TerminalStatusResponse", {"enabled": ref("Boolean"), "disabled_reason": nullable(ref("String")), "pi_available": ref("Boolean"), "pi_path": nullable(ref("String")), "cwd": ref("String"), "session_active": ref("Boolean")}, "Embedded-terminal security status, local `pi` path, working directory, and session state.")
    add_object(
        "TraceOutcome",
        "TraceOutcome",
        {"ended_with_exit_0": nullable(ref("Boolean")), "user_edited_agent_files": array(ref("String")), "has_followup_attempt": nullable(ref("Boolean"))},
        "Outcome signals inferred from a `pi` session. These values are best-effort observations, not verified success criteria.",
    )
    add_object(
        "AgentTrace",
        "AgentTrace",
        {"id": ref("NonEmptyString"), "working_dir": ref("String"), "num_turns": ref("NonNegativeInteger"), "num_tool_calls": ref("NonNegativeInteger"), "outcome": ref("TraceOutcome"), "first_event_at": nullable(ref("Rfc3339Timestamp")), "last_event_at": nullable(ref("Rfc3339Timestamp")), "forked": ref("Boolean"), "parent_id": nullable(ref("String")), "tool_manifest_sha": nullable(ref("String")), "prompt_messages": array(ref("TrainingChatMessageOutput")), "trajectory": array(ref("TurnSegmentOutput"))},
        "Indexed `pi` session and its normalized training trajectory.",
        optional=("prompt_messages", "trajectory"),
    )
    add_object("AgentTracesListResponse", "AgentTracesListResponse", {"traces": array(ref("AgentTrace"))}, "All agent traces currently indexed by Kiln.")
    add_object("DiscoverRequest", "DiscoverRequest", {"path": nullable(ref("String"))}, "Optional `pi` sessions directory on the Kiln server.", optional=("path",), open_input=True)
    add_object("DiscoverResponse", "DiscoverResponse", {"indexed": ref("NonNegativeInteger"), "path": ref("String")}, "Number of traces indexed and the server-local directory scanned.")
    add_object(
        "JudgeDistillRequest",
        "JudgeDistillRequest",
        {"name": ref("NonEmptyString"), "teacher": ref("NonEmptyString"), "include_pi_share": ref("Boolean"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Request to distill a local judge from indexed agent traces.",
        optional=("name", "teacher", "include_pi_share", "config", "post_eval"),
        open_input=True,
    )
    add_object("JudgeDistillResponse", "JudgeDistillResponse", {"job_id": ref("NonEmptyString"), "state": {"const": "queued"}, "effective_seed": ref("DecimalU64"), "message": ref("String")}, "Queue receipt for a judge-distillation request.")
    add_object(
        "SelfImproveRequest",
        "SelfImproveRequest",
        {"agent": ref("NonEmptyString"), "judge": ref("NonEmptyString"), "crisp": ref("Boolean"), "config": ref("OpdConfig"), "post_eval": nullable(ref("PostEvalConfig"))},
        "Request to queue the phases of one agent self-improvement round.",
        optional=("agent", "judge", "crisp", "config", "post_eval"),
        open_input=True,
    )
    add_object("SelfImproveResponse", "SelfImproveResponse", {"job_ids": array(ref("NonEmptyString"), min_items=1), "effective_seeds": mapping(ref("DecimalU64")), "state": {"const": "queued"}, "message": ref("String")}, "Job IDs and seeds queued for one self-improvement round.")
    add_object(
        "JudgeDriftCheckRequest",
        "JudgeDriftCheckRequest",
        {"judge": ref("NonEmptyString"), "teacher": ref("NonEmptyString"), "sample_size": ref("PositiveInteger"), "agreement_threshold": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}},
        "Inputs accepted by the judge-drift route. After validation, the endpoint currently returns HTTP 501 and performs no drift check.",
        optional=("judge", "teacher", "sample_size", "agreement_threshold"),
        open_input=True,
        extra={"x-kiln-current-runtime-result": "http_501_not_implemented"},
    )


def build_recipe_types() -> None:
    add_definition(
        "PromptsSource",
        "PromptsSource",
        {"oneOf": [object_schema({"dataset": ref("NonEmptyString")}, open_input=True), object_schema({"prompts": array(ref("OpdPrompt"), min_items=1)}, open_input=True)]},
        "Either a registered dataset name or inline prompts for an OPD recipe step.",
    )
    add_definition(
        "ExamplesSource",
        "ExamplesSource",
        {"oneOf": [object_schema({"dataset": ref("NonEmptyString")}, open_input=True), object_schema({"examples": array(ref("SftExample"), min_items=1)}, open_input=True)]},
        "Either a registered dataset name or inline examples for an SFT recipe step.",
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
    add_definition("RecipeStep", "RecipeStep", {"oneOf": recipe_steps}, "One training or post-evaluation step in a recipe.")
    add_object("Recipe", "Recipe", {"name": ref("NonEmptyString"), "description": nullable(ref("String")), "steps": array(ref("RecipeStep"), min_items=1)}, "A named, inline sequence of training and post-evaluation steps.", optional=("description",), open_input=True)
    add_definition(
        "RecipeRunRequest",
        "RecipeRunRequest",
        {"oneOf": [object_schema({"recipe": ref("NonEmptyString"), "inputs": mapping(ref("AnyJson"))}, optional=("inputs",), open_input=True), object_schema({"body": ref("Recipe"), "inputs": mapping(ref("AnyJson"))}, optional=("inputs",), open_input=True)]},
        "Run either a built-in recipe by name or an inline recipe body, with optional placeholder values.",
    )
    add_object("RecipeRunResponse", "RecipeRunResponse", {"recipe": ref("NonEmptyString"), "job_ids": array(ref("NonEmptyString")), "effective_seeds": mapping(ref("DecimalU64")), "message": ref("String")}, "Jobs and effective seeds queued for one recipe run.")
    add_object("RecipeAdmissionDescriptor", "RecipeAdmissionDescriptor", {"supported": ref("Boolean"), "unavailable_reason": nullable(ref("String"))}, "Whether the current server can run a built-in recipe and, if not, why.")
    add_object("RecipeDescriptor", "RecipeDescriptor", {"name": ref("NonEmptyString"), "description": nullable(ref("String")), "num_steps": ref("NonNegativeInteger"), "admission": ref("RecipeAdmissionDescriptor")}, "Summary of one built-in recipe and its current availability.")
    add_object("RecipesListResponse", "RecipesListResponse", {"recipes": array(ref("RecipeDescriptor"))}, "All built-in recipes and whether the current server can run each one.")


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
    add_object("CorrectionRowInput", "CorrectionRow", input_fields, "Create or update one correction. Kiln can supply timestamps and training markers when they are omitted.", optional=("agent", "adapter", "original", "ideal", "truncated", "created_at", "trained_into", "trained_at"), open_input=True)
    add_object("CorrectionRow", "CorrectionRow", fields, "One stored correction returned by Kiln.", optional=("trained_into", "trained_at"))
    add_object("ListResponse", "ListResponse", {"corrections": array(ref("CorrectionRow"))}, "Corrections from the active set or training history.")
    add_object("MarkTrainedRequest", "MarkTrainedRequest", {"request_ids": array(ref("NonEmptyString"), min_items=1), "adapter": ref("NonEmptyString")}, "Mark selected corrections as used to train a named adapter.", open_input=True)
    add_object("MarkTrainedResponse", "MarkTrainedResponse", {"status": {"const": "marked"}, "marked": ref("NonNegativeInteger")}, "Number of corrections marked as trained.")
    add_object("DeleteCorrectionResponse", "DeleteCorrectionResponse", {"status": {"const": "deleted"}, "request_id": ref("NonEmptyString")}, "Confirmation that one correction was deleted.")
    add_object("ClearCorrectionsResponse", "ClearCorrectionsResponse", {"status": {"const": "cleared"}, "removed": ref("NonNegativeInteger")}, "Number of active corrections removed. Previously trained history remains.")


def build_library_types() -> None:
    add_object(
        "LibraryAdapterEntry",
        "LibraryAdapterEntry",
        {"id": ref("NonEmptyString"), "name": ref("NonEmptyString"), "source_kind": ref("String"), "description": nullable(ref("String")), "post_eval": mapping(ref("FiniteNumber")), "uploader": nullable(ref("String")), "size_bytes": nullable(ref("NonNegativeInteger"))},
        "One adapter entry in the configured library catalog.",
    )
    add_object("LibraryListResponse", "LibraryListResponse", {"backend": ref("String"), "adapters": array(ref("LibraryAdapterEntry")), "note": nullable(ref("String"))}, "Configured library URL and catalog response. The current endpoint returns an empty list because remote catalog access is not operational.")
    add_object("PublishPayload", "PublishPayload", {"description": nullable(ref("String")), "uploader": nullable(ref("String"))}, "Optional description and uploader label for a future library publication.", optional=("description", "uploader"), open_input=True)
    add_object(
        "PublishToLibraryResponse",
        "PublishToLibraryResponse",
        {"status": {"const": "ready_to_publish"}, "backend": ref("String"), "intended_id": ref("NonEmptyString"), "uploader": nullable(ref("String")), "description": nullable(ref("String")), "receipt_schema_version": ref("PositiveInteger"), "note": ref("String")},
        "Result of validating that a local adapter has the receipt required for publication. No remote upload occurs.",
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
    opd = {"prompts": [{"messages": [message]}], "teacher": "teacher-large@local", "config": opd_config_example()}
    grpo = {"groups": [{"messages": [message], "completions": [{"text": "4", "reward": 1.0}, {"text": "5", "reward": 0.0}]}], "config": {"optimizer": {"kind": "muon"}}}
    status = status_example()
    run = run_example()
    trace = {"id": "session-1", "working_dir": "/srv/project", "num_turns": 2, "num_tool_calls": 1, "outcome": {"ended_with_exit_0": True, "user_edited_agent_files": [], "has_followup_attempt": None}, "first_event_at": "2026-07-14T12:00:00Z", "last_event_at": "2026-07-14T12:01:00Z", "forked": False, "parent_id": None, "tool_manifest_sha": None, "prompt_messages": [message], "trajectory": [{"role": "assistant", "content": "Fixed.", "kind": "action"}]}
    correction_input = {"request_id": "chat-1", "agent": "pi", "adapter": None, "user": "2 + 2?", "original": "5", "ideal": "4", "truncated": False, "created_at": ""}
    correction = {**correction_input, "created_at": "2026-07-14T12:00:00Z"}
    tier = {"tier": "prosumer", "default_logit_source": "LocalTeacher", "default_loss": "teacher_top_k (K=32)", "default_top_k": 32, "lora_rank": 32, "batch_size": 16, "samples_per_prompt_default": 4, "samples_per_prompt_data_multiplier": 32, "max_rollout_tokens": 7168, "auto_checkpoint_cadence_steps": 10, "cost_cap_default_usd": 25.0, "cold_start_overlap_threshold": 0.5, "mixture_distillation_golden_fraction": 0.25, "eval_gate_required": True, "notifications_channels": ["desktop_tray", "email", "webhook"]}
    recipe = {"name": "quick-sft", "description": "Train one adapter", "steps": [{"kind": "sft", "name": "math-v1", "examples_from": {"examples": sft["examples"]}}]}
    openenv_inspection = {
        "identity": {
            "schema": "kiln.openenv-identity.v1",
            "client_profile": "openenv-python-sdk-v1",
            "base_url": "http://127.0.0.1:8000",
            "websocket_url": "ws://127.0.0.1:8000/ws",
            "openapi_version": "3.1.0",
            "environments": ["bandit"],
            "metadata": {
                "name": "bandit",
                "description": "Choose an arm.",
                "readme_content": None,
                "version": "1.0.0",
                "author": None,
                "documentation_url": None,
            },
            "schema_sha256": "sha256:" + "0" * 64,
        },
        "schema": {
            "action": {"type": "object", "required": ["arm"]},
            "observation": {"type": "object"},
            "state": {"type": "object"},
        },
    }
    openenv_request = {
        "kind": "train",
        "environment_urls": ["http://127.0.0.1:8000"],
        "adapter": "base",
        "groups": 8,
        "group_size": 4,
        "seed_start": 0,
        "reset_options": {},
        "max_steps": 8,
        "concurrency": 4,
        "max_action_tokens": 256,
        "temperature": 1.0,
        "thinking": False,
        "protocol_error_reward": -1.0,
        "max_recoverable_errors": 3,
        "capacity_wait_seconds": 300,
        "output_adapter": "bandit-agent",
        "training_config": {"lora_rank": 8},
        "auto_load": True,
    }
    openenv_status = {
        "schema": "kiln.openenv-run.v2",
        "run_id": "80a26e21-8451-4a64-8666-890c06fd80bd",
        "kind": "train",
        "state": "training_running",
        "request": openenv_request,
        "submitted_unix_ms": 1_700_000_000_000,
        "progress": {
            "groups_completed": 3,
            "groups_total": 8,
            "rollouts_completed": 12,
            "rollouts_total": 32,
        },
        "environments": [openenv_inspection["identity"]],
        "training_job_id": "grpo-openenv-1",
        "training": {
            "job_id": "grpo-openenv-1",
            "state": "running",
            "progress": 0.375,
            "current_loss": 0.42,
            "epoch": 1,
        },
    }
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
        "DistillPumpRequest": [{"name": "rust-v1", "teacher": "teacher-large@local", "mode": {"domain": "rust"}, "rollout_budget": 1000, "config": opd_config_example()}],
        "DistillRefreshRequest": [{"name": "assistant-v1", "new_data": {"dataset": "new-docs"}, "behavioural_teacher": "prior-self", "config": opd_config_example()}],
        "DistillSelfRequest": [{"name": "concise-v1", "mode": "conciseness", "config": opd_config_example()}],
        "FrontDoorRequest": [{"kind": "sft", **sft}],
        "FrontDoorResponse": [{"picked": "sft", "training": training_response_example()}],
        "GrpoRequest": [grpo],
        "JudgeDistillRequest": [{"name": "judge-pi-v1", "teacher": "teacher-large@local", "include_pi_share": False, "config": opd_config_example()}],
        "JudgeDistillResponse": [training_response_example("judge-1")],
        "JudgeDriftCheckRequest": [{"judge": "judge-pi-v1", "teacher": "teacher-large@local", "sample_size": 50, "agreement_threshold": 0.8}],
        "LibraryListResponse": [{"backend": "https://library.kiln.run", "adapters": [], "note": "Library backend not yet operational"}],
        "ListResponse": [{"corrections": [correction]}],
        "MarkTrainedRequest": [{"request_ids": ["chat-1"], "adapter": "math-v1"}],
        "MarkTrainedResponse": [{"status": "marked", "marked": 1}],
        "MessageRequest": [{"message": "Also run the focused test."}],
        "OpdRequest": [opd],
        "OpenEnvInspectRequest": [{"environment_urls": ["http://127.0.0.1:8000"]}],
        "OpenEnvInspectResponse": [{"schema": "kiln.openenv-inspection.v1", "environments": [openenv_inspection]}],
        "OpenEnvRunList": [{"schema": "kiln.openenv-run-list.v2", "runs": [openenv_status]}],
        "OpenEnvRunRequest": [openenv_request],
        "OpenEnvRunStatus": [openenv_status],
        "PublishPayload": [{"description": "Math adapter", "uploader": "local-user"}],
        "PublishToLibraryResponse": [{"status": "ready_to_publish", "backend": "https://library.kiln.run", "intended_id": "math-v1@2026-07-14", "uploader": "local-user", "description": "Math adapter", "receipt_schema_version": 1, "note": "Local validation passed; no remote upload occurred."}],
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
    build_openenv_types()
    build_preflight_types()
    build_agent_types()
    build_recipe_types()
    build_correction_types()
    build_library_types()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ericflo.github.io/kiln/contracts/kiln-control-plane-v1.schema.json",
        "title": "Kiln Training and Agent Control Plane API",
        "description": "Request and response shapes for training, distillation, preflight estimates, embedded agents, recipes, corrections, and the adapter-library routes. Start with the entrypoint for your HTTP operation, then open only the definitions it references. Unknown-field behavior is listed per shape. Preflight compatibility and capacity values are modeled estimates, and the library routes do not yet perform remote installs or uploads.",
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
