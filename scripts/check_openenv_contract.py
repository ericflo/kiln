#!/usr/bin/env python3
"""Validate the checked-in OpenEnv artifact contract without third-party packages."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from json_schema_subset import validate_instance as validate_schema_instance


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "kiln-openenv-v1.schema.json"
CONTROL_PLANE_SCHEMA_PATH = ROOT / "contracts" / "kiln-control-plane-v1.schema.json"


def validate_instance(value: object, schema: dict, root: dict) -> list[str]:
    control_plane = json.loads(CONTROL_PLANE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return validate_schema_instance(
        value,
        schema,
        root,
        registry={"kiln-control-plane-v1.schema.json": control_plane},
    )


def hash_value(character: str) -> str:
    return f"sha256:{character * 64}"


def fixtures() -> dict[str, dict]:
    inspection = {
        "identity": {
            "schema": "kiln.openenv-identity.v1",
            "client_profile": "openenv-http/1.x",
            "base_url": "http://127.0.0.1:8990",
            "websocket_url": "ws://127.0.0.1:8990/ws",
            "authentication": "none",
            "openapi_version": "1.0",
            "environments": ["counter"],
            "metadata": {
                "name": "CounterEnvironment",
                "description": "A stateful counter.",
                "readme_content": None,
                "version": "1",
                "author": None,
                "documentation_url": None,
            },
            "schema_sha256": hash_value("a"),
        },
        "schema": {
            "action": {
                "type": "object",
                "properties": {"amount": {"type": "integer"}},
                "required": ["amount"],
            },
            "observation": {"type": "object"},
            "state": {"type": "object"},
        },
    }
    record = {
        "group_index": 0,
        "candidate_index": 0,
        "environment_name": "CounterEnvironment",
        "environment_url": "http://127.0.0.1:8990",
        "seed": 7,
        "steps": 1,
        "episode_return": 2.0,
        "terminal_done": False,
        "termination": "max_steps",
        "recoverable_protocol_errors": 0,
        "capacity_retries": 1,
        "model_tokens": 4,
        "model_latency_ms": 1.5,
    }
    stats = {
        "mean_episode_return": 2.0,
        "min_episode_return": 2.0,
        "max_episode_return": 2.0,
        "done_count": 0,
        "max_steps_count": 1,
        "invalid_model_action_count": 0,
        "protocol_error_count": 0,
        "recoverable_protocol_error_count": 0,
        "capacity_retry_count": 1,
        "total_environment_steps": 1,
        "total_model_tokens": 4,
        "mean_model_latency_ms": 1.5,
    }
    summary = {
        "schema": "kiln.openenv-rollout-summary.v4",
        "kiln_url": "http://127.0.0.1:8420",
        "adapter": None,
        "adapter_label": "base",
        "environments": [inspection],
        "groups": 1,
        "group_size": 1,
        "rollout_count": 1,
        "seed_start": 7,
        "max_steps": 1,
        "concurrency": 1,
        "max_action_tokens": 32,
        "temperature": 1.0,
        "thinking": False,
        "protocol_error_reward": -1.0,
        "max_recoverable_errors": 3,
        "capacity_wait_seconds": 300,
        "reset_plan_sha256": hash_value("b"),
        "output_path": "openenv.rollouts.jsonl",
        "replay_output_path": "openenv.replay.json",
        "summary_output_path": "openenv.rollout-summary.json",
        "dataset_sha256": hash_value("c"),
        "dataset_bytes": 512,
        "replay_sha256": hash_value("d"),
        "replay_bytes": 1024,
        "stats": stats,
        "rollouts": [record],
        "training_contract": {
            "schema": "kiln.openenv-training-contract.v1",
            "effective_config": {
                "base_adapter": None,
                "output_name": "counter-agent",
                "auto_load": True,
                "behavior_policy": "no_importance_correction",
                "lora_rank": 8,
            },
        },
        "training_submission": {"job_id": "grpo-openenv-1"},
    }
    replay = {
        "schema": "kiln.openenv-replay.v1",
        "client_profile": "openenv-http/1.x",
        "dataset_sha256": hash_value("c"),
        "protocol_error_reward": -1.0,
        "max_steps": 1,
        "environments": [inspection],
        "groups": [
            {
                "group_index": 0,
                "environment_index": 0,
                "seed": 7,
                "reset_payload": {"seed": 7},
                "reset_observation": {
                    "observation": {"total": 0},
                    "reward": None,
                    "done": False,
                },
                "candidates": [
                    {
                        "candidate_index": 0,
                        "exchanges": [
                            {
                                "step_index": 0,
                                "action": {"amount": 2},
                                "result": {
                                    "kind": "observation",
                                    "observation": {
                                        "observation": {"total": 2},
                                        "reward": 2.0,
                                        "done": False,
                                    },
                                },
                            }
                        ],
                        "final_state": {"total": 2, "step_count": 1},
                        "episode_return": 2.0,
                        "terminal_done": False,
                        "termination": "max_steps",
                        "recoverable_protocol_errors": 0,
                        "capacity_retries": 1,
                        "model_tokens": 4,
                        "model_latency_ms": 1.5,
                    }
                ],
            }
        ],
    }
    verification = {
        "schema": "kiln.openenv-verification.v1",
        "summary_path": "openenv.rollout-summary.json",
        "dataset_path": "openenv.rollouts.jsonl",
        "replay_path": "openenv.replay.json",
        "dataset_sha256": hash_value("c"),
        "replay_sha256": hash_value("d"),
        "groups": 1,
        "rollouts": 1,
        "environment_exchanges": 1,
    }
    replay_run = {
        "schema": "kiln.openenv-replay-run.v1",
        "replay_sha256": hash_value("d"),
        "environments": 1,
        "groups": 1,
        "rollouts": 1,
        "environment_exchanges": 1,
        "capacity_retries": 0,
        "environment_prefix_only_rollouts": 0,
    }
    environment_evaluation = {
        "schema": "kiln.openenv-environment-evaluation.v1",
        "run_id": "80a26e21-8451-4a64-8666-890c06fd80bd",
        "config": {
            "groups": 20,
            "group_size": 1,
            "gate": {
                "min_mean_return": 0.5,
                "min_mean_improvement": 0.1,
            },
        },
        "seed_start": 8,
        "baseline": {
            "execution_provenance_sha256": hash_value("e"),
        },
        "candidate": {
            "adapter": "bandit-agent",
            "adapter_content_revision": hash_value("f"),
            "execution_provenance_sha256": hash_value("e"),
        },
        "baseline_summary_sha256": hash_value("1"),
        "candidate_summary_sha256": hash_value("2"),
        "evidence": {
            "policy_version": "paired_return_sign_test_v1",
            "paired_groups": 20,
            "paired_episodes": 20,
            "minimum_paired_groups": 20,
            "baseline_mean_return": 0.1,
            "candidate_mean_return": 0.8,
            "mean_return_improvement": 0.7,
            "improved_groups": 20,
            "regressed_groups": 0,
            "tied_groups": 0,
            "exact_sign_test_p_value": 0.0000019073486328125,
            "exact_sign_test_alpha": 0.05,
            "decision": "passed",
            "reason": "significant paired return improvement",
        },
        "outcome": "promoted",
        "verdict": "Environment promotion gate passed and the candidate was promoted.",
    }
    artifact_download = {
        "schema": "kiln.openenv-artifact-download.v1",
        "run_id": "80a26e21-8451-4a64-8666-890c06fd80bd",
        "kind": "environment_eval_receipt",
        "source_url": "/v1/openenv/runs/80a26e21-8451-4a64-8666-890c06fd80bd/artifacts/environment_eval_receipt",
        "output_path": "evidence/environment-evaluation/receipt.json",
        "sha256": hash_value("3"),
        "bytes": 2048,
    }
    return {
        "OpenEnvRolloutSummary": summary,
        "OpenEnvReplayManifest": replay,
        "VerificationReport": verification,
        "ReplayRunReport": replay_run,
        "EnvironmentEvaluationReceipt": environment_evaluation,
        "OpenEnvArtifactDownloadReceipt": artifact_download,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="also prove representative malformed fixtures are rejected",
    )
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    values = fixtures()
    for definition, value in values.items():
        errors = validate_instance(value, schema["$defs"][definition], schema)
        failures.extend(f"{definition}: {error}" for error in errors)

    if args.self_test:
        missing_reset_plan = copy.deepcopy(values["OpenEnvRolloutSummary"])
        del missing_reset_plan["reset_plan_sha256"]
        if not validate_instance(
            missing_reset_plan, schema["$defs"]["OpenEnvRolloutSummary"], schema
        ):
            failures.append("self-test: v4 summary without reset_plan_sha256 was accepted")

        missing_training_contract = copy.deepcopy(values["OpenEnvRolloutSummary"])
        del missing_training_contract["training_contract"]
        if not validate_instance(
            missing_training_contract,
            schema["$defs"]["OpenEnvRolloutSummary"],
            schema,
        ):
            failures.append(
                "self-test: training summary without its admitted contract was accepted"
            )

        wrong_training_contract = copy.deepcopy(values["OpenEnvRolloutSummary"])
        wrong_training_contract["training_contract"]["schema"] = "unknown"
        if not validate_instance(
            wrong_training_contract,
            schema["$defs"]["OpenEnvRolloutSummary"],
            schema,
        ):
            failures.append("self-test: unknown training contract schema was accepted")

        missing_digest = copy.deepcopy(values["OpenEnvRolloutSummary"])
        del missing_digest["replay_sha256"]
        if not validate_instance(
            missing_digest, schema["$defs"]["OpenEnvRolloutSummary"], schema
        ):
            failures.append("self-test: summary without replay_sha256 was accepted")

        non_object_action = copy.deepcopy(values["OpenEnvReplayManifest"])
        non_object_action["groups"][0]["candidates"][0]["exchanges"][0]["action"] = []
        if not validate_instance(
            non_object_action, schema["$defs"]["OpenEnvReplayManifest"], schema
        ):
            failures.append("self-test: replay with a non-object action was accepted")

        unbounded_recovery = copy.deepcopy(values["OpenEnvRolloutSummary"])
        unbounded_recovery["max_recoverable_errors"] = 65
        if not validate_instance(
            unbounded_recovery, schema["$defs"]["OpenEnvRolloutSummary"], schema
        ):
            failures.append("self-test: summary exceeding the recovery bound was accepted")

        missing_candidate_hash = copy.deepcopy(values["EnvironmentEvaluationReceipt"])
        del missing_candidate_hash["candidate_summary_sha256"]
        if not validate_instance(
            missing_candidate_hash,
            schema["$defs"]["EnvironmentEvaluationReceipt"],
            schema,
        ):
            failures.append(
                "self-test: environment evaluation without a candidate summary hash was accepted"
            )

        inconsistent_outcome = copy.deepcopy(values["EnvironmentEvaluationReceipt"])
        inconsistent_outcome["outcome"] = "rejected"
        if not validate_instance(
            inconsistent_outcome,
            schema["$defs"]["EnvironmentEvaluationReceipt"],
            schema,
        ):
            failures.append(
                "self-test: passed environment evidence with a rejected outcome was accepted"
            )

        external_artifact = copy.deepcopy(values["OpenEnvArtifactDownloadReceipt"])
        external_artifact["source_url"] = "https://example.com/receipt.json"
        if not validate_instance(
            external_artifact,
            schema["$defs"]["OpenEnvArtifactDownloadReceipt"],
            schema,
        ):
            failures.append("self-test: external artifact receipt URL was accepted")

    source = (ROOT / "crates" / "kiln-server" / "src" / "openenv_replay.rs").read_text(
        encoding="utf-8"
    )
    evaluation_source = (
        ROOT / "crates" / "kiln-server" / "src" / "openenv_evaluation.rs"
    ).read_text(encoding="utf-8")
    cli = (ROOT / "crates" / "kiln-server" / "src" / "openenv_cli.rs").read_text(
        encoding="utf-8"
    )
    api_source = (
        ROOT / "crates" / "kiln-server" / "src" / "api" / "openenv.rs"
    ).read_text(encoding="utf-8")
    client = (ROOT / "crates" / "kiln-openenv" / "src" / "client.rs").read_text(
        encoding="utf-8"
    )
    corpus_source = (
        ROOT / "crates" / "kiln-train" / "src" / "openenv_provenance.rs"
    ).read_text(encoding="utf-8")
    training_api_source = (
        ROOT / "crates" / "kiln-server" / "src" / "api" / "training.rs"
    ).read_text(encoding="utf-8")
    train_receipt_source = (
        ROOT / "crates" / "kiln-train" / "src" / "train_receipt.rs"
    ).read_text(encoding="utf-8")
    adapter_manifest_source = (
        ROOT / "crates" / "kiln-train" / "src" / "adapter_output.rs"
    ).read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "OPENENV_GUIDE.md").read_text(encoding="utf-8")
    http_api = json.loads(
        (ROOT / "contracts" / "kiln-http-api-v1.openapi.json").read_text(
            encoding="utf-8"
        )
    )
    control_plane = json.loads(CONTROL_PLANE_SCHEMA_PATH.read_text(encoding="utf-8"))
    openenv_training_data_schema = control_plane.get("$defs", {}).get(
        "OpenEnvTrainingDataProvenanceV1", {}
    )
    if (
        openenv_training_data_schema.get("properties", {})
        .get("schema", {})
        .get("const")
        != "kiln.openenv-training-data.v1"
    ):
        failures.append("control-plane schema is missing OpenEnv training-data v1")
    training_data_openenv = (
        control_plane.get("$defs", {})
        .get("TrainingDataProvenance", {})
        .get("properties", {})
        .get("openenv")
    )
    if training_data_openenv != {
        "$ref": "#/$defs/OpenEnvTrainingDataProvenanceV1"
    }:
        failures.append("training status does not expose typed OpenEnv corpus provenance")
    required_source_terms = [
        "kiln.openenv-replay.v1",
        "kiln.openenv-verification.v1",
        "kiln.openenv-replay-run.v1",
    ]
    for term in required_source_terms:
        if term not in source:
            failures.append(f"openenv_replay.rs is missing contract identifier {term}")
    for term in [
        "kiln.openenv-training-data.v1",
        "OpenEnvTrainingDataAccumulator",
        "group_plan_sha256",
        "completion.reward != provenance.episode_return",
        "mixes ordinary and OpenEnv groups",
        "changed name or schema within the training corpus",
    ]:
        if term not in corpus_source:
            failures.append(f"OpenEnv corpus validator is missing contract term {term}")
    if "OpenEnvTrainingDataAccumulator" not in training_api_source:
        failures.append("native GRPO admission is missing OpenEnv corpus validation")
    if "openenv: Option<crate::OpenEnvTrainingDataProvenanceV1>" not in train_receipt_source:
        failures.append("train receipt is missing typed OpenEnv corpus provenance")
    if "openenv_training_data" not in adapter_manifest_source:
        failures.append("adapter manifest is missing OpenEnv corpus lineage")
    for term in [
        "BoundedVecWriter",
        "encode_replay_with_limit",
        "bounded_artifact_metadata",
        "open_verified_artifact",
        "regular non-symlink file",
    ]:
        if term not in source:
            failures.append(f"openenv_replay.rs is missing bounded encoder term {term}")
    for term in [
        "status.artifacts",
        "openenv_artifact_integrity_failed",
        'HeaderValue::from_static("private, no-store")',
        "CONTENT_LENGTH",
        "ETAG",
    ]:
        if term not in api_source:
            failures.append(f"api/openenv.rs is missing manifest-bound artifact term {term}")
    for term in [
        '"/v1/openenv/training/preflight"',
        "preflight_training_inner",
        "materialize_openenv_grpo_config",
        "validate_openenv_training_contract",
        "capacity_reserved: false",
        "openenv_training_preflights_accepted",
        "openenv_training_preflights_rejected",
    ]:
        if term not in api_source:
            failures.append(f"api/openenv.rs is missing direct training-preflight term {term}")
    for term in [
        "OpenEnvCommands::Start",
        "OpenEnvCommands::Artifact",
        "MAX_OPENENV_RUN_REQUEST_BYTES",
        "kiln.openenv-artifact-download.v1",
        "persist_noclobber",
        "manifest_artifact",
        "ACCEPT_ENCODING",
        '"identity"',
        "Content-Length",
        "ETag",
        "private, no-store",
        "nosniff",
    ]:
        if term not in cli:
            failures.append(f"openenv_cli.rs is missing persisted lifecycle term {term}")
    for term in [
        "kiln.openenv-training-preflight.v1",
        "preflight_openenv_training(&options).await?",
        '"{}/v1/openenv/training/preflight"',
        "&preflight.effective_config",
        "!receipt.capacity_reserved",
    ]:
        if term not in cli:
            failures.append(f"openenv_cli.rs is missing direct training-preflight term {term}")
    preflight_operation = http_api["paths"].get(
        "/v1/openenv/training/preflight", {}
    ).get("post", {})
    preflight_request = (
        preflight_operation.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema")
    )
    preflight_response = (
        preflight_operation.get("responses", {})
        .get("200", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema")
    )
    if preflight_request != {
        "$ref": "#/components/schemas/OpenEnvTrainingPreflightRequest"
    }:
        failures.append("OpenEnv preflight OpenAPI request schema is missing")
    if preflight_response != {
        "$ref": "#/components/schemas/OpenEnvTrainingPreflightReceipt"
    }:
        failures.append("OpenEnv preflight OpenAPI receipt schema is missing")
    expected_artifact_kinds = {
        "dataset",
        "replay",
        "summary",
        "environment_eval_baseline_dataset",
        "environment_eval_baseline_replay",
        "environment_eval_baseline_summary",
        "environment_eval_candidate_dataset",
        "environment_eval_candidate_replay",
        "environment_eval_candidate_summary",
        "environment_eval_receipt",
    }
    artifact_operation = http_api["paths"][
        "/v1/openenv/runs/{run_id}/artifacts/{kind}"
    ]["get"]
    artifact_kind_parameter = next(
        parameter
        for parameter in artifact_operation["parameters"]
        if parameter["name"] == "kind"
    )
    if set(artifact_kind_parameter["schema"]["enum"]) != expected_artifact_kinds:
        failures.append("OpenEnv artifact OpenAPI kind enum is incomplete")
    artifact_headers = set(artifact_operation["responses"]["200"].get("headers", {}))
    if artifact_headers != {
        "Cache-Control",
        "Content-Disposition",
        "Content-Length",
        "ETag",
        "X-Content-Type-Options",
    }:
        failures.append("OpenEnv artifact OpenAPI integrity headers are incomplete")
    if "kiln.openenv-rollout-summary.v4" not in cli:
        failures.append("openenv_cli.rs is missing summary v4")
    for term in [
        "OPENENV_TRAINING_CONTRACT_SCHEMA_V1",
        "OpenEnvTrainingContract",
        "training_contract",
    ]:
        if term not in cli or term not in api_source:
            failures.append(f"OpenEnv runtime is missing shared training-contract term {term}")
    for term in [
        "MAX_OPENENV_RETAINED_BYTES",
        "MAX_OPENENV_RESET_OPTIONS_BYTES",
        "MAX_OPENENV_SUMMARY_BYTES",
        "OpenEnvRetainedByteBudget",
        "BoundedWriter",
        "charge_serialized",
        'serialized_len(&collection, "completed collection")',
        "try_collect::<Vec<_>>()",
    ]:
        if term not in cli:
            failures.append(f"openenv_cli.rs is missing bounded-collection term {term}")
    for term in [
        "keep_alive_while",
        "UnsolicitedApplicationMessage",
        "fail_closed",
    ]:
        if term not in client:
            failures.append(f"kiln-openenv client is missing session-lifecycle term {term}")
    if "keep_alive_while" not in cli:
        failures.append("openenv_cli.rs does not pump control frames during policy generation")
    for term in [
        "kiln.openenv-environment-evaluation.v1",
        "paired_return_sign_test_v1",
    ]:
        if term not in evaluation_source:
            failures.append(
                f"openenv_evaluation.rs is missing contract identifier {term}"
            )
    for command in [
        "kiln openenv tasks",
        "kiln openenv start",
        "kiln openenv artifact",
        "kiln openenv verify",
        "kiln openenv replay",
        "pumps Ping/Pong control frames",
        "poison the socket",
        "lock-step cannot resynchronize",
        "512 MiB aggregate retained-representation budget",
        "Only manifest-declared artifacts download; each request rechecks bytes and SHA-256.",
        "Exhaustion publishes no partial bundle",
        "POST /v1/openenv/training/preflight",
        "capacity_reserved: false",
        "kiln.openenv-training-contract.v1",
    ]:
        if command not in guide:
            failures.append(f"OpenEnv guide is missing {command!r}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(
        "OpenEnv training preflight, bounded collection, manifest-gated artifacts, session lifecycle, summary, replay, paired evaluation, verification, and live-replay contracts match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
