#!/usr/bin/env python3
"""Validate the checked-in OpenEnv artifact contract without third-party packages."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from json_schema_subset import validate_instance


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "kiln-openenv-v1.schema.json"


def hash_value(character: str) -> str:
    return f"sha256:{character * 64}"


def fixtures() -> dict[str, dict]:
    inspection = {
        "identity": {
            "schema": "kiln.openenv-identity.v1",
            "client_profile": "openenv-http/1.x",
            "base_url": "http://127.0.0.1:8990",
            "websocket_url": "ws://127.0.0.1:8990/ws",
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
        "schema": "kiln.openenv-rollout-summary.v2",
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
        "reset_options_sha256": hash_value("b"),
        "output_path": "openenv.rollouts.jsonl",
        "replay_output_path": "openenv.replay.json",
        "summary_output_path": "openenv.rollout-summary.json",
        "dataset_sha256": hash_value("c"),
        "dataset_bytes": 512,
        "replay_sha256": hash_value("d"),
        "replay_bytes": 1024,
        "stats": stats,
        "rollouts": [record],
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
    return {
        "OpenEnvRolloutSummary": summary,
        "OpenEnvReplayManifest": replay,
        "VerificationReport": verification,
        "ReplayRunReport": replay_run,
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

    source = (ROOT / "crates" / "kiln-server" / "src" / "openenv_replay.rs").read_text(
        encoding="utf-8"
    )
    cli = (ROOT / "crates" / "kiln-server" / "src" / "openenv_cli.rs").read_text(
        encoding="utf-8"
    )
    guide = (ROOT / "docs" / "OPENENV_GUIDE.md").read_text(encoding="utf-8")
    required_source_terms = [
        "kiln.openenv-replay.v1",
        "kiln.openenv-verification.v1",
        "kiln.openenv-replay-run.v1",
    ]
    for term in required_source_terms:
        if term not in source:
            failures.append(f"openenv_replay.rs is missing contract identifier {term}")
    if "kiln.openenv-rollout-summary.v2" not in cli:
        failures.append("openenv_cli.rs is missing summary v2")
    for command in ["kiln openenv verify", "kiln openenv replay"]:
        if command not in guide:
            failures.append(f"OpenEnv guide is missing {command}")

    if failures:
        raise SystemExit("\n".join(failures))
    print("OpenEnv summary, replay, verification, and live-replay contracts match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
