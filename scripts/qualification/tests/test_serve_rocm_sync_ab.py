from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_rocm_sync_ab",
    QUALIFICATION_DIR / "serve_rocm_sync_ab.py",
)
assert SPEC is not None and SPEC.loader is not None
sync_ab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync_ab
SPEC.loader.exec_module(sync_ab)


def health_fixture(mode: str, *, offset: int = 0) -> dict:
    reasons = []
    for index, reason in enumerate(sync_ab.ROCM_SYNC_REASONS):
        reasons.append(
            {
                "reason": reason,
                "device_wait_count": offset + index,
                "stream_wait_count": offset + index * 2,
                "waited_ns": offset + index * 3,
                "skipped_count": offset + index * 4,
            }
        )
    return {
        "decode_runtime": {
            "accelerator_runtime": sync_ab.expected_policy(mode),
            "rocm_synchronization": {
                "active": True,
                "telemetry_available": True,
                "cleanup_quarantined": False,
                "telemetry_error": None,
                "total_device_wait_count": sum(
                    item["device_wait_count"] for item in reasons
                ),
                "total_stream_wait_count": sum(
                    item["stream_wait_count"] for item in reasons
                ),
                "total_waited_ns": sum(item["waited_ns"] for item in reasons),
                "total_skipped_count": sum(
                    item["skipped_count"] for item in reasons
                ),
                "reasons": reasons,
            },
        }
    }


def prometheus_fixture(snapshot: sync_ab.SyncSnapshot) -> str:
    lines = [
        f'kiln_rocm_synchronization_policy_info{{mode="{snapshot.mode}"}} 1',
        "kiln_rocm_cleanup_quarantined 0",
    ]
    for reason, stats in snapshot.reasons.items():
        lines.extend(
            (
                "kiln_rocm_synchronizations_total"
                f'{{reason="{reason}",scope="device"}} {stats.device_wait_count}',
                "kiln_rocm_synchronizations_total"
                f'{{reason="{reason}",scope="stream"}} {stats.stream_wait_count}',
                "kiln_rocm_synchronization_wait_seconds_total"
                f'{{reason="{reason}"}} {stats.waited_ns / 1_000_000_000.0}',
                "kiln_rocm_synchronization_skipped_total"
                f'{{reason="{reason}"}} {stats.skipped_count}',
            )
        )
    return "\n".join(lines) + "\n"


def stream_result(name: str, *, prompt_tokens: int = 20) -> sync_ab.mixed.StreamResult:
    token_times = [1.0 + index * 0.01 for index in range(sync_ab.MAX_TOKENS)]
    return sync_ab.mixed.StreamResult(
        name=name,
        marker=name,
        started=0.0,
        finished=1.5,
        semantic_times=[1.0],
        token_ready_times=token_times,
        token_queue_delays_ms=[0.0] * sync_ab.MAX_TOKENS,
        prompt_tokens=prompt_tokens,
        completion_tokens=sync_ab.MAX_TOKENS,
        usage_records=1,
        finish_reason="length",
        done=True,
        cancelled=False,
        error=None,
    )


def correctness_record() -> sync_ab.correctness.CompletionRecord:
    return sync_ab.correctness.CompletionRecord(
        scenario="synchronization-policy",
        semantic={"content": "same"},
        action_tokens=((0, 1, "sampled", -0.1),),
        sampled_logprobs=(-0.1,),
    )


class ServeRocmSynchronizationAbTests(unittest.TestCase):
    def test_reason_dimensions_exactly_match_rocm_core_contract(self) -> None:
        source = (ROOT / "crates/kiln-hip/src/lib.rs").read_text()
        start = source.index("pub const fn as_str(self) -> &'static str")
        end = source.index("const fn index(self)", start)
        labels = tuple(re.findall(r'=> "([a-z_]+)"', source[start:end]))
        self.assertEqual(labels, sync_ab.ROCM_SYNC_REASONS)
        self.assertEqual(len(labels), 23)

    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        workload = json.loads(
            (
                ROOT / "qualification/workloads/serving-rocm-sync-ab-v1.json"
            ).read_text()
        )
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], sync_ab.VARIANT_ID)
        self.assertEqual(variant["effective_config"], sync_ab.EFFECTIVE_CONFIG)
        case = variant["cases"][0]
        self.assertEqual(case["id"], sync_ab.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(sync_ab.METRIC_DEFINITIONS),
        )
        self.assertEqual(
            workload["comparison_policy"]["mode"], "same_environment_performance"
        )

    def test_policy_attestation_is_exact_and_source_aware(self) -> None:
        for mode in sync_ab.MODES:
            with self.subTest(mode=mode):
                policy = sync_ab.expected_policy(mode)
                self.assertEqual(
                    sync_ab.policy_attestation_failures(policy, mode, "health"), []
                )
                drifted = json.loads(json.dumps(policy))
                drifted["rocm_synchronization_mode"]["source"] = "default"
                self.assertTrue(
                    sync_ab.policy_attestation_failures(drifted, mode, "health")
                )

    def test_health_snapshot_closes_reason_and_total_contract(self) -> None:
        health = health_fixture("stream_ordered", offset=7)
        snapshot = sync_ab.synchronization_snapshot(health, "stream_ordered")
        self.assertEqual(tuple(snapshot.reasons), sync_ab.ROCM_SYNC_REASONS)
        self.assertEqual(snapshot.reasons["external_yield"].device_wait_count, 10)
        self.assertEqual(
            snapshot.device_wait_count,
            health["decode_runtime"]["rocm_synchronization"][
                "total_device_wait_count"
            ],
        )
        health["decode_runtime"]["rocm_synchronization"][
            "total_device_wait_count"
        ] += 1
        with self.assertRaisesRegex(sync_ab.SynchronizationQualificationError, "reason sum"):
            sync_ab.synchronization_snapshot(health, "stream_ordered")

        quarantined = health_fixture("stream_ordered")
        quarantined["decode_runtime"]["rocm_synchronization"][
            "cleanup_quarantined"
        ] = True
        with self.assertRaisesRegex(
            sync_ab.SynchronizationQualificationError, "cleanup quarantine"
        ):
            sync_ab.synchronization_snapshot(quarantined, "stream_ordered")

    def test_snapshot_delta_rejects_regression(self) -> None:
        before = sync_ab.synchronization_snapshot(
            health_fixture("legacy_host_barriers", offset=1),
            "legacy_host_barriers",
        )
        after = sync_ab.synchronization_snapshot(
            health_fixture("legacy_host_barriers", offset=4),
            "legacy_host_barriers",
        )
        delta = sync_ab.synchronization_delta(before, after)
        self.assertEqual(delta.reasons["external_yield"], sync_ab.ReasonStats(3, 3, 3, 3))
        with self.assertRaisesRegex(sync_ab.SynchronizationQualificationError, "regressed"):
            sync_ab.synchronization_delta(after, before)

    def test_prometheus_snapshot_reconciles_every_reason(self) -> None:
        health = sync_ab.synchronization_snapshot(
            health_fixture("stream_ordered", offset=2), "stream_ordered"
        )
        prometheus = sync_ab.prometheus_sync_snapshot(
            prometheus_fixture(health), "stream_ordered"
        )
        self.assertEqual(
            sync_ab.prometheus_attestation_failures(health, prometheus), []
        )
        drifted = dict(prometheus.reasons)
        drifted["external_yield"] = sync_ab.ReasonStats(999, 0, 0, 0)
        failures = sync_ab.prometheus_attestation_failures(
            health,
            sync_ab.SyncSnapshot(mode="stream_ordered", reasons=drifted),
        )
        self.assertTrue(any("external_yield" in failure for failure in failures))

    def test_prometheus_snapshot_rejects_missing_reason_and_nonintegral_count(self) -> None:
        snapshot = sync_ab.synchronization_snapshot(
            health_fixture("legacy_host_barriers"), "legacy_host_barriers"
        )
        missing = "\n".join(
            line
            for line in prometheus_fixture(snapshot).splitlines()
            if 'reason="global_state_mutation"' not in line
        )
        with self.assertRaisesRegex(sync_ab.SynchronizationQualificationError, "dimensions"):
            sync_ab.prometheus_sync_snapshot(missing, "legacy_host_barriers")
        malformed = prometheus_fixture(snapshot).replace(
            'reason="external_yield",scope="device"} 3',
            'reason="external_yield",scope="device"} 3.5',
        )
        with self.assertRaisesRegex(sync_ab.SynchronizationQualificationError, "exact integer"):
            sync_ab.prometheus_sync_snapshot(malformed, "legacy_host_barriers")
        quarantined = prometheus_fixture(snapshot).replace(
            "kiln_rocm_cleanup_quarantined 0",
            "kiln_rocm_cleanup_quarantined 1",
        )
        with self.assertRaisesRegex(
            sync_ab.SynchronizationQualificationError, "must be present exactly once"
        ):
            sync_ab.prometheus_sync_snapshot(quarantined, "legacy_host_barriers")

    def test_output_contract_compares_request_and_token_identity(self) -> None:
        results = tuple(
            stream_result(f"request-{index}", prompt_tokens=10 + index)
            for index in range(sum(len(words) for _, words in sync_ab.WAVES))
        )
        empty_sync = sync_ab.SyncSnapshot(
            mode="legacy_host_barriers",
            reasons={reason: sync_ab.ReasonStats(0, 0, 0, 0) for reason in sync_ab.ROCM_SYNC_REASONS},
        )
        base = sync_ab.ArmRun(
            mode="legacy_host_barriers",
            results=results,
            correctness_record=correctness_record(),
            elapsed_seconds=1.0,
            sync_delta=empty_sync,
            peak_memory_bytes=1,
            memory_sample_count=1,
            memory_sampler_error_count=0,
            policy_failures=(),
            prometheus_failures=(),
            device_fault_count=0,
            graph_activity_count=0,
            mutation_event_count=0,
            shutdown=sync_ab.mixed.ShutdownOutcome(0, False, 0.0),
            snapshot_residue=(),
        )
        candidate = dataclasses_replace(
            base,
            mode="stream_ordered",
            sync_delta=sync_ab.SyncSnapshot(
                mode="stream_ordered", reasons=empty_sync.reasons
            ),
        )
        self.assertEqual(sync_ab.output_contract_mismatches(base, candidate), [])
        drifted_results = list(results)
        drifted_results[0] = stream_result("request-0", prompt_tokens=999)
        candidate = dataclasses_replace(candidate, results=tuple(drifted_results))
        self.assertTrue(sync_ab.output_contract_mismatches(base, candidate))

    def test_correctness_metrics_separate_token_and_logprob_drift(self) -> None:
        results = tuple(
            stream_result(f"request-{index}")
            for index in range(sum(len(words) for _, words in sync_ab.WAVES))
        )
        empty_sync = sync_ab.SyncSnapshot(
            mode="legacy_host_barriers",
            reasons={
                reason: sync_ab.ReasonStats(0, 0, 0, 0)
                for reason in sync_ab.ROCM_SYNC_REASONS
            },
        )
        base = sync_ab.ArmRun(
            mode="legacy_host_barriers",
            results=results,
            correctness_record=correctness_record(),
            elapsed_seconds=1.0,
            sync_delta=empty_sync,
            peak_memory_bytes=1,
            memory_sample_count=1,
            memory_sampler_error_count=0,
            policy_failures=(),
            prometheus_failures=(),
            device_fault_count=0,
            graph_activity_count=0,
            mutation_event_count=0,
            shutdown=sync_ab.mixed.ShutdownOutcome(0, False, 0.0),
            snapshot_residue=(),
        )
        candidate = dataclasses_replace(
            base,
            mode="stream_ordered",
            sync_delta=sync_ab.SyncSnapshot(
                mode="stream_ordered", reasons=empty_sync.reasons
            ),
        )
        matching = sync_ab.correctness_metric_values(base, candidate)
        self.assertEqual(matching["correctness_action_token_count"], 1)
        self.assertTrue(
            all(value == 0 for name, value in matching.items() if "mismatch" in name)
        )

        token_drift = sync_ab.correctness.CompletionRecord(
            scenario="synchronization-policy",
            semantic={"content": "different"},
            action_tokens=((0, 2, "sampled", -0.1),),
            sampled_logprobs=(-0.1,),
        )
        drift = sync_ab.correctness_metric_values(
            base, dataclasses_replace(candidate, correctness_record=token_drift)
        )
        self.assertEqual(drift["correctness_output_mismatch_count"], 1)
        self.assertEqual(drift["correctness_token_id_mismatch_count"], 1)

        logprob_drift = sync_ab.correctness.CompletionRecord(
            scenario="synchronization-policy",
            semantic={"content": "different"},
            action_tokens=((0, 1, "sampled", -0.2),),
            sampled_logprobs=(-0.2,),
        )
        drift = sync_ab.correctness_metric_values(
            base, dataclasses_replace(candidate, correctness_record=logprob_drift)
        )
        self.assertEqual(drift["correctness_behavior_logprob_mismatch_count"], 1)

    def test_metric_contract_is_closed_finite_nonnegative_and_sorted(self) -> None:
        values = {name: 0 for name in sync_ab.METRIC_DEFINITIONS}
        metrics = sync_ab.metrics_from_values(values)
        self.assertEqual(
            [metric["name"] for metric in metrics], sorted(sync_ab.METRIC_DEFINITIONS)
        )
        for invalid in (-1, math.inf, math.nan, True):
            malformed = dict(values)
            malformed["legacy_request_failure_count"] = invalid
            with self.assertRaises(sync_ab.SynchronizationQualificationError):
                sync_ab.metrics_from_values(malformed)


def dataclasses_replace(value, **changes):
    return sync_ab.dataclasses.replace(value, **changes)


if __name__ == "__main__":
    unittest.main()
