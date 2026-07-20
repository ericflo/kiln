from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "qualification" / "serve_rocm_decode_width_campaign.py"
QUALIFICATION = SCRIPT.parent
if str(QUALIFICATION) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION))
SPEC = importlib.util.spec_from_file_location("serve_rocm_decode_width_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def candidate_values(
    width: int,
    *,
    deterministic: float,
    sampled: float,
    tail: float = 100.0,
    memory: int = 10_000,
) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        source: 0 for _, source in campaign.SOURCE_METRICS
    }
    values.update(
        {
            "output_token_throughput_per_second": deterministic,
            "sampled_profile_output_token_throughput_per_second": sampled,
            "completion_token_count": 1,
            "itl_ms_p99": tail,
            "ttft_ms_p99": tail,
            "e2e_latency_ms_p99": tail,
            "batching_max_decode_batch": width,
            "batching_max_observed_batch_size": width,
            "sampled_profile_batching_max_observed_batch_size_end": width,
            "sampled_profile_rocm_w8_lm_head_max_batch_rows_end": width,
            "graph_measured_capture_success_count": 1,
            "graph_measured_replay_success_count": 1,
            "rocm_w8_lm_head_argmax_dispatch_count": 1,
            "rocm_w8_lm_head_argmax_row_count": 1,
            "sampled_profile_rocm_w8_lm_head_sample_dispatch_count": 1,
            "sampled_profile_rocm_w8_lm_head_sample_row_count": 1,
            "peak_gpu_memory_used_bytes": memory,
            "sampled_profile_request_failure_count": 0,
            "zero_token_response_count": 0,
        }
    )
    return values


def outcome(
    width: int,
    *,
    deterministic: float,
    sampled: float,
    tail: float = 100.0,
    memory: int = 10_000,
) -> campaign.CandidateOutcome:
    values = candidate_values(
        width,
        deterministic=deterministic,
        sampled=sampled,
        tail=tail,
        memory=memory,
    )
    return campaign.CandidateOutcome(
        width=width,
        values=values,
        correctness_reasons=campaign.correctness_reasons(width, values, None),
        performance_reasons=[],
    )


class DecodeWidthCampaignTests(unittest.TestCase):
    def test_candidate_configs_derive_active_and_staging_widths(self) -> None:
        expected = {4: (4, 8), 6: (4, 10), 8: (4, 12)}
        for width, (staging, active) in expected.items():
            with self.subTest(width=width):
                config = campaign.candidate_config(width)
                server = config["server"]
                self.assertEqual(server["max_decode_batch"], width)
                self.assertEqual(server["max_prefill_staging_slots"], staging)
                self.assertEqual(server["max_active_requests"], active)
                self.assertTrue(config["runtime"]["rocm_graphs_enabled"])
                self.assertEqual(config["batching"]["actor_cycle_idle_ms"], 50)

    def test_correctness_requires_exact_width_graphs_and_fused_sampling(self) -> None:
        values = candidate_values(4, deterministic=12.0, sampled=13.0)
        self.assertEqual(campaign.correctness_reasons(4, values, None), [])
        values["sampled_profile_rocm_w8_lm_head_max_batch_rows_end"] = 2
        reasons = campaign.correctness_reasons(4, values, None)
        self.assertTrue(any("max_batch_rows_end" in reason for reason in reasons))
        values["sampled_profile_rocm_w8_lm_head_max_batch_rows_end"] = 4
        values["rocm_w8_lm_head_argmax_row_count"] = 0
        reasons = campaign.correctness_reasons(4, values, None)
        self.assertTrue(any("below completion_token_count" in reason for reason in reasons))

    def test_selection_uses_minimum_throughput_ratio_and_narrow_tie(self) -> None:
        outcomes = [
            outcome(4, deterministic=10.0, sampled=10.0),
            outcome(6, deterministic=15.0, sampled=14.8),
            outcome(8, deterministic=15.2, sampled=15.0),
        ]
        selected = campaign.select_candidate(outcomes)
        self.assertEqual(selected.width, 6)
        self.assertTrue(selected.selected)
        self.assertAlmostEqual(selected.score_ratio, 1.48)

    def test_selection_rejects_tail_regression(self) -> None:
        outcomes = [
            outcome(4, deterministic=10.0, sampled=10.0),
            outcome(6, deterministic=14.0, sampled=14.0),
            outcome(8, deterministic=18.0, sampled=18.0, tail=130.0),
        ]
        selected = campaign.select_candidate(outcomes)
        self.assertEqual(selected.width, 6)
        self.assertTrue(outcomes[2].performance_reasons)

    def test_selection_fails_closed_on_correctness_failure(self) -> None:
        outcomes = [
            outcome(4, deterministic=10.0, sampled=10.0),
            outcome(6, deterministic=14.0, sampled=14.0),
            outcome(8, deterministic=18.0, sampled=18.0),
        ]
        outcomes[2].correctness_reasons.append("graph replay failed")
        with self.assertRaisesRegex(campaign.CampaignError, "not every"):
            campaign.select_candidate(outcomes)

    def test_result_metric_names_are_closed_and_unique(self) -> None:
        names = campaign.declared_metric_names()
        self.assertEqual(names, sorted(set(names)))
        outcomes = [
            outcome(4, deterministic=10.0, sampled=10.0),
            outcome(6, deterministic=14.0, sampled=14.0),
            outcome(8, deterministic=15.0, sampled=15.0),
        ]
        selected = campaign.select_candidate(outcomes)
        metrics = campaign.result_metrics(outcomes, selected, 1.5)
        self.assertEqual([metric["name"] for metric in metrics], names)

    def test_campaign_builds_once_and_reuses_the_exact_binary(self) -> None:
        binary = ROOT / "target" / "release" / "kiln"
        binary_hash = "sha256:" + "a" * 64

        def fake_execute(
            _model_path: Path,
            _seed: int,
            variant: str,
            *,
            built_binary: tuple[Path, str],
        ) -> tuple[list[dict[str, object]], None]:
            self.assertEqual(built_binary, (binary, binary_hash))
            width = int(variant.rsplit("-", 1)[1])
            values = candidate_values(
                width,
                deterministic=float(width * 10),
                sampled=float(width * 10),
            )
            return [
                {"name": name, "value": value}
                for name, value in values.items()
            ], None

        with mock.patch.object(
            campaign.mixed,
            "build_binary",
            return_value=(binary, binary_hash, 1.0),
        ) as build, mock.patch.object(
            campaign.mixed, "execute", side_effect=fake_execute
        ) as execute:
            outcomes, selected, build_seconds = campaign.run_campaign(ROOT, 17)

        build.assert_called_once()
        self.assertEqual(execute.call_count, len(campaign.CANDIDATE_WIDTHS))
        self.assertEqual([item.width for item in outcomes], [4, 6, 8])
        self.assertEqual(selected.width, 8)
        self.assertGreaterEqual(build_seconds, 0.0)

    def test_campaign_does_not_start_a_wider_arm_after_failure(self) -> None:
        binary = ROOT / "target" / "release" / "kiln"
        binary_hash = "sha256:" + "b" * 64

        def fake_execute(
            _model_path: Path,
            _seed: int,
            variant: str,
            *,
            built_binary: tuple[Path, str],
        ) -> tuple[list[dict[str, object]], str | None]:
            self.assertEqual(built_binary, (binary, binary_hash))
            width = int(variant.rsplit("-", 1)[1])
            values = candidate_values(
                width,
                deterministic=float(width),
                sampled=float(width),
            )
            metrics = [
                {"name": name, "value": value}
                for name, value in values.items()
            ]
            return metrics, "synthetic width failure" if width == 6 else None

        with mock.patch.object(
            campaign.mixed,
            "build_binary",
            return_value=(binary, binary_hash, 1.0),
        ), mock.patch.object(
            campaign.mixed, "execute", side_effect=fake_execute
        ) as execute:
            with self.assertRaisesRegex(
                campaign.CampaignRunError, "not every"
            ) as raised:
                campaign.run_campaign(ROOT, 17)

        self.assertEqual(execute.call_count, 2)
        self.assertEqual(
            [item.width for item in raised.exception.outcomes], [4, 6, 8]
        )
        self.assertTrue(raised.exception.outcomes[2].not_run)

    def test_workload_matches_script_contract(self) -> None:
        workload = json.loads(
            (
                ROOT
                / "qualification"
                / "workloads"
                / "serving-rocm-decode-width-campaign-v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], campaign.VARIANT_ID)
        self.assertEqual(variant["effective_config"], campaign.EFFECTIVE_CONFIG)
        case = variant["cases"][0]
        self.assertEqual(case["id"], campaign.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            campaign.declared_metric_names(),
        )


if __name__ == "__main__":
    unittest.main()
