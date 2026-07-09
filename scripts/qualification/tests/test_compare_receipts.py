from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_compare_receipts", QUALIFICATION_DIR / "compare_receipts.py"
)
assert SPEC is not None and SPEC.loader is not None
compare_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare_module
SPEC.loader.exec_module(compare_module)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
COMMIT_A = "d" * 40
COMMIT_B = "e" * 40


def metric_rule(*, metric_class: str) -> dict:
    if metric_class == "correctness":
        return {
            "scope": "result",
            "result_id": "main-case",
            "metric": "case_pass",
            "metric_class": "correctness",
            "unit": "bool",
            "aggregation": "exact",
            "lower_is_better": False,
            "operator": "equal",
            "absolute_tolerance": 0,
            "relative_tolerance": 0,
            "required": True,
        }
    return {
        "scope": "result",
        "result_id": "main-case",
        "metric": "throughput_tps",
        "metric_class": "performance",
        "unit": "tokens/s",
        "aggregation": "rate",
        "lower_is_better": False,
        "operator": "not_less",
        "absolute_tolerance": 0,
        "relative_tolerance": 0.1,
        "required": True,
    }


def workload_case(*, seeded: bool, runner_produced: bool) -> dict:
    command = (
        ["qualification-case", "--model", "${model_path}", "--seed", "${seed}"]
        if seeded
        else ["qualification-case"]
    )
    return {
        "id": "main-case",
        "description": "Run one deterministic qualification case.",
        "required": True,
        "command": command,
        "working_directory": ".",
        "environment": {},
        "timeout_seconds": 60,
        "expected_exit_codes": [0],
        "output_assertions": [],
        "result_protocol": {
            "format": "qualification-case-result-v1",
            "producer": "runner" if runner_produced else "command",
            "path_environment_variable": "KILN_QUALIFICATION_CASE_RESULT",
            "declared_metrics": ["case_pass"] if runner_produced else ["throughput_tps"],
        },
    }


def variant(
    variant_id: str,
    backend: str,
    *,
    seeded: bool,
    runner_produced: bool,
    effective_config: dict,
) -> dict:
    return {
        "id": variant_id,
        "description": f"Run {variant_id} on {backend}.",
        "backend": backend,
        "device_requirement": "required",
        "skip_policy": "fail",
        "effective_config": effective_config,
        "cases": [workload_case(seeded=seeded, runner_produced=runner_produced)],
    }


def performance_manifest(mode: str = "same_environment_performance") -> dict:
    if mode == "same_environment_performance":
        variants = [
            variant(
                "default",
                "rocm",
                seeded=True,
                runner_produced=False,
                effective_config={"autoscale": {"enabled": True}, "graphs": {"enabled": False}},
            )
        ]
        variant_pairs: list[dict] = []
    else:
        variants = [
            variant(
                "graphs-off",
                "rocm",
                seeded=True,
                runner_produced=False,
                effective_config={"autoscale": {"enabled": True}, "graphs": {"enabled": False}},
            ),
            variant(
                "graphs-on",
                "rocm",
                seeded=True,
                runner_produced=False,
                effective_config={"autoscale": {"enabled": True}, "graphs": {"enabled": True}},
            ),
        ]
        variant_pairs = [
            {
                "baseline_variant_id": "graphs-off",
                "candidate_variant_id": "graphs-on",
                "allowed_effective_config_differences": ["graphs.enabled"],
            }
        ]
    return {
        "schema_version": 1,
        "workload_id": "performance-test-v1",
        "kind": "performance",
        "description": "Measure a deterministic local performance fixture.",
        "determinism": {
            "seed": 7,
            "seed_delivery": "argv",
            "repetitions": 1,
            "case_order": "declared",
            "max_parallel_cases": 1,
            "network_access": "forbidden",
        },
        "variables": [],
        "variants": variants,
        "comparison_policy": {
            "mode": mode,
            "variant_pairs": variant_pairs,
            "backend_pairs": [],
            "metric_rules": [metric_rule(metric_class="performance")],
        },
    }


def correctness_manifest() -> dict:
    return {
        "schema_version": 1,
        "workload_id": "correctness-test-v1",
        "kind": "correctness",
        "description": "Compare one deterministic correctness metric across backends.",
        "determinism": {
            "seed": 7,
            "seed_delivery": "fixed_fixture",
            "repetitions": 1,
            "case_order": "declared",
            "max_parallel_cases": 1,
            "network_access": "forbidden",
        },
        "variables": [],
        "variants": [
            variant("rocm", "rocm", seeded=False, runner_produced=True, effective_config={}),
            variant("vulkan", "vulkan", seeded=False, runner_produced=True, effective_config={}),
        ],
        "comparison_policy": {
            "mode": "cross_backend_correctness",
            "variant_pairs": [],
            "backend_pairs": [
                {
                    "backend_a": "rocm",
                    "variant_a_id": "rocm",
                    "backend_b": "vulkan",
                    "variant_b_id": "vulkan",
                    "allowed_environment_differences": [
                        "compiler.hipcc",
                        "compiler.shader",
                        "device.architecture",
                        "device.driver",
                        "runtime.rocm",
                        "runtime.vulkan",
                    ],
                }
            ],
            "metric_rules": [metric_rule(metric_class="correctness")],
        },
    }


def commit_manifest(root: Path, value: dict) -> compare_module.LoadedWorkload:
    path = root / "qualification/workloads/test-workload.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", str(path.relative_to(root))], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Qualification Tests",
            "-c",
            "user.email=qualification@example.invalid",
            "commit",
            "-qm",
            "Add workload",
        ],
        cwd=root,
        check=True,
    )
    return compare_module.load_committed_workload(path, root=root)


def environment(backend: str) -> dict:
    return {
        "host_id": "strix-halo",
        "os": {
            "name": "EndeavourOS",
            "version": "rolling",
            "kernel": "7.0.14",
            "architecture": "x86_64",
        },
        "device": {
            "name": "AMD Radeon 8060S",
            "architecture": "gfx1151" if backend == "rocm" else "strix_halo",
            "memory_bytes": 103079215104,
            "unified_memory": True,
            "driver": "amdgpu" if backend == "rocm" else "radv",
        },
        "runtime": {backend: "1.0"},
        "compiler": {
            "rustc": "1.94.1",
            **({"hipcc": "1.0"} if backend == "rocm" else {"shader": "1.0"}),
        },
    }


def receipt(
    receipt_id: str,
    manifest: compare_module.LoadedWorkload,
    *,
    variant_id: str,
    backend: str,
    throughput: float = 10.0,
    case_pass: float = 1,
    config: dict | None = None,
) -> dict:
    kind = manifest.value["kind"]
    selected_variant = next(item for item in manifest.value["variants"] if item["id"] == variant_id)
    result_metrics = [
        {
            "name": "throughput_tps",
            "value": throughput,
            "unit": "tokens/s",
            "aggregation": "rate",
            "lower_is_better": False,
        }
    ] if kind == "performance" else [
        {
            "name": "case_pass",
            "value": case_pass,
            "unit": "bool",
            "aggregation": "exact",
            "lower_is_better": False,
        }
    ]
    return {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "created_at_utc": "2026-07-09T20:00:02Z",
        "source": {
            "tree_hash_format": "kiln-source-tree-v1",
            "tree_hash": HASH_A,
            "git_commit": COMMIT_A,
            "git_worktree_clean": True,
        },
        "qualification": {
            "kind": kind,
            "backend": backend,
            "profile": manifest.value["workload_id"],
            "verdict": "passed",
            "started_at_utc": "2026-07-09T20:00:00Z",
            "finished_at_utc": "2026-07-09T20:00:02Z",
            "duration_seconds": 2.0,
            "command": ["python3", "scripts/qualification/run.py", variant_id],
        },
        "environment": environment(backend),
        "model": {
            "id": "Qwen/Qwen3.5-4B",
            "path": "/models/Qwen3.5-4B",
            "weight_files": [
                {"path": "model-00001.safetensors", "sha256": HASH_B, "bytes": 1024},
                {"path": "model-00002.safetensors", "sha256": HASH_C, "bytes": 2048},
            ],
            "config_hash": HASH_A,
            "tokenizer_hash": HASH_B,
            "chat_template_hash": HASH_C,
        },
        "workload": {
            "id": manifest.value["workload_id"],
            "sha256": manifest.sha256,
            "seed": manifest.value["determinism"]["seed"],
            "parameters": {"variant_id": variant_id},
        },
        "effective_config": config if config is not None else copy.deepcopy(selected_variant["effective_config"]),
        "results": [
            {
                "id": "main-case",
                "required": True,
                "status": "passed",
                "duration_seconds": 2.0,
                "metrics": result_metrics,
                "details": None,
            }
        ],
        "metrics": [],
        "artifacts": [],
        "unsupported": [],
        "notes": [],
    }


def write_receipt(root: Path, name: str, value: dict) -> compare_module.LoadedReceipt:
    path = root / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return compare_module.load_validated_receipt(path)


class ReceiptComparisonTests(unittest.TestCase):
    def test_same_environment_performance_uses_only_declared_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline = write_receipt(
                root, "baseline.json", receipt("baseline-run", manifest, variant_id="default", backend="rocm")
            )
            candidate_value = receipt(
                "candidate-run", manifest, variant_id="default", backend="rocm", throughput=9.5
            )
            candidate_value["source"]["git_commit"] = COMMIT_B
            candidate_value["model"]["path"] = "/other/host/model"
            candidate = write_receipt(root, "candidate.json", candidate_value)

            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertTrue(report["ok"])
        self.assertTrue(report["compatible"])
        self.assertEqual(len(report["metric_deltas"]), 1)
        self.assertEqual(
            report["metric_deltas"][0]["scope"],
            "results/main-case/metrics/throughput_tps",
        )
        self.assertEqual(report["metric_deltas"][0]["status"], "passed")

    def test_required_performance_rule_failure_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline = write_receipt(
                root, "baseline.json", receipt("baseline-run", manifest, variant_id="default", backend="rocm")
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt(
                    "candidate-run", manifest, variant_id="default", backend="rocm", throughput=8.0
                ),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertTrue(report["compatible"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["metric_evaluations"][0]["status"], "failed")

    def test_immutable_identity_differences_are_rejected_before_deltas(self) -> None:
        mutations = {
            "source": lambda value: value["source"].__setitem__("tree_hash", HASH_B),
            "model": lambda value: value["model"].__setitem__("tokenizer_hash", HASH_C),
            "profile": lambda value: value["qualification"].__setitem__("profile", "other-profile"),
            "backend_environment": lambda value: value["environment"]["runtime"].__setitem__(
                "rocm", "2.0"
            ),
            "effective_config": lambda value: value["effective_config"]["graphs"].__setitem__(
                "enabled", True
            ),
            "metric_identity": lambda value: value["results"][0]["metrics"][0].__setitem__(
                "unit", "req/s"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline = write_receipt(
                root, "baseline.json", receipt("baseline-run", manifest, variant_id="default", backend="rocm")
            )
            for group, mutate in mutations.items():
                with self.subTest(group=group):
                    value = receipt(f"candidate-{group}", manifest, variant_id="default", backend="rocm")
                    mutate(value)
                    candidate = write_receipt(root, f"candidate-{group}.json", value)
                    report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)
                    self.assertFalse(report["compatible"])
                    self.assertIn(f"incompatible {group} identity", report["errors"])
                    self.assertEqual(report["metric_deltas"], [])

    def test_declared_ab_pair_allows_only_named_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest("declared_ab_variants"))
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt(
                    "baseline-run",
                    manifest,
                    variant_id="graphs-off",
                    backend="rocm",
                ),
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt(
                    "candidate-run",
                    manifest,
                    variant_id="graphs-on",
                    backend="rocm",
                    throughput=11.0,
                ),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertTrue(report["ok"])
        self.assertEqual(
            [item["compatibility"] for item in report["compatibility"]["allowed_differences"]],
            ["workload_variant", "effective_config"],
        )

    def test_ab_pair_is_directional_and_rejects_other_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest("declared_ab_variants"))
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt(
                    "baseline-run",
                    manifest,
                    variant_id="graphs-on",
                    backend="rocm",
                ),
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt(
                    "candidate-run",
                    manifest,
                    variant_id="graphs-off",
                    backend="rocm",
                ),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertFalse(report["compatible"])
        self.assertTrue(any("directional A/B" in error for error in report["errors"]))
        self.assertEqual(report["metric_deltas"], [])

    def test_ab_uses_permissions_from_only_the_selected_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = performance_manifest("declared_ab_variants")
            both = copy.deepcopy(value["variants"][1])
            both["id"] = "graphs-on-autoscale-off"
            both["effective_config"]["autoscale"]["enabled"] = False
            value["variants"].append(both)
            value["comparison_policy"]["variant_pairs"].append(
                {
                    "baseline_variant_id": "graphs-off",
                    "candidate_variant_id": "graphs-on-autoscale-off",
                    "allowed_effective_config_differences": [
                        "autoscale.enabled",
                        "graphs.enabled",
                    ],
                }
            )
            manifest = commit_manifest(root, value)
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt(
                    "baseline-run",
                    manifest,
                    variant_id="graphs-off",
                    backend="rocm",
                ),
            )
            wrong_pair_candidate = write_receipt(
                root,
                "wrong-pair.json",
                receipt(
                    "wrong-pair-run",
                    manifest,
                    variant_id="graphs-on",
                    backend="rocm",
                    config={"autoscale": {"enabled": False}, "graphs": {"enabled": True}},
                ),
            )
            wrong_report = compare_module.compare_receipts(
                baseline, wrong_pair_candidate, manifest=manifest
            )
            both_candidate = write_receipt(
                root,
                "both.json",
                receipt(
                    "both-run",
                    manifest,
                    variant_id="graphs-on-autoscale-off",
                    backend="rocm",
                ),
            )
            both_report = compare_module.compare_receipts(
                baseline, both_candidate, manifest=manifest
            )

        self.assertFalse(wrong_report["compatible"])
        self.assertIn("incompatible effective_config identity", wrong_report["errors"])
        self.assertTrue(both_report["ok"])
        self.assertEqual(
            [
                item.get("path")
                for item in both_report["compatibility"]["allowed_differences"]
                if item["compatibility"] == "effective_config"
            ],
            ["autoscale.enabled", "graphs.enabled"],
        )

    def test_manifest_rejects_unused_config_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = performance_manifest("declared_ab_variants")
            value["variants"][1]["effective_config"] = copy.deepcopy(
                value["variants"][0]["effective_config"]
            )
            with self.assertRaisesRegex(compare_module.ComparisonError, "exactly equal"):
                commit_manifest(root, value)

    def test_ab_config_permission_must_name_an_exact_scalar_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = performance_manifest("declared_ab_variants")
            value["comparison_policy"]["variant_pairs"][0][
                "allowed_effective_config_differences"
            ] = ["graphs"]
            with self.assertRaisesRegex(compare_module.ComparisonError, "exactly equal"):
                commit_manifest(root, value)

    def test_cross_backend_correctness_uses_exact_endpoint_pair_and_no_perf_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, correctness_manifest())
            baseline = write_receipt(
                root, "baseline.json", receipt("baseline-run", manifest, variant_id="rocm", backend="rocm")
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt("candidate-run", manifest, variant_id="vulkan", backend="vulkan"),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertTrue(report["ok"])
        self.assertEqual(report["mode"], "cross_backend_correctness")
        self.assertEqual(report["metric_evaluations"][0]["status"], "passed")
        self.assertEqual(report["metric_deltas"], [])

    def test_cross_backend_correctness_metric_failure_is_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, correctness_manifest())
            baseline = write_receipt(
                root, "baseline.json", receipt("baseline-run", manifest, variant_id="rocm", backend="rocm")
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt(
                    "candidate-run",
                    manifest,
                    variant_id="vulkan",
                    backend="vulkan",
                    case_pass=0,
                ),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertTrue(report["compatible"])
        self.assertFalse(report["ok"])
        self.assertTrue(any("case_pass must be numeric 1" in error for error in report["errors"]))
        self.assertEqual(report["metric_evaluations"], [])
        self.assertEqual(report["metric_deltas"], [])

    def test_equal_but_incorrect_runner_metrics_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, correctness_manifest())
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt("baseline-run", manifest, variant_id="rocm", backend="rocm", case_pass=0),
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt(
                    "candidate-run",
                    manifest,
                    variant_id="vulkan",
                    backend="vulkan",
                    case_pass=0,
                ),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertFalse(report["ok"])
        self.assertEqual(report["metric_evaluations"], [])
        self.assertEqual(
            sum("case_pass must be numeric 1" in error for error in report["errors"]),
            2,
        )

    def test_cross_backend_environment_exceptions_do_not_hide_host_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, correctness_manifest())
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt("baseline-run", manifest, variant_id="rocm", backend="rocm"),
            )
            candidate_value = receipt(
                "candidate-run", manifest, variant_id="vulkan", backend="vulkan"
            )
            candidate_value["environment"]["host_id"] = "different-host"
            candidate = write_receipt(root, "candidate.json", candidate_value)
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertFalse(report["compatible"])
        self.assertTrue(
            any(
                item.get("path") == "host_id"
                for item in report["compatibility"]["rejected_differences"]
            )
        )

    def test_cross_backend_environment_permissions_cannot_be_dormant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = correctness_manifest()
            allowed = value["comparison_policy"]["backend_pairs"][0][
                "allowed_environment_differences"
            ]
            allowed.append("os.kernel")
            allowed.sort()
            manifest = commit_manifest(root, value)
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt("baseline-run", manifest, variant_id="rocm", backend="rocm"),
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt("candidate-run", manifest, variant_id="vulkan", backend="vulkan"),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertFalse(report["compatible"])
        self.assertTrue(any("os.kernel" in error and "unused" in error for error in report["errors"]))

    def test_receipt_parameters_must_match_the_committed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline_value = receipt(
                "baseline-run", manifest, variant_id="default", backend="rocm"
            )
            baseline_value["workload"]["parameters"]["undeclared"] = 1
            baseline = write_receipt(root, "baseline.json", baseline_value)
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt("candidate-run", manifest, variant_id="default", backend="rocm"),
            )
            with self.assertRaisesRegex(compare_module.ComparisonError, "undeclared keys"):
                compare_module.compare_receipts(baseline, candidate, manifest=manifest)

    def test_metric_policy_rejects_wrong_matching_metadata_and_float_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline_value = receipt(
                "baseline-run", manifest, variant_id="default", backend="rocm"
            )
            candidate_value = receipt(
                "candidate-run", manifest, variant_id="default", backend="rocm"
            )
            baseline_value["results"][0]["metrics"][0]["unit"] = "req/s"
            candidate_value["results"][0]["metrics"][0]["unit"] = "req/s"
            baseline = write_receipt(root, "baseline.json", baseline_value)
            candidate = write_receipt(root, "candidate.json", candidate_value)
            metadata_report = compare_module.compare_receipts(
                baseline, candidate, manifest=manifest
            )

        self.assertFalse(metadata_report["ok"])
        self.assertTrue(any("definition does not match" in error for error in metadata_report["errors"]))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = performance_manifest()
            value["comparison_policy"]["metric_rules"][0]["relative_tolerance"] = 2.0
            manifest = commit_manifest(root, value)
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt(
                    "baseline-run",
                    manifest,
                    variant_id="default",
                    backend="rocm",
                    throughput=1e308,
                ),
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt(
                    "candidate-run",
                    manifest,
                    variant_id="default",
                    backend="rocm",
                    throughput=-1.1e308,
                ),
            )
            overflow_report = compare_module.compare_receipts(
                baseline, candidate, manifest=manifest
            )

        self.assertFalse(overflow_report["ok"])
        self.assertEqual(overflow_report["metric_evaluations"][0]["status"], "failed")

    def test_metric_tolerance_preserves_more_than_eighty_digits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = performance_manifest()
            value["comparison_policy"]["metric_rules"][0]["relative_tolerance"] = 1e-100
            manifest = commit_manifest(root, value)
            baseline_value = 10**100
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt(
                    "baseline-run",
                    manifest,
                    variant_id="default",
                    backend="rocm",
                    throughput=baseline_value,
                ),
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt(
                    "candidate-run",
                    manifest,
                    variant_id="default",
                    backend="rocm",
                    throughput=baseline_value - 2,
                ),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertFalse(report["ok"])
        self.assertEqual(report["metric_evaluations"][0]["status"], "failed")
        self.assertIn("violation 2", report["metric_evaluations"][0]["reason"])

    def test_manifest_comparison_rejects_vacuous_self_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            loaded = write_receipt(
                root,
                "run.json",
                receipt("same-run", manifest, variant_id="default", backend="rocm"),
            )
            with self.assertRaisesRegex(compare_module.ComparisonError, "distinct receipt IDs"):
                compare_module.compare_receipts(loaded, loaded, manifest=manifest)

            injected = compare_module.LoadedReceipt(
                path=root / "copy.json",
                value={**copy.deepcopy(loaded.value), "receipt_id": "different-run"},
                sha256=loaded.sha256,
            )
            with self.assertRaisesRegex(compare_module.ComparisonError, "distinct receipt content"):
                compare_module.compare_receipts(loaded, injected, manifest=manifest)

    def test_no_manifest_mode_is_limited_to_environment_receipts(self) -> None:
        path = (
            QUALIFICATION_DIR.parents[1]
            / "qualification/receipts/rocm/strix-halo/"
            "20260709t202926z-rocm-strix-halo-environment-v1.json"
        )
        loaded = compare_module.load_validated_receipt(path)
        value = copy.deepcopy(loaded.value)
        value["qualification"]["kind"] = "correctness"
        invalid = compare_module.LoadedReceipt(
            path=loaded.path,
            value=value,
            sha256=loaded.sha256,
        )
        with self.assertRaisesRegex(compare_module.ComparisonError, "non-environment"):
            compare_module.compare_receipts(invalid, invalid, manifest=None)

    def test_comparator_json_loader_rejects_lossy_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, number in (
                ("overflow", "1e400"),
                ("underflow", "1e-4000"),
                ("rounded-boundary", "8.9999999999999999"),
                ("rounded-integer", "9007199254740993.0"),
                ("rounded-subnormal", "4e-324"),
                ("huge-integer", "1" + "0" * 5000),
            ):
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.json"
                    path.write_text('{"value":' + number + "}")
                    with self.assertRaises(compare_module.ComparisonError):
                        compare_module.load_validated_receipt(path)

    def test_cross_backend_effective_config_excludes_backend_and_variant_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, correctness_manifest())
            baseline = write_receipt(
                root,
                "baseline.json",
                receipt(
                    "baseline-run",
                    manifest,
                    variant_id="rocm",
                    backend="rocm",
                    config={"backend": "rocm"},
                ),
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt(
                    "candidate-run",
                    manifest,
                    variant_id="vulkan",
                    backend="vulkan",
                    config={"backend": "vulkan"},
                ),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertFalse(report["compatible"])
        self.assertIn("incompatible effective_config identity", report["errors"])
        self.assertEqual(report["metric_evaluations"], [])

    def test_cross_backend_rejects_undeclared_endpoint_combination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, correctness_manifest())
            baseline = write_receipt(
                root, "baseline.json", receipt("baseline-run", manifest, variant_id="rocm", backend="rocm")
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt("candidate-run", manifest, variant_id="rocm", backend="rocm"),
            )
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertFalse(report["compatible"])
        self.assertTrue(any("cross-backend pair" in error for error in report["errors"]))
        self.assertEqual(report["metric_evaluations"], [])

    def test_receipt_results_must_exactly_match_selected_variant_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline_value = receipt("baseline-run", manifest, variant_id="default", backend="rocm")
            candidate_value = receipt("candidate-run", manifest, variant_id="default", backend="rocm")
            extra = {
                "id": "undeclared-infrastructure-result",
                "required": False,
                "status": "passed",
                "duration_seconds": 0.0,
                "metrics": [],
                "details": None,
            }
            baseline_value["results"].append(copy.deepcopy(extra))
            candidate_value["results"].append(copy.deepcopy(extra))
            baseline = write_receipt(root, "baseline.json", baseline_value)
            candidate = write_receipt(root, "candidate.json", candidate_value)
            report = compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        self.assertTrue(report["compatible"])
        self.assertFalse(report["ok"])
        self.assertTrue(any("undeclared results" in error for error in report["errors"]))
        self.assertEqual(report["metric_deltas"], [])

    def test_manifest_must_match_head_and_receipt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            path = root / manifest.path
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(compare_module.ComparisonError, "match.*committed HEAD"):
                compare_module.load_committed_workload(path, root=root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline_value = receipt("baseline-run", manifest, variant_id="default", backend="rocm")
            baseline_value["workload"]["sha256"] = HASH_C
            baseline = write_receipt(root, "baseline.json", baseline_value)
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt("candidate-run", manifest, variant_id="default", backend="rocm"),
            )
            with self.assertRaisesRegex(compare_module.ComparisonError, "exact committed manifest"):
                compare_module.compare_receipts(baseline, candidate, manifest=manifest)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline = write_receipt(
                root, "baseline.json", receipt("baseline-run", manifest, variant_id="default", backend="rocm")
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt("candidate-run", manifest, variant_id="default", backend="rocm"),
            )
            injected_value = copy.deepcopy(manifest.value)
            injected_value["comparison_policy"]["metric_rules"][0][
                "relative_tolerance"
            ] = 1.0
            injected = compare_module.LoadedWorkload(
                root=manifest.root,
                path=manifest.path,
                value=injected_value,
                sha256=manifest.sha256,
            )
            with self.assertRaisesRegex(compare_module.ComparisonError, "exact committed workload"):
                compare_module.compare_receipts(baseline, candidate, manifest=injected)

    def test_cli_has_no_force_and_requires_manifest_for_workload_receipts(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(QUALIFICATION_DIR / "compare_receipts.py"), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertNotIn("--force", help_result.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = commit_manifest(root, performance_manifest())
            baseline = write_receipt(
                root, "baseline.json", receipt("baseline-run", manifest, variant_id="default", backend="rocm")
            )
            candidate = write_receipt(
                root,
                "candidate.json",
                receipt("candidate-run", manifest, variant_id="default", backend="rocm"),
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(QUALIFICATION_DIR / "compare_receipts.py"),
                    str(baseline.path),
                    str(candidate.path),
                    "--root",
                    str(root),
                    "--json",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            accepted = subprocess.run(
                [
                    sys.executable,
                    str(QUALIFICATION_DIR / "compare_receipts.py"),
                    str(baseline.path),
                    str(candidate.path),
                    "--root",
                    str(root),
                    "--workload-manifest",
                    str(manifest.path),
                    "--json",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("--workload-manifest is required", json.loads(completed.stdout)["error"])
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertTrue(json.loads(accepted.stdout)["ok"])

    def test_existing_environment_receipt_can_be_compared_strictly_without_manifest(self) -> None:
        path = (
            QUALIFICATION_DIR.parents[1]
            / "qualification/receipts/rocm/strix-halo/"
            "20260709t202926z-rocm-strix-halo-environment-v1.json"
        )
        loaded = compare_module.load_validated_receipt(path)
        report = compare_module.compare_receipts(loaded, loaded, manifest=None)
        self.assertTrue(report["ok"])
        self.assertEqual(report["mode"], "strict_no_workload")
        self.assertEqual(report["metric_deltas"], [])


if __name__ == "__main__":
    unittest.main()
