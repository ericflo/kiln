from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from typing import Any


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, QUALIFICATION_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_module = load_module("qualification_run", "run.py")
receipt_module = load_module("qualification_run_receipt", "receipt.py")
compare_module = load_module("qualification_run_compare", "compare_receipts.py")


def variable(
    name: str,
    variable_type: str,
    *,
    required: bool,
    default: Any,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {
        "allowed_values": [],
        "minimum": None,
        "maximum": None,
        "pattern": None,
    }
    if variable_type == "integer":
        constraints["minimum"] = 1
    return {
        "name": name,
        "description": f"Test variable {name}.",
        "type": variable_type,
        "required": required,
        "default": default,
        "constraints": constraints,
    }


def runner_protocol() -> dict[str, Any]:
    return {
        "format": "qualification-case-result-v1",
        "producer": "runner",
        "path_environment_variable": "KILN_QUALIFICATION_CASE_RESULT",
        "declared_metrics": [
            "case_pass",
            "exit_code",
            "output_assertion_failures",
        ],
    }


def command_protocol(metric: str = "sample_value") -> dict[str, Any]:
    return {
        "format": "qualification-case-result-v1",
        "producer": "command",
        "path_environment_variable": "KILN_QUALIFICATION_CASE_RESULT",
        "declared_metrics": [metric],
    }


def environment_workload(
    command: list[str],
    *,
    protocol: dict[str, Any] | None = None,
    assertions: list[dict[str, str]] | None = None,
    timeout_seconds: int = 10,
    repetitions: int = 1,
    expected_exit_codes: list[int] | None = None,
    effective_config: dict[str, Any] | None = None,
    variables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workload_id": "runner-unit-environment-v1",
        "kind": "environment",
        "description": "Exercise the local qualification runner without hardware.",
        "determinism": {
            "seed": None,
            "seed_delivery": "not_applicable",
            "repetitions": repetitions,
            "case_order": "declared",
            "max_parallel_cases": 1,
            "network_access": "forbidden",
        },
        "variables": variables or [],
        "variants": [
            {
                "id": "rocm",
                "description": "Fake ROCm environment used by unit tests.",
                "backend": "rocm",
                "device_requirement": "required",
                "skip_policy": "fail",
                "effective_config": effective_config or {},
                "cases": [
                    {
                        "id": "smoke-case",
                        "description": "Execute one deterministic fake case.",
                        "required": True,
                        "command": command,
                        "working_directory": ".",
                        "environment": {},
                        "timeout_seconds": timeout_seconds,
                        "expected_exit_codes": expected_exit_codes or [0],
                        "output_assertions": assertions or [],
                        "result_protocol": protocol or runner_protocol(),
                    }
                ],
            }
        ],
        "comparison_policy": None,
    }


def performance_ab_workload() -> dict[str, Any]:
    variants = []
    for variant_id, batch_size, metric_value in (
        ("baseline", 1, 10.0),
        ("candidate", 2, 11.0),
    ):
        config = {"scheduler": {"max_batch": batch_size}}
        variants.append(
            {
                "id": variant_id,
                "description": f"Synthetic {variant_id} A/B endpoint.",
                "backend": "rocm",
                "device_requirement": "required",
                "skip_policy": "fail",
                "effective_config": config,
                "cases": [
                    {
                        "id": "smoke-case",
                        "description": "Emit a strict synthetic performance result.",
                        "required": True,
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import os,sys; "
                                "assert sys.argv[1] == os.environ['TEST_MODEL_PATH']; "
                                + case_result_script(
                                    effective_config=config,
                                    metric_value=metric_value,
                                )
                            ),
                            "${model_path}",
                            "${seed}",
                        ],
                        "working_directory": ".",
                        "environment": {"TEST_MODEL_PATH": "${model_path}"},
                        "timeout_seconds": 10,
                        "expected_exit_codes": [0],
                        "output_assertions": [],
                        "result_protocol": command_protocol(),
                    }
                ],
            }
        )
    return {
        "schema_version": 1,
        "workload_id": "runner-ab-performance-v1",
        "kind": "performance",
        "description": "Exercise runner-to-comparator A/B config binding.",
        "determinism": {
            "seed": 20260709,
            "seed_delivery": "argv",
            "repetitions": 1,
            "case_order": "declared",
            "max_parallel_cases": 1,
            "network_access": "forbidden",
        },
        "variables": [],
        "variants": variants,
        "comparison_policy": {
            "mode": "declared_ab_variants",
            "variant_pairs": [
                {
                    "baseline_variant_id": "baseline",
                    "candidate_variant_id": "candidate",
                    "allowed_effective_config_differences": ["scheduler.max_batch"],
                }
            ],
            "backend_pairs": [],
            "metric_rules": [
                {
                    "scope": "result",
                    "result_id": "smoke-case",
                    "metric": "sample_value",
                    "metric_class": "performance",
                    "unit": "items",
                    "aggregation": "exact",
                    "lower_is_better": False,
                    "operator": "not_less",
                    "absolute_tolerance": 0,
                    "relative_tolerance": 0,
                    "required": True,
                }
            ],
        },
    }


def case_result_script(
    *,
    effective_config: dict[str, Any] | None = None,
    metric_value: float = 3.0,
    metric_unit: str = "items",
    lower_is_better: bool = False,
    details: str | None = None,
    status: str = "passed",
) -> str:
    value = {
        "schema_version": 1,
        "case_id": "smoke-case",
        "status": status,
        "duration_seconds": 0.01,
        "effective_config": effective_config or {},
        "metrics": [
            {
                "name": "sample_value",
                "value": metric_value,
                "unit": metric_unit,
                "aggregation": "exact",
                "lower_is_better": lower_is_better,
            }
        ],
        "tolerances": [],
        "details": details,
    }
    return (
        "import json,os; "
        f"value={value!r}; "
        "open(os.environ['KILN_QUALIFICATION_CASE_RESULT'],'w').write(json.dumps(value))"
    )


class Repository:
    def __init__(self, workload: dict[str, Any]):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "qualification/workloads").mkdir(parents=True)
        (self.root / ".gitignore").write_text(".qualification/\nqualification/receipts/\n")
        (self.root / "Cargo.toml").write_text("[workspace]\nmembers = []\n")
        self.workload_path = self.root / "qualification/workloads/test.json"
        self.workload_path.write_text(json.dumps(workload, indent=2, sort_keys=True) + "\n")
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "runner-tests@kiln.invalid"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Kiln Runner Tests"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "fixture"], cwd=self.root, check=True
        )

    def close(self) -> None:
        self.temporary.cleanup()


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hook_calls = {"environment": 0, "model": 0, "network": 0}

    def test_macos_network_isolation_preserves_loopback_and_denies_external_network(self) -> None:
        completed = subprocess.CompletedProcess([], 0, b"", b"")
        with mock.patch.object(run_module.sys, "platform", "darwin"), mock.patch.object(
            run_module.shutil, "which", return_value="/usr/bin/sandbox-exec"
        ), mock.patch.object(run_module.subprocess, "run", return_value=completed) as invoked:
            isolation = run_module.establish_network_isolation(Path("/tmp"))
        self.assertEqual(isolation.mechanism, "macos-sandbox-loopback-only-v1")
        self.assertEqual(isolation.argv_prefix[0], "/usr/bin/sandbox-exec")
        self.assertIn("(deny network*)", isolation.argv_prefix[2])
        command = invoked.call_args.args[0]
        self.assertIn("127.0.0.1", command[-1])
        self.assertIn("192.0.2.1", command[-1])

    def test_macos_network_isolation_fails_closed_when_probe_fails(self) -> None:
        completed = subprocess.CompletedProcess([], 1, b"", b"sandbox denied loopback")
        with mock.patch.object(run_module.sys, "platform", "darwin"), mock.patch.object(
            run_module.shutil, "which", return_value="/usr/bin/sandbox-exec"
        ), mock.patch.object(run_module.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(
                run_module.QualificationRunError, "did not preserve loopback"
            ):
                run_module.establish_network_isolation(Path("/tmp"))

    def test_linux_network_isolation_uses_pid_containing_unshare_fallback(self) -> None:
        completed = subprocess.CompletedProcess([], 0, b"", b"")

        def which(name: str) -> str | None:
            return {
                "bwrap": None,
                "unshare": "/usr/bin/unshare",
                "ip": "/usr/sbin/ip",
            }.get(name)

        with mock.patch.object(
            run_module.sys, "platform", "linux"
        ), mock.patch.object(
            run_module.shutil, "which", side_effect=which
        ), mock.patch.object(
            run_module.subprocess, "run", return_value=completed
        ) as invoked:
            isolation = run_module.establish_network_isolation(Path("/tmp"))

        self.assertEqual(
            isolation.mechanism,
            "util-linux-unshare-user-net-pid-landlock-v1",
        )
        self.assertIn("--map-root-user", isolation.argv_prefix)
        self.assertIn("--kill-child=SIGKILL", isolation.argv_prefix)
        self.assertIn("--mount-proc=/proc", isolation.argv_prefix)
        self.assertIn(str(run_module.LINUX_NAMESPACE_EXEC), isolation.argv_prefix)
        probe = invoked.call_args.args[0]
        self.assertIn("127.0.0.1", probe[-1])
        self.assertIn("192.0.2.1", probe[-1])
        self.assertIn("os.replace(rename_source,rename_destination)", probe[-1])
        if "microsoft-standard-wsl2" in run_module.platform.release().lower():
            self.assertIn("/mnt/c/Windows/System32/cmd.exe", probe[-1])

    def test_linux_network_isolation_fails_closed_without_complete_mechanism(
        self,
    ) -> None:
        with mock.patch.object(
            run_module.sys, "platform", "linux"
        ), mock.patch.object(
            run_module.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/unshare" if name == "unshare" else None,
        ):
            with self.assertRaisesRegex(
                run_module.QualificationRunError, "missing: ip"
            ):
                run_module.establish_network_isolation(Path("/tmp"))

    def test_wsl_supervision_keeps_scope_independent_from_optional_pacing(
        self,
    ) -> None:
        policy_path = (
            QUALIFICATION_DIR.parents[1]
            / "qualification/host-policies/"
            "rtx4090-laptop-wsl2-cgroup-pacing-v2.json"
        )
        policy = run_module.wsl_thermal_exec.load_policy(policy_path)
        command = run_module._wsl_supervised_argv(
            ["/usr/bin/unshare", "--", "/bin/true"],
            policy_path,
            policy,
            30.0,
        )
        self.assertEqual(command[1], str(run_module.WSL_THERMAL_EXEC))
        memory_index = command.index("--memory-max-bytes")
        self.assertEqual(
            command[memory_index + 1],
            str(run_module.WSL_SCOPE_PACED_MEMORY_MAX_BYTES),
        )
        pacing_index = command.index("--thermal-pacing-policy")
        self.assertEqual(command[pacing_index + 1], str(policy_path))
        self.assertEqual(command.count(str(policy_path)), 2)

        v1_path = (
            QUALIFICATION_DIR.parents[1]
            / "qualification/host-policies/"
            "rtx4090-laptop-wsl2-boundary-v1.json"
        )
        hard_limit_command = run_module._wsl_supervised_argv(
            ["/bin/true"],
            v1_path,
            run_module.wsl_thermal_exec.load_policy(v1_path),
            30.0,
        )
        self.assertNotIn("--thermal-pacing-policy", hard_limit_command)
        self.assertEqual(hard_limit_command.count(str(v1_path)), 1)

        unpaced_command = run_module._wsl_supervised_argv(
            ["/bin/true"],
            None,
            None,
            30.0,
        )
        self.assertEqual(unpaced_command[1], str(run_module.WSL_SCOPE_EXEC))
        memory_index = unpaced_command.index("--memory-max-bytes")
        self.assertEqual(
            unpaced_command[memory_index + 1],
            str(run_module.WSL_SCOPE_UNPACED_MEMORY_MAX_BYTES),
        )
        self.assertNotIn(str(run_module.WSL_THERMAL_EXEC), unpaced_command)
        self.assertNotIn("--thermal-pacing-policy", unpaced_command)

    def test_wsl_model_fingerprint_is_scoped_and_unlimited_by_default(self) -> None:
        policy_path = (
            QUALIFICATION_DIR.parents[1]
            / "qualification/host-policies/"
            "rtx4090-laptop-wsl2-cgroup-pacing-v2.json"
        )
        policy = run_module.wsl_thermal_exec.load_policy(policy_path)
        isolation = run_module.NetworkIsolation(
            mechanism="util-linux-unshare-user-net-pid-landlock-v1",
            argv_prefix=("/usr/bin/unshare", "--"),
        )
        payload = {
            "id": "fixture",
            "path": "/tmp/model",
            "weight_files": [],
            "config_hash": "sha256:" + "a" * 64,
            "tokenizer_hash": "sha256:" + "b" * 64,
            "chat_template_hash": None,
        }
        completed = subprocess.CompletedProcess(
            args=["wrapped"],
            returncode=0,
            stdout=json.dumps(payload).encode(),
            stderr=b'{"event":"wsl_scope_complete"}\n',
        )
        with tempfile.TemporaryDirectory() as temporary:
            stdout_path = Path(temporary) / "fingerprint.json"
            stderr_path = Path(temporary) / "fingerprint.stderr"
            environment = {
                "PATH": "/usr/bin",
                run_module.NETWORK_ISOLATION_ENVIRONMENT_VARIABLE: (
                    isolation.mechanism
                ),
            }
            with mock.patch.object(
                run_module,
                "_wsl_supervised_argv",
                return_value=["wrapped"],
            ) as wrapped, mock.patch.object(
                run_module.subprocess,
                "run",
                return_value=completed,
            ) as invoked:
                observed = run_module._supervised_wsl_model_fingerprint(
                    Path("/tmp/model"),
                    "fixture",
                    network_isolation=isolation,
                    policy_path=policy_path,
                    policy=policy,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    environment=environment,
                )
            self.assertEqual(observed, payload)
            contained = wrapped.call_args.args[0]
            self.assertEqual(contained[:2], list(isolation.argv_prefix))
            self.assertNotIn("--max-read-mib-per-second", contained)
            self.assertEqual(stdout_path.read_bytes(), completed.stdout)
            self.assertEqual(stderr_path.read_bytes(), completed.stderr)
            self.assertEqual(
                invoked.call_args.kwargs["env"],
                environment,
            )

    def test_wsl_model_fingerprint_nonzero_exit_fails_after_capture(self) -> None:
        policy_path = (
            QUALIFICATION_DIR.parents[1]
            / "qualification/host-policies/"
            "rtx4090-laptop-wsl2-cgroup-pacing-v2.json"
        )
        policy = run_module.wsl_thermal_exec.load_policy(policy_path)
        isolation = run_module.NetworkIsolation("test", ("/bin/true", "--"))
        completed = subprocess.CompletedProcess(
            args=["wrapped"],
            returncode=3,
            stdout=b"",
            stderr=b"thermal trip with stable handoff\n",
        )
        with tempfile.TemporaryDirectory() as temporary:
            stdout_path = Path(temporary) / "fingerprint.json"
            stderr_path = Path(temporary) / "fingerprint.stderr"
            environment = {
                run_module.NETWORK_ISOLATION_ENVIRONMENT_VARIABLE: (
                    isolation.mechanism
                )
            }
            with mock.patch.object(
                run_module,
                "_wsl_supervised_argv",
                return_value=["wrapped"],
            ), mock.patch.object(
                run_module.subprocess,
                "run",
                return_value=completed,
            ):
                with self.assertRaisesRegex(
                    run_module.QualificationRunError,
                    "failed under scope supervision",
                ):
                    run_module._supervised_wsl_model_fingerprint(
                        Path("/tmp/model"),
                        "fixture",
                        network_isolation=isolation,
                        policy_path=policy_path,
                        policy=policy,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        environment=environment,
                    )
            self.assertEqual(stdout_path.read_bytes(), b"")
            self.assertEqual(stderr_path.read_bytes(), completed.stderr)

    def test_inherited_auth_payloads_are_hashed_before_local_capture(self) -> None:
        captured = run_module._redacted_environment(
            {
                "CODEX_AUTH_JSON": '{"refresh_token":"must-not-appear"}',
                "HTTP_COOKIE": "session=must-not-appear",
                "QUALIFICATION_LABEL": "visible",
            }
        )
        self.assertEqual(
            captured["QUALIFICATION_LABEL"],
            {"redacted": False, "value": "visible"},
        )
        for name in ("CODEX_AUTH_JSON", "HTTP_COOKIE"):
            self.assertTrue(captured[name]["redacted"])
            self.assertRegex(captured[name]["value"], r"^sha256:[0-9a-f]{64}$")
            self.assertNotIn("must-not-appear", captured[name]["value"])

    def test_case_environment_excludes_undeclared_ambient_controls(self) -> None:
        workload = environment_workload(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "assert os.environ['DECLARED_CASE_VALUE'] == 'manifest'; "
                    "assert 'PATH' in os.environ; "
                    "assert 'ROCM_PATH' not in os.environ; "
                    "assert 'KILN_UNDECLARED_RUNNER_TEST' not in os.environ; "
                    "assert 'CODEX_AUTH_JSON' not in os.environ"
                ),
            ]
        )
        workload["variants"][0]["cases"][0]["environment"] = {
            "DECLARED_CASE_VALUE": "manifest"
        }
        repository = Repository(workload)
        self.addCleanup(repository.close)
        with mock.patch.dict(
            os.environ,
            {
                "ROCM_PATH": "/ambient/rocm",
                "KILN_UNDECLARED_RUNNER_TEST": "ambient-product-control",
                "CODEX_AUTH_JSON": "ambient-credential",
            },
            clear=False,
        ):
            outcome = self.execute(repository)

        self.assertEqual(outcome.exit_code, 0)
        config_artifact = next(
            artifact
            for artifact in outcome.receipt["artifacts"]
            if artifact["kind"] == "effective_run_config"
        )
        run_config = json.loads((repository.root / config_artifact["path"]).read_text())
        self.assertEqual(
            run_config["case_environment_policy"],
            run_module.CASE_ENVIRONMENT_POLICY,
        )
        self.assertNotIn("ROCM_PATH", run_config["case_base_environment"])
        self.assertNotIn(
            "KILN_UNDECLARED_RUNNER_TEST", run_config["case_base_environment"]
        )
        self.assertNotIn("CODEX_AUTH_JSON", run_config["case_base_environment"])
        self.assert_valid(outcome, repository.root)

    def test_host_id_placeholder_resolves_from_runner_input(self) -> None:
        repository = Repository(
            environment_workload(
                [
                    sys.executable,
                    "-c",
                    "import sys; assert sys.argv[1] == 'test-host'",
                    "${host_id}",
                ]
            )
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.receipt["workload"]["parameters"], {"variant_id": "rocm"})
        self.assert_valid(outcome, repository.root)

    def test_vulkan_device_probes_allow_headless_surface_skip_only(self) -> None:
        for name in ("correctness-core-v1.json", "prefill-scheduling-v1.json"):
            with self.subTest(name=name):
                workload = json.loads(
                    (run_module.ROOT / "qualification" / "workloads" / name).read_text()
                )
                variant = next(
                    item for item in workload["variants"] if item["id"] == "vulkan"
                )
                case = next(item for item in variant["cases"] if item["id"] == "device-probe")
                stdout = "Devices:\n========\nGPU0:\n\tdeviceName = Test Vulkan Device\n"
                surface_stderr = (
                    "'DISPLAY' environment variable not set... skipping surface info\n"
                )
                self.assertEqual(
                    run_module._assert_output(case, stdout, surface_stderr), []
                )
                self.assertTrue(
                    run_module._assert_output(
                        case, stdout + "skipping Vulkan physical device\n", ""
                    )
                )
                self.assertTrue(
                    run_module._assert_output(
                        case, "Devices:\n========\n", surface_stderr
                    )
                )

    def fake_environment(
        self, backend: str, host_id: str, root: Path
    ) -> run_module.EnvironmentCapture:
        self.hook_calls["environment"] += 1
        return run_module.EnvironmentCapture(
            environment={
                "host_id": host_id,
                "os": {
                    "name": "Test OS",
                    "version": "1",
                    "kernel": "test-kernel",
                    "architecture": "x86_64",
                },
                "device": {
                    "name": "Fake GPU",
                    "architecture": "fake1000",
                    "memory_bytes": 1024,
                    "unified_memory": False,
                    "driver": "fake-driver",
                },
                "runtime": {backend: "test-runtime"},
                "compiler": {"python": platform_python()},
            },
            probe_results=[
                {
                    "id": "fake-device",
                    "required": True,
                    "status": "passed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": None,
                }
            ],
            raw={"fake": True},
        )

    def fake_model(self, path: Path, model_id: str | None) -> dict[str, Any]:
        self.hook_calls["model"] += 1
        digest = "sha256:" + "a" * 64
        return {
            "id": model_id or "fake-model",
            "path": str(path),
            "weight_files": [
                {"path": "model.safetensors", "sha256": digest, "bytes": 1}
            ],
            "config_hash": digest,
            "tokenizer_hash": digest,
            "chat_template_hash": None,
        }

    def fake_network(self, root: Path) -> run_module.NetworkIsolation:
        self.hook_calls["network"] += 1
        return run_module.NetworkIsolation("unit-test-no-network", ())

    def hooks(self) -> run_module.RunnerHooks:
        return run_module.RunnerHooks(
            capture_environment=self.fake_environment,
            fingerprint_model=self.fake_model,
            network_isolation=self.fake_network,
        )

    def execute(
        self,
        repository: Repository,
        *,
        receipt_id: str = "runner-test-receipt-v1",
        assignments: list[str] | None = None,
        output: Path | None = None,
        termination_grace_seconds: float = 0.1,
    ) -> run_module.RunOutcome:
        return run_module.run_qualification(
            repository.workload_path,
            variant_id="rocm",
            host_id="test-host",
            variable_assignments=assignments or [],
            output=output,
            receipt_id=receipt_id,
            root=repository.root,
            invocation=["qualification-runner-test"],
            hooks=self.hooks(),
            termination_grace_seconds=termination_grace_seconds,
        )

    def assert_valid(self, outcome: run_module.RunOutcome, root: Path) -> None:
        self.assertEqual(receipt_module.load_receipt(outcome.receipt_path), outcome.receipt)
        self.assertEqual(
            receipt_module.validate_receipt(
                outcome.receipt,
                root=root,
                require_current_source=outcome.exit_code == 0,
                require_local_artifacts=True,
            ),
            [],
        )

    def test_success_records_typed_inputs_and_validates_strictly(self) -> None:
        definitions = [
            variable("count", "integer", required=False, default=2),
            variable("flag", "boolean", required=True, default=None),
        ]
        workload = environment_workload(
            [
                sys.executable,
                "-c",
                "import sys; print('READY', sys.argv[1:])",
                "${count}",
                "${flag}",
            ],
            variables=definitions,
            assertions=[{"stream": "stdout", "match": "required", "pattern": "READY"}],
        )
        repository = Repository(workload)
        self.addCleanup(repository.close)
        outcome = self.execute(repository, assignments=["flag=true"])

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.receipt["qualification"]["verdict"], "passed")
        self.assertEqual(
            outcome.receipt["workload"]["parameters"],
            {"variant_id": "rocm", "count": 2, "flag": True},
        )
        self.assertEqual(outcome.receipt["effective_config"], {})
        self.assertEqual([item["id"] for item in outcome.receipt["results"]], ["smoke-case"])
        metrics = {item["name"]: item for item in outcome.receipt["results"][0]["metrics"]}
        self.assertEqual(
            metrics["case_pass"],
            {
                "name": "case_pass",
                "value": 1,
                "unit": "bool",
                "aggregation": "exact",
                "lower_is_better": False,
            },
        )
        self.assert_valid(outcome, repository.root)

    def test_selected_variant_is_bound_into_case_environment(self) -> None:
        workload = environment_workload(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ['KILN_QUALIFICATION_VARIANT_ID'])",
            ],
            assertions=[
                {"stream": "stdout", "match": "required", "pattern": "^rocm$"}
            ],
        )
        repository = Repository(workload)
        self.addCleanup(repository.close)
        outcome = self.execute(repository)

        self.assertEqual(outcome.exit_code, 0)
        config_artifact = next(
            artifact
            for artifact in outcome.receipt["artifacts"]
            if artifact["kind"] == "effective_run_config"
        )
        run_config = json.loads((repository.root / config_artifact["path"]).read_text())
        self.assertEqual(
            run_config["cases"][0]["runner_environment"],
            {
                "KILN_QUALIFICATION_NETWORK_ISOLATION": "unit-test-no-network",
                "KILN_QUALIFICATION_VARIANT_ID": "rocm",
            },
        )
        self.assert_valid(outcome, repository.root)

    def test_nonzero_exit_still_emits_valid_failed_receipt(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "raise SystemExit(7)"])
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 1)
        self.assertEqual(outcome.receipt["results"][0]["status"], "failed")
        self.assertIn("exit code 7", outcome.receipt["results"][0]["details"])
        self.assertTrue(outcome.receipt_path.is_file())
        self.assert_valid(outcome, repository.root)

    def test_runner_preserves_json_details_when_adding_contract_failures(self) -> None:
        details = run_module._join_details(
            json.dumps(
                {
                    "error": "capacity failure: " + "x" * 4000,
                    "milestones": ["build", "startup", "measurement", "teardown"],
                },
                separators=(",", ":"),
            ),
            "exit code 1 not in [0]",
        )
        assert details is not None
        self.assertLessEqual(
            len(details), receipt_module.MAX_RESULT_DETAIL_CHARACTERS
        )
        parsed = json.loads(details)
        self.assertEqual(
            parsed["milestones"],
            ["build", "startup", "measurement", "teardown"],
        )
        self.assertEqual(parsed["runner_failures"], ["exit code 1 not in [0]"])
        self.assertIn("[truncated sha256:", parsed["error"])
        self.assertRegex(parsed["_truncation"]["sha256"], r"^sha256:[0-9a-f]{64}$")

    def test_runner_keeps_distinct_structured_repetition_details(self) -> None:
        details = run_module._join_details(
            '{"attempt":1,"milestones":["build"]}',
            '{"attempt":2,"milestones":["build","measurement"]}',
        )
        assert details is not None
        parsed = json.loads(details)
        self.assertEqual(parsed["attempt"], 1)
        self.assertEqual(
            parsed["additional_case_details"],
            [{"attempt": 2, "milestones": ["build", "measurement"]}],
        )

    def test_timeout_terminates_descendant_process_group(self) -> None:
        code = (
            "import pathlib,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); "
            "pathlib.Path('.qualification/child.pid').write_text(str(child.pid)); "
            "time.sleep(30)"
        )
        repository = Repository(
            environment_workload([sys.executable, "-c", code], timeout_seconds=1)
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        child_pid = int((repository.root / ".qualification/child.pid").read_text())

        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("timed out", outcome.receipt["results"][0]["details"])
        deadline = time.monotonic() + 3
        while process_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_running(child_pid), f"child {child_pid} survived timeout cleanup")
        self.assert_valid(outcome, repository.root)

    def test_process_group_cleanup_lets_driver_finish_before_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ready = root / "ready"
            cleaned = root / "cleaned"
            child_code = (
                "import pathlib,signal,time; "
                f"ready=pathlib.Path({str(ready)!r}); "
                f"cleaned=pathlib.Path({str(cleaned)!r}); "
                "signal.signal(signal.SIGINT,lambda *_: "
                "(cleaned.write_text('cleaned'),exit(0))); "
                "ready.write_text('ready'); "
                "time.sleep(30)"
            )
            wrapper_code = (
                "import subprocess,sys; "
                f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                "raise SystemExit(child.wait())"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", wrapper_code],
                start_new_session=True,
            )
            self.addCleanup(lambda: process.poll() is None and process.kill())
            deadline = time.monotonic() + 2.0
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists(), "driver child did not become ready")

            run_module._terminate_process_group(process, 2.0)

            self.assertEqual(cleaned.read_text(encoding="utf-8"), "cleaned")
            self.assertEqual(process.returncode, 0)

    def test_keyboard_interrupt_removes_owned_run_directory(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "pass"])
        )
        self.addCleanup(repository.close)
        hooks = self.hooks()
        interrupted_hooks = run_module.RunnerHooks(
            capture_environment=mock.Mock(side_effect=KeyboardInterrupt),
            fingerprint_model=hooks.fingerprint_model,
            network_isolation=hooks.network_isolation,
        )

        with self.assertRaises(KeyboardInterrupt):
            run_module.run_qualification(
                repository.workload_path,
                variant_id="rocm",
                host_id="test-host",
                receipt_id="runner-interrupted-receipt-v1",
                root=repository.root,
                invocation=["qualification-runner-test"],
                hooks=interrupted_hooks,
                termination_grace_seconds=0.1,
            )

        runs_root = repository.root / ".qualification" / "runs"
        self.assertEqual(list(runs_root.iterdir()) if runs_root.exists() else [], [])

    def test_main_reports_keyboard_interrupt_without_traceback(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                run_module, "run_qualification", side_effect=KeyboardInterrupt
            ),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = run_module.main(
                ["--variant", "rocm", "--host-id", "test-host", "fixture.json"]
            )

        self.assertEqual(exit_code, 130)
        self.assertIn("owned processes and run data were cleaned", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_command_output_is_visible_before_the_process_exits(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release = root / "release"
            stderr_path = root / "stderr.log"
            outcome: list[run_module.Execution] = []
            code = (
                "import pathlib,sys,time; "
                "sys.stderr.write('ready\\n'); sys.stderr.flush(); "
                f"release=pathlib.Path({str(release)!r}); "
                "\nwhile not release.exists(): time.sleep(0.01)"
            )

            thread = threading.Thread(
                target=lambda: outcome.append(
                    run_module.execute_argv(
                        [sys.executable, "-c", code],
                        cwd=root,
                        environment=dict(os.environ),
                        stdout_path=root / "stdout.log",
                        stderr_path=stderr_path,
                        timeout_seconds=5,
                        termination_grace_seconds=0.1,
                    )
                )
            )
            thread.start()
            try:
                deadline = time.monotonic() + 2.0
                captured = b""
                while time.monotonic() < deadline:
                    if stderr_path.exists():
                        captured = stderr_path.read_bytes()
                    if captured:
                        break
                    time.sleep(0.01)
                self.assertEqual(captured, b"ready\n")
                self.assertTrue(thread.is_alive())
            finally:
                release.write_text("release\n", encoding="utf-8")
                thread.join(timeout=3.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(outcome), 1)
            self.assertEqual(outcome[0].returncode, 0)

    def test_success_waits_for_short_lived_descendant_settlement(self) -> None:
        code = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.1)'])"
        )
        repository = Repository(
            environment_workload([sys.executable, "-c", code])
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository, termination_grace_seconds=0.0)

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.receipt["results"][0]["status"], "passed")
        self.assertNotIn(
            "descendant processes", outcome.receipt["results"][0]["details"] or ""
        )
        self.assert_valid(outcome, repository.root)

    def test_success_rejects_persistent_descendant(self) -> None:
        code = (
            "import pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "pathlib.Path('.qualification/child.pid').write_text(str(child.pid))"
        )
        repository = Repository(
            environment_workload([sys.executable, "-c", code])
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        child_pid = int((repository.root / ".qualification/child.pid").read_text())

        self.assertEqual(outcome.exit_code, 1)
        self.assertIn(
            "command left descendant processes running",
            outcome.receipt["results"][0]["details"],
        )
        deadline = time.monotonic() + 3
        while process_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_running(child_pid), f"child {child_pid} survived cleanup")
        self.assert_valid(outcome, repository.root)

    def test_missing_and_malformed_command_results_fail_with_receipts(self) -> None:
        commands = {
            "missing": [sys.executable, "-c", "print('no result')"],
            "malformed": [
                sys.executable,
                "-c",
                "import os; open(os.environ['KILN_QUALIFICATION_CASE_RESULT'],'w').write('{')",
            ],
        }
        for label, command in commands.items():
            with self.subTest(label=label):
                repository = Repository(
                    environment_workload(command, protocol=command_protocol())
                )
                try:
                    outcome = self.execute(repository, receipt_id=f"runner-{label}-receipt-v1")
                    self.assertEqual(outcome.exit_code, 1)
                    self.assertEqual(outcome.receipt["results"][0]["status"], "failed")
                    self.assertIn(
                        "case result", outcome.receipt["results"][0]["details"]
                    )
                    self.assert_valid(outcome, repository.root)
                finally:
                    repository.close()

    def test_valid_command_result_attests_effective_config(self) -> None:
        config = {"scheduler": {"max_batch": 4}}
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", case_result_script(effective_config=config)],
                protocol=command_protocol(),
                effective_config=config,
            )
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.receipt["effective_config"], config)
        self.assert_valid(outcome, repository.root)

    def test_required_accelerator_skip_becomes_a_failed_valid_receipt(self) -> None:
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", case_result_script(status="skipped")],
                protocol=command_protocol(),
            )
        )
        self.addCleanup(repository.close)

        outcome = self.execute(repository)

        self.assertEqual(outcome.exit_code, 1)
        self.assertEqual(outcome.receipt["qualification"]["verdict"], "failed")
        self.assertEqual(outcome.receipt["results"][0]["status"], "failed")
        self.assertIn(
            "variant skip_policy forbids skipped cases",
            outcome.receipt["results"][0]["details"],
        )
        self.assert_valid(outcome, repository.root)

    def test_runner_receipts_pass_declared_ab_comparison(self) -> None:
        repository = Repository(performance_ab_workload())
        self.addCleanup(repository.close)
        hooks = self.hooks()
        outcomes = []
        for variant_id in ("baseline", "candidate"):
            outcomes.append(
                run_module.run_qualification(
                    repository.workload_path,
                    variant_id=variant_id,
                    host_id="test-host",
                    model_path=repository.root / "fake-model",
                    output=(
                        repository.root
                        / f".qualification/receipts/{variant_id}-receipt.json"
                    ),
                    receipt_id=f"runner-ab-{variant_id}-v1",
                    root=repository.root,
                    invocation=["qualification-runner-test", variant_id],
                    hooks=hooks,
                    termination_grace_seconds=0.1,
                )
            )
        self.assertEqual(
            [outcome.receipt["workload"]["parameters"] for outcome in outcomes],
            [{"variant_id": "baseline"}, {"variant_id": "candidate"}],
        )
        expected_model_path = str(repository.root / "fake-model")
        for outcome in outcomes:
            config_artifact = next(
                artifact
                for artifact in outcome.receipt["artifacts"]
                if artifact["kind"] == "effective_run_config"
            )
            raw_config = json.loads((repository.root / config_artifact["path"]).read_text())
            self.assertEqual(raw_config["cases"][0]["argv"][-2], expected_model_path)
            self.assertEqual(
                raw_config["cases"][0]["environment_overrides"]["TEST_MODEL_PATH"],
                expected_model_path,
            )
        manifest = compare_module.load_committed_workload(
            repository.workload_path, root=repository.root
        )
        report = compare_module.compare_receipts(
            compare_module.load_validated_receipt(outcomes[0].receipt_path),
            compare_module.load_validated_receipt(outcomes[1].receipt_path),
            manifest=manifest,
        )
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            report["compatibility"]["allowed_differences"],
            [
                {
                    "compatibility": "workload_variant",
                    "baseline": "baseline",
                    "candidate": "candidate",
                },
                {
                    "compatibility": "effective_config",
                    "path": "scheduler.max_batch",
                    "baseline": 1,
                    "candidate": 2,
                },
            ],
        )

    def test_runner_environment_receipts_use_strict_manifest_comparison(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "print('environment')"])
        )
        self.addCleanup(repository.close)
        outcomes = [
            self.execute(
                repository,
                receipt_id=f"runner-environment-{index}-v1",
            )
            for index in (1, 2)
        ]
        manifest = compare_module.load_committed_workload(
            repository.workload_path, root=repository.root
        )
        report = compare_module.compare_receipts(
            compare_module.load_validated_receipt(outcomes[0].receipt_path),
            compare_module.load_validated_receipt(outcomes[1].receipt_path),
            manifest=manifest,
        )
        self.assertEqual(report["mode"], "strict_environment_workload")
        self.assertTrue(report["ok"], report["errors"])

        for label, expected_error, mutate in (
            (
                "metric-value",
                "result_evidence",
                lambda receipt: next(
                    metric
                    for metric in receipt["results"][0]["metrics"]
                    if metric["name"] == "exit_code"
                ).__setitem__("value", 2),
            ),
            (
                "unsupported",
                "unsupported",
                lambda receipt: receipt["unsupported"].append("synthetic unsupported state"),
            ),
        ):
            with self.subTest(label=label):
                changed = copy.deepcopy(outcomes[1].receipt)
                mutate(changed)
                changed_path = (
                    repository.root / f".qualification/receipts/{label}-changed.json"
                )
                changed_path.parent.mkdir(parents=True, exist_ok=True)
                changed_path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
                changed_report = compare_module.compare_receipts(
                    compare_module.load_validated_receipt(outcomes[0].receipt_path),
                    compare_module.load_validated_receipt(changed_path),
                    manifest=manifest,
                )
                self.assertFalse(changed_report["compatible"])
                self.assertTrue(
                    any(expected_error in error for error in changed_report["errors"]),
                    changed_report["errors"],
                )

    def test_undeclared_environment_drift_does_not_enter_ab_comparison(self) -> None:
        repository = Repository(performance_ab_workload())
        self.addCleanup(repository.close)
        outcomes = []
        for variant_id, inherited_value in (
            ("baseline", "environment-a"),
            ("candidate", "environment-b"),
        ):
            with mock.patch.dict(
                os.environ,
                {"KILN_UNDECLARED_RUNNER_TEST": inherited_value},
                clear=False,
            ):
                outcomes.append(
                    run_module.run_qualification(
                        repository.workload_path,
                        variant_id=variant_id,
                        host_id="test-host",
                        model_path=repository.root / "fake-model",
                        output=(
                            repository.root
                            / f".qualification/receipts/env-{variant_id}.json"
                        ),
                        receipt_id=f"runner-env-{variant_id}-v1",
                        root=repository.root,
                        invocation=["qualification-runner-test", variant_id],
                        hooks=self.hooks(),
                        termination_grace_seconds=0.1,
                    )
                )
        hashes = [
            outcome.receipt["environment"]["runtime"][
                "execution_environment_sha256"
            ]
            for outcome in outcomes
        ]
        self.assertEqual(hashes[0], hashes[1])
        manifest = compare_module.load_committed_workload(
            repository.workload_path, root=repository.root
        )
        report = compare_module.compare_receipts(
            compare_module.load_validated_receipt(outcomes[0].receipt_path),
            compare_module.load_validated_receipt(outcomes[1].receipt_path),
            manifest=manifest,
        )
        self.assertTrue(report["ok"], report["errors"])

    def test_model_placeholder_requires_model_before_hooks_or_artifacts(self) -> None:
        repository = Repository(performance_ab_workload())
        self.addCleanup(repository.close)
        with self.assertRaisesRegex(run_module.QualificationRunError, "--model is required"):
            run_module.run_qualification(
                repository.workload_path,
                variant_id="baseline",
                host_id="test-host",
                receipt_id="runner-missing-model-v1",
                root=repository.root,
                invocation=["qualification-runner-test"],
                hooks=self.hooks(),
            )
        self.assertEqual(self.hook_calls, {"environment": 0, "model": 0, "network": 0})
        self.assertFalse((repository.root / ".qualification").exists())

    def test_initial_model_fingerprint_failure_retains_failed_receipt(self) -> None:
        repository = Repository(performance_ab_workload())
        self.addCleanup(repository.close)
        model_path = repository.root / "model"
        model_path.mkdir()

        def fail_model(_path: Path, _model_id: str | None) -> dict[str, Any]:
            self.hook_calls["model"] += 1
            raise run_module.ModelFingerprintError("synthetic fingerprint failure")

        hooks = run_module.RunnerHooks(
            capture_environment=self.fake_environment,
            fingerprint_model=fail_model,
            network_isolation=self.fake_network,
        )
        outcome = run_module.run_qualification(
            repository.workload_path,
            variant_id="baseline",
            host_id="test-host",
            model_path=model_path,
            model_id="fixture-model",
            receipt_id="runner-initial-model-failure-v1",
            root=repository.root,
            invocation=["qualification-runner-test", "initial-model-failure"],
            hooks=hooks,
        )

        self.assertEqual(outcome.exit_code, 1)
        self.assertEqual(
            self.hook_calls,
            {"environment": 0, "model": 1, "network": 1},
        )
        self.assertIsNone(outcome.receipt["model"])
        self.assertEqual(outcome.receipt["effective_config"], {})
        self.assertEqual(outcome.receipt["qualification"]["verdict"], "failed")
        self.assertEqual(
            [result["id"] for result in outcome.receipt["results"]],
            ["smoke-case"],
        )
        self.assertIn(
            "initial model fingerprint failed",
            outcome.receipt["results"][0]["details"],
        )
        self.assertEqual(
            {artifact["kind"] for artifact in outcome.receipt["artifacts"]},
            {"environment_probes", "effective_run_config"},
        )
        self.assertTrue(
            (
                repository.root
                / ".qualification/runs/runner-initial-model-failure-v1"
                / "run-config.json"
            ).is_file()
        )
        self.assert_valid(outcome, repository.root)

    def test_supervised_initial_fingerprint_failure_retains_raw_boundary(self) -> None:
        workload = performance_ab_workload()
        for variant in workload["variants"]:
            variant["backend"] = "cuda"
        repository = Repository(workload)
        self.addCleanup(repository.close)
        policy_relative = Path(
            "qualification/host-policies/"
            "rtx4090-laptop-wsl2-cgroup-pacing-v2.json"
        )
        policy_path = repository.root / policy_relative
        policy_path.parent.mkdir(parents=True)
        shutil.copyfile(
            QUALIFICATION_DIR.parents[1] / policy_relative,
            policy_path,
        )
        subprocess.run(
            ["git", "add", policy_relative.as_posix()],
            cwd=repository.root,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-qm", "add thermal policy"],
            cwd=repository.root,
            check=True,
        )
        model_path = repository.root / ".qualification/model"
        model_path.mkdir(parents=True)
        isolation = run_module.NetworkIsolation(
            "util-linux-unshare-user-net-pid-landlock-v1",
            ("/usr/bin/unshare", "--"),
        )
        hooks = run_module.RunnerHooks(
            capture_environment=self.fake_environment,
            fingerprint_model=self.fake_model,
            network_isolation=lambda _root: isolation,
        )

        def fail_supervised(
            _model_path: Path,
            _model_id: str | None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            kwargs["stdout_path"].write_bytes(b"")
            kwargs["stderr_path"].write_bytes(
                b'wsl2-scope: {"event":"failed","scope_removed":true}\n'
                b'wsl2-thermal: {"event":"complete","child_returncode":2}\n'
            )
            raise run_module.QualificationRunError(
                "thermal pacing pause exceeded 300 seconds"
            )

        with mock.patch.object(
            run_module,
            "DEFAULT_HOOKS",
            hooks,
        ), mock.patch.object(
            run_module,
            "_supervised_wsl_model_fingerprint",
            side_effect=fail_supervised,
        ):
            outcome = run_module.run_qualification(
                repository.workload_path,
                variant_id="baseline",
                host_id="test-host",
                model_path=model_path,
                model_id="fixture-model",
                receipt_id="runner-supervised-model-failure-v1",
                wsl2_thermal_policy=policy_relative,
                root=repository.root,
                invocation=["qualification-runner-test", "supervised-failure"],
                hooks=hooks,
            )

        self.assertEqual(outcome.exit_code, 1)
        self.assertEqual(self.hook_calls["environment"], 0)
        self.assertIsNone(outcome.receipt["model"])
        self.assertEqual(
            {artifact["kind"] for artifact in outcome.receipt["artifacts"]},
            {
                "effective_run_config",
                "environment_probes",
                "model_fingerprint_initial",
                "model_fingerprint_initial_wsl2_supervision",
                "wsl2_thermal_policy",
            },
        )
        self.assertIn(
            "thermal pacing pause exceeded 300 seconds",
            outcome.receipt["results"][0]["details"],
        )
        self.assert_valid(outcome, repository.root)

    def test_model_change_and_refingerprint_failure_downgrade_receipt(self) -> None:
        mutations = {
            "changed": (
                "from pathlib import Path; import sys; "
                "Path(sys.argv[1], 'config.json').write_text('{\"changed\":true}'); "
            ),
            "missing": (
                "from pathlib import Path; import sys; "
                "Path(sys.argv[1], 'config.json').unlink(); "
            ),
        }
        for mode, mutation in mutations.items():
            with self.subTest(mode=mode):
                workload = performance_ab_workload()
                baseline = next(
                    variant for variant in workload["variants"] if variant["id"] == "baseline"
                )
                baseline["cases"][0]["command"][2] = (
                    mutation + baseline["cases"][0]["command"][2]
                )
                repository = Repository(workload)
                model_temp = tempfile.TemporaryDirectory()
                try:
                    model_path = Path(model_temp.name) / "model"
                    model_path.mkdir()
                    (model_path / "model.safetensors").write_bytes(b"weights")
                    (model_path / "config.json").write_text("{}\n")
                    (model_path / "tokenizer.json").write_text("{}\n")
                    hooks = run_module.RunnerHooks(
                        capture_environment=self.fake_environment,
                        fingerprint_model=run_module.fingerprint_model,
                        network_isolation=self.fake_network,
                    )
                    outcome = run_module.run_qualification(
                        repository.workload_path,
                        variant_id="baseline",
                        host_id="test-host",
                        model_path=model_path,
                        receipt_id=f"runner-model-{mode}-v1",
                        root=repository.root,
                        invocation=["qualification-runner-test"],
                        hooks=hooks,
                        termination_grace_seconds=0.1,
                    )
                    self.assertEqual(outcome.exit_code, 1)
                    expected = (
                        "model fingerprint changed"
                        if mode == "changed"
                        else "final model fingerprint failed"
                    )
                    self.assertIn(expected, outcome.receipt["results"][0]["details"])
                    self.assertEqual(
                        receipt_module.load_receipt(outcome.receipt_path), outcome.receipt
                    )
                    self.assert_valid(outcome, repository.root)
                finally:
                    model_temp.cleanup()
                    repository.close()

    def test_contradictory_command_effective_config_fails_closed(self) -> None:
        declared = {"scheduler": {"max_batch": 4}}
        observed = {"scheduler": {"max_batch": 8}}
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", case_result_script(effective_config=observed)],
                protocol=command_protocol(),
                effective_config=declared,
            )
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 1)
        self.assertEqual(outcome.receipt["effective_config"], {})
        result = outcome.receipt["results"][0]
        self.assertIn("does not exactly match", result["details"])
        self.assertEqual(
            result["metrics"],
            [
                {
                    "aggregation": "exact",
                    "lower_is_better": False,
                    "name": "sample_value",
                    "unit": "items",
                    "value": 3.0,
                }
            ],
        )
        normalized_artifact = next(
            artifact
            for artifact in outcome.receipt["artifacts"]
            if artifact["kind"] == "case_result"
        )
        normalized = json.loads(
            (repository.root / normalized_artifact["path"]).read_text()
        )
        self.assertEqual(normalized["effective_config"], observed)
        self.assertEqual(normalized["status"], "failed")
        self.assert_valid(outcome, repository.root)

    def test_command_metric_definition_must_match_committed_policy(self) -> None:
        mutations = {
            "unit": {"metric_unit": "wrong-unit", "lower_is_better": False},
            "direction": {"metric_unit": "items", "lower_is_better": True},
        }
        for label, metric_options in mutations.items():
            with self.subTest(label=label):
                workload = performance_ab_workload()
                baseline = next(
                    variant for variant in workload["variants"] if variant["id"] == "baseline"
                )
                config = baseline["effective_config"]
                baseline["cases"][0]["command"][2] = (
                    "import os,sys; "
                    "assert sys.argv[1] == os.environ['TEST_MODEL_PATH']; "
                    + case_result_script(
                        effective_config=config,
                        **metric_options,
                    )
                )
                repository = Repository(workload)
                try:
                    outcome = run_module.run_qualification(
                        repository.workload_path,
                        variant_id="baseline",
                        host_id="test-host",
                        model_path=repository.root / "fake-model",
                        receipt_id=f"runner-policy-{label}-v1",
                        root=repository.root,
                        invocation=["qualification-runner-test"],
                        hooks=self.hooks(),
                        termination_grace_seconds=0.1,
                    )
                    self.assertEqual(outcome.exit_code, 1)
                    self.assertIn(
                        "does not match committed policy",
                        outcome.receipt["results"][0]["details"],
                    )
                    self.assertEqual(
                        receipt_module.load_receipt(outcome.receipt_path), outcome.receipt
                    )
                    self.assert_valid(outcome, repository.root)
                finally:
                    repository.close()

    def test_output_assertion_failure_emits_failed_receipt(self) -> None:
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", "print('NOT READY')"],
                assertions=[
                    {"stream": "stdout", "match": "required", "pattern": "^READY$"}
                ],
            )
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("required stdout pattern", outcome.receipt["results"][0]["details"])
        self.assert_valid(outcome, repository.root)

    def test_output_capture_limit_terminates_and_emits_bounded_failed_receipt(self) -> None:
        code = (
            "import os,time; "
            "os.write(1,b'x'*4096); os.write(2,b'y'*4096); time.sleep(30)"
        )
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", code],
                timeout_seconds=30,
            )
        )
        self.addCleanup(repository.close)
        started = time.monotonic()
        with mock.patch.object(run_module, "CASE_OUTPUT_LIMIT_BYTES", 1024):
            outcome = self.execute(repository)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("capture limit", outcome.receipt["results"][0]["details"])
        stdout_artifact = next(
            artifact
            for artifact in outcome.receipt["artifacts"]
            if artifact["kind"] == "case_stdout"
        )
        stdout_path = repository.root / stdout_artifact["path"]
        self.assertEqual(stdout_artifact["bytes"], 1024)
        self.assertEqual(stdout_path.stat().st_size, 1024)
        self.assertEqual(receipt_module.load_receipt(outcome.receipt_path), outcome.receipt)
        self.assert_valid(outcome, repository.root)

    def test_cumulative_output_budget_scales_with_execution_count(self) -> None:
        code = (
            "import os,time; "
            "os.write(1,b'x'*60); os.write(2,b'y'*60); time.sleep(30)"
        )
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", code],
                timeout_seconds=30,
                repetitions=2,
            )
        )
        self.addCleanup(repository.close)
        with (
            mock.patch.object(run_module, "CASE_OUTPUT_LIMIT_BYTES", 1000),
            mock.patch.object(run_module, "MAX_RUN_CAPTURE_BYTES", 200),
        ):
            outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 1)
        captured = sum(
            artifact["bytes"]
            for artifact in outcome.receipt["artifacts"]
            if artifact["kind"] in {"case_stdout", "case_stderr"}
        )
        self.assertLessEqual(captured, 200)
        run_config_artifact = next(
            artifact
            for artifact in outcome.receipt["artifacts"]
            if artifact["kind"] == "effective_run_config"
        )
        run_config = json.loads(
            (repository.root / run_config_artifact["path"]).read_text()
        )
        self.assertEqual(run_config["case_execution_count"], 2)
        self.assertEqual(run_config["per_stream_output_limit_bytes"], 50)

    def test_cumulative_structured_budget_discards_oversized_repeated_results(self) -> None:
        repetitions = 3
        structured_budget = 16 * 1024
        repository = Repository(
            environment_workload(
                [
                    sys.executable,
                    "-c",
                    case_result_script(details="x" * 10_000)
                    + "; import time; time.sleep(30)",
                ],
                protocol=command_protocol(),
                repetitions=repetitions,
                timeout_seconds=30,
            )
        )
        self.addCleanup(repository.close)
        started = time.monotonic()
        with mock.patch.object(
            run_module, "MAX_RUN_STRUCTURED_BYTES", structured_budget
        ):
            outcome = self.execute(repository)

        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(outcome.exit_code, 1)
        self.assertEqual(outcome.receipt["results"][0]["status"], "failed")
        details = outcome.receipt["results"][0]["details"]
        self.assertIsInstance(details, str)
        self.assertLessEqual(len(details), receipt_module.MAX_RESULT_DETAIL_CHARACTERS)
        self.assertIn("byte limit", details)
        self.assertFalse(
            any(
                artifact["kind"] == "command_case_result"
                for artifact in outcome.receipt["artifacts"]
            )
        )
        normalized = [
            artifact
            for artifact in outcome.receipt["artifacts"]
            if artifact["kind"] == "case_result"
        ]
        self.assertEqual(len(normalized), repetitions)
        self.assertLessEqual(
            sum(artifact["bytes"] for artifact in normalized), structured_budget
        )
        self.assertEqual(
            list(
                (repository.root / ".qualification/runs/runner-test-receipt-v1").glob(
                    "cases/*/command-result.json"
                )
            ),
            [],
        )
        run_config_artifact = next(
            artifact
            for artifact in outcome.receipt["artifacts"]
            if artifact["kind"] == "effective_run_config"
        )
        run_config = json.loads(
            (repository.root / run_config_artifact["path"]).read_text()
        )
        self.assertEqual(
            run_config["per_case_result_limit_bytes"],
            structured_budget // (2 * repetitions),
        )
        self.assertEqual(run_config["max_run_structured_bytes"], structured_budget)
        self.assert_valid(outcome, repository.root)
        self.assertEqual(
            run_config["case_environment_policy"],
            run_module.CASE_ENVIRONMENT_POLICY,
        )
        self.assertIn("case_base_environment", run_config)
        self.assertNotIn("inherited_environment", run_config)
        self.assertTrue(
            all("process_environment" not in case for case in run_config["cases"])
        )
        self.assert_valid(outcome, repository.root)

    def test_dirty_tree_is_side_effect_free_preflight_failure(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "print('must not run')"])
        )
        self.addCleanup(repository.close)
        (repository.root / "Cargo.toml").write_text("[workspace]\nmembers = ['dirty']\n")
        with self.assertRaisesRegex(run_module.QualificationRunError, "must be clean"):
            self.execute(repository)
        self.assertEqual(self.hook_calls, {"environment": 0, "model": 0, "network": 0})
        self.assertFalse((repository.root / ".qualification").exists())
        self.assertFalse((repository.root / "qualification/receipts").exists())

    def test_existing_output_is_side_effect_free_preflight_failure(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "print('must not run')"])
        )
        self.addCleanup(repository.close)
        output = repository.root / "qualification/receipts/existing-receipt.json"
        output.parent.mkdir(parents=True)
        output.write_text("do not replace\n")
        with self.assertRaisesRegex(run_module.QualificationRunError, "refusing to overwrite"):
            self.execute(repository, output=output)
        self.assertEqual(output.read_text(), "do not replace\n")
        self.assertEqual(self.hook_calls, {"environment": 0, "model": 0, "network": 0})
        self.assertFalse((repository.root / ".qualification").exists())

    def test_output_rejects_source_traversal_and_symlinks_before_side_effects(self) -> None:
        outputs = {
            "source": Path("new-source-file.txt"),
            "traversal": Path("qualification/receipts/../../scripts/new-source.json"),
            "outside": Path("../outside-receipt.json"),
            "raw-collision": Path(
                ".qualification/runs/runner-test-receipt-v1/environment.json"
            ),
        }
        for label, output in outputs.items():
            with self.subTest(label=label):
                repository = Repository(
                    environment_workload([sys.executable, "-c", "print('must not run')"])
                )
                try:
                    with self.assertRaises(run_module.QualificationRunError):
                        self.execute(repository, output=output)
                    self.assertEqual(
                        self.hook_calls,
                        {"environment": 0, "model": 0, "network": 0},
                    )
                    self.assertFalse((repository.root / ".qualification").exists())
                finally:
                    repository.close()

        repository = Repository(
            environment_workload([sys.executable, "-c", "print('must not run')"])
        )
        self.addCleanup(repository.close)
        evidence_root = repository.root / "qualification/receipts"
        evidence_root.mkdir(parents=True)
        (evidence_root / "escape").symlink_to(repository.root / "qualification/workloads")
        with self.assertRaisesRegex(run_module.QualificationRunError, "symlinks"):
            self.execute(
                repository,
                output=Path("qualification/receipts/escape/receipt.json"),
            )
        self.assertEqual(self.hook_calls, {"environment": 0, "model": 0, "network": 0})
        self.assertFalse((repository.root / ".qualification").exists())

    def test_run_directory_symlink_is_rejected_before_hooks(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "print('must not run')"])
        )
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        self.addCleanup(repository.close)
        (repository.root / ".qualification").symlink_to(Path(outside.name))
        with self.assertRaisesRegex(run_module.QualificationRunError, "symlinks"):
            self.execute(repository)
        self.assertEqual(self.hook_calls, {"environment": 0, "model": 0, "network": 0})
        self.assertEqual(list(Path(outside.name).iterdir()), [])

    def test_optional_null_selected_variable_fails_before_side_effects(self) -> None:
        definitions = [variable("maybe", "string", required=False, default=None)]
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", "print('must not run')", "${maybe}"],
                variables=definitions,
            )
        )
        self.addCleanup(repository.close)
        with self.assertRaisesRegex(run_module.QualificationRunError, "no value or default"):
            self.execute(repository)
        self.assertEqual(self.hook_calls, {"environment": 0, "model": 0, "network": 0})
        self.assertFalse((repository.root / ".qualification").exists())

    def test_nonfinite_termination_grace_fails_before_side_effects(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "print('must not run')"])
        )
        self.addCleanup(repository.close)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                run_module.QualificationRunError, "finite and non-negative"
            ):
                run_module.run_qualification(
                    repository.workload_path,
                    variant_id="rocm",
                    host_id="test-host",
                    receipt_id="runner-invalid-grace-v1",
                    root=repository.root,
                    invocation=["qualification-runner-test"],
                    hooks=self.hooks(),
                    termination_grace_seconds=value,
                )
        with self.assertRaisesRegex(
            run_module.QualificationRunError,
            f"at most {run_module.MAX_TERMINATION_GRACE_SECONDS:g}",
        ):
            run_module.run_qualification(
                repository.workload_path,
                variant_id="rocm",
                host_id="test-host",
                receipt_id="runner-excessive-grace-v1",
                root=repository.root,
                invocation=["qualification-runner-test"],
                hooks=self.hooks(),
                termination_grace_seconds=(
                    run_module.MAX_TERMINATION_GRACE_SECONDS + 0.01
                ),
            )
        self.assertEqual(self.hook_calls, {"environment": 0, "model": 0, "network": 0})
        self.assertFalse((repository.root / ".qualification").exists())

    def test_workload_runtime_bounds_fail_before_hooks(self) -> None:
        workloads = {}
        value = environment_workload([sys.executable, "-c", "print('unused')"])
        value["determinism"]["repetitions"] = run_module.MAX_REPETITIONS + 1
        workloads["repetitions"] = value
        value = environment_workload([sys.executable, "-c", "print('unused')"])
        value["variants"][0]["cases"][0]["timeout_seconds"] = 10**400
        workloads["timeout"] = value
        value = environment_workload([sys.executable, "-c", "print('unused')"])
        value["determinism"]["repetitions"] = 4
        value["variants"][0]["cases"][0]["timeout_seconds"] = (
            run_module.MAX_CASE_TIMEOUT_SECONDS
        )
        workloads["budget"] = value

        for label, workload in workloads.items():
            with self.subTest(label=label):
                repository = Repository(workload)
                try:
                    with self.assertRaisesRegex(
                        run_module.QualificationRunError, "invalid workload"
                    ):
                        self.execute(repository)
                    self.assertEqual(
                        self.hook_calls,
                        {"environment": 0, "model": 0, "network": 0},
                    )
                    self.assertFalse((repository.root / ".qualification").exists())
                finally:
                    repository.close()

    def test_missing_device_still_emits_failed_receipt(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "print('case still runs')"])
        )
        self.addCleanup(repository.close)

        def missing_device(
            backend: str, host_id: str, root: Path
        ) -> run_module.EnvironmentCapture:
            capture = self.fake_environment(backend, host_id, root)
            capture.environment["device"]["name"] = "unavailable"
            return capture

        hooks = run_module.RunnerHooks(
            capture_environment=missing_device,
            fingerprint_model=self.fake_model,
            network_isolation=self.fake_network,
        )
        outcome = run_module.run_qualification(
            repository.workload_path,
            variant_id="rocm",
            host_id="test-host",
            receipt_id="runner-missing-device-v1",
            root=repository.root,
            invocation=["qualification-runner-test"],
            hooks=hooks,
        )
        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("not detected", outcome.receipt["results"][0]["details"])
        self.assert_valid(outcome, repository.root)

    def test_extreme_repetition_mean_fails_without_nonfinite_json(self) -> None:
        maximum = sys.float_info.max
        repository = Repository(
            environment_workload(
                [
                    sys.executable,
                    "-c",
                    case_result_script(metric_value=maximum),
                ],
                protocol=command_protocol(),
                repetitions=3,
            )
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("could not be aggregated", outcome.receipt["results"][0]["details"])
        json.dumps(outcome.receipt, allow_nan=False)
        self.assert_valid(outcome, repository.root)

    def test_repetition_mean_preserves_subnormal_and_large_integer(self) -> None:
        values = {
            "subnormal": 5e-324,
            "large-integer": 9007199254740993,
        }
        for label, expected in values.items():
            with self.subTest(label=label):
                repository = Repository(
                    environment_workload(
                        [
                            sys.executable,
                            "-c",
                            case_result_script(metric_value=expected),
                        ],
                        protocol=command_protocol(),
                        repetitions=2,
                    )
                )
                try:
                    outcome = self.execute(
                        repository, receipt_id=f"runner-mean-{label}-v1"
                    )
                    self.assertEqual(outcome.exit_code, 0)
                    metric = outcome.receipt["results"][0]["metrics"][0]
                    self.assertEqual(metric["value"], expected)
                    self.assertIs(type(metric["value"]), type(expected))
                    self.assert_valid(outcome, repository.root)
                finally:
                    repository.close()

    def test_repetition_metric_definition_is_data_independent(self) -> None:
        equal_script = case_result_script(metric_value=2.0)
        changing_script = (
            "import json,os,pathlib; p=pathlib.Path('.qualification/counter'); "
            "value=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(value)); "
            "result={'schema_version':1,'case_id':'smoke-case','status':'passed',"
            "'duration_seconds':0.01,'effective_config':{},'metrics':[{'name':'sample_value',"
            "'value':value,'unit':'items','aggregation':'exact','lower_is_better':False}],"
            "'tolerances':[],'details':None}; "
            "open(os.environ['KILN_QUALIFICATION_CASE_RESULT'],'w').write(json.dumps(result))"
        )
        definitions: list[dict[str, Any]] = []
        for label, script in (("equal", equal_script), ("unequal", changing_script)):
            repository = Repository(
                environment_workload(
                    [sys.executable, "-c", script],
                    protocol=command_protocol(),
                    repetitions=2,
                )
            )
            try:
                outcome = self.execute(
                    repository, receipt_id=f"runner-{label}-repetition-v1"
                )
                self.assertEqual(outcome.exit_code, 0)
                metric = outcome.receipt["results"][0]["metrics"][0]
                definitions.append(
                    {
                        key: metric[key]
                        for key in ("name", "unit", "aggregation", "lower_is_better")
                    }
                )
                self.assertEqual(metric["aggregation"], "mean_of_2_exact")
            finally:
                repository.close()
        self.assertEqual(definitions[0], definitions[1])

    def test_repeated_runner_exit_code_must_agree(self) -> None:
        code = (
            "import pathlib,sys; p=pathlib.Path('.qualification/exit-counter'); "
            "value=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(value)); "
            "raise SystemExit(0 if value == 1 else 2)"
        )
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", code],
                repetitions=2,
                expected_exit_codes=[0, 2],
            )
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("exit_code differed", outcome.receipt["results"][0]["details"])
        self.assert_valid(outcome, repository.root)

    def test_repeated_runner_assertion_failures_are_summed(self) -> None:
        code = (
            "import pathlib; p=pathlib.Path('.qualification/assert-counter'); "
            "value=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(value)); "
            "print('BAD' if value == 1 else 'GOOD')"
        )
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", code],
                repetitions=2,
                assertions=[
                    {"stream": "stdout", "match": "forbidden", "pattern": "BAD"}
                ],
            )
        )
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 1)
        metrics = {
            metric["name"]: metric for metric in outcome.receipt["results"][0]["metrics"]
        }
        self.assertEqual(metrics["output_assertion_failures"]["value"], 1)
        self.assertEqual(metrics["output_assertion_failures"]["aggregation"], "sum")
        self.assertEqual(metrics["case_pass"]["value"], 0)
        self.assertIs(type(metrics["case_pass"]["value"]), int)
        self.assert_valid(outcome, repository.root)

    def test_source_mutation_during_case_downgrades_receipt(self) -> None:
        code = "from pathlib import Path; Path('Cargo.toml').write_text('[workspace]\\nmembers=[]\\n# changed\\n')"
        repository = Repository(environment_workload([sys.executable, "-c", code]))
        self.addCleanup(repository.close)
        outcome = self.execute(repository)
        self.assertEqual(outcome.exit_code, 1)
        self.assertIn("source tree changed", outcome.receipt["results"][0]["details"])
        case_pass = next(
            metric
            for metric in outcome.receipt["results"][0]["metrics"]
            if metric["name"] == "case_pass"
        )
        self.assertEqual(case_pass["value"], 0)
        self.assert_valid(outcome, repository.root)

    def test_number_values_have_canonical_float_identity(self) -> None:
        definition = variable("ratio", "number", required=False, default=1)
        definition["constraints"]["allowed_values"] = [1]
        repository = Repository(
            environment_workload(
                [sys.executable, "-c", "print('ratio', __import__('sys').argv[1])", "${ratio}"],
                variables=[definition],
            )
        )
        self.addCleanup(repository.close)
        implicit = self.execute(repository, receipt_id="runner-number-default-v1")
        explicit = self.execute(
            repository,
            receipt_id="runner-number-explicit-v1",
            assignments=["ratio=1"],
        )
        self.assertEqual(
            implicit.receipt["workload"]["parameters"],
            {"variant_id": "rocm", "ratio": 1.0},
        )
        self.assertEqual(
            implicit.receipt["workload"]["parameters"],
            explicit.receipt["workload"]["parameters"],
        )
        self.assertIs(type(implicit.receipt["workload"]["parameters"]["ratio"]), float)
        self.assert_valid(implicit, repository.root)
        self.assert_valid(explicit, repository.root)
        zero = variable("zero", "number", required=False, default=0)
        zero_values = [
            run_module.resolve_variables(
                [zero], assignment, selected_references={"zero"}
            )["zero"]
            for assignment in ([], ["zero=-0"], ["zero=-0.0"])
        ]
        self.assertEqual(zero_values, [0.0, 0.0, 0.0])
        self.assertTrue(all(math.copysign(1.0, value) == 1.0 for value in zero_values))
        integer = variable("count", "integer", required=True, default=None)
        with self.assertRaisesRegex(run_module.QualificationRunError, "base-10 integer"):
            run_module.resolve_variables(
                [integer], ["count=1.0"], selected_references={"count"}
            )

    def test_integer_limit_matches_strict_receipt_reload(self) -> None:
        definition = variable("count", "integer", required=True, default=None)
        workload = environment_workload(
            [sys.executable, "-c", "print(len(__import__('sys').argv[1]))", "${count}"],
            variables=[definition],
        )
        accepted = "1" + "0" * (run_module.JSON_INTEGER_MAX_DIGITS - 1)
        repository = Repository(workload)
        try:
            outcome = self.execute(repository, assignments=[f"count={accepted}"])
            self.assertEqual(outcome.exit_code, 0)
            reloaded = receipt_module.load_receipt(outcome.receipt_path)
            self.assertEqual(reloaded, outcome.receipt)
            self.assertEqual(
                len(str(reloaded["workload"]["parameters"]["count"])),
                run_module.JSON_INTEGER_MAX_DIGITS,
            )
        finally:
            repository.close()

        rejected = "1" + "0" * run_module.JSON_INTEGER_MAX_DIGITS
        repository = Repository(workload)
        try:
            with self.assertRaisesRegex(run_module.QualificationRunError, "4096 digits"):
                self.execute(repository, assignments=[f"count={rejected}"])
            self.assertEqual(
                self.hook_calls,
                {"environment": 1, "model": 0, "network": 1},
            )
            self.assertFalse((repository.root / ".qualification").exists())
        finally:
            repository.close()

    def test_strict_command_numbers_reject_overflow_underflow_and_huge_integer(self) -> None:
        numeric_values = {
            "overflow": "1e400",
            "underflow": "1e-4000",
            "rounded-boundary": "8.9999999999999999",
            "rounded-integer": "9007199254740993.0",
            "rounded-subnormal": "4e-324",
            "huge-integer": "1" + "0" * run_module.JSON_INTEGER_MAX_DIGITS,
        }
        for label, numeric in numeric_values.items():
            with self.subTest(label=label):
                raw = (
                    "{\"schema_version\":1,\"case_id\":\"smoke-case\","
                    "\"status\":\"passed\",\"duration_seconds\":0.01,"
                    "\"effective_config\":{},\"metrics\":[{\"name\":\"sample_value\","
                    f"\"value\":{numeric},\"unit\":\"items\",\"aggregation\":\"exact\","
                    "\"lower_is_better\":false}],\"tolerances\":[],\"details\":null}"
                )
                code = (
                    "import os; "
                    f"raw={raw!r}; "
                    "open(os.environ['KILN_QUALIFICATION_CASE_RESULT'],'w').write(raw)"
                )
                repository = Repository(
                    environment_workload(
                        [sys.executable, "-c", code], protocol=command_protocol()
                    )
                )
                try:
                    outcome = self.execute(
                        repository, receipt_id=f"runner-{label}-number-v1"
                    )
                    self.assertEqual(outcome.exit_code, 1)
                    self.assertEqual(outcome.receipt["results"][0]["status"], "failed")
                    json.dumps(outcome.receipt, allow_nan=False)
                    self.assert_valid(outcome, repository.root)
                finally:
                    repository.close()

    def test_command_result_bytes_require_plain_utf8(self) -> None:
        payloads = {
            "invalid-utf8": b"\xff{}",
            "utf8-bom": b"\xef\xbb\xbf{}",
            "utf16": "{}".encode("utf-16"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in payloads.items():
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.json"
                    path.write_bytes(payload)
                    with self.assertRaisesRegex(
                        run_module.CaseResultError,
                        "cannot load command case result",
                    ):
                        run_module.load_case_result(
                            path,
                            expected_case_id="smoke-case",
                            declared_metrics={"sample_value"},
                        )

    def test_case_result_schema_matches_closed_runner_contract(self) -> None:
        schema = json.loads(
            (QUALIFICATION_DIR.parents[1] / "qualification/schema/case-result-v1.schema.json").read_text()
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), run_module.CASE_RESULT_KEYS)
        self.assertEqual(set(schema["properties"]), run_module.CASE_RESULT_KEYS)
        self.assertEqual(
            schema["properties"]["details"]["oneOf"][0]["maxLength"],
            receipt_module.MAX_RESULT_DETAIL_CHARACTERS,
        )
        config = schema["$defs"]["configObject"]
        self.assertEqual(config["type"], "object")
        self.assertNotIn("array", json.dumps(config))

    @unittest.skipUnless(
        sys.platform == "linux", "network namespaces are Linux-specific"
    )
    def test_production_network_wrapper_has_only_loopback(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "print('unused')"])
        )
        self.addCleanup(repository.close)
        try:
            isolation = run_module.establish_network_isolation(repository.root)
        except run_module.QualificationRunError as exc:
            self.skipTest(f"Linux network namespaces unavailable on this host: {exc}")
        probe = (
            "import json,socket; "
            "print(json.dumps([name for _,name in socket.if_nameindex()]))"
        )
        completed = subprocess.run(
            [*isolation.argv_prefix, sys.executable, "-c", probe],
            cwd=repository.root,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout), ["lo"])

    @unittest.skipUnless(
        sys.platform == "linux", "PID namespaces are Linux-specific"
    )
    def test_pid_namespace_kills_setsid_escape_on_exit_and_timeout(self) -> None:
        repository = Repository(
            environment_workload([sys.executable, "-c", "print('unused')"])
        )
        self.addCleanup(repository.close)
        try:
            isolation = run_module.establish_network_isolation(repository.root)
        except run_module.QualificationRunError as exc:
            self.skipTest(f"Linux PID namespaces unavailable on this host: {exc}")
        output_root = repository.root / ".qualification/containment"
        output_root.mkdir(parents=True)

        for mode, delay, timeout in (("exit", 0.5, 5), ("timeout", 1.5, 1)):
            with self.subTest(mode=mode):
                marker = output_root / f"{mode}-escaped.txt"
                child_code = (
                    "import os,signal,time,pathlib; "
                    "os.setsid(); "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    "signal.signal(signal.SIGHUP,signal.SIG_IGN); "
                    f"time.sleep({delay}); pathlib.Path({str(marker)!r}).write_text('escaped'); "
                    "time.sleep(0.5)"
                )
                parent_tail = "time.sleep(30)" if mode == "timeout" else "pass"
                parent_code = (
                    "import subprocess,sys,time; "
                    f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                    + parent_tail
                )
                execution = run_module.execute_argv(
                    [*isolation.argv_prefix, sys.executable, "-c", parent_code],
                    cwd=repository.root,
                    environment=dict(os.environ),
                    stdout_path=output_root / f"{mode}-stdout.log",
                    stderr_path=output_root / f"{mode}-stderr.log",
                    timeout_seconds=timeout,
                    termination_grace_seconds=0.1,
                )
                if mode == "exit":
                    self.assertEqual(execution.returncode, 0)
                    self.assertFalse(execution.timed_out)
                else:
                    self.assertTrue(execution.timed_out)
                time.sleep(delay + 0.25)
                self.assertFalse(marker.exists(), f"setsid child escaped after {mode}")


def platform_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def process_running(pid: int) -> bool:
    status = Path(f"/proc/{pid}/stat")
    try:
        fields = status.read_text().split()
    except FileNotFoundError:
        return False
    return len(fields) > 2 and fields[2] != "Z"


if __name__ == "__main__":
    unittest.main()
