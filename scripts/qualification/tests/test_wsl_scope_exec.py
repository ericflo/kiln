from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_wsl_scope_exec",
    QUALIFICATION_DIR / "wsl_scope_exec.py",
)
assert SPEC is not None and SPEC.loader is not None
scope = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scope
SPEC.loader.exec_module(scope)


class WslScopeExecTests(unittest.TestCase):
    def pacing_policy(self) -> object:
        return scope.wsl_thermal_exec.load_policy(
            QUALIFICATION_DIR.parents[1]
            / "qualification/host-policies/"
            "rtx4090-laptop-wsl2-cgroup-pacing-v2.json"
        )

    def test_missing_runtime_unit_directory_is_created_with_private_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            systemd_directory = runtime / "systemd"
            systemd_directory.mkdir(mode=0o700)

            unit_directory = scope._ensure_runtime_unit_directory(runtime)

            self.assertEqual(unit_directory, systemd_directory / "user")
            self.assertTrue(stat.S_ISDIR(unit_directory.lstat().st_mode))
            self.assertEqual(stat.S_IMODE(unit_directory.lstat().st_mode), 0o700)
            self.assertEqual(
                scope._ensure_runtime_unit_directory(runtime),
                unit_directory,
            )

    def test_runtime_unit_directory_rejects_symlink_and_writable_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            systemd_directory = runtime / "systemd"
            systemd_directory.mkdir(mode=0o700)
            target = runtime / "target"
            target.mkdir(mode=0o700)
            (systemd_directory / "user").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(scope.ScopeExecError, "unsafe"):
                scope._ensure_runtime_unit_directory(runtime)

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            systemd_directory = runtime / "systemd"
            systemd_directory.mkdir(mode=0o777)
            systemd_directory.chmod(0o777)
            with self.assertRaisesRegex(scope.ScopeExecError, "unsafe"):
                scope._ensure_runtime_unit_directory(runtime)

    def test_runtime_bus_units_are_no_clobber_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bus.socket"
            scope._atomic_runtime_unit(path, "exact\n")
            self.assertEqual(path.read_text(encoding="ascii"), "exact\n")
            scope._atomic_runtime_unit(path, "exact\n")
            with self.assertRaisesRegex(scope.ScopeExecError, "refusing to replace"):
                scope._atomic_runtime_unit(path, "different\n")

    def test_namespace_command_contract_is_closed(self) -> None:
        command = [
            "/usr/bin/unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",
            "--mount",
            "--mount-proc=/proc",
            scope.sys.executable,
            str(QUALIFICATION_DIR / "linux_namespace_exec.py"),
            "--",
            "/bin/true",
        ]
        with mock.patch.dict(
            os.environ,
            {
                scope.wsl_platform.NETWORK_ISOLATION_ENV: (
                    "util-linux-unshare-user-net-pid-landlock-v1"
                )
            },
            clear=True,
        ):
            scope._verify_network_command(command)
            with self.assertRaisesRegex(scope.ScopeExecError, "namespace boundary"):
                scope._verify_network_command(
                    [item for item in command if item != "--pid"]
                )
            with self.assertRaisesRegex(scope.ScopeExecError, "namespace boundary"):
                without_helper = list(command)
                without_helper[10] = "/tmp/untrusted.py"
                scope._verify_network_command(without_helper)

    def test_resource_arguments_reject_unenforceable_values(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                scope.parse_args(
                    [
                        "--memory-max-bytes",
                        str(64 * 1024 * 1024),
                        "--pids-max",
                        "512",
                        "--cpu-quota-percent",
                        "101",
                        "--runtime-max-seconds",
                        "10",
                        "--",
                        "/bin/true",
                    ]
                )
        parsed = scope.parse_args(
            [
                "--memory-max-bytes",
                str(10 * 1024 * 1024 * 1024),
                "--pids-max",
                "512",
                "--cpu-quota-percent",
                "50",
                "--runtime-max-seconds",
                "30",
                "--",
                "/bin/true",
            ]
        )
        self.assertEqual(parsed.cpu_quota_percent, 50)
        unthrottled = scope.parse_args(
            [
                "--memory-max-bytes",
                str(10 * 1024 * 1024 * 1024),
                "--pids-max",
                "512",
                "--cpu-quota-percent",
                "0",
                "--runtime-max-seconds",
                "30",
                "--",
                "/bin/true",
            ]
        )
        self.assertEqual(unthrottled.cpu_quota_percent, 0)

    def test_empty_host_build_inventory_is_accepted(self) -> None:
        with mock.patch.object(scope.Path, "iterdir", return_value=[]):
            self.assertEqual(scope._active_host_builds(), [])

    def test_cgroup_freeze_waits_for_kernel_state_and_times_out_closed(self) -> None:
        cgroup = Path("/fixture/cgroup")
        with mock.patch.object(scope, "_write") as write, mock.patch.object(
            scope,
            "_read",
            return_value="1",
        ), mock.patch.object(
            scope,
            "_events",
            return_value={"frozen": 1},
        ):
            scope._set_frozen(cgroup, True)
        write.assert_called_once_with(cgroup / "cgroup.freeze", "1")

        with mock.patch.object(scope, "_write"), mock.patch.object(
            scope,
            "_read",
            return_value="1",
        ), mock.patch.object(
            scope,
            "_events",
            return_value={"frozen": 0},
        ), mock.patch.object(
            scope.time,
            "monotonic",
            side_effect=[0.0, 2.0],
        ):
            with self.assertRaisesRegex(scope.ScopeExecError, "did not freeze"):
                scope._set_frozen(cgroup, True)

    def test_thermal_pacing_starts_at_threshold_and_requires_stable_resume(
        self,
    ) -> None:
        policy = self.pacing_policy()
        self.assertEqual(policy.pacing.host_resume_millicelsius, 75_050)
        state = scope.ThermalPacingState(policy)
        sample = scope.wsl_thermal_exec.ThermalSample
        self.assertFalse(state.observe(sample(0.0, 79_000, 74_000), 1.0))
        self.assertTrue(state.observe(sample(0.0, 80_000, 74_000), 10.0))
        self.assertTrue(state.observe(sample(0.0, 72_000, 70_000), 11.0))
        self.assertTrue(state.observe(sample(0.0, 72_000, 70_000), 12.0))
        self.assertFalse(state.observe(sample(0.0, 72_000, 70_000), 13.0))
        record = state.record()
        self.assertEqual(record["pause_count"], 1)
        self.assertEqual(record["completed_pause_count"], 1)
        self.assertEqual(record["total_pause_seconds"], 3.0)
        self.assertFalse(record["active"])

    def test_thermal_pacing_timeout_and_hard_limit_fail_closed(self) -> None:
        policy = self.pacing_policy()
        sample = scope.wsl_thermal_exec.ThermalSample
        state = scope.ThermalPacingState(policy)
        self.assertTrue(state.observe(sample(0.0, 80_000, 60_000), 10.0))
        with self.assertRaisesRegex(
            scope.ThermalPacingTimeoutError,
            "exceeded 300 seconds",
        ):
            state.observe(sample(0.0, 73_000, 60_000), 310.0)

        state = scope.ThermalPacingState(policy)
        with self.assertRaisesRegex(scope.ScopeExecError, "hard limit"):
            state.observe(sample(0.0, 95_000, 60_000), 1.0)

    def test_thermal_timeout_cleanup_interrupts_leaves_before_parents(
        self,
    ) -> None:
        cgroup = Path("/fixture/cgroup")
        process_layers = [
            ((10, "S", 1), (20, "S", 10), (30, "S", 20)),
            ((10, "S", 1), (20, "S", 10)),
            ((10, "S", 1),),
            (),
        ]
        with mock.patch.object(Path, "is_dir", return_value=True), mock.patch.object(
            scope,
            "_set_frozen",
        ) as set_frozen, mock.patch.object(
            scope,
            "_cgroup_process_members",
            side_effect=process_layers,
        ), mock.patch.object(
            scope,
            "_signal_cgroup_member",
            return_value=True,
        ) as signal_member, mock.patch.object(
            scope.time,
            "sleep",
        ):
            self.assertTrue(
                scope._terminate_scope_leaf_first(cgroup, grace_seconds=1.0)
            )

        set_frozen.assert_called_once_with(cgroup, False)
        self.assertEqual(
            signal_member.call_args_list,
            [
                mock.call(cgroup, 30, scope.signal.SIGINT),
                mock.call(cgroup, 20, scope.signal.SIGINT),
                mock.call(cgroup, 10, scope.signal.SIGINT),
            ],
        )

    def test_thermal_timeout_cleanup_rechecks_pidfd_membership(self) -> None:
        cgroup = Path("/fixture/cgroup")
        with mock.patch.object(
            scope.os,
            "pidfd_open",
            return_value=91,
        ) as pidfd_open, mock.patch.object(
            scope,
            "_cgroup_process_members",
            return_value=((30, "S", 20),),
        ), mock.patch.object(
            scope.signal,
            "pidfd_send_signal",
        ) as pidfd_signal, mock.patch.object(
            scope.os,
            "close",
        ) as close:
            self.assertTrue(
                scope._signal_cgroup_member(cgroup, 30, scope.signal.SIGINT)
            )

        pidfd_open.assert_called_once_with(30)
        pidfd_signal.assert_called_once_with(91, scope.signal.SIGINT)
        close.assert_called_once_with(91)

    def test_thermal_timeout_cleanup_accepts_scope_removal_race(self) -> None:
        cgroup = Path("/fixture/cgroup")
        with mock.patch.object(
            Path,
            "is_dir",
            side_effect=[True, False],
        ), mock.patch.object(
            scope,
            "_set_frozen",
        ), mock.patch.object(
            scope,
            "_cgroup_process_members",
            side_effect=scope.ScopeExecError("scope disappeared"),
        ):
            self.assertTrue(
                scope._terminate_scope_leaf_first(cgroup, grace_seconds=1.0)
            )

    def test_thermal_timeout_cleanup_accepts_unfreeze_removal_race(self) -> None:
        cgroup = Path("/fixture/cgroup")
        with mock.patch.object(
            Path,
            "is_dir",
            side_effect=[True, False],
        ), mock.patch.object(
            scope,
            "_set_frozen",
            side_effect=scope.ScopeExecError("scope disappeared"),
        ):
            self.assertTrue(
                scope._terminate_scope_leaf_first(cgroup, grace_seconds=1.0)
            )

    def test_thermal_timeout_cleanup_remains_force_bounded(self) -> None:
        cgroup = Path("/fixture/cgroup")
        with mock.patch.object(Path, "is_dir", return_value=True), mock.patch.object(
            scope,
            "_set_frozen",
        ), mock.patch.object(
            scope,
            "_cgroup_process_members",
            return_value=((10, "S", 1),),
        ), mock.patch.object(
            scope,
            "_signal_cgroup_member",
            return_value=True,
        ) as signal_member, mock.patch.object(
            scope.time,
            "monotonic",
            side_effect=[0.0, 0.0, 0.0, 2.0],
        ), mock.patch.object(
            scope.time,
            "sleep",
        ):
            self.assertFalse(
                scope._terminate_scope_leaf_first(cgroup, grace_seconds=1.0)
            )

        signal_member.assert_called_once_with(cgroup, 10, scope.signal.SIGINT)

    def test_thermal_pacing_writer_records_exact_read_only_transitions(self) -> None:
        policy = self.pacing_policy()
        sample = scope.wsl_thermal_exec.ThermalSample
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            writer = scope.ThermalPacingEventWriter(path, policy)
            state = scope.ThermalPacingState(policy)
            scope._observe_thermal_pacing(
                state,
                writer,
                sample(10.0, 80_000, 74_000),
            )
            scope._observe_thermal_pacing(
                state,
                writer,
                sample(11.0, 74_000, 69_000),
            )
            scope._observe_thermal_pacing(
                state,
                writer,
                sample(12.0, 74_000, 69_000),
            )
            scope._observe_thermal_pacing(
                state,
                writer,
                sample(13.0, 74_000, 69_000),
            )
            writer.close()

            records = [
                json.loads(line)
                for line in path.read_text(encoding="ascii").splitlines()
            ]
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            self.assertEqual(
                [(record["sequence"], record["transition"]) for record in records],
                [(0, "started"), (1, "completed")],
            )
            self.assertEqual(
                {record["schema"] for record in records},
                {scope.THERMAL_PACING_EVENT_SCHEMA},
            )
            self.assertEqual(
                {record["policy_sha256"] for record in records},
                {policy.content_sha256},
            )
            self.assertEqual(
                {record["pause_index"] for record in records},
                {1},
            )
            self.assertEqual(records[0]["duration_seconds"], 0.0)
            self.assertEqual(records[1]["duration_seconds"], 3.0)
            self.assertFalse(records[1]["active"])

    def test_thermal_pacing_writer_rejects_invalid_transition_values(self) -> None:
        policy = self.pacing_policy()
        with tempfile.TemporaryDirectory() as directory:
            writer = scope.ThermalPacingEventWriter(
                Path(directory) / "events.jsonl",
                policy,
            )
            try:
                with self.assertRaisesRegex(scope.ScopeExecError, "invalid"):
                    writer.transition(
                        active=True,
                        pause_index=0,
                        started_monotonic_seconds=1.0,
                        sample=scope.wsl_thermal_exec.ThermalSample(
                            1.0,
                            80_000,
                            74_000,
                        ),
                    )
            finally:
                writer.close()

    def test_thermal_pacing_freezes_before_publishing_start(self) -> None:
        policy = self.pacing_policy()
        state = scope.ThermalPacingState(policy)
        order: list[str] = []
        writer = mock.Mock()
        writer.transition.side_effect = lambda **_kwargs: order.append("write")

        scope._observe_thermal_pacing(
            state,
            writer,
            scope.wsl_thermal_exec.ThermalSample(
                10.0,
                80_000,
                74_000,
            ),
            before_start=lambda: order.append("freeze"),
        )

        self.assertEqual(order, ["freeze", "write"])

    def test_thermal_pacing_policy_requires_both_outer_hash_bindings(self) -> None:
        policy_path = (
            QUALIFICATION_DIR.parents[1]
            / "qualification/host-policies/"
            "rtx4090-laptop-wsl2-cgroup-pacing-v2.json"
        )
        policy = self.pacing_policy()
        with mock.patch.dict(
            os.environ,
            {
                scope.POLICY_ENV: policy.content_sha256,
                scope.PACING_POLICY_ENV: policy.content_sha256,
            },
            clear=True,
        ):
            self.assertEqual(
                scope._load_thermal_pacing_policy(
                    policy_path,
                    policy.content_sha256,
                ),
                policy,
            )
        with mock.patch.dict(
            os.environ,
            {scope.PACING_POLICY_ENV: "sha256:" + "0" * 64},
            clear=True,
        ):
            with self.assertRaisesRegex(scope.ScopeExecError, "both outer"):
                scope._load_thermal_pacing_policy(
                    policy_path,
                    policy.content_sha256,
                )

    def test_thermal_telemetry_pipe_preserves_policy_sequence_and_samples(
        self,
    ) -> None:
        policy = self.pacing_policy()
        read_descriptor, write_descriptor = os.pipe()
        reader = scope.ThermalTelemetryReader(read_descriptor, policy)
        try:
            first = scope.wsl_thermal_exec.ThermalSample(1.0, 75_000, 60_000)
            second = scope.wsl_thermal_exec.ThermalSample(2.0, 80_000, 61_000)
            scope.wsl_thermal_exec._send_telemetry_sample(
                write_descriptor,
                policy,
                0,
                first,
            )
            scope.wsl_thermal_exec._send_telemetry_sample(
                write_descriptor,
                policy,
                1,
                second,
            )
            self.assertEqual(reader.read_available(), [first, second])
            reader.require_fresh(3.0)
            with self.assertRaisesRegex(scope.ScopeExecError, "stale"):
                reader.require_fresh(8.1)
        finally:
            reader.close()
            os.close(write_descriptor)

    def test_thermal_telemetry_pipe_rejects_wrong_sequence(self) -> None:
        policy = self.pacing_policy()
        read_descriptor, write_descriptor = os.pipe()
        reader = scope.ThermalTelemetryReader(read_descriptor, policy)
        try:
            scope.wsl_thermal_exec._send_telemetry_sample(
                write_descriptor,
                policy,
                1,
                scope.wsl_thermal_exec.ThermalSample(1.0, 75_000, 60_000),
            )
            with self.assertRaisesRegex(scope.ScopeExecError, "sequence"):
                reader.read_available()
        finally:
            reader.close()
            os.close(write_descriptor)

    def test_thermal_telemetry_close_preserves_primary_failure(self) -> None:
        reader = mock.Mock()
        primary = scope.ScopeExecError("thermal pacing pause exceeded")

        self.assertIs(
            scope._close_thermal_telemetry(reader, primary),
            primary,
        )
        reader.close.assert_called_once_with()

        reader = mock.Mock()
        reader.close.side_effect = scope.ScopeExecError("cannot close telemetry")
        combined = scope._close_thermal_telemetry(reader, primary)
        self.assertEqual(
            str(combined),
            "thermal pacing pause exceeded; cannot close telemetry",
        )
        reader.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
