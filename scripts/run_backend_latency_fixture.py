#!/usr/bin/env python3
"""Run one backend latency fixture and materialize its result artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from write_backend_latency_result_artifact import (
    ArtifactError,
    GIT_COMMIT_RE,
    build_result_artifact,
    default_output_path,
    find_fixture,
    load_manifest,
    parse_metric_log,
    write_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
VALID_RESULT_STATUSES = {"passed", "failed"}


class FixtureRunError(Exception):
    pass


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "fixture"


def require_string(obj: dict[str, Any], key: str, context: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise FixtureRunError(f"{context}.{key} must be a non-empty string")
    return value


def default_log_path(fixture: dict[str, Any], log_dir: Path) -> Path:
    fixture_id = require_string(fixture, "id", "fixture")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"{safe_slug(fixture_id)}-{timestamp}.log"


def raw_log_tail(log_path: Path, max_lines: int = 20) -> str:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"<raw log unreadable: {exc}>"
    if not lines:
        return "<raw log empty>"
    return "\n".join(lines[-max_lines:])


def error_with_raw_log_tail(message: str, log_path: Path) -> FixtureRunError:
    return FixtureRunError(
        f"{message}; raw log: {log_path}; raw log tail:\n{raw_log_tail(log_path)}"
    )


def run_fixture_command(command: str, log_path: Path, echo: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if echo:
                print(line, end="")
        return_code = process.wait()
    if return_code != 0:
        raise error_with_raw_log_tail(
            f"fixture command exited with status {return_code}",
            log_path,
        )


def materialize_result_artifact(
    fixture: dict[str, Any],
    manifest_path: Path,
    manifest_schema_version: int,
    log_path: Path,
    output: Path | None,
    status: str,
) -> tuple[dict[str, Any], Path]:
    if status not in VALID_RESULT_STATUSES:
        raise FixtureRunError(f"status must be one of {sorted(VALID_RESULT_STATUSES)}")
    try:
        observations = parse_metric_log(log_path)
    except ArtifactError as exc:
        raise error_with_raw_log_tail(str(exc), log_path) from exc
    artifact = build_result_artifact(
        fixture,
        observations,
        status,
        log_path,
        manifest_path,
        manifest_schema_version,
    )
    output_path = output if output is not None else default_output_path(fixture)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    write_artifact(output_path, artifact)
    return artifact, output_path


def run_fixture(
    manifest_path: Path,
    fixture_id: str,
    log_dir: Path,
    output: Path | None,
    status: str,
    echo: bool,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    fixture = find_fixture(manifest, fixture_id)
    command = require_string(fixture, "command", fixture_id)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_path = default_log_path(fixture, log_dir)
    run_fixture_command(command, log_path, echo=echo)
    artifact, output_path = materialize_result_artifact(
        fixture,
        manifest_path,
        manifest.get("schema_version"),
        log_path,
        output,
        status,
    )
    return {
        "fixture_id": artifact["fixture_id"],
        "backend": artifact["backend"],
        "raw_log": str(log_path),
        "output": str(output_path),
        "metrics": sorted(artifact["metrics"]),
    }


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        bench_script = tmp_root / "bench.py"
        result_path = tmp_root / "result.json"
        manifest_path = tmp_root / "fixtures.json"
        log_dir = tmp_root / "logs"
        bench_script.write_text(
            "\n".join(
                [
                    "print('fixture warmup')",
                    "print('KILN_LATENCY_METRIC latency_ms 9.25 ms')",
                    "print('KILN_LATENCY_METRIC tokens_per_s 128.0 tok/s')",
                ]
            )
            + "\n"
        )
        source_sha256 = hashlib.sha256(bench_script.read_bytes()).hexdigest()
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fixtures": [
                        {
                            "id": "runner_fixture",
                            "backend": "cuda",
                            "hardware": "fixture hardware",
                            "source": str(bench_script),
                            "command": f"{sys.executable} {bench_script}",
                            "result_artifact": str(result_path),
                            "metrics": [
                                {"name": "latency_ms", "unit": "ms"},
                                {"name": "tokens_per_s", "unit": "tok/s"},
                            ],
                        }
                    ]
                }
            )
        )
        summary = run_fixture(
            manifest_path,
            "runner_fixture",
            log_dir,
            output=None,
            status="passed",
            echo=False,
        )
        result = json.loads(result_path.read_text())
        if (
            result.get("fixture_id") != "runner_fixture"
            or result.get("backend") != "cuda"
            or result.get("status") != "passed"
            or result.get("manifest") != str(manifest_path)
            or result.get("manifest_schema_version") != 1
            or not isinstance(result.get("fixture_spec_sha256"), str)
            or not GIT_COMMIT_RE.match(result.get("git_commit", ""))
            or not isinstance(result.get("git_tracked_dirty"), bool)
            or result.get("hardware") != "fixture hardware"
            or result.get("source") != str(bench_script)
            or result.get("source_sha256") != source_sha256
            or result.get("command") != f"{sys.executable} {bench_script}"
            or result.get("metrics") != {"latency_ms": 9.25, "tokens_per_s": 128.0}
            or not isinstance(result.get("raw_log_sha256"), str)
        ):
            print(json.dumps({"ok": False, "case": "result artifact", "result": result}))
            return 1
        raw_log = Path(summary["raw_log"])
        if not raw_log.is_file() or "KILN_LATENCY_METRIC latency_ms" not in raw_log.read_text():
            print(json.dumps({"ok": False, "case": "raw log", "summary": summary}))
            return 1

        no_metric_script = tmp_root / "no_metric.py"
        no_metric_script.write_text("print('fixture skipped: no hardware device')\n")
        no_metric_result_path = tmp_root / "no-metric-result.json"
        no_metric_manifest_path = tmp_root / "no-metric-fixtures.json"
        no_metric_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fixtures": [
                        {
                            "id": "no_metric_fixture",
                            "backend": "metal",
                            "hardware": "fixture hardware",
                            "source": str(no_metric_script),
                            "command": f"{sys.executable} {no_metric_script}",
                            "result_artifact": str(no_metric_result_path),
                            "metrics": [{"name": "latency_ms", "unit": "ms"}],
                        }
                    ],
                }
            )
        )
        try:
            run_fixture(
                no_metric_manifest_path,
                "no_metric_fixture",
                log_dir,
                output=None,
                status="passed",
                echo=False,
            )
        except FixtureRunError as exc:
            error = str(exc)
            if (
                "no KILN_LATENCY_METRIC lines found" not in error
                or "raw log tail:" not in error
                or "fixture skipped: no hardware device" not in error
            ):
                print(json.dumps({"ok": False, "case": "no metric tail", "error": error}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "no metric fixture did not fail"}))
            return 1

    print(json.dumps({"ok": True, "self_test": "backend latency fixture runner"}))
    return 0


def fail(message: str) -> int:
    print(json.dumps({"ok": False, "error": message}, indent=2), file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="docs/backend-latency-fixtures.json",
        help="Path to backend-latency-fixtures.json",
    )
    parser.add_argument("fixture_id", nargs="?", help="Fixture id to run")
    parser.add_argument(
        "--log-dir",
        default="bench-results/backend-latency/raw",
        help="Directory for raw fixture logs",
    )
    parser.add_argument(
        "--output",
        help="Override output JSON path; defaults to the fixture result_artifact path",
    )
    parser.add_argument(
        "--status",
        choices=sorted(VALID_RESULT_STATUSES),
        default="passed",
        help="Result status to write when the command succeeds",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Capture command output to the raw log without echoing it to stdout",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run fixture-runner self-tests instead of running a hardware fixture",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not args.fixture_id:
        return fail("fixture_id is required unless --self-test is set")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    output = Path(args.output) if args.output else None
    if output is not None and not output.is_absolute():
        output = ROOT / output

    try:
        summary = run_fixture(
            manifest_path,
            args.fixture_id,
            Path(args.log_dir),
            output=output,
            status=args.status,
            echo=not args.quiet,
        )
    except (ArtifactError, FixtureRunError) as exc:
        return fail(str(exc))

    print(json.dumps({"ok": True, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
