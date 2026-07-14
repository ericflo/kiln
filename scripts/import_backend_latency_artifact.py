#!/usr/bin/env python3
"""Import backend latency fixture artifacts downloaded from GitHub Actions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from check_backend_latency_fixtures import validate_result_artifact
from write_backend_latency_result_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    LATENCY_RAW_LOG_DIR,
    LATENCY_RESULT_ARTIFACT_DIR,
    current_git_commit,
    find_fixture,
    fixture_spec_sha256,
    is_canonical_raw_log_path,
    is_canonical_result_artifact_path,
    is_repo_relative_path,
    load_manifest,
    repo_relative_path,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


class LatencyArtifactImportError(Exception):
    pass


@dataclass
class FileSnapshot:
    path: Path
    existed: bool
    content: bytes | None
    touched: bool


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LatencyArtifactImportError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LatencyArtifactImportError(f"{label} must be a JSON object")
    return value


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise LatencyArtifactImportError(f"artifact bundle is not a valid zip: {zip_path}") from exc

    with archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise LatencyArtifactImportError(
                    f"artifact zip contains unsafe path: {member.filename!r}"
                )
            destination = (output_dir / member.filename).resolve()
            try:
                destination.relative_to(output_root)
            except ValueError as exc:
                raise LatencyArtifactImportError(
                    f"artifact zip escapes extraction root: {member.filename!r}"
                ) from exc
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


@contextmanager
def artifact_bundle_root(bundle: Path) -> Iterator[Path]:
    if bundle.is_dir():
        yield bundle
        return
    if not bundle.is_file():
        raise LatencyArtifactImportError(f"artifact bundle does not exist: {bundle}")
    if not zipfile.is_zipfile(bundle):
        raise LatencyArtifactImportError(f"artifact bundle must be a directory or zip: {bundle}")

    temp_parent = ROOT / "target"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="backend-latency-artifact-", dir=temp_parent
    ) as tmp:
        output_dir = Path(tmp)
        safe_extract_zip(bundle, output_dir)
        yield output_dir


def result_artifact_candidates(
    bundle_root: Path,
    fixture_id: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(bundle_root.rglob("*.json")):
        if "__MACOSX" in path.parts:
            continue
        try:
            result = load_json(path, str(path))
        except LatencyArtifactImportError:
            continue
        if result.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            continue
        observed_fixture_id = result.get("fixture_id")
        if not isinstance(observed_fixture_id, str) or not observed_fixture_id:
            continue
        if fixture_id is not None and observed_fixture_id != fixture_id:
            continue
        candidates.append((path, result))
    return candidates


def select_result_artifact(
    bundle_root: Path,
    fixture_id: str | None,
) -> tuple[Path, dict[str, Any]]:
    candidates = result_artifact_candidates(bundle_root, fixture_id)
    if not candidates:
        suffix = f" for fixture {fixture_id!r}" if fixture_id else ""
        raise LatencyArtifactImportError(f"no latency result artifact JSON found{suffix}")
    if len(candidates) > 1:
        labels = [str(path) for path, _ in candidates]
        raise LatencyArtifactImportError(
            "artifact bundle contains multiple result artifacts; pass --fixture-id: "
            + ", ".join(labels)
        )
    return candidates[0]


def unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def locate_raw_log(bundle_root: Path, raw_log: str) -> Path:
    if not is_canonical_raw_log_path(raw_log):
        raise LatencyArtifactImportError(
            f"result raw_log must live under {LATENCY_RAW_LOG_DIR} with a .log extension: {raw_log}"
        )

    raw_path = Path(raw_log)
    relative_to_latency_root = raw_path.relative_to(LATENCY_RESULT_ARTIFACT_DIR)
    direct_candidates = unique_existing(
        [
            bundle_root / raw_path,
            bundle_root / relative_to_latency_root,
            bundle_root / raw_path.name,
        ]
    )
    if len(direct_candidates) == 1:
        return direct_candidates[0]
    if len(direct_candidates) > 1:
        raise LatencyArtifactImportError(
            "artifact bundle contains multiple matching raw logs: "
            + ", ".join(str(path) for path in direct_candidates)
        )

    named_candidates = unique_existing(list(bundle_root.rglob(raw_path.name)))
    if len(named_candidates) == 1:
        return named_candidates[0]
    if not named_candidates:
        raise LatencyArtifactImportError(f"raw log not found in artifact bundle: {raw_log}")
    raise LatencyArtifactImportError(
        "artifact bundle contains multiple raw logs named "
        f"{raw_path.name}: " + ", ".join(str(path) for path in named_candidates)
    )


def repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def stage_copy(source: Path, destination: Path, force: bool) -> FileSnapshot:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        return FileSnapshot(destination, destination.exists(), None, touched=False)

    source_bytes = source.read_bytes()
    if destination.exists():
        if not destination.is_file():
            raise LatencyArtifactImportError(f"destination is not a regular file: {destination}")
        destination_bytes = destination.read_bytes()
        if destination_bytes == source_bytes:
            return FileSnapshot(destination, True, destination_bytes, touched=False)
        if not force:
            raise LatencyArtifactImportError(
                f"destination already exists with different content; pass --force to replace: {destination}"
            )
        snapshot = FileSnapshot(destination, True, destination_bytes, touched=True)
    else:
        snapshot = FileSnapshot(destination, False, None, touched=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return snapshot


def rollback(snapshots: list[FileSnapshot]) -> None:
    for snapshot in reversed(snapshots):
        if not snapshot.touched:
            continue
        if snapshot.existed:
            assert snapshot.content is not None
            snapshot.path.write_bytes(snapshot.content)
        else:
            try:
                snapshot.path.unlink()
            except FileNotFoundError:
                pass


def import_backend_latency_artifact(
    bundle: Path,
    manifest_path: Path,
    fixture_id: str | None,
    force: bool,
) -> dict[str, Any]:
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    try:
        manifest = load_manifest(manifest_path)
    except ArtifactError as exc:
        raise LatencyArtifactImportError(str(exc)) from exc

    with artifact_bundle_root(bundle) as bundle_root:
        result_source, result = select_result_artifact(bundle_root, fixture_id)
        observed_fixture_id = result.get("fixture_id")
        if not isinstance(observed_fixture_id, str) or not observed_fixture_id:
            raise LatencyArtifactImportError("result fixture_id must be a non-empty string")
        try:
            fixture = find_fixture(manifest, observed_fixture_id)
        except ArtifactError as exc:
            raise LatencyArtifactImportError(str(exc)) from exc

        result_artifact = fixture.get("result_artifact")
        if not isinstance(result_artifact, str) or not result_artifact:
            raise LatencyArtifactImportError(
                f"{observed_fixture_id}.result_artifact must be a non-empty string"
            )
        if not is_repo_relative_path(result_artifact) or not is_canonical_result_artifact_path(
            result_artifact
        ):
            raise LatencyArtifactImportError(
                f"{observed_fixture_id}.result_artifact must live under "
                f"{LATENCY_RESULT_ARTIFACT_DIR} with a .json extension"
            )

        raw_log = result.get("raw_log")
        if not isinstance(raw_log, str) or not raw_log:
            raise LatencyArtifactImportError("result raw_log must be a non-empty string")
        raw_source = locate_raw_log(bundle_root, raw_log)
        raw_log_sha256 = result.get("raw_log_sha256")
        if not isinstance(raw_log_sha256, str) or sha256_file(raw_source) != raw_log_sha256:
            raise LatencyArtifactImportError(
                f"raw log checksum does not match result raw_log_sha256: {raw_log}"
            )

        result_target = repo_path(result_artifact)
        raw_target = repo_path(raw_log)
        snapshots: list[FileSnapshot] = []
        try:
            snapshots.append(stage_copy(result_source, result_target, force=force))
            snapshots.append(stage_copy(raw_source, raw_target, force=force))

            errors: list[str] = []
            validate_result_artifact(
                errors,
                fixture,
                result_target,
                f"fixture {observed_fixture_id}",
                manifest.get("schema_version"),
                manifest_path,
                require_covered_provenance=True,
                enforce_git_retention=True,
                require_threshold_pass=False,
            )
            if errors:
                raise LatencyArtifactImportError(
                    "imported latency artifact failed validation:\n- "
                    + "\n- ".join(errors)
                )
        except Exception:
            rollback(snapshots)
            raise

    return {
        "fixture_id": observed_fixture_id,
        "backend": result.get("backend"),
        "result_artifact": repo_relative_path(result_target),
        "raw_log": repo_relative_path(raw_target),
        "metrics": sorted(result.get("metrics", {}).keys())
        if isinstance(result.get("metrics"), dict)
        else [],
    }


def self_test() -> int:
    temp_parent = ROOT / "target"
    result_parent = ROOT / LATENCY_RESULT_ARTIFACT_DIR
    raw_parent = ROOT / LATENCY_RAW_LOG_DIR
    temp_parent.mkdir(parents=True, exist_ok=True)
    result_parent.mkdir(parents=True, exist_ok=True)
    raw_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="backend-latency-importer-", dir=temp_parent
    ) as tmp, tempfile.TemporaryDirectory(
        prefix="backend-latency-importer-", dir=result_parent
    ) as result_tmp, tempfile.TemporaryDirectory(
        prefix="backend-latency-importer-", dir=raw_parent
    ) as raw_tmp:
        tmp_root = Path(tmp)
        manifest_path = tmp_root / "fixtures.json"
        result_target = Path(result_tmp) / "result.json"
        raw_target = Path(raw_tmp) / "fixture.log"
        source = ROOT / "crates/kiln-tensor/tests/rocm_latency_bench.rs"
        raw_text = (
            "warmup\n"
            "KILN_LATENCY_METRIC latency_ms 9.5 ms\n"
            "KILN_LATENCY_METRIC tokens_per_s 125.0 tok/s\n"
        )
        source_path = repo_relative_path(source)
        raw_target_path = repo_relative_path(raw_target)
        result_target_path = repo_relative_path(result_target)
        fixture = {
            "id": "import_fixture",
            "backend": "cuda",
            "hardware": "fixture hardware",
            "source": source_path,
            "command": "cargo bench",
            "result_artifact": result_target_path,
            "threshold_state": "pending_fixture_result",
            "metrics": [
                {"name": "latency_ms", "unit": "ms", "comparison": "<=", "max": None},
                {"name": "tokens_per_s", "unit": "tok/s", "comparison": ">=", "max": None},
            ],
        }
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "fixture_required",
                    "required_backends": ["cuda"],
                    "fixtures": [fixture],
                    "missing_fixture_slots": [],
                }
            )
        )
        bundle = tmp_root / "bundle"
        bundle_raw = bundle / "raw"
        bundle_raw.mkdir(parents=True)
        bundle_raw_log = bundle_raw / raw_target.name
        bundle_raw_log.write_text(raw_text)
        result = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at_utc": "2026-06-06T12:00:00Z",
            "git_commit": current_git_commit(),
            "git_tracked_dirty": False,
            "fixture_id": "import_fixture",
            "backend": "cuda",
            "status": "passed",
            "manifest": repo_relative_path(manifest_path),
            "manifest_schema_version": 1,
            "fixture_spec_sha256": fixture_spec_sha256(fixture),
            "hardware": "fixture hardware",
            "source": source_path,
            "source_sha256": sha256_file(source),
            "command": "cargo bench",
            "raw_log": raw_target_path,
            "raw_log_sha256": sha256_file(bundle_raw_log),
            "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
        }
        bundle_result = bundle / "result.json"
        bundle_result.write_text(json.dumps(result))

        summary = import_backend_latency_artifact(
            bundle,
            manifest_path,
            "import_fixture",
            force=False,
        )
        if (
            summary.get("fixture_id") != "import_fixture"
            or not result_target.is_file()
            or not raw_target.is_file()
            or json.loads(result_target.read_text()).get("metrics", {}).get("latency_ms")
            != 9.5
            or "KILN_LATENCY_METRIC latency_ms" not in raw_target.read_text()
        ):
            print(json.dumps({"ok": False, "case": "directory import", "summary": summary}))
            return 1

        zip_path = tmp_root / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(bundle_result, "result.json")
            archive.write(bundle_raw_log, f"raw/{raw_target.name}")
        zip_summary = import_backend_latency_artifact(
            zip_path,
            manifest_path,
            "import_fixture",
            force=True,
        )
        if zip_summary.get("raw_log") != raw_target_path:
            print(json.dumps({"ok": False, "case": "zip import", "summary": zip_summary}))
            return 1

    print(json.dumps({"ok": True, "self_test": "backend latency artifact importer"}))
    return 0


def fail(message: str) -> int:
    print(json.dumps({"ok": False, "error": message}, indent=2), file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact_bundle",
        nargs="?",
        help="Downloaded Actions artifact zip or extracted artifact directory",
    )
    parser.add_argument(
        "--manifest",
        default="docs/backend-latency-fixtures.json",
        help="Path to backend-latency-fixtures.json",
    )
    parser.add_argument(
        "--fixture-id",
        help="Fixture id to import when the bundle contains multiple result artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing canonical result/log files when content differs",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run importer self-tests instead of importing an artifact bundle",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.artifact_bundle:
        return fail("artifact_bundle is required unless --self-test is set")

    try:
        summary = import_backend_latency_artifact(
            Path(args.artifact_bundle),
            Path(args.manifest),
            args.fixture_id,
            force=args.force,
        )
    except LatencyArtifactImportError as exc:
        return fail(str(exc))

    print(json.dumps({"ok": True, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
