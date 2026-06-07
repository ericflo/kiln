#!/usr/bin/env python3
"""Plan gh workflow run dispatches and gh api runner checks for latency fixtures."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from write_backend_latency_result_artifact import ArtifactError, load_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "backend-latency-fixtures.json"
WORKFLOW_FILE = "perf-regression-nightly.yml"
DEFAULT_ARTIFACT_DOWNLOAD_DIR = "/tmp/kiln-backend-latency"


class DispatchPlanError(Exception):
    pass


def fail(message: str) -> int:
    print(json.dumps({"ok": False, "error": message}, indent=2), file=sys.stderr)
    return 1


def parse_runner_labels_json(value: str, context: str) -> list[str]:
    try:
        labels = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DispatchPlanError(f"{context} must be valid JSON: {exc}") from exc
    return validate_runner_labels(labels, context)


def validate_runner_labels(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DispatchPlanError(f"{context} must be a non-empty string array")
    labels: list[str] = []
    for index, label in enumerate(value):
        if not isinstance(label, str) or not label:
            raise DispatchPlanError(f"{context}[{index}] must be a non-empty string")
        labels.append(label)
    return labels


def current_git_ref() -> str:
    try:
        ref = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "HEAD"
    return ref or "HEAD"


def current_github_repo() -> str | None:
    try:
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

    patterns = [
        r"^git@github\.com:([^/]+/[^/]+?)(?:\.git)?$",
        r"^https://github\.com/([^/]+/[^/]+?)(?:\.git)?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, remote)
        if match:
            return match.group(1)
    return None


def repo_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def fixture_needs_dispatch(fixture: dict[str, Any], root: Path) -> bool:
    if fixture.get("threshold_state") != "locked_threshold":
        return True
    result_artifact = fixture.get("result_artifact")
    if not isinstance(result_artifact, str) or not result_artifact:
        return True
    return not repo_path(result_artifact, root).is_file()


def fixture_runner_labels(
    fixture: dict[str, Any],
    override_labels: list[str] | None,
) -> list[str] | None:
    if override_labels is not None:
        return override_labels
    labels = fixture.get("runner_labels")
    if labels is None:
        return None
    fixture_id = fixture.get("id", "<unknown>")
    return validate_runner_labels(labels, f"fixture {fixture_id!r}.runner_labels")


def gh_workflow_command(
    fixture_id: str,
    runner_labels: list[str],
    ref: str,
) -> str:
    labels_json = json.dumps(runner_labels, separators=(",", ":"))
    return shlex.join(
        [
            "gh",
            "workflow",
            "run",
            WORKFLOW_FILE,
            "--ref",
            ref,
            "--field",
            f"ref={ref}",
            "--field",
            f"latency_fixture_id={fixture_id}",
            "--field",
            f"latency_runner_labels_json={labels_json}",
        ]
    )


def artifact_name_template(fixture_id: str) -> str:
    return f"backend-latency-{fixture_id}-RUN_ID"


def artifact_download_dir_template(fixture_id: str) -> str:
    return f"{DEFAULT_ARTIFACT_DOWNLOAD_DIR}/{fixture_id}-RUN_ID"


def gh_run_download_command(fixture_id: str) -> str:
    return shlex.join(
        [
            "gh",
            "run",
            "download",
            "RUN_ID",
            "--name",
            artifact_name_template(fixture_id),
            "--dir",
            artifact_download_dir_template(fixture_id),
        ]
    )


def import_artifact_command(fixture_id: str) -> str:
    return shlex.join(
        [
            "python3",
            "scripts/import_backend_latency_artifact.py",
            artifact_download_dir_template(fixture_id),
            "--fixture-id",
            fixture_id,
        ]
    )


def lock_threshold_command(fixture_id: str) -> str:
    return shlex.join(
        [
            "python3",
            "scripts/lock_backend_latency_thresholds.py",
            "docs/backend-latency-fixtures.json",
            "--fixture-id",
            fixture_id,
        ]
    )


def covered_gate_command() -> str:
    return shlex.join(
        [
            "python3",
            "scripts/check_backend_latency_fixtures.py",
            "docs/backend-latency-fixtures.json",
            "--require-covered",
        ]
    )


def runner_label_names(runner: dict[str, Any]) -> list[str]:
    labels = runner.get("labels")
    if not isinstance(labels, list):
        return []
    names: list[str] = []
    for label in labels:
        if isinstance(label, dict) and isinstance(label.get("name"), str):
            names.append(label["name"])
        elif isinstance(label, str):
            names.append(label)
    return names


def runner_matches_labels(runner: dict[str, Any], required_labels: list[str]) -> bool:
    observed = {label.casefold() for label in runner_label_names(runner)}
    return all(label.casefold() in observed for label in required_labels)


def runner_summary(runner: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": runner.get("name"),
        "status": runner.get("status"),
        "busy": runner.get("busy"),
        "labels": sorted(runner_label_names(runner), key=str.casefold),
    }


def runner_check_summary(
    *,
    repo: str,
    runners: list[dict[str, Any]],
    required_labels: list[str] | None,
) -> dict[str, Any]:
    if required_labels is None:
        return {
            "repo": repo,
            "status": "labels_required",
            "required_labels": None,
            "total_runners": len(runners),
            "matching_runners": [],
            "matching_runner_count": 0,
            "online_matching_runner_count": 0,
            "idle_matching_runner_count": 0,
        }

    matches = [
        runner for runner in runners if runner_matches_labels(runner, required_labels)
    ]
    online_matches = [
        runner for runner in matches if runner.get("status") == "online"
    ]
    idle_matches = [
        runner for runner in online_matches if runner.get("busy") is False
    ]
    if idle_matches:
        status = "matching_online_idle"
    elif online_matches:
        status = "matching_online_busy"
    elif matches:
        status = "matching_offline"
    else:
        status = "no_matching_runner"
    return {
        "repo": repo,
        "status": status,
        "required_labels": required_labels,
        "total_runners": len(runners),
        "matching_runners": [runner_summary(runner) for runner in matches],
        "matching_runner_count": len(matches),
        "online_matching_runner_count": len(online_matches),
        "idle_matching_runner_count": len(idle_matches),
    }


def runners_from_api_payload(payload: Any) -> list[dict[str, Any]]:
    pages = payload if isinstance(payload, list) else [payload]
    runners: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_runners = page.get("runners")
        if not isinstance(page_runners, list):
            continue
        runners.extend(runner for runner in page_runners if isinstance(runner, dict))
    return runners


def fetch_github_runners(repo: str) -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/{repo}/actions/runners",
                "--paginate",
                "--slurp",
            ],
            cwd=ROOT,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise DispatchPlanError("gh CLI is required for --check-runners") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise DispatchPlanError(f"failed to list GitHub runners for {repo}: {stderr}") from exc

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise DispatchPlanError(f"gh runner response was not valid JSON: {exc}") from exc
    return runners_from_api_payload(payload)


def dispatch_plans(
    manifest: dict[str, Any],
    *,
    root: Path,
    fixture_ids: list[str] | None,
    include_covered: bool,
    override_labels: list[str] | None,
    ref: str,
    runner_repo: str | None = None,
    runner_inventory: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list):
        raise DispatchPlanError("manifest.fixtures must be an array")
    selected = set(fixture_ids or [])
    seen: set[str] = set()
    plans: list[dict[str, Any]] = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue
        fixture_id = fixture.get("id")
        if not isinstance(fixture_id, str) or not fixture_id:
            continue
        if selected and fixture_id not in selected:
            continue
        seen.add(fixture_id)
        should_plan = include_covered or selected or fixture_needs_dispatch(fixture, root)
        if not should_plan:
            continue
        labels = fixture_runner_labels(fixture, override_labels)
        result_artifact = fixture.get("result_artifact")
        artifact_exists = (
            isinstance(result_artifact, str)
            and bool(result_artifact)
            and repo_path(result_artifact, root).is_file()
        )
        command = (
            gh_workflow_command(fixture_id, labels, ref)
            if labels is not None
            else None
        )
        plan = {
            "fixture_id": fixture_id,
            "backend": fixture.get("backend"),
            "threshold_state": fixture.get("threshold_state"),
            "result_artifact": result_artifact,
            "artifact_exists": artifact_exists,
            "runner_labels": labels,
            "needs_runner_labels": labels is None,
            "workflow": f".github/workflows/{WORKFLOW_FILE}",
            "ref": ref,
            "gh_workflow_run": command,
            "artifact_name_template": artifact_name_template(fixture_id),
            "artifact_download_dir_template": artifact_download_dir_template(
                fixture_id
            ),
            "gh_run_download": gh_run_download_command(fixture_id),
            "import_artifact": import_artifact_command(fixture_id),
            "lock_threshold": lock_threshold_command(fixture_id),
            "covered_gate_check": covered_gate_command(),
        }
        if runner_inventory is not None:
            assert runner_repo is not None
            plan["github_runner_check"] = runner_check_summary(
                repo=runner_repo,
                runners=runner_inventory,
                required_labels=labels,
            )
        plans.append(plan)

    missing = sorted(selected - seen)
    if missing:
        raise DispatchPlanError(f"fixture id not found in manifest: {', '.join(missing)}")
    return plans


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        existing_artifact = tmp_root / "metal.json"
        existing_artifact.write_text("{}\n")
        manifest = {
            "fixtures": [
                {
                    "id": "cuda_fixture",
                    "backend": "cuda",
                    "threshold_state": "pending_fixture_result",
                    "result_artifact": "missing-cuda.json",
                    "runner_labels": ["self-hosted", "linux", "cuda-a6000"],
                },
                {
                    "id": "metal_fixture",
                    "backend": "metal",
                    "threshold_state": "locked_threshold",
                    "result_artifact": str(existing_artifact),
                    "runner_labels": ["macos-14"],
                },
                {
                    "id": "rocm_fixture",
                    "backend": "rocm",
                    "threshold_state": "pending_fixture_result",
                    "result_artifact": "missing-rocm.json",
                },
            ]
        }
        plans = dispatch_plans(
            manifest,
            root=tmp_root,
            fixture_ids=None,
            include_covered=False,
            override_labels=None,
            ref="feature-branch",
            runner_repo="owner/repo",
            runner_inventory=[
                {
                    "name": "a6000-1",
                    "status": "online",
                    "busy": False,
                    "labels": [
                        {"name": "self-hosted"},
                        {"name": "Linux"},
                        {"name": "cuda-a6000"},
                    ],
                }
            ],
        )
        if [plan["fixture_id"] for plan in plans] != ["cuda_fixture", "rocm_fixture"]:
            print(json.dumps({"ok": False, "case": "pending plan selection", "plans": plans}))
            return 1
        cuda_plan = plans[0]
        if (
            cuda_plan["runner_labels"] != ["self-hosted", "linux", "cuda-a6000"]
            or "latency_fixture_id=cuda_fixture" not in cuda_plan["gh_workflow_run"]
            or 'latency_runner_labels_json=["self-hosted","linux","cuda-a6000"]'
            not in cuda_plan["gh_workflow_run"]
            or "ref=feature-branch" not in cuda_plan["gh_workflow_run"]
            or cuda_plan["artifact_name_template"] != "backend-latency-cuda_fixture-RUN_ID"
            or "gh run download RUN_ID" not in cuda_plan["gh_run_download"]
            or "backend-latency-cuda_fixture-RUN_ID" not in cuda_plan["gh_run_download"]
            or "scripts/import_backend_latency_artifact.py" not in cuda_plan["import_artifact"]
            or "--fixture-id cuda_fixture" not in cuda_plan["import_artifact"]
            or "scripts/lock_backend_latency_thresholds.py" not in cuda_plan["lock_threshold"]
            or "--fixture-id cuda_fixture" not in cuda_plan["lock_threshold"]
            or "--require-covered" not in cuda_plan["covered_gate_check"]
            or cuda_plan["github_runner_check"]["status"] != "matching_online_idle"
            or cuda_plan["github_runner_check"]["idle_matching_runner_count"] != 1
        ):
            print(json.dumps({"ok": False, "case": "cuda command", "plan": cuda_plan}))
            return 1
        if (
            not plans[1]["needs_runner_labels"]
            or plans[1]["gh_workflow_run"] is not None
            or plans[1]["github_runner_check"]["status"] != "labels_required"
        ):
            print(json.dumps({"ok": False, "case": "missing label marker", "plan": plans[1]}))
            return 1

        override = parse_runner_labels_json('["self-hosted","linux","rocm"]', "--runner-labels-json")
        rocm_plan = dispatch_plans(
            manifest,
            root=tmp_root,
            fixture_ids=["rocm_fixture"],
            include_covered=False,
            override_labels=override,
            ref="feature-branch",
            runner_repo="owner/repo",
            runner_inventory=[],
        )[0]
        if (
            rocm_plan["needs_runner_labels"]
            or "latency_fixture_id=rocm_fixture" not in rocm_plan["gh_workflow_run"]
            or 'latency_runner_labels_json=["self-hosted","linux","rocm"]'
            not in rocm_plan["gh_workflow_run"]
            or rocm_plan["github_runner_check"]["status"] != "no_matching_runner"
        ):
            print(json.dumps({"ok": False, "case": "override labels", "plan": rocm_plan}))
            return 1

        if current_github_repo() is not None and "/" not in current_github_repo():
            print(json.dumps({"ok": False, "case": "repo detection"}))
            return 1

        try:
            parse_runner_labels_json('["self-hosted", ""]', "--runner-labels-json")
        except DispatchPlanError as exc:
            if "must be a non-empty string" not in str(exc):
                print(json.dumps({"ok": False, "case": "invalid labels", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "invalid labels accepted"}))
            return 1

    print(json.dumps({"ok": True, "self_test": "backend latency fixture dispatch planner"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(DEFAULT_MANIFEST),
        help="Path to backend-latency-fixtures.json",
    )
    parser.add_argument(
        "--fixture-id",
        action="append",
        help="Fixture id to plan; repeat to plan multiple fixtures",
    )
    parser.add_argument(
        "--include-covered",
        action="store_true",
        help="Include fixtures whose thresholds and artifacts are already present",
    )
    parser.add_argument(
        "--runner-labels-json",
        help="Override runner labels for selected fixtures, for example '[\"self-hosted\",\"linux\",\"rocm\"]'",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Git ref for both the workflow version and checkout input; defaults to the current branch",
    )
    parser.add_argument(
        "--check-runners",
        action="store_true",
        help="Query GitHub self-hosted runners and report whether fixture labels have a matching runner",
    )
    parser.add_argument(
        "--github-repo",
        help="Repository for --check-runners, in owner/name form; defaults to remote.origin.url",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print gh workflow run commands instead of a JSON plan",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dispatch-planner self-tests",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    try:
        override_labels = (
            parse_runner_labels_json(args.runner_labels_json, "--runner-labels-json")
            if args.runner_labels_json
            else None
        )
        manifest = load_manifest(manifest_path)
        ref = args.ref or current_git_ref()
        runner_repo = None
        runner_inventory = None
        if args.check_runners:
            runner_repo = args.github_repo or current_github_repo()
            if runner_repo is None:
                raise DispatchPlanError(
                    "--check-runners requires --github-repo or a GitHub remote.origin.url"
                )
            runner_inventory = fetch_github_runners(runner_repo)

        plans = dispatch_plans(
            manifest,
            root=ROOT,
            fixture_ids=args.fixture_id,
            include_covered=args.include_covered,
            override_labels=override_labels,
            ref=ref,
            runner_repo=runner_repo,
            runner_inventory=runner_inventory,
        )
    except (ArtifactError, DispatchPlanError) as exc:
        return fail(str(exc))

    if args.shell:
        missing_labels = [
            plan["fixture_id"] for plan in plans if plan["needs_runner_labels"]
        ]
        if missing_labels:
            return fail(
                "runner labels are not declared for "
                + ", ".join(missing_labels)
                + "; pass --runner-labels-json with site-local labels"
            )
        for plan in plans:
            print(plan["gh_workflow_run"])
        return 0

    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(manifest_path),
                "ref": plans[0]["ref"] if plans else (args.ref or current_git_ref()),
                "plans": plans,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
