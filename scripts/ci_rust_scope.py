#!/usr/bin/env python3
"""Select host-testable workspace packages affected by a git diff."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL_WORKSPACE_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
}


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def changed_paths(base: str, head: str) -> list[str] | None:
    if not base or set(base) == {"0"}:
        return None
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{base}^{{commit}}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        output = subprocess.check_output(
            ["git", "diff", "--name-only", "-z", base, head], cwd=ROOT
        )
    except subprocess.CalledProcessError:
        return None
    return [path.decode() for path in output.split(b"\0") if path]


def reverse_closure(changed: set[str], reverse: dict[str, set[str]]) -> set[str]:
    affected = set(changed)
    queue = deque(changed)
    while queue:
        dependency = queue.popleft()
        for dependent in reverse.get(dependency, set()):
            if dependent not in affected:
                affected.add(dependent)
                queue.append(dependent)
    return affected


def host_testable_members(metadata: dict[str, object]) -> set[str]:
    """Return Cargo's default members, which are safe on the CPU CI host.

    Kiln's workspace also contains backend-only crates whose default features
    require CUDA, ROCm, Metal, or Vulkan toolchains. Cargo's
    ``workspace_default_members`` is the authoritative, already-maintained
    boundary between portable tests and those dedicated backend lanes.
    """

    workspace_members = set(metadata["workspace_members"])
    return set(metadata.get("workspace_default_members") or workspace_members)


def is_openenv_change(paths: list[str], full_workspace: bool) -> bool:
    if full_workspace:
        return True
    prefixes = (
        "crates/kiln-openenv/",
        "crates/kiln-server/src/api/openenv/",
    )
    exact = {
        "crates/kiln-server/src/adapter_swap.rs",
        "crates/kiln-server/src/api/openenv.rs",
        "crates/kiln-server/src/api/training.rs",
        "crates/kiln-server/src/cli.rs",
        "crates/kiln-server/src/config.rs",
        "crates/kiln-server/src/job_cancellation.rs",
        "crates/kiln-server/src/metrics.rs",
        "crates/kiln-server/src/openenv_cli.rs",
        "crates/kiln-server/src/openenv_credentials.rs",
        "crates/kiln-server/src/openenv_evaluation.rs",
        "crates/kiln-server/src/openenv_replay.rs",
        "crates/kiln-server/src/training_queue.rs",
        "crates/kiln-server/tests/openenv_training_interop.rs",
        "crates/kiln-train/src/adapter_output.rs",
        "crates/kiln-train/src/credential_provider.rs",
        "crates/kiln-train/src/lib.rs",
        "crates/kiln-train/src/openenv_provenance.rs",
        "crates/kiln-train/src/trajectory.rs",
        "scripts/check_miniopenenv_interop.sh",
    }
    return any(path in exact or path.startswith(prefixes) for path in paths)


def select(paths: list[str] | None) -> tuple[list[str], bool, bool, bool]:
    metadata = json.loads(command("cargo", "metadata", "--locked", "--format-version", "1"))
    workspace_members = set(metadata["workspace_members"])
    testable_members = host_testable_members(metadata)
    packages = {package["id"]: package for package in metadata["packages"]}

    full_workspace = paths is None or any(path in FULL_WORKSPACE_FILES for path in paths)
    changed_members: set[str] = set()
    if not full_workspace:
        roots = sorted(
            (
                (Path(packages[package_id]["manifest_path"]).parent.relative_to(ROOT), package_id)
                for package_id in workspace_members
            ),
            key=lambda item: len(item[0].parts),
            reverse=True,
        )
        for path in paths:
            candidate = Path(path)
            matched = next(
                (package_id for root, package_id in roots if candidate == root or root in candidate.parents),
                None,
            )
            if matched is not None:
                changed_members.add(matched)
            elif path.startswith("crates/"):
                full_workspace = True
                break

    if full_workspace:
        affected = testable_members
    else:
        reverse: dict[str, set[str]] = defaultdict(set)
        for node in metadata["resolve"]["nodes"]:
            if node["id"] not in workspace_members:
                continue
            for dependency in node["deps"]:
                if dependency["pkg"] in workspace_members:
                    reverse[dependency["pkg"]].add(node["id"])
        # A backend-only crate can still affect portable reverse dependents,
        # but it cannot itself be compiled on this no-GPU runner. Its own
        # compile/test coverage belongs to the dedicated backend lane.
        affected = reverse_closure(changed_members, reverse) & testable_members

    names = sorted(packages[package_id]["name"] for package_id in affected)
    dependency_policy = paths is None or any(
        path == "Cargo.lock" or path == "deny.toml" or path.endswith("Cargo.toml")
        for path in paths
    )
    return names, full_workspace, is_openenv_change(paths or [], full_workspace), dependency_policy


def self_test() -> None:
    graph = {"core": {"train", "server"}, "train": {"server"}}
    assert reverse_closure({"core"}, graph) == {"core", "train", "server"}
    assert reverse_closure({"train"}, graph) == {"train", "server"}
    assert host_testable_members(
        {
            "workspace_members": ["core", "cuda-kernel", "server"],
            "workspace_default_members": ["core", "server"],
        }
    ) == {"core", "server"}
    assert host_testable_members({"workspace_members": ["core"]}) == {"core"}
    assert is_openenv_change(["crates/kiln-openenv/src/client.rs"], False)
    assert is_openenv_change(["crates/kiln-train/src/openenv_provenance.rs"], False)
    assert not is_openenv_change(["crates/kiln-server/src/health.rs"], False)
    assert not is_openenv_change(["crates/kiln-train/src/opd.rs"], False)
    print("CI Rust scope self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    paths = changed_paths(args.base, args.head)
    packages, full_workspace, openenv, dependency_policy = select(paths)
    values = {
        "run_tests": str(bool(packages)).lower(),
        "full_workspace": str(full_workspace).lower(),
        "openenv": str(openenv).lower(),
        "dependency_policy": str(dependency_policy).lower(),
        "package_args": " ".join(f"--package {name}" for name in packages),
        "summary": "workspace" if full_workspace else ", ".join(packages) or "no Rust packages",
    }
    print(json.dumps(values, indent=2))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            for key, value in values.items():
                print(f"{key}={value}", file=output)


if __name__ == "__main__":
    main()
