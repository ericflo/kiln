#!/usr/bin/env python3
"""Run the complete Kiln/vLLM serving benchmark profile matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "bench-concurrent-batch.py"
SCHEMA = "kiln.serving-benchmark-campaign.v11"
DEFAULT_MODEL_FINGERPRINT_READ_MIB_PER_SECOND = 0
REFERENCE_ROLES = (
    "qualification_gate",
    "same_artifact_graph_eager_discriminator",
)
PROFILES = (
    "greedy-short",
    "api-default-sampled",
    "long-prefill",
    "prefix-hit",
    "mixed",
)


class CampaignError(RuntimeError):
    """Raised when a campaign cannot be started without ambiguity."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise CampaignError(f"refusing to overwrite campaign summary: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_publish_file(source: Path, destination: Path) -> None:
    if destination.exists():
        raise CampaignError(f"refusing to overwrite campaign artifact: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("kiln", "vllm"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--runtime-artifact", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--prompt-set-id",
        required=True,
        help="stable model-visible identity shared by comparable campaigns",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help=(
            "directory containing matching *.kiln.json receipts; required for vLLM "
            "and same-artifact Kiln graph/eager discrimination"
        ),
    )
    parser.add_argument(
        "--reference-role",
        choices=REFERENCE_ROLES,
        default="qualification_gate",
        help="whether reference comparison gates qualification or discriminates graph/eager execution",
    )
    parser.add_argument("--sizes", default="1,8,16,32,64,128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--memory-source", choices=("auto", "drm", "nvml"), default="auto"
    )
    parser.add_argument("--memory-path", default="auto")
    parser.add_argument("--memory-device-index", type=int)
    parser.add_argument("--memory-device-uuid")
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--memory-sample-ms", type=int, default=50)
    parser.add_argument(
        "--model-fingerprint-read-mib-per-second",
        type=int,
        default=DEFAULT_MODEL_FINGERPRINT_READ_MIB_PER_SECOND,
        help="optional cumulative model read limit; zero disables it",
    )
    server_owner = parser.add_mutually_exclusive_group(required=True)
    server_owner.add_argument("--server-pid", type=int)
    server_owner.add_argument("--server-launch-config", type=Path)
    parser.add_argument("--slo-ttft-ms", type=float, default=5_000.0)
    parser.add_argument("--slo-itl-ms", type=float, default=250.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=60_000.0)
    parser.add_argument("--timeout-secs", type=float, default=600.0)
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--output-evidence",
        choices=("hashes", "full"),
        default="hashes",
        help="per-request hash evidence or bounded full output diagnostics",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="run later profiles after a failed profile; default stops at the first failure",
    )
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}", args.campaign_id) is None:
        parser.error("campaign-id must be 3..64 portable identifier characters")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}", args.prompt_set_id) is None:
        parser.error("prompt-set-id must be 3..64 portable identifier characters")
    if args.reference_role == "same_artifact_graph_eager_discriminator":
        if args.engine != "kiln":
            parser.error(
                "same-artifact graph/eager discrimination requires --engine kiln"
            )
        if args.reference_dir is None:
            parser.error(
                "--reference-dir is required for same-artifact graph/eager discrimination"
            )
    elif args.engine == "vllm" and args.reference_dir is None:
        parser.error("--reference-dir is required for a vLLM campaign")
    elif args.engine == "kiln" and args.reference_dir is not None:
        parser.error(
            "--reference-dir for Kiln requires "
            "--reference-role same_artifact_graph_eager_discriminator"
        )
    if args.server_pid is not None and args.server_pid <= 1:
        parser.error("--server-pid must be greater than one")
    if args.memory_device_index is not None and args.memory_device_index < 0:
        parser.error("--memory-device-index must be non-negative")
    if args.memory_device_uuid is not None and re.fullmatch(
        r"GPU-[A-Za-z0-9-]{8,120}", args.memory_device_uuid
    ) is None:
        parser.error("--memory-device-uuid must be a complete NVML GPU UUID")
    if args.memory_device_index is not None and args.memory_device_uuid is not None:
        parser.error(
            "--memory-device-index and --memory-device-uuid are mutually exclusive"
        )
    if args.memory_source == "auto":
        has_nvml_selector = (
            args.memory_device_index is not None
            or args.memory_device_uuid is not None
        )
        if args.memory_path != "auto" and has_nvml_selector:
            parser.error(
                "--memory-path and an NVML device selector cannot both select a device"
            )
        if args.memory_path != "auto":
            args.memory_source = "drm"
        elif has_nvml_selector:
            args.memory_source = "nvml"
    elif args.memory_source == "drm" and (
        args.memory_device_index is not None or args.memory_device_uuid is not None
    ):
        parser.error(
            "NVML device selectors cannot be combined with --memory-source drm"
        )
    elif args.memory_source == "nvml" and args.memory_path != "auto":
        parser.error("--memory-path cannot be combined with --memory-source nvml")
    if args.model_fingerprint_read_mib_per_second != 0 and not (
        64 <= args.model_fingerprint_read_mib_per_second <= 16_384
    ):
        parser.error(
            "--model-fingerprint-read-mib-per-second must be zero or in 64..=16384"
        )
    if args.server_launch_config is not None and (
        args.server_launch_config.is_symlink()
        or not args.server_launch_config.is_file()
    ):
        parser.error("--server-launch-config must name a regular file")
    return args


def benchmark_command(
    args: argparse.Namespace,
    profile: str,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(DRIVER),
        "--engine",
        args.engine,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--model-path",
        str(args.model_path),
        "--runtime-identity",
        args.runtime_identity,
        "--runtime-artifact",
        str(args.runtime_artifact),
        "--run-id",
        f"{args.campaign_id}-{args.engine}-{profile}",
        "--prompt-set-id",
        f"{args.prompt_set_id}-{profile}",
        "--workload-profile",
        profile,
        "--sizes",
        args.sizes,
        "--repeats",
        str(args.repeats),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-requests",
        str(args.warmup_requests),
        "--seed",
        str(args.seed),
        "--memory-source",
        args.memory_source,
        "--memory-path",
        str(args.memory_path),
        "--memory-limit-bytes",
        str(args.memory_limit_bytes),
        "--memory-sample-ms",
        str(args.memory_sample_ms),
        "--model-fingerprint-read-mib-per-second",
        str(args.model_fingerprint_read_mib_per_second),
        "--slo-ttft-ms",
        str(args.slo_ttft_ms),
        "--slo-itl-ms",
        str(args.slo_itl_ms),
        "--slo-e2e-ms",
        str(args.slo_e2e_ms),
        "--timeout-secs",
        str(args.timeout_secs),
        "--output-evidence",
        args.output_evidence,
        "--reference-role",
        args.reference_role,
        "--out",
        str(output),
    ]
    if args.server_launch_config is not None:
        command.extend(("--server-launch-config", str(args.server_launch_config)))
    else:
        command.extend(("--server-pid", str(args.server_pid)))
    if args.api_key_env is not None:
        command.extend(("--api-key-env", args.api_key_env))
    if args.memory_device_index is not None:
        command.extend(("--memory-device-index", str(args.memory_device_index)))
    if args.memory_device_uuid is not None:
        command.extend(("--memory-device-uuid", args.memory_device_uuid))
    if args.reference_dir is not None:
        command.extend(
            (
                "--reference-receipt",
                str(args.reference_dir / f"{profile}.kiln.json"),
            )
        )
    return command


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "campaign_id": args.campaign_id,
        "prompt_set_id": args.prompt_set_id,
        "engine": args.engine,
        "reference_role": args.reference_role,
        "reference_dir": (
            str(args.reference_dir.resolve())
            if args.reference_dir is not None
            else None
        ),
        "output_evidence": args.output_evidence,
        "model_fingerprint_read_mib_per_second": (
            args.model_fingerprint_read_mib_per_second
        ),
        "execution_policy": (
            "continue_after_failure"
            if args.continue_after_failure
            else "fail_fast"
        ),
        "memory_sampler": {
            "source": args.memory_source,
            "path": (
                (
                    str(Path(args.memory_path).expanduser().resolve())
                    if args.memory_path != "auto"
                    else "auto"
                )
                if args.memory_source == "drm"
                else None
            ),
            "device_index": args.memory_device_index,
            "device_uuid": args.memory_device_uuid,
            "interval_ms": args.memory_sample_ms,
            "limit_bytes": args.memory_limit_bytes,
        },
        "server_owner": (
            {
                "mode": "owned_process_group",
                "launch_config": {
                    "path": str(args.server_launch_config.resolve()),
                    "sha256": file_sha256(args.server_launch_config),
                },
                "server_pid": None,
            }
            if args.server_launch_config is not None
            else {
                "mode": "attached_process_group",
                "launch_config": None,
                "server_pid": args.server_pid,
            }
        ),
        "profiles": rows,
        "verdict": (
            "passed"
            if all(
                row["status"] == "completed"
                and row["exit_code"] == 0
                and row["receipt_sha256"]
                for row in rows
            )
            else "failed"
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.out_dir / f"campaign.{args.engine}.json"
        outputs = {
            profile: args.out_dir / f"{profile}.{args.engine}.json"
            for profile in PROFILES
        }
        conflicts = [path for path in [summary_path, *outputs.values()] if path.exists()]
        if conflicts:
            raise CampaignError(
                "refusing to overwrite campaign artifacts: "
                + ", ".join(str(path) for path in conflicts)
            )
        rows: list[dict[str, Any]] = []
        staged_outputs: dict[str, Path] = {}
        failed_profile: str | None = None
        with tempfile.TemporaryDirectory(prefix="kiln-serving-campaign-") as staging:
            staging_dir = Path(staging)
            for profile in PROFILES:
                final_output = outputs[profile]
                if failed_profile is not None and not args.continue_after_failure:
                    rows.append(
                        {
                            "profile": profile,
                            "status": "not_run_after_failure",
                            "exit_code": None,
                            "receipt": str(final_output),
                            "receipt_sha256": None,
                            "blocked_by_profile": failed_profile,
                        }
                    )
                    continue

                staged_output = staging_dir / final_output.name
                result = subprocess.run(
                    benchmark_command(args, profile, staged_output), check=False
                )
                receipt_sha256 = (
                    file_sha256(staged_output) if staged_output.is_file() else None
                )
                rows.append(
                    {
                        "profile": profile,
                        "status": "completed",
                        "exit_code": result.returncode,
                        "receipt": str(final_output),
                        "receipt_sha256": receipt_sha256,
                        "blocked_by_profile": None,
                    }
                )
                if staged_output.is_file():
                    staged_outputs[profile] = staged_output
                if result.returncode != 0 or receipt_sha256 is None:
                    failed_profile = profile
            summary = build_summary(args, rows)
            for profile in PROFILES:
                if staged_output := staged_outputs.get(profile):
                    atomic_publish_file(staged_output, outputs[profile])
            atomic_write_json(summary_path, summary)
        print(f"wrote {summary_path}")
        return 0 if summary["verdict"] == "passed" else 2
    except (CampaignError, OSError) as exc:
        print(f"campaign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
