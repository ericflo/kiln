#!/usr/bin/env python3
"""Run the complete Kiln/vLLM serving benchmark profile matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "bench-concurrent-batch.py"
SCHEMA = "kiln.serving-benchmark-campaign.v3"
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("kiln", "vllm"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--runtime-artifact", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="directory containing matching *.kiln.json receipts; required for vLLM",
    )
    parser.add_argument("--sizes", default="1,8,16,32,64,128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--memory-path", type=Path, required=True)
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--memory-sample-ms", type=int, default=50)
    parser.add_argument("--host-thermal-policy", type=Path, required=True)
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
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}", args.campaign_id) is None:
        parser.error("campaign-id must be 3..64 portable identifier characters")
    if args.engine == "vllm" and args.reference_dir is None:
        parser.error("--reference-dir is required for a vLLM campaign")
    if args.engine == "kiln" and args.reference_dir is not None:
        parser.error("--reference-dir is only valid for a vLLM campaign")
    if args.server_pid is not None and args.server_pid <= 1:
        parser.error("--server-pid must be greater than one")
    if args.host_thermal_policy.is_symlink() or not args.host_thermal_policy.is_file():
        parser.error("--host-thermal-policy must name a regular file")
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
        f"{args.campaign_id}-{profile}",
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
        "--memory-path",
        str(args.memory_path),
        "--memory-limit-bytes",
        str(args.memory_limit_bytes),
        "--memory-sample-ms",
        str(args.memory_sample_ms),
        "--host-thermal-policy",
        str(args.host_thermal_policy),
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
        "--out",
        str(output),
    ]
    if args.server_launch_config is not None:
        command.extend(("--server-launch-config", str(args.server_launch_config)))
    else:
        command.extend(("--server-pid", str(args.server_pid)))
    if args.api_key_env is not None:
        command.extend(("--api-key-env", args.api_key_env))
    if args.reference_dir is not None:
        command.extend(
            (
                "--reference-receipt",
                str(args.reference_dir / f"{profile}.kiln.json"),
            )
        )
    return command


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
        for profile in PROFILES:
            output = outputs[profile]
            result = subprocess.run(benchmark_command(args, profile, output), check=False)
            row: dict[str, Any] = {
                "profile": profile,
                "exit_code": result.returncode,
                "receipt": str(output),
                "receipt_sha256": file_sha256(output) if output.is_file() else None,
            }
            rows.append(row)
        summary: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "campaign_id": args.campaign_id,
            "engine": args.engine,
            "output_evidence": args.output_evidence,
            "host_thermal_policy": {
                "path": str(args.host_thermal_policy.resolve()),
                "sha256": file_sha256(args.host_thermal_policy),
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
                if all(row["exit_code"] == 0 and row["receipt_sha256"] for row in rows)
                else "failed"
            ),
        }
        summary["summary_sha256"] = canonical_sha256(summary)
        atomic_write_json(summary_path, summary)
        print(f"wrote {summary_path}")
        return 0 if summary["verdict"] == "passed" else 2
    except (CampaignError, OSError) as exc:
        print(f"campaign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
