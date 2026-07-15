#!/usr/bin/env python3
"""Run sequential bounded HF-reference and Vulkan full-logit forwards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ENV = "KILN_QUALIFICATION_CASE_RESULT"
CASE_ID = "hf-full-model-logit-parity"
HF_SCRIPT = ROOT / "scripts/qualification/qwen35_hf_logits.py"
HF_HOST_RESERVE_GIB = 7
HF_MAX_MEMORY_GIB = 16
HF_MIN_AVAILABLE_GIB = HF_MAX_MEMORY_GIB + HF_HOST_RESERVE_GIB
HF_RUNTIME_MAX_SECONDS = 600
VULKAN_MAX_MEMORY_GIB = 17
VULKAN_MIN_AVAILABLE_GIB = VULKAN_MAX_MEMORY_GIB + HF_HOST_RESERVE_GIB
MAX_RESULT_DETAILS_CHARACTERS = 2048
HF_PASS_PREFIX = "KILN_HF_FULL_LOGIT_REFERENCE_PASS "
HF_EVIDENCE_KEYS = {
    "argmax",
    "device",
    "duration_seconds",
    "logits_sha256",
    "memory_high_events",
    "memory_max_events",
    "memory_oom_events",
    "memory_oom_kill_events",
    "memory_peak_bytes",
    "memory_swap_bytes",
    "output_bytes",
    "torch_hip_version",
    "torch_version",
    "transformers_version",
    "vocab",
}
RUST_PASS_RE = re.compile(
    r"KILN_VULKAN_HF_FULL_LOGIT_PASS "
    r"vocab=(?P<vocab>[1-9][0-9]*) "
    r"argmax_equal=(?P<argmax_equal>[01]) "
    r"hf_argmax=(?P<hf_argmax>[0-9]+) "
    r"kiln_argmax=(?P<kiln_argmax>[0-9]+) "
    r"top10_overlap=(?P<top10_overlap>[0-9]+) "
    r"max_abs=(?P<max_abs>\S+) "
    r"mean_abs=(?P<mean_abs>\S+) "
    r"cosine=(?P<cosine>\S+)"
)


class QualificationError(RuntimeError):
    """The full-model oracle failed a declared qualification invariant."""


def _absolute_invocation_path(path: Path) -> Path:
    """Anchor a caller-provided path without dereferencing its final component."""
    return Path(os.path.abspath(path))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _available_gib(meminfo: Path = Path("/proc/meminfo")) -> int:
    try:
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) // 1024 // 1024
    except (OSError, UnicodeError, ValueError) as exc:
        raise QualificationError(f"cannot read host MemAvailable: {exc}") from exc
    raise QualificationError("host MemAvailable is absent from /proc/meminfo")


def _bounded_memory_limit_gib(available_gib: int) -> int:
    if available_gib < HF_MIN_AVAILABLE_GIB:
        raise QualificationError(
            f"refusing HF oracle with {available_gib} GiB available; "
            f"require at least {HF_MIN_AVAILABLE_GIB} GiB"
        )
    return HF_MAX_MEMORY_GIB


def _bounded_hf_command(
    *,
    unit: str,
    python: Path,
    model: Path,
    output: Path,
    temporary_directory: Path,
    memory_limit_gib: int,
) -> list[str]:
    clean_environment = [
        "/usr/bin/env",
        "-i",
        f"HOME={os.environ.get('HOME', str(Path.home()))}",
        "HF_HUB_OFFLINE=1",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        f"PATH={os.environ.get('PATH', '/usr/bin:/bin')}",
        "PYTHONHASHSEED=20260715",
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        f"TMPDIR={temporary_directory}",
        "TOKENIZERS_PARALLELISM=false",
        "TRANSFORMERS_OFFLINE=1",
    ]
    return [
        "systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--pipe",
        "--quiet",
        "--same-dir",
        "--unit",
        unit,
        "-p",
        "Type=exec",
        "-p",
        f"MemoryMax={memory_limit_gib}G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=kill",
        "-p",
        "KillMode=control-group",
        "-p",
        "SendSIGKILL=yes",
        "-p",
        "TimeoutStopSec=15s",
        "-p",
        f"RuntimeMaxSec={HF_RUNTIME_MAX_SECONDS}s",
        "-p",
        "PrivateNetwork=yes",
        *clean_environment,
        str(python),
        str(HF_SCRIPT),
        "--model",
        str(model),
        "--output",
        str(output),
    ]


def _parse_hf_evidence(output: str) -> dict[str, Any]:
    records = [
        line[len(HF_PASS_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(HF_PASS_PREFIX)
    ]
    if len(records) != 1:
        raise QualificationError(
            f"expected one bounded HF reference marker, found {len(records)}"
        )
    try:
        evidence = json.loads(records[0])
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise QualificationError(f"bounded HF reference marker is invalid JSON: {exc}") from exc
    if not isinstance(evidence, dict) or set(evidence) != HF_EVIDENCE_KEYS:
        actual = sorted(evidence) if isinstance(evidence, dict) else type(evidence).__name__
        raise QualificationError(f"bounded HF reference fields are not closed: {actual}")
    integer_fields = (
        "argmax",
        "memory_high_events",
        "memory_max_events",
        "memory_oom_events",
        "memory_oom_kill_events",
        "memory_peak_bytes",
        "memory_swap_bytes",
        "output_bytes",
        "vocab",
    )
    for name in integer_fields:
        value = evidence[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualificationError(f"bounded HF reference {name} is not a nonnegative integer")
    duration = evidence["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise QualificationError("bounded HF reference duration is not numeric")
    if not math.isfinite(float(duration)) or duration <= 0:
        raise QualificationError("bounded HF reference duration is not positive and finite")
    for name in (
        "device",
        "logits_sha256",
        "torch_hip_version",
        "torch_version",
        "transformers_version",
    ):
        if not isinstance(evidence[name], str) or not evidence[name]:
            raise QualificationError(f"bounded HF reference {name} is not a nonempty string")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", evidence["logits_sha256"]) is None:
        raise QualificationError("bounded HF reference logits_sha256 is not canonical")
    return evidence


def _run_hf_reference(
    *, python: Path, model: Path, output: Path, workspace: Path
) -> dict[str, Any]:
    if not output.is_absolute() or not workspace.is_absolute():
        raise QualificationError("HF reference paths must be absolute")
    available = _available_gib()
    limit = _bounded_memory_limit_gib(available)
    unit = f"kiln-hf-oracle-bounded-{uuid.uuid4().hex}.service"
    temporary_directory = workspace / "tmp"
    temporary_directory.mkdir(mode=0o700)
    command = _bounded_hf_command(
        unit=unit,
        python=python,
        model=model,
        output=output,
        temporary_directory=temporary_directory,
        memory_limit_gib=limit,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=HF_RUNTIME_MAX_SECONDS + 60,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise QualificationError(
            f"bounded HF reference exited {completed.returncode}: {completed.stderr[-2000:]}"
        )
    evidence = _parse_hf_evidence(completed.stdout)
    if evidence["memory_peak_bytes"] <= 0:
        raise QualificationError("bounded HF service reported zero peak memory")
    if evidence["memory_swap_bytes"] != 0:
        raise QualificationError(
            f"bounded HF service used {evidence['memory_swap_bytes']} bytes of swap"
        )
    if evidence["memory_oom_events"] != 0 or evidence["memory_oom_kill_events"] != 0:
        raise QualificationError(
            "bounded HF service reported "
            f"oom={evidence['memory_oom_events']} "
            f"oom_kill={evidence['memory_oom_kill_events']}"
        )
    if not output.is_file() or output.is_symlink():
        raise QualificationError("bounded HF reference did not create a regular artifact")
    if not (900_000 <= output.stat().st_size <= 1_100_000):
        raise QualificationError(
            f"HF reference artifact has unexpected size {output.stat().st_size}"
        )
    if evidence["output_bytes"] != output.stat().st_size:
        raise QualificationError("bounded HF reference reported the wrong artifact size")
    if evidence["vocab"] != 248_320:
        raise QualificationError("bounded HF reference reported the wrong vocabulary width")
    return {
        "available_before_gib": available,
        "memory_limit_gib": limit,
        "memory_high_events": evidence["memory_high_events"],
        "memory_max_events": evidence["memory_max_events"],
        "memory_peak_bytes": evidence["memory_peak_bytes"],
        "logits_sha256": evidence["logits_sha256"],
        "oom": evidence["memory_oom_events"],
        "oom_kill": evidence["memory_oom_kill_events"],
        "reference_sha256": _sha256_file(output),
        "swap_bytes": evidence["memory_swap_bytes"],
    }


def _wait_for_headroom(timeout_seconds: float = 60.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        available = _available_gib()
        if available >= VULKAN_MIN_AVAILABLE_GIB:
            return available
        if time.monotonic() >= deadline:
            raise QualificationError(
                f"host recovered only {available} GiB before the Vulkan forward; "
                f"require {VULKAN_MIN_AVAILABLE_GIB} GiB for a "
                f"{VULKAN_MAX_MEMORY_GIB} GiB service plus the "
                f"{HF_HOST_RESERVE_GIB} GiB host reserve"
            )
        time.sleep(1.0)


def _parse_rust_metrics(output: str) -> dict[str, int | float]:
    matches = list(RUST_PASS_RE.finditer(output))
    if len(matches) != 1:
        raise QualificationError(f"expected one Vulkan/HF pass marker, found {len(matches)}")
    raw = matches[0].groupdict()
    values: dict[str, int | float] = {}
    for name in (
        "vocab",
        "argmax_equal",
        "hf_argmax",
        "kiln_argmax",
        "top10_overlap",
    ):
        values[name] = int(raw[name])
    for name in ("max_abs", "mean_abs", "cosine"):
        value = float(raw[name])
        if not math.isfinite(value):
            raise QualificationError(f"Vulkan/HF metric {name} is non-finite")
        values[name] = value
    return values


def _run_vulkan_comparison(
    *, model: Path, reference: Path
) -> tuple[dict[str, int | float], str]:
    if not model.is_absolute() or not reference.is_absolute():
        raise QualificationError("Vulkan comparison paths must be absolute")
    environment = dict(os.environ)
    environment.update(
        {
            "CARGO_NET_OFFLINE": "true",
            "KILN_CARGO_MAX_MEMORY_GIB": str(VULKAN_MAX_MEMORY_GIB),
            "KILN_QUALIFICATION": "1",
            "KILN_QUALIFICATION_HF_LOGITS_PATH": str(reference),
            "KILN_QUALIFICATION_MODEL_PATH": str(model),
        }
    )
    completed = subprocess.run(
        [
            "scripts/qualification/cargo-test-bounded.sh",
            "test",
            "--locked",
            "--offline",
            "-p",
            "kiln-model",
            "--no-default-features",
            "--features",
            "vulkan",
            "--test",
            "vk_resident_decode_parity",
            "vk_resident_decode_matches_nonresident_on_qwen35_4b",
            "--",
            "--nocapture",
            "--test-threads=1",
        ],
        cwd=ROOT,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    combined = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        raise QualificationError(
            f"bounded Vulkan comparison exited {completed.returncode}: {combined[-3000:]}"
        )
    return _parse_rust_metrics(combined), combined


def _metric(name: str, value: int | float, unit: str, lower: bool) -> dict[str, Any]:
    return {
        "aggregation": "single_run",
        "lower_is_better": lower,
        "name": name,
        "unit": unit,
        "value": value,
    }


def _result_document(
    *, duration: float, hf: dict[str, Any], comparison: dict[str, int | float]
) -> dict[str, Any]:
    details = {
        "hf_memory_limit_gib": hf["memory_limit_gib"],
        "hf_memory_high_events": hf["memory_high_events"],
        "hf_memory_max_events": hf["memory_max_events"],
        "hf_logits_sha256": hf["logits_sha256"],
        "hf_reference_sha256": hf["reference_sha256"],
        "hf_swap_bytes": hf["swap_bytes"],
        "hf_torch_path": "pinned_rocm_torch_fallback",
        "input_token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 100],
        "kiln_path": "vulkan_resident_and_nonresident",
        "kiln_memory_limit_gib": VULKAN_MAX_MEMORY_GIB,
    }
    details_text = _canonical_json(details).decode("ascii")
    if len(details_text) > MAX_RESULT_DETAILS_CHARACTERS:
        raise QualificationError("qualification result details exceed receipt bound")
    metrics = [
        _metric("argmax_equal", comparison["argmax_equal"], "bool", False),
        _metric("cosine_similarity", comparison["cosine"], "ratio", False),
        _metric("hf_peak_memory_bytes", hf["memory_peak_bytes"], "bytes", True),
        _metric("hf_swap_bytes", hf["swap_bytes"], "bytes", True),
        _metric("max_abs_error", comparison["max_abs"], "logit", True),
        _metric("mean_abs_error", comparison["mean_abs"], "logit", True),
        _metric("top10_overlap", comparison["top10_overlap"], "count", False),
        _metric("vocab_logits", comparison["vocab"], "count", False),
    ]
    tolerances = [
        {"absolute_tolerance": 0.0, "metric": "argmax_equal", "relative_tolerance": 0.0},
        {"absolute_tolerance": 0.0, "metric": "cosine_similarity", "relative_tolerance": 0.0001},
        {"absolute_tolerance": 0.5, "metric": "max_abs_error", "relative_tolerance": 0.0},
        {"absolute_tolerance": 0.05, "metric": "mean_abs_error", "relative_tolerance": 0.0},
        {"absolute_tolerance": 1.0, "metric": "top10_overlap", "relative_tolerance": 0.0},
    ]
    return {
        "case_id": CASE_ID,
        "details": details_text,
        "duration_seconds": duration,
        "effective_config": {
            "hf_attention": "eager",
            "hf_linear_attention": "torch_fallback",
            "hf_memory_max_gib": HF_MAX_MEMORY_GIB,
            "input_token_count": 9,
            "kiln_backend": "vulkan",
            "kiln_memory_max_gib": VULKAN_MAX_MEMORY_GIB,
            "network": "forbidden",
        },
        "metrics": sorted(metrics, key=lambda item: item["name"]),
        "schema_version": 1,
        "status": "passed",
        "tolerances": sorted(tolerances, key=lambda item: item["metric"]),
    }


def _validate_inputs(model: Path, python: Path) -> tuple[Path, Path]:
    model = model.absolute()
    if model.is_symlink():
        raise QualificationError("--model must be a non-symlink directory")
    model = model.resolve(strict=True)
    if not model.is_dir():
        raise QualificationError("--model must be a non-symlink directory")
    python = python.absolute()
    try:
        metadata = python.stat()
    except OSError as exc:
        raise QualificationError(f"cannot inspect --trainer-python: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(python, os.X_OK):
        raise QualificationError("--trainer-python must resolve to an executable file")
    return model, python


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trainer-python", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result_text = os.environ.get(RESULT_ENV)
    if not result_text:
        print(f"error: {RESULT_ENV} is required", file=sys.stderr)
        return 2
    result_path = _absolute_invocation_path(Path(result_text))
    if result_path.exists() or result_path.is_symlink():
        print(f"error: refusing to replace {result_path}", file=sys.stderr)
        return 2
    workspace = result_path.parent / "hf-full-model-logit-parity"
    try:
        workspace.mkdir(mode=0o700)
        model, python = _validate_inputs(args.model, args.trainer_python)
    except (OSError, QualificationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    started = time.monotonic()
    reference = workspace / "hf-reference.safetensors"
    try:
        hf = _run_hf_reference(
            python=python, model=model, output=reference, workspace=workspace
        )
        hf["available_after_gib"] = _wait_for_headroom()
        comparison, _ = _run_vulkan_comparison(model=model, reference=reference)
        result = _result_document(
            duration=time.monotonic() - started,
            hf=hf,
            comparison=comparison,
        )
        _write_json_new(result_path, result)
    except BaseException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "KILN_VULKAN_HF_MODEL_ORACLE_PASS "
        f"vocab={comparison['vocab']} argmax_equal={comparison['argmax_equal']} "
        f"top10_overlap={comparison['top10_overlap']} max_abs={comparison['max_abs']:.8g} "
        f"mean_abs={comparison['mean_abs']:.8g} cosine={comparison['cosine']:.10g} "
        f"hf_peak_bytes={hf['memory_peak_bytes']} hf_swap_bytes={hf['swap_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
