#!/usr/bin/env python3
"""Run the production-model Kiln -> HF/TRL -> Kiln qualification path.

This command is intentionally self-contained so the committed qualification
workload can run it inside the repository's loopback-only network sandbox. It
never lets the Kiln server and the external Torch trainer own the accelerator
at the same time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULT_ENV = "KILN_QUALIFICATION_CASE_RESULT"
CASE_ID = "hf-trl-production-roundtrip"
PORT = 18420
URL = f"http://127.0.0.1:{PORT}"
SERVER_START_TIMEOUT_SECONDS = 180.0
SERVER_STOP_TIMEOUT_SECONDS = 120.0
COMMAND_TIMEOUT_SECONDS = 900.0
MAX_ARCHIVE_FILES = 32
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_RESULT_DETAILS_CHARACTERS = 2048
SUPPORTED_BACKENDS = {"rocm", "vulkan"}


class RoundTripError(RuntimeError):
    """A qualification invariant failed."""


@dataclass(frozen=True)
class CommandOutput:
    stdout: str
    stderr: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_bytes_new(path: Path, value: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o755 if executable else 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_json_new(path: Path, value: Any) -> None:
    _write_bytes_new(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoundTripError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RoundTripError(f"{path} must contain one JSON object")
    return value


def _tail(value: str, limit: int = 3000) -> str:
    return value[-limit:].strip()


def _invocation_path(path: Path) -> Path:
    """Make an invocation absolute without resolving a venv interpreter symlink."""

    return Path(os.path.abspath(path))


def _run(
    argv: Iterable[str | Path],
    *,
    workspace: Path,
    label: str,
    environment: dict[str, str] | None = None,
    timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
) -> CommandOutput:
    command = tuple(str(item) for item in argv)
    if not command:
        raise RoundTripError("refusing to run an empty command")
    process_environment = dict(os.environ)
    if environment:
        process_environment.update(environment)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=process_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RoundTripError(f"{label} could not complete: {exc}") from exc
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    _write_bytes_new(workspace / f"{label}.stdout.log", completed.stdout)
    _write_bytes_new(workspace / f"{label}.stderr.log", completed.stderr)
    if completed.returncode != 0:
        detail = _tail(stderr) or _tail(stdout) or "no command output"
        raise RoundTripError(
            f"{label} exited with {completed.returncode}: {detail}"
        )
    return CommandOutput(stdout, stderr)


def _parse_json_output(output: CommandOutput, label: str) -> dict[str, Any]:
    try:
        value = json.loads(output.stdout)
    except json.JSONDecodeError as exc:
        raise RoundTripError(f"{label} did not emit one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise RoundTripError(f"{label} output must be one JSON object")
    return value


def _parse_last_json_line(output: CommandOutput, label: str) -> dict[str, Any]:
    for line in reversed(output.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RoundTripError(f"{label} did not emit a JSON object line")


def _http_json(path: str, *, timeout_seconds: float = 5.0) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(f"{URL}{path}", method="GET")
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    except (OSError, urllib.error.URLError) as exc:
        raise RoundTripError(f"GET {path} failed: {exc}") from exc
    else:
        status = response.status
        raw = response.read()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoundTripError(f"GET {path} returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RoundTripError(f"GET {path} did not return a JSON object")
    return status, value


def _validate_health(
    health: dict[str, Any],
    backend: str,
    *,
    final: bool,
) -> dict[str, Any]:
    if health.get("status") != "ok":
        raise RoundTripError(f"server health status is not ok: {health.get('status')!r}")
    runtime = health.get("backend_runtime")
    if not isinstance(runtime, dict):
        raise RoundTripError("server health omitted backend_runtime")
    if runtime.get("healthy") is not True:
        raise RoundTripError(f"backend is unhealthy: {runtime.get('reason')!r}")
    if runtime.get("quarantined") is not False:
        raise RoundTripError(f"backend is quarantined: {runtime.get('reason')!r}")
    if runtime.get("restart_required") is not False:
        raise RoundTripError("backend reports restart_required")
    identity = health.get("execution_identity")
    if not isinstance(identity, dict) or identity.get("backend") != backend:
        actual = identity.get("backend") if isinstance(identity, dict) else None
        raise RoundTripError(
            f"server execution backend {actual!r} does not match {backend!r}"
        )
    provenance = identity.get("provenance_sha256")
    if not isinstance(provenance, str) or not provenance.startswith("sha256:"):
        raise RoundTripError("server health omitted execution provenance")
    checks = health.get("checks")
    if not isinstance(checks, list) or not checks:
        raise RoundTripError("server health omitted readiness checks")
    failed = [item for item in checks if not isinstance(item, dict) or item.get("pass") is not True]
    if failed:
        raise RoundTripError(f"server health contains failed checks: {failed!r}")
    if final:
        requests = health.get("requests")
        scheduler = health.get("scheduler")
        if not isinstance(requests, dict) or requests.get("error") != 0:
            raise RoundTripError(f"server recorded request errors: {requests!r}")
        if not isinstance(scheduler, dict):
            raise RoundTripError("server health omitted scheduler state")
        if scheduler.get("waiting") != 0 or scheduler.get("running") != 0:
            raise RoundTripError(f"server did not drain scheduler work: {scheduler!r}")
        if scheduler.get("blocks_used") != 0:
            raise RoundTripError(f"server leaked KV blocks: {scheduler!r}")
    return identity


class _Server:
    def __init__(
        self,
        *,
        kiln: Path,
        model: Path,
        adapter_dir: Path,
        backend: str,
        workspace: Path,
        phase: str,
    ) -> None:
        self._backend = backend
        self._log_path = workspace / f"server-{phase}.log"
        self._log = self._log_path.open("xb")
        environment = dict(os.environ)
        environment.update(
            {
                "KILN_MODEL_PATH": str(model),
                "KILN_ADAPTER_DIR": str(adapter_dir),
                "KILN_HOST": "127.0.0.1",
                "KILN_PORT": str(PORT),
                "KILN_NUM_BLOCKS": "256",
                "KILN_SERVING_PROFILE": "experimental",
                "KILN_CUDA_GRAPHS": "0",
                "KILN_ACCELERATOR_ROCM_GRAPH_MODE": "disabled",
                "KILN_LOG_LEVEL": "info",
            }
        )
        try:
            self.process = subprocess.Popen(
                [str(kiln), "serve", "--eval-mode"],
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            self._log.close()
            raise

    def wait_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            returncode = self.process.poll()
            if returncode is not None:
                self._log.flush()
                detail = _tail(self._log_path.read_text(errors="replace"))
                raise RoundTripError(
                    f"server exited before readiness with {returncode}: {detail}"
                )
            try:
                status, health = _http_json("/health", timeout_seconds=2.0)
                if status == 200:
                    return _validate_health(health, self._backend, final=False)
                last_error = f"health returned HTTP {status}"
            except RoundTripError as exc:
                last_error = str(exc)
            time.sleep(0.25)
        raise RoundTripError(
            f"server did not become ready within {SERVER_START_TIMEOUT_SECONDS:g}s: {last_error}"
        )

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=SERVER_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=15)
                raise RoundTripError("server did not stop after SIGINT") from exc
        self._log.flush()
        self._log.close()
        if self.process.returncode != 0:
            detail = _tail(self._log_path.read_text(errors="replace"))
            raise RoundTripError(
                f"server shutdown returned {self.process.returncode}: {detail}"
            )

    def abort(self) -> None:
        try:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGINT)
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=15)
        finally:
            if not self._log.closed:
                self._log.close()


def _safe_extract_bundle(archive: Path, destination: Path, root_name: str) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    total_bytes = 0
    file_count = 0
    try:
        bundle = tarfile.open(archive, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise RoundTripError(f"cannot open {archive}: {exc}") from exc
    with bundle:
        for member in bundle:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise RoundTripError(f"archive contains unsafe path {member.name!r}")
            if pure.parts[0] != root_name:
                raise RoundTripError(
                    f"archive root {pure.parts[0]!r} does not match {root_name!r}"
                )
            normalized = pure.as_posix()
            if normalized in seen:
                raise RoundTripError(f"archive contains duplicate path {normalized!r}")
            seen.add(normalized)
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise RoundTripError(f"archive contains non-regular entry {normalized!r}")
            file_count += 1
            total_bytes += member.size
            if file_count > MAX_ARCHIVE_FILES or total_bytes > MAX_ARCHIVE_BYTES:
                raise RoundTripError("archive exceeds qualification extraction bounds")
            target = destination.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RoundTripError(f"cannot read archive member {normalized!r}")
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                os.close(descriptor)
    extracted = destination / root_name
    if file_count == 0 or not extracted.is_dir():
        raise RoundTripError("archive did not contain a nonempty bundle root")
    return extracted


def _prepare_fixtures(workspace: Path) -> dict[str, Path]:
    fixtures = workspace / "fixtures"
    fixtures.mkdir()
    paths = {
        "sft": fixtures / "sft.jsonl",
        "grpo_tasks": fixtures / "grpo.tasks.jsonl",
        "grpo_request": fixtures / "grpo.request.json",
        "grpo_scorer": fixtures / "grpo_scorer.py",
        "eval_tasks": fixtures / "eval.tasks.jsonl",
        "eval_request": fixtures / "eval.request.json",
        "eval_scorer": fixtures / "eval_scorer.py",
    }
    _write_bytes_new(
        paths["sft"],
        (
            b'{"messages":[{"role":"user","content":'
            b'"Reply with exactly SFT_ROUNDTRIP_OK."},'
            b'{"role":"assistant","content":"SFT_ROUNDTRIP_OK"}]}\n'
        ),
    )
    _write_bytes_new(
        paths["grpo_tasks"],
        b'{"id":"grpo-roundtrip-1","prompt":"Reply with one short word.","answer":"kiln"}\n',
    )
    _write_json_new(
        paths["grpo_request"],
        {
            "max_tokens": 8,
            "messages": [{"content": "{{prompt}}", "role": "user"}],
            "model": "Qwen3.5-4B",
            "temperature": 0.8,
        },
    )
    _write_bytes_new(
        paths["grpo_scorer"],
        (
            b'#!/usr/bin/env python3\nimport json\nimport sys\n'
            b'row = json.load(sys.stdin)\nprint(float(row["seed"] % 2))\n'
        ),
        executable=True,
    )
    _write_bytes_new(
        paths["eval_tasks"],
        b'{"id":"roundtrip-eval-1","prompt":"Reply with exactly SFT_ROUNDTRIP_OK."}\n',
    )
    _write_json_new(
        paths["eval_request"],
        {
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 16,
            "messages": [{"content": "{{prompt}}", "role": "user"}],
            "model": "Qwen3.5-4B",
            "temperature": 0.0,
        },
    )
    _write_bytes_new(
        paths["eval_scorer"],
        (
            b'#!/usr/bin/env python3\nimport json\nimport sys\n'
            b'row = json.load(sys.stdin)\n'
            b'if not row["base"]["content"] or not row["candidate"]["content"]:\n'
            b'    raise SystemExit("both base and candidate must produce content")\n'
            b'print(json.dumps({"base_score": 1.0, "adapter_score": 1.0}))\n'
        ),
        executable=True,
    )
    return paths


def _external_environment(workspace: Path) -> dict[str, str]:
    return {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(workspace / "hf-home"),
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _validate_torch_probe(probe: dict[str, Any]) -> None:
    if probe.get("accelerator_available") is not True:
        raise RoundTripError(f"external Torch accelerator is unavailable: {probe!r}")
    if probe.get("device_type") != "cuda":
        raise RoundTripError(
            f"ROCm/CUDA HF qualification requires Torch device_type='cuda': {probe!r}"
        )
    if not isinstance(probe.get("device_name"), str) or not probe["device_name"]:
        raise RoundTripError("external Torch probe omitted device_name")
    if not probe.get("hip_version") and not probe.get("cuda_version"):
        raise RoundTripError("external Torch probe omitted both HIP and CUDA runtime identity")


def _validate_training_result(bundle: Path, task: str) -> dict[str, Any]:
    result = _read_json(bundle / "kiln_hf_result.json")
    if result.get("result_type") != "kiln.hf-trl-result.v1" or result.get("task") != task:
        raise RoundTripError(f"{task} result manifest has the wrong identity")
    trainer = result.get("trainer")
    if not isinstance(trainer, dict):
        raise RoundTripError(f"{task} result omitted trainer identity")
    expected_trainer = "trl_sft_trainer" if task == "sft" else "trl_grpo_trainer"
    if trainer.get("kind") != expected_trainer:
        raise RoundTripError(
            f"{task} result trainer kind is not {expected_trainer!r}"
        )
    for key in ("torch_version", "transformers_version", "trl_version", "peft_version"):
        if not isinstance(trainer.get(key), str) or not trainer[key]:
            raise RoundTripError(f"{task} result omitted trainer.{key}")
    output = result.get("output_adapter")
    if not isinstance(output, dict):
        raise RoundTripError(f"{task} result omitted output_adapter")
    model = output.get("model")
    if not isinstance(model, dict):
        raise RoundTripError(f"{task} result omitted adapter model identity")
    model_path = bundle / "adapter_model.safetensors"
    if model.get("sha256") != _sha256_file(model_path):
        raise RoundTripError(f"{task} adapter weight digest does not match its result")
    if model.get("size_bytes") != model_path.stat().st_size or model_path.stat().st_size <= 0:
        raise RoundTripError(f"{task} adapter weight size does not match its result")
    for key in ("export_sha256", "result_sha256"):
        value = result.get(key)
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise RoundTripError(f"{task} result omitted {key}")
    effective = result.get("effective_config")
    if not isinstance(effective, dict):
        raise RoundTripError(f"{task} result omitted effective_config")
    required = {
        "lora_rank": ("unsigned", 1),
        "target_modules": ("text", "q_proj"),
    }
    if task == "sft":
        required["dataset_rows"] = ("unsigned", 1)
    else:
        required.update(
            {
                "behavior_policy": ("text", "recorded"),
                "dataset_completions": ("unsigned", 2),
                "dataset_groups": ("unsigned", 1),
                "kl_reference_policy": ("text", "base_model"),
                "num_generations": ("unsigned", 2),
            }
        )
    for name, (kind, value) in required.items():
        if effective.get(name) != {"kind": kind, "value": value}:
            raise RoundTripError(
                f"{task} effective_config.{name} does not equal {kind}:{value!r}"
            )
    if task == "grpo":
        sampled = effective.get("dataset_sampled_action_tokens")
        if (
            not isinstance(sampled, dict)
            or sampled.get("kind") != "unsigned"
            or not isinstance(sampled.get("value"), int)
            or isinstance(sampled.get("value"), bool)
            or sampled["value"] <= 0
        ):
            raise RoundTripError("grpo result has no sampled action tokens")
    return result


def _validate_import_receipt(
    adapter_dir: Path,
    adapter: str,
    task: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    receipt = _read_json(adapter_dir / adapter / "kiln_hf_import.json")
    expected = {
        "adapter_name": adapter,
        "export_sha256": result["export_sha256"],
        "import_type": "kiln.hf-trl-import.v1",
        "result_sha256": result["result_sha256"],
        "task": task,
        "used_exported_reference_script": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RoundTripError(f"{task} import receipt {key} does not match")
    digest = receipt.get("import_sha256")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RoundTripError(f"{task} import receipt omitted import_sha256")
    resident = receipt.get("resident_model")
    manifest = resident.get("base_weight_shard_manifest") if isinstance(resident, dict) else None
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("aggregate_sha256"), str
    ):
        raise RoundTripError(f"{task} import receipt omitted resident base identity")
    return receipt


def _validate_adapter_verification(receipt: dict[str, Any], task: str) -> int:
    if receipt.get("status") != "ok":
        raise RoundTripError(f"{task} adapter verification did not pass")
    checks = receipt.get("checks")
    server = receipt.get("server")
    tensors = receipt.get("tensor_summary")
    if not isinstance(checks, list) or any(item.get("pass") is not True for item in checks):
        raise RoundTripError(f"{task} adapter verification contains failed offline checks")
    if not isinstance(server, dict):
        raise RoundTripError(f"{task} adapter verification omitted server checks")
    server_checks = server.get("checks")
    if not isinstance(server_checks, list) or any(
        item.get("pass") is not True for item in server_checks
    ):
        raise RoundTripError(f"{task} adapter verification contains failed server checks")
    if not isinstance(tensors, dict):
        raise RoundTripError(f"{task} adapter verification omitted tensor summary")
    nonzero = tensors.get("nonzero_tensor_count")
    if not isinstance(nonzero, int) or isinstance(nonzero, bool) or nonzero <= 0:
        raise RoundTripError(f"{task} adapter has no nonzero tensors")
    delta = receipt.get("logit_delta_summary")
    if not isinstance(delta, dict) or delta.get("measurable") is not True:
        raise RoundTripError(f"{task} adapter has no measurable delta")
    return nonzero


def _validate_eval(summary: dict[str, Any], adapter: str) -> None:
    if summary.get("adapter") != adapter or summary.get("pair_count") != 1:
        raise RoundTripError(f"{adapter} eval summary has the wrong identity or pair count")
    warnings = summary.get("warnings")
    if warnings != []:
        raise RoundTripError(f"{adapter} eval emitted warnings: {warnings!r}")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RoundTripError(f"{adapter} eval omitted its paired result")
    pair = results[0]
    if pair.get("base_score") != 1.0 or pair.get("adapter_score") != 1.0:
        raise RoundTripError(f"{adapter} eval did not score both inference paths")
    if not pair.get("base_content") or not pair.get("adapter_content"):
        raise RoundTripError(f"{adapter} eval produced empty public answer content")
    hashes = summary.get("adapter_hashes")
    if not isinstance(hashes, list) or not any(
        item.get("name") == adapter and item.get("adapter_model_sha256")
        for item in hashes
        if isinstance(item, dict)
    ):
        raise RoundTripError(f"{adapter} eval omitted the loaded adapter hash")


def _effective_config() -> dict[str, Any]:
    return {
        "external_accelerator": "torch_cuda",
        "external_trainer": "hf_trl_peft",
        "lora_rank": 1,
        "lora_target": "q_proj",
        "network": "loopback_only",
        "server_eval_mode": True,
        "training_updates_per_task": 1,
    }


def _metric(name: str, value: int | float, unit: str, lower: bool) -> dict[str, Any]:
    return {
        "aggregation": "single_run",
        "lower_is_better": lower,
        "name": name,
        "unit": unit,
        "value": value,
    }


def _result_document(
    *,
    duration: float,
    metrics: list[dict[str, Any]],
    details: dict[str, Any],
) -> dict[str, Any]:
    details_text = _canonical_json(details).decode("ascii")
    if len(details_text) > MAX_RESULT_DETAILS_CHARACTERS:
        raise RoundTripError("qualification result details exceed the receipt bound")
    return {
        "case_id": CASE_ID,
        "details": details_text,
        "duration_seconds": duration,
        "effective_config": _effective_config(),
        "metrics": sorted(metrics, key=lambda item: item["name"]),
        "schema_version": 1,
        "status": "passed",
        "tolerances": [],
    }


def _roundtrip(args: argparse.Namespace, workspace: Path) -> dict[str, Any]:
    backend: str = args.backend
    model = args.model.resolve(strict=True)
    kiln = _invocation_path(args.kiln)
    trainer_python = _invocation_path(args.trainer_python)
    if not model.is_dir() or model.is_symlink():
        raise RoundTripError("--model must be a non-symlink directory")
    try:
        kiln_metadata = kiln.lstat()
        trainer_metadata = trainer_python.stat()
    except OSError as exc:
        raise RoundTripError(f"cannot inspect an executable input: {exc}") from exc
    if (
        not stat.S_ISREG(kiln_metadata.st_mode)
        or kiln.is_symlink()
        or not os.access(kiln, os.X_OK)
    ):
        raise RoundTripError("--kiln must be an executable non-symlink file")
    # A venv's bin/python is normally a symlink. Invoking that path is what
    # activates pyvenv.cfg discovery; resolving it would silently use the base
    # interpreter and lose the pinned training environment.
    if not stat.S_ISREG(trainer_metadata.st_mode) or not os.access(trainer_python, os.X_OK):
        raise RoundTripError("--trainer-python must resolve to an executable regular file")

    adapter_dir = workspace / "adapters"
    adapter_dir.mkdir()
    fixtures = _prepare_fixtures(workspace)
    archives = workspace / "archives"
    archives.mkdir()
    extracted = workspace / "extracted"
    external_environment = _external_environment(workspace)

    torch_probe_output = _run(
        [
            trainer_python,
            "-c",
            (
                "import json,torch; "
                "ok=torch.cuda.is_available(); "
                "print(json.dumps({'accelerator_available':ok,'device_type':'cuda' if ok else None,"
                "'device_name':torch.cuda.get_device_name(0) if ok else None,"
                "'torch_version':torch.__version__,'hip_version':torch.version.hip,"
                "'cuda_version':torch.version.cuda,"
                "'bf16_supported':torch.cuda.is_bf16_supported() if ok else False},"
                "sort_keys=True))"
            ),
        ],
        workspace=workspace,
        label="torch-probe",
        environment=external_environment,
    )
    torch_probe = _parse_last_json_line(torch_probe_output, "torch probe")
    _validate_torch_probe(torch_probe)
    if torch_probe.get("bf16_supported") is not True:
        raise RoundTripError("external Torch device does not report BF16 support")

    first_server = _Server(
        kiln=kiln,
        model=model,
        adapter_dir=adapter_dir,
        backend=backend,
        workspace=workspace,
        phase="export",
    )
    try:
        export_identity = first_server.wait_ready()
        rollout_path = workspace / "grpo.rollouts.jsonl"
        rollout_summary_path = workspace / "grpo.rollouts.summary.json"
        _run(
            [
                kiln,
                "rollout-generate",
                "--adapter",
                "base",
                "--thinking",
                "false",
                "--tasks",
                fixtures["grpo_tasks"],
                "--seeds",
                "2",
                "--seed-start",
                str(args.seed),
                "--request-template",
                fixtures["grpo_request"],
                "--scorer",
                fixtures["grpo_scorer"],
                "--output",
                rollout_path,
                "--summary-output",
                rollout_summary_path,
                "--url",
                URL,
            ],
            workspace=workspace,
            label="rollout-generate",
        )
        rollout_summary = _read_json(rollout_summary_path)
        if (
            rollout_summary.get("group_count") != 1
            or rollout_summary.get("completion_count") != 2
            or rollout_summary.get("warnings") != []
        ):
            raise RoundTripError("recorded GRPO rollout summary failed its exact shape")
        stats = rollout_summary.get("stats")
        if not isinstance(stats, dict) or stats.get("mean_reward") != 0.5:
            raise RoundTripError("recorded GRPO rollout rewards are not deterministic 0/1")

        sft_archive = archives / "qualification-sft.kiln-hf.tar.gz"
        grpo_archive = archives / "qualification-grpo.kiln-hf.tar.gz"
        _run(
            [
                kiln,
                "train",
                "hf",
                "export-sft",
                "--file",
                fixtures["sft"],
                "--name",
                "qualification-sft",
                "--output",
                sft_archive,
                "--url",
                URL,
            ],
            workspace=workspace,
            label="export-sft",
        )
        _run(
            [
                kiln,
                "train",
                "hf",
                "export-grpo",
                "--file",
                rollout_path,
                "--name",
                "qualification-grpo",
                "--output",
                grpo_archive,
                "--url",
                URL,
            ],
            workspace=workspace,
            label="export-grpo",
        )
    except BaseException:
        first_server.abort()
        raise
    else:
        first_server.stop()

    sft_bundle = _safe_extract_bundle(
        sft_archive, extracted / "sft", "qualification-sft.kiln-hf"
    )
    grpo_bundle = _safe_extract_bundle(
        grpo_archive, extracted / "grpo", "qualification-grpo.kiln-hf"
    )

    train_common = [
        "--base-model",
        model,
        "--learning-rate",
        "1e-3",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--seed",
        str(args.seed),
        "--dtype",
        "bfloat16",
        "--lr-scheduler",
        "constant",
        "--warmup-ratio",
        "0",
        "--weight-decay",
        "0",
        "--max-grad-norm",
        "1",
        "--lora-rank",
        "1",
        "--lora-alpha",
        "1",
        "--lora-dropout",
        "0",
        "--target-modules",
        "q_proj",
        "--no-gradient-checkpointing",
    ]
    for task, bundle in (("sft", sft_bundle), ("grpo", grpo_bundle)):
        _run(
            [trainer_python, bundle / "train.py", bundle, "--base-model", model, "--verify-only"],
            workspace=workspace,
            label=f"verify-{task}-bundle",
            environment=external_environment,
        )
        _run(
            [trainer_python, bundle / "train.py", bundle, *train_common],
            workspace=workspace,
            label=f"train-{task}",
            environment=external_environment,
        )

    sft_result = _validate_training_result(sft_bundle, "sft")
    grpo_result = _validate_training_result(grpo_bundle, "grpo")

    second_server = _Server(
        kiln=kiln,
        model=model,
        adapter_dir=adapter_dir,
        backend=backend,
        workspace=workspace,
        phase="import",
    )
    try:
        import_identity = second_server.wait_ready()
        if import_identity.get("provenance_sha256") != export_identity.get("provenance_sha256"):
            raise RoundTripError("server execution provenance changed across the GPU handoff")
        for adapter, bundle in (
            ("roundtrip-sft", sft_bundle),
            ("roundtrip-grpo", grpo_bundle),
        ):
            _run(
                [
                    kiln,
                    "train",
                    "hf",
                    "import-peft",
                    "--bundle",
                    bundle,
                    "--name",
                    adapter,
                    "--url",
                    URL,
                ],
                workspace=workspace,
                label=f"import-{adapter}",
            )

        import_receipts = {
            "sft": _validate_import_receipt(
                adapter_dir, "roundtrip-sft", "sft", sft_result
            ),
            "grpo": _validate_import_receipt(
                adapter_dir, "roundtrip-grpo", "grpo", grpo_result
            ),
        }

        nonzero_counts: dict[str, int] = {}
        for task, adapter in (("sft", "roundtrip-sft"), ("grpo", "roundtrip-grpo")):
            verify_output = _run(
                [
                    kiln,
                    "adapters",
                    "verify",
                    adapter,
                    "--adapter-dir",
                    adapter_dir,
                    "--url",
                    URL,
                ],
                workspace=workspace,
                label=f"adapter-verify-{task}",
            )
            verification = _parse_json_output(verify_output, f"{task} adapter verifier")
            nonzero_counts[task] = _validate_adapter_verification(verification, task)

        for task, adapter in (("sft", "roundtrip-sft"), ("grpo", "roundtrip-grpo")):
            eval_path = workspace / f"eval-{task}.json"
            _run(
                [
                    kiln,
                    "eval-adapter",
                    "--adapter",
                    adapter,
                    "--tasks",
                    fixtures["eval_tasks"],
                    "--seeds",
                    "1",
                    "--request-template",
                    fixtures["eval_request"],
                    "--scorer",
                    fixtures["eval_scorer"],
                    "--output",
                    eval_path,
                    "--url",
                    URL,
                ],
                workspace=workspace,
                label=f"eval-{task}",
            )
            _validate_eval(_read_json(eval_path), adapter)

        status, adapters = _http_json("/v1/adapters")
        if status != 200:
            raise RoundTripError(f"adapter registry returned HTTP {status}")
        available = adapters.get("available")
        names = {
            item.get("name") for item in available if isinstance(item, dict)
        } if isinstance(available, list) else set()
        if names != {"roundtrip-sft", "roundtrip-grpo"}:
            raise RoundTripError(f"adapter registry has unexpected names: {sorted(names)!r}")
        status, final_health = _http_json("/health")
        if status != 200:
            raise RoundTripError(f"final health returned HTTP {status}")
        final_identity = _validate_health(final_health, backend, final=True)
        base_identity = final_health["base_weight_identity"]["aggregate_sha256"]
        for task, receipt in import_receipts.items():
            resident = receipt["resident_model"]["base_weight_shard_manifest"]
            if resident["aggregate_sha256"] != base_identity:
                raise RoundTripError(
                    f"{task} import receipt resident identity drifted from final health"
                )
    except BaseException:
        second_server.abort()
        raise
    else:
        second_server.stop()

    sft_model = sft_result["output_adapter"]["model"]
    grpo_model = grpo_result["output_adapter"]["model"]
    details = {
        "execution_provenance": final_identity["provenance_sha256"],
        "grpo_export": grpo_result["export_sha256"],
        "grpo_import": import_receipts["grpo"]["import_sha256"],
        "grpo_result": grpo_result["result_sha256"],
        "grpo_weights": grpo_model["sha256"],
        "kiln_binary": _sha256_file(kiln),
        "model_base_weights": final_health["base_weight_identity"]["aggregate_sha256"],
        "sft_export": sft_result["export_sha256"],
        "sft_import": import_receipts["sft"]["import_sha256"],
        "sft_result": sft_result["result_sha256"],
        "sft_weights": sft_model["sha256"],
        "torch_device": torch_probe["device_name"],
        "torch_version": torch_probe["torch_version"],
        "transformers_version": sft_result["trainer"]["transformers_version"],
        "trl_version": sft_result["trainer"]["trl_version"],
    }
    metrics = [
        _metric("adapter_imports", 2, "count", False),
        _metric("eval_pairs", 2, "count", False),
        _metric("grpo_nonzero_tensors", nonzero_counts["grpo"], "count", False),
        _metric("grpo_weight_bytes", grpo_model["size_bytes"], "bytes", False),
        _metric("http_errors", final_health["requests"]["error"], "count", True),
        _metric("sft_nonzero_tensors", nonzero_counts["sft"], "count", False),
        _metric("sft_weight_bytes", sft_model["size_bytes"], "bytes", False),
        _metric("training_tasks", 2, "count", False),
    ]
    return {"details": details, "metrics": metrics}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=sorted(SUPPORTED_BACKENDS))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--trainer-python", required=True, type=Path)
    parser.add_argument("--kiln", type=Path, default=Path("target/release/kiln"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result_text = os.environ.get(RESULT_ENV)
    if not result_text:
        print(f"error: {RESULT_ENV} is required", file=sys.stderr)
        return 2
    result_path = Path(result_text)
    if result_path.exists():
        print(f"error: refusing to replace {result_path}", file=sys.stderr)
        return 2
    workspace = result_path.parent / "hf-trl-roundtrip"
    try:
        workspace.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        print(f"error: cannot create qualification workspace: {exc}", file=sys.stderr)
        return 2

    started = time.monotonic()
    try:
        evidence = _roundtrip(args, workspace)
        result = _result_document(
            duration=time.monotonic() - started,
            metrics=evidence["metrics"],
            details=evidence["details"],
        )
        _write_json_new(result_path, result)
    except BaseException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"[HF/TRL ROUNDTRIP PASS] backend={args.backend} "
        "tasks=sft,grpo imports=2 eval_pairs=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
