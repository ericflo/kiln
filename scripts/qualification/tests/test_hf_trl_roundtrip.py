from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "hf_trl_roundtrip.py"
SPEC = importlib.util.spec_from_file_location("hf_trl_roundtrip", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
roundtrip = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = roundtrip
SPEC.loader.exec_module(roundtrip)


def _health(backend: str = "rocm") -> dict:
    return {
        "status": "ok",
        "backend_runtime": {
            "healthy": True,
            "quarantined": False,
            "reason": None,
            "restart_required": False,
        },
        "execution_identity": {
            "backend": backend,
            "provenance_sha256": "sha256:" + "1" * 64,
        },
        "checks": [{"name": "model_loaded", "pass": True}],
        "requests": {"error": 0},
        "scheduler": {"waiting": 0, "running": 0, "blocks_used": 0},
    }


def _tar(path: Path, entries: list[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload, kind in entries:
            item = tarfile.TarInfo(name)
            if kind == "file":
                item.size = len(payload)
                archive.addfile(item, io.BytesIO(payload))
            elif kind == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = payload.decode()
                archive.addfile(item)
            else:
                raise AssertionError(kind)


class HfTrlRoundTripTests(unittest.TestCase):
    def test_invocation_path_preserves_venv_style_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            link = Path(raw) / "python"
            link.symlink_to(Path(sys.executable))
            invoked = roundtrip._invocation_path(link)
            self.assertEqual(invoked, Path(os.path.abspath(link)))
            self.assertTrue(invoked.is_symlink())
            self.assertNotEqual(invoked, invoked.resolve())

    def test_generated_fixtures_are_valid_and_scorers_are_executable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = roundtrip._prepare_fixtures(Path(raw))
            sft = json.loads(paths["sft"].read_text())
            self.assertEqual(sft["messages"][-1]["content"], "SFT_ROUNDTRIP_OK")
            self.assertEqual(json.loads(paths["grpo_tasks"].read_text())["id"], "grpo-roundtrip-1")
            self.assertEqual(json.loads(paths["eval_request"].read_text())["max_tokens"], 16)
            self.assertTrue(os.access(paths["grpo_scorer"], os.X_OK))
            self.assertTrue(os.access(paths["eval_scorer"], os.X_OK))

    def test_health_requires_exact_backend_and_drained_runtime(self) -> None:
        identity = roundtrip._validate_health(_health(), "rocm", final=True)
        self.assertEqual(identity["backend"], "rocm")

        cases = []
        wrong_backend = _health()
        wrong_backend["execution_identity"]["backend"] = "vulkan"
        cases.append(wrong_backend)
        quarantined = _health()
        quarantined["backend_runtime"]["quarantined"] = True
        cases.append(quarantined)
        request_error = _health()
        request_error["requests"]["error"] = 1
        cases.append(request_error)
        leaked_blocks = _health()
        leaked_blocks["scheduler"]["blocks_used"] = 1
        cases.append(leaked_blocks)
        failed_check = _health()
        failed_check["checks"][0]["pass"] = False
        cases.append(failed_check)

        for health in cases:
            with self.subTest(health=health), self.assertRaises(roundtrip.RoundTripError):
                roundtrip._validate_health(health, "rocm", final=True)

    def test_safe_extract_accepts_only_one_regular_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "ok.tar.gz"
            _tar(
                archive,
                [
                    ("fixture.kiln-hf/train.py", b"print('ok')\n", "file"),
                    ("fixture.kiln-hf/kiln_hf_export.json", b"{}\n", "file"),
                ],
            )
            extracted = roundtrip._safe_extract_bundle(
                archive, root / "out", "fixture.kiln-hf"
            )
            self.assertEqual((extracted / "train.py").read_bytes(), b"print('ok')\n")

    def test_safe_extract_rejects_links_traversal_and_wrong_root(self) -> None:
        cases = [
            [("fixture.kiln-hf/link", b"train.py", "symlink")],
            [("fixture.kiln-hf/../escape", b"bad", "file")],
            [("other.kiln-hf/train.py", b"bad", "file")],
        ]
        for index, entries in enumerate(cases):
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                archive = root / "bad.tar.gz"
                _tar(archive, entries)
                with self.assertRaises(roundtrip.RoundTripError):
                    roundtrip._safe_extract_bundle(
                        archive, root / f"out-{index}", "fixture.kiln-hf"
                    )

    def test_training_result_binds_nonempty_adapter_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            bundle = Path(raw)
            weights = b"not-a-real-safetensors-fixture"
            (bundle / "adapter_model.safetensors").write_bytes(weights)
            digest = roundtrip._sha256_file(bundle / "adapter_model.safetensors")
            result = {
                "result_type": "kiln.hf-trl-result.v1",
                "task": "sft",
                "export_sha256": "sha256:" + "2" * 64,
                "result_sha256": "sha256:" + "3" * 64,
                "trainer": {
                    "kind": "trl_sft_trainer",
                    "torch_version": "2",
                    "transformers_version": "5",
                    "trl_version": "1",
                    "peft_version": "0",
                },
                "output_adapter": {
                    "model": {"sha256": digest, "size_bytes": len(weights)}
                },
                "effective_config": {
                    "dataset_rows": {"kind": "unsigned", "value": 1},
                    "lora_rank": {"kind": "unsigned", "value": 1},
                    "target_modules": {"kind": "text", "value": "q_proj"},
                },
            }
            (bundle / "kiln_hf_result.json").write_text(json.dumps(result))
            self.assertEqual(
                roundtrip._validate_training_result(bundle, "sft")["result_sha256"],
                result["result_sha256"],
            )
            result["output_adapter"]["model"]["sha256"] = "sha256:" + "4" * 64
            (bundle / "kiln_hf_result.json").write_text(json.dumps(result))
            with self.assertRaises(roundtrip.RoundTripError):
                roundtrip._validate_training_result(bundle, "sft")

    def test_import_receipt_binds_task_result_and_resident_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            adapter_dir = Path(raw)
            installed = adapter_dir / "roundtrip-grpo"
            installed.mkdir()
            result = {
                "export_sha256": "sha256:" + "1" * 64,
                "result_sha256": "sha256:" + "2" * 64,
            }
            receipt = {
                "adapter_name": "roundtrip-grpo",
                "export_sha256": result["export_sha256"],
                "import_sha256": "sha256:" + "3" * 64,
                "import_type": "kiln.hf-trl-import.v1",
                "resident_model": {
                    "base_weight_shard_manifest": {
                        "aggregate_sha256": "sha256:" + "4" * 64
                    }
                },
                "result_sha256": result["result_sha256"],
                "task": "grpo",
                "used_exported_reference_script": True,
            }
            (installed / "kiln_hf_import.json").write_text(json.dumps(receipt))
            observed = roundtrip._validate_import_receipt(
                adapter_dir, "roundtrip-grpo", "grpo", result
            )
            self.assertEqual(observed["import_sha256"], receipt["import_sha256"])
            receipt["used_exported_reference_script"] = False
            (installed / "kiln_hf_import.json").write_text(json.dumps(receipt))
            with self.assertRaises(roundtrip.RoundTripError):
                roundtrip._validate_import_receipt(
                    adapter_dir, "roundtrip-grpo", "grpo", result
                )

    def test_eval_requires_nonempty_paired_content_hash_and_no_warnings(self) -> None:
        summary = {
            "adapter": "roundtrip-sft",
            "pair_count": 1,
            "warnings": [],
            "results": [
                {
                    "base_score": 1.0,
                    "adapter_score": 1.0,
                    "base_content": "ok",
                    "adapter_content": "ok",
                }
            ],
            "adapter_hashes": [
                {
                    "name": "roundtrip-sft",
                    "adapter_model_sha256": "abc",
                }
            ],
        }
        roundtrip._validate_eval(summary, "roundtrip-sft")
        summary["results"][0]["adapter_content"] = ""
        with self.assertRaises(roundtrip.RoundTripError):
            roundtrip._validate_eval(summary, "roundtrip-sft")

    def test_case_result_metrics_are_sorted_and_config_is_portable(self) -> None:
        result = roundtrip._result_document(
            duration=1.0,
            metrics=[
                roundtrip._metric("z_metric", 1, "count", False),
                roundtrip._metric("a_metric", 0, "count", True),
            ],
            details={"hash": "sha256:" + "a" * 64},
        )
        self.assertEqual([item["name"] for item in result["metrics"]], ["a_metric", "z_metric"])
        self.assertEqual(result["effective_config"]["network"], "loopback_only")
        self.assertLessEqual(len(result["details"]), 2048)


if __name__ == "__main__":
    unittest.main()
