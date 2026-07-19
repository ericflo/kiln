from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "model_fingerprint.py"
SPEC = importlib.util.spec_from_file_location("model_fingerprint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
model_fingerprint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = model_fingerprint
SPEC.loader.exec_module(model_fingerprint)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


class ModelFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "Qwen3.5-4B"
        self.root.mkdir()
        self._write("config.json", b'{"model_type":"qwen3_5"}\n')
        self._write("tokenizer.json", b'{"version":"1.0"}\n')
        self._write("chat_template.jinja", b"{{ messages }}\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, relative: str, content: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _index(self, weight_map: dict[str, object]) -> None:
        self._write(
            model_fingerprint.INDEX_FILENAME,
            (json.dumps({"weight_map": weight_map}) + "\n").encode(),
        )

    def test_single_file_fingerprint_matches_receipt_model_shape(self) -> None:
        weight = b"single checkpoint bytes"
        self._write("model.safetensors", weight)

        result = model_fingerprint.fingerprint_model(self.root)

        self.assertEqual(result["id"], "Qwen3.5-4B")
        self.assertEqual(result["path"], str(self.root))
        self.assertEqual(
            result["weight_files"],
            [{"path": "model.safetensors", "sha256": _sha256(weight), "bytes": len(weight)}],
        )
        self.assertEqual(result["config_hash"], _sha256((self.root / "config.json").read_bytes()))
        self.assertEqual(
            result["tokenizer_hash"], _sha256((self.root / "tokenizer.json").read_bytes())
        )
        self.assertEqual(
            result["chat_template_hash"],
            _sha256((self.root / "chat_template.jinja").read_bytes()),
        )
        self.assertEqual(
            set(result),
            {
                "id",
                "path",
                "weight_files",
                "config_hash",
                "tokenizer_hash",
                "chat_template_hash",
            },
        )

    def test_index_deduplicates_references_and_sorts_output_paths(self) -> None:
        self._write("z-shard.safetensors", b"z")
        self._write("nested/a-shard.safetensors", b"a")
        self._write("unreferenced.safetensors", b"ignored")
        self._index(
            {
                "tensor.z": "z-shard.safetensors",
                "tensor.a": "nested/a-shard.safetensors",
                "tensor.a_again": "nested/a-shard.safetensors",
            }
        )

        result = model_fingerprint.fingerprint_model(self.root)

        self.assertEqual(
            [item["path"] for item in result["weight_files"]],
            ["nested/a-shard.safetensors", "z-shard.safetensors"],
        )

    def test_loader_fallback_uses_model_safetensors_then_sorted_glob(self) -> None:
        self._write("z.safetensors", b"z")
        self._write("a.safetensors", b"a")
        self._write("model.safetensors", b"single")
        single = model_fingerprint.fingerprint_model(self.root)
        self.assertEqual([item["path"] for item in single["weight_files"]], ["model.safetensors"])

        (self.root / "model.safetensors").unlink()
        fallback = model_fingerprint.fingerprint_model(self.root)
        self.assertEqual(
            [item["path"] for item in fallback["weight_files"]],
            ["a.safetensors", "z.safetensors"],
        )

    def test_hidden_suffix_without_stem_is_not_a_loader_fallback(self) -> None:
        self._write(".safetensors", b"not selected by Path::extension")
        with self.assertRaisesRegex(
            model_fingerprint.ModelFingerprintError,
            r"no \.safetensors files found",
        ):
            model_fingerprint.fingerprint_model(self.root)

    def test_missing_model_input_is_rejected(self) -> None:
        self._write("model.safetensors", b"weight")
        (self.root / "config.json").unlink()
        with self.assertRaisesRegex(
            model_fingerprint.ModelFingerprintError,
            r"config\.json.*cannot be inspected",
        ):
            model_fingerprint.fingerprint_model(self.root)

    def test_missing_index_reference_is_rejected_without_fallback(self) -> None:
        self._write("model.safetensors", b"fallback must not be used")
        self._index({"tensor": "missing.safetensors"})
        with self.assertRaisesRegex(
            model_fingerprint.ModelFingerprintError,
            r"missing\.safetensors.*referenced.*invalid",
        ):
            model_fingerprint.fingerprint_model(self.root)

    def test_index_traversal_is_rejected(self) -> None:
        outside = self.root.parent / "outside.safetensors"
        outside.write_bytes(b"outside")
        self._index({"tensor": "../outside.safetensors"})
        with self.assertRaisesRegex(
            model_fingerprint.ModelFingerprintError,
            r"must stay inside the model directory",
        ):
            model_fingerprint.fingerprint_model(self.root)

    def test_symlinked_shard_and_metadata_are_rejected(self) -> None:
        target = self.root.parent / "target.safetensors"
        target.write_bytes(b"target")
        (self.root / "linked.safetensors").symlink_to(target)
        self._index({"tensor": "linked.safetensors"})
        with self.assertRaisesRegex(
            model_fingerprint.ModelFingerprintError,
            r"must not use a symlink",
        ):
            model_fingerprint.fingerprint_model(self.root)

        (self.root / model_fingerprint.INDEX_FILENAME).unlink()
        (self.root / "linked.safetensors").unlink()
        self._write("model.safetensors", b"weight")
        tokenizer_target = self.root.parent / "tokenizer.json"
        tokenizer_target.write_bytes(b"{}")
        (self.root / "tokenizer.json").unlink()
        (self.root / "tokenizer.json").symlink_to(tokenizer_target)
        with self.assertRaisesRegex(
            model_fingerprint.ModelFingerprintError,
            r"must not use a symlink",
        ):
            model_fingerprint.fingerprint_model(self.root)

    def test_non_file_and_duplicate_physical_shards_are_rejected(self) -> None:
        (self.root / "directory.safetensors").mkdir()
        self._index({"tensor": "directory.safetensors"})
        with self.assertRaisesRegex(
            model_fingerprint.ModelFingerprintError,
            r"not a regular file",
        ):
            model_fingerprint.fingerprint_model(self.root)

        (self.root / model_fingerprint.INDEX_FILENAME).unlink()
        (self.root / "directory.safetensors").rmdir()
        first = self._write("first.safetensors", b"weight")
        os.link(first, self.root / "second.safetensors")
        self._index({"one": "first.safetensors", "two": "second.safetensors"})
        with self.assertRaisesRegex(
            model_fingerprint.ModelFingerprintError,
            r"reference the same file",
        ):
            model_fingerprint.fingerprint_model(self.root)

    def test_each_content_input_changes_its_recorded_hash(self) -> None:
        self._write("model.safetensors", b"weight-v1")
        original = model_fingerprint.fingerprint_model(self.root)
        cases = (
            ("model.safetensors", "weight_files", b"weight-v2"),
            ("config.json", "config_hash", b'{"changed":true}\n'),
            ("tokenizer.json", "tokenizer_hash", b'{"changed":true}\n'),
            ("chat_template.jinja", "chat_template_hash", b"changed template\n"),
        )
        for relative, field, content in cases:
            with self.subTest(relative=relative):
                path = self.root / relative
                before = path.read_bytes()
                path.write_bytes(content)
                changed = model_fingerprint.fingerprint_model(self.root)
                if field == "weight_files":
                    self.assertNotEqual(
                        original["weight_files"][0]["sha256"],
                        changed["weight_files"][0]["sha256"],
                    )
                else:
                    self.assertNotEqual(original[field], changed[field])
                path.write_bytes(before)

    def test_change_during_hashing_is_rejected(self) -> None:
        weight = self._write("model.safetensors", b"weight-v1")
        original_hash = model_fingerprint._OpenInput.hash

        def hash_then_change(item: object) -> str:
            digest = original_hash(item)
            if item.relative_path == "model.safetensors":
                weight.write_bytes(b"weight-v2")
            return digest

        # Force metadata equality so this test proves the second content read,
        # independent of filesystem timestamp granularity.
        unchanged_metadata = (1, 2, 3, 4, 5, 6)
        with mock.patch.object(model_fingerprint._OpenInput, "hash", hash_then_change), mock.patch.object(
            model_fingerprint, "_stat_identity", return_value=unchanged_metadata
        ):
            with self.assertRaisesRegex(
                model_fingerprint.ModelFingerprintError,
                r"changed while it was being fingerprinted",
            ):
                model_fingerprint.fingerprint_model(self.root)

    def test_missing_template_is_null_and_cli_json_honors_model_id(self) -> None:
        self._write("model.safetensors", b"weight")
        (self.root / "chat_template.jinja").unlink()
        before = set(self.root.iterdir())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            return_code = model_fingerprint.main(
                ["--model-path", str(self.root), "--model-id", "pinned-qwen", "--json"]
            )
        value = json.loads(stdout.getvalue())
        self.assertEqual(return_code, 0)
        self.assertEqual(value["id"], "pinned-qwen")
        self.assertIsNone(value["chat_template_hash"])
        self.assertEqual(set(self.root.iterdir()), before)

    def test_cli_start_gate_is_absolute_exact_and_checked_before_hashing(self) -> None:
        self._write("model.safetensors", b"weight")
        with self.assertRaisesRegex(model_fingerprint.ModelFingerprintError, "must be absolute"):
            model_fingerprint._wait_for_start_gate(Path("relative"), timeout_seconds=0.01)
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "gate"
            gate.write_bytes(b"go\n")
            model_fingerprint._wait_for_start_gate(gate, timeout_seconds=0.01)
            gate.write_bytes(b"wrong")
            with self.assertRaisesRegex(model_fingerprint.ModelFingerprintError, "payload"):
                model_fingerprint._wait_for_start_gate(gate, timeout_seconds=0.01)

        gate = Path(self.tmp.name) / "released"
        gate.write_bytes(b"go\n")
        stdout = io.StringIO()
        with mock.patch.object(
            model_fingerprint, "fingerprint_model", wraps=model_fingerprint.fingerprint_model
        ) as fingerprint, contextlib.redirect_stdout(stdout):
            return_code = model_fingerprint.main(
                [
                    "--model-path",
                    str(self.root),
                    "--json",
                    "--start-gate",
                    str(gate),
                ]
            )
        self.assertEqual(return_code, 0)
        fingerprint.assert_called_once_with(self.root, None)

    def test_tokenizer_config_template_fallback_hashes_decoded_string_bytes(self) -> None:
        self._write("model.safetensors", b"weight")
        (self.root / "chat_template.jinja").unlink()
        template = "{% if messages %}snowman: \u2603{% endif %}"
        self._write(
            "tokenizer_config.json",
            json.dumps({"chat_template": template}, ensure_ascii=True).encode("utf-8"),
        )

        result = model_fingerprint.fingerprint_model(self.root)

        self.assertEqual(result["chat_template_hash"], _sha256(template.encode("utf-8")))
        self.assertNotEqual(
            result["chat_template_hash"],
            _sha256((self.root / "tokenizer_config.json").read_bytes()),
        )

    def test_standalone_template_takes_precedence_over_invalid_fallback(self) -> None:
        self._write("model.safetensors", b"weight")
        standalone = (self.root / "chat_template.jinja").read_bytes()
        self._write("tokenizer_config.json", b"not valid JSON")

        result = model_fingerprint.fingerprint_model(self.root)

        self.assertEqual(result["chat_template_hash"], _sha256(standalone))

    def test_invalid_tokenizer_config_fallback_is_rejected(self) -> None:
        self._write("model.safetensors", b"weight")
        (self.root / "chat_template.jinja").unlink()
        invalid_values = (
            b"not valid JSON",
            b'{"chat_template":"first","chat_template":"second"}',
            b'{"chat_template":42}',
        )
        for value in invalid_values:
            with self.subTest(value=value):
                self._write("tokenizer_config.json", value)
                with self.assertRaises(model_fingerprint.ModelFingerprintError):
                    model_fingerprint.fingerprint_model(self.root)


if __name__ == "__main__":
    unittest.main()
