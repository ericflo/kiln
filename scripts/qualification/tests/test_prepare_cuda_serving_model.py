from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_prepare_cuda_serving_model",
    QUALIFICATION_DIR / "prepare_cuda_serving_model.py",
)
assert SPEC is not None and SPEC.loader is not None
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def write_model(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"fixture"}', encoding="ascii")
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="ascii")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="ascii")
    (root / "model.safetensors").write_bytes(b"fixture-weights")
    (root / "README.md").write_text("fixture", encoding="ascii")
    adapters = root / "adapters"
    adapters.mkdir()
    (adapters / "ignored-link").symlink_to("/tmp/nonexistent-adapter")
    (root / ".cache").mkdir()


class PrepareCudaServingModelTests(unittest.TestCase):
    def test_materializes_and_revalidates_a_closed_hardlink_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            parent = root / "private"
            target = parent / "serving-model"
            write_model(source)
            parent.mkdir(mode=0o700)

            created = materializer.materialize(source, target, "fixture-model")
            reused = materializer.materialize(source, target, "fixture-model")

            self.assertTrue(created["created"])
            self.assertFalse(reused["created"])
            self.assertEqual(created["content_sha256"], reused["content_sha256"])
            self.assertEqual(created["excluded_directories"], [".cache", "adapters"])
            self.assertEqual(
                sorted(path.name for path in target.iterdir()),
                [
                    "README.md",
                    "chat_template.jinja",
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                ],
            )
            self.assertEqual(target.stat().st_mode & 0o222, 0)
            self.assertTrue(
                all(
                    path.stat().st_ino == (source / path.name).stat().st_ino
                    for path in target.iterdir()
                )
            )

    def test_rejects_unknown_directories_root_symlinks_and_target_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            parent = root / "private"
            target = parent / "serving-model"
            write_model(source)
            parent.mkdir(mode=0o700)

            unknown = source / "remote-code"
            unknown.mkdir()
            with self.assertRaisesRegex(
                materializer.MaterializationError, "undeclared directory"
            ):
                materializer.materialize(source, target, "fixture-model")
            unknown.rmdir()

            (source / "root-link").symlink_to("config.json")
            with self.assertRaisesRegex(
                materializer.MaterializationError, "root contains symlink"
            ):
                materializer.materialize(source, target, "fixture-model")
            (source / "root-link").unlink()

            materializer.materialize(source, target, "fixture-model")
            target.chmod(0o755)
            with self.assertRaisesRegex(
                materializer.MaterializationError, "must be read-only"
            ):
                materializer.materialize(source, target, "fixture-model")

    def test_cli_emits_one_strict_json_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            parent = root / "private"
            target = parent / "serving-model"
            write_model(source)
            parent.mkdir(mode=0o700)
            output = root / "output"
            original_stdout = sys.stdout
            try:
                with output.open("w", encoding="ascii") as stream:
                    sys.stdout = stream
                    self.assertEqual(
                        materializer.main(
                            [
                                "--source",
                                str(source),
                                "--target",
                                str(target),
                                "--model-id",
                                "fixture-model",
                            ]
                        ),
                        0,
                    )
            finally:
                sys.stdout = original_stdout
            value = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(value["schema"], materializer.SCHEMA)
            self.assertEqual(value["file_count"], 5)


if __name__ == "__main__":
    unittest.main()
