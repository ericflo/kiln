from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[2] / "vllm_teacher.py"
SPEC = importlib.util.spec_from_file_location("vllm_teacher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
vllm_teacher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vllm_teacher
SPEC.loader.exec_module(vllm_teacher)


A_HASH = "a" * 64
B_HASH = "b" * 64
C_HASH = "c" * 64
D_HASH = "d" * 64
E_HASH = "e" * 64
RUNTIME_VERSIONS = {
    "python": "3.12.7",
    "python_implementation": "CPython",
    "vllm": "0.25.0",
    "torch": "2.9.1+rocm7.1",
    "transformers": "5.0.0",
    "tokenizers": "0.22.2",
}
ACCELERATOR = {
    "type": "rocm",
    "driver": "amdgpu:6.14.0;hip-runtime:7.1",
    "devices": [
        {
            "index": 0,
            "name": "AMD Radeon 8060S",
            "architecture": "gfx1151",
            "total_memory_bytes": 128 * 1024**3,
        }
    ],
}


def _identity(adapter: dict[str, object] | None = None) -> dict[str, object]:
    return vllm_teacher.build_identity(
        served_model_id="teacher-qwen",
        base_model_sha256=A_HASH,
        tokenizer_vocab_sha256=B_HASH,
        tokenizer_config_sha256=C_HASH,
        adapter=adapter,
        vocab_size=8,
        max_top_k=5,
        max_model_len=4096,
        max_prompt_logprob_candidates=20_000,
        implementation="vllm:0.25.1+cu129",
        inference_config_sha256=D_HASH,
    )


def _manifest(adapter: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema": vllm_teacher.INPUT_MANIFEST_SCHEMA,
        "base_model_sha256": A_HASH,
        "snapshot_content_sha256": A_HASH,
        "model_config_sha256": D_HASH,
        "tokenizer_vocab_sha256": B_HASH,
        "tokenizer_config_sha256": C_HASH,
        "adapter": adapter,
        "adapter_max_rank": 32 if adapter is not None else None,
        "vocab_size": 8,
        "implementation": "vllm:0.25.0",
        "runtime_versions": dict(RUNTIME_VERSIONS),
        "runtime_content_sha256": E_HASH,
        "accelerator": {
            **ACCELERATOR,
            "devices": [dict(device) for device in ACCELERATOR["devices"]],
        },
    }


class TokenizerVocabFingerprintTests(unittest.TestCase):
    def test_exact_binary_contract_and_order_independence(self) -> None:
        vocab = {"z": 2, "<s>": 0, "é": 1}
        digest = hashlib.sha256()
        digest.update(b"kiln.tokenizer-vocab.v1\0")
        entries = [(0, b"<s>"), (1, "é".encode()), (2, b"z")]
        digest.update(struct.pack("<Q", len(entries)))
        for token_id, token in entries:
            digest.update(struct.pack("<I", token_id))
            digest.update(struct.pack("<Q", len(token)))
            digest.update(token)

        actual, size = vllm_teacher.tokenizer_vocab_fingerprint(vocab)
        reversed_actual, _ = vllm_teacher.tokenizer_vocab_fingerprint(
            dict(reversed(tuple(vocab.items())))
        )

        self.assertEqual(size, 3)
        self.assertEqual(actual, digest.hexdigest())
        self.assertEqual(reversed_actual, actual)

    def test_matches_rust_cross_runtime_golden_vector(self) -> None:
        actual, size = vllm_teacher.tokenizer_vocab_fingerprint({"a": 0, "b": 1})
        self.assertEqual(size, 2)
        self.assertEqual(
            actual,
            "0e7ed7c4b2c375344d31b534f7cbb119b528114f1301f09d6642801b100710ff",
        )

    def test_duplicate_and_gapped_ids_follow_the_rust_pair_contract(self) -> None:
        first, size = vllm_teacher.tokenizer_vocab_fingerprint({"b": 2, "a": 0, "alias": 0})
        second, _ = vllm_teacher.tokenizer_vocab_fingerprint({"alias": 0, "a": 0, "b": 2})
        self.assertEqual(size, 3)
        self.assertEqual(first, second)

    def test_boolean_out_of_range_and_invalid_utf8_ids_are_rejected(self) -> None:
        cases = (
            ({"a": False}, "must be an integer"),
            ({"a": -1}, "outside the u32 range"),
            ({"\ud800": 0}, "not valid UTF-8"),
        )
        for vocab, message in cases:
            with self.subTest(vocab=vocab):
                with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, message):
                    vllm_teacher.tokenizer_vocab_fingerprint(vocab)

    def test_transformers_contract_uses_fast_backend_canonical_json_and_size(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class Backend:
            def to_str(self) -> str:
                return '{"version":"1.0","model":{"type":"BPE"}}'

            def get_vocab_size(self, *, with_added_tokens: bool) -> int:
                self.with_added_tokens = with_added_tokens
                return 3

        backend = Backend()

        class Tokenizer:
            backend_tokenizer = backend

            @staticmethod
            def get_vocab() -> dict[str, int]:
                return {"a": 0, "b": 1, "<special>": 2}

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> Tokenizer:
                calls.append((path, kwargs))
                return Tokenizer()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = AutoTokenizer
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            vocab, backend_json, vocab_size = vllm_teacher._load_tokenizer_contract(
                Path("/model")
            )
        self.assertEqual(vocab_size, 3)
        self.assertEqual(backend_json, backend.to_str())
        self.assertEqual(vocab["<special>"], 2)
        self.assertEqual(calls[0][1]["local_files_only"], True)
        self.assertEqual(calls[0][1]["trust_remote_code"], False)
        self.assertEqual(calls[0][1]["use_fast"], True)
        self.assertEqual(backend.with_added_tokens, True)


class IdentityTests(unittest.TestCase):
    def test_canonical_field_order_compact_json_and_round_trip(self) -> None:
        identity = _identity()
        payload = vllm_teacher.canonical_identity_json(identity)
        fingerprint = vllm_teacher.encode_system_fingerprint(identity)

        self.assertEqual(tuple(identity), vllm_teacher.IDENTITY_FIELDS)
        self.assertNotIn(b" ", payload)
        self.assertNotIn(b"sha256:", payload)
        self.assertEqual(vllm_teacher.decode_system_fingerprint(fingerprint), identity)
        prefix, encoded, digest = fingerprint.split(".")
        self.assertEqual(prefix, "kiln-teacher-v1")
        self.assertNotIn("=", encoded)
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_adapter_order_and_name_are_canonical(self) -> None:
        adapter = {
            "name": "teacher-qwen",
            "weights_sha256": A_HASH,
            "config_sha256": B_HASH,
        }
        identity = _identity(adapter)
        self.assertEqual(tuple(identity["adapter"]), vllm_teacher.ADAPTER_FIELDS)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "must equal"):
            _identity({**adapter, "name": "some-other-model"})

    def test_tampered_noncanonical_and_reordered_fingerprints_are_rejected(self) -> None:
        identity = _identity()
        fingerprint = vllm_teacher.encode_system_fingerprint(identity)
        prefix, encoded, digest = fingerprint.split(".")
        cases = (
            f"wrong.{encoded}.{digest}",
            f"{prefix}.{encoded}.{A_HASH}",
            f"{prefix}.{encoded}=.{digest}",
        )
        for value in cases:
            with self.subTest(value=value[:30]):
                with self.assertRaises(vllm_teacher.TeacherLaunchError):
                    vllm_teacher.decode_system_fingerprint(value)

        reordered = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
        reordered_encoded = base64.urlsafe_b64encode(reordered).decode().rstrip("=")
        reordered_fingerprint = (
            f"{prefix}.{reordered_encoded}.{hashlib.sha256(reordered).hexdigest()}"
        )
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "canonical order"):
            vllm_teacher.decode_system_fingerprint(reordered_fingerprint)

    def test_identity_bounds_hashes_ids_and_version_fail_closed(self) -> None:
        base = {
            "served_model_id": "teacher-qwen",
            "base_model_sha256": A_HASH,
            "tokenizer_vocab_sha256": B_HASH,
            "tokenizer_config_sha256": C_HASH,
            "adapter": None,
            "vocab_size": 8,
            "max_top_k": 5,
            "max_model_len": 4096,
            "max_prompt_logprob_candidates": 20_000,
            "implementation": "vllm:0.25.0",
            "inference_config_sha256": D_HASH,
        }
        cases = (
            ({"served_model_id": "../bad"}, "served_model_id"),
            ({"base_model_sha256": "sha256:" + A_HASH}, "64 lowercase"),
            ({"max_top_k": 9}, "max_top_k"),
            ({"max_model_len": 0}, "max_model_len"),
            ({"max_prompt_logprob_candidates": 5}, "one maximum-K row"),
            ({"max_prompt_logprob_candidates": 24_577}, "one maximum-K row"),
            ({"max_prompt_logprob_candidates": 1_000_001}, "one maximum-K row"),
            ({"implementation": "vllm:0.20.0"}, "0.20.1rc0 or newer"),
        )
        for update, message in cases:
            with self.subTest(update=update):
                with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, message):
                    vllm_teacher.build_identity(**{**base, **update})

        accepted = vllm_teacher.build_identity(
            **{**base, "implementation": "vllm:0.20.1rc0"}
        )
        self.assertEqual(accepted["implementation"], "vllm:0.20.1rc0")


class FilesystemFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "model"
        self.root.mkdir()
        self._write(self.root, "model.safetensors", b"weight-v1")
        self._write(self.root, "config.json", b'{"model_type":"qwen3_5"}\n')
        self._write(self.root, "tokenizer.json", b'{"version":"1.0"}\n')
        self._write(self.root, "tokenizer_config.json", b'{"chat_template":"x"}\n')

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _write(root: Path, relative: str, payload: bytes) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_base_hash_matches_rust_byte_contract(self) -> None:
        content = (self.root / "model.safetensors").read_bytes()
        expected_payload = (
            b"kiln.base-model-content.v1\0"
            + struct.pack("<Q", 1)
            + struct.pack("<Q", len(content))
            + hashlib.sha256(content).digest()
        )
        self.assertEqual(
            vllm_teacher.fingerprint_base_model(self.root),
            hashlib.sha256(expected_payload).hexdigest(),
        )
        self.assertEqual(
            vllm_teacher.fingerprint_base_model(self.root),
            "aa7e9961df654af2a7405c5d2c07eb5a72b58f86060035c245c3415793c459c1",
        )
        _, config_hash = vllm_teacher.fingerprint_base_model_details(self.root)
        self.assertEqual(
            config_hash,
            hashlib.sha256((self.root / "config.json").read_bytes()).hexdigest(),
        )

    def test_base_hash_sorts_multi_shard_records_by_digest_then_length(self) -> None:
        (self.root / "model.safetensors").unlink()
        shards = {"z.safetensors": b"z-shard", "a.safetensors": b"a"}
        for name, content in shards.items():
            self._write(self.root, name, content)
        records = sorted(
            (hashlib.sha256(content).digest(), len(content)) for content in shards.values()
        )
        payload = bytearray(b"kiln.base-model-content.v1\0")
        payload.extend(struct.pack("<Q", len(records)))
        for content_hash, byte_count in records:
            payload.extend(struct.pack("<Q", byte_count))
            payload.extend(content_hash)
        self.assertEqual(
            vllm_teacher.fingerprint_base_model(self.root),
            hashlib.sha256(payload).hexdigest(),
        )

    def test_base_hash_tracks_selected_weight_content_only(self) -> None:
        original = vllm_teacher.fingerprint_base_model(self.root)
        (self.root / "tokenizer.json").write_bytes(b'{"changed":true}\n')
        self.assertEqual(vllm_teacher.fingerprint_base_model(self.root), original)

        (self.root / "model.safetensors").write_bytes(b"weight-v2")
        changed_weight = vllm_teacher.fingerprint_base_model(self.root)
        self.assertNotEqual(changed_weight, original)

        (self.root / "model.safetensors").write_bytes(b"weight-v1")
        (self.root / "config.json").write_bytes(b'{"model_type":"changed"}\n')
        self.assertEqual(vllm_teacher.fingerprint_base_model(self.root), original)

        (self.root / "config.json").write_bytes(b'{"model_type":"qwen3_5"}\n')
        (self.root / "model.safetensors.index.json").write_text(
            '{"weight_map":{"tensor":"model.safetensors"}}\n'
        )
        indexed = vllm_teacher.fingerprint_base_model(self.root)
        (self.root / "model.safetensors.index.json").write_text(
            '{"weight_map":{"renamed":"model.safetensors"}}\n'
        )
        self.assertEqual(vllm_teacher.fingerprint_base_model(self.root), indexed)

    def test_alternate_bin_is_bound_by_snapshot_while_base_load_stays_safetensors(self) -> None:
        base_hash = vllm_teacher.fingerprint_base_model(self.root)
        (self.root / "pytorch_model.bin").write_bytes(b"alternate-v1")
        first_snapshot_hash = vllm_teacher.snapshot_content_fingerprint(self.root, None)
        self.assertEqual(vllm_teacher.fingerprint_base_model(self.root), base_hash)

        (self.root / "pytorch_model.bin").write_bytes(b"alternate-v2")
        self.assertNotEqual(
            vllm_teacher.snapshot_content_fingerprint(self.root, None),
            first_snapshot_hash,
        )

    def test_shared_provenance_limiter_accounts_snapshot_and_model_passes(self) -> None:
        limiter = vllm_teacher._model_fingerprint._ReadRateLimiter(None)
        expected_snapshot_bytes = sum(
            path.stat().st_size for path in self.root.iterdir() if path.is_file()
        )
        vllm_teacher.snapshot_content_fingerprint(
            self.root,
            None,
            read_rate_limiter=limiter,
        )
        self.assertEqual(limiter.total_bytes, expected_snapshot_bytes)

        before_model = limiter.total_bytes
        vllm_teacher.fingerprint_base_model(self.root, limiter)
        self.assertGreater(limiter.total_bytes, before_model)

    def test_tokenizer_config_hashes_canonical_backend_json_and_root_symlinks_fail(self) -> None:
        backend_json = '{"version":"1.0","model":{"type":"BPE"}}'
        self.assertEqual(
            vllm_teacher.tokenizer_config_fingerprint(backend_json),
            hashlib.sha256(backend_json.encode()).hexdigest(),
        )
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "duplicate"):
            vllm_teacher.tokenizer_config_fingerprint('{"version":1,"version":2}')
        linked_root = Path(self.tmp.name) / "model-link"
        linked_root.symlink_to(self.root)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "must not be a symlink"):
            vllm_teacher.fingerprint_base_model(linked_root)

    def test_static_adapter_hashes_exact_supported_weights_and_config(self) -> None:
        adapter = Path(self.tmp.name) / "adapter"
        adapter.mkdir()
        self._write(adapter, "adapter_config.json", b'{"r":8}\n')
        self._write(adapter, "adapter_model.safetensors", b"adapter-v1")

        original = vllm_teacher.fingerprint_adapter(adapter, "teacher-qwen")
        self.assertEqual(original["config_sha256"], hashlib.sha256(b'{"r":8}\n').hexdigest())
        self.assertEqual(vllm_teacher.adapter_max_rank(adapter), 8)
        (adapter / "adapter_config.json").write_text(
            '{"r":8,"rank_pattern":{"model.layers.0.q_proj":20}}'
        )
        self.assertEqual(vllm_teacher.adapter_max_rank(adapter), 32)
        (adapter / "adapter_config.json").write_bytes(b'{"r":8}\n')
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v2")
        self.assertNotEqual(
            vllm_teacher.fingerprint_adapter(adapter, "teacher-qwen")["weights_sha256"],
            original["weights_sha256"],
        )
        (adapter / "adapter_model.safetensors").unlink()
        (adapter / "adapter_model.bin").write_bytes(b"adapter-v1")
        self.assertNotEqual(
            vllm_teacher.fingerprint_adapter(adapter, "teacher-qwen")["weights_sha256"],
            original["weights_sha256"],
        )

    def test_adapter_missing_ambiguous_invalid_and_symlink_inputs_fail(self) -> None:
        adapter = Path(self.tmp.name) / "adapter-errors"
        adapter.mkdir()
        self._write(adapter, "adapter_config.json", b"{}")
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "must contain"):
            vllm_teacher.fingerprint_adapter(adapter, "teacher-qwen")

        self._write(adapter, "adapter_model.safetensors", b"safe")
        self._write(adapter, "adapter_model.bin", b"bin")
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "ambiguous"):
            vllm_teacher.fingerprint_adapter(adapter, "teacher-qwen")

        (adapter / "adapter_model.bin").unlink()
        (adapter / "adapter_config.json").write_text('{"r":8,"r":16}')
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "duplicate"):
            vllm_teacher.fingerprint_adapter(adapter, "teacher-qwen")

        (adapter / "adapter_config.json").write_text("{}")
        target = Path(self.tmp.name) / "adapter-target.safetensors"
        target.write_bytes(b"target")
        (adapter / "adapter_model.safetensors").unlink()
        (adapter / "adapter_model.safetensors").symlink_to(target)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "symlink"):
            vllm_teacher.fingerprint_adapter(adapter, "teacher-qwen")

    def test_adapter_rank_contract_rejects_missing_invalid_and_oversized_values(self) -> None:
        adapter = Path(self.tmp.name) / "adapter-ranks"
        adapter.mkdir()
        for config, message in (
            ({}, "positive integers"),
            ({"r": True}, "positive integers"),
            ({"r": 8, "rank_pattern": []}, "must be an object"),
            ({"r": 513}, "supported maximum"),
        ):
            with self.subTest(config=config):
                (adapter / "adapter_config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, message):
                    vllm_teacher.adapter_max_rank(adapter)


class ImmutableSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.model = self.base / "model"
        self.model.mkdir()
        (self.model / "weights.safetensors").write_bytes(b"weights-v1")
        (self.model / "config.json").write_text('{"vocab_size":3}')
        (self.model / "nested").mkdir()
        (self.model / "nested" / "tokenizer.json").write_bytes(b"tokenizer-v1")
        (self.model / "empty").mkdir()
        self.snapshot_root = self.base / "snapshots"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _snapshot(self) -> object:
        with mock.patch.object(vllm_teacher, "_try_reflink", return_value=False):
            return vllm_teacher.create_immutable_snapshot(
                self.model, None, self.snapshot_root
            )

    def test_copy_fallback_publishes_complete_read_only_snapshot_and_cleans_up(self) -> None:
        expected_hash = vllm_teacher.snapshot_content_fingerprint(self.model, None)
        snapshot = self._snapshot()
        try:
            self.assertTrue(snapshot.path.name.startswith("ready-"))
            self.assertEqual(snapshot.manifest_sha256, expected_hash)
            self.assertIn("model/empty", snapshot.directories)
            self.assertEqual(
                (snapshot.model_path / "weights.safetensors").read_bytes(), b"weights-v1"
            )
            self.assertEqual(snapshot.path.stat().st_mode & 0o777, 0o500)
            self.assertEqual(
                (snapshot.model_path / "weights.safetensors").stat().st_mode & 0o777,
                0o400,
            )
            snapshot.verify()
            (self.model / "weights.safetensors").write_bytes(b"mutated-source")
            snapshot.verify()
            self.assertEqual(
                (snapshot.model_path / "weights.safetensors").read_bytes(), b"weights-v1"
            )
        finally:
            snapshot.cleanup()
        self.assertFalse(snapshot.path.exists())
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_staged_file_and_empty_directory_mutation_are_detected(self) -> None:
        snapshot = self._snapshot()
        try:
            weight = snapshot.model_path / "weights.safetensors"
            weight.chmod(0o600)
            weight.write_bytes(b"mutated")
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "writable|content"):
                snapshot.verify()
        finally:
            snapshot.cleanup()

        snapshot = self._snapshot()
        try:
            empty = snapshot.model_path / "empty"
            empty.chmod(0o700)
            (empty / "new-empty").mkdir()
            (empty / "new-empty").chmod(0o500)
            empty.chmod(0o500)
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "content"):
                snapshot.verify()
        finally:
            snapshot.cleanup()

    def test_symlinks_special_files_bounds_and_capacity_fail_without_partial_snapshot(self) -> None:
        target = self.base / "outside"
        target.write_bytes(b"outside")
        (self.model / "link").symlink_to(target)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "symlink"):
            self._snapshot()
        (self.model / "link").unlink()

        if hasattr(os, "mkfifo"):
            os.mkfifo(self.model / "fifo")
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "special file"):
                self._snapshot()
            (self.model / "fifo").unlink()

        with mock.patch.object(vllm_teacher, "MAX_SNAPSHOT_FILES", 1):
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "file safety limit"):
                self._snapshot()
        with mock.patch.object(vllm_teacher, "MAX_SNAPSHOT_PATH_METADATA_BYTES", 1):
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "path metadata"):
                self._snapshot()
        with mock.patch.object(
            vllm_teacher,
            "_snapshot_filesystem_capacity",
            return_value=(0, 1_000_000, 4096),
        ):
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "free space"):
                self._snapshot()
        self.assertFalse(self.snapshot_root.exists() and any(self.snapshot_root.iterdir()))

    def test_growth_after_inventory_is_rejected_before_copying_unchecked_bytes(self) -> None:
        original = vllm_teacher._inventory_source_tree

        def mutate_after_inventory(
            path: Path,
            logical_path: str,
            budget: object,
            *,
            depth: int = 0,
        ) -> tuple[int, int]:
            result = original(path, logical_path, budget, depth=depth)
            if logical_path == "model":
                (self.model / "weights.safetensors").write_bytes(b"x" * 4096)
            return result

        with mock.patch.object(
            vllm_teacher, "_inventory_source_tree", side_effect=mutate_after_inventory
        ), mock.patch.object(vllm_teacher, "_try_reflink", return_value=False):
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "grew"):
                vllm_teacher.create_immutable_snapshot(
                    self.model, None, self.snapshot_root
                )
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_cross_file_mutation_cannot_publish_a_torn_checkpoint(self) -> None:
        coherent = self.base / "coherent-model"
        coherent.mkdir()
        (coherent / "a.safetensors").write_bytes(b"old-a")
        (coherent / "b.safetensors").write_bytes(b"old-b")
        original = vllm_teacher._copy_regular_file

        def mutate_between_files(
            source: Path,
            destination: Path,
            relative_path: str,
            budget: object,
            depth: int,
            *,
            read_rate_limiter: object | None = None,
        ) -> object:
            result = original(
                source,
                destination,
                relative_path,
                budget,
                depth,
                read_rate_limiter=read_rate_limiter,
            )
            if relative_path == "model/a.safetensors":
                (coherent / "a.safetensors").write_bytes(b"new-a")
                (coherent / "b.safetensors").write_bytes(b"new-b")
            return result

        with mock.patch.object(
            vllm_teacher, "_copy_regular_file", side_effect=mutate_between_files
        ), mock.patch.object(vllm_teacher, "_try_reflink", return_value=False):
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "multi-file copy"):
                vllm_teacher.create_immutable_snapshot(
                    coherent, None, self.snapshot_root
                )
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_snapshot_root_symlink_has_no_creation_side_effect(self) -> None:
        victim = self.base / "victim"
        victim.mkdir()
        link = self.base / "link"
        link.symlink_to(victim, target_is_directory=True)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "symlink"):
            vllm_teacher.create_immutable_snapshot(
                self.model, None, link / "must-not-exist"
            )
        self.assertFalse((victim / "must-not-exist").exists())

    def test_world_writable_nonsticky_snapshot_parent_is_rejected(self) -> None:
        unsafe_parent = self.base / "unsafe-parent"
        unsafe_parent.mkdir()
        unsafe_parent.chmod(0o777)
        try:
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "untrusted rename"):
                vllm_teacher.create_immutable_snapshot(
                    self.model, None, unsafe_parent / "snapshots"
                )
            self.assertFalse((unsafe_parent / "snapshots").exists())
        finally:
            unsafe_parent.chmod(0o700)

    def test_snapshot_ancestry_owner_must_be_current_user_root_or_verified_overflow(
        self,
    ) -> None:
        if not hasattr(os, "getuid"):
            self.skipTest("POSIX ownership is required")
        self.assertTrue(
            vllm_teacher._trusted_snapshot_directory_owner(
                types.SimpleNamespace(st_uid=os.getuid())
            )
        )
        self.assertTrue(
            vllm_teacher._trusted_snapshot_directory_owner(
                types.SimpleNamespace(st_uid=0)
            )
        )
        self.assertFalse(
            vllm_teacher._trusted_snapshot_directory_owner(
                types.SimpleNamespace(st_uid=max(os.getuid(), 0) + 1000)
            )
        )
        overflow_uid = 65_534
        self.assertTrue(
            vllm_teacher._trusted_snapshot_directory_owner(
                types.SimpleNamespace(st_uid=overflow_uid),
                overflow_uid,
            )
        )
        self.assertFalse(
            vllm_teacher._trusted_snapshot_directory_owner(
                types.SimpleNamespace(st_uid=overflow_uid)
            )
        )

    def test_overflow_owner_requires_exact_single_id_root_remap(self) -> None:
        uid_map = self.base / "uid_map"
        overflow_uid = self.base / "overflowuid"
        cases = (
            ("         0       1000          1\n", "65534\n", 65_534),
            ("0 0 1\n", "65534\n", None),
            ("0 1000 2\n", "65534\n", None),
            ("0 1000 1\n1 1001 1\n", "65534\n", None),
            ("not a map\n", "65534\n", None),
            ("0 1000 1\n", "0\n", None),
            ("0 1000 1\n", "65534 extra\n", None),
            ("0" * (vllm_teacher.MAX_PROC_IDENTITY_BYTES + 1), "65534\n", None),
        )
        with mock.patch.object(vllm_teacher.os, "getuid", return_value=0), mock.patch.object(
            vllm_teacher.os,
            "geteuid",
            return_value=0,
        ):
            for uid_map_text, overflow_text, expected in cases:
                with self.subTest(uid_map=uid_map_text, overflow=overflow_text):
                    uid_map.write_text(uid_map_text, encoding="ascii")
                    overflow_uid.write_text(overflow_text, encoding="ascii")
                    self.assertEqual(
                        vllm_teacher._verified_remapped_namespace_overflow_uid(
                            uid_map_path=uid_map,
                            overflow_uid_path=overflow_uid,
                        ),
                        expected,
                    )
        with mock.patch.object(vllm_teacher.os, "getuid", return_value=1000):
            self.assertIsNone(
                vllm_teacher._verified_remapped_namespace_overflow_uid(
                    uid_map_path=uid_map,
                    overflow_uid_path=overflow_uid,
                )
            )

    def test_private_cache_accepts_real_root_mapped_namespace_ancestry(self) -> None:
        unshare = shutil.which("unshare")
        if sys.platform != "linux" or unshare is None:
            self.skipTest("Linux util-linux unshare is unavailable")
        probe = subprocess.run(
            [unshare, "--user", "--map-root-user", "true"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode != 0:
            self.skipTest("unprivileged root-mapped user namespaces are unavailable")
        cache_root = self.base / "mapped-runtime-caches"
        code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import vllm_teacher
root = Path(sys.argv[2])
cache = vllm_teacher.create_private_runtime_cache(root)
cache.verify()
cache.cleanup()
assert list(root.iterdir()) == []
"""
        completed = subprocess.run(
            [
                unshare,
                "--user",
                "--map-root-user",
                sys.executable,
                "-c",
                code,
                str(MODULE_PATH.parent),
                str(cache_root),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_existing_nonprivate_snapshot_root_is_rejected_without_chmod(self) -> None:
        existing = self.base / "existing-root"
        existing.mkdir(mode=0o755)
        sentinel = existing / "keep"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "mode 0700"):
            vllm_teacher.create_immutable_snapshot(self.model, None, existing)
        self.assertEqual(existing.stat().st_mode & 0o777, 0o755)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_dynamic_inode_filesystem_does_not_report_false_exhaustion(self) -> None:
        statvfs = types.SimpleNamespace(
            f_bavail=100,
            f_frsize=0,
            f_bsize=4096,
            f_files=0,
            f_favail=0,
        )
        with mock.patch.object(vllm_teacher.os, "statvfs", return_value=statvfs):
            capacity = vllm_teacher._snapshot_filesystem_capacity(self.base)
        self.assertEqual(capacity, (409_600, None, 4096))

    def test_root_fd_cleanup_cannot_follow_replaced_root_or_escape(self) -> None:
        snapshot = self._snapshot()
        original_root = self.base / "renamed-snapshot-root"
        self.snapshot_root.rename(original_root)
        victim = self.base / "victim"
        victim.mkdir()
        victim_entry = victim / snapshot.path.name
        victim_entry.mkdir()
        (victim_entry / "keep").write_text("keep")
        self.snapshot_root.symlink_to(victim, target_is_directory=True)

        snapshot.cleanup()
        self.assertFalse((original_root / snapshot.path.name).exists())
        self.assertEqual((victim_entry / "keep").read_text(), "keep")

    def test_snapshot_root_inside_model_is_rejected_before_creation(self) -> None:
        nested_root = self.model / "snapshots"
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "non-nested"):
            vllm_teacher.create_immutable_snapshot(self.model, None, nested_root)
        self.assertFalse(nested_root.exists())


class PrivateRuntimeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.cache_root = self.base / "runtime-caches"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_private_cache_is_empty_writable_unique_and_recursively_removed(self) -> None:
        first = vllm_teacher.create_private_runtime_cache(self.cache_root)
        second = vllm_teacher.create_private_runtime_cache(self.cache_root)
        try:
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(first.path.stat().st_mode & 0o777, 0o700)
            self.assertEqual(list(first.path.iterdir()), [])
            (first.path / "compiled").mkdir()
            (first.path / "compiled" / "artifact.so").write_bytes(b"compiled")
            (first.path / "artifact-link").symlink_to("compiled/artifact.so")
            first.verify()
        finally:
            first.cleanup()
            second.cleanup()
        self.assertEqual(list(self.cache_root.iterdir()), [])

    def test_replaced_cache_entry_is_not_removed(self) -> None:
        cache = vllm_teacher.create_private_runtime_cache(self.cache_root)
        original = self.cache_root / "original-cache"
        cache.path.rename(original)
        cache.path.mkdir(mode=0o700)
        sentinel = cache.path / "keep"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "replaced"):
            cache.cleanup()
        self.assertEqual(sentinel.read_text(), "keep")
        sentinel.unlink()
        cache.path.rmdir()
        original.rename(cache.path)
        cache.cleanup()

    def test_cache_root_symlink_and_nonprivate_existing_root_are_rejected(self) -> None:
        victim = self.base / "victim"
        victim.mkdir()
        link = self.base / "cache-link"
        link.symlink_to(victim, target_is_directory=True)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "symlink"):
            vllm_teacher.create_private_runtime_cache(link)

        existing = self.base / "existing"
        existing.mkdir(mode=0o755)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "mode 0700"):
            vllm_teacher.create_private_runtime_cache(existing)
        self.assertEqual(existing.stat().st_mode & 0o777, 0o755)


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "identity-input.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, value: object) -> None:
        self.path.write_text(json.dumps(value))

    def test_strict_manifest_loads_without_optional_dependencies(self) -> None:
        self._write(_manifest())
        with mock.patch.object(
            vllm_teacher, "_load_tokenizer_contract", side_effect=AssertionError("must not import")
        ), mock.patch.object(
            vllm_teacher, "_installed_vllm_version", side_effect=AssertionError("must not inspect")
        ), mock.patch.object(
            vllm_teacher, "probe_accelerator", side_effect=AssertionError("must not inspect")
        ):
            result = vllm_teacher.load_identity_input(self.path)
        self.assertEqual(result["base_model_sha256"], A_HASH)
        self.assertIsNone(result["adapter"])

    def test_manifest_rejects_duplicate_extra_prefixed_hash_and_symlink(self) -> None:
        self.path.write_text('{"schema":"a","schema":"b"}')
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "duplicate"):
            vllm_teacher.load_identity_input(self.path)

        value = _manifest()
        value["extra"] = True
        self._write(value)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "extra keys"):
            vllm_teacher.load_identity_input(self.path)

        value = _manifest()
        value["base_model_sha256"] = "sha256:" + A_HASH
        self._write(value)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "64 lowercase"):
            vllm_teacher.load_identity_input(self.path)

        target = Path(self.tmp.name) / "target.json"
        target.write_text(json.dumps(_manifest()))
        self.path.unlink()
        self.path.symlink_to(target)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "must not be a symlink"):
            vllm_teacher.load_identity_input(self.path)

    def test_manifest_adapter_fields_are_strict(self) -> None:
        value = _manifest(
            {"name": "teacher-qwen", "weights_sha256": A_HASH, "config_sha256": B_HASH}
        )
        self._write(value)
        result = vllm_teacher.load_identity_input(self.path)
        self.assertEqual(result["adapter"]["name"], "teacher-qwen")
        value["adapter"]["unexpected"] = True
        self._write(value)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "wrong fields"):
            vllm_teacher.load_identity_input(self.path)

    def test_manifest_runtime_and_accelerator_contracts_are_strict(self) -> None:
        value = _manifest()
        del value["runtime_versions"]["torch"]
        self._write(value)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "runtime_versions"):
            vllm_teacher.load_identity_input(self.path)

        value = _manifest()
        value["runtime_versions"]["vllm"] = "0.25.1"
        self._write(value)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "must match"):
            vllm_teacher.load_identity_input(self.path)

        value = _manifest()
        value["accelerator"]["devices"][0]["index"] = 1
        self._write(value)
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "contiguous"):
            vllm_teacher.load_identity_input(self.path)


class RuntimeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"vocab_size":3}')

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _args(model: Path) -> object:
        return types.SimpleNamespace(
            identity_input=None,
            manifest_only=False,
            dry_run=False,
            model_path=model,
            adapter_path=None,
            served_model_id="teacher-qwen",
        )

    def _identity_inputs(self, vocab: dict[str, int], backend_size: int) -> dict[str, object]:
        with mock.patch.object(
            vllm_teacher, "snapshot_content_fingerprint", return_value=A_HASH
        ), mock.patch.object(
            vllm_teacher,
            "fingerprint_base_model_details",
            return_value=(B_HASH, C_HASH),
        ), mock.patch.object(
            vllm_teacher,
            "_load_tokenizer_contract",
            return_value=(vocab, '{"version":"1.0"}', backend_size),
        ), mock.patch.object(
            vllm_teacher, "_installed_vllm_version", return_value="0.25.0"
        ), mock.patch.object(
            vllm_teacher,
            "installed_runtime_versions",
            return_value=RUNTIME_VERSIONS,
        ), mock.patch.object(
            vllm_teacher,
            "capture_runtime_content",
            return_value=vllm_teacher.RuntimeContentSnapshot(
                sha256=E_HASH,
                python_executable=Path("/resolved/python"),
                file_count=5,
                directory_count=4,
                logical_bytes=100,
            ),
        ), mock.patch.object(
            vllm_teacher, "probe_accelerator", return_value=ACCELERATOR
        ):
            return vllm_teacher._identity_inputs(self._args(self.model))

    def test_identity_inputs_bind_model_width_and_allow_reserved_padding(self) -> None:
        inputs = self._identity_inputs({"a": 0, "b": 1, "c": 2}, 3)
        self.assertEqual(inputs["vocab_size"], 3)
        self.assertEqual(inputs["runtime_versions"], RUNTIME_VERSIONS)
        self.assertEqual(inputs["runtime_content_sha256"], E_HASH)

        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "pair count"):
            self._identity_inputs({"a": 0, "b": 1}, 3)

        (self.model / "config.json").write_text('{"vocab_size":4}')
        inputs = self._identity_inputs({"a": 0, "b": 1, "c": 2}, 3)
        self.assertEqual(inputs["vocab_size"], 4)

        (self.model / "config.json").write_text(
            '{"vocab_size":null,"text_config":{"vocab_size":4}}'
        )
        inputs = self._identity_inputs({"a": 0, "b": 1, "c": 2}, 3)
        self.assertEqual(inputs["vocab_size"], 4)

    def test_identity_inputs_reject_out_of_range_or_ambiguous_model_vocab(self) -> None:
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "maximum token ID"):
            self._identity_inputs({"a": 0, "b": 1, "c": 3}, 3)

        (self.model / "config.json").write_text('{"vocab_size":2}')
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "entry count"):
            self._identity_inputs({"a": 0, "b": 1, "c": 1}, 3)

        (self.model / "config.json").write_text(
            '{"vocab_size":3,"text_config":{"vocab_size":4}}'
        )
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "sizes disagree"):
            self._identity_inputs({"a": 0, "b": 1, "c": 2}, 3)

    def test_installed_runtime_versions_bind_exact_package_versions(self) -> None:
        versions = {
            "vllm": "0.25.2",
            "torch": "2.9.1+cu129",
            "transformers": "5.0.1",
            "tokenizers": "0.22.2",
        }
        with mock.patch.object(
            vllm_teacher.importlib.metadata,
            "version",
            side_effect=lambda package: versions[package],
        ):
            result = vllm_teacher.installed_runtime_versions()
        self.assertEqual(result["vllm"], "0.25.2")
        self.assertEqual(result["torch"], "2.9.1+cu129")
        self.assertEqual(result["python"], vllm_teacher.platform.python_version())

    def test_runtime_content_binds_resolved_executable_editable_source_and_native_extension(
        self,
    ) -> None:
        executable = Path(self.tmp.name) / "python-real"
        executable.write_bytes(b"python-executable")
        executable_link = Path(self.tmp.name) / "python"
        executable_link.symlink_to(executable)

        source_root = Path(self.tmp.name) / "editable" / "vllm"
        source_root.mkdir(parents=True)
        source = source_root / "__init__.py"
        source.write_text("VALUE = 'first'\n")
        native = source_root / "_C.native.so"
        native.write_bytes(b"native-v1")

        site = Path(self.tmp.name) / "site"
        metadata = site / "vllm-0.25.0.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text("Name: vllm\nVersion: 0.25.0\n")
        editable_pointer = site / "__editable__.vllm.pth"
        editable_pointer.write_text(str(source_root.parent) + "\n")
        distribution = types.SimpleNamespace(
            _path=metadata,
            files=[
                Path("__editable__.vllm.pth"),
                Path("vllm-0.25.0.dist-info/METADATA"),
            ],
            locate_file=lambda item: site / item,
        )
        package_spec = types.SimpleNamespace(
            submodule_search_locations=[str(source_root)],
            origin=str(source),
        )

        with mock.patch.object(
            vllm_teacher.importlib.util, "find_spec", return_value=package_spec
        ), mock.patch.object(
            vllm_teacher.importlib.metadata, "distribution", return_value=distribution
        ):
            first = vllm_teacher.capture_runtime_content(
                python_executable=executable_link,
                package_names=("vllm",),
            )
            repeated = vllm_teacher.capture_runtime_content(
                python_executable=executable_link,
                package_names=("vllm",),
            )
            limiter = vllm_teacher._model_fingerprint._ReadRateLimiter(None)
            bounded = vllm_teacher.capture_runtime_content(
                python_executable=executable_link,
                package_names=("vllm",),
                read_rate_limiter=limiter,
            )
            source.write_text("VALUE = 'other'\n")
            changed_source = vllm_teacher.capture_runtime_content(
                python_executable=executable_link,
                package_names=("vllm",),
            )
            source.write_text("VALUE = 'first'\n")
            native.write_bytes(b"native-v2")
            changed_native = vllm_teacher.capture_runtime_content(
                python_executable=executable_link,
                package_names=("vllm",),
            )
            native.write_bytes(b"native-v1")
            executable.write_bytes(b"python-executablE")
            changed_executable = vllm_teacher.capture_runtime_content(
                python_executable=executable_link,
                package_names=("vllm",),
            )

        self.assertEqual(first, repeated)
        self.assertEqual(first, bounded)
        self.assertGreater(limiter.total_bytes, 0)
        self.assertEqual(first.python_executable, executable.resolve())
        self.assertNotEqual(first.sha256, changed_source.sha256)
        self.assertNotEqual(first.sha256, changed_native.sha256)
        self.assertNotEqual(first.sha256, changed_executable.sha256)
        self.assertGreaterEqual(first.file_count, 5)

    def test_runtime_content_enforces_inventory_bounds(self) -> None:
        executable = Path(self.tmp.name) / "python"
        executable.write_bytes(b"python")
        package_root = Path(self.tmp.name) / "package"
        package_root.mkdir()
        origin = package_root / "__init__.py"
        origin.write_bytes(b"package")
        metadata = Path(self.tmp.name) / "package.dist-info"
        metadata.mkdir()
        distribution = types.SimpleNamespace(_path=metadata, files=[], locate_file=lambda item: item)
        package_spec = types.SimpleNamespace(
            submodule_search_locations=[str(package_root)],
            origin=str(origin),
        )
        limits = (
            ("MAX_RUNTIME_CONTENT_BYTES", 5, "logical-size"),
            ("MAX_RUNTIME_CONTENT_FILES", 1, "file safety limit"),
            ("MAX_RUNTIME_CONTENT_DIRECTORIES", 0, "directory safety limit"),
            ("MAX_RUNTIME_CONTENT_DEPTH", 0, "nesting limit"),
            ("MAX_RUNTIME_CONTENT_PATH_BYTES", 1, "path exceeds"),
            ("MAX_RUNTIME_CONTENT_PATH_METADATA_BYTES", 1, "path metadata"),
        )
        with mock.patch.object(
            vllm_teacher.importlib.util, "find_spec", return_value=package_spec
        ), mock.patch.object(
            vllm_teacher.importlib.metadata, "distribution", return_value=distribution
        ):
            for constant, maximum, message in limits:
                with self.subTest(constant=constant), mock.patch.object(
                    vllm_teacher, constant, maximum
                ), self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, message):
                    vllm_teacher.capture_runtime_content(
                        python_executable=executable,
                        package_names=("package",),
                    )

    def test_rocm_probe_binds_driver_name_architecture_and_memory(self) -> None:
        properties = types.SimpleNamespace(
            name="AMD Radeon 8060S",
            gcnArchName="gfx1151",
            total_memory=64 * 1024**3,
        )
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        )
        torch = types.ModuleType("torch")
        torch.cuda = cuda
        torch.version = types.SimpleNamespace(hip="7.1", cuda=None)
        with mock.patch.dict(sys.modules, {"torch": torch}), mock.patch.object(
            vllm_teacher, "_read_bounded_text", return_value="6.14.0"
        ):
            result = vllm_teacher.probe_accelerator()
        self.assertEqual(result["type"], "rocm")
        self.assertIn("hip-runtime:7.1", result["driver"])
        self.assertEqual(result["devices"][0]["name"], "AMD Radeon 8060S")
        self.assertEqual(result["devices"][0]["architecture"], "gfx1151")
        self.assertEqual(result["devices"][0]["total_memory_bytes"], 64 * 1024**3)

    def test_child_runtime_preflight_rejects_resolution_mismatch(self) -> None:
        observed = {
            "schema": vllm_teacher.RUNTIME_CONTENT_SCHEMA,
            "runtime_versions": {
                **RUNTIME_VERSIONS,
                "torch": "2.9.2+rocm7.1",
            },
            "runtime_content_sha256": E_HASH,
            "file_count": 10,
            "directory_count": 5,
            "logical_bytes": 1000,
        }
        completed = vllm_teacher.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(observed).encode(),
            stderr=b"",
        )
        with mock.patch.object(
            vllm_teacher.subprocess, "run", return_value=completed
        ) as run:
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "differ"):
                vllm_teacher.verify_child_runtime_contract(
                    RUNTIME_VERSIONS,
                    E_HASH,
                    {"PYTHONDONTWRITEBYTECODE": "1"},
                )
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["cwd"], os.path.abspath(os.sep))

    def test_child_runtime_preflight_forwards_provenance_read_ceiling(self) -> None:
        observed = {
            "schema": vllm_teacher.RUNTIME_CONTENT_SCHEMA,
            "runtime_versions": RUNTIME_VERSIONS,
            "runtime_content_sha256": E_HASH,
            "file_count": 10,
            "directory_count": 5,
            "logical_bytes": 1000,
        }
        completed = vllm_teacher.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(observed).encode(),
            stderr=b"",
        )
        with mock.patch.object(
            vllm_teacher.subprocess,
            "run",
            return_value=completed,
        ) as run:
            vllm_teacher.verify_child_runtime_contract(
                RUNTIME_VERSIONS,
                E_HASH,
                {"PYTHONDONTWRITEBYTECODE": "1"},
                256,
            )
        probe = run.call_args.args[0][-1]
        self.assertIn("runtime_contract_probe_payload(256)", probe)

    def test_child_runtime_revalidation_rejects_same_version_content_mutation(self) -> None:
        observed = {
            "schema": vllm_teacher.RUNTIME_CONTENT_SCHEMA,
            "runtime_versions": RUNTIME_VERSIONS,
            "runtime_content_sha256": A_HASH,
            "file_count": 10,
            "directory_count": 5,
            "logical_bytes": 1000,
        }
        completed = vllm_teacher.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(observed).encode(),
            stderr=b"",
        )
        with mock.patch.object(
            vllm_teacher.subprocess, "run", return_value=completed
        ) as run:
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "content changed"):
                vllm_teacher.verify_child_runtime_contract(
                    RUNTIME_VERSIONS,
                    E_HASH,
                    {"PYTHONDONTWRITEBYTECODE": "1"},
                )
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["cwd"], os.path.abspath(os.sep))


class ArgumentAndCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / "model"
        self.model.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_extra_args_are_unambiguous_unique_and_launcher_owned(self) -> None:
        self.assertEqual(
            vllm_teacher.validate_extra_vllm_args(
                [
                    "--dtype=bfloat16",
                    "--attention-backend=TRITON_ATTN",
                    "--language-model-only",
                    "--enforce-eager",
                    "--api-key=$(touch /tmp/nope)",
                ]
            ),
            [
                "--dtype=bfloat16",
                "--attention-backend=TRITON_ATTN",
                "--language-model-only",
                "--enforce-eager",
                "--api-key=$(touch /tmp/nope)",
            ],
        )
        cases = (
            (["--dtype", "bfloat16"], "key=value"),
            (["-p=8000"], "ambiguous"),
            (["--dtype=f16", "--dtype=bf16"], "duplicate"),
            (["--language-model-only=true"], "valueless option must not use"),
            (["--model=/tmp/model"], "owns or forbids"),
            (["--enable-lora"], "owns or forbids"),
            (["--enable-prompt-adapter"], "owns or forbids"),
            (["--middleware=x:y"], "owns or forbids"),
            (["--load-format=dummy"], "owns or forbids"),
            (["--quantization-param-path=/tmp/q.json"], "unbound file or path"),
            (["--speculative-model=/tmp/draft"], "not been identity-reviewed"),
            (["--compilation-config={\"cache_dir\":\"/tmp/cache\"}"], "not been identity-reviewed"),
            (["--attention-backend=package.CustomBackend"], "unsupported attention backend"),
            (["--attention-backend=ROCM_ATTN"], "unsupported attention backend"),
            (["--brand-new-option=value"], "not been identity-reviewed"),
        )
        for args, message in cases:
            with self.subTest(args=args):
                with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, message):
                    vllm_teacher.validate_extra_vllm_args(args)

    def test_inference_hash_canonicalizes_args_ignores_transport_and_tracks_runtime(self) -> None:
        kwargs = {
            "snapshot_content_sha256": D_HASH,
            "model_config_sha256": A_HASH,
            "max_top_k": 20,
            "max_model_len": 4096,
            "max_prompt_logprob_candidates": 20_000,
            "adapter_enabled": False,
            "adapter_max_rank": None,
            "runtime_versions": RUNTIME_VERSIONS,
            "runtime_content_sha256": E_HASH,
            "accelerator": ACCELERATOR,
        }
        first = vllm_teacher.inference_config_fingerprint(
            **kwargs,
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2", "--port=8000"],
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        reordered = vllm_teacher.inference_config_fingerprint(
            **kwargs,
            extra_args=["--port=9000", "--tensor-parallel-size=2", "--dtype=bfloat16"],
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        self.assertEqual(first, reordered)
        derived_cache = vllm_teacher.inference_config_fingerprint(
            **kwargs,
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2"],
            environment={
                "CUDA_VISIBLE_DEVICES": "0,1",
                "VLLM_CACHE_ROOT": "/tmp/private-random-cache",
            },
        )
        self.assertEqual(first, derived_cache)
        changed_dtype = vllm_teacher.inference_config_fingerprint(
            **kwargs,
            extra_args=["--dtype=float16", "--tensor-parallel-size=2"],
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        changed_gpu = vllm_teacher.inference_config_fingerprint(
            **kwargs,
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2"],
            environment={"CUDA_VISIBLE_DEVICES": "2,3"},
        )
        changed_model_config = vllm_teacher.inference_config_fingerprint(
            **{**kwargs, "model_config_sha256": B_HASH},
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2"],
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        changed_runtime = vllm_teacher.inference_config_fingerprint(
            **{
                **kwargs,
                "runtime_versions": {**RUNTIME_VERSIONS, "torch": "2.9.2+rocm7.1"},
            },
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2"],
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        changed_runtime_content = vllm_teacher.inference_config_fingerprint(
            **{**kwargs, "runtime_content_sha256": A_HASH},
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2"],
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        changed_accelerator = vllm_teacher.inference_config_fingerprint(
            **{
                **kwargs,
                "accelerator": {
                    **ACCELERATOR,
                    "devices": [
                        {**ACCELERATOR["devices"][0], "architecture": "gfx1201"}
                    ],
                },
            },
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2"],
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        changed_determinism = vllm_teacher.inference_config_fingerprint(
            **kwargs,
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2", "--seed=7"],
            environment={
                "CUDA_VISIBLE_DEVICES": "0,1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        )
        changed_rocm_visibility = vllm_teacher.inference_config_fingerprint(
            **kwargs,
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2"],
            environment={"ROCR_VISIBLE_DEVICES": "1"},
        )
        changed_tool_path = vllm_teacher.inference_config_fingerprint(
            **kwargs,
            extra_args=["--dtype=bfloat16", "--tensor-parallel-size=2"],
            environment={"CUDA_VISIBLE_DEVICES": "0,1", "PATH": "/opt/toolchain/bin"},
        )
        self.assertNotEqual(changed_dtype, first)
        self.assertNotEqual(changed_gpu, first)
        self.assertNotEqual(changed_model_config, first)
        self.assertNotEqual(changed_runtime, first)
        self.assertNotEqual(changed_runtime_content, first)
        self.assertNotEqual(changed_accelerator, first)
        self.assertNotEqual(changed_determinism, first)
        self.assertNotEqual(changed_rocm_visibility, first)
        self.assertNotEqual(changed_tool_path, first)

    def test_base_command_uses_native_custom_fingerprint_and_no_shell_or_middleware(self) -> None:
        fingerprint = vllm_teacher.encode_system_fingerprint(_identity())
        command = vllm_teacher.build_vllm_command(
            model_path=self.model,
            served_model_id="teacher-qwen",
            adapter_path=None,
            adapter_max_rank=None,
            max_top_k=5,
            max_model_len=4096,
            system_fingerprint=fingerprint,
            extra_args=["--host=127.0.0.1", "--api-key=$(touch /tmp/nope)"],
        )
        self.assertEqual(command[:4], [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"])
        self.assertIn("--fingerprint-mode=custom", command)
        self.assertIn(f"--fingerprint-value={fingerprint}", command)
        self.assertIn("--served-model-name=teacher-qwen", command)
        self.assertEqual(command.count("--load-format=safetensors"), 1)
        self.assertNotIn("--enable-lora", command)
        self.assertFalse(any(item.startswith("--middleware") for item in command))
        self.assertIn("--api-key=$(touch /tmp/nope)", command)
        self.assertEqual(
            vllm_teacher._redact_command(command)[-1], "--api-key=<redacted>"
        )

    def test_static_adapter_command_loads_exactly_one_adapter(self) -> None:
        adapter = Path(self.tmp.name) / "adapter"
        adapter.mkdir()
        identity = _identity(
            {"name": "teacher-qwen", "weights_sha256": A_HASH, "config_sha256": B_HASH}
        )
        fingerprint = vllm_teacher.encode_system_fingerprint(identity)
        command = vllm_teacher.build_vllm_command(
            model_path=self.model,
            served_model_id="teacher-qwen",
            adapter_path=adapter,
            adapter_max_rank=32,
            max_top_k=5,
            max_model_len=4096,
            system_fingerprint=fingerprint,
            extra_args=[],
        )
        self.assertEqual(command.count("--enable-lora"), 1)
        self.assertIn("--max-loras=1", command)
        self.assertIn("--max-lora-rank=32", command)
        module_arg = next(item for item in command if item.startswith("--lora-modules="))
        module = json.loads(module_arg.split("=", 1)[1])
        self.assertEqual(module["name"], "teacher-qwen")
        self.assertEqual(module["path"], str(adapter))
        self.assertTrue(module["base_model_name"].startswith("kiln-base-"))
        self.assertEqual(
            module_arg,
            "--lora-modules="
            + json.dumps(
                {
                    "name": "teacher-qwen",
                    "path": str(adapter),
                    "base_model_name": "kiln-base-" + A_HASH[:16],
                },
                separators=(",", ":"),
            ),
        )

    def test_launch_environment_disables_mutation_and_rejects_resolver_plugins(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
                "VLLM_CACHE_ROOT": "/tmp/ambient-cache",
                "PYTHONPATH": "/tmp/shadow",
                "LD_PRELOAD": "/tmp/inject.so",
            },
            clear=True,
        ):
            environment = vllm_teacher.launch_environment()
        self.assertEqual(environment["VLLM_ALLOW_RUNTIME_LORA_UPDATING"], "0")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertNotIn("VLLM_CACHE_ROOT", environment)
        derived = vllm_teacher.launch_environment(Path("/tmp/private-cache"))
        self.assertEqual(derived["VLLM_CACHE_ROOT"], "/tmp/private-cache")
        vllm_teacher.validate_launch_environment({})
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "VLLM_PLUGINS"):
            vllm_teacher.validate_launch_environment(
                {"VLLM_PLUGINS": "lora_filesystem_resolver"}
            )
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "MODEL_NAME_VALIDATION"):
            vllm_teacher.validate_launch_environment(
                {"VLLM_SKIP_MODEL_NAME_VALIDATION": "true"}
            )
        for key in (
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "HIPBLASLT_TUNING_OVERRIDE_FILE",
            "CC",
            "LIBRARY_PATH",
            "NVCC",
            "VLLM_NCCL_SO_PATH",
            "VLLM_LOGGING_CONFIG_PATH",
            "TRITON_CACHE_DIR",
            "VLLM_CACHE_ROOT",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                vllm_teacher.TeacherLaunchError, key
            ):
                vllm_teacher.validate_launch_environment({key: "/tmp/unbound"})


class ProcessSupervisionTests(unittest.TestCase):
    def test_child_uses_no_shell_new_session_fixed_cwd_and_maps_signal_exit(self) -> None:
        child = mock.Mock(pid=4242)
        child.wait.return_value = -signal.SIGTERM
        child.poll.return_value = -signal.SIGTERM
        with mock.patch.object(
            vllm_teacher.subprocess, "Popen", return_value=child
        ) as popen, mock.patch.object(
            vllm_teacher, "_process_group_exists", return_value=False
        ):
            code = vllm_teacher.run_vllm_child(["python", "-m", "vllm"], {"A": "B"})
        self.assertEqual(code, 128 + signal.SIGTERM)
        kwargs = popen.call_args.kwargs
        self.assertFalse(kwargs["shell"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["cwd"], os.path.abspath(os.sep))

    def test_inherited_child_stays_in_preisolated_supervisor_group(self) -> None:
        child = mock.Mock(pid=4243)
        child.wait.return_value = 0
        child.poll.return_value = 0
        with mock.patch.object(
            vllm_teacher, "_require_isolated_inherited_process_group", return_value=4242
        ), mock.patch.object(
            vllm_teacher.subprocess, "Popen", return_value=child
        ) as popen, mock.patch.object(
            vllm_teacher, "_drain_inherited_process_group"
        ) as drain:
            code = vllm_teacher.run_vllm_child(
                ["python", "-m", "vllm"],
                {"A": "B"},
                process_group_mode=vllm_teacher.PROCESS_GROUP_MODE_INHERITED,
            )
        self.assertEqual(code, 0)
        self.assertFalse(popen.call_args.kwargs["start_new_session"])
        drain.assert_called_once_with(4242)

    def test_inherited_mode_requires_launcher_to_lead_group(self) -> None:
        with mock.patch.object(
            vllm_teacher.os, "getpid", return_value=4242
        ), mock.patch.object(
            vllm_teacher.os, "getpgrp", return_value=4000
        ), mock.patch.object(
            vllm_teacher.subprocess, "Popen"
        ) as popen:
            with self.assertRaisesRegex(
                vllm_teacher.TeacherLaunchError, "lead an isolated group"
            ):
                vllm_teacher.run_vllm_child(
                    ["vllm"],
                    {},
                    process_group_mode=vllm_teacher.PROCESS_GROUP_MODE_INHERITED,
                )
        popen.assert_not_called()

    def test_inherited_signal_reaches_child_without_group_signal(self) -> None:
        handlers: dict[int, object] = {}
        child = mock.Mock(pid=4243)

        def wait(*, timeout: float | None = None) -> int:
            self.assertIsNone(timeout)
            handlers[signal.SIGINT](signal.SIGINT, None)
            return 0

        child.wait.side_effect = wait
        child.poll.return_value = None

        def install(signum: int, handler: object) -> None:
            if callable(handler):
                handlers[signum] = handler

        with mock.patch.object(
            vllm_teacher, "_require_isolated_inherited_process_group", return_value=4242
        ), mock.patch.object(
            vllm_teacher.subprocess, "Popen", return_value=child
        ), mock.patch.object(
            vllm_teacher.signal, "getsignal", return_value="previous"
        ), mock.patch.object(
            vllm_teacher.signal, "signal", side_effect=install
        ), mock.patch.object(
            vllm_teacher, "_drain_inherited_process_group"
        ), mock.patch.object(vllm_teacher.os, "killpg") as killpg:
            self.assertEqual(
                vllm_teacher.run_vllm_child(
                    ["vllm"],
                    {},
                    process_group_mode=vllm_teacher.PROCESS_GROUP_MODE_INHERITED,
                ),
                0,
            )
        child.send_signal.assert_called_once_with(signal.SIGINT)
        killpg.assert_not_called()

    def test_inherited_drain_signals_only_identity_revalidated_peers(self) -> None:
        group_states = iter(({4243: 99}, {4243: 99}, {}))
        with mock.patch.object(
            vllm_teacher,
            "_inherited_process_group_members",
            side_effect=lambda _group: next(group_states),
        ), mock.patch.object(
            vllm_teacher, "_proc_process_identity", return_value=(4242, 99)
        ), mock.patch.object(vllm_teacher.os, "kill") as kill:
            vllm_teacher._drain_inherited_process_group(4242)
        kill.assert_called_once_with(4243, signal.SIGTERM)

    @unittest.skipUnless(Path("/proc/self/stat").is_file(), "Linux /proc required")
    def test_real_inherited_child_observes_launcher_process_group(self) -> None:
        module_path = os.fspath(MODULE_PATH)
        child_code = (
            "import json,os;"
            "print(json.dumps({'role':'child','pid':os.getpid(),"
            "'process_group':os.getpgrp()}),flush=True)"
        )
        supervisor_code = (
            "import importlib.util,json,os,sys;"
            f"p={module_path!r};"
            "s=importlib.util.spec_from_file_location('_kiln_vllm_group_test',p);"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
            "s.loader.exec_module(m);"
            "print(json.dumps({'role':'supervisor','pid':os.getpid(),"
            "'process_group':os.getpgrp()}),flush=True);"
            f"c=m.run_vllm_child([sys.executable,'-c',{child_code!r}],{{}},"
            "process_group_mode=m.PROCESS_GROUP_MODE_INHERITED);"
            "print(json.dumps({'role':'exit','code':c}),flush=True)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", supervisor_code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            start_new_session=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        records = [json.loads(line) for line in completed.stdout.splitlines()]
        by_role = {record["role"]: record for record in records}
        self.assertEqual(
            by_role["supervisor"]["pid"],
            by_role["supervisor"]["process_group"],
        )
        self.assertEqual(
            by_role["child"]["process_group"],
            by_role["supervisor"]["process_group"],
        )
        self.assertEqual(by_role["exit"]["code"], 0)

    def test_forwarded_signal_is_sent_once_to_detached_process_group(self) -> None:
        handlers: dict[int, object] = {}
        child = mock.Mock(pid=4242)

        def wait(*, timeout: float | None = None) -> int:
            self.assertIsNone(timeout)
            handlers[signal.SIGINT](signal.SIGINT, None)
            return 0

        child.wait.side_effect = wait
        child.poll.return_value = None

        def install(signum: int, handler: object) -> None:
            if callable(handler):
                handlers[signum] = handler

        with mock.patch.object(
            vllm_teacher.subprocess, "Popen", return_value=child
        ), mock.patch.object(
            vllm_teacher.signal, "getsignal", return_value="previous"
        ), mock.patch.object(
            vllm_teacher.signal, "signal", side_effect=install
        ), mock.patch.object(
            vllm_teacher, "_process_group_exists", return_value=False
        ), mock.patch.object(vllm_teacher.os, "killpg") as killpg:
            self.assertEqual(vllm_teacher.run_vllm_child(["vllm"], {}), 0)
        self.assertEqual(killpg.call_args_list, [mock.call(4242, signal.SIGINT)])

    def test_leftover_descendants_are_terminated_before_return(self) -> None:
        child = mock.Mock(pid=4242)
        child.wait.return_value = 0
        child.poll.return_value = 0
        group_states = iter((True, False))
        with mock.patch.object(
            vllm_teacher.subprocess, "Popen", return_value=child
        ), mock.patch.object(
            vllm_teacher,
            "_process_group_exists",
            side_effect=lambda _pid: next(group_states),
        ), mock.patch.object(vllm_teacher.os, "killpg") as killpg:
            self.assertEqual(vllm_teacher.run_vllm_child(["vllm"], {}), 0)
        killpg.assert_called_once_with(4242, signal.SIGTERM)

    def test_wait_exception_terminates_group_and_restores_handlers(self) -> None:
        child = mock.Mock(pid=4242)
        child.poll.return_value = None
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with mock.patch.object(
            vllm_teacher.subprocess, "Popen", return_value=child
        ), mock.patch.object(
            vllm_teacher, "_process_group_exists", return_value=False
        ), mock.patch.object(vllm_teacher.os, "killpg") as killpg:
            with self.assertRaises(KeyboardInterrupt):
                vllm_teacher.run_vllm_child(["vllm"], {})
        killpg.assert_called_once_with(4242, signal.SIGTERM)
        self.assertEqual(child.wait.call_args_list[-1], mock.call(timeout=10.0))

    def test_signal_handler_failure_prevents_spawn(self) -> None:
        with mock.patch.object(
            vllm_teacher.signal, "signal", side_effect=ValueError("not main thread")
        ), mock.patch.object(vllm_teacher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "main thread"):
                vllm_teacher.run_vllm_child(["vllm"], {})
        popen.assert_not_called()


class MainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = Path(self.tmp.name) / "input.json"
        self.manifest.write_text(json.dumps(_manifest()))
        self.model = Path(self.tmp.name) / "model"
        self.model.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _common(self) -> list[str]:
        return [
            "--served-model-id",
            "teacher-qwen",
            "--max-top-k",
            "5",
            "--max-model-len",
            "4096",
            "--identity-input",
            str(self.manifest),
        ]

    def test_process_group_mode_is_closed_and_defaults_to_detached(self) -> None:
        parsed = vllm_teacher.parse_args(self._common() + ["--manifest-only"])
        self.assertEqual(
            parsed.process_group_mode, vllm_teacher.PROCESS_GROUP_MODE_DETACHED
        )
        inherited = vllm_teacher.parse_args(
            [
                *self._common(),
                "--process-group-mode",
                vllm_teacher.PROCESS_GROUP_MODE_INHERITED,
                "--manifest-only",
            ]
        )
        self.assertEqual(
            inherited.process_group_mode, vllm_teacher.PROCESS_GROUP_MODE_INHERITED
        )

    def test_provenance_read_ceiling_is_typed_and_closed(self) -> None:
        parsed = vllm_teacher.parse_args(
            [
                *self._common(),
                "--max-provenance-read-mib-per-second=256",
                "--manifest-only",
            ]
        )
        vllm_teacher._validate_requested_limits(parsed)
        self.assertEqual(parsed.max_provenance_read_mib_per_second, 256)

        for value in (0, 16_385):
            with self.subTest(value=value):
                invalid = vllm_teacher.parse_args(
                    [
                        *self._common(),
                        f"--max-provenance-read-mib-per-second={value}",
                        "--manifest-only",
                    ]
                )
                with self.assertRaisesRegex(
                    vllm_teacher.TeacherLaunchError,
                    "max-provenance-read",
                ):
                    vllm_teacher._validate_requested_limits(invalid)

    def test_manifest_only_is_dependency_free_json(self) -> None:
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            vllm_teacher, "_load_tokenizer_contract", side_effect=AssertionError("not called")
        ), contextlib.redirect_stdout(stdout):
            code = vllm_teacher.main([*self._common(), "--manifest-only"])
        self.assertEqual(code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["identity"]["schema"], vllm_teacher.IDENTITY_SCHEMA)
        self.assertEqual(
            vllm_teacher.decode_system_fingerprint(output["system_fingerprint"]),
            output["identity"],
        )
        self.assertEqual(output["runtime_content_sha256"], E_HASH)
        self.assertNotIn("command", output)

    def test_dry_run_is_dependency_free_and_redacts_api_key(self) -> None:
        stdout = io.StringIO()
        args = [
            *self._common(),
            "--model-path",
            str(self.model),
            "--dry-run",
            "--",
            "--host=127.0.0.1",
            "--api-key=secret",
        ]
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(stdout):
            code = vllm_teacher.main(args)
        self.assertEqual(code, 0)
        output = json.loads(stdout.getvalue())
        self.assertIn("--api-key=<redacted>", output["command"])
        self.assertNotIn("--api-key=secret", output["command"])
        self.assertEqual(output["runtime_lora_updates"], "disabled")

    def test_manifest_cannot_launch_and_conflicting_adapter_fails(self) -> None:
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(stderr):
            code = vllm_teacher.main(self._common())
        self.assertEqual(code, 2)
        self.assertIn("forbidden for a real launch", stderr.getvalue())

        adapter = Path(self.tmp.name) / "adapter"
        adapter.mkdir()
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(stderr):
            code = vllm_teacher.main(
                [*self._common(), "--adapter-path", str(adapter), "--manifest-only"]
            )
        self.assertEqual(code, 2)
        self.assertIn("conflicts with the null manifest adapter", stderr.getvalue())

    def test_real_launch_uses_verified_snapshot_until_supervised_child_exits(self) -> None:
        called: dict[str, object] = {}

        def fake_inputs(
            args: object,
            read_rate_limiter: object | None = None,
        ) -> dict[str, object]:
            self.assertIsNotNone(read_rate_limiter)
            called["staged_model"] = args.model_path
            self.assertTrue(args.model_path.parent.name.startswith("ready-"))
            self.assertNotEqual(args.model_path, self.model)
            return {
                "base_model_sha256": A_HASH,
                "snapshot_content_sha256": args.snapshot_content_sha256,
                "model_config_sha256": D_HASH,
                "tokenizer_vocab_sha256": B_HASH,
                "tokenizer_config_sha256": C_HASH,
                "adapter": None,
                "adapter_max_rank": None,
                "vocab_size": 8,
                "implementation": "vllm:0.25.0",
                "runtime_versions": RUNTIME_VERSIONS,
                "runtime_content_sha256": E_HASH,
                "accelerator": ACCELERATOR,
            }

        def fake_run(
            command: list[str],
            environment: dict[str, str],
            *,
            process_group_mode: str,
        ) -> int:
            staged_model = called["staged_model"]
            self.assertTrue(staged_model.exists())
            self.assertIn(str(staged_model), command)
            self.assertTrue(called.get("runtime_verified"))
            self.assertEqual(
                process_group_mode, vllm_teacher.PROCESS_GROUP_MODE_DETACHED
            )
            runtime_cache = Path(environment["VLLM_CACHE_ROOT"])
            self.assertTrue(runtime_cache.exists())
            self.assertEqual(runtime_cache.stat().st_mode & 0o777, 0o700)
            called["runtime_cache"] = runtime_cache
            called.update(command=command, environment=environment)
            return 23

        def fake_verify(
            versions: dict[str, str],
            digest: str,
            environment: dict[str, str],
            max_provenance_read_mib_per_second: int | None,
        ) -> None:
            self.assertEqual(versions, RUNTIME_VERSIONS)
            self.assertEqual(digest, E_HASH)
            self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertIsNone(max_provenance_read_mib_per_second)
            called["runtime_verified"] = True

        args = [
            "--model-path",
            str(self.model),
            "--served-model-id",
            "teacher-qwen",
            "--max-top-k",
            "5",
            "--max-model-len",
            "4096",
            "--snapshot-root",
            str(Path(self.tmp.name) / "snapshots"),
            "--cache-root",
            str(Path(self.tmp.name) / "runtime-caches"),
            "--",
            "--api-key=$(touch /tmp/must-not-run)",
        ]
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            vllm_teacher, "_identity_inputs", side_effect=fake_inputs
        ), mock.patch.object(
            vllm_teacher, "run_vllm_child", side_effect=fake_run
        ), mock.patch.object(
            vllm_teacher, "verify_child_runtime_contract", side_effect=fake_verify
        ), contextlib.redirect_stdout(io.StringIO()):
            code = vllm_teacher.main(args)
        self.assertEqual(code, 23)
        self.assertIsInstance(called["command"], list)
        self.assertIn("--api-key=$(touch /tmp/must-not-run)", called["command"])
        self.assertEqual(
            called["environment"]["VLLM_ALLOW_RUNTIME_LORA_UPDATING"], "0"
        )
        self.assertFalse(called["staged_model"].exists())
        self.assertFalse(called["runtime_cache"].exists())

    def test_runtime_cache_must_be_separate_from_model_and_snapshot_roots(self) -> None:
        parsed = vllm_teacher.parse_args(
            [
                "--model-path",
                str(self.model),
                "--served-model-id",
                "teacher-qwen",
                "--max-top-k",
                "5",
                "--max-model-len",
                "4096",
                "--snapshot-root",
                str(Path(self.tmp.name) / "snapshots"),
                "--cache-root",
                str(self.model / "cache"),
            ]
        )
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "non-nested"):
            vllm_teacher._validate_runtime_cache_separation(parsed)

    def test_invalid_limits_fail_before_snapshot_work(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            vllm_teacher,
            "create_immutable_snapshot",
            side_effect=AssertionError("must not stage"),
        ), mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(stderr):
            code = vllm_teacher.main(
                [
                    "--model-path",
                    str(self.model),
                    "--served-model-id",
                    "teacher-qwen",
                    "--max-top-k",
                    "5",
                    "--max-model-len",
                    "0",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("--max-model-len", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
