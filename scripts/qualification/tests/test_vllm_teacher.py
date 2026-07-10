from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
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
        implementation="vllm:0.25.1+cu129",
        inference_config_sha256=D_HASH,
    )


def _manifest(adapter: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema": vllm_teacher.INPUT_MANIFEST_SCHEMA,
        "base_model_sha256": A_HASH,
        "model_config_sha256": D_HASH,
        "tokenizer_vocab_sha256": B_HASH,
        "tokenizer_config_sha256": C_HASH,
        "adapter": adapter,
        "adapter_max_rank": 32 if adapter is not None else None,
        "vocab_size": 8,
        "implementation": "vllm:0.25.0",
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
            "implementation": "vllm:0.25.0",
            "inference_config_sha256": D_HASH,
        }
        cases = (
            ({"served_model_id": "../bad"}, "served_model_id"),
            ({"base_model_sha256": "sha256:" + A_HASH}, "64 lowercase"),
            ({"max_top_k": 9}, "max_top_k"),
            ({"max_model_len": 0}, "max_model_len"),
            ({"implementation": "vllm:0.24.9"}, "0.25.0 or newer"),
        )
        for update, message in cases:
            with self.subTest(update=update):
                with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, message):
                    vllm_teacher.build_identity(**{**base, **update})


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
                ["--dtype=bfloat16", "--enforce-eager", "--api-key=$(touch /tmp/nope)"]
            ),
            ["--dtype=bfloat16", "--enforce-eager", "--api-key=$(touch /tmp/nope)"],
        )
        cases = (
            (["--dtype", "bfloat16"], "key=value"),
            (["-p=8000"], "ambiguous"),
            (["--dtype=f16", "--dtype=bf16"], "duplicate"),
            (["--model=/tmp/model"], "owns or forbids"),
            (["--enable-lora"], "owns or forbids"),
            (["--enable-prompt-adapter"], "owns or forbids"),
            (["--middleware=x:y"], "owns or forbids"),
            (["--load-format=dummy"], "auto or safetensors"),
        )
        for args, message in cases:
            with self.subTest(args=args):
                with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, message):
                    vllm_teacher.validate_extra_vllm_args(args)

    def test_inference_hash_canonicalizes_args_ignores_transport_and_tracks_runtime(self) -> None:
        kwargs = {
            "model_config_sha256": A_HASH,
            "max_top_k": 20,
            "max_model_len": 4096,
            "adapter_enabled": False,
            "adapter_max_rank": None,
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
        self.assertNotEqual(changed_dtype, first)
        self.assertNotEqual(changed_gpu, first)
        self.assertNotEqual(changed_model_config, first)

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
        with mock.patch.dict(os.environ, {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}, clear=True):
            environment = vllm_teacher.launch_environment()
        self.assertEqual(environment["VLLM_ALLOW_RUNTIME_LORA_UPDATING"], "0")
        vllm_teacher.validate_launch_environment({})
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "VLLM_PLUGINS"):
            vllm_teacher.validate_launch_environment(
                {"VLLM_PLUGINS": "lora_filesystem_resolver"}
            )
        with self.assertRaisesRegex(vllm_teacher.TeacherLaunchError, "MODEL_NAME_VALIDATION"):
            vllm_teacher.validate_launch_environment(
                {"VLLM_SKIP_MODEL_NAME_VALIDATION": "true"}
            )


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

    def test_real_launch_uses_execve_directly(self) -> None:
        inputs = {
            "base_model_sha256": A_HASH,
            "model_config_sha256": D_HASH,
            "tokenizer_vocab_sha256": B_HASH,
            "tokenizer_config_sha256": C_HASH,
            "adapter": None,
            "adapter_max_rank": None,
            "vocab_size": 8,
            "implementation": "vllm:0.25.0",
        }
        called: dict[str, object] = {}

        def fake_execve(executable: str, command: list[str], environment: dict[str, str]) -> None:
            called.update(executable=executable, command=command, environment=environment)
            raise RuntimeError("exec sentinel")

        args = [
            "--model-path",
            str(self.model),
            "--served-model-id",
            "teacher-qwen",
            "--max-top-k",
            "5",
            "--max-model-len",
            "4096",
            "--",
            "--api-key=$(touch /tmp/must-not-run)",
        ]
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            vllm_teacher, "_identity_inputs", return_value=inputs
        ), mock.patch.object(
            vllm_teacher.os, "execve", side_effect=fake_execve
        ), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "exec sentinel"):
                vllm_teacher.main(args)
        self.assertEqual(called["executable"], sys.executable)
        self.assertIsInstance(called["command"], list)
        self.assertIn("--api-key=$(touch /tmp/must-not-run)", called["command"])
        self.assertEqual(
            called["environment"]["VLLM_ALLOW_RUNTIME_LORA_UPDATING"], "0"
        )


if __name__ == "__main__":
    unittest.main()
