from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "hf_trl" / "train_sft.py"
LOCK = ROOT / "scripts" / "hf_trl" / "requirements-sft.lock"
SPEC = importlib.util.spec_from_file_location("kiln_hf_trl_train_sft", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
train_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_sft)


def identity(path: Path, relative: str) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "relative_path": relative,
        "size_bytes": len(data),
        "sha256": train_sft._sha256_bytes(data),
    }


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class BundleFixture:
    def __init__(self, root: Path) -> None:
        self.bundle = root / "fixture.kiln-hf"
        self.base_model = root / "base-model"
        self.bundle.mkdir()
        self.base_model.mkdir()

        files = {
            "kiln_model_config.json": b"{}",
            "tokenizer.json": b'{"version":"1.0"}',
            "chat_template.jinja": b"{{ messages }}\n",
            "kiln_training_chat_template.jinja": b"{{ messages }}\n",
            "training_chat_template.jinja": (
                b"{% for message in messages %}{% if message.role == 'assistant' %}"
                b"{% generation %}{{ message.content }}{% endgeneration %}{% else %}"
                b"{{ message.content }}{% endif %}{% endfor %}\n"
            ),
            "train.jsonl": (
                b'{"messages":[{"role":"user","content":"a"},'
                b'{"role":"assistant","content":"b"}]}\n'
            ),
            "train.py": SCRIPT.read_bytes(),
            "requirements.lock": LOCK.read_bytes(),
        }
        for relative, data in files.items():
            write(self.bundle / relative, data)

        row_hash = "sha256:" + "1" * 64
        corpus_hash = "sha256:" + "2" * 64
        ingestion = {
            "schema": "kiln.sft-ingestion.v1",
            "source": "inline",
            "invalid_row_policy": "fail",
            "rows_read": 1,
            "rows_kept": 1,
            "rows_rejected": 0,
            "kept_row_hashes": [row_hash],
            "rejected_rows": [],
            "kept_corpus_sha256": corpus_hash,
        }
        write(
            self.bundle / "sft_ingestion.json",
            json.dumps(ingestion, indent=2).encode("utf-8"),
        )

        shard_bytes = b"fake safetensors shard"
        write(self.base_model / "model.safetensors", shard_bytes)
        write(self.base_model / "tokenizer.json", files["tokenizer.json"])
        write(self.base_model / "config.json", b"{}")

        model_config_hash = identity(
            self.bundle / "kiln_model_config.json", "kiln_model_config.json"
        )["sha256"]
        tokenizer_hash = identity(
            self.bundle / "tokenizer.json", "tokenizer.json"
        )["sha256"]
        chat_hash = identity(
            self.bundle / "chat_template.jinja", "chat_template.jinja"
        )["sha256"]
        native_template_hash = identity(
            self.bundle / "kiln_training_chat_template.jinja",
            "kiln_training_chat_template.jinja",
        )["sha256"]
        vocab_hash = "sha256:" + "3" * 64
        zero_hash = "sha256:" + "0" * 64
        source_provenance = {
            "schema_version": 1,
            "provenance_type": "kiln.execution-provenance.v1",
            "backend": {
                "name": "test",
                "device": "cpu",
                "numerical_runtime_sha256": zero_hash,
            },
            "build": {
                "package_version": "test",
                "target": "test",
                "executable_sha256": zero_hash,
            },
            "model": {
                "model_config_sha256": model_config_hash,
                "tokenizer_vocab_sha256": vocab_hash,
                "tokenizer_config_sha256": tokenizer_hash,
                "chat_template_sha256": chat_hash,
                "training_chat_template_sha256": native_template_hash,
            },
            "precision": {
                "inference_dtype": "f32",
                "training_policy": "test",
            },
            "kernels": {
                "contract_type": "kiln.kernel-contract.v1",
                "versions": {"test": "1"},
                "compiled_features": [],
                "contract_sha256": zero_hash,
            },
            "configuration": {
                "effective_server_config_sha256": zero_hash,
                "effective_environment_sha256": zero_hash,
            },
            "provenance_sha256": zero_hash,
        }
        manifest = {
            "schema_version": 1,
            "manifest_type": "kiln.hf-trl-export.v1",
            "task": "sft",
            "source_execution_provenance": source_provenance,
            "model": {
                "served_model_id": "fixture/model",
                "base_weight_shard_manifest": {
                    "schema_version": 1,
                    "manifest_type": "kiln.base-weight-shards.v1",
                    "aggregate_algorithm": "kiln.base-model-content.v1",
                    "aggregate_sha256": zero_hash,
                    "total_size_bytes": len(shard_bytes),
                    "shards": [
                        {
                            "filename": "model.safetensors",
                            "size_bytes": len(shard_bytes),
                            "sha256": train_sft._sha256_bytes(shard_bytes),
                        }
                    ],
                },
                "tokenizer_vocab_sha256": vocab_hash,
                "model_config": identity(
                    self.bundle / "kiln_model_config.json",
                    "kiln_model_config.json",
                ),
                "tokenizer": identity(
                    self.bundle / "tokenizer.json", "tokenizer.json"
                ),
                "chat_template": identity(
                    self.bundle / "chat_template.jinja", "chat_template.jinja"
                ),
                "native_training_chat_template": identity(
                    self.bundle / "kiln_training_chat_template.jinja",
                    "kiln_training_chat_template.jinja",
                ),
                "trl_training_chat_template": identity(
                    self.bundle / "training_chat_template.jinja",
                    "training_chat_template.jinja",
                ),
            },
            "data": {
                "source_name": "inline",
                "format": "sft_messages_jsonl",
                "row_count": 1,
                "ordered_corpus_sha256": corpus_hash,
                "dataset": identity(
                    self.bundle / "train.jsonl", "train.jsonl"
                ),
                "sft_selection": {
                    "invalid_row_policy": "fail",
                    "label_policy": "assistant_only_generation_spans",
                    "rows_read": 1,
                    "rows_kept": 1,
                    "rows_rejected": 0,
                    "kept_corpus_sha256": corpus_hash,
                    "ingestion_receipt": identity(
                        self.bundle / "sft_ingestion.json",
                        "sft_ingestion.json",
                    ),
                },
            },
            "reference_script": identity(self.bundle / "train.py", "train.py"),
            "environment_lock": identity(
                self.bundle / "requirements.lock", "requirements.lock"
            ),
        }
        base_manifest = manifest["model"]["base_weight_shard_manifest"]
        base_manifest["aggregate_sha256"] = train_sft._base_weight_aggregate_sha256(
            base_manifest["shards"]
        )
        manifest["export_sha256"] = train_sft._canonical_sha256(manifest)
        self.manifest = manifest
        self.write_manifest()

    def write_manifest(self) -> None:
        (self.bundle / "kiln_hf_export.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )


class GrpoBundleFixture(BundleFixture):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        (self.bundle / "sft_ingestion.json").unlink()
        messages = [{"role": "user", "content": "a"}]
        prompt_sha256 = train_sft._rollout_prompt_sha256(messages)
        behavior = {
            "served_model_id": self.manifest["model"]["served_model_id"],
            "base_model_sha256": self.manifest["model"][
                "base_weight_shard_manifest"
            ]["aggregate_sha256"],
            "inference_config_sha256": "sha256:" + "4" * 64,
            "implementation": "kiln-test/cpu",
        }
        tokenizer = {
            "vocab_sha256": self.manifest["model"]["tokenizer_vocab_sha256"],
            "config_sha256": self.manifest["model"]["tokenizer"]["sha256"],
            "chat_template_sha256": self.manifest["model"]["chat_template"][
                "sha256"
            ],
        }
        sampling = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 2,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": [],
            "thinking_budget": {
                "max_tokens": 1,
                "close_token_ids": [21],
            },
        }
        specs = [
            ("bad", 0.0, [10, 20, 21], [(1, 20, "sampled", -0.5), (2, 21, "forced", None)]),
            ("good", 1.0, [10, 22], [(1, 22, "sampled", -0.25)]),
        ]
        completions = []
        for seed, (text, reward, input_ids, actions) in enumerate(specs, 7):
            provenance = {
                "schema": train_sft.GRPO_PROVENANCE_SCHEMA,
                "input_token_ids": input_ids,
                "prompt_token_count": 1,
                "prompt_messages_sha256": prompt_sha256,
                "scored_payload_sha256": train_sft._scored_payload_sha256(
                    text, []
                ),
                "action_tokens": [
                    {
                        "sequence_index": position,
                        "token_id": token,
                        "source": source,
                        "behavior_logprob": logprob,
                    }
                    for position, token, source, logprob in actions
                ],
                "behavior_policy": copy.deepcopy(behavior),
                "tokenizer": copy.deepcopy(tokenizer),
                "sampling": copy.deepcopy(sampling),
                "seed": seed,
                "generation_backend": "cpu",
            }
            completions.append(
                {"text": text, "reward": reward, "provenance": provenance}
            )
        self.group = {"messages": messages, "completions": completions}
        self.write_groups([self.group])
        self.manifest["task"] = "grpo"
        self.manifest["data"] = {
            "source_name": "rollouts",
            "format": "grpo_groups_jsonl",
            "row_count": 1,
            "ordered_corpus_sha256": train_sft._ordered_grpo_corpus_sha256(
                [train_sft._compact_json_bytes(self.group)]
            ),
            "dataset": identity(self.bundle / "train.jsonl", "train.jsonl"),
            "rollout_provenance_schema": train_sft.GRPO_PROVENANCE_SCHEMA,
        }
        self.rehash_manifest()

    def write_groups(self, groups: list[dict[str, object]]) -> None:
        rows = [train_sft._compact_json_bytes(group) for group in groups]
        write(self.bundle / "train.jsonl", b"\n".join(rows) + b"\n")

    def rehash_manifest(self) -> None:
        self.manifest["data"]["dataset"] = identity(
            self.bundle / "train.jsonl", "train.jsonl"
        )
        rows = (self.bundle / "train.jsonl").read_bytes().splitlines()
        self.manifest["data"]["ordered_corpus_sha256"] = (
            train_sft._ordered_grpo_corpus_sha256(rows)
        )
        self.manifest["data"]["row_count"] = len(rows)
        fields = {
            key: value
            for key, value in self.manifest.items()
            if key != "export_sha256"
        }
        self.manifest["export_sha256"] = train_sft._canonical_sha256(fields)
        self.write_manifest()


class HfTrlSftReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = BundleFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_canonical_digest_matches_rust_cross_language_golden(self) -> None:
        value = {
            "z": [3, {"kind": "decimal", "value": "2e-5"}],
            "a": {"text": "qwen/3.5", "flag": True},
            "n": 42,
        }
        self.assertEqual(
            train_sft._canonical_sha256(value),
            "sha256:1a00ddf29092a9e9e9ecd4ea9faf9c48f69208e5439834c6674d1d6bfb45e258",
        )
        self.assertEqual(
            train_sft._base_weight_aggregate_sha256(
                [
                    {
                        "sha256": "sha256:" + "22" * 32,
                        "size_bytes": 22,
                    },
                    {
                        "sha256": "sha256:" + "11" * 32,
                        "size_bytes": 11,
                    },
                ]
            ),
            "sha256:813979f1d3dd938e49874427651a97344cd1af5409e75aceaa741707616cd7a5",
        )

    def test_export_and_base_bytes_verify_then_tampering_fails(self) -> None:
        root, manifest = train_sft.load_export_bundle(self.fixture.bundle)
        self.assertEqual(
            train_sft.verify_base_model_source(
                self.fixture.base_model, root, manifest
            ),
            self.fixture.base_model.resolve(),
        )
        self.assertTrue(train_sft._script_matches_export(root, manifest))

        with (self.fixture.bundle / "train.jsonl").open("ab") as handle:
            handle.write(b" ")
        with self.assertRaisesRegex(train_sft.ContractError, "size differs"):
            train_sft.load_export_bundle(self.fixture.bundle)

    def test_unknown_manifest_field_and_duplicate_json_key_fail(self) -> None:
        self.fixture.manifest["surprise"] = True
        self.fixture.manifest["export_sha256"] = train_sft._canonical_sha256(
            {key: value for key, value in self.fixture.manifest.items() if key != "export_sha256"}
        )
        self.fixture.write_manifest()
        with self.assertRaisesRegex(train_sft.ContractError, "unknown=.*surprise"):
            train_sft.load_export_bundle(self.fixture.bundle)

        duplicate = self.fixture.bundle / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaisesRegex(train_sft.ContractError, "duplicate JSON key"):
            train_sft._read_json(duplicate)

    @unittest.skipIf(os.name == "nt", "symlink permissions differ on Windows")
    def test_bundle_symlink_substitution_fails(self) -> None:
        template = self.fixture.bundle / "training_chat_template.jinja"
        saved = self.fixture.bundle / "saved-template"
        template.rename(saved)
        template.symlink_to(saved.name)
        with self.assertRaisesRegex(train_sft.ContractError, "symlink"):
            train_sft.load_export_bundle(self.fixture.bundle)

    def test_base_shard_mismatch_fails_before_training_imports(self) -> None:
        root, manifest = train_sft.load_export_bundle(self.fixture.bundle)
        (self.fixture.base_model / "model.safetensors").write_bytes(
            b"different shard bytes!!"
        )
        with self.assertRaisesRegex(train_sft.ContractError, "wrong size|differs"):
            train_sft.verify_base_model_source(
                self.fixture.base_model, root, manifest
            )

    def test_base_tokenizer_accepts_only_equivalent_merge_framing(self) -> None:
        canonical = self.fixture.bundle / "canonical-tokenizer.json"
        legacy = self.fixture.bundle / "legacy-tokenizer.json"
        changed = self.fixture.bundle / "changed-tokenizer.json"
        malformed = self.fixture.bundle / "malformed-tokenizer.json"
        write(
            canonical,
            b'{"version":"1.0","model":{"type":"BPE","merges":[["a","b"],["c","d"]]}}',
        )
        write(
            legacy,
            b'{"version":"1.0","model":{"type":"BPE","merges":["a b","c d"]}}',
        )
        write(
            changed,
            b'{"version":"1.0","model":{"type":"BPE","merges":["a b","c e"]}}',
        )
        write(
            malformed,
            b'{"version":"1.0","model":{"type":"BPE","merges":["a b c"]}}',
        )
        self.assertEqual(
            train_sft._tokenizer_semantics(canonical, "canonical tokenizer"),
            train_sft._tokenizer_semantics(legacy, "legacy tokenizer"),
        )
        self.assertNotEqual(
            train_sft._tokenizer_semantics(canonical, "canonical tokenizer"),
            train_sft._tokenizer_semantics(changed, "changed tokenizer"),
        )
        with self.assertRaisesRegex(train_sft.ContractError, "exactly two"):
            train_sft._tokenizer_semantics(malformed, "malformed tokenizer")

    def test_base_tokenizer_semantic_drift_fails_full_preflight(self) -> None:
        root, manifest = train_sft.load_export_bundle(self.fixture.bundle)
        write(self.fixture.base_model / "tokenizer.json", b'{"version":"2.0"}')
        with self.assertRaisesRegex(train_sft.ContractError, "semantics differ"):
            train_sft.verify_base_model_source(
                self.fixture.base_model, root, manifest
            )

    def test_base_aggregate_is_recomputed_instead_of_trusted(self) -> None:
        self.fixture.manifest["model"]["base_weight_shard_manifest"][
            "aggregate_sha256"
        ] = "sha256:" + "f" * 64
        fields = {
            key: value
            for key, value in self.fixture.manifest.items()
            if key != "export_sha256"
        }
        self.fixture.manifest["export_sha256"] = train_sft._canonical_sha256(fields)
        self.fixture.write_manifest()
        with self.assertRaisesRegex(train_sft.ContractError, "base-weight aggregate differs"):
            train_sft.load_export_bundle(self.fixture.bundle)

    def test_result_is_published_last_and_is_self_verifying(self) -> None:
        root, manifest = train_sft.load_export_bundle(self.fixture.bundle)
        adapter = Path(self.temporary.name) / "adapter"
        adapter.mkdir()
        write(adapter / "adapter_config.json", b'{"peft_type":"LORA"}')
        write(adapter / "adapter_model.safetensors", b"fake adapter tensors")
        versions = dict(train_sft.PINNED_PACKAGES)
        with mock.patch.object(
            train_sft.importlib.metadata,
            "version",
            side_effect=lambda name: versions[name],
        ):
            train_sft._publish_result(
                root,
                manifest,
                adapter,
                {"seed": {"kind": "unsigned", "value": 42}},
            )

        result = train_sft._read_json(root / "kiln_hf_result.json", bounded=True)
        digest_fields = dict(result)
        actual = digest_fields.pop("result_sha256")
        self.assertEqual(actual, train_sft._canonical_sha256(digest_fields))
        self.assertFalse((root / train_sft.RESULT_SENTINEL).exists())
        self.assertEqual(
            train_sft._sha256_file(root / "executed_train.py"),
            manifest["reference_script"]["sha256"],
        )
        with self.assertRaisesRegex(train_sft.ContractError, "already contains"):
            train_sft._recover_or_reject_result_state(root)

    def test_effective_config_matches_rust_bounds(self) -> None:
        train_sft._validate_effective_config(
            {
                "learning_rate": {"kind": "decimal", "value": "2e-5"},
                "seed": {"kind": "unsigned", "value": 42},
            }
        )
        with self.assertRaisesRegex(train_sft.ContractError, "bounded text"):
            train_sft._validate_effective_config(
                {"name": {"kind": "text", "value": " trailing "}}
            )
        with self.assertRaisesRegex(train_sft.ContractError, "invalid"):
            train_sft._validate_effective_config(
                {"seed": {"kind": "unsigned", "value": -1}}
            )

    def test_reference_target_modules_are_exactly_kiln_loadable(self) -> None:
        expected = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.assertEqual(list(train_sft.KILN_TARGET_MODULES), expected)
        self.assertEqual(
            train_sft._target_modules(None),
            expected,
        )
        self.assertEqual(
            train_sft._target_modules("q_proj, down_proj"),
            ["q_proj", "down_proj"],
        )
        for invalid in ("all-linear", "q_proj,q_proj", "q_proj,"):
            with self.subTest(invalid=invalid), self.assertRaises(
                train_sft.ContractError
            ):
                train_sft._target_modules(invalid)

    def test_incomplete_result_is_recovered_but_unattributed_files_fail(self) -> None:
        root, _ = train_sft.load_export_bundle(self.fixture.bundle)
        write(root / train_sft.RESULT_SENTINEL, b"incomplete")
        write(root / train_sft.ADAPTER_CONFIG, b"partial")
        train_sft._recover_or_reject_result_state(root)
        self.assertFalse((root / train_sft.RESULT_SENTINEL).exists())
        self.assertFalse((root / train_sft.ADAPTER_CONFIG).exists())

        write(root / train_sft.ADAPTER_CONFIG, b"unattributed")
        with self.assertRaisesRegex(train_sft.ContractError, "unattributed"):
            train_sft._recover_or_reject_result_state(root)


class HfTrlGrpoReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = GrpoBundleFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_grpo_bundle_and_base_verify_without_training_imports(self) -> None:
        root, manifest = train_sft.load_export_bundle(self.fixture.bundle)
        self.assertEqual(manifest["task"], "grpo")
        self.assertEqual(
            train_sft.verify_base_model_source(
                self.fixture.base_model, root, manifest
            ),
            self.fixture.base_model.resolve(),
        )
        summary = train_sft._scan_grpo_dataset(
            root / "train.jsonl", manifest["model"], None
        )
        self.assertEqual(summary["completion_count"], 2)
        self.assertEqual(summary["sampled_action_tokens"], 2)
        self.assertEqual(summary["forced_action_tokens"], 1)
        self.assertEqual(summary["num_generations"], 2)

    def test_grpo_ordered_digest_matches_rust_cross_language_golden(self) -> None:
        self.assertEqual(
            train_sft._ordered_grpo_corpus_sha256([b'{"a":1}', b"[]"]),
            "sha256:3f495e234c3946d423ff971d5ade7fc11e7139a5a24cd3ea35a36cba2c6bdeb3",
        )

    def test_noncanonical_row_fails_after_all_outer_hashes_are_updated(self) -> None:
        write(
            self.fixture.bundle / "train.jsonl",
            (json.dumps(self.fixture.group) + "\n").encode("utf-8"),
        )
        self.fixture.rehash_manifest()
        with self.assertRaisesRegex(train_sft.ContractError, "not canonical compact JSON"):
            train_sft.load_export_bundle(self.fixture.bundle)

    def test_scored_payload_drift_fails_after_dataset_and_manifest_rehash(self) -> None:
        changed = copy.deepcopy(self.fixture.group)
        changed["completions"][0]["text"] = "rescored bytes"
        self.fixture.write_groups([changed])
        self.fixture.rehash_manifest()
        with self.assertRaisesRegex(train_sft.ContractError, "scored payload identity differs"):
            train_sft.load_export_bundle(self.fixture.bundle)

    def test_recorded_rollout_replays_exact_ids_logprobs_masks_and_rewards(self) -> None:
        _, manifest = train_sft.load_export_bundle(self.fixture.bundle)
        source = train_sft._RecordedRolloutSource(
            self.fixture.bundle / "train.jsonl",
            manifest["data"]["dataset"],
            manifest["model"],
            None,
            2,
        )
        prompt = train_sft._compact_json_bytes(
            self.fixture.group["messages"]
        ).decode("utf-8")
        output = source([prompt, prompt], object())
        self.assertEqual(output["prompt_ids"], [[10], [10]])
        self.assertEqual(output["completion_ids"], [[20, 21], [22]])
        self.assertEqual(output["logprobs"], [[-0.5, 0.0], [-0.25]])
        self.assertEqual(output["env_mask"], [[1, 0], [1]])
        self.assertEqual(output["recorded_reward"], [0.0, 1.0])
        self.assertEqual(
            train_sft._recorded_reward(recorded_reward=[0.0, 1.0]),
            [0.0, 1.0],
        )
        source([prompt, prompt], object())
        self.assertEqual(source.groups_served, 2)
        self.assertEqual(source.epochs_opened, 2)

    def test_recorded_rollout_rejects_source_mutation_and_prompt_reordering(self) -> None:
        _, manifest = train_sft.load_export_bundle(self.fixture.bundle)
        source = train_sft._RecordedRolloutSource(
            self.fixture.bundle / "train.jsonl",
            manifest["data"]["dataset"],
            manifest["model"],
            None,
            2,
        )
        with self.assertRaisesRegex(train_sft.ContractError, "prompt order or repetition"):
            source(["wrong", "wrong"], object())

        source = train_sft._RecordedRolloutSource(
            self.fixture.bundle / "train.jsonl",
            manifest["data"]["dataset"],
            manifest["model"],
            None,
            2,
        )
        with (self.fixture.bundle / "train.jsonl").open("ab") as handle:
            handle.write(b" ")
        with self.assertRaisesRegex(train_sft.ContractError, "size changed"):
            source(["unused", "unused"], object())

    def test_heterogeneous_completion_counts_fail_instead_of_dropping_rows(self) -> None:
        second = copy.deepcopy(self.fixture.group)
        second["completions"].append(copy.deepcopy(second["completions"][1]))
        self.fixture.write_groups([self.fixture.group, second])
        self.fixture.rehash_manifest()
        with self.assertRaisesRegex(train_sft.ContractError, "uniform completion count"):
            train_sft.load_export_bundle(self.fixture.bundle)

    def test_grpo_result_records_task_and_trainer_kind(self) -> None:
        root, manifest = train_sft.load_export_bundle(self.fixture.bundle)
        adapter = Path(self.temporary.name) / "adapter"
        adapter.mkdir()
        write(adapter / "adapter_config.json", b'{"peft_type":"LORA"}')
        write(adapter / "adapter_model.safetensors", b"fake adapter tensors")
        versions = dict(train_sft.PINNED_PACKAGES)
        with mock.patch.object(
            train_sft.importlib.metadata,
            "version",
            side_effect=lambda name: versions[name],
        ):
            train_sft._publish_result(
                root,
                manifest,
                adapter,
                {"behavior_policy": {"kind": "text", "value": "recorded"}},
            )
        result = train_sft._read_json(root / train_sft.RESULT_MANIFEST, bounded=True)
        self.assertEqual(result["task"], "grpo")
        self.assertEqual(result["trainer"]["kind"], "trl_grpo_trainer")


if __name__ == "__main__":
    unittest.main()
