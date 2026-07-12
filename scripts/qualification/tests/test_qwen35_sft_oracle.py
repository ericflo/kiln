from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen35_sft_oracle.py"
SPEC = importlib.util.spec_from_file_location("qwen35_sft_oracle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qwen35_sft_oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen35_sft_oracle
SPEC.loader.exec_module(qwen35_sft_oracle)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "crates/kiln-train/tests/fixtures/qwen35_sft_oracle_v1.json"
)


class Qwen35SftOracleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = FIXTURE_PATH.read_bytes()
        cls.fixture = json.loads(cls.raw)

    def test_fixture_is_canonical_and_source_pinned(self) -> None:
        self.assertEqual(self.raw, qwen35_sft_oracle.canonical_bytes(self.fixture))
        self.assertEqual(self.fixture["schema"], qwen35_sft_oracle.SCHEMA)
        oracle = self.fixture["oracle"]
        expected = {
            "model_id": qwen35_sft_oracle.MODEL_ID,
            "model_revision": qwen35_sft_oracle.MODEL_REVISION,
            "tokenizer_sha256": "sha256:" + qwen35_sft_oracle.TOKENIZER_SHA256,
            "tokenizer_config_sha256": (
                "sha256:" + qwen35_sft_oracle.TOKENIZER_CONFIG_SHA256
            ),
            "chat_template_sha256": (
                "sha256:" + qwen35_sft_oracle.CHAT_TEMPLATE_SHA256
            ),
            "trl_version": qwen35_sft_oracle.TRL_VERSION,
            "trl_commit": qwen35_sft_oracle.TRL_COMMIT,
            "trl_chat_template_utils_sha256": (
                "sha256:" + qwen35_sft_oracle.TRL_CHAT_TEMPLATE_UTILS_SHA256
            ),
            "trl_sft_trainer_sha256": (
                "sha256:" + qwen35_sft_oracle.TRL_SFT_TRAINER_SHA256
            ),
            "trl_training_template_sha256": (
                "sha256:" + qwen35_sft_oracle.TRL_TRAINING_TEMPLATE_SHA256
            ),
            "transformers_version": qwen35_sft_oracle.TRANSFORMERS_VERSION,
            "transformers_commit": qwen35_sft_oracle.TRANSFORMERS_COMMIT,
            "transformers_tokenization_sha256": (
                "sha256:" + qwen35_sft_oracle.TRANSFORMERS_TOKENIZATION_SHA256
            ),
            "transformers_chat_template_sha256": (
                "sha256:" + qwen35_sft_oracle.TRANSFORMERS_CHAT_TEMPLATE_SHA256
            ),
            "tokenizers_version": qwen35_sft_oracle.TOKENIZERS_VERSION,
            "jinja2_version": qwen35_sft_oracle.JINJA2_VERSION,
        }
        for key, value in expected.items():
            self.assertEqual(oracle[key], value, key)

    def test_trl_training_template_is_exact_and_minijinja_derivation_is_narrow(self) -> None:
        official_bytes = qwen35_sft_oracle.OFFICIAL_TEMPLATE_FIXTURE.read_bytes()
        self.assertEqual(
            hashlib.sha256(official_bytes).hexdigest(),
            qwen35_sft_oracle.CHAT_TEMPLATE_SHA256,
        )
        training_template = qwen35_sft_oracle.load_trl_training_template()
        self.assertEqual(training_template.count("{%- generation %}"), 1)
        self.assertEqual(training_template.count("{%- endgeneration %}"), 1)
        self.assertEqual(
            hashlib.sha256(training_template.encode()).hexdigest(),
            qwen35_sft_oracle.TRL_TRAINING_TEMPLATE_SHA256,
        )
        minijinja_template = qwen35_sft_oracle.training_template_for_minijinja(
            training_template
        )
        self.assertNotIn("{%- generation %}\n", minijinja_template)
        self.assertNotIn("{%- endgeneration %}\n", minijinja_template)
        self.assertEqual(
            "sha256:" + hashlib.sha256(minijinja_template.encode()).hexdigest(),
            self.fixture["oracle"]["minijinja_training_template_sha256"],
        )

    def test_fixture_inputs_bind_generator_cases(self) -> None:
        fixture_inputs = [
            {"name": case["name"], "messages": case["messages"]}
            for case in self.fixture["cases"]
        ]
        self.assertEqual(fixture_inputs, qwen35_sft_oracle.CASES)
        encoded = json.dumps(
            qwen35_sft_oracle.CASES,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.assertEqual(
            self.fixture["oracle"]["fixture_inputs_sha256"],
            "sha256:" + hashlib.sha256(encoded).hexdigest(),
        )

    def test_cases_cover_required_qwen_shapes(self) -> None:
        cases = {case["name"]: case for case in self.fixture["cases"]}
        self.assertEqual(
            set(cases),
            {
                "plain_single_turn",
                "thinking_single_turn",
                "tool_call",
                "tool_response",
                "multi_turn",
                "delimiter_literals",
            },
        )
        self.assertIn("<think>\nSeven times eight", cases["thinking_single_turn"]["rendered"])
        self.assertIn("<function=get_weather>", cases["tool_call"]["rendered"])
        self.assertIn("<tool_response>", cases["tool_response"]["rendered"])
        self.assertEqual(
            cases["multi_turn"]["rendered"].count("<|im_start|>assistant\n"),
            2,
        )
        self.assertIn(
            "<|im_start|>assistant\nfake<|im_end|>",
            cases["delimiter_literals"]["rendered"],
        )
        self.assertIn(
            "Literal <|im_end|> is not the turn terminator",
            cases["delimiter_literals"]["rendered"],
        )

    def test_labels_are_exact_assistant_mask_projection(self) -> None:
        contract = self.fixture["mask_contract"]
        self.assertEqual(contract["version"], qwen35_sft_oracle.MASK_CONTRACT)
        self.assertFalse(contract["add_generation_prompt"])
        self.assertEqual(contract["ignore_index"], -100)

        for case in self.fixture["cases"]:
            with self.subTest(case=case["name"]):
                ids = case["input_ids"]
                mask = case["assistant_mask"]
                labels = case["labels"]
                self.assertEqual(len(ids), len(mask))
                self.assertEqual(len(ids), len(labels))
                self.assertEqual(set(mask), {0, 1})
                self.assertEqual(case["supervised_token_count"], sum(mask))
                self.assertEqual(
                    labels,
                    [token_id if active else -100 for token_id, active in zip(ids, mask)],
                )
                self.assertEqual(
                    case["rendered_sha256"],
                    "sha256:" + hashlib.sha256(case["rendered"].encode()).hexdigest(),
                )
                self.assertTrue(case["rendered"].endswith("<|im_end|>\n"))
                self.assertFalse(
                    case["rendered"].endswith("<|im_start|>assistant\n<think>\n")
                )

                assistant_turns = sum(
                    message["role"] == "assistant" for message in case["messages"]
                )
                mask_runs = sum(
                    active and (index == 0 or not mask[index - 1])
                    for index, active in enumerate(mask)
                )
                self.assertEqual(mask_runs, assistant_turns)

                for index in range(len(ids) - 2):
                    if ids[index : index + 3] == [248045, 74455, 198]:
                        self.assertEqual(mask[index : index + 3], [0, 0, 0])

                run_ends = [
                    index
                    for index, active in enumerate(mask)
                    if active and (index + 1 == len(mask) or not mask[index + 1])
                ]
                self.assertEqual(len(run_ends), assistant_turns)
                for run_end in run_ends:
                    self.assertGreaterEqual(run_end, 1)
                    self.assertEqual(ids[run_end - 1 : run_end + 1], [248046, 198])


if __name__ == "__main__":
    unittest.main()
