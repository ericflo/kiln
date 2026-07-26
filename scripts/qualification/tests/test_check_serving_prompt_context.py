import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/qualification/check_serving_prompt_context.py"
SPEC = importlib.util.spec_from_file_location("check_serving_prompt_context", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
context = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = context
SPEC.loader.exec_module(context)


class FakeBenchmark:
    PROFILE_CONTRACTS = {
        "short": {"prompt_profile": "short"},
        "long": {"prompt_profile": "long"},
    }

    @staticmethod
    def deterministic_prompt(
        prompt_set_id: str,
        phase: str,
        request_index: int,
        prompt_profile: str,
    ) -> str:
        length = 4 if prompt_profile == "short" else 8
        return f"{prompt_set_id}:{phase}:{request_index}:" + "x" * length


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        prompt = messages[0]["content"]
        return {"input_ids": list(range(len(prompt)))}


class PromptContextTests(unittest.TestCase):
    def test_check_covers_warmup_and_every_measured_prompt(self) -> None:
        result = context.check_prompts(
            tokenizer=FakeTokenizer(),
            benchmark=FakeBenchmark(),
            prompt_set_id="shared-prompts",
            profiles=["short", "long"],
            sizes=[1, 2],
            repeats=2,
            warmup_requests=1,
            max_tokens=4,
            context_ceiling=128,
        )
        self.assertEqual(result["checked_prompt_count"], 14)
        self.assertEqual(result["profiles"], ["short", "long"])
        self.assertEqual(result["sizes"], [1, 2])
        self.assertGreater(result["minimum_headroom_tokens"], 0)

    def test_check_rejects_prompt_plus_output_overflow(self) -> None:
        with self.assertRaisesRegex(context.PromptContextError, "above context ceiling"):
            context.check_prompts(
                tokenizer=FakeTokenizer(),
                benchmark=FakeBenchmark(),
                prompt_set_id="shared-prompts",
                profiles=["long"],
                sizes=[1],
                repeats=1,
                warmup_requests=0,
                max_tokens=64,
                context_ceiling=70,
            )


if __name__ == "__main__":
    unittest.main()
