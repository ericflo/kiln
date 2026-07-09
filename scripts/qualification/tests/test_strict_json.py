from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualification_strict_json", QUALIFICATION_DIR / "strict_json.py"
)
assert SPEC is not None and SPEC.loader is not None
strict_json = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = strict_json
SPEC.loader.exec_module(strict_json)


class StrictJSONTests(unittest.TestCase):
    def test_valid_nested_document_preserves_exact_values(self) -> None:
        value = strict_json.loads(
            b'{"nested":{"enabled":true,"count":9007199254740993,"ratio":0.5}}'
        )
        self.assertEqual(value["nested"]["count"], 9007199254740993)
        self.assertEqual(value["nested"]["ratio"], 0.5)
        self.assertIs(value["nested"]["enabled"], True)

    def test_zero_is_canonicalized_to_positive_float_zero(self) -> None:
        values = strict_json.loads('[0.0,-0.0]')
        self.assertEqual(values, [0.0, 0.0])
        self.assertTrue(all(math.copysign(1.0, value) == 1.0 for value in values))

    def test_duplicate_keys_are_rejected_at_every_depth(self) -> None:
        with self.assertRaisesRegex(
            strict_json.StrictJSONError, "duplicate JSON object key: value"
        ) as caught:
            strict_json.loads('{"nested":{"value":1,"\\u0076alue":2}}')
        self.assertEqual(caught.exception.reason, "duplicate_key")
        self.assertEqual(caught.exception.value, "value")

    def test_non_finite_and_lossy_floats_have_stable_errors(self) -> None:
        cases = {
            "NaN": "non-finite JSON number is not allowed: NaN",
            "Infinity": "non-finite JSON number is not allowed: Infinity",
            "-Infinity": "non-finite JSON number is not allowed: -Infinity",
            "1e400": "JSON number overflows finite float range: 1e400",
            "1e-4000": "JSON number underflows finite float range: 1e-4000",
            "8.9999999999999999": (
                "JSON number is not exactly representable: 8.9999999999999999"
            ),
            "9007199254740993.0": (
                "JSON number is not exactly representable: 9007199254740993.0"
            ),
            "4e-324": "JSON number is not exactly representable: 4e-324",
        }
        for token, message in cases.items():
            with self.subTest(token=token):
                with self.assertRaises(strict_json.StrictJSONError) as caught:
                    strict_json.loads('{"value":' + token + "}")
                self.assertEqual(str(caught.exception), message)

    def test_invalid_direct_numeric_tokens_have_stable_errors(self) -> None:
        with self.assertRaisesRegex(
            strict_json.StrictJSONError, "invalid JSON number: not-a-number"
        ):
            strict_json.parse_finite_float("not-a-number")
        with self.assertRaisesRegex(
            strict_json.StrictJSONError, "invalid JSON integer: 1.0"
        ):
            strict_json.parse_bounded_int("1.0")

    def test_integer_digit_limit_is_exact(self) -> None:
        accepted = "1" + "0" * (strict_json.JSON_INTEGER_MAX_DIGITS - 1)
        rejected = "1" + "0" * strict_json.JSON_INTEGER_MAX_DIGITS
        for sign in ("", "-"):
            with self.subTest(sign=sign or "positive"):
                self.assertEqual(strict_json.loads(sign + accepted), int(sign + accepted))
                with self.assertRaises(strict_json.StrictJSONError) as caught:
                    strict_json.loads(sign + rejected)
                self.assertEqual(
                    str(caught.exception),
                    f"JSON integer exceeds {strict_json.JSON_INTEGER_MAX_DIGITS} digits",
                )

    def test_json_syntax_errors_remain_json_decode_errors(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            strict_json.loads('{"value":}')

    def test_bytes_require_plain_utf8_without_a_bom(self) -> None:
        with self.assertRaises(UnicodeDecodeError):
            strict_json.loads('{"value":1}'.encode("utf-16"))
        with self.assertRaises(json.JSONDecodeError):
            strict_json.loads(b"\xef\xbb\xbf{\"value\":1}")


if __name__ == "__main__":
    unittest.main()
