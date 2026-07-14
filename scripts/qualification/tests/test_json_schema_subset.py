from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "json_schema_subset.py"
SPEC = importlib.util.spec_from_file_location("json_schema_subset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
schema_subset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = schema_subset
SPEC.loader.exec_module(schema_subset)


def errors(value: object, schema: dict, root: dict | None = None) -> list[str]:
    document = schema if root is None else root
    return schema_subset.validate_instance(value, schema, document)


class JsonSchemaSubsetTests(unittest.TestCase):
    def test_scalar_types_reject_boolean_integers_and_nonfinite_numbers(self) -> None:
        self.assertEqual(errors(3, {"type": "integer"}), [])
        self.assertTrue(errors(True, {"type": "integer"}))
        self.assertTrue(errors(math.inf, {"type": "number"}))
        self.assertEqual(errors(None, {"type": ["string", "null"]}), [])

    def test_internal_and_external_references_keep_their_document_root(self) -> None:
        external = {
            "$id": "external.json",
            "$defs": {
                "value": {"$ref": "#/$defs/nonempty"},
                "nonempty": {"type": "string", "minLength": 1},
            },
        }
        root = {"$defs": {"local": {"type": "integer", "minimum": 1}}}
        self.assertEqual(
            schema_subset.validate_instance(
                "x",
                {"$ref": "external.json#/$defs/value"},
                root,
                registry={"external.json": external},
            ),
            [],
        )
        self.assertTrue(
            schema_subset.validate_instance(
                "",
                {"$ref": "external.json#/$defs/value"},
                root,
                registry={"external.json": external},
            )
        )
        self.assertEqual(errors(1, {"$ref": "#/$defs/local"}, root), [])

    def test_unregistered_and_unresolved_references_fail_closed(self) -> None:
        root = {"$defs": {}}
        self.assertRegex(
            errors(1, {"$ref": "missing.json#/$defs/value"}, root)[0],
            "unregistered",
        )
        self.assertRegex(errors(1, {"$ref": "#/$defs/value"}, root)[0], "unresolved")

    def test_boolean_composition_and_conditionals_are_combined(self) -> None:
        schema = {
            "type": "object",
            "required": ["kind", "value"],
            "properties": {
                "kind": {"enum": ["small", "large"]},
                "value": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
            },
            "allOf": [{"not": {"properties": {"value": {"const": "forbidden"}}}}],
            "if": {"properties": {"kind": {"const": "small"}}},
            "then": {"properties": {"value": {"type": "integer", "maximum": 3}}},
            "else": {"properties": {"value": {"type": "integer", "minimum": 4}}},
        }
        self.assertEqual(errors({"kind": "small", "value": 3}, schema), [])
        self.assertTrue(errors({"kind": "small", "value": 4}, schema))
        self.assertTrue(errors({"kind": "large", "value": 3}, schema))
        self.assertTrue(errors({"kind": "large", "value": "forbidden"}, schema))

    def test_any_of_does_not_skip_sibling_object_validation(self) -> None:
        schema = {
            "type": "object",
            "anyOf": [{"required": ["left"]}, {"required": ["right"]}],
            "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
            "additionalProperties": False,
        }
        self.assertEqual(errors({"left": 1}, schema), [])
        self.assertTrue(errors({"left": "wrong"}, schema))
        self.assertTrue(errors({"other": 1}, schema))

    def test_object_contracts_cover_names_dependencies_and_additional_values(self) -> None:
        schema = {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 2,
            "propertyNames": {"pattern": "^[a-z]+$"},
            "properties": {"primary": {"type": "integer"}},
            "additionalProperties": {"type": "string"},
            "dependentRequired": {"primary": ["label"]},
        }
        self.assertEqual(errors({"primary": 1, "label": "ok"}, schema), [])
        self.assertTrue(errors({}, schema))
        self.assertTrue(errors({"primary": 1}, schema))
        self.assertTrue(errors({"Bad": "value"}, schema))
        self.assertTrue(errors({"one": "1", "two": "2", "three": "3"}, schema))

    def test_array_and_scalar_bounds_are_enforced(self) -> None:
        array = {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "uniqueItems": True,
            "items": {"type": "integer", "minimum": 1, "maximum": 3},
        }
        self.assertEqual(errors([1, 3], array), [])
        self.assertTrue(errors([], array))
        self.assertTrue(errors([1, 1], array))
        self.assertTrue(errors([4], array))
        self.assertTrue(errors([1, 2, 3], array))
        self.assertTrue(errors("x", {"type": "string", "minLength": 2}))
        self.assertTrue(errors(4, {"type": "integer", "multipleOf": 3}))

    def test_kiln_order_and_compatibility_extensions_remain_executable(self) -> None:
        ordered = {
            "type": "object",
            "x-kiln-dependent-order": [
                {"less-or-equal": "queued", "greater-or-equal": "tracked"}
            ],
        }
        self.assertEqual(errors({"queued": 2, "tracked": 3}, ordered), [])
        self.assertTrue(errors({"queued": 3, "tracked": 2}, ordered))

        toggle = {
            "type": "object",
            "x-kiln-compatible-mode-toggle": {
                "mode": "mode",
                "toggle": "enabled",
                "true-value": "enabled",
                "false-value": "disabled",
            },
        }
        self.assertEqual(errors({"mode": "enabled", "enabled": True}, toggle), [])
        self.assertTrue(errors({"mode": "disabled", "enabled": True}, toggle))


if __name__ == "__main__":
    unittest.main()
