#!/usr/bin/env python3
"""Dispatch every retained specialized oracle result to its closed validator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import rocm_hf_next_token_oracle as hf_next_token
import rocm_hf_layer_attribution as layer_attribution
import rocm_hf_path_attribution as path_attribution
from strict_json import loads as strict_json_loads


class OracleResultError(RuntimeError):
    """A retained result cannot be identified or validated."""


Validator = Callable[..., dict[str, Any]]
VALIDATORS: dict[str, Validator] = {
    hf_next_token.SCHEMA: hf_next_token.validate_result,
    layer_attribution.SCHEMA: layer_attribution.validate_result,
    path_attribution.SCHEMA: path_attribution.validate_result,
}


def schema_for(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OracleResultError(f"result is not a non-symlink regular file: {path}")
    try:
        value = strict_json_loads(path.read_bytes())
    except Exception as exc:
        raise OracleResultError(f"result is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("schema"), str):
        raise OracleResultError(f"result does not declare a string schema: {path}")
    return value["schema"]


def validate(path: Path, *, require_current_source: bool = False) -> dict[str, Any]:
    schema = schema_for(path)
    validator = VALIDATORS.get(schema)
    if validator is None:
        raise OracleResultError(f"unsupported oracle-result schema {schema!r}: {path}")
    return validator(path, require_current_source=require_current_source)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="+", type=Path)
    parser.add_argument("--require-current-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for path in args.result:
        try:
            value = validate(path, require_current_source=args.require_current_source)
        except BaseException as exc:
            print(f"oracle result is invalid: {path}: {exc}", file=sys.stderr)
            return 1
        print(f"OK {path} {value['schema']} {value['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
