"""Shared strict JSON parsing for qualification evidence and manifests."""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Any


JSON_INTEGER_MAX_DIGITS = 4096


class StrictJSONError(ValueError):
    """A JSON value is syntactically valid but violates the strict contract."""

    def __init__(self, reason: str, value: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.value = value


def reject_constant(value: str) -> None:
    raise StrictJSONError(
        "non_finite_constant",
        value,
        f"non-finite JSON number is not allowed: {value}",
    )


def parse_finite_float(value: str) -> float:
    try:
        exact = Decimal(value)
        parsed = float(value)
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise StrictJSONError(
            "invalid_number", value, f"invalid JSON number: {value}"
        ) from exc
    if not math.isfinite(parsed):
        raise StrictJSONError(
            "float_overflow",
            value,
            f"JSON number overflows finite float range: {value}",
        )
    if parsed == 0.0:
        if exact != 0:
            raise StrictJSONError(
                "float_underflow",
                value,
                f"JSON number underflows finite float range: {value}",
            )
        return 0.0
    if Decimal(str(parsed)) != exact:
        raise StrictJSONError(
            "inexact_float",
            value,
            f"JSON number is not exactly representable: {value}",
        )
    return parsed


def parse_bounded_int(
    value: str, *, max_digits: int = JSON_INTEGER_MAX_DIGITS
) -> int:
    if len(value.lstrip("-")) > max_digits:
        raise StrictJSONError(
            "integer_too_long",
            value,
            f"JSON integer exceeds {max_digits} digits",
        )
    try:
        return int(value)
    except ValueError as exc:
        raise StrictJSONError(
            "invalid_integer", value, f"invalid JSON integer: {value}"
        ) from exc


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(
                "duplicate_key", key, f"duplicate JSON object key: {key}"
            )
        result[key] = value
    return result


def loads(
    payload: str | bytes | bytearray,
    *,
    max_integer_digits: int = JSON_INTEGER_MAX_DIGITS,
) -> Any:
    """Parse JSON while rejecting duplicate keys and lossy numeric values."""

    if isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload).decode("utf-8")
    return json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
        parse_float=parse_finite_float,
        parse_int=partial(parse_bounded_int, max_digits=max_integer_digits),
    )
