"""Composite reward function — scaffold for python-algo.

Round 2 contract:

  score_one(rollout: dict) -> dict[str, float]
  RUBRIC_VERSION: str

`rollout` is one element of `rollout.jsonl` (or one entry of an OPD/SFT eval
case). Must return a dict containing every sub-score under its weight name
plus a `composite` field.

See `../pi-doctest/rubric.py` for the established multi-component pattern
(outcome * (w1*sub1 + w2*sub2 + ... + base)).

This file is a SCAFFOLD — the next agent picking this cap up needs to fill in
the real sub-score computations from the rubric defined in capability.md.
"""
from __future__ import annotations
from typing import Any

RUBRIC_VERSION = "v0-scaffold"


def score_one(rollout):
    raise NotImplementedError(
        "python-algo: rubric.py is a scaffold. Fill in score_one() per the rubric "
        "in ./capability.md and the cheat resistance bullets in section 0."
    )


CHEAT_PROBES = []
