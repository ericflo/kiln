"""Composite reward function — scaffold for pi-context-aware-edits.

Round 2 contract:

  score_one(rollout: dict) -> dict[str, float]
  RUBRIC_VERSION: str
  CHEAT_PROBES: list  (for rubric_sanity.py)

`rollout` is one element of rollout.jsonl. Must return a dict with
every sub-score under its weight name plus a `composite`.

The composite SHOULD use the round-2 multiplicative-gate pattern by
default:

    composite = outcome * format * (sum(w_i * sub_i) + base)

The pass-floor (base) ensures gradient when outcome=1 and format=1.
The multiplicative gates ensure format and outcome each provide
direct movement to composite when they shift.

See ./capability.md `## Rubric (v0)` for this cap's specific weights.
"""
from __future__ import annotations
from typing import Any

RUBRIC_VERSION = "v0-scaffold"


def score_one(rollout):
    raise NotImplementedError(
        "pi-context-aware-edits: rubric.py is a scaffold. Fill in score_one() per "
        "./capability.md ## Rubric (v0) and the §0 adversarial cheats."
    )


# Calibration adversarial cases — required for rubric_sanity.py.
# Each entry is a (rollout-shape dict, expected_band) tuple where band
# is "good" or "bad". See ./capability.md `## Adversarial design (§0)`.
CHEAT_PROBES = []
