"""Cross-cap-coherence rubric: aggregates per-cap composites.

This rubric does NOT define new sub-scores. It imports each member
cap's `rubric.py` and delegates scoring per-task.

The aggregator (capability.oracle.sh) handles the cross-cap composite
and per-cap delta computation.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

RUBRIC_VERSION = "v0"


def load_member_rubric(member_path: Path):
    """Dynamically import a member cap's rubric.py."""
    p = (member_path / "rubric.py").resolve()
    if not p.exists():
        raise FileNotFoundError(f"member rubric missing: {p}")
    spec = importlib.util.spec_from_file_location(f"member_{member_path.name}_rubric", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_one(rollout: dict[str, Any]) -> dict[str, float]:
    """Delegate to the member cap's rubric.

    `rollout` MUST contain a `_member_cap` field naming which member to
    use; the caller (build_corpus.py / capability.oracle.sh) annotates
    this when constructing the integration eval set.
    """
    member = rollout.get("_member_cap")
    if not member:
        raise ValueError("integration rollout missing _member_cap annotation")
    member_path = Path(__file__).resolve().parent.parent.parent / "agentic-grpo" / member
    if not member_path.exists():
        raise FileNotFoundError(f"member cap dir not found: {member_path}")
    rubric = load_member_rubric(member_path)
    if not hasattr(rubric, "score_one"):
        # Some legacy caps expose `score_rollout(transcript, workdir, task)` instead.
        if hasattr(rubric, "score_rollout"):
            return rubric.score_rollout(
                rollout.get("transcript", []),
                rollout.get("workdir", ""),
                rollout.get("task", {}),
            )
        raise AttributeError(f"member {member} rubric exposes neither score_one nor score_rollout")
    return rubric.score_one(rollout)


# This is referenced by the integration `capability.oracle.sh` aggregator.
# It calls score_one per rollout, then assembles the per-cap and cross-cap
# composites in the shell script via Python heredoc.
