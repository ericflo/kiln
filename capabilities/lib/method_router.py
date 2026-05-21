"""
method_router.py — apply the METHODS.md decision tree.

Given an eval summary (baseline or post-stage) plus a small set of contextual
inputs, return the recommended training method for the next stage along with
the rule that fired and a human-readable rationale.

Usage:
  python3 lib/method_router.py \\
    --eval-summary /tmp/<cap>-eval-base.json \\
    [--teacher-available] \\
    [--multi-turn] \\
    [--has-verifier] \\
    [--reward-variance 0.07] \\
    [--print]

If reward-variance isn't supplied, the router prints a warning and treats
that input as "unknown" — RULE F's variance gate then cannot fire and
RULE E/G become the dominant routes. Pass --reward-variance after sampling
20 baseline rollouts and computing group variance (PIPELINE.md §4.1 input list).

Outputs (printed JSON unless --print is set, then human-readable):
  {
    "recommended_method": "sft" | "opd" | "grpo" | "agentic-grpo" | "stop",
    "rule_fired":         "A" | "B" | "C" | ... ,
    "rationale":          "...",
    "alternates":         ["..."],
    "inputs": {...}
  }

A "stop" recommendation means the pipeline should not start a new stage at
this point. The rationale explains why (rubric saturation, no headroom, etc.).
"""

import argparse
import json
import sys
from pathlib import Path


def route(
    *,
    baseline_composite: float,
    sub_scores_mean: dict,
    sub_score_weights: dict,
    teacher_available: bool,
    multi_turn: bool,
    has_verifier: bool,
    reward_variance: float | None,
    has_hard_eval_headroom: bool | None = None,
) -> dict:
    """Apply the METHODS.md §2 decision tree.

    Returns the recommendation dict described in this module's docstring.
    """
    inputs = {
        "baseline_composite": baseline_composite,
        "sub_scores_mean": sub_scores_mean,
        "sub_score_weights": sub_score_weights,
        "teacher_available": teacher_available,
        "multi_turn": multi_turn,
        "has_verifier": has_verifier,
        "reward_variance": reward_variance,
        "has_hard_eval_headroom": has_hard_eval_headroom,
    }

    # Compute headroom per sub-score
    headroom = {
        name: sub_score_weights.get(name, 0.0) * (1.0 - sub_scores_mean.get(name, 0.0))
        for name in sub_scores_mean
    }
    total_headroom = sum(headroom.values())

    # RULE B — broken baseline
    if baseline_composite > 0.95:
        return {
            "recommended_method": "stop",
            "rule_fired": "B-lax",
            "rationale": (
                f"Baseline composite is {baseline_composite:.4f} > 0.95. "
                "Rubric is too lax. Tighten sub-scores, harden eval, re-baseline "
                "BEFORE picking any training method. See METHODS.md Rule B."
            ),
            "alternates": [],
            "inputs": inputs,
        }
    if baseline_composite < 0.3:
        return {
            "recommended_method": "stop",
            "rule_fired": "B-strict",
            "rationale": (
                f"Baseline composite is {baseline_composite:.4f} < 0.3. "
                "Inspect 3-5 base responses BEFORE picking a method. Rubric may "
                "be over-strict (round-1 OPD #5 hit this). If the model is "
                "actually trying the task in a way the rubric rejects, fix the "
                "rubric first."
            ),
            "alternates": ["sft"],  # if base genuinely doesn't try the task, SFT bootstraps
            "inputs": inputs,
        }

    # RULE A — multi-turn always routes to agentic-GRPO
    if multi_turn:
        # but sub-method choice (SFT bootstrap, OPD polish) may still apply
        alternates = []
        if total_headroom > 0 and _format_headroom_fraction(headroom) > 0.30:
            alternates.append("sft")
        if (
            teacher_available
            and 0.4 <= baseline_composite <= 0.8
            and _process_headroom_fraction(headroom) > 0.30
        ):
            alternates.append("opd")
        return {
            "recommended_method": "agentic-grpo",
            "rule_fired": "A",
            "rationale": (
                "Task is multi-turn tool-calling, so the eventual training "
                "method is agentic-GRPO with ECHO. Sub-method stages may "
                "still apply: see alternates."
            ),
            "alternates": alternates,
            "inputs": inputs,
        }

    # RULE C — cold start
    outcome = sub_scores_mean.get("outcome", 1.0)
    fmt = sub_scores_mean.get("format", 1.0)
    if outcome < 0.3 and fmt < 0.5:
        return {
            "recommended_method": "sft",
            "rule_fired": "C",
            "rationale": (
                f"Baseline outcome={outcome:.2f} < 0.3 AND format={fmt:.2f} < 0.5. "
                "Model isn't trying the task. SFT bootstrap (8-64 curated examples, "
                "rank=4, alpha=8, 1 epoch) installs the format prior cheaply."
            ),
            "alternates": [],
            "inputs": inputs,
        }

    # RULE D — format-headroom dominated
    if total_headroom > 0 and _format_headroom_fraction(headroom) > 0.30:
        return {
            "recommended_method": "sft",
            "rule_fired": "D",
            "rationale": (
                f"Format headroom is {_format_headroom_fraction(headroom):.0%} of "
                f"total headroom ({total_headroom:.3f}). SFT on curated examples "
                "is the cheapest format fix; do this before any policy gradient "
                "to keep gradient signal off malformed rollouts."
            ),
            "alternates": ["opd"] if teacher_available else [],
            "inputs": inputs,
        }

    # RULE E — distribution gap to a stronger teacher
    if (
        teacher_available
        and 0.4 <= baseline_composite <= 0.8
        and _process_or_format_headroom_fraction(headroom) > 0.30
    ):
        # CAVEAT: high-baseline OPD failure (cap #5)
        if baseline_composite > 0.80:
            return {
                "recommended_method": "sft",
                "rule_fired": "E-rescue",
                "rationale": (
                    f"Baseline composite {baseline_composite:.4f} > 0.80 with mixed "
                    "rollout quality risks the cap #5 high-baseline OPD failure mode. "
                    "Sample teacher rollouts and SFT on those instead of OPD on "
                    "raw student samples. See METHODS.md §4.3."
                ),
                "alternates": ["opd"],
                "inputs": inputs,
            }
        return {
            "recommended_method": "opd",
            "rule_fired": "E",
            "rationale": (
                f"Baseline {baseline_composite:.4f} ∈ [0.4, 0.8], teacher "
                "available, headroom concentrated in process/format. "
                "OPD against the teacher is the cheapest gradient signal."
            ),
            "alternates": ["grpo"] if has_verifier else [],
            "inputs": inputs,
        }

    # RULE F — verifier + reward variance
    if (
        has_verifier
        and 0.6 <= baseline_composite <= 0.9
        and reward_variance is not None
        and reward_variance > 0.05
    ):
        return {
            "recommended_method": "grpo",
            "rule_fired": "F",
            "rationale": (
                f"Baseline {baseline_composite:.4f} ∈ [0.6, 0.9], verifier present, "
                f"reward variance {reward_variance:.3f} > 0.05. Use GRPO with "
                "--filter-var-min 0.05. Check rubric §0 for reward-function failure modes."
            ),
            "alternates": ["opd"] if teacher_available else [],
            "inputs": inputs,
        }

    # RULE G — saturated reward with residual hard-tail
    if (
        baseline_composite > 0.85
        and (reward_variance is None or reward_variance < 0.03)
    ):
        if has_hard_eval_headroom:
            return {
                "recommended_method": "grpo",
                "rule_fired": "G-hard-eval",
                "rationale": (
                    f"Baseline {baseline_composite:.4f} > 0.85 with low reward "
                    "variance, BUT hard_eval.tasks.jsonl pool has residual headroom. "
                    "Switch eval to hard_eval and run GRPO --no-policy-loss "
                    "(ECHO-only) to avoid the policy-gradient harm vector on "
                    "saturated reward (see CONSOLIDATED_REPORT pi-diff-patch-apply)."
                ),
                "alternates": [],
                "inputs": inputs,
            }
        return {
            "recommended_method": "stop",
            "rule_fired": "G-saturated",
            "rationale": (
                f"Baseline {baseline_composite:.4f} > 0.85, reward variance is low, "
                "no hard_eval headroom available. Policy gradient on saturated "
                "reward harms the model (see CONSOLIDATED_REPORT pi-diff-patch-apply). "
                "Either build hard_eval.tasks.jsonl and re-route, or accept the "
                "ceiling and ship base."
            ),
            "alternates": [],
            "inputs": inputs,
        }

    # RULE H — closeout / no clear method
    return {
        "recommended_method": "stop",
        "rule_fired": "H",
        "rationale": (
            f"No rule fired cleanly. Baseline {baseline_composite:.4f}, "
            f"total headroom {total_headroom:.3f}. Inspect headroom distribution "
            "(lib/headroom.py) and decide manually, or harden the eval set to "
            "open a clearer signal."
        ),
        "alternates": [],
        "inputs": inputs,
    }


def _format_headroom_fraction(headroom: dict) -> float:
    total = sum(headroom.values()) or 1.0
    fmt = headroom.get("format", 0.0)
    return fmt / total


def _process_headroom_fraction(headroom: dict) -> float:
    total = sum(headroom.values()) or 1.0
    process = sum(
        v for k, v in headroom.items() if k.startswith("process") or k in ("faithfulness", "verify")
    )
    return process / total


def _process_or_format_headroom_fraction(headroom: dict) -> float:
    total = sum(headroom.values()) or 1.0
    pf = sum(
        v
        for k, v in headroom.items()
        if k.startswith("process") or k == "format" or k in ("faithfulness", "verify")
    )
    return pf / total


def _load_eval_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--eval-summary", required=True, type=Path)
    ap.add_argument("--teacher-available", action="store_true")
    ap.add_argument("--multi-turn", action="store_true")
    ap.add_argument("--has-verifier", action="store_true", default=True)
    ap.add_argument("--no-verifier", dest="has_verifier", action="store_false")
    ap.add_argument("--reward-variance", type=float, default=None)
    ap.add_argument("--has-hard-eval-headroom", action="store_true", default=None)
    ap.add_argument(
        "--sub-score-weights",
        type=str,
        default=None,
        help="JSON dict of sub-score weights; defaults to equal weights "
        "across present sub-scores",
    )
    ap.add_argument("--print", action="store_true", help="Human-readable output")

    args = ap.parse_args()
    eval_summary = _load_eval_summary(args.eval_summary)

    sub_scores = eval_summary.get("sub_scores_mean") or eval_summary.get("sub_scores") or {}
    if args.sub_score_weights:
        weights = json.loads(args.sub_score_weights)
    else:
        n = len(sub_scores) or 1
        weights = {k: 1.0 / n for k in sub_scores}

    rec = route(
        baseline_composite=eval_summary.get("mean_composite")
        or eval_summary.get("composite", 0.0),
        sub_scores_mean=sub_scores,
        sub_score_weights=weights,
        teacher_available=args.teacher_available,
        multi_turn=args.multi_turn,
        has_verifier=args.has_verifier,
        reward_variance=args.reward_variance,
        has_hard_eval_headroom=args.has_hard_eval_headroom,
    )

    if args.print:
        print(f"Recommended method: {rec['recommended_method']}")
        print(f"Rule fired:         {rec['rule_fired']}")
        print(f"Rationale:          {rec['rationale']}")
        if rec["alternates"]:
            print(f"Alternates:         {', '.join(rec['alternates'])}")
    else:
        print(json.dumps(rec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
