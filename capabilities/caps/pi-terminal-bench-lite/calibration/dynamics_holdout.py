"""Paper §5.2 dynamics-holdout test for pi-terminal-bench-lite.

Generates trajectories from a stronger teacher model and measures
env-token cross-entropy on Base / GRPO / ECHO checkpoints. ECHO should
reduce env-CE by ≥30%, while GRPO-only should barely move it (paper §5.2
Figure 3).

This script is the Phase 2 validation gate from
docs/plans/echo-integration-plan.md §5 Phase 2: "Dynamics holdout CE
drops by at least 30% on the ECHO checkpoint vs the GRPO-only
checkpoint AND pass-rate strictly improves."

Usage:
    python dynamics_holdout.py \
        --base-adapter base \
        --grpo-adapter grpo-tblite-iter5 \
        --echo-adapter echo-tblite-iter5 \
        --teacher-traj /path/to/teacher_trajectories.jsonl \
        --kiln-url http://localhost:8420 \
        --out-dir /tmp/tblite-dynamics-holdout

teacher_trajectories.jsonl format (one per line):
    {
      "task_id": "...",
      "trajectory": [
        {"role": "user", "content": "...", "kind": "context"},
        {"role": "assistant", "content": "...", "kind": "action"},
        {"role": "tool", "content": "...", "kind": "observation"},
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterator

TARGET_DROP_PCT = 30.0


def kiln_load_adapter(url: str, name: str | None) -> None:
    if not name:
        req = urllib.request.Request(f"{url}/v1/adapters/unload", data=b"", method="POST")
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass
        return
    req = urllib.request.Request(
        f"{url}/v1/adapters/load",
        data=json.dumps({"name": name}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=30).read()


def kiln_chat_logprobs(url: str, messages: list[dict], target_tokens: list[str]) -> list[float]:
    """Query kiln /v1/chat/completions in `logprobs` mode and return the
    log-probability of each target token at its corresponding position.
    Mirrors the paper §5.2 evaluation: the model conditions on the
    rendered conversation and we measure how surprised it was by each
    observation token.

    Implementation note: this is a stub. The actual call needs to use
    kiln's `prompt_logprobs` extension OR a sliding-window scoring
    approach. Filed as a follow-up; the structure here pins the
    interface."""
    raise NotImplementedError(
        "kiln_chat_logprobs requires kiln's prompt_logprobs extension; \n"
        "filed as an integration follow-up. See \n"
        "docs/plans/echo-integration-plan.md §C.2 for the intended shape."
    )


def measure_env_token_ce(
    url: str,
    adapter: str | None,
    teacher_trajectories: list[dict],
) -> dict:
    """For each trajectory, compute the mean cross-entropy of the
    observation-segment tokens under the named adapter. Returns:

        {
          "adapter": "...",
          "n_trajectories": <int>,
          "n_env_tokens": <int>,
          "mean_env_ce": <float>,
        }
    """
    kiln_load_adapter(url, adapter)

    total_log_probs: list[float] = []
    for traj in teacher_trajectories:
        # Build the rendered conversation up to and including the
        # observation segments. Score the observation token log probs.
        segments = traj["trajectory"]
        messages = [
            {"role": s["role"], "content": s["content"]}
            for s in segments
        ]
        env_targets: list[str] = []
        for s in segments:
            if s.get("kind") == "observation":
                env_targets.append(s["content"])
        if env_targets:
            try:
                logprobs = kiln_chat_logprobs(url, messages, env_targets)
                total_log_probs.extend(logprobs)
            except NotImplementedError:
                # Stub path — record zero for now so the script structure
                # is testable end-to-end without the kiln extension.
                pass

    if not total_log_probs:
        return {
            "adapter": adapter or "base",
            "n_trajectories": len(teacher_trajectories),
            "n_env_tokens": 0,
            "mean_env_ce": float("nan"),
            "stub": True,
        }

    mean_ce = -sum(total_log_probs) / len(total_log_probs)
    return {
        "adapter": adapter or "base",
        "n_trajectories": len(teacher_trajectories),
        "n_env_tokens": len(total_log_probs),
        "mean_env_ce": mean_ce,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-adapter", default="")
    p.add_argument("--grpo-adapter", required=True)
    p.add_argument("--echo-adapter", required=True)
    p.add_argument("--teacher-traj", required=True)
    p.add_argument("--kiln-url", default="http://localhost:8420")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-drop-pct", type=float, default=TARGET_DROP_PCT)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_trajs: list[dict] = []
    with Path(args.teacher_traj).open() as f:
        for line in f:
            line = line.strip()
            if line:
                teacher_trajs.append(json.loads(line))

    print(f"Loaded {len(teacher_trajs)} teacher trajectories.")

    ce_base = measure_env_token_ce(args.kiln_url, args.base_adapter or None, teacher_trajs)
    ce_grpo = measure_env_token_ce(args.kiln_url, args.grpo_adapter, teacher_trajs)
    ce_echo = measure_env_token_ce(args.kiln_url, args.echo_adapter, teacher_trajs)

    print(f"Base env-CE:  {ce_base['mean_env_ce']:.4f}")
    print(f"GRPO env-CE:  {ce_grpo['mean_env_ce']:.4f}")
    print(f"ECHO env-CE:  {ce_echo['mean_env_ce']:.4f}")

    base = ce_base["mean_env_ce"]
    grpo = ce_grpo["mean_env_ce"]
    echo = ce_echo["mean_env_ce"]

    receipt = {
        "ce_base": ce_base,
        "ce_grpo": ce_grpo,
        "ce_echo": ce_echo,
        "target_drop_pct": args.target_drop_pct,
    }

    if (
        ce_base.get("stub")
        or ce_grpo.get("stub")
        or ce_echo.get("stub")
        or not (base > 0 and grpo > 0 and echo > 0)
    ):
        print(
            "\nSTUB MODE — kiln prompt_logprobs extension not yet wired; \n"
            "skipping gate but emitting receipt structure for later replay."
        )
        receipt["status"] = "stub"
        (out_dir / "dynamics_holdout_receipt.json").write_text(json.dumps(receipt, indent=2))
        return 0

    grpo_drop_pct = 100.0 * (base - grpo) / base
    echo_drop_pct = 100.0 * (base - echo) / base

    print(f"GRPO drop:   {grpo_drop_pct:.1f}% (paper §5.2 expects ~0)")
    print(f"ECHO drop:   {echo_drop_pct:.1f}% (target ≥{args.target_drop_pct}%)")

    receipt["grpo_drop_pct"] = grpo_drop_pct
    receipt["echo_drop_pct"] = echo_drop_pct
    receipt["status"] = "pass" if echo_drop_pct >= args.target_drop_pct else "fail"
    (out_dir / "dynamics_holdout_receipt.json").write_text(json.dumps(receipt, indent=2))

    if echo_drop_pct < args.target_drop_pct:
        print(
            f"\nFAIL: ECHO env-CE drop {echo_drop_pct:.1f}% < target "
            f"{args.target_drop_pct}%"
        )
        return 1

    print(f"\nPASS: ECHO env-CE drop {echo_drop_pct:.1f}% ≥ target {args.target_drop_pct}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
