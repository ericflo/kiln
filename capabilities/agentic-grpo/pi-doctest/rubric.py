"""Reward function for pi-doctest — v1, multi-component.

v0 (rubric.py) used outcome alone. The Phase 0 baseline finding (see
capability.jsonl iter 0) showed `outcome` saturates at ~1.0 for the
4B base model even when the agentic process is wasteful (e.g.
task_0011 ran 16 bash calls including 7 redundant `python -m doctest`
invocations before timing out at 120s, but `solution.py` ended up
correct — so outcome=1.0).

v1 keeps `outcome` as a hard-floor (we don't reward incorrect
solutions) but moves most of the weight onto agentic sub-scores:

  outcome:                0.40   (hard floor — required for any signal)
  tool_call_efficiency:   0.30   (TARGET — clear signal across wall_clock variance)
  tested_before_done:     0.20   (probably partially saturated, still useful)
  format_compliance:      0.10   (saturated 1.0 in practice)

Composite = sum(weight × sub_score).

Importantly: composite IS multiplied by outcome, not summed with it.
A solution that doesn't pass doctests gets composite=0 regardless of
how efficient the agentic process was. This is the "no reward hacking
via empty solution" guard the §0 adversarial review demanded.

Effective composite = outcome × (0.30·tool_call_efficiency
                                 + 0.20·tested_before_done
                                 + 0.10·format_compliance
                                 + 0.40)

Range: [0, 1]. When outcome=1.0 and all sub-scores=1.0, composite=1.0.
When outcome=0.0, composite=0.0.
"""

import json
import subprocess
import sys
from pathlib import Path

# Re-export the helpers from the v0 module (now archived as
# rubric_v0_outcome_only.py).
sys.path.insert(0, str(Path(__file__).parent))
from rubric_v0_outcome_only import (  # type: ignore
    _iter_messages,
    _tool_calls_in,
    _tested_before_done,
    _tool_call_efficiency,
    _format_compliance,
)


def score_rollout(transcript: list, workdir: str, task: dict) -> dict:
    # Outcome: re-runs doctest on the final workdir state.
    solution = Path(workdir) / "solution.py"
    if not solution.exists():
        return {
            "outcome": 0.0,
            "tool_call_efficiency": 0.0,
            "tested_before_done": 0.0,
            "format_compliance": 0.0,
            "composite": 0.0,
            "_reason": "no solution.py",
        }
    try:
        proc = subprocess.run(
            ["python3", "-I", "-m", "doctest", "-v", str(solution)],
            capture_output=True, text=True, timeout=10,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        outcome_val = 0.0
        proc = None
    else:
        tried = (proc.stdout or "").count("Trying:")
        failed = (proc.stdout or "").count("Failed example:")
        outcome_val = max(0.0, (tried - failed) / tried) if tried > 0 else 0.0

    tbd = _tested_before_done(transcript)
    tce = _tool_call_efficiency(transcript, expected=4)
    fc = _format_compliance(transcript)

    agentic = 0.30 * tce + 0.20 * tbd + 0.10 * fc + 0.40
    composite = outcome_val * agentic

    return {
        "outcome": outcome_val,
        "tool_call_efficiency": tce,
        "tested_before_done": tbd,
        "format_compliance": fc,
        "composite": composite,
        "_n_tool_calls": sum(
            len(_tool_calls_in(m))
            for _, m in _iter_messages(transcript)
            if m.get("role") == "assistant"
        ),
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: rubric_v1.py <transcript.jsonl> <workdir> <task.json>",
              file=sys.stderr)
        sys.exit(2)
    transcript = []
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                transcript.append(json.loads(line))
            except Exception:
                pass
    task = json.loads(Path(sys.argv[3]).read_text())
    out = score_rollout(transcript, sys.argv[2], task)
    print(json.dumps(out, indent=2))
