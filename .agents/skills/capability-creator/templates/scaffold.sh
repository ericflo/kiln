#!/usr/bin/env bash
# scaffold.sh — create a new capability dir under capabilities/caps/<slug>/.
#
# Usage:
#   bash $SKILL/templates/scaffold.sh <slug> [--from-template <existing-cap>]
#
# Creates the canonical round-3 layout from LAYOUT.md. Optionally copies
# rubric.py / build_corpus.py shape from an existing cap as a starting point.
#
# Does NOT populate calibration/ — that's the agent's first job.

set -euo pipefail

SLUG="${1:-}"
if [[ -z "$SLUG" ]]; then
  echo "usage: scaffold.sh <slug> [--from-template <existing-cap>]" >&2
  exit 2
fi
shift

FROM_TEMPLATE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-template) FROM_TEMPLATE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CAP_DIR="$REPO_ROOT/capabilities/caps/$SLUG"

if [[ -e "$CAP_DIR" ]]; then
  echo "error: $CAP_DIR already exists" >&2
  exit 1
fi

mkdir -p "$CAP_DIR"/{calibration,datasets,stages,methods,hypotheses,manifest,archive}

# capability.md skeleton
cat > "$CAP_DIR/capability.md" <<EOF
# Capability: $SLUG

## Description

(One paragraph plain-English description. What the model has to do; concrete
failure modes you want training to fix.)

## Base model

Qwen3.5-4B served by kiln on http://localhost:8420.

## Rollout source

(pi / direct HTTP / teacher. How trajectories are gathered. Specify pi
config if agentic.)

## Rubric (v1)

(Sub-score table with weights. Composite formula. Cheat resistance design.)

## Adversarial design (§0)

(MANDATORY before rubric.py. Name ≥3 cheats that would score 1.0 without
doing the capability. Design mitigations.)

1. Cheat: ...
   Mitigation: ...

2. Cheat: ...
   Mitigation: ...

3. Cheat: ...
   Mitigation: ...

## Baseline + Headroom

(Filled after iter 0 baseline eval.)

- Baseline composite: TBD
- Total headroom: TBD
- Dominant sub-score: TBD
- Headroom concentration: TBD

## Hypotheses

(H1, H2, … one-sentence claim per iter. Verdicts go in capability.jsonl.)

- H1: TBD

## Standard workflow

\`\`\`bash
./run_stage.sh <method> stage-1-<slug>
\`\`\`

See [\`capabilities/PIPELINE.md\`](../../PIPELINE.md) for stage discipline,
[\`capabilities/METHODS.md\`](../../METHODS.md) for method choice.

## Kiln features used

(Filled as pipeline develops: verify, eval-adapter, dry-run, filter-var-min, etc.)
EOF

# capability.config.json — schema 3, all four methods stubbed
cat > "$CAP_DIR/capability.config.json" <<EOF
{
  "schema_version": 3,
  "shared": {
    "base_model_path": "/workspace/Qwen3.5-4B",
    "kiln_url": "http://localhost:8420",
    "adapter_dir": "/workspace/adapters",
    "sandbox_root": "/tmp/$SLUG-stages",
    "eval": {
      "seeds": 3,
      "max_tasks": null,
      "thinking_mode": "off"
    }
  },
  "methods": {
    "sft": {
      "trainer": "cuda_sft_file",
      "data_file": "datasets/sft.train.jsonl",
      "defaults": {
        "rank": 4, "alpha": 8, "lr": 1e-4, "epochs": 1,
        "dataset_cap": 128, "seed": 3141592653,
        "adapter_smoke_test": true
      }
    },
    "opd": {
      "trainer": "cuda_opd_remote",
      "prompts_file": "datasets/opd.prompts.jsonl",
      "teacher_url": "http://localhost:8002",
      "teacher_name": "qwen3.6-27b-awq",
      "defaults": {
        "rank": 16, "alpha": 32, "lr": 1e-4, "epochs": 6,
        "samples_per_prompt": 2, "seed": 3141592653,
        "adapter_smoke_test": true
      }
    },
    "grpo": {
      "trainer": "cuda_grpo_ablation",
      "data_file": "datasets/grpo.tasks.jsonl",
      "defaults": {
        "mode": "phase1", "advantage_mode": "dr_grpo",
        "loss_aggregation": "token_level", "kl_estimator": "k1",
        "kl_coeff": 0.1, "clip_epsilon": 0.20, "dynamic_sampling": true,
        "is_level": "token", "reference_policy": "base_per_step",
        "lr": 1e-5, "rank": 16, "alpha": 32, "seed": 3141592653,
        "filter_var_min": 0.05, "adapter_smoke_test": true
      }
    },
    "agentic-grpo": {
      "trainer": "cuda_grpo_ablation",
      "data_file": "datasets/grpo.tasks.jsonl",
      "rollout": {
        "pi_bin": "/usr/bin/pi",
        "pi_model_id": "Qwen3.5-4B",
        "num_generations_train": 4,
        "num_generations_eval": 1,
        "max_wall_clock_s": 120,
        "max_turns": 8,
        "max_tokens_per_turn": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "parallel": 1
      },
      "defaults": {
        "mode": "phase1", "advantage_mode": "dr_grpo",
        "loss_aggregation": "token_level", "kl_estimator": "k1",
        "kl_coeff": 0.1, "clip_epsilon": 0.20, "dynamic_sampling": true,
        "lr": 1e-5, "rank": 16, "alpha": 32, "seed": 3141592653,
        "filter_var_min": 0.05,
        "loss": {
          "echo": {"lambda": 0.05, "env_mask_mode": "env_only", "warning_filter": true},
          "no_policy_loss": false
        },
        "adapter_smoke_test": true
      }
    }
  },
  "pipeline": {
    "max_stages": 5,
    "between_stages": {
      "run_sibling_check": true,
      "stop_on_sibling_regression": true,
      "sibling_threshold": -0.02,
      "preserve_prior_stage_threshold": -0.02
    },
    "transitions": {
      "criterion_format_floor": 0.7,
      "criterion_process_headroom_min": 0.08,
      "criterion_reward_variance_min": 0.05
    }
  }
}
EOF

# Empty capability.jsonl
: > "$CAP_DIR/capability.jsonl"

# rubric.py skeleton
cat > "$CAP_DIR/rubric.py" <<'EOF'
"""rubric.py — composite reward for this capability.

Must expose:
  - score_one(rollout: dict) -> dict[str, float]  with key "composite"
  - RUBRIC_VERSION: str
  - (optional) CHEAT_PROBES: list[Callable] for rubric_sanity.py

Importable on CPU-only dev box without network.
"""

RUBRIC_VERSION = "v0"


def score_one(rollout: dict) -> dict[str, float]:
    # TODO: implement scoring
    # Return a dict with at minimum {"composite": float in [0,1]}
    # plus per-sub-score keys.
    raise NotImplementedError("populate score_one")
EOF

# rubric_sanity.py — copies the template
cp "$(dirname "$0")/rubric_sanity.py" "$CAP_DIR/rubric_sanity.py"

# build_corpus.py skeleton
cat > "$CAP_DIR/build_corpus.py" <<'EOF'
"""build_corpus.py — task generator for this capability.

Writes:
  datasets/train.tasks.jsonl       (committed, source corpus)
  datasets/eval.tasks.jsonl        (GITIGNORED, blind-eval firewall)
  datasets/hard_eval.tasks.jsonl   (GITIGNORED, hard pool)

Lazily writes method-specific data when --method is passed:
  datasets/sft.train.jsonl   if --method sft
  datasets/opd.prompts.jsonl if --method opd
  datasets/grpo.tasks.jsonl  if --method grpo or agentic-grpo

Use a deterministic seed so the eval split is reproducible.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sft", "opd", "grpo", "agentic-grpo"], default=None)
    ap.add_argument("--seed", type=int, default=3141592653)
    args = ap.parse_args()

    # TODO: populate datasets/train.tasks.jsonl from a seed corpus
    # TODO: deterministic eval split into eval.tasks.jsonl
    # TODO: hard_eval pool from failures (built incrementally per round)
    # TODO: if --method, also write the method-specific data file


if __name__ == "__main__":
    main()
EOF

# capability.oracle.sh — blind eval wrapper
cat > "$CAP_DIR/capability.oracle.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "\$(dirname "\$0")"

ADAPTER="\${1:-}"
shift || true
TASKS="\${TASKS:-datasets/eval.tasks.jsonl}"
KILN_URL="\${KILN_URL:-http://localhost:8420}"
SEEDS="\${SEEDS:-3}"
ADAPTER_DIR="\${ADAPTER_DIR:-/workspace/adapters}"
OUTPUT="\${OUTPUT:-/tmp/$SLUG-eval-\${ADAPTER:-base}.json}"

if ! curl -sf "\$KILN_URL/v1/health" > /dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln-server not reachable at \$KILN_URL" >&2
  exit 2
fi

kiln eval-adapter \\
  --url "\$KILN_URL" \\
  --adapter "\$ADAPTER" \\
  --adapter-dir "\$ADAPTER_DIR" \\
  --tasks "\$TASKS" \\
  --seeds "\$SEEDS" \\
  --scorer "./rubric.py" \\
  --output "\$OUTPUT" \\
  --thinking off \\
  "\$@"

python3 - <<PY
import json
d = json.load(open("\$OUTPUT"))
print(f"SCORE={d['mean_composite']:.4f}")
print(f"N={d['n_tasks']}")
for k, v in d['sub_scores_mean'].items():
    print(f"{k}={v:.4f}")
PY
EOF
chmod +x "$CAP_DIR/capability.oracle.sh"

# README.md placeholder
cat > "$CAP_DIR/README.md" <<EOF
# $SLUG

Cap-specific quickstart. See [capability.md](capability.md) for the contract
and [pipeline.md] (after first stage ships) for the chain.

Standard workflow:

\`\`\`bash
python3 build_corpus.py
\$EDITOR calibration/{good,bad}.jsonl
python3 rubric_sanity.py
./capability.oracle.sh                          # baseline
python3 ../../lib/method_router.py --eval-summary /tmp/$SLUG-eval-base.json --print
./run_stage.sh <method> stage-1-<slug>
\`\`\`
EOF

# calibration READMEs and empty files
cat > "$CAP_DIR/calibration/README.md" <<'EOF'
Populate `good.jsonl` and `bad.jsonl` with ≥5 known-quality rollouts each.
Each bad rollout should exercise one of the §0 cheats from capability.md.
`rubric_sanity.py` requires margin > 0.2 between good and bad means.
EOF
: > "$CAP_DIR/calibration/good.jsonl"
: > "$CAP_DIR/calibration/bad.jsonl"

# datasets README
cat > "$CAP_DIR/datasets/hard_eval.README.md" <<'EOF'
hard_eval.tasks.jsonl is a round-failures-derived pool where base composite
< 0.5. Build incrementally from failed-task IDs in capability.jsonl rows
where composite < 0.5. Gitignored — blind-eval firewall applies.
EOF

# .gitignore for datasets/eval and hard_eval
cat > "$CAP_DIR/datasets/.gitignore" <<'EOF'
eval.tasks.jsonl
hard_eval.tasks.jsonl
EOF

# Optional rollout.py stub (agentic only)
cat > "$CAP_DIR/rollout.py" <<'EOF'
"""rollout.py — pi-runner for agentic stages.

Reads a task JSONL, drives pi to produce session JSONLs, scores them via
rubric.py, writes:
  rollout.jsonl       (raw sessions + scores)
  grpo-train.jsonl    (kiln-train ScoredRollout shape)

Imports from capabilities/lib/pi_trajectory.py for session normalization.
Or shells out to `kiln trajectory inspect --json`.

Only needed when an agentic-grpo stage is on the pipeline. Delete this
file for non-agentic caps.
"""

# TODO: implement when agentic stage is in scope.
EOF

cat > "$CAP_DIR/archive/README.md" <<'EOF'
Read-only history from prior rounds. A fresh-round agent does NOT need to
read anything here to run the cap.
EOF

echo "scaffolded: $CAP_DIR"
echo ""
echo "next steps:"
echo "  1. Edit $CAP_DIR/capability.md (Description, Adversarial design, Rubric)"
echo "  2. Implement $CAP_DIR/rubric.py"
echo "  3. Populate $CAP_DIR/calibration/{good,bad}.jsonl (≥5 each)"
echo "  4. Run python3 $CAP_DIR/rubric_sanity.py to confirm separation"
echo "  5. Run python3 $CAP_DIR/build_corpus.py to produce train+eval+hard_eval"
echo "  6. Run $CAP_DIR/capability.oracle.sh for baseline (iter 0)"
echo "  7. Run lib/method_router.py to get stage-1 method recommendation"
