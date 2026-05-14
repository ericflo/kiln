#!/bin/bash
# Scaffold a new sft-capability-creator session in $PWD/sft-cap.<slug>/.
#
# Usage:  scaffold.sh <slug>
#
# Idempotent. Won't overwrite existing capability.* files; will warn instead.

set -euo pipefail

SLUG="${1-}"
if [ -z "$SLUG" ]; then
  echo "usage: scaffold.sh <slug>" >&2
  exit 2
fi

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="sft-cap.$SLUG"

mkdir -p "$DIR"/{datasets,hypotheses,adapters}
cd "$DIR"

if [ ! -f capability.md ]; then
  cat > capability.md <<'MD'
# Capability: <one-line title>

## Description
<2–4 sentences. Plain English. Copy the user's intake words verbatim.>

## Base model
Qwen/Qwen3.5-4B

## Oracle
Command: `./capability.oracle.sh <adapter_name>`
Output contract: stdout `SCORE=<float>` on the last line, optionally
also `N=<int>`. The wrapper enforces this; we never read the eval's
internals.

## Budget
- Max iterations: 20
- Per-ablation dataset cap: 128 examples
- Per-ablation training cap: 1 epoch, lr 1e-4, rank 4

## Hypothesis taxonomy
<Fill in iteration by iteration.>

## What's been tried
<Append a one-line summary per iteration.>

## Dead ends
<Falsified hypotheses. One line each.>

## Open questions
<Things we couldn't answer this round.>
MD
fi

if [ ! -f capability.config.json ]; then
  cat > capability.config.json <<'JSON'
{
  "workdir": ".",
  "base_model": "Qwen/Qwen3.5-4B",
  "server": "http://localhost:8420",
  "max_iterations": 20,
  "dataset_size_cap": 128,
  "direction": "higher",
  "eval_suite": "<REPLACE-WITH-SUITE-NAME-OR-PASTE>",
  "scorer_field": "accuracy",
  "oracle_mode": "kiln",
  "training_defaults": {
    "lr": "1e-4",
    "epochs": 1,
    "lora_rank": 4
  },
  "anchor_suite": null
}
JSON
fi

if [ ! -f capability.oracle.sh ]; then
  cp "$SKILL_DIR/oracle.sh" capability.oracle.sh
  chmod +x capability.oracle.sh
fi

if [ ! -f capability.ideas.md ]; then
  cat > capability.ideas.md <<'MD'
# Ideas backlog

Promising hypothesis families we haven't tried this round. Append as
bullets; prune when tried; the skill mines this on resume.

-
MD
fi

touch capability.jsonl

echo "scaffolded: $DIR/"
echo "next: fill in capability.md and capability.config.json, then run"
echo "      ./capability.oracle.sh \"\"   # baseline"
