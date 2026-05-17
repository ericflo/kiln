#!/usr/bin/env bash
# Scaffold an opd-cap.<slug>/ workdir for the opd-capability-creator skill.
#
# Usage:
#   $SKILL/templates/scaffold.sh <slug>
#
# Creates:
#   opd-cap.<slug>/
#     capability.md             (template; agent fills in description + rubric)
#     capability.config.json    (template; agent edits teacher URL, max iters)
#     capability.oracle.sh      (template oracle wrapper; agent points at eval)
#     capability.jsonl          (empty — append-only log)
#     kiln-polish.jsonl         (empty — separate polish ledger)
#     prompts/, hypotheses/, adapters/, responses/  (empty dirs)
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "usage: scaffold.sh <slug>" >&2
  exit 2
fi
SLUG="$1"
DIR="opd-cap.$SLUG"

if [ -e "$DIR" ]; then
  echo "$DIR already exists; refusing to overwrite" >&2
  exit 1
fi

mkdir -p "$DIR"/{prompts,hypotheses,adapters,responses}
cd "$DIR"

cat > capability.md <<EOF
# Capability: $SLUG

## Description
<2–4 sentences in the user's own words. Do NOT paraphrase to "improve".>

## Base model
<e.g. Qwen/Qwen3.5-4B at kiln serve on :8420>

## Teacher
<served-name, quantization, vLLM URL, max_logprobs>
Example: \`vllm/qwen3.6-27b-awq\` at http://localhost:8002 (AWQ-INT4, max_logprobs=64)

## Rubric
<sub-score names + weights + what each measures.
Do NOT include eval-set examples here.>

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
|           |        |                  |

## Baseline (filled by headroom.py after iter 0)
| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
|           |        |          |                     |
| **Total** |        |          | **<sum>**           |

## Target sub-score
<the sub-score with the most headroom; fill in after baseline.>

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
|      |      |        |           |        |          |         |

## Dead ends
<one line per retired family with falsifying evidence.>

## Open questions
<carry to next-session.md at close.>

## Checkpoints
<every 3rd iter, post a brief progress summary here under \`### Checkpoint at iter N\`.>
EOF

cat > capability.config.json <<EOF
{
  "slug": "$SLUG",
  "base_model_path": "/workspace/kiln/Qwen3.5-4B",
  "kiln_server_url": "http://localhost:8420",
  "teacher": {
    "url": "http://localhost:8002",
    "served_name": "qwen3.6-27b-awq",
    "max_logprobs": 64,
    "gpu_memory_utilization": 0.45
  },
  "training_defaults": {
    "rank": 16,
    "alpha": 32,
    "lr": 1e-4,
    "epochs": 6,
    "samples_per_prompt": 1,
    "max_tokens": 64,
    "top_k": 8,
    "temperature": 1.0,
    "top_p": 0.9,
    "streaming_prefill": true
  },
  "budget": {
    "max_iterations": 12,
    "max_prompt_tokens": 650
  },
  "rubric": {
    "sub_scores": [],
    "composite_formula": "<weights here, e.g. 0.4*parses+0.3*validates+...>",
    "direction": "higher_is_better"
  }
}
EOF

cat > capability.oracle.sh <<'EOF'
#!/usr/bin/env bash
# Blind oracle wrapper. Takes one positional arg (adapter name; empty = base).
#
# Output contract: on success, prints exactly these lines on stdout —
#   SCORE=<float>
#   <SUBNAME1>=<float>
#   <SUBNAME2>=<float>
#   ...
#   N=<int>
# Nothing else.
#
# On failure: exit non-zero and print `ORACLE_ERROR: <reason>` on stderr.
#
# The agent never reads anything else from this wrapper's intermediate
# files. The wrapper is the contract; the eval internals are blind.
set -euo pipefail
ADAPTER="${1:-}"

# === Customize this section per capability ===
# Example: call the user's eval, parse one JSON line, emit SCORE= lines.
# Replace the below with your eval invocation.

if [ -z "$ADAPTER" ]; then
  ADAPTER_ARG=""
else
  ADAPTER_ARG="--adapter $ADAPTER"
fi

# Customize: invoke eval, parse, emit.
# Below is a stub the user must replace.
echo "ORACLE_ERROR: capability.oracle.sh not implemented — see template" >&2
exit 1
EOF
chmod +x capability.oracle.sh

touch capability.jsonl kiln-polish.jsonl

echo "scaffolded: $DIR/"
echo "next: edit capability.md, capability.config.json, capability.oracle.sh; commit; run baseline."
