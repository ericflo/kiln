#!/usr/bin/env bash
# Run automatically after teacher completes.
set -euo pipefail
cd /workspace/kiln/sft-cap.json-schema-adherence

# 1. Build the canonical kiln-tokenized tokens file (matches what trainer sees)
/workspace/kiln/target/release/examples/tokenize_opd_prompts \
  --model-path /workspace/kiln/Qwen3.5-4B \
  --in datasets/train.opd.jsonl \
  --out datasets/train.opd.tokens.jsonl

# 2. Build short subset for OPD experiments
python3 filter_short.py --max-tokens 500

# 3. Run SFT sweep
./sweep.sh
echo "=== SFT sweep done ==="

# 4. Run eval_teacher to establish oracle ceiling
python3 eval_teacher.py --out judgments/oracle-27b.json --max-tokens 800 2>&1 | tail -5
echo "=== oracle eval done ==="

# 5. Try OPD on filtered short subset (may OOM; capture errors)
./sweep_opd.sh || echo "(OPD sweep may have failed)"

# 6. Print final summary
python3 summarize.py
