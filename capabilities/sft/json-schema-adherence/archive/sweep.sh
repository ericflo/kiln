#!/usr/bin/env bash
# Priority SFT-distillation sweep on the full 188-prompt corpus.
# Each experiment takes ~20-30 min (training) + 30s (eval) + ~10s (restart).
set -euo pipefail
cd /workspace/kiln/sft-cap.json-schema-adherence

# Wave 1: replicate sft-v1 with full data
./run_sft_exp.sh sft-v1f-r16-lr1e4-2ep --rank 16 --lr 1e-4 --epochs 2 --notes "full 188-row data, default config"

# Wave 2: more epochs on the same recipe
./run_sft_exp.sh sft-v3f-r16-lr1e4-3ep --rank 16 --lr 1e-4 --epochs 3 --notes "full data, 3 epochs"

# Wave 3: bigger rank
./run_sft_exp.sh sft-v6f-r64-lr1e4-2ep --rank 64 --lr 1e-4 --epochs 2 --notes "full data, rank 64"

# Wave 4: gentler lr
./run_sft_exp.sh sft-v4f-r16-lr5e5-3ep --rank 16 --lr 5e-5 --epochs 3 --notes "full data, lower lr more epochs"

echo "=== sweep complete ==="
python3 summarize.py
