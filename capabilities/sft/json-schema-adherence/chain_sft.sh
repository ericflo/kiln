#!/usr/bin/env bash
# Run sft-v3 + sft-v4 + restart teacher + full-data sweep.
# Uses the actual binary path in pgrep to avoid self-matching with shell processes.
set -e
cd /workspace/kiln/sft-cap.json-schema-adherence

EXAMPLE_BIN="/workspace/kiln/target/release/examples/cuda_sft_file"
KILN_BIN="/workspace/kiln/target/release/kiln"
TEACHER_BIN="teacher_inference.py"

wait_no_gpu() {
  # Wait until no GPU-using process is running
  while pgrep -f "$EXAMPLE_BIN" >/dev/null 2>&1 \
     || pgrep -f "$TEACHER_BIN" >/dev/null 2>&1 \
     || pgrep -f "$KILN_BIN serve" >/dev/null 2>&1; do
    sleep 30
  done
}

wait_no_gpu
echo "[chain] starting at $(date)"

# sft-v3 already done with composite 0.9298 — skip.

# sft-v4: rank=16 lr=5e-5 3 epochs — gentler LR
echo "[chain] running sft-v4"
./run_sft_exp.sh sft-v4-r16-lr5e5-3ep --rank 16 --lr 5e-5 --epochs 3 --notes "38-row partial, lower LR more epochs"

# sft-v5: rank=16 lr=2e-4 1 epoch — aggressive LR
echo "[chain] running sft-v5"
./run_sft_exp.sh sft-v5-r16-lr2e4-1ep --rank 16 --lr 2e-4 --epochs 1 --notes "38-row partial, aggressive LR 1 epoch"

# After both, restart teacher to fill in rows 38..188
echo "[chain] SFT mini-sweep done. Restarting teacher at $(date)."
python3 teacher_inference.py --max-new-tokens 1024 --top-k 32 --start 38 >> teacher_inference.log 2>&1 &
TEACHER_PID=$!
echo "[chain] Teacher restarted as pid $TEACHER_PID"

# Wait for teacher to fill in remaining rows
until [ "$(wc -l < datasets/teacher.fixture.jsonl)" -ge 188 ] || ! kill -0 $TEACHER_PID 2>/dev/null; do
  sleep 60
done
echo "[chain] TEACHER FULL DONE at $(date), rows=$(wc -l < datasets/teacher.fixture.jsonl)"

# Run final sweep on FULL data
echo "[chain] running sft-v1f (full data)"
./run_sft_exp.sh sft-v1f-r16-lr1e4-2ep --rank 16 --lr 1e-4 --epochs 2 --notes "FULL 188 data, default"
echo "[chain] running sft-v3f (full data, 3 epochs)"
./run_sft_exp.sh sft-v3f-r16-lr1e4-3ep --rank 16 --lr 1e-4 --epochs 3 --notes "FULL 188, 3 epochs"

# Oracle ceiling
echo "[chain] running oracle eval"
python3 eval_teacher.py --out judgments/oracle-27b.json --max-tokens 800 2>&1 | tail -3

# Summary
python3 summarize.py
echo "[chain] === CHAIN COMPLETE at $(date) ==="
