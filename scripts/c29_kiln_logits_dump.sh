#!/bin/bash
# Phase C29 kiln-side empirical MTP logits dump driver (H9).
#
# Drives kiln-bench across multiple prompt seeds with the full 8-tap dump
# infrastructure armed (pre-RoPE 5 + C14 post-block 3) so the C29 comparator
# can score top-K Jaccard agreement on the `c14__logits` tap against an HF
# reference across diverse prompt content.
#
# Layout produced (matches what c29_hf_reference_dump.py expects):
#
#     <out_root>/
#         seed-0/
#             mtp_pos-0/step-0.safetensors
#             mtp_pos-0/step-1.safetensors
#             ...
#             mtp_pos-3/step-1.safetensors
#         seed-1/
#             ...
#
# Each seed picks one of the 8 prompts from kiln-bench's PROMPT_POOL via
# `seed % 8`; we use 0..N-1 to walk distinct prompt content.
#
# C13 + C14 already proved cos_sim ≥ 0.99995 on every tap. C29's hypothesis
# (H9) is that sub-percent cosine drift on a 248K-dim logits vector still
# rotates top-K selections enough to depress α — top-K Jaccard catches that.

set -euo pipefail

KILN_BENCH="${KILN_BENCH:-./target/release/kiln-bench}"
MODEL_PATH="${MODEL_PATH:-/workspace/qwen3.5-4b}"
OUT_ROOT="${OUT_ROOT:-/workspace/captures-c29}"
SEEDS="${SEEDS:-0 1 2 3}"
MAX_STEPS="${MAX_STEPS:-2}"
POSITIONS="${POSITIONS:-0,1,2,3}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
DECODE_TOKENS="${DECODE_TOKENS:-16}"

if [ ! -x "$KILN_BENCH" ]; then
    echo "[c29_kiln] kiln-bench not found at $KILN_BENCH" >&2
    echo "[c29_kiln] build it with: KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench" >&2
    exit 2
fi

mkdir -p "$OUT_ROOT"
echo "[c29_kiln] driving $(echo $SEEDS | wc -w) prompts × $(echo $POSITIONS | tr ',' ' ' | wc -w) positions × $MAX_STEPS steps" >&2
echo "[c29_kiln] out_root=$OUT_ROOT" >&2

for seed in $SEEDS; do
    seed_dir="$OUT_ROOT/seed-$seed"
    mkdir -p "$seed_dir"
    echo "[c29_kiln] === seed $seed -> $seed_dir ===" >&2
    KILN_MTP_DUMP_PATH="$seed_dir" \
    KILN_MTP_DUMP_SPLICE=1 \
    KILN_MTP_DUMP_SPLICE_POS="$POSITIONS" \
    KILN_MTP_DUMP_SPLICE_MAX_STEPS="$MAX_STEPS" \
    KILN_MTP_DUMP_PRE_ROPE=1 \
    KILN_MTP_DUMP_C7_SDPA=1 \
    KILN_MTP_DUMP_C14_POST_BLOCK=1 \
    KILN_MTP_DUMP_SUBOPS=1 \
    KILN_MTP_DEBUG=1 \
    KILN_SPEC_METHOD=mtp \
    KILN_W4A16=1 \
    "$KILN_BENCH" \
        --model-path "$MODEL_PATH" \
        --paged \
        --skip-training \
        --seed "$seed" \
        --prompt-tokens "$PROMPT_TOKENS" \
        --max-output-tokens "$DECODE_TOKENS" \
        2>&1 | tail -40 || {
            echo "[c29_kiln] seed $seed FAILED — continuing with remaining seeds" >&2
            continue
        }
    n_dumps=$(find "$seed_dir" -name "step-*.safetensors" | wc -l)
    echo "[c29_kiln] seed $seed produced $n_dumps dumps" >&2
done

total=$(find "$OUT_ROOT" -name "step-*.safetensors" | wc -l)
echo "[c29_kiln] DONE — $total total kiln dumps under $OUT_ROOT" >&2
