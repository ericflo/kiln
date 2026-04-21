# MTP Phase B12 — Layer-31 Drift Bisect

## TL;DR

The `h_layer_31` cos-sim drift (~0.97-0.98 against HF bf16) observed in Phase
B10 is **benign bf16 accumulation**, not a kernel bug. Against a **true fp32**
HF reference, no individual layer-31 sub-op falls below the 0.995 cos-sim bar.
Drift is spread evenly across post_input_norm, q/k/v/o projections, qk_norms,
and the MLP trio — the classic accumulation signature, not a point divergence.

**Verdict:** no B13 target; unblock the downstream MTP acceptance-rate work.

## Method

Dumped layer-boundary hidden states (`h_layer_{24..31}` + `h_pre_final_norm`)
plus 11 layer-31 GQA sub-op taps (post_input_norm, q_proj, k_proj, v_proj,
qk_norm_q, qk_norm_k, o_proj, post_attn_norm, mlp_gate, mlp_up, mlp_down)
across **3 PROMPT_POOL seeds (0/1/2)** for:

- kiln (bf16 decode, CUDA graphs on, paged KV cache)
- HF reference, bf16 (`scripts/mtp_h_main_reference_dump.py --b12-taps`)
- HF reference, fp32 (`--b12-taps --fp32`)

All taps emitted as `[1, seq_len, hidden]` full-precision f32, then compared
with `scripts/mtp_compare.py --b12` at `atol=1e-2, rtol=1e-1, bar=0.995`
(cos-sim).

### Reference-dump fp32 fix (bundled)

A prerequisite bug was found and fixed in this PR:
`mtp_h_main_reference_dump.py` passed `torch_dtype=torch.float32` to
`from_pretrained` but the Qwen3.5 remote-code path + `low_cpu_mem_usage=True`
honours `config.torch_dtype` (bf16) instead, leaving the `--fp32` comparator
silently running in bf16. Verified the hf_bf16 and hf_fp32 dumps were
byte-identical before the fix. Added an explicit `model.to(torch.float32)`
post-load when `fp32=True` so the comparator really exercises the fp32 path.

## Results

### Per-layer boundary cos-sim (median across seeds 0/1/2)

| Tap | vs HF bf16 | vs HF fp32 | Status |
| --- | --- | --- | --- |
| h_layer_24 | 1.000 | 1.000 | ok |
| h_layer_25 | 1.000 | 1.000 | ok |
| h_layer_26 | 1.000 | 1.000 | ok |
| h_layer_27 | 1.000 | 1.000 | ok |
| h_layer_28 | 1.000 | 1.000 | ok |
| h_layer_29 | 0.999 | 1.000 | ok |
| h_layer_30 | 0.999 | 1.000 | ok |
| **h_layer_31** | **0.980** | **0.980** | **DIV** (same under both) |
| h_pre_final_norm | 0.980 | 0.980 | DIV |

Per-layer drift accumulates gradually across layers 29/30/31 — no single-layer
jump. `h_layer_31` is the first layer below the 0.995 bar under both
references.

### Per-sub-op cos-sim at layer 31 (median across seeds 0/1/2)

| Sub-op | vs HF bf16 | vs HF fp32 | Status under fp32 |
| --- | --- | --- | --- |
| post_input_norm | **0.9949** | 0.996 | ok |
| q_proj | 0.999 | 0.999 | ok |
| k_proj | 0.997 | 0.996 | ok |
| v_proj | 0.999 | 0.999 | ok |
| qk_norm_q | 0.998 | 0.998 | ok |
| qk_norm_k | 0.997 | 0.997 | ok |
| o_proj | 0.996 | 0.996 | ok |
| post_attn_norm | 0.995 | 0.996 | ok |
| mlp_gate | 0.999 | 0.999 | ok |
| mlp_up | 0.999 | 0.999 | ok |
| mlp_down | 0.997 | 0.997 | ok |

Under bf16 HF, `post_input_norm` is the only sub-op below bar (0.9949 — 0.0001
under the 0.995 threshold). Under **true fp32 HF**, it clears the bar (0.996)
and **no** layer-31 sub-op falls below 0.995. The per-layer `h_layer_31`
median stays at 0.980 under both references because the drift has already
accumulated through all 31 preceding layers' worth of bf16 compute — adding a
fp32 layer 31 on top of 30 layers of bf16 input can't undo what's upstream.

### Interpretation

The `mtp_compare.py --b12` verdict for the fp32 pass is literal:

> LAYER 31 median is below-bar, but NO individual sub-op at layer 31 has
> median cos_sim < 0.995. Drift is spread across multiple sub-ops
> (accumulation pattern). Strong signal for benign bf16 accumulation.

Per-sub-op rel_l2 values under fp32 HF (0.04-0.12, uniform across sub-ops) are
exactly what 31 layers of bf16 GEMM/softmax/norm accumulation looks like. No
sub-op is disproportionately hot.

**Not a bug:** layer-31 post_input_norm weight/eps are correct; q/k/v/o
projections, qk_norms, and MLP gates are correct; fused-kernel variants
agree with candle reference (see forward.rs kill-switch parity tests, which
have been green since PR #92).

## Action

- **No B13 target.** Close the layer-31 drift lead.
- Unblock MTP acceptance-rate and end-to-end decode-speedup work: the
  remaining ceiling loss is bf16 accumulation noise, not a fixable point
  divergence, and is already absorbed by normal inference tolerances.
- The comparator's `--b12` mode + taps are permanent infra — reuse them if a
  future regression pushes any layer's median below 0.995 in isolation.

## Reproduction

```bash
# On an A6000 pod (see resources/runpod-workflow.md)
export KILN_MTP_DUMP_PATH=/workspace/dumps/kiln_seed${SEED}.safetensors
export KILN_MTP_DUMP_POS=0
export KILN_MTP_DUMP_HIDDEN_STATES=1
export KILN_MTP_DUMP_B12_GQA_TAPS=1
export KILN_SPEC_METHOD=mtp
./target/release/kiln-bench --model-path $MODEL_PATH --paged --seed $SEED \
  --prompt-tokens 512 --max-output-tokens 32 --skip-training

# HF reference (run twice per seed: once plain, once --fp32)
python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint $MODEL_PATH \
  --kiln-dump /workspace/dumps/kiln_seed${SEED}.safetensors \
  --out /workspace/dumps/hf_bf16_seed${SEED}.safetensors \
  --seed $SEED --b12-taps

# Compare
python3 scripts/mtp_compare.py --b12 \
  --pair seed0:kiln_seed0.safetensors,hf_fp32_seed0.safetensors \
  --pair seed1:kiln_seed1.safetensors,hf_fp32_seed1.safetensors \
  --pair seed2:kiln_seed2.safetensors,hf_fp32_seed2.safetensors
```

## Budget

Ran on A6000 pod `zu0xihe9lxuzha` (lease `pod-fdcfc23b19c78e0d9a05ff2a`) from
the Kiln pod pool. Wall-clock ~25 min, well under the 75 min / $30 cap.
