# Phase C6 — Pre-RoPE MTP input bisect (embed / norm / splice / fc taps)

## Summary

**Pre-RoPE MTP input is NOT the dominant source of post-C3 drift.** All five pre-RoPE taps (`token_emb`, `norm_emb`, `norm_h`, `concat`, `fused`) pass the strict `cos_sim ≥ 0.9999` bar against the HF PyTorch reference across all `(seed, mtp_pos)` combinations tested (3 seeds × 2 positions = 6 samples per tap).

| Tap           | cos_sim (min across 6) | cos_sim (median) | max‖Δ‖ (worst across 6) | Verdict |
|---------------|------------------------|------------------|--------------------------|---------|
| `token_emb`   | 1.0000                 | 1.0000           | 0.00e+00                 | **PASS** (byte-identical) |
| `norm_emb`    | 1.0000                 | 1.0000           | 1.38e-02                 | **PASS** |
| `norm_h`      | 1.0000                 | 1.0000           | 3.01e-02                 | **PASS** |
| `concat`      | 1.0000                 | 1.0000           | 3.01e-02                 | **PASS** |
| `fused`       | 1.0000                 | 1.0000           | 1.60e-02                 | **PASS** |

None of the five pre-RoPE taps crosses the 0.9999 bar downward. Residual max‖Δ‖ values are ≤3.01e-02 and are consistent with bf16 arithmetic variance between Candle (kiln) and PyTorch (reference) RMSNorm + fc matmul kernels — not a semantic divergence.

**Verdict: downstream of the `fused` tap is the correct next bisect target.** The first sub-op tap to drop below `cos_sim = 1.00e+00` at `mtp_pos=2` is consistently `post_attn_raw` (SDPA output) — not any pre-RoPE or RoPE tap. Phase C7 should target attention / KV-cache / causal-mask behavior at `mtp_pos > 0`.

## Environment

| Field            | Value                                                 |
|------------------|-------------------------------------------------------|
| Repo             | `ericflo/kiln`                                         |
| Branch           | `mtp/phase-c6-pre-rope-bisect`                         |
| Base SHA         | `0feaf55` (post PR #316, Phase C5 bench report merged) |
| GPU              | NVIDIA RTX A6000 (49 140 MB)                           |
| Driver / CUDA    | 550.127.08 / 12.4.131                                  |
| Image            | `ghcr.io/ericflo/kiln-runpod:latest`                   |
| Model            | Qwen3.5-4B (2 safetensors shards, 4 206 M params)      |
| Build            | `cargo build --release --features cuda --bin kiln-bench` (sccache warm, 18 s) |
| Bench flags      | `--paged --prompt-tokens 29 --max-output-tokens 8 --skip-training --seed {42,43,44}` (seed 43 re-run with `--max-output-tokens 32` to hit `mtp_pos=2`) |
| Instrumentation  | `KILN_MTP_DUMP_PRE_ROPE=1 KILN_MTP_DUMP_POS=0,2 KILN_MTP_DUMP_SUBOPS=1` |
| Comparator bar   | `cos_sim ≥ 0.9999` (fp32-strict); max‖Δ‖ / mean‖Δ‖ / rel_l2 reported for context |

## Tap definitions

| Ordinal | kiln source                                          | HF reference parallel                                 |
|---------|------------------------------------------------------|-------------------------------------------------------|
| 1       | `token_emb = embed_tokens(draft_token_id)`           | `embed_tokens(draft_token_id)`                        |
| 2       | `norm_emb = enorm(token_emb)` (RMSNorm)              | `pre_fc_norm_embedding(token_emb)` (RMSNorm)          |
| 3       | `norm_h = hnorm(h_main)` (RMSNorm)                   | `pre_fc_norm_hidden(h_main)` (RMSNorm)                |
| 4       | `concat = cat([norm_emb, norm_h], dim=-1)` (1×5120)  | `cat([norm_emb, norm_h], dim=-1)` (1×5120)            |
| 5       | `fused = fc(concat)` (W4A16 Marlin on kiln / bf16 linear on ref) | `mtp.fc(concat)` (bf16 linear)              |

The 5 taps are dumped as `c6__<name>` safetensors entries inside each MTP dump file. The Phase B6 pre-existing taps (`tok_embed`, `fc_input`, `fc_output`) share activations 1, 4, 5 but under different names; the new `c6__` prefix keeps the bisect self-contained and back-compat with prior B-phase dumps.

## Per-tap results (3 seeds × 2 positions = 6 samples)

### `mtp_pos = 0`

| Seed | Tap         | cos_sim   | max‖Δ‖   | mean‖Δ‖  | rel_l2   | ok |
|------|-------------|-----------|-----------|-----------|-----------|----|
| 42   | token_emb   | 1.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✓ |
| 42   | norm_emb    | 1.00e+00 | 1.38e-02 | 6.50e-04 | 1.60e-03 | ✓ |
| 42   | norm_h      | 1.00e+00 | 1.14e-02 | 6.86e-04 | 1.52e-03 | ✓ |
| 42   | concat      | 1.00e+00 | 1.38e-02 | 6.68e-04 | 1.56e-03 | ✓ |
| 42   | fused       | 1.00e+00 | 5.62e-03 | 5.55e-04 | 2.87e-03 | ✓ |
| 43   | token_emb   | 1.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✓ |
| 43   | norm_emb    | 1.00e+00 | 7.40e-03 | 6.69e-04 | 1.66e-03 | ✓ |
| 43   | norm_h      | 1.00e+00 | 1.58e-02 | 7.41e-04 | 1.63e-03 | ✓ |
| 43   | concat      | 1.00e+00 | 1.58e-02 | 7.05e-04 | 1.64e-03 | ✓ |
| 43   | fused       | 1.00e+00 | 8.57e-03 | 5.67e-04 | 2.89e-03 | ✓ |
| 44   | token_emb   | 1.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✓ |
| 44   | norm_emb    | 1.00e+00 | 7.54e-03 | 6.86e-04 | 1.65e-03 | ✓ |
| 44   | norm_h      | 1.00e+00 | 1.48e-02 | 7.03e-04 | 1.52e-03 | ✓ |
| 44   | concat      | 1.00e+00 | 1.48e-02 | 6.95e-04 | 1.58e-03 | ✓ |
| 44   | fused       | 1.00e+00 | 6.55e-03 | 5.70e-04 | 2.81e-03 | ✓ |

### `mtp_pos = 2`

| Seed | Tap         | cos_sim   | max‖Δ‖   | mean‖Δ‖  | rel_l2   | ok |
|------|-------------|-----------|-----------|-----------|-----------|----|
| 42   | token_emb   | 1.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✓ |
| 42   | norm_emb    | 1.00e+00 | 7.76e-03 | 6.56e-04 | 1.66e-03 | ✓ |
| 42   | norm_h      | 1.00e+00 | 3.01e-02 | 7.49e-04 | 1.62e-03 | ✓ |
| 42   | concat      | 1.00e+00 | 3.01e-02 | 7.03e-04 | 1.63e-03 | ✓ |
| 42   | fused       | 1.00e+00 | 7.34e-03 | 6.17e-04 | 3.17e-03 | ✓ |
| 43   | token_emb   | 1.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✓ |
| 43   | norm_emb    | 1.00e+00 | 5.43e-03 | 6.64e-04 | 1.61e-03 | ✓ |
| 43   | norm_h      | 1.00e+00 | 7.66e-03 | 8.06e-04 | 1.62e-03 | ✓ |
| 43   | concat      | 1.00e+00 | 7.66e-03 | 7.35e-04 | 1.62e-03 | ✓ |
| 43   | fused       | 1.00e+00 | 5.07e-03 | 6.34e-04 | 2.71e-03 | ✓ |
| 44   | token_emb   | 1.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✓ |
| 44   | norm_emb    | 1.00e+00 | 3.98e-03 | 6.60e-04 | 1.61e-03 | ✓ |
| 44   | norm_h      | 1.00e+00 | 1.46e-02 | 7.91e-04 | 1.68e-03 | ✓ |
| 44   | concat      | 1.00e+00 | 1.46e-02 | 7.25e-04 | 1.65e-03 | ✓ |
| 44   | fused       | 1.00e+00 | 1.60e-02 | 6.29e-04 | 2.79e-03 | ✓ |

All 30 (seed × position × tap) cells report `cos_sim = 1.00e+00`, comfortably above the 0.9999 bar.

## Per-seed aggregate (top-down first-drop audit)

For each seed & position the comparator walks the 5 taps top-down and reports the first tap with `cos_sim < 0.9999`:

| Seed | Position    | First tap below bar | Result |
|------|-------------|---------------------|--------|
| 42   | mtp_pos=0   | (none)              | all 5 taps pass |
| 42   | mtp_pos=2   | (none)              | all 5 taps pass |
| 43   | mtp_pos=0   | (none)              | all 5 taps pass |
| 43   | mtp_pos=2   | (none)              | all 5 taps pass |
| 44   | mtp_pos=0   | (none)              | all 5 taps pass |
| 44   | mtp_pos=2   | (none)              | all 5 taps pass |

**Unanimous: 6/6 pass.** No pre-RoPE tap is the dominant source of drift.

## Where the drift actually emerges (downstream of `fused`)

The B6 sub-op capture (unchanged from prior phases, atol=1e-3 / rtol=1e-2 allclose bar) already walks the full MTP transformer block and final layernorm. At `mtp_pos=2`, across all three seeds, the first sub-op tap that drops below `cos_sim = 1.00e+00` is consistently **`post_attn_raw`** — the output of the attention SDPA operator inside the MTP block:

| Tap (downstream of `fused`) | seed 42 cos | seed 43 cos | seed 44 cos |
|------------------------------|-------------|-------------|-------------|
| `c6__fused`                  | 1.00e+00    | 1.00e+00    | 1.00e+00    |
| `post_pre_attn_norm`         | 1.00e+00    | 1.00e+00    | 1.00e+00    |
| `post_q_proj_raw`            | 1.00e+00    | 1.00e+00    | 1.00e+00    |
| `post_q_rope`                | 1.00e+00    | 1.00e+00    | 1.00e+00    |
| `post_k_rope`                | 1.00e+00    | 1.00e+00    | 1.00e+00    |
| **`post_attn_raw`** (SDPA)   | **9.93e-01** | **9.86e-01** | **9.96e-01** |
| `post_o_proj`                | 9.75e-01    | 9.88e-01    | 9.94e-01    |
| `post_layer`                 | 9.81e-01    | 9.92e-01    | 9.98e-01    |
| `mtp_logits`                 | 9.80e-01    | 9.93e-01    | 9.97e-01    |

By contrast, at `mtp_pos=0` every downstream tap in all three seeds stays at `cos_sim = 1.00e+00` through `mtp_logits`. The asymmetry is: the bug only surfaces when `mtp_pos > 0`, and the first signal is post-SDPA, despite `post_q_rope` / `post_k_rope` still matching bit-for-bit against the reference. This localizes the residual drift to the attention computation itself (SDPA inputs from the KV cache, causal mask extent, or how past-K/V is read when the draft query's absolute position is `base_pos + mtp_pos`).

## What this means for the next phase

- **Pre-RoPE input is ruled out.** The 5-tap bisect unanimously clears the embed → RMSNorm → concat → fc pipeline.
- **RoPE position threading is ruled out.** `post_q_rope` and `post_k_rope` match the reference at both `mtp_pos=0` and `mtp_pos=2`.
- **The SDPA step is the first divergence, and only at `mtp_pos > 0`.** This is consistent with a KV-cache / causal-mask / attention-extent bug that the C3 fix (PR #314) did not fully resolve. Phase C5 observed per-position acceptance `α(mtp_pos=0) = 6.1 %`, `α(mtp_pos=2) = 4.2 %`; the C6 downstream evidence here is directly aligned with that per-position degradation.

Phase C7 (recommended) should bisect inside the attention computation: dump `scaled_dot_product_attention` inputs (Q, K, V, attention mask) from both kiln and HF at `mtp_pos=2`, diff each, and identify whether:

1. the causal mask has the wrong right-edge for MTP draft positions;
2. the KV cache contains different past-K/V entries than HF ref assumes; or
3. the abs_pos used for the SDPA attention-window offset differs from the abs_pos used for RoPE (currently `base_pos + mtp_pos` for RoPE).

## Artifacts

- Kiln MTP dumps: `/tmp/c6_kiln/kiln_seed{42,43,44}_pos{0,2}.st` (pod `zu0xihe9lxuzha`)
- HF reference dumps: `/tmp/c6_ref/ref_seed{42,43,44}_pos{0,2}.st`
- Full compare log: `docs/archive/phase-c/phase-c6/c6-cmp.log` (3 seeds × 2 positions, ~450 lines)
- Comparator: `python3 scripts/mtp_compare.py --c6 --pair "mtp_pos=<n>:<kiln>.st,<ref>.st" [...]`

## Reproduction

```bash
# Build (requires A6000 + kiln-runpod image + sccache pointed at b2://.../kiln/)
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench

# Dump (emits c6__<name> entries alongside existing b6/b11/b12 taps)
export KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp
export KILN_MTP_DUMP_PRE_ROPE=1 KILN_MTP_DUMP_SUBOPS=1
export KILN_MTP_DUMP_POS=0,2
export KILN_MTP_DUMP_PATH='/tmp/c6_kiln/kiln_seed42_pos{pos}.st'
./target/release/kiln-bench --model-path <qwen3.5-4b-dir>/ \
  --paged --prompt-tokens 29 --max-output-tokens 32 --skip-training --seed 42

# HF reference
python3 scripts/mtp_reference_dump.py \
  --checkpoint <qwen3.5-4b-dir>/ \
  --kiln-dump /tmp/c6_kiln/kiln_seed42_pos0.st \
  --out /tmp/c6_ref/ref_seed42_pos0.st --capture-subops

# Bisect
python3 scripts/mtp_compare.py --c6 \
  --pair "mtp_pos=0:/tmp/c6_kiln/kiln_seed42_pos0.st,/tmp/c6_ref/ref_seed42_pos0.st" \
  --pair "mtp_pos=2:/tmp/c6_kiln/kiln_seed42_pos2.st,/tmp/c6_ref/ref_seed42_pos2.st"
```

Exit code is non-zero when the B-phase allclose bar (atol=1e-3) flags downstream divergence — that is expected post-C6 and is not a failure of the pre-RoPE bisect itself. The C6-specific verdict `ALL pre-RoPE taps match within cos_sim >= 0.9999.` is printed unconditionally.
