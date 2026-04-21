# Phase C8 — Fix MTP SDPA kv_len mismatch (single-site)

## Verdict

**Phase C8 fixes the MTP inner-block SDPA divergence localized in Phase C7 (PR #319).** The kiln MTP transformer block now performs single-token self-attention (`kv_len = 1`) for the inner GQA call, matching the HF / vLLM `Qwen3NextMultiTokenPredictor` reference contract. All 7 SDPA-internal taps now meet the strict `cos_sim ≥ 0.9999` bar across 3 seeds × 2 positions = 6 pairs; the C6 pre-RoPE bar continues to pass; shape parity is restored on `pre_sdpa_k`, `pre_sdpa_v`, `attn_scores_pre_softmax`, and `attn_probs`; and `attn_probs` collapses to exactly 1.0 (max|Δ| = 0, mean|Δ| = 0) — confirming the kv_len = 1 softmax-collapse contract.

The downstream `fc_output` divergence reported by the C6/C7 comparator (`pair verdict: first divergence at tap 'fc_output'`) is the pre-existing main-model `fc` matmul drift (Class B 87.6%, see Phase C6 report), out of scope for C8 — that's C9 work.

## Option chosen: A (kiln conforms to single-token self-attention)

Phase C7 explicitly recommended Option A: change kiln so the MTP inner transformer block performs single-token self-attention without participation in the main-model paged KV cache. The competing option (B) would have changed the reference dumper to attend over the full main-model history. We chose A because:

1. **Upstream parity.** HF Transformers `Qwen3NextMultiTokenPredictor` (in the `qwen3_next` modeling file shipped with the Qwen3-Next checkpoints) explicitly runs the MTP inner block as a single-token self-attention over the freshly-fused `h_fused` projection. The vLLM `Qwen3NextMultiTokenPredictor` matches. Conforming kiln to that contract keeps speculative decoding aligned with the only two production reference implementations available.
2. **Smaller blast radius.** Option A is a single-site change in `gqa_attention_paged` + a TLS arm/disarm pair in `mtp_forward_step`, gated by a thread-local flag that defaults `false`. Non-MTP paths and pre-C8 callers see the legacy paged-cache behavior unchanged. Option B would have required re-engineering the reference dumper to thread the main-model paged history through the HF attention path — fragile and harder to maintain.
3. **Speculative decoding correctness.** With the MTP layer attending to `mtp_pos + 1` previous MTP-draft tokens, the draft logits encode a stale tail of speculative attempts that the main model never accepted. Single-token self-attention removes this contamination and is the correct contract for token-by-token speculative draft.

## Single-file diff summary (2 files, +87 / −6 lines)

```
crates/kiln-model/src/forward.rs   | 46 ++++++++++++++++++++++++++++++++-----
crates/kiln-model/src/mtp_debug.rs | 47 ++++++++++++++++++++++++++++++++++++++
2 files changed, 87 insertions(+), 6 deletions(-)
```

### `crates/kiln-model/src/mtp_debug.rs` (+47 lines)

- Added `MTP_SINGLE_TOKEN_SELF_ATTN_ARMED: RefCell<bool>` to the `thread_local!` block (default `false`).
- Added three pub fns mirroring the existing C7 SDPA-capture armer:
  - `is_mtp_single_token_self_attn_armed() -> bool`
  - `arm_mtp_single_token_self_attn()`
  - `disarm_mtp_single_token_self_attn()`

The TLS slot is read on the hot path (`gqa_attention_paged`) but defaults to `false`, so non-MTP attention paths and production decode pay one TLS-borrow cost per call.

### `crates/kiln-model/src/forward.rs` (+40 / −6 lines, 1 site)

`gqa_attention_paged` reads the TLS flag once at the top, then gates three regions on it:

1. **Skip the paged KV cache write** when armed (no history is being maintained).
2. **Skip the fused paged-decode flash-attention kernel** when armed (the kernel reads the full cache history, defeating the kv_len = 1 contract).
3. **Skip the paged KV cache read** when armed; instead use the just-computed `(k, v)` directly with `kv_len = 1`.

`mtp_forward_step` brackets the inner `transformer_block_paged` call with `arm_mtp_single_token_self_attn()` / `disarm_mtp_single_token_self_attn()`. The disarm runs in both success and error paths — the inner block return is `?`-propagated below, so we cannot rely on the function tail.

## Verification

### Build

A6000 on-demand pod (image `ghcr.io/ericflo/kiln-runpod:latest`, SM 86, CUDA 12.4). Cargo release build with `--features cuda`, sccache + B2 remote cache: 145s wall-clock (97% sccache hit rate from B2 cache).

### Bench protocol

```
KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp \
KILN_MTP_DUMP_PRE_ROPE=1 KILN_MTP_DUMP_SUBOPS=1 KILN_MTP_DUMP_C7_SDPA=1 \
KILN_MTP_DUMP_POS=1,2 \
KILN_MTP_DUMP_PATH="/workspace/c8_kiln/kiln_seed${seed}_pos{pos}.st" \
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b/ --paged \
  --prompt-tokens 29 --max-output-tokens 32 --skip-training --seed $seed
```

Coverage: seeds 42 / 43 / 44, positions 1 / 2 — 6 pair dumps. Reference dumps captured with `python3 scripts/mtp_reference_dump.py --capture-subops` against the same checkpoint at `/workspace/qwen3.5-4b/`.

### C7 sweep (strict bar `cos_sim ≥ 0.9999`)

| Tap | shape (kiln, ref) | shape ok | median cos_sim | max max\|Δ\| | status |
| --- | --- | --- | --- | --- | --- |
| `c7__pre_sdpa_q` | `(1, 16, 1, 256)` / `(1, 16, 1, 256)` | yes | 1.0000 | 6.19e-02 | **ok** |
| `c7__pre_sdpa_k` | `(1, 4, 1, 256)` / `(1, 4, 1, 256)` | yes (was `kv_len=mtp_pos+1`) | 1.0000 | 5.24e-02 | **ok** |
| `c7__pre_sdpa_v` | `(1, 4, 1, 256)` / `(1, 4, 1, 256)` | yes (was `kv_len=mtp_pos+1`) | 1.0000 | 6.93e-02 | **ok** |
| `c7__causal_mask` | `()` / `()` | yes | n/a (0-d sentinel) | 0 | n/a |
| `c7__attn_scores_pre_softmax` | `(1, 16, 1, 1)` / `(1, 16, 1, 1)` | yes (was `(1, 16, 1, kv_len)`) | 1.0000 | 8.71e-02 | **ok** |
| `c7__attn_probs` | `(1, 16, 1, 1)` / `(1, 16, 1, 1)` | yes (was `(1, 16, 1, kv_len)`) | 1.0000 | 0.00 | **ok** (softmax collapse to 1.0 confirmed) |
| `c7__attn_out` | `(1, 1, 4096)` / `(1, 1, 4096)` | yes | 1.0000 | 6.93e-02 | **ok** (was 0.986–0.996 pre-fix) |

Comparator output:

```
Phase C7 verdict: ALL SDPA taps match within cos_sim >= 0.9999.
  -> SDPA is NOT the dominant source of post_attn_raw drift at this bar.
```

Full report: [`c7_after_fix.txt`](./c7_after_fix.txt) (381 lines, 6 pair verdicts).

### C6 sweep (re-run for completeness, strict bar `cos_sim ≥ 0.9999`)

| Tap | median cos_sim | max max\|Δ\| | status |
| --- | --- | --- | --- |
| `c6__token_emb` | 1.0000 | 0.00 | **ok** |
| `c6__norm_emb` | 1.0000 | 7.76e-03 | **ok** |
| `c6__norm_h` | 1.0000 | 3.01e-02 | **ok** |
| `c6__concat` | 1.0000 | 3.01e-02 | **ok** |
| `c6__fused` | 1.0000 | 1.60e-02 | **ok** |

Comparator output:

```
Phase C6 verdict: ALL pre-RoPE taps match within cos_sim >= 0.9999.
  -> Pre-RoPE input is NOT the dominant source of pre-RoPE drift at this bar.
```

Full report: [`c6_after_fix.txt`](./c6_after_fix.txt) (379 lines, 6 pair verdicts).

### Cos_sim before / after

| Tap | C7 before fix (median, 6 pairs) | C7 after fix (median, 6 pairs) | Bar |
| --- | --- | --- | --- |
| `c7__attn_out` | **0.9926** (range 0.9857–0.9957) | **1.0000** | `≥ 0.9999` |
| `c7__pre_sdpa_k` | shape mismatch (kiln `(1, 4, kv_len, 256)` vs ref `(1, 4, 1, 256)`) | 1.0000 | shape parity + `≥ 0.9999` |
| `c7__pre_sdpa_v` | shape mismatch | 1.0000 | shape parity + `≥ 0.9999` |
| `c7__attn_scores_pre_softmax` | shape mismatch | 1.0000 | shape parity + `≥ 0.9999` |
| `c7__attn_probs` | shape mismatch | 1.0000 (max\|Δ\| = 0, mean\|Δ\| = 0) | shape parity + softmax = 1.0 |
| `c6__fused` (re-check) | 1.0000 | 1.0000 | `≥ 0.9999` |

## Out of scope: C9 follow-up

The C6/C7 comparator's overall verdict still reports `first divergence at tap 'fc_output'` for every pair. This is the **pre-existing main-model `mtp.fc` matmul drift** identified in Phase C6 (Class B, 87.6% of total `fc_output` drift). Phase C8 was scoped narrowly to the SDPA kv_len mismatch. The next phase, C9, addresses the `mtp.fc` weight transpose / layout question against a fresh bf16 vs fp32 reference and is out of scope for this PR.

The MTP acceptance rate (`α`) — which Phase C7 hypothesized would lift once the SDPA contract is correct — should be re-measured in C9 once the `fc_output` divergence is also resolved. The C8 fix is a pre-condition: speculative drafts that attend over the wrong key set cannot have meaningful α, regardless of `fc` correctness.

## Files added in this PR

- `docs/phase-c8/c8_fix_report.md` (this file)
- `docs/phase-c8/c7_after_fix.txt` (full C7 comparator log, 6 pairs)
- `docs/phase-c8/c6_after_fix.txt` (full C6 comparator log, 6 pairs)

## Code changes

- `crates/kiln-model/src/mtp_debug.rs` (+47 lines): TLS arm/disarm slot + accessors
- `crates/kiln-model/src/forward.rs` (+40 / −6 lines): single-site gating in `gqa_attention_paged` + arm/disarm bracket in `mtp_forward_step`
