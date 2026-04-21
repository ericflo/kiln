# Phase C7 — MTP SDPA-internal bisect (post-C6)

## Verdict

**The kiln MTP inner-block SDPA attends over a growing slice of the main-model paged KV cache (`kv_len = mtp_pos + 1`), while the HF reference MTP block performs single-token self-attention (`kv_len = 1`).** This is the root cause of the `post_attn_raw` divergence that C6 localized. `c7__pre_sdpa_q` matches bit-for-bit (cos_sim = 1.0) across all 6 pairs; `c7__pre_sdpa_k`, `c7__pre_sdpa_v`, `c7__attn_scores_pre_softmax`, and `c7__attn_probs` are **shape-incompatible** with the reference, and `c7__attn_out` is the first comparable tap that drops below `cos_sim ≥ 0.9999` (medians 0.986 – 0.996 across seeds and positions). No further SDPA-internal bisection is possible while the kiln and reference MTP attention operators disagree on what `K`/`V` are.

Phase C8 must therefore address **what kiln's MTP inner block is attending to**, not the internal SDPA math. The most probable fix is to make the MTP inner transformer operate in single-token-self-attn mode (no participation in the main-model paged KV cache) so that the reference and production paths attend to the same tokens.

## Scope

- **Target**: the 7 SDPA-internal taps inside the MTP inner transformer block (`:kiln/mtp/attn/sdpa/*`), emitted under the `c7__` prefix in the safetensors dumps shared with B6 / B11 / B12 / C6.
- **Build**: `cargo build --release --features cuda --bin kiln-bench` (sccache warm, 1m 54s) on a fresh A40 pod (SM 86, CUDA 12.4) from the Kiln RunPod image. A40 was used as the on-demand fallback after the A6000 pool briefly hit zero-capacity; A40 shares SM 86 with A6000 so the SDPA dispatch path is identical.
- **Instrumentation**: new flag `KILN_MTP_DUMP_C7_SDPA=1` arms a thread-local capture slot in `mtp_debug.rs`. The C7 hooks are gated inside the grouped-GQA-decode SDPA path in `forward.rs`; the fused `try_flash_attn_paged_decode` is bypassed when the capture is armed so the intermediates are actually materialised.
- **Coverage**: 3 seeds × 2 positions = 6 sample pairs. Seeds 42/43/44; positions `mtp_pos=1` and `mtp_pos=2`. Prompt 29 tokens, 32 output tokens, `--paged`, `--skip-training`, MTP enabled (`KILN_SPEC_ENABLED=1`, `KILN_SPEC_METHOD=mtp`).
- **Comparator**: `scripts/mtp_compare.py --c7` (strict `cos_sim ≥ 0.9999` bar, same as C6).
- **Dumps**: `/workspace/c7-dumps/{kiln,ref,cmp}/` on pod `5dnpt5zxqih7qf`. Full comparator log: `docs/phase-c7/c7_all.log` (382 lines, 6 pair verdicts + combined summary).

## Results

### Shape comparison (kiln vs HF reference, mtp_pos=2 representative)

| Tap | Kiln shape | Ref shape | Comparable? |
| --- | --- | --- | --- |
| `c7__pre_sdpa_q` | `(1, 16, 1, 256)` | `(1, 16, 1, 256)` | yes |
| `c7__pre_sdpa_k` | `(1, 4, 3, 256)` | `(1, 4, 1, 256)` | **no** (kv_len mismatch) |
| `c7__pre_sdpa_v` | `(1, 4, 3, 256)` | `(1, 4, 1, 256)` | **no** (kv_len mismatch) |
| `c7__causal_mask` | `()` (sentinel) | `()` (sentinel) | N/A (both are 0-d placeholders) |
| `c7__attn_scores_pre_softmax` | `(1, 16, 1, 3)` | `(1, 16, 1, 1)` | **no** (kv_len mismatch) |
| `c7__attn_probs` | `(1, 16, 1, 3)` | `(1, 16, 1, 1)` | **no** (kv_len mismatch) |
| `c7__attn_out` | `(1, 1, 4096)` | `(1, 1, 4096)` | yes |

The kiln `kv_len` equals `mtp_pos + 1` (2 at pos=1, 3 at pos=2), i.e. the MTP inner block is attending over the accepted main-model token plus every previously generated MTP draft token. The HF reference does a clean single-token self-attention; it re-projects the current `h_fused` into `q/k/v` and attends only to itself.

### Comparable-tap cos_sim summary (all 6 pairs, median collapse)

| Tap | pos=1 s42 | pos=2 s42 | pos=1 s43 | pos=2 s43 | pos=1 s44 | pos=2 s44 | median | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c7__pre_sdpa_q` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **ok** |
| `c7__attn_out` | 0.9921 | 0.9931 | 0.9933 | 0.9857 | 0.9868 | 0.9957 | 0.9926 | **DIVERGENT** |

(`pre_sdpa_k/v`, `attn_scores_pre_softmax`, `attn_probs` are omitted — they are shape-incompatible and the comparator emits `nan` for cos_sim in these rows. `causal_mask` is emitted as a 0-d zero placeholder on both sides because HF's native SDPA does not materialise a mask for single-token self-attention.)

### Interpretation

- `pre_sdpa_q` passes the strict bar. Everything up to and including Q's head-split + qk-norm + rotary-on-Q is byte-identical with the HF reference. C6 already confirmed pre-RoPE parity; C7 confirms the q-side of SDPA also matches.
- `pre_sdpa_k` / `pre_sdpa_v` are **not comparable** because kiln's K/V come from the paged KV cache while the reference's K/V are the single-token self-projections of the current `h_fused`. Kiln grows its kv slice as `mtp_pos` increments; HF does not.
- `attn_scores_pre_softmax` and `attn_probs` inherit that mismatch — they are `(1, 16, 1, kv_len)` on both sides but with different `kv_len`.
- `attn_out` is the first tap whose **shape is semantically comparable** (both reduce back to `(1, 1, num_heads · head_dim)`) and it diverges at a median cos_sim of **0.9926** — a 1.7 × 10⁻² mean \|Δ\| and 4.3 max \|Δ\|.

This matches C6's finding that `post_attn_raw` (the immediate consumer of `attn_out` in the v_split / gate / o_proj path) is the earliest sub-op below the bar.

## Most-likely cause

Kiln's MTP inner transformer block reuses the main-model paged KV cache machinery: when the scheduler issues an MTP draft step, the new K/V for the draft token are appended to the same physical KV pages, and the attention kernel reads the full slice. The HF reference implementation treats each MTP step as a fresh single-token self-attention — it never populates or reads a cross-token KV cache at the MTP layer.

This divergence is structural, not numerical:

1. **Causal semantics differ.** Kiln's MTP SDPA mixes in the representations of previous accepted tokens (plus previously drafted-but-not-accepted tokens if any). The HF reference does not.
2. **Effective context differs.** At `mtp_pos=P`, kiln attends to `P+1` key/value vectors; reference attends to `1`.
3. **Numerical attribution.** Even if every scalar operation were identical, the `attn_out` values would differ by an amount proportional to the contribution of the non-self keys — which is exactly what the `~0.01–0.16` mean \|Δ\| shows.

Both C6 (`post_attn_raw` first-divergent) and the Phase C2/C3 global drift profile are consistent with this. The earlier B12 `swap_fc_norms` analysis also hinted that the MTP block was being fed a state derived from cross-token statistics rather than pure self-attention — this phase now names the mechanism.

## What C8 should do

1. **Decide the reference contract.** Is kiln's current multi-token K/V attention the intended MTP semantics, or is HF's single-token self-attention the correct reference? The upstream Qwen3 reference implementation is the authority — confirm against `transformers` `Qwen3MTP*` (or the author's reference snippet) before changing either side.
2. **If HF is correct** — the expected path — make kiln's MTP inner transformer operate as pure single-token self-attention:
   - Do **not** append the MTP K/V to the main paged KV cache at the MTP layer.
   - Allocate a throw-away per-step K/V (or reuse a scratch tensor) of length 1.
   - Keep the fused `try_flash_attn_paged_decode` path for the main-model layers; only the MTP inner block needs the new behaviour.
   - Re-run the full C6 + C7 bisects. Expected: `post_attn_raw` drops to cos_sim = 1.0, `attn_out` drops to cos_sim = 1.0, overall MTP acceptance rate rises.
3. **If kiln is correct** and the reference is wrong, update `mtp_reference_dump.py` to populate a matching K/V cache (mirroring the main-model generation history up to `mtp_pos`) and re-run. The C6 / C7 bars will then measure SDPA numerics rather than attention semantics.

Either way, the C8 work is a single-site change (the MTP attention operator), and it should not require new tap instrumentation; the existing C7 taps are sufficient to verify the fix post-hoc.

## Artefacts

- Kiln dumps: `/workspace/c7-dumps/kiln/kiln_seed{42,43,44}_pos{1,2}.st` (pod `5dnpt5zxqih7qf`, A40)
- Reference dumps: `/workspace/c7-dumps/ref/ref_seed{42,43,44}_pos{1,2}.st`
- Full comparator log: `docs/phase-c7/c7_all.log` (382 lines)
- Comparator: `python3 scripts/mtp_compare.py --c7 --pair "mtp_pos=<n>_seed<s>:<kiln>.st,<ref>.st" [...]`

## Reproduction

```bash
# Build (A6000 or A40, SM 86, kiln-runpod image, sccache pointed at b2://.../kiln/)
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench

# Dump (emits c7__<name> entries alongside existing b6 / b11 / b12 / c6 taps)
export KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp
export KILN_MTP_DUMP_PRE_ROPE=1 KILN_MTP_DUMP_SUBOPS=1 KILN_MTP_DUMP_C7_SDPA=1
export KILN_MTP_DUMP_POS=1,2
export KILN_MTP_DUMP_PATH='/tmp/c7_kiln/kiln_seed42_pos{pos}.st'
./target/release/kiln-bench --model-path <qwen3.5-4b-dir>/ \
  --paged --prompt-tokens 29 --max-output-tokens 32 --skip-training --seed 42

# HF reference (re-uses the kiln dump for alignment metadata; --capture-subops
# also emits the B6 / B11 / B12 / C6 taps for side-by-side inspection)
python3 scripts/mtp_reference_dump.py \
  --checkpoint <qwen3.5-4b-dir>/ \
  --kiln-dump /tmp/c7_kiln/kiln_seed42_pos1.st \
  --out /tmp/c7_ref/ref_seed42_pos1.st --capture-subops

# Bisect
python3 scripts/mtp_compare.py --c7 \
  --pair "mtp_pos=1_seed42:/tmp/c7_kiln/kiln_seed42_pos1.st,/tmp/c7_ref/ref_seed42_pos1.st" \
  --pair "mtp_pos=2_seed42:/tmp/c7_kiln/kiln_seed42_pos2.st,/tmp/c7_ref/ref_seed42_pos2.st"
```

Exit code is non-zero when the B-phase allclose bar (`atol=1e-3`) flags downstream divergence — expected post-C6, not a failure of the C7 bisect itself. The C7-specific verdict is printed unconditionally.
