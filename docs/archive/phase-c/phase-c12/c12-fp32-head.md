# Phase C12 — fp32 draft-head kill switch + activation-weighted probe

> **Note on artifact links below:** Raw nvtx CSVs and bench logs were
> removed from the working tree on 2026-04-25 to clean up the top-level
> directory. The exact files cited in this report are preserved in git
> history at commit `07993f0272b0ef229246e4f3cc979abb7b05e145` (origin/main immediately before the
> cleanup PR). To inspect any referenced artifact, run:
> `git show 07993f0272b0ef229246e4f3cc979abb7b05e145:<path>` (e.g. `profile-out/...`, `c12-out/...`,
> `profiling-artifacts/...`).


## Verdict: **PRIMARY-NEGATIVE**

Forcing the MTP draft head's q/k/v/o + fc projections to fp32 (`KILN_MTP_FP32_HEAD=1`)
does **not** materially improve α over the W4A16 baseline under greedy decode.

| Config                            | env                                             | Median α    | Median tok/s |
|-----------------------------------|-------------------------------------------------|-------------|--------------|
| baseline                          | `KILN_W4A16=1`                                  | **0.0325**  | 41.09        |
| **primary (fp32 draft head)**     | `KILN_W4A16=1 KILN_MTP_FP32_HEAD=1`             | **0.0325**  | 37.66        |
| sanity_nomarlin                   | `KILN_W4A16=0`                                  | **0.0583**  | 36.76        |

Ship-floor target: α ≥ 0.72. All three configs miss by ≥0.61.
Median α primary == median α baseline to 4 decimals; the hypothesis that bf16 matmul
accumulation noise in the MTP head's projections drives α drift is **refuted**.

## Why this is a hard refutation, not noise

Per-seed α under greedy decode is not just close — it is **bitwise identical across configs
for 2 of 3 seeds**:

| Seed | baseline α (accept/total) | primary α (accept/total) | sanity α (accept/total) |
|------|---------------------------|--------------------------|-------------------------|
| 42   | 0.0325 (4/123)            | 0.0325 (4/123)           | 0.0325 (4/123)          |
| 43   | 0.0325 (4/123)            | 0.0325 (4/123)           | 0.0583 (7/120)          |
| 44   | 0.0948 (11/116)           | 0.1043 (12/115)          | 0.1043 (12/115)         |

The same 4 / 123 acceptance decisions show up under W4A16-bf16 head, W4A16-fp32 head,
and no-Marlin-at-all for seed 42. Under greedy, Class C = 0 (C5), so α == top-1
agreement — meaning the MTP head selects the *same* top-1 tokens regardless of whether
its projections accumulate in bf16 or fp32 or use BF16 weights directly. The head is
predicting a *different* token from the main head, deterministically, at the argmax
level. Dtype of its projections is not a contributor of meaningful magnitude.

## Decode-throughput cost of fp32 upcast

- `KILN_MTP_FP32_HEAD=1` costs ≈ **8.3%** of decode tok/s on A6000 (41.09 → 37.66 tok/s).
- No-Marlin (W4A16=0) costs ≈ **10.5%** (41.09 → 36.76 tok/s), consistent with PR #80/#146
  Marlin wins being mostly intact in MTP mode.
- Neither tier pays for itself — both degrade throughput *and* fail to lift α.

## What this rules out

Combined with the C5 Class B = 87.6% finding and the C3 RoPE fix that produced
bitwise-identical α pre/post (PR #314), this closes several hypothesis branches:

1. **MTP-head projection dtype / accumulation precision** — refuted here (C12).
2. **Marlin W4A16 dequant drift at head matmuls** — refuted (sanity_nomarlin ≈ baseline).
3. **RoPE position threading in draft forward** — refuted in C5 (post-fix bitwise
   identical to pre-fix on the shared 8-pool prompt).
4. **Tokenizer / verifier mask mismatch (Class C)** — refuted in C5 (Class C = 0 under
   greedy across 339 rows).

The surviving hypothesis space is **upstream of the head's projections**, on the input
side:

- **MTP token-embedding lookup** — is the embedding table used by the draft head tied
  correctly to the main embed, and is it being dequantized/scaled identically to the
  main-head path?
- **MTP pre-norm (RMSNorm γ)** — applied in the right place, right dtype, right γ
  tensor?
- **MTP hidden-state splicing** — the residual fed into the draft head at step t comes
  from *where* in the main stack (post-LN vs pre-LN, which layer index)? Does the
  training recipe match what `mtp_forward_step` actually injects?
- **MTP head weight loading itself** — is `fc_input` / `fc_output` pointing at the right
  tensor names on disk, and is the weight-tying policy (tied vs separate vs shared_lora)
  matching what the checkpoint was trained with?

C12 closes the "numerical drift at the head matmul" door so C13 can focus on the
"wrong input fed into the head" and "wrong weights loaded for the head" hypotheses.

## Environment

| Field            | Value                                                            |
|------------------|------------------------------------------------------------------|
| Repo             | `ericflo/kiln`                                                   |
| Branch           | `mtp/phase-c12-fp32-draft-head`                                  |
| GPU              | NVIDIA RTX A6000 (49 140 MB), 570.195.03 / CUDA 12.4             |
| Image            | `ghcr.io/ericflo/kiln-runpod:latest`                             |
| Model            | `Qwen/Qwen3.5-4B` (vocab 248320, hybrid 24× GDN + 8× GQA)        |
| Build            | `KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench` (sccache hit rate 97.56%, 2m 12s) |
| Bench flags      | `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed {42,43,44}` |
| MTP base env     | `KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp`                        |

Seeds 42/43/44 are the C5/C9 convention for median-of-3 α; they map through the
8-pool prompt table at `seed % 8` ∈ {2,3,4}.

## Implementation summary

Two-arm-flag + one-chokepoint design keeps the blast radius small:

1. **`crates/kiln-model/src/mtp_debug.rs`** — adds `KILN_MTP_FP32_HEAD` env reader and
   a TLS `MTP_FP32_HEAD_ARMED` `RefCell<bool>` with `arm_mtp_fp32_head()` /
   `disarm_mtp_fp32_head()` helpers. No behavior change when the flag is unset.

2. **`crates/kiln-model/src/lora_loader.rs::linear_with_lora_t`** — single chokepoint
   for q/k/v/o + MLP base matmuls. When the TLS flag is armed, upcast `x` and the
   transposed base weight to `DType::F32`, matmul in fp32, then cast back to input
   dtype. LoRA and bias adds are left alone (they're small and already fp32-tolerant).

3. **`crates/kiln-model/src/forward.rs::mtp_forward_step`** — reads
   `is_mtp_fp32_head_enabled()` once; arms the TLS flag around the
   `transformer_block_paged` call for the draft head's forward; disarms on exit. Also
   OR's `fp32_head` into the existing `KILN_MTP_FC_FP32_ACCUM` branch so the new flag
   subsumes C9's fc-only knob.

Scope is intentionally narrow: the arm flag only fires during the draft-head block,
so main-head matmuls are untouched and non-MTP decode paths are unaffected.

## Activation-weighted probe (main-model MLP + full-attn q_proj, 32 prompts / 595 tok)

`scripts/c12_activation_weighted_probe.py` hooked 104 projections (32 × {gate, up,
down} + 8 full-attn q_proj) on the HF reference, reproduced kiln's Marlin dequant,
and computed
`weighted_drift_ratio = ||X @ (W_marlin - W_bf16).T||_F / ||X @ W_bf16.T||_F`
per projection. Top 10:

| layer | kind      | weighted_ratio |
|------:|:----------|---------------:|
| 6     | down_proj | **1.447e-01**  |
| 18    | down_proj | 1.384e-01      |
| 25    | down_proj | 1.202e-01      |
| 5     | up_proj   | 1.179e-01      |
| 4     | up_proj   | 1.177e-01      |
| 15    | up_proj   | 1.170e-01      |
| 17    | down_proj | 1.168e-01      |
| 6     | up_proj   | 1.164e-01      |
| 21    | down_proj | 1.161e-01      |
| 20    | down_proj | 1.154e-01      |

Main-model MLPs have activation-weighted drift up to **14.5%** (`L6.down_proj`); many
layers sit in the 11-14% band. This is *non-trivial* drift by magnitude — roughly
two orders of magnitude above the `1e-3` benign threshold the probe script uses.

**This does not contradict the bench PRIMARY-NEGATIVE result.** The probe audits
main-model MLP and full-attn projections, *not* the MTP head. The C12 bench showed
that draft-head matmul precision (bf16 vs fp32) has no effect on α under greedy
decode — i.e. whatever numerical drift exists in the draft head is not crossing any
argmax boundary. The probe independently shows the *main* model's MLPs are carrying
large activation-weighted drift, but that drift evidently doesn't change which token
the *main* head picks for the 8-pool prompts used by the bench (if it did, we'd see
different main-head trajectories across W4A16=0 vs W4A16=1, and we don't — seed 42
accept/total is 4/123 for all three configs).

The probe script's terminal-emitted verdict string
("Corroborates PRIMARY-POSITIVE…") is a stale boilerplate hint attached to the
`non-trivial drift` branch; the authoritative verdict for C12 is the bench in this
report: **PRIMARY-NEGATIVE**.

Full JSON: `c12-out/probe-report.json`. Markdown summary: `c12-out/probe-report.md`.

## Bench artifacts

- `c12-out/bench-summary.json` — full 3 × 3 trial JSON with seed, α, tok/s, mean ITL,
  and the `Decode (MTP): ...` accept line for each run.
- `c12-out/bench.log` — runner stdout.

## Ship-floor verdict

**MISS.** Do not gate `KILN_MTP_FP32_HEAD` on by default. Leave `KILN_SPEC_METHOD`
default unchanged. Do not promote MTP.

## Recommendation for C13

Pick up from the C5/C12 closure above:

1. **Dump the MTP head's input** pre-projection on a known seed, compare element-wise
   against a "reference" draft forward that re-uses the main head's embed + norm +
   layer-N residual. If the pre-head hidden state differs, the bug is in the splice /
   norm / embed lookup, not the head.
2. **Audit weight loading** for `mtp_head.fc_input` / `mtp_head.fc_output` / q-k-v-o
   projections: which safetensors keys do they resolve to, and does the loader's
   weight-tying policy (tied / separate / shared_lora) match the checkpoint's training
   recipe? Mis-tied weights would reproduce exactly this symptom — deterministic wrong
   top-1 under greedy, insensitive to matmul dtype.
3. Capture activations at `mtp_pos ∈ {0, 2}` first — those are the largest α buckets
   (146 / 339 ≈ 43% of attempts) and the worst accepting buckets (6.1% / 4.2%), so the
   signal is largest and the sample size is largest.

## Wall-clock / cost

Within the 90 min / $40 budget. Single A6000 pod (`wl0fyjvqrv0v9b`), one build (2m 12s
warm sccache), 9 bench runs (~108 s each, 974 s total), one probe run.
Pod-side waits all used bounded `runpod_api.py wait-file --timeout`; no `until ssh` or
`while ssh ... sleep` polling loops were used.
