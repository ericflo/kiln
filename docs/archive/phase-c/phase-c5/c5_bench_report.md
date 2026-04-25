# Phase C5 — Bench α + decode tok/s after C3 RoPE fix

## Summary

**Ship-floor: MISS on both criteria.**

| Criterion                 | Floor    | Measured (median) | Result |
|---------------------------|----------|-------------------|--------|
| Draft acceptance rate α   | ≥ 0.72   | **0.124**         | MISS   |
| Decode tok/s gain vs Off  | ≥ +10%   | **−25.1 %**       | MISS   |

MTP On is **25 % slower** than MTP Off and accepts **12.4 %** of draft attempts in aggregate (42 / 339 across three seeds). The C3 RoPE position threading fix (PR #314) did **not** change acceptance behavior: seed 2 / seed 42 map to the same 8-pool prompt, and both runs produced α = 6 / 121 = 4.96 % — bitwise identical outcome.

**Verdict: C6 investigates MTP head embedding/norm/proj path feeding into the now-correct RoPE.** Class B (mtp_top1 ≠ main_top1) accounts for 297 / 339 = 87.6 % of rows with Class C (mask/tokenizer mismatch) at 0 %, so the failure mode is the head predicting a different token, not an accounting/routing bug.

## Environment

| Field            | Value                                                 |
|------------------|-------------------------------------------------------|
| Repo             | `ericflo/kiln`                                         |
| Git SHA          | `f8792b91af8dd60f95701707a0752ca7607c0993` (main post PR #314) |
| Branch           | `mtp/phase-c5-bench`                                   |
| GPU              | NVIDIA RTX A6000 (49 140 MB)                           |
| Driver / CUDA    | 550.127.08 / 12.4.131                                  |
| Image            | `ghcr.io/ericflo/kiln-runpod:latest`                   |
| Model            | Qwen3.5-4B (2 safetensors shards, 4 206 M params)      |
| Build            | `cargo build --release --features cuda --bin kiln-bench` (sccache warm, ~50 s) |
| Bench flags      | `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed {42,43,44}` |

## Results — Off baseline (3 seeds)

| Seed | Prefill ms | Prefill tok/s | Decode tok/s | p50 ITL ms | p99 ITL ms |
|------|------------|---------------|--------------|------------|------------|
| 42   | 330.5      | 1 531         | 43.5         | 22.890     | 24.496     |
| 43   | 325.9      | 1 552         | 43.5         | 22.883     | 23.457     |
| 44   | 326.3      | 1 551         | 43.5         | 22.883     | 23.561     |
| **median** | **326.3** | **1 551** | **43.5**    | **22.883** | **23.561** |

## Results — MTP On (3 seeds)

| Seed | Prefill ms | Decode tok/s | p50 ITL ms | p99 ITL ms | α        |
|------|------------|--------------|------------|------------|----------|
| 42   | 331.1      | 31.9         | 31.570     | 35.503     | 0.0496 (6/121)   |
| 43   | 327.8      | 32.6         | 31.390     | 33.752     | 0.1239 (14/113)  |
| 44   | 333.5      | 33.1         | 31.424     | 33.802     | 0.2095 (22/105)  |
| **median** | **331.1** | **32.6** | **31.424** | **33.802** | **0.1239**       |

Decode-tok/s delta vs Off median: **32.6 / 43.5 − 1 = −25.1 %**.

## α breakdown by mtp_pos (aggregate over 3 seeds)

| mtp_pos | accepted / total | α       |
|---------|------------------|---------|
| 0       | 3 / 49           |  6.12 % |
| 1       | 3 / 25           | 12.00 % |
| 2       | 3 / 72           |  4.17 % |
| 3       | 3 / 20           | 15.00 % |
| 4       | 3 / 27           | 11.11 % |
| 5       | 3 / 21           | 14.29 % |
| 6       | 2 / 22           |  9.09 % |
| 7       | 2 / 23           |  8.70 % |
| 8       | 2 / 14           | 14.29 % |
| 9       | 2 / 7            | 28.57 % |
| 10      | 2 / 12           | 16.67 % |
| 11      | 2 / 10           | 20.00 % |
| 12      | 2 / 2            | 100.00 %|
| 13      | 2 / 5            | 40.00 % |
| 14      | 1 / 12           |  8.33 % |
| 15–22   | 1 / small-n       | noisy (n ≤ 4) |

Early positions (pos ∈ {0, 1, 2}) account for 146 / 339 = 43 % of attempts and accept at 9 / 146 = 6.2 %. Late positions (pos ≥ 9) aggregate at 13 / 61 = 21 %, but sample counts are tiny (several buckets have n ≤ 4) so the apparent curve is dominated by variance.

## Class breakdown (Class A / B / C)

- Class A (`mtp_top1 == main_top1` **and** accepted) = 42 / 339 = 12.4 %
- Class B (`mtp_top1 != main_top1`) = 297 / 339 = **87.6 %**
- Class C (`mtp_top1 == main_top1` **but not** accepted) = 0 / 339 = 0 %

Under greedy decode, Class C == 0 confirms the verifier is not dropping top-1 matches for mask/tokenizer reasons — every acceptance corresponds to a genuine top-1 agreement between the MTP head and the main head. The failure mode is **the MTP head predicting a different top-1 token** almost 9 times out of 10, even after the C3 RoPE position threading fix.

## Did PR #314 (C3 RoPE fix) move the needle?

No. C1 seed=2 and C5 seed=42 hit the same 8-pool prompt (`seed % 8` == 2):

| Phase | α       | accepted/total |
|-------|---------|----------------|
| C1 seed=2 (pre-fix) | 4.96 % | 6 / 121 |
| C5 seed=42 (post-fix) | 4.96 % | 6 / 121 |

The fix was correct and necessary (positions must be threaded to RoPE in the reference path) but was not sufficient on its own to recover α. The drift is upstream of RoPE — in the path feeding hidden states into the MTP head.

## Ship-floor verdict

**MISS.** Do not promote MTP to default. Leave `KILN_SPEC_METHOD` default unchanged.

## Recommendation for C6

C6 should bisect the MTP embedding / norm / projection path feeding into the (now correct) RoPE. Class B at 87.6 % means the head's **pre-RoPE** input distribution still differs from what the head was trained against. Priorities, roughly in order of cost:

1. **MTP token-embedding lookup.** Confirm the embedding table used by the draft head matches the one used by the main head (weight tie vs separate tensor, dtype, dequant scale).
2. **MTP pre-norm (RMSNorm) application.** Verify the norm γ is applied in the right place and with the right dtype; mismatched norm can shift the input magnitude enough to derail top-1 across a wide vocab.
3. **MTP hidden-state splicing.** The draft head conditions on the main-head residual from step t-1; confirm residual extraction (final pre-LN vs post-LN, which layer index) matches the training-time recipe.
4. **MTP projection (`in_proj` / `out_proj`) dtype + scale.** If the Marlin-packed weights are used for the head, verify pack/unpack matches FP path within tolerance on a single sample.

Recommended instrumentation for C6: dump pre-RoPE Q/K for a handful of steps in both (a) the MTP draft forward and (b) a "reference" draft forward that reuses the main head's stack and compare element-wise. Divergence at entry to RoPE identifies the wrong stage; divergence that appears only after embed lookup pins item 1 specifically.

Early-pos targeting: since pos ∈ {0, 2} are the worst buckets (6.12 % / 4.17 %), C6 should capture activations at those buckets first — the signal is largest and the sample size is largest.

## Wall-clock / cost

Within the 90 min / $40 budget. Single A6000 pod (`2csb4xfv8bp1jf`), six bench runs (3 Off + 3 MTP) plus build plus weights download. No SSH polling loops; all pod-side waits used `runpod_api.py wait-file --timeout`.
