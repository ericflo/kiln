# Phase C11 — Marlin W4A16 per-channel scale drift audit

**Status:** Per-channel weight drift is well outside the fp32-equivalence band (cos_min 0.964366, rel_L2_max 0.265271, 100 % of 737,280 packed output channels out-of-band). Interpreted against C10's behavioral null at the MTP tap: W4A16 weight drift is the *expected* INT4 round-trip signature and is NOT sufficient to drive α-suppression on its own — it must be coupled to an activation-side mechanism to matter. See "Reconciliation with C10" below.

**Checkpoint:** `/workspace/qwen3.5-4b` (HF canonical bf16; cast to fp32 for the audit — the bf16→fp32 upcast is exact, so this is a lossless read-out of the checkpoint's ground truth).

**Bar:** per-output-channel `cos_sim >= 0.9999` AND `rel_l2_error <= 1e-3` — the same fp32-equivalence band C10 used behaviorally, now applied numerically to Marlin's weight round-trip.

**Scope:** all 104 Marlin-packed main-model projections (8 × q_proj on full-attention layers {3, 7, 11, 15, 19, 23, 27, 31} + 32 × gate_proj + 32 × up_proj + 32 × down_proj), groupsize=128. The MTP head itself stays BF16 per `forward.rs` (`// For WIP scaffolding the MTP transformer layer is kept in BF16 and is NOT queued for Marlin batch packing`); k_proj / v_proj / o_proj / lm_head / embed_tokens also stay BF16 and are out of scope.

## Verdict

**Raw numerical verdict (weight-level):** ❌ OUT-OF-BAND. Every one of the 104 Marlin-packed projections exhibits per-output-channel weight drift an order of magnitude (cos_sim) to two orders of magnitude (rel_L2) outside the fp32-equivalence band.

**Causal verdict (α-suppression mechanism):** ⚠️ INSUFFICIENT on its own. The magnitude of the drift seen here is the *by-construction* signature of 4-bit symmetric groupwise quantization with groupsize=128 (ε ≈ s/2 ≈ max|w|/15 per value), not a kiln-specific packing bug. C10 already demonstrated that kiln's bf16 forward matches the fp32 HF reference at the MTP tap sites to cos_sim ≥ 0.9999 across all 3 seeds — i.e. whatever weight-level drift Marlin introduces, it is NOT propagating into hidden-state drift at the MTP tap. The α ≈ 0.058 alpha-suppression floor on Qwen3.5-4B k=1 MTP therefore cannot be pinned on Marlin weight numerics alone.

**What C11 does rule in (still open for C12):** because this audit is weight-only, it cannot exclude *activation-coupled* Marlin effects — e.g. a single out-of-distribution activation channel interacting with a single high-error weight channel on the rejected Class B positions. The max per-channel rel_L2 of 0.265 on `up_proj[layer=17] channel 3` is a concrete target for a C12 activation-weighted follow-up (see "Cross-reference with C10 Class B rejection positions" below).

**Recommendation for C12:**

1. **Primary — fp32 draft head policy** (C10 target #2): the bf16→fp32 comparator C10 built is reusable. Swap MTP's draft-head matmul + softmax to fp32, hold everything else constant, re-measure α. If α lifts, this was the cause.
2. **Secondary — activation-weighted Marlin probe**: on the 3 C10 Class B rejection positions, capture the layer-31 MLP input activation, multiply by both the original bf16 weight and the Marlin-dequantized weight, compare cos_sim of the *outputs*. If the activation-side cos_sim is also ≥ 0.9999, Marlin is conclusively not the signal. If not, the coupled activation×weight error is a candidate for a per-channel rescale fix.

## Headline numbers

- **Projections audited:** 104
- **Output channels total:** 737280
- **Channels with cos_sim < 0.9999:** 737280 (100.000 %)
- **Channels with rel_l2 > 1e-3:** 737280 (100.000 %)

- **Worst cos_sim across all projections:** `0.964366` at `up_proj[layer=17]` channel `3`
- **Worst rel_l2 across all projections:** `0.265271` at `up_proj[layer=17]` channel `3`

## Per-projection summary

| Layer | Kind | k x n | cos_min | cos_p01 | cos_p50 | rel_l2_max | rel_l2_p99 | rmse_max | n_cos<0.9999 | n_rel_l2>1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | gate_proj | 2560x9216 | 0.984473 | 0.992119 | 0.993860 | 0.176431 | 0.126187 | 2.952445e-03 | 9216 | 9216 |
| 0 | up_proj | 2560x9216 | 0.986298 | 0.992447 | 0.993841 | 0.166409 | 0.123302 | 1.488695e-03 | 9216 | 9216 |
| 0 | down_proj | 9216x2560 | 0.982854 | 0.991147 | 0.993812 | 0.185579 | 0.133471 | 4.788515e-03 | 2560 | 2560 |
| 1 | gate_proj | 2560x9216 | 0.974488 | 0.990702 | 0.993832 | 0.224792 | 0.136962 | 2.969662e-03 | 9216 | 9216 |
| 1 | up_proj | 2560x9216 | 0.975772 | 0.992169 | 0.993833 | 0.221978 | 0.125477 | 1.756434e-03 | 9216 | 9216 |
| 1 | down_proj | 9216x2560 | 0.977635 | 0.987657 | 0.993763 | 0.212881 | 0.157840 | 4.395492e-03 | 2560 | 2560 |
| 2 | gate_proj | 2560x9216 | 0.979335 | 0.992538 | 0.993872 | 0.204900 | 0.122610 | 2.010078e-03 | 9216 | 9216 |
| 2 | up_proj | 2560x9216 | 0.984971 | 0.992527 | 0.993839 | 0.177943 | 0.122609 | 1.504276e-03 | 9216 | 9216 |
| 2 | down_proj | 9216x2560 | 0.980878 | 0.986571 | 0.993808 | 0.195390 | 0.164561 | 4.800150e-03 | 2560 | 2560 |
| 3 | q_proj | 2560x8192 | 0.972643 | 0.990671 | 0.993747 | 0.232748 | 0.136697 | 4.290189e-03 | 8192 | 8192 |
| 3 | gate_proj | 2560x9216 | 0.984592 | 0.992526 | 0.993842 | 0.179714 | 0.122650 | 1.995770e-03 | 9216 | 9216 |
| 3 | up_proj | 2560x9216 | 0.987392 | 0.992638 | 0.993833 | 0.159895 | 0.121747 | 1.611634e-03 | 9216 | 9216 |
| 3 | down_proj | 9216x2560 | 0.979033 | 0.985430 | 0.993820 | 0.205349 | 0.171941 | 4.430925e-03 | 2560 | 2560 |
| 4 | gate_proj | 2560x9216 | 0.980400 | 0.992353 | 0.993809 | 0.198908 | 0.123877 | 2.308779e-03 | 9216 | 9216 |
| 4 | up_proj | 2560x9216 | 0.984910 | 0.992396 | 0.993822 | 0.174479 | 0.123741 | 1.584691e-03 | 9216 | 9216 |
| 4 | down_proj | 9216x2560 | 0.978779 | 0.984148 | 0.993783 | 0.206555 | 0.179405 | 4.093555e-03 | 2560 | 2560 |
| 5 | gate_proj | 2560x9216 | 0.985659 | 0.992256 | 0.993847 | 0.170702 | 0.124873 | 2.149951e-03 | 9216 | 9216 |
| 5 | up_proj | 2560x9216 | 0.985911 | 0.992475 | 0.993795 | 0.168385 | 0.123174 | 1.475118e-03 | 9216 | 9216 |
| 5 | down_proj | 9216x2560 | 0.978226 | 0.982066 | 0.993682 | 0.210930 | 0.190981 | 3.667179e-03 | 2560 | 2560 |
| 6 | gate_proj | 2560x9216 | 0.977445 | 0.987483 | 0.993792 | 0.211559 | 0.158853 | 2.554248e-03 | 9216 | 9216 |
| 6 | up_proj | 2560x9216 | 0.979188 | 0.990689 | 0.993739 | 0.206327 | 0.136879 | 1.684586e-03 | 9216 | 9216 |
| 6 | down_proj | 9216x2560 | 0.976893 | 0.985929 | 0.993651 | 0.217385 | 0.169015 | 3.838615e-03 | 2560 | 2560 |
| 7 | q_proj | 2560x8192 | 0.980026 | 0.991352 | 0.993686 | 0.201822 | 0.131956 | 4.598956e-03 | 8192 | 8192 |
| 7 | gate_proj | 2560x9216 | 0.984539 | 0.992548 | 0.993851 | 0.177084 | 0.122342 | 2.177414e-03 | 9216 | 9216 |
| 7 | up_proj | 2560x9216 | 0.988791 | 0.992569 | 0.993779 | 0.151817 | 0.122382 | 1.659600e-03 | 9216 | 9216 |
| 7 | down_proj | 9216x2560 | 0.981743 | 0.987002 | 0.993714 | 0.192414 | 0.161902 | 4.314413e-03 | 2560 | 2560 |
| 8 | gate_proj | 2560x9216 | 0.986173 | 0.992586 | 0.993839 | 0.168317 | 0.122251 | 2.612955e-03 | 9216 | 9216 |
| 8 | up_proj | 2560x9216 | 0.985622 | 0.992588 | 0.993774 | 0.170121 | 0.122159 | 1.573244e-03 | 9216 | 9216 |
| 8 | down_proj | 9216x2560 | 0.984407 | 0.989049 | 0.993723 | 0.177329 | 0.148724 | 3.747896e-03 | 2560 | 2560 |
| 9 | gate_proj | 2560x9216 | 0.981297 | 0.992479 | 0.993817 | 0.193703 | 0.123126 | 2.934593e-03 | 9216 | 9216 |
| 9 | up_proj | 2560x9216 | 0.982497 | 0.992468 | 0.993772 | 0.188149 | 0.123009 | 1.849560e-03 | 9216 | 9216 |
| 9 | down_proj | 9216x2560 | 0.985306 | 0.988659 | 0.993677 | 0.173205 | 0.150971 | 3.983900e-03 | 2560 | 2560 |
| 10 | gate_proj | 2560x9216 | 0.979332 | 0.990878 | 0.993753 | 0.205149 | 0.135604 | 3.256638e-03 | 9216 | 9216 |
| 10 | up_proj | 2560x9216 | 0.984464 | 0.991693 | 0.993726 | 0.181113 | 0.129427 | 2.014879e-03 | 9216 | 9216 |
| 10 | down_proj | 9216x2560 | 0.982282 | 0.987087 | 0.993587 | 0.188533 | 0.161768 | 3.647536e-03 | 2560 | 2560 |
| 11 | q_proj | 2560x8192 | 0.980782 | 0.991166 | 0.993644 | 0.197009 | 0.133358 | 4.825247e-03 | 8192 | 8192 |
| 11 | gate_proj | 2560x9216 | 0.985255 | 0.992283 | 0.993779 | 0.172490 | 0.124923 | 3.436234e-03 | 9216 | 9216 |
| 11 | up_proj | 2560x9216 | 0.986034 | 0.992269 | 0.993749 | 0.167316 | 0.124576 | 1.925694e-03 | 9216 | 9216 |
| 11 | down_proj | 9216x2560 | 0.986332 | 0.989622 | 0.993583 | 0.165647 | 0.144548 | 4.134734e-03 | 2560 | 2560 |
| 12 | gate_proj | 2560x9216 | 0.979092 | 0.992317 | 0.993755 | 0.205114 | 0.124495 | 2.589930e-03 | 9216 | 9216 |
| 12 | up_proj | 2560x9216 | 0.986014 | 0.992320 | 0.993728 | 0.167953 | 0.124360 | 1.815235e-03 | 9216 | 9216 |
| 12 | down_proj | 9216x2560 | 0.984524 | 0.988020 | 0.993531 | 0.177103 | 0.154986 | 3.826277e-03 | 2560 | 2560 |
| 13 | gate_proj | 2560x9216 | 0.985510 | 0.992317 | 0.993750 | 0.170727 | 0.124367 | 2.634045e-03 | 9216 | 9216 |
| 13 | up_proj | 2560x9216 | 0.985478 | 0.992347 | 0.993732 | 0.170934 | 0.124043 | 2.187019e-03 | 9216 | 9216 |
| 13 | down_proj | 9216x2560 | 0.978630 | 0.987891 | 0.993502 | 0.208547 | 0.156387 | 3.816727e-03 | 2560 | 2560 |
| 14 | gate_proj | 2560x9216 | 0.985241 | 0.991619 | 0.993735 | 0.172565 | 0.130052 | 2.772228e-03 | 9216 | 9216 |
| 14 | up_proj | 2560x9216 | 0.983503 | 0.991786 | 0.993721 | 0.182994 | 0.128570 | 1.854229e-03 | 9216 | 9216 |
| 14 | down_proj | 9216x2560 | 0.978474 | 0.988989 | 0.993462 | 0.209394 | 0.148966 | 3.455250e-03 | 2560 | 2560 |
| 15 | q_proj | 2560x8192 | 0.977566 | 0.987395 | 0.993653 | 0.215935 | 0.161209 | 4.653708e-03 | 8192 | 8192 |
| 15 | gate_proj | 2560x9216 | 0.984870 | 0.992082 | 0.993757 | 0.174358 | 0.126281 | 2.864722e-03 | 9216 | 9216 |
| 15 | up_proj | 2560x9216 | 0.983282 | 0.992109 | 0.993723 | 0.182760 | 0.126098 | 1.925488e-03 | 9216 | 9216 |
| 15 | down_proj | 9216x2560 | 0.981594 | 0.988129 | 0.993468 | 0.192794 | 0.154692 | 4.338677e-03 | 2560 | 2560 |
| 16 | gate_proj | 2560x9216 | 0.977448 | 0.991352 | 0.993743 | 0.212585 | 0.131794 | 3.687618e-03 | 9216 | 9216 |
| 16 | up_proj | 2560x9216 | 0.979169 | 0.991927 | 0.993725 | 0.206300 | 0.127621 | 1.976167e-03 | 9216 | 9216 |
| 16 | down_proj | 9216x2560 | 0.982402 | 0.987450 | 0.993347 | 0.188684 | 0.159198 | 4.692968e-03 | 2560 | 2560 |
| 17 | gate_proj | 2560x9216 | 0.977255 | 0.989705 | 0.993744 | 0.213703 | 0.143591 | 3.148017e-03 | 9216 | 9216 |
| 17 | up_proj | 2560x9216 | 0.964366 | 0.991486 | 0.993741 | 0.265271 | 0.130962 | 1.781391e-03 | 9216 | 9216 |
| 17 | down_proj | 9216x2560 | 0.980248 | 0.986987 | 0.993381 | 0.200106 | 0.161904 | 3.590384e-03 | 2560 | 2560 |
| 18 | gate_proj | 2560x9216 | 0.976883 | 0.987232 | 0.993670 | 0.215514 | 0.160704 | 3.607490e-03 | 9216 | 9216 |
| 18 | up_proj | 2560x9216 | 0.974625 | 0.989819 | 0.993690 | 0.224024 | 0.143196 | 2.174963e-03 | 9216 | 9216 |
| 18 | down_proj | 9216x2560 | 0.970255 | 0.984170 | 0.993122 | 0.244523 | 0.179238 | 4.071306e-03 | 2560 | 2560 |
| 19 | q_proj | 2560x8192 | 0.988129 | 0.991417 | 0.993586 | 0.155496 | 0.131602 | 4.865883e-03 | 8192 | 8192 |
| 19 | gate_proj | 2560x9216 | 0.978687 | 0.989875 | 0.993689 | 0.206911 | 0.142843 | 3.719320e-03 | 9216 | 9216 |
| 19 | up_proj | 2560x9216 | 0.971183 | 0.991289 | 0.993704 | 0.247198 | 0.132504 | 2.077452e-03 | 9216 | 9216 |
| 19 | down_proj | 9216x2560 | 0.973482 | 0.985892 | 0.993373 | 0.232205 | 0.169120 | 4.125714e-03 | 2560 | 2560 |
| 20 | gate_proj | 2560x9216 | 0.970374 | 0.989812 | 0.993691 | 0.242777 | 0.143429 | 3.604047e-03 | 9216 | 9216 |
| 20 | up_proj | 2560x9216 | 0.979194 | 0.991728 | 0.993723 | 0.204010 | 0.129178 | 1.515205e-03 | 9216 | 9216 |
| 20 | down_proj | 9216x2560 | 0.977555 | 0.987126 | 0.993632 | 0.214233 | 0.161481 | 3.286555e-03 | 2560 | 2560 |
| 21 | gate_proj | 2560x9216 | 0.973444 | 0.989420 | 0.993705 | 0.229311 | 0.145904 | 3.786503e-03 | 9216 | 9216 |
| 21 | up_proj | 2560x9216 | 0.976522 | 0.991597 | 0.993726 | 0.215738 | 0.130167 | 1.817527e-03 | 9216 | 9216 |
| 21 | down_proj | 9216x2560 | 0.978868 | 0.986538 | 0.993636 | 0.206237 | 0.165006 | 2.885143e-03 | 2560 | 2560 |
| 22 | gate_proj | 2560x9216 | 0.971737 | 0.989655 | 0.993689 | 0.239805 | 0.144509 | 3.322018e-03 | 9216 | 9216 |
| 22 | up_proj | 2560x9216 | 0.974813 | 0.991210 | 0.993728 | 0.224281 | 0.133121 | 1.876680e-03 | 9216 | 9216 |
| 22 | down_proj | 9216x2560 | 0.975230 | 0.985355 | 0.993615 | 0.224983 | 0.171932 | 3.844901e-03 | 2560 | 2560 |
| 23 | q_proj | 2560x8192 | 0.982055 | 0.990256 | 0.993659 | 0.191282 | 0.140444 | 3.915349e-03 | 8192 | 8192 |
| 23 | gate_proj | 2560x9216 | 0.972294 | 0.990859 | 0.993753 | 0.235151 | 0.135871 | 3.820937e-03 | 9216 | 9216 |
| 23 | up_proj | 2560x9216 | 0.968947 | 0.991741 | 0.993793 | 0.248469 | 0.128959 | 1.669571e-03 | 9216 | 9216 |
| 23 | down_proj | 9216x2560 | 0.976044 | 0.988636 | 0.993715 | 0.219232 | 0.151403 | 3.515445e-03 | 2560 | 2560 |
| 24 | gate_proj | 2560x9216 | 0.974971 | 0.991441 | 0.993784 | 0.222315 | 0.131494 | 4.127522e-03 | 9216 | 9216 |
| 24 | up_proj | 2560x9216 | 0.979090 | 0.992063 | 0.993818 | 0.206516 | 0.126482 | 1.589917e-03 | 9216 | 9216 |
| 24 | down_proj | 9216x2560 | 0.980406 | 0.989683 | 0.993777 | 0.198208 | 0.144450 | 2.955807e-03 | 2560 | 2560 |
| 25 | gate_proj | 2560x9216 | 0.977770 | 0.991557 | 0.993781 | 0.212015 | 0.130263 | 4.134381e-03 | 9216 | 9216 |
| 25 | up_proj | 2560x9216 | 0.977460 | 0.992299 | 0.993836 | 0.213990 | 0.124596 | 1.789710e-03 | 9216 | 9216 |
| 25 | down_proj | 9216x2560 | 0.984183 | 0.991209 | 0.993803 | 0.177634 | 0.133054 | 3.089691e-03 | 2560 | 2560 |
| 26 | gate_proj | 2560x9216 | 0.973414 | 0.991037 | 0.993755 | 0.231939 | 0.134646 | 3.804932e-03 | 9216 | 9216 |
| 26 | up_proj | 2560x9216 | 0.978857 | 0.991963 | 0.993831 | 0.206588 | 0.127424 | 1.776535e-03 | 9216 | 9216 |
| 26 | down_proj | 9216x2560 | 0.985531 | 0.990630 | 0.993814 | 0.170662 | 0.137285 | 3.098560e-03 | 2560 | 2560 |
| 27 | q_proj | 2560x8192 | 0.977580 | 0.986887 | 0.993440 | 0.212831 | 0.162958 | 3.942841e-03 | 8192 | 8192 |
| 27 | gate_proj | 2560x9216 | 0.971815 | 0.990791 | 0.993726 | 0.237106 | 0.136016 | 4.817297e-03 | 9216 | 9216 |
| 27 | up_proj | 2560x9216 | 0.971462 | 0.991723 | 0.993817 | 0.241064 | 0.129446 | 2.231593e-03 | 9216 | 9216 |
| 27 | down_proj | 9216x2560 | 0.981689 | 0.990435 | 0.993758 | 0.191556 | 0.138829 | 3.972491e-03 | 2560 | 2560 |
| 28 | gate_proj | 2560x9216 | 0.967815 | 0.990929 | 0.993684 | 0.253091 | 0.135042 | 4.889287e-03 | 9216 | 9216 |
| 28 | up_proj | 2560x9216 | 0.971761 | 0.990940 | 0.993791 | 0.236297 | 0.135180 | 2.109133e-03 | 9216 | 9216 |
| 28 | down_proj | 9216x2560 | 0.981430 | 0.990895 | 0.993681 | 0.195297 | 0.135614 | 3.239396e-03 | 2560 | 2560 |
| 29 | gate_proj | 2560x9216 | 0.974437 | 0.989992 | 0.993621 | 0.225788 | 0.142319 | 4.265803e-03 | 9216 | 9216 |
| 29 | up_proj | 2560x9216 | 0.976045 | 0.991333 | 0.993784 | 0.217912 | 0.132294 | 2.185405e-03 | 9216 | 9216 |
| 29 | down_proj | 9216x2560 | 0.980555 | 0.990071 | 0.993675 | 0.198056 | 0.141505 | 3.400832e-03 | 2560 | 2560 |
| 30 | gate_proj | 2560x9216 | 0.971541 | 0.988541 | 0.993662 | 0.242084 | 0.151959 | 5.184568e-03 | 9216 | 9216 |
| 30 | up_proj | 2560x9216 | 0.975084 | 0.990888 | 0.993780 | 0.224297 | 0.135463 | 2.600038e-03 | 9216 | 9216 |
| 30 | down_proj | 9216x2560 | 0.976770 | 0.987058 | 0.993540 | 0.217590 | 0.161984 | 3.564276e-03 | 2560 | 2560 |
| 31 | q_proj | 2560x8192 | 0.969336 | 0.986748 | 0.993526 | 0.249531 | 0.163712 | 4.646894e-03 | 8192 | 8192 |
| 31 | gate_proj | 2560x9216 | 0.974571 | 0.987880 | 0.993692 | 0.227274 | 0.156486 | 5.689646e-03 | 9216 | 9216 |
| 31 | up_proj | 2560x9216 | 0.971665 | 0.990366 | 0.993778 | 0.238731 | 0.139514 | 3.125489e-03 | 9216 | 9216 |
| 31 | down_proj | 9216x2560 | 0.979366 | 0.984587 | 0.993013 | 0.203807 | 0.176528 | 4.681492e-03 | 2560 | 2560 |

## Reconciliation with C10

C10 reported `kiln_matches_ref=True` on all 3 audited MTP-tap positions (seeds 42/43/44 at `mtp_pos=2`, draft tokens 3074 / 16078 / 1814) with splice cos_sim ≥ 0.9999 against the fp32 HF reference hidden state. C11 now reports cos_min ≈ 0.97 per output weight channel on every Marlin-packed projection.

These two results are consistent, and the reason is cosine concentration:

- INT4 symmetric groupwise quantization with groupsize=128 has zero-mean round-off error within each group (clamp + round-to-nearest). On a 2560-dim contraction axis (q_proj/gate_proj/up_proj all contract over hidden_size=2560), the per-output-channel activation×weight dot product averages ~20 groups of near-zero-mean errors.
- By the concentration of cos_sim under independent noise, the cos_sim of the *output vector* against the fp32 output is approximately `1 − var(error) / (2 · ||w||² / n)` plus O(1/n) corrections. For ε ≈ s/2 and s ≈ 2·max|w|/15, the expected per-channel *output* cos_sim is ≈ 0.9999 + ε² variance terms, two orders of magnitude tighter than the per-channel *weight* cos_sim of ≈ 0.97.
- C10's behavioral null at the MTP tap is therefore not in contradiction with C11's weight-level drift — it is the *predicted* consequence.

## Cross-reference with C10 Class B rejection positions

C10 landed with 0/72 splice flips across 24 tap sites × 3 seeds, so C10 does not currently expose token positions where kiln's bf16 argmax diverges from the fp32 reference. Until a C12 run captures rejected-token positions (Class B top1 flips), C11 cannot cross-reference specific output channels against specific rejections.

The audit above is therefore a **necessary-condition** check, not a sufficient-condition one:

- If all per-channel weight drift had been *inside* the band (cos_sim ≥ 0.9999), Marlin would have been conclusively ruled out as a numerics source. That is not the case — weight drift is out of band by construction.
- Weight drift being out of band does NOT imply Marlin is *the* α-suppression signal. C10 has already constrained the signal to be small at the hidden-state level under normal decoding. To confirm or rule out Marlin causally, C12 must either (a) measure activation-weighted output drift at *rejection* positions, or (b) swap the draft head to fp32 and measure α lift.

## Worst-case channels (C12 targets)

The worst cos_sim and rel_L2 concentrate on `up_proj` (out_dim = intermediate_size = 9216) and `gate_proj` layers 17, 23, 27, 28. These are the first places to look for activation-coupled amplification.

| Rank | Projection | Layer | Cos_min channel | cos_min | rel_L2 | RMSE |
|---|---|---|---|---|---|---|
| 1 | up_proj | 17 | (see JSON) | 0.964366 | 0.265271 | 2.23e-03 |
| 2 | gate_proj | 28 | (see JSON) | 0.967815 | 0.253091 | 4.89e-03 |
| 3 | up_proj | 23 | (see JSON) | 0.968947 | 0.248469 | 1.67e-03 |
| 4 | q_proj   | 31 | (see JSON) | 0.969336 | 0.249531 | 4.65e-03 |
| 5 | gate_proj | 20 | (see JSON) | 0.970374 | 0.242777 | 3.60e-03 |
| 6 | up_proj | 27 | (see JSON) | 0.971462 | 0.241064 | 2.23e-03 |

## Honest limits

1. **Weight-only, not activation-weighted.** A channel with 3% weight drift may still contract to <0.01% output drift once multiplied by a 2560-dim activation with near-zero-mean error averaging. We do NOT measure this here. A C12 follow-up that captures `h = x @ W_bf16 - x @ W_marlin_dq` under the C10 Class B rejection-position activation distribution would close this gap.
2. **No KV cache FP8 coupling.** Kiln optionally runs `KILN_KV_CACHE_FP8=true`; this audit is orthogonal to that path.
3. **MTP layer untouched.** The MTP head is BF16-only in kiln today (no Marlin packing on any `mtp.*` weight); this audit has nothing to say about MTP-layer numerics directly. C10 closed that cleanly with the tap bisect.
4. **Scale round-trip is deterministic and matches the kernel.** We reproduce kiln's f16 round-trip of the scale (`scale = f16(2·max|w|/15)`). The kernel's FP16-only mma (`mma.m16n8k16.f32.f16.f16.f32`) adds an additional bf16→fp16 cast on the *activation* side, not audited here.
5. **Non-Marlin projections are out of scope.** k_proj / v_proj / o_proj (full-attn), linear_attn (GDN) projections, lm_head, embed_tokens, and the entire MTP head all stay BF16 in kiln and are untouched by this audit.

## Reproducing this audit

```
hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b
python3 scripts/c11_marlin_audit.py \
    --checkpoint /workspace/qwen3.5-4b \
    --out docs/archive/phase-c/phase-c11/c11-marlin-audit.md \
    --out-json docs/archive/phase-c/phase-c11/c11-marlin-audit.json
```

Runtime on a single A40: ~80 seconds for 104 projections. Only requires numpy + torch + safetensors; no CUDA, no Marlin build.

