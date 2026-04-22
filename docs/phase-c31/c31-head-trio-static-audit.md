# Phase C31 — Class B head-trio bisect (already audited clean, doc-only redirect)

**Date**: 2026-04-22
**Phase**: C31 (MTP α-regression hunt, post-C30)
**Hypothesis under test**: The Class B α-regression (mtp_top1 ≠ main_top1 on
281 / 331 C1 rows, α = 15.11 %) is produced by a bug inside the MTP-head trio
— `lm_head` (base + MTP), `mtp.final_layernorm`, or `weights.final_norm`.
**Method**: CPU-first static audit per the C30 handoff, starting from the
Phase C1 attribution CSV (`mtp_top1_logit` / `main_top1_logit` columns).
**Scope**: Head trio only. Accept/reject, GDN rollback, KV rollback, and
position/index resync are **out of scope** — all cleared by C30 (H11.1–H11.3).

## TL;DR

**Verdict: Class B head bisect REFUTED on static + empirical grounds. No code
change. No pod spend.** The MTP-head trio is the single most exhaustively
audited surface in the entire kiln MTP pipeline. Three fully independent
lines of evidence — accumulated across C14, C17, C18, C25, and C29 —
converge on the same answer:

| Surface                       | Static audit (C17 / C25 / C28) | Splice cosine (C14) | Top-K selection (C29) |
| ----------------------------- | :----------------------------: | :-----------------: | :-------------------: |
| `lm_head` weight (tied)       |      ✓ tied to `embed_tokens_t` |  cos ≥ 0.9999529    | top-1 match = 100.00% |
| `mtp.final_layernorm`         |       ✓ separately loaded       | fused in `post_norm` tap | —                |
| `weights.final_norm` (base)   |       ✓ applied once (C18)      | —                   | —                     |

C29 settled the final open question left by C14: a 0.9999 cosine on a
248 320-dim head could in principle hide top-K rotation severe enough to
depress α. It does not. Across 49 paired kiln/fp32-HF reference dumps
(4 seeds × ≤ 4 positions × 4 steps), kiln's MTP head picked the **same top-1
token on every single dump** and the median Jaccard@5 / @10 / @20 was exactly
1.0. The head's selection layer is numerically indistinguishable from an fp32
PyTorch reference on every metric that matters to draft-token acceptance.

The C1 CSV's `mtp_top1_logit` < `main_top1_logit` pattern (median gap
0.375 – 1.125 units across seeds) is therefore **not** evidence of head
miscompute. It is a property of the MTP checkpoint itself: at a given
absolute position, the MTP head's greedy prediction for the next-next token
legitimately disagrees with the base model's greedy prediction for the
next token, and the MTP logit for its own top-1 is lower-magnitude than the
base logit for its own top-1 because the MTP head is fundamentally solving
a harder problem (2-token-ahead vs 1-token-ahead).

**Recommended next phase (C32)**: the one surface **no** phase has yet
independently validated — kiln's base-stack `h_main` vs an fp32 HF
reference. C29 explicitly called this out: "The `h_main` cos = 1.0 row in
C13/C14/C29 is **by construction** (the reference takes `h_main` from the
kiln dump)." Run a C15-style fp32 re-forward of the base model on the same
C1 prompts + accepted-token trajectories and compare kiln's `h_main` at each
step against HF. This is the last unfalsified surface after C30 cleared the
acceptance pipeline and C31 closes the head bisect.

**Cost**: $0 (static + cross-reference audit, doc-only PR, no pod).
~45 min wall-clock. Well under the C31 task's declared cap ($15 / 60 min).

---

## 1. Context: why the C30 handoff named "head bisect" as the live suspect

C30 (`docs/phase-c30/c30-h11-accept-reject-audit.md`) refuted H11
(accept/reject + KV-rollback) and tabulated the remaining suspects as:

| Sub-system                                               | C30 status           |
| -------------------------------------------------------- | -------------------- |
| MTP head draft logits                                    | Suspect              |
| MTP head weight tying (`lm_head` vs MTP projection)      | Suspect              |
| `final_layernorm` / `norm_gap`                           | Suspect              |
| Accept/reject decision                                   | Cleared              |
| GDN state rollback on REJECT                             | Cleared              |
| Base KV stale-slot correctness                           | Cleared              |
| MTP KV slot re-use on REJECT                             | Cleared              |
| `base_pos` / `mtp_pos` / `h_prev` threading              | Cleared              |
| RoPE position for MTP draft                              | Cleared (PR #276)    |

C30 directed C31 at the first three rows and at the Phase C1 attribution CSV
as a starting substrate. That direction is correct as stated — but it is
**partially stale**. Between C16 / C17 / C18 (the source of the "head bisect"
framing) and C31 today, Phases C14 and C29 empirically audited exactly this
trio and came back clean. C31's job is therefore to consolidate that
evidence and redirect C32, not to re-run the bisect.

---

## 2. The head trio as it exists today (HEAD `45fafdf`)

All three trio components are visible in a single `cat crates/kiln-model/src/forward.rs`.

### 2.1 `lm_head` weight is TIED via `embed_tokens_t`

Both the base and MTP paths call the same `lm_head_forward` against the same
transposed embedding tensor:

```rust
// crates/kiln-model/src/forward.rs:4676-4679  (MTP head)
let logits = {
    kiln_nvtx::range!(c"kiln/mtp/lm_head");
    lm_head_forward(&normed, &weights.embed_tokens_t)?
};

// crates/kiln-model/src/forward.rs:5083-5086  (base head, FullWithLastHidden)
let logits = {
    kiln_nvtx::range!(c"kiln/lm_head");
    lm_head_forward(&normed, &weights.embed_tokens_t)?
};
```

`MtpWeights` at `crates/kiln-model/src/weights.rs:231-245` has **no** `lm_head`
field — tying is structurally enforced at the type level, not at runtime. No
separate MTP lm_head weight exists anywhere in the loader, the GPU upload
path, or the `MtpWeights` struct. The `embed_tokens_t` tensor is constructed
once at `forward.rs:897-911` and referenced at the same offset from both
call sites; there is no aliasing, re-packing, or dtype-cast opportunity that
could make the MTP-side head numerically differ from the base-side head.
C25's static audit (`docs/phase-c25/c25-verifier-vs-draft-logit-path.md`)
reached the same conclusion from a different angle.

### 2.2 `mtp.final_layernorm` is SEPARATELY loaded, NOT reused from base

```rust
// crates/kiln-model/src/forward.rs:4668-4671
let normed = {
    kiln_nvtx::range!(c"kiln/mtp/final_layernorm");
    rms_norm(&mtp_hidden, &mtp.final_layernorm, config.rms_norm_eps)?
};
```

`mtp.final_layernorm` is uploaded from the MTP-prefixed checkpoint tensor
`mtp.final_layernorm.weight` at `forward.rs:1143` (the MTP GPU upload block
at 1128-1244). It is *not* aliased to `weights.final_norm` (the base-stack
final RMSNorm). C17's first-divergence audit verified the separation; C19
(`docs/phase-c19/c19-fc-norm-audit.md`) and C26
(`docs/phase-c26/c26-mtp-weight-reload-sanity.md`) independently verified
that every MTP-prefixed tensor lands on the GPU with the correct name, shape,
dtype, and escapes Marlin packing. H1 and H7 on the ruled-out queue.

### 2.3 `weights.final_norm` is applied ONCE per forward (post-C18)

```rust
// crates/kiln-model/src/forward.rs:5066-5087  (LmHeadMode::FullWithLastHidden)
LmHeadMode::FullWithLastHidden => {
    let normed = {
        kiln_nvtx::range!(c"kiln/final_norm");
        rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?
    };
    let last_hidden = normed.narrow(1, seq_len - 1, 1)?.contiguous()?;
    // ...
    let logits = {
        kiln_nvtx::range!(c"kiln/lm_head");
        lm_head_forward(&normed, &weights.embed_tokens_t)?
    };
    Ok((Some(logits), Some(last_hidden)))
}
```

One `rms_norm(&hidden, &weights.final_norm, ...)` call produces a tensor that
feeds *both* the lm_head projection *and* the sliced-last-row `h_prev` handed
to the MTP block. This is the C18 fix
(`docs/phase-c18/c18-h-prev-post-norm-fix.md`, PR #347): prior to C18 the
`h_prev` leaked back to the MTP head pre-norm, giving 2.0-2.4× magnitude drift
vs HF's post-norm convention. C18 recovered 4.7× α (median 0.033 → 0.153).
C28 (`docs/phase-c28/c28-mtp-h-prev-contract-audit.md`) re-verified the fix
is still wired correctly at HEAD.

---

## 3. The three independent clean audits

### 3.1 C14 — splice cosine across the full MTP window

`docs/phase-c14/` + C29's re-run on 49 dumps give the cosine envelope for
every tap between `h_main` and `mtp_logits`:

| Tap                    | n  | median cos   | min cos     |
| ---------------------- | -: | -----------: | ----------: |
| `h_main`               | 49 | 1.000000     | 1.000000    |
| `c14__post_block`      | 49 | 0.999970     | 0.999954    |
| `c14__post_norm`       | 49 | 0.999969     | 0.999958    |
| `c14__logits`          | 49 | 0.999978     | 0.999934    |

`h_main` is bit-exact by construction (the HF reference takes kiln's `h_main`
as input — this is the surface C32 must still test independently). Every
downstream tap sits well inside the bf16-forward-noise envelope for a 2560-d
residual stream and a 248K-d vocab head. C14 was already strong enough to
rule out "a sign-flip / index permutation / missing-bias-class" bug anywhere
in the MTP forward pass. C29 extended C14 from 16 dumps to 49 and from one
prompt seed to four, preserving the envelope.

### 3.2 C29 — top-K Jaccard + KL across the MTP head

`docs/phase-c29/c29-h9-verdict.md`:

| metric                   | median    | min       | max       |
| ------------------------ | --------: | --------: | --------: |
| `cos_sim`                | 0.999978  | 0.999934  | 0.999988  |
| `top-1 match rate`       | **1.0000**| 1.0000    | 1.0000    |
| `Jaccard@5`              | 1.0000    | 0.6667    | 1.0000    |
| `Jaccard@10`             | 1.0000    | 0.8182    | 1.0000    |
| `Jaccard@20`             | 1.0000    | 0.9048    | 1.0000    |
| `KL(kiln‖ref)`           | 2.06e-05  | 3.76e-07  | 6.71e-03  |
| `ref prob @ kiln top-1`  | 0.9863    | 0.4497    | 0.99985   |

**Top-1 match is 100.00% across all 49 dumps.** Median Jaccard@5/@10/@20 is
exactly 1.0 — kiln's full top-20 is the same set as the fp32 reference's
top-20 in the median case. Median KL in either direction is ~2e-5, four
orders of magnitude below any practical sampler-divergence threshold. No
position-dependent or prompt-dependent degradation.

Critically: C29 was the *empirical* test of exactly the C17 / C30 "head
bisect" hypothesis. Its null result is direct evidence that the trio is
not the regression source.

### 3.3 C17 / C25 static — zero silent-asymmetry surface

C17 (`docs/phase-c17/c17-head-forward-first-divergence.md`) located the
single wrong-frame bug in `h_prev` and confirmed every other head-trio
surface clean:
- lm_head tied via `embed_tokens_t`
- mtp.final_layernorm loaded separately at `forward.rs:1143`
- no dtype mismatch, no silent Marlin-packing path, no main/MTP aliasing

C25 (`docs/phase-c25/c25-verifier-vs-draft-logit-path.md`) re-audited
the verifier-vs-draft head path from a different angle (H6) and
re-confirmed no asymmetry.

Three separate readings of the same source at three separate code-tree
revisions, with C18 the only fix needed and applied.

---

## 4. Re-interpreting the C1 attribution CSV

The C1 summary (`docs/phase-c1/c1_summary.txt`) reports per-seed logit
medians:

| seed | mtp median | main median | gap   |
| ---: | ---------: | ----------: | ----: |
| 0    | 24.250     | 24.625      | 0.375 |
| 1    | 23.000     | 24.125      | 1.125 |
| 2    | 23.750     | 24.250      | 0.500 |

The C30 handoff framed this as suspicious — "mtp is systematically lower, so
perhaps the head is underpowered." Post-C29 that framing no longer holds.
At each C1 row the two logits are not measuring the same thing:

- `main_top1_logit` is the base model's greedy top-1 logit at absolute
  position `base_pos + mtp_pos` given the full autoregressive prefix.
- `mtp_top1_logit` is the MTP head's greedy top-1 logit at the **same**
  absolute position given the splice `[last_token, draft_token_mtp_pos]`.

Those are legitimately different computations. The MTP head predicts
"token at position N given we just committed `last_token` at N−1 and a
speculative `draft_token` chain" — a harder problem than the base model's
"token at N given the full real prefix at N−1." It is expected and benign
that the MTP head has lower top-1 confidence on average.

C29's 100% top-1 match against the fp32 HF reference is what cuts the knot:
given the same inputs, kiln's MTP head produces the same top-1 as the
reference. If the logit magnitude is lower than the base head's, that is a
property of the MTP checkpoint's training distribution, not a kiln
compute bug. The ~15% acceptance rate after the C18 4.7× recovery is the
**native α ceiling** of the Qwen3.5-4B MTP checkpoint under kiln's
speculative-decoding recipe, unless an upstream surface (h_main) is
itself drifting.

The Class A = 0 column in the C1 summary is also worth noting: every row
where MTP and base agreed on top-1 was accepted. There is no acceptance-math
bug (C16 / C30 H11.1) hiding under the attribution.

---

## 5. What the full MTP ruled-out queue looks like post-C31

| Phase | Hypothesis                                             | Status     | Mechanism of refutation   |
| ----: | ------------------------------------------------------ | ---------- | ------------------------- |
| C16   | Accept/reject math                                      | Refuted    | Static + C30 re-audit     |
| C17   | First-divergence audit                                  | Pinned h_prev frame; fixed in C18 |
| C18   | `h_prev` pre-norm vs post-norm                          | **FIXED**  | +4.7× α recovery          |
| C19   | `mtp.fc_norm` vs `model.norm` reuse (H1)                | Refuted    | Static audit              |
| C20   | MTP-block dual-norm inversion (H2)                      | Refuted    | Static audit              |
| C21   | Rotary `mtp_pos` offset (H3) — static                   | Inconclusive | Both conventions seen in vLLM |
| C22   | MTP `fc` / residual parameterization (H4)               | Refuted    | Static audit              |
| C23   | Draft sampler / temp / penalty (H5)                     | Refuted    | Static audit (greedy both sides) |
| C24   | Rotary hardware A/B (H3)                                | Refuted    | Δα = +0.010 inside null band |
| C25   | Verifier vs draft logit path (H6)                       | Refuted    | Static audit              |
| C26   | MTP weight-reload sanity (H7)                           | Refuted    | Static audit              |
| C27   | MTP block sequencing + paged-KV carryover (H8)          | Refuted    | Static audit              |
| C28   | `h_prev` post-norm contract (H10)                       | Refuted    | Static audit (C18 still wired) |
| C29   | MTP-head top-K empirical (H9)                           | Refuted    | 100% top-1 across 49 dumps|
| C30   | Accept/reject + KV rollback + position resync (H11)     | Refuted    | Static audit              |
| C31   | Class B head-trio bisect                                | **Refuted**| Triply audited: C14 cos, C29 top-K, C17/C25 static |

Every hypothesis named in the C17 handoff ladder and the C30 handoff table
is now closed.

---

## 6. Recommended C32 direction

Two candidate directions, in priority order.

### 6.1 **C32a — base-stack fp32 parity on `h_main`** (recommended)

C14/C29 both acknowledged their `h_main` cosine is 1.0 *by construction* —
they seed the HF reference forward with kiln's own `h_main`. No phase has
yet checked whether kiln's base-stack is itself computing the right
`h_main` at each decode step.

Concretely: re-forward the base model on the C1 seed prompts in fp32 HF
(with identical KV / position state across accepted-token trajectories) and
compare kiln's `h_main` tensor at each step against the HF value. The
acceptance criterion is a cosine envelope consistent with bf16 noise on a
2560-d residual (~≥ 0.9999).

This would answer the one question C14 + C29 + C30 collectively leave open.
If `h_main` is clean, the 0.15 α is genuinely the MTP checkpoint's native
ceiling under kiln's recipe and C32 closes the investigation. If `h_main`
drifts, C32 localizes the drift to a specific transformer layer or the
GDN / MoE / attention mixture — a surface the MTP side inherits but has
never isolated.

**Cost envelope**: similar to C14 / C29 — 1 A6000 pod-hour to capture,
plus a CPU-side HF re-forward. ~$0.50 – $1.50 pod spend.

**Starting artifacts**: `scripts/c14_hf_reference_dump.py`,
`scripts/c15_h_main_drift_audit.py` (already targets this surface from
the kiln-dump side), and `scripts/c29_hf_reference_dump.py` can be
retargeted — the paired-dump harness already emits `h_main` as one of
the taps.

### 6.2 **C32b — accept the native α ceiling and move on**

If the team's appetite is for a cheaper resolution: C14 + C29 are strong
enough evidence that the MTP head as shipped is not buggy, just of
modest quality. The post-C18 recovery to α ≈ 0.15 on a speculative-decode
recipe that was measuring ≈ 0.033 pre-C18 may simply be what Qwen3.5-4B's
MTP checkpoint is capable of in bf16 paged decode. Wire that into the
speculative-decoding sampler's expected-α and redirect the remaining
engineering budget to a different optimization axis (e.g., KV cache FP8,
scheduler, or a different draft model).

This would leave C32a open as a future task in case a harder-α target
becomes a priority.

### 6.3 **Explicitly NOT recommended**

- Re-running the head bisect at higher precision or on more prompts.
  C29's 49-dump 100%-top-1 result is not a sample-size problem; more
  dumps will produce more 100%-top-1 rows.
- Auditing GDN / conv1d / attention kernels one-by-one without first
  pinning the drift to a specific point in the base stack via C32a.
  That is the opposite of the C13 / C14 / C29 bracketed-bisect method
  that has worked throughout the C-series.

---

## 7. Cost accounting

- Pod time: **$0** (CPU-only static + cross-reference audit).
- Wall-clock: ~45 min (branch + read + write + PR).
- Well inside the C31 task cap of $15 / 60 min.
