# Phase C15 — `h_main` Drift Audit Across Decode Steps

**Bench**: Qwen3.5-4B BF16, paged, `KILN_SPEC_METHOD=mtp`, A6000 48GB, 512-prompt × 128-decode, seed 42.
**Captures**: `KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS="0,2" KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 KILN_MTP_DUMP_HIDDEN_STATES=1`. Fresh: 8 kiln dumps at `mtp_pos=0`, steps 0..7, all with `prompt_tokens`.
**Reference**: `scripts/c15_h_main_drift_audit.py` — single HF forward on the chained token sequence `T = step0.prompt_tokens ++ [step_k.prompt_tokens[0] for k in 1..7]` (length 519), `output_hidden_states=True`, `use_cache=False`. Per-step `h_ref_k = hidden_states[-1][0, base_pos_k - 1, :]` (Qwen3.5 attention is causal so this is equivalent to 8 independent forwards).
**Thresholds**: cos_sim ≥ 0.999 at every `(pos, step)` = CLEAN. Anything below = DRIFT.
**Bench α during fresh capture**: 0.000 (worse than the α ≈ 0.033 collapsed baseline seen in C12/C13/C14). Consequence: `mtp_pos` never advanced so `pos=2` dumps were not produced this session.

## Verdict

**DRIFT — but structural and pre-existing, not growing across decode steps.**

The drift is the same across steps 0..7 (not amplifying), identical between BF16 and FP32 references (not accumulation noise), and matches the pre-existing B10/B11 finding of cos ≈ 0.982 at step 0. The C15 hypothesis that "`h_main` drift grows with decode step and collapses α" is **refuted**: step 7 drift ≈ step 0 drift.

Handoff to C16: the α-collapse signal must be downstream of `h_main` (acceptance math, sampler, KV-rollback, or draft-token positional wiring).

## Per-step results at `mtp_pos=0`

### BF16 reference

| step | base_pos | cos_sim   | max\|Δ\|  | rel_err_mean | kiln_norm | ref_norm |
|-----:|---------:|----------:|----------:|-------------:|----------:|---------:|
| 0    | 512      | 0.982310  | 2.512e+01 | 6.469e+01    | 67.77     | 155.82   |
| 1    | 513      | 0.986500  | 3.150e+01 | 1.463e+02    | 69.44     | 159.50   |
| 2    | 514      | 0.963439  | 1.769e+01 | 2.506e+01    | 63.46     | 150.26   |
| 3    | 515      | 0.960762  | 2.338e+01 | 1.074e+02    | 80.25     | 153.28   |
| 4    | 516      | 0.986577  | 1.238e+01 | 2.864e+00    | 70.92     | 158.11   |
| 5    | 517      | 0.952823  | 2.106e+01 | 9.752e+00    | 69.11     | 146.59   |
| 6    | 518      | 0.972763  | 2.281e+01 | 5.953e-01    | 70.44     | 153.04   |
| 7    | 519      | 0.976470  | 2.088e+01 | 1.893e+01    | 62.71     | 152.68   |

### FP32 reference

| step | base_pos | cos_sim   | max\|Δ\|  | rel_err_mean | kiln_norm | ref_norm |
|-----:|---------:|----------:|----------:|-------------:|----------:|---------:|
| 0    | 512      | 0.982333  | 2.550e+01 | 7.059e-01    | 67.77     | 155.87   |
| 1    | 513      | 0.986784  | 3.139e+01 | 8.925e-01    | 69.44     | 159.40   |
| 2    | 514      | 0.963464  | 1.770e+01 | 6.296e-01    | 63.46     | 150.51   |
| 3    | 515      | 0.961570  | 2.349e+01 | 5.264e-01    | 80.25     | 153.22   |
| 4    | 516      | 0.986800  | 1.264e+01 | 5.770e-01    | 70.92     | 158.02   |
| 5    | 517      | 0.952725  | 2.112e+01 | 5.885e-01    | 69.11     | 146.79   |
| 6    | 518      | 0.972997  | 2.289e+01 | 8.605e-01    | 70.44     | 153.16   |
| 7    | 519      | 0.976439  | 2.074e+01 | 6.673e-01    | 62.71     | 152.78   |

### Drift-growth summary

| reference | step-0 drift `1 - cos` | step-7 drift `1 - cos` | growth ratio |
|-----------|-----------------------:|-----------------------:|-------------:|
| BF16      | 1.769e-02              | 2.353e-02              | 1.330x       |
| FP32      | 1.767e-02              | 2.356e-02              | 1.334x       |

The ratio is within the step-to-step noise band (step 1 ratio is 0.763x, step 4 ratio is 0.759x, step 5 ratio is 2.675x). There is no monotonic growth of drift with decode step.

## Why the drift is not a C15-new finding

- Step-0 BF16 cos = 0.98231 matches the Phase B10/B11 figure (cos ≈ 0.982) from `mtp_h_main_reference_dump.py` against a single kiln dump. Chaining 8 steps into one HF forward reproduces the single-shot reference at step 0 to 1e-4, validating the chained-sequence methodology.
- FP32 and BF16 cos_sim agree to within 1e-3 at every step. BF16 accumulation across 32 transformer layers is eliminated as the primary source of the gap: the structural mismatch is the dominant term.
- The 2x norm gap (kiln `h_main` ≈ 70, HF `hidden_states[-1]` ≈ 155) also persists in FP32, so it is not a scalar scaling bug introduced by low-precision arithmetic. It indicates kiln's tap and HF's `hidden_states[-1]` represent hidden state in related but measurably different reference frames (most plausibly a pre- vs post-partial-normalization difference in how the final decoder-layer output is observed). This is the same measurement difference C14 implicitly worked around by using `mtp_reference_dump.py`'s passthrough reference.

## Structural gap: `c13_hf_reference_dump.py` cannot detect `h_main` drift

`scripts/c13_hf_reference_dump.py` walks `<root>/mtp_pos-{N}/step-{K}.safetensors` and shells out to `scripts/mtp_reference_dump.py` per file. That script, at line ~696, emits `"h_main": h_main.contiguous()` — i.e. it echoes kiln's own `h_main` back into the reference dump instead of running an independent HF forward on the captured prompt. The comparator therefore computes cos_sim(kiln, kiln) = 1.0000 trivially, which is exactly what C14's verdict table shows:

> `h_main` | median cos_sim 1.0000000 | max|Δ| 0.00e+00

That is a known passthrough artifact, not evidence of `h_main` matching HF. C15 replaces that reference with a chained independent HF forward (`scripts/c15_h_main_drift_audit.py`) and that is where the ~0.982 cos_sim first becomes visible in the splice-dump sweep. The C13 passthrough was rerun in this session and continues to report cos = 1.0000 at every step; it is included only as an explicit control, not as a divergence signal.

## Structural gap: `mtp_pos=2` was not auditable this session

Splice capture only writes a dump at a specific `mtp_pos` value when the decode loop's running `mtp_pos` counter matches. `mtp_pos` advances only on ACCEPT (see `bench_latency_paged_mtp` in `crates/kiln-server/src/bench.rs`, line ~1130: `mtp_pos += step.mtp_advance`). Under the current build, α collapsed to 0.000 during the fresh capture run, so `mtp_pos` stayed pinned at 0 for the entire decode and no `pos=2` dumps were produced.

Older dumps at `/workspace/captures/mtp_pos-2/step-{0..7}.safetensors` (from a prior run where α ≈ 0.033 allowed `mtp_pos` to advance) exist on the pod but predate the `KILN_MTP_DUMP_HIDDEN_STATES` instrumentation: they lack `prompt_tokens`. Without per-step `prompt_tokens` the chained-HF reference cannot be built for them. The comparator flags this explicitly (`status: missing_prompt_tokens`) rather than silently substituting a guessed sequence.

What this means for the bisect: the draft-verify-accept loop is not even exercising MTP head-2 in the current broken state, because the acceptance rate never clears the threshold needed to push `mtp_pos` past 1. `pos=2` drift is therefore uninformative for the α-collapse question in this build — by the time `pos=2` matters, `pos=0` has already been rejected ~30x in a row. The immediate lever is the acceptance decision at `pos=0`, which C15's data shows is not being fed a divergent-from-HF `h_main` *any differently* at step 7 than at step 0.

## Implications for the bisect

1. **C15 hypothesis refuted.** Drift does not amplify across decode steps. An α-collapse mechanism that depends on `h_main` degrading over time is inconsistent with this data.
2. **Pre-existing ~0.98 `h_main` mismatch is stable.** It has been present since B10 and is reproduced step-by-step here. Whatever its cause, it is not sufficient on its own to drive α to 0.000 — C12 (α ≈ 0.033) and earlier phases all ran against the same `h_main` baseline and still saw an order-of-magnitude higher α.
3. **`mtp_pos=2` is unreachable without first fixing `pos=0`.** Any follow-up audit of `pos=2` requires either (a) the acceptance bug to be fixed so decoding can advance `mtp_pos` organically, or (b) a one-shot instrumentation flag that force-advances `mtp_pos` after N decode steps regardless of accept/reject so the speculative path at head-2 can at least be captured.

## Handoff to C16

C16 should audit the path from `h_main` (validated here as stably positioned relative to HF reference) through MTP head-0 output to the accept/reject decision, covering:

- MTP head-0 `mtp_logits` vs HF reference at the tokens actually being proposed (C14 already covered the 4 post-block taps; C16 should widen to the token-sampling and acceptance-probability math).
- `mtp_advance` computation in `bench_latency_paged_mtp` (verify that `step.mtp_advance` is 1 on the correct side of the accept/reject branch).
- Draft-token positional wiring (is the accepted-token's KV entry going into the slot that the next forward queries against?).
- KV-rollback semantics on reject (is the rollback leaving a consistent cache, or is it inserting an off-by-one that permanently biases `pos=0`?).

Optional groundwork for `pos=2` coverage: add a `KILN_MTP_FORCE_ADVANCE_POS` diagnostic env so future audits can exercise head-1/head-2 even when organic α is collapsed.

## Artifacts

- `scripts/c15_h_main_drift_audit.py` — comparator
- `/workspace/c15_out/audit/c15_audit_bf16.json`, `c15_audit_bf16.txt`
- `/workspace/c15_out/audit/c15_audit_fp32.json`, `c15_audit_fp32.txt`
- `/workspace/c15_out/audit/c15_bf16.log`, `c15_fp32.log`
- Kiln dumps: `/workspace/c15_out/captures/mtp_pos-0/step-{0..7}.safetensors` (with `prompt_tokens`)
- Legacy pos=2 dumps (for reference, unusable without `prompt_tokens`): `/workspace/captures/mtp_pos-2/step-{0..7}.safetensors`
