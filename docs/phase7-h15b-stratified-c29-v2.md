# Phase 7 Audit: H15b — Stratified C29 v2 Reject-Row Probe

Date: 2026-04-25
kiln main at audit time: post-PR #528 (`phase7-h15a-marlin-determinism`)
Parent PRs:
- [#527 — phase7: MTP acceptance-rate state-of-play audit (doc-only)](https://github.com/ericflo/kiln/pull/527) (merged 2026-04-24)
- [#528 — phase7: H15a Marlin pack determinism correlation (doc-only)](https://github.com/ericflo/kiln/pull/528) (merged 2026-04-24)
Scope: stratified re-run of PR #355's C29 top-K Jaccard / cos_sim probe on the
`c14__logits` tap, split by accept vs reject rows, A6000 48GB, ~30 min,
~$0.25.

## Summary

PR #527 §"Recommended next H" queued H15b as the next micro-bench after H15a
ruled Marlin pack determinism out. The hypothesis: **on the rejected
sub-population of MTP draft rows, does kiln's logits stay at the BF16-noise
cosine ceiling (≥ 0.999) or diverge materially (< 0.99) from an fp32 HF
reference?** PR #355's C29 run (2026-04-22) reported median cos_sim 0.99998
aggregated across all 49 dumps — if the reject sub-population hides cos_sim <
0.99 behind a 90%+ accept floor, the aggregate metric would miss it.

**Verdict: <VERDICT_TOKEN>.** See §"Verdict" below. Decision rule set
*before* the run:

- `reject_cos_sim_median ≥ 0.999` → **kiln_native_ceiling** → next: queue vLLM α microbench
- `reject_cos_sim_median <  0.99`  → **verifier_numerical_drift** → next: queue per-layer bisect on reject rows
- `0.99 ≤ reject_cos_sim_median < 0.999` → **ambiguous** → next: expand seeds to 0..5 and re-run

## Anti-duplication evidence

Pre-flight search (2026-04-25):

```bash
gh pr list -R ericflo/kiln --state all --search "H15b stratified C29" --limit 10
gh pr list -R ericflo/kiln --state all --search "c29 v2 reject row"   --limit 10
gh pr list -R ericflo/kiln --state all --search "H15b"                --limit 10
gh pr list -R ericflo/kiln --state all --search "C29 v2"              --limit 10
```

No prior PR (open, merged, or closed-null) matches H15b / C29 v2 / reject-row
probe / stratified C29. Closest neighbours:

- **#527** (doc-only state-of-play audit) — queued H15b as a recommendation.
  Did not execute the probe.
- **#528** (H15a Marlin pack determinism) — ruled out the alternative
  hypothesis. Did not execute the probe.
- **#355** (original C29 / H9 run) — ran the aggregate top-K Jaccard probe
  and refuted H9, but did not stratify by accept/reject.

H15b is unambiguously new work.

## Methodology

### Bench configuration (kiln side)

Reuses the PR #355 C29 harness (`scripts/c29_kiln_logits_dump.sh`) with one
additive change: **`KILN_C1_ATTR_PATH` is now also set**, so a per-seed
C1 attribution CSV (same schema as `docs/phase-c36/c1_seed*.csv`) is emitted
alongside the splice dumps from the same bench run. This guarantees
per-site accept/reject labels align with splice-dump coordinates — see
"Label alignment" below.

- Model: Qwen3.5-4B
- Mode: BF16 + Marlin W4A16 (`KILN_W4A16=1`)
- Prompts: 3 seeds from the kiln-bench PROMPT_POOL (seed 0..2, each selects
  `PROMPT_POOL[seed % 8]`)
- Splice anchors: `KILN_MTP_DUMP_SPLICE_POS=0,1,2,3`
- Splice step cap: `KILN_MTP_DUMP_SPLICE_MAX_STEPS=2`
- Prompt tokens: 512; decode tokens: 16 (matches PR #355 envelope)
- GPU: NVIDIA RTX A6000 48GB (on-demand, image `ghcr.io/ericflo/kiln-runpod:latest`)
- Build: `KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench`

### HF reference generation

`scripts/c29_hf_reference_dump.py` drives `scripts/c14_hf_reference_dump.py`
per kiln dump file, which in turn invokes `scripts/mtp_reference_dump.py
--capture-subops` to run an fp32 PyTorch re-forward of the post-block MTP
head. Produces a reference safetensors mirror under `<ref_root>/seed-*/
mtp_pos-*/step-*.safetensors`.

### Comparator

New script: `scripts/c29_logits_compare_v2.py`. Re-uses the C29 primary tap
(`c14__logits`) and all C29 metrics (cos_sim, max|Δ|, top-K Jaccard for
K∈{1,5,10,20}, KL divergences, top-1 agreement). Adds:

- **Accept/reject stratification** — two parallel stat blocks, one for
  `accepted=1` rows and one for `accepted=0` rows.
- **Per-position stratification** — cos_sim medians broken out by
  `mtp_pos` × (accept, reject).
- **Decision-rule evaluation** — writes `verdict.json` with the headline
  numbers (`reject_cos_sim_median`, `reject_cos_sim_p10`,
  `accept_cos_sim_median`, `accept_cos_sim_p10`, `n_accept`, `n_reject`,
  `n_unlabeled`) and the computed verdict token + queued next action.

### Label alignment

Each C29 v2 splice dump at `(mtp_pos=N, step=K)` is produced by the K-th
call to `speculative_mtp_decode_step` with `mtp_pos=N` in that bench run.
The C1 attribution sink (`c1_attr::push_row`) fires from the same control
flow — see `crates/kiln-model/src/speculative.rs:842-879` — so it produces
one CSV row per call to `speculative_mtp_decode_step`. Therefore the K-th
CSV row with `mtp_pos=N` corresponds to the K-th splice dump at
`mtp_pos=N`. The comparator's `load_accept_labels` enforces this by
counting per-`mtp_pos` occurrences and indexing into them.

**Why not use `docs/phase-c36/c1_seed*.csv` directly?** Those were
produced by the C36 H14a decode-length sweep with `--chat-template`,
`KILN_MTP_ARGMAX_FP32=1`, and variable `--max-output-tokens` (128/256/512/
1024). The C29 v2 harness runs with the prose PROMPT_POOL (no chat
template), default BF16 MTP head, and `--max-output-tokens 16`. The decode
trajectories and accept patterns differ — re-using those labels would
match wrong rows. See `mtp-bench-workload-sensitivity` agent note (PR #364
measured α ≈ 0.175 prose vs α ≈ 0.588 chat on the same checkpoint) for a
precedent on how sensitive MTP metrics are to prompt distribution. H15b
therefore regenerates labels in-run.

## Wall-clock + cost budget

Task-level hard cap: 90 min / $40 (per project mandate; see `kiln-ssh-polling-deadlock`
note — $99.76 incident on 2026-04-20 from an un-capped wedged-sshd loop).
Expected spend: ~30 min on A6000 ≈ $0.25 (PR #527 §"Recommended next H").

## Verdict

*(populated on completion with concrete numbers from `verdict.json`)*

- `reject_cos_sim_median`: **<FILLED>**
- `reject_cos_sim_p10`:    **<FILLED>**
- `accept_cos_sim_median`: **<FILLED>**
- `accept_cos_sim_p10`:    **<FILLED>**
- `n_accept` / `n_reject` / `n_unlabeled`: **<FILLED>**
- Decision rule applied: **<VERDICT_TOKEN>**
- Queued next action: **<FILLED>**

See `docs/phase-c29-v2/c29-v2-stratified-compare.md` for the full stratified
table and `docs/phase-c29-v2/verdict.json` for the machine-readable verdict.

## Reopen / re-revisit triggers

- Fresh C40-class anchor run with a materially different prompt distribution
  where the aggregate α shifts by ≥ 5 percentage points (e.g. moving from
  prose to chat-template).
- Any new MTP-head kernel change (fused L2 QK-norm, fp32 head upcast, etc.)
  that specifically claims to reduce reject-row divergence. Re-run H15b on
  that build before shipping.
- Evidence from `vLLM α microbench` (if `kiln_native_ceiling` verdict
  triggers the next phase) that the external ceiling is materially above
  kiln's α at this checkpoint — then per-layer bisect on reject rows
  becomes the follow-up, regardless of H15b's original verdict.
- If `verifier_numerical_drift` verdict triggers and the subsequent
  per-layer bisect fails to localize a single layer, re-open H15b with
  tighter seed fan-out (0..11) to improve reject-row n.

## References

- `scripts/c29_logits_compare_v2.py` — new stratified comparator
- `scripts/c29_kiln_logits_dump.sh` — now emits per-seed `c1_attr.csv`
- `docs/phase-c29-v2/c29-v2-stratified-compare.json` — full per-(seed, pos, step) breakdown
- `docs/phase-c29-v2/c29-v2-stratified-compare.md` — human-readable verdict tables
- `docs/phase-c29-v2/verdict.json` — machine-readable decision record
- `docs/phase-c29/c29-h9-verdict.md` — parent C29 (H9) verdict
- `docs/phase7-mtp-acceptance-state-of-play.md` — PR #527 state-of-play
- `docs/phase7-h15a-marlin-determinism.md` — PR #528 H15a result
- Agent notes: `mtp-head-audit-needs-topk-jaccard`, `mtp-bench-workload-sensitivity`,
  `kiln-ssh-polling-deadlock`, `runpod-always-on-demand`, `runpod-gpu-minimum-a6000`
