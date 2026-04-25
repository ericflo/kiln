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

**Verdict: kiln_native_ceiling.** The reject sub-population stays at the
BF16-noise cosine ceiling — `reject_cos_sim_median = 0.999978`, well above
the 0.999 floor. Top-1 agreement is 100% on both accept and reject rows.
H15b refutes verifier-numerical-drift as the α-gap source on this
checkpoint. See §"Verdict" below.

Decision rule set *before* the run:

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
C1 attribution CSV (same schema as `docs/archive/phase-c/phase-c36/c1_seed*.csv`) is emitted
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

**Why not use `docs/archive/phase-c/phase-c36/c1_seed*.csv` directly?** Those were
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

Machine-readable record: [`docs/archive/phase-c/phase-c29-v2/verdict.json`](phase-c29-v2/verdict.json).

| field | value |
| --- | --- |
| `verdict` | **kiln_native_ceiling** |
| `reject_cos_sim_median` | **0.999978** |
| `reject_cos_sim_p10`    | 0.999971 |
| `accept_cos_sim_median` | 0.999979 |
| `accept_cos_sim_p10`    | 0.999968 |
| `n_accept` / `n_reject` / `n_unlabeled` | 7 / 15 / 0 |
| `top1_match_rate` (both strata) | **1.0000** |

Decision rule applied: `reject_cos_sim_median (0.999978) ≥ 0.999` →
**kiln_native_ceiling**.

**Queued next action**: queue a vLLM α microbench to establish the
external-reference upper bound on Qwen3.5-4B A6000 bs=1 at this workload.
Kiln's MTP head logits are at the BF16-noise cosine ceiling even on
rejected drafts — the α-gap source is not kiln numerical drift.

### Headline stratified table

| stratum | n | median cos | p10 cos | min cos | top-1 match | median J@10 | median KL(k‖r) |
| --- | --: | ---: | ---: | ---: | ---: | ---: | ---: |
| **accept-only**  | 7  | 0.999979 | 0.999968 | 0.999956 | 1.000 | 1.0000 | 2.02e-04 |
| **reject-only**  | 15 | 0.999978 | 0.999971 | 0.999957 | 1.000 | 1.0000 | 1.78e-04 |
| all labeled      | 22 | 0.999978 | 0.999970 | 0.999956 | 1.000 | 1.0000 | 1.90e-04 |

The accept and reject sub-populations are statistically indistinguishable
on every cosine-based metric — there is no reject-row divergence to find.

### Per-position stratified medians

| `mtp_pos` | accept n | accept median cos | accept p10 | reject n | reject median cos | reject p10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 0.999979 | 0.999977 | 3 | 0.999977 | 0.999977 |
| 1 | 1 | 0.999980 | 0.999980 | 5 | 0.999978 | 0.999973 |
| 2 | 2 | 0.999981 | 0.999980 | 4 | 0.999977 | 0.999961 |
| 3 | 2 | 0.999966 | 0.999958 | 3 | 0.999983 | 0.999978 |

Every `mtp_pos` × (accept, reject) cell sits comfortably above the 0.999
floor. No position-dependent reject-row drift; the pos=3 accept p10
(0.999958) is the worst data point and still 0.000958 above the floor.

### What this rules out

- **Verifier numerical drift on reject rows.** If a bug in the base-model
  verify path or MTP head produced systematically worse logits when the
  draft will be rejected, `reject_cos_sim_median` should fall below 0.99.
  It did not; it's at the ceiling.
- **Position-dependent reject-row divergence.** All four `mtp_pos` cells
  show reject cos_sim within BF16 noise of accept cos_sim.
- **Top-K rotation on reject rows.** J@10 median is 1.0000 on both
  strata; 100% top-1 agreement on both strata. This mirrors PR #355's
  aggregate J@10 = 1.0000 finding but now confirms it conditional on
  rejection.

### What this does NOT rule out

- **External α ceiling.** H15b measures only kiln's internal agreement
  with an fp32 HF reference. It says nothing about whether another
  serving system (vLLM, SGLang, reference transformers) would hit a
  higher α on the same checkpoint. That is the queued next step.
- **Workload sensitivity.** This run uses prose prompts from
  PROMPT_POOL with 512-token prefill and 16-token decode. Chat-template
  prompts may expose a different α regime (per `mtp-bench-workload-sensitivity`
  note, chat α ≈ 0.588 vs prose α ≈ 0.175 on PR #364).
- **Downstream sampler divergence.** Under greedy, 100% top-1 agreement
  is conclusive. Temperature or top-p sampling may still produce
  different accept patterns if the tail of the distribution drifts
  more than the head; that is out of scope for this probe.

## Artifact summary

- **Build**: `kiln-bench` release + CUDA, `KILN_CUDA_ARCHS=86`, sccache warm
  hit rate >99% (rebuild: 2m17s).
- **Bench invocations**: 3 (one per seed), each `--prompt-tokens 512
  --max-output-tokens 16 --paged --skip-training`.
- **Kiln dumps produced**: 22 (seed 0: 7, seed 1: 7, seed 2: 8 — one
  extra at `mtp_pos=0 step=1`).
- **HF reference dumps produced**: 22/22 (0 errors, 0 missing taps).
- **C1 attribution rows**: 37 total (12 + 13 + 12). Kept under
  `docs/archive/phase-c/phase-c29-v2/artifacts/c1_attr_seed{0,1,2}.csv`.

## Pod spend

- GPU: NVIDIA RTX A6000 48GB (direct-launch fallback — pool 503'd on
  all 3 hibernated pods with `capacity_supply_exhausted`; policy lets
  us bypass the pool when it's at cap or unhealthy, see
  `runpod-capacity-fallback`).
- Wall clock: ~35 min lease (launch → kiln-setup → build → dumps → HF
  ref → compare → pull).
- Cost estimate: ~$0.30 at $0.49/hr. Well under the 90-min / \$40 cap.

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
- `docs/archive/phase-c/phase-c29-v2/c29-v2-stratified-compare.json` — full per-(seed, pos, step) breakdown
- `docs/archive/phase-c/phase-c29-v2/c29-v2-stratified-compare.md` — human-readable verdict tables
- `docs/archive/phase-c/phase-c29-v2/verdict.json` — machine-readable decision record
- `docs/archive/phase-c/phase-c29/c29-h9-verdict.md` — parent C29 (H9) verdict
- `docs/phase7-mtp-acceptance-state-of-play.md` — PR #527 state-of-play
- `docs/phase7-h15a-marlin-determinism.md` — PR #528 H15a result
- Agent notes: `mtp-head-audit-needs-topk-jaccard`, `mtp-bench-workload-sensitivity`,
  `kiln-ssh-polling-deadlock`, `runpod-always-on-demand`, `runpod-gpu-minimum-a6000`
