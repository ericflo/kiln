# Phase C45 — post-#434 narrowed row-scalar rerun

## Scope

PR #434 narrowed the remaining C45 question to three row-scalar sub-steps:

- `layer_1_input_norm_rms_inv_scalar_extracted_values`
- `layer_1_input_norm_pre_weight_row_scalar_values`
- `layer_1_input_norm_pre_weight_row_reconstructed`

This rerun executed that already-merged narrowed tap contract on the first
healthy allowed on-demand RunPod pod and captured fresh post-#435 evidence.

## Remaining-work preflight

Before spending GPU time, the fresh `origin/main` preflight was rechecked:

- PR #434 was present on `origin/main` (`7808adf`, merged 2026-04-23).
- No newer open or merged kiln PR had already recorded the earliest shared bad
  narrowed C45 tap among the three post-#434 taps.
- No separate pending or running kiln task overlapped this exact post-#434
  rerun.

Those checks passed, so the task proceeded to RunPod execution.

## Actual RunPod outcome

The first healthy allowed pod was:

- GPU: `NVIDIA RTX A6000`
- image: `ghcr.io/ericflo/kiln-runpod:latest`
- successful evidence pod: `8d7s7t6zxs827r`

The first healthy A6000 pod exposed two concrete bootstrap bugs on `main` that
had to be fixed before the rerun could finish:

- `deploy/runpod/kiln-setup.sh` no longer downloaded
  `/workspace/qwen3.5-4b`, so the bench failed at model load on the first
  healthy pod until the setup helper was restored to pull the checkpoint.
- `crates/kiln-model/src/mtp_debug.rs` serialized the narrowed splice dumps
  but never created the parent directories for
  `profiling-artifacts/.../mtp_pos-{pos}/step-{step}.safetensors`, so the
  C45 dump path warned and produced no safetensors until `create_dir_all(...)`
  was added.
- The image also lacked the Python `transformers` package required by
  `scripts/mtp_h_main_reference_dump.py`; the recovery pod installed it and
  this PR adds it to the RunPod Dockerfile.

## Verdict

The post-#435 narrowed rerun now has a stable two-seed verdict:

- seed 0 (`mtp_pos=0`, `step=1` compare): earliest shared bad narrowed C45 tap
  = `layer_1_input_norm_pre_weight_row_scalar_values`
- seed 1 (`mtp_pos=2`, `step=1` compare): earliest shared bad narrowed C45 tap
  = `layer_1_input_norm_pre_weight_row_scalar_values`
- the last shared-good narrowed tap on both seeds is
  `layer_1_input_norm_rms_inv_scalar_extracted_values`
- `layer_1_input_norm_pre_weight_row_reconstructed` also diverges, but only
  after the flat row-local scalar multiply values are already bad

So the narrowed post-#434 rerun clears the extracted scalar replay itself and
localizes the first shared row-local drift to the actual multiply that produces
the flat scalar-applied row values.

## Validation

These commands completed on the healthy A6000 pod after the two bootstrap
fixes above:

- `cargo build --release --features cuda --bin kiln-bench`
- `cargo test --locked -p kiln-model c45 -- --nocapture`
- `python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py`
- `python3 scripts/mtp_compare.py --c45-row-scalar ...` for both representative
  seeds

Recovered seed metrics from the final artifact-producing pod:

- seed 0: prompt=494, prefill=`9587.54 ms`, decode=`39.82 tok/s`,
  `alpha=0.7397`
- seed 1: prompt=508, prefill=`404.84 ms`, decode=`23.86 tok/s`,
  `alpha=0.2959`

## Recommendation

- do not widen to C46 or expand the C45 tap contract further
- focus the next audit inside the row-local scalar multiply that produces
  `layer_1_input_norm_pre_weight_row_scalar_values`
- treat the row-local `rms_inv` tensor and its extracted scalar replay as
  provisionally cleared on current `main`

## Evidence

- `profiling-artifacts/post435_c45_row_scalar_seed0.bench.json`
- `profiling-artifacts/post435_c45_row_scalar_seed0.bench.stderr`
- `profiling-artifacts/post435_c45_row_scalar_seed1.bench.json`
- `profiling-artifacts/post435_c45_row_scalar_seed1.bench.stderr`
- `profiling-artifacts/post435_c45_row_scalar_seed0_compare.txt`
- `profiling-artifacts/post435_c45_row_scalar_seed1_compare.txt`

## 2026-04-23 broadcast-mul parity rerun

After aligning the C45 replay helper with the production last-row
`broadcast_mul` shape path in `forward.rs`, a fresh A6000 rerun on current
`main` still confirms the same boundary on the representative two-seed check:

- seed 0 (`profiling-artifacts/c45_broadcast_parity_seed0_compare.txt`):
  earliest shared bad tap remains
  `layer_1_input_norm_pre_weight_row_scalar_values`
- seed 1 (`profiling-artifacts/c45_broadcast_parity_seed1_compare.txt`):
  earliest shared bad tap remains
  `layer_1_input_norm_pre_weight_row_scalar_values`

So production-op parity for the replay path does **not** move the first-bad
boundary past the row-local scalar multiply on current `main`; C45 remains
localized to the multiply itself, and this task stops here without widening the
tap contract.

## 2026-04-23 row-scalar boundary resolution

Fresh preflight on current `origin/main` (`edcc275`, with PRs #445 and #446)
confirmed that the latest checked-in C45 verdict still named
`layer_1_input_norm_pre_weight_row_scalar_values` as the earliest shared bad
tap after the broadcast-mul parity rerun.

The split-tap ordering added by PR #441 exposed that the previous
`layer_1_input_norm_pre_weight_row_scalar_values` boundary was partly an
audit-helper artifact: the helper independently recomputed a flattened
scalar-affine row even though the production path had already produced the
row-shaped `broadcast_mul` output. The helper and HF mirror now derive the
flattened tap directly from that production-shaped output, so the flat tap no
longer answers a different question from the preceding broadcast tap.

Fresh representative C45 reruns on the fallback A100 pod then moved the
earliest shared bad tap backward to the real production row-shaped multiply:

- seed 0 (`mtp_pos=0`, `step=1`):
  `layer_1_input_norm_pre_weight_row_broadcast_output`; last shared-good tap =
  `layer_1_input_norm_last_row_flat_values`; prompt=494,
  prefill=`1390.24 ms`, decode=`36.73 tok/s`, `alpha=0.7162`
- seed 1 (`mtp_pos=2`, `step=1`):
  `layer_1_input_norm_pre_weight_row_broadcast_output`; last shared-good tap =
  `layer_1_input_norm_last_row_flat_values`; prompt=508,
  prefill=`430.97 ms`, decode=`25.00 tok/s`, `alpha=0.2828`

Resolved verdict: the old flat row-scalar boundary should not be treated as a
standalone production bug, but C45 is not fully cleared. The next real target is
the production RMSNorm row-shaped `broadcast_mul` result, because both seeds
agree that the selected last-row values and row-local scalar remain shared-good
under the current C45 tolerance, while the production multiply output is the
first shared bad tap.

Evidence:

- `profiling-artifacts/c45_resolution_seed0_compare.txt`
- `profiling-artifacts/c45_resolution_seed1_compare.txt`

## 2026-04-23 row-shaped broadcast-mul boundary verdict

Fresh preflight on current `origin/main` (`4512877`, with later unrelated Metal
PR #449 also present) confirmed that PR #448 is merged and that the checked-in
C45 evidence still names `layer_1_input_norm_pre_weight_row_broadcast_output`
as the earliest shared bad tap, with
`layer_1_input_norm_last_row_flat_values` as the last shared-good tap.

A fresh A100 fallback rerun kept that tap ordering on both representative
seeds, but the added C45 operand diagnostic closes the boundary as operand-drift
amplification rather than an independent production `broadcast_mul` bug:

- seed 0 (`profiling-artifacts/c45_broadcast_seed0_compare.txt`): earliest
  shared bad tap = `layer_1_input_norm_pre_weight_row_broadcast_output`; last
  shared-good tap = `layer_1_input_norm_last_row_flat_values`; same-side product
  residuals are tiny (`kiln max=8.34e-07`, `ref max=7.15e-07`); predicted
  product max diff equals observed output max diff (`1.38e-01`); prompt=494,
  prefill=`7671.88 ms`, decode=`44.90 tok/s`, `alpha=0.6842`.
- seed 1 (`profiling-artifacts/c45_broadcast_seed1_compare.txt`): earliest
  shared bad tap = `layer_1_input_norm_pre_weight_row_broadcast_output`; last
  shared-good tap = `layer_1_input_norm_last_row_flat_values`; same-side product
  residuals are tiny (`kiln max=1.19e-06`, `ref max=1.73e-06`); predicted
  product max diff equals observed output max diff (`1.43e-01`); prompt=508,
  prefill=`361.04 ms`, decode=`28.74 tok/s`, `alpha=0.2929`.

Resolved verdict: the row-shaped broadcast multiply is not currently evidence
of a production RMSNorm shape/casting bug or an HF mirror mismatch. Both sides'
`broadcast_mul` outputs are exactly explained by their own captured row and
scalar operands; the apparent first-bad multiply output is the current C45
allclose threshold exposing amplified but previously tolerated operand drift.
The next exact target is therefore the upstream operand drift budget feeding
this multiply, starting with the row-local RMS scalar / selected last-row
provenance under a tighter C45 tolerance or a new operand-error-budget tap, not
a production `broadcast_mul` code change.
