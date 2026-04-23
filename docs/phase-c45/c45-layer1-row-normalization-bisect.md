# Phase C45 — post-#434 narrowed row-scalar rerun

## Scope

PR #434 narrowed the remaining C45 question to three row-scalar sub-steps:

- `layer_1_input_norm_rms_inv_scalar_extracted_values`
- `layer_1_input_norm_pre_weight_row_scalar_values`
- `layer_1_input_norm_pre_weight_row_reconstructed`

This rerun attempted to execute that already-merged narrowed tap contract on the
first healthy allowed on-demand RunPod pod. No code changes were required or
made in this task; the only goal was to obtain fresh post-#434 evidence.

## Remaining-work preflight

Before spending GPU time, the fresh `origin/main` preflight was rechecked:

- PR #434 was present on `origin/main` (`7808adf`, merged 2026-04-23).
- No newer open or merged kiln PR had already recorded the earliest shared bad
  narrowed C45 tap among the three post-#434 taps.
- No separate pending or running kiln task overlapped this exact post-#434
  rerun.

Those checks passed, so the task proceeded to RunPod execution.

## Actual RunPod outcome

The rerun never reached a usable SSH endpoint on any allowed GPU class:

- `NVIDIA RTX A6000`: launch failed immediately with RunPod
  `SUPPLY_CONSTRAINT`.
- `NVIDIA A40`: pod `jbagv24mf1f31l` launched on
  `ghcr.io/ericflo/kiln-runpod:latest`, but `python3 $RP wait jbagv24mf1f31l`
  stayed at `status=RUNNING, uptime=N/As` and then crashed after termination
  because the pod never exposed a runtime/SSH endpoint.
- `NVIDIA A100 80GB PCIe`: launch failed immediately with RunPod
  `SUPPLY_CONSTRAINT`.
- `NVIDIA RTX 6000 Ada Generation`: launch failed immediately with RunPod
  `SUPPLY_CONSTRAINT`.
- `NVIDIA L40S`: launch failed immediately with RunPod
  `SUPPLY_CONSTRAINT` ("no longer any instances available with enough disk
  space").
- `NVIDIA H100 PCIe`: pod `zh6zr8z5v0g1tn` launched on the kiln image, but
  `runpod_api.py info` continued to report `"runtime": null` and
  `python3 $RP wait zh6zr8z5v0g1tn` never reached SSH before the pod was
  terminated.

Because no allowed pod reached SSH, none of the required on-pod validation
commands could be run:

- `cargo build --release --features cuda --bin kiln-bench`
- `cargo test --locked -p kiln-model c45 -- --nocapture`
- `python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py`
- either `scripts/mtp_compare.py --c45-row-scalar ...` rerun

## Verdict

There is still no post-#434 narrowed C45 numerical verdict. The blocker remains
RunPod availability and pod health, not missing instrumentation:

- no earliest shared bad narrowed C45 tap was captured
- no new bench JSON, stderr, or compare outputs exist for the post-#434 rerun
- the next step is still to rerun the committed workload on the next healthy
  allowed on-demand pod without changing the tap contract

## Evidence

- `profiling-artifacts/post434_c45_row_scalar_runpod_blocker.txt`
