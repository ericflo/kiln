# Phase C37 — post-#420 seed0 vs seed1 `h_main` audit at `mtp_pos=0`

**Date:** 2026-04-23
**Hardware:** RunPod RTX A6000 48 GB, pod `d6m3ff672498v9`, image `ghcr.io/ericflo/kiln-runpod:latest`
**Build under test:** fresh `main` at `59ea3b5` (`Avoid zeroing Metal GDN recurrent prefill outputs (#419)`)
**Task question:** after PR #420's post-#417 identity-bias refresh, is the standard-workload seed split already visible in the base-model `h_main` path at `mtp_pos=0`, before any new kernel work is queued?

## Preflight

### Remaining-work gate 1: no newer PR already lands this artifact

Checked on 2026-04-23 against fresh `main`:

- PR [#420](https://github.com/ericflo/kiln/pull/420) is the latest landed post-#417 native-MTP identity-bias artifact.
- GitHub PR search for `"seed0-vs-seed1"`, `"post-#420"`, `"h_main drift"`, `"phase c37"`, and `"native MTP" "seed 0" "seed 1"` found no newer open or merged PR that already commits a post-#420 seed0-vs-seed1 `h_main` drift artifact for the standard `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training` workload.

Result: **condition 1 still holds**.

### Remaining-work gate 2: current-main native MTP still shows the post-#420 seed split

Fresh current-main rerun on `59ea3b5`:

| seed | prompt tokens | decode tok/s | mean ITL ms | p50 / p99 ITL ms | alpha |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 494 | 44.27 | 22.59 | 15.75 / 61.34 | 0.7397 |
| 1 | 508 | 24.23 | 41.27 | 57.26 / 67.55 | 0.3093 |

Result: **condition 2 still holds**. Seed 0 is the escaping case near the 0.72 floor; seed 1 remains trapped in the low-alpha regime.

## Commands

Local validation:

```bash
python3 -m py_compile scripts/c15_h_main_drift_audit.py scripts/mtp_compare.py
```

RunPod setup and build:

```bash
export RUNPOD_API_KEY=$(ce secret-get --name RUNPOD_API_KEY)
RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py

python3 $RP wait d6m3ff672498v9
B2_KEY_ID=$(ce secret-get --name B2_APPLICATION_KEY_ID)
B2_KEY=$(ce secret-get --name B2_APPLICATION_KEY)
python3 $RP ssh d6m3ff672498v9 \
  "export B2_APPLICATION_KEY_ID='$B2_KEY_ID' B2_APPLICATION_KEY='$B2_KEY'; kiln-setup --clone"
python3 $RP ssh d6m3ff672498v9 \
  "cd /workspace/kiln && git fetch origin && git checkout main && git reset --hard origin/main"
python3 $RP bg d6m3ff672498v9 /tmp/build.log \
  'source /root/.kiln-build-env && export KILN_CUDA_ARCHS=86 && cd /workspace/kiln && cargo build --release --features cuda --bin kiln-bench'
python3 $RP wait-file d6m3ff672498v9 /workspace/kiln/target/release/kiln-bench --timeout 3600
```

Environment prerequisite for the existing audit tooling on this pod image:

```bash
python3 $RP ssh d6m3ff672498v9 \
  'python3 -m pip install --upgrade transformers sentencepiece'
```

Model download:

```bash
python3 $RP ssh d6m3ff672498v9 \
  'source /root/.kiln-build-env && cd /workspace/kiln && hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b'
```

Seed 0 capture and audit:

```bash
python3 $RP bg d6m3ff672498v9 /tmp/seed0.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && \
   KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
   KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS=0 \
   KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 KILN_MTP_DUMP_HIDDEN_STATES=1 \
   KILN_MTP_DUMP_PATH=/workspace/c37/seed0/mtp_pos-{pos}/step-{step}.safetensors \
   ./target/release/kiln-bench \
     --model-path /workspace/qwen3.5-4b --paged \
     --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed 0 \
     > /workspace/c37/seed0_bench.json'

python3 $RP bg d6m3ff672498v9 /tmp/seed0-audit.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && \
   python3 scripts/c15_h_main_drift_audit.py \
     --checkpoint /workspace/qwen3.5-4b \
     --captures-root /workspace/c37/seed0 \
     --out /workspace/c37/seed0_audit \
     --device cuda'

python3 $RP wait-file d6m3ff672498v9 /workspace/c37/seed0_audit/c15_audit_bf16.json --timeout 3600
```

Seed 1 capture and audit:

```bash
python3 $RP bg d6m3ff672498v9 /tmp/seed1.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && \
   KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
   KILN_MTP_DUMP_SPLICE=1 KILN_MTP_DUMP_SPLICE_POS=0 \
   KILN_MTP_DUMP_SPLICE_MAX_STEPS=8 KILN_MTP_DUMP_HIDDEN_STATES=1 \
   KILN_MTP_DUMP_PATH=/workspace/c37/seed1/mtp_pos-{pos}/step-{step}.safetensors \
   ./target/release/kiln-bench \
     --model-path /workspace/qwen3.5-4b --paged \
     --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed 1 \
     > /workspace/c37/seed1_bench.json && \
   python3 scripts/c15_h_main_drift_audit.py \
     --checkpoint /workspace/qwen3.5-4b \
     --captures-root /workspace/c37/seed1 \
     --out /workspace/c37/seed1_audit \
     --device cuda'

python3 $RP wait-file d6m3ff672498v9 /workspace/c37/seed1_audit/c15_audit_bf16.json --timeout 3600
```

## Audit summary

The committed JSON artifacts are:

- `profiling-artifacts/post420_seed0_c15_audit_bf16.json`
- `profiling-artifacts/post420_seed1_c15_audit_bf16.json`

Per-seed `mtp_pos=0` summary:

| seed | alpha | pos-0 dumps | chained seq len | min cos | mean cos | max \|Δ\| | drift-growth ratio | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.7397 | 4 | 497 | 0.998639 | 0.999197 | 0.5999 | 0.399x | mild drift, near-clean |
| 1 | 0.3093 | 8 | 515 | 0.985204 | 0.997293 | 2.5313 | 3.345x | materially worse drift |

### Seed 0 (`profiling-artifacts/post420_seed0_c15_audit_bf16.json`)

| step | base_pos | cos_sim | max \|Δ\| |
| --- | ---: | ---: | ---: |
| 0 | 494 | 0.998639 | 0.5999 |
| 1 | 495 | 0.999080 | 0.5000 |
| 2 | 496 | 0.999613 | 0.3125 |
| 3 | 497 | 0.999457 | 0.3750 |

This lane never gets fully "strict-threshold clean" because step 0 is below `0.999`, but all four pos-0 dumps stay tightly clustered near parity and never open a large gap.

### Seed 1 (`profiling-artifacts/post420_seed1_c15_audit_bf16.json`)

| step | base_pos | cos_sim | max \|Δ\| |
| --- | ---: | ---: | ---: |
| 0 | 508 | 0.999505 | 0.7500 |
| 1 | 509 | 0.999563 | 0.3672 |
| 2 | 510 | 0.998628 | 0.5313 |
| 3 | 511 | 0.999705 | 0.2813 |
| 4 | 512 | 0.999403 | 0.3750 |
| 5 | 513 | 0.997996 | 1.1250 |
| 6 | 514 | 0.985204 | 2.5313 |
| 7 | 515 | 0.998343 | 0.7500 |

The trapped seed starts cleaner than seed 0 at step 0, but after repeated `mtp_pos=0` iterations it develops a much larger hidden-state gap, with a sharp late spike at step 6.

## Verdict

**Yes. The post-#420 seed split is already visible in the base-model `h_main` path at `mtp_pos=0` on current `main`, but not as a simple step-0 mismatch.**

The cleanest reading of the two fresh audits is:

- **Seed 0 (escaping, `alpha=0.7397`)** leaves the `mtp_pos=0` lane after four dumps and stays near parity the whole time: `cos_sim` remains in `[0.998639, 0.999613]`, `max|Δ| <= 0.60`, and the drift-growth ratio is **0.399x**.
- **Seed 1 (trapped, `alpha=0.3093`)** starts out at least as clean at step 0 (`cos_sim=0.999505`), but it stays pinned in `mtp_pos=0` long enough for the hidden-state path to diverge materially: by step 5 `cos_sim=0.997996`, by step 6 it falls to **0.985204**, `max|Δ|` rises to **2.53125**, and the drift-growth ratio is **3.345x**.

So the base-model hidden-state path does distinguish the trapped seed from the escaping seed on the standard workload. What it does **not** support is the simpler claim that seed 1 is already worse at the first pos-0 observation. The split emerges over repeated pos-0 iterations, not immediately at prompt exit.

That matters for triage:

- this is **not** evidence for a static prompt-start wiring bug unique to seed 1;
- it **is** evidence that, on the trapped workload, something in the repeated pos-0 decode path lets `h_main` walk away from the HF reference before the head ever escapes the low-alpha regime.

## Recommendation

**Do not queue new kernel work off this seed split.** The next bounded task should stay on MTP correctness and use this exact seed pair as the anchor:

1. keep seed 0 / seed 1 on fresh current `main` as the source-of-truth reproducer;
2. focus on the repeated `mtp_pos=0` path, especially seed 1 steps 5-6 where `h_main` first opens a large gap;
3. compare accept/reject bookkeeping, draft-token carry-forward, and pos-0 state evolution there before reopening any decode-kernel vendor work.

The highest-signal follow-up is a seed-1 late-step correctness bisect, not another performance pass.

## Validation

- `python3 -m py_compile scripts/c15_h_main_drift_audit.py scripts/mtp_compare.py`
- Fresh RunPod A6000 capture + audit completed on current `main` (`59ea3b5`) for both seeds.
- The committed JSON files were byte-matched against the generated pod outputs with `sha256sum`:
  - seed 0: `c8db9207fc7e4e6763a7b2e810534691cc519d78320aa6925cd561d1edaeee81`
  - seed 1: `33ec525a11a23104fa238764fc445e4ee132e96f65b8ca538d32e36e2c946120`
