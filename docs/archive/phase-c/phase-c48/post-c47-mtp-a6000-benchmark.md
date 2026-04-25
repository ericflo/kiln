# C48 post-C47 forced-MTP A6000 benchmark

## Precondition

- `ce tasks-list --project-id faf9b108c04f7f73a9a39fca --status pending` returned 0 tasks.
- `ce tasks-list --project-id faf9b108c04f7f73a9a39fca --status running` only showed this task (`58e1617792d976a5c2ade270`), so no other pending/running task covered this exact post-C47 forced-MTP A6000 artifact.
- `gh pr list -R ericflo/kiln --limit 10` returned no open PRs.
- Fresh `origin/main` included PR #459 in the last five commits: `e9c071e phase6: classify C45 tolerance artifact boundary (#459)`.
- `docs/archive/phase-c/phase-c45/c45-layer1-row-normalization-bisect.md` contains the C47 stop condition: do not change production inference math at the C45 broadcast boundary unless a fresh current-main reproducer again fails the current mask.
- Prior comparison baseline is `docs/archive/phase-c/phase-c40f/summary.json`.

## Hardware and image

- RunPod pool lease: `pod-c11fe496c432600caf0baa6a`.
- RunPod pod: `sl53yvx5seviyx`.
- GPU: `NVIDIA RTX A6000` with 49,140 MB VRAM reported by `nvidia-smi`.
- Image: `ghcr.io/ericflo/kiln-runpod:latest`.
- CUDA toolkit: 12.4; NVIDIA driver: 550.127.08; Rust/Cargo: 1.95.0.
- Repo: `/workspace/kiln` reset to `origin/main` at `e9c071e` after PR #459.

## Commands

Build and validation:

```bash
source /root/.kiln-build-env
cd /workspace/kiln
git fetch origin
git reset --hard origin/main
KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench
cargo test --locked -p kiln-server bench -- --nocapture || true
python3 -m py_compile scripts/mtp_compare.py scripts/mtp_h_main_reference_dump.py
```

Per-seed benchmark:

```bash
for seed in $(seq 0 19); do
  KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 \
    ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed "$seed" \
    --json-output docs/archive/phase-c/phase-c48/seed-${seed}.json \
    > docs/archive/phase-c/phase-c48/seed-${seed}.log 2>&1
  echo "$?" > docs/archive/phase-c/phase-c48/seed-${seed}.exit
done
```

Note: current `kiln-bench` emitted the final benchmark JSON to stdout; the checked-in `seed-*.json` files are exact extractions of the final JSON object in each successful `seed-*.log`. All `seed-*.exit` files are `0`.

The targeted test filter was valid on current main: `cargo test --locked -p kiln-server bench -- --nocapture` ran 8 bench-routing tests and all passed.

## Seed table

| Seed | Prompt tokens | Alpha | Decode tok/s | Mean ITL ms | Prefill ms | Exit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 494 | 0.7397 | 44.60 | 22.42 | 354.4 | 0 |
| 1 | 508 | 0.3093 | 26.83 | 37.27 | 353.1 | 0 |
| 2 | 501 | 0.3956 | 29.75 | 33.61 | 344.6 | 0 |
| 3 | 496 | 0.2700 | 25.71 | 38.89 | 350.3 | 0 |
| 4 | 512 | 0.3913 | 29.65 | 33.73 | 351.7 | 0 |
| 5 | 496 | 0.2451 | 24.83 | 40.27 | 346.7 | 0 |
| 6 | 512 | 0.2330 | 23.87 | 41.89 | 368.9 | 0 |
| 7 | 498 | 0.2451 | 24.71 | 40.48 | 352.3 | 0 |
| 8 | 512 | 0.6494 | 39.69 | 25.19 | 353.7 | 0 |
| 9 | 509 | 0.6623 | 40.73 | 24.55 | 347.6 | 0 |
| 10 | 488 | 0.3368 | 27.86 | 35.89 | 342.7 | 0 |
| 11 | 507 | 0.2095 | 23.76 | 42.08 | 354.4 | 0 |
| 12 | 502 | 0.2574 | 25.40 | 39.38 | 346.9 | 0 |
| 13 | 486 | 0.2800 | 25.95 | 38.54 | 345.7 | 0 |
| 14 | 494 | 0.3368 | 26.98 | 37.06 | 418.4 | 0 |
| 15 | 502 | 0.3368 | 27.62 | 36.20 | 348.8 | 0 |
| 16 | 507 | 0.2828 | 26.09 | 38.33 | 350.7 | 0 |
| 17 | 510 | 0.3656 | 29.38 | 34.04 | 361.1 | 0 |
| 18 | 497 | 0.3368 | 27.56 | 36.28 | 351.7 | 0 |
| 19 | 507 | 0.2212 | 23.16 | 43.18 | 372.8 | 0 |

## Aggregate stats

- Seeds: `20`; zero exits: `20`.
- Median alpha: `0.3231`; mean `0.3552`; min/max `0.2095` / `0.7397`; stdev `0.1524`.
- Median decode: `26.91` tok/s; mean `28.71` tok/s; min/max `23.16` / `44.60` tok/s.
- Median mean ITL: `37.16` ms; mean `35.96` ms; min/max `22.42` / `43.18` ms.
- Median prefill: `351.7` ms; mean `355.8` ms; min/max `342.7` / `418.4` ms.
- Alpha ≥ 0.72 seeds: `1` / `20`.

## Comparison to C40f

| Metric | C40f median | C48 median | Delta / ratio |
| --- | ---: | ---: | ---: |
| Alpha | 0.6888 | 0.3231 | -0.3657 |
| Decode tok/s | 38.25 | 26.91 | 0.703x |
| Mean ITL ms | 26.15 | 37.16 | 1.421x |

C47 does not change the MTP go/no-go relative to C40f. It clears the C45 production-math stop condition, but this fresh forced-MTP A6000 artifact is still below the acceptance floor and below the C40f decode-speed baseline by more than 10%.

Important comparability caveat: C40f used `--chat-template --latency-only --prompt-subset humaneval --temperature 0.0` with `KILN_MTP_ARGMAX_FP32=1`; this C48 task intentionally used the post-C47 command above. The large alpha drop should therefore be treated as the current command/workload source of truth, not as proof that C47 production math regressed.

## Recommendation

- Keep the C47 stop condition: do not change production RMSNorm or broadcast math at the C45 boundary.
- Because median alpha is below `0.65` and median decode is only `0.703x` of C40f, do not move back to performance optimization yet.
- Next diagnostic boundary: run a small A/B harness-parity check on current main that isolates prompt distribution and decode flags (`--chat-template`, `--prompt-subset humaneval`, `--temperature 0.0`, `KILN_MTP_ARGMAX_FP32=1`, and `--latency-only`) before reopening any model-math investigation.
