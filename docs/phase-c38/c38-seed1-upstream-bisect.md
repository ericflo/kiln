# Phase C38 — post-#422 seed1-only upstream `h_main` bisect

**Date:** 2026-04-23  
**Fresh main:** `c66cf41e02c53add403db3031249a6818d8efe5c` (PR #422)  
**Hardware:** RunPod `NVIDIA RTX A6000` via `ghcr.io/ericflo/kiln-runpod:latest`

## Preflight

1. **No newer PR already lands this artifact:** satisfied. Fresh `gh pr list`
   on `ericflo/kiln` still showed PR #422 as the newest relevant merged item;
   there was no newer open or merged PR landing a post-#422 upstream-base
   hidden-state bisect for the standard native-MTP workload.
2. **Fresh current `main` still reproduces the seed split:** satisfied.
3. **`PROFILING.md` still points upstream of `h_main`:** satisfied. Phase C37
   explicitly recommended bisecting the first seed1-only divergence upstream
   of the `h_main` tap rather than retrying another decode-kernel change.

## Validation

- `python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py`
- Fresh RunPod rerun of the standard workload for seeds 0 and 1
- Fresh kiln hidden-state dumps with `KILN_MTP_DUMP_HIDDEN_STATES=1` and
  `KILN_MTP_DUMP_B12_GQA_TAPS=1`
- Fresh HF reference dumps with `scripts/mtp_h_main_reference_dump.py --b12-taps`
- Fresh comparator runs with `scripts/mtp_compare.py --b10` and `--b12`
- Re-ran the comparator commands on-pod and confirmed the committed
  `.txt` artifacts are the generated outputs for this run

## Commands run

Local preflight:

```bash
python3 -m py_compile scripts/mtp_h_main_reference_dump.py scripts/mtp_compare.py
gh pr list --repo ericflo/kiln --state all --limit 30
```

RunPod setup:

```bash
export RUNPOD_API_KEY=$(ce secret-get --name RUNPOD_API_KEY)
RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py

python3 $RP wait 3slsfd81ftanhz
B2_KEY_ID=$(ce secret-get --name B2_APPLICATION_KEY_ID)
B2_KEY=$(ce secret-get --name B2_APPLICATION_KEY)
python3 $RP ssh 3slsfd81ftanhz \
  "export B2_APPLICATION_KEY_ID='$B2_KEY_ID' B2_APPLICATION_KEY='$B2_KEY'; kiln-setup --clone"
python3 $RP bg 3slsfd81ftanhz /tmp/build.log \
  'source /root/.kiln-build-env && cd /workspace/kiln && export KILN_CUDA_ARCHS=86 && cargo build --release --features cuda --bin kiln-bench'
python3 $RP wait-file 3slsfd81ftanhz /workspace/kiln/target/release/kiln-bench --timeout 3600
```

The baked image did not include `transformers`, so the existing HF reference
script required this one-time dependency install:

```bash
python3 $RP ssh 3slsfd81ftanhz 'python3 -m pip install transformers'
```

Per-seed standard workload + dump + compare:

```bash
for seed in 0 1; do
  rm -rf profiling-artifacts/post422_20260423_seed${seed}_captures
  mkdir -p profiling-artifacts/post422_20260423_seed${seed}_captures

  KILN_SPEC_METHOD=mtp \
  KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_DUMP_HIDDEN_STATES=1 \
  KILN_MTP_DUMP_B12_GQA_TAPS=1 \
  KILN_MTP_DUMP_PATH=/workspace/kiln/profiling-artifacts/post422_20260423_seed${seed}_captures/step-0.safetensors \
  ./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training \
    --seed ${seed} \
    > profiling-artifacts/post422_20260423_seed${seed}.bench.json \
    2> profiling-artifacts/post422_20260423_seed${seed}.bench.stderr

  python3 scripts/mtp_h_main_reference_dump.py \
    --checkpoint /workspace/qwen3.5-4b \
    --kiln-dump profiling-artifacts/post422_20260423_seed${seed}_captures/step-0.safetensors \
    --out profiling-artifacts/post422_20260423_seed${seed}_ref.safetensors \
    --b12-taps

  python3 scripts/mtp_compare.py --b10 \
    --pair seed${seed}:profiling-artifacts/post422_20260423_seed${seed}_captures/step-0.safetensors,profiling-artifacts/post422_20260423_seed${seed}_ref.safetensors \
    > profiling-artifacts/post422_20260423_seed${seed}_b10_compare.txt || test $? -eq 1

  python3 scripts/mtp_compare.py --b12 \
    --pair seed${seed}:profiling-artifacts/post422_20260423_seed${seed}_captures/step-0.safetensors,profiling-artifacts/post422_20260423_seed${seed}_ref.safetensors \
    > profiling-artifacts/post422_20260423_seed${seed}_b12_compare.txt || test $? -eq 1
done
```

## Fresh seed split

| Seed | prompt tokens | decode tok/s | α |
| --- | ---: | ---: | ---: |
| 0 | 494 | 41.93 | 0.7397 |
| 1 | 508 | 22.52 | 0.3093 |

This still reproduces the post-#420 / post-#422 split on fresh current
`main`: seed 0 escapes near the paper floor and seed 1 stays trapped.

## Comparator outputs

- [`profiling-artifacts/post422_20260423_seed0_b10_compare.txt`](../../profiling-artifacts/post422_20260423_seed0_b10_compare.txt)
- [`profiling-artifacts/post422_20260423_seed1_b10_compare.txt`](../../profiling-artifacts/post422_20260423_seed1_b10_compare.txt)
- [`profiling-artifacts/post422_20260423_seed0_b12_compare.txt`](../../profiling-artifacts/post422_20260423_seed0_b12_compare.txt)
- [`profiling-artifacts/post422_20260423_seed1_b12_compare.txt`](../../profiling-artifacts/post422_20260423_seed1_b12_compare.txt)

## B10 result

The first divergent boundary is the same for both seeds:

| Seed | last matching tap | first divergent tap | comment |
| --- | --- | --- | --- |
| 0 | `h_layer_8` | `h_layer_16` | seed 0 is still the escaping case end to end |
| 1 | `h_layer_8` | `h_layer_16` | seed 1 is the failing case |

So the existing B10 boundary scan does **not** isolate a seed1-only first
divergence. It only says both seeds already differ from the HF reference by
`h_layer_16`, while seed 1 later diverges more severely.

## B12 result

The first divergent GQA-tail layer is also the same for both seeds:

| Seed | first layer with median `cos_sim < 0.995` | median cos at `h_layer_31` |
| --- | --- | ---: |
| 0 | `h_layer_31` | 0.979465 |
| 1 | `h_layer_31` | 0.971042 |

That is directionally useful, but still not a seed1-only earliest boundary.
Seed 1 is worse in magnitude, not different in the first layer that trips the
comparator's bar.

## Structural blocker in the current B12 path

The current B12 route still cannot produce a trustworthy first-sub-op verdict.

Kiln-side key inspection after the run showed:

- seed 0 kiln dump: `0` `b12__*` tensors, no `h_pre_final_norm`
- seed 1 kiln dump: `0` `b12__*` tensors, no `h_pre_final_norm`

HF-reference key inspection showed:

- seed 0 reference dump: `11` `b12__*` tensors plus `h_pre_final_norm`
- seed 1 reference dump: `11` `b12__*` tensors plus `h_pre_final_norm`

The reference script also logged that `rope_q`, `rope_k`, and `attn_out`
remain hook-unreachable in the current Python path, so even the reference side
is incomplete for the full 14-tap B12 contract.

That combination means the B12 summary's "re-run with the b12__ sub-op tap env
flags set" is not actionable on this exact revision: the env flags were set,
but the kiln dump still emitted no `b12__*` tensors.

## Comparator-footnote caveat

The raw comparator footer still prints:

```text
Overall verdict: divergence(s) at: ['h_main']
```

That line is not the phase verdict here. In B10/B12 mode, `mtp_compare.py`
still walks the legacy primary taps first, and the reference dump intentionally
does not include those MTP-head tensors. The authoritative phase verdicts are
the B10/B12 summary tables, not that footer.

## Verdict

**The existing B10/B12 comparator path is structurally insufficient to
localize a seed1-only earliest upstream divergence on fresh `main`.**

What this artifact does establish:

1. The fresh seed split still holds on `c66cf41`.
2. Both seeds first diverge from the HF reference by `h_layer_16`.
3. Both seeds first cross the B12 GQA-tail bar at `h_layer_31`.
4. Seed 1 is worse than seed 0 in drift magnitude, but not different in the
   first divergent boundary that the current B10/B12 route can see.
5. The current kiln-side B12 dump path did not emit the promised `b12__*`
   tensors, and the HF reference still misses 3 of the 14 expected sub-op taps.

## Recommendation

Do not broaden this into a fix task. The next narrow task should first repair
or verify the current B12 plumbing:

- kiln-side `KILN_MTP_DUMP_B12_GQA_TAPS=1` must actually write `b12__*`
  tensors and `h_pre_final_norm`;
- the HF reference needs `rope_q`, `rope_k`, and `attn_out` taps so the
  B12 contract is complete.

Only after that should the project retry the "first seed1-only upstream
divergence" localization. As of this phase, the honest source-of-truth verdict
is that the current comparator stack stops at shared layer-level boundaries and
cannot yet name a seed1-only first divergent sub-op.
