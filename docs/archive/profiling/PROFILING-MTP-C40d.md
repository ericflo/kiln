# Phase C40d — seed-0 HumanEval anchor drift audit

## TL;DR

This audit compared current `origin/main`
(`dd7120f05f600b2af4c501801b1cf19f71cde895`, PR #375) against the last
known C40b-good merged commit
(`8870dd838a1182ce7f9d484ec321007f2bdebae3`, PR #374) that documented
the seed-0 sanity result `alpha = 0.6933333333`.

Verdict: **the seed-0 anchor is not stable enough to gate C40c as a
singleton exact-value check.** This is **not** a code-path regression in
the audited files.

- The requested static diff over
  `crates/kiln-server/src/bench.rs`,
  `crates/kiln-model/src/speculative.rs`,
  `crates/kiln-model/src/generate.rs`,
  `crates/kiln-core/src/sampling.rs`,
  `PROFILING-MTP-C39.md`,
  `PROFILING-MTP-C40b.md`,
  `PROFILING-MTP-C40c.md`
  found **no source diff at all** in the four code files and no diff in
  `PROFILING-MTP-C39.md` / `PROFILING-MTP-C40b.md`; only
  `PROFILING-MTP-C40c.md` was added after PR #374.
- The same pod, same checkpoint, and the **same `kiln-bench` binary**
  (`sha256=8895a7cdf525ea0219097c176d9d6cccfce1bade73195197f25e30806f79b6ab`)
  produced two different seed-0 results across those two git checkouts:
  `0.7297297297` on current `main` and `0.6410256410` on the reference
  commit.
- Both results miss the C39/C40b documented singleton target
  `0.6933333333`, and they miss it in opposite directions.

Exact recommendation: **resume C40c**, but do **not** treat
`seed=0 temp=0.0 == 0.6933333333` as a hard precondition anymore. The
audit does not support a `fix <file>` recommendation.

## Preflight

1. Existing work check:
   - GitHub PR search for `C40d`, `anchor drift`, and `seed-0 HumanEval`
     in `ericflo/kiln` returned no open/pending/running C40d work.
2. Fresh repo:
   - Cloned `ericflo/kiln` into `./repo`.
   - Created branch `ce/mtp-c40d-anchor-drift`.
3. Commit pair under audit:
   - Current `origin/main`:
     `dd7120f05f600b2af4c501801b1cf19f71cde895`
     (`mtp c40c: abort BF16 sweep after sanity regression`, PR #375).
   - Reference C40b-good merged commit:
     `8870dd838a1182ce7f9d484ec321007f2bdebae3`
     (`mtp c40b: HumanEval + temperature=0.1 N=20 alpha re-bench`,
     PR #374).
4. Static diff on requested paths:
   - `git diff --stat 8870dd8..dd7120f -- <requested paths>` showed only
     one changed file:
     `PROFILING-MTP-C40c.md` (new file, 142 insertions).
   - There was **no diff** in:
     `crates/kiln-server/src/bench.rs`,
     `crates/kiln-model/src/speculative.rs`,
     `crates/kiln-model/src/generate.rs`,
     `crates/kiln-core/src/sampling.rs`,
     `PROFILING-MTP-C39.md`,
     `PROFILING-MTP-C40b.md`.

## Validation

Required validation was run on the same A6000 pod used for the audit.

### `cargo test --no-run`

Required command:

```bash
cargo test -p kiln-model -p kiln-server --features cuda --no-run
```

Outcome:

- First attempt under the pod defaults failed in debug-profile CUDA
  compilation because `ptxas` was killed while building multi-arch debug
  flash-attn objects (`sm_80/sm_89/sm_90`), which is an infra/build-mode
  issue rather than a kiln source failure.
- Successful validation command:

```bash
source /root/.kiln-build-env
export KILN_CUDA_ARCHS=86 CARGO_PROFILE_DEV_DEBUG=0
cd /workspace/kiln
cargo test -p kiln-model -p kiln-server --features cuda --no-run
```

- Result: success in `2m 48s`, with the expected test executables built
  for `kiln-model` and `kiln-server`.

## GPU Environment

| Item | Value |
| --- | --- |
| Date | 2026-04-22 |
| Pod | `jol7bbddn9rfg3` |
| GPU | NVIDIA RTX A6000 |
| Image | `ghcr.io/ericflo/kiln-runpod:latest` |
| Checkpoint | `/workspace/qwen3.5-4b` from `hf download Qwen/Qwen3.5-4B` |
| Compared commits | `dd7120f05f600b2af4c501801b1cf19f71cde895` vs `8870dd838a1182ce7f9d484ec321007f2bdebae3` |

## Two-run GPU Sanity Result

Unchanged anchor command for both runs:

```bash
KILN_W4A16=1 \
KILN_MTP_ARGMAX_FP32=1 \
KILN_SPEC_ENABLED=1 \
KILN_SPEC_METHOD=mtp \
KILN_CUDA_GRAPHS=true \
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --chat-template --skip-training --latency-only \
  --prompt-tokens 512 --max-output-tokens 128 \
  --prompt-subset humaneval \
  --seed 0 --temperature 0.0
```

| Commit | Role | alpha | vs expected `0.6933333333` | decode tok/s | prefill tok/s |
| --- | --- | --- | --- | --- | --- |
| `dd7120f` | current `origin/main` | `0.7297297297` | `+0.0363963964` | `41.35` | `65.03` |
| `8870dd8` | C40b-good reference | `0.6410256410` | `-0.0523076923` | `37.94` | `1649.81` |

Additional evidence:

- `sha256(target/release/kiln-bench)` after the runs:
  `8895a7cdf525ea0219097c176d9d6cccfce1bade73195197f25e30806f79b6ab`
- The binary hash was captured after restoring the pod checkout to
  current `main`, and the static diff confirms there were no audited code
  changes across the two compared commits.
- The JSON output hashes differed:
  - `main_seed0_temp0.json`:
    `b1812534313db011e2c35e1679eb9688d2fc3c9879c03870fe7d4266fafc1b18`
  - `reference_seed0_temp0.json`:
    `26c09c8035b089a6e9b318ebfd8e73aa3b4bff124be61d9de3c5b789522348dc`

## Interpretation

1. **This is not a source regression in the audited paths.**
   The requested code files are byte-identical across the compared
   commits.
2. **The old singleton seed-0 anchor is unstable.**
   On the same pod, same checkpoint, same command line, and same bench
   binary, the measured alpha moved from `0.6410` to `0.7297` depending
   on run instance / checkout context alone.
3. **C40c should not stay blocked on exact seed-0 equality.**
   The current blocker from PR #375 assumed that missing
   `0.6933333333` implied code-path drift. This audit falsifies that
   assumption.
4. **No narrow file fix is justified.**
   There is no evidence pointing at
   `bench.rs`, `speculative.rs`, `generate.rs`, or `sampling.rs`.

## Recommendation

Exact recommendation: **resume C40c**.

Practical follow-through for the next task:

- Treat the `seed=0` sanity as a soft smoke check, not an exact numeric
  gate.
- Keep the same-pod paired-comparison design for any W4A16 vs BF16
  follow-up.
- If a hard gate is still desired, replace the singleton exact-value
  check with a short replicated window (for example 3 repeated seed-0
  runs per side, or a tiny multi-seed paired sample), because this audit
  shows one seed-0 run is not stable enough to distinguish code drift
  from environmental/nondeterministic variation.

## Artifacts

- [docs/archive/phase-c/phase-c40d/main_seed0_temp0.json](docs/archive/phase-c/phase-c40d/main_seed0_temp0.json)
- [docs/archive/phase-c/phase-c40d/main_seed0_temp0.log](docs/archive/phase-c/phase-c40d/main_seed0_temp0.log)
- [docs/archive/phase-c/phase-c40d/reference_seed0_temp0.json](docs/archive/phase-c/phase-c40d/reference_seed0_temp0.json)
- [docs/archive/phase-c/phase-c40d/reference_seed0_temp0.log](docs/archive/phase-c/phase-c40d/reference_seed0_temp0.log)
