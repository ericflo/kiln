# Local Hardware Qualification

Kiln qualifies GPU backends on named physical machines. GitHub Actions and
compile-only backend jobs are portability checks, not hardware evidence. A
qualification run starts from a clean commit, executes a checked-in workload,
keeps bounded raw output under `.qualification/`, and writes one compact JSON
receipt under `qualification/receipts/`.

Detailed Kiln/vLLM serving sweeps use the separate protocol in
[`BENCHMARKS.md`](../BENCHMARKS.md) and belong under `benchmarks/receipts/`.
Do not place that schema in `qualification/receipts/`; both trees have their
own strict validator and CI checks both.

## Prepare A Machine

Fetch and fast-forward `main`, then confirm that the checkout is clean. Do this
again before every receipt so its commit and source-tree identity are exact.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git status --short
```

Install the backend runtime and the command-line probe named by the workload.
For example, ROCm workloads expect `rocminfo` and `ROCM_PATH=/opt/rocm`; Vulkan
workloads expect `vulkaninfo`. Install the Rust toolchain and fetch dependencies
before entering an offline or network-isolated qualification environment.

Validate the workload contract before spending device time:

```bash
python3 scripts/qualification/workload.py \
  qualification/workloads/environment-v1.json \
  qualification/workloads/correctness-core-v1.json
```

The runner rejects a dirty worktree, an uncommitted workload, missing required
variables, a missing required device, silent skips, and an existing receipt or
raw-run directory. Do not bypass those checks.
The ROCm mixed-load driver also rejects ambient `KILN_*` server controls before
building. Configuration changes must be declared in a committed workload
variant; inherited shell overrides are never silently ignored or accepted as
source-bound evidence.

## Refresh The GRPO Reference Oracle

The compact fixture at
`crates/kiln-train/tests/fixtures/grpo_trl_oracle_v1.json` pins scalar GRPO
semantics independently of Kiln. Its generator hash-checks TRL 1.8.0's
`grpo_trainer.py`, calls the real `GRPOTrainer._compute_loss` with precomputed
policy/behavior/reference log-probabilities, differentiates with PyTorch
2.13.0, and takes one `torch.optim.AdamW` step. It runs entirely on CPU.

Use PyTorch's CPU wheel index so refreshing a scalar fixture does not download
CUDA libraries:

```bash
uv run \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --with 'torch==2.13.0+cpu' \
  --with 'trl==1.8.0' \
  python scripts/qualification/grpo_trl_oracle.py --check
```

Omit `--check` only when intentionally regenerating the fixture after changing
the pinned oracle or its input cases. Review the entire JSON diff. Automatic CI
validates the pins, canonical encoding, input hash, coverage, finiteness, and
shapes without installing TRL or PyTorch; Rust tests consume the numeric outputs
directly on each supported backend.

## Run A Workload

Choose a stable, non-secret host ID that identifies the physical machine. Run
one backend variant at a time. The runner prints the final receipt path and
stores bounded stdout/stderr plus their hashes under `.qualification/runs/`.

ROCm core correctness:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant rocm \
  --host-id strix-halo \
  qualification/workloads/correctness-core-v1.json
```

ROCm token-budgeted prefill correctness (Strix Halo/gfx1151):

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant rocm \
  --host-id strix-halo \
  qualification/workloads/prefill-scheduling-v1.json
```

This workload pairs ROCm with a Vulkan variant for later cross-backend receipt
comparison. Each arm combines the literal short-decode/1K-prefill/16K-prefill
actor test with a real-device deterministic hybrid-model parity test. The
latter compares monolithic prefill against six bounded quanta, including
recurrent state, the block-aligned prefix snapshot, the first following decode
token, and KV-block release. Qualification mode turns a missing device into
failure.

After the ROCm receipt is checked in, run the paired Vulkan arm from the same
source tree:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  qualification/workloads/prefill-scheduling-v1.json
```

Vulkan core correctness:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  qualification/workloads/correctness-core-v1.json
```

Model-serving workloads additionally require `--model` with the exact local
model directory and `--model-id` with its public identity. Select each declared
A/B arm explicitly; the manifest, not an ambient environment variable, owns
the effective configuration recorded in the receipt.

For the supported Strix Halo ROCm serving contract, run the `stable` arm. It
deliberately requests autoscaling, automatic allocator reclaim, and ROCm graphs,
then requires the stable profile to suppress all three while mixed SSE load,
long prefill, cancellation, and socket backpressure are active:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant stable \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-mixed-rocm-v1.json
```

The receipt also records every backend external-yield synchronization boundary,
call/failure/slow count, total time, and maximum duration. A failed sync, a sync
lasting at least 100 ms, any physical resize/reclaim/graph event, or any
unexplained ITL outlier fails the stable arm.

Experimental ROCm graph runs expose a closed fallback contract at
`/health.decode_runtime.rocm_graphs.fallbacks`. It reports the total and the
eight reason counts (`warmup_forward_failure`, `cold_cache_host_round_trip`,
`persistent_host_round_trip`, `shape_dependent_attention`, `graph_cache_capacity`,
`critical_memory_pressure`, `capture_failure`, and `replay_failure`) plus slow,
total-duration, and maximum-duration counters. The first occurrence of each
reason, every fallback lasting at least 100 ms, and every failed eager fallback
also emits `event=rocm_graph_fallback` with attempt, eager, and total duration.
Qualification validates the health invariants and attributes these events to
the exact ITL window; unknown reason strings do not receive graph attribution.

The stable serving run also attests the default 64-token prompt-work ceiling
(`server.max_prefill_tokens_per_cycle`), the default four-layer yield ceiling
(`server.max_prefill_layers_per_cycle`), and both startup provenances. Admission
and resumable prefill share the token ceiling after ready decode rows reserve
their tokens. A retained token chunk then yields between transformer-layer
groups without replaying completed layers. The receipt records both effective
values, processed-layer and layer-yield counts, plus cumulative/max actor-phase
times; a run that exercises no inter-layer yield fails. A chunk is charged to
the new-token ceiling exactly once when selected, not again when its retained
final layer completes. Every third prefill dispatch remains round-robin; the
other two may accelerate the shortest tail of at most four token chunks.
The receipt records this bounded-priority count and fails when the mixed
workload does not exercise it. Any ITL outlier remains a failure even when its
phase is explained.
The same run attests an effective decode width of eight, four bounded
short-prefill staging slots, and a total active-request ceiling of twelve in
both health and debug state. It also requires a maximum staged-priority burst
of four before the mandatory global prefill turn. Measurement must record at
least one staging admission, at least one rotating staged-priority forward, and
an observed active width above eight without ever exceeding twelve.
Staged-priority forwards must remain a subset of the bounded short-priority
count. The final cancellation drain requires ordinary decode, prefill, staged
occupancy, and the waiting queue all to reach zero. This proves that the latency
path ran without treating the staging capacity as a wider backend decode batch
or accepting an active prefill as drained.
The pressure peer also requires terminal request-scoped performance metadata.
Its actor queue, slot-admission, and admission-to-first-ready wall durations are
recorded separately and must fit inside TTFT; accumulated model prefill must fit
inside admission plus admitted-prefill wall time. Missing, duplicate,
nonnumeric, or internally impossible phase evidence fails the run. These fields
distinguish active-set saturation from slow admitted prefill before any
scheduler policy is changed.

For the historical dynamic-runtime A/B, run each of `default`,
`autoscale-off`, `graphs-off`, and `both-off` separately. These four arms now
pin `KILN_SERVING_PROFILE=experimental` so their requested graph/autoscale
differences retain the semantics they had before stable became the default:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant default \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-mixed-rocm-v1.json
```

The variant named `default` preserves the graph-on/autoscale-on A/B baseline,
not the production serving default. The manifest intentionally applies one
shared qualification transport envelope to every arm.

Never edit a receipt to make it pass. A failed receipt is useful evidence: keep
it when it identifies a reproducible product defect, fix the defect in a new
commit, and run a new receipt with a new ID.

## Validate The Result

First validate the portable schema and internal hashes:

```bash
python3 scripts/qualification/receipt.py \
  qualification/receipts/rocm/strix-halo/<receipt>.json
```

On the originating machine, require both the current committed source and the
ignored raw artifacts to match:

```bash
python3 scripts/qualification/receipt.py \
  --require-current-source \
  --require-local-artifacts \
  qualification/receipts/rocm/strix-halo/<receipt>.json
```

Review the compact verdict, skips, failures, exact effective configuration,
model/workload/source hashes, device identity, and unexplained-outlier count.
Inspect `.qualification/runs/<receipt-id>/` only for diagnosis; do not add raw
logs, traces, profiles, model output, or model weights to Git.

Compare only receipts accepted by the declared workload comparison policy:

```bash
python3 scripts/qualification/compare_receipts.py \
  qualification/receipts/rocm/strix-halo/<baseline>.json \
  qualification/receipts/rocm/strix-halo/<candidate>.json
```

The comparison command deliberately rejects mismatched source trees, models,
workloads, or undeclared configuration differences.

## Check In Evidence

Add only the compact receipt and the documentation or plan entry that explains
what it proves. Re-run portable validation on the staged receipt, inspect the
staged diff, then commit and push immediately so another machine can continue
from the same evidence chain.

```bash
git add qualification/receipts/<backend>/<host-id>/<receipt>.json \
  docs/plans/confidence-hardening-goal.md
python3 scripts/qualification/receipt.py \
  qualification/receipts/<backend>/<host-id>/<receipt>.json
git diff --cached --check
git diff --cached
git commit -m "Record <backend> <workload> qualification"
git push origin main
```

Before moving to another machine, verify `git status --short` is empty and
`git rev-parse HEAD` equals `git rev-parse origin/main`. Final cross-platform
claims require every relevant receipt to name one common source-tree hash.
