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
The runner also owns interruption containment. It starts each case in a new
process group and, on timeout or `Ctrl-C`, signals case descendants before the
outer sandbox leader so a Python serving driver can stop its independently
grouped server, delete private model snapshots, and publish no misleading
receipt. The default cleanup allowance is 65 seconds and may be set explicitly
with `--term-grace-seconds` up to the hard 75-second bound. If execution is
interrupted before receipt publication, the runner removes the exact
`.qualification/runs/<receipt-id>` directory transactionally and exits 130
without a traceback. A cleanup failure is itself a runner failure rather than
ignored residue. `SIGKILL`, kernel failure, or machine power loss cannot run
userspace cleanup; after such an event, verify process ownership before
removing the exact ignored directory.
Each case runs under `closed-qualification-case-v1`: a fixed base containing
only path, toolchain-home, locale, temporary-directory, user, and user-session
plumbing, followed by the committed case's exact `environment` object and the
runner-owned result-path and variant identifiers. Ambient backend controls,
device selectors, compiler/linker flags, product `KILN_*` values, credentials,
and library paths do not enter a case. Declare a required value in the workload
instead. The ignored effective-run artifact records the policy, redacted base,
base hash, exact overrides, and final per-case environment hash; the compact
receipt carries the same base hash as its execution-environment identity.
The ROCm mixed-load driver also rejects ambient `KILN_*` server controls before
building. Configuration changes must be declared in a committed workload
variant; inherited shell overrides are never silently ignored or accepted as
source-bound evidence.

No checked-in qualification case may invoke Cargo directly. Workload validation
rejects `cargo` by basename, including an absolute path. Standalone Rust test
cases use `scripts/qualification/cargo-test-bounded.sh`, which pins one job, a
50% aggregate CPU quota (at most half of one logical CPU on average), the
unchanged 15 GiB admission floor, offline Cargo, transient-service execution,
zero service swap, private networking, and a 1,740-second service cap. Its
`closed-qualification-test-v1` environment is the source-build allowlist plus
only the non-secret `KILN_QUALIFICATION` required-device gate; credentials,
ambient compiler flags, target directories, and unrelated `KILN_*` values do
not enter the service. The wrapper path and arguments remain part of the
committed workload and effective run artifact.

Source-building ROCm and Vulkan serving drivers likewise select an immutable
backend build specification, resolve the
requested toolchain, then execute the exact package, binary, feature, locked,
and offline build through `scripts/cargo-bounded.sh` with one job and a 15 GiB
`MemAvailable` floor. A systemd `CPUQuota=50%` bounds aggregate compiler,
linker, and helper CPU consumption even when a single Cargo job fans out inside
LLVM. ROCm alone receives `ROCM_PATH` and
`KILN_ROCM_ARCHS`; the Vulkan build strips ambient ROCm toolchain variables and
uses only the `vulkan` feature. The wrapper refuses overlapping
Cargo/rustc processes. Because the case retains bubblewrap PID isolation, the
offline build runs as a transient systemd user service rather than attempting
to attach the namespaced Cargo PID to a host scope. The service has an
aggregate `MemoryMax`, host reserve, zero swap, `PrivateNetwork=yes`,
control-group kill, a hard runtime cap, and a fail-closed package-temperature
watchdog. The typed build spec selects exactly one Linux hwmon input by
`name=k10temp` and `label=Tctl`, polls it every 250 ms, refuses to start at or
above 97,000 millicelsius, and stops the complete transient service if a later
reading reaches that limit. Missing, ambiguous, unreadable, non-integer, or
implausible telemetry also prevents or terminates the build. Ordinary ROCm and
Vulkan build services are capped at 840 seconds with a 900-second caller
timeout; the real-ROCm fault corpus uses 1140 and 1200 seconds respectively. This
60-second ordering ensures systemd can stop and collect the complete cgroup
before an outer qualification timeout can kill the wrapper. The measured server
still runs inside the case's separate network and PID namespaces. The committed
effective build config records the wrapper, job count, CPU quota, floor,
execution mode, private-network requirement, versioned environment policy, both
deadlines, memory policy, and all four thermal-selector fields. The driver mechanically
derives both `KILN_CARGO_CPU_QUOTA_PERCENT` and the wrapper's
`KILN_CARGO_HOST_THERMAL_*` controls from those typed fields; ambient values
cannot alter the quota, selector, or limit. `closed-source-build-v1` retains
only Cargo/Rustup homes, the pinned PATH and ROCm architecture/path, locale,
user/home, temporary-directory, and user-systemd connection variables. It
excludes ambient compiler flags, target directories, credentials, and API tokens before invoking the wrapper;
the wrapper independently applies the same policy when constructing the
service environment. Bounded stderr records the machine-specific
available/reserve/limit values. Do not lower the floor or bypass the wrapper to
obtain a receipt. Let the machine recover memory and rerun from the same clean
commit.

Outside a source-bound workload, `scripts/cargo-bounded.sh` also names every
ordinary systemd scope and stops that complete unit from its `EXIT` trap, so a
cancelled terminal or tool client cannot leave Cargo, rustc, or the linker
running in an orphaned scope. When exactly one `k10temp/Tctl` input exists, the
wrapper automatically applies the same 97,000-millicelsius, 250 ms guard to
ordinary commands and reports `thermal=automatic:...` in its preamble. A host
without that exact sensor runs with `thermal=disabled`; qualification source
builds do not accept that fallback because their four explicit typed fields make
missing or ambiguous telemetry a preflight failure. Operators may explicitly
configure a different stable selector only by setting all four documented
`KILN_CARGO_HOST_THERMAL_*` wrapper controls together.

For non-qualification use, `KILN_CARGO_CPU_QUOTA_PERCENT` optionally applies
the same aggregate systemd CPU limit in either execution mode; `100` represents
one logical CPU and values through `10000` are accepted. It is independent of
`KILN_CARGO_JOBS`: the latter limits Cargo's build graph concurrency, while the
quota contains all descendant threads. The wrapper reports the effective quota
in its preamble and rejects malformed or out-of-range values before launch.

Every serving driver creates a collision-resistant mode-0700 workspace below
`.qualification/serving` (the pressure driver uses
`.qualification/serving-pressure`). Names never depend on the sandbox PID,
because PID namespaces can assign the same PID on every run. Normal teardown
removes the workspace and private model snapshot. The runner sends `SIGINT` to
leaf case commands while keeping sandbox supervisors alive, allowing Python
`finally` owners to execute before hard containment. The ROCm mixed-load driver
also converts direct `SIGTERM` into a catchable interruption. A normal timeout
or `Ctrl-C` in that driver must therefore leave neither workspace nor copied
model payload. Treat residue after an uncatchable process or machine failure as
an explicit recovery condition: confirm no qualification or Kiln process still
references it before removing that exact directory.

Batching qualification must bind the complete typed startup policy, not only a
legacy actor environment switch. A serving workload that exercises the actor
declares `[batching]` values in its source-bound config, restarts the server,
and attests these exact runtime targets before measurement:

```text
GET /v1/config -> .batching.configuration
GET /v1/config -> .batching.actor_active
GET /v1/config -> .batching.direct_decode_rendezvous
GET /health -> .decode_runtime.batching_configuration
GET /health -> .decode_runtime.batching_engine
GET /health -> .decode_runtime.direct_decode_rendezvous
```

The immutable objects at `/v1/config`
`.batching.configuration` and `/health`
`.decode_runtime.batching_configuration` must be equal. The two actual
direct-rendezvous objects at `/v1/config`
`.batching.direct_decode_rendezvous` and `/health`
`.decode_runtime.direct_decode_rendezvous` must also be equal. `actor_active`
must agree with whether the optional live health `batching_engine` snapshot is
present and enabled; the snapshot is not itself equal to that boolean. The
attestation records mode
intent, backend default, effective selection and source; rowwise and
prefix-aware values and sources; admission quantum intent, backend default,
effective clamp and source; and backend-owned burst admission. A malformed
value, a canonical/deprecated-alias conflict, an unexpected source, or a
missing actor in an actor-required variant fails before device work. The direct
rendezvous policy within `batching.configuration` records configured,
backend-policy, effective, and source values for mode, max batch, wait
microseconds, and mixed sequence lengths. Its sibling status object records the
exact scope plus backend, actor, worker, and route availability. A worker may be
active while the route is unavailable because the actor is active.

Use only canonical mechanically derived names in new workload manifests:
`KILN_BATCHING_MODE`, `KILN_BATCHING_ROWWISE_DECODE`,
`KILN_BATCHING_PREFIX_AWARE_ADMISSION`, and
`KILN_BATCHING_PREFILL_ADMISSION_QUANTUM`, plus
`KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MODE`,
`KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MAX_BATCH`,
`KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_WAIT_US`, and
`KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MIXED_SEQ_LENS`. The eight historical
spellings are compatibility inputs for existing deployments, not qualification
vocabulary. A direct-rendezvous variant must disable the actor, restart, prove
`scope="direct_streaming_greedy_only"` and `route_available=true`, and send only
streaming effectively-greedy requests. Actor, sampled, and non-streaming runs
cannot qualify this fallback path.

Streaming-prefill qualification has the same source-bound rule. Declare the
complete `[streaming_prefill]` table in the committed variant, restart between
arms, and require these three serialized objects to be equal before device
work:

```text
GET /v1/config -> .streaming_prefill
GET /health -> .prefill_runtime.streaming_prefill
GET /v1/debug/model-state -> .streaming_prefill   # trusted-debug variant only
```

The attestation must retain configured source, backend dispatch, effective
dispatch and authority, threshold applicability, base/tape/detached effective
tiles and inheritance sources, derived detached boundary/replay tiles,
last-token LM-head policy, and both restart flags. New manifests use only
`KILN_STREAMING_PREFILL_MODE`,
`KILN_STREAMING_PREFILL_THRESHOLD_TOKENS`,
`KILN_STREAMING_PREFILL_TILE_TOKENS`,
`KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS`,
`KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS`, and
`KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD`. The six shorter historical names
and legacy TOML `enabled` exist for deployment compatibility, not new evidence.

A correctness arm should cover the prompt-length boundary immediately below
and at the effective threshold, compare forced `disabled` monolithic output
with forced `enabled` tiled output, exercise a prompt longer than one base tile,
and cover training work longer than the effective tape and detached tiles when
that backend supports the route. Record exact tokens/losses, cancellation and
settlement, TTFT/prefill time, per-stage timing, peak primary and host-backed
memory, allocator reclaim/resize events, device synchronizations, and any
non-finite or ownership failures. Do not interpret `mode="enabled"` on CPU or
Vulkan as support by itself; it is an explicit test route and still needs real
backend evidence.

For pause or OOM diagnosis, run one variable per arm: backend `auto`, forced
`disabled`, then a changed threshold or one tile. An explicit base tile feeds
`auto` tape and detached routes, so either set those specialized values
explicitly or record their inherited effective values. Attribute a pause only
when the trace shows the corresponding scheduler wait, tile computation,
external-yield synchronization, allocator reclaim, physical KV resize, or
memory-pressure event. A temporal gap alone is not evidence of VRAM
rebalancing. These ROCm, Vulkan, CUDA, and Metal runs belong on the named local
machines; hosted GPU CI is neither required nor accepted as qualification.

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

#### Current Vulkan prefill result

The source-bound 2026-07-15 run on RADV Strix Halo passed from clean commit
`d6d14bfaf` and tree hash
`sha256:0a135f0ad1eca6ccc1bfa6df503b9d1f7c9a0f684a993600417135f439fc0f5b`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260715t000104133598z-vulkan-strix-halo-prefill-scheduling-v1-6899c96516-v1.json`.
The headless probe enumerated `GPU0`; the literal short-decode/1K/16K actor
case passed its token-budget fairness and cancellation-cleanup contract; and
the production Vulkan hybrid-model case matched monolithic prefill across
quanta `[17, 17, 17, 17, 12, 1]`, six layer yields, block-aligned recurrent
state and prefix cache, first token 13, following decode token 13, and final
KV-block release. Both Rust cases ran through the bounded private-network
service with no ignored tests or output-assertion failures.

The receipt's execution identity is `closed-qualification-case-v1` hash
`sha256:4e2f59070351a66c62498c8952245ba078b66548fd1f7b7b4083b32b2ab41a93`.
Its effective-run artifact proves that neither `ROCM_PATH` nor
`KILN_ROCM_ARCHS` entered the Vulkan base environment. This fixture uses a
small deterministic hybrid model; it proves the named scheduling and state
transitions, not public-model tokenizer/logit/sampling parity or serving
performance.

Vulkan core correctness:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  qualification/workloads/correctness-core-v1.json
```

Both lower-level Vulkan workloads run `vulkaninfo --summary` without inheriting
`DISPLAY`. Presentation and surface discovery may therefore report that their
headless-only information was skipped. That is not a device skip: the probe
must still exit zero and emit a concrete `GPU<n>:` entry. Missing entries, “no
Vulkan device,” or an explicitly skipped Vulkan or physical device fail the
case.

### Current Vulkan core result

The source-bound 2026-07-14 run on RADV Strix Halo passed from clean commit
`ea4c0775f` and tree hash
`sha256:9f25964ce481ac7eb09c4e23d86c17ee631b6abd2a00bcd53efc61305d7b4e2f`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260714t235017349724z-vulkan-strix-halo-core-correctness-v1-0a09f3bcee-v1.json`.
The required device probe selected the AMD Radeon 8060S Graphics through RADV
Mesa 26.1.3. The bounded, private-network Cargo service then ran 16 Vulkan
tensor, cast, transfer, reduction, reshape, and autograd parity tests and nine
dense, transposed, batched-BF16, and LoRA-composition matmul parity tests. All
25 passed, none failed or were ignored, both output-skip guards remained clear,
and the complete receipt plus every local artifact hash validated against the
current source before documentation mutation.

This lower-level receipt does not load a model and therefore makes no claim
about tokenizer behavior, model logits, sampling, paged-cache lifecycle,
cancellation, eval, public serving, or throughput. Those remain separate
source-bound Vulkan gates.

### Vulkan inference oracles

Run the model-level deterministic oracle workload from a clean pushed tree:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  qualification/workloads/vulkan-inference-oracles-v1.json
```

The six required cases are sequential and network-isolated:

1. A headless `vulkaninfo` probe must enumerate a physical device.
2. Rows4 and rows8 BF16 fused argmax, including tail rows, must match CPU
   projection and tie-breaking expectations.
3. Greedy and non-greedy fused sampling must match a separate CPU contract
   across temperature, top-k, top-p, min-p, repetition/presence/frequency
   penalties, six fixed seeds, F32, BF16, and batched paths.
4. F32/BF16 selected log-probs, FLCE/GRPO losses, and backward gradients must
   match analytical CPU, finite-difference, and pinned TRL/PyTorch references.
5. Six source-pinned HF-derived Qwen chat-rendering, token-ID,
   assistant-mask, label, and tokenizer/config/template-hash cases must match
   exactly.
6. After identical Qwen prefill and decode state, all full-vocabulary logits
   from the production resident and nonresident Vulkan paths must satisfy the
   declared `1e-4` relative tolerance. Qualification fails if the model,
   Vulkan runtime, or resident pool is unavailable.

Every Rust case enters `scripts/qualification/cargo-test-bounded.sh`: offline,
one job, a 50% aggregate CPU quota, private network, zero service swap, 17 GiB
aggregate ceiling, 1,740 second deadline, and the unchanged 15 GiB host-admission
floor. The runner
derives `KILN_QUALIFICATION_MODEL_PATH` from the manifest's `${model_path}` and
the inner closed Cargo policy forwards only that test binding plus
`KILN_QUALIFICATION`; it is not a product runtime setting. Ordinary source
builds, credentials, compiler flags, backend controls, and unrelated
`KILN_*` values remain excluded.

#### Current Vulkan inference-oracle result

The source-bound 2026-07-15 run passed from clean commit `8a1edd250` and tree
hash `sha256:1ac91d8cea7f50eeaa53875fc7fe2559a99a1e1b7703e3110d2e94693e2d1c1a`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260715t002226653824z-vulkan-strix-halo-vulkan-inference-oracles-bfc6c14dd8-v1.json`.
All six cases passed with zero ignored tests or output-assertion failures. The
real Qwen case took 368.881 seconds and returned bit-identical logits across
all 248,320 entries (`max_abs=0`, `max_rel=0`); total workload time was 382.453
seconds. Systemd recorded a 17 GiB service-memory peak. Live cgroup monitoring
observed ceiling events but zero OOM/OOM-kill and zero service swap, and host
availability returned to 25 GiB after teardown.

This receipt closes deterministic tokenizer, lower-level CPU/TRL sampling and
selected-logprob math, and resident/nonresident Qwen equivalence. The Qwen
full-vocabulary comparison uses two Kiln Vulkan paths, not an independent HF
forward. It therefore does not close the independent public-model HF-logit,
broader prefix-cache/cancellation, eval-execution, soak, or throughput gates.

### Vulkan cache, cancellation, and live eval

Run the model-route workload from a clean pushed tree. It uses a deterministic
small BF16 hybrid model and therefore does not need `--model` or network
access:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  qualification/workloads/vulkan-model-routes-v1.json
```

The manifest runs a required headless physical-device probe followed by one
contained, sequential Rust case. Qualification mode makes a missing Vulkan
device, a skipped hardware test, a missing pass marker, a test failure, or any
output-assertion mismatch fail the workload. The Rust case proves four routes:

1. A cancelled retained prefill and an explicitly discarded retained prefill
   release every KV block, prefix-cache lease, recurrent-state entry, and
   pending-release record.
2. The production batching forward retains only a safe block-aligned strict
   prefix. An identical request and a longer second turn both hit that entry,
   reuse its blocks, and retire all leases without mutating a shared partial
   block.
3. The nonbatched generation path restores both paged KV state and the hybrid
   model's GDN linear state from a split entry. Its generated token IDs must be
   identical to an uncached full prefill; a plumbing-only cache hit is not
   sufficient.
4. `LiveEvalGenerator` and the real eval executor send a one-example exact-match
   suite through the production batching actor, `RealDecodeForward`, and the
   native Vulkan model runner. The deterministic fixture must consume six
   prompt tokens, emit exactly two completion tokens decoding to `t1 t1`, and
   report one pass with zero failed, invalid, or errored examples.

The focused eval fixture starts with `AppState`'s dependency-complete mock
construction and replaces only its inference backend with the real bounded
Vulkan runner, cache objects, forward implementation, and batching actor. This
avoids attempting a second process-global production memory admission inside
one test process; token generation, scoring, cache ownership, and backend
execution are not mocked. The test also requires `ModelRunner::backend_name()`
to be `vulkan` for each exercised model path and explicitly stops the batching
actor before returning.

This qualification route exposed a serving-profile bug in the eval adapter
transition. Base-model eval previously marked a base-to-base selection as
changed content, so stable serving correctly rejected it as a live weight
mutation. Base eval now performs a non-mutating selection. A nonempty named
adapter still always requests content reload because an adapter may have been
retrained under the same serving name and name equality cannot prove that its
weights are current.

#### Current Vulkan model-route result

The source-bound 2026-07-15 run passed from clean commit `bcb245ac7` and tree
hash `sha256:7cd165f542fba1e855a4a915516952b02038e2001fc1ef028d149e04fac54d16`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260715t004650723123z-vulkan-strix-halo-vulkan-model-routes-v1-e2287dab6c-v1.json`.
Both required cases passed with zero output-assertion failures on AMD Radeon
8060S Graphics, RADV Mesa 26.1.3. The model-route case took 2.543 seconds and
the complete workload took 2.760 seconds. The receipt and all eight local
artifact hashes validated against the known current commit before this
documentation mutation; teardown left no qualification service or build
process and host availability returned to 24 GiB.

This receipt closes the named deterministic cache lifecycle, cancellation, and
live eval execution subsets. It does not compare full-model logits against an
independent Hugging Face forward, measure public-model eval quality, exercise a
public HTTP eval job, establish long-duration stability, or make a throughput
claim. Those remain separate qualification gates.

### Vulkan independent Hugging Face full-model oracle

Run the independent full-model oracle only from a clean pushed tree. Prepare a
Python executable with the exact ROCm PyTorch and package versions in
`scripts/hf_trl/requirements-sft.lock` as described in
[HF/TRL Interoperability](HF_TRL_INTEROP.md). The oracle additionally pins the
PyTorch commit and hashes the installed Qwen3.5 Transformers modeling and
configuration modules. Optional fused linear-attention packages must be absent:
the reference deliberately uses Transformers' independent torch fallback.

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  --var trainer_python=/absolute/path/to/pinned-venv/bin/python \
  qualification/workloads/vulkan-hf-full-model-oracle-v1.json
```

The single required case executes two accelerator stages sequentially. It
never holds the model in ROCm and Vulkan at the same time:

1. The HF stage requires at least 23 GiB `MemAvailable`, then creates a
   private-network systemd service with `MemoryMax=16G`, zero service swap, a
   600-second cap, and control-group teardown. Inside that service, the thermal
   supervisor validates the content-hashed
   `qualification/host-policies/strix-halo-hf-oracle-v1.json` policy, requires
   20 consecutive 50 ms package samples at or below 45 C, and only then creates
   the accelerator worker in a new process group. The worker blocks on a private
   start gate until the supervisor has attached the continuous guard.
2. The guard samples the exact `k10temp/Tctl` input every 50 ms, stops the
   complete worker process group at 58 C, resumes it after 20 consecutive
   samples at or below 50 C, and terminates the group at 93 C. It remains armed
   through eager BF16 model load and forward, then requires the dead worker's
   host package to produce 20 consecutive samples at or below 45 C before the
   service can report success. Missing or ambiguous sensors, malformed samples,
   unreconciled stops, a trip, or a 300-second cooldown timeout fails closed.
3. The pinned worker uses deterministic algorithms, disables TF32, evaluates
   fixed input IDs `[1,2,3,4,5,6,7,8,100]`, and writes one safetensors artifact
   containing the input IDs and all 248,320 final-position F32 logits. It reads
   its own cgroup-v2 memory peak, swap, and
   OOM/limit events before it exits. This self-report is intentional: the
   qualification runner's bubblewrap PID/network namespace can launch and wait
   for a user service, but a post-exit `systemctl --user show` cannot reconnect
   to the host user bus. `systemd-run --wait` remains the exit verdict, and a
   missing, malformed, swapped, or OOM-bearing telemetry record fails closed.
4. After HF exits and its cooldown completes, the driver requires 24 GiB
   `MemAvailable` before Vulkan can
   start. The Rust test runs through `cargo-test-bounded.sh` with offline Cargo,
   private networking, `CPUQuota=50%`, `MemoryMax=17G`, zero service swap, a
   1,740-second cap, and a seven-GiB host reserve. The closed qualification
   environment forwards only the runner-owned model and HF-reference paths plus the hardware gate.
5. Kiln loads the same weights and input IDs through both its production
   resident and nonresident Vulkan paths. Those two paths must remain
   bit-identical. The resident result is then compared with every HF logit;
   argmax must match exactly, top-10 overlap must be at least 9/10, maximum
   absolute error at most `0.5`, mean absolute error at most `0.05`, and cosine
   similarity at least `0.9999`. Non-finite values, a missing device, an ignored
   test, an incomplete result, or any threshold failure rejects the workload.

The compact result records the comparison metrics, HF cgroup peak and service
swap, the deterministic raw-logit tensor hash, the independently computed
reference-artifact hash, memory ceilings, attention routes, exact input IDs,
content-hashed thermal policy, prelaunch samples, runtime package peak, pacing
count and duration, and post-exit cooldown evidence. Raw model output remains
below the ignored `.qualification/` run tree.
Validate a new receipt before changing documentation:

```bash
python3 scripts/qualification/receipt.py \
  --require-current-source \
  --require-local-artifacts \
  --require-known-commit \
  qualification/receipts/vulkan/<host>/<receipt>.json
```

#### Current Vulkan/HF full-model result

The source-bound 2026-07-15 run passed from clean commit `4d6697c52` and tree
hash `sha256:b21d95b47650ee831d27a678e85b3842d369b6bc78e0f21ec84c9a9da65bcfa4`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260715t013710403012z-vulkan-strix-halo-vulkan-hf-full-model-ora-39c1bc8042-v1.json`.
The 390.443-second workload compared all 248,320 logits and passed with exact
argmax 1, 10/10 top-token overlap, maximum absolute error `0.12433958`, mean
absolute error `0.020353988`, and cosine similarity `0.999941539769`. The HF
stage reported a 9,254,346,752-byte cgroup peak, zero service swap, zero high,
limit, OOM, and OOM-kill events, and deterministic raw-logit hash
`sha256:0b902c0d74a8ed54aefefcdab50adeb6fedd7adb3e45a2338c27276e90abeeaf`.
Kiln's resident and nonresident Vulkan results were bit-identical.

Live host monitoring observed the Vulkan service contact its 17 GiB cgroup
ceiling without OOM or service swap; system-wide swap rose by roughly 0.4 GiB
while host available memory remained above the reserved floor, then 24 GiB was
available after teardown. This is retained as a host-pressure signal for the
development soak rather than hidden behind the passing numerical verdict.

This historical receipt closes the numerical Phase 6 independent CPU/HF
full-model comparison when combined with the tokenizer, sampling,
selected-logprob, cache, cancellation, and live-eval receipts above. It predates
the now-required thermal supervisor evidence, so it does not satisfy the current
workload manifest or the final common-tree gate without a rerun. It covers one
deterministic next-token full-vocabulary forward. It does not establish
multi-token public-model output parity, public HTTP eval quality, long-duration
stability, large-batch throughput, or competitive performance against vLLM.

### ROCm serving first-divergence oracle

Use the focused ROCm oracle only after a source-paired exact greedy comparison
has retained both engine outputs and localized their first different token. It
is a diagnostic correctness gate, not a serving throughput measurement. The
tracked request
`qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json`
binds the exact Kiln and vLLM receipt file/content hashes, their common source
commit, the model weights/config/tokenizer/template identity, the original
user message, chat-template invocation, 163 prompt token IDs, and the first
three common continuation tokens `[1206,5517,264]`. Its complete 166-token
input has canonical hash
`sha256:709d0a314cde9072ac79b0752e795f1b76bfaea5b553ccdf26f7fbd5ac44b1a0`.
The two declared candidates are Kiln token `25045` (` baseline`) and vLLM token
`15787` (` foundation`).
The machine-readable field references are published as
[HF Next-Token Request Schema](https://ericflo.github.io/kiln/docs/hf-next-token-request-schema/)
and
[ROCm HF Next-Token Result Schema](https://ericflo.github.io/kiln/docs/rocm-hf-next-token-result-schema/).
Both close every object to unknown fields; the executable validator additionally
enforces cross-field hashes, token concatenation, source-receipt contents,
candidate ranks, thermal reconciliation, and the canonical result self-hash.

Run it only from a clean `HEAD` already pushed to `origin/main`; every path
below is intentionally absolute:

```bash
python3 scripts/qualification/rocm_hf_next_token_oracle.py run \
  --model "$(pwd)/Qwen3.5-4B" \
  --trainer-python "$(pwd)/target/qualification/hf-trl-roundtrip/.venv/bin/python" \
  --request "$(pwd)/qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json" \
  --host-thermal-policy "$(pwd)/qualification/host-policies/strix-halo-hf-oracle-v1.json" \
  --out "$(pwd)/.qualification/rocm-hf-next-token-result.json"
```

Before the service exists, the driver strictly validates both source receipts,
hashes the complete model content, verifies at least 23 GiB `MemAvailable`, and
requires `HEAD == origin/main`. Model hashing occurs before the supervised
prelaunch cooldown so provenance work cannot hand a hot package directly to
the accelerator. The systemd service then supplies the same 16 GiB memory,
zero-swap, private-network, 600-second control-group boundary used by the full
Vulkan oracle. The gated worker independently reloads the pinned tokenizer and
must reproduce every prompt ID and every retained token's decoded bytes before
loading the model. It pins ROCm PyTorch, Transformers sources and versions,
eager attention, the Transformers torch linear-attention fallback, BF16,
deterministic algorithms, and TF32-off execution.

The compact result retains the full-vocabulary F32-logit hash, argmax and text,
top ten token IDs/text/logits, both candidate logits and vocabulary ranks,
argmax margin, package versions, cgroup peak/limit/OOM/swap counters, thermal
policy and lifecycle, clean pushed source identity, implementation hashes, and
an explicit `kiln`, `vllm`, or `neither` attribution. The roughly one-MiB raw
logit artifact remains ignored. The result has a canonical `result_sha256` and
can be checked without running the model:

```bash
python3 scripts/qualification/rocm_hf_next_token_oracle.py check \
  .qualification/rocm-hf-next-token-result.json \
  --require-current-source
```

Specialized oracle results live under `qualification/oracle-results`, separate
from generic qualification receipts and serving-benchmark receipts because
each family has a different closed schema and validator. Both portable
workflows discover and check every JSON result in that tree without running an
accelerator. The current retained Strix Halo result is
`qualification/oracle-results/rocm/strix-halo/20260719t003452-rocm-strix-halo-hf-next-token-first-divergence-v1.json`,
executed from clean pushed source `c5640c090f295cd50a73aae63ef48d403fd13d98`.
The eager HF reference selects vLLM's token `15787` (` foundation`) at rank one
with logit `18.5`; Kiln's token `25045` (` baseline`) is rank two with logit
`18.25`, a `0.25` top-logit margin. The full F32 logit vector has canonical
hash `sha256:d4bc2aeb7d6bef608dfe500f6b9759b1fd4ca5eeb924b55b9e63b2e49a0e96d0`.
The guarded forward completed without memory, swap, OOM, thermal-trip, or
cooldown residue: the cgroup peaked at 9,529,192,448 bytes, runtime temperature
peaked at 60 C, and four completed pacing events occupied 9.381 seconds. This
closes the candidate attribution at that exact first divergence: vLLM matches
the independent reference and Kiln does not. It does not yet locate Kiln's
numerical error or establish broader sequence parity.

#### ROCm numerical-path attribution

Use the path-attribution runner after the independent HF result above has been
retained and its ignored raw safetensors artifact is still available. This is a
Kiln defect-localization workload, not a new oracle and not a performance
benchmark. It loads the model once and evaluates four distinct routes against
the same 248,320-element F32 HF logit vector:

1. `eager_full` uses the ordinary paged prefill/decode path and reads back the
   complete final logit vector.
2. `eager_greedy` uses the production eager greedy-selection API, isolating
   selection from full-logit readback.
3. `graph_full` primes one retained HIP-graph slot, releases its logical row,
   and requires all three exact continuation steps to replay before comparing
   the complete final logit vector.
4. `graph_greedy` uses a separate recurrent state and logical row and requires
   the same three retained replays through the production graph greedy API.

Each route starts from an independent KV cache and linear-attention state. The
worker forces the request's three known-common continuation tokens rather than
feeding one route's prediction into the next step. Therefore, all four final
comparisons describe the same 166-token state immediately before the disputed
fourth generated token. The graph prime reproduces the retained-slot condition
of the source serving process; a warmup-only call or a capture without replay
does not pass.

Run only from a clean commit already present at `origin/main`. The raw HF
reference from the accepted run currently lives at
`.qualification/.rocm-hf-next-token-result.artifacts/hf-reference.safetensors`
and is deliberately ignored; the runner accepts it only when its size and hash
match the retained compact oracle result. All paths are intentionally absolute:

```bash
python3 scripts/qualification/rocm_hf_path_attribution.py run \
  --model "$(pwd)/Qwen3.5-4B" \
  --request "$(pwd)/qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json" \
  --oracle-result "$(pwd)/qualification/oracle-results/rocm/strix-halo/20260719t003452-rocm-strix-halo-hf-next-token-first-divergence-v1.json" \
  --hf-reference "$(pwd)/.qualification/.rocm-hf-next-token-result.artifacts/hf-reference.safetensors" \
  --host-thermal-policy "$(pwd)/qualification/host-policies/strix-halo-hf-oracle-v1.json" \
  --python "$(pwd)/target/qualification/hf-trl-roundtrip/.venv/bin/python" \
  --out "$(pwd)/.qualification/rocm-hf-path-attribution-result.json"
```

Before device execution, the runner revalidates the request, both source
receipts, retained HF result, full model content, raw reference, interpreter,
clean source identity, and at least 23 GiB of host-available memory. It performs
an offline locked `gfx1151` release build through `scripts/cargo-bounded.sh`
with a 15 GiB build floor, 50 percent CPU quota, zero swap, private networking,
and the versioned `closed-source-build-v1` environment policy. Ambient
`KILN_*`, compiler flags, credentials, and target paths cannot enter that
service; only the runner-owned build controls and `KILN_ROCM_ARCHS=gfx1151`
do. The result records both the architecture and build-policy identity. The
worker itself receives a closed six-variable environment with no product
configuration variables and installs the qualified model/tensor policy,
`auto` kiln-tensor API mode, and legacy host barriers explicitly before the
primary ROCm context exists.

Execution is a private-network systemd user service with `MemoryMax=48G`, zero
swap, control-group kill semantics, and a 900-second outer lifetime. The
existing thermal supervisor completes the stable prelaunch boundary, starts a
new worker process group blocked on a private file gate, attaches the thermal
guard, and only then releases execution. The generic guarded-exec shim validates
the exact argv, working directory, complete environment, executable path, and
SHA-256. It holds an open descriptor across the gate and executes
`/proc/self/fd/<n>`, so replacing the build pathname after validation cannot
change the admitted inode. A pass additionally requires worker-observed cgroup
v2 limit/peak/current counters, zero high/limit/OOM/OOM-kill events, zero swap,
complete thermal settlement, and no graph fallback or replay failure.

The compact result schema is published as
[ROCm/HF Path Attribution Result Schema](https://ericflo.github.io/kiln/docs/rocm-hf-path-attribution-result-schema/).
It retains source/tree identity, binary and implementation hashes, request and
oracle references, model identity, full containment evidence, both full-logit
hashes and error summaries, both candidate logits/ranks, all four observed
token sequences, graph capture/replay counters, and one mechanically derived
attribution:

- `eager_full_logits`: the ordinary eager numerical path already disagrees.
- `hip_graph_full_logits`: eager full logits agree, but retained graph logits do
  not.
- `eager_greedy_selection`: full logits agree, but eager selection does not.
- `hip_graph_greedy_selection`: graph full logits agree, but graph selection
  does not.
- `serving_only_or_not_reproduced`: all isolated routes agree; the serving-only
  composition or a non-reproduced condition remains.

These labels identify the first tested boundary that disagrees; they do not by
themselves identify a layer or kernel. The executable checker recomputes the
label and canonical self-hash and cross-binds the observed prefix, candidates,
input hash/count, model, original receipts, retained HF result, and raw-artifact
identity. Check one result directly with:

```bash
python3 scripts/qualification/rocm_hf_path_attribution.py check \
  .qualification/rocm-hf-path-attribution-result.json \
  --require-current-source
```

The current retained Strix Halo result is
`qualification/oracle-results/rocm/strix-halo/20260719t014807-rocm-strix-halo-hf-path-attribution-v1.json`,
executed from clean pushed source `034e548e86910c08080d43c7c4196103fce2f9f3`.
It reports `eager_full_logits`: eager and retained-graph full logits are
bit-identical with canonical vector hash
`sha256:b5acbda785044ca46d6cddb9aea03258dd4f99fb1904dcb5983be49ef68fd603`,
and the eager and graph greedy routes select that vector's argmax. All four
routes emit the common prefix `1206, 5517, 264` followed by Kiln token `25045`.
At the disputed position, Kiln ranks token `25045` first at `18.625` and HF's
token `15787` second at `18.5`; the independent HF vector instead ranks token
`15787` first at `18.5`. Relative to HF, Kiln has `0.998147822` cosine
similarity, `0.84375` maximum absolute error, `0.140939553` mean absolute
error, and nine of ten top tokens in common.

The retained graph evidence is one successful capture and seven successful
replays with zero capture failure, replay failure, or fallback. This rules out
HIP-graph replay and greedy selection as the first tested divergence boundary;
the defect is already present in the ordinary eager full-logit forward. The
112.012-second guarded lifecycle peaked at 14,657,327,104 cgroup bytes and
59.75 C, recorded zero memory-limit, OOM, swap, or thermal-trip events,
completed 28 thermal pacing events totaling 64.109 seconds, and handed the
host back at 42.75 C. The result does not identify the first divergent model
layer or kernel. The next bounded probe must compare layer-boundary outputs for
this exact 166-token state before changing serving configuration or repeating
the concurrency matrix.

Portable repository gates discover heterogeneous files under
`qualification/oracle-results` and dispatch each declared schema through
`scripts/qualification/check_oracle_results.py`. Unknown schemas fail instead
of being interpreted as a different result family. After a passing local run,
retain only the compact JSON result in that tree; keep the raw logits and
execution workspace ignored. A route attribution is sufficient to choose the
next narrower numerical probe or repair, but it does not satisfy multi-token
parity, throughput, soak, or final common-tree acceptance.

An attributed argmax identifies which engine selected the eager HF reference's
top token at the first divergence. It does not prove multi-token parity, explain
the losing engine's numerical defect, accept either thermal pacing policy for
performance, or replace the complete concurrency/profile matrix.

Model-serving workloads additionally require `--model` with the exact local
model directory and `--model-id` with its public identity. Select each declared
A/B arm explicitly; the manifest, not an ambient environment variable, owns
the effective configuration recorded in the receipt.

The source-bound serving drivers materialize that declared policy as a private TOML
file inside the ignored run directory and start `kiln serve --config <file>`.
Server profile, bind address, model/snapshot/adapter paths, thinking default,
transport bounds, scheduling ceilings, logging, memory reclaim, synchronization,
graph mode, graph entry capacity, graph byte capacity, and Vulkan decode-weight
prewarm enable/rate policy therefore travel
through the same typed parser and source diagnostics as an operator config.
The process environment is scrubbed of ambient `KILN_*` controls before build
and launch. `memory.kv_autoscale` now carries both enabled and disabled requests.
Ordinary serving arms write `memory.kv_force_blocks = 0`; the dedicated
maintenance arm writes its declared positive target. Health and debug must
report `config_file` provenance for both fields. The same TOML sets
`server.debug_model_state = true` to grant trusted readback without enabling
eval-mode request semantics. No `KILN_*` launch exception remains; `RUST_LOG`
is the ordinary tracing filter. Qualification rejects every ambient runtime
control, including the deprecated `KILN_KV_AUTOSCALE` and
`KILN_KV_FORCE_BLOCKS` aliases.

The mixed-load and development-soak drivers bind the response policy
`ascending_zero_padded_integers_prefix_v1` in their effective configurations.
For every ordinary, warmup, stabilization, measured, and intentionally
cancelled request, the client concatenates streamed `delta.content` fragments
in order and requires the result to be a nonempty prefix of `000000 000001
000002 ...`. Tokenizer-dependent fragment boundaries and a final partial
integer are allowed; repeated numbers, punctuation, newlines, commentary,
reasoning content, tool calls, empty special-token output, multiple choices,
and malformed semantic events are not. Protocol success, positive usage, and
exact length termination are therefore necessary but not sufficient for a
passing request. A row that fails this oracle is excluded from successful
latency and throughput aggregates and increments the request-failure count.
Failure details retain the oracle reason, exact accepted token IDs when
performance metadata is enabled, and an escaped output excerpt capped at 256
characters. This bound preserves actionable corruption evidence without
allowing model output to exhaust the result-detail envelope.

The measured mixed-load prompt identity is
`variant_invariant_fixed_output_v5`. It requests exactly 64 ascending six-digit
integers and tells the model that server truncation before the target is
expected. At roughly seven model tokens per formatted integer, the target
remains beyond the 32-, 128-, and 256-token measured response caps without
inviting the output-limit refusals observed at one million and 1,024 requested
integers. The instruction requires immediate sequence output and forbids early
stopping, end markers, explanation, refusal, summaries, or discussion of output
limits. Prompt-length padding uses nonnumeric `itemNN` tokens and explicitly has
no relationship to response length. The separate unvalidated stalled-socket
request retains a 1,024-integer fill target so its 4,096-token response envelope
does not weaken. Every arm records the prompt identity,
`response_oracle_target_integer_count = 64`,
`slow_response_target_integer_count = 1024`, and
`long_prefill_marker_role = "long-prefill"`. The diagnostic and acceptance
paths therefore tokenize the same named long request; a wording, denominator,
fill target, or marker-role change invalidates direct A/B configuration identity.

Before the ordinary mixed-load measurement window, every ROCm arm also runs a
separate fixed-seed sampled profile at concurrency eight: 32 tokens per request,
temperature 0.7, top-p 0.9, top-k 40, and min-p 0.0. Those requests disable
thinking, ignore EOS, and retain terminal performance metadata, but use a
sampling-appropriate semantic oracle: exactly one nonempty plain-text choice,
with no reasoning content or tool calls. Random output is not compared with the
greedy ascending-integer sequence. The driver waits for all eight streams and
the batching engine to drain, reattests runtime policy, and captures a fresh
health baseline before starting the original workload. Sampled counters and
latencies therefore remain dedicated evidence rather than silently changing
the long-standing deterministic mixed-load denominator.

The mixed-load client and server each use the same 180-second per-request
containment bound. Cooling remains inside that wall clock. The alignment lets
the deliberately long prefill reach the server's terminal result under required
host pacing instead of having the client abandon a still-valid request at an
older, shorter local deadline; it does not remove or normalize the measured
TTFT, E2E latency, workload duration, or sustainable throughput cost.

Measured mixed-load, development-soak, and endurance receipts also retain the
complete validated request-phase population instead of discarding terminal
performance metadata. Every fixed phase `P` has a
`latency_phase_P_ms_total` and `latency_phase_P_request_count`; nullable phases
contribute neither time nor count, while a measured zero contributes one count
and zero time. `latency_phase_metadata_missing_count` must remain zero. Broad
actor phases contain narrower backend candidates and delivery can overlap the
next forward, so these totals rank candidates within their documented layer;
they must not be summed into a synthetic critical path. This makes pre/post
optimization receipts sufficient to decide whether sampling, readback,
synchronization, queueing, or model execution actually moved.

The sampled wave emits `sampled_profile_*` request, fixed-output, phase,
throughput, and batching metrics. It requires a measured `sampling_ms` value on
all eight ROCm requests, positive total sampled-tail and actor-decode duration,
zero separate `readback_ms` observations, no batching errors, at least one
batched decode forward, more decode rows than forwards, and an observed batch
width of at least two. `sampled_profile_output_token_throughput_per_second` is
the aggregate 256-token completion count divided by the concurrent wave wall
window. `sampled_profile_per_request_output_token_throughput_per_second_p50` is
the median of each request's completion count divided by its own end-to-end
wall time. Neither is a transformer-only kernel rate, and the narrower sampling
phase remains contained by the broad actor `decode_ms` interval.

Every ROCm mixed-load arm applies the same independent host thermal policy from
server launch through readiness, warmup, the isolated sampled wave, ordinary
measurement, drain, server exit, and a bounded post-exit cooldown. The typed
`host_safety` object selects
exactly one Linux hwmon input by `name=k10temp` and `label=Tctl`, polls every
250 ms, pauses the complete server process group with `SIGSTOP` at or above
88,000 millicelsius, and resumes it with `SIGCONT` at or below 86,000. A
97,000-millicelsius reading, missing or ambiguous selector, malformed input,
controller error, or signal error fails closed and terminates the server group.
Cooling consumes existing wall-clock and request deadlines, so throughput
includes the host's sustainable pacing cost rather than excluding it. Pacing
intervals join ITL attribution as `host_thermal_pacing`. Before sending the
server its teardown signal, the controller atomically disables new pacing and
releases any stopped process group while continuing to enforce the 97 C hard
limit. This prevents a late `SIGSTOP` from delaying or stranding shutdown.

After the server exits, the controller keeps sampling until eight consecutive
250 ms observations are at or below 75,000 millicelsius. The first cool reading
does not suffice: the consecutive-sample condition protects against the package
temperature rebound observed after an earlier run returned control at 88.75 C.
The cooldown is bounded to 180 seconds. A sensor failure, hard-limit reading, or
timeout remains a failed qualification; the dead server cannot resume work, and
the controller continues the bounded cooldown attempt rather than silently
calling the host ready.

The receipt retains start/peak/end package temperature, guard error and trip
counts, pacing start/completion counts, total and maximum pacing seconds,
hottest pacing start, and whether a pause remained active at teardown. Adaptive
ITL evidence retains the total attributed count and partitions it into
`host_thermal_pacing_itl_outlier_count` and
`non_thermal_attributed_itl_outlier_count`. A mixed-load pass permits the first
only when every attributed gap reconciles to one of those two counts; it still
requires both the non-thermal attributed count and unexplained count to be zero.
This makes the required external controller visible without allowing a model,
scheduler, allocator, graph, or synchronization pause to hide behind it. It also
retains cooldown active/completed/timeout counts, duration, sample count,
consecutive stable count, and post-exit peak. A pass requires zero guard error,
trip, cooldown timeout, or active final controller; exactly one completed
cooldown; and identical started and completed pacing counts. A pacing interval
that ends because the protected process has exited is completed with an explicit
`process_exited` reason rather than misreported as an interruption. These checks
apply to the short mixed-load gate, the resident-prefill oracle, and the longer
development/endurance soaks; the source-build watchdog is a separate guard
around the compiler/linker service.

### Vulkan serving baseline

Run this only after the required ROCm receipts have passed on the same clean,
pushed source tree. It is the first full-server Vulkan gate; the earlier core
and prefill workloads exercise lower-level routes but do not prove the public
SSE server, source-built executable identity, batching actor, or teardown.

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan-serving-baseline \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-vulkan-baseline-v1.json
```

The driver builds `kiln-server` once with `--no-default-features --features
vulkan`, through the bounded wrapper, offline, with one compile job and the
unchanged 15 GiB host-memory admission floor. The compile window is 900 seconds
and is separate from the 240-second server-readiness window. The generated
private TOML selects the experimental profile only to keep the batching actor
available; it explicitly disables KV autoscaling, allocator reclaim, and ROCm
graphs. Runtime attestation requires all three policies and their
`config_file` provenance before measurement and again after the final wave.
Readiness requires both the passing health check and causal log evidence. The
Vulkan-native path satisfies the latter with `Vulkan decode weight prewarm
complete`; its typed 256 MiB/s rate shapes startup pressure, and shutdown must
join the task before reporting a clean stop. It does not run or claim the
synthetic inference prewarm used by other serving paths.

After one fixed warmup, four thread-barrier waves dispatch concurrency 1, 4, 8,
and 12 with mixed prompt lengths from 16 through 1,024 deterministic words.
All 25 measured requests disable thinking and sampling, ignore EOS, stream
usage and performance metadata, and must finish by the exact 32-token limit.
The 600-second per-request limit is a correctness-containment deadline for the
longest synchronized prefill, not a latency SLO or a passing performance
threshold. Actual TTFT, end-to-end duration, output throughput, active width,
and decode batch width remain unchanged receipt evidence; the cross-engine
serving matrix owns the competitive performance verdict. Raising containment
therefore cannot turn a slow run into a fast one or suppress a timeout.
The final two waves cover the configured eight decode slots and the four
derived short-prefill staging slots. Batching counters must prove at least one
multi-row decode, more decode rows than forwards, and a prefill forward for
every request. The receipt publishes per-wave p99 TTFT, ITL, and end-to-end
latency, duration and token throughput, plus aggregate batching width, row,
prefill, synchronization, memory, and capacity evidence. These values are a
Vulkan scaling baseline, not a claim of parity with vLLM.

`/health.execution_identity` must agree exactly with the complete trusted-debug
execution provenance. The backend must be `vulkan`, the device must be
`vulkan:0`, the compiled kernel feature set must include `vulkan` and exclude
`rocm`, the source must be clean, and the executable digest must equal the
binary built by the driver. Compact result details bind that executable, the
generated TOML, the effective server/environment configuration identities, the
kernel contract, the execution-provenance envelope, and one ordered canonical
hash of all 25 streamed semantic outputs. Dynamic response IDs and timestamps
are excluded from the semantic hash; request names, usage counts, and semantic
deltas are included. The public response headers use the paired value `base`
for both loaded-adapter fields when no adapter is resident. Qualification
normalizes only that exact pair to no identity; a missing header, a mixed
base/named pair, a malformed revision, or any named identity in this baseline
fails closed.

A failed result preserves every boundary it actually crossed instead of
replacing the run with a generic zero record. Its compact `details.milestones`
array advances through `build`, `config`, `startup`, `warmup`, each completed
`wave:<name>`, `measurement`, and `teardown`. The build and generated-config
digests survive startup failure; a returned startup identity survives a later
warmup or request failure; and every completed wave immediately contributes its
real request, token, latency, termination, pause, and ordered semantic-output
hash evidence. Typed device, graph, resize, and reclaim events are collected
from server start through shutdown, while memory samples span measured load.
The final pass/fail gate therefore sees post-measurement teardown faults as
well as events observed during load.

Compact details remain valid JSON even when a failure list exceeds the receipt
limit. Long strings are bounded inside the JSON value, while `_truncation`
records the SHA-256 and original character count plus the number of omitted
top-level fields. Runner-level failures such as a nonzero process exit are
added under `runner_failures`; the runner never appends text after a JSON
object. The ignored full stdout/stderr artifacts remain hash-bound to the
receipt for causal inspection.

Metrics for a boundary absent from `milestones` mean "not observed," not a
successful zero-count measurement. For example, zero requests with no
`wave:*` milestone says that the workload never issued a complete measured
wave; zero shutdown failures with no `teardown` milestone does not attest clean
shutdown. Keep and validate these failed receipts as causal counterexamples,
but do not compare their partial performance metrics with passing baselines.

An HTTP-200 SSE response can still terminate with the closed structured error
envelope `{message, type: "server_error", code: "generation_error"}` followed
by `[DONE]`. Every serving driver treats that envelope as the primary request
failure, validates its exact shape, and retains the bounded server message in
the case result. It must never be reclassified as a successful empty stream or
reduced to a secondary missing-finish-reason error.

The gate fails on any request error, non-length or short output, missing actor
timing, unexpected named adapter identity, device fault, resize/reclaim/graph
event, batching error, failed or at-least-100-ms external-yield synchronization,
changed KV capacity, missing memory sample, unexplained adaptive ITL outlier, or ITL gap
above two seconds. Gaps above 250 ms are always counted as stall evidence even
when they remain below the hard pause gate. The server must drain, exit zero
without force inside the shared 60-second grace period, and leave no private
snapshot payload.

#### Current Strix Halo result

The source-bound 2026-07-14 run on RADV Strix Halo passed from clean commit
`e2efd5dff` and tree hash
`sha256:c5e7d485a9afb8319435e87c4eb91808652a2cc3d0a88fbac602b794a10cad66`.
Its retained receipt is
`qualification/receipts/vulkan/strix-halo/20260714t232612510178z-vulkan-strix-halo-serving-vulkan-baseline--8da450e169-v1.json`.
The run completed 25/25 measured requests and 800/800 completion tokens with
exact length termination, zero request/batching/device/policy/sampler errors,
zero graph/resize/reclaim activity, stable 5,365-block KV capacity, zero
unexplained or two-second ITL pauses, clean zero-exit shutdown, and no snapshot
residue. Four adaptive ITL outliers were causally attributed to concurrent
prefill; 395 client-visible gaps exceeded the 250 ms evidence threshold but
none crossed the two-second failure gate.

This accepted correctness result is also negative performance evidence. The
single, four-way, eight-way, and twelve-way waves took 6.473, 93.384, 324.697,
and 464.650 seconds. Overall output throughput was 0.900 tokens/second; the
eight- and twelve-way waves achieved 0.788 and 0.826 tokens/second. Although
eight active requests were observed, only two decode rows were ever combined,
with 2 batched forwards out of 773 decode forwards. Peak sampled unified-device
usage was 54,288,658,432 bytes. This receipt accepts bounded mixed/long-prompt
correctness and the absence of unexplained pauses or runtime memory-policy
mutation. It does not establish competitive Vulkan batching, and it must not
be presented as a vLLM parity result.

A passing receipt closes only the Phase 6 short/mixed/long serving-baseline
item and the no-silent-skip/no-unexplained-outlier condition for this bounded
run. It does not close the CPU/HF oracle comparison, 30-minute development
soak, eight-hour final soak, cross-engine benchmark, or final common-source-tree
release gate. Keep those items open until their separate receipts exist.

A positive `memory.kv_force_blocks` value is intentionally narrower than the
normal control loop. It is accepted only with `memory.kv_autoscale = true` and
`server.serving_profile = "maintenance"`, where inference admission is already
disabled. The one-shot resize still reserves the complete replacement pool,
drains the actor, invalidates graphs, publishes capacity transactionally, and
emits a `gpu_memory_operation` record with reason `forced_configuration`.
`/health`, `/v1/config`, and the trusted debug state expose the requested value,
effective autoscaler state, bounded reason, and `config_file` source.

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
thirteen reason counts (`cold_cache_host_round_trip`,
`persistent_host_round_trip`, `shape_dependent_attention`, `graph_cache_capacity`,
`graph_cache_byte_budget`, `graph_accounting_incomplete`,
`moderate_memory_pressure`, `tight_memory_pressure`, `critical_memory_pressure`,
`memory_reservation_denied`, `memory_governor_selector_mismatch`,
`capture_failure`, and `replay_failure`) plus slow, total-duration, and
maximum-duration counters. The first occurrence of each
reason, every fallback lasting at least 100 ms, and every failed eager fallback
also emits `event=rocm_graph_fallback` with attempt, eager, and total duration.
Qualification validates the health invariants and attributes these events to
the exact ITL window; unknown reason strings do not receive graph attribution.
The mixed-load receipt also records call, slow-call, cumulative duration, and
maximum duration for pre-candidate headroom, candidate warm, pre-native
reservation, native capture, and rejected-candidate cleanup, plus peak exact
transient-candidate bytes. These values remain distinct from retained graph
bytes and make one-time or repeated graph setup pauses comparable across runs.
Native capture timing includes the settled first launch, defensive cache
admission/publication, and blocking committed governor debit; rejected cleanup
starts only when an unretained candidate enters destruction and settlement.
The driver treats the full cache snapshot and phase telemetry independent of
the model and graph-runner locks as separate authorities. Config and trusted
debug must report
`rocm_graphs`/`rocm_graphs_unavailable_reason` independently from
`rocm_graph_telemetry`/`rocm_graph_telemetry_unavailable_reason`; health
flattens them under `decode_runtime.rocm_graphs` with separate `state`,
`unavailable_reason`, `phase_telemetry_available`, and
`phase_telemetry_unavailable_reason` fields. Missing data must use exactly one
of `backend_without_graph_runner`, `model_runner_busy`,
`model_runner_lock_poisoned`, `graph_runner_busy`, or
`graph_runner_lock_poisoned`, never fabricated zeroes. Prometheus exposes the
same distinction with `kiln_rocm_graph_telemetry_available`,
`kiln_rocm_graph_snapshot_unavailable{reason}`,
`kiln_rocm_graph_phase_telemetry_available`, and
`kiln_rocm_graph_phase_telemetry_unavailable{reason}`. The phase handle lives
outside both runner locks, so it remains available for every real backend while
model-runner or graph-runner contention/poison blocks the full snapshot;
currently only a backend without a graph runner makes the phase channel null.
The same health object exposes retained graph and reusable-slot gauges plus
lifetime slot-create/reuse counters. A logical decode row borrows one slot;
request drain removes its continuity timeline and returns the slot to an idle
pool without destroying native graphs or their graph-stable recurrent buffers.
Adapter invalidation destroys native graphs before their buffers. Graceful
server shutdown closes and joins the decode worker before accelerator teardown.

Run the graph-memory and concurrency gate after the exact source has compiled
successfully on the Strix Halo host:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant headroom-vs-tight-budget \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-rocm-graph-resilience-v1.json
```

This is one source-bound server binary and two sequential server processes. The
headroom arm permits 64 entries and 1 GiB of retained graph allocations; the
tight arm keeps the same entry limit but uses the minimum supported 64 MiB byte
budget. Each arm warms a real graph outcome, then runs mixed prompt buckets at
client concurrency 1, 8, 16, 32, and 64, for 121 measured requests per arm.
Every request is fixed to eight output tokens and the driver compares canonical
streamed semantic deltas exactly across arms, excluding only dynamic response
envelope fields such as request ID and creation time.

The gate fails on an HTTP/request error, output mismatch, non-length finish,
missing token timing, device fault, graph capture/replay failure, retained bytes
above the configured ceiling, entries above capacity, active owner/slot residue,
dirty model snapshot, forced/nonzero shutdown, or any ITL gap above
`max(250 ms, 5 * rolling p50 ITL)`. Both attributed gaps, including graph setup
and synchronization, and unexplained gaps are failures; attribution is retained
to diagnose the source, not to waive a pause. The headroom arm must capture and
replay. The tight arm must make at least one typed byte-budget decision through
pre-capture skip, post-capture rejection, deterministic budget eviction, or the
closed `graph_cache_byte_budget` eager fallback. The receipt publishes p99 TTFT,
ITL, and E2E latency at every concurrency plus graph and peak-memory counters.
A passing receipt therefore has zero attributed or unexplained ITL outliers.

The 300-second client deadline and 360-second server deadline are containment
ceilings for this correctness matrix, not performance targets. The fixed
cross-engine serving campaign owns comparative throughput and latency verdicts;
increasing these ceilings cannot erase the measured TTFT or E2E values. A failed
matrix receipt retains attempted, completed, and failed request counts for every
headroom wave, the highest fully completed concurrency, per-arm
start/attempt/completion coverage, the latest graph counters, peak memory, and
observed pause counts. Unstarted arms therefore remain explicitly distinguishable
from started-but-incomplete arms and measured zero events.

Run the destructive-identity and fallback-containment corpus separately. It
uses the bounded Cargo wrapper, one compile job, the unchanged 15 GiB host-memory
floor, offline dependencies, one test thread, and a required real ROCm device:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant real-rocm-graph-fault-corpus \
  --host-id strix-halo \
  qualification/workloads/serving-rocm-graph-failure-containment-v1.json
```

All three ignored hardware tests must actually execute. The first proves that a
shape-dependent attention geometry is cached as a typed eager fallback rather
than captured. The second crosses sequence buckets and block tables, reuses a
released slot after cancellation/prefix activity, invalidates at an adapter
boundary, and requires exact eager parity. The third poisons only a live graph's
retained pool generation while leaving the physical allocation valid; the
guard must prevent native launch, count one replay failure, clear and disable
graphs, preserve the cache identity, and return exact same-cache eager output.
Missing devices and skipped tests are failures, not successful no-ops.

A native capture failure is eligible for the `capture_failure` eager fallback
only when `capture_rollback` first settles physical work with execution
admission open and logical recurrent-state rollback also succeeds. Failure of
either proof, or any armed/unclassified capture guard leaving scope, must
publish sticky STOP, use `error_recovery` only for a post-STOP diagnostic drain,
reject the eager continuation, and require process restart. Qualification must
never classify that quarantine as an expected fallback.

### ROCm Public Mutation Lifecycle

Run the public adapter and maintenance-mutation gate with a real adapter whose
base model matches the selected Qwen3.5 model:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant public-mutation-lifecycle \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  --var adapter_path=/absolute/path/to/Qwen3.5-4B/adapters/rocm-sft-test \
  qualification/workloads/serving-rocm-public-mutation-lifecycle-v1.json
```

The adapter input must be an absolute directory containing regular,
non-symlink `adapter_config.json` and `adapter_model.safetensors` files. The
driver copies only those two files into a private per-run adapter registry,
re-hashes the copy, and records both source SHA-256 values in the command result
details. It also records the one exact source-built `kiln` binary hash and the
two generated typed-config hashes. The qualification runner separately binds
the clean source tree, model weights, tokenizer, template, manifest, variable,
stdout, and command-result artifacts in the receipt.

The first arm starts that binary under the experimental profile with KV
autoscaling and reclaim off and lazy ROCm graph execution on. It requires real
capture and replay before measuring the lifecycle. A deterministic base request
must return `x-kiln-loaded-adapter: base` and revision `base`. Public
`POST /v1/adapters/load` must return one 64-hex content revision; health,
trusted debug state, and `GET /v1/adapters` must all publish that same
name/revision. The adapter inference request explicitly sends
`"adapter": "qualification-adapter"`; its response headers must bind that exact
name/revision. Base probes explicitly send `"adapter": null`. The driver then
calls the public unload
endpoint, requires every surface and response header to return to base, and
compares canonical streamed semantic deltas from identical pre-load and
post-unload base requests exactly. Dynamic IDs and creation timestamps are the
only excluded envelope fields. Both barrier-swap reasons must appear in order,
at least one captured graph must be invalidated, and the actor, graph slots,
owners, process, and snapshot directory must drain cleanly.

The second arm reuses the same binary under a separate maintenance-profile TOML
file with graphs disabled, `memory.kv_autoscale = true`, and
`memory.kv_force_blocks = 1`. It requires exactly one structured
`gpu_memory_operation` resize with reason `forced_configuration`, an initial
capacity above the target, exact `requested_blocks` and `actual_blocks` of one,
and finite nonnegative barrier/GPU/model wait and total mutation durations.
Because this profile intentionally disables inference, `/health` must return a
structured HTTP 503 with status `maintenance`, a failed
`inference_admission` check, and every other readiness check passing. That is
maintenance readiness, not a transport or startup failure.
`/v1/config` must observe the one-block physical capacity while health and debug
report the requested target and `config_file` provenance. A public chat request
must return HTTP 503 with `inference_disabled_by_profile` without changing
batching admission, prefill, decode, or used-block counters.

Either arm fails on an HTTP or stream error, malformed identity, stale adapter
header, changed base output, missing graph invalidation, wrong resize target,
inference work in maintenance, device-fault signature, forced/nonzero shutdown,
or private snapshot residue. The receipt publishes transition and rejection
counts, adapter bytes, load/unload latency, graph invalidations, resize source
and target blocks, released bytes, coordination wait, mutation duration, and
all failure counters. This gate proves one controlled lifecycle at one source
revision; it does not replace concurrent mixed-load stress, the graduated
concurrency gate, or the long soak.

Command-result evidence is accumulated at each completed boundary rather than
synthesized only after both arms pass. A failed result therefore retains any
completed binary build, copied adapter identity, public load or unload,
semantic-output hash, graph invalidation, physical resize, rejection, and
transition already observed. `arms_started` and `arms_completed` distinguish an
unstarted arm from a partial one. Request failures count actual HTTP/stream
failures only; dirty shutdown and snapshot residue derive from teardown rather
than being invented for every failed case.

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
phase is explained unless it overlaps the required named-host thermal pacing
controller. Thermal-attributed counts must reconcile exactly; non-thermal
attributed and unexplained counts remain disqualifying.
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
The 256-token peer is twice the ordinary response length. The driver dispatches
it before the slow consumer and waits for its first
producer-ready token before opening the stalled socket. The acceptance window
then requires further producer-ready tokens during and after the slow
consumer's request-attributed backpressure interval. This ordering proves
continuity of an already-decoding peer instead of falsely requiring a queued
request to have emitted before a pressure window that already started.
Its actor queue, slot-admission, and admission-to-first-ready wall durations are
recorded separately and must fit inside TTFT; accumulated model prefill must fit
inside admission plus admitted-prefill wall time. Missing, duplicate,
nonnumeric, or internally impossible phase evidence fails the run. These fields
distinguish active-set saturation from slow admitted prefill before any
scheduler policy is changed.

For the historical dynamic-runtime A/B, run each of `default`,
`autoscale-off`, `graphs-off`, and `both-off` separately. These four arms now
pin `server.serving_profile = "experimental"` in the generated typed launch
file so their requested graph/autoscale differences retain the semantics they
had before stable became the default:

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

When deterministic output differs even with graphs and KV autoscaling disabled,
run `both-off-prefix-cache-off` against `both-off`. The diagnostic arm also
writes `prefix_cache.enabled = false`; no environment override is accepted. Its
effective configuration differs only in requested/effective prefix-cache state
and reason. Startup, post-measurement, post-sampled, and final-canary health
attestation require the batching capability to remain false and every cache
capacity, lookup, hit, lease, pending release, retained block, entry, and state
byte counter to remain zero. The receipt retains the enabled bit, measurement
baseline residency, lookup/hit deltas, and final residency. This arm isolates
prefix/recurrent-state reuse; it does not silently redefine the historical
four-arm performance matrix.

### ROCm Synchronization A/B

Run the synchronization policy checkpoint before a longer mixed-load or soak
after changing ROCm stream ownership, eager operators, graph boundaries, tensor
handoffs, allocator lifetime, or host readback. The workload is self-contained:
it builds one exact binary and sequentially starts two isolated servers from
that binary, first with `legacy_host_barriers` and then with `stream_ordered`.
Do not launch the arms by exporting old `KILN_ROCM_*` switches yourself.

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant legacy-vs-stream-ordered \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-rocm-sync-ab-v1.json
```

Both arms use the experimental serving profile because `stream_ordered` is an
explicitly gated experiment. The workload disables ROCm graphs, physical KV
autoscaling, and allocator reclaim in both arms so the first checkpoint changes
only synchronization discipline. It warms one short request, then runs fixed
32-token waves at concurrency 1, 2, and 4 with prompt lengths from 16 to 384
words. The per-request deadline is 90 seconds and build/startup/work across
both arms shares a 10-minute deadline. Each server then gets the standard
separate 60-second graceful teardown bound before a forced kill is reported as
failure. Gaps at least 250 ms are retained as stall evidence, and any
inter-token gap at least two seconds fails. This is the small admission
checkpoint; it does not replace the full mixed-load, memory pressure,
30-minute development soak, or final 24-hour soak.

Before timing begins, each arm also runs the same fixed-seed, non-streaming
provenance request through the public API. The existing full-model correctness
parser validates its action-token coverage and finite selected-token
log-probabilities. The two normalized semantic traces, action token IDs, and
behavior log-probabilities must match exactly. This probe is excluded from the
timing and synchronization deltas, so semantic confidence does not contaminate
the measured A/B window.

At startup, after warmup, and after the measured wave, the driver requires the
resolved policy from both `/v1/config.accelerator_runtime` and
`/health.decode_runtime.accelerator_runtime` to match the arm exactly. It also
requires `/health.decode_runtime.rocm_synchronization` to expose all 23 fixed
reason dimensions, internally consistent aggregate counts, no telemetry error,
`cleanup_quarantined=false`, and monotonically increasing counters. Every health
reason/count/duration is
reconciled with these Prometheus families:

```text
kiln_rocm_synchronization_policy_info{mode}
kiln_rocm_cleanup_quarantined
kiln_rocm_synchronizations_total{reason,scope}
kiln_rocm_synchronization_wait_seconds_total{reason}
kiln_rocm_synchronization_skipped_total{reason}
```

The Prometheus quarantine gauge must remain `0` while telemetry availability is
`1`. A true health field or nonzero gauge is a hard arm failure: do not classify
it as an expected graph fallback or continue collecting throughput samples.

The legacy arm must execute an `external_yield` device wait and skip no
barrier. The stream-ordered arm must execute an `external_yield` stream wait
and skip at least one proven same-stream barrier. Both must produce the same
request names, prompt-token counts, exact 32-token length termination, no
request or device fault, clean unforced shutdown, and no private snapshot
residue. The receipt records TTFT, E2E, ITL p50/p99/max, throughput, peak GPU
memory, stall and pause counts, aggregate wait scope/count/time, and skipped
barriers for each arm. After each concurrency wave, its bounded raw output
records maximum ITL, stall/pause counts, and every per-reason counter/time
delta for that interval. This narrows pause attribution to a specific wave and
reason family instead of assigning a temporal gap speculatively to VRAM
rebalancing.

Passing this checkpoint means stream ordering preserved the bounded workload
and the diagnostics are trustworthy. It does not mean stream ordering is
faster: review both arms' latency, throughput, memory, and per-reason traces,
then proceed to the existing mixed-load workload with the promising policy.
New qualification drivers use the mechanically derived
`KILN_ACCELERATOR_ROCM_SYNCHRONIZATION_MODE`,
`KILN_ACCELERATOR_ROCM_GRAPH_MODE`, and
`KILN_ACCELERATOR_ROCM_GRAPH_CACHE_ENTRIES`, and
`KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES` names. The shorter graph variables
remain deployment compatibility aliases and are not new evidence vocabulary.

### ROCm Development Soak

After a material ROCm serving change, run the committed 30-minute development
soak from a clean, pushed source tree:

```bash
PATH="$HOME/.cargo/bin:$PATH" ROCM_PATH=/opt/rocm \
python3 scripts/qualification/run.py \
  --variant autoscale-off \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-rocm-development-soak-v1.json
```

The driver builds once and starts one server process with 12 active-request
slots. Its graph-required operating point reserves one protected geometry for
each declared active owner and pre-reserves zero transition entries, so the
checked 12-entry ceiling is `12 * 1 + 0` and matches the product default. Runtime
admission consumes unused global headroom freely. Only at entry or byte
saturation does it reclaim idle owners, followed by the minimum deterministic
fair-LRU active entries; the incoming candidate counts toward its owner's share
and one graph remains protected for every active owner. This settled relief
creates transition room only when needed instead of retaining twelve additional
native graph objects continuously. Every narrow retirement preserves recurrent-
state slots and continuity. The driver warms ROCm graphs and fills the bounded
prefix cache to its declared entry/state capacity before recording the post-
warmup memory baseline or starting the 30-minute measurement clock.

The smaller operating point is evidence-based. With the same Strix Halo binary,
model, seed, workload, and 120-second minimum measurement, 12 entries completed
the identical first seven-wave sequence in 115.461 seconds versus 144.617
seconds at 24 entries, a 25.3 percent improvement. The 12-entry arm performed
seven settled measured capacity transactions and twelve captures with zero
fallback, graph failure, or live-slot loss. Its active SCLK was lower, not
higher, so the result isolates excessive retained native graph population on
this host rather than a graphics-clock advantage. This does not establish 12 as
an optimal cache size for every device; it binds the qualified Strix Halo
operating point and keeps larger deployments subject to their own receipts.
It then exercises complete fixed-prompt concurrency cycles, including periodic
cancellation, until GPU-used and server-RSS deltas remain within 64 MiB and 16
MiB respectively for two consecutive cycles. This convergence requires at
least four cycles and fails after eight instead of silently moving a growing
allocator into the baseline. The result retains completed cycle, request,
cancellation, final-delta, and maximum-delta metrics even when convergence
fails before the measured phase begins.

The measured phase keeps that process under the same fixed-output waves at
concurrency 1, 8, and 12 with prompt lengths spanning multiple sequence
buckets. Every fifth wave also cancels a longer request using a unique marker.
Slot prompts repeat across waves so prefix hits and cached-block reuse are
measured. After each wave, the driver requires the engine to drain, every used
KV block to be owned by the prefix cache, zero active cache leases or pending
releases, stable cache residency, zero active graph slots or row timelines,
at most 12 retained graphs and slots, the process to remain alive, and
runtime/debug policy attestations to remain consistent. Idle slots and their
native graphs remain resident for reuse. The final drain requires every
retained slot to be idle, and the measured phase must exercise slot reuse.

Every response in graph warmup, prefix-cache warmup, stabilization, and
measurement must also satisfy the declared ascending-sequence oracle. A
cancellation is confirmed only when its first four semantic deltas form the
same valid prefix before the client disconnects and the server proves drain;
disconnect cleanup cannot mask already-corrupt output.

The result fails on any request, deterministic-response-oracle, or cancellation
error, graph capture/replay
failure, typed eager fallback, backend synchronization failure or 100 ms slow
sync, device-fault signature in either a log message or structured error,
non-finite response error, unexplained ITL outlier, capacity change,
unaccounted block, active cache lease, pending release, dirty shutdown,
snapshot residue, or GPU/RSS peak more than 512 MiB above the post-warmup
baseline. Attributed outliers remain counted for review. The receipt also
records p50/p99/p99.9 TTFT and ITL, graph activity, prefix-cache reuse,
graph/slot residency and reuse, external-yield synchronization, memory
baselines/peaks, request/token counts, and cancellation count. Shutdown must
return zero without force after the decode worker is joined, and snapshot
cleanup must leave no residue.

ROCm results always declare the base serving metrics plus the shared host
memory, swap, temperature, thermal-pacing, and accelerator clock/power metrics.
Vulkan declares the same accelerator telemetry schema plus its resident-prefill,
process-DRM, allocator, and mapping extensions. The same backend-specific
schema is used for both complete and partial results, so a later failure cannot
replace retained ROCm warmup evidence with a metric-set mismatch caused by
Vulkan-only fields.

The accelerator sampler resolves exactly one Linux DRM device whose `vendor`
is AMD `0x1002`, then exactly one `amdgpu` hwmon directory below that device.
It selects SCLK by `freq*_label=sclk`, average package power by
`power*_label=PPT`, and edge temperature by `temp*_label=edge`; it never embeds
boot-dependent `cardN` or `hwmonN` numbers. Every 250 ms it reads those labeled
values with `gpu_busy_percent`, while `pp_dpm_sclk` supplies the advertised
maximum SCLK. Missing, ambiguous, malformed, or implausible inputs are explicit
telemetry outcomes rather than zero readings.

Receipt metrics include telemetry availability and errors, total and active
sample counts, p50/peak busy percentage, advertised SCLK, active-sample
minimum/p50/maximum SCLK, active SCLK samples below half the advertised maximum,
active p50/peak PPT power, and active p50 plus overall peak edge temperature.
An active sample is one with at least 50 percent GPU busy, preventing idle
clocks and power from distorting the workload summaries. Aggregates are scoped
to the measured phase; partial results retain samples collected after
measurement began. ROCm requires available, error-free telemetry with at least
one measured and one active sample. Vulkan uses the same sampler when AMD sysfs
is available, but permits an unavailable device so the workload remains
portable to non-AMD Vulkan implementations; once an AMD telemetry source is
selected, any read error still fails the case. The below-half-max count is
diagnostic evidence, not by itself a performance acceptance threshold.

This `kind: soak` workload is intentionally a non-comparative pass/fail gate,
so its `comparison_policy` is null. Do not use it to claim relative throughput
or latency; use the serving benchmark protocol for those claims. The 30-minute
receipt also does not replace the final 24-hour ROCm phase soak.

Build, runtime setup, and measurement use independent absolute deadlines. The
exact source-bound build has its own 900-second limit from the checked build
specification. A successful build starts a fresh 1,200-second runtime setup
envelope for server startup, warmup, and stabilization; compilation time cannot
consume the runtime evidence budget. Only after stabilization passes does a
fresh 1,920-second measurement envelope begin: 1,800 required seconds plus one
120-second request deadline so the last admitted wave can settle. The outer
4,200-second case timeout is exactly those three limits plus a separate
180-second teardown margin. Thermal pacing remains charged to the runtime setup
or measurement phase in which it occurs. A setup failure records
`soak_duration_seconds=0`; it cannot be mistaken for measured performance.

### Vulkan Resident-Prefill Oracle

Run the focused resident-state gate on a clean, pushed source tree before the
Vulkan development soak:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan-resident-prefill-oracle \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-vulkan-resident-prefill-v1.json
```

This is a source-bound `kind: correctness` gate for the experimental serving
profile. It builds the Vulkan-only `kiln` binary through the bounded Cargo
wrapper, verifies the executable and Vulkan execution identity, starts one
server from one typed configuration, and keeps cross-request prefix reuse,
graphs, KV resizing, and allocator reclaim disabled. Stable and maintenance
profiles are outside this workload because their typed policy must keep
resident prefill disabled with zero route activity.

After a singleton oracle-valid warmup, the driver dispatches two four-request
cohorts through thread barriers. Every row uses the same 16-word prompt length,
while completion limits are `8/12/16/20` in the first cohort and
`20/16/12/8` in the second. Equal prompt lengths make rows ready together;
different completion lengths force each cohort to shrink through changing
active-row sets and a singleton tail. Both cohorts remain in the same server
process so the second repetition detects poisoned parked state, stale row
identity, incorrect row strides, and unsafe capacity reuse. Every response
must terminate by length at its exact limit and remain a prefix of the
ascending zero-padded integer oracle. Terminal response metadata must attest
resident-prefill use on at least two requests.

Enablement is not execution evidence. The final health delta must report at
least one resident forward, more resident rows than forwards, at least one
completed row, and a maximum resident batch of at least two and at most four.
Attempts must reconcile exactly with forwards plus initial declines when no
route failure occurred. Initial declines, route failures, active resident rows
at drain, batching errors, device faults, graph activity, external-yield sync
failures or slow calls, response errors, and semantic-oracle failures must all
be zero. The parked batched-state cache may retain reusable capacity, but it
must end with zero active leases, no miss while leased, and no completion,
replacement, or explicit-invalidation eviction. The direct recurrent-state
registry, active KV blocks, unaccounted blocks, prefix-cache state bytes,
prefix leases, and pending prefix releases must all end at zero.

The same fail-closed host controls used by the Vulkan soak cover build, model
load, prewarm, both cohorts, drain, and teardown: at least 8 GiB Linux
`MemAvailable`, no more than 512 MiB new swap, and `k10temp/Tctl` below 97,000
millicelsius. Process-scoped DRM sampling records baseline, end, peak, and peak
growth and rejects more than 1 GiB of active growth. Shutdown must return zero
without force, remove the private model snapshot, and leave no payload residue
or request worker. The receipt retains the binary/config identities, a
canonical semantic-output digest, per-cohort duration and route metadata, all
resident counters, cache ownership, memory, temperature, synchronization, and
lifecycle metrics.

Passing this oracle permits the unchanged 30-minute development soak to test
longer-lived allocator convergence, cancellation, latency, and memory behavior.
It does not qualify stable-profile admission, eight-hour endurance, broad
prompt-length coverage, or throughput competitiveness with vLLM.

### Vulkan Development Soak

Run the Vulkan peer on a clean, pushed source tree after the serving baseline
and before a multi-hour Vulkan gate:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan-development-soak \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-vulkan-development-soak-v1.json
```

This arm uses the same source-built, one-process, post-warmup qualification
model as the ROCm soak, but its hardware accounting is deliberately different.
The checked profile selects a typed 128-token prompt-work ceiling while keeping
the shared four-layer yield ceiling. It also sets `server.max_decode_batch=2`
and `batching.prefill_admission_quantum=2`. The actor therefore admits an equal
pair together while retaining two decode rows plus two staged prefill rows, and
health/debug must attest the derived two-slot staging and four-request active
ceilings. On this Strix Halo, 64-token chunks made
regular progress but could not finish the declared eight-way long-prompt wave
before the unchanged 600-second request deadline. Repeated four-request A/B
runs at 128 tokens passed eight exact semantic oracles with stable process
history; 256 tokens was faster but corrupted every concurrent response while
the exact same prompt remained correct in isolation. The soak therefore binds
128 explicitly and fails if health/debug reports another value or provenance.
The candidate qualified load alternates one- and four-request waves with
16-token completions. Each one/four pair selects one of the fixed
16/32/64/96-word prompt slots; all four rows in the concurrent wave have the
same prompt length and distinct fixed row identities, and successive cycles
rotate through every slot before repeating. This forces an equal-ready resident
cohort and establishes every declared prompt/batch scratch capacity before the
measurement baseline. It is an enforced qualification operating point, not the
product-wide default or a claim that larger Vulkan quanta, more than four
simultaneously active requests, or longer-prompt throughput are competitive or
qualified.

Its timing envelopes are likewise independent and explicit in the effective
configuration. The exact source build gets the checked 900-second build limit;
after it succeeds, startup, warmup, and stabilization get a fresh 1,800
seconds. After those gates pass, the 30-minute measurement gets a fresh
2,400-second absolute deadline: 1,800 required seconds plus the unchanged
600-second per-request containment window. The outer case timeout is 5,280
seconds, exactly those limits plus 180 seconds for cancellation, server join,
private-snapshot cleanup, and result publication. Wave request threads get a
fixed 10-second cleanup window inside that teardown allowance even when the
setup or measurement work deadline has already expired. The driver sets their
shared abort signal, waits for them, records `request_worker_residue_count`, and
rejects any survivor; an
expired work deadline therefore cannot erase its own cleanup budget. Setup time
is never reported as measured soak time.
Raw stdout and stderr artifacts are flushed after every captured chunk, so a
monitor can follow structured progress while the case is still running rather
than waiting for a user-space file buffer to fill.
Steady-state warmup counts are committed after each response cohort has fully
completed and the server has drained, before the drained health snapshot is
judged. A graph, ownership, or cache-policy failure in that snapshot therefore
retains the completed request and wave counts while still failing the case;
requests from an incomplete or undrained cohort are never counted.

The clean 2026-07-16 run at commit `2587de1dd` is the pre-resident-prefill
counterexample, not a passing soak. It completed 30 oracle-valid stabilization
responses and one confirmed cancellation, then exhausted the 1,800-second
setup envelope during the second cycle's final width-four wave. Measurement
never started, so its zero measured request, wave, latency, and duration metrics
are explicit partial-evidence sentinels. The successful width-eight wave reported
256.9-562.5 second TTFT for 472-1,240 prompt tokens. Actor telemetry showed
continuous rowwise generic prefill in four-layer, 128-token slices of roughly
1.4-2.3 seconds, not an unexplained stall or mid-request VRAM rebalance. The
first complete cycle also grew process DRM by 91,250,688 bytes and therefore
failed the unchanged 64 MiB stabilization-delta gate. The retained receipt is
`qualification/receipts/vulkan/strix-halo/20260716t041351870594z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
It records clean bounded teardown, zero device faults, zero direct prompt-state
ownership after drain, at least 20,062,093,312 bytes of available host memory,
53,248 bytes of swap growth, and a 93,375 millicelsius package peak. The next
qualification attempt requires a safe generic-prefill throughput correction
or an explicitly narrower published service region; increasing the setup or
request deadline would not resolve this gate.

The clean `7f8903097` rerun after the focused resident-prefill oracle is the
current counterexample. It advanced farther, completing all 17 responses and
one confirmed cancellation in Cycle 1, then the one-, four-, and eight-way
waves in Cycle 2. The setup deadline expired in Cycle 2's final four-way wave
with three workers still awaiting completion. Thus 30 stabilization responses
were oracle-valid, but only one complete cycle was eligible for memory
attestation and measurement still never began. The retained receipt is
`qualification/receipts/vulkan/strix-halo/20260716t062158784737z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
Cycle 1 grew process DRM by 96,256,000 bytes, live Vulkan ownership by
95,944,768 bytes, and RSS by 319,447,040 bytes, so it correctly remained
nonstable. It recorded 1,249 exact recurrent-cache reuses, 571 resident-capacity
reuses, and 542 resident-prefix views with zero miss while leased, completion
eviction, explicit invalidation, or replacement eviction. The run had zero
batching or device faults and zero unexplained ITL outliers; host availability
stayed above 17,856,610,304 bytes, swap grew 12,288 bytes, and package
temperature peaked at 96,000 millicelsius. Shutdown was unforced and zero with
no snapshot residue or surviving process. Because that receipt predates the
failure-path evidence repair, its zero resident-prefill metrics are partial-
evidence sentinels rather than proof that the route was idle; the raw trace
contains multi-row, full-stack resident forwards. Current workloads separately
declare stabilization resident-prefill enablement, attempts, forwards, rows,
completed rows, maximum width, declines, route failures, and active rows at the
last drained boundary. Those deltas are published on both pass and failure,
while the existing unprefixed fields remain measurement-only. The performance
gate itself remains unchanged: either improve this heterogeneous
region or publish and enforce a narrower supported Vulkan service envelope
before qualifying that envelope.

ROCm's device-global memory counter cannot isolate another desktop process.
The Vulkan driver therefore additionally sums the server process's `drm-memory-vram`,
`drm-memory-gtt`, and `drm-memory-cpu` records from `/proc/<pid>/fdinfo`,
deduplicated by DRM client ID. It samples `VmRSS`, `RssAnon`, `RssFile`,
`RssShmem`, and `VmSwap` from `/proc/<pid>/status` separately.

Both committed Strix Halo development soaks use the same independent 250 ms
host controller. Its memory guard sends `SIGTERM` to the complete server process
group if Linux
`MemAvailable` falls below 8 GiB; missing or malformed host telemetry also
fails closed. It follows termination with `SIGCONT`, which is harmless for a
running group and lets a thermally paced group execute shutdown immediately.
The receipt records the starting, minimum, and ending available
memory, starting/peak/ending swap use, and swap growth. More than 512 MiB of
new swap is a failure even when the 8 GiB floor was not crossed.

The same 250 ms safety loop independently monitors the Strix Halo package
temperature from Linux hwmon. The committed workload selects the sensor by the
stable pair `name=k10temp` and `label=Tctl`, never by a boot-dependent
`hwmonN` path, and sets a 97,000 millicelsius limit. The driver resolves exactly
one matching `temp*_input` after launching the server. A missing, ambiguous,
non-integer, or implausible sensor reading fails closed and sends `SIGTERM` to
the server process group; a valid reading at or above the limit does the same.
The result retains starting, peak, and ending package temperature plus the
thermal-trip count, including startup or pre-measurement failures. The effective
configuration records the sensor name, label, limit, and poll interval, so a
receipt cannot silently inherit a different sensor or threshold. This guard
covers model load, native prewarm, warmup, stabilization, measurement, and final
drain. The preceding source build independently resolves the same stable sensor
selector and enforces the same threshold and cadence around its complete
transient compiler/linker service. A build trip exits with status 3, while a
missing or invalid sensor fails preflight with status 2; neither can be recorded
as a successful source-bound build.

Both the ROCm and Vulkan soaks also use that continuously scheduled controller for
hysteretic pacing. At or above 88,000 millicelsius it sends `SIGSTOP` to the
complete server process group, including an inference wave already in progress.
The Python qualification controller is outside that process group, so sensor,
host-memory, and deadline observation continue while the server is stopped. At
or below 86,000 millicelsius the controller sends `SIGCONT`. This bounds each
stop more tightly while retaining a two-degree hysteresis against rapid
stop/start oscillation and does not depend on reaching a request boundary. The
97,000 millicelsius safety check is evaluated first on every sample and remains
unchanged: a stopped group that reaches the limit receives `SIGTERM` followed by
`SIGCONT`, allowing termination and cleanup to run rather than leaving a stopped
process behind. Sensor ambiguity, read failure, controller-thread failure, and
signal failure all remain fail-closed conditions. Teardown first disables new
pacing under the same transition lock, releases any active stop, and leaves the
hard-limit sampler running while the server exits. This closes the race in which
a polling iteration could issue `SIGSTOP` after teardown had already checked for
an active pause.

Pacing does not suspend either phase deadline or a request deadline. Cooling
therefore consumes the existing setup or measurement envelope and cannot extend
a run, conceal an unsustainable workload, or turn a deadline failure into a
pass. Each pause and release emits a `host_thermal_pacing` observation; measured
token gaps overlapping that interval are attributed to the named host control,
not silently reported as an unexplained inference stall. The receipt publishes
`active_end`, completed and started event counts, total seconds, longest
interval, and hottest starting reading under `host_thermal_pacing_*`. A clean
result requires no active pause at teardown and requires every started event to
have completed. A deliberate teardown release or observed process exit completes
the interval with its reason; a controller close or safety trip remains an
interruption. A measured throughput result
includes the cooling time required to sustain this workload on the named host
rather than reporting only its short-burst rate.

Server exit is not the end of host containment. ROCm mixed-load, ROCm/Vulkan
development and endurance soaks, and the Vulkan resident-prefill oracle wait for
eight consecutive 250 ms `k10temp/Tctl` samples at or below 75,000
millicelsius, with a 180-second bound. The receipt publishes
`host_thermal_cooldown_active_end`, `completed_count`, `timeout_count`, seconds,
sample count, stable-sample count, and post-exit peak. Qualification requires
exactly one completed cooldown and no active or timed-out cooldown. The final
`host_temperature_end_millicelsius` is therefore a post-cooldown observation,
while `host_temperature_peak_millicelsius` includes any residual heat-soak
spike after the process exits.

When measurement starts but a later wave fails, the case result retains request,
latency, cancellation, memory, allocator, cache, and resident-route evidence
through the last fully completed and drained wave. An in-progress wave is never
counted as complete. `measurement_final_snapshot_complete=0` distinguishes that
partial evidence from a normal final drain; only a run that obtains and validates
the final health/debug/memory snapshots publishes `1`. This flag is diagnostic,
not a way for a failed case to satisfy an acceptance threshold.

The clean `1ea855a51` Strix Halo run disproved the former boundary-only policy
and motivated the continuous controller above. Six stabilization cycles completed 30 exact responses plus
three cancellations and converged to two consecutive cycles with zero DRM
growth, live-buffer growth, allocations, frees, pool misses, evictions, or
uncached allocations. Measurement began after 1,085.45 setup seconds. Its first
six waves then completed 15 exact responses plus one cancellation over 458.77
seconds with process DRM fixed at 50,001,174,528 bytes and every post-baseline
allocator counter still zero. The next 96-word singleton began below the pacing
threshold but drove the package to exactly 97,000 millicelsius before returning
to a harness boundary. The independent guard stopped the server; boundary
pacing correctly reported zero events. The retained failed receipt is
`qualification/receipts/vulkan/strix-halo/20260716t092911388875z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
It is strictly source/artifact/commit-valid, records at least 16,861,724,672
bytes available, 103,718,912 bytes of swap growth, clean unforced teardown, no
worker or snapshot residue, and no device or batching fault. Raising the 97 C
limit, deleting the 96-word supported prompt, or rerunning the boundary-only
policy would not close this gate. The continuous process-group controller must
still pass a clean pushed-source run before it supports a Vulkan qualification
claim.

That gate passed on the clean pushed `e79d3686d` source. The retained receipt is
`qualification/receipts/vulkan/strix-halo/20260716t154944163408z-vulkan-strix-halo-serving-vulkan-developme-b5eb848d54-v1.json`.
It is strictly current-source, local-artifact, and known-commit valid. Six setup
cycles completed 30 exact responses and three cancellations, covered every
prompt cohort, and converged after 1,212.42 seconds. Cycles 5 and 6 had zero DRM
growth, live-buffer growth, allocations, frees, recycler misses, evictions, or
uncached allocations; the final setup RSS delta was 401,408 bytes. Stabilization
also proved 128 resident forwards over 256 row-forwards, eight completed rows,
maximum width two, zero active rows at drain, and no decline or route failure.

The fresh measured phase then ran for 1,804.61 seconds and completed 51 exact
responses, 816 completion tokens, five cancellations, and 21 fully drained
waves. Process DRM began and ended at exactly 50,840,551,424 bytes; its measured
peak was only 98,304 bytes higher. Live Vulkan ownership and the 3.5 GiB recycler
retention were byte-identical at baseline and final drain, with zero measured
allocations, frees, cache misses, evictions, or uncached allocations and 686,925
cache hits. RSS grew 11,317,248 bytes. The resident route made 992 forwards over
1,984 row-forwards, completed 20 rows at width two, and ended with zero active
rows, initial declines, or route failures. There were zero request, batching,
device, non-finite-response, worker-residue, synchronization, shutdown, or
snapshot failures, and every KV/cache ownership gauge drained to its declared
state.

The continuous controller started and completed exactly 123 pauses, totaling
72.13 seconds; its longest pause was 1.001 seconds and its hottest starting
sample was 89,625 millicelsius. No pause remained active and the unchanged 97 C
guard did not trip. All 240 ITL outliers had bounded attribution and none was
unexplained. Host availability stayed at or above 17,548,128,256 bytes and swap
grew 341,397,504 bytes, inside their unchanged limits. Shutdown was unforced and
zero, the private snapshot was removed, and no process survived. The case result
hash is `sha256:bfc5defbb8889d2ebe73c0f5890fc6d0a0f378ed65487c0bd60245541f9bddbe`;
the receipt file hash is
`sha256:d7e4459d774e86dc6e560c834c8f4d847a1eb33ff7965dcd6b810d8274ba82ea`.

This passes the 30-minute Vulkan development-soak contract for the exact named
machine and declared four-active profile. It does not establish eight-hour
endurance, stable-profile resident admission, broader prompt/concurrency
coverage, CUDA or Metal parity, or throughput competitiveness with vLLM. In
particular, measured p99 TTFT was 150,147.60 ms; this is acceptable for the
current containment/correctness gate but remains an explicit performance
backlog item rather than a production latency claim.

At the drained warmup baseline and after each of the at most eight Vulkan
stabilization cycles, the driver also reads `/proc/<pid>/smaps`. This is bounded
diagnostic work, not a hot-loop sampler. Every mapping must contain Linux's
`Size`, `Rss`, `Pss`, `Anonymous`, `AnonHugePages`, `Private_Dirty`, and `Swap`
fields or the run fails closed. `AnonHugePages` must not exceed `Anonymous`, and
the remaining page-accounting fields are checked against mapping size and RSS.
Mappings are assigned to a fixed, low-cardinality set:

- `anonymous`: unnamed and `[anon:*]` mappings;
- `heap`: the process `[heap]` mapping;
- `stack`: main and named thread stacks;
- `shared_memory`: `/dev/shm`, `memfd`, and System V shared-memory mappings;
- `device`: other `/dev/*` mappings, including DRM render nodes;
- `file`: ordinary file-backed mappings, including model shards and libraries;
- `kernel`: kernel pseudo-mappings such as `[vdso]` and `[vvar]`.

Each cycle trace records signed RSS deltas for every category, total PSS,
anonymous-page, anonymous-huge-page, private-dirty, and swap deltas, plus the
eight largest positive mapping-level RSS changes. Each retained mapping row
includes its current anonymous-huge-page bytes. A mapping identity includes its
path (or `[anonymous]`) and virtual address range, so a fixed mapping becoming
resident can be distinguished from a newly mapped object. The bounded case-result
metrics retain start/end `smaps` RSS, positive growth by category, and positive
anonymous, anonymous-huge-page, private-dirty, and swap growth across the
complete observed stabilization-through-measurement window. The exact metric
for huge-page growth is
`vulkan_process_smaps_anonymous_huge_pages_growth_bytes`. These diagnostics
attribute a safety failure; they do not weaken, replace, or exempt it from the
RSS gate.

Vulkan stabilization alternates concurrency one and four with 16-token outputs
and a cancellation every fourth wave. Each pair uses one shared prompt-length
slot, rotating through 16/32/64/96 words; the source-bound configuration
enforces decode width two, two staged prefill rows, and four total active
requests. Stabilization must observe a multirow resident prefill before the
baseline, which also forces batch-dependent scratch growth into the setup
phase. The candidate explicitly raises the idle Vulkan scratch pool from the
3.0 GiB product default to 3.5 GiB; health must attest the exact byte cap. The
additional 512 MiB is bounded by the same host-memory, swap, process-DRM, and
thermal guards. Client concurrency above four may wait outside the active set, but this
workload makes no latency or throughput claim for that queue or for prompts
beyond the declared slots. The retained width-eight/384-word counterexamples
remain performance evidence and the separate vLLM comparison campaign remains
open. Cross-request prefix
reuse is correctness-quarantined on this backend, so warmup does not try to
fill the inert cache. Instead, the driver requires
`prefix_cache_enabled=false` and zero lookups, hits, misses, retained blocks,
retained entries, state bytes, leases, and pending releases at initial warmup,
every stabilization and measured drain, and final drain. It then requires two
consecutive cycles with no live `VulkanBuffer` ownership growth, allocation,
free, buffer-pool cache miss, eviction, or uncached allocation, plus at most
64 MiB of process-scoped DRM growth. Net-zero bytes cannot hide allocator
churn. It cannot accept before four
cycles and fails after eight. RSS is a
separate cumulative host-safety signal on unified memory: stabilization fails
if process RSS grows by more than 512 MiB from its baseline, while the host
guard independently enforces the 8 GiB `MemAvailable` floor and 512 MiB swap
growth ceiling. This distinction is intentional because an already-live fixed
Vulkan mapping can become resident without a new buffer or DRM allocation.
The measured phase runs the same fixed-slot prompt set for at least 30 minutes.
At every drained wave boundary, process DRM and RSS may grow by at most 512 MiB
from the post-stabilization baseline. The 250 ms DRM sampler separately permits
at most 1 GiB of active-workload growth, bounding concurrent scratch without
misclassifying buffers that are allocated during a wave and fully freed at
drain as retained growth. The receipt records both peak bytes and peak growth,
plus the exact `VulkanBuffer` live high-water mark. Pool retention and total live
buffer ownership must return to their measurement baselines. The same response,
drain, cache-ownership, pause, device-fault, shutdown, and residue gates apply
as in the ROCm arm. Vulkan graphs must remain disabled: any graph capture,
replay, slot, or fallback activity fails.

The allocation counter must also remain unchanged after stabilization. Prefix
cache activity must remain zero, and even one new `VulkanBuffer` allocation
fails the next measured wave. This catches allocator churn that has balanced
ownership at drain yet still creates driver-side RSS growth or inference
pauses.

#### Vulkan buffer ownership telemetry

A Vulkan build adds `vulkan_buffers` to `GET /health`. The object is omitted on
other builds and contains these process-lifetime counters:

| Field group | Meaning |
|---|---|
| `live_device_local_buffers`, `live_host_visible_buffers` | Number of live `VulkanBuffer` allocations by constructor intent. |
| `live_device_local_bytes`, `live_host_visible_bytes` | Vulkan memory-requirement bytes still bound to those live buffers. |
| `peak_live_bytes` | Process-lifetime high-water mark across both memory kinds. |
| `*_allocations`, `*_allocated_bytes` | Successful allocation count and bytes since process start. |
| `*_frees`, `*_freed_bytes` | Destructions and bytes freed since process start. |

The byte values use Vulkan's allocation requirement, which can exceed the
requested logical buffer size because of alignment. `device_local` and
`host_visible` describe the allocation route; on a unified-memory device they
do not imply physically separate VRAM and system-RAM chips. These counters
cover memory owned through `VulkanBuffer`. Driver-private allocations and
memory not created by that wrapper remain visible only through DRM, RSS, swap,
and host-availability evidence.

The same response adds `vulkan_buffer_pool`, which separates recycler
retention from all other live buffers:

| Field group | Meaning |
|---|---|
| `max_retained_bytes` | Immutable typed cap derived from `memory.vulkan_buffer_pool_gb`. |
| `bucket_count`, `buffer_count`, `retained_bytes` | Current exact cache inventory. |
| `free_*`, `borrowed_*` | Idle versus caller-owned portions; each pair must reconcile to the inventory total. |
| `cache_hits`, `cache_misses` | Process-lifetime recycler lookup outcomes. |
| `device_local_cache_misses`, `host_visible_cache_misses` | Process-lifetime miss attribution by allocation route. Their sum must equal `cache_misses`. |
| `last_cache_miss` | One bounded diagnostic record, or `null` before the first miss. A record contains the miss `sequence`, allocation `route`, requested and bucket-rounded bytes, and the Rust caller file and line. Its sequence equals `cache_misses`. |
| `eviction_count`, `evicted_bytes` | Idle entries released to admit a newer working set or satisfy pressure reclaim. |
| `uncached_allocation_count`, `uncached_allocated_bytes` | Overflow scratch allocations returned without a cache owner and freed on normal drop. |

Retained bytes may never exceed the cap. Eviction is oldest-idle-first and
never removes a borrowed buffer. Pressure reclaim runs only while the batching
actor is idle, after exclusive GPU coordination and a second health/activity
check. `GET /v1/config` exposes the pool limit, inventory, free and borrowed
bytes, both lookup outcomes, both route-specific miss counters, eviction totals,
and uncached overflow totals. It deliberately omits the source-level last-miss
record; use `GET /health` for that live diagnostic.

Device-local lookup is exact-bucket because legacy device-buffer consumers can
still use physical buffer size as a copy extent. Host-visible staging carries
an explicit logical extent, so its lookup first checks the exact bucket, then
selects the smallest sufficient idle larger bucket for the same Vulkan device
and host-memory type, with oldest-use order breaking ties. An undersized or
currently borrowed slot can never satisfy a request. Consequently, a
host-visible `cache_hit` may reuse a larger bucket; that is expected and does
not increase retained ownership or allocation count.

Prometheus exports the same state as:

```text
kiln_vulkan_buffer_live_buffers{memory="device_local|host_visible"}
kiln_vulkan_buffer_live_bytes{memory="device_local|host_visible"}
kiln_vulkan_buffer_peak_live_bytes
kiln_vulkan_buffer_allocations_total{memory="device_local|host_visible"}
kiln_vulkan_buffer_allocated_bytes_total{memory="device_local|host_visible"}
kiln_vulkan_buffer_frees_total{memory="device_local|host_visible"}
kiln_vulkan_buffer_freed_bytes_total{memory="device_local|host_visible"}
kiln_vulkan_buffer_pool_limit_bytes
kiln_vulkan_buffer_pool_bytes{state="retained|free|borrowed"}
kiln_vulkan_buffer_pool_buffers{state="retained|free|borrowed"}
kiln_vulkan_buffer_pool_buckets
kiln_vulkan_buffer_pool_requests_total{result="hit|miss"}
kiln_vulkan_buffer_pool_misses_total{route="device_local|host_visible"}
kiln_vulkan_buffer_pool_evictions_total
kiln_vulkan_buffer_pool_evicted_bytes_total
kiln_vulkan_buffer_pool_uncached_allocations_total
kiln_vulkan_buffer_pool_uncached_allocated_bytes_total
```

#### Resumable GDN prefill residency telemetry

Vulkan prompt-prefill residency is currently correctness-quarantined. Typed
runtime policy prevents production requests from entering its request/layer
scope, so ordinary prompt chunks materialize recurrent state before yielding.
The direct-registry telemetry remains a closed contract for detecting accidental
activation, retaining failed-experiment evidence, and qualifying a future
repair. It is distinct from the persistent batched decode-state cache described
below. Trusted `GET /v1/debug/model-state` exposes the complete current direct
registry at `caches.resident_recurrent_state`:

The clean `7dcad0d95` no-override qualification probe is the current quarantine
acceptance anchor. With the checked profile's historical decode opt-in intact,
the 138-token q128 request emitted the exact required 32-token ascending prefix,
reported no semantic failure, and left no direct prompt-registry ownership or
process residue. Its 31.34-second prefill does not satisfy a throughput or soak
gate; those remain separate.

| Field | Meaning |
|---|---|
| `entry_count` | Current logical recurrent-state slots mapped to backend-private buffers. Resumable prefill owns a slot by request ID plus linear-layer index; the aggregate deliberately does not expose either value. |
| `buffer_bytes` | Sum of addressable Vulkan buffer extents for those entries. |
| `allocation_bytes` | Sum of Vulkan allocation requirements, including alignment beyond the addressable extent. This must be at least `buffer_bytes`. |

The object is deliberately aggregate and bounded: it contains no request ID,
prompt data, tensor ID, or per-layer label. Internally, a secondary tensor alias
supports decode and handle-local lookup, but it is not the resumable-prefill
owner. All three values must remain zero throughout current production prompt
execution and after successful decode handoff, cancellation, generation error,
actor discard, and final server drain. A future explicitly test-enabled arm may
observe nonzero values only while a prefill quantum is live. Its cleanup must be
scoped to the stable request owner; clearing the whole registry would be an
invalid implementation because it could corrupt concurrent rows.

No environment variable can enable the quarantined prompt or decode scope;
`kiln.vulkan-kernel-policy.v3` fixes both off. A fully materialized control
requires a reviewed source-policy change and separately attested binary. A
control that merely reports `resident_prefill_used=false` is
insufficient: that field describes the native multi-row prefill route, not this
direct GDN registry. A valid control additionally observes zero direct-registry
ownership and records no direct resident GDN use. Each source-policy A/B arm
requires a fresh server process and its own source/binary identity.

Any future device-resident repair must preserve the ordinary external-dtype boundary. For a
BF16 recurrent tensor, every completed nonfinal token chunk performs a
device-side F32 -> BF16 -> F32 round-to-nearest-even conversion before the row
can resume; an F32 tensor is unchanged. Unsupported external dtypes materialize
at the boundary. Layer-group yields within a token chunk must not add rounding,
and final materialization supplies the final chunk's handoff conversion. This
rule is independent of registry counts: all gauges can drain correctly while a
missing precision boundary still corrupts subsequent decode.

An empty registry with nonzero bytes, a nonempty registry with zero buffer
bytes, or allocation bytes below buffer bytes is internally inconsistent and
fails qualification.

Prometheus exports the same state:

```text
kiln_gdn_recurrent_state_resident_entries
kiln_gdn_recurrent_state_resident_bytes{kind="buffer|allocation"}
```

The Vulkan development-soak driver treats the debug object as a closed
contract. It validates exact field names and nonnegative integer types at
startup, after steady warmup, at the stabilization baseline, after every
stabilization wave and cancellation, at the measurement baseline, after every
measured wave and cancellation, and at final drain. Any nonzero drained value
fails immediately with the phase and all three ownership values. The checked-in
workload retains final values as
`resident_recurrent_state_entries_end`,
`resident_recurrent_state_buffer_bytes_end`, and
`resident_recurrent_state_allocation_bytes_end`; each has a required-zero
acceptance threshold.

The quarantined implementation's real-device parity regressions cover two prompt chunks,
multiple unrelated work-handle identities, reuse from an intentionally stale
zero-valued host handle on a different thread, stable-key materialization into
the caller's chosen handle, and a zero registry afterward. The BF16 arm compares
both outputs and final state with a host-materialized oracle that explicitly
quantizes between chunks; a separate F32 arm compares split execution with one
monolithic chunk. A second two-owner regression proves that evicting one
interrupted row preserves the other row's state. The serving semantic oracle
and cancellation workload remain required because kernel parity alone cannot
prove actor teardown or decode handoff. In particular, a cancellation probe
must first observe a nonzero registry during prefill, abort before a semantic
token is delivered, wait for the actor to drain, and then require both trusted
debug and Prometheus ownership to return to zero before issuing a follow-up
request in the same process. These focused tests pass bit-for-bit today, but
the clean full-model q128 prompt-resident arm failed the semantic oracle while
the fully materialized control passed. They therefore cannot authorize route
activation without the full-model gate.

#### Batched recurrent-state cache telemetry

Native batched decode also owns a persistent `LinearAttentionState` cache for
the recurrent GDN layers. Trusted `GET /v1/debug/model-state` exposes its full
snapshot at `caches.batched_recurrent_state` on every backend. The four current
ownership fields are `entry_present`, `capacity_rows`, `logical_rows`, and
`resident`. A parked resident entry may have more capacity rows than logical
rows because smaller batches use an identity-preserving prefix view of the
maximum observed allocation. An entry is temporarily absent while a forward
call owns its lease.

The remaining fields are process-lifetime monotonic counters:

| Field group | Meaning |
|---|---|
| `active_leases`, `max_active_leases` | Current and peak simultaneous checked-out or newly assembled states. More than one proves overlapping batched-state forwards. |
| `take_hit_count`, `take_miss_count` | Cache checkout outcomes. The first eligible forward normally misses. |
| `take_miss_while_leased_count` | Misses observed while another state lease is active. This is the direct signal for concurrent checkout of a single-slot cache. |
| `exact_reuse_count` | Reuse with the same ordered request-ID fingerprint; no state-row refresh is needed. |
| `resident_capacity_reuse_count` | Reuse of backend-resident allocation capacity. This includes same-width and smaller-prefix reuse. |
| `resident_prefix_view_count` | Capacity reuse through a smaller logical axis-0 prefix. |
| `resident_refresh_count` | In-place row refresh because the ordered request IDs changed. |
| `fresh_assembly_count` | New batched state assembled after a miss or rejected entry. |
| `rejected_*_count` | A checked-out entry could not be reused because row IDs were absent, input rows were nonresident, the cache was nonresident, or capacity was insufficient. Exactly one rejection reason is recorded for each rejected entry. |
| `park_count` | A forward returned its state to the persistent cache. |
| `park_replacement_eviction_count` | A returning lease found another parked entry and evicted it. This should remain zero when one cache owner cannot overlap another. |
| `explicit_invalidation_count`, `explicit_invalidation_eviction_count` | Adapter/model lifecycle invalidations requested, and those that actually removed a parked entry. |
| `completed_row_preservation_count`, `completed_row_eviction_count` | Completed request rows that appeared in the cache fingerprint and caused resident preservation or nonresident eviction. |
| `lease_drop_eviction_count` | Checked-out capacity released instead of parked, including rejected entries and forward/error exits. Use the rejection counters and request failures to distinguish expected capacity replacement from errors. |
| `resident_prefix_snapshot_suppression_count` | Whole-prompt, strict-prefix split, or rolling prefix snapshots rejected because backend-resident recurrent/conv state, rather than the logical tensors, was authoritative. This is a correctness guard, not a cache or request failure. |

Prometheus exports the same bounded-cardinality state as:

```text
kiln_batched_recurrent_state_cache_entry
kiln_batched_recurrent_state_cache_rows{kind="capacity|logical"}
kiln_batched_recurrent_state_cache_resident
kiln_batched_recurrent_state_cache_leases{kind="active|max"}
kiln_batched_recurrent_state_cache_takes_total{result="hit|miss"}
kiln_batched_recurrent_state_cache_misses_while_leased_total
kiln_batched_recurrent_state_cache_reuses_total{kind="exact|resident_capacity|prefix_view|refresh"}
kiln_batched_recurrent_state_cache_assemblies_total
kiln_batched_recurrent_state_cache_rejections_total{reason="missing_row_ids|nonresident_rows|nonresident_cache|insufficient_capacity"}
kiln_batched_recurrent_state_cache_parks_total
kiln_batched_recurrent_state_cache_invalidations_total
kiln_batched_recurrent_state_cache_completed_rows_total{action="preserve|evict"}
kiln_batched_recurrent_state_cache_evictions_total{reason="park_replacement|explicit_invalidation|completed_row|lease_drop"}
kiln_prefix_cache_snapshot_suppressions_total
```

The reuse counters deliberately describe overlapping properties: an exact
fingerprint can also reuse resident capacity and a prefix view. Rejection
reasons, by contrast, are mutually exclusive. In a warmed, serialized,
fixed-maximum-capacity workload, fresh assemblies and insufficient-capacity
rejections stop increasing after the largest batch is seen;
`take_miss_while_leased_count`, `park_replacement_eviction_count`, and
`max_active_leases - 1` remain zero. Growth in those three concurrency signals
alongside flat semantic/device error counters identifies ownership overlap,
not allocator pressure. Increasing `rejected_insufficient_capacity_count`
without overlap identifies legitimate high-water growth. Increasing
`lease_drop_eviction_count` without a rejection requires an accompanying
forward/error investigation.

Vulkan full-attention KV seed flags and recurrent GDN state have different
lifetimes. The former are indexed by layer and reset when an unidentified
single-request decode crosses a start-position session boundary, so a new
request cannot read the prior request's KV contents. GDN buffers are indexed
by tensor ID and remain owned by that request row or the batched-state cache;
a session boundary must not clear their initialization markers. Only explicit
eviction of the same tensor ID may remove its recurrent buffer, convolution
buffer, and initialization marker. A rising
`rejected_nonresident_cache_count` with live buffers still present indicates a
violation of this lifetime split, not buffer-pool eviction.

Prefix-cache snapshots have a related authority boundary. Generic execution
does not by itself prove that logical GDN tensors remain authoritative: an
accelerated prefill kernel may advance backend-private recurrent and
convolution buffers without updating those logical tensors. Native resident
decode can additionally advance backend-private KV without writing generic
decoded-token KV positions. Copying the logical tensors in either state would
publish an internally inconsistent cache entry. Kiln therefore puts every
whole-prompt, strict-prefix split, and rolling snapshot through the same
authority gate. It captures only while no layer reports backend-resident GDN
authority; otherwise it omits the cache registration. This may deliberately
forgo a prefix-cache hit, but never changes the live request state and never
introduces a hidden device readback on the prefill or decode hot path.

`resident_prefix_snapshot_suppression_count` and
`kiln_prefix_cache_snapshot_suppressions_total` prove that guard fired. The
counter is expected to rise only when an admitted cache encounters an otherwise
eligible capture under native resident decode authority. Vulkan still
quarantines the cross-request prefix cache. Stable and maintenance profiles
also quarantine resident token-prefill and require its advertised capability
and every activity counter to remain zero. The experimental profile admits
resident token-prefill after backend and request eligibility checks; its soak
must instead prove at least one multi-row forward, positive row and completion
counts, zero active rows at drain, and zero route failures or initial declines.
Captures remain subject to the same authority gate. The suppression counter
must be monotonic; a zero delta is meaningful only when the workload also proves
that an eligible cache capture and resident execution occurred. Qualification
retains its delta but does not treat a positive value as an error on an admitted
backend.

The Vulkan development soak treats this snapshot as a closed qualification
contract. It validates the exact field set and types at startup, after warmup,
after every drained stabilization cycle, after every measured wave, and at
final drain. Per-cycle traces include the current ownership gauges and the
delta of every lifecycle counter. The case result retains the complete
observed stabilization-through-measurement delta, including runs that fail
before measurement. A drained wave fails if it leaves an active lease; after
stabilization, any new miss while leased or park-replacement eviction fails
immediately. The final verdict also requires one resident parked entry, no
active lease, a peak of at most one active lease, and zero completed-row,
explicit-invalidation, replacement, or lease-overlap eviction in the observed
window. These gates distinguish cache ownership overlap from buffer-pool
pressure without inferring either from RSS alone.

The soak validates that allocation/free deltas reconcile exactly with the
change in live count and bytes, that pool free plus borrowed ownership equals
retention, that the cap never changes or overflows, and that every cache
counter remains monotonic. Initial stabilization cycles may fill the
fixed-shape recycler working set. Two later cycles must show no positive live
ownership growth before measurement begins; any measurement-phase live or pool
retention growth fails the run. Allocation and free deltas that match while
live bytes stay flat indicate temporary buffer churn; if RSS continues to grow
within the cumulative host limit, investigate driver or unified-memory page
residency rather than mislabeling it as retained Rust ownership. Flat
allocation/free counters and flat live bytes with growing RSS instead point to
pages becoming resident in already-live allocations. The stabilization and
measured-wave traces record total and route-specific pool miss deltas. They
include the single `vulkan_pool_last_cache_miss` record only for an interval
that made at least one miss, so an unexpected warmed-path allocation identifies
its final request size, pool bucket, route, and source callsite without
unbounded event capture. The stabilization trace also records buffer, pool,
DRM, RSS, swap, fixed-category `smaps`, anonymous huge-page, and bounded
per-mapping deltas per cycle. A run that fails
before measurement retains the observed stabilization window in the Vulkan
allocation, pool, cache-lifecycle, and mapping metrics instead of replacing it
with zeroes.
The cumulative stabilization RSS gate runs only after the completed cycle's
health, debug snapshot, lifecycle deltas, and memory deltas have been traced
and stored, so the cycle that crosses the safety limit is not lost from the
diagnostic evidence.

### Vulkan Final Endurance Soak

Run the final Strix Halo endurance gate from a clean, pushed source tree after
the 30-minute Vulkan development soak passes:

```bash
PATH="$HOME/.cargo/bin:$PATH" \
python3 scripts/qualification/run.py \
  --variant vulkan-endurance \
  --host-id strix-halo \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen3.5-4B \
  qualification/workloads/serving-vulkan-endurance-v1.json
```

This is a distinct source-bound qualification identity, not a duration override
on a development receipt. It deliberately retains the development soak's exact
four-active-request service envelope, one/four-way prompt cohorts, periodic
cancellation, semantic oracle, fixed KV and Vulkan recycler policy, resident
route gates, 8 GiB host-memory floor, 512 MiB swap-growth cap, continuous
88/86 C process-group pacing, and independent 97 C hard stop. The different
deterministic seed and eight-hour duration exercise a longer request sequence
without silently widening the already-qualified operating point.

The timing contract gives the exact source build its checked 900-second limit,
then gives startup, warmup, and stabilization a fresh 1,800 seconds. Successful
stabilization starts a fresh 29,400-second measurement deadline: 28,800 required
seconds plus the unchanged 600-second request containment window. The outer
case timeout is 32,280 seconds, exactly those limits plus a separate 180 seconds
for cancellation, process-group shutdown, private-snapshot cleanup, and result
publication. Pacing counts against runtime setup and measurement deadlines. A
setup failure still reports zero measured duration, and a measurement failure
retains evidence only through the last fully completed and drained wave with
`measurement_final_snapshot_complete=0`.

A pass requires the same exact response, device, ownership, memory, thermal,
worker, and lifecycle verdicts as the development soak across the full measured
window. In particular, hardware absence or a Vulkan skip fails the required
case; every ITL outlier must have a bounded known attribution; every thermal
pause must be released; final health, debug, allocator, cache, process-memory,
and DRM snapshots must complete; and teardown must leave no server, request
worker, or private snapshot. This endurance result qualifies only this named
host and declared experimental operating point. It does not establish a
high-concurrency performance claim, stable-profile resident admission, or
portability to CUDA, Metal, or another Vulkan machine.

The clean pushed `3897239fe` source passed this contract on the named Strix
Halo. The retained receipt is
`qualification/receipts/vulkan/strix-halo/20260716t165320275412z-vulkan-strix-halo-serving-vulkan-endurance-7db5d986fd-v1.json`
with file hash
`sha256:118274f07578024cd1a65af2342a388f8be66dee636853d6e0d99698575ce604`.
Before any evidence documentation changed the tree, strict current-source,
local-artifact, and known-commit validation passed. The required case selected
`AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, reported no unsupported arm, and
left no server, qualification worker, compiler, or transient build unit.

Setup completed six cycles, 30 exact responses, and three cancellations in
1,099.02 seconds. The first four cycles established the fixed working set;
cycles five and six had zero DRM or live-buffer growth, allocation, free,
recycler miss, eviction, or uncached allocation. The final stabilization RSS
delta was 282,624 bytes and the second flat cycle admitted measurement. Setup
also exercised 520 resident forwards over 1,040 rows, completed ten rows at
width two, and ended with no active row, initial decline, or route failure.

Measurement ran for 28,867.72 seconds and completed 820 exact responses,
13,120 completion tokens, 82 confirmed cancellations, and 328 fully drained
waves. Process DRM began and ended at exactly 51,510,386,688 bytes, with only a
98,304-byte sampled active peak. Live Vulkan ownership began and ended at
51,492,696,448 bytes; the recycler retained exactly 3,574,202,368 bytes at both
boundaries. Across the measured window there were zero Vulkan allocations,
frees, cache misses, evictions, or uncached allocations and 11,366,760 recycler
hits. RSS grew 21,610,496 bytes from a 398,393,344-byte baseline and peaked at
468,049,920 bytes. The resident route made 18,512 forwards over 37,024 rows,
completed 320 rows at width two, and drained with no decline, failure, or active
row.

The external controller completed all 1,063 pacing events and left none active.
Cooling consumed 758.36 seconds of the unchanged deadlines; the longest pause
was 1.502 seconds, the highest pacing start and sampled package peak were
90,125 millicelsius, and the 97 C guard never tripped. All 3,907 ITL outliers
were attributed and none was unexplained. Host availability stayed at or above
17,928,245,248 bytes, host swap growth was 24,604,672 bytes inside the 512 MiB
cap, and the server's `smaps` swap growth was zero. Request, batching, device,
non-finite-response, synchronization, graph, ownership, worker, shutdown, and
snapshot failures were all zero; final snapshots completed, shutdown was
unforced and zero, and the private snapshot was removed.

This result establishes eight-hour endurance only for this declared
four-active-request operating point on this named host. It does not make the
point fast: aggregate measured completion goodput was 0.454 tokens/second,
p99 TTFT was 152,290.00 ms, and p99 ITL was 2,477.96 ms, including required
cooling time. Those values remain explicit performance limitations and do not
support a vLLM competitiveness, production-latency, higher-concurrency, broader
prompt, stable-profile, or cross-machine claim.

Never edit a receipt to make it pass. A failed receipt is useful evidence: keep
it when it identifies a reproducible product defect, fix the defect in a new
commit, and run a new receipt with a new ID. When a structurally valid command
result reports an effective configuration that differs from the selected
variant, the runner fails the case and clears the receipt-level
`effective_config` attestation, but retains the command's metrics, tolerances,
and details as counterexample evidence. Those measurements are diagnostic only;
they cannot support an accepted performance comparison because their effective
configuration was not verified.

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
