# Native Training Checkpoints

Kiln uses two deliberately separate training artifacts:

- A **resumable checkpoint** is an immutable directory ending in
  `.kiln-checkpoint`. It contains `checkpoint_manifest.json`, exact adapter and
  optimizer state, scheduler/cursor/RNG state, and checksums for every file.
- A **PEFT adapter snapshot** contains `adapter_config.json` and
  `adapter_model.safetensors`. It is portable to serving and Hugging Face PEFT,
  but it does not contain enough state to resume an interrupted optimizer run.

Never pass a PEFT adapter directory as a resume checkpoint. The loader rejects
directories without the resumable manifest instead of silently restarting the
optimizer from zero.

## Durability contract

Checkpoint schema v1 is defined by
`kiln_train::checkpoint::TrainingCheckpointManifest`. A writer creates a
hidden sibling staging directory, writes `.incomplete`, writes and synchronizes
the declared state files, computes their sizes and SHA-256 hashes, writes and
synchronizes the manifest, removes the sentinel, and atomically renames the
directory into place. Checkpoint names are immutable and cannot be overwritten.

The loader accepts only a canonical `.kiln-checkpoint` basename and rejects an
incomplete sentinel, unsupported schema/type, unknown manifest fields, invalid
or escaping paths, symlinks, missing or untracked files, size drift, and
checksum drift before returning any state to a trainer.

The publication state machine is process-kill qualified at all seven durable
boundaries: staging creation, sentinel sync, artifact write, manifest/artifact
sync, ready-to-rename, atomic rename, and parent-directory sync. Deterministic
fault injection and real child-process `SIGKILL` prove that the canonical final
basename is absent before rename and fully checksum-valid after rename. A crash
may retain a hidden UUID-suffixed staging directory, including the narrow state
after its sentinel was removed, but its noncanonical basename is never
loadable or discoverable as a resume checkpoint. A new writer can safely retry
the intended immutable basename beside that orphan.

On a shared serving GPU, checkpoint publication has two phases. Kiln takes the
same interruptible serving write lock used by an optimizer step only while it
copies authoritative adapter and optimizer buffers into CPU-owned tensors. It
then releases the lock before safetensors encoding, checksumming, file writes,
fsync, and rename. Logs report `gpu_wait_ms`, `device_snapshot_ms`, and
`publish_ms` separately, so an operator can distinguish inference contention or
device-transfer latency from storage latency. Final SFT adapter export follows
the same device-snapshot boundary and fails closed if resident synchronization
fails.

The manifest records the exact resolved training configuration, precision
policy, optimizer and scheduler step, next epoch/cursor and item order, data
identity, and every named RNG stream. Objective-specific reference, EMA,
reward-normalization, and sampler state is carried either in a checksummed
state file or in the manifest's versioned auxiliary state.

## Admission and compatibility ordering

Checkpoint resume is not an escape hatch around the running server's training
contract. Every dedicated SFT/GRPO/OPD endpoint, the intent-tagged `/v1/train`
front door, recipes, judge/self-improve, the distinct DistillRefresh route, and
OPD-backed distillation first run the cheap workload and resident
optimizer-tuple guards. These checks happen
before checkpoint loading or corpus scanning. Cheap teacher-alias validation
and metadata pinning may happen first to preserve established request errors;
remote/local teacher materialization, checkpoint checksumming, memory preflight,
and GPU reservation happen only after the workload guard. The queue worker
repeats the same static
guards before loading the checkpoint again, memory reservation, and device
residency.

The static workload gate requires a real readable runner, serving-profile
training GPU ownership, configured/resident weight-device agreement, a runtime
that resolves those weights, exact native backend/device identity, no
Marlin-packed projection, and the authoritative `kt_tape_authoritative`
forward/backward route. SFT additionally rejects multi-segment checkpointing on
the `full_logits` route. OPD additionally requires its loss and phase-B
backward routes. A CPU portable-reference optimizer tuple or raw Vulkan native
hook does not satisfy those workload requirements.

DistillRefresh has no exact-resume contract yet and is not admitted as OPD. Its
`distill_refresh` workload row always fails closed with
`distill_refresh is unavailable until admission pins separate exact SFT and OPD
phase plans, prepares the exact SFT rows, and reserves the maximum sequential
working set`. A correct future admission record must bind the SFT knowledge
phase and OPD behavior-recovery phase separately, include the exact phase-one
rows, and reserve the maximum of the two sequential phase peaks. Until then,
the endpoint and every recipe containing that step reject, after any cheap
teacher-alias validation/pinning, before checkpoint loading, remote/local
teacher materialization, corpus scanning, memory preflight, or GPU reservation.

After those checks, exact resume compares the checkpoint's backend/device,
base-weight manifest, base and resolved LoRA dtypes, optimizer kind and complete
state, rank, round-to-nearest mode, execution provenance, data identity,
checkpoint plan, and objective-specific tape/loss/backward route. Any mismatch
fails closed before GPU ownership; Kiln never changes optimizer, lowers rank,
falls back to a host update, or reinterprets a checkpoint through another
route. Live-memory admission remains dynamic and can still reject an otherwise
compatible workload/tuple.

The running snapshot is visible under `GET /v1/config` field
`training.optimizer_support`, schema
`{"id":"kiln.training-optimizer-support","version":1}`. Raw
`backend_implementation`, resident `optimizer_tuple`, and per-workload
`allowed_optimizer_kinds` are deliberately separate. `GET /v1/recipes`
provides `admission {supported, unavailable_reason}` for each built-in recipe,
but this static preview is not a checkpoint validation or memory reservation.

The tuple's `lora_rank.maximum` is the effective static minimum of
`backend_maximum` and model-derived `model_maximum`. A null backend maximum is
not unbounded resume authority: the concrete model maximum still applies, and
live memory can impose a lower admission ceiling. Exact resume compares the
recorded rank and model/backend envelope rather than clamping it into the
current range.

## Queued resume revalidation

Resume admission runs the full strict checkpoint loader, including every
declared artifact size and SHA-256. The queue retains only a compact identity:
the checkpoint ID and a digest of that fully validated manifest, whose artifact
entries cover the validated file hashes. It separately retains the admitted
effective seed. This keeps queue state bounded without weakening the admitted
identity.

At dequeue, before memory reservation, Kiln fully reloads the checkpoint,
revalidates all declared artifacts, recomputes the compact manifest identity,
and derives the effective seed again. The worker requires both identity and
seed to equal the admitted values. A replaced manifest, changed artifact hash,
different checkpoint ID, or changed seed rejects the queued job instead of
continuing from different state.

This invariant is revalidation, not a filesystem snapshot. The queue does not
copy or pin externally mutable checkpoint files, and the comparison cannot
eliminate a mutation race after the dequeue reload. Operators must keep exact
checkpoint directories immutable and access-controlled for their entire use;
later strict loaders remain authoritative for any bytes they open.

## Effective seed admission contract

Every admitted SFT, GRPO, OPD, and OPD-backed distillation path resolves one
effective seed before publishing either its tracking record or queue entry.
This includes the dedicated endpoints, `/v1/train`, recipe steps, judge
distillation, scheduled or manual self-improvement, and the distill
merge/pump/self endpoints. DistillRefresh is a separate workload and currently
fails its static guard before an effective seed or queue entry exists. For an
admitted fresh run, a request-provided `config.seed` wins; otherwise the server
draws one value exactly once and writes it back into the queued effective
configuration.

The one-job submission response contains `effective_seed`. Status, queue, job
detail, and on-disk training history retain the same value. JSON-facing fields
are decimal strings, not JSON numbers, so JavaScript clients preserve the full
`u64` range. Composite recipe and self-improvement responses expose an
`effective_seeds` object keyed by job ID. The CLI prints the value on submit
and status, and the browser exposes it on every job card and as a copyable job
detail field. Legacy archived jobs may omit it; new jobs may not.

For resume, the checkpoint's `rng_states["lora-init"].seed` is authoritative.
Admission copies it into the effective configuration when the caller omits a
seed and rejects a conflicting explicit value before queue publication. The
worker validates the same checkpoint and seed again before GPU ownership.

The seed is one input to reproducibility, not a standalone replay guarantee.
Same-seed runs can diverge when model weights, data/order, tokenizer/template,
build, backend, device, driver/runtime, precision, kernels, or effective
environment differ. Exact checkpoint continuation is guaranteed only inside
the deterministic envelope recorded and validated by the checkpoint contract.

## Base-weight binding

Every new exact SFT, GRPO, and OPD checkpoint embeds the loader-owned
`kiln.base-weight-shards.v1` manifest in
`auxiliary_state.base_weight_shard_manifest`. The existing
`base_model_weights_sha256` field must equal that manifest's validated aggregate.
Capture fails before publication if either value is absent or inconsistent.

Resume validates both full manifests before GPU ownership and compares their
byte-content identity. A changed shard digest, size, or multiplicity is
incompatible. Paths, safetensors index order, and audit filenames do not affect
content identity, so relocating or renaming otherwise identical shards remains
valid. The filenames are still persisted to identify the audited source files.

Legacy exact checkpoints that recorded only the aggregate fail closed because
they cannot prove the constituent shards. Use them only as non-resumable
serving or weights-only artifacts; start a new run to publish a checkpoint with
the complete binding. Successful `train_receipt.json` and
`adapter_manifest.json` files retain the same full manifest, and their readers
reject internal tampering. See
[Base-Weight Provenance](BASE_WEIGHT_PROVENANCE.md) for the strict JSON schema
and aggregate algorithm.

## Execution binding

Every new exact SFT, GRPO, and OPD checkpoint also embeds the complete
`kiln.execution-provenance.v1` record in
`auxiliary_state.execution_provenance`. Capture validates the self-verifying
record before publishing any checkpoint. It binds the exact running executable,
optional source revision, backend/device and bounded driver/runtime evidence,
model/tokenizer/inference-template identity, the effective supervised-training
template identity, inference and training precision policy, compiled kernel
contract, and effective server configuration/environment. SFT checkpoint
auxiliary state repeats `training_chat_template_sha256` explicitly so an exact
resume comparison cannot silently cross assistant-label contracts.

Resume validates both records before GPU ownership and requires their canonical
`provenance_sha256` values to match. The checkpoint's existing concrete
`precision_policy` separately records the parameter, optimizer-state,
activation, gradient, and stochastic-rounding policy used by the trainer.
Changing either the execution envelope or those concrete dtypes is not an exact
continuation.

Current server training records `{"mode":"round_to_nearest"}` and admits only
that policy. Stochastic rounding remains an explicit optimizer-library option,
not a server/config/environment option. Therefore a legacy exact checkpoint
whose precision policy records `{"mode":"stochastic","seed":...}` fails
closed during precision comparison before GPU ownership; Kiln does not discard
the recorded seed, silently convert the optimizer state, or resume under a
different rounding rule. `KILN_BF16_STOCHASTIC_ROUND` has been removed and
cannot be used to make such a checkpoint resumable. The removed
`KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK` and
`KILN_CUDA_TRAINING_OPTIMIZER_FALLBACK`,
`KILN_ROCM_TRAINING_OPTIMIZER_FALLBACK`,
`KILN_METAL_TRAINING_OPTIMIZER_FALLBACK`, and
`KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK` likewise have no configuration fields
or compatibility aliases. Use the artifact as a non-resumable record or start
a new round-to-nearest run.

Legacy exact checkpoints that contain only `backend_runtime`, package version,
or other partial runtime strings fail closed because they cannot prove the
complete envelope. They remain usable only as non-resumable serving or
weights-only artifacts. Successful `train_receipt.json` files retain the full
record under `runtime.execution_provenance` and the concrete dtypes under
`runtime.training_precision`; `adapter_manifest.json` copies both. Readers
validate these fields when present while continuing to read legacy non-resume
artifacts that predate them. SFT receipts additionally record
`tokenizer.training_chat_template_hash`, and adapter manifests copy it; readers
reject malformed values or disagreement with the execution record. See
[Execution Provenance](EXECUTION_PROVENANCE.md) for the canonical schema and
evidence sources.

## Checkpoint planning identity

Every exact SFT, GRPO, and OPD checkpoint binds the immutable
`TrainingRuntimeContext` used to plan the run. Every planning identity records
effective VRAM, gradient-checkpoint policy, runtime device, resolved
streaming-prefill policy, and this complete checkpoint-boundary policy:

```json
{
  "recompute_mode": "auto",
  "recompute_threshold_tokens": 8192,
  "anchor_stride": null,
  "cache_target_bytes": 6442450944
}
```

GRPO and OPD use schema `kiln.training-checkpoint-planning.v3`. SFT uses
`kiln.training-checkpoint-planning.v4`, which adds the backend-owned loss route:

```json
{
  "schema": "kiln.training-checkpoint-planning.v4",
  "sft_loss_route": "kt_tape_flce",
  "checkpoint_boundary_policy": {
    "recompute_mode": "auto",
    "recompute_threshold_tokens": 8192,
    "anchor_stride": null,
    "cache_target_bytes": 6442450944
  }
}
```

The other current route values are `vulkan_active_rows` and `full_logits`.
The enum comes from the admitted backend capability, not checkpoint JSON,
request input, TOML, CLI, or process environment. SFT admission pins it in the
queued job, the worker compares it with the resident runner before memory
reservation, and the trainer compares it with its execution backend before
allocation. The pinned value then drives every loss step and this v4 identity.

SFT and GRPO store the planning object under
`auxiliary_state.training_runtime_planning_identity`; OPD retains it under its
existing `auxiliary_state.checkpoint_planning` key. Exact resume compares the
complete auxiliary state rather than silently adopting the new process policy.
A checkpoint that lacks the boundary policy or contains any different mode,
threshold, stride, cache target, runtime route, or schema is rejected as
planning drift before continuation.

This nested version change does not change the outer
`TrainingCheckpointManifest` envelope, which remains checkpoint schema v1.
Older artifacts can still be inspected and checksum-validated, but a v2
planning identity cannot authorize exact continuation under v3, and an SFT v3
identity cannot authorize continuation under SFT v4 because it cannot prove
which loss implementation admission budgeted and execution used. Start a fresh
training run to create a resumable checkpoint with the current planning
contract; do not edit the manifest or auxiliary JSON.

Sparse boundary replay is an SFT execution policy. In `auto` mode, sequences
at least `recompute_threshold_tokens` replay between sparse anchors;
`enabled` always replays and `disabled` retains every boundary. An explicit
positive `anchor_stride` wins over cache-based shape selection. With
`anchor_stride = null`, the same pure shape calculation used by admission and
execution derives a stride from sequence length, planned segment count, hidden
width, boundary dtype, and `cache_target_bytes`. No trainer or admission path
re-reads process environment after startup.

GRPO and OPD currently retain all `num_segments + 1` checkpoint boundaries and
do not use this policy to choose their live layout. The policy is nevertheless
part of their common v3 runtime identity. Changing one of the four startup
fields therefore rejects exact GRPO or OPD resume too, even though the setting
is execution-inert for those modes today. This conservative rule prevents a
future boundary-layout expansion from reinterpreting an older checkpoint.

For SFT, `kt_tape_flce` and `vulkan_active_rows` are compatible with a
multi-segment plan. `full_logits` is not: checkpoint tails execute outside an
active kt tape, so admission rejects more than one segment and the trainer
rechecks the invariant before a forward. This compatibility fact is part of
planning, but route names are not operator controls. The retired
`KILN_USE_FLCE` spelling has no alias or current effect.

## Integration status

Native SFT, inline and streamed-JSONL GRPO, and OPD support exact resume. A
legacy PEFT snapshot from any training mode remains serving-only and is never
accepted as an exact checkpoint.

SFT checkpoint scheduler state additionally binds the fixed
`native_online_lora_v1` microbatch, accumulation, warmup, and clipping values.
See [Native SFT Profile](NATIVE_SFT_PROFILE.md).

### Repeatability qualification

The opt-in real-hardware qualification runs each public SFT, inline GRPO,
streamed-JSONL GRPO, and OPD route twice from a clean adapter/checkpoint
directory with the same resident model, data, effective configuration, and
seed. It requires exact equality of the loss sequence, final PEFT bytes and
SHA-256, checkpoint adapter and optimizer tensors, objective-specific reference
state, semantic manifest, and loop state. Each completed run must also record an
`adapter_model_sha256` in `train_receipt.json` that matches the actual
`adapter_model.safetensors` bytes. The same fixture then cancels a third run at
a committed boundary, resumes its immutable checkpoint through the public HTTP
route, and compares the combined result with the first clean run.

These tests currently run locally on real ROCm and Vulkan devices when
`KILN_QUALIFICATION=1`; the default CPU CI job does not execute them. ROCm and
Vulkan feature builds can compile the gated paths without opting into the
hardware run. Passing establishes repeatability inside that one process's
recorded model, executable, source, backend, device, driver/runtime, precision,
kernel, tokenizer/template, configuration, environment, data, and seed
envelope. It does not claim that different devices, backends, builds, drivers,
or machines produce identical bytes.

### SFT

`SftConfig.checkpoint_interval` publishes a resumable directory after every N
committed optimizer steps. Cooperative cancellation also publishes one at the
next step boundary. SFT restores:

- adapter parameters and AdamW moments or Muon momentum by stable parameter
  name (SGD has no optimizer artifact);
- optimizer/scheduler step, next epoch, exact shuffled item order, and cursor;
- LoRA initialization and shuffle seed streams;
- loss history, partial epoch loss, divergence state, and gradient diagnostics;
- effective configuration, precision policy, data identity, model config,
  base-weight shard bytes, tokenizer identity, backend runtime, and the derived
  per-example gradient-checkpoint plan;
- the pinned backend loss route through SFT planning identity v4.

Completed SFT runs also expose the executed route at
`train_receipt.json -> runtime.sft_loss_route`; the receipt field is not a
replacement for the checksummed planning identity used by resume.

The native parameter codec validates the complete tensor set, shape, dtype,
finite values, and optimizer step before mutation. It restores both
resident-device state and the portable F32 checkpoint representation so a
resume cannot silently reset momentum or change optimizer route. Direct-library
CPU portable-reference tests and ROCm continuation tests cover byte-identical
next-step state. Real ROCm BF16 and Vulkan F32 qualification compares two fresh
runs and an uninterrupted run with its cancelled-and-resumed counterpart
through final adapter, optimizer, receipt, manifest, and loop-state artifacts.

### GRPO

`GrpoConfig.checkpoint_interval` publishes an exact checkpoint after every N
committed optimizer groups. Cooperative cancellation publishes one after the
current group settles. GRPO restores:

- policy adapter parameters and AdamW moments or Muon momentum by stable name;
- the frozen/EMA reference tensors and exact EMA refresh cadence;
- committed group cursor, loss history, data/token/gradient/ECHO/policy-audit
  accumulators, and phase plus GPU-writer timings;
- effective configuration, precision, model/base-weight/tokenizer/backend
  identities, RNG streams, trainable parameter order, and the derived
  gradient-checkpoint plan;
- for inline batches, the exact filtered group order and content identity;
- for streamed JSONL, the physical line number and byte offset plus every
  consumed line hash, token count, and gradient plan.

The JSONL route performs a memory-bounded CPU preflight before model upload.
Resume loads and validates the complete bundle before GPU setup, seeks directly
to the committed JSONL offset, and revalidates each next group before its
optimizer step. The route is part of checkpoint identity: an inline checkpoint
must resume from identical inline groups, and a JSONL checkpoint must resume
from the identical JSONL bytes through the streamed route.

Real ROCm and Vulkan qualification compares two fresh runs and an uninterrupted
run with its cancelled-and-resumed counterpart for both routes. Losses, receipt
hashes, final adapter bytes, intermediate adapter/optimizer/reference artifacts,
EMA cadence, and diagnostic state match exactly.

### OPD

`OpdConfig.checkpoint_interval` defaults to 25 and publishes an exact
checkpoint after every N committed optimizer steps. Cooperative cancellation
publishes at the next settled source/sample candidate boundary. OPD keeps the
optimizer-step counter separate from the candidate cursor because a sampled
rollout can be empty and consume a deterministic candidate without applying an
update. OPD restores:

- adapter parameters plus AdamW moments or Muon momentum by stable parameter
  name (SGD has no optimizer artifact), with optimizer/scheduler step;
- the next epoch and source/sample candidate cursor, prepared source order,
  loss history, token/data/ECHO/gradient diagnostics, phase and GPU-writer
  timings, and stateful collapse-guardrail accumulators;
- independent LoRA-initialization and on-policy rollout RNG streams, including
  the one effective seed resolved for the original run;
- effective configuration, auto-resolved sample count, precision policy,
  prompt or off-policy dataset identity, model config, base-weight content,
  tokenizer identity, backend runtime, and any requested `base_adapter`;
- teacher capabilities, canonical teacher identity when one exists, and the
  authoritative content revision of the exact numeric source. Deterministic or
  composite fixture teachers bind their generated numeric rows or algorithm
  contract rather than pretending to be a model identity.

Resume requires the identical prompt array or off-policy dataset bytes,
training mode and effective configuration, output adapter, and exact teacher
content revision. A same-name teacher that was re-registered with different
model, tokenizer, runtime, adapter, protocol, scoring bounds, fixture rows, or
algorithm identity is rejected. The complete immutable bundle is restored
before continuation; a `base_adapter` is not a substitute for checkpoint
state.

Real ROCm BF16 and Vulkan F32 qualification compares two fresh OPD runs and an
uninterrupted run with its cancelled-and-resumed counterpart. Loss history,
receipt hashes, final and intermediate adapter bytes, optimizer tensors,
cursor/RNG state, and diagnostics match exactly, while inference can acquire
the shared device between settled candidate phases.

### Legacy snapshots

The strict loader rejects older SFT, GRPO, and OPD PEFT snapshots that lack
`checkpoint_manifest.json`. Capability-distillation modes use the same OPD
checkpoint contract, but a generated teacher that is not available as a
registered alias cannot be prepared automatically by the browser. Restore the
exact source and submit through a compatible API/CLI route; never relabel a
serving snapshot as a checkpoint.

## API, CLI, and browser workflow

Start checkpointed SFT, GRPO, or OPD from the CLI:

```bash
kiln train sft \
  --file corrections.jsonl \
  --adapter support-bot \
  --epochs 3 \
  --checkpoint-interval 25

kiln train grpo \
  --file scored-groups.jsonl \
  --adapter reward-bot \
  --checkpoint-interval 25

kiln train opd \
  --file opd-request.json \
  --adapter distilled-bot \
  --teacher qwen35@vllm \
  --checkpoint-interval 25
```

Each successful command prints the materialized effective seed alongside its
job ID. `kiln train status` prints the same exact decimal value; use the job
detail or dashboard copy action when recording a run manifest.

The GRPO command treats `.jsonl` as the memory-bounded streamed route and a
JSON request/batch containing `groups` as the inline route. The equivalent
streamed API requests are:

```json
{
  "dataset_path": "/absolute/path/corrections.jsonl",
  "config": {
    "output_name": "support-bot",
    "epochs": 3,
    "checkpoint_interval": 25
  }
}
```

```json
{
  "dataset_path": "/absolute/path/scored-groups.jsonl",
  "config": {
    "output_name": "reward-bot",
    "checkpoint_interval": 25
  }
}
```

For inline GRPO, replace `dataset_path` with the exact `groups` array. Named
datasets submitted by the browser are resolved to the server's JSONL copy and
use the streamed route.

The direct OPD API uses the same configuration fields. OPD defaults to a
25-step cadence, but spelling it out makes the durability policy visible:

```json
{
  "prompts": [
    {"messages": [{"role": "user", "content": "Explain why the sky is blue."}]}
  ],
  "teacher": "qwen35@vllm",
  "config": {
    "output_name": "distilled-bot",
    "checkpoint_interval": 25
  }
}
```

Submit that object from `opd-request.json`, or continue it with the identical
file and the exact basename reported by status:

```bash
kiln train opd \
  --file opd-request.json \
  --adapter distilled-bot \
  --teacher qwen35@vllm \
  --checkpoint-interval 25 \
  --resume-checkpoint distilled-bot-checkpoint-step-00000025.kiln-checkpoint
```

Exact checkpoints are direct children of the configured adapter registry, for
example:

```text
support-bot-checkpoint-step-00000025.kiln-checkpoint/
```

They are published there while training is running, independently of the
temporary final-adapter staging tree. A process crash can therefore lose the
in-flight step, group, or OPD candidate, but not the last committed checkpoint;
an interrupted checkpoint publication is either absent or complete at its
canonical basename.

`GET /v1/train/jobs/{job_id}` reports `latest_checkpoint`, including the
`resume_checkpoint` basename, `training_kind`, `data_source_kind`, committed
step, total steps, next epoch/group/candidate cursor, and completion state. Job
detail also exposes the validated effective configuration, data hash/count,
and OPD teacher alias, `teacher_identity_revision`, and
`teacher_content_revision`. The identity revision is comparable with
`GET /v1/teachers`; the content revision separately binds exact materialized
logit rows or another numeric source. `kiln train status --job-id JOB_ID`
prints the same basename. The training detail panel
labels SFT epoch/example, inline/JSONL GRPO group, and OPD candidate cursors
separately and provides copy and prepare-resume actions. Preparation clears
inline SFT/GRPO data and OPD prompts that cannot be recovered from replay
metadata. OPD preparation selects a teacher only when the currently registered
alias has the checkpoint's exact identity revision; missing legacy bindings or
teacher drift fail closed. The server then reconstructs and verifies the exact
content revision during resume admission. Reinsert the identical source before
submit.

Status discovery validates the bounded strict manifest without rehashing large
state files on every poll. Resume admission performs the full file-set, size,
and checksum validation before GPU work.

The older `replay.jsonl` and `lineage.json` audit trail is not a checkpoint.
`kiln-replay verify` checks only request-lineage hash integrity; it does not
execute training or compare losses, tensors, or outputs. See
[Request-Lineage Integrity](REPLAY_INTEGRITY.md).

Resume with the identical dataset, route, and training configuration:

```bash
kiln train sft \
  --file corrections.jsonl \
  --adapter support-bot \
  --epochs 3 \
  --checkpoint-interval 25 \
  --resume-checkpoint support-bot-checkpoint-step-00000025.kiln-checkpoint

kiln train grpo \
  --file scored-groups.jsonl \
  --adapter reward-bot \
  --checkpoint-interval 25 \
  --resume-checkpoint reward-bot-checkpoint-step-00000025.kiln-checkpoint
```

```json
{
  "dataset_path": "/absolute/path/scored-groups.jsonl",
  "config": {
    "output_name": "reward-bot",
    "checkpoint_interval": 25,
    "resume_checkpoint": "reward-bot-checkpoint-step-00000025.kiln-checkpoint"
  }
}
```

The server accepts a single basename beneath the adapter registry. It also
accepts an absolute path only when that path is directly beneath the same
registry. Traversal, nested paths, other training kinds, and an adapter-name
mismatch fail before final-adapter staging or GPU training.

Resume is continuation, not warm-starting. The output name, data bytes and
route, resolved learning rate, optimizer, SFT epochs or GRPO objective/filter
settings, LoRA shape/scaling, precision, model and base-weight shards,
tokenizer, seed, backend runtime, and derived gradient-checkpoint segmentation
must match the checkpoint. Use `base_adapter` for a weights-only warm start
instead.

Checkpoint names are immutable. Prefer the newest checkpoint. If deliberately
resuming an older checkpoint, first archive any later same-adapter checkpoints
outside the adapter registry; otherwise the resumed run will fail when it
reaches an already-published step rather than overwrite it.
