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

## Effective seed admission contract

Every public SFT, GRPO, OPD, and OPD-backed distillation path resolves one
effective seed before publishing either its tracking record or queue entry.
This includes the dedicated endpoints, `/v1/train`, recipe steps, judge
distillation, scheduled or manual self-improvement, and the distill
refresh/merge/pump/self endpoints. A request-provided `config.seed` wins for a
fresh run; otherwise the server draws one value exactly once and writes it back
into the queued effective configuration.

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

## Integration status

Native SFT, inline and streamed-JSONL GRPO, and OPD support exact resume. A
legacy PEFT snapshot from any training mode remains serving-only and is never
accepted as an exact checkpoint.

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
  per-example gradient-checkpoint plan.

The native parameter codec validates the complete tensor set, shape, dtype,
finite values, and optimizer step before mutation. It restores both
resident-device and host-fallback state so a later optimizer route cannot
silently reset momentum. CPU and ROCm continuation tests cover byte-identical
next-step state; the ROCm SFT qualification compares an uninterrupted run with
a cancelled-and-resumed run through final adapter and optimizer artifacts.

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

Real ROCm and Vulkan qualification compares uninterrupted runs with
cancelled-and-resumed runs for both routes. Losses, final adapter bytes,
intermediate adapter/optimizer/reference artifacts, EMA cadence, and diagnostic
state match exactly.

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

Real ROCm BF16 and Vulkan F32 qualification compares uninterrupted OPD with a
cancelled-and-resumed run. Loss history, final and intermediate adapter bytes,
optimizer tensors, cursor/RNG state, and diagnostics match exactly, while
inference can acquire the shared device between settled candidate phases.

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
