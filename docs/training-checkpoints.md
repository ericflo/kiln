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

The loader rejects an incomplete sentinel, unsupported schema/type, unknown
manifest fields, invalid or escaping paths, symlinks, missing or untracked
files, size drift, and checksum drift before returning any state to a trainer.

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

## Integration status

Native SFT and GRPO support exact resume. OPD does not yet. A legacy PEFT
snapshot from any training mode remains serving-only and is never accepted as
an exact checkpoint.

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

### OPD and legacy snapshots

OPD `checkpoint_interval` output remains a PEFT adapter snapshot without exact
optimizer, cursor, RNG, or reference state. It is not resumable. The strict
loader likewise rejects older SFT/GRPO PEFT snapshots that lack
`checkpoint_manifest.json`. Capability-distillation modes must explicitly
document exact resume support before their snapshots may be treated as
checkpoints.

## API, CLI, and browser workflow

Start checkpointed SFT or GRPO from the CLI:

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
```

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

Exact checkpoints are direct children of the configured adapter registry, for
example:

```text
support-bot-checkpoint-step-00000025.kiln-checkpoint/
```

They are published there while training is running, independently of the
temporary final-adapter staging tree. A process crash can therefore lose the
in-flight step or group, but not the last committed checkpoint.

`GET /v1/train/jobs/{job_id}` reports `latest_checkpoint`, including the
`resume_checkpoint` basename, `training_kind`, `data_source_kind`, committed
step, total steps, next epoch/group cursor, and completion state. `kiln train
status --job-id JOB_ID` prints the same basename. The training detail panel
labels SFT epoch/example cursors separately from inline/JSONL GRPO group
cursors, provides copy and prepare-resume actions, and exposes checkpoint fields
in both advanced forms. Preparing a resume clears inline data that cannot be
verified from replay metadata; select the identical source before submit.

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
