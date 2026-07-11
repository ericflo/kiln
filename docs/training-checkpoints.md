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

The manifest records the exact resolved training configuration, precision
policy, optimizer and scheduler step, next epoch/cursor and item order, data
identity, and every named RNG stream. Objective-specific reference, EMA,
reward-normalization, and sampler state is carried either in a checksummed
state file or in the manifest's versioned auxiliary state.

## Integration status

Native SFT supports exact resume. `SftConfig.checkpoint_interval` publishes a
resumable directory after every N committed optimizer steps. Cooperative
cancellation also publishes one at the next step boundary. SFT restores:

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

GRPO and OPD have not yet migrated their full loop state. Their existing
`checkpoint_interval` directories are PEFT snapshots and are not resumable.
The strict loader rejects them. Capability-distillation modes must explicitly
document resume support before their snapshots may be treated as checkpoints.

## SFT API and CLI

Start a checkpointed SFT job with either interface:

```bash
kiln train sft \
  --file corrections.jsonl \
  --adapter support-bot \
  --epochs 3 \
  --checkpoint-interval 25
```

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

Exact checkpoints are direct children of the configured adapter registry, for
example:

```text
support-bot-checkpoint-step-00000025.kiln-checkpoint/
```

They are published there while training is running, independently of the
temporary final-adapter staging tree. A process crash can therefore lose the
in-flight step, but not the last committed checkpoint.

`GET /v1/train/jobs/{job_id}` reports `latest_checkpoint`, including the
`resume_checkpoint` basename, committed step, total steps, next epoch/cursor,
and completion state. `kiln train status --job-id JOB_ID` prints the same
basename, and the training detail panel provides a copy action. Discovery
validates the bounded strict manifest without rehashing large state files on
every UI poll. Resume admission performs the full file-set, size, and checksum
validation before GPU work.

Resume with the identical dataset and training configuration:

```bash
kiln train sft \
  --file corrections.jsonl \
  --adapter support-bot \
  --epochs 3 \
  --checkpoint-interval 25 \
  --resume-checkpoint support-bot-checkpoint-step-00000025.kiln-checkpoint
```

```json
{
  "dataset_path": "/absolute/path/corrections.jsonl",
  "config": {
    "output_name": "support-bot",
    "epochs": 3,
    "checkpoint_interval": 25,
    "resume_checkpoint": "support-bot-checkpoint-step-00000025.kiln-checkpoint"
  }
}
```

The server accepts a single basename beneath the adapter registry. It also
accepts an absolute path only when that path is directly beneath the same
registry. Traversal, nested paths, other training kinds, and an adapter-name
mismatch fail before final-adapter staging or GPU training.

Resume is continuation, not warm-starting. The output name, data bytes,
resolved learning rate, optimizer, epochs, LoRA shape/scaling, precision,
model and base-weight shard, tokenizer, seed, backend runtime, and derived
gradient-checkpoint segmentation must match the checkpoint. Use `base_adapter`
for a weights-only warm start instead.

Checkpoint names are immutable. Prefer the newest checkpoint. If deliberately
resuming an older checkpoint, first archive any later same-adapter checkpoints
outside the adapter registry; otherwise the resumed run will fail when it
reaches an already-published step rather than overwrite it.
