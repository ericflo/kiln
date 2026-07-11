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

The schema and atomic storage API are available. The native parameter codec can
save and restore exact adapter tensors plus AdamW moments or Muon momentum by
stable PEFT-compatible parameter name. It validates the complete tensor set,
shape, dtype, finite values, and optimizer step before mutation, and restores
both resident-device and host-fallback state so a later routing change cannot
silently reset momentum. CPU and ROCm continuation tests prove that the next
optimizer step produces byte-identical adapter and optimizer files.

Native SFT, GRPO, and OPD loop integration is still being completed as part of
the confidence-hardening program. The codec alone does not turn an existing
weight snapshot into a resumable checkpoint. Until a training surface
documents a `resume_checkpoint` field, its existing `checkpoint_interval`
output remains a PEFT weight snapshot and is not resumable.
