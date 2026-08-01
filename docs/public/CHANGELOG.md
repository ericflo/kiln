# Changelog

This page summarizes user-visible changes. It separates work on `main` from
published releases and links to the full engineering ledger for exhaustive
implementation detail.

## kiln-v0.5.2 — 2026-08-01

### First-class OpenEnv RL, fast stable serving, and the new dashboard

- Added native, reproducible OpenEnv discovery, rollout, replay, GRPO training,
  held-out evaluation, provenance, artifact verification, CLI, API, telemetry,
  and dashboard workflows for any protocol-compatible environment.
- Made Stable the normal complete product across CUDA, ROCm, Metal, and Vulkan,
  including inference, training, adapters, memory management, and qualified
  graph acceleration. Experimental is no longer required for normal speed or
  training.
- Added the new embedded ember dashboard with guided training and distillation
  on-ramps, contextual next actions, training-time hints, loss/run visibility,
  bundled fonts, and resilient empty/error/retry behavior.
- Vulkan execution derives route legality from device capabilities rather than
  device names or IDs. The tracked source receipt measures 13.46 decode tok/s
  at 74.29 ms mean inter-token latency on its recorded workload.

[Release notes and artifacts](https://github.com/ericflo/kiln/releases/tag/kiln-v0.5.2)

## kiln-v0.5.1 — 2026-07-30

### Vulkan recovery and a usable documentation path

- Restored the generic Vulkan compute routes that a broad portable fallback had
  disabled, recovering the tracked single-stream decode workload from
  0.142 tok/s to 13.43 tok/s.
- Kept dispatch independent of qualification-machine identity. The
  capability-derived policy now on `main` supersedes the release's global route
  table.
- Reworked the documentation around task-focused guides, a searchable reference
  library, an evidence-bound benchmark explanation, and a demo that distinguishes
  live UI from seeded or historical material.

[Release notes and artifacts](https://github.com/ericflo/kiln/releases/tag/kiln-v0.5.1)

## kiln-v0.5.0 — 2026-07-29

### Bounded thinking and typed startup policy

- Added token and wall-clock thinking budgets with explicit request, server
  default, streaming, and telemetry behavior.
- Made public environment overrides mechanically derive from typed configuration
  fields and rejected removed or conflicting configuration authorities.
- Expanded effective-configuration, runtime-policy, provenance, checkpoint,
  teacher, and artifact identity contracts.
- Moved CUDA, Metal, ROCm, Vulkan, batching, streaming-prefill, and training
  decisions toward immutable typed startup policy instead of hot-path environment
  reads.

[Release notes and artifacts](https://github.com/ericflo/kiln/releases/tag/kiln-v0.5.0)

## kiln-v0.4.1 — 2026-06-12

### Multi-turn prefix-cache repair

- Restored reusable prefix snapshots for multi-turn ROCm traffic.
- Added the same split-snapshot behavior to non-batched serving paths.
- Prevented an oversized entry from evicting useful cache contents when that
  entry could never fit.

[Release notes and artifacts](https://github.com/ericflo/kiln/releases/tag/kiln-v0.4.1)

## kiln-v0.4.0 — 2026-06-11

### Managed pi runs and Muon training

- Added server-managed pi runs with queued execution, event feeds, steering,
  follow-up, abort, persistence, and trace ingestion.
- Added the dashboard's Distill → Agent runs workflow.
- Made Muon the default optimizer for supported training routes, with AdamW and
  SGD still selectable per request.

[Release notes and artifacts](https://github.com/ericflo/kiln/releases/tag/kiln-v0.4.0)

## kiln-v0.3.5 — 2026-06-09

### Task-first dashboard workflows

- Added in-place dataset upload and training flows, post-training eval setup,
  adapter-focused comparisons, completion notifications, correction capture,
  and the embedded pi terminal.
- Improved first-run recovery and made adapter, health, and flywheel state agree
  after swaps or failures.

[Release notes and artifacts](https://github.com/ericflo/kiln/releases/tag/kiln-v0.3.5)

## Older releases and exhaustive detail

- [All GitHub releases](https://github.com/ericflo/kiln/releases)
- [Full engineering changelog](https://github.com/ericflo/kiln/blob/main/CHANGELOG.md)

Historical entries preserve the terminology and evidence available when each
change landed. Use current product guides for current configuration and support
claims.
