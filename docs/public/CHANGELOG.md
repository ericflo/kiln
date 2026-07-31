# Changelog

This page summarizes user-visible changes. It separates work on `main` from
published releases and links to the full engineering ledger for exhaustive
implementation detail.

## Unreleased on `main`

### Capability-derived Vulkan execution

- Vulkan policy v6 derives route legality from the selected device's reported
  Vulkan capabilities. Device names, vendor IDs, device IDs, PCI identities,
  and driver names are not routing inputs.
- Instance creation negotiates the loader API up to Vulkan 1.2. Most shaders
  target Vulkan 1.0; subgroup-tiled attention shaders require the corresponding
  Vulkan 1.1 subgroup capabilities and otherwise use an untiled route.
- The tracked short diagnostic measures 13.46 decode tok/s at 74.29 ms mean
  inter-token latency. This is source-verified evidence, not a published-release
  or cross-device claim. See [Benchmarks](BENCHMARKS.md) for the workload,
  hardware, source revision, and remaining prefill gap.

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
