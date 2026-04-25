---
name: Feature request
about: Propose a new feature or API
title: "[feature] "
labels: ["enhancement"]
---

## Use case

What task, what workload, and why current capability is insufficient.

## Scope check

CONTRIBUTING.md lists three core constraints kiln preserves:

- Single-model: Qwen3.5-4B only
- Single-process: no Python sidecar as the primary path
- Consumer-GPU friendly: must run on a single 24GB GPU (RTX 3090/4090) for the core path

Confirm this proposal preserves all three, or argue why an exception is justified.

## Proposed API or config surface

Endpoint shape, env flag, CLI flag, config field — whatever the user-facing surface is.

## Alternatives considered

What else you tried or considered, and why this proposal is preferable.
