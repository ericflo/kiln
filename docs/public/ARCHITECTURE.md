# Architecture

Kiln is one process with four product responsibilities: serving, training,
evaluation, and artifact lifecycle. They share model identity, accelerator
ownership, memory accounting, and adapter state instead of communicating
through a collection of opaque sidecars.

## Request path

```text
OpenAI-compatible request
        ↓
HTTP validation and request policy
        ↓
scheduler / continuous batching
        ↓
backend model execution
        ↓
CUDA · ROCm · Metal · Vulkan
        ↓
streamed events, usage, latency phases, lineage
```

The HTTP layer owns schema and error behavior. The scheduler owns admission,
batch assembly, cancellation, and request-local progress. Model code owns
architecture and adapter application. Backend code owns accelerator-specific
execution without changing the public request contract.

## Learning path

```text
examples or scored rollouts
        ↓
training admission and memory plan
        ↓
SFT · GRPO · OPD execution
        ↓
checkpoint and adapter manifest
        ↓
eval / comparison gate
        ↓
explicit adapter activation
```

Training does not silently replace the model a client is using. Jobs produce
named artifacts; requests or operators select an adapter explicitly. Manifests
bind the artifact to its base model, training inputs, configuration, and
provenance.

## Shared accelerator ownership

Inference and training share one accelerator budget. Kiln plans allocations,
rejects unsafe work before execution when possible, and exposes the effective
runtime state. “One process” does not mean every workload can run
simultaneously without a memory or latency tradeoff.

Each backend has two separate responsibilities:

1. implement correct operations for the model and training routes;
2. select efficient kernels from capabilities available on the active device.

Backend selection must not depend on a marketing device name. Qualification
receipts prove behavior on their recorded systems; they do not become product
routing rules.

## State and artifacts

| State | Lifetime | Owner |
|---|---|---|
| Base weights | model process | model loader and accelerator backend |
| KV cache and decode buffers | request / scheduler scope | batching and model execution |
| Training optimizer state | training job | trainer and checkpoint manager |
| LoRA adapters | durable artifact plus loaded runtime state | adapter registry |
| Eval suites and outcomes | durable local data | eval service |
| Receipts and lineage | durable evidence | qualification and provenance layers |

## Failure boundaries

Kiln distinguishes validation errors, admission failures, request failures,
accelerator failures, and process-terminal device loss. Readiness may degrade
without pretending an unavailable route is healthy. A device-loss error
requires process restart; a malformed request does not.

## Go deeper

- [HTTP API guide](https://ericflo.github.io/kiln/api.html)
- [Latency observability](../LATENCY_OBSERVABILITY.md)
- [Adapter manifest](../ADAPTER_MANIFEST.md)
- [Execution provenance](../EXECUTION_PROVENANCE.md)
- [Full architecture source](https://github.com/ericflo/kiln/blob/main/ARCHITECTURE.md)
