# Serving Benchmark Protocol

This protocol compares source-bound Kiln and reference servers with ordinary
wall-clock timing. It does not impose a host temperature policy, CPU quota,
machine-specific accelerator target, or fixed host-memory ceiling.

## Scope

The serving campaign is responsible for:

- binding every run to a clean pushed source revision;
- binding model, tokenizer, runtime, binary, configuration, and workload
  artifacts by content hash;
- launching one owned server process group;
- waiting for bounded readiness;
- executing the declared request corpus;
- retaining closed-schema request, response, timing, and lifecycle evidence;
- shutting down the owned process group and rejecting residue; and
- writing a strict qualification receipt.

Hardware-specific results describe only the hardware that produced them. They
do not change portable product defaults or become admission requirements for
other hosts.

## Workload Contract

Each workload JSON file under `qualification/workloads/` declares:

- backend and device requirements;
- deterministic seed and request order;
- concurrency and repetition counts;
- command, environment, timeout, and output assertions;
- result protocol and declared metrics; and
- comparison rules.

Backend-specific cases may require CUDA, ROCm, Vulkan, Metal, or another
accelerator API. They must not require a particular laptop, product name, GPU
UUID, temperature sensor, or single architecture unless the case is explicitly
a hardware regression fixture.

## Source And Runtime Identity

Performance publication requires a clean worktree and `HEAD` equal to
`origin/main`. The receipt records the commit and Git tree.

Kiln runs bind the exact binary and configuration. Reference runs bind the
runtime manifest, interpreter, package set, model, tokenizer, and launch
implementation. Captured runtime manifests are portable descriptions of the
runtime; they do not encode host temperature or host resource policy.

## Process Boundary

Owned launches use a new process group and bounded shutdown. A successful run
requires:

1. readiness before the declared startup deadline;
2. no unexpected listener owner;
3. no request timeout or output-contract failure;
4. graceful termination, followed by forced group cleanup only when needed;
5. no remaining owned process or listener; and
6. a complete lifecycle record.

Linux and WSL2 qualification may use systemd user scopes and cgroup accounting
when available. The generic WSL2 boundary defaults to no CPU or memory ceiling
and a bounded PID count. Unsupported isolation is reported explicitly rather
than silently substituted.

## Timing

All deadlines and throughput calculations use monotonic wall-clock time.
Startup, request, shutdown, and campaign timeouts are not extended by host
temperature, scheduler delay, or CPU feedback.

The core serving metrics are:

- request-window aggregate output tokens per second;
- per-request output tokens per second;
- time to first token;
- inter-token latency p50, p95, p99, and maximum;
- request and output counts;
- lifecycle and cleanup status; and
- backend-specific graph, cache, and memory telemetry when available.

Telemetry is diagnostic evidence. It does not alter elapsed time or pause the
measured process.

## Running A Campaign

Select a checked-in workload and a variant supported by the current host:

```bash
python3 scripts/qualification/run.py \
  qualification/workloads/<workload>.json \
  --variant <variant> \
  --host-id <host-id> \
  --model /absolute/path/to/model \
  --model-id <model-id> \
  --output .qualification/receipts/result.json
```

Run `--help` for the exact arguments supported by the current driver. Do not
reuse arguments from historical receipts or older documentation.

## ROCm And Vulkan Diagnostics

ROCm and Vulkan qualification drivers build for the available toolchain target.
They do not inject `gfx1151` or any other single architecture. Full-logit and
layer/path attribution drivers use:

- an offline private-network worker where supported;
- a start gate;
- a finite runtime;
- process-group termination and cleanup;
- exact model/request/source hashes; and
- numerical evidence with a self-hashed result.

The current schemas are:

- `rocm-hf-next-token-oracle-v2.schema.json`;
- `rocm-hf-path-attribution-v2.schema.json`; and
- `rocm-hf-layer-attribution-v2.schema.json`.

Version 1 results that contain machine-specific temperature policy are
historical records and are not accepted by the current validators.

## Retention

Retain passing and diagnostically useful failing receipts without rewriting
their contents. A receipt is evidence of one run, not configuration policy.

Before publication:

1. validate the workload and result schemas;
2. validate source and artifact hashes;
3. verify cleanup and listener ownership;
4. verify correctness and numerical tolerances;
5. compare only like-for-like workload rows; and
6. state the exact host/backend scope of the result.

Historical machine-specific receipts may remain for auditability. Current
product configuration, workload admission, and documentation must not depend
on them.
