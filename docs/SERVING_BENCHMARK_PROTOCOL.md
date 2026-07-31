# Serving benchmark protocol

This protocol compares source-bound Kiln and reference servers with ordinary
wall-clock timing. It does not impose a host temperature policy, CPU quota,
machine-specific accelerator target, or fixed host-memory ceiling.

Use it when you need a result that another reader can trace to an exact
workload, source tree, model, runtime, server lifecycle, and set of output
bytes. Do not use one receipt as a universal performance claim.

## What the campaign owns

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
other hosts. A host name or device name in a receipt is provenance, not
dispatch policy.

## Workload contract

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
a hardware regression fixture. Even then, the identity belongs to the fixture
and its receipt—not to Kiln's product routing.

The [workload schema](../qualification/schema/workload-v1.schema.json) is the
machine-readable authority for variants, variables, case order, timeouts,
output assertions, metrics, and comparison policy.

## Source and runtime identity

Performance publication requires a clean worktree and `HEAD` equal to
`origin/main`. The receipt records the commit and Git tree.

Kiln runs bind the exact binary and configuration. Reference runs bind the
runtime manifest, interpreter, package set, model, tokenizer, and launch
implementation. Captured runtime manifests are portable descriptions of the
runtime; they do not encode host temperature or host resource policy.

## Process boundary

Owned launches use a new process group and bounded shutdown. A successful run
requires:

1. readiness before the declared startup deadline;
2. no unexpected listener owner;
3. no request timeout or output-contract failure;
4. graceful termination without forced cleanup;
5. no remaining owned process or listener; and
6. a complete lifecycle record.

The runner still attempts forced group cleanup when graceful shutdown fails so
that a broken campaign does not leave a server behind. That cleanup is a
failure signal, not a successful shutdown.

Linux and WSL2 qualification may use systemd user scopes and cgroup accounting
when available. The generic WSL2 boundary defaults to no CPU or memory ceiling
and a bounded PID count. Unsupported isolation is reported explicitly rather
than silently substituted.

An owned-server recipe uses the
[launch schema](../qualification/schema/serving-benchmark-server-launch-v1.schema.json).
Its `command` is a direct argv vector, not a shell command. Startup waits for
`/v1/models`; shutdown verifies both the process group and listener.

## Timing

All deadlines and throughput calculations use monotonic wall-clock time.
Startup, request, shutdown, and campaign timeouts are not extended by host
temperature, scheduler delay, or CPU feedback.

The core metrics have different denominators:

| Metric | Definition | What it answers |
|---|---|---|
| Request-window output throughput | All completed output tokens divided by the full measured request window | What did this concurrent workload deliver end to end? |
| Per-request output throughput | One request's output tokens divided by that request's elapsed time | How quickly did that request finish, including its non-decode time? |
| Time to first token (TTFT) | Request dispatch to the first semantic output token | How long did the user wait before output began? |
| Inter-token latency (ITL) | Wall time between consecutive semantic output tokens after the first | How quickly did decode deliver later tokens? |
| SLO goodput | Output tokens from requests that pass the declared latency, validity, and route gates, divided by the request window | How much useful work met the workload's service objective? |

Receipts also retain request and output counts, lifecycle and cleanup status,
and backend-specific graph, cache, route, and memory telemetry when available.

Telemetry is diagnostic evidence. It does not alter elapsed time or pause the
measured process.

Do not compare a request-window throughput number with a decode rate. Decode
rate is derived from ITL for a defined path; request-window throughput includes
queueing, prefill, decode, and the workload's concurrency shape.

## Run a campaign

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

`--host-id` is a stable, non-secret evidence identifier for the physical host.
It does not select a device or enable a kernel. `--output` must be under
`qualification/receipts/` or `.qualification/receipts/`, and the runner refuses
to replace an existing run or receipt.

Run `python3 scripts/qualification/run.py --help` for the current arguments.
Do not reuse command lines from historical receipts or older documentation.

## ROCm and Vulkan diagnostics

ROCm and Vulkan qualification drivers build for the available toolchain target.
They do not inject `gfx1151` or any other single architecture. Full-logit and
layer/path attribution drivers use:

- an offline private-network worker where supported;
- a start gate;
- a finite runtime;
- process-group termination and cleanup;
- exact model/request/source hashes; and
- numerical evidence with a self-hashed result.

The current evidence contracts are:

- [ROCm Hugging Face next-token result](../qualification/schema/rocm-hf-next-token-oracle-v2.schema.json);
- [ROCm/HF path attribution result](../qualification/schema/rocm-hf-path-attribution-v2.schema.json); and
- [ROCm/HF layer attribution result](../qualification/schema/rocm-hf-layer-attribution-v2.schema.json).

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

The [case-result schema](../qualification/schema/case-result-v1.schema.json)
defines one case outcome. The
[receipt schema](../qualification/schema/receipt-v1.schema.json) binds the
campaign as a whole.

Historical machine-specific receipts may remain for auditability. Current
product configuration, workload admission, and documentation must not depend
on them.
