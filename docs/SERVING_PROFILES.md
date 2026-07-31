# Serving profiles

A serving profile is an immutable, process-wide GPU ownership policy. It
controls inference admission, training writers, adapter weight transitions,
physical KV-cache changes, allocator reclaim, live graph capture, and Vulkan
resident prefill. Requests cannot select or override a profile.

`stable` is the default. Choose another profile only when its additional
operations are required:

| Profile | Use it for | Do not use it for |
|---|---|---|
| **stable** | Predictable base-model inference without live GPU mutations | Training, loading or unloading an adapter, or live memory and graph changes |
| **experimental** | Development and qualification that require inference plus training or adapter changes in one process | A latency or correctness baseline |
| **maintenance** | Drained training, adapter changes, and physical memory maintenance | Any inference or evaluation |

## Select a profile

Set the profile in TOML:

```toml
[server]
serving_profile = "stable"
```

Or override TOML for one process:

```bash
KILN_SERVER_SERVING_PROFILE=experimental kiln serve
```

Accepted values are exactly `stable`, `experimental`, and `maintenance`.
Matching is case-insensitive and ignores surrounding whitespace. An empty or
unknown value is a fatal startup error that identifies the invalid source and
value.

The retired `KILN_SERVING_PROFILE` spelling is ignored. Use
`KILN_SERVER_SERVING_PROFILE` or TOML, then run `kiln config --json` to verify
the resolved value and its source.

## Exact policy

| Effective policy | `stable` (default) | `experimental` | `maintenance` |
|---|---:|---:|---:|
| Inference admission | yes | yes | no |
| Training GPU ownership | no | yes | yes |
| Adapter weight transitions | no | yes | yes |
| Dynamic physical KV resize | no | yes | yes |
| Allocator reclaim | no | yes | yes |
| Live graph capture | no | yes | no |
| Vulkan resident prefill | no | yes | no |
| Exclusive GPU behavior | reject | writer priority | inference disabled; drain, then run exclusively |

The profile is one policy boundary, not a hardware selector. Kiln still
selects backend routes from runtime capabilities, tensor shapes, data types,
and correctness gates. Device marketing names, vendor or device IDs, driver
strings, benchmark host names, and qualification receipts never choose a
profile or a Vulkan kernel.

## Stable

Use `stable` for predictable inference. Logical batching, request
cancellation, eviction, and supported prefix-cache reuse remain available, but
the process rejects training GPU ownership and real adapter weight changes
before either can wait for the GPU writer. Dynamic physical KV resize,
allocator reclaim, and live graph capture are disabled.

Vulkan currently disables cross-request prefix reuse under a correctness
quarantine, so stable Vulkan requests use fresh prefill. Stable also keeps the
generic layer-resumable prompt path authoritative instead of enabling the
experimental resident-prefill route.

There is an important current limitation: Kiln starts a serving process with
the base model active and has no startup setting for selecting a saved adapter.
Loading or unloading an adapter is a live weight transition, which `stable`
rejects. A stable process therefore cannot currently serve or evaluate a saved,
unmerged adapter. Use `experimental` when adapter inference is required. This
is a product limitation, not a hidden configuration option.

## Experimental

Use `experimental` for controlled development and qualification that require
inference and GPU-writer operations in the same process. Training, adapter
transitions, dynamic KV changes, allocator reclaim, and live graph capture are
allowed. Writer-priority work can pause inference, and memory or graph changes
can alter latency, so this profile is not the stable performance baseline.

On Vulkan, `experimental` permits resident token prefill only after the
backend's capability and request-shape checks pass. The route batches one
prompt token per active row and retains KV and recurrent state on the Vulkan
device. Admission has no device-name, host-name, or receipt-based exception.
Portable defaults remain portable; hardware-specific experiments and results
belong in the [benchmark report](public/BENCHMARKS.md) and
[serving benchmark protocol](SERVING_BENCHMARK_PROTOCOL.md), not in this
profile's product policy.

## Maintenance

`maintenance` is a drained work mode, not a serving mode. Inference prewarm is
skipped. `/health` and `/v1/health` return HTTP 503 with
`status: "maintenance"`. Completion, prompt-logprob, eval, and agent-run
admission return HTTP 503 with `code: "inference_disabled_by_profile"`.

Training, real adapter transitions, dynamic KV changes, and allocator reclaim
remain available. Live graph capture stays disabled because no inference
request should be present to justify a serving graph.

## Move between profiles

A profile cannot be changed through an API request or reloaded in place. Use a
restart as the ownership boundary:

1. Remove the current instance from traffic and stop it gracefully.
2. Start a new process with the required profile.
3. Run only operations admitted by that profile.
4. Stop the process, start the next profile, and wait for `/health` before
   restoring traffic.

For drained training, use `maintenance` while the writer owns the GPU. The
result remains on disk, but `maintenance` cannot evaluate it and `stable`
cannot load it. Start a separate `experimental` process to load, evaluate, or
serve the unmerged adapter. If you need generation, training, and evaluation
in one process, use `experimental` for the entire controlled loop.

## Observe the resolved policy

At startup, Kiln logs the selected profile, its source, its immutability, and
every effective policy field. `/health`, `/v1/health`, and `/v1/config` expose
the same `serving_profile` object:

```json
{
  "profile": "stable",
  "source": "default",
  "immutable_after_startup": true,
  "request_overrides_allowed": false,
  "effective_policy_source": "serving_profile",
  "effective_policy": {
    "inference_admission": true,
    "training_gpu_ownership": false,
    "adapter_weight_transitions": false,
    "dynamic_kv_resize": false,
    "allocator_reclaim": false,
    "live_graph_capture": false,
    "vulkan_resident_prefill": false,
    "exclusive_gpu_behavior": "reject"
  }
}
```

`source` is `default`, `config_file`, or `environment`. Request selection is
never a source. In stable mode,
`decode_runtime.memory_governor.reclaim_mode` is `"off"` even when
`requested_reclaim_mode` contains another configured value, and
`disabled_by_serving_profile` is `true`.

Backend health and readiness are separate. A healthy maintenance process can
report `backend_runtime.healthy: true` while its `inference_admission` check is
false and its overall HTTP status is 503. A quarantined backend reports
`backend_runtime.healthy: false` and requires a restart.
