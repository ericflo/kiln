# Serving profiles

A serving profile is an immutable, process-wide GPU ownership policy. Normal
users do not need to choose one: `stable` is the default and supports
inference, training, adapter transitions, guarded memory management, and live
graph capture. Requests cannot select or override a profile.

Choose another profile only for backend development or drained maintenance:

| Profile | Use it for | Do not use it for |
|---|---|---|
| **stable** | Normal inference, training, adapters, evaluation, and automatic performance paths | Backend routes that are still under an explicit correctness quarantine |
| **experimental** | Backend development and qualification of quarantined routes | Ordinary use; it is never required for normal speed or training |
| **maintenance** | Drained training, adapter changes, and physical memory maintenance | Any inference or evaluation |

## Select a profile

Set the profile in TOML:

```toml
[server]
serving_profile = "stable"
```

Or override TOML for one process:

```bash
KILN_SERVER_SERVING_PROFILE=maintenance kiln serve
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
| Training GPU ownership | yes | yes | yes |
| Adapter weight transitions | yes | yes | yes |
| Dynamic physical KV resize | yes | yes | yes |
| Allocator reclaim | yes | yes | yes |
| Live graph capture | yes | yes | no |
| Vulkan resident prefill | no | yes | no |
| Exclusive GPU behavior | writer priority | writer priority | inference disabled; drain, then run exclusively |

The profile is one policy boundary, not a hardware selector. Kiln still
selects backend routes from runtime capabilities, tensor shapes, data types,
and correctness gates. Device marketing names, vendor or device IDs, driver
strings, benchmark host names, and qualification receipts never choose a
profile or a Vulkan kernel.

## Stable

Use `stable` for normal product operation. Logical batching, request
cancellation, eviction, supported prefix-cache reuse, training, real adapter
weight changes, dynamic physical KV resize, allocator reclaim, and guarded live
graph capture are available. Writer-priority ownership serializes accelerator
mutations against inference instead of making the user restart into a special
mode.

Vulkan currently disables cross-request prefix reuse under a correctness
quarantine, so stable Vulkan requests use fresh prefill. Stable also keeps the
generic layer-resumable prompt path authoritative instead of enabling the
experimental resident-prefill route.

## Experimental

`experimental` is a backend-development profile, not a faster product mode.
It has the same inference/training/adapter/memory/graph ownership contract as
`stable`; its only purpose is admitting a route that remains under an explicit
correctness quarantine. Ordinary users should not select it.

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

For ordinary training, evaluation, and adapter serving, remain on `stable`.
Use `maintenance` only when inference must be fully drained while a writer or
physical-memory operation runs exclusively.

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
    "training_gpu_ownership": true,
    "adapter_weight_transitions": true,
    "dynamic_kv_resize": true,
    "allocator_reclaim": true,
    "live_graph_capture": true,
    "vulkan_resident_prefill": false,
    "exclusive_gpu_behavior": "writer_priority"
  }
}
```

`source` is `default`, `config_file`, or `environment`. Request selection is
never a source.

Backend health and readiness are separate. A healthy maintenance process can
report `backend_runtime.healthy: true` while its `inference_admission` check is
false and its overall HTTP status is 503. A quarantined backend reports
`backend_runtime.healthy: false` and requires a restart.
