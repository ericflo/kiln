# Serving Profiles

Kiln resolves one serving profile at startup and keeps it immutable for the
life of the process. The profile is the GPU ownership contract: it decides
whether inference, training writers, adapter weight transitions, physical KV
cache changes, allocator reclaim, and live graph capture may run.

`stable` is the default. Select a different profile only for a deliberate
development or drained-maintenance session:

```toml
[server]
serving_profile = "stable"
```

`KILN_SERVER_SERVING_PROFILE` overrides the TOML value. Accepted values are exactly
`stable`, `experimental`, and `maintenance` (case-insensitive, with surrounding
whitespace ignored). An empty or unknown value is a fatal startup error that
names the invalid field or environment variable and value.

The older `KILN_SERVING_PROFILE` spelling remains a deprecated compatibility
alias and emits a startup warning. Do not use it in new deployments.

## Policy matrix

| Effective policy | `stable` (default) | `experimental` | `maintenance` |
|---|---:|---:|---:|
| Inference admission | yes | yes | no |
| Training GPU ownership | no | yes | yes |
| Adapter weight transitions | no | yes | yes |
| Dynamic physical KV resize | no | yes | yes |
| Allocator reclaim | no | yes | yes |
| Live graph capture | no | yes | no |
| Vulkan resident token prefill | no | yes | no |
| Exclusive GPU behavior | reject | writer priority | inference disabled, drain, then exclusive |

The stable profile keeps ordinary logical scheduling available. On a backend
that admits cross-request prefix reuse, requests may be admitted, cancelled,
evicted, and served from the prefix cache without moving live allocation
pointers. Vulkan currently forces that effective capability off under a
correctness quarantine and fresh-prefills every request. The profile rejects
training ownership and real adapter weight changes before either can wait for
the actor-wide GPU writer.
A request that selects the adapter already active is a no-op and remains
allowed.

The experimental profile retains dynamic behavior for controlled development
and comparison work. It is not the supported latency or correctness baseline.
Writer-priority operations may interrupt inference progress, and live graph
capture or physical memory changes may alter latency. On Vulkan, experimental
also admits native resident token prefill after the backend and request-level
eligibility checks pass. This route batches one prompt token per active row and
retains its KV and recurrent state on the Vulkan device. Stable serving keeps
the generic layer-resumable prompt path authoritative until the resident route
passes the final repeated-cohort and soak gates. There is no per-request or
environment override for this policy.

The maintenance profile is intentionally not a serving profile. Inference
prewarm is skipped, `/health` and `/v1/health` return HTTP 503 with
`status: "maintenance"`, and completion, prompt-logprob, eval, and agent-run
admission return HTTP 503 with `code: "inference_disabled_by_profile"`.
Training, real adapter transitions, and physical memory maintenance remain
available. Live graph capture stays disabled because no request should be
present to justify a new serving graph.

## Entering maintenance

The profile cannot be changed by an API request or reloaded in place. Use a
restart as the ownership boundary:

1. Remove the stable instance from traffic and stop it gracefully. Wait for
   admitted requests to finish within the configured shutdown timeout.
2. Start a new process with `KILN_SERVER_SERVING_PROFILE=maintenance`. Keep it out of
   the serving pool; its 503 health response is an additional readiness guard.
3. Run the training, adapter activation, or physical memory operation that
   requires exclusive GPU ownership.
4. Stop the maintenance process and restart with
   `KILN_SERVER_SERVING_PROFILE=stable` (or remove the override).
5. Wait for `/health` to return 200 before restoring traffic.

Post-training evaluation requires inference. Do not schedule it inside the
maintenance process; run it after the stable restart. Use `experimental` only
when the experiment specifically requires inference and GPU-writer transitions
inside one process.

## Observability

At startup, Kiln logs the selected profile, its source, immutability, request
override policy, and every effective policy field. The same
`serving_profile` object is returned by `/health`, `/v1/health`, and
`/v1/config`:

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
never a source because request overrides are prohibited. Under stable mode,
`decode_runtime.memory_governor` reports `reclaim_mode: "off"` even if
`requested_reclaim_mode` was configured differently, and sets
`disabled_by_serving_profile: true`. This distinction prevents requested
configuration from being mistaken for behavior that is actually running.

Backend health is independent of readiness. A healthy maintenance process
reports `backend_runtime.healthy: true` while its `inference_admission` health
check is false and its overall HTTP status remains 503. A quarantined backend
continues to report `backend_runtime.healthy: false` and requires a restart.
