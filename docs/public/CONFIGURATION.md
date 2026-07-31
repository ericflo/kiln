# Configuration Reference

Most Kiln deployments need only a model path, a listener, an accelerator build,
and an adapter directory. Start with those. Add scheduling, memory, training,
or observability policy only when you can name the behavior you are changing.

## Minimal server

```toml
[server]
host = "127.0.0.1"
port = 8420

[model]
path = "/models/Qwen3.5-4B"
adapter_dir = "/var/lib/kiln/adapters"
```

Run `kiln config` before starting a long-lived service. It resolves defaults,
validates cross-field rules, and reports the effective backend policy without
loading the model.

## The settings people change first

| Goal | Setting or command | Guidance |
|---|---|---|
| Listen beyond localhost | `server.host` | Add authentication and network controls before exposing the port. |
| Select a Vulkan GPU | `accelerator.vulkan_device_index` | Use `auto` or an explicit zero-based Vulkan index. The current memory-probe identity gate still admits only index zero on a single-GPU host; verify the selected device in `/health`. |
| Move adapter storage | `model.adapter_dir` | Use durable local storage and preserve manifests with adapter files. |
| Bound request work | request `max_tokens`, thinking budgets, server batch budgets | Set explicit client limits before tuning throughput. |
| Change serving behavior | `serving_profile` and batching fields | Measure TTFT, inter-token latency, and request-window throughput separately. |
| Enable diagnostics | observability and protected debug settings | Treat debug endpoints as operational data, not public APIs. |

## Configuration precedence

Kiln reports both the effective value and its source. The exact precedence and
the canonical environment-variable names are defined in the full reference;
do not create aliases by guessing an uppercase name.

After startup, inspect the result:

```bash
curl -fsS http://127.0.0.1:8420/v1/config | jq
```

When debugging, capture this response with the server version, source revision,
backend, device, and driver. A TOML snippet alone does not prove the runtime
used it.

## Accelerator policy

Builds select one accelerator implementation: CUDA, ROCm, Metal, or Vulkan.
Backend kernel policy is immutable for the process lifetime. Restart after a
policy change, and verify the resolved policy schema in `/v1/config`.

Portable does not mean “turn every optimized route off.” A fast path should be
admitted from the API and hardware capabilities it requires, with a narrow
fallback when that capability is absent. Device marketing names and
qualification-machine identities are not configuration.

## Safety defaults

- Keep the listener on loopback unless you have an explicit exposure plan.
- Use explicit output and thinking limits for untrusted requests.
- Put adapters, checkpoints, and receipts on durable storage.
- Do not weaken model, executable, or artifact identity checks to make a
  benchmark pass.
- Do not compare performance runs that changed workload, source, driver, or
  metric definition.

## Exact references

This page is an operational guide, not a duplicate of generated contracts.

- [Complete typed configuration reference](https://ericflo.github.io/kiln/docs/configuration-complete/)
- [Machine-readable configuration schema](https://github.com/ericflo/kiln/blob/main/contracts/kiln-config-v1.schema.json)
- [Serving profiles](../SERVING_PROFILES.md)
- [Troubleshooting guide](https://ericflo.github.io/kiln/troubleshooting.html)
