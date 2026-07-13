# Configuration Reference

This document is the canonical operator reference for Kiln's typed server
configuration. It describes the configuration accepted by the current
`KilnConfig` and `RequestLogConfig` implementations, including current defaults,
validation, working environment-variable aliases, and runtime provenance.

Kiln's intended public environment-variable naming rule is mechanical:

```text
KILN_<SECTION>_<FIELD>
```

Both the section and field are converted to uppercase snake case. For example,
`server.host` maps to the canonical target `KILN_SERVER_HOST`, and
`request_log.max_file_bytes` maps to
`KILN_REQUEST_LOG_MAX_FILE_BYTES`.

Every fixed field with an environment override implements its canonical name.
The "Working spelling(s) today" column records either that same canonical name
or a temporary compatibility spelling:

- **implemented** means the mechanically derived canonical name works;
- **deprecated compatibility** means a non-canonical alias still works but
  emits a structured startup warning naming the field, alias, and replacement;
- **none** means there is no supported environment override for that field;
- **target only; not implemented** means setting the mechanically derived name
  currently has no effect because the field is intentionally config-file-only.

When a canonical name and deprecated compatibility alias are both present,
Kiln parses both strictly. They are accepted only when they resolve to the same
typed value; a conflict stops startup and names both variables. Values are
never included in compatibility warnings.

Do not assume that an arbitrary `KILN_*` name is public configuration. The
repository contains internal diagnostics, qualification switches, kernel
experiments, and temporary compatibility flags. Only the canonical names and
deprecated compatibility spellings listed here are supported inputs.

## Resolution and startup

### Config file selection

Kiln selects at most one TOML file, in this order:

1. The explicit path supplied by the command, such as
   `kiln --config /path/kiln.toml serve`.
2. The path in `KILN_CONFIG`.
3. `./kiln.toml`, if it exists in the process working directory.
4. No file; built-in defaults are used.

An explicitly selected file that cannot be read or parsed is a startup error.
`KILN_CONFIG` selects the file; it is not an override for a TOML field and is
not included in the field counts below.

### Value precedence

For fields with a working environment alias, effective startup values resolve
in this order, highest priority first:

1. Typed CLI overrides applied after environment resolution
   (`serve --served-model-id` and `serve --eval-mode`).
2. A canonical environment name or deprecated compatibility alias listed here.
3. The selected TOML file.
4. The built-in default.

The `-v`/`-vv` and `--quiet` logging flags override `logging.level` for the
running command. `RUST_LOG`, when present, takes precedence over that resolved
level when the tracing filter is built.

Environment resolution happens once at startup. Restart the server after
changing a file or environment value. Model, training, and request execution
receive resolved policy rather than re-reading public configuration from the
process environment.

### Strict failure behavior

The top-level document and every typed section reject unknown fields. This
includes `[request_log]`, `[eval]`, `[agent]`, `[teachers]`, and each dynamic
teacher credential. A misspelled key does not fall back to a default. The one
intentional open subtree is `agent.self_improve`, whose structured request body
is validated later by the self-improvement subsystem.

TOML type errors, invalid enum values, malformed URLs, out-of-range values, and
semantically invalid combinations stop startup. Present centralized environment
overrides must be valid UTF-8 and must parse successfully; malformed values stop
startup and the diagnostic names the variable and raw value. The centralized
loader's boolean aliases accept, case-insensitively and with surrounding
whitespace ignored:

```text
true:  true, 1, yes, on
false: false, 0, no, off
```

Optional TOML values are represented by omitting the field; TOML has no `null`
literal. Environment-specific clearing syntax is documented per field.

Validate without starting the server:

```bash
kiln config --file kiln.toml
```

This command uses the same typed loader and environment precedence as serving,
but its pretty output is a summary rather than a complete effective-config
dump.

## Coverage summary

The accepted TOML surface contains 14 top-level sections and 82 fixed leaf
fields. Dynamic `teachers.credentials.<id>` entries add two leaf fields per
credential. Of the 82 fixed fields:

- 74 implement the canonical mechanical environment name;
- 52 also retain one or more deprecated compatibility spellings (54 aliases
  total);
- 8 are config-file-only and have no environment override;
- the 54 aliases include `KILN_DEFAULT_NO_THINK`, the second deprecated
  compatibility spelling for `server.default_thinking_enabled`.

The tables below cover all 82 fixed fields and both dynamic credential fields.

## `[server]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `server.serving_profile` | string enum; `"stable"` | `KILN_SERVER_SERVING_PROFILE` (implemented) | `KILN_SERVING_PROFILE` (deprecated compatibility) | `stable`, `experimental`, or `maintenance`, case-insensitive. Process-lifetime policy; restart required. |
| `server.deterministic` | boolean; `false` | `KILN_SERVER_DETERMINISTIC` (implemented) | `KILN_DETERMINISTIC` (deprecated compatibility) | Enables deterministic tensor behavior and forces the effective concurrent decode width to one. |
| `server.host` | string; `"127.0.0.1"` | `KILN_SERVER_HOST` (implemented) | `KILN_HOST` (deprecated compatibility) | Must be non-empty. Binding beyond loopback exposes an unauthenticated inference and training API; use a trusted network or authenticated reverse proxy. |
| `server.port` | unsigned 16-bit integer; `8420` | `KILN_SERVER_PORT` (implemented) | `KILN_PORT` (deprecated compatibility) | `1..=65535`. |
| `server.request_timeout_secs` | unsigned integer; `600` | `KILN_SERVER_REQUEST_TIMEOUT_SECS` (implemented) | `KILN_REQUEST_TIMEOUT_SECS` (deprecated compatibility) | Must be greater than zero. Bounds a request, including model work and cleanup settlement. |
| `server.http_send_buffer_bytes` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_HTTP_SEND_BUFFER_BYTES` (implemented) | `KILN_HTTP_SEND_BUFFER_BYTES` (deprecated compatibility) | When set, `1024..=16777216`. Applied to accepted sockets. Startup preflights the listener and rejects an OS read-back smaller than requested. |
| `server.stream_stall_grace_ms` | unsigned integer; `2000` | `KILN_SERVER_STREAM_STALL_GRACE_MS` (implemented) | `KILN_STREAM_STALL_GRACE_MS` (deprecated compatibility) | `10..=2000`. A request retaining KV state with no streaming delivery progress is selected for cancellation after this grace. |
| `server.max_batch_tokens` | unsigned integer; `512` | `KILN_SERVER_MAX_BATCH_TOKENS` (implemented) | `KILN_MAX_BATCH_TOKENS` (deprecated compatibility) | `2..=65536`. Combined decode-plus-prefill token budget for one batching-actor cycle. |
| `server.max_prefill_tokens_per_cycle` | unsigned integer; `64` | `KILN_SERVER_MAX_PREFILL_TOKENS_PER_CYCLE` (implemented) | `KILN_MAX_PREFILL_TOKENS_PER_CYCLE` (deprecated compatibility) | `1..=65536`. Independent new-prompt-token ceiling within the combined actor budget. |
| `server.max_prefill_layers_per_cycle` | unsigned integer; `4` | `KILN_SERVER_MAX_PREFILL_LAYERS_PER_CYCLE` (implemented) | `KILN_MAX_PREFILL_LAYERS_PER_CYCLE` (deprecated compatibility) | `1..=1024`. Number of transformer layers an in-flight prefill chunk may execute before yielding to decode. |
| `server.max_decode_batch` | `"auto"` or unsigned integer; `"auto"` | `KILN_SERVER_MAX_DECODE_BATCH` (implemented) | `KILN_MAX_DECODE_BATCH` (deprecated compatibility) | `auto`, `backend`, and `backend_policy` all delegate to backend policy; an integer must be `1..=65536`. Deterministic mode and `max_batch_tokens` may lower the final width. |
| `server.eval_mode` | boolean; `false` | `KILN_SERVER_EVAL_MODE` (implemented) | `KILN_EVAL_MODE` (deprecated compatibility) | Enables deterministic eval-serving defaults, headers, adapter warnings, and transient-cache cleanup behavior. `serve --eval-mode` applies a typed override after environment resolution and wins without mutating process environment. |
| `server.default_thinking_enabled` | optional boolean; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_ENABLED` (implemented) | `KILN_DEFAULT_THINKING_ENABLED` (deprecated compatibility); `KILN_DEFAULT_NO_THINK` is a deprecated presence-only alias | `None` preserves the model template default. Requests may override with `chat_template_kwargs.enable_thinking`. Without the canonical spelling, any presence of `KILN_DEFAULT_NO_THINK`, even a value such as `0`, first selects `false`; the explicit `KILN_DEFAULT_THINKING_ENABLED` value is applied afterward and wins. When the canonical spelling is present, every present compatibility alias must resolve to the same boolean. There is no environment spelling that restores `None`. |
| `server.default_thinking_budget_tokens` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_BUDGET_TOKENS` (implemented) | `KILN_DEFAULT_THINKING_BUDGET_TOKENS` (deprecated compatibility) | Integer values include `0`, which closes thinking immediately. Case-insensitive `unlimited` clears a TOML limit back to `None`. Requests may inherit, replace, or explicitly disable the limit. |
| `server.default_thinking_budget_ms` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_BUDGET_MS` (implemented) | `KILN_DEFAULT_THINKING_BUDGET_MS` (deprecated compatibility) | Integer values include `0`. `unlimited` clears a TOML limit. The clock starts at the first decode candidate, after queueing and prefill. The first token or time limit reached forces the model's closing `</think>` sequence. |
| `server.fold_reasoning_into_content` | boolean; `false` | `KILN_SERVER_FOLD_REASONING_INTO_CONTENT` (implemented) | `KILN_FOLD_REASONING_INTO_CONTENT` (deprecated compatibility) | Copies separated reasoning into `content` for compatibility clients. A request can override it. |
| `server.chat_performance_metadata` | boolean; `false` | `KILN_SERVER_CHAT_PERFORMANCE_METADATA` (implemented) | `KILN_CHAT_PERFORMANCE_METADATA` (deprecated compatibility) | Default for chat response performance metadata; requests can override with `include_performance`. |
| `server.chat_config_hash_metadata` | boolean; `false` | `KILN_SERVER_CHAT_CONFIG_HASH_METADATA` (implemented) | `KILN_CHAT_CONFIG_HASH_METADATA` (deprecated compatibility) | Default for chat response config hashes; requests can override with `include_config_hashes`. |
| `server.slow_request_warn_secs` | unsigned integer; `30` | `KILN_SERVER_SLOW_REQUEST_WARN_SECS` (implemented) | `KILN_SLOW_REQUEST_WARN_SECS` (deprecated compatibility) | `0` disables slow-request warnings; otherwise a request at least this old emits a structured warning. |
| `server.shutdown_timeout_secs` | unsigned integer; `5` | `KILN_SERVER_SHUTDOWN_TIMEOUT_SECS` (implemented) | `KILN_SHUTDOWN_TIMEOUT_SECS` (deprecated compatibility) | Must be greater than zero. Hard ceiling for graceful drain before forced exit. |

### Serving profiles

The profile owns GPU mutation and admission policy; individual requests cannot
override it.

| Profile | Inference | Training GPU ownership | Adapter weight transitions | Dynamic KV resize / allocator reclaim | Live graph capture | Exclusive behavior |
|---|---|---|---|---|---|---|
| `stable` | admitted | rejected | rejected | disabled | disabled | reject |
| `experimental` | admitted | allowed | allowed | enabled | enabled | writer priority |
| `maintenance` | not admitted | allowed | allowed | enabled | disabled | drain, then exclusive |

`memory.cuda_graphs = true` does not override the profile: the default `stable`
profile selects eager-only model-runner options because live graph capture is
disabled.

## `[batching]`

These values are resolved once, after backend selection and after
`server.deterministic`, `server.max_decode_batch`, and
`server.max_batch_tokens` determine the effective decode width. Every change
requires a process restart; none is a live tuning control.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `batching.mode` | string enum; `"auto"` | `KILN_BATCHING_MODE` (implemented) | `KILN_BATCHING_ENGINE` (deprecated compatibility) | TOML accepts `auto`, `enabled`, or `disabled`, case-insensitively. Environment input also accepts the strict boolean spellings, mapping true to `enabled` and false to `disabled`. `auto` delegates actor selection to the immutable backend policy; an explicit value wins. Restart required. |
| `batching.rowwise_decode` | boolean; `false` | `KILN_BATCHING_ROWWISE_DECODE` (implemented) | `KILN_BATCH_DECODE_ROWWISE` (deprecated compatibility) | `false` sends each ready cohort through one true batched forward. `true` issues one forward per row while retaining actor ownership; it is an emergency correctness comparison, normally reduces throughput, and does not increase the effective decode width. Restart required. |
| `batching.prefix_aware_admission` | boolean; `true` | `KILN_BATCHING_PREFIX_AWARE_ADMISSION` (implemented) | `KILN_BATCH_PREFIX_AWARE_ADMISSION` (deprecated compatibility) | When true, a queued same-adapter strict descendant waits while its active shorter prefix can become reusable; independent rows may still be admitted. Disable only for a controlled admission A/B. Restart required. |
| `batching.prefill_admission_quantum` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_PREFILL_ADMISSION_QUANTUM` (implemented) | `KILN_BATCH_PREFILL_ADMISSION_QUANTUM` (deprecated compatibility) | An integer must be `1..=65536` and caps how many queued prompts the actor admits in one cycle before returning to decode. `auto` is case-insensitive and uses the backend policy below. The selected value is then clamped to `1..=effective max_decode_batch`; the diagnostics name `effective_decode_width` as final authority when it performs that clamp. Restart required. |
| `batching.direct_decode_rendezvous_mode` | string enum; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MODE` (implemented) | `KILN_DECODE_BATCHER` (deprecated compatibility) | Selects only the fallback worker used by actor-absent direct streaming effectively-greedy requests. TOML accepts `auto`, `enabled`, or `disabled`, case-insensitively; environment input also accepts the strict boolean spellings. `auto` enables the worker on every real backend. Sampled requests, non-streaming requests, and every route using the batching actor bypass this worker. Restart required. |
| `batching.direct_decode_rendezvous_max_batch` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MAX_BATCH` (implemented) | `KILN_DECODE_BATCH_MAX` (deprecated compatibility) | An explicit integer must be `1..=65536`. `auto` uses the backend policy below. Either selection is clamped to the already-resolved effective decode width; diagnostics use `effective_decode_width` when that ceiling wins. Restart required. |
| `batching.direct_decode_rendezvous_wait_us` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_WAIT_US` (implemented) | `KILN_DECODE_BATCH_WAIT_US` (deprecated compatibility) | An explicit value is any non-negative `u64` number of microseconds, including `0`; `auto` uses backend policy. A negative, overflowing, malformed, or non-UTF-8 environment value stops startup. In particular, malformed legacy `KILN_DECODE_BATCH_WAIT_US` now fails startup instead of silently becoming zero. Restart required. |
| `batching.direct_decode_rendezvous_mixed_seq_lens` | `"auto"` or boolean; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MIXED_SEQ_LENS` (implemented) | `KILN_DECODE_BATCH_MIXED_SEQ` (deprecated compatibility) | TOML accepts only the string `auto` or a native boolean; quoted boolean strings are rejected. Environment input accepts `auto` or the strict boolean spellings. The value controls whether one compatible fallback cohort may contain different decode positions. Restart required. |

### Backend-owned `auto` policy

`auto` preserves backend qualification rather than imposing one cross-device
default. The matrix is evaluated against the final effective decode width, not
the unconstrained backend maximum.

| Backend | `batching.mode = "auto"` | Backend decode width before server constraints | `prefill_admission_quantum = "auto"` | `burst_prefill_admission` diagnostic |
|---|---:|---:|---:|---:|
| CUDA | enabled | 8 | effective decode width | true |
| ROCm | enabled | 8 | 4, clamped to effective width | false |
| Vulkan | enabled | 64 | effective decode width | false |
| Metal | disabled | 8 | 4, clamped to effective width | false |
| CPU/other | enabled | 8 | 4, clamped to effective width | false |

`burst_prefill_admission` is backend-owned and intentionally has no TOML or
public environment field. On CUDA it lets admission refill the decode width in
one actor turn. It is reported with the four primary-actor settings so an observed
CUDA/Vulkan/ROCm difference is not mistaken for a hidden runtime override.

The fallback direct-stream rendezvous has a separate backend-owned `auto`
matrix. Its mode is enabled on every real backend, but a live worker is not the
same as an available request route:

| Backend | `direct_decode_rendezvous_mode = "auto"` | `max_batch = "auto"` | `wait_us = "auto"` | `mixed_seq_lens = "auto"` |
|---|---:|---:|---:|---:|
| CPU/other | enabled | 8 | 0 | false |
| CUDA | enabled | 1 | 0 | false |
| ROCm | enabled | 8 | 0 | false |
| Metal | enabled | 8 | 100 | true |
| Vulkan | enabled | 64 | 5000 | true |

The selected `max_batch` is always clamped to the effective decode width. The
worker is constructed independently of the primary actor so startup and worker
failures remain observable, but its route is available only for a real backend
when the actor is absent and the worker is active. An actor-active process can
therefore legitimately report `worker_active=true` and
`route_available=false`. Under the complete default policy, this makes the
route available only on Metal: every real backend auto-enables the worker, but
Metal is the only backend whose primary actor is auto-disabled. Explicitly
disabling the actor can make the route available on another real backend.

### Effective policy and provenance

`GET /v1/config` reports the exact startup result at
`batching.configuration` and whether an actor was actually constructed at
`batching.actor_active`. A typical ROCm default is:

```json
{
  "batching": {
    "configuration": {
      "mode": {
        "configured": "auto",
        "configured_source": "default",
        "backend_policy_enabled": true,
        "effective_enabled": true,
        "effective_source": "backend_policy"
      },
      "rowwise_decode": { "enabled": false, "source": "default" },
      "prefix_aware_admission": { "enabled": true, "source": "default" },
      "prefill_admission_quantum": {
        "configured": null,
        "configured_source": "default",
        "backend_policy": 4,
        "effective": 4,
        "effective_source": "backend_policy"
      },
      "direct_decode_rendezvous": {
        "mode": {
          "configured": "auto",
          "configured_source": "default",
          "backend_policy_enabled": true,
          "effective_enabled": true,
          "effective_source": "backend_policy"
        },
        "max_batch": {
          "configured": null,
          "configured_source": "default",
          "backend_policy": 8,
          "effective": 8,
          "effective_source": "backend_policy"
        },
        "wait_us": {
          "configured": null,
          "configured_source": "default",
          "backend_policy": 0,
          "effective": 0,
          "effective_source": "backend_policy"
        },
        "mixed_seq_lens": {
          "configured": null,
          "configured_source": "default",
          "backend_policy": false,
          "effective": false,
          "effective_source": "backend_policy"
        }
      },
      "burst_prefill_admission": false
    },
    "actor_active": true,
    "direct_decode_rendezvous": {
      "scope": "direct_streaming_greedy_only",
      "backend_available": true,
      "actor_active": true,
      "worker_active": true,
      "route_available": false
    }
  }
}
```

For explicit values, the configured source is `config_file` or `environment`.
The complete effective-source enum is `default`, `backend_policy`,
`config_file`, `environment`, or `effective_decode_width`. The mode normally
uses the middle three; `default` is reachable only for an explicit
programmatic value carrying default provenance, not for ordinary built-in
`auto`, which resolves through `backend_policy`. The admission quantum uses
`effective_decode_width` when the final decode width lowers the selected value.
A configured quantum or direct-rendezvous width is retained even when clamped,
so intent and execution cannot be confused. The direct policy is nested at
`batching.configuration.direct_decode_rendezvous`; the sibling
`batching.direct_decode_rendezvous` is actual process state, with exact fields
`scope`, `backend_available`, optional `backend_unavailable_reason`,
`actor_active`, `worker_active`, and `route_available`. Mock mode reports
`backend_available=false` and `backend_unavailable_reason="mock_backend"`.

`GET /health` repeats the same immutable object at
`decode_runtime.batching_configuration` beside the optional live
`decode_runtime.batching_engine` snapshot and reports actual fallback state at
`decode_runtime.direct_decode_rendezvous`. The trusted debug response keeps the
existing `batching_engine.enabled` actor-activity flag, adds the immutable
object at `batching_engine.configuration`, and reports actual fallback state at
`batching_engine.direct_decode_rendezvous`. Actor activity can be false even
when configured intent is visible, and worker activity does not prove routing;
clients must use `route_available` for this deliberately narrow direct route.

`kiln config --file <path>` validates and prints all eight `[batching]` startup
values: actor mode, rowwise decode, prefix-aware admission, prefill quantum,
direct rendezvous mode, maximum batch, wait microseconds, and mixed-sequence
policy. It cannot report backend-effective or live route state because it does
not construct a model; inspect `/v1/config` after restart for those facts.

## `[model]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `model.path` | optional string; omitted (`None`) | `KILN_MODEL_PATH` (implemented) | `KILN_MODEL_PATH` | Must be non-empty when set. Omitted starts the server in mock mode; a real model path enables real inference. |
| `model.model_id` | string; `"Qwen/Qwen3.5-4B"` | `KILN_MODEL_MODEL_ID` (implemented) | `KILN_MODEL_ID` (deprecated compatibility) | Must be non-empty. Used for model/tokenizer identity and as the source of the default served id. Kiln still applies its built-in Qwen3.5-4B runtime profile. |
| `model.tokenizer_path` | optional string; omitted (`None`) | `KILN_MODEL_TOKENIZER_PATH` (implemented) | `KILN_TOKENIZER_PATH` (deprecated compatibility) | Must be non-empty when set. |
| `model.adapter_dir` | optional string; omitted (`None`) | `KILN_MODEL_ADAPTER_DIR` (implemented) | `KILN_ADAPTER_DIR` (deprecated compatibility) | Must be non-empty when set. For the Qwen3.5-4B profile, omission resolves to `<model.path>/adapters`. |
| `model.snapshot_dir` | optional string; omitted (`None`) | `KILN_MODEL_SNAPSHOT_DIR` (implemented) | `KILN_MODEL_SNAPSHOT_DIR` | Must be non-empty when set. The environment alias uniquely treats an empty or whitespace-only value as a request to clear the TOML value. Without a value, Kiln tries a location beside the model and then the system temporary directory. |
| `model.served_model_id` | optional string; omitted (`None`) | `KILN_MODEL_SERVED_MODEL_ID` (implemented) | `KILN_SERVED_MODEL_ID` (deprecated compatibility) | Must be non-empty when set. Otherwise the effective id is the final slash-separated component of `model.model_id` (`Qwen3.5-4B` by default). `serve --served-model-id` applies a typed, validated override after environment resolution. |

## `[memory]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `memory.num_blocks` | optional unsigned integer; omitted (`None`) | `KILN_MEMORY_NUM_BLOCKS` (implemented) | `KILN_NUM_BLOCKS` (deprecated compatibility) | Must be greater than zero when set. Omission invokes backend-aware automatic KV-block sizing. |
| `memory.gpu_memory_gb` | optional finite number; omitted (`None`) | `KILN_MEMORY_GPU_MEMORY_GB` (implemented) | `KILN_GPU_MEMORY_GB` (deprecated compatibility) | Must be finite, greater than zero, and representable as bytes. Units are GiB. This is a capacity cap, not a hardware override: it may reduce the detected safe capacity but never expands physical VRAM, host-backed unified memory, or a cgroup-bounded capacity. A request above the safe detected capacity is clamped down. |
| `memory.inference_memory_fraction` | finite number; `0.7` | `KILN_MEMORY_INFERENCE_MEMORY_FRACTION` (implemented) | `KILN_INFERENCE_MEMORY_FRACTION` (deprecated compatibility) | Loader validation accepts `0.0..=1.0`; real-state construction clamps the configured value to `0.1..=1.0` before KV sizing. The remainder is available to the training budget. |
| `memory.training_memory_gb` | optional finite number; omitted (`None`) | `KILN_MEMORY_TRAINING_MEMORY_GB` (implemented) | `KILN_TRAINING_MEMORY_GB` (deprecated compatibility) | Must be finite, greater than zero, and representable as bytes when set. Optional training-budget cap in GiB; it can reduce but never expand the capacity remaining after resident model and KV allocations. |
| `memory.floor_gb` | finite number; `1.0` | `KILN_MEMORY_FLOOR_GB` (implemented) | `KILN_MEMORY_FLOOR_GB` | Must be finite, non-negative, representable as bytes, and strictly smaller than the selected accelerator's effective capacity after `memory.gpu_memory_gb` is applied. Units are GiB. Accelerator startup rejects an equal or larger floor before model upload and reports both configured and effective byte values. The process-wide governor subtracts this additional floor, then outstanding soft reservations, from live free memory when computing allocation headroom. On unified-memory devices it is separate from the physical-memory reserve applied during safe-capacity detection. |
| `memory.probe_ms` | unsigned integer; `500` | `KILN_MEMORY_PROBE_MS` (implemented) | `KILN_MEMORY_PROBE_MS` | Must be greater than zero. Sets the background memory-sampler cadence. Request, inference, health, and metrics paths read only the published sample and never run a driver/OS probe synchronously. Cached admission fails closed when the sample is older than `max(5000 ms, 4 * probe_ms)`, the latest probe failed, or a required sampler is not running. An explicit refresh after a material allocation or release bypasses the cadence. |
| `memory.reclaim_mode` | string enum; `"off"` | `KILN_MEMORY_RECLAIM_MODE` (implemented) | `KILN_MEMORY_RECLAIM_MODE` | Exactly `off`, `on-demand`, or `automatic`, case-insensitive with surrounding whitespace ignored. `off` prevents execution of registered allocator reclaim hooks; `on-demand` permits explicit pressure and allocation-retry reclaim calls; `automatic` also permits the background pressure monitor. The immutable serving profile remains authoritative: a profile with allocator reclaim disabled keeps the effective mode off and does not start the monitor. |
| `memory.kv_cache_fp8` | boolean; `false` | `KILN_MEMORY_KV_CACHE_FP8` (implemented) | `KILN_KV_CACHE_FP8` (deprecated compatibility) | Requests E4M3FN KV storage. Backend storage policy may reject or disable the request when unsupported. |
| `memory.cuda_graphs` | boolean; `true` | `KILN_MEMORY_CUDA_GRAPHS` (implemented) | `KILN_CUDA_GRAPHS` (deprecated compatibility) | CUDA-only request. Non-CUDA backends ignore it, and a serving profile with live graph capture disabled selects eager-only execution regardless of this value. |

Capacity detection is device-scoped. Discrete accelerators use the selected
device's driver-reported VRAM and never count GTT as device-local capacity.
Linux DRM topology keeps the primary VRAM heap and a separately admissible
host-backed GTT tier distinct. Any nonzero VRAM heap remains the primary pool,
even when GTT is larger, because that is also a common discrete-AMD shape. This
preserves large carved pools such as Strix Halo's 96 GiB VRAM rather than
incorrectly capping it to Linux's smaller CPU-online pool.

The host-backed tier is independently bounded by GTT free bytes,
`MemAvailable`, the most restrictive finite `memory.max` and `memory.high`
headroom across the visible cgroup hierarchy, and the unified-memory reserve.
Kiln's current Vulkan paged KV cache and recurrent prefix state are
host-resident, so startup requires them to fit both the primary accelerator
budget and this host-backed budget. Prefix state is reserved first; automatic
KV sizing uses the configured inference fraction of the remaining host-backed
headroom. Missing or exhausted host-tier evidence yields a startup error before
the zero-filled Rust allocation is attempted. CUDA and ROCm KV pools remain
device-resident and are not capped by this separate tier.

Only a driver reporting no local VRAM and a nonzero GTT heap is automatically
treated as fully host-shared. A host-shared primary pool retains
`max(6 GiB, 25% of backing capacity)` for the OS and CPU workloads. Apple
Silicon applies the same reserve to its unified pool and samples live host
pressure. The optional `memory.gpu_memory_gb` cap is applied only after safe
physical capacity is established.

Physical-device identity is currently fail-closed while backend selection and
memory probes still expose unrelated logical ordinals. CUDA, ROCm, and Vulkan
startup therefore accepts only logical device `0` when the relevant NVIDIA or
DRM candidate set is provably singular. Multi-device hosts, nonzero ordinals,
failed candidate enumeration, and visibility/remapping controls such as
`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`,
`KILN_VULKAN_DEVICE`, or `GGML_VK_VISIBLE_DEVICES` are rejected before model
upload. `Auto` probing remains diagnostic-only; CPU performs no accelerator
probe, and Apple Silicon uses its single unified physical memory pool. This is
an interim safety restriction until backend selection and probing share a
typed PCI-address or UUID identity.

## `[training]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `training.grad_checkpoint_segments` | optional unsigned integer; omitted (`None`) | `KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS` (implemented) | `KILN_GRAD_CHECKPOINT_SEGMENTS` (deprecated compatibility) | Must be greater than zero when set. When present, selects an explicit process-lifetime gradient-checkpoint segment count for native training; omission leaves workload- and capacity-aware automatic planning enabled. |
| `training.no_grad_checkpoint` | boolean; `false` | `KILN_TRAINING_NO_GRAD_CHECKPOINT` (implemented) | `KILN_NO_GRAD_CHECKPOINT` (deprecated compatibility) | Disables gradient checkpoint execution for native training. The disabled state and any explicit segment count are retained together in the immutable training policy and exact-resume identity. Disabling checkpointing can materially increase training memory. |
| `training.checkpoint_interval` | optional unsigned integer; omitted (`None`) | `KILN_TRAINING_CHECKPOINT_INTERVAL` (implemented) | `KILN_CHECKPOINT_INTERVAL` (deprecated compatibility) | Must be greater than zero when set. Number of committed optimizer steps between checkpoints; per-job configuration overrides it. Omission disables periodic checkpoints. |
| `training.webhook_url` | optional string; omitted (`None`) | `KILN_TRAINING_WEBHOOK_URL` (implemented) | `KILN_TRAINING_WEBHOOK_URL` | Must be a non-empty valid HTTP(S) URL. An exactly empty environment value clears a TOML URL; whitespace is not a clearing value and fails validation. Delivery is fire-and-forget with a five-second timeout after terminal state is recorded. |
| `training.max_queued_jobs` | unsigned integer; `32` | `KILN_TRAINING_MAX_QUEUED_JOBS` (implemented) | `KILN_TRAINING_MAX_QUEUED_JOBS` | Must be greater than zero. At capacity, submissions return HTTP 503 with `Retry-After: 30`. |
| `training.max_tracked_jobs` | unsigned integer; `1024` | `KILN_TRAINING_MAX_TRACKED_JOBS` (implemented) | `KILN_TRAINING_MAX_TRACKED_JOBS` | Must be greater than zero and at least `max_queued_jobs`. Counts queued, running, completed, and failed entries. |
| `training.tracked_job_ttl_secs` | unsigned integer; `604800` | `KILN_TRAINING_TRACKED_JOB_TTL_SECS` (implemented) | `KILN_TRAINING_TRACKED_JOB_TTL_SECS` | Must be greater than zero. Terminal entries older than the TTL are removed; active jobs are never age-evicted. |

Training GPU work is also governed by `server.serving_profile`; the default
`stable` profile does not grant training GPU ownership.

## `[logging]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `logging.level` | string; `"info"` | `KILN_LOGGING_LEVEL` (implemented) | `KILN_LOG_LEVEL` (deprecated compatibility) | One of `trace`, `debug`, `info`, `warn`, `error`, or a valid `tracing_subscriber::EnvFilter` directive such as `kiln=debug,tower_http=warn`. `RUST_LOG` overrides the effective filter but does not make an invalid typed value valid. |
| `logging.format` | string; `"auto"` | `KILN_LOGGING_FORMAT` (implemented) | `KILN_LOG_FORMAT` (deprecated compatibility) | Exactly `auto`, `json`, `pretty`, `text`, or `human`. `auto` is pretty on a stderr TTY and JSON otherwise; `text` and `human` select pretty output. |

Logging is bootstrapped before the full file is validated so parse failures can
be reported through the requested renderer. The authoritative loader then
validates the complete file and stops startup if it differs or is invalid.
Malformed or non-UTF-8 `RUST_LOG` is fatal.

## `[prefix_cache]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `prefix_cache.enabled` | boolean; `true` | `KILN_PREFIX_CACHE_ENABLED` (implemented) | `KILN_PREFIX_CACHE_ENABLED` | Enables reuse of KV blocks and recurrent-state snapshots for shared prompt prefixes. |
| `prefix_cache.max_blocks` | optional unsigned integer; omitted (`None`) | `KILN_PREFIX_CACHE_MAX_BLOCKS` (implemented) | `KILN_PREFIX_CACHE_MAX_BLOCKS` | Must be greater than zero when set. `None` resolves to half of the allocated KV block pool. |
| `prefix_cache.max_entries` | optional unsigned integer; omitted (`None`) | `KILN_PREFIX_CACHE_MAX_ENTRIES` (implemented) | `KILN_PREFIX_CACHE_MAX_ENTRIES` | Must be greater than zero when set. `None` resolves from the relevant safe allocation tier and per-entry recurrent-state bytes, with at least one entry. Vulkan reserves this state from the separately bounded host-backed tier before sizing its host-resident KV pool; an explicit count that cannot fit stops startup. |

## `[speculative]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `speculative.enabled` | boolean; `false` | `KILN_SPECULATIVE_ENABLED` (implemented) | `KILN_SPEC_ENABLED` (deprecated compatibility) | `false` forces the effective method to `off`. For compatibility, `true` with `method = "off"` selects `skip_layer`; that effective method is then subject to the fail-closed startup gate below. |
| `speculative.method` | string enum; `"off"` | `KILN_SPECULATIVE_METHOD` (implemented) | `KILN_SPEC_METHOD` (deprecated compatibility) | TOML accepts `off`, `skip_layer`, or `mtp`. The environment parser also accepts `none`, `0`, `false`; `skiplayer`, `skip-layer`, `self`; and `native_mtp`, `native-mtp`. The deprecated method alias also controls `enabled` when neither enabled spelling is present: non-off enables and off disables. The canonical method name has literal field semantics and does not implicitly enable. |
| `speculative.num_speculative_tokens` | unsigned integer; `4` | `KILN_SPECULATIVE_NUM_SPECULATIVE_TOKENS` (implemented) | `KILN_SPEC_NUM_TOKENS` (deprecated compatibility) | Draft proposal bound K. Must be in `1..=4`; out-of-range values stop startup before accelerator allocation. This conservative ceiling matches the planned local K=1/2/4 qualification matrix and cannot be raised without new accelerator evidence. |
| `speculative.draft_layers` | unsigned integer; `8` | `KILN_SPECULATIVE_DRAFT_LAYERS` (implemented) | `KILN_SPEC_DRAFT_LAYERS` (deprecated compatibility) | Must be greater than zero. When speculative decoding is enabled, model-dependent startup validation also requires this value to be less than the selected model's transformer-layer count. Invalid geometry stops startup; it does not fall back at request time. |

The loaded `SpeculativeDecodingConfig` is the sole serving authority. TOML,
canonical environment overrides, and deprecated aliases are resolved and
validated once during typed startup, then the immutable result is passed to
serving state. Request dispatch does not re-read speculative environment
variables. Restart the server to apply a change.

**Current fail-closed status:** `off` is the only serving method available.
Every enabled method, including `skip_layer` and `mtp`, stops startup until its
cancellation, owner-settlement, EOS, context-capacity, and burst-admission
contracts pass local accelerator qualification. The fields remain part of the
typed configuration so their intended contract is explicit and can be qualified
without introducing a second configuration path. `kiln config --file ...`
applies the same availability gate as `serve`, native MTP weights remain
deferred (`load_mtp=false`), request dispatch contains no speculative branch,
and Desktop always launches with speculative serving off. Benchmark-only
speculative paths require both `KILN_QUALIFICATION=1` and the qualification
harness result-path contract before model loading.

## `[streaming_prefill]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `streaming_prefill.mode` | string enum; `"auto"` | `KILN_STREAMING_PREFILL_MODE` (implemented) | `KILN_STREAMING_PREFILL` and `KILN_STREAMING_PREFILL_ENABLED` (deprecated compatibility) | `auto`, `enabled`, or `disabled`, case-insensitive with surrounding whitespace ignored. `auto` delegates dispatch to backend policy, `enabled` selects every non-empty prompt, and `disabled` selects none. The canonical environment name accepts only those three words. Deprecated aliases also accept the strict boolean forms listed above, mapping true to `enabled` and false to `disabled`. Legacy TOML `enabled = true` or `enabled = false` remains accepted; if `mode` is also present, both must express the same non-auto intent or startup fails. |
| `streaming_prefill.threshold_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_THRESHOLD_TOKENS` (implemented) | `KILN_STREAMING_PREFILL_THRESHOLD_TOKENS` | In `auto` mode, an integer replaces the backend crossover only when that backend already has a threshold-based automatic policy. It does not make CPU or Vulkan auto-dispatch streaming. `0`, negative values, and strings other than `auto` fail startup. `enabled` and `disabled` modes ignore this crossover for dispatch while retaining it in diagnostics. |
| `streaming_prefill.tile_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_TILE_TOKENS` (implemented) | `KILN_STREAMING_TILE_TOKENS` (deprecated compatibility) | Base tile for ordinary tiled prefill and non-tape GDN segment execution. Concrete values must be positive multiples of 64. When this field is concrete and either specialized tile below is `auto`, that specialized route inherits this base value rather than its backend default. |
| `streaming_prefill.tape_tile_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS` (implemented) | `KILN_TAPE_STREAMING_TILE_TOKENS` (deprecated compatibility) | Tile used by tape-authoritative training forward paths. Concrete values must be positive multiples of 64. `auto` inherits an explicit `tile_tokens`; when both are `auto`, backend policy owns the value. |
| `streaming_prefill.detached_full_attn_tile_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS` (implemented) | `KILN_DETACHED_FULL_ATTN_TILE_TOKENS` (deprecated compatibility) | Tile for detached materialized full-attention training work. A concrete value also controls its derived boundary-forward and tape-replay variants. `auto` inherits an explicit `tile_tokens`; when both are `auto`, each variant keeps its backend default. Every concrete value must be a positive multiple of 64. |
| `streaming_prefill.last_token_lm_head` | boolean; `true` | `KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD` (implemented) | `KILN_STREAMING_LAST_TOKEN_LM_HEAD` (deprecated compatibility) | When true, the final inference streaming tile computes the LM head only for its last row. All centralized strict boolean spellings, including `on` and `off`, work identically. |

Every deprecated alias is parsed strictly and emits a value-free startup
warning naming its canonical replacement. When a canonical name is present,
each present alias must normalize to the same typed value or startup fails and
names both variables. For `mode`, if both deprecated aliases are present
without `KILN_STREAMING_PREFILL_MODE`, both are validated and the historical
higher-precedence `KILN_STREAMING_PREFILL_ENABLED` value wins over
`KILN_STREAMING_PREFILL`. A malformed lower-precedence alias still fails; it is
never ignored because another spelling is present.

The selected backend supplies the following `auto` policy. Dispatch thresholds
are inclusive. Tile counts are tokens:

| Backend | Automatic dispatch | Base | Tape | Detached full attention | Detached boundary | Detached tape replay |
|---|---:|---:|---:|---:|---:|---:|
| CPU | never | 8192 | 8192 | 8192 | 8192 | 8192 |
| CUDA | prompt tokens >= 2048 | 1024 | 1024 | 8192 | 65536 | 65536 |
| ROCm | prompt tokens >= 2048 | 1024 | 1024 | 8192 | 8192 | 8192 |
| Metal | prompt tokens >= 2048 | 2048 | 2048 | 8192 | 8192 | 8192 |
| Vulkan | never | 2048 | 2048 | 8192 | 8192 | 8192 |

`mode = "enabled"` can deliberately force non-empty CPU or Vulkan inputs onto
the streaming path; this is an operator override, not a claim that the route is
qualified or faster on that backend. Conversely, `mode = "disabled"` is the
clean isolation control for a monolithic-prefill A/B. A threshold override
only adjusts an existing CUDA, ROCm, or Metal automatic crossover. Dispatch
eligibility and physical splitting are distinct: an eligible route creates
multiple tiles only when its sequence is longer than the effective tile for
that route.

Startup resolves this section once after backend selection and injects the same
immutable execution policy into ordinary generation, prompt-logprob scoring,
native SFT/GRPO/OPD admission and training forwards, local OPD teachers, MTP
alignment, checkpoint planning, and the benchmark harness. Server-owned
training and teacher construction fail closed if that startup policy is
missing. No production model or trainer path re-reads these public environment
names. Every change requires a restart; an existing job or request cannot
observe a mid-process change.

Kiln prompt-logprob teacher identity uses
`kiln.prompt-logprobs.inference-config.v2` and hashes the complete resolved
execution policy: mode, threshold, every base/tape/detached tile variant, and
last-token LM-head behavior. Changing any of these fields therefore changes
`inference_config_sha256`, the canonical teacher content revision, and every
identity-bound logit-cache key. A deployment must re-register the teacher and
must not resume an OPD checkpoint pinned to the previous revision.

`GET /v1/config` exposes the complete resolved object at
`streaming_prefill`. Health repeats it at
`prefill_runtime.streaming_prefill`, and trusted debug exposes it at top-level
`streaming_prefill`. The object separates:

- configured mode/source, backend dispatch rule, effective dispatch rule, and
  effective authority;
- configured/backend/effective threshold, including whether an override
  actually changed a threshold-based backend policy;
- configured/backend/effective values and sources for base, tape, detached,
  detached-boundary, and detached-replay tiles;
- last-token LM-head configured/effective state and source; and
- `immutable_after_startup` plus `restart_required_to_change`.

An inherited specialized tile reports a closed source such as
`inherited_from_tile_tokens_config_file`; an untouched automatic value reports
`backend_policy`. There is intentionally no single `effective_enabled` boolean:
in automatic mode dispatch depends on the current prompt length.

Examples:

```toml
# Keep backend dispatch and tile policy (recommended default).
[streaming_prefill]
mode = "auto"
threshold_tokens = "auto"
tile_tokens = "auto"
tape_tile_tokens = "auto"
detached_full_attn_tile_tokens = "auto"
last_token_lm_head = true
```

```toml
# Isolate a suspected tiled-prefill pause.
[streaming_prefill]
mode = "disabled"
```

```toml
# ROCm tuning experiment: stream from 4096 tokens, use one base tile for all
# ordinary, tape, detached, boundary, and replay routes through inheritance.
[streaming_prefill]
mode = "auto"
threshold_tokens = 4096
tile_tokens = 2048
tape_tile_tokens = "auto"
detached_full_attn_tile_tokens = "auto"
```

## `[adapters]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `adapters.max_disk_bytes` | optional unsigned integer; `107374182400` (100 GiB) | `KILN_ADAPTERS_MAX_DISK_BYTES` (implemented) | `KILN_ADAPTERS_MAX_DISK_BYTES` | Caps finalized adapter bytes under `adapter_dir`, excluding upload staging and the composed cache. For the environment alias only, empty or `0` means `None` and disables the cap. TOML `0` is accepted as a literal zero cap, not as `None`. |
| `adapters.composed_cache_max_bytes` | optional unsigned integer; `10737418240` (10 GiB) | `KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES` (implemented) | `KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES` | LRU byte cap for `.composed`. Environment empty or `0` disables this dimension; TOML `0` remains a zero cap. |
| `adapters.composed_cache_max_entries` | optional unsigned integer; `64` | `KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES` (implemented) | `KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES` | LRU entry-count cap for `.composed`. Environment empty or `0` disables this dimension; TOML `0` remains a zero cap. |

Because all three typed defaults are `Some(...)` and TOML has no null value,
the supported way to opt out is the working environment alias with `0` or an
empty value.

## `[teachers]`

`teachers.credentials` is a dynamic map of server-owned credential handles.
There is no field-level environment override for the map. Each credential
points to a separately named secret environment variable; that secret is not
stored in TOML, `AppState`, API responses, or receipts.

```toml
[teachers.credentials.primary-vllm]
origin = "https://vllm.example.com:8443"
api_key_env = "VLLM_TEACHER_API_KEY"
```

| TOML field template | Type and default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `teachers.credentials.<id>.origin` | required string for each entry; no entries by default | none; structured trust policy is TOML-only | none | Must be the exact canonical `scheme://host[:port]` origin, with no path, query, fragment, or embedded credentials. HTTPS is required unless the host is loopback. |
| `teachers.credentials.<id>.api_key_env` | required string for each entry | none; this value names the secret variable | the named secret variable itself | Must match `[A-Za-z_][A-Za-z0-9_]*`, length `1..=128`. The named variable must exist and contain a non-whitespace value at startup. |

Credential ids must be `1..=64` ASCII letters, digits, `_`, or `-`. A handle is
accepted only for its exact configured origin. Non-loopback teachers require a
configured credential handle; credential-free teachers are restricted to
loopback. Secret availability is checked again when resolving a teacher, but
operators should still treat secret changes as restart-scoped configuration.

## `[eval]`

The section is optional. When absent, the runtime uses the same queue defaults
shown below and roots eval data at `<adapter_dir>/.eval`. It creates `suites`,
`datasets`, and `judgments` below that root.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `eval.eval_dir` | optional path string; omitted (`None`) | `KILN_EVAL_EVAL_DIR` (target only; not implemented) | none | Must be non-empty when set. Despite the historical field name, runtime treats it as the shared eval root and creates the three registry subdirectories below it. |
| `eval.max_queued_jobs` | unsigned integer; `32` | `KILN_EVAL_MAX_QUEUED_JOBS` (target only; not implemented) | none | Must be greater than zero. |
| `eval.max_tracked_jobs` | unsigned integer; `1024` | `KILN_EVAL_MAX_TRACKED_JOBS` (target only; not implemented) | none | Must be greater than zero and at least `max_queued_jobs`. |
| `eval.webhook_url` | optional string; omitted (`None`) | `KILN_EVAL_WEBHOOK_URL` (target only; not implemented) | none | Must be a non-empty valid HTTP(S) URL. Terminal eval notifications are fire-and-forget. |

Names such as `eval.root`, `eval.max_tracked_eval_jobs`, and an
`[eval.generation]` table are not accepted by the current typed schema.

## `[request_log]`

Request logging is enabled by default. It records inference request and response
JSON on a dedicated bounded writer thread; overload drops log rows instead of
blocking inference. Rotation and retention bound disk use.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `request_log.enabled` | boolean; `true` | `KILN_REQUEST_LOG_ENABLED` (implemented) | `KILN_REQUEST_LOG_ENABLED` | Master switch. Initialization failure logs a warning and disables logging rather than aborting an otherwise valid server startup. |
| `request_log.dir` | optional path string; omitted (`None`) | `KILN_REQUEST_LOG_DIR` (implemented) | `KILN_REQUEST_LOG_DIR` | Must be non-empty when set. `None` resolves to `<adapter_dir>/.requests`. The environment alias cannot clear a TOML directory; an empty value is fatal. |
| `request_log.max_file_bytes` | unsigned integer; `67108864` (64 MiB) | `KILN_REQUEST_LOG_MAX_FILE_BYTES` (implemented) | `KILN_REQUEST_LOG_MAX_FILE_BYTES` | Must be at least `4096`. The active JSONL file rotates after reaching the threshold. |
| `request_log.max_total_bytes` | unsigned integer; `2147483648` (2 GiB) | `KILN_REQUEST_LOG_MAX_TOTAL_BYTES` (implemented) | `KILN_REQUEST_LOG_MAX_TOTAL_BYTES` | Must be greater than zero. Oldest rotated files are removed until retained bytes fit. There is no validation requiring this value to exceed `max_file_bytes`. |
| `request_log.compress` | boolean; `true` | `KILN_REQUEST_LOG_COMPRESS` (implemented) | `KILN_REQUEST_LOG_COMPRESS` | Gzip rotated files when true. |
| `request_log.max_capture_bytes` | unsigned integer; `4194304` (4 MiB) | `KILN_REQUEST_LOG_MAX_CAPTURE_BYTES` (implemented) | `KILN_REQUEST_LOG_MAX_CAPTURE_BYTES` | Must be greater than zero. Per-request and per-response storage cap; truncation affects the log only, never the wire response. |

The default is intentionally data-bearing. Review log-directory permissions,
retention, and the captured request content before exposing the service to
untrusted traffic.

## `[agent]`

The section is optional. Its absence disables the scheduled self-improvement
loop, while the embedded run engine still receives the default concurrency and
timeout.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `agent.self_improve_interval_hours` | optional unsigned integer; omitted (`None`) | `KILN_AGENT_SELF_IMPROVE_INTERVAL_HOURS` (target only; not implemented) | none | Omission disables scheduling. `0` is accepted and also results in no scheduler. The first run occurs one full interval after startup; cadence is persisted under the adapter directory. |
| `agent.self_improve` | optional structured value; omitted (`None`) | `KILN_AGENT_SELF_IMPROVE` (target only; not implemented) | none | Request template submitted to the same self-improvement path as the API. This value is intentionally open structured data; its inner request contract is validated by that subsystem rather than by `KilnConfig`. |
| `agent.max_concurrent_runs` | unsigned integer; `2` | `KILN_AGENT_MAX_CONCURRENT_RUNS` (target only; not implemented) | none | Must be greater than zero. Embedded pi runs above the limit queue FIFO. |
| `agent.run_timeout_secs` | unsigned integer; `900` | `KILN_AGENT_RUN_TIMEOUT_SECS` (target only; not implemented) | none | Must be at least `10`. A per-run timeout can override the server default. |

## Effective values and provenance

### Library runtime boundaries

Applications embedding `kiln-model` directly must initialize accelerator
memory governance as an explicit startup step. Call
`InferenceMemoryRuntime::initialize(device, governor_config)` before building
the runner, then pass the returned binding to
`ModelRunner::new_with_initialized_runtime`. Initialization validates the
backend/probe physical-device identity, detects safe capacity, treats
`GovernorConfig::capacity_limit_bytes` as a cap, publishes a live sample, and
starts the background sampler. The fallible runner constructor verifies the
binding against the model-weight device and installed process policy without
probing again. `ModelRunner::new`, `new_with_options`, and
`new_with_runtime_options` remain non-probing compatibility constructors for
owners such as `kiln-server` that install the same process-wide policy before
model construction.

Accelerator training has the same explicit identity rule. Construct
`TrainingRuntimeContext::new_for_device(device, effective_vram, policy)` and
use the `*_with_runtime` SFT, GRPO, streamed-GRPO, or OPD entry point. A context
without a device binding cannot authorize accelerator-backed weights. Weight
storage and native training execution must currently name the same device.
Vulkan serving uses CPU-host weight handles, but its full-model resident upload
path is not production-qualified and is therefore rejected before queue or
dataset admission with `training_backend_unsupported`; there is no environment
escape hatch. Vulkan-device training fixtures remain valid when their weights
are genuinely Vulkan-resident. Use the HF/TRL export/import workflow for
production training while serving through the hybrid Vulkan backend.

Kiln currently exposes several complementary, partial views:

- `kiln config --file <path>` performs authoritative parsing and validation,
  then prints a human-oriented subset of resolved fields.
- `GET /v1/config` reports runtime diagnostics for serving profile, effective
  decode width, speculative configuration and availability, VRAM/KV state,
  native-training runtime/weight devices and
  support reason, checkpoint segmentation, memory budgets, and generation
  defaults. Its `speculative` object distinguishes `configured_method` and
  `configured_effective_method` from the actual `serving_effective_method`
  (`off`), reports `serving_routable=false`, the stable
  `serving_unavailable_reason`, the K ceiling, and immutable backend MTP
  support facts. It does not expose live routing, admission, or weight-readiness
  fields because those are not serving authorities. A running process cannot
  have a non-off configured method through the supported startup path.
  `training.native_training_supported=false` is accompanied by
  `training.native_training_unavailable_reason` and matches the admission
  error without scanning a corpus. When checkpoint execution is disabled,
  `training.checkpoint_segments` is `0`; an optional segment count retained in
  `training.checkpoint_policy` is provenance, not an active execution plan. The
  endpoint is not a serialization of all accepted TOML.
- `GET /v1/config` also reports cached-sample age, maximum age, staleness,
  sampler requirement/liveness/health, cgroup `memory.high`, and the separately
  bounded `vram.live.raw_observations.host_backed` tier. `/health` exposes the
  same liveness decision, while `/metrics` exports fixed-cardinality gauges for
  alerting. A stale or failed sample remains visible diagnostically but
  contributes zero allocation headroom.
- `/health` and `/v1/debug/model-state` expose config hashes and other runtime
  identity data.
- Startup logs record serving-profile provenance and configured/backend/final
  decode-width sources.

Explicit source tracking (`default`, `config_file`, or `environment`) currently
exists for `server.serving_profile`, `server.deterministic`,
`server.stream_stall_grace_ms`, the three server batching/prefill budgets,
`server.max_decode_batch`, all eight `[batching]` fields, all six
`[streaming_prefill]` fields, and `memory.reclaim_mode`. Other fields have
resolved values but do not yet carry per-field source metadata.

The `kiln_env_config_hash` binds the serialized effective typed configuration
and the process's complete `KILN_*` environment map. It is an identity digest,
not a human-readable effective-config dump.

There is not yet one endpoint or CLI mode that dumps every effective field,
its source, backend-derived adjustment, and restart requirement. Until that
lands, use the typed validation command plus startup/runtime diagnostics.

## Known configuration migration limitations

These are current implementation facts, not recommended architecture:

1. **Effective dump:** neither `kiln config` nor `/v1/config` covers the whole
   typed object with provenance and backend-derived values.
2. **Deprecated aliases:** 54 non-canonical spellings across 52 fields remain
   temporarily for compatibility, including `KILN_DEFAULT_NO_THINK`. Each use
   warns at startup; canonical and compatibility names cannot silently
   disagree.

The intended migration is to resolve every public runtime field exactly once,
pass immutable typed configuration into its owner, generate the environment
contract mechanically, and reject direct production environment reads outside
that boundary. Internal experiment and qualification flags should remain
separate from this public API rather than being promoted by documentation.
