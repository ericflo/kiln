# Configuration Reference

This document is the canonical operator reference for Kiln's typed server
configuration. It describes the configuration accepted by the current
`KilnConfig` and `RequestLogConfig` implementations, including current defaults,
validation, working environment-variable aliases, and runtime provenance.

The machine-readable contract is
[`contracts/kiln-config-v1.schema.json`](../contracts/kiln-config-v1.schema.json).
The documentation website renders its complete field reference directly from
that schema. `python3 scripts/check_config_schema.py --self-test` verifies that
the schema, these tables, and `kiln.example.toml` agree.

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

The generated [Runtime Environment Inventory](RUNTIME_ENVIRONMENT_INVENTORY.md)
catalogs every direct environment read and mutation in crate-owned source,
classifies its owner boundary, and exposes the remaining typed-config migration
queue. It is an engineering inventory, not a second list of supported settings.

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
kiln config --file kiln.toml --backend rocm
```

This command uses the same typed loader and environment precedence as serving,
but its pretty output is a summary rather than a complete effective-config
dump. The optional `--backend cpu|cuda|rocm|metal|vulkan` selector resolves that
target's static scheduling capabilities and applies the same actor-prefill
startup validator as `kiln serve`. It does not enumerate devices, probe memory,
open an accelerator, or load model weights.

## Coverage summary

The accepted TOML surface contains 15 top-level sections and 114 fixed leaf
fields. Dynamic `teachers.credentials.<id>` entries add two leaf fields per
credential. Of the 114 fixed fields:

- 109 implement the canonical mechanical environment name;
- 71 also retain one or more deprecated compatibility spellings (76 aliases
  total);
- 5 are config-file-only and have no environment override;
- the 76 aliases include `KILN_DEFAULT_NO_THINK`, the second deprecated
  compatibility spelling for `server.default_thinking_enabled`.

The tables below cover all 114 fixed fields and both dynamic credential fields.
The schema additionally records the accepted deprecated TOML-only
`streaming_prefill.enabled` compatibility field so validators match the loader.

## `[server]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `server.serving_profile` | string enum; `"stable"` | `KILN_SERVER_SERVING_PROFILE` (implemented) | `KILN_SERVING_PROFILE` (deprecated compatibility) | `stable`, `experimental`, or `maintenance`, case-insensitive. Process-lifetime policy; restart required. |
| `server.deterministic` | boolean; `false` | `KILN_SERVER_DETERMINISTIC` (implemented) | `KILN_DETERMINISTIC` (deprecated compatibility) | Enables deterministic tensor behavior and forces the effective concurrent decode width to one. |
| `server.host` | string; `"127.0.0.1"` | `KILN_SERVER_HOST` (implemented) | `KILN_HOST` (deprecated compatibility) | Must be non-empty. Binding beyond loopback exposes an unauthenticated inference and training API; use a trusted network or authenticated reverse proxy. |
| `server.port` | unsigned 16-bit integer; `8420` | `KILN_SERVER_PORT` (implemented) | `KILN_PORT` (deprecated compatibility) | `1..=65535`. |
| `server.request_timeout_secs` | unsigned integer; `600` | `KILN_SERVER_REQUEST_TIMEOUT_SECS` (implemented) | `KILN_REQUEST_TIMEOUT_SECS` (deprecated compatibility) | Must be greater than zero. Bounds a request, including model work and cleanup settlement. |
| `server.terminal_access` | string enum; `"loopback_only"` | `KILN_SERVER_TERMINAL_ACCESS` (implemented) | `KILN_TERMINAL` (deprecated compatibility; boolean spellings map to enabled/disabled) | `loopback_only`, `enabled`, or `disabled`. Compatibility boolean spellings are accepted from the environment. This capability can execute arbitrary code; changing it requires restart. |
| `server.http_send_buffer_bytes` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_HTTP_SEND_BUFFER_BYTES` (implemented) | `KILN_HTTP_SEND_BUFFER_BYTES` (deprecated compatibility) | When set, `1024..=16777216`. Applied to accepted sockets. Startup preflights the listener and rejects an OS read-back smaller than requested. |
| `server.stream_stall_grace_ms` | unsigned integer; `2000` | `KILN_SERVER_STREAM_STALL_GRACE_MS` (implemented) | `KILN_STREAM_STALL_GRACE_MS` (deprecated compatibility) | `10..=2000`. A request retaining KV state with no streaming delivery progress is selected for cancellation after this grace. |
| `server.max_batch_tokens` | unsigned integer; `512` | `KILN_SERVER_MAX_BATCH_TOKENS` (implemented) | `KILN_MAX_BATCH_TOKENS` (deprecated compatibility) | `2..=65536`. Combined decode-plus-prefill token budget for one batching-actor cycle. |
| `server.max_prefill_tokens_per_cycle` | unsigned integer; `256` | `KILN_SERVER_MAX_PREFILL_TOKENS_PER_CYCLE` (implemented) | `KILN_MAX_PREFILL_TOKENS_PER_CYCLE` (deprecated compatibility) | `1..=65536`. Independent new-prompt-token ceiling within the combined actor budget. ROCm actor serving requires equality with the effective streaming tile, direct streaming dispatch no later than that boundary, and `server.max_batch_tokens >= tile + effective max_decode_batch`; unsafe combinations fail startup. The checked Strix Halo Vulkan development profile pins `128`; concurrent Vulkan serving at `256` is not qualified. |
| `server.max_prefill_layers_per_cycle` | unsigned integer; `4` | `KILN_SERVER_MAX_PREFILL_LAYERS_PER_CYCLE` (implemented) | `KILN_MAX_PREFILL_LAYERS_PER_CYCLE` (deprecated compatibility) | `1..=1024`. Number of transformer layers an in-flight prefill chunk may execute before yielding to decode. |
| `server.max_decode_batch` | `"auto"` or unsigned integer; `"auto"` | `KILN_SERVER_MAX_DECODE_BATCH` (implemented) | `KILN_MAX_DECODE_BATCH` (deprecated compatibility) | `auto`, `backend`, and `backend_policy` all delegate to backend policy; an integer must be `1..=65536`. Deterministic mode and `max_batch_tokens` may lower the final width. The Strix Halo Vulkan development-soak candidate sets `2`; together with a prefill admission quantum of two, this yields four total active requests and admits an equal pair together. |
| `server.eval_mode` | boolean; `false` | `KILN_SERVER_EVAL_MODE` (implemented) | `KILN_EVAL_MODE` (deprecated compatibility) | Enables deterministic eval-serving defaults, headers, adapter warnings, and transient-cache cleanup behavior. `serve --eval-mode` applies a typed override after environment resolution and wins without mutating process environment. |
| `server.debug_model_state` | boolean; `false` | `KILN_SERVER_DEBUG_MODEL_STATE` (implemented) | None | Enables trusted `GET /v1/debug/model-state` diagnostics without changing inference, cache, or eval semantics. `server.eval_mode=true` also enables the endpoint. The response contains model/configuration/runtime state but no prompt or user-message contents. |
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

## `[accelerator]`

This section owns process-lifetime accelerator execution behavior that must be
fixed before the primary device context or model runner is created. The
resolved object uses schema `kiln.accelerator-runtime-policy.v12`. Startup,
`kiln config`, `GET /v1/config`, `/health`, trusted debug state, and the
dashboard all report the same configured/effective/source values, plus the
compiled Vulkan kernel-policy schema ID; lower model, tensor, and kernel paths
do not re-read these public environment names.

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `accelerator.kt_api_mode` | string enum; `"auto"` | `KILN_ACCELERATOR_KT_API_MODE` (implemented) | none | `auto`, `all`, or `disabled`, case-insensitive. `auto` enables the qualified kiln-tensor adapter routes while leaving the experimental generic matmul and paged-KV routes inactive. `all` enables every adapter route; `disabled` forces legacy fallbacks. The two explicit modes require `server.serving_profile = "experimental"`. The mode is resolved before accelerator execution, remains immutable for the process lifetime, and is reported with source attribution. Restart required. |
| `accelerator.full_attention_score_budget_mib` | unsigned integer MiB; `2048` | `KILN_ACCELERATOR_FULL_ATTENTION_SCORE_BUDGET_MIB` (implemented) | none | `64..=2048`. Immutable ceiling for exact full-attention materialized score and scratch geometry across CPU, CUDA, ROCm, Metal, and Vulkan model routes. ROCm online attention derives `min(value, 1024)` MiB while retaining fixed qualified 2048-query/4096-key tiles. Runtime memory observations remain fail-closed admission and reservation checks; they never resize the geometry or switch to smaller tiles during a request. Restart required. |
| `accelerator.vulkan_device_index` | `"auto"` or unsigned integer; `"auto"` | `KILN_ACCELERATOR_VULKAN_DEVICE_INDEX` (implemented) | `KILN_VULKAN_DEVICE` (deprecated compatibility) | `auto` preserves automatic discrete-GPU preference and otherwise chooses the first enumerated Vulkan physical device. An integer strictly selects that zero-based Vulkan enumeration index. An unavailable index fails logical-device startup; it is never ignored or replaced with another device. The immutable selection is installed before Vulkan device creation and reported with source attribution. Restart required. |
| `accelerator.vulkan_validation` | boolean; `false` | `KILN_ACCELERATOR_VULKAN_VALIDATION` (implemented) | `KILN_VULKAN_VALIDATION` (deprecated compatibility) | `true` requires `server.serving_profile = "experimental"` and enables `VK_LAYER_KHRONOS_validation` when the Vulkan instance is created. Startup fails if the layer is not installed. This is not mutable per request or dispatch. Restart required. |
| `accelerator.cuda_kernel_profile` | string enum; `"native_default"` | `KILN_ACCELERATOR_CUDA_KERNEL_PROFILE` (implemented) | none | `native_default` or `portable_fallback`, case-insensitive. `native_default` preserves the fourteen CUDA backend routes that were enabled by default before consolidation; this name deliberately makes no current-hardware qualification claim. `portable_fallback` declines every owned route. The complete route set is installed before CUDA backend construction, immutable for the process lifetime, and reported with source attribution. Restart required. |
| `accelerator.rocm_synchronization_mode` | string enum; `"legacy_host_barriers"` | `KILN_ACCELERATOR_ROCM_SYNCHRONIZATION_MODE` (implemented) | none | `legacy_host_barriers` or `stream_ordered`, case-insensitive. `stream_ordered` requires `server.serving_profile = "experimental"`; other profiles fail startup rather than silently weakening the request. Restart required. |
| `accelerator.rocm_strided_batched_matmul_mode` | string enum; `"auto"` | `KILN_ACCELERATOR_ROCM_STRIDED_BATCHED_MATMUL_MODE` (implemented) | `KILN_FORCE_ROCM_STRIDED_BATCHED_MATMUL` and `KILN_DISABLE_ROCM_STRIDED_BATCHED_MATMUL` (deprecated compatibility) | `auto`, `enabled`, or `disabled`, case-insensitive. `auto` applies the qualified gfx115x large-attention guard; `enabled` always permits the strided-batched route and `disabled` always uses per-row GEMMs. Either explicit route requires the experimental profile. Conflicting aliases, malformed values, and canonical-plus-alias inputs fail startup. Restart required. |
| `accelerator.rocm_bf16_matmul_output_mode` | string enum; `"auto"` | `KILN_ACCELERATOR_ROCM_BF16_MATMUL_OUTPUT_MODE` (implemented) | `KILN_FORCE_ROCM_BF16_MATMUL_F32_OUTPUT` and `KILN_DISABLE_ROCM_BF16_MATMUL_F32_OUTPUT` (deprecated compatibility) | `auto`, `native_bf16`, or `f32_then_cast`, case-insensitive. `auto` applies the qualified ROCm 7.2 shape guard; the explicit routes require the experimental profile. Conflicting aliases, malformed values, and canonical-plus-alias inputs fail startup. Restart required. |
| `accelerator.rocm_kernel_profile` | string enum; `"qualified"` | `KILN_ACCELERATOR_ROCM_KERNEL_PROFILE` (implemented) | none | `qualified`, `portable_fallback`, or `experimental_multiblock`, case-insensitive. `qualified` installs forty-three of forty-five accelerated model/tensor routes, leaving multi-block GDN prefill and the correctness-disabled fused RMSNorm route off, and retains thirty fixed correctness/bounded-work leaves. `portable_fallback` declines all forty-five accelerated routes while retaining the same safety behavior. `experimental_multiblock` enables the multi-block GDN prefill experiment but retains fused RMSNorm off, installing forty-four of forty-five accelerated routes, and requires `server.serving_profile = "experimental"`. The complete 75-leaf policy is immutable after startup and governs backend/model dispatch, loading, forward/tape execution, W8 decode, training geometry, paged-attention specialization and splitting, concat assembly, finite checks, RMSNorm row tiling, and flash-attention forward/backward route selection and fixed tiling through Rust and C++ kernel boundaries. Retired per-kernel variables are not aliases. Restart required. |
| `accelerator.rocm_graph_mode` | string enum; `"profile"` | `KILN_ACCELERATOR_ROCM_GRAPH_MODE` (implemented) | `KILN_ROCM_GRAPHS` and `KILN_ROCM_GRAPH_CAPTURE` (deprecated compatibility) | `profile`, `disabled`, `warmup_then_eager`, or `lazy_capture_replay`, case-insensitive. `profile` resolves to `disabled` under stable/maintenance and `lazy_capture_replay` under experimental. The two explicit non-disabled modes require the experimental profile. Restart required. |
| `accelerator.rocm_graph_cache_entries` | unsigned integer; `8` | `KILN_ACCELERATOR_ROCM_GRAPH_CACHE_ENTRIES` (implemented) | `KILN_ROCM_GRAPH_CACHE_MAX` (deprecated compatibility) | `1..=64`. Bounds retained native graph entries in every product and embedding constructor. At saturation, admission reclaims idle owners first and then the minimum fair-LRU active entries while preserving one graph per active owner after the incoming candidate. Zero or unbounded capacities are rejected. Restart required. |
| `accelerator.rocm_graph_cache_max_bytes` | unsigned integer bytes; `1073741824` (1 GiB) | `KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES` (implemented) | none | `67108864..=17179869184` (64 MiB through 16 GiB). Independently bounds requested physical bytes retained by graph-owned stable tensors, capture arenas, private-stream hipBLASLt workspaces, and owner slot state. Opaque HIP graph/exec/stream/event overhead is counted as objects and remains subject to live driver-pressure policy. Restart required. |

`accelerator.kt_api_mode` replaces the former `KILN_USE_KT_API_*`,
`KILN_DISABLE_KT_API_*`, and `KILN_USE_KT_PAGED_KV_*` debugging switches.
Those per-operation names are not compatibility aliases: they created
order-dependent route combinations that could not be represented, validated,
or reported as one startup policy. Use the typed field or its mechanically
derived canonical environment name instead.

`accelerator.cuda_kernel_profile` owns these fourteen CUDA backend decisions:

| Policy leaf | `native_default` | `portable_fallback` |
|---|---:|---:|
| `full_attn_qkv_in_proj` | on | off |
| `gdn_ab_in_proj` | on | off |
| `gdn_prefill_ab_in_proj` | on | off |
| `gdn_prefill_gates` | on | off |
| `gdn` | on | off |
| `gdn_gates` | on | off |
| `gdn_gated_rms_norm` | on | off |
| `gdn_decode_fused` | on | off |
| `gdn_decode_unexpanded_qk` | on | off |
| `gdn_decode_qk_norm_recurrent` | on | off |
| `gdn_decode_qk_norm_recurrent_rmsnorm` | on | off |
| `fused_conv1d` | on | off |
| `lora_decode_add` | on | off |
| `gdn_full_chunk_forward_multiblock` | on | off |

The CUDA-specific variables `KILN_DISABLE_CUDA_FULL_ATTN_QKV_IN_PROJ`,
`KILN_DISABLE_CUDA_GDN_AB_IN_PROJ`,
`KILN_DISABLE_CUDA_GDN_PREFILL_AB_IN_PROJ`,
`KILN_DISABLE_CUDA_GDN_PREFILL_GATES`,
`KILN_DISABLE_FUSED_GDN_DECODE`,
`KILN_DISABLE_GDN_DECODE_UNEXPANDED_QK`,
`KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT`,
`KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM`,
`KILN_DISABLE_CUDA_LORA_DECODE_ADD`, and
`KILN_DISABLE_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK` no longer control CUDA and
are not aliases. The generic GDN/convolution spellings are also no longer read
by the CUDA backend, but remain documented only for backends not yet migrated.

`accelerator.rocm_kernel_profile` replaces the model-layer ROCm controls
`KILN_DISABLE_ROCM_ATTN_DECODE_QKV_PREP`,
`KILN_ENABLE_ROCM_GDN_STREAMING_FASTPATHS`,
`KILN_DISABLE_ROCM_LONG_FLASH_ATTN`,
`KILN_ROCM_FLASH_NATIVE_RECTANGULAR_CAUSAL`,
`KILN_DISABLE_ROCM_TRAINING_MLP_CHUNKING`,
`KILN_ROCM_TRAINING_MLP_CHUNK_TOKENS`,
`KILN_DISABLE_ROCM_W8_SAMPLED_LM_HEAD`,
`KILN_DISABLE_ROCM_SPLIT_Q_GATE_TRAINING`,
`KILN_ROCM_SPLIT_Q_GATE_ROW_TILE_TOKENS`,
`KILN_DISABLE_ROCM_SPLIT_Q_GATE_F32_OUTPUT`,
`KILN_DISABLE_ROCM_GQA_SDPA_F32`,
`KILN_ROCM_TAPE_FLASH_MATERIALIZED`, `KILN_ROCM_W8A16`,
`KILN_ROCM_W8A8`, `KILN_ROCM_W8A8_SAMPLED_LM_HEAD`, and
`KILN_DISABLE_ROCM_W8_SWIGLU`, plus the lower-level tensor spellings
`KILN_DISABLE_ROCM_SPLIT_PAGED_ATTN`,
`KILN_ROCM_PAGED_ATTN_SPLIT_TOKENS`,
`KILN_ROCM_PAGED_ATTN_MAX_SPLITS`,
`KILN_DISABLE_ROCM_GQA_PAGED_ATTN`,
`KILN_DISABLE_ROCM_GQA_D128_PARALLEL`,
`KILN_DISABLE_ROCM_GQA_D256_PARALLEL`,
`KILN_DISABLE_ROCM_CONCAT_SAFE_ROW_ASSEMBLY`,
`KILN_ROCM_CONCAT_AXIS0_ROW_COPY`,
`KILN_ROCM_IS_FINITE_LARGE_HOST_SCAN`,
`KILN_ROCM_IS_FINITE_HOST_SCAN_ELEMENTS`, and
`KILN_ROCM_RMSNORM_ROW_TILE_ROWS`. None is an alias. The two default-off
investigation switches for streaming GDN and materialized tape flash have no
profile leaf because no qualified product uses them. The default-off axis-zero
concat experiment and its dead vectorized kernel were also deleted rather than
promoted.

The context-owned flash-attention subsection replaces or deletes all 50 former
flash build/runtime spellings: `KILN_ROCM_ATTENTION_COOPERATIVE_YIELD`,
`KILN_ROCM_ATTENTION_YIELD_MS`, `KILN_ROCM_DISABLE_CK_FMHA`,
`KILN_ROCM_ENABLE_CK_FMHA`, `KILN_ROCM_ENABLE_CK_FMHA_FWD`,
`KILN_ROCM_F32_MATMUL_INNER_TILE`,
`KILN_ROCM_FLASH_BWD_PRECOMPUTE_DELTA`,
`KILN_ROCM_FLASH_BWD_PRECOMPUTE_DELTA_MAX_SEQ`, `KILN_ROCM_FLASH_CK`,
`KILN_ROCM_FLASH_COLLAPSED_GQA_BWD`,
`KILN_ROCM_FLASH_DISABLE_CK_BWD_COLLAPSED_GQA`,
`KILN_ROCM_FLASH_DISABLE_WMMA_GQA_QBLOCK`,
`KILN_ROCM_FLASH_DISABLE_WMMA_GQA_R64K16`,
`KILN_ROCM_FLASH_DISABLE_WMMA_GQA_R64K32`,
`KILN_ROCM_FLASH_DISABLE_WMMA_GQA_R64K32_LOG2`,
`KILN_ROCM_FLASH_MATMUL_BWD`, `KILN_ROCM_FLASH_NATIVE_BWD`,
`KILN_ROCM_FLASH_NATIVE_BWD_FORCE`,
`KILN_ROCM_FLASH_NATIVE_BWD_LONG_MIN_SEQ`,
`KILN_ROCM_FLASH_NATIVE_BWD_MAX_SEQ`,
`KILN_ROCM_FLASH_NATIVE_DIRECT_COLLAPSED_GQA_BWD`,
`KILN_ROCM_FLASH_NATIVE_DIRECT_COLLAPSED_GQA_QPAR`,
`KILN_ROCM_FLASH_NATIVE_FFI_SYNC`, `KILN_ROCM_FLASH_NATIVE_GQA_QBLOCK`,
`KILN_ROCM_FLASH_NATIVE_GQA_QBLOCK_MIN_SEQ`,
`KILN_ROCM_FLASH_NATIVE_KEYSPLIT`,
`KILN_ROCM_FLASH_NATIVE_KEYSPLIT_MIN_SEQ`,
`KILN_ROCM_FLASH_NATIVE_KEY_TILE`, `KILN_ROCM_FLASH_NATIVE_MAX_SEQ`,
`KILN_ROCM_FLASH_NATIVE_QUERY_TILE`,
`KILN_ROCM_FLASH_NATIVE_RECTANGULAR_CAUSAL`,
`KILN_ROCM_FLASH_NATIVE_SCALAR`, `KILN_ROCM_FLASH_NATIVE_SCALAR_FORCE`,
`KILN_ROCM_FLASH_NATIVE_SINGLE_MAX_SEQ`,
`KILN_ROCM_FLASH_NATIVE_STREAMING`,
`KILN_ROCM_FLASH_NATIVE_STREAMING_FORCE`,
`KILN_ROCM_FLASH_NATIVE_STREAMING_MIN_SEQ`,
`KILN_ROCM_FLASH_NATIVE_TILED`, `KILN_ROCM_FLASH_NATIVE_TILED_FORCE`,
`KILN_ROCM_FLASH_NATIVE_WMMA_QBLOCK`,
`KILN_ROCM_FLASH_NATIVE_WMMA_QBLOCK_MIN_SEQ`, `KILN_ROCM_FLASH_ONLINE`,
`KILN_ROCM_FLASH_ONLINE_BWD`,
`KILN_ROCM_FLASH_ONLINE_MATMUL_BATCH_GROUP`,
`KILN_ROCM_FLASH_USE_WMMA_GQA_R64K16`,
`KILN_ROCM_FLASH_USE_WMMA_GQA_R64K32`,
`KILN_ROCM_FLASH_USE_WMMA_GQA_R64K32_LOG2`,
`KILN_ROCM_FLASH_WMMA_GQA_R64K16_MIN_SEQ`,
`KILN_ROCM_FLASH_WMMA_GQA_R64K32_LOG2_MIN_SEQ`, and
`KILN_ROCM_FLASH_WMMA_GQA_R64K32_MIN_SEQ`. None is accepted as an alias.
Positive/negative/force combinations collapse into the closed profile;
cooperative sleeps and forced FFI synchronization are deleted; the unused
key-split/WMMA branches and numerically incorrect CK backward route are removed
rather than exposed as policy.

The generic
`KILN_DISABLE_GPU_TRAINING_MLP_CHUNKING`,
`KILN_GPU_TRAINING_MLP_CHUNK_TOKENS`, and
`KILN_SPLIT_Q_GATE_OUTPUT_CHUNK_FEATURES` controls remain migration inputs for
other backends only; ROCm ignores them and uses the complete typed policy.
The same ROCm boundary also ignores the still-live CUDA/Metal comparison
switches `KILN_DISABLE_FUSED_PAGED_DECODE`,
`KILN_DISABLE_FUSED_PAGED_DECODE_DYN_SEQLEN_BATCH`,
`KILN_FORCE_FUSED_PAGED_DECODE_DYN_SEQLEN_BATCH`,
`KILN_DISABLE_FUSED_CUDA_MLP_SILU_MUL`,
`KILN_DISABLE_FUSED_MLP_GATE_UP_PREFILL`, and
`KILN_DISABLE_RMSNORM_KERNEL`; their ROCm decisions are leaves of the typed
profile instead.

`accelerator.full_attention_score_budget_mib` replaces the undocumented
`KILN_FULL_ATTN_SCORE_BUDGET_MB`, `KILN_ROCM_FLASH_SCORE_BUDGET_MB`,
`KILN_FULL_ATTN_SCORE_TILE_MAX_ELEMENTS`,
`KILN_ROCM_FULL_ATTN_SCORE_ELEMENT_CAP`,
`KILN_ROCM_FLASH_SCORE_TILE_MAX_ELEMENTS`,
`KILN_ROCM_FLASH_ONLINE_SCORE_BUDGET_MB`,
`KILN_ROCM_FLASH_ONLINE_QUERY_TILE`, and
`KILN_ROCM_FLASH_ONLINE_KEY_TILE` controls. Those names are intentionally not
aliases. Previously, model and ROCm flash paths parsed different permissive
values during execution and recomputed score budgets from changing free-memory
snapshots, permitting route geometry to change between layers or requests.
Policy v12 fixes one bounded ceiling before execution; the ROCm allocator
governor may still reject a planned operation when its exact working set is no
longer admissible, but it cannot silently shrink or expand that plan.

### Vulkan device policy

`kiln.vulkan-device-policy.v1` is the model/kernel boundary installed from the
two typed fields above before the first logical Vulkan device is created. A
same-value reinstall is idempotent; a conflicting late owner fails closed. The
selected `VulkanDevice` retains its physical enumeration index for diagnostics.
The removed `GGML_VK_VISIBLE_DEVICES` compatibility behavior is not a Vulkan
loader contract and is not accepted as a Kiln alias. Use the typed field or its
mechanically derived canonical environment name. External loader/driver
selectors such as `MESA_VK_DEVICE_SELECT`, `DRI_PRIME`, and Vulkan ICD selectors
remain identity remaps and cause device-scoped memory-probe validation to fail
closed rather than claim the wrong physical device.

### Vulkan kernel policy

Vulkan kernel selection is an immutable implementation contract, not a public
configuration surface. Product execution uses
`kiln.vulkan-kernel-policy.v3`, defined by the typed
`QUALIFIED_VULKAN_KERNEL_POLICY` object before any dispatch. There is no TOML,
CLI, request, or environment override for its leaves. Changing one requires a
reviewed source change, a new policy version, backend/oracle parity, and new
Vulkan performance and soak receipts.

The qualified policy fixes these exact route decisions:

| Family | Qualified selection |
|---|---|
| Model route availability | GDN, GDN prefill input projection, fused GDN gates/norm, single-submit GDN chunkwise prefill, fused conv1d prefill and its single-submit path, recurrent unexpanded Q/K plus Q/K norm, fused batch GDN state, linear decode and batched argmax, full-attention QKV, SDPA prefill, paged decode with GPU block-table gather, fused MLP decode, resident decode, and final GDN state-readback elision are enabled. Every resident-decode entry point uses the same policy leaf. |
| Packed model weights | BF16-packed linear, GDN input-projection, full-attention QKV, and MLP decode weights are enabled. The MLP BF16 gate/up with F32 down route is enabled. |
| Quarantined or slower model routes | Full-chunk GDN, single-token fused conv1d update, GDN forward substitution, batch-one fused GDN decode, the standalone MLP gate/up route, decode recurrent-state residency, resumable-prefill recurrent-state residency, fallback after a failed/disabled single-submit GDN chunkwise route, and the CPU-bridged/generic-tensor native Vulkan RMSNorm experiment are disabled. The device-resident model RMSNorm route remains enabled. |
| Dispatch safety | Vulkan linear submissions are capped at exactly 20,000,000,000 estimated FLOP and oversized work is sub-chunked. The former floating-point environment parser and its zero-means-unbounded escape are removed. |
| Bounded operator tiling | Flash attention uses a 2048-row tile and a 10,000,000-element row-work budget. Frozen-BF16-weight matmul uses 128-row tiles. Generic elementwise work uses 1,048,576 elements per dispatch and exponentiation uses 65,536. All are positive immutable policy leaves. |
| BF16 MLP | Four-row gate/up at batch 8, four-row down at batch 16, four-row F32 down at batch 8, and eight-row kernels at batch 256; all are enabled. |
| BF16 linear | Four-row kernels at batch 16 and eight-row kernels at batch 64; both are enabled. |
| BF16 full-attention QKV | Four-row kernels at batch 2 and eight-row kernels at batch 64; both are enabled. |
| GDN input projection | Pair QKV/Z, two-row grouping, and four-row grouping are enabled; eight-row grouping is disabled. Four-row selection starts at batch 16 and its eight-row threshold remains 64 for a future qualified policy. |
| GDN recurrence | Single-submit and parallel reduction are enabled. Host-visible recurrence is enabled for batch one and disabled for batched state. Q/K-normalization recurrence fusion is enabled; input-projection/conv-split fusion is disabled. |
| Submission and transfer fusion | Paged attention, Qwen RMSNorm, GDN gates/norm/input projection, MLP gate/up, causal conv1d, full-attention QKV, linear decode/argmax, chained MLP dispatch/transfer, and the applicable batched upload/transfer routes are enabled. |
| Paged-attention split-K | Batch one uses 32 chunks. Batch 16 or wider uses 4. Smaller batches use 2 at 64 or more blocks per sequence and 4 otherwise, with an absolute 256-chunk bound. |
| Prefill and profiling | Pair-row prefill matmul is enabled from batch 8. Kernel-stage and resident-decode profiling are disabled in product execution. |
| CPU references | GDN chunk preparation/scan/state-exit backward, triangular solve/transpose, and gated RMSNorm backward expose explicit CPU-reference functions for parity tests and offline diagnosis. Production entry points always use the GPU implementations; no process environment value can switch a live request to CPU or race another test. |

The former variables represented by this table, including the applicable
`KILN_DISABLE_VULKAN_*`, `KILN_ENABLE_VULKAN_*`, kernel-threshold, split-K,
kernel-stage profiling, model-route, packed-weight, recurrence-residency, and
resident-decode controls, were deleted without aliases or replacement fields.
They permitted route mixtures and permissive fallback below the typed startup
boundary, and the model duplicated a different copy of several thresholds.
Historical optimization documents may name those variables as experiment
evidence; they do not describe current runtime controls. The research
microbenchmark remains a separate executable surface and must not be treated as
product configuration.

The serving library and product kernel paths no longer read residual
`KILN_VK_*` tiling or CPU-fallback switches. The separate
`vulkan_decode_microbench` research executable also has no `KILN_*` input: its
complete experiment policy is a typed, fail-fast command line documented
below. It is not server configuration, and invoking it does not create a
product qualification receipt. Phase 8.1 remains open for non-Vulkan runtime
controls elsewhere in the repository.

### Standalone benchmark configuration

`kiln-bench` loads the same typed `KilnConfig` as the server. `--config` selects
the TOML file; in its absence normal config discovery applies. In particular,
the `[speculative]` section supplies the method, draft proposal count, and draft
depth. The three `--spec-*` arguments are explicit per-invocation overrides for
controlled A/B work. The executable has no direct `KILN_BENCH_*`,
`KILN_SPEC_*`, or qualification environment reads.

| Argument | Type and exact default | Validation and effect |
|---|---|---|
| `--config <path>` | path; normal discovery | Loads typed startup configuration before device selection. |
| `--model-path <path>` | path; required | Local model and tokenizer directory. |
| `--max-output-tokens <n>` | positive integer; `128` | Maximum generated tokens per request. |
| `--prompt-tokens <n>` | positive integer; `512` | Approximate generated benchmark prompt length. |
| `--training-steps <n>` | unsigned integer; `10` | SFT steps when training is not skipped. |
| `--skip-training` | flag; false | Omits the SFT benchmark. |
| `--paged` | flag; false | Uses the production paged-KV latency path. |
| `--latency-only` | flag; false | Stops after latency and emits empty throughput/training results. |
| `--latency-warmup-runs <n>` | unsigned integer; `0` | Runs unmeasured complete latency passes first. |
| `--seed <u64>` | integer; `42` | Selects prompt and deterministic sampler trajectory. |
| `--chat-template` | flag; false | Applies Qwen ChatML framing to the MTP arm. |
| `--prompt-subset <name>` | `all`, `gsm8k`, `humaneval`, or `c4`; `all` | Selects the MTP prompt pool. Unknown values are fatal. |
| `--temperature <f32>` | finite non-negative number; `0.0` | Sampling temperature for every benchmark arm. |
| `--spec-method <method>` | `off`, `skip_layer`, or `mtp`; typed `[speculative]` effective method | Overrides only this benchmark invocation. Unknown values are fatal. |
| `--spec-num-tokens <n>` | integer; typed `speculative.num_speculative_tokens` | Must pass the model speculative window validation, currently `1..=4`. |
| `--spec-draft-layers <n>` | integer; typed `speculative.draft_layers` | Must be positive and less than model layer count. |
| `--force-mtp` | flag; false | Bypasses benchmark shape routing; it does not enable serving. |
| `--log-tokens` | flag; false | Emits generated token IDs for the paged non-speculative arm. |
| `--log-itl` | flag; false | Emits each measured ITL for the paged non-speculative arm. |
| `--allow-experimental-speculative` | flag; false | Required for a non-`off` method. Acknowledges unsupported research behavior; it does not make a run qualification evidence. |

The speculative benchmark opt-in is deliberately not named `qualification`.
Only a declared workload executed by `scripts/qualification/run.py` and a
strictly checked retained receipt constitute qualification evidence. A direct
benchmark command, including one using the experimental opt-in, does not.
| `-v`, `--verbose`, `-vv` | count; `0` | Selects info or trace diagnostics. |
| `-q`, `--quiet` | flag; false | Selects warning diagnostics and wins over verbosity. |

`vulkan_decode_microbench` is an offline kernel research tool. It parses all
arguments before opening a Vulkan device. Unknown arguments/cases, empty list
items, malformed integers, and zero numeric values are fatal. Its complete base
and geometry surface is:

| Argument | Type and exact default | Effect |
|---|---|---|
| `--only <case,...>` | validated CSV; all 16 cases | Runs a subset of `full_attn_qkv`, `mlp_bf16_gu_f32_d`, `mlp_bf16w`, `linear_decode`, `causal_conv1d_update`, `gdn_gated_norm`, `qwen_rmsnorm`, `gdn_gates`, `gdn_in_proj`, `gdn_block_resident_batched`, `full_step_resident`, `full_step_resident_batched`, `full_token_resident_batched`, `full_token_resident_mixed_batched`, `full_token_resident_mixed_paged`, and `full_token_resident_paged`. |
| `--batches <n,...>` | positive-integer CSV; `1,4,8,16,32,64` | Batch sweep. |
| `--warmup-iters <n>` | positive integer; `10` | Untimed iterations per result. |
| `--timed-iters <n>` | positive integer; `30` | Iterations in each timed block. |
| `--repeats <n>` | positive integer; `5` | Timed blocks; the minimum block mean is reported. |
| `--attention-history <n>` | positive integer; `256` | Contiguous attention history for the mixed resident case. |
| `--paged-history <n>` | positive integer; `256` | Paged-attention history. |
| `--paged-block-size <n>` | positive integer; `16` | Paged KV block size. |

The microbenchmark kernel-policy arguments are explicit research A/B controls;
they never modify product policy:

| Argument | Exact default/effect |
|---|---|
| `--mlp-rows8-min-batch <n>` | `256` |
| `--mlp-gate-up-rows4-min-batch <n>` | `8` |
| `--mlp-down-rows4-min-batch <n>` | `16` |
| `--linear-rows8-min-batch <n>` | `64` |
| `--linear-rows4-min-batch <n>` | `16` |
| `--gdn-in-proj-rows8-min-batch <n>` | `64` |
| `--gdn-in-proj-rows4-min-batch <n>` | `16` |
| `--full-attn-qkv-rows8-min-batch <n>` | `64` |
| `--full-attn-qkv-rows4-min-batch <n>` | `2` |
| `--disable-linear-rows4`, `--disable-linear-rows8` | Both routes default enabled. |
| `--disable-mlp-rows8`, `--disable-mlp-gate-up-rows4`, `--disable-mlp-down-rows4` | All three routes default enabled. |
| `--disable-full-attn-qkv-rows4`, `--disable-full-attn-qkv-rows8` | Both routes default enabled. |
| `--disable-gdn-in-proj-pair-qkv-z`, `--disable-gdn-in-proj-row-pair`, `--disable-gdn-in-proj-row-quad` | All three routes default enabled. |
| `--enable-gdn-in-proj-row-octet` | Eight-row grouping defaults disabled. |
| `--enable-gdn-in-proj-conv-split-fusion` | Projection/conv fusion defaults disabled. |
| `--disable-gdn-qk-norm-recurrent-fusion` | Q/K-normalization recurrence fusion defaults enabled. |

### ROCm synchronization semantics

`legacy_host_barriers` is the qualified default. It preserves the historical
device-wide barriers around large hipBLASLt results and BF16/F32 cast
boundaries, the historical active-stream waits after eager tensor operations,
and a device-wide external-yield drain before generated progress is published.
Choosing it is intended to produce the same ordering as the pre-policy runtime,
with the new reasoned counters added.

`stream_ordered` is an explicit local-qualification mode. It omits only
barriers whose call site asserts that producer and consumer are FIFO-ordered on
the same HIP stream. It does not omit host readback, host-buffer lifetime,
in-place mutation, allocator reclaim, graph transition, error recovery, or
global-state drains. External yield waits for the primary producer stream
rather than the whole device. A replayed graph records completion on its
capture stream and makes the primary stream wait on that event before the
external-yield stream wait, so replay work is included.

### ROCm matmul route semantics

Both matmul fields default to `auto`, preserving the routes used by the accepted
Strix Halo ROCm receipts. The strided-batched guard uses per-row GEMMs for
large attention-shaped BF16/F32 batches where ROCm 7.2 hipBLASLt has returned
incorrect values. The BF16-output guard requests F32 output followed by an
on-device BF16 cast for large projections, large outputs, and tall low-rank
compression shapes where the native BF16 epilogue has returned non-finite
values. These are deterministic shape and dtype decisions; they do not inspect
live memory and cannot change after the primary ROCm context is created.

The explicit routes exist for controlled A/B qualification, require
`server.serving_profile = "experimental"`, and are reported with source
attribution by `kiln config`, `/v1/config`, health, and trusted debug state. The
former force/disable variables are warning compatibility aliases resolved once
by the typed startup registry. The two ad hoc trace variables were deleted;
they are not configuration and no matmul path reads the process environment.

### ROCm token-only LM-head contract

The `qualified` and `experimental_multiblock` ROCm kernel profiles pack the tied
BF16 embedding/LM-head rows once during model load and own both
`w8_sampled_lm_head` and `w8a8_sampled_lm_head`. These are immutable leaves of
`accelerator.rocm_kernel_profile`, not additional user settings. The
`portable_fallback` profile disables them and retains the ordinary BF16 LM head.

Greedy contiguous decode batches use W8A16 projection into an internal F32
score scratch, perform a stable lower-token-id argmax on the device, and copy
one token vector to the host. Sampled batches may mix greedy and stochastic
rows. Stochastic rows support `top_k` from 1 through 64 and apply the public
sampling contract in this order: sign-conditional repetition penalty, presence
and frequency penalties over unique generated-token counts, temperature,
top-k, stable softmax, min-p, top-p, and categorical selection. The qualified
profile quantizes each final-normalized activation row once and uses W8A8 for
the sampled projection; the same kernel contract has a W8A16 implementation for
profiles that retain W8 projection but do not select sampled W8A8. Parameter
and sparse-history uploads are ordered on the active HIP stream without an
extra trailing synchronization. The full vocabulary score scratch never leaves
the device; successful dispatch reads back only the `[batch]` I64 token tensor.

`temperature = 0` and positive-temperature `top_k = 1` are greedy and ignore
history penalties, matching `SamplingParams`. A stochastic `top_k = 0`, a
value above 64, an incompatible dtype/shape, active gradient tracking, a profile
that disables the route, or an unavailable packed LM head declines to the
existing full-logit sampler. The single-row unfiltered `top_k = 0` ROCm route
may use the existing full-distribution W8 Gumbel sampler. Invalid dimensions,
non-finite sampling parameters, non-positive repetition penalty, duplicate
row/token history, zero counts, and out-of-range row/token indices fail closed;
they are not coerced into a supported request.

Seeded categorical rows use the device SplitMix64 mapping defined by this ROCm
kernel. The same binary, device path, inputs, and seed replay exactly; the seed
does not promise the same token as Rust `StdRng`, Vulkan's device RNG, another
backend, or a later kernel contract. Use greedy decoding where a comparison
requires backend-independent bit-exact output.

`GET /health` and `GET /v1/health` expose process-lifetime route evidence at
`decode_runtime.rocm_w8_lm_head`: successful argmax/sample dispatches and rows,
sample history entries, W8A16/W8A8 sampled dispatches, failures, maximum batch
rows, and maximum sampled top-k. `/metrics` exports the same fixed-cardinality
state as `kiln_rocm_w8_lm_head_dispatches_total{operation}`,
`kiln_rocm_w8_lm_head_rows_total{operation}`,
`kiln_rocm_w8_lm_head_sample_dispatches_total{activation}`,
`kiln_rocm_w8_lm_head_sample_history_entries_total`,
`kiln_rocm_w8_lm_head_dispatch_failures_total`,
`kiln_rocm_w8_lm_head_max_batch_rows`, and
`kiln_rocm_w8_lm_head_max_sample_top_k`. Reading either surface only loads host
atomics. It does not probe or synchronize the accelerator.

These counters describe route ownership, not total response ownership. Prefill
emits the first completion token for each admitted request before the request
becomes a decode-continuation row. Consequently an eight-request sampled wave
with 32 completions per request has 256 response tokens but exactly 248 fused
sample rows. Qualification requires both exact counts: the response oracle and
usage fields cover all 256 tokens, while the LM-head counter covers the 248
decode-continuation tokens. Treating the eight prefill-owned tokens as missing
fused work is a contract error.

The primary ROCm context stores the immutable policy. A second caller asking
for a different policy on the same device receives a startup error naming the
already-installed and requested policies. This prevents two model/runtime
owners from silently mixing synchronization disciplines.

Synchronization telemetry has 23 fixed reason values. `/health` reports device
wait count, stream wait count, waited nanoseconds, and skipped count for every
reason under `decode_runtime.rocm_synchronization`. The same object exposes
`cleanup_quarantined`: when true, a fatal execution or synchronization failure
made device state unsafe. The flag is shared by every context for that device
ordinal and is process-lifetime sticky: a later recovery drain may collect
diagnostics, but cannot re-enable execution or cleanup. New ROCm execution is
rejected, possibly in-flight HIP resources are retained instead of being
destroyed unsafely, backend admission fails closed, and the process must be
restarted. Active ROCm telemetry becoming unavailable is treated as the same
fail-closed health condition rather than as a false quarantine value. The
central server health gate observes and latches this state before adapter,
training, import, or other backend mutation admission, so those paths cannot
slip through before the model runner notices the fault. This is live state, so it is
reported by health and trusted debug rather than `/v1/config`; the latter remains
the immutable policy authority. `/metrics` exports the same
fixed dimensions as `kiln_rocm_synchronizations_total`,
`kiln_rocm_synchronization_wait_seconds_total`, and
`kiln_rocm_synchronization_skipped_total`, plus policy, availability, and
`kiln_rocm_cleanup_quarantined` gauges. Treat the quarantine gauge as unknown
unless `kiln_rocm_synchronization_telemetry_available` is `1`.
Reading these surfaces loads atomics only; it never synchronizes the device or
probes the driver.

`capture_rollback` and `error_recovery` are deliberately different reasons.
After a native capture error, `capture_rollback` may settle the device while
execution admission remains open; eager continuation is allowed only after the
runner also proves that its logical recurrent state was restored. A settlement
failure, a logical-rollback failure, or an armed/unclassified capture guard
leaving scope publishes the process-lifetime STOP state first, then uses
`error_recovery` only as a diagnostic drain. That fatal path remains
cleanup-quarantined even if the later drain succeeds and always requires a
process restart.

Stream-ordered mode is not promoted by configuration alone. Qualify legacy and
stream-ordered runs on the same source tree using token/logit parity,
throughput at concurrency 1/8/16/32/64, p50/p95/p99/max inter-token latency,
reasoned synchronization time, and stable memory plateaus before changing the
default.

### ROCm graph lifecycle

- `disabled` bypasses graph-shaped warmup, capture, and replay.
- `warmup_then_eager` runs one graph-shaped eager warmup, never captures, and
  remains eager. It exists for controlled graph-state-machine comparisons.
- `lazy_capture_replay` warms eagerly and may capture/replay eligible decode
  shapes while serving. Capture/fallback/replay state remains visible in
  `/health`; use this only in the experimental profile.
- `profile` is the recommended default because the serving profile remains the
  authority: stable and maintenance stay eager, while experimental selects
  lazy capture/replay.

Entry count and retained bytes are independent hard limits. Byte accounting
deduplicates physical ROCm allocations by device pointer, counts graph-stable
I/O, freeze-pointer capture arenas, each private stream's actual hipBLASLt
workspace, and persistent recurrent/conv slot state once per owner. Health,
trusted debug, and Prometheus split those categories and report peak bytes,
evictions, admission rejections, and the number of opaque HIP graph, exec,
stream, and event objects. The byte budget deliberately does not pretend that
HIP exposes the allocator overhead of those opaque objects; live driver memory
pressure remains a separate complementary signal.

Before native capture, the runner reclaims eligible idle owners at entry/byte
saturation. If those cannot make room, it computes a deterministic fair-LRU
order across active owners. The incoming candidate counts toward its owner's
projected share; each step selects the most represented owner, then its oldest
exact geometry with stable owner/geometry tie-breakers. Admission retires only
the minimum entries needed and preserves at least one graph per active owner
after the candidate, plus every live recurrent-state slot, row assignment, and
decode-continuity timeline. No active graph is retired while unused global
entry and byte headroom remains. The runner then performs one settled warm
Record pass, measures the candidate's exact queryable allocation identities,
rechecks matching-device pressure, and atomically reserves global governor
headroom. Native capture begins only after that sequence admits it. A governor
device-selector mismatch fails closed rather than consuming another
accelerator's headroom. The Record pass necessarily allocates the candidate
before its exact size is knowable. Phase telemetry independent of the model and
graph-runner locks plus last/peak transient-candidate bytes make that residual
exposure observable. Before any capture-parity snapshot allocation, a separate
matching-device reservation covers two GDN state snapshots, the eager hidden
copy, all current K/V rows, one current-row gather, and the largest exact U8
comparison mask. The later exact candidate reservation overlaps it through
native capture, so the transient high-water mark includes both working sets. A
candidate that alone exceeds the byte budget or cannot
be accounted exactly makes its geometry non-capture-safe for this runner.
Aggregate byte-budget suppression clears after ownership or budget relief;
global-reservation denial retries after the matching device reports enough
cached headroom. A selector mismatch remains fail-closed rather than consuming
another accelerator's budget. None repeats a full warm pass on every token.

Ordinary budget eviction is deterministic least-recently-used eviction of
idle owners as a unit, including every graph for the owner and its retained
slot state. Those owners are always considered before fair active relief. At
actual saturation, boundary `pre_capture_active_fair_share` settles and retires
the selected active graph entries in one narrow transaction. A cache hit is
never an admission candidate, and the algorithm never removes an active
owner's last projected graph merely to grow another owner's share.
Every completed transaction emits `rocm_graph_cache_eviction` with its
`boundary`, typed `reason`, removed graph/slot counts, released requested bytes,
and `active_graph_only_owner_count`. A nonzero active count identifies the
settled prior-geometry retirement path rather than an idle-owner eviction.
Moderate pressure permits retained replay but blocks cache growth. Tight
pressure additionally reclaims eligible idle owners while preserving an active
cache hit. Critical or unavailable pressure disables replay,
reclaims all graphs only after device settlement, and uses eager execution only
if both pre-drop and post-drop settlement prove cleanup safe. A destructor,
async-free, or settlement failure activates the process-lifetime device
quarantine; no reclaimed-byte success is reported and no eager retry follows.

### ROCm graph observability contract

`GET /v1/config` and trusted `GET /v1/debug/model-state` expose two independent
live values and a reason beside each one:

- `rocm_graphs` is the nonblocking full cache/counter snapshot, with
  `rocm_graphs_unavailable_reason`.
- `rocm_graph_telemetry` is the current/completed phase and transient-candidate
  snapshot, independent of the model and graph-runner locks, with
  `rocm_graph_telemetry_unavailable_reason`.

Each value is either an object with a null reason or `null` with exactly one of
`backend_without_graph_runner`, `model_runner_busy`,
`model_runner_lock_poisoned`, `graph_runner_busy`, or
`graph_runner_lock_poisoned`. The full snapshot can use any of those five. The
phase handle is stored outside both runner locks, so a real backend retains live
phase telemetry through model-runner and graph-runner contention or poison;
currently only `backend_without_graph_runner` makes that value null. The phase
reason family nevertheless uses the same closed schema. Do not interpret any
null value as an empty cache or zero activity.

Health flattens both authorities into `decode_runtime.rocm_graphs` for an
operational view. Its `state` is `enabled` or `disabled` when full statistics
are present, `busy` only for `model_runner_busy` or `graph_runner_busy`, and
`unavailable` for the no-runner and poisoned-lock reasons. `unavailable_reason`
belongs to the full statistics. `phase_telemetry_available` and
`phase_telemetry_unavailable_reason` independently govern `current_phase`,
`current_phase_elapsed_micros`, five phase summary objects, and last/peak
transient candidate bytes. Every unavailable field is serialized as `null`;
the server never fabricates a zero snapshot.

For a coherent point-in-time full snapshot, `rocm_graphs` also serializes the
five completed phase summaries and last/peak transient bytes. The separate
`rocm_graph_telemetry` object is the authoritative current-phase source and
continues advancing when full statistics cannot acquire either runner lock.
Health sources its phase and transient fields from that separate channel.

The phase names are exactly `pre_candidate_headroom`, `candidate_warm`,
`pre_native_reservation`, `native_capture`, and
`rejected_candidate_cleanup`. Each completed phase object contains `calls`,
`slow`, `total_duration_micros`, and `max_duration_micros`; `slow` means at
least 100 ms. `current_phase_elapsed_micros` uses a monotonic clock while a
phase is active. `native_capture` remains active through capture, the settled
first launch, defensive cache admission/publication, and the blocking committed
governor debit. `rejected_candidate_cleanup` begins only when an unretained
candidate enters destruction and settlement. `last_transient_candidate_bytes` and
`peak_transient_candidate_bytes` measure the peak matching-device reservation
for the exact deduplicated pre-admission candidate plus the capture-parity
snapshot/gather/mask working set. Already-owned recurrent slot state and opaque
native objects remain excluded. These values are not retained-cache bytes; only
the candidate subset is committed when admission retains the graph.

Native ROCm HIP-graph capture supports both single-row decode and the contiguous
BF16 multi-row route. Batched graphs are keyed by row count and bucketed
attention geometry. They retain graph-stable token IDs, positions, RoPE tables,
block tables, sequence lengths, KV slots, per-layer attention scratch, output
hidden state, and any GDN recurrent state. Final normalization, LM head, and
sampling remain eager after the graph. A width-four graph therefore reports one
active graph slot across changing request cohorts; it is not an idle single-row
owner. `multi_row_batch_unsupported` remains in the closed schema so historical
receipts stay valid, but a supported current batched route must leave it zero
and show real capture/replay activity instead.

Before a new batched graph becomes replayable, its first launch is compared
exactly with the already-required eager warm pass. The comparison covers hidden
output, every recurrent and convolution state tensor, and the current K/V rows
for every full-attention layer. Native same-dtype equality produces one bounded
U8 mask at a time and reads back only the reduced scalar. Structured event
`rocm_graph_capture_parity_check` reports `passed`, `failed`, or `error`, whether
comparison completed, compared bytes, duration, hidden equality and the first
recurrent, convolution, K, and V layer mismatch, or the comparison error.
Comparison failure follows the capture rollback/settlement contract, counts as
`capture_failure`, disables further graph execution for the runner, and reaches
eager only after state restoration and containment succeed.

Health makes the admission proof durable with process-lifetime
`batched_capture_attempts`, `batched_capture_successes`,
`batched_capture_deferrals`, `batched_capture_failures`,
`capture_parity_checks`, `capture_parity_passes`,
`capture_parity_failures`, `capture_parity_errors`,
`capture_parity_compared_bytes` for completed equality operations, and
`capture_parity_duration_micros`. Batched attempts and parity checks each
reconcile to their closed outcome sets. Every
successful batched capture has a prior passed check, so cumulative successes
cannot exceed cumulative passes. Equality is required for a clean qualification
window; a passed check followed by a later cache-admission error is retained as
a pass plus capture failure rather than falsified into a successful admission.

The eager-fallback reasons are exactly `multi_row_batch_unsupported`,
`cold_cache_host_round_trip`,
`persistent_host_round_trip`, `shape_dependent_attention`,
`graph_cache_capacity`, `graph_cache_byte_budget`,
`graph_accounting_incomplete`, `moderate_memory_pressure`,
`tight_memory_pressure`, `critical_memory_pressure`,
`memory_reservation_denied`, `memory_governor_selector_mismatch`,
`capture_failure`, and `replay_failure`. The full snapshot reports one counter
per reason plus total, slow, total-duration, and maximum-duration values.

Prometheus always emits two availability gauges and two closed one-hot reason
families. A zero `kiln_rocm_graph_telemetry_available` is explained by
`kiln_rocm_graph_snapshot_unavailable{reason}` and omits only the full
cache/counter families. A zero
`kiln_rocm_graph_phase_telemetry_available` is explained by
`kiln_rocm_graph_phase_telemetry_unavailable{reason}` and omits only current
phase, phase latency, and transient-candidate families. Both reason labels use
the same five-value set above. Phase availability remains one for a real backend
while full snapshot availability is zero during either runner-lock contention
or poison. Full
families cover state; cache, slot, owner, retained-byte and opaque-object
gauges; admissions, evictions and their four causes; three post-capture
rejection reasons; five pre-capture skip reasons; capture/replay outcomes;
batched capture attempts/outcomes; first-launch parity checks/outcomes, logical
bytes, and duration; and all 14 fallback reasons and latency. The new
fixed-cardinality families are
`kiln_rocm_graph_batched_capture_attempts_total`,
`kiln_rocm_graph_batched_capture_outcomes_total{outcome="success|deferred|failure"}`,
`kiln_rocm_graph_capture_parity_checks_total`,
`kiln_rocm_graph_capture_parity_outcomes_total{outcome="passed|failed|error"}`,
`kiln_rocm_graph_capture_parity_compared_bytes_total`, and
`kiln_rocm_graph_capture_parity_duration_seconds_total`. Live families cover
the one-hot current phase, active elapsed seconds, calls/slow/total/max duration
for all five phases, and last/peak transient bytes. Every label set is closed;
request, shape, allocation, and configured-byte values never become labels.

Graph-stable paged metadata is now a correctness invariant rather than a knob.
The runtime no longer reads `KILN_ROCM_GRAPH_STABLE_PAGED_METADATA`, graph
cache/capture flags, matmul synchronization thresholds, or full-attention
handoff flags inside decode. The three historical graph variables listed in
the table are parsed once by the typed startup loader. Setting both historical
graph-mode aliases is rejected even when their booleans happen to agree; use
the canonical enum to state one unambiguous lifecycle.

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
| `batching.prefill_admission_quantum` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_PREFILL_ADMISSION_QUANTUM` (implemented) | `KILN_BATCH_PREFILL_ADMISSION_QUANTUM` (deprecated compatibility) | An integer must be `1..=65536` and caps how many queued prompts the actor admits in one cycle before returning to decode. `auto` is case-insensitive and uses the backend policy below. The selected value is then clamped to `1..=effective max_decode_batch`; the diagnostics name `effective_decode_width` as final authority when it performs that clamp. With non-burst admission, total active capacity is effective decode width plus this staging quantum, capped internally at four staging slots. The Strix Halo Vulkan development-soak candidate sets `2`, admitting an equal pair while retaining a four-request active ceiling. Restart required. |
| `batching.actor_cycle_idle_ms` | unsigned integer milliseconds; `0` | `KILN_BATCHING_ACTOR_CYCLE_IDLE_MS` (implemented) | none | `0..=60000`. Zero preserves the unpaced actor. A nonzero value inserts one intentional cooperative wait after an actor cycle that advanced prefill or decode, only after synchronous accelerator work has returned. The actor polls control commands at intervals no longer than 5 ms, so shutdown remains responsive, and the independent response-delivery worker and HTTP process remain live. This deliberately trades request throughput and inter-token latency for a lower sustained accelerator duty cycle; it is not a temperature controller and never changes itself from a live sensor. Config, health, debug, Prometheus, and serving-benchmark receipts expose the policy and observed waits. Restart required. |
| `batching.direct_decode_rendezvous_mode` | string enum; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MODE` (implemented) | `KILN_DECODE_BATCHER` (deprecated compatibility) | Selects only the fallback worker used by actor-absent direct streaming effectively-greedy requests. TOML accepts `auto`, `enabled`, or `disabled`, case-insensitively; environment input also accepts the strict boolean spellings. `auto` enables the worker on every real backend. Sampled requests, non-streaming requests, and every route using the batching actor bypass this worker. Restart required. |
| `batching.direct_decode_rendezvous_max_batch` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MAX_BATCH` (implemented) | `KILN_DECODE_BATCH_MAX` (deprecated compatibility) | An explicit integer must be `1..=65536`. `auto` uses the backend policy below. Either selection is clamped to the already-resolved effective decode width; diagnostics use `effective_decode_width` when that ceiling wins. Restart required. |
| `batching.direct_decode_rendezvous_wait_us` | `"auto"` or unsigned integer; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_WAIT_US` (implemented) | `KILN_DECODE_BATCH_WAIT_US` (deprecated compatibility) | An explicit value is any non-negative `u64` number of microseconds, including `0`; `auto` uses backend policy. A negative, overflowing, malformed, or non-UTF-8 environment value stops startup. In particular, malformed legacy `KILN_DECODE_BATCH_WAIT_US` now fails startup instead of silently becoming zero. Restart required. |
| `batching.direct_decode_rendezvous_mixed_seq_lens` | `"auto"` or boolean; `"auto"` | `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MIXED_SEQ_LENS` (implemented) | `KILN_DECODE_BATCH_MIXED_SEQ` (deprecated compatibility) | TOML accepts only the string `auto` or a native boolean; quoted boolean strings are rejected. Environment input accepts `auto` or the strict boolean spellings. The value controls whether one compatible fallback cohort may contain different decode positions. Restart required. |

### Backend-owned `auto` policy

`auto` preserves backend qualification rather than imposing one cross-device
default. The matrix is evaluated against the final effective decode width, not
the unconstrained backend maximum.

| Backend | `batching.mode = "auto"` | Backend decode width before server constraints | `prefill_admission_quantum = "auto"` | `burst_prefill_admission` diagnostic | `actor_prefill_tile_alignment_required` diagnostic |
|---|---:|---:|---:|---:|---:|
| CUDA | enabled | 8 | effective decode width | true | false |
| ROCm | enabled | 8 | 4, clamped to effective width | false | true |
| Vulkan | enabled | 64 | effective decode width | false | false |
| Metal | disabled | 8 | 4, clamped to effective width | false | false |
| CPU/other | enabled | 8 | 4, clamped to effective width | false | false |

`burst_prefill_admission` is backend-owned and intentionally has no TOML or
public environment field. On CUDA it lets admission refill the decode width in
one actor turn. It is reported with the four primary-actor settings so an observed
CUDA/Vulkan/ROCm difference is not mistaken for a hidden runtime override.
`actor_prefill_tile_alignment_required` is also backend-owned. ROCm sets it
because local deterministic parity proved that a 64-token actor partition and
monolithic direct prefill could select different greedy tokens. Startup now
requires the actor ceiling, direct streaming crossover, combined token budget,
and effective decode width to preserve one 256-token numerical partition.

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
      "actor_cycle_idle": {
        "milliseconds": 0,
        "source": "default",
        "enabled": false,
        "command_poll_milliseconds": 5
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
      "burst_prefill_admission": false,
      "actor_prefill_tile_alignment_required": true
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
A nonzero actor-cycle idle has no backend-derived effective value: its exact
configured milliseconds and source are immutable for the process. The actor
applies it only after a cycle performed model work, not while idle, queued, or
waiting for a request. Because the delay begins only after synchronous
accelerator return, it does not suspend outstanding device work. It is
intentionally separate from external thermal containment: operators must still
use a hardware guard for qualification and measure the resulting throughput and
ITL cost.
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

When the actor exists, the live health/debug snapshot reports
`actor_cycle_idle_ms`, `actor_cycle_idle_source`, `actor_cycle_idle_active`,
`actor_cycle_idle_count`, `total_actor_cycle_idle_ms`, and
`max_actor_cycle_idle_ms`. Prometheus exports the same process-lifetime state as
`kiln_batching_engine_actor_cycle_idle_configured_seconds`,
`kiln_batching_engine_actor_cycle_idle_active`,
`kiln_batching_engine_actor_cycle_idles_total`,
`kiln_batching_engine_actor_cycle_idle_seconds_total`, and
`kiln_batching_engine_actor_cycle_idle_max_seconds`. These counters make an
intentional duty-cycle delay distinguishable from an unexplained inference
stall. Serving benchmark driver v13 snapshots an idle boundary outside each
measured request window and fails its `actor_cycle_idle_accounted` gate when the
source, count, elapsed time, maximum, or final active state contradicts the
configured policy.

`kiln config --file <path>` validates and prints all nine `[batching]` startup
values: actor mode, rowwise decode, prefix-aware admission, prefill quantum,
actor-cycle idle, direct rendezvous mode, maximum batch, wait microseconds, and
mixed-sequence policy. It also prints the combined actor-cycle budget, prompt-token and layer
ceilings, configured decode-width ceiling, streaming mode, threshold, base,
tape, and detached full-attention tiles with source provenance. It does not
construct an accelerator. Add `--backend rocm` (or another target) to resolve
the effective actor selection, alignment requirement, decode width, streaming
dispatch, and tiles from the same backend policies used at serve startup. The
command applies the real actor-prefill contract and exits nonzero before any
hardware access when the combination is invalid. Without `--backend`, it names
those values as a serve-startup contract instead of guessing a target. Inspect
`/v1/config` after restart for actual selected-device and live route facts.

## `[model]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `model.path` | optional string; omitted (`None`) | `KILN_MODEL_PATH` (implemented) | `KILN_MODEL_PATH` | Must be non-empty when set. Omitted starts the server in mock mode; a real model path enables real inference. |
| `model.model_id` | string; `"Qwen/Qwen3.5-4B"` | `KILN_MODEL_MODEL_ID` (implemented) | `KILN_MODEL_ID` (deprecated compatibility) | Must be non-empty. Used for model/tokenizer identity and as the source of the default served id. Kiln still applies its built-in Qwen3.5-4B runtime profile. |
| `model.tokenizer_path` | optional string; omitted (`None`) | `KILN_MODEL_TOKENIZER_PATH` (implemented) | `KILN_TOKENIZER_PATH` (deprecated compatibility) | Must be non-empty when set. |
| `model.adapter_dir` | optional string; omitted (`None`) | `KILN_MODEL_ADAPTER_DIR` (implemented) | `KILN_ADAPTER_DIR` (deprecated compatibility) | Must be non-empty when set. For the Qwen3.5-4B profile, omission resolves to `<model.path>/adapters`. |
| `model.snapshot_dir` | optional string; omitted (`None`) | `KILN_MODEL_SNAPSHOT_DIR` (implemented) | `KILN_MODEL_SNAPSHOT_DIR` | Must be non-empty when set. The environment alias uniquely treats an empty or whitespace-only value as a request to clear the TOML value. Without a value, Kiln tries a location beside the model and then the system temporary directory. |
| `model.checkpoint_read_mib_per_second` | optional unsigned integer MiB/s; omitted (`None`) | `KILN_MODEL_CHECKPOINT_READ_MIB_PER_SECOND` (implemented) | `KILN_MODEL_CHECKPOINT_READ_MIB_PER_SECOND` | `1..=16384` when set. Independently bounds the private snapshot copy, initial full content verification, and post-upload full verification. Reflinked bytes are not charged as reads. Shutdown is checked between bounded chunks and at most every 25 ms during waits. Omission removes rate limiting but preserves cancellation. Applies to real-model startup on every backend, requires restart, and is never active during inference. `GET /v1/config.model_startup.checkpoint_read` reports all three phase observations. |
| `model.accelerator_weight_upload_mib_per_second` | optional unsigned integer MiB/s; omitted (`None`) | `KILN_MODEL_ACCELERATOR_WEIGHT_UPLOAD_MIB_PER_SECOND` (implemented) | `KILN_MODEL_ACCELERATOR_WEIGHT_UPLOAD_MIB_PER_SECOND` | `1..=16384` when set. Reserves the cumulative eager base-model source-byte schedule before the base group and each layer, then checks it again after the unit. The base group adds shutdown boundaries after embedding upload, transpose, pack, final norm, and rotary initialization. Transforms can add device work, so this is not a bus-throughput cap. The current backend operation is not interruptible; reservation waits poll cancellation every 25 ms. Omission removes rate limiting but preserves boundary cancellation. Inapplicable to mock and CPU-only execution. Startup-only, restart required, and never active during inference. `GET /v1/config.model_startup.accelerator_weight_upload` reports reserved/completed bytes and layers. |
| `model.vulkan_decode_weight_prewarm` | boolean; `true` | `KILN_MODEL_VULKAN_DECODE_WEIGHT_PREWARM` (implemented) | `KILN_MODEL_VULKAN_DECODE_WEIGHT_PREWARM` | Populates backend-private Vulkan decode-weight caches during startup. Disable only to trade first-request latency for lower startup work. Restart required. |
| `model.vulkan_decode_weight_prewarm_mib_per_second` | unsigned integer MiB/s; `256` | `KILN_MODEL_VULKAN_DECODE_WEIGHT_PREWARM_MIB_PER_SECOND` (implemented) | `KILN_MODEL_VULKAN_DECODE_WEIGHT_PREWARM_MIB_PER_SECOND` | `1..=16384`. Bounds the average Vulkan decode-weight cache materialization rate. Pacing checks shutdown at least every 25 ms between uploads. Restart required. |
| `model.served_model_id` | optional string; omitted (`None`) | `KILN_MODEL_SERVED_MODEL_ID` (implemented) | `KILN_SERVED_MODEL_ID` (deprecated compatibility) | Must be non-empty when set. Otherwise the effective id is the final slash-separated component of `model.model_id` (`Qwen3.5-4B` by default). `serve --served-model-id` applies a typed, validated override after environment resolution. |

## `[memory]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `memory.num_blocks` | optional unsigned integer; omitted (`None`) | `KILN_MEMORY_NUM_BLOCKS` (implemented) | `KILN_NUM_BLOCKS` (deprecated compatibility) | Must be greater than zero when set. Omission invokes backend-aware automatic KV-block sizing. |
| `memory.gpu_memory_gb` | optional finite number; omitted (`None`) | `KILN_MEMORY_GPU_MEMORY_GB` (implemented) | `KILN_GPU_MEMORY_GB` (deprecated compatibility) | Must be finite, greater than zero, and representable as bytes. Units are GiB. This is a capacity cap, not a hardware override: it may reduce the detected safe capacity but never expands physical VRAM, host-backed unified memory, or a cgroup-bounded capacity. A request above the safe detected capacity is clamped down. |
| `memory.inference_memory_fraction` | finite number; `0.7` | `KILN_MEMORY_INFERENCE_MEMORY_FRACTION` (implemented) | `KILN_INFERENCE_MEMORY_FRACTION` (deprecated compatibility) | Loader validation accepts `0.0..=1.0`; real-state construction clamps the configured value to `0.1..=1.0` before KV sizing. The remainder is available to the training budget. |
| `memory.training_memory_gb` | optional finite number; omitted (`None`) | `KILN_MEMORY_TRAINING_MEMORY_GB` (implemented) | `KILN_TRAINING_MEMORY_GB` (deprecated compatibility) | Must be finite, greater than zero, and representable as bytes when set. Optional training-budget cap in GiB; it can reduce but never expand the capacity remaining after resident model and KV allocations. |
| `memory.vulkan_buffer_pool_gb` | finite number; `3.0` | `KILN_MEMORY_VULKAN_BUFFER_POOL_GB` (implemented) | `KILN_VULKAN_BUFFER_POOL_GB` (deprecated compatibility) | Vulkan-only process-wide cap, in GiB, on idle scratch buffers retained for reuse. Must be finite, non-negative, and representable as bytes; `0` disables retention. Active operations may allocate beyond the cap, but overflow buffers are freed when their final caller drops. A new cache entry evicts the oldest idle buffers before admission, and pressure reclaim releases idle entries under exclusive GPU coordination. `/v1/config`, `/health`, and Prometheus expose the cap, retained/free/borrowed bytes, hits/misses by allocation route, evictions, and uncached overflow. Health also exposes one bounded last-miss record with requested and bucket bytes plus the source callsite. The default remains `3.0`; the Strix Halo Vulkan development-soak candidate explicitly sets `3.5` after the smaller cap showed deterministic eviction churn. |
| `memory.floor_gb` | finite number; `1.0` | `KILN_MEMORY_FLOOR_GB` (implemented) | `KILN_MEMORY_FLOOR_GB` | Must be finite, non-negative, representable as bytes, and strictly smaller than the selected accelerator's effective capacity after `memory.gpu_memory_gb` is applied. Units are GiB. Accelerator startup rejects an equal or larger floor before model upload and reports both configured and effective byte values. The process-wide governor subtracts this additional floor, then outstanding soft reservations, from live free memory when computing allocation headroom. On unified-memory devices it is separate from the physical-memory reserve applied during safe-capacity detection. |
| `memory.probe_ms` | unsigned integer; `500` | `KILN_MEMORY_PROBE_MS` (implemented) | `KILN_MEMORY_PROBE_MS` | Must be greater than zero. Sets the background memory-sampler cadence. Request, inference, health, and metrics paths read only the published sample and never run a driver/OS probe synchronously. Cached admission fails closed when the sample is older than `max(5000 ms, 4 * probe_ms)`, the latest probe failed, or a required sampler is not running. An explicit refresh after a material allocation or release bypasses the cadence. |
| `memory.reclaim_mode` | string enum; `"off"` | `KILN_MEMORY_RECLAIM_MODE` (implemented) | `KILN_MEMORY_RECLAIM_MODE` | Exactly `off`, `on-demand`, or `automatic`, case-insensitive with surrounding whitespace ignored. `off` prevents execution of registered allocator reclaim hooks; `on-demand` permits explicit pressure and allocation-retry reclaim calls; `automatic` also permits the background pressure monitor. The immutable serving profile remains authoritative: a profile with allocator reclaim disabled keeps the effective mode off and does not start the monitor. |
| `memory.kv_autoscale` | boolean; `true` | `KILN_MEMORY_KV_AUTOSCALE` (implemented) | `KILN_KV_AUTOSCALE` (deprecated compatibility) | Requests the pressure-driven physical KV-cache control loop. The serving profile and backend remain authoritative: stable mode and backends without device-resident KV pressure report the request as unavailable rather than silently enabling mutation. `/health`, `/v1/config`, and the trusted debug state expose the request, effective state, bounded reason, and source. |
| `memory.kv_force_blocks` | unsigned integer; `0` (disabled) | `KILN_MEMORY_KV_FORCE_BLOCKS` (implemented) | `KILN_KV_FORCE_BLOCKS` (deprecated compatibility) | A positive value requests one exact startup resize before the normal autoscaler loop. It requires `memory.kv_autoscale=true` and `server.serving_profile="maintenance"`; every other combination fails configuration validation. Zero disables the one-shot operation. The resize still uses full replacement-pool reservation, exclusive GPU ownership, graph invalidation, transactional publication, and typed `forced_configuration` attribution. This is an offline maintenance/qualification control, not a per-request tuning knob. |
| `memory.kv_cache_fp8` | boolean; `false` | `KILN_MEMORY_KV_CACHE_FP8` (implemented) | `KILN_KV_CACHE_FP8` (deprecated compatibility) | Requests E4M3FN KV storage. Backend storage policy may reject or disable the request when unsupported. |
| `memory.cuda_graphs` | boolean; `true` | `KILN_MEMORY_CUDA_GRAPHS` (implemented) | `KILN_CUDA_GRAPHS` (deprecated compatibility) | CUDA-only request. Non-CUDA backends ignore it, and a serving profile with live graph capture disabled selects eager-only execution regardless of this value. |
| `memory.cuda_graph_cache_entries` | unsigned integer; `8` | `KILN_MEMORY_CUDA_GRAPH_CACHE_ENTRIES` (implemented) | None | `1..=64`. Bounds retained single-row CUDA decode graphs and their graph-stable device buffers. Resolved once before device selection; decode never re-reads process environment. The unqualified batched CUDA graph route remains unavailable. |

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

The Vulkan scratch recycler is a cache, not an allocation entitlement. Its
`memory.vulkan_buffer_pool_gb` cap is applied to Vulkan's exact bound memory
requirements, including alignment, across device-local and host-visible
routes. Borrowed buffers count against the cap and are never evicted while a
caller owns them. When there is insufficient idle capacity, Kiln either evicts
the least-recently-used idle entries or returns an uncached buffer that is
destroyed normally. The memory governor can trim only idle entries and only
when the serving profile permits allocator reclaim; changing the cap requires
a restart. Recycler misses are counted separately for device-local and
host-visible allocation routes. `GET /health` retains only the most recent miss,
including its process-lifetime sequence, requested size, rounded bucket, route,
and Rust caller file and line; it reports `null` before any miss. This diagnostic
record has constant memory cost and is not a per-allocation log.

Device-local scratch lookup requires the exact rounded bucket. Host-visible
staging lookup checks its exact bucket first, then selects the smallest
sufficient idle larger bucket for the same Vulkan device and memory type, with
oldest-use order breaking ties. This is sound because every host staging
operation retains its logical copy/read extent; it prevents arbitrary
prompt-tail shapes from allocating a new staging buffer while a larger
compatible one is idle. Undersized and borrowed buffers are never candidates,
and cross-device or cross-memory-type reuse is prohibited.

The paged KV block manager also favors residency locality: a completed
request's block run returns to the front of the free list in its original
logical order. The next compatible request reuses those already-touched pages
before walking untouched sparse-zero capacity. On host-resident Vulkan KV
pools, this bounds delayed anonymous/transparent-huge-page first touch without
prefaulting the full configured cache; logical block ownership, prefix-cache
leases, shrink retirement, and physical-resize safety are unchanged.

Physical-device identity remains fail-closed where backend enumeration and OS
memory probes expose unrelated logical ordinals. Typed
`accelerator.vulkan_device_index` now selects and reports the same Vulkan index
through kernel, model, and server code, but the subsequent memory-probe gate
still accepts it only when ordinal zero is the sole relevant DRM candidate.
This means an available nonzero Vulkan index is selected faithfully and then
rejected before model upload rather than being budgeted against another GPU.
CUDA and ROCm have the same single-candidate/ordinal-zero restriction.
Multi-device hosts, failed candidate enumeration, and driver-level visibility
or remapping controls such as `CUDA_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `MESA_VK_DEVICE_SELECT`, or
`DRI_PRIME` are rejected. The deprecated `KILN_VULKAN_DEVICE` alias is typed
application configuration, not a driver remap; `GGML_VK_VISIBLE_DEVICES` is
ignored and no longer supported. `Auto` memory probing remains diagnostic-only;
CPU performs no accelerator probe, and Apple Silicon uses its single unified
physical memory pool. Multi-device startup remains unavailable until backend
selection and probing share a typed PCI address or UUID.

## `[training]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `training.grad_checkpoint_segments` | optional unsigned integer; omitted (`None`) | `KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS` (implemented) | `KILN_GRAD_CHECKPOINT_SEGMENTS` (deprecated compatibility) | Must be greater than zero when set. When present, selects an explicit process-lifetime gradient-checkpoint segment count for native training; omission leaves workload- and capacity-aware automatic planning enabled. |
| `training.no_grad_checkpoint` | boolean; `false` | `KILN_TRAINING_NO_GRAD_CHECKPOINT` (implemented) | `KILN_NO_GRAD_CHECKPOINT` (deprecated compatibility) | Disables gradient checkpoint execution for native training. The disabled state and any explicit segment count are retained together in the immutable training policy and exact-resume identity. Disabling checkpointing can materially increase training memory. |
| `training.recompute_checkpoint_boundaries` | string enum; `"auto"` | `KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES` (implemented) | `KILN_RECOMPUTE_CHECKPOINT_BOUNDARIES` (deprecated compatibility) | `auto`, `enabled`, or `disabled`, case-insensitive with surrounding whitespace ignored. `auto` replays sparse SFT boundaries when sequence length is at least `recompute_boundary_threshold_tokens`; `enabled` always replays them and `disabled` always retains them. The canonical environment name accepts only those three words. The deprecated alias additionally accepts historical `1`/`true`/`yes` and `0`/`false`/`no`; `on` and `off` are not accepted for this alias. |
| `training.recompute_boundary_threshold_tokens` | positive unsigned integer; `8192` | `KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS` (implemented) | `KILN_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS` (deprecated compatibility) | Inclusive sequence-length threshold used only by automatic SFT boundary replay. Zero, negative, overflowing, malformed, and non-UTF-8 environment values stop startup. |
| `training.checkpoint_boundary_anchor_stride` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE` (implemented) | `KILN_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE` (deprecated compatibility) | When sparse SFT replay is active, a concrete value retains every Nth segment boundary as an anchor. `auto` derives a shape-specific positive stride from sequence length, segment count, hidden width, boundary dtype, and the cache target. Zero and strings other than `auto` fail startup. |
| `training.checkpoint_boundary_cache_gb` | positive floating-point GiB value; `6.0` | `KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB` (implemented) | `KILN_CHECKPOINT_BOUNDARY_CACHE_GB` (deprecated compatibility) | Automatic anchor-stride memory target. Despite the historical `_gb` spelling, the unit is GiB (`2^30` bytes). The value must be finite, positive, convert to at least one byte, and remain below the `u64` byte limit. Startup converts it once to integral bytes using the historical truncating conversion. |
| `training.checkpoint_interval` | optional unsigned integer; omitted (`None`) | `KILN_TRAINING_CHECKPOINT_INTERVAL` (implemented) | `KILN_CHECKPOINT_INTERVAL` (deprecated compatibility) | Must be greater than zero when set. Number of committed optimizer steps between checkpoints; per-job configuration overrides it. Omission disables periodic checkpoints. |
| `training.webhook_url` | optional string; omitted (`None`) | `KILN_TRAINING_WEBHOOK_URL` (implemented) | `KILN_TRAINING_WEBHOOK_URL` | Must be a non-empty valid HTTP(S) URL. An exactly empty environment value clears a TOML URL; whitespace is not a clearing value and fails validation. Delivery is fire-and-forget with a five-second timeout after terminal state is recorded. |
| `training.logit_cache_dir` | optional path string; omitted (beside the effective adapter directory) | `KILN_TRAINING_LOGIT_CACHE_DIR` (implemented) | `KILN_LOGIT_CACHE_DIR` (deprecated compatibility) | Must be a non-empty path when set. Resolved once at startup; request handlers never reread the environment. |
| `training.max_queued_jobs` | unsigned integer; `32` | `KILN_TRAINING_MAX_QUEUED_JOBS` (implemented) | `KILN_TRAINING_MAX_QUEUED_JOBS` | Must be greater than zero. At capacity, submissions return HTTP 503 with `Retry-After: 30`. |
| `training.max_tracked_jobs` | unsigned integer; `1024` | `KILN_TRAINING_MAX_TRACKED_JOBS` (implemented) | `KILN_TRAINING_MAX_TRACKED_JOBS` | Must be greater than zero and at least `max_queued_jobs`. Counts queued, running, completed, and failed entries. |
| `training.tracked_job_ttl_secs` | unsigned integer; `604800` | `KILN_TRAINING_TRACKED_JOB_TTL_SECS` (implemented) | `KILN_TRAINING_TRACKED_JOB_TTL_SECS` | Must be greater than zero. Terminal entries older than the TTL are removed; active jobs are never age-evicted. |

Training GPU work is also governed by `server.serving_profile`; the default
`stable` profile does not grant training GPU ownership.

### Optimizer support is a resident capability, not configuration

There is intentionally no startup or request switch for optimizer fallback or
rounding. Product SFT, GRPO, OPD, the distinct DistillRefresh workload, and
OPD-backed distillation use immutable round-to-nearest. Stochastic rounding
remains available only to explicit Rust optimizer-library callers; it is not a
server product mode. Consequently, the
mechanical `KILN_<SECTION>_<FIELD>` rule produces no optimizer-policy
environment variable.

The following old process-global inputs were removed without aliases:

- `KILN_BF16_STOCHASTIC_ROUND`
- `KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK`
- `KILN_CUDA_TRAINING_OPTIMIZER_FALLBACK`
- `KILN_ROCM_TRAINING_OPTIMIZER_FALLBACK`
- `KILN_METAL_TRAINING_OPTIMIZER_FALLBACK`
- `KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK`

Delete them from service definitions. There is no replacement field: silent
host fallback during product training would change performance, precision,
residency, and checkpoint identity. Unknown environment variables are not a
supported way to select behavior.

`GET /v1/config` exposes the resident contract at
`training.optimizer_support`. A real runner returns schema ID
`kiln.training-optimizer-support` and version `1`; a mock runner or a runner
whose snapshot cannot be read returns `null`. `immutable_after_startup=true`
means this is a process-lifetime capability snapshot, not mutable
configuration. The object has this complete shape (the arrays always enumerate
all four workloads and all three optimizer kinds):

```json
{
  "schema": {"id": "kiln.training-optimizer-support", "version": 1},
  "backend": "rocm",
  "device": "rocm:0",
  "base_weight_dtype": "bf16",
  "resolved_lora_parameter_dtype": "bf16",
  "immutable_after_startup": true,
  "rounding_modes": ["round_to_nearest"],
  "backend_implementation_rounding_modes": ["round_to_nearest"],
  "optimizer_tuple_kinds": ["muon", "adam_w", "sgd"],
  "workloads": [
    {
      "workload": "sft",
      "supported": true,
      "unavailable_reason": null,
      "allowed_optimizer_kinds": ["muon", "adam_w", "sgd"]
    },
    {
      "workload": "grpo",
      "supported": true,
      "unavailable_reason": null,
      "allowed_optimizer_kinds": ["muon", "adam_w", "sgd"]
    },
    {
      "workload": "opd",
      "supported": true,
      "unavailable_reason": null,
      "allowed_optimizer_kinds": ["muon", "adam_w", "sgd"]
    },
    {
      "workload": "distill_refresh",
      "supported": false,
      "unavailable_reason": "distill_refresh is unavailable until admission pins separate exact SFT and OPD phase plans, prepares the exact SFT rows, and reserves the maximum sequential working set",
      "allowed_optimizer_kinds": []
    }
  ],
  "optimizers": [
    {
      "kind": "muon",
      "backend_implementation": {
        "supported": true,
        "route": "native_device_hook",
        "native_device_hook": true,
        "parameter_dtypes": ["f32", "bf16"]
      },
      "optimizer_tuple": {
        "supported": true,
        "unavailable_reason": null,
        "lora_rank": {
          "minimum": 2,
          "maximum": 48,
          "backend_maximum": 48,
          "model_maximum": 1024,
          "live_memory_admission_required": true
        }
      }
    },
    {
      "kind": "adam_w",
      "backend_implementation": {
        "supported": true,
        "route": "native_device_hook",
        "native_device_hook": true,
        "parameter_dtypes": ["f32", "bf16"]
      },
      "optimizer_tuple": {
        "supported": true,
        "unavailable_reason": null,
        "lora_rank": {
          "minimum": 1,
          "maximum": 1024,
          "backend_maximum": null,
          "model_maximum": 1024,
          "live_memory_admission_required": true
        }
      }
    },
    {
      "kind": "sgd",
      "backend_implementation": {
        "supported": true,
        "route": "native_device_hook",
        "native_device_hook": true,
        "parameter_dtypes": ["f32", "bf16"]
      },
      "optimizer_tuple": {
        "supported": true,
        "unavailable_reason": null,
        "lora_rank": {
          "minimum": 1,
          "maximum": 1024,
          "backend_maximum": null,
          "model_maximum": 1024,
          "live_memory_admission_required": true
        }
      }
    }
  ]
}
```

Read the response as four distinct gates:

1. `backend_implementation` reports whether the optimizer library contains a
   raw executable update implementation, its `portable_reference` or
   `native_device_hook` route, whether that route is an accelerator-native
   hook, and its parameter dtypes. It does not claim that resident server
   weights can execute a training workload. CPU therefore reports
   `portable_reference` with `native_device_hook=false`. An unavailable raw
   implementation reports `supported=false`, route `unavailable`, and
   `native_device_hook=false`; `parameter_dtypes` remains the backend's
   advertised kind-specific dtype set.
2. `optimizer_tuple` combines that implementation with the exact resident
   backend/device identity, base-weight dtype, resolved LoRA dtype, immutable
   product rounding rule, optimizer kind, and static LoRA-rank range. Within
   `lora_rank`, `backend_maximum` is the optimizer backend's optional ceiling,
   `model_maximum` is derived from the smallest dimension among every trained
   projection, and `maximum` is their effective minimum. A null
   `backend_maximum` means the backend adds no ceiling; it never makes the
   effective `maximum` null because the resident model still supplies one.
   `optimizer_tuple_kinds` is the summary of kinds whose resident tuple is
   supported. A supported tuple has `unavailable_reason: null`; an unsupported
   tuple retains its concrete reason next to the kind.
3. Each member of `workloads` independently reports whether the complete
   server substrate for `sft`, `grpo`, `opd`, or `distill_refresh` is
   executable. Its
   `allowed_optimizer_kinds` is the intersection of that workload gate and
   `optimizer_tuple_kinds`; it is empty whenever the workload is unavailable,
   and that entry carries a concrete `unavailable_reason`.
   This per-workload array, not a raw hook or tuple, is the static HTTP and
   dashboard admission authority.
4. `live_memory_admission_required=true` is deliberately outside the static
   promise. A supported tuple and workload can still be rejected when the
   exact request shape does not fit the current memory budget. Admission
   rejects the request; it never lowers rank, changes optimizer, or switches
   execution route.

`rounding_modes` is the server-product policy and currently contains only
`round_to_nearest`. `backend_implementation_rounding_modes` describes what the
resident backend optimizer implementation reports. Those fields are separate
so a future implementation capability cannot silently become a product mode.
`resolved_lora_parameter_dtype: null` means the resident base dtype cannot be
resolved under the backend precision policy; the per-kind tuple reasons carry
the failure rather than treating the dtype as unknown.

The sibling `training.native_training_supported` and
`native_training_unavailable_reason` fields are a coarse compatibility summary.
They do not replace `workloads`: a true summary does not promise that all four
workloads are available, and clients selecting SFT, GRPO, OPD, or
DistillRefresh must inspect the matching workload member.

For canonical Qwen3.5-4B, whose trained-projection model ceiling is 1024, the
resident optimizer-tuple matrix before workload and live-memory admission is:

| Backend | Base -> LoRA dtype | SGD | AdamW | Muon |
|---|---|---|---|---|
| CPU portable reference | F32 -> F32 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=1024 (backend unbounded) |
| CUDA | F32 -> F32; BF16 -> BF16 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=48 |
| ROCm | F32 -> F32; BF16 -> BF16 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=48 |
| Metal | BF16 -> BF16 | unsupported | rank 1..=1024 (backend unbounded) | rank 2..=32 |
| Vulkan tuple | F32/BF16 -> F32 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=32 |

This table is not a server-execution matrix. Current CPU processes expose the
portable F32 optimizer tuples for diagnostics and direct library testing, while
their `sft`, `grpo`, `opd`, and `distill_refresh` workload entries remain
unsupported. A Vulkan
process can likewise expose native hooks and resident optimizer tuples while a
workload fails closed. In the usual hybrid Vulkan server, model weights are
CPU-hosted, so exact native backend/device identity and resident-weight checks
reject every workload even though raw Vulkan hooks exist.

Static workload admission requires all of the following: a real readable
runner; a serving profile that grants training GPU ownership; agreement between
configured and resident weight devices; a training runtime that resolves those
weights; exact native backend and device identity; no Marlin-packed model
projection; and the authoritative `kt_tape_authoritative` forward/backward
route. SFT additionally rejects effective multi-segment checkpointing with a
`full_logits` loss route. OPD additionally requires both its loss route and
phase-B backward route. GRPO uses the common authoritative-tape gate. The
specific failed condition appears in `workloads[].unavailable_reason`.

DistillRefresh is not an OPD alias for admission. It is a sequential composite
with an SFT knowledge phase and an OPD behavior-recovery phase. The
`distill_refresh` row is unconditionally fail-closed with this stable reason:
`distill_refresh is unavailable until admission pins separate exact SFT and OPD
phase plans, prepares the exact SFT rows, and reserves the maximum sequential
working set`. Enabling it requires one admission artifact that binds both exact
phase plans, the precise SFT rows that phase one will consume, and a reservation
for the larger of the two sequential working sets. Adding the phases' estimates
or admitting only the OPD-shaped request is not equivalent.

F16 is inference-only on CUDA/ROCm. `maximum` is the effective static ceiling,
computed as `min(backend_maximum, model_maximum)` while treating a null backend
maximum as unbounded. Live memory may impose a lower request-time ceiling but
does not rewrite these static fields. The server validates
the cheap workload gate, optimizer kind, hyperparameters, base/LoRA dtype,
rounding, and rank before checkpoint loading or corpus scanning across dedicated
training endpoints, the intent-tagged front door, recipes, judge/self-improve,
DistillRefresh, and OPD-backed distillation. Checkpoint loading and corpus
preparation happen only after those checks. The worker
revalidates the checkpoint and the workload/tuple before memory reservation and
residency.
Invalid kind, rank, or hyperparameters use structured
`training_invalid_request`; an unsupported base dtype, backend/device identity,
or workload substrate uses `training_backend_unsupported`.

Cheap teacher-alias validation and metadata pinning retain their established
request-error ordering and may precede the workload guard. Remote/local teacher
materialization, checkpoint loading, corpus scanning, memory preflight, and GPU
reservation do not.

For a queued resume, admission fully validates the manifest and every declared
artifact hash, then retains a compact checkpoint-ID/manifest-digest identity
plus the effective seed. The manifest digest covers its artifact hash entries.
Before memory reservation at dequeue, the worker fully reloads the checkpoint
and requires both the recomputed identity and effective seed to match. This is
not an external-file snapshot: Kiln does not copy or pin checkpoint files in
the queue, and a mutation after the reload remains a filesystem race. Keep the
checkpoint directory immutable for its entire use.

`GET /v1/recipes` applies the same static checks to every built-in recipe step
and returns `admission: {supported, unavailable_reason}` on each descriptor. A
false descriptor identifies the first unsupported workload or optimizer/rank
tuple. A true descriptor is not a memory reservation: recipe submission repeats
the full preflight before any step is prepared, and live-memory admission still
occurs for each queued job.

### Tape scope is internal execution authority, not configuration

The kt autograd tape is a required workload substrate. Kiln opens its
thread-local scope internally around the forward/backward region that must
record gradients and closes it when that region completes. SFT, GRPO, and OPD
admission first require an authoritative tape route; a request cannot open a
scope, suppress one recorder, or choose a partially tape-backed execution.
There is intentionally no TOML field, API member, CLI flag, or mechanically
derived environment name for tape activation.

Scope presence is also the complete inference boundary. Ordinary inference
does not open a training tape scope and therefore retains forward-only fast
paths. In particular, GDN chunkwise recurrence and CUDA's weight-aware
embedding lookup defer to their tape-aware alternatives only while a scope is
actually active. They do not consult a cached process-global tape setting. This
prevents a default-on global flag from changing inference routing when no
gradient graph exists.

The following former correctness switches were removed without compatibility
aliases or replacement fields:

- `KILN_USE_TAPE_FORWARD`
- `KILN_USE_TAPE_FLASH_ATTN`
- `KILN_USE_TAPE_SDPA`
- `KILN_USE_TAPE_LORA_ADD`
- `KILN_USE_TAPE_GDN`
- `KILN_USE_TAPE_GDN_CONV`
- `KILN_USE_TAPE_GDN_QK_NORM`
- `KILN_USE_TAPE_GDN_GATED_NORM`

Delete these names from service definitions instead of translating them. A
debug or performance control may select an implementation only when the active
scope still records the required analytic backward and preserves the graph. It
must not disable the tape, bypass an individual recorder, or fall through to a
forward-only kernel. If no graph-preserving route exists, the workload fails
closed through `training.optimizer_support.workloads[]` rather than accepting a
correctness opt-out.

This contract proves static ownership and route isolation; it does not by
itself qualify numerical correctness, latency, memory use, or stability on a
particular accelerator. Those claims require the platform-specific
qualification workload and committed evidence described in
[`NATIVE_SFT_PROFILE.md`](NATIVE_SFT_PROFILE.md).

#### Frozen parameters are constants, not optimizer leaves

Native SFT, GRPO, and OPD train only LoRA A/B tensors. Embedding tables, base
projection matrices, RMSNorm and GDN gated-RMSNorm weights, GDN gate parameters,
MTP projection weights, and loss-head transposes are saved constants. Their
tape nodes record only differentiable activations and LoRA leaves. Backward
therefore returns only required activation and LoRA gradients; it does not
allocate, retain, or deposit frozen-weight gradients, and frozen projection
paths do not execute a dWeight matrix multiplication.

The split Q/gate projection keeps the original full LoRA A/B tensor IDs on both
recorded chunks. Each chunk pads its B contribution to the full shape before
the tape combines them, and tape-aware reshapes preserve the activation chain.
Sliced base matrices and temporary B slices are never trainable leaves. This is
an execution invariant with no TOML, request, CLI, debug, or environment
override. It reduces frozen-gradient memory and compute by construction, but
does not by itself qualify backend throughput.

The optimizer boundary independently requires an exact LoRA gradient set. The
observed tensor IDs must equal the configured trainable leaves; shape,
backward-compute dtype, master device, and finite values must match before any
parameter, optimizer state, or step counter is changed. Missing or unknown
entries reject the step, while a valid all-zero gradient is accepted.
Checkpointed execution applies the same identity and metadata contract to the
exact leaves in each layer range. The tape bridge reduces distinct recorded
input contributions to one accumulated entry per leaf, but rejects the result
if any registered differentiable input is absent. A connected clone therefore
cannot hide a disconnected use of the same leaf. Split output-projection chunks
assemble zero-padded contributions under the original full leaf ID rather than
temporary slice IDs. A checkpoint range with no configured LoRA leaves accepts
only an empty gradient set, and the merge rejects duplicate leaf IDs across
disjoint segments. Token-level GRPO
accumulation validates identity and metadata for each source but defers its
finite-value reduction to the single final optimizer boundary, avoiding a
per-completion device synchronization multiplier. None of these checks has a
configuration or environment bypass. CUDA, ROCm, and Vulkan
use backend finite reducers. Metal performs a synchronized full-gradient host
scan at the same boundary until a native Metal reducer is qualified; this is a
correctness fallback, not an advertised throughput path.

### Backend-owned SFT loss route is not configuration

There is intentionally no `[training]` field for the native SFT loss route.
The resident backend reports one typed capability: CUDA and ROCm report
`kt_tape_flce`, Vulkan reports `vulkan_active_rows`, and Metal reports
`full_logits`. These values are capability and artifact names, not accepted
TOML enums, request fields, CLI flags, or `/v1/config` controls. Because no
configuration field exists, the mechanical naming rule does not produce an
environment variable for this choice.

The old `KILN_USE_FLCE` input has been removed. It is not a deprecated alias,
the typed loader does not consume it, and setting it has no effect on current
SFT loss selection. Operators should remove it from service definitions rather
than translating it to a new name. Backend loss routing is not an A/B switch.

SFT admission resolves the route from the resident model runner and applies a
route-specific, saturating memory upper bound. CUDA/ROCm account for kt-tape
FLCE buffers and any F32 head promotion; Vulkan accounts for its maximum legal
active-row chunk and F32 workspace; `full_logits` accounts for dense `[T, V]`
logits and cross-entropy forward/backward residency. Overflow saturates toward
rejection. An over-budget request returns HTTP 413 with estimated and available
capacity and a breakdown containing `loss workspace ... (route=<route>)`.

The admitted enum is stored with the queued SFT job. The worker revalidates it
against the resident runner before memory reservation or reclamation, and the
trainer revalidates it against its execution backend before resident or
trainable allocations. The pinned value drives each forward/backward step,
appears in new SFT receipts as `runtime.sft_loss_route`, and participates in
the SFT exact-resume planning identity. `full_logits` is not compatible with a
multi-segment checkpoint plan; that combination returns
`training_invalid_request` before queue publication and is checked again by the
trainer. These are immutable execution contracts, not omitted configuration
knobs.

### SFT checkpoint-boundary policy

These four fields control activation boundaries inside gradient-checkpointed
native SFT. They do not enable or disable gradient checkpointing; that remains
the job of `no_grad_checkpoint` and the resolved segment plan. When replay is
inactive, SFT retains every segment input. When replay is active, it retains a
sparse set of anchors and recomputes intervening segment inputs during the
backward pass.

For `checkpoint_boundary_anchor_stride = "auto"`, Kiln computes the bytes in
one boundary as `sequence length * hidden size * bytes per element`, determines
how many anchors fit in `checkpoint_boundary_cache_gb`, reserves one slot for
replay, and spreads the remaining anchors across the planned segments. The
automatically selected result is always at least one, and auto mode selects
stride one for a one-segment plan. An explicit stride wins even for a
one-segment plan and bypasses only this shape calculation. It does not override
the recompute mode.

Startup resolves all four values into one immutable
`CheckpointBoundaryPolicy`. The typed loader retains `default`, `config_file`,
or `environment` source attribution for each configured value. Admission and
execution receive the same copyable policy and call the same pure threshold and
shape functions, so a request cannot be admitted under one boundary-memory
estimate and execute another. Trainer and preflight code do not re-read any of
the four environment names. Changing any value requires restarting the server;
queued and running jobs retain the policy installed at startup.

Sparse checkpoint-boundary replay is currently SFT-only. Native GRPO and OPD
retain all `num_segments + 1` boundaries and do not use these fields to choose
their live boundary layout. The common training runtime still records the
resolved policy in exact-resume planning identity. GRPO and OPD use
`kiln.training-checkpoint-planning.v3`. SFT extends the same object to v4 with
its pinned backend loss route. A changed boundary value, including one that is
execution-inert for GRPO or OPD today, is therefore exact-resume drift. An SFT
v3 identity also cannot authorize continuation under v4 because it cannot
prove the admitted loss implementation. See [Native Training
Checkpoints](training-checkpoints.md#checkpoint-planning-identity).

The resolved object has this stable JSON shape:

```json
{
  "recompute_mode": "auto",
  "recompute_threshold_tokens": 8192,
  "anchor_stride": null,
  "cache_target_bytes": 6442450944
}
```

`GET /v1/config` exposes it at `training.checkpoint_boundary_policy` alongside
the gradient-checkpoint plan. `GET /health` and its `/v1/health` alias expose it
at `training.checkpoint_boundary_policy`; trusted
`GET /v1/debug/model-state` exposes the same object at
`training.checkpoint_boundary_policy`. The dashboard's Runtime Config training
group renders mode, threshold, explicit-or-auto stride, cache target, and the
restart requirement. These are observations of immutable application state,
not live controls or fresh environment reads.

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
| `prefix_cache.enabled` | boolean; `true` | `KILN_PREFIX_CACHE_ENABLED` (implemented) | `KILN_PREFIX_CACHE_ENABLED` | Requests reuse of KV blocks and recurrent-state snapshots for shared prompt prefixes. CPU, CUDA, ROCm, and Metal honor the request. Vulkan currently forces the effective capability off because repeated production-model runs proved semantic corruption after cross-request restoration. This is a source-level correctness quarantine, not a second setting. |
| `prefix_cache.max_blocks` | optional unsigned integer; omitted (`None`) | `KILN_PREFIX_CACHE_MAX_BLOCKS` (implemented) | `KILN_PREFIX_CACHE_MAX_BLOCKS` | Must be greater than zero when set. On an admitted backend, `None` resolves to half of the allocated KV block pool. It has no runtime allocation effect while the effective capability is off. |
| `prefix_cache.max_entries` | optional unsigned integer; omitted (`None`) | `KILN_PREFIX_CACHE_MAX_ENTRIES` (implemented) | `KILN_PREFIX_CACHE_MAX_ENTRIES` | Must be greater than zero when set. On an admitted backend, `None` resolves from the relevant safe allocation tier and per-entry recurrent-state bytes, with at least one entry. It has no runtime allocation effect while the effective capability is off; quarantined Vulkan reserves no host-backed prefix state. |

The TOML object records operator intent. `GET /v1/config` reports that object at
`prefix_cache.configuration` and reports the backend-qualified result beside it:
`effective_enabled`, `effective_reason`, `effective_max_blocks`,
`effective_max_entries`, and `effective_max_state_bytes`. Reasons are `active`,
`configuration`, `vulkan_correctness_quarantine`, or `backend_unavailable`.
`GET /health` and `GET /v1/health` expose the live capability both as
`prefix_cache.enabled` and, when the batching actor exists,
`decode_runtime.batching_engine.prefix_cache_enabled`. Trusted debug state
publishes the same two views. Prometheus publishes
`kiln_batching_engine_prefix_cache_enabled` for the actor capability plus the
existing `kiln_prefix_cache_*` activity and residency series.

While Vulkan is quarantined, all prefix-cache lookup, hit, miss, retained-block,
retained-entry, recurrent-state, lease, and pending-release values must remain
zero. Requests use fresh generic prefill, including exact repeats. There is no
request override or alternate environment flag that bypasses the quarantine.
Re-enablement requires a source change and production-model semantic parity
across first use, exact repeats, strict descendants, changing concurrency,
cancellation, and repeated process history; cache mechanics tests alone are not
sufficient.

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

The former Phase B/C `KILN_MTP_DEBUG`, `KILN_MTP_DUMP_*`,
`KILN_MTP_SWAP_*`, and `KILN_MTP_*FP32*` experiment controls are retired and
are not configuration inputs. They have no typed replacements because they
selected completed diagnostic branches rather than supported runtime policy.
Current source retains one canonical MTP execution path; historical commands
and conclusions remain labeled as such in `PROFILING.md`.

## `[streaming_prefill]`

| TOML field | Type and exact default | Canonical env target | Working spelling(s) today | Validation and effective semantics |
|---|---|---|---|---|
| `streaming_prefill.mode` | string enum; `"auto"` | `KILN_STREAMING_PREFILL_MODE` (implemented) | `KILN_STREAMING_PREFILL` and `KILN_STREAMING_PREFILL_ENABLED` (deprecated compatibility) | `auto`, `enabled`, or `disabled`, case-insensitive with surrounding whitespace ignored. `auto` delegates dispatch to backend policy, `enabled` selects every non-empty prompt, and `disabled` selects none. When the ROCm batching actor is effective, dispatch must cover its first effective tile; disabling streaming therefore also requires disabling the actor. The canonical environment name accepts only those three words. Deprecated aliases also accept the strict boolean forms listed above, mapping true to `enabled` and false to `disabled`. Legacy TOML `enabled = true` or `enabled = false` remains accepted; if `mode` is also present, both must express the same non-auto intent or startup fails. |
| `streaming_prefill.threshold_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_THRESHOLD_TOKENS` (implemented) | `KILN_STREAMING_PREFILL_THRESHOLD_TOKENS` | In `auto` mode, an integer replaces the backend crossover only when that backend already has a threshold-based automatic policy. It does not make CPU or Vulkan auto-dispatch streaming. With the ROCm batching actor effective, the crossover cannot be later than the effective tile. `0`, negative values, and strings other than `auto` fail startup. `enabled` and `disabled` modes ignore this crossover for dispatch while retaining it in diagnostics. |
| `streaming_prefill.tile_tokens` | `"auto"` or positive unsigned integer; `"auto"` | `KILN_STREAMING_PREFILL_TILE_TOKENS` (implemented) | `KILN_STREAMING_TILE_TOKENS` (deprecated compatibility) | Base tile for ordinary tiled prefill and non-tape GDN segment execution. Concrete values must be positive multiples of 64. With the ROCm batching actor effective, this tile must equal `server.max_prefill_tokens_per_cycle` and fit beside effective decode width in `server.max_batch_tokens`. When this field is concrete and either specialized tile below is `auto`, that specialized route inherits this base value rather than its backend default. |
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
| ROCm | prompt tokens >= 256 | 256 | 256 | 8192 | 8192 | 8192 |
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

Vulkan resumable-prompt GDN state residency is intentionally absent from this
configuration section because the production route is correctness-quarantined.
There is no TOML field, canonical environment override, compatibility alias, or
internal enable variable that can activate it. The typed Vulkan runtime policy
sets prompt residency to false. Ordinary actor chunking therefore materializes
each layer's recurrent state at the completed token-chunk boundary, including
the BF16 dtype boundary, before a request can yield and resume. Layer-group
yields inside one token chunk retain their existing in-progress state and do
not add an extra numerical boundary.

The quarantine is evidence-based. On the clean `967e1f799` Vulkan release,
disabling both prompt and decode recurrent-state residency produced the exact
required 32-token prefix `000000 000001 000002 000003 0000` for the 138-token
q128 production prompt. The same binary with prompt residency active and decode
residency disabled produced plausible but incorrect sequence text. The prior
fully resident arm was more severely corrupt. Focused device tests still match
the small host BF16 oracle bit-for-bit, which proves those tests are necessary
but not sufficient; they do not justify exposing the full-model route.

The exact pushed quarantine revision `7dcad0d95` also passed without injecting
the disable guard, while the checked profile retained its historical decode
opt-in. It emitted the identical 32 token IDs and required text with no direct
prompt-registry activation. This proves that neither public configuration nor
the legacy decode opt-in can bypass the source quarantine. Its 31.34-second
prefill was slower than the fully materialized rollback arm's 24.16 seconds, so
the result is semantic acceptance only; it does not close the q128 performance
or soak gates.

The lower-level registry, stable `(serving row ID, linear-attention layer
index)` ownership, TensorId alias transfer, device-side BF16 boundary, exact
owner eviction, and observability remain compiled and test-covered so the fault
can be investigated without reconstructing the experiment. Production cannot
enter those prompt scopes. Re-enabling them requires a source change, the
focused real-device suite, a clean full-model semantic pass with prompt
residency enabled and decode residency disabled, the cancellation/drain probe,
and the checked soak. A micro-kernel or small-state parity result alone is not a
release gate.

The historical decode residency experiment is also disabled by immutable
`kiln.vulkan-kernel-policy.v3`. Its former enable and disable variables were
deleted without aliases; neither decode nor prompt residency can be activated
by process environment. Re-enabling either route requires a reviewed source
policy version and the full evidence sequence above.

Trusted `GET /v1/debug/model-state` exposes current ownership as
`caches.resident_recurrent_state.entry_count`, `buffer_bytes`, and
`allocation_bytes`. `/metrics` publishes the same state as
`kiln_gdn_recurrent_state_resident_entries` and
`kiln_gdn_recurrent_state_resident_bytes{kind="buffer|allocation"}`. A drained
server must report zero for all three values. While prompt residency is
quarantined, production also expects zero throughout prompt execution; a
nonzero value is an invalid route activation or ownership defect. Their wire
types are generated from `contracts/kiln-observability-v1.schema.json`; no
copied environment variable or undocumented toggle controls them.

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
# Isolate a suspected tiled-prefill pause through the direct route. ROCm must
# disable the actor as well because actor/direct tile alignment is mandatory.
[batching]
mode = "disabled"

[streaming_prefill]
mode = "disabled"
```

```toml
# ROCm actor-disabled tuning experiment: stream from 4096 tokens and use one
# base tile for all ordinary, tape, detached, boundary, and replay routes.
# This is an isolation control, not a production actor configuration.
[batching]
mode = "disabled"

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
| `adapters.library_url` | string URL; `"https://library.kiln.run"` | `KILN_ADAPTERS_LIBRARY_URL` (implemented) | `KILN_ADAPTER_LIBRARY_URL` (deprecated compatibility) | Must be a non-empty valid HTTP(S) URL. Resolved once at startup. |
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
| `agent.self_improve_interval_hours` | optional unsigned integer; omitted (`None`) | `KILN_AGENT_SELF_IMPROVE_INTERVAL_HOURS` (implemented) | `KILN_AGENT_SELF_IMPROVE_INTERVAL_HOURS` | Omission disables scheduling. `0` is accepted and also results in no scheduler. The first run occurs one full interval after startup; cadence is persisted under the adapter directory. |
| `agent.self_improve` | optional structured value; omitted (`None`) | `KILN_AGENT_SELF_IMPROVE` (target only; not implemented) | none | Request template submitted to the same self-improvement path as the API. This value is intentionally open structured data; its inner request contract is validated by that subsystem rather than by `KilnConfig`. |
| `agent.max_concurrent_runs` | unsigned integer; `2` | `KILN_AGENT_MAX_CONCURRENT_RUNS` (implemented) | `KILN_AGENT_MAX_CONCURRENT_RUNS` | Must be greater than zero. Embedded pi runs above the limit queue FIFO. |
| `agent.run_timeout_secs` | unsigned integer; `900` | `KILN_AGENT_RUN_TIMEOUT_SECS` (implemented) | `KILN_AGENT_RUN_TIMEOUT_SECS` | Must be at least `10`. A per-run timeout can override the server default. |
| `agent.runs_access` | string enum; `"loopback_only"` | `KILN_AGENT_RUNS_ACCESS` (implemented) | `KILN_AGENT_RUNS` (deprecated compatibility; boolean spellings map to enabled/disabled) | `loopback_only`, `enabled`, or `disabled`. Compatibility boolean spellings are accepted from the environment. Embedded runs can execute arbitrary code; changing this requires restart. |
| `agent.pi_bin` | optional path string; omitted (search startup PATH) | `KILN_AGENT_PI_BIN` (implemented) | `KILN_PI_BIN` (deprecated compatibility) | Must name a non-empty existing file when explicitly configured. The resolved executable is immutable for the process lifetime. |
| `agent.pi_sessions_dir` | optional path string; omitted (`$HOME/.pi/agent/sessions`) | `KILN_AGENT_PI_SESSIONS_DIR` (implemented) | `KILN_PI_SESSIONS_DIR` (deprecated compatibility) | Must be a non-empty path when set. Relative paths and the HOME fallback are resolved once at startup. |

Terminal and embedded-run access default to `loopback_only`. The host decision,
the pi executable, session root, adapter-library URL, and teacher-logit cache
root are resolved once during startup and published under `operational` in
`GET /v1/config`. Request handlers use that immutable snapshot; changing TOML,
`PATH`, `HOME`, or a compatibility environment alias requires a restart.

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
  then prints a human-oriented subset of resolved fields, including the ROCm
  synchronization mode and configured/effective graph policy. Adding
  `--backend <target>` resolves hardware-independent scheduling capabilities
  and rejects an invalid actor-prefill contract, but deliberately does not
  claim device availability or live route state.
- `GET /v1/config` reports runtime diagnostics for serving profile, the
  versioned accelerator runtime policy, effective
  decode width, speculative configuration and availability, VRAM/KV state,
  native-training runtime/weight devices and
  support reason, the versioned implementation/resident-tuple/per-workload
  optimizer contract, gradient-checkpoint segmentation, the immutable SFT
  checkpoint-boundary policy, memory budgets, and generation defaults. Its
  `training.checkpoint_boundary_policy` object reports resolved recompute mode,
  inclusive automatic threshold, optional explicit stride, and integral cache
  target bytes. Its `speculative` object distinguishes `configured_method` and
  `configured_effective_method` from the actual `serving_effective_method`
  (`off`), reports `serving_routable=false`, the stable
  `serving_unavailable_reason`, the K ceiling, and immutable backend MTP
  support facts. It does not expose live routing, admission, or weight-readiness
  fields because those are not serving authorities. A running process cannot
  have a non-off configured method through the supported startup path.
  `training.native_training_supported=false` is accompanied by
  `training.native_training_unavailable_reason` and matches the admission
  error without scanning a corpus. `training.optimizer_support` separately
  exposes backend implementations, exact resident optimizer tuples, and
  per-workload executable kinds. Consumers must use `workloads[].supported` and
  `workloads[].allowed_optimizer_kinds`, not infer product support from
  `backend_implementation` or `optimizer_tuple_kinds`. When
  checkpoint execution is disabled,
  `training.checkpoint_segments` is `0`; an optional segment count retained in
  `training.checkpoint_policy` is provenance, not an active execution plan. The
  endpoint is not a serialization of all accepted TOML.
- `GET /health` and `/v1/health` repeat the accelerator policy and reasoned
  ROCm synchronization counters under `decode_runtime`, and repeat the resolved boundary policy under
  `training.checkpoint_boundary_policy`; trusted `GET /v1/debug/model-state`
  reports the identical object at the same path. The dashboard reads
  `/v1/config` and renders the same process-lifetime values. None of these
  surfaces parses configuration or rereads process environment.
- `GET /v1/config` also reports cached-sample age, maximum age, staleness,
  sampler requirement/liveness/health, cgroup `memory.high`, and the separately
  bounded `vram.live.raw_observations.host_backed` tier. `/health` exposes the
  same liveness decision, while `/metrics` exports fixed-cardinality gauges for
  alerting. A stale or failed sample remains visible diagnostically but
  contributes zero allocation headroom.
- `/health` and `/v1/debug/model-state` expose config hashes and other runtime
  identity data. Trusted debug additionally exposes the current and
  process-lifetime batched recurrent-state cache lifecycle at
  `caches.batched_recurrent_state`, plus the separate direct prompt/scoped-decode
  GDN registry at `caches.resident_recurrent_state`; `/metrics` publishes the
  same fixed-cardinality ownership, reuse, rejection, concurrency, and eviction
  counters. The complete field and interpretation contracts are in
  [resumable prefill residency telemetry](qualification.md#resumable-gdn-prefill-residency-telemetry)
  and [batched recurrent-state cache telemetry](qualification.md#batched-recurrent-state-cache-telemetry).
- Startup logs record serving-profile provenance and configured/backend/final
  decode-width sources.

Explicit source tracking (`default`, `config_file`, or `environment`) currently
exists for `server.serving_profile`, `server.deterministic`,
`server.stream_stall_grace_ms`, the three server batching/prefill budgets,
`server.max_decode_batch`, all nine `[batching]` fields, all six
`[streaming_prefill]` fields, all six `[accelerator]` fields, and
`memory.reclaim_mode`. Other fields have
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
   typed object with provenance and backend-derived values. The CLI's optional
   backend preview is intentionally limited to static scheduling policy.
2. **Deprecated aliases:** 61 non-canonical spellings across 58 fields remain
   temporarily for compatibility, including `KILN_DEFAULT_NO_THINK`. Each use
   warns at startup; canonical and compatibility names cannot silently
   disagree.

The intended migration is to resolve every public runtime field exactly once,
pass immutable typed configuration into its owner, generate the environment
contract mechanically, and reject direct production environment reads outside
that boundary. Internal experiment and qualification flags should remain
separate from this public API rather than being promoted by documentation.
