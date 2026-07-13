# Configuration Reference

This document is the canonical operator reference for Kiln's typed server
configuration. It describes the configuration accepted by the current
`KilnConfig` and `RequestLogConfig` implementations, including current defaults,
validation, working environment-variable aliases, and known migration gaps.

Kiln's intended public environment-variable naming rule is mechanical:

```text
KILN_<SECTION>_<FIELD>
```

Both the section and field are converted to uppercase snake case. For example,
`server.host` maps to the canonical target `KILN_SERVER_HOST`, and
`request_log.max_file_bytes` maps to
`KILN_REQUEST_LOG_MAX_FILE_BYTES`.

**A canonical target shown below is not necessarily implemented yet.** The
"Working override today" column is authoritative for the current release:

- **implemented** means the canonical name already works;
- **legacy; works** means a non-canonical compatibility alias works today;
- **none** means there is no supported environment override for that field;
- **target only; not implemented** means setting the mechanically derived name
  currently has no effect.

Do not assume that an arbitrary `KILN_*` name is public configuration. The
repository contains internal diagnostics, qualification switches, kernel
experiments, and temporary compatibility flags. Only the working aliases in
this reference are supported as configuration inputs.

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

1. CLI behavior that deliberately sets an override before loading config
   (`serve --served-model-id` and `serve --eval-mode`).
2. A working environment alias listed in this document.
3. The selected TOML file.
4. The built-in default.

The `-v`/`-vv` and `--quiet` logging flags override `logging.level` for the
running command. `RUST_LOG`, when present, takes precedence over that resolved
level when the tracing filter is built.

Environment resolution is intended to happen once at startup. Restart the
server after changing a file or environment value. The split-brain migration
limitations documented below still contain lower-level environment reads; they
are limitations, not a supported live-reconfiguration mechanism.

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

The accepted TOML surface contains 13 top-level sections and 68 fixed leaf
fields. Dynamic `teachers.credentials.<id>` entries add two leaf fields per
credential. Of the 68 fixed fields:

- 60 have a primary working environment override;
- 19 already use the canonical mechanical name;
- 41 use a working legacy spelling;
- 8 have no environment override;
- `KILN_DEFAULT_NO_THINK` is one additional compatibility alias for
  `server.default_thinking_enabled`.

The tables below cover all 68 fixed fields and both dynamic credential fields.

## `[server]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `server.serving_profile` | string enum; `"stable"` | `KILN_SERVER_SERVING_PROFILE` (target only; not implemented) | `KILN_SERVING_PROFILE` (legacy; works) | `stable`, `experimental`, or `maintenance`, case-insensitive. Process-lifetime policy; restart required. |
| `server.deterministic` | boolean; `false` | `KILN_SERVER_DETERMINISTIC` (target only; not implemented) | `KILN_DETERMINISTIC` (legacy; works) | Enables deterministic tensor behavior and forces the effective concurrent decode width to one. |
| `server.host` | string; `"127.0.0.1"` | `KILN_SERVER_HOST` (target only; not implemented) | `KILN_HOST` (legacy; works) | Must be non-empty. Binding beyond loopback exposes an unauthenticated inference and training API; use a trusted network or authenticated reverse proxy. |
| `server.port` | unsigned 16-bit integer; `8420` | `KILN_SERVER_PORT` (target only; not implemented) | `KILN_PORT` (legacy; works) | `1..=65535`. |
| `server.request_timeout_secs` | unsigned integer; `600` | `KILN_SERVER_REQUEST_TIMEOUT_SECS` (target only; not implemented) | `KILN_REQUEST_TIMEOUT_SECS` (legacy; works) | Must be greater than zero. Bounds a request, including model work and cleanup settlement. |
| `server.http_send_buffer_bytes` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_HTTP_SEND_BUFFER_BYTES` (target only; not implemented) | `KILN_HTTP_SEND_BUFFER_BYTES` (legacy; works) | When set, `1024..=16777216`. Applied to accepted sockets. Startup preflights the listener and rejects an OS read-back smaller than requested. |
| `server.stream_stall_grace_ms` | unsigned integer; `2000` | `KILN_SERVER_STREAM_STALL_GRACE_MS` (target only; not implemented) | `KILN_STREAM_STALL_GRACE_MS` (legacy; works) | `10..=2000`. A request retaining KV state with no streaming delivery progress is selected for cancellation after this grace. |
| `server.max_batch_tokens` | unsigned integer; `512` | `KILN_SERVER_MAX_BATCH_TOKENS` (target only; not implemented) | `KILN_MAX_BATCH_TOKENS` (legacy; works) | `2..=65536`. Combined decode-plus-prefill token budget for one batching-actor cycle. |
| `server.max_prefill_tokens_per_cycle` | unsigned integer; `64` | `KILN_SERVER_MAX_PREFILL_TOKENS_PER_CYCLE` (target only; not implemented) | `KILN_MAX_PREFILL_TOKENS_PER_CYCLE` (legacy; works) | `1..=65536`. Independent new-prompt-token ceiling within the combined actor budget. |
| `server.max_prefill_layers_per_cycle` | unsigned integer; `4` | `KILN_SERVER_MAX_PREFILL_LAYERS_PER_CYCLE` (target only; not implemented) | `KILN_MAX_PREFILL_LAYERS_PER_CYCLE` (legacy; works) | `1..=1024`. Number of transformer layers an in-flight prefill chunk may execute before yielding to decode. |
| `server.max_decode_batch` | `"auto"` or unsigned integer; `"auto"` | `KILN_SERVER_MAX_DECODE_BATCH` (target only; not implemented) | `KILN_MAX_DECODE_BATCH` (legacy; works) | `auto`, `backend`, and `backend_policy` all delegate to backend policy; an integer must be `1..=65536`. Deterministic mode and `max_batch_tokens` may lower the final width. |
| `server.eval_mode` | boolean; `false` | `KILN_SERVER_EVAL_MODE` (target only; not implemented) | `KILN_EVAL_MODE` (legacy; works) | Enables deterministic eval-serving defaults, headers, adapter warnings, and transient-cache cleanup behavior. `serve --eval-mode` wins by setting this alias before load. |
| `server.default_thinking_enabled` | optional boolean; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_ENABLED` (target only; not implemented) | `KILN_DEFAULT_THINKING_ENABLED` (legacy; works); `KILN_DEFAULT_NO_THINK` is a presence-only compatibility alias | `None` preserves the model template default. Requests may override with `chat_template_kwargs.enable_thinking`. Any presence of `KILN_DEFAULT_NO_THINK`, even a value such as `0`, first selects `false`; the explicit `KILN_DEFAULT_THINKING_ENABLED` value is applied afterward and wins. There is no environment spelling that restores `None`. |
| `server.default_thinking_budget_tokens` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_BUDGET_TOKENS` (target only; not implemented) | `KILN_DEFAULT_THINKING_BUDGET_TOKENS` (legacy; works) | Integer values include `0`, which closes thinking immediately. Case-insensitive `unlimited` clears a TOML limit back to `None`. Requests may inherit, replace, or explicitly disable the limit. |
| `server.default_thinking_budget_ms` | optional unsigned integer; omitted (`None`) | `KILN_SERVER_DEFAULT_THINKING_BUDGET_MS` (target only; not implemented) | `KILN_DEFAULT_THINKING_BUDGET_MS` (legacy; works) | Integer values include `0`. `unlimited` clears a TOML limit. The clock starts at the first decode candidate, after queueing and prefill. The first token or time limit reached forces the model's closing `</think>` sequence. |
| `server.fold_reasoning_into_content` | boolean; `false` | `KILN_SERVER_FOLD_REASONING_INTO_CONTENT` (target only; not implemented) | `KILN_FOLD_REASONING_INTO_CONTENT` (legacy; works) | Copies separated reasoning into `content` for compatibility clients. A request can override it. |
| `server.chat_performance_metadata` | boolean; `false` | `KILN_SERVER_CHAT_PERFORMANCE_METADATA` (target only; not implemented) | `KILN_CHAT_PERFORMANCE_METADATA` (legacy; works) | Default for chat response performance metadata; requests can override with `include_performance`. |
| `server.chat_config_hash_metadata` | boolean; `false` | `KILN_SERVER_CHAT_CONFIG_HASH_METADATA` (target only; not implemented) | `KILN_CHAT_CONFIG_HASH_METADATA` (legacy; works) | Default for chat response config hashes; requests can override with `include_config_hashes`. |
| `server.slow_request_warn_secs` | unsigned integer; `30` | `KILN_SERVER_SLOW_REQUEST_WARN_SECS` (target only; not implemented) | `KILN_SLOW_REQUEST_WARN_SECS` (legacy; works) | `0` disables slow-request warnings; otherwise a request at least this old emits a structured warning. |
| `server.shutdown_timeout_secs` | unsigned integer; `5` | `KILN_SERVER_SHUTDOWN_TIMEOUT_SECS` (target only; not implemented) | `KILN_SHUTDOWN_TIMEOUT_SECS` (legacy; works) | Must be greater than zero. Hard ceiling for graceful drain before forced exit. |

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

## `[model]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `model.path` | optional string; omitted (`None`) | `KILN_MODEL_PATH` (implemented) | `KILN_MODEL_PATH` | Must be non-empty when set. Omitted starts the server in mock mode; a real model path enables real inference. |
| `model.model_id` | string; `"Qwen/Qwen3.5-4B"` | `KILN_MODEL_ID` (implemented) | `KILN_MODEL_ID` | Must be non-empty. Used for model/tokenizer identity and as the source of the default served id. Kiln still applies its built-in Qwen3.5-4B runtime profile. |
| `model.tokenizer_path` | optional string; omitted (`None`) | `KILN_MODEL_TOKENIZER_PATH` (target only; not implemented) | `KILN_TOKENIZER_PATH` (legacy; works) | Must be non-empty when set. |
| `model.adapter_dir` | optional string; omitted (`None`) | `KILN_MODEL_ADAPTER_DIR` (target only; not implemented) | `KILN_ADAPTER_DIR` (legacy; works) | Must be non-empty when set. For the Qwen3.5-4B profile, omission resolves to `<model.path>/adapters`. |
| `model.snapshot_dir` | optional string; omitted (`None`) | `KILN_MODEL_SNAPSHOT_DIR` (implemented) | `KILN_MODEL_SNAPSHOT_DIR` | Must be non-empty when set. The environment alias uniquely treats an empty or whitespace-only value as a request to clear the TOML value. Without a value, Kiln tries a location beside the model and then the system temporary directory. |
| `model.served_model_id` | optional string; omitted (`None`) | `KILN_MODEL_SERVED_MODEL_ID` (target only; not implemented) | `KILN_SERVED_MODEL_ID` (legacy; works) | Must be non-empty when set. Otherwise the effective id is the final slash-separated component of `model.model_id` (`Qwen3.5-4B` by default). `serve --served-model-id` sets the working alias before load. |

## `[memory]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `memory.num_blocks` | optional unsigned integer; omitted (`None`) | `KILN_MEMORY_NUM_BLOCKS` (target only; not implemented) | `KILN_NUM_BLOCKS` (legacy; works) | Must be greater than zero when set. Omission invokes backend-aware automatic KV-block sizing. |
| `memory.gpu_memory_gb` | optional finite number; omitted (`None`) | `KILN_MEMORY_GPU_MEMORY_GB` (target only; not implemented) | `KILN_GPU_MEMORY_GB` (legacy; works) | Must be finite and greater than zero. Units are GiB. **Migration limitation:** the TOML value is validated but is not forwarded to the lower-level VRAM detector; use the working environment alias when an override must take effect today. |
| `memory.inference_memory_fraction` | finite number; `0.7` | `KILN_MEMORY_INFERENCE_MEMORY_FRACTION` (target only; not implemented) | `KILN_INFERENCE_MEMORY_FRACTION` (legacy; works) | Loader validation accepts `0.0..=1.0`; real-state construction clamps the configured value to `0.1..=1.0` before KV sizing. The remainder is available to the training budget. |
| `memory.training_memory_gb` | optional finite number; omitted (`None`) | `KILN_MEMORY_TRAINING_MEMORY_GB` (target only; not implemented) | `KILN_TRAINING_MEMORY_GB` (legacy; works) | Must be finite and greater than zero when set. Explicit training-memory budget in GiB. |
| `memory.kv_cache_fp8` | boolean; `false` | `KILN_MEMORY_KV_CACHE_FP8` (target only; not implemented) | `KILN_KV_CACHE_FP8` (legacy; works) | Requests E4M3FN KV storage. Backend storage policy may reject or disable the request when unsupported. |
| `memory.cuda_graphs` | boolean; `true` | `KILN_MEMORY_CUDA_GRAPHS` (target only; not implemented) | `KILN_CUDA_GRAPHS` (legacy; works) | CUDA-only request. Non-CUDA backends ignore it, and a serving profile with live graph capture disabled selects eager-only execution regardless of this value. |

## `[training]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `training.grad_checkpoint_segments` | optional unsigned integer; omitted (`None`) | `KILN_TRAINING_GRAD_CHECKPOINT_SEGMENTS` (target only; not implemented) | `KILN_GRAD_CHECKPOINT_SEGMENTS` (legacy; works) | Must be greater than zero when set. **Migration limitation:** training admission and lower training crates still consult the environment directly; use the working alias for effective runtime behavior today. |
| `training.no_grad_checkpoint` | boolean; `false` | `KILN_TRAINING_NO_GRAD_CHECKPOINT` (target only; not implemented) | `KILN_NO_GRAD_CHECKPOINT` (legacy; works) | **Migration limitation:** the TOML mirror is validated but is not the runtime authority; use the working alias today. Disabling checkpointing can materially increase training memory. |
| `training.checkpoint_interval` | optional unsigned integer; omitted (`None`) | `KILN_TRAINING_CHECKPOINT_INTERVAL` (target only; not implemented) | `KILN_CHECKPOINT_INTERVAL` (legacy; works) | Must be greater than zero when set. Number of committed optimizer steps between checkpoints; per-job configuration overrides it. Omission disables periodic checkpoints. |
| `training.webhook_url` | optional string; omitted (`None`) | `KILN_TRAINING_WEBHOOK_URL` (implemented) | `KILN_TRAINING_WEBHOOK_URL` | Must be a non-empty valid HTTP(S) URL. An exactly empty environment value clears a TOML URL; whitespace is not a clearing value and fails validation. Delivery is fire-and-forget with a five-second timeout after terminal state is recorded. |
| `training.max_queued_jobs` | unsigned integer; `32` | `KILN_TRAINING_MAX_QUEUED_JOBS` (implemented) | `KILN_TRAINING_MAX_QUEUED_JOBS` | Must be greater than zero. At capacity, submissions return HTTP 503 with `Retry-After: 30`. |
| `training.max_tracked_jobs` | unsigned integer; `1024` | `KILN_TRAINING_MAX_TRACKED_JOBS` (implemented) | `KILN_TRAINING_MAX_TRACKED_JOBS` | Must be greater than zero and at least `max_queued_jobs`. Counts queued, running, completed, and failed entries. |
| `training.tracked_job_ttl_secs` | unsigned integer; `604800` | `KILN_TRAINING_TRACKED_JOB_TTL_SECS` (implemented) | `KILN_TRAINING_TRACKED_JOB_TTL_SECS` | Must be greater than zero. Terminal entries older than the TTL are removed; active jobs are never age-evicted. |

Training GPU work is also governed by `server.serving_profile`; the default
`stable` profile does not grant training GPU ownership.

## `[logging]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `logging.level` | string; `"info"` | `KILN_LOGGING_LEVEL` (target only; not implemented) | `KILN_LOG_LEVEL` (legacy; works) | One of `trace`, `debug`, `info`, `warn`, `error`, or a valid `tracing_subscriber::EnvFilter` directive such as `kiln=debug,tower_http=warn`. `RUST_LOG` overrides the effective filter but does not make an invalid typed value valid. |
| `logging.format` | string; `"auto"` | `KILN_LOGGING_FORMAT` (target only; not implemented) | `KILN_LOG_FORMAT` (legacy; works) | Exactly `auto`, `json`, `pretty`, `text`, or `human`. `auto` is pretty on a stderr TTY and JSON otherwise; `text` and `human` select pretty output. |

Logging is bootstrapped before the full file is validated so parse failures can
be reported through the requested renderer. The authoritative loader then
validates the complete file and stops startup if it differs or is invalid.
Malformed or non-UTF-8 `RUST_LOG` is fatal.

## `[prefix_cache]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `prefix_cache.enabled` | boolean; `true` | `KILN_PREFIX_CACHE_ENABLED` (implemented) | `KILN_PREFIX_CACHE_ENABLED` | Enables reuse of KV blocks and recurrent-state snapshots for shared prompt prefixes. |
| `prefix_cache.max_blocks` | optional unsigned integer; omitted (`None`) | `KILN_PREFIX_CACHE_MAX_BLOCKS` (implemented) | `KILN_PREFIX_CACHE_MAX_BLOCKS` | Must be greater than zero when set. `None` resolves to half of the allocated KV block pool. |
| `prefix_cache.max_entries` | optional unsigned integer; omitted (`None`) | `KILN_PREFIX_CACHE_MAX_ENTRIES` (implemented) | `KILN_PREFIX_CACHE_MAX_ENTRIES` | Must be greater than zero when set. `None` resolves from detected VRAM and per-entry recurrent-state bytes, with at least one entry. |

## `[speculative]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `speculative.enabled` | boolean; `false` | `KILN_SPECULATIVE_ENABLED` (target only; not implemented) | `KILN_SPEC_ENABLED` (legacy; works) | If false, effective method is off. If true while `method = "off"`, backward compatibility selects `skip_layer`. Because of the lower runtime re-read described below, use the exact value `1` or `0` (or whitespace-free `true` or `false`) today. |
| `speculative.method` | string enum; `"off"` | `KILN_SPECULATIVE_METHOD` (target only; not implemented) | `KILN_SPEC_METHOD` (legacy; works) | TOML accepts `off`, `skip_layer`, or `mtp`. The environment parser also accepts `none`, `0`, `false`; `skiplayer`, `skip-layer`, `self`; and `native_mtp`, `native-mtp`. A non-off environment method implicitly enables speculative decoding when `KILN_SPEC_ENABLED` is absent. |
| `speculative.num_speculative_tokens` | unsigned integer; `256` | `KILN_SPECULATIVE_NUM_SPECULATIVE_TOKENS` (target only; not implemented) | `KILN_SPEC_NUM_TOKENS` (legacy; works) | Must be greater than zero. Draft proposal upper bound; MTP may have fewer native draft layers. Use an undecorated decimal value: the typed loader trims whitespace, but the lower runtime re-read does not. |
| `speculative.draft_layers` | unsigned integer; `8` | `KILN_SPECULATIVE_DRAFT_LAYERS` (target only; not implemented) | `KILN_SPEC_DRAFT_LAYERS` (legacy; works) | Must be greater than zero. The model-specific speculative validator additionally requires fewer than the model's transformer-layer count; an invalid skip-layer shape currently falls back to non-speculative generation at request dispatch. Use an undecorated decimal value because the lower runtime re-read does not trim whitespace. |

**Known migration limitation:** direct real completion dispatch currently builds
its speculative settings from `KILN_SPEC_*` again instead of consuming the
loaded `[speculative]` object. Consequently, TOML-only speculative settings are
not authoritative today, while the listed environment aliases are. This also
means `kiln config` and config hashes can describe a TOML value that the direct
request path does not use. The lower boolean reader recognizes only exact `1`
or case-insensitive, whitespace-free `true` as true, so centralized-valid
values such as `yes` and `on` can still result in runtime false. Speculative
dispatch is currently relevant to direct non-streaming generation;
batching-engine generation is non-speculative, and direct SSE generation
settles through single-token decode.

## `[streaming_prefill]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `streaming_prefill.enabled` | boolean; `false` | `KILN_STREAMING_PREFILL_ENABLED` (target only; not implemented) | `KILN_STREAMING_PREFILL` (legacy; works) | Environment value forces streaming on or off when spelled `1`/`true`/`yes` or `0`/`false`/`no`. Do not use `on` or `off`: the typed loader accepts them, but the lower reader treats them as absent. Without a recognized override, runtime backend policy automatically enables it for CUDA and ROCm prompts of at least 2048 tokens and Metal prompts of at least 2048 tokens; Vulkan and CPU default off. |
| `streaming_prefill.tile_tokens` | unsigned integer; `8192` | `KILN_STREAMING_PREFILL_TILE_TOKENS` (target only; not implemented) | `KILN_STREAMING_TILE_TOKENS` (legacy; works) | Must be a positive multiple of 64. Use an undecorated decimal value because the lower runtime re-read does not trim whitespace. Runtime backend defaults, absent a recognized environment override, are currently 1024 on CUDA/ROCm and 2048 on Metal/Vulkan rather than this generic typed default. |
| `streaming_prefill.last_token_lm_head` | boolean; `true` | `KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD` (target only; not implemented) | `KILN_STREAMING_LAST_TOKEN_LM_HEAD` (legacy; works) | When true, the final streaming tile computes the LM head only for its last row. Use `0`, `false`, or `no` to disable it; the lower reader incorrectly treats centralized-valid `off` as true. |

**Known migration limitation:** these typed fields are currently documentation
and validation mirrors. Lower model helpers read the listed environment aliases
directly and otherwise use backend policy; TOML-only values are not forwarded.
Use `KILN_STREAMING_PREFILL=0` to force streaming off today rather than relying
on the typed default, because the backend can auto-enable it for long prompts.
Use bare decimal tile sizes; avoid the centralized-only `on`/`off` boolean
spellings until the duplicate parsers are removed.
Internal threshold and training-tile controls are intentionally not part of
this public configuration surface.

## `[adapters]`

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
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

| TOML field template | Type and default | Canonical env target | Working override today | Validation and effective semantics |
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

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
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

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
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

| TOML field | Type and exact default | Canonical env target | Working override today | Validation and effective semantics |
|---|---|---|---|---|
| `agent.self_improve_interval_hours` | optional unsigned integer; omitted (`None`) | `KILN_AGENT_SELF_IMPROVE_INTERVAL_HOURS` (target only; not implemented) | none | Omission disables scheduling. `0` is accepted and also results in no scheduler. The first run occurs one full interval after startup; cadence is persisted under the adapter directory. |
| `agent.self_improve` | optional structured value; omitted (`None`) | `KILN_AGENT_SELF_IMPROVE` (target only; not implemented) | none | Request template submitted to the same self-improvement path as the API. This value is intentionally open structured data; its inner request contract is validated by that subsystem rather than by `KilnConfig`. |
| `agent.max_concurrent_runs` | unsigned integer; `2` | `KILN_AGENT_MAX_CONCURRENT_RUNS` (target only; not implemented) | none | Must be greater than zero. Embedded pi runs above the limit queue FIFO. |
| `agent.run_timeout_secs` | unsigned integer; `900` | `KILN_AGENT_RUN_TIMEOUT_SECS` (target only; not implemented) | none | Must be at least `10`. A per-run timeout can override the server default. |

## Effective values and provenance

Kiln currently exposes several complementary, partial views:

- `kiln config --file <path>` performs authoritative parsing and validation,
  then prints a human-oriented subset of resolved fields.
- `GET /v1/config` reports runtime diagnostics for serving profile, effective
  decode width, VRAM/KV state, checkpoint segmentation, memory budgets, and
  generation defaults. It is not a serialization of all accepted TOML.
- `/health` and `/v1/debug/model-state` expose config hashes and other runtime
  identity data.
- Startup logs record serving-profile provenance and configured/backend/final
  decode-width sources.

Explicit source tracking (`default`, `config_file`, or `environment`) currently
exists for `server.serving_profile`, `server.deterministic`,
`server.stream_stall_grace_ms`, the three batching/prefill budgets, and
`server.max_decode_batch`. Other fields have resolved values but do not yet
carry per-field source metadata.

The `kiln_env_config_hash` binds the serialized effective typed configuration
and the process's complete `KILN_*` environment map. It is an identity digest,
not a human-readable effective-config dump.

There is not yet one endpoint or CLI mode that dumps every effective field,
its source, backend-derived adjustment, and restart requirement. Until that
lands, use the typed validation command plus startup/runtime diagnostics, and
account for the migration limitations below.

## Known configuration migration limitations

These are current implementation facts, not recommended architecture:

1. **Speculative decoding:** `[speculative]` is parsed and validated, but the
   direct request path rebuilds defaults from `KILN_SPEC_*`. TOML-only values do
   not control serving.
2. **Streaming prefill:** `[streaming_prefill]` is parsed and validated, but
   lower model helpers read `KILN_STREAMING_*` and backend policy directly.
   TOML-only values do not control dispatch.
3. **Gradient checkpoint selection:**
   `training.grad_checkpoint_segments` and `training.no_grad_checkpoint` are
   typed mirrors while training admission and lower crates still read
   `KILN_GRAD_CHECKPOINT_SEGMENTS` and `KILN_NO_GRAD_CHECKPOINT`.
4. **VRAM override:** `memory.gpu_memory_gb` is typed and validated, but the
   detector currently consumes `KILN_GPU_MEMORY_GB` directly.
5. **Effective dump:** neither `kiln config` nor `/v1/config` covers the whole
   typed object with provenance and backend-derived values.
6. **Canonical names:** only 19 fixed fields currently implement the
   `KILN_<SECTION>_<FIELD>` name. Target-only names in this document must not be
   treated as aliases until implementation and conformance tests land.

The intended migration is to resolve every public runtime field exactly once,
pass immutable typed configuration into its owner, generate the environment
contract mechanically, and reject direct production environment reads outside
that boundary. Internal experiment and qualification flags should remain
separate from this public API rather than being promoted by documentation.
