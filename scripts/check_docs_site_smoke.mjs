#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, extname, relative, sep, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import process from 'node:process';

const repoRoot = resolve(import.meta.dirname, '..');
const configuredSiteRoot = process.env.KILN_DOCS_SITE_ROOT?.trim() || 'docs/site';
const siteRoot = resolve(repoRoot, configuredSiteRoot);
const generatedDocsRequired = process.env.KILN_DOCS_REQUIRE_GENERATED === 'true';
const staticOnly = process.env.KILN_DOCS_SMOKE_STATIC_ONLY === 'true';
const mobileViewport = { width: 390, height: 844, deviceScaleFactor: 2, isMobile: true };
const desktopViewport = { width: 1440, height: 900, deviceScaleFactor: 1 };
const mobileOverflowTolerancePx = 2;
const dashboardImageSrcset = 'assets/server-ui-dashboard-720.webp 720w, assets/server-ui-dashboard-1440.webp 1440w, assets/server-ui-dashboard.webp 2880w';
const dashboardImageSizes = '(max-width: 760px) calc(100vw - 32px), 960px';
const demoImageSizes = '(max-width: 780px) calc(100vw - 32px), 1180px';
const demoImageNames = ['dashboard', 'playground', 'training', 'adapters'];
const runtimeEnvironmentContract = JSON.parse(
  readFileSync(resolve(repoRoot, 'contracts/runtime-env-direct-reads-v1.json'), 'utf8'),
);
const runtimeEnvironmentSummary = runtimeEnvironmentContract.summary;
const docsManifest = JSON.parse(
  readFileSync(resolve(repoRoot, 'docs/site/docs-manifest.json'), 'utf8'),
);

function publishedPath(relativePath) {
  const path = relative(repoRoot, resolve(siteRoot, relativePath)).split(sep).join('/');
  return relativePath.endsWith('/') && !path.endsWith('/') ? `${path}/` : path;
}

const pages = [
  { label: 'Home', path: publishedPath('index.html'), currentLabel: null },
  { label: 'Quickstart', path: publishedPath('quickstart.html'), currentLabel: 'Quickstart' },
  { label: 'GRPO Guide', path: publishedPath('grpo.html'), currentLabel: 'GRPO Guide' },
  { label: 'Evals', path: publishedPath('evals.html'), currentLabel: 'Evals' },
  { label: 'API Reference', path: publishedPath('api.html'), currentLabel: 'API Reference' },
  { label: 'CLI Reference', path: publishedPath('cli.html'), currentLabel: 'CLI Reference' },
  { label: 'Troubleshooting', path: publishedPath('troubleshooting.html'), currentLabel: 'Troubleshooting' },
  { label: 'Architecture', path: publishedPath('architecture.html'), currentLabel: 'Architecture' },
  { label: 'Demo', path: publishedPath('demo/index.html'), currentLabel: 'Demo' },
];

const expectedNavLabels = [
  'Quickstart',
  'Documentation',
  'GRPO Guide',
  'Evals',
  'API Reference',
  'CLI Reference',
  'Demo',
  'Troubleshooting',
  'Architecture',
];

const expectedFooterLinks = [
  { label: 'Documentation', localPath: publishedPath('docs/') },
  { label: 'Quickstart', localPath: publishedPath('quickstart.html') },
  { label: 'GRPO Guide', localPath: publishedPath('grpo.html') },
  { label: 'Evals', localPath: publishedPath('evals.html') },
  { label: 'API Reference', localPath: publishedPath('api.html') },
  { label: 'CLI Reference', localPath: publishedPath('cli.html') },
  { label: 'Demo', localPath: publishedPath('demo/') },
  { label: 'Troubleshooting', localPath: publishedPath('troubleshooting.html') },
  { label: 'Architecture', localPath: publishedPath('architecture.html') },
  { label: 'Benchmarks', localPath: publishedPath('docs/benchmarks/') },
  { label: 'Changelog', href: 'https://github.com/ericflo/kiln/blob/main/CHANGELOG.md' },
  { label: 'Security', href: 'https://github.com/ericflo/kiln/blob/main/SECURITY.md' },
  { label: 'License', href: 'https://github.com/ericflo/kiln/blob/main/LICENSE' },
];

const expectedProductFooterOrder = [
  'Home',
  'README',
  'Quickstart',
  'Documentation',
  'GRPO Guide',
  'Evals',
  'API Reference',
  'CLI Reference',
  'Demo',
  'Troubleshooting',
  'Architecture',
  'Benchmarks',
  'Changelog',
  'Security',
  'License',
];

const expectedEmbeddedUiHelpLinks = [
  { label: 'Quickstart', href: 'https://ericflo.github.io/kiln/quickstart.html' },
  { label: 'Documentation', href: 'https://ericflo.github.io/kiln/docs/' },
  { label: 'OpenEnv Guide', href: 'https://ericflo.github.io/kiln/docs/openenv/' },
  { label: 'GRPO Guide', href: 'https://ericflo.github.io/kiln/grpo.html' },
  { label: 'API Reference', href: 'https://ericflo.github.io/kiln/api.html' },
  { label: 'CLI Reference', href: 'https://ericflo.github.io/kiln/cli.html' },
  { label: 'Demo', href: 'https://ericflo.github.io/kiln/demo/' },
  { label: 'Troubleshooting', href: 'https://ericflo.github.io/kiln/troubleshooting.html' },
  { label: 'Architecture', href: 'https://ericflo.github.io/kiln/architecture.html' },
];

const expectedEmbeddedUiAccessibleControls = [
  { selector: '#chat-input', labelTerms: ['message'], descriptionTerms: ['quick prompt', 'selected adapter', 'base model'] },
  { selector: '#chat-send', labelTerms: ['send'] },
  { selector: '#chat-stop', labelTerms: ['stop'] },
  { selector: '#chat-clear', labelTerms: ['clear'] },
  { selector: '#sft-examples', labelTerms: ['training examples'], descriptionTerms: ['json array', 'messages'] },
  { selector: '#grpo-groups', labelTerms: ['grpo groups'], descriptionTerms: ['json array', 'completions', 'reward'] },
  { selector: '#upload-name', labelTerms: ['adapter name'], descriptionTerms: ['adapter archive', 'saved adapter directory'] },
  { selector: '#upload-archive', labelTerms: ['adapter archive'], descriptionTerms: ['adapter archive', '.tar.gz'] },
  { selector: '#merge-output-name', labelTerms: ['output name'], descriptionTerms: ['saved adapter name', 'path-safe'] },
];

const demoPagePath = publishedPath('demo/index.html');
const homePagePath = publishedPath('index.html');
const quickstartPagePath = publishedPath('quickstart.html');
const apiPagePath = publishedPath('api.html');
const cliPagePath = publishedPath('cli.html');
const architecturePagePath = publishedPath('architecture.html');
const troubleshootingPagePath = publishedPath('troubleshooting.html');

const generatedDocsPages = [
  {
    label: 'Documentation hub',
    path: publishedPath('docs/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/',
    h1: 'Kiln documentation',
    terms: [
      'Configuration reference',
      'GRPO Guide',
      'Evals Guide',
      'Configuration schema',
      'HTTP API contract',
      'Observability API schema',
      'Architecture deep dive',
      'Native SFT profile',
      'SFT ingestion and row identity',
      'SFT tokenization and loss',
      'Native training checkpoints',
      'Dataset splits and train/eval separation',
      'Thinking budgets',
      'CI and local qualification policy',
      'Verification policy',
      'Runtime environment inventory',
      'Source-parsing test debt',
    ],
  },
  {
    label: 'Serving benchmark protocol',
    path: publishedPath('docs/serving-benchmark-protocol/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/serving-benchmark-protocol/',
    h1: 'Serving benchmark protocol',
    anchors: ['process-boundary', 'timing', 'retention'],
    terms: [
      'ordinary wall-clock timing',
      'does not impose a host temperature policy',
      'must not require a particular laptop',
      'no CPU or memory ceiling',
      'monotonic wall-clock time',
      'Telemetry is diagnostic evidence',
      'do not inject gfx1151',
      'historical records and are not accepted',
    ],
  },
  {
    label: 'Serving profiles',
    path: publishedPath('docs/serving-profiles/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/serving-profiles/',
    h1: 'Serving profiles',
    anchors: ['exact-policy', 'stable', 'experimental', 'maintenance', 'move-between-profiles'],
    terms: [
      'A serving profile is an immutable, process-wide GPU ownership policy',
      'The profile is one policy boundary, not a hardware selector',
      'Device marketing names, vendor or device IDs, driver strings',
      'has no startup setting for selecting a saved adapter',
      'This is a product limitation, not a hidden configuration option',
      'Start a separate experimental process to load, evaluate, or serve the unmerged adapter',
    ],
  },
  {
    label: 'Latency observability',
    path: publishedPath('docs/latency-observability/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/latency-observability/',
    h1: 'Latency observability',
    anchors: ['choose-an-observation-surface', 'understand-the-clocks', 'use-the-rolling-decode-endpoint', 'diagnose-a-pause'],
    terms: [
      'Tokens from concurrent streams are never paired with one another',
      'Decode cadence, not end-to-end throughput',
      'does not mean that the remote client received or rendered the bytes',
      'Never sum every phase',
      'It is not evidence by itself of allocator activity',
      'Do not compare it with end-to-end output throughput',
      'Hardware receipts and host names are evidence about specific benchmark runs',
    ],
  },
  {
    label: 'Thinking budgets',
    path: publishedPath('docs/thinking-budgets/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/thinking-budgets/',
    h1: 'Thinking budgets',
    anchors: ['set-a-request-budget', 'know-when-a-budget-applies', 'understand-enforcement', 'find-the-record-on-each-api-surface'],
    terms: [
      'limits an open model reasoning block without disabling thinking or ending the completion',
      'Explicitly unlimited, even when the server has a finite default',
      'If both are reached at the same token boundary, the token trigger is reported',
      'A budget does not turn thinking on',
      'Those close tokens enter KV history',
      'Do not treat triggered as proof of successful closure',
      'Kiln does not manufacture one aggregate batch outcome',
    ],
  },
  {
    label: 'Thinking budget schema',
    path: publishedPath('docs/thinking-budget-schema/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/thinking-budget-schema/',
    h1: 'Thinking budget schema',
    anchors: [],
    terms: [
      'The token and decode-time dimensions resolve independently',
      'Maximum generated thinking tokens before Kiln closes an open reasoning block',
      'Maximum decode time in milliseconds, starting at the first decode candidate',
      "Whether the tokenizer's complete reasoning close sequence entered model history",
      'Recent-request form of the effective record',
    ],
  },
  {
    label: 'Benchmarks',
    path: publishedPath('docs/benchmarks/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/benchmarks/',
    h1: 'Benchmarks',
    anchors: ['current-measured-position', 'what-happened', 'why-the-old-0455-number-did-not-describe-decode-speed'],
    terms: [
      'Vulkan decode is back in its historical range',
      '13.46 tok/s',
      'Latest verified source result',
      'not yet a published release',
      '0.142 tok/s',
      'roughly 87×',
      '28d8c6028',
      'same Qwen3.5-4B short serial workload',
      'Device name, vendor ID, device ID, PCI identity, and driver name are never policy inputs',
      '0.455 aggregate output',
      '83.2 seconds',
      'about 13.0 decode tok/s',
      'predates the July 27 regression',
      'does not claim high-concurrency parity',
    ],
  },
  {
    label: 'Configuration Reference',
    path: publishedPath('docs/configuration/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/configuration/',
    h1: 'Configuration Reference',
    anchors: ['optimizer-support-is-a-resident-capability-not-configuration', 'tape-scope-is-internal-execution-authority-not-configuration', 'frozen-parameters-are-constants-not-optimizer-leaves', 'backend-owned-sft-loss-route-is-not-configuration'],
    terms: [
      'KILN_<SECTION>_<FIELD>',
      'Strict failure behavior',
      '16 top-level sections and 112 fixed leaf fields',
      '107 implement the canonical mechanical environment name',
      'no field has an alternate environment spelling',
      'complete profile-gated set',
      'server.serving_profile is the single opt-in authority',
      'accelerator.cuda_marlin_profile',
      'memory.kv_force_blocks',
      'not an execution experiment',
      '82 former public spellings',
      '5 are config-file-only',
      'paths.cache_root',
      'KILN_PATHS_CACHE_ROOT',
      'startup working directory',
      'operating-system account database',
      'XDG_CACHE_HOME',
      'BLAS/rocBLAS autotune data',
      'Vulkan pipeline caches',
      'transposed model weights',
      'GET /v1/config.paths',
      'cannot change during the process lifetime',
      'External accelerator environment safety',
      'CUDA_VISIBLE_DEVICES',
      'NVIDIA_VISIBLE_DEVICES',
      'CUDA_DEVICE_ORDER',
      'ROCR_VISIBLE_DEVICES',
      'HIP_VISIBLE_DEVICES',
      'GPU_DEVICE_ORDINAL',
      'ZE_AFFINITY_MASK',
      'ONEAPI_DEVICE_SELECTOR',
      'SYCL_DEVICE_FILTER',
      'MESA_VK_DEVICE_SELECT',
      'DRI_PRIME',
      'VK_ICD_FILENAMES',
      'VK_DRIVER_FILES',
      'presence of this closed set',
      'before model weights are loaded',
      'credential-provider adapter performs every process-environment secret lookup',
      'missing, non-Unicode, or whitespace-only values fail closed',
      'memory.cuda_graph_cache_entries',
      'KILN_MEMORY_CUDA_GRAPH_CACHE_ENTRIES',
      'model.accelerator_weight_upload_mib_per_second',
      'model.checkpoint_read_mib_per_second',
      'KILN_MODEL_ACCELERATOR_WEIGHT_UPLOAD_MIB_PER_SECOND',
      'active during inference',
      'cumulative eager base-model source-byte schedule',
      'The unqualified batched CUDA graph route remains unavailable',
      'KILN_BATCHING_MODE',
      'KILN_BATCHING_ROWWISE_DECODE',
      'KILN_BATCHING_PREFIX_AWARE_ADMISSION',
      'KILN_BATCHING_PREFILL_ADMISSION_QUANTUM',
      'batching.actor_cycle_idle_ms',
      'KILN_BATCHING_ACTOR_CYCLE_IDLE_MS',
      'kiln_batching_engine_actor_cycle_idle_seconds_total',
      'KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MODE',
      'KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MAX_BATCH',
      'KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_WAIT_US',
      'KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_MIXED_SEQ_LENS',
      'KILN_DECODE_BATCHER',
      'KILN_DECODE_BATCH_MAX',
      'KILN_DECODE_BATCH_WAIT_US',
      'KILN_DECODE_BATCH_MIXED_SEQ',
      'effective_decode_width',
      'decode_runtime.batching_configuration',
      'streaming_prefill.mode',
      'streaming_prefill.threshold_tokens',
      'streaming_prefill.tile_tokens',
      'streaming_prefill.tape_tile_tokens',
      'streaming_prefill.detached_full_attn_tile_tokens',
      'streaming_prefill.last_token_lm_head',
      'KILN_STREAMING_PREFILL_MODE',
      'KILN_STREAMING_PREFILL_THRESHOLD_TOKENS',
      'KILN_STREAMING_PREFILL_TILE_TOKENS',
      'KILN_STREAMING_PREFILL_TAPE_TILE_TOKENS',
      'KILN_STREAMING_PREFILL_DETACHED_FULL_ATTN_TILE_TOKENS',
      'KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD',
      'KILN_STREAMING_PREFILL_ENABLED',
      'KILN_STREAMING_TILE_TOKENS',
      'KILN_TAPE_STREAMING_TILE_TOKENS',
      'KILN_DETACHED_FULL_ATTN_TILE_TOKENS',
      'KILN_STREAMING_LAST_TOKEN_LM_HEAD',
      'TOML also has two removed fields',
      'speculative.enabled',
      'streaming_prefill.enabled',
      'prompt tokens >= 2048',
      'Detached tape replay',
      'inherited_from_tile_tokens_config_file',
      'prefill_runtime.streaming_prefill',
      'restart_required_to_change',
      'training.recompute_checkpoint_boundaries',
      'training.recompute_boundary_threshold_tokens',
      'training.checkpoint_boundary_anchor_stride',
      'training.checkpoint_boundary_cache_gb',
      'KILN_TRAINING_RECOMPUTE_CHECKPOINT_BOUNDARIES',
      'KILN_TRAINING_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS',
      'KILN_TRAINING_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE',
      'KILN_TRAINING_CHECKPOINT_BOUNDARY_CACHE_GB',
      'KILN_RECOMPUTE_CHECKPOINT_BOUNDARIES',
      'KILN_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS',
      'KILN_CHECKPOINT_BOUNDARY_ANCHOR_STRIDE',
      'KILN_CHECKPOINT_BOUNDARY_CACHE_GB',
      'Retired non-canonical names are ignored',
      'string enum; "auto"',
      'positive unsigned integer; 8192',
      '"auto" or positive unsigned integer; "auto"',
      'positive floating-point GiB value; 6.0',
      'Zero, negative, overflowing, malformed, and non-UTF-8 environment values stop startup',
      'Zero and strings other than auto fail startup',
      'must be finite, positive, convert to at least one byte',
      'Optimizer support is a resident capability, not configuration',
      'kiln.training-optimizer-support',
      'backend_implementation',
      'portable_reference',
      'native_device_hook',
      'optimizer_tuple_kinds',
      'optimizer_tuple',
      'workloads',
      'unavailable_reason',
      'allowed_optimizer_kinds',
      'live_memory_admission_required',
      'backend_maximum',
      'model_maximum',
      'maximum is their effective minimum',
      'null backend_maximum means the backend adds no ceiling',
      'model still supplies one',
      'rank 2..=1024',
      'never lowers rank',
      'CPU processes expose the portable F32 optimizer tuples',
      'This table is not a server-execution matrix',
      'exact native backend and device identity',
      'no Marlin-packed model projection',
      'kt_tape_authoritative',
      'full_logits',
      'phase-B backward route',
      'resolved_lora_parameter_dtype: null',
      'coarse compatibility summary',
      'all four workloads and all three optimizer kinds',
      'distill_refresh',
      'separate exact SFT and OPD phase plans',
      'prepares the exact SFT rows',
      'reserves the maximum sequential working set',
      'larger of the two sequential working sets',
      'before checkpoint loading or corpus scanning',
      'Cheap teacher-alias validation and metadata pinning',
      'Remote/local teacher materialization, checkpoint loading, corpus scanning, memory preflight, and GPU reservation do not',
      'For a queued resume',
      'compact checkpoint-ID/manifest-digest identity',
      'Before memory reservation at dequeue',
      'not an external-file snapshot',
      'filesystem race',
      'GET /v1/recipes',
      'admission: {supported, unavailable_reason}',
      'Metal',
      'KILN_BF16_STOCHASTIC_ROUND',
      'KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK',
      'KILN_CUDA_TRAINING_OPTIMIZER_FALLBACK',
      'KILN_ROCM_TRAINING_OPTIMIZER_FALLBACK',
      'KILN_METAL_TRAINING_OPTIMIZER_FALLBACK',
      'KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK',
      'Tape scope is internal execution authority, not configuration',
      'required workload substrate',
      'thread-local scope internally',
      'complete inference boundary',
      'GDN chunkwise recurrence',
      "CUDA's weight-aware embedding lookup",
      'KILN_USE_TAPE_FORWARD',
      'KILN_USE_TAPE_FLASH_ATTN',
      'KILN_USE_TAPE_SDPA',
      'KILN_USE_TAPE_LORA_ADD',
      'KILN_USE_TAPE_GDN',
      'KILN_USE_TAPE_GDN_CONV',
      'KILN_USE_TAPE_GDN_QK_NORM',
      'KILN_USE_TAPE_GDN_GATED_NORM',
      'removed without compatibility aliases or replacement fields',
      'must not disable the tape, bypass an individual recorder',
      'does not by itself qualify numerical correctness',
      'Frozen parameters are constants, not optimizer leaves',
      'train only LoRA A/B tensors',
      'Embedding tables',
      'base projection matrices',
      'RMSNorm and GDN gated-RMSNorm weights',
      'GDN gate parameters',
      'MTP projection weights',
      'loss-head transposes',
      'saved constants',
      'do not execute a dWeight matrix multiplication',
      'original full LoRA A/B tensor IDs',
      'pads its B contribution to the full shape',
      'tape-aware reshapes preserve the activation chain',
      'temporary B slices are never trainable leaves',
      'does not by itself qualify backend throughput',
      'Backend-owned SFT loss route is not configuration',
      'kt_tape_flce',
      'vulkan_active_rows',
      'full_logits',
      'not accepted TOML enums',
      'mechanical naming rule does not produce an environment variable for this choice',
      'KILN_USE_FLCE',
      'not a deprecated alias',
      'typed loader does not consume it',
      'HTTP 413',
      'loss workspace ... (route=<route>)',
      'training_invalid_request',
      'runtime.sft_loss_route',
      'SFT extends the same object to v4',
      'kiln.training-checkpoint-planning.v3',
      'exact-resume drift',
      'GET /v1/config',
      'GET /health',
      '/v1/health',
      'GET /v1/debug/model-state',
      'training.checkpoint_boundary_policy',
      'Runtime Config training group',
    ],
  },
  {
    label: 'Configuration Schema',
    path: publishedPath('docs/configuration-schema/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/configuration-schema/',
    h1: 'Configuration Schema',
    terms: [
      'Kiln server configuration v1',
      'Canonical environment target',
      'Alternate environment spelling',
      'Profile gate',
      'experimental when enum "attention_mlp", "attention_mlp_gdn"',
      'maintenance when minimum 1',
      'server.serving_profile',
      'KILN_SERVER_SERVING_PROFILE',
      'x-kiln-retired-environment-count',
      'KILN_SERVING_PROFILE',
      'KILN_SERVER_SERVING_PROFILE',
      'memory.cuda_graph_cache_entries',
      'KILN_MEMORY_CUDA_GRAPH_CACHE_ENTRIES',
      'accelerator.rocm_synchronization_mode',
      'accelerator.cuda_marlin_profile',
      'memory.kv_force_blocks',
      'server.terminal_access',
      'training.logit_cache_dir',
      'adapters.library_url',
      'agent.runs_access',
      'agent.pi_bin',
      'agent.pi_sessions_dir',
      'streaming_prefill.enabled',
      'x-kiln-removed-toml-field-replacements',
      'teachers.credentials.<id>.api_key_env',
      'Composition and conditional rules',
    ],
  },
  {
    label: 'HTTP API Contract',
    path: publishedPath('docs/http-api/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/http-api/',
    h1: 'HTTP API Contract',
    terms: [
      'Kiln HTTP API',
      '108 paths',
      '122 operations',
      'complete',
      'DELETE 13, GET 57, POST 51, PUT 1',
      '/v1/agent/runs/{id}/events',
      '/v1/corrections/mark_trained',
      '/v1/eval/datasets/{name}/preview',
      '/v1/eval/jobs/{job_id}/replay',
      '/v1/library/install/{id}',
      '/v1/openenv/tasks',
      '/v1/preflight/tier_defaults',
      '/v1/terminal/ws',
      '/v1/training/grpo',
      'multipart/form-data',
      'text/event-stream',
      'application/gzip',
      'completions::chat_completions',
      'ApiError',
      'OpenEnvTaskCatalogResponse',
      'Payload components',
      '141 are field-complete',
      '0 remain migration pending',
    ],
  },
  {
    label: 'Inference API Schema',
    path: publishedPath('docs/inference-schema/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/inference-schema/',
    h1: 'Inference API Schema',
    terms: [
      'Kiln inference API contract v1',
      'BatchCompletionRequest',
      'ChatCompletionRequest',
      'ChatCompletionResponse',
      'ChatCompletionChunkStream',
      'TextCompletionRequest',
      'TextCompletionResponse',
      'ThinkingBudgetConfigurationMetadata',
      'RolloutProvenanceV1',
      'accepted_and_ignored',
      'fallback_to_qwen3_thinking_general',
      'finite_json_number_only',
      'text/event-stream',
      'prompt_logprobs',
      'thinking_budget_tokens',
      'rollout_provenance',
      'Composition and conditional rules',
    ],
  },
  {
    label: 'Observability API Schema',
    path: publishedPath('docs/observability-schema/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/observability-schema/',
    h1: 'Observability API Schema',
    terms: [
      'Kiln Read-only Serving and Observability API v1',
      'HealthResponse',
      'ConfigResponse',
      'ModelStateResponse',
      'DebugDisabledResponse',
      'DebugProvenanceErrorResponse',
      'DecodeStatsSnapshot',
      'ModelsResponse',
      'RequestRecord',
      'CacheStatsResponse',
      'RocmGraphStats',
      'RequestThinkingBudget',
      'ModelStartupConfigResponse',
      'AcceleratorWeightUploadConfigResponse',
      'AcceleratorWeightUploadReport',
      'PrefixCacheConfigResponse',
      'closed object',
      'x-kiln-entrypoints',
    ],
  },
  {
    label: 'Artifact Lifecycle API Schema',
    path: publishedPath('docs/artifact-api-schema/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/artifact-api-schema/',
    h1: 'Artifact Lifecycle API Schema',
    terms: [
      'Kiln Artifact Lifecycle API',
      'AdapterUploadMultipart',
      'AdaptersResponse',
      'MergeAdapterRequest',
      'HfTrlExportManifestV1',
      'ImportPeftResponse',
      'RegisterTeacherRequest',
      'TeacherIdentityV1',
      'kiln_train_AdapterReceipt',
      'accepted_and_ignored',
      'source_execution_provenance',
      'identity_revision',
      'Composition and conditional rules',
      'x-kiln-entrypoints',
      'explicit CPU and placeholder build identities',
      'not hardware receipts or backend support claims',
      'runtime defaults, support predicates, qualification evidence, or benchmark receipts',
    ],
  },
  {
    label: 'Eval and Judgment API Schema',
    path: publishedPath('docs/eval-api-schema/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/eval-api-schema/',
    h1: 'Eval and Judgment API Schema',
    terms: [
      'Kiln Eval, Dataset Synthesis, and Judgment API',
      'EvalRunRequest',
      'EvalRunResponse',
      'EvalResult',
      'EvalJobInfo',
      'EvalSuite',
      'ScorerChoice',
      'CancelEvalJobResponse',
      'SynthesisPreview',
      'SynthesizeDatasetResponse',
      'CompileJudgmentResponse',
      'ValidateJudgmentResponse',
      'accepted_and_ignored',
      'thinking-budget-v1.schema.json',
      'Composition and conditional rules',
      'x-kiln-entrypoints',
    ],
  },
  {
    label: 'Training and Agent Control Plane API Schema',
    path: publishedPath('docs/control-plane-api-schema/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/control-plane-api-schema/',
    h1: 'Training and Agent Control Plane API Schema',
    terms: [
      'Kiln Training and Agent Control Plane API',
      'SftRequest',
      'GrpoRequest',
      'OpdRequest',
      'DistillRefreshRequest',
      'FrontDoorRequest',
      'TrainingJobDetail',
      'CancelTrainingJobResponse',
      'AgentRunEventsResponse',
      'AgentTrace',
      'RecipeRunRequest',
      'CorrectionRowInput',
      'PublishToLibraryResponse',
      'accepted_and_ignored',
      'http_501_not_implemented',
      'Composition and conditional rules',
      'x-kiln-entrypoints',
    ],
  },
  {
    label: 'Runtime Environment Inventory',
    path: publishedPath('docs/runtime-environment-inventory/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/runtime-environment-inventory/',
    h1: 'Runtime Environment Inventory',
    anchors: [
      'current-baseline',
      'classification-policy',
      'production-migration-owners',
      'literal-kiln_-catalog',
      'dynamic-read-catalog',
      'mutation-boundary',
      'migration-rule',
    ],
    terms: [
      `${runtimeEnvironmentSummary.read_call_sites} direct read call sites`,
      `${runtimeEnvironmentSummary.process_mutation_call_sites} process-mutation call sites`,
      `${runtimeEnvironmentSummary.literal_kiln_read_names} distinct literal KILN_*`,
      'Startup safety',
      'Credential provider',
      'Experimental/debug migration',
      `${runtimeEnvironmentSummary.read_call_sites_by_classification.experimental_debug}`,
      'Production migration owners',
      'the migration queue is empty',
      'crates/kiln-memory/src/startup_environment.rs',
      'crates/kiln-train/src/credential_provider.rs',
      'Process-environment mutation is forbidden in production execution',
      'rejects every experimental/debug migration read',
      'KILN_<SECTION>_<FIELD>',
      'do not edit by hand',
    ],
  },
  {
    label: 'Repository Artifact Retention',
    path: publishedPath('docs/artifact-retention/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/artifact-retention/',
    h1: 'Repository Artifact Retention',
    anchors: [
      'retention-boundary',
      'enforced-limits',
      'local-check',
      'historical-audit-artifacts',
      'creating-a-removal-manifest',
      'ci-scope',
    ],
    terms: [
      'contracts/repository-artifact-policy-v1.json',
      'Raw artifact suffixes',
      '1 MiB',
      '10 MiB',
      'hash of any raw local artifact',
      'NUL-delimited index records',
      'removed-raw-artifacts-2026-07-13-v1.json',
      'did not run a history filter',
      '--archive-current-offenders',
      'not hardware qualification',
    ],
  },
  {
    label: 'Verification Test Inventory',
    path: publishedPath('docs/verification-test-inventory/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/verification-test-inventory/',
    h1: 'Verification Test Inventory',
    anchors: [
      'current-baseline',
      'replacement-policy',
      'ownership-concentration',
      'migration-queue',
      'gate',
    ],
    terms: [
      'Tests: 0',
      'source reads: 0',
      'assertions: 0',
      'implementation_source_text',
      'qualification_driver_source_text',
      'generated_artifact_text',
      'zero-debt gate',
      'technical debt, not correctness evidence',
      'contracts/source-parsing-test-inventory-v1.json',
      'runtime test',
      'property/state-machine test',
      'python3 scripts/check_source_parsing_tests.py',
    ],
  },
  {
    label: 'Verification Policy',
    path: publishedPath('docs/verification-policy/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/verification-policy/',
    h1: 'Verification Policy',
    anchors: ['prohibited-evidence', 'evidence-ownership', 'migration-record'],
    terms: [
      'Tests must not infer correctness from implementation source text',
      'zero allowed tests, reads, and text assertions',
      'dynamic implementation paths',
      '112 source-text tests',
      'Backend capability and dispatch shape',
      'Tape scope and exact gradients',
      'ROCm stream and pointer lifetime',
      'Accelerator behavior is accepted only from a non-skipped, source-bound local qualification receipt',
      'python3 scripts/check_source_parsing_tests.py',
    ],
  },
  {
    label: 'Backend capability report',
    path: publishedPath('docs/backend-capabilities/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/backend-capabilities/',
    h1: 'Backend capability report',
    anchors: [
      'read-this-first',
      'current-source-snapshot',
      'open-coverage-work',
      'runtime-fallback-policy',
      'how-to-read-status-labels',
      'detailed-generated-inventory',
      'regenerate-and-validate',
    ],
    terms: [
      'It is a static source inventory',
      'does not compile a backend, execute the listed commands, detect hardware, or measure throughput',
      'determine support from capabilities and request constraints, not a device name',
      'must never become runtime dispatch, a default, or a support promise',
      'Feature fanout exists; it does not prove operation coverage or performance',
      'the host-fallback counter test validates counter attribution',
      'The report therefore marks that gate partial',
      'Commands are prescriptions, not execution receipts',
      'regeneration alone is not verification',
    ],
  },
  {
    label: 'Architecture deep dive',
    path: publishedPath('docs/architecture-deep-dive/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/architecture-deep-dive/',
    h1: 'Architecture deep dive',
    anchors: ['vulkan-policy-is-capability-driven', 'accelerator-ownership-and-serving-profiles', 'failure-and-health-boundaries'],
    terms: [
      'Startup establishes the runtime contract',
      'The production block manager divides K/V storage into 64-token blocks',
      'Vulkan runtime selection does not branch on a marketing device name',
      'vendor ID, or device ID',
      'VulkanKernelPolicy::from_capabilities',
      'A benchmark result is evidence about that recorded run',
      'it is never a device-name dispatch rule',
      'stable',
      'experimental',
      'maintenance',
      'inference takes shared ownership',
      'training and physical mutations take exclusive ownership',
      'vulkan_correctness_quarantine',
      "Kiln's HTTP API has no built-in authentication",
      'unknown accelerator state',
    ],
  },
  {
    label: 'Native SFT Profile',
    path: publishedPath('docs/native-sft-profile/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/native-sft-profile/',
    h1: 'Native SFT Profile',
    anchors: ['backend-owned-sft-loss-routing', 'tape-authority-and-inference-isolation', 'frozen-parameter-ownership', 'exact-gradient-set-boundary'],
    terms: [
      'config.optimizer is a tagged object',
      '"kind":"adam_w"',
      '"kind":"muon"',
      'rank 2..=48',
      'rank 2..=32',
      'Metal does not implement a native SGD update',
      'F16 remains inference-only',
      'kiln.training-optimizer-support',
      'optimizer_tuple_kinds',
      'optimizer_tuple {supported, unavailable_reason, lora_rank}',
      'The workloads array contains exactly',
      'distill_refresh',
      'distinct sequential workload, not an OPD alias',
      'separate exact SFT and OPD phase plans',
      'prepares the exact SFT rows',
      'reserves the maximum sequential working set',
      'larger phase peak',
      'allowed_optimizer_kinds',
      'CPU server workloads remain unsupported',
      'no Marlin-packed projection',
      'kt_tape_authoritative',
      'phase-B backward routes',
      'GET /v1/recipes',
      'admission {supported, unavailable_reason}',
      'live_memory_admission_required=true',
      'backend_maximum',
      'model_maximum',
      'effective maximum',
      'optimizer backend is unbounded',
      'maximum remains the model ceiling',
      'CPU reference Muon requires rank 2+',
      'never lowers rank',
      'fixed to round-to-nearest',
      'legacy exact checkpoint that records stochastic rounding cannot resume',
      'Resume first passes the current cheap SFT workload and resident optimizer-tuple gates',
      'newly Marlin-packed projection',
      'Backend-owned SFT loss routing',
      'SftConfig',
      'kt_tape_flce',
      'vulkan_active_rows',
      'full_logits',
      'multi-segment checkpoint plan is rejected',
      'training_invalid_request',
      'not a hardware-qualification receipt',
      'no TOML field',
      'no mechanically derived environment name',
      'KILN_USE_FLCE',
      'not a compatibility alias',
      'typed loader',
      'HTTP 413',
      'loss workspace ... (route=<route>)',
      'PreparedSftAdmission',
      'before governor reservation, KV replacement, or allocator reclamation',
      'TrainingRuntimeContext',
      'freshly constructed execution backend',
      'runtime.sft_loss_route',
      'kiln.training-checkpoint-planning.v4',
      'older v3 planning identity',
      'GRPO and OPD continue to use the common v3 planning identity',
      'Tape authority and inference isolation',
      'part of the native workload contract, not a tunable feature',
      'presence of that scope as their only activation authority',
      'GDN chunkwise recurrence',
      "CUDA's weight-aware embedding lookup",
      'KILN_USE_TAPE_FORWARD',
      'KILN_USE_TAPE_FLASH_ATTN',
      'KILN_USE_TAPE_SDPA',
      'KILN_USE_TAPE_LORA_ADD',
      'KILN_USE_TAPE_GDN',
      'KILN_USE_TAPE_GDN_CONV',
      'KILN_USE_TAPE_GDN_QK_NORM',
      'KILN_USE_TAPE_GDN_GATED_NORM',
      'removed without aliases or replacements',
      'not accelerator qualification',
      'Frozen parameter ownership',
      'trains LoRA A/B tensors and no base-model parameters',
      'Base projection matrices are saved constants',
      'Embedding tables and token IDs are not differentiable inputs',
      'RMSNorm and GDN gated-RMSNorm weights are saved constants',
      'Frozen GDN gate parameters remain constants',
      'MTP projection weights',
      'loss roots save their head transpose as a constant',
      'frozen dWeight GEMMs and frozen gradient storage',
      'both chunks register the original full A/B IDs',
      'zero-pads its B gradient to the full parameter shape',
      'tape-aware reshapes keep the post-split activation graph connected',
      'Exact gradient-set boundary',
      'observed gradient tensor IDs',
      'must contain only finite values',
      'finite all-zero gradient remains valid',
      'One connected clone cannot mask a disconnected sibling',
      'range with no configured leaves accepts only an empty result',
      'one finite-value reduction immediately before the optimizer',
      'correctness guarantee, not large-batch performance evidence',
    ],
  },
  {
    label: 'Native Training Checkpoints',
    path: publishedPath('docs/training-checkpoints/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/training-checkpoints/',
    h1: 'Native Training Checkpoints',
    anchors: ['checkpoint-planning-identity'],
    terms: [
      'Checkpoint or adapter?',
      'A resume continues the same optimizer run',
      'Start a checkpointed run',
      'Find the checkpoint to resume',
      'latest_checkpoint',
      'Resume the run',
      'Resume is continuation, not warm-starting',
      'capability-derived backend routes',
      'not device-name',
      'Queue-time revalidation',
      'before memory reservation',
      'GRPO and OPD use schema kiln.training-checkpoint-planning.v3',
      'SFT uses schema kiln.training-checkpoint-planning.v4',
      'sft_loss_route',
      'kt_tape_flce',
      'vulkan_active_rows',
      'full_logits',
      'Crash and cancellation behavior',
      'What the checkpoint restores',
      'Checkpoint validation and security',
      'Promotion is a separate decision',
      'Recovery checklist',
      'does not prune checkpoints on a timer',
    ],
  },
  {
    label: 'Dataset Splits and Train/Eval Separation',
    path: publishedPath('docs/dataset-splits/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/dataset-splits/',
    h1: 'Dataset Splits and Train/Eval Separation',
    anchors: [
      'recommended-workflow',
      'split-configuration',
      'row-and-group-identity',
      'post-training-evaluation-policy',
      'training-data-provenance',
      'what-this-does-and-does-not-prove',
    ],
    terms: [
      'dataset_split',
      'source_split',
      'split_manifest_sha256',
      'train-set-eval',
      'arbitrary paraphrases',
      'group_id',
      'session_id',
      'before queue publication',
    ],
  },
  {
    label: 'Local Hardware Qualification',
    path: publishedPath('docs/hardware-qualification/index.html'),
    canonical: 'https://ericflo.github.io/kiln/docs/hardware-qualification/',
    h1: 'Local Hardware Qualification',
    terms: [
      'did this declared workload pass',
      'Portable runtime policy must continue to discover and use the capabilities',
      'does not require a named laptop, GPU model, or temperature sensor',
      'a conformant Vulkan loader and driver',
      'invoke case commands through a shell',
      'command case-result file is not the qualification verdict',
      'no fixed CPU quota or machine-sized memory ceiling',
      'does not pin one ROCm architecture',
      'The receipt itself has no signature or self-digest',
      '--require-local-artifacts',
      'comparison policy',
      'Retained Evidence Examples',
      'historical snapshots, not current defaults or a device allowlist',
      'Metal/macOS environment',
      'Core correctness',
      'Controlled admission, allocation-failure, reclaim, and cleanup cases',
      'Targeted serving closure',
      'Hosted build checkpoint',
      'Version 1 oracle results',
      'Do not “fix” a missing capability',
      'Keep every machine-specific fact out of portable product defaults',
    ],
  },
];

for (const expected of generatedDocsPages) {
  if (expected.label === 'Documentation hub') continue;
  const slug = expected.path.split('/').filter(Boolean).at(-2);
  const document = docsManifest.documents.find((candidate) => candidate.slug === slug);
  if (document) expected.h1 = document.title;
}

const configurationGuideExpectation = generatedDocsPages.find(
  (page) => page.label === 'Configuration Reference',
);
configurationGuideExpectation.anchors = [
  'minimal-server',
  'the-settings-people-change-first',
  'configuration-precedence',
  'accelerator-policy',
  'exact-references',
];
configurationGuideExpectation.terms = [
  'Most Kiln deployments need only',
  'Run kiln config before starting',
  'effective value and its source',
  'Portable does not mean',
  'Device marketing names',
  'Complete typed configuration reference',
  'Machine-readable configuration schema',
];

const expectedQuickstartSections = [
  { label: 'Kiln Desktop path', terms: ['kiln desktop', 'recommended'] },
  { label: 'server binary path', terms: ['server binary', 'terminal-first'] },
  { label: 'Docker path', terms: ['docker', 'nvidia container toolkit'] },
  { label: 'requirements', terms: ['before you download', '20 gb', 'accelerator runtime', 'cpu architecture'] },
  { label: 'start server', terms: ['run the server', 'kiln serve'] },
  { label: 'test inference', terms: ['send chat', '/v1/chat/completions'] },
  { label: 'pi agent setup', terms: ['configure pi', 'kiln pi-setup', 'Qwen3.5-4B'] },
  { label: 'open UI', terms: ['open the ui', '/ui'] },
  { label: 'first inference checkpoint', terms: ['get one response before moving on to training'] },
  { label: 'SFT next step', terms: ['correct with sft', 'kiln train sft'] },
  { label: 'GRPO next step', terms: ['optimize with grpo', 'generate candidates', 'score them', 'train'] },
  { label: 'training payload shapes', terms: ['sft jsonl', 'one chat correction', 'messages array', 'grpo json request or batch', 'jsonl with one candidate-and-reward group per line', 'opd request object', 'kiln train sft', 'kiln train grpo', 'kiln train opd', 'registered exact teacher identity'] },
  { label: 'exact training resume', terms: ['checkpoint-interval', '.kiln-checkpoint', 'resume-checkpoint', 'same source', 'restores optimizer and loop state', 'opd defaults to 25', 'exact registered teacher revision'] },
  { label: 'base-weight provenance', terms: ['base-weight shard manifest', 'changed shard bytes', 'legacy aggregate-only checkpoints', 'base-weight provenance contract'] },
  { label: 'next-task guides', terms: ['open the guide for your next task'] },
];

const expectedQuickstartDashboardTerms = [
  'dashboard',
  'status',
  'adapters',
  'training',
  'quick inference',
];

const expectedQuickstartLinks = [
  { label: 'GRPO Guide', href: 'grpo.html' },
  { label: 'API Reference', href: 'api.html' },
  { label: 'CLI Reference', href: 'cli.html' },
  { label: 'Demo', href: 'demo/' },
  { label: 'Troubleshooting', href: 'troubleshooting.html' },
  { label: 'Architecture', href: 'architecture.html' },
];

const expectedDemoSections = [
  { label: 'freshness', terms: ['captured july 30, 2026', 'ui source 1eacf5100'] },
  { label: 'provenance', terms: ['ui: embedded dashboard source at', '1eacf5100', 'data: deterministic seeded demo fixtures', 'not a live performance run'] },
  { label: 'observe', terms: ['know what is actually running', 'backend truth', 'request evidence'] },
  { label: 'serve', terms: ['test the exact request path', 'openai-compatible', 'adapter-explicit'] },
  { label: 'teach', terms: ['turn evidence into a named training job', 'admission first', 'evaluation separate'] },
  { label: 'promote', terms: ['compare, inspect, then activate', 'base identity bound', 'hot-swap explicit'] },
  { label: 'profile boundary', terms: ['works in the default stable profile', 'interactive train, evaluate, and adapter-activation loop', 'requires the experimental profile', 'current saved-adapter limitation'] },
  { label: 'historical archive', terms: ['historical terminal recordings', 'captured may 4, 2026', 'not current product proof'] },
];

const expectedDemoCastFiles = [
  'first-token.cast',
  'bench.cast',
  'hot-swap.cast',
  'openai.cast',
  'grpo.cast',
  'kiln-60s.cast',
];

const expectedDemoReadmeDrivers = new Map([
  ['first-token.cast', 'demo-first-token.sh'],
  ['bench.cast', 'demo-bench.sh'],
  ['hot-swap.cast', 'demo-hot-swap.sh'],
  ['openai.cast', 'demo-openai.sh'],
  ['grpo.cast', 'demo-grpo.sh'],
  ['kiln-60s.cast', 'demo.sh'],
]);

const expectedDemoReadmeLinks = [
  'SCRIPTS.md',
  'SCRIPT.md',
  'index.html',
  'QUICKSTART.md',
  'README.md',
  '../launch/README.md',
];

const expectedApiEndpoints = [
  '/health',
  '/v1/health',
  '/metrics',
  '/ui',
  '/v1/models',
  '/v1/config',
  '/v1/debug/model-state',
  '/v1/chat/completions',
  '/v1/completions/batch',
  '/v1/train/hf/sft/exports',
  '/v1/train/hf/exports',
  '/v1/train/hf/exports/{name}',
  '/v1/train/hf/exports/{name}/download',
  '/v1/train/hf/peft/imports/{name}',
  '/v1/adapters',
  '/v1/adapters/default/download',
  '/v1/adapters/upload',
  '/v1/adapters/merge',
  '/v1/train/sft',
  '/v1/train/grpo',
  '/v1/train/status',
  '/v1/train/status/{job_id}',
  '/v1/train/queue',
];

// /v1/train/jobs/{job_id} is a REAL route (GET job detail + DELETE archived
// job — see crates/kiln-server/src/api/training.rs routes); the api page
// must document it rather than deny it.
const trainingJobDetailEndpoint = '/v1/train/jobs/{job_id}';
const staleAdapterListPhrases = [
  'List loaded adapters',
  'List loaded and available adapters',
  'List loaded LoRA adapters',
  'list loaded LoRA adapters',
  'loaded LoRA adapters',
];
const adapterListStaleWordingSurfaces = [
  'README.md',
  'QUICKSTART.md',
  'docs/site/api.html',
  'crates/kiln-server/src/api/adapters.rs',
];
const expectedAdapterListSemantics = [
  'saved/available LoRA adapters',
  'active adapter',
  'content revision',
];

const expectedApiSections = [
  { label: 'server status', terms: ['server status'] },
  { label: 'Metal platform boundary evidence', terms: ['Retained Metal platform receipt', '20260728t223911446266z-metal-macbook-air-m1-local-environment-v1', 'clean pushed source c8d9f5856ce9', '17,179,869,184 unified-memory bytes', 'real shader compiled to AIR and linked to a metallib', 'permission denial for external networking', 'temperature readings were explicitly unavailable', 'platform-boundary evidence only'] },
  { label: 'Metal core evidence', terms: ['Retained Metal core receipt', '20260728t225405419496z-metal-macbook-air-m1-cuda-metal-core-correctn-d119e83143-v1', 'clean pushed source 94d03199f334', 'strided-view', 'left-transposed', 'two-row Metal ICB replay', 'twenty-step BF16 AdamW trajectory', 'bounded core-correctness evidence'] },
  { label: 'Metal unified-memory lifecycle evidence', terms: ['Retained Metal unified-memory lifecycle receipt', '20260728t230542216939z-metal-macbook-air-m1-metal-memory-lifecycle-v-dfc8d17c13-v1', 'clean pushed source ee39b573566d', 'controlled floor left a 64 MiB budget', '10,241 blocks before', 'injected first allocator failure', 'dropping its sole owner reclaimed all', 'controlled UMA lifecycle evidence'] },
  { label: 'complete effective configuration', terms: ['kiln.effective-configuration.v1', 'all 112 fixed typed startup leaves', 'two dynamic leaves for each teacher credential', 'post-precedence typed value', 'command_line', 'canonical environment spelling', 'explicitly empty compatibility-name list', 'redacted null entries', 'kiln config --json'] },
  { label: 'typed CUDA graph diagnostics', terms: ['top-level cuda_graphs configured/effective/cache/invariant state', 'decode_runtime.cuda_graphs', 'requested and profile-effective single-row CUDA graph policy', 'mandatory stable metadata', 'unavailable batched route', 'trusted debug response repeats that exact object', 'qualification receipt can bind the launch policy'] },
  { label: 'immutable operational runtime', terms: ['immutable operational snapshot', 'terminal/agent access', 'pi resolution', 'library URL', 'cache/session paths'] },
  { label: 'typed accelerator retained-byte diagnostics', terms: ['kiln.accelerator-runtime-policy.v16', '"version": 16', 'All fifteen typed [accelerator] fields', 'kt_api_mode', 'full_attention_score_budget_mib', 'vulkan_device_policy_schema_id', 'kiln.vulkan-device-policy.v1', 'cuda_kernel_profile', 'metal_kernel_profile', 'rocm_synchronization_mode', 'rocm_strided_batched_matmul_mode', 'disabled is the portable default', 'rocm_bf16_matmul_output_mode', 'f32_then_cast is the portable default', 'rocm_kernel_profile', 'portable_fallback is the only product value', 'Retired machine-qualified values fail startup', 'hardware-qualification', 'rocm_graph_mode', 'rocm_graph_cache_entries', 'rocm_graph_cache_max_bytes', '1073741824', '67108864..=17179869184', 'KILN_ACCELERATOR_ROCM_GRAPH_CACHE_MAX_BYTES', 'retained_bytes', 'opaque_native_object_count', 'quarantined_retained_bytes', 'decode_runtime.rocm_graphs', 'rocm_graphs_unavailable_reason', 'rocm_graph_telemetry', 'rocm_graph_telemetry_unavailable_reason'] },
  { label: 'ROCm live API and Prometheus contract', terms: ['independent top-level rocm_graphs full statistics', 'rocm_graphs_unavailable_reason', 'rocm_graph_telemetry', 'rocm_graph_telemetry_unavailable_reason', 'busy only for model_runner_busy or graph_runner_busy', 'unavailable for backend_without_graph_runner', 'model_runner_lock_poisoned', 'graph_runner_lock_poisoned', 'Owner and slot lifecycle', 'decode_owner_release_count', 'tracked_decode_owner_count', 'ROCm graph Prometheus contract', 'kiln_rocm_graph_telemetry_available', 'kiln_rocm_graph_snapshot_unavailable{reason}', 'kiln_rocm_graph_phase_telemetry_available', 'kiln_rocm_graph_phase_telemetry_unavailable{reason}', 'kiln_rocm_graph_state{kind}', 'kiln_rocm_graph_slots{state=', 'kiln_rocm_graph_owner_lifecycle_total{event=', 'kiln_rocm_graph_retained_bytes{kind}', 'kiln_rocm_graph_cache_admissions_total', 'kiln_rocm_graph_cache_evictions_by_cause_total', 'kiln_rocm_graph_cache_admission_rejections_total', 'kiln_rocm_graph_pre_capture_skips_total', 'kiln_rocm_graph_capture_outcomes_total', 'kiln_rocm_graph_replay_outcomes_total', 'kiln_rocm_graph_current_phase{phase}', 'kiln_rocm_graph_phase_calls_total{phase}', 'kiln_rocm_graph_transient_candidate_bytes{kind="last|peak"}', 'kiln_rocm_graph_fallbacks_total{reason}', 'kiln_rocm_graph_fallback_slow_total', 'kiln_rocm_graph_fallback_duration_seconds_total', 'kiln_rocm_graph_fallback_duration_seconds_max', 'cold_cache_host_round_trip', 'persistent_host_round_trip', 'shape_dependent_attention', 'graph_cache_capacity', 'graph_cache_byte_budget', 'graph_accounting_incomplete', 'moderate_memory_pressure', 'tight_memory_pressure', 'critical_memory_pressure', 'memory_reservation_denied', 'memory_governor_selector_mismatch', 'capture_failure', 'replay_failure', 'Counter labels are closed sets'] },
  { label: 'ROCm multi-row graph-route accounting', terms: ['all 14 reason counters', 'multi_row_batch_unsupported', 'Current ROCm capture includes contiguous BF16 multi-row decode', 'real capture/replay activity'] },
  { label: 'ROCm graph outcome, phase, and rollback distinctions', terms: ['recurrent/conv state in every runner-owned owner slot', 'active slot state that outlives graph eviction', 'peak_retained_bytes', 'never reported below the current total', 'capture_successes means native graph instantiation and its first launch succeeded', 'cache_admission_successes is the retained, replayable subset', 'point-in-time phase/transient copy', 'six phase summary objects', 'pre_candidate_headroom_phase', 'candidate_warm_phase', 'pre_native_reservation_phase', 'native_capture_phase', 'native_replay_phase', 'rejected_candidate_cleanup_phase', 'last_transient_candidate_bytes', 'peak_transient_candidate_bytes', 'capture_rollback', 'logical recurrent-state restoration', 'sticky STOP', 'post-STOP diagnostic drain', 'fixed 23-reason array'] },
  { label: 'typed batching diagnostics', terms: ['batching', 'actor_active', 'configured_source', 'prefill_admission_quantum', 'effective_decode_width', 'burst_prefill_admission', 'actor_prefill_tile_alignment_required', 'one scheduling state machine', 'decode_runtime.batching_configuration', 'batching_engine.configuration', 'Every real backend constructs this actor', 'Serving benchmark driver v18'] },
  { label: 'typed streaming-prefill diagnostics', terms: ['streaming_prefill', 'configured_mode', 'prompt_tokens_at_least', 'effective_for_auto_mode', 'override_applied_to_backend_auto_policy', 'tape_tile_tokens', 'detached_full_attn_boundary_tile_tokens', 'detached_full_attn_tape_replay_tile_tokens', 'last_token_lm_head', 'inherited_from_tile_tokens_config_file', 'prefill_runtime.streaming_prefill', 'restart_required_to_change', '(1024,1024,8192,65536,65536)', '(256,256,8192,8192,8192)', '(2048,2048,8192,8192,8192)'] },
  { label: 'typed SFT checkpoint-boundary diagnostics', terms: ['checkpoint_boundary_policy', 'recompute_mode', 'recompute_threshold_tokens', 'anchor_stride', 'cache_target_bytes', '/v1/config', '/health', '/v1/health', '/v1/debug/model-state', 'Runtime Config training group', 'immutability', 'restart requirement'] },
  { label: 'optimizer capability diagnostics', terms: ['optimizer_support', 'kiln.training-optimizer-support', 'backend_implementation', 'portable_reference', 'native_device_hook', 'optimizer_tuple_kinds', 'optimizer_tuple', 'workloads', 'workload', 'supported', 'unavailable_reason', 'allowed_optimizer_kinds', 'base_weight_dtype', 'resolved_lora_parameter_dtype', 'cannot be resolved by the backend precision policy', 'round_to_nearest', 'backend_implementation_rounding_modes', 'live_memory_admission_required', 'backend_maximum', 'model_maximum', 'effective minimum as maximum', 'null backend_maximum means the backend adds no ceiling', 'maximum remains the concrete model ceiling', 'does not lower the requested rank', 'CPU can expose', 'portable F32 tuples', 'all four server workloads remain unsupported', 'Hybrid Vulkan can expose raw native hooks or tuples', 'exact native backend/device identity', 'no Marlin-packed projection', 'kt_tape_authoritative', 'phase-B backward routes', 'distill_refresh', 'separate exact SFT and OPD phase plans', 'prepare the precise SFT rows', 'reserve the larger of the two sequential working sets', 'Cheap teacher-alias validation and metadata pinning may occur first', 'checkpoint loading, remote/local teacher materialization, corpus scanning, memory preflight, and GPU reservation occur only after the static workload check', 'Metal', 'unsupported', 'rank 2+', 'rank 2..=48', 'rank 2..=32', 'F16 is inference-only', 'KILN_BF16_STOCHASTIC_ROUND', 'KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK', 'KILN_CUDA_TRAINING_OPTIMIZER_FALLBACK', 'KILN_ROCM_TRAINING_OPTIMIZER_FALLBACK', 'KILN_METAL_TRAINING_OPTIMIZER_FALLBACK', 'KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK'] },
  { label: 'training tape scope authority', terms: ['tape scope authority and inference isolation', 'required workload substrate, not a configuration feature', 'internal thread-local training scope', 'sole activation authority', 'no TOML field, API property, CLI flag, or mechanically derived environment name', 'ordinary inference never opens that scope', 'GDN chunkwise recurrence', 'weight-aware embedding lookup', 'neither consults a cached global tape default', 'cannot suppress a recorder, sever the gradient graph', 'KILN_USE_TAPE_FORWARD', 'KILN_USE_TAPE_FLASH_ATTN', 'KILN_USE_TAPE_SDPA', 'KILN_USE_TAPE_LORA_ADD', 'KILN_USE_TAPE_GDN', 'KILN_USE_TAPE_GDN_CONV', 'KILN_USE_TAPE_GDN_QK_NORM', 'KILN_USE_TAPE_GDN_GATED_NORM', 'removed without aliases or replacement fields', 'not hardware qualification'] },
  { label: 'frozen training parameter ownership', terms: ['Frozen parameter ownership', 'train LoRA A/B tensors and no base-model parameters', 'Embedding tables', 'base projection matrices', 'RMSNorm and gated-RMSNorm weights', 'GDN gate parameters', 'MTP projection weights', 'loss-head transposes', 'saved constants rather than differentiable tape inputs', 'do not allocate, retain, deposit, or run matrix products for frozen-weight gradients', 'omit dWeight outputs and atomics', 'original full LoRA A/B IDs', 'pads its B contribution to the full parameter shape', 'tape-aware reshapes', 'temporary B slice can never become an optimizer leaf', 'not accelerator throughput evidence'] },
  { label: 'exact LoRA gradient-set boundary', terms: ['Exact LoRA gradient-set boundary', 'observed gradient tensor IDs must equal the configured trainable LoRA leaves', 'shape, backward-compute dtype, and master device', 'only finite values', 'Missing or unknown IDs', 'all-zero gradient remains valid', 'no request, config, debug, or environment bypass', 'CUDA, ROCm, and Vulkan use backend finite reducers', 'Metal currently synchronizes and copies each complete gradient to the host', 'correctness fallback, not a Metal large-batch throughput claim', 'cannot be disabled at the optimizer boundary', 'exactly the leaves in their layer range', 'one accumulated entry per leaf', 'Every registered differentiable input must have a gradient', 'one connected clone cannot hide a disconnected use', 'zero-padded contributions under the original full LoRA leaf ID', 'range with no configured LoRA leaves accepts only an empty gradient set', 'duplicate leaf IDs across disjoint segments', 'one finite-value scan', 'SFT, GRPO, and OPD share this consumer contract'] },
  { label: 'optimizer request, recipe, and resume contract', terms: ['config.optimizer', '{"kind":"sgd"}', '{"kind":"adam_w"', '{"kind":"muon"', 'Muon iterations must be in', 'cheap per-workload gate and resident optimizer tuple are checked before checkpoint or corpus materialization', 'before memory reservation', 'before device residency', '/v1/recipes', 'admission {supported, unavailable_reason}', 'preflights every step again', 'training_invalid_request', 'training_backend_unsupported', 'legacy checkpoint recording stochastic rounding fails closed', 'newly Marlin-packed weight', 'unavailable authoritative route', 'Queued resume admission fully validates', 'checkpoint ID', 'digest of that validated manifest', 'effective seed', 'Before memory reservation at dequeue', 'revalidation, not a filesystem snapshot', 'mutation race after the reload'] },
  { label: 'DistillRefresh fail-closed admission', terms: ['/v1/distill/refresh', 'distinct fail-closed DistillRefresh workload', 'no job or seed is created', 'separate exact SFT and OPD phase plans', 'exact SFT rows', 'maximum sequential working set', 'currently rejects before a seed or job exists', 'Any descriptor containing a DistillRefresh step is currently unsupported'] },
  { label: 'SFT checkpoint-boundary startup and resume contract', terms: ['recompute_checkpoint_boundaries', 'checkpoint_boundary_anchor_stride', 'checkpoint_boundary_cache_gb', 'defaults to auto', 'inclusive sequence threshold defaults to 8,192 tokens', 'default 6 GiB', 'explicit positive strides', 'four historical unsectioned names are deprecated', 'parsed strictly', 'must agree with a present canonical value', 'sft_loss_route', 'kiln.training-checkpoint-planning.v4', 'prior SFT v3 checkpoints fail closed', 'GRPO and OPD', 'kiln.training-checkpoint-planning.v3', 'planning drift', 'cannot resume exactly', 'outer checkpoint envelope remains schema v1'] },
  { label: 'copy-paste first requests', terms: ['copy-paste first requests'] },
  { label: 'power-user requests', terms: ['power-user requests'] },
  { label: 'OpenAI-compatible generation', terms: ['openai-compatible generation'] },
  { label: 'first-token performance attribution', terms: ['attribute first-token latency', 'include_performance', 'actor_queue_ms', 'actor_admission_ms', 'actor_prefill_wall_ms', 'kiln.token_timing'] },
  { label: 'adapter lifecycle', terms: ['lora lifecycle'] },
  { label: 'training', terms: ['training'] },
  { label: 'exact SFT/GRPO/OPD resume', terms: ['latest exact sft/grpo/opd resume checkpoint', '/v1/train/opd', 'checkpoint_interval', 'resume_checkpoint', 'training_kind', 'data_source_kind', 'candidate cursor', 'teacher content revision', 'peft snapshots remain serving-only'] },
  { label: 'base-weight provenance', terms: ['kiln.base-weight-shards.v1', 'base-weight aggregate', 'complete base-weight shard manifest', 'legacy aggregate-only checkpoints'] },
  { label: 'execution provenance', terms: ['kiln.execution-provenance.v1', 'execution-provenance record', 'admission-time base-weight and execution identities', 'validated execution-provenance record'] },
  { label: 'HF/TRL conditional export cleanup', terms: ['/v1/train/hf/exports/{name}', 'etag', 'if-match', 'http 412', 'damaged bundles'] },
  { label: 'HF/TRL resident-validated PEFT import', terms: ['/v1/train/hf/peft/imports/{name}', '{name}.kiln-hf-import', 'resident model/tokenizer/template identity', 'peft tensor shapes', 'import-receipt digest', 'content revision', 'kiln train hf import-peft', 'strong etag', 'installed byte count', 'source bundle on every outcome'] },
  { label: 'training data safety', terms: ['training data changes', 'adapter'] },
  {
    label: 'typed CUDA weight and backward policy',
    terms: [
      'cuda_marlin_profile',
      'KILN_ACCELERATOR_CUDA_MARLIN_PROFILE',
      'attention_mlp',
      'attention_mlp_gdn',
      'server.serving_profile = "experimental"',
      'cuda_flash_backward_mode',
      'KILN_ACCELERATOR_CUDA_FLASH_BACKWARD_MODE',
      'deterministic',
      'KILN_W4A16',
      'KILN_W4A16_GDN_OUT_PROJ',
      'KILN_DISABLE_PARALLEL_PACK',
      'KILN_FLASH_ATTN_BWD_DETERMINISTIC',
      'KILN_DISABLE_OPD_LOSS_KERNEL',
      'KILN_FLCE_ACTIVE_ROW_TILE',
      '4,096',
    ],
  },
  { label: 'response shapes', terms: ['response shapes'] },
];

const expectedApiCodeExamples = [
  { label: 'accelerator runtime v16 shape', terms: ['"accelerator_runtime"', '"schema_id": "kiln.accelerator-runtime-policy.v16"', '"version": 16', '"vulkan_kernel_policy_schema_id": "kiln.vulkan-kernel-policy.v6"', '"vulkan_device_policy_schema_id": "kiln.vulkan-device-policy.v1"', '"kt_api_mode"', '"full_attention_score_budget_mib"', '"configured": 2048', '"effective": 2048', '"vulkan_device_index"', '"configured": null', '"effective": null', '"vulkan_validation"', '"configured": false', '"effective": false', '"cuda_kernel_profile"', '"metal_kernel_profile"', '"configured": "native_default"', '"effective": "native_default"', '"rocm_synchronization_mode"', '"rocm_strided_batched_matmul_mode"', '"rocm_bf16_matmul_output_mode"', '"rocm_kernel_profile"', '"configured": "portable_fallback"', '"effective": "portable_fallback"', '"rocm_graph_mode"', '"rocm_graph_cache_entries"', '"rocm_graph_cache_max_bytes"', '"configured": 1073741824', '"effective": 1073741824'] },
  { label: 'CUDA startup policy shape', terms: ['"cuda_marlin_profile"', '"configured": "disabled"', '"effective": "disabled"', '"cuda_flash_backward_mode"', '"configured": "fast"', '"effective": "fast"'] },
  { label: 'batching configuration shape', terms: ['"batching"', '"configuration"', '"rowwise_decode"', '"prefix_aware_admission"', '"prefill_admission_quantum"', '"actor_cycle_idle"', '"actor_active"'] },
  { label: 'streaming-prefill configuration shape', terms: ['"streaming_prefill"', '"dispatch"', '"threshold_tokens"', '"tile_tokens"', '"tape_tile_tokens"', '"detached_full_attn_tile_tokens"', '"immutable_after_startup"', '"restart_required_to_change"'] },
  { label: 'checkpoint-boundary configuration shape', terms: ['"checkpoint_boundary_policy"', '"recompute_mode"', '"recompute_threshold_tokens"', '"anchor_stride"', '"cache_target_bytes"', '8192', '6442450944'] },
  { label: 'optimizer support configuration shape', terms: ['"optimizer_support"', '"kiln.training-optimizer-support"', '"version": 1', '"base_weight_dtype"', '"resolved_lora_parameter_dtype"', '"immutable_after_startup"', '"rounding_modes"', '"backend_implementation_rounding_modes"', '"optimizer_tuple_kinds"', '"workloads"', '"workload": "sft"', '"workload": "grpo"', '"workload": "opd"', '"workload": "distill_refresh"', '"allowed_optimizer_kinds": []', '"backend_implementation"', '"native_device_hook"', '"optimizer_tuple"', '"unavailable_reason"', '"minimum"', '"maximum"', '"backend_maximum"', '"model_maximum"', '"live_memory_admission_required"', '"maximum": 48', '"maximum": 1024', '"backend_maximum": null', '"model_maximum": 1024'] },
  { label: 'recipe admission descriptor shape', terms: ['"recipes"', '"name": "frontier-pump"', '"num_steps": 1', '"admission"', '"supported": false', '"unavailable_reason": "step 1 (opd) is unavailable'] },
  { label: 'first chat', terms: ['/v1/chat/completions', 'messages', 'max_tokens'] },
  { label: 'first SFT', terms: ['/v1/train/sft', 'examples', 'config', 'epochs'] },
  { label: 'first GRPO', terms: ['/v1/train/grpo', 'groups', 'completions', 'reward'] },
  { label: 'exact GRPO resume', terms: ['/v1/train/grpo', 'scored-groups.jsonl', 'checkpoint_interval', 'resume_checkpoint'] },
  { label: 'exact OPD resume', terms: ['/v1/train/opd', 'qwen35@vllm', 'distilled-bot', 'checkpoint_interval', 'resume_checkpoint'] },
  { label: 'training status', terms: ['/v1/train/status'] },
  { label: 'batch completions', terms: ['/v1/completions/batch', 'prompts'] },
  { label: 'adapter download/upload', terms: ['/v1/adapters/default/download', '/v1/adapters/upload'] },
  { label: 'merge', terms: ['/v1/adapters/merge', 'mode', 'ties'] },
  { label: 'composition', terms: ['/v1/chat/completions', 'adapters', 'scale'] },
  { label: 'webhook', terms: ['kiln.toml', 'webhook_url', 'kiln_training_webhook_url'] },
];

// The public API page is a task-oriented map, not a duplicate of every generated
// runtime contract. Keep this smoke coverage focused on its visible reading path;
// the generated references and schema tests own exhaustive field-level coverage.
const expectedApiReaderSections = [
  { label: 'server status', terms: ['server status, metrics, ui, and config'] },
  { label: 'first requests', terms: ['copy-paste first requests'] },
  { label: 'advanced requests', terms: ['copyable advanced requests'] },
  { label: 'generation', terms: ['openai-compatible generation'] },
  { label: 'adapter lifecycle', terms: ['lora lifecycle', 'stable profile cannot serve or evaluate one'] },
  { label: 'training', terms: ['sft, grpo, opd, status, and queue control', 'current hybrid vulkan server rejects it before dataset admission'] },
  { label: 'teachers', terms: ['verified remote teachers and identity-bound cache'] },
  { label: 'evals', terms: ['run eval suites, synthesize datasets, train a local judge'] },
  { label: 'training safety', terms: ['treat every training endpoint as privileged'] },
  { label: 'response shapes', terms: ['response shapes'] },
  { label: 'structured errors', terms: ['branch on http status and error.code', 'do not infer retry safety from prose alone'] },
];

const expectedApiReaderCodeExamples = [
  { label: 'first chat', terms: ['/v1/chat/completions', 'messages', 'max_tokens'] },
  { label: 'first SFT', terms: ['/v1/train/sft', 'examples', 'config', 'epochs'] },
  { label: 'first GRPO', terms: ['/v1/train/grpo', 'groups', 'completions', 'reward'] },
  { label: 'training status', terms: ['/v1/train/status'] },
  { label: 'batch completions', terms: ['/v1/completions/batch', 'prompts'] },
  { label: 'adapter download/upload', terms: ['/v1/adapters/default/download', '/v1/adapters/upload'] },
  { label: 'merge', terms: ['/v1/adapters/merge', 'mode', 'ties'] },
  { label: 'composition', terms: ['/v1/chat/completions', 'adapters', 'scale'] },
  { label: 'webhook', terms: ['kiln.toml', 'webhook_url', 'kiln_training_webhook_url'] },
];

const expectedCliSections = [
  { label: 'command chooser', terms: ['choose a task'] },
  { label: 'serve/start-server path', terms: ['start serving Qwen3.5-4B', 'kiln_model_path', 'kiln serve'] },
  { label: 'no-subcommand serve path', terms: ['running kiln with no subcommand starts the server'] },
  { label: 'health/readiness path', terms: ['check server readiness', 'kiln health'] },
  { label: 'pi setup path', terms: ['pi integration', 'kiln pi-setup', 'Qwen3.5-4B'] },
  { label: 'native and HF/TRL training path', terms: ['run native jobs or move a verified hf/trl handoff', 'kiln train sft', 'kiln train grpo', 'kiln train opd', 'kiln train hf export-sft', 'kiln train hf export-grpo', 'kiln train hf import-peft'] },
  { label: 'SFT payload shape', terms: ['sft reads jsonl', 'one chat correction example per line', 'messages array'] },
  { label: 'GRPO payload shape', terms: ['grpo accepts either one json request/batch', 'streamed jsonl with one group per line', 'groups', 'messages', 'candidate completions', 'text', 'reward scores'] },
  { label: 'OPD payload shape', terms: ['opd reads one', '/v1/train/opd', 'json prompt array', '--teacher', 'exact content revision'] },
  { label: 'verified HF/TRL handoff', terms: ['verified hf/trl sft and grpo handoff', '--file is read by the server process', 'sft requires exactly one --file or --dataset', 'grpo requires canonical compact, final-lf --file input', 'uniform-width groups carry exact recorded provenance', 'both exports refuse redirects and existing outputs', 'duplicate paths', 'trailing gzip data', '{name}.kiln-hf', '--keep-server-copy', 'identity-conditional if-match', '--export-sha256', 'omit that flag only for deliberate operator cleanup', 'completed extracted .kiln-hf directory', 'fails local verification before connecting', 'deterministic ten-file envelope', 'locally predicted receipt', 'never changes or removes the source bundle', 'rejected without replacement'] },
  { label: 'exact training resume', terms: ['checkpoint-interval', 'resume-checkpoint', 'exact resume requires the identical file, route, adapter, and options', 'opd defaults to an exact checkpoint every 25 committed optimizer steps'] },
  { label: 'request-lineage integrity', terms: ['kiln-replay verify', 'recomputes request-lineage hashes only', 'does not load a model', 'prove reproducibility', '.kiln-checkpoint'] },
  { label: 'adapter lifecycle path', terms: ['manage lora adapters', 'kiln adapters list', 'kiln adapters load', 'kiln adapters unload'] },
  { label: 'config validation path', terms: ['validate config', 'kiln config --file', 'kiln config --file kiln.toml --backend rocm', 'without probing hardware or loading model weights'] },
  { label: 'typed CUDA graph policy', terms: ['bound retained decode graphs at startup', 'cuda_graphs = true', 'cuda_graph_cache_entries = 8', 'KILN_MEMORY_CUDA_GRAPH_CACHE_ENTRIES', "jq '.cuda_graphs'", "jq '.decode_runtime.cuda_graphs'", 'stable paged metadata is mandatory', 'batched CUDA graph capture is unavailable'] },
  { label: 'typed batching policy', terms: ['resolve scheduler controls at startup', 'batching', 'rowwise_decode', 'prefix_aware_admission', 'prefill_admission_quantum', 'actor_cycle_idle_ms', '0–60,000 ms safe-boundary duty-cycle delay', 'all four batching fields', 'KILN_BATCHING_ACTOR_CYCLE_IDLE_MS', 'combined actor-cycle budget', 'prompt-token and layer ceilings', 'streaming mode, threshold, and three configured tiles', 'reject an invalid contract without probing hardware or loading model weights', 'actor_active', 'Every real backend uses one batching actor'] },
  { label: 'typed streaming-prefill policy', terms: ['resolve long-prefill and training tiles once', 'streaming_prefill', 'threshold_tokens', 'tape_tile_tokens', 'detached_full_attn_tile_tokens', 'positive multiple of 64', 'disable for an a/b', 'force non-empty prompts', 'tune rocm crossover', 'prefill_runtime.streaming_prefill', 'KILN_STREAMING_PREFILL_LAST_TOKEN_LM_HEAD', 'restart after every change'] },
  { label: 'typed offline benchmark tools', terms: ['configure benchmark runs explicitly', 'kiln-bench', 'vulkan_decode_microbench', 'complete kiln-bench arguments', 'complete vulkan microbenchmark arguments', '--spec-method', '--allow-experimental-speculative', '--only', '--batches', '--timed-iters', '--paged-history', '--disable-linear-rows4', '--enable-gdn-in-proj-row-octet', 'every experiment-specific choice is visible in the command', 'not a serving qualification receipt'] },
  { label: 'help and verbosity flags', terms: ['--help', '--verbose', '--quiet', '-vv'] },
  { label: 'UI handoff', terms: ['http://127.0.0.1:8420/ui', '/ui'] },
  { label: 'related docs', terms: ['related docs', 'quickstart', 'api reference', 'grpo guide', 'troubleshooting', 'architecture'] },
];

const expectedCliCodeExamples = [
  { label: 'serve command', terms: ['kiln_model_path=./Qwen3.5-4B', 'kiln serve'] },
  { label: 'health commands', terms: ['kiln health', 'kiln health --json'] },
  { label: 'OpenEnv commands', terms: ['kiln openenv inspect', 'kiln openenv tasks', 'kiln openenv train', '--environment http://127.0.0.1:8990'] },
  { label: 'pi setup command', terms: ['kiln pi-setup', '--kiln-url http://office-kiln:8420'] },
  { label: 'SFT training command', terms: ['kiln train sft', '--file corrections.jsonl', '--adapter support-bot', '--checkpoint-interval 25'] },
  { label: 'GRPO training command', terms: ['kiln train grpo', '--file scored-groups.jsonl', '--adapter support-bot', '--checkpoint-interval 25', '--resume-checkpoint'] },
  { label: 'OPD training command', terms: ['kiln train opd', '--file opd-request.json', '--adapter distilled-bot', '--teacher qwen35@vllm', '--checkpoint-interval 25', '--resume-checkpoint'] },
  { label: 'HF/TRL export and import commands', terms: ['kiln train hf export-sft', '--file /data/corrections.jsonl', '--dataset corrections:active', '--name support-hf-01', 'kiln train hf import-peft', '--bundle ./support-hf-01.kiln-hf', '--name support-v2', 'kiln train hf list --json', 'kiln train hf delete', '--name retained-export', '--export-sha256'] },
  { label: 'training status command', terms: ['kiln train status'] },
  { label: 'adapter commands', terms: ['kiln adapters list', 'kiln adapters load support-bot', 'kiln adapters unload'] },
  { label: 'config validation commands', terms: ['kiln config --file kiln.toml', 'kiln serve --config kiln.toml'] },
  { label: 'CUDA graph policy config', terms: ['[memory]', 'cuda_graphs = true', 'cuda_graph_cache_entries = 8'] },
  { label: 'CUDA graph policy verification', terms: ['kiln config --file kiln.toml', 'kiln serve --config kiln.toml', "jq '.cuda_graphs'", "jq '.decode_runtime.cuda_graphs'"] },
  { label: 'batching policy config', terms: ['[batching]', 'rowwise_decode = false', 'prefix_aware_admission = true', 'prefill_admission_quantum = "auto"', 'actor_cycle_idle_ms = 0'] },
  { label: 'batching policy verification', terms: ['kiln config --file kiln.toml', 'kiln serve --config kiln.toml', "jq '.batching'"] },
  { label: 'streaming-prefill policy config', terms: ['[streaming_prefill]', 'mode = "auto"', 'threshold_tokens = "auto"', 'tile_tokens = "auto"', 'tape_tile_tokens = "auto"', 'detached_full_attn_tile_tokens = "auto"', 'last_token_lm_head = true'] },
  { label: 'streaming-prefill policy verification', terms: ["jq '.streaming_prefill'", "jq '.prefill_runtime.streaming_prefill'"] },
  { label: 'typed benchmark commands', terms: ['kiln-bench --model-path', '--config kiln.toml', '--paged --latency-only', 'vulkan_decode_microbench --only', 'full_step_resident,full_token_resident_paged', '--warmup-iters 10', '--timed-iters 30', '--repeats 5'] },
  { label: 'verbosity commands', terms: ['kiln -v serve', 'kiln -vv serve', 'kiln -q health'] },
];

const expectedCliReaderSections = [
  { label: 'command chooser', terms: ['choose a task'] },
  { label: 'binary availability', terms: ['know which commands the release archive includes'] },
  { label: 'serve', terms: ['start serving qwen3.5-4b'] },
  { label: 'config', terms: ['use config files and model ids'] },
  { label: 'startup policy', terms: ['advanced startup policy'] },
  { label: 'health', terms: ['check server readiness'] },
  { label: 'agent clients', terms: ['point pi or opencode at kiln'] },
  { label: 'OpenEnv', terms: ['act, observe, reward, and train in one command'] },
  { label: 'training', terms: ['run native jobs or move a verified hf/trl handoff'] },
  { label: 'evals', terms: ['run, compare, and strictly replay evals'] },
  { label: 'offline tools', terms: ['configure benchmark runs explicitly'] },
  { label: 'adapters', terms: ['manage lora adapters'] },
  { label: 'related docs', terms: ['related docs'] },
];

const expectedCliReaderCodeExamples = [
  { label: 'serve command', terms: ['kiln_model_path=./qwen3.5-4b', 'kiln serve'] },
  { label: 'health commands', terms: ['kiln health', 'kiln health --json'] },
  { label: 'OpenEnv commands', terms: ['kiln openenv inspect', 'kiln openenv tasks', 'kiln openenv train', '--environment http://127.0.0.1:8990'] },
  { label: 'pi setup command', terms: ['kiln pi-setup', '--kiln-url http://office-kiln:8420'] },
  { label: 'SFT training command', terms: ['kiln train sft', '--file corrections.jsonl', '--adapter support-bot'] },
  { label: 'GRPO training command', terms: ['kiln train grpo', '--file scored-groups.jsonl', '--adapter support-bot'] },
  { label: 'OPD training command', terms: ['kiln train opd', '--file opd-request.json', '--teacher qwen35@vllm'] },
  { label: 'training status command', terms: ['kiln train status'] },
  { label: 'eval commands', terms: ['kiln-eval run', 'kiln-eval compare', 'kiln-eval replay'] },
  { label: 'offline benchmark commands', terms: ['kiln-bench --model-path', 'vulkan_decode_microbench --only'] },
  { label: 'adapter commands', terms: ['kiln adapters list', 'kiln adapters load support-bot', 'kiln adapters unload'] },
  { label: 'config validation commands', terms: ['kiln config --file kiln.toml', 'kiln serve --config kiln.toml'] },
];

const expectedCliLinks = [
  { label: 'Quickstart', href: 'quickstart.html' },
  { label: 'API Reference', href: 'api.html' },
  { label: 'GRPO Guide', href: 'grpo.html' },
  { label: 'Troubleshooting', href: 'troubleshooting.html' },
  { label: 'Architecture', href: 'architecture.html' },
];

const expectedCliHeroFragments = [
  { label: 'Start the server', href: '#serve' },
  { label: 'Check health', href: '#health' },
  { label: 'Configure pi', href: '#pi-setup' },
  { label: 'Train with OpenEnv', href: '#openenv' },
  { label: 'Submit training', href: '#training' },
  { label: 'Run evals', href: '#evals' },
  { label: 'Run offline tools', href: '#benchmarks' },
  { label: 'Manage adapters', href: '#adapters' },
];

const expectedCliPageFragments = [
  '#serve',
  '#health',
  '#openenv',
  '#training',
  '#benchmarks',
  '#adapters',
  '#where-to-go-next',
];

const expectedCliModelSetupCue = {
  label: 'Qwen/Qwen3.5-4B setup cue',
  terms: ['Qwen/Qwen3.5-4B', 'quickstart', 'setup', 'model download'],
  href: 'quickstart.html',
};

const expectedArchitectureReaderSections = [
  { label: 'single process', terms: ['single-process server', 'one deployable rust binary'] },
  { label: 'prefill authority', terms: ['one streaming-prefill authority for inference and training', 'immutable streamingprefillruntimeconfig'] },
  { label: 'accelerator policy', terms: ['accelerator policy is resolved once before execution', 'kiln.accelerator-runtime-policy.v16'] },
  { label: 'Vulkan capability policy', terms: ['device-capability policy derived from standardized limits and features', 'not a device name'] },
  { label: 'request path', terms: ['request path and batching', 'iteration-level scheduler', 'paged kv block manager'] },
  { label: 'Vulkan correctness quarantine', terms: ['vulkan currently uses generic prefill', 'disables cross-request prefix reuse on every vulkan device'] },
  { label: 'model architecture', terms: ['why the gated deltanet hybrid matters', '24 gated deltanet', '8 full gqa layers'] },
  { label: 'training queue', terms: ['lora hot-swap and training queue', 'fifo background queue'] },
  { label: 'SFT route', terms: ['backend-owned sft loss routing', 'does not claim that the complete training path is available'] },
  { label: 'GPU crates', terms: ['gpu backend crates', 'kiln-vulkan-kernel'] },
  { label: 'eval pipeline', terms: ['evals as a peer of training', 'waits while training holds exclusive gpu ownership'] },
  { label: 'next steps', terms: ['where to go next'] },
];

const expectedArchitectureFlowTerms = [
  'http/api',
  'scheduler',
  'block manager',
  'Qwen/Qwen3.5-4B engine',
  'lora training queue',
  'hot-swapped adapter',
];

const expectedArchitectureLinks = [
  { label: 'architecture deep dive', href: 'docs/architecture-deep-dive/' },
  { label: 'quickstart', href: 'quickstart.html' },
  { label: 'troubleshooting', href: 'troubleshooting.html' },
  { label: 'API reference', href: 'api.html' },
  { label: 'GRPO guide', href: 'grpo.html' },
  { label: 'generated Native SFT profile', href: 'docs/native-sft-profile/#backend-owned-sft-loss-routing' },
];

const expectedArchitectureFragments = ['rocm-execution-policy', 'backend-owned-sft-loss-routing'];

const expectedTroubleshootingReaderSections = [
  { label: 'diagnostic framing', terms: ['start with the symptom', 'task-oriented diagnosis guide'] },
  { label: 'three probes', terms: ['start with three probes'] },
  { label: 'Desktop App first launch', terms: ['desktop app first launch'] },
  { label: 'GPU selection', terms: ['wrong binary or gpu path', 'vulkan-capable gpus', 'identity-safety gate'] },
  { label: 'memory pressure', terms: ['memory admission fails or vulkan pauses under pressure'] },
  { label: 'batching diagnosis', terms: ['inference pauses or throughput collapses at concurrency', 'do not label a scheduling pause as vram rebalancing'] },
  { label: 'CUDA graph policy', terms: ['cuda graph capture is absent, unstable, or unexpectedly eager'] },
  { label: 'ROCm latency', terms: ['rocm token pauses or irregular decode latency', 'a pause alone is not evidence', 'rollback'] },
  { label: 'ROCm prefill', terms: ['rocm long prefill pauses or runs out of memory'] },
  { label: 'training tape', terms: ['tape switches do not isolate training or inference'] },
  { label: 'model weights', terms: ['model weights are not found'] },
  { label: 'health', terms: ['/health is not green'] },
  { label: 'remote access', terms: ['remote server is not reachable'] },
  { label: 'historical timeout', terms: ['older-release long-prefill and tool-call timeouts'] },
  { label: 'mock mode', terms: ['mock mode is not real training'] },
  { label: 'adapter directory', terms: ['adapters are in a different directory than expected'] },
  { label: 'adapter conflicts', terms: ['an adapter mutation or training publication was rejected'] },
];

const expectedTroubleshootingProbeExamples = [
  { label: 'health probe', terms: ['/health'] },
  { label: 'models probe', terms: ['/v1/models'] },
  { label: 'minimal chat probe', terms: ['/v1/chat/completions', 'messages', 'max_tokens'] },
  { label: 'batching policy probe', terms: ['/v1/config', "jq '.batching, .decode_runtime.max_decode_batch'"] },
  { label: 'CUDA graph policy probe', terms: ["jq '.cuda_graphs'", "jq '.decode_runtime.cuda_graphs'", 'kiln config --file kiln.toml'] },
  { label: 'ROCm execution-policy probe', terms: ['rocm_graphs_unavailable_reason', 'rocm_graph_telemetry', 'rocm_graph_telemetry_unavailable_reason', 'rocm_synchronization', "grep -e '^kiln_rocm_"] },
  { label: 'streaming-prefill policy probe', terms: ["jq '.streaming_prefill'", "jq '.prefill_runtime.streaming_prefill'"] },
];

const expectedTroubleshootingFragments = ['cuda-graph-policy', 'rocm-token-pauses'];

const expectedTroubleshootingLinks = [
  { label: 'Quickstart', href: 'quickstart.html' },
  { label: 'GRPO Guide', href: 'grpo.html' },
  { label: 'Architecture', href: 'architecture.html' },
  { label: 'API Reference', href: 'api.html' },
  { label: 'CLI Reference', href: 'cli.html' },
  { label: 'Configuration', href: 'docs/configuration/' },
  { label: 'Benchmarks', href: 'docs/benchmarks/' },
];

function fail(message) {
  throw new Error(message);
}

function normalizedHtmlText(html) {
  return html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/&hellip;/g, '...')
    .replace(/&rarr;/g, '->')
    .replace(/&mdash;/g, '-')
    .replace(/&ndash;/g, '-')
    .replace(/&rsquo;/g, "'")
    .replace(/&lsquo;/g, "'")
    .replace(/&ldquo;/g, '"')
    .replace(/&rdquo;/g, '"')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function missingNormalizedTerms(text, terms) {
  const normalized = text.toLowerCase();
  return terms.filter((term) => !normalized.includes(term.toLowerCase()));
}

function validateRuntimeEnvironmentBoundaryDocumentationSourceContract() {
  const expectedDeviceRemapEnvironmentNames = [
    'CUDA_VISIBLE_DEVICES',
    'NVIDIA_VISIBLE_DEVICES',
    'CUDA_DEVICE_ORDER',
    'ROCR_VISIBLE_DEVICES',
    'HIP_VISIBLE_DEVICES',
    'GPU_DEVICE_ORDINAL',
    'ZE_AFFINITY_MASK',
    'ONEAPI_DEVICE_SELECTOR',
    'SYCL_DEVICE_FILTER',
    'MESA_VK_DEVICE_SELECT',
    'DRI_PRIME',
    'VK_ICD_FILENAMES',
    'VK_DRIVER_FILES',
  ];
  const startupEnvironmentSource = readFileSync(
    resolve(repoRoot, 'crates/kiln-memory/src/startup_environment.rs'),
    'utf8',
  );
  const unionBody = startupEnvironmentSource.match(
    /const ALL_DEVICE_REMAP_ENV: &\[&str\] = &\[([\s\S]*?)\];/,
  )?.[1];
  if (!unionBody) {
    fail('startup_environment.rs: missing closed ALL_DEVICE_REMAP_ENV list');
  }
  const sourceNames = [...unionBody.matchAll(/"([A-Z0-9_]+)"/g)]
    .map((match) => match[1]);
  if (JSON.stringify(sourceNames) !== JSON.stringify(expectedDeviceRemapEnvironmentNames)) {
    fail(
      `startup_environment.rs: expected exact closed driver-remap list ${expectedDeviceRemapEnvironmentNames.join(', ')}, got ${sourceNames.join(', ')}`,
    );
  }

  const configurationSource = readFileSync(resolve(repoRoot, 'docs/CONFIGURATION.md'), 'utf8');
  const missingNames = missingNormalizedTerms(
    configurationSource,
    expectedDeviceRemapEnvironmentNames,
  );
  if (missingNames.length > 0) {
    fail(`docs/CONFIGURATION.md: incomplete driver-remap safety reference: ${missingNames.join(', ')}`);
  }
}

function validateSftLossRouteDocumentationSourceContract() {
  const manifest = JSON.parse(readFileSync(resolve(repoRoot, 'docs/site/docs-manifest.json'), 'utf8'));
  const expectedTrainingOrder = [
    'native-sft-profile',
    'sft-ingestion',
    'sft-tokenization',
    'training-checkpoints',
    'grpo',
    'echo',
    'dataset-splits',
    'evals',
  ];
  const trainingSlugs = manifest.documents
    .filter((document) => document.section === 'training')
    .map((document) => document.slug);
  const orderedTrainingSlugs = trainingSlugs.filter((slug) => expectedTrainingOrder.includes(slug));
  if (JSON.stringify(orderedTrainingSlugs) !== JSON.stringify(expectedTrainingOrder)) {
    fail(`docs/site/docs-manifest.json: training references must lead with ${expectedTrainingOrder.join(', ')}`);
  }

  const requiredDescriptions = new Map([
    ['configuration', ['mechanically derived environment override', 'backend-owned policy', 'SFT loss routing']],
    ['architecture', ['route-bound training', 'checkpoint identity']],
    ['native-sft-profile', ['backend-owned loss route', 'memory admission', 'receipt', 'checkpoint identity']],
    ['training-checkpoints', ['route-bound SFT planning identity']],
  ]);
  for (const [slug, terms] of requiredDescriptions) {
    const description = manifest.documents.find((document) => document.slug === slug)?.description || '';
    const missing = missingNormalizedTerms(description, terms);
    if (missing.length > 0) {
      fail(`docs/site/docs-manifest.json: ${slug} description missing route discoverability terms: ${missing.join(', ')}`);
    }
  }

  const configurationSource = readFileSync(resolve(repoRoot, 'docs/CONFIGURATION.md'), 'utf8');
  const retiredOverrideTableRows = configurationSource
    .split('\n')
    .filter((line) => line.trimStart().startsWith('|') && line.includes('KILN_USE_FLCE'));
  if (retiredOverrideTableRows.length > 0) {
    fail('docs/CONFIGURATION.md: retired KILN_USE_FLCE must not appear as a supported or deprecated configuration-table entry');
  }

  const metalPolicySource = readFileSync(
    resolve(repoRoot, 'crates/kiln-model/src/metal_policy.rs'),
    'utf8',
  );
  const metalPolicyBody = metalPolicySource.match(
    /pub struct MetalKernelPolicy \{([\s\S]*?)\n\}/,
  )?.[1];
  if (!metalPolicyBody) {
    fail('crates/kiln-model/src/metal_policy.rs: missing MetalKernelPolicy definition');
  }
  const metalPolicyRoutes = [...metalPolicyBody.matchAll(/pub\(crate\)\s+([a-z0-9_]+): bool,/g)]
    .map((match) => match[1]);
  if (metalPolicyRoutes.length !== 46 || new Set(metalPolicyRoutes).size !== 46) {
    fail(`MetalKernelPolicy must expose exactly 46 unique routes, got ${metalPolicyRoutes.length}`);
  }
  const retiredMetalEnvironmentNames = [
    'KILN_DISABLE_FUSED_CONV1D',
    'KILN_DISABLE_FUSED_GDN_GATES',
    'KILN_DISABLE_GDN_KERNEL',
    'KILN_DISABLE_METAL_ATTN_GATE_FUSION',
    'KILN_DISABLE_METAL_CONV1D_PREFILL',
    'KILN_DISABLE_METAL_FUSED_CONV1D',
    'KILN_DISABLE_METAL_FUSED_QKV_PROJ',
    'KILN_DISABLE_METAL_GATED_RMSNORM',
    'KILN_DISABLE_METAL_GDN_DECODE_GATES_RECURRENT',
    'KILN_DISABLE_METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM',
    'KILN_DISABLE_METAL_GDN_FORWARD_SUBSTITUTION',
    'KILN_DISABLE_METAL_GDN_GATES',
    'KILN_DISABLE_METAL_GDN_IN_PROJ_FUSION',
    'KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_PAIR',
    'KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_QUAD',
    'KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_TRIPLE',
    'KILN_DISABLE_METAL_GDN_IN_PROJ_SERIAL_VECTOR_LOAD',
    'KILN_DISABLE_METAL_GDN_IN_PROJ_SERIAL_X2_LOAD',
    'KILN_DISABLE_METAL_GDN_PREFILL_AB_IN_PROJ',
    'KILN_DISABLE_METAL_GDN_PREFILL_DECAY_RECURRENT',
    'KILN_DISABLE_METAL_GDN_PREFILL_QKV_CONV_SPLIT',
    'KILN_DISABLE_METAL_GDN_QKV_CONV_NORM',
    'KILN_DISABLE_METAL_GDN_QK_NORM',
    'KILN_DISABLE_METAL_GDN_RECURRENT',
    'KILN_DISABLE_METAL_LM_HEAD_ARGMAX',
    'KILN_DISABLE_METAL_LM_HEAD_ARGMAX_GPU_REDUCE',
    'KILN_DISABLE_METAL_LM_HEAD_ARGMAX_ROWS',
    'KILN_DISABLE_METAL_LM_HEAD_SAMPLE',
    'KILN_DISABLE_METAL_LORA_DELTA_DECODE',
    'KILN_DISABLE_METAL_MLP_GATE_UP_FUSION',
    'KILN_DISABLE_METAL_MLP_GATE_UP_ROW_PAIR',
    'KILN_DISABLE_METAL_MLP_GATE_UP_ROW_QUAD',
    'KILN_DISABLE_METAL_MLP_GATE_UP_ROW_QUAD_VECTOR_LOAD',
    'KILN_DISABLE_METAL_MLP_GATE_UP_ROW_TRIPLE',
    'KILN_DISABLE_METAL_MLP_GATE_UP_SERIAL_DEDICATED',
    'KILN_DISABLE_METAL_MLP_GATE_UP_SERIAL_VECTOR_LOAD',
    'KILN_DISABLE_METAL_MLP_SILU_MUL',
    'KILN_DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS',
    'KILN_DISABLE_METAL_PAGED_KV_WRITE_TOKEN_MAJOR',
    'KILN_DISABLE_METAL_RMSNORM',
    'KILN_DISABLE_METAL_SDPA',
    'KILN_DISABLE_METAL_SDPA_FULL',
    'KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV',
    'KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_PAIR',
    'KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD',
    'KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD_TILE8',
    'KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_TRIPLE_TILE8',
    'KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE16',
    'KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8',
    'KILN_DISABLE_RMSNORM_KERNEL',
    'KILN_ENABLE_METAL_LM_HEAD_ARGMAX',
  ];
  if (
    retiredMetalEnvironmentNames.length !== 51
    || new Set(retiredMetalEnvironmentNames).size !== 51
  ) {
    fail(
      `Metal policy docs must cover exactly 51 unique retired environment names, got ${retiredMetalEnvironmentNames.length}`,
    );
  }
  const missingMetalConfigurationTerms = missingNormalizedTerms(configurationSource, [
    'accelerator.metal_kernel_profile',
    'KILN_ACCELERATOR_METAL_KERNEL_PROFILE',
    ...metalPolicyRoutes,
    ...retiredMetalEnvironmentNames,
  ]);
  if (missingMetalConfigurationTerms.length > 0) {
    fail(`docs/CONFIGURATION.md: incomplete Metal policy reference: ${missingMetalConfigurationTerms.join(', ')}`);
  }

  const staticArchitecturePath = resolve(repoRoot, 'docs/site/architecture.html');
  const staticArchitectureHtml = readFileSync(staticArchitecturePath, 'utf8');
  const staticArchitectureText = normalizedHtmlText(staticArchitectureHtml);
  const missingArchitectureTerms = missingNormalizedTerms(staticArchitectureText, [
    'device-capability policy derived from standardized limits and features',
    'not a device name',
    'qualification records correctness and performance',
    'never rewrites product dispatch',
    'each optional Vulkan route is admitted only when',
    'unrelated fast paths remain eligible',
    'without appearing in an allowlist',
    'source-level quarantine',
    'not device-name routing',
    'SftFlceLossRoute',
    'requests cannot select it',
    'no TOML field',
    'does not claim that the complete training path is available',
    'OpenEnv episodes are a native training input',
    'ordered seed-free plan',
    'production path depends only on OpenEnv compatibility',
    'pumps Ping/Pong control frames while the model thinks',
    'poisons the session permanently',
  ]);
  if (missingArchitectureTerms.length > 0) {
    fail(`docs/site/architecture.html: reader architecture contract missing terms: ${missingArchitectureTerms.join(', ')}`);
  }
  if (!staticArchitectureHtml.includes('id="backend-owned-sft-loss-routing"')) {
    fail('docs/site/architecture.html: missing stable #backend-owned-sft-loss-routing anchor');
  }
  if (!staticArchitectureHtml.includes('id="openenv-rl-workflow"')) {
    fail('docs/site/architecture.html: missing stable #openenv-rl-workflow anchor');
  }
  const nativeSftLink = expectedArchitectureLinks
    .find((link) => link.label === 'generated Native SFT profile')?.href;
  if (!nativeSftLink || !staticArchitectureHtml.includes(`href="${nativeSftLink}"`)) {
    fail('docs/site/architecture.html: missing generated Native SFT profile route-contract link');
  }

  const troubleshootingHtml = readFileSync(resolve(repoRoot, 'docs/site/troubleshooting.html'), 'utf8');
  const troubleshootingText = normalizedHtmlText(troubleshootingHtml);
  const missingTroubleshootingTerms = missingNormalizedTerms(troubleshootingText, [
    'An OpenEnv run cannot start, stalls, or fails verification',
    'exactly one environment_reset_options object per URL',
    'groups must cover every endpoint',
    'pumps Ping/Pong while inference is pending',
    'poisons ambiguous lock-step sessions',
    'do not edit the receipt',
  ]);
  if (missingTroubleshootingTerms.length > 0) {
    fail(`docs/site/troubleshooting.html: OpenEnv diagnostics missing terms: ${missingTroubleshootingTerms.join(', ')}`);
  }
  if (!troubleshootingHtml.includes('id="openenv-runs"')) {
    fail('docs/site/troubleshooting.html: missing stable #openenv-runs anchor');
  }

  const apiHtml = readFileSync(resolve(repoRoot, 'docs/site/api.html'), 'utf8');
  const apiText = normalizedHtmlText(apiHtml);
  const missingApiTerms = missingNormalizedTerms(apiText, [
    'sft_loss_route',
    'kiln.training-checkpoint-planning.v4',
    'prior SFT v3 checkpoints fail closed',
    'GRPO and OPD',
    'kiln.training-checkpoint-planning.v3',
    'variant_invariant_fixed_output_v5',
    'exactly 64 ascending six-digit integers',
    'server truncation before the target is expected',
    'response_oracle_target_integer_count = 64',
    'slow_response_target_integer_count = 1024',
    'long_prefill_marker_role = "long-prefill"',
    'before_slow_start_after_first_token',
    'first producer-ready token',
    '256-token pressure peer',
    '36 fixed bytes plus 12 bytes per requested candidate',
    'device-to-host transfer and host work are O(TK)',
    'kernel still scans all V logits',
    "observed token's exact full-vocabulary rank",
    'ascending token-ID tie breaking',
    'same cancellation, external-yield settlement, and backend-quarantine boundary',
    'serialized teacher query, not a high-throughput generation route',
    'kiln_prompt_logprob_selection_chunks_total',
    'kiln_prompt_logprob_selection_rows_total',
    'compact_device',
    'bounded_host_fallback',
  ]);
  if (missingApiTerms.length > 0) {
    fail(`docs/site/api.html: retained deep-reference wording missing terms: ${missingApiTerms.join(', ')}`);
  }
  if (apiText.includes('current path performs correctness-first o(tv) host readback')) {
    fail('docs/site/api.html: remove the stale blanket O(TV) prompt-logprob claim');
  }
}

function validateReadmeStartupBanner() {
  const readmePath = resolve(repoRoot, 'README.md');
  const readme = readFileSync(readmePath, 'utf8');
  const bannerMatch = readme.match(/```[\s\S]*?K I L N[\s\S]*?Endpoints:[\s\S]*?```/);
  if (!bannerMatch) {
    fail('README.md: missing Quick Start startup banner snippet');
  }

  const banner = bannerMatch[0];
  const expectedLabels = ['Mode:', 'CUDA:', 'GPU:', 'VRAM:', 'Listen:', 'Endpoints:'];
  const missingLabels = expectedLabels.filter((label) => !banner.includes(label));
  if (missingLabels.length > 0) {
    fail(`README.md: startup banner missing labels: ${missingLabels.join(', ')}`);
  }
  if (!/Mode:\s+GPU inference/.test(banner)) {
    fail('README.md: startup banner Mode line must show GPU inference');
  }

  const labelPositions = expectedLabels.map((label) => [label, banner.indexOf(label)]);
  const outOfOrder = labelPositions.find(([, position], index) => (
    index > 0 && position < labelPositions[index - 1][1]
  ));
  if (outOfOrder) {
    fail(`README.md: startup banner label order drifted near ${outOfOrder[0]}`);
  }
}

function validateReadmeMedia() {
  const readmePath = resolve(repoRoot, 'README.md');
  const readme = readFileSync(readmePath, 'utf8');
  const dashboardImagePath = 'docs/site/assets/server-ui-dashboard.png';

  if (!readme.includes(dashboardImagePath)) {
    fail(`README.md: missing dashboard screenshot reference ${dashboardImagePath}`);
  }
  if (!existsSync(resolve(repoRoot, dashboardImagePath))) {
    fail(`README.md: referenced dashboard screenshot does not exist: ${dashboardImagePath}`);
  }

  const dashboardImagePattern = new RegExp(`!\\[([^\\]]+)\\]\\(${dashboardImagePath.replaceAll('/', '\\/')}\\)`);
  const dashboardImageMatch = readme.match(dashboardImagePattern);
  if (!dashboardImageMatch) {
    fail(`README.md: dashboard screenshot reference must include alt text for ${dashboardImagePath}`);
  }

  const altText = dashboardImageMatch[1].toLowerCase().replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim();
  const requiredAltTerms = ['dashboard', 'status', 'adapters', 'training'];
  const missingAltTerms = requiredAltTerms.filter((term) => !altText.includes(term));
  if (!altText.includes('chat') && !altText.includes('quick inference')) {
    missingAltTerms.push('chat or quick inference');
  }
  if (missingAltTerms.length > 0) {
    fail(`README.md: dashboard screenshot alt text missing terms: ${missingAltTerms.join(', ')}`);
  }

  const requiredDemoLinks = [
    'https://ericflo.github.io/kiln/demo/',
    'docs/site/demo/',
  ];
  const missingDemoLinks = requiredDemoLinks.filter((link) => !readme.includes(link));
  if (missingDemoLinks.length > 0) {
    fail(`README.md: missing demo/asciicast links: ${missingDemoLinks.join(', ')}`);
  }
}

function validateCurrentPerformancePositioning() {
  const indexPath = resolve(repoRoot, 'docs/site/index.html');
  const index = readFileSync(indexPath, 'utf8');
  const requiredTerms = [
    'Latest verified performance and release provenance',
    '13.46',
    '74.29',
    '2.81',
    'Vulkan decode · recovered',
    'verified source · f3ae29e4a',
    'Capability-led Vulkan',
    'never a GPU name, vendor ID, PCI ID, or',
    'machine allowlist',
    'Latest server release',
    'Five native artifacts',
    'CUDA 12.4',
    'ROCm 7.2.4',
    'Apple Silicon',
    'Checksums · attestations · third-party licenses',
    'Honest boundary',
    'verified source evidence, not yet a published',
    'Kiln makes no vLLM parity claim',
    'Serving-profile boundary:',
    'stable',
    'experimental',
    'docs/benchmarks/',
    'assets/og-image-v3.png',
  ];
  const missingTerms = requiredTerms.filter((term) => !index.includes(term));
  if (missingTerms.length > 0) {
    fail(`docs/site/index.html: current performance positioning missing terms: ${missingTerms.join(', ')}`);
  }

  const retiredTerms = [
    '44.75',
    'A6000 benchmarks',
    'KILN_W4A16=1',
    'assets/og-image.png',
    '0.455',
    'Vulkan, stability only',
    'unacceptable serving performance',
    'Source-bound Strix Halo development receipt',
  ];
  const presentRetiredTerms = retiredTerms.filter((term) => index.includes(term));
  if (presentRetiredTerms.length > 0) {
    fail(`docs/site/index.html: retired performance claims remain: ${presentRetiredTerms.join(', ')}`);
  }

  const socialAssetPath = resolve(repoRoot, 'docs/site/assets/og-image-v3.png');
  if (!existsSync(socialAssetPath)) {
    fail('docs/site/assets/og-image-v3.png: current social preview is missing');
  }
  const socialAsset = readFileSync(socialAssetPath);
  const pngSignature = socialAsset.subarray(0, 8).toString('hex');
  const width = socialAsset.length >= 24 ? socialAsset.readUInt32BE(16) : 0;
  const height = socialAsset.length >= 24 ? socialAsset.readUInt32BE(20) : 0;
  if (pngSignature !== '89504e470d0a1a0a' || width !== 1200 || height !== 630) {
    fail(`docs/site/assets/og-image-v3.png: expected a 1200x630 PNG, got ${width}x${height}`);
  }
  if (socialAsset.length > 300 * 1024) {
    fail(`docs/site/assets/og-image-v3.png: social preview must stay at or below 300 KiB, got ${socialAsset.length} bytes`);
  }

  const staleSocialReferences = [];
  for (const sitePage of pages) {
    const source = readFileSync(resolve(repoRoot, sitePage.path), 'utf8');
    if (source.includes('assets/og-image.png') || source.includes('assets/og-image-v2.png')) {
      staleSocialReferences.push(sitePage.path);
    }
  }
  if (staleSocialReferences.length > 0) {
    fail(`static site pages still reference the retired social preview: ${staleSocialReferences.join(', ')}`);
  }

  const retiredAssets = [
    'desktop-about.png',
    'desktop-dashboard.png',
    'desktop-logs.png',
    'desktop-settings.png',
    'og-image.png',
    'og-image-v2.png',
    'server-ui-adapters.png',
    'server-ui-playground.png',
    'server-ui-training.png',
  ];
  const presentRetiredAssets = retiredAssets.filter((name) => (
    existsSync(resolve(repoRoot, 'docs/site/assets', name))
  ));
  if (presentRetiredAssets.length > 0) {
    fail(`docs/site/assets: retired, unreferenced public assets must not ship: ${presentRetiredAssets.join(', ')}`);
  }
}

function tagAttributeValue(tag, attribute) {
  const pattern = '\\b' + attribute + '\\s*=\\s*(?:"([^"]*)"|\\\'([^\\\']*)\\\'|([^\\s"\\\'=<>`]+))';
  const match = tag.match(new RegExp(pattern, 'i'));
  return decodeHtmlAttribute(match?.[1] ?? match?.[2] ?? match?.[3] ?? '').trim();
}

function metaContent(html, selectorAttribute, selectorValue) {
  for (const match of html.matchAll(/<meta\b[^>]*>/gi)) {
    const tag = match[0];
    if (tagAttributeValue(tag, selectorAttribute).toLowerCase() === selectorValue.toLowerCase()) {
      return tagAttributeValue(tag, 'content');
    }
  }
  return '';
}

function validateSocialMetadataForHtml(displayPath, html, canonical, { enforceSnippetLength = false } = {}) {
  const titleMatch = html.match(/<title>([\s\S]*?)<\/title>/i);
  const title = decodeHtmlAttribute(titleMatch?.[1] ?? '').replace(/<[^>]+>/g, '').trim();
  const description = metaContent(html, 'name', 'description');
  const socialImage = 'https://ericflo.github.io/kiln/assets/og-image-v3.png';
  const expectedImageAlt = 'Kiln — Serve it. Teach it. Watch it get better. OpenAI-compatible inference, live LoRA training, and local evals in one server.';
  const expected = [
    ['theme-color', metaContent(html, 'name', 'theme-color'), '#0a0908'],
    ['color-scheme', metaContent(html, 'name', 'color-scheme'), 'dark'],
    ['og:type', metaContent(html, 'property', 'og:type'), 'website', 'article'],
    ['og:locale', metaContent(html, 'property', 'og:locale'), 'en_US'],
    ['og:site_name', metaContent(html, 'property', 'og:site_name'), 'Kiln'],
    ['og:url', metaContent(html, 'property', 'og:url'), canonical],
    ['og:image', metaContent(html, 'property', 'og:image'), socialImage],
    ['og:image:type', metaContent(html, 'property', 'og:image:type'), 'image/png'],
    ['og:image:width', metaContent(html, 'property', 'og:image:width'), '1200'],
    ['og:image:height', metaContent(html, 'property', 'og:image:height'), '630'],
    ['og:image:alt', metaContent(html, 'property', 'og:image:alt'), expectedImageAlt],
    ['twitter:card', metaContent(html, 'name', 'twitter:card'), 'summary_large_image'],
    ['twitter:image', metaContent(html, 'name', 'twitter:image'), socialImage],
    ['twitter:image:alt', metaContent(html, 'name', 'twitter:image:alt'), expectedImageAlt],
  ];

  for (const [label, actual, ...allowed] of expected) {
    if (!allowed.includes(actual)) {
      fail(`${displayPath}: ${label} must be ${allowed.join(' or ')}, got ${actual || '(missing)'}`);
    }
  }

  for (const label of ['og:title', 'og:description']) {
    if (!metaContent(html, 'property', label)) fail(`${displayPath}: ${label} must not be empty`);
  }
  for (const label of ['twitter:title', 'twitter:description']) {
    if (!metaContent(html, 'name', label)) fail(`${displayPath}: ${label} must not be empty`);
  }
  if (!html.includes('rel="apple-touch-icon"')) {
    fail(`${displayPath}: missing apple-touch-icon`);
  }
  if (html.includes('og-image-v2.png') || html.includes('/assets/og-image.png')) {
    fail(`${displayPath}: references a retired social preview`);
  }

  if (enforceSnippetLength) {
    if (title.length < 20 || title.length > 65) {
      fail(`${displayPath}: title must be 20–65 characters, got ${title.length}`);
    }
    if (description.length < 80 || description.length > 160) {
      fail(`${displayPath}: meta description must be 80–160 characters, got ${description.length}`);
    }
  }
}

function validateStaticSocialMetadata() {
  for (const sitePage of pages) {
    const html = readFileSync(resolve(repoRoot, sitePage.path), 'utf8');
    validateSocialMetadataForHtml(
      sitePage.path,
      html,
      expectedCanonicalHref(sitePage.path),
      { enforceSnippetLength: true },
    );
  }
}

function validateNotFoundPage() {
  const notFoundPath = resolve(siteRoot, '404.html');
  if (!existsSync(notFoundPath)) {
    fail(`${publishedPath('404.html')}: GitHub Pages requires a useful custom not-found page`);
  }
  const html = readFileSync(notFoundPath, 'utf8');
  const requiredTerms = [
    '<meta name="robots" content="noindex, follow">',
    'That Kiln page isn’t here.',
    'The documentation directory and search cover every published Kiln guide and reference.',
    'Open the documentation',
    'Start the quickstart',
    'Diagnose a problem',
    'Check current performance',
    'Inspect the API',
    'Return to Kiln home',
  ];
  const missingTerms = requiredTerms.filter((term) => !html.includes(term));
  if (missingTerms.length > 0) {
    fail(`${publishedPath('404.html')}: incomplete recovery copy: ${missingTerms.join(', ')}`);
  }
  if (/<link\b[^>]*\brel\s*=\s*["']?canonical/i.test(html)) {
    fail(`${publishedPath('404.html')}: a not-found response must not declare a canonical content URL`);
  }
  const requiredLinks = [
    'https://ericflo.github.io/kiln/',
    'https://ericflo.github.io/kiln/docs/',
    'https://ericflo.github.io/kiln/quickstart.html',
    'https://ericflo.github.io/kiln/troubleshooting.html',
    'https://ericflo.github.io/kiln/docs/benchmarks/',
    'https://ericflo.github.io/kiln/api.html',
  ];
  const missingLinks = requiredLinks.filter((href) => !html.includes(`href="${href}"`));
  if (missingLinks.length > 0) {
    fail(`${publishedPath('404.html')}: missing path-safe recovery links: ${missingLinks.join(', ')}`);
  }
}

function normalizeReadmeForColdReader(value) {
  return value
    .toLowerCase()
    .replace(/<[^>]+>/g, ' ')
    .replace(/[`*_~[\]()]/g, ' ')
    .replace(/[-_/]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function assertReadmeColdReaderTerms(source, label, terms) {
  const missingTerms = terms.filter((term) => !source.includes(normalizeReadmeForColdReader(term)));
  if (missingTerms.length > 0) {
    fail(`README.md: missing cold-reader ${label}: ${missingTerms.join(', ')}`);
  }
}

function assertReadmeColdReaderAny(source, label, terms) {
  if (!terms.some((term) => source.includes(normalizeReadmeForColdReader(term)))) {
    fail(`README.md: missing cold-reader ${label}: expected one of ${terms.join(', ')}`);
  }
}

function validateReadmeColdReaderCoverage() {
  const readmePath = resolve(repoRoot, 'README.md');
  const readme = readFileSync(readmePath, 'utf8');
  const normalizedReadme = normalizeReadmeForColdReader(readme);

  const requiredTermGroups = [
    ['what-it-is paragraph', ['single-GPU inference server', 'live LoRA training', 'one GPU', 'one Rust binary']],
    ['serving-profile contract', ['stable profile', 'experimental', 'maintenance', 'Serving Profiles', 'docs/SERVING_PROFILES.md']],
    ['install/run paths', ['Desktop App', 'server binary', 'Docker', 'source CLI', 'kiln serve']],
    ['GRPO loop', ['GRPO Loop', 'killer feature', 'generate completions', 'score them', 'reward function']],
    ['embedded dashboard', ['/ui', 'dashboard']],
    ['demo/asciicast coverage', ['asciicasts', 'https://ericflo.github.io/kiln/demo/', 'docs/site/demo/']],
    ['embedded dashboard screenshot', ['docs/site/assets/server-ui-dashboard.png']],
    ['required links', ['QUICKSTART.md', 'CHANGELOG.md', 'LICENSE']],
  ];

  for (const [label, terms] of requiredTermGroups) {
    assertReadmeColdReaderTerms(normalizedReadme, label, terms);
  }
  assertReadmeColdReaderAny(normalizedReadme, 'docs site or website link', [
    'https://ericflo.github.io/kiln/',
    'docs/site',
  ]);
}

function validateReadmeImageReferences() {
  const sourcePath = 'README.md';
  const readme = readFileSync(resolve(repoRoot, sourcePath), 'utf8');
  const imageTargets = extractMarkdownLocalTargets(readme)
    .filter((link) => link.text === 'image' || /\.(?:avif|gif|jpe?g|png|svg|webp)(?:[?#]|$)/i.test(link.target))
    .map((link) => link.target.trim())
    .filter((target) => !isIgnoredMarkdownTarget(target));

  if (imageTargets.length === 0) {
    fail('README.md: missing image references for cold-reader visual context');
  }

  for (const target of imageTargets) {
    const { pathPart } = markdownTargetParts(target);
    const resolvedPath = resolveMarkdownTargetPath(sourcePath, pathPart);
    if (!existsSync(resolvedPath) || statSync(resolvedPath).isDirectory()) {
      fail(`README.md: broken image reference ${target} (resolved path: ${relative(repoRoot, resolvedPath)})`);
    }
  }
}

function validateQuickstartMarkdownMedia() {
  const quickstartPath = resolve(repoRoot, 'QUICKSTART.md');
  const quickstart = readFileSync(quickstartPath, 'utf8');
  const dashboardImagePath = 'docs/site/assets/server-ui-dashboard.png';

  if (!quickstart.includes(dashboardImagePath)) {
    fail(`QUICKSTART.md: missing dashboard screenshot reference ${dashboardImagePath}`);
  }
  if (!existsSync(resolve(repoRoot, dashboardImagePath))) {
    fail(`QUICKSTART.md: referenced dashboard screenshot does not exist: ${dashboardImagePath}`);
  }

  const dashboardImagePattern = new RegExp(`!\\[([^\\]]+)\\]\\(${dashboardImagePath.replaceAll('/', '\\/')}\\)`);
  const dashboardImageMatch = quickstart.match(dashboardImagePattern);
  if (!dashboardImageMatch) {
    fail(`QUICKSTART.md: dashboard screenshot reference must include alt text for ${dashboardImagePath}`);
  }

  const altText = dashboardImageMatch[1].toLowerCase().replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim();
  const requiredAltTerms = ['dashboard', 'status', 'adapters', 'training'];
  const missingAltTerms = requiredAltTerms.filter((term) => !altText.includes(term));
  if (!altText.includes('chat') && !altText.includes('quick inference')) {
    missingAltTerms.push('chat or quick inference');
  }
  if (missingAltTerms.length > 0) {
    fail(`QUICKSTART.md: dashboard screenshot alt text missing terms: ${missingAltTerms.join(', ')}`);
  }
}

function validateEmbeddedUiHelpLinks() {
  const uiPath = resolve(repoRoot, 'crates/kiln-server/src/ui/index.html');
  const ui = readFileSync(uiPath, 'utf8');
  const navMatch = ui.match(/<nav class="header-help" aria-label="Help links">[\s\S]*?<\/nav>/);
  if (!navMatch) {
    fail('crates/kiln-server/src/ui/index.html: missing embedded UI help nav');
  }

  const nav = navMatch[0];
  const missingLinks = expectedEmbeddedUiHelpLinks.filter(({ label, href }) => {
    const escapedHref = href.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const linkPattern = new RegExp(`<a\\s+href="${escapedHref}"[^>]*>${label}<\\/a>`);
    return !linkPattern.test(nav);
  });
  if (missingLinks.length > 0) {
    fail(`crates/kiln-server/src/ui/index.html: embedded UI help nav missing links: ${missingLinks.map(({ label }) => label).join(', ')}`);
  }
}

function validateDesktopDocumentationLinks() {
  const expectedLinks = [
    'desktop/ui/dashboard.html',
    'desktop/ui/settings.html',
  ];
  const documentationLink = /<a\b[^>]*href="https:\/\/ericflo\.github\.io\/kiln\/docs\/"[^>]*>\s*Documentation\s*<\/a>/i;
  const missing = expectedLinks.filter((sourcePath) => (
    !documentationLink.test(readFileSync(resolve(repoRoot, sourcePath), 'utf8'))
  ));
  if (missing.length > 0) {
    fail(`desktop UI surfaces missing the Documentation entry point: ${missing.join(', ')}`);
  }
}

async function validateEmbeddedUiControlAccessibleNames(browser) {
  const uiPath = resolve(repoRoot, 'crates/kiln-server/src/ui/index.html');
  const page = await browser.newPage();
  await page.goto(pathToFileURL(uiPath).href, { waitUntil: 'domcontentloaded', timeout: 10000 });

  try {
    const result = await page.evaluate((expectedControls) => {
      const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
      const byIdText = (id) => normalize(document.getElementById(id)?.textContent || '');
      const labelTextFor = (element) => {
        const explicitLabels = element.id
          ? Array.from(document.querySelectorAll(`label[for="${CSS.escape(element.id)}"]`))
          : [];
        const implicitLabel = element.closest('label');
        return normalize([
          ...explicitLabels.map((label) => label.textContent || ''),
          implicitLabel?.textContent || '',
        ].join(' '));
      };
      const referencedText = (element, attribute) => normalize(
        (element.getAttribute(attribute) || '')
          .split(/\s+/)
          .filter(Boolean)
          .map(byIdText)
          .join(' '),
      );
      const accessibleName = (element) => normalize([
        element.getAttribute('aria-label') || '',
        referencedText(element, 'aria-labelledby'),
        labelTextFor(element),
        element.tagName === 'BUTTON' ? element.textContent || '' : '',
      ].join(' '));
      const accessibleDescription = (element) => referencedText(element, 'aria-describedby');

      return expectedControls.map((control) => {
        const element = document.querySelector(control.selector);
        if (!element) {
          return { selector: control.selector, missing: true };
        }

        const name = accessibleName(element);
        const description = accessibleDescription(element);
        return {
          selector: control.selector,
          name,
          description,
          missingLabelTerms: control.labelTerms.filter((term) => !name.includes(term)),
          missingDescriptionTerms: (control.descriptionTerms || [])
            .filter((term) => !description.includes(term)),
        };
      });
    }, expectedEmbeddedUiAccessibleControls);

    const missingControls = result.filter((control) => control.missing).map((control) => control.selector);
    if (missingControls.length > 0) {
      fail(`crates/kiln-server/src/ui/index.html: missing embedded UI controls: ${missingControls.join(', ')}`);
    }

    const unnamedControls = result
      .filter((control) => control.missingLabelTerms?.length > 0)
      .map((control) => `${control.selector} missing label terms ${control.missingLabelTerms.join(', ')}`);
    if (unnamedControls.length > 0) {
      fail(`crates/kiln-server/src/ui/index.html: embedded UI controls need accessible names: ${unnamedControls.join('; ')}`);
    }

    const undescribedControls = result
      .filter((control) => control.missingDescriptionTerms?.length > 0)
      .map((control) => `${control.selector} missing description terms ${control.missingDescriptionTerms.join(', ')}`);
    if (undescribedControls.length > 0) {
      fail(`crates/kiln-server/src/ui/index.html: embedded UI controls need aria-describedby help text: ${undescribedControls.join('; ')}`);
    }
  } finally {
    await page.close();
  }
}

function validateQuickstartServerBinaryPath() {
  const quickstart = readFileSync(resolve(repoRoot, 'QUICKSTART.md'), 'utf8');
  const choosePathSection = extractMarkdownSection(quickstart, 'Choose your path');
  if (!choosePathSection) {
    fail('QUICKSTART.md: missing ## Choose your path section');
  }

  const expectedPathRows = [
    ['Desktop App path', '**Desktop App (recommended)**'],
    ['Server binary path', '**Server binary (terminal-first)**'],
    ['Container path', '**Container**'],
    ['Source / CLI path', '**Source / CLI**'],
  ];
  for (const [label, term] of expectedPathRows) {
    assertIncludes(choosePathSection, term, `QUICKSTART.md: Choose your path ${label}`);
  }

  if (!/\[Running with Docker\]\(#running-with-docker\)/.test(choosePathSection)) {
    fail('QUICKSTART.md: Choose your path Container row must link to Running with Docker');
  }

  const prerequisitesSection = extractMarkdownSection(quickstart, 'Prerequisites');
  if (!prerequisitesSection) {
    fail('QUICKSTART.md: missing ## Prerequisites section');
  }

  const requiredPrerequisiteTerms = [
    'Container path',
    'Docker/GHCR',
    'NVIDIA Container Toolkit',
    'Qwen/Qwen3.5-4B',
    'No Rust toolchain',
    'prebuilt `ghcr.io/ericflo/kiln-server:latest` image',
  ];
  for (const term of requiredPrerequisiteTerms) {
    assertIncludes(prerequisitesSection, term, 'QUICKSTART.md: Container prerequisites');
  }

  const serverBinarySection = extractMarkdownSection(quickstart, 'Quick path: Server binary (terminal-first, no source build)');
  if (!serverBinarySection) {
    fail('QUICKSTART.md: missing ## Quick path: Server binary (terminal-first, no source build) section');
  }

  const requiredTerms = [
    'terminal-first',
    'no source build',
    'Qwen/Qwen3.5-4B',
    'SHA-256 sidecars',
    'kiln-v${KILN_VERSION}',
    'x86_64-unknown-linux-gnu-cuda124.tar.gz',
    'x86_64-unknown-linux-gnu-vulkan.tar.gz',
    'aarch64-apple-darwin-metal.tar.gz',
    'x86_64-pc-windows-msvc-cuda124.zip',
  ];
  for (const term of requiredTerms) {
    assertIncludes(serverBinarySection, term, 'QUICKSTART.md: Server binary path');
  }

  if (!/https:\/\/github\.com\/ericflo\/kiln\/releases\/download\/kiln-v/.test(serverBinarySection)) {
    fail('QUICKSTART.md: Server binary path must include at least one kiln-v release download command');
  }

  if (/^## 1\. Build Kiln\s*$/m.test(quickstart)) {
    fail('QUICKSTART.md: source build heading must stay optional, not generic mandatory-sounding "## 1. Build Kiln"');
  }

  const sourceBuildSection = extractMarkdownSection(quickstart, '1. Optional Source / CLI Branch: Build Kiln');
  if (!sourceBuildSection) {
    fail('QUICKSTART.md: missing optional Source / CLI build section');
  }

  const requiredSourceBuildTerms = [
    'Skip this section',
    'Desktop App',
    'prebuilt server binary',
    'container image',
    'do not require a source checkout or Rust build',
    '[Download Model Weights](#2-download-model-weights)',
  ];
  for (const term of requiredSourceBuildTerms) {
    assertIncludes(sourceBuildSection, term, 'QUICKSTART.md: optional Source / CLI branch');
  }
}

function validateReadmeQuickStartPaths() {
  const readme = readFileSync(resolve(repoRoot, 'README.md'), 'utf8');
  const quickStartSection = extractMarkdownSection(readme, 'Quick Start');
  if (!quickStartSection) {
    fail('README.md: missing ## Quick Start section');
  }

  const requiredPaths = [
    ['Desktop App path', 'Desktop App (recommended)'],
    ['Server binary path', 'Server binary (terminal-first, no source build)'],
    ['Container path', 'Container'],
    ['Source / CLI path', 'Source / CLI'],
  ];
  for (const [label, term] of requiredPaths) {
    assertIncludes(quickStartSection, term, `README.md: Quick Start ${label}`);
  }

  const serverBinaryMatch = quickStartSection.match(/\*\*Path 2 — Server binary \(terminal-first, no source build\):\*\*([\s\S]*?)(?=\n\*\*Path 3 — Container:\*\*)/);
  if (!serverBinaryMatch) {
    fail('README.md: Quick Start missing distinct Server binary path before Container path');
  }
  const serverBinaryPath = serverBinaryMatch[1];
  const requiredServerTerms = [
    'terminal-first',
    'no source build',
    'Qwen/Qwen3.5-4B',
    'kiln-v${KILN_VERSION}',
    'x86_64-unknown-linux-gnu-cuda124.tar.gz',
  ];
  for (const term of requiredServerTerms) {
    assertIncludes(serverBinaryPath, term, 'README.md: Server binary path');
  }
  if (!/https:\/\/github\.com\/ericflo\/kiln\/releases\/download\/kiln-v/.test(serverBinaryPath)) {
    fail('README.md: Server binary path must include at least one kiln-v release download command');
  }
  if (!/QUICKSTART\.md#quick-path-server-binary-terminal-first-no-source-build/.test(serverBinaryPath)) {
    fail('README.md: Server binary path must link to QUICKSTART.md full artifact matrix');
  }
}

function validateReadmeAdapterListSemantics() {
  const readme = readFileSync(resolve(repoRoot, 'README.md'), 'utf8');
  const adaptersRow = readme
    .split('\n')
    .find((line) => line.includes('`/v1/adapters`'));
  if (!adaptersRow) {
    fail('README.md: missing GET /v1/adapters API table row');
  }
  for (const phrase of staleAdapterListPhrases) {
    if (readme.includes(phrase)) {
      fail(`README.md: GET /v1/adapters wording must not say "${phrase}"`);
    }
  }
  for (const term of expectedAdapterListSemantics) {
    assertIncludes(adaptersRow, term, 'README.md: GET /v1/adapters saved/active semantics');
  }
}

function validateAdapterListStaleWordingSurfaces() {
  for (const filePath of adapterListStaleWordingSurfaces) {
    const contents = readFileSync(resolve(repoRoot, filePath), 'utf8');
    for (const phrase of staleAdapterListPhrases) {
      if (contents.includes(phrase)) {
        fail(`${filePath}: adapter-list wording must not say \"${phrase}\"`);
      }
    }
  }
}

function validateGrpoOverviewRequestsImports() {
  const readme = readFileSync(resolve(repoRoot, 'README.md'), 'utf8');
  const readmeSection = extractMarkdownSection(readme, 'The GRPO Loop');
  if (!readmeSection) {
    fail('README.md: missing ## The GRPO Loop section');
  }
  assertRequestsImportNearPost(readmeSection, 'README.md: The GRPO Loop');

  const index = readFileSync(resolve(repoRoot, 'docs/site/index.html'), 'utf8');
  const requiredLandingLoopTerms = [
    'id="learning-pass"',
    'POST /v1/train/grpo',
    'reward accepted',
    'adapter updated',
    'href="grpo.html"',
    'Walk through GRPO',
  ];
  const missingLandingLoopTerms = requiredLandingLoopTerms.filter((term) => !index.includes(term));
  if (missingLandingLoopTerms.length > 0) {
    fail(`docs/site/index.html: learning-pass overview missing terms: ${missingLandingLoopTerms.join(', ')}`);
  }
}

function validateGrpoDemoPayloadCue() {
  const driverPath = 'docs/site/demo/demo-grpo.sh';
  const driver = readFileSync(resolve(repoRoot, driverPath), 'utf8');

  if (driver.includes('-d @demo-grpo.json')) {
    fail(`${driverPath}: displayed GRPO curl command must use a repo-root path, not -d @demo-grpo.json`);
  }

  if (!driver.includes('-d @docs/site/demo/demo-grpo.json')) {
    fail(`${driverPath}: displayed GRPO curl command must cue docs/site/demo/demo-grpo.json for repo-root copy/paste`);
  }
}

function validateLandingDesktopVersionSplitCue() {
  const index = readFileSync(resolve(repoRoot, 'docs/site/index.html'), 'utf8');
  const requiredTerms = [
    'The latest server release publishes five provenance-attested archives',
    'macOS binary is signed',
    'notarized',
    'current provenance-attested server release',
    'Linux',
    'ROCm',
    'Vulkan',
    'Apple Silicon',
    'Windows CUDA',
    'href="quickstart.html"',
    'https://github.com/ericflo/kiln/releases/latest',
  ];
  for (const term of requiredTerms) {
    assertIncludes(index, term, 'docs/site/index.html: release-platform cue');
  }
  if (index.includes('Signed releases are available')) {
    fail('docs/site/index.html: release copy must distinguish provenance attestation from platform code signing');
  }
}

function validateQuickstartDesktopVersionSplitCue() {
  const quickstart = readFileSync(resolve(repoRoot, 'docs/site/quickstart.html'), 'utf8');
  const desktopInstallMatch = quickstart.match(/<h3[^>]*>Kiln Desktop &middot; recommended<\/h3>([\s\S]*?)<table class="installer-table/);
  if (!desktopInstallMatch) {
    fail('docs/site/quickstart.html: missing Kiln Desktop install card copy');
  }

  const desktopInstallCopy = desktopInstallMatch[1];
  const requiredTerms = [
    'desktop-v0.2.16',
    'Desktop',
    'server',
    'release',
    'kiln-v*',
    'downloads',
    'verifies',
  ];
  for (const term of requiredTerms) {
    assertIncludes(desktopInstallCopy, term, 'docs/site/quickstart.html: Kiln Desktop version-split cue');
  }

  const desktopAssetLinks = [
    'releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_aarch64.dmg',
    'releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_x64-setup.exe',
    'releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_x64_en-US.msi',
    'releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_amd64.deb',
    'releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_amd64.AppImage',
  ];
  for (const assetLink of desktopAssetLinks) {
    assertIncludes(quickstart, assetLink, 'docs/site/quickstart.html: direct Desktop installer link');
  }
  assertIncludes(quickstart, 'class="install-alternatives"', 'docs/site/quickstart.html: advanced installer disclosure');
  const releaseIntegrityTerms = [
    'Release integrity',
    'Verify provenance before extraction',
    'Sigstore-backed GitHub',
    'build-provenance attestation',
    'macOS binary is',
    'also code-signed and notarized',
    'Linux and Windows archives do not carry',
    'gh attestation verify ARCHIVE',
    'gh attestation verify kiln-linux-cuda.tar.gz',
    '--repo ericflo/kiln',
    '--signer-workflow ericflo/kiln/.github/workflows/server-release.yml',
  ];
  for (const term of releaseIntegrityTerms) {
    assertIncludes(quickstart, term, 'docs/site/quickstart.html: server release integrity cue');
  }
  for (const term of [
    '-p 127.0.0.1:8420:8420',
    '-e KILN_SERVER_HOST=0.0.0.0',
    'ghcr.io/ericflo/kiln-server:latest',
    'replace <code>latest</code> with the release version or immutable digest you validated',
    'Device names and IDs are never routing inputs',
  ]) {
    assertIncludes(quickstart, term, 'docs/site/quickstart.html: safe capability-based runtime guidance');
  }
  if (quickstart.includes('Strix Halo')) {
    fail('docs/site/quickstart.html: install guidance must not route by a named machine or product');
  }
  assertIncludes(quickstart, 'Four checkpoints. One working endpoint.', 'docs/site/quickstart.html: first-success path');
}

function validateGrpoSafetyAndPromotionCues() {
  const grpo = readFileSync(resolve(repoRoot, 'docs/site/grpo.html'), 'utf8')
    .replace(/\s+/g, ' ');
  for (const term of [
    'experimental training workflow',
    'cannot start a <code>stable</code> process with a saved LoRA adapter',
    '<code>KILN_SERVER_SERVING_PROFILE=experimental</code>',
    '"auto_load": false',
    'evaluation should gate activation',
    'Recommended: bind activation to held-out evidence',
    '"post_eval": {',
    '"data_scope": "held-out"',
    'only <code>promoted</code> activates the adapter',
    'Manual review for an ungated artifact',
    '/v1/adapters/load',
    'The current <code>stable</code> profile cannot load or start with a saved LoRA adapter',
  ]) {
    assertIncludes(grpo, term, 'docs/site/grpo.html: safe profile-gated GRPO workflow');
  }
}

function validateEvalsOperationAndDistributionCues() {
  const evals = readFileSync(resolve(repoRoot, 'docs/site/evals.html'), 'utf8')
    .replace(/\s+/g, ' ');
  for (const term of [
    'subset diagnostic, not reproduction',
    'same bound identities and raw decoder bytes',
    'did this exact trained revision satisfy the declared evidence gate',
    '<code>eval_dir = "/path/to/eval"</code> in <code>[eval]</code>',
    '<code>eval.root</code> is not an accepted field',
    '<code>auto_detect</code> checks the target in this order',
    'It never selects <code>llm_judge</code> implicitly',
    'A compare job is diagnostic: it never activates an adapter',
    '"auto_load": true',
    'It is not included in the current prebuilt server archives',
    '-p kiln-server --bin kiln-eval',
  ]) {
    assertIncludes(evals, term, 'docs/site/evals.html: exact eval workflow and distribution guidance');
  }
  if (evals.includes('set the root with <code>[eval] root</code>')) {
    fail('docs/site/evals.html: eval.root is not a valid configuration field');
  }
}

function validateApiProfileAndMutationCues() {
  const api = readFileSync(resolve(repoRoot, 'docs/site/api.html'), 'utf8')
    .replace(/\s+/g, ' ');
  for (const term of [
    '<code>KILN_SERVER_SERVING_PROFILE=experimental</code>',
    '"output_name": "first-sft"',
    '"behavior_policy": "no_importance_correction"',
    '"auto_load": false',
    'returns text, not trainer-ready scored groups',
    'current hybrid Vulkan server rejects it before dataset admission',
    '<code>?force=true</code> to replace an existing name deliberately',
    '<code>data_scope: "held-out"</code>',
    'Diagnostic post-eval runs do not imply promotion',
    'A job can create or replace an adapter artifact',
    '"code": "serving_profile_conflict"',
    'curl -sS -w',
  ]) {
    assertIncludes(api, term, 'docs/site/api.html: profile-safe API workflow');
  }
  if (api.includes('<template>')) {
    fail('docs/site/api.html: dead template payload must not ship in the public API guide');
  }
}

function validateCliDistributionAndProfileCues() {
  const cli = readFileSync(resolve(repoRoot, 'docs/site/cli.html'), 'utf8')
    .replace(/\s+/g, ' ');
  for (const term of [
    'Release archives contain the <code>kiln</code> binary only',
    'Current prebuilt server archives contain only <code>kiln</code>',
    '--bin kiln-eval --bin kiln-bench',
    '--bin kiln-replay',
    '--bin vulkan_decode_microbench',
    'These CLI commands do not expose <code>auto_load</code> or <code>post_eval</code>',
    'server&rsquo;s <code>auto_load: true</code> training default',
    '<code>auto_load: false</code> or a held-out',
    '<code>kiln-eval</code> is a source-built client',
    'source-built developer tools rather than release-archive binaries',
    'The default <code>stable</code> profile cannot load a saved LoRA',
    'There is no trash or automatic undo',
    'OpenCode&rsquo;s OpenAI-compatible provider accepts a local endpoint without an <code>apiKey</code>',
    'https://opencode.ai/docs/providers/#custom-provider',
  ]) {
    assertIncludes(cli, term, 'docs/site/cli.html: exact distribution and profile boundary');
  }
  if (cli.includes('"apiKey": "unused"')) {
    fail('docs/site/cli.html: OpenCode custom-provider apiKey is optional for Kiln and must not be presented as required');
  }
}

function validateSelfHostedProductPageAssets() {
  const productPages = [
    'quickstart.html',
    'grpo.html',
    'evals.html',
    'api.html',
    'cli.html',
    'troubleshooting.html',
    'architecture.html',
    'demo/index.html',
  ];
  for (const relativePath of productPages) {
    const html = readFileSync(resolve(repoRoot, 'docs/site', relativePath), 'utf8');
    if (html.includes('cdn.tailwindcss.com')) {
      fail(`docs/site/${relativePath}: runtime Tailwind CDN is forbidden; use the self-hosted utility sheet`);
    }
    const prefix = relativePath.includes('/') ? '../' : '';
    if (relativePath === 'demo/index.html') {
      if (html.includes('css/utilities.css') || html.includes('css/kiln.css')) {
        fail(`docs/site/${relativePath}: standalone tour must not import the unused product-page stylesheets`);
      }
    } else {
      assertIncludes(
        html,
        `href="${prefix}css/utilities.css"`,
        `docs/site/${relativePath}: self-hosted utility sheet`,
      );
    }
    assertIncludes(
      html,
      `src="${prefix}js/site.js"`,
      `docs/site/${relativePath}: shared product-page shell`,
    );
    assertIncludes(
      html,
      "<script>document.documentElement.classList.add('product-page-enhanced');</script>",
      `docs/site/${relativePath}: pre-render product-page shell`,
    );
    const externalRuntimeAssets = Array.from(
      html.matchAll(/\b(?:src|href)="(https?:\/\/[^"]+\.(?:css|js)(?:[?#][^"]*)?)"/gi),
      (match) => match[1],
    );
    if (externalRuntimeAssets.length > 0) {
      fail(`docs/site/${relativePath}: third-party runtime assets are forbidden: ${externalRuntimeAssets.join(', ')}`);
    }
  }

  const asciinemaAssets = [
    {
      path: 'docs/site/demo/vendor/asciinema-player/asciinema-player.css',
      sha256: 'a2b6c49acad107f7c1358c30a7496fe89e7ec1939dd3e49b0645c03f291a4993',
    },
    {
      path: 'docs/site/demo/vendor/asciinema-player/asciinema-player.min.js',
      sha256: '704f17a2607f4e8e8d5c95b2ea6cc6a15dc63f5811daaabe46971470e2c39571',
    },
  ];
  for (const asset of asciinemaAssets) {
    const path = resolve(repoRoot, asset.path);
    if (!existsSync(path)) fail(`${asset.path}: missing pinned asciinema-player asset`);
    const actualSha256 = createHash('sha256').update(readFileSync(path)).digest('hex');
    if (actualSha256 !== asset.sha256) {
      fail(`${asset.path}: pinned asciinema-player SHA-256 mismatch; expected ${asset.sha256}, got ${actualSha256}`);
    }
  }

  for (const relativePath of ['index.html', 'quickstart.html']) {
    const html = readFileSync(resolve(repoRoot, 'docs/site', relativePath), 'utf8');
    assertIncludes(html, `srcset="${dashboardImageSrcset}"`, `docs/site/${relativePath}: responsive dashboard srcset`);
    assertIncludes(html, `sizes="${dashboardImageSizes}"`, `docs/site/${relativePath}: responsive dashboard sizes`);
  }
  const demoHtml = readFileSync(resolve(repoRoot, 'docs/site/demo/index.html'), 'utf8');
  for (const name of demoImageNames) {
    const srcset = `../assets/server-ui-${name}-720.webp 720w, ../assets/server-ui-${name}-1440.webp 1440w, ../assets/server-ui-${name}.webp 2880w`;
    assertIncludes(demoHtml, `src="../assets/server-ui-${name}.webp"`, `docs/site/demo/index.html: optimized ${name} image`);
    assertIncludes(demoHtml, `srcset="${srcset}"`, `docs/site/demo/index.html: responsive ${name} srcset`);
    assertIncludes(demoHtml, `sizes="${demoImageSizes}"`, `docs/site/demo/index.html: responsive ${name} sizes`);
    for (const suffix of ['-720.webp', '-1440.webp', '.webp']) {
      const assetPath = resolve(repoRoot, `docs/site/assets/server-ui-${name}${suffix}`);
      if (!existsSync(assetPath)) fail(`${relative(repoRoot, assetPath)}: missing responsive demo image`);
      const maximumBytes = suffix === '-720.webp' ? 60_000 : suffix === '-1440.webp' ? 160_000 : 320_000;
      const bytes = statSync(assetPath).size;
      if (bytes > maximumBytes) {
        fail(`${relative(repoRoot, assetPath)}: responsive demo image is ${bytes} bytes; expected at most ${maximumBytes}`);
      }
    }
  }

  const asciinemaLicense = readFileSync(
    resolve(repoRoot, 'docs/site/demo/vendor/asciinema-player/LICENSE'),
    'utf8',
  );
  assertIncludes(asciinemaLicense, 'Apache License', 'vendored asciinema-player license');
  assertIncludes(asciinemaLicense, 'Version 2.0', 'vendored asciinema-player license version');
}

function assertRequestsImportNearPost(section, context) {
  const requestPosts = Array.from(section.matchAll(/requests\.post/g));
  if (requestPosts.length === 0) {
    fail(`${context}: missing requests.post GRPO submit call`);
  }

  for (const requestPost of requestPosts) {
    const nearbyPrefix = section.slice(Math.max(0, requestPost.index - 800), requestPost.index);
    if (!nearbyPrefix.includes('import requests')) {
      fail(`${context}: requests.post must have import requests nearby before use`);
    }
  }
}

function assertOpenAIClientSetupNearChatCreate(section, context) {
  const chatCreates = Array.from(section.matchAll(/client\.chat\.completions\.create/g));
  if (chatCreates.length === 0) {
    fail(`${context}: missing client.chat.completions.create generate call`);
  }

  for (const chatCreate of chatCreates) {
    const nearbyPrefix = section.slice(Math.max(0, chatCreate.index - 800), chatCreate.index);
    if (!nearbyPrefix.includes('import openai')) {
      fail(`${context}: client.chat.completions.create must have import openai nearby before use`);
    }
    if (!nearbyPrefix.includes('openai.OpenAI(base_url="http://localhost:8420/v1", api_key="unused")')) {
      fail(`${context}: client.chat.completions.create must set OpenAI base_url and api_key nearby before use`);
    }
  }
}

function extractMarkdownSection(markdown, heading) {
  const headingPattern = new RegExp(`^## ${heading.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*$`, 'm');
  const headingMatch = markdown.match(headingPattern);
  if (!headingMatch) return null;

  const sectionStart = headingMatch.index + headingMatch[0].length;
  const nextHeadingMatch = markdown.slice(sectionStart).match(/^##\s+/m);
  const sectionEnd = nextHeadingMatch ? sectionStart + nextHeadingMatch.index : markdown.length;
  return markdown.slice(sectionStart, sectionEnd);
}

function assertIncludes(source, needle, context) {
  if (!source.includes(needle)) {
    fail(`${context}: missing ${needle}`);
  }
}

function assertMatches(source, pattern, context) {
  if (!pattern.test(source)) {
    fail(`${context}: missing ${pattern}`);
  }
}

function extractRustRawStringConstant(source, constantName) {
  const escapedName = constantName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(`const\\s+${escapedName}\\s*:\\s*&str\\s*=\\s*r(#+)"([\\s\\S]*?)"\\1;`);
  const match = source.match(pattern);
  if (!match) {
    fail(`crates/kiln-server/src/cli.rs: missing raw string constant ${constantName}`);
  }
  return match[2];
}

function assertHelpCopyIncludes(helpCopy, constantName, term) {
  if (!helpCopy.includes(term)) {
    fail(`crates/kiln-server/src/cli.rs: ${constantName} missing ${term}`);
  }
}

function validateCliHelpOnboardingCopy() {
  const cliParser = readFileSync(resolve(repoRoot, 'crates/kiln-server/src/cli.rs'), 'utf8');
  const constants = new Map(
    [
      'TOP_LEVEL_OVERVIEW',
      'TOP_LEVEL_EXAMPLES',
      'SERVE_OVERVIEW',
      'SERVE_EXAMPLES',
      'HEALTH_OVERVIEW',
      'HEALTH_EXAMPLES',
      'TRAIN_OVERVIEW',
      'TRAIN_SFT_OVERVIEW',
      'TRAIN_GRPO_OVERVIEW',
      'TRAIN_OPD_OVERVIEW',
      'TRAIN_EXAMPLES',
      'ADAPTERS_EXAMPLES',
      'CONFIG_EXAMPLES',
    ]
      .map((constantName) => [constantName, extractRustRawStringConstant(cliParser, constantName)]),
  );

  const requiredTerms = new Map([
    ['TOP_LEVEL_OVERVIEW', [
      'Qwen3.5-4B',
      'live LoRA training',
      'no subcommand starts',
      'http://127.0.0.1:8420/ui',
      'kiln health',
      'kiln train sft',
      'kiln train grpo',
      'kiln train opd',
      'kiln adapters list',
    ]],
    ['TOP_LEVEL_EXAMPLES', [
      'kiln serve',
      'kiln health',
      'kiln train sft --file examples.jsonl --adapter my-task',
      'kiln train grpo --file grpo-batch.json --adapter my-task',
      'kiln train opd --file opd-request.json --adapter distilled-task --teacher teacher-v1',
      'kiln adapters list',
    ]],
    ['SERVE_OVERVIEW', [
      'Qwen3.5-4B',
      'KILN_MODEL_PATH',
      '--config',
      'http://127.0.0.1:8420/ui',
      'kiln health',
      'quickstart.html',
      'troubleshooting.html',
    ]],
    ['SERVE_EXAMPLES', [
      'KILN_MODEL_PATH',
      'kiln serve --config kiln.toml',
      'http://127.0.0.1:8420/ui',
      'kiln health',
      'troubleshooting.html',
    ]],
    ['HEALTH_OVERVIEW', [
      'kiln health',
      'http://localhost:8420',
      '/health',
      '--url',
      'quickstart.html',
      'troubleshooting.html',
    ]],
    ['HEALTH_EXAMPLES', [
      'kiln health',
      'kiln health --url http://localhost:8420',
      'kiln health --json',
      'curl http://localhost:8420/health',
      '/health',
      '--url',
      '--json',
      'Troubleshooting',
    ]],
    ['TRAIN_OVERVIEW', [
      'SFT reads JSONL',
      'messages array',
      'GRPO reads',
      'one JSON request/batch',
      'groups',
      'prompt messages',
      'candidate completions',
      'text',
      'completions',
      'reward scores',
      'http://127.0.0.1:8420/ui',
      'guided submission and status',
      'docs/GRPO_GUIDE.md',
      'reward-loop examples',
    ]],
    ['TRAIN_SFT_OVERVIEW', [
      'SFT JSONL',
      'one chat correction example per line',
      'messages array',
      'http://127.0.0.1:8420/ui',
      'training status',
    ]],
    ['TRAIN_GRPO_OVERVIEW', [
      'Train from GRPO',
      'JSON request/batch',
      'groups',
      'JSONL',
      'one group per line',
      'http://127.0.0.1:8420/ui',
      'training status',
      'docs/GRPO_GUIDE.md',
      'reward-loop examples',
    ]],
    ['TRAIN_EXAMPLES', [
      'SFT JSONL',
      'messages array',
      'GRPO JSON',
      'groups',
      'GRPO JSONL',
      'one group per line',
      'kiln train status',
      'kiln train status --job-id train_123',
    ]],
    ['ADAPTERS_EXAMPLES', [
      'kiln adapters unload',
      'revert the running server to the base model',
    ]],
    ['CONFIG_EXAMPLES', [
      'kiln config',
      'kiln config --file kiln.toml',
      'kiln serve --config kiln.toml',
    ]],
  ]);

  for (const [constantName, terms] of requiredTerms) {
    const helpCopy = constants.get(constantName);
    for (const term of terms) {
      assertHelpCopyIncludes(helpCopy, constantName, term);
    }
  }

  if (cliParser.includes('prompts/groups')) {
    fail('crates/kiln-server/src/cli.rs: kiln train grpo CLI help must describe scored groups, not prompts/groups; prompts belong to /v1/completions/batch');
  }

  assertMatches(
    cliParser,
    /Path to one GRPO JSON request\/batch, or JSONL with one group per line/,
    'crates/kiln-server/src/cli.rs: TrainCommands::Grpo file arg payload help',
  );

  assertMatches(
    cliParser,
    /pub enum Commands[\s\S]*?\n\s+#\[command\(long_about = SERVE_OVERVIEW, after_help = SERVE_EXAMPLES\)\]\n\s+Serve\s*\{/,
    'crates/kiln-server/src/cli.rs: Commands::Serve onboarding help wiring',
  );
  assertMatches(
    cliParser,
    /pub enum Commands[\s\S]*?\n\s+#\[command\(long_about = HEALTH_OVERVIEW, after_help = HEALTH_EXAMPLES\)\]\n\s+Health\s*\{/,
    'crates/kiln-server/src/cli.rs: Commands::Health onboarding help wiring',
  );
}

function validateQuickstartCliReference() {
  const quickstart = readFileSync(resolve(repoRoot, 'QUICKSTART.md'), 'utf8');
  const cliReference = extractMarkdownSection(quickstart, 'CLI Reference');
  if (!cliReference) {
    fail('QUICKSTART.md: missing ## CLI Reference section');
  }

  const cliReferenceCodeBlock = cliReference.match(/```(?:bash|sh)?\n([\s\S]*?)```/i)?.[1];
  if (!cliReferenceCodeBlock) {
    fail('QUICKSTART.md: CLI Reference section must include a fenced command block');
  }

  const expectedCommands = [
    'kiln serve --served-model-id <id>',
    'kiln health',
    'kiln health --json',
    'kiln config --file kiln.toml',
    'kiln config -f kiln.toml',
    'kiln train sft --file corrections.jsonl --adapter support-bot',
    'kiln train grpo --file grpo-batch.json --adapter support-bot',
    'kiln train opd --file opd-request.json --adapter distilled-bot --teacher qwen35@vllm',
    'kiln train status --job-id train_123',
    'kiln adapters list',
    'kiln adapters load support-bot',
    'kiln adapters unload',
    'kiln adapters delete support-bot',
  ];
  const missingCommands = expectedCommands.filter((command) => !cliReferenceCodeBlock.includes(command));
  if (missingCommands.length > 0) {
    fail(`QUICKSTART.md: CLI Reference command block missing commands: ${missingCommands.join(', ')}`);
  }

  const expectedTrainingPayloadTerms = [
    'SFT JSONL',
    'messages array',
    'GRPO JSON request/batch',
    'groups',
    'completions',
    'reward scores',
    'OPD request object',
    'registered teacher',
  ];
  const missingTrainingPayloadTerms = expectedTrainingPayloadTerms.filter((term) => !cliReferenceCodeBlock.includes(term));
  if (missingTrainingPayloadTerms.length > 0) {
    fail(`QUICKSTART.md: CLI Reference command block missing training payload cues: ${missingTrainingPayloadTerms.join(', ')}`);
  }

  const cliParser = readFileSync(resolve(repoRoot, 'crates/kiln-server/src/cli.rs'), 'utf8');
  const parserChecks = [
    ['Commands::Serve', /pub enum Commands[\s\S]*?\n\s+Serve\s*\{[\s\S]*?served_model_id:\s*Option<String>/],
    ['Commands::Health', /pub enum Commands[\s\S]*?\n\s+Health\s*\{[\s\S]*?url:\s*String[\s\S]*?json:\s*bool/],
    ['Commands::ConfigCheck', /pub enum Commands[\s\S]*?\n\s+ConfigCheck\s*\{[\s\S]*?file:\s*Option<String>/],
    ['Commands::Train', /pub enum Commands[\s\S]*?Train\(TrainCommands\)/],
    ['Commands::Adapters', /pub enum Commands[\s\S]*?Adapters\(AdapterCommands\)/],
    ['TrainCommands::Sft', /pub enum TrainCommands[\s\S]*?\n\s+Sft\s*\{[\s\S]*?file:\s*String[\s\S]*?adapter:\s*String[\s\S]*?url:\s*String/],
    ['TrainCommands::Grpo', /pub enum TrainCommands[\s\S]*?\n\s+Grpo\s*\{[\s\S]*?file:\s*String[\s\S]*?adapter:\s*String[\s\S]*?url:\s*String/],
    ['TrainCommands::Opd', /pub enum TrainCommands[\s\S]*?\n\s+Opd\s*\{[\s\S]*?file:\s*String[\s\S]*?adapter:\s*String[\s\S]*?teacher:\s*Option<String>[\s\S]*?url:\s*String/],
    ['TrainCommands::Status', /pub enum TrainCommands[\s\S]*?\n\s+Status\s*\{[\s\S]*?job_id:\s*Option<String>[\s\S]*?url:\s*String/],
    ['AdapterCommands::List', /pub enum AdapterCommands[\s\S]*?\n\s+List\s*\{[\s\S]*?url:\s*String/],
    ['AdapterCommands::Load', /pub enum AdapterCommands[\s\S]*?\n\s+Load\s*\{[\s\S]*?name:\s*String[\s\S]*?url:\s*String/],
    ['AdapterCommands::Unload', /pub enum AdapterCommands[\s\S]*?\n\s+Unload\s*\{[\s\S]*?name:\s*Option<String>[\s\S]*?url:\s*String/],
    ['AdapterCommands::Delete', /pub enum AdapterCommands[\s\S]*?\n\s+Delete\s*\{[\s\S]*?name:\s*String[\s\S]*?url:\s*String/],
  ];
  for (const [label, pattern] of parserChecks) {
    assertMatches(cliParser, pattern, `crates/kiln-server/src/cli.rs: ${label}`);
  }

  const expectedArgs = ['served_model_id', 'json', 'file', 'job_id', 'adapter', 'url'];
  for (const arg of expectedArgs) {
    assertIncludes(cliParser, arg, 'crates/kiln-server/src/cli.rs');
  }
}

function validateLaunchSentinel() {
  const launchDir = resolve(repoRoot, 'docs/site/launch');
  const sentinelPath = resolve(launchDir, 'README.md');

  if (!existsSync(sentinelPath)) {
    fail('docs/site/launch/README.md: missing no-publicity sentinel');
  }

  const entries = readdirSync(launchDir, { withFileTypes: true });
  const unexpectedEntries = entries
    .filter((entry) => entry.name !== 'README.md')
    .map((entry) => `${entry.name}${entry.isDirectory() ? '/' : ''}`)
    .sort();
  if (unexpectedEntries.length > 0) {
    fail(`docs/site/launch/: unexpected draft/content files: ${unexpectedEntries.join(', ')}`);
  }

  const sentinel = readFileSync(sentinelPath, 'utf8').toLowerCase().replace(/\s+/g, ' ');
  const requiredPhrases = [
    'publicity draft sentinel',
    'intentionally does not contain external launch, announcement',
    'agents must not recreate publicity materials',
    'eric handles publicity himself',
    'keep phase 11 work limited to internal onboarding',
  ];
  const missingPhrases = requiredPhrases.filter((phrase) => !sentinel.includes(phrase));
  if (missingPhrases.length > 0) {
    fail(`docs/site/launch/README.md: missing no-publicity sentinel wording: ${missingPhrases.join(', ')}`);
  }
}

function expectedLocalHref(localPath) {
  const href = pathToFileURL(resolve(repoRoot, localPath)).href;
  return localPath.endsWith('/') && !href.endsWith('/') ? `${href}/` : href;
}

function hasKnownExternalScheme(href) {
  return /^(?:https?:|mailto:)/i.test(href);
}

function isServerRoute(href) {
  return /^\/(?:ui(?:[/?#]|$)|health(?:[/?#]|$)|metrics(?:[/?#]|$)|v1(?:[/?#\/]|$))/.test(href);
}

function isIgnoredHref(href) {
  return href === ''
    || href.includes('${')
    || /^javascript:/i.test(href)
    || hasKnownExternalScheme(href)
    || isServerRoute(href);
}

function decodeHtmlAttribute(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/&mdash;/g, '—')
    .replace(/&ndash;/g, '–');
}

function hrefPathOnly(href) {
  return href.split('#')[0].split('?')[0];
}

function hrefFragment(href) {
  const fragmentIndex = href.indexOf('#');
  if (fragmentIndex === -1) return '';

  const fragment = href.slice(fragmentIndex + 1);
  try {
    return decodeURIComponent(fragment);
  } catch {
    return fragment;
  }
}

function docsSiteHtmlPaths() {
  const htmlPaths = [];

  function visit(directoryPath) {
    for (const entry of readdirSync(directoryPath).sort()) {
      const entryPath = resolve(directoryPath, entry);
      const entryStat = statSync(entryPath);
      if (entryStat.isDirectory()) {
        visit(entryPath);
      } else if (entryStat.isFile() && entry.endsWith('.html')) {
        htmlPaths.push(relative(repoRoot, entryPath).split(sep).join('/'));
      }
    }
  }

  visit(siteRoot);
  return htmlPaths;
}

function htmlIdsForPath(htmlPath, idCache) {
  if (!idCache.has(htmlPath)) {
    const html = readFileSync(resolve(repoRoot, htmlPath), 'utf8');
    const ids = new Set();
    const idMatches = html.matchAll(/\bid\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/gi);
    for (const match of idMatches) {
      ids.add(decodeHtmlAttribute(match[1] ?? match[2] ?? match[3] ?? ''));
    }
    idCache.set(htmlPath, ids);
  }

  return idCache.get(htmlPath);
}

function resolveLocalHref(sourceHtmlPath, href) {
  const sourceDir = dirname(resolve(repoRoot, sourceHtmlPath));
  const hrefPath = hrefPathOnly(href);
  if (hrefPath === '') return resolve(repoRoot, sourceHtmlPath);

  const resolvedPath = hrefPath.startsWith('/')
    ? resolve(siteRoot, `.${hrefPath}`)
    : resolve(sourceDir, hrefPath);

  if (hrefPath.endsWith('/')) {
    return resolve(resolvedPath, 'index.html');
  }

  if (existsSync(resolvedPath) && statSync(resolvedPath).isDirectory()) {
    return resolve(resolvedPath, 'index.html');
  }

  return resolvedPath;
}

function expectedCanonicalHref(localPath) {
  const siteRelativePath = relative(siteRoot, resolve(repoRoot, localPath)).split(sep).join('/');
  if (siteRelativePath === 'index.html') return 'https://ericflo.github.io/kiln/';
  if (siteRelativePath === 'demo/index.html') return 'https://ericflo.github.io/kiln/demo/';
  return `https://ericflo.github.io/kiln/${siteRelativePath}`;
}

function validateDocsSiteCanonicalLinks() {
  for (const sitePage of pages) {
    const pagePath = resolve(repoRoot, sitePage.path);
    const html = readFileSync(pagePath, 'utf8');
    const canonicalMatches = Array.from(html.matchAll(/<link\b[^>]*\brel\s*=\s*(?:"canonical"|'canonical'|canonical)[^>]*>/gi));

    if (canonicalMatches.length !== 1) {
      fail(`${sitePage.path}: expected exactly one canonical link, found ${canonicalMatches.length}`);
    }

    const canonicalTag = canonicalMatches[0][0];
    const hrefMatch = canonicalTag.match(/\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/i);
    const href = decodeHtmlAttribute(hrefMatch?.[1] ?? hrefMatch?.[2] ?? hrefMatch?.[3] ?? '').trim();
    const expectedHref = expectedCanonicalHref(sitePage.path);
    if (href !== expectedHref) {
      fail(`${sitePage.path}: canonical href must be ${expectedHref}, found ${href || '(missing)'}`);
    }
  }
}

function validateDocsSiteLocalLinks() {
  const idCache = new Map();

  for (const sourcePath of docsSiteHtmlPaths()) {
    const pagePath = resolve(repoRoot, sourcePath);
    const html = readFileSync(pagePath, 'utf8');
    const hrefMatches = html.matchAll(/\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/gi);

    for (const match of hrefMatches) {
      const href = decodeHtmlAttribute(match[1] ?? match[2] ?? match[3] ?? '').trim();
      if (isIgnoredHref(href)) continue;

      const targetPath = resolveLocalHref(sourcePath, href);
      if (!existsSync(targetPath) || statSync(targetPath).isDirectory()) {
        fail(`${sourcePath}: broken local href ${href} (resolved target: ${relative(repoRoot, targetPath)})`);
      }

      const fragment = hrefFragment(href);
      if (fragment) {
        const targetRelativePath = relative(repoRoot, targetPath).split(sep).join('/');
        if (!htmlIdsForPath(targetRelativePath, idCache).has(fragment)) {
          fail(`${sourcePath}: broken local href anchor ${href} (resolved target: ${targetRelativePath}, missing fragment: #${fragment})`);
        }
      }
    }
  }
}


function markdownLocalLinkSourcePaths() {
  const docsDir = resolve(repoRoot, 'docs');
  const topLevelDocs = readdirSync(docsDir)
    .filter((entry) => entry.endsWith('.md') && !/^archive(?:\.|$|-)/i.test(entry))
    .map((entry) => join('docs', entry))
    .sort();

  return ['README.md', 'QUICKSTART.md', ...topLevelDocs];
}

function stripMarkdownCode(markdown) {
  return markdown
    .replace(/^```[\s\S]*?^```/gm, (block) => '\n'.repeat(block.split('\n').length - 1))
    .replace(/^~~~[\s\S]*?^~~~/gm, (block) => '\n'.repeat(block.split('\n').length - 1))
    .replace(/`[^`\n]*(?:`|$)/g, '');
}

function normalizeMarkdownLinkText(value) {
  return value
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/<[^>]+>/g, '')
    .replace(/[`*_~\[\]]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function githubHeadingSlug(headingText) {
  return normalizeMarkdownLinkText(headingText)
    .toLowerCase()
    .replace(/&(?:amp|lt|gt|quot|#39);/g, (entity) => ({
      '&amp;': '&',
      '&lt;': '<',
      '&gt;': '>',
      '&quot;': '',
      '&#39;': '',
    })[entity] ?? '')
    .replace(/[^\p{Letter}\p{Number}\s_-]/gu, '')
    .trim()
    .replace(/\s+/g, '-');
}

function markdownHeadingAnchors(markdown) {
  const anchors = new Set();
  const seen = new Map();
  const headingMatches = stripMarkdownCode(markdown).matchAll(/^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/gm);

  for (const match of headingMatches) {
    const baseSlug = githubHeadingSlug(match[2]);
    if (!baseSlug) continue;
    const duplicateCount = seen.get(baseSlug) ?? 0;
    seen.set(baseSlug, duplicateCount + 1);
    anchors.add(duplicateCount === 0 ? baseSlug : `${baseSlug}-${duplicateCount}`);
  }

  return anchors;
}

function splitMarkdownTarget(rawTarget) {
  const trimmed = rawTarget.trim();
  if (trimmed.startsWith('<')) {
    const closingBracket = trimmed.indexOf('>');
    return closingBracket === -1 ? trimmed : trimmed.slice(1, closingBracket);
  }

  const match = trimmed.match(/^(?:\\.|[^\s"'])+/);
  return match ? match[0].replace(/\\([()])/g, '$1') : trimmed;
}

function extractMarkdownLocalTargets(markdown) {
  const source = stripMarkdownCode(markdown);
  const targets = [];
  const inlineLinkPattern = /(!?)\[([^\]\n]*(?:\][^\[\]\n]*)*)\]\(\s*([^\n)]*(?:\([^\n)]*\)[^\n)]*)*)\)/g;
  const referenceDefinitionPattern = /^ {0,3}\[([^\]\n]+)\]:\s*(\S[^\n]*)$/gm;
  const htmlAttrPattern = /\b(?:href|src)\s*=\s*(?:"([^"]*)"|'([^']*)')/gi;

  for (const match of source.matchAll(inlineLinkPattern)) {
    targets.push({
      text: match[2].replace(/\s+/g, ' ').trim() || (match[1] ? 'image' : 'link'),
      target: splitMarkdownTarget(match[3]),
    });
  }

  for (const match of source.matchAll(referenceDefinitionPattern)) {
    targets.push({
      text: `[${match[1]}]`,
      target: splitMarkdownTarget(match[2]),
    });
  }

  for (const match of source.matchAll(htmlAttrPattern)) {
    targets.push({
      text: '<html attribute>',
      target: decodeHtmlAttribute(match[1] ?? match[2] ?? ''),
    });
  }

  return targets;
}

function isDynamicMarkdownTarget(target) {
  return target.includes('${')
    || target.includes('{{')
    || target.includes('}}')
    || target.includes('<')
    || target.includes('>')
    || target.includes('*')
    || target.includes('…')
    || target.includes('...');
}

function isIgnoredMarkdownTarget(target) {
  const trimmed = target.trim();
  return trimmed === ''
    || hasKnownExternalScheme(trimmed)
    || /^javascript:/i.test(trimmed)
    || /^data:/i.test(trimmed)
    || isDynamicMarkdownTarget(trimmed);
}

function markdownTargetParts(target) {
  const [pathAndQuery, rawAnchor = ''] = target.split('#');
  const pathPart = pathAndQuery.split('?')[0];
  const anchor = rawAnchor.split('?')[0];
  return { pathPart, anchor };
}

function decodeLocalPath(pathPart) {
  try {
    return decodeURIComponent(pathPart);
  } catch {
    return pathPart;
  }
}

function resolveMarkdownTargetPath(sourceMarkdownPath, pathPart) {
  const decodedPath = decodeLocalPath(pathPart);
  if (decodedPath === '') return resolve(repoRoot, sourceMarkdownPath);
  const sourceDir = dirname(resolve(repoRoot, sourceMarkdownPath));
  return decodedPath.startsWith('/')
    ? resolve(repoRoot, `.${decodedPath}`)
    : resolve(sourceDir, decodedPath);
}

function directoryHasMarkdownIndex(directoryPath) {
  return ['index.html', 'README.md', 'Readme.md', 'readme.md']
    .some((entry) => existsSync(resolve(directoryPath, entry)));
}

function validateMarkdownTargetFile(sourcePath, link, resolvedPath, pathPart) {
  if (!existsSync(resolvedPath)) {
    fail(`${sourcePath}: broken local Markdown link "${link.text}" -> ${link.target} (resolved path: ${relative(repoRoot, resolvedPath)})`);
  }

  const targetStat = statSync(resolvedPath);
  if (targetStat.isDirectory()) {
    if (!directoryHasMarkdownIndex(resolvedPath)) {
      fail(`${sourcePath}: directory Markdown link "${link.text}" -> ${link.target} must contain index.html or README.md (resolved path: ${relative(repoRoot, resolvedPath)})`);
    }
    return;
  }

  if (pathPart.endsWith('/')) {
    fail(`${sourcePath}: directory-style Markdown link "${link.text}" -> ${link.target} resolved to a file (${relative(repoRoot, resolvedPath)})`);
  }
}

function anchorLooksLikeIssueOrPrShorthand(anchor) {
  return /^\d+$/.test(anchor);
}

function safeDecodeAnchor(anchor) {
  try {
    return decodeURIComponent(anchor).toLowerCase();
  } catch {
    return anchor.toLowerCase();
  }
}

function validateMarkdownLocalLinks() {
  const anchorCache = new Map();
  const markdownPaths = markdownLocalLinkSourcePaths();

  function anchorsFor(relativePath) {
    if (!anchorCache.has(relativePath)) {
      anchorCache.set(relativePath, markdownHeadingAnchors(readFileSync(resolve(repoRoot, relativePath), 'utf8')));
    }
    return anchorCache.get(relativePath);
  }

  for (const sourcePath of markdownPaths) {
    const markdown = readFileSync(resolve(repoRoot, sourcePath), 'utf8');
    for (const link of extractMarkdownLocalTargets(markdown)) {
      const target = link.target.trim();
      if (isIgnoredMarkdownTarget(target)) continue;

      const { pathPart, anchor } = markdownTargetParts(target);
      if (pathPart === '' && anchorLooksLikeIssueOrPrShorthand(anchor)) continue;

      const resolvedPath = resolveMarkdownTargetPath(sourcePath, pathPart);
      validateMarkdownTargetFile(sourcePath, link, resolvedPath, pathPart);

      if (anchor) {
        if (statSync(resolvedPath).isDirectory()) continue;
        if (extname(resolvedPath).toLowerCase() !== '.md') continue;

        const targetRelativePath = relative(repoRoot, resolvedPath).split(sep).join('/');
        const normalizedAnchor = safeDecodeAnchor(anchor);
        if (!anchorsFor(targetRelativePath).has(normalizedAnchor)) {
          fail(`${sourcePath}: broken Markdown anchor "${link.text}" -> ${link.target} (missing anchor #${normalizedAnchor} in ${targetRelativePath})`);
        }
      }
    }
  }
}

function validateDemoCasts(sitePagePath, referencedCasts) {
  const demoDir = resolve(repoRoot, dirname(sitePagePath));
  const uniqueCasts = [...new Set(referencedCasts)];
  const missingExpected = expectedDemoCastFiles.filter((cast) => !uniqueCasts.includes(cast));

  if (missingExpected.length > 0) {
    fail(`${sitePagePath}: missing expected demo cast references: ${missingExpected.join(', ')}`);
  }

  for (const cast of uniqueCasts) {
    const castPath = resolve(demoDir, cast);
    const castRelativePath = relative(demoDir, castPath);
    if (castRelativePath.startsWith('..') || castRelativePath.includes(`..${sep}`)) {
      fail(`${sitePagePath}: demo cast escapes docs/site/demo/: ${cast}`);
    }
    if (!existsSync(castPath)) {
      fail(`${sitePagePath}: referenced demo cast does not exist: ${cast}`);
    }
  }

  for (const cast of expectedDemoCastFiles) {
    validateDemoCastContent(demoDir, cast);
  }
}

function parseDemoCastJsonLine(castPath, line, lineNumber) {
  try {
    return JSON.parse(line);
  } catch (error) {
    fail(`${castPath}: line ${lineNumber} is not valid JSON: ${error.message}`);
  }
}

function validateDemoCastContent(demoDir, cast) {
  const castPath = resolve(demoDir, cast);
  const castDisplayPath = relative(repoRoot, castPath).split(sep).join('/');
  const lines = readFileSync(castPath, 'utf8').trimEnd().split(/\r?\n/);

  if (lines.length === 0 || lines[0].trim() === '') {
    fail(`${castDisplayPath}: missing asciinema v2 header row`);
  }

  const header = parseDemoCastJsonLine(castDisplayPath, lines[0], 1);
  if (!header || Array.isArray(header) || typeof header !== 'object') {
    fail(`${castDisplayPath}: header row must be a JSON object`);
  }
  if (header.version !== 2) {
    fail(`${castDisplayPath}: header version must be 2, got ${JSON.stringify(header.version)}`);
  }
  if (header.width !== 120) {
    fail(`${castDisplayPath}: header width must be 120, got ${JSON.stringify(header.width)}`);
  }
  if (header.height !== 32) {
    fail(`${castDisplayPath}: header height must be 32, got ${JSON.stringify(header.height)}`);
  }
  if (typeof header.title !== 'string' || header.title.trim() === '') {
    fail(`${castDisplayPath}: header title must be a non-empty string`);
  }

  const eventLines = lines.slice(1).filter((line) => line.trim() !== '');
  if (eventLines.length === 0) {
    fail(`${castDisplayPath}: missing asciinema JSONL event rows`);
  }

  let hasOutputEvent = false;
  for (const [eventIndex, eventLine] of eventLines.entries()) {
    const lineNumber = eventIndex + 2;
    const event = parseDemoCastJsonLine(castDisplayPath, eventLine, lineNumber);
    if (!Array.isArray(event) || typeof event[0] !== 'number' || typeof event[1] !== 'string' || typeof event[2] !== 'string') {
      fail(`${castDisplayPath}: line ${lineNumber} must be an event row shaped like [number, string, string]`);
    }
    if (event[1] === 'o') {
      hasOutputEvent = true;
    }
  }

  if (!hasOutputEvent) {
    fail(`${castDisplayPath}: missing output event shaped like [number, "o", string]`);
  }
}

function validateDemoReadmeInventory() {
  const readmePath = resolve(repoRoot, 'docs/site/demo/README.md');
  if (!existsSync(readmePath)) {
    fail('docs/site/demo/README.md: missing demo cast inventory');
  }

  const readme = readFileSync(readmePath, 'utf8');
  const missingCasts = expectedDemoCastFiles.filter((cast) => !readme.includes(cast));
  if (missingCasts.length > 0) {
    fail(`docs/site/demo/README.md: missing demo cast inventory entries: ${missingCasts.join(', ')}`);
  }

  const missingDrivers = expectedDemoCastFiles
    .map((cast) => expectedDemoReadmeDrivers.get(cast))
    .filter((driver) => driver && !readme.includes(driver));
  if (missingDrivers.length > 0) {
    fail(`docs/site/demo/README.md: missing demo driver inventory entries: ${missingDrivers.join(', ')}`);
  }

  const missingLinks = expectedDemoReadmeLinks.filter((link) => !readme.includes(link));
  if (missingLinks.length > 0) {
    fail(`docs/site/demo/README.md: missing expected cross-links: ${missingLinks.join(', ')}`);
  }
}

function validatePrimaryDocsEditorialBudget() {
  const manifestPath = resolve(repoRoot, 'docs/site/docs-manifest.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const sectionNavigation = new Map(
    manifest.sections.map((section) => [section.id, section.navigation || 'primary']),
  );
  const expectedCuratedSources = new Map([
    ['overview', 'docs/public/OVERVIEW.md'],
    ['quickstart-reference', 'docs/public/QUICKSTART.md'],
    ['configuration', 'docs/public/CONFIGURATION.md'],
    ['architecture', 'docs/public/ARCHITECTURE.md'],
    ['benchmarks', 'docs/public/BENCHMARKS.md'],
  ]);

  for (const [slug, expectedSource] of expectedCuratedSources) {
    const document = manifest.documents.find((candidate) => candidate.slug === slug);
    if (document?.source !== expectedSource) {
      fail(`docs/site/docs-manifest.json: ${slug} must use curated source ${expectedSource}`);
    }
  }

  for (const document of manifest.documents) {
    const navigation = document.navigation || sectionNavigation.get(document.section) || 'primary';
    const kind = document.kind || 'markdown';
    if (navigation !== 'primary' || kind !== 'markdown') continue;

    const source = readFileSync(resolve(repoRoot, document.source), 'utf8');
    const wordCount = source.trim() ? source.trim().split(/\s+/).length : 0;
    const lineCount = source.split(/\r?\n/).length;
    if (wordCount > 1500 || lineCount > 250) {
      fail(
        `${document.source}: primary documentation exceeds the editorial budget ` +
        `(${wordCount} words, ${lineCount} lines; maximum 1500 words and 250 lines); ` +
        'move exhaustive material to the reference library',
      );
    }
  }
}

function validateGeneratedDocsArtifacts() {
  const requiredPaths = [
    ...generatedDocsPages.map((page) => page.path),
    publishedPath('docs/search-index.json'),
    publishedPath('css/docs.css'),
    publishedPath('js/docs.js'),
    publishedPath('sitemap.xml'),
    publishedPath('llms.txt'),
  ];
  const missing = requiredPaths.filter((path) => !existsSync(resolve(repoRoot, path)));
  if (missing.length > 0) {
    if (!generatedDocsRequired) return false;
    fail(`generated documentation artifact is incomplete: ${missing.join(', ')}`);
  }

  const routeForStaticPage = (sitePage) => {
    const outputRelative = relative(siteRoot, resolve(repoRoot, sitePage.path)).split(sep).join('/');
    if (outputRelative === 'index.html') return '/';
    if (outputRelative.endsWith('/index.html')) return `/${outputRelative.slice(0, -'index.html'.length)}`;
    return `/${outputRelative}`;
  };
  const expectedChecklistRoutes = [
    ...pages.map(routeForStaticPage),
    '/docs/',
    ...docsManifest.documents.map((document) => `/docs/${document.slug}/`),
  ].sort();
  const auditPlanPath = resolve(repoRoot, 'docs/plans/public-site-audit-and-copyediting-plan.md');
  const auditPlan = readFileSync(auditPlanPath, 'utf8');
  const checklistRoutes = Array.from(
    auditPlan.matchAll(/^\| `([^`]+)` — /gm),
    (match) => match[1],
  ).sort();
  if (JSON.stringify(checklistRoutes) !== JSON.stringify(expectedChecklistRoutes)) {
    const missingRoutes = expectedChecklistRoutes.filter((route) => !checklistRoutes.includes(route));
    const staleRoutes = checklistRoutes.filter((route) => !expectedChecklistRoutes.includes(route));
    fail(`public-site audit checklist drifted; missing: ${missingRoutes.join(', ') || 'none'}; stale: ${staleRoutes.join(', ') || 'none'}`);
  }
  if (!auditPlan.includes(`Scope: ${expectedChecklistRoutes.length} public HTML routes`)) {
    fail(`public-site audit plan must declare the current ${expectedChecklistRoutes.length}-route scope`);
  }

  const publishedHtmlPaths = [
    ...pages.map((page) => resolve(repoRoot, page.path)),
    resolve(siteRoot, 'docs/index.html'),
    ...docsManifest.documents.map((document) => (
      resolve(siteRoot, 'docs', document.slug, 'index.html')
    )),
  ];
  for (const [label, values] of [
    ['title', publishedHtmlPaths.map((path) => {
      const html = readFileSync(path, 'utf8');
      return decodeHtmlAttribute(html.match(/<title>([\s\S]*?)<\/title>/i)?.[1] ?? '')
        .replace(/<[^>]+>/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    })],
    ['description', publishedHtmlPaths.map((path) => (
      metaContent(readFileSync(path, 'utf8'), 'name', 'description')
    ))],
  ]) {
    const counts = new Map();
    for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
    const emptyCount = values.filter((value) => !value).length;
    const duplicates = [...counts]
      .filter(([value, count]) => value && count > 1)
      .map(([value]) => value);
    if (emptyCount > 0 || duplicates.length > 0) {
      fail(`published HTML ${label} values must be non-empty and unique; empty: ${emptyCount}; duplicates: ${duplicates.join(' | ') || 'none'}`);
    }
  }

  let searchIndex;
  try {
    searchIndex = JSON.parse(readFileSync(resolve(siteRoot, 'docs/search-index.json'), 'utf8'));
  } catch (error) {
    fail(`generated documentation search index is invalid JSON: ${error.message}`);
  }
  if (!Array.isArray(searchIndex) || searchIndex.length === 0) {
    fail('generated documentation search index must contain published documents');
  }
  const expectedSearchCount = docsManifest.site.product_guides.length + docsManifest.documents.length;
  if (searchIndex.length !== expectedSearchCount) {
    fail(`generated documentation search index must contain exactly ${expectedSearchCount} published entries, got ${searchIndex.length}`);
  }
  const expectedReferenceSlugs = docsManifest.documents.map((document) => document.slug).sort();
  const actualReferenceSlugs = searchIndex
    .filter((entry) => entry?.kind === 'reference')
    .map((entry) => entry.slug)
    .sort();
  if (JSON.stringify(actualReferenceSlugs) !== JSON.stringify(expectedReferenceSlugs)) {
    fail('generated documentation search index contains a missing or unpublished reference');
  }
  const expectedProductGuideOrder = [
    'Quickstart',
    'GRPO Guide',
    'OpenEnv Training',
    'Evals Guide',
    'API Reference',
    'CLI Reference',
    'Architecture',
    'Troubleshooting',
    'Demo',
  ];
  const productGuideOrder = searchIndex
    .filter((entry) => entry?.kind === 'product_guide')
    .map((entry) => entry.title);
  if (JSON.stringify(productGuideOrder) !== JSON.stringify(expectedProductGuideOrder)) {
    fail(`generated documentation search index has the wrong product-guide order: ${productGuideOrder.join(', ')}`);
  }
  for (const expected of generatedDocsPages.filter((page) => page.label !== 'Documentation hub')) {
    const slug = expected.path.split('/').filter(Boolean).at(-2);
    const entry = searchIndex.find((candidate) => candidate?.slug === slug);
    const manifestDocument = docsManifest.documents.find((candidate) => candidate.slug === slug);
    if (!entry || !manifestDocument || entry.title !== manifestDocument.title) {
      fail(`generated documentation search index is missing the manifest title for ${expected.label}`);
    }
  }

  const expectedTrainingOrder = [
    'native-sft-profile',
    'sft-ingestion',
    'sft-tokenization',
    'training-checkpoints',
    'grpo',
    'echo',
    'dataset-splits',
    'evals',
  ];
  const trainingSearchOrder = searchIndex
    .filter((entry) => entry?.kind === 'reference' && expectedTrainingOrder.includes(entry.slug))
    .map((entry) => entry.slug);
  if (JSON.stringify(trainingSearchOrder) !== JSON.stringify(expectedTrainingOrder)) {
    fail(`generated documentation search index has the wrong training reference order: ${trainingSearchOrder.join(', ')}`);
  }

  for (const expected of generatedDocsPages) {
    const htmlPath = resolve(repoRoot, expected.path);
    const html = readFileSync(htmlPath, 'utf8');
    validateSocialMetadataForHtml(expected.path, html, expected.canonical);
    const missingTerms = missingNormalizedTerms(normalizedHtmlText(html), expected.terms);
    if (missingTerms.length > 0) {
      fail(`${expected.path}: generated documentation content missing terms: ${missingTerms.join(', ')}`);
    }
    const ids = new Set(Array.from(html.matchAll(/\bid\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s"'=<>`]+))/gi))
      .map((match) => decodeHtmlAttribute(match[1] ?? match[2] ?? match[3] ?? '')));
    const missingAnchors = (expected.anchors || []).filter((anchor) => !ids.has(anchor));
    if (missingAnchors.length > 0) {
      fail(`${expected.path}: generated documentation content missing anchors: ${missingAnchors.join(', ')}`);
    }
  }

  const sitemap = readFileSync(resolve(siteRoot, 'sitemap.xml'), 'utf8');
  const sitemapUrls = Array.from(sitemap.matchAll(/<loc>([^<]+)<\/loc>/g), (match) => match[1]);
  const expectedSitemapUrls = [
    ...pages.map((page) => expectedCanonicalHref(page.path)),
    `${docsManifest.site.base_url}/docs/`,
    ...docsManifest.documents.map((document) => `${docsManifest.site.base_url}/docs/${document.slug}/`),
  ].sort();
  const uniqueSitemapUrls = [...new Set(sitemapUrls)].sort();
  if (sitemapUrls.length !== uniqueSitemapUrls.length) {
    fail('generated sitemap contains a canonical URL more than once');
  }
  if (JSON.stringify(uniqueSitemapUrls) !== JSON.stringify(expectedSitemapUrls)) {
    const missing = expectedSitemapUrls.filter((url) => !uniqueSitemapUrls.includes(url));
    const unexpected = uniqueSitemapUrls.filter((url) => !expectedSitemapUrls.includes(url));
    fail(`generated sitemap must contain exactly every canonical HTML route; missing: ${missing.join(', ') || 'none'}; unexpected: ${unexpected.join(', ') || 'none'}`);
  }
  if (sitemapUrls.some((url) => /(?:^|\/)404(?:\.html|\/|$)/.test(url))) {
    fail('generated sitemap must not publish the custom not-found page');
  }

  const robots = readFileSync(resolve(siteRoot, 'robots.txt'), 'utf8');
  for (const term of [
    'User-agent: *',
    'Allow: /',
    'Sitemap: https://ericflo.github.io/kiln/sitemap.xml',
  ]) {
    if (!robots.includes(term)) fail(`robots.txt is missing ${term}`);
  }

  const llms = readFileSync(resolve(siteRoot, 'llms.txt'), 'utf8');
  const requiredLlmsTerms = [
    '# Kiln',
    '> Kiln is a pure-Rust, single-GPU server for Qwen3.5-4B',
    'CUDA, ROCm, Metal, and Vulkan builds are available',
    'reported capabilities, not device names or machine allowlists',
    'performance claims are bounded by the published benchmark receipts',
    '## Product guides',
    '[Quickstart](https://ericflo.github.io/kiln/quickstart.html)',
    '[GRPO Guide](https://ericflo.github.io/kiln/grpo.html)',
    '[Evals Guide](https://ericflo.github.io/kiln/evals.html)',
    '## Core documentation',
    'https://raw.githubusercontent.com/ericflo/kiln/refs/heads/main/docs/public/OVERVIEW.md',
    'https://raw.githubusercontent.com/ericflo/kiln/refs/heads/main/docs/GRPO_GUIDE.md',
    'https://raw.githubusercontent.com/ericflo/kiln/refs/heads/main/docs/EVAL_GUIDE.md',
    '## Machine-readable contracts',
    'https://raw.githubusercontent.com/ericflo/kiln/refs/heads/main/contracts/kiln-http-api-v1.openapi.json',
    '## Optional',
    '[Documentation search index](https://ericflo.github.io/kiln/docs/search-index.json)',
  ];
  const missingLlmsTerms = requiredLlmsTerms.filter((term) => !llms.includes(term));
  if (missingLlmsTerms.length > 0) {
    fail(`generated llms.txt is missing agent-discovery terms: ${missingLlmsTerms.join(', ')}`);
  }
  return true;
}

async function loadPuppeteer() {
  const packageJson = resolve(repoRoot, 'scripts/docs-site/package.json');
  const require = createRequire(packageJson);
  try {
    const module = require('puppeteer-core');
    return module.default || module;
  } catch (error) {
    if (error?.code !== 'MODULE_NOT_FOUND') throw error;
    fail('Missing pinned puppeteer-core dependency; run npm ci --prefix scripts/docs-site.');
  }
}

function chromiumPath() {
  const path = process.env.CHROME_BIN
    || process.env.PUPPETEER_EXECUTABLE_PATH
    || process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  if (!path) {
    fail('Set CHROME_BIN, PUPPETEER_EXECUTABLE_PATH, or PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH to an installed Chromium/Chrome binary.');
  }
  return path;
}

// Chromium launch flags tuned for headless CI containers.
//
// `--disable-dev-shm-usage` is the canonical fix for the
// "Timed out after N ms while waiting for the WS endpoint URL to appear in
// stdout!" failure: GitHub Actions runners give /dev/shm only ~64 MiB, the
// Chromium renderer fills it during early init, the renderer dies before
// printing the DevTools WebSocket URL, and Puppeteer's launch then times
// out. Routing shared memory through /tmp avoids the cliff entirely.
//
// The other flags shave several seconds off cold start (no GPU init, no
// extension load, no first-run prompts, no zygote prefork) and remove
// failure modes that don't matter for a headless DOM scrape.
const CHROMIUM_CI_FLAGS = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--disable-dev-shm-usage',
  '--disable-gpu',
  '--no-zygote',
  '--disable-extensions',
  '--disable-background-networking',
  '--disable-default-apps',
  '--no-first-run',
  '--no-default-browser-check',
  '--mute-audio',
  '--hide-scrollbars',
];

const CHROMIUM_LAUNCH_TIMEOUT_MS = 90_000;
const CHROMIUM_LAUNCH_ATTEMPTS = 3;

async function launchChromiumWithRetry(puppeteer) {
  let lastErr;
  for (let attempt = 1; attempt <= CHROMIUM_LAUNCH_ATTEMPTS; attempt++) {
    try {
      return await puppeteer.launch({
        executablePath: chromiumPath(),
        headless: true,
        args: CHROMIUM_CI_FLAGS,
        timeout: CHROMIUM_LAUNCH_TIMEOUT_MS,
        protocolTimeout: CHROMIUM_LAUNCH_TIMEOUT_MS,
        dumpio: attempt > 1,
      });
    } catch (err) {
      lastErr = err;
      console.warn(`Chromium launch attempt ${attempt}/${CHROMIUM_LAUNCH_ATTEMPTS} failed: ${err && err.message ? err.message : err}`);
      if (attempt < CHROMIUM_LAUNCH_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, 2000 * attempt));
      }
    }
  }
  fail(`Chromium failed to launch after ${CHROMIUM_LAUNCH_ATTEMPTS} attempts: ${lastErr && lastErr.message ? lastErr.message : lastErr}`);
}

function formatLikelyOverflowingElements(elements) {
  if (!elements || elements.length === 0) return 'none found';
  return elements
    .map((element) => `${element.selector} rect=[${element.left},${element.right}] scrollWidth=${element.scrollWidth} clientWidth=${element.clientWidth} text="${element.text}"`)
    .join('; ');
}

async function validateGeneratedDocsBrowser(browser) {
  const viewports = [
    { label: 'desktop', value: desktopViewport },
    { label: 'mobile', value: mobileViewport },
  ];

  for (const viewport of viewports) {
    const page = await browser.newPage();
    await page.setViewport(viewport.value);
    try {
      for (const expected of generatedDocsPages) {
        await page.goto(pathToFileURL(resolve(repoRoot, expected.path)).href, {
          waitUntil: 'load',
          timeout: 10000,
        });
        const result = await page.evaluate((pageSpec) => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim();
          const bodyText = normalize(document.body.innerText);
          const documentText = normalize(document.body.textContent);
          const h1 = normalize(document.querySelector('h1')?.textContent);
          const canonical = document.querySelector('link[rel="canonical"]')?.getAttribute('href') || '';
          const logo = document.querySelector('.docs-brand img');
          const configurationLink = document.querySelector('main a[href="./configuration/"]');
          const current = document.querySelector('.docs-sidebar a[aria-current="page"]');
          const tableCells = Array.from(document.querySelectorAll('.docs-table-scroll tbody td'));
          const firstTableRow = document.querySelector('.docs-table-scroll tbody tr');

          return {
            bodyText,
            h1,
            canonical,
            hasMain: Boolean(document.querySelector('main#main-content')),
            hasTopbar: Boolean(document.querySelector('.docs-topbar')),
            hasSidebar: Boolean(document.querySelector('aside.docs-sidebar')),
            hasSearch: Boolean(document.querySelector('[data-docs-search] input[type="search"]')),
            hasStylesheet: Array.from(document.styleSheets)
              .some((stylesheet) => stylesheet.href?.endsWith('/css/docs.css')),
            logoLoaded: Boolean(logo?.complete && logo.naturalWidth > 0 && logo.naturalHeight > 0),
            configurationHref: configurationLink?.getAttribute('href') || '',
            currentText: normalize(current?.textContent),
            hasConfigTable: Boolean(document.querySelector('.docs-table-scroll table')),
            unlabeledTableCells: tableCells
              .filter((cell) => !normalize(cell.dataset.label))
              .length,
            unwrappedTableCells: tableCells
              .filter((cell) => !cell.querySelector(':scope > .docs-table-cell-value'))
              .length,
            firstTableRowDisplay: firstTableRow
              ? window.getComputedStyle(firstTableRow).display
              : '',
            hasMobileMenu: Boolean(document.querySelector('button[data-docs-menu]')),
            sidebarInert: document.querySelector('#docs-sidebar')?.inert ?? null,
            scrollWidth: document.documentElement.scrollWidth,
            clientWidth: document.documentElement.clientWidth,
            missingTerms: pageSpec.terms.filter((term) => !documentText.includes(term)),
            missingAnchors: (pageSpec.anchors || [])
              .filter((anchor) => !document.getElementById(anchor)),
          };
        }, expected);

        if (result.h1 !== expected.h1) {
          fail(`${expected.path}: ${viewport.label} h1 must be ${expected.h1}, got ${result.h1 || '(missing)'}`);
        }
        if (result.canonical !== expected.canonical) {
          fail(`${expected.path}: canonical href must be ${expected.canonical}, got ${result.canonical || '(missing)'}`);
        }
        if (!result.hasMain || !result.hasTopbar || !result.hasSidebar || !result.hasSearch) {
          fail(`${expected.path}: ${viewport.label} generated documentation shell is incomplete`);
        }
        if (!result.hasStylesheet || !result.logoLoaded) {
          fail(`${expected.path}: ${viewport.label} generated documentation assets did not load`);
        }
        if (!result.hasMobileMenu) {
          fail(`${expected.path}: generated documentation navigation is missing its menu control`);
        }
        if (viewport.label === 'mobile' && result.sidebarInert !== true) {
          fail(`${expected.path}: closed mobile documentation navigation must start inert`);
        }
        if (viewport.label === 'desktop' && result.sidebarInert !== false) {
          fail(`${expected.path}: desktop documentation navigation must remain interactive`);
        }
        if (result.missingTerms.length > 0) {
          fail(`${expected.path}: ${viewport.label} content missing terms: ${result.missingTerms.join(', ')}`);
        }
        if (result.missingAnchors.length > 0) {
          fail(`${expected.path}: ${viewport.label} content missing anchors: ${result.missingAnchors.join(', ')}`);
        }
        if (result.scrollWidth > result.clientWidth + mobileOverflowTolerancePx) {
          fail(`${expected.path}: ${viewport.label} horizontal overflow: scrollWidth ${result.scrollWidth} > clientWidth ${result.clientWidth} + tolerance ${mobileOverflowTolerancePx}`);
        }
        if (result.hasConfigTable && result.unlabeledTableCells > 0) {
          fail(`${expected.path}: ${viewport.label} table has ${result.unlabeledTableCells} cells without mobile labels`);
        }
        if (result.hasConfigTable && result.unwrappedTableCells > 0) {
          fail(`${expected.path}: ${viewport.label} table has ${result.unwrappedTableCells} cells without a single grouped value`);
        }
        if (viewport.label === 'mobile' && result.hasConfigTable && result.firstTableRowDisplay !== 'block') {
          fail(`${expected.path}: mobile documentation tables must render as labeled row cards`);
        }
        if (expected.label === 'Documentation hub' && result.configurationHref !== './configuration/') {
          fail(`${expected.path}: documentation hub must link to ./configuration/`);
        }
        if (expected.label === 'Configuration Reference') {
          if (result.currentText !== expected.h1) {
            fail(`${expected.path}: configuration sidebar current-page state is missing`);
          }
          if (!result.hasConfigTable) {
            fail(`${expected.path}: configuration reference is missing its responsive field table`);
          }
        }
        if (viewport.label === 'mobile') {
          await page.click('[data-docs-menu]');
          const openNavigation = await page.evaluate(() => ({
            expanded: document.querySelector('[data-docs-menu]')?.getAttribute('aria-expanded'),
            open: document.body.classList.contains('docs-nav-open'),
            sidebarInert: document.querySelector('#docs-sidebar')?.inert,
          }));
          if (
            openNavigation.expanded !== 'true'
            || !openNavigation.open
            || openNavigation.sidebarInert !== false
          ) {
            fail(`${expected.path}: opening mobile documentation navigation must make it interactive`);
          }

          await page.keyboard.press('Escape');
          const closedNavigation = await page.evaluate(() => ({
            expanded: document.querySelector('[data-docs-menu]')?.getAttribute('aria-expanded'),
            open: document.body.classList.contains('docs-nav-open'),
            sidebarInert: document.querySelector('#docs-sidebar')?.inert,
            focusRestored: document.activeElement === document.querySelector('[data-docs-menu]'),
          }));
          if (
            closedNavigation.expanded !== 'false'
            || closedNavigation.open
            || closedNavigation.sidebarInert !== true
            || !closedNavigation.focusRestored
          ) {
            fail(`${expected.path}: Escape must close and inert mobile documentation navigation`);
          }
        }
      }
    } finally {
      await page.close();
    }
  }
}

async function validateNotFoundBrowser(browser) {
  for (const viewport of [
    { label: 'desktop', value: desktopViewport },
    { label: 'mobile', value: mobileViewport },
  ]) {
    const page = await browser.newPage();
    await page.setViewport(viewport.value);
    try {
      await page.goto(pathToFileURL(resolve(siteRoot, '404.html')).href, {
        waitUntil: 'load',
        timeout: 10000,
      });
      const result = await page.evaluate(() => ({
        h1: document.querySelector('h1')?.textContent?.replace(/\s+/g, ' ').trim(),
        main: Boolean(document.querySelector('main#main-content')),
        skipTarget: document.querySelector('.skip-link')?.getAttribute('href'),
        navLabels: Array.from(document.querySelectorAll('nav[aria-label]'))
          .map((nav) => nav.getAttribute('aria-label')),
        emptyLinks: Array.from(document.querySelectorAll('a[href]'))
          .filter((link) => !(link.textContent || '').trim() && !link.getAttribute('aria-label'))
          .length,
        scrollWidth: document.documentElement.scrollWidth,
        clientWidth: document.documentElement.clientWidth,
      }));
      if (result.h1 !== 'That Kiln page isn’t here.') {
        fail(`${publishedPath('404.html')}: ${viewport.label} H1 is missing or incorrect`);
      }
      if (!result.main || result.skipTarget !== '#main-content') {
        fail(`${publishedPath('404.html')}: ${viewport.label} skip-link target is incomplete`);
      }
      for (const label of ['Primary', 'Useful destinations']) {
        if (!result.navLabels.includes(label)) {
          fail(`${publishedPath('404.html')}: ${viewport.label} is missing the ${label} navigation label`);
        }
      }
      if (result.emptyLinks > 0) {
        fail(`${publishedPath('404.html')}: ${viewport.label} has ${result.emptyLinks} empty links`);
      }
      if (result.scrollWidth > result.clientWidth + mobileOverflowTolerancePx) {
        fail(`${publishedPath('404.html')}: ${viewport.label} horizontal overflow`);
      }
    } finally {
      await page.close();
    }
  }
}

async function validateEveryPublishedRouteAccessibilityShell(browser) {
  const routePaths = [
    ...pages.map((page) => page.path),
    publishedPath('docs/index.html'),
    ...docsManifest.documents.map((document) => (
      publishedPath(`docs/${document.slug}/index.html`)
    )),
    publishedPath('404.html'),
  ];
  const page = await browser.newPage();
  await page.setViewport(mobileViewport);
  try {
    for (const routePath of routePaths) {
      await page.goto('about:blank');
      await page.goto(pathToFileURL(resolve(repoRoot, routePath)).href, {
        waitUntil: 'load',
        timeout: 10000,
      });
      const result = await page.evaluate((overflowTolerancePx) => {
        const duplicateIds = Array.from(document.querySelectorAll('[id]'))
          .map((element) => element.id)
          .filter((id, index, ids) => id && ids.indexOf(id) !== index);
        const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'))
          .filter((heading) => (
            heading.getClientRects().length > 0
            && !heading.closest('[inert]')
            && !heading.closest('details:not([open])')
          ))
          .map((heading) => Number(heading.tagName.slice(1)));
        const skippedHeadings = headings
          .slice(1)
          .filter((level, index) => level > headings[index] + 1);
        const likelyOverflowingElements = Array.from(document.body.querySelectorAll('*'))
          .filter((element) => {
            const rect = element.getBoundingClientRect();
            return (
              rect.left < -overflowTolerancePx
              || rect.right > window.innerWidth + overflowTolerancePx
            );
          })
          .slice(0, 5)
          .map((element) => `${element.tagName.toLowerCase()}${element.id ? `#${element.id}` : ''}`);
        return {
          lang: document.documentElement.lang,
          h1Count: document.querySelectorAll('h1').length,
          hasMain: Boolean(document.querySelector('main[id]')),
          hasSkipLink: document.querySelector('.skip-link')?.getAttribute('href')?.startsWith('#'),
          skipLinkIsFirstInteractive: document.querySelector(
            'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])',
          )?.classList.contains('skip-link'),
          sidebarInert: document.querySelector('#docs-sidebar')?.inert ?? null,
          imagesMissingAlt: document.querySelectorAll('img:not([alt])').length,
          duplicateIds: [...new Set(duplicateIds)],
          skippedHeadingCount: skippedHeadings.length,
          scrollWidth: document.documentElement.scrollWidth,
          clientWidth: document.documentElement.clientWidth,
          likelyOverflowingElements,
        };
      }, mobileOverflowTolerancePx);

      if (result.lang !== 'en') fail(`${routePath}: document language must be en`);
      if (result.h1Count !== 1) fail(`${routePath}: expected one H1, got ${result.h1Count}`);
      if (!result.hasMain || !result.hasSkipLink || !result.skipLinkIsFirstInteractive) {
        fail(`${routePath}: main landmark or skip link is missing`);
      }
      if (result.sidebarInert === false) {
        fail(`${routePath}: closed mobile documentation navigation must be inert`);
      }
      if (result.imagesMissingAlt > 0) {
        fail(`${routePath}: ${result.imagesMissingAlt} images have no alt attribute`);
      }
      if (result.duplicateIds.length > 0) {
        fail(`${routePath}: duplicate IDs: ${result.duplicateIds.join(', ')}`);
      }
      if (result.skippedHeadingCount > 0) {
        fail(`${routePath}: visible heading outline skips ${result.skippedHeadingCount} levels`);
      }
      if (result.scrollWidth > result.clientWidth + mobileOverflowTolerancePx) {
        fail(`${routePath}: mobile accessibility pass found horizontal overflow; likely elements: ${result.likelyOverflowingElements.join(', ') || 'none'}`);
      }

      await page.$eval('.skip-link', (link) => link.focus());
      const focusedSkipLink = await page.evaluate(() => {
        const active = document.activeElement;
        const rect = active?.getBoundingClientRect();
        return {
          active: active?.classList.contains('skip-link'),
          visible: Boolean(rect && rect.bottom > 0 && rect.top < window.innerHeight),
          target: active?.getAttribute('href'),
        };
      });
      if (!focusedSkipLink.active || !focusedSkipLink.visible) {
        fail(`${routePath}: the skip link cannot receive visible keyboard focus`);
      }
      await page.keyboard.press('Enter');
      const skipLinkActivated = await page.evaluate((target) => (
        window.location.hash === target
      ), focusedSkipLink.target);
      if (!skipLinkActivated) {
        fail(`${routePath}: keyboard activation of the skip link did not reach its target`);
      }

      const accessibilityTree = await page.accessibility.snapshot({ interestingOnly: true });
      const unnamedInteractiveRoles = [];
      const visit = (node) => {
        if (
          ['button', 'checkbox', 'combobox', 'link', 'radio', 'searchbox', 'textbox'].includes(node.role)
          && !node.name?.trim()
        ) {
          unnamedInteractiveRoles.push(node.role);
        }
        for (const child of node.children || []) visit(child);
      };
      if (accessibilityTree) visit(accessibilityTree);
      if (unnamedInteractiveRoles.length > 0) {
        fail(`${routePath}: accessibility tree has unnamed interactive roles: ${unnamedInteractiveRoles.join(', ')}`);
      }
    }
  } finally {
    await page.close();
  }
}

async function runSmoke() {
  validateReadmeStartupBanner();
  validateReadmeMedia();
  validateCurrentPerformancePositioning();
  validateStaticSocialMetadata();
  validateNotFoundPage();
  validateReadmeColdReaderCoverage();
  validateReadmeImageReferences();
  validateReadmeQuickStartPaths();
  validateAdapterListStaleWordingSurfaces();
  validateReadmeAdapterListSemantics();
  validateGrpoOverviewRequestsImports();
  validateGrpoDemoPayloadCue();
  validateLandingDesktopVersionSplitCue();
  validateQuickstartDesktopVersionSplitCue();
  validateGrpoSafetyAndPromotionCues();
  validateEvalsOperationAndDistributionCues();
  validateApiProfileAndMutationCues();
  validateCliDistributionAndProfileCues();
  validateSelfHostedProductPageAssets();
  validateQuickstartMarkdownMedia();
  validateQuickstartServerBinaryPath();
  validateQuickstartCliReference();
  validateEmbeddedUiHelpLinks();
  validateDesktopDocumentationLinks();
  validateCliHelpOnboardingCopy();
  validateLaunchSentinel();
  validateDemoReadmeInventory();
  validatePrimaryDocsEditorialBudget();
  validateRuntimeEnvironmentBoundaryDocumentationSourceContract();
  validateSftLossRouteDocumentationSourceContract();
  const hasGeneratedDocs = validateGeneratedDocsArtifacts();
  validateDocsSiteCanonicalLinks();
  validateDocsSiteLocalLinks();
  validateMarkdownLocalLinks();
  if (staticOnly) return;

  const puppeteer = await loadPuppeteer();
  const browser = await launchChromiumWithRetry(puppeteer);

  try {
    await validateEmbeddedUiControlAccessibleNames(browser);
    if (hasGeneratedDocs) await validateGeneratedDocsBrowser(browser);
    await validateNotFoundBrowser(browser);
    await validateEveryPublishedRouteAccessibilityShell(browser);

    const page = await browser.newPage();
    await page.setViewport(mobileViewport);

    for (const sitePage of pages) {
      const filePath = resolve(repoRoot, sitePage.path);
      await page.goto(pathToFileURL(filePath).href, { waitUntil: 'domcontentloaded', timeout: 10000 });

      const expectedFooterLinksWithUrls = expectedFooterLinks.map((link) => ({
        label: link.label,
        href: link.href || expectedLocalHref(link.localPath),
      }));

      const result = await page.evaluate((expectedLabels, currentLabel, expectedLinks, overflowTolerancePx) => {
        const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim();
        const selectorFor = (element) => {
          if (!element || element === document.documentElement) return 'html';
          if (element.id) return `${element.tagName.toLowerCase()}#${CSS.escape(element.id)}`;

          const classes = Array.from(element.classList || [])
            .slice(0, 3)
            .map((className) => `.${CSS.escape(className)}`)
            .join('');
          const tagSelector = `${element.tagName.toLowerCase()}${classes}`;
          const parent = element.parentElement;
          if (!parent) return tagSelector;

          const sameTagSiblings = Array.from(parent.children)
            .filter((sibling) => sibling.tagName === element.tagName);
          if (sameTagSiblings.length <= 1) return tagSelector;
          return `${tagSelector}:nth-of-type(${sameTagSiblings.indexOf(element) + 1})`;
        };
        const likelyOverflowingElements = Array.from(document.body.querySelectorAll('*'))
          .map((element) => {
            const rect = element.getBoundingClientRect();
            return {
              selector: selectorFor(element),
              left: Math.floor(rect.left),
              right: Math.ceil(rect.right),
              scrollWidth: element.scrollWidth,
              clientWidth: element.clientWidth,
              text: normalize(element.innerText || element.textContent).slice(0, 80),
              exceedsViewport: rect.left < -overflowTolerancePx || rect.right > window.innerWidth + overflowTolerancePx,
              overflowsInternally: element.scrollWidth > element.clientWidth + overflowTolerancePx,
            };
          })
          .filter((element) => element.exceedsViewport || element.overflowsInternally)
          .slice(0, 6);
        const h1 = document.querySelector('h1');
        const nav = document.querySelector('nav.site-nav');
        const navLinks = Array.from(nav?.querySelectorAll('a') || []);
        const navLabels = navLinks.map((link) => normalize(link.textContent));
        const missingLabels = expectedLabels.filter((label) => !navLabels.includes(label));
        const current = navLinks.find((link) => link.getAttribute('aria-current') === 'page');
        const homeCurrent = document.querySelector('[aria-current="page"]');
        const footer = document.querySelector('footer');
        const footerLinks = Array.from(footer?.querySelectorAll('a[href]') || []).map((link) => ({
          label: normalize(link.textContent),
          href: link.href,
        }));
        const missingFooterLinks = expectedLinks
          .filter((expectedLink) => !footerLinks.some((link) => (
            link.label === expectedLink.label && link.href === expectedLink.href
          )))
          .map((link) => `${link.label} -> ${link.href}`);

        return {
          h1Text: normalize(h1?.textContent),
          hasNav: Boolean(nav),
          hasFooter: Boolean(footer),
          footerLabels: footerLinks.map((link) => link.label),
          missingLabels,
          missingFooterLinks,
          currentLabel: normalize(current?.textContent),
          hasHomeCurrent: Boolean(homeCurrent),
          scrollWidth: document.documentElement.scrollWidth,
          clientWidth: document.documentElement.clientWidth,
          likelyOverflowingElements,
          currentMatches: currentLabel ? normalize(current?.textContent) === currentLabel : Boolean(homeCurrent),
        };
      }, expectedNavLabels, sitePage.currentLabel, expectedFooterLinksWithUrls, mobileOverflowTolerancePx);

      if (!result.h1Text) fail(`${sitePage.path}: missing h1`);
      if (!result.hasNav) fail(`${sitePage.path}: missing nav.site-nav`);
      if (!result.hasFooter) fail(`${sitePage.path}: missing footer`);
      if (result.missingLabels.length > 0) {
        fail(`${sitePage.path}: nav.site-nav missing labels: ${result.missingLabels.join(', ')}`);
      }
      if (result.missingFooterLinks.length > 0) {
        fail(`${sitePage.path}: footer missing visible links: ${result.missingFooterLinks.join(', ')}`);
      }
      if (
        sitePage.path !== homePagePath
        && JSON.stringify(result.footerLabels) !== JSON.stringify(expectedProductFooterOrder)
      ) {
        fail(`${sitePage.path}: footer links must use the shared order; got ${result.footerLabels.join(', ')}`);
      }
      if (!result.currentMatches) {
        const expected = sitePage.currentLabel || 'an aria-current="page" marker';
        fail(`${sitePage.path}: expected current marker for ${expected}, got ${result.currentLabel || 'none'}`);
      }
      if (result.scrollWidth > result.clientWidth + mobileOverflowTolerancePx) {
        fail(`${sitePage.path}: mobile horizontal overflow at ${mobileViewport.width}x${mobileViewport.height}: scrollWidth ${result.scrollWidth} > clientWidth ${result.clientWidth} + tolerance ${mobileOverflowTolerancePx}; likely overflowing elements: ${formatLikelyOverflowingElements(result.likelyOverflowingElements)}`);
      }

      if (sitePage.path === homePagePath) {
        const homeResult = await page.evaluate(() => {
          const learningPass = document.querySelector('[data-learning-pass]');
          const replay = document.querySelector('[data-replay-learning]');
          const productImage = document.querySelector('.product-shot-window img');
          const navToggle = document.querySelector('.nav-toggle');
          return {
            hasLearningPass: Boolean(learningPass),
            hasBefore: Boolean(learningPass?.querySelector('.learning-answer-before')),
            hasAfter: Boolean(learningPass?.querySelector('.learning-answer-after')),
            hasTrainingEvent: Boolean(learningPass?.querySelector('.learning-heat-event')),
            hasReplay: Boolean(replay),
            hasNavToggle: Boolean(navToggle),
            productImageSrc: productImage?.getAttribute('src') || '',
            productImageSrcset: productImage?.getAttribute('srcset') || '',
            productImageSizes: productImage?.getAttribute('sizes') || '',
            productImageLoading: productImage?.getAttribute('loading') || '',
          };
        });

        if (!homeResult.hasLearningPass || !homeResult.hasBefore || !homeResult.hasAfter || !homeResult.hasTrainingEvent || !homeResult.hasReplay) {
          fail(`${sitePage.path}: homepage learning-pass visual is incomplete`);
        }
        if (!homeResult.hasNavToggle) {
          fail(`${sitePage.path}: homepage mobile navigation toggle is missing`);
        }
        if (homeResult.productImageSrc !== 'assets/server-ui-dashboard.webp' || homeResult.productImageLoading !== 'lazy') {
          fail(`${sitePage.path}: homepage product image must use the optimized lazy-loaded WebP asset`);
        }
        if (homeResult.productImageSrcset !== dashboardImageSrcset
            || homeResult.productImageSizes !== dashboardImageSizes) {
          fail(`${sitePage.path}: homepage product image must declare the responsive dashboard sources`);
        }

        await page.click('.nav-toggle');
        const openNav = await page.evaluate(() => ({
          expanded: document.querySelector('.nav-toggle')?.getAttribute('aria-expanded'),
          open: document.querySelector('.site-nav')?.dataset.open,
          bodyLocked: document.body.classList.contains('nav-open'),
          visible: getComputedStyle(document.querySelector('.site-nav')).display !== 'none',
        }));
        if (openNav.expanded !== 'true' || openNav.open !== 'true' || !openNav.bodyLocked || !openNav.visible) {
          fail(`${sitePage.path}: homepage mobile navigation does not open accessibly`);
        }

        await page.keyboard.press('Escape');
        const closedNav = await page.evaluate(() => ({
          expanded: document.querySelector('.nav-toggle')?.getAttribute('aria-expanded'),
          open: document.querySelector('.site-nav')?.dataset.open,
          bodyLocked: document.body.classList.contains('nav-open'),
          focused: document.activeElement === document.querySelector('.nav-toggle'),
        }));
        if (closedNav.expanded !== 'false' || closedNav.open !== 'false' || closedNav.bodyLocked || !closedNav.focused) {
          fail(`${sitePage.path}: Escape must close homepage navigation and restore focus`);
        }

        await page.click('[data-replay-learning]');
        const replayStarted = await page.evaluate(() => ({
          animating: document.querySelector('[data-learning-pass]')?.classList.contains('is-replaying'),
          disabled: document.querySelector('[data-replay-learning]')?.disabled,
        }));
        if (!replayStarted.animating || !replayStarted.disabled) {
          fail(`${sitePage.path}: learning-pass replay does not expose its active state`);
        }
        await new Promise((resolve) => setTimeout(resolve, 1200));
        const replayFinished = await page.evaluate(() => ({
          animating: document.querySelector('[data-learning-pass]')?.classList.contains('is-replaying'),
          disabled: document.querySelector('[data-replay-learning]')?.disabled,
          status: document.querySelector('[data-learning-status]')?.textContent || '',
        }));
        if (replayFinished.animating || replayFinished.disabled || !replayFinished.status.includes('improved adapter is serving')) {
          fail(`${sitePage.path}: learning-pass replay does not settle with an announced result`);
        }

        await page.$eval('.product-shot-window img', (image) => image.scrollIntoView({ block: 'center' }));
        await page.waitForFunction(() => document.querySelector('.product-shot-window img')?.naturalWidth > 0);
      }

      if (sitePage.path !== homePagePath) {
        const shellResult = await page.evaluate(() => ({
          enhanced: document.body.classList.contains('product-page-shell'),
          hasToggle: Boolean(document.querySelector('.nav-toggle')),
          hasExplore: Boolean(document.querySelector('.nav-explore-toggle')),
          closed: getComputedStyle(document.querySelector('.site-nav')).display === 'none',
        }));
        if (!shellResult.enhanced || !shellResult.hasToggle || !shellResult.hasExplore || !shellResult.closed) {
          fail(`${sitePage.path}: shared mobile product navigation shell is incomplete`);
        }

        await page.click('.nav-toggle');
        const openNav = await page.evaluate(() => ({
          expanded: document.querySelector('.nav-toggle')?.getAttribute('aria-expanded'),
          open: document.querySelector('.site-nav')?.dataset.open,
          bodyLocked: document.body.classList.contains('nav-open'),
          visible: getComputedStyle(document.querySelector('.site-nav')).display !== 'none',
        }));
        if (openNav.expanded !== 'true' || openNav.open !== 'true' || !openNav.bodyLocked || !openNav.visible) {
          fail(`${sitePage.path}: shared mobile product navigation does not open accessibly`);
        }

        await page.keyboard.press('Escape');
        const closedNav = await page.evaluate(() => ({
          expanded: document.querySelector('.nav-toggle')?.getAttribute('aria-expanded'),
          open: document.querySelector('.site-nav')?.dataset.open,
          bodyLocked: document.body.classList.contains('nav-open'),
          focused: document.activeElement === document.querySelector('.nav-toggle'),
        }));
        if (closedNav.expanded !== 'false' || closedNav.open !== 'false' || closedNav.bodyLocked || !closedNav.focused) {
          fail(`${sitePage.path}: Escape must close shared product navigation and restore focus`);
        }
      }

      if (sitePage.path === demoPagePath) {
        const demoImages = await page.$$('.screen img');
        for (const image of demoImages) {
          await image.evaluate((element) => element.scrollIntoView({ block: 'center' }));
          await page.waitForFunction(
            (element) => Boolean(element?.complete && element.naturalWidth > 0 && element.naturalHeight > 0),
            { timeout: 5000 },
            image,
          );
        }
        const demoResult = await page.evaluate(() => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
          const referencedCasts = Array.from(
            document.querySelectorAll('a[href$=".cast"]'),
            (link) => link.getAttribute('href'),
          );
          const tourImages = Array.from(document.querySelectorAll('.screen img'), (image) => ({
            src: image.getAttribute('src') || '',
            srcset: image.getAttribute('srcset') || '',
            sizes: image.getAttribute('sizes') || '',
            loading: image.getAttribute('loading') || '',
            decoding: image.getAttribute('decoding') || '',
            currentSrc: image.currentSrc,
            complete: image.complete,
            naturalWidth: image.naturalWidth,
            naturalHeight: image.naturalHeight,
          }));

          return {
            bodyText: normalize(document.body.innerText),
            referencedCasts,
            tourImages,
            externalResources: performance.getEntriesByType('resource')
              .map((entry) => entry.name)
              .filter((url) => /^https?:/i.test(url)),
          };
        });

        const missingSections = expectedDemoSections
          .filter((section) => !section.terms.every((term) => demoResult.bodyText.includes(term)))
          .map((section) => section.label);
        if (missingSections.length > 0) {
          fail(`${sitePage.path}: missing demo sections: ${missingSections.join(', ')}`);
        }
        if (demoResult.externalResources.length > 0) {
          fail(`${sitePage.path}: Demo must not fetch third-party runtime resources: ${demoResult.externalResources.join(', ')}`);
        }
        if (demoResult.tourImages.length !== demoImageNames.length) {
          fail(`${sitePage.path}: expected ${demoImageNames.length} responsive tour images, got ${demoResult.tourImages.length}`);
        }
        demoResult.tourImages.forEach((image, index) => {
          const name = demoImageNames[index];
          const expectedSrc = `../assets/server-ui-${name}.webp`;
          const expectedSrcset = `../assets/server-ui-${name}-720.webp 720w, ../assets/server-ui-${name}-1440.webp 1440w, ../assets/server-ui-${name}.webp 2880w`;
          if (image.src !== expectedSrc
              || image.srcset !== expectedSrcset
              || image.sizes !== demoImageSizes
              || image.loading !== 'lazy'
              || image.decoding !== 'async') {
            fail(`${sitePage.path}: ${name} tour image is not fully responsive and deferred: ${JSON.stringify(image)}`);
          }
          if (!image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) {
            fail(`${sitePage.path}: ${name} responsive tour image did not load`);
          }
          if (!image.currentSrc.endsWith(`/assets/server-ui-${name}-720.webp`)) {
            fail(`${sitePage.path}: mobile ${name} tour image must load the 720px source, got ${image.currentSrc || 'no source'}`);
          }
        });

        validateDemoCasts(sitePage.path, demoResult.referencedCasts);
      }

      if (sitePage.path === quickstartPagePath) {
        await page.evaluate(() => {
          document.querySelector('main img[src="assets/server-ui-dashboard.webp"]')?.scrollIntoView({
            block: 'center',
          });
        });
        await page.waitForFunction(
          () => {
            const image = document.querySelector('main img[src="assets/server-ui-dashboard.webp"]');
            return Boolean(image?.complete && image.naturalWidth > 0 && image.naturalHeight > 0);
          },
          { timeout: 5000 },
        );
        const quickstartResult = await page.evaluate(() => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
          // Include closed disclosure content: the quickstart deliberately keeps
          // exact training contracts behind a progressive-disclosure summary.
          const bodyText = normalize(document.body.textContent);
          const headings = normalize(Array.from(document.querySelectorAll('h1, h2, h3'))
            .map((heading) => heading.textContent || '')
            .join('\n'));
          const codeText = normalize(Array.from(document.querySelectorAll('pre > code, code'))
            .map((code) => code.innerText || code.textContent)
            .join('\n'));
          const links = Array.from(document.querySelectorAll('main a[href]')).map((link) => ({
            href: link.getAttribute('href'),
            text: normalize(link.textContent),
          }));
          const dashboardImage = document.querySelector('main img[src="assets/server-ui-dashboard.webp"]');
          const nextTaskHeading = Array.from(document.querySelectorAll('#train h2'))
            .find((heading) => heading.textContent?.includes('Open the guide for your next task'));

          return {
            bodyText,
            headings,
            codeText,
            links,
            nextTaskInsideContainer: nextTaskHeading?.parentElement?.classList.contains('max-w-5xl') === true,
            dashboardImage: dashboardImage ? {
              alt: normalize(dashboardImage.getAttribute('alt')),
              complete: dashboardImage.complete,
              currentSrc: dashboardImage.currentSrc,
              naturalWidth: dashboardImage.naturalWidth,
              naturalHeight: dashboardImage.naturalHeight,
              sizes: dashboardImage.getAttribute('sizes') || '',
              srcset: dashboardImage.getAttribute('srcset') || '',
            } : null,
          };
        });

        const missingSections = expectedQuickstartSections
          .filter((section) => !section.terms.every((term) => {
            const normalizedTerm = term.toLowerCase();
            return quickstartResult.headings.includes(normalizedTerm)
              || quickstartResult.bodyText.includes(normalizedTerm)
              || quickstartResult.codeText.includes(normalizedTerm);
          }))
          .map((section) => section.label);
        if (missingSections.length > 0) {
          fail(`${sitePage.path}: missing quickstart cold-reader coverage: ${missingSections.join(', ')}`);
        }

        const missingDashboardTerms = expectedQuickstartDashboardTerms
          .filter((term) => !quickstartResult.bodyText.includes(term));
        if (missingDashboardTerms.length > 0) {
          fail(`${sitePage.path}: dashboard checkpoint missing terms: ${missingDashboardTerms.join(', ')}`);
        }
        if (!quickstartResult.nextTaskInsideContainer) {
          fail(`${sitePage.path}: next-task guide grid escaped the bounded training-section container`);
        }

        if (!quickstartResult.dashboardImage) {
          fail(`${sitePage.path}: missing dashboard screenshot assets/server-ui-dashboard.webp`);
        }
        if (!quickstartResult.dashboardImage.complete
            || quickstartResult.dashboardImage.naturalWidth <= 0
            || quickstartResult.dashboardImage.naturalHeight <= 0) {
          fail(`${sitePage.path}: dashboard screenshot did not load locally`);
        }
        if (quickstartResult.dashboardImage.srcset !== dashboardImageSrcset
            || quickstartResult.dashboardImage.sizes !== dashboardImageSizes) {
          fail(`${sitePage.path}: dashboard screenshot must declare the responsive dashboard sources`);
        }
        if (!quickstartResult.dashboardImage.currentSrc.endsWith('/assets/server-ui-dashboard-720.webp')) {
          fail(`${sitePage.path}: mobile viewport must load the 720px dashboard source, got ${quickstartResult.dashboardImage.currentSrc || 'no source'}`);
        }
        if (!expectedQuickstartDashboardTerms.every((term) => quickstartResult.dashboardImage.alt.includes(term))) {
          fail(`${sitePage.path}: dashboard screenshot alt text must mention status, adapters, training, and quick inference`);
        }

        const missingLinks = expectedQuickstartLinks
          .filter((expectedLink) => !quickstartResult.links.some((link) => link.href === expectedLink.href))
          .map((link) => link.label);
        if (missingLinks.length > 0) {
          fail(`${sitePage.path}: missing quickstart onboarding links: ${missingLinks.join(', ')}`);
        }
      }

      if (sitePage.path === apiPagePath) {
        const apiResult = await page.evaluate(() => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
          const bodyText = normalize(document.body.textContent);
          const endpointText = normalize(Array.from(document.querySelectorAll('.endpoint, code'))
            .map((element) => element.textContent || '')
            .join('\n'));
          const endpointElements = Array.from(document.querySelectorAll('.endpoint'));
          const endpointRoutes = endpointElements
            .map((element) => normalize(element.querySelector('code')?.textContent || ''));
          const endpointDescriptions = endpointElements
            .map((element) => normalize(element.textContent || ''));
          const headings = normalize(Array.from(document.querySelectorAll('h2, h3'))
            .map((heading) => heading.textContent || '')
            .join('\n'));
          const copyableCodeBlocks = Array.from(document.querySelectorAll('pre > code'))
            .map((code) => normalize(code.innerText || code.textContent));
          const copyButtons = Array.from(document.querySelectorAll('.copy-code-button'));
          const deepReference = document.querySelector('.reference-details');

          return {
            bodyText,
            endpointText,
            endpointRoutes,
            endpointDescriptions,
            headings,
            copyableCodeBlocks,
            copyButtonCount: copyButtons.length,
            deepReference: deepReference ? {
              open: deepReference.open,
              summary: normalize(deepReference.querySelector('summary')?.textContent),
            } : null,
          };
        });

        if (!apiResult.deepReference
            || apiResult.deepReference.open
            || !apiResult.deepReference.summary.includes('deep reference: resolved runtime configuration')) {
          fail(`${sitePage.path}: exhaustive runtime policy must be present behind a clearly labeled, closed disclosure`);
        }

        const missingEndpoints = expectedApiEndpoints.filter((endpoint) => {
          const normalizedEndpoint = endpoint.toLowerCase();
          return !apiResult.endpointText.includes(normalizedEndpoint)
            && !apiResult.bodyText.includes(normalizedEndpoint);
        });
        if (missingEndpoints.length > 0) {
          fail(`${sitePage.path}: missing API endpoint coverage: ${missingEndpoints.join(', ')}`);
        }

        const normalizedJobDetailEndpoint = trainingJobDetailEndpoint.toLowerCase();
        if (!apiResult.endpointRoutes.includes(normalizedJobDetailEndpoint)) {
          fail(`${sitePage.path}: missing ${trainingJobDetailEndpoint} endpoint (rich job detail; GET + DELETE)`);
        }
        if (apiResult.bodyText.includes(`no separate ${normalizedJobDetailEndpoint} route`)) {
          fail(`${sitePage.path}: ${trainingJobDetailEndpoint} is a real route — remove the denial wording`);
        }

        for (const phrase of staleAdapterListPhrases) {
          if (apiResult.bodyText.includes(phrase.toLowerCase())) {
            fail(`${sitePage.path}: GET /v1/adapters wording must not say "${phrase}"`);
          }
        }
        const adaptersEndpoint = apiResult.endpointDescriptions
          .find((line) => line.startsWith('get /v1/adapters ')) || '';
        if (!adaptersEndpoint) {
          fail(`${sitePage.path}: missing GET /v1/adapters endpoint wording`);
        }
        for (const term of expectedAdapterListSemantics) {
          if (!adaptersEndpoint.includes(term.toLowerCase())) {
            fail(`${sitePage.path}: GET /v1/adapters wording missing ${term}`);
          }
        }

        const missingSections = expectedApiReaderSections
          .map((section) => ({
            label: section.label,
            missingTerms: section.terms.filter((term) => {
            const normalizedTerm = term.toLowerCase();
              return !apiResult.headings.includes(normalizedTerm)
                && !apiResult.bodyText.includes(normalizedTerm);
            }),
          }))
          .filter((section) => section.missingTerms.length > 0);
        if (missingSections.length > 0) {
          fail(`${sitePage.path}: missing API cold-reader sections: ${missingSections
            .map((section) => `${section.label} (${section.missingTerms.join(', ')})`)
            .join('; ')}`);
        }

        const missingCodeExamples = expectedApiReaderCodeExamples
          .filter((example) => !apiResult.copyableCodeBlocks.some((codeBlock) => (
            example.terms.every((term) => codeBlock.includes(term.toLowerCase()))
          )))
          .map((example) => example.label);
        if (missingCodeExamples.length > 0) {
          fail(`${sitePage.path}: missing copy-paste API examples: ${missingCodeExamples.join(', ')}`);
        }

        const adapterUploadCode = apiResult.copyableCodeBlocks.find((codeBlock) => codeBlock.includes('/v1/adapters/upload'));
        if (!adapterUploadCode) {
          fail(`${sitePage.path}: missing /v1/adapters/upload copy-paste example`);
        }
        if (!adapterUploadCode.includes('-f "archive=@default-adapter.tar.gz"')) {
          fail(`${sitePage.path}: adapter upload example must use multipart field archive for default-adapter.tar.gz`);
        }
        if (adapterUploadCode.includes('-f "file=@')) {
          fail(`${sitePage.path}: adapter upload example uses stale multipart field file; use archive`);
        }

        const adapterMergeCodeBlocks = apiResult.copyableCodeBlocks.filter((codeBlock) => codeBlock.includes('/v1/adapters/merge'));
        if (adapterMergeCodeBlocks.length < 2) {
          fail(`${sitePage.path}: expected TIES and concat /v1/adapters/merge copy-paste examples`);
        }
        for (const [index, codeBlock] of adapterMergeCodeBlocks.entries()) {
          if (!codeBlock.includes('"sources"')) {
            fail(`${sitePage.path}: adapter merge example ${index + 1} must use sources array`);
          }
          if (codeBlock.includes('"adapters"')) {
            fail(`${sitePage.path}: adapter merge example ${index + 1} uses stale adapters array; use sources`);
          }
        }

        if (apiResult.copyButtonCount < apiResult.copyableCodeBlocks.length) {
          fail(`${sitePage.path}: expected each API code block to have a copy button; got ${apiResult.copyButtonCount} for ${apiResult.copyableCodeBlocks.length} code blocks`);
        }
      }


      if (sitePage.path === cliPagePath) {
        const cliResult = await page.evaluate(() => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
          const bodyText = normalize(document.body.innerText);
          const headings = normalize(Array.from(document.querySelectorAll('h1, h2, h3'))
            .map((heading) => heading.textContent || '')
            .join('\n'));
          const copyableCodeBlocks = Array.from(document.querySelectorAll('pre > code'))
            .map((code) => normalize(code.innerText || code.textContent));
          const copyButtons = Array.from(document.querySelectorAll('.copy-code-button'));
          const links = Array.from(document.querySelectorAll('main a[href]')).map((link) => ({
            href: link.getAttribute('href'),
            text: normalize(link.textContent),
          }));
          const hero = document.querySelector('main > section:first-of-type');
          const heroText = normalize(hero?.innerText || '');
          const heroLinks = Array.from(hero?.querySelectorAll('a[href]') || []).map((link) => ({
            href: link.getAttribute('href'),
            text: normalize(link.textContent),
          }));
          const fragmentIds = Array.from(document.querySelectorAll('[id]'))
            .map((element) => element.getAttribute('id'));

          return {
            bodyText,
            headings,
            copyableCodeBlocks,
            copyButtonCount: copyButtons.length,
            links,
            heroText,
            heroLinks,
            fragmentIds,
          };
        });

        const missingModelSetupCueTerms = expectedCliModelSetupCue.terms
          .filter((term) => !cliResult.heroText.includes(term.toLowerCase()));
        if (missingModelSetupCueTerms.length > 0) {
          fail(`${sitePage.path}: missing ${expectedCliModelSetupCue.label}: ${missingModelSetupCueTerms.join(', ')}`);
        }
        if (!cliResult.heroLinks.some((link) => link.href === expectedCliModelSetupCue.href)) {
          fail(`${sitePage.path}: ${expectedCliModelSetupCue.label} must link to Quickstart`);
        }

        const missingSections = expectedCliReaderSections
          .filter((section) => !section.terms.every((term) => {
            const normalizedTerm = term.toLowerCase();
            return cliResult.headings.includes(normalizedTerm)
              || cliResult.bodyText.includes(normalizedTerm)
              || cliResult.copyableCodeBlocks.some((codeBlock) => codeBlock.includes(normalizedTerm));
          }))
          .map((section) => section.label);
        if (missingSections.length > 0) {
          fail(`${sitePage.path}: missing CLI cold-reader coverage: ${missingSections.join(', ')}`);
        }

        const missingCodeExamples = expectedCliReaderCodeExamples
          .filter((example) => !cliResult.copyableCodeBlocks.some((codeBlock) => (
            example.terms.every((term) => codeBlock.includes(term.toLowerCase()))
          )))
          .map((example) => example.label);
        if (missingCodeExamples.length > 0) {
          fail(`${sitePage.path}: missing copy-paste CLI examples: ${missingCodeExamples.join(', ')}`);
        }

        if (cliResult.bodyText.includes('prompts/groups') || cliResult.copyableCodeBlocks.some((codeBlock) => codeBlock.includes('prompts/groups'))) {
          fail(`${sitePage.path}: kiln train grpo docs must describe scored groups, not prompts/groups; prompts belong to /v1/completions/batch`);
        }

        if (cliResult.copyButtonCount < cliResult.copyableCodeBlocks.length) {
          fail(`${sitePage.path}: expected each CLI code block to have a copy button; got ${cliResult.copyButtonCount} for ${cliResult.copyableCodeBlocks.length} code blocks`);
        }

        const missingLinks = expectedCliLinks
          .filter((expectedLink) => !cliResult.links.some((link) => link.href === expectedLink.href))
          .map((link) => link.label);
        if (missingLinks.length > 0) {
          fail(`${sitePage.path}: missing CLI next-step links: ${missingLinks.join(', ')}`);
        }

        const missingHeroFragments = expectedCliHeroFragments
          .filter((expectedLink) => !cliResult.heroLinks.some((link) => link.href === expectedLink.href))
          .map((link) => `${link.label} -> ${link.href}`);
        if (missingHeroFragments.length > 0) {
          fail(`${sitePage.path}: missing CLI hero deep links: ${missingHeroFragments.join(', ')}`);
        }

        const missingPageFragments = expectedCliPageFragments
          .filter((fragment) => !cliResult.fragmentIds.includes(fragment.slice(1)));
        if (missingPageFragments.length > 0) {
          fail(`${sitePage.path}: missing CLI page fragments: ${missingPageFragments.join(', ')}`);
        }
      }

      if (sitePage.path === architecturePagePath) {
        const architectureResult = await page.evaluate(() => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
          const bodyText = normalize(document.body.innerText);
          const headings = normalize(Array.from(document.querySelectorAll('h2, h3'))
            .map((heading) => heading.textContent || '')
            .join('\n'));
          const copyableCodeBlocks = Array.from(document.querySelectorAll('pre > code'))
            .map((code) => normalize(code.innerText || code.textContent));
          const links = Array.from(document.querySelectorAll('a[href]')).map((link) => ({
            href: link.getAttribute('href'),
            text: normalize(link.textContent),
          }));
          const fragmentIds = Array.from(document.querySelectorAll('[id]'))
            .map((element) => element.id);

          return {
            bodyText,
            headings,
            copyableCodeBlocks,
            links,
            fragmentIds,
          };
        });

        const missingSections = expectedArchitectureReaderSections
          .filter((section) => !section.terms.every((term) => {
            const normalizedTerm = term.toLowerCase();
            return architectureResult.headings.includes(normalizedTerm)
              || architectureResult.bodyText.includes(normalizedTerm);
          }))
          .map((section) => section.label);
        if (missingSections.length > 0) {
          fail(`${sitePage.path}: missing architecture cold-reader coverage: ${missingSections.join(', ')}`);
        }

        const hasRequestFlow = architectureResult.copyableCodeBlocks.some((codeBlock) => (
          expectedArchitectureFlowTerms.every((term) => codeBlock.includes(term.toLowerCase()))
        ));
        if (!hasRequestFlow) {
          fail(`${sitePage.path}: missing copy-paste architecture/request-flow block covering HTTP/API, scheduler, block manager, Qwen engine, and adapter/training path`);
        }

        const missingLinks = expectedArchitectureLinks
          .filter((expectedLink) => !architectureResult.links.some((link) => link.href === expectedLink.href))
          .map((link) => link.label);
        if (missingLinks.length > 0) {
          fail(`${sitePage.path}: missing architecture next-step links: ${missingLinks.join(', ')}`);
        }

        const missingFragments = expectedArchitectureFragments
          .filter((fragment) => !architectureResult.fragmentIds.includes(fragment));
        if (missingFragments.length > 0) {
          fail(`${sitePage.path}: missing stable architecture fragments: ${missingFragments.join(', ')}`);
        }
      }

      if (sitePage.path === troubleshootingPagePath) {
        const troubleshootingResult = await page.evaluate(() => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
          const bodyText = normalize(document.body.innerText);
          const headings = normalize(Array.from(document.querySelectorAll('h1, h2, h3'))
            .map((heading) => heading.textContent || '')
            .join('\n'));
          const copyableCodeBlocks = Array.from(document.querySelectorAll('pre > code'))
            .map((code) => normalize(code.innerText || code.textContent));
          const links = Array.from(document.querySelectorAll('a[href]')).map((link) => ({
            href: link.getAttribute('href'),
            text: normalize(link.textContent),
          }));
          const fragmentIds = Array.from(document.querySelectorAll('[id]'))
            .map((element) => element.id);

          return {
            bodyText,
            headings,
            copyableCodeBlocks,
            links,
            fragmentIds,
          };
        });

        const missingSections = expectedTroubleshootingReaderSections
          .filter((section) => !section.terms.every((term) => {
            const normalizedTerm = term.toLowerCase();
            return troubleshootingResult.headings.includes(normalizedTerm)
              || troubleshootingResult.bodyText.includes(normalizedTerm);
          }))
          .map((section) => section.label);
        if (missingSections.length > 0) {
          fail(`${sitePage.path}: missing troubleshooting cold-reader coverage: ${missingSections.join(', ')}`);
        }

        const missingProbes = expectedTroubleshootingProbeExamples
          .filter((probe) => !troubleshootingResult.copyableCodeBlocks.some((codeBlock) => (
            probe.terms.every((term) => codeBlock.includes(term.toLowerCase()))
          )))
          .map((probe) => probe.label);
        if (missingProbes.length > 0) {
          fail(`${sitePage.path}: missing troubleshooting first-run probes: ${missingProbes.join(', ')}`);
        }

        const missingLinks = expectedTroubleshootingLinks
          .filter((expectedLink) => !troubleshootingResult.links.some((link) => link.href === expectedLink.href))
          .map((link) => link.label);
        if (missingLinks.length > 0) {
          fail(`${sitePage.path}: missing troubleshooting next-step links: ${missingLinks.join(', ')}`);
        }

        const missingFragments = expectedTroubleshootingFragments
          .filter((fragment) => !troubleshootingResult.fragmentIds.includes(fragment));
        if (missingFragments.length > 0) {
          fail(`${sitePage.path}: missing stable troubleshooting fragments: ${missingFragments.join(', ')}`);
        }
      }
    }
  } finally {
    await browser.close();
  }
}

runSmoke().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
