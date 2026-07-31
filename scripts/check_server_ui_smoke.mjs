#!/usr/bin/env node
import http from 'node:http';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import process from 'node:process';
import { tmpdir } from 'node:os';
import vm from 'node:vm';

const repoRoot = resolve(import.meta.dirname, '..');
const uiDir = resolve(repoRoot, 'crates/kiln-server/src/ui');
const uiIndexPath = resolve(uiDir, 'index.html');
const uiStylesPath = resolve(uiDir, 'styles.css');
const uiDemoJsPath = resolve(uiDir, 'demo.js');
const uiAppFragmentPaths = [
  'shell.js',
  'adapters.js',
  'training.js',
  'playground.js',
  'evaluations.js',
  'command_palette.js',
  'charts.js',
  'adapter_drill.js',
  'training_drill.js',
  'playground_compare.js',
  'terminal.js',
  'distillation.js',
  'agents.js',
  'preflight.js',
  'bootstrap.js',
].map((name) => resolve(uiDir, 'app', name));
const thinkingBudgetContractPath = resolve(repoRoot, 'contracts/thinking-budget-v1.conformance.json');
const thinkingBudgetContract = JSON.parse(await readFile(thinkingBudgetContractPath, 'utf8'));
const GIB = 1024 ** 3;
const uiVendorDir = resolve(uiDir, 'vendor');
const uiVendorFiles = {
  'xterm.js': 'application/javascript',
  'xterm.css': 'text/css',
  'xterm-addon-fit.js': 'application/javascript',
};
const expectedHeaderHelpLinks = [
  ['Quickstart', 'https://ericflo.github.io/kiln/quickstart.html'],
  ['OpenEnv Guide', 'https://ericflo.github.io/kiln/docs/openenv/'],
  ['GRPO Guide', 'https://ericflo.github.io/kiln/grpo.html'],
  ['API Reference', 'https://ericflo.github.io/kiln/api.html'],
  ['CLI Reference', 'https://ericflo.github.io/kiln/cli.html'],
  ['Demo', 'https://ericflo.github.io/kiln/demo/'],
  ['Troubleshooting', 'https://ericflo.github.io/kiln/troubleshooting.html'],
];
const forbiddenPublicityTerms = [
  'launch post',
  'announcement',
  'press release',
  'marketing',
  'outreach',
  'social media',
  'twitter',
  'x-twitter',
  'hacker news',
  'hn launch',
  'lobste.rs',
  'localllama',
  'discord',
  'reddit',
  'product hunt',
  'community post',
  'community launch',
  'community announcement',
  'external community',
];

function fail(message) {
  throw new Error(message);
}

async function readUiAppBundle() {
  const fragments = await Promise.all(
    uiAppFragmentPaths.map((path) => readFile(path, 'utf8')),
  );
  return `(function() {\n'use strict';\n\n${fragments.join('')}\n})();\n`;
}

function checkThinkingBudgetParserContract(source) {
  const parserStart = source.indexOf('function strictThinkingBudgetInteger(raw, max)');
  const parserEnd = source.indexOf('function thinkingBudgetInputRaw(', parserStart);
  if (parserStart < 0 || parserEnd < 0) fail('Server thinking-budget parsers are missing');

  const context = vm.createContext({});
  vm.runInContext(`${source.slice(parserStart, parserEnd)}\nthis.parsers = { strictThinkingBudgetInteger, strictThinkingBudgetMilliseconds };`, context);
  const integerCases = new Map([
    ['0', 0], ['0002', 2], ['131072', 131072],
    ['1.5', null], ['1e2', null], ['+1', null], ['-1', null], ['131073', null],
  ]);
  for (const [raw, expected] of integerCases) {
    const actual = context.parsers.strictThinkingBudgetInteger(raw, 131072);
    if (actual !== expected) {
      fail(`Server token parser returned ${String(actual)} for ${JSON.stringify(raw)}; expected ${String(expected)}`);
    }
  }

  const millisecondCases = new Map([
    ['0', 0], ['.001', 1], ['0.010', 10], ['1.25', 1250], ['86400', 86_400_000],
    ['1.0001', null], ['1e2', null], ['+1', null], ['-1', null], ['1.', null],
    ['86400.001', null],
  ]);
  for (const [raw, expected] of millisecondCases) {
    const actual = context.parsers.strictThinkingBudgetMilliseconds(raw, 86_400_000);
    if (actual !== expected) {
      fail(`Server time parser returned ${String(actual)} for ${JSON.stringify(raw)}; expected ${String(expected)}`);
    }
  }
}

function checkRuntimeConfigSchemaContract(source) {
  const rendererStart = source.indexOf('function graphUnavailableLabel(reason)');
  const rendererEnd = source.indexOf('async function loadRuntimeConfig(', rendererStart);
  if (rendererStart < 0 || rendererEnd < 0) fail('Server runtime-config renderer is missing');
  const renderer = source.slice(rendererStart, rendererEnd);

  for (const field of [
    'physical_capacity_gib',
    'configured_capacity_gib',
    'effective_capacity_gib',
    'usable_after_governor_floor_gib',
    'capacity_limit_gib',
    'floor_gib',
    'probe_ms',
    'reclaim_mode_requested',
    'reclaim_mode_effective',
    'reclaim_disabled_by_serving_profile',
    'cfg.rocm_graphs',
    'cfg.rocm_graphs_unavailable_reason',
    'cfg.rocm_graph_telemetry',
    'cfg.rocm_graph_telemetry_unavailable_reason',
    'rocmGraphTelemetry.current_phase',
    'rocmGraphTelemetry.current_phase_elapsed_micros',
    'runtime_device',
    'model_weight_device',
    'native_training_supported',
    'native_training_unavailable_reason',
    'train.optimizer_support',
    'optimizerSupport?.schema',
    'optimizerSupport?.backend',
    'optimizerSupport?.device',
    'optimizerSupport?.base_weight_dtype',
    'optimizerSupport?.resolved_lora_parameter_dtype',
    'optimizerSupport?.rounding_modes',
    'optimizerSupport?.immutable_after_startup',
    'optimizerSupport?.optimizer_tuple_kinds',
    'optimizerSupport?.workloads',
    "workloadSummary('distill_refresh')",
    'entry.allowed_optimizer_kinds',
    'entry?.backend_implementation?.supported',
    'entry?.backend_implementation?.native_device_hook',
    'muonSupport?.optimizer_tuple?.lora_rank',
    'muonRank?.backend_maximum',
    'muonRank?.model_maximum',
    'checkpoint_policy',
    'train.checkpoint_boundary_policy',
    'checkpointBoundaryPolicy.recompute_mode',
    'checkpointBoundaryPolicy.recompute_threshold_tokens',
    'checkpointBoundaryPolicy.anchor_stride',
    'checkpointBoundaryPolicy.cache_target_bytes',
    'training_budget_bytes',
    'training_budget_gib',
    'batching.actor_active',
    'rowwiseDecode.enabled',
    'rowwiseDecode.source',
    'prefixAwareAdmission.enabled',
    'prefixAwareAdmission.source',
    'prefillAdmissionQuantum.configured',
    'prefillAdmissionQuantum.configured_source',
    'prefillAdmissionQuantum.backend_policy',
    'prefillAdmissionQuantum.effective',
    'prefillAdmissionQuantum.effective_source',
    'actorCycleIdle.milliseconds',
    'actorCycleIdle.source',
    'actorCycleIdle.enabled',
    'actorCycleIdle.command_poll_milliseconds',
    'batchingConfiguration.burst_prefill_admission',
    'batchingConfiguration.actor_prefill_tile_alignment_required',
    'cfg.streaming_prefill',
    'streamingPrefill.dispatch',
    'streamingPrefill.threshold_tokens',
    'streamingPrefill.tile_tokens',
    'streamingPrefill.tape_tile_tokens',
    'streamingPrefill.detached_full_attn_tile_tokens',
    'streamingPrefill.detached_full_attn_boundary_tile_tokens',
    'streamingPrefill.detached_full_attn_tape_replay_tile_tokens',
    'streamingPrefill.last_token_lm_head',
    'streamingDispatch.configured_mode',
    'streamingDispatch.configured_source',
    'streamingDispatch.backend_policy',
    'streamingDispatch.effective',
    'streamingDispatch.effective_source',
    'streamingThreshold.configured',
    'streamingThreshold.configured_source',
    'streamingThreshold.override_applied_to_backend_auto_policy',
    'streamingBaseTile.configured_source',
    'streamingBaseTile.backend_policy',
    'streamingBaseTile.effective',
    'streamingBaseTile.effective_source',
    'streamingTapeTile.configured_source',
    'streamingTapeTile.backend_policy',
    'streamingTapeTile.effective',
    'streamingTapeTile.effective_source',
    'streamingDetachedTile.configured_source',
    'streamingDetachedTile.backend_policy',
    'streamingDetachedTile.effective',
    'streamingDetachedTile.effective_source',
    'streamingBoundaryTile.backend_policy',
    'streamingBoundaryTile.effective',
    'streamingBoundaryTile.effective_source',
    'streamingReplayTile.backend_policy',
    'streamingReplayTile.effective',
    'streamingReplayTile.effective_source',
    'streamingLastTokenLmHead.configured',
    'streamingLastTokenLmHead.configured_source',
    'streamingLastTokenLmHead.effective',
    'streamingLastTokenLmHead.effective_source',
    'streamingPrefill.immutable_after_startup',
    'streamingPrefill.restart_required_to_change',
  ]) {
    if (!renderer.includes(field)) fail(`Runtime-config renderer does not consume ${field}`);
  }
  for (const stale of [
    'detected_gb',
    'total_vram_gb',
    'training_budget_gb',
    'KILN_GPU_MEMORY_GB',
    'KILN_NUM_BLOCKS',
    'KILN_GRAD_CHECKPOINT_SEGMENTS',
  ]) {
    if (renderer.includes(stale)) fail(`Runtime-config renderer still depends on stale contract wording ${stale}`);
  }

  // Newer policy diagnostics were added after the original runtime-config
  // response. Older servers and partial diagnostic payloads must keep the
  // dashboard usable while rendering unknown values, never throw during the
  // first paint or leak JavaScript's `undefined` into operator-facing text.
  const context = vm.createContext({
    escapeHtml: (value) => String(value),
    runtimeConfigRow: (label, valueHtml) => `${label}:${valueHtml}`,
    isFinite,
  });
  vm.runInContext(`${renderer}\nthis.renderRuntimeConfigBody = renderRuntimeConfigBody;`, context);
  for (const [label, config] of [
    ['missing response fields', {}],
    ['actor-only batching response', {
      batching: {
        configuration: {},
        actor_active: true,
      },
    }],
    ['legacy null effective Muon maximum', {
      training: {
        optimizer_support: {
          schema: { id: 'kiln.training-optimizer-support', version: 1 },
          backend: 'cpu',
          device: 'cpu',
          base_weight_dtype: 'f32',
          resolved_lora_parameter_dtype: 'f32',
          immutable_after_startup: true,
          rounding_modes: ['round_to_nearest'],
          optimizer_tuple_kinds: [],
          workloads: [{ workload: 'sft', supported: false, unavailable_reason: 'CPU tape unavailable', allowed_optimizer_kinds: [] }],
          optimizers: [{
            kind: 'muon',
            backend_implementation: { supported: true, route: 'portable_reference', native_device_hook: false, parameter_dtypes: ['f32'] },
            optimizer_tuple: { supported: false, unavailable_reason: 'legacy response omitted effective maximum', lora_rank: { minimum: 2, maximum: null, backend_maximum: null, model_maximum: 256, live_memory_admission_required: true } },
          }],
        },
      },
    }],
  ]) {
    let html;
    try {
      html = context.renderRuntimeConfigBody(config);
    } catch (error) {
      fail(`Runtime-config renderer crashed for ${label}: ${error.message}`);
    }
    if (/undefined/.test(html)) fail(`Runtime-config renderer exposed undefined for ${label}`);
    if (!/Primary actor/.test(html)) {
      fail(`Runtime-config renderer omitted actor ownership for ${label}`);
    }
    if (!/Streaming prefill/.test(html)) {
      fail(`Runtime-config renderer omitted streaming-prefill diagnostics for ${label}`);
    }
    if (label === 'legacy null effective Muon maximum') {
      if (!/Muon rank:<strong>2\+<\/strong>/.test(html)
        || !/Muon rank ceilings:<strong>backend none · model 256<\/strong>/.test(html)
        || /dynamic/.test(html)) {
        fail(`Runtime-config renderer should display a null legacy effective maximum as 2+ while preserving the backend/model ceilings, got ${html}`);
      }
    }
  }
}

function checkTrainingOptimizerSupportContract(appSource, indexSource, demoSource) {
  for (const term of [
    'kiln.training-optimizer-support',
    'function updateTrainingOptimizerSupport(cfg)',
    'entry.optimizer_tuple',
    'candidate.optimizer_tuple_kinds',
    'candidate.workloads',
    'candidate.immutable_after_startup === true',
    'candidate.rounding_modes.length === 1',
    "candidate.rounding_modes[0] === 'round_to_nearest'",
    'nonEmptyIdentity(candidate.backend)',
    'nonEmptyIdentity(candidate.device)',
    'nonEmptyIdentity(candidate.base_weight_dtype)',
    'nonEmptyIdentity(candidate.resolved_lora_parameter_dtype)',
    'function validOptimizerRankContract(rank)',
    'Number.isSafeInteger(value) && value > 0',
    'rank.maximum === effectiveMaximum',
    'candidate.optimizers.every(entry => validOptimizerRankContract',
    'option.disabled',
    'function applyOptimizerRankBounds(input, entry)',
    'function trainingOptimizerAdmissionState(workload, kind, rawRank)',
    'function requireTrainingOptimizerAdmission(workload, kind, rank, modeLabel)',
    "if (!['muon', 'adam_w', 'sgd'].includes(value))",
    'Optimizer capability details are still loading',
    'function markTrainingOptimizerSupportFetchFailed(error)',
    'trainingOptimizerSupportSnapshot = null',
    'runtimeConfigLoaded = false',
    'if (runtimeConfigLoaded && runtimeConfigSnapshot && !force) return;',
    'function runtimeConfigFailureHtml(error)',
    'function updateOpdSubmitState()',
    '|| !optimizerState.ready',
    'live_memory_admission_required',
    "optimizerSupportEntry('muon')",
    'round-to-nearest',
    "requireTrainingOptimizerAdmission('sft', 'muon', 8, 'Corrections SFT')",
    "requireTrainingOptimizerAdmission('distill_refresh', 'muon', 16, 'Distill refresh')",
    "requireTrainingOptimizerAdmission('opd', 'muon', form.rank.value, 'Boost')",
    "requireTrainingOptimizerAdmission('opd', 'muon', 16, 'Distill merge')",
    "requireTrainingOptimizerAdmission('opd', 'muon', 16, 'Self-improvement')",
    'function recipeRunAdmissionState(button)',
    "button?.dataset.recipeAdmissionSupported !== 'true'",
  ]) {
    if (!appSource.includes(term)) fail(`Training optimizer product support is missing ${term}`);
  }
  for (const forbidden of [
    'server_execution',
    'select.value = replacement',
    'input.value = String(minimum)',
    'input.value = String(maximum)',
    "form.optimizer.value || 'muon'",
    "document.getElementById('sft-optimizer')?.value || 'muon'",
    "document.getElementById('grpo-optimizer')?.value || 'muon'",
  ]) {
    if (appSource.includes(forbidden)) fail(`Training optimizer UI must not retain ${forbidden}`);
  }
  const assertGuardBefore = (label, startMarker, endMarker, guardMarker, sideEffectMarker) => {
    const start = appSource.indexOf(startMarker);
    const end = appSource.indexOf(endMarker, start + startMarker.length);
    if (start < 0 || end < 0) fail(`${label} training handler is missing`);
    const handler = appSource.slice(start, end);
    const guard = handler.indexOf(guardMarker);
    const sideEffect = handler.indexOf(sideEffectMarker);
    if (guard < 0 || sideEffect < 0 || guard >= sideEffect) {
      fail(`${label} must enforce optimizer admission before ${sideEffectMarker}`);
    }
  };
  assertGuardBefore('Corrections', 'async function trainFromCorrections()', '// Resolve an in-flight corrections-train receipt', "requireTrainingOptimizerAdmission('sft', 'muon', 8", 'corrFlushToServer(trainable)');
  assertGuardBefore('SFT', "document.getElementById('sft-form').addEventListener('submit'", '// --- GRPO Form ---', "requireTrainingOptimizerAdmission('sft'", "api('/v1/train/sft'");
  assertGuardBefore('GRPO', "document.getElementById('grpo-form').addEventListener('submit'", '// --- Chat ---', "requireTrainingOptimizerAdmission('grpo'", "api('/v1/train/grpo'");
  assertGuardBefore('OpenEnv', "document.getElementById('openenv-form')?.addEventListener('submit'", 'syncOpenEnvKind();', "requireTrainingOptimizerAdmission('grpo'", "api('/v1/openenv/runs'");
  assertGuardBefore('OPD', "document.getElementById('opd-form')?.addEventListener('submit'", '// --- Distill / Refresh', "requireTrainingOptimizerAdmission('opd'", "api('/v1/train/opd'");
  assertGuardBefore('Refresh', "document.getElementById('distill-refresh-form')?.addEventListener('submit'", '// --- Distill / Pump', "requireTrainingOptimizerAdmission('distill_refresh'", "api('/v1/distill/refresh'");
  assertGuardBefore('Pump', "document.getElementById('distill-pump-form')?.addEventListener('submit'", '// --- Distill / Merge', "requireTrainingOptimizerAdmission('opd'", "api('/v1/distill/pump'");
  assertGuardBefore('Merge', "document.getElementById('distill-merge-form')?.addEventListener('submit'", '// --- Distill / Self', "requireTrainingOptimizerAdmission('opd'", "api('/v1/adapters/distill_merge'");
  assertGuardBefore('Self', "document.getElementById('distill-self-form')?.addEventListener('submit'", '// --- Cache', "requireTrainingOptimizerAdmission('opd'", "api('/v1/distill/self'");
  assertGuardBefore('Recipe', "document.addEventListener('click', async (ev) => {\n  const btn = ev.target.closest('[data-recipe-run]')", '// --- Submit OPD', 'recipeRunAdmissionState(btn)', "btn.dataset.recipeBusy = 'true'");
  for (const id of [
    'corr-optimizer-support',
    'sft-optimizer-support',
    'grpo-optimizer-support',
    'openenv-optimizer-support',
    'opd-optimizer-support',
    'refresh-optimizer-support',
    'pump-optimizer-support',
    'merge-optimizer-support',
    'self-optimizer-support',
  ]) {
    if (!indexSource.includes(`id="${id}"`)) fail(`Training optimizer status node #${id} is missing`);
  }
  for (const id of [
    'openenv-credential-ids',
    'openenv-environment-eval-mode',
    'openenv-environment-eval-groups',
    'openenv-environment-eval-group-size',
    'openenv-environment-gate-floor',
    'openenv-environment-gate-improvement',
  ]) {
    if (!indexSource.includes(`id="${id}"`)) fail(`OpenEnv held-out evaluation control #${id} is missing`);
  }
  for (const term of [
    'request.environment_eval = {',
    'environmentEvaluation?.evidence',
    "state === 'environment_evaluating'",
    'exact_sign_test_p_value',
  ]) {
    if (!appSource.includes(term)) fail(`OpenEnv held-out evaluation UI is missing ${term}`);
  }
  for (const describedBy of [
    'aria-describedby="corr-optimizer-support"',
    'aria-describedby="sft-optimizer-support"',
    'aria-describedby="grpo-optimizer-support"',
    'aria-describedby="openenv-optimizer-support"',
    'aria-describedby="opd-optimizer-support"',
    'aria-describedby="refresh-optimizer-support"',
    'aria-describedby="pump-optimizer-support"',
    'aria-describedby="merge-optimizer-support"',
    'aria-describedby="self-optimizer-support"',
  ]) {
    if (!indexSource.includes(describedBy)) fail(`Training optimizer controls are missing ${describedBy}`);
  }
  const loadingStatusCount = (indexSource.match(/Optimizer capability details are still loading\. Training remains disabled\./g) || []).length;
  if (loadingStatusCount < 9) {
    fail(`Every training-producing surface must start with visible loading/disabled optimizer copy; found ${loadingStatusCount}`);
  }
  for (const formId of [
    'sft-form',
    'grpo-form',
    'openenv-form',
    'opd-form',
    'distill-refresh-form',
    'distill-pump-form',
    'distill-merge-form',
    'distill-self-form',
  ]) {
    const start = indexSource.indexOf(`<form id="${formId}"`);
    const end = indexSource.indexOf('</form>', start);
    const form = start >= 0 && end >= 0 ? indexSource.slice(start, end) : '';
    if (!/<button[^>]+type="submit"[^>]+disabled/.test(form)) {
      fail(`#${formId} must be disabled in static HTML until optimizer capabilities load`);
    }
  }
  for (const controlId of ['corr-train', 'sft-optimizer', 'sft-rank', 'grpo-optimizer', 'grpo-rank', 'openenv-lora-rank', 'opd-rank', 'pump-rank']) {
    const control = indexSource.match(new RegExp(`<[^>]+id="${controlId}"[^>]*>`))?.[0] || '';
    if (!/\sdisabled(?:\s|>)/.test(control)) fail(`#${controlId} must start disabled in static HTML`);
  }
  for (const term of [
    "id: 'kiln.training-optimizer-support'",
    "optimizer_tuple_kinds: ['muon', 'adam_w', 'sgd']",
    "workload: 'distill_refresh'",
    'optimizer_tuple:',
    'resolved_lora_parameter_dtype',
    'backend_implementation_rounding_modes',
    'live_memory_admission_required: true',
    'backend_maximum:',
    'model_maximum:',
    'admission: { supported: true, unavailable_reason: null }',
    "admission: { supported: false, unavailable_reason: 'distill_refresh is unavailable until admission pins separate exact SFT and OPD phase plans, prepares the exact SFT rows, and reserves the maximum sequential working set' }",
  ]) {
    if (!demoSource.includes(term)) fail(`Demo optimizer support fixture is missing ${term}`);
  }
}

function checkTrainingOptimizerRankValidator(appSource) {
  const validatorStart = appSource.indexOf('function validOptimizerRankContract(rank)');
  const validatorEnd = appSource.indexOf('function updateTrainingOptimizerSupport(cfg)', validatorStart);
  if (validatorStart < 0 || validatorEnd < 0) fail('Training optimizer rank validator is missing');
  const context = vm.createContext({});
  vm.runInContext(`${appSource.slice(validatorStart, validatorEnd)}\nthis.validate = validOptimizerRankContract;`, context);

  const valid = [
    { minimum: 2, maximum: 48, backend_maximum: 48, model_maximum: 1024, live_memory_admission_required: true },
    { minimum: 1, maximum: 1024, backend_maximum: null, model_maximum: 1024, live_memory_admission_required: true },
    { minimum: 2, maximum: 32, backend_maximum: 64, model_maximum: 32, live_memory_admission_required: true },
  ];
  for (const rank of valid) {
    if (!context.validate(rank)) fail(`Valid optimizer rank contract was rejected: ${JSON.stringify(rank)}`);
  }

  const invalid = [
    { minimum: 2, maximum: null, backend_maximum: 48, model_maximum: 1024, live_memory_admission_required: true },
    { minimum: 2, maximum: 49, backend_maximum: 48, model_maximum: 1024, live_memory_admission_required: true },
    { minimum: 2, maximum: 48, backend_maximum: 64, model_maximum: 1024, live_memory_admission_required: true },
    { minimum: 49, maximum: 48, backend_maximum: 48, model_maximum: 1024, live_memory_admission_required: true },
    { minimum: 2, maximum: Number.MAX_SAFE_INTEGER + 1, backend_maximum: null, model_maximum: Number.MAX_SAFE_INTEGER + 1, live_memory_admission_required: true },
    { minimum: 2, maximum: 48, backend_maximum: 48, model_maximum: 1024, live_memory_admission_required: false },
    { minimum: 2, maximum: 48, backend_maximum: 48, model_maximum: 1024 },
  ];
  for (const rank of invalid) {
    if (context.validate(rank)) fail(`Malformed optimizer rank contract was accepted: ${JSON.stringify(rank)}`);
  }
}

async function loadPuppeteer() {
  try {
    const module = await import('puppeteer');
    return module.default || module;
  } catch (error) {
    if (error?.code !== 'ERR_MODULE_NOT_FOUND') throw error;
  }

  const installDir = '/tmp/kiln-server-ui-smoke-puppeteer';
  const packageJson = join(installDir, 'package.json');
  const puppeteerPackageJson = join(installDir, 'node_modules/puppeteer/package.json');
  await mkdir(installDir, { recursive: true });
  if (!existsSync(packageJson)) {
    await writeFile(packageJson, '{"private":true,"type":"commonjs"}\n');
  }
  if (!existsSync(puppeteerPackageJson)) {
    execFileSync('npm', ['install', '--silent', '--no-save', 'puppeteer@latest'], {
      cwd: installDir,
      stdio: 'inherit',
      env: { ...process.env, PUPPETEER_SKIP_DOWNLOAD: 'true' },
    });
  }
  const require = createRequire(packageJson);
  return require('puppeteer');
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

function json(res, body) {
  res.writeHead(200, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
}

function text(res, body, contentType = 'text/plain; charset=utf-8') {
  res.writeHead(200, { 'content-type': contentType });
  res.end(body);
}

function apiFailure(res, panelName, path) {
  // Kiln's canonical error shape ({ error: { code, message, hint } }) — the
  // dashboard must render message + hint, never "[object Object]".
  res.writeHead(503, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify({
    error: {
      code: 'smoke_failure',
      message: `${panelName} smoke failure from ${path}`,
      hint: 'Smoke hint: retry after startup.',
    },
  }));
}

function apiBadRequest(res, detail) {
  res.writeHead(400, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify({ detail }));
}

async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

async function readBufferBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function validateSftPayload(body) {
  const namedDataset = typeof body?.dataset === 'string';
  if (namedDataset) {
    if (body.dataset !== 'sample-coding-agent') return 'Named SFT should preserve the selected dataset';
    if (body.dataset_split !== 'train') return 'Named SFT should explicitly select the persisted train partition';
    if ('examples' in body || 'dataset_path' in body) return 'Named SFT should submit exactly one data source';
    if (body?.post_eval?.suite !== 'smoke-heldout' || body.post_eval.data_scope !== 'train-set-eval') {
      return 'Named SFT should preserve the deliberate train-set diagnostic label';
    }
    if (body.post_eval.include_baseline !== true) return 'Named SFT diagnostic should compare with the base model';
  } else {
    if (!Array.isArray(body?.examples) || body.examples.length !== 1) return 'SFT examples should be a one-item array from the sample payload';
    const messages = body.examples[0]?.messages;
    if (!Array.isArray(messages) || messages.length !== 2) return 'SFT sample example should include user and assistant messages';
    if (messages[0]?.role !== 'user' || messages[1]?.role !== 'assistant') return 'SFT sample messages should preserve chat roles';
  }
  if (body?.config?.training_profile !== 'native_online_lora_v1') return 'SFT should submit the explicit native_online_lora_v1 profile';
  const expectedOutput = namedDataset ? 'sample-coding-agent-sft' : 'sft-adapter';
  if (body?.config?.output_name !== expectedOutput) return 'SFT output_name should be nested under config and follow the selected source';
  if (body?.config?.auto_load !== true) return 'SFT auto_load should be true by default';
  if ('learning_rate' in (body?.config || {})) return 'SFT learning_rate should be omitted when the field is blank (server resolves the per-optimizer default)';
  if (body?.config?.epochs !== 3) return 'SFT epochs should be numeric and nested under config';
  if (body?.config?.lora_rank !== 8) return 'SFT lora_rank should be numeric and nested under config';
  if (body?.config?.lora_alpha !== 16) return 'SFT lora_alpha should pair with rank (2×rank, capped at 32) so the trainer scale gate passes';
  if (body?.config?.optimizer?.kind !== 'muon') return 'SFT should submit the exact selected Muon optimizer tuple';
  if (body?.config?.checkpoint_interval !== 2) return 'SFT checkpoint_interval should be a positive integer nested under config';
  if ('resume_checkpoint' in (body?.config || {})) return 'Fresh SFT submission should omit resume_checkpoint when the field is blank';
  if ('output_name' in body || 'adapter_name' in body || 'num_epochs' in body) return 'SFT payload should not use stale top-level training config fields';
  return null;
}

function validateGrpoPayload(body) {
  if (!Array.isArray(body?.groups) || body.groups.length !== 1) return 'GRPO groups should be a one-item array from the sample payload';
  const group = body.groups[0];
  if (!Array.isArray(group?.messages) || group.messages[0]?.role !== 'user') return 'GRPO sample group should preserve prompt messages';
  if (!Array.isArray(group?.completions) || group.completions.length !== 2) return 'GRPO sample group should include scored completions';
  if (!group.completions.every((completion) => typeof completion.text === 'string' && isFiniteNumber(completion.reward))) return 'GRPO completions should include text and numeric rewards';
  if (body?.config?.output_name !== 'grpo-adapter') return 'GRPO output_name should be nested under config';
  if (body?.config?.auto_load !== true) return 'GRPO auto_load should be true by default';
  if ('learning_rate' in (body?.config || {})) return 'GRPO learning_rate should be omitted when the field is blank (server resolves the per-optimizer default)';
  if (!isFiniteNumber(body?.config?.kl_coeff) || body.config.kl_coeff !== 0.1) return 'GRPO kl_coeff should be numeric';
  if (body?.config?.lora_rank !== 8) return 'GRPO lora_rank should be numeric and nested under config';
  if (body?.config?.lora_alpha !== 16) return 'GRPO lora_alpha should pair with rank (2×rank, capped at 32) so the trainer scale gate passes';
  if (body?.config?.optimizer?.kind !== 'muon') return 'GRPO should submit the exact selected Muon optimizer tuple';
  if (body?.config?.checkpoint_interval !== 3) return 'GRPO checkpoint_interval should be a positive integer nested under config';
  if ('resume_checkpoint' in (body?.config || {})) return 'Fresh GRPO submission should omit resume_checkpoint when the field is blank';
  if ('epochs' in (body?.config || {}) || 'output_name' in body || 'adapter_name' in body || 'num_epochs' in body) return 'GRPO payload should not use stale SFT/top-level training config fields';
  return null;
}

function validateOpdPayload(body) {
  if (!Array.isArray(body?.prompts) || body.prompts.length !== 2) return 'OPD prompts should be the two-item sample array';
  if (!body.prompts.every((prompt) => Array.isArray(prompt?.messages) && prompt.messages.length > 0)) return 'OPD sample prompts should preserve messages';
  if (body?.teacher !== 'teacher-v1') return 'OPD should submit the selected registered teacher';
  if (body?.config?.output_name !== 'opd-adapter') return 'OPD output_name should be nested under config';
  if (body?.config?.training_mode !== 'on_policy') return 'OPD browser form should submit the on-policy mode it presents';
  if (body?.config?.lora_rank !== 32) return 'OPD lora_rank should be numeric and nested under config';
  if (body?.config?.optimizer?.kind !== 'muon') return 'OPD should submit the exact gated Muon optimizer tuple';
  if (body?.config?.checkpoint_interval !== 25) return 'OPD checkpoint_interval should preserve the exact-resume default';
  if ('resume_checkpoint' in (body?.config || {})) return 'Fresh OPD submission should omit resume_checkpoint when the field is blank';
  if (body?.config?.auto_load !== true) return 'OPD auto_load should be true by default';
  return null;
}

function isPathSafeAdapterDirectoryName(name) {
  return typeof name === 'string'
    && name.length > 0
    && name !== '.'
    && name !== '..'
    && !name.includes('/')
    && !name.includes('\\');
}

function validateAdapterMergePayload(body) {
  if (!Array.isArray(body?.sources) || body.sources.length !== 2) return 'Merge payload should include exactly two sources from the dashboard flow';
  const names = body.sources.map((source) => source?.name);
  if (!names.every((name) => typeof name === 'string' && name.length > 0)) return 'Merge sources should include adapter names';
  if (new Set(names).size !== names.length) return 'Merge sources should be distinct adapters';
  if (!body.sources.every((source) => isFiniteNumber(source?.weight))) return 'Merge source weights should be numeric';
  if (!isPathSafeAdapterDirectoryName(body?.output_name)) return 'Merge output_name should be path-safe';
  if (!['weighted_average', 'ties', 'concat'].includes(body?.mode)) return 'Merge mode should be a supported dashboard mode';
  if (body.mode === 'ties') {
    if (!isFiniteNumber(body?.density) || body.density <= 0 || body.density > 1) return 'TIES merge should include numeric density in (0, 1]';
  } else if ('density' in body) {
    return 'Density should only be sent for TIES merges';
  }
  return null;
}

function parseMultipartFormData(contentType, body) {
  const match = /(?:^|;)\s*boundary=(?:("[^"]+")|([^;]+))/i.exec(contentType || '');
  if (!match) return null;
  const boundary = (match[1] || match[2]).replace(/^"|"$/g, '');
  const parts = [];
  const delimiter = Buffer.from(`--${boundary}`);
  let searchOffset = 0;
  while (true) {
    const start = body.indexOf(delimiter, searchOffset);
    if (start === -1) break;
    let partStart = start + delimiter.length;
    if (body.subarray(partStart, partStart + 2).toString() === '--') break;
    if (body.subarray(partStart, partStart + 2).toString() === '\r\n') partStart += 2;
    const next = body.indexOf(delimiter, partStart);
    if (next === -1) break;
    let part = body.subarray(partStart, next);
    if (part.subarray(part.length - 2).toString() === '\r\n') part = part.subarray(0, part.length - 2);
    const headerEnd = part.indexOf(Buffer.from('\r\n\r\n'));
    if (headerEnd !== -1) {
      const headers = part.subarray(0, headerEnd).toString('utf8');
      const content = part.subarray(headerEnd + 4);
      const disposition = /content-disposition:\s*form-data;([^\r\n]+)/i.exec(headers)?.[1] || '';
      const name = /name="([^"]+)"/i.exec(disposition)?.[1];
      const filename = /filename="([^"]*)"/i.exec(disposition)?.[1];
      if (name) parts.push({ name, filename, content });
    }
    searchOffset = next;
  }
  return parts;
}

function validateAdapterUploadRequest(req, body) {
  if (req.method !== 'POST') return { status: 405, detail: 'Use POST for adapter upload' };
  const contentType = req.headers['content-type'] || '';
  if (!/^multipart\/form-data\b/i.test(contentType)) {
    return { status: 400, detail: 'Adapter upload should use multipart/form-data' };
  }
  const parts = parseMultipartFormData(contentType, body);
  if (!parts) return { status: 400, detail: 'Adapter upload should include a multipart boundary' };
  const name = parts.find((part) => part.name === 'name')?.content.toString('utf8').trim();
  const archive = parts.find((part) => part.name === 'archive');
  if (!isPathSafeAdapterDirectoryName(name)) return { status: 400, detail: 'Adapter upload name should be path-safe' };
  if (!archive?.filename) return { status: 400, detail: 'Adapter upload should include an archive file field' };
  if (archive.content.length === 0) return { status: 400, detail: 'Adapter upload archive should be non-empty' };
  return { name, archiveSize: archive.content.length };
}

function validateExistingAdapterName(name, availableAdapters, action) {
  if (!isPathSafeAdapterDirectoryName(name)) return `${action} adapter name should be path-safe`;
  if (!availableAdapters.some((adapter) => adapter.name === name)) return `${action} adapter should already exist`;
  return null;
}

function parseAdapterRoute(pathname) {
  const match = /^\/v1\/adapters\/([^/]+)(?:\/(download|detail|receipt))?$/.exec(pathname);
  if (!match) return null;
  if (['load', 'unload', 'upload', 'merge', 'distill_merge'].includes(match[1])) return null;
  return { name: decodeURIComponent(match[1]), action: match[2] || null };
}

function adapterNotFound(res, name) {
  // Mirrors error.rs ApiError::adapter_not_found.
  res.writeHead(404, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify({
    error: {
      code: 'adapter_not_found',
      message: `Adapter '${name}' does not exist`,
      hint: 'List available adapters with GET /v1/adapters.',
    },
  }));
}

// §8.11 reproducibility receipt for adapter-alpha — field-for-field the
// kiln-train/src/receipt.rs AdapterReceipt shape (the /receipt endpoint
// serializes that struct directly). Other adapters 404, exactly like
// AdapterReceipt::read_from_adapter_dir returning Ok(None) for uploaded
// or pre-receipt adapters.
const smokeAdapterReceipt = {
  schema_version: 1,
  adapter: 'adapter-alpha',
  produced_at: '2026-06-10T18:30:00Z',
  kiln_version: '0.1.0',
  kernel_versions: { 'kiln-opd-loss-kernel': '0.1.0' },
  seed: 4218,
  source_kind: 'opd',
  teacher: { alias: 'qwen3.6-27b@openrouter', model_id: 'qwen/qwen-3.6-27b' },
  prompts: {
    source: 'kiln-canonical:math_reasoning:v3',
    manifest_hash: 'f00dfeedf00dfeedf00dfeedf00dfeedf00dfeedf00dfeedf00dfeedf00dfeed',
    count: 128,
  },
  hyperparameters: {
    learning_rate: 0.0001,
    lora_rank: 16,
    lora_alpha: 32,
    epochs: 2,
    temperature: 0.7,
    top_k: 64,
    top_p: 0.95,
  },
  diagnostic_summary: { overlap_ratio_final: 0.91, rep_rate_max: 0.0, final_loss: 0.0421 },
  post_eval: { 'math-reasoning-suite': 0.84 },
};


// Mirrors api/eval.rs upload_dataset: multipart fields `name`, `format`
// (sft_chat | grpo_groups | raw), optional `description`, and `file` whose
// every JSONL line must parse into the SftConversation contract — messages[]
// of { role, content?, tool_calls?, name?, tool_call_id? }.
function validateEvalDatasetUploadRequest(req, body) {
  if (req.method !== 'POST') return { status: 405, detail: 'Use POST for dataset upload' };
  const contentType = req.headers['content-type'] || '';
  if (!/^multipart\/form-data\b/i.test(contentType)) {
    return { status: 400, detail: 'Dataset upload should use multipart/form-data' };
  }
  const parts = parseMultipartFormData(contentType, body);
  if (!parts) return { status: 400, detail: 'Dataset upload should include a multipart boundary' };
  const name = parts.find((part) => part.name === 'name')?.content.toString('utf8').trim();
  const format = parts.find((part) => part.name === 'format')?.content.toString('utf8').trim() || 'sft_chat';
  const description = parts.find((part) => part.name === 'description')?.content.toString('utf8') || null;
  const file = parts.find((part) => part.name === 'file');
  if (!isPathSafeAdapterDirectoryName(name) || name.includes('..')) return { status: 400, detail: 'Dataset name should be path-safe' };
  if (!['sft_chat', 'sft', 'grpo_groups', 'grpo', 'raw'].includes(format)) return { status: 400, detail: `unknown format \`${format}\`` };
  if (!file) return { status: 400, detail: 'Dataset upload should include a file field' };
  const jsonl = file.content.toString('utf8');
  const lines = jsonl.split('\n').filter((line) => line.trim().length > 0);
  if (lines.length === 0) return { status: 400, detail: 'Dataset file should contain at least one JSONL row' };
  const stats = {
    num_assistant_turns: 0,
    num_with_tool_calls: 0,
    num_tool_messages: 0,
    max_messages_per_conv: 0,
    max_content_chars: 0,
    avg_messages_per_conv: 0,
    sample_role_patterns: [],
  };
  let totalMessages = 0;
  if (format !== 'raw') {
    for (const [index, line] of lines.entries()) {
      let row;
      try {
        row = JSON.parse(line);
      } catch (error) {
        return { status: 400, detail: `line ${index + 1} is not valid JSON: ${error.message}` };
      }
      if (!Array.isArray(row.messages) || row.messages.length === 0) {
        return { status: 400, detail: `line ${index + 1} should have a non-empty messages array` };
      }
      for (const message of row.messages) {
        if (typeof message.role !== 'string' || message.role.length === 0) {
          return { status: 400, detail: `line ${index + 1} has a message without a string role` };
        }
        if (message.content !== undefined && typeof message.content !== 'string') {
          return { status: 400, detail: `line ${index + 1} has a non-string content field` };
        }
        if (message.tool_calls !== undefined && !Array.isArray(message.tool_calls)) {
          return { status: 400, detail: `line ${index + 1} has a non-array tool_calls field` };
        }
        if (message.role === 'tool' && typeof message.tool_call_id !== 'string') {
          return { status: 400, detail: `line ${index + 1} has a tool reply without tool_call_id` };
        }
        if (message.role === 'assistant') {
          stats.num_assistant_turns += 1;
          if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) stats.num_with_tool_calls += 1;
        }
        if (message.role === 'tool') stats.num_tool_messages += 1;
        stats.max_content_chars = Math.max(stats.max_content_chars, (message.content || '').length);
      }
      totalMessages += row.messages.length;
      stats.max_messages_per_conv = Math.max(stats.max_messages_per_conv, row.messages.length);
      if (stats.sample_role_patterns.length < 5) {
        stats.sample_role_patterns.push(row.messages.map((message) => message.role).join(' '));
      }
    }
    stats.avg_messages_per_conv = totalMessages / lines.length;
  }
  const now = new Date().toISOString();
  return {
    manifest: {
      name,
      format: format === 'sft' ? 'sft_chat' : format === 'grpo' ? 'grpo_groups' : format,
      description,
      num_rows: lines.length,
      size_bytes: file.content.length,
      created_at: now,
      updated_at: now,
      corpus_sha256: `sha256:${'1'.repeat(64)}`,
      normalized_corpus_sha256: `sha256:${'2'.repeat(64)}`,
      split_manifest_sha256: `sha256:${'3'.repeat(64)}`,
      split_config: { seed: '0', train_percent: 80, validation_percent: 10 },
      split_counts: {
        train: Math.max(1, lines.length - 2),
        validation: lines.length > 2 ? 1 : 0,
        holdout: lines.length > 1 ? 1 : 0,
      },
      num_groups: 0,
      num_sessions: 0,
      stats,
    },
  };
}

const defaultAvailableAdapters = [
  { name: 'adapter-alpha', active: false, size_bytes: 4096 },
  { name: 'adapter-beta', active: false, size_bytes: 8192 },
];

// Recent-requests row factory for the journey-strip truth assertions. The
// `client` field mirrors the server echoing the X-Kiln-Client header:
// dashboard-originated rows carry `client: 'dashboard'` and must never
// complete the "Agent connected" milestone; external rows (curl, pi) must.
let smokeRowSeq = 0;
function smokeRecentRow(overrides = {}) {
  smokeRowSeq += 1;
  return {
    id: `chatcmpl-smoke-row-${smokeRowSeq}`,
    timestamp_unix_ms: Date.now() - smokeRowSeq * 1000,
    model: 'Qwen3.5-4B',
    prompt_preview: 'smoke prompt',
    completion_preview: 'smoke completion',
    prompt_tokens: 12,
    completion_tokens: 8,
    duration_ms: 120,
    streamed: false,
    finish_reason: 'stop',
    thinking_budget: {
      configured: true,
      max_tokens: 64,
      max_time_ms: 1500,
      tokens_source: 'request',
      time_source: 'server_default',
      applied: true,
      triggered: true,
      trigger: 'tokens',
      closed: true,
      thinking_tokens: 64,
      thinking_time_ms: 800,
    },
    ...overrides,
  };
}

function smokeRequestLatency() {
  return {
    emitted_tokens: 8,
    gap_samples: 7,
    retained_gap_samples: 7,
    gap_samples_truncated: false,
    ttft_ms: 38,
    itl_ms_p50: 9,
    itl_ms_p99: 28,
    itl_ms_p999: 29.8,
    max_itl_ms: 30,
    stall_threshold_ms: 250,
    stall_count: 0,
    unexplained_stall_count: 0,
    stall_reasons: {
      actor_queue: 0,
      actor_admission: 0,
      actor_prefill: 0,
      actor_decode: 0,
      actor_cycle_idle: 0,
      response_delivery: 0,
      handler_queue: 0,
      client_delivery: 0,
      sampling: 0,
      readback: 0,
      gpu_lock_wait: 0,
      graph_capture: 0,
      graph_replay: 0,
      synchronization: 0,
      resize: 0,
      trim: 0,
      adapter: 0,
      training: 0,
      unexplained: 0,
    },
    phases: {
      actor_queue_ms: 4,
      actor_admission_ms: 2,
      tokenization_ms: 1,
      prefill_ms: 290,
      decode_ms: 63,
      actor_cycle_idle_ms: 0,
      sampling_ms: null,
      readback_ms: null,
      response_delivery_ms: 2,
      handler_queue_ms: 1,
      client_delivery_ms: 1,
      gpu_lock_wait_ms: 2,
      graph_capture_ms: null,
      graph_replay_ms: null,
      synchronization_ms: 4,
      resize_ms: null,
      trim_ms: null,
      adapter_ms: null,
      training_ms: null,
      unexplained_ms: 0,
    },
  };
}
const dashboardRow = () => smokeRecentRow({
  user_agent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36',
  client: 'dashboard',
  prompt_preview: 'Reply with the single word: connected',
  completion_preview: 'connected',
});
const curlRow = () => smokeRecentRow({
  user_agent: 'curl/8.7.1',
  prompt_preview: 'hello from the curl tab',
});
const piRow = (overrides = {}) => smokeRecentRow({
  user_agent: 'pi/1.2.0',
  prompt_preview: 'hello from pi',
  ...overrides,
});

function sse(res, chunks) {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  for (const chunk of chunks) {
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  }
  res.write('data: [DONE]\n\n');
  res.end();
}

function smokeThinkingBudgetMetadata(body, { triggered = true, trigger = 'tokens' } = {}) {
  const hasTokens = Object.hasOwn(body, 'thinking_budget_tokens');
  const hasTime = Object.hasOwn(body, 'thinking_budget_ms');
  const maxTokens = hasTokens ? body.thinking_budget_tokens : 8;
  const maxTimeMs = hasTime ? body.thinking_budget_ms : null;
  const tokensSource = hasTokens
    ? (maxTokens === null ? 'request_unlimited' : 'request')
    : 'server_default';
  const timeSource = hasTime
    ? (maxTimeMs === null ? 'request_unlimited' : 'request')
    : 'unlimited';
  const configured = maxTokens !== null || maxTimeMs !== null;
  const metadata = {
    configured,
    applied: configured,
    tokens_source: tokensSource,
    time_source: timeSource,
    triggered: configured && triggered,
  };
  if (maxTokens !== null) metadata.max_tokens = maxTokens;
  if (maxTimeMs !== null) metadata.max_time_ms = maxTimeMs;
  if (configured) {
    if (triggered) metadata.trigger = trigger;
    metadata.closed = true;
    metadata.thinking_tokens = maxTokens ?? 2;
    metadata.thinking_time_ms = 7;
  }
  return metadata;
}

function smokeThinkingBudgetOutcome(metadata) {
  if (!metadata.applied) return undefined;
  return {
    triggered: metadata.triggered,
    ...(metadata.trigger ? { trigger: metadata.trigger } : {}),
    closed: metadata.closed,
    thinking_tokens: metadata.thinking_tokens,
    thinking_time_ms: metadata.thinking_time_ms,
  };
}

function smokeTrainingRuntimeConfig(available) {
  const unavailableReason = 'native Vulkan training is unavailable for CPU-host serving weights';
  const common = {
    checkpoint_policy: { mode: 'explicit_segments', segments: 4 },
    checkpoint_boundary_policy: {
      recompute_mode: 'enabled',
      recompute_threshold_tokens: 4_096,
      anchor_stride: 3,
      cache_target_bytes: 6 * GIB,
    },
    checkpoint_segments: 4,
    checkpoint_segments_source: 'configured',
    checkpointing_enabled: true,
  };
  const schema = { id: 'kiln.training-optimizer-support', version: 1 };
  if (available === 'missing') {
    return {
      runtime_device: 'vulkan:0',
      model_weight_device: 'cpu',
      native_training_supported: false,
      native_training_unavailable_reason: 'Optimizer capability details are unavailable',
      ...common,
    };
  }
  if (!available) {
    return {
      runtime_device: 'vulkan:0',
      model_weight_device: 'cpu',
      native_training_supported: false,
      native_training_unavailable_reason: unavailableReason,
      optimizer_support: {
        schema,
        backend: 'vulkan',
        device: 'vulkan:0',
        base_weight_dtype: 'bf16',
        resolved_lora_parameter_dtype: 'f32',
        immutable_after_startup: true,
        rounding_modes: ['round_to_nearest'],
        backend_implementation_rounding_modes: ['round_to_nearest'],
        optimizer_tuple_kinds: ['muon', 'adam_w', 'sgd'],
        workloads: ['sft', 'grpo', 'opd', 'distill_refresh'].map(workload => ({
          workload,
          supported: false,
          unavailable_reason: unavailableReason,
          allowed_optimizer_kinds: [],
        })),
        optimizers: ['muon', 'adam_w', 'sgd'].map(kind => ({
          kind,
          backend_implementation: { supported: true, route: 'native_device_hook', native_device_hook: true, parameter_dtypes: ['f32', 'bf16'] },
          optimizer_tuple: {
            supported: true,
            unavailable_reason: null,
            lora_rank: kind === 'muon'
              ? { minimum: 2, maximum: 32, backend_maximum: 32, model_maximum: 256, live_memory_admission_required: true }
              : { minimum: 1, maximum: 256, backend_maximum: null, model_maximum: 256, live_memory_admission_required: true },
          },
        })),
      },
      ...common,
    };
  }
  const supported = {
    runtime_device: 'metal:0',
    model_weight_device: 'metal:0',
    native_training_supported: true,
    optimizer_support: {
      schema,
      backend: 'metal',
      device: 'metal:0',
      base_weight_dtype: 'bf16',
      resolved_lora_parameter_dtype: 'bf16',
      immutable_after_startup: true,
      rounding_modes: ['round_to_nearest'],
      backend_implementation_rounding_modes: ['round_to_nearest'],
      optimizer_tuple_kinds: ['muon', 'adam_w'],
      workloads: [
        ...['sft', 'grpo', 'opd'].map(workload => ({
          workload,
          supported: true,
          unavailable_reason: null,
          allowed_optimizer_kinds: ['muon', 'adam_w'],
        })),
        {
          workload: 'distill_refresh',
          supported: false,
          unavailable_reason: 'distill_refresh is unavailable until admission pins separate exact SFT and OPD phase plans, prepares the exact SFT rows, and reserves the maximum sequential working set',
          allowed_optimizer_kinds: [],
        },
      ],
      optimizers: [
        {
          kind: 'muon',
          backend_implementation: { supported: true, route: 'native_device_hook', native_device_hook: true, parameter_dtypes: ['f32', 'bf16'] },
          optimizer_tuple: { supported: true, unavailable_reason: null, lora_rank: { minimum: 2, maximum: 32, backend_maximum: 32, model_maximum: 256, live_memory_admission_required: true } },
        },
        {
          kind: 'adam_w',
          backend_implementation: { supported: true, route: 'native_device_hook', native_device_hook: true, parameter_dtypes: ['f32', 'bf16'] },
          optimizer_tuple: { supported: true, unavailable_reason: null, lora_rank: { minimum: 1, maximum: 256, backend_maximum: null, model_maximum: 256, live_memory_admission_required: true } },
        },
        {
          kind: 'sgd',
          backend_implementation: { supported: false, route: 'unavailable', native_device_hook: false, parameter_dtypes: [] },
          optimizer_tuple: { supported: false, unavailable_reason: 'Metal has no native SGD optimizer hook', lora_rank: { minimum: 1, maximum: 256, backend_maximum: null, model_maximum: 256, live_memory_admission_required: true } },
        },
      ],
    },
    ...common,
  };
  if (available === 'unsupported-schema') supported.optimizer_support.schema.version = 2;
  if (available === 'mutable-policy') supported.optimizer_support.immutable_after_startup = false;
  if (available === 'wrong-rounding') supported.optimizer_support.rounding_modes = ['stochastic'];
  if (available === 'invalid-rank-contract') {
    supported.optimizer_support.optimizers[0].optimizer_tuple.lora_rank.maximum = null;
  }
  if (available === 'missing-tuple-identity') {
    supported.optimizer_support.backend = '';
    supported.optimizer_support.device = '';
    supported.optimizer_support.base_weight_dtype = '';
    supported.optimizer_support.resolved_lora_parameter_dtype = '';
  }
  return supported;
}

async function startServer({
  failDashboardApis = false,
  availableAdapters = defaultAvailableAdapters,
  // Cold-start fixture: /v1/models 503s until healed via setModelsCold(false),
  // then serves `servedModelId`. Distinct from the UI fallback ('Qwen3.5-4B')
  // when the scenario needs to prove the snippets upgraded without reload.
  modelsCold = false,
  servedModelId = 'Qwen3.5-4B',
} = {}) {
  // Mutable so the failure scenario can heal/re-break the APIs mid-run and
  // assert the dashboard recovers (Retry buttons + dedupe-key invalidation).
  const apiState = {
    failDashboardApis,
    failRuntimeConfig: false,
    modelsCold,
    recentRequests: [],
    modelsRequests: 0,
    evalSuites: [],
    trainingOptimizerAvailable: false,
    trainingSubmitRequests: { sft: 0, grpo: 0, openenv: 0, opd: 0, refresh: 0, pump: 0, merge: 0, self: 0, recipe: 0 },
  };
  const uiHtml = await readFile(uiIndexPath, 'utf8');
  const uiStyles = await readFile(uiStylesPath, 'utf8');
  const uiDemoJs = await readFile(uiDemoJsPath, 'utf8');
  const uiAppJs = await readUiAppBundle();
  availableAdapters = availableAdapters.map((adapter) => ({ ...adapter }));
  let activeAdapter = availableAdapters.find((adapter) => adapter.active)?.name || null;
  const completedTrainingJobs = [];
  const openEnvRuns = [];
  const smokeTeacherRevision = `sha256:${'7'.repeat(64)}`;
  const smokeTeacherContentRevision = `sha256:${'6'.repeat(64)}`;
  const smokeTeachers = [{
    spec: { alias: 'teacher-v1', kind: 'fixture', model_id: 'smoke-teacher-v1' },
    usable: true,
    status: 'configured',
    identity_revision: smokeTeacherRevision,
    capabilities: { teacher_id: 'teacher-v1', max_top_k: 32, vocab_size: 248320 },
  }];
  // Mirrors the real server's single running slot. SFT submit lands here
  // (state Running) so the drill-modal Stop flow can exercise the
  // cooperative running-job cancel; DELETE /v1/train/queue/:id moves it to
  // `completedTrainingJobs` as Failed("cancelled by user"), exactly like
  // the trainer aborting at the next step boundary.
  let runningTrainingJob = null;
  // Health-poll counter — see the /health handler.
  let healthTick = 0;
  // Eval datasets created through POST /v1/eval/datasets/upload (the
  // "Try a sample dataset" golden path) — GET /v1/eval/datasets reflects
  // them so the Datasets list refresh after upload is observable.
  const uploadedEvalDatasets = [];
  // Judgment datasets created/judged through the A/B walk. The rows POST
  // mirrors api/eval.rs append_judgment: it returns `judgment_id` plus the
  // flattened manifest, so the smoke can assert the record → Undo → DELETE
  // round-trip restores the visible counts.
  const judgmentDatasets = [];
  let judgmentRowCounter = 0;
  const judgmentManifest = (dataset) => ({
    name: dataset.name,
    description: dataset.description,
    num_rows: dataset.rows.length,
    created_at: dataset.created_at,
    updated_at: dataset.updated_at,
    winner_histogram: { ...dataset.winner_histogram },
  });
  const judgmentNotFound = (res, detail) => {
    res.writeHead(404, { 'content-type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: {
      code: 'judgment_not_found',
      message: `Judgment dataset '${detail}' not found`,
      hint: 'List judgments with GET /v1/judgments.',
    } }));
  };
  // Eval drill fixtures, mirroring kiln-eval's ExampleOutcome / SuiteResult
  // and the queue's EvalJobInfo: one completed compare job with real
  // outcomes (the raw-JSON toggle + outcomes-JSONL export walk reads these)
  // and one queued job with no runs yet (the export must disable on it).
  const smokeEvalOutcome = (exampleId, kind, text, detail) => ({
    example_id: exampleId,
    completion_index: 0,
    generation_seed: exampleId === 'ex-1' ? '18446744073709551614' : exampleId === 'ex-2' ? '18446744073709551613' : '18446744073709551612',
    completion_text: text,
    kind,
    score: kind === 'pass' ? 1 : 0,
    ...(detail ? { detail } : {}),
    latency_ms: 42,
    prompt_tokens: 12,
    completion_tokens: 4,
    tags: ['math'],
  });
  const smokeEvalRun = (adapter, outcomes) => {
    const numPass = outcomes.filter((outcome) => outcome.kind === 'pass').length;
    const accuracy = numPass / outcomes.length;
    return {
      suite_name: 'smoke-suite',
      adapter,
      metrics: {
        num_examples: outcomes.length,
        num_pass: numPass,
        num_fail: outcomes.length - numPass,
        num_invalid: 0,
        num_error: 0,
        accuracy,
        mean_score: accuracy,
        weighted_mean_score: accuracy,
        latency: { p50_ms: 42, p90_ms: 55, p99_ms: 61 },
        total_prompt_tokens: 36,
        total_completion_tokens: 12,
        elapsed_secs: 1.5,
        pass_rate_by_tag: { math: accuracy },
        by_scorer: [{ scorer_kind: 'exact_match', num_examples: outcomes.length, num_pass: numPass, accuracy }],
      },
      outcomes,
      started_at: '2026-06-11T00:00:00Z',
      finished_at: '2026-06-11T00:00:02Z',
      suite_hash: 'smoke-suite-hash',
    };
  };
  const smokeBaseWeightManifest = {
    schema_version: 1,
    manifest_type: 'kiln.base-weight-shards.v1',
    aggregate_algorithm: 'kiln.base-model-content.v1',
    aggregate_sha256: 'sha256:c62f9f56234c61c943716ae3b8783c851fb41a2551e31f17d15f1b0c346339b5',
    total_size_bytes: 11,
    shards: [{
      filename: 'model.safetensors',
      size_bytes: 11,
      sha256: `sha256:${'42'.repeat(32)}`,
    }],
  };
  const smokeExecutionProvenance = {
    schema_version: 1,
    provenance_type: 'kiln.execution-provenance.v1',
    backend: {
      name: 'rocm',
      device: 'gfx1151',
      numerical_runtime_sha256: `sha256:${'1'.repeat(64)}`,
    },
    build: {
      package_version: '0.4.1',
      target: 'x86_64-unknown-linux-gnu',
      executable_sha256: `sha256:${'2'.repeat(64)}`,
      git_commit: '6002d836',
      source_tree_sha256: `sha256:${'3'.repeat(64)}`,
      source_dirty: false,
    },
    model: {
      model_config_sha256: `sha256:${'4'.repeat(64)}`,
      tokenizer_vocab_sha256: `sha256:${'5'.repeat(64)}`,
      tokenizer_config_sha256: `sha256:${'6'.repeat(64)}`,
      chat_template_sha256: `sha256:${'7'.repeat(64)}`,
    },
    precision: {
      inference_dtype: 'bf16',
      training_policy: 'rocm_native_float',
    },
    kernels: {
      contract_type: 'kiln.kernel-contract.v1',
      versions: { 'kiln-model': '0.4.1' },
      compiled_features: ['rocm'],
      contract_sha256: `sha256:${'8'.repeat(64)}`,
    },
    configuration: {
      effective_server_config_sha256: `sha256:${'9'.repeat(64)}`,
      effective_environment_sha256: `sha256:${'a'.repeat(64)}`,
    },
    provenance_sha256: `sha256:${'b'.repeat(64)}`,
  };
  const smokeEvalJobs = [
    {
      job_id: 'smoke-eval-full',
      suite_name: 'smoke-suite',
      adapters: [null, 'smoke-tuned'],
      submission_kind: 'compare',
      base_weight_shard_manifest: smokeBaseWeightManifest,
      execution_provenance: smokeExecutionProvenance,
      effective_seed: '18446744073709551615',
      state: 'completed',
      progress: { examples_completed: 3, examples_total: 3, running_accuracy: 2 / 3, running_mean_score: 2 / 3 },
      finished_runs: [
        smokeEvalRun(null, [
          smokeEvalOutcome('ex-1', 'pass', '4'),
          smokeEvalOutcome('ex-2', 'fail', '41', 'expected 42, got 41'),
          smokeEvalOutcome('ex-3', 'fail', '7', 'expected 9, got 7'),
        ]),
        smokeEvalRun('smoke-tuned', [
          smokeEvalOutcome('ex-1', 'pass', '4'),
          smokeEvalOutcome('ex-2', 'fail', '41', 'expected 42, got 41'),
          smokeEvalOutcome('ex-3', 'pass', '9'),
        ]),
      ],
      headline_accuracy: 2 / 3,
      error: null,
      source_training_job_id: null,
      submitted_at_iso: '2026-06-11T00:00:00Z',
      started_at_iso: '2026-06-11T00:00:00Z',
      finished_at_iso: '2026-06-11T00:00:02Z',
    },
    {
      job_id: 'smoke-eval-empty',
      suite_name: 'smoke-suite',
      adapters: [null],
      submission_kind: 'on_demand',
      effective_seed: '73',
      state: 'queued',
      progress: { examples_completed: 0, examples_total: 0, running_accuracy: 0, running_mean_score: 0 },
      finished_runs: [],
      headline_accuracy: null,
      error: null,
      source_training_job_id: null,
      submitted_at_iso: '2026-06-11T00:00:01Z',
      started_at_iso: null,
      finished_at_iso: null,
    },
  ];
  // Agent-trace fixtures, mirroring api/agent_traces.rs `AgentTrace` /
  // `TraceOutcome` (trajectory segments are kiln-train `TurnSegment`s with
  // snake_case `kind`; assistant actions embed <think>/<tool_call> blocks
  // exactly as the pi trace normalizer renders them). One exit-0 session
  // and one failed+forked session so the outcome chips have something to
  // split.
  const smokeAgentTraces = [
    {
      id: 'trace-good-1111',
      working_dir: '/home/smoke/projects/widget',
      num_turns: 3,
      num_tool_calls: 1,
      outcome: { ended_with_exit_0: true, user_edited_agent_files: [], has_followup_attempt: false },
      first_event_at: '2026-06-10T10:00:00Z',
      last_event_at: '2026-06-10T10:05:00Z',
      forked: false,
      parent_id: null,
      tool_manifest_sha: 'sha256:smoke-manifest',
      prompt_messages: [
        { role: 'system', content: 'You are pi.' },
        { role: 'user', content: 'Fix the widget test' },
      ],
      trajectory: [
        { role: 'assistant', content: '<think>run the tests first</think><tool_call>{"name": "bash", "arguments": {"cmd": "cargo test -p widget"}}</tool_call>', kind: 'action' },
        { role: 'tool', content: 'test result: ok. 12 passed; 0 failed', kind: 'observation', tool_call_id: 'call-1' },
        { role: 'assistant', content: 'All 12 widget tests pass.', kind: 'action' },
      ],
    },
    {
      id: 'trace-fail-2222',
      working_dir: '/home/smoke/projects/gadget',
      num_turns: 2,
      num_tool_calls: 1,
      outcome: { ended_with_exit_0: false, user_edited_agent_files: ['src/main.rs'], has_followup_attempt: true },
      first_event_at: '2026-06-09T09:00:00Z',
      last_event_at: '2026-06-09T09:30:00Z',
      forked: true,
      parent_id: 'trace-parent-0000',
      tool_manifest_sha: null,
      prompt_messages: [
        { role: 'user', content: 'Refactor the gadget pipeline' },
      ],
      trajectory: [
        { role: 'assistant', content: '<tool_call>{"name": "bash", "arguments": {"cmd": "cargo build -p gadget"}}</tool_call>', kind: 'action' },
        { role: 'tool', content: 'error[E0308]: mismatched types', kind: 'observation', tool_call_id: 'call-2' },
      ],
    },
  ];
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', 'http://127.0.0.1');
    const adapterRoute = parseAdapterRoute(url.pathname);
    // Mirror kiln-server's routing: `/` and `/ui` redirect to `/ui/` (query
    // preserved) so the harness exercises the same redirect the real server
    // performs; the split assets are served alongside the page.
    if (url.pathname === '/') {
      res.writeHead(303, { location: '/ui/' });
      res.end();
      return;
    }
    if (url.pathname === '/ui') {
      res.writeHead(307, { location: `/ui/${url.search || ''}` });
      res.end();
      return;
    }
    if (url.pathname === '/ui/') {
      text(res, uiHtml, 'text/html; charset=utf-8');
      return;
    }
    if (url.pathname === '/ui/styles.css') {
      text(res, uiStyles, 'text/css');
      return;
    }
    if (url.pathname === '/ui/demo.js') {
      text(res, uiDemoJs, 'application/javascript');
      return;
    }
    if (url.pathname === '/ui/app.js') {
      text(res, uiAppJs, 'application/javascript');
      return;
    }
    if (url.pathname.startsWith('/ui/vendor/')) {
      const name = url.pathname.slice('/ui/vendor/'.length);
      const contentType = uiVendorFiles[name];
      if (contentType) {
        text(res, await readFile(resolve(uiVendorDir, name), 'utf8'), contentType);
        return;
      }
    }
    if (url.pathname === '/favicon.ico') {
      res.writeHead(204);
      res.end();
      return;
    }
    if (apiState.failDashboardApis) {
      if (url.pathname === '/health') {
        apiFailure(res, 'Server status', url.pathname);
        return;
      }
      if (url.pathname === '/v1/stats/decode') {
        apiFailure(res, 'Decode performance', url.pathname);
        return;
      }
      if (url.pathname === '/v1/config') {
        apiFailure(res, 'Runtime config', url.pathname);
        return;
      }
      if (url.pathname === '/v1/stats/recent-requests') {
        apiFailure(res, 'Recent requests', url.pathname);
        return;
      }
      if (url.pathname === '/v1/adapters') {
        apiFailure(res, 'Adapters', url.pathname);
        return;
      }
      if (url.pathname === '/v1/train/queue' || url.pathname === '/v1/train/status') {
        apiFailure(res, 'Training queue', url.pathname);
        return;
      }
      if (url.pathname === '/v1/openenv/runs') {
        apiFailure(res, 'OpenEnv runs', url.pathname);
        return;
      }
      // The eval-jobs background poll runs in every scenario — return
      // the configured failure shape rather than letting it 404 and
      // poison the failure-state browser-error assertion.
      if (
        url.pathname === '/v1/eval/jobs' ||
        url.pathname === '/v1/eval/suites' ||
        url.pathname === '/v1/eval/datasets' ||
        url.pathname === '/v1/judgments'
      ) {
        apiFailure(res, 'Evals', url.pathname);
        return;
      }
    }
    if (apiState.failRuntimeConfig && url.pathname === '/v1/config') {
      apiFailure(res, 'Runtime config', url.pathname);
      return;
    }
    if (url.pathname === '/health') {
      // blocks_used cycles so renderServerStatus's content key changes on
      // every poll — each 2s tick genuinely innerHTML-swaps #server-status,
      // which is what the runtime-config-expander survival assertion (and
      // the VRAM-donut regression assertion) must hold against.
      healthTick += 1;
      json(res, {
        status: 'ok',
        model: 'Qwen3.5-4B',
        backend: 'mock',
        uptime_seconds: 42,
        active_adapter: activeAdapter,
        scheduler: { waiting: 0, running: 0, blocks_used: healthTick % 2, blocks_free: 1024 },
        decode_runtime: {
          rocm_graphs: {
            state: 'busy',
            unavailable_reason: 'model_runner_busy',
            capture_enabled: null,
            phase_telemetry_available: true,
            phase_telemetry_unavailable_reason: null,
            current_phase: 'native_capture',
            current_phase_elapsed_micros: healthTick * 1_000_000,
          },
        },
        gpu_memory: { total_vram_gb: 24, model_gb: 8, kv_cache_gb: 2, training_budget_gb: 4 },
        checks: [{ name: 'mock smoke server', pass: true }],
      });
      return;
    }
    // Mirrors api/config.rs ConfigResponse. Playground and the runtime-config
    // expander share this immutable snapshot instead of polling it.
    if (url.pathname === '/v1/config') {
      json(res, {
        paths: {
          cache_root: '/var/cache/kiln',
          cache_root_source: 'config_file',
          restart_required_to_change: true,
        },
        vram: {
          probe_selector: 'linux-drm:amd:0',
          unified: true,
          physical_capacity_bytes: 25.75 * GIB,
          physical_capacity_gib: 25.75,
          physical_capacity_source: 'linux-drm-sysfs-unified',
          configured_capacity_bytes: 24 * GIB,
          configured_capacity_gib: 24.0,
          effective_capacity_bytes: 24 * GIB,
          effective_capacity_gib: 24.0,
          effective_capacity_source: 'memory.gpu_memory_gb',
          configured_capacity_clamped: false,
          live: {
            total_bytes: 25.75 * GIB,
            total_gib: 25.75,
            used_bytes: 9.5 * GIB,
            used_gib: 9.5,
            available_bytes: 16.25 * GIB,
            available_gib: 16.25,
            effective_capacity_available_bytes: 14.5 * GIB,
            effective_capacity_available_gib: 14.5,
            usable_after_governor_floor_bytes: 12.5 * GIB,
            usable_after_governor_floor_gib: 12.5,
            soft_reserved_bytes: 0,
            soft_reserved_gib: 0,
            pressure: 'Moderate',
            source: 'linux-drm-sysfs-unified',
            raw_observations: {
              probe_failed: false,
              driver_total_bytes: 64 * GIB,
              driver_used_bytes: 9.5 * GIB,
              driver_free_bytes: 54.5 * GIB,
              driver_vram_total_bytes: 16 * GIB,
              driver_vram_used_bytes: 4.5 * GIB,
              driver_gtt_total_bytes: 48 * GIB,
              driver_gtt_used_bytes: 5 * GIB,
              host_total_bytes: 32 * GIB,
              host_available_bytes: 18.25 * GIB,
              cgroup_limit_bytes: 28 * GIB,
              cgroup_current_bytes: 7.75 * GIB,
              cgroup_remaining_bytes: 20.25 * GIB,
              unified_reserve_bytes: 6.25 * GIB,
            },
          },
          governor: {
            floor_bytes: 2 * GIB,
            floor_gib: 2.0,
            capacity_limit_bytes: 24 * GIB,
            capacity_limit_gib: 24.0,
            probe_ms: 750,
            reclaim_mode_requested: 'automatic',
            reclaim_mode_effective: 'off',
            reclaim_mode_source: 'config_file',
            reclaim_disabled_by_serving_profile: true,
          },
        },
        accelerator_runtime: {
          schema_id: 'kiln.accelerator-runtime-policy.v16',
          version: 16,
          vulkan_kernel_policy_schema_id: 'kiln.vulkan-kernel-policy.v5',
          vulkan_device_policy_schema_id: 'kiln.vulkan-device-policy.v1',
          serving_profile: 'stable',
          serving_profile_source: 'default',
          kt_api_mode: {
            configured: 'auto',
            effective: 'auto',
            source: 'default',
          },
          full_attention_score_budget_mib: {
            configured: 2048,
            effective: 2048,
            source: 'default',
          },
          vulkan_device_index: { configured: null, effective: null, source: 'default' },
          vulkan_validation: { configured: false, effective: false, source: 'default' },
          cuda_kernel_profile: {
            configured: 'native_default',
            effective: 'native_default',
            source: 'config_file',
          },
          cuda_marlin_profile: {
            configured: 'disabled',
            effective: 'disabled',
            source: 'config_file',
          },
          cuda_flash_backward_mode: {
            configured: 'fast',
            effective: 'fast',
            source: 'config_file',
          },
          metal_kernel_profile: {
            configured: 'native_default',
            effective: 'native_default',
            source: 'config_file',
          },
          rocm_synchronization_mode: {
            configured: 'legacy_host_barriers',
            effective: 'legacy_host_barriers',
            source: 'default',
          },
          rocm_strided_batched_matmul_mode: {
            configured: 'disabled',
            effective: 'disabled',
            source: 'default',
          },
          rocm_bf16_matmul_output_mode: {
            configured: 'f32_then_cast',
            effective: 'f32_then_cast',
            source: 'default',
          },
          rocm_kernel_profile: {
            configured: 'portable_fallback',
            effective: 'portable_fallback',
            source: 'config_file',
          },
          rocm_graph_mode: { configured: 'profile', effective: 'disabled', source: 'default' },
          rocm_graph_cache_entries: { configured: 8, effective: 8, source: 'default' },
          rocm_graph_cache_max_bytes: { configured: GIB, effective: GIB, source: 'default' },
        },
        cuda_graphs: {
          requested: true,
          capture_allowed_by_serving_profile: false,
          effective_policy_enabled: false,
          max_cached_graphs: 8,
          stable_paged_metadata: true,
          batched_capture_available: false,
          restart_required_to_change: true,
        },
        kv_cache: { num_blocks: 1024, num_blocks_source: 'auto', fp8_enabled: true },
        batching: {
          configuration: {
            rowwise_decode: { enabled: false, source: 'default' },
            prefix_aware_admission: { enabled: true, source: 'environment' },
            prefill_admission_quantum: {
              configured: 32,
              configured_source: 'config_file',
              backend_policy: 16,
              effective: 16,
              effective_source: 'effective_decode_width',
            },
            actor_cycle_idle: {
              milliseconds: 75,
              source: 'config_file',
              enabled: true,
              command_poll_milliseconds: 5,
            },
            burst_prefill_admission: true,
            actor_prefill_tile_alignment_required: false,
          },
          actor_active: true,
        },
        streaming_prefill: {
          dispatch: {
            configured_mode: 'auto',
            configured_source: 'config_file',
            backend_policy: { policy: 'prompt_tokens_at_least', minimum_prompt_tokens: 2_048 },
            effective: { policy: 'prompt_tokens_at_least', minimum_prompt_tokens: 4_096 },
            effective_source: 'environment',
          },
          threshold_tokens: {
            configured: 4_096,
            configured_source: 'environment',
            backend_policy: 2_048,
            effective_for_auto_mode: 4_096,
            override_applied_to_backend_auto_policy: true,
          },
          tile_tokens: {
            configured: 4_096,
            configured_source: 'config_file',
            backend_policy: 2_048,
            effective: 4_096,
            effective_source: 'config_file',
          },
          tape_tile_tokens: {
            configured: null,
            configured_source: 'default',
            backend_policy: 2_048,
            effective: 4_096,
            effective_source: 'inherited_from_tile_tokens_config_file',
          },
          detached_full_attn_tile_tokens: {
            configured: null,
            configured_source: 'default',
            backend_policy: 8_192,
            effective: 4_096,
            effective_source: 'inherited_from_tile_tokens_config_file',
          },
          detached_full_attn_boundary_tile_tokens: {
            backend_policy: 8_192,
            effective: 4_096,
            effective_source: 'inherited_from_tile_tokens_config_file',
          },
          detached_full_attn_tape_replay_tile_tokens: {
            backend_policy: 8_192,
            effective: 4_096,
            effective_source: 'inherited_from_tile_tokens_config_file',
          },
          last_token_lm_head: {
            configured: true,
            configured_source: 'default',
            effective: true,
            effective_source: 'default',
          },
          immutable_after_startup: true,
          restart_required_to_change: true,
        },
        training: smokeTrainingRuntimeConfig(apiState.trainingOptimizerAvailable),
        memory_budget: {
          total_vram_bytes: 24 * GIB,
          total_vram_gib: 24.0,
          model_bytes: 8.25 * GIB,
          model_gib: 8.25,
          kv_cache_bytes: 2.125 * GIB,
          kv_cache_gib: 2.125,
          training_budget_bytes: 4 * GIB,
          training_budget_gib: 4.0,
          inference_memory_fraction: 0.55,
        },
        generation: {
          default_thinking_enabled: true,
          default_thinking_budget_tokens: 64,
          default_thinking_budget_ms: 1500,
          fold_reasoning_into_content: false,
        },
      });
      return;
    }
    if (url.pathname === '/metrics') {
      text(res, '# HELP kiln_mock_info Mock metrics for UI smoke\n# TYPE kiln_mock_info gauge\nkiln_mock_info 1\n');
      return;
    }
    // Durable corrections store: the basket's init sync (GET) and
    // write-through (POST/DELETE) run in every scenario.
    if (url.pathname === '/v1/corrections') {
      if (req.method === 'GET') { json(res, { corrections: [] }); return; }
      if (req.method === 'POST') { json(res, { status: 'ok' }); return; }
      if (req.method === 'DELETE') { json(res, { status: 'cleared', removed: 0 }); return; }
    }
    if (url.pathname.startsWith('/v1/corrections/')) {
      json(res, { status: 'ok' });
      return;
    }
    if (url.pathname === '/v1/adapters') {
      json(res, { active: activeAdapter, available: availableAdapters });
      return;
    }
    if (url.pathname === '/v1/adapters/load') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for adapter load' }));
        return;
      }
      const body = await readJsonBody(req);
      const validationError = validateExistingAdapterName(body?.name, availableAdapters, 'Load');
      if (validationError) {
        apiBadRequest(res, validationError);
        return;
      }
      activeAdapter = body.name;
      setTimeout(() => json(res, { active: activeAdapter }), 75);
      return;
    }
    if (url.pathname === '/v1/adapters/unload') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for adapter unload' }));
        return;
      }
      const body = await readBufferBody(req);
      if (body.length > 0) {
        apiBadRequest(res, 'Unload should not require or send a request body');
        return;
      }
      activeAdapter = null;
      setTimeout(() => json(res, { active: null }), 75);
      return;
    }
    // Mirrors api/adapters.rs AdapterDetail — the drill modal's main body.
    if (adapterRoute?.action === 'detail') {
      const adapter = availableAdapters.find((candidate) => candidate.name === adapterRoute.name);
      if (!adapter) {
        adapterNotFound(res, adapterRoute.name);
        return;
      }
      json(res, {
        name: adapter.name,
        is_active: activeAdapter === adapter.name,
        has_config: true,
        has_weights: true,
        size_bytes: adapter.size_bytes,
        files: [
          { name: 'adapter_config.json', size_bytes: 512 },
          { name: 'adapter_model.safetensors', size_bytes: adapter.size_bytes },
        ],
        training_jobs: [],
        eval_jobs: [],
      });
      return;
    }
    // GET /v1/adapters/:name/receipt — adapter-alpha ships the smoke
    // receipt; everything else 404s with the adapter_not_found envelope
    // (api/adapters.rs adapter_receipt's Ok(None) branch).
    if (adapterRoute?.action === 'receipt') {
      const adapterExists = availableAdapters.some((candidate) => candidate.name === adapterRoute.name);
      if (!adapterExists || adapterRoute.name !== 'adapter-alpha') {
        adapterNotFound(res, `${adapterRoute.name}/receipt.json`);
        return;
      }
      json(res, smokeAdapterReceipt);
      return;
    }
    if (adapterRoute?.action === 'download') {
      if (req.method !== 'GET') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use GET for adapter download' }));
        return;
      }
      const validationError = validateExistingAdapterName(adapterRoute.name, availableAdapters, 'Download');
      if (validationError) {
        apiBadRequest(res, validationError);
        return;
      }
      const archive = Buffer.from(`kiln smoke adapter archive: ${adapterRoute.name}\n`);
      res.writeHead(200, {
        'content-type': 'application/gzip',
        'content-disposition': `attachment; filename="${adapterRoute.name}.tar.gz"`,
        'content-length': String(archive.length),
      });
      res.end(archive);
      return;
    }
    if (adapterRoute && adapterRoute.action === null) {
      if (req.method !== 'DELETE') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use DELETE for adapter deletion' }));
        return;
      }
      const validationError = validateExistingAdapterName(adapterRoute.name, availableAdapters, 'Delete');
      if (validationError) {
        apiBadRequest(res, validationError);
        return;
      }
      availableAdapters = availableAdapters.filter((adapter) => adapter.name !== adapterRoute.name);
      if (activeAdapter === adapterRoute.name) activeAdapter = null;
      setTimeout(() => json(res, { deleted: adapterRoute.name }), 75);
      return;
    }
    if (url.pathname === '/v1/adapters/upload') {
      const body = await readBufferBody(req);
      const validation = validateAdapterUploadRequest(req, body);
      if (validation.detail) {
        res.writeHead(validation.status, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: validation.detail }));
        return;
      }
      if (!availableAdapters.some((adapter) => adapter.name === validation.name)) {
        availableAdapters.push({ name: validation.name, active: false, size_bytes: validation.archiveSize });
      }
      setTimeout(() => json(res, {
        name: validation.name,
        size_bytes: validation.archiveSize,
        files: 2,
      }), 300);
      return;
    }
    if (url.pathname === '/v1/adapters/merge') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for adapter merge' }));
        return;
      }
      const body = await readJsonBody(req);
      const validationError = validateAdapterMergePayload(body);
      if (validationError) {
        apiBadRequest(res, validationError);
        return;
      }
      setTimeout(() => json(res, {
        sources: body.sources,
        output_name: body.output_name,
        mode: body.mode,
        density: body.density,
        num_tensors: 32,
      }), 75);
      return;
    }
    if (url.pathname === '/v1/teachers' && req.method === 'GET') {
      json(res, { teachers: smokeTeachers });
      return;
    }
    if (url.pathname === '/v1/recipes' && req.method === 'GET') {
      const supported = apiState.trainingOptimizerAvailable === true;
      json(res, { recipes: [
        {
          name: 'smoke-supported-recipe',
          description: 'Exercises an admitted single-workload recipe in the dashboard',
          num_steps: 2,
          admission: {
            supported,
            unavailable_reason: supported ? null : 'recipe optimizer tuple is unavailable',
          },
        },
        {
          name: 'smoke-refresh-recipe',
          description: 'Exercises fail-closed distill-refresh recipe admission',
          num_steps: 2,
          admission: {
            supported: false,
            unavailable_reason: 'distill_refresh is unavailable until admission pins separate exact SFT and OPD phase plans, prepares the exact SFT rows, and reserves the maximum sequential working set',
          },
        },
        {
          name: 'smoke-missing-admission',
          description: 'Deliberately omits admission for fail-closed coverage',
          num_steps: 1,
        },
      ] });
      return;
    }
    if (url.pathname === '/v1/recipes/run' && req.method === 'POST') {
      apiState.trainingSubmitRequests.recipe += 1;
      json(res, { message: 'Recipe queued', effective_seeds: {} });
      return;
    }
    const directTrainingRoutes = {
      '/v1/distill/refresh': ['refresh', 'Refresh queued'],
      '/v1/distill/pump': ['pump', 'Boost job queued'],
      '/v1/adapters/distill_merge': ['merge', 'Merge queued'],
      '/v1/distill/self': ['self', 'Self-improvement job queued'],
    };
    if (req.method === 'POST' && directTrainingRoutes[url.pathname]) {
      const [counter, message] = directTrainingRoutes[url.pathname];
      apiState.trainingSubmitRequests[counter] += 1;
      await readJsonBody(req);
      json(res, { message, job_id: `smoke-${counter}` });
      return;
    }
    if (url.pathname === '/v1/train/queue' || url.pathname === '/v1/train/status') {
      json(res, { running: runningTrainingJob, queued: [], completed: completedTrainingJobs });
      return;
    }
    if (url.pathname === '/v1/openenv/runs' && req.method === 'GET') {
      json(res, { schema: 'kiln.openenv-run-list.v3', runs: openEnvRuns });
      return;
    }
    if (url.pathname === '/v1/openenv/inspect' && req.method === 'POST') {
      await readJsonBody(req);
      json(res, {
        schema: 'kiln.openenv-inspection.v1',
        environments: [{
          identity: {
            schema: 'kiln.openenv-identity.v1',
            client_profile: 'openenv-python-sdk-v1',
            base_url: 'http://127.0.0.1:8990',
            websocket_url: 'ws://127.0.0.1:8990/ws',
            openapi_version: '3.1.0',
            environments: ['smoke-bandit'],
            metadata: { name: 'smoke-bandit', description: 'Choose an arm.', version: '1.0.0' },
            schema_sha256: `sha256:${'a'.repeat(64)}`,
          },
          schema: {
            action: { type: 'object', required: ['arm'], properties: { arm: { type: 'integer' } } },
            observation: { type: 'object' },
            state: { type: 'object' },
          },
        }],
      });
      return;
    }
    if (url.pathname === '/v1/openenv/tasks' && req.method === 'POST') {
      const request = await readJsonBody(req);
      json(res, {
        schema: 'kiln.openenv-task-catalog.v1',
        catalogs: [{
          base_url: 'http://127.0.0.1:8990',
          catalog: {
            environment_name: 'smoke-bandit',
            task_api: 'available',
            splits: [
              { name: 'train', type: 'train' },
              { name: 'holdout', type: 'validation' },
            ],
            ...(request.split ? {
              selected_split: 'train',
              num_tasks: 3,
              start: request.start,
              stop: Math.min(request.start + request.limit, 3),
              tasks: [{ id: 1, prompt: '2 + 2', answer: '4' }],
            } : {}),
          },
        }],
      });
      return;
    }
    if (url.pathname === '/v1/openenv/runs' && req.method === 'POST') {
      const request = await readJsonBody(req);
      apiState.trainingSubmitRequests.openenv += 1;
      const status = {
        schema: 'kiln.openenv-run.v3',
        run_id: 'smoke-openenv-run',
        kind: request.kind,
        state: 'completed',
        request,
        submitted_unix_ms: 1_700_000_000_000,
        progress: {
          groups_completed: 2,
          groups_total: request.groups,
          rollouts_completed: 8,
          rollouts_total: request.groups * request.group_size,
        },
        environments: [],
        artifacts: [],
        environment_evaluation: {
          state: 'completed',
          seed_start: request.groups,
          groups: request.environment_eval?.groups || 20,
          group_size: request.environment_eval?.group_size || 1,
          baseline: {},
          candidate: {
            adapter: request.output_adapter,
            adapter_content_revision: `sha256:${'b'.repeat(64)}`,
          },
          progress: {
            state: 'completed',
            groups_completed: request.environment_eval?.groups || 20,
            groups_total: request.environment_eval?.groups || 20,
            rollouts_completed: (request.environment_eval?.groups || 20) * (request.environment_eval?.group_size || 1),
            rollouts_total: (request.environment_eval?.groups || 20) * (request.environment_eval?.group_size || 1),
          },
          evidence: {
            policy_version: 'paired_return_sign_test_v1',
            paired_groups: 20,
            paired_episodes: 20,
            minimum_paired_groups: 20,
            baseline_mean_return: 0.1,
            candidate_mean_return: 0.8,
            mean_return_improvement: 0.7,
            improved_groups: 20,
            regressed_groups: 0,
            tied_groups: 0,
            exact_sign_test_p_value: 0.0000019073486328125,
            exact_sign_test_alpha: 0.05,
            decision: 'passed',
            reason: 'significant paired return improvement',
          },
          outcome: 'promoted',
          verdict: 'Environment promotion gate passed.',
        },
      };
      openEnvRuns.unshift(status);
      json(res, status, 202);
      return;
    }
    // Parameterized training-job routes, mirroring the real API:
    //   GET    /v1/train/jobs/:id   — drill-modal detail payload
    //   DELETE /v1/train/queue/:id  — cooperative cancel (running jobs)
    const trainJobDetailMatch = /^\/v1\/train\/jobs\/([^/]+)$/.exec(url.pathname);
    if (trainJobDetailMatch && req.method === 'GET') {
      const jobId = decodeURIComponent(trainJobDetailMatch[1]);
      const job = (runningTrainingJob && runningTrainingJob.job_id === jobId)
        ? runningTrainingJob
        : completedTrainingJobs.find((candidate) => candidate.job_id === jobId);
      if (!job) {
        res.writeHead(404, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: { code: 'training_job_not_found', message: `Training job '${jobId}' not found` } }));
        return;
      }
      json(res, {
        ...job,
        progress: job.progress ?? 0,
        loss_history: job.loss_history || [],
        linked_eval_job_ids: job.linked_eval_job_ids || [],
        auto_load: job.auto_load ?? false,
      });
      return;
    }
    const trainCancelMatch = /^\/v1\/train\/queue\/([^/]+)$/.exec(url.pathname);
    if (trainCancelMatch && req.method === 'DELETE') {
      const jobId = decodeURIComponent(trainCancelMatch[1]);
      if (runningTrainingJob && runningTrainingJob.job_id === jobId) {
        completedTrainingJobs.unshift({
          ...runningTrainingJob,
          state: 'Failed',
          error: 'cancelled by user',
        });
        runningTrainingJob = null;
        json(res, {
          job_id: jobId,
          status: 'cancelling',
          message: 'stop requested — the trainer aborts at the next step boundary',
        });
        return;
      }
      res.writeHead(409, { 'content-type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: { code: 'training_job_not_cancellable', message: `Cannot cancel job '${jobId}'` } }));
      return;
    }
    if (url.pathname === '/v1/train/sft') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for SFT training' }));
        return;
      }
      apiState.trainingSubmitRequests.sft += 1;
      const body = await readJsonBody(req);
      const validationError = validateSftPayload(body);
      if (validationError) {
        apiBadRequest(res, validationError);
        return;
      }
      // SFT lands as the RUNNING job so the drill-modal Stop flow can
      // assert running-job cooperative cancel against the mock. The loss
      // samples mirror state.rs TrainingLossSample (epoch/progress/loss/
      // elapsed_secs — no step, no wall-clock) and feed the drill modal's
      // "Copy loss CSV" assertion.
      const namedDataset = typeof body.dataset === 'string';
      const jobId = namedDataset ? 'smoke-sft-named' : 'smoke-sft';
      runningTrainingJob = {
        job_id: jobId,
        job_type: 'sft',
        effective_seed: '18446744073709551615',
        state: 'Running',
        progress: 0.42,
        current_loss: 1.234,
        epoch: 1,
        adapter_name: body.config.output_name,
        elapsed_secs: 12,
        loss_history: [
          { epoch: 1, progress: 0.1, loss: 2.5, elapsed_secs: 2 },
          { epoch: 1, progress: 0.25, loss: 1.9, elapsed_secs: 5 },
          { epoch: 1, progress: 0.42, loss: 1.234, elapsed_secs: 12 },
        ],
        linked_eval_job_ids: [],
        auto_load: false,
        training_data: namedDataset ? {
          source: 'named_dataset',
          dataset: body.dataset,
          split: body.dataset_split,
          dataset_corpus_sha256: `sha256:${'1'.repeat(64)}`,
          split_manifest_sha256: `sha256:${'3'.repeat(64)}`,
          admitted_corpus_sha256: `sha256:${'4'.repeat(64)}`,
          rows: 8,
        } : {
          source: 'inline',
          admitted_corpus_sha256: `sha256:${'5'.repeat(64)}`,
          rows: 1,
        },
      };
      setTimeout(() => json(res, { message: 'SFT job submitted', job_id: jobId, effective_seed: '18446744073709551615' }), 75);
      return;
    }
    if (url.pathname === '/v1/train/grpo') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for GRPO training' }));
        return;
      }
      apiState.trainingSubmitRequests.grpo += 1;
      const body = await readJsonBody(req);
      const validationError = validateGrpoPayload(body);
      if (validationError) {
        apiBadRequest(res, validationError);
        return;
      }
      completedTrainingJobs.unshift({
        job_id: 'smoke-grpo',
        job_type: 'grpo',
        effective_seed: '18446744073709551614',
        state: 'Completed',
        progress: 1,
        adapter_name: body.config.output_name,
        elapsed_secs: 1,
        train_receipt: {
          model: {
            path: '/models/qwen3.5-4b',
            config_hash: `sha256:${'3'.repeat(64)}`,
            base_weight_shard_manifest: smokeBaseWeightManifest,
          },
          runtime: {
            execution_provenance: smokeExecutionProvenance,
            training_precision: {
              parameter_dtype: 'bf16',
              optimizer_state_dtype: 'f32',
              activation_dtype: 'f32',
              gradient_dtype: 'f32',
              stochastic_rounding: { mode: 'round_to_nearest' },
            },
          },
          hyperparameters: {
            mode: 'grpo',
            seed: '18446744073709551614',
          },
        },
        replay_request: {
          kind: 'grpo',
          request_body: {
            groups_count: body.groups.length,
            config: body.config,
          },
        },
        latest_checkpoint: {
          resume_checkpoint: 'grpo-adapter-checkpoint-step-00000003.kiln-checkpoint',
          checkpoint_id: 'smoke-grpo-checkpoint-3',
          training_kind: 'grpo',
          data_source_kind: 'jsonl-grpo-trainable-order-v1',
          global_step: 3,
          total_steps: 5,
          next_epoch_index: 0,
          next_cursor_in_epoch: 3,
          complete: false,
          created_at: '2026-07-10T12:00:00Z',
        },
      });
      setTimeout(() => json(res, { message: 'GRPO job submitted', job_id: 'smoke-grpo', effective_seed: '18446744073709551614' }), 75);
      return;
    }
    if (url.pathname === '/v1/train/opd') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for OPD training' }));
        return;
      }
      apiState.trainingSubmitRequests.opd += 1;
      const body = await readJsonBody(req);
      const validationError = validateOpdPayload(body);
      if (validationError) {
        apiBadRequest(res, validationError);
        return;
      }
      completedTrainingJobs.unshift({
        job_id: 'smoke-opd',
        job_type: 'opd',
        effective_seed: '18446744073709551613',
        state: 'Completed',
        progress: 1,
        adapter_name: body.config.output_name,
        elapsed_secs: 2,
        latest_checkpoint: {
          resume_checkpoint: 'opd-adapter-checkpoint-step-00000002.kiln-checkpoint',
          checkpoint_id: 'smoke-opd-checkpoint-2',
          training_kind: 'opd',
          data_source_kind: 'inline-opd-prompts-v1',
          global_step: 2,
          total_steps: 4,
          next_epoch_index: 0,
          next_cursor_in_epoch: 2,
          complete: false,
          created_at: '2026-07-10T12:05:00Z',
          effective_config: { ...body.config, seed: 73 },
          data_content_sha256: '9'.repeat(64),
          data_item_count: body.prompts.length,
          teacher_id: body.teacher,
          teacher_identity_revision: smokeTeacherRevision,
          teacher_content_revision: smokeTeacherContentRevision,
        },
      });
      setTimeout(() => json(res, { message: 'OPD job submitted', job_id: 'smoke-opd', effective_seed: '18446744073709551613' }), 75);
      return;
    }
    if (url.pathname === '/v1/models') {
      apiState.modelsRequests += 1;
      if (apiState.modelsCold) {
        apiFailure(res, 'Models', url.pathname);
        return;
      }
      json(res, { object: 'list', data: [{ id: servedModelId, object: 'model', owned_by: 'kiln' }] });
      return;
    }
    if (url.pathname === '/v1/stats/decode') {
      json(res, {
        window_secs: 60,
        sample_count: 0,
        tok_per_sec: 0,
        p50_itl_ms: 0,
        p99_itl_ms: 0,
        p999_itl_ms: 0,
        mean_itl_ms: 0,
        max_itl_ms: 0,
        stall_threshold_ms: 250,
        stall_count: 0,
        unexplained_stall_count: 0,
        stall_reasons: {
          actor_queue: 0, actor_admission: 0, actor_prefill: 0, actor_decode: 0,
          actor_cycle_idle: 0,
          response_delivery: 0, handler_queue: 0, client_delivery: 0, sampling: 0,
          readback: 0, gpu_lock_wait: 0, graph_capture: 0, graph_replay: 0,
          synchronization: 0, resize: 0, trim: 0, adapter: 0, training: 0,
          unexplained: 0,
        },
      });
      return;
    }
    if (url.pathname === '/v1/terminal/status') {
      json(res, {
        enabled: false,
        disabled_reason: 'not available in the smoke fixture',
        pi_available: false,
        pi_path: null,
        cwd: '/tmp',
        session_active: false,
      });
      return;
    }
    if (url.pathname === '/v1/stats/recent-requests') {
      json(res, apiState.recentRequests);
      return;
    }
    // The UI polls /v1/eval/jobs at startup to keep the Evals badge
    // accurate before the user has visited the tab. The fixture jobs feed
    // the eval drill walk (raw JSON toggle + outcomes JSONL export) and the
    // deep-link drill walk (#evals/jobs/smoke-eval-full); their adapters
    // reference no real adapter card, so the rest of the dashboard is
    // unaffected.
    if (url.pathname === '/v1/eval/jobs') {
      json(res, { jobs: smokeEvalJobs });
      return;
    }
    // GET /v1/eval/jobs/:id — mirrors EvalJobInfo::to_result: the public
    // detail payload renames finished_runs to `runs` and only ships
    // `progress` while the job is still active.
    const evalJobDetailMatch = /^\/v1\/eval\/jobs\/([^/]+)$/.exec(url.pathname);
    if (evalJobDetailMatch && req.method === 'GET') {
      const jobId = decodeURIComponent(evalJobDetailMatch[1]);
      const job = smokeEvalJobs.find((candidate) => candidate.job_id === jobId);
      if (!job) {
        res.writeHead(404, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: { code: 'eval_job_not_found', message: `Eval job '${jobId}' not found` } }));
        return;
      }
      json(res, {
        job_id: job.job_id,
        state: job.state,
        base_weight_shard_manifest: job.base_weight_shard_manifest,
        execution_provenance: job.execution_provenance,
        effective_seed: job.effective_seed,
        seed_derivation: 'kiln.eval-seed.v1',
        runs: job.finished_runs,
        ...(job.state === 'queued' || job.state === 'running' ? { progress: job.progress } : {}),
      });
      return;
    }
    // Same shape as the production endpoints — only the empty cases
    // are exercised in smoke, but keep them registered so future UI
    // background polls don't surprise the mock.
    if (url.pathname === '/v1/eval/suites') {
      json(res, { suites: apiState.evalSuites });
      return;
    }
    // The drill modal lazily fetches the suite content so it can show the
    // prompt next to each outcome. Serve the fixture suite (kept OUT of
    // the suites list so the Suites empty-state assertions still hold).
    if (url.pathname === '/v1/eval/suites/smoke-suite' && req.method === 'GET') {
      json(res, {
        name: 'smoke-suite',
        examples: [
          { id: 'ex-1', messages: [{ role: 'user', content: 'What is 2 + 2?' }], target: '4', tags: ['math'] },
          { id: 'ex-2', messages: [{ role: 'user', content: 'What is 6 x 7?' }], target: '42', tags: ['math'] },
          { id: 'ex-3', messages: [{ role: 'user', content: 'What is 3 squared?' }], target: '9', tags: ['math'] },
        ],
      });
      return;
    }
    if (url.pathname === '/v1/eval/datasets') {
      json(res, { datasets: uploadedEvalDatasets });
      return;
    }
    // Mirrors api/eval.rs upload_dataset: multipart in, DatasetManifest out
    // (the manifest JSON is the entire response body, not an envelope).
    if (url.pathname === '/v1/eval/datasets/upload') {
      const body = await readBufferBody(req);
      const result = validateEvalDatasetUploadRequest(req, body);
      if (result.detail) {
        res.writeHead(result.status, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: { code: 'dataset_invalid', message: result.detail, hint: '' } }));
        return;
      }
      if (uploadedEvalDatasets.some((dataset) => dataset.name === result.manifest.name)) {
        res.writeHead(409, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: {
          code: 'dataset_exists',
          message: `Eval dataset '${result.manifest.name}' already exists`,
          hint: 'Delete or rename the existing dataset, or use a different name.',
        } }));
        return;
      }
      uploadedEvalDatasets.push(result.manifest);
      json(res, result.manifest);
      return;
    }
    if (url.pathname === '/v1/judgments') {
      if (req.method === 'POST') {
        const body = await readJsonBody(req);
        const name = (body?.name || '').trim();
        if (!isPathSafeAdapterDirectoryName(name)) {
          apiBadRequest(res, 'Judgment dataset name should be path-safe');
          return;
        }
        if (judgmentDatasets.some((dataset) => dataset.name === name)) {
          res.writeHead(409, { 'content-type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ error: {
            code: 'dataset_exists',
            message: `Judgment dataset '${name}' already exists`,
            hint: 'Append rows to the existing dataset or pick a new name.',
          } }));
          return;
        }
        const now = new Date().toISOString();
        const dataset = {
          name,
          description: body?.description || null,
          created_at: now,
          updated_at: now,
          winner_histogram: {},
          rows: [],
        };
        judgmentDatasets.push(dataset);
        json(res, judgmentManifest(dataset));
        return;
      }
      json(res, { judgments: judgmentDatasets.map(judgmentManifest) });
      return;
    }
    // POST /v1/judgments/:name/rows — append a judgment. Mirrors the real
    // handler's response shape: `judgment_id` for the appended row plus the
    // flattened manifest (the UI's Undo DELETEs by that id).
    const judgmentRowsMatch = /^\/v1\/judgments\/([^/]+)\/rows$/.exec(url.pathname);
    if (judgmentRowsMatch && req.method === 'POST') {
      const name = decodeURIComponent(judgmentRowsMatch[1]);
      const dataset = judgmentDatasets.find((candidate) => candidate.name === name);
      if (!dataset) {
        judgmentNotFound(res, name);
        return;
      }
      const body = await readJsonBody(req);
      if (!Array.isArray(body?.prompt) || body.prompt.length === 0) {
        apiBadRequest(res, 'Judgment rows should carry the prompt messages');
        return;
      }
      if (!['a', 'b', 'tie', 'skip'].includes(body?.winner)) {
        apiBadRequest(res, 'Judgment winner should be one of a|b|tie|skip');
        return;
      }
      if (typeof body?.response_a !== 'string' || typeof body?.response_b !== 'string') {
        apiBadRequest(res, 'Judgment rows should carry both responses');
        return;
      }
      judgmentRowCounter += 1;
      const id = body.id || `smoke-judgment-${judgmentRowCounter}`;
      dataset.rows.push({ id, winner: body.winner });
      dataset.winner_histogram[body.winner] = (dataset.winner_histogram[body.winner] || 0) + 1;
      dataset.updated_at = new Date().toISOString();
      json(res, { judgment_id: id, ...judgmentManifest(dataset) });
      return;
    }
    // DELETE /v1/judgments/:name/rows/:id — the Undo path. Unknown ids 404
    // (a double-fired Undo must not silently "succeed").
    const judgmentRowDeleteMatch = /^\/v1\/judgments\/([^/]+)\/rows\/([^/]+)$/.exec(url.pathname);
    if (judgmentRowDeleteMatch && req.method === 'DELETE') {
      const name = decodeURIComponent(judgmentRowDeleteMatch[1]);
      const id = decodeURIComponent(judgmentRowDeleteMatch[2]);
      const dataset = judgmentDatasets.find((candidate) => candidate.name === name);
      const rowIndex = dataset ? dataset.rows.findIndex((row) => row.id === id) : -1;
      if (!dataset || rowIndex === -1) {
        judgmentNotFound(res, `${name}/${id}`);
        return;
      }
      const [removed] = dataset.rows.splice(rowIndex, 1);
      dataset.winner_histogram[removed.winner] = Math.max(0, (dataset.winner_histogram[removed.winner] || 1) - 1);
      dataset.updated_at = new Date().toISOString();
      json(res, judgmentManifest(dataset));
      return;
    }
    // Agent traces (api/agent_traces.rs): discover rescans a sessions dir
    // (optional `path` in the POST body), the index GET lists every trace,
    // and the per-trace GET returns one full record by id.
    if (url.pathname === '/v1/agent/traces/discover' && req.method === 'POST') {
      const body = await readJsonBody(req);
      if ('path' in body && typeof body.path !== 'string') {
        apiBadRequest(res, 'discover `path` must be a string when present');
        return;
      }
      json(res, {
        indexed: smokeAgentTraces.length,
        path: body.path || '/home/smoke/.pi/agent/sessions',
      });
      return;
    }
    if (url.pathname === '/v1/agent/traces' && req.method === 'GET') {
      json(res, { traces: smokeAgentTraces });
      return;
    }
    const traceDetailMatch = /^\/v1\/agent\/traces\/([^/]+)$/.exec(url.pathname);
    if (traceDetailMatch && req.method === 'GET') {
      const traceId = decodeURIComponent(traceDetailMatch[1]);
      const trace = smokeAgentTraces.find((candidate) => candidate.id === traceId);
      if (!trace) {
        res.writeHead(400, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: {
          code: 'training_invalid_request',
          message: `trace ${traceId} not indexed`,
          hint: 'Rescan with POST /v1/agent/traces/discover.',
        } }));
        return;
      }
      json(res, trace);
      return;
    }
    if (url.pathname === '/v1/chat/completions') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for chat completions' }));
        return;
      }
      // Bug-A regression: every dashboard-originated inference request must
      // self-identify so the server can mark it `client: 'dashboard'` and
      // onboarding milestones don't count the dashboard as an agent.
      if (req.headers['x-kiln-client'] !== 'dashboard') {
        res.writeHead(400, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Dashboard inference request is missing the X-Kiln-Client: dashboard header' }));
        return;
      }
      const body = await readJsonBody(req);
      const prompt = body?.messages?.findLast((message) => message.role === 'user')?.content || '';
      // The judgment viewer streams the same prompt twice (slots A and B).
      if (body?.stream && /Judge this smoke pair\./.test(prompt)) {
        sse(res, [
          { choices: [{ delta: { role: 'assistant' } }] },
          { choices: [{ delta: { content: 'smoke reply for judging' } }] },
        ]);
        return;
      }
      if (body?.stream && /Compare stream failure\./.test(prompt)) {
        if (body.adapter === 'adapter-alpha') {
          sse(res, [{
            error: {
              message: 'Injected compare stream failure.',
              type: 'server_error',
              code: 'generation_error',
            },
          }]);
        } else {
          const metadata = smokeThinkingBudgetMetadata(body, { triggered: false });
          sse(res, [
            { choices: [{ delta: { content: 'Healthy side.' } }] },
            {
              choices: [{
                delta: {},
                finish_reason: 'stop',
                thinking_budget: smokeThinkingBudgetOutcome(metadata),
              }],
              metadata: { thinking_budget: metadata },
            },
          ]);
        }
        return;
      }
      if (body?.stream && /Compare budget outcomes\./.test(prompt)) {
        const capped = body.adapter === 'adapter-alpha';
        const metadata = smokeThinkingBudgetMetadata(body, {
          triggered: capped,
          trigger: 'tokens',
        });
        sse(res, [
          { choices: [{ delta: { role: 'assistant' } }] },
          { choices: [{ delta: { reasoning_content: capped ? 'Adapter reasoning.' : 'Base reasoning.' } }] },
          { choices: [{ delta: { content: capped ? 'Adapter final.' : 'Base final.' } }] },
          {
            choices: [{
              delta: {},
              finish_reason: capped ? 'length' : 'stop',
              thinking_budget: smokeThinkingBudgetOutcome(metadata),
            }],
            metadata: { thinking_budget: metadata },
          },
        ]);
        return;
      }
      if (!body?.stream || !/Explain Kiln in one sentence\./.test(prompt)) {
        res.writeHead(400, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Unexpected Quick Inference smoke request' }));
        return;
      }
      const metadata = smokeThinkingBudgetMetadata(body);
      sse(res, [
        { choices: [{ delta: { role: 'assistant' } }] },
        { choices: [{ delta: { reasoning_content: 'Checked the active thinking budget.' } }] },
        { choices: [{ delta: { content: 'Kiln serves one tuned model' } }] },
        { choices: [{ delta: { content: ' and learns from feedback live.' } }] },
        {
          choices: [{
            delta: {},
            finish_reason: 'stop',
            thinking_budget: smokeThinkingBudgetOutcome(metadata),
          }],
          metadata: { thinking_budget: metadata },
        },
      ]);
      return;
    }
    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    res.end(`No smoke stub for ${url.pathname}`);
  });

  await new Promise((accept, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', accept);
  });
  const address = server.address();
  if (!address || typeof address === 'string') fail('Could not bind local smoke server.');
  return {
    server,
    baseUrl: `http://127.0.0.1:${address.port}`,
    setFailDashboardApis: (value) => { apiState.failDashboardApis = value; },
    setFailRuntimeConfig: (value) => { apiState.failRuntimeConfig = value; },
    setRecentRequests: (rows) => { apiState.recentRequests = rows; },
    setModelsCold: (value) => { apiState.modelsCold = value; },
    setTrainingOptimizerAvailable: (value) => { apiState.trainingOptimizerAvailable = value; },
    setEvalSuites: (suites) => { apiState.evalSuites = suites.map((suite) => ({ ...suite })); },
    getTrainingSubmitRequests: () => ({ ...apiState.trainingSubmitRequests }),
    getModelsRequests: () => apiState.modelsRequests,
  };
}

async function expectText(page, selector, pattern, message) {
  const handle = await page.$(selector);
  if (!handle) fail(`${message}: missing selector ${selector}`);
  const textContent = await page.evaluate((el) => el.textContent || '', handle);
  if (!pattern.test(textContent)) fail(`${message}: selector ${selector} text was ${JSON.stringify(textContent.trim())}`);
}

async function expectNoForbiddenPublicityCopy(page, label) {
  const text = await page.evaluate(() => (document.body.innerText || '').replace(/\s+/g, ' ').trim().toLowerCase());
  for (const term of forbiddenPublicityTerms) {
    if (text.includes(term.toLowerCase())) {
      fail(`${label} should not use external publicity wording: ${term}`);
    }
  }
}

async function expectDisabled(page, selector, expected, message) {
  await page.waitForFunction(
    (targetSelector, targetDisabled) => {
      const element = document.querySelector(targetSelector);
      return element && Boolean(element.disabled) === targetDisabled;
    },
    { timeout: 5000 },
    selector,
    expected,
  ).catch(async () => {
    const actual = await page.$eval(selector, (el) => Boolean(el.disabled)).catch(() => 'missing');
    fail(`${message}: expected ${selector} disabled=${expected}, got ${actual}`);
  });
}

async function clickAndWait(page, selector, message) {
  const handle = await page.waitForSelector(selector, { visible: true, timeout: 5000 })
    .catch((error) => fail(`${message}: clickAndWait timed out for ${selector}: ${error.message}`));
  if (!handle) fail(`${message}: missing selector ${selector}`);
  await page.evaluate((element) => element.click(), handle);
}

async function goToPrimaryTab(page, name) {
  const pageId = `#page-${name}`;
  await page.evaluate((targetName) => {
    const tab = document.querySelector(`#primary-tab-${targetName}`);
    if (tab) tab.click();
  }, name);
  await page.waitForFunction(
    (selector) => {
      const section = document.querySelector(selector);
      return section
        && section.classList.contains('active')
        && !section.hidden
        && !section.hasAttribute('inert');
    },
    { timeout: 5000 },
    pageId,
  ).catch(() => fail(`Primary tab ${name} did not activate (${pageId})`));
}

// Asserts BOTH halves of hash navigation: the page section is active AND
// location.hash agrees. Used by the Back/Forward history assertions, where
// either half regressing (page without hash, hash without page) is a bug.
// `expectedHash` defaults to the page-level #name; deep-link assertions pass
// the full #page/subtab[/id] spelling.
async function expectActivePageAndHash(page, name, message, expectedHash = `#${name}`) {
  await page.waitForFunction(
    (targetName, targetHash) => {
      const section = document.querySelector(`#page-${targetName}`);
      return section
        && section.classList.contains('active')
        && !section.hidden
        && !section.hasAttribute('inert')
        && window.location.hash === targetHash;
    },
    { timeout: 5000 },
    name,
    expectedHash,
  ).catch(async () => {
    const actual = await page.evaluate(() => ({
      hash: window.location.hash,
      page: document.querySelector('.page.active')?.id || 'none',
    })).catch(() => ({ hash: 'unknown', page: 'unknown' }));
    fail(`${message}: expected page-${name} active with hash ${expectedHash}, got page=${actual.page} hash=${actual.hash}`);
  });
}

// Eval-drill + hash agreement for the deep-link close state machine: either
// half drifting (modal without its id hash, id hash without the modal) is
// exactly the class of bug PR 17 guards against.
async function expectEvalDrillState(page, { open, hash }, message) {
  await page.waitForFunction(
    (wantOpen, wantHash) => {
      const modal = document.getElementById('eval-drill-modal');
      return modal && modal.hidden === !wantOpen && window.location.hash === wantHash;
    },
    { timeout: 5000 },
    open,
    hash,
  ).catch(async () => {
    const actual = await page.evaluate(() => ({
      hidden: document.getElementById('eval-drill-modal')?.hidden,
      hash: window.location.hash,
    })).catch(() => ({ hidden: 'unknown', hash: 'unknown' }));
    fail(`${message}: expected drill ${open ? 'open' : 'closed'} with hash ${hash}, got modalHidden=${actual.hidden} hash=${actual.hash}`);
  });
}

async function waitForVisiblePanel(page, selector, message) {
  await page.waitForFunction(
    (panelSelector) => {
      const panel = document.querySelector(panelSelector);
      if (!panel || panel.hidden || panel.inert || !panel.classList.contains('active')) return false;
      const rect = panel.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    },
    { timeout: 5000 },
    selector,
  ).catch(() => fail(`${message}: ${selector} did not become active and visible`));
}

async function waitForVisibleElement(page, selector, message) {
  await page.waitForFunction(
    (targetSelector) => {
      const element = document.querySelector(targetSelector);
      if (!element) return false;
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
    },
    { timeout: 5000 },
    selector,
  ).catch(() => fail(`${message}: ${selector} did not become visible`));
}

async function waitForPanelText(page, selector, pattern, message) {
  await page.waitForFunction(
    (panelSelector, source) => {
      const element = document.querySelector(panelSelector);
      return element && new RegExp(source).test(element.textContent || '');
    },
    { timeout: 5000 },
    selector,
    pattern.source,
  ).catch(() => fail(`${message}: ${selector} did not render text matching ${pattern}`));
}

async function evalDrillTabbableState(page, focusTarget = null) {
  return page.evaluate((target) => {
    const modal = document.getElementById('eval-drill-modal');
    const selector = 'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"]), [contenteditable="true"]';
    const tabbables = Array.from(modal.querySelectorAll(selector)).filter((node) => {
      if (node.closest('[hidden]')) return false;
      const rect = node.getBoundingClientRect();
      return rect.width > 0 || rect.height > 0;
    });
    if (target === 'first') tabbables[0]?.focus();
    if (target === 'last') tabbables[tabbables.length - 1]?.focus();
    return {
      count: tabbables.length,
      activeIsFirst: document.activeElement === tabbables[0],
      activeIsLast: document.activeElement === tabbables[tabbables.length - 1],
      active: document.activeElement?.id || document.activeElement?.tagName || 'none',
    };
  }, focusTarget);
}

async function expectRocmMatmulRuntimeConfig(page, scenarioLabel) {
  const prefix = `${scenarioLabel} ROCm matmul runtime config`;
  const selector = '#runtime-config-body';
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Policy schema[\s\S]*kiln\.accelerator-runtime-policy\.v16[\s\S]*v16/, `${prefix} should render policy schema v16`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Vulkan kernel policy[\s\S]*kiln\.vulkan-kernel-policy\.v5/, `${prefix} should render the device-neutral Vulkan kernel policy schema`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Vulkan device policy[\s\S]*kiln\.vulkan-device-policy\.v1/, `${prefix} should render the immutable Vulkan device policy schema`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Vulkan physical device[\s\S]*automatic[\s\S]*config_file/, `${prefix} should render automatic Vulkan physical-device selection and source`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Vulkan validation[\s\S]*disabled[\s\S]*config_file/, `${prefix} should render Vulkan validation policy and source`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Kiln-tensor API routes[\s\S]*auto[\s\S]*default/, `${prefix} should render the immutable adapter route set and source`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Full-attention score ceiling[\s\S]*2,048 MiB[\s\S]*default/, `${prefix} should render immutable full-attention geometry and source`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*CUDA kernel profile[\s\S]*native_default[\s\S]*config_file/, `${prefix} should render the immutable CUDA backend route set and source`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Metal kernel profile[\s\S]*native_default[\s\S]*config_file/, `${prefix} should render the immutable Metal backend route set and source`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*CUDA graph policy[\s\S]*disabled[\s\S]*profile blocked/, `${prefix} should render the profile-resolved CUDA graph request`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*CUDA graph cache[\s\S]*8/, `${prefix} should render the typed CUDA graph cache bound`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*CUDA graph contract[\s\S]*stable metadata[\s\S]*single-row only/, `${prefix} should render fixed CUDA graph safety invariants`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*Strided batched matmul[\s\S]*disabled[\s\S]*default/, `${prefix} should render the portable strided-batched route and source`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*BF16 matmul output[\s\S]*f32_then_cast[\s\S]*default/, `${prefix} should render the portable BF16-output route and source`);
  await waitForPanelText(page, selector, /Accelerator execution[\s\S]*ROCm kernel profile[\s\S]*portable_fallback[\s\S]*config_file/, `${prefix} should render the portable ROCm kernel route set and source`);
}

async function expectStreamingPrefillRuntimeConfig(page, scenarioLabel) {
  const prefix = `${scenarioLabel} streaming-prefill runtime config`;
  await page.waitForFunction(() => Array.from(document.querySelectorAll('#runtime-config-body .rc-group'))
    .some((group) => group.querySelector('.rc-group-title')?.textContent?.trim() === 'Streaming prefill'),
  { timeout: 5000 }).catch(() => fail(`${prefix} group did not render`));
  const groupText = await page.$$eval('#runtime-config-body .rc-group', (groups) => groups
    .find((group) => group.querySelector('.rc-group-title')?.textContent?.trim() === 'Streaming prefill')
    ?.innerText || '');
  for (const [pattern, message] of [
    [/Dispatch input[\s\S]*uncached prompt tokens/, 'should identify the uncached-token dispatch input'],
    [/Policy consumers[\s\S]*inference \+ training/, 'should identify the shared inference and training policy'],
    [/Mode configured[\s\S]*auto[\s\S]*config_file/, 'should render configured mode and source'],
    [/Dispatch backend[\s\S]*at least 2,048 tokens[\s\S]*backend_policy/, 'should render backend dispatch and source'],
    [/Dispatch effective[\s\S]*at least 4,096 tokens[\s\S]*environment/, 'should render effective dispatch and source'],
    [/Threshold configured[\s\S]*4,096 tokens[\s\S]*environment/, 'should render configured threshold and source'],
    [/Threshold backend[\s\S]*2,048 tokens[\s\S]*backend_policy/, 'should render backend threshold and source'],
    [/Threshold effective[\s\S]*4,096 tokens[\s\S]*environment/, 'should render effective threshold and source'],
    [/Threshold override[\s\S]*applied/, 'should render threshold override resolution'],
    [/Base configured[\s\S]*4,096 tokens[\s\S]*config_file[\s\S]*Base backend[\s\S]*2,048 tokens[\s\S]*backend_policy[\s\S]*Base effective[\s\S]*4,096 tokens[\s\S]*config_file/, 'should render configured, backend, and effective base tiles'],
    [/Tape configured[\s\S]*auto[\s\S]*default[\s\S]*Tape backend[\s\S]*2,048 tokens[\s\S]*backend_policy[\s\S]*Tape effective[\s\S]*4,096 tokens[\s\S]*inherited_from_tile_tokens_config_file/, 'should render tape-tile inheritance'],
    [/Detached configured[\s\S]*auto[\s\S]*default[\s\S]*Detached backend[\s\S]*8,192 tokens[\s\S]*backend_policy[\s\S]*Detached effective[\s\S]*4,096 tokens[\s\S]*inherited_from_tile_tokens_config_file/, 'should render detached-tile inheritance'],
    [/Boundary configured[\s\S]*derived from detached[\s\S]*inherited_from_tile_tokens_config_file[\s\S]*Boundary backend[\s\S]*8,192 tokens[\s\S]*backend_policy[\s\S]*Boundary effective[\s\S]*4,096 tokens[\s\S]*inherited_from_tile_tokens_config_file/, 'should render derived boundary tiles and source'],
    [/Replay configured[\s\S]*derived from detached[\s\S]*inherited_from_tile_tokens_config_file[\s\S]*Replay backend[\s\S]*8,192 tokens[\s\S]*backend_policy[\s\S]*Replay effective[\s\S]*4,096 tokens[\s\S]*inherited_from_tile_tokens_config_file/, 'should render derived tape-replay tiles and source'],
    [/Last-token configured[\s\S]*on[\s\S]*default[\s\S]*Last-token effective[\s\S]*on[\s\S]*default/, 'should render last-token LM-head resolution'],
    [/Startup policy[\s\S]*immutable[\s\S]*Change requires restart[\s\S]*required/, 'should render immutable and restart-required lifecycle flags'],
  ]) {
    if (!pattern.test(groupText)) fail(`${prefix} ${message}; got ${JSON.stringify(groupText)}`);
  }
}

async function expectCheckpointBoundaryRuntimeConfig(page, scenarioLabel) {
  const prefix = `${scenarioLabel} checkpoint-boundary runtime config`;
  await page.waitForFunction(() => Array.from(document.querySelectorAll('#runtime-config-body .rc-group'))
    .some((group) => group.querySelector('.rc-group-title')?.textContent?.trim() === 'Training'),
  { timeout: 5000 }).catch(() => fail(`${prefix} group did not render`));
  const groupText = await page.$$eval('#runtime-config-body .rc-group', (groups) => groups
    .find((group) => group.querySelector('.rc-group-title')?.textContent?.trim() === 'Training')
    ?.innerText || '');
  for (const [pattern, message] of [
    [/Boundary mode[\s\S]*enabled/, 'should render the recompute mode'],
    [/Recompute threshold[\s\S]*4,096 tokens/, 'should render the automatic-mode threshold'],
    [/Anchor stride[\s\S]*3[\s\S]*explicit/, 'should distinguish an explicit anchor stride from auto'],
    [/Anchor cache target[\s\S]*6\.00 GiB/, 'should render the anchor-cache target in GiB'],
    [/Startup policy[\s\S]*immutable[\s\S]*Change requires restart[\s\S]*required/, 'should render immutable and restart-required lifecycle semantics'],
  ]) {
    if (!pattern.test(groupText)) fail(`${prefix} ${message}; got ${JSON.stringify(groupText)}`);
  }
}

async function expectCheckpointBoundaryRuntimeConfigLayout(page, scenarioLabel) {
  const layout = await page.evaluate(() => {
    const group = Array.from(document.querySelectorAll('#runtime-config-body .rc-group'))
      .find((candidate) => candidate.querySelector('.rc-group-title')?.textContent?.trim() === 'Training');
    if (!group) return { found: false };
    const bounds = group.getBoundingClientRect();
    const rows = Array.from(group.querySelectorAll('.rc-row')).map((row) => {
      const rect = row.getBoundingClientRect();
      const label = row.querySelector('.rc-label');
      const value = row.querySelector('.rc-value');
      const labelRect = label?.getBoundingClientRect();
      const valueRect = value?.getBoundingClientRect();
      const horizontalOverlap = labelRect && valueRect
        ? Math.min(labelRect.right, valueRect.right) - Math.max(labelRect.left, valueRect.left)
        : 0;
      return {
        name: label?.textContent?.trim() || '<missing>',
        left: rect.left,
        right: rect.right,
        top: rect.top,
        bottom: rect.bottom,
        scrollWidth: row.scrollWidth,
        clientWidth: row.clientWidth,
        labelValueOverlap: horizontalOverlap > 1,
      };
    });
    const rowOverlaps = [];
    for (let left = 0; left < rows.length; left += 1) {
      for (let right = left + 1; right < rows.length; right += 1) {
        if (Math.min(rows[left].bottom, rows[right].bottom) - Math.max(rows[left].top, rows[right].top) > 1) {
          rowOverlaps.push([rows[left].name, rows[right].name]);
        }
      }
    }
    return {
      found: true,
      clientWidth: group.clientWidth,
      scrollWidth: group.scrollWidth,
      outside: rows.filter((row) => row.left < bounds.left - 1 || row.right > bounds.right + 1).map((row) => row.name),
      overflowing: rows.filter((row) => row.scrollWidth > row.clientWidth + 1).map((row) => row.name),
      labelValueOverlaps: rows.filter((row) => row.labelValueOverlap).map((row) => row.name),
      rowOverlaps,
    };
  });
  if (!layout.found) fail(`${scenarioLabel} checkpoint-boundary runtime config layout is missing its Training group`);
  if (layout.scrollWidth > layout.clientWidth + 1
    || layout.outside.length
    || layout.overflowing.length
    || layout.labelValueOverlaps.length
    || layout.rowOverlaps.length) {
    fail(`${scenarioLabel} checkpoint-boundary diagnostics overlap or overflow: ${JSON.stringify(layout)}`);
  }
}

async function expectActiveTrainingTab(page, tabName, message) {
  await page.waitForFunction(
    (name) => {
      const tab = document.querySelector(`[data-tab="${name}"]`);
      const panel = document.querySelector(`#tab-${name}`);
      return tab?.getAttribute('aria-selected') === 'true'
        && panel
        && !panel.hidden
        && !panel.inert
        && panel.classList.contains('active');
    },
    { timeout: 5000 },
    tabName,
  ).catch(() => fail(message));
}

async function expectTrainingTabA11yState(page, activeName, message, { focused = true } = {}) {
  const state = await page.evaluate(() => {
    const tabNames = ['queue', 'sft', 'grpo', 'openenv'];
    return Object.fromEntries(tabNames.map((tabName) => {
      const tab = document.querySelector(`#training-tab-${tabName}`);
      const panel = document.querySelector(`#tab-${tabName}`);
      const rect = panel?.getBoundingClientRect();
      return [tabName, {
        ariaSelected: tab?.getAttribute('aria-selected'),
        classActive: Boolean(tab?.classList.contains('active')),
        focused: document.activeElement === tab,
        panelClassActive: Boolean(panel?.classList.contains('active')),
        panelHidden: Boolean(panel?.hidden),
        panelInert: Boolean(panel?.inert),
        panelVisible: Boolean(rect && rect.width > 0 && rect.height > 0),
        tabIndex: tab?.getAttribute('tabindex'),
      }];
    }));
  });

  for (const [name, tabState] of Object.entries(state)) {
    const isActive = name === activeName;
    if (tabState.ariaSelected !== String(isActive)) fail(`${message}: ${name} aria-selected=${tabState.ariaSelected}`);
    if (tabState.classActive !== isActive) fail(`${message}: ${name} active class=${tabState.classActive}`);
    if (tabState.tabIndex !== (isActive ? '0' : '-1')) fail(`${message}: ${name} tabindex=${tabState.tabIndex}`);
    if (tabState.panelClassActive !== isActive) fail(`${message}: ${name} panel active class=${tabState.panelClassActive}`);
    if (tabState.panelHidden !== !isActive) fail(`${message}: ${name} panel hidden=${tabState.panelHidden}`);
    if (tabState.panelInert !== !isActive) fail(`${message}: ${name} panel inert=${tabState.panelInert}`);
    if (isActive && !tabState.panelVisible) fail(`${message}: ${name} panel should be visible`);
    if (focused && isActive && !tabState.focused) fail(`${message}: ${name} tab should retain keyboard focus`);
    if (focused && !isActive && tabState.focused) fail(`${message}: ${name} inactive tab should not be focused`);
  }
}

async function expectTrainingTabKeyboardNavigation(page) {
  await page.focus('#training-tab-queue');
  await expectTrainingTabA11yState(page, 'queue', 'Queue should start as the active focused training tab');

  await page.keyboard.press('ArrowRight');
  await expectTrainingTabA11yState(page, 'sft', 'ArrowRight should activate the SFT training tab');

  await page.keyboard.press('ArrowRight');
  await expectTrainingTabA11yState(page, 'grpo', 'ArrowRight should activate the GRPO training tab');

  await page.keyboard.press('ArrowRight');
  await expectTrainingTabA11yState(page, 'openenv', 'ArrowRight should activate the OpenEnv training tab');

  await page.keyboard.press('ArrowLeft');
  await expectTrainingTabA11yState(page, 'grpo', 'ArrowLeft should return to the GRPO training tab');

  await page.keyboard.press('ArrowLeft');
  await expectTrainingTabA11yState(page, 'sft', 'ArrowLeft should return to the SFT training tab');

  await page.keyboard.press('Home');
  await expectTrainingTabA11yState(page, 'queue', 'Home should activate the Queue training tab');

  await page.keyboard.press('End');
  await expectTrainingTabA11yState(page, 'openenv', 'End should activate the OpenEnv training tab');
}

async function expectTrainingToast(page, text) {
  await page.waitForFunction(
    (expectedText) => Array.from(document.querySelectorAll('#toasts .toast')).some((toast) => toast.textContent?.trim() === expectedText),
    { timeout: 5000 },
    text,
  ).catch(async () => {
    const toasts = await page.evaluate(() => Array.from(document.querySelectorAll('#toasts .toast')).map((toast) => toast.textContent?.trim())).catch(() => []);
    fail(`Expected training success toast ${JSON.stringify(text)}, got ${JSON.stringify(toasts)}`);
  });
}

async function expectPanelLink(page, selector, label, href) {
  await page.waitForFunction(
    (panelSelector, expectedLabel, expectedHref) => {
      const panel = document.querySelector(panelSelector);
      if (!panel) return false;
      return Array.from(panel.querySelectorAll('a')).some((anchor) => (
        anchor.textContent?.trim() === expectedLabel && anchor.getAttribute('href') === expectedHref
      ));
    },
    { timeout: 5000 },
    selector,
    label,
    href,
  ).catch(() => fail(`${selector} missing expected ${label} link ${href}`));
}

async function expectHeaderHelpLinks(page, { visible = false } = {}) {
  const helpLinks = await page.$$eval('nav.header-help a', (links) => links.map((link) => {
    const rect = link.getBoundingClientRect();
    const style = window.getComputedStyle(link);
    return {
      text: link.textContent?.trim(),
      href: link.getAttribute('href'),
      visible: rect.width > 0
        && rect.height > 0
        && rect.bottom > 0
        && rect.right > 0
        && rect.top < window.innerHeight
        && rect.left < window.innerWidth
        && style.visibility !== 'hidden'
        && style.display !== 'none',
    };
  }));
  for (const [label, href] of expectedHeaderHelpLinks) {
    const link = helpLinks.find((candidate) => candidate.text === label && candidate.href === href);
    if (!link) fail(`nav.header-help missing expected link ${label} -> ${href}`);
    if (visible && !link.visible) fail(`nav.header-help link ${label} should be visible on mobile`);
  }
}

async function expectNoMobileOverflow(page) {
  const overflow = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    bodyScrollWidth: document.body.scrollWidth,
    bodyClientWidth: document.body.clientWidth,
  }));
  if (overflow.scrollWidth > overflow.clientWidth + 1 || overflow.bodyScrollWidth > overflow.bodyClientWidth + 1) {
    fail(`Mobile viewport has horizontal overflow at 390x844: document ${overflow.scrollWidth}/${overflow.clientWidth}, body ${overflow.bodyScrollWidth}/${overflow.bodyClientWidth}`);
  }
}

async function expectAdvancedTrainingLayout(page, kind, viewportLabel) {
  const layout = await page.evaluate((trainingKind) => {
    const body = document.getElementById(trainingKind + '-advanced');
    if (!body || body.hidden) return { visible: false };
    const bounds = body.getBoundingClientRect();
    const groups = Array.from(body.querySelectorAll('.form-group')).map((group) => {
      const rect = group.getBoundingClientRect();
      return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
    });
    const overlaps = [];
    for (let left = 0; left < groups.length; left += 1) {
      for (let right = left + 1; right < groups.length; right += 1) {
        const a = groups[left];
        const b = groups[right];
        if (Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1
          && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1) {
          overlaps.push([left, right]);
        }
      }
    }
    return {
      visible: true,
      bodyWidth: bounds.width,
      clientWidth: body.clientWidth,
      scrollWidth: body.scrollWidth,
      outside: groups.filter((rect) => rect.left < bounds.left - 1 || rect.right > bounds.right + 1),
      overlaps,
    };
  }, kind);
  if (!layout.visible) fail(`${viewportLabel} ${kind.toUpperCase()} advanced settings should be visible`);
  if (layout.scrollWidth > layout.clientWidth + 1) {
    fail(`${viewportLabel} ${kind.toUpperCase()} advanced settings overflow horizontally: ${JSON.stringify(layout)}`);
  }
  if (layout.outside.length || layout.overlaps.length) {
    fail(`${viewportLabel} ${kind.toUpperCase()} advanced controls overlap or escape their panel: ${JSON.stringify(layout)}`);
  }
}

async function expectMobilePanelFlow(page) {
  const tabPanels = [
    // Recent requests is promoted to the top of Overview — live agent traffic is
    // the primary thing a pi/opencode operator wants to see — followed by the
    // server-status and decode-performance panels.
    { tab: 'overview', selectors: ['#recent-requests-panel', '#server-status', '#decode-perf-panel'] },
    { tab: 'adapters', selectors: ['#adapters-panel'] },
    { tab: 'training', selectors: ['[data-training-tabs]'] },
    { tab: 'playground', selectors: ['#chat-output'] },
  ];

  for (const { tab, selectors } of tabPanels) {
    await goToPrimaryTab(page, tab);
    const panelFlow = await page.evaluate((panelSelectors) => panelSelectors.map((selector) => {
      const element = document.querySelector(selector);
      const panel = element?.closest('.panel') || element;
      const rect = panel?.getBoundingClientRect();
      return rect && {
        selector,
        left: Math.round(rect.left),
        top: Math.round(rect.top + window.scrollY),
        width: Math.round(rect.width),
      };
    }), selectors);

    if (panelFlow.some((panel) => !panel)) fail(`Mobile ${tab} tab is missing a main panel: ${JSON.stringify(panelFlow)}`);
    for (let index = 1; index < panelFlow.length; index += 1) {
      const previous = panelFlow[index - 1];
      const current = panelFlow[index];
      if (current.top <= previous.top) fail(`Mobile ${tab} panels should stack in source order: ${JSON.stringify(panelFlow)}`);
      if (Math.abs(current.left - panelFlow[0].left) > 2) fail(`Mobile ${tab} panels should align in one column: ${JSON.stringify(panelFlow)}`);
      if (current.width > 390) fail(`Mobile ${tab} panel exceeds viewport width: ${JSON.stringify(current)}`);
    }

    for (const selector of selectors) {
      await page.evaluate((targetSelector) => {
        const element = document.querySelector(targetSelector);
        const panel = element?.closest('.panel') || element;
        panel?.scrollIntoView({ block: 'center' });
      }, selector);
      await page.waitForFunction((targetSelector) => {
        const element = document.querySelector(targetSelector);
        const panel = element?.closest('.panel') || element;
        const rect = panel?.getBoundingClientRect();
        return Boolean(rect && rect.bottom > 0 && rect.top < window.innerHeight && rect.width > 0 && rect.height > 0);
      }, { timeout: 5000 }, selector).catch(() => fail(`Mobile ${tab} panel ${selector} should be reachable by scrolling`));
    }
  }
}

async function clickAdapterAction(page, adapterName, actionLabel) {
  const clicked = await page.evaluate((targetAdapterName, targetActionLabel) => {
    const items = Array.from(document.querySelectorAll('#adapters-panel .adapter-item'));
    const item = items.find((candidate) => candidate.querySelector('.adapter-name')?.textContent?.trim() === targetAdapterName);
    const button = Array.from(item?.querySelectorAll('button') || [])
      .find((candidate) => candidate.textContent?.trim() === targetActionLabel);
    if (!button || button.disabled) return false;
    button.click();
    return true;
  }, adapterName, actionLabel);
  if (!clicked) fail(`Could not click ${actionLabel} for ${adapterName}`);
}

async function expectAdapterAction(page, adapterName, actionLabel, message) {
  await page.waitForFunction(
    (targetAdapterName, targetActionLabel) => {
      const items = Array.from(document.querySelectorAll('#adapters-panel .adapter-item'));
      const item = items.find((candidate) => candidate.querySelector('.adapter-name')?.textContent?.trim() === targetAdapterName);
      return Array.from(item?.querySelectorAll('button') || [])
        .some((button) => button.textContent?.trim() === targetActionLabel && !button.disabled);
    },
    { timeout: 5000 },
    adapterName,
    actionLabel,
  ).catch(() => fail(message));
}

async function expectAdapterAbsent(page, adapterName, message) {
  await page.waitForFunction(
    (targetAdapterName) => !Array.from(document.querySelectorAll('#adapters-panel .adapter-name'))
      .some((name) => name.textContent?.trim() === targetAdapterName),
    { timeout: 5000 },
    adapterName,
  ).catch(() => fail(message));
}

function escapeRegExp(value) {
  return String(value).split('').map((char) => ('\\^$.*+?()[]{}|'.includes(char) ? `\\${char}` : char)).join('');
}

async function expectApiFailurePanel(page, selector, action, detail) {
  await page.waitForFunction(
    (panelSelector) => {
      const panel = document.querySelector(panelSelector);
      return panel && panel.querySelector('.api-failure');
    },
    { timeout: 5000 },
    selector,
  ).catch(() => fail(`${action} did not render an api-failure state in ${selector}`));

  await expectText(page, selector, new RegExp(`${action} could not load yet`, 'i'), `${action} failure copy missing panel/action name`);
  await expectText(page, selector, /Retry/i, `${action} failure copy missing retry affordance`);
  await expectText(page, selector, /Quickstart/, `${action} failure copy missing Quickstart link`);
  await expectText(page, selector, /Troubleshooting/, `${action} failure copy missing Troubleshooting link`);
  await expectText(page, selector, new RegExp(escapeRegExp(detail)), `${action} failure copy missing error detail`);
  await expectText(page, selector, /Smoke hint: retry after startup\./, `${action} failure copy missing the structured error hint`);
  const panelText = await page.$eval(selector, (el) => el.textContent || '');
  if (panelText.includes('[object Object]')) fail(`${action} failure panel rendered [object Object] — api() is not unwrapping the structured error body`);

  await expectPanelLink(page, `${selector} .api-failure`, 'Quickstart', 'https://ericflo.github.io/kiln/quickstart.html');
  await expectPanelLink(page, `${selector} .api-failure`, 'Troubleshooting', 'https://ericflo.github.io/kiln/troubleshooting.html');

  const retryLabel = await page.$eval(`${selector} .api-failure button`, (button) => ({
    text: button.textContent?.trim(),
    ariaLabel: button.getAttribute('aria-label'),
  })).catch(() => null);
  if (!retryLabel || retryLabel.text !== `Retry ${action}` || retryLabel.ariaLabel !== `Retry ${action}`) {
    fail(`${action} failure retry button should be labelled "Retry ${action}"`);
  }
}

// ── aria-live hygiene (roadmap PR 24) ───────────────────────────────────────
// Polled data panels must NOT be live regions: every content-keyed repaint
// would be re-read wholesale by screen readers every 2-3s. Each affected card
// owns a visually-hidden role="status" node instead, written only on genuine
// transitions; aria-busy flips only during a panel's FIRST load.
const polledDataPanelIds = [
  'tab-queue',
  'server-status',
  'decode-perf-panel',
  'recent-requests-panel',
  'adapters-panel',
  'recent-heartbeat',
];
const transitionStatusNodeIds = ['training-queue-status', 'eval-jobs-status', 'recent-requests-status'];

async function expectAriaLiveHygieneAtBoot(page) {
  // Static markup: the polled panels ship aria-busy="true" in index.html
  // itself, so "busy at boot" holds before app.js ever runs (the live DOM is
  // already past the first poll by the time puppeteer can look).
  const staticHtml = await readFile(uiIndexPath, 'utf8');
  for (const id of polledDataPanelIds) {
    const tag = staticHtml.match(new RegExp(`<[a-z][^>]*\\bid="${id}"[^>]*>`))?.[0];
    if (!tag) fail(`index.html is missing #${id}`);
    if (/aria-live/.test(tag)) fail(`#${id} is a polled data panel and must not be aria-live in static markup: ${tag}`);
    if (id !== 'recent-heartbeat' && !/aria-busy="true"/.test(tag)) {
      fail(`#${id} must boot with aria-busy="true" in static markup, got: ${tag}`);
    }
  }

  const state = await page.evaluate((panelIds, statusIds) => ({
    missingPanels: panelIds.filter((id) => !document.getElementById(id)),
    livePanels: panelIds.filter((id) => document.getElementById(id)?.hasAttribute('aria-live')),
    serverStatusBusy: document.getElementById('server-status')?.getAttribute('aria-busy'),
    statusNodes: statusIds.map((id) => {
      const el = document.getElementById(id);
      return { id, exists: Boolean(el), role: el?.getAttribute('role') || '', text: (el?.textContent || '').trim() };
    }),
    toastsRole: document.getElementById('toasts')?.getAttribute('role') || '',
    toastsLive: document.getElementById('toasts')?.getAttribute('aria-live') || '',
  }), polledDataPanelIds, transitionStatusNodeIds);
  if (state.missingPanels.length > 0) fail(`Polled panels missing from the DOM: ${state.missingPanels.join(', ')}`);
  if (state.livePanels.length > 0) fail(`Polled data panels must not be aria-live: ${state.livePanels.join(', ')}`);
  // First load finished before networkidle0 resolved, so the live DOM must
  // already be past busy — and (asserted later) it must never flip back.
  if (state.serverStatusBusy !== 'false') {
    fail(`#server-status aria-busy should be "false" after the first render, got "${state.serverStatusBusy}"`);
  }
  for (const node of state.statusNodes) {
    if (!node.exists) fail(`Missing visually-hidden status node #${node.id}`);
    if (node.role !== 'status') fail(`#${node.id} must have role="status", got "${node.role}"`);
    if (node.text !== '') fail(`#${node.id} must be empty at boot (no announcement before a real transition), got ${JSON.stringify(node.text)}`);
  }
  // Toasts are the intentional announcement channel — they must stay live.
  if (state.toastsRole !== 'status' || state.toastsLive !== 'polite') {
    fail(`#toasts must keep role="status" aria-live="polite", got role="${state.toastsRole}" aria-live="${state.toastsLive}"`);
  }

  await installServerStatusBusyObserver(page);
}

// Record every aria-busy value change on #server-status from install time on.
// The mock /health cycles blocks_used, so each 2s poll is a REAL repaint —
// aria-busy must still never return to "true" (first-load-only guard). Wiped
// by full page navigations; re-install after any page.goto/reload that
// precedes reading window.__serverStatusBusyFlips.
async function installServerStatusBusyObserver(page) {
  // Arm only after the first load completed — the boot true→false flip is
  // legitimate and must not pollute the "never flips again" record.
  await page.waitForFunction(
    () => document.getElementById('server-status')?.getAttribute('aria-busy') === 'false',
    { timeout: 5000 },
  ).catch(() => fail('#server-status never reached aria-busy="false" before arming the busy observer'));
  await page.evaluate(() => {
    const el = document.getElementById('server-status');
    window.__serverStatusBusyFlips = [];
    new MutationObserver((records) => {
      for (let i = 0; i < records.length; i += 1) {
        const next = i + 1 < records.length ? records[i + 1].oldValue : el.getAttribute('aria-busy');
        if (records[i].oldValue !== next) window.__serverStatusBusyFlips.push(`${records[i].oldValue}→${next}`);
      }
    }).observe(el, { attributes: true, attributeFilter: ['aria-busy'], attributeOldValue: true });
  });
}

async function expectStatusAnnouncement(page, nodeId, pattern, message) {
  await page.waitForFunction(
    (id, source) => new RegExp(source).test(document.getElementById(id)?.textContent || ''),
    { timeout: 6000 },
    nodeId,
    pattern.source,
  ).catch(async () => {
    const text = await page.$eval(`#${nodeId}`, (el) => el.textContent).catch(() => '<missing>');
    fail(`${message}: #${nodeId} should match ${pattern}, got ${JSON.stringify(text)}`);
  });
}

async function runMobileOnboardingSmoke(baseUrl) {
  const puppeteer = await loadPuppeteer();
  const browser = await puppeteer.launch({
    executablePath: chromiumPath(),
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const pageErrors = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', (error) => pageErrors.push(error.message));
    page.on('console', (entry) => {
      if (entry.type() === 'error') pageErrors.push(entry.text());
    });
    page.on('requestfailed', (request) => {
      pageErrors.push(`${request.method()} ${request.url()} failed: ${request.failure()?.errorText || 'unknown error'}`);
    });

    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true });
    await page.goto(`${baseUrl}/ui`, { waitUntil: 'networkidle0', timeout: 10000 });

    if (pageErrors.length > 0) fail(`Mobile UI emitted browser errors: ${pageErrors.join('; ')}`);
    await expectNoMobileOverflow(page);
    await waitForVisibleElement(page, '.header h1', 'Mobile header title did not render');
    await expectText(page, '.header h1', /^\s*kiln\s*$/i, 'Mobile header title text missing');
    await expectHeaderHelpLinks(page, { visible: true });
    await expectNoForbiddenPublicityCopy(page, 'Mobile server dashboard');
    await expectMobilePanelFlow(page);
    await goToPrimaryTab(page, 'overview');
    await clickAndWait(page, '#runtime-config > summary', 'Could not open the runtime config expander on mobile');
    await expectCheckpointBoundaryRuntimeConfig(page, 'Mobile');
    await expectCheckpointBoundaryRuntimeConfigLayout(page, 'Mobile');
    await expectNoMobileOverflow(page);
    await goToPrimaryTab(page, 'training');
    // The desktop pass earlier submitted jobs on this same server, so the
    // empty-queue→SFT landing (asserted there) doesn't apply here. Make the
    // starting sub-tab explicit before the keyboard checks.
    await clickAndWait(page, '#training-tab-queue', 'Could not activate Queue tab before keyboard checks');
    await expectTrainingTabKeyboardNavigation(page);
    await clickAndWait(page, '#training-tab-queue', 'Could not activate mobile Queue tab');
    await waitForVisiblePanel(page, '#tab-queue', 'Mobile Queue tab did not activate');
    await clickAndWait(page, '#training-tab-sft', 'Could not activate mobile SFT tab');
    await waitForVisiblePanel(page, '#tab-sft', 'Mobile SFT tab did not activate');
    await clickAndWait(page, '#sft-adv-toggle', 'Could not open mobile SFT advanced settings');
    await expectAdvancedTrainingLayout(page, 'sft', 'Mobile');
    await expectNoMobileOverflow(page);
    await clickAndWait(page, '#training-tab-grpo', 'Could not activate mobile GRPO tab');
    await waitForVisiblePanel(page, '#tab-grpo', 'Mobile GRPO tab did not activate');
    await clickAndWait(page, '#grpo-adv-toggle', 'Could not open mobile GRPO advanced settings');
    await expectAdvancedTrainingLayout(page, 'grpo', 'Mobile');
    await expectNoMobileOverflow(page);

    await goToPrimaryTab(page, 'playground');
    await page.select('#chat-thinking-budget-tokens-mode', 'limit');
    await page.select('#chat-thinking-budget-time-mode', 'limit');
    await page.waitForFunction(() => (
      document.getElementById('chat-advanced')?.hidden === false
      && document.getElementById('chat-thinking-budget-tokens')?.disabled === false
      && document.getElementById('chat-thinking-budget-seconds')?.disabled === false
    ), { timeout: 5000 }).catch(() => fail('Mobile finite thinking-budget controls should be visible and editable'));
    await page.$eval('#chat-thinking-budget-tokens', (input) => {
      input.value = '131072';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.$eval('#chat-thinking-budget-seconds', (input) => {
      input.value = '86400';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.waitForFunction(() => (
      document.getElementById('chat-thinking-budget-preview-tokens')?.textContent === '131,072'
      && document.getElementById('chat-thinking-budget-preview-time')?.textContent === '86,400 s'
    ), { timeout: 5000 }).catch(() => fail('Mobile thinking-budget preview should render the maximum finite pair'));
    await expectNoMobileOverflow(page);
  } finally {
    await browser.close();
  }
}

// ── Bug B: cold-start model-id resolution ──────────────────────────────
// Open the dashboard while /v1/models still 503s (weights loading): the
// Connect snippets render the fallback id. Heal the endpoint → the next
// 2s health poll piggybacks a /v1/models retry, and the copyable model-id
// field + snippets silently upgrade to the real id without a reload. Once
// resolved, the retry stops for good.
async function runModelColdStartSmoke(baseUrl, { setModelsCold, getModelsRequests }) {
  const puppeteer = await loadPuppeteer();
  const browser = await puppeteer.launch({
    executablePath: chromiumPath(),
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const pageErrors = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', (error) => pageErrors.push(error.message));
    page.on('console', (entry) => {
      if (entry.type() !== 'error') return;
      const text = entry.text();
      // The cold phase intentionally 503s /v1/models — that resource noise
      // is the fixture, not a dashboard defect.
      if (/Failed to load resource: the server responded with a status of 503/.test(text)) return;
      pageErrors.push(text);
    });

    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
    await page.goto(`${baseUrl}/ui`, { waitUntil: 'networkidle0', timeout: 10000 });

    // Cold: the panel runs on the fallback id.
    await page.waitForFunction(
      () => document.getElementById('connect-model')?.textContent === 'Qwen3.5-4B',
      { timeout: 8000 },
    ).catch(() => fail('Connect model id should show the fallback while /v1/models is cold'));
    const coldSnippets = await page.$eval('#connect-snippets', (el) => el.textContent || '');
    if (!coldSnippets.includes('Qwen3.5-4B')) fail('Connect snippets should render the fallback model id during cold start');
    if (coldSnippets.includes('Qwen3.5-4B-resolved')) fail('Connect snippets must not know the real model id before /v1/models answers');
    await clickAndWait(page, '#runtime-config > summary', 'Could not open the runtime config expander during model cold start');
    await expectCheckpointBoundaryRuntimeConfig(page, 'Cold-start');

    // Heal. The piggybacked retry on the 2s health poll must upgrade the
    // copyable model-id field and the rendered snippets without a reload.
    setModelsCold(false);
    await page.waitForFunction(
      () => document.getElementById('connect-model')?.textContent === 'Qwen3.5-4B-resolved',
      { timeout: 10000 },
    ).catch(() => fail('Connect model id did not upgrade to the served id after /v1/models recovered (no-reload retry broken)'));
    const warmSnippets = await page.$eval('#connect-snippets', (el) => el.textContent || '');
    if (!warmSnippets.includes('Qwen3.5-4B-resolved')) fail('Connect snippets did not re-render with the served model id after cold start');
    // The fallback id is a PREFIX of the resolved one — assert no snippet
    // still carries a bare fallback (every occurrence must be the resolved id).
    const bareFallbacks = warmSnippets.split('Qwen3.5-4B').slice(1).filter((tail) => !tail.startsWith('-resolved')).length;
    if (bareFallbacks > 0) fail(`Connect snippets still contain ${bareFallbacks} stale fallback model id(s) after resolution`);

    // Resolved → the retry stops: no further /v1/models hits across >2 polls.
    const settled = getModelsRequests();
    await new Promise((resolve) => setTimeout(resolve, 5200));
    const after = getModelsRequests();
    if (after !== settled) fail(`/v1/models retry did not stop after resolution (${settled} → ${after} requests)`);

    if (pageErrors.length > 0) fail(`Cold-start UI emitted browser errors: ${pageErrors.join('; ')}`);
  } finally {
    await browser.close();
  }
}

async function runSmoke(baseUrl, {
  expectFailureStates = false,
  expectEmptyAdapters = false,
  setFailDashboardApis = null,
  setFailRuntimeConfig = null,
  setRecentRequests = null,
  setTrainingOptimizerAvailable = null,
  setEvalSuites = null,
  getTrainingSubmitRequests = null,
} = {}) {
  const puppeteer = await loadPuppeteer();
  const browser = await puppeteer.launch({
    executablePath: chromiumPath(),
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const pageErrors = [];
  let expectedRuntimeConfigFailure = false;
  try {
    const page = await browser.newPage();
    page.on('pageerror', (error) => pageErrors.push(error.message));
    page.on('console', (entry) => {
      if (entry.type() !== 'error') return;
      const text = entry.text();
      if ((expectFailureStates || expectedRuntimeConfigFailure)
        && /Failed to load resource: the server responded with a status of 503/.test(text)) return;
      pageErrors.push(text);
    });
    page.on('requestfailed', (request) => {
      pageErrors.push(`${request.method()} ${request.url()} failed: ${request.failure()?.errorText || 'unknown error'}`);
    });

    await page.evaluateOnNewDocument(() => { window.__kilnThinkingBudgetTest = {}; });
    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
    await page.goto(`${baseUrl}/ui`, { waitUntil: 'networkidle0', timeout: 10000 });

    if (pageErrors.length > 0) fail(`UI emitted browser errors: ${pageErrors.join('; ')}`);

    await expectText(page, '.header h1', /^\s*kiln\s*$/i, 'Header did not render');
    await expectHeaderHelpLinks(page);
    await expectNoForbiddenPublicityCopy(page, 'Server dashboard');
    await expectAriaLiveHygieneAtBoot(page);

    if (expectFailureStates) {
      await goToPrimaryTab(page, 'overview');
      await expectApiFailurePanel(page, '#server-status', 'Server status', 'Server status smoke failure from /health');
      await expectApiFailurePanel(page, '#decode-perf-panel', 'Decode performance', 'Decode performance smoke failure from /v1/stats/decode');
      await expectApiFailurePanel(page, '#recent-requests-panel', 'Recent requests', 'Recent requests smoke failure from /v1/stats/recent-requests');

      // Opening the runtime-config expander while /v1/config 503s must
      // render a quiet retry line INSIDE the expander — the Server-status
      // card keeps its own failure state, nothing throws (the pageErrors
      // assertion at the end of this scenario backs that up).
      await clickAndWait(page, '#runtime-config > summary', 'Could not open the runtime config expander in the failure scenario');
      await waitForPanelText(page, '#runtime-config-body', /Couldn't load \/v1\/config/, 'Runtime config should render its graceful failure copy');
      await page.$('#runtime-config [data-rc-refresh]')
        .then((handle) => { if (!handle) fail('Runtime config failure copy should offer a Retry button'); });
      await goToPrimaryTab(page, 'playground');
      await page.waitForFunction(() => (
        document.getElementById('chat-thinking-budget-preview')?.dataset.state === 'defaults unavailable'
        && document.getElementById('chat-thinking-budget-refresh')?.hidden === false
      ), { timeout: 5000 }).catch(() => fail('Playground should show unavailable inherited defaults with a Retry button'));

      await goToPrimaryTab(page, 'adapters');
      await expectApiFailurePanel(page, '#adapters-panel', 'Adapters', 'Adapters smoke failure from /v1/adapters');
      await goToPrimaryTab(page, 'training');
      await expectApiFailurePanel(page, '#tab-queue', 'Training queue', 'Training queue smoke failure from /v1/train/queue');

      // Retry buttons must dispatch through the app's delegated listener.
      // (The app is IIFE-scoped: the old inline onclick threw ReferenceError
      // on every click, which the pageErrors assertion below would catch.)
      await page.click('#tab-queue [data-retry]')
        .catch((e) => fail(`Training queue failure panel has no clickable Retry button: ${e.message}`));
      await new Promise((resolve) => setTimeout(resolve, 300));
      if (pageErrors.length > 0) fail(`Retry click emitted browser errors: ${pageErrors.join('; ')}`);

      if (setFailDashboardApis) {
        const recoverPanel = async (selector, pattern, message) => {
          // Click Retry for an instant repaint when the button is still there;
          // the interval polls (2-5s) cover the already-recovered case.
          await page.click(`${selector} [data-retry]`).catch(() => {});
          await waitForPanelText(page, selector, pattern, message);
        };

        // Heal the APIs: every panel must recover (Retry click or next poll).
        setFailDashboardApis(false);
        await goToPrimaryTab(page, 'playground');
        await page.click('#chat-thinking-budget-refresh')
          .catch(() => fail('Could not retry Playground thinking-budget defaults after the APIs healed'));
        await page.waitForFunction(() => (
          document.getElementById('chat-thinking-budget-preview')?.dataset.state === 'ready'
          && document.getElementById('chat-thinking-budget-preview-tokens')?.textContent === '64'
          && document.getElementById('chat-thinking-budget-preview-time')?.textContent === '1.5 s'
        ), { timeout: 5000 }).catch(() => fail('Playground thinking-budget defaults did not recover after Retry'));
        await goToPrimaryTab(page, 'overview');
        await recoverPanel('#server-status', /GPU VRAM/, 'Server status did not recover after the APIs healed');
        await page.waitForSelector('#server-status .vram-donut svg', { timeout: 5000 })
          .catch(() => fail('VRAM donut missing from recovered server status panel'));
        // The expander's Retry refetches /v1/config (no poll loop covers it).
        await page.click('#runtime-config [data-rc-refresh]')
          .catch(() => fail('Could not click the runtime config Retry button after the APIs healed'));
        await waitForPanelText(page, '#runtime-config-body', /linux-drm-sysfs-unified/, 'Runtime config did not recover after Retry');
        await expectCheckpointBoundaryRuntimeConfig(page, 'Recovered failure');
        await recoverPanel('#decode-perf-panel', /No recent token gaps/i, 'Decode panel did not recover after the APIs healed');
        await recoverPanel('#recent-requests-panel', /No recent requests yet\./, 'Recent requests did not recover after the APIs healed');
        await goToPrimaryTab(page, 'training');
        await recoverPanel('#tab-queue', /No training jobs yet\./, 'Training queue did not recover after the APIs healed');
        await goToPrimaryTab(page, 'adapters');
        await recoverPanel('#adapters-panel', /adapter-alpha/, 'Adapters did not recover after the APIs healed');

        // Break the APIs again, then heal again. The second recovery is the
        // real regression test for the stuck-panel class: by now every render
        // dedupe key holds the pre-failure value, so a failure writer that
        // forgets to invalidate its key leaves the panel frozen on the error
        // HTML forever (the data hasn't changed, so the repaint is skipped).
        setFailDashboardApis(true);
        await goToPrimaryTab(page, 'overview');
        await expectApiFailurePanel(page, '#server-status', 'Server status', 'Server status smoke failure from /health');
        await expectApiFailurePanel(page, '#decode-perf-panel', 'Decode performance', 'Decode performance smoke failure from /v1/stats/decode');
        await expectApiFailurePanel(page, '#recent-requests-panel', 'Recent requests', 'Recent requests smoke failure from /v1/stats/recent-requests');
        await goToPrimaryTab(page, 'training');
        await expectApiFailurePanel(page, '#tab-queue', 'Training queue', 'Training queue smoke failure from /v1/train/queue');
        await goToPrimaryTab(page, 'adapters');
        await expectApiFailurePanel(page, '#adapters-panel', 'Adapters', 'Adapters smoke failure from /v1/adapters');

        setFailDashboardApis(false);
        await recoverPanel('#adapters-panel', /adapter-alpha/, 'Adapters stuck on failure HTML after second recovery (dedupe key not invalidated)');
        await goToPrimaryTab(page, 'training');
        await recoverPanel('#tab-queue', /No training jobs yet\./, 'Training queue stuck on failure HTML after second recovery (dedupe key not invalidated)');
        await goToPrimaryTab(page, 'overview');
        await recoverPanel('#server-status', /GPU VRAM/, 'Server status stuck on failure HTML after second recovery');
        await recoverPanel('#decode-perf-panel', /No recent token gaps/i, 'Decode panel stuck on failure HTML after second recovery');
        await recoverPanel('#recent-requests-panel', /No recent requests yet\./, 'Recent requests stuck on failure HTML after second recovery (dedupe key not invalidated)');
      }

      if (pageErrors.length > 0) fail(`Failure state UI emitted browser errors: ${pageErrors.join('; ')}`);
      return;
    }

    if (expectEmptyAdapters) {
      await goToPrimaryTab(page, 'adapters');
      await waitForPanelText(page, '#adapters-panel', /No adapters found yet\./, 'Empty adapter state missing');
      await expectPanelLink(page, '#adapters-panel .empty', 'Quickstart', 'https://ericflo.github.io/kiln/quickstart.html');
      await expectPanelLink(page, '#adapters-panel .empty', 'Troubleshooting', 'https://ericflo.github.io/kiln/troubleshooting.html');
      await expectDisabled(page, '#merge-btn', true, 'Adapter merge should stay disabled when fewer than two adapters exist');

      // ── Journey-strip truth (Bug A), novice golden path ────────────────
      // No adapters trained yet → the strip stays on screen, so the agent
      // step's honesty is directly observable through every transition.
      if (setRecentRequests) {
        await goToPrimaryTab(page, 'overview');
        await page.waitForFunction(
          () => { const strip = document.getElementById('journey-strip'); return strip && !strip.hidden; },
          { timeout: 8000 },
        ).catch(() => fail('Journey strip should be visible on a fresh server with no adapters'));

        // Dashboard-only traffic (Test connection / Playground) must NOT
        // complete "Agent connected" nor collapse the Connect panel.
        setRecentRequests([dashboardRow()]);
        await waitForPanelText(page, '#recent-requests-panel', /Reply with the single word: connected/, 'Dashboard-client row did not render in recent requests');
        const dashOnly = await page.evaluate(() => ({
          agentDone: !!document.querySelector('#journey-strip [data-journey="agent"]')?.classList.contains('is-done'),
          sub: document.querySelector('#journey-strip [data-journey="agent"] .journey-sub')?.textContent || '',
          connectExpandedHidden: !!document.getElementById('connect-expanded')?.hidden,
          fwAgents: document.getElementById('fw-agents')?.textContent || '',
        }));
        if (dashOnly.agentDone) fail('Dashboard-only rows must NOT complete the "Agent connected" milestone');
        if (dashOnly.sub !== 'point pi or opencode here') fail(`Agent step sub-text should stay default on dashboard-only rows, got "${dashOnly.sub}"`);
        if (dashOnly.connectExpandedHidden) fail('Connect panel must not auto-collapse on dashboard-only traffic');
        if (dashOnly.fwAgents !== '0') fail(`Flywheel client count should ignore dashboard rows, got "${dashOnly.fwAgents}"`);

        // A curl row IS an external connection: the milestone completes,
        // with the inline (not hover-only) "next move" hint.
        setRecentRequests([dashboardRow(), curlRow()]);
        await page.waitForFunction(
          () => {
            const strip = document.getElementById('journey-strip');
            const step = strip?.querySelector('[data-journey="agent"]');
            const sub = step?.querySelector('.journey-sub')?.textContent || '';
            return strip && !strip.hidden && !!step?.classList.contains('is-done')
              && sub === 'curl seen — point a coding agent here next';
          },
          { timeout: 8000 },
        ).catch(() => fail('A curl row should complete "Agent connected" with the inline coding-agent hint'));
        const collapsedAfterCurl = await page.evaluate(() => !document.getElementById('connect-collapsed')?.hidden);
        if (!collapsedAfterCurl) fail('Connect panel should auto-collapse once external (curl) traffic flows');

        // A recognized coding agent arrives → the curl hint clears, and pi
        // leads the client-chip enumeration (hard preference).
        setRecentRequests([dashboardRow(), curlRow(), piRow()]);
        await page.waitForFunction(
          () => {
            const step = document.querySelector('#journey-strip [data-journey="agent"]');
            const sub = step?.querySelector('.journey-sub')?.textContent || '';
            return !!step?.classList.contains('is-done') && sub === 'point pi or opencode here';
          },
          { timeout: 8000 },
        ).catch(() => fail('The curl hint should clear once a recognized coding agent connects'));
        const chips = await page.$$eval('#recent-requests-panel .agent-chip', (els) => els.map((el) => el.textContent.trim()));
        if (chips.length === 0) fail('Client filter chips should render when multiple clients are present');
        if (!(chips[1] || '').startsWith('pi')) fail(`pi should lead the client chips after "All agents", got ${JSON.stringify(chips)}`);
        if (!chips.some((chip) => chip.startsWith('dashboard'))) fail(`Dashboard rows should be labeled honestly in the client chips, got ${JSON.stringify(chips)}`);
      }

      if (pageErrors.length > 0) fail(`Empty adapter UI emitted browser errors: ${pageErrors.join('; ')}`);
      return;
    }

    // ── Journey-strip truth (Bug A), default scenario ───────────────────
    // Dashboard-client rows must not complete "Agent connected"; a curl row
    // must. With smoke adapters already present, the curl row completes all
    // three milestones, so the strip retires itself — that retirement IS the
    // milestone assertion here (the empty-adapter scenario watches the step
    // classes directly while the strip stays visible).
    if (setRecentRequests) {
      setRecentRequests([dashboardRow()]);
      await waitForPanelText(page, '#recent-requests-panel', /Reply with the single word: connected/, 'Dashboard-client row did not render in recent requests');
      const dashOnly = await page.evaluate(() => ({
        stripHidden: !document.getElementById('journey-strip') || document.getElementById('journey-strip').hidden,
        agentDone: !!document.querySelector('#journey-strip [data-journey="agent"]')?.classList.contains('is-done'),
        connectExpandedHidden: !!document.getElementById('connect-expanded')?.hidden,
      }));
      if (dashOnly.stripHidden) fail('Journey strip should stay visible while only dashboard-client rows exist');
      if (dashOnly.agentDone) fail('Dashboard-only rows must NOT complete the "Agent connected" milestone');
      if (dashOnly.connectExpandedHidden) fail('Connect panel must not auto-collapse on dashboard-only traffic');

      setRecentRequests([dashboardRow(), curlRow()]);
      await page.waitForFunction(
        () => document.getElementById('journey-strip')?.hidden === true,
        { timeout: 8000 },
      ).catch(() => fail('A curl row should complete "Agent connected" (all milestones done → journey strip retires)'));
      const collapsedAfterCurl = await page.evaluate(() => !document.getElementById('connect-collapsed')?.hidden);
      if (!collapsedAfterCurl) fail('Connect panel should auto-collapse once external (curl) traffic flows');

      // Drain the fixture rows so the later empty-state assertions hold.
      setRecentRequests([]);
      await waitForPanelText(page, '#recent-requests-panel', /No recent requests yet\./, 'Recent requests did not drain after the journey-strip truth checks');

      // ── Recent-requests status line: announce only on attention CHANGES ──
      // Routine traffic first: a clean row arrives, the needs-attention count
      // stays 0 → the status node must stay silent across the next poll tick.
      const cleanRow = piRow({ latency: smokeRequestLatency() });
      setRecentRequests([cleanRow]);
      await waitForPanelText(page, '#recent-requests-panel', /hello from pi/, 'Clean pi row did not render before the attention-announcement checks');
      await clickAndWait(
        page,
        `#recent-requests-panel .recent-row[data-id="${cleanRow.id}"]`,
        'Could not open the recent-request drill for thinking-budget telemetry',
      );
      await page.waitForFunction(
        () => document.getElementById('request-drill-modal')?.hidden === false
          && !!document.querySelector('#request-drill-content [data-request-thinking-budget]'),
        { timeout: 5000 },
      ).catch(() => fail('Request drill did not render the thinking-budget section'));
      const budgetText = await page.$eval(
        '#request-drill-content [data-request-thinking-budget]',
        (el) => el.textContent || '',
      );
      for (const expected of [
        '64 tokens · request',
        '1.5 s · server default',
        'Applied',
        'Yes',
        'Token limit · closed',
        '64 tokens · 800 ms',
      ]) {
        if (!budgetText.includes(expected)) {
          fail(`Request drill thinking-budget telemetry is missing ${JSON.stringify(expected)}: ${JSON.stringify(budgetText)}`);
        }
      }
      const latencyText = await page.$eval(
        '#request-drill-content [data-request-latency]',
        (el) => el.textContent || '',
      ).catch(() => fail('Request drill did not render latency diagnosis'));
      for (const expected of [
        'p99.9 ITL',
        '30 ms',
        'Stalls',
        'Actor prefill',
        'GPU lock wait',
        'Synchronization',
        '7 of 7 request-local gaps retained',
        'Not measured on this path: Sampling, Readback, Graph capture',
      ]) {
        if (!latencyText.includes(expected)) {
          fail(`Request latency diagnosis is missing ${JSON.stringify(expected)}: ${JSON.stringify(latencyText)}`);
        }
      }
      await clickAndWait(page, '#request-drill-close', 'Could not close the recent-request drill');
      await page.waitForFunction(
        () => document.getElementById('request-drill-modal')?.hidden === true,
        { timeout: 5000 },
      ).catch(() => fail('Request drill did not close after telemetry assertions'));
      await new Promise((resolve) => setTimeout(resolve, 2500)); // ≥1 recent-requests poll
      const routineText = await page.$eval('#recent-requests-status', (el) => el.textContent || '');
      if (routineText.trim() !== '') fail(`Routine traffic must not announce; #recent-requests-status got ${JSON.stringify(routineText)}`);

      // An errored row flips the count 0→1: exactly one terse line, counting
      // arrivals since the last announcement plus the attention total.
      setRecentRequests([smokeRecentRow({ user_agent: 'pi/1.2.0', prompt_preview: 'attention row', finish_reason: 'error' }), cleanRow]);
      await expectStatusAnnouncement(page, 'recent-requests-status', /needs attention\./, 'Attention-count increase was not announced');

      // Clearing the errored row flips 1→0: the recovery is announced too.
      setRecentRequests([cleanRow]);
      await expectStatusAnnouncement(page, 'recent-requests-status', /no requests need attention\./, 'Attention-count recovery was not announced');

      // A coherent attributed stall exercises the complete closed reason
      // vocabulary in the drill, independent of the routine-traffic status
      // assertions above.
      const diagnosticLatency = smokeRequestLatency();
      diagnosticLatency.itl_ms_p99 = 282.6;
      diagnosticLatency.itl_ms_p999 = 298.3;
      diagnosticLatency.max_itl_ms = 300;
      diagnosticLatency.stall_count = 1;
      diagnosticLatency.stall_reasons.synchronization = 1;
      diagnosticLatency.phases.decode_ms = 320;
      diagnosticLatency.phases.synchronization_ms = 280;
      const diagnosticRow = piRow({
        id: 'chatcmpl-latency-diagnostic',
        prompt_preview: 'diagnose backend pause',
        latency: diagnosticLatency,
      });
      setRecentRequests([diagnosticRow]);
      await waitForPanelText(page, '#recent-requests-panel', /diagnose backend pause/, 'Attributed-stall row did not render');
      await clickAndWait(
        page,
        `#recent-requests-panel .recent-row[data-id="${diagnosticRow.id}"]`,
        'Could not open the attributed-stall request drill',
      );
      const diagnosticText = await page.$eval(
        '#request-drill-content [data-request-latency]',
        (el) => el.textContent || '',
      ).catch(() => fail('Attributed-stall drill did not render latency diagnosis'));
      for (const expected of ['300 ms', 'synchronization 1', 'Synchronization', '280 ms']) {
        if (!diagnosticText.includes(expected)) {
          fail(`Attributed-stall diagnosis is missing ${JSON.stringify(expected)}: ${JSON.stringify(diagnosticText)}`);
        }
      }
      await clickAndWait(page, '#request-drill-close', 'Could not close the attributed-stall drill');

      // Re-drain so the later empty-state assertions hold.
      setRecentRequests([]);
      await waitForPanelText(page, '#recent-requests-panel', /No recent requests yet\./, 'Recent requests did not drain after the attention-announcement checks');
    }

    // --- Hash navigation: tab clicks mint history entries, browser
    // Back/Forward walks them, and live hash edits route through the
    // page whitelist (roadmap PR 16). Deep links extend the grammar to
    // #page/subtab[/id] (roadmap PR 17): sub-tabs and drill modals are
    // addressable, junk segments repair to the canonical spelling via
    // replaceState, and a user-closed modal CONSUMES the history entry
    // its open minted. ---
    await expectActivePageAndHash(page, 'overview', 'Landing on /ui (no fragment) should repair the URL to #overview in place');

    // ---- Deep-link boots (PR 17). On a pre-PR17 build the whole fragment
    // fails the page whitelist and falls back to #overview — the negative
    // check for this branch. ----
    await page.goto('about:blank');
    await page.goto(`${baseUrl}/ui#training/sft`, { waitUntil: 'domcontentloaded' });
    await expectActivePageAndHash(page, 'training', 'Deep-link boot #training/sft should land on the Training page with the hash intact', '#training/sft');
    await expectActiveTrainingTab(page, 'sft', 'Deep-link boot #training/sft should activate the SFT sub-tab');

    // Boot straight onto the evals Jobs sub-tab; the fixture job renders.
    await page.goto('about:blank');
    await page.goto(`${baseUrl}/ui#evals/jobs`, { waitUntil: 'domcontentloaded' });
    await expectActivePageAndHash(page, 'evals', 'Deep-link boot #evals/jobs should land on the Jobs sub-tab', '#evals/jobs');
    await waitForVisiblePanel(page, '#tab-evals-jobs', 'Deep-link boot #evals/jobs should activate the Jobs tab panel');
    await waitForPanelText(page, '#eval-jobs-list', /smoke-suite/, 'Jobs list should render the smoke eval job');

    // Mint a sub-tab trail so Back-after-close has a meaningful place to land.
    await clickAndWait(page, '#evals-tab-datasets', 'Could not click the Datasets sub-tab');
    await expectActivePageAndHash(page, 'evals', 'Sub-tab click should push #evals/datasets', '#evals/datasets');
    await clickAndWait(page, '#evals-tab-jobs', 'Could not click back to the Jobs sub-tab');
    await expectActivePageAndHash(page, 'evals', 'Sub-tab click should push #evals/jobs', '#evals/jobs');

    // Open the eval drill: the job id lands in the hash (pushState).
    await clickAndWait(page, '#eval-jobs-list .job-card[data-job-id="smoke-eval-full"]', 'Could not open the eval drill modal from the jobs list');
    await expectEvalDrillState(page, { open: true, hash: '#evals/jobs/smoke-eval-full' }, 'Opening the eval drill should push the job id into the hash');

    // Escape-close CONSUMES the modal entry (open pushed → close walks back)…
    await page.keyboard.press('Escape');
    await expectEvalDrillState(page, { open: false, hash: '#evals/jobs' }, 'Escape should close the eval drill and return the hash to #evals/jobs');
    // …so Back now keeps walking pages instead of re-opening the modal.
    await page.goBack();
    await expectEvalDrillState(page, { open: false, hash: '#evals/datasets' }, 'Back after close should land on the prior sub-tab, not re-open the modal');
    await page.goForward();
    await expectEvalDrillState(page, { open: false, hash: '#evals/jobs' }, 'Forward should return to the Jobs sub-tab with the modal still closed');
    // The modal entry itself stays LIVE in forward history (never dead):
    // Forward onto it re-opens the drill.
    await page.goForward();
    await expectEvalDrillState(page, { open: true, hash: '#evals/jobs/smoke-eval-full' }, 'Forward onto the modal entry should re-open the eval drill');
    // Closing a modal re-entered via history (we pushed nothing this time)
    // repairs the entry in place instead of walking back out of the app.
    await page.keyboard.press('Escape');
    await expectEvalDrillState(page, { open: false, hash: '#evals/jobs' }, 'Escape on a history-reopened drill should close it and repair the hash in place');

    // ---- Shared modal manager (roadmap PR 18): focus moves INTO the
    // dialog on open and back to the pre-open element on close, Tab is
    // trapped within the top modal, and stacked modals (palette over a
    // drill) close outside-in with the body scroll lock held until the
    // stack empties. The hash assertions interleaved below prove the
    // manager composes with the deep-link close state machine instead of
    // replacing it. ----

    // Focus a known element, then open the drill with a programmatic
    // .click() (which does not move focus) so the manager must capture
    // THIS element as the restore target.
    await page.focus('#evals-tab-jobs');
    await page.evaluate(() => document.querySelector('#eval-jobs-list .job-card[data-job-id="smoke-eval-full"]').click());
    await expectEvalDrillState(page, { open: true, hash: '#evals/jobs/smoke-eval-full' }, 'Opening the eval drill for the focus checks should push the job id into the hash');
    const drillFocus = await page.evaluate(() => ({
      inModal: document.getElementById('eval-drill-modal').contains(document.activeElement),
      active: document.activeElement?.id || document.activeElement?.tagName || 'none',
    }));
    if (!drillFocus.inModal) fail(`Opening the eval drill should move focus into the dialog, got activeElement=${drillFocus.active}`);

    // Tab trap: from the LAST tabbable a real Tab press wraps to the
    // first; Shift+Tab from the first wraps back to the last. Outcomes
    // must finish rendering first so the tabbable set is stable.
    await page.waitForFunction(() => document.querySelector('#eval-drill-modal .outcome-item'), { timeout: 5000 })
      .catch(() => fail('Eval drill outcomes did not render before the Tab-trap checks'));
    const trapReady = await evalDrillTabbableState(page, 'last');
    if (trapReady.count < 2) fail('Eval drill should expose at least two tabbables for the Tab-trap checks');
    await page.keyboard.press('Tab');
    const forwardFocus = await evalDrillTabbableState(page);
    if (!forwardFocus.activeIsFirst) {
      fail(`Tab from the last tabbable should wrap to the current first inside the eval drill, got activeElement=${forwardFocus.active}`);
    }
    await evalDrillTabbableState(page, 'first');
    await page.keyboard.down('Shift');
    await page.keyboard.press('Tab');
    await page.keyboard.up('Shift');
    const backwardFocus = await evalDrillTabbableState(page);
    if (!backwardFocus.activeIsLast) {
      fail(`Shift+Tab from the first tabbable should wrap to the current last inside the eval drill, got activeElement=${backwardFocus.active}`);
    }

    // Escape closes the drill through its own user-close fn (the hash
    // entry is consumed exactly like the X button) AND focus returns to
    // the element that was focused before the open.
    await page.keyboard.press('Escape');
    await expectEvalDrillState(page, { open: false, hash: '#evals/jobs' }, 'Escape via the modal manager should close the eval drill and consume the hash entry');
    const restoredFocus = await page.evaluate(() => document.activeElement?.id || 'none');
    if (restoredFocus !== 'evals-tab-jobs') fail(`Closing the eval drill should return focus to the pre-open element #evals-tab-jobs, got ${restoredFocus}`);

    // Stacking: the command palette over the drill. Escape peels layers
    // outside-in (palette first, drill second) and the body scroll lock
    // holds until the LAST layer closes.
    await page.evaluate(() => document.querySelector('#eval-jobs-list .job-card[data-job-id="smoke-eval-full"]').click());
    await expectEvalDrillState(page, { open: true, hash: '#evals/jobs/smoke-eval-full' }, 'Re-opening the eval drill for the stacking checks should push the job id');
    const lockedWithDrill = await page.evaluate(() => document.body.style.overflow);
    if (lockedWithDrill !== 'hidden') fail(`Opening the eval drill should lock body scroll, got overflow=${JSON.stringify(lockedWithDrill)}`);
    await page.keyboard.down('Control');
    await page.keyboard.press('k');
    await page.keyboard.up('Control');
    await page.waitForFunction(() => document.getElementById('cmdk-modal')?.hidden === false, { timeout: 5000 })
      .catch(() => fail('Ctrl+K should open the command palette over the eval drill'));
    const cmdkFocus = await page.evaluate(() => document.activeElement?.id || 'none');
    if (cmdkFocus !== 'cmdk-input') fail(`Opening the palette should focus its input, got ${cmdkFocus}`);
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => document.getElementById('cmdk-modal')?.hidden === true, { timeout: 5000 })
      .catch(() => fail('Escape should close the stacked command palette'));
    const afterCmdkClose = await page.evaluate(() => ({
      drillHidden: document.getElementById('eval-drill-modal')?.hidden,
      overflow: document.body.style.overflow,
      hash: window.location.hash,
    }));
    if (afterCmdkClose.drillHidden !== false) fail('Escape over a stacked palette should close ONLY the palette — the eval drill underneath must stay open');
    if (afterCmdkClose.hash !== '#evals/jobs/smoke-eval-full') fail(`Closing the palette must not touch the drill's hash entry, got ${afterCmdkClose.hash}`);
    if (afterCmdkClose.overflow !== 'hidden') fail(`Body scroll must stay locked while the drill is still open under the closed palette, got overflow=${JSON.stringify(afterCmdkClose.overflow)}`);
    await page.keyboard.press('Escape');
    await expectEvalDrillState(page, { open: false, hash: '#evals/jobs' }, 'The second Escape should close the drill itself and consume its hash entry');
    const unlockedAfterStack = await page.evaluate(() => document.body.style.overflow);
    if (unlockedAfterStack !== '') fail(`Body scroll lock should release once the modal stack empties, got overflow=${JSON.stringify(unlockedAfterStack)}`);

    // Junk sub-tab: the page still activates, the segment is repaired via
    // replaceState to a real sub-tab spelling (which one depends on the
    // localStorage restore + empty-queue redirect, both legitimate).
    await page.goto('about:blank');
    await page.goto(`${baseUrl}/ui#training/bogus`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.querySelector('#page-training')?.classList.contains('active')
        && /^#training\/(queue|sft|grpo)$/.test(window.location.hash),
      { timeout: 5000 },
    ).catch(async () => {
      const got = await page.evaluate(() => ({ hash: window.location.hash, page: document.querySelector('.page.active')?.id || 'none' })).catch(() => ({}));
      fail(`Junk sub-tab deep link #training/bogus should keep Training and repair the hash to a real sub-tab, got page=${got.page} hash=${got.hash}`);
    });
    // Arrow-key tab navigation mints hashes too (the write lives in the
    // tab-select fn, so keyboard nav and programmatic .click() are covered).
    await clickAndWait(page, '#training-tab-sft', 'Could not activate the SFT tab for keyboard hash checks');
    await page.focus('#training-tab-sft');
    await page.keyboard.press('ArrowRight');
    await expectActivePageAndHash(page, 'training', 'Arrow-key sub-tab navigation should push #training/grpo', '#training/grpo');

    // Adapter deep link (a page with no sub-tabs: #adapters/{name} is a
    // drill id): the modal opens over the Adapters page.
    await page.goto('about:blank');
    await page.goto(`${baseUrl}/ui#adapters/adapter-alpha`, { waitUntil: 'domcontentloaded' });
    await expectActivePageAndHash(page, 'adapters', 'Deep-link boot #adapters/adapter-alpha should land on Adapters with the hash intact', '#adapters/adapter-alpha');
    await page.waitForFunction(
      () => document.getElementById('adapter-drill-modal')?.hidden === false
        && document.getElementById('adapter-drill-title')?.textContent === 'adapter-alpha',
      { timeout: 5000 },
    ).catch(() => fail('Adapter deep link should open the adapter drill modal for adapter-alpha'));
    // X-closing a BOOT-opened modal has no entry of ours to consume — the
    // hash repairs in place to the parent page instead of history.back()
    // (which would exit the dashboard).
    await clickAndWait(page, '#adapter-drill-close', 'Could not close the deep-linked adapter drill modal');
    await page.waitForFunction(
      () => document.getElementById('adapter-drill-modal')?.hidden === true && window.location.hash === '#adapters',
      { timeout: 5000 },
    ).catch(async () => {
      const got = await page.evaluate(() => window.location.hash).catch(() => 'unknown');
      fail(`Closing a boot-deep-linked adapter drill should repair the hash to #adapters, got ${got}`);
    });

    // ---- PR 16 page-level walk, re-run on a pristine boot (localStorage
    // cleared so lastPage / sub-tab restores don't steer the canonical
    // hashes minted below). ----
    await page.evaluate(() => localStorage.clear());
    await page.goto('about:blank');
    await page.goto(`${baseUrl}/ui`, { waitUntil: 'domcontentloaded' });
    await expectActivePageAndHash(page, 'overview', 'A pristine /ui boot should land on #overview');
    // The Training click below asserts the canonical ONE-entry push
    // (#training/sft via the empty-queue redirect); that redirect requires a
    // LOADED queue cache, so gate on the queue render first.
    await waitForPanelText(page, '#tab-queue', /No training jobs yet\./, 'Training queue poll should land before the hash-navigation walk');
    await goToPrimaryTab(page, 'adapters');
    await expectActivePageAndHash(page, 'adapters', 'Clicking the Adapters tab should push #adapters');
    await goToPrimaryTab(page, 'training');
    await expectActivePageAndHash(page, 'training', 'Clicking the Training tab should push ONE canonical entry — #training/sft via the empty-queue redirect, never #training plus a second sub-tab entry', '#training/sft');
    await page.goBack();
    await expectActivePageAndHash(page, 'adapters', 'Browser Back from Training should return to Adapters');
    await page.goBack();
    await expectActivePageAndHash(page, 'overview', 'Second browser Back should return to Overview');
    await page.goForward();
    await expectActivePageAndHash(page, 'adapters', 'Browser Forward should re-land on Adapters');
    // A live hash edit (address bar / location.hash) must activate the page
    // and canonicalize the bare page hash to its sub-tab spelling in place.
    await page.evaluate(() => { window.location.hash = '#evals'; });
    await expectActivePageAndHash(page, 'evals', 'Setting location.hash = #evals should activate the Evals page and canonicalize the hash', '#evals/datasets');
    // A junk hash falls back to Overview and is repaired via replaceState —
    // the junk entry must NOT survive in history for Back to trip over.
    await page.evaluate(() => { window.location.hash = '#nonsense'; });
    await expectActivePageAndHash(page, 'overview', 'A junk hash should fall back to Overview with the URL repaired');
    await page.goBack();
    await expectActivePageAndHash(page, 'evals', 'Back after a junk hash should land on Evals — #nonsense must not pollute history', '#evals/datasets');

    // --- Shared tablist keyboard contract (roadmap PR 19): every
    // [role=tablist] — primary nav, training (asserted separately via
    // expectTrainingTabKeyboardNavigation), evals, distill, Connect —
    // answers ArrowLeft/ArrowRight/Home/End with automatic activation
    // (selection follows focus) and a roving tabindex. ---

    // Primary nav: arrows walk the pages through the same selectPage click
    // path, so pages activate AND the canonical hashes get minted.
    await goToPrimaryTab(page, 'overview');
    await expectActivePageAndHash(page, 'overview', 'Primary-nav keyboard checks should start from Overview');
    await page.focus('#primary-tab-overview');
    await page.keyboard.press('ArrowRight');
    await expectActivePageAndHash(page, 'adapters', 'ArrowRight on the primary nav should activate the Adapters page with its hash');
    const navArrowState = await page.evaluate(() => ({
      focused: document.activeElement?.id || null,
      adaptersTabIndex: document.getElementById('primary-tab-adapters')?.tabIndex,
      overviewTabIndex: document.getElementById('primary-tab-overview')?.tabIndex,
    }));
    if (navArrowState.focused !== 'primary-tab-adapters') fail(`ArrowRight should move focus to the Adapters nav tab, got ${navArrowState.focused}`);
    if (navArrowState.adaptersTabIndex !== 0 || navArrowState.overviewTabIndex !== -1) {
      fail(`Primary-nav roving tabindex should follow selection, got adapters=${navArrowState.adaptersTabIndex} overview=${navArrowState.overviewTabIndex}`);
    }
    await page.keyboard.press('ArrowLeft');
    await expectActivePageAndHash(page, 'overview', 'ArrowLeft on the primary nav should return to Overview');
    await page.keyboard.press('End');
    await expectActivePageAndHash(page, 'terminal', 'End on the primary nav should jump to the last page (pi Terminal)');
    await page.keyboard.press('Home');
    await expectActivePageAndHash(page, 'overview', 'Home on the primary nav should jump back to Overview');

    // Connect panel tabs: the JS-rendered snippet panes carry real tabpanel
    // semantics paired to the tab button ids, and arrows walk the clients.
    // (Earlier journey-strip traffic auto-collapsed the panel — re-expand.)
    await page.evaluate(() => window.openConnect());
    await page.waitForFunction(() => document.getElementById('connect-expanded')?.hidden === false, { timeout: 5000 })
      .catch(() => fail('Connect panel did not expand for the tablist keyboard checks'));
    const connectPaneAria = await page.evaluate(() => Array.from(document.querySelectorAll('.connect-snippet')).map((pane) => ({
      key: pane.dataset.connectPane,
      role: pane.getAttribute('role'),
      id: pane.id,
      labelledby: pane.getAttribute('aria-labelledby'),
      tabControls: document.getElementById(`connect-tab-${pane.dataset.connectPane}`)?.getAttribute('aria-controls'),
    })));
    if (connectPaneAria.length === 0) fail('Connect snippet panes did not render');
    for (const pane of connectPaneAria) {
      if (pane.role !== 'tabpanel') fail(`Connect pane ${pane.key} should carry role=tabpanel, got ${JSON.stringify(pane.role)}`);
      if (pane.labelledby !== `connect-tab-${pane.key}`) fail(`Connect pane ${pane.key} aria-labelledby should point at its tab, got ${JSON.stringify(pane.labelledby)}`);
      if (pane.tabControls !== pane.id) fail(`Connect tab ${pane.key} aria-controls should point at its pane id ${pane.id}, got ${JSON.stringify(pane.tabControls)}`);
    }
    await clickAndWait(page, '#connect-tab-pi', 'Could not activate the pi connect tab before keyboard checks');
    await page.focus('#connect-tab-pi');
    await page.keyboard.press('ArrowRight');
    const connectArrowState = await page.evaluate(() => ({
      selected: document.getElementById('connect-tab-opencode')?.getAttribute('aria-selected'),
      focused: document.activeElement?.id || null,
      tabIndex: document.getElementById('connect-tab-opencode')?.tabIndex,
      piTabIndex: document.getElementById('connect-tab-pi')?.tabIndex,
      paneActive: !!document.getElementById('connect-snippet-opencode')?.classList.contains('active'),
      paneVisible: (() => {
        const rect = document.getElementById('connect-snippet-opencode')?.getBoundingClientRect();
        return Boolean(rect && rect.width > 0 && rect.height > 0);
      })(),
    }));
    if (connectArrowState.selected !== 'true') fail(`ArrowRight should select the opencode connect tab, aria-selected=${connectArrowState.selected}`);
    if (connectArrowState.focused !== 'connect-tab-opencode') fail(`ArrowRight should focus the opencode connect tab, got ${connectArrowState.focused}`);
    if (connectArrowState.tabIndex !== 0 || connectArrowState.piTabIndex !== -1) {
      fail(`Connect-tabs roving tabindex should follow selection, got opencode=${connectArrowState.tabIndex} pi=${connectArrowState.piTabIndex}`);
    }
    if (!connectArrowState.paneActive || !connectArrowState.paneVisible) fail('ArrowRight should reveal the opencode snippet pane');
    await page.keyboard.press('Home');
    const connectHomeState = await page.evaluate(() => ({
      selected: document.getElementById('connect-tab-pi')?.getAttribute('aria-selected'),
      paneActive: !!document.getElementById('connect-snippet-pi')?.classList.contains('active'),
    }));
    if (connectHomeState.selected !== 'true' || !connectHomeState.paneActive) fail('Home should return the connect tabs to pi with its pane visible');

    // Distill: the headline fix. Its 12 sub-tabs were completely
    // unreachable by keyboard (roving tabindex=-1 with no arrow-key
    // handler). Arrows now activate them, the decorative group
    // labels/separators inside the tablist are skipped, and — per the
    // deep-link wiring — every keyboard activation mints #distill/<tab>.
    await goToPrimaryTab(page, 'distill');
    await clickAndWait(page, '#distill-tab-opd', 'Could not activate the Distill (OPD) tab before keyboard checks');
    await page.focus('#distill-tab-opd');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await expectActivePageAndHash(page, 'distill', 'ArrowRight x2 from Distill should select the third tab (Boost) and mint its deep-link hash', '#distill/pump');
    const distillBoostState = await page.evaluate(() => {
      const tab = document.getElementById('distill-tab-pump');
      const pane = document.getElementById('distill-tab-pump-pane');
      const rect = pane?.getBoundingClientRect();
      return {
        selected: tab?.getAttribute('aria-selected'),
        focused: document.activeElement === tab,
        tabIndex: tab?.tabIndex,
        opdSelected: document.getElementById('distill-tab-opd')?.getAttribute('aria-selected'),
        opdTabIndex: document.getElementById('distill-tab-opd')?.tabIndex,
        paneHidden: Boolean(pane?.hidden),
        paneVisible: Boolean(rect && rect.width > 0 && rect.height > 0),
      };
    });
    if (distillBoostState.selected !== 'true') fail(`ArrowRight x2 should select the Boost tab, aria-selected=${distillBoostState.selected}`);
    if (!distillBoostState.focused) fail('Boost tab should hold keyboard focus after arrow navigation');
    if (distillBoostState.tabIndex !== 0 || distillBoostState.opdTabIndex !== -1) {
      fail(`Distill roving tabindex should follow selection, got pump=${distillBoostState.tabIndex} opd=${distillBoostState.opdTabIndex}`);
    }
    if (distillBoostState.opdSelected !== 'false') fail('OPD tab should deselect when Boost activates');
    if (distillBoostState.paneHidden || !distillBoostState.paneVisible) fail('Boost pane should be visible after arrow-key activation');
    // Crossing a group boundary: Merge → Teachers skips the decorative
    // label/separator spans inside the tablist.
    await clickAndWait(page, '#distill-tab-merge', 'Could not activate the Merge distill tab');
    await page.focus('#distill-tab-merge');
    await page.keyboard.press('ArrowRight');
    await expectActivePageAndHash(page, 'distill', 'ArrowRight from Merge should skip the group label/separator spans and land on Teachers', '#distill/teachers');
    await page.keyboard.press('End');
    await expectActivePageAndHash(page, 'distill', 'End should jump to the last distill tab (Agent runs)', '#distill/runs');
    const runsPaneVisible = await page.evaluate(() => {
      const pane = document.getElementById('distill-tab-runs-pane');
      const rect = pane?.getBoundingClientRect();
      return Boolean(pane && !pane.hidden && rect && rect.width > 0 && rect.height > 0);
    });
    if (!runsPaneVisible) fail('End should reveal the Agent runs pane');
    await page.keyboard.press('Home');
    await expectActivePageAndHash(page, 'distill', 'Home should jump back to the first distill tab', '#distill/opd');

    await goToPrimaryTab(page, 'adapters');
    await waitForPanelText(page, '#adapters-panel', /adapter-alpha/, 'Adapter list should show the first smoke adapter');
    await waitForPanelText(page, '#adapters-panel', /adapter-beta/, 'Adapter list should show the second smoke adapter');

    // Swap interaction: non-active cards offer "Make active" (hot-swap), the
    // active card offers "Unload (use base)" — plain-language state labels.
    await clickAdapterAction(page, 'adapter-alpha', 'Make active');
    await expectTrainingToast(page, "adapter-alpha is now serving — pi's next request uses it");
    await expectAdapterAction(page, 'adapter-alpha', 'Unload (use base)', 'Loaded adapter should refresh as active with an Unload button');
    await clickAdapterAction(page, 'adapter-alpha', 'Unload (use base)');
    await expectTrainingToast(page, 'Adapter unloaded — requests now use the base model');
    await expectAdapterAction(page, 'adapter-alpha', 'Make active', 'Unloaded adapter should refresh with a Make active button');

    const downloadResponsePromise = page.waitForResponse(
      (response) => response.url().endsWith('/v1/adapters/adapter-beta/download') && response.status() === 200,
      { timeout: 5000 },
    );
    await clickAdapterAction(page, 'adapter-beta', 'Download');
    const downloadResponse = await downloadResponsePromise.catch(() => fail('Adapter download did not request /v1/adapters/adapter-beta/download'));
    const downloadHeaders = downloadResponse.headers();
    if (!/^attachment\b/i.test(downloadHeaders['content-disposition'] || '')) fail('Adapter download should return an attachment content-disposition header');
    if (!/^application\/gzip\b/i.test(downloadHeaders['content-type'] || '')) fail('Adapter download should return an archive content-type header');
    if (Number(downloadHeaders['content-length'] || 0) <= 0) fail('Adapter download should return non-empty archive bytes');
    await page.goto(`${baseUrl}/ui`, { waitUntil: 'domcontentloaded' });
    // The reload wiped the boot-time aria-busy observer — re-arm it so the
    // overview steady-state assertions below can read it.
    await installServerStatusBusyObserver(page);
    await goToPrimaryTab(page, 'adapters');
    await waitForPanelText(page, '#adapters-panel', /adapter-alpha/, 'Adapter list should reload after adapter download');
    await waitForPanelText(page, '#adapters-panel', /adapter-beta/, 'Adapter list should still include adapter-beta after download');

    await expectDisabled(page, '#upload-adapter-btn', true, 'Adapter upload should start disabled until name and archive are provided');
    const uploadFixtureDir = await mkdtemp(join(tmpdir(), 'kiln-ui-upload-'));
    try {
      const uploadBytes = Buffer.from('tiny adapter archive\n');
      const uploadFixture = join(uploadFixtureDir, 'uploaded-smoke-adapter.tgz');
      await writeFile(uploadFixture, uploadBytes);
      await page.type('#upload-name', 'uploaded-smoke-adapter');
      await expectDisabled(page, '#upload-adapter-btn', true, 'Adapter upload should stay disabled until an archive is attached');
      const uploadInput = await page.$('#upload-archive');
      if (!uploadInput) fail('Adapter upload archive input missing');
      await uploadInput.uploadFile(uploadFixture);
      await expectDisabled(page, '#upload-adapter-btn', false, 'Adapter upload should enable after path-safe name and archive are provided');
      const uploadResponsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/v1/adapters/upload'),
        { timeout: 5000 },
      );
      await clickAndWait(page, '#upload-adapter-btn', 'Could not submit adapter upload');
      await expectDisabled(page, '#upload-adapter-btn', true, 'Adapter upload should disable while submitting');
      const uploadResponse = await uploadResponsePromise.catch(() => fail('Adapter upload did not request /v1/adapters/upload'));
      if (uploadResponse.status() !== 200) {
        const uploadDetail = await uploadResponse.text().catch(() => '');
        fail(`Adapter upload should return 200, got ${uploadResponse.status()}: ${uploadDetail}`);
      }
      await expectTrainingToast(page, `Uploaded uploaded-smoke-adapter (${uploadBytes.length} B, 2 files)`);
      await waitForPanelText(page, '#adapters-panel', /uploaded-smoke-adapter/, 'Adapter list should refresh with the uploaded smoke adapter');
    } finally {
      await rm(uploadFixtureDir, { recursive: true, force: true });
    }

    // ---- Adapter receipt viewer in the drill modal: adapter-alpha carries
    // a §8.11 receipt.json → provenance fields + raw-JSON toggle render; the
    // just-uploaded adapter 404s → the graceful no-receipt copy renders and
    // the rest of the modal is unaffected.
    const openAdapterDrill = async (name) => {
      const clicked = await page.evaluate((target) => {
        const card = document.querySelector(`#adapters-panel .adapter-card[data-adapter-name="${target}"]`);
        if (!card) return false;
        card.click();
        return true;
      }, name);
      if (!clicked) fail(`Could not find the adapter card for ${name}`);
      await page.waitForFunction(() => document.getElementById('adapter-drill-modal')?.hidden === false, { timeout: 5000 })
        .catch(() => fail(`Adapter drill modal did not open for ${name}`));
    };
    const closeAdapterDrill = async () => {
      await clickAndWait(page, '#adapter-drill-close', 'Could not close the adapter drill modal');
      await page.waitForFunction(() => document.getElementById('adapter-drill-modal')?.hidden === true, { timeout: 5000 })
        .catch(() => fail('Adapter drill modal did not close'));
    };

    await openAdapterDrill('adapter-alpha');
    await waitForPanelText(page, '#adapter-receipt-section', /Trained via/, 'Receipt section should render provenance for adapter-alpha');
    await waitForPanelText(page, '#adapter-receipt-section', /opd/, 'Receipt should render the source kind');
    await waitForPanelText(page, '#adapter-receipt-section', /kiln-canonical:math_reasoning:v3/, 'Receipt should render the dataset source');
    await waitForPanelText(page, '#adapter-receipt-section', /128 prompts/, 'Receipt should render the prompt count');
    await waitForPanelText(page, '#adapter-receipt-section', /qwen3\.6-27b@openrouter/, 'Receipt should render the teacher alias');
    await waitForPanelText(page, '#adapter-receipt-section', /lora_rank/, 'Receipt should render key hyperparameters');
    await clickAndWait(page, '#adapter-receipt-section [data-receipt-raw]', 'Could not toggle the receipt raw JSON');
    const receiptRawShown = await page.$eval(
      '#adapter-receipt-section [data-receipt-raw-pre]',
      (el) => !el.hidden && /"schema_version"/.test(el.textContent || ''),
    );
    if (!receiptRawShown) fail('Receipt raw JSON toggle should reveal the pretty-printed receipt payload');
    await closeAdapterDrill();

    await openAdapterDrill('uploaded-smoke-adapter');
    await waitForPanelText(page, '#adapter-receipt-section', /No receipt — uploaded or legacy adapter/, 'A 404 receipt should render the graceful no-receipt copy');
    await waitForPanelText(page, '#adapter-drill-content', /Files on disk/, 'The drill modal body should render fine alongside a 404 receipt');
    await closeAdapterDrill();

    page.once('dialog', async (dialog) => {
      if (!/Delete adapter "adapter-beta"\?/.test(dialog.message())) fail(`Unexpected delete confirmation text: ${dialog.message()}`);
      await dialog.accept();
    });
    const deleteRequestPromise = page.waitForRequest(
      (request) => request.method() === 'DELETE' && request.url().endsWith('/v1/adapters/adapter-beta'),
      { timeout: 5000 },
    );
    await clickAdapterAction(page, 'adapter-beta', 'Delete');
    await deleteRequestPromise.catch(() => fail('Adapter delete did not send DELETE /v1/adapters/adapter-beta'));
    await expectTrainingToast(page, 'Deleted adapter: adapter-beta');
    await expectAdapterAbsent(page, 'adapter-beta', 'Deleted adapter should disappear after list refresh');

    await waitForPanelText(page, '#merge-helper', /Select at least two source adapters/i, 'Adapter merge helper should ask for source selections before merge');
    await expectDisabled(page, '#add-merge-source', false, 'Add merge source should enable when at least two adapters exist');
    await expectDisabled(page, '#merge-btn', true, 'Adapter merge should stay disabled before source adapters are selected');
    await page.select('#merge-src-name-1', 'adapter-alpha');
    await page.select('#merge-src-name-2', 'uploaded-smoke-adapter');

    // Regression: the 5s adapters poll re-renders merge sources; it must not
    // steal focus (or wipe selections) from someone mid-edit in a weight input.
    // (Triple-click doesn't select-all in number inputs — use el.select().)
    await page.$eval('#merge-src-weight-1', (el) => { el.focus(); el.select(); });
    await page.keyboard.type('0.75');
    await new Promise((resolve) => setTimeout(resolve, 5600)); // sit through one adapters poll
    const mergeFocusState = await page.evaluate(() => ({
      focusedId: document.activeElement?.id || null,
      weight: document.getElementById('merge-src-weight-1')?.value,
      src1: document.getElementById('merge-src-name-1')?.value,
      src2: document.getElementById('merge-src-name-2')?.value,
    }));
    if (mergeFocusState.focusedId !== 'merge-src-weight-1') fail(`Merge weight input lost focus to ${mergeFocusState.focusedId || 'nothing'} during the adapters poll`);
    if (mergeFocusState.weight !== '0.75') fail(`Merge weight value lost during the adapters poll: "${mergeFocusState.weight}"`);
    if (mergeFocusState.src1 !== 'adapter-alpha' || mergeFocusState.src2 !== 'uploaded-smoke-adapter') fail(`Merge source selections lost during the adapters poll: ${mergeFocusState.src1}/${mergeFocusState.src2}`);

    await page.click('#merge-output-name', { clickCount: 3 });
    await page.type('#merge-output-name', 'merged-smoke-adapter');
    await expectDisabled(page, '#merge-btn', false, 'Adapter merge should enable after two distinct sources and path-safe output are selected');
    await clickAndWait(page, '#merge-btn', 'Could not submit adapter merge');
    await expectDisabled(page, '#merge-btn', true, 'Adapter merge should disable while submitting');
    await expectTrainingToast(page, 'Merged 2 sources → merged-smoke-adapter (32 tensors, mode=weighted_average)');

    await goToPrimaryTab(page, 'training');
    await waitForPanelText(page, '#tab-queue', /No training jobs yet\./, 'Empty training queue state missing');
    await expectPanelLink(page, '#tab-queue .empty', 'Quickstart', 'https://ericflo.github.io/kiln/quickstart.html');
    await expectPanelLink(page, '#tab-queue .empty', 'GRPO Guide', 'https://ericflo.github.io/kiln/grpo.html');

    await goToPrimaryTab(page, 'overview');
    await waitForPanelText(page, '#recent-requests-panel', /No recent requests yet\./, 'Empty recent requests state missing');
    await expectPanelLink(page, '#recent-requests-panel .empty', 'Quickstart', 'https://ericflo.github.io/kiln/quickstart.html');

    await waitForPanelText(page, '#decode-perf-panel', /No recent token gaps/i, 'Empty decode performance state missing');
    await expectPanelLink(page, '#decode-perf-panel', '/health', '/health');

    // ---- Hidden diagnostics: Prometheus /metrics line in the Connect panel
    // with a live-origin scrape config behind the standard copy button.
    await waitForPanelText(page, '#connect-metrics', /Prometheus metrics at/, 'Connect panel missing the /metrics line');
    await expectPanelLink(page, '#connect-metrics', '/metrics', '/metrics');
    const scrapeSnippet = await page.$eval('#connect-metrics-snippet', (el) => el.innerText || '');
    if (!scrapeSnippet.includes(`targets: ["${new URL(baseUrl).host}"]`)) {
      fail(`Prometheus scrape config should target the live origin, got: ${JSON.stringify(scrapeSnippet)}`);
    }
    if (!(await page.$('#connect-metrics [data-copy-code]'))) fail('Prometheus scrape config snippet missing its copy button');

    // ---- Hidden diagnostics: the runtime-config expander on the Server
    // status card fetches GET /v1/config on first open and renders the real
    // ConfigResponse fields (capacity resolution, live memory, governor,
    // application paths, batching, streaming-prefill and checkpoint policy,
    // KV geometry, and exact budget partitions).
    await clickAndWait(page, '#runtime-config > summary', 'Could not open the runtime config expander');
    await waitForPanelText(page, '#runtime-config-body', /Application paths[\s\S]*Cache root[\s\S]*\/var\/cache\/kiln[\s\S]*config_file/, 'Runtime config should render the absolute application cache root and source');
    await waitForPanelText(page, '#runtime-config-body', /Change requires restart[\s\S]*required/, 'Runtime config should render cache-root immutability');
    await waitForPanelText(page, '#runtime-config-body', /linux-drm-sysfs-unified/, 'Runtime config should render the device-scoped memory source');
    await waitForPanelText(page, '#runtime-config-body', /Physical[\s\S]*25\.75 GiB/, 'Runtime config should render physical capacity');
    await waitForPanelText(page, '#runtime-config-body', /Effective cap[\s\S]*24\.00 GiB/, 'Runtime config should render the effective capacity cap');
    await waitForPanelText(page, '#runtime-config-body', /Usable[\s\S]*12\.50 GiB/, 'Runtime config should render live usable memory after the governor floor');
    await waitForPanelText(page, '#runtime-config-body', /Probe cadence[\s\S]*750 ms/, 'Runtime config should render the governor probe cadence');
    await waitForPanelText(page, '#runtime-config-body', /Reclaim[\s\S]*off[\s\S]*requested automatic/, 'Runtime config should distinguish effective reclaim from a profile-constrained request');
    await waitForPanelText(page, '#runtime-config-body', /Graph live[\s\S]*busy[\s\S]*model runner busy/, 'Runtime config should take live graph contention from the health poll');
    await waitForPanelText(page, '#runtime-config-body', /Graph current phase[\s\S]*native capture[\s\S]*\d+\.\d s/, 'Runtime config should show the runner-lock-independent in-progress graph phase from health');
    await waitForPanelText(page, '#runtime-config-body', /1,024/, 'Runtime config should render the KV cache block count');
    await waitForPanelText(page, '#runtime-config-body', /FP8 cache/, 'Runtime config should render the fp8 cache mode');
    await waitForPanelText(page, '#runtime-config-body', /Primary actor[\s\S]*active/, 'Runtime config should render primary batching actor activity');
    await waitForPanelText(page, '#runtime-config-body', /Prefix admission[\s\S]*on[\s\S]*environment/, 'Runtime config should render prefix-aware admission and source');
    await waitForPanelText(page, '#runtime-config-body', /Prefill configured[\s\S]*32[\s\S]*config_file/, 'Runtime config should render configured prefill admission quantum and source');
    await waitForPanelText(page, '#runtime-config-body', /Prefill effective[\s\S]*16[\s\S]*effective_decode_width/, 'Runtime config should render bounded effective prefill quantum and source');
    await waitForPanelText(page, '#runtime-config-body', /Cycle idle[\s\S]*75 ms[\s\S]*config_file[\s\S]*poll 5 ms/, 'Runtime config should render cooperative actor-cycle idle, source, and command polling bound');
    await waitForPanelText(page, '#runtime-config-body', /Burst prefill[\s\S]*on/, 'Runtime config should render backend burst-prefill policy');
    await expectRocmMatmulRuntimeConfig(page, 'Desktop');
    await expectStreamingPrefillRuntimeConfig(page, 'Desktop');
    await expectCheckpointBoundaryRuntimeConfig(page, 'Desktop');
    await expectCheckpointBoundaryRuntimeConfigLayout(page, 'Desktop');
    await waitForPanelText(page, '#runtime-config-body', /Runtime device[\s\S]*vulkan:0/, 'Runtime config should render the native-training runtime device');
    await waitForPanelText(page, '#runtime-config-body', /Weight device[\s\S]*cpu/, 'Runtime config should render the frozen model-weight device');
    await waitForPanelText(page, '#runtime-config-body', /Native training[\s\S]*unavailable/, 'Runtime config should render fail-closed native-training support');
    await waitForPanelText(page, '#runtime-config-body', /Optimizer contract[\s\S]*kiln\.training-optimizer-support[\s\S]*v1/, 'Runtime config should render the optimizer schema identity');
    await waitForPanelText(page, '#runtime-config-body', /Optimizer backend[\s\S]*vulkan · vulkan:0/, 'Runtime config should render the optimizer backend and device');
    await waitForPanelText(page, '#runtime-config-body', /Base \/ LoRA dtype[\s\S]*bf16 \/ f32/, 'Runtime config should render base and resolved LoRA dtypes');
    await waitForPanelText(page, '#runtime-config-body', /Optimizer tuples[\s\S]*Muon, AdamW, SGD/, 'Runtime config should preserve optimizer tuple facts when the workload path is unavailable');
    await waitForPanelText(page, '#runtime-config-body', /SFT workload[\s\S]*unavailable/, 'Runtime config should report SFT workload admission separately from optimizer tuples');
    await waitForPanelText(page, '#runtime-config-body', /GRPO workload[\s\S]*unavailable/, 'Runtime config should report GRPO workload admission separately from optimizer tuples');
    await waitForPanelText(page, '#runtime-config-body', /OPD workload[\s\S]*unavailable/, 'Runtime config should report OPD workload admission separately from optimizer tuples');
    await waitForPanelText(page, '#runtime-config-body', /Distill refresh workload[\s\S]*unavailable/, 'Runtime config should report joint refresh admission separately from its SFT and OPD phases');
    await waitForPanelText(page, '#runtime-config-body', /Optimizer implementations[\s\S]*Muon, AdamW, SGD/, 'Runtime config should render backend optimizer implementations');
    await waitForPanelText(page, '#runtime-config-body', /Native device hooks[\s\S]*Muon, AdamW, SGD/, 'Runtime config should distinguish native device hooks from server execution');
    await waitForPanelText(page, '#runtime-config-body', /Optimizer rounding[\s\S]*round_to_nearest[\s\S]*immutable/, 'Runtime config should render immutable round-to-nearest product policy');
    await waitForPanelText(page, '#runtime-config-body', /Muon rank[\s\S]*2\.\.32[\s\S]*admission required/, 'Runtime config should render the effective Muon rank constraint and separate live-memory rejection gate');
    await waitForPanelText(page, '#runtime-config-body', /Muon rank ceilings[\s\S]*backend 32 · model 256/, 'Runtime config should distinguish the backend ceiling from the resident-model ceiling');
    const unavailableOptimizerControls = await page.evaluate(() => ({
      sftSelectDisabled: document.getElementById('sft-optimizer')?.disabled,
      grpoSelectDisabled: document.getElementById('grpo-optimizer')?.disabled,
      openenvSubmitDisabled: document.querySelector('#openenv-form button[type="submit"]')?.disabled,
      opdSubmitDisabled: document.querySelector('#opd-form button[type="submit"]')?.disabled,
      refreshSubmitDisabled: document.querySelector('#distill-refresh-form button[type="submit"]')?.disabled,
      pumpSubmitDisabled: document.querySelector('#distill-pump-form button[type="submit"]')?.disabled,
      mergeSubmitDisabled: document.querySelector('#distill-merge-form button[type="submit"]')?.disabled,
      selfSubmitDisabled: document.querySelector('#distill-self-form button[type="submit"]')?.disabled,
      sftSubmitTitle: document.querySelector('#sft-form button[type="submit"]')?.title,
      openenvSubmitTitle: document.querySelector('#openenv-form button[type="submit"]')?.title,
      opdSubmitTitle: document.querySelector('#opd-form button[type="submit"]')?.title,
      refreshStatus: document.getElementById('refresh-optimizer-support')?.textContent,
    }));
    if (!unavailableOptimizerControls.sftSelectDisabled
      || !unavailableOptimizerControls.grpoSelectDisabled
      || !unavailableOptimizerControls.openenvSubmitDisabled
      || !unavailableOptimizerControls.opdSubmitDisabled
      || !unavailableOptimizerControls.refreshSubmitDisabled
      || !unavailableOptimizerControls.pumpSubmitDisabled
      || !unavailableOptimizerControls.mergeSubmitDisabled
      || !unavailableOptimizerControls.selfSubmitDisabled
      || !/CPU-host serving weights/.test(unavailableOptimizerControls.sftSubmitTitle || '')
      || !/CPU-host serving weights/.test(unavailableOptimizerControls.openenvSubmitTitle || '')
      || !/CPU-host serving weights/.test(unavailableOptimizerControls.opdSubmitTitle || '')
      || !/Distill refresh is unavailable/.test(unavailableOptimizerControls.refreshStatus || '')) {
      fail(`Unavailable optimizer support did not gate every training form: ${JSON.stringify(unavailableOptimizerControls)}`);
    }
    if (getTrainingSubmitRequests) {
      const before = getTrainingSubmitRequests();
      await page.$eval('#sft-form', form => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      await expectTrainingToast(page, 'SFT cannot submit: SFT is unavailable: native Vulkan training is unavailable for CPU-host serving weights.');
      await page.$eval('#grpo-form', form => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      await expectTrainingToast(page, 'GRPO cannot submit: GRPO is unavailable: native Vulkan training is unavailable for CPU-host serving weights.');
      await page.$eval('#openenv-environments', input => { input.value = 'http://127.0.0.1:8990'; });
      await page.$eval('#openenv-form', form => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      await expectTrainingToast(page, 'OpenEnv GRPO cannot submit: GRPO is unavailable: native Vulkan training is unavailable for CPU-host serving weights.');
      await page.$eval('#opd-form', form => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      await expectTrainingToast(page, 'OPD cannot submit: OPD is unavailable: native Vulkan training is unavailable for CPU-host serving weights.');
      for (const formId of ['distill-refresh-form', 'distill-pump-form', 'distill-merge-form', 'distill-self-form']) {
        await page.$eval(`#${formId}`, form => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      }
      await page.evaluate(() => {
        window.addCorrectionFromRequest({
          id: 'optimizer-capability-correction',
          prompt_full: 'Correct this response.',
          completion_full: 'Wrong response.',
          timestamp_unix_ms: Date.now(),
        });
        const ideal = document.querySelector('[data-corr-ideal="optimizer-capability-correction"]');
        ideal.value = 'Correct response.';
        ideal.dispatchEvent(new Event('input', { bubbles: true }));
        document.getElementById('corr-train').dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      });
      await goToPrimaryTab(page, 'distill');
      await clickAndWait(page, '#distill-tab-recipes', 'Could not open Recipes for fail-closed admission coverage');
      await page.waitForSelector('[data-recipe-run="smoke-supported-recipe"]', { timeout: 5000 })
        .catch(() => fail('Recipes did not render their admission descriptors'));
      await expectDisabled(page, '[data-recipe-run="smoke-supported-recipe"]', true, 'An unavailable recipe descriptor should disable Run');
      await expectDisabled(page, '[data-recipe-run="smoke-refresh-recipe"]', true, 'A refresh recipe without joint phase admission should disable Run');
      await expectDisabled(page, '[data-recipe-run="smoke-missing-admission"]', true, 'A missing recipe admission descriptor should fail closed');
      await page.$eval('[data-recipe-run="smoke-missing-admission"]', button => button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })));
      await page.$eval('[data-corr-remove="optimizer-capability-correction"]', button => button.click());
      await goToPrimaryTab(page, 'overview');
      const after = getTrainingSubmitRequests();
      if (JSON.stringify(after) !== JSON.stringify(before)) {
        fail(`Unsupported programmatic training submit reached an API: ${JSON.stringify(before)} -> ${JSON.stringify(after)}`);
      }
    }
    if (!setTrainingOptimizerAvailable) fail('Optimizer smoke requires a runtime capability transition setter');

    for (const [mode, reason] of [
      ['missing', /Optimizer capability details are unavailable/],
      ['unsupported-schema', /unsupported optimizer capability contract/],
      ['mutable-policy', /unsupported optimizer capability contract/],
      ['wrong-rounding', /unsupported optimizer capability contract/],
      ['invalid-rank-contract', /unsupported optimizer capability contract/],
      ['missing-tuple-identity', /unsupported optimizer capability contract/],
    ]) {
      setTrainingOptimizerAvailable(mode);
      const invalidConfigResponse = page.waitForResponse(
        response => response.url().endsWith('/v1/config') && response.status() === 200,
        { timeout: 5000 },
      );
      await clickAndWait(page, '#runtime-config [data-rc-refresh]', `Could not refresh the ${mode} optimizer schema fixture`);
      await invalidConfigResponse.catch(() => fail(`The ${mode} optimizer schema refresh did not fetch /v1/config`));
      await waitForPanelText(page, '#sft-optimizer-support', reason, `The ${mode} optimizer schema must fail closed with a visible reason`);
      await expectDisabled(page, '#sft-form button[type="submit"]', true, `The ${mode} optimizer schema must disable SFT submit`);
    }

    await page.evaluate(() => {
      const optimizer = document.getElementById('sft-optimizer');
      const rank = document.getElementById('sft-rank');
      optimizer.value = 'sgd';
      rank.value = '99';
    });
    setTrainingOptimizerAvailable(true);
    const refreshedConfig = page.waitForResponse(
      response => response.url().endsWith('/v1/config') && response.status() === 200,
      { timeout: 5000 },
    );
    await clickAndWait(page, '#runtime-config [data-rc-refresh]', 'Could not refresh runtime optimizer support');
    await refreshedConfig.catch(() => fail('Runtime optimizer refresh did not fetch /v1/config'));
    await waitForPanelText(page, '#runtime-config-body', /Optimizer backend[\s\S]*metal · metal:0/, 'Runtime config refresh should apply the new resident optimizer capability');
    await waitForPanelText(page, '#runtime-config-body', /Optimizer tuples[\s\S]*Muon, AdamW/, 'Runtime config refresh should show only Metal optimizer tuples');
    await waitForPanelText(page, '#runtime-config-body', /SFT workload[\s\S]*Muon, AdamW/, 'Runtime config refresh should show the Metal SFT workload allowlist');
    await waitForPanelText(page, '#runtime-config-body', /Distill refresh workload[\s\S]*unavailable/, 'Runtime config refresh must preserve the distinct fail-closed joint refresh workload');
    const supportedOptimizerControls = await page.evaluate(() => ({
      sftSelectDisabled: document.getElementById('sft-optimizer')?.disabled,
      grpoSelectDisabled: document.getElementById('grpo-optimizer')?.disabled,
      selectedKind: document.getElementById('sft-optimizer')?.value,
      sgdDisabled: document.querySelector('#sft-optimizer option[value="sgd"]')?.disabled,
      muonDisabled: document.querySelector('#sft-optimizer option[value="muon"]')?.disabled,
      opdSubmitDisabled: document.querySelector('#opd-form button[type="submit"]')?.disabled,
      rankValue: document.getElementById('sft-rank')?.value,
      rankMin: document.getElementById('sft-rank')?.min,
      rankMax: document.getElementById('sft-rank')?.max,
      sftStatus: document.getElementById('sft-optimizer-support')?.textContent,
    }));
    if (supportedOptimizerControls.sftSelectDisabled
      || supportedOptimizerControls.grpoSelectDisabled
      || supportedOptimizerControls.selectedKind !== 'sgd'
      || !supportedOptimizerControls.sgdDisabled
      || supportedOptimizerControls.muonDisabled
      || supportedOptimizerControls.opdSubmitDisabled
      || supportedOptimizerControls.rankValue !== '99'
      || !/SGD is not allowed for SFT/.test(supportedOptimizerControls.sftStatus || '')) {
      fail(`Supported Metal optimizer controls rewrote or hid an unsupported selection: ${JSON.stringify(supportedOptimizerControls)}`);
    }
    if (getTrainingSubmitRequests) {
      const before = getTrainingSubmitRequests();
      await page.$eval('#sft-form', form => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      const after = getTrainingSubmitRequests();
      if (JSON.stringify(after) !== JSON.stringify(before)) fail('A programmatic submit with selected unsupported SGD reached the API');
    }
    await page.select('#sft-optimizer', 'adam_w');
    const adamwRankBounds = await page.$eval('#sft-rank', input => ({
      min: input.min,
      max: input.max,
      value: input.value,
      describedBy: input.getAttribute('aria-describedby'),
      status: document.getElementById('sft-optimizer-support')?.textContent,
    }));
    if (adamwRankBounds.min !== '1'
      || adamwRankBounds.max !== '256'
      || adamwRankBounds.value !== '99'
      || adamwRankBounds.describedBy !== 'sft-optimizer-support'
      || !/supported 1–256/.test(adamwRankBounds.status || '')) {
      fail(`Effective AdamW model rank support should preserve the value and render 1–256 accessibly: ${JSON.stringify(adamwRankBounds)}`);
    }
    await page.select('#sft-optimizer', 'muon');
    const muonRankBounds = await page.$eval('#sft-rank', input => ({ min: input.min, max: input.max, value: input.value }));
    if (muonRankBounds.min !== '2' || muonRankBounds.max !== '32' || muonRankBounds.value !== '99') {
      fail(`Switching back to Muon should set constraints without clamping the entered rank: ${JSON.stringify(muonRankBounds)}`);
    }
    await waitForPanelText(page, '#sft-optimizer-support', /rank 99 is outside supported ranks 2–32/, 'Out-of-range Muon rank should remain visible with its rejection reason');
    await expectDisabled(page, '#sft-form button[type="submit"]', true, 'Out-of-range Muon rank should disable SFT submit');
    if (getTrainingSubmitRequests) {
      const before = getTrainingSubmitRequests();
      await page.$eval('#sft-form', form => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      const after = getTrainingSubmitRequests();
      if (JSON.stringify(after) !== JSON.stringify(before)) fail('A programmatic submit with out-of-range Muon rank reached the API');
    }
    await page.$eval('#sft-rank', input => {
      input.value = '8';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const fixedSurfaceState = await page.evaluate(() => ({
      correction: document.getElementById('corr-train')?.disabled,
      opd: document.querySelector('#opd-form button[type="submit"]')?.disabled,
      refresh: document.querySelector('#distill-refresh-form button[type="submit"]')?.disabled,
      pump: document.querySelector('#distill-pump-form button[type="submit"]')?.disabled,
      merge: document.querySelector('#distill-merge-form button[type="submit"]')?.disabled,
      self: document.querySelector('#distill-self-form button[type="submit"]')?.disabled,
      pumpRank: document.getElementById('pump-rank')?.value,
      pumpMin: document.getElementById('pump-rank')?.min,
      pumpMax: document.getElementById('pump-rank')?.max,
      refreshStatus: document.getElementById('refresh-optimizer-support')?.textContent,
      refreshTitle: document.querySelector('#distill-refresh-form button[type="submit"]')?.title,
    }));
    if (fixedSurfaceState.opd
      || !fixedSurfaceState.refresh
      || fixedSurfaceState.pump
      || fixedSurfaceState.merge
      || fixedSurfaceState.self
      || fixedSurfaceState.pumpRank !== '32'
      || fixedSurfaceState.pumpMin !== '2'
      || fixedSurfaceState.pumpMax !== '32'
      || !/pins separate exact SFT and OPD phase plans, prepares the exact SFT rows/.test(fixedSurfaceState.refreshStatus || '')
      || !/pins separate exact SFT and OPD phase plans, prepares the exact SFT rows/.test(fixedSurfaceState.refreshTitle || '')) {
      fail(`Fixed training surfaces did not reflect their exact workload and Muon tuple admissions: ${JSON.stringify(fixedSurfaceState)}`);
    }
    // Corrections remains disabled because its basket is empty, independently
    // of the now-supported optimizer tuple.
    if (!fixedSurfaceState.correction) fail('An empty corrections basket should remain disabled after optimizer admission succeeds');
    if (getTrainingSubmitRequests) {
      const before = getTrainingSubmitRequests();
      await page.$eval('#distill-refresh-form', form => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
      const after = getTrainingSubmitRequests();
      if (after.refresh !== before.refresh) {
        fail(`A programmatic refresh submit bypassed distinct joint-workload admission: ${JSON.stringify(before)} -> ${JSON.stringify(after)}`);
      }
    }

    await goToPrimaryTab(page, 'distill');
    await clickAndWait(page, '#distill-tab-recipes', 'Could not refresh recipe descriptors after optimizer support became available');
    await page.waitForFunction(
      () => document.querySelector('[data-recipe-run="smoke-supported-recipe"]')?.dataset.recipeAdmissionSupported === 'true',
      { timeout: 5000 },
    ).catch(() => fail('Supported recipe descriptor did not refresh'));
    await expectDisabled(page, '[data-recipe-run="smoke-supported-recipe"]', false, 'A supported recipe descriptor should enable Run');
    await expectDisabled(page, '[data-recipe-run="smoke-refresh-recipe"]', true, 'A refresh recipe must remain disabled without a supported joint phase descriptor');
    const refreshRecipeAdmission = await page.$eval('[data-recipe-run="smoke-refresh-recipe"]', button => ({
      title: button.title,
      status: document.getElementById(button.getAttribute('aria-describedby'))?.textContent,
    }));
    if (!/pins separate exact SFT and OPD phase plans, prepares the exact SFT rows/.test(refreshRecipeAdmission.title || '')
      || !/Recipe execution remains disabled/.test(refreshRecipeAdmission.status || '')) {
      fail(`Refresh recipe admission reason was not exposed accessibly: ${JSON.stringify(refreshRecipeAdmission)}`);
    }
    await expectDisabled(page, '[data-recipe-run="smoke-missing-admission"]', true, 'A missing recipe admission descriptor must stay disabled');
    if (getTrainingSubmitRequests) {
      const before = getTrainingSubmitRequests();
      await clickAndWait(page, '[data-recipe-run="smoke-supported-recipe"]', 'Could not run a recipe with an admitted descriptor');
      await expectTrainingToast(page, 'Recipe queued');
      const after = getTrainingSubmitRequests();
      if (after.recipe !== before.recipe + 1) {
        fail(`An admitted recipe should issue exactly one run request: ${JSON.stringify(before)} -> ${JSON.stringify(after)}`);
      }
    }
    await goToPrimaryTab(page, 'overview');

    if (setFailRuntimeConfig) {
      await clickAndWait(page, '#runtime-config > summary', 'Could not close Runtime Config before the cross-surface failure check');
      const configClosed = await page.$eval('#runtime-config', details => !details.open);
      if (!configClosed) fail('Runtime Config should be closed before the Playground-triggered refresh failure');
      await goToPrimaryTab(page, 'playground');
      expectedRuntimeConfigFailure = true;
      setFailRuntimeConfig(true);
      const failedConfigResponse = page.waitForResponse(
        response => response.url().endsWith('/v1/config') && response.status() === 503,
        { timeout: 5000 },
      );
      await page.$eval('#chat-thinking-budget-refresh', button => button.click());
      await failedConfigResponse.catch(() => fail('Playground-triggered optimizer capability refresh did not receive the expected failure'));
      await page.waitForFunction(
        () => document.getElementById('chat-thinking-budget-preview')?.dataset.state === 'defaults unavailable',
        { timeout: 5000 },
      ).catch(() => fail('Playground must expose its config refresh failure and reveal Retry'));
      await waitForPanelText(page, '#sft-optimizer-support', /Optimizer capability lookup failed/, 'A refresh failure must clear the previously valid optimizer snapshot');
      await expectDisabled(page, '#sft-form button[type="submit"]', true, 'A refresh failure must fail closed instead of retaining stale optimizer support');
      await expectDisabled(page, '[data-recipe-run="smoke-supported-recipe"]', true, 'A refresh failure must disable a recipe even when its last descriptor was supported');
      setFailRuntimeConfig(false);
      await goToPrimaryTab(page, 'overview');
      const recoveredConfigResponse = page.waitForResponse(
        response => response.url().endsWith('/v1/config') && response.status() === 200,
        { timeout: 5000 },
      );
      await clickAndWait(page, '#runtime-config > summary', 'Could not reopen Runtime Config after the Playground-triggered failure');
      await recoveredConfigResponse.catch(() => fail('Reopening Runtime Config reused stale loaded HTML instead of refetching after the shared failure'));
      await waitForPanelText(page, '#runtime-config-body', /Optimizer backend[\s\S]*metal · metal:0/, 'Runtime Config did not repaint after recovering from the Playground-triggered failure');
      await waitForPanelText(page, '#sft-optimizer-support', /Muon · bf16 LoRA/, 'Recovered optimizer capability should re-enable the selected supported tuple');
      await expectDisabled(page, '[data-recipe-run="smoke-supported-recipe"]', false, 'A recovered capability should reapply the still-current supported recipe descriptor');
      expectedRuntimeConfigFailure = false;
    }
    await waitForPanelText(page, '#runtime-config-body', /Checkpoint policy[\s\S]*explicit segments/, 'Runtime config should render the typed checkpoint policy');
    await waitForPanelText(page, '#runtime-config-body', /Training budget/, 'Runtime config should render the memory budget rows');
    await waitForPanelText(page, '#runtime-config-body', /8,858,370,048 B/, 'Runtime config should render exact partition bytes alongside GiB');
    await clickAndWait(page, '#runtime-config [data-rc-raw]', 'Could not toggle the runtime config raw JSON');
    const configRawShown = await page.$eval(
      '#runtime-config [data-rc-raw-pre]',
      (el) => !el.hidden
        && /"memory_budget"/.test(el.textContent || '')
        && /"cache_root": "\/var\/cache\/kiln"/.test(el.textContent || '')
        && /"actor_active": true/.test(el.textContent || '')
        && /"streaming_prefill"/.test(el.textContent || ''),
    );
    if (!configRawShown) fail('Runtime config raw JSON toggle should reveal the pretty-printed /v1/config payload including resolved paths, batching, and streaming-prefill state');

    // Regression (the "pie chart disappears" bug): the VRAM donut and header
    // stats must survive subsequent 2s health polls. The donut used to render
    // once, then renderServerStatus clobbered the panel while the donut
    // refresher's dedupe key skipped re-appending it — gone until reload.
    // The mock /health cycles scheduler.blocks_used, so every poll below is
    // a REAL innerHTML repaint of #server-status, not a deduped no-op.
    await page.waitForSelector('#server-status .vram-donut svg', { timeout: 5000 })
      .catch(() => fail('VRAM donut should render in the server status panel'));
    await new Promise((resolve) => setTimeout(resolve, 5200)); // sit through ≥2 health polls
    const overviewSteadyState = await page.evaluate(() => ({
      donuts: document.querySelectorAll('#server-status .vram-donut').length,
      svg: Boolean(document.querySelector('#server-status .vram-donut svg')),
      model: document.getElementById('header-model')?.textContent || '',
      uptime: document.getElementById('header-uptime')?.textContent || '',
      configOpen: document.getElementById('runtime-config')?.open === true,
      configIntact: /linux-drm-sysfs-unified/.test(document.getElementById('runtime-config-body')?.textContent || ''),
      serverStatusBusy: document.getElementById('server-status')?.getAttribute('aria-busy'),
      busyFlips: window.__serverStatusBusyFlips || null,
    }));
    if (!overviewSteadyState.svg) fail('VRAM donut disappeared after subsequent health polls');
    if (overviewSteadyState.donuts !== 1) fail(`Expected exactly one VRAM donut, found ${overviewSteadyState.donuts}`);
    if (overviewSteadyState.model !== 'Qwen3.5-4B') fail(`Header model stat should render from /health, got "${overviewSteadyState.model}"`);
    if (!/^\d+[sm]/.test(overviewSteadyState.uptime)) fail(`Header uptime stat should render, got "${overviewSteadyState.uptime}"`);
    // The expander is a static SIBLING of the keyed #server-status region —
    // the ≥2 genuine repaints above must not close it or destroy its content.
    if (!overviewSteadyState.configOpen) fail('Runtime config expander lost its open state across server-status repaints');
    if (!overviewSteadyState.configIntact) fail('Runtime config content was destroyed by the server-status repaint');
    // aria-busy hygiene: the ≥2 genuine repaints above (blocks_used cycles, so
    // content really changed) must not flip aria-busy — it toggles only on the
    // FIRST load, then stays false for the life of the page.
    if (overviewSteadyState.serverStatusBusy !== 'false') {
      fail(`#server-status aria-busy should stay "false" across poll ticks, got "${overviewSteadyState.serverStatusBusy}"`);
    }
    if (overviewSteadyState.busyFlips === null) fail('aria-busy mutation observer was not installed on #server-status');
    if (overviewSteadyState.busyFlips.length > 0) {
      fail(`aria-busy on #server-status flipped on poll ticks after the first render: ${overviewSteadyState.busyFlips.join(', ')}`);
    }

    await goToPrimaryTab(page, 'playground');
    await waitForPanelText(page, '#chat-output', /Send a message to test inference\./, 'Quick Inference empty state missing');
    await expectPanelLink(page, '#chat-output .empty', '/health', '/health');
    await expectPanelLink(page, '#chat-output .empty', 'Troubleshooting guide', 'https://ericflo.github.io/kiln/troubleshooting.html');

    await goToPrimaryTab(page, 'training');
    // Empty queue lands on the Train·SFT form by design (see mobile flow note).
    await waitForVisiblePanel(page, '#tab-sft', 'Empty training queue should land on the Train·SFT form');
    await clickAndWait(page, '#training-tab-queue', 'Could not activate Queue tab before keyboard checks');
    await expectTrainingTabKeyboardNavigation(page);

    await clickAndWait(page, '#training-tab-openenv', 'Could not open OpenEnv training tab');
    await waitForVisiblePanel(page, '#tab-openenv', 'OpenEnv training tab did not activate');
    await waitForPanelText(page, '#openenv-optimizer-support', /Muon · bf16 LoRA · round-to-nearest · rank 8 \(supported 2–32\)/, 'OpenEnv train should render the native GRPO optimizer tuple');
    await expectDisabled(page, '#openenv-form button[type="submit"]', false, 'OpenEnv train should enable after native GRPO capability admission');
    await page.$eval('#openenv-environments', input => { input.value = 'http://127.0.0.1:8990'; });
    await page.$eval('#openenv-credential-ids', input => { input.value = 'smoke-arcade'; });
    const inspectRequest = page.waitForRequest(
      request => request.method() === 'POST' && request.url().endsWith('/v1/openenv/inspect'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#openenv-inspect', 'Could not inspect the OpenEnv server');
    const openEnvInspectRequest = await inspectRequest.catch(() => fail('OpenEnv Inspect did not POST /v1/openenv/inspect'));
    const openEnvInspectBody = JSON.parse(openEnvInspectRequest.postData() || '{}');
    if (openEnvInspectBody.credential_ids?.[0] !== 'smoke-arcade') {
      fail(`OpenEnv Inspect lost its origin-scoped credential handle: ${JSON.stringify(openEnvInspectBody)}`);
    }
    await waitForPanelText(page, '#openenv-inspection', /smoke-bandit[\s\S]*schema sha256:aaaa/, 'OpenEnv inspection should render discovered identity and action schema');
    await page.$eval('#openenv-task-split', input => { input.value = 'train'; });
    await page.$eval('#openenv-task-start', input => { input.value = '1'; });
    await page.$eval('#openenv-task-limit', input => { input.value = '1'; });
    const taskCatalogRequest = page.waitForRequest(
      request => request.method() === 'POST' && request.url().endsWith('/v1/openenv/tasks'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#openenv-task-inspect', 'Could not inspect the OpenEnv task catalog');
    const openEnvTaskRequest = await taskCatalogRequest.catch(() => fail('OpenEnv task catalog did not POST /v1/openenv/tasks'));
    const openEnvTaskBody = JSON.parse(openEnvTaskRequest.postData() || '{}');
    if (openEnvTaskBody.credential_ids?.[0] !== 'smoke-arcade'
      || openEnvTaskBody.split !== 'train'
      || openEnvTaskBody.start !== 1
      || openEnvTaskBody.limit !== 1) {
      fail(`OpenEnv task catalog lost its bounded selection or credential handle: ${JSON.stringify(openEnvTaskBody)}`);
    }
    await waitForPanelText(page, '#openenv-task-catalog', /smoke-bandit[\s\S]*train[\s\S]*tasks 1\.\.2 of 3[\s\S]*2 \+ 2/, 'OpenEnv task catalog should render the bounded arbitrary task page');
    await page.$eval('#openenv-environments', input => { input.value = 'http://127.0.0.1:8990\nhttp://127.0.0.1:8991'; });
    await page.$eval('#openenv-credential-ids', input => { input.value = 'smoke-arcade\n-'; });
    await page.$eval('#openenv-environment-reset-options', input => { input.value = '[{"difficulty":"hard"},{"split":"train"}]'; });
    await page.select('#openenv-environment-eval-mode', 'gate');
    await page.$eval('#openenv-environment-gate-floor', input => { input.value = '0.5'; });
    await page.$eval('#openenv-environment-gate-improvement', input => { input.value = '0.1'; });
    const openEnvSubmitRequest = page.waitForRequest(
      request => request.method() === 'POST' && request.url().endsWith('/v1/openenv/runs'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#openenv-form button[type="submit"]', 'Could not submit the OpenEnv train run');
    const openEnvRequest = await openEnvSubmitRequest.catch(() => fail('OpenEnv form did not POST /v1/openenv/runs'));
    const openEnvBody = JSON.parse(openEnvRequest.postData() || '{}');
    if (openEnvBody.kind !== 'train'
      || openEnvBody.environment_urls?.[0] !== 'http://127.0.0.1:8990'
      || openEnvBody.environment_urls?.[1] !== 'http://127.0.0.1:8991'
      || openEnvBody.credential_ids?.[0] !== 'smoke-arcade'
      || openEnvBody.credential_ids?.[1] !== null
      || openEnvBody.environment_reset_options?.[0]?.difficulty !== 'hard'
      || openEnvBody.environment_reset_options?.[1]?.split !== 'train'
      || openEnvBody.adapter !== 'base'
      || openEnvBody.output_adapter !== 'openenv-agent'
      || openEnvBody.training_config?.lora_rank !== 8
      || openEnvBody.environment_eval?.groups !== 20
      || openEnvBody.environment_eval?.group_size !== 1
      || openEnvBody.environment_eval?.gate?.min_mean_return !== 0.5
      || openEnvBody.environment_eval?.gate?.min_mean_improvement !== 0.1) {
      fail(`OpenEnv train request lost its protocol or native-GRPO controls: ${JSON.stringify(openEnvBody)}`);
    }
    await expectTrainingToast(page, 'OpenEnv train run smoke-op started');
    await waitForPanelText(page, '#openenv-runs', /OpenEnv train[\s\S]*smoke-op[\s\S]*completed[\s\S]*promoted[\s\S]*8 \/ 32 episodes[\s\S]*Held-out environment · completed · return 0\.100 → 0\.800 \(\+0\.700\) · exact p=0\.0000/, 'OpenEnv run history should render paired-return evidence and promotion outcome');

    await clickAndWait(page, '#training-tab-sft', 'Could not open SFT tab');
    await waitForVisiblePanel(page, '#tab-sft', 'SFT tab did not activate');
    await waitForPanelText(page, '#sft-optimizer-support', /Muon · bf16 LoRA · round-to-nearest · rank 8 \(supported 2–32\)/, 'SFT should render the selected executable optimizer tuple');
    const sftOptimizerCapability = await page.evaluate(() => ({
      selectDisabled: document.getElementById('sft-optimizer')?.disabled,
      disabledKinds: Array.from(document.querySelectorAll('#sft-optimizer option:disabled')).map(option => option.value),
      rankMin: document.getElementById('sft-rank')?.min,
      rankMax: document.getElementById('sft-rank')?.max,
    }));
    if (sftOptimizerCapability.selectDisabled
      || sftOptimizerCapability.disabledKinds.join(',') !== 'sgd'
      || sftOptimizerCapability.rankMin !== '2'
      || sftOptimizerCapability.rankMax !== '32') {
      fail(`SFT optimizer controls did not preserve Metal capability constraints: ${JSON.stringify(sftOptimizerCapability)}`);
    }
    await expectDisabled(page, '#sft-form button[type="submit"]', true, 'SFT submit should start disabled until examples are provided');
    await clickAndWait(page, '#use-sft-sample', 'Could not click SFT sample payload button');
    await expectDisabled(page, '#sft-form button[type="submit"]', false, 'SFT submit should enable after sample payload is clicked');
    await clickAndWait(page, '#sft-adv-toggle', 'Could not open SFT advanced settings');
    await page.$eval('#sft-checkpoint-interval', (input) => {
      input.value = '2';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expectAdvancedTrainingLayout(page, 'sft', 'Desktop');
    await waitForPanelText(page, '#sft-adv-summary', /checkpoint every 2 · fresh run/, 'SFT advanced summary should expose checkpoint cadence');
    await clickAndWait(page, '#sft-form button[type="submit"]', 'Could not submit sample SFT payload');
    await expectDisabled(page, '#sft-form button[type="submit"]', true, 'SFT submit should disable while the job is submitting');
    await expectTrainingToast(page, 'SFT job submitted · seed 18446744073709551615');
    await expectActiveTrainingTab(page, 'queue', 'Submitting SFT should switch back to the training queue tab');
    await waitForPanelText(page, '#tab-queue', /smoke-sf/, 'Training queue should refresh after SFT submit');
    await waitForPanelText(page, '#tab-queue', /Adapter:\s*sft-adapter/, 'Training queue should show the submitted SFT adapter name');
    await waitForPanelText(page, '#tab-queue', /seed 18446744073709551615/, 'Training queue should preserve and show the exact effective seed');
    await waitForPanelText(page, '#tab-queue', /running/, 'Training queue should show the SFT job as running');
    // The queue panel is no longer aria-live; the card's visually-hidden
    // status node must carry the start transition instead.
    await expectStatusAnnouncement(page, 'training-queue-status', /Training started: sft-adapter\./, 'Training start was not announced');

    // Drill modal for the RUNNING job: Stop must be live (running jobs are
    // cancellable cooperatively — the trainer aborts at the next step
    // boundary) and must route through the same DELETE /v1/train/queue/:id
    // path the queue card uses. The modal stays open across the cancel so
    // failures (and the cancelled repaint) surface in it.
    await clickAndWait(page, '[data-train-job-id="smoke-sft"]', 'Could not open the train drill modal for the running SFT job');
    // Deep-link grammar: opening the train drill pushes the job id segment.
    await expectActivePageAndHash(page, 'training', 'Opening the train drill should push the job id into the hash', '#training/queue/smoke-sft');
    await page.waitForFunction(
      () => {
        const modal = document.getElementById('train-drill-modal');
        const stop = document.getElementById('train-drill-stop');
        return modal && !modal.hidden && stop && !stop.hidden && !stop.disabled
          && stop.title === 'Stop at the next training step'
          && stop.dataset.jobId === 'smoke-sft';
      },
      { timeout: 5000 },
    ).catch(() => fail('Train drill Stop should be enabled for a running job with title "Stop at the next training step"'));
    await waitForPanelText(page, '#train-drill-content', /18446744073709551615/, 'Training drill should show the exact effective seed');
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '#train-drill-content [data-copy-training-seed]', 'Could not copy the effective training seed');
    await page.waitForFunction(
      () => window.__copiedText === '18446744073709551615',
      { timeout: 5000 },
    ).catch(async () => {
      const copiedText = await page.evaluate(() => window.__copiedText).catch(() => undefined);
      fail(`Copy effective seed should preserve all u64 digits, got ${JSON.stringify(copiedText)}`);
    });
    await expectTrainingToast(page, 'Effective seed copied');
    // Copy loss CSV: the running SFT job carries three TrainingLossSample
    // rows (epoch/progress/loss/elapsed_secs — no step, no timestamps), so
    // the header button must be live and put exactly that CSV on the
    // clipboard via the shared __copiedText test hook.
    await expectDisabled(page, '#train-drill-copy-loss', false, 'Copy loss CSV should enable once the job has loss samples');
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '#train-drill-copy-loss', 'Could not click Copy loss CSV');
    const expectedLossCsv = 'sample,epoch,progress,loss,elapsed_secs\n1,1,0.1,2.5,2\n2,1,0.25,1.9,5\n3,1,0.42,1.234,12';
    await page.waitForFunction(
      (expected) => window.__copiedText === expected,
      { timeout: 5000 },
      expectedLossCsv,
    ).catch(async () => {
      const copiedText = await page.evaluate(() => window.__copiedText).catch(() => undefined);
      fail(`Copy loss CSV should copy the loss history rows, got ${JSON.stringify(copiedText)}`);
    });
    await expectTrainingToast(page, 'Loss history copied as CSV');
    page.once('dialog', async (dialog) => {
      if (!/Stop this running job at the next training step\?/.test(dialog.message())) fail(`Unexpected stop confirmation text: ${dialog.message()}`);
      await dialog.accept();
    });
    const stopRequestPromise = page.waitForRequest(
      (request) => request.method() === 'DELETE' && request.url().endsWith('/v1/train/queue/smoke-sft'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#train-drill-stop', 'Could not click the train drill Stop button');
    await stopRequestPromise.catch(() => fail('Drill-modal Stop did not send DELETE /v1/train/queue/smoke-sft'));
    await expectTrainingToast(page, 'Cancelled job smoke-sf');
    const drillStillOpen = await page.$eval('#train-drill-modal', (el) => !el.hidden);
    if (!drillStillOpen) fail('Train drill modal should stay open across Stop so failures and the cancelled state surface in it');
    // The 1.5s drill poll repaints the now-terminal job: Stop hides, Delete shows.
    await page.waitForFunction(
      () => {
        const stop = document.getElementById('train-drill-stop');
        const del = document.getElementById('train-drill-delete');
        return stop && stop.hidden && del && !del.hidden;
      },
      { timeout: 6000 },
    ).catch(() => fail('Train drill modal did not repaint to the cancelled state after Stop'));
    // The running→failed transition (cancel lands as Failed) must also speak
    // through the status node — same channel as completed/failed jobs.
    await expectStatusAnnouncement(page, 'training-queue-status', /Training failed: sft-adapter\./, 'Training cancel (failed) was not announced');
    await clickAndWait(page, '#train-drill-close', 'Could not close the train drill modal');
    // X-close consumes the entry the open minted: modal hidden AND the hash
    // back to the queue sub-tab (the close routes through history.back()).
    await page.waitForFunction(
      () => document.getElementById('train-drill-modal')?.hidden === true && window.location.hash === '#training/queue',
      { timeout: 5000 },
    ).catch(async () => {
      const got = await page.evaluate(() => window.location.hash).catch(() => 'unknown');
      fail(`Train drill close should pop the hash back to #training/queue, got ${got}`);
    });

    await clickAndWait(page, '#training-tab-grpo', 'Could not open GRPO tab');
    await waitForVisiblePanel(page, '#tab-grpo', 'GRPO tab did not activate');
    await waitForPanelText(page, '#grpo-optimizer-support', /Muon · bf16 LoRA · round-to-nearest · rank 8 \(supported 2–32\)/, 'GRPO should render the selected executable optimizer tuple');
    await waitForPanelText(page, '#opd-optimizer-support', /OPD uses Muon\.[\s\S]*Muon · bf16 LoRA · round-to-nearest · rank 32 \(supported 2–32\)/, 'OPD should surface its fixed executable Muon tuple');
    await expectDisabled(page, '#grpo-form button[type="submit"]', true, 'GRPO submit should start disabled until groups are provided');
    await clickAndWait(page, '#use-grpo-sample', 'Could not click GRPO sample payload button');
    await expectDisabled(page, '#grpo-form button[type="submit"]', false, 'GRPO submit should enable after sample payload is clicked');
    await clickAndWait(page, '#grpo-adv-toggle', 'Could not open GRPO advanced settings');
    await page.$eval('#grpo-checkpoint-interval', (input) => {
      input.value = '3';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await expectAdvancedTrainingLayout(page, 'grpo', 'Desktop');
    await waitForPanelText(page, '#grpo-adv-summary', /checkpoint every 3 · fresh run/, 'GRPO advanced summary should expose checkpoint cadence');
    await clickAndWait(page, '#grpo-form button[type="submit"]', 'Could not submit sample GRPO payload');
    await expectDisabled(page, '#grpo-form button[type="submit"]', true, 'GRPO submit should disable while the job is submitting');
    await expectTrainingToast(page, 'GRPO job submitted · seed 18446744073709551614');
    await expectActiveTrainingTab(page, 'queue', 'Submitting GRPO should switch back to the training queue tab');
    await waitForPanelText(page, '#tab-queue', /smoke-gr/, 'Training queue should refresh after GRPO submit');
    await waitForPanelText(page, '#tab-queue', /Adapter:\s*grpo-adapter/, 'Training queue should show the submitted GRPO adapter name');

    // The completed GRPO job recorded no loss samples — Copy loss CSV must
    // disable with a title that explains what unlocks it.
    await clickAndWait(page, '[data-train-job-id="smoke-grpo"]', 'Could not open the train drill modal for the completed GRPO job');
    await page.waitForFunction(() => document.getElementById('train-drill-modal')?.hidden === false, { timeout: 5000 })
      .catch(() => fail('Train drill modal did not open for the GRPO job'));
    await expectDisabled(page, '#train-drill-copy-loss', true, 'Copy loss CSV should disable when the job has no loss samples');
    const copyLossTitle = await page.$eval('#train-drill-copy-loss', (el) => el.title);
    if (!/No loss samples recorded yet/.test(copyLossTitle)) fail(`Disabled Copy loss CSV should explain what unlocks it, got: ${JSON.stringify(copyLossTitle)}`);
    await waitForPanelText(page, '#train-drill-content', /GRPO JSONL · next group cursor 3/, 'GRPO checkpoint status should use route-aware group cursor wording');
    await waitForPanelText(page, '#train-drill-content', /Base weights[\s\S]*1 shard/, 'Training drill should show the persisted base-weight identity');
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '#train-drill-content [data-copy-base-weight]', 'Could not copy the training base-weight identity');
    await page.waitForFunction(
      () => window.__copiedText === 'sha256:c62f9f56234c61c943716ae3b8783c851fb41a2551e31f17d15f1b0c346339b5',
      { timeout: 5000 },
    ).catch(() => fail('Training base-weight copy should preserve the complete aggregate digest'));
    await expectTrainingToast(page, 'Base-weight identity copied');
    await waitForPanelText(page, '#train-drill-content', /Execution[\s\S]*rocm · gfx1151/, 'Training drill should show the persisted execution identity');
    await waitForPanelText(page, '#train-drill-content', /Concrete precision[\s\S]*bf16 parameters[\s\S]*f32 optimizer[\s\S]*round_to_nearest/, 'Training drill should show the concrete precision contract');
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '#train-drill-content [data-copy-execution]', 'Could not copy the training execution identity');
    await page.waitForFunction(
      () => window.__copiedText === `sha256:${'b'.repeat(64)}`,
      { timeout: 5000 },
    ).catch(() => fail('Training execution copy should preserve the complete canonical digest'));
    await expectTrainingToast(page, 'Execution identity copied');
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '[data-copy-resume-checkpoint]', 'Could not copy the GRPO resume checkpoint');
    await page.waitForFunction(
      () => window.__copiedText === 'grpo-adapter-checkpoint-step-00000003.kiln-checkpoint',
      { timeout: 5000 },
    ).catch(() => fail('GRPO resume checkpoint copy should use the direct immutable basename'));
    await expectTrainingToast(page, 'Resume checkpoint copied');

    await clickAndWait(page, '[data-prepare-training-resume]', 'Could not prepare the GRPO resume form');
    await waitForVisiblePanel(page, '#tab-grpo', 'Preparing a GRPO resume should open the GRPO form');
    await page.waitForFunction(() => document.getElementById('train-drill-modal')?.hidden === true, { timeout: 5000 })
      .catch(() => fail('Preparing a resume should close the train drill modal'));
    await expectTrainingToast(page, 'Checkpoint loaded — re-select the exact original training data before submitting.');
    const preparedResume = await page.evaluate(() => ({
      adapter: document.getElementById('grpo-output-name')?.value,
      cadence: document.getElementById('grpo-checkpoint-interval')?.value,
      checkpoint: document.getElementById('grpo-resume-checkpoint')?.value,
      advancedOpen: document.getElementById('grpo-advanced')?.hidden === false,
    }));
    if (preparedResume.adapter !== 'grpo-adapter'
      || preparedResume.cadence !== '3'
      || preparedResume.checkpoint !== 'grpo-adapter-checkpoint-step-00000003.kiln-checkpoint'
      || !preparedResume.advancedOpen) {
      fail(`Prepared GRPO resume fields were incomplete: ${JSON.stringify(preparedResume)}`);
    }
    await waitForPanelText(page, '#grpo-adv-summary', /checkpoint every 3 · resume selected/, 'Prepared GRPO resume should be visible in the collapsed summary');
    await expectDisabled(page, '#grpo-form button[type="submit"]', true, 'Preparing an inline resume must clear unverifiable prior data');

    await page.$eval('#grpo-checkpoint-interval', (input) => {
      input.value = '0';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await clickAndWait(page, '#use-grpo-sample', 'Could not restore sample data for resume validation');
    await expectDisabled(page, '#grpo-form button[type="submit"]', false, 'GRPO submit should re-enable after exact data is selected');
    const zeroCadenceValidity = await page.$eval('#grpo-checkpoint-interval', (input) => ({
      valid: input.checkValidity(),
      rangeUnderflow: input.validity.rangeUnderflow,
      formValid: input.form?.checkValidity(),
    }));
    if (zeroCadenceValidity.valid || !zeroCadenceValidity.rangeUnderflow || zeroCadenceValidity.formValid) {
      fail(`Native GRPO checkpoint cadence validation should reject zero: ${JSON.stringify(zeroCadenceValidity)}`);
    }
    await page.$eval('#grpo-checkpoint-interval', (input) => {
      input.value = '3';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.$eval('#grpo-resume-checkpoint', (input) => {
      input.value = '../bad.kiln-checkpoint';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await clickAndWait(page, '#grpo-form button[type="submit"]', 'Could not exercise invalid GRPO resume validation');
    await expectTrainingToast(page, 'GRPO resume checkpoint must be one direct .kiln-checkpoint basename, without a path.');

    await goToPrimaryTab(page, 'distill');
    await clickAndWait(page, '#distill-tab-opd', 'Could not open the OPD distillation form');
    await waitForVisiblePanel(page, '#distill-tab-opd-pane', 'OPD distillation pane did not activate');
    await page.waitForFunction(
      () => Array.from(document.getElementById('opd-teacher')?.options || []).some((option) => option.value === 'teacher-v1'),
      { timeout: 5000 },
    ).catch(() => fail('OPD teacher dropdown did not load the usable registered teacher'));
    await page.select('#opd-teacher', 'teacher-v1');
    await clickAndWait(page, '#opd-use-sample', 'Could not insert OPD sample prompts');
    const freshOpdState = await page.evaluate(() => ({
      cadence: document.getElementById('opd-checkpoint-interval')?.value,
      checkpoint: document.getElementById('opd-resume-checkpoint')?.value,
      promptCount: JSON.parse(document.getElementById('opd-prompts')?.value || '[]').length,
    }));
    if (freshOpdState.cadence !== '25' || freshOpdState.checkpoint !== '' || freshOpdState.promptCount !== 2) {
      fail(`Fresh OPD exact-checkpoint defaults are wrong: ${JSON.stringify(freshOpdState)}`);
    }
    const opdSubmitRequest = page.waitForRequest(
      (request) => request.method() === 'POST' && request.url().endsWith('/v1/train/opd'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#opd-form button[type="submit"]', 'Could not submit the OPD sample payload');
    await opdSubmitRequest.catch(() => fail('OPD form did not POST /v1/train/opd'));
    await expectTrainingToast(page, 'OPD job submitted · seed 18446744073709551613');
    await expectActivePageAndHash(page, 'training', 'Submitting OPD should open the training queue', '#training/queue');
    await waitForPanelText(page, '#tab-queue', /smoke-op/, 'Training queue should refresh after OPD submit');
    await waitForPanelText(page, '#tab-queue', /Adapter:\s*opd-adapter/, 'Training queue should show the submitted OPD adapter name');

    await clickAndWait(page, '[data-train-job-id="smoke-opd"]', 'Could not open the completed OPD job detail');
    await waitForPanelText(page, '#train-drill-content', /OPD inline-opd-prompts-v1 · next candidate cursor 2/, 'OPD checkpoint status should use candidate-cursor wording');
    await clickAndWait(page, '[data-prepare-training-resume]', 'Could not prepare the OPD resume form');
    await waitForVisiblePanel(page, '#distill-tab-opd-pane', 'Preparing OPD resume should open the Distill form');
    await expectActivePageAndHash(page, 'distill', 'Preparing OPD resume should deep-link to the Distill form', '#distill/opd');
    await expectTrainingToast(page, 'OPD checkpoint loaded — reinsert the exact original prompts before submitting.');
    const preparedOpd = await page.evaluate(() => ({
      adapter: document.getElementById('opd-output-name')?.value,
      teacher: document.getElementById('opd-teacher')?.value,
      rank: document.getElementById('opd-rank')?.value,
      cadence: document.getElementById('opd-checkpoint-interval')?.value,
      checkpoint: document.getElementById('opd-resume-checkpoint')?.value,
      prompts: document.getElementById('opd-prompts')?.value,
      noteHidden: document.getElementById('opd-resume-note')?.hidden,
      note: document.getElementById('opd-resume-note')?.textContent || '',
    }));
    if (preparedOpd.adapter !== 'opd-adapter'
      || preparedOpd.teacher !== 'teacher-v1'
      || preparedOpd.rank !== '32'
      || preparedOpd.cadence !== '25'
      || preparedOpd.checkpoint !== 'opd-adapter-checkpoint-step-00000002.kiln-checkpoint'
      || preparedOpd.prompts !== ''
      || preparedOpd.noteHidden
      || !/2 training candidates \(data 999999999999…\)/.test(preparedOpd.note)) {
      fail(`Prepared OPD resume fields were incomplete or unsafe: ${JSON.stringify(preparedOpd)}`);
    }

    await page.$eval('#opd-checkpoint-interval', (input) => { input.value = '0'; });
    const opdZeroCadence = await page.$eval('#opd-checkpoint-interval', (input) => ({
      valid: input.checkValidity(),
      rangeUnderflow: input.validity.rangeUnderflow,
      formValid: input.form?.checkValidity(),
    }));
    if (opdZeroCadence.valid || !opdZeroCadence.rangeUnderflow || opdZeroCadence.formValid) {
      fail(`Native OPD checkpoint cadence validation should reject zero: ${JSON.stringify(opdZeroCadence)}`);
    }
    await page.$eval('#opd-checkpoint-interval', (input) => { input.value = '25'; });
    await clickAndWait(page, '#opd-use-sample', 'Could not restore OPD prompts for validation checks');
    await page.$eval('#opd-resume-checkpoint', (input) => { input.value = '../bad.kiln-checkpoint'; });
    await clickAndWait(page, '#opd-form button[type="submit"]', 'Could not exercise invalid OPD resume validation');
    await expectTrainingToast(page, 'OPD resume checkpoint must be one direct .kiln-checkpoint basename, without a path.');
    await page.$eval('#opd-resume-checkpoint', (input) => { input.value = 'opd-adapter-checkpoint-step-00000002.kiln-checkpoint'; });
    await page.$eval('#opd-form', (form) => { form.dataset.resumeTeacherRevision = `sha256:${'8'.repeat(64)}`; });
    await clickAndWait(page, '#opd-form button[type="submit"]', 'Could not exercise OPD teacher-revision validation');
    await expectTrainingToast(page, 'OPD resume requires the exact teacher revision recorded by the checkpoint. Restore or re-register that teacher before submitting.');
    await page.$eval('#opd-form', (form) => { form.dataset.resumeTeacherRevision = ''; });
    await clickAndWait(page, '#opd-form button[type="submit"]', 'Could not exercise OPD missing teacher-binding validation');
    await expectTrainingToast(page, 'This OPD checkpoint does not expose an exact teacher identity and revision, so it cannot be prepared safely in the browser.');

    await goToPrimaryTab(page, 'playground');
    await expectDisabled(page, '#chat-send', true, 'Quick Inference send should start disabled until text is entered');
    await page.waitForFunction(() => document.getElementById('chat-thinking-budget-preview')?.dataset.state === 'ready', { timeout: 5000 })
      .catch(() => fail('Playground did not resolve the effective server thinking-budget defaults'));
    const initialThinkingBudget = await page.evaluate(() => ({
      tokensMode: document.getElementById('chat-thinking-budget-tokens-mode')?.value,
      timeMode: document.getElementById('chat-thinking-budget-time-mode')?.value,
      tokensModeDisabled: document.getElementById('chat-thinking-budget-tokens-mode')?.disabled,
      timeModeDisabled: document.getElementById('chat-thinking-budget-time-mode')?.disabled,
      customHidden: document.getElementById('chat-thinking-budget-custom')?.hidden,
      tokensFieldHidden: document.getElementById('chat-thinking-budget-tokens-field')?.hidden,
      timeFieldHidden: document.getElementById('chat-thinking-budget-time-field')?.hidden,
      tokensDisabled: document.getElementById('chat-thinking-budget-tokens')?.disabled,
      secondsDisabled: document.getElementById('chat-thinking-budget-seconds')?.disabled,
      previewTokens: document.getElementById('chat-thinking-budget-preview-tokens')?.textContent,
      previewTime: document.getElementById('chat-thinking-budget-preview-time')?.textContent,
      previewTokensSource: document.getElementById('chat-thinking-budget-preview-tokens-source')?.textContent,
      previewTimeSource: document.getElementById('chat-thinking-budget-preview-time-source')?.textContent,
    }));
    if (initialThinkingBudget.tokensMode !== 'inherit'
        || initialThinkingBudget.timeMode !== 'inherit'
        || initialThinkingBudget.tokensModeDisabled
        || initialThinkingBudget.timeModeDisabled) {
      fail(`Thinking budgets should start as independently enabled Inherit controls, got ${JSON.stringify(initialThinkingBudget)}`);
    }
    if (!initialThinkingBudget.customHidden
        || !initialThinkingBudget.tokensFieldHidden
        || !initialThinkingBudget.timeFieldHidden
        || !initialThinkingBudget.tokensDisabled
        || !initialThinkingBudget.secondsDisabled) {
      fail(`Inherited thinking budgets should hide and disable finite-limit fields, got ${JSON.stringify(initialThinkingBudget)}`);
    }
    if (initialThinkingBudget.previewTokens !== '64'
        || initialThinkingBudget.previewTime !== '1.5 s'
        || initialThinkingBudget.previewTokensSource !== 'server'
        || initialThinkingBudget.previewTimeSource !== 'server') {
      fail(`Inherited thinking-budget preview should show the effective server pair, got ${JSON.stringify(initialThinkingBudget)}`);
    }

    const browserBudgetCases = thinkingBudgetContract.resolution_cases
      .filter((budgetCase) => budgetCase.scope === 'request')
      .map((budgetCase) => {
        let secondsValue = '';
        if (budgetCase.time.state === 'limit') {
          const wholeSeconds = Math.floor(budgetCase.time.value / 1000);
          const remainingMs = budgetCase.time.value % 1000;
          secondsValue = remainingMs === 0
            ? String(wholeSeconds)
            : `${wholeSeconds}.${String(remainingMs).padStart(3, '0').replace(/0+$/, '')}`;
        }
        return {
          name: budgetCase.name,
          tokensMode: budgetCase.tokens.state,
          timeMode: budgetCase.time.state,
          tokensValue: budgetCase.tokens.state === 'limit' ? String(budgetCase.tokens.value) : '',
          secondsValue,
          expectedRequest: budgetCase.request,
        };
      });
    const browserBudgetResults = await page.evaluate((budgetCases) => budgetCases.map((budgetCase) => {
      const tokensMode = document.getElementById('chat-thinking-budget-tokens-mode');
      const timeMode = document.getElementById('chat-thinking-budget-time-mode');
      const tokens = document.getElementById('chat-thinking-budget-tokens');
      const seconds = document.getElementById('chat-thinking-budget-seconds');
      tokensMode.value = budgetCase.tokensMode;
      tokensMode.dispatchEvent(new Event('change', { bubbles: true }));
      timeMode.value = budgetCase.timeMode;
      timeMode.dispatchEvent(new Event('change', { bubbles: true }));
      tokens.value = budgetCase.tokensValue;
      seconds.value = budgetCase.secondsValue;
      const request = {};
      const hooks = window.__kilnThinkingBudgetTest;
      hooks.applyRequest(request, hooks.readRequest());
      return { name: budgetCase.name, request };
    }), browserBudgetCases);
    for (let i = 0; i < browserBudgetCases.length; i += 1) {
      const expected = browserBudgetCases[i];
      const actual = browserBudgetResults[i];
      if (actual?.name !== expected.name
          || JSON.stringify(actual.request) !== JSON.stringify(expected.expectedRequest)) {
        fail(`Browser thinking-budget contract ${expected.name} produced ${JSON.stringify(actual?.request)}; expected ${JSON.stringify(expected.expectedRequest)}`);
      }
    }

    const legacyBudgetMigration = await page.evaluate(() => {
      window.__kilnThinkingBudgetTest.applySettings({
        thinkingBudgetMode: 'custom',
        thinkingBudgetTokens: '7',
        thinkingBudgetSeconds: '',
      });
      return {
        tokensMode: document.getElementById('chat-thinking-budget-tokens-mode')?.value,
        timeMode: document.getElementById('chat-thinking-budget-time-mode')?.value,
        tokensValue: document.getElementById('chat-thinking-budget-tokens')?.value,
        secondsValue: document.getElementById('chat-thinking-budget-seconds')?.value,
      };
    });
    if (JSON.stringify(legacyBudgetMigration) !== JSON.stringify({
      tokensMode: 'limit',
      timeMode: 'unlimited',
      tokensValue: '7',
      secondsValue: '',
    })) {
      fail(`Legacy combined thinking-budget settings should migrate dimension by dimension, got ${JSON.stringify(legacyBudgetMigration)}`);
    }

    await page.select('#chat-thinking-budget-tokens-mode', 'limit');
    await page.select('#chat-thinking-budget-time-mode', 'inherit');
    await page.waitForFunction(() => (
      document.getElementById('chat-advanced')?.hidden === false
      && document.getElementById('chat-thinking-budget-custom')?.hidden === false
      && document.getElementById('chat-thinking-budget-tokens-field')?.hidden === false
      && document.getElementById('chat-thinking-budget-tokens')?.disabled === false
      && document.getElementById('chat-thinking-budget-time-field')?.hidden === true
      && document.getElementById('chat-thinking-budget-seconds')?.disabled === true
    ), { timeout: 5000 }).catch(() => fail('A finite token budget should preserve the inherited time dimension'));

    await page.$eval('#chat-thinking-budget-tokens', (input) => {
      input.value = '5';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.waitForFunction(() => (
      document.getElementById('chat-thinking-budget-preview-tokens')?.textContent === '5'
      && document.getElementById('chat-thinking-budget-preview-tokens-source')?.textContent === 'request'
      && document.getElementById('chat-thinking-budget-preview-time')?.textContent === '1.5 s'
      && document.getElementById('chat-thinking-budget-preview-time-source')?.textContent === 'server'
    ), { timeout: 5000 }).catch(() => fail('Independent token override should preserve the previewed server time default'));
    await page.select('#chat-thinking-budget-time-mode', 'limit');
    await page.$eval('#chat-thinking-budget-seconds', (input) => {
      input.value = '1';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.waitForFunction(() => {
      const settings = JSON.parse(localStorage.getItem('kiln.playground.settings.v1') || '{}');
      return settings.thinkingBudgetTokens === '5' && settings.thinkingBudgetSeconds === '1';
    }, { timeout: 5000 }).catch(() => fail('A valid custom thinking budget should persist before malformed-edit coverage'));

    let malformedBudgetRequests = 0;
    const trackMalformedBudgetRequest = (request) => {
      if (request.method() === 'POST' && request.url().endsWith('/v1/chat/completions')) {
        malformedBudgetRequests += 1;
      }
    };
    page.on('request', trackMalformedBudgetRequest);
    await page.$eval('#chat-thinking-budget-tokens', (input) => {
      input.value = '1.5';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.$eval('#chat-thinking-budget-seconds', (input) => { input.value = '1'; });
    await page.waitForFunction(() => document.getElementById('chat-thinking-budget-preview')?.dataset.state === 'incomplete', { timeout: 5000 })
      .catch(() => fail('Malformed finite thinking budget should mark the effective preview incomplete'));
    await page.type('#chat-input', 'This malformed budget must not be sent.');
    await page.click('#chat-send');
    await page.waitForFunction(() => (
      /Thinking tokens must be a whole number/.test(document.getElementById('toasts')?.textContent || '')
      && document.activeElement?.id === 'chat-thinking-budget-tokens'
    ), { timeout: 5000 }).catch(async () => {
      const state = await page.evaluate(() => {
        const input = document.getElementById('chat-thinking-budget-tokens');
        const send = document.getElementById('chat-send');
        return {
          activeElement: document.activeElement?.id,
          inputValue: input?.value,
          inputValid: input?.checkValidity(),
          inputBadInput: input?.validity?.badInput,
          inputStepMismatch: input?.validity?.stepMismatch,
          sendDisabled: send?.disabled,
          sendText: send?.textContent,
          stopHidden: document.getElementById('chat-stop')?.hidden,
          compareToggle: document.getElementById('chat-compare-toggle')?.checked,
          sendCenterTarget: (() => {
            const rect = send?.getBoundingClientRect();
            if (!rect) return null;
            const target = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
            return { id: target?.id, tag: target?.tagName, className: target?.className };
          })(),
          visibleDialogs: Array.from(document.querySelectorAll('[role="dialog"]'))
            .filter((dialog) => !dialog.hidden && getComputedStyle(dialog).display !== 'none')
            .map((dialog) => dialog.id),
          toasts: document.getElementById('toasts')?.textContent || '',
        };
      });
      fail(`A decimal token budget should show an error and focus the token field: ${JSON.stringify(state)}`);
    });

    await page.$eval('#toasts', (toasts) => toasts.replaceChildren());
    await page.$eval('#chat-thinking-budget-tokens', (input) => { input.value = ''; input.focus(); });
    await page.keyboard.type('e');
    const nativeBadInput = await page.$eval('#chat-thinking-budget-tokens', (input) => input.validity.badInput);
    if (!nativeBadInput) fail('The browser did not enter the native malformed number state used by the budget regression');
    await page.click('#chat-send');
    await page.waitForFunction(() => (
      /Thinking tokens must be a whole number/.test(document.getElementById('toasts')?.textContent || '')
      && document.activeElement?.id === 'chat-thinking-budget-tokens'
    ), { timeout: 5000 }).catch(() => fail('A native malformed token state should show an error instead of reading as blank'));
    await new Promise((resolve) => setTimeout(resolve, 250));
    page.off('request', trackMalformedBudgetRequest);
    if (malformedBudgetRequests !== 0) {
      fail(`Malformed thinking budgets must not produce a request; observed ${malformedBudgetRequests}`);
    }
    const persistedAfterBadInput = await page.evaluate(() => {
      const settings = JSON.parse(localStorage.getItem('kiln.playground.settings.v1') || '{}');
      return [settings.thinkingBudgetTokens, settings.thinkingBudgetSeconds];
    });
    if (JSON.stringify(persistedAfterBadInput) !== JSON.stringify(['5', '1'])) {
      fail(`A malformed native number state must not overwrite the last valid persisted budget, got ${JSON.stringify(persistedAfterBadInput)}`);
    }
    await page.$eval('#chat-input', (input) => {
      input.value = '';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });

    await page.$eval('#chat-thinking-budget-tokens', (input) => {
      input.value = '0';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.$eval('#chat-thinking-budget-seconds', (input) => {
      input.value = '1.25';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.waitForFunction(() => {
      const settings = JSON.parse(localStorage.getItem('kiln.playground.settings.v1') || '{}');
      return settings.thinkingBudgetTokensMode === 'limit'
        && settings.thinkingBudgetTimeMode === 'limit'
        && settings.thinkingBudgetTokens === '0'
        && settings.thinkingBudgetSeconds === '1.25'
        && settings.advancedOpen === true;
    }, { timeout: 5000 }).catch(() => fail('Independent finite thinking budgets should persist with the Playground settings'));
    await page.waitForFunction(() => (
      document.getElementById('chat-thinking-budget-preview-tokens')?.textContent === '0'
      && document.getElementById('chat-thinking-budget-preview-time')?.textContent === '1.25 s'
      && document.getElementById('chat-thinking-budget-preview-tokens-source')?.textContent === 'request'
      && document.getElementById('chat-thinking-budget-preview-time-source')?.textContent === 'request'
    ), { timeout: 5000 }).catch(() => fail('Finite request pair should update the effective thinking-budget preview'));

    const thinkingBeforeDisable = await page.$eval('#chat-enable-thinking', (input) => {
      input.scrollIntoView({ block: 'center' });
      return { checked: input.checked, disabled: input.disabled };
    });
    if (!thinkingBeforeDisable.checked) {
      fail(`Thinking should still be enabled before the disable-state regression, got ${JSON.stringify(thinkingBeforeDisable)}`);
    }
    await page.click('#chat-enable-thinking');
    await expectDisabled(page, '#chat-thinking-budget-tokens-mode', true, 'Thinking off should disable the token budget mode');
    await expectDisabled(page, '#chat-thinking-budget-time-mode', true, 'Thinking off should disable the time budget mode');
    await expectDisabled(page, '#chat-thinking-budget-tokens', true, 'Thinking off should disable the token budget');
    await expectDisabled(page, '#chat-thinking-budget-seconds', true, 'Thinking off should disable the time budget');
    await page.waitForFunction(() => document.getElementById('chat-thinking-budget-preview')?.dataset.state === 'inactive', { timeout: 5000 })
      .catch(() => fail('Thinking off should mark the configured budget preview inactive'));
    await page.click('#chat-enable-thinking');
    await expectDisabled(page, '#chat-thinking-budget-tokens-mode', false, 'Thinking on should re-enable the token budget mode');
    await expectDisabled(page, '#chat-thinking-budget-time-mode', false, 'Thinking on should re-enable the time budget mode');
    await expectDisabled(page, '#chat-thinking-budget-tokens', false, 'Thinking on should restore custom token budget editing');
    await expectDisabled(page, '#chat-thinking-budget-seconds', false, 'Thinking on should restore custom time budget editing');
    await page.waitForFunction(() => document.getElementById('chat-thinking-budget-preview')?.dataset.state === 'ready', { timeout: 5000 })
      .catch(() => fail('Thinking on should reactivate the effective budget preview'));

    await page.type('#chat-input', 'Explain Kiln in one sentence.');
    await expectDisabled(page, '#chat-send', false, 'Quick Inference send should enable after text is entered');
    await page.evaluate(() => { window.__copiedText = ''; });
    const customBudgetRequestPromise = page.waitForRequest(
      (request) => request.method() === 'POST' && request.url().endsWith('/v1/chat/completions'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#chat-send', 'Could not click Quick Inference send');
    const customBudgetRequest = await customBudgetRequestPromise.catch(() => fail('Custom budget Quick Inference request was not sent'));
    const customBudgetBody = JSON.parse(customBudgetRequest.postData() || '{}');
    if (customBudgetBody.thinking_budget_tokens !== 0 || customBudgetBody.thinking_budget_ms !== 1250) {
      fail(`Custom thinking budget should send token zero and fractional seconds as integer ms, got ${JSON.stringify(customBudgetBody)}`);
    }
    await waitForPanelText(page, '#chat-output', /Kiln serves one tuned model and learns from feedback live\./, 'Quick Inference response missing');
    await waitForPanelText(page, '#chat-output', /token cap/, 'Quick Inference should render the final thinking-budget outcome');
    await expectDisabled(page, '#copy-chat-response', false, 'Copy response should enable after an assistant response renders');
    await clickAndWait(page, '#copy-chat-response', 'Could not click Copy response');
    await page.waitForFunction(
      () => window.__copiedText === 'Kiln serves one tuned model and learns from feedback live.',
      { timeout: 5000 },
    ).catch(async () => {
      const copiedText = await page.evaluate(() => window.__copiedText).catch(() => undefined);
      fail(`Copy response should copy the latest assistant response, got ${JSON.stringify(copiedText)}`);
    });
    await clickAndWait(page, '#chat-clear', 'Could not click Quick Inference clear');

    await page.select('#chat-thinking-budget-tokens-mode', 'unlimited');
    await page.select('#chat-thinking-budget-time-mode', 'unlimited');
    const unlimitedState = await page.evaluate(() => ({
      customHidden: document.getElementById('chat-thinking-budget-custom')?.hidden,
      tokensFieldHidden: document.getElementById('chat-thinking-budget-tokens-field')?.hidden,
      timeFieldHidden: document.getElementById('chat-thinking-budget-time-field')?.hidden,
      tokensDisabled: document.getElementById('chat-thinking-budget-tokens')?.disabled,
      secondsDisabled: document.getElementById('chat-thinking-budget-seconds')?.disabled,
      previewTokens: document.getElementById('chat-thinking-budget-preview-tokens')?.textContent,
      previewTime: document.getElementById('chat-thinking-budget-preview-time')?.textContent,
      previewTokensSource: document.getElementById('chat-thinking-budget-preview-tokens-source')?.textContent,
      previewTimeSource: document.getElementById('chat-thinking-budget-preview-time-source')?.textContent,
    }));
    if (!unlimitedState.customHidden
        || !unlimitedState.tokensFieldHidden
        || !unlimitedState.timeFieldHidden
        || !unlimitedState.tokensDisabled
        || !unlimitedState.secondsDisabled
        || unlimitedState.previewTokens !== 'unlimited'
        || unlimitedState.previewTime !== 'unlimited'
        || unlimitedState.previewTokensSource !== 'request'
        || unlimitedState.previewTimeSource !== 'request') {
      fail(`Unlimited thinking budget should hide and disable custom fields, got ${JSON.stringify(unlimitedState)}`);
    }
    await page.type('#chat-input', 'Explain Kiln in one sentence.');
    const unlimitedRequestPromise = page.waitForRequest(
      (request) => request.method() === 'POST' && request.url().endsWith('/v1/chat/completions'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#chat-send', 'Could not send Unlimited thinking-budget request');
    const unlimitedRequest = await unlimitedRequestPromise.catch(() => fail('Unlimited budget Quick Inference request was not sent'));
    const unlimitedBody = JSON.parse(unlimitedRequest.postData() || '{}');
    if (unlimitedBody.thinking_budget_tokens !== null || unlimitedBody.thinking_budget_ms !== null) {
      fail(`Unlimited thinking budget should send explicit nulls, got ${JSON.stringify(unlimitedBody)}`);
    }
    await waitForPanelText(page, '#chat-output', /Kiln serves one tuned model and learns from feedback live\./, 'Unlimited budget response missing');
    await clickAndWait(page, '#chat-clear', 'Could not clear Unlimited thinking-budget response');

    await page.select('#chat-thinking-budget-tokens-mode', 'inherit');
    await page.select('#chat-thinking-budget-time-mode', 'inherit');
    await page.waitForFunction(() => (
      document.getElementById('chat-thinking-budget-preview-tokens')?.textContent === '64'
      && document.getElementById('chat-thinking-budget-preview-time')?.textContent === '1.5 s'
      && document.getElementById('chat-thinking-budget-preview-tokens-source')?.textContent === 'server'
      && document.getElementById('chat-thinking-budget-preview-time-source')?.textContent === 'server'
    ), { timeout: 5000 }).catch(() => fail('Inherited request should restore the effective server-default preview'));
    await page.type('#chat-input', 'Explain Kiln in one sentence.');
    const inheritedRequestPromise = page.waitForRequest(
      (request) => request.method() === 'POST' && request.url().endsWith('/v1/chat/completions'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#chat-send', 'Could not send inherited thinking-budget request');
    const inheritedRequest = await inheritedRequestPromise.catch(() => fail('Inherited budget Quick Inference request was not sent'));
    const inheritedBody = JSON.parse(inheritedRequest.postData() || '{}');
    if ('thinking_budget_tokens' in inheritedBody || 'thinking_budget_ms' in inheritedBody) {
      fail(`Inherited thinking budget should omit both request fields, got ${JSON.stringify(inheritedBody)}`);
    }
    await waitForPanelText(page, '#chat-output', /Kiln serves one tuned model and learns from feedback live\./, 'Inherited budget response missing');
    await clickAndWait(page, '#chat-clear', 'Could not clear inherited thinking-budget response');
    await waitForPanelText(page, '#chat-output', /Send a message to test inference\./, 'Quick Inference clear should restore the empty state');
    await expectDisabled(page, '#copy-chat-response', true, 'Copy response should disable after clearing chat');

    // Compare must consume and render the exact same SSE finish metadata as
    // normal Playground chat. Side A closes naturally; side B hits the token
    // cap and finishes by length, proving per-side outcomes stay independent.
    await page.$eval('#chat-compare-toggle', (input) => {
      if (!input.checked) input.click();
    });
    await page.waitForFunction(() => {
      const pair = document.getElementById('chat-compare-pair');
      return pair && getComputedStyle(pair).display !== 'none';
    }, { timeout: 5000 }).catch(async () => {
      const state = await page.evaluate(() => ({
        checked: document.getElementById('chat-compare-toggle')?.checked,
        pairInlineDisplay: document.getElementById('chat-compare-pair')?.style.display,
        pairComputedDisplay: getComputedStyle(document.getElementById('chat-compare-pair')).display,
        adapterBDisplay: document.getElementById('chat-adapter-b')?.style.display,
      }));
      fail(`Compare mode should reveal the side-by-side response panel, got ${JSON.stringify(state)}`);
    });
    await page.waitForFunction(() => (
      Array.from(document.getElementById('chat-adapter-b')?.options || [])
        .some((option) => option.value === 'adapter-alpha')
    ), { timeout: 5000 }).catch(() => fail('Compare adapter options should include adapter-alpha'));
    await page.select('#chat-adapter', '');
    await page.select('#chat-adapter-b', 'adapter-alpha');
    await page.type('#chat-input', 'Compare budget outcomes.');
    await expectDisabled(page, '#chat-send', false, 'Compare send should enable after text is entered');
    await clickAndWait(page, '#chat-send', 'Could not send the Playground comparison');
    await page.waitForFunction(() => {
      const a = document.getElementById('chat-compare-a-body')?.textContent || '';
      const b = document.getElementById('chat-compare-b-body')?.textContent || '';
      return /Base final\./.test(a)
        && /natural close/.test(a)
        && /Adapter final\./.test(b)
        && /token cap/.test(b)
        && /truncated/.test(b)
        && document.getElementById('chat-save-judgment')?.disabled === false;
    }, { timeout: 5000 }).catch(() => fail('Compare mode should render independent content, budget outcomes, and finish reasons'));
    await page.type('#chat-input', 'Compare stream failure.');
    await clickAndWait(page, '#chat-send', 'Could not send the compare stream-error regression');
    await page.waitForFunction(() => {
      const a = document.getElementById('chat-compare-a-body')?.textContent || '';
      const b = document.getElementById('chat-compare-b-body')?.textContent || '';
      return /Healthy side\./.test(a) && /Injected compare stream failure\./.test(b);
    }, { timeout: 5000 }).catch(() => fail('A structured generation_error must fail only its compare side and remain visible'));
    await page.$eval('#chat-compare-toggle', (input) => {
      if (input.checked) input.click();
    });

    // ---- Evals golden path: a first-five-minutes user with no data must not
    // dead-end. Suites empty state routes to Datasets (no raw-API copy);
    // Datasets empty state offers a one-click sample upload through the real
    // endpoint, then hands off to suite synthesis.
    await goToPrimaryTab(page, 'evals');
    await clickAndWait(page, '#evals-tab-suites', 'Could not open the Suites sub-tab');
    await waitForPanelText(page, '#suites-list', /No eval suites yet/, 'Suites empty state missing');
    await waitForPanelText(page, '#suites-list', /report card/, 'Suites empty state should explain suites in plain language');
    const suitesEmptyText = await page.$eval('#suites-list', (el) => el.textContent || '');
    if (/POST|\/v1\/eval\/suites/.test(suitesEmptyText)) fail('Suites empty state should not surface raw API instructions in primary copy');
    const suitesCtaTitle = await page.$eval('#suites-list .eval-empty-cta', (el) => el.title).catch(() => '');
    if (!suitesCtaTitle.includes('/v1/eval/suites')) fail('Suites empty-state CTA should keep the API mention as a title for power users');
    await clickAndWait(page, '#suites-list .eval-empty-cta', 'Could not click the suites empty-state CTA');
    await waitForVisiblePanel(page, '#tab-evals-datasets', 'Suites empty-state CTA should land on the Datasets sub-tab');

    await waitForPanelText(page, '#datasets-list', /No datasets yet/, 'Datasets empty state missing');
    await waitForPanelText(page, '#datasets-list', /Try a sample dataset/, 'Datasets empty state should offer the sample dataset CTA');
    await expectDisabled(page, '#dataset-from-corrections', true, 'Corrections dataset CTA should be disabled while the basket is empty');
    const corrCtaHint = await page.$eval('#dataset-from-corrections', (el) => el.title);
    if (!/write the ideal answer/i.test(corrCtaHint)) fail(`Disabled corrections CTA should explain what unlocks it, got: ${JSON.stringify(corrCtaHint)}`);

    const sampleUploadRequestPromise = page.waitForRequest(
      (request) => request.method() === 'POST' && request.url().endsWith('/v1/eval/datasets/upload'),
      { timeout: 5000 },
    );
    const sampleUploadResponsePromise = page.waitForResponse(
      (response) => response.url().endsWith('/v1/eval/datasets/upload'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#use-sample-dataset', 'Could not click Try a sample dataset');
    await sampleUploadRequestPromise.catch(() => fail('Try a sample dataset did not POST /v1/eval/datasets/upload'));
    const sampleUploadResponse = await sampleUploadResponsePromise.catch(() => fail('Sample dataset upload got no response'));
    if (sampleUploadResponse.status() !== 200) {
      const detail = await sampleUploadResponse.text().catch(() => '');
      fail(`Sample dataset upload should return 200 (mock validates every JSONL row against the SftConversation contract), got ${sampleUploadResponse.status()}: ${detail}`);
    }
    await expectTrainingToast(page, 'Sample dataset added (10 rows) — next: synthesize an eval suite from it');
    await waitForPanelText(page, '#datasets-list', /sample-coding-agent/, 'Datasets list should refresh with the uploaded sample');
    // The handoff lands on the next step of the golden path: the synthesize
    // panel, pre-filled from the sample dataset.
    await page.waitForFunction(
      () => {
        const panel = document.getElementById('synthesize-panel');
        return panel && !panel.hidden;
      },
      { timeout: 5000 },
    ).catch(() => fail('Sample upload should open the synthesize panel (create a suite is the next step)'));
    const synthSuiteName = await page.$eval('#synth-suite-name', (el) => el.value);
    if (synthSuiteName !== 'sample-coding-agent-eval') fail(`Synthesize panel should pre-fill the suite name from the sample dataset, got ${JSON.stringify(synthSuiteName)}`);
    const synthSourceSplit = await page.$eval('#synth-source-split', (el) => el.value);
    if (synthSourceSplit !== 'holdout') fail(`Dataset synthesis should default to the persisted holdout partition, got ${JSON.stringify(synthSourceSplit)}`);

    // Named training consumes the persisted train partition and makes a
    // held-out-vs-diagnostic choice explicit. Exercise the complete browser
    // request and rendered provenance rather than checking source strings.
    if (!setEvalSuites) fail('Train/eval separation smoke controls are unavailable');
    setEvalSuites([{ name: 'smoke-heldout', num_examples: 2 }]);
    await goToPrimaryTab(page, 'training');
    await clickAndWait(page, '#training-tab-sft', 'Could not open SFT for named-dataset training');
    await page.waitForFunction(
      () => document.getElementById('sft-prove-row')?.hidden === false,
      { timeout: 5000 },
    ).catch(() => fail('A registered eval suite should reveal post-training evaluation controls'));
    const initialProveState = await page.evaluate(() => ({
      scope: document.getElementById('sft-prove-scope')?.value,
      scopeDisabled: document.getElementById('sft-prove-scope')?.disabled,
      suiteDisabled: document.getElementById('sft-prove-suite')?.disabled,
      hint: document.getElementById('sft-prove-hint')?.textContent,
    }));
    if (initialProveState.scope !== 'held-out'
      || !initialProveState.scopeDisabled
      || !initialProveState.suiteDisabled
      || !/rejects the submission.*overlaps/i.test(initialProveState.hint || '')) {
      fail(`Post-training eval should default visibly and disabled to held-out: ${JSON.stringify(initialProveState)}`);
    }
    await clickAndWait(page, '#sft-pick-toggle', 'Could not open the named SFT dataset picker');
    await page.waitForFunction(
      () => Array.from(document.getElementById('sft-dataset-pick')?.options || [])
        .some((option) => /sample-coding-agent.*8 train.*1 validation.*1 holdout/.test(option.textContent || '')),
      { timeout: 5000 },
    ).catch(() => fail('Named SFT picker should show exact train, validation, and holdout counts'));
    await page.select('#sft-dataset-pick', 'sample-coding-agent');
    await waitForPanelText(page, '#sft-data-status', /8 train examples.*1 validation.*1 holdout.*persisted train partition/, 'Named SFT status should identify the exact train partition');
    await clickAndWait(page, '#sft-prove', 'Could not enable post-training evaluation');
    await page.select('#sft-prove-scope', 'train-set-eval');
    await waitForPanelText(page, '#sft-prove-hint', /Diagnostic only:.*cannot satisfy a minimum-accuracy promotion gate/, 'Train-set evaluation should be visibly diagnostic-only');
    const namedSftRequestPromise = page.waitForRequest(
      (request) => request.method() === 'POST' && request.url().endsWith('/v1/train/sft'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#sft-form button[type="submit"]', 'Could not submit named SFT train partition');
    const namedSftRequest = await namedSftRequestPromise.catch(() => fail('Named SFT did not POST /v1/train/sft'));
    const namedSftBody = JSON.parse(namedSftRequest.postData() || '{}');
    if (namedSftBody.dataset !== 'sample-coding-agent'
      || namedSftBody.dataset_split !== 'train'
      || namedSftBody.post_eval?.suite !== 'smoke-heldout'
      || namedSftBody.post_eval?.data_scope !== 'train-set-eval'
      || namedSftBody.post_eval?.include_baseline !== true) {
      fail(`Named SFT request lost partition or eval-scope intent: ${JSON.stringify(namedSftBody)}`);
    }
    await expectTrainingToast(page, 'SFT job submitted · seed 18446744073709551615');
    await waitForPanelText(page, '#tab-queue', /sample-coding-agent.*train.*8 rows/, 'Training card should show admitted named-dataset provenance');
    await clickAndWait(page, '[data-train-job-id="smoke-sft-named"]', 'Could not open named SFT job provenance');
    await waitForPanelText(page, '#train-drill-content', /Data source[\s\S]*named_dataset[\s\S]*Dataset[\s\S]*sample-coding-agent[\s\S]*Partition[\s\S]*train[\s\S]*Rows admitted[\s\S]*8/, 'Training drill should show source, dataset, partition, and admitted count');
    await waitForPanelText(page, '#train-drill-content', new RegExp(`Admitted corpus SHA-256[\\s\\S]*sha256:${'4'.repeat(64)}[\\s\\S]*Split manifest SHA-256[\\s\\S]*sha256:${'3'.repeat(64)}`), 'Training drill should show exact admitted and split-manifest identities');
    await clickAndWait(page, '#train-drill-close', 'Could not close named SFT provenance drill');
    await goToPrimaryTab(page, 'evals');

    // ---- Judgment Undo: record an A/B pick, then the toast's Undo must
    // DELETE exactly the row the POST response identified (judgment_id)
    // and restore the visible counts.
    await clickAndWait(page, '#evals-tab-judgments', 'Could not open the Judgments sub-tab');
    await waitForVisiblePanel(page, '#tab-evals-judgments', 'Judgments sub-tab did not activate');
    await page.type('#judgment-create-name', 'smoke-judgments');
    await clickAndWait(page, '#judgment-create-btn', 'Could not create the smoke judgment dataset');
    await waitForPanelText(page, '#judgment-rows-count', /Judging into "smoke-judgments"/, 'Judgment viewer should open on the new dataset');
    await page.type('#judgment-prompt', 'Judge this smoke pair.');
    await clickAndWait(page, '#judgment-generate-btn', 'Could not generate the judgment pair');
    await page.waitForFunction(
      () => !document.getElementById('judgment-actions')?.hidden
        && /smoke reply for judging/.test(document.getElementById('judgment-a-text')?.textContent || '')
        && /smoke reply for judging/.test(document.getElementById('judgment-b-text')?.textContent || ''),
      { timeout: 5000 },
    ).catch(() => fail('Judgment pair did not stream into both compare cards'));
    const judgmentRecordResponsePromise = page.waitForResponse(
      (response) => response.url().endsWith('/v1/judgments/smoke-judgments/rows')
        && response.request().method() === 'POST',
      { timeout: 5000 },
    );
    await clickAndWait(page, '#judgment-pick-a', 'Could not record the A vote');
    const judgmentRecordResponse = await judgmentRecordResponsePromise
      .catch(() => fail('Recording a judgment did not POST /v1/judgments/smoke-judgments/rows'));
    const judgmentRecordBody = await judgmentRecordResponse.json().catch(() => ({}));
    if (judgmentRecordBody.judgment_id !== 'smoke-judgment-1') {
      fail(`Judgment POST response should carry the new row's judgment_id, got ${JSON.stringify(judgmentRecordBody.judgment_id)}`);
    }
    if (judgmentRecordBody.num_rows !== 1) {
      fail(`Judgment POST response should keep the manifest fields (num_rows), got ${JSON.stringify(judgmentRecordBody.num_rows)}`);
    }
    await waitForPanelText(page, '#judgment-rows-count', /1 judgments in "smoke-judgments"/, 'Row count should reflect the recorded judgment');
    const undoDeleteRequestPromise = page.waitForRequest(
      (request) => request.method() === 'DELETE'
        && request.url().endsWith('/v1/judgments/smoke-judgments/rows/smoke-judgment-1'),
      { timeout: 5000 },
    );
    const undoClicked = await page.evaluate(() => {
      const toastEl = Array.from(document.querySelectorAll('#toasts .toast-action'))
        .find((candidate) => /Recorded A wins in "smoke-judgments"/.test(candidate.textContent || ''));
      const undoBtn = Array.from(toastEl?.querySelectorAll('.toast-action-btn') || [])
        .find((button) => button.textContent?.trim() === 'Undo');
      if (!undoBtn) return false;
      undoBtn.click();
      return true;
    });
    if (!undoClicked) fail('Recorded-judgment toast should offer an Undo action');
    await undoDeleteRequestPromise
      .catch(() => fail('Undo did not DELETE /v1/judgments/smoke-judgments/rows/smoke-judgment-1'));
    await expectTrainingToast(page, 'Undone — judgment removed from "smoke-judgments"');
    await waitForPanelText(page, '#judgment-rows-count', /0 judgments in "smoke-judgments"/, 'Undo should restore the judgment viewer count');
    await waitForPanelText(page, '#judgments-list', /0 judgments/, 'Judgments list should refresh to zero rows after Undo');

    // ---- Eval drill power-user depth: the raw-JSON toggle mirrors the
    // request drill modal's `raw` button; the outcomes export downloads one
    // JSON line per outcome across every run of the job.
    await clickAndWait(page, '#evals-tab-jobs', 'Could not open the Jobs sub-tab');
    await waitForVisiblePanel(page, '#tab-evals-jobs', 'Jobs sub-tab did not activate');
    await waitForPanelText(page, '#eval-jobs-list', /seed 18446744073709551615/, 'Eval job cards should show the exact decimal effective seed without JavaScript rounding');
    await clickAndWait(page, '[data-job-id="smoke-eval-full"]', 'Could not open the eval drill modal for the completed compare job');
    await page.waitForFunction(
      () => document.getElementById('eval-drill-modal')?.hidden === false
        && document.getElementById('drill-title')?.textContent === 'smoke-suite',
      { timeout: 5000 },
    ).catch(() => fail('Eval drill modal did not open on the completed compare job'));
    await waitForPanelText(page, '#drill-meta', /seed 18446744073709551615/, 'Eval drill metadata should expose the exact effective seed');
    await waitForPanelText(page, '#drill-headline', /Base weights[\s\S]*1 shard/, 'Eval drill should show the compact base-weight identity');
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '#drill-headline [data-copy-base-weight]', 'Could not copy the eval base-weight identity');
    await page.waitForFunction(
      () => window.__copiedText === 'sha256:c62f9f56234c61c943716ae3b8783c851fb41a2551e31f17d15f1b0c346339b5',
      { timeout: 5000 },
    ).catch(() => fail('Eval base-weight copy should preserve the complete aggregate digest'));
    await expectTrainingToast(page, 'Base-weight identity copied');
    await waitForPanelText(page, '#drill-headline', /Execution[\s\S]*rocm · gfx1151/, 'Eval drill should show the admission-time execution identity');
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '#drill-headline [data-copy-execution]', 'Could not copy the eval execution identity');
    await page.waitForFunction(
      () => window.__copiedText === `sha256:${'b'.repeat(64)}`,
      { timeout: 5000 },
    ).catch(() => fail('Eval execution copy should preserve the complete canonical digest'));
    await expectTrainingToast(page, 'Execution identity copied');
    // The drill defaults to the first failure (ex-2), not the first outcome.
    await waitForPanelText(page, '#drill-detail', /seed 18446744073709551613/, 'Eval outcome detail should expose the exact derived decoder seed');
    await page.waitForSelector('#drill-detail [data-outcome-copy="seed"]', { timeout: 5000 })
      .catch(() => fail('Eval outcome detail should provide a copy-seed action'));
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '#drill-detail [data-outcome-copy="seed"]', 'Could not copy the eval completion seed');
    await page.waitForFunction(
      () => window.__copiedText === '18446744073709551613',
      { timeout: 5000 },
    ).catch(async () => {
      const copiedText = await page.evaluate(() => window.__copiedText).catch(() => undefined);
      fail(`Eval copy-seed should preserve all u64 digits, got ${JSON.stringify(copiedText)}`);
    });
    await expectTrainingToast(page, 'Copied seed');

    // Raw JSON toggle: first click appends the pretty-printed cached job
    // payload, second click removes it.
    await clickAndWait(page, '#drill-raw', 'Could not click the eval drill raw JSON toggle');
    await page.waitForSelector('#drill-raw-block', { timeout: 5000 })
      .catch(() => fail('Raw JSON toggle did not render #drill-raw-block'));
    const rawPayload = await page.$eval('#drill-raw-block', (el) => el.textContent || '');
    let parsedRaw = null;
    try { parsedRaw = JSON.parse(rawPayload); } catch { fail('Eval drill raw JSON block should contain valid JSON'); }
    if (parsedRaw.job_id !== 'smoke-eval-full') fail(`Raw JSON should show the drilled job, got job_id ${JSON.stringify(parsedRaw.job_id)}`);
    if (parsedRaw.effective_seed !== '18446744073709551615') fail(`Raw JSON should preserve the exact decimal effective seed, got ${JSON.stringify(parsedRaw.effective_seed)}`);
    if (parsedRaw.base_weight_shard_manifest?.shards?.[0]?.filename !== 'model.safetensors') fail('Raw JSON should preserve the complete per-shard base-weight manifest');
    if (parsedRaw.execution_provenance?.backend?.device !== 'gfx1151' || parsedRaw.execution_provenance?.provenance_sha256 !== `sha256:${'b'.repeat(64)}`) fail('Raw JSON should preserve the complete execution provenance record');
    if (!Array.isArray(parsedRaw.runs) || parsedRaw.runs.length !== 2) fail('Raw JSON should carry both runs of the compare job');
    if (!rawPayload.includes('\n  "runs"')) fail('Raw JSON should be pretty-printed with 2-space indentation');
    if (!rawPayload.includes('"detail": "expected 42, got 41"')) fail('Raw JSON should surface per-outcome fields like detail');
    await clickAndWait(page, '#drill-raw', 'Could not click the raw JSON toggle a second time');
    await page.waitForFunction(() => !document.getElementById('drill-raw-block'), { timeout: 5000 })
      .catch(() => fail('Second raw JSON click should remove the block'));

    // Outcomes JSONL download: stub object URLs and anchor clicks so the
    // blob is inspectable in-page, then assert the line schema and that
    // every created URL is revoked (no leak across repeated downloads).
    await expectDisabled(page, '#drill-download-outcomes', false, 'Download outcomes should enable when the job has outcomes');
    await page.evaluate(() => {
      window.__smokeDownloads = { created: 0, revoked: 0, name: '', blob: null };
      URL.createObjectURL = (blob) => {
        window.__smokeDownloads.created += 1;
        window.__smokeDownloads.blob = blob;
        return `blob:smoke-${window.__smokeDownloads.created}`;
      };
      URL.revokeObjectURL = () => { window.__smokeDownloads.revoked += 1; };
      // Keep headless Chrome from navigating to the fake blob URL; buttons
      // are unaffected (they use HTMLElement.prototype.click).
      HTMLAnchorElement.prototype.click = function () { window.__smokeDownloads.name = this.download; };
    });
    await clickAndWait(page, '#drill-download-outcomes', 'Could not click Download outcomes (.jsonl)');
    await page.waitForFunction(
      () => window.__smokeDownloads?.created === 1 && window.__smokeDownloads?.revoked === 1,
      { timeout: 5000 },
    ).catch(async () => {
      const counts = await page.evaluate(() => window.__smokeDownloads).catch(() => undefined);
      fail(`Download should create then revoke exactly one object URL, got ${JSON.stringify(counts)}`);
    });
    // Toast asserted on the FIRST download — a repeat within 4s dedupes
    // into the same toast with an appended ×2 counter, which exact-match
    // would miss.
    await expectTrainingToast(page, 'Downloaded 6 outcomes as smoke-suite-smoke-ev.outcomes.jsonl');
    const download = await page.evaluate(async () => ({
      name: window.__smokeDownloads.name,
      text: window.__smokeDownloads.blob ? await window.__smokeDownloads.blob.text() : null,
    }));
    if (download.name !== 'smoke-suite-smoke-ev.outcomes.jsonl') {
      fail(`Outcomes download filename should be <suite>-<job8>.outcomes.jsonl, got ${JSON.stringify(download.name)}`);
    }
    const jsonlLines = (download.text || '').split('\n').filter((line) => line.length > 0);
    if (jsonlLines.length !== 6) fail(`Outcomes JSONL should carry 6 lines (2 runs x 3 outcomes), got ${jsonlLines.length}`);
    const parsedLines = jsonlLines.map((line, index) => {
      try { return JSON.parse(line); } catch { fail(`Outcomes JSONL line ${index + 1} is not valid JSON: ${line}`); }
      return null;
    });
    const firstLine = parsedLines[0];
    if (firstLine.suite !== 'smoke-suite' || firstLine.job_id !== 'smoke-eval-full' || firstLine.adapter !== 'base') {
      fail(`First JSONL line should carry standalone context (suite/job_id/adapter), got ${JSON.stringify(firstLine)}`);
    }
    if (firstLine.example_id !== 'ex-1' || firstLine.example_index !== 0 || firstLine.kind !== 'pass' || firstLine.score !== 1 || firstLine.completion_text !== '4') {
      fail(`First JSONL line should carry the outcome verdict fields, got ${JSON.stringify(firstLine)}`);
    }
    if (firstLine.effective_seed !== '18446744073709551615' || firstLine.seed_derivation !== 'kiln.eval-seed.v1') {
      fail(`Outcomes JSONL should preserve job-level seed provenance, got ${JSON.stringify(firstLine)}`);
    }
    if (firstLine.base_weight_shard_manifest?.aggregate_sha256 !== 'sha256:c62f9f56234c61c943716ae3b8783c851fb41a2551e31f17d15f1b0c346339b5') {
      fail(`Outcomes JSONL should preserve exact base-weight provenance, got ${JSON.stringify(firstLine.base_weight_shard_manifest)}`);
    }
    if (firstLine.execution_provenance?.backend?.device !== 'gfx1151' || firstLine.execution_provenance?.provenance_sha256 !== `sha256:${'b'.repeat(64)}`) {
      fail(`Outcomes JSONL should preserve exact execution provenance, got ${JSON.stringify(firstLine.execution_provenance)}`);
    }
    if (firstLine.generation_seed !== '18446744073709551614') fail(`Outcomes JSONL should preserve the exact per-completion seed, got ${JSON.stringify(firstLine.generation_seed)}`);
    if (!parsedLines.some((line) => line.adapter === 'smoke-tuned')) fail('Outcomes JSONL should include the second run, tagged with its adapter');
    const failLine = parsedLines.find((line) => line.adapter === 'base' && line.example_id === 'ex-2');
    if (!failLine || failLine.kind !== 'fail' || failLine.detail !== 'expected 42, got 41' || failLine.latency_ms !== 42) {
      fail(`Failing JSONL line should carry kind/detail/latency, got ${JSON.stringify(failLine)}`);
    }
    // A second download must mint and revoke a fresh URL (no leak on repeats).
    await clickAndWait(page, '#drill-download-outcomes', 'Could not click Download outcomes a second time');
    await page.waitForFunction(
      () => window.__smokeDownloads?.created === 2 && window.__smokeDownloads?.revoked === 2,
      { timeout: 5000 },
    ).catch(() => fail('Repeated downloads should revoke every object URL they create'));
    await clickAndWait(page, '#drill-close', 'Could not close the eval drill modal');
    await page.waitForFunction(() => document.getElementById('eval-drill-modal')?.hidden === true, { timeout: 5000 })
      .catch(() => fail('Eval drill modal did not close'));

    // No-outcomes job: the export button must disable with an explanatory
    // title (the raw toggle stays live — the job JSON itself exists).
    await clickAndWait(page, '[data-job-id="smoke-eval-empty"]', 'Could not open the eval drill modal for the queued job');
    await page.waitForFunction(() => document.getElementById('eval-drill-modal')?.hidden === false, { timeout: 5000 })
      .catch(() => fail('Eval drill modal did not open on the queued job'));
    await expectDisabled(page, '#drill-download-outcomes', true, 'Download outcomes should disable on a job with no outcomes');
    const emptyDownloadTitle = await page.$eval('#drill-download-outcomes', (el) => el.title);
    if (!/No outcomes yet/.test(emptyDownloadTitle)) fail(`Disabled outcomes download should explain what unlocks it, got: ${JSON.stringify(emptyDownloadTitle)}`);
    await clickAndWait(page, '#drill-close', 'Could not close the queued-job eval drill modal');
    await page.waitForFunction(() => document.getElementById('eval-drill-modal')?.hidden === true, { timeout: 5000 })
      .catch(() => fail('Queued-job eval drill modal did not close'));

    // ---- Agent traces (roadmap PR 23): the Distill → Agent traces tab is
    // actionable — entering it lists the existing index, outcome chips and
    // a working-dir filter narrow the list client-side, the scan path
    // round-trips into the discover POST body (persisted in localStorage),
    // and each card drills into the recorded conversation at
    // #distill/traces/{id} through the shared modal manager.
    await goToPrimaryTab(page, 'distill');
    await clickAndWait(page, '#distill-tab-traces', 'Could not open the Agent traces distill tab');
    await waitForPanelText(page, '#agent-traces-list', /trace-good-1111/, 'Traces tab should list the first mock pi session on entry (no manual scan needed)');
    await waitForPanelText(page, '#agent-traces-list', /trace-fail-2222/, 'Traces tab should list the second mock pi session');
    const tracesPaneCopy = await page.$eval('#distill-tab-traces-pane', (el) => el.innerText || '');
    if (/\/v1\/agent\/traces/.test(tracesPaneCopy)) fail('Traces tab primary copy should not name raw API routes');

    // Outcome chips: exit ≠ 0 narrows to the failing session.
    const traceChips = await page.$$eval('#agent-traces-chips .agent-chip', (els) => els.map((el) => el.textContent.trim()));
    if (traceChips.length < 5) fail(`Traces tab should render the outcome filter chips, got ${JSON.stringify(traceChips)}`);
    if (!traceChips[0].startsWith('All sessions')) fail(`The first outcome chip should be "All sessions", got ${JSON.stringify(traceChips[0])}`);
    await clickAndWait(page, '[data-trace-chip="exitnz"]', 'Could not click the exit ≠ 0 outcome chip');
    await page.waitForFunction(() => {
      const text = document.getElementById('agent-traces-list')?.textContent || '';
      return text.includes('trace-fail-2222') && !text.includes('trace-good-1111');
    }, { timeout: 5000 }).catch(() => fail('exit ≠ 0 chip should narrow the list to the failing session'));
    await clickAndWait(page, '[data-trace-chip="all"]', 'Could not reset the outcome chip filter');
    await waitForPanelText(page, '#agent-traces-list', /trace-good-1111/, 'All-sessions chip should restore the full list');

    // working_dir filter narrows client-side over the fetched index.
    await page.type('#agent-traces-dir', 'widget');
    await page.waitForFunction(() => {
      const text = document.getElementById('agent-traces-list')?.textContent || '';
      return text.includes('trace-good-1111') && !text.includes('trace-fail-2222');
    }, { timeout: 5000 }).catch(() => fail('Working-dir filter should narrow the list to sessions under …/widget'));
    await page.$eval('#agent-traces-dir', (el) => {
      el.value = '';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitForPanelText(page, '#agent-traces-list', /trace-fail-2222/, 'Clearing the working-dir filter should restore the list');

    // Custom scan path: rides the discover POST body and persists.
    await page.$eval('#agent-traces-path', (el) => { el.value = ''; });
    await page.type('#agent-traces-path', '/tmp/smoke-pi-sessions');
    const discoverResponsePromise = page.waitForResponse(
      (response) => response.url().endsWith('/v1/agent/traces/discover'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#agent-traces-refresh', 'Could not click the pi session scan button');
    const discoverResponse = await discoverResponsePromise.catch(() => fail('Scanning should POST the discover endpoint'));
    let discoverBody = null;
    try { discoverBody = JSON.parse(discoverResponse.request().postData() || '{}'); } catch { fail('Discover POST body should be JSON'); }
    if (discoverBody.path !== '/tmp/smoke-pi-sessions') fail(`Custom scan path should ride the discover body, got ${JSON.stringify(discoverBody)}`);
    await waitForPanelText(page, '#agent-traces-list', /Indexed 2 pi sessions from/, 'Scan headline should report the indexed count and folder');
    const persistedScanPath = await page.evaluate(() => localStorage.getItem('kiln.traces.scanPath'));
    if (persistedScanPath !== '/tmp/smoke-pi-sessions') fail(`Scan path should persist to localStorage, got ${JSON.stringify(persistedScanPath)}`);
    // Empty input = server default: the path key is omitted entirely.
    await page.$eval('#agent-traces-path', (el) => { el.value = ''; });
    const defaultDiscoverPromise = page.waitForResponse(
      (response) => response.url().endsWith('/v1/agent/traces/discover'),
      { timeout: 5000 },
    );
    await clickAndWait(page, '#agent-traces-refresh', 'Could not rescan with the default sessions folder');
    const defaultDiscover = await defaultDiscoverPromise.catch(() => fail('Default rescan should POST the discover endpoint'));
    let defaultDiscoverBody = null;
    try { defaultDiscoverBody = JSON.parse(defaultDiscover.request().postData() || '{}'); } catch { fail('Default discover POST body should be JSON'); }
    if ('path' in defaultDiscoverBody) fail(`Empty scan path should omit the path field (server default), got ${JSON.stringify(defaultDiscoverBody)}`);
    await waitForPanelText(page, '#agent-traces-list', /trace-good-1111/, 'Rescan should re-render the session list');

    // Drill-in: focus the card, open it, and the shared modal manager
    // moves focus into the dialog while the hash gains the id segment.
    await page.focus('[data-trace-open="trace-good-1111"]');
    await clickAndWait(page, '[data-trace-open="trace-good-1111"]', 'Could not open the pi session drill');
    await page.waitForFunction(
      () => document.getElementById('trace-drill-modal')?.hidden === false
        && window.location.hash === '#distill/traces/trace-good-1111',
      { timeout: 5000 },
    ).catch(async () => {
      const actual = await page.evaluate(() => ({
        hidden: document.getElementById('trace-drill-modal')?.hidden,
        hash: window.location.hash,
      })).catch(() => ({ hidden: 'unknown', hash: 'unknown' }));
      fail(`Trace drill should open with its id in the hash, got ${JSON.stringify(actual)}`);
    });
    const traceFocus = await page.evaluate(() => ({
      inModal: document.getElementById('trace-drill-modal').contains(document.activeElement),
      active: document.activeElement?.id || document.activeElement?.tagName || 'none',
    }));
    if (!traceFocus.inModal) fail(`Opening the trace drill should move focus into the dialog, got activeElement=${traceFocus.active}`);
    await waitForPanelText(page, '#trace-drill-content', /\/home\/smoke\/projects\/widget/, 'Trace drill should show the session working directory');
    await waitForPanelText(page, '#trace-drill-content', /Fix the widget test/, 'Trace drill should render the user task turn');
    await waitForPanelText(page, '#trace-drill-content', /cargo test -p widget/, 'Trace drill should render the tool-call arguments');
    await waitForPanelText(page, '#trace-drill-content', /test result: ok\. 12 passed/, 'Trace drill should render the tool-result turn');
    const traceDrillText = await page.$eval('#trace-drill-content', (el) => el.textContent || '');
    for (const expected of ['system', 'user', 'assistant', 'tool result', 'bash', 'run the tests first']) {
      if (!traceDrillText.includes(expected)) fail(`Trace drill should role-label turns and name tool calls; missing ${JSON.stringify(expected)}`);
    }

    // Raw JSON toggle mirrors the other drills' raw buttons.
    await clickAndWait(page, '#trace-drill-raw', 'Could not toggle the trace raw JSON view');
    await page.waitForSelector('#trace-drill-raw-block', { timeout: 5000 })
      .catch(() => fail('Trace raw JSON toggle did not render its block'));
    const traceRawPayload = await page.$eval('#trace-drill-raw-block', (el) => el.textContent || '');
    let parsedTraceRaw = null;
    try { parsedTraceRaw = JSON.parse(traceRawPayload); } catch { fail('Trace raw JSON block should contain valid JSON'); }
    if (parsedTraceRaw.id !== 'trace-good-1111') fail(`Trace raw JSON should show the drilled session, got id ${JSON.stringify(parsedTraceRaw.id)}`);
    if (!Array.isArray(parsedTraceRaw.trajectory) || parsedTraceRaw.trajectory.length !== 3) fail('Trace raw JSON should carry the full trajectory');
    await clickAndWait(page, '#trace-drill-raw', 'Could not untoggle the trace raw JSON view');
    await page.waitForFunction(() => !document.getElementById('trace-drill-raw-block'), { timeout: 5000 })
      .catch(() => fail('Second trace raw JSON click should remove the block'));

    // Escape closes through the modal manager: the hash entry the open
    // minted is consumed, and focus returns to the card that opened it.
    await page.keyboard.press('Escape');
    await page.waitForFunction(
      () => document.getElementById('trace-drill-modal')?.hidden === true
        && window.location.hash === '#distill/traces',
      { timeout: 5000 },
    ).catch(async () => {
      const actual = await page.evaluate(() => ({
        hidden: document.getElementById('trace-drill-modal')?.hidden,
        hash: window.location.hash,
      })).catch(() => ({ hidden: 'unknown', hash: 'unknown' }));
      fail(`Escape should close the trace drill and consume its hash entry, got ${JSON.stringify(actual)}`);
    });
    const traceRestoredFocus = await page.evaluate(() => document.activeElement?.getAttribute('data-trace-open') || 'none');
    if (traceRestoredFocus !== 'trace-good-1111') fail(`Closing the trace drill should restore focus to its card, got ${traceRestoredFocus}`);

    // Deep link: a fresh boot on #distill/traces/{id} opens the drill on
    // that session (the per-trace GET feeds it; no list fetch required).
    // Via about:blank so the hash change is a real navigation, not a
    // same-document fragment jump.
    await page.goto('about:blank');
    await page.goto(`${baseUrl}/ui#distill/traces/trace-fail-2222`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.getElementById('trace-drill-modal')?.hidden === false
        && (document.getElementById('trace-drill-content')?.textContent || '').includes('/home/smoke/projects/gadget'),
      { timeout: 5000 },
    ).catch(() => fail('Booting on #distill/traces/{id} should deep-link into the trace drill'));

  } finally {
    await browser.close();
  }
}

const uiAppSource = await readUiAppBundle();
const uiIndexSource = await readFile(uiIndexPath, 'utf8');
const uiDemoSource = await readFile(uiDemoJsPath, 'utf8');
checkThinkingBudgetParserContract(uiAppSource);
checkRuntimeConfigSchemaContract(uiAppSource);
checkTrainingOptimizerSupportContract(uiAppSource, uiIndexSource, uiDemoSource);
checkTrainingOptimizerRankValidator(uiAppSource);

if (process.argv.includes('--static-only')) {
  console.log('server UI static smoke check passed');
  process.exit(0);
}

const emptyAdapterScenario = await startServer({ availableAdapters: [] });
try {
  console.log('[smoke] empty adapter scenario start');
  await runSmoke(emptyAdapterScenario.baseUrl, {
    expectEmptyAdapters: true,
    setRecentRequests: emptyAdapterScenario.setRecentRequests,
    setTrainingOptimizerAvailable: emptyAdapterScenario.setTrainingOptimizerAvailable,
    getTrainingSubmitRequests: emptyAdapterScenario.getTrainingSubmitRequests,
  });
  console.log('[smoke] empty adapter scenario passed');
} finally {
  await new Promise((accept) => emptyAdapterScenario.server.close(accept));
}

const {
  server,
  baseUrl,
  setFailRuntimeConfig,
  setRecentRequests,
  setTrainingOptimizerAvailable,
  setEvalSuites,
  getTrainingSubmitRequests,
} = await startServer();
try {
  console.log('[smoke] default scenario desktop start');
  await runSmoke(baseUrl, {
    setFailRuntimeConfig,
    setRecentRequests,
    setTrainingOptimizerAvailable,
    setEvalSuites,
    getTrainingSubmitRequests,
  });
  console.log('[smoke] default scenario desktop passed; mobile start');
  await runMobileOnboardingSmoke(baseUrl);
  console.log('[smoke] default scenario mobile passed');
} finally {
  await new Promise((accept) => server.close(accept));
}

const coldStartScenario = await startServer({ modelsCold: true, servedModelId: 'Qwen3.5-4B-resolved' });
try {
  console.log('[smoke] model cold-start scenario start');
  await runModelColdStartSmoke(coldStartScenario.baseUrl, {
    setModelsCold: coldStartScenario.setModelsCold,
    getModelsRequests: coldStartScenario.getModelsRequests,
  });
  console.log('[smoke] model cold-start scenario passed');
} finally {
  await new Promise((accept) => coldStartScenario.server.close(accept));
}

const failureScenario = await startServer({ failDashboardApis: true });
try {
  console.log('[smoke] failure scenario start');
  await runSmoke(failureScenario.baseUrl, {
    expectFailureStates: true,
    setFailDashboardApis: failureScenario.setFailDashboardApis,
    setTrainingOptimizerAvailable: failureScenario.setTrainingOptimizerAvailable,
    getTrainingSubmitRequests: failureScenario.getTrainingSubmitRequests,
  });
  console.log('[smoke] failure scenario passed');
} finally {
  await new Promise((accept) => failureScenario.server.close(accept));
}

console.log('server UI smoke check passed');
