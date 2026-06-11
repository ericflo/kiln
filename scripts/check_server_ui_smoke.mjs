#!/usr/bin/env node
import http from 'node:http';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import process from 'node:process';
import { tmpdir } from 'node:os';

const repoRoot = resolve(import.meta.dirname, '..');
const uiDir = resolve(repoRoot, 'crates/kiln-server/src/ui');
const uiIndexPath = resolve(uiDir, 'index.html');
const uiStylesPath = resolve(uiDir, 'styles.css');
const uiDemoJsPath = resolve(uiDir, 'demo.js');
const uiAppJsPath = resolve(uiDir, 'app.js');
const uiVendorDir = resolve(uiDir, 'vendor');
const uiVendorFiles = {
  'xterm.js': 'application/javascript',
  'xterm.css': 'text/css',
  'xterm-addon-fit.js': 'application/javascript',
};
const expectedHeaderHelpLinks = [
  ['Quickstart', 'https://ericflo.github.io/kiln/quickstart.html'],
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
  if (!Array.isArray(body?.examples) || body.examples.length !== 1) return 'SFT examples should be a one-item array from the sample payload';
  const messages = body.examples[0]?.messages;
  if (!Array.isArray(messages) || messages.length !== 2) return 'SFT sample example should include user and assistant messages';
  if (messages[0]?.role !== 'user' || messages[1]?.role !== 'assistant') return 'SFT sample messages should preserve chat roles';
  if (body?.config?.output_name !== 'sft-adapter') return 'SFT output_name should be nested under config';
  if (body?.config?.auto_load !== true) return 'SFT auto_load should be true by default';
  if ('learning_rate' in (body?.config || {})) return 'SFT learning_rate should be omitted when the field is blank (server resolves the per-optimizer default)';
  if (body?.config?.epochs !== 3) return 'SFT epochs should be numeric and nested under config';
  if (body?.config?.lora_rank !== 8) return 'SFT lora_rank should be numeric and nested under config';
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
  if ('epochs' in (body?.config || {}) || 'output_name' in body || 'adapter_name' in body || 'num_epochs' in body) return 'GRPO payload should not use stale SFT/top-level training config fields';
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
  if (['load', 'unload', 'upload', 'merge'].includes(match[1])) return null;
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
    ...overrides,
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
const piRow = () => smokeRecentRow({
  user_agent: 'pi/1.2.0',
  prompt_preview: 'hello from pi',
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
  const apiState = { failDashboardApis, modelsCold, recentRequests: [], modelsRequests: 0 };
  const uiHtml = await readFile(uiIndexPath, 'utf8');
  const uiStyles = await readFile(uiStylesPath, 'utf8');
  const uiDemoJs = await readFile(uiDemoJsPath, 'utf8');
  const uiAppJs = await readFile(uiAppJsPath, 'utf8');
  availableAdapters = availableAdapters.map((adapter) => ({ ...adapter }));
  let activeAdapter = availableAdapters.find((adapter) => adapter.active)?.name || null;
  const completedTrainingJobs = [];
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
  const smokeEvalJobs = [
    {
      job_id: 'smoke-eval-full',
      suite_name: 'smoke-suite',
      adapters: [null, 'smoke-tuned'],
      submission_kind: 'compare',
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
        gpu_memory: { total_vram_gb: 24, model_gb: 8, kv_cache_gb: 2, training_budget_gb: 4 },
        checks: [{ name: 'mock smoke server', pass: true }],
      });
      return;
    }
    // Mirrors api/config.rs ConfigResponse (vram / kv_cache / training /
    // memory_budget) — the runtime-config expander fetches this once per
    // open, never on a poll loop.
    if (url.pathname === '/v1/config') {
      json(res, {
        vram: { detected_gb: 25.8, source: 'nvidia-smi' },
        kv_cache: { num_blocks: 1024, num_blocks_source: 'auto', fp8_enabled: true },
        training: { checkpoint_segments: 4, checkpoint_segments_source: 'auto', checkpointing_enabled: true },
        memory_budget: {
          total_vram_gb: 25.8,
          model_gb: 8.2,
          kv_cache_gb: 2.1,
          training_budget_gb: 4.0,
          inference_memory_fraction: 0.55,
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
    if (url.pathname === '/v1/train/queue' || url.pathname === '/v1/train/status') {
      json(res, { running: runningTrainingJob, queued: [], completed: completedTrainingJobs });
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
      runningTrainingJob = {
        job_id: 'smoke-sft',
        job_type: 'sft',
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
      };
      setTimeout(() => json(res, { message: 'SFT job submitted', job_id: 'smoke-sft' }), 75);
      return;
    }
    if (url.pathname === '/v1/train/grpo') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for GRPO training' }));
        return;
      }
      const body = await readJsonBody(req);
      const validationError = validateGrpoPayload(body);
      if (validationError) {
        apiBadRequest(res, validationError);
        return;
      }
      completedTrainingJobs.unshift({
        job_id: 'smoke-grpo',
        job_type: 'grpo',
        state: 'Completed',
        progress: 1,
        adapter_name: body.config.output_name,
        elapsed_secs: 1,
      });
      setTimeout(() => json(res, { message: 'GRPO job submitted', job_id: 'smoke-grpo' }), 75);
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
      json(res, { window_secs: 60, sample_count: 0, tok_per_sec: 0, p50_itl_ms: 0, p99_itl_ms: 0, mean_itl_ms: 0 });
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
    // the eval drill walk (raw JSON toggle + outcomes JSONL export); their
    // adapters reference no real adapter card, so the rest of the
    // dashboard is unaffected.
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
        runs: job.finished_runs,
        ...(job.state === 'queued' || job.state === 'running' ? { progress: job.progress } : {}),
      });
      return;
    }
    // Same shape as the production endpoints — only the empty cases
    // are exercised in smoke, but keep them registered so future UI
    // background polls don't surprise the mock.
    if (url.pathname === '/v1/eval/suites') {
      json(res, { suites: [] });
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
      if (!body?.stream || !/Explain Kiln in one sentence\./.test(prompt)) {
        res.writeHead(400, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Unexpected Quick Inference smoke request' }));
        return;
      }
      sse(res, [
        { choices: [{ delta: { role: 'assistant' } }] },
        { choices: [{ delta: { content: 'Kiln serves one tuned model' } }] },
        { choices: [{ delta: { content: ' and learns from feedback live.' } }] },
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
    setRecentRequests: (rows) => { apiState.recentRequests = rows; },
    setModelsCold: (value) => { apiState.modelsCold = value; },
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
async function expectActivePageAndHash(page, name, message) {
  await page.waitForFunction(
    (targetName) => {
      const section = document.querySelector(`#page-${targetName}`);
      return section
        && section.classList.contains('active')
        && !section.hidden
        && !section.hasAttribute('inert')
        && window.location.hash === `#${targetName}`;
    },
    { timeout: 5000 },
    name,
  ).catch(async () => {
    const actual = await page.evaluate(() => ({
      hash: window.location.hash,
      page: document.querySelector('.page.active')?.id || 'none',
    })).catch(() => ({ hash: 'unknown', page: 'unknown' }));
    fail(`${message}: expected page-${name} active with hash #${name}, got page=${actual.page} hash=${actual.hash}`);
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
    const tabNames = ['queue', 'sft', 'grpo'];
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

  await page.keyboard.press('ArrowLeft');
  await expectTrainingTabA11yState(page, 'sft', 'ArrowLeft should return to the SFT training tab');

  await page.keyboard.press('Home');
  await expectTrainingTabA11yState(page, 'queue', 'Home should activate the Queue training tab');

  await page.keyboard.press('End');
  await expectTrainingTabA11yState(page, 'grpo', 'End should activate the GRPO training tab');
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
    await clickAndWait(page, '#training-tab-grpo', 'Could not activate mobile GRPO tab');
    await waitForVisiblePanel(page, '#tab-grpo', 'Mobile GRPO tab did not activate');
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

async function runSmoke(baseUrl, { expectFailureStates = false, expectEmptyAdapters = false, setFailDashboardApis = null, setRecentRequests = null } = {}) {
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
      if (expectFailureStates && /Failed to load resource: the server responded with a status of 503/.test(text)) return;
      pageErrors.push(text);
    });
    page.on('requestfailed', (request) => {
      pageErrors.push(`${request.method()} ${request.url()} failed: ${request.failure()?.errorText || 'unknown error'}`);
    });

    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
    await page.goto(`${baseUrl}/ui`, { waitUntil: 'networkidle0', timeout: 10000 });

    if (pageErrors.length > 0) fail(`UI emitted browser errors: ${pageErrors.join('; ')}`);

    await expectText(page, '.header h1', /^\s*kiln\s*$/i, 'Header did not render');
    await expectHeaderHelpLinks(page);
    await expectNoForbiddenPublicityCopy(page, 'Server dashboard');

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
        await goToPrimaryTab(page, 'overview');
        await recoverPanel('#server-status', /GPU VRAM/, 'Server status did not recover after the APIs healed');
        await page.waitForSelector('#server-status .vram-donut svg', { timeout: 5000 })
          .catch(() => fail('VRAM donut missing from recovered server status panel'));
        // The expander's Retry refetches /v1/config (no poll loop covers it).
        await page.click('#runtime-config [data-rc-refresh]')
          .catch(() => fail('Could not click the runtime config Retry button after the APIs healed'));
        await waitForPanelText(page, '#runtime-config-body', /nvidia-smi/, 'Runtime config did not recover after Retry');
        await recoverPanel('#decode-perf-panel', /No streaming completions/i, 'Decode panel did not recover after the APIs healed');
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
        await recoverPanel('#decode-perf-panel', /No streaming completions/i, 'Decode panel stuck on failure HTML after second recovery');
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
    }

    // --- Hash navigation: tab clicks mint history entries, browser
    // Back/Forward walks them, and live hash edits route through the
    // page whitelist (roadmap PR 16). ---
    await expectActivePageAndHash(page, 'overview', 'Landing on /ui (no fragment) should repair the URL to #overview in place');
    await goToPrimaryTab(page, 'adapters');
    await expectActivePageAndHash(page, 'adapters', 'Clicking the Adapters tab should push #adapters');
    await goToPrimaryTab(page, 'training');
    await expectActivePageAndHash(page, 'training', 'Clicking the Training tab should push #training');
    await page.goBack();
    await expectActivePageAndHash(page, 'adapters', 'Browser Back from Training should return to Adapters');
    await page.goBack();
    await expectActivePageAndHash(page, 'overview', 'Second browser Back should return to Overview');
    await page.goForward();
    await expectActivePageAndHash(page, 'adapters', 'Browser Forward should re-land on Adapters');
    // A live hash edit (address bar / location.hash) must activate the page.
    await page.evaluate(() => { window.location.hash = '#evals'; });
    await expectActivePageAndHash(page, 'evals', 'Setting location.hash = #evals should activate the Evals page');
    // A junk hash falls back to Overview and is repaired via replaceState —
    // the junk entry must NOT survive in history for Back to trip over.
    await page.evaluate(() => { window.location.hash = '#nonsense'; });
    await expectActivePageAndHash(page, 'overview', 'A junk hash should fall back to Overview with the URL repaired');
    await page.goBack();
    await expectActivePageAndHash(page, 'evals', 'Back after a junk hash should land on Evals — #nonsense must not pollute history');

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

    await waitForPanelText(page, '#decode-perf-panel', /No streaming completions/i, 'Empty decode performance state missing');
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
    // ConfigResponse fields (VRAM detection, KV cache geometry, budgets).
    await clickAndWait(page, '#runtime-config > summary', 'Could not open the runtime config expander');
    await waitForPanelText(page, '#runtime-config-body', /nvidia-smi/, 'Runtime config should render the VRAM detection source');
    await waitForPanelText(page, '#runtime-config-body', /25\.8 GB/, 'Runtime config should render detected VRAM');
    await waitForPanelText(page, '#runtime-config-body', /1,024/, 'Runtime config should render the KV cache block count');
    await waitForPanelText(page, '#runtime-config-body', /FP8 cache/, 'Runtime config should render the fp8 cache mode');
    await waitForPanelText(page, '#runtime-config-body', /Training reserve/, 'Runtime config should render the memory budget rows');
    await clickAndWait(page, '#runtime-config [data-rc-raw]', 'Could not toggle the runtime config raw JSON');
    const configRawShown = await page.$eval(
      '#runtime-config [data-rc-raw-pre]',
      (el) => !el.hidden && /"memory_budget"/.test(el.textContent || ''),
    );
    if (!configRawShown) fail('Runtime config raw JSON toggle should reveal the pretty-printed /v1/config payload');

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
      configIntact: /nvidia-smi/.test(document.getElementById('runtime-config-body')?.textContent || ''),
    }));
    if (!overviewSteadyState.svg) fail('VRAM donut disappeared after subsequent health polls');
    if (overviewSteadyState.donuts !== 1) fail(`Expected exactly one VRAM donut, found ${overviewSteadyState.donuts}`);
    if (overviewSteadyState.model !== 'Qwen3.5-4B') fail(`Header model stat should render from /health, got "${overviewSteadyState.model}"`);
    if (!/^\d+[sm]/.test(overviewSteadyState.uptime)) fail(`Header uptime stat should render, got "${overviewSteadyState.uptime}"`);
    // The expander is a static SIBLING of the keyed #server-status region —
    // the ≥2 genuine repaints above must not close it or destroy its content.
    if (!overviewSteadyState.configOpen) fail('Runtime config expander lost its open state across server-status repaints');
    if (!overviewSteadyState.configIntact) fail('Runtime config content was destroyed by the server-status repaint');

    await goToPrimaryTab(page, 'playground');
    await waitForPanelText(page, '#chat-output', /Send a message to test inference\./, 'Quick Inference empty state missing');
    await expectPanelLink(page, '#chat-output .empty', '/health', '/health');
    await expectPanelLink(page, '#chat-output .empty', 'Troubleshooting guide', 'https://ericflo.github.io/kiln/troubleshooting.html');

    await goToPrimaryTab(page, 'training');
    // Empty queue lands on the Train·SFT form by design (see mobile flow note).
    await waitForVisiblePanel(page, '#tab-sft', 'Empty training queue should land on the Train·SFT form');
    await clickAndWait(page, '#training-tab-queue', 'Could not activate Queue tab before keyboard checks');
    await expectTrainingTabKeyboardNavigation(page);

    await clickAndWait(page, '#training-tab-sft', 'Could not open SFT tab');
    await waitForVisiblePanel(page, '#tab-sft', 'SFT tab did not activate');
    await expectDisabled(page, '#sft-form button[type="submit"]', true, 'SFT submit should start disabled until examples are provided');
    await clickAndWait(page, '#use-sft-sample', 'Could not click SFT sample payload button');
    await expectDisabled(page, '#sft-form button[type="submit"]', false, 'SFT submit should enable after sample payload is clicked');
    await clickAndWait(page, '#sft-form button[type="submit"]', 'Could not submit sample SFT payload');
    await expectDisabled(page, '#sft-form button[type="submit"]', true, 'SFT submit should disable while the job is submitting');
    await expectTrainingToast(page, 'SFT job submitted');
    await expectActiveTrainingTab(page, 'queue', 'Submitting SFT should switch back to the training queue tab');
    await waitForPanelText(page, '#tab-queue', /smoke-sf/, 'Training queue should refresh after SFT submit');
    await waitForPanelText(page, '#tab-queue', /Adapter:\s*sft-adapter/, 'Training queue should show the submitted SFT adapter name');
    await waitForPanelText(page, '#tab-queue', /running/, 'Training queue should show the SFT job as running');

    // Drill modal for the RUNNING job: Stop must be live (running jobs are
    // cancellable cooperatively — the trainer aborts at the next step
    // boundary) and must route through the same DELETE /v1/train/queue/:id
    // path the queue card uses. The modal stays open across the cancel so
    // failures (and the cancelled repaint) surface in it.
    await clickAndWait(page, '[data-train-job-id="smoke-sft"]', 'Could not open the train drill modal for the running SFT job');
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
    await clickAndWait(page, '#train-drill-close', 'Could not close the train drill modal');
    await page.waitForFunction(() => document.getElementById('train-drill-modal')?.hidden === true, { timeout: 5000 })
      .catch(() => fail('Train drill modal did not close'));

    await clickAndWait(page, '#training-tab-grpo', 'Could not open GRPO tab');
    await waitForVisiblePanel(page, '#tab-grpo', 'GRPO tab did not activate');
    await expectDisabled(page, '#grpo-form button[type="submit"]', true, 'GRPO submit should start disabled until groups are provided');
    await clickAndWait(page, '#use-grpo-sample', 'Could not click GRPO sample payload button');
    await expectDisabled(page, '#grpo-form button[type="submit"]', false, 'GRPO submit should enable after sample payload is clicked');
    await clickAndWait(page, '#grpo-form button[type="submit"]', 'Could not submit sample GRPO payload');
    await expectDisabled(page, '#grpo-form button[type="submit"]', true, 'GRPO submit should disable while the job is submitting');
    await expectTrainingToast(page, 'GRPO job submitted');
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
    await clickAndWait(page, '#train-drill-close', 'Could not close the GRPO train drill modal');
    await page.waitForFunction(() => document.getElementById('train-drill-modal')?.hidden === true, { timeout: 5000 })
      .catch(() => fail('GRPO train drill modal did not close'));

    await goToPrimaryTab(page, 'playground');
    await expectDisabled(page, '#chat-send', true, 'Quick Inference send should start disabled until text is entered');
    await page.type('#chat-input', 'Explain Kiln in one sentence.');
    await expectDisabled(page, '#chat-send', false, 'Quick Inference send should enable after text is entered');
    await page.evaluate(() => { window.__copiedText = ''; });
    await clickAndWait(page, '#chat-send', 'Could not click Quick Inference send');
    await waitForPanelText(page, '#chat-output', /Kiln serves one tuned model and learns from feedback live\./, 'Quick Inference response missing');
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
    await waitForPanelText(page, '#chat-output', /Send a message to test inference\./, 'Quick Inference clear should restore the empty state');
    await expectDisabled(page, '#copy-chat-response', true, 'Copy response should disable after clearing chat');

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
    await clickAndWait(page, '[data-job-id="smoke-eval-full"]', 'Could not open the eval drill modal for the completed compare job');
    await page.waitForFunction(
      () => document.getElementById('eval-drill-modal')?.hidden === false
        && document.getElementById('drill-title')?.textContent === 'smoke-suite',
      { timeout: 5000 },
    ).catch(() => fail('Eval drill modal did not open on the completed compare job'));

    // Raw JSON toggle: first click appends the pretty-printed cached job
    // payload, second click removes it.
    await clickAndWait(page, '#drill-raw', 'Could not click the eval drill raw JSON toggle');
    await page.waitForSelector('#drill-raw-block', { timeout: 5000 })
      .catch(() => fail('Raw JSON toggle did not render #drill-raw-block'));
    const rawPayload = await page.$eval('#drill-raw-block', (el) => el.textContent || '');
    let parsedRaw = null;
    try { parsedRaw = JSON.parse(rawPayload); } catch { fail('Eval drill raw JSON block should contain valid JSON'); }
    if (parsedRaw.job_id !== 'smoke-eval-full') fail(`Raw JSON should show the drilled job, got job_id ${JSON.stringify(parsedRaw.job_id)}`);
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

  } finally {
    await browser.close();
  }
}

const emptyAdapterScenario = await startServer({ availableAdapters: [] });
try {
  console.log('[smoke] empty adapter scenario start');
  await runSmoke(emptyAdapterScenario.baseUrl, {
    expectEmptyAdapters: true,
    setRecentRequests: emptyAdapterScenario.setRecentRequests,
  });
  console.log('[smoke] empty adapter scenario passed');
} finally {
  await new Promise((accept) => emptyAdapterScenario.server.close(accept));
}

const { server, baseUrl, setRecentRequests } = await startServer();
try {
  console.log('[smoke] default scenario desktop start');
  await runSmoke(baseUrl, { setRecentRequests });
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
  });
  console.log('[smoke] failure scenario passed');
} finally {
  await new Promise((accept) => failureScenario.server.close(accept));
}

console.log('server UI smoke check passed');
