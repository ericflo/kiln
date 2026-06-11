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
  const match = /^\/v1\/adapters\/([^/]+)(?:\/(download))?$/.exec(pathname);
  if (!match) return null;
  if (['load', 'unload', 'upload', 'merge'].includes(match[1])) return null;
  return { name: decodeURIComponent(match[1]), action: match[2] || null };
}

const defaultAvailableAdapters = [
  { name: 'adapter-alpha', active: false, size_bytes: 4096 },
  { name: 'adapter-beta', active: false, size_bytes: 8192 },
];

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

async function startServer({ failDashboardApis = false, availableAdapters = defaultAvailableAdapters } = {}) {
  // Mutable so the failure scenario can heal/re-break the APIs mid-run and
  // assert the dashboard recovers (Retry buttons + dedupe-key invalidation).
  const apiState = { failDashboardApis };
  const uiHtml = await readFile(uiIndexPath, 'utf8');
  const uiStyles = await readFile(uiStylesPath, 'utf8');
  const uiDemoJs = await readFile(uiDemoJsPath, 'utf8');
  const uiAppJs = await readFile(uiAppJsPath, 'utf8');
  availableAdapters = availableAdapters.map((adapter) => ({ ...adapter }));
  let activeAdapter = availableAdapters.find((adapter) => adapter.active)?.name || null;
  const completedTrainingJobs = [];
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
      json(res, {
        status: 'ok',
        model: 'Qwen3.5-4B',
        backend: 'mock',
        uptime_seconds: 42,
        active_adapter: activeAdapter,
        scheduler: { waiting: 0, running: 0, blocks_used: 0, blocks_free: 1024 },
        gpu_memory: { total_vram_gb: 24, model_gb: 8, kv_cache_gb: 2, training_budget_gb: 4 },
        checks: [{ name: 'mock smoke server', pass: true }],
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
      json(res, { running: null, queued: [], completed: completedTrainingJobs });
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
      completedTrainingJobs.unshift({
        job_id: 'smoke-sft',
        job_type: 'sft',
        state: 'Completed',
        progress: 1,
        adapter_name: body.config.output_name,
        elapsed_secs: 1,
      });
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
      json(res, { object: 'list', data: [{ id: 'Qwen3.5-4B', object: 'model', owned_by: 'kiln' }] });
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
      json(res, []);
      return;
    }
    // The UI polls /v1/eval/jobs at startup to keep the Evals badge
    // accurate before the user has visited the tab. Stub an empty list
    // so the empty-adapter smoke run doesn't see a 404 in the console.
    if (url.pathname === '/v1/eval/jobs') {
      json(res, { jobs: [] });
      return;
    }
    // Same shape as the production endpoints — only the empty cases
    // are exercised in smoke, but keep them registered so future UI
    // background polls don't surprise the mock.
    if (url.pathname === '/v1/eval/suites') {
      json(res, { suites: [] });
      return;
    }
    if (url.pathname === '/v1/eval/datasets') {
      json(res, { datasets: [] });
      return;
    }
    if (url.pathname === '/v1/judgments') {
      json(res, { judgments: [] });
      return;
    }
    if (url.pathname === '/v1/chat/completions') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ detail: 'Use POST for chat completions' }));
        return;
      }
      const body = await readJsonBody(req);
      const prompt = body?.messages?.findLast((message) => message.role === 'user')?.content || '';
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

async function runSmoke(baseUrl, { expectFailureStates = false, expectEmptyAdapters = false, setFailDashboardApis = null } = {}) {
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
      if (pageErrors.length > 0) fail(`Empty adapter UI emitted browser errors: ${pageErrors.join('; ')}`);
      return;
    }

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

    // Regression (the "pie chart disappears" bug): the VRAM donut and header
    // stats must survive subsequent 2s health polls. The donut used to render
    // once, then renderServerStatus clobbered the panel while the donut
    // refresher's dedupe key skipped re-appending it — gone until reload.
    await page.waitForSelector('#server-status .vram-donut svg', { timeout: 5000 })
      .catch(() => fail('VRAM donut should render in the server status panel'));
    await new Promise((resolve) => setTimeout(resolve, 5200)); // sit through ≥2 health polls
    const overviewSteadyState = await page.evaluate(() => ({
      donuts: document.querySelectorAll('#server-status .vram-donut').length,
      svg: Boolean(document.querySelector('#server-status .vram-donut svg')),
      model: document.getElementById('header-model')?.textContent || '',
      uptime: document.getElementById('header-uptime')?.textContent || '',
    }));
    if (!overviewSteadyState.svg) fail('VRAM donut disappeared after subsequent health polls');
    if (overviewSteadyState.donuts !== 1) fail(`Expected exactly one VRAM donut, found ${overviewSteadyState.donuts}`);
    if (overviewSteadyState.model !== 'Qwen3.5-4B') fail(`Header model stat should render from /health, got "${overviewSteadyState.model}"`);
    if (!/^\d+[sm]/.test(overviewSteadyState.uptime)) fail(`Header uptime stat should render, got "${overviewSteadyState.uptime}"`);

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

  } finally {
    await browser.close();
  }
}

const emptyAdapterScenario = await startServer({ availableAdapters: [] });
try {
  console.log('[smoke] empty adapter scenario start');
  await runSmoke(emptyAdapterScenario.baseUrl, { expectEmptyAdapters: true });
  console.log('[smoke] empty adapter scenario passed');
} finally {
  await new Promise((accept) => emptyAdapterScenario.server.close(accept));
}

const { server, baseUrl } = await startServer();
try {
  console.log('[smoke] default scenario desktop start');
  await runSmoke(baseUrl);
  console.log('[smoke] default scenario desktop passed; mobile start');
  await runMobileOnboardingSmoke(baseUrl);
  console.log('[smoke] default scenario mobile passed');
} finally {
  await new Promise((accept) => server.close(accept));
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
