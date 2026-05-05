#!/usr/bin/env node
import http from 'node:http';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import process from 'node:process';

const repoRoot = resolve(import.meta.dirname, '..');
const uiPath = resolve(repoRoot, 'crates/kiln-server/src/ui.html');

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
  res.writeHead(503, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify({
    detail: `${panelName} smoke failure from ${path}`,
  }));
}

async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

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

async function startServer({ failDashboardApis = false } = {}) {
  const uiHtml = await readFile(uiPath, 'utf8');
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', 'http://127.0.0.1');
    if (url.pathname === '/' || url.pathname === '/ui') {
      text(res, uiHtml, 'text/html; charset=utf-8');
      return;
    }
    if (url.pathname === '/favicon.ico') {
      res.writeHead(204);
      res.end();
      return;
    }
    if (failDashboardApis) {
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
    }
    if (url.pathname === '/health') {
      json(res, {
        status: 'ok',
        model: 'Qwen3.5-4B',
        backend: 'mock',
        uptime_seconds: 42,
        active_adapter: null,
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
    if (url.pathname === '/v1/adapters') {
      json(res, { active: null, available: [] });
      return;
    }
    if (url.pathname === '/v1/train/queue' || url.pathname === '/v1/train/status') {
      json(res, { running: null, queued: [], completed: [] });
      return;
    }
    if (url.pathname === '/v1/models') {
      json(res, { object: 'list', data: [{ id: 'qwen3.5-4b', object: 'model', owned_by: 'kiln' }] });
      return;
    }
    if (url.pathname === '/v1/stats/decode') {
      json(res, { window_secs: 60, sample_count: 0, tok_per_sec: 0, p50_itl_ms: 0, p99_itl_ms: 0, mean_itl_ms: 0 });
      return;
    }
    if (url.pathname === '/v1/stats/recent-requests') {
      json(res, []);
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
  return { server, baseUrl: `http://127.0.0.1:${address.port}` };
}

async function expectText(page, selector, pattern, message) {
  const handle = await page.$(selector);
  if (!handle) fail(`${message}: missing selector ${selector}`);
  const textContent = await page.evaluate((el) => el.textContent || '', handle);
  if (!pattern.test(textContent)) fail(`${message}: selector ${selector} text was ${JSON.stringify(textContent.trim())}`);
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
  const handle = await page.waitForSelector(selector, { visible: true, timeout: 5000 });
  if (!handle) fail(`${message}: missing selector ${selector}`);
  await page.evaluate((element) => element.click(), handle);
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

async function runSmoke(baseUrl, { expectFailureStates = false } = {}) {
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
    await expectText(page, 'nav.header-help', /Quickstart/, 'Header Quickstart link missing');
    await expectText(page, 'nav.header-help', /GRPO Guide/, 'Header GRPO link missing');
    await expectText(page, 'nav.header-help', /API Reference/, 'Header docs link missing');

    const helpLinks = await page.$$eval('nav.header-help a', (links) => links.map((link) => ({ text: link.textContent?.trim(), href: link.getAttribute('href') })));
    for (const expected of [
      ['Quickstart', 'https://ericflo.github.io/kiln/quickstart.html'],
      ['GRPO Guide', 'https://ericflo.github.io/kiln/grpo.html'],
    ]) {
      const [label, href] = expected;
      if (!helpLinks.some((link) => link.text === label && link.href === href)) {
        fail(`Header help link missing expected ${label} -> ${href}`);
      }
    }

    if (expectFailureStates) {
      await expectApiFailurePanel(page, '#server-status', 'Server status', 'Server status smoke failure from /health');
      await expectApiFailurePanel(page, '#decode-perf-panel', 'Decode performance', 'Decode performance smoke failure from /v1/stats/decode');
      await expectApiFailurePanel(page, '#recent-requests-panel', 'Recent requests', 'Recent requests smoke failure from /v1/stats/recent-requests');
      await expectApiFailurePanel(page, '#adapters-panel', 'Adapters', 'Adapters smoke failure from /v1/adapters');
      await expectApiFailurePanel(page, '#tab-queue', 'Training queue', 'Training queue smoke failure from /v1/train/queue');
      if (pageErrors.length > 0) fail(`Failure state UI emitted browser errors: ${pageErrors.join('; ')}`);
      return;
    }

    await waitForPanelText(page, '#adapters-panel', /No adapters found yet\./, 'Empty adapter state missing');
    await expectPanelLink(page, '#adapters-panel .empty', 'Quickstart', 'https://ericflo.github.io/kiln/quickstart.html');
    await expectPanelLink(page, '#adapters-panel .empty', 'Troubleshooting', 'https://ericflo.github.io/kiln/troubleshooting.html');

    await waitForPanelText(page, '#tab-queue', /No training jobs yet\./, 'Empty training queue state missing');
    await expectPanelLink(page, '#tab-queue .empty', 'Quickstart', 'https://ericflo.github.io/kiln/quickstart.html');
    await expectPanelLink(page, '#tab-queue .empty', 'GRPO Guide', 'https://ericflo.github.io/kiln/grpo.html');

    await waitForPanelText(page, '#recent-requests-panel', /No recent requests yet\./, 'Empty recent requests state missing');
    await expectPanelLink(page, '#recent-requests-panel .empty', 'Quickstart', 'https://ericflo.github.io/kiln/quickstart.html');

    await waitForPanelText(page, '#decode-perf-panel', /No streaming completions/i, 'Empty decode performance state missing');
    await expectPanelLink(page, '#decode-perf-panel', '/health', '/health');

    await waitForPanelText(page, '#chat-output', /Send a message to test inference\./, 'Quick Inference empty state missing');
    await expectPanelLink(page, '#chat-output .empty', '/health', '/health');
    await expectPanelLink(page, '#chat-output .empty', 'Troubleshooting guide', 'https://ericflo.github.io/kiln/troubleshooting.html');

    await clickAndWait(page, '#training-tab-sft', 'Could not open SFT tab');
    await waitForVisiblePanel(page, '#tab-sft', 'SFT tab did not activate');
    await expectDisabled(page, '#sft-form button[type="submit"]', true, 'SFT submit should start disabled until examples are provided');
    await clickAndWait(page, '#use-sft-sample', 'Could not click SFT sample payload button');
    await expectDisabled(page, '#sft-form button[type="submit"]', false, 'SFT submit should enable after sample payload is clicked');

    await clickAndWait(page, '#training-tab-grpo', 'Could not open GRPO tab');
    await waitForVisiblePanel(page, '#tab-grpo', 'GRPO tab did not activate');
    await expectDisabled(page, '#grpo-form button[type="submit"]', true, 'GRPO submit should start disabled until groups are provided');
    await clickAndWait(page, '#use-grpo-sample', 'Could not click GRPO sample payload button');
    await expectDisabled(page, '#grpo-form button[type="submit"]', false, 'GRPO submit should enable after sample payload is clicked');

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

    await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1, isMobile: true });
    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      bodyScrollWidth: document.body.scrollWidth,
      bodyClientWidth: document.body.clientWidth,
    }));
    if (overflow.scrollWidth > overflow.clientWidth + 1 || overflow.bodyScrollWidth > overflow.bodyClientWidth + 1) {
      fail(`Mobile viewport has horizontal overflow at 390x844: document ${overflow.scrollWidth}/${overflow.clientWidth}, body ${overflow.bodyScrollWidth}/${overflow.bodyClientWidth}`);
    }
  } finally {
    await browser.close();
  }
}

const { server, baseUrl } = await startServer();
try {
  await runSmoke(baseUrl);
} finally {
  await new Promise((accept) => server.close(accept));
}

const failureScenario = await startServer({ failDashboardApis: true });
try {
  await runSmoke(failureScenario.baseUrl, { expectFailureStates: true });
} finally {
  await new Promise((accept) => failureScenario.server.close(accept));
}

console.log('server UI smoke check passed');
