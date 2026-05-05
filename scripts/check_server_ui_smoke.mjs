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

async function startServer() {
  const uiHtml = await readFile(uiPath, 'utf8');
  const server = http.createServer((req, res) => {
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
  const actual = await page.$eval(selector, (el) => Boolean(el.disabled));
  if (actual !== expected) fail(`${message}: expected ${selector} disabled=${expected}, got ${actual}`);
}

async function clickAndWait(page, selector, message) {
  const handle = await page.$(selector);
  if (!handle) fail(`${message}: missing selector ${selector}`);
  await handle.click();
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

async function runSmoke(baseUrl) {
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

    await waitForPanelText(page, '#adapters-panel', /No adapters found yet\./, 'Empty adapter state missing');
    await waitForPanelText(page, '#tab-queue', /No training jobs yet\./, 'Empty training queue state missing');
    await waitForPanelText(page, '#recent-requests-panel', /No recent requests yet\./, 'Empty recent requests state missing');
    await waitForPanelText(page, '#decode-perf-panel', /No streaming completions/i, 'Empty decode performance state missing');

    await clickAndWait(page, '#training-tab-sft', 'Could not open SFT tab');
    await expectDisabled(page, '#sft-form button[type="submit"]', true, 'SFT submit should start disabled until examples are provided');
    await clickAndWait(page, '#use-sft-sample', 'Could not click SFT sample payload button');
    await expectDisabled(page, '#sft-form button[type="submit"]', false, 'SFT submit should enable after sample payload is clicked');

    await clickAndWait(page, '#training-tab-grpo', 'Could not open GRPO tab');
    await expectDisabled(page, '#grpo-form button[type="submit"]', true, 'GRPO submit should start disabled until groups are provided');
    await clickAndWait(page, '#use-grpo-sample', 'Could not click GRPO sample payload button');
    await expectDisabled(page, '#grpo-form button[type="submit"]', false, 'GRPO submit should enable after sample payload is clicked');

    await expectDisabled(page, '#chat-send', true, 'Quick Inference send should start disabled until text is entered');
    await page.type('#chat-input', 'Explain Kiln in one sentence.');
    await expectDisabled(page, '#chat-send', false, 'Quick Inference send should enable after text is entered');

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
  console.log('server UI smoke check passed');
} finally {
  await new Promise((accept) => server.close(accept));
}
