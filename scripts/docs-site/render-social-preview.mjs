#!/usr/bin/env node

import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import puppeteer from 'puppeteer-core';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');
const template = resolve(scriptDir, 'social-preview.html');
const output = resolve(repoRoot, 'docs/site/assets/og-image-v3.png');
const executablePath = process.env.CHROME_BIN
  || process.env.PUPPETEER_EXECUTABLE_PATH
  || ['/usr/bin/google-chrome-stable', '/usr/bin/google-chrome', '/usr/bin/chromium']
    .find((candidate) => existsSync(candidate));

if (!executablePath) {
  throw new Error('Set CHROME_BIN or PUPPETEER_EXECUTABLE_PATH to render the social preview.');
}

const browser = await puppeteer.launch({
  headless: 'shell',
  executablePath,
  args: ['--no-sandbox', '--allow-file-access-from-files'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 1 });
  await page.goto(pathToFileURL(template).href, { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: output, type: 'png', omitBackground: false });
  process.stdout.write(`Rendered ${output}\n`);
} finally {
  await browser.close();
}
