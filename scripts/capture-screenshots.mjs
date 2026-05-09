#!/usr/bin/env node
/* Capture demo screenshots of the kiln dashboard with ?demo=1 fixtures.
   Headless Chromium, file:// URL, no GPU pod required. */

import puppeteer from "puppeteer-core";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdir } from "node:fs/promises";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");
const uiHtml = resolve(repoRoot, "crates/kiln-server/src/ui.html");

const targets = [
  { page: "overview",   out: "docs/site/assets/server-ui-dashboard.png", scrollWait: 0 },
  { page: "adapters",   out: "docs/site/assets/server-ui-adapters.png",  scrollWait: 0 },
  { page: "training",   out: "docs/site/assets/server-ui-training.png",  scrollWait: 0 },
  { page: "playground", out: "docs/site/assets/server-ui-playground.png", scrollWait: 0 },
];

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium-browser",
  headless: "new",
  args: ["--no-sandbox", "--disable-setuid-sandbox", "--font-render-hinting=medium"],
  defaultViewport: { width: 1440, height: 900, deviceScaleFactor: 2 },
});

try {
  for (const t of targets) {
    const page = await browser.newPage();
    const url = `file://${uiHtml}?demo=1#${t.page}`;
    console.log(`→ ${t.page}: ${url}`);
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    // Give polling a chance to populate from the demo fixture
    await new Promise(r => setTimeout(r, 1500));

    const outPath = resolve(repoRoot, t.out);
    await mkdir(dirname(outPath), { recursive: true });
    await page.screenshot({ path: outPath, fullPage: true });
    console.log(`  saved ${t.out}`);
    await page.close();
  }
} finally {
  await browser.close();
}
