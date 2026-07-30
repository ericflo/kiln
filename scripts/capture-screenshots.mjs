#!/usr/bin/env node
/* Capture demo screenshots of the kiln dashboard with ?demo=1 fixtures.
   Headless Chromium + ImageMagick, file:// URL, no GPU pod required.
   Each PNG source is followed by 720/1440/2880 WebP delivery variants. */

import puppeteer from "./docs-site/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdir, readFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");
const uiHtml = resolve(repoRoot, "crates/kiln-server/src/ui/index.html");
const uiAppDir = resolve(repoRoot, "crates/kiln-server/src/ui/app");
const appFragments = [
  "shell.js",
  "adapters.js",
  "training.js",
  "playground.js",
  "evaluations.js",
  "command_palette.js",
  "charts.js",
  "adapter_drill.js",
  "training_drill.js",
  "playground_compare.js",
  "terminal.js",
  "distillation.js",
  "agents.js",
  "preflight.js",
  "bootstrap.js",
];
const appJs = `(function() {\n'use strict';\n\n${(
  await Promise.all(appFragments.map((name) => readFile(resolve(uiAppDir, name), "utf8")))
).join("\n")}\n})();\n`;
const execFileAsync = promisify(execFile);

async function writeResponsiveWebp(pngPath) {
  const stem = pngPath.replace(/\.png$/i, "");
  const common = ["-strip", "-quality", "82", "-define", "webp:method=6"];
  await execFileAsync("magick", [pngPath, ...common, `${stem}.webp`]);
  await execFileAsync("magick", [pngPath, "-filter", "Lanczos", "-resize", "1440x>", ...common, `${stem}-1440.webp`]);
  await execFileAsync("magick", [pngPath, "-filter", "Lanczos", "-resize", "720x>", ...common, `${stem}-720.webp`]);
}

const targets = [
  { page: "overview",   out: "docs/site/assets/server-ui-dashboard.png", scrollWait: 0 },
  { page: "adapters",   out: "docs/site/assets/server-ui-adapters.png",  scrollWait: 0 },
  { page: "training",   out: "docs/site/assets/server-ui-training.png",  scrollWait: 0 },
  { page: "playground", out: "docs/site/assets/server-ui-playground.png", scrollWait: 0 },
];

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_BIN || "/usr/bin/chromium-browser",
  headless: "new",
  args: ["--no-sandbox", "--disable-setuid-sandbox", "--font-render-hinting=medium"],
  defaultViewport: { width: 1440, height: 900, deviceScaleFactor: 2 },
});

try {
  for (const t of targets) {
    const page = await browser.newPage();
    await page.setRequestInterception(true);
    page.on("request", async (request) => {
      if (new URL(request.url()).pathname.endsWith("/ui/app.js")) {
        await request.respond({
          status: 200,
          contentType: "application/javascript",
          body: appJs,
        });
        return;
      }
      await request.continue();
    });
    const url = `file://${uiHtml}?demo=1#${t.page}`;
    console.log(`→ ${t.page}: ${url}`);
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    // Give polling a chance to populate from the demo fixture
    await new Promise(r => setTimeout(r, 1500));

    const outPath = resolve(repoRoot, t.out);
    await mkdir(dirname(outPath), { recursive: true });
    await page.screenshot({ path: outPath, fullPage: true });
    await writeResponsiveWebp(outPath);
    console.log(`  saved ${t.out} and responsive WebP variants`);
    await page.close();
  }
} finally {
  await browser.close();
}
