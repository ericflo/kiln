#!/usr/bin/env node
/* Capture demo screenshots of the kiln Tauri shell windows.
   Stubs window.__TAURI__ with canned demo data so the dashboard, settings,
   and logs windows render their populated state under headless Chromium. */

import puppeteer from "puppeteer-core";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdir } from "node:fs/promises";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");
const uiDir = resolve(repoRoot, "desktop/ui");
const serverUiHtml = resolve(repoRoot, "crates/kiln-server/src/ui.html");
const serverUiUrl = `file://${serverUiHtml}?demo=1#overview`;

const targets = [
  {
    name: "dashboard",
    file: "dashboard.html",
    out: "docs/desktop/dashboard.png",
    viewport: { width: 1024, height: 768, deviceScaleFactor: 2 },
    settle: 2200,
    iframeServerUi: true,
  },
  {
    name: "settings",
    file: "settings.html",
    out: "docs/desktop/settings.png",
    viewport: { width: 520, height: 760, deviceScaleFactor: 2 },
    settle: 1200,
  },
  {
    name: "logs",
    file: "logs.html",
    out: "docs/desktop/logs.png",
    viewport: { width: 900, height: 600, deviceScaleFactor: 2 },
    settle: 1200,
  },
  {
    name: "about",
    file: "about.html",
    out: "docs/desktop/about.png",
    viewport: { width: 420, height: 380, deviceScaleFactor: 2 },
    settle: 800,
  },
];

const tauriStub = () => {
  const demoSettings = {
    model_path: "/Users/eric/models/Qwen3.5-4B-bf16",
    auto_start: true,
    auto_install_updates: false,
    server_port: 8420,
    log_level: "info",
    binary_path: "/Applications/Kiln Desktop.app/Contents/Resources/kiln",
  };
  const demoBinaryStatus = {
    platform_supported: true,
    has_binary: true,
    installed: true,
    version: "0.6.4",
  };
  const demoServerState = { kind: "TrainingActive", message: null };
  const demoActiveAdapter = {
    active: "qwen35-4b-style-warm",
    available_count: 7,
  };
  const demoTraining = {
    active: {
      job_id: "tr_5f2c1",
      state: "Running",
      progress: 0.46,
      current_loss: 0.812,
      adapter_name: "qwen35-4b-style-warm",
    },
    total_jobs: 3,
  };
  const demoUpdate = {
    status: "up_to_date",
    current_version: "0.2.2",
    latest_version: "0.2.2",
  };
  const demoDiag = {
    version: "0.2.2",
    platform: "macos",
    arch: "aarch64",
    server_binary_version: "0.6.4",
  };
  const demoLogs = [
    "[2026-05-09T16:32:04Z INFO  kiln_server] kiln 0.6.4 booting on 127.0.0.1:8420",
    "[2026-05-09T16:32:04Z INFO  kiln_model] loaded Qwen3.5-4B (bf16) in 1.84s",
    "[2026-05-09T16:32:04Z INFO  kiln_server] OpenAI-compatible API ready",
    "[2026-05-09T16:32:09Z INFO  kiln_train] GRPO job tr_5f2c1 accepted (3 prompts × 4 rollouts)",
    "[2026-05-09T16:32:11Z INFO  kiln_train] step 12/64 loss=0.94 reward=0.31 lr=2.0e-04",
    "[2026-05-09T16:32:14Z INFO  kiln_train] step 18/64 loss=0.88 reward=0.40 lr=2.0e-04",
    "[2026-05-09T16:32:18Z INFO  kiln_train] step 24/64 loss=0.84 reward=0.44 lr=2.0e-04",
    "[2026-05-09T16:32:22Z INFO  kiln_lora] hot-swapped adapter qwen35-4b-style-warm (rank=16)",
    "[2026-05-09T16:32:26Z INFO  kiln_train] step 29/64 loss=0.82 reward=0.49 lr=2.0e-04",
    "[2026-05-09T16:32:31Z INFO  kiln_server] decode 86 tok/s · ttft 38ms · adapter qwen35-4b-style-warm",
  ];
  const demoLogsStderr = [
    "kiln-server: warming caches…",
    "kiln-train: replay buffer 12% full",
  ];

  const handlers = {
    server_state: () => demoServerState,
    get_binary_status: () => demoBinaryStatus,
    get_settings: () => demoSettings,
    default_settings: () => demoSettings,
    get_kiln_url: () => window.__KILN_DEMO_IFRAME_URL || "http://127.0.0.1:8420",
    get_openai_base_url: () => "http://127.0.0.1:8420/v1",
    get_active_adapter: () => demoActiveAdapter,
    get_training_status: () => demoTraining,
    get_app_version: () => "0.2.2",
    get_diagnostic_info: () => demoDiag,
    check_for_updates: () => demoUpdate,
    check_for_kiln_update: () => demoUpdate,
    server_logs: () => demoLogs.concat(demoLogsStderr),
    path_info: () => ({ exists: true, kind: "directory" }),
  };

  const noop = async () => null;

  const invoke = async (cmd, _args) => {
    const fn = handlers[cmd];
    if (fn) return fn();
    return noop();
  };

  const event = {
    listen: async (_evt, _cb) => {
      return () => {};
    },
    emit: async () => null,
    once: async (_evt, _cb) => {
      return () => {};
    },
  };

  const dialog = {
    open: async () => null,
    save: async () => null,
    message: async () => null,
    ask: async () => false,
  };

  const shell = {
    open: async () => null,
  };

  const webviewWindow = {
    getCurrent: () => ({
      show: async () => null,
      hide: async () => null,
      close: async () => null,
      setFocus: async () => null,
      onCloseRequested: async () => () => {},
    }),
    getAllWebviewWindows: async () => [],
  };

  const windowNs = {
    getCurrentWindow: () => ({
      show: async () => null,
      hide: async () => null,
      close: async () => null,
      setFocus: async () => null,
    }),
  };

  const notification = {
    isPermissionGranted: async () => true,
    requestPermission: async () => "granted",
    sendNotification: () => null,
  };

  window.__TAURI__ = {
    core: { invoke },
    invoke,
    event,
    dialog,
    shell,
    webviewWindow,
    window: windowNs,
    notification,
  };
  window.__TAURI_INTERNALS__ = {
    invoke,
    transformCallback: () => 0,
    metadata: { currentWindow: { label: "main" } },
  };
};

const browser = await puppeteer.launch({
  executablePath: "/usr/bin/chromium-browser",
  headless: "new",
  args: ["--no-sandbox", "--disable-setuid-sandbox", "--font-render-hinting=medium"],
  defaultViewport: { width: 1024, height: 768, deviceScaleFactor: 2 },
});

try {
  for (const t of targets) {
    const page = await browser.newPage();
    await page.setViewport(t.viewport);
    await page.evaluateOnNewDocument(tauriStub);
    if (t.iframeServerUi) {
      await page.evaluateOnNewDocument((iframeUrl) => {
        window.__KILN_DEMO_IFRAME_URL = iframeUrl;
      }, serverUiUrl);
    }

    const url = `file://${resolve(uiDir, t.file)}`;
    console.log(`→ ${t.name}: ${url}`);
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    await new Promise((r) => setTimeout(r, t.settle));

    const outPath = resolve(repoRoot, t.out);
    await mkdir(dirname(outPath), { recursive: true });
    await page.screenshot({ path: outPath, fullPage: false });
    console.log(`  saved ${t.out}`);
    await page.close();
  }
} finally {
  await browser.close();
}
