#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, extname, relative, sep, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import process from 'node:process';

const repoRoot = resolve(import.meta.dirname, '..');
const mobileViewport = { width: 390, height: 844, deviceScaleFactor: 2, isMobile: true };
const mobileOverflowTolerancePx = 2;

const pages = [
  { label: 'Home', path: 'docs/site/index.html', currentLabel: null },
  { label: 'Quickstart', path: 'docs/site/quickstart.html', currentLabel: 'Quickstart' },
  { label: 'GRPO Guide', path: 'docs/site/grpo.html', currentLabel: 'GRPO Guide' },
  { label: 'API Reference', path: 'docs/site/api.html', currentLabel: 'API Reference' },
  { label: 'CLI Reference', path: 'docs/site/cli.html', currentLabel: 'CLI Reference' },
  { label: 'Troubleshooting', path: 'docs/site/troubleshooting.html', currentLabel: 'Troubleshooting' },
  { label: 'Architecture', path: 'docs/site/architecture.html', currentLabel: 'Architecture' },
  { label: 'Demo', path: 'docs/site/demo/index.html', currentLabel: 'Demo' },
];

const expectedNavLabels = [
  'Quickstart',
  'GRPO Guide',
  'API Reference',
  'CLI Reference',
  'Demo',
  'Troubleshooting',
  'Architecture',
];

const expectedFooterLinks = [
  { label: 'Quickstart', localPath: 'docs/site/quickstart.html' },
  { label: 'GRPO Guide', localPath: 'docs/site/grpo.html' },
  { label: 'API Reference', localPath: 'docs/site/api.html' },
  { label: 'CLI Reference', localPath: 'docs/site/cli.html' },
  { label: 'Demo', localPath: 'docs/site/demo/' },
  { label: 'Troubleshooting', localPath: 'docs/site/troubleshooting.html' },
  { label: 'Architecture', localPath: 'docs/site/architecture.html' },
  { label: 'Changelog', href: 'https://github.com/ericflo/kiln/blob/main/CHANGELOG.md' },
  { label: 'License', href: 'https://github.com/ericflo/kiln/blob/main/LICENSE' },
];

const expectedEmbeddedUiHelpLinks = [
  { label: 'Quickstart', href: 'https://ericflo.github.io/kiln/quickstart.html' },
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

const demoPagePath = 'docs/site/demo/index.html';
const quickstartPagePath = 'docs/site/quickstart.html';
const apiPagePath = 'docs/site/api.html';
const cliPagePath = 'docs/site/cli.html';
const architecturePagePath = 'docs/site/architecture.html';
const troubleshootingPagePath = 'docs/site/troubleshooting.html';

const expectedQuickstartSections = [
  { label: 'Desktop App path', terms: ['desktop app', 'recommended'] },
  { label: 'server binary path', terms: ['server binary', 'terminal-first'] },
  { label: 'Docker path', terms: ['docker', 'nvidia container toolkit'] },
  { label: 'prerequisites', terms: ['prerequisites'] },
  { label: 'start server', terms: ['run the server', 'kiln serve'] },
  { label: 'test inference', terms: ['send chat', '/v1/chat/completions'] },
  { label: 'open UI', terms: ['open the ui', '/ui'] },
  { label: 'first inference checkpoint', terms: ['first inference checkpoint'] },
  { label: 'SFT next step', terms: ['sft corrections', '/v1/train/sft'] },
  { label: 'GRPO next step', terms: ['grpo guide', 'generate', 'score', 'train'] },
  { label: 'training payload shapes', terms: ['sft jsonl', 'one chat correction per line', 'messages array', 'grpo json request/batch', 'groups', 'candidate completions', 'reward scores', 'kiln train sft', 'kiln train grpo'] },
  { label: 'Where to go next', terms: ['where to go next'] },
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
  { label: 'first token', terms: ['first token'] },
  { label: 'benchmark', terms: ['benchmark'] },
  { label: 'hot-swap', terms: ['hot-swap'] },
  { label: 'OpenAI client', terms: ['openai client'] },
  { label: 'GRPO', terms: ['grpo'] },
  { label: '60-second loop', terms: ['60-second', 'loop'] },
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
  '/v1/chat/completions',
  '/v1/completions/batch',
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

const staleTrainingJobEndpoint = '/v1/train/jobs/{job_id}';
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
];

const expectedApiSections = [
  { label: 'server status', terms: ['server status'] },
  { label: 'copy-paste first requests', terms: ['copy-paste first requests'] },
  { label: 'power-user requests', terms: ['power-user requests'] },
  { label: 'OpenAI-compatible generation', terms: ['openai-compatible generation'] },
  { label: 'adapter lifecycle', terms: ['lora lifecycle'] },
  { label: 'training', terms: ['training'] },
  { label: 'training data safety', terms: ['training data changes', 'adapter'] },
  { label: 'response shapes', terms: ['response shapes'] },
];

const expectedApiCodeExamples = [
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
  { label: 'command chooser', terms: ['if you want to'] },
  { label: 'serve/start-server path', terms: ['start serving qwen3.5-4b', 'kiln_model_path', 'kiln serve'] },
  { label: 'no-subcommand serve path', terms: ['running kiln with no subcommand starts the server'] },
  { label: 'health/readiness path', terms: ['check server readiness', 'kiln health'] },
  { label: 'SFT/GRPO training path', terms: ['submit sft and grpo jobs', 'kiln train sft', 'kiln train grpo'] },
  { label: 'SFT payload shape', terms: ['sft reads jsonl', 'one chat correction example per line', 'messages array'] },
  { label: 'GRPO payload shape', terms: ['grpo reads one json request/batch', 'groups', 'messages', 'candidate completions', 'text', 'reward scores'] },
  { label: 'adapter lifecycle path', terms: ['manage lora adapters', 'kiln adapters list', 'kiln adapters load', 'kiln adapters unload'] },
  { label: 'config validation path', terms: ['validate config', 'kiln config --file'] },
  { label: 'help and verbosity flags', terms: ['--help', '--verbose', '--quiet', '-vv'] },
  { label: 'UI handoff', terms: ['http://127.0.0.1:8420/ui', '/ui'] },
  { label: 'related docs', terms: ['related docs', 'quickstart', 'api reference', 'grpo guide', 'troubleshooting', 'architecture'] },
];

const expectedCliCodeExamples = [
  { label: 'serve command', terms: ['kiln_model_path=./qwen3.5-4b', 'kiln serve'] },
  { label: 'health commands', terms: ['kiln health', 'kiln health --json'] },
  { label: 'SFT training command', terms: ['kiln train sft', '--file corrections.jsonl', '--adapter support-bot'] },
  { label: 'GRPO training command', terms: ['kiln train grpo', '--file grpo-batch.json', '--adapter support-bot'] },
  { label: 'training status command', terms: ['kiln train status'] },
  { label: 'adapter commands', terms: ['kiln adapters list', 'kiln adapters load support-bot', 'kiln adapters unload'] },
  { label: 'config validation commands', terms: ['kiln config --file kiln.toml', 'kiln serve --config kiln.toml'] },
  { label: 'verbosity commands', terms: ['kiln -v serve', 'kiln -vv serve', 'kiln -q health'] },
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
  { label: 'Submit training', href: '#training' },
  { label: 'Manage adapters', href: '#adapters' },
];

const expectedCliPageFragments = [
  '#serve',
  '#health',
  '#training',
  '#adapters',
  '#where-to-go-next',
];

const expectedCliModelSetupCue = {
  label: 'Qwen/Qwen3.5-4B setup cue',
  terms: ['qwen/qwen3.5-4b', 'quickstart', 'setup', 'model download'],
  href: 'quickstart.html',
};

const expectedArchitectureSections = [
  { label: 'single-process server', terms: ['single process', 'rust binary', 'axum http api'] },
  { label: 'request path and batching', terms: ['request path and batching', 'iteration-level scheduler', 'continuous batching'] },
  { label: 'Gated DeltaNet/GDN hybrid', terms: ['gated deltanet', 'gdn', 'hybrid'] },
  { label: 'paged KV/block manager', terms: ['paged kv', 'block manager'] },
  { label: 'Qwen3.5-4B', terms: ['qwen3.5-4b'] },
  { label: 'LoRA hot-swap', terms: ['lora hot-swap', 'iteration boundary'] },
  { label: 'training queue', terms: ['training queue', 'fifo background queue'] },
  { label: 'GPU backend crates', terms: ['gpu backend crates', 'kiln-flash-attn', 'kiln-vulkan-kernel'] },
  { label: 'where-to-go-next links', terms: ['where to go next', 'deep dive', 'grpo guide', 'quickstart', 'troubleshooting'] },
];

const expectedArchitectureFlowTerms = [
  'http/api',
  'scheduler',
  'block manager',
  'qwen/qwen3.5-4b engine',
  'lora training queue',
  'hot-swapped adapter',
];

const expectedArchitectureLinks = [
  { label: 'full ARCHITECTURE.md', href: 'https://github.com/ericflo/kiln/blob/main/ARCHITECTURE.md' },
  { label: 'quickstart', href: 'quickstart.html' },
  { label: 'troubleshooting', href: 'troubleshooting.html' },
  { label: 'API reference', href: 'api.html' },
  { label: 'GRPO guide', href: 'grpo.html' },
];

const expectedTroubleshootingSections = [
  { label: 'first-run diagnostic framing', terms: ['first-run', 'diagnostic'] },
  { label: 'three probes', terms: ['start with three probes'] },
  { label: 'Desktop App first launch', terms: ['desktop app first launch'] },
  { label: 'wrong binary/GPU path', terms: ['wrong binary or gpu path'] },
  { label: 'model weights not found', terms: ['model weights are not found'] },
  { label: 'health not green', terms: ['/health', 'not green'] },
  { label: 'remote server not reachable', terms: ['remote server is not reachable'] },
  { label: 'long-prefill/tool-call timeouts', terms: ['long-prefill', 'tool-call', 'timeouts'] },
  { label: 'mock mode', terms: ['mock mode is not real training'] },
  { label: 'adapter directory', terms: ['adapters are in a different directory'] },
];

const expectedTroubleshootingProbeExamples = [
  { label: 'health probe', terms: ['/health'] },
  { label: 'models probe', terms: ['/v1/models'] },
  { label: 'minimal chat probe', terms: ['/v1/chat/completions', 'messages', 'max_tokens'] },
];

const expectedTroubleshootingLinks = [
  { label: 'Quickstart', href: 'quickstart.html' },
  { label: 'GRPO Guide', href: 'grpo.html' },
  { label: 'Architecture', href: 'architecture.html' },
  { label: 'API Reference', href: 'api.html' },
  { label: 'CLI Reference', href: 'cli.html' },
];

function fail(message) {
  throw new Error(message);
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
    ['what-it-is paragraph', ['single-GPU inference server', 'live LoRA training', 'same process', 'same GPU']],
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
  const uiPath = resolve(repoRoot, 'crates/kiln-server/src/ui.html');
  const ui = readFileSync(uiPath, 'utf8');
  const navMatch = ui.match(/<nav class="header-help" aria-label="Help links">[\s\S]*?<\/nav>/);
  if (!navMatch) {
    fail('crates/kiln-server/src/ui.html: missing embedded UI help nav');
  }

  const nav = navMatch[0];
  const missingLinks = expectedEmbeddedUiHelpLinks.filter(({ label, href }) => {
    const escapedHref = href.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const linkPattern = new RegExp(`<a\\s+href="${escapedHref}"[^>]*>${label}<\\/a>`);
    return !linkPattern.test(nav);
  });
  if (missingLinks.length > 0) {
    fail(`crates/kiln-server/src/ui.html: embedded UI help nav missing links: ${missingLinks.map(({ label }) => label).join(', ')}`);
  }
}

async function validateEmbeddedUiControlAccessibleNames(browser) {
  const uiPath = resolve(repoRoot, 'crates/kiln-server/src/ui.html');
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
      fail(`crates/kiln-server/src/ui.html: missing embedded UI controls: ${missingControls.join(', ')}`);
    }

    const unnamedControls = result
      .filter((control) => control.missingLabelTerms?.length > 0)
      .map((control) => `${control.selector} missing label terms ${control.missingLabelTerms.join(', ')}`);
    if (unnamedControls.length > 0) {
      fail(`crates/kiln-server/src/ui.html: embedded UI controls need accessible names: ${unnamedControls.join('; ')}`);
    }

    const undescribedControls = result
      .filter((control) => control.missingDescriptionTerms?.length > 0)
      .map((control) => `${control.selector} missing description terms ${control.missingDescriptionTerms.join(', ')}`);
    if (undescribedControls.length > 0) {
      fail(`crates/kiln-server/src/ui.html: embedded UI controls need aria-describedby help text: ${undescribedControls.join('; ')}`);
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
  const indexSection = index.match(/<!-- the GRPO loop -->[\s\S]*?<\/section>/);
  if (!indexSection) {
    fail('docs/site/index.html: missing The GRPO loop section');
  }
  assertRequestsImportNearPost(indexSection[0], 'docs/site/index.html: The GRPO loop');
  assertOpenAIClientSetupNearChatCreate(indexSection[0], 'docs/site/index.html: The GRPO loop');
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
  const desktopInstallMatch = index.match(/<h3[^>]*>Desktop App[\s\S]*?<\/h3>([\s\S]*?)<table class="installer-table/);
  if (!desktopInstallMatch) {
    fail('docs/site/index.html: missing Desktop App install copy');
  }

  const desktopInstallCopy = desktopInstallMatch[1];
  const requiredTerms = [
    'separate GitHub release tags/version numbers',
    'desktop-v0.2.15',
    'latest <code>kiln-v*</code>',
  ];
  for (const term of requiredTerms) {
    assertIncludes(desktopInstallCopy, term, 'docs/site/index.html: Desktop App version-split cue');
  }
}

function validateQuickstartDesktopVersionSplitCue() {
  const quickstart = readFileSync(resolve(repoRoot, 'docs/site/quickstart.html'), 'utf8');
  const desktopInstallMatch = quickstart.match(/<h3[^>]*>Desktop App &middot; recommended<\/h3>([\s\S]*?)<\/div>\s*<div class="install-card">/);
  if (!desktopInstallMatch) {
    fail('docs/site/quickstart.html: missing Desktop App install card copy');
  }

  const desktopInstallCopy = desktopInstallMatch[1];
  const requiredTerms = [
    'desktop-v0.2.15',
    'Desktop',
    'server',
    'release',
    'kiln-v*',
    'downloads',
    'verifies',
  ];
  for (const term of requiredTerms) {
    assertIncludes(desktopInstallCopy, term, 'docs/site/quickstart.html: Desktop App version-split cue');
  }
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
      'kiln adapters list',
    ]],
    ['TOP_LEVEL_EXAMPLES', [
      'kiln serve',
      'kiln health',
      'kiln train sft --file examples.jsonl --adapter my-task',
      'kiln train grpo --file grpo-batch.json --adapter my-task',
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
    /Path to one GRPO JSON request\/batch with groups, prompt messages, candidate completions, text, and reward scores/,
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
    .replace(/&apos;/g, "'");
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
  const docsSiteRoot = resolve(repoRoot, 'docs/site');
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

  visit(docsSiteRoot);
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
    ? resolve(repoRoot, `.${hrefPath}`)
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
  const siteRelativePath = localPath.replace(/^docs\/site\//, '');
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

async function loadPuppeteer() {
  try {
    const module = await import('puppeteer');
    return module.default || module;
  } catch (error) {
    if (error?.code !== 'ERR_MODULE_NOT_FOUND') throw error;
  }

  const installDir = '/tmp/kiln-docs-site-smoke-puppeteer';
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

function formatLikelyOverflowingElements(elements) {
  if (!elements || elements.length === 0) return 'none found';
  return elements
    .map((element) => `${element.selector} rect=[${element.left},${element.right}] scrollWidth=${element.scrollWidth} clientWidth=${element.clientWidth} text="${element.text}"`)
    .join('; ');
}

async function runSmoke() {
  validateReadmeStartupBanner();
  validateReadmeMedia();
  validateReadmeColdReaderCoverage();
  validateReadmeImageReferences();
  validateReadmeQuickStartPaths();
  validateAdapterListStaleWordingSurfaces();
  validateReadmeAdapterListSemantics();
  validateGrpoOverviewRequestsImports();
  validateGrpoDemoPayloadCue();
  validateLandingDesktopVersionSplitCue();
  validateQuickstartDesktopVersionSplitCue();
  validateQuickstartMarkdownMedia();
  validateQuickstartServerBinaryPath();
  validateQuickstartCliReference();
  validateEmbeddedUiHelpLinks();
  validateCliHelpOnboardingCopy();
  validateLaunchSentinel();
  validateDemoReadmeInventory();
  validateDocsSiteCanonicalLinks();
  validateDocsSiteLocalLinks();
  validateMarkdownLocalLinks();

  const puppeteer = await loadPuppeteer();
  const browser = await puppeteer.launch({
    executablePath: chromiumPath(),
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    await validateEmbeddedUiControlAccessibleNames(browser);

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
      if (!result.currentMatches) {
        const expected = sitePage.currentLabel || 'an aria-current="page" marker';
        fail(`${sitePage.path}: expected current marker for ${expected}, got ${result.currentLabel || 'none'}`);
      }
      if (result.scrollWidth > result.clientWidth + mobileOverflowTolerancePx) {
        fail(`${sitePage.path}: mobile horizontal overflow at ${mobileViewport.width}x${mobileViewport.height}: scrollWidth ${result.scrollWidth} > clientWidth ${result.clientWidth} + tolerance ${mobileOverflowTolerancePx}; likely overflowing elements: ${formatLikelyOverflowingElements(result.likelyOverflowingElements)}`);
      }

      if (sitePage.path === demoPagePath) {
        const demoResult = await page.evaluate(() => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
          const scriptText = Array.from(document.querySelectorAll('script'))
            .map((script) => script.textContent || '')
            .join('\n');
          const referencedCasts = Array.from(scriptText.matchAll(/cast:\s*["']([^"']+\.cast)["']/g), (match) => match[1]);

          return {
            bodyText: normalize(document.body.innerText),
            referencedCasts,
          };
        });

        const missingSections = expectedDemoSections
          .filter((section) => !section.terms.every((term) => demoResult.bodyText.includes(term)))
          .map((section) => section.label);
        if (missingSections.length > 0) {
          fail(`${sitePage.path}: missing demo sections: ${missingSections.join(', ')}`);
        }

        validateDemoCasts(sitePage.path, demoResult.referencedCasts);
      }

      if (sitePage.path === quickstartPagePath) {
        const quickstartResult = await page.evaluate(() => {
          const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim().toLowerCase();
          const bodyText = normalize(document.body.innerText);
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
          const dashboardImage = document.querySelector('main img[src="assets/server-ui-dashboard.png"]');

          return {
            bodyText,
            headings,
            codeText,
            links,
            dashboardImage: dashboardImage ? {
              alt: normalize(dashboardImage.getAttribute('alt')),
              complete: dashboardImage.complete,
              naturalWidth: dashboardImage.naturalWidth,
              naturalHeight: dashboardImage.naturalHeight,
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

        if (!quickstartResult.dashboardImage) {
          fail(`${sitePage.path}: missing dashboard screenshot assets/server-ui-dashboard.png`);
        }
        if (!quickstartResult.dashboardImage.complete
            || quickstartResult.dashboardImage.naturalWidth <= 0
            || quickstartResult.dashboardImage.naturalHeight <= 0) {
          fail(`${sitePage.path}: dashboard screenshot did not load locally`);
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
          const bodyText = normalize(document.body.innerText);
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

          return {
            bodyText,
            endpointText,
            endpointRoutes,
            endpointDescriptions,
            headings,
            copyableCodeBlocks,
            copyButtonCount: copyButtons.length,
          };
        });

        const missingEndpoints = expectedApiEndpoints.filter((endpoint) => {
          const normalizedEndpoint = endpoint.toLowerCase();
          return !apiResult.endpointText.includes(normalizedEndpoint)
            && !apiResult.bodyText.includes(normalizedEndpoint);
        });
        if (missingEndpoints.length > 0) {
          fail(`${sitePage.path}: missing API endpoint coverage: ${missingEndpoints.join(', ')}`);
        }

        const normalizedStaleEndpoint = staleTrainingJobEndpoint.toLowerCase();
        if (apiResult.endpointRoutes.includes(normalizedStaleEndpoint)) {
          fail(`${sitePage.path}: stale route ${staleTrainingJobEndpoint} is presented as a real API endpoint; use /v1/train/status/{job_id} instead`);
        }
        if (
          apiResult.bodyText.includes(normalizedStaleEndpoint)
          && !apiResult.bodyText.includes(`no separate ${normalizedStaleEndpoint} route`)
        ) {
          fail(`${sitePage.path}: stale route ${staleTrainingJobEndpoint} must appear only in explicit negative wording`);
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

        const missingSections = expectedApiSections
          .filter((section) => !section.terms.every((term) => {
            const normalizedTerm = term.toLowerCase();
            return apiResult.headings.includes(normalizedTerm) || apiResult.bodyText.includes(normalizedTerm);
          }))
          .map((section) => section.label);
        if (missingSections.length > 0) {
          fail(`${sitePage.path}: missing API cold-reader sections: ${missingSections.join(', ')}`);
        }

        const missingCodeExamples = expectedApiCodeExamples
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
          .filter((term) => !cliResult.heroText.includes(term));
        if (missingModelSetupCueTerms.length > 0) {
          fail(`${sitePage.path}: missing ${expectedCliModelSetupCue.label}: ${missingModelSetupCueTerms.join(', ')}`);
        }
        if (!cliResult.heroLinks.some((link) => link.href === expectedCliModelSetupCue.href)) {
          fail(`${sitePage.path}: ${expectedCliModelSetupCue.label} must link to Quickstart`);
        }

        const missingSections = expectedCliSections
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

        const missingCodeExamples = expectedCliCodeExamples
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

          return {
            bodyText,
            headings,
            copyableCodeBlocks,
            links,
          };
        });

        const missingSections = expectedArchitectureSections
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

          return {
            bodyText,
            headings,
            copyableCodeBlocks,
            links,
          };
        });

        const missingSections = expectedTroubleshootingSections
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
