#!/usr/bin/env node
import fs from 'node:fs';
import vm from 'node:vm';

const files = {
  dashboard: 'desktop/ui/dashboard.html',
  settings: 'desktop/ui/settings.html',
  runtimeDefaults: 'desktop/ui/_kiln-runtime-defaults.js',
  runtimeDefaultsContract: 'contracts/runtime-defaults-v1.json',
  thinkingBudgetContract: 'contracts/thinking-budget-v1.conformance.json',
};

const quickstartHref = 'https://ericflo.github.io/kiln/quickstart.html';
const troubleshootingHref = 'https://ericflo.github.io/kiln/troubleshooting.html';
const forbiddenPublicityTerms = [
  'launch post',
  'announcement',
  'press release',
  'twitter',
  'x-twitter',
  'lobste.rs',
  'localLLaMA',
  'discord',
  'hacker news',
  'hn launch',
];

function read(path) {
  return fs.readFileSync(path, 'utf8');
}

function stripHtml(html) {
  return html
    .replace(/<script\b[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

function assertContains(source, needle, label) {
  if (!source.includes(needle)) {
    throw new Error(`${label} is missing ${needle}`);
  }
}

function assertAny(source, needles, label) {
  if (!needles.some((needle) => source.includes(needle))) {
    throw new Error(`${label} is missing one of: ${needles.join(', ')}`);
  }
}

function assertNoForbiddenPublicityCopy(text, label) {
  for (const term of forbiddenPublicityTerms) {
    if (text.includes(term.toLowerCase())) {
      throw new Error(`${label} should not use external publicity wording: ${term}`);
    }
  }
}

function checkDashboard(html) {
  const text = stripHtml(html);
  assertContains(html, '_kiln-runtime-defaults.js', 'dashboard runtime defaults');
  assertContains(html, 'KILN_RUNTIME_DEFAULTS.serverPort', 'dashboard port fallback');
  assertContains(html, quickstartHref, 'dashboard first-run help');
  assertContains(html, troubleshootingHref, 'dashboard first-run help');
  assertContains(text, 'quickstart', 'dashboard first-run help text');
  assertContains(text, 'troubleshooting', 'dashboard first-run help text');
  assertAny(text, ['model path', 'server binary', 'kiln server binary'], 'dashboard first-run setup copy');
  assertContains(text, 'build from source', 'dashboard source-build link copy');
  assertNoForbiddenPublicityCopy(text, 'dashboard first-run help');
}

function checkSettings(html) {
  const text = stripHtml(html);
  assertContains(html, '_kiln-runtime-defaults.js', 'settings runtime defaults');
  assertContains(html, 'KILN_RUNTIME_DEFAULTS.serverPort', 'settings port fallback');
  assertContains(html, quickstartHref, 'settings setup help');
  assertContains(html, troubleshootingHref, 'settings setup help');
  assertContains(text, 'quickstart', 'settings setup help text');
  assertContains(text, 'troubleshooting', 'settings setup help text');
  assertContains(text, 'model path', 'settings setup help copy');
  assertAny(text, ['server binary', 'kiln binary'], 'settings binary setup copy');
  assertNoForbiddenPublicityCopy(text, 'settings setup help');

  const parserStart = html.indexOf('function strictThinkingBudgetInteger(raw)');
  const parserEnd = html.indexOf('function readForm()', parserStart);
  if (parserStart < 0 || parserEnd < 0) {
    throw new Error('settings thinking-budget parsers are missing');
  }
  const inputs = {
    default_thinking_budget_tokens: { value: '', validity: { badInput: false } },
    default_thinking_budget_seconds: { value: '', validity: { badInput: false } },
  };
  const context = vm.createContext({
    document: { getElementById: (id) => inputs[id] },
    thinkingBudgetMode: 'custom',
  });
  vm.runInContext(
    `${html.slice(parserStart, parserEnd)}\nthis.parsers = { strictThinkingBudgetInteger, strictThinkingBudgetMilliseconds, readThinkingBudget };`,
    context,
  );

  const integerCases = new Map([
    ['0', 0], ['0002', 2], ['9007199254740991', Number.MAX_SAFE_INTEGER],
    ['1.5', null], ['1e2', null], ['+1', null], ['-1', null],
    ['9007199254740992', null],
  ]);
  for (const [raw, expected] of integerCases) {
    const actual = context.parsers.strictThinkingBudgetInteger(raw);
    if (actual !== expected) {
      throw new Error(`settings token parser returned ${String(actual)} for ${JSON.stringify(raw)}; expected ${String(expected)}`);
    }
  }

  const millisecondCases = new Map([
    ['0', 0], ['.001', 1], ['0.010', 10], ['1.25', 1250],
    ['1.0001', null], ['1e2', null], ['+1', null], ['-1', null], ['1.', null],
    ['9007199254740.991', Number.MAX_SAFE_INTEGER],
    ['9007199254740.992', null],
  ]);
  for (const [raw, expected] of millisecondCases) {
    const actual = context.parsers.strictThinkingBudgetMilliseconds(raw);
    if (actual !== expected) {
      throw new Error(`settings time parser returned ${String(actual)} for ${JSON.stringify(raw)}; expected ${String(expected)}`);
    }
  }

  assertContains(html, 'input.validity.badInput', 'settings malformed number-state guard');
  if (html.includes('Math.round(milliseconds)')) {
    throw new Error('settings time budget must not round a floating-point conversion');
  }

  inputs.default_thinking_budget_tokens.value = '1.5';
  inputs.default_thinking_budget_seconds.value = '1';
  assertThrowsBudgetRead(context, /whole number/, 'settings decimal token budget');

  inputs.default_thinking_budget_tokens.value = '';
  inputs.default_thinking_budget_tokens.validity.badInput = true;
  assertThrowsBudgetRead(context, /whole number/, 'settings native malformed token state');

  inputs.default_thinking_budget_tokens.validity.badInput = false;
  inputs.default_thinking_budget_tokens.value = '0';
  inputs.default_thinking_budget_seconds.value = '1.25';
  const custom = context.parsers.readThinkingBudget();
  if (custom.default_thinking_budget_tokens !== 0 || custom.default_thinking_budget_ms !== 1250) {
    throw new Error(`settings custom budget produced ${JSON.stringify(custom)}`);
  }

  context.thinkingBudgetMode = 'unlimited';
  const unlimited = context.parsers.readThinkingBudget();
  if (unlimited.default_thinking_budget_tokens !== null || unlimited.default_thinking_budget_ms !== null) {
    throw new Error(`settings Unlimited mode produced ${JSON.stringify(unlimited)}`);
  }

  const contract = JSON.parse(read(files.thinkingBudgetContract));
  if (contract.contract_version !== 1 || !Array.isArray(contract.server_default_cases)) {
    throw new Error('thinking-budget contract is missing v1 server-default cases');
  }
  for (const testCase of contract.server_default_cases) {
    context.thinkingBudgetMode = testCase.mode;
    inputs.default_thinking_budget_tokens.validity.badInput = false;
    inputs.default_thinking_budget_seconds.validity.badInput = false;
    inputs.default_thinking_budget_tokens.value = testCase.tokens_input;
    inputs.default_thinking_budget_seconds.value = testCase.seconds_input;
    const actual = context.parsers.readThinkingBudget();
    if (JSON.stringify(actual) !== JSON.stringify(testCase.settings)) {
      throw new Error(`${testCase.name} produced ${JSON.stringify(actual)}; expected ${JSON.stringify(testCase.settings)}`);
    }
  }
}

function checkRuntimeDefaults() {
  const contract = JSON.parse(read(files.runtimeDefaultsContract));
  if (contract.contract_version !== 1 || !contract.server) {
    throw new Error('runtime-defaults contract is missing the v1 server record');
  }

  const context = vm.createContext({});
  vm.runInContext(read(files.runtimeDefaults), context);
  const defaults = context.KILN_RUNTIME_DEFAULTS;
  if (!defaults) {
    throw new Error('desktop runtime defaults did not define KILN_RUNTIME_DEFAULTS');
  }
  if (defaults.serverHost !== contract.server.bind_host) {
    throw new Error(`desktop default host ${defaults.serverHost} does not match ${contract.server.bind_host}`);
  }
  if (defaults.serverPort !== contract.server.port) {
    throw new Error(`desktop default port ${defaults.serverPort} does not match ${contract.server.port}`);
  }
  const expectedBase = `http://${contract.server.bind_host}:${contract.server.port}`;
  if (defaults.serverBaseUrl !== expectedBase || defaults.openAiBaseUrl !== `${expectedBase}/v1`) {
    throw new Error(`desktop default URLs do not match ${expectedBase}`);
  }
}

function assertThrowsBudgetRead(context, pattern, label) {
  try {
    context.parsers.readThinkingBudget();
  } catch (error) {
    if (pattern.test(error.message)) return;
    throw new Error(`${label} failed with unexpected error: ${error.message}`);
  }
  throw new Error(`${label} was accepted`);
}

checkRuntimeDefaults();
checkDashboard(read(files.dashboard));
checkSettings(read(files.settings));
console.log('Desktop UI smoke checks passed');
