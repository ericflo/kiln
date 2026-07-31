#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(import.meta.dirname, '..');
const schemaPath = 'contracts/thinking-budget-v1.schema.json';
const vectorsPath = 'contracts/thinking-budget-v1.conformance.json';
const referencePath = 'docs/THINKING_BUDGET_CONTRACT.md';
const generatedStart = '<!-- thinking-budget-contract-v1:generated:start -->';
const generatedEnd = '<!-- thinking-budget-contract-v1:generated:end -->';

function read(path) {
  return readFileSync(resolve(repoRoot, path), 'utf8');
}

function readJson(path) {
  try {
    return JSON.parse(read(path));
  } catch (error) {
    throw new Error(`${path} is not valid JSON: ${error.message}`);
  }
}

function requireArray(value, label) {
  if (!Array.isArray(value) || value.length === 0 || value.some((item) => typeof item !== 'string')) {
    throw new Error(`${label} must be a non-empty string array`);
  }
  return value;
}

function requireExactSet(actual, expected, label) {
  const actualSorted = [...new Set(actual)].sort();
  const expectedSorted = [...expected].sort();
  if (JSON.stringify(actualSorted) !== JSON.stringify(expectedSorted)) {
    throw new Error(`${label} is ${JSON.stringify(actualSorted)}; expected ${JSON.stringify(expectedSorted)}`);
  }
}

const schema = readJson(schemaPath);
const vectors = readJson(vectorsPath);
if (vectors.contract_version !== 1) {
  throw new Error(`${vectorsPath} contract_version must be 1`);
}
if (schema.$id !== 'https://ericflo.github.io/kiln/contracts/thinking-budget-v1.schema.json') {
  throw new Error(`${schemaPath} has an unexpected $id: ${JSON.stringify(schema.$id)}`);
}
if (!Array.isArray(vectors.resolution_cases) || vectors.resolution_cases.length === 0) {
  throw new Error(`${vectorsPath} must contain resolution_cases`);
}

const overrideStates = vectors.resolution_cases.flatMap((testCase) => [
  testCase.tokens?.state,
  testCase.time?.state,
]);
requireExactSet(overrideStates, ['inherit', 'unlimited', 'limit'], 'thinking-budget override states');
const sources = requireArray(schema.$defs?.source?.enum, `${schemaPath} source enum`);
const triggers = requireArray(schema.$defs?.trigger?.enum, `${schemaPath} trigger enum`);

const generatedReference = [
  generatedStart,
  `- Contract version: \`${vectors.contract_version}\``,
  '- Request override states: `inherit`, `unlimited`, `limit`',
  `- Source vocabulary: ${sources.map((value) => `\`${value}\``).join(', ')}`,
  `- Trigger vocabulary: ${triggers.map((value) => `\`${value}\``).join(', ')}`,
  generatedEnd,
].join('\n');

if (process.argv.includes('--print-generated')) {
  console.log(generatedReference);
  process.exit(0);
}

const reference = read(referencePath);
const startIndex = reference.indexOf(generatedStart);
const endIndex = reference.indexOf(generatedEnd, startIndex + generatedStart.length);
if (startIndex < 0 || endIndex < 0) {
  throw new Error(`${referencePath} is missing the generated contract vocabulary block`);
}
const actualReference = reference.slice(startIndex, endIndex + generatedEnd.length);
if (actualReference !== generatedReference) {
  throw new Error(
    `${referencePath} vocabulary drifted from the schema/vectors; run `
    + '`node scripts/check_thinking_budget_contract.mjs --print-generated` for the expected block',
  );
}

const requiredLinks = new Map([
  ['README.md', 'docs/THINKING_BUDGET_CONTRACT.md'],
  ['QUICKSTART.md', 'docs/THINKING_BUDGET_CONTRACT.md'],
  ['docs/EVAL_GUIDE.md', 'THINKING_BUDGET_CONTRACT.md'],
  ['docs/site/api.html', 'docs/thinking-budgets/'],
  ['docs/site/quickstart.html', 'docs/thinking-budgets/'],
]);
for (const [path, target] of requiredLinks) {
  if (!read(path).includes(target)) {
    throw new Error(`${path} must link to the canonical thinking-budget reference at ${target}`);
  }
}

console.log('thinking-budget schema and documentation contract passed');
