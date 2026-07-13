#!/usr/bin/env node

import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { DocsBuildError, buildDocsSite } from './lib.mjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../..');

function usage() {
  return `Usage: node scripts/docs-site/build.mjs --out <directory> [options]

Build the static documentation site from docs/site/docs-manifest.json.

Options:
  --out <directory>   Destination to replace with the complete static site
  --validate-only     Validate the manifest and Markdown without writing output
  --manifest <path>   Override the manifest path (primarily for tests)
  --site <path>       Override the static site source directory
  --help              Show this help
`;
}

function parseArguments(argv) {
  const options = {
    outDir: null,
    validateOnly: false,
    manifestPath: resolve(repoRoot, 'docs/site/docs-manifest.json'),
    siteSourceDir: resolve(repoRoot, 'docs/site'),
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help') return { help: true };
    if (argument === '--validate-only') {
      options.validateOnly = true;
      continue;
    }
    if (['--out', '--manifest', '--site'].includes(argument)) {
      const value = argv[index + 1];
      if (!value || value.startsWith('--')) throw new DocsBuildError(`${argument} requires a path`);
      index += 1;
      if (argument === '--out') options.outDir = resolve(value);
      if (argument === '--manifest') options.manifestPath = resolve(value);
      if (argument === '--site') options.siteSourceDir = resolve(value);
      continue;
    }
    throw new DocsBuildError(`unknown argument: ${argument}`);
  }
  if (!options.validateOnly && !options.outDir) {
    throw new DocsBuildError('--out is required unless --validate-only is used');
  }
  return options;
}

try {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(usage());
    process.exit(0);
  }
  const result = await buildDocsSite({ repoRoot, ...options });
  const action = options.validateOnly ? 'validated' : `built ${options.outDir}`;
  process.stdout.write(`Documentation ${action}: ${result.documentCount} documents, ${result.assetCount} copied assets\n`);
} catch (error) {
  if (error instanceof DocsBuildError) {
    process.stderr.write(`Documentation build failed:\n${error.message}\n`);
  } else {
    process.stderr.write(`${error?.stack ?? error}\n`);
  }
  process.exit(1);
}
