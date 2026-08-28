import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { constants as fsConstants } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { access, mkdtemp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, relative, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer-core';

import {
  DocsBuildError,
  buildDocsSite,
  loadAndValidateManifest,
  slugifyHeading,
} from '../lib.mjs';

const testDir = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(testDir, '../../..');

async function write(path, contents) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, contents);
}

function fixtureManifest() {
  return {
    version: 1,
    site: {
      title: 'Fixture Docs',
      base_url: 'https://example.test/kiln',
      repository_url: 'https://github.com/example/kiln',
      product_guides: [
        { title: 'Product', href: '../index.html', description: 'Product guide.' },
      ],
    },
    sections: [
      { id: 'start', title: 'Start Here' },
    ],
    documents: [
      {
        source: 'docs/contracts/CONFIGURATION.md',
        slug: 'configuration',
        title: 'Configuration',
        section: 'start',
        description: 'Complete configuration reference.',
      },
      {
        source: 'docs/contracts/GUIDE.md',
        slug: 'guide',
        title: 'Guide',
        section: 'start',
        description: 'Complete guide.',
      },
    ],
  };
}

async function createFixture() {
  const root = await mkdtemp(resolve(tmpdir(), 'kiln-docs-builder-'));
  const site = resolve(root, 'docs/site');
  const manifestPath = resolve(site, 'docs-manifest.json');
  await write(
    resolve(site, 'index.html'),
    '<!doctype html><html><head><link rel="canonical" href="https://example.test/kiln/"></head><body><a id="home"></a></body></html>\n',
  );
  await write(resolve(site, 'assets/logo.png'), 'logo');
  await write(resolve(site, 'assets/favicon.svg'), '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n');
  await write(resolve(site, 'css/docs.css'), 'body { color: white; }\n');
  await write(resolve(site, 'js/docs.js'), 'void 0;\n');
  await write(manifestPath, `${JSON.stringify(fixtureManifest(), null, 2)}\n`);
  await write(
    resolve(root, 'docs/contracts/CONFIGURATION.md'),
    '# Original configuration title\n\n## Load order\n\n[Guide](GUIDE.md#deep-dive)\n\n![Diagram](diagram.png)\n\n```sh\nkiln config\n```\n\n````python\nCODE_BLOCK = re.compile(r"```python\\n(.*?)```")\nsrc = m.group(1) if m else text\n````\n',
  );
  await write(
    resolve(root, 'docs/contracts/GUIDE.md'),
    '# Guide\n\n## Deep dive\n\n[Configuration](CONFIGURATION.md#load-order)\n\n[Same heading](#deep-dive)\n\n## Deep dive\n',
  );
  await write(resolve(root, 'docs/contracts/diagram.png'), 'diagram');
  await write(resolve(root, 'Cargo.toml'), '[workspace]\n');
  return { root, site, manifestPath };
}

async function treeDigest(root) {
  const entries = [];
  async function visit(directory) {
    const children = await readdir(directory, { withFileTypes: true });
    children.sort((left, right) => left.name.localeCompare(right.name));
    for (const child of children) {
      const path = resolve(directory, child.name);
      if (child.isDirectory()) await visit(path);
      else if (child.isFile()) {
        const contents = await readFile(path);
        entries.push(`${relative(root, path)}\0${createHash('sha256').update(contents).digest('hex')}`);
      }
    }
  }
  await visit(root);
  return entries;
}

async function executable(path) {
  if (!path) return false;
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findChromium() {
  const candidates = [process.env.CHROME_BIN, process.env.PUPPETEER_EXECUTABLE_PATH];
  for (const command of ['chromium-browser', 'chromium', 'google-chrome', 'google-chrome-stable']) {
    try {
      candidates.push(execFileSync('which', [command], {
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
      }).trim());
    } catch {
      // Continue to the pinned Puppeteer cache used by local docs smoke runs.
    }
  }
  const cacheRoot = resolve(homedir(), '.cache/puppeteer/chrome');
  try {
    const versions = await readdir(cacheRoot);
    versions.sort().reverse();
    for (const version of versions) {
      candidates.push(resolve(cacheRoot, version, 'chrome-linux64/chrome'));
    }
  } catch {
    // A system browser may still have been found above.
  }
  for (const candidate of candidates) {
    if (await executable(candidate)) return candidate;
  }
  return null;
}

async function serveStatic(root) {
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      let target = resolve(root, `.${pathname}`);
      if (!target.startsWith(`${root}/`) && target !== root) {
        response.writeHead(403).end('forbidden');
        return;
      }
      if (pathname.endsWith('/')) target = resolve(target, 'index.html');
      const body = await readFile(target);
      const contentType = target.endsWith('.html')
        ? 'text/html; charset=utf-8'
        : target.endsWith('.json')
          ? 'application/json; charset=utf-8'
          : target.endsWith('.js')
            ? 'text/javascript; charset=utf-8'
            : target.endsWith('.css')
              ? 'text/css; charset=utf-8'
              : 'application/octet-stream';
      response.writeHead(200, { 'content-type': contentType }).end(body);
    } catch {
      response.writeHead(404).end('not found');
    }
  });
  await new Promise((accept, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', accept);
  });
  const address = server.address();
  return {
    url: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((accept, reject) => server.close((error) => (error ? reject(error) : accept()))),
  };
}

test('heading slugs are stable and punctuation-insensitive', () => {
  assert.equal(slugifyHeading('Thinking Budget: Tokens + Time'), 'thinking-budget-tokens-time');
  assert.equal(slugifyHeading('Caf\u00e9 d\u00e9code'), 'cafe-decode');
  assert.equal(slugifyHeading('***'), 'section');
});

test('build copies the static site and emits complete deterministic docs', async () => {
  const fixture = await createFixture();
  const first = resolve(fixture.root, '.build-one');
  const second = resolve(fixture.root, '.build-two');
  const options = {
    repoRoot: fixture.root,
    siteSourceDir: fixture.site,
    manifestPath: fixture.manifestPath,
  };

  const result = await buildDocsSite({ ...options, outDir: first });
  await buildDocsSite({ ...options, outDir: second });
  assert.equal(result.documentCount, 2);
  assert.equal(result.assetCount, 1);

  const configuration = await readFile(resolve(first, 'docs/configuration/index.html'), 'utf8');
  assert.match(configuration, /<link rel="canonical" href="https:\/\/example\.test\/kiln\/docs\/configuration\/">/);
  assert.match(configuration, /<meta name="theme-color" content="#0a0908">/);
  assert.match(configuration, /<meta property="og:title" content="Configuration — Kiln Documentation">/);
  assert.match(configuration, /<meta property="og:url" content="https:\/\/example\.test\/kiln\/docs\/configuration\/">/);
  assert.match(configuration, /<meta property="og:image" content="https:\/\/example\.test\/kiln\/assets\/og-image-v3\.png">/);
  assert.match(configuration, /<meta property="og:image:width" content="1200">/);
  assert.match(configuration, /<meta property="og:image:height" content="630">/);
  assert.match(configuration, /<meta name="twitter:card" content="summary_large_image">/);
  assert.match(configuration, /<meta name="twitter:image:alt" content="Kiln — Serve it\. Teach it\. Watch it get better\./);
  assert.match(configuration, /<h1 id="original-configuration-title">Configuration<\/h1>/);
  assert.match(configuration, /href="\.\.\/guide\/#deep-dive"/);
  assert.match(configuration, /src="\.\.\/_assets\/docs\/contracts\/diagram\.png"/);
  assert.match(configuration, /class="docs-breadcrumbs"/);
  assert.match(configuration, /class="docs-sidebar"/);
  assert.match(configuration, /class="docs-toc"/);
  assert.match(configuration, /class="docs-pager"/);
  assert.match(configuration, /data-copy-code/);

  const guide = await readFile(resolve(first, 'docs/guide/index.html'), 'utf8');
  assert.match(guide, /id="deep-dive"/);
  assert.match(guide, /id="deep-dive-1"/);
  assert.match(guide, /href="\.\.\/configuration\/#load-order"/);

  const hub = await readFile(resolve(first, 'docs/index.html'), 'utf8');
  assert.match(hub, /Choose a path/);
  assert.match(hub, /Start with a product workflow below/);
  assert.match(hub, /Search 3 guides and references/);
  assert.match(hub, /<title>Guides and reference &mdash; Kiln Documentation<\/title>/);
  assert.match(hub, /Start with a workflow/);
  assert.match(hub, /Core documentation/);
  assert.match(hub, /Reference library/);
  assert.match(hub, /href="\.\/configuration\/"/);

  const index = JSON.parse(await readFile(resolve(first, 'docs/search-index.json'), 'utf8'));
  assert.deepEqual(index.map((entry) => entry.kind), ['product_guide', 'reference', 'reference']);
  assert.equal(index[0].url, '../index.html');
  assert.deepEqual(index.slice(1).map((entry) => entry.slug), ['configuration', 'guide']);
  assert.match(index[1].content, /Load order/);
  assert.equal(await readFile(resolve(first, 'docs/_assets/docs/contracts/diagram.png'), 'utf8'), 'diagram');

  const sitemap = await readFile(resolve(first, 'sitemap.xml'), 'utf8');
  assert.match(sitemap, /https:\/\/example\.test\/kiln\//);
  assert.match(sitemap, /https:\/\/example\.test\/kiln\/docs\/configuration\//);
  const llms = await readFile(resolve(first, 'llms.txt'), 'utf8');
  assert.match(llms, /^# Kiln\n\n> Kiln is a pure-Rust, single-GPU server/m);
  assert.match(llms, /\[Product\]\(https:\/\/example\.test\/kiln\/index\.html\): Product guide\./);
  assert.match(llms, /\[Configuration\]\(https:\/\/raw\.githubusercontent\.com\/example\/kiln\/refs\/heads\/main\/docs\/contracts\/CONFIGURATION\.md\): Complete configuration reference\./);
  assert.match(llms, /\[Documentation search index\]\(https:\/\/example\.test\/kiln\/docs\/search-index\.json\)/);
  assert.deepEqual(await treeDigest(first), await treeDigest(second));
});

test('validate-only checks Markdown without writing output', async () => {
  const fixture = await createFixture();
  const result = await buildDocsSite({
    repoRoot: fixture.root,
    siteSourceDir: fixture.site,
    manifestPath: fixture.manifestPath,
    outDir: null,
    validateOnly: true,
  });
  assert.equal(result.documentCount, 2);
});

test('JSON Schema documents render fields, constraints, definitions, and search text', async () => {
  const fixture = await createFixture();
  const manifest = fixtureManifest();
  manifest.documents.push({
    source: 'contracts/receipt.schema.json',
    kind: 'json_schema',
    slug: 'receipt-schema',
    title: 'Receipt Schema',
    section: 'start',
    description: 'Generated receipt contract.',
  });
  await write(fixture.manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  await write(
    resolve(fixture.root, 'contracts/receipt.schema.json'),
    `${JSON.stringify({
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      $id: 'https://example.test/receipt.schema.json',
      title: 'Receipt contract v1',
      description: 'Canonical receipt envelope.',
      type: 'object',
      additionalProperties: false,
      'x-kiln-field-schema-status': 'complete',
      required: ['schema_version', 'result'],
      properties: {
        schema_version: { type: 'integer', const: 1, description: 'Envelope version.' },
        result: { $ref: '#/$defs/result' },
      },
      allOf: [
        {
          if: { properties: { schema_version: { const: 1 } } },
          then: { required: ['result'] },
        },
      ],
      $defs: {
        result: {
          description: 'Closed result record.',
          type: 'object',
          additionalProperties: false,
          required: ['status'],
          properties: {
            status: { type: 'string', enum: ['passed', 'failed'], description: 'Final status.' },
            duration_ms: { type: 'number', minimum: 0, default: 0 },
          },
        },
        config: {
          description: 'Typed configuration fixture.',
          type: 'object',
          additionalProperties: false,
          properties: {
            configured: {
              type: 'boolean',
              default: false,
              'x-kiln-path': 'server.configured',
              'x-kiln-type-and-default': 'boolean; false',
              'x-kiln-canonical-env': 'KILN_SERVER_CONFIGURED (implemented)',
              'x-kiln-environment': 'none',
              'x-kiln-profile-gate': {
                profile: 'experimental',
                when: { const: true },
              },
              'x-kiln-validation': 'Controls the fixture behavior.',
            },
          },
        },
        eventStream: {
          type: 'string',
          contentMediaType: 'text/event-stream',
          'x-kiln-event-types': ['result'],
        },
      },
    }, null, 2)}\n`,
  );

  const output = resolve(fixture.root, '.schema-output');
  await buildDocsSite({
    repoRoot: fixture.root,
    siteSourceDir: fixture.site,
    manifestPath: fixture.manifestPath,
    outDir: output,
  });
  const html = await readFile(resolve(output, 'docs/receipt-schema/index.html'), 'utf8');
  assert.match(html, /Schema identity/);
  assert.match(html, /Receipt contract v1/);
  assert.match(html, /Root fields/);
  assert.match(html, /<code>schema_version<\/code>/);
  assert.match(html, /const 1/);
  assert.match(html, /<td>Result\. Closed result record\.<\/td>/);
  assert.match(html, /<td>Duration ms\.<\/td>/);
  assert.match(html, /Definitions/);
  assert.match(html, /id="result"/);
  assert.match(html, /enum &quot;passed&quot;, &quot;failed&quot;/);
  assert.match(html, /Canonical environment target/);
  assert.match(html, /server\.configured/);
  assert.match(html, /KILN_SERVER_CONFIGURED/);
  assert.match(html, /Alternate environment spelling/);
  assert.match(html, /Profile gate/);
  assert.match(html, /experimental when const true/);
  assert.match(html, /Composition and conditional rules/);
  assert.match(html, /Show exact composition rules/);
  assert.match(html, /&quot;then&quot;: \{/);
  assert.match(html, /Kiln contract annotations/);
  assert.match(html, /Show exact Kiln annotations and examples/);
  assert.match(html, /x-kiln-field-schema-status/);
  assert.match(html, /content media type text\/event-stream/);
  assert.match(html, /x-kiln-event-types/);

  const index = JSON.parse(await readFile(resolve(output, 'docs/search-index.json'), 'utf8'));
  const entry = index.find((item) => item.slug === 'receipt-schema');
  assert.match(entry.content, /Canonical receipt envelope/);
  assert.match(entry.content, /Envelope version/);

  await write(resolve(fixture.root, 'contracts/receipt.schema.json'), '{invalid');
  await assert.rejects(
    loadAndValidateManifest({ repoRoot: fixture.root, manifestPath: fixture.manifestPath }),
    (error) => error instanceof DocsBuildError && /not valid JSON/.test(error.message),
  );
});

test('union JSON Schemas lead with linked public entrypoints', async () => {
  const fixture = await createFixture();
  const manifest = fixtureManifest();
  manifest.documents.push({
    source: 'contracts/union.schema.json',
    kind: 'json_schema',
    slug: 'union-schema',
    title: 'Union schema',
    section: 'start',
    description: 'Generated union contract.',
  });
  await write(fixture.manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  await write(
    resolve(fixture.root, 'contracts/union.schema.json'),
    `${JSON.stringify({
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      title: 'Union fixture',
      oneOf: [
        { $ref: '#/$defs/AlphaRequest' },
        { $ref: '#/$defs/AlphaResponse' },
      ],
      $defs: {
        AlphaRequest: {
          description: 'Submit one alpha.',
          type: 'object',
          additionalProperties: true,
          required: ['value'],
          properties: { value: { type: 'string' } },
        },
        AlphaResponse: {
          description: 'Return the accepted alpha.',
          type: 'object',
          additionalProperties: false,
          required: ['value', 'accepted'],
          properties: {
            value: { type: 'string' },
            accepted: { type: 'boolean' },
          },
        },
      },
    }, null, 2)}\n`,
  );

  const output = resolve(fixture.root, '.union-schema-output');
  await buildDocsSite({
    repoRoot: fixture.root,
    siteSourceDir: fixture.site,
    manifestPath: fixture.manifestPath,
    outDir: output,
  });
  const html = await readFile(resolve(output, 'docs/union-schema/index.html'), 'utf8');
  assert.match(html, /<h2 id="entrypoints">Entrypoints<\/h2>/);
  assert.match(html, /Union of 2 public entrypoints/);
  assert.match(html, /href="#alpharequest"/);
  assert.match(html, /Submit one alpha/);
  assert.match(html, /Unknown fields accepted and ignored/);
  assert.match(html, /Unknown fields rejected/);
  assert.doesNotMatch(html, /<h2 id="root-fields">Root fields<\/h2>/);
});

test('OpenAPI documents render every operation, transport, owner, and payload status', async () => {
  const fixture = await createFixture();
  const manifest = fixtureManifest();
  manifest.documents.push({
    source: 'contracts/http.openapi.json',
    kind: 'openapi',
    slug: 'http-api',
    title: 'HTTP API',
    section: 'start',
    description: 'Generated HTTP contract.',
  });
  await write(fixture.manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  await write(
    resolve(fixture.root, 'contracts/http.openapi.json'),
    `${JSON.stringify({
      openapi: '3.1.1',
      jsonSchemaDialect: 'https://json-schema.org/draft/2020-12/schema',
      info: {
        title: 'Fixture HTTP API',
        version: '1.0.0',
        description: 'Canonical fixture operation inventory.',
      },
      servers: [{ url: 'http://127.0.0.1:8420' }],
      security: [],
      tags: [
        { name: 'status', description: 'Operational status.' },
        { name: 'terminal', description: 'Terminal transport.' },
      ],
      paths: {
        '/health': {
          get: {
            tags: ['status'],
            summary: 'Get health',
            operationId: 'get_health',
            'x-kiln-handler': 'health::health',
            'x-kiln-transport': 'http',
            'x-kiln-query-rust-type': 'HealthQuery',
            responses: {
              200: {
                description: 'Success.',
                'x-kiln-rust-type': 'HealthResponse',
                content: {
                  'application/json': { schema: { $ref: '#/components/schemas/HealthResponse' } },
                },
              },
            },
          },
        },
        '/v1/terminal/ws': {
          get: {
            tags: ['terminal'],
            summary: 'Open terminal',
            operationId: 'open_terminal',
            'x-kiln-handler': 'terminal::terminal_ws',
            'x-kiln-transport': 'websocket',
            parameters: [{ name: 'Origin', in: 'header', schema: { type: 'string' } }],
            responses: {
              101: {
                description: 'Upgrade.',
                'x-kiln-rust-type': 'WebSocketUpgrade',
                content: {
                  'application/octet-stream': { schema: { $ref: '#/components/schemas/WebSocketUpgrade' } },
                },
              },
            },
          },
        },
      },
      components: {
        schemas: {
          HealthResponse: {
            type: 'object',
            additionalProperties: true,
            'x-kiln-rust-type': 'HealthResponse',
          },
          WebSocketUpgrade: {
            type: 'string',
            format: 'binary',
            'x-kiln-rust-type': 'WebSocketUpgrade',
          },
        },
      },
      'x-kiln-field-schema-status': 'migration_pending',
    }, null, 2)}\n`,
  );

  const output = resolve(fixture.root, '.openapi-output');
  await buildDocsSite({
    repoRoot: fixture.root,
    siteSourceDir: fixture.site,
    manifestPath: fixture.manifestPath,
    outDir: output,
  });
  const html = await readFile(resolve(output, 'docs/http-api/index.html'), 'utf8');
  assert.match(html, /Fixture HTTP API/);
  assert.match(html, /2 paths/);
  assert.match(html, /2 operations/);
  assert.match(html, /<code>GET<\/code>/);
  assert.match(html, /<code>\/v1\/terminal\/ws<\/code>/);
  assert.match(html, /WebSocket/);
  assert.match(html, /query: HealthQuery/);
  assert.match(html, /headers: Origin/);
  assert.match(html, /Authentication/);
  assert.match(html, /none declared/);
  assert.match(html, /terminal::terminal_ws/);
  assert.match(html, /Payload components/);
  assert.match(html, /migration pending/);

  const index = JSON.parse(await readFile(resolve(output, 'docs/search-index.json'), 'utf8'));
  const entry = index.find((item) => item.slug === 'http-api');
  assert.match(entry.content, /\/v1\/terminal\/ws/);
  assert.match(entry.content, /WebSocket/);

  await write(resolve(fixture.root, 'contracts/http.openapi.json'), '{invalid');
  await assert.rejects(
    loadAndValidateManifest({ repoRoot: fixture.root, manifestPath: fixture.manifestPath }),
    (error) => error instanceof DocsBuildError && /not valid JSON/.test(error.message),
  );
});

test('broken Markdown anchors fail before publication', async () => {
  const fixture = await createFixture();
  await write(
    resolve(fixture.root, 'docs/contracts/CONFIGURATION.md'),
    '# Configuration\n\n[Broken](GUIDE.md#not-a-heading)\n',
  );
  await assert.rejects(
    buildDocsSite({
      repoRoot: fixture.root,
      siteSourceDir: fixture.site,
      manifestPath: fixture.manifestPath,
      outDir: resolve(fixture.root, '.output'),
    }),
    (error) => error instanceof DocsBuildError && /broken Markdown anchor/.test(error.message),
  );
});

test('manifest rejects duplicate slugs, missing sources, and internal trees', async () => {
  const fixture = await createFixture();
  const manifest = fixtureManifest();
  manifest.documents.push({
    source: 'docs/plans/private.md',
    slug: 'guide',
    title: 'Private',
    section: 'start',
    description: 'Must not publish.',
  });
  await write(fixture.manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  await assert.rejects(
    loadAndValidateManifest({ repoRoot: fixture.root, manifestPath: fixture.manifestPath }),
    (error) => (
      error instanceof DocsBuildError
      && /excluded internal documentation tree/.test(error.message)
      && /duplicate or reserved document slug guide/.test(error.message)
      && /does not exist/.test(error.message)
    ),
  );
});

test('output cannot be nested under the source site', async () => {
  const fixture = await createFixture();
  await assert.rejects(
    buildDocsSite({
      repoRoot: fixture.root,
      siteSourceDir: fixture.site,
      manifestPath: fixture.manifestPath,
      outDir: resolve(fixture.site, '_output'),
    }),
    /output directory must not overlap docs\/site/,
  );
});

test('output replacement rejects destructive paths and unmanaged content', async () => {
  const fixture = await createFixture();
  const options = {
    repoRoot: fixture.root,
    siteSourceDir: fixture.site,
    manifestPath: fixture.manifestPath,
  };
  await assert.rejects(
    buildDocsSite({ ...options, outDir: fixture.root }),
    /repository or one of its ancestors/,
  );

  const unmanaged = resolve(fixture.root, '.unmanaged');
  await write(resolve(unmanaged, 'keep.txt'), 'do not delete');
  await assert.rejects(
    buildDocsSite({ ...options, outDir: unmanaged }),
    /refusing to replace unmanaged non-empty output directory/,
  );
  assert.equal(await readFile(resolve(unmanaged, 'keep.txt'), 'utf8'), 'do not delete');

  const managed = resolve(fixture.root, '.managed');
  await buildDocsSite({ ...options, outDir: managed });
  await buildDocsSite({ ...options, outDir: managed });
  assert.match(await readFile(resolve(managed, '.kiln-docs-site-output'), 'utf8'), /output-v1/);
});

test('search loads its index and resolves product and reference results over HTTP', async (context) => {
  const chromium = await findChromium();
  if (!chromium) {
    context.skip('Chromium is not available; Pages smoke supplies CHROME_BIN');
    return;
  }
  const fixture = await createFixture();
  await write(
    resolve(fixture.site, 'js/docs.js'),
    await readFile(resolve(repositoryRoot, 'docs/site/js/docs.js'), 'utf8'),
  );
  const output = resolve(fixture.root, '.http-output');
  await buildDocsSite({
    repoRoot: fixture.root,
    siteSourceDir: fixture.site,
    manifestPath: fixture.manifestPath,
    outDir: output,
  });
  const server = await serveStatic(output);
  let browser;
  try {
    browser = await puppeteer.launch({
      executablePath: chromium,
      headless: true,
      args: ['--disable-gpu', '--no-sandbox'],
    });
    const page = await browser.newPage();
    await page.goto(`${server.url}/docs/`, { waitUntil: 'domcontentloaded' });
    await page.keyboard.press('/');
    assert.equal(
      await page.$eval('#docs-search-hub', (element) => document.activeElement === element),
      true,
    );
    await page.locator('#docs-search-hub').fill('Product');
    await page.waitForSelector('.docs-search-result');
    assert.equal(await page.$eval('#docs-search-hub', (element) => element.getAttribute('aria-expanded')), 'true');
    const product = await page.$eval('.docs-search-result', (element) => ({
      text: element.textContent,
      href: element.getAttribute('href'),
    }));
    assert.match(product.text, /Product/);
    assert.equal(product.href, '../index.html');

    await page.locator('#docs-search-hub').fill('Load order');
    await page.waitForFunction(() => document.querySelector('.docs-search-result')?.textContent.includes('Configuration'));
    const referenceHref = await page.$eval('.docs-search-result', (element) => element.getAttribute('href'));
    assert.equal(referenceHref, './configuration/');

    await page.locator('#docs-search-hub').fill('not-a-published-kiln-term');
    await page.waitForFunction(() => document.querySelector('.docs-search-empty')?.textContent.includes('Try fewer or broader words.'));
    assert.match(
      await page.$eval('.docs-search-empty', (element) => element.textContent),
      /No documents match .* Try fewer or broader words\./,
    );
  } finally {
    await browser?.close();
    await server.close();
  }
});
