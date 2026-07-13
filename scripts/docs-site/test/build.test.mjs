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
        source: 'docs/CONFIGURATION.md',
        slug: 'configuration',
        title: 'Configuration',
        section: 'start',
        description: 'Complete configuration reference.',
      },
      {
        source: 'docs/GUIDE.md',
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
    resolve(root, 'docs/CONFIGURATION.md'),
    '# Original configuration title\n\n## Load order\n\n[Guide](GUIDE.md#deep-dive)\n\n![Diagram](diagram.png)\n\n```sh\nkiln config\n```\n\n````python\nCODE_BLOCK = re.compile(r"```python\\n(.*?)```")\nsrc = m.group(1) if m else text\n````\n',
  );
  await write(
    resolve(root, 'docs/GUIDE.md'),
    '# Guide\n\n## Deep dive\n\n[Configuration](CONFIGURATION.md#load-order)\n\n[Same heading](#deep-dive)\n\n## Deep dive\n',
  );
  await write(resolve(root, 'docs/diagram.png'), 'diagram');
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
  assert.match(configuration, /<h1 id="original-configuration-title">Configuration<\/h1>/);
  assert.match(configuration, /href="\.\.\/guide\/#deep-dive"/);
  assert.match(configuration, /src="\.\.\/_assets\/docs\/diagram\.png"/);
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
  assert.match(hub, /Complete Kiln documentation/);
  assert.match(hub, /href="\.\/configuration\/"/);

  const index = JSON.parse(await readFile(resolve(first, 'docs/search-index.json'), 'utf8'));
  assert.deepEqual(index.map((entry) => entry.kind), ['product_guide', 'reference', 'reference']);
  assert.equal(index[0].url, '../index.html');
  assert.deepEqual(index.slice(1).map((entry) => entry.slug), ['configuration', 'guide']);
  assert.match(index[1].content, /Load order/);
  assert.equal(await readFile(resolve(first, 'docs/_assets/docs/diagram.png'), 'utf8'), 'diagram');

  const sitemap = await readFile(resolve(first, 'sitemap.xml'), 'utf8');
  assert.match(sitemap, /https:\/\/example\.test\/kiln\//);
  assert.match(sitemap, /https:\/\/example\.test\/kiln\/docs\/configuration\//);
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

test('broken Markdown anchors fail before publication', async () => {
  const fixture = await createFixture();
  await write(
    resolve(fixture.root, 'docs/CONFIGURATION.md'),
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
  const browser = await puppeteer.launch({
    executablePath: chromium,
    headless: true,
    args: ['--disable-gpu', '--no-sandbox'],
  });
  try {
    const page = await browser.newPage();
    await page.goto(`${server.url}/docs/`, { waitUntil: 'domcontentloaded' });
    await page.locator('#docs-search-hub').fill('Product');
    await page.waitForSelector('.docs-search-result');
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
  } finally {
    await browser.close();
    await server.close();
  }
});
