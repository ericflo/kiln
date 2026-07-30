import MarkdownIt from 'markdown-it';
import {
  cp,
  copyFile,
  lstat,
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import {
  basename,
  dirname,
  extname,
  isAbsolute,
  relative,
  resolve,
  sep,
} from 'node:path';

const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const SECTION_PATTERN = /^[a-z][a-z0-9-]*$/;
const EXCLUDED_SOURCE_SEGMENTS = new Set([
  'archive',
  'audits',
  'plans',
  '.ipynb_checkpoints',
]);
const IMAGE_EXTENSIONS = new Set([
  '.avif',
  '.gif',
  '.jpeg',
  '.jpg',
  '.png',
  '.svg',
  '.webp',
]);
const OUTPUT_MARKER = '.kiln-docs-site-output';
const DOCUMENT_KINDS = new Set(['markdown', 'json_schema', 'openapi']);

export class DocsBuildError extends Error {
  constructor(messages) {
    const errors = Array.isArray(messages) ? messages : [messages];
    super(errors.join('\n'));
    this.name = 'DocsBuildError';
    this.errors = errors;
  }
}

function asPosix(path) {
  return path.split(sep).join('/');
}

function isWithin(path, root) {
  const rel = relative(root, path);
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function escapeXml(value) {
  return escapeHtml(value);
}

function decodeHtmlAttribute(value) {
  return value
    .replaceAll('&amp;', '&')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
    .replaceAll('&apos;', "'")
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>');
}

function encodeRepoPath(path) {
  return path.split('/').map(encodeURIComponent).join('/');
}

function normalizeWhitespace(value) {
  return value.replace(/\s+/g, ' ').trim();
}

export function slugifyHeading(value) {
  const slug = normalizeWhitespace(value)
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^\p{Letter}\p{Number}\s_-]/gu, '')
    .trim()
    .replace(/\s+/g, '-');
  return slug || 'section';
}

function textFromInlineToken(token) {
  if (!token?.children) return normalizeWhitespace(token?.content ?? '');
  const parts = [];
  for (const child of token.children) {
    if (['text', 'code_inline', 'emoji'].includes(child.type)) {
      parts.push(child.content);
    } else if (['softbreak', 'hardbreak'].includes(child.type)) {
      parts.push(' ');
    } else if (child.type === 'image') {
      parts.push(child.content || child.attrGet('alt') || '');
    }
  }
  return normalizeWhitespace(parts.join(' '));
}

function explicitHtmlAnchors(markdown) {
  const anchors = new Set();
  const pattern = /\b(?:id|name)\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s"'=<>`]+))/gi;
  for (const match of markdown.matchAll(pattern)) {
    anchors.add(decodeHtmlAttribute(match[1] ?? match[2] ?? match[3] ?? ''));
  }
  return anchors;
}

function decorateHeadings(tokens, markdown) {
  const seen = new Map();
  const headings = [];
  const anchors = explicitHtmlAnchors(markdown);
  let firstH1 = null;

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token.type !== 'heading_open') continue;
    const level = Number.parseInt(token.tag.slice(1), 10);
    const inline = tokens[index + 1];
    const text = textFromInlineToken(inline);
    const base = slugifyHeading(text);
    const duplicate = seen.get(base) ?? 0;
    seen.set(base, duplicate + 1);
    const id = duplicate === 0 ? base : `${base}-${duplicate}`;
    token.attrSet('id', id);
    anchors.add(id);
    const heading = { level, id, text };
    headings.push(heading);
    if (level === 1 && firstH1 === null) {
      firstH1 = { ...heading, tokenIndex: index };
    }
  }

  return { anchors, headings, firstH1 };
}

function stripFirstH1(tokens, firstH1) {
  if (!firstH1) return tokens;
  const index = firstH1.tokenIndex;
  if (
    tokens[index]?.type === 'heading_open'
    && tokens[index + 1]?.type === 'inline'
    && tokens[index + 2]?.type === 'heading_close'
  ) {
    return [...tokens.slice(0, index), ...tokens.slice(index + 3)];
  }
  return tokens;
}

function searchableText(tokens) {
  const parts = [];
  for (const token of tokens) {
    if (token.type === 'inline') {
      const text = textFromInlineToken(token);
      if (text) parts.push(text);
    }
  }
  return normalizeWhitespace(parts.join(' '));
}

function createMarkdown() {
  const markdown = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: false,
  });

  markdown.renderer.rules.fence = (tokens, index) => {
    const token = tokens[index];
    const language = token.info.trim().split(/\s+/, 1)[0];
    const className = language && /^[A-Za-z0-9_+-]+$/.test(language)
      ? ` class="language-${escapeHtml(language)}"`
      : '';
    return `<div class="docs-code"><button type="button" class="docs-copy" data-copy-code aria-label="Copy code" title="Copy code">Copy</button><pre><code${className}>${escapeHtml(token.content)}</code></pre></div>\n`;
  };
  markdown.renderer.rules.code_block = (tokens, index) => (
    `<div class="docs-code"><button type="button" class="docs-copy" data-copy-code aria-label="Copy code" title="Copy code">Copy</button><pre><code>${escapeHtml(tokens[index].content)}</code></pre></div>\n`
  );
  markdown.renderer.rules.table_open = () => '<div class="docs-table-scroll"><table>\n';
  markdown.renderer.rules.table_close = () => '</table></div>\n';
  return markdown;
}

function nonEmptyString(value) {
  return typeof value === 'string' && value.trim() !== '';
}

function documentKind(document) {
  return document.kind ?? 'markdown';
}

function markdownTableCell(value) {
  return String(value ?? '')
    .replaceAll('\\', '\\\\')
    .replaceAll('|', '\\|')
    .replace(/\s+/g, ' ')
    .trim() || '-';
}

function schemaType(schema) {
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) return 'unknown';
  if (nonEmptyString(schema.$ref)) return schema.$ref.split('/').at(-1);
  if (Array.isArray(schema.type)) return schema.type.join(' | ');
  if (nonEmptyString(schema.type)) {
    if (schema.type === 'array' && schema.items) return `array<${schemaType(schema.items)}>`;
    return schema.type;
  }
  for (const keyword of ['oneOf', 'anyOf', 'allOf']) {
    if (Array.isArray(schema[keyword])) {
      return schema[keyword].map(schemaType).join(keyword === 'allOf' ? ' & ' : ' | ');
    }
  }
  if (schema.properties) return 'object';
  return 'any';
}

function compactJson(value) {
  return JSON.stringify(value).replaceAll('|', '\\u007c');
}

function schemaConstraints(schema) {
  const constraints = [];
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) return constraints;
  if (Object.hasOwn(schema, 'const')) constraints.push(`const ${compactJson(schema.const)}`);
  if (Array.isArray(schema.enum)) constraints.push(`enum ${schema.enum.map(compactJson).join(', ')}`);
  if (Object.hasOwn(schema, 'default')) constraints.push(`default ${compactJson(schema.default)}`);
  for (const [keyword, label] of [
    ['minimum', 'minimum'],
    ['exclusiveMinimum', 'exclusive minimum'],
    ['maximum', 'maximum'],
    ['exclusiveMaximum', 'exclusive maximum'],
    ['minLength', 'minimum length'],
    ['maxLength', 'maximum length'],
    ['minItems', 'minimum items'],
    ['maxItems', 'maximum items'],
    ['minProperties', 'minimum properties'],
    ['maxProperties', 'maximum properties'],
  ]) {
    if (Object.hasOwn(schema, keyword)) constraints.push(`${label} ${schema[keyword]}`);
  }
  if (nonEmptyString(schema.pattern)) constraints.push(`pattern ${schema.pattern}`);
  if (nonEmptyString(schema.format)) constraints.push(`format ${schema.format}`);
  if (nonEmptyString(schema.contentMediaType)) {
    constraints.push(`content media type ${schema.contentMediaType}`);
  }
  if (schema.uniqueItems === true) constraints.push('unique items');
  if (schema.additionalProperties === false) constraints.push('closed object');
  return constraints;
}

function schemaFieldTable(schema) {
  const properties = schema?.properties;
  if (!properties || typeof properties !== 'object' || Array.isArray(properties)) {
    return '_This schema node has no named object fields._\n';
  }
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  const entries = Object.entries(properties);
  const hasKilnConfigMetadata = entries.length > 0
    && entries.every(([, field]) => nonEmptyString(field?.['x-kiln-path']));
  if (hasKilnConfigMetadata) {
    const rows = entries.map(([name, field]) => {
      const path = field?.['x-kiln-path'] ?? name;
      const typeAndDefault = field?.['x-kiln-type-and-default'] ?? schemaType(field);
      const canonicalEnvironment = field?.['x-kiln-canonical-env'] ?? '';
      const environment = field?.['x-kiln-environment'] ?? '';
      const profileGate = field?.['x-kiln-profile-gate'];
      const profileGateText = nonEmptyString(profileGate?.profile)
        ? `${profileGate.profile} when ${schemaConstraints(profileGate.when).join('; ')}`
        : 'none';
      const validation = field?.['x-kiln-validation'] ?? field?.description ?? '';
      return `| \`${markdownTableCell(path)}\` | ${required.has(name) ? 'yes' : 'no'} | ${markdownTableCell(typeAndDefault)} | \`${markdownTableCell(canonicalEnvironment)}\` | ${markdownTableCell(environment)} | ${markdownTableCell(profileGateText)} | ${markdownTableCell(validation)} |`;
    });
    return [
      '| Field | Required | Type and default | Canonical environment target | Alternate environment spelling | Profile gate | Validation and semantics |',
      '| --- | --- | --- | --- | --- | --- | --- |',
      ...rows,
      '',
    ].join('\n');
  }
  const rows = entries.map(([name, field]) => {
    const description = field?.description ?? field?.title ?? '';
    return `| \`${markdownTableCell(name)}\` | ${required.has(name) ? 'yes' : 'no'} | \`${markdownTableCell(schemaType(field))}\` | ${markdownTableCell(schemaConstraints(field).join('; '))} | ${markdownTableCell(description)} |`;
  });
  return [
    '| Field | Required | Type | Constraints and default | Description |',
    '| --- | --- | --- | --- | --- |',
    ...rows,
    '',
  ].join('\n');
}

function schemaStructuralRules(schema) {
  const rules = {};
  for (const keyword of [
    'allOf',
    'anyOf',
    'oneOf',
    'not',
    'if',
    'then',
    'else',
    'dependentRequired',
    'dependentSchemas',
    'propertyNames',
    'patternProperties',
    'contains',
  ]) {
    if (Object.hasOwn(schema ?? {}, keyword)) rules[keyword] = schema[keyword];
  }
  if (schema?.additionalProperties && typeof schema.additionalProperties === 'object') {
    rules.additionalProperties = schema.additionalProperties;
  }
  return rules;
}

function schemaStructuralMarkdown(schema, headingLevel) {
  const rules = schemaStructuralRules(schema);
  if (Object.keys(rules).length === 0) return '';
  const heading = '#'.repeat(headingLevel);
  return `${heading} Composition and conditional rules\n\nThe following JSON is copied exactly from this schema node.\n\n\`\`\`json\n${JSON.stringify(rules, null, 2)}\n\`\`\`\n`;
}

function schemaContractAnnotations(schema, headingLevel) {
  const annotations = Object.fromEntries(
    Object.entries(schema ?? {})
      .filter(([key]) => key.startsWith('x-kiln-'))
      .sort(([left], [right]) => left.localeCompare(right)),
  );
  if (Object.keys(annotations).length === 0) return '';
  const heading = '#'.repeat(headingLevel);
  return `${heading} Kiln contract annotations\n\nThese machine-readable annotations are copied exactly from this schema node.\n\n\`\`\`json\n${JSON.stringify(annotations, null, 2)}\n\`\`\`\n`;
}

function renderJsonSchemaMarkdown(schema, document) {
  const lines = [`# ${document.title}`, ''];
  if (nonEmptyString(schema.description)) lines.push(schema.description.trim(), '');
  lines.push('## Schema identity', '');
  lines.push('| Property | Value |', '| --- | --- |');
  if (nonEmptyString(schema.title)) lines.push(`| Title | ${markdownTableCell(schema.title)} |`);
  if (nonEmptyString(schema.$id)) lines.push(`| \`$id\` | \`${markdownTableCell(schema.$id)}\` |`);
  if (nonEmptyString(schema.$schema)) lines.push(`| Dialect | \`${markdownTableCell(schema.$schema)}\` |`);
  lines.push(`| Root type | \`${markdownTableCell(schemaType(schema))}\` |`);
  lines.push(`| Root object | ${schema.additionalProperties === false ? 'closed' : 'open or unspecified'} |`, '');
  lines.push('## Root fields', '', schemaFieldTable(schema));
  const rootRules = schemaStructuralMarkdown(schema, 3);
  if (rootRules) lines.push(rootRules);
  const rootAnnotations = schemaContractAnnotations(schema, 3);
  if (rootAnnotations) lines.push(rootAnnotations);

  const definitions = schema.$defs ?? schema.definitions ?? {};
  if (definitions && typeof definitions === 'object' && !Array.isArray(definitions)) {
    const entries = Object.entries(definitions);
    if (entries.length > 0) lines.push('## Definitions', '');
    for (const [name, definition] of entries) {
      lines.push(`### ${name}`, '');
      if (nonEmptyString(definition?.description)) lines.push(definition.description.trim(), '');
      const constraints = schemaConstraints(definition);
      lines.push(`Type: \`${schemaType(definition)}\`${constraints.length ? `. Constraints: ${constraints.join('; ')}.` : '.'}`, '');
      if (definition?.properties) lines.push(schemaFieldTable(definition));
      const definitionRules = schemaStructuralMarkdown(definition, 4);
      if (definitionRules) lines.push(definitionRules);
      const definitionAnnotations = schemaContractAnnotations(definition, 4);
      if (definitionAnnotations) lines.push(definitionAnnotations);
    }
  }
  return `${lines.join('\n').trim()}\n`;
}

function openApiOperations(spec) {
  const methods = ['get', 'post', 'put', 'patch', 'delete'];
  const operations = [];
  for (const [path, item] of Object.entries(spec?.paths ?? {})) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
    for (const method of methods) {
      const operation = item[method];
      if (operation && typeof operation === 'object' && !Array.isArray(operation)) {
        operations.push({ path, method: method.toUpperCase(), operation });
      }
    }
  }
  return operations;
}

function openApiRequestSummary(operation) {
  const parts = [];
  const pathParameters = (operation.parameters ?? [])
    .filter((parameter) => parameter?.in === 'path')
    .map((parameter) => parameter.name);
  const headerParameters = (operation.parameters ?? [])
    .filter((parameter) => parameter?.in === 'header')
    .map((parameter) => parameter.name);
  if (pathParameters.length > 0) parts.push(`path: ${pathParameters.join(', ')}`);
  if (headerParameters.length > 0) parts.push(`headers: ${headerParameters.join(', ')}`);
  if (nonEmptyString(operation['x-kiln-query-rust-type'])) {
    parts.push(`query: ${operation['x-kiln-query-rust-type']}`);
  }
  const body = operation.requestBody;
  if (body?.content && typeof body.content === 'object') {
    const media = Object.keys(body.content).join(', ');
    const type = body['x-kiln-rust-type'] ?? 'declared body';
    parts.push(`body: ${type} (${media})`);
  }
  return parts.join('; ') || 'none';
}

function openApiResponseSummary(operation) {
  const successes = Object.entries(operation.responses ?? {})
    .filter(([status]) => /^\d+$/.test(status) && Number(status) >= 100 && Number(status) < 400);
  if (successes.length === 0) return 'unspecified';
  return successes.map(([status, response]) => {
    const media = Object.keys(response?.content ?? {}).join(', ') || 'no body';
    const type = response?.['x-kiln-rust-type'] ?? 'declared response';
    return `${status}: ${type} (${media})`;
  }).join('; ');
}

function renderOpenApiMarkdown(spec, document) {
  const operations = openApiOperations(spec);
  const schemaCounts = spec?.['x-kiln-component-schema-counts'];
  const schemaProgress = schemaCounts && typeof schemaCounts === 'object'
    ? ` Of ${schemaCounts.total} top-level payload components, **${schemaCounts.complete} are field-complete** and **${schemaCounts.migration_pending} remain migration pending**.`
    : '';
  const lines = [`# ${document.title}`, ''];
  if (nonEmptyString(spec?.info?.description)) lines.push(spec.info.description.trim(), '');
  lines.push('## Contract status', '');
  lines.push(
    `This contract contains **${Object.keys(spec?.paths ?? {}).length} paths** and **${operations.length} operations**. `
      + `The aggregate field status is \`${spec?.['x-kiln-field-schema-status'] ?? 'unspecified'}\`; every operation and transport remains canonical while the remaining payload components are migrated.${schemaProgress}`,
    '',
  );
  lines.push('## OpenAPI identity', '');
  lines.push('| Property | Value |', '| --- | --- |');
  lines.push(`| Title | ${markdownTableCell(spec?.info?.title ?? document.title)} |`);
  lines.push(`| Version | \`${markdownTableCell(spec?.info?.version ?? 'unknown')}\` |`);
  lines.push(`| OpenAPI | \`${markdownTableCell(spec?.openapi ?? 'unknown')}\` |`);
  if (nonEmptyString(spec?.jsonSchemaDialect)) {
    lines.push(`| JSON Schema dialect | \`${markdownTableCell(spec.jsonSchemaDialect)}\` |`);
  }
  const servers = (spec?.servers ?? []).map((server) => server?.url).filter(nonEmptyString);
  lines.push(`| Servers | ${markdownTableCell(servers.join(', ') || 'none')} |`, '');
  if (spec?.['x-kiln-method-counts'] && typeof spec['x-kiln-method-counts'] === 'object') {
    const counts = Object.entries(spec['x-kiln-method-counts'])
      .map(([method, count]) => `${method} ${count}`)
      .join(', ');
    lines.splice(lines.length - 1, 0, `| Method counts | ${markdownTableCell(counts)} |`);
  }

  const tags = Array.isArray(spec?.tags) ? spec.tags : [];
  for (const tag of tags) {
    const tagged = operations.filter(({ operation }) => operation.tags?.includes(tag.name));
    lines.push(`## ${tag.name}`, '');
    if (nonEmptyString(tag.description)) lines.push(tag.description.trim(), '');
    lines.push(
      '| Method | Path | Summary | Request inputs | Successful response | Owner |',
      '| --- | --- | --- | --- | --- | --- |',
    );
    for (const { path, method, operation } of tagged) {
      const transport = operation['x-kiln-transport'] === 'websocket' ? ' (WebSocket)' : '';
      lines.push(
        `| \`${method}\` | \`${markdownTableCell(path)}\` | ${markdownTableCell(operation.summary)}${transport} | ${markdownTableCell(openApiRequestSummary(operation))} | ${markdownTableCell(openApiResponseSummary(operation))} | \`${markdownTableCell(operation['x-kiln-handler'] ?? 'unknown')}\` |`,
      );
    }
    lines.push('');
  }

  const schemas = spec?.components?.schemas ?? {};
  lines.push('## Payload components', '');
  lines.push(
    'Open components are explicit migration markers, not an assertion that arbitrary fields are accepted by the runtime.',
    '',
    '| Component | Rust type | Shape | Field status |',
    '| --- | --- | --- | --- |',
  );
  for (const [name, schema] of Object.entries(schemas)) {
    let shape = schemaType(schema);
    if (schema?.type === 'object') {
      shape += schema.additionalProperties === false ? '; closed' : '; open';
    }
    const explicitStatus = schema?.['x-kiln-field-schema-status'];
    const status = nonEmptyString(explicitStatus)
      ? explicitStatus.replaceAll('_', ' ')
      : (schema?.type === 'object' && schema.additionalProperties === true
        ? 'migration pending'
        : 'declared');
    lines.push(`| \`${markdownTableCell(name)}\` | \`${markdownTableCell(schema?.['x-kiln-rust-type'] ?? 'unknown')}\` | ${markdownTableCell(shape)} | ${status} |`);
  }
  lines.push('');
  return `${lines.join('\n').trim()}\n`;
}

async function pathIsFile(path) {
  try {
    return (await lstat(path)).isFile();
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

async function pathKind(path) {
  try {
    const info = await lstat(path);
    if (info.isFile()) return 'file';
    if (info.isDirectory()) return 'directory';
    return 'other';
  } catch (error) {
    if (error?.code === 'ENOENT') return 'missing';
    throw error;
  }
}

export async function loadAndValidateManifest({ repoRoot, manifestPath }) {
  let manifest;
  try {
    manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  } catch (error) {
    throw new DocsBuildError(`cannot read docs manifest ${manifestPath}: ${error.message}`);
  }

  const errors = [];
  if (manifest?.version !== 1) errors.push('docs manifest version must be 1');
  if (!nonEmptyString(manifest?.site?.title)) errors.push('site.title must be a non-empty string');
  if (!nonEmptyString(manifest?.site?.base_url)) errors.push('site.base_url must be a non-empty string');
  if (!nonEmptyString(manifest?.site?.repository_url)) errors.push('site.repository_url must be a non-empty string');
  if (nonEmptyString(manifest?.site?.base_url) && manifest.site.base_url.endsWith('/')) {
    errors.push('site.base_url must not end with a slash');
  }
  if (!Array.isArray(manifest?.site?.product_guides)) {
    errors.push('site.product_guides must be an array');
  }
  if (!Array.isArray(manifest?.sections) || manifest.sections.length === 0) {
    errors.push('sections must be a non-empty array');
  }
  if (!Array.isArray(manifest?.documents) || manifest.documents.length === 0) {
    errors.push('documents must be a non-empty array');
  }

  const sectionIds = new Set();
  for (const [index, section] of (manifest?.sections ?? []).entries()) {
    const label = `sections[${index}]`;
    if (!nonEmptyString(section?.id) || !SECTION_PATTERN.test(section.id)) {
      errors.push(`${label}.id must match ${SECTION_PATTERN}`);
    } else if (sectionIds.has(section.id)) {
      errors.push(`duplicate section id ${section.id}`);
    } else {
      sectionIds.add(section.id);
    }
    if (!nonEmptyString(section?.title)) errors.push(`${label}.title must be non-empty`);
  }

  for (const [index, guide] of (manifest?.site?.product_guides ?? []).entries()) {
    const label = `site.product_guides[${index}]`;
    if (!nonEmptyString(guide?.title)) errors.push(`${label}.title must be non-empty`);
    if (!nonEmptyString(guide?.href)) errors.push(`${label}.href must be non-empty`);
    if (!nonEmptyString(guide?.description)) errors.push(`${label}.description must be non-empty`);
  }

  const sources = new Set();
  const slugs = new Set(['index', '_assets']);
  let hasConfiguration = false;
  for (const [index, document] of (manifest?.documents ?? []).entries()) {
    const label = `documents[${index}]`;
    if (!nonEmptyString(document?.source)) {
      errors.push(`${label}.source must be non-empty`);
      continue;
    }
    if (document.source === 'docs/CONFIGURATION.md') hasConfiguration = true;
    if (isAbsolute(document.source) || document.source.includes('\\')) {
      errors.push(`${label}.source must be a repository-relative POSIX path`);
    }
    const segments = document.source.split('/');
    if (segments.includes('..') || segments.includes('.')) {
      errors.push(`${label}.source must not traverse directories`);
    }
    if (segments.some((segment) => EXCLUDED_SOURCE_SEGMENTS.has(segment))) {
      errors.push(`${label}.source points at an excluded internal documentation tree: ${document.source}`);
    }
    const kind = documentKind(document);
    if (!DOCUMENT_KINDS.has(kind)) {
      errors.push(`${label}.kind must be one of ${[...DOCUMENT_KINDS].join(', ')}`);
    }
    const extension = extname(document.source).toLowerCase();
    if (kind === 'markdown' && extension !== '.md') {
      errors.push(`${label}.source must be a Markdown file for kind markdown`);
    }
    if (kind === 'json_schema' && extension !== '.json') {
      errors.push(`${label}.source must be a JSON file for kind json_schema`);
    }
    if (kind === 'openapi' && extension !== '.json') {
      errors.push(`${label}.source must be a JSON file for kind openapi`);
    }
    if (sources.has(document.source)) {
      errors.push(`duplicate document source ${document.source}`);
    } else {
      sources.add(document.source);
    }
    if (!nonEmptyString(document?.slug) || !SLUG_PATTERN.test(document.slug)) {
      errors.push(`${label}.slug must match ${SLUG_PATTERN}`);
    } else if (slugs.has(document.slug)) {
      errors.push(`duplicate or reserved document slug ${document.slug}`);
    } else {
      slugs.add(document.slug);
    }
    if (!nonEmptyString(document?.title)) errors.push(`${label}.title must be non-empty`);
    if (!nonEmptyString(document?.description)) errors.push(`${label}.description must be non-empty`);
    if (!sectionIds.has(document?.section)) {
      errors.push(`${label}.section references unknown section ${JSON.stringify(document?.section)}`);
    }

    const absoluteSource = resolve(repoRoot, document.source);
    if (!isWithin(absoluteSource, repoRoot)) {
      errors.push(`${label}.source escapes the repository: ${document.source}`);
    } else if (!(await pathIsFile(absoluteSource))) {
      errors.push(`${label}.source does not exist or is not a file: ${document.source}`);
    } else if (kind === 'json_schema' || kind === 'openapi') {
      try {
        const parsed = JSON.parse(await readFile(absoluteSource, 'utf8'));
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
          errors.push(`${label}.source must contain a JSON object`);
        } else if (kind === 'json_schema' && !nonEmptyString(parsed.$schema)) {
          errors.push(`${label}.source must declare a non-empty $schema dialect`);
        } else if (kind === 'openapi' && (!nonEmptyString(parsed.openapi) || !parsed.openapi.startsWith('3.1.'))) {
          errors.push(`${label}.source must declare an OpenAPI 3.1 version`);
        } else if (kind === 'openapi' && (!parsed.paths || typeof parsed.paths !== 'object' || Array.isArray(parsed.paths))) {
          errors.push(`${label}.source must declare an OpenAPI paths object`);
        }
      } catch (error) {
        errors.push(`${label}.source is not valid JSON: ${error.message}`);
      }
    }
  }
  if (!hasConfiguration) {
    errors.push('documents must include docs/CONFIGURATION.md');
  }
  if (errors.length > 0) throw new DocsBuildError(errors);
  return manifest;
}

function isExternalReference(href) {
  return /^[A-Za-z][A-Za-z0-9+.-]*:/.test(href) || href.startsWith('//');
}

function isDynamicReference(href) {
  return href.includes('${')
    || href.includes('{{')
    || href.includes('}}')
    || href.includes('<')
    || href.includes('>')
    || href.includes('*')
    || href.includes('…')
    || href.includes('...');
}

function splitReference(href) {
  const hashIndex = href.indexOf('#');
  const beforeHash = hashIndex === -1 ? href : href.slice(0, hashIndex);
  const fragment = hashIndex === -1 ? '' : href.slice(hashIndex + 1);
  const queryIndex = beforeHash.indexOf('?');
  return {
    path: queryIndex === -1 ? beforeHash : beforeHash.slice(0, queryIndex),
    query: queryIndex === -1 ? '' : beforeHash.slice(queryIndex),
    fragment,
  };
}

function safeDecode(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function fragmentSuffix(query, fragment) {
  return `${query}${fragment ? `#${fragment}` : ''}`;
}

function walkInlineTokens(tokens) {
  const children = [];
  for (const token of tokens) {
    if (token.type === 'inline' && token.children) children.push(...token.children);
  }
  return children;
}

async function replaceHtmlReferences(html, resolver) {
  const pattern = /\b(href|src)\s*=\s*(?:"([^"]*)"|'([^']*)')/gi;
  let output = '';
  let cursor = 0;
  for (const match of html.matchAll(pattern)) {
    output += html.slice(cursor, match.index);
    const attribute = match[1].toLowerCase();
    const raw = decodeHtmlAttribute(match[2] ?? match[3] ?? '');
    const rewritten = await resolver(raw, attribute === 'src');
    output += `${match[1]}="${escapeHtml(rewritten)}"`;
    cursor = match.index + match[0].length;
  }
  return output + html.slice(cursor);
}

function renderToc(headings) {
  const visible = headings.filter((heading) => heading.level === 2 || heading.level === 3);
  if (visible.length === 0) return '<p class="docs-toc-empty">No subsections</p>';
  return `<ol>${visible.map((heading) => (
    `<li class="docs-toc-level-${heading.level}"><a href="#${escapeHtml(heading.id)}">${escapeHtml(heading.text)}</a></li>`
  )).join('')}</ol>`;
}

function sectionMap(manifest) {
  return new Map(manifest.sections.map((section) => [section.id, section]));
}

function renderSidebar(manifest, activeSlug, depth) {
  const docsPrefix = depth === 'hub' ? '.' : '..';
  const rootPrefix = depth === 'hub' ? '..' : '../..';
  const sections = manifest.sections.map((section) => {
    const links = manifest.documents
      .filter((document) => document.section === section.id)
      .map((document) => {
        const current = document.slug === activeSlug ? ' aria-current="page"' : '';
        return `<li><a href="${docsPrefix}/${escapeHtml(document.slug)}/"${current}>${escapeHtml(document.title)}</a></li>`;
      })
      .join('');
    return `<section class="docs-sidebar-section"><h2>${escapeHtml(section.title)}</h2><ul>${links}</ul></section>`;
  }).join('');
  return `
    <aside class="docs-sidebar" id="docs-sidebar" aria-label="Documentation navigation">
      <div class="docs-sidebar-inner">
        <a class="docs-sidebar-home" href="${docsPrefix}/">Documentation home</a>
        ${sections}
        <div class="docs-sidebar-product">
          <h2>Product guides</h2>
          <ul>${manifest.site.product_guides.map((guide) => {
            const href = depth === 'hub' ? guide.href : `../${guide.href}`;
            return `<li><a href="${escapeHtml(href)}">${escapeHtml(guide.title)}</a></li>`;
          }).join('')}</ul>
        </div>
        <a class="docs-sidebar-repo" href="${escapeHtml(manifest.site.repository_url)}">GitHub repository</a>
        <a class="docs-sidebar-repo" href="${rootPrefix}/">Kiln home</a>
      </div>
    </aside>`;
}

function renderTopbar(manifest, depth) {
  const rootPrefix = depth === 'hub' ? '..' : '../..';
  const docsPrefix = depth === 'hub' ? '.' : '..';
  return `
    <header class="docs-topbar">
      <a class="docs-brand" href="${rootPrefix}/" aria-label="Kiln home">
        <img src="${rootPrefix}/assets/favicon.svg" width="30" height="30" alt="">
        <span>Kiln</span>
      </a>
      <a class="docs-product-name" href="${docsPrefix}/">Documentation</a>
      <div class="docs-search" data-docs-search>
        <label class="sr-only" for="docs-search-${depth}">Search documentation</label>
        <input id="docs-search-${depth}" type="search" placeholder="Search documentation" autocomplete="off" spellcheck="false">
        <div class="docs-search-results" data-docs-search-results hidden></div>
      </div>
      <button class="docs-menu-button" type="button" data-docs-menu aria-controls="docs-sidebar" aria-expanded="false" aria-label="Menu, open documentation navigation" title="Open navigation">Menu</button>
    </header>`;
}

function pageHead({ manifest, title, description, canonical, depth }) {
  const rootPrefix = depth === 'hub' ? '..' : '../..';
  const socialTitle = `${title} — Kiln Documentation`;
  const socialImage = `${manifest.site.base_url}/assets/og-image-v3.png`;
  const socialImageAlt = 'Kiln — Serve it. Teach it. Watch it get better. OpenAI-compatible inference, live LoRA training, and local evals in one server.';
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)} &mdash; Kiln Documentation</title>
  <meta name="description" content="${escapeHtml(description)}">
  <meta name="theme-color" content="#0a0908">
  <meta name="color-scheme" content="dark">
  <link rel="canonical" href="${escapeHtml(canonical)}">
  <link rel="icon" type="image/svg+xml" href="${rootPrefix}/assets/favicon.svg">
  <link rel="alternate icon" type="image/png" href="${rootPrefix}/assets/logo.png">
  <link rel="apple-touch-icon" href="${rootPrefix}/assets/logo.png">

  <meta property="og:title" content="${escapeHtml(socialTitle)}">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:type" content="website">
  <meta property="og:locale" content="en_US">
  <meta property="og:site_name" content="Kiln">
  <meta property="og:url" content="${escapeHtml(canonical)}">
  <meta property="og:image" content="${escapeHtml(socialImage)}">
  <meta property="og:image:type" content="image/png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="${escapeHtml(socialImageAlt)}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${escapeHtml(socialTitle)}">
  <meta name="twitter:description" content="${escapeHtml(description)}">
  <meta name="twitter:image" content="${escapeHtml(socialImage)}">
  <meta name="twitter:image:alt" content="${escapeHtml(socialImageAlt)}">

  <link rel="stylesheet" href="${rootPrefix}/css/docs.css">
</head>`;
}

function pageFooter({ manifest, depth, source }) {
  const rootPrefix = depth === 'hub' ? '..' : '../..';
  const sourceLink = source
    ? `<a href="${escapeHtml(manifest.site.repository_url)}/blob/main/${encodeRepoPath(source)}">View source</a>`
    : '';
  return `
    <footer class="docs-footer">
      <span>Kiln documentation</span>
      <nav aria-label="Footer">${sourceLink}<a href="${rootPrefix}/">Home</a><a href="${escapeHtml(manifest.site.repository_url)}">GitHub</a></nav>
    </footer>`;
}

function pageEnd(depth) {
  const rootPrefix = depth === 'hub' ? '..' : '../..';
  return `
  <script src="${rootPrefix}/js/docs.js" defer></script>
</body>
</html>\n`;
}

function renderDocumentPage({ manifest, document, section, html, headings, firstH1, previous, next }) {
  const canonical = `${manifest.site.base_url}/docs/${document.slug}/`;
  const titleId = firstH1?.id ?? slugifyHeading(document.title);
  const previousLink = previous
    ? `<a class="docs-pager-link docs-pager-previous" href="../${escapeHtml(previous.slug)}/"><span>Previous</span><strong>${escapeHtml(previous.title)}</strong></a>`
    : '<span></span>';
  const nextLink = next
    ? `<a class="docs-pager-link docs-pager-next" href="../${escapeHtml(next.slug)}/"><span>Next</span><strong>${escapeHtml(next.title)}</strong></a>`
    : '<span></span>';
  return `${pageHead({ manifest, title: document.title, description: document.description, canonical, depth: 'document' })}
<body class="docs-body" data-docs-root="..">
  <a class="skip-link" href="#main-content">Skip to content</a>
  ${renderTopbar(manifest, 'document')}
  <div class="docs-shell">
    ${renderSidebar(manifest, document.slug, 'document')}
    <main class="docs-main" id="main-content">
      <nav class="docs-breadcrumbs" aria-label="Breadcrumb">
        <a href="../../">Home</a><span aria-hidden="true">/</span><a href="../">Docs</a><span aria-hidden="true">/</span><span>${escapeHtml(section.title)}</span>
      </nav>
      <header class="docs-article-header">
        <p class="docs-section-label">${escapeHtml(section.title)}</p>
        <h1 id="${escapeHtml(titleId)}">${escapeHtml(document.title)}</h1>
        <p>${escapeHtml(document.description)}</p>
      </header>
      <article class="docs-article">${html}</article>
      <nav class="docs-pager" aria-label="Adjacent documentation">${previousLink}${nextLink}</nav>
      ${pageFooter({ manifest, depth: 'document', source: document.source })}
    </main>
    <aside class="docs-toc" aria-label="On this page"><div><h2>On this page</h2>${renderToc(headings)}</div></aside>
  </div>
${pageEnd('document')}`;
}

function renderHubPage(manifest) {
  const canonical = `${manifest.site.base_url}/docs/`;
  const sections = manifest.sections.map((section) => {
    const documents = manifest.documents.filter((document) => document.section === section.id);
    return `<section class="docs-directory-section" id="${escapeHtml(section.id)}">
      <h2>${escapeHtml(section.title)}</h2>
      <div class="docs-directory-list">${documents.map((document) => (
        `<a href="./${escapeHtml(document.slug)}/"><strong>${escapeHtml(document.title)}</strong><span>${escapeHtml(document.description)}</span></a>`
      )).join('')}</div>
    </section>`;
  }).join('');
  const guides = manifest.site.product_guides.map((guide) => (
    `<a class="docs-guide" href="${escapeHtml(guide.href)}"><strong>${escapeHtml(guide.title)}</strong><span>${escapeHtml(guide.description)}</span></a>`
  )).join('');
  return `${pageHead({
    manifest,
    title: 'Documentation',
    description: 'Complete serving, configuration, training, evaluation, interoperability, and operations documentation for Kiln.',
    canonical,
    depth: 'hub',
  })}
<body class="docs-body docs-hub-body" data-docs-root=".">
  <a class="skip-link" href="#main-content">Skip to content</a>
  ${renderTopbar(manifest, 'hub')}
  <div class="docs-shell docs-hub-shell">
    ${renderSidebar(manifest, null, 'hub')}
    <main class="docs-main docs-hub" id="main-content">
      <nav class="docs-breadcrumbs" aria-label="Breadcrumb"><a href="../">Home</a><span aria-hidden="true">/</span><span>Docs</span></nav>
      <header class="docs-hub-header">
        <p class="docs-section-label">Reference library</p>
        <h1>Complete Kiln documentation</h1>
        <p>Serving, configuration, training, evaluation, interoperability, integrity, and hardware qualification in one searchable reference.</p>
      </header>
      <section class="docs-product-guides" aria-labelledby="product-guides-heading">
        <h2 id="product-guides-heading">Product guides</h2>
        <div>${guides}</div>
      </section>
      <div class="docs-directory">${sections}</div>
      ${pageFooter({ manifest, depth: 'hub' })}
    </main>
  </div>
${pageEnd('hub')}`;
}

function canonicalUrlsFromHtml(html) {
  const urls = [];
  const pattern = /<link\b[^>]*\brel\s*=\s*(?:"canonical"|'canonical'|canonical)[^>]*>/gi;
  for (const match of html.matchAll(pattern)) {
    const href = match[0].match(/\bhref\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s"'=<>`]+))/i);
    const value = decodeHtmlAttribute(href?.[1] ?? href?.[2] ?? href?.[3] ?? '');
    if (value) urls.push(value);
  }
  return urls;
}

async function listFiles(root, predicate) {
  const files = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name, 'en'));
    for (const entry of entries) {
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (entry.isFile() && predicate(path)) files.push(path);
    }
  }
  await visit(root);
  return files;
}

async function writeSitemap(outDir) {
  const htmlFiles = await listFiles(outDir, (path) => extname(path).toLowerCase() === '.html');
  const urls = new Set();
  for (const path of htmlFiles) {
    for (const url of canonicalUrlsFromHtml(await readFile(path, 'utf8'))) urls.add(url);
  }
  const body = [...urls].sort().map((url) => `  <url><loc>${escapeXml(url)}</loc></url>`).join('\n');
  await writeFile(
    resolve(outDir, 'sitemap.xml'),
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${body}\n</urlset>\n`,
  );
}

function escapeMarkdownLabel(value) {
  return String(value)
    .replaceAll('\\', '\\\\')
    .replaceAll('[', '\\[')
    .replaceAll(']', '\\]');
}

function rawRepositorySourceUrl(manifest, source) {
  const repository = new URL(manifest.site.repository_url);
  const repositoryPath = repository.pathname
    .replace(/^\/+/, '')
    .replace(/\.git$/, '');
  if (repository.hostname !== 'github.com' || repositoryPath.split('/').length !== 2) {
    return `${manifest.site.repository_url}/blob/main/${encodeRepoPath(source)}`;
  }
  return `https://raw.githubusercontent.com/${repositoryPath}/refs/heads/main/${encodeRepoPath(source)}`;
}

function llmsFileEntry(title, url, description) {
  return `- [${escapeMarkdownLabel(title)}](${url}): ${normalizeWhitespace(description)}`;
}

async function writeLlmsTxt(outDir, manifest) {
  const documentBySlug = new Map(manifest.documents.map((document) => [document.slug, document]));
  const documentsFor = (slugs) => slugs
    .map((slug) => documentBySlug.get(slug))
    .filter(Boolean)
    .map((document) => llmsFileEntry(
      document.title,
      rawRepositorySourceUrl(manifest, document.source),
      document.description,
    ));
  const productGuides = manifest.site.product_guides.map((guide) => llmsFileEntry(
    guide.title,
    new URL(guide.href, `${manifest.site.base_url}/docs/`).href,
    guide.description,
  ));
  const coreDocumentation = documentsFor([
    'overview',
    'quickstart-reference',
    'configuration',
    'architecture-reference',
    'grpo',
    'evals',
    'benchmarks',
  ]);
  const machineReadableContracts = documentsFor([
    'http-api',
    'configuration-schema',
    'inference-schema',
    'observability-schema',
    'eval-api-schema',
    'control-plane-api-schema',
  ]);
  const projectDocumentation = documentsFor([
    'security',
    'changelog',
    'contributing',
  ]);
  const sections = [
    '# Kiln',
    '',
    '> Kiln is a pure-Rust, single-GPU server for Qwen3.5-4B that combines OpenAI-compatible inference, live LoRA training, local evals, and strict replay in one process.',
    '',
    'Kiln deliberately targets one model family and one local improvement loop. The server owns inference, SFT, GRPO, OPD, adapter lifecycle, evaluation, receipts, and the dashboard. CUDA, ROCm, Metal, and Vulkan are supported; performance claims are bounded by the published benchmark receipts.',
    '',
    'Prefer the source documents and machine-readable contracts below for implementation details. Use the product guides for task-oriented orientation.',
    '',
    '## Product guides',
    '',
    ...productGuides,
  ];
  if (coreDocumentation.length > 0) {
    sections.push('', '## Core documentation', '', ...coreDocumentation);
  }
  if (machineReadableContracts.length > 0) {
    sections.push('', '## Machine-readable contracts', '', ...machineReadableContracts);
  }
  if (projectDocumentation.length > 0) {
    sections.push('', '## Project and operations', '', ...projectDocumentation);
  }
  sections.push(
    '',
    '## Optional',
    '',
    llmsFileEntry('Complete documentation directory', `${manifest.site.base_url}/docs/`, 'Searchable HTML reference for every published Kiln document and contract.'),
    llmsFileEntry('Documentation search index', `${manifest.site.base_url}/docs/search-index.json`, 'Structured titles, descriptions, headings, and searchable text for all published references.'),
    llmsFileEntry('Sitemap', `${manifest.site.base_url}/sitemap.xml`, 'Complete canonical URL inventory.'),
    llmsFileEntry('Source repository', manifest.site.repository_url, 'Canonical source, issues, releases, and development history.'),
    llmsFileEntry('Latest release', `${manifest.site.repository_url}/releases/latest`, 'Current signed Kiln release and platform artifacts.'),
    '',
  );
  await writeFile(resolve(outDir, 'llms.txt'), `${sections.join('\n')}\n`);
}

async function validateGeneratedHtml(outDir) {
  const errors = [];
  const htmlFiles = await listFiles(outDir, (path) => extname(path).toLowerCase() === '.html');
  const idCache = new Map();

  async function idsFor(path) {
    if (!idCache.has(path)) {
      const html = await readFile(path, 'utf8');
      const ids = new Set();
      for (const match of html.matchAll(/\bid\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s"'=<>`]+))/gi)) {
        ids.add(decodeHtmlAttribute(match[1] ?? match[2] ?? match[3] ?? ''));
      }
      idCache.set(path, ids);
    }
    return idCache.get(path);
  }

  for (const source of htmlFiles) {
    const html = await readFile(source, 'utf8');
    const tags = html.matchAll(/<(?:a|area|audio|iframe|img|link|script|source|track|video)\b[^>]*>/gi);
    for (const tag of tags) {
      const references = tag[0].matchAll(/\b(?:href|src)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/gi);
      for (const match of references) {
        const href = decodeHtmlAttribute(match[1] ?? match[2] ?? match[3] ?? '').trim();
        if (!href || href.startsWith('/') || href.startsWith('#') || isExternalReference(href) || isDynamicReference(href)) continue;
        const { path: pathPart, fragment } = splitReference(href);
        let target = resolve(dirname(source), safeDecode(pathPart));
        const kind = await pathKind(target);
        if (kind === 'directory') target = resolve(target, 'index.html');
        if ((await pathKind(target)) !== 'file') {
          errors.push(`${asPosix(relative(outDir, source))}: broken generated reference ${href}`);
          continue;
        }
        if (fragment && extname(target).toLowerCase() === '.html') {
          const decoded = safeDecode(fragment);
          if (!(await idsFor(target)).has(decoded)) {
            errors.push(`${asPosix(relative(outDir, source))}: missing generated anchor #${decoded} in ${asPosix(relative(outDir, target))}`);
          }
        }
      }
    }
  }
  if (errors.length > 0) throw new DocsBuildError(errors);
}

async function validateOutputDestination({ outDir, repoRoot, siteSourceDir }) {
  if (dirname(outDir) === outDir) {
    throw new DocsBuildError('output directory must not be a filesystem root');
  }
  if (outDir === repoRoot || isWithin(repoRoot, outDir)) {
    throw new DocsBuildError('output directory must not be the repository or one of its ancestors');
  }
  if (isWithin(outDir, siteSourceDir) || isWithin(siteSourceDir, outDir)) {
    throw new DocsBuildError('output directory must not overlap docs/site');
  }
  const kind = await pathKind(outDir);
  if (kind === 'file' || kind === 'other') {
    throw new DocsBuildError('output path exists and is not a directory');
  }
  if (kind === 'directory') {
    const entries = await readdir(outDir);
    if (entries.length > 0 && !entries.includes(OUTPUT_MARKER)) {
      throw new DocsBuildError(
        `refusing to replace unmanaged non-empty output directory ${outDir}; remove it explicitly before the first build`,
      );
    }
  }
}

export async function buildDocsSite({
  repoRoot,
  siteSourceDir,
  manifestPath,
  outDir,
  validateOnly = false,
}) {
  const resolvedRepoRoot = resolve(repoRoot);
  const resolvedSiteSource = resolve(siteSourceDir);
  const resolvedOut = outDir ? resolve(outDir) : null;
  if (resolvedOut) {
    await validateOutputDestination({
      outDir: resolvedOut,
      repoRoot: resolvedRepoRoot,
      siteSourceDir: resolvedSiteSource,
    });
  }

  const manifest = await loadAndValidateManifest({ repoRoot: resolvedRepoRoot, manifestPath });
  const markdown = createMarkdown();
  const sections = sectionMap(manifest);
  const documentBySource = new Map(manifest.documents.map((document) => [document.source, document]));
  const analysisCache = new Map();
  const assetCopies = new Map();
  const renderedDocuments = [];

  async function analyze(source) {
    if (analysisCache.has(source)) return analysisCache.get(source);
    const sourcePath = resolve(resolvedRepoRoot, source);
    const sourceRaw = await readFile(sourcePath, 'utf8');
    const document = documentBySource.get(source);
    let raw = sourceRaw;
    const kind = documentKind(document ?? {});
    if (kind === 'json_schema') {
      let schema;
      try {
        schema = JSON.parse(sourceRaw);
      } catch (error) {
        throw new DocsBuildError(`${source}: cannot render JSON Schema: ${error.message}`);
      }
      raw = renderJsonSchemaMarkdown(schema, document);
    } else if (kind === 'openapi') {
      let spec;
      try {
        spec = JSON.parse(sourceRaw);
      } catch (error) {
        throw new DocsBuildError(`${source}: cannot render OpenAPI: ${error.message}`);
      }
      raw = renderOpenApiMarkdown(spec, document);
    }
    const tokens = markdown.parse(raw, {});
    const headingData = decorateHeadings(tokens, raw);
    const analysis = {
      raw,
      anchors: headingData.anchors,
      headings: headingData.headings,
      firstH1: headingData.firstH1,
      text: searchableText(tokens),
    };
    analysisCache.set(source, analysis);
    return analysis;
  }

  async function validateAnchor(source, fragment, fromSource, originalHref) {
    if (!fragment || /^\d+$/.test(fragment)) return;
    const decoded = safeDecode(fragment);
    const target = await analyze(source);
    if (!target.anchors.has(decoded)) {
      throw new DocsBuildError(`${fromSource}: broken Markdown anchor ${originalHref} (missing #${decoded} in ${source})`);
    }
  }

  async function rewriteReference(originalHref, fromDocument, image = false) {
    const href = originalHref.trim();
    if (!href || isExternalReference(href) || href.startsWith('/') || isDynamicReference(href)) return originalHref;
    const parts = splitReference(href);
    if (!parts.path) {
      await validateAnchor(fromDocument.source, parts.fragment, fromDocument.source, originalHref);
      return originalHref;
    }
    const decodedPath = safeDecode(parts.path);
    const targetPath = resolve(dirname(resolve(resolvedRepoRoot, fromDocument.source)), decodedPath);
    if (!isWithin(targetPath, resolvedRepoRoot)) {
      throw new DocsBuildError(`${fromDocument.source}: local reference escapes the repository: ${originalHref}`);
    }
    const kind = await pathKind(targetPath);
    if (kind === 'missing' || kind === 'other') {
      throw new DocsBuildError(`${fromDocument.source}: broken local Markdown reference ${originalHref}`);
    }
    const targetSource = asPosix(relative(resolvedRepoRoot, targetPath));
    if (kind === 'file' && extname(targetPath).toLowerCase() === '.md') {
      await validateAnchor(targetSource, parts.fragment, fromDocument.source, originalHref);
      const published = documentBySource.get(targetSource);
      if (published) {
        if (published.slug === fromDocument.slug) return fragmentSuffix(parts.query, parts.fragment) || './';
        return `../${published.slug}/${fragmentSuffix(parts.query, parts.fragment)}`;
      }
    }
    if (image && kind === 'file') {
      const destination = `_assets/${targetSource}`;
      assetCopies.set(targetPath, destination);
      return `../${encodeRepoPath(destination)}${fragmentSuffix(parts.query, parts.fragment)}`;
    }
    const route = kind === 'directory' ? 'tree' : 'blob';
    return `${manifest.site.repository_url}/${route}/main/${encodeRepoPath(targetSource)}${fragmentSuffix(parts.query, parts.fragment)}`;
  }

  for (const [index, document] of manifest.documents.entries()) {
    const analysis = await analyze(document.source);
    let tokens = markdown.parse(analysis.raw, {});
    const headingData = decorateHeadings(tokens, analysis.raw);
    for (const token of walkInlineTokens(tokens)) {
      if (token.type === 'link_open') {
        token.attrSet('href', await rewriteReference(token.attrGet('href') ?? '', document, false));
      } else if (token.type === 'image') {
        token.attrSet('src', await rewriteReference(token.attrGet('src') ?? '', document, true));
      }
    }
    for (const token of tokens) {
      if (token.type === 'html_inline' || token.type === 'html_block') {
        token.content = await replaceHtmlReferences(
          token.content,
          (href, image) => rewriteReference(href, document, image),
        );
      }
    }
    tokens = stripFirstH1(tokens, headingData.firstH1);
    const rendered = markdown.renderer.render(tokens, markdown.options, {});
    renderedDocuments.push({
      document,
      html: rendered,
      headings: headingData.headings.filter((heading) => heading.level !== 1),
      firstH1: headingData.firstH1,
      searchText: analysis.text,
      index,
    });
  }

  if (validateOnly) {
    return { manifest, documentCount: manifest.documents.length, assetCount: assetCopies.size };
  }
  if (!resolvedOut) throw new DocsBuildError('an output directory is required unless --validate-only is used');

  await mkdir(dirname(resolvedOut), { recursive: true });
  const buildOut = await mkdtemp(resolve(dirname(resolvedOut), `.${basename(resolvedOut)}.kiln-docs-build-`));
  let published = false;
  try {
    await cp(resolvedSiteSource, buildOut, { recursive: true, force: true });
    await mkdir(resolve(buildOut, 'docs'), { recursive: true });

    for (const entry of renderedDocuments) {
      const previous = manifest.documents[entry.index - 1] ?? null;
      const next = manifest.documents[entry.index + 1] ?? null;
      const html = renderDocumentPage({
        manifest,
        document: entry.document,
        section: sections.get(entry.document.section),
        html: entry.html,
        headings: entry.headings,
        firstH1: entry.firstH1,
        previous,
        next,
      });
      const directory = resolve(buildOut, 'docs', entry.document.slug);
      await mkdir(directory, { recursive: true });
      await writeFile(resolve(directory, 'index.html'), html);
    }

    await writeFile(resolve(buildOut, 'docs', 'index.html'), renderHubPage(manifest));
    const searchIndex = [
      ...manifest.site.product_guides.map((guide) => ({
        kind: 'product_guide',
        url: guide.href,
        title: guide.title,
        description: guide.description,
        section: 'Product guides',
        headings: [],
        content: `${guide.title} ${guide.description}`,
      })),
      ...renderedDocuments.map((entry) => ({
        kind: 'reference',
        url: `./${entry.document.slug}/`,
        slug: entry.document.slug,
        title: entry.document.title,
        description: entry.document.description,
        section: sections.get(entry.document.section).title,
        headings: entry.headings.map((heading) => heading.text),
        content: entry.searchText,
      })),
    ];
    await writeFile(
      resolve(buildOut, 'docs', 'search-index.json'),
      `${JSON.stringify(searchIndex, null, 2)}\n`,
    );

    for (const [source, destination] of [...assetCopies.entries()].sort((left, right) => left[1].localeCompare(right[1], 'en'))) {
      const target = resolve(buildOut, 'docs', destination);
      await mkdir(dirname(target), { recursive: true });
      await copyFile(source, target);
    }

    await writeSitemap(buildOut);
    await writeLlmsTxt(buildOut, manifest);
    await validateGeneratedHtml(buildOut);
    await writeFile(resolve(buildOut, OUTPUT_MARKER), 'kiln-docs-site-output-v1\n');
    if ((await pathKind(resolvedOut)) === 'directory') {
      await rm(resolvedOut, { recursive: true, force: true });
    }
    await rename(buildOut, resolvedOut);
    published = true;
  } finally {
    if (!published) await rm(buildOut, { recursive: true, force: true });
  }
  return { manifest, documentCount: manifest.documents.length, assetCount: assetCopies.size };
}

export function isPublishedImagePath(path) {
  return IMAGE_EXTENSIONS.has(extname(path).toLowerCase());
}
