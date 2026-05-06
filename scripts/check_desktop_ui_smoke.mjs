#!/usr/bin/env node
import fs from 'node:fs';

const files = {
  dashboard: 'desktop/ui/dashboard.html',
  settings: 'desktop/ui/settings.html',
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
  assertContains(html, quickstartHref, 'settings setup help');
  assertContains(html, troubleshootingHref, 'settings setup help');
  assertContains(text, 'quickstart', 'settings setup help text');
  assertContains(text, 'troubleshooting', 'settings setup help text');
  assertContains(text, 'model path', 'settings setup help copy');
  assertAny(text, ['server binary', 'kiln binary'], 'settings binary setup copy');
  assertNoForbiddenPublicityCopy(text, 'settings setup help');
}

checkDashboard(read(files.dashboard));
checkSettings(read(files.settings));
console.log('Desktop UI smoke checks passed');
