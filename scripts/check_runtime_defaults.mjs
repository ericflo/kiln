#!/usr/bin/env node

import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const files = {
  contract: 'contracts/runtime-defaults-v1.json',
  serverConfig: 'crates/kiln-server/src/config.rs',
  serverCli: 'crates/kiln-server/src/cli.rs',
  evalCli: 'crates/kiln-server/src/bin/kiln_eval_cli.rs',
  agentRuns: 'crates/kiln-server/src/agent_runs.rs',
  terminal: 'crates/kiln-server/src/api/terminal.rs',
  desktopRust: 'desktop/src/runtime_defaults.rs',
  desktopSettings: 'desktop/src/settings.rs',
  desktopSupervisor: 'desktop/src/supervisor.rs',
  desktopJs: 'desktop/ui/_kiln-runtime-defaults.js',
  desktopDashboard: 'desktop/ui/dashboard.html',
  desktopSettingsUi: 'desktop/ui/settings.html',
  desktopReadme: 'desktop/README.md',
  exampleConfig: 'kiln.example.toml',
  readme: 'README.md',
  quickstart: 'QUICKSTART.md',
  siteQuickstart: 'docs/site/quickstart.html',
  phase2Validation: 'scripts/phase2_validation_steps_1_2_3.sh',
};

function read(path) {
  return readFileSync(path, 'utf8');
}

function requireText(source, expected, label) {
  if (!source.includes(expected)) {
    throw new Error(`${label} is missing ${JSON.stringify(expected)}`);
  }
}

function rejectText(source, forbidden, label) {
  if (source.includes(forbidden)) {
    throw new Error(`${label} still contains ${JSON.stringify(forbidden)}`);
  }
}

const contract = JSON.parse(read(files.contract));
if (contract.contract_version !== 1 || !contract.server) {
  throw new Error('runtime-defaults contract is missing the v1 server record');
}
const { bind_host: bindHost, client_host: clientHost, port } = contract.server;
if (bindHost !== '127.0.0.1' || clientHost !== 'localhost' || port !== 8420) {
  throw new Error(`unexpected runtime-defaults v1 server record: ${JSON.stringify(contract.server)}`);
}

const serverConfig = read(files.serverConfig);
requireText(serverConfig, `pub const DEFAULT_SERVER_HOST: &str = "${bindHost}";`, 'server bind default');
requireText(serverConfig, `pub const DEFAULT_SERVER_CLIENT_HOST: &str = "${clientHost}";`, 'server client default');
requireText(serverConfig, `pub const DEFAULT_SERVER_PORT: u16 = ${port};`, 'server port default');
requireText(serverConfig, 'host: DEFAULT_SERVER_HOST.into()', 'server config default');
requireText(serverConfig, 'port: DEFAULT_SERVER_PORT', 'server config default');

const serverCli = read(files.serverCli);
const cliDefaultCount = (serverCli.match(/default_value_t = default_server_url\(\)/g) || []).length;
if (cliDefaultCount !== 15) {
  throw new Error(`server CLI has ${cliDefaultCount} centralized URL defaults; expected 15`);
}
rejectText(serverCli, 'default_value = "http://localhost:', 'server CLI');

const evalCli = read(files.evalCli);
requireText(evalCli, 'default_value_t = default_server_url()', 'eval CLI');
rejectText(evalCli, 'const DEFAULT_SERVER_URL', 'eval CLI');

requireText(read(files.agentRuns), 'crate::config::DEFAULT_SERVER_PORT', 'embedded agent fallback');
requireText(read(files.terminal), 'crate::config::DEFAULT_SERVER_PORT', 'terminal fallback');

const desktopRust = read(files.desktopRust);
requireText(desktopRust, `pub const DEFAULT_SERVER_HOST: &str = "${bindHost}";`, 'desktop Rust bind default');
requireText(desktopRust, `pub const DEFAULT_SERVER_PORT: u16 = ${port};`, 'desktop Rust port default');
requireText(read(files.desktopSettings), 'port: DEFAULT_SERVER_PORT', 'desktop settings default');
requireText(read(files.desktopSupervisor), 'port: DEFAULT_SERVER_PORT', 'desktop supervisor default');

const jsContext = vm.createContext({});
vm.runInContext(read(files.desktopJs), jsContext);
const jsDefaults = jsContext.KILN_RUNTIME_DEFAULTS;
if (jsDefaults?.serverHost !== bindHost || jsDefaults?.serverPort !== port) {
  throw new Error(`desktop JavaScript defaults do not match the contract: ${JSON.stringify(jsDefaults)}`);
}
const expectedDesktopBase = `http://${bindHost}:${port}`;
if (jsDefaults.serverBaseUrl !== expectedDesktopBase || jsDefaults.openAiBaseUrl !== `${expectedDesktopBase}/v1`) {
  throw new Error(`desktop JavaScript URLs do not match ${expectedDesktopBase}`);
}

for (const [label, path] of [
  ['desktop dashboard', files.desktopDashboard],
  ['desktop settings UI', files.desktopSettingsUi],
]) {
  const source = read(path);
  requireText(source, '_kiln-runtime-defaults.js', label);
  requireText(source, 'KILN_RUNTIME_DEFAULTS.serverPort', label);
  rejectText(source, '127.0.0.1:8000/v1', label);
  rejectText(source, '|| 8000', label);
  rejectText(source, '?? 8000', label);
}

requireText(read(files.exampleConfig), `port = ${port}`, 'example config');
requireText(read(files.readme), `| \`server.port\` | \`KILN_PORT\` | ${port} |`, 'README config table');
requireText(read(files.quickstart), `| \`server.port\` | \`KILN_PORT\` | ${port} |`, 'Quickstart config table');
requireText(read(files.readme), '(contracts/runtime-defaults-v1.json)', 'README runtime-defaults link');
requireText(read(files.quickstart), '(contracts/runtime-defaults-v1.json)', 'Quickstart runtime-defaults link');
requireText(read(files.siteQuickstart), `${bindHost}:${port}</code> by default`, 'site quickstart');
requireText(read(files.desktopReadme), '(../contracts/runtime-defaults-v1.json)', 'desktop runtime-defaults link');
requireText(read(files.desktopReadme), `Default server port is \`${port}\``, 'desktop guide');

const phase2Validation = read(files.phase2Validation);
requireText(phase2Validation, `KILN_URL:-http://${clientHost}:${port}`, 'phase 2 validation default URL');
rejectText(phase2Validation, 'http://localhost:8080', 'phase 2 validation script');

console.log(`runtime defaults v${contract.contract_version} passed (${bindHost}:${port})`);
