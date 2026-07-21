
// --- Chat ---
const chatMessages = [];
let chatAbort = null;
let chatGenerating = false;
let servedModelId = null;
// True once the SERVER reported a model id. Until then the Connect panel and
// Playground run on the fallback id, and every successful health poll retries
// /v1/models (cold start: the endpoint 503s or lists nothing while weights
// load). Once resolved, the flag short-circuits — the retry stops for good.
let servedModelIdResolved = false;

async function loadServedModelId() {
  if (servedModelIdResolved) return;
  try {
    const res = await fetch('/v1/models');
    if (!res.ok) return;
    const data = await res.json();
    const id = data?.data?.[0]?.id;
    if (!id) return;
    servedModelId = id;
    servedModelIdResolved = true;
    // Upgrade the copyable snippets / model-id field that rendered with the
    // fallback while weights were still loading.
    applyServedModelId(id);
  } catch {}
}

/* ---------------------------------------------------------------------
   Playground settings persistence

   Sampling controls + system prompt round-trip through localStorage so
   reloading doesn't reset every knob. Conversation history is *not*
   persisted by default — that would surprise people running quick
   tests against different adapters. Use the "Restore last" affordance
   in the empty state to bring back the previous session.
   --------------------------------------------------------------------- */
const PLAYGROUND_SETTINGS_KEY = 'kiln.playground.settings.v1';
const PLAYGROUND_HISTORY_KEY  = 'kiln.playground.history.v1';

function readPlaygroundSettings() {
  try { return JSON.parse(localStorage.getItem(PLAYGROUND_SETTINGS_KEY)) || {}; }
  catch { return {}; }
}

function writePlaygroundSettings(settings) {
  try { localStorage.setItem(PLAYGROUND_SETTINGS_KEY, JSON.stringify(settings)); }
  catch { /* storage full / disabled / private mode */ }
}

function capturePlaygroundSettings() {
  const get = (id) => document.getElementById(id);
  return {
    temperature:        get('chat-temp')?.value ?? '1.0',
    maxTokens:          get('chat-max-tokens')?.value ?? '16384',
    enableThinking:     !!get('chat-enable-thinking')?.checked,
    thinkingBudgetTokensMode: get('chat-thinking-budget-tokens-mode')?.value ?? 'inherit',
    thinkingBudgetTimeMode: get('chat-thinking-budget-time-mode')?.value ?? 'inherit',
    thinkingBudgetTokens: get('chat-thinking-budget-tokens')?.value ?? '',
    thinkingBudgetSeconds: get('chat-thinking-budget-seconds')?.value ?? '',
    preset:             get('chat-preset')?.value ?? 'qwen3-thinking-general',
    topP:               get('chat-top-p')?.value ?? '',
    topK:               get('chat-top-k')?.value ?? '',
    minP:               get('chat-min-p')?.value ?? '',
    presencePenalty:    get('chat-presence-penalty')?.value ?? '',
    frequencyPenalty:   get('chat-frequency-penalty')?.value ?? '',
    repetitionPenalty:  get('chat-repetition-penalty')?.value ?? '',
    seed:               get('chat-seed')?.value ?? '',
    stop:               get('chat-stop-sequences')?.value ?? '',
    system:             get('chat-system')?.value ?? '',
    advancedOpen:       !get('chat-advanced')?.hidden,
    compareMode:        !!document.getElementById('chat-compare-toggle')?.checked,
  };
}

function applyPlaygroundSettings(settings) {
  if (!settings || typeof settings !== 'object') return;
  const set = (id, v) => { const el = document.getElementById(id); if (el != null && v != null) el.value = v; };
  set('chat-temp',                settings.temperature);
  // One-shot migration: the old default was '1024', which is too low
  // for thinking-capable models (the reasoning block alone routinely
  // exceeds it and the answer arrives "truncated"). Anyone with
  // '1024' persisted is almost certainly riding the old default
  // rather than having explicitly chosen it, so upgrade to the new
  // 16384 default. Users who really want 1024 can re-set it.
  if (settings.maxTokens && settings.maxTokens !== '1024') {
    set('chat-max-tokens', settings.maxTokens);
  }
  const budgetModes = new Set(['inherit', 'unlimited', 'limit']);
  let tokensMode = settings.thinkingBudgetTokensMode;
  let timeMode = settings.thinkingBudgetTimeMode;
  if (!budgetModes.has(tokensMode) || !budgetModes.has(timeMode)) {
    // Migrate the former combined server/unlimited/custom selector once.
    const legacyMode = settings.thinkingBudgetMode;
    if (legacyMode === 'unlimited') {
      tokensMode = 'unlimited';
      timeMode = 'unlimited';
    } else if (legacyMode === 'custom') {
      tokensMode = settings.thinkingBudgetTokens ? 'limit' : 'unlimited';
      timeMode = settings.thinkingBudgetSeconds ? 'limit' : 'unlimited';
    } else {
      tokensMode = 'inherit';
      timeMode = 'inherit';
    }
  }
  set('chat-thinking-budget-tokens-mode', tokensMode);
  set('chat-thinking-budget-time-mode', timeMode);
  set('chat-thinking-budget-tokens', settings.thinkingBudgetTokens);
  set('chat-thinking-budget-seconds', settings.thinkingBudgetSeconds);
  set('chat-preset',              settings.preset);
  set('chat-top-p',               settings.topP);
  set('chat-top-k',               settings.topK);
  set('chat-min-p',               settings.minP);
  set('chat-presence-penalty',    settings.presencePenalty);
  set('chat-frequency-penalty',   settings.frequencyPenalty);
  set('chat-repetition-penalty',  settings.repetitionPenalty);
  set('chat-seed',                settings.seed);
  set('chat-stop-sequences',      settings.stop);
  set('chat-system',              settings.system);
  const thinking = document.getElementById('chat-enable-thinking');
  if (thinking && typeof settings.enableThinking === 'boolean') thinking.checked = settings.enableThinking;
  syncThinkingBudgetControls();
  const adv = document.getElementById('chat-advanced');
  const advBtn = document.getElementById('chat-toggle-advanced');
  if (adv && advBtn && settings.advancedOpen) {
    adv.hidden = false;
    advBtn.setAttribute('aria-expanded', 'true');
  }
}

/// Apply a Qwen3.5 preset by filling in every sampling input + the
/// thinking toggle. Mirrors the SamplingParams::qwen3_* helpers in
/// kiln-core so the UI shows the same numbers the server would pick if
/// the client sent `"sampling_preset": "..."`.
const QWEN3_PRESETS = {
  'qwen3-thinking-general': {
    temperature: '1.0', topP: '0.95', topK: '20', minP: '0.0',
    presencePenalty: '1.5', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: true,
  },
  'qwen3-thinking-coding': {
    temperature: '0.6', topP: '0.95', topK: '20', minP: '0.0',
    presencePenalty: '0.0', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: true,
  },
  'qwen3-non-thinking-general': {
    temperature: '0.7', topP: '0.8', topK: '20', minP: '0.0',
    presencePenalty: '1.5', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: false,
  },
  'qwen3-non-thinking-reasoning': {
    temperature: '1.0', topP: '0.95', topK: '20', minP: '0.0',
    presencePenalty: '1.5', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: false,
  },
  'greedy': {
    temperature: '0.0', topP: '1.0', topK: '0', minP: '0.0',
    presencePenalty: '0.0', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: true,
  },
};

function applyChatPreset(name) {
  if (name === 'custom') return;
  const preset = QWEN3_PRESETS[name];
  if (!preset) return;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  set('chat-temp',               preset.temperature);
  set('chat-top-p',              preset.topP);
  set('chat-top-k',              preset.topK);
  set('chat-min-p',              preset.minP);
  set('chat-presence-penalty',   preset.presencePenalty);
  set('chat-frequency-penalty',  preset.frequencyPenalty);
  set('chat-repetition-penalty', preset.repetitionPenalty);
  const thinking = document.getElementById('chat-enable-thinking');
  if (thinking) thinking.checked = preset.enableThinking;
  syncThinkingBudgetControls();
  persistPlaygroundSettingsSoon();
}

function persistPlaygroundSettingsSoon() {
  if (persistPlaygroundSettingsSoon._h) clearTimeout(persistPlaygroundSettingsSoon._h);
  persistPlaygroundSettingsSoon._h = setTimeout(() => {
    try {
      readThinkingBudgetRequest({ validateDisabled: true });
    } catch {
      return;
    }
    writePlaygroundSettings(capturePlaygroundSettings());
  }, 200);
}

function parseChatStopSequences(raw) {
  if (!raw) return undefined;
  const parts = String(raw).split(',').map(s => s.trim()).filter(Boolean);
  return parts.length ? parts : undefined;
}

function parseOptionalPositiveInt(raw) {
  if (raw == null || raw === '') return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) return undefined;
  return Math.floor(n);
}

function parseOptionalFloat(raw, { min, max } = {}) {
  if (raw == null || raw === '') return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n)) return undefined;
  if (min != null && n < min) return undefined;
  if (max != null && n > max) return undefined;
  return n;
}

function thinkingBudgetError(message, fieldId) {
  const error = new Error(message);
  error.fieldId = fieldId;
  return error;
}

function strictThinkingBudgetInteger(raw, max) {
  if (!/^\d+$/.test(raw)) return null;
  const value = Number(raw);
  return Number.isSafeInteger(value) && value <= max ? value : null;
}

function strictThinkingBudgetMilliseconds(raw, max) {
  const match = /^(?:(\d+)(?:\.(\d{1,3}))?|\.(\d{1,3}))$/.exec(raw);
  if (!match) return null;

  const wholeSeconds = BigInt(match[1] || '0');
  const fractionalMilliseconds = BigInt((match[2] || match[3] || '').padEnd(3, '0'));
  const milliseconds = wholeSeconds * 1000n + fractionalMilliseconds;
  return milliseconds <= BigInt(max) ? Number(milliseconds) : null;
}

const THINKING_BUDGET_TOKEN_MAX = 131_072;
const THINKING_BUDGET_TIME_MS_MAX = 86_400_000;
let playgroundThinkingBudgetDefaults = {
  loaded: false,
  loading: false,
  error: null,
  tokens: null,
  timeMs: null,
};

function validThinkingBudgetDefault(value) {
  return value === null || (Number.isSafeInteger(value) && value >= 0);
}

function updatePlaygroundThinkingBudgetDefaults(cfg) {
  const generation = cfg?.generation;
  const hasTokens = generation
    && Object.prototype.hasOwnProperty.call(generation, 'default_thinking_budget_tokens');
  const hasTime = generation
    && Object.prototype.hasOwnProperty.call(generation, 'default_thinking_budget_ms');
  const tokens = generation?.default_thinking_budget_tokens;
  const timeMs = generation?.default_thinking_budget_ms;
  if (!hasTokens || !hasTime || !validThinkingBudgetDefault(tokens) || !validThinkingBudgetDefault(timeMs)) {
    playgroundThinkingBudgetDefaults = {
      ...playgroundThinkingBudgetDefaults,
      loaded: false,
      loading: false,
      error: 'Server defaults unavailable',
    };
    renderThinkingBudgetPreview();
    return;
  }
  playgroundThinkingBudgetDefaults = {
    loaded: true,
    loading: false,
    error: null,
    tokens,
    timeMs,
  };
  renderThinkingBudgetPreview();
}

async function loadPlaygroundThinkingBudgetDefaults(force = false) {
  playgroundThinkingBudgetDefaults.loading = true;
  playgroundThinkingBudgetDefaults.error = null;
  renderThinkingBudgetPreview();
  try {
    await fetchRuntimeConfig(force);
  } catch (error) {
    playgroundThinkingBudgetDefaults = {
      loaded: false,
      loading: false,
      error: (error && error.message) || 'Server defaults unavailable',
      tokens: null,
      timeMs: null,
    };
    renderThinkingBudgetPreview();
  }
}

function thinkingBudgetPreviewDimension(mode, input, kind) {
  if (mode === 'unlimited') return { value: null, source: 'request', invalid: false };
  if (mode === 'limit') {
    const raw = (input?.value || '').trim();
    if (input?.validity?.badInput || !raw) {
      return { label: raw ? 'invalid' : 'required', source: '', invalid: true };
    }
    const value = kind === 'tokens'
      ? strictThinkingBudgetInteger(raw, THINKING_BUDGET_TOKEN_MAX)
      : strictThinkingBudgetMilliseconds(raw, THINKING_BUDGET_TIME_MS_MAX);
    if (value === null) return { label: 'invalid', source: '', invalid: true };
    return { value, source: 'request', invalid: false };
  }
  if (!playgroundThinkingBudgetDefaults.loaded) {
    return {
      label: playgroundThinkingBudgetDefaults.loading ? 'loading...' : 'unavailable',
      source: '',
      invalid: false,
    };
  }
  return {
    value: kind === 'tokens'
      ? playgroundThinkingBudgetDefaults.tokens
      : playgroundThinkingBudgetDefaults.timeMs,
    source: 'server',
    invalid: false,
  };
}

function formatThinkingBudgetPreviewValue(dimension, kind) {
  if (dimension.label) return dimension.label;
  if (dimension.value === null) return 'unlimited';
  if (kind === 'tokens') return dimension.value.toLocaleString();
  if (dimension.value < 1000) return `${dimension.value.toLocaleString()} ms`;
  return `${(dimension.value / 1000).toLocaleString(undefined, { maximumFractionDigits: 3 })} s`;
}

function renderThinkingBudgetPreview() {
  const preview = document.getElementById('chat-thinking-budget-preview');
  if (!preview) return;
  const enabled = document.getElementById('chat-enable-thinking')?.checked !== false;
  const tokensMode = document.getElementById('chat-thinking-budget-tokens-mode')?.value || 'inherit';
  const timeMode = document.getElementById('chat-thinking-budget-time-mode')?.value || 'inherit';
  const tokens = thinkingBudgetPreviewDimension(
    tokensMode,
    document.getElementById('chat-thinking-budget-tokens'),
    'tokens',
  );
  const time = thinkingBudgetPreviewDimension(
    timeMode,
    document.getElementById('chat-thinking-budget-seconds'),
    'time',
  );
  const setDimension = (valueId, sourceId, dimension, kind) => {
    const value = document.getElementById(valueId);
    const source = document.getElementById(sourceId);
    if (value) value.textContent = formatThinkingBudgetPreviewValue(dimension, kind);
    if (source) {
      source.textContent = dimension.source;
      source.hidden = !dimension.source;
    }
  };
  setDimension('chat-thinking-budget-preview-tokens', 'chat-thinking-budget-preview-tokens-source', tokens, 'tokens');
  setDimension('chat-thinking-budget-preview-time', 'chat-thinking-budget-preview-time-source', time, 'time');

  const invalid = tokens.invalid || time.invalid;
  const state = document.getElementById('chat-thinking-budget-preview-state');
  let stateText = '';
  if (playgroundThinkingBudgetDefaults.error) {
    stateText = playgroundThinkingBudgetDefaults.loaded ? 'refresh failed' : 'defaults unavailable';
  } else if (!enabled) stateText = 'inactive';
  else if (invalid) stateText = 'incomplete';
  if (state) state.textContent = stateText;
  preview.classList.toggle('is-inactive', !enabled);
  preview.classList.toggle('has-error', enabled && (invalid || !!playgroundThinkingBudgetDefaults.error));
  preview.dataset.state = stateText || 'ready';
  preview.title = playgroundThinkingBudgetDefaults.error || '';

  const retry = document.getElementById('chat-thinking-budget-refresh');
  if (retry) {
    retry.hidden = !playgroundThinkingBudgetDefaults.error;
    retry.disabled = playgroundThinkingBudgetDefaults.loading;
  }
}

function thinkingBudgetInputRaw(input, message, fieldId) {
  if (input?.validity?.badInput) {
    throw thinkingBudgetError(message, fieldId);
  }
  return (input?.value || '').trim();
}

function readThinkingBudgetRequest({ validateDisabled = false } = {}) {
  const thinkingEnabled = document.getElementById('chat-enable-thinking')?.checked !== false;
  if (!thinkingEnabled && !validateDisabled) {
    return { tokensMode: 'inherit', timeMode: 'inherit' };
  }
  const tokensMode = document.getElementById('chat-thinking-budget-tokens-mode')?.value || 'inherit';
  const timeMode = document.getElementById('chat-thinking-budget-time-mode')?.value || 'inherit';
  const tokensInput = document.getElementById('chat-thinking-budget-tokens');
  const secondsInput = document.getElementById('chat-thinking-budget-seconds');
  const tokensMessage = 'Thinking tokens must be a whole number from 0 to 131072.';
  const secondsMessage = 'Thinking seconds must be between 0 and 86400 with at most three decimal places.';
  let tokens;
  if (tokensMode === 'limit') {
    const tokensRaw = thinkingBudgetInputRaw(
      tokensInput,
      tokensMessage,
      'chat-thinking-budget-tokens',
    );
    if (!tokensRaw) throw thinkingBudgetError(tokensMessage, 'chat-thinking-budget-tokens');
    tokens = strictThinkingBudgetInteger(tokensRaw, THINKING_BUDGET_TOKEN_MAX);
    if (tokens === null) {
      throw thinkingBudgetError(
        tokensMessage,
        'chat-thinking-budget-tokens',
      );
    }
  }
  let ms;
  if (timeMode === 'limit') {
    const secondsRaw = thinkingBudgetInputRaw(
      secondsInput,
      secondsMessage,
      'chat-thinking-budget-seconds',
    );
    if (!secondsRaw) throw thinkingBudgetError(secondsMessage, 'chat-thinking-budget-seconds');
    ms = strictThinkingBudgetMilliseconds(secondsRaw, THINKING_BUDGET_TIME_MS_MAX);
    if (ms === null) {
      throw thinkingBudgetError(
        secondsMessage,
        'chat-thinking-budget-seconds',
      );
    }
  }
  return { tokensMode, timeMode, tokens, ms };
}

function readThinkingBudgetRequestOrNotify() {
  try {
    return readThinkingBudgetRequest();
  } catch (error) {
    const field = error?.fieldId && document.getElementById(error.fieldId);
    if (field) field.focus();
    toast(error?.message || 'Invalid thinking budget.', 'err');
    return null;
  }
}

function openChatAdvancedControls() {
  const panel = document.getElementById('chat-advanced');
  const button = document.getElementById('chat-toggle-advanced');
  if (!panel || !button || !panel.hidden) return;
  panel.hidden = false;
  button.setAttribute('aria-expanded', 'true');
}

function syncThinkingBudgetControls({ revealCustom = false } = {}) {
  const enabled = document.getElementById('chat-enable-thinking')?.checked !== false;
  const tokensModeInput = document.getElementById('chat-thinking-budget-tokens-mode');
  const timeModeInput = document.getElementById('chat-thinking-budget-time-mode');
  const tokensMode = tokensModeInput?.value || 'inherit';
  const timeMode = timeModeInput?.value || 'inherit';
  const custom = document.getElementById('chat-thinking-budget-custom');
  const tokensField = document.getElementById('chat-thinking-budget-tokens-field');
  const timeField = document.getElementById('chat-thinking-budget-time-field');
  const tokens = document.getElementById('chat-thinking-budget-tokens');
  const seconds = document.getElementById('chat-thinking-budget-seconds');
  const tokenLimit = tokensMode === 'limit';
  const timeLimit = timeMode === 'limit';
  const hasLimit = tokenLimit || timeLimit;

  if (tokensModeInput) tokensModeInput.disabled = !enabled;
  if (timeModeInput) timeModeInput.disabled = !enabled;
  if (custom) {
    custom.hidden = !hasLimit;
    custom.classList.toggle('is-disabled', !enabled);
    custom.setAttribute('aria-disabled', String(!enabled));
  }
  if (tokensField) tokensField.hidden = !tokenLimit;
  if (timeField) timeField.hidden = !timeLimit;
  if (tokens) tokens.disabled = !enabled || !tokenLimit;
  if (seconds) seconds.disabled = !enabled || !timeLimit;
  if (enabled && hasLimit && revealCustom) openChatAdvancedControls();
  renderThinkingBudgetPreview();
}

function applyThinkingBudgetRequest(body, thinkingBudget) {
  if (thinkingBudget?.tokensMode === 'unlimited') body.thinking_budget_tokens = null;
  if (thinkingBudget?.tokensMode === 'limit') body.thinking_budget_tokens = thinkingBudget.tokens;
  if (thinkingBudget?.timeMode === 'unlimited') body.thinking_budget_ms = null;
  if (thinkingBudget?.timeMode === 'limit') body.thinking_budget_ms = thinkingBudget.ms;
}

if (Object.prototype.hasOwnProperty.call(window, '__kilnThinkingBudgetTest')) {
  window.__kilnThinkingBudgetTest = Object.freeze({
    readRequest: readThinkingBudgetRequest,
    applyRequest: applyThinkingBudgetRequest,
    applySettings: applyPlaygroundSettings,
  });
}

function buildChatRequestBody({ messages, temperature, thinkingBudget }) {
  const body = {
    messages,
    stream: true,
    temperature,
  };
  const maxTokens = parseOptionalPositiveInt(document.getElementById('chat-max-tokens')?.value);
  body.max_tokens = maxTokens || 16384;

  const topP = parseOptionalFloat(document.getElementById('chat-top-p')?.value, { min: 0, max: 1 });
  if (topP !== undefined) body.top_p = topP;
  const topK = parseOptionalPositiveInt(document.getElementById('chat-top-k')?.value);
  if (topK !== undefined) body.top_k = topK; // 0 disables; still send so server doesn't fall back to its default of 20
  const minP = parseOptionalFloat(document.getElementById('chat-min-p')?.value, { min: 0, max: 1 });
  if (minP !== undefined) body.min_p = minP;
  const presencePenalty = parseOptionalFloat(
    document.getElementById('chat-presence-penalty')?.value,
    { min: -2, max: 2 },
  );
  if (presencePenalty !== undefined) body.presence_penalty = presencePenalty;
  const frequencyPenalty = parseOptionalFloat(
    document.getElementById('chat-frequency-penalty')?.value,
    { min: -2, max: 2 },
  );
  if (frequencyPenalty !== undefined) body.frequency_penalty = frequencyPenalty;
  const repetitionPenalty = parseOptionalFloat(
    document.getElementById('chat-repetition-penalty')?.value,
    { min: 0, max: 4 },
  );
  if (repetitionPenalty !== undefined) body.repetition_penalty = repetitionPenalty;
  const seed = parseOptionalPositiveInt(document.getElementById('chat-seed')?.value);
  if (seed !== undefined) body.seed = seed;
  const stop = parseChatStopSequences(document.getElementById('chat-stop-sequences')?.value);
  if (stop) body.stop = stop;

  applyThinkingBudgetRequest(body, thinkingBudget);

  const enableThinking = document.getElementById('chat-enable-thinking');
  if (enableThinking && !enableThinking.checked) {
    body.chat_template_kwargs = { enable_thinking: false };
  }
  return body;
}

function getSystemPromptMessage() {
  const text = (document.getElementById('chat-system')?.value || '').trim();
  return text ? { role: 'system', content: text } : null;
}

function serializableChatMessages() {
  // Strip the volatile streaming-state fields. We persist role +
  // content + reasoning only; on restore the message is "frozen"
  // (non-pending, no timing) but its text is preserved.
  return chatMessages
    .filter(m => m.role !== 'assistant' || (m.content && !m.error))
    .map(m => ({
      role: m.role,
      content: m.content || '',
      reasoning: m.reasoning || '',
      adapter: m.adapter || null,
      temperature: m.temperature ?? null,
      thinkingBudget: m.thinkingBudget || null,
    }));
}

function persistChatHistory() {
  try {
    const slim = serializableChatMessages();
    if (!slim.length) {
      localStorage.removeItem(PLAYGROUND_HISTORY_KEY);
    } else {
      localStorage.setItem(PLAYGROUND_HISTORY_KEY, JSON.stringify({ ts: Date.now(), messages: slim }));
    }
  } catch { /* ignore quota / disabled */ }
}

function readPersistedChatHistory() {
  try { return JSON.parse(localStorage.getItem(PLAYGROUND_HISTORY_KEY)) || null; }
  catch { return null; }
}

function restorePlaygroundHistoryBanner() {
  const stash = readPersistedChatHistory();
  if (!stash || !Array.isArray(stash.messages) || !stash.messages.length) return;
  if (chatMessages.length) return;  // already populated (HMR / re-init)
  const ageMin = Math.max(0, Math.round((Date.now() - (stash.ts || 0)) / 60000));
  const out = document.getElementById('chat-output');
  if (!out) return;
  const banner = document.createElement('div');
  banner.className = 'restore-banner';
  banner.innerHTML = `
    <div>
      <strong>Restore previous chat?</strong>
      <span style="color:var(--text-muted);"> — ${stash.messages.length} message${stash.messages.length === 1 ? '' : 's'}${ageMin ? `, ${ageMin} min ago` : ''}.</span>
    </div>
    <div style="display:flex; gap:6px;">
      <button class="btn btn-sm btn-primary" type="button" data-restore="yes">Restore</button>
      <button class="btn btn-sm" type="button" data-restore="no">Discard</button>
    </div>`;
  out.parentNode.insertBefore(banner, out);
  banner.addEventListener('click', (ev) => {
    const which = ev.target?.dataset?.restore;
    if (!which) return;
    if (which === 'yes') {
      for (const m of stash.messages) {
        chatMessages.push({
          _id: newChatMsgId(),
          role: m.role,
          content: m.content || '',
          reasoning: m.reasoning || '',
          pending: false,
          thinkOpen: false,
          adapter: m.adapter || null,
          temperature: m.temperature ?? null,
          thinkingBudget: m.thinkingBudget || null,
        });
      }
      renderChat();
    } else {
      try { localStorage.removeItem(PLAYGROUND_HISTORY_KEY); } catch {}
    }
    banner.remove();
  });
}

/* ---------------------------------------------------------------------
   Tiny safe-by-construction markdown renderer

   Lives inside the playground because we don't want to ship marked.js
   or any pin-anything for the dashboard. Handles only what assistant
   completions actually emit:

     - Fenced code blocks (``` and ~~~) with optional language tag.
     - Inline `code`.
     - **bold**, *italic*, ~~strike~~.
     - ATX headers (#…######).
     - Unordered / ordered lists.
     - Block quotes (>).
     - Horizontal rules (---).
     - Inline [text](url) links — http/https/relative only.

   Everything else falls through as escaped text, so a model that emits
   raw HTML can't inject anything dangerous. We HTML-escape on the way
   in and only re-introduce tags from a fixed, restricted set.
   --------------------------------------------------------------------- */
function _mdInline(text) {
  // The caller has already HTML-escaped `text`. We now re-introduce a
  // small set of inline tags; each placeholder we emit uses entities so
  // none can be mistaken for re-entrant markdown by a later pass.

  // Inline code: backticked spans. Greedy on inner content but stops at
  // the matching backtick run length (e.g. ``foo`bar`` works).
  text = text.replace(/(`+)([^`]+?)\1/g, (_, ticks, body) =>
    `<code>${body.replace(/\n/g, ' ')}</code>`);

  // Links: [text](url). Allow http(s) and relative paths; reject `javascript:`
  // and other schemes outright so a model can't smuggle XSS through here.
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, href) => {
    const ok = /^(https?:\/\/|\/|\.{1,2}\/|#)/.test(href);
    if (!ok) return m;
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });

  // **bold** and __bold__
  text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');

  // *italic* and _italic_ — narrower than bold so we don't eat **bold**.
  text = text.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:!?)\]]|$)/g, '$1<em>$2</em>');
  text = text.replace(/(^|[\s(])_([^_\n]+)_(?=[\s.,;:!?)\]]|$)/g, '$1<em>$2</em>');

  // ~~strike~~
  text = text.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');
  return text;
}

function renderMarkdown(raw) {
  if (!raw) return '';
  // 1) Escape, but extract fenced code blocks first so their contents
  //    don't get interpreted as inline markdown.
  const fenced = [];
  let fenceSentinel = '__KILN_FENCE__';
  while (raw.includes(fenceSentinel)) fenceSentinel += '_';
  const fenceRe = /```([a-zA-Z0-9_+\-.]*)\n([\s\S]*?)```|~~~([a-zA-Z0-9_+\-.]*)\n([\s\S]*?)~~~/g;
  const withPlaceholders = raw.replace(fenceRe, (_m, lang1, body1, lang2, body2) => {
    const lang = (lang1 || lang2 || '').trim();
    const body = body1 != null ? body1 : body2;
    const idx = fenced.length;
    const placeholder = `${fenceSentinel}${idx}${fenceSentinel}`;
    fenced.push({ lang, body, placeholder });
    return placeholder;
  });
  const escaped = escapeHtml(withPlaceholders);

  // 2) Split into block-level pieces by blank-line runs. Each block is
  //    classified once (header / list / quote / hr / paragraph).
  const blocks = escaped.split(/\n{2,}/);
  const html = blocks.map(block => {
    if (!block.trim()) return '';

    // Fenced-code placeholder: emit verbatim, no inline processing.
    const fencedBlock = fenced.find(item => item.placeholder === block);
    if (fencedBlock) {
      const { lang, body } = fencedBlock;
      const escBody = escapeHtml(body.replace(/\n$/, ''));
      const langAttr = lang ? ` data-lang="${escapeHtml(lang)}"` : '';
      return `<pre class="md-code"${langAttr}><code>${escBody}</code></pre>`;
    }

    // Horizontal rule.
    if (/^---+$/.test(block.trim())) return `<hr>`;

    // ATX header (# …)
    const h = block.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      return `<h${level}>${_mdInline(h[2])}</h${level}>`;
    }

    // Block quote — every line starts with `>`.
    if (/^>/.test(block) && block.split('\n').every(l => /^>\s?/.test(l) || !l.trim())) {
      const inner = block.split('\n').map(l => l.replace(/^>\s?/, '')).join('\n');
      return `<blockquote>${_mdInline(inner).replace(/\n/g, '<br>')}</blockquote>`;
    }

    // Lists: an unordered if every non-empty line matches `- |* `, or
    // ordered if every non-empty line matches `\d+\.`. Mixed → paragraph.
    const lines = block.split('\n');
    if (lines.every(l => !l.trim() || /^\s*[-*]\s+/.test(l))) {
      const items = lines.filter(l => l.trim()).map(l => l.replace(/^\s*[-*]\s+/, ''));
      return `<ul>${items.map(i => `<li>${_mdInline(i)}</li>`).join('')}</ul>`;
    }
    if (lines.every(l => !l.trim() || /^\s*\d+\.\s+/.test(l))) {
      const items = lines.filter(l => l.trim()).map(l => l.replace(/^\s*\d+\.\s+/, ''));
      return `<ol>${items.map(i => `<li>${_mdInline(i)}</li>`).join('')}</ol>`;
    }

    // Plain paragraph. Soft line breaks become <br>.
    return `<p>${_mdInline(block).replace(/\n/g, '<br>')}</p>`;
  }).join('');
  return html;
}

function formatChatDuration(ms) {
  if (ms == null || !Number.isFinite(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
}

function approximateTokenCount(text) {
  if (!text) return 0;
  // Cheap token estimate: ~4 chars/token for English. Good enough for
  // a "tokens/sec" readout that doesn't require a wire-side counter.
  return Math.max(1, Math.round(text.length / 4));
}

function chatTokensPerSec(message) {
  if (!message || !message.durationMs || message.durationMs <= 0) return null;
  const tokens = approximateTokenCount(message.reasoning || '') + approximateTokenCount(message.content || '');
  if (!tokens) return null;
  return (tokens * 1000) / message.durationMs;
}

function thinkingBudgetSummary(outcome) {
  if (!outcome?.applied) return '';
  if (outcome.triggered) {
    const trigger = outcome.trigger === 'tokens'
      ? 'token cap'
      : outcome.trigger === 'time'
        ? 'time cap'
        : 'completion limit';
    return trigger;
  }
  return outcome.closed ? 'natural close' : 'unclosed';
}

function appendCompletionOutcomeStats(stats, message, hasReasoning) {
  const budgetOutcome = !message.pending && !hasReasoning
    ? thinkingBudgetSummary(message.thinkingBudget)
    : '';
  if (budgetOutcome) {
    stats.push(`<span class="stat"><strong>Thinking</strong> ${escapeHtml(budgetOutcome)}</span>`);
  }
  if (!message.pending && message.finishReason && message.finishReason !== 'stop') {
    const kind = message.finishReason === 'length' ? 'truncated' : message.finishReason;
    const cls = message.finishReason === 'length' ? 'stat finish-warn' : 'stat';
    const title = message.finishReason === 'length'
      ? 'Response was cut off — increase Max tokens to let the model finish.'
      : `Generation ended with finish_reason=${message.finishReason}.`;
    stats.push(`<span class="${cls}" title="${escapeHtml(title)}">${icon('warning','icn-sm')} ${escapeHtml(kind)}</span>`);
  }
}

function applyChatCompletionStreamChunk(message, chunk, now = performance.now()) {
  const choice = chunk?.choices?.[0];
  const delta = choice?.delta;
  let changed = false;

  if (choice?.finish_reason) {
    message.finishReason = choice.finish_reason;
    changed = true;
  }
  const thinkingBudget = chunk?.metadata?.thinking_budget;
  if (thinkingBudget && typeof thinkingBudget === 'object') {
    message.thinkingBudget = thinkingBudget;
    changed = true;
  }

  const reasoning = typeof delta?.reasoning_content === 'string'
    ? delta.reasoning_content
    : '';
  const content = typeof delta?.content === 'string' ? delta.content : '';
  if (!reasoning && !content) return changed;

  if (message.firstTokenMs == null) {
    message.firstTokenMs = now;
    message.ttftMs = now - message.startMs;
  }
  if (reasoning) {
    if (message.thinkStartMs == null) message.thinkStartMs = now;
    message.reasoning += reasoning;
  }
  if (content) {
    if (message.thinkStartMs != null && message.thinkEndMs == null) {
      message.thinkEndMs = now;
    }
    if (message.firstContentTokenMs == null) message.firstContentTokenMs = now;
    message.content += content;
  }
  message.pending = true;
  message.lastTokenMs = now;
  message.durationMs = now - message.startMs;
  return true;
}

async function consumeChatCompletionSse(response, message, onUpdate, logLabel) {
  if (!response.body) throw new Error('Streaming response did not include a body.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let sawDone = false;

  const consumeLine = (rawLine) => {
    let line = rawLine;
    if (line.endsWith('\r')) line = line.slice(0, -1);
    if (!line.startsWith('data:')) return;
    let payload = line.slice(5);
    if (payload.startsWith(' ')) payload = payload.slice(1);
    if (payload === '[DONE]') {
      sawDone = true;
      return;
    }
    if (!payload) return;

    let chunk;
    try {
      chunk = JSON.parse(payload);
    } catch (parseErr) {
      console.warn(`[${logLabel}] skipped malformed SSE chunk`, parseErr, payload.slice(0, 120));
      return;
    }
    if (chunk?.error) {
      const error = new Error(chunk.error.message || chunk.error.detail || 'Streaming generation failed.');
      if (chunk.error.code) error.code = chunk.error.code;
      throw error;
    }
    if (applyChatCompletionStreamChunk(message, chunk)) onUpdate(message);
  };

  const drainLines = (flush) => {
    let newline;
    while ((newline = buffer.indexOf('\n')) !== -1) {
      consumeLine(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (sawDone) return;
    }
    if (flush && buffer) {
      consumeLine(buffer);
      buffer = '';
    }
  };

  try {
    while (!sawDone) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        drainLines(true);
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      drainLines(false);
    }
    if (!sawDone) {
      throw new Error('Streaming response ended before the [DONE] sentinel.');
    }
  } finally {
    try { await reader.cancel(); } catch (_) {}
  }
}

function renderAssistantBubble(m) {
  const parts = [];
  const hasReasoning = !!(m.reasoning && m.reasoning.length);
  const hasContent   = !!(m.content   && m.content.length);

  if (hasReasoning) {
    // While content has not started arriving, keep the thinking block
    // open so the user can watch the chain-of-thought stream live.
    // Once content arrives, collapse by default but let the user
    // pin-open if they had it open already.
    const live = m.pending && !hasContent;
    const open = live || m.thinkOpen;
    const summary = (() => {
      if (live) {
        const elapsed = m.thinkStartMs ? formatChatDuration(performance.now() - m.thinkStartMs) : '';
        return `<span class="think-label">Thinking</span>${elapsed ? `<span class="think-meta">· ${escapeHtml(elapsed)}</span>` : ''}`;
      }
      const dur = (m.thinkStartMs && m.thinkEndMs) ? formatChatDuration(m.thinkEndMs - m.thinkStartMs) : null;
      const outcome = thinkingBudgetSummary(m.thinkingBudget);
      return `<span class="think-label">Thought</span>${dur ? `<span class="think-meta">· for ${escapeHtml(dur)}</span>` : ''}${outcome ? `<span class="think-meta">· ${escapeHtml(outcome)}</span>` : ''}`;
    })();
    parts.push(`
      <details class="think-block${live ? ' live' : ''}"${open ? ' open' : ''} data-think-toggle="${escapeHtml(m._id)}">
        <summary>${summary}</summary>
        <div class="think-body">${escapeHtml(m.reasoning)}</div>
      </details>
    `);
  }

  // Main answer body — pending without any content shows a "Generating…"
  // placeholder unless the reasoning block is already live (in which
  // case the chain-of-thought is the visible activity). Finished
  // answers go through the lightweight markdown renderer; still-
  // streaming output stays in a plain <pre> so partial fences and
  // mid-list states don't render as flicker.
  let body = '';
  if (m.error) {
    body = `<div class="err-block">${escapeHtml(m.error)}</div>`;
  } else if (hasContent && m.pending) {
    body = `<pre>${escapeHtml(m.content)}</pre>`;
  } else if (hasContent) {
    body = `<div class="md-body">${renderMarkdown(m.content)}</div>`;
  } else if (m.pending && !hasReasoning) {
    // After ~5 s with no token, swap the bare "Generating…" placeholder
    // for a hint about prompt-processing latency + an explicit Stop
    // reminder. Cheap UX guard against the most common "is it stuck?"
    // moment: long prompts or a cold model where the model is still
    // doing prefill before the first token streams.
    const waited = m.startMs != null ? (performance.now() - m.startMs) : 0;
    if (waited > 5000) {
      body = `<pre style="color:var(--text-muted);">Waiting for first token (${escapeHtml(formatChatDuration(waited))}) — long prompts or a cold model can take several seconds. Use the Stop button if this is hung.</pre>`;
    } else {
      body = `<pre>Generating…</pre>`;
    }
  } else if (m.pending && hasReasoning && !hasContent) {
    body = `<pre style="color:var(--text-muted);font-style:italic;">Drafting answer…</pre>`;
  } else if (!hasContent && !hasReasoning) {
    body = `<pre style="color:var(--text-muted);">(empty response)</pre>`;
  }
  parts.push(body);

  // Per-turn footer with timing + actions. Hidden while we have nothing
  // useful to display yet (no first-token timing and no content).
  const showFoot = !!(m.ttftMs || m.durationMs || (!m.pending && (hasContent || m.error)));
  if (showFoot) {
    const stats = [];
    const tag = (m.adapter || m.adapter === null) && m.temperature != null
      ? `<span class="badge-tag" title="Adapter + sampling temperature for this turn">${escapeHtml(m.adapter || 'base')} · t=${escapeHtml(String(m.temperature))}</span>`
      : '';
    if (tag) stats.push(tag);
    if (m.ttftMs != null)     stats.push(`<span class="stat"><strong>TTFT</strong> ${escapeHtml(formatChatDuration(m.ttftMs))}</span>`);
    if (m.durationMs != null) stats.push(`<span class="stat"><strong>${m.pending ? 'Elapsed' : 'Total'}</strong> ${escapeHtml(formatChatDuration(m.durationMs))}</span>`);
    const tps = chatTokensPerSec(m);
    if (tps != null)          stats.push(`<span class="stat"><strong>~${tps.toFixed(tps >= 100 ? 0 : 1)}</strong> tok/s</span>`);
    appendCompletionOutcomeStats(stats, m, hasReasoning);
    stats.push(`<span class="spacer"></span>`);
    if (!m.pending && m.error) {
      stats.push(`<button class="turn-btn" type="button" data-chat-action="regenerate" data-chat-id="${escapeHtml(m._id)}" title="Retry this request"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-refresh"></use></svg> retry</button>`);
    } else if (!m.pending && hasContent) {
      stats.push(`<button class="turn-btn" type="button" data-chat-action="copy" data-chat-id="${escapeHtml(m._id)}" title="Copy assistant answer"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg> copy</button>`);
      stats.push(`<button class="turn-btn" type="button" data-chat-action="regenerate" data-chat-id="${escapeHtml(m._id)}" title="Regenerate this response"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-refresh"></use></svg> regenerate</button>`);
    } else if (m.pending) {
      stats.push(`<button class="turn-btn" type="button" data-chat-action="stop" title="Stop generation">■ stop</button>`);
    }
    parts.push(`<div class="turn-foot">${stats.join('')}</div>`);
  }

  return parts.join('');
}

/* ---------------------------------------------------------------------
   Auto-scroll behavior

   Snapping to the bottom on every chunk fights the user when they
   scroll up to read an earlier turn. Track "is the viewport pinned to
   the bottom?" right before each re-render and only snap if it was.
   --------------------------------------------------------------------- */
let chatStickToBottom = true;
function captureScrollAffinity() {
  const el = document.getElementById('chat-output');
  if (!el) return;
  // 12 px slack so the user doesn't have to be pixel-perfect.
  chatStickToBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) <= 12;
}
function restoreScrollAffinity() {
  const el = document.getElementById('chat-output');
  if (!el || !chatStickToBottom) return;
  el.scrollTop = el.scrollHeight;
}

function updateChatTurnCount() {
  const el = document.getElementById('chat-turn-count');
  const exportBtn = document.getElementById('chat-export');
  if (!el) return;
  const userTurns = chatMessages.filter(m => m.role === 'user').length;
  const finalAssistant = chatMessages.filter(m => m.role === 'assistant' && !m.pending && m.content).length;
  if (userTurns === 0) {
    el.hidden = true;
    if (exportBtn) exportBtn.disabled = true;
  } else {
    el.hidden = false;
    el.textContent = `${userTurns} turn${userTurns === 1 ? '' : 's'} · ${finalAssistant} reply${finalAssistant === 1 ? '' : 'ies'}`;
    if (exportBtn) exportBtn.disabled = finalAssistant === 0;
  }
}

/* In-place update of a single assistant bubble's contents — replaces
   only the children of the existing `[data-msg-id]` wrapper, so every
   *other* bubble's DOM is left untouched. Streaming chunks call this
   instead of the global `renderChat()` to avoid the relayout-and-
   re-animate storm that re-creating every <div> caused. Falls back to
   a full render when the bubble isn't in the DOM yet (first paint).

   Note: scroll affinity is *not* sampled here. A targeted innerHTML
   update doesn't reset scrollTop, so we'd actively fight the user's
   scroll position if we tried to re-pin. The full `renderChat()` —
   used on user-message push, clear, and turn end — still handles
   sticking to the bottom on layout-changing events. */
function patchAssistantBubble(m) {
  const wrapper = document.querySelector(`.chat-msg.assistant[data-msg-id="${cssEscape(m._id)}"]`);
  if (!wrapper) { renderChat(); return; }
  wrapper.className = `chat-msg assistant${m.pending ? ' pending' : ''}`;
  wrapper.innerHTML = `<div class="role">assistant</div>${renderAssistantBubble(m)}`;
  // If user was pinned to the bottom, follow the growing bubble.
  const out = document.getElementById('chat-output');
  if (out) {
    const slack = out.scrollHeight - out.scrollTop - out.clientHeight;
    if (slack <= 24) out.scrollTop = out.scrollHeight;
  }
  // Refresh footer-derived state without a global render.
  updateCopyChatResponseState();
}

// Tiny CSS.escape polyfill — we control the IDs (alphanumeric + `-`),
// but be defensive against future ID schemes that include CSS-special
// characters like `.` or `:`.
function cssEscape(s) {
  return String(s).replace(/[^a-zA-Z0-9_-]/g, c => '\\' + c);
}

function renderChat() {
  const el = document.getElementById('chat-output');
  if (chatMessages.length === 0) {
    el.innerHTML = `<div class="empty">
      <div style="font-weight:600;color:var(--text);margin-bottom:6px;">Send a message to test inference.</div>
      <div>Quick Inference sends a chat completion to the currently selected adapter, or the <strong>Base model</strong>, using the temperature above.</div>
      <div style="margin-top:var(--space-3);"><button type="button" class="btn btn-sm btn-primary" data-chat-example="Explain Kiln in one sentence.">Try an example prompt</button></div>
      <div style="margin-top:var(--space-3);color:var(--text-3);">Tip: toggle <strong>Compare</strong> (above) to race two adapters side-by-side on the same prompt — the fastest way to eyeball whether a freshly trained adapter actually answers better.</div>
      <div style="margin-top:var(--space-2);">If the server is still starting, check <a href="/health" target="_blank" rel="noopener noreferrer">/health</a> or the <a href="https://ericflo.github.io/kiln/troubleshooting.html" target="_blank" rel="noopener noreferrer">Troubleshooting guide</a>.</div>
    </div>`;
    el.querySelector('[data-chat-example]')?.addEventListener('click', (ev) => {
      const input = document.getElementById('chat-input');
      if (!input) return;
      input.value = ev.currentTarget.dataset.chatExample || '';
      if (typeof autoresizeChatInput === 'function') autoresizeChatInput();
      if (typeof updateChatSendState === 'function') updateChatSendState();
      input.focus();
    });
    updateCopyChatResponseState();
    return;
  }
  captureScrollAffinity();
  el.innerHTML = chatMessages.map(m => {
    if (m.role === 'assistant') {
      return `<div class="chat-msg assistant${m.pending ? ' pending' : ''}" data-msg-id="${escapeHtml(m._id)}">
        <div class="role">assistant</div>
        ${renderAssistantBubble(m)}
      </div>`;
    }
    if (m.role === 'user') {
      // Inline-edit affordance: pencil floats on hover; clicking
      // promotes the row to an editable textarea + save/cancel
      // controls. Save trims downstream messages and re-streams.
      return `<div class="chat-msg user${m._editing ? ' editing' : ''}">
        <div class="role">user</div>
        <pre>${escapeHtml(m.content)}</pre>
        <button class="user-edit-btn" type="button" data-chat-action="edit" data-chat-id="${escapeHtml(m._id)}" title="Edit and resend"><svg class="icn icn-sm"><use href="#i-pencil"></use></svg></button>
        <div class="user-edit-area">
          <textarea class="user-edit-input">${escapeHtml(m.content)}</textarea>
          <div class="user-edit-actions">
            <button class="btn btn-sm" type="button" data-chat-action="edit-cancel" data-chat-id="${escapeHtml(m._id)}">Cancel</button>
            <button class="btn btn-sm btn-primary" type="button" data-chat-action="edit-save" data-chat-id="${escapeHtml(m._id)}">Save & resend</button>
          </div>
        </div>
      </div>`;
    }
    if (m.role === 'system') {
      // System messages aren't typically pushed into chatMessages
      // (we synthesize from the system-prompt textarea), but render
      // defensively in case a future flow drops one in.
      return `<div class="chat-msg system">
        <div class="role">system</div>
        <pre>${escapeHtml(m.content)}</pre>
      </div>`;
    }
    return `<div class="chat-msg ${m.role}">
      <div class="role">${m.role}</div>
      <pre>${escapeHtml(m.content)}</pre>
    </div>`;
  }).join('');
  restoreScrollAffinity();
  updateCopyChatResponseState();
  updateChatTurnCount();
}

function getLatestAssistantResponseText() {
  for (let i = chatMessages.length - 1; i >= 0; i--) {
    const message = chatMessages[i];
    if (message.role === 'assistant' && message.content.trim()) {
      return message.content.trim();
    }
  }

  const output = document.getElementById('chat-output');
  const assistantMessages = output ? output.querySelectorAll('.chat-msg.assistant pre, .msg.assistant') : [];
  for (let i = assistantMessages.length - 1; i >= 0; i--) {
    const text = assistantMessages[i].textContent.trim();
    if (text && text !== 'Generating…') return text;
  }
  return '';
}

function updateCopyChatResponseState() {
  const button = document.getElementById('copy-chat-response');
  if (!button) return;
  button.disabled = !getLatestAssistantResponseText();
}
window.updateCopyChatResponseState = updateCopyChatResponseState;

function updateChatSendState() {
  const input = document.getElementById('chat-input');
  const send = document.getElementById('chat-send');
  if (!input || !send) return;
  send.disabled = chatGenerating || !input.value.trim();
}

function fallbackCopyText(text) {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  try {
    if (!document.execCommand('copy')) throw new Error('copy command failed');
    if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = text;
  } finally {
    textarea.remove();
  }
}

async function copyLatestAssistantResponse() {
  const text = getLatestAssistantResponseText();
  if (!text) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = text;
    } else {
      fallbackCopyText(text);
    }
    toast('Copied response');
  } catch (error) {
    try {
      fallbackCopyText(text);
      toast('Copied response');
    } catch {
      toast('Could not copy response. Select the answer text and copy it manually.', 'err');
    }
  }
}

function setChatGenerating(isGenerating) {
  chatGenerating = isGenerating;
  const send = document.getElementById('chat-send');
  const stop = document.getElementById('chat-stop');
  send.textContent = isGenerating ? 'Generating…' : 'Send';
  updateChatSendState();
  stop.hidden = !isGenerating;
  stop.disabled = !isGenerating;
}

function removeEmptyPendingAssistant() {
  const last = chatMessages[chatMessages.length - 1];
  if (last?.role === 'assistant' && last.pending && !last.content) {
    chatMessages.pop();
  } else if (last?.role === 'assistant') {
    last.pending = false;
  }
}

function formatQuickInferenceError(error) {
  const message = error?.message || String(error || 'Unknown error');
  return [
    '',
    'Quick Inference could not complete this request.',
    `Server error: ${message}`,
    '',
    'Next steps:',
    '1. If kiln serve just started, wait for model startup to finish and try again.',
    '2. Open /health to check whether the server is ready.',
    '3. Check the kiln serve logs for model path or GPU initialization errors.',
    '4. See Troubleshooting: https://ericflo.github.io/kiln/troubleshooting.html',
  ].join('\n');
}

let chatMsgIdCounter = 0;
function newChatMsgId() { return `m${++chatMsgIdCounter}-${Date.now().toString(36)}`; }

function makeAssistantPlaceholder() {
  return {
    _id: newChatMsgId(),
    role: 'assistant',
    content: '',
    reasoning: '',
    pending: true,
    startMs: performance.now(),
    firstTokenMs: null,
    firstContentTokenMs: null,
    lastTokenMs: null,
    thinkStartMs: null,
    thinkEndMs: null,
    thinkOpen: false,
    ttftMs: null,
    durationMs: null,
    error: null,
    aborted: false,
    thinkingBudget: null,
  };
}

async function sendChat() {
  if (chatAbort) return;
  // When compare mode is on, the dedicated A/B handler (wired further
  // down) owns the send. Skipping here prevents the single-side bubble
  // from rendering on top of the side-by-side compare panel.
  if (typeof chatCompareMode !== 'undefined' && chatCompareMode) return;
  const input = document.getElementById('chat-input');
  const tempInput = document.getElementById('chat-temp');
  const msg = input.value.trim();
  if (!msg) return;

  let temp;
  try {
    temp = parseQuickInferenceTemperature(tempInput);
  } catch (error) {
    tempInput.focus();
    toast(error.message, 'err');
    return;
  }
  const thinkingBudget = readThinkingBudgetRequestOrNotify();
  if (!thinkingBudget) return;

  input.value = '';
  autoresizeChatInput();
  updateChatSendState();

  chatMessages.push({ _id: newChatMsgId(), role: 'user', content: msg });
  chatMessages.push(makeAssistantPlaceholder());
  renderChat();
  await streamAssistantTurn(temp, thinkingBudget);
}

async function streamAssistantTurn(temp, thinkingBudget) {
  setChatGenerating(true);

  const adapter = document.getElementById('chat-adapter').value || undefined;
  const assistant = chatMessages[chatMessages.length - 1];

  // Snapshot the request shape on this assistant turn so the per-turn
  // footer + the persisted history can show *what* produced this
  // answer even after the user mutates the controls.
  assistant.adapter = adapter || null;
  assistant.temperature = temp;

  const convo = chatMessages
    .filter(m => m.role !== 'assistant' || (m.content && !m.error))
    .filter(m => m.content || m.role === 'system')
    .map(m => ({ role: m.role, content: m.content }));
  const sys = getSystemPromptMessage();
  const messages = sys ? [sys, ...convo] : convo;

  const body = buildChatRequestBody({ messages, temperature: temp, thinkingBudget });
  if (servedModelId) body.model = servedModelId;
  if (adapter) body.adapter = adapter;

  // Tick the per-turn footer so users see the elapsed counter advance
  // even before the first token lands. 500 ms cadence — fast enough
  // for a visible counter, slow enough to never feel jittery — and we
  // patch only the assistant bubble in place (not a full re-render)
  // so neighboring bubbles don't relayout.
  const tickHandle = setInterval(() => {
    if (!assistant.pending) return;
    assistant.durationMs = performance.now() - assistant.startMs;
    patchAssistantBubble(assistant);
  }, 500);

  try {
    const controller = new AbortController();
    chatAbort = controller;

    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Kiln-Client': 'dashboard' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }

    await consumeChatCompletionSse(
      res,
      assistant,
      () => patchAssistantBubble(assistant),
      'playground',
    );
    assistant.pending = false;
    assistant.durationMs = (assistant.lastTokenMs || performance.now()) - assistant.startMs;
    if (assistant.thinkStartMs != null && assistant.thinkEndMs == null && assistant.content) {
      assistant.thinkEndMs = assistant.lastTokenMs || performance.now();
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      assistant.aborted = true;
      assistant.pending = false;
      // Keep partial output if anything streamed; drop the empty
      // placeholder + its paired user message otherwise so the user
      // doesn't accrue empty turns on rapid stop-clicks. (The original
      // behavior popped only the placeholder, leaving an orphaned
      // user turn that re-sent the same prompt on the next regen.)
      if (!assistant.content && !assistant.reasoning) {
        chatMessages.pop();  // assistant placeholder
      } else {
        assistant.durationMs = (assistant.lastTokenMs || performance.now()) - assistant.startMs;
      }
    } else {
      assistant.pending = false;
      assistant.error = formatQuickInferenceError(e);
    }
  } finally {
    // Order matters here: clear the abort handle *and* the UI flag
    // before anything that could throw (renderChat → renderMarkdown →
    // arbitrary user content), so a render failure can't leave the
    // Send button stuck on "Generating…" forever.
    clearInterval(tickHandle);
    chatAbort = null;
    setChatGenerating(false);
    try { persistChatHistory(); } catch (e) { console.warn('[playground] persistChatHistory threw', e); }
    try { renderChat(); }        catch (e) { console.warn('[playground] renderChat threw', e); }
  }
}

function autoresizeChatInput() {
  const input = document.getElementById('chat-input');
  if (!input || input.tagName !== 'TEXTAREA') return;
  input.style.height = 'auto';
  const next = Math.min(input.scrollHeight, 180);
  input.style.height = next + 'px';
}

async function regenerateAssistantMessage(messageId) {
  if (chatAbort) return;
  const thinkingBudget = readThinkingBudgetRequestOrNotify();
  if (!thinkingBudget) return;
  // Find the assistant message and the chain of user/assistant turns
  // *before* it. We replace it in-place with a fresh placeholder and
  // re-stream, so the user's prior message and the conversation
  // upstream of it stay intact.
  const idx = chatMessages.findIndex(m => m._id === messageId);
  if (idx < 0 || chatMessages[idx].role !== 'assistant') return;
  // Drop the target assistant message and any messages after it; the
  // upstream context (everything before idx, ending in user) is what
  // we want to re-send.
  chatMessages.splice(idx);
  // Push a fresh placeholder and stream against the trimmed history.
  chatMessages.push(makeAssistantPlaceholder());
  renderChat();
  const tempInput = document.getElementById('chat-temp');
  let temp;
  try {
    temp = parseQuickInferenceTemperature(tempInput);
  } catch (error) {
    tempInput.focus();
    toast(error.message, 'err');
    chatMessages.pop();
    renderChat();
    return;
  }
  await streamAssistantTurn(temp, thinkingBudget);
}

function handleChatActionClick(event) {
  const btn = event.target.closest('[data-chat-action]');
  if (!btn) return;
  const action = btn.dataset.chatAction;
  if (action === 'stop') {
    if (chatAbort) chatAbort.abort();
    return;
  }
  const id = btn.dataset.chatId;
  if (action === 'copy') {
    const m = chatMessages.find(x => x._id === id);
    if (!m || !m.content) return;
    const writeText = navigator.clipboard?.writeText
      ? navigator.clipboard.writeText.bind(navigator.clipboard)
      : (text) => { fallbackCopyText(text); return Promise.resolve(); };
    writeText(m.content).then(() => {
      if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = m.content;
      toast('Copied response');
    }).catch(() => {
      try { fallbackCopyText(m.content); toast('Copied response'); }
      catch { toast('Could not copy response.', 'err'); }
    });
    return;
  }
  if (action === 'regenerate') {
    regenerateAssistantMessage(id);
    return;
  }
  if (action === 'edit') {
    const m = chatMessages.find(x => x._id === id);
    if (!m || m.role !== 'user') return;
    m._editing = true;
    renderChat();
    // After re-render, focus the new textarea so the user starts typing.
    const ta = document.querySelector(`.chat-msg.user.editing[data-edit-host="${id}"] .user-edit-input`)
      || document.querySelector('.chat-msg.user.editing .user-edit-input');
    if (ta) { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }
    return;
  }
  if (action === 'edit-cancel') {
    const m = chatMessages.find(x => x._id === id);
    if (!m) return;
    m._editing = false;
    renderChat();
    return;
  }
  if (action === 'edit-save') {
    const m = chatMessages.find(x => x._id === id);
    if (!m || m.role !== 'user') return;
    const host = btn.closest('.chat-msg.user');
    const ta = host?.querySelector('.user-edit-input');
    const next = (ta?.value || '').trim();
    if (!next) {
      toast('Message can not be empty.', 'err');
      return;
    }
    const thinkingBudget = readThinkingBudgetRequestOrNotify();
    if (!thinkingBudget) return;
    if (chatAbort) chatAbort.abort();
    const idx = chatMessages.indexOf(m);
    m.content = next;
    m._editing = false;
    // Drop everything after this user turn — we're re-running from here.
    chatMessages.splice(idx + 1);
    chatMessages.push(makeAssistantPlaceholder());
    renderChat();
    const tempInput = document.getElementById('chat-temp');
    let temp;
    try {
      temp = parseQuickInferenceTemperature(tempInput);
    } catch (error) {
      tempInput.focus();
      toast(error.message, 'err');
      chatMessages.pop();
      renderChat();
      return;
    }
    streamAssistantTurn(temp, thinkingBudget);
    return;
  }
}

function handleThinkToggle(event) {
  // Persist whether the user has the thinking panel pinned-open so the
  // next renderChat() (every streaming chunk) doesn't snap it shut.
  const details = event.target.closest('details.think-block');
  if (!details) return;
  const id = details.dataset.thinkToggle;
  if (!id) return;
  const m = chatMessages.find(x => x._id === id);
  if (m) m.thinkOpen = details.open;
}

document.getElementById('chat-send').addEventListener('click', sendChat);
document.getElementById('chat-input').addEventListener('input', () => {
  autoresizeChatInput();
  updateChatSendState();
});
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
document.querySelectorAll('[data-chat-starter-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    const input = document.getElementById('chat-input');
    input.value = button.dataset.chatStarterPrompt || '';
    autoresizeChatInput();
    updateChatSendState();
    input.focus();
  });
});
document.getElementById('chat-stop').addEventListener('click', () => {
  if (chatAbort) chatAbort.abort();
  updateChatSendState();
});
// Esc aborts the active streaming generation when no modal owns the key.
// Standard chat-app keyboard shortcut — saves a mouse trip to the Stop
// button mid-stream. Any open modal claims Escape via the shared modal
// manager, so this only fires when the modal stack is empty.
document.addEventListener('keydown', (ev) => {
  if (ev.key !== 'Escape') return;
  // Only intervene when chat is actually streaming.
  if (!chatAbort && !chatCompareAbort) return;
  // Don't fight any open modal — Escape there closes the top of the stack.
  if (modalStack.length) return;
  if (chatAbort) { chatAbort.abort(); }
  if (chatCompareAbort) { chatCompareAbort.abort(); }
  ev.preventDefault();
});
document.getElementById('chat-clear').addEventListener('click', () => {
  if (chatAbort) { chatAbort.abort(); chatAbort = null; }
  chatMessages.length = 0;
  setChatGenerating(false);
  updateChatSendState();
  persistChatHistory();
  renderChat();
});
document.getElementById('chat-output').addEventListener('click', handleChatActionClick);
document.getElementById('chat-output').addEventListener('toggle', handleThinkToggle, true);
document.getElementById('copy-chat-response').addEventListener('click', copyLatestAssistantResponse);

/* ---------------------------------------------------------------------
   Conversation export

   Renders the current `chatMessages` as a portable markdown document
   so users can paste it into a PR/Slack/dataset without retyping. We
   include the per-turn adapter+temperature badge so a recipient can
   see what produced each answer.
   --------------------------------------------------------------------- */
function exportChatAsMarkdown() {
  const sys = (document.getElementById('chat-system')?.value || '').trim();
  const lines = [`# Kiln playground transcript`, '', `_Exported ${new Date().toISOString()}_`, ''];
  if (sys) {
    lines.push('## System prompt', '', '```', sys, '```', '');
  }
  for (const m of chatMessages) {
    if (m.role === 'user') {
      lines.push('## User', '', m.content || '_(empty)_', '');
    } else if (m.role === 'assistant') {
      const tag = (m.adapter || m.adapter === null) && m.temperature != null
        ? ` — ${m.adapter || 'base'}, t=${m.temperature}`
        : '';
      lines.push(`## Assistant${tag}`, '');
      if (m.reasoning) {
        lines.push('<details><summary>Thinking</summary>', '', '```', m.reasoning, '```', '', '</details>', '');
      }
      if (m.error) {
        lines.push(`> **Error:** ${m.error}`, '');
      } else {
        lines.push(m.content || '_(empty)_', '');
      }
    }
  }
  const text = lines.join('\n');
  const writeText = navigator.clipboard?.writeText
    ? navigator.clipboard.writeText.bind(navigator.clipboard)
    : (t) => { fallbackCopyText(t); return Promise.resolve(); };
  writeText(text).then(() => {
    if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = text;
    toast('Conversation copied as markdown');
  }).catch(() => {
    try { fallbackCopyText(text); toast('Conversation copied as markdown'); }
    catch { toast('Could not copy conversation.', 'err'); }
  });
}
document.getElementById('chat-export')?.addEventListener('click', exportChatAsMarkdown);

/* Escape inside the chat input stops an in-flight generation without
   stealing focus or wiping the input. Falls through when nothing is
   streaming so the user can still type literal escape sequences. */
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && (chatAbort || chatCompareAbort)) {
    e.preventDefault();
    if (chatAbort) chatAbort.abort();
    if (chatCompareAbort) chatCompareAbort.abort();
  }
});

/* ---------------------------------------------------------------------
   Wire up advanced-settings toggle + persistence

   The Advanced panel and every sampling control round-trip through
   localStorage on input/change so a refresh restores the user's setup.
   --------------------------------------------------------------------- */
const chatAdvBtn = document.getElementById('chat-toggle-advanced');
const chatAdvPanel = document.getElementById('chat-advanced');
if (chatAdvBtn && chatAdvPanel) {
  chatAdvBtn.addEventListener('click', () => {
    const open = chatAdvPanel.hidden;
    chatAdvPanel.hidden = !open;
    chatAdvBtn.setAttribute('aria-expanded', String(open));
    persistPlaygroundSettingsSoon();
  });
}

const PLAYGROUND_SETTING_IDS = [
  'chat-temp', 'chat-max-tokens', 'chat-enable-thinking',
  'chat-thinking-budget-tokens-mode', 'chat-thinking-budget-time-mode',
  'chat-thinking-budget-tokens', 'chat-thinking-budget-seconds',
  'chat-preset',
  'chat-top-p', 'chat-top-k', 'chat-min-p',
  'chat-presence-penalty', 'chat-frequency-penalty', 'chat-repetition-penalty',
  'chat-seed', 'chat-stop-sequences', 'chat-system',
];
PLAYGROUND_SETTING_IDS.forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  const ev = (el.type === 'checkbox' || el.tagName === 'SELECT') ? 'change' : 'input';
  el.addEventListener(ev, persistPlaygroundSettingsSoon);
});

['chat-thinking-budget-tokens-mode', 'chat-thinking-budget-time-mode'].forEach(id => {
  document.getElementById(id)?.addEventListener('change', () => {
    syncThinkingBudgetControls({ revealCustom: true });
  });
});
['chat-thinking-budget-tokens', 'chat-thinking-budget-seconds'].forEach(id => {
  document.getElementById(id)?.addEventListener('input', renderThinkingBudgetPreview);
});
document.getElementById('chat-thinking-budget-refresh')?.addEventListener('click', () => {
  loadPlaygroundThinkingBudgetDefaults(true);
});
const thinkingEnabled = document.getElementById('chat-enable-thinking');
thinkingEnabled?.addEventListener('change', () => {
  syncThinkingBudgetControls();
});

// Wire the preset dropdown: changing it applies the preset's values
// to every advanced sampling input and flips the thinking toggle. Any
// later manual edit silently switches the selector to "custom" so the
// preset isn't lying about what's actually in the form.
const presetSelect = document.getElementById('chat-preset');
if (presetSelect) {
  presetSelect.addEventListener('change', (e) => {
    applyChatPreset(e.target.value);
  });
  const FIELDS_THAT_DESYNC_PRESET = [
    'chat-temp', 'chat-top-p', 'chat-top-k', 'chat-min-p',
    'chat-presence-penalty', 'chat-frequency-penalty', 'chat-repetition-penalty',
  ];
  FIELDS_THAT_DESYNC_PRESET.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      // Only mark custom if the user manually changed *after* page load.
      // We compare current value against the preset's spec.
      const currentPreset = presetSelect.value;
      const preset = QWEN3_PRESETS[currentPreset];
      if (!preset) return;
      const map = {
        'chat-temp': 'temperature', 'chat-top-p': 'topP', 'chat-top-k': 'topK',
        'chat-min-p': 'minP', 'chat-presence-penalty': 'presencePenalty',
        'chat-frequency-penalty': 'frequencyPenalty', 'chat-repetition-penalty': 'repetitionPenalty',
      };
      if (String(el.value) !== String(preset[map[id]])) {
        presetSelect.value = 'custom';
      }
    });
  });
  // The thinking checkbox also affects which preset is "consistent".
  const thinkingEl = document.getElementById('chat-enable-thinking');
  if (thinkingEl) {
    thinkingEl.addEventListener('change', () => {
      const preset = QWEN3_PRESETS[presetSelect.value];
      if (preset && preset.enableThinking !== thinkingEl.checked) {
        presetSelect.value = 'custom';
      }
    });
  }
}

// Restore settings + last conversation on first load. Settings always
// apply; conversation restore prompts a banner so users don't get a
// stale conversation invisibly attached to a fresh request.
applyPlaygroundSettings(readPlaygroundSettings());
loadPlaygroundThinkingBudgetDefaults();
restorePlaygroundHistoryBanner();

document.getElementById('upload-name').addEventListener('input', handleUploadNameInput);
document.getElementById('upload-archive').addEventListener('change', handleUploadArchiveChange);
updateUploadAdapterState();
document.getElementById('sft-output-name').addEventListener('input', updateSftSubmitState);
document.getElementById('sft-examples').addEventListener('input', (e) => { if (e.target.value.trim()) clearTrainingData('sft'); updateSftSubmitState(); });
document.getElementById('grpo-output-name').addEventListener('input', updateGrpoSubmitState);
document.getElementById('grpo-groups').addEventListener('input', (e) => { if (e.target.value.trim()) clearTrainingData('grpo'); updateGrpoSubmitState(); });
updateSftSubmitState();
updateGrpoSubmitState();
updateOpdSubmitState();
document.getElementById('merge-output-name').addEventListener('input', updateMergeButtonState);
document.getElementById('merge-mode').addEventListener('change', updateMergeButtonState);
document.getElementById('merge-density').addEventListener('input', updateMergeButtonState);
updateMergeButtonState();
