
/* =====================================================================
   Cmd-K command palette
   ---------------------------------------------------------------------
   Aggregates everything-actionable into one text-searchable list:
   navigations (open page X), actions (run an eval, switch adapter),
   and direct drill-ins (click a job/adapter/suite directly from search).
   The palette stays cheap by reusing the data the dashboard already
   polls — no separate index, no extra requests.
   ===================================================================== */

let cmdkOpen = false;
let cmdkActiveIdx = 0;
let cmdkResultsCache = [];

function openCmdk() {
  cmdkOpen = true;
  cmdkActiveIdx = 0;
  const modal = document.getElementById('cmdk-modal');
  modal.hidden = false;
  // The palette stacks over whatever is open (e.g. ⌘K from inside a drill):
  // the manager keeps the drill's layer + scroll lock and Escape peels the
  // palette off first.
  openModal(modal, { onClose: closeCmdk });
  const input = document.getElementById('cmdk-input');
  input.value = '';
  input.focus();
  renderCmdkResults('');
}
function closeCmdk() {
  cmdkOpen = false;
  const modal = document.getElementById('cmdk-modal');
  modal.hidden = true;
  closeModal(modal);
}

// Build the searchable index from cached state. Cheap to recompute on
// every keystroke — N is at most low hundreds.
function buildCmdkIndex() {
  const items = [];
  // Navigation
  items.push({ kind: 'nav', icon: icon('home'), title: 'Overview',   sub: 'Live stats, recent requests, quick actions', action: () => selectPage('overview') });
  items.push({ kind: 'nav', icon: icon('layers'), title: 'Adapters',   sub: 'Saved LoRAs, upload, merge', action: () => selectPage('adapters') });
  items.push({ kind: 'nav', icon: icon('flask'), title: 'Training',   sub: 'SFT/GRPO queue + submit', action: () => selectPage('training') });
  items.push({ kind: 'nav', icon: icon('chart'), title: 'Evals',      sub: 'Datasets, suites, jobs, judgments', action: () => selectPage('evals') });
  items.push({ kind: 'nav', icon: icon('flask'), title: 'Distill',    sub: 'Teachers, boost, refresh, merge, self-improve', action: () => selectPage('distill') });
  items.push({ kind: 'nav', icon: icon('terminal'), title: 'pi Terminal', sub: 'Run pi against this Kiln, right here', action: () => selectPage('terminal') });
  items.push({ kind: 'nav', icon: icon('chat'), title: 'Playground', sub: 'Quick inference + A/B compare', action: () => selectPage('playground') });
  // Actions
  items.push({ kind: 'action', icon: icon('link'), title: 'Connect your agent', sub: 'Base URL, model id, pi / opencode setup, test connection', action: () => openConnect() });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Run a new eval',   sub: 'Submit a suite against an adapter', action: () => { selectPage('evals'); document.getElementById('evals-tab-suites')?.click(); } });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Train a new SFT adapter', sub: 'Open the SFT submit form', action: () => { selectPage('training'); document.getElementById('training-tab-sft')?.click(); } });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Train a new GRPO adapter', sub: 'Open the GRPO submit form', action: () => { selectPage('training'); document.getElementById('training-tab-grpo')?.click(); } });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Upload a dataset', sub: 'Drop an SFT JSONL', action: () => { selectPage('evals'); document.getElementById('evals-tab-datasets')?.click(); document.getElementById('dataset-name')?.focus(); } });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Create judgment dataset', sub: 'Start the A/B flywheel', action: () => { selectPage('evals'); document.getElementById('evals-tab-judgments')?.click(); document.getElementById('judgment-create-name')?.focus(); } });
  // Adapters
  for (const name of evalAdaptersCache) {
    items.push({
      kind: 'adapter', icon: icon('layers'),
      title: name,
      sub: name === evalActiveAdapter ? 'Adapter · ACTIVE' : 'Adapter',
      action: async () => {
        selectPage('adapters');
        await openAdapterDrillModal(name);
      },
    });
    items.push({
      kind: 'adapter-load', icon: icon('refresh'),
      title: 'Load adapter ' + name,
      sub: 'Switch active LoRA',
      action: async () => {
        try {
          await api('/v1/adapters/load', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
          toast(`Loaded ${name}`, 'ok');
          refreshAdapterDropdowns();
          pollAdapters && pollAdapters();
        } catch (e) { toast('Load failed: ' + e.message, 'err'); }
      },
    });
  }
  // Suites (from the cached jobs list — list endpoint not always loaded yet)
  const suiteNames = new Set();
  for (const j of evalJobsCache) suiteNames.add(j.suite_name);
  for (const name of suiteNames) {
    items.push({
      kind: 'suite', icon: icon('target'),
      title: name,
      sub: 'Eval suite',
      action: async () => {
        selectPage('evals');
        document.getElementById('evals-tab-suites')?.click();
      },
    });
    items.push({
      kind: 'suite-run', icon: icon('play'),
      title: `Run "${name}" vs active adapter`,
      sub: 'Queue an eval immediately',
      action: async () => {
        try {
          const res = await api('/v1/eval/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ suite: name, adapter: evalActiveAdapter || '' }) });
          toast(`Queued eval ${res.job_id.slice(0, 8)} · seed ${res.effective_seed}`, 'ok');
          selectPage('evals');
          document.getElementById('evals-tab-jobs')?.click();
          refreshEvalJobs();
        } catch (e) { toast('Run failed: ' + e.message, 'err'); }
      },
    });
  }
  // Jobs (recent — clickable to drill in)
  for (const j of evalJobsCache.slice(0, 20)) {
    items.push({
      kind: 'job', icon: icon('chart'),
      title: `${j.suite_name}`,
      sub: `Eval · ${j.state} · ${j.headline_accuracy != null ? (j.headline_accuracy*100).toFixed(0)+'%' : '—'} · ${j.job_id.slice(0, 8)}`,
      action: () => { selectPage('evals'); document.getElementById('evals-tab-jobs')?.click(); openDrillModal(j.job_id); },
    });
  }
  // Training runs (running + queued + most-recent completed) — same
  // drill-modal jump as clicking a card on the Training tab. Lets
  // power users find a finished run by adapter name without scrolling.
  const trainingPool = trainingJobsCache ? [
    ...(trainingJobsCache.running ? [trainingJobsCache.running] : []),
    ...(trainingJobsCache.queued || []),
    ...(trainingJobsCache.completed || []).slice(0, 30),
  ] : [];
  for (const j of trainingPool) {
    const stateNorm = (j.state || '').toString().toLowerCase() || 'queued';
    const lossLbl = j.current_loss != null ? `loss ${j.current_loss.toFixed(3)}` : 'no loss yet';
    items.push({
      kind: 'train-job', icon: icon('flask'),
      title: j.adapter_name || j.job_id,
      sub: `${(j.job_type || 'train').toString().toUpperCase()} · ${stateNorm} · ${lossLbl} · ${j.job_id.slice(0, 8)}`,
      action: () => {
        selectPage('training');
        document.getElementById('training-tab-queue')?.click();
        if (typeof openTrainDrillModal === 'function') openTrainDrillModal(j.job_id);
      },
    });
  }
  // Recent requests (last 20) — jump back into the request inspect modal
  // by short id or by prompt content. Most useful for retrieving "what
  // was the prompt that produced that weird answer five minutes ago".
  for (const r of (recentRequestsCache || []).slice(0, 20)) {
    const preview = (r.prompt_preview || '').replace(/\s+/g, ' ').slice(0, 60) || '(no prompt)';
    items.push({
      kind: 'recent-req', icon: icon('arrow-right'),
      title: preview,
      sub: `Request · ${r.streamed ? 'stream' : 'unary'} · ${r.completion_tokens || 0} tok · ${(r.id || '').replace(/^chatcmpl-/, '').slice(0, 8)}`,
      action: () => {
        selectPage('overview');
        if (typeof openRequestDrillModal === 'function') openRequestDrillModal(r.id);
      },
    });
  }
  return items;
}

function renderCmdkResults(query) {
  const items = buildCmdkIndex();
  const q = query.trim().toLowerCase();
  const filtered = !q ? items : items.filter(it => {
    return it.title.toLowerCase().includes(q) || (it.sub && it.sub.toLowerCase().includes(q));
  });
  cmdkResultsCache = filtered;
  cmdkActiveIdx = Math.max(0, Math.min(cmdkActiveIdx, filtered.length - 1));
  const el = document.getElementById('cmdk-results');
  if (!filtered.length) {
    el.innerHTML = `<div class="cmdk-empty">No matches for <code>${escapeHtml(q || '(all)')}</code>.</div>`;
    return;
  }
  // Group by kind label.
  const groups = {
    nav: 'Navigate', action: 'Actions',
    adapter: 'Adapters', 'adapter-load': 'Adapter actions',
    suite: 'Suites', 'suite-run': 'Suite actions',
    job: 'Jobs',
  };
  let html = '';
  let lastGroup = '';
  filtered.forEach((it, i) => {
    const groupLabel = groups[it.kind] || it.kind;
    if (groupLabel !== lastGroup) {
      html += `<div class="cmdk-section-label">${escapeHtml(groupLabel)}</div>`;
      lastGroup = groupLabel;
    }
    html += `<div class="cmdk-item ${i === cmdkActiveIdx ? 'cmdk-active' : ''}" data-cmdk-idx="${i}">
      <span class="cmdk-item-icon">${it.icon || '·'}</span>
      <div class="cmdk-item-body">
        <div class="cmdk-item-title">${escapeHtml(it.title)}</div>
        <div class="cmdk-item-sub">${escapeHtml(it.sub || '')}</div>
      </div>
      <span class="cmdk-item-action">↵</span>
    </div>`;
  });
  el.innerHTML = html;
  el.querySelectorAll('.cmdk-item').forEach(item => {
    item.addEventListener('mouseover', () => {
      cmdkActiveIdx = parseInt(item.dataset.cmdkIdx, 10);
      el.querySelectorAll('.cmdk-item').forEach((other, idx) => other.classList.toggle('cmdk-active', idx === cmdkActiveIdx));
    });
    item.addEventListener('click', () => runCmdkActive());
  });
  // Scroll the active row into view.
  const active = el.querySelector('.cmdk-active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function runCmdkActive() {
  const item = cmdkResultsCache[cmdkActiveIdx];
  if (!item) return;
  closeCmdk();
  // Defer to next tick so any open-modal action sees a clean state.
  setTimeout(() => item.action(), 10);
}

document.getElementById('cmdk-trigger')?.addEventListener('click', openCmdk);
document.getElementById('cmdk-input')?.addEventListener('input', ev => {
  cmdkActiveIdx = 0;
  renderCmdkResults(ev.target.value);
});
document.getElementById('cmdk-modal')?.addEventListener('click', ev => {
  if (ev.target.id === 'cmdk-modal') closeCmdk();
});

// Keyboard cheatsheet — opened with '?'. Lists the shortcuts that already exist
// so power users can discover triage/judging/playground keys without hunting.
function toggleShortcutsSheet() {
  const existing = document.getElementById('shortcuts-modal');
  if (existing) { closeModal(existing); existing.remove(); return; }
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || '');
  const mod = isMac ? '⌘' : 'Ctrl';
  const groups = [
    ['Global', [[[mod + 'K', '/'], 'Command palette'], [['?'], 'This shortcuts list'], [['Esc'], 'Close any modal or palette']]],
    ['Recent requests', [[['Enter', 'Space'], 'Inspect the focused request'], [['←', '→'], 'Previous / next request in the inspector']]],
    ['Eval results drill', [[['/'], 'Search outcomes'], [['r'], 'Re-run the suite'], [['j', 'k'], 'Next / previous outcome']]],
    ['A/B judging', [[['a', 'b'], 'Prefer A / B'], [['t'], 'Tie'], [['s'], 'Skip']]],
    ['Playground', [[['Enter'], 'Send'], [['⇧Enter'], 'Newline'], [['Esc'], 'Stop generating']]],
  ];
  const kbd = keys => keys.map(k => `<kbd>${escapeHtml(k)}</kbd>`).join('<span class="kbd-or">or</span>');
  const body = groups.map(([title, rows]) => `
    <div class="shortcuts-group">
      <div class="shortcuts-group-title">${escapeHtml(title)}</div>
      ${rows.map(([keys, desc]) => `<div class="shortcut-row"><span class="shortcut-keys">${kbd(keys)}</span><span class="shortcut-desc">${escapeHtml(desc)}</span></div>`).join('')}
    </div>`).join('');
  const m = document.createElement('div');
  m.id = 'shortcuts-modal';
  m.className = 'modal-backdrop';
  m.setAttribute('role', 'dialog');
  m.setAttribute('aria-modal', 'true');
  m.setAttribute('aria-label', 'Keyboard shortcuts');
  m.innerHTML = `<div class="modal-shell modal-shell-fit shortcuts-shell" tabindex="-1">
    <div class="modal-head"><h2>Keyboard shortcuts</h2><span style="flex:1 1 auto;"></span>
      <button class="modal-close" id="shortcuts-close" aria-label="Close"><svg class="icn" aria-hidden="true"><use href="#i-close"></use></svg></button></div>
    <div class="shortcuts-body">${body}</div>
  </div>`;
  document.body.appendChild(m);
  // Escape, focus, and the scroll lock come from the shared modal manager.
  const close = () => { closeModal(m); m.remove(); };
  m.querySelector('#shortcuts-close')?.addEventListener('click', close);
  m.addEventListener('click', ev => { if (ev.target === m) close(); });
  openModal(m, { onClose: close });
}

document.addEventListener('keydown', ev => {
  // Open: ⌘K / Ctrl+K (anywhere except inside an input that already has its own handler)
  if ((ev.key === 'k' || ev.key === 'K') && (ev.metaKey || ev.ctrlKey)) {
    ev.preventDefault();
    if (cmdkOpen) closeCmdk(); else openCmdk();
    return;
  }
  // Open: just '/' when nothing else is focused (mirrors GitHub behaviour)
  if (!cmdkOpen && ev.key === '/' && !['INPUT','TEXTAREA','SELECT'].includes((ev.target.tagName||'').toUpperCase())) {
    ev.preventDefault();
    openCmdk();
    return;
  }
  // '?' (Shift+/) opens the keyboard cheatsheet when not typing.
  if (!cmdkOpen && ev.key === '?' && !['INPUT','TEXTAREA','SELECT'].includes((ev.target.tagName||'').toUpperCase()) && !ev.target.isContentEditable) {
    ev.preventDefault();
    toggleShortcutsSheet();
    return;
  }
  if (!cmdkOpen) return;
  // Escape is handled by the shared modal manager (closes the TOP of the
  // stack — the palette when it's frontmost).
  if (ev.key === 'ArrowDown') {
    ev.preventDefault();
    cmdkActiveIdx = Math.min(cmdkResultsCache.length - 1, cmdkActiveIdx + 1);
    renderCmdkResults(document.getElementById('cmdk-input').value);
  } else if (ev.key === 'ArrowUp') {
    ev.preventDefault();
    cmdkActiveIdx = Math.max(0, cmdkActiveIdx - 1);
    renderCmdkResults(document.getElementById('cmdk-input').value);
  } else if (ev.key === 'Enter') {
    ev.preventDefault();
    runCmdkActive();
  }
});
