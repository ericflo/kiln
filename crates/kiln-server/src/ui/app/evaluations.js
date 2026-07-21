/* =====================================================================
   Evals page — datasets, suites, jobs, judgments (the flywheel)

   This is intentionally one large module: every refresh is content-
   addressed by the active sub-tab and shares one drill-in modal so
   data flows across tabs (suite → run → drill, judgment → adapter
   validate → drill) without losing context.
   ===================================================================== */

function selectEvalsTab(tab, focus = false) {
  const panel = tab.closest('.card');
  panel.querySelectorAll('[role="tab"]').forEach(item => {
    const selected = item === tab;
    item.classList.toggle('active', selected);
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  panel.querySelectorAll('[role="tabpanel"]').forEach(tabPanel => {
    const selected = tabPanel.id === tab.getAttribute('aria-controls');
    tabPanel.classList.toggle('active', selected);
    tabPanel.hidden = !selected;
    if (selected) tabPanel.removeAttribute('inert'); else tabPanel.setAttribute('inert', '');
  });
  if (focus) tab.focus();
  const which = tab.dataset.tab;
  try { localStorage.setItem('kiln.evalsSubTab', which); } catch {}
  // Deep-link hash for the sub-tab — covers clicks, arrow keys, and every
  // programmatic .click() caller (cmdk, quick actions, "View result" toasts).
  pushSubTabHash('evals');
  if (which === 'datasets') refreshDatasets();
  else if (which === 'suites') refreshSuites();
  else if (which === 'jobs') refreshEvalJobs();
  else if (which === 'judgments') refreshJudgments();
}
wireTablist(document.querySelector('[data-evals-tabs]'), {
  onSelect: (tab, { focus }) => selectEvalsTab(tab, focus),
});

// Restore the last visited eval sub-tab so users return to Jobs (or
// Suites / Judgments) instead of always-Datasets after a refresh.
// Hash-suppressed: the no-hash fallback — an explicit hash sub-tab is
// applied after this in the boot route pass and wins.
try {
  const lastEvalsSubTab = localStorage.getItem('kiln.evalsSubTab');
  if (lastEvalsSubTab && lastEvalsSubTab !== 'datasets') {
    const target = document.getElementById(`evals-tab-${lastEvalsSubTab}`);
    if (target) withHashWritesSuppressed(() => selectEvalsTab(target));
  }
} catch {}

let evalAdaptersCache = [];
let evalActiveAdapter = null;
async function refreshAdapterDropdowns() {
  try {
    const d = await api('/v1/adapters');
    evalAdaptersCache = (d.available || []).map(a => a.name);
    evalActiveAdapter = d.active || '';
    const targets = ['judgment-adapter-a', 'judgment-adapter-b', 'compile-judge-adapter'];
    // Rebuild the <option> lists only when the adapter name set changed —
    // this runs on the Evals poll tick, and an unconditional rebuild snaps
    // an open dropdown shut mid-pick. Unchanged options always skip, so a
    // focused select is never rebuilt under the user.
    const optionsKey = 'opts:' + JSON.stringify(evalAdaptersCache);
    const optionsHtml = ['<option value="">Base model</option>']
      .concat(evalAdaptersCache.map(n => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`))
      .join('');
    for (const id of targets) {
      const sel = document.getElementById(id);
      if (!sel) continue;
      const cur = sel.value;
      if (setListHtml(sel, optionsKey, optionsHtml)) {
        // Preserve the user's in-flight selection across the rebuild.
        if (cur && evalAdaptersCache.includes(cur)) sel.value = cur;
      }
    }
  } catch (_) { /* best-effort */ }
}

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function truncate(s, n) {
  s = String(s || '');
  if (s.length <= n) return s;
  return s.slice(0, n) + '…';
}
function fmtPct(x, digits = 1) {
  if (x == null || !isFinite(x)) return '—';
  return (x * 100).toFixed(digits) + '%';
}

/* ---------- Accuracy ring (color graded by score) ---------- */
function ringHtml(accuracy, size = '') {
  const pct = (accuracy != null && isFinite(accuracy)) ? Math.max(0, Math.min(1, accuracy)) : 0;
  // Color gradient: red → orange → green
  let color;
  if (pct >= 0.8) color = 'var(--success-fg)';
  else if (pct >= 0.5) color = 'var(--warning-fg)';
  else if (pct > 0) color = 'var(--danger-fg)';
  else color = 'var(--text-quiet)';
  const sizeClass = size ? ` ${size}` : '';
  return `<span class="acc-ring${sizeClass}" style="--ring-pct:${(pct*100).toFixed(0)}; --ring-color:${color};"><span class="acc-ring-num">${(pct*100).toFixed(0)}</span></span>`;
}

/* ---------- Sparkline (suite history) ---------- */
function sparkSvg(values, width = 64, height = 18) {
  if (!values || values.length < 2) return '';
  const pad = 1;
  const w = width - 2 * pad;
  const h = height - 2 * pad;
  const xs = values.map((_, i) => pad + (i * w) / (values.length - 1));
  const ys = values.map(v => pad + h - Math.max(0, Math.min(1, v)) * h);
  const linePath = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(' ');
  const areaPath = `${linePath} L${xs[xs.length-1].toFixed(1)} ${(pad+h).toFixed(1)} L${xs[0].toFixed(1)} ${(pad+h).toFixed(1)} Z`;
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
    <path class="spark-area" d="${areaPath}"/>
    <path d="${linePath}"/>
  </svg>`;
}

/* ---------- Datasets ---------- */

let activeSynthDataset = null;
let activeSynthManifest = null;
async function refreshDatasets() {
  try {
    const d = await api('/v1/eval/datasets');
    const datasets = d.datasets || [];
    const el = document.getElementById('datasets-list');
    if (!datasets.length) {
      el.className = 'eval-empty';
      // The corrections CTA tracks the basket: enabled the moment a finished
      // correction exists. Key on that count so the 1.5s poll repaints the
      // button state as corrections arrive (or get their ideal answers).
      const corrReady = (typeof correctionsBasket !== 'undefined' && typeof corrTrainable === 'function')
        ? correctionsBasket.filter(corrTrainable).length : 0;
      const corrHint = corrReady > 0
        ? `Turn your ${corrReady} finished correction${corrReady === 1 ? '' : 's'} into a dataset you can build evals from`
        : 'Nothing to build yet — when pi gives a wrong answer, add it to Corrections (Overview page) and write the ideal answer first';
      const wrote = setListHtml(el, 'empty:' + corrReady, `
        <div class="eval-empty-icon"><svg class="icn"><use href="#i-folder"></use></svg></div>
        <div class="eval-empty-title">No datasets yet</div>
        <div class="eval-empty-body">A dataset is a list of conversations — the raw material Kiln turns into eval suites and training runs. Upload your own above, or start with one of these:</div>
        <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
          <button class="eval-empty-cta" type="button" id="use-sample-dataset" title="Adds a small built-in dataset of coding-agent conversations — tool calls, code review, commit messages — so you can try the eval flow without bringing your own data">Try a sample dataset</button>
          <button class="eval-empty-cta" type="button" id="dataset-from-corrections" ${corrReady > 0 ? '' : 'disabled '}title="${escapeHtml(corrHint)}">Build a dataset from your corrections</button>
        </div>`);
      if (wrote) {
        document.getElementById('use-sample-dataset')?.addEventListener('click', ev => uploadSampleDataset(ev.currentTarget));
        document.getElementById('dataset-from-corrections')?.addEventListener('click', ev => buildDatasetFromCorrections(ev.currentTarget));
      }
      return;
    }
    el.className = '';
    // Key on every payload field the rows display: stats covers the
    // role-pattern column, the assistant/tool_calls counts, and the
    // recommendStrategy badge (derived solely from stats).
    const listKey = 'list:' + JSON.stringify(datasets.map(m =>
      [m.name, m.format, m.description, m.num_rows, m.size_bytes, m.split_counts, m.stats]));
    const listHtml = datasets.map(m => {
      const stats = m.stats || {};
      const pattern = (stats.sample_role_patterns || []).slice(0, 1).join(' · ') || '';
      const recommendation = recommendStrategy(stats);
      const splits = m.split_counts || {};
      const splitSummary = `train ${(splits.train || 0).toLocaleString()} · validation ${(splits.validation || 0).toLocaleString()} · holdout ${(splits.holdout || 0).toLocaleString()}`;
      return `<div class="eval-row eval-row-datasets">
        <div>
          <div class="row-title">${escapeHtml(m.name)}</div>
          <div class="row-sub">${escapeHtml(m.format)} · ${escapeHtml(m.description || 'No description')}</div>
        </div>
        <div class="tabular-nums" title="${escapeHtml(splitSummary)}">${m.num_rows.toLocaleString()} rows · ${fmtBytes(m.size_bytes)}<div class="row-sub">T ${splits.train || 0} · V ${splits.validation || 0} · H ${splits.holdout || 0}</div></div>
        <div class="row-sub" title="Detected from the first ${stats.num_assistant_turns ? '1000' : 0} rows">
          ${stats.num_assistant_turns ? stats.num_assistant_turns.toLocaleString() + ' assistant · ' + (stats.num_with_tool_calls || 0) + ' tool_calls' : '—'}
          ${recommendation ? `<div style="margin-top:2px;"><span class="scorer-badge" title="Recommended synthesis strategy">${escapeHtml(recommendation)}</span></div>` : ''}
        </div>
        <div class="row-sub" style="font-family:var(--font-mono);">${escapeHtml(pattern)}</div>
        <div class="row-actions">
          ${m.format === 'sft_chat' ? `<button type="button" class="btn btn-primary btn-sm" data-action="train-sft" data-name="${escapeHtml(m.name)}" title="Open Training with this dataset loaded — one click from here to a queued job">Train SFT →</button>` : ''}
          ${m.format === 'grpo_groups' ? `<button type="button" class="btn btn-primary btn-sm" data-action="train-grpo" data-name="${escapeHtml(m.name)}" title="Open Training with this dataset loaded — one click from here to a queued job">Train GRPO →</button>` : ''}
          <button type="button" class="btn ${m.format === 'raw' ? 'btn-primary ' : ''}btn-sm" data-action="synth" data-name="${escapeHtml(m.name)}">Synthesize eval</button>
          <button type="button" class="btn btn-sm" data-action="del" data-name="${escapeHtml(m.name)}">Delete</button>
        </div>
      </div>`;
    }).join('');
    if (!setListHtml(el, listKey, listHtml)) return; // unchanged — old nodes keep their listeners
    el.querySelectorAll('button[data-action]').forEach(b => {
      const name = b.dataset.name;
      if (b.dataset.action === 'train-sft') {
        b.addEventListener('click', () => trainFromDataset(name, 'sft'));
      } else if (b.dataset.action === 'train-grpo') {
        b.addEventListener('click', () => trainFromDataset(name, 'grpo'));
      } else if (b.dataset.action === 'synth') {
        b.addEventListener('click', () => openSynthPanel(name, datasets.find(item => item.name === name)));
      } else if (b.dataset.action === 'del') {
        b.addEventListener('click', async () => {
          if (!confirm(`Delete dataset "${name}"?`)) return;
          try {
            await api('/v1/eval/datasets/' + encodeURIComponent(name), { method: 'DELETE' });
            toast('Dataset deleted', 'ok');
            refreshDatasets();
          } catch (e) { toast('Delete failed: ' + e.message, 'err'); }
        });
      }
    });
  } catch (e) {
    // Route the failure write through setListHtml too: it stamps an
    // error-specific key, so the post-recovery payload (even an identical
    // empty list) compares unequal and repaints (#1547 regression shape).
    setListHtml(document.getElementById('datasets-list'), 'err:' + e.message,
      `<div class="eval-empty"><div class="eval-empty-title">Failed to load</div><div class="eval-empty-body">${escapeHtml(e.message)}</div></div>`);
  }
}

function recommendStrategy(stats) {
  if (!stats || !stats.num_assistant_turns) return null;
  // Tool-call heavy → tool_call_predict
  const toolFraction = stats.num_with_tool_calls / Math.max(1, stats.num_assistant_turns);
  if (toolFraction > 0.3) return 'tool_call_predict ↘';
  // Multi-turn → every_assistant_turn
  if (stats.avg_messages_per_conv > 8) return 'every_assistant_turn';
  // Otherwise → final_assistant
  return 'final_assistant';
}

function evalAggregationLabel(aggregation) {
  const kind = aggregation?.kind || 'single';
  if (kind === 'single') return 'single';
  const stem = { mean_at_k: 'mean', pass_at_k: 'pass', majority_at_k: 'majority' }[kind] || kind;
  const k = Number(aggregation?.k);
  return `${stem}@${Number.isInteger(k) && k > 0 ? k : '?'}`;
}

function updateSynthAggregationControls() {
  const kind = document.getElementById('synth-aggregation')?.value || 'single';
  const group = document.getElementById('synth-k-group');
  if (group) group.hidden = kind === 'single';
  const temperature = document.getElementById('synth-temperature');
  if (temperature && kind !== 'single' && Number(temperature.value) === 0) temperature.value = '0.7';
}

function openSynthPanel(name, manifest = null) {
  activeSynthDataset = name;
  activeSynthManifest = manifest;
  document.getElementById('synth-dataset-name').textContent = name;
  document.getElementById('synth-suite-name').value = name + '-eval';
  document.getElementById('synth-preview-output').innerHTML = '';
  const source = document.getElementById('synth-source-split');
  const counts = manifest?.split_counts || null;
  if (source) {
    for (const option of source.options) {
      const label = option.value === 'train'
        ? 'Train-set diagnostic'
        : option.value[0].toUpperCase() + option.value.slice(1);
      if (counts) {
        const count = Number(counts[option.value] || 0);
        option.disabled = count === 0;
        option.textContent = `${label} (${count.toLocaleString()})`;
      } else {
        option.disabled = false;
        option.textContent = label;
      }
    }
    source.value = !counts || counts.holdout > 0
      ? 'holdout'
      : counts.validation > 0 ? 'validation' : 'train';
  }
  document.getElementById('synthesize-panel').hidden = false;
  document.getElementById('synthesize-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

document.getElementById('synth-close')?.addEventListener('click', () => {
  document.getElementById('synthesize-panel').hidden = true;
  activeSynthDataset = null;
  activeSynthManifest = null;
});
document.getElementById('synth-aggregation')?.addEventListener('change', updateSynthAggregationControls);

// The judge scorer needs to know WHICH adapter judges — typically the
// judge LoRA trained from A/B picks. Reveal + populate the picker only
// when the judge scorer is selected.
document.getElementById('synth-scorer')?.addEventListener('change', () => {
  const isJudge = document.getElementById('synth-scorer').value === 'judge';
  const group = document.getElementById('synth-judge-adapter-group');
  if (group) group.hidden = !isJudge;
  if (isJudge) populateSynthJudgeAdapters();
});

async function populateSynthJudgeAdapters() {
  const sel = document.getElementById('synth-judge-adapter');
  if (!sel) return;
  const current = sel.value;
  try {
    const res = await api('/v1/adapters');
    const names = (res.available || []).map(a => a.name);
    sel.innerHTML = '<option value="">Base model</option>' +
      names.map(n => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('');
    if (names.includes(current)) sel.value = current;
  } catch (_) { /* adapter list unavailable — base-model option remains */ }
}

function readSynthConfig() {
  const suite_name = document.getElementById('synth-suite-name').value.trim();
  if (!suite_name) { toast('Suite name is required', 'err'); return null; }
  const strategy = document.getElementById('synth-strategy').value;
  const scorerChoice = document.getElementById('synth-scorer').value;
  let scorer;
  if (scorerChoice === 'auto')       scorer = { kind: 'auto_detect' };
  else if (scorerChoice === 'judge') {
    const judgeAdapter = document.getElementById('synth-judge-adapter')?.value || null;
    scorer = { kind: 'judge', judge_adapter: judgeAdapter };
  }
  else if (scorerChoice === 'exact_match') scorer = { kind: 'fixed', scorer: { kind: 'exact_match', case_sensitive: false, strip_whitespace: true } };
  else if (scorerChoice === 'contains')    scorer = { kind: 'fixed', scorer: { kind: 'contains', phrases: [], mode: 'any', case_sensitive: false } };
  else if (scorerChoice === 'numeric')     scorer = { kind: 'fixed', scorer: { kind: 'numeric_tolerance', atol: 0, rtol: 0, integer_only: false } };
  else if (scorerChoice === 'tool_call')   scorer = { kind: 'fixed', scorer: { kind: 'tool_call' } };
  else if (scorerChoice === 'code')        scorer = { kind: 'fixed', scorer: { kind: 'code', style: { kind: 'token_similarity', min_jaccard: 0.6 } } };
  const max_examples = parseInt(document.getElementById('synth-max-examples').value, 10);
  const aggregationKind = document.getElementById('synth-aggregation')?.value || 'single';
  const k = aggregationKind === 'single'
    ? 1
    : parseInt(document.getElementById('synth-k')?.value || '', 10);
  if (!Number.isInteger(k) || k < 1 || k > 128) {
    toast('Completions (k) must be an integer from 1 to 128', 'err');
    return null;
  }
  if (aggregationKind === 'majority_at_k' && k % 2 === 0) {
    toast('Majority @ k requires an odd number of completions', 'err');
    return null;
  }
  const temperature = Number(document.getElementById('synth-temperature')?.value || 0);
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    toast('Temperature must be between 0 and 2', 'err');
    return null;
  }
  const aggregation = aggregationKind === 'single'
    ? { kind: 'single' }
    : { kind: aggregationKind, k };
  const seedVal = document.getElementById('synth-seed').value;
  const sampling = {
    max_examples: isFinite(max_examples) && max_examples > 0 ? max_examples : null,
    max_prompt_chars: 32768,
    max_target_chars: 4096,
    seed: seedVal ? parseInt(seedVal, 10) : null,
    dedupe: true,
  };
  return {
    suite_name,
    strategy,
    scorer,
    generation: { temperature, top_p: 1.0, top_k: 0, max_tokens: 256, n: k, stop: [], seed: null },
    aggregation,
    source_split: document.getElementById('synth-source-split')?.value || 'holdout',
    sampling,
    strip_system_prompt: document.getElementById('synth-strip-system').checked,
  };
}

document.getElementById('synth-preview-btn')?.addEventListener('click', async () => {
  if (!activeSynthDataset) return;
  const config = readSynthConfig();
  if (!config) return;
  const out = document.getElementById('synth-preview-output');
  out.innerHTML = '<div class="hint">Synthesizing preview…</div>';
  try {
    const res = await api('/v1/eval/datasets/' + encodeURIComponent(activeSynthDataset) + '/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...config, head_n: 5 }),
    });
    renderSynthPreview(out, res);
  } catch (e) { out.innerHTML = '<div class="eval-empty"><div class="eval-empty-body">Preview failed: ' + escapeHtml(e.message) + '</div></div>'; }
});

function renderSynthPreview(container, preview) {
  const s = preview.stats || {};
  const examples = preview.examples || [];
  const hist = s.auto_scorer_histogram || {};
  const histStr = Object.entries(hist).map(([k, v]) => `<span class="scorer-badge">${escapeHtml(k)}×${v}</span>`).join(' ');
  const exHtml = examples.slice(0, 5).map((ex, i) => {
    const userMsg = (ex.messages || []).filter(m => m.role === 'user').slice(-1)[0];
    const userText = userMsg ? userMsg.content : '';
    const tags = (ex.tags || []).map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('');
    const scorerKind = ex.scorer ? ex.scorer.kind : '(suite default)';
    return `<div style="border:1px solid var(--border); border-radius:6px; padding:10px; margin-top:6px; background:var(--surface);">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <div class="eyebrow">Example ${i+1}</div>
        <span class="scorer-badge">${escapeHtml(scorerKind)}</span>
      </div>
      <div style="font-size:11px; color:var(--text-muted); margin-bottom:2px;">prompt</div>
      <div style="font-family:var(--font-mono); font-size:12px; max-height:60px; overflow:auto; margin-bottom:6px;">${escapeHtml(truncate(userText, 240))}</div>
      <div style="font-size:11px; color:var(--text-muted); margin-bottom:2px;">target</div>
      <div style="font-family:var(--font-mono); font-size:12px; max-height:80px; overflow:auto;">${escapeHtml(truncate(ex.target || '', 320))}</div>
      <div style="margin-top:6px;">${tags}</div>
    </div>`;
  }).join('');
  container.innerHTML = `
    <div style="margin-bottom:8px; padding:10px; background:var(--surface-2); border-radius:6px; display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
      <div>
        <div class="hint" style="font-size:11px; color:var(--text-muted);">source partition</div>
        <div style="font-size:13px; font-weight:600;">${escapeHtml(preview.source_split || 'holdout')}</div>
      </div>
      <div>
        <div class="hint" style="font-size:11px; color:var(--text-muted);">examples generated</div>
        <div style="font-size:18px; font-weight:700; font-variant-numeric:tabular-nums;">${(s.examples_generated || 0).toLocaleString()}</div>
      </div>
      <div>
        <div class="hint" style="font-size:11px; color:var(--text-muted);">trajectories used</div>
        <div style="font-size:18px; font-weight:700; font-variant-numeric:tabular-nums;">${(s.trajectories_used || 0).toLocaleString()}</div>
      </div>
      <div style="flex:1; min-width:200px;">
        <div class="hint" style="font-size:11px; color:var(--text-muted); margin-bottom:4px;">auto-detected scorers</div>
        <div>${histStr || '<span class="hint">n/a</span>'}</div>
      </div>
      <div>
        <div class="hint" style="font-size:11px; color:var(--text-muted);">completion reduction</div>
        <div style="font-size:13px; font-weight:600;">${escapeHtml(evalAggregationLabel(preview.aggregation))}</div>
      </div>
    </div>
    <div class="hint" style="margin-bottom:8px; font-size:11px;">Skipped: empty target=${s.skipped_no_target || 0} · prompt-too-long=${s.skipped_prompt_too_long || 0} · target-too-long=${s.skipped_target_too_long || 0} · duplicate=${s.skipped_duplicate || 0}</div>
    ${exHtml || '<div class="eval-empty"><div class="eval-empty-body">No examples produced — try a different strategy or relax the sampling caps.</div></div>'}
  `;
}

async function doSynthesize(runAgainst) {
  if (!activeSynthDataset) return;
  const config = readSynthConfig();
  if (!config) return;
  try {
    const body = { ...config, force: false };
    if (runAgainst && runAgainst.length) body.run_against = runAgainst;
    const res = await api('/v1/eval/datasets/' + encodeURIComponent(activeSynthDataset) + '/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const queued = (res.queued_eval_job_ids || []).length;
    toast(`Saved suite "${res.suite.name}" (${res.stats.examples_generated} examples)${queued ? ', queued ' + queued + ' eval job(s)' : ''}`, 'ok');
    refreshSuites();
    refreshEvalJobs();
    if (queued > 0) {
      // Hop to the Jobs tab so the user immediately sees the run.
      document.getElementById('evals-tab-jobs')?.click();
    }
  } catch (e) { toast('Synthesize failed: ' + e.message, 'err'); }
}

document.getElementById('synth-save-btn')?.addEventListener('click', () => doSynthesize([]));
document.getElementById('synth-save-and-run-btn')?.addEventListener('click', async () => {
  await doSynthesize([evalActiveAdapter || '']);
});

// Shared multipart POST for every dataset-upload surface (the form, the
// sample-dataset CTA, the corrections builder). Matches the server contract:
// fields `name`, `format`, optional `description`, and `file` (JSONL bytes).
async function postDatasetUpload(name, format, description, fileOrBlob) {
  const fd = new FormData();
  fd.append('name', name);
  fd.append('format', format);
  if (description) fd.append('description', description);
  fd.append('file', fileOrBlob, fileOrBlob.name || name + '.jsonl');
  const res = await fetch('/v1/eval/datasets/upload', { method: 'POST', body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const e = new Error(err.error?.message || `HTTP ${res.status}`);
    e.code = err.error?.code;
    throw e;
  }
  return res.json();
}

document.getElementById('dataset-upload-form')?.addEventListener('submit', async ev => {
  ev.preventDefault();
  const name = document.getElementById('dataset-name').value.trim();
  const format = document.getElementById('dataset-format').value;
  const description = document.getElementById('dataset-description').value.trim();
  const file = document.getElementById('dataset-file').files[0];
  if (!name || !file) { toast('Name and file are required', 'err'); return; }
  try {
    const m = await postDatasetUpload(name, format, description, file);
    toast(`Uploaded "${m.name}" (${m.num_rows.toLocaleString()} rows)`, 'ok');
    document.getElementById('dataset-upload-form').reset();
    refreshDatasets();
    // Next step depends on what they uploaded: training data should lead to
    // TRAINING (one click), not get hijacked into the eval-synthesis flow.
    if (format === 'sft_chat' || format === 'grpo_groups') {
      const kind = format === 'sft_chat' ? 'sft' : 'grpo';
      showDatasetUploadedNext(m.name, kind, m.num_rows, m);
    } else {
      openSynthPanel(m.name, m);
    }
  } catch (e) { toast('Upload failed: ' + e.message, 'err'); }
});

/* ---------- First-run CTAs: sample dataset + corrections → dataset ---------- */

// A small built-in sft_chat dataset themed on realistic coding-agent traffic
// (what pi actually sends Kiln: tool calls, code review, commit messages,
// test writing) so "Run an eval" works in the first five minutes without
// bringing your own data. Every row matches the server's SftConversation
// contract exactly: messages[] of {role, content}, OpenAI-style tool_calls
// on assistant turns, and name/tool_call_id on tool replies.
const SAMPLE_DATASET_NAME = 'sample-coding-agent';
function sampleDatasetRows() {
  const call = (id, name, args) => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } });
  return [
    { messages: [
      { role: 'user', content: 'What files import the deprecated `legacy_client` module?' },
      { role: 'assistant', content: '', tool_calls: [call('call_1', 'grep_repo', { pattern: 'legacy_client', glob: '**/*.rs' })] },
      { role: 'tool', name: 'grep_repo', tool_call_id: 'call_1', content: 'src/api/handlers.rs:12\nsrc/sync/worker.rs:7\ntests/integration/client_test.rs:3' },
      { role: 'assistant', content: 'Three files still import `legacy_client`: src/api/handlers.rs, src/sync/worker.rs, and tests/integration/client_test.rs. The two src/ call sites need migrating to `TokenStore` before the module can be removed; the test can be deleted with it.' },
    ] },
    { messages: [
      { role: 'user', content: 'Run the cache tests and tell me if anything fails.' },
      { role: 'assistant', content: '', tool_calls: [call('call_2', 'run_tests', { package: 'cache', filter: '' })] },
      { role: 'tool', name: 'run_tests', tool_call_id: 'call_2', content: 'running 14 tests\ntest read_through_hit ... ok\ntest eviction_lru ... FAILED\n13 passed; 1 failed' },
      { role: 'assistant', content: 'One failure: `eviction_lru`. The test expects the least-recently-used entry to be evicted first, but the current implementation evicts by insertion order. The fix is to touch the entry timestamp on read, not just on write.' },
    ] },
    { messages: [
      { role: 'user', content: 'What does the retry helper in src/net/retry.rs actually do?' },
      { role: 'assistant', content: '', tool_calls: [call('call_3', 'read_file', { path: 'src/net/retry.rs' })] },
      { role: 'tool', name: 'read_file', tool_call_id: 'call_3', content: 'pub async fn with_retry<F, T>(op: F, max: u32) -> Result<T> { /* exponential backoff, jitter 0-50ms, caps at 5s */ }' },
      { role: 'assistant', content: '`with_retry` wraps an async operation with exponential backoff: it retries up to `max` times, doubling the delay each attempt with 0-50ms of jitter, capped at 5 seconds per wait. Errors from the final attempt propagate to the caller unchanged.' },
    ] },
    { messages: [
      { role: 'user', content: 'Generate a conventional-commit message for this change: swapped the blocking Mutex for an async RwLock in cache.rs so reads no longer serialize.' },
      { role: 'assistant', content: 'fix(cache): replace blocking Mutex with async RwLock so concurrent reads no longer serialize' },
    ] },
    { messages: [
      { role: 'user', content: 'Review this diff and flag anything risky: the PR holds a `.read()` guard across an `await` on the network fetch in cache.rs line 88.' },
      { role: 'assistant', content: 'Risky: holding the RwLock read guard across the awaited network fetch blocks every writer for the full request latency, and can deadlock if the fetch path ever takes the write lock. Scope the guard to the lookup only — clone the value out, drop the guard, then await the fetch.' },
    ] },
    { messages: [
      { role: 'user', content: 'How do I store handlers with different concrete types in one Vec in Rust?' },
      { role: 'assistant', content: 'Use trait objects: define a common trait and store boxed instances — `Vec<Box<dyn Handler>>`. Each handler implements `Handler`, and you push `Box::new(MyHandler)` into the vec. If the handlers are cloned or shared across threads, use `Vec<Arc<dyn Handler + Send + Sync>>` instead.' },
    ] },
    { messages: [
      { role: 'user', content: 'Write a unit test for the rate limiter covering burst then steady-state.' },
      { role: 'assistant', content: '#[tokio::test]\nasync fn limiter_allows_burst_then_throttles() {\n    let lim = RateLimiter::new(5, Duration::from_secs(1));\n    for _ in 0..5 {\n        assert!(lim.try_acquire());\n    }\n    assert!(!lim.try_acquire(), "burst budget should be exhausted");\n    tokio::time::sleep(Duration::from_secs(1)).await;\n    assert!(lim.try_acquire(), "tokens should refill after the window");\n}' },
    ] },
    { messages: [
      { role: 'user', content: 'Explain what this query does: SELECT user_id, COUNT(*) AS n FROM events WHERE ts > now() - interval \'7 days\' GROUP BY user_id HAVING COUNT(*) > 100;' },
      { role: 'assistant', content: 'It finds heavy users from the last week: it filters `events` to rows newer than 7 days, groups them by `user_id`, counts each user\'s events, and keeps only users with more than 100 events. The result is one row per heavy user with their event count `n`.' },
    ] },
    { messages: [
      { role: 'user', content: 'Why would the integration test `flaky_reconnect` fail intermittently but pass on retry?' },
      { role: 'assistant', content: 'The test asserts reconnection within 100ms, but the client\'s backoff adds random jitter that can push the first retry to ~150ms under load. Either widen the assertion window to cover worst-case jitter or inject a deterministic backoff in tests.' },
    ] },
    { messages: [
      { role: 'system', content: 'You are a careful coding agent working in a Rust codebase.' },
      { role: 'user', content: 'Refactor src/auth/session.rs to use the new TokenStore trait. What is your plan?' },
      { role: 'assistant', content: 'Plan: 1) add a `store: Arc<dyn TokenStore>` field to `Session` and thread it through `Session::new`; 2) replace the three direct `token_db` calls in handlers/ with `self.store` methods; 3) keep the old constructor behind a deprecated shim for one release; 4) run the auth test suite and fix call sites until green.' },
    ] },
  ];
}

// "Try a sample dataset" — upload the embedded rows through the same endpoint
// a real JSONL file goes through, then hand off to the eval-suite synthesizer
// (that is the next step on the eval golden path).
async function uploadSampleDataset(btn) {
  if (btn) btn.disabled = true;
  try {
    const jsonl = sampleDatasetRows().map(r => JSON.stringify(r)).join('\n') + '\n';
    const blob = new Blob([jsonl], { type: 'application/jsonl' });
    const m = await postDatasetUpload(SAMPLE_DATASET_NAME, 'sft_chat',
      'Built-in sample: coding-agent conversations with tool calls', blob);
    refreshDatasets();
    toast(`Sample dataset added (${m.num_rows} rows) — next: synthesize an eval suite from it`, 'ok');
    openSynthPanel(m.name, m);
  } catch (e) {
    if (e.code === 'dataset_exists' || /already exists/i.test(e.message || '')) {
      toast('The sample dataset is already here — synthesize an eval suite from it', 'info');
      refreshDatasets();
      openSynthPanel(SAMPLE_DATASET_NAME);
    } else {
      toast('Could not add the sample dataset: ' + e.message, 'err');
      if (btn) btn.disabled = false;
    }
  }
}

// "Build a dataset from your corrections" — the durable corrections store
// (your hand-written ideal answers, including rows already trained into an
// adapter) becomes an sft_chat dataset via the SAME transform the Corrections
// card trains with, so you can eval exactly what you taught.
async function buildDatasetFromCorrections(btn) {
  if (btn) btn.disabled = true;
  try {
    let rows = correctionsBasket;
    try {
      const d = await api('/v1/corrections?include_trained=1');
      if (d && Array.isArray(d.corrections)) rows = d.corrections;
    } catch (_) { /* server store unreachable — the local basket still works */ }
    const finished = rows.filter(corrTrainable);
    if (!finished.length) {
      toast('Your corrections need ideal answers first — open Corrections on the Overview page and write what pi should have said', 'info');
      if (btn) btn.disabled = false;
      return;
    }
    const jsonl = correctionsToSftExamples(finished).map(r => JSON.stringify(r)).join('\n') + '\n';
    const name = 'corrections-' + new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    const m = await postDatasetUpload(name, 'sft_chat',
      'Your corrections: each row pairs a prompt with the answer you said pi should have given', new Blob([jsonl], { type: 'application/jsonl' }));
    refreshDatasets();
    toast(`Dataset "${m.name}" built from ${finished.length} correction${finished.length === 1 ? '' : 's'} — next: synthesize an eval suite from it`, 'ok');
    openSynthPanel(m.name, m);
  } catch (e) {
    toast('Could not build the dataset: ' + e.message, 'err');
    if (btn) btn.disabled = false;
  }
}

// Inline "uploaded — what next?" strip on the Datasets tab. Primary action is
// training (that's why most people upload SFT/GRPO data); synthesizing an eval
// from the same rows is offered alongside.
function showDatasetUploadedNext(name, kind, numRows, manifest = null) {
  const old = document.getElementById('dataset-uploaded-next');
  if (old) old.remove();
  const form = document.getElementById('dataset-upload-form');
  if (!form) return;
  const strip = document.createElement('div');
  strip.id = 'dataset-uploaded-next';
  strip.className = 'corr-receipt';
  strip.setAttribute('role', 'status');
  strip.innerHTML = `
    <span class="corr-receipt-icon">${icon('check', 'icn-sm')}</span>
    <span class="corr-receipt-text"><strong>${escapeHtml(name)}</strong> uploaded (${Number(numRows || 0).toLocaleString()} rows). Train on it now, or build an eval from it.</span>
    <button type="button" class="btn btn-sm btn-primary" id="dataset-next-train">Train on this dataset ${icon('arrow-right', 'icn-sm')}</button>
    <button type="button" class="btn btn-sm" id="dataset-next-synth">Synthesize an eval</button>
    <button type="button" class="btn btn-sm btn-ghost corr-receipt-dismiss" id="dataset-next-dismiss" aria-label="Dismiss">${icon('close', 'icn-sm')}</button>`;
  form.insertAdjacentElement('afterend', strip);
  document.getElementById('dataset-next-train')?.addEventListener('click', () => { strip.remove(); trainFromDataset(name, kind); });
  document.getElementById('dataset-next-synth')?.addEventListener('click', () => { strip.remove(); openSynthPanel(name, manifest); });
  document.getElementById('dataset-next-dismiss')?.addEventListener('click', () => strip.remove());
}

/* ---------- Suites ---------- */

// Cache job results so we can compute per-suite sparkline trends.
let evalJobsCache = [];
// Lifecycle counts derived from evalJobsCache on every refresh. `null` until
// the first /v1/eval/jobs response lands so consumers (updateFlywheel) can
// tell "not loaded yet" apart from "zero evals ever" — an unfetched jobs
// list is unknown, not empty.
let evalJobCounts = null;

async function refreshSuites() {
  try {
    const d = await api('/v1/eval/suites');
    const suites = d.suites || [];
    const el = document.getElementById('suites-list');
    if (!suites.length) {
      el.className = 'eval-empty';
      setListHtml(el, 'empty', `
        <div class="eval-empty-icon"><svg class="icn"><use href="#i-target"></use></svg></div>
        <div class="eval-empty-title">No eval suites yet</div>
        <div class="eval-empty-body">A suite is a set of prompts with expected answers and a scorer — your model's report card. Create one from a dataset on the Datasets tab; no data yet? The built-in sample dataset works out of the box.</div>
        <button class="eval-empty-cta" type="button" title="Synthesize a suite from any dataset — power users can also POST an EvalSuite document to /v1/eval/suites" onclick="document.getElementById('evals-tab-datasets').click()">Create a suite from a dataset</button>`);
      return;
    }
    el.className = '';
    // Build a per-suite history from the cached job list. The server returns
    // jobs NEWEST-first (sorted descending by submitted_at_iso), but the
    // sparkline draws points left-to-right in array order and the badge takes
    // the LAST entry — so accumulate each history oldest→newest. Sort a copy
    // (never mutate the shared evalJobsCache) rather than blindly reversing,
    // matching the defensive re-sorts in adapterEvalChip/adapterCompareVerdict.
    const suiteHistory = {};
    const completedOldestFirst = evalJobsCache
      .filter(j => j.state === 'completed' && j.headline_accuracy != null)
      .sort((a, b) => String(a.submitted_at_iso || '').localeCompare(String(b.submitted_at_iso || '')));
    for (const j of completedOldestFirst) {
      (suiteHistory[j.suite_name] = suiteHistory[j.suite_name] || []).push(j.headline_accuracy);
    }
    // Key on everything the cards display: the suites payload, the
    // sparkline/badge history derived from evalJobsCache (#1548 — a new
    // completed run must repaint even when the suites payload is byte-
    // identical), and evalActiveAdapter (the Run/A-B button titles and the
    // A/B disabled state embed it).
    const listKey = 'list:' + JSON.stringify([
      evalActiveAdapter,
      suites.map(s => [s.name, s.description, s.num_examples, s.default_scorer_kind, s.aggregation, s.completions_per_example, s.schema_version]),
      completedOldestFirst.map(j => [j.suite_name, j.headline_accuracy]),
    ]);
    const listHtml = suites.map(s => {
      const hist = (suiteHistory[s.name] || []).slice(-10);
      const recent = hist.length ? hist[hist.length - 1] : null;
      const sparkline = hist.length >= 2 ? sparkSvg(hist) : '';
      const recentBadge = recent != null
        ? `<span class="job-state-pill completed" title="Latest run accuracy">${(recent*100).toFixed(0)}%</span>`
        : '';
      return `<div class="eval-row eval-row-suites">
        <div>
          <div class="row-title">${escapeHtml(s.name)}</div>
          <div class="row-sub">${escapeHtml(truncate(s.description || 'No description', 120))}</div>
        </div>
        <div class="tabular-nums">${s.num_examples.toLocaleString()} examples · <span class="scorer-badge">${escapeHtml(s.default_scorer_kind)}</span> · <span class="scorer-badge">${escapeHtml(evalAggregationLabel(s.aggregation))}</span></div>
        <div style="display:flex; gap:6px; align-items:center;">${recentBadge} ${sparkline}</div>
        <div class="row-actions">
          <button type="button" class="btn btn-primary btn-sm" data-suite="${escapeHtml(s.name)}" data-action="run" ${evalActiveAdapter ? `title="Score ${escapeHtml(evalActiveAdapter)} (the active adapter) on this suite"` : 'title="Score the base model on this suite"'}>Run</button>
          <button type="button" class="btn btn-sm" data-suite="${escapeHtml(s.name)}" data-action="compare" ${evalActiveAdapter ? `title="Compare base vs ${escapeHtml(evalActiveAdapter)} (the active adapter) — to compare a different adapter, use Run eval… on its card under Adapters"` : 'disabled title="No adapter is active — load one on the Adapters page, or use Run eval… on any adapter card"'}>A/B${evalActiveAdapter ? '' : ''}</button>
          <button type="button" class="btn btn-sm" data-suite="${escapeHtml(s.name)}" data-action="preview" title="Show the first few examples without running">Preview</button>
          <button type="button" class="btn btn-sm" data-suite="${escapeHtml(s.name)}" data-action="del">Delete</button>
        </div>
      </div>`;
    }).join('');
    if (!setListHtml(el, listKey, listHtml)) return; // unchanged — old nodes keep their listeners
    el.querySelectorAll('button[data-suite]').forEach(b => {
      const suite = b.dataset.suite;
      b.addEventListener('click', async () => {
        const action = b.dataset.action;
        try {
          if (action === 'run') {
            const res = await api('/v1/eval/run', {
              method: 'POST', headers: {'Content-Type':'application/json'},
              body: JSON.stringify({ suite, adapter: evalActiveAdapter || '' }),
            });
            toast(`Queued eval ${res.job_id.slice(0, 8)} · seed ${res.effective_seed}`, 'ok');
            document.getElementById('evals-tab-jobs')?.click();
            refreshEvalJobs();
          } else if (action === 'compare') {
            const res = await api('/v1/eval/compare', {
              method: 'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({ suite, adapters: ['', evalActiveAdapter || ''] }),
            });
            toast(`Queued compare ${res.job_id.slice(0, 8)} · seed ${res.effective_seed}`, 'ok');
            document.getElementById('evals-tab-jobs')?.click();
            refreshEvalJobs();
          } else if (action === 'preview') {
            await openSuitePreview(suite);
          } else if (action === 'del') {
            if (!confirm(`Delete suite "${suite}"?`)) return;
            await api('/v1/eval/suites/' + encodeURIComponent(suite), { method: 'DELETE' });
            toast('Suite deleted', 'ok');
            refreshSuites();
          }
        } catch (e) { toast(action + ' failed: ' + e.message, 'err'); }
      });
    });
  } catch (e) {
    // Error-specific key: recovery payloads (even identical ones) repaint.
    setListHtml(document.getElementById('suites-list'), 'err:' + e.message,
      `<div class="eval-empty"><div class="eval-empty-body">Failed: ${escapeHtml(e.message)}</div></div>`);
  }
}

/* ---------- Suite preview (lightweight modal — first N examples) ---------- */
function closeSuitePreviewModal() {
  const modal = document.getElementById('suite-preview-modal');
  if (!modal || modal.hidden) return;
  modal.hidden = true;
  closeModal(modal);
}
async function openSuitePreview(name) {
  // Lazy-create the modal scaffolding on first use. Reuses the same
  // CSS classes as the other drill-ins for consistency. Escape, focus,
  // and the scroll lock come from the shared modal manager.
  let modal = document.getElementById('suite-preview-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'suite-preview-modal';
    modal.className = 'modal-backdrop';
    modal.role = 'dialog';
    modal.setAttribute('aria-modal', 'true');
    modal.innerHTML = `<div class="modal-shell" tabindex="-1">
      <div class="modal-head">
        <h2 id="suite-preview-title">Suite preview</h2>
        <span class="modal-meta" id="suite-preview-meta"></span>
        <button class="modal-close" id="suite-preview-close" aria-label="Close"><svg class="icn" aria-hidden="true"><use href="#i-close"></use></svg></button>
      </div>
      <div class="modal-body" style="grid-template-columns: 1fr;">
        <div class="modal-content" id="suite-preview-content"><div class="detail-empty">Loading…</div></div>
      </div>
    </div>`;
    document.body.appendChild(modal);
    document.getElementById('suite-preview-close').addEventListener('click', closeSuitePreviewModal);
    modal.addEventListener('click', (ev) => {
      if (ev.target === modal) closeSuitePreviewModal();
    });
  }
  modal.hidden = false;
  openModal(modal, { onClose: closeSuitePreviewModal });
  document.getElementById('suite-preview-title').textContent = `Suite: ${name}`;
  document.getElementById('suite-preview-meta').textContent = '';
  const content = document.getElementById('suite-preview-content');
  content.innerHTML = '<div class="detail-empty">Loading…</div>';
  try {
    const suite = await api('/v1/eval/suites/' + encodeURIComponent(name));
    const examples = suite.examples || [];
    const preview = examples.slice(0, 20);
    document.getElementById('suite-preview-meta').innerHTML =
      `${examples.length} example${examples.length === 1 ? '' : 's'}` +
      (suite.default_scorer ? ` · <span class="scorer-badge">${escapeHtml(suite.default_scorer.kind || 'scorer')}</span>` : '');
    if (!preview.length) {
      content.innerHTML = '<div class="detail-empty">This suite has no examples.</div>';
      return;
    }
    const rows = preview.map((ex, i) => {
      const msgs = (ex.messages || [])
        .map(m => `<div style="margin-bottom:4px;"><span class="role ${escapeHtml(m.role)}" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps); color:var(--text-muted); margin-right:6px;">${escapeHtml(m.role)}</span><span style="white-space:pre-wrap; font-family:var(--font-mono); font-size:12px;">${escapeHtml(truncate(m.content || '', 600))}</span></div>`)
        .join('');
      const target = ex.target != null
        ? `<div style="margin-top:6px;"><span class="hint" style="font-size:11px;">target:</span> <code style="font-family:var(--font-mono); font-size:12px;">${escapeHtml(truncate(String(ex.target), 200))}</code></div>`
        : '';
      const tags = (ex.tags || []).map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('');
      return `<div style="border:1px solid var(--border); border-radius:var(--radius-md); padding:var(--space-3); margin-bottom:var(--space-3); background:var(--surface-2);">
        <div style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono); margin-bottom:6px;">#${i + 1}${ex.id ? ' · ' + escapeHtml(ex.id) : ''}</div>
        ${msgs}
        ${target}
        ${tags ? `<div style="margin-top:6px;">${tags}</div>` : ''}
      </div>`;
    }).join('');
    const more = examples.length > preview.length
      ? `<div class="hint" style="text-align:center; padding:var(--space-3);">…showing first ${preview.length} of ${examples.length}. The Run action evaluates all of them.</div>`
      : '';
    content.innerHTML = `<div style="padding:var(--space-4) var(--space-5); overflow-y:auto;">${rows}${more}</div>`;
  } catch (e) {
    content.innerHTML = `<div class="detail-empty">Failed to load suite: ${escapeHtml(e.message)}</div>`;
  }
}

/* ---------- Jobs ---------- */

let evalJobsFilter = { query: '', state: 'all' };
function matchesEvalJobsFilter(j) {
  const q = (evalJobsFilter.query || '').trim().toLowerCase();
  if (q) {
    const hay = [
      j.suite_name || '',
      j.job_id || '',
      ...(j.adapters || []).map(a => a || 'base'),
    ].join(' ').toLowerCase();
    if (!hay.includes(q)) return false;
  }
  const st = (j.state || '').toString().toLowerCase();
  if (evalJobsFilter.state === 'running') return st === 'queued' || st === 'running';
  if (evalJobsFilter.state === 'completed') return st === 'completed';
  if (evalJobsFilter.state === 'failed') return st === 'failed' || st === 'cancelled';
  return true;
}
async function refreshEvalJobs() {
  try {
    const d = await api('/v1/eval/jobs');
    const jobs = d.jobs || [];
    evalJobsCache = jobs;
    // Lifecycle counts as JS state for the flywheel (and any other consumer):
    // data must not round-trip through a badge's rendered textContent.
    const stateOf = j => (j.state || '').toString().toLowerCase();
    evalJobCounts = {
      completed: jobs.filter(j => stateOf(j) === 'completed').length,
      running: jobs.filter(j => stateOf(j) === 'running').length,
      queued: jobs.filter(j => stateOf(j) === 'queued').length,
    };
    detectEvalTransitions(jobs);
    // Adapter cards show each adapter's latest eval score — refresh them now that
    // eval results changed (the dedup key includes the completed-eval signature).
    if (typeof refreshAdapterCards === 'function') refreshAdapterCards();
    // Header badge counts active jobs (queued + running), mirroring the
    // training badge — total job history is shown inside the tab so the
    // badge should signal "needs attention now", not "lifetime count".
    const liveCount = evalJobCounts.running + evalJobCounts.queued;
    setText('evals-count', String(liveCount));
    const evalsBadge = document.getElementById('evals-count');
    if (evalsBadge) evalsBadge.title = `${liveCount} eval job${liveCount === 1 ? '' : 's'} queued or running`;
    // The flywheel's Eval node reads evalJobCounts — repaint it now instead of
    // waiting for the next training/requests poll tick.
    updateFlywheel();
    const el = document.getElementById('eval-jobs-list');
    const filtered = jobs.filter(matchesEvalJobsFilter);
    if (jobs.length && !filtered.length) {
      el.className = 'eval-empty';
      if (setListHtml(el, 'nomatch', `<div class="eval-empty-body">No eval jobs match the current filter. <button class="btn btn-sm" type="button" data-eval-jobs-filter="all">Clear filter</button></div>`)) {
        el.querySelectorAll('[data-eval-jobs-filter]').forEach(btn => {
          btn.addEventListener('click', () => {
            document.querySelectorAll('[data-eval-jobs-filter]').forEach(b => b.classList.toggle('active', b.dataset.evalJobsFilter === 'all'));
            evalJobsFilter.state = 'all';
            const inp = document.getElementById('eval-jobs-filter');
            if (inp) inp.value = '';
            evalJobsFilter.query = '';
            refreshEvalJobs();
          });
        });
      }
      return;
    }
    if (!jobs.length) {
      el.className = 'eval-empty';
      setListHtml(el, 'empty', `
        <div class="eval-empty-icon"><svg class="icn"><use href="#i-chart"></use></svg></div>
        <div class="eval-empty-title">No eval jobs yet</div>
        <div class="eval-empty-body">Run a suite from the Suites tab. Jobs land here as they complete; click any job to drill into the per-example outcomes.</div>
        <button class="eval-empty-cta" type="button" onclick="document.getElementById('evals-tab-suites').click()">Browse suites</button>`);
      return;
    }
    el.className = '';
    // Key on the active filter (query + state pill) plus every field a job
    // card displays — id/state, headline accuracy, the whole progress object
    // (examples_completed/total, running accuracy/mean), per-run metrics and
    // tag pass-rates, and the error line. The filter belongs in the key so a
    // filter keystroke always repaints even when it yields the same set.
    const listKey = 'jobs:' + JSON.stringify([
      evalJobsFilter.query, evalJobsFilter.state,
      filtered.map(j => [
        j.job_id, j.state, j.suite_name, j.adapters, j.submission_kind,
        j.effective_seed, j.headline_accuracy, j.progress, j.error, j.replay_verdict,
        (j.finished_runs || []).map(r => [r.adapter, r.metrics]),
      ]),
    ]);
    if (setListHtml(el, listKey, filtered.map(j => renderJobCard(j)).join(''))) {
      el.querySelectorAll('.job-card').forEach(card => {
        card.addEventListener('click', () => openDrillModal(card.dataset.jobId));
      });
    }
  } catch (e) {
    // Error-specific key: the recovered list (even an identical empty
    // payload) compares unequal and repaints (#1547 regression shape).
    setListHtml(document.getElementById('eval-jobs-list'), 'err:' + e.message,
      `<div class="eval-empty"><div class="eval-empty-body">Failed: ${escapeHtml(e.message)}</div></div>`);
  }
  // Refreshing jobs also updates suite sparklines.
  if (document.getElementById('evals-tab-suites')?.classList.contains('active')) {
    refreshSuites();
  }
}

// The flywheel's headline answer: did the adapter beat base, and by how much?
// Reads the per-run accuracies (base run keyed by adapter==null) and renders the
// existing-but-unused .delta-badge so the verdict isn't left as mental math.
// Eval completions are announced with the VERDICT attached — the number the
// user queued the job to learn, delivered instead of buried in the Jobs tab.
let prevEvalStates = null;
function detectEvalTransitions(jobs) {
  const now = new Map();
  (jobs || []).forEach(j => now.set(j.job_id, (j.state || '').toString().toLowerCase()));
  if (prevEvalStates) {
    for (const [id, state] of now) {
      const prev = prevEvalStates.get(id);
      if (!prev || prev === state || (prev !== 'running' && prev !== 'queued')) continue;
      const j = (jobs || []).find(x => x.job_id === id) || {};
      const suite = j.suite_name || 'eval';
      if (state === 'completed') {
        let verdict = '';
        // Same gate as the adapter card: win/loss phrasing only when the
        // paired sign test clears SIGN_TEST_ALPHA; otherwise the toast stays
        // neutral. One verdict per candidate — no best-of-N reduce(max).
        const verdicts = gatedCompareVerdicts(j.finished_runs || []);
        if (verdicts.length) {
          const phrase = (v) => v.significant
            ? (Math.abs(v.delta) <= 0.5
              ? `matches base (${fmtSignTestP(v.p)})`
              : `${v.delta > 0 ? '+' : ''}${v.delta.toFixed(1)} pts vs base (${fmtSignTestP(v.p)})`)
            : `no significant difference vs base (${fmtSignTestP(v.p)})`;
          verdict = verdicts.length === 1
            ? ` Verdict: ${phrase(verdicts[0])}.`
            : ` Verdicts: ${verdicts.map(v => `${v.candidate} ${phrase(v)}`).join('; ')}.`;
        } else if (typeof j.headline_accuracy === 'number') {
          verdict = ` Accuracy: ${(j.headline_accuracy * 100).toFixed(0)}%.`;
        }
        announceStatus('eval-jobs-status', `Eval ${suite} finished.${verdict}`);
        actionToast(`Eval ${suite} finished.${verdict}`, 'ok', [
          { label: 'View result', onClick: () => { selectPage('evals'); document.getElementById('evals-tab-jobs')?.click(); setTimeout(() => openDrillModal(id), 250); } },
        ]);
      } else if (state === 'failed') {
        announceStatus('eval-jobs-status', `Eval ${suite} failed.`);
        actionToast(`Eval ${suite} failed${j.error ? ': ' + String(j.error).slice(0, 80) : ''}.`, 'err', [
          { label: 'View job', onClick: () => { selectPage('evals'); document.getElementById('evals-tab-jobs')?.click(); } },
        ]);
      }
    }
  }
  prevEvalStates = now;
}

function compareVerdictBadge(runs) {
  // Same gate as the adapter card (gatedCompareVerdicts): a colored win/loss
  // badge only renders at p < SIGN_TEST_ALPHA; below that it's the neutral
  // "not enough evidence" treatment. One badge per candidate — no picking the
  // max of N candidates (best-of-N selection bias dressed up as a verdict).
  const verdicts = gatedCompareVerdicts(runs);
  if (!verdicts.length) return '';
  const multi = verdicts.length > 1;
  return verdicts.map(v => {
    const name = multi ? `${escapeHtml(v.candidate)}: ` : '';
    const title = `${escapeHtml(v.candidate)} ${(v.accuracy * 100).toFixed(0)}% vs base ${(v.baseAccuracy * 100).toFixed(0)}% — sign test improved ${v.improved} / regressed ${v.regressed}, ${fmtSignTestP(v.p)}`;
    if (!v.significant && Math.abs(v.delta) > 0.5) {
      return `<span class="delta-badge delta-flat" title="${title}">${name}${v.delta > 0 ? '+' : ''}${v.delta.toFixed(1)} pts — not enough evidence (${fmtSignTestP(v.p)})</span>`;
    }
    const cls = v.delta > 0.5 ? 'delta-up' : (v.delta < -0.5 ? 'delta-down' : 'delta-flat');
    const label = cls === 'delta-flat' ? 'matches base' : `${v.delta > 0 ? '+' : ''}${v.delta.toFixed(1)} pts vs base`;
    return `<span class="delta-badge ${cls}" title="${title}">${name}${label}</span>`;
  }).join('');
}

// Non-completed jobs have no score — show a state figure, never a giant "0"
// (which reads as "lost to base" and obscures the actual win answer).
function jobStateFigure(stateClass) {
  const g = stateClass === 'running' ? 'activity' : stateClass === 'queued' ? 'play' : stateClass === 'failed' ? 'warning' : 'activity';
  return `<span class="job-statefig ${stateClass}" aria-hidden="true">${icon(g)}</span>`;
}

function renderBaseWeightSummary(manifest) {
  if (!manifest || typeof manifest.aggregate_sha256 !== 'string') return '';
  const digest = manifest.aggregate_sha256;
  const shortDigest = digest.length > 28 ? `${digest.slice(0, 18)}…${digest.slice(-8)}` : digest;
  const shardCount = Array.isArray(manifest.shards) ? manifest.shards.length : 0;
  const shardLabel = `${shardCount} shard${shardCount === 1 ? '' : 's'}`;
  const byteLabel = Number.isFinite(manifest.total_size_bytes) ? ` · ${fmtBytes(manifest.total_size_bytes)}` : '';
  return `<div class="hint" style="display:flex; align-items:center; gap:6px; min-width:0; flex-wrap:wrap; font-size:11px;">
    <strong style="color:var(--text);">Base weights</strong>
    <code title="${escapeHtml(digest)}" style="overflow-wrap:anywhere;">${escapeHtml(shortDigest)}</code>
    <button class="btn btn-sm btn-ghost" type="button" data-copy-base-weight="${escapeHtml(digest)}" title="Copy exact base-weight aggregate" aria-label="Copy exact base-weight aggregate"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg></button>
    <span>${escapeHtml(shardLabel + byteLabel)}</span>
  </div>`;
}

function renderExecutionProvenanceSummary(provenance) {
  if (!provenance || typeof provenance.provenance_sha256 !== 'string') return '';
  const digest = provenance.provenance_sha256;
  const shortDigest = digest.length > 28 ? `${digest.slice(0, 18)}…${digest.slice(-8)}` : digest;
  const backend = provenance.backend?.name || 'unknown';
  const device = provenance.backend?.device || 'unknown device';
  return `<div class="hint" style="display:flex; align-items:center; gap:6px; min-width:0; flex-wrap:wrap; font-size:11px;">
    <strong style="color:var(--text);">Execution</strong>
    <span>${escapeHtml(`${backend} · ${device}`)}</span>
    <code title="${escapeHtml(digest)}" style="overflow-wrap:anywhere;">${escapeHtml(shortDigest)}</code>
    <button class="btn btn-sm btn-ghost" type="button" data-copy-execution="${escapeHtml(digest)}" title="Copy exact execution provenance digest" aria-label="Copy exact execution provenance digest"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg></button>
  </div>`;
}

function renderTrainingPrecisionSummary(precision) {
  if (!precision || typeof precision !== 'object') return '';
  const parameter = precision.parameter_dtype || 'unknown';
  const optimizer = precision.optimizer_state_dtype || 'unknown';
  const activation = precision.activation_dtype || 'unknown';
  const gradient = precision.gradient_dtype || 'unknown';
  const rounding = precision.stochastic_rounding?.mode
    || (typeof precision.stochastic_rounding?.enabled === 'boolean'
      ? (precision.stochastic_rounding.enabled ? 'enabled' : 'disabled')
      : 'declared');
  return `<div class="hint" style="display:flex; align-items:center; gap:6px; min-width:0; flex-wrap:wrap; font-size:11px;">
    <strong style="color:var(--text);">Concrete precision</strong>
    <span>${escapeHtml(`${parameter} parameters · ${optimizer} optimizer · ${activation} activations · ${gradient} gradients · ${rounding}`)}</span>
  </div>`;
}

function wireBaseWeightCopy(root) {
  root?.querySelectorAll('[data-copy-base-weight]').forEach(btn => {
    btn.addEventListener('click', () => {
      const value = btn.dataset.copyBaseWeight;
      if (!value) return;
      copyText(value, btn).then(() => {
        if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = value;
        toast('Base-weight identity copied', 'ok');
      });
    });
  });
}

function wireExecutionProvenanceCopy(root) {
  root?.querySelectorAll('[data-copy-execution]').forEach(btn => {
    btn.addEventListener('click', () => {
      const value = btn.dataset.copyExecution;
      if (!value) return;
      copyText(value, btn).then(() => {
        if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = value;
        toast('Execution identity copied', 'ok');
      });
    });
  });
}

function replayStatusPresentation(status) {
  if (status === 'matched') return { cls: 'completed', label: 'Replay matched' };
  if (status === 'mismatch') return { cls: 'failed', label: 'Replay mismatch' };
  if (status === 'error') return { cls: 'failed', label: 'Replay error' };
  return { cls: 'running', label: 'Replay pending' };
}

function renderReplaySummary(job, compact = false) {
  if (!job?.replay_expectation) return '';
  const verdict = job.replay_verdict;
  const present = replayStatusPresentation(verdict?.status);
  if (compact) {
    return `<span class="job-state-pill ${present.cls}" title="${escapeHtml(verdict?.message || 'Waiting for a terminal byte-comparison verdict')}">${present.label}</span>`;
  }
  const record = verdict?.expected_record_sha256 || job.replay_expectation.expected_record_sha256;
  const raw = verdict?.expected_raw_completion_set_sha256 || job.replay_expectation.expected_raw_completion_set_sha256;
  const actualRecord = verdict?.actual_record_sha256;
  const actualRaw = verdict?.actual_raw_completion_set_sha256;
  const short = value => value?.length > 28 ? `${value.slice(0, 18)}…${value.slice(-8)}` : (value || 'missing');
  return `<div class="hint" style="display:flex; align-items:center; gap:6px; min-width:0; flex-wrap:wrap; font-size:11px;">
    <span class="job-state-pill ${present.cls}" title="${escapeHtml(verdict?.message || 'Waiting for a terminal byte-comparison verdict')}">${present.label}</span>
    <strong style="color:var(--text);">Record</strong><code title="${escapeHtml(record)}">${escapeHtml(short(record))}</code>
    <button class="btn btn-sm btn-ghost" type="button" data-copy-replay="${escapeHtml(record)}" title="Copy expected replay-record digest" aria-label="Copy expected replay-record digest"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg></button>
    <strong style="color:var(--text);">Raw set</strong><code title="${escapeHtml(raw)}">${escapeHtml(short(raw))}</code>
    <button class="btn btn-sm btn-ghost" type="button" data-copy-replay="${escapeHtml(raw)}" title="Copy expected raw-completion-set digest" aria-label="Copy expected raw-completion-set digest"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg></button>
    ${actualRecord && actualRecord !== record ? `<strong style="color:var(--text);">Actual record</strong><code title="${escapeHtml(actualRecord)}">${escapeHtml(short(actualRecord))}</code><button class="btn btn-sm btn-ghost" type="button" data-copy-replay="${escapeHtml(actualRecord)}" title="Copy actual replay-record digest" aria-label="Copy actual replay-record digest"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg></button>` : ''}
    ${actualRaw && actualRaw !== raw ? `<strong style="color:var(--text);">Actual raw set</strong><code title="${escapeHtml(actualRaw)}">${escapeHtml(short(actualRaw))}</code><button class="btn btn-sm btn-ghost" type="button" data-copy-replay="${escapeHtml(actualRaw)}" title="Copy actual raw-completion-set digest" aria-label="Copy actual raw-completion-set digest"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg></button>` : ''}
  </div>`;
}

function wireReplayCopy(root) {
  root?.querySelectorAll('[data-copy-replay]').forEach(btn => {
    btn.addEventListener('click', () => {
      const value = btn.dataset.copyReplay;
      if (!value) return;
      copyText(value, btn).then(() => toast('Replay identity copied', 'ok'));
    });
  });
}

function renderJobCard(j) {
  const acc = j.headline_accuracy;
  const adapters = (j.adapters || []).map(a => a == null ? '<span class="hint">base</span>' : escapeHtml(a)).join(' vs ');
  const stateClass = (j.state || 'queued').toLowerCase();
  const showRing = stateClass === 'completed' && typeof acc === 'number' && isFinite(acc);
  const progress = j.progress || {};
  const progFrac = progress.examples_total > 0 ? progress.examples_completed / progress.examples_total : 0;
  const isRunning = stateClass === 'running' || stateClass === 'queued';
  const seed = j.effective_seed == null ? '' : String(j.effective_seed);

  // Compact tag bars for the most-recent finished run (max 3)
  let tagSummary = '';
  if (j.finished_runs && j.finished_runs.length > 0) {
    const lastRun = j.finished_runs[j.finished_runs.length - 1];
    const rates = Object.entries(lastRun.metrics?.pass_rate_by_tag || {}).slice(0, 3);
    if (rates.length) {
      tagSummary = `<div style="display:flex; gap:8px; margin-top:6px; flex-wrap:wrap; font-size:11px;">`
        + rates.map(([k, v]) => `<span class="tag-chip">${escapeHtml(k)} ${(v*100).toFixed(0)}%</span>`).join('') + `</div>`;
    }
  }

  let progressOrCounts = '';
  if (isRunning) {
    progressOrCounts = `
      <div class="job-card-progress">
        <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${(progFrac*100).toFixed(1)}%;"></div></div>
        <span class="tabular-nums hint" style="font-size:11px;">${progress.examples_completed || 0}/${progress.examples_total || 0}</span>
      </div>`;
    if ((progress.examples_completed || 0) > 0) {
      progressOrCounts += `<div class="hint" style="font-size:11px; margin-top:4px;">running ${(progress.running_accuracy*100).toFixed(0)}% accuracy · mean ${(progress.running_mean_score).toFixed(2)}</div>`;
    }
  } else if (j.finished_runs && j.finished_runs.length > 0) {
    // Per-run mini bars when compare-mode
    const runsHtml = j.finished_runs.map(r => {
      const a = r.adapter || 'base';
      return `<span class="hint" style="font-size:11px; display:inline-flex; gap:4px; align-items:center; margin-right:10px;">
        <strong>${escapeHtml(a)}</strong>: <span class="tabular-nums">${(r.metrics.accuracy*100).toFixed(0)}%</span>
        <span class="hint" style="font-size:10px;">(${r.metrics.num_pass}/${r.metrics.num_examples})</span>
      </span>`;
    }).join('');
    const verdict = compareVerdictBadge(j.finished_runs);
    progressOrCounts = `<div style="margin-top:6px; display:flex; align-items:center; gap:8px; flex-wrap:wrap;">${runsHtml}${verdict}</div>`;
  } else if (j.error) {
    progressOrCounts = `<div class="hint" style="color:var(--danger-fg); margin-top:4px;">${escapeHtml(j.error)}</div>`;
  }

  return `<div class="job-card" data-job-id="${escapeHtml(j.job_id)}">
    ${showRing ? ringHtml(acc, 'large') : jobStateFigure(stateClass)}
    <div class="job-card-meta">
      <div class="job-card-suite">${escapeHtml(j.suite_name)}</div>
      <div class="job-card-sub">
        <span class="job-state-pill ${stateClass}">${escapeHtml(j.state || '')}</span>
        <span>${adapters}</span>
        <span class="hint">${escapeHtml(j.submission_kind)}</span>
        ${seed ? `<span class="hint" style="font-family:var(--font-mono);" title="Immutable effective eval seed">seed ${escapeHtml(seed)}</span>` : ''}
        ${renderReplaySummary(j, true)}
        <span class="hint" style="font-family:var(--font-mono);">${escapeHtml(j.job_id.slice(0, 8))}</span>
      </div>
      ${progressOrCounts}
      ${tagSummary}
    </div>
  </div>`;
}

/* ---------- Drill-in modal ---------- */

let drillJob = null;
// The job id the drill modal is showing (set before the fetch lands, unlike
// drillJob) — the deep-link router diffs against it.
let evalDrillJobId = null;
let drillFilter = 'all';
let drillSearch = '';
let drillSelectedRun = 0;
let drillSelectedOutcome = null;
let drillPollHandle = null;
// Map of example_id → { messages, target, scorer, weight, tags } for the
// suite the current drill job ran. Lets the detail panel show the prompt
// the model actually saw, not just the model's reply. Cached per-suite.
let drillExamplesById = new Map();
let drillSuiteCacheKey = null;

// Join each independent example reduction to the raw completion selected by
// the reducer. The UI makes decisions and filters on these rows while the
// complete raw completion set remains available in the result and export.
function drillExampleRows(run) {
  const raw = run?.outcomes || [];
  const aggregated = run?.aggregated_outcomes || [];
  if (!aggregated.length) return raw;
  return aggregated.map(outcome => {
    const representative = raw.find(item =>
      item.example_id === outcome.example_id
      && item.completion_index === outcome.representative_completion_index) || {};
    return {
      ...representative,
      raw_kind: representative.kind,
      raw_score: representative.score,
      example_id: outcome.example_id,
      completion_index: outcome.representative_completion_index,
      kind: outcome.kind,
      score: outcome.score,
      tags: outcome.tags || representative.tags || [],
      metadata: outcome.metadata ?? representative.metadata,
      aggregation_outcome: outcome,
    };
  });
}

async function openDrillModal(jobId) {
  evalDrillJobId = jobId;
  modalHashOnOpen('eval', '#evals/jobs/' + encodeURIComponent(jobId));
  drillFilter = 'all';
  drillSearch = '';
  drillSelectedRun = 0;
  drillSelectedOutcome = null;
  document.getElementById('drill-search').value = '';
  document.querySelectorAll('[data-drill-filter]').forEach(b => b.classList.toggle('active', b.dataset.drillFilter === 'all'));
  // A leftover raw-JSON block from a previously drilled job would show the
  // wrong payload until the user re-toggles — drop it on every open.
  document.getElementById('drill-raw-block')?.remove();
  const modal = document.getElementById('eval-drill-modal');
  modal.hidden = false;
  openModal(modal, { onClose: userCloseDrillModal });
  await fetchDrillJob(jobId);
  // If the job is still running, poll every second so the modal updates live.
  drillPollHandle = setInterval(async () => {
    if (!drillJob) return;
    if (drillJob.state === 'running' || drillJob.state === 'queued') {
      await fetchDrillJob(drillJob.job_id, /*preserveSelection*/ true);
    }
  }, 1500);
}

function closeDrillModal() {
  const modal = document.getElementById('eval-drill-modal');
  modal.hidden = true;
  closeModal(modal);
  document.getElementById('drill-raw-block')?.remove();
  drillJob = null;
  evalDrillJobId = null;
  drillSelectedOutcome = null;
  drillSuiteCacheKey = null;
  drillExamplesById = new Map();
  if (drillPollHandle) { clearInterval(drillPollHandle); drillPollHandle = null; }
}
// User-initiated close (X / backdrop / Esc / Cancel-Delete): walk history per
// the deep-link state machine. "Replay in playground" and re-run keep calling
// closeDrillModal directly — they navigate FORWARD from the modal, so its
// entry should stay behind them for Back.
function userCloseDrillModal() {
  modalHashOnUserClose('eval', '#evals/jobs', closeDrillModal);
}

async function fetchDrillJob(jobId, preserveSelection = false) {
  try {
    const j = await api('/v1/eval/jobs/' + encodeURIComponent(jobId));
    const jobMeta = evalJobsCache.find(item => item.job_id === jobId);
    drillJob = {
      ...j,
      suite_name: jobMeta?.suite_name || j.runs?.[0]?.suite_name || 'eval',
      adapters: jobMeta?.adapters || j.runs?.map(r => r.adapter ?? null) || [],
      submission_kind: jobMeta?.submission_kind || 'on_demand',
    };
    // Lazily fetch the suite content the *first* time we draw a drill for
    // it. The outcomes don't carry the example prompts (only the model's
    // reply) — without this the user can't actually debug a failure.
    const suiteName = drillJob.suite_name;
    if (suiteName && drillSuiteCacheKey !== suiteName) {
      drillSuiteCacheKey = suiteName;
      drillExamplesById = new Map();
      try {
        const suite = await api('/v1/eval/suites/' + encodeURIComponent(suiteName));
        // EvalExample.id is optional — when omitted the server uses a
        // sha256 prefix derived from messages+target+aliases. We mirror
        // the algorithm here so the outcome's example_id keys back to
        // the right prompt. Hashing is async (crypto.subtle), so we
        // resolve all of them in parallel.
        const examples = suite.examples || [];
        const ids = await Promise.all(examples.map(ex => ex.id ? Promise.resolve(ex.id) : hashExampleId(ex)));
        for (let i = 0; i < examples.length; i++) {
          drillExamplesById.set(ids[i], examples[i]);
        }
      } catch (_) {
        // Inline-suite jobs aren't registered, so this 404s — we degrade
        // to no-prompt mode silently.
      }
    }
    renderDrillModal(preserveSelection);
  } catch (e) {
    toast('Failed to load job: ' + e.message, 'err');
    // userClose (not plain close): consumes/repairs the hash entry too, so a
    // junk #evals/jobs/{id} deep link degrades to #evals/jobs cleanly.
    userCloseDrillModal();
  }
}

/// Recompute the same example ID the server uses when one is not provided.
/// Mirrors `EvalExample::resolved_id` in kiln-eval/src/suite.rs (sha256
/// over role|content|target|aliases, hex prefix of 8 bytes).
async function hashExampleId(ex) {
  const enc = new TextEncoder();
  const parts = [];
  for (const m of (ex.messages || [])) {
    parts.push(enc.encode(m.role));
    parts.push(new Uint8Array([0]));
    parts.push(enc.encode(m.content));
    parts.push(new Uint8Array([0]));
  }
  if (ex.target != null) {
    parts.push(enc.encode('|t|'));
    parts.push(enc.encode(ex.target));
  }
  for (const a of (ex.aliases || [])) {
    parts.push(enc.encode('|a|'));
    parts.push(enc.encode(a));
  }
  const total = parts.reduce((s, p) => s + p.length, 0);
  const buf = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { buf.set(p, off); off += p.length; }
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(digest, 0, 8)).map(b => b.toString(16).padStart(2, '0')).join('');
}

function renderDrillModal(preserveSelection) {
  const j = drillJob;
  if (!j) return;
  document.getElementById('drill-title').textContent = j.suite_name || 'Eval results';
  document.getElementById('drill-meta').innerHTML = `
    <span class="job-state-pill ${j.state}">${escapeHtml(j.state)}</span>
    ${j.effective_seed == null ? '' : `<span class="hint" style="margin-left:8px; font-family:var(--font-mono);" title="${escapeHtml(j.seed_derivation || 'kiln.eval-seed.v1')}">seed ${escapeHtml(String(j.effective_seed))}</span>`}
    <span class="hint" style="margin-left:8px; font-family:var(--font-mono);">${escapeHtml(j.job_id)}</span>`;
  // Cancel / Delete: same DELETE endpoint, different copy. Active jobs
  // get cancelled; terminal jobs get deleted from memory + archive.
  const stateLower = (j.state || '').toString().toLowerCase();
  const isActive = stateLower === 'queued' || stateLower === 'running';
  const cancelBtn = document.getElementById('drill-cancel');
  if (cancelBtn) {
    cancelBtn.hidden = false;
    cancelBtn.innerHTML = isActive ? icon('stop','icn-sm') + ' Cancel' : icon('trash','icn-sm') + ' Delete';
    cancelBtn.title = isActive
      ? 'Cancel this eval job (queued or running)'
      : 'Permanently delete this terminal job from memory and the on-disk archive';
    cancelBtn.dataset.mode = isActive ? 'cancel' : 'delete';
  }
  const rerunBtn = document.getElementById('drill-rerun');
  if (rerunBtn) {
    const failingInAnyRun = (j.runs || []).some(r =>
      drillExampleRows(r).some(o => o.kind !== 'pass'));
    rerunBtn.hidden = isActive || !failingInAnyRun;
  }
  const replayBtn = document.getElementById('drill-replay');
  if (replayBtn) {
    const selectedRun = (j.runs || [])[Math.min(drillSelectedRun, Math.max(0, (j.runs || []).length - 1))];
    replayBtn.hidden = j.state !== 'completed' || !selectedRun?.replay_record;
  }
  // Download outcomes (.jsonl): live across every run of the job (compare
  // jobs export all adapters, one line per outcome). Disabled until the
  // first outcome lands so the click never produces an empty file.
  const exportBtn = document.getElementById('drill-download-outcomes');
  if (exportBtn) {
    const outcomeCount = (j.runs || []).reduce((n, r) => n + (r.outcomes || []).length, 0);
    exportBtn.disabled = outcomeCount === 0;
    exportBtn.title = outcomeCount
      ? `Download all ${outcomeCount} raw completions with their example reductions across ${(j.runs || []).length} run(s) as JSON Lines`
      : 'No outcomes yet — the download unlocks as examples finish';
  }

  const runs = j.runs || [];
  const isCompare = (j.adapters && j.adapters.length > 1) || runs.length > 1;
  const headerEl = document.getElementById('drill-headline');
  const compareEl = document.getElementById('drill-compare');
  const tagsEl = document.getElementById('drill-tags');

  if (runs.length === 0) {
    headerEl.innerHTML = `
      <div class="hint">${j.state === 'queued' ? 'Job is queued. Will start shortly.' : (j.state === 'running' ? 'Job is running. Live progress streaming…' : 'No completed runs yet.')}</div>
      ${j.progress && j.progress.examples_total > 0 ? `<div style="flex:1;"><div class="progress-bar-wrap" style="height:8px;"><div class="progress-bar-fill" style="width:${(j.progress.examples_completed / j.progress.examples_total * 100).toFixed(1)}%;"></div></div><div class="hint" style="font-size:11px; margin-top:4px;">${j.progress.examples_completed}/${j.progress.examples_total} · running ${(j.progress.running_accuracy*100).toFixed(0)}%</div></div>` : ''}
      ${renderBaseWeightSummary(j.base_weight_shard_manifest)}
      ${renderExecutionProvenanceSummary(j.execution_provenance)}
      ${renderReplaySummary(j)}`;
    wireBaseWeightCopy(headerEl);
    wireExecutionProvenanceCopy(headerEl);
    wireReplayCopy(headerEl);
    compareEl.hidden = true;
    tagsEl.hidden = true;
    document.getElementById('drill-outcomes').innerHTML = '<div class="eval-empty"><div class="eval-empty-body">Outcomes will appear here as they complete.</div></div>';
    document.getElementById('drill-detail').innerHTML = '<div class="detail-empty">Waiting on first results…</div>';
    updateDrillFilterCounts([]);
    return;
  }

  // Headline shows the *selected* run (default first).
  const run = runs[Math.min(drillSelectedRun, runs.length - 1)];
  const adapter = run.adapter || 'base';
  const m = run.metrics || {};
  headerEl.innerHTML = `
    ${ringHtml(m.accuracy, 'large')}
    <div style="flex:1; min-width:0;">
      <div style="font-size:14px; font-weight:600; margin-bottom:6px;">Adapter: <span style="color:var(--text);">${escapeHtml(adapter)}</span></div>
      <div class="hint" style="font-size:11px; margin-bottom:6px;">${escapeHtml(evalAggregationLabel(run.aggregation))} · ${(m.num_examples || 0).toLocaleString()} independent examples · ${(m.num_completions || 0).toLocaleString()} raw completions</div>
      <div class="drill-counts">
        <div class="count-cell"><span class="count-num" style="color:var(--success-fg);">${m.num_pass || 0}</span><span class="count-label">pass</span></div>
        <div class="count-cell"><span class="count-num" style="color:var(--danger-fg);">${m.num_fail || 0}</span><span class="count-label">fail</span></div>
        <div class="count-cell"><span class="count-num" style="color:var(--warning-fg);">${m.num_invalid || 0}</span><span class="count-label">invalid</span></div>
        <div class="count-cell"><span class="count-num" style="color:var(--text-muted);">${m.num_error || 0}</span><span class="count-label">error</span></div>
        ${m.latency && m.latency.p50_ms > 0 ? `<div class="count-cell"><span class="count-num">${m.latency.p50_ms.toFixed(0)}ms</span><span class="count-label">p50</span></div>` : ''}
        ${m.total_completion_tokens ? `<div class="count-cell"><span class="count-num">${(m.total_completion_tokens/1000).toFixed(1)}k</span><span class="count-label">tok out</span></div>` : ''}
      </div>
    </div>
    ${renderBaseWeightSummary(j.base_weight_shard_manifest)}
    ${renderExecutionProvenanceSummary(j.execution_provenance)}
    ${renderReplaySummary(j)}`;
  wireBaseWeightCopy(headerEl);
  wireExecutionProvenanceCopy(headerEl);
  wireReplayCopy(headerEl);

  // Compare matrix when multi-run
  if (isCompare && runs.length >= 2) {
    compareEl.hidden = false;
    const total = (run.metrics?.num_examples || 1);
    const verdictBadge = compareVerdictBadge(runs);
    compareEl.innerHTML = `<div class="eval-section-head" style="background:transparent; border:none; padding:0 0 4px 0; display:flex; align-items:center; gap:10px;">Adapter comparison${verdictBadge}</div>` +
      runs.map((r, i) => {
        const rm = r.metrics || {};
        const tot = Math.max(1, rm.num_examples || total);
        const pp = (rm.num_pass || 0) / tot * 100;
        const fp = (rm.num_fail || 0) / tot * 100;
        const ip = (rm.num_invalid || 0) / tot * 100;
        const ep = (rm.num_error || 0) / tot * 100;
        const a = r.adapter || 'base';
        const isSel = i === drillSelectedRun;
        return `<div class="compare-row" data-run-idx="${i}" style="cursor:pointer; ${isSel ? 'opacity:1;' : 'opacity:0.7;'}" title="Click to view this adapter's outcomes">
          <span class="compare-name" ${isSel ? 'style="color:var(--accent);"' : ''}>${escapeHtml(a)}</span>
          <div class="compare-bar">
            <div class="seg-pass" style="width:${pp}%;" title="pass ${rm.num_pass}"></div>
            <div class="seg-fail" style="width:${fp}%;" title="fail ${rm.num_fail}"></div>
            <div class="seg-invalid" style="width:${ip}%;" title="invalid ${rm.num_invalid}"></div>
            <div class="seg-error" style="width:${ep}%;" title="error ${rm.num_error}"></div>
          </div>
          <span class="compare-acc">${(rm.accuracy*100).toFixed(0)}%</span>
        </div>`;
      }).join('');
    compareEl.querySelectorAll('.compare-row').forEach(row => {
      row.addEventListener('click', () => {
        drillSelectedRun = parseInt(row.dataset.runIdx, 10);
        drillSelectedOutcome = null;
        renderDrillModal(false);
      });
    });
  } else {
    compareEl.hidden = true;
  }

  // Tag pass-rate bars
  const tagRates = m.pass_rate_by_tag || {};
  const tagEntries = Object.entries(tagRates);
  if (tagEntries.length) {
    tagsEl.hidden = false;
    tagsEl.innerHTML = '<div class="eval-section-head" style="background:transparent; border:none; padding:0 0 4px 0;">Pass rate by tag</div>' +
      tagEntries.map(([k, v]) => `<div class="tag-bar">
        <span class="tag-name">${escapeHtml(k)}</span>
        <div class="tag-track"><div class="tag-fill" style="width:${(v*100).toFixed(1)}%;"></div></div>
        <span class="tag-pct">${(v*100).toFixed(0)}%</span>
      </div>`).join('');
  } else {
    tagsEl.hidden = true;
  }

  // Outcomes list, filtered + searched
  renderDrillOutcomes();
  if (!preserveSelection || drillSelectedOutcome === null) {
    // Default: show first failure if any, else first outcome
    const exampleRows = drillExampleRows(run);
    const first = exampleRows.find(o => o.kind !== 'pass') || exampleRows[0];
    if (first) selectDrillOutcome(first);
    else document.getElementById('drill-detail').innerHTML = '<div class="detail-empty">No outcomes for this run.</div>';
  } else {
    // Re-find the selected outcome by id (it may have changed kind on a re-poll)
    const found = drillExampleRows(run).find(o => o.example_id === drillSelectedOutcome.example_id);
    if (found) selectDrillOutcome(found);
  }
}

function renderDrillOutcomes() {
  const j = drillJob;
  if (!j || !j.runs) return;
  const run = j.runs[Math.min(drillSelectedRun, j.runs.length - 1)];
  const all = drillExampleRows(run);
  // Counts always reflect the whole run (so the filter pills are stable)
  const counts = { all: all.length, pass: 0, fail: 0, invalid: 0, error: 0 };
  for (const o of all) counts[o.kind] = (counts[o.kind] || 0) + 1;
  updateDrillFilterCounts(counts);

  const filtered = all.filter(o => {
    if (drillFilter !== 'all' && o.kind !== drillFilter) return false;
    if (drillSearch) {
      const needle = drillSearch.toLowerCase();
      const hay = (o.example_id + ' ' + (o.completion_text || '') + ' ' + (o.detail || '')).toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    return true;
  });
  const el = document.getElementById('drill-outcomes');
  if (!filtered.length) {
    el.innerHTML = '<div class="eval-empty" style="border:none; background:transparent;"><div class="eval-empty-body">No examples match the current filter.</div></div>';
    return;
  }
  el.innerHTML = filtered.map(o => {
    const tags = (o.tags || []).slice(0, 2).map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('');
    const isSel = drillSelectedOutcome
      && drillSelectedOutcome.example_id === o.example_id
      && drillSelectedOutcome.completion_index === o.completion_index;
    const aggregate = o.aggregation_outcome;
    const completionSummary = aggregate && aggregate.completion_indices?.length > 1
      ? `<span class="hint">${aggregate.num_pass}/${aggregate.completion_indices.length} raw pass · rep #${aggregate.representative_completion_index}</span>`
      : '';
    return `<div class="outcome-item ${isSel ? 'selected' : ''}" data-example-id="${escapeHtml(o.example_id)}" data-completion-index="${o.completion_index}">
      <span class="outcome-badge ${o.kind}">${o.kind}</span>
      <div class="outcome-preview" title="${escapeHtml(o.completion_text || '')}">${escapeHtml(truncate(o.completion_text || '(empty)', 110))}</div>
      <div class="outcome-meta">
        ${o.latency_ms != null ? `<span class="hint">${o.latency_ms.toFixed(0)}ms</span>` : ''}
        ${completionSummary}
        ${tags}
      </div>
    </div>`;
  }).join('');
  el.querySelectorAll('.outcome-item').forEach(item => {
    item.addEventListener('click', () => {
      const id = item.dataset.exampleId;
      const idx = parseInt(item.dataset.completionIndex, 10);
      const found = all.find(o => o.example_id === id && o.completion_index === idx);
      if (found) selectDrillOutcome(found);
    });
  });
}

function updateDrillFilterCounts(counts) {
  for (const k of ['all', 'pass', 'fail', 'invalid', 'error']) {
    const el = document.getElementById('drill-count-' + k);
    if (el) el.textContent = (counts[k] || 0).toLocaleString();
  }
}

function selectDrillOutcome(o) {
  drillSelectedOutcome = { example_id: o.example_id, completion_index: o.completion_index };
  // Highlight in list
  document.querySelectorAll('.outcome-item').forEach(item => {
    const match = item.dataset.exampleId === o.example_id && parseInt(item.dataset.completionIndex, 10) === o.completion_index;
    item.classList.toggle('selected', match);
  });
  renderOutcomeDetail(o);
}

function renderOutcomeDetail(o) {
  const tags = (o.tags || []).map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('');
  const detail = document.getElementById('drill-detail');
  const example = drillExamplesById.get(o.example_id);
  const aggregate = o.aggregation_outcome;
  const aggregateSummary = aggregate && aggregate.completion_indices?.length > 1
    ? `<span class="tabular-nums hint" style="font-size:11px;">raw: ${aggregate.num_pass} pass · ${aggregate.num_fail} fail · ${aggregate.num_invalid} invalid · ${aggregate.num_error} error</span>`
    : '';
  // Prompt section: the chat history the model actually saw. When we have
  // it (suite was loadable + example_id matched), render each message as a
  // role-coded bubble; otherwise show a hint that the suite isn't available.
  const promptSection = example && example.messages && example.messages.length
    ? `<div class="detail-section">
        <h4>Prompt</h4>
        <div class="messages-list">
          ${example.messages.map(m => `<div class="chat-msg">
            <div class="role ${escapeHtml(m.role)}">${escapeHtml(m.role)}</div>
            <div class="body">${escapeHtml(m.content)}</div>
          </div>`).join('')}
        </div>
      </div>`
    : `<div class="detail-section">
        <h4>Prompt</h4>
        <div class="hint" style="font-size:11px;">Suite content not available locally — drill into a registered suite to see the prompt the model saw.</div>
      </div>`;
  // Side-by-side target vs got. Only render when we have a target; the
  // "json_validity" / "any_block" scorers don't always set one.
  const target = example && example.target;
  const passClass = o.kind === 'pass' ? 'tg-pass' : (o.kind === 'fail' ? 'tg-fail' : '');
  const tgSection = target != null
    ? `<div class="detail-section">
        <h4>Target ↔ Got</h4>
        <div class="detail-tg">
          <div class="tg-cell">
            <div class="tg-label">Expected target</div>
            <pre>${escapeHtml(target)}</pre>
          </div>
          <div class="tg-cell ${passClass}">
            <div class="tg-label">Model output</div>
            <pre>${escapeHtml(o.completion_text || '(empty)')}</pre>
          </div>
        </div>
      </div>`
    : `<div class="detail-section">
        <h4>Model output</h4>
        <pre class="${passClass}" style="margin:0; font-family:var(--font-mono); font-size:12px; line-height:1.55; white-space:pre-wrap; word-break:break-word; padding:10px; background:var(--surface); border:1px solid var(--border); border-radius:6px;">${escapeHtml(o.completion_text || '(empty)')}</pre>
      </div>`;
  // Scorer section: kind + per-example detail
  const scorerKind = (example && example.scorer && example.scorer.kind) || drillJob?.runs?.[0]?.metrics?.by_scorer?.[0]?.scorer_kind || '';
  const scorerSection = `<div class="detail-section">
    <h4>Scorer</h4>
    <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
      ${scorerKind ? `<span class="scorer-badge">${escapeHtml(scorerKind)}</span>` : ''}
      <span class="tabular-nums hint">${aggregate && aggregate.completion_indices?.length > 1 ? 'aggregate' : ''} score ${(o.score).toFixed(3)}</span>
      ${aggregate && aggregate.completion_indices?.length > 1 && o.raw_score != null ? `<span class="tabular-nums hint">representative ${escapeHtml(o.raw_kind || '')} ${(o.raw_score).toFixed(3)}</span>` : ''}
    </div>
    ${o.detail ? `<div style="font-family:var(--font-mono); font-size:12px; padding:10px; background:var(--surface); border:1px solid var(--border); border-radius:6px;">${escapeHtml(o.detail)}</div>` : '<div class="hint" style="font-size:11px;">No scorer commentary.</div>'}
  </div>`;
  // Per-outcome action toolbar: copy raw outputs, replay the failing
  // prompt in the playground. Stash the prompt text on a dataset attribute
  // so the click handler doesn't have to walk back through drillExamplesById.
  const promptForReplay = example && example.messages && example.messages.length
    ? example.messages
        .filter(m => m.role !== 'system')
        .map(m => `[${m.role}] ${m.content}`)
        .join('\n\n')
    : '';
  const userMsg = example && example.messages
    ? (example.messages.filter(m => m.role === 'user').pop()?.content || '')
    : '';
  const actionsHtml = `<div class="outcome-actions" style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px;">
    <button type="button" class="btn btn-sm" data-outcome-copy="completion" title="Copy the model's output"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg> Copy output</button>
    ${promptForReplay ? `<button type="button" class="btn btn-sm" data-outcome-copy="prompt" title="Copy the full prompt as role-prefixed text"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg> Copy prompt</button>` : ''}
    ${o.generation_seed == null ? '' : `<button type="button" class="btn btn-sm" data-outcome-copy="seed" title="Copy the exact per-completion decoder seed"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg> Copy seed</button>`}
    ${userMsg ? `<button type="button" class="btn btn-sm" data-outcome-replay="1" title="Drop the last user message into the playground so you can iterate"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-chat"></use></svg> Replay in playground</button>` : ''}
    ${(o.kind === 'fail' || o.kind === 'invalid') && userMsg ? `<button type="button" class="btn btn-sm" data-outcome-correct="1" title="Capture this failing example into the corrections basket — write the ideal answer, then train"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-pencil"></use></svg> Add to corrections</button>` : ''}
  </div>`;
  detail.innerHTML = `
    <div class="detail-section">
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">
        <span class="outcome-badge ${o.kind}">${o.kind}</span>
        <strong style="font-family:var(--font-mono); font-size:13px;">${escapeHtml(o.example_id)}</strong>
        ${o.completion_index > 0 ? `<span class="hint">completion #${o.completion_index}</span>` : ''}
        ${o.latency_ms != null ? `<span class="tabular-nums hint" style="font-size:11px;">${o.latency_ms.toFixed(0)}ms</span>` : ''}
        ${o.prompt_tokens != null ? `<span class="tabular-nums hint" style="font-size:11px;">${o.prompt_tokens}→${o.completion_tokens || 0} tok</span>` : ''}
        ${o.generation_seed == null ? '' : `<span class="tabular-nums hint" style="font-size:11px;" title="Derived per-completion decoder seed">seed ${escapeHtml(String(o.generation_seed))}</span>`}
        ${aggregateSummary}
      </div>
      ${actionsHtml}
      <div>${tags}</div>
    </div>
    ${promptSection}
    ${tgSection}
    ${scorerSection}
  `;
  // Wire the per-outcome action buttons.
  detail.querySelectorAll('[data-outcome-copy]').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.outcomeCopy;
      const text = key === 'completion'
        ? (o.completion_text || '')
        : key === 'seed' ? String(o.generation_seed) : promptForReplay;
      copyText(text, btn).then(() => {
        if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = text;
        toast(`Copied ${key}`, 'ok');
      });
    });
  });
  detail.querySelectorAll('[data-outcome-correct]').forEach(btn => {
    btn.addEventListener('click', () => {
      addCorrectionFromEvalOutcome(o, example, scorerKind);
    });
  });
  detail.querySelectorAll('[data-outcome-replay]').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.getElementById('chat-input');
      if (input) {
        input.value = userMsg;
        if (typeof autoresizeChatInput === 'function') autoresizeChatInput();
        if (typeof updateChatSendState === 'function') updateChatSendState();
      }
      closeDrillModal();
      selectPage('playground');
      if (input) setTimeout(() => input.focus(), 50);
    });
  });
}

document.getElementById('drill-close')?.addEventListener('click', userCloseDrillModal);
document.getElementById('eval-drill-modal')?.addEventListener('click', ev => {
  if (ev.target.id === 'eval-drill-modal') userCloseDrillModal();
});
// Raw JSON toggle — same pattern as the request drill modal's `raw` button:
// click appends a pretty-printed <pre> of the cached job to the modal
// content, click again removes it.
document.getElementById('drill-raw')?.addEventListener('click', () => {
  if (!drillJob) return;
  const content = document.getElementById('drill-content');
  if (!content) return;
  const existing = content.querySelector('#drill-raw-block');
  if (existing) { existing.remove(); return; }
  const pre = document.createElement('pre');
  pre.id = 'drill-raw-block';
  pre.className = 'req-pre';
  pre.style.cssText = 'max-height:50vh; margin:var(--space-4) var(--space-5);';
  pre.textContent = JSON.stringify(drillJob, null, 2);
  content.appendChild(pre);
  pre.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
/// Trigger a browser download via a temporary object URL. The URL is
/// revoked right after the click so repeated downloads don't pin every
/// blob in memory for the lifetime of the page.
function downloadBlobAsFile(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
/// One JSON line per raw completion, across every finished run of the job. Each
/// line is standalone: suite/job/adapter context first, then the outcome's
/// verdict, its independent example reduction, optional diagnostics, and the
/// completion last.
function buildDrillOutcomesJsonl(j) {
  const lines = [];
  for (const run of (j.runs || [])) {
    (run.outcomes || []).forEach((o, i) => {
      const aggregate = (run.aggregated_outcomes || []).find(a => a.example_id === o.example_id);
      const line = {
        suite: run.suite_name || j.suite_name || 'eval',
        job_id: j.job_id,
        adapter: run.adapter || 'base',
        example_index: i,
        example_id: o.example_id,
        completion_index: o.completion_index,
        kind: o.kind,
        score: o.score,
        aggregation: run.aggregation || { kind: 'single' },
      };
      if (aggregate) line.aggregated_example_outcome = aggregate;
      if (j.effective_seed != null) line.effective_seed = String(j.effective_seed);
      if (j.seed_derivation != null) line.seed_derivation = j.seed_derivation;
      if (j.base_weight_shard_manifest != null) line.base_weight_shard_manifest = j.base_weight_shard_manifest;
      if (j.execution_provenance != null) line.execution_provenance = j.execution_provenance;
      if (o.generation_seed != null) line.generation_seed = String(o.generation_seed);
      if (o.detail != null) line.detail = o.detail;
      if (o.latency_ms != null) line.latency_ms = o.latency_ms;
      if (o.prompt_tokens != null) line.prompt_tokens = o.prompt_tokens;
      if (o.completion_tokens != null) line.completion_tokens = o.completion_tokens;
      if (o.tags && o.tags.length) line.tags = o.tags;
      if (o.metadata != null) line.metadata = o.metadata;
      if (o.reasoning_text != null) line.reasoning_text = o.reasoning_text;
      if (o.raw_completion_text != null) line.raw_completion_text = o.raw_completion_text;
      if (o.thinking_budget != null) line.thinking_budget = o.thinking_budget;
      if (o.unclosed_thinking === true) line.unclosed_thinking = true;
      line.completion_text = o.completion_text || '';
      lines.push(JSON.stringify(line));
    });
  }
  return lines;
}
document.getElementById('drill-download-outcomes')?.addEventListener('click', () => {
  if (!drillJob) return;
  const lines = buildDrillOutcomesJsonl(drillJob);
  if (!lines.length) return; // button is disabled in this state; belt and braces
  const suiteSlug = String(drillJob.suite_name || 'eval').replace(/[^A-Za-z0-9._-]+/g, '-');
  const filename = `${suiteSlug}-${drillJob.job_id.slice(0, 8)}.outcomes.jsonl`;
  downloadBlobAsFile(filename, new Blob([lines.join('\n') + '\n'], { type: 'application/jsonl' }));
  toast(`Downloaded ${lines.length} outcome${lines.length === 1 ? '' : 's'} as ${filename}`, 'ok');
});
document.getElementById('drill-search')?.addEventListener('input', ev => {
  drillSearch = ev.target.value;
  renderDrillOutcomes();
});
document.querySelectorAll('[data-drill-filter]').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('[data-drill-filter]').forEach(other => other.classList.toggle('active', other === b));
    drillFilter = b.dataset.drillFilter;
    renderDrillOutcomes();
  });
});

document.getElementById('drill-cancel')?.addEventListener('click', async () => {
  if (!drillJob) return;
  const mode = document.getElementById('drill-cancel')?.dataset?.mode || 'cancel';
  const verbMsg = mode === 'delete'
    ? `Permanently delete eval job ${drillJob.job_id.slice(0, 8)}? Adapter weights are untouched; only the tracking entry and the on-disk archive file are removed.`
    : `Cancel eval job ${drillJob.job_id.slice(0, 8)}?`;
  if (!confirm(verbMsg)) return;
  try {
    await api('/v1/eval/jobs/' + encodeURIComponent(drillJob.job_id), { method: 'DELETE' });
    toast(mode === 'delete' ? 'Eval job deleted' : 'Cancelled eval job', 'ok');
    userCloseDrillModal();
    refreshEvalJobs();
  } catch (e) { toast((mode === 'delete' ? 'Delete' : 'Cancel') + ' failed: ' + e.message, 'err'); }
});

document.getElementById('drill-rerun')?.addEventListener('click', async () => {
  if (!drillJob) return;
  const selectedRun = drillJob.runs?.[Math.min(drillSelectedRun, (drillJob.runs?.length || 1) - 1)];
  const failing = drillExampleRows(selectedRun)
    .filter(o => o.kind !== 'pass').length;
  if (!failing) {
    toast('No non-passing examples to re-run', 'ok');
    return;
  }
  if (!confirm(`Re-run ${failing} failing example(s)?`)) return;
  try {
    const res = await api('/v1/eval/jobs/' + encodeURIComponent(drillJob.job_id) + '/rerun', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({}),
    });
    toast('Queued re-run as ' + res.job_id.slice(0, 8), 'ok');
    closeDrillModal();
    refreshEvalJobs();
    setTimeout(() => openDrillModal(res.job_id), 200);
  } catch (e) { toast('Re-run failed: ' + e.message, 'err'); }
});

document.getElementById('drill-replay')?.addEventListener('click', async () => {
  if (!drillJob) return;
  const runIndex = Math.min(drillSelectedRun, Math.max(0, (drillJob.runs?.length || 1) - 1));
  const run = drillJob.runs?.[runIndex];
  if (!run?.replay_record) {
    toast('This run predates strict replay evidence', 'err');
    return;
  }
  if (!confirm(`Strictly replay run ${runIndex} and compare every raw decoder completion byte?`)) return;
  try {
    const res = await api('/v1/eval/jobs/' + encodeURIComponent(drillJob.job_id) + '/replay', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ run_index: runIndex }),
    });
    toast('Queued strict replay as ' + res.job_id.slice(0, 8), 'ok');
    closeDrillModal();
    refreshEvalJobs();
    setTimeout(() => openDrillModal(res.job_id), 200);
  } catch (e) { toast('Strict replay refused: ' + e.message, 'err'); }
});

// Modal-scoped keyboard shortcuts: / focuses search; J/K scroll through
// outcomes (vim-style); R triggers re-run. Esc is the shared modal
// manager's (routes through userCloseDrillModal via the layer's onClose).
document.addEventListener('keydown', ev => {
  const modal = document.getElementById('eval-drill-modal');
  if (modal.hidden) return;
  // Only while this drill is the TOP modal — cmdk over it owns the keys.
  if (modalStackTop()?.el !== modal) return;
  const tag = (ev.target.tagName || '').toUpperCase();
  // When focused in an input, only Cmd/Ctrl shortcuts fire.
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (ev.key === '/') {
    ev.preventDefault();
    document.getElementById('drill-search').focus();
  } else if (ev.key === 'r' || ev.key === 'R') {
    ev.preventDefault();
    document.getElementById('drill-rerun').click();
  } else if (ev.key === 'j' || ev.key === 'ArrowDown') {
    ev.preventDefault();
    moveDrillSelection(1);
  } else if (ev.key === 'k' || ev.key === 'ArrowUp') {
    ev.preventDefault();
    moveDrillSelection(-1);
  }
});

function moveDrillSelection(delta) {
  const list = Array.from(document.querySelectorAll('.outcome-item'));
  if (!list.length) return;
  const cur = list.findIndex(el => el.classList.contains('selected'));
  const next = Math.max(0, Math.min(list.length - 1, (cur < 0 ? 0 : cur + delta)));
  list[next].click();
  list[next].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

/* ---------- Judgments — keyboard-first A/B with streaming ---------- */

let activeJudgmentDataset = null;
let pendingJudgmentPair = null;
let judgmentStreams = { a: null, b: null };  // AbortControllers
let judgmentAutoAdvance = true;

async function refreshJudgments() {
  try {
    const d = await api('/v1/judgments');
    const items = d.judgments || [];
    const el = document.getElementById('judgments-list');
    if (!items.length) {
      el.className = 'eval-empty';
      setListHtml(el, 'empty', `
        <div class="eval-empty-icon"><svg class="icn"><use href="#i-scale"></use></svg></div>
        <div class="eval-empty-title">No judgment datasets yet</div>
        <div class="eval-empty-body">Create a dataset, then judge model outputs A/B/Tie. After ~20 picks you can compile them into SFT data and train a local judge LoRA — no frontier LLM in the loop.</div>
        <button class="eval-empty-cta" type="button" onclick="document.getElementById('judgment-create-name').focus()">Create your first dataset</button>`);
    } else {
      el.className = '';
      // Key on the displayed payload fields plus activeJudgmentDataset —
      // the "(active)" hint and the Continue/Judge button label depend on
      // it, so switching datasets must repaint even with identical data.
      const listKey = 'list:' + JSON.stringify([
        activeJudgmentDataset,
        items.map(m => [m.name, m.description, m.num_rows, m.winner_histogram]),
      ]);
      const listHtml = items.map(m => {
        const winners = m.winner_histogram || {};
        const total = (winners.a || 0) + (winners.b || 0) + (winners.tie || 0);
        const aPct = total ? ((winners.a || 0) / total * 100).toFixed(0) : 0;
        const bPct = total ? ((winners.b || 0) / total * 100).toFixed(0) : 0;
        const isActive = activeJudgmentDataset === m.name;
        const winnerBar = total > 0 ? `<div style="display:flex; height:6px; border-radius:3px; overflow:hidden; background:var(--surface-3); width:120px;">
          <div style="background:var(--info-fg); width:${aPct}%;" title="A: ${winners.a || 0}"></div>
          <div style="background:var(--warning-fg); width:${total ? ((winners.tie || 0) / total * 100).toFixed(0) : 0}%;" title="Tie: ${winners.tie || 0}"></div>
          <div style="background:var(--accent); width:${bPct}%;" title="B: ${winners.b || 0}"></div>
        </div>` : '';
        return `<div class="eval-row eval-row-judgments">
          <div>
            <div class="row-title">${escapeHtml(m.name)}${isActive ? ' <span class="hint">(active)</span>' : ''}</div>
            <div class="row-sub">${escapeHtml(m.description || 'No description')}</div>
          </div>
          <div class="tabular-nums">${m.num_rows} judgments</div>
          <div style="display:flex; gap:8px; align-items:center; font-size:11px;">${winnerBar}<span class="hint">A ${winners.a||0} · T ${winners.tie||0} · B ${winners.b||0}</span></div>
          <div class="row-actions">
            <button type="button" class="btn btn-primary btn-sm" data-action="judge" data-name="${escapeHtml(m.name)}">${isActive ? 'Continue' : 'Judge →'}</button>
            <button type="button" class="btn btn-sm" data-action="promote" data-name="${escapeHtml(m.name)}">Promote</button>
            <button type="button" class="btn btn-sm" data-action="del" data-name="${escapeHtml(m.name)}">Delete</button>
          </div>
        </div>`;
      }).join('');
      if (setListHtml(el, listKey, listHtml)) {
        el.querySelectorAll('button[data-action]').forEach(b => {
          const name = b.dataset.name;
          if (b.dataset.action === 'judge') {
            b.addEventListener('click', () => openJudgmentViewer(name));
          } else if (b.dataset.action === 'promote') {
            b.addEventListener('click', () => openJudgmentCompile(name));
          } else if (b.dataset.action === 'del') {
            b.addEventListener('click', async () => {
              if (!confirm(`Delete judgment dataset "${name}"? Provenance is gone for good.`)) return;
              try {
                await api('/v1/judgments/' + encodeURIComponent(name), { method: 'DELETE' });
                if (activeJudgmentDataset === name) {
                  activeJudgmentDataset = null;
                  document.getElementById('judgment-viewer').hidden = true;
                  document.getElementById('judgment-compile').hidden = true;
                }
                refreshJudgments();
              } catch (e) { toast('Delete failed: ' + e.message, 'err'); }
            });
          }
        });
      }
    }
  } catch (e) {
    // Error-specific key: recovery payloads (even identical ones) repaint.
    setListHtml(document.getElementById('judgments-list'), 'err:' + e.message,
      `<div class="eval-empty"><div class="eval-empty-body">Failed: ${escapeHtml(e.message)}</div></div>`);
  }
  refreshAdapterDropdowns();
}

document.getElementById('judgment-create-btn')?.addEventListener('click', async () => {
  const name = document.getElementById('judgment-create-name').value.trim();
  if (!name) { toast('Name is required', 'err'); return; }
  try {
    await api('/v1/judgments', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
    toast('Created judgment dataset', 'ok');
    document.getElementById('judgment-create-name').value = '';
    refreshJudgments();
    openJudgmentViewer(name);
  } catch (e) { toast('Create failed: ' + e.message, 'err'); }
});

document.getElementById('judgment-autoadvance')?.addEventListener('change', ev => {
  judgmentAutoAdvance = ev.target.checked;
});

function openJudgmentViewer(name) {
  activeJudgmentDataset = name;
  document.getElementById('judgment-viewer').hidden = false;
  document.getElementById('judgment-pair').hidden = true;
  document.getElementById('judgment-actions').hidden = true;
  document.getElementById('judgment-rows-count').textContent = `Judging into "${name}". Press G to generate, A/B/T/S to vote.`;
  document.getElementById('judgment-compile').hidden = true;
  document.getElementById('judgment-prompt').focus();
  document.getElementById('judgment-viewer').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function openJudgmentCompile(name) {
  activeJudgmentDataset = name;
  document.getElementById('judgment-compile').hidden = false;
  document.getElementById('compile-sft-name').value = name + '-sft';
  document.getElementById('compile-output').innerHTML = '';
  document.getElementById('judgment-compile').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function abortPendingStreams() {
  for (const k of ['a', 'b']) {
    if (judgmentStreams[k]) {
      try { judgmentStreams[k].abort(); } catch (_) {}
      judgmentStreams[k] = null;
    }
  }
}

async function streamCompletion(slot, body) {
  const ctrl = new AbortController();
  judgmentStreams[slot] = ctrl;
  const target = document.getElementById('judgment-' + slot + '-text');
  target.textContent = '';
  target.classList.add('token-cursor');
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream', 'X-Kiln-Client': 'dashboard' },
      body: JSON.stringify({ ...body, stream: true }),
      signal: ctrl.signal,
    });
    if (!res.ok || !res.body) {
      const errText = await res.text().catch(() => `HTTP ${res.status}`);
      // The mock backend rejects streaming with a clear error code. Fall
      // back to a plain non-streaming completion so the judgment flow
      // still works without a real model loaded.
      if (errText.includes('streaming_not_supported')) {
        return await nonStreamingCompletion(slot, body, ctrl, target);
      }
      throw new Error(errText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let acc = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') return acc;
        try {
          const chunk = JSON.parse(payload);
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) {
            acc += delta;
            target.textContent = acc;
          }
        } catch (_) { /* tolerate non-JSON keepalives */ }
      }
    }
    return acc;
  } finally {
    target.classList.remove('token-cursor');
    if (judgmentStreams[slot] === ctrl) judgmentStreams[slot] = null;
  }
}

async function nonStreamingCompletion(slot, body, ctrl, target) {
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Kiln-Client': 'dashboard' },
    body: JSON.stringify({ ...body, stream: false }),
    signal: ctrl.signal,
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.error?.message || `HTTP ${res.status}`);
  }
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content || '';
  target.textContent = text;
  return text;
}

async function generateJudgmentPair() {
  if (!activeJudgmentDataset) { toast('Pick a judgment dataset first', 'err'); return; }
  const promptText = document.getElementById('judgment-prompt').value.trim();
  if (!promptText) { toast('Enter a prompt to compare on', 'err'); return; }
  abortPendingStreams();
  const adapterA = document.getElementById('judgment-adapter-a').value;
  const adapterB = document.getElementById('judgment-adapter-b').value;
  const temperature = parseFloat(document.getElementById('judgment-temp').value || '0.7');
  const baseBody = {
    messages: [{ role: 'user', content: promptText }],
    temperature, top_p: 1.0, max_tokens: 512,
  };
  const aBody = { ...baseBody }; if (adapterA) aBody.adapter = adapterA;
  const bBody = { ...baseBody }; if (adapterB) bBody.adapter = adapterB;

  document.getElementById('judgment-pair').hidden = false;
  document.getElementById('judgment-actions').hidden = false;
  document.getElementById('judgment-a-adapter').textContent = adapterA || 'base';
  document.getElementById('judgment-b-adapter').textContent = adapterB || 'base';
  // Stub the pair so vote actions know what to record. Final text is set
  // after streams complete.
  pendingJudgmentPair = {
    prompt: [{ role: 'user', content: promptText }],
    adapter_a: adapterA || null,
    adapter_b: adapterB || null,
    response_a: '',
    response_b: '',
  };

  // Run both streams concurrently. Either one's failure shouldn't kill the other.
  const [a, b] = await Promise.allSettled([
    streamCompletion('a', aBody),
    streamCompletion('b', bBody),
  ]);
  if (a.status === 'fulfilled') pendingJudgmentPair.response_a = a.value;
  else document.getElementById('judgment-a-text').innerHTML = icon('warning','icn-sm') + ' ' + escapeHtml(a.reason?.message || 'failed');
  if (b.status === 'fulfilled') pendingJudgmentPair.response_b = b.value;
  else document.getElementById('judgment-b-text').innerHTML = icon('warning','icn-sm') + ' ' + escapeHtml(b.reason?.message || 'failed');
}

document.getElementById('judgment-generate-btn')?.addEventListener('click', () => generateJudgmentPair());

// One toast with an Undo action for a just-recorded judgment. The rows POST
// returns `judgment_id` (the appended row's stable id) — Undo DELETEs that
// exact row, refreshes the visible counts, and confirms. A misclicked vote
// no longer poisons the dataset permanently. `fired` guards double-fires:
// actionToast removes the toast on click, but a queued second click must
// not double-DELETE (the second DELETE would 404 and toast a scary error).
function recordedJudgmentToast(message, datasetName, judgmentId) {
  if (!judgmentId) { toast(message, 'ok'); return; }  // no id, no Undo
  let fired = false;
  actionToast(message, 'ok', [{
    label: 'Undo',
    onClick: async () => {
      if (fired) return;
      fired = true;
      try {
        const m = await api('/v1/judgments/' + encodeURIComponent(datasetName) + '/rows/' + encodeURIComponent(judgmentId), { method: 'DELETE' });
        if (activeJudgmentDataset === datasetName) {
          document.getElementById('judgment-rows-count').textContent =
            `${m.num_rows} judgments in "${datasetName}". Press G to generate the next pair (A/B/T/S to vote).`;
        }
        refreshJudgments();
        toast(`Undone — judgment removed from "${datasetName}"`, 'ok');
      } catch (e) {
        // Leave the counts as they are — the row may still exist server-side.
        toast('Undo failed: ' + e.message, 'err');
      }
    },
  }]);
}

async function recordJudgment(winner) {
  if (!activeJudgmentDataset || !pendingJudgmentPair) return;
  // If a stream is still going, capture whatever has been emitted so far so
  // the user can vote mid-stream.
  if (!pendingJudgmentPair.response_a) {
    pendingJudgmentPair.response_a = document.getElementById('judgment-a-text').textContent;
  }
  if (!pendingJudgmentPair.response_b) {
    pendingJudgmentPair.response_b = document.getElementById('judgment-b-text').textContent;
  }
  abortPendingStreams();
  const note = document.getElementById('judgment-note').value.trim();
  const tags = document.getElementById('judgment-tags').value.split(',').map(s => s.trim()).filter(Boolean);
  const body = {
    ...pendingJudgmentPair,
    winner,
    note: note || null,
    tags,
  };
  const dataset = activeJudgmentDataset;
  try {
    const m = await api('/v1/judgments/' + encodeURIComponent(dataset) + '/rows', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    document.getElementById('judgment-rows-count').textContent =
      `${m.num_rows} judgments in "${dataset}". Press G to generate the next pair (A/B/T/S to vote).`;
    document.getElementById('judgment-note').value = '';
    pendingJudgmentPair = null;
    document.getElementById('judgment-actions').hidden = true;
    document.getElementById('judgment-prompt').value = '';
    document.getElementById('judgment-a-text').textContent = '';
    document.getElementById('judgment-b-text').textContent = '';
    document.getElementById('judgment-pair').hidden = true;
    refreshJudgments();
    const winnerLabel = { a: 'A wins', b: 'B wins', tie: 'Tie', skip: 'Skip' }[winner] || winner;
    recordedJudgmentToast(`Recorded ${winnerLabel} in "${dataset}"`, dataset, m.judgment_id);
    if (judgmentAutoAdvance) {
      // Re-focus the prompt for the next round.
      setTimeout(() => document.getElementById('judgment-prompt').focus(), 50);
    }
  } catch (e) { toast('Save failed: ' + e.message, 'err'); }
}

document.getElementById('judgment-pick-a')?.addEventListener('click', () => recordJudgment('a'));
document.getElementById('judgment-pick-b')?.addEventListener('click', () => recordJudgment('b'));
document.getElementById('judgment-pick-tie')?.addEventListener('click', () => recordJudgment('tie'));
document.getElementById('judgment-pick-skip')?.addEventListener('click', () => recordJudgment('skip'));

// Click reply card itself to vote — visual and obvious.
document.getElementById('judgment-card-a')?.addEventListener('click', () => {
  if (pendingJudgmentPair && document.getElementById('judgment-actions') && !document.getElementById('judgment-actions').hidden) {
    recordJudgment('a');
  }
});
document.getElementById('judgment-card-b')?.addEventListener('click', () => {
  if (pendingJudgmentPair && document.getElementById('judgment-actions') && !document.getElementById('judgment-actions').hidden) {
    recordJudgment('b');
  }
});

// Keyboard shortcuts for judgment voting.
document.addEventListener('keydown', ev => {
  // Only when judgment view is active and visible
  const evalsActive = document.getElementById('page-evals')?.classList.contains('active');
  const judgmentTabActive = document.getElementById('evals-tab-judgments')?.classList.contains('active');
  if (!evalsActive || !judgmentTabActive) return;
  // Don't intercept when typing in inputs
  const tag = (ev.target.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    // Special case: in the prompt textarea, Cmd/Ctrl+Enter still triggers generate.
    if ((ev.key === 'Enter') && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      generateJudgmentPair();
    }
    return;
  }
  if (document.getElementById('judgment-actions')?.hidden) {
    // Pre-vote: G or Enter generates a pair.
    if (ev.key === 'g' || ev.key === 'G' || ev.key === 'Enter') {
      ev.preventDefault();
      generateJudgmentPair();
    }
    return;
  }
  // Voting mode
  if (ev.key === 'a' || ev.key === 'A' || ev.key === 'ArrowLeft') { ev.preventDefault(); recordJudgment('a'); }
  else if (ev.key === 'b' || ev.key === 'B' || ev.key === 'ArrowRight') { ev.preventDefault(); recordJudgment('b'); }
  else if (ev.key === 't' || ev.key === 'T' || ev.key === 'ArrowUp') { ev.preventDefault(); recordJudgment('tie'); }
  else if (ev.key === 's' || ev.key === 'S' || ev.key === 'ArrowDown') { ev.preventDefault(); recordJudgment('skip'); }
});

document.getElementById('compile-btn')?.addEventListener('click', async () => {
  if (!activeJudgmentDataset) { toast('No judgment dataset selected', 'err'); return; }
  const output_dataset = document.getElementById('compile-sft-name').value.trim();
  if (!output_dataset) { toast('Provide an output SFT dataset name', 'err'); return; }
  try {
    const res = await api('/v1/judgments/' + encodeURIComponent(activeJudgmentDataset) + '/compile', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ output_dataset, include_skips: false }),
    });
    // One-click crank: the dataset is ready — go straight to training a
    // judge LoRA on it instead of sending the user off to find the
    // Training tab with an instruction string.
    document.getElementById('compile-output').innerHTML =
      `<div style="padding:10px; background:var(--success-bg); border:1px solid var(--success-bd); border-radius:6px; color:var(--success-fg); font-size:12px;">
        ${icon('check','icn-sm')} Compiled <strong>${res.rows}</strong> judgments into SFT dataset <code>${escapeHtml(res.dataset.name)}</code> (${res.dataset.num_rows} rows).
        <button type="button" class="btn btn-primary btn-sm" style="margin-left:8px;" id="compile-train-judge-btn">${icon('flask','icn-sm')} Train judge LoRA now</button>
      </div>`;
    document.getElementById('compile-train-judge-btn')?.addEventListener('click', () => {
      trainFromDataset(res.dataset.name, 'sft');
    });
    toast('Compiled to SFT', 'ok');
    refreshDatasets();
  } catch (e) { toast('Compile failed: ' + e.message, 'err'); }
});

document.getElementById('validate-btn')?.addEventListener('click', async () => {
  if (!activeJudgmentDataset) return;
  const adapter = document.getElementById('compile-judge-adapter').value;
  const holdout_n = parseInt(document.getElementById('compile-holdout').value, 10) || 20;
  if (!adapter) { toast('Pick an adapter to validate', 'err'); return; }
  try {
    const res = await api('/v1/judgments/' + encodeURIComponent(activeJudgmentDataset) + '/validate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ adapter, holdout_n }),
    });
    document.getElementById('compile-output').innerHTML =
      `<div style="padding:10px; background:var(--info-bg); border:1px solid var(--info-bd); border-radius:6px; color:var(--info-fg); font-size:12px;">
        Queued validation as eval job <code>${escapeHtml(res.eval_job_id)}</code>. Switching to the Jobs tab…
      </div>`;
    refreshEvalJobs();
    setTimeout(() => {
      document.getElementById('evals-tab-jobs')?.click();
      openDrillModal(res.eval_job_id);
    }, 400);
  } catch (e) { toast('Validate failed: ' + e.message, 'err'); }
});

// Show the first-time onboarding banner unless the user has dismissed it.
if (!localStorage.getItem('kiln-evals-onboarded')) {
  document.getElementById('evals-onboarding').hidden = false;
}

// Initial adapter dropdown population is universal (used by Training and
// Playground forms too). Eval-scoped lists are lazy: they fetch when the
// Evals page is first activated (see selectPage / refreshActiveEvalSubTab)
// and on every polling tick while the page is visible (below). This avoids
// 4× /v1/eval/* + /v1/judgments fetches firing on every dashboard load.
refreshAdapterDropdowns();
if (document.getElementById('page-evals')?.classList.contains('active')) {
  refreshActiveEvalSubTab();
}

// Periodic refresh — only updates the active sub-tab so we don't thrash.
// Every sub-tab refreshes on the same 1.5s tick (so a running job's progress
// feels alive); the content-keyed renders (setListHtml) make the unchanged
// ticks free instead of clobbering hover/selection/open dropdowns.
setInterval(() => {
  const evalsPage = document.getElementById('page-evals');
  if (!evalsPage || !evalsPage.classList.contains('active')) return;
  const active = evalsPage.querySelector('.tab.active')?.dataset?.tab;
  if (active === 'jobs')      refreshEvalJobs();
  else if (active === 'datasets') refreshDatasets();
  else if (active === 'suites')   refreshSuites();
  else if (active === 'judgments') refreshJudgments();
}, 1500);
