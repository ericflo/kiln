
// --- Training Queue ---
const cancellingTrainingJobIds = new Set();
// Flat snapshot of the latest /v1/train/queue payload so the command
// palette (and any other consumer) can search training jobs without
// re-issuing the request. Updated on every pollTraining tick.
// null until the first SUCCESSFUL fetch: an unfetched (or failing) queue is
// unknown, not empty — seeding an empty shape here made selectPage('training')
// auto-switch to the SFT form during outages, hiding the failure panel and
// its Retry button.
let trainingJobsCache = null;
// Skip the wholesale `tab-queue` innerHTML rewrite when nothing changed
// (running progress, queued list identity, recent finish-state). Mirrors
// the `lastAdaptersKey` guard on the adapters tab.
let lastTrainingKey = null;

// Job states from the previous poll, so finishing is an EVENT the UI announces
// (with the next action attached) instead of a silent row moving between lists.
let prevTrainingStates = null;
function detectTrainingTransitions(data) {
  const now = new Map();
  if (data.running) now.set(data.running.job_id, 'running');
  (data.queued || []).forEach(j => now.set(j.job_id, 'queued'));
  (data.completed || []).forEach(j => now.set(j.job_id, (j.state || 'completed').toString().toLowerCase()));
  if (prevTrainingStates) {
    for (const [id, state] of now) {
      const prev = prevTrainingStates.get(id);
      if (prev === state) continue;
      // Start is an announce-only event (no toast — the submit flow already
      // confirms visually): a job begins running that wasn't running before,
      // whether it stepped queued→running or appeared mid-poll already running.
      if (state === 'running') {
        const adapter = (data.running && data.running.adapter_name) || 'adapter';
        announceStatus('training-queue-status', `Training started: ${adapter}.`);
        continue;
      }
      // Only announce jobs we watched run/queue in THIS session — never history.
      if (!prev || (prev !== 'running' && prev !== 'queued')) continue;
      const j = (data.completed || []).find(x => x.job_id === id) || {};
      const adapter = j.adapter_name || 'adapter';
      if (state === 'completed') {
        announceStatus('training-queue-status', `Training completed: ${adapter} is ready.`);
        actionToast(`${adapter} finished training — it's ready${j.job_type ? ' (' + j.job_type + ')' : ''}.`, 'ok', [
          { label: 'Prove it vs base', onClick: () => openAdapterEvalModal(adapter) },
          { label: 'View job', onClick: () => { selectPage('training'); document.querySelector('#page-training [data-tab="queue"]')?.click(); } },
        ]);
      } else if (state === 'failed' || state === 'error') {
        announceStatus('training-queue-status', `Training failed: ${adapter}.`);
        actionToast(`Training ${adapter} failed.`, 'err', [
          { label: 'View job', onClick: () => { selectPage('training'); document.querySelector('#page-training [data-tab="queue"]')?.click(); } },
        ]);
      }
    }
  }
  prevTrainingStates = now;
}

async function pollTraining() {
  const queuePanel = setPanelBusy('tab-queue', true);
  if (!queuePanel) return;
  try {
    const data = await api('/v1/train/queue');
    trainingJobsCache = {
      running: data.running || null,
      queued: data.queued || [],
      completed: data.completed || [],
    };
    detectTrainingTransitions(data);
    watchCorrectionsJob(data);
    const r = data.running;
    const key = [
      r ? `${r.job_id}:${(r.progress || 0).toFixed(3)}:${r.current_loss != null ? r.current_loss.toFixed(4) : ''}` : '',
      (data.queued || []).map(j => j.job_id).join(','),
      // Completed jobs are NOT immutable: the §8.7 gate eval stamps
      // post_eval_verdict/gate_outcome minutes AFTER state flips to
      // 'completed', and failed jobs carry an error message. Key on their
      // presence too, or the verdict pill / error line never repaints
      // until some unrelated change touches the list.
      (data.completed || []).map(j => `${j.job_id}:${j.state}:${j.gate_outcome || (j.post_eval_verdict ? 'v' : '')}:${Array.isArray(j.post_eval_gate_evidence) ? j.post_eval_gate_evidence.length : 0}:${j.error ? 'e' : ''}`).join(','),
    ].join('|');
    if (key !== lastTrainingKey) {
      lastTrainingKey = key;
      renderTrainingQueue(data);
    }
    const liveCount = (data.running ? 1 : 0) + (data.queued ? data.queued.length : 0);
    setText('training-count', String(liveCount));
    updateFlywheel();
  } catch (e) {
    // Invalidate the queue's render key — the failure HTML replaced the list.
    lastTrainingKey = null;
    queuePanel.innerHTML = apiFailureHtml('Training queue', e, 'pollTraining');
  } finally {
    setPanelBusy('tab-queue', false);
  }
}

let trainingQueueFilter = '';
function matchTraining(j) {
  const q = trainingQueueFilter.trim().toLowerCase();
  if (!q) return true;
  return [
    j.job_id || '',
    j.adapter_name || '',
    j.effective_seed || '',
    (j.job_type || '').toString(),
    (j.state || '').toString(),
  ].join(' ').toLowerCase().includes(q);
}
function renderTrainingQueue(data) {
  const el = document.getElementById('tab-queue');
  // Snapshot filter focus/selection BEFORE rewriting innerHTML so a
  // background poll that fires while the user is typing doesn't yank
  // focus away mid-keystroke.
  let restoreFocus = false;
  let restoreSelStart = 0;
  let restoreSelEnd = 0;
  const prevFilter = document.getElementById('training-queue-filter');
  if (prevFilter && document.activeElement === prevFilter) {
    restoreFocus = true;
    restoreSelStart = prevFilter.selectionStart || 0;
    restoreSelEnd = prevFilter.selectionEnd || 0;
  }
  const totalAll = (data.running ? 1 : 0)
    + (data.queued ? data.queued.length : 0)
    + (data.completed ? data.completed.length : 0);
  // Filter input is always present (even when 0 results) so the user can
  // clear / change the filter without re-navigating.
  const filterBar = totalAll > 0
    ? `<div class="evals-toolbar" style="padding:0 0 var(--space-3) 0;">
        <input class="search-input" id="training-queue-filter" type="search" placeholder="Filter by adapter, type, state, job id…" aria-label="Filter training jobs" value="${escapeHtml(trainingQueueFilter)}">
      </div>`
    : '';

  const runningMatch = data.running && matchTraining(data.running);
  const queuedMatch = (data.queued || []).filter(matchTraining);
  const completedMatch = (data.completed || []).filter(matchTraining);

  // Always render in the same flat container (training-cards) — separate
  // queue/running/completed visually with section labels but a uniform
  // card style so eye-tracking is constant across job states.
  let html = filterBar + '<div class="training-cards">';
  if (runningMatch) {
    html += `<div class="queue-section-label">Running</div>`;
    html += renderTrainingCard(data.running, 'running');
  }
  if (queuedMatch.length > 0) {
    html += `<div class="queue-section-label">Queued</div>`;
    queuedMatch.forEach(q => {
      html += renderTrainingCard(q, 'queued');
    });
  }
  if (completedMatch.length > 0) {
    const total = completedMatch.length;
    const totalLabel = totalAll && total !== (data.completed?.length || 0)
      ? `${total} of ${data.completed?.length || 0}`
      : `${total}`;
    html += `<div class="queue-section-label">Recent <span class="hint" style="font-weight:400;">· ${totalLabel} job${total === 1 ? '' : 's'}</span></div>`;
    completedMatch.forEach(j => {
      html += renderTrainingCard(j, 'completed');
    });
  }
  html += '</div>';
  if (totalAll > 0 && !runningMatch && !queuedMatch.length && !completedMatch.length) {
    html += `<div class="eval-empty" style="margin-top:var(--space-3);"><div class="eval-empty-body">No training jobs match <code>${escapeHtml(trainingQueueFilter)}</code>.</div></div>`;
  }

  if (!data.running && (!data.queued || !data.queued.length) && (!data.completed || !data.completed.length)) {
    html = `<div class="eval-empty empty">
      <div class="eval-empty-icon"><svg class="icn"><use href="#i-flask"></use></svg></div>
      <div class="eval-empty-title">No training jobs yet.</div>
      <div class="eval-empty-body">Submit SFT examples to teach a correction, or use GRPO for scored completions. Datasets uploaded under Evals can be picked directly in the SFT/GRPO submit forms. New here? Read the <a href="https://ericflo.github.io/kiln/quickstart.html" target="_blank" rel="noopener">Quickstart</a> or the <a href="https://ericflo.github.io/kiln/grpo.html" target="_blank" rel="noopener">GRPO Guide</a>.</div>
      <button class="eval-empty-cta" type="button" onclick="document.getElementById('training-tab-sft').click();">Train your first adapter</button>
    </div>`;
  }
  el.innerHTML = html;
  // Wire card clicks for drill-in (queued/running/completed all open the modal).
  el.querySelectorAll('[data-train-job-id]').forEach(card => {
    card.addEventListener('click', ev => {
      // Don't trigger drill if user clicked an inline action button.
      if (ev.target.closest('[data-train-cancel],[data-train-prove]')) return;
      openTrainDrillModal(card.dataset.trainJobId);
    });
  });
  el.querySelectorAll('[data-train-cancel]').forEach(b => {
    b.addEventListener('click', ev => {
      ev.stopPropagation();
      cancelJobFromButton(b);
    });
  });
  // Persistent "Prove it vs base" on completed cards — same modal the
  // completion toast offers, so a missed toast is never a dead-end.
  el.querySelectorAll('[data-train-prove]').forEach(b => {
    b.addEventListener('click', ev => {
      ev.stopPropagation();
      openAdapterEvalModal(b.dataset.adapter);
    });
  });
  if (restoreFocus) {
    const f = document.getElementById('training-queue-filter');
    if (f) {
      f.focus();
      try { f.setSelectionRange(restoreSelStart, restoreSelEnd); } catch {}
    }
  }
}

/// Render a training job as a rich card with progress bar, loss curve
/// (when history is available), and per-job stats. State drives layout:
/// `queued` shows position; `running` shows live progress + curve;
/// `completed` shows final loss + duration.
function renderTrainingCard(j, state) {
  const pct = ((j.progress || 0) * 100).toFixed(0);
  const adapterLabel = j.adapter_name ? escapeHtml(j.adapter_name) : `<span class="hint">(unnamed)</span>`;
  const jobType = (j.job_type || '').toString().toLowerCase();
  // Loss curve (only present when we've run job_detail at least once
  // for this job — populated by openTrainDrillModal). We just leave a
  // placeholder for now.
  const isRunning = state === 'running';
  // State class drives the amber rule: only a RUNNING job's bar is hot (amber);
  // completed → green, failed → red, queued → neutral.
  const stateClass = isRunning ? 'training-card-running' : 'training-card-' + (j.state || state || 'done').toString().toLowerCase();
  const cardClass = 'training-card ' + stateClass;
  let stateBadge;
  if (state === 'queued') {
    stateBadge = `<span class="job-state-pill queued">queued${j.position ? ' · #'+j.position : ''}</span>`;
  } else if (state === 'running') {
    stateBadge = `<span class="job-state-pill running">running</span>`;
  } else {
    const stateNorm = (j.state || '').toString().toLowerCase();
    stateBadge = `<span class="job-state-pill ${stateNorm}">${escapeHtml(stateNorm || 'completed')}</span>`;
  }
  const stateNormForActions = (j.state || state || '').toString().toLowerCase();
  // Completed cards carry the next action PERSISTENTLY — the completion toast
  // is a courtesy, not the only door. Failed cards surface the reason inline.
  let actionBtn = '';
  if (state === 'queued') {
    actionBtn = `<button class="btn btn-sm" data-train-cancel data-job-id="${escapeHtml(j.job_id)}" type="button" style="margin-left:auto;">Cancel</button>`;
  } else if (state === 'running') {
    // Running jobs are stoppable too: the server sets a cooperative flag
    // and the trainer aborts at the next step boundary.
    actionBtn = `<button class="btn btn-sm" data-train-cancel data-job-id="${escapeHtml(j.job_id)}" type="button" style="margin-left:auto;" title="Stop at the next training step">Stop</button>`;
  } else if (stateNormForActions === 'completed' && j.adapter_name) {
    actionBtn = `<button class="btn btn-sm" data-train-prove data-adapter="${escapeHtml(j.adapter_name)}" type="button" style="margin-left:auto;" title="Grade ${escapeHtml(j.adapter_name)} against base on an eval suite">Prove it vs base</button>`;
  }
  // §8.7 promotion-gate verdict pill. Color keys off the server's
  // machine-readable `gate_outcome` (stamped next to the prose verdict):
  //   promoted          → green (gate passed, adapter serving)
  //   kept              → amber chip with a CHECK icon — a pass without a
  //                       requested promotion is a success, not a warning
  //   regression/demoted → red (rejected vs baseline / demoted to .failed)
  //   inconclusive/error → amber + warning icon (insufficient evidence or
  //                        gate couldn't measure)
  // Pill text stays the prose verdict. Rendered whenever the backend
  // stamped a verdict so a silent demotion can't hide.
  let gateLine = '';
  if (j.post_eval_verdict || j.gate_outcome) {
    const v = String(j.post_eval_verdict || j.gate_outcome);
    const OUTCOME_CLS = { promoted: 'ok', kept: 'warn', regression: 'err', demoted: 'err', inconclusive: 'warn', error: 'warn' };
    let cls = OUTCOME_CLS[j.gate_outcome] || '';
    if (!cls) {
      // Fallback ONLY for jobs archived before `gate_outcome` existed
      // (and for older servers): those carry prose alone, so classify by
      // substring as the UI historically did. Known-imperfect — that
      // heuristic is exactly why gate_outcome was added.
      cls = (v.includes('promoted') && !v.includes('NOT')) ? 'ok'
        : (v.includes('.failed') || v.includes('demoted') || v.includes('REGRESSION')) ? 'err' : 'warn';
    }
    const iconName = (cls === 'ok' || j.gate_outcome === 'kept') ? 'check' : 'warning';
    const evidenceRows = Array.isArray(j.post_eval_gate_evidence) ? j.post_eval_gate_evidence : [];
    const evidence = evidenceRows.length ? evidenceRows[evidenceRows.length - 1] : null;
    const evidenceSummary = evidence
      ? `${evidenceRows.length > 1 ? ` · gates=${evidenceRows.length}` : ''} · n=${Number(evidence.paired_examples || 0).toLocaleString()} · p=${Number(evidence.exact_sign_test_p_value).toPrecision(3)} · LB=${Number(evidence.candidate_accuracy_lower_bound).toFixed(3)}`
      : '';
    const evidenceTitle = evidence
      ? `${v}\n${evidenceRows.map((row) => `Policy ${row.policy_version}; suite ${row.suite_name} (${row.suite_hash}); outcome ${row.outcome}; improved ${row.improved}, regressed ${row.regressed}, tied ${row.tied}; 95% CI [${Number(row.candidate_accuracy_lower_bound).toFixed(3)}, ${Number(row.candidate_accuracy_upper_bound).toFixed(3)}]`).join('\n')}`
      : v;
    gateLine = `<div class="training-card-gate gate-${cls}" title="${escapeHtml(evidenceTitle)}">${icon(iconName, 'icn-sm')} ${escapeHtml(v.slice(0, 140))}${escapeHtml(evidenceSummary)}</div>`;
  }
  const errLine = (stateNormForActions === 'failed' && j.error)
    ? `<div class="training-card-error">${icon('warning', 'icn-sm')} ${escapeHtml(String(j.error).slice(0, 220))}</div>`
    : '';
  const cancelBtn = actionBtn;
  const admitted = j.training_data || null;
  const admittedOpenEnv = admitted?.openenv || null;
  const admittedOpenEnvNames = Array.isArray(admittedOpenEnv?.environments)
    ? admittedOpenEnv.environments.map((environment) => environment.environment_name).join(', ')
    : '';
  const admittedDataLine = admitted
    ? `<div class="training-card-data" title="Exact admitted training corpus ${escapeHtml(admitted.admitted_corpus_sha256 || '')}${admittedOpenEnv?.group_plan_sha256 ? `; OpenEnv plan ${escapeHtml(admittedOpenEnv.group_plan_sha256)}` : ''}">${icon('stack', 'icn-sm')} ${admittedOpenEnv ? `OpenEnv · ${escapeHtml(admittedOpenEnvNames || 'compatible environment')} · ${Number(admittedOpenEnv.groups || 0).toLocaleString()} group${Number(admittedOpenEnv.groups || 0) === 1 ? '' : 's'} · ${Number(admittedOpenEnv.rollouts || 0).toLocaleString()} rollout${Number(admittedOpenEnv.rollouts || 0) === 1 ? '' : 's'}` : `${escapeHtml(admitted.dataset || admitted.source || 'training data')}${admitted.split ? ` · ${escapeHtml(admitted.split)}` : ''} · ${Number(admitted.rows || 0).toLocaleString()} row${Number(admitted.rows || 0) === 1 ? '' : 's'}`}</div>`
    : '';
  // Prefer the wall-clock timestamps (`submitted_unix_ms` /
  // `finished_unix_ms`) introduced with the on-disk archive — those
  // survive restarts. Fall back to `elapsed_secs` only when the server
  // is on an older payload (no wall-clock fields).
  let timeBadge = '';
  if (state === 'completed' && j.finished_unix_ms) {
    timeBadge = `<span class="hint" style="font-size:11px;" title="${escapeHtml(new Date(j.finished_unix_ms).toISOString())}">finished ${escapeHtml(fmtSmartTime(j.finished_unix_ms))}</span>`;
  } else if (j.submitted_unix_ms) {
    timeBadge = `<span class="hint" style="font-size:11px;" title="${escapeHtml(new Date(j.submitted_unix_ms).toISOString())}">started ${escapeHtml(fmtSmartTime(j.submitted_unix_ms))}</span>`;
  } else if (j.elapsed_secs != null) {
    timeBadge = `<span class="hint" style="font-size:11px;">${escapeHtml(Math.floor(j.elapsed_secs) + 's elapsed')}</span>`;
  }
  return `<div class="${cardClass}" data-train-job-id="${escapeHtml(j.job_id)}">
    <div class="training-card-head">
      ${stateBadge}
      <span class="training-card-name"><span class="hint" style="font-weight:400;margin-right:4px;">Adapter:</span>${adapterLabel}</span>
      <span class="training-card-type ${escapeHtml(jobType)}">${escapeHtml(jobType)}</span>
      <span class="hint" style="font-family:var(--font-mono); font-size:11px;">${escapeHtml(shortId(j.job_id))}</span>
      ${j.effective_seed == null ? '' : `<span class="hint tabular-nums" style="font-family:var(--font-mono); font-size:11px;" title="Immutable effective training seed">seed ${escapeHtml(String(j.effective_seed))}</span>`}
      ${timeBadge}
      ${cancelBtn}
    </div>
    ${admittedDataLine}
    <div class="training-card-progress">
      <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${pct}%;"></div></div>
      <div class="training-stat"><span class="training-stat-num">${pct}%</span><span class="training-stat-label">progress</span></div>
      <div class="training-stat">
        <span class="training-stat-num">${j.current_loss != null ? j.current_loss.toFixed(3) : '—'}</span>
        <span class="training-stat-label">${j.current_loss != null ? 'loss' : 'not started'}</span>
      </div>
    </div>
    ${gateLine}${errLine}
    <div class="training-card-curve" id="training-curve-${escapeHtml(j.job_id)}"></div>
  </div>`;
}

// Cancel buttons receive the click via stopPropagation; this is a thin
// indirection that resolves the job_id and forwards to the existing
// cancellation flow.
function cancelJobFromButton(btn) {
  const jobId = btn.dataset.jobId || '';
  if (!jobId || cancellingTrainingJobIds.has(jobId)) return;
  btn.disabled = true;
  btn.textContent = 'Cancelling…';
  cancelJob(jobId, btn);
}

window.cancelJob = async function(jobId, button) {
  if (!jobId || cancellingTrainingJobIds.has(jobId)) return;
  cancellingTrainingJobIds.add(jobId);
  try {
    await api('/v1/train/queue/' + jobId, { method: 'DELETE' });
    toast('Cancelled job ' + jobId.slice(0, 8));
    cancellingTrainingJobIds.delete(jobId);
    pollTraining();
  } catch (e) {
    cancellingTrainingJobIds.delete(jobId);
    if (button) {
      button.disabled = false;
      button.textContent = 'Cancel';
    }
    toast(e.message, 'err');
  }
};

function fillSftSamplePayload() {
  const sample = [
    {
      messages: [
        { role: 'user', content: 'Translate to French: Hello' },
        { role: 'assistant', content: 'Bonjour' },
      ],
    },
  ];
  const textarea = document.getElementById('sft-examples');
  const samplePayload = JSON.stringify(sample, null, 2);
  if (textarea.value.trim() && textarea.value !== samplePayload && !confirm('Replace the current SFT examples with the sample payload?')) {
    return;
  }
  textarea.value = samplePayload;
  const pasteRow = document.getElementById('sft-paste-row'); if (pasteRow) pasteRow.hidden = false;
  clearTrainingData('sft');
  updateSftSubmitState();
  textarea.focus();
  toast('Sample SFT payload inserted — edit it or Train as-is');
}

function fillGrpoSamplePayload() {
  const sample = [
    {
      messages: [
        { role: 'user', content: 'Write a haiku about the moon' },
      ],
      completions: [
        {
          text: 'Silent moonlit night / Silver clouds drift softly by / Dreams bloom in starlight',
          reward: 0.9,
        },
        { text: 'The moon is bright tonight.', reward: 0.2 },
      ],
    },
  ];
  const textarea = document.getElementById('grpo-groups');
  const samplePayload = JSON.stringify(sample, null, 2);
  if (textarea.value.trim() && textarea.value !== samplePayload && !confirm('Replace the current GRPO groups with the sample payload?')) {
    return;
  }
  textarea.value = samplePayload;
  const pasteRow = document.getElementById('grpo-paste-row'); if (pasteRow) pasteRow.hidden = false;
  clearTrainingData('grpo');
  updateGrpoSubmitState();
  textarea.focus();
  toast('Sample GRPO payload inserted — edit it or Train as-is');
}

document.getElementById('use-sft-sample').addEventListener('click', fillSftSamplePayload);
document.getElementById('use-grpo-sample').addEventListener('click', fillGrpoSamplePayload);

/* ====== Direct "drop a file and train" data input (SFT + GRPO) ==============
   /v1/train/{sft,grpo} take inline examples/groups, so the primary path is:
   drop a .jsonl/.json file -> parse + validate + preview in place -> train.
   No Evals detour, no dropdown round-trip, no megabytes pasted into a textarea.
   trainingData[kind] holds parsed items from the file/dataset path; the textarea
   is the secondary "paste" path. Exactly one source is active at a time. */
const trainingData = { sft: null, grpo: null };
const TRAIN_KIND = {
  sft:  { noun: 'example', datasetFormat: 'sft_chat',  pickId: 'sft-dataset-pick',  textareaId: 'sft-examples', update: () => updateSftSubmitState(),  valid: (it) => sftItemValid(it) },
  grpo: { noun: 'group',   datasetFormat: 'grpo_groups', pickId: 'grpo-dataset-pick', textareaId: 'grpo-groups',  update: () => updateGrpoSubmitState(), valid: (it) => grpoItemValid(it) },
};

function parseTrainingText(text) {
  const t = (text || '').trim();
  if (!t) return [];
  if (t[0] === '[') {
    const arr = JSON.parse(t);
    if (!Array.isArray(arr)) throw new Error('Top-level JSON must be an array.');
    return arr;
  }
  const items = [];
  t.split('\n').forEach((line, i) => {
    const s = line.trim(); if (!s) return;
    try { items.push(JSON.parse(s)); }
    catch { throw new Error(`Line ${i + 1} isn't valid JSON. Use JSONL (one object per line) or a JSON array.`); }
  });
  return items;
}
function sftItemValid(it) {
  if (!it || !Array.isArray(it.messages) || !it.messages.length) return false;
  const roles = it.messages.map(m => m && m.role);
  return roles.includes('user') && roles.includes('assistant');
}
function grpoItemValid(it) {
  if (!it || !Array.isArray(it.messages) || !it.messages.length) return false;
  if (!Array.isArray(it.completions) || !it.completions.length) return false;
  return it.completions.every(c => c && typeof c.text === 'string' && c.text.trim() && typeof c.reward === 'number' && Number.isFinite(c.reward));
}
function suggestAdapterName(filename, kind) {
  let base = (filename || '').replace(/\.[^.]+$/, '').toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40);
  if (!base) base = kind + '-data';
  return base.endsWith('-' + kind) ? base : base + '-' + kind;
}
function trainingDatasetNameForFile(filename, kind) {
  const base = suggestAdapterName(filename, kind)
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || (kind + '-data');
  return `${base}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
// Auto-fill the adapter name from the data source UNLESS the user typed their
// own. Tracks a dirty flag instead of matching the literal default, so loading
// a second file/dataset re-suggests (no silent name collisions) while a
// hand-chosen name is never clobbered.
function maybeSuggestAdapterName(kind, sourceLabel) {
  const input = document.getElementById(kind + '-output-name');
  if (!input) return;
  if (input.dataset.userEdited === '1' && input.value.trim()) return;
  input.value = suggestAdapterName(sourceLabel, kind);
  TRAIN_KIND[kind].update();
}
['sft', 'grpo'].forEach(kind => {
  document.getElementById(kind + '-output-name')?.addEventListener('input', e => {
    e.target.dataset.userEdited = e.target.value.trim() ? '1' : '';
  });
});
function renderTrainingDataStatus(kind, total, valid, label, badIdx) {
  const el = document.getElementById(kind + '-data-status');
  const K = TRAIN_KIND[kind];
  if (!el) return;
  const why = kind === 'sft' ? 'needs a user message and an assistant reply' : 'needs messages plus scored completions';
  if (!valid) {
    el.hidden = false; el.className = 'train-data-status is-bad';
    el.innerHTML = `${icon('warning', 'icn-sm')} No usable ${K.noun}s in ${escapeHtml(label || 'that input')}. Each ${why}.`;
    return;
  }
  const skipped = total - valid;
  // Name the offending entries so nobody has to open the file and hunt.
  let skipDetail = '';
  if (skipped > 0 && Array.isArray(badIdx) && badIdx.length) {
    const shown = badIdx.slice(0, 3).map(i => '#' + (i + 1)).join(', ');
    skipDetail = ` · skipped ${escapeHtml(shown)}${badIdx.length > 3 ? ` +${badIdx.length - 3} more` : ''} (each ${why})`;
  } else if (skipped > 0) {
    skipDetail = ` · ${skipped} skipped (invalid)`;
  }
  el.hidden = false; el.className = 'train-data-status is-good';
  el.innerHTML = `${icon('check', 'icn-sm')} <strong>${valid}</strong> ${K.noun}${valid === 1 ? '' : 's'} ready`
    + (skipDetail ? `<span class="train-data-skip">${skipDetail}</span>` : '')
    + ` <span class="train-data-src">from ${escapeHtml(label || 'input')}</span>`;
}
// Set the file/dataset source for a kind. Clears the textarea so there's one
// source of truth, validates, previews, and re-checks submit readiness.
function setTrainingData(kind, items, label) {
  const K = TRAIN_KIND[kind];
  const valid = [], badIdx = [];
  (items || []).forEach((it, i) => { if (K.valid(it)) valid.push(it); else badIdx.push(i); });
  trainingData[kind] = valid.length ? { items: valid, total: (items || []).length, label } : null;
  const ta = document.getElementById(K.textareaId);
  if (ta) ta.value = '';                       // file/dataset is now the source
  renderTrainingDataStatus(kind, (items || []).length, valid.length, label, badIdx);
  K.update();
  return valid.length;
}
function clearTrainingData(kind) {
  trainingData[kind] = null;
  const el = document.getElementById(kind + '-data-status'); if (el) el.hidden = true;
}
async function loadTrainingFile(kind, file) {
  if (!file) return;
  const K = TRAIN_KIND[kind];
  const previous = trainingData[kind];
  try {
    const datasetName = trainingDatasetNameForFile(file.name, kind);
    trainingData[kind] = null;
    const el = document.getElementById(kind + '-data-status');
    if (el) {
      el.hidden = false; el.className = 'train-data-status';
      el.innerHTML = `${icon('upload', 'icn-sm')} Uploading ${escapeHtml(file.name)} into the local dataset store…`;
    }
    K.update();
    const manifest = await postDatasetUpload(
      datasetName,
      K.datasetFormat,
      `Uploaded from ${file.name} for ${kind.toUpperCase()} training`,
      file,
    );
    const n = await loadNamedDatasetIntoTraining(kind, manifest.name || datasetName);
    if (n) maybeSuggestAdapterName(kind, file.name);
    if (typeof refreshDatasets === 'function') refreshDatasets();
    refreshDatasetPicker(kind);
    toast(`Uploaded "${manifest.name || datasetName}" (${Number(manifest.num_rows || 0).toLocaleString()} rows)`, 'ok');
  } catch (e) {
    // A bad drop must never destroy data you already loaded — keep it and say so.
    trainingData[kind] = previous;
    const el = document.getElementById(kind + '-data-status');
    if (el) {
      el.hidden = false; el.className = 'train-data-status is-bad';
      el.innerHTML = `${icon('warning', 'icn-sm')} ${escapeHtml(e.message)}`
        + (previous ? ` <span class="train-data-src">— kept your previous data (${escapeHtml(previous.label || 'loaded input')})</span>` : '');
    }
    K.update();
  }
}
function wireDropzone(kind) {
  const zone = document.getElementById(kind + '-dropzone');
  const file = document.getElementById(kind + '-file');
  if (!zone || !file) return;
  zone.addEventListener('click', () => file.click());
  zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); } });
  file.addEventListener('change', () => { loadTrainingFile(kind, file.files[0]); file.value = ''; });
  ['dragenter', 'dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('is-drag'); }));
  ['dragleave', 'dragend'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('is-drag'); }));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('is-drag'); const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]; if (f) loadTrainingFile(kind, f); });
}
// Secondary affordance toggles (pick a dataset / paste JSON) reveal their row.
function wireTrainingAlts(kind) {
  const pickRow = document.getElementById(kind + '-pick-row');
  const pasteRow = document.getElementById(kind + '-paste-row');
  document.getElementById(kind + '-pick-toggle')?.addEventListener('click', () => { if (pickRow) { pickRow.hidden = !pickRow.hidden; if (!pickRow.hidden) refreshDatasetPicker(kind); } });
  document.getElementById(kind + '-paste-toggle')?.addEventListener('click', () => { if (pasteRow) { pasteRow.hidden = !pasteRow.hidden; if (!pasteRow.hidden) document.getElementById(TRAIN_KIND[kind].textareaId)?.focus(); } });
}
wireDropzone('sft'); wireDropzone('grpo');
wireTrainingAlts('sft'); wireTrainingAlts('grpo');

// Advanced hyperparameters live behind a toggle; a one-line summary narrates the
// current values so collapsing never hides information.
function wireAdvanced(kind, summarize) {
  const btn = document.getElementById(kind + '-adv-toggle');
  const body = document.getElementById(kind + '-advanced');
  const summary = document.getElementById(kind + '-adv-summary');
  if (!btn || !body) return;
  btn.addEventListener('click', () => {
    const open = body.hidden;
    body.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
  });
  const update = () => { if (summary) summary.textContent = summarize(); };
  body.querySelectorAll('input, select').forEach(i => {
    i.addEventListener('input', update);
    i.addEventListener('change', update);
  });
  update();
}
// A blank learning-rate field means "auto" — the server resolves the
// per-optimizer default (Muon vs AdamW want very different bands).
const lrSummary = id => (document.getElementById(id)?.value || '').trim() || 'auto';
const checkpointSummary = kind => {
  const interval = (document.getElementById(kind + '-checkpoint-interval')?.value || '').trim();
  const resume = (document.getElementById(kind + '-resume-checkpoint')?.value || '').trim();
  return `${interval ? `checkpoint every ${interval}` : 'checkpoints off'} · ${resume ? 'resume selected' : 'fresh run'}`;
};
const optimizerLabel = id => {
  const value = document.getElementById(id)?.value;
  return optimizerLabelForKind(value);
};
function optimizerLabelForKind(kind) {
  if (kind === 'adam_w') return 'AdamW';
  if (kind === 'sgd') return 'SGD';
  if (kind === 'muon') return 'Muon';
  return String(kind || 'unknown');
}

function trainingWorkloadLabel(workload) {
  if (workload === 'distill_refresh') return 'Distill refresh';
  return String(workload || 'unknown').toUpperCase();
}

let trainingOptimizerSupportSnapshot = null;
let trainingOptimizerSupportUnavailableReason = 'Optimizer capability details are still loading';

function optimizerSupportEntry(kind) {
  const entries = trainingOptimizerSupportSnapshot?.optimizers;
  return Array.isArray(entries) ? entries.find(entry => entry?.kind === kind) || null : null;
}

function optimizerWorkloadEntry(workload) {
  const workloads = trainingOptimizerSupportSnapshot?.workloads;
  return Array.isArray(workloads)
    ? workloads.find(entry => entry?.workload === workload) || null
    : null;
}

function rememberGeneralRankBounds(input) {
  if (!input) return;
  if (!input.dataset.optimizerGeneralMin) input.dataset.optimizerGeneralMin = input.min || '1';
  if (!input.dataset.optimizerGeneralMax) input.dataset.optimizerGeneralMax = input.max || '';
}

function applyOptimizerRankBounds(input, entry) {
  if (!input) return;
  rememberGeneralRankBounds(input);
  const generalMin = Number.parseInt(input.dataset.optimizerGeneralMin, 10) || 1;
  const generalMax = Number.parseInt(input.dataset.optimizerGeneralMax, 10);
  const rank = entry?.optimizer_tuple?.lora_rank;
  const minimum = Number.isInteger(rank?.minimum) ? rank.minimum : generalMin;
  const maximum = rank && Object.prototype.hasOwnProperty.call(rank, 'maximum')
    ? rank.maximum
    : generalMax;
  input.min = String(minimum);
  if (Number.isInteger(maximum)) input.max = String(maximum);
  else input.removeAttribute('max');
}

function optimizerRankRangeLabel(rank) {
  if (!Number.isInteger(rank?.minimum)) return 'unavailable';
  return Number.isInteger(rank.maximum)
    ? `${rank.minimum}–${rank.maximum}`
    : `${rank.minimum}+`;
}

function trainingOptimizerKindState(workload, kind) {
  const support = trainingOptimizerSupportSnapshot;
  if (!support) return { ready: false, reason: trainingOptimizerSupportUnavailableReason, entry: null };
  const workloadEntry = optimizerWorkloadEntry(workload);
  const workloadLabel = trainingWorkloadLabel(workload);
  if (!workloadEntry) {
    return { ready: false, reason: `The optimizer capability contract does not describe the ${workloadLabel} workload`, entry: null };
  }
  if (workloadEntry.supported !== true) {
    return {
      ready: false,
      reason: `${workloadLabel} is unavailable: ${workloadEntry.unavailable_reason || 'the resident server path is unsupported'}`,
      entry: null,
    };
  }
  const entry = optimizerSupportEntry(kind);
  if (!Array.isArray(workloadEntry.allowed_optimizer_kinds)) {
    return { ready: false, reason: `${workloadLabel} is missing its optimizer allowlist`, entry };
  }
  if (!workloadEntry.allowed_optimizer_kinds.includes(kind)) {
    return { ready: false, reason: `${optimizerLabelForKind(kind)} is not allowed for ${workloadLabel}`, entry };
  }
  if (!support.optimizer_tuple_kinds.includes(kind)) {
    return { ready: false, reason: `${optimizerLabelForKind(kind)} is absent from the admitted optimizer tuples`, entry };
  }
  if (!entry) {
    return { ready: false, reason: `${optimizerLabelForKind(kind)} is absent from the optimizer capability contract`, entry: null };
  }
  const tuple = entry.optimizer_tuple;
  if (tuple?.supported !== true) {
    return {
      ready: false,
      reason: `${optimizerLabelForKind(kind)} optimizer tuple is unavailable: ${tuple?.unavailable_reason || 'the resident weights do not admit it'}`,
      entry,
    };
  }
  return { ready: true, reason: null, entry };
}

function trainingOptimizerAdmissionState(workload, kind, rawRank) {
  const kindState = trainingOptimizerKindState(workload, kind);
  if (!kindState.ready) return kindState;
  const rank = typeof rawRank === 'number' ? rawRank : Number(rawRank);
  if (!Number.isSafeInteger(rank) || rank < 1) {
    return { ...kindState, ready: false, reason: `${optimizerLabelForKind(kind)} LoRA rank must be a positive whole number` };
  }
  const rankSupport = kindState.entry?.optimizer_tuple?.lora_rank;
  if (!Number.isInteger(rankSupport?.minimum)) {
    return { ...kindState, ready: false, reason: `${optimizerLabelForKind(kind)} is missing its LoRA rank contract` };
  }
  if (rank < rankSupport.minimum || (Number.isInteger(rankSupport.maximum) && rank > rankSupport.maximum)) {
    return {
      ...kindState,
      ready: false,
      reason: `${optimizerLabelForKind(kind)} LoRA rank ${rank} is outside supported ranks ${optimizerRankRangeLabel(rankSupport)}`,
    };
  }
  return { ...kindState, rank };
}

function optimizerSupportStatusFromState(state, kind) {
  if (!state.ready) return `${state.reason}. Training remains disabled.`;
  const rank = state.entry.optimizer_tuple.lora_rank;
  const liveAdmission = rank.live_memory_admission_required === true
    ? '; live memory admission can still reject a request that does not fit'
    : '';
  return `${optimizerLabelForKind(kind)} · ${trainingOptimizerSupportSnapshot.resolved_lora_parameter_dtype || 'resolved'} LoRA · round-to-nearest · rank ${state.rank} (supported ${optimizerRankRangeLabel(rank)})${liveAdmission}.`;
}

function optimizerSupportStatus(workload, kind, rawRank) {
  return optimizerSupportStatusFromState(
    trainingOptimizerAdmissionState(workload, kind, rawRank),
    kind,
  );
}

function requireTrainingOptimizerAdmission(workload, kind, rank, modeLabel) {
  const state = trainingOptimizerAdmissionState(workload, kind, rank);
  if (!state.ready) throw new Error(`${modeLabel} cannot submit: ${state.reason}.`);
  return state;
}

function applyTrainingOptimizerForm(kind) {
  const select = document.getElementById(kind + '-optimizer');
  const rankInput = document.getElementById(kind + '-rank');
  const status = document.getElementById(kind + '-optimizer-support');
  if (!select) return;
  rememberGeneralRankBounds(rankInput);
  const support = trainingOptimizerSupportSnapshot;
  for (const option of select.options) {
    const optionState = trainingOptimizerKindState(kind, option.value);
    option.disabled = !optionState.ready;
    option.title = option.disabled
      ? optionState.reason || 'Unsupported by the complete server training path'
      : '';
  }
  const entry = optimizerSupportEntry(select.value);
  const kindState = trainingOptimizerKindState(kind, select.value);
  select.disabled = !support || !Array.from(select.options).some(option => !option.disabled);
  if (rankInput) rankInput.disabled = !kindState.ready;
  applyOptimizerRankBounds(rankInput, entry);
  if (status) status.textContent = optimizerSupportStatus(kind, select.value, rankInput?.value);
  if (kind === 'sft') updateSftSubmitState();
  if (kind === 'grpo') updateGrpoSubmitState();
}

function applyOpdOptimizerSupport() {
  const rankInput = document.getElementById('opd-rank');
  const status = document.getElementById('opd-optimizer-support');
  const entry = optimizerSupportEntry('muon');
  applyOptimizerRankBounds(rankInput, entry);
  const kindState = trainingOptimizerKindState('opd', 'muon');
  if (rankInput) rankInput.disabled = !kindState.ready;
  if (status) status.textContent = `OPD uses Muon. ${optimizerSupportStatus('opd', 'muon', rankInput?.value)}`;
  updateOpdSubmitState();
}

function applyFixedTrainingSurface(formId, statusId, rank, workload = 'opd') {
  const form = document.getElementById(formId);
  const status = document.getElementById(statusId);
  const state = trainingOptimizerAdmissionState(workload, 'muon', rank);
  const submit = form?.querySelector('button[type="submit"]');
  if (submit) {
    submit.disabled = !state.ready;
    submit.title = state.ready ? '' : state.reason || 'Training capability unavailable';
  }
  if (status) status.textContent = optimizerSupportStatus(workload, 'muon', rank);
}

function updateFixedTrainingSurfaceStates() {
  const pumpRank = document.getElementById('pump-rank');
  const pumpEntry = optimizerSupportEntry('muon');
  applyOptimizerRankBounds(pumpRank, pumpEntry);
  const pumpKindState = trainingOptimizerKindState('opd', 'muon');
  if (pumpRank) pumpRank.disabled = !pumpKindState.ready;
  applyFixedTrainingSurface('distill-pump-form', 'pump-optimizer-support', pumpRank?.value);
  applyFixedTrainingSurface('distill-refresh-form', 'refresh-optimizer-support', 16, 'distill_refresh');
  applyFixedTrainingSurface('distill-merge-form', 'merge-optimizer-support', 16);
  applyFixedTrainingSurface('distill-self-form', 'self-optimizer-support', 16);
}

function applyTrainingOptimizerSupportState() {
  applyTrainingOptimizerForm('sft');
  applyTrainingOptimizerForm('grpo');
  applyOpdOptimizerSupport();
  updateFixedTrainingSurfaceStates();
  syncOpenEnvKind();
  updateCorrFoot();
  applyRecipeAdmissionButtons();
}

function validOptimizerRankContract(rank) {
  const positiveSafeInteger = value => Number.isSafeInteger(value) && value > 0;
  if (!positiveSafeInteger(rank?.minimum)
    || !positiveSafeInteger(rank?.maximum)
    || !positiveSafeInteger(rank?.model_maximum)
    || rank.live_memory_admission_required !== true
    || rank.minimum > rank.maximum
    || rank.maximum > rank.model_maximum) {
    return false;
  }
  if (rank.backend_maximum !== null && !positiveSafeInteger(rank.backend_maximum)) return false;
  const effectiveMaximum = Math.min(
    rank.model_maximum,
    rank.backend_maximum ?? rank.model_maximum,
  );
  return rank.maximum === effectiveMaximum;
}

function updateTrainingOptimizerSupport(cfg) {
  const candidate = cfg?.training?.optimizer_support;
  const nonEmptyIdentity = value => typeof value === 'string' && value.trim().length > 0;
  const supportedSchema = candidate?.schema?.id === 'kiln.training-optimizer-support'
    && candidate?.schema?.version === 1
    && candidate.immutable_after_startup === true
    && Array.isArray(candidate.rounding_modes)
    && candidate.rounding_modes.length === 1
    && candidate.rounding_modes[0] === 'round_to_nearest'
    && nonEmptyIdentity(candidate.backend)
    && nonEmptyIdentity(candidate.device)
    && nonEmptyIdentity(candidate.base_weight_dtype)
    && nonEmptyIdentity(candidate.resolved_lora_parameter_dtype)
    && Array.isArray(candidate.optimizer_tuple_kinds)
    && Array.isArray(candidate.workloads)
    && Array.isArray(candidate.optimizers)
    && candidate.optimizers.length > 0
    && candidate.optimizers.every(entry => validOptimizerRankContract(entry?.optimizer_tuple?.lora_rank));
  trainingOptimizerSupportSnapshot = supportedSchema ? candidate : null;
  trainingOptimizerSupportUnavailableReason = supportedSchema
    ? null
    : candidate
      ? 'The server returned an unsupported optimizer capability contract'
      : cfg?.training?.native_training_unavailable_reason
        || 'Optimizer capability details are unavailable';
  applyTrainingOptimizerSupportState();
}

function markTrainingOptimizerSupportFetchFailed(error) {
  trainingOptimizerSupportSnapshot = null;
  trainingOptimizerSupportUnavailableReason = `Optimizer capability lookup failed: ${error?.message || 'request failed'}`;
  applyTrainingOptimizerSupportState();
}

for (const kind of ['sft', 'grpo']) {
  document.getElementById(kind + '-optimizer')?.addEventListener('change', () => {
    applyTrainingOptimizerForm(kind);
  });
  document.getElementById(kind + '-rank')?.addEventListener('input', () => {
    applyTrainingOptimizerForm(kind);
  });
}
document.getElementById('opd-rank')?.addEventListener('input', applyOpdOptimizerSupport);
document.getElementById('pump-rank')?.addEventListener('input', updateFixedTrainingSurfaceStates);
rememberGeneralRankBounds(document.getElementById('opd-rank'));
rememberGeneralRankBounds(document.getElementById('pump-rank'));

function readTrainingOptimizer(kind) {
  const value = document.getElementById(kind + '-optimizer')?.value;
  if (!['muon', 'adam_w', 'sgd'].includes(value)) {
    throw new Error(`${kind.toUpperCase()} optimizer selection is missing or invalid.`);
  }
  return { kind: value };
}
wireAdvanced('sft', () => {
  const v = id => document.getElementById(id)?.value || '?';
  const raw = id => document.getElementById(id)?.value || '';
  const lr = lrSummary('sft-learning-rate');
  const opt = optimizerLabel('sft-optimizer');
  const invalidRows = v('sft-invalid-row-policy') === 'skip' ? 'skip invalid rows' : 'invalid rows fail';
  const isDefault = v('sft-epochs') === '3' && opt === 'Muon' && lr === 'auto' && v('sft-rank') === '8'
    && v('sft-invalid-row-policy') === 'fail'
    && !raw('sft-checkpoint-interval') && !raw('sft-resume-checkpoint');
  if (typeof updateSftOverfitHint === 'function') updateSftOverfitHint();
  return `${v('sft-epochs')} epochs · ${opt} · learning rate ${lr} · LoRA rank ${v('sft-rank')} · ${invalidRows} · ${checkpointSummary('sft')}`
    + (isDefault ? ' — sensible defaults, no tuning needed' : ' — customized');
});
wireAdvanced('grpo', () => {
  const v = id => document.getElementById(id)?.value || '?';
  const raw = id => document.getElementById(id)?.value || '';
  const lr = lrSummary('grpo-learning-rate');
  const opt = optimizerLabel('grpo-optimizer');
  const isDefault = v('grpo-kl-coeff') === '0.1' && opt === 'Muon' && lr === 'auto' && v('grpo-rank') === '8'
    && !raw('grpo-checkpoint-interval') && !raw('grpo-resume-checkpoint');
  return `KL ${v('grpo-kl-coeff')} · ${opt} · learning rate ${lr} · LoRA rank ${v('grpo-rank')} · ${checkpointSummary('grpo')}`
    + (isDefault ? ' — sensible defaults, no tuning needed' : ' — customized');
});

// "Prove it after training" — wires the server's post_eval auto-hook: when
// checked, the train request carries an explicit suite and data scope
// and Kiln grades the fresh adapter AND base the moment training completes.
// The row only appears when eval suites actually exist (no dead control).
function updateProveControls(kind) {
  const check = document.getElementById(kind + '-prove');
  const suite = document.getElementById(kind + '-prove-suite');
  const scope = document.getElementById(kind + '-prove-scope');
  const hint = document.getElementById(kind + '-prove-hint');
  const enabled = Boolean(check?.checked);
  if (suite) suite.disabled = !enabled;
  if (scope) scope.disabled = !enabled;
  if (hint) {
    hint.textContent = scope?.value === 'train-set-eval'
      ? 'Diagnostic only: this mode may reuse training rows and cannot satisfy a minimum-accuracy promotion gate.'
      : kind === 'openenv'
        ? 'Kiln preflights the installed suite before opening an environment session, then grades the trained adapter and base after native GRPO.'
        : 'Kiln rejects the submission if this suite overlaps the admitted training partition, then grades the adapter and base when training finishes.';
  }
}
async function refreshProveRows() {
  let suites = [];
  try { const d = await api('/v1/eval/suites'); suites = d.suites || []; } catch (_) { /* leave hidden */ }
  for (const kind of ['sft', 'grpo', 'openenv']) {
    const row = document.getElementById(kind + '-prove-row');
    const sel = document.getElementById(kind + '-prove-suite');
    const check = document.getElementById(kind + '-prove');
    if (!row || !sel || !check) continue;
    row.dataset.hasSuites = suites.length ? 'true' : 'false';
    if (!suites.length) {
      check.checked = false;
      row.hidden = true;
      updateProveControls(kind);
      continue;
    }
    row.hidden = kind === 'openenv' && document.getElementById('openenv-kind')?.value !== 'train';
    const cur = sel.value;
    sel.innerHTML = suites.map(s => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.name)}${s.num_examples ? ' · ' + s.num_examples + ' examples' : ''}</option>`).join('');
    if (cur && suites.some(s => s.name === cur)) sel.value = cur;
    updateProveControls(kind);
  }
}
for (const kind of ['sft', 'grpo', 'openenv']) {
  document.getElementById(kind + '-prove')?.addEventListener('change', () => updateProveControls(kind));
  document.getElementById(kind + '-prove-scope')?.addEventListener('change', () => updateProveControls(kind));
}
function provePostEval(kind) {
  const check = document.getElementById(kind + '-prove');
  const sel = document.getElementById(kind + '-prove-suite');
  const scope = document.getElementById(kind + '-prove-scope');
  if (!check || !check.checked || !sel || !sel.value) return null;
  return { suite: sel.value, data_scope: scope?.value || 'held-out', include_baseline: true };
}

// Dataset picker, per-form and format-correct (fixes the old SFT-only picker
// that filtered the wrong format and was always empty). Loading sets
// trainingData directly — never dumps rows into a textarea.
async function refreshDatasetPicker(kind) {
  const K = TRAIN_KIND[kind];
  const sel = document.getElementById(K.pickId);
  if (!sel) return;
  try {
    const d = await api('/v1/eval/datasets');
    const datasets = (d.datasets || []).filter(m => m.format === K.datasetFormat);
    const cur = sel.value;
    sel.innerHTML = '<option value="">Select an uploaded dataset…</option>'
      + datasets.map(m => {
        const counts = m.split_counts || {};
        const train = Number.isFinite(Number(counts.train)) ? Number(counts.train) : Number(m.num_rows || 0);
        const validation = Number(counts.validation || 0);
        const holdout = Number(counts.holdout || 0);
        return `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)} · ${train.toLocaleString()} train · ${validation.toLocaleString()} validation · ${holdout.toLocaleString()} holdout</option>`;
      }).join('');
    if (cur) sel.value = cur;
    // An empty picker is a dead-end without directions — say where data comes from.
    const empty = document.getElementById(kind + '-pick-empty');
    if (empty) empty.hidden = datasets.length > 0;
    sel.hidden = datasets.length === 0;
  } catch (_) { /* best-effort */ }
}
async function loadNamedDatasetIntoTraining(kind, name) {
  const K = TRAIN_KIND[kind];
  // Reference the dataset BY NAME — the server trains on its own copy
  // (/v1/train/* accepts `dataset`), so rows never round-trip through the
  // browser and nothing is truncated. We only fetch the manifest for the count.
  let manifest = null;
  try {
    const d = await api('/v1/eval/datasets');
    manifest = (d.datasets || []).find(x => x.name === name) || null;
  } catch (_) {}
  const counts = manifest?.split_counts || {};
  const count = manifest
    ? (Number.isFinite(Number(counts.train)) ? Number(counts.train) : Number(manifest.num_rows || 0))
    : null;
  const validation = Number(counts.validation || 0);
  const holdout = Number(counts.holdout || 0);
  trainingData[kind] = {
    datasetName: name,
    split: 'train',
    count,
    label: name,
    splitManifestSha256: manifest?.split_manifest_sha256 || null,
    datasetCorpusSha256: manifest?.corpus_sha256 || null,
  };
  const ta = document.getElementById(K.textareaId);
  if (ta) ta.value = '';
  const el = document.getElementById(kind + '-data-status');
  if (el) {
    el.hidden = false; el.className = 'train-data-status is-good';
    el.innerHTML = `${icon('check', 'icn-sm')} <strong>${escapeHtml(name)}</strong>`
      + (count != null ? ` · ${Number(count).toLocaleString()} train ${K.noun}${count === 1 ? '' : 's'}` : '')
      + (manifest ? ` <span class="train-data-src">(${validation.toLocaleString()} validation · ${holdout.toLocaleString()} holdout) · persisted train partition</span>` : '');
  }
  K.update();
  maybeSuggestAdapterName(kind, name);
  return count == null ? 1 : count;
}
async function loadDatasetIntoTraining(kind) {
  const sel = document.getElementById(TRAIN_KIND[kind].pickId);
  const name = sel && sel.value;
  if (!name) return; // placeholder option — nothing to load
  try { await loadNamedDatasetIntoTraining(kind, name); }
  catch (e) { toast('Load failed: ' + e.message, 'err'); }
}
// One-click bridge from anywhere a dataset is listed: jump to the right
// training form with the dataset already loaded and the adapter pre-named —
// the next action is just "Train adapter".
async function trainFromDataset(name, kind) {
  selectPage('training');
  document.getElementById('training-tab-' + kind)?.click();
  try {
    const n = await loadNamedDatasetIntoTraining(kind, name);
    if (n) toast(`${name} loaded — review the name, then Train adapter`, 'ok');
  } catch (e) { toast('Could not load ' + name + ': ' + e.message, 'err'); }
}
// Selecting a dataset LOADS it immediately — no separate Load click, and no
// way for the dropdown to show one dataset while another is actually held.
document.getElementById('sft-dataset-pick')?.addEventListener('change', () => loadDatasetIntoTraining('sft'));
document.getElementById('grpo-dataset-pick')?.addEventListener('change', () => loadDatasetIntoTraining('grpo'));
document.querySelectorAll('[data-goto-datasets]').forEach(b => b.addEventListener('click', () => {
  selectPage('evals');
  document.getElementById('evals-tab-datasets')?.click();
  setTimeout(() => document.getElementById('dataset-name')?.focus(), 120);
}));
document.getElementById('training-tab-sft')?.addEventListener('click', () => { refreshDatasetPicker('sft'); refreshProveRows(); });
document.getElementById('training-tab-grpo')?.addEventListener('click', () => { refreshDatasetPicker('grpo'); refreshProveRows(); });
refreshProveRows();

/* ====== Native OpenEnv RL =================================================
   This surface drives the persisted /v1/openenv lifecycle. Collection never
   happens in the browser: Kiln owns discovery, policy inference, WebSocket
   sessions, reward capture, replay artifacts, training, post-evaluation, and
   cancellation. The OpenEnv run remains authoritative until learning ends. */
let openEnvRunsKey = null;
let openEnvPendingSubmission = null;

function openEnvIdempotencyKey() {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function openEnvUrls() {
  return (document.getElementById('openenv-environments')?.value || '')
    .split(/\r?\n/)
    .map(url => url.trim())
    .filter(Boolean);
}

function openEnvCredentialIds(environmentCount) {
  const raw = document.getElementById('openenv-credential-ids')?.value || '';
  if (!raw.trim()) return [];
  const values = raw.trim().split(/\r?\n/).map(value => value.trim());
  if (values.length !== environmentCount) {
    throw new Error(`Credential handles must be empty or contain exactly one line per environment URL (expected ${environmentCount}, got ${values.length}).`);
  }
  return values.map((value, index) => {
    if (value === '-') return null;
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(value)) {
      throw new Error(`Credential handle ${index + 1} must use 1–64 letters, digits, "_" or "-", or be exactly "-" for a public slot.`);
    }
    return value;
  });
}

function openEnvEnvironmentResetOptions(environmentCount) {
  const text = (document.getElementById('openenv-environment-reset-options')?.value || '').trim();
  if (!text) return [];
  let values;
  try {
    values = JSON.parse(text);
  } catch (error) {
    throw new Error(`Per-environment reset options must be valid JSON: ${error.message}`);
  }
  if (!Array.isArray(values) || values.length !== environmentCount) {
    throw new Error(`Per-environment reset options must be an array with exactly one object per environment URL (expected ${environmentCount}, got ${Array.isArray(values) ? values.length : 'a non-array value'}).`);
  }
  values.forEach((value, index) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(`Per-environment reset option ${index + 1} must be one JSON object.`);
    }
  });
  return values;
}

function openEnvNumber(id, label, integer = true) {
  const value = Number(document.getElementById(id)?.value);
  if (!Number.isFinite(value) || (integer && !Number.isSafeInteger(value))) {
    throw new Error(`${label} must be a ${integer ? 'whole ' : ''}number.`);
  }
  return value;
}

function openEnvOptionalNumber(id, label) {
  const text = (document.getElementById(id)?.value || '').trim();
  if (!text) return null;
  const value = Number(text);
  if (!Number.isFinite(value)) throw new Error(`${label} must be a finite number.`);
  return value;
}

function openEnvObject(id, label) {
  const text = (document.getElementById(id)?.value || '').trim() || '{}';
  let value;
  try { value = JSON.parse(text); }
  catch { throw new Error(`${label} must be valid JSON.`); }
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    throw new Error(`${label} must be one JSON object.`);
  }
  return value;
}

function openEnvOptimizerKind(config) {
  if (config.optimizer == null) return 'muon';
  if (Array.isArray(config.optimizer) || typeof config.optimizer !== 'object') {
    throw new Error('Native GRPO optimizer must be an object such as {"kind":"muon"}.');
  }
  const kind = config.optimizer.kind;
  if (!['muon', 'adam_w', 'sgd'].includes(kind)) {
    throw new Error('Native GRPO optimizer.kind must be muon, adam_w, or sgd.');
  }
  return kind;
}

function syncOpenEnvKind() {
  const train = document.getElementById('openenv-kind')?.value === 'train';
  const environmentEvalMode = document.getElementById('openenv-environment-eval-mode');
  const environmentEvalEnabled = train && environmentEvalMode?.value !== 'off';
  const environmentGateEnabled = environmentEvalEnabled && environmentEvalMode?.value === 'gate';
  const outputGroup = document.getElementById('openenv-output-group');
  const output = document.getElementById('openenv-output-adapter');
  const rank = document.getElementById('openenv-lora-rank');
  const optimizerStatus = document.getElementById('openenv-optimizer-support');
  const proveRow = document.getElementById('openenv-prove-row');
  const proveCheck = document.getElementById('openenv-prove');
  if (outputGroup) outputGroup.hidden = !train;
  if (output) {
    output.disabled = !train;
    output.required = train;
  }
  if (environmentEvalMode) environmentEvalMode.disabled = !train;
  if (proveRow) proveRow.hidden = !train || proveRow.dataset.hasSuites !== 'true';
  if (proveCheck) proveCheck.disabled = !train;
  if (train) updateProveControls('openenv');
  else {
    const proveSuite = document.getElementById('openenv-prove-suite');
    const proveScope = document.getElementById('openenv-prove-scope');
    if (proveSuite) proveSuite.disabled = true;
    if (proveScope) proveScope.disabled = true;
  }
  ['openenv-environment-eval-groups', 'openenv-environment-eval-group-size'].forEach(id => {
    const input = document.getElementById(id);
    if (input) input.disabled = !environmentEvalEnabled;
  });
  ['openenv-environment-gate-floor', 'openenv-environment-gate-improvement'].forEach(id => {
    const input = document.getElementById(id);
    if (input) input.disabled = !environmentGateEnabled;
  });
  const floorGroup = document.getElementById('openenv-environment-gate-floor-group');
  const improvementGroup = document.getElementById('openenv-environment-gate-improvement-group');
  if (floorGroup) floorGroup.hidden = !environmentGateEnabled;
  if (improvementGroup) improvementGroup.hidden = !environmentGateEnabled;
  const submit = document.querySelector('#openenv-form button[type="submit"]');
  if (!train) {
    if (rank) rank.disabled = true;
    if (optimizerStatus) optimizerStatus.textContent = 'Rollout-only runs do not require native optimizer admission.';
    if (submit) {
      submit.disabled = false;
      submit.title = '';
      submit.textContent = 'Collect replayable rollouts';
    }
    return;
  }
  let admission;
  let optimizerKind = 'muon';
  try {
    const config = openEnvObject('openenv-training-config', 'Native GRPO overrides');
    optimizerKind = openEnvOptimizerKind(config);
    admission = trainingOptimizerAdmissionState('grpo', optimizerKind, rank?.value);
  } catch (error) {
    admission = { ready: false, reason: error.message };
  }
  if (rank) {
    rank.disabled = !trainingOptimizerKindState('grpo', optimizerKind).ready;
    applyOptimizerRankBounds(rank, optimizerSupportEntry(optimizerKind));
  }
  if (optimizerStatus) {
    optimizerStatus.textContent = admission.ready
      ? optimizerSupportStatusFromState(admission, optimizerKind)
      : `${admission.reason}. Training remains disabled.`;
  }
  if (submit) {
    submit.disabled = !admission.ready;
    submit.title = admission.ready ? '' : admission.reason || 'Native GRPO is unavailable';
    submit.textContent = 'Collect & train';
  }
}

async function inspectOpenEnv() {
  const result = document.getElementById('openenv-inspection');
  const button = document.getElementById('openenv-inspect');
  const environment_urls = openEnvUrls();
  if (!environment_urls.length) {
    if (result) result.textContent = 'Add at least one environment URL first.';
    return;
  }
  let credential_ids;
  try {
    credential_ids = openEnvCredentialIds(environment_urls.length);
  } catch (error) {
    if (result) result.textContent = error.message;
    toast(error.message, 'err');
    return;
  }
  if (button) button.disabled = true;
  if (result) result.textContent = 'Discovering health, metadata, schemas, and protocol identity…';
  try {
    const response = await api('/v1/openenv/inspect', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ environment_urls, ...(credential_ids.length ? {credential_ids} : {}) }),
    });
    const environments = response.environments || [];
    if (result) {
      result.innerHTML = environments.map(environment => {
        const identity = environment.identity || {};
        const metadata = identity.metadata || {};
        const action = environment.schema?.action || {};
        return `<span><strong>${escapeHtml(metadata.name || 'OpenEnv')}</strong> · ${escapeHtml(identity.client_profile || 'compatible')} · schema <code>${escapeHtml((identity.schema_sha256 || '').slice(0, 12))}</code> · action <code>${escapeHtml(JSON.stringify(action))}</code></span>`;
      }).join('<br>');
    }
  } catch (error) {
    if (result) result.textContent = error.message;
    toast(error.message, 'err');
  } finally {
    if (button) button.disabled = false;
  }
}

async function inspectOpenEnvTasks() {
  const result = document.getElementById('openenv-task-catalog');
  const button = document.getElementById('openenv-task-inspect');
  const environment_urls = openEnvUrls();
  if (!environment_urls.length) {
    if (result) result.textContent = 'Add at least one environment URL first.';
    return;
  }
  let credential_ids;
  try {
    credential_ids = openEnvCredentialIds(environment_urls.length);
  } catch (error) {
    if (result) result.textContent = error.message;
    toast(error.message, 'err');
    return;
  }
  const environment_name = document.getElementById('openenv-task-environment-name')?.value.trim() || '';
  const split = document.getElementById('openenv-task-split')?.value.trim() || '';
  const start = Number(document.getElementById('openenv-task-start')?.value || 0);
  const limit = Number(document.getElementById('openenv-task-limit')?.value || 20);
  if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 200) {
    const message = 'Task page requires a non-negative safe-integer start and a page size from 1 to 200.';
    if (result) result.textContent = message;
    toast(message, 'err');
    return;
  }
  if (button) button.disabled = true;
  if (result) result.textContent = split
    ? 'Reading the bounded OpenEnv task page…'
    : 'Discovering OpenEnv Task API splits…';
  try {
    const response = await api('/v1/openenv/tasks', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        environment_urls,
        ...(credential_ids.length ? {credential_ids} : {}),
        ...(environment_name ? {environment_name} : {}),
        ...(split ? {split} : {}),
        start,
        limit,
      }),
    });
    if (result) {
      result.innerHTML = (response.catalogs || []).map(entry => {
        const catalog = entry.catalog || {};
        const name = catalog.environment_name || entry.base_url || 'OpenEnv';
        if (catalog.task_api === 'unsupported') {
          return `<span><strong>${escapeHtml(name)}</strong> · Task API unsupported · seeded reset/options training remains available</span>`;
        }
        const splits = (catalog.splits || []).map(item =>
          `<code>${escapeHtml(item.name || '')}</code> (${escapeHtml(item.type || '')})`
        ).join(', ');
        if (!catalog.selected_split) {
          return `<span><strong>${escapeHtml(name)}</strong> · Task API available · ${splits || 'no splits advertised'}</span>`;
        }
        const rows = (catalog.tasks || []).map((task, offset) =>
          `<span><code>[${Number(catalog.start || 0) + offset}]</code> <code>${escapeHtml(JSON.stringify(task))}</code></span>`
        ).join('<br>');
        return `<span><strong>${escapeHtml(name)}</strong> · <code>${escapeHtml(catalog.selected_split)}</code> tasks ${Number(catalog.start || 0)}..${Number(catalog.stop || 0)} of ${Number(catalog.num_tasks || 0)}${rows ? `<br>${rows}` : '<br>No tasks in this page.'}</span>`;
      }).join('<br>');
    }
  } catch (error) {
    if (result) result.textContent = error.message;
    toast(error.message, 'err');
  } finally {
    if (button) button.disabled = false;
  }
}

function openEnvStateLabel(state) {
  return String(state || 'unknown').replaceAll('_', ' ');
}

function openEnvRunCard(run) {
  const state = String(run.state || 'unknown');
  const progress = run.progress || {};
  const groupTotal = Number(progress.groups_total || 0);
  const groupDone = Number(progress.groups_completed || 0);
  const training = run.training || null;
  const trainingContract = run.training_contract || null;
  const admission = run.admission || null;
  const evaluations = Array.isArray(run.post_evaluations) ? run.post_evaluations : [];
  const environmentEvaluation = run.environment_evaluation || null;
  const evalDone = evaluations.reduce((sum, item) => sum + Number(item.examples_completed || 0), 0);
  const evalTotal = evaluations.reduce((sum, item) => sum + Number(item.examples_total || 0), 0);
  let pct = groupTotal ? Math.min(100, Math.round(groupDone / groupTotal * 100)) : 0;
  let statValue = `${groupDone}/${groupTotal}`;
  let statLabel = 'seed groups';
  if (state === 'queued') {
    pct = 0;
    statValue = admission?.queue_position ? `#${Number(admission.queue_position).toLocaleString()}` : 'queued';
    statLabel = 'execution queue';
  } else if (state === 'training_queued') {
    pct = 0;
    statValue = 'queued';
    statLabel = 'native GRPO';
  } else if (state === 'training_running' && training) {
    pct = Math.min(100, Math.max(0, Math.round(Number(training.progress || 0) * 100)));
    statValue = `${pct}%`;
    statLabel = 'native GRPO';
  } else if (state === 'post_evaluating') {
    pct = evalTotal ? Math.min(100, Math.round(evalDone / evalTotal * 100)) : 0;
    statValue = evalTotal ? `${evalDone}/${evalTotal}` : 'queued';
    statLabel = 'eval examples';
  } else if (state === 'environment_evaluating' && environmentEvaluation) {
    const environmentProgress = environmentEvaluation.progress || {};
    const environmentDone = Number(environmentProgress.groups_completed || 0);
    const environmentTotal = Number(environmentProgress.groups_total || 0);
    pct = environmentTotal ? Math.min(100, Math.round(environmentDone / environmentTotal * 100)) : 0;
    statValue = environmentTotal ? `${environmentDone}/${environmentTotal}` : 'queued';
    statLabel = String(environmentEvaluation.state || 'environment eval').replaceAll('_', ' ');
  } else if (state === 'completed' || state === 'rollout_ready') {
    pct = 100;
  }
  const terminal = ['rollout_ready', 'completed', 'failed', 'cancelled'].includes(state);
  const environments = (run.environments || []).map(item => item.metadata?.name || item.identity?.metadata?.name).filter(Boolean);
  const artifacts = (run.artifacts || []).map(artifact => {
    const digest = String(artifact.sha256 || '');
    const shortDigest = digest.startsWith('sha256:') ? digest.slice(7, 19) : digest.slice(0, 12);
    return `<a class="btn btn-sm" href="${escapeHtml(artifact.url)}" download title="Manifest-bound artifact ${escapeHtml(digest)}">${escapeHtml(artifact.kind)}${artifact.bytes ? ` · ${fmtBytes(artifact.bytes)}` : ''}${shortDigest ? ` · <code>${escapeHtml(shortDigest)}</code>` : ''}</a>`;
  }).join('');
  const job = run.training_job_id
    ? `<button class="btn btn-sm" type="button" data-openenv-training-job="${escapeHtml(run.training_job_id)}">Training ${escapeHtml(run.training_job_id.slice(0, 8))}</button>`
    : '';
  const cancel = terminal ? '' : `<button class="btn btn-sm btn-danger" type="button" data-openenv-cancel="${escapeHtml(run.run_id)}">Cancel</button>`;
  const failure = run.failure || null;
  const error = failure
    ? `<div class="training-card-error"><strong>${escapeHtml(String(failure.code || 'internal_error'))}</strong> · ${escapeHtml(openEnvStateLabel(failure.stage || 'orchestration'))} · ${failure.retryable ? 'retryable' : 'not retryable'}${failure.protocol_code ? ` · OpenEnv ${escapeHtml(failure.protocol_code)}` : ''}${failure.http_status ? ` · HTTP ${Number(failure.http_status)}` : ''}<br>${escapeHtml(failure.message || run.error || 'OpenEnv run failed.')}${failure.hint ? `<br><span class="hint">Next: ${escapeHtml(failure.hint)}</span>` : ''}</div>`
    : run.error
      ? `<div class="training-card-error">${escapeHtml(run.error)}</div>`
      : '';
  const admissionDetail = admission
    ? state === 'queued'
      ? `<div class="training-card-meta">FIFO execution queue · position ${Number(admission.queue_position || 0).toLocaleString()} · ${Number(admission.max_active_runs || 0).toLocaleString()} active slot${Number(admission.max_active_runs || 0) === 1 ? '' : 's'}</div>`
      : admission.queue_wait_ms != null
        ? `<div class="training-card-meta">Execution admitted after ${escapeHtml(fmtMsShort(Number(admission.queue_wait_ms)))}</div>`
        : ''
    : '';
  const submissionDetail = run.request?.idempotency_key
    ? `<div class="training-card-meta">Retry key · <code>${escapeHtml(run.request.idempotency_key)}</code></div>`
    : '';
  const effectiveConfig = trainingContract?.effective_config || null;
  const contractPolicy = trainingContract?.behavior_policy || null;
  const contractPolicyDigest = contractPolicy?.adapter?.content_sha256 || contractPolicy?.base_model_sha256 || '';
  const contractPolicyShort = contractPolicyDigest.startsWith('sha256:')
    ? contractPolicyDigest.slice(7, 19)
    : contractPolicyDigest.slice(0, 12);
  const contractDetail = effectiveConfig
    ? `<div class="training-card-meta">Admitted contract · ${escapeHtml(effectiveConfig.optimizer?.kind || 'muon')} · rank ${Number(effectiveConfig.lora_rank || 8).toLocaleString()} · policy <code>${escapeHtml(contractPolicy?.adapter?.name || 'base')}${contractPolicyShort ? `@${escapeHtml(contractPolicyShort)}` : ''}</code> · output <code>${escapeHtml(effectiveConfig.output_name || 'unknown')}</code> · auto-load ${effectiveConfig.auto_load === false ? 'off' : 'on'}${trainingContract.post_eval?.suite ? ` · post-eval ${escapeHtml(trainingContract.post_eval.suite)}` : ''}</div>`
    : '';
  const trainingDetail = training
    ? `<div class="training-card-meta">Trainer · ${escapeHtml(String(training.state || 'unknown'))} · ${Math.round(Number(training.progress || 0) * 100)}%${training.current_loss != null ? ` · loss ${Number(training.current_loss).toFixed(4)}` : ''}${training.epoch != null ? ` · epoch ${Number(training.epoch).toLocaleString()}` : ''}</div>`
    : '';
  const trainingLineage = training?.training_data?.openenv || null;
  const lineageEnvironmentNames = Array.isArray(trainingLineage?.environments)
    ? trainingLineage.environments.map(environment => environment.environment_name).filter(Boolean).join(', ')
    : '';
  const lineagePolicy = trainingLineage?.behavior_policy || null;
  const lineagePolicyDigest = lineagePolicy?.adapter?.content_sha256 || lineagePolicy?.base_model_sha256 || '';
  const lineagePolicyShort = lineagePolicyDigest.startsWith('sha256:')
    ? lineagePolicyDigest.slice(7, 19)
    : lineagePolicyDigest.slice(0, 12);
  const lineageDetail = trainingLineage
    ? `<div class="training-card-data" title="Admitted corpus ${escapeHtml(training.training_data.admitted_corpus_sha256 || '')}; OpenEnv task plan ${escapeHtml(trainingLineage.group_plan_sha256 || '')}; behavior policy ${escapeHtml(lineagePolicyDigest)}">${icon('stack', 'icn-sm')} OpenEnv corpus · ${escapeHtml(lineageEnvironmentNames || 'compatible environment')} · ${Number(trainingLineage.groups || 0).toLocaleString()} groups · ${Number(trainingLineage.rollouts || 0).toLocaleString()} rollouts · policy ${escapeHtml(lineagePolicy?.adapter?.name || 'base')}${lineagePolicyShort ? `@${escapeHtml(lineagePolicyShort)}` : ''} · seeds ${escapeHtml(String(trainingLineage.seed_min ?? 'unknown'))}–${escapeHtml(String(trainingLineage.seed_max ?? 'unknown'))}</div>`
    : '';
  const evalDetail = evaluations.length
    ? `<div class="training-card-meta">${evaluations.map(item => `${escapeHtml(item.suite_name)} · ${escapeHtml(String(item.state || 'unknown'))}${item.headline_accuracy != null ? ` · ${(Number(item.headline_accuracy) * 100).toFixed(1)}%` : ''}`).join('<br>')}</div>`
    : '';
  const environmentEvidence = environmentEvaluation?.evidence || null;
  const environmentDetail = environmentEvaluation
    ? `<div class="training-card-meta">Held-out environment · ${escapeHtml(String(environmentEvaluation.state || 'pending').replaceAll('_', ' '))}${environmentEvidence ? ` · return ${Number(environmentEvidence.baseline_mean_return).toFixed(3)} → ${Number(environmentEvidence.candidate_mean_return).toFixed(3)} (${Number(environmentEvidence.mean_return_improvement) >= 0 ? '+' : ''}${Number(environmentEvidence.mean_return_improvement).toFixed(3)}) · exact p=${Number(environmentEvidence.exact_sign_test_p_value).toFixed(4)}` : ''}</div>`
    : '';
  const gateOutcome = environmentEvaluation?.outcome || training?.gate_outcome;
  const gate = gateOutcome
    ? `<span class="training-card-type">${escapeHtml(gateOutcome)}</span>`
    : '';
  const cardState = state === 'failed'
    ? 'training-card-failed'
    : terminal
      ? 'training-card-completed'
      : 'training-card-running';
  return `<div class="training-card ${cardState}">
    <div class="training-card-head">
      <div><strong>${run.kind === 'train' ? 'OpenEnv train' : 'OpenEnv rollout'}</strong> <code>${escapeHtml(run.run_id.slice(0, 8))}</code></div>
      <span class="training-card-type">${escapeHtml(openEnvStateLabel(state))}</span>${gate}
    </div>
    <div class="training-card-meta">${escapeHtml(run.request?.adapter || 'base')} policy · ${Number(progress.rollouts_completed || 0).toLocaleString()} / ${Number(progress.rollouts_total || 0).toLocaleString()} episodes${environments.length ? ` · ${escapeHtml(environments.join(', '))}` : ''}</div>
    ${submissionDetail}
    ${contractDetail}
    ${admissionDetail}
    ${trainingDetail}
    ${lineageDetail}
    ${evalDetail}
    ${environmentDetail}
    <div class="training-card-progress">
      <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
      <div class="training-stat"><span class="training-stat-num">${escapeHtml(statValue)}</span><span class="training-stat-label">${escapeHtml(statLabel)}</span></div>
    </div>
    ${error}
    <div style="display:flex;gap:var(--space-2);flex-wrap:wrap;margin-top:var(--space-3);">${artifacts}${job}${cancel}</div>
  </div>`;
}

async function pollOpenEnvRuns(force = false) {
  const list = document.getElementById('openenv-runs');
  if (!list) return;
  try {
    const response = await api('/v1/openenv/runs');
    const runs = response.runs || [];
    const key = JSON.stringify(runs);
    if (force || key !== openEnvRunsKey) {
      openEnvRunsKey = key;
      list.className = runs.length ? '' : 'empty';
      list.innerHTML = runs.length
        ? runs.map(openEnvRunCard).join('')
        : '<div class="empty">No OpenEnv runs yet. Inspect an environment, then launch your first rollout.</div>';
    }
    list.setAttribute('aria-busy', 'false');
  } catch (error) {
    list.className = 'empty error';
    list.textContent = error.message;
    list.setAttribute('aria-busy', 'false');
  }
}

document.getElementById('openenv-kind')?.addEventListener('change', syncOpenEnvKind);
document.getElementById('openenv-lora-rank')?.addEventListener('input', syncOpenEnvKind);
document.getElementById('openenv-training-config')?.addEventListener('input', syncOpenEnvKind);
document.getElementById('openenv-environment-eval-mode')?.addEventListener('change', syncOpenEnvKind);
document.getElementById('openenv-inspect')?.addEventListener('click', inspectOpenEnv);
document.getElementById('openenv-task-inspect')?.addEventListener('click', inspectOpenEnvTasks);
document.getElementById('openenv-refresh')?.addEventListener('click', () => pollOpenEnvRuns(true));
document.getElementById('training-tab-openenv')?.addEventListener('click', () => {
  pollOpenEnvRuns(true);
  refreshProveRows();
});
document.getElementById('openenv-adv-toggle')?.addEventListener('click', event => {
  const body = document.getElementById('openenv-advanced');
  const open = body?.hidden;
  if (body) body.hidden = !open;
  event.currentTarget.setAttribute('aria-expanded', String(open));
});
document.getElementById('openenv-runs')?.addEventListener('click', async event => {
  const cancel = event.target.closest('[data-openenv-cancel]');
  if (cancel) {
    cancel.disabled = true;
    cancel.textContent = 'Cancelling…';
    try {
      await api('/v1/openenv/runs/' + encodeURIComponent(cancel.dataset.openenvCancel), {method:'DELETE'});
      toast('OpenEnv cancellation requested', 'ok');
      pollOpenEnvRuns(true);
    } catch (error) {
      cancel.disabled = false;
      cancel.textContent = 'Cancel';
      toast(error.message, 'err');
    }
    return;
  }
  const job = event.target.closest('[data-openenv-training-job]');
  if (job) document.getElementById('training-tab-queue')?.click();
});
document.getElementById('openenv-form')?.addEventListener('submit', async event => {
  event.preventDefault();
  const submit = event.currentTarget.querySelector('button[type="submit"]');
  const status = document.getElementById('openenv-submit-state');
  try {
    const kind = document.getElementById('openenv-kind').value;
    const environment_urls = openEnvUrls();
    if (!environment_urls.length) throw new Error('Add at least one OpenEnv environment URL.');
    const credential_ids = openEnvCredentialIds(environment_urls.length);
    const environment_reset_options = openEnvEnvironmentResetOptions(environment_urls.length);
    const training_config = openEnvObject('openenv-training-config', 'Native GRPO overrides');
    const rank = openEnvNumber('openenv-lora-rank', 'LoRA rank');
    const optimizerKind = openEnvOptimizerKind(training_config);
    if (kind === 'train') requireTrainingOptimizerAdmission('grpo', optimizerKind, rank, 'OpenEnv GRPO');
    training_config.lora_rank = rank;
    const reset_options = openEnvObject('openenv-reset-options', 'Reset options');
    if (environment_reset_options.length && Object.keys(reset_options).length) {
      throw new Error('Use either shared reset options or per-environment reset options, not both.');
    }
    const request = {
      kind,
      environment_urls,
      adapter: document.getElementById('openenv-adapter').value.trim() || 'base',
      groups: openEnvNumber('openenv-groups', 'Seed groups'),
      group_size: openEnvNumber('openenv-group-size', 'Episodes per seed'),
      seed_start: openEnvNumber('openenv-seed-start', 'First seed'),
      reset_options,
      max_steps: openEnvNumber('openenv-max-steps', 'Max actions'),
      concurrency: openEnvNumber('openenv-concurrency', 'Concurrency'),
      max_action_tokens: openEnvNumber('openenv-max-action-tokens', 'Max action tokens'),
      temperature: openEnvNumber('openenv-temperature', 'Action temperature', false),
      thinking: document.getElementById('openenv-thinking').checked,
      protocol_error_reward: openEnvNumber('openenv-protocol-error-reward', 'Protocol-error reward', false),
      max_recoverable_errors: openEnvNumber('openenv-max-recoverable-errors', 'Recoverable errors'),
      capacity_wait_seconds: openEnvNumber('openenv-capacity-wait', 'Capacity wait'),
      auto_load: document.getElementById('openenv-auto-load').checked,
    };
    if (credential_ids.length) request.credential_ids = credential_ids;
    if (environment_reset_options.length) request.environment_reset_options = environment_reset_options;
    if (request.groups < environment_urls.length) {
      throw new Error(`Seed groups must be at least the number of environment URLs (${environment_urls.length}) so every endpoint is exercised.`);
    }
    if (kind === 'train') {
      request.output_adapter = document.getElementById('openenv-output-adapter').value.trim();
      if (!request.output_adapter) throw new Error('Choose an output adapter name.');
      request.training_config = training_config;
      const postEval = provePostEval('openenv');
      if (postEval) request.post_eval = postEval;
      const environmentEvalMode = document.getElementById('openenv-environment-eval-mode').value;
      if (environmentEvalMode !== 'off') {
        request.environment_eval = {
          groups: openEnvNumber('openenv-environment-eval-groups', 'Held-out seed groups'),
          group_size: openEnvNumber('openenv-environment-eval-group-size', 'Held-out episodes per seed'),
        };
        if (request.environment_eval.groups < environment_urls.length) {
          throw new Error(`Held-out seed groups must be at least the number of environment URLs (${environment_urls.length}) so every endpoint is evaluated.`);
        }
        if (environmentEvalMode === 'gate') {
          const minMeanReturn = openEnvOptionalNumber('openenv-environment-gate-floor', 'Minimum mean return');
          request.environment_eval.gate = {
            min_mean_improvement: openEnvNumber('openenv-environment-gate-improvement', 'Minimum mean improvement', false),
          };
          if (minMeanReturn != null) request.environment_eval.gate.min_mean_return = minMeanReturn;
        }
      }
    }
    const requestFingerprint = JSON.stringify(request);
    if (!openEnvPendingSubmission || openEnvPendingSubmission.fingerprint !== requestFingerprint) {
      openEnvPendingSubmission = {
        fingerprint: requestFingerprint,
        key: openEnvIdempotencyKey(),
      };
    }
    request.idempotency_key = openEnvPendingSubmission.key;
    submit.disabled = true;
    if (status) status.textContent = 'Creating persisted OpenEnv run…';
    const run = await api('/v1/openenv/runs', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(request),
    });
    openEnvPendingSubmission = null;
    if (status) status.textContent = `Run ${run.run_id.slice(0, 8)} accepted. Kiln owns it through collection, training, and requested evaluation.`;
    toast(`OpenEnv ${kind} run ${run.run_id.slice(0, 8)} started`, 'ok');
    pollOpenEnvRuns(true);
  } catch (error) {
    if (status) status.textContent = error.message;
    toast(error.message, 'err');
  } finally {
    syncOpenEnvKind();
  }
});
syncOpenEnvKind();
pollOpenEnvRuns();
setInterval(pollOpenEnvRuns, 4000);

function parseJsonArrayField(value, label) {
  const text = value.trim();
  if (!text) {
    throw new Error(`${label} cannot be empty. Paste a JSON array or use the sample payload.`);
  }

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} must be valid JSON. Check commas, quotes, and brackets.`);
  }

  if (!Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON array, not an object or single item.`);
  }
  if (parsed.length === 0) {
    throw new Error(`${label} must include at least one item.`);
  }
  return parsed;
}

function parseFiniteNumberField(value, label) {
  const text = value.trim();
  if (!text) {
    throw new Error(`${label} is required.`);
  }
  const parsed = Number(text);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return parsed;
}

// Blank means "omit the field" (the server resolves a default); anything
// non-blank must still be a real number.
function parseOptionalFiniteNumberField(value, label) {
  if (!value.trim()) return null;
  return parseFiniteNumberField(value, label);
}

function parsePositiveIntegerField(value, label) {
  const parsed = parseFiniteNumberField(value, label);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive whole number.`);
  }
  return parsed;
}

function parseOptionalPositiveIntegerField(value, label) {
  if (!value.trim()) return null;
  return parsePositiveIntegerField(value, label);
}

function parseResumeCheckpointField(value, label) {
  const checkpoint = value.trim();
  if (!checkpoint) return null;
  if (!isPathSafeAdapterDirectoryName(checkpoint) || !checkpoint.endsWith('.kiln-checkpoint')) {
    throw new Error(`${label} must be one direct .kiln-checkpoint basename, without a path.`);
  }
  return checkpoint;
}

function parseQuickInferenceTemperature(input) {
  const text = input.value.trim();
  const parsed = Number(text);
  if (!text || !Number.isFinite(parsed) || parsed < 0 || parsed > 2) {
    throw new Error('Temperature must be between 0 and 2.');
  }
  return parsed;
}

function validateMessages(messages, label) {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error(`${label} needs a non-empty messages array.`);
  }
  const roles = messages.map((message) => message && message.role);
  if (!roles.includes('user') || !roles.includes('assistant')) {
    throw new Error(`${label} messages need both user and assistant roles.`);
  }
}

function validateSftExamples(examples) {
  examples.forEach((example, index) => {
    validateMessages(example && example.messages, `SFT example ${index + 1}`);
  });
}

function validateGrpoGroups(groups) {
  groups.forEach((group, groupIndex) => {
    const label = `GRPO group ${groupIndex + 1}`;
    if (!group || typeof group !== 'object') {
      throw new Error(`${label} must be an object with messages and completions.`);
    }
    if (!Array.isArray(group.messages) || group.messages.length === 0) {
      throw new Error(`${label} needs a non-empty messages array.`);
    }
    if (!Array.isArray(group.completions) || group.completions.length === 0) {
      throw new Error(`${label} needs a non-empty completions array.`);
    }
    group.completions.forEach((completion, completionIndex) => {
      if (!completion || typeof completion.text !== 'string' || !completion.text.trim()) {
        throw new Error(`${label} completion ${completionIndex + 1} needs non-empty text.`);
      }
      if (typeof completion.reward !== 'number' || !Number.isFinite(completion.reward)) {
        throw new Error(`${label} completion ${completionIndex + 1} needs a numeric reward, not a quoted string.`);
      }
    });
  });
}

function parseAdapterNameField(input) {
  const adapterName = input.value.trim();
  if (!adapterName) {
    input.focus();
    throw new Error('Adapter name is required. Use a short, path-safe name.');
  }
  return adapterName;
}

function trainingOutputNameReadinessState(formId, label) {
  const form = document.getElementById(formId);
  const input = form ? form.querySelector('input[name="output_name"]') : null;
  const outputName = input ? input.value.trim() : '';
  if (!outputName) {
    return { ready: false, message: `Enter a path-safe ${label} adapter name to enable submit.` };
  }
  if (!isPathSafeAdapterDirectoryName(outputName)) {
    return { ready: false, message: pathSafeAdapterDirectoryNameMessage() };
  }
  return { ready: true, message: 'Ready to submit with this path-safe adapter name.' };
}

function trainingPayloadReadinessState(textareaId, label, dataKind) {
  // A file (parsed items) or a server-side dataset reference held in
  // trainingData[dataKind] makes the form ready; the status chip narrates it.
  const held = (dataKind && typeof trainingData !== 'undefined') ? trainingData[dataKind] : null;
  if (held && ((held.items && held.items.length) || held.datasetName)) {
    // Mirror the visual status chip in this aria-live line so screen-reader
    // users tracking it hear the readiness change too.
    const what = held.datasetName
      ? `Dataset ${held.label || held.datasetName} ready — trains on the server's copy.`
      : `${held.items.length} item${held.items.length === 1 ? '' : 's'} ready from ${held.label || 'your file'}.`;
    return { ready: true, message: what + ' Train adapter is enabled.' };
  }
  const textarea = document.getElementById(textareaId);
  if (!textarea || !textarea.value.trim()) {
    return { ready: false, message: `Drop a file, pick a dataset, paste ${label} JSON, or try a sample to enable training.` };
  }
  return { ready: true, message: `${label} pasted — Train will validate before queuing.` };
}

function updateTrainingSubmitState(options) {
  const outputState = trainingOutputNameReadinessState(options.formId, options.outputLabel);
  const payloadState = trainingPayloadReadinessState(options.payloadId, options.payloadLabel, options.dataKind);
  const optimizerKind = typeof options.optimizerKind === 'function' ? options.optimizerKind() : null;
  const optimizerRank = typeof options.optimizerRank === 'function' ? options.optimizerRank() : null;
  const optimizerState = trainingOptimizerAdmissionState(options.workload, optimizerKind, optimizerRank);
  const outputHelper = document.getElementById(options.outputStateId);
  const payloadHelper = document.getElementById(options.payloadStateId);
  if (outputHelper) outputHelper.textContent = outputState.message;
  if (payloadHelper) payloadHelper.textContent = payloadState.message;
  const form = document.getElementById(options.formId);
  const submitButton = form ? form.querySelector('button[type="submit"]') : null;
  if (submitButton) {
    submitButton.disabled = form?.dataset.trainingBusy === 'true'
      || !outputState.ready
      || !payloadState.ready
      || !optimizerState.ready;
    submitButton.title = optimizerState.ready ? '' : optimizerState.reason || 'Optimizer unavailable';
  }
  return {
    ready: outputState.ready && payloadState.ready && optimizerState.ready,
    outputState,
    payloadState,
    optimizerState,
  };
}

// Cross-check epochs against how much data is actually loaded — many passes
// over a handful of examples just memorizes them. Advisory, never blocking.
function updateSftOverfitHint() {
  const hint = document.getElementById('sft-overfit-hint');
  if (!hint) return;
  const epochs = parseInt(document.getElementById('sft-epochs')?.value || '0', 10) || 0;
  const held = trainingData.sft;
  const n = held && held.items ? held.items.length : (held && held.count ? held.count : 0);
  if (n > 0 && n < 20 && epochs > 10) {
    hint.hidden = false;
    hint.textContent = `${epochs} passes over only ${n} example${n === 1 ? '' : 's'} will likely memorize them — 3 is usually plenty.`;
  } else {
    hint.hidden = true;
  }
}

function updateSftSubmitState() {
  updateSftOverfitHint();
  return updateTrainingSubmitState({
    formId: 'sft-form',
    outputStateId: 'sft-output-name-state',
    outputLabel: 'SFT output',
    payloadId: 'sft-examples',
    payloadStateId: 'sft-examples-state',
    payloadLabel: 'examples',
    dataKind: 'sft',
    workload: 'sft',
    optimizerKind: () => document.getElementById('sft-optimizer')?.value,
    optimizerRank: () => document.getElementById('sft-rank')?.value,
  });
}

function updateGrpoSubmitState() {
  return updateTrainingSubmitState({
    formId: 'grpo-form',
    outputStateId: 'grpo-output-name-state',
    outputLabel: 'GRPO output',
    payloadId: 'grpo-groups',
    payloadStateId: 'grpo-groups-state',
    payloadLabel: 'groups',
    dataKind: 'grpo',
    workload: 'grpo',
    optimizerKind: () => document.getElementById('grpo-optimizer')?.value,
    optimizerRank: () => document.getElementById('grpo-rank')?.value,
  });
}

function updateOpdSubmitState() {
  const form = document.getElementById('opd-form');
  const submitButton = form?.querySelector('button[type="submit"]');
  if (!submitButton) return;
  const optimizerState = trainingOptimizerAdmissionState(
    'opd',
    'muon',
    document.getElementById('opd-rank')?.value,
  );
  submitButton.disabled = form.dataset.trainingBusy === 'true' || !optimizerState.ready;
  submitButton.title = optimizerState.ready ? '' : optimizerState.reason || 'Muon unavailable';
}

function updateSftOutputNameState() {
  return updateSftSubmitState().outputState;
}

function updateGrpoOutputNameState() {
  return updateGrpoSubmitState().outputState;
}

function parsePathSafeAdapterNameField(input, updateState) {
  const adapterName = parseAdapterNameField(input);
  if (!isPathSafeAdapterDirectoryName(adapterName)) {
    if (typeof updateState === 'function') updateState();
    input.focus();
    throw new Error(pathSafeAdapterDirectoryNameMessage());
  }
  return adapterName;
}

function setTrainingSubmitBusy(form, busy, pendingLabel) {
  const submitButton = form.querySelector('button[type="submit"]');
  if (!submitButton) return;
  if (!submitButton.dataset.originalLabel) {
    submitButton.dataset.originalLabel = submitButton.textContent;
  }
  form.dataset.trainingBusy = busy ? 'true' : 'false';
  submitButton.disabled = busy;
  submitButton.textContent = busy ? pendingLabel : submitButton.dataset.originalLabel;
  if (!busy) {
    if (form.id === 'sft-form') updateSftSubmitState();
    if (form.id === 'grpo-form') updateGrpoSubmitState();
    if (form.id === 'opd-form') updateOpdSubmitState();
  }
}

function toastTrainingSubmission(res, fallback) {
  const seed = res?.effective_seed;
  const suffix = seed == null ? '' : ` · seed ${String(seed)}`;
  toast(`${res?.message || fallback}${suffix}`, 'ok');
}

function readAdapterSmokePrompts(form) {
  const raw = form.adapter_smoke_prompts?.value || '';
  return raw.split(/\r?\n/).map(prompt => prompt.trim()).filter(Boolean);
}

// --- SFT Form ---
document.getElementById('sft-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    requireTrainingOptimizerAdmission('sft', form.optimizer.value, form.rank.value, 'SFT');
    const outputName = parsePathSafeAdapterNameField(form.output_name, updateSftOutputNameState);
    const learningRate = parseOptionalFiniteNumberField(form.learning_rate.value, 'SFT learning rate');
    const epochs = parsePositiveIntegerField(form.epochs.value, 'SFT epochs');
    const rank = parsePositiveIntegerField(form.rank.value, 'SFT LoRA rank');
    const checkpointInterval = parseOptionalPositiveIntegerField(form.checkpoint_interval.value, 'SFT checkpoint interval');
    const resumeCheckpoint = parseResumeCheckpointField(form.resume_checkpoint.value, 'SFT resume checkpoint');
    const invalidRowPolicy = form.invalid_row_policy.value === 'skip' ? 'skip' : 'fail';
    const config = {
      training_profile: 'native_online_lora_v1',
      output_name: outputName,
      auto_load: form.auto_load.checked,
      epochs,
      lora_rank: rank,
      // Paired explicitly: the server's default alpha (32) over the form's
      // default rank (8) trips the trainer's alpha/rank safety gate.
      lora_alpha: loraAlphaFor(rank),
      optimizer: readTrainingOptimizer('sft'),
      invalid_row_policy: invalidRowPolicy,
    };
    // Blank lr is omitted so the server resolves the per-optimizer default.
    if (learningRate !== null) config.learning_rate = learningRate;
    if (checkpointInterval !== null) config.checkpoint_interval = checkpointInterval;
    if (resumeCheckpoint !== null) config.resume_checkpoint = resumeCheckpoint;
    if (form.detect_anomaly.checked) config.detect_anomaly = true;
    const smokePrompts = readAdapterSmokePrompts(form);
    if (form.adapter_smoke_test.checked || smokePrompts.length) config.adapter_smoke_test = true;
    if (smokePrompts.length) config.adapter_smoke_prompts = smokePrompts;
    const held = trainingData.sft;
    let body;
    if (held && held.datasetName) {
      // Server-side dataset reference: the server reads its own copy — no rows
      // travel in the request and nothing is truncated.
      body = { dataset: held.datasetName, dataset_split: held.split || 'train', config };
    } else {
      let examples;
      if (held && held.items && held.items.length) {
        examples = held.items;
      } else {
        // Paste path accepts a JSON array OR JSONL — exactly what the help says.
        examples = parseTrainingText(form.examples.value);
        if (!examples.length) throw new Error('SFT examples cannot be empty. Drop a file, pick a dataset, paste JSON, or try a sample.');
      }
      if (invalidRowPolicy === 'fail') validateSftExamples(examples);
      body = { examples, config };
    }
    const postEval = provePostEval('sft');
    if (postEval) body.post_eval = postEval;
    setTrainingSubmitBusy(form, true, 'Submitting SFT…');
    const res = await api('/v1/train/sft', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toastTrainingSubmission(res, 'SFT job submitted');
    // Switch to queue tab
    document.querySelector('[data-tab="queue"]').click();
    pollTraining();
  } catch (e) { toast(e.message, 'err'); }
  finally { setTrainingSubmitBusy(form, false, 'Submitting SFT…'); }
});

// --- GRPO Form ---
document.getElementById('grpo-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    requireTrainingOptimizerAdmission('grpo', form.optimizer.value, form.rank.value, 'GRPO');
    const outputName = parsePathSafeAdapterNameField(form.output_name, updateGrpoOutputNameState);
    const learningRate = parseOptionalFiniteNumberField(form.learning_rate.value, 'GRPO learning rate');
    const klCoeff = parseFiniteNumberField(form.kl_coeff.value, 'GRPO KL coefficient');
    const rank = parsePositiveIntegerField(form.rank.value, 'GRPO LoRA rank');
    const checkpointInterval = parseOptionalPositiveIntegerField(form.checkpoint_interval.value, 'GRPO checkpoint interval');
    const resumeCheckpoint = parseResumeCheckpointField(form.resume_checkpoint.value, 'GRPO resume checkpoint');
    const config = {
      output_name: outputName,
      auto_load: form.auto_load.checked,
      kl_coeff: klCoeff,
      lora_rank: rank,
      // Paired explicitly: the server's default alpha (32) over the form's
      // default rank (8) trips the trainer's alpha/rank safety gate.
      lora_alpha: loraAlphaFor(rank),
      optimizer: readTrainingOptimizer('grpo'),
    };
    // Blank lr is omitted so the server resolves the per-optimizer default.
    if (learningRate !== null) config.learning_rate = learningRate;
    if (checkpointInterval !== null) config.checkpoint_interval = checkpointInterval;
    if (resumeCheckpoint !== null) config.resume_checkpoint = resumeCheckpoint;
    if (form.detect_anomaly.checked) config.detect_anomaly = true;
    const smokePrompts = readAdapterSmokePrompts(form);
    if (form.adapter_smoke_test.checked || smokePrompts.length) config.adapter_smoke_test = true;
    if (smokePrompts.length) config.adapter_smoke_prompts = smokePrompts;
    config.shared_prefix_reference = form.shared_prefix_reference.checked;
    const held = trainingData.grpo;
    let body;
    if (held && held.datasetName) {
      body = { dataset: held.datasetName, dataset_split: held.split || 'train', config };
    } else {
      let groups;
      if (held && held.items && held.items.length) {
        groups = held.items;
      } else {
        groups = parseTrainingText(form.groups.value);
        if (!groups.length) throw new Error('GRPO groups cannot be empty. Drop a file, pick a dataset, paste JSON, or try a sample.');
      }
      validateGrpoGroups(groups);
      body = { groups, config };
    }
    const postEval = provePostEval('grpo');
    if (postEval) body.post_eval = postEval;
    setTrainingSubmitBusy(form, true, 'Submitting GRPO…');
    const res = await api('/v1/train/grpo', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toastTrainingSubmission(res, 'GRPO job submitted');
    document.querySelector('[data-tab="queue"]').click();
    pollTraining();
  } catch (e) { toast(e.message, 'err'); }
  finally { setTrainingSubmitBusy(form, false, 'Submitting GRPO…'); }
});
