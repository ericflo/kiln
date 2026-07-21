
/* =====================================================================
   Adapter cards renderer + drill-in modal
   ===================================================================== */

let adaptersFilter = '';
// The Saved-adapters list is where you pick what to load — so surface each
// adapter's latest eval score (from the already-polled evalJobsCache) as a chip,
// turning a file browser into a glanceable leaderboard. No new endpoint.
function adapterEvalChip(name) {
  const jobs = (typeof evalJobsCache !== 'undefined' ? evalJobsCache : []) || [];
  const done = jobs.filter(j => (j.state || '').toLowerCase() === 'completed'
    && Array.isArray(j.adapters) && j.adapters.includes(name));
  if (!done.length) return `<span class="adapter-eval-chip none" title="No completed eval for this adapter yet — Run eval… below">not evaluated</span>`;
  done.sort((a, b) => String(b.submitted_at_iso || '').localeCompare(String(a.submitted_at_iso || '')));
  const j = done[0];
  let acc = null;
  const run = (j.finished_runs || []).find(r => r.adapter === name);
  if (run && typeof run.metrics?.accuracy === 'number') acc = run.metrics.accuracy;
  else if (typeof j.headline_accuracy === 'number' && (j.adapters || []).filter(a => a != null).length === 1) acc = j.headline_accuracy;
  if (acc == null) return `<span class="adapter-eval-chip none" title="Eval completed but no per-adapter accuracy recorded">not evaluated</span>`;
  const pct = (acc * 100).toFixed(0);
  return `<span class="adapter-eval-chip" title="${escapeHtml(j.suite_name || 'eval')}: ${pct}% accuracy (latest completed eval)">${escapeHtml(j.suite_name || 'eval')} <strong>${pct}%</strong></span>`;
}

// The strongest signal for "is the loaded adapter actually better than base?":
// the newest completed COMPARE eval (base run + this adapter's run). Returns
// { delta, suite } in accuracy points, or null. Powers the active-card verdict.
// Two-sided exact binomial sign test over discordant flips — mirrors
// kiln-eval's SignTest so the dashboard verdicts use the same math as
// the CLI. p=1 when there are no discordant examples.
// One decision threshold for EVERY surface that turns a compare eval into a
// win/loss claim (adapter card, job-card badge, completion toast, flywheel
// ribbon). §8.7's promise is "promotion is gated on a paired sign test" — a
// verdict colored green at p >= alpha anywhere breaks that promise.
const SIGN_TEST_ALPHA = 0.05;
// Shared p-value formatting so every surface prints the same string.
function fmtSignTestP(p) { return p < 0.005 ? 'p<0.01' : 'p=' + p.toFixed(2); }
function signTestP(improved, regressed) {
  const n = improved + regressed;
  if (n === 0) return 1.0;
  const k = Math.min(improved, regressed);
  let lnC = 0;            // ln C(n, 0)
  let lnTerms = [];
  for (let i = 0; i <= k; i++) {
    lnTerms.push(lnC - n * Math.LN2);
    lnC += Math.log(n - i) - Math.log(i + 1);
  }
  const max = Math.max(...lnTerms);
  const tail = lnTerms.reduce((acc, t) => acc + Math.exp(t - max), 0) * Math.exp(max);
  return Math.min(2 * tail, 1.0);
}
// Paired pass/fail flips over independent examples after the suite's declared
// completion reduction.
function compareFlips(baseRun, adapterRun) {
  const verdictOf = (run) => {
    const m = new Map();
    for (const o of run.aggregated_outcomes || []) m.set(o.example_id, o.kind === 'pass');
    return m;
  };
  const b = verdictOf(baseRun), a = verdictOf(adapterRun);
  let improved = 0, regressed = 0;
  for (const [id, basePass] of b) {
    if (!a.has(id)) continue;
    const adapterPass = a.get(id);
    if (!basePass && adapterPass) improved++;
    else if (basePass && !adapterPass) regressed++;
  }
  return { improved, regressed };
}
// The one gate, shared by all surfaces: pair every candidate run against the
// base run and attach the paired sign test, so "beats base" can only ever be
// claimed at p < SIGN_TEST_ALPHA. Returns one verdict per candidate — never a
// best-of-N pick (selecting the max of N noisy deltas is itself a bias) — or
// [] when there is no base/candidate accuracy pair.
function gatedCompareVerdicts(runs) {
  if (!Array.isArray(runs) || runs.length < 2) return [];
  const base = runs.find(r => r.adapter == null || r.adapter === 'base');
  if (!base || typeof base.metrics?.accuracy !== 'number') return [];
  return runs
    .filter(r => r.adapter != null && r.adapter !== 'base' && typeof r.metrics?.accuracy === 'number')
    .map(run => {
      const flips = compareFlips(base, run);
      const p = signTestP(flips.improved, flips.regressed);
      return {
        candidate: run.adapter,
        delta: Math.round((run.metrics.accuracy - base.metrics.accuracy) * 1000) / 10,
        accuracy: run.metrics.accuracy,
        baseAccuracy: base.metrics.accuracy,
        improved: flips.improved,
        regressed: flips.regressed,
        p,
        significant: p < SIGN_TEST_ALPHA,
      };
    });
}
function adapterCompareVerdict(name) {
  const jobs = ((typeof evalJobsCache !== 'undefined' ? evalJobsCache : []) || [])
    .filter(j => (j.state || '').toLowerCase() === 'completed' && Array.isArray(j.finished_runs)
      && j.finished_runs.length >= 2 && Array.isArray(j.adapters) && j.adapters.includes(name));
  jobs.sort((a, b) => String(b.submitted_at_iso || '').localeCompare(String(a.submitted_at_iso || '')));
  for (const j of jobs) {
    const v = gatedCompareVerdicts(j.finished_runs).find(x => x.candidate === name);
    if (v) return { ...v, suite: j.suite_name };
  }
  return null;
}
function verdictDeltaHtml(v) {
  if (!v) return '';
  // A green/red verdict is a claim — gate it on the paired sign test so
  // a 2-example wobble doesn't render as "beats base".
  const significant = v.significant === true;
  const detail = typeof v.p === 'number'
    ? ` — sign test improved ${v.improved} / regressed ${v.regressed}, ${fmtSignTestP(v.p)}`
    : '';
  if (!significant && Math.abs(v.delta) > 0.5) {
    const pTxt = typeof v.p === 'number' ? ` (${fmtSignTestP(v.p)})` : '';
    return `<span class="delta-badge delta-flat" title="vs base on ${escapeHtml(v.suite || 'eval')}${detail}">${v.delta > 0 ? '+' : ''}${v.delta.toFixed(1)} pts — not enough evidence${pTxt}</span>`;
  }
  const cls = v.delta > 0.5 ? 'delta-up' : (v.delta < -0.5 ? 'delta-down' : 'delta-flat');
  const label = cls === 'delta-flat' ? 'matches base' : `${v.delta > 0 ? '+' : ''}${v.delta.toFixed(1)} pts vs base`;
  return `<span class="delta-badge ${cls}" title="vs base on ${escapeHtml(v.suite || 'eval')}${detail}">${label}</span>`;
}

function renderAdaptersAsCards(data) {
  const panel = document.getElementById('adapters-panel');
  if (!panel) return;
  const adapters = data.available || [];
  const active = data.active || '';
  const q = adaptersFilter.trim().toLowerCase();
  const filtered = q ? adapters.filter(a => (a.name || '').toLowerCase().includes(q)) : adapters;
  if (!adapters.length) {
    panel.innerHTML = `<div class="eval-empty empty">
      <div class="eval-empty-icon"><svg class="icn"><use href="#i-layers"></use></svg></div>
      <div class="eval-empty-title">No adapters found yet.</div>
      <div class="eval-empty-body">An adapter is a small LoRA layer that personalizes the base model. Train your first from a JSONL of examples — drop the file on the Training page and you're one click away. New here? Read the <a href="https://ericflo.github.io/kiln/quickstart.html" target="_blank" rel="noopener">Quickstart</a> or the <a href="https://ericflo.github.io/kiln/troubleshooting.html" target="_blank" rel="noopener">Troubleshooting</a> guide.</div>
      <div style="display:flex; gap: var(--space-2); justify-content:center;">
        <button class="eval-empty-cta" type="button" data-train-first>Train your first adapter</button>
        <button class="btn btn-sm" type="button" data-focus-id="upload-name" style="align-self:center;">Or upload one</button>
      </div>
    </div>`;
    panel.querySelector('[data-train-first]')?.addEventListener('click', () => {
      selectPage('training');
      document.getElementById('training-tab-sft')?.click();
    });
    panel.querySelector('[data-focus-id]')?.addEventListener('click', ev => {
      const id = ev.currentTarget.getAttribute('data-focus-id');
      if (id) document.getElementById(id)?.focus();
    });
    return;
  }
  if (q && filtered.length === 0) {
    panel.innerHTML = `<div class="eval-empty"><div class="eval-empty-body">No adapters match <code>${escapeHtml(q)}</code>.</div></div>`;
    return;
  }
  // Active card first — the one serving pi is the one you came to check.
  const ordered = [...filtered].sort((a, b) => (b.name === active) - (a.name === active));
  const cards = ordered.map(a => {
    const isActive = a.name === active;
    return `<div class="adapter-card adapter-item ${isActive ? 'adapter-card-active' : ''}" data-adapter-name="${escapeHtml(a.name)}">
      ${isActive ? '<span class="adapter-card-active-pill">active</span>' : ''}
      <div class="adapter-card-name adapter-name">${escapeHtml(a.name)}</div>
      <div class="adapter-card-meta">
        <span><span class="tabular-nums">${fmtBytes(a.size_bytes)}</span> on disk</span>
        ${a.modified_at ? `<span title="modified ${escapeHtml(a.modified_at)}">${escapeHtml(fmtSmartTime(Date.parse(a.modified_at)))}</span>` : ''}
        ${a.files ? `<span class="tabular-nums">${a.files.length} file${a.files.length === 1 ? '' : 's'}</span>` : ''}
        ${adapterEvalChip(a.name)}
        ${isActive ? verdictDeltaHtml(adapterCompareVerdict(a.name)) : ''}
      </div>
      <div class="adapter-card-actions">
        ${isActive
          ? `<button class="btn btn-sm" type="button" data-adapter-action="unload" title="Stop serving this adapter — requests fall back to the base model">Unload (use base)</button>`
          : `<button class="btn btn-sm btn-primary" type="button" data-adapter-action="load" title="Hot-swap this adapter in — pi's next request uses it, no restart">Make active</button>`}
        <button class="btn btn-sm" type="button" data-adapter-action="eval" title="Grade this adapter on an eval suite — compare it against base">Run eval…</button>
        <button class="btn btn-sm" type="button" data-adapter-action="download">Download</button>
        <button class="btn btn-sm" type="button" data-adapter-action="delete" title="Delete this adapter from disk" style="margin-left:auto;">Delete</button>
      </div>
    </div>`;
  }).join('');
  panel.innerHTML = `<div class="adapter-cards">${cards}</div>`;
  document.getElementById('adapters-card-eyebrow').textContent =
    adapters.length + ' adapter' + (adapters.length === 1 ? '' : 's') + (active ? ' · active: ' + active : ' · base model active');
  panel.querySelectorAll('.adapter-card').forEach(card => {
    const name = card.dataset.adapterName;
    card.addEventListener('click', ev => {
      // Skip the open-drill behaviour when an action button was clicked.
      if (ev.target.closest('[data-adapter-action]')) return;
      openAdapterDrillModal(name);
    });
    card.querySelectorAll('[data-adapter-action]').forEach(b => {
      b.addEventListener('click', async ev => {
        ev.stopPropagation();
        const action = b.dataset.adapterAction;
        try {
          if (action === 'load') {
            b.disabled = true; b.textContent = 'Swapping…';
            await api('/v1/adapters/load', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
            toast(name + ' is now serving — pi\'s next request uses it', 'ok');
            // Refresh BOTH sources of "active" so cards and flywheel agree.
            pollAdapters && pollAdapters();
            pollHealth && pollHealth();
          } else if (action === 'unload') {
            b.disabled = true; b.textContent = 'Unloading…';
            await api('/v1/adapters/unload', { method: 'POST' });
            toast('Adapter unloaded — requests now use the base model', 'ok');
            pollAdapters && pollAdapters();
            pollHealth && pollHealth();
          } else if (action === 'download') {
            window.location.href = '/v1/adapters/' + encodeURIComponent(name) + '/download';
          } else if (action === 'delete') {
            if (!confirm(`Delete adapter "${name}"? This cannot be undone.`)) return;
            await api('/v1/adapters/' + encodeURIComponent(name), { method: 'DELETE' });
            toast('Deleted adapter: ' + name, 'ok');
            pollAdapters && pollAdapters();
          } else if (action === 'eval') {
            // Prove THIS adapter: suite picker + compare-vs-base, scoped to it.
            openAdapterEvalModal(name);
          }
        } catch (e) {
          toast(action + ' failed: ' + e.message, 'err');
          // Repaint so an in-flight "Swapping…" button never sticks around.
          lastAdaptersKey = null;
          refreshAdapterCards && refreshAdapterCards();
        }
      });
    });
  });
}

// Adapter cards renderer. Driven off `lastAdapters` (populated by the
// original `pollAdapters`) so we never issue a second `/v1/adapters`
// request, and dedup'd on a content key so we don't re-paint the cards
// when nothing changed (which would destroy hover/focus state).
let lastAdaptersKey = null;
function refreshAdapterCards() {
  const d = lastAdapters;
  if (!d) return;
  // Include a signature of completed evals so the per-card eval-score chips
  // refresh when a job finishes (the dedup must not pin the cold-start render).
  const evalSig = ((typeof evalJobsCache !== 'undefined' ? evalJobsCache : []) || [])
    .filter(j => (j.state || '').toLowerCase() === 'completed')
    .map(j => j.job_id).join(',');
  const key = (d.active || '') + '|' + (d.available || [])
    .map(a => `${a.name}:${a.size_bytes}:${a.modified_at || ''}`)
    .join(',') + '|' + evalSig;
  if (key === lastAdaptersKey) return;
  lastAdaptersKey = key;
  renderAdaptersAsCards(d);
}
// Driven from `pollAdapters` end-of-success directly — no standalone
// interval, no first-render kick needed.

// Adapter drill-in modal state. `adapterDrillName` is the currently-
// viewed adapter; `adapterDrillIsActive` mirrors the server's `is_active`
// flag so the Load/Unload button doesn't read its own label as state.
let adapterDrillName = null;
let adapterDrillIsActive = false;

async function openAdapterDrillModal(name) {
  adapterDrillName = name;
  modalHashOnOpen('adapter', '#adapters/' + encodeURIComponent(name));
  adapterDrillIsActive = false;
  const adapterModal = document.getElementById('adapter-drill-modal');
  adapterModal.hidden = false;
  openModal(adapterModal, { onClose: userCloseAdapterDrillModal });
  document.getElementById('adapter-drill-title').textContent = name;
  document.getElementById('adapter-drill-meta').textContent = 'Loading…';
  document.getElementById('adapter-drill-content').innerHTML = '<div class="detail-empty">Loading…</div>';
  try {
    const d = await api('/v1/adapters/' + encodeURIComponent(name) + '/detail');
    adapterDrillIsActive = !!d.is_active;
    document.getElementById('adapter-drill-meta').innerHTML =
      `<span class="hint">${d.is_active ? 'ACTIVE · ' : ''}${fmtBytes(d.size_bytes)} · ${d.files.length} file${d.files.length === 1 ? '' : 's'}</span>`;
    const loadBtn = document.getElementById('adapter-drill-load');
    loadBtn.textContent = d.is_active ? 'Unload' : 'Load';
    loadBtn.classList.toggle('btn-primary', !d.is_active);
    const content = document.getElementById('adapter-drill-content');
    content.innerHTML = renderAdapterDrillBody(d);
    content.querySelectorAll('[data-eval-job]').forEach(row => {
      row.addEventListener('click', () => openDrillModal(row.dataset.evalJob));
    });
    content.querySelectorAll('[data-train-job]').forEach(row => {
      row.addEventListener('click', () => openTrainDrillModal(row.dataset.trainJob));
    });
    // Provenance receipt loads after the detail body renders — its failure
    // (404 is the normal case for uploaded/legacy adapters) must never take
    // the rest of the modal down, so it's a separate fire-and-forget fetch.
    loadAdapterReceipt(name);
  } catch (e) {
    document.getElementById('adapter-drill-content').innerHTML = `<div class="detail-empty">Failed to load: ${escapeHtml(e.message)}</div>`;
  }
}

/* ---- Adapter receipt (GET /v1/adapters/:name/receipt) -----------------
   The §8.11 reproducibility receipt (kiln-train/src/receipt.rs
   AdapterReceipt): training provenance — source kind, seed, teacher,
   prompt corpus, hyperparameters, run diagnostics, post-train eval
   scores. Fetched when the drill modal opens; 404 means no receipt.json
   on disk (uploaded or pre-receipt adapters) and renders as a quiet
   explanation; any other failure renders a one-line hint. */
async function loadAdapterReceipt(name) {
  // Re-resolve on every write: the modal may have switched to another
  // adapter (or been repainted) while this fetch was in flight.
  const section = () => (adapterDrillName === name
    ? document.getElementById('adapter-receipt-section')
    : null);
  let receipt;
  try {
    receipt = await api('/v1/adapters/' + encodeURIComponent(name) + '/receipt');
  } catch (e) {
    const el = section();
    if (!el) return;
    el.innerHTML = (e && e.status === 404)
      ? '<h4>Receipt</h4><div class="hint">No receipt — uploaded or legacy adapter. Adapters trained on this server ship a reproducibility receipt (<code>receipt.json</code>).</div>'
      : `<h4>Receipt</h4><div class="hint">Couldn't load receipt — ${escapeHtml((e && e.message) || 'request failed')}</div>`;
    return;
  }
  const el = section();
  if (!el) return;
  el.innerHTML = renderAdapterReceipt(receipt);
  el.querySelectorAll('[data-train-job]').forEach(row => {
    row.addEventListener('click', () => openTrainDrillModal(row.dataset.trainJob));
  });
  const rawBtn = el.querySelector('[data-receipt-raw]');
  const rawPre = el.querySelector('[data-receipt-raw-pre]');
  if (rawBtn && rawPre) {
    rawBtn.addEventListener('click', () => {
      rawPre.hidden = !rawPre.hidden;
      rawBtn.setAttribute('aria-expanded', String(!rawPre.hidden));
    });
  }
}

function renderAdapterReceipt(r) {
  const rows = [];
  const line = (label, html) => rows.push(`<div><span class="hint">${escapeHtml(label)}:</span> ${html}</div>`);
  if (r.source_kind) line('Trained via', `<code>${escapeHtml(String(r.source_kind))}</code>`);
  if (r.produced_at) {
    const t = Date.parse(r.produced_at);
    line('Produced', escapeHtml(isFinite(t) ? fmtSmartTime(t) : String(r.produced_at)));
  }
  if (r.kiln_version) line('Kiln version', `<code>${escapeHtml(String(r.kiln_version))}</code>`);
  if (r.seed != null) line('Seed', `<code>${escapeHtml(String(r.seed))}</code>`);
  // The receipt schema has no dedicated job-id field today, but when a
  // producer recorded one (top-level or inside the free-form
  // hyperparameters object) link it through to the train drill.
  const hp = (r.hyperparameters && typeof r.hyperparameters === 'object' && !Array.isArray(r.hyperparameters)) ? r.hyperparameters : null;
  const jobId = r.job_id || r.training_job_id || (hp && (hp.job_id || hp.training_job_id)) || null;
  if (jobId) line('Training job', `<a data-train-job="${escapeHtml(String(jobId))}" style="font-family:var(--font-mono); cursor:pointer;">${escapeHtml(String(jobId))}</a>`);
  if (r.teacher && r.teacher.alias) {
    const tid = r.teacher.model_id && r.teacher.model_id !== r.teacher.alias
      ? ` <span class="hint">(${escapeHtml(String(r.teacher.model_id))})</span>` : '';
    line('Teacher', `<code>${escapeHtml(String(r.teacher.alias))}</code>${tid}`);
  }
  if (r.prompts && r.prompts.source) {
    const count = typeof r.prompts.count === 'number' ? ` <span class="hint">· ${r.prompts.count} prompts</span>` : '';
    line('Dataset', `<code>${escapeHtml(String(r.prompts.source))}</code>${count}`);
  }
  const diag = r.diagnostic_summary || {};
  if (typeof diag.final_loss === 'number') line('Final loss', `<code>${diag.final_loss.toFixed(4)}</code>`);
  if (Array.isArray(diag.guardrail_triggers) && diag.guardrail_triggers.length) {
    line('Guardrails fired', escapeHtml(diag.guardrail_triggers.join(', ')));
  }
  if (r.post_eval && typeof r.post_eval === 'object') {
    const evals = Object.entries(r.post_eval).slice(0, 6)
      .map(([suite, score]) => `${escapeHtml(suite)} <code>${typeof score === 'number' ? score.toFixed(3) : escapeHtml(String(score))}</code>`);
    if (evals.length) line('Post-train evals', evals.join(' · '));
  }
  let hyperHtml = '';
  if (hp) {
    const chips = Object.entries(hp)
      .filter(([, v]) => v === null || ['number', 'string', 'boolean'].includes(typeof v))
      .slice(0, 12)
      .map(([k, v]) => `<span class="receipt-chip"><span class="hint">${escapeHtml(k)}</span> ${escapeHtml(v === null ? 'default' : String(v))}</span>`);
    if (chips.length) hyperHtml = `<div class="receipt-chips">${chips.join('')}</div>`;
  }
  return `<h4>Receipt</h4>
    <div style="display:flex; flex-direction:column; gap:4px; font-size:13px;">${rows.join('') || '<div class="hint">Receipt present, but it carries no provenance fields.</div>'}</div>
    ${hyperHtml}
    <div style="margin-top:8px;"><button class="btn btn-sm btn-ghost" type="button" data-receipt-raw aria-expanded="false">Raw JSON</button></div>
    <pre class="rc-raw" data-receipt-raw-pre hidden>${escapeHtml(JSON.stringify(r, null, 2))}</pre>`;
}

function renderAdapterDrillBody(d) {
  const filesHtml = d.files.map(f => `<div style="display:grid; grid-template-columns:1fr auto; gap:8px; padding:6px 0; border-bottom:1px solid var(--border); font-family:var(--font-mono); font-size:12px;">
    <span>${escapeHtml(f.name)}</span>
    <span class="tabular-nums hint">${fmtBytes(f.size_bytes)}</span>
  </div>`).join('') || '<div class="hint">No files.</div>';

  const trainHtml = d.training_jobs.length
    ? d.training_jobs.map(j => `<div class="eval-row" data-train-job="${escapeHtml(j.job_id)}" style="grid-template-columns:auto 1fr auto auto auto; cursor:pointer;">
        <span class="job-state-pill ${(j.state||'').toString().toLowerCase()}">${escapeHtml((j.state||'').toString())}</span>
        <span style="font-family:var(--font-mono);">${escapeHtml(j.job_id.slice(0,12))}</span>
        <span class="hint">${escapeHtml(j.job_type.toString())}</span>
        <span class="tabular-nums hint">${j.final_loss != null ? 'loss '+j.final_loss.toFixed(3) : '—'}</span>
        <span class="tabular-nums hint">${fmtDuration(j.elapsed_secs)}</span>
      </div>`).join('')
    : '<div class="hint">No training jobs have produced this adapter (yet). Submit one from the Training tab.</div>';

  const evalHtml = d.eval_jobs.length
    ? d.eval_jobs.map(j => `<div class="eval-row" data-eval-job="${escapeHtml(j.job_id)}" style="grid-template-columns:auto 1fr auto auto; cursor:pointer;">
        <span class="job-state-pill ${(j.state||'').toString()}">${escapeHtml((j.state||'').toString())}</span>
        <span><strong>${escapeHtml(j.suite_name)}</strong></span>
        <span class="tabular-nums" style="color:var(--text);">${j.accuracy != null ? (j.accuracy*100).toFixed(0)+'%' : '—'}</span>
        <span class="hint" style="font-family:var(--font-mono);">${escapeHtml(j.job_id.slice(0,8))}</span>
      </div>`).join('')
    : '<div class="hint">No evals against this adapter yet. Click "Run eval…" above.</div>';

  // Lineage section: surface the on-disk lineage.json fields when present
  // so users can see base model + Kiln build + created_at without
  // opening the file. Falls back gracefully when the adapter was
  // uploaded or pre-dates the lineage format.
  let lineageHtml = '';
  if (d.lineage && typeof d.lineage === 'object') {
    const lin = d.lineage;
    const rows = [];
    if (lin.base_model && lin.base_model.id) {
      rows.push(`<div><span class="hint">Base model:</span> <code>${escapeHtml(lin.base_model.id)}</code></div>`);
    }
    if (lin.created_at) {
      const t = Date.parse(lin.created_at);
      rows.push(`<div><span class="hint">Created:</span> ${escapeHtml(isFinite(t) ? fmtSmartTime(t) : lin.created_at)} <span class="hint" title="${escapeHtml(lin.created_at)}">(${escapeHtml(lin.created_at.split('T')[0])})</span></div>`);
    }
    if (lin.kiln_commit) {
      rows.push(`<div><span class="hint">Kiln build:</span> <code>${escapeHtml(lin.kiln_commit)}</code></div>`);
    }
    if (lin.replay_hash) {
      rows.push(`<div><span class="hint">Replay hash:</span> <code style="font-size:11px;">${escapeHtml(String(lin.replay_hash).slice(0, 16))}…</code></div>`);
    }
    if (rows.length) {
      lineageHtml = `<div class="detail-section">
        <h4>Lineage</h4>
        <div style="display:flex; flex-direction:column; gap:4px; font-size:13px;">${rows.join('')}</div>
      </div>`;
    }
  }

  return `<div style="padding: var(--space-4) var(--space-5); border-bottom:1px solid var(--border);">
    <div style="display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
      <div>
        <div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">Disk</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${fmtBytes(d.size_bytes)}</div>
      </div>
      <div>
        <div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">Files</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${d.files.length}</div>
      </div>
      <div>
        <div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">Training</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${d.training_jobs.length}</div>
      </div>
      <div>
        <div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">Evals</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${d.eval_jobs.length}</div>
      </div>
    </div>
  </div>
  ${lineageHtml}
  <div class="detail-section" id="adapter-receipt-section">
    <h4>Receipt</h4>
    <div class="hint">Loading receipt…</div>
  </div>
  <div class="detail-section">
    <h4>Eval history</h4>
    ${evalHtml}
  </div>
  <div class="detail-section">
    <h4>Training history</h4>
    ${trainHtml}
  </div>
  <div class="detail-section">
    <h4>Files on disk</h4>
    ${filesHtml}
  </div>`;
}

function closeAdapterDrillModal() {
  adapterDrillName = null;
  const adapterModal = document.getElementById('adapter-drill-modal');
  adapterModal.hidden = true;
  closeModal(adapterModal);
}
// User-initiated close: walk history per the deep-link state machine.
function userCloseAdapterDrillModal() {
  modalHashOnUserClose('adapter', '#adapters', closeAdapterDrillModal);
}
document.getElementById('adapter-drill-close')?.addEventListener('click', userCloseAdapterDrillModal);
document.getElementById('adapter-drill-modal')?.addEventListener('click', ev => {
  if (ev.target.id === 'adapter-drill-modal') userCloseAdapterDrillModal();
});
document.getElementById('adapter-drill-load')?.addEventListener('click', async () => {
  if (!adapterDrillName) return;
  const name = adapterDrillName;
  const isUnload = adapterDrillIsActive;
  try {
    if (isUnload) {
      await api('/v1/adapters/unload', { method: 'POST' });
      toast('Unloaded — base model active', 'ok');
    } else {
      await api('/v1/adapters/load', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
      toast('Loaded ' + name, 'ok');
    }
    userCloseAdapterDrillModal();
    pollAdapters && pollAdapters();
  } catch (e) { toast(e.message, 'err'); }
});
/* Prove-an-adapter modal — scoped to the adapter the user actually clicked.
   Fixes the old dead-end where "Run eval…" dropped the adapter name and just
   navigated to the suites list. */
let adapterEvalName = null;
async function openAdapterEvalModal(name) {
  adapterEvalName = name;
  const modal = document.getElementById('adapter-eval-modal');
  if (!modal) return;
  setText('adapter-eval-name', name);
  const sel = document.getElementById('adapter-eval-suite');
  const help = document.getElementById('adapter-eval-suite-help');
  const go = document.getElementById('adapter-eval-compare');
  const solo = document.getElementById('adapter-eval-solo');
  let suites = [];
  try { const d = await api('/v1/eval/suites'); suites = d.suites || []; } catch (_) {}
  if (sel) {
    sel.innerHTML = suites.map(s => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.name)}${s.num_examples ? ' · ' + s.num_examples + ' examples' : ''}</option>`).join('');
    sel.disabled = !suites.length;
  }
  if (help) help.hidden = suites.length > 0;
  if (go) go.disabled = !suites.length;
  if (solo) solo.disabled = !suites.length;
  modal.hidden = false;
  openModal(modal, { onClose: closeAdapterEvalModal });
  if (sel && suites.length) sel.focus();
}
function closeAdapterEvalModal() {
  const modal = document.getElementById('adapter-eval-modal');
  if (!modal) return;
  modal.hidden = true;
  closeModal(modal);
}
async function submitAdapterEval(compare) {
  const suite = document.getElementById('adapter-eval-suite')?.value;
  const name = adapterEvalName;
  if (!suite || name == null) return;
  const btn = document.getElementById(compare ? 'adapter-eval-compare' : 'adapter-eval-solo');
  if (btn) btn.disabled = true;
  try {
    const res = compare
      ? await api('/v1/eval/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ suite, adapters: ['', name] }) })
      : await api('/v1/eval/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ suite, adapter: name }) });
    toast(`Queued ${compare ? 'compare' : 'eval'} ${res.job_id.slice(0, 8)} · seed ${res.effective_seed}`, 'ok');
    closeAdapterEvalModal();
    selectPage('evals');
    document.getElementById('evals-tab-jobs')?.click();
    if (typeof refreshEvalJobs === 'function') refreshEvalJobs();
    toast(compare
      ? `Comparing ${name} vs base on ${suite} — the verdict shows here when it finishes`
      : `Scoring ${name} on ${suite} — results show here when it finishes`, 'ok');
  } catch (e) { toast('Could not queue eval: ' + e.message, 'err'); }
  finally { if (btn) btn.disabled = false; }
}
document.getElementById('adapter-eval-goto-datasets')?.addEventListener('click', () => {
  closeAdapterEvalModal();
  selectPage('evals');
  document.getElementById('evals-tab-datasets')?.click();
  setTimeout(() => document.getElementById('dataset-name')?.focus(), 120);
});
document.getElementById('adapter-eval-close')?.addEventListener('click', closeAdapterEvalModal);
document.getElementById('adapter-eval-modal')?.addEventListener('click', ev => { if (ev.target.id === 'adapter-eval-modal') closeAdapterEvalModal(); });
document.getElementById('adapter-eval-compare')?.addEventListener('click', () => submitAdapterEval(true));
document.getElementById('adapter-eval-solo')?.addEventListener('click', () => submitAdapterEval(false));
// Escape is handled by the shared modal manager (closeAdapterEvalModal is
// the layer's onClose).

document.getElementById('adapter-drill-eval')?.addEventListener('click', () => {
  const name = adapterDrillName;
  userCloseAdapterDrillModal();
  if (name) openAdapterEvalModal(name);
});
