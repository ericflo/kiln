
// --- Agent traces (pi sessions → distillation source) ----------------
// The Distill → Agent traces tab: every pi session saved on this machine,
// browsable before you distill from it. Entering the tab lists the
// existing index; the scan button rebuilds it (optionally from a custom
// sessions folder, persisted in localStorage). Outcome chips and a
// working-dir filter narrow the list client-side — the index rows carry
// the full §10.3 outcome heuristics — and every card drills into the
// recorded conversation at #distill/traces/{id}.
const TRACES_SCAN_PATH_KEY = 'kiln.traces.scanPath';
let agentTracesCache = null;   // last fetched index; null = never loaded
let agentTracesScanNote = '';  // headline HTML after an explicit scan
let agentTraceOutcomeFilter = 'all';

// Outcome buckets for the filter chips, derived from the heuristics the
// index actually carries: last bash exit code, /tree forks, follow-up
// attempts, and user-edited agent files. A trace can land in several.
function agentTraceOutcomeBuckets(t) {
  const buckets = [];
  if (t.outcome?.ended_with_exit_0 === true) buckets.push('exit0');
  if (t.outcome?.ended_with_exit_0 === false) buckets.push('exitnz');
  if (t.forked || t.outcome?.has_followup_attempt === true || (t.outcome?.user_edited_agent_files || []).length > 0) buckets.push('sideways');
  if (buckets.length === 0) buckets.push('nosignal');
  return buckets;
}

// Human-readable outcome summary shared by the cards and the drill modal.
function agentTraceOutcomeBits(t) {
  const bits = [];
  if (t.outcome?.ended_with_exit_0 === true) bits.push('exit 0');
  if (t.outcome?.ended_with_exit_0 === false) bits.push('exit ≠ 0');
  const edited = (t.outcome?.user_edited_agent_files || []).length;
  if (edited) bits.push(`${edited} user-edited file${edited === 1 ? '' : 's'}`);
  if (t.outcome?.has_followup_attempt === true) bits.push('has follow-up');
  if (t.forked) bits.push('forked');
  return bits;
}

// List the existing index (no rescan) — fired on tab entry so the tab is
// useful without touching the scan button.
async function refreshAgentTraces() {
  const node = document.getElementById('agent-traces-list');
  if (!node) return;
  try {
    const list = await api('/v1/agent/traces');
    agentTracesCache = list.traces || [];
    renderAgentTracesList();
  } catch (e) {
    setListHtml(node, 'err:' + e.message, `<div class="empty">Couldn't load pi sessions: ${escapeHtml(e.message)}</div>`);
    setListHtml(document.getElementById('agent-traces-chips'), 'err', '');
  }
}

document.getElementById('agent-traces-refresh')?.addEventListener('click', async () => {
  const node = document.getElementById('agent-traces-list');
  if (!node) return;
  const customPath = (document.getElementById('agent-traces-path')?.value || '').trim();
  // Remember the last-used folder (empty = pi's default) for next visit.
  try { localStorage.setItem(TRACES_SCAN_PATH_KEY, customPath); } catch {}
  setListHtml(node, 'scanning', '<div class="empty">Scanning for pi sessions…</div>');
  try {
    // Rescan first (rebuilds the index), then list what it indexed. An
    // omitted path means the server scans pi's default sessions folder.
    const discover = await api('/v1/agent/traces/discover', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(customPath ? { path: customPath } : {}),
    });
    agentTracesScanNote = `Indexed ${discover.indexed} pi session${discover.indexed === 1 ? '' : 's'} from <code>${escapeHtml(discover.path || '')}</code>.`;
    const list = await api('/v1/agent/traces');
    agentTracesCache = list.traces || [];
    renderAgentTracesList();
  } catch (e) {
    agentTracesScanNote = '';
    setListHtml(node, 'scanerr:' + e.message, `<div class="empty">Scan failed: ${escapeHtml(e.message)}</div>`);
  }
});

// Restore the last-used scan path (empty = server default).
try {
  const savedScanPath = localStorage.getItem(TRACES_SCAN_PATH_KEY);
  const scanPathInput = document.getElementById('agent-traces-path');
  if (savedScanPath && scanPathInput) scanPathInput.value = savedScanPath;
} catch {}

document.getElementById('agent-traces-dir')?.addEventListener('input', () => renderAgentTracesList());

function renderAgentTracesList() {
  const node = document.getElementById('agent-traces-list');
  const chipsNode = document.getElementById('agent-traces-chips');
  if (!node) return;
  const all = agentTracesCache || [];
  const dirNeedle = (document.getElementById('agent-traces-dir')?.value || '').trim().toLowerCase();

  const counts = { all: all.length, exit0: 0, exitnz: 0, sideways: 0, nosignal: 0 };
  for (const t of all) for (const b of agentTraceOutcomeBuckets(t)) counts[b] += 1;
  // A rescan can empty the active bucket; degrade to All instead of
  // pinning the list on a filter that now matches nothing.
  if (agentTraceOutcomeFilter !== 'all' && counts[agentTraceOutcomeFilter] === 0) agentTraceOutcomeFilter = 'all';

  // Outcome chips — same pattern as the recent-requests client chips.
  const chip = (key, label, n, title) =>
    `<button type="button" class="agent-chip${agentTraceOutcomeFilter === key ? ' active' : ''}" data-trace-chip="${key}" title="${escapeHtml(title)}">${escapeHtml(label)}<span class="count">${n}</span></button>`;
  const chipsHtml = all.length === 0 ? '' : `<div class="agent-chips" role="group" aria-label="Filter pi sessions by outcome" style="margin-bottom:0;">`
    + chip('all', 'All sessions', counts.all, 'Every indexed pi session')
    + chip('exit0', 'exit 0', counts.exit0, 'Sessions whose last shell command exited 0 — the likely successes worth distilling')
    + chip('exitnz', 'exit ≠ 0', counts.exitnz, 'Sessions whose last shell command failed')
    + chip('sideways', 'went sideways', counts.sideways, 'Forked with /tree, retried in a follow-up session, or hand-edited afterwards — signs the original branch went wrong')
    + chip('nosignal', 'no signal', counts.nosignal, 'Sessions with no outcome heuristics extracted')
    + '</div>';
  if (chipsNode && setListHtml(chipsNode, 'chips:' + JSON.stringify([agentTraceOutcomeFilter, counts]), chipsHtml)) {
    chipsNode.querySelectorAll('[data-trace-chip]').forEach(c => c.addEventListener('click', () => {
      agentTraceOutcomeFilter = c.dataset.traceChip;
      renderAgentTracesList();
    }));
  }

  if (agentTracesCache === null) return; // first load pending — keep the static hint
  const noteHtml = agentTracesScanNote ? `<div class="form-help" style="margin-bottom: var(--space-3);">${agentTracesScanNote}</div>` : '';
  if (all.length === 0) {
    setListHtml(node, 'empty:' + agentTracesScanNote,
      noteHtml + '<div class="empty">No pi sessions found yet. Use pi against this server, then scan again — every session it saves becomes distillable here.</div>');
    return;
  }

  const filtered = all.filter(t => {
    if (agentTraceOutcomeFilter !== 'all' && !agentTraceOutcomeBuckets(t).includes(agentTraceOutcomeFilter)) return false;
    if (dirNeedle && !String(t.working_dir || '').toLowerCase().includes(dirNeedle)) return false;
    return true;
  });
  if (filtered.length === 0) {
    setListHtml(node, 'nomatch:' + JSON.stringify([agentTraceOutcomeFilter, dirNeedle, agentTracesScanNote]),
      noteHtml + '<div class="empty">No pi sessions match the current filters.</div>');
    return;
  }

  const listKey = 'list:' + JSON.stringify([agentTraceOutcomeFilter, dirNeedle, agentTracesScanNote,
    filtered.map(t => [t.id, t.num_turns, t.num_tool_calls, t.last_event_at])]);
  const cards = filtered.map(t => {
    const bits = agentTraceOutcomeBits(t);
    const when = t.last_event_at || t.first_event_at || '';
    return `<button type="button" class="adapter-card" data-trace-open="${escapeHtml(t.id || '')}" style="display:block; width:100%; text-align:left; font:inherit; color:inherit; margin-bottom:var(--space-2);" title="Open this pi session — read the conversation and tool calls before distilling from it">
      <div style="display:flex; justify-content:space-between; gap:var(--space-3); align-items:baseline; flex-wrap:wrap;">
        <span style="font-weight:600; font-family:var(--font-mono); font-size:var(--text-xs);">${escapeHtml(t.id || '?')}</span>
        ${when ? `<span style="font-size:var(--text-2xs); color:var(--text-muted);">${escapeHtml(when)}</span>` : ''}
      </div>
      <div style="font-size:var(--text-xs); color:var(--text-muted);">${t.num_turns || 0} turns · ${t.num_tool_calls || 0} tool calls · ${escapeHtml(t.working_dir || '')}</div>
      ${bits.length ? `<div style="font-size:var(--text-2xs); color:var(--text-muted); margin-top:var(--space-1);">${bits.map(b => escapeHtml(b)).join(' · ')}</div>` : ''}
    </button>`;
  }).join('');
  if (setListHtml(node, listKey, noteHtml + cards)) {
    node.querySelectorAll('[data-trace-open]').forEach(btn => {
      btn.addEventListener('click', () => openTraceDrillModal(btn.dataset.traceOpen));
    });
  }
}

/* =====================================================================
   pi session trace drill-in modal — the recorded conversation: turns,
   tool calls, outcome heuristics. Read it before you distill from it.
   ===================================================================== */
let traceDrillId = null;
let traceDrillData = null;
// Full text per clamped block in the current drill render, keyed by the
// data-trace-clamp attribute; the Show-all buttons swap it in on demand.
let traceDrillTexts = new Map();
const TRACE_CLAMP_CHARS = 700;

// Trace ids are pi session ids — stable UUID-like file stems — so they
// ride the #distill/traces/{id} deep-link grammar like the other drills.
async function openTraceDrillModal(id) {
  traceDrillId = id;
  modalHashOnOpen('trace', '#distill/traces/' + encodeURIComponent(id));
  const modal = document.getElementById('trace-drill-modal');
  if (!modal) return;
  modal.hidden = false;
  openModal(modal, { onClose: userCloseTraceDrillModal });
  document.getElementById('trace-drill-title').textContent = 'pi session';
  document.getElementById('trace-drill-meta').textContent = id;
  const content = document.getElementById('trace-drill-content');
  content.innerHTML = '<div class="detail-empty">Loading…</div>';
  traceDrillData = null;
  try {
    const t = await api('/v1/agent/traces/' + encodeURIComponent(id));
    if (traceDrillId !== id) return; // closed or re-targeted while fetching
    traceDrillData = t;
    document.getElementById('trace-drill-title').textContent = `pi session ${String(t.id || id).slice(0, 8)}`;
    const metaBits = [`${t.num_turns || 0} turns`, `${t.num_tool_calls || 0} tool calls`];
    const outcomeBits = agentTraceOutcomeBits(t);
    if (outcomeBits.length) metaBits.push(outcomeBits.join(' · '));
    document.getElementById('trace-drill-meta').textContent = metaBits.join(' · ');
    content.innerHTML = renderTraceDrillBody(t);
    content.querySelectorAll('[data-trace-expand]').forEach(btn => {
      btn.addEventListener('click', () => {
        const pre = content.querySelector(`pre[data-trace-clamp="${btn.dataset.traceExpand}"]`);
        const full = traceDrillTexts.get(btn.dataset.traceExpand);
        if (pre && full != null) { pre.textContent = full; btn.remove(); }
      });
    });
  } catch (e) {
    if (traceDrillId !== id) return;
    content.innerHTML = `<div class="detail-empty">Couldn't load this pi session: ${escapeHtml(e.message)}</div>`;
  }
}

function closeTraceDrillModal() {
  traceDrillId = null;
  traceDrillData = null;
  traceDrillTexts = new Map();
  const modal = document.getElementById('trace-drill-modal');
  if (!modal) return;
  modal.hidden = true;
  closeModal(modal);
}
// User-initiated close (X / backdrop / Esc): walk history per the
// deep-link state machine, exactly like the other drills.
function userCloseTraceDrillModal() {
  modalHashOnUserClose('trace', '#distill/traces', closeTraceDrillModal);
}
document.getElementById('trace-drill-close')?.addEventListener('click', userCloseTraceDrillModal);
document.getElementById('trace-drill-modal')?.addEventListener('click', ev => {
  if (ev.target.id === 'trace-drill-modal') userCloseTraceDrillModal();
});
// Raw JSON toggle — same pattern as the other drill modals' `raw` buttons.
document.getElementById('trace-drill-raw')?.addEventListener('click', () => {
  if (!traceDrillData) return;
  const content = document.getElementById('trace-drill-content');
  if (!content) return;
  const existing = content.querySelector('#trace-drill-raw-block');
  if (existing) { existing.remove(); return; }
  const pre = document.createElement('pre');
  pre.id = 'trace-drill-raw-block';
  pre.className = 'req-pre';
  pre.style.cssText = 'max-height:50vh; margin:var(--space-4) var(--space-5);';
  pre.textContent = JSON.stringify(traceDrillData, null, 2);
  content.appendChild(pre);
  pre.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

// Split an assistant action segment into displayable pieces: <think>
// blocks, <tool_call>{json}</tool_call> blocks (the form the trace
// normalizer emits), and the plain-text runs between them.
function traceSegmentPieces(content) {
  const pieces = [];
  const re = /<think>([\s\S]*?)<\/think>|<tool_call>([\s\S]*?)<\/tool_call>/g;
  let last = 0;
  let m;
  while ((m = re.exec(content)) !== null) {
    if (m.index > last) pieces.push({ kind: 'text', text: content.slice(last, m.index) });
    if (m[1] !== undefined) pieces.push({ kind: 'think', text: m[1] });
    else pieces.push({ kind: 'tool_call', text: m[2] });
    last = re.lastIndex;
  }
  if (last < content.length) pieces.push({ kind: 'text', text: content.slice(last) });
  return pieces.filter(p => p.kind === 'tool_call' || p.text.trim().length > 0);
}

function renderTraceDrillBody(t) {
  traceDrillTexts = new Map();
  let clampSeq = 0;
  // Long content is clamped with a Show-all expander so a 200-line tool
  // result doesn't bury the conversation.
  const clamped = (text) => {
    const full = String(text ?? '');
    if (full.length <= TRACE_CLAMP_CHARS) return `<pre class="req-pre">${escapeHtml(full)}</pre>`;
    const key = 'seg' + (clampSeq++);
    traceDrillTexts.set(key, full);
    return `<pre class="req-pre" data-trace-clamp="${key}">${escapeHtml(full.slice(0, TRACE_CLAMP_CHARS))}…</pre>
      <button type="button" class="btn btn-sm btn-ghost" data-trace-expand="${key}">Show all ${full.length.toLocaleString()} characters</button>`;
  };

  // Metadata header: where it ran, when, how big, how it ended.
  const stats = [
    ['Working dir', t.working_dir || '—'],
    ['Turns', String(t.num_turns || 0)],
    ['Tool calls', String(t.num_tool_calls || 0)],
    ['Started', t.first_event_at || '—'],
    ['Last event', t.last_event_at || '—'],
  ];
  if (t.parent_id) stats.push(['Forked from', t.parent_id]);
  const outcomeBits = agentTraceOutcomeBits(t);
  stats.push(['Outcome', outcomeBits.length ? outcomeBits.join(' · ') : 'no signal extracted']);
  const statRow = stats
    .map(([k, v]) => `<div class="req-stat"><span class="req-stat-k">${escapeHtml(k)}</span><span class="req-stat-v">${escapeHtml(v)}</span></div>`)
    .join('');

  const turnHtml = (role, kindLabel, bodyHtml) => `
    <div class="req-section">
      <div class="req-section-head">${escapeHtml(role)}${kindLabel ? ` <span class="hint" style="text-transform:none; letter-spacing:normal;">${escapeHtml(kindLabel)}</span>` : ''}</div>
      ${bodyHtml}
    </div>`;

  // Leading system/user context — the task the session started from.
  const promptHtml = (t.prompt_messages || [])
    .map(m => turnHtml(m.role || '?', 'task scaffold', clamped(m.content)))
    .join('');

  // The trajectory proper: actions (with tool calls broken out by name),
  // observations (tool results), and mid-session user/system context.
  const segHtml = (t.trajectory || []).map(seg => {
    const kind = seg.kind || 'context';
    const content = seg.content || '';
    if (seg.role === 'assistant' && /<tool_call>|<think>/.test(content)) {
      const piecesHtml = traceSegmentPieces(content).map(p => {
        if (p.kind === 'tool_call') {
          let name = '?';
          let argsText = p.text.trim();
          try {
            const parsed = JSON.parse(p.text);
            if (parsed && typeof parsed === 'object') {
              name = parsed.name || '?';
              argsText = JSON.stringify(parsed.arguments ?? {});
            }
          } catch { /* malformed call JSON — show it verbatim */ }
          return `<div style="border:1px solid var(--border); border-radius:var(--radius-sm); padding:var(--space-2) var(--space-3); margin:var(--space-1) 0; background:var(--surface);">
            <div style="font-size:var(--text-xs); color:var(--text-muted); margin-bottom:4px;">tool call · <strong style="font-family:var(--font-mono); color:var(--text);">${escapeHtml(name)}</strong></div>
            ${clamped(argsText)}
          </div>`;
        }
        if (p.kind === 'think') {
          return `<div style="margin:var(--space-1) 0;"><div style="font-size:var(--text-2xs); color:var(--text-muted); text-transform:uppercase; letter-spacing:var(--tracking-caps); margin-bottom:4px;">thinking</div>${clamped(p.text.trim())}</div>`;
        }
        return clamped(p.text.trim());
      }).join('');
      return turnHtml('assistant', null, piecesHtml);
    }
    const role = seg.role || '?';
    const label = kind === 'observation'
      ? `tool result${seg.tool_call_id ? ' · ' + seg.tool_call_id : ''}`
      : (kind === 'action' ? null : 'context');
    return turnHtml(role, label, clamped(content));
  }).join('');

  const conversationHtml = (promptHtml || segHtml)
    ? promptHtml + segHtml
    : '<div class="empty">This index entry predates turn-level capture — scan again to re-read the session with the current parser.</div>';

  return `<div class="req-detail">
    <div class="req-stats">${statRow}</div>
    ${conversationHtml}
  </div>`;
}

/* =====================================================================
   Agent runs — the embedded pi run engine (/v1/agent/runs). Submit a
   task, watch the live event feed, steer / follow up / abort mid-flight.
   Every finished run leaves a pi session the Agent traces tab can
   distill from.
   ===================================================================== */
const AGENT_RUN_TERMINAL = new Set(['completed', 'failed', 'aborted', 'timed_out', 'interrupted']);

// queued/running/completed/failed map straight onto the job-state-pill
// palette; the run-only terminals reuse the closest existing tone (no
// new CSS): aborted/interrupted read as cancelled, timed_out as failed.
function agentRunPill(status) {
  const s = String(status || 'queued');
  let cls = (s === 'aborted' || s === 'interrupted') ? 'cancelled' : (s === 'timed_out' ? 'failed' : s);
  // The status lands in a class attribute — only known tokens pass (an
  // unexpected server value must not write arbitrary attribute text).
  if (!/^[a-z_]+$/.test(cls)) cls = 'queued';
  return `<span class="job-state-pill ${cls}">${escapeHtml(s.replace(/_/g, ' '))}</span>`;
}

// List the run engine's status line + run history — fired on tab entry
// and every 3s while the pane is visible.
async function refreshAgentRuns() {
  const statusNode = document.getElementById('agent-runs-status');
  const node = document.getElementById('agent-runs-list');
  if (!node) return;
  const startBtn = document.getElementById('agent-run-start');
  try {
    const st = await api('/v1/agent/runs/status');
    const ready = st.enabled && st.pi_available;
    let line;
    if (!st.enabled) {
      line = `Embedded runs are disabled — ${escapeHtml(st.disabled_reason || 'gate closed')}.`;
    } else if (!st.pi_available) {
      line = 'Embedded runs need <code>pi</code> on the server’s PATH — <code>npm install -g @earendil-works/pi-coding-agent</code>, then come back here.';
    } else {
      line = `Run engine ready — pi at <code>${escapeHtml(st.pi_path || '')}</code> · ${st.active_runs}/${st.max_concurrent_runs} active · sessions land in <code>${escapeHtml(st.sessions_dir || '')}</code>.`;
    }
    // The key carries every field the line renders, or a changed
    // disabled_reason/path would paint stale.
    setListHtml(statusNode, 'status:' + JSON.stringify([st.enabled, st.disabled_reason, st.pi_available, st.pi_path, st.sessions_dir, st.active_runs, st.max_concurrent_runs]), line);
    if (startBtn) startBtn.disabled = !ready;
  } catch (e) {
    setListHtml(statusNode, 'statuserr:' + e.message, `Couldn't reach the run engine: ${escapeHtml(e.message)}`);
  }
  try {
    const res = await api('/v1/agent/runs');
    renderAgentRunsList(res.runs || []);
  } catch (e) {
    setListHtml(node, 'err:' + e.message, `<div class="empty">Couldn't load runs: ${escapeHtml(e.message)}</div>`);
  }
}

function renderAgentRunsList(runs) {
  const node = document.getElementById('agent-runs-list');
  if (!node) return;
  if (runs.length === 0) {
    setListHtml(node, 'empty',
      '<div class="empty">No runs yet. Describe a task above and start one — every run saves a pi session you can distill from.</div>');
    return;
  }
  // The minute bucket keeps the relative "Nm ago" stamps moving — the
  // setListHtml key must change whenever rendered content would.
  const listKey = 'list:' + Math.floor(Date.now() / 60000) + ':' +
    JSON.stringify(runs.map(r => [r.id, r.status, r.num_turns, r.num_tool_calls, r.finished_unix_ms]));
  const cards = runs.map(r => {
    const task = String(r.task || '');
    const preview = task.length > 90 ? task.slice(0, 90) + '…' : task;
    const errLine = (r.status === 'failed' || r.status === 'timed_out') && r.error
      ? `<div style="font-size:var(--text-2xs); color:var(--danger-fg); margin-top:var(--space-1);">${escapeHtml(String(r.error).length > 160 ? String(r.error).slice(0, 160) + '…' : String(r.error))}</div>`
      : '';
    return `<button type="button" class="adapter-card" data-run-open="${escapeHtml(r.id || '')}" style="display:block; width:100%; text-align:left; font:inherit; color:inherit; margin-bottom:var(--space-2);" title="Open this run — watch the live event feed, steer it, or read how it ended">
      <div style="display:flex; gap:var(--space-3); align-items:baseline; flex-wrap:wrap;">
        <span style="font-weight:600; font-family:var(--font-mono); font-size:var(--text-xs);">${escapeHtml(shortId(r.id))}</span>
        ${agentRunPill(r.status)}
        ${r.label ? `<span class="hint">${escapeHtml(r.label)}</span>` : ''}
        <span style="margin-left:auto; font-size:var(--text-2xs); color:var(--text-muted);">${escapeHtml(fmtSmartTime(r.created_unix_ms))}</span>
      </div>
      <div style="font-size:var(--text-xs); margin-top:var(--space-1);">${escapeHtml(preview)}</div>
      <div style="font-size:var(--text-xs); color:var(--text-muted);">${r.num_turns || 0} turns · ${r.num_tool_calls || 0} tool calls · ${escapeHtml(r.cwd || '')}</div>
      ${errLine}
    </button>`;
  }).join('');
  if (setListHtml(node, listKey, cards)) {
    node.querySelectorAll('[data-run-open]').forEach(btn => {
      btn.addEventListener('click', () => openRunDrillModal(btn.dataset.runOpen));
    });
  }
}

// New-run form: POST /v1/agent/runs, then drill straight into the run.
async function submitAgentRun() {
  const taskEl = document.getElementById('agent-run-task');
  const task = (taskEl?.value || '').trim();
  if (!task) { toast('Describe a task for the agent first', 'err'); taskEl?.focus(); return; }
  const cwd = (document.getElementById('agent-run-cwd')?.value || '').trim();
  const label = (document.getElementById('agent-run-label')?.value || '').trim();
  const body = { task };
  if (cwd) body.cwd = cwd;
  if (label) body.label = label;
  const startBtn = document.getElementById('agent-run-start');
  if (startBtn) startBtn.disabled = true;
  try {
    const rec = await api('/v1/agent/runs', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    if (taskEl) taskEl.value = '';
    toast(`Run ${shortId(rec.id)} queued`, 'ok');
    refreshAgentRuns();
    openRunDrillModal(rec.id);
  } catch (e) {
    toast(e.message, 'err');
  } finally {
    // refreshAgentRuns re-applies the status gate on its next pass.
    if (startBtn) startBtn.disabled = false;
  }
}
document.getElementById('agent-run-start')?.addEventListener('click', submitAgentRun);
document.getElementById('agent-run-task')?.addEventListener('keydown', (ev) => {
  if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') { ev.preventDefault(); submitAgentRun(); }
});

// Keep the list live while the runs pane is showing — gated on the pane
// AND the Distill page being frontmost, mirroring the eval-badge pattern
// of visibility-gated background intervals.
setInterval(() => {
  const pane = document.getElementById('distill-tab-runs-pane');
  if (!pane || pane.hidden) return;
  if (!document.getElementById('page-distill')?.classList.contains('active')) return;
  refreshAgentRuns();
}, 3000);

/* =====================================================================
   Agent run drill-in modal — live event feed (1s ?after= cursor polls)
   + steer / follow-up / abort for one embedded run.
   ===================================================================== */
let runDrillId = null;
let runDrillCursor = 0;
let runDrillStatus = null;
let runDrillPollHandle = null;
// Generation token: bumped on every open AND close. Post-await guards
// compare against it instead of the run id — id equality can't tell
// "same run, new modal session" apart, which let a stale in-flight poll
// regress the fresh cursor or leak a second interval on quick
// close-then-reopen of the same run.
let runDrillGen = 0;
// In-flight guard: a poll that outlives its 1s slot must not overlap
// the next one — overlapping polls share a cursor and append the same
// events twice (this feed appends; it can't repaint idempotently).
let runDrillPollBusy = false;

const RUN_EVENT_CLAMP_CHARS = 400;
const RUN_TEXT_CLAMP_CHARS = 700;

function runEventClamp(text, limit) {
  const s = String(text ?? '');
  return s.length > limit ? s.slice(0, limit) + '…' : s;
}

// Run ids ride the #distill/runs/{id} deep-link grammar like the other
// drills.
async function openRunDrillModal(id) {
  const gen = ++runDrillGen;
  runDrillId = id;
  modalHashOnOpen('run', '#distill/runs/' + encodeURIComponent(id));
  const modal = document.getElementById('run-drill-modal');
  if (!modal) return;
  modal.hidden = false;
  openModal(modal, { onClose: userCloseRunDrillModal });
  document.getElementById('run-drill-title').textContent = 'Agent run';
  document.getElementById('run-drill-meta').textContent = id;
  const feed = document.getElementById('run-drill-events');
  feed.innerHTML = '<div class="detail-empty">Loading…</div>';
  delete feed.dataset.gapNoted;
  runDrillCursor = 0;
  runDrillStatus = null;
  runDrillPollBusy = false;
  if (runDrillPollHandle) { clearInterval(runDrillPollHandle); runDrillPollHandle = null; }
  let rec;
  try {
    rec = await api('/v1/agent/runs/' + encodeURIComponent(id));
    if (gen !== runDrillGen) return; // closed or re-targeted while fetching
    renderRunDrillHead(rec);
  } catch (e) {
    if (gen !== runDrillGen) return;
    feed.innerHTML = `<div class="detail-empty">Couldn't load this run: ${escapeHtml(e.message)}</div>`;
    return;
  }
  feed.innerHTML = '';
  await pollRunDrillEvents(gen);
  if (gen !== runDrillGen) return;
  // A run that's already over still owes the reader its ending — the
  // live path appends the error on the status flip, but a deep link or
  // reopen arrives after the flip already happened.
  if (AGENT_RUN_TERMINAL.has(rec.status) && rec.error) {
    feed.insertAdjacentHTML('beforeend',
      `<div class="req-section req-error"><div class="req-section-head">error</div><pre class="req-pre">${escapeHtml(runEventClamp(rec.error, RUN_EVENT_CLAMP_CHARS))}</pre></div>`);
    feed.scrollTop = feed.scrollHeight;
  }
  if (!(runDrillStatus && AGENT_RUN_TERMINAL.has(runDrillStatus))) {
    runDrillPollHandle = setInterval(() => pollRunDrillEvents(gen), 1000);
  }
}

function renderRunDrillHead(rec) {
  runDrillStatus = rec.status || null;
  document.getElementById('run-drill-title').textContent = `Agent run ${shortId(rec.id)}`;
  const bits = [`${rec.num_turns || 0} turns`, `${rec.num_tool_calls || 0} tool calls`];
  if (rec.label) bits.push(rec.label);
  if (rec.cwd) bits.push(rec.cwd);
  document.getElementById('run-drill-meta').innerHTML =
    `${agentRunPill(rec.status)} <span class="hint" style="margin-left:8px;">${bits.map(b => escapeHtml(b)).join(' · ')}</span>`;
  const abortBtn = document.getElementById('run-drill-abort');
  if (abortBtn) abortBtn.disabled = AGENT_RUN_TERMINAL.has(rec.status);
}

async function pollRunDrillEvents(gen) {
  if (gen !== runDrillGen) return;
  if (runDrillPollBusy) return; // previous poll still in flight
  const id = runDrillId;
  if (!id) return;
  const feed = document.getElementById('run-drill-events');
  if (!feed) return;
  runDrillPollBusy = true;
  try {
    const res = await api('/v1/agent/runs/' + encodeURIComponent(id) + '/events?after=' + runDrillCursor);
    if (gen !== runDrillGen) return;
    runDrillCursor = res.next_after ?? runDrillCursor;
    // Auto-scroll only when the user is already reading the bottom — a
    // scrolled-up reader keeps their place while events keep landing.
    const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 40;
    if (res.truncated && !feed.dataset.gapNoted) {
      feed.dataset.gapNoted = '1';
      feed.insertAdjacentHTML('beforeend',
        '<div style="font-size:var(--text-2xs); color:var(--text-muted);">… earlier events are no longer buffered (long run or server restart) — the full trajectory lives in the session trace …</div>');
    }
    const html = (res.events || []).map(item => renderRunEvent(item.event)).filter(Boolean).join('');
    if (html) feed.insertAdjacentHTML('beforeend', html);
    if (atBottom) feed.scrollTop = feed.scrollHeight;
    if (res.status && res.status !== runDrillStatus) {
      // Status flipped (queued→running or →terminal): re-pull the record
      // so the header pill, counts, and error are fresh.
      try {
        const rec = await api('/v1/agent/runs/' + encodeURIComponent(id));
        if (gen !== runDrillGen) return;
        renderRunDrillHead(rec);
        if (AGENT_RUN_TERMINAL.has(rec.status) && rec.error) {
          feed.insertAdjacentHTML('beforeend',
            `<div class="req-section req-error"><div class="req-section-head">error</div><pre class="req-pre">${escapeHtml(runEventClamp(rec.error, RUN_EVENT_CLAMP_CHARS))}</pre></div>`);
          if (atBottom) feed.scrollTop = feed.scrollHeight;
        }
      } catch {
        // Record fetch is best-effort, but the status flip must still
        // land — on a terminal flip there IS no next poll to catch up.
        runDrillStatus = res.status;
        const pill = document.querySelector('#run-drill-meta .job-state-pill');
        if (pill) pill.outerHTML = agentRunPill(res.status);
        const abortBtn = document.getElementById('run-drill-abort');
        if (abortBtn) abortBtn.disabled = AGENT_RUN_TERMINAL.has(res.status);
      }
    }
    if (res.status && AGENT_RUN_TERMINAL.has(res.status) && runDrillPollHandle) {
      clearInterval(runDrillPollHandle);
      runDrillPollHandle = null;
    }
  } catch (e) {
    // Run vanished (e.g. server restart): stop hammering the endpoint.
    if (gen === runDrillGen && e.status === 404 && runDrillPollHandle) {
      clearInterval(runDrillPollHandle);
      runDrillPollHandle = null;
    }
  } finally {
    runDrillPollBusy = false;
  }
}

// One pi agent event → one compact feed line (or '' for noise).
function renderRunEvent(ev) {
  if (!ev || typeof ev !== 'object') return '';
  const ty = ev.type || '';
  const dim = (text) => `<div style="font-size:var(--text-2xs); color:var(--text-muted);">${escapeHtml(text)}</div>`;
  if (ty === 'agent_start') return dim('— agent start —');
  if (ty === 'agent_end') return dim('— agent end —');
  if (ty === 'kiln_note') return dim('kiln: ' + (ev.note || ''));
  if (ty === 'message_end') {
    const msg = ev.message || {};
    if (msg.role !== 'assistant') return '';
    const blocks = Array.isArray(msg.content) ? msg.content : [];
    const parts = blocks.map(b => {
      if (!b || typeof b !== 'object') return '';
      if (b.type === 'text' && b.text) {
        return `<pre class="req-pre">${escapeHtml(runEventClamp(b.text, RUN_TEXT_CLAMP_CHARS))}</pre>`;
      }
      if (b.type === 'thinking' && b.thinking) {
        return `<div><div style="font-size:var(--text-2xs); color:var(--text-muted); text-transform:uppercase; letter-spacing:var(--tracking-caps); margin-bottom:4px;">thinking</div><pre class="req-pre">${escapeHtml(runEventClamp(b.thinking, RUN_EVENT_CLAMP_CHARS))}</pre></div>`;
      }
      return ''; // toolCall blocks are covered by tool_execution_start
    }).filter(Boolean).join('');
    if (!parts) return '';
    return `<div class="req-section"><div class="req-section-head">assistant</div>${parts}</div>`;
  }
  if (ty === 'tool_execution_start') {
    let args = '';
    try { args = JSON.stringify(ev.args ?? {}); } catch { args = String(ev.args ?? ''); }
    return `<div style="font-family:var(--font-mono); font-size:var(--text-xs); color:var(--text-muted);">→ ${escapeHtml(ev.toolName || '?')}(${escapeHtml(runEventClamp(args, 160))})</div>`;
  }
  if (ty === 'tool_execution_end') {
    let result = ev.result;
    if (result != null && typeof result !== 'string') {
      try { result = JSON.stringify(result); } catch { result = String(result); }
    }
    return `<div class="req-section${ev.isError ? ' req-error' : ''}"><div class="req-section-head">${escapeHtml(ev.toolName || 'tool')}${ev.isError ? ' · error' : ''}</div><pre class="req-pre">${escapeHtml(runEventClamp(result || '', RUN_EVENT_CLAMP_CHARS))}</pre></div>`;
  }
  if (ty === 'response') {
    if (ev.success === false) {
      return `<div class="req-section req-error"><div class="req-section-head">${escapeHtml(ev.command || 'command')} failed</div><pre class="req-pre">${escapeHtml(runEventClamp(ev.error || JSON.stringify(ev), RUN_EVENT_CLAMP_CHARS))}</pre></div>`;
    }
    return '';
  }
  return '';
}

function closeRunDrillModal() {
  runDrillGen++; // invalidate any in-flight fetches from this session
  runDrillId = null;
  runDrillCursor = 0;
  runDrillStatus = null;
  runDrillPollBusy = false;
  if (runDrillPollHandle) { clearInterval(runDrillPollHandle); runDrillPollHandle = null; }
  const modal = document.getElementById('run-drill-modal');
  if (!modal) return;
  const feed = document.getElementById('run-drill-events');
  if (feed) delete feed.dataset.gapNoted;
  modal.hidden = true;
  closeModal(modal);
}
// User-initiated close (X / backdrop / Esc): walk history per the
// deep-link state machine, exactly like the other drills.
function userCloseRunDrillModal() {
  modalHashOnUserClose('run', '#distill/runs', closeRunDrillModal);
}
document.getElementById('run-drill-close')?.addEventListener('click', userCloseRunDrillModal);
document.getElementById('run-drill-modal')?.addEventListener('click', ev => {
  if (ev.target.id === 'run-drill-modal') userCloseRunDrillModal();
});

// Steer interrupts the current turn; Follow-up queues after agent_end.
// Both share the one input row at the bottom of the modal.
async function sendRunDrillMessage(endpoint, verb) {
  const id = runDrillId;
  if (!id) return;
  const input = document.getElementById('run-drill-steer-input');
  const message = (input?.value || '').trim();
  if (!message) { toast(`Type a message to ${verb.toLowerCase()} with first`, 'err'); input?.focus(); return; }
  try {
    await api('/v1/agent/runs/' + encodeURIComponent(id) + '/' + endpoint, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ message }),
    });
    if (input) input.value = '';
    toast(`${verb} queued`, 'ok');
  } catch (e) {
    toast(e.message, 'err');
  }
}
document.getElementById('run-drill-steer-send')?.addEventListener('click', () => sendRunDrillMessage('steer', 'Steer'));
document.getElementById('run-drill-followup-send')?.addEventListener('click', () => sendRunDrillMessage('follow_up', 'Follow-up'));
document.getElementById('run-drill-steer-input')?.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') { ev.preventDefault(); sendRunDrillMessage('steer', 'Steer'); }
});

document.getElementById('run-drill-abort')?.addEventListener('click', async () => {
  const id = runDrillId;
  if (!id) return;
  if (!confirm('Abort this run? pi stops at the next opportunity.')) return;
  try {
    await api('/v1/agent/runs/' + encodeURIComponent(id) + '/abort', { method: 'POST' });
    toast('Abort requested', 'ok');
  } catch (e) {
    toast(e.message, 'err');
  }
});
