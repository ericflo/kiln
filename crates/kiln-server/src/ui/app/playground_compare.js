
/* =====================================================================
   Playground: A/B compare mode + send-to-judgments
   ===================================================================== */

// We keep the simple chat mode entirely intact. Compare mode toggles a
// second adapter dropdown + a second reply column, fans the same prompt
// out to both, and offers a "Save A/B preference" button that ships the
// pair into a chosen judgment dataset. Saves the inevitable copy-paste
// dance into the Evals tab.

// Inject the compare-mode controls into the top .chat-controls row so
// the toggle is right next to the adapter dropdown — previously the
// toggle hid below the input row where users never found it.
const playgroundCard = document.querySelector('.playground-card');
if (playgroundCard) {
  const topControls = playgroundCard.querySelector('.chat-controls');
  if (topControls) {
    const compareFrag = document.createElement('span');
    compareFrag.style.cssText = 'display:flex; align-items:center; gap:var(--space-3);';
    compareFrag.innerHTML = `
      <label class="chat-toggle-label" style="user-select:none; cursor:pointer;" title="Send the same prompt to two adapters side-by-side for direct comparison.">
        <input type="checkbox" id="chat-compare-toggle">
        <span>Compare</span>
      </label>
      <label for="chat-adapter-b" id="chat-adapter-b-label" style="display:none;">vs</label>
      <select id="chat-adapter-b" style="display:none;"><option value="">Base model</option></select>`;
    const advanced = topControls.querySelector('#chat-toggle-advanced');
    if (advanced) topControls.insertBefore(compareFrag, advanced);
    else topControls.appendChild(compareFrag);
  }
  // Save-pair action sits in chat-output-actions next to Copy/Export.
  const outputActions = playgroundCard.querySelector('.chat-output-actions');
  if (outputActions) {
    const saveBtn = document.createElement('button');
    saveBtn.type = 'button';
    saveBtn.className = 'btn btn-sm';
    saveBtn.id = 'chat-save-judgment';
    saveBtn.disabled = true;
    saveBtn.title = 'Send this A/B pair into a judgment dataset';
    saveBtn.innerHTML = icon('arrow-right','icn-sm') + ' Save A/B preference';
    const exportBtn = outputActions.querySelector('#chat-export');
    if (exportBtn) outputActions.insertBefore(saveBtn, exportBtn);
    else outputActions.appendChild(saveBtn);
  }
  // The compare reply panel — appears under the existing chat-output.
  const comparePair = document.createElement('div');
  comparePair.id = 'chat-compare-pair';
  comparePair.className = 'compare-pair';
  comparePair.style.display = 'none';
  comparePair.style.padding = '0 var(--space-5) var(--space-4)';
  const sidePlaceholder = `<div style="color:var(--text-muted); font-style:italic; font-size:12px; padding:8px 0;">Pick adapters above and send a prompt to fan it out side-by-side.</div>`;
  comparePair.innerHTML = `
    <div class="compare-side"><div class="compare-side-head">A · <span id="chat-compare-a-name">base</span></div><div class="compare-side-body" id="chat-compare-a-body">${sidePlaceholder}</div></div>
    <div class="compare-side"><div class="compare-side-head">B · <span id="chat-compare-b-name">base</span></div><div class="compare-side-body" id="chat-compare-b-body">${sidePlaceholder}</div></div>`;
  const chatOutput = playgroundCard.querySelector('.chat-output-actions');
  if (chatOutput) chatOutput.parentNode.insertBefore(comparePair, chatOutput);
}

let chatCompareMode = false;
let chatComparePair = null;
const chatCompareToggle = document.getElementById('chat-compare-toggle');
chatCompareToggle?.addEventListener('change', ev => {
  chatCompareMode = ev.target.checked;
  document.getElementById('chat-adapter-b-label').style.display = chatCompareMode ? '' : 'none';
  document.getElementById('chat-adapter-b').style.display = chatCompareMode ? '' : 'none';
  document.getElementById('chat-compare-pair').style.display = chatCompareMode ? '' : 'none';
  // Hide the simple-mode chat history when compare is on — compare runs
  // are stateless, single-prompt, so showing prior turns is confusing.
  // The chat history isn't cleared (toggling back off restores it).
  const chatOutput = document.getElementById('chat-output');
  if (chatOutput) chatOutput.style.display = chatCompareMode ? 'none' : '';
  const chatExport = document.getElementById('chat-export');
  if (chatExport) chatExport.style.display = chatCompareMode ? 'none' : '';
  const chatCopy = document.querySelector('[data-copy-chat-response]');
  if (chatCopy) chatCopy.style.display = chatCompareMode ? 'none' : '';
  // Sync the B dropdown to the same options as A.
  const a = document.getElementById('chat-adapter');
  const b = document.getElementById('chat-adapter-b');
  if (a && b) b.innerHTML = a.innerHTML;
});

/* ---------------------------------------------------------------------
   Compare-mode streaming

   Previously this fanned out two *non-streaming* completions and
   awaited Promise.all, so:
     - You stared at "Generating…" until both sides finished, with no
       per-side progress.
     - Reasoning-capable models discarded the entire `<think>` block
       since non-streaming responses only return the post-`</think>`
       content as `message.content`.
   The streaming variant solves both: each side renders text as it
   arrives, with a live "Thinking…" header populated from
   `delta.reasoning_content`, and the existing Save A/B preference
   flow keeps working off the final content.
   --------------------------------------------------------------------- */

function _renderCompareSide(side, m) {
  const head = document.getElementById(`chat-compare-${side}-body`);
  if (!head) return;
  let html = '';
  if (m.reasoning) {
    const live = m.pending && !m.content;
    const dur = (m.thinkStartMs && m.thinkEndMs)
      ? formatChatDuration(m.thinkEndMs - m.thinkStartMs)
      : (live && m.thinkStartMs ? formatChatDuration(performance.now() - m.thinkStartMs) : null);
    const label = live ? 'Thinking' : 'Thought';
    const outcome = live ? '' : thinkingBudgetSummary(m.thinkingBudget);
    const meta = `${dur ? `<span class="think-meta"> · ${live ? '' : 'for '}${escapeHtml(dur)}</span>` : ''}`
      + `${outcome ? `<span class="think-meta"> · ${escapeHtml(outcome)}</span>` : ''}`;
    html += `<details class="think-block compare-think${live ? ' live' : ''}"${live ? ' open' : ''}>
      <summary><span class="think-label">${label}</span>${meta}</summary>
      <div class="think-body">${escapeHtml(m.reasoning)}</div>
    </details>`;
  }
  if (m.error) {
    html += `<div class="err-block">${escapeHtml(m.error)}</div>`;
  } else if (m.content) {
    html += m.pending
      ? `<pre style="white-space:pre-wrap; margin:0;">${escapeHtml(m.content)}</pre>`
      : `<div class="md-body">${renderMarkdown(m.content)}</div>`;
  } else if (m.pending) {
    html += `<div style="color:var(--text-muted); font-style:italic; font-size:11px;">Generating…</div>`;
  }
  if (m.ttftMs != null || m.durationMs != null) {
    const stats = [];
    if (m.ttftMs != null)     stats.push(`<span class="stat"><strong>TTFT</strong> ${escapeHtml(formatChatDuration(m.ttftMs))}</span>`);
    if (m.durationMs != null) stats.push(`<span class="stat"><strong>${m.pending ? 'Elapsed' : 'Total'}</strong> ${escapeHtml(formatChatDuration(m.durationMs))}</span>`);
    const tps = chatTokensPerSec(m);
    if (tps != null) stats.push(`<span class="stat"><strong>~${tps.toFixed(tps >= 100 ? 0 : 1)}</strong> tok/s</span>`);
    appendCompletionOutcomeStats(stats, m, !!m.reasoning);
    html += `<div class="turn-foot" style="margin-top:6px;">${stats.join('')}</div>`;
  }
  head.innerHTML = html;
}

async function streamCompareSide(side, adapterName, prompt, temp, thinkingBudget, signal) {
  const m = {
    role: 'assistant', content: '', reasoning: '',
    pending: true,
    startMs: performance.now(),
    firstTokenMs: null, lastTokenMs: null,
    thinkStartMs: null, thinkEndMs: null,
    ttftMs: null, durationMs: null,
    error: null, adapter: adapterName || null,
  };
  _renderCompareSide(side, m);
  const tick = setInterval(() => {
    if (!m.pending) return;
    m.durationMs = performance.now() - m.startMs;
    _renderCompareSide(side, m);
  }, 250);
  try {
    const body = buildChatRequestBody({
      messages: (() => {
        const sys = getSystemPromptMessage();
        const user = { role: 'user', content: prompt };
        return sys ? [sys, user] : [user];
      })(),
      temperature: temp,
      thinkingBudget,
    });
    if (servedModelId) body.model = servedModelId;
    if (adapterName) body.adapter = adapterName;

    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Kiln-Client': 'dashboard' },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }
    await consumeChatCompletionSse(
      res,
      m,
      () => _renderCompareSide(side, m),
      'playground compare',
    );
    m.pending = false;
    m.durationMs = (m.lastTokenMs || performance.now()) - m.startMs;
    if (m.thinkStartMs != null && m.thinkEndMs == null && m.content) {
      m.thinkEndMs = m.lastTokenMs || performance.now();
    }
  } catch (e) {
    m.pending = false;
    if (e.name === 'AbortError') {
      m.error = 'Aborted before completion.';
    } else {
      m.error = e?.message || String(e);
    }
  } finally {
    clearInterval(tick);
    _renderCompareSide(side, m);
  }
  return m;
}

// Hook send button: when compare mode is on, fan out to A and B in
// parallel and render side-by-side with live streaming. Otherwise
// let the existing `sendChat` handler take it (it early-returns when
// compare mode is on, so the two handlers don't fight).
let chatCompareAbort = null;
async function sendChatCompare() {
  if (chatCompareAbort) return;
  const promptEl = document.getElementById('chat-input');
  const prompt = (promptEl?.value || '').trim();
  if (!prompt) return;
  let temp;
  try { temp = parseQuickInferenceTemperature(document.getElementById('chat-temp')); }
  catch (error) { toast(error.message, 'err'); return; }
  const thinkingBudget = readThinkingBudgetRequestOrNotify();
  if (!thinkingBudget) return;

  const adapterA = document.getElementById('chat-adapter').value;
  const adapterB = document.getElementById('chat-adapter-b').value;
  document.getElementById('chat-compare-a-name').textContent = adapterA || 'base';
  document.getElementById('chat-compare-b-name').textContent = adapterB || 'base';
  promptEl.value = '';
  autoresizeChatInput();
  updateChatSendState();
  setChatGenerating(true);
  document.getElementById('chat-save-judgment').disabled = true;

  chatCompareAbort = new AbortController();
  try {
    const [a, b] = await Promise.all([
      streamCompareSide('a', adapterA, prompt, temp, thinkingBudget, chatCompareAbort.signal),
      streamCompareSide('b', adapterB, prompt, temp, thinkingBudget, chatCompareAbort.signal),
    ]);
    if (a.content || b.content) {
      chatComparePair = {
        prompt: [{ role: 'user', content: prompt }],
        adapter_a: adapterA || null,
        adapter_b: adapterB || null,
        response_a: a.content || '',
        response_b: b.content || '',
      };
      document.getElementById('chat-save-judgment').disabled = false;
    }
  } finally {
    chatCompareAbort = null;
    setChatGenerating(false);
  }
}
const chatSendBtn = document.getElementById('chat-send');
if (chatSendBtn) {
  chatSendBtn.addEventListener('click', () => {
    if (!chatCompareMode) return;  // sendChat handler covers simple mode
    sendChatCompare();
  });
}
// Enter-key in the textarea routes through sendChat, which early-returns
// when compare mode is on. Route those Enter presses to the compare flow.
document.getElementById('chat-input')?.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' || e.shiftKey) return;
  if (!chatCompareMode) return;
  e.preventDefault();
  e.stopImmediatePropagation();
  sendChatCompare();
}, { capture: true });

// Wire the existing #chat-stop button so it aborts compare-mode
// streams too (the simple-mode handler already aborts `chatAbort`).
document.getElementById('chat-stop').addEventListener('click', () => {
  if (chatCompareAbort) chatCompareAbort.abort();
}, { capture: false });

// Save the current A/B pair into a judgment dataset. Renders a small
// inline form (replaces two consecutive `prompt()` dialogs which were
// terrible UX and blocked the event loop).
document.getElementById('chat-save-judgment')?.addEventListener('click', () => {
  if (!chatComparePair) return;
  const existing = document.getElementById('chat-save-judgment-form');
  if (existing) { existing.remove(); return; }
  const host = document.getElementById('chat-compare-pair');
  if (!host) return;
  const form = document.createElement('div');
  form.id = 'chat-save-judgment-form';
  form.style.cssText = 'display:flex; gap:8px; align-items:center; padding:10px; margin-top:8px; background:var(--surface-2); border:1px solid var(--border); border-radius:6px; flex-wrap:wrap;';
  form.innerHTML = `
    <input id="chat-save-dataset" type="text" placeholder="dataset name" value="playground-pair" style="flex:1; min-width:140px; padding:6px 10px; background:var(--surface); border:1px solid var(--border); border-radius:4px; color:var(--text); font-family:inherit;">
    <select id="chat-save-winner" style="padding:6px 10px; background:var(--surface); border:1px solid var(--border); border-radius:4px; color:var(--text); font-family:inherit;">
      <option value="a">A wins</option>
      <option value="b">B wins</option>
      <option value="tie" selected>Tie</option>
      <option value="skip">Skip</option>
    </select>
    <button class="btn btn-sm btn-primary" id="chat-save-confirm" type="button">Save</button>
    <button class="btn btn-sm" id="chat-save-cancel" type="button">Cancel</button>`;
  host.parentNode.insertBefore(form, host.nextSibling);
  document.getElementById('chat-save-cancel').addEventListener('click', () => form.remove());
  document.getElementById('chat-save-confirm').addEventListener('click', async () => {
    const datasetName = (document.getElementById('chat-save-dataset').value || '').trim();
    if (!datasetName) { toast('Dataset name required', 'err'); return; }
    const winner = document.getElementById('chat-save-winner').value;
    try {
      // 409 (already-exists) is fine — we just append a row below.
      try { await api('/v1/judgments', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name: datasetName }) }); } catch (_) { /* already exists is fine */ }
      const m = await api('/v1/judgments/' + encodeURIComponent(datasetName) + '/rows', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ ...chatComparePair, winner, note: 'from playground', tags: ['playground'] }),
      });
      recordedJudgmentToast('Saved into ' + datasetName, datasetName, m.judgment_id);
      form.remove();
    } catch (e) { toast('Save failed: ' + e.message, 'err'); }
  });
});
