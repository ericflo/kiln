
// --- Adapters ---
let lastAdapters = null;

async function pollAdapters() {
  const adaptersPanel = setPanelBusy('adapters-panel', true);
  if (!adaptersPanel) return;
  try {
    const data = await api('/v1/adapters');
    lastAdapters = data;
    window.lastAdapters = data;
    // The cards renderer owns `#adapters-panel`; these two helpers
    // update orthogonal UI (chat-adapter dropdown, merge-sources panel)
    // that the cards don't touch.
    updateAdapterSelect(data);
    renderMergeSources();
    if (typeof refreshAdapterCards === 'function') refreshAdapterCards();
    const count = (data.available || []).length;
    setText('adapters-count', String(count));
  } catch (e) {
    // refreshAdapterCards owns this panel and dedupes on lastAdaptersKey;
    // reset it so the card list repaints once the server recovers.
    lastAdaptersKey = null;
    adaptersPanel.innerHTML = apiFailureHtml('Adapters', e, 'pollAdapters');
  } finally {
    setPanelBusy('adapters-panel', false);
  }
}

function updateAdapterSelect(data) {
  const sel = document.getElementById('chat-adapter');
  const b = document.getElementById('chat-adapter-b');
  // Rebuild the <option> list only when its rendered content (names +
  // active marker) actually changed — an unconditional rebuild on every
  // adapters poll snaps an open dropdown shut mid-pick. In particular the
  // option set is never rebuilt while the select has focus with unchanged
  // options, because unchanged options always skip.
  const names = (data.available || []).map(a => a.name);
  const optionsKey = 'opts:' + JSON.stringify([names, data.active || '']);
  const optionsHtml = '<option value="">Base model</option>' + names.map(n =>
    `<option value="${escapeHtml(n)}">${escapeHtml(n)}${data.active === n ? ' (active)' : ''}</option>`
  ).join('');
  const current = sel.value;
  if (setListHtml(sel, optionsKey, optionsHtml)) {
    sel.value = current; // preserve the user's in-flight selection
  }
  // Keep the compare (B) dropdown's options in sync, preserving its selection.
  if (b) {
    const bCurrent = b.value;
    if (setListHtml(b, optionsKey, optionsHtml)) {
      b.value = bCurrent;
    }
  }
  // Apply any deferred selection ("Verify the fix" names an adapter that's
  // still training) the moment the option actually exists.
  for (const el of [sel, b]) {
    if (!el) continue;
    const want = el.dataset.pendingValue;
    if (want && Array.from(el.options).some(o => o.value === want)) {
      el.value = want;
      delete el.dataset.pendingValue;
      toast(`${want} finished training — it's now selected for compare`, 'ok');
    }
  }
}

window.loadAdapter = async function(name) {
  try {
    await api('/v1/adapters/load', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name}) });
    toast('Loaded adapter: ' + name);
    pollAdapters();
    pollHealth();
  } catch (e) { toast(e.message, 'err'); }
};

window.unloadAdapter = async function() {
  try {
    await api('/v1/adapters/unload', { method: 'POST' });
    toast('Unloaded adapter');
    pollAdapters();
    pollHealth();
  } catch (e) { toast(e.message, 'err'); }
};

window.deleteAdapter = async function(name) {
  if (!confirm('Delete adapter "' + name + '"? This cannot be undone.')) return;
  try {
    await api('/v1/adapters/' + encodeURIComponent(name), { method: 'DELETE' });
    toast('Deleted adapter: ' + name);
    pollAdapters();
  } catch (e) { toast(e.message, 'err'); }
};

window.downloadAdapter = function(name) {
  // Browser saves the response via Content-Disposition: attachment.
  window.location.href = '/v1/adapters/' + encodeURIComponent(name) + '/download';
};

let uploadAdapterBusy = false;
let uploadNameWasAutofilled = false;
let lastAutofilledUploadName = '';

function pathSafeAdapterStemFromArchiveName(fileName) {
  const baseName = String(fileName || '').split(/[\\/]/).pop() || '';
  const stem = baseName.replace(/\.tar\.gz$/i, '').replace(/\.tgz$/i, '');
  return stem
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[\\/]+/g, '-')
    .replace(/[^a-z0-9._-]+/gi, '-')
    .replace(/\.\.+/g, '.')
    .replace(/-+/g, '-')
    .replace(/^[.-]+|[.-]+$/g, '');
}

function maybeAutofillUploadName() {
  const nameEl = document.getElementById('upload-name');
  const fileEl = document.getElementById('upload-archive');
  if (!nameEl || !fileEl || fileEl.files.length === 0) return;

  const currentName = nameEl.value.trim();
  if (currentName && (!uploadNameWasAutofilled || currentName !== lastAutofilledUploadName)) return;

  const autoName = pathSafeAdapterStemFromArchiveName(fileEl.files[0].name);
  if (!autoName) return;
  nameEl.value = autoName;
  uploadNameWasAutofilled = true;
  lastAutofilledUploadName = autoName;
}

function handleUploadNameInput() {
  const nameEl = document.getElementById('upload-name');
  if (!nameEl) return;
  if (uploadNameWasAutofilled && nameEl.value.trim() === lastAutofilledUploadName) {
    updateUploadAdapterState();
    return;
  }
  uploadNameWasAutofilled = false;
  updateUploadAdapterState();
}

function handleUploadArchiveChange() {
  maybeAutofillUploadName();
  updateUploadAdapterState();
}

function updateUploadAdapterState() {
  const nameEl = document.getElementById('upload-name');
  const fileEl = document.getElementById('upload-archive');
  const button = document.getElementById('upload-adapter-btn');
  const state = document.getElementById('upload-adapter-state');
  if (!nameEl || !fileEl || !button) return;
  if (uploadAdapterBusy) {
    button.disabled = true;
    if (state) state.textContent = 'Uploading adapter…';
    return;
  }
  const uploadName = nameEl.value.trim();
  const hasName = uploadName.length > 0;
  const hasPathSafeName = isPathSafeAdapterDirectoryName(uploadName);
  const hasFile = fileEl.files.length > 0;
  button.disabled = !(hasName && hasPathSafeName && hasFile);
  if (state) {
    if (!hasName && !hasFile) state.textContent = 'Enter a name and choose an archive to enable upload.';
    else if (!hasName) state.textContent = 'Enter an adapter name to enable upload.';
    else if (!hasPathSafeName) state.textContent = pathSafeAdapterDirectoryNameMessage();
    else if (!hasFile) state.textContent = 'Choose a .tar.gz or .tgz archive to enable upload.';
    else if (uploadNameWasAutofilled && uploadName === lastAutofilledUploadName) state.textContent = 'Ready to upload with the auto-filled adapter name.';
    else state.textContent = 'Ready to upload.';
  }
}

window.uploadAdapter = async function() {
  const nameEl = document.getElementById('upload-name');
  const fileEl = document.getElementById('upload-archive');
  let name;
  try {
    name = parseAdapterNameField(nameEl);
  } catch (e) {
    toast(e.message, 'err');
    return;
  }
  if (!isPathSafeAdapterDirectoryName(name)) {
    nameEl.focus();
    toast(pathSafeAdapterDirectoryNameMessage(), 'err');
    updateUploadAdapterState();
    return;
  }
  const file = fileEl.files[0];
  if (!file) { fileEl.focus(); toast('Choose a .tar.gz or .tgz adapter archive', 'err'); return; }
  const lowerName = file.name.toLowerCase();
  if (!lowerName.endsWith('.tar.gz') && !lowerName.endsWith('.tgz')) {
    toast('Adapter upload expects a .tar.gz or .tgz archive', 'err');
    return;
  }
  const fd = new FormData();
  fd.append('name', name);
  fd.append('archive', file);
  const button = document.getElementById('upload-adapter-btn');
  const originalLabel = button ? button.textContent : '';
  uploadAdapterBusy = true;
  if (button) {
    button.disabled = true;
    button.textContent = 'Uploading…';
  }
  updateUploadAdapterState();
  try {
    // NOTE: do not set Content-Type — the browser sets the multipart boundary.
    const res = await fetch('/v1/adapters/upload', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    toast(`Uploaded ${data.name} (${fmtBytes(data.size_bytes)}, ${data.files} files)`);
    nameEl.value = '';
    fileEl.value = '';
    uploadNameWasAutofilled = false;
    lastAutofilledUploadName = '';
    updateUploadAdapterState();
    pollAdapters();
  } catch (e) { toast(e.message, 'err'); }
  finally {
    uploadAdapterBusy = false;
    if (button) button.textContent = originalLabel;
    updateUploadAdapterState();
  }
};

// --- Adapter Merge ---
let mergeSourceCount = 2;
let mergeAdaptersBusy = false;

function isPathSafeAdapterDirectoryName(name) {
  return Boolean(name)
    && name !== '.'
    && name !== '..'
    && !name.includes('/')
    && !name.includes('\\');
}

window.isPathSafeAdapterDirectoryName = isPathSafeAdapterDirectoryName;

function pathSafeAdapterDirectoryNameMessage() {
  return 'Name must be a single adapter directory name with no / or \\, and not . or ..';
}

function mergeReadinessState() {
  const adapterState = window.lastAdapters || lastAdapters;
  const available = (adapterState && adapterState.available) || [];
  if (available.length < 2) {
    return {
      ready: false,
      message: 'Merging requires at least two saved adapters. Create one with SFT/GRPO, or upload an adapter first.',
    };
  }
  if (mergeAdaptersBusy) {
    return { ready: false, message: 'Merging adapters…' };
  }

  // Source selection comes first: you can't name an output for a merge
  // that has no inputs, and the helper text reads more naturally when the
  // user is asked to pick sources before they're asked to name the result.
  const list = document.getElementById('merge-sources');
  const rows = list ? Array.from(list.querySelectorAll('.merge-source')) : [];
  const selected = [];
  for (const row of rows) {
    const name = row.querySelector('.merge-src-name')?.value.trim() || '';
    if (!name) continue;
    selected.push(name);
    const weightText = row.querySelector('.merge-src-weight')?.value || '';
    const weight = parseFloat(weightText);
    if (!Number.isFinite(weight)) {
      return { ready: false, message: `Enter a numeric weight for ${name}.` };
    }
  }
  if (selected.length < 2) {
    return { ready: false, message: 'Select at least two source adapters to enable merge.' };
  }
  if (new Set(selected).size !== selected.length) {
    return { ready: false, message: 'Choose distinct source adapters; duplicates cannot be merged.' };
  }

  const outputEl = document.getElementById('merge-output-name');
  const outputName = outputEl ? outputEl.value.trim() : '';
  if (!outputName) {
    return { ready: false, message: 'Enter a path-safe output adapter name to enable merge.' };
  }
  if (!isPathSafeAdapterDirectoryName(outputName)) {
    return { ready: false, message: 'Output name must be a single path-safe adapter name, not a path.' };
  }

  const mode = document.getElementById('merge-mode')?.value;
  if (mode === 'ties') {
    const density = parseFloat(document.getElementById('merge-density')?.value || '');
    if (!Number.isFinite(density) || density <= 0 || density > 1) {
      return { ready: false, message: 'TIES density must be a number in (0, 1].' };
    }
  }

  return { ready: true, message: 'Ready to merge the selected adapters into a new saved adapter.' };
}

function updateMergeButtonState() {
  const state = mergeReadinessState();
  const helper = document.getElementById('merge-helper');
  if (helper) helper.textContent = state.message;
  const mergeBtn = document.getElementById('merge-btn');
  if (mergeBtn) mergeBtn.disabled = !state.ready;
  const addBtn = document.getElementById('add-merge-source');
  if (addBtn) {
    const adapterState = window.lastAdapters || lastAdapters;
    const available = (adapterState && adapterState.available) || [];
    addBtn.disabled = available.length < 2 || mergeAdaptersBusy;
  }
  return state;
}

// Structural signature of the last merge-sources render. This function runs
// on every 5s adapters poll; rebuilding the rows when nothing changed would
// steal focus/caret from someone mid-typing a weight and snap open adapter
// selects shut. Rebuild only when the adapter set or the row count changes.
let lastMergeSourcesKey = null;
function renderMergeSources() {
  const list = document.getElementById('merge-sources');
  if (!list) return;
  const adapterState = window.lastAdapters || lastAdapters;
  const available = (adapterState && adapterState.available) || [];
  const canMerge = available.length >= 2;
  if (!canMerge) {
    lastMergeSourcesKey = null;
    if (list.firstChild) list.innerHTML = '';
    updateMergeButtonState();
    return;
  }
  const structureKey = available.map(a => a.name).join('|') + '::' + mergeSourceCount;
  if (structureKey === lastMergeSourcesKey && list.querySelector('.merge-source')) {
    updateMergeButtonState();
    return;
  }
  lastMergeSourcesKey = structureKey;
  // A structural rebuild is required — if the user is focused in one of our
  // inputs (e.g. an adapter was saved mid-edit), put them back afterwards.
  const active = document.activeElement;
  const restoreFocusId = active && list.contains(active) ? active.id : null;
  let restoreSelStart = null, restoreSelEnd = null;
  if (restoreFocusId) {
    try { restoreSelStart = active.selectionStart; restoreSelEnd = active.selectionEnd; } catch {}
  }
  // Preserve current values across re-renders.
  const existing = Array.from(list.querySelectorAll('.merge-source')).map(row => ({
    name: row.querySelector('.merge-src-name').value,
    weight: row.querySelector('.merge-src-weight').value,
  }));
  const adapterOptions = available
    .map(a => `<option value="${escapeHtml(a.name)}">${escapeHtml(a.name)}</option>`)
    .join('');
  let html = '';
  for (let i = 0; i < mergeSourceCount; i++) {
    const sel = existing[i] ? existing[i].name : '';
    const w = existing[i] ? existing[i].weight : '0.5';
    const rowNumber = i + 1;
    const nameId = `merge-src-name-${rowNumber}`;
    const weightId = `merge-src-weight-${rowNumber}`;
    html += `<div class="merge-source" style="display:grid;grid-template-columns:1fr 90px auto;gap:var(--space-2);margin-bottom:var(--space-2);align-items:center;">
      <select id="${nameId}" class="merge-src-name" aria-label="Merge source ${rowNumber} adapter"><option value="">(select adapter)</option>${adapterOptions}</select>
      <input id="${weightId}" type="number" class="merge-src-weight" step="0.05" value="${w}" aria-label="Merge source ${rowNumber} weight">
      <button type="button" class="btn btn-sm btn-danger" onclick="removeMergeSource(${i})" aria-label="Remove merge source ${rowNumber}" ${mergeSourceCount <= 2 ? 'disabled' : ''}>−</button>
    </div>`;
  }
  list.innerHTML = html;
  // Re-apply preserved selections after innerHTML replacement.
  Array.from(list.querySelectorAll('.merge-source')).forEach((row, i) => {
    if (existing[i]) row.querySelector('.merge-src-name').value = existing[i].name;
    row.querySelector('.merge-src-name').addEventListener('change', updateMergeButtonState);
    row.querySelector('.merge-src-weight').addEventListener('input', updateMergeButtonState);
  });
  if (restoreFocusId) {
    const el = document.getElementById(restoreFocusId);
    if (el) {
      el.focus();
      // setSelectionRange throws on <input type=number> in some browsers.
      try { if (restoreSelStart != null && el.setSelectionRange) el.setSelectionRange(restoreSelStart, restoreSelEnd); } catch {}
    }
  }
  updateMergeButtonState();
}

window.renderMergeSources = renderMergeSources;
window.updateMergeButtonState = updateMergeButtonState;
window.addMergeSource = function() { mergeSourceCount += 1; renderMergeSources(); updateMergeButtonState(); };
window.removeMergeSource = function(idx) {
  if (mergeSourceCount <= 2) return;
  // Drop the row at idx by reading current values, removing it, then re-rendering.
  const list = document.getElementById('merge-sources');
  const rows = Array.from(list.querySelectorAll('.merge-source'));
  const kept = rows.filter((_, i) => i !== idx).map(row => ({
    name: row.querySelector('.merge-src-name').value,
    weight: row.querySelector('.merge-src-weight').value,
  }));
  mergeSourceCount = Math.max(2, kept.length);
  renderMergeSources();
  // Re-apply preserved values to the freshly rendered rows.
  const newRows = list.querySelectorAll('.merge-source');
  kept.forEach((v, i) => {
    if (!newRows[i]) return;
    newRows[i].querySelector('.merge-src-name').value = v.name;
    newRows[i].querySelector('.merge-src-weight').value = v.weight;
  });
  updateMergeButtonState();
};

window.onMergeModeChange = function() {
  const mode = document.getElementById('merge-mode').value;
  const densityWrap = document.getElementById('merge-density-wrap');
  if (densityWrap) densityWrap.style.display = (mode === 'ties') ? '' : 'none';
  updateMergeButtonState();
};

window.mergeAdapters = async function() {
  const adapterState = window.lastAdapters || lastAdapters;
  const available = (adapterState && adapterState.available) || [];
  if (available.length < 2) {
    toast('Merging requires at least two saved adapters', 'err');
    return;
  }
  const list = document.getElementById('merge-sources');
  const rows = Array.from(list.querySelectorAll('.merge-source'));
  const sources = [];
  for (const row of rows) {
    const name = row.querySelector('.merge-src-name').value.trim();
    const weight = parseFloat(row.querySelector('.merge-src-weight').value);
    if (!name) continue;
    if (!Number.isFinite(weight)) { toast('Each merge source needs a numeric weight', 'err'); return; }
    sources.push({ name, weight });
  }
  if (sources.length < 2) { toast('Choose at least two source adapters to merge', 'err'); return; }
  if (new Set(sources.map(source => source.name)).size !== sources.length) {
    toast('Choose distinct source adapters to merge', 'err');
    return;
  }
  let outputName;
  try {
    outputName = parseAdapterNameField(document.getElementById('merge-output-name'));
  } catch (e) {
    toast(e.message, 'err');
    return;
  }
  const mode = document.getElementById('merge-mode').value;
  const body = { sources, output_name: outputName, mode };
  if (mode === 'ties') {
    const d = parseFloat(document.getElementById('merge-density').value);
    if (!Number.isFinite(d) || d <= 0 || d > 1) { toast('Density must be in (0, 1]', 'err'); return; }
    body.density = d;
  }
  const mergeBtn = document.getElementById('merge-btn');
  const originalLabel = mergeBtn ? mergeBtn.textContent : '';
  mergeAdaptersBusy = true;
  if (mergeBtn) {
    mergeBtn.textContent = 'Merging…';
  }
  updateMergeButtonState();
  try {
    const res = await api('/v1/adapters/merge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    toast(`Merged ${res.sources.length} sources → ${res.output_name} (${res.num_tensors} tensors, mode=${res.mode})`);
    pollAdapters();
  } catch (e) { toast(e.message, 'err'); }
  finally {
    mergeAdaptersBusy = false;
    if (mergeBtn) mergeBtn.textContent = originalLabel;
    renderMergeSources();
    updateMergeButtonState();
  }
};
