
/* =====================================================================
   Distill page — §3 + §10.6 on-policy distillation workflows
   ===================================================================== */

// Sub-tab activation for the Distill page mirrors the evals/training
// pattern: selecting a `.tab[data-tab="X"]` inside `[data-distill-tabs]`
// hides every `.tab-content` and shows the one with id
// `distill-tab-X-pane`. Click + keyboard (arrow/Home/End) wiring comes
// from the shared wireTablist helper, which skips the decorative
// group-label/separator spans (they carry no role=tab).
(function wireDistillTabs() {
  const root = document.querySelector('[data-distill-tabs]');
  if (!root) return;
  function selectDistillTab(btn) {
    const wanted = btn.dataset.tab;
    root.querySelectorAll('.tab').forEach(t => {
      const active = t.dataset.tab === wanted;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', String(active));
      t.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll('#page-distill .tab-content').forEach(p => {
      const active = p.id === `distill-tab-${wanted}-pane`;
      p.classList.toggle('active', active);
      p.hidden = !active;
      if (active) p.removeAttribute('inert'); else p.setAttribute('inert', '');
    });
    refreshActiveDistillSubTab();
    try { localStorage.setItem('kiln.distill.lastTab', wanted); } catch {}
    // Deep-link hash for the sub-tab (no-op for hash-driven activation and
    // for the suppressed localStorage restore below).
    pushSubTabHash('distill');
  }
  wireTablist(root, { onSelect: selectDistillTab });
  // Restore the last-used sub-tab. Hash-suppressed: the no-hash fallback —
  // an explicit hash sub-tab is applied after this in the boot route pass.
  try {
    const last = localStorage.getItem('kiln.distill.lastTab');
    if (last) {
      const btn = root.querySelector(`button.tab[data-tab="${last}"]`);
      if (btn) withHashWritesSuppressed(() => btn.click());
    }
  } catch {}
})();

// Click-handler for inline distill-tab cross-links in form help text
// (e.g. "register one first" → Teachers tab).
document.addEventListener('click', (ev) => {
  const link = ev.target.closest('[data-distill-tab-link]');
  if (!link) return;
  ev.preventDefault();
  const tab = link.getAttribute('data-distill-tab-link');
  const btn = document.querySelector(`[data-distill-tabs] button.tab[data-tab="${tab}"]`);
  if (btn) btn.click();
});

function refreshActiveDistillSubTab() {
  const root = document.querySelector('[data-distill-tabs]');
  if (!root) return;
  const active = root.querySelector('.tab.active')?.dataset?.tab || 'opd';
  if (active === 'opd' || active === 'refresh' || active === 'pump') {
    refreshTeacherDropdowns();
  } else if (active === 'teachers') {
    refreshTeachersList();
  } else if (active === 'recipes') {
    refreshRecipesList();
  } else if (active === 'cache') {
    refreshCacheStats();
  } else if (active === 'library') {
    refreshLibraryList();
  } else if (active === 'traces') {
    refreshAgentTraces();
  } else if (active === 'runs') {
    refreshAgentRuns();
  } else if (active === 'preflight') {
    refreshPreflightSurfaces();
  }
}

// --- Teachers (/v1/teachers) ----------------------------------------
async function refreshTeachersList() {
  const node = document.getElementById('teachers-list');
  if (!node) return;
  try {
    const res = await api('/v1/teachers');
    const teachers = res.teachers || [];
    if (teachers.length === 0) {
      node.innerHTML = '<div class="empty">No teachers registered. Add one below.</div>';
    } else {
      const rows = teachers.map(t => {
        const identity = t.spec?.identity;
        const caps = t.capabilities
          ? `${t.capabilities.max_top_k || '?'} top-K · ${(t.capabilities.vocab_size || 0).toLocaleString()} vocab`
          : 'capabilities unavailable';
        const bounds = identity
          ? ` · ${identity.max_model_len.toLocaleString()} context · ${identity.max_prompt_logprob_candidates.toLocaleString()} candidates`
          : '';
        const adapter = t.spec?.adapter ? ` · adapter ${escapeHtml(t.spec.adapter)}` : '';
        const statusClass = t.usable ? 'completed' : 'failed';
        const status = String(t.status || (t.usable ? 'configured' : 'unavailable')).replaceAll('_', ' ');
        const revision = t.identity_revision
          ? `<span class="hint" title="${escapeHtml(t.identity_revision)}">revision ${escapeHtml(t.identity_revision.replace('sha256:', '').slice(0, 12))}</span>`
          : '';
        const problem = t.status_message
          ? `<div class="training-card-error">${icon('warning', 'icn-sm')} ${escapeHtml(t.status_message)}</div>`
          : '';
        return `<div class="adapter-card" style="display:flex; align-items:center; gap:var(--space-3);">
          <div style="flex:1; min-width:0;">
            <div style="display:flex; align-items:center; gap:var(--space-2); font-weight:600;">${escapeHtml(t.spec?.alias || '?')}<span class="job-state-pill ${statusClass}">${escapeHtml(status)}</span></div>
            <div style="font-size:var(--text-xs); color:var(--text-muted);">${escapeHtml(t.spec?.kind || '?')} · ${escapeHtml(t.spec?.model_id || '?')}${adapter}</div>
            <div style="font-size:var(--text-xs); color:var(--text-muted);">${escapeHtml(caps)}${escapeHtml(bounds)}${revision ? ' · ' + revision : ''}</div>
            ${problem}
          </div>
          <button class="btn btn-sm" data-teacher-delete="${escapeHtml(t.spec?.alias || '')}">Delete</button>
        </div>`;
      }).join('');
      node.innerHTML = rows;
    }
    refreshTeacherDropdowns(teachers);
  } catch (e) {
    node.innerHTML = `<div class="empty">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

async function refreshTeacherDropdowns(prefetched) {
  let teachers;
  try {
    if (prefetched) teachers = prefetched;
    else teachers = (await api('/v1/teachers'))?.teachers || [];
  } catch { return; }
  const selectors = ['#opd-teacher', '[data-distill-teacher-select]'];
  for (const sel of selectors) {
    document.querySelectorAll(sel).forEach(node => {
      const prev = node.value;
      const opts = ['<option value="">— pick a registered teacher —</option>'];
      for (const t of teachers) {
        const alias = t.spec?.alias || '';
        if (!alias || t.usable !== true) continue;
        opts.push(`<option value="${escapeHtml(alias)}">${escapeHtml(alias)}</option>`);
      }
      node.innerHTML = opts.join('');
      if (prev && teachers.some(t => t.spec?.alias === prev && t.usable === true)) node.value = prev;
    });
  }
}

document.addEventListener('click', async (ev) => {
  const btn = ev.target.closest('[data-teacher-delete]');
  if (!btn) return;
  const alias = btn.getAttribute('data-teacher-delete');
  if (!alias || !confirm(`Delete teacher "${alias}"?`)) return;
  try {
    await api('/v1/teachers/' + encodeURIComponent(alias), { method: 'DELETE' });
    toast(`Deleted teacher ${alias}`);
    refreshTeachersList();
  } catch (e) { toast('Delete failed: ' + e.message, 'err'); }
});

document.querySelectorAll('#teacher-form select[name="kind"]').forEach(select => {
  const sync = () => {
    document.querySelectorAll('#teacher-form [data-teacher-kind-field]').forEach(node => {
      node.hidden = node.getAttribute('data-teacher-kind-field') !== select.value;
    });
    const url = document.getElementById('teacher-url');
    if (url) url.required = select.value === 'remote';
  };
  select.addEventListener('change', sync);
  sync();
});

document.getElementById('teacher-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const submit = document.getElementById('teacher-register-btn');
  const status = document.getElementById('teacher-register-status');
  const originalLabel = submit?.textContent || 'Register teacher';
  if (submit) {
    submit.disabled = true;
    submit.textContent = form.kind.value === 'remote' ? 'Verifying teacher…' : 'Registering…';
  }
  if (status) status.textContent = form.kind.value === 'remote' ? 'Running identity and capability probes…' : '';
  try {
    const body = {
      alias: form.alias.value.trim(),
      kind: form.kind.value,
      model_id: form.model_id.value.trim(),
    };
    if (body.kind === 'remote') {
      body.provider = 'vllm';
      body.url = form.url.value.trim();
      if (form.credential_id.value.trim()) body.credential_id = form.credential_id.value.trim();
    }
    if (body.kind === 'local' && form.adapter.value.trim()) body.adapter = form.adapter.value.trim();
    const registered = await api('/v1/teachers', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(`Registered teacher ${body.alias}`);
    if (status) {
      const revision = registered.identity_revision ? ` · ${registered.identity_revision.replace('sha256:', '').slice(0, 12)}` : '';
      status.textContent = `${registered.status || 'configured'}${revision}`;
    }
    form.reset();
    form.kind.dispatchEvent(new Event('change'));
    refreshTeachersList();
  } catch (err) {
    if (status) status.textContent = err.message;
    toast('Register failed: ' + err.message, 'err');
  } finally {
    if (submit) {
      submit.disabled = false;
      submit.textContent = originalLabel;
    }
  }
});

// --- Recipes (/v1/recipes + /v1/recipes/run) ------------------------
function recipeRunAdmissionState(button) {
  if (!trainingOptimizerSupportSnapshot) {
    return { ready: false, reason: trainingOptimizerSupportUnavailableReason };
  }
  if (button?.dataset.recipeAdmissionSupported !== 'true') {
    return {
      ready: false,
      reason: button?.dataset.recipeAdmissionReason || 'The recipe response is missing a supported admission descriptor',
    };
  }
  return { ready: true, reason: null };
}

function applyRecipeAdmissionButtons() {
  document.querySelectorAll('[data-recipe-run]').forEach(button => {
    const state = recipeRunAdmissionState(button);
    const busy = button.dataset.recipeBusy === 'true';
    button.disabled = busy || !state.ready;
    button.title = state.ready ? '' : state.reason;
    const statusId = button.getAttribute('aria-describedby');
    const status = statusId ? document.getElementById(statusId) : null;
    if (status) {
      status.textContent = state.ready
        ? 'Available for the resident training path.'
        : `${state.reason}. Recipe execution remains disabled.`;
    }
  });
}

async function refreshRecipesList() {
  const node = document.getElementById('recipes-list');
  if (!node) return;
  try {
    const res = await api('/v1/recipes');
    const recipes = Array.isArray(res?.recipes) ? res.recipes : [];
    if (recipes.length === 0) {
      node.innerHTML = '<div class="empty">No bundled recipes.</div>';
      return;
    }
    node.innerHTML = recipes.map((r, index) => {
      const hasName = typeof r?.name === 'string' && r.name.trim().length > 0;
      const admissionSupported = hasName && r?.admission?.supported === true;
      const admissionReason = admissionSupported
        ? ''
        : !hasName
          ? 'The server returned a recipe without a valid name'
          : r?.admission?.unavailable_reason || 'The server did not provide a supported recipe admission descriptor';
      const statusId = `recipe-admission-${index}`;
      return `<div class="adapter-card" style="display:flex; align-items:center; gap:var(--space-3); margin-bottom:var(--space-2);">
      <div style="flex:1; min-width:0;">
        <div style="font-weight:600;">${escapeHtml(hasName ? r.name : 'Invalid recipe')}</div>
        <div style="font-size:var(--text-xs); color:var(--text-muted);">${escapeHtml(r.description || '')}</div>
        <div style="font-size:var(--text-2xs); color:var(--text-muted); margin-top:var(--space-1);">${r.num_steps || 0} step${(r.num_steps || 0) === 1 ? '' : 's'}</div>
        <div class="form-help" id="${statusId}" role="status" aria-live="polite"></div>
      </div>
      <button class="btn btn-sm" data-recipe-run="${escapeHtml(hasName ? r.name : '')}" data-recipe-admission-supported="${admissionSupported}" data-recipe-admission-reason="${escapeHtml(admissionReason)}" aria-describedby="${statusId}" disabled>Run</button>
    </div>`;
    }).join('');
    applyRecipeAdmissionButtons();
  } catch (e) {
    node.innerHTML = `<div class="empty">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

document.addEventListener('click', async (ev) => {
  const btn = ev.target.closest('[data-recipe-run]');
  if (!btn) return;
  const name = btn.getAttribute('data-recipe-run');
  if (!name) return;
  const admission = recipeRunAdmissionState(btn);
  if (!admission.ready) {
    toast(`Recipe ${name} cannot run: ${admission.reason}.`, 'err');
    return;
  }
  try {
    btn.dataset.recipeBusy = 'true';
    btn.disabled = true; btn.textContent = 'Queuing…';
    const res = await api('/v1/recipes/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ recipe: name }) });
    const seedCount = Object.keys(res.effective_seeds || {}).length;
    toast(`${res.message || `Queued recipe ${name}`}${seedCount ? ` · ${seedCount} effective seed${seedCount === 1 ? '' : 's'} recorded` : ''}`, 'ok');
  } catch (e) { toast('Run failed: ' + e.message, 'err'); }
  finally {
    btn.dataset.recipeBusy = 'false';
    btn.textContent = 'Run';
    applyRecipeAdmissionButtons();
  }
});

// --- Submit OPD (/v1/train/opd) -------------------------------------
document.getElementById('opd-use-sample')?.addEventListener('click', () => {
  document.getElementById('opd-prompts').value = JSON.stringify([
    { messages: [{ role: 'user', content: 'Solve for x: 2x^2 - 5x + 3 = 0.' }, { role: 'assistant', content: 'Use the quadratic formula: x = (5 ± √(25 - 24)) / 4 = (5 ± 1)/4, so x = 3/2 or x = 1.' }] },
    { messages: [{ role: 'user', content: 'What is the derivative of sin(x^2)?' }, { role: 'assistant', content: 'Chain rule: d/dx sin(x²) = cos(x²) · 2x.' }] },
  ], null, 2);
});

document.getElementById('opd-resume-checkpoint')?.addEventListener('input', (event) => {
  const form = document.getElementById('opd-form');
  if (!form || event.target.value.trim() === form.dataset.resumeCheckpoint) return;
  const note = document.getElementById('opd-resume-note');
  if (note) note.hidden = true;
});

document.getElementById('opd-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    requireTrainingOptimizerAdmission('opd', 'muon', form.lora_rank.value, 'OPD');
    const opdRank = parsePositiveIntegerField(form.lora_rank.value, 'OPD LoRA rank');
    const outputName = parsePathSafeAdapterNameField(form.output_name);
    const promptsText = document.getElementById('opd-prompts').value.trim();
    const prompts = promptsText ? JSON.parse(promptsText) : [];
    if (!Array.isArray(prompts) || prompts.length === 0) {
      throw new Error('Prompts must be a non-empty JSON array');
    }
    const teacher = document.getElementById('opd-teacher').value;
    if (!teacher) throw new Error('Pick a teacher first (Teachers tab)');
    const opdLearningRate = parseOptionalFiniteNumberField(
      document.getElementById('opd-lr').value, 'Learning rate');
    const checkpointInterval = parseOptionalPositiveIntegerField(
      form.checkpoint_interval.value, 'OPD checkpoint interval');
    const resumeCheckpoint = parseResumeCheckpointField(
      form.resume_checkpoint.value, 'OPD resume checkpoint');
    if (resumeCheckpoint && resumeCheckpoint === form.dataset.resumeCheckpoint) {
      const expectedTeacher = form.dataset.resumeTeacher;
      const expectedRevision = form.dataset.resumeTeacherRevision;
      if (!expectedTeacher || !expectedRevision) {
        throw new Error('This OPD checkpoint does not expose an exact teacher identity and revision, so it cannot be prepared safely in the browser.');
      }
      if (teacher !== expectedTeacher) {
        throw new Error(`OPD resume requires the checkpoint teacher ${expectedTeacher}.`);
      }
      const teachers = (await api('/v1/teachers'))?.teachers || [];
      const current = teachers.find(t => t.spec?.alias === teacher && t.usable === true);
      if (!current || current.identity_revision !== expectedRevision) {
        throw new Error('OPD resume requires the exact teacher revision recorded by the checkpoint. Restore or re-register that teacher before submitting.');
      }
    }
    const body = {
      prompts,
      teacher,
      config: {
        output_name: outputName,
        loss: document.getElementById('opd-loss').value,
        top_k: parseInt(document.getElementById('opd-top-k').value, 10),
        samples_per_prompt: parseInt(document.getElementById('opd-samples').value, 10),
        lora_rank: opdRank,
        lora_alpha: loraAlphaFor(opdRank),
        optimizer: { kind: 'muon' },
        max_tokens: parseInt(document.getElementById('opd-max-tokens').value, 10),
        temperature: parseFloat(document.getElementById('opd-temperature').value),
        top_p: parseFloat(document.getElementById('opd-top-p').value),
        training_mode: 'on_policy',
        objective: 'reverse_kl',
        stable_opd: { mode: 'off' },
        discount: 0,
        clip_epsilon: 0,
        auto_load: document.getElementById('opd-auto-load').checked,
      },
    };
    // Blank lr is omitted so the server resolves the per-optimizer default.
    if (opdLearningRate !== null) body.config.learning_rate = opdLearningRate;
    if (checkpointInterval !== null) body.config.checkpoint_interval = checkpointInterval;
    if (resumeCheckpoint !== null) body.config.resume_checkpoint = resumeCheckpoint;
    if (form.detect_anomaly.checked) body.config.detect_anomaly = true;
    const samplerSegments = parseOptionalPositiveIntegerField(
      form.sampler_segments.value, 'OPD sampler segments');
    if (samplerSegments !== null) body.config.sampler_segments = samplerSegments;
    body.config.rollout_prompt_rendering = form.rollout_prompt_rendering.value;
    setTrainingSubmitBusy(form, true, 'Submitting OPD…');
    const res = await api('/v1/train/opd', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toastTrainingSubmission(res, 'Distillation job queued');
    selectPage('training');
    document.getElementById('training-tab-queue')?.click();
    pollTraining();
  } catch (err) { toast(err.message, 'err'); }
  finally { setTrainingSubmitBusy(form, false, 'Submitting OPD…'); }
});

// --- Distill / Refresh (/v1/distill/refresh) ------------------------
document.getElementById('distill-refresh-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    requireTrainingOptimizerAdmission('distill_refresh', 'muon', 16, 'Distill refresh');
    const examplesText = document.getElementById('refresh-new-data').value.trim();
    const examples = examplesText ? JSON.parse(examplesText) : [];
    if (!Array.isArray(examples) || examples.length === 0) {
      throw new Error('new_data must be a non-empty JSON array');
    }
    const body = {
      name: form.name.value.trim(),
      new_data: { examples },
      behavioural_teacher: form.behavioural_teacher.value,
      background_chat: form.background_chat.value.trim() || 'tulu3',
      require_if_eval_recovery: parseFloat(form.require_if_eval_recovery.value),
      require_internal_qa_gain: parseFloat(form.require_internal_qa_gain.value),
      config: { optimizer: { kind: 'muon' }, lora_rank: 16, lora_alpha: 32 },
    };
    if (form.if_eval_suite.value.trim()) body.if_eval_suite = form.if_eval_suite.value.trim();
    if (form.new_knowledge_eval_suite.value.trim()) body.new_knowledge_eval_suite = form.new_knowledge_eval_suite.value.trim();
    const res = await api('/v1/distill/refresh', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toastTrainingSubmission(res, 'Refresh queued');
    selectPage('training');
  } catch (err) { toast(err.message, 'err'); }
});

// --- Distill / Pump (/v1/distill/pump) ------------------------------
document.querySelectorAll('#distill-pump-form select[name="mode"]').forEach(sel => {
  const sync = () => {
    const wanted = sel.value;
    document.querySelectorAll('#distill-pump-form [data-pump-mode-field]').forEach(node => {
      node.hidden = node.getAttribute('data-pump-mode-field') !== wanted;
    });
  };
  sel.addEventListener('change', sync);
  sync();
});

document.getElementById('distill-pump-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    requireTrainingOptimizerAdmission('opd', 'muon', form.rank.value, 'Boost');
    const rank = parsePositiveIntegerField(form.rank.value, 'Boost LoRA rank');
    const mode = form.mode.value;
    let modeBody;
    if (mode === 'domain') modeBody = { domain: form.domain.value.trim() };
    else if (mode === 'wide') modeBody = { wide: true };
    else if (mode === 'examples') {
      const text = document.getElementById('pump-examples').value.trim();
      const examples = text ? JSON.parse(text) : [];
      if (!Array.isArray(examples) || examples.length === 0) throw new Error('Inline examples must be a non-empty JSON array');
      modeBody = { examples };
    }
    const body = {
      name: form.name.value.trim(),
      teacher: form.teacher.value,
      mode: modeBody,
      rank,
      rollout_budget: parseInt(form.rollout_budget.value, 10),
      use_cache: form.use_cache.checked,
      config: { optimizer: { kind: 'muon' }, lora_rank: rank, lora_alpha: loraAlphaFor(rank) },
    };
    const res = await api('/v1/distill/pump', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toastTrainingSubmission(res, 'Boost job queued');
    selectPage('training');
  } catch (err) { toast(err.message, 'err'); }
});

// --- Distill / Merge (/v1/adapters/distill_merge) -------------------
document.getElementById('distill-merge-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    requireTrainingOptimizerAdmission('opd', 'muon', 16, 'Distill merge');
    const sources = JSON.parse(form.sources.value);
    if (!Array.isArray(sources) || sources.length === 0) throw new Error('sources must be a non-empty JSON array');
    const body = {
      name: form.name.value.trim(),
      sources,
      student: form.student.value.trim() || 'base',
      rollout_budget: parseInt(form.rollout_budget.value, 10),
      config: { training_mode: 'off_policy', optimizer: { kind: 'muon' }, lora_rank: 16, lora_alpha: 32 },
    };
    const res = await api('/v1/adapters/distill_merge', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toastTrainingSubmission(res, 'Merge queued');
    selectPage('training');
  } catch (err) { toast(err.message, 'err'); }
});

// --- Distill / Self (/v1/distill/self) ------------------------------
document.querySelectorAll('#distill-self-form select[name="mode"]').forEach(sel => {
  const sync = () => {
    const requiresContext = sel.value === 'ground_truth_conditioning' || sel.value === 'document_as_pi';
    const prompts = document.getElementById('self-prompts');
    const groundTruth = document.getElementById('self-ground-truth');
    const documents = document.getElementById('self-documents');
    if (prompts) prompts.required = true;
    if (groundTruth) groundTruth.required = sel.value === 'ground_truth_conditioning';
    if (documents) documents.required = sel.value === 'document_as_pi';
    const promptsLabel = document.getElementById('self-prompts-label');
    const promptsHelp = document.getElementById('self-prompts-help');
    if (promptsLabel) promptsLabel.textContent = 'Prompts with assistant actions (JSON array)';
    if (promptsHelp) promptsHelp.textContent = requiresContext
      ? 'Required. Every prompt needs an assistant action, and the context array must have the same number of entries.'
      : 'Required. Every prompt needs a non-empty assistant action for the privileged teacher to rescore.';
    document.querySelectorAll('#distill-self-form [data-self-mode-field]').forEach(node => {
      node.hidden = node.getAttribute('data-self-mode-field') !== sel.value;
    });
  };
  sel.addEventListener('change', sync);
  sync();
});

document.getElementById('distill-self-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    requireTrainingOptimizerAdmission('opd', 'muon', 16, 'Self-improvement');
    const body = {
      name: form.name.value.trim(),
      mode: form.mode.value,
      config: { training_mode: 'off_policy', optimizer: { kind: 'muon' }, lora_rank: 16, lora_alpha: 32 },
    };
    const prompts = JSON.parse(form.prompts.value);
    if (!Array.isArray(prompts) || prompts.length === 0) throw new Error('Prompts must be a non-empty JSON array');
    body.prompts = prompts;
    const gt = document.getElementById('self-ground-truth')?.value?.trim();
    if (gt) body.ground_truth = JSON.parse(gt);
    const docs = document.getElementById('self-documents')?.value?.trim();
    if (docs) body.documents = JSON.parse(docs);
    const res = await api('/v1/distill/self', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toastTrainingSubmission(res, 'Self-improvement job queued');
    selectPage('training');
  } catch (err) { toast(err.message, 'err'); }
});

// --- Cache (/v1/cache/{stats,export}) -------------------------------
async function refreshCacheStats() {
  const node = document.getElementById('cache-stats');
  if (!node) return;
  try {
    const res = await api('/v1/cache/stats');
    const stats = res.stats || {};
    const teachers = Object.entries(stats.per_teacher || {});
    const perTeacherHtml = teachers.length
      ? `<div style="margin-top: var(--space-3);"><div class="form-help" style="margin-bottom: var(--space-2);">Per-teacher entries</div>
          ${teachers.map(([k, n]) => `<div style="display:flex; justify-content:space-between; font-size:var(--text-xs); padding: var(--space-1) 0;"><span>${escapeHtml(k)}</span><span style="font-variant-numeric: tabular-nums;">${n.toLocaleString()}</span></div>`).join('')}
        </div>`
      : '';
    node.innerHTML = `<div class="stat-grid" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-3);">
        <div class="stat-card" style="padding: var(--space-3); border: 1px solid var(--border); border-radius: var(--radius-md);"><div class="stat-label" style="font-size: var(--text-xs); color: var(--text-muted);">Total entries</div><div class="stat-val" style="font-size: var(--text-2xl); font-weight: 600;">${(stats.total_entries ?? 0).toLocaleString()}</div></div>
        <div class="stat-card" style="padding: var(--space-3); border: 1px solid var(--border); border-radius: var(--radius-md);"><div class="stat-label" style="font-size: var(--text-xs); color: var(--text-muted);">Size on disk</div><div class="stat-val" style="font-size: var(--text-2xl); font-weight: 600;">${formatBytes(stats.total_bytes || 0)}</div></div>
        <div class="stat-card" style="padding: var(--space-3); border: 1px solid var(--border); border-radius: var(--radius-md);"><div class="stat-label" style="font-size: var(--text-xs); color: var(--text-muted);">Teachers</div><div class="stat-val" style="font-size: var(--text-2xl); font-weight: 600;">${teachers.length}</div></div>
      </div>
      <div class="form-help" style="margin-top: var(--space-3);">Cache root: <code>${escapeHtml(res.root || '')}</code></div>
      ${perTeacherHtml}`;
  } catch (e) {
    node.innerHTML = `<div class="empty">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

document.getElementById('cache-export-btn')?.addEventListener('click', () => {
  window.location.href = '/v1/cache/export';
});

// --- Library (/v1/library) ------------------------------------------
async function refreshLibraryList() {
  const node = document.getElementById('library-list');
  if (!node) return;
  try {
    const res = await api('/v1/library');
    const adapters = res.adapters || [];
    if (adapters.length === 0) {
      node.innerHTML = '<div class="empty">No published adapters yet.</div>';
      return;
    }
    node.innerHTML = adapters.map(a => `<div class="adapter-card" style="display:flex; align-items:center; gap:var(--space-3); margin-bottom:var(--space-2);">
      <div style="flex:1; min-width:0;">
        <div style="font-weight:600;">${escapeHtml(a.name || a.id || '?')}</div>
        <div style="font-size:var(--text-xs); color:var(--text-muted);">${escapeHtml(a.source_kind || '')}${a.description ? ' · ' + escapeHtml(a.description) : ''}</div>
      </div>
      <button class="btn btn-sm" data-library-install="${escapeHtml(a.id || a.name || '')}">Install</button>
    </div>`).join('');
    const note = res.note;
    if (note) {
      const noteEl = document.createElement('div');
      noteEl.className = 'empty';
      noteEl.style.cssText = 'margin-top: var(--space-3); font-size: var(--text-xs);';
      noteEl.textContent = note;
      node.appendChild(noteEl);
    }
  } catch (e) {
    node.innerHTML = `<div class="empty">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

document.addEventListener('click', async (ev) => {
  const btn = ev.target.closest('[data-library-install]');
  if (!btn) return;
  const id = btn.getAttribute('data-library-install');
  if (!id) return;
  try {
    btn.disabled = true; btn.textContent = 'Installing…';
    await api('/v1/library/install/' + encodeURIComponent(id), { method: 'POST' });
    toast(`Installed ${id}`);
    pollAdapters();
  } catch (err) { toast('Install failed: ' + err.message, 'err'); }
  finally { btn.disabled = false; btn.textContent = 'Install'; }
});

document.getElementById('library-publish-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    const name = form.adapter_name.value.trim();
    if (!name) throw new Error('Adapter name required');
    const body = {};
    if (form.description.value.trim()) body.description = form.description.value.trim();
    if (form.uploader.value.trim()) body.uploader = form.uploader.value.trim();
    const res = await api('/v1/library/publish/' + encodeURIComponent(name), {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    toast(res.status === 'ready_to_publish' ? `Publish prepared for ${name} (${res.intended_id})` : `Published ${name}`);
    refreshLibraryList();
  } catch (err) { toast('Publish failed: ' + err.message, 'err'); }
});
