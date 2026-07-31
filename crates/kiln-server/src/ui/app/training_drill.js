
/* =====================================================================
   Training drill-in modal — full loss curve + linked evals + cancel
   ===================================================================== */

let trainDrillJobId = null;
let trainDrillPollHandle = null;
// Tracks the last (sample_count, state) tuple we rendered so we can skip
// the SVG/DOM rebuild when nothing meaningful has changed. The drill
// modal polls every 1.5s; for a finished job that's a 1024-sample
// loss_history shipping every poll otherwise.
let trainDrillLastKey = null;
// Loss samples of the currently drilled job, refreshed on every poll —
// the header's "Copy loss CSV" button reads this instead of re-fetching.
let trainDrillLossHistory = [];

const TRAIN_TERMINAL_STATES = new Set(['completed', 'failed', 'cancelled']);

async function openTrainDrillModal(jobId) {
  trainDrillJobId = jobId;
  modalHashOnOpen('train', '#training/queue/' + encodeURIComponent(jobId));
  trainDrillLastKey = null;
  const trainModal = document.getElementById('train-drill-modal');
  trainModal.hidden = false;
  openModal(trainModal, { onClose: userCloseTrainDrillModal });
  document.getElementById('train-drill-content').innerHTML = '<div class="detail-empty">Loading…</div>';
  await fetchTrainDrill();
  if (trainDrillPollHandle) clearInterval(trainDrillPollHandle);
  trainDrillPollHandle = setInterval(() => {
    if (!trainDrillJobId) return;
    fetchTrainDrill();
  }, 1500);
}

function closeTrainDrillModal() {
  trainDrillJobId = null;
  trainDrillLastKey = null;
  trainDrillLossHistory = [];
  const copyLossBtn = document.getElementById('train-drill-copy-loss');
  if (copyLossBtn) {
    copyLossBtn.disabled = true;
    copyLossBtn.title = 'No loss samples recorded yet — the CSV unlocks once training reports its first loss';
  }
  const trainModal = document.getElementById('train-drill-modal');
  trainModal.hidden = true;
  closeModal(trainModal);
  if (trainDrillPollHandle) { clearInterval(trainDrillPollHandle); trainDrillPollHandle = null; }
}
// User-initiated close (X / backdrop / Delete): walk history per the
// deep-link state machine. The linked-eval jump keeps calling
// closeTrainDrillModal directly — it navigates FORWARD to the eval drill,
// so Back should return here.
function userCloseTrainDrillModal() {
  modalHashOnUserClose('train', '#training/queue', closeTrainDrillModal);
}

async function fetchTrainDrill() {
  if (!trainDrillJobId) return;
  try {
    const j = await api('/v1/train/jobs/' + encodeURIComponent(trainDrillJobId));
    const stateLow = (j.state || '').toString().toLowerCase();
    const sampleCount = (j.loss_history || []).length;
    const key = `${stateLow}|${sampleCount}|${j.progress.toFixed(4)}`;
    if (key === trainDrillLastKey) {
      // No new sample, no state change: skip the DOM/SVG rebuild. Also
      // stop polling once the job has terminated — there's nothing more
      // to learn from a completed/failed/cancelled job.
      if (TRAIN_TERMINAL_STATES.has(stateLow) && trainDrillPollHandle) {
        clearInterval(trainDrillPollHandle);
        trainDrillPollHandle = null;
      }
      return;
    }
    trainDrillLastKey = key;

    document.getElementById('train-drill-title').textContent = j.adapter_name || 'Training job';
    document.getElementById('train-drill-meta').innerHTML =
      `<span class="job-state-pill ${stateLow}">${escapeHtml(stateLow)}</span>
       <span class="training-card-type ${(j.job_type||'').toString().toLowerCase()}" style="margin-left:8px;">${escapeHtml((j.job_type||'').toString())}</span>
       <span class="hint" style="margin-left:8px; font-family:var(--font-mono);">${escapeHtml(j.job_id)}</span>
       ${j.effective_seed == null ? '' : `<span class="hint tabular-nums" style="margin-left:8px; font-family:var(--font-mono);">seed ${escapeHtml(String(j.effective_seed))}</span>`}`;

    const stopBtn = document.getElementById('train-drill-stop');
    const deleteBtn = document.getElementById('train-drill-delete');
    if (stateLow === 'queued') {
      stopBtn.disabled = false;
      stopBtn.title = 'Cancel this queued job';
      stopBtn.hidden = false;
      if (deleteBtn) deleteBtn.hidden = true;
    } else if (stateLow === 'running') {
      // Running jobs are stoppable too — DELETE /v1/train/queue/{id} sets
      // the cooperative cancel flag and the trainer aborts at the next
      // step boundary. Same path as the queue card's Stop button.
      stopBtn.disabled = false;
      stopBtn.title = 'Stop at the next training step';
      stopBtn.hidden = false;
      if (deleteBtn) deleteBtn.hidden = true;
    } else {
      // Terminal (Completed / Failed) — hide Stop, show Delete instead.
      stopBtn.hidden = true;
      if (deleteBtn) {
        deleteBtn.hidden = false;
        deleteBtn.dataset.jobId = j.job_id;
      }
    }
    stopBtn.dataset.jobId = j.job_id;
    // The click handler words its confirm() by state (queued = removed
    // from queue immediately; running = cooperative stop at the next step).
    stopBtn.dataset.jobState = stateLow;

    // Copy loss CSV: enabled the moment the first loss sample lands.
    // Samples may be downsampled past TRAINING_LOSS_HISTORY_CAP, so the
    // CSV column is `sample` (recorded order), not a training step.
    trainDrillLossHistory = Array.isArray(j.loss_history) ? j.loss_history : [];
    const copyLossBtn = document.getElementById('train-drill-copy-loss');
    if (copyLossBtn) {
      copyLossBtn.disabled = trainDrillLossHistory.length === 0;
      copyLossBtn.title = trainDrillLossHistory.length
        ? `Copy ${trainDrillLossHistory.length} loss sample${trainDrillLossHistory.length === 1 ? '' : 's'} as CSV (sample,epoch,progress,loss,elapsed_secs)`
        : 'No loss samples recorded yet — the CSV unlocks once training reports its first loss';
    }

    document.getElementById('train-drill-content').innerHTML = renderTrainDrillBody(j);
    const curveEl = document.getElementById('train-drill-curve-host');
    if (curveEl && j.loss_history && j.loss_history.length >= 2) {
      const series = [{
        points: j.loss_history.map(s => [s.elapsed_secs, s.loss]),
        color: 'var(--accent)',
      }];
      renderLineChart(curveEl, series, { width: 800, height: 280, large: true });
    } else if (curveEl) {
      curveEl.innerHTML = `<div class="hint" style="padding:24px; text-align:center;">${stateLow === 'queued' ? 'Job hasn\'t started yet.' : (stateLow === 'running' ? 'Awaiting first loss sample…' : 'No loss history recorded.')}</div>`;
    }
    // Stop polling once the job is in a terminal state — `loss_history`
    // and `state` are now frozen; the modal can sit on the last render.
    if (TRAIN_TERMINAL_STATES.has(stateLow) && trainDrillPollHandle) {
      clearInterval(trainDrillPollHandle);
      trainDrillPollHandle = null;
    }
  } catch (e) {
    // Reset the drill key so the next successful poll repaints over this
    // error instead of being deduped away.
    trainDrillLastKey = null;
    document.getElementById('train-drill-content').innerHTML = `<div class="detail-empty">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

function drillValue(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    return Math.abs(value) >= 10000 ? value.toLocaleString() : String(value);
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

function drillOptimizerName(config, replayRequest) {
  const opt = config?.optimizer || replayRequest?.request_body?.config?.optimizer;
  const kind = typeof opt === 'string' ? opt : opt?.kind;
  if (kind === 'adam_w') return 'AdamW';
  if (kind === 'sgd') return 'SGD';
  if (kind === 'muon') return 'Muon';
  return '—';
}

function renderDrillKv(label, value) {
  return `<div class="req-stat"><span class="req-stat-k">${escapeHtml(label)}</span><span class="req-stat-v"><code>${escapeHtml(drillValue(value))}</code></span></div>`;
}

function renderTrainMetadata(j) {
  const receipt = j.train_receipt || null;
  const replay = j.replay_request || null;
  const hp = receipt?.hyperparameters || {};
  const data = receipt?.data || {};
  const source = receipt?.training_data || {};
  const admitted = j.training_data || {};
  const openenv = admitted.openenv || source.openenv || null;
  const openenvNames = Array.isArray(openenv?.environments)
    ? openenv.environments.map((environment) => environment.environment_name).join(', ')
    : null;
  const openenvTerminations = openenv?.terminations
    ? `done=${openenv.terminations.done || 0}, max_steps=${openenv.terminations.max_steps || 0}, invalid=${openenv.terminations.invalid_model_action || 0}, protocol=${openenv.terminations.protocol_error || 0}`
    : null;
  const openenvPolicy = openenv?.behavior_policy || null;
  const openenvPolicyLabel = openenvPolicy
    ? `${openenvPolicy.adapter?.name || 'base'} @ ${openenvPolicy.adapter?.content_sha256 || openenvPolicy.base_model_sha256}`
    : null;
  const config = replay?.request_body?.config || null;
  const rows = [
    renderDrillKv('Mode', hp.mode || replay?.kind || j.job_type),
    renderDrillKv('Optimizer', drillOptimizerName(config, replay)),
    renderDrillKv('Learning rate', hp.learning_rate ?? config?.learning_rate ?? 'auto'),
    renderDrillKv('Epochs', hp.epochs ?? config?.epochs),
    renderDrillKv('LoRA rank', hp.rank ?? config?.lora_rank),
    renderDrillKv('LoRA alpha', hp.alpha ?? config?.lora_alpha),
    renderDrillKv('Alpha / rank', hp.alpha_over_rank),
    renderDrillKv('Seed', hp.seed ?? replay?.seed),
    renderDrillKv('Examples trained', data.examples_trained),
    renderDrillKv('Groups trained', data.groups_trained),
    renderDrillKv('Completions trained', data.completions_trained),
    renderDrillKv('Data source', admitted.source || source.source || replay?.request_body?.dataset || replay?.request_body?.dataset_path),
    renderDrillKv('Dataset', admitted.dataset),
    renderDrillKv('Partition', admitted.split),
    renderDrillKv('Rows admitted', admitted.rows),
    renderDrillKv('Admitted corpus SHA-256', admitted.admitted_corpus_sha256),
    renderDrillKv('Split manifest SHA-256', admitted.split_manifest_sha256),
    renderDrillKv('Full dataset SHA-256', admitted.dataset_corpus_sha256),
    renderDrillKv('OpenEnv environments', openenvNames),
    renderDrillKv('OpenEnv groups / rollouts', openenv ? `${openenv.groups} / ${openenv.rollouts}` : null),
    renderDrillKv('OpenEnv seed range', openenv ? `${openenv.seed_min}–${openenv.seed_max} (${openenv.unique_seeds} unique)` : null),
    renderDrillKv('OpenEnv total steps', openenv?.total_steps),
    renderDrillKv('OpenEnv terminations', openenvTerminations),
    renderDrillKv('OpenEnv group plan SHA-256', openenv?.group_plan_sha256),
    renderDrillKv('OpenEnv behavior policy', openenvPolicyLabel),
    renderDrillKv('OpenEnv inference config SHA-256', openenvPolicy?.inference_config_sha256),
  ].join('');
  const receiptRaw = receipt
    ? `<details style="margin-top:12px;"><summary>Raw train receipt</summary><pre class="req-pre">${escapeHtml(JSON.stringify(receipt, null, 2))}</pre></details>`
    : '';
  const replayRaw = replay
    ? `<details style="margin-top:8px;"><summary>Replay request summary</summary><pre class="req-pre">${escapeHtml(JSON.stringify(replay, null, 2))}</pre></details>`
    : '';
  const error = j.metadata_error
    ? `<div class="training-card-error" style="margin-top:10px;">${icon('warning', 'icn-sm')} ${escapeHtml(j.metadata_error)}</div>`
    : '';
  const empty = !receipt && !replay && !j.training_data && !j.metadata_error
    ? '<div class="hint">No receipt or replay metadata was found for this job.</div>'
    : '';
  return `<div class="detail-section">
    <h4>Run metadata</h4>
    ${receipt || replay || j.training_data ? `<div class="req-stats" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));">${rows}</div>` : empty}
    ${error}
    ${receiptRaw}
    ${replayRaw}
  </div>`;
}

function checkpointTrainingKind(j, checkpoint) {
  const kind = String(checkpoint?.training_kind || j?.job_type || '').toLowerCase();
  return kind === 'sft' || kind === 'grpo' || kind === 'opd' ? kind : null;
}

function setTrainingFormValue(id, value) {
  const input = document.getElementById(id);
  if (!input) return;
  input.value = value === null || value === undefined ? '' : String(value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

function replayOptimizerKind(config) {
  const optimizer = config?.optimizer;
  const kind = typeof optimizer === 'string' ? optimizer : optimizer?.kind;
  return ['muon', 'adam_w', 'sgd'].includes(kind) ? kind : 'muon';
}

async function prepareTrainingResume(j, checkpoint) {
  const kind = checkpointTrainingKind(j, checkpoint);
  if (!kind) {
    toast('This checkpoint type cannot be resumed from the browser.', 'err');
    return;
  }

  if (kind === 'opd') {
    closeTrainDrillModal();
    selectPage('distill');
    document.getElementById('distill-tab-opd')?.click();

    const form = document.getElementById('opd-form');
    form?.reset();
    const config = checkpoint?.effective_config || {};
    setTrainingFormValue('opd-output-name', j.adapter_name || config.output_name || 'opd-adapter');
    setTrainingFormValue('opd-lr', Number.isFinite(config.learning_rate) ? config.learning_rate : '');
    setTrainingFormValue('opd-rank', Number.isInteger(config.lora_rank) && config.lora_rank > 0 ? config.lora_rank : 32);
    setTrainingFormValue('opd-loss', typeof config.loss === 'string' ? config.loss : 'teacher_top_k');
    setTrainingFormValue('opd-top-k', Number.isInteger(config.top_k) ? config.top_k : 16);
    setTrainingFormValue('opd-samples', Number.isInteger(config.samples_per_prompt) && config.samples_per_prompt > 0 ? config.samples_per_prompt : 4);
    setTrainingFormValue('opd-max-tokens', Number.isInteger(config.max_tokens) && config.max_tokens > 0 ? config.max_tokens : 7168);
    setTrainingFormValue('opd-temperature', Number.isFinite(config.temperature) ? config.temperature : 1);
    setTrainingFormValue('opd-top-p', Number.isFinite(config.top_p) ? config.top_p : 0.9);
    setTrainingFormValue(
      'opd-checkpoint-interval',
      Number.isInteger(config.checkpoint_interval) && config.checkpoint_interval > 0
        ? config.checkpoint_interval
        : '',
    );
    setTrainingFormValue('opd-resume-checkpoint', checkpoint.resume_checkpoint);
    const autoLoad = document.getElementById('opd-auto-load');
    if (autoLoad && typeof config.auto_load === 'boolean') autoLoad.checked = config.auto_load;
    const detectAnomaly = document.getElementById('opd-detect-anomaly');
    if (detectAnomaly) detectAnomaly.checked = config.detect_anomaly === true;
    setTrainingFormValue(
      'opd-sampler-segments',
      Number.isInteger(config.sampler_segments) && config.sampler_segments > 0
        ? config.sampler_segments
        : '',
    );
    setTrainingFormValue(
      'opd-rollout-prompt-rendering',
      config.rollout_prompt_rendering === 'chat_template'
        ? 'chat_template'
        : 'legacy_action_boundary',
    );
    setTrainingFormValue('opd-prompts', '');

    let teachers = [];
    try {
      teachers = (await api('/v1/teachers'))?.teachers || [];
      await refreshTeacherDropdowns(teachers);
    } catch { /* Submission still performs authoritative server validation. */ }
    const expectedTeacher = String(checkpoint.teacher_id || '');
    const currentTeacher = teachers.find(t => t.spec?.alias === expectedTeacher && t.usable === true);
    const teacherBound = Boolean(
      expectedTeacher
      && checkpoint.teacher_identity_revision
      && currentTeacher?.identity_revision === checkpoint.teacher_identity_revision,
    );
    if (teacherBound) {
      setTrainingFormValue('opd-teacher', expectedTeacher);
    } else {
      setTrainingFormValue('opd-teacher', '');
    }

    if (form) {
      form.dataset.resumeCheckpoint = checkpoint.resume_checkpoint || '';
      form.dataset.resumeTeacher = expectedTeacher;
      form.dataset.resumeTeacherRevision = checkpoint.teacher_identity_revision || '';
    }
    const note = document.getElementById('opd-resume-note');
    if (note) {
      const dataHash = String(checkpoint.data_content_sha256 || '').replace(/^sha256:/, '');
      const count = Number.isInteger(checkpoint.data_item_count) ? checkpoint.data_item_count : '?';
      note.hidden = false;
      note.className = `train-data-status ${teacherBound ? 'is-good' : 'is-bad'}`;
      note.textContent = teacherBound
        ? `Exact checkpoint loaded for ${count} training candidate${count === 1 ? '' : 's'} (data ${dataHash.slice(0, 12)}…). Reinsert the identical prompt array before submitting.`
        : `Checkpoint loaded, but its exact teacher ${expectedTeacher || 'identity'} is not currently registered. Restore that teacher revision and the identical prompt array before submitting.`;
    }
    document.getElementById('opd-prompts')?.focus();
    toast(teacherBound
      ? 'OPD checkpoint loaded — reinsert the exact original prompts before submitting.'
      : 'OPD checkpoint loaded, but its exact teacher is unavailable.',
      teacherBound ? undefined : 'err');
    return;
  }

  closeTrainDrillModal();
  selectPage('training');
  document.getElementById('training-tab-' + kind)?.click();

  // A replay summary does not retain inline rows. Clear every current source
  // before preparing the form so an unrelated paste or upload cannot be sent
  // with a valid-looking checkpoint basename.
  clearTrainingData(kind);
  const textarea = document.getElementById(TRAIN_KIND[kind].textareaId);
  if (textarea) textarea.value = '';
  const picker = document.getElementById(TRAIN_KIND[kind].pickId);
  if (picker) picker.value = '';

  const requestBody = j?.replay_request?.request_body || {};
  const config = requestBody.config || {};
  let restoredDataset = null;
  const recordedDataset = requestBody.dataset || j?.training_data?.dataset;
  if (typeof recordedDataset === 'string' && recordedDataset.trim()) {
    restoredDataset = recordedDataset.trim();
    await loadNamedDatasetIntoTraining(kind, restoredDataset);
  }

  setTrainingFormValue(kind + '-output-name', j.adapter_name || config.output_name || '');
  setTrainingFormValue(kind + '-learning-rate', Number.isFinite(config.learning_rate) ? config.learning_rate : '');
  setTrainingFormValue(kind + '-rank', Number.isInteger(config.lora_rank) && config.lora_rank > 0 ? config.lora_rank : 8);
  setTrainingFormValue(kind + '-optimizer', replayOptimizerKind(config));
  setTrainingFormValue(
    kind + '-checkpoint-interval',
    Number.isInteger(config.checkpoint_interval) && config.checkpoint_interval > 0
      ? config.checkpoint_interval
      : '',
  );
  setTrainingFormValue(kind + '-resume-checkpoint', checkpoint.resume_checkpoint);
  if (kind === 'sft') {
    setTrainingFormValue('sft-epochs', Number.isInteger(config.epochs) && config.epochs > 0 ? config.epochs : 3);
    setTrainingFormValue('sft-invalid-row-policy', config.invalid_row_policy === 'skip' ? 'skip' : 'fail');
  } else {
    setTrainingFormValue('grpo-kl-coeff', Number.isFinite(config.kl_coeff) ? config.kl_coeff : 0.1);
  }
  const autoLoad = document.getElementById(kind + '-auto-load');
  if (autoLoad && typeof config.auto_load === 'boolean') autoLoad.checked = config.auto_load;
  const detectAnomaly = document.getElementById(kind + '-detect-anomaly');
  if (detectAnomaly) detectAnomaly.checked = config.detect_anomaly === true;
  const smokeTest = document.getElementById(kind + '-adapter-smoke-test');
  if (smokeTest) smokeTest.checked = config.adapter_smoke_test === true;
  setTrainingFormValue(
    kind + '-adapter-smoke-prompts',
    Array.isArray(config.adapter_smoke_prompts) ? config.adapter_smoke_prompts.join('\n') : '',
  );
  if (kind === 'grpo') {
    const sharedPrefix = document.getElementById('grpo-shared-prefix-reference');
    if (sharedPrefix) sharedPrefix.checked = config.shared_prefix_reference !== false;
  }

  const advanced = document.getElementById(kind + '-advanced');
  const advancedToggle = document.getElementById(kind + '-adv-toggle');
  if (advanced && advanced.hidden) {
    advanced.hidden = false;
    advancedToggle?.setAttribute('aria-expanded', 'true');
  }
  TRAIN_KIND[kind].update();

  if (restoredDataset) {
    document.getElementById(kind + '-resume-checkpoint')?.focus();
    toast(`Checkpoint and dataset ${restoredDataset} loaded — review the settings, then submit.`, 'ok');
  } else {
    document.getElementById(kind + '-dropzone')?.focus();
    toast('Checkpoint loaded — re-select the exact original training data before submitting.');
  }
}

function renderTrainCheckpoint(j) {
  const checkpoint = j.latest_checkpoint || null;
  const error = j.checkpoint_error
    ? `<div class="training-card-error" style="margin-top:10px;">${icon('warning', 'icn-sm')} ${escapeHtml(j.checkpoint_error)}</div>`
    : '';
  if (!checkpoint) {
    return `<div class="detail-section">
      <h4>Resume checkpoint</h4>
      <div class="hint">None</div>
      ${error}
    </div>`;
  }
  const status = checkpoint.complete
    ? 'complete'
    : `step ${drillValue(checkpoint.global_step)} / ${drillValue(checkpoint.total_steps)}`;
  const kind = checkpointTrainingKind(j, checkpoint);
  const sourceKind = String(checkpoint.data_source_kind || '');
  const cursor = kind === 'grpo'
    ? `${sourceKind.startsWith('jsonl-') ? 'GRPO JSONL' : 'GRPO inline'} · next group cursor ${drillValue(checkpoint.next_cursor_in_epoch)}`
    : kind === 'opd'
      ? `OPD ${sourceKind || 'source'} · next candidate cursor ${drillValue(checkpoint.next_cursor_in_epoch)}`
      : `SFT · next epoch index ${drillValue(checkpoint.next_epoch_index)} · example cursor ${drillValue(checkpoint.next_cursor_in_epoch)}`;
  const state = String(j.state || '').toLowerCase();
  const prepareButton = kind && state !== 'queued' && state !== 'running'
    ? `<button class="btn btn-sm" type="button" data-prepare-training-resume title="Load this checkpoint and its recorded settings into the ${kind.toUpperCase()} form"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-arrow-right"></use></svg> Prepare resume</button>`
    : '';
  return `<div class="detail-section">
    <h4>Resume checkpoint</h4>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <code style="background:var(--surface); padding:4px 8px; border-radius:4px; border:1px solid var(--border); overflow-wrap:anywhere;">${escapeHtml(checkpoint.resume_checkpoint)}</code>
      <button class="btn btn-sm btn-ghost" type="button" data-copy-resume-checkpoint="${escapeHtml(checkpoint.resume_checkpoint)}" title="Copy resume checkpoint basename" aria-label="Copy resume checkpoint basename"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg></button>
      ${prepareButton}
      <span class="hint tabular-nums">${escapeHtml(status)} · ${escapeHtml(cursor)}</span>
    </div>
    ${error}
  </div>`;
}

function renderTrainDrillBody(j) {
  const linkedIds = j.linked_eval_job_ids || [];
  const linkedHtml = linkedIds.length
    ? linkedIds.map(id => `<button class="btn btn-sm" type="button" data-linked-eval="${escapeHtml(id)}"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-arrow-right"></use></svg> Eval ${escapeHtml(id.slice(0, 8))}</button>`).join(' ')
    : '<span class="hint">None</span>';
  const finalLoss = j.current_loss != null ? j.current_loss.toFixed(4) : '—';
  const epoch = j.epoch != null ? j.epoch.toString() : '—';
  const samples = (j.loss_history || []).length;
  // Prefer the on-disk wall-clock fields when present — `elapsed_secs`
  // is wrong for archived jobs because the in-memory `Instant` reset
  // when we restored from disk.
  let durationSecs = j.elapsed_secs;
  if (j.submitted_unix_ms && j.finished_unix_ms) {
    durationSecs = Math.max(0, (j.finished_unix_ms - j.submitted_unix_ms) / 1000);
  } else if (j.submitted_unix_ms) {
    durationSecs = Math.max(0, (Date.now() - j.submitted_unix_ms) / 1000);
  }
  const timeRow = (j.submitted_unix_ms || j.finished_unix_ms)
    ? `<div style="margin-top:6px; font-size:11px; color:var(--text-muted);">
        ${j.submitted_unix_ms ? `submitted ${escapeHtml(fmtSmartTime(j.submitted_unix_ms))}` : ''}
        ${j.finished_unix_ms ? ` · finished ${escapeHtml(fmtSmartTime(j.finished_unix_ms))}` : ''}
      </div>`
    : '';
  const seedSection = j.effective_seed == null ? '' : `<div class="detail-section">
    <h4>Effective seed</h4>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <code style="background:var(--surface); padding:4px 8px; border-radius:4px; border:1px solid var(--border);">${escapeHtml(String(j.effective_seed))}</code>
      <button class="btn btn-sm btn-ghost" type="button" data-copy-training-seed="${escapeHtml(String(j.effective_seed))}" title="Copy exact effective seed" aria-label="Copy exact effective seed"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg></button>
      <span class="hint">Materialized before queue publication</span>
    </div>
  </div>`;
  const baseWeightManifest = j.train_receipt?.model?.base_weight_shard_manifest
    || j.adapter_manifest?.base_weight_shard_manifest
    || null;
  const baseWeightSection = baseWeightManifest == null ? '' : `<div class="detail-section">
    <h4>Base weights</h4>
    ${renderBaseWeightSummary(baseWeightManifest)}
  </div>`;
  const executionProvenance = j.train_receipt?.runtime?.execution_provenance
    || j.adapter_manifest?.execution_provenance
    || null;
  const trainingPrecision = j.train_receipt?.runtime?.training_precision
    || j.adapter_manifest?.training_precision
    || null;
  const executionSection = executionProvenance == null && trainingPrecision == null ? '' : `<div class="detail-section">
    <h4>Execution and precision</h4>
    ${renderExecutionProvenanceSummary(executionProvenance)}
    ${renderTrainingPrecisionSummary(trainingPrecision)}
  </div>`;
  const html = `<div style="padding: var(--space-4) var(--space-5); border-bottom:1px solid var(--border);">
    <div style="display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
      <div><div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">Progress</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${(j.progress*100).toFixed(0)}%</div></div>
      <div><div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">${j.state === 'completed' || j.state === 'failed' ? 'Final loss' : 'Current loss'}</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${finalLoss}</div></div>
      <div><div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">Epoch</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${epoch}</div></div>
      <div><div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">Duration</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${fmtDuration(durationSecs)}</div></div>
      <div><div class="hint" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">Samples</div>
        <div style="font-size:18px; font-weight:600;" class="tabular-nums">${samples}</div></div>
    </div>
    ${timeRow}
  </div>
  <div class="detail-section">
    <h4>Loss curve</h4>
    <div id="train-drill-curve-host"></div>
  </div>
  ${seedSection}
  ${baseWeightSection}
  ${executionSection}
  ${renderTrainMetadata(j)}
  ${renderTrainCheckpoint(j)}
  <div class="detail-section">
    <h4>Adapter</h4>
    <div style="display:flex; gap:8px; align-items:center;">
      <code style="background:var(--surface); padding:4px 8px; border-radius:4px; border:1px solid var(--border);">${escapeHtml(j.adapter_name || '—')}</code>
      ${j.adapter_path ? `<span class="hint" style="font-family:var(--font-mono); font-size:11px;">${escapeHtml(j.adapter_path)}</span>` : ''}
      <span class="hint">${j.auto_load ? 'auto-load on completion' : ''}</span>
    </div>
  </div>
  <div class="detail-section">
    <h4>Linked evals</h4>
    ${linkedHtml}
  </div>`;
  // Defer wiring to after innerHTML set
  setTimeout(() => {
    wireBaseWeightCopy(document.getElementById('train-drill-content'));
    wireExecutionProvenanceCopy(document.getElementById('train-drill-content'));
    document.querySelectorAll('[data-linked-eval]').forEach(b => {
      b.addEventListener('click', () => {
        closeTrainDrillModal();
        selectPage('evals');
        document.getElementById('evals-tab-jobs')?.click();
        openDrillModal(b.dataset.linkedEval);
      });
    });
    document.querySelectorAll('[data-copy-resume-checkpoint]').forEach(b => {
      b.addEventListener('click', () => {
        const value = b.dataset.copyResumeCheckpoint;
        if (!value) return;
        const writeText = navigator.clipboard?.writeText
          ? navigator.clipboard.writeText.bind(navigator.clipboard)
          : (text) => { fallbackCopyText(text); return Promise.resolve(); };
        writeText(value)
          .then(() => {
            if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = value;
            toast('Resume checkpoint copied', 'ok');
          })
          .catch(() => {
            try { fallbackCopyText(value); toast('Resume checkpoint copied', 'ok'); }
            catch { toast('Copy failed', 'err'); }
          });
      });
    });
    document.querySelectorAll('[data-copy-training-seed]').forEach(b => {
      b.addEventListener('click', () => {
        const value = b.dataset.copyTrainingSeed;
        if (!value) return;
        const writeText = navigator.clipboard?.writeText
          ? navigator.clipboard.writeText.bind(navigator.clipboard)
          : (text) => { fallbackCopyText(text); return Promise.resolve(); };
        writeText(value)
          .then(() => {
            if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = value;
            toast('Effective seed copied', 'ok');
          })
          .catch(() => {
            try { fallbackCopyText(value); toast('Effective seed copied', 'ok'); }
            catch { toast('Copy failed', 'err'); }
          });
      });
    });
    document.querySelectorAll('[data-prepare-training-resume]').forEach(b => {
      b.addEventListener('click', () => {
        prepareTrainingResume(j, j.latest_checkpoint).catch(error => {
          toast('Could not prepare resume: ' + error.message, 'err');
        });
      });
    });
  }, 0);
  return html;
}

document.getElementById('train-drill-close')?.addEventListener('click', userCloseTrainDrillModal);
document.getElementById('train-drill-modal')?.addEventListener('click', ev => {
  if (ev.target.id === 'train-drill-modal') userCloseTrainDrillModal();
});
document.getElementById('train-drill-stop')?.addEventListener('click', async () => {
  const stopBtn = document.getElementById('train-drill-stop');
  const jobId = stopBtn.dataset.jobId;
  if (!jobId) return;
  const running = stopBtn.dataset.jobState === 'running';
  const msg = running
    ? 'Stop this running job at the next training step?'
    : 'Cancel queued job?';
  if (!confirm(msg)) return;
  // Reuse the in-flight set + toast + pollTraining refresh that
  // window.cancelJob already implements; calling DELETE directly
  // would let rapid clicks fire duplicate requests. Keep the modal
  // OPEN until the DELETE resolves — a failure surfaces right here
  // instead of in a closed modal, and on success the 1.5s drill poll
  // repaints the state to cancelled on its own.
  trainDrillLastKey = null; // bypass change-detection so the repaint lands
  await window.cancelJob(jobId);
});
document.getElementById('train-drill-delete')?.addEventListener('click', async () => {
  const jobId = document.getElementById('train-drill-delete').dataset.jobId;
  if (!jobId) return;
  if (!confirm('Permanently delete this training job? The adapter weights on disk are untouched; only the tracking entry and the on-disk archive file are removed.')) return;
  try {
    await api('/v1/train/jobs/' + encodeURIComponent(jobId), { method: 'DELETE' });
    toast('Training job deleted', 'ok');
    userCloseTrainDrillModal();
    lastTrainingKey = null; // bypass change-detection so re-render happens
    pollTraining();
  } catch (e) {
    toast('Delete failed: ' + e.message, 'err');
  }
});
// Copy loss history (CSV) — `sample` is the recorded order (the in-memory
// history downsamples past 512 points, so it is NOT the optimizer step).
// Loss samples carry no wall-clock timestamps; elapsed_secs is the offset
// from job start.
document.getElementById('train-drill-copy-loss')?.addEventListener('click', () => {
  if (!trainDrillLossHistory.length) return;
  const csv = ['sample,epoch,progress,loss,elapsed_secs']
    .concat(trainDrillLossHistory.map((s, i) => `${i + 1},${s.epoch},${s.progress},${s.loss},${s.elapsed_secs}`))
    .join('\n');
  const writeText = navigator.clipboard?.writeText
    ? navigator.clipboard.writeText.bind(navigator.clipboard)
    : (t) => { fallbackCopyText(t); return Promise.resolve(); };
  writeText(csv).then(() => {
    if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = csv;
    toast('Loss history copied as CSV', 'ok');
  }).catch(() => {
    try { fallbackCopyText(csv); toast('Loss history copied as CSV', 'ok'); }
    catch { toast('Copy failed', 'err'); }
  });
});
