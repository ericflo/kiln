
// --- Preflight (/v1/preflight/*) ------------------------------------
async function refreshPreflightSurfaces() {
  const compatNode = document.getElementById('preflight-compat-list');
  const tierNode = document.getElementById('preflight-tier-defaults');
  try {
    const compat = await api('/v1/preflight/compatibility');
    const rows = compat.matches || [];
    compatNode.innerHTML = rows.length === 0
      ? `<div class="empty">${escapeHtml(compat.note || 'No rows.')}</div>`
      : `<div style="overflow-x:auto;"><table style="width:100%; border-collapse:collapse; font-size:var(--text-xs);">
          <thead><tr style="text-align:left; color:var(--text-muted);">
            <th style="padding:var(--space-2);">Teacher</th><th style="padding:var(--space-2);">Student</th><th style="padding:var(--space-2);">Domain</th><th style="padding:var(--space-2);">Init overlap</th><th style="padding:var(--space-2);">Rank</th><th style="padding:var(--space-2);">GPU-hr</th><th style="padding:var(--space-2);">$</th><th style="padding:var(--space-2);">Eval</th>
          </tr></thead>
          <tbody>${rows.map(r => `<tr style="border-top:1px solid var(--border);">
            <td style="padding:var(--space-2);">${escapeHtml(r.teacher)}</td>
            <td style="padding:var(--space-2);">${escapeHtml(r.student)}</td>
            <td style="padding:var(--space-2);">${escapeHtml(r.domain)}</td>
            <td style="padding:var(--space-2);">${(r.predicted_initial_overlap || 0).toFixed(2)}</td>
            <td style="padding:var(--space-2);">${r.recommended_rank}</td>
            <td style="padding:var(--space-2);">${(r.expected_gpu_hours || 0).toFixed(1)}</td>
            <td style="padding:var(--space-2);">${r.expected_cost_usd != null ? '$' + r.expected_cost_usd.toFixed(2) : '—'}</td>
            <td style="padding:var(--space-2);">${escapeHtml(r.validation_eval || '')}</td>
          </tr>`).join('')}</tbody>
        </table></div>`;
  } catch (e) {
    compatNode.innerHTML = `<div class="empty">Failed: ${escapeHtml(e.message)}</div>`;
  }
  try {
    const res = await api('/v1/preflight/tiers');
    const tiers = res.tiers || [];
    tierNode.innerHTML = tiers.length === 0
      ? '<div class="empty">No tiers configured.</div>'
      : tiers.map(t => `<div class="adapter-card" style="margin-bottom:var(--space-2);">
          <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div style="font-weight:600; text-transform:capitalize;">${escapeHtml(t.tier)}</div>
            <div style="font-size:var(--text-xs); color:var(--text-muted);">${escapeHtml(t.default_logit_source || '')}</div>
          </div>
          <div style="font-size:var(--text-xs); color:var(--text-muted); margin-top:var(--space-1);">rank ${t.lora_rank} · top-K ${t.default_top_k} · loss ${escapeHtml(t.default_loss || '')} · batch ${t.batch_size}</div>
          <div style="font-size:var(--text-xs); color:var(--text-muted);">cost cap ${t.cost_cap_default_usd == null ? '—' : '$' + t.cost_cap_default_usd.toFixed(0)} · max rollout ${(t.max_rollout_tokens || 0).toLocaleString()} tok · checkpoint every ${t.auto_checkpoint_cadence_steps} steps</div>
          <div style="font-size:var(--text-2xs); color:var(--text-muted); margin-top:var(--space-1);">samples/prompt: ${t.samples_per_prompt_default} (data-multiplier: ${t.samples_per_prompt_data_multiplier}) · cold-start ≥ ${t.cold_start_overlap_threshold} · goldens ${(t.mixture_distillation_golden_fraction * 100).toFixed(0)}%</div>
        </div>`).join('');
  } catch (e) {
    tierNode.innerHTML = `<div class="empty">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

// Helper used by cache stats (small + dep-free; no risk of name clash).
function formatBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB';
  return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}
