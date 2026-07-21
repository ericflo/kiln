
/* =====================================================================
   Charts: line/sparkline/donut renderers
   ===================================================================== */

/// Render a multi-axis line chart of (x, y) pairs into the given container.
/// `series` is an array of {points: [[x, y], ...], color, label}. Auto-scales
/// X linearly between min/max; Y linearly between 0 and max(y) with a small
/// headroom. Suitable for loss curves and tok/s timelines.
let lineChartSeq = 0;
function renderLineChart(container, series, opts = {}) {
  const w = opts.width || 600;
  const h = opts.height || 280;
  const padL = 40, padR = 12, padT = 12, padB = 24;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;
  // Single-pass min/max: avoids `Math.min/max(...xs)` which would spread
  // every sample as a function arg. With training-loss curves capped at
  // 1024 samples we're still within engine limits today, but the spread
  // pattern crashes around ~125k args on Chrome — better to never use it.
  let xMin = Infinity, xMax = -Infinity, yMaxRaw = 0, yMinRaw = Infinity, count = 0;
  for (const s of series) {
    for (const p of (s.points || [])) {
      const x = p[0], y = p[1];
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (isFinite(y)) { if (y > yMaxRaw) yMaxRaw = y; if (y < yMinRaw) yMinRaw = y; }
      count++;
    }
  }
  if (count < 2) {
    container.innerHTML = `<div class="hint" style="padding:12px; text-align:center;">Awaiting first samples…</div>`;
    return;
  }
  // Default: baseline at 0. opts.yZoom auto-scales to the data range (+padding)
  // so a steady-but-live series (e.g. tok/s hovering ~145) reads as a living
  // trend instead of a dead-flat line pinned to the top of a 0-based axis.
  let yMin = 0;
  let yMax = yMaxRaw <= 0 ? 1 : yMaxRaw * 1.1;
  if (opts.yZoom && isFinite(yMinRaw) && yMaxRaw > yMinRaw) {
    const pad = (yMaxRaw - yMinRaw) * 0.35 || 1;
    yMin = Math.max(0, yMinRaw - pad);
    yMax = yMaxRaw + pad;
  }
  const xRange = (xMax - xMin) || 1;
  const yRange = (yMax - yMin) || 1;
  const xx = x => padL + ((x - xMin) / xRange) * innerW;
  const yy = y => padT + innerH - ((y - yMin) / yRange) * innerH;
  // Y gridlines at 0/25/50/75/100% of range.
  const grid = [];
  for (let i = 0; i <= 4; i++) {
    const yVal = yMin + (yRange * i / 4);
    const yPx = yy(yVal);
    grid.push(`<line class="grid" x1="${padL}" y1="${yPx.toFixed(1)}" x2="${(padL+innerW).toFixed(1)}" y2="${yPx.toFixed(1)}" />`);
    grid.push(`<text class="axis-label" x="${padL - 4}" y="${(yPx + 3).toFixed(1)}" text-anchor="end">${yVal.toFixed(yVal < 1 ? 2 : (yVal < 10 ? 1 : 0))}</text>`);
  }
  // X axis
  const xAxisLabels = [
    `<text class="axis-label" x="${padL}" y="${(h - 6).toFixed(1)}" text-anchor="start">${xMin.toFixed(0)}s</text>`,
    `<text class="axis-label" x="${(padL + innerW).toFixed(1)}" y="${(h - 6).toFixed(1)}" text-anchor="end">${xMax.toFixed(0)}s</text>`,
  ];
  // Series paths — the area uses a vertical gradient that fades to
  // transparent at the baseline, so even a flat/constant series reads as a
  // soft glow under the line rather than a solid block.
  const cid = 'lc' + (++lineChartSeq);
  const defs = [];
  const seriesHtml = series.map((s, idx) => {
    const color = s.color || ['var(--accent)', 'var(--info-fg)', 'var(--success-fg)', 'var(--warning-fg)'][idx % 4];
    const pts = s.points || [];
    if (pts.length < 2) return '';
    const gid = `${cid}-a${idx}`;
    defs.push(`<linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:${color};stop-opacity:0.26"/><stop offset="0.85" style="stop-color:${color};stop-opacity:0.02"/><stop offset="1" style="stop-color:${color};stop-opacity:0"/></linearGradient>`);
    const linePath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xx(p[0]).toFixed(1)} ${yy(p[1]).toFixed(1)}`).join(' ');
    const areaPath = `${linePath} L${xx(pts[pts.length-1][0]).toFixed(1)} ${(padT+innerH).toFixed(1)} L${xx(pts[0][0]).toFixed(1)} ${(padT+innerH).toFixed(1)} Z`;
    return `<path class="data-area" d="${areaPath}" style="fill: url(#${gid});"/>
            <path class="data-line" d="${linePath}" style="stroke: ${color};"/>`;
  }).join('');
  container.innerHTML = `<svg class="line-chart ${opts.large ? 'line-chart-large' : ''}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
    <defs>${defs.join('')}</defs>
    ${grid.join('')}
    <line class="axis" x1="${padL}" y1="${padT}" x2="${padL}" y2="${(padT+innerH).toFixed(1)}" />
    <line class="axis" x1="${padL}" y1="${(padT+innerH).toFixed(1)}" x2="${(padL+innerW).toFixed(1)}" y2="${(padT+innerH).toFixed(1)}" />
    ${xAxisLabels.join('')}
    ${seriesHtml}
  </svg>`;
}

/// Render a donut chart representing memory or any partition into slices.
/// `slices` is [{label, value, color}]. Returns SVG markup as a string.
function donutChartSvg(slices, opts = {}) {
  const size = opts.size || 110;
  const stroke = opts.stroke || 18;
  const r = (size - stroke) / 2;
  const c = size / 2;
  const total = slices.reduce((s, sl) => s + sl.value, 0);
  if (total <= 0) {
    return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="var(--surface-3)" stroke-width="${stroke}"/></svg>`;
  }
  let offset = 0;
  const C = 2 * Math.PI * r;
  const segs = slices.map(sl => {
    const portion = sl.value / total;
    const dash = portion * C;
    const seg = `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${sl.color}" stroke-width="${stroke}"
      stroke-dasharray="${dash.toFixed(2)} ${(C - dash).toFixed(2)}"
      stroke-dashoffset="${(-offset).toFixed(2)}"
      transform="rotate(-90 ${c} ${c})"/>`;
    offset += dash;
    return seg;
  }).join('');
  const center = opts.centerLabel
    ? `<text x="${c}" y="${c - 2}" text-anchor="middle" style="fill:var(--text); font-weight:700; font-size:14px; font-variant-numeric:tabular-nums;">${escapeHtml(opts.centerLabel)}</text>
       <text x="${c}" y="${c + 12}" text-anchor="middle" style="fill:var(--text-muted); font-size:9px; text-transform:uppercase; letter-spacing: var(--tracking-caps);">${escapeHtml(opts.centerSub || '')}</text>`
    : '';
  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">
    <circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="var(--surface-3)" stroke-width="${stroke}"/>
    ${segs}
    ${center}
  </svg>`;
}

/* =====================================================================
   Overview: tok/s sparkline + quick actions
   ===================================================================== */

const tpsHistory = [];
const TPS_HISTORY_CAP = 60;

// Real elapsed span of the tok/s history in seconds, derived from the stored
// sample timestamps. Samples arrive once per poll tick (~2s), NOT once per
// second, so counting entries would understate the window by ~2x. Each sample
// represents one whole poll interval, so the span counts N intervals (last-to-
// first delta plus one average gap); snapping to a 5s grid beyond 10s keeps
// the label from flickering with poll-timer jitter. Returns null until two
// samples exist (no honest span to claim yet).
function decodeSparkSpanSecs(history) {
  if (!history || history.length < 2) return null;
  const spanMs = history[history.length - 1].ts - history[0].ts;
  if (!(spanMs > 0)) return null;
  const avgGapMs = spanMs / (history.length - 1);
  const secs = Math.round((spanMs + avgGapMs) / 1000);
  return secs >= 10 ? Math.round(secs / 5) * 5 : Math.max(secs, 1);
}

// Decode-perf sparkline. Driven from the end of `pollDecodePerf` so we
// share the upstream fetch and never issue a second `/v1/stats/decode`
// request. A change-detection guard skips the SVG repaint when tok/s is
// unchanged (idle server), avoiding a layout reflow every 2s for nothing.
let lastTpsRendered = null;
function refreshDecodeSparkline() {
  const data = lastDecode;
  if (!data || typeof data.tok_per_sec !== 'number') return;
  const tps = data.tok_per_sec;
  // Always advance the sliding window so the visualised range stays
  // anchored to "now"; only short-circuit the SVG repaint when the value
  // hasn't changed and the buffer is full enough to look stable.
  tpsHistory.push({ ts: Date.now(), tps });
  while (tpsHistory.length > TPS_HISTORY_CAP) tpsHistory.shift();
  if (tps === lastTpsRendered && tpsHistory.length >= TPS_HISTORY_CAP) return;
  lastTpsRendered = tps;
  const panel = document.getElementById('decode-perf-panel');
  if (!panel) return;
  let spark = panel.querySelector('.decode-spark-host');
  if (!spark) {
    spark = document.createElement('div');
    spark.className = 'decode-spark-host';
    spark.style.marginTop = '12px';
    spark.style.paddingTop = '8px';
    spark.style.borderTop = '1px solid var(--border)';
    const header = document.createElement('div');
    header.className = 'hint';
    header.style.fontSize = '11px';
    header.style.marginBottom = '4px';
    spark.appendChild(header);
    const body = document.createElement('div');
    body.className = 'decode-spark-body';
    spark.appendChild(body);
    panel.appendChild(spark);
  }
  let peakTps = 0;
  for (const s of tpsHistory) if (s.tps > peakTps) peakTps = s.tps;
  const spanSecs = decodeSparkSpanSecs(tpsHistory);
  spark.firstChild.innerHTML = `${spanSecs != null ? `tok/s over the last ${spanSecs}s` : 'tok/s'} · peak <span class="tabular-nums" style="color:var(--text-2);">${peakTps.toFixed(0)}</span> · now <span class="tabular-nums" style="color:var(--text-2);">${tps.toFixed(1)}</span>`;
  const series = [{ points: tpsHistory.map((s, i) => [i, s.tps]), color: 'var(--accent)' }];
  renderLineChart(spark.querySelector('.decode-spark-body'), series, { width: 520, height: 100, yZoom: true });
}

// The VRAM donut renders inside `renderServerStatus` — the server-status card
// has exactly one writer. (A second writer appending the donut here used to
// race the card repaint and the donut vanished on the second poll.)

// The sparkline refresher is driven event-style from the bottom of
// `pollDecodePerf` (success path). No need for a standalone interval —
// the poll is already on the right cadence.

const QUICK_ACTIONS = {
  'new-eval':   () => { selectPage('evals');    document.getElementById('evals-tab-suites')?.click(); },
  'train-sft':  () => { selectPage('training'); document.getElementById('training-tab-sft')?.click(); },
  'judge':      () => { selectPage('evals');    document.getElementById('evals-tab-judgments')?.click(); },
  'playground': () => { selectPage('playground'); },
};
document.querySelectorAll('[data-quick-action]').forEach(btn => {
  btn.addEventListener('click', () => QUICK_ACTIONS[btn.dataset.quickAction]?.());
});
