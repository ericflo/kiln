
/* =====================================================================
   pi Terminal — embedded, pre-configured pi over a PTY WebSocket.
   The server runs the `kiln pi-setup` merge before spawning, so this pi
   is the user's pi, already pointed at this Kiln. xterm.js renders; its
   assets are vendored into the binary and lazy-loaded on first launch.
   ===================================================================== */
let termInstance = null;   // xterm Terminal
let termSocket = null;     // WebSocket
let termFit = null;        // FitAddon
let termAssetsLoaded = false;

function termStateEl() { return document.getElementById('terminal-state'); }

async function loadTerminalAssets() {
  if (termAssetsLoaded) return true;
  try {
    await new Promise((resolve, reject) => {
      const link = document.createElement('link');
      link.rel = 'stylesheet'; link.href = './vendor/xterm.css';
      link.onload = resolve; link.onerror = reject;
      document.head.appendChild(link);
    });
    const loadScript = src => new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src; s.onload = resolve; s.onerror = reject;
      document.head.appendChild(s);
    });
    await loadScript('./vendor/xterm.js');
    await loadScript('./vendor/xterm-addon-fit.js');
    termAssetsLoaded = !!window.Terminal;
    return termAssetsLoaded;
  } catch (_) { return false; }
}

async function initTerminalPage() {
  const state = termStateEl();
  if (!state) return;
  if (termSocket && termSocket.readyState === WebSocket.OPEN) return; // session live
  state.innerHTML = '<div class="empty">Checking terminal availability…</div>';
  let st = null;
  try { st = await api('/v1/terminal/status'); } catch (e) {
    state.innerHTML = `<div class="empty">${escapeHtml(e.message)}</div>`;
    return;
  }
  setText('terminal-cwd', st.cwd ? ('runs in ' + st.cwd) : '');
  if (!st.enabled) {
    state.innerHTML = `<div class="eval-empty">
      <div class="eval-empty-icon"><svg class="icn"><use href="#i-terminal"></use></svg></div>
      <div class="eval-empty-title">Terminal disabled</div>
      <div class="eval-empty-body">${escapeHtml(st.disabled_reason || 'Not available on this server.')}</div>
    </div>`;
    return;
  }
  if (!st.pi_available) {
    state.innerHTML = `<div class="eval-empty">
      <div class="eval-empty-icon"><svg class="icn"><use href="#i-terminal"></use></svg></div>
      <div class="eval-empty-title">pi isn't installed on the server</div>
      <div class="eval-empty-body">Install the <strong>pi</strong> coding agent on the machine running <code>kiln serve</code>, then come back — Kiln configures it automatically. See the <a href="https://ericflo.github.io/kiln/quickstart.html" target="_blank" rel="noopener">Quickstart</a> for the pi setup walkthrough.</div>
    </div>`;
    return;
  }
  state.innerHTML = `<div class="eval-empty">
    <div class="eval-empty-icon"><svg class="icn"><use href="#i-terminal"></use></svg></div>
    <div class="eval-empty-title">Launch pi in your browser</div>
    <div class="eval-empty-body">Kiln merges its connection into pi's config (the same non-destructive merge as <code>kiln pi-setup</code> — your other providers are kept), then starts a real <code>pi</code> session in <code>${escapeHtml(st.cwd || '.')}</code>. Everything it does flows through this Kiln — watch it land on the Overview.</div>
    <button class="eval-empty-cta" type="button" id="terminal-launch">Launch pi</button>
  </div>`;
  document.getElementById('terminal-launch')?.addEventListener('click', launchTerminal);
}

async function launchTerminal() {
  const state = termStateEl();
  const host = document.getElementById('terminal-host');
  if (!state || !host) return;
  state.innerHTML = '<div class="empty">Loading terminal…</div>';
  if (!(await loadTerminalAssets())) {
    state.innerHTML = '<div class="empty">Could not load the terminal renderer (xterm). Is the server up to date?</div>';
    return;
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const sock = new WebSocket(`${proto}//${window.location.host}/v1/terminal/ws`);
  sock.binaryType = 'arraybuffer';
  termSocket = sock;

  sock.onopen = () => {
    state.hidden = true;
    host.hidden = false;
    if (!termInstance) {
      termInstance = new window.Terminal({
        fontFamily: "'SF Mono', 'JetBrains Mono', 'Cascadia Code', Consolas, monospace",
        fontSize: 13,
        cursorBlink: true,
        theme: {
          background: '#0a0908', foreground: '#f7f4ef',
          cursor: '#f97316', cursorAccent: '#0a0908',
          selectionBackground: 'rgba(249,115,22,0.30)',
        },
      });
      termFit = new window.FitAddon.FitAddon();
      termInstance.loadAddon(termFit);
      termInstance.open(host);
      termInstance.onData(data => {
        if (termSocket && termSocket.readyState === WebSocket.OPEN) {
          termSocket.send(new TextEncoder().encode(data));
        }
      });
      new ResizeObserver(() => {
        if (!termFit) return;
        try { termFit.fit(); } catch (_) {}
        if (termSocket && termSocket.readyState === WebSocket.OPEN && termInstance) {
          termSocket.send(JSON.stringify({ type: 'resize', cols: termInstance.cols, rows: termInstance.rows }));
        }
      }).observe(host);
    } else {
      termInstance.reset();
    }
    setTimeout(() => {
      try { termFit.fit(); } catch (_) {}
      if (termInstance) sock.send(JSON.stringify({ type: 'resize', cols: termInstance.cols, rows: termInstance.rows }));
      termInstance.focus();
    }, 60);
    const restart = document.getElementById('terminal-restart');
    if (restart) restart.hidden = false;
  };
  sock.onmessage = ev => {
    if (typeof ev.data === 'string') {
      try {
        const m = JSON.parse(ev.data);
        if (m.type === 'ready') {
          setText('terminal-cwd', 'runs in ' + (m.cwd || '.'));
          toast('pi is live — configured for ' + (m.kiln_url || 'this Kiln'), 'ok');
        } else if (m.type === 'exit') {
          if (termInstance) termInstance.write('\r\n\x1b[90m— pi exited. Restart session to relaunch. —\x1b[0m\r\n');
        } else if (m.type === 'error') {
          if (termInstance) termInstance.write(`\r\n\x1b[31m${m.message || 'terminal error'}\x1b[0m\r\n`);
          else { const s = termStateEl(); if (s) { s.hidden = false; s.innerHTML = `<div class="empty">${escapeHtml(m.message || 'terminal error')}</div>`; } }
        }
      } catch (_) {}
      return;
    }
    if (termInstance) termInstance.write(new Uint8Array(ev.data));
  };
  sock.onclose = () => {
    if (termInstance) termInstance.write('\r\n\x1b[90m— session closed —\x1b[0m\r\n');
    termSocket = null;
  };
  sock.onerror = () => {
    const s = termStateEl();
    if (s && host.hidden) { s.innerHTML = '<div class="empty">Could not open the terminal socket.</div>'; }
  };
}

document.getElementById('terminal-restart')?.addEventListener('click', () => {
  if (termSocket) { try { termSocket.close(); } catch (_) {} termSocket = null; }
  launchTerminal();
});

// Surface "try pi right here" on the Connect panel + journey step when the
// embedded terminal is actually usable on this server (live, gated-on, pi
// installed). One status fetch at boot; quiet no-op otherwise.
async function revealTerminalShortcuts() {
  try {
    const st = await api('/v1/terminal/status');
    if (!st || !st.enabled || !st.pi_available) return;
    const btn = document.getElementById('connect-try-pi');
    if (btn) {
      btn.hidden = false;
      btn.addEventListener('click', () => selectPage('terminal'));
    }
    const agentStep = document.querySelector('.journey-step[data-journey="agent"]');
    if (agentStep) {
      agentStep.title = 'Checks itself when a request from pi, opencode, or an OpenAI SDK arrives. Click to try pi right here in the browser — already configured.';
      agentStep.dataset.terminalReady = '1';
    }
  } catch (_) { /* static demo / unreachable: leave hidden */ }
}
revealTerminalShortcuts();

// Distill "use a Recipe" hint — shown until dismissed (or used).
(function initDistillRecipesHint() {
  const hint = document.getElementById('distill-recipes-hint');
  if (!hint) return;
  let dismissed = false;
  try { dismissed = localStorage.getItem('kiln.distill.recipes.hint') === '1'; } catch {}
  hint.hidden = dismissed;
  document.getElementById('distill-recipes-go')?.addEventListener('click', () => {
    document.getElementById('distill-tab-recipes')?.click();
  });
  document.getElementById('distill-recipes-dismiss')?.addEventListener('click', () => {
    try { localStorage.setItem('kiln.distill.recipes.hint', '1'); } catch {}
    hint.hidden = true;
  });
})();

// --- Init ---
renderChat();
updateChatSendState();
initConnect();
initCorrections();
pollHealth();
pollAdapters();
pollTraining();
pollDecodePerf();
pollRecentRequests();
// Pull eval jobs once at startup so the Evals tab badge reflects archived
// runs immediately. The Evals tab also refreshes on its own tab-switch
// handler — this just makes the dashboard's header count accurate before
// the user has visited the tab.
if (typeof refreshEvalJobs === 'function') refreshEvalJobs();

setInterval(pollHealth, 2000);
setInterval(pollAdapters, 5000);
setInterval(pollTraining, 3000);
setInterval(pollDecodePerf, 2000);
setInterval(pollRecentRequests, 2000);
// Refresh relative timestamps every second without re-fetching the list.
setInterval(refreshRecentTimes, 1000);
// Refresh eval-jobs badge count every 5s so it tracks queued / running
// changes even when the user is on a different tab.
setInterval(() => {
  if (typeof refreshEvalJobs !== 'function') return;
  // Only refresh the badge when the Evals tab isn't already actively
  // polling — its own polling cadence is more aggressive when visible.
  const evalsActive = document.querySelector('[data-page="evals"]')?.classList.contains('active');
  if (!evalsActive) refreshEvalJobs();
}, 5000);
