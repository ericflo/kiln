/* =====================================================================
   Page switcher + deep-link hash router
   ===================================================================== */
// The 7 primary pages — single source of truth shared by boot-time hash
// resolution and the hashchange handler below.
const PRIMARY_PAGES = ['overview', 'adapters', 'training', 'evals', 'distill', 'playground', 'terminal'];

/* Deep-link grammar (roadmap PR 17, extending the PR 16 page hashes):

     #page                          — primary page
     #training/queue|sft|grpo       — training sub-tabs
     #training/queue/{job_id}       — train drill modal over the queue
     #evals/datasets|suites|jobs|judgments
     #evals/jobs/{job_id}           — eval drill modal
     #distill/{tab}                 — distill sub-tabs
     #adapters/{name}               — adapter drill modal (page has no sub-tabs)
     #overview/requests/{id}        — request drill modal. Ids are
                                      `chatcmpl-{uuid}`, minted once per
                                      request server-side (completions.rs) and
                                      stable across /v1/stats/recent-requests
                                      polls; FIFO eviction from the 100-entry
                                      ring degrades to the modal's graceful
                                      "no record" body.

   State machine — who writes the hash, and how:

   - Primary-tab click          → selectPage pushes the CANONICAL hash
                                  (#page/active-subtab). One entry even when
                                  selectPage internally redirects the sub-tab
                                  (empty queue → SFT form): the internal click
                                  runs hash-suppressed and the single write
                                  happens after it, so "click Training" never
                                  double-pushes (#training then #training/sft).
   - Sub-tab activation         → pushState #page/subtab. The write lives at
                                  the END of the tab-select fns, so arrow-key
                                  navigation and every programmatic .click()
                                  caller (cmdk, quick actions, "View job"
                                  toasts) mint correct hashes for free.
   - Modal open (user action)   → pushState the id segment and remember WE
                                  minted that entry (modalHashPushed). In-modal
                                  navigation to a sibling id (request drill
                                  prev/next) replaceState's the id instead of
                                  stacking an entry per arrow press.
   - Modal close (X / Esc / backdrop / a modal action that closes it)
                                → if we pushed on open: history.back(). The
                                  hashchange handler lands on the parent hash
                                  and closes the modal idempotently, so the
                                  modal entry is CONSUMED — Back after close
                                  keeps walking pages instead of re-opening
                                  the modal, and closing via browser Back
                                  directly takes the exact same path. Forward
                                  after close re-opens the modal: that entry
                                  is live on purpose, never dead.
                                  If we did NOT push (a deep-link boot or live
                                  hash edit opened it): history.back() could
                                  exit the dashboard, so instead replaceState
                                  to the parent #page/subtab and close
                                  directly.
                                  (A few flows intentionally close WITHOUT
                                  touching history — e.g. "Replay in
                                  playground" — so that Back returns to the
                                  modal the user came from.)
   - Boot + hashchange          → never mint entries here; junk sub-tab/id
                                  segments keep the page and repair the URL to
                                  the canonical #page/subtab via replaceState.
   - localStorage sub-tab restores stay hash-suppressed: they are the no-hash
     fallback only. An explicit hash sub-tab is applied AFTER them during the
     boot route pass (applyHashRoute at the end of this file), so it wins.
*/

// While >0, the tab-select + modal-open hash writers below are no-ops: the
// activation is hash-driven (boot / hashchange) or an internal redirect whose
// caller owns the single history write for the whole gesture.
let hashWriteDepth = 0;
function withHashWritesSuppressed(fn) {
  hashWriteDepth += 1;
  try { fn(); } finally { hashWriteDepth -= 1; }
}

// Active sub-tab name for the pages that have a tablist (null for the rest).
function activeSubTab(name) {
  if (name === 'training') return document.querySelector('[data-training-tabs] [role="tab"].active')?.dataset.tab || null;
  if (name === 'evals') return document.querySelector('[data-evals-tabs] [role="tab"].active')?.dataset.tab || null;
  if (name === 'distill') return document.querySelector('[data-distill-tabs] [role="tab"].active')?.dataset.tab || null;
  return null;
}
function canonicalPageHash(name) {
  const sub = activeSubTab(name);
  return '#' + name + (sub ? '/' + sub : '');
}

// The sub-tab BUTTON for a page/segment pair, or null when the segment is
// junk. Doubles as the whitelist: a button only exists for real sub-tabs.
function subTabButton(pageName, sub) {
  let el = null;
  if (pageName === 'training') el = document.getElementById('training-tab-' + sub);
  else if (pageName === 'evals') el = document.getElementById('evals-tab-' + sub);
  else if (pageName === 'distill') el = document.getElementById('distill-tab-' + sub);
  // Guard against id-shaped collisions (e.g. "opd-pane" resolving to the
  // PANEL div distill-tab-opd-pane): only role=tab buttons qualify.
  return (el && el.getAttribute('role') === 'tab' && el.dataset.tab === sub) ? el : null;
}

// Push #page/subtab for a user-driven sub-tab change. No-op while writes are
// suppressed (hash-driven activation) or when the page isn't frontmost (the
// boot-time localStorage restores run while another page is showing).
function pushSubTabHash(pageName) {
  if (hashWriteDepth > 0 || !history.pushState) return;
  if (!document.getElementById('page-' + pageName)?.classList.contains('active')) return;
  const target = canonicalPageHash(pageName);
  if (location.hash !== target) history.pushState(null, '', target);
}

// --- Drill-modal hash bookkeeping (see the state machine above) --------
// Whether WE minted the history entry for the currently-open modal of each
// kind; decides between history.back() and replaceState on user close.
const modalHashPushed = { eval: false, train: false, adapter: false, request: false, trace: false, run: false };

function modalHashOnOpen(kind, hash, alreadyOpen = false) {
  if (hashWriteDepth > 0 || !history.pushState) { modalHashPushed[kind] = false; return; }
  if (location.hash === hash) return;
  if (alreadyOpen) { history.replaceState(null, '', hash); return; }
  history.pushState(null, '', hash);
  modalHashPushed[kind] = true;
}

function modalHashOnUserClose(kind, parentHash, closeFn) {
  const pushed = modalHashPushed[kind];
  modalHashPushed[kind] = false;
  if (pushed) {
    // Consume the entry we minted on open: the hashchange handler lands on
    // the parent hash and closes the modal. closeFn is idempotent, so the
    // traversal-driven close racing a direct one is harmless.
    history.back();
    return;
  }
  if (history.replaceState && location.hash !== parentHash) history.replaceState(null, '', parentHash);
  closeFn();
}

/* =====================================================================
   Shared modal manager — focus, stacking, scroll lock, Escape (PR 18)
   ---------------------------------------------------------------------
   Every dialog claims aria-modal="true"; this is the machinery that makes
   it true. One stack tracks the open dialogs (a drill with the command
   palette over it is two layers). Per layer the manager owns exactly:

     - focus IN on open (the explicit .modal-close button, else the shell);
     - focus BACK on close (the element focused before that layer opened);
     - the Tab trap (Tab/Shift+Tab wrap within the TOP layer's tabbables);
     - the body scroll lock (released only when the stack empties);
     - Escape (closes the TOP layer only).

   Composition with the deep-link hash state machine above: the manager
   NEVER touches history. openModal/closeModal are called by the per-modal
   open/close fns, which keep their own modalHashOnOpen/OnUserClose calls.
   Escape routes through the layer's `onClose` — the modal's USER-close fn
   (e.g. userCloseDrillModal) — so an Esc press consumes the history entry
   the open minted exactly like the X button does. The traversal path
   (hashchange → syncDrillModalsToRoute → direct close fn) also lands in
   closeModal because the direct close fns call it; closeModal is
   idempotent for untracked elements, so the race is harmless.
   ===================================================================== */
const modalStack = []; // [{ el, onClose, restoreFocus }]

function modalStackTop() {
  return modalStack.length ? modalStack[modalStack.length - 1] : null;
}

// Tabbables: the standard focusable set, filtered to what a Tab press can
// actually reach (visible, not disabled, not tabindex=-1).
const MODAL_TABBABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]',
].join(', ');

function modalTabbables(el) {
  return Array.from(el.querySelectorAll(MODAL_TABBABLE_SELECTOR)).filter(node => {
    if (node.closest('[hidden]')) return false;
    const rect = node.getBoundingClientRect();
    return rect.width > 0 || rect.height > 0;
  });
}

// Move focus into a modal: the close button is the canonical first stop
// (screen readers announce the dialog label, and the most likely next
// action is "get me out"); fall back to the shell (tabindex="-1") and
// finally the backdrop element itself. Open fns with a better target
// (cmdk's search input, adapter-eval's suite select) focus it themselves
// AFTER calling openModal — last write wins.
function focusIntoModal(el) {
  const target = el.querySelector('.modal-close')
    || el.querySelector('.modal-shell, .cmdk-shell, [tabindex="-1"]')
    || el;
  try { target.focus(); } catch {}
}

// opts.onClose: the modal's USER-close fn — the one that runs its hash
// state machine (history.back() vs replaceState). Escape routes through
// it so keyboard close and X-button close are indistinguishable.
function openModal(el, opts = {}) {
  if (!el) return;
  const existing = modalStack.find(layer => layer.el === el);
  if (existing) {
    // Re-entrant open of an already-tracked modal (request drill prev/next
    // re-runs its open fn; route sync re-targets a drill by id): keep the
    // original layer — same restore target, same scroll lock.
    if (opts.onClose) existing.onClose = opts.onClose;
    return;
  }
  const active = document.activeElement;
  modalStack.push({
    el,
    onClose: opts.onClose || null,
    restoreFocus: (active && active !== document.body) ? active : null,
  });
  document.body.style.overflow = 'hidden';
  focusIntoModal(el);
}

function closeModal(el) {
  const idx = modalStack.findIndex(layer => layer.el === el);
  if (idx < 0) return; // not tracked (already closed) — idempotent
  const layer = modalStack.splice(idx, 1)[0];
  // Scroll stays locked while ANY layer remains (cmdk closing over a drill
  // must not unlock the page under the drill).
  if (!modalStack.length) document.body.style.overflow = '';
  // A layer above the closed one (modal A closed underneath modal B, e.g.
  // adapter drill → "Run eval…" hands off to adapter-eval) would restore
  // focus into the now-hidden A; re-point it at A's own restore target.
  const above = modalStack[idx];
  if (above && layer.el.contains(above.restoreFocus)) above.restoreFocus = layer.restoreFocus;
  const top = modalStackTop();
  if (top) {
    // Still a modal under this one: keep focus inside it.
    const rf = layer.restoreFocus;
    if (rf && top.el.contains(rf) && rf.isConnected) { try { rf.focus(); } catch {} }
    else if (!top.el.contains(document.activeElement)) focusIntoModal(top.el);
    return;
  }
  const rf = layer.restoreFocus;
  if (rf && rf.isConnected && !rf.closest('[hidden]')) { try { rf.focus(); } catch {} }
}

// ONE delegated keydown for every dialog. Escape closes the TOP of the
// stack only, through that modal's own close fn (hash machine included).
// Tab/Shift+Tab wrap within the top modal so focus can't escape into the
// inert page behind it.
document.addEventListener('keydown', (ev) => {
  const top = modalStackTop();
  if (!top) return;
  if (ev.key === 'Escape') {
    ev.preventDefault();
    if (top.onClose) top.onClose();
    else closeModal(top.el);
    return;
  }
  if (ev.key !== 'Tab') return;
  const tabbables = modalTabbables(top.el);
  if (!tabbables.length) { ev.preventDefault(); return; }
  const first = tabbables[0];
  const last = tabbables[tabbables.length - 1];
  const active = document.activeElement;
  const inside = top.el.contains(active);
  if (ev.shiftKey) {
    if (!inside || active === first) { ev.preventDefault(); last.focus(); }
  } else if (!inside || active === last) {
    ev.preventDefault();
    first.focus();
  }
});

// opts (optional):
//   fromHash: true  — the call ORIGINATED from the hashchange handler
//                     (browser Back/Forward or a live hash edit). The URL is
//                     already correct; pushing here would bury the entry the
//                     user just traveled to, breaking Forward.
//   replace:  true  — boot-time landing: repair the URL in place instead of
//                     minting an entry, so Back still exits the dashboard.
function selectPage(name, opts) {
  document.querySelectorAll('.primary-tab').forEach(t => {
    const active = t.dataset.page === name;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', String(active));
    // Roving tabindex lives here (not only in the wireTablist wrapper) so
    // the MANY direct selectPage callers — boot, hashchange, quick actions,
    // window.selectPage — keep exactly one nav tab in the Tab order.
    t.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll('.page').forEach(p => {
    const active = p.id === 'page-' + name;
    p.classList.toggle('active', active);
    p.hidden = !active;
    if (active) p.removeAttribute('inert'); else p.setAttribute('inert', '');
  });
  // Landing on an EMPTY queue helps nobody — if nothing ever ran and nothing
  // is queued, the person opening Training came to train. Take them to the
  // form. Requires a LOADED cache: an unfetched queue is unknown, not empty.
  // Runs BEFORE the hash write below (suppressed) so the redirect folds into
  // the same history entry instead of minting #training AND #training/sft.
  if (name === 'training') {
    // try/catch, not typeof: a #training/... DEEP-LINK BOOT reaches here
    // while `let trainingJobsCache` (declared further down this module) is
    // still in its temporal dead zone, where even typeof throws.
    let tj = null;
    try { tj = trainingJobsCache; } catch { tj = null; }
    const queueEmpty = !!tj && !tj.running && (!tj.queued || !tj.queued.length) && (!tj.completed || !tj.completed.length);
    const queueTabActive = document.getElementById('training-tab-queue')?.classList.contains('active');
    if (queueEmpty && queueTabActive) withHashWritesSuppressed(() => document.getElementById('training-tab-sft')?.click());
  }
  // Real history entries (pushState) so browser Back/Forward walks the tab
  // trail. The entry is the CANONICAL #page/active-subtab so traversing back
  // to it restores the sub-tab too. Same-page guard: re-selecting the page
  // already in the hash (tab re-click, polling re-entry) must not stack
  // duplicate entries. Note that pushState itself fires neither hashchange
  // nor popstate, so writing here cannot re-trigger the hashchange handler.
  if (!(opts && opts.fromHash) && hashWriteDepth === 0 && history.pushState) {
    const target = canonicalPageHash(name);
    if (location.hash !== target) {
      if (opts && opts.replace) history.replaceState(null, '', target);
      else history.pushState(null, '', target);
    }
  }
  // Remember the user's last tab so a fresh visit (no hash, no bookmark)
  // lands them back where they were instead of always on Overview.
  // The hash still wins — bookmarks deep-link as expected.
  try { localStorage.setItem('kiln.lastPage', name); } catch {}
  // Lazy-refresh page-scoped data when activating a tab. Avoids firing
  // /v1/eval/* and /v1/judgments/* at page load when the user might never
  // visit the Evals page — and catches the case of someone landing on
  // #evals via deep link without having to mirror the polling guard.
  if (name === 'evals') refreshActiveEvalSubTab();
  if (name === 'distill') refreshActiveDistillSubTab();
  if (name === 'terminal' && typeof initTerminalPage === 'function') initTerminalPage();
}

// Trigger the refresh function matching the currently-active Evals sub-tab.
// Mirrors the polling guard at the bottom of the file so the same selection
// logic isn't duplicated across initial-load + polling + nav-click paths.
function refreshActiveEvalSubTab() {
  if (typeof refreshDatasets !== 'function') return; // not yet defined
  const evalsPage = document.getElementById('page-evals');
  const active = evalsPage?.querySelector('.tab.active')?.dataset?.tab;
  if (active === 'jobs')           refreshEvalJobs();
  else if (active === 'datasets')  refreshDatasets();
  else if (active === 'suites')    refreshSuites();
  else if (active === 'judgments') refreshJudgments();
  else                             refreshDatasets(); // default sub-tab
}
// Primary nav is a tablist too: clicks AND arrow/Home/End keys route
// through selectPage (the existing nav click path), so history entries and
// canonical #page/subtab hashes stay exactly as the click path mints them.
wireTablist(document.querySelector('.primary-nav'), {
  onSelect: tab => selectPage(tab.dataset.page),
});
// Prefer the URL hash (bookmarkable; first segment = page), then the user's
// last tab from localStorage, then Overview. Sub-tab and drill-id segments
// are applied by applyHashRoute at the END of the file, once every tablist
// and modal open fn is wired; this first pass is hash-suppressed so the
// deep link survives untouched in the URL until then.
let initialPage = (location.hash || '').slice(1).split('/')[0];
if (!initialPage) {
  try { initialPage = localStorage.getItem('kiln.lastPage') || ''; } catch {}
}
if (!PRIMARY_PAGES.includes(initialPage)) {
  initialPage = 'overview';
}
withHashWritesSuppressed(() => selectPage(initialPage, { replace: true }));

// Parse location.hash against the full deep-link grammar and drive the UI to
// match: page → sub-tab → drill modal (closing any modal whose id segment is
// gone). Shared by the ONE hashchange listener below and the boot pass at the
// end of the file. Writes back to the URL only as replaceState repairs — this
// function never mints history entries, it consumes them.
function applyHashRoute(opts = {}) {
  const raw = (location.hash || '').slice(1);
  const segs = raw.split('/').map(s => { try { return decodeURIComponent(s); } catch { return s; } });
  const name = segs[0];
  if (!PRIMARY_PAGES.includes(name)) {
    // In-page anchor (e.g. the "Skip to content" link targets #content):
    // leave the browser's native scroll/focus behavior alone.
    if (raw && segs.length === 1 && document.getElementById(raw)) return;
    if (opts.boot) {
      // No (or junk) hash at boot: the localStorage landing above already
      // picked the page — just canonicalize the URL in place.
      const cur = document.querySelector('.page.active')?.id?.replace(/^page-/, '') || 'overview';
      if (history.replaceState) history.replaceState(null, '', canonicalPageHash(cur));
      withHashWritesSuppressed(() => syncDrillModalsToRoute(null));
      return;
    }
    // Unknown/garbage hash: land on Overview and repair the URL in place —
    // replaceState, NOT pushState, so the junk never survives as its own
    // history entry for Back to trip over.
    if (history.replaceState) history.replaceState(null, '', '#overview');
    withHashWritesSuppressed(() => {
      selectPage('overview', { fromHash: true });
      syncDrillModalsToRoute(null);
    });
    return;
  }
  let sub = segs[1] || null;
  let id = segs[2] || null;
  let drill = null; // { kind: 'eval'|'train'|'adapter'|'request'|'trace'|'run', id }
  withHashWritesSuppressed(() => {
    // Re-activating an already-active page would re-fire its lazy refreshes
    // on every sub-tab/modal traversal — only switch when actually needed.
    if (!document.getElementById('page-' + name)?.classList.contains('active')) {
      selectPage(name, { fromHash: true });
    }
    if (name === 'training' || name === 'evals' || name === 'distill') {
      const btn = sub ? subTabButton(name, sub) : null;
      if (btn) {
        if (!btn.classList.contains('active')) btn.click();
      } else {
        // Missing or junk sub-tab segment: keep the page (and whatever
        // sub-tab is already showing) and let the repair below canonicalize.
        sub = activeSubTab(name);
        id = null;
      }
      // Drill ids ride on training/queue, evals/jobs, distill/traces, and
      // distill/runs.
      if (name === 'training' && sub === 'queue' && id) drill = { kind: 'train', id };
      else if (name === 'evals' && sub === 'jobs' && id) drill = { kind: 'eval', id };
      else if (name === 'distill' && sub === 'traces' && id) drill = { kind: 'trace', id };
      else if (name === 'distill' && sub === 'runs' && id) drill = { kind: 'run', id };
      else id = null;
    } else if (name === 'adapters' && sub) {
      // #adapters/{name} — the second segment is a drill id, not a sub-tab.
      drill = { kind: 'adapter', id: sub };
      id = null;
    } else if (name === 'overview' && sub === 'requests' && id) {
      drill = { kind: 'request', id };
    } else {
      // Pages without segments (overview/adapters/playground/terminal), or
      // segment shapes outside the grammar: degrade to the page itself.
      sub = null;
      id = null;
    }
    syncDrillModalsToRoute(drill);
  });
  // Repair the URL in place to the canonical spelling of what actually
  // activated (drops junk segments). Never mints an entry.
  const canonical = '#' + name
    + (drill && drill.kind === 'adapter' ? '/' + encodeURIComponent(drill.id) : (sub ? '/' + sub : ''))
    + (drill && drill.kind !== 'adapter' ? '/' + encodeURIComponent(drill.id) : '');
  if (history.replaceState && location.hash !== canonical) history.replaceState(null, '', canonical);
}

// Open/close the six drill modals so they match the route. Closes here are
// direct (never history.back()): this runs FROM a traversal or boot, where
// the URL is already where it should be. Opens run hash-suppressed by the
// caller, so the open fns' own pushState helpers stay quiet. Each modal's
// open fn fetches its own data by id and degrades gracefully (error body or
// toast-and-close), so routing ahead of the first poll is safe; the one
// exception is the request drill, whose data source is the polled
// recent-requests ring — when that hasn't loaded yet the open is DEFERRED to
// the first poll (pendingRequestDrillId in pollRecentRequests) instead of
// flashing a false "no record for that id".
function syncDrillModalsToRoute(drill) {
  const want = drill || {};
  const evalModal = document.getElementById('eval-drill-modal');
  if (evalModal) {
    const openId = evalModal.hidden ? null : (evalDrillJobId || null);
    if (want.kind === 'eval') {
      if (openId !== want.id) openDrillModal(want.id);
    } else if (openId !== null) {
      modalHashPushed.eval = false;
      closeDrillModal();
    }
  }
  const trainModal = document.getElementById('train-drill-modal');
  if (trainModal) {
    const openId = trainModal.hidden ? null : (trainDrillJobId || null);
    if (want.kind === 'train') {
      if (openId !== want.id) openTrainDrillModal(want.id);
    } else if (openId !== null) {
      modalHashPushed.train = false;
      closeTrainDrillModal();
    }
  }
  const adapterModal = document.getElementById('adapter-drill-modal');
  if (adapterModal) {
    const openId = adapterModal.hidden ? null : (adapterDrillName || null);
    if (want.kind === 'adapter') {
      if (openId !== want.id) openAdapterDrillModal(want.id);
    } else if (openId !== null) {
      modalHashPushed.adapter = false;
      closeAdapterDrillModal();
    }
  }
  const requestModal = document.getElementById('request-drill-modal');
  if (requestModal) {
    const openId = requestModal.hidden ? null : (requestModal.dataset.requestId || null);
    if (want.kind === 'request') {
      if (!recentRequestsLoaded) pendingRequestDrillId = want.id;
      else if (openId !== want.id) openRequestDrillModal(want.id);
    } else {
      pendingRequestDrillId = null;
      if (openId !== null) {
        modalHashPushed.request = false;
        closeRequestDrillModal();
      }
    }
  }
  const traceModal = document.getElementById('trace-drill-modal');
  if (traceModal) {
    const openId = traceModal.hidden ? null : (traceDrillId || null);
    if (want.kind === 'trace') {
      if (openId !== want.id) openTraceDrillModal(want.id);
    } else if (openId !== null) {
      modalHashPushed.trace = false;
      closeTraceDrillModal();
    }
  }
  const runModal = document.getElementById('run-drill-modal');
  if (runModal) {
    const openId = runModal.hidden ? null : (runDrillId || null);
    if (want.kind === 'run') {
      if (openId !== want.id) openRunDrillModal(want.id);
    } else if (openId !== null) {
      modalHashPushed.run = false;
      closeRunDrillModal();
    }
  }
}

// Browser Back/Forward + live hash edits re-resolve through the same
// whitelist as boot. ONE hashchange listener suffices — no separate
// popstate handler. Evidence (probed in headless Chrome 148, the same
// binary the smoke suite drives):
//   - history.pushState('#x') fires NEITHER hashchange nor popstate;
//   - Back/Forward across pushState'd entries fires BOTH (the entries
//     differ only by fragment, and fragment-differing same-document
//     traversals fire hashchange per spec);
//   - location.hash = '#x' assignment fires BOTH.
// Every entry the writers above create differs by fragment (same-hash
// guards), so hashchange covers every traversal we can produce, and it is
// the spec-guaranteed event for address-bar hash edits. Listening to both
// events would double-invoke this handler on every Back/Forward.
window.addEventListener('hashchange', () => applyHashRoute({ fromHash: true }));

// --- Toast Notifications ---
function toast(msg, type) {
  const c = document.getElementById('toasts');
  // Dedupe: a poll that errors every 2s shouldn't stack 30 identical
  // toasts. If the most-recent live toast matches the message + type,
  // bump a small (xN) counter on it and reset its dismiss timer
  // instead of stacking a new one.
  const last = c.lastElementChild;
  if (last && last.dataset.toastKey === `${type || 'ok'}|${msg}`) {
    const cnt = (Number(last.dataset.toastCount) || 1) + 1;
    last.dataset.toastCount = String(cnt);
    last.querySelector('.toast-count')?.remove();
    const counter = document.createElement('span');
    counter.className = 'toast-count';
    counter.style.cssText = 'margin-left:8px; opacity:0.7; font-variant-numeric: tabular-nums;';
    counter.textContent = `×${cnt}`;
    last.insertBefore(counter, last.querySelector('.toast-action-close'));
    clearTimeout(Number(last.dataset.toastTimer));
    last.dataset.toastTimer = String(setTimeout(() => last.remove(), 4000));
    return;
  }
  const el = document.createElement('div');
  el.className = 'toast ' + (type || 'ok');
  el.dataset.toastKey = `${type || 'ok'}|${msg}`;
  if (type === 'err') {
    el.setAttribute('role', 'alert');
    el.setAttribute('aria-live', 'assertive');
    el.setAttribute('aria-atomic', 'true');
  }
  const text = document.createElement('span');
  text.textContent = msg;
  el.appendChild(text);
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'toast-action-close';
  close.setAttribute('aria-label', 'Dismiss');
  close.innerHTML = icon('close', 'icn-sm');
  close.addEventListener('click', () => el.remove());
  el.appendChild(close);
  c.appendChild(el);
  el.dataset.toastTimer = String(setTimeout(() => el.remove(), 4000));
}

// A completion notice that carries the NEXT ACTION. Sticks around (25s, plus
// hover-to-keep) because "your adapter finished" deserves more than a 4-second
// flash the user was never going to catch from another tab.
function actionToast(msg, type, actions) {
  const c = document.getElementById('toasts');
  if (!c) return;
  const el = document.createElement('div');
  el.className = 'toast ' + (type || 'ok') + ' toast-action';
  el.setAttribute('role', 'status');
  const text = document.createElement('span');
  text.textContent = msg;
  el.appendChild(text);
  (actions || []).forEach(a => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-sm toast-action-btn';
    btn.textContent = a.label;
    btn.addEventListener('click', ev => { ev.stopPropagation(); el.remove(); a.onClick && a.onClick(); });
    el.appendChild(btn);
  });
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'toast-action-close';
  close.setAttribute('aria-label', 'Dismiss');
  close.textContent = '×';
  close.addEventListener('click', ev => { ev.stopPropagation(); el.remove(); });
  el.appendChild(close);
  c.appendChild(el);
  let timer = setTimeout(() => el.remove(), 25000);
  el.addEventListener('mouseenter', () => clearTimeout(timer));
  el.addEventListener('mouseleave', () => { timer = setTimeout(() => el.remove(), 8000); });
}

// --- Icon helper ---
// Returns inline-SVG markup for a sprite symbol (see #i-* defs at top of body).
// Use in template strings instead of emoji so glyphs inherit currentColor.
function icon(name, extraCls) {
  return `<svg class="icn${extraCls ? ' ' + extraCls : ''}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
}

// --- API Helpers ---
async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    // A thrown fetch = the server didn't answer at all (down / wrong host /
    // CORS) — say so plainly instead of the browser's opaque "Failed to fetch".
    const err = new Error(`Can't reach Kiln at ${window.location.origin} — is \`kiln serve\` running on this host/port?`);
    err.unreachable = true;
    throw err;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    // Kiln's canonical error shape is { error: { code, message, hint } };
    // body.error may also be a plain string behind generic proxies.
    const errObj = (body.error && typeof body.error === 'object') ? body.error : null;
    const msg = (typeof body.error === 'string' && body.error)
      || errObj?.message || body.detail || '';
    const hint = errObj?.hint || '';
    const err = new Error(msg ? (hint ? `${msg} — ${hint}` : msg) : `HTTP ${res.status} from ${path}`);
    if (errObj?.code) err.code = errObj.code;
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/* =====================================================================
   Connect-your-agent panel — onboarding centerpiece. Live base URL + model
   id + per-client setup snippets + test-connection. The #1 task for an
   operator: point pi / opencode / any OpenAI client at this server.
   ===================================================================== */
const CONNECT_FALLBACK_MODEL = 'Qwen3.5-4B';
let connectModelId = CONNECT_FALLBACK_MODEL;
let connectManualState = null; // null = auto-collapse on traffic; true/false = user override

function connectBaseUrl() { return window.location.origin + '/v1'; }

// Token highlighter operating on RAW code (so quotes/comments survive escaping).
function renderHighlighted(code, lang) {
  const esc = s => escapeHtml(s);
  return code.split('\n').map(line => {
    const cm = (lang === 'py' || lang === 'sh') ? line.match(/^(\s*)(#.*)$/)
             : (lang === 'js') ? line.match(/^(\s*)(\/\/.*)$/) : null;
    if (cm) return esc(cm[1]) + '<span class="tok-com">' + esc(cm[2]) + '</span>';
    let res = '', last = 0, m; const re = /"(?:[^"\\]|\\.)*"/g;
    while ((m = re.exec(line))) { res += esc(line.slice(last, m.index)) + '<span class="tok-str">' + esc(m[0]) + '</span>'; last = m.index + m[0].length; }
    return res + esc(line.slice(last));
  }).join('\n');
}

function connectSnippets() {
  const base = connectBaseUrl(), origin = window.location.origin, model = connectModelId;
  return {
    pi: { lang: 'sh',
      note: 'One command — backs up &amp; merges <code>~/.pi/agent/{models,settings}.json</code>, then makes Kiln pi&rsquo;s default model.',
      code: `# Point pi at this Kiln server\nkiln pi-setup\n\n# Remote server? pass its URL — /v1 is appended for you\nkiln pi-setup --kiln-url ${origin}` },
    opencode: { lang: 'js', path: '~/.config/opencode/opencode.json',
      note: 'Add this provider, then run <code>opencode</code> and pick the model with <code>/models</code>. No API key needed.',
      code: `{\n  "$schema": "https://opencode.ai/config.json",\n  "provider": {\n    "kiln": {\n      "npm": "@ai-sdk/openai-compatible",\n      "name": "Kiln (local)",\n      "options": { "baseURL": "${base}", "apiKey": "unused" },\n      "models": { "${model}": { "name": "Kiln · ${model}" } }\n    }\n  }\n}` },
    python: { lang: 'py',
      note: 'Drop-in OpenAI SDK — only <code>base_url</code> changes.',
      code: `from openai import OpenAI\n\nclient = OpenAI(base_url="${base}", api_key="unused")\n\nresp = client.chat.completions.create(\n    model="${model}",\n    messages=[{"role": "user", "content": "Hello from my agent"}],\n)\nprint(resp.choices[0].message.content)` },
    js: { lang: 'js',
      note: 'Drop-in OpenAI SDK for Node / Bun / Deno.',
      code: `import OpenAI from "openai";\n\nconst client = new OpenAI({ baseURL: "${base}", apiKey: "unused" });\n\nconst resp = await client.chat.completions.create({\n  model: "${model}",\n  messages: [{ role: "user", content: "Hello from my agent" }],\n});\nconsole.log(resp.choices[0].message.content);` },
    curl: { lang: 'sh',
      note: 'Raw HTTP — drop into any shell, CI job, or script.',
      code: `curl -s ${base}/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "${model}",\n    "messages": [{"role": "user", "content": "Hello from my agent"}]\n  }'` },
  };
}

function renderConnectSnippets(active) {
  const host = document.getElementById('connect-snippets');
  if (!host) return;
  const snips = connectSnippets();
  host.innerHTML = Object.entries(snips).map(([key, s]) => {
    const pathLine = s.path ? `<div class="code-block-path">${escapeHtml(s.path)}</div>` : '';
    // Tabpanel pairing mirrors the training tabs: the static tab buttons in
    // index.html carry id="connect-tab-{key}" + aria-controls back at these.
    return `<div class="connect-snippet${key === active ? ' active' : ''}" data-connect-pane="${key}" role="tabpanel" id="connect-snippet-${key}" aria-labelledby="connect-tab-${key}">
      <div class="connect-snippet-note">${s.note}</div>
      <div class="code-block">${pathLine}<button class="copy-btn" type="button" data-copy-code aria-label="Copy code"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg>Copy</button><pre>${renderHighlighted(s.code, s.lang)}</pre></div>
    </div>`;
  }).join('');
}

// Push a freshly-resolved served-model id into every copyable surface: the
// model-id field and the pi/opencode/SDK/curl snippets. Cold starts open the
// dashboard before /v1/models can answer, so the panel initially renders the
// fallback id — this silently upgrades it the moment the real id arrives.
// Re-renders only on an actual change, preserving the selected tab.
function applyServedModelId(id) {
  if (!id || connectModelId === id) return;
  connectModelId = id;
  const modelEl = document.getElementById('connect-model');
  if (modelEl) { modelEl.textContent = id; modelEl.title = id; }
  const activeTab = document.querySelector('.connect-tabs .tab.active')?.dataset.connectTab || 'pi';
  renderConnectSnippets(activeTab);
}

// Prometheus scrape config for this server's /metrics endpoint, built from
// the live origin. Rendered once at init (the origin can't change without a
// reload); the Copy button reuses the same delegated data-copy-code handler
// as the client snippets above.
function renderConnectMetricsSnippet() {
  const pre = document.getElementById('connect-metrics-snippet');
  if (!pre) return;
  const code = `# prometheus.yml — scrape this Kiln server\nscrape_configs:\n  - job_name: "kiln"\n    metrics_path: "/metrics"\n    static_configs:\n      - targets: ["${window.location.host}"]`;
  pre.innerHTML = renderHighlighted(code, 'sh');
}

function selectConnectTab(key) {
  document.querySelectorAll('.connect-tabs .tab').forEach(t => {
    const on = t.dataset.connectTab === key;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll('.connect-snippet').forEach(p => p.classList.toggle('active', p.dataset.connectPane === key));
}

async function copyText(text, btn) {
  try { await navigator.clipboard.writeText(text); }
  catch { const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); } catch {} ta.remove(); }
  if (btn) {
    const orig = btn.innerHTML;
    btn.classList.add('copied');
    btn.innerHTML = `${icon('check', 'icn-sm')}Copied`;
    setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = orig; }, 1400);
  }
}

function setConnectCollapsed(collapsed) {
  const exp = document.getElementById('connect-expanded');
  const col = document.getElementById('connect-collapsed');
  if (!exp || !col) return;
  exp.hidden = collapsed; col.hidden = !collapsed;
}

function openConnect() {
  connectManualState = true;
  setConnectCollapsed(false);
  selectPage('overview');
  const panel = document.getElementById('connect-panel');
  if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Auto-collapse to the slim "connected" bar once EXTERNAL traffic flows
// (unless the operator has manually picked a state this session). The
// dashboard's own requests (Test connection, Playground — `client:
// "dashboard"`) don't prove anything connected and must not hide the setup
// snippets a first-run user still needs.
function updateConnectSummary(rows) {
  const external = (Array.isArray(rows) ? rows : []).filter(r => !rowIsDashboard(r));
  const n = external.length;
  const txt = document.getElementById('connect-collapsed-text');
  if (txt && n > 0) {
    const last = external[0]?.timestamp_unix_ms;
    txt.innerHTML = `<strong>Connected</strong> · <strong>${n}</strong> recent request${n === 1 ? '' : 's'}${last ? ' · last ' + fmtRelTime(last) : ''}`;
  }
  if (connectManualState === null && n > 0) setConnectCollapsed(true);
}

// Exposed for inline onclick handlers in dynamically-rendered empty states.
window.openConnect = openConnect;
window.selectPage = selectPage;

/* === Request health & ambient awareness (the in-flow pi coder) ===========
   The dashboard usually lives in a background tab while the dev codes with pi.
   These derive what pi ACTUALLY experiences from the recent-requests ring —
   errors, truncations, end-to-end TTFT, which adapter actually served, and
   freshness — and surface it as a live heartbeat, a silent-base-fallback
   warning, and ambient document.title + favicon so real problems break
   through without alt-tabbing back to the dashboard. */
let lastRequestHealth = null;
// Adapters demoted by a §8.7 eval gate are renamed `<name>.failed` (or
// `<name>.failed-<unix_ms>` on collision) by eval/worker.rs and will never
// serve again under that name; canary-quarantined and invalid registry
// entries can't be hot-swapped either. None of those count as "a trained
// adapter you could be serving" — a user whose only adapter was demoted is
// serving base CORRECTLY, and must not have every request flagged forever.
const GATE_DEMOTED_RE = /\.failed(-\d+)?$/;
function nonBaseAdapterCount() {
  const av = (lastAdapters && lastAdapters.available) || [];
  return av.filter(a => a && a.name && a.name !== 'base'
    && !GATE_DEMOTED_RE.test(a.name)
    && a.status !== 'quarantined' && a.status !== 'invalid').length;
}
// A base-served request is a SILENT fallback only when the server itself
// claims a non-base adapter is active (/v1/adapters `active`) — that request
// demonstrably bypassed the adapter the server says it serves. When `active`
// is null the server is intentionally configured for base (the unload button
// sets exactly this), so base-served requests are the requested behaviour,
// not a per-request defect. NOTE (API follow-up): `active == null` cannot
// distinguish "user explicitly unloaded" from "server restarted and nothing
// re-promoted an adapter" — the server would need to ship WHY active is null
// (e.g. explicit_unload vs never_loaded) for the ambient heartbeat nudge
// below to also quiet the deliberate case.
function servedBaseSilently(adapterName) {
  if ((adapterName || 'base') !== 'base') return false;
  const active = lastAdapters && lastAdapters.active;
  return !!(active && active !== 'base');
}
// Nearest-rank quantile of an ascending-sorted numeric array: q(0.99) over
// 100 sorted samples returns the 99th-percentile element. Kept as a named
// top-level function so the quantile math is testable in isolation.
function sortedQuantile(sorted, p) {
  return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))] : null;
}
function computeRequestHealth(rows) {
  rows = rows || [];
  const now = Date.now();
  const sample = rows.slice(0, 30);
  let errors = 0, truncated = 0; const ttfts = [];
  for (const r of sample) {
    const f = (r.finish_reason || '').toLowerCase();
    // 'client_disconnect' is the caller hanging up (Ctrl-C in pi mid-stream)
    // — deliberate, not degraded service. It stays neutral so it never
    // drives the error heartbeat/ambient states.
    if (r.error || f === 'error') errors++;
    else if (f === 'length') truncated++;
    if (typeof r.ttft_ms === 'number') ttfts.push(r.ttft_ms);
  }
  ttfts.sort((a, b) => a - b);
  const q = p => sortedQuantile(ttfts, p);
  const lastTs = rows.length ? rows[0].timestamp_unix_ms : null;
  return {
    // ttftP99 really is q(0.99) — the heartbeat tooltip renders it as "p99"
    // (tail latency, matching /v1/stats/decode's p99_itl_ms), so computing
    // anything else here would mislabel the number.
    total: rows.length, errors, truncated, ttftP50: q(0.5), ttftP99: q(0.99),
    lastTs, sinceMs: lastTs ? now - lastTs : null,
    servedBy: rows.length ? (rows[0].adapter || 'base') : null,
  };
}
function fmtMsShort(ms) { return ms == null ? '—' : (ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(1) + ' s'); }

function updateHeartbeat(h) {
  const el = document.getElementById('recent-heartbeat');
  if (!el) return;
  const txt = el.querySelector('.hb-text');
  // Deliberately broader than servedBaseSilently(): the ambient nudge fires
  // whenever live traffic is on base while servable trained adapters sit
  // idle — that catches the post-restart "active adapter was lost" hole that
  // per-request attention no longer flags. The label is a factual status
  // ("serving base model"), not an error count.
  const baseFallback = h.servedBy === 'base' && nonBaseAdapterCount() > 0;
  let cls, label;
  if (h.errors > 0) {
    cls = 'err';
    label = `${h.errors} error${h.errors === 1 ? '' : 's'}${h.truncated ? ` · ${h.truncated} truncated` : ''} recently`;
  } else if (h.lastTs == null) {
    cls = 'idle'; label = 'No requests yet';
  } else if (h.sinceMs != null && h.sinceMs < 15000) {
    if (baseFallback) { cls = 'warn'; label = `Live · serving base model · last ${fmtRelTime(h.lastTs)}`; }
    else { cls = 'live'; label = `Live · last ${fmtRelTime(h.lastTs)}${h.ttftP50 != null ? ` · TTFT ${fmtMsShort(h.ttftP50)}` : ''}`; }
  } else if (h.truncated > 0) {
    cls = 'warn'; label = `${h.truncated} truncated · last ${fmtRelTime(h.lastTs)}`;
  } else if (h.sinceMs != null && h.sinceMs < 120000) {
    cls = 'idle'; label = `Last request ${fmtRelTime(h.lastTs)}`;
  } else {
    cls = 'quiet'; label = `Quiet · ${fmtRelTime(h.lastTs)}`;
  }
  el.className = 'heartbeat hb-' + cls;
  if (txt) txt.textContent = label;
  el.title = baseFallback
    ? 'pi is hitting the BASE model even though you have trained adapters. Hot-swap one on the Adapters page so your usage actually improves the model.'
    : (h.ttftP99 != null ? `End-to-end TTFT p50 ${fmtMsShort(h.ttftP50)} · p99 ${fmtMsShort(h.ttftP99)} (the latency pi feels)` : '');
}

// Ambient document.title + favicon so the in-flow coder catches problems from
// a background tab. Green = actively serving, amber = truncations/base
// fallback, red = errors, none = idle.
function updateAmbient(h) {
  let title = 'Kiln Dashboard', dot = null;
  const live = h.sinceMs != null && h.sinceMs < 15000;
  if (h.errors > 0) { title = `⚠ ${h.errors} error${h.errors === 1 ? '' : 's'} · Kiln`; dot = '#f87171'; }
  else if (live && h.servedBy === 'base' && nonBaseAdapterCount() > 0) { title = '● base model · Kiln'; dot = '#fbbf24'; }
  else if (live) {
    const tps = lastDecode && lastDecode.tok_per_sec ? Math.round(lastDecode.tok_per_sec) : null;
    title = tps ? `● ${tps} tok/s · Kiln` : '● serving · Kiln'; dot = '#4ade80';
  } else if (h.truncated > 0) { title = `${h.truncated} truncated · Kiln`; dot = '#fbbf24'; }
  else if (h.lastTs != null) { title = 'Kiln — idle'; }
  if (document.title !== title) document.title = title;
  setFaviconDot(dot);
}
let _faviconDot = 'init';
function setFaviconDot(color) {
  if (color === _faviconDot) return; _faviconDot = color;
  const dot = color ? `<circle cx="24" cy="8" r="6.5" fill="${color}" stroke="#0a0908" stroke-width="2"/>` : '';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#fbbf24"/><stop offset="0.5" stop-color="#f97316"/><stop offset="1" stop-color="#c2410c"/></linearGradient></defs><rect width="32" height="32" rx="7" fill="#0a0908"/><rect x="4" y="4" width="24" height="24" rx="5" fill="url(#g)"/><g fill="none" stroke="#0a0908" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"><path d="M8 12.5l8 4 8-4"/><path d="M10 17.5l6 3 6-3"/><path d="M12 22.5l4 2 4-2"/></g>${dot}</svg>`;
  const link = document.querySelector('link[rel="icon"]');
  if (link) link.setAttribute('href', 'data:image/svg+xml,' + encodeURIComponent(svg));
}

// Recompute + repaint all request-health surfaces. Called every recent-requests
// poll (2s).
function refreshRequestHealth() {
  lastRequestHealth = computeRequestHealth(recentRequestsCache);
  updateHeartbeat(lastRequestHealth);
  updateAmbient(lastRequestHealth);
}

/* === Corrections basket — the flywheel's core loop, one-click =============
   pi gives a bad answer → "Use as correction" APPENDS it here (not overwrite)
   → you fix the ideal answer inline → "Train" turns the whole basket into ONE
   SFT job and hot-swaps the result. This is the literal mechanism of "your
   model gets better every time you use it".

   Durability: the server's /v1/corrections store is the source of truth —
   corrections survive across browsers/machines, pi can file them
   programmatically, and trained rows are MARKED (kept as history with their
   hand-written ideal answers) rather than deleted. localStorage remains a
   write-behind cache so the basket still works in the static demo and
   through transient server hiccups. */
const CORR_KEY = 'kiln.corrections.v1';
let correctionsBasket = [];
// A persistent "training started" receipt shown in the Corrections card after a
// train submit, so the handoff isn't a silent page-jump. {name, count}.
let corrReceipt = null;
function loadCorrections() {
  try { const v = JSON.parse(localStorage.getItem(CORR_KEY) || '[]'); correctionsBasket = Array.isArray(v) ? v : []; }
  catch { correctionsBasket = []; }
}
function saveCorrections() { try { localStorage.setItem(CORR_KEY, JSON.stringify(correctionsBasket)); } catch {} }
// ── server write-through (best-effort; the local basket never blocks) ──
function corrRowForServer(c) {
  return {
    request_id: c.request_id,
    agent: c.agent || '',
    adapter: (c.adapter && c.adapter !== 'base') ? c.adapter : null,
    user: c.user || '',
    original: c.original || '',
    ideal: c.ideal || '',
    truncated: !!c.truncated,
  };
}
function corrSyncUpsert(c) {
  try {
    api('/v1/corrections', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(corrRowForServer(c)) }).catch(() => {});
  } catch (_) {}
}
function corrSyncRemove(rid) {
  try { api('/v1/corrections/' + encodeURIComponent(rid), { method: 'DELETE' }).catch(() => {}); } catch (_) {}
}
// Debounced upsert per row so typing in the ideal editor doesn't spam the
// server — flushes 800ms after the last keystroke.
const corrSyncTimers = {};
function corrSyncUpsertDebounced(c) {
  clearTimeout(corrSyncTimers[c.request_id]);
  corrSyncTimers[c.request_id] = setTimeout(() => corrSyncUpsert(c), 800);
}
// On load, the server list wins: rows existing there (incl. ones captured
// from another browser or filed by pi) replace the local cache; local-only
// rows (offline captures) are pushed up.
async function corrSyncFromServer() {
  let server;
  try { server = await api('/v1/corrections'); } catch (_) { return; }
  if (!server || !Array.isArray(server.corrections)) return;
  const serverRows = server.corrections.map(r => ({
    request_id: r.request_id, agent: r.agent || 'client', ts: Date.parse(r.created_at) || Date.now(),
    model: '', adapter: r.adapter || 'base', user: r.user || '',
    original: r.original || '', truncated: !!r.truncated, ideal: r.ideal || '',
  }));
  const serverIds = new Set(serverRows.map(r => r.request_id));
  const localOnly = correctionsBasket.filter(c => !serverIds.has(c.request_id));
  localOnly.forEach(corrSyncUpsert);
  correctionsBasket = serverRows.concat(localOnly);
  saveCorrections();
  renderCorrections();
}
// Shared basket insert for every capture surface (recent requests, eval
// drill). Dedupes on request_id; persists + repaints + toasts.
function addCorrectionItem(item) {
  if (correctionsBasket.some(c => c.request_id === item.request_id)) {
    toast('Already in your corrections', 'info');
    return false;
  }
  correctionsBasket.unshift(item);
  saveCorrections();
  corrSyncUpsert(item);
  renderCorrections();
  toast(`Added to corrections (${correctionsBasket.length}) — ${item.ideal ? 'review the ideal answer, then Train' : 'write the ideal answer, then Train'}`, 'ok');
  return true;
}

function addCorrectionFromRequest(r) {
  if (!r) return false;
  return addCorrectionItem({
    request_id: r.id || ('req-' + Date.now()),
    agent: (clientForRow(r) || {}).label || 'client',
    ts: r.timestamp_unix_ms || Date.now(),
    model: r.model || '', adapter: r.adapter || 'base',
    user: r.prompt_full || r.prompt_preview || '',
    original: r.completion_full || r.completion_preview || '',
    // Only a preview was retained for this request — flag it so the operator
    // knows the captured text may be cut short.
    truncated: !(r.prompt_full && r.completion_full),
    // Start EMPTY — never pre-seed with the bad answer. A correction left
    // untouched must never silently train the model on its own mistake.
    ideal: '',
  });
}

// Eval-drill capture: turn a failing outcome into a correction. The ideal
// answer is pre-seeded from the example target ONLY for scorers whose
// target IS the verbatim expected reply (exact_match / contains) — for
// everything else the target is a pattern/choice/number, not a reply, and
// pre-seeding would train garbage. corrTrainable still requires the ideal
// to differ from the model's output before anything trains.
function addCorrectionFromEvalOutcome(o, example, scorerKind) {
  const rid = `eval-${drillJob?.job_id || 'job'}-${o.example_id}-${o.completion_index || 0}`;
  const userMsg = example && example.messages
    ? (example.messages.filter(m => m.role === 'user').pop()?.content || '')
    : '';
  if (!userMsg) { toast('Suite content unavailable — cannot capture this outcome', 'err'); return false; }
  const seedIdeal = (scorerKind === 'exact_match' || scorerKind === 'contains')
    ? (example.target || '')
    : '';
  return addCorrectionItem({
    request_id: rid,
    agent: 'eval:' + (drillJob?.suite_name || 'suite'),
    ts: Date.now(),
    model: '',
    adapter: drillJob?.adapters?.[drillSelectedRun] ?? 'base',
    user: userMsg,
    original: o.completion_text || '',
    truncated: false,
    ideal: seedIdeal,
  });
}
function removeCorrection(rid) { correctionsBasket = correctionsBasket.filter(c => c.request_id !== rid); saveCorrections(); corrSyncRemove(rid); renderCorrections(); }
function clearCorrections() {
  if (correctionsBasket.length && !confirm(`Discard ${correctionsBasket.length} correction${correctionsBasket.length === 1 ? '' : 's'}?`)) return;
  correctionsBasket = []; saveCorrections(); renderCorrections();
  try { api('/v1/corrections', { method: 'DELETE' }).catch(() => {}); } catch (_) {}
}
// A correction trains ONLY if you supplied a genuinely different answer. Empty
// (untouched) or identical-to-original (reverted) is excluded — the whole point
// is that the model never learns from the very output you flagged as wrong.
function corrTrainable(c) {
  const ideal = (c.ideal || '').trim();
  return ideal.length > 0 && ideal !== (c.original || '').trim();
}
function corrStateHtml(ready) {
  return ready ? icon('check', 'icn-sm') + 'ready to train' : 'needs your answer';
}
function renderCorrections() {
  const card = document.getElementById('corrections-card');
  const list = document.getElementById('corrections-list');
  const n = correctionsBasket.length;
  setText('corr-count', String(n));
  // Keep the card visible while a training receipt is showing, even if the
  // basket emptied out (all corrections were just trained in).
  if (card) card.hidden = n === 0 && !corrReceipt;
  if (!list) { updateCorrFoot(); return; }
  // The receipt tracks the submitted job through the queue poll
  // (watchCorrectionsJob): 'training' while queued/running, 'done' on
  // completion, 'failed' with the job's error once the worker gives up —
  // at which point the durable rows (never marked, per the completion-time
  // contract) are pulled back into the basket for fix-and-retrain.
  const dismissBtnHtml = `<button type="button" class="btn btn-sm btn-ghost corr-receipt-dismiss" id="corr-receipt-dismiss" aria-label="Dismiss receipt">${icon('close', 'icn-sm')}</button>`;
  const receiptHtml = corrReceipt ? (() => {
    const nameHtml = `<strong>${escapeHtml(corrReceipt.name)}</strong>`;
    const plural = corrReceipt.count === 1 ? '' : 's';
    if (corrReceipt.state === 'failed') {
      return `
    <div class="corr-receipt corr-receipt-failed" role="alert">
      <span class="corr-receipt-icon">${icon('warning', 'icn-sm')}</span>
      <span class="corr-receipt-text">Training ${nameHtml} failed: ${escapeHtml(String(corrReceipt.error || 'unknown error').slice(0, 220))} — your correction${plural} ${corrReceipt.count === 1 ? 'is' : 'are'} back below. Fix and train again.</span>
      <button type="button" class="btn btn-sm" id="corr-receipt-view">View job ${icon('arrow-right', 'icn-sm')}</button>
      ${dismissBtnHtml}
    </div>`;
    }
    if (corrReceipt.state === 'done') {
      return `
    <div class="corr-receipt" role="status">
      <span class="corr-receipt-icon">${icon('check', 'icn-sm')}</span>
      <span class="corr-receipt-text">Trained ${nameHtml} from ${corrReceipt.count} correction${plural} — hot-swapped in.</span>
      ${corrReceipt.firstPrompt ? `<button type="button" class="btn btn-sm" id="corr-receipt-verify" title="Replay the corrected prompt in Playground compare — base vs ${escapeHtml(corrReceipt.name)} — to see the fix land">Verify the fix</button>` : ''}
      <button type="button" class="btn btn-sm" id="corr-receipt-view">View in queue ${icon('arrow-right', 'icn-sm')}</button>
      ${dismissBtnHtml}
    </div>`;
    }
    return `
    <div class="corr-receipt" role="status">
      <span class="corr-receipt-icon">${icon('check', 'icn-sm')}</span>
      <span class="corr-receipt-text">Training ${nameHtml} from ${corrReceipt.count} correction${plural} — it'll hot-swap in when done.</span>
      <button type="button" class="btn btn-sm" id="corr-receipt-view">View in queue ${icon('arrow-right', 'icn-sm')}</button>
      ${dismissBtnHtml}
    </div>`;
  })() : '';
  list.innerHTML = receiptHtml + correctionsBasket.map(c => {
    const prev = (c.user || '').slice(0, 180);
    const rid = escapeHtml(c.request_id);
    const ready = corrTrainable(c);
    return `<div class="corr-item ${ready ? 'is-ready' : 'is-todo'}" data-corr="${rid}">
      <div class="corr-item-head">
        <span class="recent-agent">${escapeHtml(c.agent)}</span>
        <span class="corr-prompt" title="${escapeHtml(c.user)}">${escapeHtml(prev)}${c.user.length > 180 ? '…' : ''}</span>
        ${c.truncated ? `<span class="corr-trunc" title="Only a preview of this request was retained — the captured text may be cut short. Check it before training.">${icon('warning', 'icn-sm')} preview only</span>` : ''}
        <span class="corr-state" data-corr-state="${rid}">${corrStateHtml(ready)}</span>
        <button type="button" class="btn btn-sm btn-ghost corr-remove" data-corr-remove="${rid}" aria-label="Remove correction">${icon('close', 'icn-sm')}</button>
      </div>
      <div class="corr-grid">
        <div class="corr-col">
          <div class="corr-label">pi answered${c.adapter && c.adapter !== 'base' ? ' (' + escapeHtml(c.adapter) + ')' : ''}</div>
          <pre class="corr-orig">${escapeHtml(c.original) || '—'}</pre>
          <button type="button" class="btn btn-sm btn-ghost corr-seed" data-corr-seed="${rid}" title="Copy pi's answer into the editor so you can fix it in place">${icon('copy', 'icn-sm')} Start from this &amp; edit</button>
        </div>
        <div class="corr-col">
          <label class="corr-label" for="corr-ideal-${rid}">should have answered</label>
          <textarea class="corr-ideal" id="corr-ideal-${rid}" data-corr-ideal="${rid}" rows="6" spellcheck="false" placeholder="Write the answer pi should have given…">${escapeHtml(c.ideal)}</textarea>
        </div>
      </div>
    </div>`;
  }).join('');
  list.querySelectorAll('[data-corr-ideal]').forEach(ta => ta.addEventListener('input', () => {
    const c = correctionsBasket.find(x => x.request_id === ta.dataset.corrIdeal);
    if (c) { c.ideal = ta.value; saveCorrections(); corrSyncUpsertDebounced(c); markCorrState(c); updateCorrFoot(); }
  }));
  list.querySelectorAll('[data-corr-seed]').forEach(b => b.addEventListener('click', () => {
    const c = correctionsBasket.find(x => x.request_id === b.dataset.corrSeed);
    if (!c) return;
    const ta = document.getElementById('corr-ideal-' + c.request_id);
    if (ta) { ta.value = c.original; c.ideal = c.original; saveCorrections(); corrSyncUpsertDebounced(c); markCorrState(c); updateCorrFoot();
      ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }
  }));
  list.querySelectorAll('[data-corr-remove]').forEach(b => b.addEventListener('click', () => removeCorrection(b.dataset.corrRemove)));
  document.getElementById('corr-receipt-view')?.addEventListener('click', () => {
    selectPage('training');
    setTimeout(() => document.querySelector('#page-training [data-tab="queue"]')?.click(), 40);
  });
  document.getElementById('corr-receipt-verify')?.addEventListener('click', () => {
    if (!corrReceipt || !corrReceipt.firstPrompt) return;
    selectPage('playground');
    setTimeout(() => setupCompareReplay(corrReceipt.firstPrompt, '', corrReceipt.name), 60);
  });
  document.getElementById('corr-receipt-dismiss')?.addEventListener('click', () => { corrReceipt = null; renderCorrections(); });
  updateCorrFoot();
}
// Repaint one item's ready/todo affordance in place — no full re-render, so the
// textarea keeps focus and caret position while the operator types.
function markCorrState(c) {
  const ready = corrTrainable(c);
  document.querySelectorAll('.corr-item[data-corr]').forEach(item => {
    if (item.getAttribute('data-corr') !== c.request_id) return;
    item.classList.toggle('is-ready', ready); item.classList.toggle('is-todo', !ready);
  });
  document.querySelectorAll('[data-corr-state]').forEach(st => {
    if (st.getAttribute('data-corr-state') === c.request_id) st.innerHTML = corrStateHtml(ready);
  });
}
function updateCorrFoot() {
  const ready = correctionsBasket.filter(corrTrainable).length;
  const todo = correctionsBasket.length - ready;
  const admission = trainingOptimizerAdmissionState('sft', 'muon', 8);
  setText('corr-train-n', String(ready));
  const note = document.getElementById('corr-foot-note');
  if (note) note.textContent = todo > 0
    ? `${todo} still need${todo === 1 ? 's' : ''} an answer · only edited items train`
    : (ready > 0 ? 'These become one SFT job — the new adapter hot-swaps in when done' : '');
  const btn = document.getElementById('corr-train');
  if (btn) {
    btn.disabled = ready === 0 || !admission.ready;
    btn.title = admission.ready ? '' : admission.reason || 'Training capability unavailable';
  }
  const support = document.getElementById('corr-optimizer-support');
  if (support) support.textContent = optimizerSupportStatus('sft', 'muon', 8);
}
// The client-side corrections→SFT transform, used by "Build a dataset from
// your corrections" on the Evals Datasets tab. The Corrections card's Train
// button no longer builds rows client-side — it submits dataset
// "corrections:active" and the server applies the same (user prompt, ideal
// answer) transform in CorrectionsStore::trainable_rows, so a correction
// becomes the same chat row whichever path it takes.
function correctionsToSftExamples(corrections) {
  return corrections.map(c => ({ messages: [{ role: 'user', content: c.user }, { role: 'assistant', content: c.ideal }] }));
}
// The trainer refuses alpha/rank > 2.0 (unsafe LoRA scaling) and the server
// default alpha is 32 — so any rank below 16 MUST send a matching alpha or
// the job is rejected. 2×rank, capped at the 32 default, is the standard pair.
function loraAlphaFor(rank) { return Math.min(32, 2 * rank); }
// Pre-train flush: push every locally-edited row to the durable store and
// WAIT for the writes. The server resolves corrections:active from its own
// copy, so a still-debounced ideal edit would otherwise train stale text.
async function corrFlushToServer(rows) {
  const ids = new Set(rows.map(c => c.request_id));
  Object.keys(corrSyncTimers).forEach(id => {
    clearTimeout(corrSyncTimers[id]); delete corrSyncTimers[id]; ids.add(id);
  });
  const byId = new Map(correctionsBasket.map(c => [c.request_id, c]));
  await Promise.all([...ids].map(id => {
    const c = byId.get(id);
    if (!c) return Promise.resolve();
    return api('/v1/corrections', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(corrRowForServer(c)) });
  }));
}
async function trainFromCorrections() {
  const trainable = correctionsBasket.filter(corrTrainable);
  if (!trainable.length) { toast('Write at least one ideal answer (different from pi’s) before training', 'err'); return; }
  try {
    requireTrainingOptimizerAdmission('sft', 'muon', 8, 'Corrections SFT');
  } catch (error) {
    toast(error.message, 'err');
    return;
  }
  const nameInput = document.getElementById('corr-adapter-name');
  const name = ((nameInput && nameInput.value) || '').trim() || 'codebase-corrections';
  if (!/^[A-Za-z0-9._-]+$/.test(name)) { toast('Adapter name: letters, digits, . _ - only', 'err'); nameInput && nameInput.focus(); return; }
  const btn = document.getElementById('corr-train');
  if (btn) btn.disabled = true;
  try {
    // The durable store is what trains: dataset "corrections:active" makes
    // the SERVER resolve the trainable rows and mark them trained only when
    // the job COMPLETES. A failed job never marks anything, so every
    // hand-written ideal stays re-trainable instead of being burned by an
    // optimistic submit-time mark (the pre-0.4.2 dead-end).
    await corrFlushToServer(trainable);
    // learning_rate omitted on purpose: the server resolves the
    // per-optimizer default (Muon and AdamW want very different bands).
    const body = { dataset: 'corrections:active', config: { output_name: name, auto_load: true, epochs: 3, lora_rank: 8, lora_alpha: loraAlphaFor(8), optimizer: { kind: 'muon' } } };
    const res = await api('/v1/train/sft', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    toastTrainingSubmission(res, `Training ${name} from ${trainable.length} correction${trainable.length === 1 ? '' : 's'} — it will hot-swap in when done`);
    // Clear the submitted rows from the LOCAL view only. The durable rows
    // stay active server-side until the job completes; if it fails,
    // watchCorrectionsJob pulls them straight back into the basket.
    const submitted = new Set(trainable.map(c => c.request_id));
    correctionsBasket = correctionsBasket.filter(c => !submitted.has(c.request_id));
    // Leave a persistent in-card receipt instead of a disorienting auto-jump —
    // the user stays in context and chooses when to view the job. Keep the
    // first corrected prompt so "Verify the fix" can replay it base-vs-adapter,
    // and the job id so the queue poll can resolve this receipt's outcome.
    corrReceipt = { name, count: trainable.length, firstPrompt: trainable[0].user, jobId: res.job_id, state: 'training', error: null };
    saveCorrections(); renderCorrections();
    if (typeof pollTraining === 'function') pollTraining();
  } catch (e) { toast(e.message || 'Could not submit training', 'err'); }
  finally { updateCorrFoot(); }
}
// Resolve an in-flight corrections-train receipt against the queue poll.
// Failed → flip the receipt to its error state and pull the (never-marked)
// rows back from the durable store so the user can fix and train again.
function watchCorrectionsJob(data) {
  if (!corrReceipt || !corrReceipt.jobId || corrReceipt.state !== 'training') return;
  const job = (data.completed || []).find(j => j.job_id === corrReceipt.jobId);
  if (!job) return;
  if (job.state === 'failed') {
    corrReceipt.state = 'failed';
    corrReceipt.error = job.error || 'training failed';
    corrSyncFromServer(); // restores the basket, then re-renders
  } else {
    corrReceipt.state = 'done';
  }
  renderCorrections();
}
function initCorrections() {
  loadCorrections();
  renderCorrections();
  // Pull the durable store: corrections captured on another machine (or
  // filed by pi via POST /v1/corrections) appear here too.
  corrSyncFromServer();
  document.getElementById('corr-clear')?.addEventListener('click', clearCorrections);
  document.getElementById('corr-train')?.addEventListener('click', trainFromCorrections);
  // Inline adapter-name validation so a bad name isn't a submit-time surprise.
  const nameInput = document.getElementById('corr-adapter-name');
  if (nameInput) nameInput.addEventListener('input', () => {
    const ok = !nameInput.value.trim() || /^[A-Za-z0-9._-]+$/.test(nameInput.value.trim());
    nameInput.classList.toggle('input-invalid', !ok);
  });
}
window.addCorrectionFromRequest = addCorrectionFromRequest;

/* First-run journey strip: three milestones computed from live caches.
   - server:  /health answered (lastHealth set)
   - agent:   a request from a RECOGNIZED coding agent seen (not just any curl)
   - adapter: at least one trained adapter exists
   Each pending step is a button that takes you to the action; the strip
   disappears forever once all three are done (or on dismiss). */
const JOURNEY_KEY = 'kiln.journey.dismissed.v1';
function updateJourneyStrip() {
  const strip = document.getElementById('journey-strip');
  if (!strip) return;
  let dismissed = false;
  try { dismissed = localStorage.getItem(JOURNEY_KEY) === '1'; } catch {}
  const serverUp = !!lastHealth;
  const rows = recentRequestsCache || [];
  // Only EXTERNAL traffic completes "Agent connected": the dashboard's own
  // requests (Test connection, Playground — `client: "dashboard"`) never
  // count, while curl DOES — a user who followed the curl tab really did
  // connect something. When curl is the only client seen, the step carries
  // an inline nudge to point a coding agent here next.
  const externalKeys = rows.filter(r => !rowIsDashboard(r)).map(r => clientFromUA(r.user_agent).key);
  const agentSeen = externalKeys.some(k => k && k !== 'unknown');
  const curlOnly = agentSeen && externalKeys.every(k => !k || k === 'unknown' || k === 'curl');
  const adapterTrained = nonBaseAdapterCount() > 0;
  const allDone = serverUp && agentSeen && adapterTrained;
  if (dismissed || allDone) {
    // Once everything is done, retire the guide permanently so it never
    // reappears if an adapter is later deleted.
    if (allDone) { try { localStorage.setItem(JOURNEY_KEY, '1'); } catch {} }
    strip.hidden = true;
    return;
  }
  strip.hidden = false;
  const states = { server: serverUp, agent: agentSeen, adapter: adapterTrained };
  let nextMarked = false;
  for (const [k, done] of Object.entries(states)) {
    const el = strip.querySelector(`[data-journey="${k}"]`);
    if (!el) continue;
    el.classList.toggle('is-done', done);
    const isNext = !done && !nextMarked;
    if (isNext) nextMarked = true;
    el.classList.toggle('is-next', isNext);
  }
  // Inline (not hover-only) state on the agent step: when curl is the only
  // external client seen, the milestone is honestly complete but the natural
  // next move — pointing a real coding agent here — gets said out loud.
  const agentSub = strip.querySelector('[data-journey="agent"] .journey-sub');
  if (agentSub) {
    agentSub.textContent = curlOnly
      ? 'curl seen — point a coding agent here next'
      : 'point pi or opencode here';
  }
}
document.getElementById('journey-dismiss')?.addEventListener('click', () => {
  try { localStorage.setItem(JOURNEY_KEY, '1'); } catch {}
  const strip = document.getElementById('journey-strip');
  if (strip) strip.hidden = true;
});
document.querySelectorAll('.journey-step').forEach(el => el.addEventListener('click', () => {
  const k = el.dataset.journey;
  if (k === 'server') { window.open('https://ericflo.github.io/kiln/quickstart.html', '_blank', 'noopener'); }
  else if (k === 'agent') {
    // When the embedded terminal is usable, the fastest "agent connected" is
    // one click away — launch pi right here instead of copying snippets.
    if (el.dataset.terminalReady === '1') selectPage('terminal');
    else openConnect();
  }
  else if (k === 'adapter') { selectPage('training'); document.getElementById('training-tab-sft')?.click(); }
}));

// Update the flywheel ribbon from whatever live caches are populated. Called
// from the various polls; tolerant of missing data (cold start).
function updateFlywheel() {
  const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
  const rows = recentRequestsCache || [];
  // Client count means EXTERNAL clients — the dashboard's own traffic
  // (`client: "dashboard"`) is not an agent and never inflates this node.
  const agents = new Set(rows.filter(r => !rowIsDashboard(r)).map(r => clientFromUA(r.user_agent).key));
  set('fw-agents', agents.size || 0);
  set('fw-agents-sub', agents.size ? (agents.size === 1 ? '1 client' : agents.size + ' clients') : 'none yet');
  set('fw-traffic', rows.length || 0);
  set('fw-traffic-sub', rows.length ? 'recent requests' : 'no traffic yet');

  const tj = (typeof trainingJobsCache !== 'undefined') ? trainingJobsCache : null;
  const running = tj && tj.running ? 1 : 0;
  const queued = tj && Array.isArray(tj.queued) ? tj.queued.length : 0;
  const done = tj && Array.isArray(tj.completed) ? tj.completed.length : 0;
  const activeJobs = running + queued;
  // Corrections in the basket are pending training — surface them on the Train
  // node so the ribbon reflects the live to-do, not just submitted jobs.
  const corrReady = (typeof correctionsBasket !== 'undefined' && typeof corrTrainable === 'function')
    ? correctionsBasket.filter(corrTrainable).length : 0;
  const corrTotal = (typeof correctionsBasket !== 'undefined') ? correctionsBasket.length : 0;
  set('fw-train', activeJobs > 0 ? activeJobs : (corrTotal || done));
  set('fw-train-sub',
    corrReady ? corrReady + ' correction' + (corrReady === 1 ? '' : 's') + ' ready'
    : running ? 'running now'
    : corrTotal ? corrTotal + ' correction' + (corrTotal === 1 ? '' : 's') + ' to answer'
    : queued ? queued + ' queued'
    : done ? done + ' completed' : 'idle');

  // Eval node derives from JS state (evalJobCounts, set by refreshEvalJobs) —
  // never from the nav badge's textContent. The badge counts LIVE jobs only
  // (queued + running), so reading it back here used to claim "no evals yet"
  // the moment the last job finished, right next to a "+N pts vs base"
  // verdict computed from those same completed evals.
  const ec = (typeof evalJobCounts !== 'undefined') ? evalJobCounts : null;
  const evalRunning = ec ? ec.running : 0;
  const evalQueued = ec ? ec.queued : 0;
  const evalLive = evalRunning + evalQueued;
  const evalCompleted = ec ? ec.completed : 0;
  // "No evals" is a strong claim — only make it from a LOADED jobs list
  // (ec !== null). Before the first /v1/eval/jobs response lands the count is
  // unknown, so the node keeps its neutral placeholder instead of asserting
  // absence (same loaded-cache rule trainingJobsCache consumers follow).
  const noEvals = !!ec && evalCompleted === 0 && evalLive === 0;
  set('fw-eval', ec ? String(evalLive > 0 ? evalLive : evalCompleted) : '—');

  const active = lastHealth && lastHealth.active_adapter;
  set('fw-active', active || 'base');
  // Lead with the win verdict when the active adapter has a compare eval — the
  // flywheel's payoff ("is what's serving pi better than base?") right on the ribbon.
  const activeVerdict = active && typeof adapterCompareVerdict === 'function' ? adapterCompareVerdict(active) : null;
  const subEl = document.getElementById('fw-active-sub');
  if (subEl) {
    subEl.classList.remove('fw-win', 'fw-loss');
    if (activeVerdict) {
      // Same gate as every other verdict surface: fw-win/fw-loss (and the
      // "+N pts vs base" claim) only at p < SIGN_TEST_ALPHA. Ungated → neutral
      // text, neither ribbon class.
      const sig = activeVerdict.significant === true;
      const detail = typeof activeVerdict.p === 'number'
        ? `sign test improved ${activeVerdict.improved} / regressed ${activeVerdict.regressed}, ${fmtSignTestP(activeVerdict.p)}`
        : '';
      subEl.title = detail;
      if (!sig && Math.abs(activeVerdict.delta) > 0.5) {
        subEl.textContent = `${activeVerdict.delta > 0 ? '+' : ''}${activeVerdict.delta.toFixed(1)} pts — not enough evidence`;
      } else if (Math.abs(activeVerdict.delta) <= 0.5) {
        subEl.textContent = 'matches base';
      } else {
        subEl.textContent = `${activeVerdict.delta > 0 ? '+' : ''}${activeVerdict.delta.toFixed(1)} pts vs base`;
        if (activeVerdict.delta > 0.5) subEl.classList.add('fw-win');  // green: proven better (significant)
        else subEl.classList.add('fw-loss');                           // red: proven regression (significant)
      }
    } else {
      subEl.textContent = active ? 'hot-swapped LoRA' : 'base model';
      subEl.removeAttribute('title');
    }
  }
  // Caution the eval node when an adapter is live but never evaluated.
  set('fw-eval-sub',
    evalRunning ? evalRunning + ' running'
    : evalQueued ? evalQueued + ' queued'
    : (active && noEvals) ? 'not evaluated yet'
    : noEvals ? 'no evals yet'
    : evalCompleted ? evalCompleted + ' completed'
    : 'suites & jobs');

  // Highlight the single most valuable next stage (the "hot" node). A genuine
  // downstream GAP (active adapter that was never evaluated) wins over an
  // already-running stage — that's the move the operator should make next.
  let hot = 'connect';
  if (corrReady > 0) hot = 'train';            // you authored fixes — train them in
  else if (active && noEvals) hot = 'eval';    // adapter live but unproven — verify it
  else if (running) hot = 'train';
  else if (rows.length > 0 && !active) hot = 'train';
  else if (active) hot = 'swap';
  else if (rows.length > 0) hot = 'train';
  document.querySelectorAll('.flywheel-node').forEach(n => n.classList.toggle('hot', n.dataset.fw === hot));

  // Explanatory + clickable-cue tooltips, keyed to the live value so they read
  // as state ("3 clients seen") not generic help. Every node is a button → say
  // where clicking goes.
  const tip = (fw, t) => { const n = document.querySelector(`.flywheel-node[data-fw="${fw}"]`); if (n) n.title = t; };
  tip('connect', `${agents.size || 0} distinct client${agents.size === 1 ? '' : 's'} seen in recent traffic — click for connection setup`);
  tip('traffic', `${rows.length} request${rows.length === 1 ? '' : 's'} in the live ring — click to jump to Recent requests`);
  tip('train', running ? 'A training job is running — click to view the queue'
    : corrReady ? `${corrReady} correction${corrReady === 1 ? '' : 's'} ready to train — click to the Corrections basket`
    : 'Train an adapter from your corrections — click for the Training queue');
  tip('eval', (active && noEvals) ? 'Your active adapter has no eval yet — click to prove it beats base'
    : noEvals ? 'No evals run yet — click to score an adapter' : 'Eval suites & results — click to view');
  tip('swap', active ? `Serving adapter "${active}" — click to manage / hot-swap adapters` : 'Serving the base model — click to load a trained adapter');

  // The journey strip reads the same caches — keep it in lockstep.
  updateJourneyStrip();
}

function bindFlywheel() {
  const go = {
    connect: () => openConnect(),
    traffic: () => { selectPage('overview'); document.getElementById('recent-requests-panel')?.scrollIntoView({ behavior: 'smooth', block: 'center' }); },
    // When corrections are pending, the actionable place is the Corrections card
    // on Overview (answer + train), not the Training queue.
    train: () => {
      const hasCorr = (typeof correctionsBasket !== 'undefined') && correctionsBasket.length > 0;
      if (hasCorr) { selectPage('overview'); setTimeout(() => document.getElementById('corrections-card')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 40); }
      else selectPage('training');
    },
    eval: () => selectPage('evals'),
    swap: () => selectPage('adapters'),
  };
  document.querySelectorAll('.flywheel-node').forEach(n => n.addEventListener('click', () => go[n.dataset.fw]?.()));
}

async function initConnect() {
  const baseEl = document.getElementById('connect-base-url');
  if (baseEl) { baseEl.textContent = connectBaseUrl(); baseEl.title = connectBaseUrl(); }
  // One shared resolver with the Playground: if the server is mid-cold-start
  // this fails harmlessly and pollHealth keeps retrying until a real id
  // arrives (applyServedModelId then upgrades the rendered snippets in place).
  await loadServedModelId();
  const modelEl = document.getElementById('connect-model');
  if (modelEl) { modelEl.textContent = connectModelId; modelEl.title = connectModelId; }
  renderConnectSnippets('pi');
  renderConnectMetricsSnippet();
  const tr = document.getElementById('connect-test-result');
  if (tr && !tr.textContent.trim()) tr.innerHTML = 'Once connected, your agent&rsquo;s calls stream into <strong>Recent requests</strong> below.';
  wireTablist(document.querySelector('.connect-tabs'), {
    onSelect: tab => selectConnectTab(tab.dataset.connectTab),
  });
  document.getElementById('connect-panel')?.addEventListener('click', (e) => {
    const f = e.target.closest('[data-copy-target]');
    if (f) { copyText(document.getElementById(f.dataset.copyTarget)?.textContent || '', f); return; }
    const c = e.target.closest('[data-copy-code]');
    if (c) { copyText(c.parentElement.querySelector('pre')?.innerText || '', c); return; }
  });
  const col = document.getElementById('connect-collapsed');
  col?.addEventListener('click', openConnect);
  col?.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openConnect(); } });
  document.getElementById('connect-trigger')?.addEventListener('click', openConnect);
  document.getElementById('connect-test')?.addEventListener('click', testConnection);
  bindFlywheel();
  updateFlywheel();
}

async function testConnection() {
  const btn = document.getElementById('connect-test');
  const out = document.getElementById('connect-test-result');
  if (!out) return;
  out.className = 'connect-test-result'; out.textContent = 'Testing…';
  if (btn) btn.disabled = true;
  const t0 = performance.now();
  try {
    const res = await fetch(connectBaseUrl() + '/chat/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Kiln-Client': 'dashboard' },
      body: JSON.stringify({ model: connectModelId, messages: [{ role: 'user', content: 'Reply with the single word: connected' }], max_tokens: 8, temperature: 0 }),
    });
    const ms = Math.round(performance.now() - t0);
    if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b.error?.message || b.detail || `HTTP ${res.status}`); }
    const data = await res.json();
    const reply = (data.choices?.[0]?.message?.content || '').trim().slice(0, 40);
    out.className = 'connect-test-result ok';
    out.innerHTML = `${icon('check', 'icn-sm')} Connected — <strong>${escapeHtml(connectModelId)}</strong> replied in ${ms} ms${reply ? ' · &ldquo;' + escapeHtml(reply) + '&rdquo;' : ''}`;
  } catch (e) {
    out.className = 'connect-test-result err';
    out.innerHTML = `${icon('warning', 'icn-sm')} ${escapeHtml(e.message || 'Request failed')} — the model may still be loading. See <a href="https://ericflo.github.io/kiln/troubleshooting.html" target="_blank" rel="noopener">Troubleshooting</a>.`;
  } finally { if (btn) btn.disabled = false; }
}

// aria-busy may flip only during a panel's FIRST load. The pollers call this
// on every tick (2-3s), and once content has rendered, busy/idle thrash is
// pure screen-reader noise. The `false` arm — every poller's `finally`, which
// runs after both the success render and the failure HTML — marks the panel
// loaded, so the guard is central and call sites stay untouched.
function setPanelBusy(id, busy) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (busy) {
    if (!el.dataset.loaded) el.setAttribute('aria-busy', 'true');
  } else {
    el.setAttribute('aria-busy', 'false');
    el.dataset.loaded = '1';
  }
  return el;
}

// Write a terse one-line transition announcement into a visually-hidden
// role="status" node. The polled data panels are deliberately NOT aria-live
// (a content-keyed repaint of a 30-row list would be re-read in full); these
// nodes are the screen-reader channel for the few transitions a sighted user
// would actually notice. Re-setting identical text fires no live-region
// mutation, so repeats get a non-breaking-space nudge to still announce.
function announceStatus(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = (el.textContent === msg) ? msg + '\u00a0' : msg;
}

// --- Shared tablist wiring (roadmap PR 19) ---
// One keyboard + click contract for every [role=tablist] in the dashboard:
// ArrowRight/ArrowLeft wrap around, Home/End jump to the edges, and
// selection FOLLOWS focus on arrow keys (automatic activation — matching
// the original training-tabs behavior). Only [role=tab] descendants
// participate: decorative children (the Distill group labels/separators)
// are skipped by construction.
//
// The helper owns NO selection logic. onSelect(tab, { focus }) must route
// to the tablist's existing select path (selectTrainingTab, selectEvalsTab,
// the distill select fn, selectConnectTab, selectPage) so localStorage
// restores, lazy refreshes, and the deep-link hash writes at the END of
// those functions keep working unchanged. After onSelect runs, the roving
// tabindex is re-asserted from aria-selected so tablists whose select fn
// predates roving focus still end up with exactly one Tab stop.
function wireTablist(root, { onSelect }) {
  if (!root) return;
  const tabsOf = () => Array.from(root.querySelectorAll('[role="tab"]'));
  const applyRovingTabindex = () => {
    tabsOf().forEach(t => { t.tabIndex = t.getAttribute('aria-selected') === 'true' ? 0 : -1; });
  };
  const select = (tab, focus) => {
    onSelect(tab, { focus });
    applyRovingTabindex();
    if (focus) tab.focus();
  };
  root.addEventListener('click', event => {
    const tab = event.target.closest('[role="tab"]');
    if (!tab || !root.contains(tab)) return;
    select(tab, false);
  });
  root.addEventListener('keydown', event => {
    const tab = event.target.closest('[role="tab"]');
    if (!tab) return;
    const tabs = tabsOf();
    const index = tabs.indexOf(tab);
    if (index === -1) return;
    let nextTab = null;
    if (event.key === 'ArrowRight') nextTab = tabs[(index + 1) % tabs.length];
    if (event.key === 'ArrowLeft') nextTab = tabs[(index - 1 + tabs.length) % tabs.length];
    if (event.key === 'Home') nextTab = tabs[0];
    if (event.key === 'End') nextTab = tabs[tabs.length - 1];
    if (nextTab) {
      event.preventDefault();
      select(nextTab, true);
    }
  });
  // Normalize the initial state: static markup without explicit tabindex
  // attributes (Connect tabs, primary nav) starts with every tab in the
  // Tab order — collapse that to the selected tab only.
  applyRovingTabindex();
}

// --- Training Tabs ---
function selectTrainingTab(tab, focus = false) {
  const panel = tab.closest('.card');
  panel.querySelectorAll('[role="tab"]').forEach(item => {
    const selected = item === tab;
    item.classList.toggle('active', selected);
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  panel.querySelectorAll('[role="tabpanel"]').forEach(tabPanel => {
    const selected = tabPanel.id === tab.getAttribute('aria-controls');
    tabPanel.classList.toggle('active', selected);
    tabPanel.hidden = !selected;
    tabPanel.inert = !selected;
  });
  if (focus) tab.focus();
  try { localStorage.setItem('kiln.trainingSubTab', tab.dataset.tab); } catch {}
  // Deep-link hash for the sub-tab (no-op when this activation is itself
  // hash-driven, or when Training isn't the frontmost page). Living here —
  // not in the click handler — covers arrow-key navigation and programmatic
  // .click() callers too.
  pushSubTabHash('training');
}

wireTablist(document.querySelector('[data-training-tabs]'), {
  onSelect: (tab, { focus }) => selectTrainingTab(tab, focus),
});

// Restore last visited training sub-tab so users return to Submit SFT
// (or GRPO) instead of always-Queue after a refresh. Hash-suppressed: this
// is the NO-HASH fallback — when the URL carries an explicit sub-tab the
// boot route pass applies it after this and wins.
try {
  const lastTrainingSubTab = localStorage.getItem('kiln.trainingSubTab');
  if (lastTrainingSubTab && lastTrainingSubTab !== 'queue') {
    const target = document.getElementById(`training-tab-${lastTrainingSubTab}`);
    if (target) withHashWritesSuppressed(() => selectTrainingTab(target));
  }
} catch {}

// --- Formatting ---
function fmtBytes(b) {
  if (b == null) return '-';
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  if (b < 1073741824) return (b / 1048576).toFixed(1) + ' MB';
  return (b / 1073741824).toFixed(2) + ' GB';
}
function fmtGb(gb) {
  if (gb == null) return '-';
  return gb.toFixed(1) + ' GB';
}
function fmtDuration(secs) {
  if (secs == null) return '-';
  secs = Math.floor(secs);
  if (secs < 60) return secs + 's';
  if (secs < 3600) return Math.floor(secs / 60) + 'm ' + (secs % 60) + 's';
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return h + 'h ' + m + 'm';
}

// --- Hero stats helpers ---
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}
function setHtml(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

// --- Health Polling ---
let lastHealth = null;

async function pollHealth() {
  const statusPanel = setPanelBusy('server-status', true);
  try {
    const h = await api('/health');
    lastHealth = h;
    const ub = document.getElementById('unreachable-banner');
    if (ub) ub.hidden = true;
    renderHeader(h);
    renderServerStatus(h);
    updateRuntimeGraphLive(h);
    updateFlywheel();
    // Cold-start follow-up: /v1/models may have failed (or listed nothing)
    // while weights loaded, leaving the fallback model id baked into the
    // Connect snippets and Playground bodies. Piggyback the retry on this
    // 2s poll — no extra interval — until a real id resolves, then stop
    // (the resolved flag short-circuits).
    if (!servedModelIdResolved) loadServedModelId();
  } catch (e) {
    document.getElementById('status-dot').className = 'status-dot offline';
    document.getElementById('status-text').textContent = 'offline';
    if (statusPanel) statusPanel.innerHTML = apiFailureHtml('Server status', e, 'pollHealth');
    // A server that doesn't answer at all gets a loud banner with the fix,
    // not just a quiet topbar dot. Clears itself on the next good poll.
    if (e && e.unreachable) {
      lastHealth = null;
      const b = document.getElementById('unreachable-banner');
      if (b) { setText('unreachable-origin', window.location.origin); b.hidden = false; }
    }
  } finally {
    setPanelBusy('server-status', false);
  }
}

document.getElementById('unreachable-retry')?.addEventListener('click', () => {
  const b = document.getElementById('unreachable-retry');
  if (b) { b.disabled = true; b.textContent = 'Checking…'; setTimeout(() => { b.disabled = false; b.textContent = 'Retry now'; }, 1200); }
  pollHealth();
});

function renderHeader(h) {
  const ok = h.status === 'ok';
  const dot = document.getElementById('status-dot');
  if (dot) dot.className = 'status-dot ' + (ok ? 'ok' : 'degraded');
  setText('status-text', ok ? 'Running' : (h.status || 'unknown'));
  const model = document.getElementById('header-model');
  if (model) { model.textContent = h.model || '—'; model.title = h.model || ''; }
  setText('header-backend', h.backend || '—');
  setText('header-uptime', fmtDuration(h.uptime_seconds));
  setText('header-adapter', h.active_adapter || 'base');
}

// Content key for the server-status card. The panel is innerHTML-swapped, so
// repainting identical content on the 2s poll would flash the donut and wipe
// hover/selection state. The card renders from exactly ONE place (here) —
// a second writer racing this one is how the donut used to vanish.
let lastServerStatusKey = null;
function renderServerStatus(h) {
  const el = document.getElementById('server-status');
  const gpu = h.gpu_memory;
  const sched = h.scheduler;
  const key = JSON.stringify([gpu, sched, h.checks]);
  // Repaint when content changed, or when the failure path (`.api-failure`
  // from pollHealth's catch) owns the panel and we're recovering.
  if (key === lastServerStatusKey && el.firstElementChild && !el.querySelector('.api-failure')) return;
  lastServerStatusKey = key;

  let vramHtml = '';
  let donutHtml = '';
  if (gpu && gpu.total_vram_gb > 0) {
    const total = gpu.total_vram_gb;
    const model = gpu.model_gb || 0;
    const kv = gpu.kv_cache_gb || 0;
    const train = gpu.training_budget_gb || 0;
    const free = Math.max(0, total - model - kv - train);
    const pct = v => ((v / total) * 100).toFixed(1);
    vramHtml = `
      <div class="vram-bar-wrap">
        <div class="vram-meta">
          <span style="font-size: var(--text-2xs); text-transform: uppercase; letter-spacing: var(--tracking-caps); color: var(--text-muted); font-weight: 600;">GPU VRAM</span>
          <span class="vram-meta-total"><strong>${fmtGb(total)}</strong> total</span>
        </div>
        <div class="vram-bar">
          <div class="vram-seg" style="width:${pct(model)}%">${model > 1.5 ? fmtGb(model) : ''}</div>
          <div class="vram-seg" style="width:${pct(kv)}%">${kv > 1.5 ? fmtGb(kv) : ''}</div>
          <div class="vram-seg" style="width:${pct(train)}%">${train > 1.5 ? fmtGb(train) : ''}</div>
          <div class="vram-seg" style="width:${pct(free)}%"></div>
        </div>
        <div class="vram-legend">
          <span class="model" title="GPU memory used by the model weights (Qwen3.5-4B + Marlin packed quantized projections).">Model ${fmtGb(model)}</span>
          <span class="kv" title="GPU memory reserved for the paged attention KV cache.">KV cache ${fmtGb(kv)}</span>
          <span class="train" title="GPU memory budget reserved for online LoRA training (SFT and GRPO).">Training ${fmtGb(train)}</span>
          <span class="free" title="Unallocated GPU memory.">Free ${fmtGb(free)}</span>
        </div>
      </div>
    `;
    const slices = [
      { label: 'Model',            value: model, color: 'var(--accent)' },
      { label: 'KV cache',         value: kv,    color: 'var(--info-fg)' },
      { label: 'Training reserve', value: train, color: 'var(--success-fg)' },
      { label: 'Free',             value: free,  color: 'var(--surface-3)' },
    ].filter(s => s.value > 0);
    const legendRows = slices.map(s => `<div class="vram-legend-row">
      <span class="vram-legend-swatch" style="background:${s.color};"></span>
      <span class="vram-legend-name">${escapeHtml(s.label)}</span>
      <span class="vram-legend-value">${s.value.toFixed(1)}G</span>
    </div>`).join('');
    donutHtml = `<div class="vram-donut" style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">
      ${donutChartSvg(slices, { size: 120, stroke: 20, centerLabel: (model + kv).toFixed(1) + 'G', centerSub: 'used / ' + total.toFixed(1) + 'G' })}
      <div class="vram-legend">${legendRows}</div>
    </div>`;
  }

  let schedHtml = '';
  if (sched) {
    schedHtml = `
      <div class="sched-stats" style="margin-top: var(--space-4);">
        <div class="sched-stat" title="Requests admitted but not yet running."><div class="num">${sched.waiting}</div><div class="lbl">Waiting</div></div>
        <div class="sched-stat" title="Requests currently decoding tokens this step."><div class="num">${sched.running}</div><div class="lbl">Running</div></div>
        <div class="sched-stat" title="KV cache blocks allocated to active requests."><div class="num">${sched.blocks_used}</div><div class="lbl">Blocks used</div></div>
        <div class="sched-stat" title="KV cache blocks available for new requests."><div class="num">${sched.blocks_free}</div><div class="lbl">Blocks free</div></div>
      </div>
    `;
  }

  let checksHtml = '';
  if (h.checks && h.checks.length > 0) {
    checksHtml = `<div class="checks-row">
      ${h.checks.map(c => `<span class="check-chip ${c.pass ? 'pass' : 'fail'}">${escapeHtml(c.name)}</span>`).join('')}
    </div>`;
  }

  el.innerHTML = (vramHtml + schedHtml + checksHtml + donutHtml) || '<div class="empty">No data</div>';
}

/* =====================================================================
   Runtime config expander — GET /v1/config (device-scoped capacity,
   live usable memory, governor policy, batching and streaming-prefill
   resolution, KV geometry, training policy, and exact memory-budget
   partitions).
   The <details> shell is static in index.html as a SIBLING of the keyed
   #server-status region: renderServerStatus innerHTML-swaps that element
   whenever its content key changes (and pollHealth's failure path
   overwrites it wholesale), so anything rendered inside it is destroyed
   by the 2s poll — exactly how the VRAM donut used to vanish. Out here
   the open state and the rendered content survive repaints by
   construction. Policy is fetched once per open; the existing health poll
   refreshes only the graph live/current-phase values. Refresh re-fetches the
   full contract; failures render a quiet retry line and never throw.
   ===================================================================== */
let runtimeConfigLoaded = false;
let runtimeConfigRenderSeq = 0;
let runtimeConfigSnapshot = null;
let runtimeConfigRequest = null;
let runtimeConfigRequestSeq = 0;

function fetchRuntimeConfig(force = false) {
  if (!force && runtimeConfigSnapshot) return Promise.resolve(runtimeConfigSnapshot);
  if (!force && runtimeConfigRequest) return runtimeConfigRequest;

  const seq = ++runtimeConfigRequestSeq;
  const request = api('/v1/config')
    .then(cfg => {
      if (seq === runtimeConfigRequestSeq) {
        runtimeConfigSnapshot = cfg;
        updatePlaygroundThinkingBudgetDefaults(cfg);
        updateTrainingOptimizerSupport(cfg);
      }
      return cfg;
    })
    .catch(error => {
      if (seq === runtimeConfigRequestSeq) {
        runtimeConfigSnapshot = null;
        runtimeConfigLoaded = false;
        markTrainingOptimizerSupportFetchFailed(error);
        const details = document.getElementById('runtime-config');
        const body = document.getElementById('runtime-config-body');
        if (details?.open && body) body.innerHTML = runtimeConfigFailureHtml(error);
      }
      throw error;
    })
    .finally(() => {
      if (runtimeConfigRequest === request) runtimeConfigRequest = null;
    });
  runtimeConfigRequest = request;
  return request;
}

function runtimeConfigRow(label, valueHtml, title) {
  return `<div class="rc-row"${title ? ` title="${escapeHtml(title)}"` : ''}>
    <span class="rc-label">${escapeHtml(label)}</span>
    <span class="rc-value">${valueHtml}</span>
  </div>`;
}

function runtimeConfigFailureHtml(error) {
  return `<div class="hint">Couldn't load /v1/config — ${escapeHtml(error?.message || 'request failed')}</div>
    <div class="rc-actions"><button class="btn btn-sm" type="button" data-rc-refresh>Retry</button></div>`;
}

function graphUnavailableLabel(reason) {
  return typeof reason === 'string' && reason.includes('busy') ? 'busy' : 'unavailable';
}

function graphReasonChip(reason) {
  return reason
    ? ` <span class="rc-source" title="${escapeHtml(`Closed missing-data reason: ${reason}.`)}">${escapeHtml(String(reason).replaceAll('_', ' '))}</span>`
    : '';
}

function updateRuntimeGraphLive(health) {
  const graph = health?.decode_runtime?.rocm_graphs;
  if (!graph || typeof graph !== 'object') return;

  const live = document.getElementById('runtime-graph-live-value');
  if (live) {
    const state = typeof graph.state === 'string'
      ? graph.state
      : graphUnavailableLabel(graph.unavailable_reason);
    const armed = graph.capture_enabled === true
      ? ' <span class="rc-source" title="Native capture and replay remain armed.">capture armed</span>'
      : '';
    live.innerHTML = `<strong>${escapeHtml(state)}</strong>${graphReasonChip(graph.unavailable_reason)}${armed}`;
  }

  const current = document.getElementById('runtime-graph-current-phase-value');
  if (current) {
    const available = graph.phase_telemetry_available === true;
    const phase = available
      ? typeof graph.current_phase === 'string'
        ? graph.current_phase.replaceAll('_', ' ')
        : 'idle'
      : graphUnavailableLabel(graph.phase_telemetry_unavailable_reason);
    const elapsed = available
      && graph.current_phase != null
      && Number.isFinite(graph.current_phase_elapsed_micros)
      ? ` <span class="rc-source" title="Monotonic time spent in the currently active graph lifecycle phase.">${escapeHtml(fmtMsShort(graph.current_phase_elapsed_micros / 1000))}</span>`
      : '';
    current.innerHTML = `<strong>${escapeHtml(phase)}</strong>${elapsed}${graphReasonChip(graph.phase_telemetry_unavailable_reason)}`;
  }
}

// Renders the operational subset of /v1/config (shape: api/config.rs
// ConfigResponse) plus a raw pretty-printed JSON toggle so diagnostics that
// do not belong in the compact summary remain available.
function renderRuntimeConfigBody(cfg) {
  const applicationPaths = cfg.paths || {};
  const vram = cfg.vram || {};
  const live = vram.live || {};
  const governor = vram.governor || {};
  const acceleratorRuntime = cfg.accelerator_runtime || {};
  const ktApiMode = acceleratorRuntime.kt_api_mode || {};
  const fullAttentionScoreBudget = acceleratorRuntime.full_attention_score_budget_mib || {};
  const vulkanDeviceIndex = acceleratorRuntime.vulkan_device_index || {};
  const vulkanValidation = acceleratorRuntime.vulkan_validation || {};
  const cudaKernelProfile = acceleratorRuntime.cuda_kernel_profile || {};
  const cudaMarlinProfile = acceleratorRuntime.cuda_marlin_profile || {};
  const cudaFlashBackwardMode = acceleratorRuntime.cuda_flash_backward_mode || {};
  const metalKernelProfile = acceleratorRuntime.metal_kernel_profile || {};
  const rocmSynchronization = acceleratorRuntime.rocm_synchronization_mode || {};
  const rocmStridedBatchedMatmul = acceleratorRuntime.rocm_strided_batched_matmul_mode || {};
  const rocmBf16MatmulOutput = acceleratorRuntime.rocm_bf16_matmul_output_mode || {};
  const rocmKernelProfile = acceleratorRuntime.rocm_kernel_profile || {};
  const rocmGraphMode = acceleratorRuntime.rocm_graph_mode || {};
  const rocmGraphCache = acceleratorRuntime.rocm_graph_cache_entries || {};
  const rocmGraphBudget = acceleratorRuntime.rocm_graph_cache_max_bytes || {};
  const cudaGraphs = cfg.cuda_graphs || {};
  const rocmGraphs = cfg.rocm_graphs || {};
  const rocmGraphUnavailableReason = cfg.rocm_graphs_unavailable_reason;
  const rocmGraphTelemetry = cfg.rocm_graph_telemetry || {};
  const rocmGraphTelemetryUnavailableReason = cfg.rocm_graph_telemetry_unavailable_reason;
  const rocmGraphFallbacks = rocmGraphs.fallbacks || {};
  const kv = cfg.kv_cache || {};
  const train = cfg.training || {};
  const b = cfg.memory_budget || {};
  const generation = cfg.generation || {};
  const batching = cfg.batching || {};
  const batchingConfiguration = batching.configuration || {};
  const rowwiseDecode = batchingConfiguration.rowwise_decode || {};
  const prefixAwareAdmission = batchingConfiguration.prefix_aware_admission || {};
  const prefillAdmissionQuantum = batchingConfiguration.prefill_admission_quantum || {};
  const actorCycleIdle = batchingConfiguration.actor_cycle_idle || {};
  const streamingPrefill = cfg.streaming_prefill || {};
  const streamingDispatch = streamingPrefill.dispatch || {};
  const streamingThreshold = streamingPrefill.threshold_tokens || {};
  const streamingBaseTile = streamingPrefill.tile_tokens || {};
  const streamingTapeTile = streamingPrefill.tape_tile_tokens || {};
  const streamingDetachedTile = streamingPrefill.detached_full_attn_tile_tokens || {};
  const streamingBoundaryTile = streamingPrefill.detached_full_attn_boundary_tile_tokens || {};
  const streamingReplayTile = streamingPrefill.detached_full_attn_tape_replay_tile_tokens || {};
  const streamingLastTokenLmHead = streamingPrefill.last_token_lm_head || {};
  const srcChip = s => s == null ? '' : ` <span class="rc-source" title="Where this value came from">${escapeHtml(String(s))}</span>`;
  const flagChip = (label, title) => ` <span class="rc-source"${title ? ` title="${escapeHtml(title)}"` : ''}>${escapeHtml(label)}</span>`;
  const onOff = v => v ? 'on' : 'off';
  const enabledState = v => v === true ? 'on' : v === false ? 'off' : '—';
  const activeState = v => v === true ? 'active' : v === false ? 'inactive' : '—';
  const hasOwn = (object, field) => Object.prototype.hasOwnProperty.call(object, field);
  const num = v => (typeof v === 'number' && isFinite(v)) ? v.toLocaleString() : '—';
  const closedSum = (object, fields) => fields.every(field => Number.isFinite(object[field]))
    ? fields.reduce((total, field) => total + object[field], 0)
    : null;
  const graphPostCaptureRejections = closedSum(rocmGraphs, [
    'entry_capacity_rejections',
    'byte_budget_rejections',
    'accounting_incomplete_rejections',
  ]);
  const graphPreCaptureSkips = closedSum(rocmGraphs, [
    'pre_capture_entry_capacity_skips',
    'pre_capture_byte_budget_skips',
    'pre_capture_accounting_incomplete_skips',
    'pre_capture_memory_reservation_denied_skips',
    'memory_governor_selector_mismatch_skips',
  ]);
  const graphPreCandidateHeadroom = rocmGraphTelemetry.pre_candidate_headroom_phase || {};
  const graphCandidateWarm = rocmGraphTelemetry.candidate_warm_phase || {};
  const graphPreNativeReservation = rocmGraphTelemetry.pre_native_reservation_phase || {};
  const graphNativeCapture = rocmGraphTelemetry.native_capture_phase || {};
  const graphRejectedCandidateCleanup = rocmGraphTelemetry.rejected_candidate_cleanup_phase || {};
  const graphLiveState = cfg.rocm_graphs == null
    ? graphUnavailableLabel(rocmGraphUnavailableReason)
    : enabledState(rocmGraphs.enabled);
  const graphCurrentPhase = typeof rocmGraphTelemetry.current_phase === 'string'
    ? rocmGraphTelemetry.current_phase.replaceAll('_', ' ')
    : cfg.rocm_graph_telemetry == null ? graphUnavailableLabel(rocmGraphTelemetryUnavailableReason) : 'idle';
  const graphCurrentPhaseElapsed = Number.isFinite(rocmGraphTelemetry.current_phase_elapsed_micros)
    && rocmGraphTelemetry.current_phase != null
    ? flagChip(
      fmtMsShort(rocmGraphTelemetry.current_phase_elapsed_micros / 1000),
      'Monotonic time spent in the currently active graph lifecycle phase.',
    )
    : '';
  const graphPhaseMax = phase => Number.isFinite(phase.max_duration_micros)
    ? fmtMsShort(phase.max_duration_micros / 1000)
    : '—';
  const graphPhaseSlowChip = (phase, label) => Number.isFinite(phase.slow) && phase.slow > 0
    ? flagChip(`${num(phase.slow)} ${label} slow`, `${label} phase calls taking at least 100 ms.`)
    : '';
  const tokens = value => (typeof value === 'number' && isFinite(value))
    ? `${num(value)} tokens`
    : '—';
  const configuredTokens = object => {
    if (!hasOwn(object, 'configured')) return '—';
    return object.configured == null ? 'auto' : tokens(object.configured);
  };
  const optionalTokens = (object, field) => {
    if (!hasOwn(object, field)) return '—';
    return object[field] == null ? 'none' : tokens(object[field]);
  };
  const streamingDispatchRule = rule => {
    if (!rule || typeof rule !== 'object') return '—';
    if (rule.policy === 'never') return 'never';
    if (rule.policy === 'all_non_empty') return 'all non-empty prompts';
    if (rule.policy === 'prompt_tokens_at_least') {
      return typeof rule.minimum_prompt_tokens === 'number' && isFinite(rule.minimum_prompt_tokens)
        ? `at least ${num(rule.minimum_prompt_tokens)} tokens`
        : 'at configured threshold';
    }
    return rule.policy == null ? '—' : String(rule.policy).replaceAll('_', ' ');
  };
  const streamingConfiguredMode = streamingDispatch.configured_mode == null
    ? '—'
    : String(streamingDispatch.configured_mode);
  const streamingThresholdOverride = hasOwn(streamingThreshold, 'override_applied_to_backend_auto_policy')
    ? streamingThreshold.override_applied_to_backend_auto_policy ? 'applied' : 'not applied'
    : '—';
  const streamingEffectiveThresholdSource = !hasOwn(streamingThreshold, 'effective_for_auto_mode')
    ? null
    : streamingThreshold.override_applied_to_backend_auto_policy === true
      ? streamingThreshold.configured_source
      : 'backend_policy';
  const derivedTileInput = object => hasOwn(object, 'effective') ? 'derived from detached' : '—';
  const immutableState = streamingPrefill.immutable_after_startup === true
    ? 'immutable'
    : streamingPrefill.immutable_after_startup === false ? 'mutable' : '—';
  const restartState = streamingPrefill.restart_required_to_change === true
    ? 'required'
    : streamingPrefill.restart_required_to_change === false ? 'not required' : '—';
  const policySource = (object, field) => hasOwn(object, field) ? srcChip('backend_policy') : '';
  const gib = v => (typeof v === 'number' && isFinite(v)) ? v.toFixed(2) + ' GiB' : '—';
  const memory = (gibValue, bytesValue) => {
    const exact = typeof bytesValue === 'number' && Number.isSafeInteger(bytesValue) && bytesValue >= 0
      ? `<span class="rc-memory-exact">${bytesValue.toLocaleString()} B</span>`
      : '';
    return `<strong>${gib(gibValue)}</strong>${exact}`;
  };
  const configuredCap = vram.configured_capacity_gib == null
    ? '<strong>not set</strong>'
    : `${memory(vram.configured_capacity_gib, vram.configured_capacity_bytes)}${vram.configured_capacity_clamped ? flagChip('clamped', 'The requested cap exceeded safely detected capacity and was reduced.') : ''}`;
  const checkpointPolicy = train.checkpoint_policy || {};
  const checkpointPolicyLabel = checkpointPolicy.mode === 'explicit_segments'
    ? 'explicit segments'
    : checkpointPolicy.mode === 'disabled'
      ? 'disabled'
      : checkpointPolicy.mode === 'auto'
        ? 'auto'
        : '—';
  const retainedSegments = checkpointPolicy.mode === 'disabled' && checkpointPolicy.segments != null
    ? flagChip(`${num(checkpointPolicy.segments)} retained`, 'The segment count remains part of checkpoint identity while execution is disabled.')
    : '';
  const checkpointBoundaryPolicy = train.checkpoint_boundary_policy || {};
  const checkpointBoundaryMode = checkpointBoundaryPolicy.recompute_mode == null
    ? '—'
    : String(checkpointBoundaryPolicy.recompute_mode).replaceAll('_', ' ');
  const checkpointBoundaryStride = !hasOwn(checkpointBoundaryPolicy, 'anchor_stride')
    ? '—'
    : checkpointBoundaryPolicy.anchor_stride == null
      ? 'auto'
      : num(checkpointBoundaryPolicy.anchor_stride);
  const checkpointBoundaryStrideChip = typeof checkpointBoundaryPolicy.anchor_stride === 'number'
    && isFinite(checkpointBoundaryPolicy.anchor_stride)
    ? flagChip('explicit', 'The configured stride overrides cache-target-based automatic selection.')
    : '';
  const checkpointBoundaryCacheTargetGib = typeof checkpointBoundaryPolicy.cache_target_bytes === 'number'
    && isFinite(checkpointBoundaryPolicy.cache_target_bytes)
    && checkpointBoundaryPolicy.cache_target_bytes >= 0
    ? gib(checkpointBoundaryPolicy.cache_target_bytes / (1024 ** 3))
    : '—';
  const optimizerSupport = train.optimizer_support;
  const optimizerEntries = Array.isArray(optimizerSupport?.optimizers) ? optimizerSupport.optimizers : [];
  const optimizerName = kind => kind === 'adam_w' ? 'AdamW' : kind === 'sgd' ? 'SGD' : kind === 'muon' ? 'Muon' : String(kind || 'unknown');
  const optimizerWorkloadName = workload => workload === 'distill_refresh'
    ? 'Distill refresh'
    : String(workload || 'unknown').toUpperCase();
  const optimizerTupleKinds = Array.isArray(optimizerSupport?.optimizer_tuple_kinds)
    ? optimizerSupport.optimizer_tuple_kinds.map(kind => optimizerName(kind)).join(', ') || 'none'
    : 'unavailable';
  const optimizerWorkloads = Array.isArray(optimizerSupport?.workloads) ? optimizerSupport.workloads : [];
  const workloadSummary = workload => {
    const workloadLabel = optimizerWorkloadName(workload);
    const entry = optimizerWorkloads.find(candidate => candidate?.workload === workload);
    if (!entry) return { value: 'unavailable', detail: `The ${workloadLabel} workload descriptor is missing.` };
    const allowed = Array.isArray(entry.allowed_optimizer_kinds)
      ? entry.allowed_optimizer_kinds.map(kind => optimizerName(kind)).join(', ') || 'none'
      : 'invalid allowlist';
    return {
      value: entry.supported === true ? allowed : 'unavailable',
      detail: entry.supported === true
        ? `Optimizer kinds admitted for ${workloadLabel}. Exact rank admission remains tuple-specific.`
        : entry.unavailable_reason || `${workloadLabel} is unsupported by the resident server path.`,
    };
  };
  const backendOptimizerImplementations = optimizerEntries
    .filter(entry => entry?.backend_implementation?.supported === true)
    .map(entry => optimizerName(entry.kind))
    .join(', ') || (optimizerSupport ? 'none' : 'unavailable');
  const nativeHookOptimizers = optimizerEntries
    .filter(entry => entry?.backend_implementation?.native_device_hook === true)
    .map(entry => optimizerName(entry.kind))
    .join(', ') || (optimizerSupport ? 'none' : 'unavailable');
  const muonSupport = optimizerEntries.find(entry => entry?.kind === 'muon');
  const muonRank = muonSupport?.optimizer_tuple?.lora_rank;
  const muonRankLabel = muonRank && Number.isInteger(muonRank.minimum)
    ? `${muonRank.minimum}${Number.isInteger(muonRank.maximum) ? `..${muonRank.maximum}` : '+'}`
    : 'unavailable';
  const muonBackendMaximum = Number.isInteger(muonRank?.backend_maximum)
    ? num(muonRank.backend_maximum)
    : 'none';
  const muonModelMaximum = Number.isInteger(muonRank?.model_maximum)
    ? num(muonRank.model_maximum)
    : 'unavailable';
  const sftWorkload = workloadSummary('sft');
  const grpoWorkload = workloadSummary('grpo');
  const opdWorkload = workloadSummary('opd');
  const distillRefreshWorkload = workloadSummary('distill_refresh');
  const reclaimRequested = governor.reclaim_mode_requested;
  const reclaimEffective = governor.reclaim_mode_effective;
  const reclaimDisabledByProfile = governor.reclaim_disabled_by_serving_profile === true;
  const reclaimRequestedChip = reclaimRequested != null && reclaimEffective != null && reclaimRequested !== reclaimEffective
    ? flagChip(
        `requested ${String(reclaimRequested)}`,
        reclaimDisabledByProfile
          ? 'The serving profile disabled the requested reclaim behavior.'
          : 'The effective reclaim behavior differs from the requested configuration.',
      )
    : '';
  const configuredPrefillQuantum = Object.prototype.hasOwnProperty.call(prefillAdmissionQuantum, 'configured')
    ? prefillAdmissionQuantum.configured == null ? 'auto' : num(prefillAdmissionQuantum.configured)
    : '—';
  return `
    <div class="rc-groups">
      <div class="rc-group">
        <div class="rc-group-title">Application paths</div>
        ${runtimeConfigRow('Cache root', `<strong>${escapeHtml(applicationPaths.cache_root == null ? '—' : String(applicationPaths.cache_root))}</strong>${srcChip(applicationPaths.cache_root_source)}`, 'Absolute process-lifetime root shared by autotune, Vulkan pipeline, and transposed-weight caches.')}
        ${runtimeConfigRow('Change requires restart', `<strong>${applicationPaths.restart_required_to_change === true ? 'required' : '—'}</strong>`, 'The cache root is installed before model and accelerator cache construction.')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Capacity</div>
        ${runtimeConfigRow('Device', `<strong>${escapeHtml(vram.probe_selector == null ? '—' : String(vram.probe_selector))}</strong>${vram.unified ? flagChip('unified', 'The accelerator and host share one physical memory pool.') : ''}`, 'The device-scoped probe selected for the running backend.')}
        ${runtimeConfigRow('Physical', `${memory(vram.physical_capacity_gib, vram.physical_capacity_bytes)}${srcChip(vram.physical_capacity_source)}`, 'Safe capacity detected at startup for the selected device.')}
        ${runtimeConfigRow('Configured cap', configuredCap, 'Optional typed capacity cap. It can reduce detected capacity but cannot expand it.')}
        ${runtimeConfigRow('Effective cap', `${memory(vram.effective_capacity_gib, vram.effective_capacity_bytes)}${srcChip(vram.effective_capacity_source)}`, 'Immutable capacity used by planning and memory admission.')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Live memory</div>
        ${runtimeConfigRow('Sample total', memory(live.total_gib, live.total_bytes), 'Current bounded total reported by the device-scoped live probe.')}
        ${runtimeConfigRow('Used now', memory(live.used_gib, live.used_bytes), 'Memory currently in use at the time of this snapshot.')}
        ${runtimeConfigRow('Probe available', `${memory(live.available_gib, live.available_bytes)}${srcChip(live.source)}`, 'Current availability after driver, host, cgroup, and unified-memory safety bounds.')}
        ${runtimeConfigRow('Cap-aware', memory(live.effective_capacity_available_gib, live.effective_capacity_available_bytes), 'Current availability after the immutable effective capacity cap.')}
        ${runtimeConfigRow('Usable', memory(live.usable_after_governor_floor_gib, live.usable_after_governor_floor_bytes), 'Live cap-aware memory remaining after the governor floor.')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Governor</div>
        ${runtimeConfigRow('Capacity limit', memory(governor.capacity_limit_gib, governor.capacity_limit_bytes), 'Immutable capacity limit enforced by memory admission.')}
        ${runtimeConfigRow('Free floor', memory(governor.floor_gib, governor.floor_bytes), 'Memory kept free rather than admitted to model work.')}
        ${runtimeConfigRow('Probe cadence', `<strong>${typeof governor.probe_ms === 'number' && isFinite(governor.probe_ms) ? num(governor.probe_ms) + ' ms' : '—'}</strong>`, 'How often the memory governor refreshes live observations.')}
        ${runtimeConfigRow('Reclaim', `<strong>${escapeHtml(reclaimEffective == null ? '—' : String(reclaimEffective))}</strong>${srcChip(governor.reclaim_mode_source)}${reclaimRequestedChip}`, 'Effective process-lifetime reclaim policy. A differing requested value is shown alongside it.')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Accelerator execution</div>
        ${runtimeConfigRow('Policy schema', `<strong>${escapeHtml(acceleratorRuntime.schema_id || '—')}</strong>${acceleratorRuntime.version == null ? '' : flagChip(`v${acceleratorRuntime.version}`, 'Accelerator runtime policy schema version.')}`, 'Versioned process-lifetime accelerator policy installed before device initialization.')}
        ${runtimeConfigRow('Vulkan kernel policy', `<strong>${escapeHtml(acceleratorRuntime.vulkan_kernel_policy_schema_id || '—')}</strong>`, 'Device-neutral immutable Vulkan selection and dispatch policy compiled into this build.')}
        ${runtimeConfigRow('Vulkan device policy', `<strong>${escapeHtml(acceleratorRuntime.vulkan_device_policy_schema_id || '—')}</strong>`, 'Immutable Vulkan physical-device and validation-layer policy installed before logical-device creation.')}
        ${runtimeConfigRow('Vulkan physical device', `<strong>${vulkanDeviceIndex.effective == null ? 'automatic' : num(vulkanDeviceIndex.effective)}</strong>${srcChip(vulkanDeviceIndex.source)}`, 'Automatic prefers a discrete Vulkan GPU and otherwise selects the first enumerated device. An explicit zero-based index fails startup when unavailable.')}
        ${runtimeConfigRow('Vulkan validation', `<strong>${vulkanValidation.effective === true ? 'enabled' : vulkanValidation.effective === false ? 'disabled' : '—'}</strong>${srcChip(vulkanValidation.source)}`, 'Validation layers are startup-only and intended for backend diagnosis.')}
        ${runtimeConfigRow('Kiln-tensor API routes', `<strong>${escapeHtml(ktApiMode.effective || '—')}</strong>${srcChip(ktApiMode.source)}`, 'Immutable adapter route set. Auto uses qualified defaults; all and disabled are diagnostic comparisons.')}
        ${runtimeConfigRow('Full-attention score ceiling', `<strong>${Number.isFinite(fullAttentionScoreBudget.effective) ? num(fullAttentionScoreBudget.effective) + ' MiB' : '—'}</strong>${srcChip(fullAttentionScoreBudget.source)}`, 'Immutable exact-attention scratch ceiling. ROCm online attention uses at most 1024 MiB and live pressure rejects admission instead of changing tile geometry.')}
        ${runtimeConfigRow('CUDA kernel profile', `<strong>${escapeHtml(cudaKernelProfile.effective || '—')}</strong>${srcChip(cudaKernelProfile.source)}`, 'Immutable twenty-five-route CUDA model and backend policy. Native default preserves established accelerated dispatch without making a hardware-qualification claim; portable fallback declines every owned route.')}
        ${runtimeConfigRow('CUDA Marlin layout', `<strong>${escapeHtml(cudaMarlinProfile.effective || '—')}</strong>${srcChip(cudaMarlinProfile.source)}`, 'Immutable projection layout. Disabled preserves BF16 weights; attention MLP adds W4A16 full-attention Q and MLP projections; attention MLP GDN also packs the quality-sensitive GDN output projection.')}
        ${runtimeConfigRow('CUDA FlashAttention backward', `<strong>${escapeHtml(cudaFlashBackwardMode.effective || '—')}</strong>${srcChip(cudaFlashBackwardMode.source)}`, 'Startup-owned training backward mode. Fast preserves the established accumulation path; deterministic selects split accumulation for exact replay and diagnosis.')}
        ${runtimeConfigRow('Metal kernel profile', `<strong>${escapeHtml(metalKernelProfile.effective || '—')}</strong>${srcChip(metalKernelProfile.source)}`, 'Immutable forty-six-route Metal backend policy. Native default preserves forty-five established native routes while leaving custom LM-head argmax disabled; portable fallback declines every owned route.')}
        ${runtimeConfigRow('CUDA graph policy', `<strong>${cudaGraphs.effective_policy_enabled === true ? 'enabled' : cudaGraphs.effective_policy_enabled === false ? 'disabled' : '—'}</strong>${cudaGraphs.requested === true && cudaGraphs.capture_allowed_by_serving_profile === false ? flagChip('profile blocked', 'The serving profile disables live graph capture.') : ''}`, 'Configured CUDA graph request after immutable serving-profile resolution.')}
        ${runtimeConfigRow('CUDA graph cache', `<strong>${Number.isFinite(cudaGraphs.max_cached_graphs) ? num(cudaGraphs.max_cached_graphs) : '—'}</strong>`, 'Startup-fixed retained single-row CUDA graph entry limit; valid range 1 through 64.')}
        ${runtimeConfigRow('CUDA graph contract', `<strong>${cudaGraphs.stable_paged_metadata === true ? 'stable metadata' : '—'}</strong>${cudaGraphs.batched_capture_available === false ? flagChip('single-row only', 'Batched CUDA graph capture is unavailable until it has NVIDIA correctness and resilience evidence.') : ''}`, 'Graph-stable paged metadata is mandatory and the unqualified batched route cannot be enabled by a hidden switch.')}
        ${runtimeConfigRow('Synchronization', `<strong>${escapeHtml(rocmSynchronization.effective || '—')}</strong>${srcChip(rocmSynchronization.source)}`, 'Effective ROCm synchronization discipline. Legacy host barriers remain the portable default.')}
        ${runtimeConfigRow('Strided batched matmul', `<strong>${escapeHtml(rocmStridedBatchedMatmul.effective || '—')}</strong>${srcChip(rocmStridedBatchedMatmul.source)}`, 'Portable execution uses per-row GEMMs. Strided batching is an explicit diagnostic route.')}
        ${runtimeConfigRow('BF16 matmul output', `<strong>${escapeHtml(rocmBf16MatmulOutput.effective || '—')}</strong>${srcChip(rocmBf16MatmulOutput.source)}`, 'Portable execution uses F32 output followed by an on-device BF16 cast. Native BF16 output is an explicit diagnostic route.')}
        ${runtimeConfigRow('ROCm kernel profile', `<strong>${escapeHtml(rocmKernelProfile.effective || '—')}</strong>${srcChip(rocmKernelProfile.source)}`, 'Native default enables correctness-qualified paged decode and falls back portably for ineligible requests. Portable fallback is diagnostic.')}
        ${runtimeConfigRow('Graph configured', `<strong>${escapeHtml(rocmGraphMode.configured || '—')}</strong>${srcChip(rocmGraphMode.source)}`, 'Configured graph lifecycle before serving-profile resolution.')}
        ${runtimeConfigRow('Graph effective', `<strong>${escapeHtml(rocmGraphMode.effective || '—')}</strong>`, 'Effective immutable graph lifecycle. Stable serving resolves profile mode to guarded lazy capture; maintenance resolves to eager execution.')}
        ${runtimeConfigRow('Graph cache', `<strong>${num(rocmGraphCache.effective)}</strong>${srcChip(rocmGraphCache.source)}`, 'Bounded number of retained ROCm graph entries; valid range 1 through 64.')}
        ${runtimeConfigRow('Graph byte budget', `<strong>${Number.isFinite(rocmGraphBudget.effective) ? fmtBytes(rocmGraphBudget.effective) : '—'}</strong>${srcChip(rocmGraphBudget.source)}`, 'Bounded requested physical bytes retained by graph-owned tensors, capture arenas, workspaces, and owner state; opaque HIP object overhead is reported separately.')}
        ${runtimeConfigRow('Graph live', `<span id="runtime-graph-live-value"><strong>${escapeHtml(graphLiveState)}</strong>${graphReasonChip(rocmGraphUnavailableReason)}${rocmGraphs.capture_enabled === true ? flagChip('capture armed', 'Native capture and replay remain armed.') : ''}</span>`, 'Live graph-runner state. Failed or unclassified physical settlement quarantines the circuit breaker until restart; an acknowledged capture rollback may continue in eager mode.')}
        ${runtimeConfigRow('Graph current phase', `<span id="runtime-graph-current-phase-value"><strong>${escapeHtml(graphCurrentPhase)}</strong>${graphCurrentPhaseElapsed}${graphReasonChip(rocmGraphTelemetryUnavailableReason)}</span>`, 'Live attribution independent of the model and graph-runner locks for graph headroom checks, candidate preparation, native publication, and rejected-candidate cleanup.')}
        ${runtimeConfigRow('Graph retained', `<strong>${Number.isFinite(rocmGraphs.retained_bytes) ? fmtBytes(rocmGraphs.retained_bytes) : '—'}</strong>${Number.isFinite(rocmGraphs.peak_retained_bytes) ? flagChip(`peak ${fmtBytes(rocmGraphs.peak_retained_bytes)}`, 'Highest deduplicated retained-byte measurement after admission.') : ''}`, 'Deduplicated requested physical bytes held by stable tensors, capture arenas, private-stream workspaces, and owner state.')}
        ${runtimeConfigRow('Graph entries live', `<strong>${num(rocmGraphs.captured_graph_count)} / ${num(rocmGraphs.max_cached_graphs)}</strong>`, 'Current retained native graph entries and the installed entry limit.')}
        ${runtimeConfigRow('Graph slots', `<strong>${num(rocmGraphs.active_graph_slot_count)} active · ${num(rocmGraphs.idle_graph_slot_count)} idle</strong>`, 'Persistent recurrent and convolution state slots. Active means assigned to a live logical row; idle includes reusable slots reserved for a batched width.')}
        ${runtimeConfigRow('Graph accounting', `<strong>${rocmGraphs.retained_bytes_accounting_complete === true ? 'exact' : rocmGraphs.retained_bytes_accounting_complete === false ? 'incomplete' : '—'}</strong>${Number.isFinite(rocmGraphs.opaque_native_object_count) ? flagChip(`${num(rocmGraphs.opaque_native_object_count)} opaque`, 'HIP graph, executable, stream, and event object sizes are not queryable from the driver.') : ''}`, 'Exact means every retained tensor mapped to physical ROCm allocation metadata.')}
        ${runtimeConfigRow('Graph transient', `<strong>${Number.isFinite(rocmGraphTelemetry.last_transient_candidate_bytes) ? fmtBytes(rocmGraphTelemetry.last_transient_candidate_bytes) : '—'}</strong>${Number.isFinite(rocmGraphTelemetry.peak_transient_candidate_bytes) ? flagChip(`peak ${fmtBytes(rocmGraphTelemetry.peak_transient_candidate_bytes)}`, 'Largest exact pre-admission candidate measured after its settled warm pass.') : ''}`, 'Exact requested physical bytes in the latest graph candidate before cache admission, excluding already owned recurrent slot state.')}
        ${runtimeConfigRow('Graph headroom latency', `<strong>max ${graphPhaseMax(graphPreCandidateHeadroom)}</strong>${graphPhaseSlowChip(graphPreCandidateHeadroom, 'headroom')}`, 'Longest observed matching-device governor check and safely settled idle-owner or prior-geometry reclamation before candidate allocation.')}
        ${runtimeConfigRow('Graph prepare latency', `<strong>warm ${graphPhaseMax(graphCandidateWarm)} · reserve ${graphPhaseMax(graphPreNativeReservation)}</strong>${graphPhaseSlowChip(graphCandidateWarm, 'warm')}${graphPhaseSlowChip(graphPreNativeReservation, 'reserve')}`, 'Longest observed candidate allocation and settled warm pass, followed by exact accounting, pressure reconciliation, reservation, and any pre-native cleanup.')}
        ${runtimeConfigRow('Graph native latency', `<strong>capture ${graphPhaseMax(graphNativeCapture)} · cleanup ${graphPhaseMax(graphRejectedCandidateCleanup)}</strong>${graphPhaseSlowChip(graphNativeCapture, 'capture')}${graphPhaseSlowChip(graphRejectedCandidateCleanup, 'cleanup')}`, 'Longest observed native capture through settled first launch, defensive cache admission and publication, and committed governor debit; cleanup covers settled rejected candidates.')}
        ${runtimeConfigRow('Graph cache actions', `<strong>${num(rocmGraphs.cache_admission_successes)} admitted · ${num(rocmGraphs.cache_evictions)} evicted</strong>${graphPostCaptureRejections == null ? '' : flagChip(`${num(graphPostCaptureRejections)} post-capture rejected`, 'Successfully launched candidates rejected by exact entry, byte, or accounting admission.')}`, 'Lifetime cache admissions, safely settled evictions, and post-capture admission rejections.')}
        ${runtimeConfigRow('Graph capture skips', `<strong>${num(graphPreCaptureSkips)}</strong>${Number.isFinite(rocmGraphs.pre_capture_accounting_incomplete_skips) && rocmGraphs.pre_capture_accounting_incomplete_skips > 0 ? flagChip(`${num(rocmGraphs.pre_capture_accounting_incomplete_skips)} accounting`, 'Capture was skipped because exact retained-allocation accounting could not be completed.') : ''}${Number.isFinite(rocmGraphs.pre_capture_memory_reservation_denied_skips) && rocmGraphs.pre_capture_memory_reservation_denied_skips > 0 ? flagChip(`${num(rocmGraphs.pre_capture_memory_reservation_denied_skips)} governor`, 'The process-wide memory governor denied transient candidate headroom.') : ''}${Number.isFinite(rocmGraphs.memory_governor_selector_mismatch_skips) && rocmGraphs.memory_governor_selector_mismatch_skips > 0 ? flagChip(`${num(rocmGraphs.memory_governor_selector_mismatch_skips)} device mismatch`, 'The process-wide governor was observing a different accelerator, so graph growth failed closed.') : ''}`, 'Candidates declined before native capture by entry capacity, retained-byte budget, incomplete accounting, global reservation denial, or governor device mismatch.')}
        ${runtimeConfigRow('Graph fallbacks', `<strong>${num(rocmGraphFallbacks.total)}</strong>${Number.isFinite(rocmGraphFallbacks.slow) ? flagChip(`${num(rocmGraphFallbacks.slow)} slow`, 'Fallbacks taking at least 100 ms end to end.') : ''}${Number.isFinite(rocmGraphFallbacks.max_duration_micros) ? flagChip(`max ${fmtMsShort(rocmGraphFallbacks.max_duration_micros / 1000)}`, 'Longest observed end-to-end eager graph fallback.') : ''}`, 'Closed-reason eager fallbacks and their observed pause envelope.')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">KV cache</div>
        ${runtimeConfigRow('Blocks', `<strong>${num(kv.num_blocks)}</strong>${srcChip(kv.num_blocks_source)}`, 'Paged-attention blocks allocated by the running backend, either automatically sized or explicitly configured.')}
        ${runtimeConfigRow('FP8 cache', `<strong>${onOff(kv.fp8_enabled)}</strong>`, 'Whether the KV cache stores keys/values in FP8 (halves cache memory per token).')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Streaming prefill</div>
        ${runtimeConfigRow('Dispatch input', '<strong>uncached prompt tokens</strong>', 'Dispatch is based on the prompt suffix not already satisfied by the prefix cache.')}
        ${runtimeConfigRow('Policy consumers', '<strong>inference + training</strong>', 'Inference and native training share this exact resolved process-lifetime policy.')}
        ${runtimeConfigRow('Mode configured', `<strong>${escapeHtml(streamingConfiguredMode)}</strong>${srcChip(streamingDispatch.configured_source)}`, 'Typed startup mode before backend policy resolution.')}
        ${runtimeConfigRow('Dispatch backend', `<strong>${escapeHtml(streamingDispatchRule(streamingDispatch.backend_policy))}</strong>${policySource(streamingDispatch, 'backend_policy')}`, 'Backend-selected automatic dispatch rule.')}
        ${runtimeConfigRow('Dispatch effective', `<strong>${escapeHtml(streamingDispatchRule(streamingDispatch.effective))}</strong>${srcChip(streamingDispatch.effective_source)}`, 'Immutable dispatch rule used by inference and native training.')}
        ${runtimeConfigRow('Threshold configured', `<strong>${configuredTokens(streamingThreshold)}</strong>${srcChip(streamingThreshold.configured_source)}`, 'Optional automatic-mode threshold override.')}
        ${runtimeConfigRow('Threshold backend', `<strong>${optionalTokens(streamingThreshold, 'backend_policy')}</strong>${policySource(streamingThreshold, 'backend_policy')}`, 'Backend automatic-dispatch threshold; none means the backend never dispatches automatically.')}
        ${runtimeConfigRow('Threshold effective', `<strong>${optionalTokens(streamingThreshold, 'effective_for_auto_mode')}</strong>${srcChip(streamingEffectiveThresholdSource)}`, 'Threshold used when the configured mode is auto.')}
        ${runtimeConfigRow('Threshold override', `<strong>${streamingThresholdOverride}</strong>`, 'Whether the configured threshold replaced a threshold-bearing backend auto policy.')}
        ${runtimeConfigRow('Base configured', `<strong>${configuredTokens(streamingBaseTile)}</strong>${srcChip(streamingBaseTile.configured_source)}`, 'Configured base streaming tile size; auto delegates to backend policy.')}
        ${runtimeConfigRow('Base backend', `<strong>${tokens(streamingBaseTile.backend_policy)}</strong>${policySource(streamingBaseTile, 'backend_policy')}`, 'Backend base streaming tile size.')}
        ${runtimeConfigRow('Base effective', `<strong>${tokens(streamingBaseTile.effective)}</strong>${srcChip(streamingBaseTile.effective_source)}`, 'Resolved base streaming tile size.')}
        ${runtimeConfigRow('Tape configured', `<strong>${configuredTokens(streamingTapeTile)}</strong>${srcChip(streamingTapeTile.configured_source)}`, 'Configured tape-forward tile size; auto inherits an explicit base tile before using backend policy.')}
        ${runtimeConfigRow('Tape backend', `<strong>${tokens(streamingTapeTile.backend_policy)}</strong>${policySource(streamingTapeTile, 'backend_policy')}`, 'Backend tape-forward tile size.')}
        ${runtimeConfigRow('Tape effective', `<strong>${tokens(streamingTapeTile.effective)}</strong>${srcChip(streamingTapeTile.effective_source)}`, 'Resolved tape-forward tile size and inheritance source.')}
        ${runtimeConfigRow('Detached configured', `<strong>${configuredTokens(streamingDetachedTile)}</strong>${srcChip(streamingDetachedTile.configured_source)}`, 'Configured detached full-attention tile size; auto inherits an explicit base tile before using backend policy.')}
        ${runtimeConfigRow('Detached backend', `<strong>${tokens(streamingDetachedTile.backend_policy)}</strong>${policySource(streamingDetachedTile, 'backend_policy')}`, 'Backend detached full-attention tile size.')}
        ${runtimeConfigRow('Detached effective', `<strong>${tokens(streamingDetachedTile.effective)}</strong>${srcChip(streamingDetachedTile.effective_source)}`, 'Resolved detached full-attention tile size and inheritance source.')}
        ${runtimeConfigRow('Boundary configured', `<strong>${derivedTileInput(streamingBoundaryTile)}</strong>${srcChip(streamingBoundaryTile.effective_source)}`, 'Boundary tiles are derived from the detached full-attention configuration rather than configured independently.')}
        ${runtimeConfigRow('Boundary backend', `<strong>${tokens(streamingBoundaryTile.backend_policy)}</strong>${policySource(streamingBoundaryTile, 'backend_policy')}`, 'Backend detached full-attention boundary tile size.')}
        ${runtimeConfigRow('Boundary effective', `<strong>${tokens(streamingBoundaryTile.effective)}</strong>${srcChip(streamingBoundaryTile.effective_source)}`, 'Resolved detached full-attention boundary tile size and inheritance source.')}
        ${runtimeConfigRow('Replay configured', `<strong>${derivedTileInput(streamingReplayTile)}</strong>${srcChip(streamingReplayTile.effective_source)}`, 'Tape-replay tiles are derived from the detached full-attention configuration rather than configured independently.')}
        ${runtimeConfigRow('Replay backend', `<strong>${tokens(streamingReplayTile.backend_policy)}</strong>${policySource(streamingReplayTile, 'backend_policy')}`, 'Backend detached full-attention tape-replay tile size.')}
        ${runtimeConfigRow('Replay effective', `<strong>${tokens(streamingReplayTile.effective)}</strong>${srcChip(streamingReplayTile.effective_source)}`, 'Resolved detached full-attention tape-replay tile size and inheritance source.')}
        ${runtimeConfigRow('Last-token configured', `<strong>${enabledState(streamingLastTokenLmHead.configured)}</strong>${srcChip(streamingLastTokenLmHead.configured_source)}`, 'Whether streaming prefill computes the LM head only for the final prompt token.')}
        ${runtimeConfigRow('Last-token effective', `<strong>${enabledState(streamingLastTokenLmHead.effective)}</strong>${srcChip(streamingLastTokenLmHead.effective_source)}`, 'Resolved final-token LM-head projection policy for streaming inference tiles.')}
        ${runtimeConfigRow('Startup policy', `<strong>${immutableState}</strong>`, 'The resolved policy does not change during this server process.')}
        ${runtimeConfigRow('Change requires restart', `<strong>${restartState}</strong>`, 'Configuration changes take effect on the next server start.')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Batching</div>
        ${runtimeConfigRow('Primary actor', `<strong>${activeState(batching.actor_active)}</strong>`, 'Whether the primary production batching actor is active in the current model state.')}
        ${runtimeConfigRow('Rowwise decode', `<strong>${enabledState(rowwiseDecode.enabled)}</strong>${srcChip(rowwiseDecode.source)}`, 'Whether batched decode executes one row at a time.')}
        ${runtimeConfigRow('Prefix admission', `<strong>${enabledState(prefixAwareAdmission.enabled)}</strong>${srcChip(prefixAwareAdmission.source)}`, 'Whether admission accounts for reusable prompt prefixes.')}
        ${runtimeConfigRow('Prefill configured', `<strong>${configuredPrefillQuantum}</strong>${srcChip(prefillAdmissionQuantum.configured_source)}`, 'Configured prompt-admission quantum; auto delegates to backend policy.')}
        ${runtimeConfigRow('Prefill policy', `<strong>${num(prefillAdmissionQuantum.backend_policy)}</strong>`, 'Backend-selected prompt-admission quantum before an explicit value or decode-width bound is applied.')}
        ${runtimeConfigRow('Prefill effective', `<strong>${num(prefillAdmissionQuantum.effective)}</strong>${srcChip(prefillAdmissionQuantum.effective_source)}`, 'Prompt-admission quantum used by the batching actor.')}
        ${runtimeConfigRow('Cycle idle', `<strong>${num(actorCycleIdle.milliseconds)}${Number.isFinite(actorCycleIdle.milliseconds) ? ' ms' : ''}</strong>${srcChip(actorCycleIdle.source)}${actorCycleIdle.enabled === true ? flagChip(`poll ${num(actorCycleIdle.command_poll_milliseconds)} ms`, 'Maximum control-command polling interval during the cooperative wait.') : ''}`, 'Intentional safe-boundary idle after actor cycles that advanced prefill or decode. Zero disables pacing; nonzero values trade throughput and latency for lower sustained accelerator duty cycle.')}
        ${runtimeConfigRow('Burst prefill', `<strong>${enabledState(batchingConfiguration.burst_prefill_admission)}</strong>`, 'Whether the backend admits a burst of prefill work between decode steps.')}
        ${runtimeConfigRow('Tile alignment', `<strong>${enabledState(batchingConfiguration.actor_prefill_tile_alignment_required)}</strong>`, 'Whether startup requires actor prompt chunks to preserve the backend-qualified streaming-prefill numerical boundary. ROCm enables this correctness contract.')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Training</div>
        ${runtimeConfigRow('Runtime device', `<strong>${escapeHtml(train.runtime_device == null ? '—' : String(train.runtime_device))}</strong>`, 'Immutable execution device bound to native training.')}
        ${runtimeConfigRow('Weight device', `<strong>${escapeHtml(train.model_weight_device == null ? '—' : String(train.model_weight_device))}</strong>`, 'Device representation of the frozen model weights.')}
        ${runtimeConfigRow('Native training', `<strong>${train.native_training_supported === true ? 'available' : 'unavailable'}</strong>`, train.native_training_supported === true ? 'The bound runtime and model-weight representation can execute native training.' : (train.native_training_unavailable_reason || 'Native training is unavailable on this backend.'))}
        ${runtimeConfigRow('Optimizer contract', `<strong>${escapeHtml(optimizerSupport?.schema?.id || 'unavailable')}</strong>${optimizerSupport?.schema?.version == null ? '' : flagChip(`v${optimizerSupport.schema.version}`, 'Optimizer support response schema version.')}`, 'Versioned product optimizer support derived from the resident runner.')}
        ${runtimeConfigRow('Optimizer backend', `<strong>${escapeHtml(optimizerSupport?.backend || '—')} · ${escapeHtml(optimizerSupport?.device || '—')}</strong>`, 'Backend and device identity used for optimizer admission.')}
        ${runtimeConfigRow('Base / LoRA dtype', `<strong>${escapeHtml(optimizerSupport?.base_weight_dtype || '—')} / ${escapeHtml(optimizerSupport?.resolved_lora_parameter_dtype || '—')}</strong>`, 'Resident base-weight dtype and the LoRA parameter dtype resolved by the training precision policy.')}
        ${runtimeConfigRow('Optimizer tuples', `<strong>${escapeHtml(optimizerTupleKinds)}</strong>`, 'Optimizer kinds whose immutable backend, dtype, rounding, and rank tuple is admitted for the resident weights. Workload admission is reported separately.')}
        ${runtimeConfigRow('SFT workload', `<strong>${escapeHtml(sftWorkload.value)}</strong>`, sftWorkload.detail)}
        ${runtimeConfigRow('GRPO workload', `<strong>${escapeHtml(grpoWorkload.value)}</strong>`, grpoWorkload.detail)}
        ${runtimeConfigRow('OPD workload', `<strong>${escapeHtml(opdWorkload.value)}</strong>`, opdWorkload.detail)}
        ${runtimeConfigRow('Distill refresh workload', `<strong>${escapeHtml(distillRefreshWorkload.value)}</strong>`, distillRefreshWorkload.detail)}
        ${runtimeConfigRow('Optimizer implementations', `<strong>${escapeHtml(backendOptimizerImplementations)}</strong>`, 'Backend optimizer implementations, including the CPU portable reference route. These facts do not by themselves make server training available.')}
        ${runtimeConfigRow('Native device hooks', `<strong>${escapeHtml(nativeHookOptimizers)}</strong>`, 'Accelerator-native optimizer hooks only; the CPU reference route is deliberately excluded.')}
        ${runtimeConfigRow('Optimizer rounding', `<strong>${escapeHtml((optimizerSupport?.rounding_modes || []).join(', ') || 'unavailable')}</strong>${optimizerSupport?.immutable_after_startup === true ? flagChip('immutable', 'Product optimizer rounding is fixed for the process lifetime.') : ''}`, 'Product training uses only the reported rounding policy; backend-implementation modes are retained separately in raw JSON.')}
        ${runtimeConfigRow('Muon rank', `<strong>${escapeHtml(muonRankLabel)}</strong>${muonRank?.live_memory_admission_required === true ? flagChip('admission required', 'Live memory admission revalidates this tuple and rejects the request if it does not fit.') : ''}`, 'Effective resident-model/backend Muon rank range. Live memory admission is a separate request-time rejection gate; it does not change this range or the requested rank.')}
        ${runtimeConfigRow('Muon rank ceilings', `<strong>backend ${escapeHtml(muonBackendMaximum)} · model ${escapeHtml(muonModelMaximum)}</strong>`, 'The effective maximum above is the model ceiling bounded by the optional backend ceiling. Backend none means the model ceiling is effective.')}
        ${runtimeConfigRow('Checkpoint policy', `<strong>${checkpointPolicyLabel}</strong>${retainedSegments}`, 'Immutable typed checkpoint policy for native training runs.')}
        ${runtimeConfigRow('Execution', `<strong>${onOff(train.checkpointing_enabled)}</strong>`, 'Gradient checkpointing trades recompute for activation memory during LoRA training.')}
        ${runtimeConfigRow('Effective segments', `<strong>${num(train.checkpoint_segments)}</strong>${srcChip(train.checkpoint_segments_source)}`, 'Resolved segment count and whether it was measured, conservatively selected, explicitly configured, or disabled.')}
        ${runtimeConfigRow('Boundary mode', `<strong>${escapeHtml(checkpointBoundaryMode)}</strong>`, 'Whether checkpointed SFT replays sparse segment boundaries automatically, always, or never.')}
        ${runtimeConfigRow('Recompute threshold', `<strong>${tokens(checkpointBoundaryPolicy.recompute_threshold_tokens)}</strong>`, 'In automatic mode, sequences at or above this token count replay sparse boundaries.')}
        ${runtimeConfigRow('Anchor stride', `<strong>${checkpointBoundaryStride}</strong>${checkpointBoundaryStrideChip}`, 'Explicit boundaries-per-anchor stride, or auto when the cache target selects the stride for each training shape.')}
        ${runtimeConfigRow('Anchor cache target', `<strong>${checkpointBoundaryCacheTargetGib}</strong>`, 'Process-lifetime memory target used to derive an automatic sparse-boundary anchor stride.')}
        ${runtimeConfigRow('Startup policy', '<strong>immutable</strong>', 'Checkpoint-boundary policy is resolved once and shared by every native training run in this process.')}
        ${runtimeConfigRow('Change requires restart', '<strong>required</strong>', 'Restart the server to apply checkpoint-boundary policy changes.')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Generation</div>
        ${runtimeConfigRow('Thinking default', `<strong>${generation.default_thinking_enabled == null ? 'template' : onOff(generation.default_thinking_enabled)}</strong>`)}
        ${runtimeConfigRow('Thinking tokens', `<strong>${generation.default_thinking_budget_tokens == null ? 'unlimited' : num(generation.default_thinking_budget_tokens)}</strong>`)}
        ${runtimeConfigRow('Thinking time', `<strong>${generation.default_thinking_budget_ms == null ? 'unlimited' : num(generation.default_thinking_budget_ms) + ' ms'}</strong>`)}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Memory budget</div>
        ${runtimeConfigRow('Budget total', memory(b.total_vram_gib, b.total_vram_bytes), 'Effective capacity partitioned by the startup memory budget.')}
        ${runtimeConfigRow('Model weights', memory(b.model_gib, b.model_bytes))}
        ${runtimeConfigRow('KV cache', memory(b.kv_cache_gib, b.kv_cache_bytes))}
        ${runtimeConfigRow('Training budget', memory(b.training_budget_gib, b.training_budget_bytes))}
        ${runtimeConfigRow('Inference fraction', `<strong>${(typeof b.inference_memory_fraction === 'number' && isFinite(b.inference_memory_fraction)) ? (b.inference_memory_fraction * 100).toFixed(0) + '%' : '—'}</strong>`, 'Fraction of usable VRAM reserved for inference (model + KV cache); the remainder is the training budget.')}
      </div>
    </div>
    <div class="rc-actions">
      <button class="btn btn-sm" type="button" data-rc-refresh>Refresh</button>
      <button class="btn btn-sm btn-ghost" type="button" data-rc-raw aria-expanded="false">Raw JSON</button>
    </div>
    <pre class="rc-raw" data-rc-raw-pre hidden>${escapeHtml(JSON.stringify(cfg, null, 2))}</pre>`;
}

async function loadRuntimeConfig(force = false) {
  const body = document.getElementById('runtime-config-body');
  if (!body) return;
  if (runtimeConfigLoaded && runtimeConfigSnapshot && !force) return;
  const seq = ++runtimeConfigRenderSeq;
  body.innerHTML = '<div class="hint">Loading GET /v1/config…</div>';
  try {
    const cfg = await fetchRuntimeConfig(force);
    if (seq !== runtimeConfigRenderSeq) return; // superseded by a newer refresh
    runtimeConfigLoaded = true;
    body.innerHTML = renderRuntimeConfigBody(cfg);
    updateRuntimeGraphLive(lastHealth);
  } catch (e) {
    if (seq !== runtimeConfigRenderSeq) return;
    runtimeConfigLoaded = false; // the next open retries automatically
    body.innerHTML = runtimeConfigFailureHtml(e);
  }
}

// Static shell: wire once at startup. `toggle` fires on open and close; the
// first open consumes Playground's shared snapshot, while Refresh forces one
// new request for both surfaces.
(function initRuntimeConfig() {
  const details = document.getElementById('runtime-config');
  if (!details) return;
  details.addEventListener('toggle', () => { if (details.open) loadRuntimeConfig(); });
  details.addEventListener('click', e => {
    if (e.target.closest('[data-rc-refresh]')) { loadRuntimeConfig(true); return; }
    const raw = e.target.closest('[data-rc-raw]');
    if (raw) {
      const pre = details.querySelector('[data-rc-raw-pre]');
      if (pre) { pre.hidden = !pre.hidden; raw.setAttribute('aria-expanded', String(!pre.hidden)); }
    }
  });
})();


// --- Decode Performance ---
// Cached last response from `/v1/stats/decode`, populated by `pollDecodePerf`.
// The Overview tab's tok/s sparkline reads from this so it never needs to
// re-fetch what the original poll already produced.
let lastDecode = null;
// Content key for the decode stats block. Stats live in their own sub-host so
// repaints never destroy the sparkline that refreshDecodeSparkline appends
// after it — two writers clobbering one panel is how the VRAM donut vanished.
let lastDecodeStatsKey = null;
async function pollDecodePerf() {
  const el = setPanelBusy('decode-perf-panel', true);
  if (!el) return;
  try {
    const data = await api('/v1/stats/decode');
    lastDecode = data;
    const window = data.window_secs ? Math.round(data.window_secs) : 60;
    const idle = !data.sample_count || data.sample_count < 1;

    let host = el.querySelector('.decode-stats-host');
    if (!host) {
      el.innerHTML = '';
      host = document.createElement('div');
      host.className = 'decode-stats-host';
      el.appendChild(host);
      lastDecodeStatsKey = null;
    }

    let key, html;
    if (idle) {
      key = 'idle|' + window;
      html = `<div class="sched-stats">
        <div class="sched-stat" title="Decode cadence across recent request-local token gaps."><div class="num">&mdash;</div><div class="lbl">tok/s</div></div>
        <div class="sched-stat" title="Median inter-token latency."><div class="num">&mdash;</div><div class="lbl">p50 ITL</div></div>
        <div class="sched-stat" title="99th-percentile inter-token latency."><div class="num">&mdash;</div><div class="lbl">p99 ITL</div></div>
        <div class="sched-stat" title="99.9th-percentile inter-token latency."><div class="num">&mdash;</div><div class="lbl">p99.9 ITL</div></div>
        <div class="sched-stat" title="Request-local inter-token gaps counted in this rolling window."><div class="num">0</div><div class="lbl">token gaps</div></div>
      </div>
      <div class="empty" style="margin-top: var(--space-4);">No recent token gaps in the last ${window}s. Send a message in <strong>Playground</strong> to populate metrics, or check <a href="/health" target="_blank" rel="noopener noreferrer">/health</a> if the server is still warming up.</div>`;
    } else {
      const tps = data.tok_per_sec.toFixed(1);
      const p50 = data.p50_itl_ms.toFixed(1);
      const p99 = data.p99_itl_ms.toFixed(1);
      const p999 = data.p999_itl_ms.toFixed(1);
      const mean = data.mean_itl_ms.toFixed(1);
      const stalls = Number(data.stall_count || 0);
      const unexplained = Number(data.unexplained_stall_count || 0);
      key = ['live', tps, p50, p99, p999, mean, stalls, unexplained, data.sample_count, window].join('|');
      html = `<div class="sched-stats">
        <div class="sched-stat" title="Decode cadence across recent request-local token gaps."><div class="num">${tps}</div><div class="lbl">tok/s</div></div>
        <div class="sched-stat" title="Median inter-token latency."><div class="num">${p50}<span style="font-size:0.55em;color:var(--text-muted);font-weight:500;"> ms</span></div><div class="lbl">p50 ITL</div></div>
        <div class="sched-stat" title="99th-percentile inter-token latency."><div class="num">${p99}<span style="font-size:0.55em;color:var(--text-muted);font-weight:500;"> ms</span></div><div class="lbl">p99 ITL</div></div>
        <div class="sched-stat" title="99.9th-percentile inter-token latency."><div class="num">${p999}<span style="font-size:0.55em;color:var(--text-muted);font-weight:500;"> ms</span></div><div class="lbl">p99.9 ITL</div></div>
        <div class="sched-stat" title="Request-local inter-token gaps counted in this rolling window."><div class="num">${data.sample_count}</div><div class="lbl">token gaps · ${window}s</div></div>
      </div>
      <div style="margin-top: var(--space-3); font-size: var(--text-xs); color: var(--text-muted);">Mean <span class="tabular-nums" style="color: var(--text-2);">${mean} ms</span> · max <span class="tabular-nums" style="color: var(--text-2);">${data.max_itl_ms.toFixed(1)} ms</span> · stalls <strong class="tabular-nums">${stalls}</strong> · unexplained <strong class="tabular-nums">${unexplained}</strong></div>`;
    }
    if (key !== lastDecodeStatsKey || !host.firstChild) {
      lastDecodeStatsKey = key;
      host.innerHTML = html;
    }

    if (idle) {
      // The rolling window drained: a stale sparkline under "no streaming
      // completions" would contradict it. Drop it and start fresh next time.
      const spark = el.querySelector('.decode-spark-host');
      if (spark) spark.remove();
      tpsHistory.length = 0;
      lastTpsRendered = null;
    } else if (typeof refreshDecodeSparkline === 'function') {
      refreshDecodeSparkline();
    }
  } catch (e) {
    // The failure HTML destroys both sub-hosts, so every dedupe key that
    // could skip rebuilding them must be invalidated or the panel stays
    // stuck on this message after the server recovers.
    lastDecodeStatsKey = null;
    lastTpsRendered = null;
    el.innerHTML = apiFailureHtml('Decode performance', e, 'pollDecodePerf');
  } finally {
    setPanelBusy('decode-perf-panel', false);
  }
}

// --- Recent Requests ---
let recentRequestsCache = [];
// First /v1/stats/recent-requests poll landed? A #overview/requests/{id}
// deep link defers its modal open until then (pendingRequestDrillId), so a
// boot deep link doesn't flash "no record" before the ring ever loaded.
let recentRequestsLoaded = false;
let pendingRequestDrillId = null;
// The ids currently shown in Recent requests, in display order, AFTER the active
// agent/status/text filters. The inspect modal's prev/next steps through THIS —
// so under "Needs attention" you walk only the problem requests.
let lastRenderedRequestIds = [];
// Ids from the previous recent-requests paint — null until the first render
// completes so the initial load doesn't flash every row as "new".
let previousRequestIds = null;

// Scroll shadow: once the page scrolls under the sticky bars, give them a
// quiet drop shadow so content visibly passes *under* the chrome.
(function initScrollShadow() {
  const update = () => document.body.classList.toggle('is-scrolled', window.scrollY > 4);
  window.addEventListener('scroll', update, { passive: true });
  update();
})();

function fmtRelTime(unixMs) {
  if (!unixMs) return '-';
  const now = Date.now();
  const delta = Math.max(0, Math.round((now - unixMs) / 1000));
  if (delta < 1) return 'now';
  if (delta < 60) return delta + 's ago';
  if (delta < 3600) return Math.floor(delta / 60) + 'm ago';
  if (delta < 86400) return Math.floor(delta / 3600) + 'h ago';
  return Math.floor(delta / 86400) + 'd ago';
}

/// Like `fmtRelTime` but switches to an absolute date once the entry is
/// older than a week — relative ago times stop being meaningful after that.
function fmtSmartTime(unixMs) {
  if (!unixMs) return '—';
  const delta = (Date.now() - unixMs) / 1000;
  if (delta < 7 * 86400) return fmtRelTime(unixMs);
  try {
    const d = new Date(unixMs);
    // e.g. "May 14"
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch { return fmtRelTime(unixMs); }
}

// `escapeHtml` is defined further down (the canonical version also
// escapes single quotes); function-declaration hoisting makes the later
// one win regardless of where this comment lives. Keeping a stub here
// would be dead code that future readers might mistake for the source
// of truth, so we just rely on the hoisted version.

function safeClassToken(s) {
  return String(s == null ? '' : s).replace(/[^a-z0-9_-]/gi, '_');
}

function shortId(s) {
  return String(s == null ? '' : s).slice(0, 8);
}

// Skip an innerHTML rewrite when the content key hasn't changed — poll
// loops must not repaint identical DOM (a rewrite destroys hover state,
// in-progress text selection, click targets, and open <select> dropdowns
// mid-interaction). The key lives on the element node itself rather than
// in a module-level variable so "someone else replaced this panel" is
// self-invalidating: a recreated node carries no key, and every writer
// that paints INTO the same node (error/empty/list branches alike) must
// route through this helper with its own distinct key, so a failure →
// recovery transition always compares unequal — the #1547 lesson, where
// module-level dedupe keys survived the failure writer and froze panels
// on stale error HTML. Returns true when it wrote; callers that wire
// listeners after rendering must skip that wiring when the DOM was left
// untouched (the old nodes still hold their old listeners).
function setListHtml(el, key, html) {
  if (!el) return false;
  if (el._kilnListKey === key) return false;
  el.innerHTML = html;
  el._kilnListKey = key;
  return true;
}

function apiFailureHtml(action, e, retryFn) {
  const retryButton = retryFn ? `<div style="margin-top:var(--space-3);"><button type="button" class="btn btn-primary" data-retry="${escapeHtml(retryFn)}" aria-label="Retry ${escapeHtml(action)}">Retry ${escapeHtml(action)}</button></div>` : '';
  return `<div class="empty api-failure">
    <div style="font-weight:600;color:var(--text);margin-bottom:6px;">Dashboard is waiting for Kiln server APIs.</div>
    <div>${escapeHtml(action)} could not load yet. If this is a cold start, wait for <code>kiln serve</code> to finish model startup, then retry.</div>
    ${retryButton}
    <div style="margin-top:var(--space-2);">Need setup help? See the <a href="https://ericflo.github.io/kiln/quickstart.html" target="_blank" rel="noopener noreferrer">Quickstart</a> or <a href="https://ericflo.github.io/kiln/troubleshooting.html" target="_blank" rel="noopener noreferrer">Troubleshooting</a>.</div>
    <div style="margin-top:var(--space-2);font-family:var(--font-mono);font-size:var(--text-xs);">Details: ${escapeHtml(e.message)}</div>
  </div>`;
}

// The whole app lives inside an IIFE, so the failure panels' Retry buttons
// can't use inline onclick (the poll functions aren't globals — a click would
// throw ReferenceError). They dispatch through this delegated listener instead.
const RETRY_ACTIONS = {
  pollHealth,
  pollDecodePerf,
  pollRecentRequests,
  pollAdapters,
  pollTraining,
};
document.addEventListener('click', (ev) => {
  const btn = ev.target.closest('[data-retry]');
  if (!btn) return;
  const action = RETRY_ACTIONS[btn.dataset.retry];
  if (action) action();
});

let recentRequestsFilter = '';
let recentAgentFilter = 'all';
// 'all' | 'attention' — the "show me what went wrong" toggle. A pi user lands
// on the dashboard precisely when something looked off; this isolates the rows
// worth correcting (errored, truncated, or silently base-served) so they can be
// opened and dropped into the Corrections basket without scrolling.
let recentStatusFilter = 'all';

// A request "needs attention" if pi got a degraded result from it: an error, a
// truncated (max_tokens-clipped) completion, a request-local token stall, or a
// silent fallback to the base model while the server claims a non-base adapter is active. These are the
// rows worth turning into corrections — the same predicate drives the
// "Needs attention" filter chip and the per-row warning tint. Deliberate
// non-problems stay neutral: 'client_disconnect' is the user pressing Ctrl-C
// mid-stream, and base-served requests while NO adapter is active (explicit
// unload, or none ever loaded) are the server doing exactly what it was told.
function requestNeedsAttention(r) {
  if (!r) return false;
  const f = (r.finish_reason || '').toLowerCase();
  if (r.error || f === 'error') return true;
  if (f === 'length') return true;
  if (Number(r.latency?.stall_count || 0) > 0) return true;
  if (servedBaseSilently(r.adapter)) return true;
  return false;
}

// Map a raw User-Agent to a friendly client identity for per-agent attribution.
// Best-effort: known agents/SDKs get a clean label; anything else shows its
// leading token. The raw UA is always available on hover.
function clientFromUA(ua) {
  if (!ua) return { key: 'unknown', label: 'unknown' };
  const s = ua.toLowerCase();
  if (/(^|[\/ _-])pi([\/ _-]|$)|pi-agent|earendil/.test(s)) return { key: 'pi', label: 'pi' };
  if (s.includes('opencode')) return { key: 'opencode', label: 'opencode' };
  // A bare Vercel AI SDK UA tells us the SDK, not the agent — don't credit
  // it to any particular client.
  if (s.includes('ai-sdk')) return { key: 'ai-sdk', label: 'AI SDK' };
  if (s.includes('openai-python') || s.includes('python-openai')) return { key: 'openai-python', label: 'OpenAI Python' };
  if (s.includes('openai') && (s.includes('node') || s.includes('js') || s.includes('deno') || s.includes('bun'))) return { key: 'openai-js', label: 'OpenAI JS' };
  if (s.includes('openai')) return { key: 'openai', label: 'OpenAI SDK' };
  if (s.includes('curl')) return { key: 'curl', label: 'curl' };
  if (s.includes('httpx') || s.includes('python-requests') || s.includes('aiohttp') || s.startsWith('python')) return { key: 'python', label: 'Python' };
  if (s.includes('undici') || s.includes('node') || s.includes('bun') || s.includes('deno')) return { key: 'node', label: 'Node' };
  return { key: 'other', label: (ua.split(/[\/ ]/)[0] || 'client').slice(0, 16) };
}

// The dashboard's own inference traffic (Test connection, Playground,
// Compare, judgment generation) self-identifies via the `X-Kiln-Client:
// dashboard` request header, which the server echoes back as `client` on
// each recent-requests row. Those rows are labeled honestly and must NEVER
// count as a connected agent — only external clients prove the
// "Agent connected" milestone.
function rowIsDashboard(r) { return !!r && r.client === 'dashboard'; }

// Resolve a row to its client identity: trusted self-identification first,
// User-Agent sniffing otherwise.
function clientForRow(r) {
  if (rowIsDashboard(r)) return { key: 'dashboard', label: 'dashboard' };
  return clientFromUA(r && r.user_agent);
}

function renderRecentRequests(rows) {
  const el = document.getElementById('recent-requests-panel');
  if (!el) return;
  if (!rows || rows.length === 0) {
    el.innerHTML = `<div class="empty">
      <div style="font-weight:600;color:var(--text);margin-bottom:6px;">No recent requests yet. Waiting for your first agent request.</div>
      <div style="color:var(--text-3);">Point <strong>pi</strong>, <strong>opencode</strong>, or any OpenAI client at <code>${escapeHtml(connectBaseUrl())}</code> (model <code>${escapeHtml(connectModelId)}</code>) and every chat completion shows up here — prompt &amp; completion previews, tokens, latency, and finish status. That live traffic is the raw material your model learns from.</div>
      <div style="margin-top:var(--space-3);"><button type="button" class="btn btn-primary btn-sm" onclick="openConnect()">${icon('link','icn-sm')} Connect your agent</button> <button type="button" class="btn btn-sm" onclick="selectPage('playground')">${icon('chat','icn-sm')} Or try the Playground</button></div>
      <div style="margin-top:var(--space-2);color:var(--text-3);">New to Kiln? Follow the <a href="https://ericflo.github.io/kiln/quickstart.html" target="_blank" rel="noopener noreferrer">Quickstart</a>.</div>
    </div>`;
    return;
  }
  // Tag every row with its calling agent so we can both badge it and offer
  // per-client filter chips — the operator confirms "is opencode getting
  // through?" in one click.
  rows.forEach(r => { r._client = clientForRow(r); });

  // Build the agent-filter chips from the clients actually present.
  // pi is the first-class agent integration — it always leads the
  // enumeration; everything else orders by traffic volume.
  const counts = new Map();
  rows.forEach(r => counts.set(r._client.key, (counts.get(r._client.key) || 0) + 1));
  const ordered = [...counts.entries()].sort((a, b) =>
    a[0] === 'pi' ? -1 : b[0] === 'pi' ? 1 : b[1] - a[1]);
  const labelFor = k => (rows.find(r => r._client.key === k)?._client.label) || k;
  if (!counts.has(recentAgentFilter) && recentAgentFilter !== 'all') recentAgentFilter = 'all';
  const chip = (key, label, n) =>
    `<button type="button" class="agent-chip${recentAgentFilter === key ? ' active' : ''}" data-agent-chip="${escapeHtml(key)}">${key === 'all' ? '' : icon('terminal')}${escapeHtml(label)}<span class="count">${n}</span></button>`;

  // "Needs attention" toggle — only shown when there's actually something wrong.
  // It sits with the agent chips but reads as a distinct, warning-tinted pill.
  const attentionCount = rows.filter(requestNeedsAttention).length;
  if (recentStatusFilter === 'attention' && attentionCount === 0) recentStatusFilter = 'all';
  const attentionChip = attentionCount > 0
    ? `<button type="button" class="agent-chip attn-chip${recentStatusFilter === 'attention' ? ' active' : ''}" data-status-chip="attention" title="Errored, truncated, or silently served by the base model while an adapter is active — the requests worth correcting">${icon('warning')}Needs attention<span class="count">${attentionCount}</span></button>`
    : '';
  const chipsInner = (ordered.length > 1 || attentionChip)
    ? chip('all', 'All agents', rows.length) + ordered.map(([k, n]) => chip(k, labelFor(k), n)).join('') + attentionChip
    : '';
  const chipsHtml = chipsInner
    ? `<div class="agent-chips" role="group" aria-label="Filter requests">${chipsInner}</div>`
    : '';

  // Apply text filter (prompt/completion/id) + agent filter + status filter together.
  const q = (recentRequestsFilter || '').trim().toLowerCase();
  const filtered = rows.filter(r =>
    (recentAgentFilter === 'all' || r._client.key === recentAgentFilter) &&
    (recentStatusFilter !== 'attention' || requestNeedsAttention(r)) &&
    (!q || (r.prompt_preview || '').toLowerCase().includes(q)
        || (r.completion_preview || '').toLowerCase().includes(q)
        || (r.id || '').toLowerCase().includes(q)));
  // Record display order so the inspect modal can step prev/next within the
  // current filter (e.g. walk only the "Needs attention" rows).
  lastRenderedRequestIds = filtered.map(r => r.id).filter(Boolean);
  // Snapshot this paint's ids so the NEXT poll can tell which rows just
  // arrived (drives the .row-new arrival flash). Kept separate from
  // lastRenderedRequestIds, which the inspect modal mutates for navigation.
  previousRequestIds = new Set(lastRenderedRequestIds);
  if (filtered.length === 0) {
    el.innerHTML = chipsHtml + `<div class="empty">No requests match this filter.</div>`;
    bindAgentChips(el);
    return;
  }
  const items = filtered.map(r => {
    const finish = r.finish_reason || 'stop';
    const finishClass = finish.replace(/[^a-z_]/gi, '_');
    const streamPill = r.streamed ? '<span class="recent-pill streamed">stream</span>' : '';
    const finishPill = `<span class="recent-pill ${finishClass}">${escapeHtml(finish)}</span>`;
    const agentPill = `<span class="recent-agent" title="${escapeHtml(r.user_agent || 'unknown client')}">${escapeHtml(r._client.label)}</span>`;
    // Which adapter actually served this request — so a silent fallback to the
    // base model (your trained LoRA quietly off) is visible per request, not
    // buried in the inspect modal. Base-while-an-adapter-is-ACTIVE gets a
    // warning tint; base while nothing is active is the configured behaviour.
    const adapterName = r.adapter || 'base';
    const baseFallback = servedBaseSilently(adapterName);
    const adapterPill = `<span class="recent-served${baseFallback ? ' is-base' : ''}" title="${baseFallback ? 'Served by the BASE model — your trained adapters are not being used for this request' : 'Served by ' + escapeHtml(adapterName)}">${baseFallback ? icon('warning', 'icn-sm') : ''}${escapeHtml(adapterName)}</span>`;
    const tokens = `${r.prompt_tokens || 0} → ${r.completion_tokens || 0} tok`;
    const dur = (r.duration_ms != null) ? `${r.duration_ms} ms` : '';
    const ttft = (r.ttft_ms != null) ? `<span class="recent-ttft tabular-nums" title="Time to first token — the latency pi feels before output starts">${fmtMsShort(r.ttft_ms)} ttft</span>` : '';
    const stalls = Number(r.latency?.stall_count || 0);
    const stallPill = stalls > 0
      ? `<span class="recent-pill length" title="${Number(r.latency?.unexplained_stall_count || 0)} unexplained">${stalls} stall${stalls === 1 ? '' : 's'}</span>`
      : '';
    const promptText = r.prompt_preview || '—';
    const completionText = r.completion_preview || '—';
    const attn = requestNeedsAttention(r);
    // Newly-arrived rows flash a brief ember tint so live traffic feels alive.
    // lastRenderedRequestIds holds the previous paint's ids (undefined on the
    // very first render — nothing is "new" then).
    const isNew = previousRequestIds !== null && r.id && !previousRequestIds.has(r.id);
    return `
      <li class="recent-row${attn ? ' attn' : ''}${isNew ? ' row-new' : ''}" data-ts="${r.timestamp_unix_ms || 0}" data-id="${escapeHtml(r.id || '')}" tabindex="0" role="button" aria-label="Inspect request ${escapeHtml(shortId(r.id || ''))} from ${escapeHtml(r._client.label)}${attn ? ' — needs attention' : ''}">
        <div class="recent-time">${fmtRelTime(r.timestamp_unix_ms)}</div>
        <div class="recent-previews">
          <div class="recent-prompt" title="${escapeHtml(promptText)}">${agentPill}${adapterPill}${streamPill}${escapeHtml(promptText)}</div>
          <div class="recent-completion" title="${escapeHtml(completionText)}">${escapeHtml(completionText)}</div>
        </div>
        <div class="recent-meta">
          <span class="recent-tokens">${tokens}${ttft}</span>
          <span>${finishPill}${stallPill}${dur ? `<span class="tabular-nums">${escapeHtml(dur)}</span>` : ''}</span>
          <button type="button" class="recent-correct${attn ? '' : ' quiet'}" data-correct-id="${escapeHtml(r.id || '')}" title="Add this to your Corrections basket — write the fix on the Overview, then train">${icon('flask', 'icn-sm')} Correct</button>
        </div>
      </li>
    `;
  }).join('');
  el.innerHTML = chipsHtml + `<ul class="recent-list">${items}</ul>`;
  bindAgentChips(el);
  el.querySelectorAll('.recent-row').forEach(row => {
    const id = row.dataset.id;
    if (!id) return;
    row.addEventListener('click', () => openRequestDrillModal(id));
    row.addEventListener('keydown', (ev) => {
      // Only the row itself opens the modal — let the inline Correct button
      // (a focusable child) handle its own Enter/Space.
      if ((ev.key === 'Enter' || ev.key === ' ') && ev.target === row) {
        ev.preventDefault();
        openRequestDrillModal(id);
      }
    });
  });
  // Inline one-click capture for the rows worth correcting — sweep the whole
  // "Needs attention" column into the basket without opening each modal.
  el.querySelectorAll('[data-correct-id]').forEach(btn => {
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const r = findRecentRequest(btn.dataset.correctId);
      if (!r) return;
      const added = window.addCorrectionFromRequest(r);
      if (added) {
        const orig = btn.innerHTML;
        btn.classList.add('is-added'); btn.disabled = true;
        btn.innerHTML = `${icon('check', 'icn-sm')} Added`;
        setTimeout(() => { btn.innerHTML = orig; btn.classList.remove('is-added'); btn.disabled = false; }, 1100);
      }
    });
  });
}

function bindAgentChips(el) {
  el.querySelectorAll('[data-agent-chip]').forEach(c => c.addEventListener('click', () => {
    recentAgentFilter = c.dataset.agentChip;
    lastRecentRequestsKey = null; // force re-render
    renderRecentRequests(recentRequestsCache);
  }));
  el.querySelectorAll('[data-status-chip]').forEach(c => c.addEventListener('click', () => {
    // Toggle the attention filter off when re-clicking the active chip.
    recentStatusFilter = (recentStatusFilter === 'attention') ? 'all' : 'attention';
    lastRecentRequestsKey = null; // force re-render
    renderRecentRequests(recentRequestsCache);
  }));
}

/* ---------------------------------------------------------------------
   Recent-request inspect modal

   The recent-requests panel only shows previews; clicking a row pops
   this modal which exposes everything we captured on the server side
   (model, adapter, finish reason, tokens, timings, full prompt + full
   completion). Replay button drops the prompt into the playground.
   --------------------------------------------------------------------- */
function findRecentRequest(id) {
  return (recentRequestsCache || []).find(r => r.id === id) || null;
}

function formatUnixMs(ms) {
  if (!ms) return '—';
  try {
    const d = new Date(ms);
    return d.toISOString().replace('T', ' ').replace('Z', ' UTC');
  } catch { return String(ms); }
}

function requestThinkingBudgetSource(source) {
  return ({
    request: 'request',
    server_default: 'server default',
    request_unlimited: 'request unlimited',
    unlimited: 'unlimited',
  })[source] || 'unknown';
}

function requestThinkingBudgetOutcome(r) {
  const budget = r?.thinking_budget;
  if (!budget || !budget.configured) return 'Not configured';
  if (budget.applied == null) return 'Unresolved';
  if (!budget.applied) return 'Inert';
  if (budget.triggered) {
    const trigger = ({
      tokens: 'Token limit',
      time: 'Time limit',
      max_tokens: 'Completion limit',
    })[budget.trigger] || 'Budget limit';
    return `${trigger} · ${budget.closed ? 'closed' : 'close incomplete'}`;
  }
  if (budget.triggered === false && budget.closed === true) return 'Natural close';
  if (['error', 'timeout', 'client_disconnect'].includes(r.finish_reason)) return 'Interrupted';
  if (budget.closed === false) return 'Unclosed';
  return 'Unresolved';
}

function requestThinkingBudgetSection(r) {
  const budget = r?.thinking_budget;
  if (!budget || typeof budget !== 'object') return '';
  const tokenLimit = budget.max_tokens == null ? 'Unlimited' : `${budget.max_tokens} tokens`;
  const timeLimit = budget.max_time_ms == null ? 'Unlimited' : fmtMsShort(budget.max_time_ms);
  const applied = budget.applied == null ? 'Unresolved' : budget.applied ? 'Yes' : 'No';
  const measured = [];
  if (budget.thinking_tokens != null) measured.push(`${budget.thinking_tokens} tokens`);
  if (budget.thinking_time_ms != null) measured.push(fmtMsShort(budget.thinking_time_ms));
  const rows = [
    ['Token limit', `${tokenLimit} · ${requestThinkingBudgetSource(budget.tokens_source)}`],
    ['Time limit', `${timeLimit} · ${requestThinkingBudgetSource(budget.time_source)}`],
    ['Applied', applied],
    ['Outcome', requestThinkingBudgetOutcome(r)],
  ];
  if (measured.length) rows.push(['Measured thinking', measured.join(' · ')]);
  return `
    <div class="req-section req-thinking-budget" data-request-thinking-budget>
      <div class="req-section-head">Thinking budget</div>
      <div class="req-stats">
        ${rows.map(([key, value]) => `<div class="req-stat"><span class="req-stat-k">${escapeHtml(key)}</span><span class="req-stat-v">${escapeHtml(value)}</span></div>`).join('')}
      </div>
    </div>`;
}

function requestLatencySection(r) {
  const latency = r?.latency;
  if (!latency || typeof latency !== 'object') return '';
  const value = milliseconds => milliseconds == null ? '—' : fmtMsShort(milliseconds);
  const summary = [
    ['TTFT', value(latency.ttft_ms)],
    ['p50 ITL', value(latency.itl_ms_p50)],
    ['p99 ITL', value(latency.itl_ms_p99)],
    ['p99.9 ITL', value(latency.itl_ms_p999)],
    ['Max ITL', value(latency.max_itl_ms)],
    ['Stalls', String(latency.stall_count || 0)],
    ['Unexplained', String(latency.unexplained_stall_count || 0)],
  ];
  const phaseLabels = {
    actor_queue_ms: 'Actor queue', actor_admission_ms: 'Admission', tokenization_ms: 'Tokenization',
    prefill_ms: 'Actor prefill', decode_ms: 'Actor decode',
    actor_cycle_idle_ms: 'Actor cycle idle', sampling_ms: 'Sampling',
    readback_ms: 'Readback', response_delivery_ms: 'Response delivery',
    handler_queue_ms: 'Handler queue', client_delivery_ms: 'Body enqueue',
    gpu_lock_wait_ms: 'GPU lock wait', graph_capture_ms: 'Graph capture',
    graph_replay_ms: 'Graph replay', synchronization_ms: 'Synchronization', resize_ms: 'Resize',
    trim_ms: 'Trim', adapter_ms: 'Adapter', training_ms: 'Training', unexplained_ms: 'Unexplained',
  };
  const phases = latency.phases && typeof latency.phases === 'object' ? latency.phases : {};
  const measured = Object.entries(phaseLabels)
    .filter(([key]) => typeof phases[key] === 'number')
    .map(([key, label]) => `<div class="req-stat"><span class="req-stat-k">${escapeHtml(label)}</span><span class="req-stat-v tabular-nums">${escapeHtml(value(phases[key]))}</span></div>`)
    .join('');
  const missing = Object.entries(phaseLabels).filter(([key]) => phases[key] == null).map(([, label]) => label);
  const reasonLabels = {
    actor_queue: 'actor queue', actor_admission: 'admission', actor_prefill: 'prefill',
    actor_decode: 'decode', actor_cycle_idle: 'actor cycle idle',
    response_delivery: 'response delivery', handler_queue: 'handler queue',
    client_delivery: 'body enqueue', sampling: 'sampling', readback: 'readback',
    gpu_lock_wait: 'GPU lock wait', graph_capture: 'graph capture', graph_replay: 'graph replay',
    synchronization: 'synchronization', resize: 'resize', trim: 'trim', adapter: 'adapter',
    training: 'training', unexplained: 'unexplained',
  };
  const reasons = Object.entries(reasonLabels)
    .filter(([key]) => Number(latency.stall_reasons?.[key] || 0) > 0)
    .map(([key, label]) => `${label} ${latency.stall_reasons[key]}`)
    .join(' · ');
  const coverage = `${latency.retained_gap_samples || 0} of ${latency.gap_samples || 0} request-local gaps retained${latency.gap_samples_truncated ? ' (bounded)' : ''}`;
  return `
    <div class="req-section" data-request-latency>
      <div class="req-section-head">Latency diagnosis</div>
      <div class="req-stats">
        ${summary.map(([key, display]) => `<div class="req-stat"><span class="req-stat-k">${escapeHtml(key)}</span><span class="req-stat-v tabular-nums">${escapeHtml(display)}</span></div>`).join('')}
      </div>
      ${measured ? `<div class="req-stats" style="margin-top:var(--space-3);">${measured}</div>` : ''}
      <div class="hint" style="margin-top:var(--space-2);">${escapeHtml(coverage)}${reasons ? ` · ${escapeHtml(reasons)}` : ''}</div>
      ${missing.length ? `<div class="hint" style="margin-top:var(--space-1);">Not measured on this path: ${escapeHtml(missing.join(', '))}</div>` : ''}
    </div>`;
}

function openRequestDrillModal(id) {
  const modal = document.getElementById('request-drill-modal');
  if (!modal) return;
  // Prev/next triage inside the open modal replaces the id segment instead
  // of minting a history entry per arrow press.
  modalHashOnOpen('request', '#overview/requests/' + encodeURIComponent(id), !modal.hidden);
  const r = findRecentRequest(id);
  const titleEl = document.getElementById('request-drill-title');
  const metaEl = document.getElementById('request-drill-meta');
  const content = document.getElementById('request-drill-content');
  if (!r) {
    titleEl.textContent = 'Request';
    metaEl.textContent = id;
    content.innerHTML = '<div class="detail-empty">No record for that request id (it may have been evicted from the in-memory ring).</div>';
    modal.hidden = false;
    modal.dataset.requestId = id;
    openModal(modal, { onClose: userCloseRequestDrillModal });
    updateRequestDrillNav(id);
    return;
  }
  // Strip the API-style `chatcmpl-` prefix so the shortened id is the
  // distinguishing tail (the first 8 chars of a UUID), not the prefix
  // every id shares.
  const trimmedId = String(r.id || '').replace(/^chatcmpl-/, '');
  titleEl.textContent = `Request ${trimmedId.slice(0, 8) || 'unknown'}`;
  const pieces = [];
  pieces.push(escapeHtml(r.model || '—'));
  if (r.user_agent || rowIsDashboard(r)) pieces.push(`via <strong>${escapeHtml(clientForRow(r).label)}</strong>`);
  if (r.adapter) pieces.push(`adapter <code>${escapeHtml(r.adapter)}</code>`);
  pieces.push(r.streamed ? 'streamed' : 'unary');
  metaEl.innerHTML = pieces.join(' · ');

  // Two different tok/s figures appear in this modal and they legitimately
  // disagree: this one divides by the WHOLE request duration (prefill +
  // queueing + TTFT + streaming), while the latency bar below divides by
  // streaming time only. Both are labeled explicitly so they never read as
  // a contradiction.
  const stats = [
    ['Started',         formatUnixMs(r.timestamp_unix_ms)],
    ['Finish reason',   r.finish_reason || '—'],
    ['Duration',        r.duration_ms != null ? `${r.duration_ms} ms` : '—'],
    ['Prompt tokens',   r.prompt_tokens != null ? String(r.prompt_tokens) : '—'],
    ['Completion tok',  r.completion_tokens != null ? String(r.completion_tokens) : '—'],
    ['tok/s (end-to-end)', (r.duration_ms && r.completion_tokens) ? (r.completion_tokens / (r.duration_ms / 1000)).toFixed(1) : '—',
      'Completion tokens ÷ total request duration. Includes prefill, queueing, and the wait for the first token, so it reads lower than the decode-only rate in the latency bar.'],
  ];
  if (r.ttft_ms != null) stats.push(['TTFT', `${r.ttft_ms} ms`]);
  if (r.temperature != null) stats.push(['Temperature', String(r.temperature)]);
  if (r.top_p != null) stats.push(['top_p', String(r.top_p)]);
  if (r.max_tokens != null) stats.push(['max_tokens', String(r.max_tokens)]);
  const statRow = stats
    .map(([k, v, title]) => `<div class="req-stat"${title ? ` title="${escapeHtml(title)}"` : ''}><span class="req-stat-k">${escapeHtml(k)}</span><span class="req-stat-v">${escapeHtml(v)}</span></div>`)
    .join('');

  const prompt = r.prompt_full || r.prompt_preview || '';
  const completion = r.completion_full || r.completion_preview || '';
  const errBlock = r.error
    ? `<div class="req-section req-error"><div class="req-section-head">Error</div><pre class="req-pre">${escapeHtml(r.error)}</pre></div>`
    : '';
  const thinkingBudgetHtml = requestThinkingBudgetSection(r);
  const latencyDiagnosisHtml = requestLatencySection(r);
  // Latency breakdown — the experience pi actually felt: the wait for the first
  // token (TTFT), then how fast the rest streamed. Only meaningful when we have
  // both a TTFT and a total duration (i.e. a streamed completion).
  let latencyHtml = '';
  if (!r.latency && r.ttft_ms != null && r.duration_ms != null && r.duration_ms >= r.ttft_ms) {
    const total = r.duration_ms, ttft = r.ttft_ms, decode = Math.max(0, total - ttft);
    const ttftPct = total > 0 ? (ttft / total * 100) : 0;
    const tps = (r.completion_tokens && decode > 0) ? (r.completion_tokens / (decode / 1000)).toFixed(0) : null;
    latencyHtml = `
      <div class="req-section">
        <div class="req-section-head">Latency pi felt</div>
        <div class="lat-bar" role="img" aria-label="Time to first token ${fmtMsShort(ttft)}, then ${fmtMsShort(decode)} streaming ${r.completion_tokens || 0} tokens, ${fmtMsShort(total)} total">
          <div class="lat-seg lat-ttft" style="width:${ttftPct.toFixed(1)}%" title="Time to first token (${fmtMsShort(ttft)}) — the wait pi feels before any output"></div>
          <div class="lat-seg lat-decode" style="width:${(100 - ttftPct).toFixed(1)}%" title="Streaming the response (${fmtMsShort(decode)}${tps ? ', ' + tps + ' tok/s decode-only' : ''})"></div>
        </div>
        <div class="lat-legend">
          <span><span class="lat-key lat-ttft"></span>first token in <strong>${fmtMsShort(ttft)}</strong></span>
          <span><span class="lat-key lat-decode"></span>${r.completion_tokens || 0} tokens in <strong>${fmtMsShort(decode)}</strong>${tps ? ` · <span title="Completion tokens ÷ streaming time after the first token (inter-token rate). Excludes prefill and queueing, so it reads higher than the end-to-end tok/s above.">${tps} tok/s (decode)</span>` : ''}</span>
          <span class="lat-total">total <strong>${fmtMsShort(total)}</strong></span>
        </div>
      </div>`;
  }
  content.innerHTML = `
    <div class="req-detail">
      <div class="req-stats">${statRow}</div>
      ${latencyDiagnosisHtml}
      ${latencyHtml}
      ${thinkingBudgetHtml}
      ${errBlock}
      <div class="req-section">
        <div class="req-section-head">Prompt
          <button class="btn btn-sm" type="button" data-copy="prompt">Copy</button>
        </div>
        <pre class="req-pre" data-pre="prompt">${escapeHtml(prompt)}</pre>
      </div>
      <div class="req-section">
        <div class="req-section-head">Completion
          <button class="btn btn-sm" type="button" data-copy="completion">Copy</button>
        </div>
        <pre class="req-pre" data-pre="completion">${escapeHtml(completion)}</pre>
      </div>
      <div class="req-section req-meta">
        <div class="req-section-head">Identity</div>
        <div class="req-stat"><span class="req-stat-k">Request id</span><span class="req-stat-v"><code>${escapeHtml(r.id || '')}</code></span></div>
        ${r.user_agent ? `<div class="req-stat"><span class="req-stat-k">Client</span><span class="req-stat-v"><code>${escapeHtml(r.user_agent)}</code></span></div>` : ''}
      </div>
    </div>`;
  content.querySelectorAll('button[data-copy]').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.copy;
      const pre = content.querySelector(`pre[data-pre="${key}"]`);
      const text = pre ? pre.textContent : '';
      navigator.clipboard.writeText(text).then(() => toast(`Copied ${key}`, 'ok'), () => toast('Copy failed', 'err'));
    });
  });
  modal.hidden = false;
  modal.dataset.requestId = id;
  openModal(modal, { onClose: userCloseRequestDrillModal });
  updateRequestDrillNav(id);
}

// Prev/next walk the CURRENT filtered Recent-requests order, so triaging under
// "Needs attention" steps through only the problem requests. Hidden when there's
// nothing to step through (single result or the row isn't in the current view).
function updateRequestDrillNav(id) {
  const nav = document.getElementById('request-drill-nav');
  const pos = document.getElementById('request-drill-pos');
  const prev = document.getElementById('request-drill-prev');
  const next = document.getElementById('request-drill-next');
  const ids = lastRenderedRequestIds || [];
  const idx = ids.indexOf(id);
  const has = idx >= 0 && ids.length > 1;
  if (nav) nav.style.display = has ? '' : 'none';
  if (!has) return;
  if (pos) pos.textContent = `${idx + 1} / ${ids.length}`;
  if (prev) prev.disabled = idx <= 0;
  if (next) next.disabled = idx >= ids.length - 1;
}
function navigateRequestDrill(dir) {
  const modal = document.getElementById('request-drill-modal');
  if (!modal || modal.hidden) return;
  const ids = lastRenderedRequestIds || [];
  const idx = ids.indexOf(modal.dataset.requestId || '');
  const ni = idx + dir;
  if (idx < 0 || ni < 0 || ni >= ids.length) return;
  openRequestDrillModal(ids[ni]);
}

function closeRequestDrillModal() {
  const modal = document.getElementById('request-drill-modal');
  if (!modal) return;
  modal.hidden = true;
  delete modal.dataset.requestId;
  closeModal(modal);
}
// User-initiated close (X / backdrop / Esc): walk history per the deep-link
// state machine. Flows that close on the way SOMEWHERE ELSE (Replay,
// Verify A/B) keep calling closeRequestDrillModal directly so Back returns
// to the modal they left.
function userCloseRequestDrillModal() {
  modalHashOnUserClose('request', '#overview', closeRequestDrillModal);
}

// Load the Playground in compare mode with a prompt and a before/after adapter
// pair, ready to Send. Used by the drill modal's "Verify A/B" to prove a fix.
// Tolerant of adapters not present in the selectors (falls back to base).
function setupCompareReplay(prompt, adapterA, adapterB) {
  const input = document.getElementById('chat-input');
  if (input) {
    input.value = prompt || '';
    if (typeof autoresizeChatInput === 'function') autoresizeChatInput();
    if (typeof updateChatSendState === 'function') updateChatSendState();
  }
  // Turn compare mode on via the toggle so its change-handler wires up the B
  // selector + the side-by-side panel exactly as a manual toggle would.
  const toggle = document.getElementById('chat-compare-toggle');
  if (toggle && !toggle.checked) { toggle.checked = true; toggle.dispatchEvent(new Event('change', { bubbles: true })); }
  const setSel = (id, val) => {
    const sel = document.getElementById(id);
    if (!sel) return false;
    if (Array.from(sel.options).some(o => o.value === val)) { sel.value = val; return true; }
    // The adapter may still be training — remember the intent and apply it
    // automatically when the option shows up (see updateAdapterSelect).
    sel.value = '';
    if (val) sel.dataset.pendingValue = val;
    return !val;
  };
  setSel('chat-adapter', adapterA || '');
  const bApplied = setSel('chat-adapter-b', adapterB || '');
  const aName = adapterA || 'base', bName = adapterB || 'base';
  if (!bApplied && adapterB) {
    toast(`Prompt loaded — ${adapterB} is still training; it'll be selected here automatically when it's ready`, 'info');
  } else if (aName === bName) {
    toast(`Loaded for A/B — pick a second adapter to compare against ${aName}, then Send`, 'info');
  } else {
    toast(`Loaded A/B: ${aName} vs ${bName} — press Send to see if the swap helped`, 'ok');
  }
  if (input) setTimeout(() => input.focus(), 60);
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('request-drill-close')?.addEventListener('click', userCloseRequestDrillModal);
  document.getElementById('request-drill-modal')?.addEventListener('click', (ev) => {
    if (ev.target.id === 'request-drill-modal') userCloseRequestDrillModal();
  });
  document.getElementById('request-drill-prev')?.addEventListener('click', () => navigateRequestDrill(-1));
  document.getElementById('request-drill-next')?.addEventListener('click', () => navigateRequestDrill(1));
  document.getElementById('request-drill-copy-id')?.addEventListener('click', () => {
    const modal = document.getElementById('request-drill-modal');
    const id = modal?.dataset.requestId || '';
    if (!id) return;
    navigator.clipboard.writeText(id).then(() => toast('Request id copied', 'ok'), () => toast('Copy failed', 'err'));
  });
  document.getElementById('request-drill-raw')?.addEventListener('click', () => {
    const modal = document.getElementById('request-drill-modal');
    const id = modal?.dataset.requestId || '';
    const r = findRecentRequest(id);
    if (!r) return;
    const content = document.getElementById('request-drill-content');
    if (!content) return;
    const existing = content.querySelector('#request-raw-block');
    if (existing) { existing.remove(); return; }
    const pre = document.createElement('pre');
    pre.id = 'request-raw-block';
    pre.className = 'req-pre';
    pre.style.cssText = 'max-height:50vh; margin:var(--space-4) var(--space-5);';
    pre.textContent = JSON.stringify(r, null, 2);
    content.appendChild(pre);
    pre.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
  document.getElementById('request-drill-replay')?.addEventListener('click', () => {
    const modal = document.getElementById('request-drill-modal');
    const id = modal?.dataset.requestId || '';
    const r = findRecentRequest(id);
    if (!r) return;
    const input = document.getElementById('chat-input');
    if (input) {
      input.value = r.prompt_full || r.prompt_preview || '';
      if (typeof autoresizeChatInput === 'function') autoresizeChatInput();
      if (typeof updateChatSendState === 'function') updateChatSendState();
    }
    if (r.adapter) {
      const sel = document.getElementById('chat-adapter');
      if (sel && Array.from(sel.options).some(o => o.value === r.adapter)) sel.value = r.adapter;
    }
    closeRequestDrillModal();
    const playgroundTab = document.querySelector('[data-page="playground"]');
    if (playgroundTab) playgroundTab.click();
    if (input) setTimeout(() => input.focus(), 50);
  });
  // Verify A/B — close the flywheel loop. Re-run this exact prompt in the
  // Playground's compare mode, side A = whatever served it (the "before"),
  // side B = the currently active adapter (the "after" you hot-swapped in).
  // This is the "did the swap actually help?" proof the loop was missing.
  document.getElementById('request-drill-verify')?.addEventListener('click', () => {
    const modal = document.getElementById('request-drill-modal');
    const r = findRecentRequest(modal?.dataset.requestId || '');
    if (!r) return;
    const prompt = r.prompt_full || r.prompt_preview || '';
    const before = r.adapter || '';                              // '' = base
    const after = (lastHealth && lastHealth.active_adapter) || ''; // current hot-swapped adapter
    closeRequestDrillModal();
    selectPage('playground');
    setTimeout(() => setupCompareReplay(prompt, before, after), 60);
  });
  // Use-as-correction — the literal flywheel hinge. Append this captured
  // request to the Corrections basket on the Overview (it persists across
  // reloads and accumulates over a coding session). The modal stays open so
  // the operator can flick through Recent requests and grab several bad
  // answers in a row; a brief "Added ✓" pulse confirms each capture.
  document.getElementById('request-drill-correct')?.addEventListener('click', () => {
    const modal = document.getElementById('request-drill-modal');
    const id = modal?.dataset.requestId || '';
    const r = findRecentRequest(id);
    if (!r) return;
    const btn = document.getElementById('request-drill-correct');
    const added = addCorrectionFromRequest(r);
    if (btn && added) {
      const orig = btn.innerHTML;
      btn.classList.add('is-added');
      btn.innerHTML = `${icon('check', 'icn-sm')} Added`;
      btn.disabled = true;
      setTimeout(() => { btn.innerHTML = orig; btn.classList.remove('is-added'); btn.disabled = false; }, 1100);
    }
  });
  // Escape is handled by the shared modal manager (routes through
  // userCloseRequestDrillModal via the layer's onClose).
  document.addEventListener('keydown', (ev) => {
    const m = document.getElementById('request-drill-modal');
    if (!m || m.hidden) return;
    // Only while this drill is the TOP modal — cmdk over it owns the keys.
    if (modalStackTop()?.el !== m) return;
    // Arrow-key triage through the filtered list — but not while typing in the
    // "Use as correction" editor or any field.
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test((ev.target.tagName || '')) || ev.target.isContentEditable;
    if (typing) return;
    if (ev.key === 'ArrowLeft') { ev.preventDefault(); navigateRequestDrill(-1); }
    else if (ev.key === 'ArrowRight') { ev.preventDefault(); navigateRequestDrill(1); }
  });
});

function refreshRecentTimes() {
  const el = document.getElementById('recent-requests-panel');
  if (!el) return;
  // No aria-live suppression dance needed: the panel is deliberately not a
  // live region (see index.html), so in-place timestamp ticks are silent.
  el.querySelectorAll('.recent-row').forEach(row => {
    const ts = Number(row.dataset.ts || 0);
    if (!ts) return;
    const tcell = row.querySelector('.recent-time');
    if (tcell) tcell.textContent = fmtRelTime(ts);
  });
}

// Wire the filter input. Re-render on every input event using the
// already-cached list (no extra fetch). Done outside DOMContentLoaded
// so it runs as soon as the input exists.
document.addEventListener('input', (ev) => {
  if (ev.target && ev.target.id === 'recent-requests-filter') {
    recentRequestsFilter = ev.target.value || '';
    // Skip the change-detection guard for the next render so the filter
    // applies immediately even when the upstream cache is unchanged.
    lastRecentRequestsKey = null;
    renderRecentRequests(recentRequestsCache);
  } else if (ev.target && ev.target.id === 'eval-jobs-filter') {
    evalJobsFilter.query = ev.target.value || '';
    if (typeof refreshEvalJobs === 'function') refreshEvalJobs();
  } else if (ev.target && ev.target.id === 'training-queue-filter') {
    trainingQueueFilter = ev.target.value || '';
    // Bypass the change-detection guard for an immediate re-render.
    lastTrainingKey = null;
    if (trainingJobsCache) renderTrainingQueue(trainingJobsCache);
  } else if (ev.target && ev.target.id === 'adapters-filter') {
    adaptersFilter = ev.target.value || '';
    if (typeof lastAdaptersKey !== 'undefined') lastAdaptersKey = null;
    if (typeof pollAdapters === 'function') pollAdapters();
  }
});

// State-pill clicks for the eval-jobs filter. Delegated so a re-render
// of the pill DOM doesn't tear off the handler.
document.addEventListener('click', (ev) => {
  const t = ev.target.closest('[data-eval-jobs-filter]');
  if (!t) return;
  // Don't double-handle the "Clear filter" button rendered inside the
  // empty state — it already re-runs refresh on its own listener.
  if (!t.classList.contains('filter-pill')) return;
  evalJobsFilter.state = t.dataset.evalJobsFilter;
  document.querySelectorAll('.filter-pill[data-eval-jobs-filter]').forEach(p => {
    p.classList.toggle('active', p.dataset.evalJobsFilter === evalJobsFilter.state);
  });
  if (typeof refreshEvalJobs === 'function') refreshEvalJobs();
});

// Content key from the most recent request fingerprints. Skipping the
// re-render when unchanged preserves text-selection / hover / scroll
// position on idle servers (`refreshRecentTimes` already updates the
// relative timestamps in place every second).
let lastRecentRequestsKey = null;

// One-line screen-reader summary for the Recent requests card, spoken ONLY
// when the needs-attention count changes — the moment the warning tint
// appears or clears for a sighted user. Routine traffic never narrates (the
// panel is not a live region), and "new" counts arrivals since the last
// announcement so the line carries scale without ticking every poll.
let lastAnnouncedAttentionCount = null;
let lastAnnouncedRequestIds = new Set();
function announceRecentAttention(rows) {
  const ids = new Set(rows.map(r => `${r.timestamp_unix_ms || ''}:${r.id || ''}`));
  const attn = rows.filter(requestNeedsAttention).length;
  if (lastAnnouncedAttentionCount === null) {
    // First poll is baseline — page load must not announce history.
    lastAnnouncedAttentionCount = attn;
    lastAnnouncedRequestIds = ids;
    return;
  }
  if (attn === lastAnnouncedAttentionCount) return;
  let fresh = 0;
  ids.forEach(id => { if (!lastAnnouncedRequestIds.has(id)) fresh++; });
  const newPart = fresh > 0 ? `${fresh} new request${fresh === 1 ? '' : 's'}, ` : '';
  const attnPart = attn === 0
    ? 'no requests need attention'
    : `${attn} request${attn === 1 ? '' : 's'} need${attn === 1 ? 's' : ''} attention`;
  announceStatus('recent-requests-status', `${newPart}${attnPart}.`);
  lastAnnouncedAttentionCount = attn;
  lastAnnouncedRequestIds = ids;
}

async function pollRecentRequests() {
  const el = setPanelBusy('recent-requests-panel', true);
  if (!el) return;
  try {
    const data = await api('/v1/stats/recent-requests');
    recentRequestsCache = Array.isArray(data) ? data : [];
    recentRequestsLoaded = true;
    // A #overview/requests/{id} deep link that arrived before the ring
    // loaded parked its id here — open it now that lookup is meaningful
    // (still hash-suppressed: the id is already in the URL).
    if (pendingRequestDrillId) {
      const wantId = pendingRequestDrillId;
      pendingRequestDrillId = null;
      withHashWritesSuppressed(() => openRequestDrillModal(wantId));
    }
    updateConnectSummary(recentRequestsCache);
    updateFlywheel();
    refreshRequestHealth();
    announceRecentAttention(recentRequestsCache);
    const key = recentRequestsCache
      .map(r => `${r.timestamp_unix_ms || r.id || ''}:${r.completion_tokens || 0}`)
      .join('|');
    if (key === lastRecentRequestsKey) return;
    lastRecentRequestsKey = key;
    renderRecentRequests(recentRequestsCache);
  } catch (e) {
    // Invalidate the dedupe key: the failure HTML replaced the list, so an
    // unchanged key after recovery would leave this panel stuck on the error.
    lastRecentRequestsKey = null;
    el.innerHTML = apiFailureHtml('Recent requests', e, 'pollRecentRequests');
  } finally {
    setPanelBusy('recent-requests-panel', false);
  }
}
