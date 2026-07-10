(function() {
'use strict';

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
    last.appendChild(counter);
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
  el.textContent = msg;
  el.title = 'Click to dismiss';
  el.style.cursor = 'pointer';
  el.addEventListener('click', () => el.remove());
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
  setText('corr-train-n', String(ready));
  const note = document.getElementById('corr-foot-note');
  if (note) note.textContent = todo > 0
    ? `${todo} still need${todo === 1 ? 's' : ''} an answer · only edited items train`
    : (ready > 0 ? 'These become one SFT job — the new adapter hot-swaps in when done' : '');
  const btn = document.getElementById('corr-train');
  if (btn) btn.disabled = ready === 0;
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
    const body = { dataset: 'corrections:active', config: { output_name: name, auto_load: true, epochs: 3, lora_rank: 8, lora_alpha: loraAlphaFor(8) } };
    const res = await api('/v1/train/sft', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    toast(res.message || `Training ${name} from ${trainable.length} correction${trainable.length === 1 ? '' : 's'} — it will hot-swap in when done`, 'ok');
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
  finally { if (btn) btn.disabled = correctionsBasket.filter(corrTrainable).length === 0; }
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
   Runtime config expander — GET /v1/config (detected VRAM + detection
   source, KV cache geometry, training checkpointing, memory budgets).
   The <details> shell is static in index.html as a SIBLING of the keyed
   #server-status region: renderServerStatus innerHTML-swaps that element
   whenever its content key changes (and pollHealth's failure path
   overwrites it wholesale), so anything rendered inside it is destroyed
   by the 2s poll — exactly how the VRAM donut used to vanish. Out here
   the open state and the rendered content survive repaints by
   construction. Fetched once per open (no poll loop); Refresh re-fetches;
   failures render a quiet retry line and never throw.
   ===================================================================== */
let runtimeConfigLoaded = false;
let runtimeConfigFetchSeq = 0;

function runtimeConfigRow(label, valueHtml, title) {
  return `<div class="rc-row"${title ? ` title="${escapeHtml(title)}"` : ''}>
    <span class="rc-label">${escapeHtml(label)}</span>
    <span class="rc-value">${valueHtml}</span>
  </div>`;
}

// Renders the genuinely useful subset of /v1/config (shape: api/config.rs
// ConfigResponse — vram / kv_cache / training / memory_budget) plus a raw
// pretty-printed JSON toggle so nothing the server reports is hidden.
function renderRuntimeConfigBody(cfg) {
  const vram = cfg.vram || {};
  const kv = cfg.kv_cache || {};
  const train = cfg.training || {};
  const b = cfg.memory_budget || {};
  const generation = cfg.generation || {};
  const srcChip = s => s == null ? '' : ` <span class="rc-source" title="Where this value came from">${escapeHtml(String(s))}</span>`;
  const onOff = v => v ? 'on' : 'off';
  const num = v => (typeof v === 'number' && isFinite(v)) ? v.toLocaleString() : '—';
  const gb = v => (typeof v === 'number' && isFinite(v)) ? v.toFixed(1) + ' GB' : '—';
  return `
    <div class="rc-groups">
      <div class="rc-group">
        <div class="rc-group-title">VRAM detection</div>
        ${runtimeConfigRow('Detected', `<strong>${gb(vram.detected_gb)}</strong>${srcChip(vram.source)}`, 'GPU memory detected at startup, plus the detector that reported it (nvidia-smi, linux-drm-sysfs, KILN_GPU_MEMORY_GB, …).')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">KV cache</div>
        ${runtimeConfigRow('Blocks', `<strong>${num(kv.num_blocks)}</strong>${srcChip(kv.num_blocks_source)}`, 'Paged-attention blocks allocated by the running backend (auto-sized, or pinned via KILN_NUM_BLOCKS).')}
        ${runtimeConfigRow('FP8 cache', `<strong>${onOff(kv.fp8_enabled)}</strong>`, 'Whether the KV cache stores keys/values in FP8 (halves cache memory per token).')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Training</div>
        ${runtimeConfigRow('Grad checkpointing', `<strong>${onOff(train.checkpointing_enabled)}</strong>`, 'Gradient checkpointing trades recompute for activation memory during LoRA training.')}
        ${runtimeConfigRow('Segments', `<strong>${num(train.checkpoint_segments)}</strong>${srcChip(train.checkpoint_segments_source)}`, 'Checkpoint segment count (auto-sized, or pinned via KILN_GRAD_CHECKPOINT_SEGMENTS).')}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Generation</div>
        ${runtimeConfigRow('Thinking default', `<strong>${generation.default_thinking_enabled == null ? 'template' : onOff(generation.default_thinking_enabled)}</strong>`)}
        ${runtimeConfigRow('Thinking tokens', `<strong>${generation.default_thinking_budget_tokens == null ? 'unlimited' : num(generation.default_thinking_budget_tokens)}</strong>`)}
        ${runtimeConfigRow('Thinking time', `<strong>${generation.default_thinking_budget_ms == null ? 'unlimited' : num(generation.default_thinking_budget_ms) + ' ms'}</strong>`)}
      </div>
      <div class="rc-group">
        <div class="rc-group-title">Memory budget</div>
        ${runtimeConfigRow('Total VRAM', `<strong>${gb(b.total_vram_gb)}</strong>`)}
        ${runtimeConfigRow('Model weights', `<strong>${gb(b.model_gb)}</strong>`)}
        ${runtimeConfigRow('KV cache', `<strong>${gb(b.kv_cache_gb)}</strong>`)}
        ${runtimeConfigRow('Training reserve', `<strong>${gb(b.training_budget_gb)}</strong>`)}
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
  if (runtimeConfigLoaded && !force) return;
  const seq = ++runtimeConfigFetchSeq;
  body.innerHTML = '<div class="hint">Loading GET /v1/config…</div>';
  try {
    const cfg = await api('/v1/config');
    if (seq !== runtimeConfigFetchSeq) return; // superseded by a newer refresh
    runtimeConfigLoaded = true;
    body.innerHTML = renderRuntimeConfigBody(cfg);
  } catch (e) {
    if (seq !== runtimeConfigFetchSeq) return;
    runtimeConfigLoaded = false; // the next open retries automatically
    body.innerHTML = `<div class="hint">Couldn't load /v1/config — ${escapeHtml((e && e.message) || 'request failed')}</div>
      <div class="rc-actions"><button class="btn btn-sm" type="button" data-rc-refresh>Retry</button></div>`;
  }
}

// Static shell: wire once at startup. `toggle` fires on open and close;
// fetch on open only (and only the first time, unless Refresh forces it).
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
        <div class="sched-stat" title="Decoded tokens per second across recent streaming completions."><div class="num">&mdash;</div><div class="lbl">tok/s</div></div>
        <div class="sched-stat" title="Median inter-token latency."><div class="num">&mdash;</div><div class="lbl">p50 ITL</div></div>
        <div class="sched-stat" title="99th-percentile inter-token latency."><div class="num">&mdash;</div><div class="lbl">p99 ITL</div></div>
        <div class="sched-stat" title="Streaming completions counted in this rolling window."><div class="num">0</div><div class="lbl">samples</div></div>
      </div>
      <div class="empty" style="margin-top: var(--space-4);">No streaming completions in the last ${window}s. Send a message in <strong>Playground</strong> to populate metrics, or check <a href="/health" target="_blank" rel="noopener noreferrer">/health</a> if the server is still warming up.</div>`;
    } else {
      const tps = data.tok_per_sec.toFixed(1);
      const p50 = data.p50_itl_ms.toFixed(1);
      const p99 = data.p99_itl_ms.toFixed(1);
      const mean = data.mean_itl_ms.toFixed(1);
      key = ['live', tps, p50, p99, mean, data.sample_count, window].join('|');
      html = `<div class="sched-stats">
        <div class="sched-stat" title="Decoded tokens per second across recent streaming completions."><div class="num">${tps}</div><div class="lbl">tok/s</div></div>
        <div class="sched-stat" title="Median inter-token latency."><div class="num">${p50}<span style="font-size:0.55em;color:var(--text-muted);font-weight:500;"> ms</span></div><div class="lbl">p50 ITL</div></div>
        <div class="sched-stat" title="99th-percentile inter-token latency."><div class="num">${p99}<span style="font-size:0.55em;color:var(--text-muted);font-weight:500;"> ms</span></div><div class="lbl">p99 ITL</div></div>
        <div class="sched-stat" title="Streaming completions counted in this rolling window."><div class="num">${data.sample_count}</div><div class="lbl">samples · ${window}s</div></div>
      </div>
      <div style="margin-top: var(--space-3); font-size: var(--text-xs); color: var(--text-muted);">Mean inter-token latency: <span class="tabular-nums" style="color: var(--text-2);">${mean} ms</span></div>`;
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
// truncated (max_tokens-clipped) completion, or a silent fallback to the base
// model while the server claims a non-base adapter is active. These are the
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
    const promptText = r.prompt_preview || '—';
    const completionText = r.completion_preview || '—';
    const attn = requestNeedsAttention(r);
    return `
      <li class="recent-row${attn ? ' attn' : ''}" data-ts="${r.timestamp_unix_ms || 0}" data-id="${escapeHtml(r.id || '')}" tabindex="0" role="button" aria-label="Inspect request ${escapeHtml(shortId(r.id || ''))} from ${escapeHtml(r._client.label)}${attn ? ' — needs attention' : ''}">
        <div class="recent-time">${fmtRelTime(r.timestamp_unix_ms)}</div>
        <div class="recent-previews">
          <div class="recent-prompt" title="${escapeHtml(promptText)}">${agentPill}${adapterPill}${streamPill}${escapeHtml(promptText)}</div>
          <div class="recent-completion" title="${escapeHtml(completionText)}">${escapeHtml(completionText)}</div>
        </div>
        <div class="recent-meta">
          <span class="recent-tokens">${tokens}${ttft}</span>
          <span>${finishPill}${dur ? `<span class="tabular-nums">${escapeHtml(dur)}</span>` : ''}</span>
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
  // Latency breakdown — the experience pi actually felt: the wait for the first
  // token (TTFT), then how fast the rest streamed. Only meaningful when we have
  // both a TTFT and a total duration (i.e. a streamed completion).
  let latencyHtml = '';
  if (r.ttft_ms != null && r.duration_ms != null && r.duration_ms >= r.ttft_ms) {
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
      ${latencyHtml}
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

// --- Training Queue ---
const cancellingTrainingJobIds = new Set();
// Flat snapshot of the latest /v1/train/queue payload so the command
// palette (and any other consumer) can search training jobs without
// re-issuing the request. Updated on every pollTraining tick.
// null until the first SUCCESSFUL fetch: an unfetched (or failing) queue is
// unknown, not empty — seeding an empty shape here made selectPage('training')
// auto-switch to the SFT form during outages, hiding the failure panel and
// its Retry button.
let trainingJobsCache = null;
// Skip the wholesale `tab-queue` innerHTML rewrite when nothing changed
// (running progress, queued list identity, recent finish-state). Mirrors
// the `lastAdaptersKey` guard on the adapters tab.
let lastTrainingKey = null;

// Job states from the previous poll, so finishing is an EVENT the UI announces
// (with the next action attached) instead of a silent row moving between lists.
let prevTrainingStates = null;
function detectTrainingTransitions(data) {
  const now = new Map();
  if (data.running) now.set(data.running.job_id, 'running');
  (data.queued || []).forEach(j => now.set(j.job_id, 'queued'));
  (data.completed || []).forEach(j => now.set(j.job_id, (j.state || 'completed').toString().toLowerCase()));
  if (prevTrainingStates) {
    for (const [id, state] of now) {
      const prev = prevTrainingStates.get(id);
      if (prev === state) continue;
      // Start is an announce-only event (no toast — the submit flow already
      // confirms visually): a job begins running that wasn't running before,
      // whether it stepped queued→running or appeared mid-poll already running.
      if (state === 'running') {
        const adapter = (data.running && data.running.adapter_name) || 'adapter';
        announceStatus('training-queue-status', `Training started: ${adapter}.`);
        continue;
      }
      // Only announce jobs we watched run/queue in THIS session — never history.
      if (!prev || (prev !== 'running' && prev !== 'queued')) continue;
      const j = (data.completed || []).find(x => x.job_id === id) || {};
      const adapter = j.adapter_name || 'adapter';
      if (state === 'completed') {
        announceStatus('training-queue-status', `Training completed: ${adapter} is ready.`);
        actionToast(`${adapter} finished training — it's ready${j.job_type ? ' (' + j.job_type + ')' : ''}.`, 'ok', [
          { label: 'Prove it vs base', onClick: () => openAdapterEvalModal(adapter) },
          { label: 'View job', onClick: () => { selectPage('training'); document.querySelector('#page-training [data-tab="queue"]')?.click(); } },
        ]);
      } else if (state === 'failed' || state === 'error') {
        announceStatus('training-queue-status', `Training failed: ${adapter}.`);
        actionToast(`Training ${adapter} failed.`, 'err', [
          { label: 'View job', onClick: () => { selectPage('training'); document.querySelector('#page-training [data-tab="queue"]')?.click(); } },
        ]);
      }
    }
  }
  prevTrainingStates = now;
}

async function pollTraining() {
  const queuePanel = setPanelBusy('tab-queue', true);
  if (!queuePanel) return;
  try {
    const data = await api('/v1/train/queue');
    trainingJobsCache = {
      running: data.running || null,
      queued: data.queued || [],
      completed: data.completed || [],
    };
    detectTrainingTransitions(data);
    watchCorrectionsJob(data);
    const r = data.running;
    const key = [
      r ? `${r.job_id}:${(r.progress || 0).toFixed(3)}:${r.current_loss != null ? r.current_loss.toFixed(4) : ''}` : '',
      (data.queued || []).map(j => j.job_id).join(','),
      // Completed jobs are NOT immutable: the §8.7 gate eval stamps
      // post_eval_verdict/gate_outcome minutes AFTER state flips to
      // 'completed', and failed jobs carry an error message. Key on their
      // presence too, or the verdict pill / error line never repaints
      // until some unrelated change touches the list.
      (data.completed || []).map(j => `${j.job_id}:${j.state}:${j.gate_outcome || (j.post_eval_verdict ? 'v' : '')}:${j.error ? 'e' : ''}`).join(','),
    ].join('|');
    if (key !== lastTrainingKey) {
      lastTrainingKey = key;
      renderTrainingQueue(data);
    }
    const liveCount = (data.running ? 1 : 0) + (data.queued ? data.queued.length : 0);
    setText('training-count', String(liveCount));
    updateFlywheel();
  } catch (e) {
    // Invalidate the queue's render key — the failure HTML replaced the list.
    lastTrainingKey = null;
    queuePanel.innerHTML = apiFailureHtml('Training queue', e, 'pollTraining');
  } finally {
    setPanelBusy('tab-queue', false);
  }
}

let trainingQueueFilter = '';
function matchTraining(j) {
  const q = trainingQueueFilter.trim().toLowerCase();
  if (!q) return true;
  return [
    j.job_id || '',
    j.adapter_name || '',
    (j.job_type || '').toString(),
    (j.state || '').toString(),
  ].join(' ').toLowerCase().includes(q);
}
function renderTrainingQueue(data) {
  const el = document.getElementById('tab-queue');
  // Snapshot filter focus/selection BEFORE rewriting innerHTML so a
  // background poll that fires while the user is typing doesn't yank
  // focus away mid-keystroke.
  let restoreFocus = false;
  let restoreSelStart = 0;
  let restoreSelEnd = 0;
  const prevFilter = document.getElementById('training-queue-filter');
  if (prevFilter && document.activeElement === prevFilter) {
    restoreFocus = true;
    restoreSelStart = prevFilter.selectionStart || 0;
    restoreSelEnd = prevFilter.selectionEnd || 0;
  }
  const totalAll = (data.running ? 1 : 0)
    + (data.queued ? data.queued.length : 0)
    + (data.completed ? data.completed.length : 0);
  // Filter input is always present (even when 0 results) so the user can
  // clear / change the filter without re-navigating.
  const filterBar = totalAll > 0
    ? `<div class="evals-toolbar" style="padding:0 0 var(--space-3) 0;">
        <input class="search-input" id="training-queue-filter" type="search" placeholder="Filter by adapter, type, state, job id…" aria-label="Filter training jobs" value="${escapeHtml(trainingQueueFilter)}">
      </div>`
    : '';

  const runningMatch = data.running && matchTraining(data.running);
  const queuedMatch = (data.queued || []).filter(matchTraining);
  const completedMatch = (data.completed || []).filter(matchTraining);

  // Always render in the same flat container (training-cards) — separate
  // queue/running/completed visually with section labels but a uniform
  // card style so eye-tracking is constant across job states.
  let html = filterBar + '<div class="training-cards">';
  if (runningMatch) {
    html += `<div class="queue-section-label">Running</div>`;
    html += renderTrainingCard(data.running, 'running');
  }
  if (queuedMatch.length > 0) {
    html += `<div class="queue-section-label">Queued</div>`;
    queuedMatch.forEach(q => {
      html += renderTrainingCard(q, 'queued');
    });
  }
  if (completedMatch.length > 0) {
    const total = completedMatch.length;
    const totalLabel = totalAll && total !== (data.completed?.length || 0)
      ? `${total} of ${data.completed?.length || 0}`
      : `${total}`;
    html += `<div class="queue-section-label">Recent <span class="hint" style="font-weight:400;">· ${totalLabel} job${total === 1 ? '' : 's'}</span></div>`;
    completedMatch.forEach(j => {
      html += renderTrainingCard(j, 'completed');
    });
  }
  html += '</div>';
  if (totalAll > 0 && !runningMatch && !queuedMatch.length && !completedMatch.length) {
    html += `<div class="eval-empty" style="margin-top:var(--space-3);"><div class="eval-empty-body">No training jobs match <code>${escapeHtml(trainingQueueFilter)}</code>.</div></div>`;
  }

  if (!data.running && (!data.queued || !data.queued.length) && (!data.completed || !data.completed.length)) {
    html = `<div class="eval-empty empty">
      <div class="eval-empty-icon"><svg class="icn"><use href="#i-flask"></use></svg></div>
      <div class="eval-empty-title">No training jobs yet.</div>
      <div class="eval-empty-body">Submit SFT examples to teach a correction, or use GRPO for scored completions. Datasets uploaded under Evals can be picked directly in the SFT/GRPO submit forms. New here? Read the <a href="https://ericflo.github.io/kiln/quickstart.html" target="_blank" rel="noopener">Quickstart</a> or the <a href="https://ericflo.github.io/kiln/grpo.html" target="_blank" rel="noopener">GRPO Guide</a>.</div>
      <button class="eval-empty-cta" type="button" onclick="document.getElementById('training-tab-sft').click();">Train your first adapter</button>
    </div>`;
  }
  el.innerHTML = html;
  // Wire card clicks for drill-in (queued/running/completed all open the modal).
  el.querySelectorAll('[data-train-job-id]').forEach(card => {
    card.addEventListener('click', ev => {
      // Don't trigger drill if user clicked an inline action button.
      if (ev.target.closest('[data-train-cancel],[data-train-prove]')) return;
      openTrainDrillModal(card.dataset.trainJobId);
    });
  });
  el.querySelectorAll('[data-train-cancel]').forEach(b => {
    b.addEventListener('click', ev => {
      ev.stopPropagation();
      cancelJobFromButton(b);
    });
  });
  // Persistent "Prove it vs base" on completed cards — same modal the
  // completion toast offers, so a missed toast is never a dead-end.
  el.querySelectorAll('[data-train-prove]').forEach(b => {
    b.addEventListener('click', ev => {
      ev.stopPropagation();
      openAdapterEvalModal(b.dataset.adapter);
    });
  });
  if (restoreFocus) {
    const f = document.getElementById('training-queue-filter');
    if (f) {
      f.focus();
      try { f.setSelectionRange(restoreSelStart, restoreSelEnd); } catch {}
    }
  }
}

/// Render a training job as a rich card with progress bar, loss curve
/// (when history is available), and per-job stats. State drives layout:
/// `queued` shows position; `running` shows live progress + curve;
/// `completed` shows final loss + duration.
function renderTrainingCard(j, state) {
  const pct = ((j.progress || 0) * 100).toFixed(0);
  const adapterLabel = j.adapter_name ? escapeHtml(j.adapter_name) : `<span class="hint">(unnamed)</span>`;
  const jobType = (j.job_type || '').toString().toLowerCase();
  // Loss curve (only present when we've run job_detail at least once
  // for this job — populated by openTrainDrillModal). We just leave a
  // placeholder for now.
  const isRunning = state === 'running';
  // State class drives the amber rule: only a RUNNING job's bar is hot (amber);
  // completed → green, failed → red, queued → neutral.
  const stateClass = isRunning ? 'training-card-running' : 'training-card-' + (j.state || state || 'done').toString().toLowerCase();
  const cardClass = 'training-card ' + stateClass;
  let stateBadge;
  if (state === 'queued') {
    stateBadge = `<span class="job-state-pill queued">queued${j.position ? ' · #'+j.position : ''}</span>`;
  } else if (state === 'running') {
    stateBadge = `<span class="job-state-pill running">running</span>`;
  } else {
    const stateNorm = (j.state || '').toString().toLowerCase();
    stateBadge = `<span class="job-state-pill ${stateNorm}">${escapeHtml(stateNorm || 'completed')}</span>`;
  }
  const stateNormForActions = (j.state || state || '').toString().toLowerCase();
  // Completed cards carry the next action PERSISTENTLY — the completion toast
  // is a courtesy, not the only door. Failed cards surface the reason inline.
  let actionBtn = '';
  if (state === 'queued') {
    actionBtn = `<button class="btn btn-sm" data-train-cancel data-job-id="${escapeHtml(j.job_id)}" type="button" style="margin-left:auto;">Cancel</button>`;
  } else if (state === 'running') {
    // Running jobs are stoppable too: the server sets a cooperative flag
    // and the trainer aborts at the next step boundary.
    actionBtn = `<button class="btn btn-sm" data-train-cancel data-job-id="${escapeHtml(j.job_id)}" type="button" style="margin-left:auto;" title="Stop at the next training step">Stop</button>`;
  } else if (stateNormForActions === 'completed' && j.adapter_name) {
    actionBtn = `<button class="btn btn-sm" data-train-prove data-adapter="${escapeHtml(j.adapter_name)}" type="button" style="margin-left:auto;" title="Grade ${escapeHtml(j.adapter_name)} against base on an eval suite">Prove it vs base</button>`;
  }
  // §8.7 promotion-gate verdict pill. Color keys off the server's
  // machine-readable `gate_outcome` (stamped next to the prose verdict):
  //   promoted          → green (gate passed, adapter serving)
  //   kept              → amber chip with a CHECK icon — a pass without a
  //                       requested promotion is a success, not a warning
  //   regression/demoted → red (rejected vs baseline / demoted to .failed)
  //   error             → amber + warning icon (gate couldn't measure)
  // Pill text stays the prose verdict. Rendered whenever the backend
  // stamped a verdict so a silent demotion can't hide.
  let gateLine = '';
  if (j.post_eval_verdict || j.gate_outcome) {
    const v = String(j.post_eval_verdict || j.gate_outcome);
    const OUTCOME_CLS = { promoted: 'ok', kept: 'warn', regression: 'err', demoted: 'err', error: 'warn' };
    let cls = OUTCOME_CLS[j.gate_outcome] || '';
    if (!cls) {
      // Fallback ONLY for jobs archived before `gate_outcome` existed
      // (and for older servers): those carry prose alone, so classify by
      // substring as the UI historically did. Known-imperfect — that
      // heuristic is exactly why gate_outcome was added.
      cls = (v.includes('promoted') && !v.includes('NOT')) ? 'ok'
        : (v.includes('.failed') || v.includes('demoted') || v.includes('REGRESSION')) ? 'err' : 'warn';
    }
    const iconName = (cls === 'ok' || j.gate_outcome === 'kept') ? 'check' : 'warning';
    gateLine = `<div class="training-card-gate gate-${cls}" title="${escapeHtml(v)}">${icon(iconName, 'icn-sm')} ${escapeHtml(v.slice(0, 160))}</div>`;
  }
  const errLine = (stateNormForActions === 'failed' && j.error)
    ? `<div class="training-card-error">${icon('warning', 'icn-sm')} ${escapeHtml(String(j.error).slice(0, 220))}</div>`
    : '';
  const cancelBtn = actionBtn;
  // Prefer the wall-clock timestamps (`submitted_unix_ms` /
  // `finished_unix_ms`) introduced with the on-disk archive — those
  // survive restarts. Fall back to `elapsed_secs` only when the server
  // is on an older payload (no wall-clock fields).
  let timeBadge = '';
  if (state === 'completed' && j.finished_unix_ms) {
    timeBadge = `<span class="hint" style="font-size:11px;" title="${escapeHtml(new Date(j.finished_unix_ms).toISOString())}">finished ${escapeHtml(fmtSmartTime(j.finished_unix_ms))}</span>`;
  } else if (j.submitted_unix_ms) {
    timeBadge = `<span class="hint" style="font-size:11px;" title="${escapeHtml(new Date(j.submitted_unix_ms).toISOString())}">started ${escapeHtml(fmtSmartTime(j.submitted_unix_ms))}</span>`;
  } else if (j.elapsed_secs != null) {
    timeBadge = `<span class="hint" style="font-size:11px;">${escapeHtml(Math.floor(j.elapsed_secs) + 's elapsed')}</span>`;
  }
  return `<div class="${cardClass}" data-train-job-id="${escapeHtml(j.job_id)}">
    <div class="training-card-head">
      ${stateBadge}
      <span class="training-card-name"><span class="hint" style="font-weight:400;margin-right:4px;">Adapter:</span>${adapterLabel}</span>
      <span class="training-card-type ${escapeHtml(jobType)}">${escapeHtml(jobType)}</span>
      <span class="hint" style="font-family:var(--font-mono); font-size:11px;">${escapeHtml(shortId(j.job_id))}</span>
      ${timeBadge}
      ${cancelBtn}
    </div>
    <div class="training-card-progress">
      <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${pct}%;"></div></div>
      <div class="training-stat"><span class="training-stat-num">${pct}%</span><span class="training-stat-label">progress</span></div>
      <div class="training-stat">
        <span class="training-stat-num">${j.current_loss != null ? j.current_loss.toFixed(3) : '—'}</span>
        <span class="training-stat-label">${j.current_loss != null ? 'loss' : 'not started'}</span>
      </div>
    </div>
    ${gateLine}${errLine}
    <div class="training-card-curve" id="training-curve-${escapeHtml(j.job_id)}"></div>
  </div>`;
}

// Cancel buttons receive the click via stopPropagation; this is a thin
// indirection that resolves the job_id and forwards to the existing
// cancellation flow.
function cancelJobFromButton(btn) {
  const jobId = btn.dataset.jobId || '';
  if (!jobId || cancellingTrainingJobIds.has(jobId)) return;
  btn.disabled = true;
  btn.textContent = 'Cancelling…';
  cancelJob(jobId, btn);
}

window.cancelJob = async function(jobId, button) {
  if (!jobId || cancellingTrainingJobIds.has(jobId)) return;
  cancellingTrainingJobIds.add(jobId);
  try {
    await api('/v1/train/queue/' + jobId, { method: 'DELETE' });
    toast('Cancelled job ' + jobId.slice(0, 8));
    cancellingTrainingJobIds.delete(jobId);
    pollTraining();
  } catch (e) {
    cancellingTrainingJobIds.delete(jobId);
    if (button) {
      button.disabled = false;
      button.textContent = 'Cancel';
    }
    toast(e.message, 'err');
  }
};

function fillSftSamplePayload() {
  const sample = [
    {
      messages: [
        { role: 'user', content: 'Translate to French: Hello' },
        { role: 'assistant', content: 'Bonjour' },
      ],
    },
  ];
  const textarea = document.getElementById('sft-examples');
  const samplePayload = JSON.stringify(sample, null, 2);
  if (textarea.value.trim() && textarea.value !== samplePayload && !confirm('Replace the current SFT examples with the sample payload?')) {
    return;
  }
  textarea.value = samplePayload;
  const pasteRow = document.getElementById('sft-paste-row'); if (pasteRow) pasteRow.hidden = false;
  clearTrainingData('sft');
  updateSftSubmitState();
  textarea.focus();
  toast('Sample SFT payload inserted — edit it or Train as-is');
}

function fillGrpoSamplePayload() {
  const sample = [
    {
      messages: [
        { role: 'user', content: 'Write a haiku about the moon' },
      ],
      completions: [
        {
          text: 'Silent moonlit night / Silver clouds drift softly by / Dreams bloom in starlight',
          reward: 0.9,
        },
        { text: 'The moon is bright tonight.', reward: 0.2 },
      ],
    },
  ];
  const textarea = document.getElementById('grpo-groups');
  const samplePayload = JSON.stringify(sample, null, 2);
  if (textarea.value.trim() && textarea.value !== samplePayload && !confirm('Replace the current GRPO groups with the sample payload?')) {
    return;
  }
  textarea.value = samplePayload;
  const pasteRow = document.getElementById('grpo-paste-row'); if (pasteRow) pasteRow.hidden = false;
  clearTrainingData('grpo');
  updateGrpoSubmitState();
  textarea.focus();
  toast('Sample GRPO payload inserted — edit it or Train as-is');
}

document.getElementById('use-sft-sample').addEventListener('click', fillSftSamplePayload);
document.getElementById('use-grpo-sample').addEventListener('click', fillGrpoSamplePayload);

/* ====== Direct "drop a file and train" data input (SFT + GRPO) ==============
   /v1/train/{sft,grpo} take inline examples/groups, so the primary path is:
   drop a .jsonl/.json file -> parse + validate + preview in place -> train.
   No Evals detour, no dropdown round-trip, no megabytes pasted into a textarea.
   trainingData[kind] holds parsed items from the file/dataset path; the textarea
   is the secondary "paste" path. Exactly one source is active at a time. */
const trainingData = { sft: null, grpo: null };
const TRAIN_KIND = {
  sft:  { noun: 'example', datasetFormat: 'sft_chat',  pickId: 'sft-dataset-pick',  textareaId: 'sft-examples', update: () => updateSftSubmitState(),  valid: (it) => sftItemValid(it) },
  grpo: { noun: 'group',   datasetFormat: 'grpo_groups', pickId: 'grpo-dataset-pick', textareaId: 'grpo-groups',  update: () => updateGrpoSubmitState(), valid: (it) => grpoItemValid(it) },
};

function parseTrainingText(text) {
  const t = (text || '').trim();
  if (!t) return [];
  if (t[0] === '[') {
    const arr = JSON.parse(t);
    if (!Array.isArray(arr)) throw new Error('Top-level JSON must be an array.');
    return arr;
  }
  const items = [];
  t.split('\n').forEach((line, i) => {
    const s = line.trim(); if (!s) return;
    try { items.push(JSON.parse(s)); }
    catch { throw new Error(`Line ${i + 1} isn't valid JSON. Use JSONL (one object per line) or a JSON array.`); }
  });
  return items;
}
function sftItemValid(it) {
  if (!it || !Array.isArray(it.messages) || !it.messages.length) return false;
  const roles = it.messages.map(m => m && m.role);
  return roles.includes('user') && roles.includes('assistant');
}
function grpoItemValid(it) {
  if (!it || !Array.isArray(it.messages) || !it.messages.length) return false;
  if (!Array.isArray(it.completions) || !it.completions.length) return false;
  return it.completions.every(c => c && typeof c.text === 'string' && c.text.trim() && typeof c.reward === 'number' && Number.isFinite(c.reward));
}
function suggestAdapterName(filename, kind) {
  let base = (filename || '').replace(/\.[^.]+$/, '').toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40);
  if (!base) base = kind + '-data';
  return base.endsWith('-' + kind) ? base : base + '-' + kind;
}
function trainingDatasetNameForFile(filename, kind) {
  const base = suggestAdapterName(filename, kind)
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || (kind + '-data');
  return `${base}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
// Auto-fill the adapter name from the data source UNLESS the user typed their
// own. Tracks a dirty flag instead of matching the literal default, so loading
// a second file/dataset re-suggests (no silent name collisions) while a
// hand-chosen name is never clobbered.
function maybeSuggestAdapterName(kind, sourceLabel) {
  const input = document.getElementById(kind + '-output-name');
  if (!input) return;
  if (input.dataset.userEdited === '1' && input.value.trim()) return;
  input.value = suggestAdapterName(sourceLabel, kind);
  TRAIN_KIND[kind].update();
}
['sft', 'grpo'].forEach(kind => {
  document.getElementById(kind + '-output-name')?.addEventListener('input', e => {
    e.target.dataset.userEdited = e.target.value.trim() ? '1' : '';
  });
});
function renderTrainingDataStatus(kind, total, valid, label, badIdx) {
  const el = document.getElementById(kind + '-data-status');
  const K = TRAIN_KIND[kind];
  if (!el) return;
  const why = kind === 'sft' ? 'needs a user message and an assistant reply' : 'needs messages plus scored completions';
  if (!valid) {
    el.hidden = false; el.className = 'train-data-status is-bad';
    el.innerHTML = `${icon('warning', 'icn-sm')} No usable ${K.noun}s in ${escapeHtml(label || 'that input')}. Each ${why}.`;
    return;
  }
  const skipped = total - valid;
  // Name the offending entries so nobody has to open the file and hunt.
  let skipDetail = '';
  if (skipped > 0 && Array.isArray(badIdx) && badIdx.length) {
    const shown = badIdx.slice(0, 3).map(i => '#' + (i + 1)).join(', ');
    skipDetail = ` · skipped ${escapeHtml(shown)}${badIdx.length > 3 ? ` +${badIdx.length - 3} more` : ''} (each ${why})`;
  } else if (skipped > 0) {
    skipDetail = ` · ${skipped} skipped (invalid)`;
  }
  el.hidden = false; el.className = 'train-data-status is-good';
  el.innerHTML = `${icon('check', 'icn-sm')} <strong>${valid}</strong> ${K.noun}${valid === 1 ? '' : 's'} ready`
    + (skipDetail ? `<span class="train-data-skip">${skipDetail}</span>` : '')
    + ` <span class="train-data-src">from ${escapeHtml(label || 'input')}</span>`;
}
// Set the file/dataset source for a kind. Clears the textarea so there's one
// source of truth, validates, previews, and re-checks submit readiness.
function setTrainingData(kind, items, label) {
  const K = TRAIN_KIND[kind];
  const valid = [], badIdx = [];
  (items || []).forEach((it, i) => { if (K.valid(it)) valid.push(it); else badIdx.push(i); });
  trainingData[kind] = valid.length ? { items: valid, total: (items || []).length, label } : null;
  const ta = document.getElementById(K.textareaId);
  if (ta) ta.value = '';                       // file/dataset is now the source
  renderTrainingDataStatus(kind, (items || []).length, valid.length, label, badIdx);
  K.update();
  return valid.length;
}
function clearTrainingData(kind) {
  trainingData[kind] = null;
  const el = document.getElementById(kind + '-data-status'); if (el) el.hidden = true;
}
async function loadTrainingFile(kind, file) {
  if (!file) return;
  const K = TRAIN_KIND[kind];
  const previous = trainingData[kind];
  try {
    const datasetName = trainingDatasetNameForFile(file.name, kind);
    trainingData[kind] = null;
    const el = document.getElementById(kind + '-data-status');
    if (el) {
      el.hidden = false; el.className = 'train-data-status';
      el.innerHTML = `${icon('upload', 'icn-sm')} Uploading ${escapeHtml(file.name)} into the local dataset store…`;
    }
    K.update();
    const manifest = await postDatasetUpload(
      datasetName,
      K.datasetFormat,
      `Uploaded from ${file.name} for ${kind.toUpperCase()} training`,
      file,
    );
    const n = await loadNamedDatasetIntoTraining(kind, manifest.name || datasetName);
    if (n) maybeSuggestAdapterName(kind, file.name);
    if (typeof refreshDatasets === 'function') refreshDatasets();
    refreshDatasetPicker(kind);
    toast(`Uploaded "${manifest.name || datasetName}" (${Number(manifest.num_rows || 0).toLocaleString()} rows)`, 'ok');
  } catch (e) {
    // A bad drop must never destroy data you already loaded — keep it and say so.
    trainingData[kind] = previous;
    const el = document.getElementById(kind + '-data-status');
    if (el) {
      el.hidden = false; el.className = 'train-data-status is-bad';
      el.innerHTML = `${icon('warning', 'icn-sm')} ${escapeHtml(e.message)}`
        + (previous ? ` <span class="train-data-src">— kept your previous data (${escapeHtml(previous.label || 'loaded input')})</span>` : '');
    }
    K.update();
  }
}
function wireDropzone(kind) {
  const zone = document.getElementById(kind + '-dropzone');
  const file = document.getElementById(kind + '-file');
  if (!zone || !file) return;
  zone.addEventListener('click', () => file.click());
  zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); } });
  file.addEventListener('change', () => { loadTrainingFile(kind, file.files[0]); file.value = ''; });
  ['dragenter', 'dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('is-drag'); }));
  ['dragleave', 'dragend'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('is-drag'); }));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('is-drag'); const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]; if (f) loadTrainingFile(kind, f); });
}
// Secondary affordance toggles (pick a dataset / paste JSON) reveal their row.
function wireTrainingAlts(kind) {
  const pickRow = document.getElementById(kind + '-pick-row');
  const pasteRow = document.getElementById(kind + '-paste-row');
  document.getElementById(kind + '-pick-toggle')?.addEventListener('click', () => { if (pickRow) { pickRow.hidden = !pickRow.hidden; if (!pickRow.hidden) refreshDatasetPicker(kind); } });
  document.getElementById(kind + '-paste-toggle')?.addEventListener('click', () => { if (pasteRow) { pasteRow.hidden = !pasteRow.hidden; if (!pasteRow.hidden) document.getElementById(TRAIN_KIND[kind].textareaId)?.focus(); } });
}
wireDropzone('sft'); wireDropzone('grpo');
wireTrainingAlts('sft'); wireTrainingAlts('grpo');

// Advanced hyperparameters live behind a toggle; a one-line summary narrates the
// current values so collapsing never hides information.
function wireAdvanced(kind, summarize) {
  const btn = document.getElementById(kind + '-adv-toggle');
  const body = document.getElementById(kind + '-advanced');
  const summary = document.getElementById(kind + '-adv-summary');
  if (!btn || !body) return;
  btn.addEventListener('click', () => {
    const open = body.hidden;
    body.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
  });
  const update = () => { if (summary) summary.textContent = summarize(); };
  body.querySelectorAll('input, select').forEach(i => {
    i.addEventListener('input', update);
    i.addEventListener('change', update);
  });
  update();
}
// A blank learning-rate field means "auto" — the server resolves the
// per-optimizer default (Muon vs AdamW want very different bands).
const lrSummary = id => (document.getElementById(id)?.value || '').trim() || 'auto';
const optimizerLabel = id => {
  const value = document.getElementById(id)?.value || 'muon';
  if (value === 'adam_w') return 'AdamW';
  if (value === 'sgd') return 'SGD';
  return 'Muon';
};
function readTrainingOptimizer(kind) {
  const value = document.getElementById(kind + '-optimizer')?.value || 'muon';
  if (value === 'adam_w') return { kind: 'adam_w' };
  if (value === 'sgd') return { kind: 'sgd' };
  return { kind: 'muon' };
}
wireAdvanced('sft', () => {
  const v = id => document.getElementById(id)?.value || '?';
  const lr = lrSummary('sft-learning-rate');
  const opt = optimizerLabel('sft-optimizer');
  const isDefault = v('sft-epochs') === '3' && opt === 'Muon' && lr === 'auto' && v('sft-rank') === '8';
  if (typeof updateSftOverfitHint === 'function') updateSftOverfitHint();
  return `${v('sft-epochs')} epochs · ${opt} · learning rate ${lr} · LoRA rank ${v('sft-rank')}`
    + (isDefault ? ' — sensible defaults, no tuning needed' : ' — customized');
});
wireAdvanced('grpo', () => {
  const v = id => document.getElementById(id)?.value || '?';
  const lr = lrSummary('grpo-learning-rate');
  const opt = optimizerLabel('grpo-optimizer');
  const isDefault = v('grpo-kl-coeff') === '0.1' && opt === 'Muon' && lr === 'auto' && v('grpo-rank') === '8';
  return `KL ${v('grpo-kl-coeff')} · ${opt} · learning rate ${lr} · LoRA rank ${v('grpo-rank')}`
    + (isDefault ? ' — sensible defaults, no tuning needed' : ' — customized');
});

// "Prove it after training" — wires the server's post_eval auto-hook: when
// checked, the train request carries post_eval:{suite, include_baseline:true}
// and Kiln grades the fresh adapter AND base the moment training completes.
// The row only appears when eval suites actually exist (no dead control).
async function refreshProveRows() {
  let suites = [];
  try { const d = await api('/v1/eval/suites'); suites = d.suites || []; } catch (_) { /* leave hidden */ }
  for (const kind of ['sft', 'grpo']) {
    const row = document.getElementById(kind + '-prove-row');
    const sel = document.getElementById(kind + '-prove-suite');
    const check = document.getElementById(kind + '-prove');
    if (!row || !sel || !check) continue;
    if (!suites.length) { row.hidden = true; continue; }
    row.hidden = false;
    const cur = sel.value;
    sel.innerHTML = suites.map(s => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.name)}${s.num_examples ? ' · ' + s.num_examples + ' examples' : ''}</option>`).join('');
    if (cur && suites.some(s => s.name === cur)) sel.value = cur;
    sel.disabled = !check.checked;
  }
}
document.getElementById('sft-prove')?.addEventListener('change', e => { const s = document.getElementById('sft-prove-suite'); if (s) s.disabled = !e.target.checked; });
document.getElementById('grpo-prove')?.addEventListener('change', e => { const s = document.getElementById('grpo-prove-suite'); if (s) s.disabled = !e.target.checked; });
function provePostEval(kind) {
  const check = document.getElementById(kind + '-prove');
  const sel = document.getElementById(kind + '-prove-suite');
  if (!check || !check.checked || !sel || !sel.value) return null;
  return { suite: sel.value, include_baseline: true };
}

// Dataset picker, per-form and format-correct (fixes the old SFT-only picker
// that filtered the wrong format and was always empty). Loading sets
// trainingData directly — never dumps rows into a textarea.
async function refreshDatasetPicker(kind) {
  const K = TRAIN_KIND[kind];
  const sel = document.getElementById(K.pickId);
  if (!sel) return;
  try {
    const d = await api('/v1/eval/datasets');
    const datasets = (d.datasets || []).filter(m => m.format === K.datasetFormat);
    const cur = sel.value;
    sel.innerHTML = '<option value="">Select an uploaded dataset…</option>'
      + datasets.map(m => `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)} · ${m.num_rows} rows</option>`).join('');
    if (cur) sel.value = cur;
    // An empty picker is a dead-end without directions — say where data comes from.
    const empty = document.getElementById(kind + '-pick-empty');
    if (empty) empty.hidden = datasets.length > 0;
    sel.hidden = datasets.length === 0;
  } catch (_) { /* best-effort */ }
}
async function loadNamedDatasetIntoTraining(kind, name) {
  const K = TRAIN_KIND[kind];
  // Reference the dataset BY NAME — the server trains on its own copy
  // (/v1/train/* accepts `dataset`), so rows never round-trip through the
  // browser and nothing is truncated. We only fetch the manifest for the count.
  let count = null;
  try {
    const d = await api('/v1/eval/datasets');
    const m = (d.datasets || []).find(x => x.name === name);
    if (m) count = m.num_rows;
  } catch (_) {}
  trainingData[kind] = { datasetName: name, count, label: name };
  const ta = document.getElementById(K.textareaId);
  if (ta) ta.value = '';
  const el = document.getElementById(kind + '-data-status');
  if (el) {
    el.hidden = false; el.className = 'train-data-status is-good';
    el.innerHTML = `${icon('check', 'icn-sm')} <strong>${escapeHtml(name)}</strong>`
      + (count != null ? ` · ${Number(count).toLocaleString()} ${K.noun}${count === 1 ? '' : 's'}` : '')
      + ` <span class="train-data-src">trains on the server's copy — nothing to re-upload, no row limit</span>`;
  }
  K.update();
  maybeSuggestAdapterName(kind, name);
  return count == null ? 1 : count;
}
async function loadDatasetIntoTraining(kind) {
  const sel = document.getElementById(TRAIN_KIND[kind].pickId);
  const name = sel && sel.value;
  if (!name) return; // placeholder option — nothing to load
  try { await loadNamedDatasetIntoTraining(kind, name); }
  catch (e) { toast('Load failed: ' + e.message, 'err'); }
}
// One-click bridge from anywhere a dataset is listed: jump to the right
// training form with the dataset already loaded and the adapter pre-named —
// the next action is just "Train adapter".
async function trainFromDataset(name, kind) {
  selectPage('training');
  document.getElementById('training-tab-' + kind)?.click();
  try {
    const n = await loadNamedDatasetIntoTraining(kind, name);
    if (n) toast(`${name} loaded — review the name, then Train adapter`, 'ok');
  } catch (e) { toast('Could not load ' + name + ': ' + e.message, 'err'); }
}
// Selecting a dataset LOADS it immediately — no separate Load click, and no
// way for the dropdown to show one dataset while another is actually held.
document.getElementById('sft-dataset-pick')?.addEventListener('change', () => loadDatasetIntoTraining('sft'));
document.getElementById('grpo-dataset-pick')?.addEventListener('change', () => loadDatasetIntoTraining('grpo'));
document.querySelectorAll('[data-goto-datasets]').forEach(b => b.addEventListener('click', () => {
  selectPage('evals');
  document.getElementById('evals-tab-datasets')?.click();
  setTimeout(() => document.getElementById('dataset-name')?.focus(), 120);
}));
document.getElementById('training-tab-sft')?.addEventListener('click', () => { refreshDatasetPicker('sft'); refreshProveRows(); });
document.getElementById('training-tab-grpo')?.addEventListener('click', () => { refreshDatasetPicker('grpo'); refreshProveRows(); });
refreshProveRows();

function parseJsonArrayField(value, label) {
  const text = value.trim();
  if (!text) {
    throw new Error(`${label} cannot be empty. Paste a JSON array or use the sample payload.`);
  }

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} must be valid JSON. Check commas, quotes, and brackets.`);
  }

  if (!Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON array, not an object or single item.`);
  }
  if (parsed.length === 0) {
    throw new Error(`${label} must include at least one item.`);
  }
  return parsed;
}

function parseFiniteNumberField(value, label) {
  const text = value.trim();
  if (!text) {
    throw new Error(`${label} is required.`);
  }
  const parsed = Number(text);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return parsed;
}

// Blank means "omit the field" (the server resolves a default); anything
// non-blank must still be a real number.
function parseOptionalFiniteNumberField(value, label) {
  if (!value.trim()) return null;
  return parseFiniteNumberField(value, label);
}

function parsePositiveIntegerField(value, label) {
  const parsed = parseFiniteNumberField(value, label);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive whole number.`);
  }
  return parsed;
}

function parseQuickInferenceTemperature(input) {
  const text = input.value.trim();
  const parsed = Number(text);
  if (!text || !Number.isFinite(parsed) || parsed < 0 || parsed > 2) {
    throw new Error('Temperature must be between 0 and 2.');
  }
  return parsed;
}

function validateMessages(messages, label) {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error(`${label} needs a non-empty messages array.`);
  }
  const roles = messages.map((message) => message && message.role);
  if (!roles.includes('user') || !roles.includes('assistant')) {
    throw new Error(`${label} messages need both user and assistant roles.`);
  }
}

function validateSftExamples(examples) {
  examples.forEach((example, index) => {
    validateMessages(example && example.messages, `SFT example ${index + 1}`);
  });
}

function validateGrpoGroups(groups) {
  groups.forEach((group, groupIndex) => {
    const label = `GRPO group ${groupIndex + 1}`;
    if (!group || typeof group !== 'object') {
      throw new Error(`${label} must be an object with messages and completions.`);
    }
    if (!Array.isArray(group.messages) || group.messages.length === 0) {
      throw new Error(`${label} needs a non-empty messages array.`);
    }
    if (!Array.isArray(group.completions) || group.completions.length === 0) {
      throw new Error(`${label} needs a non-empty completions array.`);
    }
    group.completions.forEach((completion, completionIndex) => {
      if (!completion || typeof completion.text !== 'string' || !completion.text.trim()) {
        throw new Error(`${label} completion ${completionIndex + 1} needs non-empty text.`);
      }
      if (typeof completion.reward !== 'number' || !Number.isFinite(completion.reward)) {
        throw new Error(`${label} completion ${completionIndex + 1} needs a numeric reward, not a quoted string.`);
      }
    });
  });
}

function parseAdapterNameField(input) {
  const adapterName = input.value.trim();
  if (!adapterName) {
    input.focus();
    throw new Error('Adapter name is required. Use a short, path-safe name.');
  }
  return adapterName;
}

function trainingOutputNameReadinessState(formId, label) {
  const form = document.getElementById(formId);
  const input = form ? form.querySelector('input[name="output_name"]') : null;
  const outputName = input ? input.value.trim() : '';
  if (!outputName) {
    return { ready: false, message: `Enter a path-safe ${label} adapter name to enable submit.` };
  }
  if (!isPathSafeAdapterDirectoryName(outputName)) {
    return { ready: false, message: pathSafeAdapterDirectoryNameMessage() };
  }
  return { ready: true, message: 'Ready to submit with this path-safe adapter name.' };
}

function trainingPayloadReadinessState(textareaId, label, dataKind) {
  // A file (parsed items) or a server-side dataset reference held in
  // trainingData[dataKind] makes the form ready; the status chip narrates it.
  const held = (dataKind && typeof trainingData !== 'undefined') ? trainingData[dataKind] : null;
  if (held && ((held.items && held.items.length) || held.datasetName)) {
    // Mirror the visual status chip in this aria-live line so screen-reader
    // users tracking it hear the readiness change too.
    const what = held.datasetName
      ? `Dataset ${held.label || held.datasetName} ready — trains on the server's copy.`
      : `${held.items.length} item${held.items.length === 1 ? '' : 's'} ready from ${held.label || 'your file'}.`;
    return { ready: true, message: what + ' Train adapter is enabled.' };
  }
  const textarea = document.getElementById(textareaId);
  if (!textarea || !textarea.value.trim()) {
    return { ready: false, message: `Drop a file, pick a dataset, paste ${label} JSON, or try a sample to enable training.` };
  }
  return { ready: true, message: `${label} pasted — Train will validate before queuing.` };
}

function updateTrainingSubmitState(options) {
  const outputState = trainingOutputNameReadinessState(options.formId, options.outputLabel);
  const payloadState = trainingPayloadReadinessState(options.payloadId, options.payloadLabel, options.dataKind);
  const outputHelper = document.getElementById(options.outputStateId);
  const payloadHelper = document.getElementById(options.payloadStateId);
  if (outputHelper) outputHelper.textContent = outputState.message;
  if (payloadHelper) payloadHelper.textContent = payloadState.message;
  const form = document.getElementById(options.formId);
  const submitButton = form ? form.querySelector('button[type="submit"]') : null;
  if (submitButton) {
    submitButton.disabled = form?.dataset.trainingBusy === 'true' || !outputState.ready || !payloadState.ready;
  }
  return { ready: outputState.ready && payloadState.ready, outputState, payloadState };
}

// Cross-check epochs against how much data is actually loaded — many passes
// over a handful of examples just memorizes them. Advisory, never blocking.
function updateSftOverfitHint() {
  const hint = document.getElementById('sft-overfit-hint');
  if (!hint) return;
  const epochs = parseInt(document.getElementById('sft-epochs')?.value || '0', 10) || 0;
  const held = trainingData.sft;
  const n = held && held.items ? held.items.length : (held && held.count ? held.count : 0);
  if (n > 0 && n < 20 && epochs > 10) {
    hint.hidden = false;
    hint.textContent = `${epochs} passes over only ${n} example${n === 1 ? '' : 's'} will likely memorize them — 3 is usually plenty.`;
  } else {
    hint.hidden = true;
  }
}

function updateSftSubmitState() {
  updateSftOverfitHint();
  return updateTrainingSubmitState({
    formId: 'sft-form',
    outputStateId: 'sft-output-name-state',
    outputLabel: 'SFT output',
    payloadId: 'sft-examples',
    payloadStateId: 'sft-examples-state',
    payloadLabel: 'examples',
    dataKind: 'sft',
  });
}

function updateGrpoSubmitState() {
  return updateTrainingSubmitState({
    formId: 'grpo-form',
    outputStateId: 'grpo-output-name-state',
    outputLabel: 'GRPO output',
    payloadId: 'grpo-groups',
    payloadStateId: 'grpo-groups-state',
    payloadLabel: 'groups',
    dataKind: 'grpo',
  });
}

function updateSftOutputNameState() {
  return updateSftSubmitState().outputState;
}

function updateGrpoOutputNameState() {
  return updateGrpoSubmitState().outputState;
}

function parsePathSafeAdapterNameField(input, updateState) {
  const adapterName = parseAdapterNameField(input);
  if (!isPathSafeAdapterDirectoryName(adapterName)) {
    if (typeof updateState === 'function') updateState();
    input.focus();
    throw new Error(pathSafeAdapterDirectoryNameMessage());
  }
  return adapterName;
}

function setTrainingSubmitBusy(form, busy, pendingLabel) {
  const submitButton = form.querySelector('button[type="submit"]');
  if (!submitButton) return;
  if (!submitButton.dataset.originalLabel) {
    submitButton.dataset.originalLabel = submitButton.textContent;
  }
  form.dataset.trainingBusy = busy ? 'true' : 'false';
  submitButton.disabled = busy;
  submitButton.textContent = busy ? pendingLabel : submitButton.dataset.originalLabel;
  if (!busy) {
    if (form.id === 'sft-form') updateSftSubmitState();
    if (form.id === 'grpo-form') updateGrpoSubmitState();
  }
}

// --- SFT Form ---
document.getElementById('sft-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    const outputName = parsePathSafeAdapterNameField(form.output_name, updateSftOutputNameState);
    const learningRate = parseOptionalFiniteNumberField(form.learning_rate.value, 'SFT learning rate');
    const epochs = parsePositiveIntegerField(form.epochs.value, 'SFT epochs');
    const rank = parsePositiveIntegerField(form.rank.value, 'SFT LoRA rank');
    const config = {
      output_name: outputName,
      auto_load: form.auto_load.checked,
      epochs,
      lora_rank: rank,
      // Paired explicitly: the server's default alpha (32) over the form's
      // default rank (8) trips the trainer's alpha/rank safety gate.
      lora_alpha: loraAlphaFor(rank),
      optimizer: readTrainingOptimizer('sft'),
    };
    // Blank lr is omitted so the server resolves the per-optimizer default.
    if (learningRate !== null) config.learning_rate = learningRate;
    const held = trainingData.sft;
    let body;
    if (held && held.datasetName) {
      // Server-side dataset reference: the server reads its own copy — no rows
      // travel in the request and nothing is truncated.
      body = { dataset: held.datasetName, config };
    } else {
      let examples;
      if (held && held.items && held.items.length) {
        examples = held.items;
      } else {
        // Paste path accepts a JSON array OR JSONL — exactly what the help says.
        examples = parseTrainingText(form.examples.value);
        if (!examples.length) throw new Error('SFT examples cannot be empty. Drop a file, pick a dataset, paste JSON, or try a sample.');
      }
      validateSftExamples(examples);
      body = { examples, config };
    }
    const postEval = provePostEval('sft');
    if (postEval) body.post_eval = postEval;
    setTrainingSubmitBusy(form, true, 'Submitting SFT…');
    const res = await api('/v1/train/sft', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(res.message || 'SFT job submitted');
    // Switch to queue tab
    document.querySelector('[data-tab="queue"]').click();
    pollTraining();
  } catch (e) { toast(e.message, 'err'); }
  finally { setTrainingSubmitBusy(form, false, 'Submitting SFT…'); }
});

// --- GRPO Form ---
document.getElementById('grpo-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    const outputName = parsePathSafeAdapterNameField(form.output_name, updateGrpoOutputNameState);
    const learningRate = parseOptionalFiniteNumberField(form.learning_rate.value, 'GRPO learning rate');
    const klCoeff = parseFiniteNumberField(form.kl_coeff.value, 'GRPO KL coefficient');
    const rank = parsePositiveIntegerField(form.rank.value, 'GRPO LoRA rank');
    const config = {
      output_name: outputName,
      auto_load: form.auto_load.checked,
      kl_coeff: klCoeff,
      lora_rank: rank,
      // Paired explicitly: the server's default alpha (32) over the form's
      // default rank (8) trips the trainer's alpha/rank safety gate.
      lora_alpha: loraAlphaFor(rank),
      optimizer: readTrainingOptimizer('grpo'),
    };
    // Blank lr is omitted so the server resolves the per-optimizer default.
    if (learningRate !== null) config.learning_rate = learningRate;
    const held = trainingData.grpo;
    let body;
    if (held && held.datasetName) {
      body = { dataset: held.datasetName, config };
    } else {
      let groups;
      if (held && held.items && held.items.length) {
        groups = held.items;
      } else {
        groups = parseTrainingText(form.groups.value);
        if (!groups.length) throw new Error('GRPO groups cannot be empty. Drop a file, pick a dataset, paste JSON, or try a sample.');
      }
      validateGrpoGroups(groups);
      body = { groups, config };
    }
    const postEval = provePostEval('grpo');
    if (postEval) body.post_eval = postEval;
    setTrainingSubmitBusy(form, true, 'Submitting GRPO…');
    const res = await api('/v1/train/grpo', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(res.message || 'GRPO job submitted');
    document.querySelector('[data-tab="queue"]').click();
    pollTraining();
  } catch (e) { toast(e.message, 'err'); }
  finally { setTrainingSubmitBusy(form, false, 'Submitting GRPO…'); }
});

// --- Chat ---
const chatMessages = [];
let chatAbort = null;
let chatGenerating = false;
let servedModelId = null;
// True once the SERVER reported a model id. Until then the Connect panel and
// Playground run on the fallback id, and every successful health poll retries
// /v1/models (cold start: the endpoint 503s or lists nothing while weights
// load). Once resolved, the flag short-circuits — the retry stops for good.
let servedModelIdResolved = false;

async function loadServedModelId() {
  if (servedModelIdResolved) return;
  try {
    const res = await fetch('/v1/models');
    if (!res.ok) return;
    const data = await res.json();
    const id = data?.data?.[0]?.id;
    if (!id) return;
    servedModelId = id;
    servedModelIdResolved = true;
    // Upgrade the copyable snippets / model-id field that rendered with the
    // fallback while weights were still loading.
    applyServedModelId(id);
  } catch {}
}

/* ---------------------------------------------------------------------
   Playground settings persistence

   Sampling controls + system prompt round-trip through localStorage so
   reloading doesn't reset every knob. Conversation history is *not*
   persisted by default — that would surprise people running quick
   tests against different adapters. Use the "Restore last" affordance
   in the empty state to bring back the previous session.
   --------------------------------------------------------------------- */
const PLAYGROUND_SETTINGS_KEY = 'kiln.playground.settings.v1';
const PLAYGROUND_HISTORY_KEY  = 'kiln.playground.history.v1';

function readPlaygroundSettings() {
  try { return JSON.parse(localStorage.getItem(PLAYGROUND_SETTINGS_KEY)) || {}; }
  catch { return {}; }
}

function writePlaygroundSettings(settings) {
  try { localStorage.setItem(PLAYGROUND_SETTINGS_KEY, JSON.stringify(settings)); }
  catch { /* storage full / disabled / private mode */ }
}

function capturePlaygroundSettings() {
  const get = (id) => document.getElementById(id);
  return {
    temperature:        get('chat-temp')?.value ?? '1.0',
    maxTokens:          get('chat-max-tokens')?.value ?? '16384',
    enableThinking:     !!get('chat-enable-thinking')?.checked,
    thinkingBudgetMode: get('chat-thinking-budget-mode')?.value ?? 'server',
    thinkingBudgetTokens: get('chat-thinking-budget-tokens')?.value ?? '',
    thinkingBudgetSeconds: get('chat-thinking-budget-seconds')?.value ?? '',
    preset:             get('chat-preset')?.value ?? 'qwen3-thinking-general',
    topP:               get('chat-top-p')?.value ?? '',
    topK:               get('chat-top-k')?.value ?? '',
    minP:               get('chat-min-p')?.value ?? '',
    presencePenalty:    get('chat-presence-penalty')?.value ?? '',
    frequencyPenalty:   get('chat-frequency-penalty')?.value ?? '',
    repetitionPenalty:  get('chat-repetition-penalty')?.value ?? '',
    seed:               get('chat-seed')?.value ?? '',
    stop:               get('chat-stop-sequences')?.value ?? '',
    system:             get('chat-system')?.value ?? '',
    advancedOpen:       !get('chat-advanced')?.hidden,
    compareMode:        !!document.getElementById('chat-compare-toggle')?.checked,
  };
}

function applyPlaygroundSettings(settings) {
  if (!settings || typeof settings !== 'object') return;
  const set = (id, v) => { const el = document.getElementById(id); if (el != null && v != null) el.value = v; };
  set('chat-temp',                settings.temperature);
  // One-shot migration: the old default was '1024', which is too low
  // for thinking-capable models (the reasoning block alone routinely
  // exceeds it and the answer arrives "truncated"). Anyone with
  // '1024' persisted is almost certainly riding the old default
  // rather than having explicitly chosen it, so upgrade to the new
  // 16384 default. Users who really want 1024 can re-set it.
  if (settings.maxTokens && settings.maxTokens !== '1024') {
    set('chat-max-tokens', settings.maxTokens);
  }
  const budgetModes = new Set(['server', 'unlimited', 'custom']);
  set('chat-thinking-budget-mode', budgetModes.has(settings.thinkingBudgetMode) ? settings.thinkingBudgetMode : 'server');
  set('chat-thinking-budget-tokens', settings.thinkingBudgetTokens);
  set('chat-thinking-budget-seconds', settings.thinkingBudgetSeconds);
  set('chat-preset',              settings.preset);
  set('chat-top-p',               settings.topP);
  set('chat-top-k',               settings.topK);
  set('chat-min-p',               settings.minP);
  set('chat-presence-penalty',    settings.presencePenalty);
  set('chat-frequency-penalty',   settings.frequencyPenalty);
  set('chat-repetition-penalty',  settings.repetitionPenalty);
  set('chat-seed',                settings.seed);
  set('chat-stop-sequences',      settings.stop);
  set('chat-system',              settings.system);
  const thinking = document.getElementById('chat-enable-thinking');
  if (thinking && typeof settings.enableThinking === 'boolean') thinking.checked = settings.enableThinking;
  syncThinkingBudgetControls();
  const adv = document.getElementById('chat-advanced');
  const advBtn = document.getElementById('chat-toggle-advanced');
  if (adv && advBtn && settings.advancedOpen) {
    adv.hidden = false;
    advBtn.setAttribute('aria-expanded', 'true');
  }
}

/// Apply a Qwen3.5 preset by filling in every sampling input + the
/// thinking toggle. Mirrors the SamplingParams::qwen3_* helpers in
/// kiln-core so the UI shows the same numbers the server would pick if
/// the client sent `"sampling_preset": "..."`.
const QWEN3_PRESETS = {
  'qwen3-thinking-general': {
    temperature: '1.0', topP: '0.95', topK: '20', minP: '0.0',
    presencePenalty: '1.5', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: true,
  },
  'qwen3-thinking-coding': {
    temperature: '0.6', topP: '0.95', topK: '20', minP: '0.0',
    presencePenalty: '0.0', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: true,
  },
  'qwen3-non-thinking-general': {
    temperature: '0.7', topP: '0.8', topK: '20', minP: '0.0',
    presencePenalty: '1.5', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: false,
  },
  'qwen3-non-thinking-reasoning': {
    temperature: '1.0', topP: '0.95', topK: '20', minP: '0.0',
    presencePenalty: '1.5', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: false,
  },
  'greedy': {
    temperature: '0.0', topP: '1.0', topK: '0', minP: '0.0',
    presencePenalty: '0.0', frequencyPenalty: '0.0', repetitionPenalty: '1.0',
    enableThinking: true,
  },
};

function applyChatPreset(name) {
  if (name === 'custom') return;
  const preset = QWEN3_PRESETS[name];
  if (!preset) return;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  set('chat-temp',               preset.temperature);
  set('chat-top-p',              preset.topP);
  set('chat-top-k',              preset.topK);
  set('chat-min-p',              preset.minP);
  set('chat-presence-penalty',   preset.presencePenalty);
  set('chat-frequency-penalty',  preset.frequencyPenalty);
  set('chat-repetition-penalty', preset.repetitionPenalty);
  const thinking = document.getElementById('chat-enable-thinking');
  if (thinking) thinking.checked = preset.enableThinking;
  syncThinkingBudgetControls();
  persistPlaygroundSettingsSoon();
}

function persistPlaygroundSettingsSoon() {
  if (persistPlaygroundSettingsSoon._h) clearTimeout(persistPlaygroundSettingsSoon._h);
  persistPlaygroundSettingsSoon._h = setTimeout(() => {
    writePlaygroundSettings(capturePlaygroundSettings());
  }, 200);
}

function parseChatStopSequences(raw) {
  if (!raw) return undefined;
  const parts = String(raw).split(',').map(s => s.trim()).filter(Boolean);
  return parts.length ? parts : undefined;
}

function parseOptionalPositiveInt(raw) {
  if (raw == null || raw === '') return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) return undefined;
  return Math.floor(n);
}

function parseOptionalFloat(raw, { min, max } = {}) {
  if (raw == null || raw === '') return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n)) return undefined;
  if (min != null && n < min) return undefined;
  if (max != null && n > max) return undefined;
  return n;
}

function thinkingBudgetError(message, fieldId) {
  const error = new Error(message);
  error.fieldId = fieldId;
  return error;
}

function readThinkingBudgetRequest() {
  const thinkingEnabled = document.getElementById('chat-enable-thinking')?.checked !== false;
  const mode = document.getElementById('chat-thinking-budget-mode')?.value || 'server';
  if (!thinkingEnabled || mode === 'server') return { mode: 'server' };
  if (mode === 'unlimited') return { mode, tokens: null, ms: null };

  const tokensInput = document.getElementById('chat-thinking-budget-tokens');
  const secondsInput = document.getElementById('chat-thinking-budget-seconds');
  const tokensRaw = (tokensInput?.value || '').trim();
  const secondsRaw = (secondsInput?.value || '').trim();
  if (!tokensRaw && !secondsRaw) {
    throw thinkingBudgetError(
      'Set a thinking token limit, a time limit, or choose Unlimited.',
      'chat-thinking-budget-tokens',
    );
  }

  let tokens = null;
  if (tokensRaw) {
    const parsed = Number(tokensRaw);
    if (!/^\d+$/.test(tokensRaw) || !Number.isSafeInteger(parsed) || parsed > 131072) {
      throw thinkingBudgetError(
        'Thinking tokens must be a whole number from 0 to 131072.',
        'chat-thinking-budget-tokens',
      );
    }
    tokens = parsed;
  }

  let ms = null;
  if (secondsRaw) {
    const decimalSeconds = /^(?:\d+|\d*\.\d{1,3})$/;
    const seconds = Number(secondsRaw);
    if (!decimalSeconds.test(secondsRaw) || !Number.isFinite(seconds) || seconds > 86400) {
      throw thinkingBudgetError(
        'Thinking seconds must be between 0 and 86400 with at most three decimal places.',
        'chat-thinking-budget-seconds',
      );
    }
    ms = Math.round(seconds * 1000);
  }
  return { mode, tokens, ms };
}

function readThinkingBudgetRequestOrNotify() {
  try {
    return readThinkingBudgetRequest();
  } catch (error) {
    const field = error?.fieldId && document.getElementById(error.fieldId);
    if (field) field.focus();
    toast(error?.message || 'Invalid thinking budget.', 'err');
    return null;
  }
}

function openChatAdvancedControls() {
  const panel = document.getElementById('chat-advanced');
  const button = document.getElementById('chat-toggle-advanced');
  if (!panel || !button || !panel.hidden) return;
  panel.hidden = false;
  button.setAttribute('aria-expanded', 'true');
}

function syncThinkingBudgetControls({ revealCustom = false } = {}) {
  const enabled = document.getElementById('chat-enable-thinking')?.checked !== false;
  const modeInput = document.getElementById('chat-thinking-budget-mode');
  const mode = modeInput?.value || 'server';
  const custom = document.getElementById('chat-thinking-budget-custom');
  const tokens = document.getElementById('chat-thinking-budget-tokens');
  const seconds = document.getElementById('chat-thinking-budget-seconds');
  const customMode = mode === 'custom';

  if (modeInput) modeInput.disabled = !enabled;
  if (custom) {
    custom.hidden = !customMode;
    custom.classList.toggle('is-disabled', !enabled);
    custom.setAttribute('aria-disabled', String(!enabled));
  }
  if (tokens) tokens.disabled = !enabled || !customMode;
  if (seconds) seconds.disabled = !enabled || !customMode;
  if (enabled && customMode && revealCustom) openChatAdvancedControls();
}

function buildChatRequestBody({ messages, temperature, thinkingBudget }) {
  const body = {
    messages,
    stream: true,
    temperature,
  };
  const maxTokens = parseOptionalPositiveInt(document.getElementById('chat-max-tokens')?.value);
  body.max_tokens = maxTokens || 16384;

  const topP = parseOptionalFloat(document.getElementById('chat-top-p')?.value, { min: 0, max: 1 });
  if (topP !== undefined) body.top_p = topP;
  const topK = parseOptionalPositiveInt(document.getElementById('chat-top-k')?.value);
  if (topK !== undefined) body.top_k = topK; // 0 disables; still send so server doesn't fall back to its default of 20
  const minP = parseOptionalFloat(document.getElementById('chat-min-p')?.value, { min: 0, max: 1 });
  if (minP !== undefined) body.min_p = minP;
  const presencePenalty = parseOptionalFloat(
    document.getElementById('chat-presence-penalty')?.value,
    { min: -2, max: 2 },
  );
  if (presencePenalty !== undefined) body.presence_penalty = presencePenalty;
  const frequencyPenalty = parseOptionalFloat(
    document.getElementById('chat-frequency-penalty')?.value,
    { min: -2, max: 2 },
  );
  if (frequencyPenalty !== undefined) body.frequency_penalty = frequencyPenalty;
  const repetitionPenalty = parseOptionalFloat(
    document.getElementById('chat-repetition-penalty')?.value,
    { min: 0, max: 4 },
  );
  if (repetitionPenalty !== undefined) body.repetition_penalty = repetitionPenalty;
  const seed = parseOptionalPositiveInt(document.getElementById('chat-seed')?.value);
  if (seed !== undefined) body.seed = seed;
  const stop = parseChatStopSequences(document.getElementById('chat-stop-sequences')?.value);
  if (stop) body.stop = stop;

  if (thinkingBudget?.mode && thinkingBudget.mode !== 'server') {
    body.thinking_budget_tokens = thinkingBudget.tokens;
    body.thinking_budget_ms = thinkingBudget.ms;
  }

  const enableThinking = document.getElementById('chat-enable-thinking');
  if (enableThinking && !enableThinking.checked) {
    body.chat_template_kwargs = { enable_thinking: false };
  }
  return body;
}

function getSystemPromptMessage() {
  const text = (document.getElementById('chat-system')?.value || '').trim();
  return text ? { role: 'system', content: text } : null;
}

function serializableChatMessages() {
  // Strip the volatile streaming-state fields. We persist role +
  // content + reasoning only; on restore the message is "frozen"
  // (non-pending, no timing) but its text is preserved.
  return chatMessages
    .filter(m => m.role !== 'assistant' || (m.content && !m.error))
    .map(m => ({
      role: m.role,
      content: m.content || '',
      reasoning: m.reasoning || '',
      adapter: m.adapter || null,
      temperature: m.temperature ?? null,
      thinkingBudget: m.thinkingBudget || null,
    }));
}

function persistChatHistory() {
  try {
    const slim = serializableChatMessages();
    if (!slim.length) {
      localStorage.removeItem(PLAYGROUND_HISTORY_KEY);
    } else {
      localStorage.setItem(PLAYGROUND_HISTORY_KEY, JSON.stringify({ ts: Date.now(), messages: slim }));
    }
  } catch { /* ignore quota / disabled */ }
}

function readPersistedChatHistory() {
  try { return JSON.parse(localStorage.getItem(PLAYGROUND_HISTORY_KEY)) || null; }
  catch { return null; }
}

function restorePlaygroundHistoryBanner() {
  const stash = readPersistedChatHistory();
  if (!stash || !Array.isArray(stash.messages) || !stash.messages.length) return;
  if (chatMessages.length) return;  // already populated (HMR / re-init)
  const ageMin = Math.max(0, Math.round((Date.now() - (stash.ts || 0)) / 60000));
  const out = document.getElementById('chat-output');
  if (!out) return;
  const banner = document.createElement('div');
  banner.className = 'restore-banner';
  banner.innerHTML = `
    <div>
      <strong>Restore previous chat?</strong>
      <span style="color:var(--text-muted);"> — ${stash.messages.length} message${stash.messages.length === 1 ? '' : 's'}${ageMin ? `, ${ageMin} min ago` : ''}.</span>
    </div>
    <div style="display:flex; gap:6px;">
      <button class="btn btn-sm btn-primary" type="button" data-restore="yes">Restore</button>
      <button class="btn btn-sm" type="button" data-restore="no">Discard</button>
    </div>`;
  out.parentNode.insertBefore(banner, out);
  banner.addEventListener('click', (ev) => {
    const which = ev.target?.dataset?.restore;
    if (!which) return;
    if (which === 'yes') {
      for (const m of stash.messages) {
        chatMessages.push({
          _id: newChatMsgId(),
          role: m.role,
          content: m.content || '',
          reasoning: m.reasoning || '',
          pending: false,
          thinkOpen: false,
          adapter: m.adapter || null,
          temperature: m.temperature ?? null,
          thinkingBudget: m.thinkingBudget || null,
        });
      }
      renderChat();
    } else {
      try { localStorage.removeItem(PLAYGROUND_HISTORY_KEY); } catch {}
    }
    banner.remove();
  });
}

/* ---------------------------------------------------------------------
   Tiny safe-by-construction markdown renderer

   Lives inside the playground because we don't want to ship marked.js
   or any pin-anything for the dashboard. Handles only what assistant
   completions actually emit:

     - Fenced code blocks (``` and ~~~) with optional language tag.
     - Inline `code`.
     - **bold**, *italic*, ~~strike~~.
     - ATX headers (#…######).
     - Unordered / ordered lists.
     - Block quotes (>).
     - Horizontal rules (---).
     - Inline [text](url) links — http/https/relative only.

   Everything else falls through as escaped text, so a model that emits
   raw HTML can't inject anything dangerous. We HTML-escape on the way
   in and only re-introduce tags from a fixed, restricted set.
   --------------------------------------------------------------------- */
function _mdInline(text) {
  // The caller has already HTML-escaped `text`. We now re-introduce a
  // small set of inline tags; each placeholder we emit uses entities so
  // none can be mistaken for re-entrant markdown by a later pass.

  // Inline code: backticked spans. Greedy on inner content but stops at
  // the matching backtick run length (e.g. ``foo`bar`` works).
  text = text.replace(/(`+)([^`]+?)\1/g, (_, ticks, body) =>
    `<code>${body.replace(/\n/g, ' ')}</code>`);

  // Links: [text](url). Allow http(s) and relative paths; reject `javascript:`
  // and other schemes outright so a model can't smuggle XSS through here.
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, href) => {
    const ok = /^(https?:\/\/|\/|\.{1,2}\/|#)/.test(href);
    if (!ok) return m;
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });

  // **bold** and __bold__
  text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');

  // *italic* and _italic_ — narrower than bold so we don't eat **bold**.
  text = text.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:!?)\]]|$)/g, '$1<em>$2</em>');
  text = text.replace(/(^|[\s(])_([^_\n]+)_(?=[\s.,;:!?)\]]|$)/g, '$1<em>$2</em>');

  // ~~strike~~
  text = text.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');
  return text;
}

function renderMarkdown(raw) {
  if (!raw) return '';
  // 1) Escape, but extract fenced code blocks first so their contents
  //    don't get interpreted as inline markdown.
  const fenced = [];
  const fenceRe = /```([a-zA-Z0-9_+\-.]*)\n([\s\S]*?)```|~~~([a-zA-Z0-9_+\-.]*)\n([\s\S]*?)~~~/g;
  const withPlaceholders = raw.replace(fenceRe, (_m, lang1, body1, lang2, body2) => {
    const lang = (lang1 || lang2 || '').trim();
    const body = body1 != null ? body1 : body2;
    const idx = fenced.length;
    fenced.push({ lang, body });
    return ` FENCE${idx} `;
  });
  const escaped = escapeHtml(withPlaceholders);

  // 2) Split into block-level pieces by blank-line runs. Each block is
  //    classified once (header / list / quote / hr / paragraph).
  const blocks = escaped.split(/\n{2,}/);
  const html = blocks.map(block => {
    if (!block.trim()) return '';

    // Fenced-code placeholder: emit verbatim, no inline processing.
    const fenceMatch = block.match(/^ FENCE(\d+) $/);
    if (fenceMatch) {
      const { lang, body } = fenced[Number(fenceMatch[1])];
      const escBody = escapeHtml(body.replace(/\n$/, ''));
      const langAttr = lang ? ` data-lang="${escapeHtml(lang)}"` : '';
      return `<pre class="md-code"${langAttr}><code>${escBody}</code></pre>`;
    }

    // Horizontal rule.
    if (/^---+$/.test(block.trim())) return `<hr>`;

    // ATX header (# …)
    const h = block.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      return `<h${level}>${_mdInline(h[2])}</h${level}>`;
    }

    // Block quote — every line starts with `>`.
    if (/^>/.test(block) && block.split('\n').every(l => /^>\s?/.test(l) || !l.trim())) {
      const inner = block.split('\n').map(l => l.replace(/^>\s?/, '')).join('\n');
      return `<blockquote>${_mdInline(inner).replace(/\n/g, '<br>')}</blockquote>`;
    }

    // Lists: an unordered if every non-empty line matches `- |* `, or
    // ordered if every non-empty line matches `\d+\.`. Mixed → paragraph.
    const lines = block.split('\n');
    if (lines.every(l => !l.trim() || /^\s*[-*]\s+/.test(l))) {
      const items = lines.filter(l => l.trim()).map(l => l.replace(/^\s*[-*]\s+/, ''));
      return `<ul>${items.map(i => `<li>${_mdInline(i)}</li>`).join('')}</ul>`;
    }
    if (lines.every(l => !l.trim() || /^\s*\d+\.\s+/.test(l))) {
      const items = lines.filter(l => l.trim()).map(l => l.replace(/^\s*\d+\.\s+/, ''));
      return `<ol>${items.map(i => `<li>${_mdInline(i)}</li>`).join('')}</ol>`;
    }

    // Plain paragraph. Soft line breaks become <br>.
    return `<p>${_mdInline(block).replace(/\n/g, '<br>')}</p>`;
  }).join('');
  return html;
}

function formatChatDuration(ms) {
  if (ms == null || !Number.isFinite(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
}

function approximateTokenCount(text) {
  if (!text) return 0;
  // Cheap token estimate: ~4 chars/token for English. Good enough for
  // a "tokens/sec" readout that doesn't require a wire-side counter.
  return Math.max(1, Math.round(text.length / 4));
}

function chatTokensPerSec(message) {
  if (!message || !message.durationMs || message.durationMs <= 0) return null;
  const tokens = approximateTokenCount(message.reasoning || '') + approximateTokenCount(message.content || '');
  if (!tokens) return null;
  return (tokens * 1000) / message.durationMs;
}

function thinkingBudgetSummary(outcome) {
  if (!outcome?.applied) return '';
  if (outcome.triggered) {
    const trigger = outcome.trigger === 'tokens'
      ? 'token cap'
      : outcome.trigger === 'time'
        ? 'time cap'
        : 'completion limit';
    return trigger;
  }
  return outcome.closed ? 'natural close' : 'unclosed';
}

function renderAssistantBubble(m) {
  const parts = [];
  const hasReasoning = !!(m.reasoning && m.reasoning.length);
  const hasContent   = !!(m.content   && m.content.length);

  if (hasReasoning) {
    // While content has not started arriving, keep the thinking block
    // open so the user can watch the chain-of-thought stream live.
    // Once content arrives, collapse by default but let the user
    // pin-open if they had it open already.
    const live = m.pending && !hasContent;
    const open = live || m.thinkOpen;
    const summary = (() => {
      if (live) {
        const elapsed = m.thinkStartMs ? formatChatDuration(performance.now() - m.thinkStartMs) : '';
        return `<span class="think-label">Thinking</span>${elapsed ? `<span class="think-meta">· ${escapeHtml(elapsed)}</span>` : ''}`;
      }
      const dur = (m.thinkStartMs && m.thinkEndMs) ? formatChatDuration(m.thinkEndMs - m.thinkStartMs) : null;
      const outcome = thinkingBudgetSummary(m.thinkingBudget);
      return `<span class="think-label">Thought</span>${dur ? `<span class="think-meta">· for ${escapeHtml(dur)}</span>` : ''}${outcome ? `<span class="think-meta">· ${escapeHtml(outcome)}</span>` : ''}`;
    })();
    parts.push(`
      <details class="think-block${live ? ' live' : ''}"${open ? ' open' : ''} data-think-toggle="${escapeHtml(m._id)}">
        <summary>${summary}</summary>
        <div class="think-body">${escapeHtml(m.reasoning)}</div>
      </details>
    `);
  }

  // Main answer body — pending without any content shows a "Generating…"
  // placeholder unless the reasoning block is already live (in which
  // case the chain-of-thought is the visible activity). Finished
  // answers go through the lightweight markdown renderer; still-
  // streaming output stays in a plain <pre> so partial fences and
  // mid-list states don't render as flicker.
  let body = '';
  if (m.error) {
    body = `<div class="err-block">${escapeHtml(m.error)}</div>`;
  } else if (hasContent && m.pending) {
    body = `<pre>${escapeHtml(m.content)}</pre>`;
  } else if (hasContent) {
    body = `<div class="md-body">${renderMarkdown(m.content)}</div>`;
  } else if (m.pending && !hasReasoning) {
    // After ~5 s with no token, swap the bare "Generating…" placeholder
    // for a hint about prompt-processing latency + an explicit Stop
    // reminder. Cheap UX guard against the most common "is it stuck?"
    // moment: long prompts or a cold model where the model is still
    // doing prefill before the first token streams.
    const waited = m.startMs != null ? (performance.now() - m.startMs) : 0;
    if (waited > 5000) {
      body = `<pre style="color:var(--text-muted);">Waiting for first token (${escapeHtml(formatChatDuration(waited))}) — long prompts or a cold model can take several seconds. Use the Stop button if this is hung.</pre>`;
    } else {
      body = `<pre>Generating…</pre>`;
    }
  } else if (m.pending && hasReasoning && !hasContent) {
    body = `<pre style="color:var(--text-muted);font-style:italic;">Drafting answer…</pre>`;
  } else if (!hasContent && !hasReasoning) {
    body = `<pre style="color:var(--text-muted);">(empty response)</pre>`;
  }
  parts.push(body);

  // Per-turn footer with timing + actions. Hidden while we have nothing
  // useful to display yet (no first-token timing and no content).
  const showFoot = !!(m.ttftMs || m.durationMs || (!m.pending && (hasContent || m.error)));
  if (showFoot) {
    const stats = [];
    const tag = (m.adapter || m.adapter === null) && m.temperature != null
      ? `<span class="badge-tag" title="Adapter + sampling temperature for this turn">${escapeHtml(m.adapter || 'base')} · t=${escapeHtml(String(m.temperature))}</span>`
      : '';
    if (tag) stats.push(tag);
    if (m.ttftMs != null)     stats.push(`<span class="stat"><strong>TTFT</strong> ${escapeHtml(formatChatDuration(m.ttftMs))}</span>`);
    if (m.durationMs != null) stats.push(`<span class="stat"><strong>${m.pending ? 'Elapsed' : 'Total'}</strong> ${escapeHtml(formatChatDuration(m.durationMs))}</span>`);
    const tps = chatTokensPerSec(m);
    if (tps != null)          stats.push(`<span class="stat"><strong>~${tps.toFixed(tps >= 100 ? 0 : 1)}</strong> tok/s</span>`);
    const budgetOutcome = !m.pending && !hasReasoning ? thinkingBudgetSummary(m.thinkingBudget) : '';
    if (budgetOutcome)        stats.push(`<span class="stat"><strong>Thinking</strong> ${escapeHtml(budgetOutcome)}</span>`);
    if (!m.pending && m.finishReason && m.finishReason !== 'stop') {
      const kind = m.finishReason === 'length' ? 'truncated' : m.finishReason;
      const cls = m.finishReason === 'length' ? 'stat finish-warn' : 'stat';
      const title = m.finishReason === 'length'
        ? 'Response was cut off — increase Max tokens to let the model finish.'
        : `Generation ended with finish_reason=${m.finishReason}.`;
      stats.push(`<span class="${cls}" title="${escapeHtml(title)}">${icon('warning','icn-sm')} ${escapeHtml(kind)}</span>`);
    }
    stats.push(`<span class="spacer"></span>`);
    if (!m.pending && m.error) {
      stats.push(`<button class="turn-btn" type="button" data-chat-action="regenerate" data-chat-id="${escapeHtml(m._id)}" title="Retry this request"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-refresh"></use></svg> retry</button>`);
    } else if (!m.pending && hasContent) {
      stats.push(`<button class="turn-btn" type="button" data-chat-action="copy" data-chat-id="${escapeHtml(m._id)}" title="Copy assistant answer"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg> copy</button>`);
      stats.push(`<button class="turn-btn" type="button" data-chat-action="regenerate" data-chat-id="${escapeHtml(m._id)}" title="Regenerate this response"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-refresh"></use></svg> regenerate</button>`);
    } else if (m.pending) {
      stats.push(`<button class="turn-btn" type="button" data-chat-action="stop" title="Stop generation">■ stop</button>`);
    }
    parts.push(`<div class="turn-foot">${stats.join('')}</div>`);
  }

  return parts.join('');
}

/* ---------------------------------------------------------------------
   Auto-scroll behavior

   Snapping to the bottom on every chunk fights the user when they
   scroll up to read an earlier turn. Track "is the viewport pinned to
   the bottom?" right before each re-render and only snap if it was.
   --------------------------------------------------------------------- */
let chatStickToBottom = true;
function captureScrollAffinity() {
  const el = document.getElementById('chat-output');
  if (!el) return;
  // 12 px slack so the user doesn't have to be pixel-perfect.
  chatStickToBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) <= 12;
}
function restoreScrollAffinity() {
  const el = document.getElementById('chat-output');
  if (!el || !chatStickToBottom) return;
  el.scrollTop = el.scrollHeight;
}

function updateChatTurnCount() {
  const el = document.getElementById('chat-turn-count');
  const exportBtn = document.getElementById('chat-export');
  if (!el) return;
  const userTurns = chatMessages.filter(m => m.role === 'user').length;
  const finalAssistant = chatMessages.filter(m => m.role === 'assistant' && !m.pending && m.content).length;
  if (userTurns === 0) {
    el.hidden = true;
    if (exportBtn) exportBtn.disabled = true;
  } else {
    el.hidden = false;
    el.textContent = `${userTurns} turn${userTurns === 1 ? '' : 's'} · ${finalAssistant} reply${finalAssistant === 1 ? '' : 'ies'}`;
    if (exportBtn) exportBtn.disabled = finalAssistant === 0;
  }
}

/* In-place update of a single assistant bubble's contents — replaces
   only the children of the existing `[data-msg-id]` wrapper, so every
   *other* bubble's DOM is left untouched. Streaming chunks call this
   instead of the global `renderChat()` to avoid the relayout-and-
   re-animate storm that re-creating every <div> caused. Falls back to
   a full render when the bubble isn't in the DOM yet (first paint).

   Note: scroll affinity is *not* sampled here. A targeted innerHTML
   update doesn't reset scrollTop, so we'd actively fight the user's
   scroll position if we tried to re-pin. The full `renderChat()` —
   used on user-message push, clear, and turn end — still handles
   sticking to the bottom on layout-changing events. */
function patchAssistantBubble(m) {
  const wrapper = document.querySelector(`.chat-msg.assistant[data-msg-id="${cssEscape(m._id)}"]`);
  if (!wrapper) { renderChat(); return; }
  wrapper.className = `chat-msg assistant${m.pending ? ' pending' : ''}`;
  wrapper.innerHTML = `<div class="role">assistant</div>${renderAssistantBubble(m)}`;
  // If user was pinned to the bottom, follow the growing bubble.
  const out = document.getElementById('chat-output');
  if (out) {
    const slack = out.scrollHeight - out.scrollTop - out.clientHeight;
    if (slack <= 24) out.scrollTop = out.scrollHeight;
  }
  // Refresh footer-derived state without a global render.
  updateCopyChatResponseState();
}

// Tiny CSS.escape polyfill — we control the IDs (alphanumeric + `-`),
// but be defensive against future ID schemes that include CSS-special
// characters like `.` or `:`.
function cssEscape(s) {
  return String(s).replace(/[^a-zA-Z0-9_-]/g, c => '\\' + c);
}

function renderChat() {
  const el = document.getElementById('chat-output');
  if (chatMessages.length === 0) {
    el.innerHTML = `<div class="empty">
      <div style="font-weight:600;color:var(--text);margin-bottom:6px;">Send a message to test inference.</div>
      <div>Quick Inference sends a chat completion to the currently selected adapter, or the <strong>Base model</strong>, using the temperature above.</div>
      <div style="margin-top:var(--space-3);"><button type="button" class="btn btn-sm btn-primary" data-chat-example="Explain Kiln in one sentence.">Try an example prompt</button></div>
      <div style="margin-top:var(--space-3);color:var(--text-3);">Tip: toggle <strong>Compare</strong> (above) to race two adapters side-by-side on the same prompt — the fastest way to eyeball whether a freshly trained adapter actually answers better.</div>
      <div style="margin-top:var(--space-2);">If the server is still starting, check <a href="/health" target="_blank" rel="noopener noreferrer">/health</a> or the <a href="https://ericflo.github.io/kiln/troubleshooting.html" target="_blank" rel="noopener noreferrer">Troubleshooting guide</a>.</div>
    </div>`;
    el.querySelector('[data-chat-example]')?.addEventListener('click', (ev) => {
      const input = document.getElementById('chat-input');
      if (!input) return;
      input.value = ev.currentTarget.dataset.chatExample || '';
      if (typeof autoresizeChatInput === 'function') autoresizeChatInput();
      if (typeof updateChatSendState === 'function') updateChatSendState();
      input.focus();
    });
    updateCopyChatResponseState();
    return;
  }
  captureScrollAffinity();
  el.innerHTML = chatMessages.map(m => {
    if (m.role === 'assistant') {
      return `<div class="chat-msg assistant${m.pending ? ' pending' : ''}" data-msg-id="${escapeHtml(m._id)}">
        <div class="role">assistant</div>
        ${renderAssistantBubble(m)}
      </div>`;
    }
    if (m.role === 'user') {
      // Inline-edit affordance: pencil floats on hover; clicking
      // promotes the row to an editable textarea + save/cancel
      // controls. Save trims downstream messages and re-streams.
      return `<div class="chat-msg user${m._editing ? ' editing' : ''}">
        <div class="role">user</div>
        <pre>${escapeHtml(m.content)}</pre>
        <button class="user-edit-btn" type="button" data-chat-action="edit" data-chat-id="${escapeHtml(m._id)}" title="Edit and resend"><svg class="icn icn-sm"><use href="#i-pencil"></use></svg></button>
        <div class="user-edit-area">
          <textarea class="user-edit-input">${escapeHtml(m.content)}</textarea>
          <div class="user-edit-actions">
            <button class="btn btn-sm" type="button" data-chat-action="edit-cancel" data-chat-id="${escapeHtml(m._id)}">Cancel</button>
            <button class="btn btn-sm btn-primary" type="button" data-chat-action="edit-save" data-chat-id="${escapeHtml(m._id)}">Save & resend</button>
          </div>
        </div>
      </div>`;
    }
    if (m.role === 'system') {
      // System messages aren't typically pushed into chatMessages
      // (we synthesize from the system-prompt textarea), but render
      // defensively in case a future flow drops one in.
      return `<div class="chat-msg system">
        <div class="role">system</div>
        <pre>${escapeHtml(m.content)}</pre>
      </div>`;
    }
    return `<div class="chat-msg ${m.role}">
      <div class="role">${m.role}</div>
      <pre>${escapeHtml(m.content)}</pre>
    </div>`;
  }).join('');
  restoreScrollAffinity();
  updateCopyChatResponseState();
  updateChatTurnCount();
}

function getLatestAssistantResponseText() {
  for (let i = chatMessages.length - 1; i >= 0; i--) {
    const message = chatMessages[i];
    if (message.role === 'assistant' && message.content.trim()) {
      return message.content.trim();
    }
  }

  const output = document.getElementById('chat-output');
  const assistantMessages = output ? output.querySelectorAll('.chat-msg.assistant pre, .msg.assistant') : [];
  for (let i = assistantMessages.length - 1; i >= 0; i--) {
    const text = assistantMessages[i].textContent.trim();
    if (text && text !== 'Generating…') return text;
  }
  return '';
}

function updateCopyChatResponseState() {
  const button = document.getElementById('copy-chat-response');
  if (!button) return;
  button.disabled = !getLatestAssistantResponseText();
}
window.updateCopyChatResponseState = updateCopyChatResponseState;

function updateChatSendState() {
  const input = document.getElementById('chat-input');
  const send = document.getElementById('chat-send');
  if (!input || !send) return;
  send.disabled = chatGenerating || !input.value.trim();
}

function fallbackCopyText(text) {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  try {
    if (!document.execCommand('copy')) throw new Error('copy command failed');
    if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = text;
  } finally {
    textarea.remove();
  }
}

async function copyLatestAssistantResponse() {
  const text = getLatestAssistantResponseText();
  if (!text) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = text;
    } else {
      fallbackCopyText(text);
    }
    toast('Copied response');
  } catch (error) {
    try {
      fallbackCopyText(text);
      toast('Copied response');
    } catch {
      toast('Could not copy response. Select the answer text and copy it manually.', 'err');
    }
  }
}

function setChatGenerating(isGenerating) {
  chatGenerating = isGenerating;
  const send = document.getElementById('chat-send');
  const stop = document.getElementById('chat-stop');
  send.textContent = isGenerating ? 'Generating…' : 'Send';
  updateChatSendState();
  stop.hidden = !isGenerating;
  stop.disabled = !isGenerating;
}

function removeEmptyPendingAssistant() {
  const last = chatMessages[chatMessages.length - 1];
  if (last?.role === 'assistant' && last.pending && !last.content) {
    chatMessages.pop();
  } else if (last?.role === 'assistant') {
    last.pending = false;
  }
}

function formatQuickInferenceError(error) {
  const message = error?.message || String(error || 'Unknown error');
  return [
    '',
    'Quick Inference could not complete this request.',
    `Server error: ${message}`,
    '',
    'Next steps:',
    '1. If kiln serve just started, wait for model startup to finish and try again.',
    '2. Open /health to check whether the server is ready.',
    '3. Check the kiln serve logs for model path or GPU initialization errors.',
    '4. See Troubleshooting: https://ericflo.github.io/kiln/troubleshooting.html',
  ].join('\n');
}

let chatMsgIdCounter = 0;
function newChatMsgId() { return `m${++chatMsgIdCounter}-${Date.now().toString(36)}`; }

function makeAssistantPlaceholder() {
  return {
    _id: newChatMsgId(),
    role: 'assistant',
    content: '',
    reasoning: '',
    pending: true,
    startMs: performance.now(),
    firstTokenMs: null,
    firstContentTokenMs: null,
    lastTokenMs: null,
    thinkStartMs: null,
    thinkEndMs: null,
    thinkOpen: false,
    ttftMs: null,
    durationMs: null,
    error: null,
    aborted: false,
    thinkingBudget: null,
  };
}

async function sendChat() {
  if (chatAbort) return;
  // When compare mode is on, the dedicated A/B handler (wired further
  // down) owns the send. Skipping here prevents the single-side bubble
  // from rendering on top of the side-by-side compare panel.
  if (typeof chatCompareMode !== 'undefined' && chatCompareMode) return;
  const input = document.getElementById('chat-input');
  const tempInput = document.getElementById('chat-temp');
  const msg = input.value.trim();
  if (!msg) return;

  let temp;
  try {
    temp = parseQuickInferenceTemperature(tempInput);
  } catch (error) {
    tempInput.focus();
    toast(error.message, 'err');
    return;
  }
  const thinkingBudget = readThinkingBudgetRequestOrNotify();
  if (!thinkingBudget) return;

  input.value = '';
  autoresizeChatInput();
  updateChatSendState();

  chatMessages.push({ _id: newChatMsgId(), role: 'user', content: msg });
  chatMessages.push(makeAssistantPlaceholder());
  renderChat();
  await streamAssistantTurn(temp, thinkingBudget);
}

async function streamAssistantTurn(temp, thinkingBudget) {
  setChatGenerating(true);

  const adapter = document.getElementById('chat-adapter').value || undefined;
  const assistant = chatMessages[chatMessages.length - 1];

  // Snapshot the request shape on this assistant turn so the per-turn
  // footer + the persisted history can show *what* produced this
  // answer even after the user mutates the controls.
  assistant.adapter = adapter || null;
  assistant.temperature = temp;

  const convo = chatMessages
    .filter(m => m.role !== 'assistant' || (m.content && !m.error))
    .filter(m => m.content || m.role === 'system')
    .map(m => ({ role: m.role, content: m.content }));
  const sys = getSystemPromptMessage();
  const messages = sys ? [sys, ...convo] : convo;

  const body = buildChatRequestBody({ messages, temperature: temp, thinkingBudget });
  if (servedModelId) body.model = servedModelId;
  if (adapter) body.adapter = adapter;

  // Tick the per-turn footer so users see the elapsed counter advance
  // even before the first token lands. 500 ms cadence — fast enough
  // for a visible counter, slow enough to never feel jittery — and we
  // patch only the assistant bubble in place (not a full re-render)
  // so neighboring bubbles don't relayout.
  const tickHandle = setInterval(() => {
    if (!assistant.pending) return;
    assistant.durationMs = performance.now() - assistant.startMs;
    patchAssistantBubble(assistant);
  }, 500);

  try {
    const controller = new AbortController();
    chatAbort = controller;

    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Kiln-Client': 'dashboard' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    streamLoop:
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\n');
      buf = lines.pop();

      for (let line of lines) {
        // Strip CR for servers that emit `\r\n` line terminators (some
        // proxies, some SSE polyfills). Without this, `payload` ends in
        // `\r` and the `=== '[DONE]'` check never matches, so the loop
        // only exits when the server closes the socket — which can be
        // significantly after [DONE] in keep-alive setups.
        if (line.endsWith('\r')) line = line.slice(0, -1);
        if (!line.startsWith('data:')) continue;
        // The SSE spec allows `data:value` (no space) too; accept both.
        let payload = line.slice(5);
        if (payload.startsWith(' ')) payload = payload.slice(1);
        if (payload === '[DONE]') break streamLoop;
        try {
          const chunk = JSON.parse(payload);
          const choice = chunk.choices?.[0];
          const delta = choice?.delta;
          if (choice?.finish_reason) {
            assistant.finishReason = choice.finish_reason;
          }
          if (chunk.metadata?.thinking_budget?.applied) {
            assistant.thinkingBudget = chunk.metadata.thinking_budget;
          }
          if (!delta) continue;
          const now = performance.now();
          if (assistant.firstTokenMs == null) {
            assistant.firstTokenMs = now;
            assistant.ttftMs = now - assistant.startMs;
          }
          if (delta.reasoning_content) {
            if (assistant.thinkStartMs == null) assistant.thinkStartMs = now;
            assistant.reasoning += delta.reasoning_content;
            assistant.pending = true;
            assistant.lastTokenMs = now;
            assistant.durationMs = now - assistant.startMs;
            patchAssistantBubble(assistant);
          }
          if (delta.content) {
            if (assistant.thinkStartMs != null && assistant.thinkEndMs == null) {
              assistant.thinkEndMs = now;
            }
            if (assistant.firstContentTokenMs == null) assistant.firstContentTokenMs = now;
            assistant.content += delta.content;
            assistant.pending = true;
            assistant.lastTokenMs = now;
            assistant.durationMs = now - assistant.startMs;
            patchAssistantBubble(assistant);
          }
        } catch (parseErr) {
          // Don't swallow the *whole* stream when one chunk is
          // malformed. Log so we can see this in a user's devtools if
          // they report a stuck stream.
          console.warn('[playground] skipped malformed SSE chunk', parseErr, payload.slice(0, 120));
        }
      }
    }
    assistant.pending = false;
    assistant.durationMs = (assistant.lastTokenMs || performance.now()) - assistant.startMs;
    if (assistant.thinkStartMs != null && assistant.thinkEndMs == null && assistant.content) {
      assistant.thinkEndMs = assistant.lastTokenMs || performance.now();
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      assistant.aborted = true;
      assistant.pending = false;
      // Keep partial output if anything streamed; drop the empty
      // placeholder + its paired user message otherwise so the user
      // doesn't accrue empty turns on rapid stop-clicks. (The original
      // behavior popped only the placeholder, leaving an orphaned
      // user turn that re-sent the same prompt on the next regen.)
      if (!assistant.content && !assistant.reasoning) {
        chatMessages.pop();  // assistant placeholder
      } else {
        assistant.durationMs = (assistant.lastTokenMs || performance.now()) - assistant.startMs;
      }
    } else {
      assistant.pending = false;
      assistant.error = formatQuickInferenceError(e);
    }
  } finally {
    // Order matters here: clear the abort handle *and* the UI flag
    // before anything that could throw (renderChat → renderMarkdown →
    // arbitrary user content), so a render failure can't leave the
    // Send button stuck on "Generating…" forever.
    clearInterval(tickHandle);
    chatAbort = null;
    setChatGenerating(false);
    try { persistChatHistory(); } catch (e) { console.warn('[playground] persistChatHistory threw', e); }
    try { renderChat(); }        catch (e) { console.warn('[playground] renderChat threw', e); }
  }
}

function autoresizeChatInput() {
  const input = document.getElementById('chat-input');
  if (!input || input.tagName !== 'TEXTAREA') return;
  input.style.height = 'auto';
  const next = Math.min(input.scrollHeight, 180);
  input.style.height = next + 'px';
}

async function regenerateAssistantMessage(messageId) {
  if (chatAbort) return;
  const thinkingBudget = readThinkingBudgetRequestOrNotify();
  if (!thinkingBudget) return;
  // Find the assistant message and the chain of user/assistant turns
  // *before* it. We replace it in-place with a fresh placeholder and
  // re-stream, so the user's prior message and the conversation
  // upstream of it stay intact.
  const idx = chatMessages.findIndex(m => m._id === messageId);
  if (idx < 0 || chatMessages[idx].role !== 'assistant') return;
  // Drop the target assistant message and any messages after it; the
  // upstream context (everything before idx, ending in user) is what
  // we want to re-send.
  chatMessages.splice(idx);
  // Push a fresh placeholder and stream against the trimmed history.
  chatMessages.push(makeAssistantPlaceholder());
  renderChat();
  const tempInput = document.getElementById('chat-temp');
  let temp;
  try {
    temp = parseQuickInferenceTemperature(tempInput);
  } catch (error) {
    tempInput.focus();
    toast(error.message, 'err');
    chatMessages.pop();
    renderChat();
    return;
  }
  await streamAssistantTurn(temp, thinkingBudget);
}

function handleChatActionClick(event) {
  const btn = event.target.closest('[data-chat-action]');
  if (!btn) return;
  const action = btn.dataset.chatAction;
  if (action === 'stop') {
    if (chatAbort) chatAbort.abort();
    return;
  }
  const id = btn.dataset.chatId;
  if (action === 'copy') {
    const m = chatMessages.find(x => x._id === id);
    if (!m || !m.content) return;
    const writeText = navigator.clipboard?.writeText
      ? navigator.clipboard.writeText.bind(navigator.clipboard)
      : (text) => { fallbackCopyText(text); return Promise.resolve(); };
    writeText(m.content).then(() => {
      if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = m.content;
      toast('Copied response');
    }).catch(() => {
      try { fallbackCopyText(m.content); toast('Copied response'); }
      catch { toast('Could not copy response.', 'err'); }
    });
    return;
  }
  if (action === 'regenerate') {
    regenerateAssistantMessage(id);
    return;
  }
  if (action === 'edit') {
    const m = chatMessages.find(x => x._id === id);
    if (!m || m.role !== 'user') return;
    m._editing = true;
    renderChat();
    // After re-render, focus the new textarea so the user starts typing.
    const ta = document.querySelector(`.chat-msg.user.editing[data-edit-host="${id}"] .user-edit-input`)
      || document.querySelector('.chat-msg.user.editing .user-edit-input');
    if (ta) { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }
    return;
  }
  if (action === 'edit-cancel') {
    const m = chatMessages.find(x => x._id === id);
    if (!m) return;
    m._editing = false;
    renderChat();
    return;
  }
  if (action === 'edit-save') {
    const m = chatMessages.find(x => x._id === id);
    if (!m || m.role !== 'user') return;
    const host = btn.closest('.chat-msg.user');
    const ta = host?.querySelector('.user-edit-input');
    const next = (ta?.value || '').trim();
    if (!next) {
      toast('Message can not be empty.', 'err');
      return;
    }
    const thinkingBudget = readThinkingBudgetRequestOrNotify();
    if (!thinkingBudget) return;
    if (chatAbort) chatAbort.abort();
    const idx = chatMessages.indexOf(m);
    m.content = next;
    m._editing = false;
    // Drop everything after this user turn — we're re-running from here.
    chatMessages.splice(idx + 1);
    chatMessages.push(makeAssistantPlaceholder());
    renderChat();
    const tempInput = document.getElementById('chat-temp');
    let temp;
    try {
      temp = parseQuickInferenceTemperature(tempInput);
    } catch (error) {
      tempInput.focus();
      toast(error.message, 'err');
      chatMessages.pop();
      renderChat();
      return;
    }
    streamAssistantTurn(temp, thinkingBudget);
    return;
  }
}

function handleThinkToggle(event) {
  // Persist whether the user has the thinking panel pinned-open so the
  // next renderChat() (every streaming chunk) doesn't snap it shut.
  const details = event.target.closest('details.think-block');
  if (!details) return;
  const id = details.dataset.thinkToggle;
  if (!id) return;
  const m = chatMessages.find(x => x._id === id);
  if (m) m.thinkOpen = details.open;
}

document.getElementById('chat-send').addEventListener('click', sendChat);
document.getElementById('chat-input').addEventListener('input', () => {
  autoresizeChatInput();
  updateChatSendState();
});
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
document.querySelectorAll('[data-chat-starter-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    const input = document.getElementById('chat-input');
    input.value = button.dataset.chatStarterPrompt || '';
    autoresizeChatInput();
    updateChatSendState();
    input.focus();
  });
});
document.getElementById('chat-stop').addEventListener('click', () => {
  if (chatAbort) chatAbort.abort();
  updateChatSendState();
});
// Esc aborts the active streaming generation when no modal owns the key.
// Standard chat-app keyboard shortcut — saves a mouse trip to the Stop
// button mid-stream. Any open modal claims Escape via the shared modal
// manager, so this only fires when the modal stack is empty.
document.addEventListener('keydown', (ev) => {
  if (ev.key !== 'Escape') return;
  // Only intervene when chat is actually streaming.
  if (!chatAbort && !chatCompareAbort) return;
  // Don't fight any open modal — Escape there closes the top of the stack.
  if (modalStack.length) return;
  if (chatAbort) { chatAbort.abort(); }
  if (chatCompareAbort) { chatCompareAbort.abort(); }
  ev.preventDefault();
});
document.getElementById('chat-clear').addEventListener('click', () => {
  if (chatAbort) { chatAbort.abort(); chatAbort = null; }
  chatMessages.length = 0;
  setChatGenerating(false);
  updateChatSendState();
  persistChatHistory();
  renderChat();
});
document.getElementById('chat-output').addEventListener('click', handleChatActionClick);
document.getElementById('chat-output').addEventListener('toggle', handleThinkToggle, true);
document.getElementById('copy-chat-response').addEventListener('click', copyLatestAssistantResponse);

/* ---------------------------------------------------------------------
   Conversation export

   Renders the current `chatMessages` as a portable markdown document
   so users can paste it into a PR/Slack/dataset without retyping. We
   include the per-turn adapter+temperature badge so a recipient can
   see what produced each answer.
   --------------------------------------------------------------------- */
function exportChatAsMarkdown() {
  const sys = (document.getElementById('chat-system')?.value || '').trim();
  const lines = [`# Kiln playground transcript`, '', `_Exported ${new Date().toISOString()}_`, ''];
  if (sys) {
    lines.push('## System prompt', '', '```', sys, '```', '');
  }
  for (const m of chatMessages) {
    if (m.role === 'user') {
      lines.push('## User', '', m.content || '_(empty)_', '');
    } else if (m.role === 'assistant') {
      const tag = (m.adapter || m.adapter === null) && m.temperature != null
        ? ` — ${m.adapter || 'base'}, t=${m.temperature}`
        : '';
      lines.push(`## Assistant${tag}`, '');
      if (m.reasoning) {
        lines.push('<details><summary>Thinking</summary>', '', '```', m.reasoning, '```', '', '</details>', '');
      }
      if (m.error) {
        lines.push(`> **Error:** ${m.error}`, '');
      } else {
        lines.push(m.content || '_(empty)_', '');
      }
    }
  }
  const text = lines.join('\n');
  const writeText = navigator.clipboard?.writeText
    ? navigator.clipboard.writeText.bind(navigator.clipboard)
    : (t) => { fallbackCopyText(t); return Promise.resolve(); };
  writeText(text).then(() => {
    if (Object.prototype.hasOwnProperty.call(window, '__copiedText')) window.__copiedText = text;
    toast('Conversation copied as markdown');
  }).catch(() => {
    try { fallbackCopyText(text); toast('Conversation copied as markdown'); }
    catch { toast('Could not copy conversation.', 'err'); }
  });
}
document.getElementById('chat-export')?.addEventListener('click', exportChatAsMarkdown);

/* Escape inside the chat input stops an in-flight generation without
   stealing focus or wiping the input. Falls through when nothing is
   streaming so the user can still type literal escape sequences. */
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && (chatAbort || chatCompareAbort)) {
    e.preventDefault();
    if (chatAbort) chatAbort.abort();
    if (chatCompareAbort) chatCompareAbort.abort();
  }
});

/* ---------------------------------------------------------------------
   Wire up advanced-settings toggle + persistence

   The Advanced panel and every sampling control round-trip through
   localStorage on input/change so a refresh restores the user's setup.
   --------------------------------------------------------------------- */
const chatAdvBtn = document.getElementById('chat-toggle-advanced');
const chatAdvPanel = document.getElementById('chat-advanced');
if (chatAdvBtn && chatAdvPanel) {
  chatAdvBtn.addEventListener('click', () => {
    const open = chatAdvPanel.hidden;
    chatAdvPanel.hidden = !open;
    chatAdvBtn.setAttribute('aria-expanded', String(open));
    persistPlaygroundSettingsSoon();
  });
}

const PLAYGROUND_SETTING_IDS = [
  'chat-temp', 'chat-max-tokens', 'chat-enable-thinking',
  'chat-thinking-budget-mode', 'chat-thinking-budget-tokens', 'chat-thinking-budget-seconds',
  'chat-preset',
  'chat-top-p', 'chat-top-k', 'chat-min-p',
  'chat-presence-penalty', 'chat-frequency-penalty', 'chat-repetition-penalty',
  'chat-seed', 'chat-stop-sequences', 'chat-system',
];
PLAYGROUND_SETTING_IDS.forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  const ev = (el.type === 'checkbox' || el.tagName === 'SELECT') ? 'change' : 'input';
  el.addEventListener(ev, persistPlaygroundSettingsSoon);
});

const thinkingBudgetMode = document.getElementById('chat-thinking-budget-mode');
thinkingBudgetMode?.addEventListener('change', () => {
  syncThinkingBudgetControls({ revealCustom: true });
});
const thinkingEnabled = document.getElementById('chat-enable-thinking');
thinkingEnabled?.addEventListener('change', () => {
  syncThinkingBudgetControls();
});

// Wire the preset dropdown: changing it applies the preset's values
// to every advanced sampling input and flips the thinking toggle. Any
// later manual edit silently switches the selector to "custom" so the
// preset isn't lying about what's actually in the form.
const presetSelect = document.getElementById('chat-preset');
if (presetSelect) {
  presetSelect.addEventListener('change', (e) => {
    applyChatPreset(e.target.value);
  });
  const FIELDS_THAT_DESYNC_PRESET = [
    'chat-temp', 'chat-top-p', 'chat-top-k', 'chat-min-p',
    'chat-presence-penalty', 'chat-frequency-penalty', 'chat-repetition-penalty',
  ];
  FIELDS_THAT_DESYNC_PRESET.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      // Only mark custom if the user manually changed *after* page load.
      // We compare current value against the preset's spec.
      const currentPreset = presetSelect.value;
      const preset = QWEN3_PRESETS[currentPreset];
      if (!preset) return;
      const map = {
        'chat-temp': 'temperature', 'chat-top-p': 'topP', 'chat-top-k': 'topK',
        'chat-min-p': 'minP', 'chat-presence-penalty': 'presencePenalty',
        'chat-frequency-penalty': 'frequencyPenalty', 'chat-repetition-penalty': 'repetitionPenalty',
      };
      if (String(el.value) !== String(preset[map[id]])) {
        presetSelect.value = 'custom';
      }
    });
  });
  // The thinking checkbox also affects which preset is "consistent".
  const thinkingEl = document.getElementById('chat-enable-thinking');
  if (thinkingEl) {
    thinkingEl.addEventListener('change', () => {
      const preset = QWEN3_PRESETS[presetSelect.value];
      if (preset && preset.enableThinking !== thinkingEl.checked) {
        presetSelect.value = 'custom';
      }
    });
  }
}

// Restore settings + last conversation on first load. Settings always
// apply; conversation restore prompts a banner so users don't get a
// stale conversation invisibly attached to a fresh request.
applyPlaygroundSettings(readPlaygroundSettings());
restorePlaygroundHistoryBanner();

document.getElementById('upload-name').addEventListener('input', handleUploadNameInput);
document.getElementById('upload-archive').addEventListener('change', handleUploadArchiveChange);
updateUploadAdapterState();
document.getElementById('sft-output-name').addEventListener('input', updateSftSubmitState);
document.getElementById('sft-examples').addEventListener('input', (e) => { if (e.target.value.trim()) clearTrainingData('sft'); updateSftSubmitState(); });
document.getElementById('grpo-output-name').addEventListener('input', updateGrpoSubmitState);
document.getElementById('grpo-groups').addEventListener('input', (e) => { if (e.target.value.trim()) clearTrainingData('grpo'); updateGrpoSubmitState(); });
updateSftSubmitState();
updateGrpoSubmitState();
document.getElementById('merge-output-name').addEventListener('input', updateMergeButtonState);
document.getElementById('merge-mode').addEventListener('change', updateMergeButtonState);
document.getElementById('merge-density').addEventListener('input', updateMergeButtonState);
updateMergeButtonState();
/* =====================================================================
   Evals page — datasets, suites, jobs, judgments (the flywheel)

   This is intentionally one large module: every refresh is content-
   addressed by the active sub-tab and shares one drill-in modal so
   data flows across tabs (suite → run → drill, judgment → adapter
   validate → drill) without losing context.
   ===================================================================== */

function selectEvalsTab(tab, focus = false) {
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
    if (selected) tabPanel.removeAttribute('inert'); else tabPanel.setAttribute('inert', '');
  });
  if (focus) tab.focus();
  const which = tab.dataset.tab;
  try { localStorage.setItem('kiln.evalsSubTab', which); } catch {}
  // Deep-link hash for the sub-tab — covers clicks, arrow keys, and every
  // programmatic .click() caller (cmdk, quick actions, "View result" toasts).
  pushSubTabHash('evals');
  if (which === 'datasets') refreshDatasets();
  else if (which === 'suites') refreshSuites();
  else if (which === 'jobs') refreshEvalJobs();
  else if (which === 'judgments') refreshJudgments();
}
wireTablist(document.querySelector('[data-evals-tabs]'), {
  onSelect: (tab, { focus }) => selectEvalsTab(tab, focus),
});

// Restore the last visited eval sub-tab so users return to Jobs (or
// Suites / Judgments) instead of always-Datasets after a refresh.
// Hash-suppressed: the no-hash fallback — an explicit hash sub-tab is
// applied after this in the boot route pass and wins.
try {
  const lastEvalsSubTab = localStorage.getItem('kiln.evalsSubTab');
  if (lastEvalsSubTab && lastEvalsSubTab !== 'datasets') {
    const target = document.getElementById(`evals-tab-${lastEvalsSubTab}`);
    if (target) withHashWritesSuppressed(() => selectEvalsTab(target));
  }
} catch {}

let evalAdaptersCache = [];
let evalActiveAdapter = null;
async function refreshAdapterDropdowns() {
  try {
    const d = await api('/v1/adapters');
    evalAdaptersCache = (d.available || []).map(a => a.name);
    evalActiveAdapter = d.active || '';
    const targets = ['judgment-adapter-a', 'judgment-adapter-b', 'compile-judge-adapter'];
    // Rebuild the <option> lists only when the adapter name set changed —
    // this runs on the Evals poll tick, and an unconditional rebuild snaps
    // an open dropdown shut mid-pick. Unchanged options always skip, so a
    // focused select is never rebuilt under the user.
    const optionsKey = 'opts:' + JSON.stringify(evalAdaptersCache);
    const optionsHtml = ['<option value="">Base model</option>']
      .concat(evalAdaptersCache.map(n => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`))
      .join('');
    for (const id of targets) {
      const sel = document.getElementById(id);
      if (!sel) continue;
      const cur = sel.value;
      if (setListHtml(sel, optionsKey, optionsHtml)) {
        // Preserve the user's in-flight selection across the rebuild.
        if (cur && evalAdaptersCache.includes(cur)) sel.value = cur;
      }
    }
  } catch (_) { /* best-effort */ }
}

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function truncate(s, n) {
  s = String(s || '');
  if (s.length <= n) return s;
  return s.slice(0, n) + '…';
}
function fmtPct(x, digits = 1) {
  if (x == null || !isFinite(x)) return '—';
  return (x * 100).toFixed(digits) + '%';
}

/* ---------- Accuracy ring (color graded by score) ---------- */
function ringHtml(accuracy, size = '') {
  const pct = (accuracy != null && isFinite(accuracy)) ? Math.max(0, Math.min(1, accuracy)) : 0;
  // Color gradient: red → orange → green
  let color;
  if (pct >= 0.8) color = 'var(--success-fg)';
  else if (pct >= 0.5) color = 'var(--warning-fg)';
  else if (pct > 0) color = 'var(--danger-fg)';
  else color = 'var(--text-quiet)';
  const sizeClass = size ? ` ${size}` : '';
  return `<span class="acc-ring${sizeClass}" style="--ring-pct:${(pct*100).toFixed(0)}; --ring-color:${color};"><span class="acc-ring-num">${(pct*100).toFixed(0)}</span></span>`;
}

/* ---------- Sparkline (suite history) ---------- */
function sparkSvg(values, width = 64, height = 18) {
  if (!values || values.length < 2) return '';
  const pad = 1;
  const w = width - 2 * pad;
  const h = height - 2 * pad;
  const xs = values.map((_, i) => pad + (i * w) / (values.length - 1));
  const ys = values.map(v => pad + h - Math.max(0, Math.min(1, v)) * h);
  const linePath = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(' ');
  const areaPath = `${linePath} L${xs[xs.length-1].toFixed(1)} ${(pad+h).toFixed(1)} L${xs[0].toFixed(1)} ${(pad+h).toFixed(1)} Z`;
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
    <path class="spark-area" d="${areaPath}"/>
    <path d="${linePath}"/>
  </svg>`;
}

/* ---------- Datasets ---------- */

let activeSynthDataset = null;
async function refreshDatasets() {
  try {
    const d = await api('/v1/eval/datasets');
    const datasets = d.datasets || [];
    const el = document.getElementById('datasets-list');
    if (!datasets.length) {
      el.className = 'eval-empty';
      // The corrections CTA tracks the basket: enabled the moment a finished
      // correction exists. Key on that count so the 1.5s poll repaints the
      // button state as corrections arrive (or get their ideal answers).
      const corrReady = (typeof correctionsBasket !== 'undefined' && typeof corrTrainable === 'function')
        ? correctionsBasket.filter(corrTrainable).length : 0;
      const corrHint = corrReady > 0
        ? `Turn your ${corrReady} finished correction${corrReady === 1 ? '' : 's'} into a dataset you can build evals from`
        : 'Nothing to build yet — when pi gives a wrong answer, add it to Corrections (Overview page) and write the ideal answer first';
      const wrote = setListHtml(el, 'empty:' + corrReady, `
        <div class="eval-empty-icon"><svg class="icn"><use href="#i-folder"></use></svg></div>
        <div class="eval-empty-title">No datasets yet</div>
        <div class="eval-empty-body">A dataset is a list of conversations — the raw material Kiln turns into eval suites and training runs. Upload your own above, or start with one of these:</div>
        <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
          <button class="eval-empty-cta" type="button" id="use-sample-dataset" title="Adds a small built-in dataset of coding-agent conversations — tool calls, code review, commit messages — so you can try the eval flow without bringing your own data">Try a sample dataset</button>
          <button class="eval-empty-cta" type="button" id="dataset-from-corrections" ${corrReady > 0 ? '' : 'disabled '}title="${escapeHtml(corrHint)}">Build a dataset from your corrections</button>
        </div>`);
      if (wrote) {
        document.getElementById('use-sample-dataset')?.addEventListener('click', ev => uploadSampleDataset(ev.currentTarget));
        document.getElementById('dataset-from-corrections')?.addEventListener('click', ev => buildDatasetFromCorrections(ev.currentTarget));
      }
      return;
    }
    el.className = '';
    // Key on every payload field the rows display: stats covers the
    // role-pattern column, the assistant/tool_calls counts, and the
    // recommendStrategy badge (derived solely from stats).
    const listKey = 'list:' + JSON.stringify(datasets.map(m =>
      [m.name, m.format, m.description, m.num_rows, m.size_bytes, m.stats]));
    const listHtml = datasets.map(m => {
      const stats = m.stats || {};
      const pattern = (stats.sample_role_patterns || []).slice(0, 1).join(' · ') || '';
      const recommendation = recommendStrategy(stats);
      return `<div class="eval-row eval-row-datasets">
        <div>
          <div class="row-title">${escapeHtml(m.name)}</div>
          <div class="row-sub">${escapeHtml(m.format)} · ${escapeHtml(m.description || 'No description')}</div>
        </div>
        <div class="tabular-nums">${m.num_rows.toLocaleString()} rows · ${fmtBytes(m.size_bytes)}</div>
        <div class="row-sub" title="Detected from the first ${stats.num_assistant_turns ? '1000' : 0} rows">
          ${stats.num_assistant_turns ? stats.num_assistant_turns.toLocaleString() + ' assistant · ' + (stats.num_with_tool_calls || 0) + ' tool_calls' : '—'}
          ${recommendation ? `<div style="margin-top:2px;"><span class="scorer-badge" title="Recommended synthesis strategy">${escapeHtml(recommendation)}</span></div>` : ''}
        </div>
        <div class="row-sub" style="font-family:var(--font-mono);">${escapeHtml(pattern)}</div>
        <div class="row-actions">
          ${m.format === 'sft_chat' ? `<button type="button" class="btn btn-primary btn-sm" data-action="train-sft" data-name="${escapeHtml(m.name)}" title="Open Training with this dataset loaded — one click from here to a queued job">Train SFT →</button>` : ''}
          ${m.format === 'grpo_groups' ? `<button type="button" class="btn btn-primary btn-sm" data-action="train-grpo" data-name="${escapeHtml(m.name)}" title="Open Training with this dataset loaded — one click from here to a queued job">Train GRPO →</button>` : ''}
          <button type="button" class="btn ${m.format === 'raw' ? 'btn-primary ' : ''}btn-sm" data-action="synth" data-name="${escapeHtml(m.name)}">Synthesize eval</button>
          <button type="button" class="btn btn-sm" data-action="del" data-name="${escapeHtml(m.name)}">Delete</button>
        </div>
      </div>`;
    }).join('');
    if (!setListHtml(el, listKey, listHtml)) return; // unchanged — old nodes keep their listeners
    el.querySelectorAll('button[data-action]').forEach(b => {
      const name = b.dataset.name;
      if (b.dataset.action === 'train-sft') {
        b.addEventListener('click', () => trainFromDataset(name, 'sft'));
      } else if (b.dataset.action === 'train-grpo') {
        b.addEventListener('click', () => trainFromDataset(name, 'grpo'));
      } else if (b.dataset.action === 'synth') {
        b.addEventListener('click', () => openSynthPanel(name));
      } else if (b.dataset.action === 'del') {
        b.addEventListener('click', async () => {
          if (!confirm(`Delete dataset "${name}"?`)) return;
          try {
            await api('/v1/eval/datasets/' + encodeURIComponent(name), { method: 'DELETE' });
            toast('Dataset deleted', 'ok');
            refreshDatasets();
          } catch (e) { toast('Delete failed: ' + e.message, 'err'); }
        });
      }
    });
  } catch (e) {
    // Route the failure write through setListHtml too: it stamps an
    // error-specific key, so the post-recovery payload (even an identical
    // empty list) compares unequal and repaints (#1547 regression shape).
    setListHtml(document.getElementById('datasets-list'), 'err:' + e.message,
      `<div class="eval-empty"><div class="eval-empty-title">Failed to load</div><div class="eval-empty-body">${escapeHtml(e.message)}</div></div>`);
  }
}

function recommendStrategy(stats) {
  if (!stats || !stats.num_assistant_turns) return null;
  // Tool-call heavy → tool_call_predict
  const toolFraction = stats.num_with_tool_calls / Math.max(1, stats.num_assistant_turns);
  if (toolFraction > 0.3) return 'tool_call_predict ↘';
  // Multi-turn → every_assistant_turn
  if (stats.avg_messages_per_conv > 8) return 'every_assistant_turn';
  // Otherwise → final_assistant
  return 'final_assistant';
}

function openSynthPanel(name) {
  activeSynthDataset = name;
  document.getElementById('synth-dataset-name').textContent = name;
  document.getElementById('synth-suite-name').value = name + '-eval';
  document.getElementById('synth-preview-output').innerHTML = '';
  document.getElementById('synthesize-panel').hidden = false;
  document.getElementById('synthesize-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

document.getElementById('synth-close')?.addEventListener('click', () => {
  document.getElementById('synthesize-panel').hidden = true;
  activeSynthDataset = null;
});

// The judge scorer needs to know WHICH adapter judges — typically the
// judge LoRA trained from A/B picks. Reveal + populate the picker only
// when the judge scorer is selected.
document.getElementById('synth-scorer')?.addEventListener('change', () => {
  const isJudge = document.getElementById('synth-scorer').value === 'judge';
  const group = document.getElementById('synth-judge-adapter-group');
  if (group) group.hidden = !isJudge;
  if (isJudge) populateSynthJudgeAdapters();
});

async function populateSynthJudgeAdapters() {
  const sel = document.getElementById('synth-judge-adapter');
  if (!sel) return;
  const current = sel.value;
  try {
    const res = await api('/v1/adapters');
    const names = (res.available || []).map(a => a.name);
    sel.innerHTML = '<option value="">Base model</option>' +
      names.map(n => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('');
    if (names.includes(current)) sel.value = current;
  } catch (_) { /* adapter list unavailable — base-model option remains */ }
}

function readSynthConfig() {
  const suite_name = document.getElementById('synth-suite-name').value.trim();
  if (!suite_name) { toast('Suite name is required', 'err'); return null; }
  const strategy = document.getElementById('synth-strategy').value;
  const scorerChoice = document.getElementById('synth-scorer').value;
  let scorer;
  if (scorerChoice === 'auto')       scorer = { kind: 'auto_detect' };
  else if (scorerChoice === 'judge') {
    const judgeAdapter = document.getElementById('synth-judge-adapter')?.value || null;
    scorer = { kind: 'judge', judge_adapter: judgeAdapter };
  }
  else if (scorerChoice === 'exact_match') scorer = { kind: 'fixed', scorer: { kind: 'exact_match', case_sensitive: false, strip_whitespace: true } };
  else if (scorerChoice === 'contains')    scorer = { kind: 'fixed', scorer: { kind: 'contains', phrases: [], mode: 'any', case_sensitive: false } };
  else if (scorerChoice === 'numeric')     scorer = { kind: 'fixed', scorer: { kind: 'numeric_tolerance', atol: 0, rtol: 0, integer_only: false } };
  else if (scorerChoice === 'tool_call')   scorer = { kind: 'fixed', scorer: { kind: 'tool_call' } };
  else if (scorerChoice === 'code')        scorer = { kind: 'fixed', scorer: { kind: 'code', style: { kind: 'token_similarity', min_jaccard: 0.6 } } };
  const max_examples = parseInt(document.getElementById('synth-max-examples').value, 10);
  const seedVal = document.getElementById('synth-seed').value;
  const sampling = {
    max_examples: isFinite(max_examples) && max_examples > 0 ? max_examples : null,
    max_prompt_chars: 32768,
    max_target_chars: 4096,
    seed: seedVal ? parseInt(seedVal, 10) : null,
    dedupe: true,
  };
  return {
    suite_name,
    strategy,
    scorer,
    generation: { temperature: 0.0, top_p: 1.0, top_k: 0, max_tokens: 256, n: 1, stop: [], seed: null },
    sampling,
    strip_system_prompt: document.getElementById('synth-strip-system').checked,
  };
}

document.getElementById('synth-preview-btn')?.addEventListener('click', async () => {
  if (!activeSynthDataset) return;
  const config = readSynthConfig();
  if (!config) return;
  const out = document.getElementById('synth-preview-output');
  out.innerHTML = '<div class="hint">Synthesizing preview…</div>';
  try {
    const res = await api('/v1/eval/datasets/' + encodeURIComponent(activeSynthDataset) + '/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...config, head_n: 5 }),
    });
    renderSynthPreview(out, res);
  } catch (e) { out.innerHTML = '<div class="eval-empty"><div class="eval-empty-body">Preview failed: ' + escapeHtml(e.message) + '</div></div>'; }
});

function renderSynthPreview(container, preview) {
  const s = preview.stats || {};
  const examples = preview.examples || [];
  const hist = s.auto_scorer_histogram || {};
  const histStr = Object.entries(hist).map(([k, v]) => `<span class="scorer-badge">${escapeHtml(k)}×${v}</span>`).join(' ');
  const exHtml = examples.slice(0, 5).map((ex, i) => {
    const userMsg = (ex.messages || []).filter(m => m.role === 'user').slice(-1)[0];
    const userText = userMsg ? userMsg.content : '';
    const tags = (ex.tags || []).map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('');
    const scorerKind = ex.scorer ? ex.scorer.kind : '(suite default)';
    return `<div style="border:1px solid var(--border); border-radius:6px; padding:10px; margin-top:6px; background:var(--surface);">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <div class="eyebrow">Example ${i+1}</div>
        <span class="scorer-badge">${escapeHtml(scorerKind)}</span>
      </div>
      <div style="font-size:11px; color:var(--text-muted); margin-bottom:2px;">prompt</div>
      <div style="font-family:var(--font-mono); font-size:12px; max-height:60px; overflow:auto; margin-bottom:6px;">${escapeHtml(truncate(userText, 240))}</div>
      <div style="font-size:11px; color:var(--text-muted); margin-bottom:2px;">target</div>
      <div style="font-family:var(--font-mono); font-size:12px; max-height:80px; overflow:auto;">${escapeHtml(truncate(ex.target || '', 320))}</div>
      <div style="margin-top:6px;">${tags}</div>
    </div>`;
  }).join('');
  container.innerHTML = `
    <div style="margin-bottom:8px; padding:10px; background:var(--surface-2); border-radius:6px; display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
      <div>
        <div class="hint" style="font-size:11px; color:var(--text-muted);">examples generated</div>
        <div style="font-size:18px; font-weight:700; font-variant-numeric:tabular-nums;">${(s.examples_generated || 0).toLocaleString()}</div>
      </div>
      <div>
        <div class="hint" style="font-size:11px; color:var(--text-muted);">trajectories used</div>
        <div style="font-size:18px; font-weight:700; font-variant-numeric:tabular-nums;">${(s.trajectories_used || 0).toLocaleString()}</div>
      </div>
      <div style="flex:1; min-width:200px;">
        <div class="hint" style="font-size:11px; color:var(--text-muted); margin-bottom:4px;">auto-detected scorers</div>
        <div>${histStr || '<span class="hint">n/a</span>'}</div>
      </div>
    </div>
    <div class="hint" style="margin-bottom:8px; font-size:11px;">Skipped: empty target=${s.skipped_no_target || 0} · prompt-too-long=${s.skipped_prompt_too_long || 0} · target-too-long=${s.skipped_target_too_long || 0} · duplicate=${s.skipped_duplicate || 0}</div>
    ${exHtml || '<div class="eval-empty"><div class="eval-empty-body">No examples produced — try a different strategy or relax the sampling caps.</div></div>'}
  `;
}

async function doSynthesize(runAgainst) {
  if (!activeSynthDataset) return;
  const config = readSynthConfig();
  if (!config) return;
  try {
    const body = { ...config, force: false };
    if (runAgainst && runAgainst.length) body.run_against = runAgainst;
    const res = await api('/v1/eval/datasets/' + encodeURIComponent(activeSynthDataset) + '/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const queued = (res.queued_eval_job_ids || []).length;
    toast(`Saved suite "${res.suite.name}" (${res.stats.examples_generated} examples)${queued ? ', queued ' + queued + ' eval job(s)' : ''}`, 'ok');
    refreshSuites();
    refreshEvalJobs();
    if (queued > 0) {
      // Hop to the Jobs tab so the user immediately sees the run.
      document.getElementById('evals-tab-jobs')?.click();
    }
  } catch (e) { toast('Synthesize failed: ' + e.message, 'err'); }
}

document.getElementById('synth-save-btn')?.addEventListener('click', () => doSynthesize([]));
document.getElementById('synth-save-and-run-btn')?.addEventListener('click', async () => {
  await doSynthesize([evalActiveAdapter || '']);
});

// Shared multipart POST for every dataset-upload surface (the form, the
// sample-dataset CTA, the corrections builder). Matches the server contract:
// fields `name`, `format`, optional `description`, and `file` (JSONL bytes).
async function postDatasetUpload(name, format, description, fileOrBlob) {
  const fd = new FormData();
  fd.append('name', name);
  fd.append('format', format);
  if (description) fd.append('description', description);
  fd.append('file', fileOrBlob, fileOrBlob.name || name + '.jsonl');
  const res = await fetch('/v1/eval/datasets/upload', { method: 'POST', body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const e = new Error(err.error?.message || `HTTP ${res.status}`);
    e.code = err.error?.code;
    throw e;
  }
  return res.json();
}

document.getElementById('dataset-upload-form')?.addEventListener('submit', async ev => {
  ev.preventDefault();
  const name = document.getElementById('dataset-name').value.trim();
  const format = document.getElementById('dataset-format').value;
  const description = document.getElementById('dataset-description').value.trim();
  const file = document.getElementById('dataset-file').files[0];
  if (!name || !file) { toast('Name and file are required', 'err'); return; }
  try {
    const m = await postDatasetUpload(name, format, description, file);
    toast(`Uploaded "${m.name}" (${m.num_rows.toLocaleString()} rows)`, 'ok');
    document.getElementById('dataset-upload-form').reset();
    refreshDatasets();
    // Next step depends on what they uploaded: training data should lead to
    // TRAINING (one click), not get hijacked into the eval-synthesis flow.
    if (format === 'sft_chat' || format === 'grpo_groups') {
      const kind = format === 'sft_chat' ? 'sft' : 'grpo';
      showDatasetUploadedNext(m.name, kind, m.num_rows);
    } else {
      openSynthPanel(m.name);
    }
  } catch (e) { toast('Upload failed: ' + e.message, 'err'); }
});

/* ---------- First-run CTAs: sample dataset + corrections → dataset ---------- */

// A small built-in sft_chat dataset themed on realistic coding-agent traffic
// (what pi actually sends Kiln: tool calls, code review, commit messages,
// test writing) so "Run an eval" works in the first five minutes without
// bringing your own data. Every row matches the server's SftConversation
// contract exactly: messages[] of {role, content}, OpenAI-style tool_calls
// on assistant turns, and name/tool_call_id on tool replies.
const SAMPLE_DATASET_NAME = 'sample-coding-agent';
function sampleDatasetRows() {
  const call = (id, name, args) => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } });
  return [
    { messages: [
      { role: 'user', content: 'What files import the deprecated `legacy_client` module?' },
      { role: 'assistant', content: '', tool_calls: [call('call_1', 'grep_repo', { pattern: 'legacy_client', glob: '**/*.rs' })] },
      { role: 'tool', name: 'grep_repo', tool_call_id: 'call_1', content: 'src/api/handlers.rs:12\nsrc/sync/worker.rs:7\ntests/integration/client_test.rs:3' },
      { role: 'assistant', content: 'Three files still import `legacy_client`: src/api/handlers.rs, src/sync/worker.rs, and tests/integration/client_test.rs. The two src/ call sites need migrating to `TokenStore` before the module can be removed; the test can be deleted with it.' },
    ] },
    { messages: [
      { role: 'user', content: 'Run the cache tests and tell me if anything fails.' },
      { role: 'assistant', content: '', tool_calls: [call('call_2', 'run_tests', { package: 'cache', filter: '' })] },
      { role: 'tool', name: 'run_tests', tool_call_id: 'call_2', content: 'running 14 tests\ntest read_through_hit ... ok\ntest eviction_lru ... FAILED\n13 passed; 1 failed' },
      { role: 'assistant', content: 'One failure: `eviction_lru`. The test expects the least-recently-used entry to be evicted first, but the current implementation evicts by insertion order. The fix is to touch the entry timestamp on read, not just on write.' },
    ] },
    { messages: [
      { role: 'user', content: 'What does the retry helper in src/net/retry.rs actually do?' },
      { role: 'assistant', content: '', tool_calls: [call('call_3', 'read_file', { path: 'src/net/retry.rs' })] },
      { role: 'tool', name: 'read_file', tool_call_id: 'call_3', content: 'pub async fn with_retry<F, T>(op: F, max: u32) -> Result<T> { /* exponential backoff, jitter 0-50ms, caps at 5s */ }' },
      { role: 'assistant', content: '`with_retry` wraps an async operation with exponential backoff: it retries up to `max` times, doubling the delay each attempt with 0-50ms of jitter, capped at 5 seconds per wait. Errors from the final attempt propagate to the caller unchanged.' },
    ] },
    { messages: [
      { role: 'user', content: 'Generate a conventional-commit message for this change: swapped the blocking Mutex for an async RwLock in cache.rs so reads no longer serialize.' },
      { role: 'assistant', content: 'fix(cache): replace blocking Mutex with async RwLock so concurrent reads no longer serialize' },
    ] },
    { messages: [
      { role: 'user', content: 'Review this diff and flag anything risky: the PR holds a `.read()` guard across an `await` on the network fetch in cache.rs line 88.' },
      { role: 'assistant', content: 'Risky: holding the RwLock read guard across the awaited network fetch blocks every writer for the full request latency, and can deadlock if the fetch path ever takes the write lock. Scope the guard to the lookup only — clone the value out, drop the guard, then await the fetch.' },
    ] },
    { messages: [
      { role: 'user', content: 'How do I store handlers with different concrete types in one Vec in Rust?' },
      { role: 'assistant', content: 'Use trait objects: define a common trait and store boxed instances — `Vec<Box<dyn Handler>>`. Each handler implements `Handler`, and you push `Box::new(MyHandler)` into the vec. If the handlers are cloned or shared across threads, use `Vec<Arc<dyn Handler + Send + Sync>>` instead.' },
    ] },
    { messages: [
      { role: 'user', content: 'Write a unit test for the rate limiter covering burst then steady-state.' },
      { role: 'assistant', content: '#[tokio::test]\nasync fn limiter_allows_burst_then_throttles() {\n    let lim = RateLimiter::new(5, Duration::from_secs(1));\n    for _ in 0..5 {\n        assert!(lim.try_acquire());\n    }\n    assert!(!lim.try_acquire(), "burst budget should be exhausted");\n    tokio::time::sleep(Duration::from_secs(1)).await;\n    assert!(lim.try_acquire(), "tokens should refill after the window");\n}' },
    ] },
    { messages: [
      { role: 'user', content: 'Explain what this query does: SELECT user_id, COUNT(*) AS n FROM events WHERE ts > now() - interval \'7 days\' GROUP BY user_id HAVING COUNT(*) > 100;' },
      { role: 'assistant', content: 'It finds heavy users from the last week: it filters `events` to rows newer than 7 days, groups them by `user_id`, counts each user\'s events, and keeps only users with more than 100 events. The result is one row per heavy user with their event count `n`.' },
    ] },
    { messages: [
      { role: 'user', content: 'Why would the integration test `flaky_reconnect` fail intermittently but pass on retry?' },
      { role: 'assistant', content: 'The test asserts reconnection within 100ms, but the client\'s backoff adds random jitter that can push the first retry to ~150ms under load. Either widen the assertion window to cover worst-case jitter or inject a deterministic backoff in tests.' },
    ] },
    { messages: [
      { role: 'system', content: 'You are a careful coding agent working in a Rust codebase.' },
      { role: 'user', content: 'Refactor src/auth/session.rs to use the new TokenStore trait. What is your plan?' },
      { role: 'assistant', content: 'Plan: 1) add a `store: Arc<dyn TokenStore>` field to `Session` and thread it through `Session::new`; 2) replace the three direct `token_db` calls in handlers/ with `self.store` methods; 3) keep the old constructor behind a deprecated shim for one release; 4) run the auth test suite and fix call sites until green.' },
    ] },
  ];
}

// "Try a sample dataset" — upload the embedded rows through the same endpoint
// a real JSONL file goes through, then hand off to the eval-suite synthesizer
// (that is the next step on the eval golden path).
async function uploadSampleDataset(btn) {
  if (btn) btn.disabled = true;
  try {
    const jsonl = sampleDatasetRows().map(r => JSON.stringify(r)).join('\n') + '\n';
    const blob = new Blob([jsonl], { type: 'application/jsonl' });
    const m = await postDatasetUpload(SAMPLE_DATASET_NAME, 'sft_chat',
      'Built-in sample: coding-agent conversations with tool calls', blob);
    refreshDatasets();
    toast(`Sample dataset added (${m.num_rows} rows) — next: synthesize an eval suite from it`, 'ok');
    openSynthPanel(m.name);
  } catch (e) {
    if (e.code === 'dataset_exists' || /already exists/i.test(e.message || '')) {
      toast('The sample dataset is already here — synthesize an eval suite from it', 'info');
      refreshDatasets();
      openSynthPanel(SAMPLE_DATASET_NAME);
    } else {
      toast('Could not add the sample dataset: ' + e.message, 'err');
      if (btn) btn.disabled = false;
    }
  }
}

// "Build a dataset from your corrections" — the durable corrections store
// (your hand-written ideal answers, including rows already trained into an
// adapter) becomes an sft_chat dataset via the SAME transform the Corrections
// card trains with, so you can eval exactly what you taught.
async function buildDatasetFromCorrections(btn) {
  if (btn) btn.disabled = true;
  try {
    let rows = correctionsBasket;
    try {
      const d = await api('/v1/corrections?include_trained=1');
      if (d && Array.isArray(d.corrections)) rows = d.corrections;
    } catch (_) { /* server store unreachable — the local basket still works */ }
    const finished = rows.filter(corrTrainable);
    if (!finished.length) {
      toast('Your corrections need ideal answers first — open Corrections on the Overview page and write what pi should have said', 'info');
      if (btn) btn.disabled = false;
      return;
    }
    const jsonl = correctionsToSftExamples(finished).map(r => JSON.stringify(r)).join('\n') + '\n';
    const name = 'corrections-' + new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    const m = await postDatasetUpload(name, 'sft_chat',
      'Your corrections: each row pairs a prompt with the answer you said pi should have given', new Blob([jsonl], { type: 'application/jsonl' }));
    refreshDatasets();
    toast(`Dataset "${m.name}" built from ${finished.length} correction${finished.length === 1 ? '' : 's'} — next: synthesize an eval suite from it`, 'ok');
    openSynthPanel(m.name);
  } catch (e) {
    toast('Could not build the dataset: ' + e.message, 'err');
    if (btn) btn.disabled = false;
  }
}

// Inline "uploaded — what next?" strip on the Datasets tab. Primary action is
// training (that's why most people upload SFT/GRPO data); synthesizing an eval
// from the same rows is offered alongside.
function showDatasetUploadedNext(name, kind, numRows) {
  const old = document.getElementById('dataset-uploaded-next');
  if (old) old.remove();
  const form = document.getElementById('dataset-upload-form');
  if (!form) return;
  const strip = document.createElement('div');
  strip.id = 'dataset-uploaded-next';
  strip.className = 'corr-receipt';
  strip.setAttribute('role', 'status');
  strip.innerHTML = `
    <span class="corr-receipt-icon">${icon('check', 'icn-sm')}</span>
    <span class="corr-receipt-text"><strong>${escapeHtml(name)}</strong> uploaded (${Number(numRows || 0).toLocaleString()} rows). Train on it now, or build an eval from it.</span>
    <button type="button" class="btn btn-sm btn-primary" id="dataset-next-train">Train on this dataset ${icon('arrow-right', 'icn-sm')}</button>
    <button type="button" class="btn btn-sm" id="dataset-next-synth">Synthesize an eval</button>
    <button type="button" class="btn btn-sm btn-ghost corr-receipt-dismiss" id="dataset-next-dismiss" aria-label="Dismiss">${icon('close', 'icn-sm')}</button>`;
  form.insertAdjacentElement('afterend', strip);
  document.getElementById('dataset-next-train')?.addEventListener('click', () => { strip.remove(); trainFromDataset(name, kind); });
  document.getElementById('dataset-next-synth')?.addEventListener('click', () => { strip.remove(); openSynthPanel(name); });
  document.getElementById('dataset-next-dismiss')?.addEventListener('click', () => strip.remove());
}

/* ---------- Suites ---------- */

// Cache job results so we can compute per-suite sparkline trends.
let evalJobsCache = [];
// Lifecycle counts derived from evalJobsCache on every refresh. `null` until
// the first /v1/eval/jobs response lands so consumers (updateFlywheel) can
// tell "not loaded yet" apart from "zero evals ever" — an unfetched jobs
// list is unknown, not empty.
let evalJobCounts = null;

async function refreshSuites() {
  try {
    const d = await api('/v1/eval/suites');
    const suites = d.suites || [];
    const el = document.getElementById('suites-list');
    if (!suites.length) {
      el.className = 'eval-empty';
      setListHtml(el, 'empty', `
        <div class="eval-empty-icon"><svg class="icn"><use href="#i-target"></use></svg></div>
        <div class="eval-empty-title">No eval suites yet</div>
        <div class="eval-empty-body">A suite is a set of prompts with expected answers and a scorer — your model's report card. Create one from a dataset on the Datasets tab; no data yet? The built-in sample dataset works out of the box.</div>
        <button class="eval-empty-cta" type="button" title="Synthesize a suite from any dataset — power users can also POST an EvalSuite document to /v1/eval/suites" onclick="document.getElementById('evals-tab-datasets').click()">Create a suite from a dataset</button>`);
      return;
    }
    el.className = '';
    // Build a per-suite history from the cached job list. The server returns
    // jobs NEWEST-first (sorted descending by submitted_at_iso), but the
    // sparkline draws points left-to-right in array order and the badge takes
    // the LAST entry — so accumulate each history oldest→newest. Sort a copy
    // (never mutate the shared evalJobsCache) rather than blindly reversing,
    // matching the defensive re-sorts in adapterEvalChip/adapterCompareVerdict.
    const suiteHistory = {};
    const completedOldestFirst = evalJobsCache
      .filter(j => j.state === 'completed' && j.headline_accuracy != null)
      .sort((a, b) => String(a.submitted_at_iso || '').localeCompare(String(b.submitted_at_iso || '')));
    for (const j of completedOldestFirst) {
      (suiteHistory[j.suite_name] = suiteHistory[j.suite_name] || []).push(j.headline_accuracy);
    }
    // Key on everything the cards display: the suites payload, the
    // sparkline/badge history derived from evalJobsCache (#1548 — a new
    // completed run must repaint even when the suites payload is byte-
    // identical), and evalActiveAdapter (the Run/A-B button titles and the
    // A/B disabled state embed it).
    const listKey = 'list:' + JSON.stringify([
      evalActiveAdapter,
      suites.map(s => [s.name, s.description, s.num_examples, s.default_scorer_kind]),
      completedOldestFirst.map(j => [j.suite_name, j.headline_accuracy]),
    ]);
    const listHtml = suites.map(s => {
      const hist = (suiteHistory[s.name] || []).slice(-10);
      const recent = hist.length ? hist[hist.length - 1] : null;
      const sparkline = hist.length >= 2 ? sparkSvg(hist) : '';
      const recentBadge = recent != null
        ? `<span class="job-state-pill completed" title="Latest run accuracy">${(recent*100).toFixed(0)}%</span>`
        : '';
      return `<div class="eval-row eval-row-suites">
        <div>
          <div class="row-title">${escapeHtml(s.name)}</div>
          <div class="row-sub">${escapeHtml(truncate(s.description || 'No description', 120))}</div>
        </div>
        <div class="tabular-nums">${s.num_examples.toLocaleString()} examples · <span class="scorer-badge">${escapeHtml(s.default_scorer_kind)}</span></div>
        <div style="display:flex; gap:6px; align-items:center;">${recentBadge} ${sparkline}</div>
        <div class="row-actions">
          <button type="button" class="btn btn-primary btn-sm" data-suite="${escapeHtml(s.name)}" data-action="run" ${evalActiveAdapter ? `title="Score ${escapeHtml(evalActiveAdapter)} (the active adapter) on this suite"` : 'title="Score the base model on this suite"'}>Run</button>
          <button type="button" class="btn btn-sm" data-suite="${escapeHtml(s.name)}" data-action="compare" ${evalActiveAdapter ? `title="Compare base vs ${escapeHtml(evalActiveAdapter)} (the active adapter) — to compare a different adapter, use Run eval… on its card under Adapters"` : 'disabled title="No adapter is active — load one on the Adapters page, or use Run eval… on any adapter card"'}>A/B${evalActiveAdapter ? '' : ''}</button>
          <button type="button" class="btn btn-sm" data-suite="${escapeHtml(s.name)}" data-action="preview" title="Show the first few examples without running">Preview</button>
          <button type="button" class="btn btn-sm" data-suite="${escapeHtml(s.name)}" data-action="del">Delete</button>
        </div>
      </div>`;
    }).join('');
    if (!setListHtml(el, listKey, listHtml)) return; // unchanged — old nodes keep their listeners
    el.querySelectorAll('button[data-suite]').forEach(b => {
      const suite = b.dataset.suite;
      b.addEventListener('click', async () => {
        const action = b.dataset.action;
        try {
          if (action === 'run') {
            const res = await api('/v1/eval/run', {
              method: 'POST', headers: {'Content-Type':'application/json'},
              body: JSON.stringify({ suite, adapter: evalActiveAdapter || '' }),
            });
            toast('Queued eval ' + res.job_id.slice(0, 8), 'ok');
            document.getElementById('evals-tab-jobs')?.click();
            refreshEvalJobs();
          } else if (action === 'compare') {
            const res = await api('/v1/eval/compare', {
              method: 'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({ suite, adapters: ['', evalActiveAdapter || ''] }),
            });
            toast('Queued compare ' + res.job_id.slice(0, 8), 'ok');
            document.getElementById('evals-tab-jobs')?.click();
            refreshEvalJobs();
          } else if (action === 'preview') {
            await openSuitePreview(suite);
          } else if (action === 'del') {
            if (!confirm(`Delete suite "${suite}"?`)) return;
            await api('/v1/eval/suites/' + encodeURIComponent(suite), { method: 'DELETE' });
            toast('Suite deleted', 'ok');
            refreshSuites();
          }
        } catch (e) { toast(action + ' failed: ' + e.message, 'err'); }
      });
    });
  } catch (e) {
    // Error-specific key: recovery payloads (even identical ones) repaint.
    setListHtml(document.getElementById('suites-list'), 'err:' + e.message,
      `<div class="eval-empty"><div class="eval-empty-body">Failed: ${escapeHtml(e.message)}</div></div>`);
  }
}

/* ---------- Suite preview (lightweight modal — first N examples) ---------- */
function closeSuitePreviewModal() {
  const modal = document.getElementById('suite-preview-modal');
  if (!modal || modal.hidden) return;
  modal.hidden = true;
  closeModal(modal);
}
async function openSuitePreview(name) {
  // Lazy-create the modal scaffolding on first use. Reuses the same
  // CSS classes as the other drill-ins for consistency. Escape, focus,
  // and the scroll lock come from the shared modal manager.
  let modal = document.getElementById('suite-preview-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'suite-preview-modal';
    modal.className = 'modal-backdrop';
    modal.role = 'dialog';
    modal.setAttribute('aria-modal', 'true');
    modal.innerHTML = `<div class="modal-shell" tabindex="-1">
      <div class="modal-head">
        <h2 id="suite-preview-title">Suite preview</h2>
        <span class="modal-meta" id="suite-preview-meta"></span>
        <button class="modal-close" id="suite-preview-close" aria-label="Close"><svg class="icn" aria-hidden="true"><use href="#i-close"></use></svg></button>
      </div>
      <div class="modal-body" style="grid-template-columns: 1fr;">
        <div class="modal-content" id="suite-preview-content"><div class="detail-empty">Loading…</div></div>
      </div>
    </div>`;
    document.body.appendChild(modal);
    document.getElementById('suite-preview-close').addEventListener('click', closeSuitePreviewModal);
    modal.addEventListener('click', (ev) => {
      if (ev.target === modal) closeSuitePreviewModal();
    });
  }
  modal.hidden = false;
  openModal(modal, { onClose: closeSuitePreviewModal });
  document.getElementById('suite-preview-title').textContent = `Suite: ${name}`;
  document.getElementById('suite-preview-meta').textContent = '';
  const content = document.getElementById('suite-preview-content');
  content.innerHTML = '<div class="detail-empty">Loading…</div>';
  try {
    const suite = await api('/v1/eval/suites/' + encodeURIComponent(name));
    const examples = suite.examples || [];
    const preview = examples.slice(0, 20);
    document.getElementById('suite-preview-meta').innerHTML =
      `${examples.length} example${examples.length === 1 ? '' : 's'}` +
      (suite.default_scorer ? ` · <span class="scorer-badge">${escapeHtml(suite.default_scorer.kind || 'scorer')}</span>` : '');
    if (!preview.length) {
      content.innerHTML = '<div class="detail-empty">This suite has no examples.</div>';
      return;
    }
    const rows = preview.map((ex, i) => {
      const msgs = (ex.messages || [])
        .map(m => `<div style="margin-bottom:4px;"><span class="role ${escapeHtml(m.role)}" style="font-size:10px; text-transform:uppercase; letter-spacing: var(--tracking-caps); color:var(--text-muted); margin-right:6px;">${escapeHtml(m.role)}</span><span style="white-space:pre-wrap; font-family:var(--font-mono); font-size:12px;">${escapeHtml(truncate(m.content || '', 600))}</span></div>`)
        .join('');
      const target = ex.target != null
        ? `<div style="margin-top:6px;"><span class="hint" style="font-size:11px;">target:</span> <code style="font-family:var(--font-mono); font-size:12px;">${escapeHtml(truncate(String(ex.target), 200))}</code></div>`
        : '';
      const tags = (ex.tags || []).map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('');
      return `<div style="border:1px solid var(--border); border-radius:var(--radius-md); padding:var(--space-3); margin-bottom:var(--space-3); background:var(--surface-2);">
        <div style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono); margin-bottom:6px;">#${i + 1}${ex.id ? ' · ' + escapeHtml(ex.id) : ''}</div>
        ${msgs}
        ${target}
        ${tags ? `<div style="margin-top:6px;">${tags}</div>` : ''}
      </div>`;
    }).join('');
    const more = examples.length > preview.length
      ? `<div class="hint" style="text-align:center; padding:var(--space-3);">…showing first ${preview.length} of ${examples.length}. The Run action evaluates all of them.</div>`
      : '';
    content.innerHTML = `<div style="padding:var(--space-4) var(--space-5); overflow-y:auto;">${rows}${more}</div>`;
  } catch (e) {
    content.innerHTML = `<div class="detail-empty">Failed to load suite: ${escapeHtml(e.message)}</div>`;
  }
}

/* ---------- Jobs ---------- */

let evalJobsFilter = { query: '', state: 'all' };
function matchesEvalJobsFilter(j) {
  const q = (evalJobsFilter.query || '').trim().toLowerCase();
  if (q) {
    const hay = [
      j.suite_name || '',
      j.job_id || '',
      ...(j.adapters || []).map(a => a || 'base'),
    ].join(' ').toLowerCase();
    if (!hay.includes(q)) return false;
  }
  const st = (j.state || '').toString().toLowerCase();
  if (evalJobsFilter.state === 'running') return st === 'queued' || st === 'running';
  if (evalJobsFilter.state === 'completed') return st === 'completed';
  if (evalJobsFilter.state === 'failed') return st === 'failed' || st === 'cancelled';
  return true;
}
async function refreshEvalJobs() {
  try {
    const d = await api('/v1/eval/jobs');
    const jobs = d.jobs || [];
    evalJobsCache = jobs;
    // Lifecycle counts as JS state for the flywheel (and any other consumer):
    // data must not round-trip through a badge's rendered textContent.
    const stateOf = j => (j.state || '').toString().toLowerCase();
    evalJobCounts = {
      completed: jobs.filter(j => stateOf(j) === 'completed').length,
      running: jobs.filter(j => stateOf(j) === 'running').length,
      queued: jobs.filter(j => stateOf(j) === 'queued').length,
    };
    detectEvalTransitions(jobs);
    // Adapter cards show each adapter's latest eval score — refresh them now that
    // eval results changed (the dedup key includes the completed-eval signature).
    if (typeof refreshAdapterCards === 'function') refreshAdapterCards();
    // Header badge counts active jobs (queued + running), mirroring the
    // training badge — total job history is shown inside the tab so the
    // badge should signal "needs attention now", not "lifetime count".
    const liveCount = evalJobCounts.running + evalJobCounts.queued;
    setText('evals-count', String(liveCount));
    const evalsBadge = document.getElementById('evals-count');
    if (evalsBadge) evalsBadge.title = `${liveCount} eval job${liveCount === 1 ? '' : 's'} queued or running`;
    // The flywheel's Eval node reads evalJobCounts — repaint it now instead of
    // waiting for the next training/requests poll tick.
    updateFlywheel();
    const el = document.getElementById('eval-jobs-list');
    const filtered = jobs.filter(matchesEvalJobsFilter);
    if (jobs.length && !filtered.length) {
      el.className = 'eval-empty';
      if (setListHtml(el, 'nomatch', `<div class="eval-empty-body">No eval jobs match the current filter. <button class="btn btn-sm" type="button" data-eval-jobs-filter="all">Clear filter</button></div>`)) {
        el.querySelectorAll('[data-eval-jobs-filter]').forEach(btn => {
          btn.addEventListener('click', () => {
            document.querySelectorAll('[data-eval-jobs-filter]').forEach(b => b.classList.toggle('active', b.dataset.evalJobsFilter === 'all'));
            evalJobsFilter.state = 'all';
            const inp = document.getElementById('eval-jobs-filter');
            if (inp) inp.value = '';
            evalJobsFilter.query = '';
            refreshEvalJobs();
          });
        });
      }
      return;
    }
    if (!jobs.length) {
      el.className = 'eval-empty';
      setListHtml(el, 'empty', `
        <div class="eval-empty-icon"><svg class="icn"><use href="#i-chart"></use></svg></div>
        <div class="eval-empty-title">No eval jobs yet</div>
        <div class="eval-empty-body">Run a suite from the Suites tab. Jobs land here as they complete; click any job to drill into the per-example outcomes.</div>
        <button class="eval-empty-cta" type="button" onclick="document.getElementById('evals-tab-suites').click()">Browse suites</button>`);
      return;
    }
    el.className = '';
    // Key on the active filter (query + state pill) plus every field a job
    // card displays — id/state, headline accuracy, the whole progress object
    // (examples_completed/total, running accuracy/mean), per-run metrics and
    // tag pass-rates, and the error line. The filter belongs in the key so a
    // filter keystroke always repaints even when it yields the same set.
    const listKey = 'jobs:' + JSON.stringify([
      evalJobsFilter.query, evalJobsFilter.state,
      filtered.map(j => [
        j.job_id, j.state, j.suite_name, j.adapters, j.submission_kind,
        j.headline_accuracy, j.progress, j.error,
        (j.finished_runs || []).map(r => [r.adapter, r.metrics]),
      ]),
    ]);
    if (setListHtml(el, listKey, filtered.map(j => renderJobCard(j)).join(''))) {
      el.querySelectorAll('.job-card').forEach(card => {
        card.addEventListener('click', () => openDrillModal(card.dataset.jobId));
      });
    }
  } catch (e) {
    // Error-specific key: the recovered list (even an identical empty
    // payload) compares unequal and repaints (#1547 regression shape).
    setListHtml(document.getElementById('eval-jobs-list'), 'err:' + e.message,
      `<div class="eval-empty"><div class="eval-empty-body">Failed: ${escapeHtml(e.message)}</div></div>`);
  }
  // Refreshing jobs also updates suite sparklines.
  if (document.getElementById('evals-tab-suites')?.classList.contains('active')) {
    refreshSuites();
  }
}

// The flywheel's headline answer: did the adapter beat base, and by how much?
// Reads the per-run accuracies (base run keyed by adapter==null) and renders the
// existing-but-unused .delta-badge so the verdict isn't left as mental math.
// Eval completions are announced with the VERDICT attached — the number the
// user queued the job to learn, delivered instead of buried in the Jobs tab.
let prevEvalStates = null;
function detectEvalTransitions(jobs) {
  const now = new Map();
  (jobs || []).forEach(j => now.set(j.job_id, (j.state || '').toString().toLowerCase()));
  if (prevEvalStates) {
    for (const [id, state] of now) {
      const prev = prevEvalStates.get(id);
      if (!prev || prev === state || (prev !== 'running' && prev !== 'queued')) continue;
      const j = (jobs || []).find(x => x.job_id === id) || {};
      const suite = j.suite_name || 'eval';
      if (state === 'completed') {
        let verdict = '';
        // Same gate as the adapter card: win/loss phrasing only when the
        // paired sign test clears SIGN_TEST_ALPHA; otherwise the toast stays
        // neutral. One verdict per candidate — no best-of-N reduce(max).
        const verdicts = gatedCompareVerdicts(j.finished_runs || []);
        if (verdicts.length) {
          const phrase = (v) => v.significant
            ? (Math.abs(v.delta) <= 0.5
              ? `matches base (${fmtSignTestP(v.p)})`
              : `${v.delta > 0 ? '+' : ''}${v.delta.toFixed(1)} pts vs base (${fmtSignTestP(v.p)})`)
            : `no significant difference vs base (${fmtSignTestP(v.p)})`;
          verdict = verdicts.length === 1
            ? ` Verdict: ${phrase(verdicts[0])}.`
            : ` Verdicts: ${verdicts.map(v => `${v.candidate} ${phrase(v)}`).join('; ')}.`;
        } else if (typeof j.headline_accuracy === 'number') {
          verdict = ` Accuracy: ${(j.headline_accuracy * 100).toFixed(0)}%.`;
        }
        announceStatus('eval-jobs-status', `Eval ${suite} finished.${verdict}`);
        actionToast(`Eval ${suite} finished.${verdict}`, 'ok', [
          { label: 'View result', onClick: () => { selectPage('evals'); document.getElementById('evals-tab-jobs')?.click(); setTimeout(() => openDrillModal(id), 250); } },
        ]);
      } else if (state === 'failed') {
        announceStatus('eval-jobs-status', `Eval ${suite} failed.`);
        actionToast(`Eval ${suite} failed${j.error ? ': ' + String(j.error).slice(0, 80) : ''}.`, 'err', [
          { label: 'View job', onClick: () => { selectPage('evals'); document.getElementById('evals-tab-jobs')?.click(); } },
        ]);
      }
    }
  }
  prevEvalStates = now;
}

function compareVerdictBadge(runs) {
  // Same gate as the adapter card (gatedCompareVerdicts): a colored win/loss
  // badge only renders at p < SIGN_TEST_ALPHA; below that it's the neutral
  // "not enough evidence" treatment. One badge per candidate — no picking the
  // max of N candidates (best-of-N selection bias dressed up as a verdict).
  const verdicts = gatedCompareVerdicts(runs);
  if (!verdicts.length) return '';
  const multi = verdicts.length > 1;
  return verdicts.map(v => {
    const name = multi ? `${escapeHtml(v.candidate)}: ` : '';
    const title = `${escapeHtml(v.candidate)} ${(v.accuracy * 100).toFixed(0)}% vs base ${(v.baseAccuracy * 100).toFixed(0)}% — sign test improved ${v.improved} / regressed ${v.regressed}, ${fmtSignTestP(v.p)}`;
    if (!v.significant && Math.abs(v.delta) > 0.5) {
      return `<span class="delta-badge delta-flat" title="${title}">${name}${v.delta > 0 ? '+' : ''}${v.delta.toFixed(1)} pts — not enough evidence (${fmtSignTestP(v.p)})</span>`;
    }
    const cls = v.delta > 0.5 ? 'delta-up' : (v.delta < -0.5 ? 'delta-down' : 'delta-flat');
    const label = cls === 'delta-flat' ? 'matches base' : `${v.delta > 0 ? '+' : ''}${v.delta.toFixed(1)} pts vs base`;
    return `<span class="delta-badge ${cls}" title="${title}">${name}${label}</span>`;
  }).join('');
}

// Non-completed jobs have no score — show a state figure, never a giant "0"
// (which reads as "lost to base" and obscures the actual win answer).
function jobStateFigure(stateClass) {
  const g = stateClass === 'running' ? 'activity' : stateClass === 'queued' ? 'play' : stateClass === 'failed' ? 'warning' : 'activity';
  return `<span class="job-statefig ${stateClass}" aria-hidden="true">${icon(g)}</span>`;
}

function renderJobCard(j) {
  const acc = j.headline_accuracy;
  const adapters = (j.adapters || []).map(a => a == null ? '<span class="hint">base</span>' : escapeHtml(a)).join(' vs ');
  const stateClass = (j.state || 'queued').toLowerCase();
  const showRing = stateClass === 'completed' && typeof acc === 'number' && isFinite(acc);
  const progress = j.progress || {};
  const progFrac = progress.examples_total > 0 ? progress.examples_completed / progress.examples_total : 0;
  const isRunning = stateClass === 'running' || stateClass === 'queued';

  // Compact tag bars for the most-recent finished run (max 3)
  let tagSummary = '';
  if (j.finished_runs && j.finished_runs.length > 0) {
    const lastRun = j.finished_runs[j.finished_runs.length - 1];
    const rates = Object.entries(lastRun.metrics?.pass_rate_by_tag || {}).slice(0, 3);
    if (rates.length) {
      tagSummary = `<div style="display:flex; gap:8px; margin-top:6px; flex-wrap:wrap; font-size:11px;">`
        + rates.map(([k, v]) => `<span class="tag-chip">${escapeHtml(k)} ${(v*100).toFixed(0)}%</span>`).join('') + `</div>`;
    }
  }

  let progressOrCounts = '';
  if (isRunning) {
    progressOrCounts = `
      <div class="job-card-progress">
        <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${(progFrac*100).toFixed(1)}%;"></div></div>
        <span class="tabular-nums hint" style="font-size:11px;">${progress.examples_completed || 0}/${progress.examples_total || 0}</span>
      </div>`;
    if ((progress.examples_completed || 0) > 0) {
      progressOrCounts += `<div class="hint" style="font-size:11px; margin-top:4px;">running ${(progress.running_accuracy*100).toFixed(0)}% accuracy · mean ${(progress.running_mean_score).toFixed(2)}</div>`;
    }
  } else if (j.finished_runs && j.finished_runs.length > 0) {
    // Per-run mini bars when compare-mode
    const runsHtml = j.finished_runs.map(r => {
      const a = r.adapter || 'base';
      return `<span class="hint" style="font-size:11px; display:inline-flex; gap:4px; align-items:center; margin-right:10px;">
        <strong>${escapeHtml(a)}</strong>: <span class="tabular-nums">${(r.metrics.accuracy*100).toFixed(0)}%</span>
        <span class="hint" style="font-size:10px;">(${r.metrics.num_pass}/${r.metrics.num_examples})</span>
      </span>`;
    }).join('');
    const verdict = compareVerdictBadge(j.finished_runs);
    progressOrCounts = `<div style="margin-top:6px; display:flex; align-items:center; gap:8px; flex-wrap:wrap;">${runsHtml}${verdict}</div>`;
  } else if (j.error) {
    progressOrCounts = `<div class="hint" style="color:var(--danger-fg); margin-top:4px;">${escapeHtml(j.error)}</div>`;
  }

  return `<div class="job-card" data-job-id="${escapeHtml(j.job_id)}">
    ${showRing ? ringHtml(acc, 'large') : jobStateFigure(stateClass)}
    <div class="job-card-meta">
      <div class="job-card-suite">${escapeHtml(j.suite_name)}</div>
      <div class="job-card-sub">
        <span class="job-state-pill ${stateClass}">${escapeHtml(j.state || '')}</span>
        <span>${adapters}</span>
        <span class="hint">${escapeHtml(j.submission_kind)}</span>
        <span class="hint" style="font-family:var(--font-mono);">${escapeHtml(j.job_id.slice(0, 8))}</span>
      </div>
      ${progressOrCounts}
      ${tagSummary}
    </div>
  </div>`;
}

/* ---------- Drill-in modal ---------- */

let drillJob = null;
// The job id the drill modal is showing (set before the fetch lands, unlike
// drillJob) — the deep-link router diffs against it.
let evalDrillJobId = null;
let drillFilter = 'all';
let drillSearch = '';
let drillSelectedRun = 0;
let drillSelectedOutcome = null;
let drillPollHandle = null;
// Map of example_id → { messages, target, scorer, weight, tags } for the
// suite the current drill job ran. Lets the detail panel show the prompt
// the model actually saw, not just the model's reply. Cached per-suite.
let drillExamplesById = new Map();
let drillSuiteCacheKey = null;

async function openDrillModal(jobId) {
  evalDrillJobId = jobId;
  modalHashOnOpen('eval', '#evals/jobs/' + encodeURIComponent(jobId));
  drillFilter = 'all';
  drillSearch = '';
  drillSelectedRun = 0;
  drillSelectedOutcome = null;
  document.getElementById('drill-search').value = '';
  document.querySelectorAll('[data-drill-filter]').forEach(b => b.classList.toggle('active', b.dataset.drillFilter === 'all'));
  // A leftover raw-JSON block from a previously drilled job would show the
  // wrong payload until the user re-toggles — drop it on every open.
  document.getElementById('drill-raw-block')?.remove();
  const modal = document.getElementById('eval-drill-modal');
  modal.hidden = false;
  openModal(modal, { onClose: userCloseDrillModal });
  await fetchDrillJob(jobId);
  // If the job is still running, poll every second so the modal updates live.
  drillPollHandle = setInterval(async () => {
    if (!drillJob) return;
    if (drillJob.state === 'running' || drillJob.state === 'queued') {
      await fetchDrillJob(drillJob.job_id, /*preserveSelection*/ true);
    }
  }, 1500);
}

function closeDrillModal() {
  const modal = document.getElementById('eval-drill-modal');
  modal.hidden = true;
  closeModal(modal);
  document.getElementById('drill-raw-block')?.remove();
  drillJob = null;
  evalDrillJobId = null;
  drillSelectedOutcome = null;
  drillSuiteCacheKey = null;
  drillExamplesById = new Map();
  if (drillPollHandle) { clearInterval(drillPollHandle); drillPollHandle = null; }
}
// User-initiated close (X / backdrop / Esc / Cancel-Delete): walk history per
// the deep-link state machine. "Replay in playground" and re-run keep calling
// closeDrillModal directly — they navigate FORWARD from the modal, so its
// entry should stay behind them for Back.
function userCloseDrillModal() {
  modalHashOnUserClose('eval', '#evals/jobs', closeDrillModal);
}

async function fetchDrillJob(jobId, preserveSelection = false) {
  try {
    const j = await api('/v1/eval/jobs/' + encodeURIComponent(jobId));
    const jobMeta = evalJobsCache.find(item => item.job_id === jobId);
    drillJob = {
      ...j,
      suite_name: jobMeta?.suite_name || j.runs?.[0]?.suite_name || 'eval',
      adapters: jobMeta?.adapters || j.runs?.map(r => r.adapter ?? null) || [],
      submission_kind: jobMeta?.submission_kind || 'on_demand',
    };
    // Lazily fetch the suite content the *first* time we draw a drill for
    // it. The outcomes don't carry the example prompts (only the model's
    // reply) — without this the user can't actually debug a failure.
    const suiteName = drillJob.suite_name;
    if (suiteName && drillSuiteCacheKey !== suiteName) {
      drillSuiteCacheKey = suiteName;
      drillExamplesById = new Map();
      try {
        const suite = await api('/v1/eval/suites/' + encodeURIComponent(suiteName));
        // EvalExample.id is optional — when omitted the server uses a
        // sha256 prefix derived from messages+target+aliases. We mirror
        // the algorithm here so the outcome's example_id keys back to
        // the right prompt. Hashing is async (crypto.subtle), so we
        // resolve all of them in parallel.
        const examples = suite.examples || [];
        const ids = await Promise.all(examples.map(ex => ex.id ? Promise.resolve(ex.id) : hashExampleId(ex)));
        for (let i = 0; i < examples.length; i++) {
          drillExamplesById.set(ids[i], examples[i]);
        }
      } catch (_) {
        // Inline-suite jobs aren't registered, so this 404s — we degrade
        // to no-prompt mode silently.
      }
    }
    renderDrillModal(preserveSelection);
  } catch (e) {
    toast('Failed to load job: ' + e.message, 'err');
    // userClose (not plain close): consumes/repairs the hash entry too, so a
    // junk #evals/jobs/{id} deep link degrades to #evals/jobs cleanly.
    userCloseDrillModal();
  }
}

/// Recompute the same example ID the server uses when one is not provided.
/// Mirrors `EvalExample::resolved_id` in kiln-eval/src/suite.rs (sha256
/// over role|content|target|aliases, hex prefix of 8 bytes).
async function hashExampleId(ex) {
  const enc = new TextEncoder();
  const parts = [];
  for (const m of (ex.messages || [])) {
    parts.push(enc.encode(m.role));
    parts.push(new Uint8Array([0]));
    parts.push(enc.encode(m.content));
    parts.push(new Uint8Array([0]));
  }
  if (ex.target != null) {
    parts.push(enc.encode('|t|'));
    parts.push(enc.encode(ex.target));
  }
  for (const a of (ex.aliases || [])) {
    parts.push(enc.encode('|a|'));
    parts.push(enc.encode(a));
  }
  const total = parts.reduce((s, p) => s + p.length, 0);
  const buf = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { buf.set(p, off); off += p.length; }
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(digest, 0, 8)).map(b => b.toString(16).padStart(2, '0')).join('');
}

function renderDrillModal(preserveSelection) {
  const j = drillJob;
  if (!j) return;
  document.getElementById('drill-title').textContent = j.suite_name || 'Eval results';
  document.getElementById('drill-meta').innerHTML = `
    <span class="job-state-pill ${j.state}">${escapeHtml(j.state)}</span>
    <span class="hint" style="margin-left:8px; font-family:var(--font-mono);">${escapeHtml(j.job_id)}</span>`;
  // Cancel / Delete: same DELETE endpoint, different copy. Active jobs
  // get cancelled; terminal jobs get deleted from memory + archive.
  const stateLower = (j.state || '').toString().toLowerCase();
  const isActive = stateLower === 'queued' || stateLower === 'running';
  const cancelBtn = document.getElementById('drill-cancel');
  if (cancelBtn) {
    cancelBtn.hidden = false;
    cancelBtn.innerHTML = isActive ? icon('stop','icn-sm') + ' Cancel' : icon('trash','icn-sm') + ' Delete';
    cancelBtn.title = isActive
      ? 'Cancel this eval job (queued or running)'
      : 'Permanently delete this terminal job from memory and the on-disk archive';
    cancelBtn.dataset.mode = isActive ? 'cancel' : 'delete';
  }
  const rerunBtn = document.getElementById('drill-rerun');
  if (rerunBtn) {
    const failingInAnyRun = (j.runs || []).some(r =>
      (r.outcomes || []).some(o => o.kind !== 'pass'));
    rerunBtn.hidden = isActive || !failingInAnyRun;
  }
  // Download outcomes (.jsonl): live across every run of the job (compare
  // jobs export all adapters, one line per outcome). Disabled until the
  // first outcome lands so the click never produces an empty file.
  const exportBtn = document.getElementById('drill-download-outcomes');
  if (exportBtn) {
    const outcomeCount = (j.runs || []).reduce((n, r) => n + (r.outcomes || []).length, 0);
    exportBtn.disabled = outcomeCount === 0;
    exportBtn.title = outcomeCount
      ? `Download all ${outcomeCount} per-example outcomes across ${(j.runs || []).length} run(s) as JSON Lines`
      : 'No outcomes yet — the download unlocks as examples finish';
  }

  const runs = j.runs || [];
  const isCompare = (j.adapters && j.adapters.length > 1) || runs.length > 1;
  const headerEl = document.getElementById('drill-headline');
  const compareEl = document.getElementById('drill-compare');
  const tagsEl = document.getElementById('drill-tags');

  if (runs.length === 0) {
    headerEl.innerHTML = `
      <div class="hint">${j.state === 'queued' ? 'Job is queued. Will start shortly.' : (j.state === 'running' ? 'Job is running. Live progress streaming…' : 'No completed runs yet.')}</div>
      ${j.progress && j.progress.examples_total > 0 ? `<div style="flex:1;"><div class="progress-bar-wrap" style="height:8px;"><div class="progress-bar-fill" style="width:${(j.progress.examples_completed / j.progress.examples_total * 100).toFixed(1)}%;"></div></div><div class="hint" style="font-size:11px; margin-top:4px;">${j.progress.examples_completed}/${j.progress.examples_total} · running ${(j.progress.running_accuracy*100).toFixed(0)}%</div></div>` : ''}`;
    compareEl.hidden = true;
    tagsEl.hidden = true;
    document.getElementById('drill-outcomes').innerHTML = '<div class="eval-empty"><div class="eval-empty-body">Outcomes will appear here as they complete.</div></div>';
    document.getElementById('drill-detail').innerHTML = '<div class="detail-empty">Waiting on first results…</div>';
    updateDrillFilterCounts([]);
    return;
  }

  // Headline shows the *selected* run (default first).
  const run = runs[Math.min(drillSelectedRun, runs.length - 1)];
  const adapter = run.adapter || 'base';
  const m = run.metrics || {};
  headerEl.innerHTML = `
    ${ringHtml(m.accuracy, 'large')}
    <div style="flex:1; min-width:0;">
      <div style="font-size:14px; font-weight:600; margin-bottom:6px;">Adapter: <span style="color:var(--text);">${escapeHtml(adapter)}</span></div>
      <div class="drill-counts">
        <div class="count-cell"><span class="count-num" style="color:var(--success-fg);">${m.num_pass || 0}</span><span class="count-label">pass</span></div>
        <div class="count-cell"><span class="count-num" style="color:var(--danger-fg);">${m.num_fail || 0}</span><span class="count-label">fail</span></div>
        <div class="count-cell"><span class="count-num" style="color:var(--warning-fg);">${m.num_invalid || 0}</span><span class="count-label">invalid</span></div>
        <div class="count-cell"><span class="count-num" style="color:var(--text-muted);">${m.num_error || 0}</span><span class="count-label">error</span></div>
        ${m.latency && m.latency.p50_ms > 0 ? `<div class="count-cell"><span class="count-num">${m.latency.p50_ms.toFixed(0)}ms</span><span class="count-label">p50</span></div>` : ''}
        ${m.total_completion_tokens ? `<div class="count-cell"><span class="count-num">${(m.total_completion_tokens/1000).toFixed(1)}k</span><span class="count-label">tok out</span></div>` : ''}
      </div>
    </div>`;

  // Compare matrix when multi-run
  if (isCompare && runs.length >= 2) {
    compareEl.hidden = false;
    const total = (run.metrics?.num_examples || 1);
    const verdictBadge = compareVerdictBadge(runs);
    compareEl.innerHTML = `<div class="eval-section-head" style="background:transparent; border:none; padding:0 0 4px 0; display:flex; align-items:center; gap:10px;">Adapter comparison${verdictBadge}</div>` +
      runs.map((r, i) => {
        const rm = r.metrics || {};
        const tot = Math.max(1, rm.num_examples || total);
        const pp = (rm.num_pass || 0) / tot * 100;
        const fp = (rm.num_fail || 0) / tot * 100;
        const ip = (rm.num_invalid || 0) / tot * 100;
        const ep = (rm.num_error || 0) / tot * 100;
        const a = r.adapter || 'base';
        const isSel = i === drillSelectedRun;
        return `<div class="compare-row" data-run-idx="${i}" style="cursor:pointer; ${isSel ? 'opacity:1;' : 'opacity:0.7;'}" title="Click to view this adapter's outcomes">
          <span class="compare-name" ${isSel ? 'style="color:var(--accent);"' : ''}>${escapeHtml(a)}</span>
          <div class="compare-bar">
            <div class="seg-pass" style="width:${pp}%;" title="pass ${rm.num_pass}"></div>
            <div class="seg-fail" style="width:${fp}%;" title="fail ${rm.num_fail}"></div>
            <div class="seg-invalid" style="width:${ip}%;" title="invalid ${rm.num_invalid}"></div>
            <div class="seg-error" style="width:${ep}%;" title="error ${rm.num_error}"></div>
          </div>
          <span class="compare-acc">${(rm.accuracy*100).toFixed(0)}%</span>
        </div>`;
      }).join('');
    compareEl.querySelectorAll('.compare-row').forEach(row => {
      row.addEventListener('click', () => {
        drillSelectedRun = parseInt(row.dataset.runIdx, 10);
        drillSelectedOutcome = null;
        renderDrillModal(false);
      });
    });
  } else {
    compareEl.hidden = true;
  }

  // Tag pass-rate bars
  const tagRates = m.pass_rate_by_tag || {};
  const tagEntries = Object.entries(tagRates);
  if (tagEntries.length) {
    tagsEl.hidden = false;
    tagsEl.innerHTML = '<div class="eval-section-head" style="background:transparent; border:none; padding:0 0 4px 0;">Pass rate by tag</div>' +
      tagEntries.map(([k, v]) => `<div class="tag-bar">
        <span class="tag-name">${escapeHtml(k)}</span>
        <div class="tag-track"><div class="tag-fill" style="width:${(v*100).toFixed(1)}%;"></div></div>
        <span class="tag-pct">${(v*100).toFixed(0)}%</span>
      </div>`).join('');
  } else {
    tagsEl.hidden = true;
  }

  // Outcomes list, filtered + searched
  renderDrillOutcomes();
  if (!preserveSelection || drillSelectedOutcome === null) {
    // Default: show first failure if any, else first outcome
    const first = (run.outcomes || []).find(o => o.kind !== 'pass') || (run.outcomes || [])[0];
    if (first) selectDrillOutcome(first);
    else document.getElementById('drill-detail').innerHTML = '<div class="detail-empty">No outcomes for this run.</div>';
  } else {
    // Re-find the selected outcome by id (it may have changed kind on a re-poll)
    const found = (run.outcomes || []).find(o => o.example_id === drillSelectedOutcome.example_id && o.completion_index === drillSelectedOutcome.completion_index);
    if (found) selectDrillOutcome(found);
  }
}

function renderDrillOutcomes() {
  const j = drillJob;
  if (!j || !j.runs) return;
  const run = j.runs[Math.min(drillSelectedRun, j.runs.length - 1)];
  const all = run.outcomes || [];
  // Counts always reflect the whole run (so the filter pills are stable)
  const counts = { all: all.length, pass: 0, fail: 0, invalid: 0, error: 0 };
  for (const o of all) counts[o.kind] = (counts[o.kind] || 0) + 1;
  updateDrillFilterCounts(counts);

  const filtered = all.filter(o => {
    if (drillFilter !== 'all' && o.kind !== drillFilter) return false;
    if (drillSearch) {
      const needle = drillSearch.toLowerCase();
      const hay = (o.example_id + ' ' + (o.completion_text || '') + ' ' + (o.detail || '')).toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    return true;
  });
  const el = document.getElementById('drill-outcomes');
  if (!filtered.length) {
    el.innerHTML = '<div class="eval-empty" style="border:none; background:transparent;"><div class="eval-empty-body">No examples match the current filter.</div></div>';
    return;
  }
  el.innerHTML = filtered.map(o => {
    const tags = (o.tags || []).slice(0, 2).map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('');
    const isSel = drillSelectedOutcome
      && drillSelectedOutcome.example_id === o.example_id
      && drillSelectedOutcome.completion_index === o.completion_index;
    return `<div class="outcome-item ${isSel ? 'selected' : ''}" data-example-id="${escapeHtml(o.example_id)}" data-completion-index="${o.completion_index}">
      <span class="outcome-badge ${o.kind}">${o.kind}</span>
      <div class="outcome-preview" title="${escapeHtml(o.completion_text || '')}">${escapeHtml(truncate(o.completion_text || '(empty)', 110))}</div>
      <div class="outcome-meta">
        ${o.latency_ms != null ? `<span class="hint">${o.latency_ms.toFixed(0)}ms</span>` : ''}
        ${tags}
      </div>
    </div>`;
  }).join('');
  el.querySelectorAll('.outcome-item').forEach(item => {
    item.addEventListener('click', () => {
      const id = item.dataset.exampleId;
      const idx = parseInt(item.dataset.completionIndex, 10);
      const found = all.find(o => o.example_id === id && o.completion_index === idx);
      if (found) selectDrillOutcome(found);
    });
  });
}

function updateDrillFilterCounts(counts) {
  for (const k of ['all', 'pass', 'fail', 'invalid', 'error']) {
    const el = document.getElementById('drill-count-' + k);
    if (el) el.textContent = (counts[k] || 0).toLocaleString();
  }
}

function selectDrillOutcome(o) {
  drillSelectedOutcome = { example_id: o.example_id, completion_index: o.completion_index };
  // Highlight in list
  document.querySelectorAll('.outcome-item').forEach(item => {
    const match = item.dataset.exampleId === o.example_id && parseInt(item.dataset.completionIndex, 10) === o.completion_index;
    item.classList.toggle('selected', match);
  });
  renderOutcomeDetail(o);
}

function renderOutcomeDetail(o) {
  const tags = (o.tags || []).map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('');
  const detail = document.getElementById('drill-detail');
  const example = drillExamplesById.get(o.example_id);
  // Prompt section: the chat history the model actually saw. When we have
  // it (suite was loadable + example_id matched), render each message as a
  // role-coded bubble; otherwise show a hint that the suite isn't available.
  const promptSection = example && example.messages && example.messages.length
    ? `<div class="detail-section">
        <h4>Prompt</h4>
        <div class="messages-list">
          ${example.messages.map(m => `<div class="chat-msg">
            <div class="role ${escapeHtml(m.role)}">${escapeHtml(m.role)}</div>
            <div class="body">${escapeHtml(m.content)}</div>
          </div>`).join('')}
        </div>
      </div>`
    : `<div class="detail-section">
        <h4>Prompt</h4>
        <div class="hint" style="font-size:11px;">Suite content not available locally — drill into a registered suite to see the prompt the model saw.</div>
      </div>`;
  // Side-by-side target vs got. Only render when we have a target; the
  // "json_validity" / "any_block" scorers don't always set one.
  const target = example && example.target;
  const passClass = o.kind === 'pass' ? 'tg-pass' : (o.kind === 'fail' ? 'tg-fail' : '');
  const tgSection = target != null
    ? `<div class="detail-section">
        <h4>Target ↔ Got</h4>
        <div class="detail-tg">
          <div class="tg-cell">
            <div class="tg-label">Expected target</div>
            <pre>${escapeHtml(target)}</pre>
          </div>
          <div class="tg-cell ${passClass}">
            <div class="tg-label">Model output</div>
            <pre>${escapeHtml(o.completion_text || '(empty)')}</pre>
          </div>
        </div>
      </div>`
    : `<div class="detail-section">
        <h4>Model output</h4>
        <pre class="${passClass}" style="margin:0; font-family:var(--font-mono); font-size:12px; line-height:1.55; white-space:pre-wrap; word-break:break-word; padding:10px; background:var(--surface); border:1px solid var(--border); border-radius:6px;">${escapeHtml(o.completion_text || '(empty)')}</pre>
      </div>`;
  // Scorer section: kind + per-example detail
  const scorerKind = (example && example.scorer && example.scorer.kind) || drillJob?.runs?.[0]?.metrics?.by_scorer?.[0]?.scorer_kind || '';
  const scorerSection = `<div class="detail-section">
    <h4>Scorer</h4>
    <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
      ${scorerKind ? `<span class="scorer-badge">${escapeHtml(scorerKind)}</span>` : ''}
      <span class="tabular-nums hint">score ${(o.score).toFixed(3)}</span>
    </div>
    ${o.detail ? `<div style="font-family:var(--font-mono); font-size:12px; padding:10px; background:var(--surface); border:1px solid var(--border); border-radius:6px;">${escapeHtml(o.detail)}</div>` : '<div class="hint" style="font-size:11px;">No scorer commentary.</div>'}
  </div>`;
  // Per-outcome action toolbar: copy raw outputs, replay the failing
  // prompt in the playground. Stash the prompt text on a dataset attribute
  // so the click handler doesn't have to walk back through drillExamplesById.
  const promptForReplay = example && example.messages && example.messages.length
    ? example.messages
        .filter(m => m.role !== 'system')
        .map(m => `[${m.role}] ${m.content}`)
        .join('\n\n')
    : '';
  const userMsg = example && example.messages
    ? (example.messages.filter(m => m.role === 'user').pop()?.content || '')
    : '';
  const actionsHtml = `<div class="outcome-actions" style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px;">
    <button type="button" class="btn btn-sm" data-outcome-copy="completion" title="Copy the model's output"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg> Copy output</button>
    ${promptForReplay ? `<button type="button" class="btn btn-sm" data-outcome-copy="prompt" title="Copy the full prompt as role-prefixed text"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-copy"></use></svg> Copy prompt</button>` : ''}
    ${userMsg ? `<button type="button" class="btn btn-sm" data-outcome-replay="1" title="Drop the last user message into the playground so you can iterate"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-chat"></use></svg> Replay in playground</button>` : ''}
    ${(o.kind === 'fail' || o.kind === 'invalid') && userMsg ? `<button type="button" class="btn btn-sm" data-outcome-correct="1" title="Capture this failing example into the corrections basket — write the ideal answer, then train"><svg class="icn icn-sm" aria-hidden="true"><use href="#i-pencil"></use></svg> Add to corrections</button>` : ''}
  </div>`;
  detail.innerHTML = `
    <div class="detail-section">
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">
        <span class="outcome-badge ${o.kind}">${o.kind}</span>
        <strong style="font-family:var(--font-mono); font-size:13px;">${escapeHtml(o.example_id)}</strong>
        ${o.completion_index > 0 ? `<span class="hint">completion #${o.completion_index}</span>` : ''}
        ${o.latency_ms != null ? `<span class="tabular-nums hint" style="font-size:11px;">${o.latency_ms.toFixed(0)}ms</span>` : ''}
        ${o.prompt_tokens != null ? `<span class="tabular-nums hint" style="font-size:11px;">${o.prompt_tokens}→${o.completion_tokens || 0} tok</span>` : ''}
      </div>
      ${actionsHtml}
      <div>${tags}</div>
    </div>
    ${promptSection}
    ${tgSection}
    ${scorerSection}
  `;
  // Wire the per-outcome action buttons.
  detail.querySelectorAll('[data-outcome-copy]').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.outcomeCopy;
      const text = key === 'completion' ? (o.completion_text || '') : promptForReplay;
      navigator.clipboard.writeText(text).then(() => toast(`Copied ${key}`, 'ok'), () => toast('Copy failed', 'err'));
    });
  });
  detail.querySelectorAll('[data-outcome-correct]').forEach(btn => {
    btn.addEventListener('click', () => {
      addCorrectionFromEvalOutcome(o, example, scorerKind);
    });
  });
  detail.querySelectorAll('[data-outcome-replay]').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.getElementById('chat-input');
      if (input) {
        input.value = userMsg;
        if (typeof autoresizeChatInput === 'function') autoresizeChatInput();
        if (typeof updateChatSendState === 'function') updateChatSendState();
      }
      closeDrillModal();
      selectPage('playground');
      if (input) setTimeout(() => input.focus(), 50);
    });
  });
}

document.getElementById('drill-close')?.addEventListener('click', userCloseDrillModal);
document.getElementById('eval-drill-modal')?.addEventListener('click', ev => {
  if (ev.target.id === 'eval-drill-modal') userCloseDrillModal();
});
// Raw JSON toggle — same pattern as the request drill modal's `raw` button:
// click appends a pretty-printed <pre> of the cached job to the modal
// content, click again removes it.
document.getElementById('drill-raw')?.addEventListener('click', () => {
  if (!drillJob) return;
  const content = document.getElementById('drill-content');
  if (!content) return;
  const existing = content.querySelector('#drill-raw-block');
  if (existing) { existing.remove(); return; }
  const pre = document.createElement('pre');
  pre.id = 'drill-raw-block';
  pre.className = 'req-pre';
  pre.style.cssText = 'max-height:50vh; margin:var(--space-4) var(--space-5);';
  pre.textContent = JSON.stringify(drillJob, null, 2);
  content.appendChild(pre);
  pre.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
/// Trigger a browser download via a temporary object URL. The URL is
/// revoked right after the click so repeated downloads don't pin every
/// blob in memory for the lifetime of the page.
function downloadBlobAsFile(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
/// One JSON line per outcome, across every finished run of the job. Each
/// line is standalone: suite/job/adapter context first, then the outcome's
/// verdict and (when present) its optional diagnostics, completion last.
function buildDrillOutcomesJsonl(j) {
  const lines = [];
  for (const run of (j.runs || [])) {
    (run.outcomes || []).forEach((o, i) => {
      const line = {
        suite: run.suite_name || j.suite_name || 'eval',
        job_id: j.job_id,
        adapter: run.adapter || 'base',
        example_index: i,
        example_id: o.example_id,
        completion_index: o.completion_index,
        kind: o.kind,
        score: o.score,
      };
      if (o.detail != null) line.detail = o.detail;
      if (o.latency_ms != null) line.latency_ms = o.latency_ms;
      if (o.prompt_tokens != null) line.prompt_tokens = o.prompt_tokens;
      if (o.completion_tokens != null) line.completion_tokens = o.completion_tokens;
      if (o.tags && o.tags.length) line.tags = o.tags;
      if (o.metadata != null) line.metadata = o.metadata;
      if (o.reasoning_text != null) line.reasoning_text = o.reasoning_text;
      line.completion_text = o.completion_text || '';
      lines.push(JSON.stringify(line));
    });
  }
  return lines;
}
document.getElementById('drill-download-outcomes')?.addEventListener('click', () => {
  if (!drillJob) return;
  const lines = buildDrillOutcomesJsonl(drillJob);
  if (!lines.length) return; // button is disabled in this state; belt and braces
  const suiteSlug = String(drillJob.suite_name || 'eval').replace(/[^A-Za-z0-9._-]+/g, '-');
  const filename = `${suiteSlug}-${drillJob.job_id.slice(0, 8)}.outcomes.jsonl`;
  downloadBlobAsFile(filename, new Blob([lines.join('\n') + '\n'], { type: 'application/jsonl' }));
  toast(`Downloaded ${lines.length} outcome${lines.length === 1 ? '' : 's'} as ${filename}`, 'ok');
});
document.getElementById('drill-search')?.addEventListener('input', ev => {
  drillSearch = ev.target.value;
  renderDrillOutcomes();
});
document.querySelectorAll('[data-drill-filter]').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('[data-drill-filter]').forEach(other => other.classList.toggle('active', other === b));
    drillFilter = b.dataset.drillFilter;
    renderDrillOutcomes();
  });
});

document.getElementById('drill-cancel')?.addEventListener('click', async () => {
  if (!drillJob) return;
  const mode = document.getElementById('drill-cancel')?.dataset?.mode || 'cancel';
  const verbMsg = mode === 'delete'
    ? `Permanently delete eval job ${drillJob.job_id.slice(0, 8)}? Adapter weights are untouched; only the tracking entry and the on-disk archive file are removed.`
    : `Cancel eval job ${drillJob.job_id.slice(0, 8)}?`;
  if (!confirm(verbMsg)) return;
  try {
    await api('/v1/eval/jobs/' + encodeURIComponent(drillJob.job_id), { method: 'DELETE' });
    toast(mode === 'delete' ? 'Eval job deleted' : 'Cancelled eval job', 'ok');
    userCloseDrillModal();
    refreshEvalJobs();
  } catch (e) { toast((mode === 'delete' ? 'Delete' : 'Cancel') + ' failed: ' + e.message, 'err'); }
});

document.getElementById('drill-rerun')?.addEventListener('click', async () => {
  if (!drillJob) return;
  const failing = (drillJob.runs?.[Math.min(drillSelectedRun, (drillJob.runs?.length || 1) - 1)]?.outcomes || [])
    .filter(o => o.kind !== 'pass').length;
  if (!failing) {
    toast('No non-passing examples to re-run', 'ok');
    return;
  }
  if (!confirm(`Re-run ${failing} failing example(s)?`)) return;
  try {
    const res = await api('/v1/eval/jobs/' + encodeURIComponent(drillJob.job_id) + '/rerun', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({}),
    });
    toast('Queued re-run as ' + res.job_id.slice(0, 8), 'ok');
    closeDrillModal();
    refreshEvalJobs();
    setTimeout(() => openDrillModal(res.job_id), 200);
  } catch (e) { toast('Re-run failed: ' + e.message, 'err'); }
});

// Modal-scoped keyboard shortcuts: / focuses search; J/K scroll through
// outcomes (vim-style); R triggers re-run. Esc is the shared modal
// manager's (routes through userCloseDrillModal via the layer's onClose).
document.addEventListener('keydown', ev => {
  const modal = document.getElementById('eval-drill-modal');
  if (modal.hidden) return;
  // Only while this drill is the TOP modal — cmdk over it owns the keys.
  if (modalStackTop()?.el !== modal) return;
  const tag = (ev.target.tagName || '').toUpperCase();
  // When focused in an input, only Cmd/Ctrl shortcuts fire.
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (ev.key === '/') {
    ev.preventDefault();
    document.getElementById('drill-search').focus();
  } else if (ev.key === 'r' || ev.key === 'R') {
    ev.preventDefault();
    document.getElementById('drill-rerun').click();
  } else if (ev.key === 'j' || ev.key === 'ArrowDown') {
    ev.preventDefault();
    moveDrillSelection(1);
  } else if (ev.key === 'k' || ev.key === 'ArrowUp') {
    ev.preventDefault();
    moveDrillSelection(-1);
  }
});

function moveDrillSelection(delta) {
  const list = Array.from(document.querySelectorAll('.outcome-item'));
  if (!list.length) return;
  const cur = list.findIndex(el => el.classList.contains('selected'));
  const next = Math.max(0, Math.min(list.length - 1, (cur < 0 ? 0 : cur + delta)));
  list[next].click();
  list[next].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

/* ---------- Judgments — keyboard-first A/B with streaming ---------- */

let activeJudgmentDataset = null;
let pendingJudgmentPair = null;
let judgmentStreams = { a: null, b: null };  // AbortControllers
let judgmentAutoAdvance = true;

async function refreshJudgments() {
  try {
    const d = await api('/v1/judgments');
    const items = d.judgments || [];
    const el = document.getElementById('judgments-list');
    if (!items.length) {
      el.className = 'eval-empty';
      setListHtml(el, 'empty', `
        <div class="eval-empty-icon"><svg class="icn"><use href="#i-scale"></use></svg></div>
        <div class="eval-empty-title">No judgment datasets yet</div>
        <div class="eval-empty-body">Create a dataset, then judge model outputs A/B/Tie. After ~20 picks you can compile them into SFT data and train a local judge LoRA — no frontier LLM in the loop.</div>
        <button class="eval-empty-cta" type="button" onclick="document.getElementById('judgment-create-name').focus()">Create your first dataset</button>`);
    } else {
      el.className = '';
      // Key on the displayed payload fields plus activeJudgmentDataset —
      // the "(active)" hint and the Continue/Judge button label depend on
      // it, so switching datasets must repaint even with identical data.
      const listKey = 'list:' + JSON.stringify([
        activeJudgmentDataset,
        items.map(m => [m.name, m.description, m.num_rows, m.winner_histogram]),
      ]);
      const listHtml = items.map(m => {
        const winners = m.winner_histogram || {};
        const total = (winners.a || 0) + (winners.b || 0) + (winners.tie || 0);
        const aPct = total ? ((winners.a || 0) / total * 100).toFixed(0) : 0;
        const bPct = total ? ((winners.b || 0) / total * 100).toFixed(0) : 0;
        const isActive = activeJudgmentDataset === m.name;
        const winnerBar = total > 0 ? `<div style="display:flex; height:6px; border-radius:3px; overflow:hidden; background:var(--surface-3); width:120px;">
          <div style="background:var(--info-fg); width:${aPct}%;" title="A: ${winners.a || 0}"></div>
          <div style="background:var(--warning-fg); width:${total ? ((winners.tie || 0) / total * 100).toFixed(0) : 0}%;" title="Tie: ${winners.tie || 0}"></div>
          <div style="background:var(--accent); width:${bPct}%;" title="B: ${winners.b || 0}"></div>
        </div>` : '';
        return `<div class="eval-row eval-row-judgments">
          <div>
            <div class="row-title">${escapeHtml(m.name)}${isActive ? ' <span class="hint">(active)</span>' : ''}</div>
            <div class="row-sub">${escapeHtml(m.description || 'No description')}</div>
          </div>
          <div class="tabular-nums">${m.num_rows} judgments</div>
          <div style="display:flex; gap:8px; align-items:center; font-size:11px;">${winnerBar}<span class="hint">A ${winners.a||0} · T ${winners.tie||0} · B ${winners.b||0}</span></div>
          <div class="row-actions">
            <button type="button" class="btn btn-primary btn-sm" data-action="judge" data-name="${escapeHtml(m.name)}">${isActive ? 'Continue' : 'Judge →'}</button>
            <button type="button" class="btn btn-sm" data-action="promote" data-name="${escapeHtml(m.name)}">Promote</button>
            <button type="button" class="btn btn-sm" data-action="del" data-name="${escapeHtml(m.name)}">Delete</button>
          </div>
        </div>`;
      }).join('');
      if (setListHtml(el, listKey, listHtml)) {
        el.querySelectorAll('button[data-action]').forEach(b => {
          const name = b.dataset.name;
          if (b.dataset.action === 'judge') {
            b.addEventListener('click', () => openJudgmentViewer(name));
          } else if (b.dataset.action === 'promote') {
            b.addEventListener('click', () => openJudgmentCompile(name));
          } else if (b.dataset.action === 'del') {
            b.addEventListener('click', async () => {
              if (!confirm(`Delete judgment dataset "${name}"? Provenance is gone for good.`)) return;
              try {
                await api('/v1/judgments/' + encodeURIComponent(name), { method: 'DELETE' });
                if (activeJudgmentDataset === name) {
                  activeJudgmentDataset = null;
                  document.getElementById('judgment-viewer').hidden = true;
                  document.getElementById('judgment-compile').hidden = true;
                }
                refreshJudgments();
              } catch (e) { toast('Delete failed: ' + e.message, 'err'); }
            });
          }
        });
      }
    }
  } catch (e) {
    // Error-specific key: recovery payloads (even identical ones) repaint.
    setListHtml(document.getElementById('judgments-list'), 'err:' + e.message,
      `<div class="eval-empty"><div class="eval-empty-body">Failed: ${escapeHtml(e.message)}</div></div>`);
  }
  refreshAdapterDropdowns();
}

document.getElementById('judgment-create-btn')?.addEventListener('click', async () => {
  const name = document.getElementById('judgment-create-name').value.trim();
  if (!name) { toast('Name is required', 'err'); return; }
  try {
    await api('/v1/judgments', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
    toast('Created judgment dataset', 'ok');
    document.getElementById('judgment-create-name').value = '';
    refreshJudgments();
    openJudgmentViewer(name);
  } catch (e) { toast('Create failed: ' + e.message, 'err'); }
});

document.getElementById('judgment-autoadvance')?.addEventListener('change', ev => {
  judgmentAutoAdvance = ev.target.checked;
});

function openJudgmentViewer(name) {
  activeJudgmentDataset = name;
  document.getElementById('judgment-viewer').hidden = false;
  document.getElementById('judgment-pair').hidden = true;
  document.getElementById('judgment-actions').hidden = true;
  document.getElementById('judgment-rows-count').textContent = `Judging into "${name}". Press G to generate, A/B/T/S to vote.`;
  document.getElementById('judgment-compile').hidden = true;
  document.getElementById('judgment-prompt').focus();
  document.getElementById('judgment-viewer').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function openJudgmentCompile(name) {
  activeJudgmentDataset = name;
  document.getElementById('judgment-compile').hidden = false;
  document.getElementById('compile-sft-name').value = name + '-sft';
  document.getElementById('compile-output').innerHTML = '';
  document.getElementById('judgment-compile').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function abortPendingStreams() {
  for (const k of ['a', 'b']) {
    if (judgmentStreams[k]) {
      try { judgmentStreams[k].abort(); } catch (_) {}
      judgmentStreams[k] = null;
    }
  }
}

async function streamCompletion(slot, body) {
  const ctrl = new AbortController();
  judgmentStreams[slot] = ctrl;
  const target = document.getElementById('judgment-' + slot + '-text');
  target.textContent = '';
  target.classList.add('token-cursor');
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream', 'X-Kiln-Client': 'dashboard' },
      body: JSON.stringify({ ...body, stream: true }),
      signal: ctrl.signal,
    });
    if (!res.ok || !res.body) {
      const errText = await res.text().catch(() => `HTTP ${res.status}`);
      // The mock backend rejects streaming with a clear error code. Fall
      // back to a plain non-streaming completion so the judgment flow
      // still works without a real model loaded.
      if (errText.includes('streaming_not_supported')) {
        return await nonStreamingCompletion(slot, body, ctrl, target);
      }
      throw new Error(errText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let acc = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') return acc;
        try {
          const chunk = JSON.parse(payload);
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) {
            acc += delta;
            target.textContent = acc;
          }
        } catch (_) { /* tolerate non-JSON keepalives */ }
      }
    }
    return acc;
  } finally {
    target.classList.remove('token-cursor');
    if (judgmentStreams[slot] === ctrl) judgmentStreams[slot] = null;
  }
}

async function nonStreamingCompletion(slot, body, ctrl, target) {
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Kiln-Client': 'dashboard' },
    body: JSON.stringify({ ...body, stream: false }),
    signal: ctrl.signal,
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.error?.message || `HTTP ${res.status}`);
  }
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content || '';
  target.textContent = text;
  return text;
}

async function generateJudgmentPair() {
  if (!activeJudgmentDataset) { toast('Pick a judgment dataset first', 'err'); return; }
  const promptText = document.getElementById('judgment-prompt').value.trim();
  if (!promptText) { toast('Enter a prompt to compare on', 'err'); return; }
  abortPendingStreams();
  const adapterA = document.getElementById('judgment-adapter-a').value;
  const adapterB = document.getElementById('judgment-adapter-b').value;
  const temperature = parseFloat(document.getElementById('judgment-temp').value || '0.7');
  const baseBody = {
    messages: [{ role: 'user', content: promptText }],
    temperature, top_p: 1.0, max_tokens: 512,
  };
  const aBody = { ...baseBody }; if (adapterA) aBody.adapter = adapterA;
  const bBody = { ...baseBody }; if (adapterB) bBody.adapter = adapterB;

  document.getElementById('judgment-pair').hidden = false;
  document.getElementById('judgment-actions').hidden = false;
  document.getElementById('judgment-a-adapter').textContent = adapterA || 'base';
  document.getElementById('judgment-b-adapter').textContent = adapterB || 'base';
  // Stub the pair so vote actions know what to record. Final text is set
  // after streams complete.
  pendingJudgmentPair = {
    prompt: [{ role: 'user', content: promptText }],
    adapter_a: adapterA || null,
    adapter_b: adapterB || null,
    response_a: '',
    response_b: '',
  };

  // Run both streams concurrently. Either one's failure shouldn't kill the other.
  const [a, b] = await Promise.allSettled([
    streamCompletion('a', aBody),
    streamCompletion('b', bBody),
  ]);
  if (a.status === 'fulfilled') pendingJudgmentPair.response_a = a.value;
  else document.getElementById('judgment-a-text').innerHTML = icon('warning','icn-sm') + ' ' + escapeHtml(a.reason?.message || 'failed');
  if (b.status === 'fulfilled') pendingJudgmentPair.response_b = b.value;
  else document.getElementById('judgment-b-text').innerHTML = icon('warning','icn-sm') + ' ' + escapeHtml(b.reason?.message || 'failed');
}

document.getElementById('judgment-generate-btn')?.addEventListener('click', () => generateJudgmentPair());

// One toast with an Undo action for a just-recorded judgment. The rows POST
// returns `judgment_id` (the appended row's stable id) — Undo DELETEs that
// exact row, refreshes the visible counts, and confirms. A misclicked vote
// no longer poisons the dataset permanently. `fired` guards double-fires:
// actionToast removes the toast on click, but a queued second click must
// not double-DELETE (the second DELETE would 404 and toast a scary error).
function recordedJudgmentToast(message, datasetName, judgmentId) {
  if (!judgmentId) { toast(message, 'ok'); return; }  // no id, no Undo
  let fired = false;
  actionToast(message, 'ok', [{
    label: 'Undo',
    onClick: async () => {
      if (fired) return;
      fired = true;
      try {
        const m = await api('/v1/judgments/' + encodeURIComponent(datasetName) + '/rows/' + encodeURIComponent(judgmentId), { method: 'DELETE' });
        if (activeJudgmentDataset === datasetName) {
          document.getElementById('judgment-rows-count').textContent =
            `${m.num_rows} judgments in "${datasetName}". Press G to generate the next pair (A/B/T/S to vote).`;
        }
        refreshJudgments();
        toast(`Undone — judgment removed from "${datasetName}"`, 'ok');
      } catch (e) {
        // Leave the counts as they are — the row may still exist server-side.
        toast('Undo failed: ' + e.message, 'err');
      }
    },
  }]);
}

async function recordJudgment(winner) {
  if (!activeJudgmentDataset || !pendingJudgmentPair) return;
  // If a stream is still going, capture whatever has been emitted so far so
  // the user can vote mid-stream.
  if (!pendingJudgmentPair.response_a) {
    pendingJudgmentPair.response_a = document.getElementById('judgment-a-text').textContent;
  }
  if (!pendingJudgmentPair.response_b) {
    pendingJudgmentPair.response_b = document.getElementById('judgment-b-text').textContent;
  }
  abortPendingStreams();
  const note = document.getElementById('judgment-note').value.trim();
  const tags = document.getElementById('judgment-tags').value.split(',').map(s => s.trim()).filter(Boolean);
  const body = {
    ...pendingJudgmentPair,
    winner,
    note: note || null,
    tags,
  };
  const dataset = activeJudgmentDataset;
  try {
    const m = await api('/v1/judgments/' + encodeURIComponent(dataset) + '/rows', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    document.getElementById('judgment-rows-count').textContent =
      `${m.num_rows} judgments in "${dataset}". Press G to generate the next pair (A/B/T/S to vote).`;
    document.getElementById('judgment-note').value = '';
    pendingJudgmentPair = null;
    document.getElementById('judgment-actions').hidden = true;
    document.getElementById('judgment-prompt').value = '';
    document.getElementById('judgment-a-text').textContent = '';
    document.getElementById('judgment-b-text').textContent = '';
    document.getElementById('judgment-pair').hidden = true;
    refreshJudgments();
    const winnerLabel = { a: 'A wins', b: 'B wins', tie: 'Tie', skip: 'Skip' }[winner] || winner;
    recordedJudgmentToast(`Recorded ${winnerLabel} in "${dataset}"`, dataset, m.judgment_id);
    if (judgmentAutoAdvance) {
      // Re-focus the prompt for the next round.
      setTimeout(() => document.getElementById('judgment-prompt').focus(), 50);
    }
  } catch (e) { toast('Save failed: ' + e.message, 'err'); }
}

document.getElementById('judgment-pick-a')?.addEventListener('click', () => recordJudgment('a'));
document.getElementById('judgment-pick-b')?.addEventListener('click', () => recordJudgment('b'));
document.getElementById('judgment-pick-tie')?.addEventListener('click', () => recordJudgment('tie'));
document.getElementById('judgment-pick-skip')?.addEventListener('click', () => recordJudgment('skip'));

// Click reply card itself to vote — visual and obvious.
document.getElementById('judgment-card-a')?.addEventListener('click', () => {
  if (pendingJudgmentPair && document.getElementById('judgment-actions') && !document.getElementById('judgment-actions').hidden) {
    recordJudgment('a');
  }
});
document.getElementById('judgment-card-b')?.addEventListener('click', () => {
  if (pendingJudgmentPair && document.getElementById('judgment-actions') && !document.getElementById('judgment-actions').hidden) {
    recordJudgment('b');
  }
});

// Keyboard shortcuts for judgment voting.
document.addEventListener('keydown', ev => {
  // Only when judgment view is active and visible
  const evalsActive = document.getElementById('page-evals')?.classList.contains('active');
  const judgmentTabActive = document.getElementById('evals-tab-judgments')?.classList.contains('active');
  if (!evalsActive || !judgmentTabActive) return;
  // Don't intercept when typing in inputs
  const tag = (ev.target.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    // Special case: in the prompt textarea, Cmd/Ctrl+Enter still triggers generate.
    if ((ev.key === 'Enter') && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      generateJudgmentPair();
    }
    return;
  }
  if (document.getElementById('judgment-actions')?.hidden) {
    // Pre-vote: G or Enter generates a pair.
    if (ev.key === 'g' || ev.key === 'G' || ev.key === 'Enter') {
      ev.preventDefault();
      generateJudgmentPair();
    }
    return;
  }
  // Voting mode
  if (ev.key === 'a' || ev.key === 'A' || ev.key === 'ArrowLeft') { ev.preventDefault(); recordJudgment('a'); }
  else if (ev.key === 'b' || ev.key === 'B' || ev.key === 'ArrowRight') { ev.preventDefault(); recordJudgment('b'); }
  else if (ev.key === 't' || ev.key === 'T' || ev.key === 'ArrowUp') { ev.preventDefault(); recordJudgment('tie'); }
  else if (ev.key === 's' || ev.key === 'S' || ev.key === 'ArrowDown') { ev.preventDefault(); recordJudgment('skip'); }
});

document.getElementById('compile-btn')?.addEventListener('click', async () => {
  if (!activeJudgmentDataset) { toast('No judgment dataset selected', 'err'); return; }
  const output_dataset = document.getElementById('compile-sft-name').value.trim();
  if (!output_dataset) { toast('Provide an output SFT dataset name', 'err'); return; }
  try {
    const res = await api('/v1/judgments/' + encodeURIComponent(activeJudgmentDataset) + '/compile', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ output_dataset, include_skips: false }),
    });
    // One-click crank: the dataset is ready — go straight to training a
    // judge LoRA on it instead of sending the user off to find the
    // Training tab with an instruction string.
    document.getElementById('compile-output').innerHTML =
      `<div style="padding:10px; background:var(--success-bg); border:1px solid var(--success-bd); border-radius:6px; color:var(--success-fg); font-size:12px;">
        ${icon('check','icn-sm')} Compiled <strong>${res.rows}</strong> judgments into SFT dataset <code>${escapeHtml(res.dataset.name)}</code> (${res.dataset.num_rows} rows).
        <button type="button" class="btn btn-primary btn-sm" style="margin-left:8px;" id="compile-train-judge-btn">${icon('flask','icn-sm')} Train judge LoRA now</button>
      </div>`;
    document.getElementById('compile-train-judge-btn')?.addEventListener('click', () => {
      trainFromDataset(res.dataset.name, 'sft');
    });
    toast('Compiled to SFT', 'ok');
    refreshDatasets();
  } catch (e) { toast('Compile failed: ' + e.message, 'err'); }
});

document.getElementById('validate-btn')?.addEventListener('click', async () => {
  if (!activeJudgmentDataset) return;
  const adapter = document.getElementById('compile-judge-adapter').value;
  const holdout_n = parseInt(document.getElementById('compile-holdout').value, 10) || 20;
  if (!adapter) { toast('Pick an adapter to validate', 'err'); return; }
  try {
    const res = await api('/v1/judgments/' + encodeURIComponent(activeJudgmentDataset) + '/validate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ adapter, holdout_n }),
    });
    document.getElementById('compile-output').innerHTML =
      `<div style="padding:10px; background:var(--info-bg); border:1px solid var(--info-bd); border-radius:6px; color:var(--info-fg); font-size:12px;">
        Queued validation as eval job <code>${escapeHtml(res.eval_job_id)}</code>. Switching to the Jobs tab…
      </div>`;
    refreshEvalJobs();
    setTimeout(() => {
      document.getElementById('evals-tab-jobs')?.click();
      openDrillModal(res.eval_job_id);
    }, 400);
  } catch (e) { toast('Validate failed: ' + e.message, 'err'); }
});

// Show the first-time onboarding banner unless the user has dismissed it.
if (!localStorage.getItem('kiln-evals-onboarded')) {
  document.getElementById('evals-onboarding').hidden = false;
}

// Initial adapter dropdown population is universal (used by Training and
// Playground forms too). Eval-scoped lists are lazy: they fetch when the
// Evals page is first activated (see selectPage / refreshActiveEvalSubTab)
// and on every polling tick while the page is visible (below). This avoids
// 4× /v1/eval/* + /v1/judgments fetches firing on every dashboard load.
refreshAdapterDropdowns();
if (document.getElementById('page-evals')?.classList.contains('active')) {
  refreshActiveEvalSubTab();
}

// Periodic refresh — only updates the active sub-tab so we don't thrash.
// Every sub-tab refreshes on the same 1.5s tick (so a running job's progress
// feels alive); the content-keyed renders (setListHtml) make the unchanged
// ticks free instead of clobbering hover/selection/open dropdowns.
setInterval(() => {
  const evalsPage = document.getElementById('page-evals');
  if (!evalsPage || !evalsPage.classList.contains('active')) return;
  const active = evalsPage.querySelector('.tab.active')?.dataset?.tab;
  if (active === 'jobs')      refreshEvalJobs();
  else if (active === 'datasets') refreshDatasets();
  else if (active === 'suites')   refreshSuites();
  else if (active === 'judgments') refreshJudgments();
}, 1500);

/* =====================================================================
   Cmd-K command palette
   ---------------------------------------------------------------------
   Aggregates everything-actionable into one text-searchable list:
   navigations (open page X), actions (run an eval, switch adapter),
   and direct drill-ins (click a job/adapter/suite directly from search).
   The palette stays cheap by reusing the data the dashboard already
   polls — no separate index, no extra requests.
   ===================================================================== */

let cmdkOpen = false;
let cmdkActiveIdx = 0;
let cmdkResultsCache = [];

function openCmdk() {
  cmdkOpen = true;
  cmdkActiveIdx = 0;
  const modal = document.getElementById('cmdk-modal');
  modal.hidden = false;
  // The palette stacks over whatever is open (e.g. ⌘K from inside a drill):
  // the manager keeps the drill's layer + scroll lock and Escape peels the
  // palette off first.
  openModal(modal, { onClose: closeCmdk });
  const input = document.getElementById('cmdk-input');
  input.value = '';
  input.focus();
  renderCmdkResults('');
}
function closeCmdk() {
  cmdkOpen = false;
  const modal = document.getElementById('cmdk-modal');
  modal.hidden = true;
  closeModal(modal);
}

// Build the searchable index from cached state. Cheap to recompute on
// every keystroke — N is at most low hundreds.
function buildCmdkIndex() {
  const items = [];
  // Navigation
  items.push({ kind: 'nav', icon: icon('home'), title: 'Overview',   sub: 'Live stats, recent requests, quick actions', action: () => selectPage('overview') });
  items.push({ kind: 'nav', icon: icon('layers'), title: 'Adapters',   sub: 'Saved LoRAs, upload, merge', action: () => selectPage('adapters') });
  items.push({ kind: 'nav', icon: icon('flask'), title: 'Training',   sub: 'SFT/GRPO queue + submit', action: () => selectPage('training') });
  items.push({ kind: 'nav', icon: icon('chart'), title: 'Evals',      sub: 'Datasets, suites, jobs, judgments', action: () => selectPage('evals') });
  items.push({ kind: 'nav', icon: icon('flask'), title: 'Distill',    sub: 'Teachers, boost, refresh, merge, self-improve', action: () => selectPage('distill') });
  items.push({ kind: 'nav', icon: icon('terminal'), title: 'pi Terminal', sub: 'Run pi against this Kiln, right here', action: () => selectPage('terminal') });
  items.push({ kind: 'nav', icon: icon('chat'), title: 'Playground', sub: 'Quick inference + A/B compare', action: () => selectPage('playground') });
  // Actions
  items.push({ kind: 'action', icon: icon('link'), title: 'Connect your agent', sub: 'Base URL, model id, pi / opencode setup, test connection', action: () => openConnect() });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Run a new eval',   sub: 'Submit a suite against an adapter', action: () => { selectPage('evals'); document.getElementById('evals-tab-suites')?.click(); } });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Train a new SFT adapter', sub: 'Open the SFT submit form', action: () => { selectPage('training'); document.getElementById('training-tab-sft')?.click(); } });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Train a new GRPO adapter', sub: 'Open the GRPO submit form', action: () => { selectPage('training'); document.getElementById('training-tab-grpo')?.click(); } });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Upload a dataset', sub: 'Drop an SFT JSONL', action: () => { selectPage('evals'); document.getElementById('evals-tab-datasets')?.click(); document.getElementById('dataset-name')?.focus(); } });
  items.push({ kind: 'action', icon: icon('plus'), title: 'Create judgment dataset', sub: 'Start the A/B flywheel', action: () => { selectPage('evals'); document.getElementById('evals-tab-judgments')?.click(); document.getElementById('judgment-create-name')?.focus(); } });
  // Adapters
  for (const name of evalAdaptersCache) {
    items.push({
      kind: 'adapter', icon: icon('layers'),
      title: name,
      sub: name === evalActiveAdapter ? 'Adapter · ACTIVE' : 'Adapter',
      action: async () => {
        selectPage('adapters');
        await openAdapterDrillModal(name);
      },
    });
    items.push({
      kind: 'adapter-load', icon: icon('refresh'),
      title: 'Load adapter ' + name,
      sub: 'Switch active LoRA',
      action: async () => {
        try {
          await api('/v1/adapters/load', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
          toast(`Loaded ${name}`, 'ok');
          refreshAdapterDropdowns();
          pollAdapters && pollAdapters();
        } catch (e) { toast('Load failed: ' + e.message, 'err'); }
      },
    });
  }
  // Suites (from the cached jobs list — list endpoint not always loaded yet)
  const suiteNames = new Set();
  for (const j of evalJobsCache) suiteNames.add(j.suite_name);
  for (const name of suiteNames) {
    items.push({
      kind: 'suite', icon: icon('target'),
      title: name,
      sub: 'Eval suite',
      action: async () => {
        selectPage('evals');
        document.getElementById('evals-tab-suites')?.click();
      },
    });
    items.push({
      kind: 'suite-run', icon: icon('play'),
      title: `Run "${name}" vs active adapter`,
      sub: 'Queue an eval immediately',
      action: async () => {
        try {
          const res = await api('/v1/eval/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ suite: name, adapter: evalActiveAdapter || '' }) });
          toast('Queued eval ' + res.job_id.slice(0, 8), 'ok');
          selectPage('evals');
          document.getElementById('evals-tab-jobs')?.click();
          refreshEvalJobs();
        } catch (e) { toast('Run failed: ' + e.message, 'err'); }
      },
    });
  }
  // Jobs (recent — clickable to drill in)
  for (const j of evalJobsCache.slice(0, 20)) {
    items.push({
      kind: 'job', icon: icon('chart'),
      title: `${j.suite_name}`,
      sub: `Eval · ${j.state} · ${j.headline_accuracy != null ? (j.headline_accuracy*100).toFixed(0)+'%' : '—'} · ${j.job_id.slice(0, 8)}`,
      action: () => { selectPage('evals'); document.getElementById('evals-tab-jobs')?.click(); openDrillModal(j.job_id); },
    });
  }
  // Training runs (running + queued + most-recent completed) — same
  // drill-modal jump as clicking a card on the Training tab. Lets
  // power users find a finished run by adapter name without scrolling.
  const trainingPool = trainingJobsCache ? [
    ...(trainingJobsCache.running ? [trainingJobsCache.running] : []),
    ...(trainingJobsCache.queued || []),
    ...(trainingJobsCache.completed || []).slice(0, 30),
  ] : [];
  for (const j of trainingPool) {
    const stateNorm = (j.state || '').toString().toLowerCase() || 'queued';
    const lossLbl = j.current_loss != null ? `loss ${j.current_loss.toFixed(3)}` : 'no loss yet';
    items.push({
      kind: 'train-job', icon: icon('flask'),
      title: j.adapter_name || j.job_id,
      sub: `${(j.job_type || 'train').toString().toUpperCase()} · ${stateNorm} · ${lossLbl} · ${j.job_id.slice(0, 8)}`,
      action: () => {
        selectPage('training');
        document.getElementById('training-tab-queue')?.click();
        if (typeof openTrainDrillModal === 'function') openTrainDrillModal(j.job_id);
      },
    });
  }
  // Recent requests (last 20) — jump back into the request inspect modal
  // by short id or by prompt content. Most useful for retrieving "what
  // was the prompt that produced that weird answer five minutes ago".
  for (const r of (recentRequestsCache || []).slice(0, 20)) {
    const preview = (r.prompt_preview || '').replace(/\s+/g, ' ').slice(0, 60) || '(no prompt)';
    items.push({
      kind: 'recent-req', icon: icon('arrow-right'),
      title: preview,
      sub: `Request · ${r.streamed ? 'stream' : 'unary'} · ${r.completion_tokens || 0} tok · ${(r.id || '').replace(/^chatcmpl-/, '').slice(0, 8)}`,
      action: () => {
        selectPage('overview');
        if (typeof openRequestDrillModal === 'function') openRequestDrillModal(r.id);
      },
    });
  }
  return items;
}

function renderCmdkResults(query) {
  const items = buildCmdkIndex();
  const q = query.trim().toLowerCase();
  const filtered = !q ? items : items.filter(it => {
    return it.title.toLowerCase().includes(q) || (it.sub && it.sub.toLowerCase().includes(q));
  });
  cmdkResultsCache = filtered;
  cmdkActiveIdx = Math.max(0, Math.min(cmdkActiveIdx, filtered.length - 1));
  const el = document.getElementById('cmdk-results');
  if (!filtered.length) {
    el.innerHTML = `<div class="cmdk-empty">No matches for <code>${escapeHtml(q || '(all)')}</code>.</div>`;
    return;
  }
  // Group by kind label.
  const groups = {
    nav: 'Navigate', action: 'Actions',
    adapter: 'Adapters', 'adapter-load': 'Adapter actions',
    suite: 'Suites', 'suite-run': 'Suite actions',
    job: 'Jobs',
  };
  let html = '';
  let lastGroup = '';
  filtered.forEach((it, i) => {
    const groupLabel = groups[it.kind] || it.kind;
    if (groupLabel !== lastGroup) {
      html += `<div class="cmdk-section-label">${escapeHtml(groupLabel)}</div>`;
      lastGroup = groupLabel;
    }
    html += `<div class="cmdk-item ${i === cmdkActiveIdx ? 'cmdk-active' : ''}" data-cmdk-idx="${i}">
      <span class="cmdk-item-icon">${it.icon || '·'}</span>
      <div class="cmdk-item-body">
        <div class="cmdk-item-title">${escapeHtml(it.title)}</div>
        <div class="cmdk-item-sub">${escapeHtml(it.sub || '')}</div>
      </div>
      <span class="cmdk-item-action">↵</span>
    </div>`;
  });
  el.innerHTML = html;
  el.querySelectorAll('.cmdk-item').forEach(item => {
    item.addEventListener('mouseover', () => {
      cmdkActiveIdx = parseInt(item.dataset.cmdkIdx, 10);
      el.querySelectorAll('.cmdk-item').forEach((other, idx) => other.classList.toggle('cmdk-active', idx === cmdkActiveIdx));
    });
    item.addEventListener('click', () => runCmdkActive());
  });
  // Scroll the active row into view.
  const active = el.querySelector('.cmdk-active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function runCmdkActive() {
  const item = cmdkResultsCache[cmdkActiveIdx];
  if (!item) return;
  closeCmdk();
  // Defer to next tick so any open-modal action sees a clean state.
  setTimeout(() => item.action(), 10);
}

document.getElementById('cmdk-trigger')?.addEventListener('click', openCmdk);
document.getElementById('cmdk-input')?.addEventListener('input', ev => {
  cmdkActiveIdx = 0;
  renderCmdkResults(ev.target.value);
});
document.getElementById('cmdk-modal')?.addEventListener('click', ev => {
  if (ev.target.id === 'cmdk-modal') closeCmdk();
});

// Keyboard cheatsheet — opened with '?'. Lists the shortcuts that already exist
// so power users can discover triage/judging/playground keys without hunting.
function toggleShortcutsSheet() {
  const existing = document.getElementById('shortcuts-modal');
  if (existing) { closeModal(existing); existing.remove(); return; }
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || '');
  const mod = isMac ? '⌘' : 'Ctrl';
  const groups = [
    ['Global', [[[mod + 'K', '/'], 'Command palette'], [['?'], 'This shortcuts list'], [['Esc'], 'Close any modal or palette']]],
    ['Recent requests', [[['Enter', 'Space'], 'Inspect the focused request'], [['←', '→'], 'Previous / next request in the inspector']]],
    ['Eval results drill', [[['/'], 'Search outcomes'], [['r'], 'Re-run the suite'], [['j', 'k'], 'Next / previous outcome']]],
    ['A/B judging', [[['a', 'b'], 'Prefer A / B'], [['t'], 'Tie'], [['s'], 'Skip']]],
    ['Playground', [[['Enter'], 'Send'], [['⇧Enter'], 'Newline'], [['Esc'], 'Stop generating']]],
  ];
  const kbd = keys => keys.map(k => `<kbd>${escapeHtml(k)}</kbd>`).join('<span class="kbd-or">or</span>');
  const body = groups.map(([title, rows]) => `
    <div class="shortcuts-group">
      <div class="shortcuts-group-title">${escapeHtml(title)}</div>
      ${rows.map(([keys, desc]) => `<div class="shortcut-row"><span class="shortcut-keys">${kbd(keys)}</span><span class="shortcut-desc">${escapeHtml(desc)}</span></div>`).join('')}
    </div>`).join('');
  const m = document.createElement('div');
  m.id = 'shortcuts-modal';
  m.className = 'modal-backdrop';
  m.setAttribute('role', 'dialog');
  m.setAttribute('aria-modal', 'true');
  m.setAttribute('aria-label', 'Keyboard shortcuts');
  m.innerHTML = `<div class="modal-shell modal-shell-fit shortcuts-shell" tabindex="-1">
    <div class="modal-head"><h2>Keyboard shortcuts</h2><span style="flex:1 1 auto;"></span>
      <button class="modal-close" id="shortcuts-close" aria-label="Close"><svg class="icn" aria-hidden="true"><use href="#i-close"></use></svg></button></div>
    <div class="shortcuts-body">${body}</div>
  </div>`;
  document.body.appendChild(m);
  // Escape, focus, and the scroll lock come from the shared modal manager.
  const close = () => { closeModal(m); m.remove(); };
  m.querySelector('#shortcuts-close')?.addEventListener('click', close);
  m.addEventListener('click', ev => { if (ev.target === m) close(); });
  openModal(m, { onClose: close });
}

document.addEventListener('keydown', ev => {
  // Open: ⌘K / Ctrl+K (anywhere except inside an input that already has its own handler)
  if ((ev.key === 'k' || ev.key === 'K') && (ev.metaKey || ev.ctrlKey)) {
    ev.preventDefault();
    if (cmdkOpen) closeCmdk(); else openCmdk();
    return;
  }
  // Open: just '/' when nothing else is focused (mirrors GitHub behaviour)
  if (!cmdkOpen && ev.key === '/' && !['INPUT','TEXTAREA','SELECT'].includes((ev.target.tagName||'').toUpperCase())) {
    ev.preventDefault();
    openCmdk();
    return;
  }
  // '?' (Shift+/) opens the keyboard cheatsheet when not typing.
  if (!cmdkOpen && ev.key === '?' && !['INPUT','TEXTAREA','SELECT'].includes((ev.target.tagName||'').toUpperCase()) && !ev.target.isContentEditable) {
    ev.preventDefault();
    toggleShortcutsSheet();
    return;
  }
  if (!cmdkOpen) return;
  // Escape is handled by the shared modal manager (closes the TOP of the
  // stack — the palette when it's frontmost).
  if (ev.key === 'ArrowDown') {
    ev.preventDefault();
    cmdkActiveIdx = Math.min(cmdkResultsCache.length - 1, cmdkActiveIdx + 1);
    renderCmdkResults(document.getElementById('cmdk-input').value);
  } else if (ev.key === 'ArrowUp') {
    ev.preventDefault();
    cmdkActiveIdx = Math.max(0, cmdkActiveIdx - 1);
    renderCmdkResults(document.getElementById('cmdk-input').value);
  } else if (ev.key === 'Enter') {
    ev.preventDefault();
    runCmdkActive();
  }
});

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
// First-completion pass/fail flips between a base run and an adapter run.
function compareFlips(baseRun, adapterRun) {
  const verdictOf = (run) => {
    const m = new Map();
    for (const o of run.outcomes || []) {
      if ((o.completion_index || 0) === 0) m.set(o.example_id, o.kind === 'pass');
    }
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
       <span class="hint" style="margin-left:8px; font-family:var(--font-mono);">${escapeHtml(j.job_id)}</span>`;

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
    renderDrillKv('Data source', source.source || replay?.request_body?.dataset || replay?.request_body?.dataset_path),
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
  const empty = !receipt && !replay && !j.metadata_error
    ? '<div class="hint">No receipt or replay metadata was found for this job.</div>'
    : '';
  return `<div class="detail-section">
    <h4>Run metadata</h4>
    ${receipt || replay ? `<div class="req-stats" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));">${rows}</div>` : empty}
    ${error}
    ${receiptRaw}
    ${replayRaw}
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
  ${renderTrainMetadata(j)}
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
    document.querySelectorAll('[data-linked-eval]').forEach(b => {
      b.addEventListener('click', () => {
        closeTrainDrillModal();
        selectPage('evals');
        document.getElementById('evals-tab-jobs')?.click();
        openDrillModal(b.dataset.linkedEval);
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
    const meta  = dur ? `<span class="think-meta"> · ${live ? '' : 'for '}${escapeHtml(dur)}</span>` : '';
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
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    streamLoop:
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (let line of lines) {
        if (line.endsWith('\r')) line = line.slice(0, -1);
        if (!line.startsWith('data:')) continue;
        let payload = line.slice(5);
        if (payload.startsWith(' ')) payload = payload.slice(1);
        if (payload === '[DONE]') break streamLoop;
        try {
          const chunk = JSON.parse(payload);
          const choice = chunk.choices?.[0];
          const delta = choice?.delta;
          if (choice?.finish_reason) m.finishReason = choice.finish_reason;
          if (!delta) continue;
          const now = performance.now();
          if (m.firstTokenMs == null) { m.firstTokenMs = now; m.ttftMs = now - m.startMs; }
          if (delta.reasoning_content) {
            if (m.thinkStartMs == null) m.thinkStartMs = now;
            m.reasoning += delta.reasoning_content;
          }
          if (delta.content) {
            if (m.thinkStartMs != null && m.thinkEndMs == null) m.thinkEndMs = now;
            m.content += delta.content;
          }
          m.lastTokenMs = now;
          m.durationMs = now - m.startMs;
          _renderCompareSide(side, m);
        } catch (parseErr) {
          console.warn('[playground compare] skipped malformed SSE chunk', parseErr, payload.slice(0, 120));
        }
      }
    }
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
        const caps = t.capabilities ? `<span class="hint">${t.capabilities.max_top_k || '?'} top-K · ${t.capabilities.vocab_size || '?'} vocab</span>` : '';
        const adapter = t.spec?.adapter ? ` · adapter ${escapeHtml(t.spec.adapter)}` : '';
        return `<div class="adapter-card" style="display:flex; align-items:center; gap:var(--space-3);">
          <div style="flex:1; min-width:0;">
            <div style="font-weight:600;">${escapeHtml(t.spec?.alias || '?')}</div>
            <div style="font-size:var(--text-xs); color:var(--text-muted);">${escapeHtml(t.spec?.kind || '?')} · ${escapeHtml(t.spec?.model_id || '?')}${adapter}</div>
          </div>
          ${caps}
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
        if (!alias) continue;
        opts.push(`<option value="${escapeHtml(alias)}">${escapeHtml(alias)}</option>`);
      }
      node.innerHTML = opts.join('');
      if (prev && teachers.some(t => t.spec?.alias === prev)) node.value = prev;
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
  };
  select.addEventListener('change', sync);
  sync();
});

document.getElementById('teacher-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    const body = {
      alias: form.alias.value.trim(),
      kind: form.kind.value,
      model_id: form.model_id.value.trim(),
    };
    if (body.kind === 'remote') body.provider = 'vllm';
    if (form.url.value.trim()) body.url = form.url.value.trim();
    if (form.api_key_env.value.trim()) body.api_key_env = form.api_key_env.value.trim();
    if (body.kind === 'local' && form.adapter.value.trim()) body.adapter = form.adapter.value.trim();
    if (form.max_top_k.value) body.max_top_k = parseInt(form.max_top_k.value, 10);
    if (form.vocab_size.value) body.vocab_size = parseInt(form.vocab_size.value, 10);
    await api('/v1/teachers', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(`Registered teacher ${body.alias}`);
    form.reset();
    form.kind.dispatchEvent(new Event('change'));
    refreshTeachersList();
  } catch (err) { toast('Register failed: ' + err.message, 'err'); }
});

// --- Recipes (/v1/recipes + /v1/recipes/run) ------------------------
async function refreshRecipesList() {
  const node = document.getElementById('recipes-list');
  if (!node) return;
  try {
    const res = await api('/v1/recipes');
    const recipes = res.recipes || [];
    if (recipes.length === 0) {
      node.innerHTML = '<div class="empty">No bundled recipes.</div>';
      return;
    }
    node.innerHTML = recipes.map(r => `<div class="adapter-card" style="display:flex; align-items:center; gap:var(--space-3); margin-bottom:var(--space-2);">
      <div style="flex:1; min-width:0;">
        <div style="font-weight:600;">${escapeHtml(r.name)}</div>
        <div style="font-size:var(--text-xs); color:var(--text-muted);">${escapeHtml(r.description || '')}</div>
        <div style="font-size:var(--text-2xs); color:var(--text-muted); margin-top:var(--space-1);">${r.num_steps || 0} step${(r.num_steps || 0) === 1 ? '' : 's'}</div>
      </div>
      <button class="btn btn-sm" data-recipe-run="${escapeHtml(r.name)}">Run</button>
    </div>`).join('');
  } catch (e) {
    node.innerHTML = `<div class="empty">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

document.addEventListener('click', async (ev) => {
  const btn = ev.target.closest('[data-recipe-run]');
  if (!btn) return;
  const name = btn.getAttribute('data-recipe-run');
  if (!name) return;
  try {
    btn.disabled = true; btn.textContent = 'Queuing…';
    const res = await api('/v1/recipes/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ recipe: name }) });
    toast(res.message || `Queued recipe ${name}`);
  } catch (e) { toast('Run failed: ' + e.message, 'err'); }
  finally { btn.disabled = false; btn.textContent = 'Run'; }
});

// --- Submit OPD (/v1/train/opd) -------------------------------------
document.getElementById('opd-use-sample')?.addEventListener('click', () => {
  document.getElementById('opd-prompts').value = JSON.stringify([
    { messages: [{ role: 'user', content: 'Solve for x: 2x^2 - 5x + 3 = 0.' }, { role: 'assistant', content: 'Use the quadratic formula: x = (5 ± √(25 - 24)) / 4 = (5 ± 1)/4, so x = 3/2 or x = 1.' }] },
    { messages: [{ role: 'user', content: 'What is the derivative of sin(x^2)?' }, { role: 'assistant', content: 'Chain rule: d/dx sin(x²) = cos(x²) · 2x.' }] },
  ], null, 2);
});

document.getElementById('opd-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    const promptsText = document.getElementById('opd-prompts').value.trim();
    const prompts = promptsText ? JSON.parse(promptsText) : [];
    if (!Array.isArray(prompts) || prompts.length === 0) {
      throw new Error('Prompts must be a non-empty JSON array');
    }
    const teacher = document.getElementById('opd-teacher').value;
    if (!teacher) throw new Error('Pick a teacher first (Teachers tab)');
    const opdLearningRate = parseOptionalFiniteNumberField(
      document.getElementById('opd-lr').value, 'Learning rate');
    const body = {
      prompts,
      teacher,
      config: {
        output_name: document.getElementById('opd-output-name').value.trim(),
        loss: document.getElementById('opd-loss').value,
        top_k: parseInt(document.getElementById('opd-top-k').value, 10),
        samples_per_prompt: parseInt(document.getElementById('opd-samples').value, 10),
        lora_rank: parseInt(document.getElementById('opd-rank').value, 10),
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
    const res = await api('/v1/train/opd', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(res.message || 'Distillation job queued');
    selectPage('training');
  } catch (err) { toast(err.message, 'err'); }
});

// --- Distill / Refresh (/v1/distill/refresh) ------------------------
document.getElementById('distill-refresh-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
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
    };
    if (form.if_eval_suite.value.trim()) body.if_eval_suite = form.if_eval_suite.value.trim();
    if (form.new_knowledge_eval_suite.value.trim()) body.new_knowledge_eval_suite = form.new_knowledge_eval_suite.value.trim();
    const res = await api('/v1/distill/refresh', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(res.message || 'Refresh queued');
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
      rank: parseInt(form.rank.value, 10),
      rollout_budget: parseInt(form.rollout_budget.value, 10),
      use_cache: form.use_cache.checked,
    };
    const res = await api('/v1/distill/pump', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(res.message || 'Boost job queued');
    selectPage('training');
  } catch (err) { toast(err.message, 'err'); }
});

// --- Distill / Merge (/v1/adapters/distill_merge) -------------------
document.getElementById('distill-merge-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  try {
    const sources = JSON.parse(form.sources.value);
    if (!Array.isArray(sources) || sources.length === 0) throw new Error('sources must be a non-empty JSON array');
    const body = {
      name: form.name.value.trim(),
      sources,
      student: form.student.value.trim() || 'base',
      rollout_budget: parseInt(form.rollout_budget.value, 10),
      config: { training_mode: 'off_policy' },
    };
    const res = await api('/v1/adapters/distill_merge', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(res.message || 'Merge queued');
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
    const body = {
      name: form.name.value.trim(),
      mode: form.mode.value,
      config: { training_mode: 'off_policy' },
    };
    const prompts = JSON.parse(form.prompts.value);
    if (!Array.isArray(prompts) || prompts.length === 0) throw new Error('Prompts must be a non-empty JSON array');
    body.prompts = prompts;
    const gt = document.getElementById('self-ground-truth')?.value?.trim();
    if (gt) body.ground_truth = JSON.parse(gt);
    const docs = document.getElementById('self-documents')?.value?.trim();
    if (docs) body.documents = JSON.parse(docs);
    const res = await api('/v1/distill/self', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast(res.message || 'Self-improvement job queued');
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

/* =====================================================================
   Boot pass 2: apply the full deep-link route (#page/subtab/id).
   Runs LAST on purpose — every tablist, localStorage restore, and modal
   open fn above is wired by now, so an explicit hash sub-tab overrides
   the restores (the hash always wins) and drill ids can open their
   modals. Only replaceState repairs happen here; the boot landing never
   mints a history entry, so Back still exits the dashboard.
   ===================================================================== */
applyHashRoute({ boot: true });

})();
