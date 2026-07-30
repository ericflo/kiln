(() => {
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.copy-btn');
    if (!btn) return;
    const id = btn.dataset.target;
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    const text = target.textContent.replace(/\u00a0/g, ' ');
    const restore = btn.textContent;
    const done = () => {
      btn.dataset.copied = '1';
      btn.textContent = 'copied';
      setTimeout(() => {
        delete btn.dataset.copied;
        btn.textContent = restore;
      }, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
    } else {
      fallbackCopy(text, done);
    }
  });

  function fallbackCopy(text, cb) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (_) {}
    document.body.removeChild(ta);
    cb();
  }

  $$('.tabs').forEach((group) => {
    const buttons = $$('[role="tab"]', group);
    if (!buttons.length) return;
    buttons.forEach((btn) => {
      btn.addEventListener('click', () => activate(btn));
      btn.addEventListener('keydown', (e) => {
        const idx = buttons.indexOf(btn);
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
          e.preventDefault();
          buttons[(idx + 1) % buttons.length].focus();
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
          e.preventDefault();
          buttons[(idx - 1 + buttons.length) % buttons.length].focus();
        } else if (e.key === 'Home') {
          e.preventDefault();
          buttons[0].focus();
        } else if (e.key === 'End') {
          e.preventDefault();
          buttons[buttons.length - 1].focus();
        } else if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          activate(btn);
        }
      });
    });

    function activate(btn) {
      buttons.forEach((b) => {
        const selected = b === btn;
        b.setAttribute('aria-selected', selected ? 'true' : 'false');
        b.tabIndex = selected ? 0 : -1;
        const panelId = b.getAttribute('aria-controls');
        if (!panelId) return;
        const panel = document.getElementById(panelId);
        if (!panel) return;
        if (selected) {
          panel.removeAttribute('hidden');
          panel.setAttribute('aria-hidden', 'false');
        } else {
          panel.setAttribute('hidden', '');
          panel.setAttribute('aria-hidden', 'true');
        }
      });
      btn.focus();
    }
  });

  const nav = document.querySelector('.site-nav');
  const navToggle = document.querySelector('.nav-toggle');
  const navExplore = document.querySelector('.nav-explore');
  const navExploreToggle = document.querySelector('.nav-explore-toggle');

  function setNavigation(open, { restoreFocus = false } = {}) {
    if (!nav || !navToggle) return;
    nav.dataset.open = open ? 'true' : 'false';
    navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    const label = navToggle.querySelector('.sr-only');
    if (label) label.textContent = open ? 'Close navigation' : 'Open navigation';
    document.body.classList.toggle('nav-open', open);
    if (!open && restoreFocus) navToggle.focus();
  }

  function setExplore(open) {
    if (!navExplore || !navExploreToggle) return;
    navExplore.dataset.open = open ? 'true' : 'false';
    navExploreToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  if (navToggle && nav) {
    navToggle.addEventListener('click', () => {
      setNavigation(nav.dataset.open !== 'true');
    });

    nav.addEventListener('click', (event) => {
      if (event.target.closest('a')) setNavigation(false);
    });

    window.addEventListener('resize', () => {
      if (window.innerWidth > 980) setNavigation(false);
    });
  }

  if (navExploreToggle && navExplore) {
    navExploreToggle.addEventListener('click', () => {
      setExplore(navExplore.dataset.open !== 'true');
    });
  }

  document.addEventListener('click', (event) => {
    if (navExplore && !navExplore.contains(event.target)) setExplore(false);
    if (
      nav
      && navToggle
      && nav.dataset.open === 'true'
      && !nav.contains(event.target)
      && !navToggle.contains(event.target)
    ) {
      setNavigation(false);
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    if (nav?.dataset.open === 'true') {
      setNavigation(false, { restoreFocus: true });
      return;
    }
    if (navExplore?.dataset.open === 'true') {
      setExplore(false);
      navExploreToggle?.focus();
    }
  });

  const learningPass = document.querySelector('[data-learning-pass]');
  const replayButton = document.querySelector('[data-replay-learning]');
  const learningStatus = document.querySelector('[data-learning-status]');

  if (learningPass && replayButton) {
    replayButton.addEventListener('click', () => {
      learningPass.classList.remove('is-replaying');
      void learningPass.offsetWidth;
      learningPass.classList.add('is-replaying');
      replayButton.disabled = true;
      if (learningStatus) learningStatus.textContent = 'Training pass replaying.';

      window.setTimeout(() => {
        learningPass.classList.remove('is-replaying');
        replayButton.disabled = false;
        if (learningStatus) {
          learningStatus.textContent = 'Training pass complete. The improved adapter is serving.';
        }
      }, 1100);
    });
  }
})();
