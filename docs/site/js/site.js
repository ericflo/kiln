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
})();
