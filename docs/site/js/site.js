(() => {
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.copy-btn');
    if (!btn || !document.body.classList.contains('home')) return;
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

  function enhanceProductNavigation() {
    const shell = document.querySelector('.site-nav-bar');
    if (!shell || shell.querySelector('.nav-toggle')) return;

    const nav = shell.querySelector('.site-nav');
    const brand = shell.querySelector(':scope > a');
    if (!nav || !brand) return;

    document.documentElement.classList.add('product-page-enhanced');
    document.body.classList.add('product-page-shell');
    shell.closest('header')?.classList.add('product-topbar');
    brand.classList.add('product-brand');
    brand.setAttribute('aria-label', 'Kiln home');

    nav.id ||= 'site-nav';
    nav.setAttribute('aria-label', 'Primary');
    nav.dataset.open = 'false';

    const links = Array.from(nav.querySelectorAll(':scope > a'));
    const text = (link) => link.textContent.trim();
    const primaryNames = new Set(['Quickstart', 'Documentation', 'Demo']);
    const github = links.find((link) => text(link) === 'GitHub');
    const primary = links.filter((link) => primaryNames.has(text(link)));
    const explore = links.filter((link) => !primaryNames.has(text(link)) && link !== github);

    nav.replaceChildren(...primary);

    if (explore.length) {
      const exploreShell = document.createElement('div');
      exploreShell.className = 'nav-explore';
      exploreShell.dataset.open = 'false';

      const exploreToggle = document.createElement('button');
      exploreToggle.className = 'nav-explore-toggle';
      exploreToggle.type = 'button';
      exploreToggle.setAttribute('aria-expanded', 'false');
      exploreToggle.innerHTML = 'Explore <span aria-hidden="true">⌄</span>';

      const exploreMenu = document.createElement('div');
      exploreMenu.className = 'nav-explore-menu';
      exploreMenu.append(...explore);

      const current = explore.find((link) => link.getAttribute('aria-current') === 'page');
      if (current) {
        exploreToggle.classList.add('has-current');
        exploreToggle.setAttribute('aria-label', `Explore, current page: ${text(current)}`);
      }

      exploreShell.append(exploreToggle, exploreMenu);
      nav.append(exploreShell);
    }

    if (github) {
      github.classList.remove('hidden', 'sm:inline');
      github.classList.add('nav-ext', 'nav-github');
      github.rel = 'noopener';
      nav.append(github);
    }

    const toggle = document.createElement('button');
    toggle.className = 'nav-toggle';
    toggle.type = 'button';
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-controls', nav.id);
    toggle.innerHTML = `
      <span class="sr-only">Open navigation</span>
      <svg class="icn nav-toggle-open" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
      <svg class="icn nav-toggle-close" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>
    `;
    shell.insertBefore(toggle, nav);
  }

  enhanceProductNavigation();

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

  function updateScrollableCodeRegions() {
    $$('pre, .overflow-x-auto').forEach((region) => {
      const overflows = region.scrollWidth > region.clientWidth + 1;
      if (overflows) {
        region.tabIndex = 0;
        region.dataset.keyboardScroll = 'true';
        if (!region.hasAttribute('aria-label')) {
          region.setAttribute(
            'aria-label',
            region.matches('pre') ? 'Scrollable code example' : 'Scrollable content',
          );
        }
      } else if (region.dataset.keyboardScroll === 'true') {
        region.removeAttribute('tabindex');
        region.removeAttribute('data-keyboard-scroll');
        if (['Scrollable code example', 'Scrollable content'].includes(region.getAttribute('aria-label'))) {
          region.removeAttribute('aria-label');
        }
      }
    });
  }

  let codeResizeFrame = 0;
  const queueCodeRegionUpdate = () => {
    window.cancelAnimationFrame(codeResizeFrame);
    codeResizeFrame = window.requestAnimationFrame(updateScrollableCodeRegions);
  };
  queueCodeRegionUpdate();
  window.addEventListener('load', queueCodeRegionUpdate, { once: true });
  window.addEventListener('resize', queueCodeRegionUpdate);
  document.addEventListener('toggle', queueCodeRegionUpdate, true);

  function labelEmbeddedPlayerControls(root = document) {
    $$('.ap-timer[role="textbox"]', root).forEach((timer) => {
      if (!timer.hasAttribute('aria-label')) timer.setAttribute('aria-label', 'Playback time');
    });
  }

  if (document.querySelector('.asciinema-frame')) {
    labelEmbeddedPlayerControls();
    const playerControlObserver = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.addedNodes) {
          if (!(node instanceof Element)) continue;
          if (node.matches('.ap-timer[role="textbox"]')) labelEmbeddedPlayerControls(node.parentElement);
          else if (node.querySelector('.ap-timer[role="textbox"]')) labelEmbeddedPlayerControls(node);
        }
      }
    });
    playerControlObserver.observe(document.body, { childList: true, subtree: true });
  }

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
