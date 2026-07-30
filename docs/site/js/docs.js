(() => {
  const body = document.body;
  const menuButton = document.querySelector('[data-docs-menu]');
  const sidebar = document.getElementById('docs-sidebar');

  function closeMenu({ restoreFocus = false } = {}) {
    if (!menuButton) return;
    body.classList.remove('docs-nav-open');
    menuButton.setAttribute('aria-expanded', 'false');
    menuButton.setAttribute('aria-label', 'Menu, open documentation navigation');
    menuButton.title = 'Open navigation';
    if (restoreFocus) menuButton.focus();
  }

  if (menuButton && sidebar) {
    menuButton.addEventListener('click', () => {
      const opening = !body.classList.contains('docs-nav-open');
      body.classList.toggle('docs-nav-open', opening);
      menuButton.setAttribute('aria-expanded', String(opening));
      menuButton.setAttribute('aria-label', opening ? 'Menu, close documentation navigation' : 'Menu, open documentation navigation');
      menuButton.title = opening ? 'Close navigation' : 'Open navigation';
    });

    sidebar.addEventListener('click', (event) => {
      if (event.target.closest('a') && window.matchMedia('(max-width: 820px)').matches) closeMenu();
    });

    document.addEventListener('click', (event) => {
      if (!body.classList.contains('docs-nav-open')) return;
      if (sidebar.contains(event.target) || menuButton.contains(event.target)) return;
      closeMenu();
    });
  }

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && body.classList.contains('docs-nav-open')) {
      closeMenu({ restoreFocus: true });
    }
  });

  const currentSidebarLink = sidebar?.querySelector('[aria-current="page"]');
  currentSidebarLink?.scrollIntoView({ block: 'nearest' });

  document.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-copy-code]');
    if (!button) return;
    const code = button.closest('.docs-code')?.querySelector('code');
    if (!code) return;
    const value = code.textContent.replaceAll('\u00a0', ' ');
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = value;
      textarea.readOnly = true;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.append(textarea);
      textarea.select();
      document.execCommand('copy');
      textarea.remove();
    }
    button.dataset.copied = 'true';
    button.textContent = 'Copied';
    window.setTimeout(() => {
      delete button.dataset.copied;
      button.textContent = 'Copy';
    }, 1400);
  });

  const search = document.querySelector('[data-docs-search]');
  const searchInput = search?.querySelector('input[type="search"]');
  const searchResults = search?.querySelector('[data-docs-search-results]');
  const docsRoot = body.dataset.docsRoot || '.';
  let indexPromise = null;
  let activeResult = -1;

  function loadSearchIndex() {
    if (!indexPromise) {
      indexPromise = fetch(`${docsRoot}/search-index.json`, { credentials: 'same-origin' })
        .then((response) => {
          if (!response.ok) throw new Error(`search index returned HTTP ${response.status}`);
          return response.json();
        });
    }
    return indexPromise;
  }

  function searchScore(entry, terms) {
    const title = entry.title.toLowerCase();
    const headings = entry.headings.join(' ').toLowerCase();
    const content = entry.content.toLowerCase();
    let score = 0;
    for (const term of terms) {
      if (!title.includes(term) && !headings.includes(term) && !content.includes(term)) return -1;
      if (title === term) score += 60;
      else if (title.startsWith(term)) score += 30;
      else if (title.includes(term)) score += 20;
      if (headings.includes(term)) score += 8;
      if (content.includes(term)) score += 1;
    }
    return score;
  }

  function setActiveResult(next) {
    const links = Array.from(searchResults.querySelectorAll('.docs-search-result'));
    if (links.length === 0) {
      activeResult = -1;
      return;
    }
    activeResult = ((next % links.length) + links.length) % links.length;
    links.forEach((link, index) => {
      if (index === activeResult) {
        link.dataset.active = 'true';
        link.scrollIntoView({ block: 'nearest' });
      } else {
        delete link.dataset.active;
      }
    });
  }

  function renderSearchResults(entries, query) {
    searchResults.replaceChildren();
    activeResult = -1;
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (terms.length === 0) {
      searchResults.hidden = true;
      return;
    }
    const matches = entries
      .map((entry) => ({ entry, score: searchScore(entry, terms) }))
      .filter((result) => result.score >= 0)
      .sort((left, right) => right.score - left.score || left.entry.title.localeCompare(right.entry.title))
      .slice(0, 10);

    if (matches.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'docs-search-empty';
      empty.textContent = 'No matching documentation';
      searchResults.append(empty);
    } else {
      for (const { entry } of matches) {
        const link = document.createElement('a');
        link.className = 'docs-search-result';
        link.href = docsRoot === '.'
          ? entry.url
          : entry.url.startsWith('../')
            ? `../${entry.url}`
            : `${docsRoot}/${entry.url.replace(/^\.\//, '')}`;
        const title = document.createElement('strong');
        title.textContent = entry.title;
        const context = document.createElement('span');
        context.textContent = `${entry.section} · ${entry.description}`;
        link.append(title, context);
        searchResults.append(link);
      }
    }
    searchResults.hidden = false;
  }

  if (searchInput && searchResults) {
    searchInput.addEventListener('input', async () => {
      try {
        renderSearchResults(await loadSearchIndex(), searchInput.value);
      } catch {
        searchResults.hidden = true;
      }
    });
    searchInput.addEventListener('focus', () => {
      if (searchInput.value.trim()) searchInput.dispatchEvent(new Event('input'));
      else loadSearchIndex().catch(() => {});
    });
    searchInput.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        setActiveResult(activeResult + 1);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        setActiveResult(activeResult - 1);
      } else if (event.key === 'Enter' && activeResult >= 0) {
        const link = searchResults.querySelectorAll('.docs-search-result')[activeResult];
        if (link) window.location.assign(link.href);
      } else if (event.key === 'Escape') {
        searchResults.hidden = true;
        activeResult = -1;
      }
    });
    document.addEventListener('click', (event) => {
      if (!search.contains(event.target)) searchResults.hidden = true;
    });
  }

  const tocLinks = Array.from(document.querySelectorAll('.docs-toc a[href^="#"]'));
  if (tocLinks.length > 0 && 'IntersectionObserver' in window) {
    const linksById = new Map(tocLinks.map((link) => [decodeURIComponent(link.hash.slice(1)), link]));
    const visible = new Map();
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) visible.set(entry.target.id, entry.isIntersecting ? entry.boundingClientRect.top : null);
      const active = [...visible.entries()]
        .filter(([, top]) => top !== null)
        .sort((left, right) => left[1] - right[1])[0]?.[0];
      for (const [id, link] of linksById) {
        if (id === active) link.setAttribute('aria-current', 'true');
        else link.removeAttribute('aria-current');
      }
    }, { rootMargin: '-70px 0px -72% 0px', threshold: [0, 1] });
    for (const id of linksById.keys()) {
      const heading = document.getElementById(id);
      if (heading) observer.observe(heading);
    }
  }
})();
