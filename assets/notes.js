/* Bilingual reading state; the text and diagrams are already present in the HTML. */
(() => {
  let language = 'en';
  let view = 'corpus';
  let position;
  const headings = () => [...document.querySelector(`[data-note-language="${language}"]`).querySelectorAll('h2,h3')];
  const panels = [...document.querySelectorAll('.pipeline-panel')];
  const tabs = [...document.querySelectorAll('.pipeline-tabs [role="tab"]')];
  const details = document.querySelector('.reading-sidebar details');
  const compact = matchMedia('(max-width: 760px)');
  details.open = !compact.matches;
  compact.addEventListener('change', event => { details.open = !event.matches; });

  function showPanel(next = view) {
    view = next;
    panels.forEach(panel => {
      const selected = panel.dataset.language === language;
      panel.hidden = !selected || panel.dataset.view !== view;
      panel.dataset.printActive = String(selected);
    });
    tabs.forEach(tab => {
      const selected = tab.dataset.view === view;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      tab.setAttribute('aria-controls', `panel-${tab.dataset.view}-${language}`);
    });
  }

  function setLanguage(next, initial = false) {
    const savedPosition = position;
    position = undefined;
    language = next;
    document.querySelectorAll('[data-note-language], [data-toc-language]').forEach(element => {
      element.hidden = (element.dataset.noteLanguage || element.dataset.tocLanguage) !== language;
    });
    document.title = language === 'zh' ? '分词之前的文本预处理 · 第 1 讲 · CS40008.01' : 'Text preprocessing before tokenization · Lecture 01 · CS40008.01';
    showPanel();
    const hash = decodeURIComponent(location.hash.slice(1));
    if (/^(en|zh)-/.test(hash)) {
      const key = hash.replace(/^(en|zh)-/, '');
      const url = new URL(location.href);
      url.hash = `${language}-${key}`;
      history.replaceState(history.state, '', url);
      if (initial) document.getElementById(`${language}-${key}`)?.scrollIntoView();
    }
    if (!initial && savedPosition) {
      const target = document.getElementById(`${language}-${savedPosition.key}`);
      if (target) window.scrollBy(0, target.getBoundingClientRect().top - savedPosition.offset);
    }
    updateTOC();
  }

  function updateTOC() {
    const current = headings().filter(heading => heading.tagName === 'H2' && heading.getBoundingClientRect().top <= 140).at(-1);
    document.querySelectorAll('.reading-sidebar a[data-section]').forEach(link => {
      if (link.dataset.section === current?.dataset.section) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { updateTOC(); ticking = false; });
  }, { passive: true });
  window.addEventListener('course:before-language', () => {
    const anchor = headings().filter(heading => heading.getBoundingClientRect().top <= 120).at(-1);
    position = anchor ? { key: anchor.dataset.section, offset: anchor.getBoundingClientRect().top } : undefined;
  });
  window.addEventListener('course:language', event => setLanguage(event.detail.language));
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => showPanel(tab.dataset.view));
    tab.addEventListener('keydown', event => {
      const delta = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
      if (!delta && !['Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const target = event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs.at(-1) : tabs[(index + delta + tabs.length) % tabs.length];
      showPanel(target.dataset.view);
      target.focus();
    });
  });
  document.querySelector('.print-note').addEventListener('click', () => window.print());
  document.querySelectorAll('.copy-code').forEach(button => {
    button.addEventListener('click', async () => {
      const code = button.parentElement.querySelector('code');
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = language === 'zh' ? '已复制' : 'Copied';
      } catch {
        const range = document.createRange();
        range.selectNodeContents(code);
        getSelection().removeAllRanges();
        getSelection().addRange(range);
        button.textContent = language === 'zh' ? '已选中，请复制' : 'Selected; copy manually';
      }
      setTimeout(() => { button.textContent = language === 'zh' ? '复制' : 'Copy'; }, 2000);
    });
  });
  window.addEventListener('hashchange', () => {
    const match = location.hash.match(/^#(en|zh)-/);
    if (match && match[1] !== language) {
      CourseLanguage.set(match[1]);
      document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
    }
  });
  setLanguage(CourseLanguage.get(), true);
})();
