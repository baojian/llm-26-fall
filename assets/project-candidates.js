/* Progressive enhancement: every candidate and source is readable without JS. */
(() => {
  const form = document.getElementById('catalog-filters');
  const search = document.getElementById('project-search');
  const category = document.getElementById('project-category');
  const topic = document.getElementById('project-topic');
  const compute = document.getElementById('project-compute');
  const allCards = [...document.querySelectorAll('.project-card')];
  const cards = [...document.querySelectorAll('#project-list .project-card')];
  const groups = [...document.querySelectorAll('.topic-group')];
  const counter = document.getElementById('result-count');
  const empty = document.getElementById('no-results');
  const expand = document.getElementById('expand-projects');
  const toc = document.querySelector('.catalog-toc');
  const tocLinks = [...toc.querySelectorAll('a')];
  const compactLayout = window.matchMedia('(max-width: 980px)');
  const translations = Object.fromEntries(
    Object.entries(window.projectCandidateTranslations || {}).map(([key, value]) => [key.trim(), value])
  );
  const englishTitle = document.title;
  const description = document.querySelector('meta[name="description"]');
  const englishDescription = description.content;
  const searchPlaceholder = search.placeholder;
  const staticTextNodes = [];

  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement.closest('script, style, code, pre, svg, .lang-toggle, [data-en][data-zh]')) continue;
    const original = node.nodeValue;
    const key = original.trim();
    if (Object.prototype.hasOwnProperty.call(translations, key)) {
      staticTextNodes.push({
        node,
        en: original,
        zh: original.replace(key, translations[key]),
      });
    }
  }

  function isChinese() {
    return window.CourseLanguage?.get() === 'zh';
  }

  function translateStatic() {
    const chinese = isChinese();
    staticTextNodes.forEach(entry => { entry.node.nodeValue = chinese ? entry.zh : entry.en; });
    document.title = chinese ? (translations[englishTitle] || englishTitle) : englishTitle;
    description.content = chinese
      ? (translations[englishDescription] || englishDescription)
      : englishDescription;
    search.placeholder = chinese
      ? '试试：上下文、分词器、验证器、中文……'
      : searchPlaceholder;
  }

  function updateCurrentSection() {
    let current = tocLinks[0];
    for (const link of tocLinks) {
      const target = document.getElementById(link.hash.slice(1));
      if (target && !target.hidden && target.getBoundingClientRect().top <= 140) current = link;
    }
    for (const link of tocLinks) {
      if (link === current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }

  function updateExpandLabel() {
    const visible = cards.filter(card => !card.hidden);
    expand.disabled = visible.length === 0;
    const collapse = visible.length && visible.every(card => card.querySelector('details').open);
    expand.textContent = isChinese()
      ? (collapse ? '收起当前显示项目的详细信息' : '展开当前显示项目的详细信息')
      : (collapse ? 'Collapse visible project details' : 'Expand visible project details');
  }

  function update() {
    const words = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    for (const card of cards) {
      card.hidden = !(
        words.every(word => card.dataset.search.includes(word)) &&
        (!category.value || card.dataset.category.split('|').includes(category.value)) &&
        (!topic.value || card.dataset.topic === topic.value) &&
        (!compute.value || card.dataset.compute === compute.value)
      );
    }
    for (const group of groups) {
      group.hidden = ![...group.querySelectorAll('.project-card')].some(card => !card.hidden);
    }
    const count = cards.filter(card => !card.hidden).length;
    counter.textContent = isChinese()
      ? `${count} 个候选项目，共 ${cards.length} 个`
      : `${count} of ${cards.length} projects`;
    empty.hidden = count > 0;
    updateExpandLabel();
    updateCurrentSection();
  }

  function revealLinkedTarget() {
    let identifier;
    try { identifier = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(identifier);
    if (!target) return;
    const group = groups.find(item => item.id === identifier);
    const card = allCards.find(item => item.id === identifier);
    if (group) {
      form.reset();
      topic.value = group.dataset.topic;
      update();
    } else if (identifier === 'catalog' || (card && card.hidden)) {
      // A bookmarked project remains reachable while other filters are active.
      form.reset();
      update();
    }
    if (card) card.querySelector('details').open = true;
    updateExpandLabel();
    target.scrollIntoView({ block: 'start' });
    updateCurrentSection();
  }

  function resizeContents() {
    toc.open = !compactLayout.matches;
    updateCurrentSection();
  }

  resizeContents();
  compactLayout.addEventListener('change', resizeContents);
  tocLinks.forEach(link => link.addEventListener('click', () => {
    if (compactLayout.matches) toc.open = false;
    if (location.hash === link.hash) revealLinkedTarget();
  }));
  let scrollPending = false;
  window.addEventListener('scroll', () => {
    if (scrollPending) return;
    scrollPending = true;
    requestAnimationFrame(() => {
      scrollPending = false;
      updateCurrentSection();
    });
  }, { passive: true });

  form.hidden = false;
  expand.hidden = false;
  form.addEventListener('submit', event => event.preventDefault());
  form.addEventListener('input', update);
  form.addEventListener('change', update);
  // A native reset-button click runs microtasks before the controls reset.
  // Read their restored values in the next task, after the default action.
  form.addEventListener('reset', () => setTimeout(update, 0));
  expand.addEventListener('click', () => {
    const visible = cards.filter(card => !card.hidden);
    const open = visible.some(card => !card.querySelector('details').open);
    visible.forEach(card => { card.querySelector('details').open = open; });
    updateExpandLabel();
  });
  cards.forEach(card => card.querySelector('details').addEventListener('toggle', updateExpandLabel));
  window.addEventListener('course:language', () => {
    translateStatic();
    update();
  });
  window.addEventListener('hashchange', revealLinkedTarget);
  let printState;
  window.addEventListener('beforeprint', () => {
    printState = allCards.map(card => card.querySelector('details').open);
    allCards.filter(card => !card.hidden).forEach(card => { card.querySelector('details').open = true; });
  });
  window.addEventListener('afterprint', () => {
    if (printState) allCards.forEach((card, i) => { card.querySelector('details').open = printState[i]; });
  });
  translateStatic();
  update();
  revealLinkedTarget();
})();
