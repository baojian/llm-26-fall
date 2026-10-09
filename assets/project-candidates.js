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
    expand.textContent = visible.length && visible.every(card => card.querySelector('details').open)
      ? 'Collapse visible project details' : 'Expand visible project details';
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
    counter.textContent = `${count} of ${cards.length} projects`;
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
  form.addEventListener('reset', () => queueMicrotask(update));
  expand.addEventListener('click', () => {
    const visible = cards.filter(card => !card.hidden);
    const open = visible.some(card => !card.querySelector('details').open);
    visible.forEach(card => { card.querySelector('details').open = open; });
    updateExpandLabel();
  });
  cards.forEach(card => card.querySelector('details').addEventListener('toggle', updateExpandLabel));
  window.addEventListener('hashchange', revealLinkedTarget);
  let printState;
  window.addEventListener('beforeprint', () => {
    printState = allCards.map(card => card.querySelector('details').open);
    allCards.filter(card => !card.hidden).forEach(card => { card.querySelector('details').open = true; });
  });
  window.addEventListener('afterprint', () => {
    if (printState) allCards.forEach((card, i) => { card.querySelector('details').open = printState[i]; });
  });
  update();
  revealLinkedTarget();
})();
