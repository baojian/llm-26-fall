/* Progressive enhancement: every candidate and source is readable without JS. */
(() => {
  const form = document.getElementById('catalog-filters');
  const search = document.getElementById('project-search');
  const category = document.getElementById('project-category');
  const topic = document.getElementById('project-topic');
  const compute = document.getElementById('project-compute');
  const cards = [...document.querySelectorAll('.project-card')];
  const counter = document.getElementById('result-count');
  const empty = document.getElementById('no-results');
  const expand = document.getElementById('expand-projects');

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
    const count = cards.filter(card => !card.hidden).length;
    counter.textContent = `${count} of ${cards.length} projects`;
    empty.hidden = count > 0;
    expand.disabled = count === 0;
    expand.textContent = 'Expand visible project details';
  }

  function revealLinkedProject() {
    let identifier;
    try { identifier = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const card = cards.find(item => item.id === identifier);
    if (!card) return;
    // A bookmarked project remains reachable even while other filters are active.
    if (card.hidden) {
      form.reset();
      update();
    }
    card.querySelector('details').open = true;
    card.scrollIntoView({ block: 'start' });
  }

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
    expand.textContent = open ? 'Collapse visible project details' : 'Expand visible project details';
  });
  window.addEventListener('hashchange', revealLinkedProject);
  let printState;
  window.addEventListener('beforeprint', () => {
    printState = cards.map(card => card.querySelector('details').open);
    cards.filter(card => !card.hidden).forEach(card => { card.querySelector('details').open = true; });
  });
  window.addEventListener('afterprint', () => {
    if (printState) cards.forEach((card, i) => { card.querySelector('details').open = printState[i]; });
  });
  update();
  revealLinkedProject();
})();
