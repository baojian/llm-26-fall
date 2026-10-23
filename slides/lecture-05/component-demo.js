// Follow each diagram's teaching sequence in either design.
// Explicit visibility avoids SVG opacity layers clipping newly shown artwork.
import { initializePositions } from './position-interactions.js';

export async function initialize(Reveal) {
  const parameters = new URLSearchParams(location.search);
  const original = parameters.get('design') === 'original';
  const diagrams = await Promise.all([...document.querySelectorAll('.attention-diagram,.position-diagram,.teaching-diagram')].map(async container => {
    // Keep the running example in both teaching diagrams. Only the complete
    // GPT architecture has a separate reference view in the integrated deck.
    const reference = original && container.dataset.sourceSlide === '40';
    const design = reference ? '' : '-modern';
    const suffix = container.dataset.sourceSlide === '24' ? '' : `-${container.dataset.sourceSlide}`;
    const artwork = container.dataset.diagram || `self-attention${suffix}${design}`;
    const manifest = container.dataset.diagram || `animation${suffix}${design}`;
    const [response, metadata] = await Promise.all([
      fetch(`assets/${artwork}.svg`),
      fetch(`assets/${manifest}.json`),
    ]);
    if (!response.ok || !metadata.ok) throw new Error(`The diagram ${artwork} could not be loaded.`);
    const xml = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
    if (xml.querySelector('parsererror')) throw new Error(`Invalid diagram SVG: ${artwork}.`);
    // Standalone artwork is complete; hide fragments before interactive insertion.
    if (!Reveal.isPrintView()) {
      xml.querySelectorAll('[data-appear]').forEach(group => group.setAttribute('visibility', 'hidden'));
    }
    container.dataset.design = reference ? 'original' : 'modern';
    container.append(document.importNode(xml.documentElement, true));
    const fragments = [...container.querySelectorAll('.fragment')];
    fragments.forEach(fragment => {
      if (!fragment.dataset.animationObject) fragment.dataset.animationObject = fragment.dataset.sourceShape;
    });
    return { container, fragments, section: container.closest('section'), manifest: await metadata.json() };
  }));
  const positions = initializePositions();
  const sections = [...document.querySelectorAll('.slides > section')];
  const countSteps = section => Math.max(0, ...[...section.querySelectorAll('.fragment')]
    .map(fragment => Number(fragment.dataset.fragmentIndex) + 1));

  const status = document.getElementById('step-status');
  const previous = document.getElementById('previous-step');
  const next = document.getElementById('next-step');
  const designLink = document.getElementById('design-switch');
  designLink.textContent = original ? 'Modern' : 'Original';
  document.getElementById('print-link').href = original ? '?print-pdf&design=original' : '?print-pdf';

  function update() {
    const printed = Reveal.isPrintView();
    for (const diagram of diagrams) {
      let step = 0;
      for (const fragment of diagram.fragments) {
        const visible = printed || fragment.classList.contains('visible');
        fragment.querySelector('[data-appear]')?.setAttribute('visibility', visible ? 'visible' : 'hidden');
        fragment.setAttribute('aria-hidden', String(!visible));
        if (visible) step = Math.max(step, Number(fragment.dataset.fragmentIndex) + 1);
      }
      diagram.container.dataset.step = String(step);
    }
    const current = Reveal.getCurrentSlide();
    if (!current) return;
    const diagram = diagrams.find(item => item.section === current);
    const count = countSteps(current);
    const step = Math.max(0, ...[...current.querySelectorAll('.fragment')]
      .filter(fragment => printed || fragment.classList.contains('visible'))
      .map(fragment => Number(fragment.dataset.fragmentIndex) + 1));
    const interactive = Boolean(current.querySelector('[data-interactive],[data-plotly]'));
    const reference = diagram?.container.dataset.design === 'original' && count === 0;
    status.textContent = count ? `Step ${step} / ${count}` : reference ? 'Reference' : interactive ? 'Explore' : 'Overview';
    status.title = count ? (step ? diagram?.manifest.steps[step - 1]?.label || 'Answer revealed' : 'Press Space, Right, or Next to begin')
      : interactive ? 'Use the controls or hover over the chart' : 'Complete diagram';
    previous.disabled = current === sections[0] && step === 0;
    next.disabled = current === sections.at(-1) && step === count;
    document.getElementById('reset-animation').disabled = count === 0 && !current.querySelector('[data-interactive]');
    designLink.hidden = diagram?.container.dataset.sourceSlide !== '40';
    if (!designLink.hidden) {
      designLink.textContent = original ? 'Modern' : diagram.container.dataset.referenceLabel || 'Original';
      designLink.href = `${original ? './' : '?design=original'}#/${diagram.section.id}`;
    }
    document.getElementById('source-link').href = current.querySelector('[data-source-url]')?.dataset.sourceUrl
      || (current.id === 'sinusoidal-example' ? 'https://arxiv.org/html/1706.03762v7#S3.S5' : 'https://www.youtube.com/watch?v=tIvKXrEDMhk');
  }
  const observer = new MutationObserver(update);
  observer.observe(document.querySelector('.slides'), { subtree: true, attributes: true, attributeFilter: ['class'] });
  for (const diagram of diagrams) {
    if (!diagram.container.dataset.interactive) diagram.container.addEventListener('click', () => Reveal.next());
  }
  Reveal.sync();
  // Reveal reads the route before the fetched SVG fragments exist. Restore
  // a named fragment link once those fragments have joined the slide.
  const requested = /^#\/([a-z0-9-]+)\/(-?\d+)$/.exec(window.lectureStartHash || '');
  if (requested && !Reveal.isPrintView()) {
    const section = sections.find(item => item.id === requested[1]);
    if (section) {
      const { h, v } = Reveal.getIndices(section);
      const fragment = Math.max(-1, Math.min(Number(requested[2]), countSteps(section) - 1));
      Reveal.slide(h, v, fragment);
    }
  }
  Reveal.on('fragmentshown', update);
  Reveal.on('fragmenthidden', update);
  Reveal.on('slidechanged', update);
  previous.addEventListener('click', () => Reveal.prev());
  next.addEventListener('click', () => Reveal.next());
  document.getElementById('reset-animation').addEventListener('click', () => {
    positions.reset(Reveal.getCurrentSlide());
    const { h, v } = Reveal.getIndices();
    Reveal.slide(h, v, -1);
    update();
  });
  document.getElementById('notebook-link').hidden = false;
  update();
}
