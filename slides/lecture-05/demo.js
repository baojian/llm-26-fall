import { initializeCausalDemo } from './causal-demo.js';
import { initialState, computePositionExample, figureFor, describeState } from './position-demo.js';
import { initialize as initializeComponents } from './component-demo.js';

export async function initialize(Reveal) {
  const attention = document.getElementById('attention-visual');
  const position = document.getElementById('position-visual');
  if (attention) {
    attention.dataset.interactive = 'causal-attention';
    await initializeCausalDemo();
  }
  if (position) {
    position.dataset.interactive = 'position-permutation';
    await initializePositionDemo();
  }
  await initializeComponents(Reveal);
  // Reveal throttles URL updates for one second. Keep a restored fragment in
  // the address immediately so an early reload preserves the visible stage.
  if (!Reveal.isPrintView() && /^#\/[a-z0-9-]+\/-?\d+$/.test(window.lectureStartHash || '')) {
    const current = Reveal.getCurrentSlide();
    const { f } = Reveal.getIndices();
    const fragment = f >= 0 ? `/${f}` : '';
    history.replaceState(history.state, '', `#/${current.id}${fragment}`);
  }
  document.getElementById('reset-animation').addEventListener('click', () => {
    Reveal.getCurrentSlide()?.querySelector('#attention-reset,#position-reset')?.click();
  });

  // The component walkthrough supplies its own attribution. Original lecture
  // sections without a source control retain their citations in the notes.
  const updateSource = () => {
    const current = Reveal.getCurrentSlide();
    document.getElementById('source-link').hidden = !current || (
      !current.querySelector('[data-source-url],.attention-diagram')
      && current.id !== 'sinusoidal-example'
    );
  };
  Reveal.on('slidechanged', updateSource);
  updateSource();
}

export async function initializePositionDemo() {
  const graph = document.getElementById('position-visual');
  const response = await fetch(new URL('./assets/position-values.json', import.meta.url));
  if (!response.ok) throw new Error('The local position example could not be loaded.');
  const fixture = await response.json();
  const positionButton = document.getElementById('position-toggle');
  const swapButton = document.getElementById('position-swap');
  let state = { ...initialState };
  let queue = Promise.resolve();

  async function render(snapshot) {
    const figure = figureFor(fixture, snapshot);
    await Plotly.react(graph, figure.data, figure.layout, figure.config);
    graph.setAttribute('aria-label', describeState(fixture, snapshot));
    graph.dataset.positions = String(snapshot.positions);
    graph.dataset.swapped = String(snapshot.swapped);
    graph.dataset.difference = String(computePositionExample(fixture, snapshot).difference);
    positionButton.textContent = 'Positions: ' + (snapshot.positions ? 'on' : 'off');
    positionButton.setAttribute('aria-pressed', String(snapshot.positions));
    swapButton.textContent = snapshot.swapped ? 'Restore token order' : 'Swap Noa / she';
    swapButton.setAttribute('aria-pressed', String(snapshot.swapped));
  }

  function update(change) {
    state = { ...state, ...change };
    const snapshot = { ...state };
    queue = queue.then(() => render(snapshot)).catch(error => {
      graph.setAttribute('aria-label', 'The position chart could not update: ' + error.message);
      console.error(error);
    });
  }

  positionButton.addEventListener('click', () => update({ positions: !state.positions }));
  swapButton.addEventListener('click', () => update({ swapped: !state.swapped }));
  document.getElementById('position-reset').addEventListener('click', () => update(initialState));
  await render(state);
}
