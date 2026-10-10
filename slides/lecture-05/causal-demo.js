import { initialState, computeAttention, figureFor, describeState } from './attention-demo.js';

export async function initializeCausalDemo() {
  const graph = document.getElementById('attention-visual');
  const response = await fetch(new URL('./assets/attention-values.json', import.meta.url));
  if (!response.ok) throw new Error('The local attention example could not be loaded.');
  const fixture = await response.json();
  let state = { ...initialState };
  let queue = Promise.resolve();
  const maskButton = document.getElementById('attention-mask');
  const valueButton = document.getElementById('attention-value');
  const queryButtons = [...document.querySelectorAll('[data-query]')];

  async function render(snapshot) {
    const figure = figureFor(fixture, snapshot);
    await Plotly.react(graph, figure.data, figure.layout, figure.config);
    graph.setAttribute('aria-label', describeState(fixture, snapshot));
    graph.dataset.query = String(snapshot.query);
    graph.dataset.causal = String(snapshot.causal);
    graph.dataset.changed = String(snapshot.changed);
    graph.dataset.output = JSON.stringify(computeAttention(fixture, snapshot).outputs[snapshot.query]);
    queryButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.query) === snapshot.query)));
    maskButton.textContent = `Mask: ${snapshot.causal ? 'on' : 'off'}`;
    maskButton.setAttribute('aria-pressed', String(snapshot.causal));
    valueButton.textContent = snapshot.changed ? 'Restore V3' : 'Change V3';
    valueButton.setAttribute('aria-pressed', String(snapshot.changed));
  }

  function update(change) {
    state = { ...state, ...change };
    const snapshot = { ...state };
    queue = queue.then(() => render(snapshot)).catch(error => {
      graph.setAttribute('aria-label', `The attention chart could not update: ${error.message}`);
      console.error(error);
    });
  }

  queryButtons.forEach(button => button.addEventListener('click', () => update({ query: Number(button.dataset.query) })));
  maskButton.addEventListener('click', () => update({ causal: !state.causal }));
  valueButton.addEventListener('click', () => update({ changed: !state.changed }));
  document.getElementById('attention-reset').addEventListener('click', () => update(initialState));
  await render(state);
}
