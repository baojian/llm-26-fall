import { initialState, computePositionExample, figureFor, describeState } from './position-demo.js';

export async function initialize() {
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
    swapButton.textContent = snapshot.swapped ? 'Restore token order' : 'Swap bank / river';
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
