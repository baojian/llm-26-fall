export async function initialize() {
  const response = await fetch(new URL('./assets/evaluation-states.json', import.meta.url));
  if (!response.ok) throw new Error('Could not load the teaching evaluation examples.');
  const states = await response.json();
  const plot = document.querySelector('#evaluation-plot');
  const toggle = document.querySelector('#evaluation-toggle');
  const reset = document.querySelector('#reset-button');
  const status = document.querySelector('#evaluation-status');
  let index = 0;
  const update = async () => {
    toggle.disabled = reset.disabled = true;
    try {
      const state = states[index];
      await Plotly.restyle(plot, {
        y: [state.scores],
        text: [state.scores.map(value => value.toFixed(1))],
      }, [0]);
      toggle.textContent = state.label;
      toggle.setAttribute('aria-pressed', String(index === 1));
      status.textContent = state.description;
      plot.dataset.evaluationSize = String(state.size);
    } finally {
      toggle.disabled = reset.disabled = false;
    }
  };
  toggle.addEventListener('click', () => {
    index = (index + 1) % states.length;
    void update();
  });
  reset.addEventListener('click', () => {
    index = 0;
    void update();
  });
  await update();
}
