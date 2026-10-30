export async function initialize() {
  const response = await fetch(new URL('./assets/budget-states.json', import.meta.url));
  if (!response.ok) throw new Error('Could not load the synthetic budget examples.');
  const states = await response.json();
  const plot = document.querySelector('#budget-plot');
  const button = document.querySelector('#budget-button');
  const reset = document.querySelector('#reset-button');
  const status = document.querySelector('#budget-status');
  let index = 0;
  const update = async () => {
    const state = states[index];
    button.disabled = reset.disabled = true;
    try {
      await Plotly.restyle(plot, {
        x: state.data.map(trace => trace.x),
        y: state.data.map(trace => trace.y),
      }, [0, 1]);
      button.textContent = `Budget: ${state.multiple}×`;
      status.textContent = `Synthetic optimum: ${state.parameters / 1e6}M parameters, ${state.tokens / 1e9}B tokens; loss ${state.loss.toFixed(3)}.`;
      plot.dataset.computeMultiple = String(state.multiple);
    } finally {
      button.disabled = reset.disabled = false;
    }
  };
  button.addEventListener('click', () => {
    index = (index + 1) % states.length;
    void update();
  });
  reset.addEventListener('click', () => {
    index = 0;
    void update();
  });
  await update();
}
