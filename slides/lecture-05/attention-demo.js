/* Original toy computation shared by the chart and its small numerical tests. */
export const initialState = Object.freeze({ query: 5, causal: true, changed: false });

export function computeAttention(fixture, state) {
  const values = fixture.value.map((row, index) => row.map(value =>
    value + (state.changed && index === fixture.future_position ? 10 : 0)));
  const scores = fixture.query.map(query => fixture.key.map(key =>
    query.reduce((sum, value, index) => sum + value * key[index], 0)
      / Math.sqrt(query.length)));
  const weights = scores.map((row, query) => {
    const masked = row.map((value, key) => state.causal && key > query ? -Infinity : value);
    const maximum = Math.max(...masked);
    const numerators = masked.map(value => Math.exp(value - maximum));
    const total = numerators.reduce((sum, value) => sum + value, 0);
    return numerators.map(value => value / total);
  });
  const outputs = weights.map(row => values[0].map((_, dimension) =>
    row.reduce((sum, weight, key) => sum + weight * values[key][dimension], 0)));
  return { scores, weights, outputs, values };
}

export function figureFor(fixture, state) {
  const result = computeAttention(fixture, state);
  const selected = state.query;
  const vector = (row, digits) => `(${row.map(value => value.toFixed(digits)).join(', ')})`;
  const positions = fixture.labels.map((_, index) => index);
  const future = fixture.future_position;
  const annotation = (y, text, size = 26) => ({
    x: 0.65, y, xref: 'paper', yref: 'paper', text,
    xanchor: 'left', align: 'left', showarrow: false,
    font: { family: 'Arial, sans-serif', size, color: '#142e4b' },
  });
  return {
    data: [{
      type: 'heatmap', x: positions, y: positions, z: result.weights,
      zmin: 0, zmax: 1, colorscale: [[0, '#f0f4f7'], [1, '#20578c']],
      showscale: false, text: result.weights.map(row => row.map(value => value.toFixed(2))),
      texttemplate: '%{text}', textfont: { size: 26 },
      hovertemplate: 'Query %{y}<br>Key %{x}<br>Weight %{z:.4f}<extra></extra>',
    }],
    layout: {
      width: 1152, height: 430,
      paper_bgcolor: '#fbfbf9', plot_bgcolor: '#fbfbf9',
      font: { family: 'Arial, sans-serif', size: 26, color: '#202b38' },
      margin: { l: 70, r: 10, t: 60, b: 100 },
      xaxis: { domain: [0, 0.58], tickvals: positions, tickangle: -25,
        ticktext: fixture.labels, fixedrange: true },
      yaxis: { range: [positions.length - 0.5, -0.5], tickvals: positions,
        ticktext: positions.map(position => String(position)), title: { text: 'Query position' }, fixedrange: true },
      annotations: [
        { x: 0.29, y: 1.2, xref: 'paper', yref: 'paper', text: 'Attention weights',
          showarrow: false, font: { size: 28 } },
        annotation(1.16, `Query: ${fixture.labels[selected]} (${selected})`, 28),
        annotation(0.88, state.causal ? `Allowed: positions 0–${selected}` : 'Allowed: all seven positions'),
        annotation(0.61, `Weight on is: ${result.weights[selected][future].toFixed(3)}`),
        annotation(0.32, `Output: ${vector(result.outputs[selected], 3)}`, 28),
        annotation(0.02, `V_is = ${vector(result.values[future], 0)}`),
      ],
      shapes: [{ type: 'rect', x0: -0.5, x1: positions.length - 0.5, y0: selected - 0.5, y1: selected + 0.5,
        line: { color: '#28673f', width: 4 }, fillcolor: 'rgba(0,0,0,0)' }],
    },
    config: { displayModeBar: false, displaylogo: false, scrollZoom: false, responsive: false },
  };
}

export function describeState(fixture, state) {
  const result = computeAttention(fixture, state);
  return `Query ${fixture.labels[state.query]} at position ${state.query}. Causal mask ${state.causal ? 'on' : 'off'}. `
    + 'Columns are key tokens; rows are query positions. '
    + `Weights ${result.weights[state.query].map(value => value.toFixed(4)).join(', ')}. `
    + `Output ${result.outputs[state.query].map(value => value.toFixed(4)).join(', ')}. `
    + `Value for is at position ${fixture.future_position}: ${result.values[fixture.future_position].join(', ')}.`;
}
