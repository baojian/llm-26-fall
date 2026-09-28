/* Original toy computation shared by the chart and its small numerical tests. */
export const initialState = Object.freeze({ query: 1, causal: true, changed: false });

export function computeAttention(fixture, state) {
  const values = fixture.value.map((row, index) => row.map(value =>
    value + (state.changed && index === 2 ? 10 : 0)));
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
  const annotation = (y, text, size = 26) => ({
    x: 0.59, y, xref: 'paper', yref: 'paper', text,
    xanchor: 'left', align: 'left', showarrow: false,
    font: { family: 'Arial, sans-serif', size, color: '#142e4b' },
  });
  return {
    data: [{
      type: 'heatmap', x: [1, 2, 3], y: [1, 2, 3], z: result.weights,
      zmin: 0, zmax: 1, colorscale: [[0, '#f0f4f7'], [1, '#20578c']],
      showscale: false, text: result.weights.map(row => row.map(value => value.toFixed(3))),
      texttemplate: '%{text}', textfont: { size: 28 },
      hovertemplate: 'Query %{y}<br>Key %{x}<br>Weight %{z:.4f}<extra></extra>',
    }],
    layout: {
      width: 1152, height: 410,
      paper_bgcolor: '#fbfbf9', plot_bgcolor: '#fbfbf9',
      font: { family: 'Arial, sans-serif', size: 26, color: '#202b38' },
      margin: { l: 75, r: 15, t: 65, b: 75 },
      xaxis: { domain: [0, 0.49], tickvals: [1, 2, 3],
        ticktext: ['1: red', '2: key', '3: opens'], title: { text: 'Key position' }, fixedrange: true },
      yaxis: { range: [3.5, 0.5], tickvals: [1, 2, 3],
        ticktext: ['Q1', 'Q2', 'Q3'], fixedrange: true },
      annotations: [
        { x: 0.245, y: 1.2, xref: 'paper', yref: 'paper', text: 'Attention weights',
          showarrow: false, font: { size: 28 } },
        annotation(1.2, `Selected query: ${selected + 1}`, 28),
        annotation(0.9, `Scores: ${vector(result.scores[selected], 0)}`),
        annotation(0.63, `Weights: ${vector(result.weights[selected], 2)}`),
        annotation(0.3, `Output: ${vector(result.outputs[selected], 3)}`, 28),
        annotation(-0.02, state.changed ? 'V3 = (13, 11)' : 'V3 = (3, 1)'),
      ],
      shapes: [{ type: 'rect', x0: 0.5, x1: 3.5, y0: selected + 0.5, y1: selected + 1.5,
        line: { color: '#28673f', width: 4 }, fillcolor: 'rgba(0,0,0,0)' }],
    },
    config: { displayModeBar: false, displaylogo: false, scrollZoom: false, responsive: false },
  };
}

export function describeState(fixture, state) {
  const result = computeAttention(fixture, state);
  return `Query ${state.query + 1}. Causal mask ${state.causal ? 'on' : 'off'}. `
    + `Weights ${result.weights[state.query].map(value => value.toFixed(4)).join(', ')}. `
    + `Output ${result.outputs[state.query].map(value => value.toFixed(4)).join(', ')}. `
    + `Value 3 ${state.changed ? 'changed to 13, 11' : 'is 3, 1'}.`;
}
