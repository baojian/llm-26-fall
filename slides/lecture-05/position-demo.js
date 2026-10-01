/* The unmasked E02 fixture; token labels do not imply trained semantics. */
export const initialState = Object.freeze({ positions: false, swapped: false });

export function sinusoidalPositions(length, width) {
  if (width <= 0 || width % 2) throw new RangeError('Use a positive even width.');
  return Array.from({ length }, (_, position) => Array.from({ length: width }, (_, dimension) => {
    const angle = position / 10000 ** (2 * Math.floor(dimension / 2) / width);
    return dimension % 2 ? Math.cos(angle) : Math.sin(angle);
  }));
}

function attention(rows) {
  const scores = rows.map(query => rows.map(key =>
    query.reduce((sum, value, d) => sum + value * key[d], 0) / Math.sqrt(query.length)));
  const weights = scores.map(row => {
    const maximum = Math.max(...row);
    const numerators = row.map(value => Math.exp(value - maximum));
    const total = numerators.reduce((a, b) => a + b, 0);
    return numerators.map(value => value / total);
  });
  const outputs = weights.map(row => rows[0].map((_, d) =>
    row.reduce((sum, weight, key) => sum + weight * rows[key][d], 0)));
  return { weights, outputs };
}

export function computePositionExample(fixture, state) {
  const order = state.swapped ? fixture.permutation : fixture.tokens.map((_, index) => index);
  const positions = sinusoidalPositions(fixture.tokens.length, fixture.vectors[0].length);
  const addPositions = (row, slot) => row.map((value, d) =>
    value + (state.positions ? positions[slot][d] : 0));
  const baseline = attention(fixture.vectors.map(addPositions));
  const current = attention(order.map((source, slot) => addPositions(fixture.vectors[source], slot)));
  const restored = fixture.tokens.map((_, token) => current.outputs[order.indexOf(token)]);
  const difference = Math.max(...restored.flatMap((row, i) =>
    row.map((value, j) => Math.abs(value - baseline.outputs[i][j]))));
  return { ...current, restored, difference, order, labels: order.map(index => fixture.tokens[index]) };
}

export function describeState(fixture, state) {
  const result = computePositionExample(fixture, state);
  return 'Unmasked toy attention. Positions ' + (state.positions ? 'on' : 'off')
    + '. Order: ' + result.labels.join(', ')
    + '. Maximum output change after undoing the row permutation: ' + result.difference.toFixed(4)
    + '. Q, K, and V use identity projections on invented four-dimensional inputs.';
}

export function figureFor(fixture, state) {
  const result = computePositionExample(fixture, state);
  const note = (y, text, size = 26) => ({
    x: 0.62, y, xref: 'paper', yref: 'paper', text, showarrow: false,
    xanchor: 'left', align: 'left', font: { family: 'Arial, sans-serif', size, color: '#142e4b' },
  });
  return {
    data: [{
      type: 'heatmap', x: [0, 1, 2, 3], y: [0, 1, 2, 3], z: result.weights,
      zmin: 0, zmax: 1, colorscale: [[0, '#f0f4f7'], [1, '#20578c']],
      showscale: false, text: result.weights.map(row => row.map(value => value.toFixed(2))),
      texttemplate: '%{text}', textfont: { size: 28 },
      hovertemplate: 'Query slot %{y}<br>Key slot %{x}<br>Weight %{z:.4f}<extra></extra>',
    }],
    layout: {
      width: 1152, height: 410, paper_bgcolor: '#fbfbf9', plot_bgcolor: '#fbfbf9',
      font: { family: 'Arial, sans-serif', size: 26, color: '#202b38' },
      margin: { l: 90, r: 20, t: 40, b: 70 },
      xaxis: { domain: [0, 0.51], tickvals: [0, 1, 2, 3], ticktext: result.labels,
        title: { text: 'Key token' }, fixedrange: true },
      yaxis: { range: [3.5, -0.5], tickvals: [0, 1, 2, 3], ticktext: result.labels,
        fixedrange: true },
      annotations: [
        { x: 0.255, y: 1.13, xref: 'paper', yref: 'paper', text: 'Attention weights',
          showarrow: false, font: { size: 28 } },
        note(1.04, state.positions ? 'Sinusoidal positions' : 'No position vectors', 28),
        note(0.79, state.swapped ? 'bank / river swapped' : 'Original token order'),
        note(0.53, 'Compare the same token<br>after restoring row order'),
        note(0.21, 'Max change: ' + result.difference.toFixed(4), 28),
        note(-0.02, 'Unmasked; Q = K = V', 24),
      ],
    },
    config: { displayModeBar: false, displaylogo: false, scrollZoom: false, responsive: false },
  };
}
