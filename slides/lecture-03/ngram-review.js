// Pure calculations and Plotly figure construction for the Lecture 02 recap.
const BLUE = '#20578c';
const GREEN = '#28673f';
const ORANGE = '#a85a29';

export function matchingPairs(model, context) {
  return model.corpus.flatMap((sentence, row) => sentence.slice(0, -1)
    .flatMap((token, index) => token === context
      ? [{ row, index, next: sentence[index + 1] }] : []));
}

export function nextTokenDistribution(model, context, seen = Infinity) {
  const pairs = matchingPairs(model, context).slice(0, seen);
  const counts = model.vocabulary.map(token => pairs.filter(pair => pair.next === token).length);
  return { counts, total: pairs.length,
    probabilities: pairs.length ? counts.map(count => count / pairs.length) : null };
}

export function sampleNext(model, context, draw) {
  if (!(draw >= 0 && draw < 1)) throw new RangeError('Use a draw in [0, 1).');
  const { probabilities } = nextTokenDistribution(model, context);
  if (!probabilities) return null;
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (draw < cumulative) return model.vocabulary[i];
  }
  // Protect the final positive bin against floating-point roundoff.
  return model.vocabulary.findLast((_, i) => probabilities[i] > 0);
}

export function reviewFigure(model, state) {
  const { context, phase, seen, sampled } = state;
  const pairs = matchingPairs(model, context);
  const counting = phase === 'counts';
  const distribution = nextTokenDistribution(model, context, counting ? seen : Infinity);
  const full = nextTokenDistribution(model, context);
  const shown = model.vocabulary.filter((_, i) => full.counts[i] > 0);
  const counts = shown.map(token => distribution.counts[model.vocabulary.indexOf(token)]);
  const active = counting ? pairs.slice(seen - 1, seen) : pairs;
  const annotations = [];
  const annotate = (x, y, text, extra = {}) => annotations.push({
    x, y, xref: 'paper', yref: 'paper', showarrow: false,
    xanchor: 'left', yanchor: 'middle', text, font: { size: 30 }, ...extra,
  });
  annotate(0, 0.97, 'Lecture 02 toy word corpus');
  // Widths reserve room for 30px Arial text and 12px gaps; all positions are local.
  const widths = { BOS: 75, EOS: 75, I: 28, am: 57, Sam: 72, do: 47,
    not: 52, like: 56, eggs: 75, and: 62, ham: 69 };
  model.corpus.forEach((sentence, row) => {
    let x = 0;
    sentence.forEach((token, index) => {
      const isContext = active.some(pair => pair.row === row && pair.index === index);
      const isNext = active.some(pair => pair.row === row && pair.index + 1 === index);
      annotate(x / 1128, 0.78 - row * 0.18, token, {
        bgcolor: isContext ? '#d7e5f1' : isNext ? '#e0efdf' : '#fbfbf9',
        bordercolor: isContext ? BLUE : isNext ? GREEN : '#fbfbf9',
        borderpad: 3,
        font: { size: 30, color: isContext ? BLUE : isNext ? GREEN : '#202b38' },
      });
      x += widths[token];
    });
  });
  annotate(0, 0.21, 'IDs: I → 1, am → 2, Sam → 3');
  const progress = counting ? `Pair ${seen} / ${pairs.length}: ${context} → ${pairs[seen - 1].next}`
    : sampled ? `Sample: ${context} → ${sampled}${sampled === 'EOS' ? ' (stop)' : ''}`
      : `All ${pairs.length} pairs counted; no smoothing.`;
  annotate(0, 0.035, progress);
  annotate(0.855, 0.97, counting ? `Counts after ${context}` : `p(next | ${context})`, { xanchor: 'center' });
  annotate(0.855, 0.015, `${model.vocabulary.length - shown.length} other tokens: 0`,
    { xanchor: 'center', font: { size: 26 } });
  const fractions = counts.map(count => `${count} / ${distribution.total}`);
  return {
    data: [{ type: 'bar', x: shown,
      y: counting ? counts : counts.map(count => count / distribution.total),
      text: counting ? counts.map(String) : fractions, textposition: 'outside',
      textfont: { size: 30 }, cliponaxis: false,
      customdata: counts.map((count, i) => [count, distribution.total, fractions[i]]),
      hovertemplate: `Next: %{x}<br>Count: %{customdata[0]} / %{customdata[1]}`
        + (counting ? '' : '<br>Probability: %{customdata[2]}') + '<extra></extra>',
      marker: { color: shown.map(token => token === sampled ? ORANGE : BLUE) },
    }],
    layout: {
      meta: model, margin: { l: 8, r: 16, t: 12, b: 12 },
      font: { family: 'Arial, sans-serif', size: 30, color: '#202b38' },
      paper_bgcolor: '#fbfbf9', plot_bgcolor: '#fbfbf9',
      showlegend: false, bargap: 0.45, annotations,
      xaxis: { domain: [0.73, 0.99], title: { text: 'Next token', standoff: 8 },
        fixedrange: true, tickfont: { size: 30 }, automargin: false },
      yaxis: { domain: [0.28, 0.85], range: counting ? [0, 3.8] : [0, 1.15],
        title: { text: counting ? 'Count' : 'Probability', standoff: 8 },
        tickvals: counting ? [0, 1, 2, 3] : [0, 0.5, 1],
        fixedrange: true, gridcolor: '#dce2e7', zeroline: true, zerolinecolor: '#ccd5dd' },
    },
    config: { displayModeBar: false, displaylogo: false, responsive: false, scrollZoom: false },
  };
}
