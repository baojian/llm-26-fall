import { matchingPairs, nextTokenDistribution, reviewFigure, sampleNext } from './ngram-review.js';

// Small, deterministic classroom mechanisms. All values and assets stay local.
const SVG = 'http://www.w3.org/2000/svg';
const colors = { ink: '#142e4b', blue: '#20578c', green: '#28673f', orange: '#a85a29' };

function element(tag, attributes, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function label(svg, x, y, text, color = colors.ink, anchor = 'start') {
  svg.append(element('text', { x, y, fill: color, 'font-size': 30,
    'font-family': 'Arial, sans-serif', 'text-anchor': anchor }, text));
}

function rectangle(svg, x, y, width, height, fill, stroke = colors.blue) {
  svg.append(element('rect', { x, y, width, height, fill, stroke, 'stroke-width': 2 }));
}

function canvas(id, description) {
  const svg = document.getElementById(id);
  svg.replaceChildren();
  svg.setAttribute('aria-label', description);
  return svg;
}

export async function initialize() {
  await initializeNgramReview();
  const response = await fetch(new URL('assets/lookup-values.json', import.meta.url));
  if (!response.ok) throw new Error('The lookup example could not load.');
  const { values } = await response.json();
  const lookupForm = document.getElementById('lookup-controls');
  const memoryForm = document.getElementById('memory-controls');
  lookupForm.addEventListener('submit', event => event.preventDefault());
  memoryForm.addEventListener('submit', event => event.preventDefault());

  function lookup(id) {
    const vector = values[id].map(value => value.toFixed(2)).join(', ');
    const svg = canvas('lookup-visual', `Toy E01 table, 10 rows by 4 values. ID ${id} selects row ${id}: ${vector}. Repeating this ID returns the same row.`);
    svg.dataset.selectedId = id;
    label(svg, 20, 42, 'Token ID', colors.blue);
    rectangle(svg, 40, 120, 120, 76, '#e8f0f7');
    label(svg, 100, 168, id, colors.blue, 'middle');
    label(svg, 245, 42, 'E: four of ten rows shown');
    for (const [position, row] of [0, 1, 5, 7].entries()) {
      const y = 67 + position * 54;
      const selected = row === id;
      rectangle(svg, 290, y, 415, 48, selected ? '#dcecdf' : '#f4f6f8', selected ? colors.green : '#ccd5dd');
      label(svg, 265, y + 34, row, selected ? colors.green : colors.ink, 'end');
      label(svg, 310, y + 34, values[row].map(value => value.toFixed(2)).join('   '));
    }
    label(svg, 775, 42, `E[${id}]`, colors.green);
    const rowY = 91 + [0, 1, 5, 7].indexOf(id) * 54;
    svg.append(element('path', { d: `M 165 158 H 212 V ${rowY} H 232 M 226 ${rowY - 6} L 234 ${rowY} L 226 ${rowY + 6}`,
      fill: 'none', stroke: colors.green, 'stroke-width': 3 }));
    label(svg, 734, 151, '→', colors.green, 'middle');
    for (let i = 0; i < 4; i++) {
      rectangle(svg, 780 + i * 90, 110, 86, 64, '#eaf3ed', colors.green);
      label(svg, 823 + i * 90, 152, values[id][i].toFixed(2), colors.green, 'middle');
    }
    label(svg, 775, 219, 'Same ID → same row');
    label(svg, 775, 262, 'One-hot × E = E[id]');
    lookupForm.querySelectorAll('[data-token-id]').forEach(button => {
      button.setAttribute('aria-pressed', String(Number(button.dataset.tokenId) === id));
    });
    document.getElementById('lookup-status').textContent = `ID ${id} selects a trainable vector; the table is trained with the model.`;
  }
  lookupForm.querySelectorAll('[data-token-id]').forEach(button => {
    button.addEventListener('click', () => lookup(Number(button.dataset.tokenId)));
  });
  document.getElementById('lookup-reset').addEventListener('click', () => lookup(5));

  let rows = 32000;
  let width = 512;
  function memory() {
    const parameters = rows * width;
    const unitMiB = parameters * 4 / 2 ** 20;
    const total = 4 * unitMiB;
    const svg = canvas('memory-visual', `fp32 tensor payload for ${parameters.toLocaleString('en-US')} unique parameters: parameters ${unitMiB} MiB, gradients ${unitMiB} MiB, Adam moments ${2 * unitMiB} MiB; subtotal ${total} MiB. Additional memory is excluded.`);
    svg.dataset.totalMib = total;
    label(svg, 15, 34, `P = ${rows.toLocaleString('en-US')} × ${width} = ${parameters.toLocaleString('en-US')} unique parameters`);
    const items = [['Parameters', 1, colors.blue], ['Gradients', 1, colors.green], ['Adam moments', 2, colors.orange]];
    for (const [i, [name, multiplier, color]] of items.entries()) {
      const y = 63 + i * 61;
      label(svg, 15, y + 33, name);
      // A fixed 0–500 MiB scale makes growth visible when dimensions change.
      rectangle(svg, 300, y, unitMiB * multiplier / 500 * 590, 42, color, color);
      label(svg, 930, y + 33, `${unitMiB * multiplier} MiB`);
    }
    svg.append(element('path', { d: 'M 300 241 V 254 M 300 247 H 890 M 595 241 V 254 M 890 241 V 254',
      fill: 'none', stroke: '#8495a5', 'stroke-width': 2 }));
    label(svg, 300, 285, '0', colors.ink, 'middle');
    label(svg, 595, 285, '250', colors.ink, 'middle');
    label(svg, 890, 285, '500 MiB', colors.ink, 'middle');
    label(svg, 15, 331, `Subtotal: ${total} MiB = 16 bytes per parameter`, colors.green);
    document.getElementById('memory-rows').textContent = `Rows: ${rows.toLocaleString('en-US')}`;
    document.getElementById('memory-width').textContent = `Width: ${width}`;
  }
  document.getElementById('memory-rows').addEventListener('click', () => { rows = rows === 32000 ? 64000 : 32000; memory(); });
  document.getElementById('memory-width').addEventListener('click', () => { width = width === 512 ? 1024 : 512; memory(); });
  document.getElementById('memory-reset').addEventListener('click', () => { rows = 32000; width = 512; memory(); });
  lookup(5);
  memory();
}


async function initializeNgramReview() {
  const chart = document.getElementById('ngram-visual');
  const controls = document.getElementById('ngram-controls');
  const trace = document.getElementById('ngram-trace');
  const sample = document.getElementById('ngram-sample');
  const model = chart.layout.meta;
  let state;
  let busy = false;

  async function render() {
    busy = true;
    try {
      const figure = reviewFigure(model, state);
      await Plotly.react(chart, figure.data, { ...figure.layout, width: 1152, height: 410 }, figure.config);
      const counting = state.phase === 'counts';
      const total = matchingPairs(model, state.context).length;
      trace.textContent = counting ? state.seen < total ? 'Next pair' : 'Normalize' : 'Trace pairs';
      sample.disabled = counting;
      for (const button of controls.querySelectorAll('[data-ngram-context]')) {
        button.setAttribute('aria-pressed', String(button.dataset.ngramContext === state.context));
      }
      chart.dataset.context = state.context;
      chart.dataset.phase = state.phase;
      chart.dataset.seen = String(counting ? state.seen : total);
      chart.dataset.sampled = state.sampled || '';
      const values = nextTokenDistribution(model, state.context, counting ? state.seen : Infinity);
      const nonzero = model.vocabulary.flatMap((token, i) => values.counts[i]
        ? [`${token}: ${values.counts[i]}${counting ? '' : ` / ${values.total}`}`] : []);
      chart.setAttribute('aria-label', `Toy bigram ${state.phase} after ${state.context}. `
        + nonzero.join('; ') + '. All other output tokens: zero.'
        + (state.sampled ? ` Sampled ${state.sampled}${state.sampled === 'EOS' ? '; stop' : ''}.` : ''));
    } finally {
      busy = false;
    }
  }

  function reset(context = 'I') {
    state = { context, phase: 'probabilities', seen: 0, sampled: null };
    return render();
  }
  controls.addEventListener('submit', event => event.preventDefault());
  controls.addEventListener('click', async event => {
    const button = event.target.closest('button');
    if (!button || busy) return;
    if (button.dataset.ngramContext) return reset(button.dataset.ngramContext);
    if (button.id === 'ngram-reset') return reset();
    if (button.id === 'ngram-trace') {
      const total = matchingPairs(model, state.context).length;
      state.sampled = null;
      if (state.phase !== 'counts') {
        state.phase = 'counts';
        state.seen = 1;
      } else if (state.seen < total) state.seen++;
      else state.phase = 'probabilities';
    } else if (button.id === 'ngram-sample' && state.phase === 'probabilities') {
      state.sampled = sampleNext(model, state.context, Math.random());
    }
    await render();
  });
  await reset();
}
