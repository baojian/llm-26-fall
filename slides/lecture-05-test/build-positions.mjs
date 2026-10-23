// Original editable diagrams and computed curves for the positional-encoding section.
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import katex from 'katex';
import { sinusoidal, toyFrequency, rotatePair, relativeExample } from './position-math.js';

const assets = fileURLToPath(new URL('./assets/', import.meta.url));
const c = { ink: '#142e4b', muted: '#526374', line: '#bccbd7', pale: '#e5edf3',
  blue: '#527596', purple: '#7550b6', purplePale: '#eee8f7', teal: '#147d73', tealPale: '#e5f3ef' };
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const el = (tag, attributes, content = '') => `<${tag} ${Object.entries(attributes).map(([key, value]) => `${key}="${escape(value)}"`).join(' ')}>${content}</${tag}>`;
const text = (x, y, value, { size = 30, fill = c.ink, anchor = 'start', weight = 400, ...attributes } = {}) => el('text', {
  x, y, fill, 'font-size': size, 'text-anchor': anchor, 'font-weight': weight, ...attributes,
}, escape(value));
const rect = (x, y, width, height, fill = c.pale) => el('rect', { x, y, width, height, fill, rx: 12 });
const path = (d, color = c.line, attributes = {}) => el('path', { d, fill: 'none', stroke: color, 'stroke-width': 2.5, ...attributes });
const line = (x1, y1, x2, y2, color = c.line, arrow = false, attributes = {}) => el('line', {
  x1, y1, x2, y2, stroke: color, 'stroke-width': 2.5,
  ...(arrow ? { 'marker-end': `url(#pos-arrow-${color.slice(1)})` } : {}), ...attributes,
});
const math = (x, y, value, { width = 560, height = 60, size = 32, anchor = 'start', fill = c.ink } = {}) => {
  const html = katex.renderToString(value, { output: 'htmlAndMathml', throwOnError: true, strict: 'error', trust: false });
  return el('foreignObject', { x, y, width, height, 'font-size': size, color: fill, 'data-math-box': '' },
    el('div', { xmlns: 'http://www.w3.org/1999/xhtml', class: 'attention-math', 'data-align': anchor, 'data-tex': value }, html));
};
const bullet = (y, value) => el('circle', { cx: 9, cy: y - 10, r: 4, fill: c.ink }) + text(28, y, value);
const cells = (x, y, colors) => colors.map((fill, i) => rect(x + i * 26, y, 21, 22, fill)).join('');
const box = (x, y, width, height, label, fill = c.pale, size = 30) => rect(x, y, width, height, fill)
  + text(x + width / 2, y + height / 2 + size * 0.35, label, { anchor: 'middle', size });
const plus = (x, y) => el('circle', { cx: x, cy: y, r: 19, fill: '#fff', stroke: c.line, 'stroke-width': 2 })
  + text(x, y + 10, '+', { anchor: 'middle', size: 32 });

function diagram(name, title, description, height) {
  const steps = [];
  let body = '';
  return {
    add: content => { body += content; },
    step(label, content) {
      const id = `${name}-${steps.length}`;
      steps.push({ click: steps.length + 1, label, object_ids: [id] });
      body += el('g', { class: 'fragment custom', 'data-fragment-index': steps.length - 1, 'data-animation-object': id },
        el('g', { 'data-appear': '', visibility: 'visible' }, content));
    },
    async save() {
      const definitions = Object.values(c).map(color => el('marker', {
        id: `pos-arrow-${color.slice(1)}`, viewBox: '0 0 10 10', refX: 8, refY: 5,
        markerWidth: 5, markerHeight: 5, orient: 'auto', markerUnits: 'strokeWidth',
      }, el('path', { d: 'M 1 1 L 8 5 L 1 9', fill: 'none', stroke: color, 'stroke-width': 1.8 }))).join('');
      // Prefix IDs per asset so several inline SVGs cannot resolve another slide's marker.
      const svg = el('svg', { xmlns: 'http://www.w3.org/2000/svg', width: 1152, height, viewBox: `0 0 1152 ${height}`,
        'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img', 'aria-labelledby': `${name}-title ${name}-description` },
      el('title', { id: `${name}-title` }, escape(title)) + el('desc', { id: `${name}-description` }, escape(description))
        + el('defs', {}, definitions) + body).replaceAll('pos-arrow-', `${name}-arrow-`);
      await writeFile(`${assets}${name}.svg`, svg + '\n');
      await writeFile(`${assets}${name}.json`, JSON.stringify({ title, steps }, null, 2) + '\n');
    },
  };
}

// The first component after the architecture: identity and position are two inputs.
const inputs = diagram('position-inputs', 'Token identity and position',
  'The same token at positions zero and three has the same lookup embedding. Separate position vectors are added before the Transformer stack. Read upward.', 552);
inputs.add(text(2, 30, 'POSITION · COMPONENT 01', { size: 24, fill: c.muted, weight: 600 })
  + text(640, 45, 'Position 0', { anchor: 'middle' }) + text(970, 45, 'Position 3', { anchor: 'middle' })
  + line(437, 18, 437, 512, c.pale));
let content = bullet(112, 'Same token identity') + text(28, 155, 'Same lookup embedding', { fill: c.blue });
for (const center of [640, 970]) {
  content += box(center - 145, 347, 130, 86, '', c.pale)
    + cells(center - 132, 365, Array(4).fill(c.blue))
    + text(center - 80, 417, 'Token', { anchor: 'middle' })
    + line(center - 80, 470, center - 80, 439, c.blue, true)
    + text(center - 80, 506, '“the”', { anchor: 'middle', weight: 600 });
}
inputs.step('The same token uses the same embedding lookup.', content);
content = bullet(229, 'Add a position vector') + text(28, 272, 'Same width as the embedding', { size: 28, fill: c.muted });
for (const [index, center] of [640, 970].entries()) {
  content += box(center + 15, 347, 130, 86, '', c.purplePale)
    + cells(center + 28, 365, index ? [c.purple, '#c7b6e3', '#9b7ec6', c.purple] : ['#c7b6e3', c.purple, '#c7b6e3', c.purple])
    + text(center + 80, 417, 'Position', { anchor: 'middle', fill: c.purple })
    + line(center + 80, 470, center + 80, 439, c.purple, true)
    + math(center + 30, 477, `p=${index ? 3 : 0}`, { width: 100, height: 46, anchor: 'middle' })
    + path(`M ${center - 80} 347 V 294 H ${center - 20}`, c.blue)
    + path(`M ${center + 80} 347 V 294 H ${center + 20}`, c.purple)
    + plus(center, 294);
}
inputs.step('Add a position vector to each token embedding.', content);
content = bullet(355, 'Different model inputs')
  + math(28, 380, 'x_p=e_{t_p}+PE(p)', { width: 390, height: 66, size: 31 })
  + text(28, 470, 'Add once, before the stack.', { size: 28, fill: c.muted });
for (const [index, center] of [640, 970].entries()) {
  content += line(center, 273, center, 227, c.teal, true)
    + box(center - 101, 133, 202, 88, '', c.tealPale)
    + cells(center - 49, 149, index ? [c.teal, '#6caaa3', '#99c7c0', c.teal] : ['#6caaa3', c.teal, '#6caaa3', c.teal])
    + math(center - 95, 173, `x_${index ? 3 : 0}`, { width: 190, height: 48, anchor: 'middle', fill: c.teal })
    + line(center, 133, center, 75, c.teal, true);
}
content += text(805, 542, 'Schematic vectors; numerical example follows.', { size: 24, anchor: 'middle', fill: c.muted });
inputs.step('Token identity and position together form the model input.', content);
await inputs.save();

// Real values, sampled densely to show the sine curves; tokens occupy integer p.
const positions = Array.from({ length: 513 }, (_, i) => i / 8);
const labels = ['Dimension 0 · fast', 'Dimension 2 · slower', 'Dimension 4 · slower still'];
const colors = [c.purple, c.teal, c.blue];
const figure = {
  data: [0, 2, 4].map((dimension, index) => ({
    type: 'scatter', mode: 'lines', x: positions, y: positions.map(p => sinusoidal(p, 8)[dimension]),
    yaxis: index ? `y${index + 1}` : 'y', name: labels[index],
    line: { color: colors[index], width: 3 },
    hovertemplate: 'Position p = %{x:g}<br>Encoding = %{y:.3f}<extra>' + labels[index] + '</extra>',
  })),
  layout: {
    margin: { l: 92, r: 32, t: 48, b: 65 }, showlegend: false,
    xaxis: { range: [0, 64], title: { text: 'Position p (token index)' }, tickvals: [0, 16, 32, 48, 64], fixedrange: true, anchor: 'y3' },
    annotations: labels.map((label, i) => ({ x: 0.995, y: [1.02, 0.64, 0.26][i], xref: 'paper', yref: 'paper',
      text: label, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 24, color: colors[i] } })),
  },
  config: { displayModeBar: false, responsive: false },
};
for (const [i, domain] of [[0, [0.76, 1]], [1, [0.38, 0.62]], [2, [0, 0.24]]]) {
  figure.layout[i ? `yaxis${i + 1}` : 'yaxis'] = { domain, range: [-1.15, 1.15], tickvals: [-1, 0, 1], fixedrange: true, zerolinecolor: c.line, gridcolor: c.pale };
}
await writeFile(`${assets}position-frequencies.json`, JSON.stringify(figure) + '\n');

function axes(cx, cy, radius) {
  return el('circle', { cx, cy, r: radius, fill: 'none', stroke: c.pale, 'stroke-width': 2 })
    + line(cx - radius - 18, cy, cx + radius + 22, cy, c.line, true)
    + line(cx, cy + radius + 16, cx, cy - radius - 20, c.line, true)
    + text(cx + radius + 15, cy + 34, '1', { size: 24, anchor: 'middle', fill: c.muted })
    + text(cx - 22, cy - radius - 10, '1', { size: 24, anchor: 'middle', fill: c.muted });
}
function ray(cx, cy, radius, angle, color, attributes = {}) {
  const [x, y] = rotatePair([radius, 0], angle);
  return line(cx, cy, cx + x, cy - y, color, true, { 'stroke-width': 4, ...attributes });
}
const rotation = diagram('position-rotation', 'RoPE rotates coordinate pairs',
  'One unit vector is rotated counterclockwise by position times a toy angle of thirty degrees. The real RoPE frequencies vary across coordinate pairs. Apply rotations to queries and keys, leaving values unchanged.', 470);
rotation.add(bullet(32, 'Project the hidden state to Q, K, V.')
  + bullet(83, 'Rotate pairs of coordinates in Q and K.')
  + math(28, 111, 'R(\\phi)=\\begin{bmatrix}\\cos\\phi&-\\sin\\phi\\\\\\sin\\phi&\\cos\\phi\\end{bmatrix}', { width: 615, height: 122, size: 34 })
  + math(28, 250, '\\widetilde q_p=R(p\\theta)q_p,\\quad\\widetilde k_p=R(p\\theta)k_p', { width: 615, height: 70, size: 31 })
  + bullet(354, 'Keep V unchanged.')
  + math(28, 386, '\\theta_i=10000^{-2i/d_h}', { width: 540, height: 66, size: 32 })
  + math(28, 432, 'd_h\\text{: head width; one frequency per pair.}', { width: 615, height: 36, size: 24, fill: c.muted })
  + text(885, 32, 'Toy pair · 30° per token', { anchor: 'middle', weight: 600, size: 28 })
  + axes(885, 232, 154)
  + ray(885, 232, 154, 0, c.blue, { 'stroke-dasharray': '7 5' })
  + ray(885, 232, 154, 2 * toyFrequency, c.purple, { 'data-rotation-ray': '' })
  + path('M 939 232 A 54 54 0 0 0 912 185.235', c.purple, { 'data-rotation-arc': '' })
  + text(885, 417, 'p = 2 · angle = 60°', { anchor: 'middle', 'data-rotation-label': '' })
  + text(885, 461, 'Rotated pair: (0.500, 0.866)', { size: 26, anchor: 'middle', fill: c.purple, 'data-rotation-values': '' }));
await rotation.save();

const relative = diagram('position-relative', 'RoPE encodes the relative offset in a dot product',
  'Two circles compare the same fixed query and key vectors at different absolute positions. Both pairs have the same offset and the same dot product. Controls shift both positions or change the gap.', 480);
const example = relativeExample();
relative.add(text(2, 28, 'Practice E02 · 2 min · Shift both positions. Does the score change?', { size: 26 })
  + text(2, 63, 'Toy: q = k = (1, 0), θ = 30°. Query is purple; key is teal.', { size: 24, fill: c.muted }));
for (const [index, cx] of [278, 866].entries()) {
  relative.add(text(cx, 111, index ? 'Shift both by s = 3' : 'Original positions', { anchor: 'middle', weight: 600, ...(index ? { 'data-shift-label': '' } : {}) })
    + axes(cx, 237, 93)
    + ray(cx, 237, 93, (index ? example.shiftedQuery : example.query) * toyFrequency, c.purple, { 'data-relative-ray': index ? 'shifted-query' : 'query' })
    + ray(cx, 237, 93, (index ? example.shiftedKey : example.key) * toyFrequency, c.teal, { 'data-relative-ray': index ? 'shifted-key' : 'key' })
    + text(cx, 369, index ? 'm = 6, n = 4' : 'm = 3, n = 1', { anchor: 'middle', 'data-relative-positions': index ? 'shifted' : 'original' }));
}
relative.add(text(572, 216, 'Same gap', { anchor: 'middle', size: 28, fill: c.muted })
  + text(572, 257, 'm − n = 2', { anchor: 'middle', size: 28, 'data-gap-label': '' }));
relative.step('For fixed content vectors, a shared shift preserves the dot product.',
  text(278, 409, 'Dot product = 0.500', { anchor: 'middle', fill: c.teal, 'data-relative-score': 'original' })
  + text(866, 409, 'Dot product = 0.500', { anchor: 'middle', fill: c.teal, 'data-relative-score': 'shifted' }));
relative.step('R(mθ)ᵀR(nθ) equals R((n−m)θ): the positions enter through their difference.',
  math(30, 422, '\\widetilde q_m^{\\mathsf T}\\widetilde k_n=q_m^{\\mathsf T}R((n-m)\\theta)k_n', { width: 1092, height: 58, size: 30, anchor: 'middle' }));
await relative.save();

const placement = diagram('position-placement', 'Add at the input or rotate within attention',
  'The original Transformer adds sinusoidal encodings at the embedding input. RoPE instead rotates queries and keys in each attention layer, while values bypass rotation. Both models retain the causal mask for language modeling. Read upward.', 268);
placement.add(box(43, 0, 470, 47, 'Causal attention', c.tealPale)
  + box(81, 79, 394, 47, 'Q, K, V projections')
  + line(278, 79, 278, 49, c.line, true)
  + math(53, 160, 'e_{t_p}+PE(p)', { width: 450, height: 50, anchor: 'middle', fill: c.purple })
  + line(278, 158, 278, 130, c.purple, true)
  + box(676, 0, 394, 47, 'Causal attention', c.tealPale)
  + box(676, 67, 266, 44, 'Rotate Q and K', c.purplePale, 28)
  + line(809, 67, 809, 49, c.purple, true)
  + box(701, 135, 346, 44, 'Q, K, V projections')
  + line(809, 135, 809, 114, c.purple, true)
  + path('M 1047 157 H 1124 V 24 H 1074', c.blue, { 'marker-end': `url(#pos-arrow-${c.blue.slice(1)})`, 'data-value-bypass': '' })
  + text(1098, 114, 'V', { fill: c.blue })
  + math(754, 202, 'e_{t_p}', { width: 240, height: 44, anchor: 'middle' })
  + line(874, 202, 874, 182, c.line, true)
  + text(576, 260, 'Keep the causal mask in both.', { anchor: 'middle', size: 26, fill: c.muted }));
await placement.save();

console.log('Built four editable position diagrams and the sinusoidal frequency plot.');
