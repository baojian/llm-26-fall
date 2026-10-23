// Condense the three-role explanation from the updated Desktop pages 5–7.
// Keep the builder and assets portable when moving this section into Lecture 05.
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import katex from 'katex';

const assets = fileURLToPath(new URL('./assets/', import.meta.url));
const name = 'attention-roles';
const c = { ink: '#142e4b', muted: '#526374', line: '#bccbd7',
  blue: '#436eae', bluePale: '#eef3fa', purple: '#7550b6', purplePale: '#f2edfa',
  teal: '#147d73', tealPale: '#edf7f5' };
const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const el = (tag, attributes, content = '') => `<${tag} ${Object.entries(attributes)
  .map(([key, value]) => `${key}="${esc(value)}"`).join(' ')}>${content}</${tag}>`;
const text = (x, y, label, { size = 30, fill = c.ink, anchor = 'start', weight = 400, ...attributes } = {}) =>
  el('text', { x, y, fill, 'font-size': size, 'text-anchor': anchor, 'font-weight': weight, ...attributes }, esc(label));
const rect = (x, y, width, height, fill, rx = 12) => el('rect', { x, y, width, height, fill, rx });
const tex = (x, y, formula, color) => el('foreignObject', {
  x, y, width: 328, height: 76, 'font-size': 36, color, 'data-math-box': '',
}, el('div', { xmlns: 'http://www.w3.org/1999/xhtml', class: 'attention-math',
  'data-align': 'middle', 'data-tex': formula },
katex.renderToString(formula, { output: 'htmlAndMathml', throwOnError: true, strict: 'error', trust: false })));
const steps = [];
let body = text(2, 360, 'Input') + text(2, 402, 'embeddings');
body += rect(220, 327, 930, 101, c.bluePale);
const tokens = ['Noa', 'can', 'be', 'annoying', 'but', 'she'];
for (const [i, word] of tokens.entries()) {
  const x = 298 + i * 154;
  const fill = i === 5 ? c.purple : c.blue;
  let token = '';
  for (let cell = 0; cell < 4; cell++) token += rect(x - 36 + cell * 19, 348, 15, 20, fill, 3);
  token += text(x, 406, word, { size: 28, anchor: 'middle', fill: i === 5 ? c.purple : c.ink });
  body += el('g', { 'data-token-position': i, 'data-token': word }, token);
}

function step(label, content) {
  const id = `${name}-${steps.length}`;
  steps.push({ click: steps.length + 1, label, object_ids: [id] });
  body += el('g', { class: 'fragment custom', 'data-fragment-index': steps.length - 1,
    'data-animation-object': id }, el('g', { 'data-appear': '', visibility: 'visible' }, content));
}

let roles = el('path', { d: 'M 576 326 V 282', fill: 'none', stroke: c.line, 'stroke-width': 2.5 });
const roleData = [
  ['Query', String.raw`q_5=x_5`, ['Token seeking', 'context: “she”'], c.purple, c.purplePale],
  ['Key', String.raw`k_j=x_j`, ['Compared with', 'the query'], c.blue, c.bluePale],
  ['Value', String.raw`v_j=x_j`, ['Information in', 'the weighted sum'], c.teal, c.tealPale],
];
for (const [i, [label, formula, explanation, color, pale]] of roleData.entries()) {
  const x = 4 + i * 392;
  const center = x + 180;
  roles += el('g', { 'data-attention-role': label.toLowerCase() },
    rect(x, 5, 360, 230, pale)
    + text(center, 48, label, { size: 34, weight: 600, anchor: 'middle', fill: color })
    + tex(x + 16, 67, formula, color)
    + text(center, 175, explanation[0], { anchor: 'middle' })
    + text(center, 215, explanation[1], { anchor: 'middle' })
    + el('path', { d: `M 576 282 H ${center} V 240`, fill: 'none', stroke: color,
      'stroke-width': 2.5, 'stroke-linejoin': 'round', 'marker-end': `url(#${name}-${label.toLowerCase()}-arrow)` }));
}
step('Name the three uses of the same input vectors: query, key, and value.', roles);
step('The simplified operation has no learned Q/K/V projections; learn a representation for each role next.',
  el('g', { 'data-projection-motivation': '' },
    text(2, 477, 'So far: no learned Q/K/V projections.', { size: 32 })
    + text(2, 529, 'Learn a separate representation for each role.', { size: 32, fill: c.teal, weight: 600 })));

const title = 'One input, three roles';
const description = 'The six input embeddings for Noa can be annoying but she are reused without projection. '
  + 'For “she” at zero-based position 5, the query is x_5. Each x_j at positions 0 through 5 is used as both a key for comparison and a value in the weighted sum. '
  + 'The next step introduces learned projections for the three roles. The input embeddings themselves may already be trainable.';
const defs = el('defs', {}, roleData.map(([label, , , color]) => el('marker', {
  id: `${name}-${label.toLowerCase()}-arrow`, viewBox: '0 0 10 10', refX: 8, refY: 5,
  markerWidth: 5, markerHeight: 5, orient: 'auto',
}, el('path', { d: 'M 1 1 L 8 5 L 1 9', fill: 'none', stroke: color, 'stroke-width': 1.8 }))).join(''));
const svg = el('svg', { xmlns: 'http://www.w3.org/2000/svg', width: 1152, height: 552,
  viewBox: '0 0 1152 552', 'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img',
  'aria-labelledby': `${name}-title ${name}-description` },
el('title', { id: `${name}-title` }, title)
  + el('desc', { id: `${name}-description` }, esc(description)) + defs + body);
await writeFile(`${assets}${name}.svg`, svg + '\n');
await writeFile(`${assets}${name}.json`, JSON.stringify({ title,
  source_file: 'lecture-05-slides-transformers.pptx',
  source_copy: 'Desktop copy, 43-slide revision inspected on October 10, 2026 (Asia/Shanghai)',
  source_pages: [5, 6, 7],
  source_titles: ['Self-Attention Whole block', 'Self-Attention Adding parameters'],
  original_attribution: 'https://www.youtube.com/watch?v=tIvKXrEDMhk',
  adaptation: 'Name the three roles in the six-token Noa prefix, then motivate learned projections. Qualify the source claim about having no trainable weights.',
  tokens, query_position: 5,
  steps,
}, null, 2) + '\n');
console.log('Built the attention-roles bridge and its two-step manifest.');
