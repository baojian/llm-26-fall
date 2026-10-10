// Adapt the updated Desktop pages 2–3 to a causal, final-token example.
// All paths remain relative when this section moves into Lecture 05.
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const assets = fileURLToPath(new URL('./assets/', import.meta.url));
const name = 'multihead-intuition';
const c = { ink: '#142e4b', muted: '#526374', pale: '#eef3fa',
  purple: '#7550b6', purplePale: '#f2edfa', teal: '#147d73', blue: '#436eae' };
const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const el = (tag, attributes, content = '') => `<${tag} ${Object.entries(attributes)
  .map(([key, value]) => `${key}="${esc(value)}"`).join(' ')}>${content}</${tag}>`;
const text = (x, y, label, { size = 30, fill = c.ink, anchor = 'start', weight = 400, ...attributes } = {}) =>
  el('text', { x, y, fill, 'font-size': size, 'text-anchor': anchor, 'font-weight': weight, ...attributes }, esc(label));
const rect = (x, y, width, height, fill, attributes = {}) =>
  el('rect', { x, y, width, height, fill, rx: 10, ...attributes });
const words = ['Noa', 'can', 'be', 'annoying', 'but', 'she'];
const rows = [
  { question: 'Referent?', color: c.purple, weights: [0.60, 0.05, 0.04, 0.10, 0.06, 0.15] },
  { question: 'Description?', color: c.teal, weights: [0.10, 0.06, 0.08, 0.58, 0.10, 0.08] },
  { question: 'Contrast?', color: c.blue, weights: [0.10, 0.05, 0.05, 0.12, 0.60, 0.08] },
];
for (const row of rows) {
  if (row.weights.length !== words.length || row.weights.some(weight => weight <= 0)
    || Math.abs(row.weights.reduce((sum, weight) => sum + weight, 0) - 1) > 1e-12) {
    throw new Error('Each illustrative head needs a positive, normalized weight for every available position.');
  }
}
const center = i => 352 + 148 * i;
let body = text(2, 35, 'Query: “she” at position 5 · six available positions, 0–5.')
  + rect(center(5) - 62, 74, 124, 54, c.purplePale)
  + text(2, 546, 'Invented weights; learned heads may have overlapping roles.', { fill: c.muted });
for (const [i, word] of words.entries()) {
  body += text(center(i), 112, word, { size: 32, anchor: 'middle',
    fill: i === 5 ? c.purple : c.ink, weight: i === 5 ? 600 : 400,
    'data-token-position': i, 'data-token': word, 'data-token-visibility': 'available' });
}

function rowDiagram(index) {
  const row = rows[index];
  const y = 183 + index * 92;
  let content = text(2, y - 4, `Head ${index + 1}`, { weight: 600, fill: row.color })
    + text(2, y + 33, row.question);
  for (const [j, weight] of row.weights.entries()) {
    content += rect(center(j) - 57, y - 30, 114, 66, c.pale)
      + rect(center(j) - 57, y - 30, 114, 66, row.color, { 'fill-opacity': weight })
      + text(center(j), y + 14, weight.toFixed(2), { size: 32, anchor: 'middle',
        fill: weight > 0.5 ? '#ffffff' : c.ink, weight: weight > 0.5 ? 600 : 400,
        'data-attention-weight': weight, 'data-key-position': j });
  }
  return el('g', { 'data-illustrative-head': index + 1, 'data-query-position': 5 }, content);
}
const steps = [];
function step(label, content) {
  const id = `${name}-${steps.length}`;
  steps.push({ click: steps.length + 1, label, object_ids: [id] });
  body += el('g', { class: 'fragment custom', 'data-fragment-index': steps.length - 1,
    'data-animation-object': id }, el('g', { 'data-appear': '', visibility: 'visible' }, content));
}
step('One head forms one weighting of the available positions for this query. Its row sums to one.', rowDiagram(0));
step('Other heads can form different context mixtures for the same query position, in parallel.',
  rowDiagram(1) + rowDiagram(2)
  + text(2, 477, 'Multiple heads → separate context mixtures.', { size: 34, weight: 600, fill: c.teal }));

const title = 'Why use multiple heads?';
const description = 'An invented causal example at she, position 5 of “Noa can be annoying but she”. '
  + 'Three heads have distinct normalized rows over the same six available positions, emphasizing Noa, annoying, and but. '
  + 'Referent, description, and contrast illustrate possible relations. Real heads learn their behavior and may overlap. '
  + 'Each head uses its own projected query, keys, and values; the example displays only the attention weights.';
const svg = el('svg', { xmlns: 'http://www.w3.org/2000/svg', width: 1152, height: 552,
  viewBox: '0 0 1152 552', 'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img',
  'aria-labelledby': `${name}-title ${name}-description` },
el('title', { id: `${name}-title` }, title)
  + el('desc', { id: `${name}-description` }, esc(description)) + body);
await writeFile(`${assets}${name}.svg`, svg + '\n');
await writeFile(`${assets}${name}.json`, JSON.stringify({ title,
  source_file: 'lecture-05-slides-transformers.pptx',
  source_copy: 'Desktop copy, 26-slide revision inspected on October 10, 2026 (Asia/Shanghai)',
  source_pages: [2, 3], source_titles: ['Is one self-attention enough?', 'Multi-attention'],
  adaptation: 'Use the six-token Noa prefix and she query at position 5 to illustrate three possible context mixtures. All weights are invented.',
  query_position: 5, tokens: words, illustrative_weights: rows.map(row => row.weights),
  steps,
}, null, 2) + '\n');
console.log('Built the multi-head intuition slide; all three illustrative rows sum to one.');
