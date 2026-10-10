// One portable introduction adapted from Desktop PowerPoint pages 4–6.
// Relative asset paths let this builder move with the lecture into lecture-05.
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import katex from 'katex';

const assets = fileURLToPath(new URL('./assets/', import.meta.url));
const name = 'context-intuition';
const c = { ink: '#142e4b', muted: '#526374', line: '#bccbd7',
  future: '#8493a1', purple: '#7550b6', teal: '#147d73' };
const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const el = (tag, attributes, content = '') => `<${tag} ${Object.entries(attributes)
  .map(([key, value]) => `${key}="${esc(value)}"`).join(' ')}>${content}</${tag}>`;
const text = (x, y, label, { size = 30, fill = c.ink, anchor = 'start', weight = 400, ...attributes } = {}) =>
  el('text', { x, y, fill, 'font-size': size, 'text-anchor': anchor, 'font-weight': weight, ...attributes }, esc(label));
const path = (d, color, attributes = {}) => el('path', {
  d, fill: 'none', stroke: color, 'stroke-width': 3, 'stroke-linecap': 'round', ...attributes,
});
const steps = [];
let body = text(2, 37, 'Which earlier word helps interpret “she”?', { size: 32 });
const words = ['Noa', 'can', 'be', 'annoying', 'but', 'she', 'is', 'a', 'great', 'cat'];
const centers = [65, 165, 257, 400, 542, 650, 748, 818, 927, 1078];
for (const [i, word] of words.entries()) {
  body += text(centers[i], 166, word, { size: 38, anchor: 'middle',
    fill: i === 5 ? c.purple : i > 5 ? c.future : c.ink,
    weight: i === 5 ? 600 : 400,
    'data-token-position': i, 'data-token': word,
    'data-token-visibility': i > 5 ? 'future' : 'available',
  });
}
body += path('M 614 182 H 686', c.purple);

function step(label, content) {
  const id = `${name}-${steps.length}`;
  steps.push({ click: steps.length + 1, label, object_ids: [id] });
  body += el('g', { class: 'fragment custom', 'data-fragment-index': steps.length - 1,
    'data-animation-object': id }, el('g', { 'data-appear': '', visibility: 'visible' }, content));
}

step('The pronoun “she” refers back to Noa; relevance can span intervening words.',
  path('M 29 182 H 101', c.teal)
  + path('M 646 194 C 573 299 144 299 70 194', c.teal,
    { 'marker-end': 'url(#context-intuition-arrow)', 'data-referent-link': '' })
  + text(357, 309, '“she” refers to Noa', { size: 32, anchor: 'middle', fill: c.teal })
  + text(2, 547, 'Illustrative linguistic relation', { size: 24, fill: c.muted }));

step('At “she”, causal attention can use the prefix including “she”; later words are masked.',
  path('M 707 118 V 194', c.line, { 'stroke-dasharray': '5 6', 'data-causal-boundary': '' })
  + path('M 724 116 V 97 H 1126 V 116', c.future)
  + text(925, 80, 'Future positions: masked', { size: 27, anchor: 'middle', fill: c.muted })
  + text(753, 255, 'At “she”, GPT uses', { size: 30 })
  + text(753, 297, 'the available prefix.', { size: 30 }));

const formula = String.raw`z_5=\sum_{j=0}^{5}\alpha_{5,j}v_j`;
const html = katex.renderToString(formula, {
  output: 'htmlAndMathml', throwOnError: true, strict: 'error', trust: false,
});
step('Compute a context vector by weighting the available value vectors.',
  text(2, 391, 'Build a context vector for this position', { size: 32, weight: 600 })
  + el('foreignObject', { x: 2, y: 412, width: 510, height: 110,
    'font-size': 36, color: c.ink, 'data-math-box': '' },
  el('div', { xmlns: 'http://www.w3.org/1999/xhtml', class: 'attention-math',
    'data-align': 'start', 'data-tex': formula }, html))
  + text(574, 451, 'Learn weights over the', { size: 30 })
  + text(574, 493, 'available value vectors.', { size: 30 }));

const title = 'Context for a token';
const description = 'In “Noa can be annoying but she is a great cat”, the pronoun she refers to Noa. '
  + 'At she, causal attention can use Noa through she and masks is, a, great, and cat. '
  + 'The link is an illustrative linguistic relation. The model computes a weighted context over available value vectors.';
const defs = el('defs', {}, el('marker', { id: 'context-intuition-arrow', viewBox: '0 0 10 10',
  refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto' },
el('path', { d: 'M 1 1 L 8 5 L 1 9', fill: 'none', stroke: c.teal, 'stroke-width': 1.8 })));
const svg = el('svg', { xmlns: 'http://www.w3.org/2000/svg', width: 1152, height: 552,
  viewBox: '0 0 1152 552', 'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img',
  'aria-labelledby': `${name}-title ${name}-description` },
el('title', { id: `${name}-title` }, title)
  + el('desc', { id: `${name}-description` }, esc(description)) + defs + body);
await writeFile(`${assets}${name}.svg`, svg + '\n');
await writeFile(`${assets}${name}.json`, JSON.stringify({ title,
  source_file: 'lecture-05-slides-transformers.pptx',
  source_copy: 'Desktop copy, 49-slide revision inspected on October 10, 2026 (Asia/Shanghai)',
  source_pages: [4, 5, 6], source_titles: ['Get more context for text'],
  original_attribution: 'https://www.youtube.com/watch?v=tIvKXrEDMhk',
  adaptation: 'One animated language example with zero-based positions, explicit causal visibility, and an illustrative referent link.',
  tokens: words, query_position: 5,
  steps,
}, null, 2) + '\n');
console.log('Built the context-for-text introduction and its three-step manifest.');
