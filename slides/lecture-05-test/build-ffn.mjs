// Editable FFN mechanism and parameter-share figure, using the existing shell.
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import katex from 'katex';

const assets = fileURLToPath(new URL('./assets/', import.meta.url));
const name = 'ffn-tokenwise';
const c = { ink: '#142e4b', muted: '#526374', line: '#bccbd7', pale: '#eef3fa',
  purple: '#7550b6', purplePale: '#f2edfa', teal: '#147d73', tealPale: '#edf7f5' };
const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const el = (tag, attributes, content = '') => `<${tag} ${Object.entries(attributes)
  .map(([key, value]) => `${key}="${esc(value)}"`).join(' ')}>${content}</${tag}>`;
const text = (x, y, label, { size = 30, fill = c.ink, anchor = 'start', weight = 400 } = {}) =>
  el('text', { x, y, fill, 'font-size': size, 'text-anchor': anchor, 'font-weight': weight }, esc(label));
const rect = (x, y, width, height, fill) => el('rect', { x, y, width, height, fill, rx: 10 });
const tex = (x, y, formula, { width = 430, height = 64, size = 32, color = c.ink, align = 'start' } = {}) =>
  el('foreignObject', { x, y, width, height, 'font-size': size, color, 'data-math-box': '' },
    el('div', { xmlns: 'http://www.w3.org/1999/xhtml', class: 'attention-math',
      'data-align': align, 'data-tex': formula },
    katex.renderToString(formula, { output: 'htmlAndMathml', throwOnError: true, strict: 'error', trust: false })));
const arrow = (x, from, to) => el('path', { d: `M ${x} ${from} V ${to}`, fill: 'none',
  stroke: c.line, 'stroke-width': 2.5, 'marker-end': `url(#${name}-arrow)` });
const bullet = (y, label) => el('circle', { cx: 7, cy: y - 10, r: 4, fill: c.ink }) + text(27, y, label);
const centers = [584, 804, 1024];
let body = tex(2, 0, String.raw`f_t=\sigma(x_tW_1+b_1)W_2+b_2`, { width: 1148, height: 82, size: 38 })
  + text(2, 112, 'Each token separately. Same weights at all positions.')
  + el('path', { d: 'M 461 143 V 502', stroke: c.line, 'stroke-width': 1.5 })
  + text(2, 546, '2017 base model: 512 → 2048 → 512 features', { fill: c.muted });
for (const [i, x] of centers.entries()) {
  body += tex(x - 80, 475, `x_${i + 1}`, { width: 160, height: 45, size: 32, align: 'middle', color: c.purple });
}
const steps = [];
function step(label, content) {
  const id = `${name}-${steps.length}`;
  steps.push({ click: steps.length + 1, label, object_ids: [id] });
  body += el('g', { class: 'fragment custom', 'data-fragment-index': steps.length - 1,
    'data-animation-object': id }, el('g', { 'data-appear': '', visibility: 'visible' }, content));
}
function layer(y, formula, fill, stage) {
  return centers.map((x, i) => el('g', { 'data-ffn-position': i + 1, 'data-ffn-stage': stage },
    rect(x - 92, y, 184, 54, fill)
    + tex(x - 87, y + 2, formula, { width: 174, height: 50, size: 32, align: 'middle' }))).join('');
}
step('Expand the feature width using the same first affine map at every token.',
  bullet(186, 'Expand features')
  + tex(27, 204, String.raw`d_{\rm model}\to d_{\rm ff}`)
  + centers.map(x => arrow(x, 474, 450)).join('')
  + layer(392, 'W_1,b_1', c.purplePale, 'expand'));
step('Apply the activation to each hidden coordinate. The 2017 model uses ReLU; GPT-2 uses GELU.',
  bullet(299, 'Apply a nonlinearity')
  + text(27, 345, 'ReLU (2017)') + text(27, 385, 'GELU (GPT-2)')
  + centers.map(x => arrow(x, 392, 350)).join('')
  + layer(292, String.raw`\sigma`, c.pale, 'activate'));
step('Project back to the model width. The FFN preserves batch and token axes.',
  bullet(432, 'Project back')
  + tex(27, 447, String.raw`d_{\rm ff}\to d_{\rm model}`)
  + centers.map(x => arrow(x, 292, 250)).join('')
  + layer(192, 'W_2,b_2', c.tealPale, 'project')
  + centers.map((x, i) => arrow(x, 192, 179)
    + tex(x - 80, 132, `f_${i + 1}`, { width: 160, height: 45, size: 32, align: 'middle', color: c.teal })).join(''));

const title = 'Feedforward network: transform each token';
const description = 'Three independent token columns use identical first projection, activation, and second projection. '
  + 'The feature width expands and returns to the model width. Input and output retain the same token positions. '
  + 'The original Transformer uses ReLU and widths 512, 2048, 512. GPT-2 uses GELU.';
const defs = el('defs', {}, el('marker', { id: `${name}-arrow`, viewBox: '0 0 10 10', refX: 8, refY: 5,
  markerWidth: 5, markerHeight: 5, orient: 'auto' },
el('path', { d: 'M 1 1 L 8 5 L 1 9', fill: 'none', stroke: c.line, 'stroke-width': 1.8 })));
await writeFile(`${assets}${name}.svg`, el('svg', { xmlns: 'http://www.w3.org/2000/svg', width: 1152, height: 552,
  viewBox: '0 0 1152 552', 'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img',
  'aria-labelledby': `${name}-title ${name}-description` },
el('title', { id: `${name}-title` }, title)
  + el('desc', { id: `${name}-description` }, esc(description)) + defs + body) + '\n');
await writeFile(`${assets}${name}.json`, JSON.stringify({ title,
  source_file: 'lecture-05-slides-transformers.pptx',
  source_copy: 'Desktop copy, 19-slide revision inspected on October 10, 2026 (Asia/Shanghai)',
  source_pages: [6], source_title: 'Transformer Feed forward',
  adaptation: 'Native token columns, affine maps, and activation; distinguish the 2017 ReLU from the GPT-2 GELU. Dropout omitted.',
  steps,
}, null, 2) + '\n');

// The ratio is for one dense GPT-style block's matrices at d_ff = 4 d.
const attentionUnits = 4;
const ffnUnits = 2 * 4;
const boundary = 1152 * attentionUnits / (attentionUnits + ffnUnits);
await writeFile(`${assets}ffn-parameter-share.svg`, el('svg', {
  xmlns: 'http://www.w3.org/2000/svg', width: 1152, height: 56, viewBox: '0 0 1152 56',
  'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img', 'aria-labelledby': 'ffn-share-title',
}, el('title', { id: 'ffn-share-title' }, 'Block matrix parameters: attention one third, FFN two thirds, when FFN width is four times model width.')
  + el('rect', { x: 0, y: 0, width: boundary, height: 56, fill: c.purple })
  + el('rect', { x: boundary, y: 0, width: 1152 - boundary, height: 56, fill: c.teal })
  + text(boundary / 2, 38, 'Attention: 1/3', { fill: '#ffffff', anchor: 'middle' })
  + text((boundary + 1152) / 2, 38, 'FFN: 2/3', { fill: '#ffffff', anchor: 'middle', weight: 600 })) + '\n');
console.log('Built the three-step FFN diagram and the 1:2 parameter-share figure.');
