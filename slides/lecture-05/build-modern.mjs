// Editable SVG text and geometry. No raster assets or new runtime libraries.
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import katex from 'katex';

const directory = path.dirname(fileURLToPath(import.meta.url));
const assets = path.join(directory, 'assets');
const source = JSON.parse(await readFile(path.join(assets, 'animation.json'), 'utf8'));
const color = {
  ink: '#142e4b', muted: '#526374', line: '#bccbd7', pale: '#e5edf3',
  blue: '#527596', purple: '#7550b6', purplePale: '#eee8f7',
  teal: '#147d73', tealPale: '#e5f3ef', white: '#ffffff',
};
const tokens = ['Noa', 'can', 'be', 'annoying', 'but', 'she'];
const centers = [370, 514, 658, 802, 946, 1090];
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const element = (tag, attributes, content = '') => `<${tag} ${Object.entries(attributes).map(([key, value]) => `${key}="${escape(value)}"`).join(' ')}>${content}</${tag}>`;
const text = (x, y, content, { size = 28, fill = color.ink, anchor = 'start', weight = 400, ...attributes } = {}) => element('text', {
  x: x === 0 ? 2 : x, y, 'font-size': size, fill, 'text-anchor': anchor, 'font-weight': weight, ...attributes,
}, escape(content));
const rect = (x, y, width, height, fill, radius = 0) => element('rect', { x, y, width, height, fill, rx: radius });
const line = (x1, y1, x2, y2, { stroke = color.line, arrow = false, width = 1.8, prefix = 'arrow' } = {}) => element('line', {
  x1, y1, x2, y2, stroke, 'stroke-width': width,
  ...(arrow ? { 'marker-end': `url(#${prefix}-${stroke === color.teal ? 'teal' : 'neutral'})` } : {}),
});

function item(y, label) {
  return element('g', { 'data-explanation-item': label, role: 'listitem' },
    element('circle', { cx: 9, cy: y - 9, r: 3.6, fill: color.ink }) + text(28, y, label));
}

// The same pinned KaTeX package and local fonts used by the shared course shell.
// XHTML inside foreignObject preserves selectable text, MathML, and TeX source.
function tex(x, y, latex, { size = 30, fill = color.ink, anchor = 'start', width } = {}) {
  width ??= x < 328 ? 298 : anchor === 'middle' ? 150 : 110;
  const left = anchor === 'middle' ? x - width / 2 : anchor === 'end' ? x - width : x;
  width = Math.min(width, 1152 - left);
  const html = katex.renderToString(latex, { output: 'htmlAndMathml', throwOnError: true, strict: 'error', trust: false });
  return element('foreignObject', {
    x: left, y: y - size * 1.15, width, height: size * 1.8,
    'font-size': size, color: fill, 'data-math-box': '',
  }, element('div', {
    xmlns: 'http://www.w3.org/1999/xhtml', class: 'attention-math',
    'data-align': anchor, 'data-tex': latex,
  }, html));
}

function math(x, y, runs, options = {}) {
  const latex = runs.map(run => {
    if (typeof run === 'string') return run;
    let value = ['x', 'z'].includes(run.base) ? `\\mathbf{${run.base}}` : run.base === '∑' ? '\\sum' : run.base;
    if (run.sub) value += `_{${run.sub}}`;
    if (run.sup) value += `^{${run.sup === 'T' ? '\\mathsf T' : run.sup}}`;
    return run.fill ? `\\textcolor{${run.fill}}{${value}}` : value;
  }).join('');
  return tex(x, y, latex, options);
}
const variable = (base, sub, fill, sup) => ({ base, sub, fill, sup });
function vector(x, y, { fill = color.blue, width = 76, height = 22 } = {}) {
  const gap = 3;
  const cell = (width - 3 * gap) / 4;
  return Array.from({ length: 4 }, (_, i) => rect(x + i * (cell + gap), y, cell, height, fill, 3)).join('');
}

const groups = [];
function group(id, label, content) {
  const index = groups.length;
  groups.push({ id, label, content: element('g', { class: 'fragment custom', 'data-fragment-index': index, 'data-animation-object': id },
    element('g', { 'data-appear': '', visibility: 'visible' }, content)) });
}

let content = text(0, 38, 'Query: “she”', { fill: color.purple, weight: 600 });
content += item(84, 'Input embeddings');
content += text(28, 120, 'Available positions 0–5', { size: 24, fill: color.muted });
for (const [i, word] of tokens.entries()) {
  const x = centers[i];
  const query = i === 5;
  let token = query ? rect(x - 51, 8, 102, 43, color.purplePale, 12) : '';
  token += text(x, 38, word, { size: 28, anchor: 'middle', weight: query ? 600 : 400, fill: query ? color.purple : color.ink });
  token += vector(x - 37, 68, { width: 74, fill: query ? color.purple : color.blue });
  token += math(x, 123, [variable('x', String(i))], { fill: query ? color.purple : color.ink, anchor: 'middle', width: 110 });
  content += element('g', { 'data-token-position': i, 'data-token': word, 'data-token-visibility': 'available' }, token);
}
group('inputs', 'The query is she at position 5; positions 0–5 form the available causal prefix.', content);

content = item(180, 'Compare with the query');
content += tex(28, 213, String.raw`s_{5j}=\mathbf x_5\mathbf x_j^{\mathsf T}`, { size: 27, width: 285 });
for (const [i, x] of centers.entries()) {
  content += line(x, 139, x, 160, { arrow: true });
  content += tex(x, 196, `s_{5,${i}}`, { anchor: 'middle', width: 116 });
}
group('scores', 'Compare the she embedding with each of the six available embeddings.', content);

content = item(250, 'Normalize');
content += rect(326, 221, 816, 42, color.tealPale, 10);
content += text(734, 250, 'softmax across the six available scores', { size: 26, anchor: 'middle', fill: color.teal });
group('normalization', 'Normalize the six available scores with softmax.', content);

content = item(295, 'Attention weights');
content += tex(28, 326, String.raw`\sum_{j=0}^{5} w_{5j}=1`, { size: 26, fill: color.teal, width: 285 });
for (const [i, x] of centers.entries()) {
  content += line(x, 263, x, 277, { arrow: true, stroke: color.teal });
  content += tex(x, 309, `w_{5,${i}}`, { anchor: 'middle', width: 116, fill: color.teal });
}
group('weights', 'The query has one normalized weight for every prefix token.', content);

content = item(371, 'Reuse the embeddings');
for (const [i, x] of centers.entries()) {
  content += vector(x - 42, 352, { width: 60, fill: i === 5 ? color.purple : color.blue });
  content += math(x + 24, 372, [variable('x', String(i))], { size: 27, width: 43, fill: i === 5 ? color.purple : color.ink });
}
group('values', 'For this first calculation, use the input embeddings themselves as values.', content);

content = item(430, 'Weight each input');
for (const [i, x] of centers.entries()) {
  content += line(x, 319, x, 344, { arrow: true, stroke: color.teal });
  content += line(x, 382, x, 401, { arrow: true });
  content += tex(x, 431, `\\textcolor{${color.teal}}{w_{5,${i}}}\\mathbf x_${i}`, { anchor: 'middle', width: 138, size: 27 });
}
group('products', 'Multiply each available value by its weight for she.', content);

content = item(499, 'Weighted sum');
content += tex(28, 538, String.raw`\mathbf z_5=\sum_{j=0}^{5}w_{5j}\mathbf x_j`, { size: 27, width: 288 });
for (const x of centers) content += line(x, 442, x, 462);
content += line(centers[0], 462, centers.at(-1), 462);
content += rect(718, 448, 36, 28, color.white, 8);
content += text(736, 470, '+', { size: 30, anchor: 'middle', fill: color.teal });
content += element('path', { d: 'M 736 478 V 485 H 1090 V 500', fill: 'none', stroke: color.teal,
  'stroke-width': 1.8, 'marker-end': 'url(#arrow-teal)' });
content += vector(1052, 508, { fill: color.teal, width: 76, height: 24 });
content += tex(1090, 558, String.raw`\mathbf z_5`, { fill: color.purple, anchor: 'middle', size: 28, width: 110 });
group('context', 'The weighted sum is the contextual output z_5 for she.', content);

content = text(365, 495, 'Other queries: causal prefixes', { size: 24, fill: color.muted });
for (let i = 0; i < 5; i++) {
  content += element('g', { 'data-output-position': i },
    vector(centers[i] - 38, 508, { fill: color.teal, width: 76, height: 24 })
    + tex(centers[i], 558, `\\mathbf z_${i}`, { fill: color.teal, anchor: 'middle', size: 28, width: 110 }));
}
group('other-outputs', 'Repeat for positions 0–4; each query can use only its own available prefix.', content);

let definitions = '';
for (const [name, stroke] of [['neutral', color.line], ['teal', color.teal]]) {
  definitions += element('marker', { id: `arrow-${name}`, viewBox: '0 0 8 8', refX: 6, refY: 4, markerWidth: 5, markerHeight: 5, orient: 'auto', markerUnits: 'strokeWidth' },
    element('path', { d: 'M 1 1 L 6 4 L 1 7', fill: 'none', stroke, 'stroke-width': 1.4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
}
const svg = element('svg', {
  xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 1152 580', width: 1152, height: 552,
  'font-family': 'Arial, Helvetica Neue, sans-serif',
  role: 'img', 'aria-labelledby': 'modern-attention-title modern-attention-description',
}, element('title', { id: 'modern-attention-title' }, 'Self-attention for she in the six-token Noa prefix')
  + element('desc', { id: 'modern-attention-description' }, 'Noa, can, be, annoying, but, she occupy positions 0 through 5. Compare the she embedding at position 5 with the six available embeddings, normalize with softmax, and form the weighted sum. Later words are excluded. Other queries use their own causal prefixes. Feature cells are schematic.')
  + element('defs', {}, definitions) + groups.map(item => item.content).join('\n'));
await writeFile(path.join(assets, 'self-attention-modern.svg'), svg + '\n');
await writeFile(path.join(assets, 'animation-modern.json'), JSON.stringify({
  source: source.source, source_slide: source.slide, effect: 'Appear',
  adaptation: 'The source teaching sequence is retained with the six-token Noa prefix, zero-based positions, and a causal final-position query.',
  tokens, query_position: 5,
  steps: groups.map((item, index) => ({ click: index + 1, label: item.label, object_ids: [item.id] })),
}, null, 2) + '\n');

// Learned projections: six token rows; symbolic feature widths are explicit.
const source29 = JSON.parse(await readFile(path.join(assets, 'animation-29.json'), 'utf8'));
const groups29 = [];
function group29(id, label, content) {
  groups29.push({ id, label, content: element('g', {
    class: 'fragment custom', 'data-fragment-index': groups29.length, 'data-animation-object': id,
  }, element('g', { 'data-appear': '', visibility: 'visible' }, content)) });
}
function matrix(x, y, fill, { name, columns, width = 78, height = 80 } = {}) {
  const gap = 3;
  const rowHeight = (height - 5 * gap) / 6;
  return element('g', { 'data-matrix': name, 'data-rows': 6, 'data-columns': columns, 'data-feature-width-schematic': 'true' },
    tokens.map((word, row) => element('g', { 'data-matrix-row': row, 'data-token-position': row, 'data-token': word },
      vector(x, y + row * (rowHeight + gap), { fill, width, height: rowHeight }))).join(''));
}
const route29 = (d, { stroke = color.line, arrow = true } = {}) => element('path', {
  d, fill: 'none', stroke, 'stroke-width': 1.8, 'stroke-linejoin': 'round',
  ...(arrow ? { 'marker-end': `url(#qkv-arrow-${stroke === color.teal ? 'teal' : 'neutral'})` } : {}),
});
const down29 = (x, y1, y2) => line(x, y1, x, y2, { arrow: true, prefix: 'qkv-arrow' });
let base29 = text(0, 35, 'One attention head', { fill: color.purple, weight: 600 });
base29 += text(750, 31, 'Rows ↓: Noa, can, be, annoying, but, she · schematic features →', { size: 24, anchor: 'middle', fill: color.muted });
base29 += tex(373, 177, String.raw`\mathbf X`, { anchor: 'middle', width: 100, size: 30 });
base29 += matrix(334, 197, color.blue, { name: 'X', columns: 'd_model', width: 78, height: 96 });
base29 += tex(373, 331, String.raw`6\times d_{\rm model}`, { anchor: 'middle', width: 185, size: 25 });
base29 += text(373, 367, 'Token rows ↓', { anchor: 'middle', size: 24, fill: color.muted });
base29 += rect(836, 126, 294, 50, color.pale, 12);
base29 += text(983, 160, 'Scaled scores', { anchor: 'middle', size: 27 });
base29 += element('g', { 'data-tensor': 'S', 'data-rows': 6, 'data-columns': 6 },
  tex(983, 212, String.raw`QK^{\mathsf T}/\sqrt{d_k}:6\times6`, { anchor: 'middle', width: 294, size: 24 }));
base29 += down29(983, 221, 235);
base29 += rect(836, 239, 294, 48, color.tealPale, 12);
base29 += text(983, 270, 'Mask, then softmax', { anchor: 'middle', size: 26, fill: color.teal });
base29 += element('g', { 'data-tensor': 'A', 'data-rows': 6, 'data-columns': 6 },
  tex(983, 321, String.raw`A:6\times6`, { anchor: 'middle', width: 220, size: 25, fill: color.teal }));
base29 += down29(983, 327, 341);
base29 += rect(836, 346, 294, 48, color.tealPale, 12);
base29 += text(983, 378, 'Weighted sum', { anchor: 'middle', size: 27, fill: color.teal });
base29 += down29(983, 394, 416);
base29 += matrix(947, 425, color.teal, { name: 'Z', columns: 'd_v', width: 72, height: 60 });
base29 += tex(885, 465, String.raw`Z`, { size: 30, anchor: 'middle', width: 74, fill: color.teal });
base29 += tex(1074, 465, String.raw`6\times d_v`, { size: 24, anchor: 'middle', width: 130, fill: color.teal });

for (const [index, [letter, label, hint, fill, pale, dim]] of [
  ['Q', 'Query projection', 'What to look for', color.purple, color.purplePale, 'd_k'],
  ['K', 'Key projection', 'What to match', color.blue, color.pale, 'd_k'],
  ['V', 'Value projection', 'What to pass on', color.teal, color.tealPale, 'd_v'],
].entries()) {
  const y = 110 + index * 130;
  let part = item(y - 17, label);
  part += text(28, y + 18, hint, { size: 24, fill: color.muted });
  part += route29(`M 412 245 H 461 V ${y} H 493`);
  part += element('g', { 'data-matrix': `W_${letter}`, 'data-rows': 'd_model', 'data-columns': dim },
    rect(495, y - 28, 112, 56, pale, 12)
    + tex(551, y + 12, `\\mathbf W_${letter}`, { size: 30, fill, anchor: 'middle', width: 105 }));
  part += tex(551, y + 69, `d_{\\rm model}\\times ${dim}`, { size: 24, anchor: 'middle', width: 196, fill });
  part += line(608, y, 676, y, { arrow: true, prefix: 'qkv-arrow' });
  part += matrix(678, y - 35, fill, { name: letter, columns: dim, width: 90, height: 70 });
  part += tex(723, y - 49, `\\mathbf ${letter}:6\\times ${dim}`, { size: 25, fill, anchor: 'middle', width: 210 });
  const route = index === 0 ? 'M 769 110 H 807 V 144 H 834'
    : index === 1 ? 'M 769 240 H 807 V 163 H 834' : 'M 769 370 H 834';
  part += route29(route, { stroke: index === 2 ? color.teal : color.line });
  group29(['query-projection', 'key-projection', 'value-projection'][index], `${letter} uses a learned d_model by ${dim} projection and has six token rows.`, part);
}
content = item(468, 'Trainable projections');
content += text(28, 500, 'Shared across token positions', { size: 24, fill: color.muted });
content += rect(489, 450, 325, 45, color.purplePale, 12);
content += tex(651, 480, String.raw`\mathbf W_Q,\;\mathbf W_K,\;\mathbf W_V`, { anchor: 'middle', width: 309, size: 28, fill: color.purple });
group29('trainable-projections', 'Projection matrices are shared across all six token positions, while Q, K, and V have separate parameters.', content);
content = item(552, 'Matrix form');
content += line(326, 516, 1130, 516, { stroke: color.pale, width: 1.5 });
for (const [index, letter] of ['Q', 'K', 'V'].entries()) {
  content += tex(455 + index * 269, 552, `\\mathbf ${letter}=\\mathbf X\\mathbf W_${letter}`, {
    size: 28, anchor: 'middle', width: 252, fill: [color.purple, color.blue, color.teal][index],
  });
}
group29('matrix-form', 'Rows index the same six token positions; columns are learned features. Q and K share width d_k, while V can use width d_v.', content);
const svg29 = element('svg', {
  xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 1152 580', width: 1152, height: 552,
  'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img',
  'aria-labelledby': 'qkv-attention-title qkv-attention-description',
}, element('title', { id: 'qkv-attention-title' }, 'Six-token query, key, and value projections with dimensions')
  + element('desc', { id: 'qkv-attention-description' }, 'The Noa prefix has six token rows. X has shape 6 by d_model. W_Q and W_K have shape d_model by d_k; W_V has shape d_model by d_v. Q and K have shape 6 by d_k and V has shape 6 by d_v. Scores and masked attention weights have shape 6 by 6; Z has shape 6 by d_v. Matrix drawings show exactly six rows, with schematic feature widths.')
  + element('defs', {}, definitions.replaceAll('id="arrow-', 'id="qkv-arrow-'))
  + base29 + groups29.map(item => item.content).join('\n'));
await writeFile(path.join(assets, 'self-attention-29-modern.svg'), svg29 + '\n');
await writeFile(path.join(assets, 'animation-29-modern.json'), JSON.stringify({
  source: source29.source, source_slide: source29.slide, effect: 'Appear',
  adaptation: 'Six token rows from the Noa prefix; explicit one-head matrix dimensions; causal mask in the score path.',
  tokens, dimensions: { X: [6, 'd_model'], W_Q: ['d_model', 'd_k'], W_K: ['d_model', 'd_k'], W_V: ['d_model', 'd_v'], Q: [6, 'd_k'], K: [6, 'd_k'], V: [6, 'd_v'], S: [6, 6], A: [6, 6], Z: [6, 'd_v'] },
  steps: groups29.map((item, index) => ({ click: index + 1, label: item.label, object_ids: [item.id] })),
}, null, 2) + '\n');

// Page 40 adaptation: a GPT-2-style causal decoder with pre-normalization.
// The expanded block and language-model head follow openai/gpt-2/src/model.py.
const referenceURL = 'https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.svg';
const groups40 = [];
function group40(id, label, content) {
  groups40.push({ id, label, content: element('g', {
    class: 'fragment custom', 'data-fragment-index': groups40.length, 'data-animation-object': id,
  }, element('g', { 'data-appear': '', visibility: 'visible' }, content)) });
}
const up40 = (y1, y2) => line(735, y1, 735, y2, { arrow: true, prefix: 'gpt-arrow' });
function box40(y, height, label, { width = 400, fill = color.pale, ink = color.ink, size = 28 } = {}) {
  return rect(735 - width / 2, y, width, height, fill, 10)
    + text(735, y + height / 2 + size * 0.34, label, { size, anchor: 'middle', fill: ink });
}
function residual40(startY, endY) {
  return element('path', {
    d: `M 735 ${startY} H 961 V ${endY} H 749`, fill: 'none', stroke: color.blue,
    'stroke-width': 1.8, 'marker-end': 'url(#gpt-arrow-neutral)', 'data-residual-path': '',
  }) + element('circle', { cx: 735, cy: startY, r: 3, fill: color.blue })
    + element('circle', { cx: 735, cy: endY, r: 13, fill: color.white, stroke: color.blue, 'stroke-width': 1.8 })
    + text(735, endY + 8, '+', { anchor: 'middle', size: 28, fill: color.blue });
}
let base40 = text(661, 556, 'Noa · can · be · … · she', { anchor: 'middle', size: 24, fill: color.purple });
base40 += text(960, 556, '0 · 1 · 2 · 3 · 4 · 5', { anchor: 'middle', size: 24, fill: color.blue });

content = item(498, 'Embed the prefix');
content += tex(28, 537, '\\mathbf h_i^{(0)}=\\mathbf E(t_i)+\\mathbf P_i', { size: 28, width: 394 });
content += element('g', { 'data-input-component': 'token' },
  rect(519, 487, 284, 36, color.purplePale, 10)
  + text(661, 513, 'Token embedding', { size: 26, anchor: 'middle', fill: color.purple }));
content += element('g', { 'data-input-component': 'position' },
  rect(824, 487, 272, 36, color.pale, 10)
  + text(960, 513, 'Positional embedding', { size: 26, anchor: 'middle', fill: color.blue }));
for (const x of [661, 960]) content += line(x, 532, x, 524, { arrow: true, prefix: 'gpt-arrow' });
for (const [x, endX] of [[661, 721], [960, 749]]) {
  content += element('path', {
    d: `M ${x} 486 V 462 H ${endX}`, fill: 'none', stroke: color.blue,
    'stroke-width': 1.8, 'marker-end': 'url(#gpt-arrow-neutral)',
  });
}
content += element('g', { 'data-embedding-sum': '' },
  element('circle', { cx: 735, cy: 462, r: 12, fill: color.white, stroke: color.blue, 'stroke-width': 1.8 })
  + text(735, 470, '+', { anchor: 'middle', size: 28, fill: color.blue }));
group40('gpt-embeddings', 'Embed each input token and add its position embedding', content);

content = item(355, 'Causal self-attention');
content += tex(28, 394, 'a_{ij}=0\\quad\\text{for }j>i', { size: 28, width: 394 });
content += up40(450, 433);
content += box40(400, 32, 'LayerNorm', { width: 180, size: 25, fill: '#f0f3f6' });
content += up40(400, 385);
content += box40(344, 40, 'Causal multi-head attention', { size: 27, fill: color.tealPale, ink: color.teal });
content += line(735, 344, 735, 337);
content += residual40(442, 324);
group40('gpt-attention', 'Normalize, apply causal multi-head attention, then add the residual', content);

content = item(216, 'Process each token');
content += text(28, 253, 'One MLP, applied per position', { size: 24, fill: color.muted });
content += up40(311, 293);
content += box40(260, 32, 'LayerNorm', { width: 180, size: 25, fill: '#f0f3f6' });
content += up40(260, 245);
content += box40(204, 40, 'MLP', { fill: color.purplePale, ink: color.purple });
content += line(735, 204, 735, 197);
content += residual40(302, 184);
group40('gpt-mlp', 'Normalize, apply the position-wise MLP, then add the residual', content);

content = item(135, 'Repeat L blocks');
content += text(28, 172, 'New parameters in each block', { size: 24, fill: color.muted });
content += element('rect', {
  x: 486, y: 151, width: 511, height: 298, rx: 15, fill: 'none', stroke: color.blue,
  'stroke-width': 1.7, 'stroke-dasharray': '7 5', 'data-transformer-block': '',
});
content += tex(1054, 188, '\\times L', { anchor: 'middle', width: 115, size: 32, fill: color.blue });
group40('gpt-stack', 'Repeat the complete block L times, with separate parameters at each layer', content);

content = item(44, 'Predict the next token');
content += tex(28, 83, 'p(t_6\\mid t_0,\\ldots,t_5)', { width: 394, size: 29, fill: color.teal });
content += up40(171, 130);
content += box40(97, 32, 'Final LayerNorm', { width: 270, size: 25, fill: '#f0f3f6' });
content += up40(97, 69);
content += box40(26, 42, 'LM head + softmax', { width: 420, size: 28, fill: color.tealPale, ink: color.teal });
content += line(946, 47, 985, 47, { stroke: color.teal, arrow: true, prefix: 'gpt-arrow' });
content += text(1048, 98, 'Choose', { size: 24, anchor: 'middle', fill: color.muted });
content += rect(988, 26, 120, 42, color.tealPale, 10);
content += text(1048, 55, 'is', { anchor: 'middle', size: 28, fill: color.teal, weight: 600 });
content += element('path', {
  d: 'M 1109 47 H 1134 V 566 H 495 V 543 H 516', fill: 'none', stroke: color.teal,
  'stroke-width': 1.8, 'stroke-dasharray': '6 5', 'marker-end': 'url(#gpt-arrow-teal)',
  'data-generation-loop': '',
});
content += text(1061, 275, 'Append', { size: 24, anchor: 'middle', fill: color.teal });
content += text(1061, 307, 'and repeat', { size: 24, anchor: 'middle', fill: color.teal });
group40('gpt-prediction', 'At she in position 5, predict is at position 6; append it and repeat during generation', content);

const svg40 = element('svg', {
  xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 1152 568', width: 1152, height: 552,
  'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img',
  'aria-labelledby': 'gpt-architecture-title gpt-architecture-description',
}, element('title', { id: 'gpt-architecture-title' }, 'GPT-style causal Transformer')
  + element('desc', { id: 'gpt-architecture-description' }, 'Read upward from the bottom. Separate token and positional embeddings are added before entering L Transformer blocks. Each block uses layer normalization before causal multi-head attention and before an MLP, with a residual addition after each. Final layer normalization and the language-model head at the top predict the next token. Generation appends that token to the prefix at the bottom and repeats.')
  + element('defs', {}, definitions.replaceAll('id="arrow-', 'id="gpt-arrow-'))
  + base40 + groups40.map(item => item.content).join('\n'));
await writeFile(path.join(assets, 'self-attention-40-modern.svg'), svg40 + '\n');
await writeFile(path.join(assets, 'animation-40-modern.json'), JSON.stringify({
  source_slide: 40, source: referenceURL, effect: 'Appear',
  adaptation: 'Five-stage GPT-2-style diagram with the Noa prefix at positions 0–5 and the illustrative next token is at position 6.',
  tokens, next_token: 'is', next_position: 6,
  steps: groups40.map((item, index) => ({ click: index + 1, label: item.label, object_ids: [item.id] })),
}, null, 2) + '\n');

// Keep the linked reference available through the existing comparison control.
// Preserve the downloaded artwork and put it in a separately titled wrapper.
const originalGPT = (await readFile(path.join(assets, 'gpt-architecture-reference.svg'), 'utf8'))
  .replace(/<\?xml[^>]*\?>\s*/, '');
const referenceSVG = element('svg', {
  xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 1152 568', width: 1152, height: 552,
  role: 'img', 'aria-labelledby': 'gpt-reference-title', 'font-family': 'Arial, Helvetica Neue, sans-serif',
}, element('title', { id: 'gpt-reference-title' }, 'Full GPT architecture reference by Marxav and Mrmw')
  + text(28, 72, 'Full GPT reference', { size: 30, weight: 600 })
  + text(28, 117, 'Marxav / Mrmw · CC0', { size: 24, fill: color.muted })
  + text(28, 198, 'Open Modern to present', { size: 24 })
  + text(28, 232, 'the simplified architecture.', { size: 24 })
  + element('g', { transform: 'translate(445 2) scale(0.94)' }, originalGPT));
await writeFile(path.join(assets, 'self-attention-40.svg'), referenceSVG + '\n');
await writeFile(path.join(assets, 'animation-40.json'), JSON.stringify({
  source_slide: 40, source: referenceURL, effect: 'Static reference', steps: [],
  author: 'Marxav (original), Mrmw (vectorization)', license: 'CC0 1.0',
}, null, 2) + '\n');
console.log('Built three modern SVGs: 8 + 5 + 5 stages, bullet explanations, and KaTeX equations.');
