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
const centers = [388, 548, 708, 868];
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const element = (tag, attributes, content = '') => `<${tag} ${Object.entries(attributes).map(([key, value]) => `${key}="${escape(value)}"`).join(' ')}>${content}</${tag}>`;
const text = (x, y, content, { size = 28, fill = color.ink, anchor = 'start', weight = 400 } = {}) => element('text', {
  x: x === 0 ? 2 : x, y, 'font-size': size, fill, 'text-anchor': anchor, 'font-weight': weight,
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
function group(id, content) {
  const index = groups.length;
  groups.push({ id, content: element('g', { class: 'fragment custom', 'data-fragment-index': index, 'data-animation-object': id },
    element('g', { 'data-appear': '', visibility: 'visible' }, content)) });
}

let content = text(0, 38, 'Query: “the”', { fill: color.purple, weight: 600 });
content += item(84, 'Input embeddings');
content += text(28, 120, 'Attends to all four words', { size: 24, fill: color.muted });
for (const [i, word] of ['Bank', 'of', 'the', 'river'].entries()) {
  const x = centers[i];
  const query = i === 2;
  if (query) content += rect(x - 58, 8, 116, 43, color.purplePale, 12);
  content += text(x, 38, word, { size: 30, anchor: 'middle', weight: query ? 600 : 400, fill: query ? color.purple : color.ink });
  content += vector(x - 38, 68, { fill: query ? color.purple : color.blue });
  content += math(x, 124, [variable('x', String(i + 1))], { fill: query ? color.purple : color.ink, anchor: 'middle' });
}
group('inputs', content);

content = item(180, 'Compare with the query');
content += math(28, 213, [variable('s', '3j'), ' = ', variable('x', '3', color.purple, 'T'), ' ', variable('x', 'j')], { size: 28 });
for (const [i, x] of centers.entries()) {
  content += line(x, 139, x, 160, { arrow: true });
  content += math(x, 196, [variable('s', `3${i + 1}`)], { anchor: 'middle' });
}
group('scores', content);

content = item(250, 'Normalize');
content += rect(328, 221, 600, 42, color.tealPale, 10);
content += text(628, 250, 'softmax across the four scores', { size: 26, anchor: 'middle', fill: color.teal });
group('normalization', content);

content = item(295, 'Attention weights');
content += math(28, 326, [variable('∑', 'j'), ' ', variable('w', '3j'), ' = 1'], { size: 26, fill: color.teal });
for (const [i, x] of centers.entries()) {
  content += line(x, 263, x, 277, { arrow: true, stroke: color.teal });
  content += math(x, 309, [variable('w', `3${i + 1}`)], { anchor: 'middle', fill: color.teal });
}
group('weights', content);

content = item(371, 'Reuse the embeddings');
for (const [i, x] of centers.entries()) {
  content += vector(x - 46, 352, { width: 64, fill: i === 2 ? color.purple : color.blue });
  content += math(x + 29, 372, [variable('x', String(i + 1))], { size: 27, fill: i === 2 ? color.purple : color.ink });
}
group('values', content);

content = item(430, 'Weight each input');
for (const [i, x] of centers.entries()) {
  content += line(x, 319, x, 344, { arrow: true, stroke: color.teal });
  content += line(x, 382, x, 401, { arrow: true });
  content += math(x, 431, [variable('w', `3${i + 1}`, color.teal), ' ', variable('x', String(i + 1), i === 2 ? color.purple : color.ink)], { anchor: 'middle', size: 29 });
}
group('products', content);

content = item(499, 'Weighted sum');
content += math(28, 532, [variable('z', '3', color.purple), ' = ', variable('∑', 'j'), ' ', variable('w', '3j', color.teal), ' ', variable('x', 'j')], { size: 28 });
for (const x of centers) content += line(x, 442, x, 462);
content += line(centers[0], 462, centers[3], 462);
content += rect(610, 448, 36, 28, color.white, 8);
content += text(628, 470, '+', { size: 30, anchor: 'middle', fill: color.teal });
content += line(628, 478, 628, 496, { arrow: true, stroke: color.teal });
content += vector(583, 507, { fill: color.teal, width: 90, height: 27 });
content += math(549, 529, [variable('z', '3')], { fill: color.purple, anchor: 'end', size: 34 });
content += text(732, 511, 'Context for “the”', { size: 27, fill: color.purple, weight: 600 });
content += text(732, 541, 'from all four words', { size: 24, fill: color.muted });
group('context', content);

content = line(977, 8, 977, 473, { stroke: color.pale, width: 1.5 });
content += text(1006, 171, 'Repeat for', { size: 24, fill: color.muted });
content += text(1006, 201, 'each word', { size: 24, fill: color.muted });
for (const [i, index] of [1, 2, 4].entries()) {
  const y = 258 + i * 74;
  content += math(1005, y + 20, [variable('z', String(index))], { size: 28, fill: color.teal });
  content += vector(1053, y, { fill: color.teal, width: 78, height: 24 });
}
group('other-outputs', content);

let definitions = '';
for (const [name, stroke] of [['neutral', color.line], ['teal', color.teal]]) {
  definitions += element('marker', { id: `arrow-${name}`, viewBox: '0 0 8 8', refX: 6, refY: 4, markerWidth: 5, markerHeight: 5, orient: 'auto', markerUnits: 'strokeWidth' },
    element('path', { d: 'M 1 1 L 6 4 L 1 7', fill: 'none', stroke, 'stroke-width': 1.4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
}
const svg = element('svg', {
  xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 1152 552', width: 1152, height: 552,
  'font-family': 'Arial, Helvetica Neue, sans-serif',
  role: 'img', 'aria-labelledby': 'modern-attention-title modern-attention-description',
}, element('title', { id: 'modern-attention-title' }, 'Self-attention for the word “the”')
  + element('desc', { id: 'modern-attention-description' }, 'Compare the third embedding with all four embeddings, normalize the scores with softmax, and sum the weighted embeddings. Eight stages use vector shapes and live text.')
  + element('defs', {}, definitions) + groups.map(item => item.content).join('\n'));
await writeFile(path.join(assets, 'self-attention-modern.svg'), svg + '\n');
await writeFile(path.join(assets, 'animation-modern.json'), JSON.stringify({
  source: source.source, source_slide: source.slide, effect: 'Appear',
  steps: groups.map((item, index) => ({ click: index + 1, label: source.steps[index].label, object_ids: [item.id] })),
}, null, 2) + '\n');

// Source page 29: learned projections, in the original five-click order.
const source29 = JSON.parse(await readFile(path.join(assets, 'animation-29.json'), 'utf8'));
const groups29 = [];
function group29(id, content) {
  groups29.push({ id, content: element('g', {
    class: 'fragment custom', 'data-fragment-index': groups29.length, 'data-animation-object': id,
  }, element('g', { 'data-appear': '', visibility: 'visible' }, content)) });
}
function matrix(x, y, fill, { width = 76, height = 68 } = {}) {
  return Array.from({ length: 4 }, (_, row) => vector(x, y + row * (height + 3) / 4, {
    fill, width, height: (height - 9) / 4,
  })).join('');
}
const route29 = (d, { stroke = color.line, arrow = true } = {}) => element('path', {
  d, fill: 'none', stroke, 'stroke-width': 1.8, 'stroke-linejoin': 'round',
  ...(arrow ? { 'marker-end': `url(#qkv-arrow-${stroke === color.teal ? 'teal' : 'neutral'})` } : {}),
});
const down29 = (x, y1, y2) => line(x, y1, x, y2, { arrow: true, prefix: 'qkv-arrow' });

let base29 = text(0, 38, 'Three learned views', { fill: color.purple, weight: 600 });
base29 += text(391, 174, 'Embeddings', { size: 24, anchor: 'middle', fill: color.muted });
base29 += matrix(353, 205, color.blue);
base29 += tex(391, 313, '\\mathbf X', { anchor: 'middle', width: 90, size: 32 });
base29 += text(391, 342, 'four words', { size: 24, anchor: 'middle', fill: color.muted });
base29 += rect(844, 153, 282, 55, color.pale, 12);
base29 += text(985, 189, 'Dot products', { anchor: 'middle' });
base29 += down29(985, 208, 239);
base29 += rect(844, 240, 282, 55, color.tealPale, 12);
base29 += text(985, 276, 'Softmax', { anchor: 'middle', fill: color.teal });
base29 += down29(985, 295, 340);
base29 += rect(844, 341, 282, 55, color.tealPale, 12);
base29 += text(985, 377, 'Weighted sum', { anchor: 'middle', fill: color.teal });
base29 += down29(985, 396, 418);
base29 += matrix(953, 425, color.teal, { width: 64, height: 49 });
base29 += tex(912, 460, '\\mathbf Z', { size: 32, anchor: 'middle', width: 70, fill: color.teal });

for (const [index, [letter, label, hint, fill, pale]] of [
  ['Q', 'Query projection', 'What to look for', color.purple, color.purplePale],
  ['K', 'Key projection', 'What to match', color.blue, color.pale],
  ['V', 'Value projection', 'What to pass on', color.teal, color.tealPale],
].entries()) {
  const y = 104 + index * 131;
  let part = item(y - 6, label);
  part += text(28, y + 29, hint, { size: 24, fill: color.muted });
  part += route29(`M 429 239 H 478 V ${y} H 527`);
  part += rect(528, y - 28, 102, 56, pale, 12);
  part += tex(579, y + 13, `\\mathbf W_${letter}`, { size: 32, fill, anchor: 'middle', width: 100 });
  part += line(631, y, 685, y, { arrow: true, prefix: 'qkv-arrow' });
  part += matrix(686, y - 34, fill);
  part += tex(724, y - 47, `\\mathbf ${letter}`, { size: 31, fill, anchor: 'middle', width: 90 });
  const path = index === 0 ? 'M 763 104 H 804 V 172 H 842'
    : index === 1 ? 'M 763 235 H 803 V 190 H 842'
      : 'M 763 366 H 842';
  part += route29(path, { stroke: index === 2 ? color.teal : color.line });
  group29(['query-projection', 'key-projection', 'value-projection'][index], part);
}

content = item(448, 'Trainable projections');
content += text(28, 483, 'Shared across word positions', { size: 24, fill: color.muted });
content += rect(510, 421, 325, 60, color.purplePale, 12);
content += tex(672, 458, '\\mathbf W_Q,\\;\\mathbf W_K,\\;\\mathbf W_V', { anchor: 'middle', width: 309, size: 29, fill: color.purple });
group29('trainable-projections', content);

content = item(538, 'Matrix form');
content += line(328, 498, 1126, 498, { stroke: color.pale, width: 1.5 });
for (const [index, letter] of ['Q', 'K', 'V'].entries()) {
  content += tex(458 + index * 268, 539, `\\mathbf ${letter}=\\mathbf X\\mathbf W_${letter}`, {
    size: 30, anchor: 'middle', width: 252, fill: [color.purple, color.blue, color.teal][index],
  });
}
group29('matrix-form', content);

const svg29 = element('svg', {
  xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 1152 568', width: 1152, height: 552,
  'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img',
  'aria-labelledby': 'qkv-attention-title qkv-attention-description',
}, element('title', { id: 'qkv-attention-title' }, 'Learned query, key, and value projections')
  + element('desc', { id: 'qkv-attention-description' }, 'Five stages introduce the query, key, and value projections of the same input embeddings, identify their learned parameters, and show Q = X W_Q, K = X W_K, and V = X W_V. Queries and keys determine the weights used to combine values.')
  + element('defs', {}, definitions.replaceAll('id="arrow-', 'id="qkv-arrow-'))
  + base29 + groups29.map(item => item.content).join('\n'));
await writeFile(path.join(assets, 'self-attention-29-modern.svg'), svg29 + '\n');
await writeFile(path.join(assets, 'animation-29-modern.json'), JSON.stringify({
  source: source29.source, source_slide: source29.slide, effect: 'Appear',
  steps: groups29.map((item, index) => ({ click: index + 1, label: source29.steps[index].label, object_ids: [item.id] })),
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
let base40 = text(661, 556, 'Bank · of · the', { anchor: 'middle', size: 24, fill: color.purple });
base40 += text(960, 556, '0 · 1 · 2', { anchor: 'middle', size: 24, fill: color.blue });

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
content += tex(28, 83, 'p(t_{n+1}\\mid t_1,\\ldots,t_n)', { width: 394, size: 29, fill: color.teal });
content += up40(171, 130);
content += box40(97, 32, 'Final LayerNorm', { width: 270, size: 25, fill: '#f0f3f6' });
content += up40(97, 69);
content += box40(26, 42, 'LM head + softmax', { width: 420, size: 28, fill: color.tealPale, ink: color.teal });
content += line(946, 47, 985, 47, { stroke: color.teal, arrow: true, prefix: 'gpt-arrow' });
content += text(1048, 98, 'Choose', { size: 24, anchor: 'middle', fill: color.muted });
content += rect(988, 26, 120, 42, color.tealPale, 10);
content += text(1048, 55, 'river', { anchor: 'middle', size: 28, fill: color.teal, weight: 600 });
content += element('path', {
  d: 'M 1109 47 H 1134 V 566 H 778 V 546 H 761', fill: 'none', stroke: color.teal,
  'stroke-width': 1.8, 'stroke-dasharray': '6 5', 'marker-end': 'url(#gpt-arrow-teal)',
  'data-generation-loop': '',
});
content += text(1061, 275, 'Append', { size: 24, anchor: 'middle', fill: color.teal });
content += text(1061, 307, 'and repeat', { size: 24, anchor: 'middle', fill: color.teal });
group40('gpt-prediction', 'Predict a token from the final position; append it and repeat during generation', content);

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
  adaptation: 'New five-stage GPT-2-style teaching diagram replacing the encoder-decoder overview',
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
