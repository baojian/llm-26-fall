// Editable redraws of pages 38–43 in the instructor's Desktop PowerPoint.
// The existing Reveal shell supplies all fonts, KaTeX CSS, and reveal controls.
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import katex from 'katex';

const assets = fileURLToPath(new URL('./assets/', import.meta.url));
const c = { ink: '#142e4b', muted: '#526374', line: '#bccbd7', pale: '#e5edf3',
  blue: '#527596', purple: '#7550b6', purplePale: '#eee8f7', teal: '#147d73', tealPale: '#e5f3ef' };
const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const el = (tag, attributes, content = '') => `<${tag} ${Object.entries(attributes).map(([key, value]) => `${key}="${esc(value)}"`).join(' ')}>${content}</${tag}>`;
const text = (x, y, value, { size = 30, fill = c.ink, anchor = 'start', weight = 400, ...attributes } = {}) => el('text', {
  x, y, fill, 'font-size': size, 'text-anchor': anchor, 'font-weight': weight, ...attributes,
}, esc(value));
const rect = (x, y, width, height, fill = c.pale, attributes = {}) => el('rect', { x, y, width, height, fill, rx: 11, ...attributes });
const path = (d, color = c.line, arrow = false, attributes = {}) => el('path', { d, fill: 'none', stroke: color, 'stroke-width': 2.5,
  ...(arrow ? { 'marker-end': `url(#mh-arrow-${color.slice(1)})` } : {}), ...attributes });
const line = (x1, y1, x2, y2, color = c.line, arrow = false) => path(`M ${x1} ${y1} L ${x2} ${y2}`, color, arrow);
const math = (x, y, value, { width = 480, height = 60, size = 32, anchor = 'start', fill = c.ink } = {}) => {
  const html = katex.renderToString(value, { output: 'htmlAndMathml', throwOnError: true, strict: 'error', trust: false });
  return el('foreignObject', { x, y, width, height, 'font-size': size, color: fill, 'data-math-box': '' },
    el('div', { xmlns: 'http://www.w3.org/1999/xhtml', class: 'attention-math', 'data-align': anchor, 'data-tex': value }, html));
};
const bullet = (x, y, value, options = {}) => el('circle', { cx: x + 7, cy: y - 10, r: 4, fill: c.ink }) + text(x + 27, y, value, options);
const box = (x, y, width, height, label, fill = c.pale, size = 30) => rect(x, y, width, height, fill)
  + text(x + width / 2, y + height / 2 + size * 0.35, label, { anchor: 'middle', size });
const mathBox = (x, y, width, height, label, fill = c.pale, size = 30) => rect(x, y, width, height, fill)
  + math(x + 7, y + 2, label, { width: width - 14, height: height - 4, size, anchor: 'middle' });
const twoLineBox = (x, y, width, height, lines, fill = c.pale, size = 28) => rect(x, y, width, height, fill)
  + text(x + width / 2, y + height / 2 - 3, lines[0], { anchor: 'middle', size })
  + text(x + width / 2, y + height / 2 + size + 2, lines[1], { anchor: 'middle', size });

function diagram(name, title, description, pages) {
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
        id: `mh-arrow-${color.slice(1)}`, viewBox: '0 0 10 10', refX: 8, refY: 5,
        markerWidth: 5, markerHeight: 5, orient: 'auto', markerUnits: 'strokeWidth',
      }, el('path', { d: 'M 1 1 L 8 5 L 1 9', fill: 'none', stroke: color, 'stroke-width': 1.8 }))).join('');
      const svg = el('svg', { xmlns: 'http://www.w3.org/2000/svg', width: 1152, height: 552, viewBox: '0 0 1152 552',
        'font-family': 'Arial, Helvetica Neue, sans-serif', role: 'img', 'aria-labelledby': `${name}-title ${name}-description` },
      el('title', { id: `${name}-title` }, esc(title)) + el('desc', { id: `${name}-description` }, esc(description))
        + el('defs', {}, definitions) + body).replaceAll('mh-arrow-', `${name}-arrow-`);
      await writeFile(`${assets}${name}.svg`, svg + '\n');
      await writeFile(`${assets}${name}.json`, JSON.stringify({ title, source_file: 'lecture-05-slides-transformers.pptx',
        source_copy: 'Desktop copy supplied by the instructor', source_pages: pages, steps }, null, 2) + '\n');
    },
  };
}

// Page 38, left: retain the upward computation and give the mask a causal meaning.
const single = diagram('mha-single-head', 'Scaled dot-product attention in one head',
  'Read upward: query-key dot products, scale by the square root of the key width, mask future keys, normalize each query row, and multiply by values. Values bypass the score calculation.', [38]);
single.add(text(2, 28, 'ATTENTION · COMPONENT 02', { size: 24, fill: c.muted, weight: 600 })
  + line(528, 8, 528, 535, c.pale)
  + math(612, 504, 'Q', { width: 100, height: 44, anchor: 'middle', fill: c.purple })
  + math(757, 504, 'K', { width: 100, height: 44, anchor: 'middle', fill: c.blue })
  + math(1025, 504, 'V', { width: 100, height: 44, anchor: 'middle', fill: c.teal }));
single.step('Compare queries and keys, then scale by the key dimension.',
  bullet(0, 88, 'Compare queries with keys')
  + math(27, 106, 'S=QK^{\\mathsf T}/\\sqrt{d_k}', { width: 492, height: 74, size: 32 })
  + line(662, 504, 662, 476, c.purple, true) + line(807, 504, 807, 476, c.blue, true)
  + mathBox(604, 425, 346, 47, 'QK^{\\mathsf T}')
  + line(777, 425, 777, 396, c.line, true)
  + mathBox(604, 345, 346, 47, '\\text{Scale by }1/\\sqrt{d_k}'));
single.step('For causal attention, set future-key logits to negative infinity.',
  bullet(0, 220, 'Block future positions')
  + math(27, 237, 'S_{tj}\\leftarrow-\\infty\\quad(j>t)', { width: 495, height: 64, size: 31 })
  + line(777, 345, 777, 316, c.line, true)
  + box(604, 265, 346, 47, 'Causal mask', c.purplePale));
single.step('Softmax normalizes over the keys for each query.',
  bullet(0, 348, 'Normalize each query row')
  + math(27, 364, 'A=\\operatorname{softmax}_{\\rm row}(S)', { width: 495, height: 68, size: 30 })
  + line(777, 265, 777, 236, c.line, true)
  + box(604, 185, 346, 47, 'Softmax over keys', c.tealPale));
single.step('Use the attention weights to combine the value vectors.',
  bullet(0, 473, 'Mix the value vectors')
  + math(27, 488, 'Z=AV', { width: 495, height: 60, size: 34, fill: c.teal })
  + line(777, 185, 777, 156, c.teal, true)
  + mathBox(604, 105, 346, 47, 'AV', c.tealPale, 32)
  + path('M 1075 504 V 129 H 956', c.teal, true, { 'data-single-value-path': '' })
  + line(777, 105, 777, 73, c.teal, true)
  + math(677, 12, 'Z', { width: 200, height: 58, anchor: 'middle', size: 36, fill: c.teal })
  + text(958, 204, 'Values', { size: 26, fill: c.teal }));
await single.save();

// Page 38, right: independent projections, parallel attention, concatenate, W^O.
const heads = diagram('mha-parallel-heads', 'Parallel heads, then an output projection',
  'The same hidden states feed several learned Q/K/V projections. Each head computes its own attention map. Concatenate the head outputs along features and apply the learned output matrix. Eight heads at model width 512 have width 64 each in the original base model.', [38, 39]);
heads.add(text(2, 27, 'SAME TOKENS · DIFFERENT PROJECTIONS', { size: 24, fill: c.muted, weight: 600 })
  + mathBox(460, 491, 674, 54, 'X\\in\\mathbb R^{n\\times d_{\\rm model}}', c.pale, 30));
let content = bullet(0, 87, 'Project the same inputs')
  + math(27, 101, 'Q_i=XW_i^Q', { width: 395, height: 58, size: 31 })
  + text(27, 182, 'Likewise for keys and values.', { size: 26, fill: c.muted });
const headCenters = [553, 780, 1039];
const headNames = ['1', '2', 'h'];
const headColors = [c.purple, c.blue, c.teal];
const headFills = [c.purplePale, c.pale, c.tealPale];
content += path('M 797 491 V 464 M 553 464 H 1039', c.line);
for (const [i, cx] of headCenters.entries()) {
  content += line(cx, 464, cx, 437, headColors[i], true)
    + mathBox(cx - 98, 363, 196, 70, `W_{${headNames[i]}}^Q,W_{${headNames[i]}}^K,W_{${headNames[i]}}^V`, headFills[i], 28);
}
content += text(909, 407, '…', { size: 36, anchor: 'middle', fill: c.muted });
heads.step('Each head learns separate Q, K, and V projections of all input features.', content);
content = bullet(0, 246, 'Attend in parallel') + text(27, 286, 'One attention map per head', { size: 26, fill: c.muted });
for (const [i, cx] of headCenters.entries()) {
  content += line(cx, 363, cx, 325, headColors[i], true)
    + box(cx - 98, 257, 196, 64, `Head ${headNames[i]}`, headFills[i])
    + math(cx - 90, 207, `H_{${headNames[i]}}`, { width: 180, height: 47, size: 30, anchor: 'middle', fill: headColors[i] })
    + line(cx, 257, cx, 250, headColors[i]);
}
content += text(909, 301, '…', { size: 36, anchor: 'middle', fill: c.muted });
heads.step('Compute scaled dot-product attention independently in every head.', content);
content = bullet(0, 349, 'Concatenate the outputs') + text(27, 389, 'Join features for the same token.', { size: 26, fill: c.muted });
for (const [i, cx] of headCenters.entries()) content += line(cx, 208, cx, 185, headColors[i], true);
content += rect(455, 137, 682, 44, c.pale)
  + rect(455, 137, 215, 44, c.purplePale) + rect(919, 137, 218, 44, c.tealPale)
  + math(470, 135, '\\operatorname{Concat}(H_1,\\ldots,H_h)', { width: 650, height: 48, size: 28, anchor: 'middle' });
heads.step('Concatenate along the feature axis; keep the token rows aligned.', content);
heads.step('The output projection mixes the concatenated head features.',
  bullet(0, 450, 'Mix head outputs')
  + line(797, 137, 797, 113, c.line, true)
  + mathBox(641, 59, 311, 50, 'W^O', c.tealPale, 32)
  + line(797, 59, 797, 43, c.teal, true)
  + math(612, 0, 'Y\\in\\mathbb R^{n\\times d_{\\rm model}}', { width: 367, height: 42, size: 29, anchor: 'middle', fill: c.teal })
  + text(27, 502, 'Practice E03 · 1 min · width per head?', { size: 24, fill: c.muted })
  + math(27, 514, '512\\,/\\,8=', { width: 186, height: 38, size: 28 }));
heads.step('In the original base model, 512 dimensions divided among 8 heads gives 64 per head.',
  math(155, 514, '64', { width: 85, height: 38, size: 28, fill: c.teal }));
await heads.save();

// Pages 39–40: show the three attention roles, then isolate the course's causal path.
const architecture = diagram('mha-architecture', 'Where multi-head attention is used',
  'The original encoder-decoder has encoder self-attention, causal decoder self-attention, and cross-attention to encoder outputs. The GPT-style decoder uses causal self-attention within one prefix. Read upward. Residual connections and normalization are omitted to isolate attention paths.', [39, 40]);
architecture.add(text(308, 32, '2017 encoder–decoder', { size: 32, anchor: 'middle', weight: 600 })
  + line(651, 8, 651, 509, c.pale)
  + text(925, 32, 'GPT-style model', { size: 32, anchor: 'middle', weight: 600 })
  + text(576, 546, 'Attention paths only; residuals and normalization omitted.', { size: 24, anchor: 'middle', fill: c.muted }));
architecture.step('The encoder attends over the source sequence.',
  box(16, 466, 225, 46, 'Source states', c.pale, 28)
  + line(128, 466, 128, 434, c.line, true)
  + rect(3, 238, 251, 200, 'none', { stroke: c.line, 'stroke-dasharray': '6 5', 'stroke-width': 2 })
  + twoLineBox(16, 354, 225, 76, ['Self-attention', 'all source tokens'], c.tealPale, 26)
  + line(128, 354, 128, 316, c.line, true)
  + box(16, 259, 225, 53, 'Feed-forward', c.pale, 28)
  + line(128, 259, 128, 188, c.teal, true)
  + text(128, 167, 'Encoder output', { size: 27, anchor: 'middle', fill: c.teal })
  + math(175, 198, '\\times N', { width: 78, height: 40, size: 25, fill: c.muted }));
architecture.step('The decoder uses causal self-attention and cross-attention to the encoder.',
  box(369, 466, 252, 46, 'Target states', c.pale, 28)
  + line(495, 466, 495, 434, c.line, true)
  + rect(356, 133, 278, 305, 'none', { stroke: c.line, 'stroke-dasharray': '6 5', 'stroke-width': 2 })
  + twoLineBox(369, 354, 252, 76, ['Self-attention', 'causal mask'], c.purplePale, 28)
  + line(495, 354, 495, 321, c.purple, true)
  + math(508, 323, 'Q', { width: 57, height: 30, size: 25, fill: c.purple })
  + box(369, 258, 252, 59, 'Cross-attention', c.tealPale, 28)
  + path('M 128 178 H 298 V 287 H 365', c.teal, true, { 'data-cross-attention-path': '' })
  + math(270, 292, 'K,V', { width: 95, height: 43, size: 27, fill: c.teal })
  + line(495, 258, 495, 213, c.line, true)
  + box(369, 156, 252, 53, 'Feed-forward', c.pale, 28)
  + math(560, 218, '\\times N', { width: 66, height: 33, size: 25, fill: c.muted })
  + line(495, 156, 495, 121, c.line, true)
  + box(369, 63, 252, 54, 'Linear + softmax', c.tealPale, 27));
architecture.step('The GPT-style model keeps the causal self-attention path.',
  box(745, 466, 362, 46, 'Prefix states', c.pale, 30)
  + line(926, 466, 926, 434, c.line, true)
  + rect(732, 238, 388, 200, 'none', { stroke: c.line, 'stroke-dasharray': '6 5', 'stroke-width': 2 })
  + twoLineBox(745, 354, 362, 76, ['Self-attention', 'causal mask'], c.purplePale, 30)
  + line(926, 354, 926, 316, c.line, true)
  + box(745, 259, 362, 53, 'Feed-forward / MLP', c.pale, 30)
  + line(926, 259, 926, 188, c.line, true)
  + math(1034, 198, '\\times L', { width: 84, height: 40, size: 25, fill: c.muted })
  + box(745, 130, 362, 54, 'LM head + softmax', c.tealPale, 30)
  + text(926, 91, 'Next-token probabilities', { size: 28, anchor: 'middle', fill: c.teal })
  + line(926, 130, 926, 101, c.teal, true));
await architecture.save();

// Pages 41–42 on the left; page 43 on the right, as requested.
const tokenizer = diagram('tokenizer-inputs', 'From raw text to token embeddings',
  'Left: words, characters, and data-learned subwords; a toy sentence splits into We, are, play, ing, with invented token IDs leading to learned embeddings. Right: BPE, Unigram, and WordPiece share separate vocabulary-learning and segmentation stages.', [41, 42, 43]);
tokenizer.add(text(2, 34, 'Text → tokens → embeddings', { size: 32, weight: 600 })
  + line(576, 8, 576, 541, c.pale)
  + text(607, 34, 'Three common tokenizers', { size: 32, weight: 600 }));
let left = bullet(0, 88, 'Words, characters, or subwords', { size: 29 })
  + bullet(0, 133, 'Learn subword units from data', { size: 29 })
  + text(280, 167, 'Additive input example', { size: 24, fill: c.muted, anchor: 'middle' })
  + mathBox(25, 178, 510, 64, 'x_p=e_{t_p}+PE(p)', c.tealPale, 32)
  + line(280, 273, 280, 247, c.teal, true)
  + box(25, 273, 510, 74, '', c.pale)
  + text(280, 304, 'Learned embedding lookup', { anchor: 'middle', size: 28 });
const tokens = ['We', 'are', 'play', 'ing'];
for (const [i, token] of tokens.entries()) {
  const cx = 76 + 132 * i;
  left += Array.from({ length: 4 }, (_, j) => rect(cx - 46 + j * 24, 325, 19, 17, c.blue)).join('')
    + line(cx, 366, cx, 351, c.blue, true)
    + text(cx, 389, String(10 * (i + 1)), { anchor: 'middle', size: 28, fill: c.blue })
    + line(cx, 415, cx, 400, c.line, true)
    + box(cx - 55, 421, 110, 43, token, c.purplePale, 29);
}
left += line(280, 493, 280, 469, c.purple, true)
  + text(280, 526, '“We are playing”', { anchor: 'middle', size: 30 })
  + text(280, 547, 'Illustrative split and token IDs', { anchor: 'middle', size: 24, fill: c.muted });
tokenizer.step('Convert text into token IDs, then look up trainable embeddings.', left);
tokenizer.step('BPE, Unigram, and WordPiece are three subword approaches.',
  bullet(605, 90, 'Byte-Pair Encoding (BPE)', { size: 29 })
  + text(632, 124, 'Sennrich et al., 2016', { size: 24, fill: c.muted })
  + bullet(605, 175, 'Unigram language model', { size: 29 })
  + text(632, 209, 'Kudo, 2018', { size: 24, fill: c.muted })
  + bullet(605, 260, 'WordPiece', { size: 29 })
  + text(632, 294, 'Schuster & Nakajima, 2012', { size: 24, fill: c.muted }));
tokenizer.step('Fit the tokenizer on a corpus to obtain its vocabulary and rules.',
  text(607, 345, 'Train, then apply', { size: 30, weight: 600 })
  + text(610, 399, 'Corpus', { size: 27 })
  + line(703, 389, 754, 389, c.line, true)
  + box(759, 364, 171, 50, 'Learner', c.purplePale, 28)
  + line(934, 389, 970, 389, c.purple, true)
  + text(1060, 383, 'Vocabulary', { anchor: 'middle', size: 26, fill: c.purple })
  + text(1060, 412, '+ rules', { anchor: 'middle', size: 24, fill: c.muted }));
tokenizer.step('Use the fitted tokenizer to segment new text; the embedding table is learned separately with the model.',
  text(610, 518, 'New text', { size: 27 })
  + line(720, 507, 754, 507, c.line, true)
  + box(759, 482, 171, 50, 'Segmenter', c.tealPale, 27)
  + line(934, 507, 981, 507, c.teal, true)
  + text(990, 518, 'Token IDs', { size: 27, fill: c.teal })
  + path('M 1060 420 V 451 H 845 V 477', c.purple, true, { 'data-tokenizer-rules-path': '' }));
await tokenizer.save();

console.log('Built three multi-head attention diagrams and the split input/tokenizer slide.');
