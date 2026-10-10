import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { readFile, readdir, mkdir, realpath, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const slidesRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.dirname(slidesRoot);
const args = process.argv.slice(2);
const folder = args.find(arg => !arg.startsWith('--')) || 'example';
const exportPDF = args.includes('--pdf');
assert.match(folder, /^(example|template|lecture-\d{2}(?:-test)?|\d{2}-[a-z0-9-]+)$/, 'Use a lecture folder name, such as lecture-01 or lecture-05-test.');
const deck = path.join(slidesRoot, folder);
const source = await readFile(path.join(deck, 'slides.md'), 'utf8');
assert.doesNotMatch(source, /REPLACE:|\{\{[^}]+\}\}/, 'Replace the template prompts before checking a lecture.');
assert.doesNotMatch(source, /\bstyle\s*=|<style\b|r-fit-text/, 'Use the shared layouts instead of slide-specific styles or automatic text shrinking.');
const metadata = JSON.parse(await readFile(path.join(deck, 'lecture.json'), 'utf8'));
const defaultNotebook = metadata.notebook || 'practice.ipynb';
const notebooks = new Map(await Promise.all([defaultNotebook, ...(metadata.additional_notebooks || [])].map(async filename => {
  assert.match(filename, /^[a-z0-9][a-z0-9_-]*\.ipynb$/, 'Keep registered notebooks inside the lecture folder.');
  const notebook = JSON.parse(await readFile(path.join(deck, filename), 'utf8'));
  assert.equal(notebook.nbformat, 4);
  return [filename, notebook];
})));
const notebook = notebooks.get(defaultNotebook);
const slideSources = text => text.split(/^---$/m).map(section => section.trim()).filter(Boolean).map(section => ({
  id: /<!--\s*\.slide:[^>]*\bid="([^"]+)"/.exec(section)?.[1],
  section,
}));
const componentSources = ['lecture-05-test', 'lecture-05'].includes(folder)
  ? slideSources(await readFile(path.join(slidesRoot, 'lecture-05-test/slides.md'), 'utf8')) : [];
const adaptations = folder === 'lecture-05'
  ? JSON.parse(await readFile(path.join(deck, 'adaptations.json'), 'utf8')) : null;
for (const { id: slideId, section } of slideSources(source)) {
  const attributes = /<!--\s*\.slide:([^>]+)-->/m.exec(section)?.[1] || '';
  const filename = /\bdata-exercise-notebook="([^"]+)"/.exec(attributes)?.[1] || defaultNotebook;
  assert.ok(notebooks.has(filename), `${slideId}: register the exercise notebook ${filename} in lecture.json.`);
  for (const [, id] of section.matchAll(/(?:Exercise|Practice|Implementation)\s+([EP]\d+)/g)) {
    assert.ok(notebooks.get(filename).cells.some(cell => cell.cell_type === 'markdown'
      && new RegExp(`\\b${id}\\b`).test(cell.source.join(''))), `${slideId}: ${filename} is missing ${id}.`);
  }
}
if (folder === 'lecture-05') {
  const pilot = path.join(slidesRoot, 'lecture-05-test');
  assert.equal(createHash('sha256').update(await readFile(path.join(pilot, 'slides.md'))).digest('hex'),
    adaptations.source_sha256, 'Keep the archived source deck unchanged while adapting Lecture 05.');
  const adaptedSlides = new Set(Object.keys(adaptations.slides));
  const adaptedFiles = new Set(Object.keys(adaptations.files));
  const integrated = new Map(slideSources(source).map(({ id, section }) => [id, section]));
  assert.ok(componentSources.length > 0, 'Load the complete source deck for preservation checks.');
  const sourceIds = componentSources.map(({ id }) => id);
  const integratedIds = slideSources(source).map(({ id }) => id);
  assert.deepEqual(integratedIds.filter(id => sourceIds.includes(id)), sourceIds,
    'Keep the source sections in their approved teaching order.');
  for (const bundle of [
    ['context-for-text', 'self-attention', 'attention-roles', 'learned-projections'],
    ['gpt-architecture', 'position-inputs'],
    ['why-multiple-heads', 'multi-head-attention', 'attention-in-architecture'],
    ['feedforward-network', 'why-ffn', 'ffn-practice'],
    ['paper-experimental-setup', 'paper-parameter-count'],
  ]) {
    const start = integratedIds.indexOf(bundle[0]);
    assert.deepEqual(integratedIds.slice(start, start + bundle.length), bundle,
      `Keep the ${bundle[0]} teaching sequence adjacent.`);
  }
  for (const { id, section } of componentSources) {
    assert.ok(integrated.has(id), `${id}: retain the complete source topic sequence.`);
    if (!adaptedSlides.has(id)) {
      assert.equal(integrated.get(id), section, `${id}: record an authorized adaptation or preserve the source text.`);
    }
  }
  assert.ok([...adaptedSlides].every(id => sourceIds.includes(id)), 'Adaptation records must name source slide IDs.');
  const copied = ['practice.ipynb', 'position-math.js', 'position-interactions.js', 'import-slide.py'];
  for (const entry of await readdir(pilot, { withFileTypes: true })) {
    if (entry.isFile() && /^build-.*\.mjs$/.test(entry.name)) copied.push(entry.name);
  }
  async function includeAssets(directory = 'assets') {
    for (const entry of await readdir(path.join(pilot, directory), { withFileTypes: true })) {
      const relative = path.join(directory, entry.name);
      if (entry.isDirectory()) await includeAssets(relative);
      else if (entry.isFile()) copied.push(relative);
    }
  }
  await includeAssets();
  for (const filename of copied) {
    const [original, integrated] = await Promise.all([
      readFile(path.join(pilot, filename)), readFile(path.join(deck, filename)),
    ]);
    if (!adaptedFiles.has(filename)) {
      assert.ok(original.equals(integrated), `${filename}: record an authorized adaptation or preserve the source asset.`);
    }
  }
  if (!adaptedFiles.has('component-demo.js')) {
    assert.deepEqual(await readFile(path.join(deck, 'component-demo.js')), await readFile(path.join(pilot, 'demo.js')),
      'Record the integrated diagram loader adaptation.');
  }
}
const output = path.join(slidesRoot, '.checks', folder);
await mkdir(output, { recursive: true });
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.md': 'text/plain', '.json': 'application/json', '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml', '.ipynb': 'application/json', '.mp4': 'video/mp4', '.webm': 'video/webm' };
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    let target = path.resolve(root, '.' + pathname);
    const relative = path.relative(root, target);
    if (relative.startsWith('..') || relative.split(path.sep).some(part => part.startsWith('.') || ['workspace', 'node_modules'].includes(part))) throw new Error('Not found');
    if ((await stat(target)).isDirectory()) target = path.join(target, 'index.html');
    target = await realpath(target);
    if (!target.startsWith(root + path.sep)) throw new Error('Not found');
    response.writeHead(200, { 'Content-Type': mime[path.extname(target)] || 'application/octet-stream' });
    response.end(await readFile(target));
  } catch {
    response.writeHead(404);
    response.end('Not found');
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  await context.route('**/*', route => {
    if (new URL(route.request().url()).origin !== origin) {
      errors.push(`External runtime dependency: ${route.request().url()}`);
      return route.abort();
    }
    return route.continue();
  });
  const url = `${origin}/slides/${folder}/`;
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.evaluate(() => window.courseReady);
  assert.equal(await page.locator('#course-error').isVisible(), false);
  const count = await page.evaluate(() => Reveal.getTotalSlides());
  assert.ok(count > 0, 'No slides rendered.');
  if (folder === 'lecture-05') {
    assert.equal(count, 60, 'Retain the complete integrated topic sequence.');
    assert.deepEqual(adaptations.running_example.tokens, ['Noa', 'can', 'be', 'annoying', 'but', 'she']);
    const causal = JSON.parse(await readFile(path.join(deck, 'assets/attention-values.json'), 'utf8'));
    const positions = JSON.parse(await readFile(path.join(deck, 'assets/position-values.json'), 'utf8'));
    assert.deepEqual(causal.labels, [...adaptations.running_example.tokens, 'is']);
    assert.equal(causal.query_position, 5);
    assert.equal(causal.future_position, 6);
    assert.deepEqual(positions.tokens, adaptations.running_example.tokens);
    const teachingText = await page.locator('.slides > section').evaluateAll(sections => sections.map(section => {
      const content = section.cloneNode(true);
      content.querySelectorAll('aside.notes').forEach(note => note.remove());
      return content.textContent;
    }).join('\n'));
    assert.doesNotMatch(teachingText, /\bbank\b|\briver\b|\bCharlie\b|We are playing/i,
      'Use the Noa sentence throughout the teaching examples.');
  }
  const ids = await page.locator('.slides > section').evaluateAll(sections => sections.map(section => section.id));
  assert.ok(ids.every(Boolean), 'Give every slide a stable ID.');
  assert.equal(new Set(ids).size, ids.length, 'Slide IDs must be unique.');
  const checkedSlides = [];
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 720 }, { width: 1024, height: 768 }]) {
    await page.setViewportSize(viewport);
    for (let i = 0; i < count; i++) {
      await page.evaluate(async index => {
        Reveal.slide(index);
        Reveal.getCurrentSlide().querySelectorAll('.fragment').forEach(el => el.classList.add('visible'));
        await document.fonts.ready;
        Reveal.layout();
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      }, i);
      const result = await page.evaluate(() => {
        const slide = Reveal.getCurrentSlide();
        const box = slide.getBoundingClientRect();
        const scale = Reveal.getScale();
        const problems = [];
        const elements = [...slide.querySelectorAll('h1,h2,h3,p,li,pre,table,.columns,.katex-display,input,button,.plot,.plot text,img,video')].filter(el => !el.closest('aside.notes'));
        if (!slide.querySelector('h1,h2')) problems.push('Missing heading');
        if (slide.querySelector('.katex-error')) problems.push('Unrendered equation');
        for (const el of elements) {
          if (el.closest('.katex') && !el.classList.contains('katex-display')) continue;
          const bounds = el.getBoundingClientRect();
          const label = el.textContent.trim().slice(0, 70) || el.tagName;
          if (!bounds.width || !bounds.height) continue;
          if (bounds.left < box.left - 1 || bounds.right > box.right + 1 || bounds.top < box.top - 1 || bounds.bottom > box.bottom - 35 * scale) problems.push(`Outside slide content area: ${label}`);
          if (el.scrollWidth > el.clientWidth + 3 && ['PRE', 'TABLE', 'P', 'H1', 'H2', 'H3', 'LI'].includes(el.tagName)) problems.push(`Horizontal overflow: ${label}`);
          const font = parseFloat(getComputedStyle(el).fontSize);
          const minimum = el.closest('.source,.caption,.eyebrow,.exercise-meta,.plot') ? 24 : el.tagName === 'PRE' ? 26 : ['H1', 'H2'].includes(el.tagName) ? 44 : 26;
          if (font < minimum && (el.textContent.trim() || el.matches('input'))) problems.push(`Text too small (${font}px): ${label}`);
        }
        return { id: slide.id, title: slide.querySelector('h1,h2')?.textContent, problems };
      });
      if (result.problems.length) {
        await page.screenshot({ path: path.join(output, `failure-${viewport.width}.png`), animations: 'disabled' });
      }
      assert.deepEqual(result.problems, [], `${folder}/${result.id} at ${viewport.width}×${viewport.height}: ${result.problems.join('; ')}`);
      if (viewport.width === 1440) {
        checkedSlides.push(result);
        await page.screenshot({ path: path.join(output, `slide-${String(i + 1).padStart(2, '0')}.png`), animations: 'disabled' });
      }
    }
  }
  const assets = await page.locator('.slides [src], .slides [poster], .slides [data-plotly], .slides [data-excalidraw-source], .slides [data-manim-source]').evaluateAll(elements => elements.flatMap(el => {
    return ['src', 'poster', 'data-plotly', 'data-excalidraw-source', 'data-manim-source'].map(name => el.getAttribute(name)).filter(Boolean);
  }));
  for (const asset of new Set(assets)) {
    const target = new URL(asset, url);
    assert.equal(target.origin, origin, `Host visual assets with the course: ${asset}`);
    const response = await context.request.get(target.href);
    assert.ok(response.ok(), `Missing visual asset or editable source: ${asset}`);
    if (asset.endsWith('.excalidraw')) assert.equal((await response.json()).type, 'excalidraw');
  }
  assert.equal(await page.locator('.slides img:not([alt]), .slides video:not([aria-label]), .slides video:not([controls]), .slides video:not([poster])').count(), 0, 'Give images alt text and videos labels, controls, and posters.');
  const animations = page.locator('.slides video.animation');
  const posters = await animations.evaluateAll(videos => videos.map(video => {
    const poster = video.nextElementSibling;
    return {
      slide: video.closest('section').id,
      ready: poster?.matches('img.animation-poster') && poster.complete && poster.naturalWidth > 0,
      matchingSource: poster?.src === video.poster,
      matchingLabel: poster?.alt === (video.getAttribute('aria-label') || 'Animation summary'),
    };
  }));
  for (const poster of posters) {
    assert.ok(poster.ready && poster.matchingSource && poster.matchingLabel, `${poster.slide}: provide a loaded, labeled poster fallback for printing.`);
  }
  await page.emulateMedia({ media: 'print' });
  const printFallbacks = await animations.evaluateAll(videos => videos.map(video => ({
    slide: video.closest('section').id,
    videoHidden: getComputedStyle(video).display === 'none',
    posterVisible: getComputedStyle(video.nextElementSibling).display !== 'none',
  })));
  for (const fallback of printFallbacks) {
    assert.ok(fallback.videoHidden && fallback.posterVisible, `${fallback.slide}: printing must replace the video with its poster.`);
  }
  await page.emulateMedia({ media: 'screen' });
  for (let i = 0; i < await animations.count(); i++) {
    const video = animations.nth(i);
    const slide = await video.evaluate(element => {
      const slide = element.closest('section');
      Reveal.slide(Reveal.getIndices(slide).h);
      element.currentTime = 0;
      return slide.id;
    });
    await video.evaluate(element => element.play());
    await page.waitForFunction(element => element.currentTime > 0.1 && !element.paused, await video.elementHandle(), { timeout: 10000 });
    await page.evaluate(() => Reveal.slide((Reveal.getIndices().h + 1) % Reveal.getTotalSlides()));
    assert.equal(await video.evaluate(element => element.paused), true, `${slide}: leaving the slide must pause playback.`);
    await video.evaluate(element => { element.currentTime = 0; });
  }
  const links = await page.locator('a[href]').evaluateAll(anchors => anchors.map(a => ({ href: a.getAttribute('href'), target: a.target, rel: a.rel })));
  for (const link of links) {
    if (/^https?:/.test(link.href)) {
      if (!link.href.startsWith(origin)) assert.equal(link.target, '_blank');
    } else if (link.href.startsWith('mailto:')) {
      assert.match(link.href, /^mailto:[^\s@]+@[^\s@]+$/, `Invalid email link: ${link.href}`);
    } else if (link.href.startsWith('#/')) {
      assert.ok(ids.includes(link.href.slice(2)) || /^#\/\d/.test(link.href), `Broken slide link: ${link.href}`);
    } else if (!link.href.startsWith('#')) {
      const target = new URL(link.href, url);
      const response = await context.request.get(target.href);
      assert.ok(response.ok(), `Broken local link: ${target.href}`);
    }
  }
  await page.getByRole('button', { name: 'Help', exact: true }).click();
  assert.equal(await page.locator('#course-help').isVisible(), true);
  await page.getByRole('button', { name: 'Close', exact: true }).click();
  assert.equal(await page.locator('#course-help').isVisible(), false);
  if (folder === 'lecture-05-test' || folder === 'lecture-05') {
    if (folder === 'lecture-05-test') assert.equal(count, componentSources.length, 'Render every source component slide.');
    const showSlide = (id, fragment = -1) => page.evaluate(({ id, fragment }) =>
      Reveal.slide(Reveal.getIndices(document.getElementById(id)).h, 0, fragment), { id, fragment });
    const openFragmentLink = async (id, fragment) => {
      // A reset can precede Reveal's debounced URL update. Opening that same
      // hash would then do nothing. Load a fresh document to test an incoming
      // fragment link, including initialization of its fetched SVG artwork.
      await page.goto('about:blank');
      await page.goto(`${url}#/${id}/${fragment}`, { waitUntil: 'networkidle' });
      await page.evaluate(() => window.courseReady);
    };
    await page.setViewportSize({ width: 1440, height: 900 });
    for (const design of ['modern', 'original']) {
      await page.goto(`${url}${design === 'original' ? '?design=original' : ''}`, { waitUntil: 'networkidle' });
      await page.evaluate(() => window.courseReady);
      for (const [id, sourcePage, steps] of [['self-attention', 24, 8], ['learned-projections', 29, 5], ['gpt-architecture', 40, design === 'modern' ? 5 : 0]]) {
        const effectiveDesign = folder === 'lecture-05' && sourcePage !== 40 ? 'modern' : design;
        const index = await page.locator(`#${id}`).evaluate(section => Reveal.getIndices(section).h);
        const selector = `.attention-diagram[data-source-slide="${sourcePage}"]`;
        const diagram = page.locator(selector);
        const suffix = sourcePage === 24 ? '' : `-${sourcePage}`;
        const manifest = `animation${suffix}${effectiveDesign === 'original' ? '' : '-modern'}.json`;
        const animation = JSON.parse(await readFile(path.join(deck, 'assets', manifest), 'utf8'));
        assert.equal(animation.steps.length, steps, 'Keep the specified teaching stages.');
        assert.equal(await diagram.getAttribute('data-design'), effectiveDesign);
        if (steps === 0) {
          await page.evaluate(index => Reveal.slide(index, 0, -1), index);
          assert.equal(await diagram.locator('.fragment').count(), 0, 'The full GPT reference is a static comparison.');
          assert.equal(await page.locator('#step-status').textContent(), 'Reference');
          assert.equal(await page.getByRole('button', { name: 'Reset', exact: true }).isDisabled(), true);
          await page.screenshot({ path: path.join(output, 'original-gpt-architecture-reference.png'), animations: 'disabled' });
          continue;
        }
        if (effectiveDesign === 'modern') {
          assert.equal(await diagram.locator('image').count(), 0, 'Keep the modern artwork vector-based.');
          assert.equal(await diagram.locator('[data-explanation-item]').count(), sourcePage === 24 ? 7 : 5);
          const formulas = await diagram.locator('[data-tex]').count();
          assert.ok(formulas > 0 && await diagram.locator('[data-tex] .katex').count() === formulas, 'Render every equation with KaTeX.');
          assert.equal(await diagram.locator('.katex-error').count(), 0);
          if (sourcePage === 40) {
            assert.equal(await diagram.locator('[data-residual-path]').count(), 2, 'Both attention and MLP need residual connections.');
            assert.equal(await diagram.locator('[data-generation-loop]').count(), 1, 'Show how generation appends the next token.');
          }
          if (folder === 'lecture-05' && sourcePage === 29) {
            const expected = { X: ['6', 'd_model'], W_Q: ['d_model', 'd_k'], W_K: ['d_model', 'd_k'],
              W_V: ['d_model', 'd_v'], Q: ['6', 'd_k'], K: ['6', 'd_k'], V: ['6', 'd_v'],
              S: ['6', '6'], A: ['6', '6'], Z: ['6', 'd_v'] };
            const shapes = await diagram.locator('[data-matrix],[data-tensor]').evaluateAll(matrices =>
              Object.fromEntries(matrices.map(matrix => [matrix.dataset.matrix || matrix.dataset.tensor,
                [matrix.dataset.rows, matrix.dataset.columns]])));
            assert.deepEqual(shapes, expected, 'Show dimensions on inputs, projections, scores, and outputs.');
            for (const name of ['X', 'Q', 'K', 'V', 'Z']) {
              const rows = await diagram.locator(`[data-matrix="${name}"] [data-matrix-row]`).evaluateAll(rows =>
                rows.map(row => row.dataset.token));
              assert.deepEqual(rows, adaptations.running_example.tokens, `${name}: use the same six token rows.`);
            }
          }
        }
        const idsThrough = step => animation.steps.slice(0, step).flatMap(group => group.object_ids || group.shape_ids).sort();
        const visibleObjects = () => diagram.locator('[data-animation-object]').evaluateAll(shapes => shapes
          .filter(shape => getComputedStyle(shape.querySelector('[data-appear]')).visibility === 'visible')
          .map(shape => shape.dataset.animationObject).sort());
        await page.evaluate(index => Reveal.slide(index, 0, -1), index);
        await page.getByRole('button', { name: 'Reset', exact: true }).click();
        assert.deepEqual(await visibleObjects(), []);
        assert.equal(await page.locator('#step-status').textContent(), `Step 0 / ${steps}`);
        const prefix = `${design}-slide-${index + 1}`;
        await page.screenshot({ path: path.join(output, `${prefix}-step-00.png`), animations: 'disabled' });
        for (let step = 1; step <= steps; step++) {
          if (step % 2) await page.keyboard.press('ArrowRight');
          else await page.getByRole('button', { name: 'Next', exact: true }).click();
          assert.deepEqual(await visibleObjects(), idsThrough(step), `${design}, page ${sourcePage}: incorrect objects after click ${step}.`);
          assert.equal(await page.locator('#step-status').textContent(), `Step ${step} / ${steps}`);
          await page.waitForFunction(selector => [...document.querySelectorAll(`${selector} .fragment.visible`)]
            .every(fragment => Number(getComputedStyle(fragment).opacity) === 1), selector);
          await page.screenshot({ path: path.join(output, `${prefix}-step-${String(step).padStart(2, '0')}.png`), animations: 'disabled' });
        }
        if (effectiveDesign === 'modern') {
          const clippedText = await diagram.locator(':scope > svg').evaluate(svg => {
            const view = svg.viewBox.baseVal;
            const problems = [...svg.querySelectorAll('text,foreignObject')].filter(text => {
              const box = text.getBBox();
              return box.x < view.x || box.y < view.y || box.x + box.width > view.x + view.width || box.y + box.height > view.y + view.height;
            }).map(text => text.textContent);
            for (const math of svg.querySelectorAll('[data-math-box]')) {
              const box = math.getBoundingClientRect();
              const glyphs = [...math.querySelectorAll('.katex-html *')].filter(el => !el.childElementCount && el.textContent.replaceAll('\u200b', '').trim());
              if (glyphs.some(el => {
                const glyph = el.getBoundingClientRect();
                return glyph.left < box.left - 1 || glyph.right > box.right + 1 || glyph.top < box.top - 1 || glyph.bottom > box.bottom + 1;
              })) problems.push(math.querySelector('[data-tex]').dataset.tex);
            }
            return problems;
          });
          assert.deepEqual(clippedText, [], `Page ${sourcePage}: a label or KaTeX glyph is clipped.`);
        }
        for (let step = steps - 1; step >= 0; step--) {
          if (step % 2) await page.keyboard.press('ArrowLeft');
          else await page.getByRole('button', { name: 'Previous', exact: true }).click();
          assert.deepEqual(await visibleObjects(), idsThrough(step));
        }
        await diagram.click({ position: { x: 20, y: 20 } });
        assert.deepEqual(await visibleObjects(), idsThrough(1));
        await page.keyboard.press('Space');
        assert.equal(await page.locator('#step-status').textContent(), `Step 2 / ${steps}`);
        await page.getByRole('button', { name: 'Reset', exact: true }).click();
        assert.deepEqual(await visibleObjects(), []);
        assert.equal(await page.evaluate(() => Reveal.getIndices().h), index, 'Reset must stay on the current slide.');
      }
      // The merged lecture inserts ordinary sections between component groups.
      // Traverse every real boundary by its stable ID in both designs.
      const navigation = await page.evaluate(folder => Reveal.getSlides().map(section => ({
        id: section.id,
        steps: Math.max(0, ...[...section.querySelectorAll('.fragment')]
          .map(fragment => Number(fragment.dataset.fragmentIndex) + 1)),
        design: Boolean(section.querySelector(folder === 'lecture-05'
          ? '.attention-diagram[data-source-slide="40"]' : '.attention-diagram')),
      })), folder);
      await showSlide(navigation[0].id);
      assert.equal(await page.getByRole('button', { name: 'Previous', exact: true }).isDisabled(), true);
      for (let index = 0; index < navigation.length; index++) {
        const current = navigation[index];
        await showSlide(current.id, current.steps - 1);
        assert.equal(await page.locator('#design-switch').isVisible(), current.design);
        const last = index === navigation.length - 1;
        assert.equal(await page.getByRole('button', { name: 'Next', exact: true }).isDisabled(), last);
        if (last) continue;
        const following = navigation[index + 1];
        await page.getByRole('button', { name: 'Next', exact: true }).click();
        assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), following.id);
        if (following.steps) assert.equal(await page.locator('#step-status').textContent(), `Step 0 / ${following.steps}`);
        await page.getByRole('button', { name: 'Previous', exact: true }).click();
        assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), current.id);
        if (current.steps) assert.equal(await page.locator('#step-status').textContent(), `Step ${current.steps} / ${current.steps}`);
      }
      await showSlide('gpt-architecture', 4);
      await page.locator('#design-switch').click();
      await page.waitForLoadState('networkidle');
      await page.waitForFunction(() => Boolean(window.courseReady));
      await page.evaluate(() => window.courseReady);
      assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), 'gpt-architecture', 'Switching design keeps the current slide.');
      assert.equal(await page.locator('#source-link').getAttribute('href'), 'https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.svg');
      if (folder === 'lecture-05') {
        for (const viewport of [{ width: 1280, height: 720 }, { width: 1024, height: 768 }]) {
          await page.setViewportSize(viewport);
          const toolbar = await page.locator('.course-toolbar').boundingBox();
          assert.ok(toolbar.x >= 0 && toolbar.x + toolbar.width <= viewport.width,
            `Keep every lecture control within the ${viewport.width}px viewport.`);
          await page.screenshot({ path: path.join(output, `toolbar-${design}-${viewport.width}.png`), animations: 'disabled' });
        }
        await page.setViewportSize({ width: 1440, height: 900 });
      }
    }
    await openFragmentLink('gpt-architecture', 4);
    assert.equal(await page.locator('#step-status').textContent(), 'Step 5 / 5', 'Opening a fragment link must restore the requested reveal after SVG loading.');
    // Reveal debounces URL writes; reload only after the restored fragment is
    // reflected in the address bar, and verify that URL as part of this check.
    await page.waitForURL(url => url.hash === '#/gpt-architecture/4');
    await page.reload({ waitUntil: 'networkidle' });
    await page.evaluate(() => window.courseReady);
    assert.equal(await page.locator('#step-status').textContent(), 'Step 5 / 5', 'Reloading must preserve the current reveal.');
    // Exercise answers and SVG reveals use the same toolbar as the earlier slides.
    for (const design of ['modern', 'original']) {
      await page.goto(`${url}${design === 'original' ? '?design=original' : ''}`, { waitUntil: 'networkidle' });
      await page.evaluate(() => window.courseReady);
      const screenshotPrefix = design === 'modern' ? '' : 'original-';
      for (const [id, steps] of [['context-for-text', 3], ['attention-roles', 2], ['position-inputs', 3], ['sinusoidal-example', 2], ['rope-relative-offset', 2],
        ['scaled-dot-product', 4], ['why-multiple-heads', 2], ['multi-head-attention', 5], ['attention-in-architecture', 3],
        ['residual-addition', 1], ['layernorm-features', 1], ['normalization-order', 1], ['add-norm-practice', 1],
        ['feedforward-network', 3], ['why-ffn', 1], ['ffn-practice', 1], ['input-tokenization', 4],
        ...(ids.includes('paper-parameter-count') ? [['paper-parameter-count', 2], ['paper-translation-results', 1]] : [])]) {
        await page.evaluate(id => Reveal.slide(Reveal.getIndices(document.getElementById(id)).h, 0, -1), id);
        await page.getByRole('button', { name: 'Reset', exact: true }).click();
        for (let step = 0; step <= steps; step++) {
          if (step) await page.getByRole('button', { name: 'Next', exact: true }).click();
          assert.equal(await page.locator('#step-status').textContent(), `Step ${step} / ${steps}`);
          const indices = await page.locator(`#${id} .fragment.visible`).evaluateAll(fragments =>
            [...new Set(fragments.map(fragment => Number(fragment.dataset.fragmentIndex)))].sort());
          assert.deepEqual(indices, Array.from({ length: step }, (_, i) => i));
          if (id === 'context-for-text') {
            assert.equal(await page.locator(`#${id} [data-referent-link]`).isVisible(), step >= 1);
            assert.equal(await page.locator(`#${id}`).getByText('Future positions: masked', { exact: true }).isVisible(), step >= 2);
            assert.equal(await page.locator(`#${id} [data-math-box]`).isVisible(), step >= 3);
            await page.screenshot({ path: path.join(output, `${screenshotPrefix}context-for-text-step-${step}.png`), animations: 'disabled' });
          }
          if (id === 'attention-roles') {
            for (const role of ['query', 'key', 'value']) {
              assert.equal(await page.locator(`#${id} [data-attention-role="${role}"]`).isVisible(), step >= 1);
            }
            assert.equal(await page.locator(`#${id} [data-projection-motivation]`).isVisible(), step >= 2);
            await page.screenshot({ path: path.join(output, `${screenshotPrefix}attention-roles-step-${step}.png`), animations: 'disabled' });
          }
          if (id === 'why-multiple-heads') {
            for (const head of [1, 2, 3]) {
              const row = page.locator(`#${id} [data-illustrative-head="${head}"]`);
              assert.equal(await row.isVisible(), step >= (head === 1 ? 1 : 2));
              const weights = await row.locator('[data-attention-weight]').evaluateAll(cells =>
                cells.map(cell => ({ weight: Number(cell.dataset.attentionWeight), key: Number(cell.dataset.keyPosition) })));
              const query = Number(await row.getAttribute('data-query-position'));
              assert.equal(weights.length, adaptations?.running_example.tokens.length || 7);
              assert.ok(weights.every(({ weight, key }) => weight >= 0 && key <= query), 'All illustrated keys must be causally available.');
              assert.ok(Math.abs(weights.reduce((sum, { weight }) => sum + weight, 0) - 1) < 1e-12, 'Each illustrated row must sum to one.');
            }
            await page.screenshot({ path: path.join(output, `${screenshotPrefix}why-multiple-heads-step-${step}.png`), animations: 'disabled' });
          }
          if (['feedforward-network', 'why-ffn', 'ffn-practice'].includes(id)) {
            await page.screenshot({ path: path.join(output, `${screenshotPrefix}${id}-step-${step}.png`), animations: 'disabled' });
          }
        }
        await page.getByRole('button', { name: 'Previous', exact: true }).click();
        assert.equal(await page.locator('#step-status').textContent(), `Step ${steps - 1} / ${steps}`);
        if (['context-for-text', 'attention-roles', 'scaled-dot-product', 'why-multiple-heads', 'multi-head-attention', 'attention-in-architecture', 'feedforward-network', 'input-tokenization'].includes(id)) {
          await page.getByRole('button', { name: 'Reset', exact: true }).click();
          assert.equal(await page.locator('#step-status').textContent(), `Step 0 / ${steps}`);
          assert.equal(await page.locator(`#${id} .fragment.visible`).count(), 0);
          await page.locator(`#${id} .teaching-diagram`).click({ position: { x: 20, y: 20 } });
          assert.equal(await page.locator('#step-status').textContent(), `Step 1 / ${steps}`);
        }
      }
    }
    await openFragmentLink('context-for-text', 2);
    assert.equal(await page.locator('#step-status').textContent(), 'Step 3 / 3');
    await page.waitForURL(url => url.hash === '#/context-for-text/2');
    await page.reload({ waitUntil: 'networkidle' });
    await page.evaluate(() => window.courseReady);
    assert.equal(await page.locator('#step-status').textContent(), 'Step 3 / 3');
    await showSlide('rope-rotation');
    assert.equal(await page.locator('#step-status').textContent(), 'Explore');
    await page.locator('#rope-position').fill('3');
    const pair = JSON.parse(await page.locator('#rope-rotation-visual').getAttribute('data-pair'));
    assert.ok(Math.abs(pair[0]) < 1e-12 && Math.abs(pair[1] - 1) < 1e-12, 'At p=3, the toy vector must rotate by 90 degrees.');
    await page.locator('#rope-position').focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#rope-position').inputValue(), '4', 'Keyboard arrows should operate the focused slider.');
    assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), 'rope-rotation');
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.equal(await page.locator('#rope-position').inputValue(), folder === 'lecture-05' ? '5' : '2');
    await showSlide('rope-relative-offset', 1);
    await page.locator('#rope-shift').click();
    const relative = page.locator('#rope-relative-visual');
    assert.equal(await relative.getAttribute('data-shift'), '4');
    const initialRopeScore = folder === 'lecture-05' ? -Math.sqrt(3) / 2 : 0.5;
    assert.ok(Math.abs(Number(await relative.getAttribute('data-shifted-score')) - initialRopeScore) < 1e-12);
    await page.locator('#rope-gap').click();
    assert.equal(await relative.getAttribute('data-gap'), folder === 'lecture-05' ? '6' : '3');
    const changedRopeScore = folder === 'lecture-05' ? -1 : 0;
    assert.ok(Math.abs(Number(await relative.getAttribute('data-score')) - changedRopeScore) < 1e-12);
    assert.ok(Math.abs(Number(await relative.getAttribute('data-shifted-score')) - changedRopeScore) < 1e-12);
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.equal(await relative.getAttribute('data-shift'), '3');
    assert.equal(await relative.getAttribute('data-gap'), folder === 'lecture-05' ? '5' : '2');
    assert.equal(await page.locator('#step-status').textContent(), 'Step 0 / 2');
    await openFragmentLink('rope-relative-offset', 1);
    assert.equal(await page.locator('#step-status').textContent(), 'Step 2 / 2');
    await showSlide('position-methods');
    assert.equal(await page.getByRole('button', { name: 'Next', exact: true }).isDisabled(), false);
    await page.getByRole('button', { name: 'Next', exact: true }).click();
    assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), ids[ids.indexOf('position-methods') + 1]);
    assert.equal(await page.locator('[data-value-bypass]').count(), 1, 'Values bypass the RoPE rotation.');
    assert.equal(await page.locator('[data-single-value-path]').count(), 1, 'Values enter at the weighted sum.');
    assert.equal(await page.locator('[data-cross-attention-path]').count(), 1, 'Show encoder states feeding decoder cross-attention.');
    assert.equal(await page.locator('[data-tokenizer-rules-path]').count(), 1, 'The segmenter uses the learned vocabulary and rules.');
    if (folder === 'lecture-05') {
      assert.deepEqual(await page.locator('#input-tokenization [data-token-id]').evaluateAll(labels =>
        labels.map(label => Number(label.dataset.tokenId))), [0, 1, 2, 3, 4, 5],
      'Keep the illustrated prefix IDs consistent with the notebook vocabulary.');
    }
    await openFragmentLink('multi-head-attention', 4);
    assert.equal(await page.locator('#step-status').textContent(), 'Step 5 / 5');
    await page.waitForURL(url => url.hash === '#/multi-head-attention/4');
    await page.reload({ waitUntil: 'networkidle' });
    await page.evaluate(() => window.courseReady);
    assert.equal(await page.locator('#step-status').textContent(), 'Step 5 / 5');
    await showSlide('residual-addition', 0);
    assert.equal(await page.locator('#residual-addition .katex-display').count(), 1, 'Render the residual equation without Markdown interpreting its subscripts.');
    await showSlide('input-tokenization', 3);
    assert.equal(await page.getByRole('button', { name: 'Next', exact: true }).isDisabled(), ids.at(-1) === 'input-tokenization');
    for (const diagram of await page.locator('.position-diagram,.teaching-diagram').all()) {
      await diagram.evaluate(element => {
        const section = element.closest('section');
        Reveal.slide(Reveal.getIndices(section).h);
        section.querySelectorAll('.fragment').forEach(fragment => fragment.classList.add('visible'));
      });
      const problems = await diagram.locator(':scope > svg').evaluate(svg => {
        const view = svg.viewBox.baseVal;
        const problems = [...svg.querySelectorAll('text,foreignObject')].filter(el => {
          const box = el.getBBox();
          return box.x < -1 || box.y < -1 || box.x + box.width > view.width + 1 || box.y + box.height > view.height + 1;
        }).map(el => el.textContent);
        for (const math of svg.querySelectorAll('[data-math-box]')) {
          const box = math.getBoundingClientRect();
          const glyphs = [...math.querySelectorAll('.katex-html *')].filter(el => !el.childElementCount && el.textContent.replaceAll('\u200b', '').trim());
          if (glyphs.some(el => {
            const glyph = el.getBoundingClientRect();
            return glyph.left < box.left - 1 || glyph.right > box.right + 1 || glyph.top < box.top - 1 || glyph.bottom > box.bottom + 1;
          })) problems.push(math.querySelector('[data-tex]').dataset.tex);
        }
        const slideBox = svg.closest('section').getBoundingClientRect();
        if (svg.getBoundingClientRect().bottom > slideBox.bottom - 35 * Reveal.getScale()) problems.push('Diagram extends below the slide content area.');
        return problems;
      });
      assert.deepEqual(problems, [], 'Keep the new diagrams inside their SVG view boxes.');
      assert.equal(await diagram.locator('image').count(), 0, 'Keep the redraws as native vector artwork and text.');
    }
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.evaluate(() => window.courseReady);
  }
  if (folder === 'example' || folder === 'lecture-01') {
    const chartId = folder === 'lecture-01' ? 'heldout-results' : 'interactive-results';
    const expectedHover = folder === 'lecture-01' ? 'Chinese tokens: 48' : 'Total tokens: 18';
    await page.evaluate(id => Reveal.slide(Reveal.getIndices(document.getElementById(id)).h), chartId);
    const point = await page.locator(`#${chartId} .point`).nth(1).boundingBox();
    await page.mouse.move(point.x + point.width / 2, point.y + point.height / 2);
    await page.waitForFunction(({ id, expected }) => document.querySelector(`#${id} .hoverlayer`).textContent.includes(expected), { id: chartId, expected: expectedHover });
    assert.ok((await page.locator(`#${chartId} .hoverlayer`).textContent()).includes(expectedHover));
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('browser-demo')).h));
    await page.getByLabel('Try English, Chinese, or emoji').fill('🙂');
    assert.equal(await page.locator('#point-count').textContent(), '1');
    assert.equal(await page.locator('#byte-count').textContent(), '4');
    await page.getByRole('button', { name: 'Reset example' }).click();
    assert.equal(await page.locator('#point-count').textContent(), '3');
    assert.equal(await page.locator('#byte-count').textContent(), '10');
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('exercise-01')).h, 0, -1));
    assert.equal(await page.locator('#exercise-01 .answer').evaluate(el => el.classList.contains('visible')), false);
    await page.keyboard.press('Space');
    assert.equal(await page.locator('#exercise-01 .answer').evaluate(el => el.classList.contains('visible')), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(() => Reveal.isOverview()), true);
    await page.keyboard.press('Escape');
    await page.waitForURL(url => /^#\/exercise-01(?:\/\d+)?$/.test(url.hash));
    await page.reload({ waitUntil: 'networkidle' });
    await page.evaluate(() => window.courseReady);
    assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), 'exercise-01');
    assert.ok(await page.locator('.katex').count() > 0, 'Sample equations did not render.');
  }
  if (folder === 'lecture-05') {
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('position-demo')).h));
    const graph = page.locator('#position-visual');
    const initialDescription = await graph.getAttribute('aria-label');
    assert.equal(await graph.getAttribute('data-positions'), 'false');
    assert.equal(await graph.getAttribute('data-swapped'), 'false');
    assert.equal(Number(await graph.getAttribute('data-difference')), 0);
    await page.locator('#position-swap').click();
    await page.waitForFunction(() => document.getElementById('position-visual').dataset.swapped === 'true');
    assert.equal(Number(await graph.getAttribute('data-difference')), 0);
    assert.equal(await page.locator('#position-swap').getAttribute('aria-pressed'), 'true');
    await page.locator('#position-toggle').focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.getElementById('position-visual').dataset.positions === 'true');
    assert.ok(Math.abs(Number(await graph.getAttribute('data-difference')) - 0.44920448566702703) < 1e-12);
    assert.equal(await page.locator('#position-toggle').getAttribute('aria-pressed'), 'true');
    assert.match(await graph.getAttribute('aria-label'), /Positions on.*she, can, be, annoying, but, Noa.*0\.4492/);
    await page.locator('#position-swap').click();
    await page.waitForFunction(() => document.getElementById('position-visual').dataset.swapped === 'false');
    assert.equal(Number(await graph.getAttribute('data-difference')), 0);
    await page.locator('#position-reset').click();
    await page.waitForFunction(expected => document.getElementById('position-visual').getAttribute('aria-label') === expected, initialDescription);
    assert.equal(await page.locator('#position-toggle').getAttribute('aria-pressed'), 'false');
    assert.equal(await page.locator('#position-swap').getAttribute('aria-pressed'), 'false');
    await page.locator('#position-toggle').click();
    await page.waitForFunction(() => document.getElementById('position-visual').dataset.positions === 'true');
    await page.locator('#reset-animation').click();
    await page.waitForFunction(expected => document.getElementById('position-visual').getAttribute('aria-label') === expected, initialDescription);
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('exercise-05')).h, 0, -1));
    assert.equal(await page.locator('#exercise-05 .answer').evaluate(el => el.classList.contains('visible')), false);
    await page.keyboard.press('Space');
    assert.equal(await page.locator('#exercise-05 .answer').evaluate(el => el.classList.contains('visible')), true);
  }
  if (folder === 'lecture-04') {
    assert.equal(await page.locator('#shape-ledger tbody tr:last-child td').count(), 2);
    assert.equal(await page.locator('#shape-ledger tbody tr:last-child .katex').count(), 1, 'Render vocabulary bars inside the table as math.');
    for (const id of ['exercise-01', 'exercise-02', 'exercise-03', 'exercise-04', 'exercise-05']) {
      await page.evaluate(id => Reveal.slide(Reveal.getIndices(document.getElementById(id)).h, 0, -1), id);
      assert.equal(await page.locator(`#${id} .answer`).evaluate(el => el.classList.contains('visible')), false);
      await page.keyboard.press('Space');
      assert.equal(await page.locator(`#${id} .answer`).evaluate(el => el.classList.contains('visible')), true);
      if (id === 'exercise-04') {
        const limitations = page.locator('#recurrent-limitations');
        assert.equal(await limitations.evaluate(el => el.closest('.fragment').classList.contains('visible')), false);
        await page.keyboard.press('Space');
        assert.equal(await limitations.evaluate(el => el.closest('.fragment').classList.contains('visible')), true);
        assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), id);
      }
    }
  }
  if (folder === 'lecture-05') {
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('attention-demo')).h));
    const graph = page.locator('#attention-visual');
    const initialDescription = await graph.getAttribute('aria-label');
    const initialOutput = JSON.parse(await graph.getAttribute('data-output'));
    assert.equal(await graph.getAttribute('data-query'), '5');
    assert.equal(await graph.getAttribute('data-causal'), 'true');
    assert.ok(Math.abs(initialOutput[0] - 1) < 1e-12);
    assert.ok(Math.abs(initialOutput[1] - 8 / 11) < 1e-12);
    await page.locator('#attention-value').click();
    await page.waitForFunction(() => document.getElementById('attention-visual').dataset.changed === 'true');
    assert.deepEqual(JSON.parse(await graph.getAttribute('data-output')), initialOutput);
    await page.locator('#attention-mask').click();
    await page.waitForFunction(() => document.getElementById('attention-visual').dataset.causal === 'false');
    const leaked = JSON.parse(await graph.getAttribute('data-output'));
    assert.ok(Math.abs(leaked[0] - 25 / 7) < 1e-12);
    assert.ok(Math.abs(leaked[1] - 41 / 14) < 1e-12);
    const futureQuery = page.locator('#attention-demo button[data-query="6"]');
    await futureQuery.focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => document.getElementById('attention-visual').dataset.query === '6');
    assert.equal(await futureQuery.getAttribute('aria-pressed'), 'true');
    await page.locator('#attention-reset').click();
    await page.waitForFunction(expected => document.getElementById('attention-visual').getAttribute('aria-label') === expected, initialDescription);
    assert.deepEqual(JSON.parse(await graph.getAttribute('data-output')), initialOutput);
    assert.equal(await page.locator('#attention-mask').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#attention-value').getAttribute('aria-pressed'), 'false');
    await page.locator('#attention-value').click();
    await page.waitForFunction(() => document.getElementById('attention-visual').dataset.changed === 'true');
    await page.locator('#reset-animation').click();
    await page.waitForFunction(expected => document.getElementById('attention-visual').getAttribute('aria-label') === expected, initialDescription);
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('exercise-04')).h, 0, -1));
    assert.equal(await page.locator('#exercise-04 .answer').evaluate(el => el.classList.contains('visible')), false);
    await page.keyboard.press('Space');
    assert.equal(await page.locator('#exercise-04 .answer').evaluate(el => el.classList.contains('visible')), true);
  }
  if (folder === 'lecture-03') {
    assert.equal(count, 60, 'Keep the revised lecture at 60 slides.');
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('lookup-table')).h));
    assert.equal(await page.locator('#lookup-visual').getAttribute('data-selected-id'), '5');
    const initialLookup = await page.locator('#lookup-visual').textContent();
    await page.getByRole('button', { name: 'ID 1', exact: true }).click();
    assert.equal(await page.locator('#lookup-visual').getAttribute('data-selected-id'), '1');
    assert.match(await page.locator('#lookup-visual').getAttribute('aria-label'), /0\.85, 0\.69, -0\.32, -2\.12/);
    await page.getByRole('button', { name: 'ID 7', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('#lookup-visual').getAttribute('data-selected-id'), '7');
    await page.getByRole('button', { name: 'Reset lookup', exact: true }).click();
    assert.equal(await page.locator('#lookup-visual').textContent(), initialLookup);
    assert.equal(await page.getByRole('button', { name: 'ID 5', exact: true }).getAttribute('aria-pressed'), 'true');
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('training-memory')).h));
    assert.equal(await page.locator('#memory-visual').getAttribute('data-total-mib'), '250');
    await page.locator('#memory-rows').click();
    assert.equal(await page.locator('#memory-visual').getAttribute('data-total-mib'), '500');
    await page.locator('#memory-width').click();
    assert.equal(await page.locator('#memory-visual').getAttribute('data-total-mib'), '1000');
    assert.match(await page.locator('#memory-visual').getAttribute('aria-label'), /Adam moments 500 MiB/);
    const clipped = await page.locator('#memory-visual text').evaluateAll(labels => labels.filter(label => {
      const bounds = label.getBBox();
      return bounds.x < 0 || bounds.x + bounds.width > 1152 || bounds.y + bounds.height > 345;
    }).map(label => label.textContent));
    assert.deepEqual(clipped, [], 'Keep maximum-size memory labels inside the visual.');
    await page.locator('#memory-reset').click();
    assert.equal(await page.locator('#memory-visual').getAttribute('data-total-mib'), '250');
    await page.locator('#memory-rows').click();
    await page.locator('#memory-rows').click();
    assert.equal(await page.locator('#memory-visual').getAttribute('data-total-mib'), '250');
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('ngram-recap')).h));
    const graph = page.locator('#ngram-visual');
    assert.equal(await graph.getAttribute('data-context'), 'I');
    assert.deepEqual(await graph.evaluate(el => el.data[0].y), [2 / 3, 1 / 3]);
    const initialReview = await graph.getAttribute('aria-label');
    const trace = page.locator('#ngram-trace');
    const sample = page.locator('#ngram-sample');
    await trace.click();
    assert.equal(await graph.getAttribute('data-phase'), 'counts');
    assert.deepEqual(await graph.evaluate(el => el.data[0].y), [1, 0]);
    assert.equal(await sample.isDisabled(), true);
    await trace.focus();
    await page.keyboard.press('Enter');
    assert.deepEqual(await graph.evaluate(el => el.data[0].y), [2, 0]);
    await trace.click();
    assert.deepEqual(await graph.evaluate(el => el.data[0].y), [2, 1]);
    assert.equal(await trace.textContent(), 'Normalize');
    await trace.click();
    assert.equal(await graph.evaluate(el => el.layout.yaxis.title.text), 'Probability');
    assert.deepEqual(await graph.evaluate(el => el.data[0].y), [2 / 3, 1 / 3]);
    assert.equal(await sample.isDisabled(), false);
    await sample.click();
    assert.ok(['am', 'do'].includes(await graph.getAttribute('data-sampled')));
    await page.getByRole('button', { name: 'Use context Sam', exact: true }).click();
    assert.deepEqual(await graph.evaluate(el => el.data[0].y), [0.5, 0.5]);
    await sample.click();
    assert.ok(['EOS', 'I'].includes(await graph.getAttribute('data-sampled')));
    if (await graph.getAttribute('data-sampled') === 'EOS') assert.match(await graph.getAttribute('aria-label'), /stop/);
    await page.getByRole('button', { name: 'Use context BOS', exact: true }).click();
    assert.deepEqual(await graph.evaluate(el => el.data[0].x), ['I', 'Sam']);
    await page.getByRole('button', { name: 'Reset review', exact: true }).click();
    assert.equal(await graph.getAttribute('aria-label'), initialReview);
    assert.equal(await page.getByRole('button', { name: 'Use context I', exact: true }).getAttribute('aria-pressed'), 'true');
  }
  if (folder === 'lecture-01') {
    const developmentStart = ids.indexOf('outline-development');
    const currentModelsStart = ids.indexOf('models-2026');
    const preprocessingStart = ids.indexOf('outline-preprocessing');
    assert.equal(currentModelsStart - developmentStart - 1, 10, 'Lecture 01 needs ten historical slides before the 2026 updates.');
    assert.deepEqual(ids.slice(currentModelsStart, preprocessingStart), ['models-2026', 'terminal-bench-science', 'navier-stokes-2026'], 'The 2026 updates should connect models, scientific evaluation, and research results.');
    const slidePractice = [...source.matchAll(/(?:Exercise|Practice) ([EP]\d+)/g)].map(match => match[1]);
    const notebookPractice = notebook.cells.filter(cell => cell.cell_type === 'markdown')
      .flatMap(cell => [...cell.source.join('').matchAll(/^#{2,3} ([EP]\d+) ·/gm)].map(match => match[1]));
    assert.deepEqual(slidePractice, notebookPractice, 'Slide and notebook practice IDs must appear in the same order.');
    assert.doesNotMatch(source, /https?:\/\/(?:127\.0\.0\.1|localhost):\d+\/lab\//, 'Use the shared notebook launcher.');
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('bpe-live')).h));
    assert.match(await page.locator('#bpe-step').textContent(), /25 tokens/);
    await page.getByRole('button', { name: 'Next merge', exact: true }).click();
    assert.match(await page.locator('#bpe-step').textContent(), /18 tokens/);
    await page.getByRole('button', { name: 'Next merge', exact: true }).click();
    assert.match(await page.locator('#bpe-step').textContent(), /11 tokens/);
    assert.equal(await page.locator('#bpe-advance').isDisabled(), true);
    assert.match(await page.locator('#bpe-corpus').textContent(), /low · e · r/);
    await page.getByRole('button', { name: 'Overlap example', exact: true }).click();
    assert.match(await page.locator('#bpe-step').textContent(), /10 tokens/);
    await page.getByRole('button', { name: 'Next merge', exact: true }).click();
    assert.match(await page.locator('#bpe-step').textContent(), /8 tokens/);
    assert.match(await page.locator('#bpe-next').textContent(), /Pair count 4; replacements 2/);
    await page.getByRole('button', { name: 'Reset BPE', exact: true }).click();
    assert.match(await page.locator('#bpe-step').textContent(), /25 tokens/);
    assert.equal(await page.locator('#bpe-advance').isEnabled(), true);
    for (const id of ['example-video-johannesburg']) {
      const video = page.locator(`#${id} video.animation`);
      assert.equal(await video.count(), 1, `${id}: include the text-to-video example.`);
      assert.equal(await video.evaluate(element => element.autoplay || element.hasAttribute('autoplay')), false, `${id}: start playback only on request.`);
      assert.equal(await video.evaluate(element => element.controls && element.playsInline), true, `${id}: retain inline playback controls.`);
    }
    const outlines = await page.locator('.outline-topics').evaluateAll(lists => lists.map(list => ({
      topics: [...list.children].map(item => item.textContent),
      active: [...list.children].flatMap((item, index) => item.matches('[aria-current="step"]') ? [index] : []),
    })));
    assert.equal(outlines.length, 5);
    for (const outline of outlines) assert.deepEqual(outline.topics, outlines[0].topics);
    assert.deepEqual(outlines.map(outline => outline.active), [[0], [1], [2], [3], [3]]);
  }
  if (exportPDF) {
    if (folder === 'lecture-03') {
      // Reproduce a chart fetch that finishes after Reveal replaces the print
      // slide nodes; the demonstration must initialize on the final chart.
      await page.route('**/lecture-03/assets/ngram-review.json', async route => {
        await page.waitForFunction(() => document.querySelector('.pdf-page'));
        await route.continue();
      });
    }
    await page.goto(url + '?print-pdf', { waitUntil: 'networkidle' });
    await page.evaluate(() => window.courseReady);
    await page.waitForFunction(expected => document.querySelectorAll('.pdf-page').length === expected, count);
    if (folder === 'lecture-03') {
      assert.equal(await page.locator('#lookup-visual').getAttribute('data-selected-id'), '5');
      assert.equal(await page.locator('#memory-visual').getAttribute('data-total-mib'), '250');
      assert.equal(await page.locator('.pdf-page svg[role="img"]').count(), 2, 'Print both initial interactive examples.');
    }
    await page.evaluate(() => document.fonts.ready);
    await page.setViewportSize({ width: 1600, height: 1000 });
    await page.screenshot({ path: path.join(output, 'print-preview.png'), animations: 'disabled' });
    await page.emulateMedia({ media: 'print' });
    const printPadding = await page.locator('.pdf-page section').evaluateAll(sections => sections.map(section => parseFloat(getComputedStyle(section).paddingLeft)));
    assert.ok(printPadding.every(padding => padding >= 64), 'PDF export must preserve the slide content margins.');
    const printOverflow = await page.locator('.pdf-page section').evaluateAll(sections => sections.flatMap(section => {
      const box = section.getBoundingClientRect();
      return [...section.querySelectorAll('h1,h2,p,li,pre,table,.plot')]
        .filter(el => !el.closest('aside.notes'))
        .filter(el => el.getBoundingClientRect().bottom > box.bottom - 35)
        .map(el => `${section.id}: ${el.textContent.trim().slice(0, 60)}`);
    }));
    assert.deepEqual(printOverflow, [], 'Content extends into the PDF slide footer.');
    await page.pdf({ path: path.join(output, `${folder}.pdf`), printBackground: true, preferCSSPageSize: true });
  }
  assert.deepEqual(errors, [], 'Browser errors or missing runtime assets.');
  await writeFile(path.join(output, 'report.json'), JSON.stringify({ folder, slides: checkedSlides, viewports: 3, exportPDF, errors }, null, 2) + '\n');
  console.log(`Passed ${count} slides at three viewport sizes. Links, local assets, math, and interactions checked.`);
  console.log(`Review files: ${path.relative(root, output)}`);
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
