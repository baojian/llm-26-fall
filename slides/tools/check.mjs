import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, realpath, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const slidesRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.dirname(slidesRoot);
const args = process.argv.slice(2);
const folder = args.find(arg => !arg.startsWith('--')) || 'example';
const exportPDF = args.includes('--pdf');
assert.match(folder, /^(example|template|\d{2}-[a-z0-9-]+)$/, 'Use a lecture folder name, such as 01-tokenization.');
const deck = path.join(slidesRoot, folder);
const source = await readFile(path.join(deck, 'slides.md'), 'utf8');
assert.doesNotMatch(source, /REPLACE:|\{\{[^}]+\}\}/, 'Replace the template prompts before checking a lecture.');
assert.doesNotMatch(source, /\bstyle\s*=|<style\b|r-fit-text/, 'Use the shared layouts instead of slide-specific styles or automatic text shrinking.');
const notebook = JSON.parse(await readFile(path.join(deck, 'practice.ipynb'), 'utf8'));
assert.equal(notebook.nbformat, 4);
for (const [, id] of source.matchAll(/Exercise (E\d+)/g)) {
  assert.ok(notebook.cells.some(cell => cell.cell_type === 'markdown' && cell.source.join('').includes(id)), `Notebook is missing ${id}.`);
}
const output = path.join(slidesRoot, '.checks', folder);
await mkdir(output, { recursive: true });
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.md': 'text/plain', '.json': 'application/json', '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf', '.png': 'image/png', '.svg': 'image/svg+xml', '.ipynb': 'application/json', '.mp4': 'video/mp4', '.webm': 'video/webm' };
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
  const links = await page.locator('a[href]').evaluateAll(anchors => anchors.map(a => ({ href: a.getAttribute('href'), target: a.target, rel: a.rel })));
  for (const link of links) {
    if (/^https?:/.test(link.href)) {
      if (!link.href.startsWith(origin)) assert.equal(link.target, '_blank');
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
  if (folder === 'example') {
    await page.evaluate(() => Reveal.slide(Reveal.getIndices(document.getElementById('interactive-results')).h));
    const point = await page.locator('#interactive-results .point').nth(1).boundingBox();
    await page.mouse.move(point.x + point.width / 2, point.y + point.height / 2);
    await page.waitForFunction(() => document.querySelector('#interactive-results .hoverlayer').textContent.includes('Total tokens: 18'));
    assert.match(await page.locator('#interactive-results .hoverlayer').textContent(), /Total tokens: 18/);
    await page.evaluate(() => Reveal.slide(4));
    await page.getByLabel('Try English, Chinese, or emoji').fill('🙂');
    assert.equal(await page.locator('#point-count').textContent(), '1');
    assert.equal(await page.locator('#byte-count').textContent(), '4');
    await page.getByRole('button', { name: 'Reset example' }).click();
    assert.equal(await page.locator('#point-count').textContent(), '3');
    assert.equal(await page.locator('#byte-count').textContent(), '10');
    await page.evaluate(() => Reveal.slide(7, 0, -1));
    assert.equal(await page.locator('#exercise-01 .answer').evaluate(el => el.classList.contains('visible')), false);
    await page.keyboard.press('Space');
    assert.equal(await page.locator('#exercise-01 .answer').evaluate(el => el.classList.contains('visible')), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(() => Reveal.isOverview()), true);
    await page.keyboard.press('Escape');
    await page.waitForURL(url => url.hash === '#/exercise-01');
    await page.reload({ waitUntil: 'networkidle' });
    await page.evaluate(() => window.courseReady);
    assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), 'exercise-01');
    assert.ok(await page.locator('.katex').count() > 0, 'Sample equations did not render.');
  }
  if (exportPDF) {
    await page.goto(url + '?print-pdf', { waitUntil: 'networkidle' });
    await page.evaluate(() => window.courseReady);
    await page.waitForFunction(expected => document.querySelectorAll('.pdf-page').length === expected, count);
    await page.evaluate(() => document.fonts.ready);
    const printPadding = await page.locator('.pdf-page section').evaluateAll(sections => sections.map(section => parseFloat(getComputedStyle(section).paddingLeft)));
    assert.ok(printPadding.every(padding => padding >= 64), 'PDF export must preserve the slide content margins.');
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
