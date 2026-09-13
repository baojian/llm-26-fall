/* Check the bilingual reader and shared language controls in a real browser. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, realpath, stat } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(root, 'slides/package.json'));
const { chromium } = require('playwright');
const output = path.join(root, 'slides/.checks/notes');
await mkdir(output, { recursive: true });
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json', '.svg': 'image/svg+xml', '.md': 'text/plain', '.mmd': 'text/plain', '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf' };
const server = createServer(async (request, response) => {
  try {
    let target = path.resolve(root, '.' + decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
    const relative = path.relative(root, target);
    if (relative.startsWith('..') || relative.split(path.sep).some(part => part.startsWith('.') || ['workspace', 'node_modules'].includes(part))) throw new Error('Not found');
    if ((await stat(target)).isDirectory()) target = path.join(target, 'index.html');
    target = await realpath(target);
    if (!target.startsWith(root + path.sep)) throw new Error('Not found');
    response.writeHead(200, { 'Content-Type': mime[path.extname(target)] || 'application/octet-stream' });
    response.end(await readFile(target));
  } catch { response.writeHead(404); response.end('Not found'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const notePath = '/docs/lecture-01-pre-tokenization.html';
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1050 }, permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  await context.route('**/*', route => {
    if (route.request().url().startsWith(`file://${root}/`)) return route.continue();
    if (new URL(route.request().url()).origin !== origin) {
      errors.push(`External runtime request: ${route.request().url()}`);
      return route.abort();
    }
    if (route.request().url().endsWith('/_course/notebook')) return route.fulfill({ status: 200, contentType: 'text/plain', body: 'Static preview' });
    return route.continue();
  });
  await page.goto(origin + notePath, { waitUntil: 'networkidle' });
  const ids = await page.locator('[id]').evaluateAll(elements => elements.map(element => element.id));
  assert.equal(new Set(ids).size, ids.length, 'Duplicate HTML/SVG IDs.');
  const brokenAnchors = await page.locator('a[href^="#"]').evaluateAll(links => links.filter(link => !document.getElementById(decodeURIComponent(link.hash.slice(1)))).map(link => link.hash));
  assert.deepEqual(brokenAnchors, [], 'Broken section links.');
  const localLinks = await page.locator('a[href]:not([href^="#"])').evaluateAll(links => links.map(link => link.href).filter(href => new URL(href).origin === location.origin));
  for (const href of new Set(localLinks)) assert.ok((await context.request.get(href)).ok(), `Missing local link: ${href}`);
  assert.equal(await page.locator('.pipeline-panel svg').count(), 6);
  assert.equal(await page.locator('svg foreignObject').count(), 0, 'Use portable SVG text.');
  for (const lang of ['en', 'zh']) {
    await page.locator(`.lang-toggle [data-lang="${lang}"]`).click();
    assert.equal(await page.locator('html').getAttribute('lang'), lang === 'en' ? 'en' : 'zh-CN');
    assert.equal(await page.locator('.reading-article:visible').count(), 1);
    assert.equal(await page.locator('.reading-article:visible').getAttribute('data-note-language'), lang);
    assert.equal(await page.locator('.reading-sidebar ol:visible a').count(), 9);
    for (const view of ['corpus', 'learning', 'encoding']) {
      await page.locator(`#tab-${view}`).click();
      assert.equal(await page.locator('.pipeline-panel:visible').count(), 1);
      assert.equal(await page.locator('.pipeline-panel:visible').getAttribute('id'), `panel-${view}-${lang}`);
      const diagram = page.locator('.pipeline-panel:visible svg');
      assert.ok(await diagram.locator('title').textContent(), 'Diagram needs accessible title.');
      assert.ok(await diagram.locator('desc').textContent(), 'Diagram needs accessible description.');
      await page.locator('#pipeline').screenshot({ path: path.join(output, `pipeline-${view}-${lang}.png`) });
    }
    await page.locator('#tab-corpus').click();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#tab-learning').getAttribute('aria-selected'), 'true');
    await page.keyboard.press('End');
    assert.equal(await page.locator('#tab-encoding').getAttribute('aria-selected'), 'true');
    await page.keyboard.press('Home');
    assert.equal(await page.locator('#tab-corpus').getAttribute('aria-selected'), 'true');
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({ path: path.join(output, `reader-desktop-${lang}.png`) });
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      assert.equal(await page.locator('.reading-sidebar details').getAttribute('open'), null);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      assert.equal(overflow, false, `Reader overflows at ${width}px in ${lang}.`);
      for (const view of ['corpus', 'learning', 'encoding']) {
        await page.locator(`#tab-${view}`).click();
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      }
      await page.locator('.reading-sidebar summary').click();
      assert.equal(await page.locator('.reading-sidebar ol:visible a').count(), 9);
      await page.locator('.reading-sidebar summary').click();
      await page.locator('#tab-corpus').click();
      await page.evaluate(() => scrollTo(0, 0));
      if (width === 390) await page.screenshot({ path: path.join(output, `reader-mobile-${lang}.png`) });
    }
    await page.setViewportSize({ width: 1440, height: 1050 });
    await page.emulateMedia({ media: 'print' });
    assert.equal(await page.locator('.pipeline-panel:visible').count(), 3, 'Print all three diagrams in only the chosen language.');
    assert.equal(await page.locator('.reading-article:visible').count(), 1);
    await page.emulateMedia({ media: 'screen' });
  }
  const section = '5-which-steps-repeat-during-encoding';
  await page.goto(`${origin}${notePath}?lang=en#en-${section}`, { waitUntil: 'networkidle' });
  const before = await page.locator(`#en-${section}`).evaluate(element => element.getBoundingClientRect().top);
  await page.evaluate(() => {
    window.languageTrace = [];
    for (const type of ['course:before-language', 'course:language']) window.addEventListener(type, () => {
      const article = document.querySelector('.reading-article:not([hidden])');
      window.languageTrace.push({ type, scroll: scrollY, anchor: [...article.querySelectorAll('h2,h3')].filter(h => h.getBoundingClientRect().top <= 120).at(-1)?.id });
    });
  });
  await page.locator('.lang-toggle [data-lang="zh"]').click();
  const after = await page.locator(`#zh-${section}`).evaluate(element => element.getBoundingClientRect().top);
  assert.ok(Math.abs(before - after) <= 2, `Language switch should preserve reading position (${before} → ${after}). ${JSON.stringify(await page.evaluate(() => window.languageTrace))}`);
  assert.ok(page.url().endsWith(`#zh-${section}`));
  const code = page.locator('[data-note-language="zh"] pre code.language-python').first();
  const expectedCode = await code.textContent();
  await code.locator('xpath=../..').locator('.copy-code').click();
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), expectedCode);
  await page.goto(origin + '/index.html', { waitUntil: 'networkidle' });
  assert.equal(await page.locator('html').getAttribute('lang'), 'zh-CN', 'Remember preference across course pages.');
  await page.screenshot({ path: path.join(output, 'course-zh.png') });
  await page.locator('a[href="docs/lecture-01-pre-tokenization.html"]').click();
  assert.equal(await page.locator('.reading-article:visible').getAttribute('data-note-language'), 'zh');
  await page.goto(origin + '/slides/example/', { waitUntil: 'networkidle' });
  await page.evaluate(() => window.courseReady);
  assert.equal(await page.locator('html').getAttribute('lang'), 'zh-CN');
  assert.equal(await page.locator('.slides').getAttribute('lang'), 'en');
  assert.equal(await page.locator('.course-language-note').isVisible(), true);
  assert.equal(await page.locator('#help-button').textContent(), '帮助');
  await page.locator('#help-button').click();
  assert.equal(await page.locator('#course-help h2').textContent(), '课件使用说明');
  await page.locator('#course-help button').click();
  await page.screenshot({ path: path.join(output, 'slide-shell-zh.png') });
  await page.locator('.lang-toggle [data-lang="en"]').click();
  assert.equal(await page.locator('.course-language-note').isVisible(), false);
  await page.goto(origin + '/slides/shared/notebook.html?lecture=example&lang=zh', { waitUntil: 'networkidle' });
  assert.equal(await page.locator('#notebook-title').textContent(), 'JupyterLab 尚未就绪');
  assert.equal(await page.locator('#notebook-status').textContent(), '请使用下方的课程预览命令启动 JupyterLab。');
  await page.locator('.lang-toggle [data-lang="en"]').click();
  assert.equal(await page.locator('#notebook-title').textContent(), 'JupyterLab is not ready');
  await page.goto(origin + '/slides/shared/notebook.html?lang=zh', { waitUntil: 'networkidle' });
  assert.equal(await page.locator('#notebook-title').textContent(), '请先选择一讲课件');
  await page.goto(origin + '/slides/example/?print-pdf&lang=zh', { waitUntil: 'networkidle' });
  await page.evaluate(() => window.courseReady);
  assert.equal(await page.locator('.print-preview-toolbar span').textContent(), '预览');
  assert.equal(await page.getByRole('button', { name: '适合页面', exact: true }).count(), 1);
  // The reading HTML also works directly from disk without a build server or CDN.
  await page.goto(`file://${path.join(root, 'docs', path.basename(notePath))}`);
  await page.locator('.lang-toggle [data-lang="zh"]').click();
  assert.equal(await page.locator('.reading-article:visible').getAttribute('data-note-language'), 'zh');
  await page.locator('#tab-learning').click();
  assert.equal(await page.locator('#panel-learning-zh').isVisible(), true);
  assert.deepEqual(errors, [], 'Browser errors or external runtime dependencies.');
  console.log('Passed bilingual articles, six accessible diagrams, tabs, copying, anchors, mobile layout, print, offline viewing, and shared language controls.');
  console.log('Review images: slides/.checks/notes/');
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
