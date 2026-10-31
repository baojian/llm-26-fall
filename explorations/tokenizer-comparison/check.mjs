/* Run from any directory after build; screenshots stay in ignored workspace/. */
import assert from 'node:assert/strict';
import { readFile, mkdir, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const topic = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(topic, '../..');
const require = createRequire(path.join(root, 'slides/package.json'));
const { chromium } = require('playwright');
const output = path.join(root, 'workspace/tokenizer-comparison/browser-check');
await mkdir(output, { recursive: true });
const html = await readFile(path.join(topic, 'index.html'), 'utf8');
const report = JSON.parse(html.match(/<script id="measurements-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const errors = [];
const external = [];
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    let file = path.resolve(root, `.${pathname}`);
    assert.ok(file.startsWith(`${root}${path.sep}`));
    if ((await stat(file)).isDirectory()) file = path.join(file, 'index.html');
    const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json' };
    response.setHeader('Content-Type', `${types[path.extname(file)] || 'text/plain'}; charset=utf-8`);
    response.end(await readFile(file));
  } catch { response.writeHead(404).end('Not found'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/explorations/tokenizer-comparison/index.html`;
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 950 } });
  await context.route('**/*', route => {
    const target = new URL(route.request().url());
    if (target.protocol === 'file:' || target.hostname === '127.0.0.1') return route.continue();
    external.push(target.href);
    return route.abort();
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(url);
  assert.equal(await page.locator('#case option').count(), 32);
  assert.equal(await page.locator('.pipeline').count(), 2);
  await page.screenshot({ path: path.join(output, 'desktop.png') });
  await page.locator('#left').selectOption('bert');
  await page.locator('#right').selectOption('qwen3');
  for (const sample of report.cases) {
    await page.locator('#case').selectOption(sample.id);
    assert.equal(await page.locator('#input').textContent(), JSON.stringify(sample.text));
    assert.deepEqual(await page.locator('.count-row strong').allTextContents(),
      Object.values(sample.observed).map(value => String(value.token_count)));
    for (const [i, name] of ['bert', 'qwen3'].entries()) {
      const card = page.locator('.pipeline').nth(i);
      assert.equal(await card.locator('h3').textContent(), name);
      const decoded = card.locator('dt').filter({ hasText: '4 · Decoded text' }).locator('+ dd');
      assert.equal(await decoded.textContent(), JSON.stringify(sample.observed[name].decoded_text));
    }
  }
  await page.locator('#case').selectOption('emoji');
  assert.match(await page.locator('.pipeline').first().textContent(), /Unknown tokens/);
  await page.locator('#explore').screenshot({ path: path.join(output, 'inspector.png') });
  await page.reload();
  assert.equal(await page.locator('#case').inputValue(), 'emoji');
  assert.equal(await page.locator('#left').inputValue(), 'bert');
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Document overflow at ${width}px`);
    await page.locator('#explore').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, `mobile-${width}.png`) });
  }
  await page.locator('[data-lang="zh"]').click();
  assert.equal(await page.locator('label[for="case"]').textContent(), '示例');
  await page.locator('#case').selectOption('normalization');
  assert.match(await page.locator('.pipeline').last().textContent(), /NFC equivalent/);
  await page.locator('[data-lang="en"]').click();
  await page.setViewportSize({ width: 794, height: 1123 });
  await page.emulateMedia({ media: 'print' });
  assert.equal(await page.locator('#explore').isVisible(), false);
  assert.equal(await page.locator('#measurements').isVisible(), true);
  await page.locator('#results-vocabulary').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, 'print-tables.png') });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Print layout overflow');
  await page.emulateMedia({ media: 'screen' });
  // Every local publication link must resolve; internal anchors must exist.
  for (const href of await page.locator('a[href]').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')))) {
    if (/^https?:/.test(href)) continue;
    if (href.startsWith('#')) {
      assert.equal(await page.locator(`[id="${href.slice(1)}"]`).count(), 1, `Broken anchor ${href}`);
    } else {
      assert.ok(await stat(path.resolve(topic, href.split('#')[0])), `Broken local link ${href}`);
    }
  }
  await page.goto(pathToFileURL(path.join(topic, 'index.html')).href);
  await page.locator('#case').selectOption('mixed-script');
  assert.equal(await page.locator('.pipeline').count(), 2);
  const staticContext = await browser.newContext({ javaScriptEnabled: false });
  const staticPage = await staticContext.newPage();
  await staticPage.goto(pathToFileURL(path.join(topic, 'index.html')).href);
  assert.equal(await staticPage.locator('#measurements table').count(), 2);
  await page.goto(pathToFileURL(path.join(root, 'explorations/index.html')).href);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Collection overflow');
  await page.screenshot({ path: path.join(output, 'collection-mobile.png') });
  await page.getByRole('link', { name: 'Explore the report →', exact: true }).click();
  assert.equal(await page.locator('#case option').count(), 32);
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  console.log('Passed: 32 examples, pipeline values, URL persistence, language controls, 390/320px, print, links, file/offline, and no-JS reading.');
  console.log(`Screenshots: ${output}`);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
