/* Exercise the standalone reader offline, its filters, exports, and mobile layout. */
import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(root, 'slides/package.json'));
const { chromium } = require('playwright');
const output = path.join(root, 'workspace/tokenizer-note-check');
await mkdir(output, { recursive: true });
const source = path.join(root, 'docs/model-tokenizer-statistics/foundation-models.html');
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
const requests = [], errors = [];
await context.route(/^https?:/, route => { requests.push(route.request().url()); return route.abort(); });
const page = await context.newPage();
page.on('pageerror', error => errors.push(String(error)));
try {
  await page.goto(pathToFileURL(source).href, { waitUntil: 'load' });
  assert.equal(await page.locator('figure svg').count(), 2);
  const modelCount = await page.locator('#models tr').count();
  assert.ok(modelCount > 20);
  const ids = await page.locator('[id]').evaluateAll(nodes => nodes.map(n => n.id));
  assert.equal(new Set(ids).size, ids.length, 'Inline SVG identifiers must not collide.');
  await page.screenshot({ path: path.join(output, 'reader-desktop.png'), fullPage: true });
  await page.screenshot({ path: path.join(output, 'reader-top.png') });
  for (const [index, figure] of (await page.locator('figure').all()).entries()) {
    await figure.screenshot({ path: path.join(output, `figure-${index + 1}.png`) });
  }
  await page.locator('#data summary').focus();
  await page.keyboard.press('Enter');
  await page.locator('#limit').selectOption('1000');
  assert.equal(await page.locator('#models tr:visible').count(), modelCount);
  await page.locator('#search').fill('Qwen');
  const searchModels = await page.locator('#models tr:visible').evaluateAll(nodes => nodes.map(n => n.dataset.model));
  assert.ok(searchModels.length >= 3 && searchModels.every(id => id.startsWith('Qwen/')));
  await page.locator('#search').fill('');
  await page.locator('#compact').check();
  const compact = await page.locator('#models tr:visible').evaluateAll(nodes => nodes.map(n => Number(n.dataset.vocab)));
  assert.ok(compact.length >= 3 && compact.every(value => value >= 0 && value < 50000));
  await page.locator('#compact').uncheck();
  await page.locator('#sort').selectOption('year');
  const years = await page.locator('#models tr:visible').evaluateAll(nodes => nodes.map(n => Number(n.dataset.year)));
  assert.deepEqual(years, [...years].sort((a, b) => b - a));
  await page.locator('#search').fill('no-model-has-this-name');
  assert.equal(await page.locator('#models tr:visible').count(), 0);
  assert.ok(await page.locator('#empty').isVisible());
  await page.locator('#search').fill('');
  const downloadEvent = page.waitForEvent('download');
  await page.locator('#download-json').click();
  const download = await downloadEvent;
  const exportPath = path.join(output, download.suggestedFilename());
  await download.saveAs(exportPath);
  const exported = JSON.parse(await readFile(exportPath, 'utf8'));
  assert.equal(exported.models.length, modelCount, 'Export the full cohort, not just filtered rows.');
  const csvEvent = page.waitForEvent('download');
  await page.locator('#download-csv').click();
  const csv = await csvEvent;
  await csv.saveAs(path.join(output, csv.suggestedFilename()));
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(overflow, false, `No page-level overflow at ${width}px; figures/tables scroll internally.`);
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({ path: path.join(output, `reader-${width}.png`), fullPage: true });
    if (width === 390) await page.screenshot({ path: path.join(output, 'reader-mobile-top.png') });
  }
  assert.deepEqual(requests, [], 'The standalone reader must not need external requests.');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: 'passed', modelCount, figures: 2, externalRequests: requests.length,
    checks: ['direct-file/offline', 'keyboard', 'search', 'vocabulary filter', 'sort', 'empty state', 'JSON/CSV download', '390px', '320px'] }));
} finally {
  await browser.close();
}
