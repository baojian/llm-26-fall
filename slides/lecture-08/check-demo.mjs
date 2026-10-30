import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const origin = process.argv[2] || 'http://127.0.0.1:8000';
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const errors = [];
  const external = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (!request.url().startsWith(origin) && !request.url().startsWith('data:')) {
      external.push(request.url());
    }
  });
  await page.goto(`${origin}/slides/lecture-08/#/evaluation-demo`);
  await page.evaluate(() => window.courseReady);
  assert.equal(await page.locator('#historical-shortcuts .ytitle').textContent(), 'Accuracy (%)');
  assert.equal(await page.locator('#evaluation-demo .ytitle').textContent(), 'Accuracy (%)');
  assert.equal(await page.locator('#bootstrap-result .ytitle').textContent(), 'Bootstrap samples');
  assert.match(await page.locator('#bootstrap-result .xtitle').textContent(), /percentage points/);
  for (const [size, scores, pressed] of [
    [120, [100 * 79 / 120, 62.5], 'false'],
    [100, [60, 64], 'true'],
  ]) {
    if (size === 100) await page.locator('#evaluation-toggle').click();
    await page.waitForFunction(expected =>
      document.querySelector('#evaluation-plot').dataset.evaluationSize === String(expected), size);
    const actual = await page.locator('#evaluation-plot').evaluate(plot => plot.data[0].y);
    actual.forEach((value, i) => assert.ok(Math.abs(value - scores[i]) < 1e-10));
    assert.equal(await page.locator('#evaluation-toggle').getAttribute('aria-pressed'), pressed);
    assert.match(await page.locator('#evaluation-status').textContent(), /Synthetic/);
  }
  await page.locator('#reset-button').click();
  await page.waitForFunction(() =>
    document.querySelector('#evaluation-plot').dataset.evaluationSize === '120');
  assert.equal(await page.locator('#evaluation-toggle').textContent(), 'Mixed evaluation');
  assert.match(await page.locator('#evaluation-status').textContent(), /65\.8%/);
  assert.deepEqual(errors, []);
  assert.deepEqual(external, [], 'The classroom deck must load without external services.');
  console.log('Evaluation populations, chart values, status, reset, and offline loading checked.');
} finally {
  await browser.close();
}
