import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const origin = process.argv[2] || 'http://127.0.0.1:8000';
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`${origin}/slides/lecture-07/#/budget-demo`);
  await page.evaluate(() => window.courseReady);
  for (const [multiple, parameters, tokens, loss] of [
    [1, 100, 2, '3.100'], [4, 200, 4, '2.631'], [16, 400, 8, '2.300'],
  ]) {
    if (multiple !== 1) await page.locator('#budget-button').click();
    await page.waitForFunction(value =>
      document.querySelector('#budget-plot').dataset.computeMultiple === String(value), multiple);
    assert.equal(await page.locator('#budget-status').textContent(),
      `Synthetic optimum: ${parameters}M parameters, ${tokens}B tokens; loss ${loss}.`);
    const optimum = await page.locator('#budget-plot').evaluate(plot => plot.data[1].x[0]);
    assert.equal(optimum, parameters);
  }
  await page.locator('#reset-button').click();
  await page.waitForFunction(() =>
    document.querySelector('#budget-plot').dataset.computeMultiple === '1');
  assert.match(await page.locator('#budget-status').textContent(), /100M parameters, 2B tokens/);
  assert.deepEqual(errors, []);
  console.log('Budget cycle, chart values, status, reset, and page errors checked.');
} finally {
  await browser.close();
}
