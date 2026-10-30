/* Lecture-specific interaction checks against the running local course server. */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const origin = new URL(process.argv[2] || 'http://127.0.0.1:8000');
assert.ok(['127.0.0.1', 'localhost'].includes(origin.hostname));
const output = new URL('../.checks/lecture-06/', import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  await context.route('**/*', route => {
    if (new URL(route.request().url()).origin !== origin.origin) {
      errors.push('Unexpected external runtime request');
      return route.abort();
    }
    return route.continue();
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(new URL('/slides/lecture-06/#/sampling-demo', origin).href, { waitUntil: 'networkidle' });
  await page.evaluate(() => window.courseReady);
  const chart = page.locator('#sampling-plot');
  const status = page.locator('#sampling-status');
  const vector = async () => JSON.parse(await chart.getAttribute('data-probabilities'));
  const initial = await vector();
  initial.forEach((v,i) => assert.ok(Math.abs(v-[4/7,3/7,0,0][i]) < 1e-12));
  await page.getByRole('button', { name: 'Draw token', exact: true }).click();
  const firstDraw = await status.textContent();
  assert.match(firstDraw, /^Draw 1: [AB]\./);
  await page.getByRole('button', { name: 'τ = 1', exact: true }).click();
  await page.getByRole('button', { name: 'τ = 0.7', exact: true }).waitFor();
  assert.notDeepEqual(await vector(), initial);
  await page.getByRole('button', { name: 'p = 0.6', exact: true }).click();
  await page.getByRole('button', { name: 'p = 0.9', exact: true }).waitFor();
  await page.getByRole('button', { name: 'p = 0.9', exact: true }).click();
  await page.getByRole('button', { name: 'p = 1', exact: true }).waitFor();
  assert.ok((await vector()).every(v => v > 0));
  await page.screenshot({ path: fileURLToPath(new URL('sampling-changed.png', output)) });
  await page.getByRole('button', { name: 'Reset seed 29', exact: true }).click();
  await page.getByRole('button', { name: 'τ = 1', exact: true }).waitFor();
  assert.deepEqual(await vector(), initial);
  await page.getByRole('button', { name: 'Draw token', exact: true }).click();
  assert.equal(await status.textContent(), firstDraw);
  await page.getByRole('button', { name: 'Reset seed 29', exact: true }).click();
  await page.screenshot({ path: fileURLToPath(new URL('sampling-reset.png', output)) });
  assert.deepEqual(errors, []);
  await writeFile(new URL('demo-check.json', output), JSON.stringify({
    passed: true, checked: ['initial example', 'temperature', 'top-p', 'all-token case',
      'sample draw', 'seed reset', 'offline runtime'], firstDraw,
  }, null, 2)+'\n');
  console.log('Passed Lecture 06 controls, probabilities, seeded reset, and offline runtime.');
} finally {
  await browser.close();
}
