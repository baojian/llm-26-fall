import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const origin = process.argv[2] || 'http://127.0.0.1:8000';
const manifest = JSON.parse(await readFile(new URL('./assets/source-manifest.json', import.meta.url)));
for (const media of manifest.media) {
  const bytes = await readFile(new URL(media.asset, import.meta.url));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), media.sha256);
}
const output = new URL('../.checks/lecture-11/', import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(response.url()); });
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin === origin || url.protocol === 'data:') return route.continue();
    errors.push(`External runtime request: ${url}`);
    return route.abort();
  });
  await page.goto(`${origin}/slides/lecture-11/#/source-006`);
  await page.evaluate(() => window.courseReady);
  assert.equal(await page.evaluate(() => Reveal.getTotalSlides()), 69);
  const initial = page.locator('.source-initial-diagram');
  assert.equal(await initial.evaluate(el => getComputedStyle(el).opacity), '1');
  await page.screenshot({ path: new URL('slide-06-before.png', output).pathname });
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), 'source-006');
  assert.equal(await initial.evaluate(el => getComputedStyle(el).opacity), '0');
  await page.screenshot({ path: new URL('slide-06-after.png', output).pathname });
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), 'source-007');

  for (const [number, count] of [[31, 1], [33, 1], [34, 3]]) {
    await page.evaluate(index => Reveal.slide(index), number - 1);
    await page.waitForFunction(expected =>
      [...document.querySelectorAll('.source-gif')].filter(el => !el.hidden && el.complete).length === expected, count);
    const gifs = page.locator('.present .source-gif');
    const before = await gifs.first().screenshot();
    let advanced = false;
    for (let attempt = 0; attempt < 10 && !advanced; attempt++) {
      await page.waitForTimeout(700);
      advanced = !before.equals(await gifs.first().screenshot());
    }
    assert.ok(advanced, `Slide ${number}: the original GIF must advance frames.`);
    await page.screenshot({ path: new URL(`slide-${number}-animated.png`, output).pathname });
    await page.evaluate(() => Reveal.toggleOverview(true));
    assert.equal(await page.locator('.source-gif[src]').count(), 0);
    await page.evaluate(() => Reveal.toggleOverview(false));
  }
  const links = await page.locator('.source-link').evaluateAll(items => items.map(el => ({
    slide: el.closest('section').id, uri: el.href, label: el.getAttribute('aria-label'),
  })));
  assert.deepEqual(links, manifest.slides.flatMap(slide => slide.links.map(link => ({
    slide: slide.id, uri: link.uri, label: link.uri,
  }))));
  assert.ok(links.some(link => link.slide === 'source-044' && link.uri === 'https://aclanthology.org/K16-1002.pdf'));
  await page.locator('.course-toolbar a[href="#/references"]').click();
  await page.waitForFunction(() => Reveal.getCurrentSlide().id === 'references');
  assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), 'references');
  assert.equal(await page.locator('#references .source-link').count(), 6);
  assert.equal(await page.locator('#notebook-link').isVisible(), false);

  await page.goto(`${origin}/slides/lecture-11/?print-pdf`);
  await page.evaluate(() => window.courseReady);
  assert.equal(await page.locator('.pdf-page').count(), 69);
  assert.equal(await page.locator('.source-gif:visible,.source-initial-diagram:visible').count(), 0);
  assert.deepEqual(errors, []);
  console.log('Passed: 69 pages, original media hashes, click reveal, GIF playback, source links, references and print posters.');
} finally {
  await browser.close();
}
