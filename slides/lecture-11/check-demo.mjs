import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const origin = new URL(process.argv[2] || 'http://127.0.0.1:8000').origin;
const manifest = JSON.parse(await readFile(new URL('./assets/source-manifest.json', import.meta.url)));
assert.equal(manifest.mode, 'native', 'Check the editable HTML conversion.');
assert.equal(manifest.slides.length, 69);
const illustrations = manifest.slides.flatMap(slide => slide.illustrations);
const textBoxes = manifest.slides.flatMap(slide => slide.text_boxes);
const originalBoxes = manifest.slides.flatMap(slide => slide.original_text_boxes);
const originalNumbers = originalBoxes.filter(box => box.role === 'slide-number');
assert.equal(illustrations.length, 69);
assert.equal(originalBoxes.length, 180, 'Keep the original text inventory for comparison.');
assert.equal(originalNumbers.length, 69);
assert.equal(textBoxes.length, 113, 'Retain 41 body boxes, 68 revised titles, and four cover boxes.');
assert.ok(originalNumbers.every(original => !textBoxes.some(box => box.id === original.id)),
  'Remove the original numeric footers from every slide.');
const expectedCover = [
  ['cover-course', 'CS40008.01'],
  ['cover-title', 'Diffusion models and diffusion language models'],
  ['cover-subtitle', 'Lecture 11 · NLP and LLMs'],
  ['cover-byline', 'Baojian Zhou\nSchool of Data Science\nFudan University\nNovember 18, 2026'],
];
assert.deepEqual(manifest.slides[0].text_boxes.map(box => [box.id, box.paragraphs.join('\n')]), expectedCover);
for (const illustration of illustrations) {
  const bytes = await readFile(new URL(illustration.asset, import.meta.url));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), illustration.sha256,
    `${illustration.id}: preserve the original embedded image bytes.`);
}
const output = new URL('../.checks/lecture-11/', import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if (response.status() >= 400) errors.push(response.url()); });
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin === origin || url.protocol === 'data:') return route.continue();
    errors.push(`External runtime request: ${url}`);
    return route.abort();
  });
  await page.goto(`${origin}/slides/lecture-11/`, { waitUntil: 'networkidle' });
  await page.evaluate(() => window.courseReady);
  assert.equal(await page.locator('#course-error').isVisible(), false);
  assert.equal(await page.evaluate(() => Reveal.getTotalSlides()), 69);
  assert.equal(await page.locator('.slides .native-text').count(), textBoxes.length);
  assert.equal(await page.locator('.slides .katex').count(), 31);
  assert.equal(await page.locator('.slides .katex-error').count(), 0);
  assert.equal(await page.locator('.slides .native-figure').count(), 69);
  assert.deepEqual(await page.locator('.native-text').evaluateAll(items =>
    items.map(item => item.dataset.sourceTextId)), textBoxes.map(box => box.id));
  assert.deepEqual(await page.locator('.native-figure').evaluateAll(items => items.map(item => ({
    id: item.id,
    count: item.querySelectorAll('img').length,
    asset: item.querySelector('img')?.dataset.gifSource || item.querySelector('img')?.getAttribute('data-src')
      || item.querySelector('img')?.getAttribute('src'),
  }))), illustrations.map(item => ({ id: item.id, count: 1, asset: item.asset })));
  assert.equal(await page.locator('.slides img').count(), 69,
    'Every image is an original separate illustration, not full-slide artwork.');
  assert.equal(await page.locator('.slides [data-background-image], .slides canvas, .slides object, .slides embed').count(), 0);
  assert.equal(await page.locator('.slides svg').evaluateAll(items =>
    items.filter(item => !item.closest('.katex')).length), 0,
  'There must be no outlined full-slide SVG standing in for native text.');
  assert.equal(await page.locator('.slides img[src$=".svg"], .slides img[data-src$=".svg"]').count(), 0);

  for (const [id, expected] of expectedCover) {
    const element = page.locator(`[data-source-text-id="${id}"]`);
    assert.equal(await element.count(), 1);
    assert.equal((await element.innerText()).replace(/\s+/g, ' ').trim(), expected.replace(/\s+/g, ' '));
  }
  assert.doesNotMatch(await page.locator('#source-001').innerText(), /DATA130030|11\/18\/2026/);
  assert.equal(await page.locator('[data-source-text-id]').evaluateAll((items, ids) =>
    items.filter(item => ids.includes(item.dataset.sourceTextId)).length, originalNumbers.map(box => box.id)), 0);

  async function showSlide(index) {
    await page.evaluate(value => Reveal.slide(value, 0, -1), index);
    await page.waitForFunction(id => Reveal.getCurrentSlide().id === id, manifest.slides[index].id);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  }
  for (const [index, slide] of manifest.slides.entries()) {
    await showSlide(index);
    const heading = page.locator(`#${slide.id} h1, #${slide.id} h2`);
    assert.equal(await heading.count(), 1, `${slide.id}: one native heading.`);
    assert.equal(await heading.isVisible(), true, `${slide.id}: the heading must be visible.`);
    assert.equal((await heading.innerText()).replace(/\s+/g, ' ').trim(), slide.title);
    assert.equal(await page.locator(`#${slide.id} .native-text`).evaluateAll((items, number) =>
      items.filter(item => item.textContent.trim() === String(number)).length, slide.number), 0,
    `${slide.id}: no original numeric footer remains in the slide content.`);
  }

  // Select and temporarily edit a visible text node. Separate image elements
  // must remain intact; restore the original node before continuing checks.
  await showSlide(3);
  const editable = page.locator('[data-source-text-id="s004-text6"]');
  const selected = await editable.evaluate(element => {
    const range = document.createRange();
    range.selectNodeContents(element.querySelector('p'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    return { text: selection.toString(), collapsed: selection.isCollapsed };
  });
  assert.equal(selected.text, "Théâtre D'opéra Spatial");
  assert.equal(selected.collapsed, false);
  await page.screenshot({ path: new URL('native-text-selection.png', output).pathname });
  const originalText = await editable.evaluate(element => {
    const node = document.createTreeWalker(element.querySelector('p'), NodeFilter.SHOW_TEXT, {
      acceptNode: node => node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT,
    }).nextNode();
    const original = node.nodeValue;
    node.nodeValue = 'Editable HTML';
    window.getSelection().removeAllRanges();
    return original;
  });
  try {
    assert.equal(await editable.innerText(), 'Editable HTML');
    assert.equal(await editable.isVisible(), true);
    assert.equal(await page.locator('#source-004 .native-figure img').count(), 1);
    assert.equal(await page.locator('#s004-pic2050 img').getAttribute('src'), 'assets/media/image1.jpeg');
    await page.screenshot({ path: new URL('native-text-edit.png', output).pathname });
  } finally {
    await editable.evaluate((element, original) => {
      document.createTreeWalker(element.querySelector('p'), NodeFilter.SHOW_TEXT, {
        acceptNode: node => node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT,
      }).nextNode().nodeValue = original;
    }, originalText);
  }
  assert.equal(await editable.innerText(), selected.text);

  await showSlide(5);
  const initial = page.locator('#s006-pic5');
  const reveal = page.locator('#s006-pic6');
  assert.equal(await initial.isVisible(), true);
  assert.equal(await reveal.evaluate(element => getComputedStyle(element).opacity), '0');
  await page.screenshot({ path: new URL('slide-06-before.png', output).pathname });
  await page.keyboard.press('ArrowRight');
  await page.waitForFunction(() => document.querySelector('#s006-pic6').classList.contains('visible'));
  assert.equal(await page.evaluate(() => Reveal.getCurrentSlide().id), 'source-006');
  await page.waitForFunction(() => getComputedStyle(document.querySelector('#s006-pic6')).opacity === '1');
  await page.screenshot({ path: new URL('slide-06-after.png', output).pathname });
  await page.keyboard.press('ArrowRight');
  await page.waitForFunction(() => Reveal.getCurrentSlide().id === 'source-007');

  const animatedAssets = [];
  for (const [number, count] of [[31, 1], [33, 1], [34, 3]]) {
    await showSlide(number - 1);
    await page.waitForFunction(expected => {
      const images = [...document.querySelectorAll('.native-gif')];
      return images.filter(image => image.getAttribute('src') === image.dataset.gifSource
        && image.complete && image.naturalWidth > 0).length === expected;
    }, count);
    const gifs = page.locator('.present .native-gif');
    assert.equal(await gifs.count(), count);
    const snapshots = await Promise.all(Array.from({ length: count }, (_, index) => gifs.nth(index).screenshot()));
    const advanced = Array(count).fill(false);
    for (let attempt = 0; attempt < 12 && !advanced.every(Boolean); attempt++) {
      await page.waitForTimeout(700);
      for (let index = 0; index < count; index++) {
        if (!advanced[index]) advanced[index] = !snapshots[index].equals(await gifs.nth(index).screenshot());
      }
    }
    assert.ok(advanced.every(Boolean), `Slide ${number}: every original GIF must advance frames.`);
    animatedAssets.push(...await gifs.evaluateAll(images => images.map(image => image.dataset.gifSource)));
    await page.screenshot({ path: new URL(`slide-${number}-animated.png`, output).pathname });
    await page.evaluate(() => Reveal.toggleOverview(true));
    await page.waitForFunction(() => [...document.querySelectorAll('.native-gif')]
      .every(image => image.getAttribute('src') === image.dataset.poster));
    await page.evaluate(() => Reveal.toggleOverview(false));
    await page.waitForFunction(() => [...document.querySelectorAll('.present .native-gif')]
      .every(image => image.getAttribute('src') === image.dataset.gifSource));
  }
  assert.equal(new Set(animatedAssets).size, 5);
  assert.equal(await page.locator('#s044-pic5 a').getAttribute('href'), 'https://aclanthology.org/K16-1002.pdf');
  await page.locator('.course-toolbar a[href="#/references"]').click();
  await page.waitForFunction(() => Reveal.getCurrentSlide().id === 'references');
  const referenceLinks = await page.locator('#references .native-text a[href]').evaluateAll(items =>
    [...new Set(items.map(item => item.href))].sort());
  const expectedReferences = [...new Set(manifest.slides.at(-1).text_boxes.flatMap(box =>
    box.original_paragraphs.flatMap(paragraph => paragraph.match(/https?:\/\/\S+/g) || [])))].sort();
  assert.equal(referenceLinks.length, 6);
  assert.deepEqual(referenceLinks, expectedReferences);
  assert.equal(await page.locator('#notebook-link').isVisible(), false);

  await page.goto(`${origin}/slides/lecture-11/?print-pdf`, { waitUntil: 'networkidle' });
  await page.evaluate(() => window.courseReady);
  await page.waitForFunction(() => document.querySelectorAll('.pdf-page').length === 69);
  assert.equal(await page.locator('.pdf-page').count(), 69);
  assert.equal(await page.locator('.pdf-page .native-text').count(), textBoxes.length);
  assert.equal(await page.locator('.pdf-page .katex').count(), 31);
  assert.equal(await page.locator('.native-gif').count(), 5);
  assert.ok(await page.locator('.native-gif').evaluateAll(images => images.every(image =>
    image.getAttribute('src') === image.dataset.poster && image.complete && image.naturalWidth > 0)),
  'Print uses complete static GIF posters.');
  assert.equal(await page.locator('#s006-pic6').evaluate(element => getComputedStyle(element).opacity), '1',
    'Print retains the final state of the original click reveal.');
  assert.deepEqual(errors, []);
  await writeFile(new URL('native-demo-report.json', output), JSON.stringify({
    slides: 69, nativeTextBoxes: textBoxes.length, originalTextBoxes: originalBoxes.length,
    removedOriginalSlideNumbers: originalNumbers.length, updatedCourseCover: true,
    equations: 31, illustrationInstances: 69,
    uniqueOriginalIllustrations: new Set(illustrations.map(item => item.asset)).size,
    visibleHeadings: 69, selectedAndEditedRealText: true, originalMediaHashes: true,
    animatedAssets, referenceLinks, printPages: 69, printUsesPosters: true, errors,
  }, null, 2) + '\n');
  console.log('Passed: 69 revised titles, course cover, 113 editable text boxes, no original numeric footers, 31 equations, 69 original illustrations, text selection/editing, click reveal, all five GIFs, six references, and 69 print pages.');
} finally {
  await browser.close();
}
