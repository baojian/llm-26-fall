/* Integration check against the running course preview. Does not edit or run cells. */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const slides = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const preview = new URL(process.argv[2] || 'http://127.0.0.1:8000');
assert.ok(['localhost', '127.0.0.1'].includes(preview.hostname));
const output = path.join(slides, '.checks', 'notebook');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
let context;
try {
  context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  let captureLaunch;
  await context.route('**/_course/notebook', async route => {
    const response = await route.fetch({ timeout: 45000 });
    const result = await response.json();
    captureLaunch({ status: response.status(), result });
    await route.fulfill({ response });
  });
  const page = await context.newPage();
  await page.goto(new URL('/slides/example/', preview).href, { waitUntil: 'networkidle' });
  await page.evaluate(() => window.courseReady);
  const link = page.getByRole('link', { name: 'Notebook', exact: true });
  assert.equal(await link.getAttribute('target'), '_blank');
  assert.equal(await link.getAttribute('download'), null);

  async function openNotebook() {
    const pendingTab = context.waitForEvent('page');
    const pendingResult = new Promise(resolve => { captureLaunch = resolve; });
    await link.click();
    const tab = await pendingTab;
    const { status, result } = await pendingResult;
    assert.equal(status, 200, result.error || 'The notebook launcher failed.');
    await tab.waitForURL(url => url.pathname.includes('/lab/workspaces/'), { timeout: 60000 });
    await tab.locator('.jp-Notebook').waitFor({ state: 'visible', timeout: 60000 });
    await tab.getByText('Python 3 (course uv)', { exact: true }).filter({ visible: true }).first().waitFor({ state: 'visible', timeout: 60000 });
    return { tab, result };
  }

  const first = await openNotebook();
  await first.tab.screenshot({ path: path.join(output, 'jupyterlab.png') });
  const second = await openNotebook();
  assert.equal(second.result.reused, true, 'The second click must reuse the running server.');
  const firstURL = new URL(first.result.url);
  const secondURL = new URL(second.result.url);
  assert.equal(secondURL.origin, firstURL.origin, 'Repeated clicks started different servers.');
  assert.notEqual(secondURL.pathname, firstURL.pathname, 'Tabs should have independent Lab workspaces.');
  assert.equal(first.result.notebook, 'workspace/slides/example/practice.ipynb');

  const serverBase = firstURL.origin + firstURL.pathname.split('/lab/workspaces/')[0];
  const specsResponse = await context.request.get(serverBase + '/api/kernelspecs');
  assert.equal(specsResponse.status(), 200);
  const specs = await specsResponse.json();
  const executable = specs.kernelspecs.python3.spec.argv[0];
  assert.equal(path.dirname(executable), path.join(path.dirname(slides), '.venv', process.platform === 'win32' ? 'Scripts' : 'bin'));
  const sessionsResponse = await context.request.get(serverBase + '/api/sessions');
  const sessions = await sessionsResponse.json();
  assert.ok(sessions.some(session => session.path.endsWith(first.result.notebook) && session.kernel.name === 'python3'), 'The notebook must have a live Python kernel session.');
  assert.equal(await second.tab.getByText('Please use a different workspace.').count(), 0);

  // A generic static server shows setup instructions instead of downloading JSON.
  const setup = await context.newPage();
  await setup.route('**/_course/notebook', route => route.fulfill({ status: 404, contentType: 'text/html', body: 'Not found' }));
  await setup.goto(new URL('/slides/shared/notebook.html?lecture=example', preview).href);
  await setup.getByRole('heading', { name: 'JupyterLab is not ready' }).waitFor();
  assert.equal(await setup.locator('#notebook-setup').isVisible(), true);
  await setup.screenshot({ path: path.join(output, 'setup-instructions.png') });
  await writeFile(path.join(output, 'report.json'), JSON.stringify({
    passed: true, firstClickReusedServer: first.result.reused,
    secondClickReusedServer: second.result.reused, notebook: first.result.notebook,
    kernelExecutable: executable, editedOrExecutedCells: false
  }, null, 2) + '\n');
  console.log('Passed: notebook opens in a new tab, repeat clicks reuse JupyterLab, the kernel uses the course environment, and setup instructions handle a static server.');
} catch (error) {
  for (const [index, tab] of (context?.pages() || []).entries()) {
    await tab.screenshot({ path: path.join(output, `failure-${index}.png`) }).catch(() => {});
  }
  // Jupyter authentication tokens must not appear in test output.
  console.error(String(error).replace(/token=[^\s&"'<>]+/g, 'token=[redacted]'));
  process.exitCode = 1;
} finally {
  await browser.close();
}
