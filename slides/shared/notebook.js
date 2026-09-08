/* Runs in a new tab so server startup cannot be blocked as an async popup. */
(async () => {
  const lecture = new URLSearchParams(location.search).get('lecture');
  const title = document.getElementById('notebook-title');
  const status = document.getElementById('notebook-status');
  const setup = document.getElementById('notebook-setup');
  if (!/^(example|lecture-[0-9]{2}|[0-9]{2}-[a-z0-9-]+)$/.test(lecture || '')) {
    title.textContent = 'Choose a lecture first';
    status.textContent = 'Open the lecture slides and use their Notebook link.';
    return;
  }
  document.getElementById('notebook-back').href = `../${lecture}/`;
  if (!['127.0.0.1', 'localhost'].includes(location.hostname)) {
    title.textContent = 'Open the notebook locally';
    status.textContent = 'The course website hosts the slides. Run the local course server to work in JupyterLab.';
    setup.hidden = false;
    return;
  }
  try {
    const response = await fetch('/_course/notebook', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Course-Notebook': '1' },
      body: JSON.stringify({ lecture })
    });
    if (!response.headers.get('Content-Type')?.includes('application/json')) {
      throw new Error('Use the course preview command below to enable JupyterLab.');
    }
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The notebook could not be opened.');
    const target = new URL(result.url);
    if (!['http:', 'https:'].includes(target.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(target.hostname)) {
      throw new Error('The notebook launcher did not return a local JupyterLab address.');
    }
    location.replace(target.href);
  } catch (error) {
    title.textContent = 'JupyterLab is not ready';
    status.textContent = error.message;
    setup.hidden = false;
  }
})();
