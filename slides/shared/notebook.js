/* Runs in a new tab so server startup cannot be blocked as an async popup. */
(async () => {
  const params = new URLSearchParams(location.search);
  const lecture = params.get('lecture');
  const notebook = params.get('notebook');
  const title = document.getElementById('notebook-title');
  const status = document.getElementById('notebook-status');
  const setup = document.getElementById('notebook-setup');
  function copy(element, en, zh) {
    element.dataset.en = en;
    element.dataset.zh = zh;
    CourseLanguage.translate();
  }
  function updateTitle() {
    document.title = CourseLanguage.get() === 'zh' ? '打开课程练习本' : 'Open course notebook';
  }
  window.addEventListener('course:language', updateTitle);
  updateTitle();
  if (!/^(example|lecture-[0-9]{2}|[0-9]{2}-[a-z0-9-]+)$/.test(lecture || '')) {
    copy(title, 'Choose a lecture first', '请先选择一讲课件');
    copy(status, 'Open the lecture slides and use their Notebook link.', '打开课件，然后点击其中的“练习本”链接。');
    return;
  }
  document.getElementById('notebook-back').href = `../${lecture}/`;
  if (!['127.0.0.1', 'localhost'].includes(location.hostname)) {
    copy(title, 'Open the notebook locally', '请在本地打开练习本');
    copy(status, 'The course website hosts the slides. Run the local course server to work in JupyterLab.', '课程网站提供课件。请启动本地课程服务器，以便在 JupyterLab 中完成练习。');
    setup.hidden = false;
    return;
  }
  try {
    const response = await fetch('/_course/notebook', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Course-Notebook': '1' },
      body: JSON.stringify({ lecture, notebook })
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
    copy(title, 'JupyterLab is not ready', 'JupyterLab 尚未就绪');
    const messages = {
      'Use the course preview command below to enable JupyterLab.': '请使用下方的课程预览命令启动 JupyterLab。',
      'The notebook could not be opened.': '无法打开练习本。',
      'The notebook launcher did not return a local JupyterLab address.': '练习本启动器未返回本地 JupyterLab 地址。',
    };
    copy(status, error.message, messages[error.message] || `启动失败，服务器详情：${error.message}`);
    setup.hidden = false;
  }
})();
