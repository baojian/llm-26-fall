/* Screen zoom for the fixed-size PDF layout; printed page sizes stay unchanged. */
export async function initialize(reveal) {
  const stylesheet = document.createElement('link');
  stylesheet.rel = 'stylesheet';
  stylesheet.href = new URL('./print-preview.css', import.meta.url).href;
  await new Promise((resolve, reject) => {
    stylesheet.onload = resolve;
    stylesheet.onerror = () => reject(new Error('The print preview styles could not be loaded.'));
    document.head.append(stylesheet);
  });
  if (!document.querySelector('.pdf-page')) {
    await new Promise(resolve => {
      const ready = () => { reveal.off('pdf-ready', ready); resolve(); };
      reveal.on('pdf-ready', ready);
    });
  }

  const pageWidth = parseFloat(document.body.style.width);
  const pageHeight = parseFloat(document.querySelector('.pdf-page').style.height);
  const root = document.documentElement;
  root.style.setProperty('--print-page-width', `${pageWidth}px`);
  root.classList.add('course-print-preview');

  const toolbar = document.createElement('nav');
  toolbar.className = 'print-preview-toolbar';
  toolbar.setAttribute('aria-label', 'Print preview zoom');
  const label = document.createElement('span');
  label.textContent = 'Preview';
  toolbar.append(label);
  const output = document.createElement('output');
  output.setAttribute('aria-label', 'Preview zoom');
  output.setAttribute('aria-live', 'polite');
  let scale = 1;
  let zoomOut;
  let zoomIn;
  function setScale(value) {
    const previous = scale;
    scale = Math.max(0.25, Math.min(2, value));
    root.style.setProperty('--print-preview-scale', scale);
    output.value = `${Math.round(scale * 100)}%`;
    zoomOut.disabled = scale <= 0.25;
    zoomIn.disabled = scale >= 2;
    window.scrollTo(window.scrollX * scale / previous, window.scrollY * scale / previous);
  }
  function button(text, name, action) {
    const control = document.createElement('button');
    control.type = 'button';
    control.textContent = text;
    control.setAttribute('aria-label', name);
    control.addEventListener('click', action);
    toolbar.append(control);
    return control;
  }
  zoomOut = button('−', 'Zoom out', () => setScale(scale - 0.1));
  toolbar.append(output);
  zoomIn = button('+', 'Zoom in', () => setScale(scale + 0.1));
  button('100%', 'Actual size', () => setScale(1));
  const fit = () => setScale(Math.min(1, (window.innerWidth - 32) / pageWidth, (window.innerHeight - 100) / pageHeight));
  button('Fit page', 'Fit page', fit);
  document.body.append(toolbar);
  fit();
}
