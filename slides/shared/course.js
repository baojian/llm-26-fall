/* Shared lecture behavior. Lecture content belongs in each deck's slides.md. */
(() => {
  const base = new URL('../', document.currentScript.src);
  const errorBox = document.getElementById('course-error');
  const help = document.getElementById('course-help');
  document.getElementById('help-button').addEventListener('click', () => help.showModal());
  const notebookLink = document.getElementById('notebook-link');
  const lecture = new URL('./', location.href).pathname.split('/').filter(Boolean).pop();
  notebookLink.href = new URL(`shared/notebook.html?lecture=${encodeURIComponent(lecture)}`, base);
  notebookLink.target = '_blank';
  notebookLink.rel = 'noopener noreferrer';
  notebookLink.removeAttribute('download');
  const languageNote = document.querySelector('.course-language-note');
  function updateLanguage() {
    languageNote.hidden = CourseLanguage.get() !== 'zh';
  }
  window.addEventListener('course:language', updateLanguage);
  updateLanguage();
  window.courseReady = (async () => {
    if (location.protocol === 'file:') {
      throw new Error('Open the published course website, or use the local preview command in slides/README.md.');
    }
    const response = await fetch('lecture.json');
    if (!response.ok) throw new Error('The lecture details could not be loaded.');
    const metadata = await response.json();
    document.title = `${metadata.title} · CS40008.01`;
    document.querySelector('.slides').lang = metadata.language || 'en';
    await Reveal.initialize({
      width: 1280,
      height: 720,
      margin: 0.035,
      center: false,
      hash: true,
      slideNumber: 'c/t',
      transition: 'none',
      backgroundTransition: 'none',
      controlsTutorial: false,
      controls: true,
      progress: true,
      pdfSeparateFragments: false,
      pdfMaxPagesPerSlide: 1,
      plugins: [RevealMarkdown, RevealHighlight, RevealNotes]
    });
    renderMathInElement(Reveal.getSlidesElement(), {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: true,
      strict: 'warn'
    });
    const visuals = await import(new URL('shared/visuals.js', base));
    await visuals.initialize(Reveal);
    if (metadata.demo) {
      const demoURL = new URL(metadata.demo, location.href);
      if (demoURL.origin !== location.origin || !demoURL.pathname.startsWith(new URL('./', location.href).pathname)) {
        throw new Error('The lecture demonstration must be stored in this lecture folder.');
      }
      const demo = await import(demoURL.href);
      await demo.initialize(Reveal);
    }
    Reveal.getSlidesElement().querySelectorAll('a[href]').forEach(link => {
      const href = link.getAttribute('href').trim();
      if (!href || href.startsWith('#')) return;
      if (!['http:', 'https:'].includes(new URL(href, document.baseURI).protocol)) return;
      link.target = '_blank';
      link.relList.add('noopener', 'noreferrer');
    });
    await document.fonts.ready;
    Reveal.layout();
    if (Reveal.isPrintView()) {
      const preview = await import(new URL('shared/print-preview.js', base));
      await preview.initialize(Reveal);
    }
    return metadata;
  })();
  window.courseReady.catch(error => {
    errorBox.hidden = false;
    errorBox.textContent = `The lecture could not start. ${error.message}`;
    console.error(error);
  });
})();
