/* Shared language preference for the course, reading pages, and slide tools. */
(() => {
  const valid = value => value === 'zh' || value === 'en';
  const requested = new URLSearchParams(location.search).get('lang');
  let saved = 'en';
  try { saved = localStorage.getItem('course-language') || 'en'; } catch { /* Optional storage. */ }
  let language = valid(requested) ? requested : valid(saved) ? saved : 'en';
  try { localStorage.setItem('course-language', language); } catch { /* Optional storage. */ }

  function translate(root = document) {
    root.querySelectorAll('[data-en][data-zh]').forEach(element => {
      element.textContent = element.dataset[language];
    });
    root.querySelectorAll('[data-en-label][data-zh-label]').forEach(element => {
      element.setAttribute('aria-label', element.dataset[`${language}Label`]);
    });
    document.querySelectorAll('.lang-toggle [data-lang]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.lang === language));
    });
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  }

  function set(value, { updateURL = true } = {}) {
    if (!valid(value)) return;
    const previous = language;
    window.dispatchEvent(new CustomEvent('course:before-language', { detail: { language: value, previous } }));
    language = value;
    try { localStorage.setItem('course-language', language); } catch { /* Optional storage. */ }
    if (updateURL && ['http:', 'https:'].includes(location.protocol)) {
      const url = new URL(location.href);
      url.searchParams.set('lang', language);
      history.replaceState(history.state, '', url);
    }
    translate();
    window.dispatchEvent(new CustomEvent('course:language', { detail: { language, previous } }));
  }

  window.CourseLanguage = { get: () => language, set, translate };
  document.addEventListener('click', event => {
    const button = event.target.closest('.lang-toggle button[data-lang]');
    if (button) set(button.dataset.lang);
  });
  window.addEventListener('storage', event => {
    if (event.key === 'course-language' && valid(event.newValue)) set(event.newValue);
  });
  window.addEventListener('popstate', () => {
    const value = new URLSearchParams(location.search).get('lang');
    if (valid(value)) set(value, { updateURL: false });
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => translate());
  else translate();
})();
