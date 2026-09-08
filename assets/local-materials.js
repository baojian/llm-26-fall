/* Public course links open the student's local server. Keep a custom port
   when the course page itself is served by the local preview. */
(() => {
  if (!['localhost', '127.0.0.1'].includes(location.hostname)) return;
  document.querySelectorAll('a[data-local-path]').forEach(link => {
    link.href = new URL(link.dataset.localPath, location.origin).href;
  });
})();
