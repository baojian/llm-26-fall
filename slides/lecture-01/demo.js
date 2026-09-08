export function initialize() {
  const form = document.getElementById('utf8-demo');
  const input = document.getElementById('demo-text');
  const points = document.getElementById('point-count');
  const bytes = document.getElementById('byte-count');
  function update() {
    points.textContent = String(Array.from(input.value).length);
    bytes.textContent = String(new TextEncoder().encode(input.value).length);
  }
  form.addEventListener('submit', event => event.preventDefault());
  input.addEventListener('input', update);
  document.getElementById('demo-reset').addEventListener('click', () => {
    input.value = '你好🙂';
    update();
  });
  update();
}
