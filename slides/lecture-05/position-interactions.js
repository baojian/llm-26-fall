import { rotatePair, relativeExample, toyFrequency } from './position-math.js';

const format = number => (Math.abs(number) < 0.0005 ? 0 : number).toFixed(3);
const setRay = (element, cx, cy, radius, angle) => {
  const [x, y] = rotatePair([radius, 0], angle);
  element.setAttribute('x2', cx + x);
  element.setAttribute('y2', cy - y);
};

export function initializePositions() {
  const rotation = document.getElementById('rope-rotation-visual');
  const slider = document.getElementById('rope-position');
  function updateRotation() {
    const position = Number(slider.value);
    const angle = position * toyFrequency;
    const [x, y] = rotatePair([1, 0], angle);
    setRay(rotation.querySelector('[data-rotation-ray]'), 885, 232, 154, angle);
    const arc = position ? `M 939 232 A 54 54 0 0 0 ${885 + 54 * x} ${232 - 54 * y}` : 'M 939 232';
    rotation.querySelector('[data-rotation-arc]').setAttribute('d', arc);
    rotation.querySelector('[data-rotation-label]').textContent = `p = ${position} · angle = ${position * 30}°`;
    rotation.querySelector('[data-rotation-values]').textContent = `Rotated pair: (${format(x)}, ${format(y)})`;
    document.getElementById('rope-position-value').value = String(position);
    rotation.dataset.position = String(position);
    rotation.dataset.pair = JSON.stringify([x, y]);
    rotation.querySelector('desc').textContent = `A unit vector at position ${position} rotates by ${position * 30} degrees to (${format(x)}, ${format(y)}). Toy frequency: thirty degrees per token. RoPE rotates query and key coordinate pairs and leaves values unchanged.`;
  }
  slider.addEventListener('input', updateRotation);
  updateRotation();

  const relative = document.getElementById('rope-relative-visual');
  const shiftButton = document.getElementById('rope-shift');
  const gapButton = document.getElementById('rope-gap');
  let gap = 5;
  let shift = 3;
  function updateRelative() {
    const result = relativeExample(gap, shift);
    for (const [role, cx, position] of [
      ['query', 278, result.query], ['key', 278, result.key],
      ['shifted-query', 866, result.shiftedQuery], ['shifted-key', 866, result.shiftedKey],
    ]) setRay(relative.querySelector(`[data-relative-ray="${role}"]`), cx, 237, 93, position * toyFrequency);
    relative.querySelector('[data-shift-label]').textContent = `Shift both by s = ${shift}`;
    relative.querySelector('[data-gap-label]').textContent = `m − n = ${gap}`;
    relative.querySelector('[data-relative-positions="original"]').textContent = `m = ${result.query}, n = ${result.key}`;
    relative.querySelector('[data-relative-positions="shifted"]').textContent = `m = ${result.shiftedQuery}, n = ${result.shiftedKey}`;
    relative.querySelector('[data-relative-score="original"]').textContent = `Dot product = ${format(result.score)}`;
    relative.querySelector('[data-relative-score="shifted"]').textContent = `Dot product = ${format(result.shiftedScore)}`;
    relative.dataset.gap = String(gap);
    relative.dataset.shift = String(shift);
    relative.dataset.score = String(result.score);
    relative.dataset.shiftedScore = String(result.shiftedScore);
    shiftButton.disabled = shift >= 6;
    gapButton.disabled = gap >= 7;
    relative.querySelector('desc').textContent = `Fixed query and key vectors (1, 0); thirty degrees per token. Positions (${result.query}, ${result.key}) and (${result.shiftedQuery}, ${result.shiftedKey}) have gap ${gap} and identical dot products ${format(result.score)}. The shared shift is ${shift}.`;
  }
  shiftButton.addEventListener('click', () => { shift = Math.min(6, shift + 1); updateRelative(); });
  gapButton.addEventListener('click', () => { gap = Math.min(7, gap + 1); updateRelative(); });
  updateRelative();
  return {
    reset(section) {
      if (section.id === 'rope-rotation') { slider.value = '5'; updateRotation(); }
      if (section.id === 'rope-relative-offset') { gap = 5; shift = 3; updateRelative(); }
    },
  };
}
