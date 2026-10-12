/* Pure probability helpers for the four-token browser teaching example. */
export function probabilities(temperature = 1, topP = 0.6) {
  if (!Number.isFinite(temperature) || temperature <= 0 || !Number.isFinite(topP) || topP <= 0 || topP > 1) {
    throw new RangeError('Require temperature > 0 and 0 < top-p <= 1.');
  }
  const logits = [0.4, 0.3, 0.2, 0.1].map(Math.log);
  const largest = Math.max(...logits);
  const weights = logits.map(value => Math.exp((value - largest) / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  const before = weights.map(value => value / total);
  const order = before.map((_, index) => index).sort((a, b) => before[b] - before[a] || a - b);
  const retained = [];
  let mass = 0;
  for (const index of order) {
    retained.push(index);
    mass += before[index];
    if (topP < 1 && mass >= topP) break;
  }
  const after = before.map((value, index) => retained.includes(index) ? value / mass : 0);
  return { before, after, retained, mass };
}

export function seededRandom(seed) {
  // Mulberry32: a small deterministic teaching RNG, distinct from PyTorch's RNG.
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

export function draw(probabilities, randomValue) {
  if (!(randomValue >= 0 && randomValue < 1)) throw new RangeError('Draw must be in [0,1).');
  let mass = 0;
  for (let index = 0; index < probabilities.length; index++) {
    mass += probabilities[index];
    if (randomValue < mass) return index;
  }
  return probabilities.findLastIndex(value => value > 0);
}
