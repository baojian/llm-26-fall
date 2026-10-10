// Column-vector convention. Pair i uses coordinates 2i and 2i + 1.
export function frequencies(width, base = 10000) {
  if (!Number.isInteger(width) || width < 2 || width % 2) {
    throw new RangeError('Use a positive even dimension.');
  }
  return Array.from({ length: width / 2 }, (_, i) => base ** (-2 * i / width));
}

export function sinusoidal(position, width) {
  return frequencies(width).flatMap(frequency => [
    Math.sin(position * frequency), Math.cos(position * frequency),
  ]);
}

export function rotatePair([x, y], angle) {
  return [x * Math.cos(angle) - y * Math.sin(angle),
    x * Math.sin(angle) + y * Math.cos(angle)];
}

export function rotary(vector, position) {
  return frequencies(vector.length).flatMap((frequency, i) =>
    rotatePair(vector.slice(2 * i, 2 * i + 2), position * frequency));
}

export const dot = (left, right) => left.reduce((sum, value, i) => sum + value * right[i], 0);

// An explicitly labeled teaching frequency, not RoPE's standard first frequency.
export const toyFrequency = Math.PI / 6;

export function relativeExample(gap = 5, shift = 3) {
  const key = 0; // Noa; the default query is she at position 5.
  const query = key + gap;
  const score = (m, n) => dot(rotatePair([1, 0], m * toyFrequency),
    rotatePair([1, 0], n * toyFrequency));
  return {
    gap, shift, query, key, shiftedQuery: query + shift, shiftedKey: key + shift,
    score: score(query, key), shiftedScore: score(query + shift, key + shift),
  };
}
