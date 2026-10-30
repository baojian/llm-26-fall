import assert from 'node:assert/strict';
import test from 'node:test';
import { frequencies, sinusoidal, rotatePair, rotary, dot, relativeExample } from '../slides/lecture-05-test/position-math.js';
import { relativeExample as noaRelativeExample } from '../slides/lecture-05/position-math.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

test('the sinusoidal worked example uses one frequency per sine/cosine pair', () => {
  assert.deepEqual(frequencies(4), [1, 0.01]);
  assert.deepEqual(sinusoidal(0, 4), [0, 1, 0, 1]);
  sinusoidal(1, 4).forEach((value, i) => close(value,
    [0.8414709848078965, 0.5403023058681398, 0.009999833334166664, 0.9999500004166653][i]));
  assert.throws(() => frequencies(3), RangeError);
});

test('the rotation convention is counterclockwise for column vectors', () => {
  const [x, y] = rotatePair([1, 0], Math.PI / 2);
  close(x, 0);
  close(y, 1);
});

test('RoPE preserves norms and uses the signed offset n minus m', () => {
  const q = [0.3, -0.8, 1.5, 0.2, -1, 2, 0.5, -0.4];
  const k = [-0.2, 0.6, 0.9, -1.2, 0.5, 0.8, -0.1, 2];
  for (const [m, n, shift] of [[3, 1, 3], [0, 0, 12], [82, 37, 500]]) {
    const qm = rotary(q, m);
    close(dot(qm, qm), dot(q, q));
    close(dot(qm, rotary(k, n)), dot(q, rotary(k, n - m)));
    close(dot(qm, rotary(k, n)), dot(rotary(q, m + shift), rotary(k, n + shift)));
  }
});

test('the classroom example changes with gap and stays fixed under a shared shift', () => {
  for (let shift = 0; shift <= 6; shift++) {
    const result = relativeExample(2, shift);
    close(result.score, 0.5);
    close(result.shiftedScore, 0.5);
    assert.ok(result.shiftedKey <= result.shiftedQuery);
  }
  close(relativeExample(3, 3).score, 0);
  close(relativeExample(4, 3).score, -0.5);
});

test('the integrated example defaults to she at 5 and Noa at 0', () => {
  const example = noaRelativeExample();
  assert.deepEqual([example.query, example.key, example.shiftedQuery, example.shiftedKey], [5, 0, 8, 3]);
  close(example.score, -Math.sqrt(3) / 2);
  for (const gap of [5, 6, 7]) {
    for (const shift of [3, 4, 5, 6]) {
      const result = noaRelativeExample(gap, shift);
      close(result.score, Math.cos(gap * Math.PI / 6));
      close(result.shiftedScore, result.score);
    }
  }
});
