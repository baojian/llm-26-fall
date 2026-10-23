import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { computeAttention, describeState, figureFor, initialState } from '../slides/lecture-05/attention-demo.js';

const fixture = JSON.parse(await readFile(new URL('../slides/lecture-05/assets/attention-values.json', import.meta.url), 'utf8'));
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

test('the initial masked example matches the independent hand calculation', () => {
  const { weights, outputs } = computeAttention(fixture, initialState);
  [4, 1, 1, 2, 1, 2, 0].forEach((numerator, key) => close(weights[5][key], numerator / 11));
  close(outputs[5][0], 1);
  close(outputs[5][1], 8 / 11);
});

test('changing the future value affects only permitted outputs', () => {
  const before = computeAttention(fixture, initialState);
  const changed = computeAttention(fixture, { ...initialState, changed: true });
  assert.deepEqual(changed.outputs.slice(0, 6), before.outputs.slice(0, 6));
  assert.notDeepEqual(changed.outputs[6], before.outputs[6]);
  const open = computeAttention(fixture, { ...initialState, causal: false });
  const leaked = computeAttention(fixture, { ...initialState, causal: false, changed: true });
  for (let coordinate = 0; coordinate < 2; coordinate++) {
    close(leaked.outputs[5][coordinate] - open.outputs[5][coordinate], 15 / 7);
  }
});

test('every row stays normalized in all control states and the fixture is unchanged', () => {
  const original = structuredClone(fixture);
  for (const causal of [false, true]) for (const changed of [false, true]) {
    const result = computeAttention(fixture, { query: 5, causal, changed });
    result.weights.forEach((row, query) => {
      close(row.reduce((sum, value) => sum + value, 0), 1);
      if (causal) row.slice(query + 1).forEach(value => close(value, 0));
    });
  }
  assert.deepEqual(fixture, original);
});

test('the printable initial figure and accessible description match the computation', async () => {
  const stored = JSON.parse(await readFile(new URL('../slides/lecture-05/assets/attention-demo.json', import.meta.url), 'utf8'));
  assert.deepEqual(stored, figureFor(fixture, initialState));
  assert.match(describeState(fixture, initialState), /Query she at position 5.*mask on.*1.0000, 0.7273/);
  assert.match(describeState(fixture, { ...initialState, query: 6, changed: true }), /Query is at position 6.*Value for is at position 6: 13, 11/);
});
