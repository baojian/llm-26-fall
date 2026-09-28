import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { computeAttention, describeState, figureFor, initialState } from '../slides/lecture-04/attention-demo.js';

const fixture = JSON.parse(await readFile(new URL('../slides/lecture-04/assets/attention-values.json', import.meta.url), 'utf8'));
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

test('the initial masked example matches the independent hand calculation', () => {
  const { weights, outputs } = computeAttention(fixture, initialState);
  close(weights[1][0], Math.E / (Math.E + 1));
  close(weights[1][1], 1 / (Math.E + 1));
  close(weights[1][2], 0);
  close(outputs[1][0], 0.7310585786300049);
  close(outputs[1][1], 0.5378828427399903);
});

test('changing the future value affects only permitted outputs', () => {
  const before = computeAttention(fixture, initialState);
  const changed = computeAttention(fixture, { ...initialState, changed: true });
  assert.deepEqual(changed.outputs.slice(0, 2), before.outputs.slice(0, 2));
  assert.notDeepEqual(changed.outputs[2], before.outputs[2]);
  const open = computeAttention(fixture, { ...initialState, causal: false });
  const leaked = computeAttention(fixture, { ...initialState, causal: false, changed: true });
  for (let coordinate = 0; coordinate < 2; coordinate++) {
    close(leaked.outputs[1][coordinate] - open.outputs[1][coordinate], 10 * Math.E / (2 * Math.E + 1));
  }
});

test('every row stays normalized in all control states and the fixture is unchanged', () => {
  const original = structuredClone(fixture);
  for (const causal of [false, true]) for (const changed of [false, true]) {
    const result = computeAttention(fixture, { query: 1, causal, changed });
    result.weights.forEach((row, query) => {
      close(row.reduce((sum, value) => sum + value, 0), 1);
      if (causal) row.slice(query + 1).forEach(value => close(value, 0));
    });
  }
  assert.deepEqual(fixture, original);
});

test('the printable initial figure and accessible description match the computation', async () => {
  const stored = JSON.parse(await readFile(new URL('../slides/lecture-04/assets/attention-demo.json', import.meta.url), 'utf8'));
  assert.deepEqual(stored, figureFor(fixture, initialState));
  assert.match(describeState(fixture, initialState), /Query 2.*mask on.*0.7311, 0.5379/);
  assert.match(describeState(fixture, { ...initialState, query: 2, changed: true }), /Query 3.*changed to 13, 11/);
});
