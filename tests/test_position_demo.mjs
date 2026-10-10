import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { initialState, computePositionExample, sinusoidalPositions, figureFor, describeState }
  from '../slides/lecture-05/position-demo.js';
import { initializePositionDemo as initialize } from '../slides/lecture-05/demo.js';

const asset = name => new URL('../slides/lecture-05/assets/' + name, import.meta.url);
const fixture = JSON.parse(await readFile(asset('position-values.json'), 'utf8'));
const close = (a, b, tolerance = 1e-12) => assert.ok(Math.abs(a - b) <= tolerance, a + ' != ' + b);

test('unmasked permutation moves outputs with their tokens', () => {
  const result = computePositionExample(fixture, { positions: false, swapped: true });
  assert.equal(result.difference, 0);
  const diagonal = Math.exp(0.5) / (Math.exp(0.5) + 3);
  result.weights.forEach((row, i) => {
    close(row.reduce((sum, value) => sum + value, 0), 1);
    row.forEach((value, j) => close(value, i === j ? diagonal : 1 / (Math.exp(0.5) + 3)));
  });
  assert.deepEqual(result.labels, ['river', 'of', 'the', 'bank']);
});

test('fixed sinusoidal slots break that equivariance', () => {
  const result = computePositionExample(fixture, { positions: true, swapped: true });
  close(result.difference, 1.3650501767846626);
  const original = computePositionExample(fixture, { positions: true, swapped: false });
  assert.equal(original.difference, 0);
  close(original.outputs[0][0], 0.8130890422270126);
  close(result.restored[0][1], -0.258452593886434);
  result.weights.forEach(row => close(row.reduce((sum, value) => sum + value, 0), 1));
});

test('each sinusoidal frequency has a sine and cosine coordinate', () => {
  const pe = sinusoidalPositions(4, 4);
  assert.deepEqual(pe[0], [0, 1, 0, 1]);
  close(pe[1][0], Math.sin(1));
  close(pe[1][1], Math.cos(1));
  close(pe[1][2], Math.sin(0.01));
  close(pe[1][3], Math.cos(0.01));
  pe.forEach(row => {
    close(row[0] ** 2 + row[1] ** 2, 1);
    close(row[2] ** 2 + row[3] ** 2, 1);
  });
  assert.throws(() => sinusoidalPositions(4, 3), /even/);
});

test('the PDF initial figure and accessible description match the computation', async () => {
  const figure = JSON.parse(await readFile(asset('position-demo.json'), 'utf8'));
  assert.deepEqual(figure, figureFor(fixture, initialState));
  assert.match(describeState(fixture, initialState), /Positions off.*0\.0000/);
  assert.match(describeState(fixture, { positions: true, swapped: true }), /1\.3651/);
});

test('reset recovers after a failed chart update', async () => {
  const controls = Object.fromEntries(['position-visual', 'position-toggle', 'position-swap', 'position-reset']
    .map(id => [id, {
      dataset: {}, attributes: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener(name, listener) { this[name] = listener; },
    }]));
  const saved = { fetch: globalThis.fetch, document: globalThis.document, Plotly: globalThis.Plotly };
  const savedError = console.error;
  const errors = [];
  let failNext = false;
  try {
    globalThis.fetch = async () => ({ ok: true, json: async () => fixture });
    globalThis.document = { getElementById: id => controls[id] };
    globalThis.Plotly = { react: async () => {
      if (failNext) { failNext = false; throw new Error('simulated chart failure'); }
    } };
    console.error = error => errors.push(error.message);
    await initialize();
    failNext = true;
    controls['position-toggle'].click();
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(errors, ['simulated chart failure']);
    assert.match(controls['position-visual'].attributes['aria-label'], /could not update/);
    controls['position-reset'].click();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(controls['position-visual'].dataset.positions, 'false');
    assert.equal(controls['position-toggle'].attributes['aria-pressed'], 'false');
    assert.equal(controls['position-visual'].attributes['aria-label'], describeState(fixture, initialState));
    assert.equal(errors.length, 1);
  } finally {
    for (const [name, value] of Object.entries(saved)) {
      if (value === undefined) delete globalThis[name];
      else globalThis[name] = value;
    }
    console.error = savedError;
  }
});
