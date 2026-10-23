import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { matchingPairs, nextTokenDistribution, reviewFigure, sampleNext } from '../slides/lecture-03/ngram-review.js';

const initial = JSON.parse(await readFile(new URL('../slides/lecture-03/assets/ngram-review.json', import.meta.url)));
const model = initial.layout.meta;

test('count all 16 within-sentence pairs and never predict BOS or continue EOS', () => {
  const histories = ['BOS', ...model.vocabulary];
  assert.equal(histories.reduce((total, context) => total + matchingPairs(model, context).length, 0), 16);
  assert.equal(model.vocabulary.includes('BOS'), false);
  assert.deepEqual(matchingPairs(model, 'EOS'), []);
  for (const context of histories) {
    assert.ok(matchingPairs(model, context).every(pair => pair.next !== 'BOS'));
  }
});

test('trace I contexts in corpus order, then normalize counts over all outputs', () => {
  assert.deepEqual(matchingPairs(model, 'I'), [
    { row: 0, index: 1, next: 'am' }, { row: 1, index: 2, next: 'am' },
    { row: 2, index: 1, next: 'do' },
  ]);
  for (const [seen, am, doCount] of [[1, 1, 0], [2, 2, 0], [3, 2, 1]]) {
    const result = nextTokenDistribution(model, 'I', seen);
    assert.equal(result.total, seen);
    assert.equal(result.counts[2], am);
    assert.equal(result.counts[4], doCount);
  }
  assert.deepEqual(nextTokenDistribution(model, 'I').probabilities, [0, 0, 2 / 3, 0, 1 / 3, 0, 0, 0, 0, 0]);
});

test('BOS and Sam reproduce the Lecture 02 estimates; deterministic contexts sum to one', () => {
  assert.deepEqual(nextTokenDistribution(model, 'BOS').probabilities, [0, 2 / 3, 0, 1 / 3, 0, 0, 0, 0, 0, 0]);
  assert.deepEqual(nextTokenDistribution(model, 'Sam').probabilities, [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]);
  for (const context of ['BOS', ...model.vocabulary.filter(token => token !== 'EOS')]) {
    assert.equal(nextTokenDistribution(model, context).probabilities.reduce((a, b) => a + b), 1);
  }
});

test('missing and terminal histories have no learned conditional distribution', () => {
  for (const context of ['EOS', 'unseen']) {
    assert.equal(nextTokenDistribution(model, context).probabilities, null);
    assert.equal(sampleNext(model, context, 0.5), null);
  }
});

test('inverse-CDF sampling respects boundaries and never draws zero-count tokens', () => {
  assert.equal(sampleNext(model, 'I', 0), 'am');
  assert.equal(sampleNext(model, 'I', 2 / 3 - 1e-10), 'am');
  assert.equal(sampleNext(model, 'I', 2 / 3), 'do');
  assert.equal(sampleNext(model, 'I', 1 - Number.EPSILON), 'do');
  assert.equal(sampleNext(model, 'Sam', 0), 'EOS');
  assert.equal(sampleNext(model, 'Sam', 0.5), 'I');
  for (const draw of [-1, 1, NaN]) assert.throws(() => sampleNext(model, 'I', draw), RangeError);
});

test('static print example agrees with calculated values, units, and zero bins', () => {
  assert.deepEqual(initial, reviewFigure(model, { context: 'I', phase: 'probabilities', seen: 0, sampled: null }));
  assert.deepEqual(initial.data[0].y, [2 / 3, 1 / 3]);
  const counted = reviewFigure(model, { context: 'I', phase: 'counts', seen: 3, sampled: null });
  assert.deepEqual(counted.data[0].y, [2, 1]);
  assert.equal(counted.layout.yaxis.title.text, 'Count');
  assert.equal(initial.layout.yaxis.title.text, 'Probability');
});
