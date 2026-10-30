import assert from 'node:assert/strict';
import { probabilities, seededRandom, draw } from '../slides/lecture-06/sampling.js';

const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);
const base = probabilities();
assert.deepEqual(base.retained, [0, 1]);
base.after.forEach((value, i) => near(value, [4/7,3/7,0,0][i]));
assert.deepEqual(probabilities(1, .01).retained, [0]);
assert.deepEqual(probabilities(1, 1).retained, [0,1,2,3]);
for (const temperature of [.7,1,1.5]) for (const p of [.6,.9,1]) {
  const result = probabilities(temperature, p);
  near(result.before.reduce((a,b)=>a+b,0), 1);
  near(result.after.reduce((a,b)=>a+b,0), 1);
  assert.ok(result.mass + 1e-12 >= p);
  assert.ok(result.before[0] > result.before[1]);
}
for (const args of [[0,1],[-1,1],[NaN,1],[1,0],[1,1.1],[1,NaN]]) {
  assert.throws(()=>probabilities(...args), RangeError);
}
near(draw([.4,.6,0,0],0),0);
near(draw([.4,.6,0,0],.4),1);
near(draw([.4,.6,0,0],.999999),1);
const first = seededRandom(29), reset = seededRandom(29);
assert.deepEqual(Array.from({length:20},()=>draw(base.after,first())),
                 Array.from({length:20},()=>draw(base.after,reset())));
console.log('Lecture 06 sampling checks passed.');
