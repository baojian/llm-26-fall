/** Small browser demonstrations; all data stay on this page. */

export function buildBpeTrace(corpus, budget) {
  if (!Number.isInteger(budget) || budget < 0) throw new Error('Use a non-negative merge budget.');
  const encoder = new TextEncoder();
  let sequences = corpus.map(([text, frequency]) => {
    if (typeof text !== 'string' || !Number.isInteger(frequency) || frequency <= 0) {
      throw new Error('Each sequence needs text and a positive integer frequency.');
    }
    return { text, frequency, ids: [...encoder.encode(text)] };
  });
  const vocabulary = Array.from({ length: 256 }, (_, id) => [id]);
  const states = [];
  let applied = null;
  for (let step = 0; step <= budget; step++) {
    const counts = new Map();
    for (const { ids, frequency } of sequences) {
      for (let i = 0; i + 1 < ids.length; i++) {
        const pair = [ids[i], ids[i + 1]];
        const key = pair.join(',');
        const previous = counts.get(key);
        counts.set(key, { pair, count: (previous?.count || 0) + frequency });
      }
    }
    const next = [...counts.values()].sort((a, b) =>
      b.count - a.count || a.pair[0] - b.pair[0] || a.pair[1] - b.pair[1])[0] || null;
    const total = sequences.reduce((sum, seq) => sum + seq.ids.length * seq.frequency, 0);
    states.push({ step, total, sequences, next, applied, vocabulary: vocabulary.map(bytes => [...bytes]) });
    if (step === budget || !next) break;
    const newId = vocabulary.length;
    vocabulary.push([...vocabulary[next.pair[0]], ...vocabulary[next.pair[1]]]);
    let replacements = 0;
    sequences = sequences.map(seq => {
      const ids = [];
      for (let i = 0; i < seq.ids.length;) {
        if (seq.ids[i] === next.pair[0] && seq.ids[i + 1] === next.pair[1]) {
          ids.push(newId);
          i += 2;
          replacements += seq.frequency;
        } else {
          ids.push(seq.ids[i]);
          i++;
        }
      }
      return { ...seq, ids };
    });
    applied = { ...next, newId, replacements };
  }
  return states;
}

function initializeBpe() {
  const body = document.getElementById('bpe-corpus');
  if (!body) return;
  const status = document.getElementById('bpe-step');
  const next = document.getElementById('bpe-next');
  const advance = document.getElementById('bpe-advance');
  const decoder = new TextDecoder();
  let trace;
  let index;
  function render() {
    const state = trace[index];
    const piece = id => decoder.decode(new Uint8Array(state.vocabulary[id]));
    body.replaceChildren(...state.sequences.map(seq => {
      const row = document.createElement('tr');
      const label = document.createElement('td');
      label.textContent = `${seq.text} × ${seq.frequency}`;
      const pieces = document.createElement('td');
      const code = document.createElement('code');
      code.textContent = seq.ids.map(piece).join(' · ');
      pieces.append(code);
      row.append(label, pieces);
      return row;
    }));
    status.textContent = index === 0 ? `Initial corpus · ${state.total} tokens`
      : `Merge ${index}: ${state.applied.pair.map(piece).join(' + ')} → ${piece(state.applied.newId)} · ${state.total} tokens`;
    advance.disabled = index === trace.length - 1;
    if (advance.disabled) {
      next.textContent = state.applied
        ? `Pair count ${state.applied.count}; replacements ${state.applied.replacements}. Totals: ${trace.map(s => s.total).join(' → ')}.`
        : 'No adjacent pair remains.';
    } else {
      next.textContent = `Next: ${state.next.pair.map(piece).join(' + ')} · count ${state.next.count}`;
    }
  }
  function reset(overlap = false) {
    trace = buildBpeTrace(overlap ? [['aaab', 2], ['ab', 1]] : [['low', 5], ['lower', 2]], overlap ? 1 : 2);
    index = 0;
    render();
  }
  advance.addEventListener('click', () => {
    if (index + 1 < trace.length) index++;
    render();
  });
  document.getElementById('bpe-reset').addEventListener('click', () => reset());
  document.getElementById('bpe-overlap').addEventListener('click', () => reset(true));
  reset();
}

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
  initializeBpe();
}
