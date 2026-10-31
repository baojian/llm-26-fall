/* Inspect the embedded observations; no downloads or browser-side tokenization. */
(() => {
  const report = JSON.parse(document.getElementById('measurements-data').textContent);
  const names = Object.keys(report.tokenizers);
  const select = document.getElementById('case');
  const left = document.getElementById('left');
  const right = document.getElementById('right');
  const labels = { exact: 'Exact', 'nfc-equivalent': 'NFC equivalent', 'unknown-tokens': 'Unknown tokens', 'text-changed': 'Text changed' };
  const params = new URLSearchParams(location.search);
  select.value = report.cases.some(c => c.id === params.get('case')) ? params.get('case') : 'digits';
  left.value = names.includes(params.get('left')) ? params.get('left') : 'cl100k_base';
  right.value = names.includes(params.get('right')) ? params.get('right') : 'qwen3';
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function field(parent, label, value) {
    parent.append(element('dt', label), element('dd', value));
  }
  function render() {
    const current = report.cases.find(c => c.id === select.value);
    document.getElementById('question').textContent = current.question;
    document.getElementById('input').textContent = JSON.stringify(current.text);
    const sample = current.observed[names[0]];
    document.getElementById('input-size').textContent = `${sample.input_codepoints} Unicode code points · ${sample.input_utf8_bytes} UTF-8 bytes`;
    const counts = document.getElementById('counts');
    counts.replaceChildren();
    const maximum = Math.max(1, ...names.map(name => current.observed[name].token_count));
    for (const name of names) {
      const observed = current.observed[name];
      const row = element('div', undefined, 'count-row');
      const track = element('div', undefined, 'bar-track');
      const bar = element('div', undefined, `bar ${observed.fidelity}`);
      bar.style.width = `${100 * observed.token_count / maximum}%`;
      track.setAttribute('aria-hidden', 'true');
      track.append(bar);
      row.append(element('span', name, 'count-name'), track,
        element('strong', String(observed.token_count)), element('span', labels[observed.fidelity], `status ${observed.fidelity}`));
      counts.append(row);
    }
    const pipelines = document.getElementById('pipelines');
    pipelines.replaceChildren();
    for (const name of [left.value, right.value]) {
      const observed = current.observed[name];
      const card = element('section', undefined, 'pipeline');
      card.append(element('h3', name), element('p', `${report.tokenizers[name].algorithm} · ${observed.token_count} ${observed.token_count === 1 ? 'token' : 'tokens'} · ${labels[observed.fidelity]}`, 'pipeline-meta'));
      const details = element('dl');
      field(details, '1 · Normalized text', JSON.stringify(observed.normalized_text));
      field(details, '2 · Pre-tokenizer preview', JSON.stringify(observed.pre_tokenizer_preview));
      field(details, '3 · Final pieces', JSON.stringify(observed.token_pieces));
      field(details, 'Token IDs', JSON.stringify(observed.token_ids));
      field(details, 'Piece notation', observed.piece_format);
      field(details, '4 · Decoded text', JSON.stringify(observed.decoded_text));
      field(details, 'Unknown-token IDs emitted', String(observed.unknown_token_count));
      field(details, 'Original input bytes per token', observed.input_bytes_per_token === null ? 'Undefined (zero tokens)' : observed.input_bytes_per_token.toFixed(2));
      card.append(details);
      const rules = element('details');
      rules.append(element('summary', 'Inspect the published normalization and splitting rules'),
        element('pre', JSON.stringify({ normalizer: report.tokenizers[name].normalizer,
          pre_tokenizer: report.tokenizers[name].pre_tokenizer }, null, 2)));
      card.append(rules);
      pipelines.append(card);
    }
    if (['http:', 'https:'].includes(location.protocol)) {
      const url = new URL(location.href);
      url.searchParams.set('case', select.value);
      url.searchParams.set('left', left.value);
      url.searchParams.set('right', right.value);
      history.replaceState(history.state, '', url);
    }
  }
  [select, left, right].forEach(node => node.addEventListener('change', render));
  document.getElementById('print').addEventListener('click', () => window.print());
  render();
})();
