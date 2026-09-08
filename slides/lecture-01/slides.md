<!-- .slide: class="title-slide" id="title" -->

<p class="eyebrow">CS40008.01 · Lecture 01</p>

# Tokenization

<p class="subtitle">How does text become model input?</p>

<p class="byline">Baojian Zhou<br>Fudan University · Fall 2026</p>

Note:
Ask students how they think a model represents a Chinese sentence.

---

<!-- .slide: id="representation" -->

## Text representation

A tokenizer maps text to a sequence of token IDs.

The model reads those IDs through an embedding table.

> The tokenizer defines which pieces of text share a vocabulary entry.

Note:
Distinguish a tokenizer from the model that uses it. A token ID has meaning only within its tokenizer's vocabulary.

---

<!-- .slide: id="bytes" -->

## Code points and UTF-8 bytes

<div class="columns">
<div>
<h3>Different measurements</h3>
<p>A Unicode code point identifies a character value.</p>
<p>UTF-8 encodes that value using one or more bytes.</p>
</div>
<div>
<table>
<thead><tr><th>Text</th><th class="num">Code points</th><th class="num">Bytes</th></tr></thead>
<tbody>
<tr><td><code>hello</code></td><td class="num">5</td><td class="num">5</td></tr>
<tr><td><code>你好</code></td><td class="num">2</td><td class="num">6</td></tr>
<tr><td><code>🙂</code></td><td class="num">1</td><td class="num">4</td></tr>
</tbody>
</table>
</div>
</div>

<p class="source">These counts use the exact strings shown. A visible symbol can contain multiple code points.</p>

Note:
Python len counts code points for these strings. JavaScript string.length counts UTF-16 code units, so the live demonstration uses Array.from.

---

<!-- .slide: id="code" -->

## Measuring the same text in Python

```python
examples = ["hello", "你好", "🙂"]

for text in examples:
    code_points = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    print(text, code_points, utf8_bytes)
```

Predict each line before running the cell in the notebook.

Note:
The notebook contains this exact example. Pause before showing the output.

---

<!-- .slide: id="browser-demo" -->

## Exploring the counts

<form class="demo-form" id="utf8-demo">
<label for="demo-text">Try English, Chinese, or emoji</label>
<input id="demo-text" type="text" value="你好🙂" maxlength="48" autocomplete="off" spellcheck="false">
<button type="button" id="demo-reset">Reset example</button>
</form>

<div class="demo-result" aria-live="polite">
<p><strong id="point-count">3</strong> Unicode code points</p>
<p><strong id="byte-count">10</strong> UTF-8 bytes</p>
</div>

<p class="caption">Try a space, a line of digits, or an accented letter. Explain each change.</p>

Note:
This demonstration runs entirely in the browser. Input stays on this page.
For a decomposed accent, compare the precomposed é with e followed by a combining acute accent.

---

<!-- .slide: class="section-slide" id="bpe" -->

<p class="eyebrow">A small algorithm</p>

## Byte pair encoding

The training procedure repeatedly merges a frequent adjacent pair.

Note:
Introduce the distinction between learning merges and applying an existing merge list.
The next example begins with characters to keep the arithmetic visible.

---

<!-- .slide: id="worked-example" -->

## A toy training corpus

| Initial sequence | Frequency |
| :--- | ---: |
| `l o w` | 5 |
| `l o w e r` | 2 |

The pairs `l o` and `o w` each occur 7 times.

<div class="fragment">
<p>Choose <code>l o</code> to break this tie. The next merge combines <code>lo w</code>.</p>
</div>

<p class="caption">Count pairs inside each word. Weight the counts by word frequency.</p>

Note:
This is a toy character-based example with no end-of-word symbol. State these choices explicitly.
Different tie-breaking choices can lead to different learned merges.

---

<!-- .slide: class="exercise" id="exercise-01" -->

<p class="exercise-meta">Exercise E01 · 3 minutes · Notebook E01</p>

## Which pair merges first?

| Initial sequence | Frequency |
| :--- | ---: |
| `a a a b` | 2 |
| `a b` | 1 |

Count every adjacent pair, including overlapping occurrences.

<div class="answer fragment">
<p><code>a a</code> occurs 4 times. <code>a b</code> occurs 3 times.</p>
</div>

Note:
Ask students to explain why a a is counted twice in a a a b.
Pair frequency can include overlaps. Applying the merge left to right replaces non-overlapping occurrences.

---

<!-- .slide: id="results" -->

## Sequence length after two merges

| Stage | Sequence for `low` | Sequence for `lower` | Total tokens |
| :--- | :--- | :--- | ---: |
| Initial | `l o w` | `l o w e r` | 25 |
| Merge `l o` | `lo w` | `lo w e r` | 18 |
| Merge `lo w` | `low` | `low e r` | 11 |

<p class="caption">Totals weight <code>low</code> by 5 and <code>lower</code> by 2.</p>

This example measures sequence length on the training corpus.

Note:
Verify the totals as 5*3+2*5, 5*2+2*4, and 5*1+2*3.
Ask what additional evidence would be needed to assess a tokenizer on unseen text.

---

<!-- .slide: id="interactive-results" -->

## Two merges: 25 → 11 tokens

<div class="plot" data-plotly="assets/sequence-length.json" role="img" aria-label="Toy corpus total tokens decrease from 25 to 18 to 11 after zero, one, and two merges."></div>

<p class="caption">The same toy corpus: <code>low</code> × 5 and <code>lower</code> × 2. Hover for exact counts.</p>

Note:
The chart loads automatically from a local Plotly specification. It plots the checked totals from the previous table, not model benchmark results. Drag to zoom; double-click to reset.

---

<!-- .slide: id="equation" -->

## Average sequence length

For a fixed collection of $N$ texts:

$$
\bar{L} = \frac{1}{N} \sum_{i=1}^{N} \left|\operatorname{encode}(x_i)\right|
$$

Use the same texts when comparing tokenizers.

Report the tokenizer version and any text normalization.

Note:
This is an average per text, so document length affects it. Explain why a comparison must hold the input collection fixed.

---

<!-- .slide: class="references" id="references" -->

## References

- [Jurafsky and Martin, *Words and Tokens*](https://web.stanford.edu/~jurafsky/slp3/2.pdf)<br>Read Sections 2.3–2.4: Unicode and BPE (August 2026 draft).

- <a href="../../papers/sennrich-2016-subword-units.pdf" target="_blank" rel="noopener noreferrer">Sennrich, Haddow, and Birch (2016)</a><br>Read Section 3.2 for the BPE-based subword method.

- <a href="../../papers/kudo-2018-sentencepiece.pdf" target="_blank" rel="noopener noreferrer">Kudo and Richardson (2018), SentencePiece</a><br>Further reading on language-independent tokenization.

- [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html)<br>Review Unicode strings and UTF-8 encoding.

Note:
The notebook is available through the Notebook link at the top of the page.
