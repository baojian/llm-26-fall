# A1 · Tokenization and language models

**Released:** Week 2, September 16, 2026 · **Due:** Week 4, September 30, 23:59 (Asia/Shanghai), on eLearning · **Weight:** 15% of the course grade

One assignment for every student, undergraduate or master's, with the same parts and the same points. You will build what Lectures 01 to 03 describe: a byte-level BPE tokenizer, an
n-gram language model, and a neural n-gram language model whose embeddings you
inspect. You measure every model on the same held-out text and argue for one
design choice with your own numbers. Parts 1, 2, and 4 use the Python standard
library; Part 3 uses PyTorch (already in the course environment). The data is
in `data/`.

## What you submit (on eLearning, one zip)

```text
a1/__init__.py    keep this file: it makes a1/ a package the graders can import
a1/bpe.py         Part 1
a1/ngram.py       Part 2
a1/nplm.py        Part 3
a1/evaluate.py    Part 4, plus the results.json it writes
results.json
report.md         the Part 4 table and Part 5 (about 400 words), from the template, with the AI-use disclosure box filled in
```

`manifest.toml` is the submission contract (required files, entry point, results
layout, runtime expectations). Package with the script, which checks the files and
`results.json` before writing the zip:

```sh
python make_submission.py <your-student-id>    # checks the files, writes a1-<id>.zip
```

## Setup

If using the course repository, copy this assignment folder into your ignored
`workspace/` directory before editing. Keep solutions and results out of public
commits; submit them only through eLearning.

Unzip the handout anywhere. It needs Python 3.11 or newer and PyTorch (Part 3
only). Inside the course checkout, run `uv sync --extra tokenization --group dev`;
this installs PyTorch and pytest from the course lockfile. Standalone,
create an environment with `uv venv && uv pip install "torch>=2.6" pytest` or
use any Python with `torch` and `pytest` installed. Then, from the handout
folder:

```sh
uv run python -m pytest tests/      # the public tests
OMP_NUM_THREADS=2 uv run python a1/evaluate.py data/  # full CPU experiment
```

The completed reference experiment took **12 min 00 s and 1.03 GiB peak
process memory** on an Apple M1 laptop with 16 GiB RAM and two PyTorch threads.
An 8 GB RAM laptop should be sufficient; allow at least an hour for the first
full run, with more time for slower hardware or implementations. No GPU is
required. Chinese BPE training accounts for most of the wait; see
[measured runtime and the English/Chinese explanation](runtime.md).

The public tests check shapes and small examples; the graders run a larger
hidden suite that includes everything the public tests check. Expect the whole
assignment to take 12–15 hours over the two weeks; Parts 1, 2, and 4 can be
done from Week 2, Part 3 after Lecture 03 (September 23).

## Data

`data/` contains six fixed UTF-8 text files, bundled with the repository and
student release. No SSH access or full-corpus download is needed. One MB means
1,000,000 bytes; complete documents make each file slightly larger than its target.

| File | Size | Documents | Use |
| --- | ---: | ---: | --- |
| `tinystories-train.txt` | 10 MB | 12,352 | train tokenizer and models |
| `tinystories-dev.txt` | 1 MB | 1,235 | tune settings and monitor neural training |
| `tinystories-test.txt` | 1 MB | 1,276 | final reported evaluation |
| `chinese-web-train.txt` | 10 MB | 2,307 | train the Chinese tokenizer and n-gram models |
| `chinese-web-dev.txt` | 1 MB | 229 | tune Chinese settings; use this fixed file |
| `chinese-web-test.txt` | 1 MB | 232 | final reported evaluation |

The text files total 24 MB. Documents are separated by one blank line; internal
empty lines were removed during preparation. Every student uses the same files.
Do not take another development split from training or tune on test.

TinyStories V2 GPT-4 is sampled from the official training file, with dev/test
from its official validation file. Chinese text comes from 64 sampled shards of
Fineweb-Edu-Chinese V2.1's 4-5 score bucket. These length-filtered teaching samples
support comparisons within a corpus; a score difference between corpora does not
by itself measure language difficulty. See [data provenance and source terms](data/README.md),
[data manifest](data/manifest.json), and `data/SHA256SUMS` for the recipe, exact sizes,
checksums, and source cards. Do not include the supplied data in your submission.

## Part 1 · Tokenizer (25 points) — `a1/bpe.py`

Implement four functions.

```python
def pretokenize(text: str) -> list[str]
def train_bpe(text: str, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
def encode(text: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> list[int]
def decode(ids: list[int], vocab: dict[int, bytes]) -> str
```

- `pretokenize` applies the GPT-2 rules (the Week 1 task `gpt2-pretokenizer`
  is exactly this function; reuse your solution).
- `pretokenize` must put every character of the input into exactly one chunk
  (so the chunks concatenate back to the input); the GPT-2 rules in `re`
  syntax are written out in the starter file's pattern comment.
- `train_bpe` starts from the 256 byte tokens, counts adjacent byte pairs
  inside pre-tokenized chunks (never across chunks), and repeatedly merges the
  most frequent pair until the vocabulary has `vocab_size` entries. **Tie
  break:** among pairs with the same count, merge the lexicographically
  greater pair of byte strings. Return the vocabulary (id → bytes, ids 0–255
  are the single bytes) and the merges in the order they were learned.
- `encode` applies the merges in learned order to each chunk; `decode`
  concatenates the bytes and decodes UTF-8 with `errors="replace"`.
- Property: `decode(encode(x)) == x` for every string `x`.

**Efficiency hint:** count each distinct pre-tokenized chunk once, weighted by
its frequency. Keep an index from each pair to the chunks containing it. After
a merge, subtract the old pair counts and add the new counts for only those
affected chunks; preserve overlapping pair counts and the stated tie break.
A full recount at every merge is a useful small-input correctness baseline,
but can take much longer on the supplied Chinese data. Efficiency is not a
separate graded part; the required inputs, outputs, and points are unchanged.

Train with `vocab_size=1000` on `tinystories-train.txt`. The evaluation script
records bytes per token on the test file; you will need it in Part 4.

## Part 2 · N-gram models (25 points) — `a1/ngram.py`

```python
class NGramLM:
    def __init__(self, order: int, vocab_size: int): ...
    def fit(self, sequences: list[list[int]]) -> "NGramLM"           # counts for orders 1..order, with BOS/EOS
    def prob_mle(self, history: list[int], token: int) -> float       # 0.0 for unseen histories
    def prob_add(self, history: list[int], token: int, delta: float) -> float
    def prob_interp(self, history: list[int], token: int) -> float    # uses self.weights; weights[0] is a uniform floor
    def tune(self, dev_sequences: list[list[int]], iterations: int = 20) -> list[float]   # EM for self.weights
    def log2_prob(self, ids: list[int], kind: str = "interp", delta: float = 1.0) -> tuple[float, int]
    def sample(self, rng: random.Random, max_tokens: int = 60) -> list[int]   # draw from the interpolated model
```

Every smoothed estimator and the floored MLE scoring distribution must sum
to one over the `vocab_size` ids plus EOS for *every* history, including
histories never seen in training. The raw `prob_mle` function returns zero
for unseen histories; the floor is applied by `log2_prob`. For
interpolation that means: an order whose history is unseen contributes the
next lower order's estimate, not zeros (otherwise the mixture loses that
order's λ and no longer sums to one); the uniform floor counts EOS as an
outcome; and `sample` draws from exactly the same mixture that `prob_interp`
scores. The graders check these sums.

Rules: `BOS = -1` and `EOS = -2` are reserved ids; every sequence is padded
with `order-1` BOS and one EOS; the next token ranges over the `vocab_size`
ids plus EOS, so add-δ's denominator uses `vocab_size + 1`; `tune` only ever
sees dev data, and the graders check that the dev log-likelihood does not
decrease across EM iterations; `log2_prob` with `kind="mle"` mixes in a
uniform floor of 10⁻⁶ so that no sequence scores −∞.

## Part 3 · Neural n-gram LM with embeddings (25 points) — `a1/nplm.py`

Bengio et al. (2003), Lecture 02's last section, in PyTorch:

```python
class NeuralNGramLM(nn.Module):
    def __init__(self, vocab_size, context=3, dim=64, hidden=64)   # embedding table, hidden layer, output projection
    def forward(self, contexts) -> logits                           # (batch, context) -> (batch, vocab_size + 2)
def make_examples(sequences, model) -> (contexts, targets)          # every window and its next id, BOS/EOS mapped to the two extra rows
def train(model, sequences, dev_sequences, epochs=3, ...) -> history  # Adam on cross-entropy; train and dev loss per epoch
def log2_prob(model, sequences) -> (total_log2, n_tokens)
def nearest_neighbours(model, token, k=5) -> [(id, cosine)]         # from the embedding table
```

Evaluate development and test examples in minibatches so the full split does not
produce one large vocabulary-logit tensor. Use minibatches
when completing these functions.

Train on the TinyStories BPE ids (the evaluation script does this), then answer
in the report: which five tokens are the nearest neighbours of a few frequent
tokens, and do they make sense? How many parameters does the model have, and
how many does `tie_output=True` remove? Tying shares the output projection's
weight matrix with the embedding table, which is only possible when `hidden`
equals `dim` (the starter uses 64 for both); the count is the Lecture 03
practice question. The lookup, the concatenation, the projection, and the loss
are the same four pieces the Transformer in Week 6 uses.

## Part 4 · Held-out evaluation (10 points) — `a1/evaluate.py`

The starter `a1/evaluate.py` is complete scaffolding except for one function,
`score`, which turns a model's total log₂ probability into token perplexity and
bits per byte. Running `python a1/evaluate.py data/` then writes `results.json`
in the layout shown by `SCHEMA` at the top of the file: for each of
`tinystories` and `chinese`, the `vocab_size`, the `bytes_per_token` on the
test file, and for orders 1, 2, 3 the perplexity and bits per byte under `mle`
(with the floor), `add` (best δ chosen on dev from 0.01, 0.1, 0.5, 1) and
`interp` (λ from `tune`), plus the chosen δ and weights; and for `tinystories`
an `nplm` entry with the epoch history, the parameter counts with and without
weight tying, the test perplexity and bits per byte, and the neighbour lists.
The neural model is trained on TinyStories only. Put the numbers in a table in
`report.md`.

Bits per byte is −log₂ p(test) divided by the number of UTF-8 bytes in the
test documents (each document stripped of surrounding whitespace; the blank
lines between documents are not counted). Its value changes with the tokenizer
and the model, as any score does; what stays comparable is the unit, because
the denominator is the same text for every tokenizer, whereas perplexity is
per token and the token units change with the tokenizer.

## Part 5 · Report (15 points) — `report.md`

Start from the `report.md` template in this folder and replace every `TODO`
(the packager refuses a report with one left). About 400 words, three things:

1. **Samples.** Three stories sampled from your interpolated trigram (`sample`)
   and, if you like, three from the neural model. Two sentences on what the
   samples get right and wrong.
2. **One design choice.** Pick one: the vocabulary size (train a second
   tokenizer with `vocab_size=4000` and compare), the digit rule, δ, the
   interpolation weights, or the embedding dimension. State a claim, support
   it with at least two numbers from `results.json`, and say in one sentence
   what those numbers cannot tell you.
3. **Embeddings.** The neighbour lists from Part 3 and the parameter counts,
   with one sentence on why counts cannot generalize the way embeddings do.

Fill in the AI-use disclosure box at the end of the template (a missing box
costs 5 points; using assistants is allowed, hiding it is not).

## Late submissions

The penalty multiplies your earned assignment score after ordinary grading.
Count each started 24-hour period after September 30, 2026, 23:59
(Asia/Shanghai) as one late day, using the eLearning submission timestamp.

| Late days | Final score |
| --- | --- |
| 0 (on time) | earned score × 100% |
| 1 | earned score × 90% |
| 2 | earned score × 80% |
| 3 | earned score × 70% |
| 4 | earned score × 60% |
| 5 | earned score × 50% |
| 6 or more | 0 |

For example, an earned score of 80 submitted two days late becomes 64.
The same rule applies to every student.

## Rules

- Standard library only in Parts 1, 2, and 4; PyTorch only in Part 3. No
  `tokenizers`, `tiktoken`, `numpy`, or the course's `pipeline/` package in
  submitted code (you may use them to check your own results).
- Your own code and words. Discussing ideas with classmates is fine; sharing
  code is not.
- Tune only on dev. Report only on test.
