# The same text, six tokenizers

**Question:** What changes when the same text goes through different published
tokenizers, and which part of the pipeline helps explain the difference?

Open the [interactive report](index.html) to choose an example, compare two
pipelines, and inspect token counts across all six tokenizers. The
[measured tables](results.md) provide the same numerical evidence without
JavaScript. This is an instructor-authored exploration for Lecture 01.

## 1. What we measured

We run 32 short, authored examples through `r50k_base`, `cl100k_base`,
`o200k_base`, Qwen3-8B, BERT uncased, and T5-small. The sample covers BPE,
WordPiece, and Unigram. The three tiktoken names identify exact published
encodings; they are not claims about every current commercial model's tokenizer.

The [source manifest](tokenizers.json) records exact artifact revisions, SHA-256
hashes, runtime versions, regex rules, and special-token mappings. The only
downloaded files are tokenizer artifacts. No model weights, paid API, GPU,
or training run is needed; measurements run on a CPU.

The encoding policy is **ordinary text**: no chat template, automatic BOS/EOS
wrappers, padding, or truncation. Literal special-token strings are processed
as ordinary input; other added tokens retain their artifact's matching rules.
These counts therefore describe the displayed text under this policy, rather
than a complete model or API request.

For every case we record the original input, Unicode code-point and UTF-8 byte
counts, normalized text, pre-tokenizer preview, final pieces and IDs, token
count, decoded text, and unknown-token count. We distinguish four outcomes:

| Outcome | Meaning |
| --- | --- |
| Exact | Decoding reproduces the original string exactly. |
| NFC equivalent | The strings differ, but are equal after Unicode NFC normalization. This can change the byte representation without changing the canonical Unicode text. |
| Unknown tokens | Decoding differs and the output IDs contain the tokenizer's unknown-token ID. |
| Text changed | Decoding differs beyond canonical Unicode equivalence, without an unknown-token ID explaining the change. Inspect normalization and decoding to understand it. |

These are diagnostic categories, not quality scores. An exact round trip also
does not show that a language model understands the text.

## 2. Vocabulary entries are not ID capacity

Start with the generated vocabulary table. `cl100k_base` and `o200k_base` have
gaps in their token IDs, so their highest ID plus one exceeds the number of
actual entries. BERT and T5 already include their marked special entries in
their base vocabularies: adding those counts again would double-count them.

Token IDs are local labels. An equal integer in two vocabularies does not
identify the same text. Displayed pieces also use different conventions:
tiktoken byte representations, BERT's `##` continuation marker, T5's `▁` space
marker, and byte-level alphabets. A byte token can contain only part of a UTF-8
character; decode the complete sequence when checking preservation.

In the course abstraction, a vocabulary entry indexes an embedding vector.
The allocated embedding rows and vector width must be checked in the model's
architecture or configuration. This tokenizer-only exploration does not
measure them. The [model table project](https://github.com/baojian/llm-26-fall/issues/154)
can reuse the sourced vocabulary counts while collecting its model metadata
separately.

## 3. Boundaries before and after vocabulary lookup

Choose `digits`, then compare the shorter digit runs. On `item123456789`, the
`r50k_base` pre-tokenizer keeps the digits together, while its learned byte
vocabulary produces several final tokens. `cl100k_base` and `o200k_base`
separate digit runs into chunks of at most three digits. Qwen3's splitting
rule separates individual digits on these inputs.

The one-, two-, three-, and four-digit cases help locate the boundary. The
decimal-point and inserted-space cases then change one character in the
nine-digit example. Inspect both the preview and the final pieces: a chunk
is a region available to the segmentation algorithm, not a promise that the
whole region is one token.

The preview runs normalization and pre-tokenization in isolation. Added-token
matching can happen earlier in the complete tokenizer pipeline, so this is
not a full encoding trace. The final pieces and IDs come from the complete
published tokenizer pipeline under the stated ordinary-text policy.

These observations explain the segmentation of the supplied strings. They
do not establish which associated language model performs arithmetic better.

## 4. Language and script change the representation cost

Choose `mixed-script` and inspect the same Chinese/English string across all
six tokenizers. Several byte-based encodings retain the full string with
different token counts. BERT uncased and T5-small produce unknown tokens on
this case, so their shorter representations cannot be interpreted as equally
faithful compression.

Then compare `chinese`, `mixed-spaced`, `english`, and `arabic`. Adding spaces
around `Python` provides a focused intervention; comparing unrelated English
and Arabic sentences does not isolate the effect of language. The examples
are intentionally small and are not a balanced multilingual benchmark.

UTF-8 bytes per token uses the **original** input size. It is useful when
comparing exact representations of the same input, but can be misleading
when normalization or unknown tokens discard information. It is undefined
for the empty input. We do not average the selected examples into a global
tokenizer ranking.

## 5. Similar-looking strings can have different code points

The `normalization` example is `CAFÉ café`: the first accent belongs to
precomposed `É` (U+00C9), while the last accent is U+0301 after `e`. Compare
`accent-composed` and `accent-decomposed`, then `case-upper` and `case-lower`.

On the combined example, the tiktoken encodings preserve the original bytes.
Qwen3 applies NFC normalization; T5's normalizer also produces the composed
accent on this case. Their decoded strings are canonically equivalent to the
input. BERT lowercases and removes accents, which goes beyond canonical
equivalence. The lowercase unaccented input helps separate these effects.

The `fullwidth` example probes another transformation. Compare its original,
normalized, and decoded strings before generalizing from the accent example:
different normalization pipelines need not make the same transformations.

## 6. Whitespace and unknown tokens matter

Compare `space-single`, `space-leading`, and `space-double`; then inspect the
tab, newline, and Python indentation in `whitespace` and `code`. A lower count
can reflect collapsed layout. That matters when the layout carries information,
as indentation does in Python. `code-flat` is a deliberately changed source
string. This comparison tests preservation of the layout; tokenizer
measurements alone do not test whether two programs behave identically.

The `emoji` case gives a particularly clear counterexample to ranking by
count: BERT emits one unknown token, which decodes as `[UNK]`. The original
emoji sequence cannot be recovered. T5 also emits unknown tokens. The byte
tokenizers in this comparison recover the example even when individual pieces
contain incomplete UTF-8 sequences.

Finally, inspect the literal special-token strings and the empty input. Their
behavior tests the stated encoding policy. It should not be extrapolated to
chat templates or a model wrapper that explicitly inserts special IDs.

## 7. What follows, and what remains untested

For an exactly preserved input, fewer tokens mean fewer sequence positions
under the measured policy. Converting that difference to latency, memory,
training cost, or context capacity requires a model, workload, and compute
measurement. We measured none of those quantities here.

The six artifacts differ in vocabulary, training data, normalization,
pre-tokenization, and segmentation algorithm. This comparison does not isolate
BPE versus WordPiece versus Unigram as a causal factor. The controlled input
variations support local explanations of boundaries and transformations;
broader language-coverage or model-quality claims need a separate study.

The useful conclusion is a way to read tokenizer evidence: inspect the
pipeline, compare the same input, verify what decoding preserves, and then
state the scope of the observation.

## Reproduce or explore another input

Run these commands from the repository root. Install the pinned Python
environment and the existing course page-building dependencies once:

```sh
uv sync --locked --extra tokenizer-comparison
npm ci --prefix slides
uv run --locked --extra tokenizer-comparison python explorations/tokenizer-comparison/run.py fetch
```

Rebuild all measured tables and the interactive report with one command:

```sh
uv run --locked --extra tokenizer-comparison python explorations/tokenizer-comparison/run.py build
```

Artifacts and the full measurement record stay in ignored
`workspace/tokenizer-comparison/`. `results.md` and `index.html` are generated
teaching publications. Do not edit their numerical values by hand. The page
embeds its examples and needs no model or network requests to explore them.
The browser chooses among the measured examples; it does not tokenize arbitrary
new text. Use the CLI for a new input:

```sh
uv run --locked --extra tokenizer-comparison python explorations/tokenizer-comparison/run.py compare --tokenizers bert qwen3 --text 'CAFÉ café'
```

Only `fetch` downloads artifacts; `compare` and `build` use the verified local
cache. A missing or changed artifact fails explicitly. A build check compares
the regenerated publications without rewriting them:

```sh
uv run --locked --extra tokenizer-comparison python explorations/tokenizer-comparison/run.py build --check
TOKENIZER_COMPARISON_INTEGRATION=1 uv run --locked --extra tokenizer-comparison python -m pytest -q tests/test_tokenizer_comparison.py
node explorations/tokenizer-comparison/check.mjs
```

The browser check needs the course's Playwright Chromium installed with
`npm --prefix slides run browser:install`. It checks the local page, including
mobile width, printing, and operation with external requests blocked.

## Sources and scope

- [Pinned manifest](tokenizers.json): exact tokenizer URLs, revisions, hashes,
  and runtime versions for every measurement. The generated vocabulary table
  links each artifact directly.
- [Published tiktoken encoding definitions](https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py):
  regexes, mergeable ranks, and special-token mappings. See the
  [third-party notice](third-party-notices.md).
- [Hugging Face tokenizer pipeline](https://huggingface.co/docs/tokenizers/main/en/pipeline):
  the distinction between normalization, pre-tokenization, tokenization,
  and post-processing. Measurements use the pinned local artifacts rather
  than a moving model alias.
- [Unicode normalization](https://www.unicode.org/reports/tr15/): canonical
  equivalence and NFC. Python's `unicodedata` performs the diagnostic NFC
  comparison; it does not modify inputs sent to the tokenizers.
- [Lecture 01 preprocessing reader](../../docs/lecture-01-pre-tokenization.html)
  and [tokenizer reading guide](../../docs/tokenizer-reading-list.md):
  broader course context and literature. Their separate corpus studies are
  not measurements from these short examples.
