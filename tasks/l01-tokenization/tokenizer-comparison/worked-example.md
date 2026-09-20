# Worked example: the same text, six tokenizers

These are selected teaching examples from the pinned artifacts in
[tokenizers.json](tokenizers.json), measured on September 17, 2026 with
`tiktoken 0.14.0`, `tokenizers 0.23.2`, and `regex 2026.9.3`.
[worked-example.json](worked-example.json) records the selected measurements
and source provenance for the integration check. The full local report also
contains IDs, pieces, normalization, and pre-tokenizer previews for all ten
supplied inputs.

Run the `fetch` and `compare` commands in [the task instructions](instruction.md)
to reproduce the report. The policy is ordinary text with no chat template,
automatic special-token wrappers, padding, or truncation.

## Vocabulary size is a count of entries

| Tokenizer | Algorithm | Actual vocabulary entries | Highest token ID + 1 | Sources and notes |
| --- | --- | ---: | ---: | --- |
| `r50k_base` | BPE | 50,257 | 50,257 | [Encoding definitions](https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py); 50,256 mergeable entries plus one special entry |
| `cl100k_base` | BPE | 100,261 | 100,277 | [Encoding definitions](https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py); 100,256 mergeable entries plus five special entries; IDs have gaps |
| `o200k_base` | BPE | 200,000 | 200,019 | [Encoding definitions](https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py); 199,998 mergeable entries plus two special entries; IDs have gaps |
| `qwen3` | BPE | 151,669 | 151,669 | [Pinned tokenizer](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json); 151,643 base entries and 26 additional entries, of which 14 are marked special |
| `bert` | WordPiece | 30,522 | 30,522 | [Pinned tokenizer](https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json); its five marked special entries are already counted in the base vocabulary |
| `t5` | Unigram | 32,100 | 32,100 | [Pinned tokenizer](https://huggingface.co/google-t5/t5-small/blob/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/tokenizer.json); its 103 marked special entries are already counted in the base vocabulary |

In the course abstraction, each token in $V$ has an embedding vector of
dimension $d$. Real implementations can reserve token IDs or allocate extra
embedding rows. The highest ID, actual entry count, and allocated rows should
therefore be recorded separately. This comparison does not measure embedding
dimension, context window, or a model's release date; those require separate
model sources in [the companion table project](https://github.com/baojian/llm-26-fall/issues/154).

## A pre-tokenizer chunk can become several final tokens

Input: `item123456789`.

| Tokenizer | Pre-tokenizer preview | Final pieces | Tokens |
| --- | --- | --- | ---: |
| `r50k_base` | `item`, `123456789` | `item`, `123`, `45`, `67`, `89` | 5 |
| `cl100k_base` | `item`, `123`, `456`, `789` | `item`, `123`, `456`, `789` | 4 |
| `o200k_base` | `item`, `123`, `456`, `789` | `item`, `123`, `456`, `789` | 4 |
| `qwen3` | `item`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` | same pieces on this input | 10 |
| `bert` | `item123456789` | `item`, `##12`, `##34`, `##56`, `##7`, `##8`, `##9` | 7 |
| `t5` | `▁item123456789` | `▁item`, `123`, `45`, `67`, `89` | 5 |

The ASCII byte pieces for tiktoken are shown as text here for readability.
The complete report uses explicit byte representations. BERT's `##` marks
continuations; T5's `▁` is its whitespace marker.

For `r50k_base`, pre-tokenization keeps the digit run together, but the learned
vocabulary still splits it into several tokens. Qwen's pre-tokenizer separates
individual digits, preventing merges across those boundaries. All six
decoders recover this input exactly. These observations explain segmentation;
they do not tell us which associated model is better at arithmetic.

## Visually similar text may not round-trip exactly

Input: `CAFÉ café`. The first accent is part of `É` (U+00C9); the final accent
is a separate U+0301 following `e`. An unambiguous representation is
`"CAF\u00c9 cafe\u0301"`.

| Tokenizer | Tokens | Decoded text, with Unicode escapes | Exact round trip? |
| --- | ---: | --- | --- |
| `r50k_base` | 6 | `"CAF\u00c9 cafe\u0301"` | yes |
| `cl100k_base` | 4 | `"CAF\u00c9 cafe\u0301"` | yes |
| `o200k_base` | 4 | `"CAF\u00c9 cafe\u0301"` | yes |
| `qwen3` | 3 | `"CAF\u00c9 caf\u00e9"` | no |
| `bert` | 2 | `"cafe cafe"` | no |
| `t5` | 4 | `"CAF\u00c9 caf\u00e9"` | no |

Qwen's NFC normalizer combines the final accent with `e`; T5's published
normalizer produces the same text on this case. BERT lowercases and removes
the accents. The tiktoken encodings retain the original UTF-8 bytes. A student
could change just the capitalization or accent representation to test which
transformation explains each result.

## One token can mean lost information

For the emoji input `👩🏽‍💻🙂`, the counts are 12 (`r50k_base`), 12
(`cl100k_base`), 8 (`o200k_base`), 6 (`qwen3`), 1 (`bert`), and 4 (`t5`).
BERT decodes its one token as `[UNK]`; T5 decodes to `<unk> <unk>`. The other
four recover the input exactly. This is why a token-count chart should be
read alongside decoding and coverage, rather than used alone to rank
tokenizers.

Before contributing, make your own prediction and add a short input that
tests your explanation. A peer reviewer should be able to reproduce the
observation and distinguish the measured fact from the proposed explanation.

## Verify the teaching example

After fetching the artifacts:

```sh
TOKENIZER_COMPARISON_INTEGRATION=1 uv run --locked --extra tokenizer-comparison python -m pytest -q tests/test_tokenizer_comparison.py
```

The normal test suite uses small local fixtures and needs no downloads.
This additional check verifies the published vocabulary counts and selected
examples against all six cached artifacts, with network requests blocked
inside the test.
