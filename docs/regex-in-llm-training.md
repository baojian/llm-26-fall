# Where regular expressions do real work in LLM training

Reference notes for designing the first task (regular expressions). Each
example names the pipeline stage, what the regex decides, and a source that
states it. Quotes are verbatim from the source; everything else is a
paraphrase to check against the source.

## 1. Pre-tokenization: the regex that runs before BPE

Byte-pair encoding never sees raw text. A regex first cuts the text into
chunks, and merges are only learned inside a chunk. The pattern therefore
decides that `'s` is its own token, that a space attaches to the following
word, and how digits are grouped.

GPT-2 (`r50k`), from OpenAI's `tiktoken` (`tiktoken_ext/openai_public.py`):

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
```

GPT-4 (`cl100k_base`), same file. Note `\p{N}{1,3}`: numbers are split into
groups of at most three digits, so `2026` becomes `202` + `6`:

```text
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s
```

`o200k_base` (GPT-4o) keeps the three-digit rule and adds case-aware word
pieces. Llama 3 and Qwen use tiktoken-style patterns of the same family.

Source: <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>.
Stanford CS336 Lecture 1 walks through the GPT-2 pattern.

Why it matters for the course: this is the exact boundary between Lecture 01
(tokenization) and Lecture 02 (counting tokens). Change the pattern and the
vocabulary, the token counts, and the perplexities all change.

## 2. Web-corpus quality filters (C4, Gopher, Dolma, FineWeb)

C4 (Raffel et al., 2020) kept a Common Crawl page only if its lines passed
line-level rules. The `datatrove` library used to build FineWeb implements
them in `C4QualityFilter`:

- a line must end in terminal punctuation (`.`, `?`, `!`, `"`, `'`) and not in `...`;
- a line must have at least 3 words; drop lines containing `javascript`;
- drop lines with "terms of use", "privacy policy", "cookie policy", "uses cookies";
- drop the whole document if it contains `lorem ipsum` or a `{` character;
- strip Wikipedia-style citations with the regex `\[\d*\]|\[edit\]|\[citation needed\]`;
- keep documents with at least 5 sentences afterwards.

Source: <https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/c4_filters.py>.

Gopher (Rae et al., 2021, Appendix A) adds document-level rules that are
regex-shaped: fraction of lines starting with a bullet, fraction ending with
an ellipsis, symbol-to-word ratio for `#` and `...`, fraction of words
containing an alphabetic character. Dolma (Soldaini et al., 2024, Section 5.2)
states that it combines "heuristics introduced by Gopher and C4", using the
Gopher rules and "a single heuristic from C4 designed to remove paragraphs
that do not end in punctuation".

## 3. PII removal before training (Dolma, StarCoder)

Dolma, Section 5.3: "we rely on carefully-crafted regular expressions that
sacrifice some accuracy for significant speed-up", targeting "three kinds of
PII that are detectable with high precision: email addresses, IP addresses
and phone numbers". Matches are replaced with a special token such as
`|||EMAIL_ADDRESS|||`; documents with many matches are dropped.
Source: <https://arxiv.org/abs/2402.00159>.

StarCoder (Li et al., 2023, Section 4) trained a named-entity model (StarPII)
for names, emails, keys, passwords, IP addresses, and usernames, and compared
it with regexes: on emails the regex reached "96.20% precision, 97.47%
recall" against the model's "97.73% precision, 98.94% recall" (Table 8). The
regex baseline is close on structured PII and much cheaper.
Source: <https://arxiv.org/abs/2305.06161>.

## 4. Code corpus filters (StarCoder, Section 3.1)

Files were dropped by simple text rules before tokenization: a "long-line
filter (which requires lines to be less than 1,000 characters)", an "alpha
filter that removed files with fewer than 25% alphabetic characters", an XML
filter that "checked for the presence of `<?xml`", an HTML rule requiring
visible text to be at least 20% of the file, and a JSON size window of
50–5,000 characters. The Codex paper (Chen et al., 2021) used the same kind
of rules (auto-generated files, average line length, alphanumeric fraction).

## 5. Math corpora: keeping LaTeX through HTML extraction (OpenWebMath)

OpenWebMath (Paster et al., 2023) is 14.7B tokens of mathematical web pages
whose whole point is that "existing open datasets do not faithfully preserve
mathematical notation".

- Prefilter (Section 3.3): "Our first filters check for common mathematical
  strings ... such as the presence of tex classes, `<math>` tags, and the
  word 'mathjax'"; if none match, "we search for the presence of the top 100
  most-popular LaTeX symbols in the text".
- Extraction (Section 3.4): "We use regular expressions to search for code
  that calls the configuration function for MathJax to extract the delimiters
  used for equations", so `$...$`, `\(...\)`, `\[...\]` and site-specific
  delimiters are recognised; `<math>` tags with
  `<annotation encoding="application/x-tex">` are read directly; equations
  encoded in image URLs (for example `latex.codecogs.com`) and in `alttext`
  attributes are recovered.
- Then language identification, a math classifier, and a KenLM perplexity
  filter (Section 3.5).

Source: <https://arxiv.org/abs/2310.06786>. Proof-Pile-2 (Llemma) and later
math corpora reuse this extraction.

## 6. Answer checking and reward functions (Minerva, GSM8K, DeepSeek-R1)

- GSM8K answers end with `#### <number>`; evaluation code extracts the number
  after `####` with a regex and compares it with the model's extracted number
  (<https://github.com/openai/grade-school-math>; stated from memory, not
  re-checked for these notes).
- Minerva-style evaluation on MATH extracts the last `\boxed{...}` from the
  model output before comparing (Lewkowycz et al., 2022,
  <https://arxiv.org/abs/2206.14858>; same caveat).
- DeepSeek-R1 (Section 2.2) uses rule-based rewards during reinforcement
  learning: "the model is required to provide the final answer in a specified
  format (e.g., within a box), enabling reliable rule-based verification of
  correctness", plus a format reward under which "the model is incentivized
  to encapsulate its reasoning process within designated tags, specifically
  `<think>` and `</think>`". Both checks are regex matches on the output.
  Source: <https://arxiv.org/abs/2501.12948>.

This connects to the course's GRPO week: the reward that trains the model is
often a regular expression.

## Candidate first tasks

Each is one function checked by input/output cases, the shape the task
template expects. Difficulty is a guess for a student who has not used
regular expressions before.

| Task | Function | Modelled on | Difficulty |
| --- | --- | --- | --- |
| GPT-style pre-tokenizer | `solve(text) -> list[str]`: split text into chunks following the GPT-2 rules (contractions, space + letters, space + digits, punctuation runs, whitespace). Python's `re` has no `\p{L}`; use `[^\W\d_]` | tiktoken `r50k` pattern | medium |
| Three-digit numbers | `solve(text) -> list[str]`: same as above but digits in groups of at most three, as in `cl100k_base` | tiktoken `cl100k_base` | easy add-on |
| C4 line filter | `solve(text) -> str`: keep only lines that end in terminal punctuation, have 3+ words, and mention no policy phrases; drop the document (return `""`) on `lorem ipsum` or `{` | datatrove `C4QualityFilter` | easy |
| PII masking | `solve(text) -> str`: replace emails, IPv4 addresses, and phone numbers with `|||EMAIL_ADDRESS|||`, `|||IP_ADDRESS|||`, `|||PHONE_NUMBER|||` | Dolma Section 5.3 | easy–medium |
| LaTeX span extraction | `solve(html_text) -> list[str]`: return the LaTeX inside `$...$`, `$$...$$`, `\(...\)`, `\[...\]` in order, ignoring escaped dollars | OpenWebMath Section 3.4 | medium |
| Answer extraction | `solve(output) -> str`: return the last `\boxed{...}` content (with nested braces) or the number after `####`; `""` if absent | Minerva / GSM8K / DeepSeek-R1 | medium |

A natural first pair for Week 2 is the C4 line filter (easy, everyone) and the
GPT-style pre-tokenizer (medium, connects both lectures). The LaTeX and
answer-extraction tasks fit the math and reasoning weeks later.
