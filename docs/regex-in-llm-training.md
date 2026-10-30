# Where regular expressions do real work in LLM training

Reference notes for designing the regular-expression tasks. Five pipeline
stages, each with what the regex decides and a source that states it. Quotes
are verbatim from the source; everything else is a paraphrase to check
against the source. Model families checked: GPT, Llama, Qwen, DeepSeek, GLM,
Kimi.

## 1. Pre-tokenization: the regex that runs before BPE

Byte-pair encoding never sees raw text. A regex first cuts the text into
chunks, and merges are only learned inside a chunk. The pattern decides that
`'s` is its own token, that a space attaches to the following word, whether
Chinese characters and Latin letters can share a token, and how digits are
grouped. Every family below ships such a pattern; they differ in the details.

| Model | Pattern feature | Digits | Source |
| --- | --- | --- | --- |
| GPT-2 (`r50k`) | `'(?:[sdmt]\|ll\|ve\|re)\| ?\p{L}++\| ?\p{N}++\| ?[^\s\p{L}\p{N}]++\|\s++$\|\s+(?!\S)\|\s` | whole run `\p{N}++` | tiktoken `openai_public.py` |
| GPT-4 (`cl100k_base`) | adds case-insensitive contractions, `[^\r\n\p{L}\p{N}]?+\p{L}++`, newline handling | groups of 1–3, `\p{N}{1,3}+` | same file |
| GPT-4o (`o200k_base`) | splits words by letter case (`\p{Lu}...\p{Ll}...`) | 1–3 | same file |
| Qwen (Qwen-7B `tokenization_qwen.py`; Qwen3 keeps the tokenizer, 151,669 vocab) | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` | **one digit at a time**, `\p{N}` | Hugging Face `Qwen/Qwen-7B`; Qwen3 report Section 2 |
| GLM-4 (`tokenization_chatglm.py`) | same shape as Qwen | 1–3, `\p{N}{1,3}` | Hugging Face `THUDM/glm-4-9b-chat` |
| DeepSeek LLM | "Pre-tokenization was employed to prevent the merging of tokens from different character categories such as new lines, punctuation, and Chinese-Japanese-Korean (CJK) symbols, similar to GPT-2." | split into single digits (stated in the paper; not re-quoted here) | arXiv 2401.02954, tokenizer paragraph |
| DeepSeek-V3 | "Byte-level BPE with an extended vocabulary of 128K tokens"; the pretokenizer "introduces tokens that combine punctuations and line breaks", which caused a token-boundary bias on prompts without a final newline, fixed by randomly splitting such tokens during training | as DeepSeek LLM | arXiv 2412.19437, Tokenizer subsection |
| Kimi K2 (`tokenization_kimi.py`) | first alternative is `[\p{Han}]+`: a run of Chinese characters is its own chunk; Latin pieces exclude Han (`&&[^\p{Han}]`) and split by case like `o200k` | 1–3 | Hugging Face `moonshotai/Kimi-K2-Instruct` |
| Kimi K3 | 160K vocabulary, same as K2; the K3 report PDF was not checked for these notes | — | github.com/MoonshotAI/Kimi-K3 |

GLM-5 / GLM-5.2 reports describe data pipelines but not the tokenizer; GLM-5.2
"builds upon the GLM-4.5 data pipeline". Llama 3 uses a tiktoken-style
pattern of the `o200k` family.

Why it matters: this is the exact boundary between Lecture 01 (tokenization)
and Lecture 02 (counting tokens). The same corpus gives different vocabularies,
token counts, and perplexities under Qwen's one-digit rule and GPT-4's
three-digit rule. Stanford CS336 Lecture 1 walks through the GPT-2 pattern.

Sources: <https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py>,
<https://huggingface.co/Qwen/Qwen-7B/raw/main/tokenization_qwen.py>,
<https://huggingface.co/THUDM/glm-4-9b-chat/raw/main/tokenization_chatglm.py>,
<https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/tokenization_kimi.py>,
<https://arxiv.org/abs/2401.02954>, <https://arxiv.org/abs/2412.19437>,
<https://arxiv.org/abs/2505.09388>.

## 2. Web-corpus quality filters (C4, Gopher, Dolma; what changed by 2026)

C4 (Raffel et al., 2020) kept a Common Crawl page only if its lines passed
line-level rules. The `datatrove` library that built FineWeb implements them
in `C4QualityFilter`:

- a line must end in terminal punctuation (`.`, `?`, `!`, `"`, `'`) and not in `...`;
- a line must have at least 3 words; drop lines containing `javascript`;
- drop lines with "terms of use", "privacy policy", "cookie policy", "uses cookies";
- drop the whole document if it contains `lorem ipsum` or a `{` character;
- strip Wikipedia-style citations with the regex `\[\d*\]|\[edit\]|\[citation needed\]`;
- keep documents with at least 5 sentences afterwards.

Gopher (Rae et al., 2021, Appendix A) adds document-level rules of the same
kind: fraction of lines starting with a bullet or ending with an ellipsis,
symbol-to-word ratio for `#` and `...`, fraction of words with an alphabetic
character. Dolma (2024, Section 5.2) "combin[es] heuristics introduced by
Gopher and C4". Dolma also masks emails, IP addresses, and phone numbers with
regexes (Section 5.3), the one place PII removal appears in these notes.

What changed: Qwen3 (2025) annotates "over 30 trillion tokens" with a
model-based labelling system and picks the mixture "at the instance-level";
GLM-5 (2026) uses "another DCLM classifier based on sentence embeddings" and
a "World Knowledge classifier". The regex rules did not disappear: they run
first, cheaply, and the classifiers rank what survives.

Sources: <https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/c4_filters.py>,
<https://arxiv.org/abs/2402.00159>, <https://arxiv.org/abs/2505.09388>,
<https://arxiv.org/abs/2602.15763>.

## 3. Math corpora: keeping LaTeX and finding math pages (OpenWebMath, DeepSeekMath, Qwen2.5-Math)

OpenWebMath (Paster et al., 2023), 14.7B tokens, exists because "existing open
datasets do not faithfully preserve mathematical notation".

- Prefilter (Section 3.3): "Our first filters check for common mathematical
  strings ... such as the presence of tex classes, `<math>` tags, and the
  word 'mathjax'"; otherwise "we search for the presence of the top 100
  most-popular LaTeX symbols in the text".
- Extraction (Section 3.4): "We use regular expressions to search for code
  that calls the configuration function for MathJax to extract the delimiters
  used for equations", so `$...$`, `\(...\)`, `\[...\]` and site-specific
  delimiters are recognised; `<math>` tags with
  `<annotation encoding="application/x-tex">` are read directly; equations in
  image URLs (`latex.codecogs.com`) and `alttext` attributes are recovered.
- Then language identification, a math classifier, and a KenLM perplexity
  filter (Section 3.5).

DeepSeekMath (2024, Section 2.1) starts from OpenWebMath: "we train a fastText
model to recall more OpenWebMath-like mathematical web pages", then groups
Common Crawl by domain, marks domains "where over 10% of the web pages have
been collected" as math-related (e.g. `mathoverflow.net`), and "manually
annotate[s] the URLs associated with mathematical content within these
identified domains" (URL-path patterns such as `mathoverflow.net/questions`).
Decontamination is string matching: "any text segment containing a 10-gram
string that matches exactly with any sub-string from the evaluation benchmarks
is removed". Qwen2.5-Math reuses this recall-by-classifier design.

Sources: <https://arxiv.org/abs/2310.06786>, <https://arxiv.org/abs/2402.03300>.

## 4. Code corpora: file filters and training formats (StarCoder, DeepSeek-V3)

StarCoder (2023, Section 3.1) dropped files by text rules before tokenization:
a "long-line filter (which requires lines to be less than 1,000 characters)",
an "alpha filter that removed files with fewer than 25% alphabetic
characters", an XML filter that "checked for the presence of `<?xml`", an
HTML rule requiring visible text to be at least 20% of the file, and a JSON
size window of 50–5,000 characters. Its PII study found a regex within about
1.5 points of the NER model on emails (Table 8).

DeepSeek-V3 (Section 4.1) packs documents and formats 10% of them for
fill-in-the-middle as `<|fim_begin|>prefix<|fim_hole|>suffix<|fim_end|>middle<|eos_token|>`:
a fixed template that tooling parses back with a pattern. GLM-5 fixed
"metadata alignment issues in Software Heritage code files" and reports
"fuzzily deduplicated unique tokens" for code.

Sources: <https://arxiv.org/abs/2305.06161>, <https://arxiv.org/abs/2412.19437>,
<https://arxiv.org/abs/2602.15763>.

## 5. Answer checking and reward functions (GSM8K, MATH, DeepSeek-R1)

- GSM8K answers end with `#### <number>`; evaluation code extracts the number
  after `####` and compares (<https://github.com/openai/grade-school-math>;
  from memory, not re-checked).
- Minerva-style evaluation on MATH extracts the last `\boxed{...}` from the
  model output (Lewkowycz et al., 2022, <https://arxiv.org/abs/2206.14858>;
  same caveat).
- DeepSeek-R1 (Section 2.2) trains with rule-based rewards: "the model is
  required to provide the final answer in a specified format (e.g., within a
  box), enabling reliable rule-based verification of correctness", plus a
  format reward under which "the model is incentivized to encapsulate its
  reasoning process within designated tags, specifically `<think>` and
  `</think>`". Both are pattern matches on the model's output. Source:
  <https://arxiv.org/abs/2501.12948>.

This connects to the course's GRPO week: the reward that trains the model is
often a regular expression.

## Design rule for every task: think first, then code

A task that is only "make the tests pass" teaches pattern matching against
the checker. Each task therefore has three parts, and the checker verifies
the first two mechanically:

1. **Predict before you run.** The instruction gives three inputs. The
   student writes the expected outputs into `PREDICTIONS` in their file
   before writing `solve`. The checker verifies that `solve` reproduces the
   predictions, so a wrong prediction forces the student to reconcile the
   rule and the code.
2. **Break it.** The student adds two inputs of their own to `MY_CASES`
   where a naive approach fails (for example `"I'm"` for a word splitter, or
   `"3.14"` for the three-digit rule), with the expected outputs. The
   checker verifies that the cases are new (not the given examples) and that
   `solve` passes them.
3. **Explain a choice.** `NOTES` answers one "why" question from the
   instruction in three to five sentences, for example: why does GPT-4 split
   numbers into at most three digits, and what would change if Qwen's
   one-digit rule were used instead? The checker only verifies length; a TA
   reads it when merging.

## Candidate tasks

Each is one function checked by input/output cases, in the task-template
shape. Difficulty is a guess for a student new to regular expressions.

| Task | Function | Modelled on | Think-first question | Difficulty |
| --- | --- | --- | --- | --- |
| GPT-style pre-tokenizer | `solve(text) -> list[str]`: chunk text by the GPT-2 rules (contractions, space + letters, space + digits, punctuation runs, whitespace). Python's `re` has no `\p{L}`; use `[^\W\d_]` | tiktoken `r50k` | Why does the space attach to the following word and not the preceding one? | medium |
| Digit grouping | same interface; digits one at a time (Qwen) or in groups of three (GPT-4), chosen by a flag | Qwen `\p{N}` vs `cl100k` `\p{N}{1,3}` | Which rule gives fewer tokens for a table of years, and which for prices? | easy |
| Han runs | same interface; a run of Chinese characters is its own chunk, never merged with Latin letters | Kimi K2 `[\p{Han}]+` | What could go wrong for mixed sentences like `用GPT写代码` without this rule? | medium |
| C4 line filter | `solve(text) -> str`: keep lines that end in terminal punctuation, have 3+ words, mention no policy phrase; return `""` on `lorem ipsum` or `{` | datatrove `C4QualityFilter` | Which good documents does the `{` rule throw away, and why did C4 accept that? | easy |
| LaTeX span extraction | `solve(html_text) -> list[str]`: LaTeX inside `$...$`, `$$...$$`, `\(...\)`, `\[...\]` in order, ignoring escaped dollars | OpenWebMath Section 3.4 | Why does OpenWebMath read the MathJax configuration instead of assuming `$`? | medium |
| Math-URL recall | `solve(url) -> bool`: is the URL a math page by the DeepSeekMath domain and path rules given in the instruction? | DeepSeekMath Section 2.1 | Why annotate paths within a domain instead of whole domains? | easy |
| Answer extraction | `solve(output) -> str`: the last `\boxed{...}` (nested braces) or the number after `####`; `""` if absent | MATH / GSM8K / DeepSeek-R1 | What can a model do to fool a regex reward, and how would you harden it? | medium |

A natural first pair for Week 2 is the C4 line filter (easy, everyone) and the
GPT-style pre-tokenizer (medium, joins Lecture 01 and Lecture 02). Digit
grouping and Han runs are one-line follow-ups to the pre-tokenizer. LaTeX
extraction, math-URL recall, and answer extraction fit the math and GRPO
weeks.

For the wider question of how tokenizers affect model quality and what their
training data should be, see the [tokenizer reading list](tokenizer-reading-list.md).
