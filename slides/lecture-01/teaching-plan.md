# Lecture 01: Introduction and Tokenization

September 9, 2026 · CS40008.01 · Three 45-minute teaching periods

## Central question

How does an LLM application turn our text into something a model can predict?

By the end, students should be able to call a local model, distinguish code
points from bytes and tokens, trace BPE training and encoding, and compare
tokenizers on held-out text without confusing compression with model quality.

## Classroom plan

| Period | Minutes within the period | Topic and activity |
| --- | --- | --- |
| 1 | 0–25 | Course overview, app survey, language ambiguity, E01: sentiment with Ollama |
| 1 | 25–45 | Development of NLP and LLMs; next-token prediction and the training pipeline |
| 2 | 0–20 | Text preprocessing, Unicode, UTF-8, E02: count and round-trip |
| 2 | 20–45 | Tokenization choices, BPE training, E03: overlapping pairs and merge counts |
| 3 | 0–20 | Fixed merge order, E04: encode unseen text, boundaries and special tokens |
| 3 | 20–40 | Vocabulary tradeoffs, E05: held-out English and Chinese comparison |
| 3 | 40–45 | Exit questions and readings |

Breaks fall between periods and are outside these 135 teaching minutes. Repeat
the four-topic outline at each section transition, with only the current topic
in bold black. Repeat the tokenization outline after the second break.

The five timed exercises total 25 minutes (5, 3, 4, 5, and 8 minutes).
Students predict first, run the matching notebook section, then explain one
observation. These are ungraded classroom practice. Use the course website for
assessment rules and dates; A1 is released in Week 2.

## Source map

- [Fudan Spring Lecture 01](https://baojian.github.io/llm-26/slides/lecture-01-slides/):
  the four-part outline, natural-language ambiguity, sentiment examples,
  translation and generation demonstrations, and the motivation for subwords.
- [Stanford CS336 Spring 2026, Lecture 1](https://cs336.stanford.edu/lectures/?trace=lecture_01):
  learning by building, resource tradeoffs, the tokenizer interface, character
  and byte baselines, BPE training versus encoding, and compression measurement.
  The local source used for comparison is `lecture_01.py` in the instructor's
  `stanford-cs336-lectures` checkout. Stanford's assessment rules do not apply.
- [Sennrich et al. (2016), Section 3.2](../../papers/sennrich-2016-subword-units.pdf):
  BPE-based subword segmentation. Our implementation begins with all 256 bytes;
  the paper's character vocabulary and word-boundary convention differ.
- [Ollama API](https://docs.ollama.com/api/generate): executable versions of the
  previous browser demos. The notebook uses the REST API from Python's standard
  library, so no additional Python package is required.

## Notebook and preparation

Use `lecture-01-exercise.ipynb` for executable work. The classroom core includes
short text calls, Unicode, and a self-contained byte BPE implementation. The
tokenization work runs without Ollama or network access. Ollama is a separate
application: install it, start it, and fetch `qwen3:0.6b` before class. Model calls
are short, but runtime depends on the classroom machine.

Log probabilities, thinking output, image understanding, and a comparison with
named `tiktoken` encodings are further notebook experiments. They are opt-in and
are not required to complete the 135-minute lesson. Vision requires a suitable
model and a student-provided image; no large model is downloaded by a cell.

The toy corpus is `low` × 5 and `lower` × 2. The first two merges are `l o` and
`lo w`, producing weighted token totals 25, 18, and 11. Pair ties use ascending
token-ID pairs. Count overlapping pairs, but replace non-overlapping occurrences
left to right. Each training string is a separate sequence, so no merge crosses
a document boundary. The notebook deliberately omits production pre-tokenizers
and chat-template processing and explains those limitations.

The launcher preserves existing notebook answers. To receive a revised handout,
rename the existing personal notebook in JupyterLab, then click **Notebook**
again to create a fresh copy. Keep the renamed copy for your previous work.
