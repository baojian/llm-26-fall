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
| 1 | 0–25 | Course introduction (7 min), language and multimodal example gallery (13 min), E01: sentiment with Ollama (5 min) |
| 1 | 25–45 | Ten development slides: language difficulties and history (13 min); next-token prediction, building a model, and resource choices (7 min) |
| 2 | 0–20 | Text preprocessing, Unicode, UTF-8, E02: count and round-trip |
| 2 | 20–45 | Tokenization choices, BPE training, E03: overlapping pairs and merge counts |
| 3 | 0–20 | Fixed merge order, E04: encode unseen text, boundaries and special tokens |
| 3 | 20–40 | Vocabulary tradeoffs, E05: held-out English and Chinese comparison |
| 3 | 40–45 | Exit questions and readings |

Breaks fall between periods and are outside these 135 teaching minutes. Repeat
the four-topic outline at each section transition, with only the current topic
in bold black. Repeat the tokenization outline after the second break.

Open with the title, **About me**, and the first four-topic outline, in that
order. Continue with **NLP, LLMs, and this course**, then the course logic and
coursework to establish the field, the model family, and the course's purpose
and progression. Continue with students'
applications and the example gallery. Place **From examples to today’s goals**
at the end of Course Overview, immediately before the development outline, to
connect observed model behavior to the technical lesson.

Prepare one short Qwen response before class; keep the context example immediately before E01. Show both full
translation inputs, the article-authorship poll, and the two bundled vision
images. Play about ten seconds of each recorded video manually. Further live
translation, article, vision, and image generation are optional; do not wait for
model downloads or long inference during the 13-minute gallery.

Development of NLP & LLMs has ten content slides, following its outline:
ambiguity; messy text and reasoning; Weaver and Turing; the Turing test;
rules to learned representations; Transformers and pretraining; the 2019–2024
LLM timeline; next-token prediction; the model-building pipeline; and resource
budgets. The historical examples motivate the final three slides and the next
section's question of how to represent text.

The five timed exercises total 25 minutes (5, 3, 4, 5, and 8 minutes).
Students predict first, run the matching notebook section, then explain one
observation. The gallery's numbered tasks, authorship poll, and optional model
demonstrations add no timed exercises beyond E01–E05. These are ungraded
classroom practice. Use the course website for assessment rules and dates;
A1 is released in Week 2.

## Source map

- [Jurafsky and Martin, Chapter 1](https://web.stanford.edu/~jurafsky/slp3/1.pdf),
  introduction and Section 1.1 (August 19, 2026 draft): the NLP field and LLMs.
  The course progression and coursework follow the [Fall course page](../../index.html).
- [Fudan Spring Lecture 01](https://baojian.github.io/llm-26/slides/lecture-01-slides/):
  instructor introduction, four-part outline, and the motivation for subwords.
  The local source is the sibling checkout's `llm-26/slides/lecture-01-slides/index.md`.
  Original slides [#5–11](https://baojian.github.io/llm-26/slides/lecture-01-slides/index.html#/5)
  supply the Qwen prompt, camera reviews, both complete translation passages,
  both complete article excerpts and their GPT-3 attribution, vision images,
  Stable Diffusion prompt/settings, and two recorded video examples. Complete
  text examples are retained in the notebook where slides use excerpts.
- Original development slides
  [#21–27](https://baojian.github.io/llm-26/slides/lecture-01-slides/index.html#/21)
  map respectively to ambiguity, messy text/reasoning, Weaver/Turing, the
  Turing test, rules-to-neural history, Transformers/pretraining, and the LLM
  timeline. The revised history corrects the LSTM date to 1997. The timeline
  comes from Zhao et al., [*A Survey of Large Language Models*, v16, Figure 3](https://arxiv.org/abs/2303.18223v16):
  it covers **2019–2024**, not the current model inventory. Media files and
  original filenames are documented in [assets/README.md](assets/README.md).
- [Stanford CS336 Spring 2026, Lecture 1](https://cs336.stanford.edu/lectures/?trace=lecture_01):
  learning by building, resource tradeoffs, the tokenizer interface, character
  and byte baselines, BPE training versus encoding, and compression measurement.
  The local source is the sibling checkout's `stanford-cs336-lectures/lecture_01.py`,
  file revision [`607a238629cf5332f71085d19c97dff41decf661`](https://github.com/stanford-cs336/lectures/blob/607a238629cf5332f71085d19c97dff41decf661/lecture_01.py).
  Precise sections: `why_this_course_exists()` (lines 65–123: building and budgets),
  `current_lm_landscape()` (126–185: historical ingredients and openness),
  `course_syllabus()` (235–253: resource constraints), `basics()` (265–316:
  components and design choices), and `tokenization()` with the tokenizer
  implementations below it (484 onward). Stanford's assessment rules do not apply.
- [Sennrich et al. (2016), Section 3.2](../../papers/sennrich-2016-subword-units.pdf):
  BPE-based subword segmentation. Our implementation begins with all 256 bytes;
  the paper's character vocabulary and word-boundary convention differ.
- [Ollama API](https://docs.ollama.com/api/generate): executable versions of the
  previous browser demos. The notebook uses the REST API from Python's standard
  library, so no additional Python package is required.

## Notebook and preparation

Use `lecture-01-exercise.ipynb` for executable work. The classroom
core includes short text calls, Unicode, and a self-contained byte BPE implementation. The
tokenization work runs without Ollama or network access. Ollama is a separate
application: install it, start it, and fetch `qwen3:0.6b` before class. Model calls
are short, but runtime depends on the classroom machine.

Log probabilities, thinking output, image understanding, Stable Diffusion, video
playback, and named `tiktoken` encodings are further notebook experiments. Vision
uses the bundled rainfall and Big Data images with an already installed
vision-capable model. The two recorded videos and their posters are bundled;
the deck uses manual playback and the notebook can display the same clips.
These opt-in experiments are not required to complete the 135-minute lesson.

For optional Stable Diffusion generation and extended tokenization practice:

```sh
uv sync --extra tokenization --extra multimodal
uv run --extra tokenization --extra multimodal python scripts/slides.py serve
```

Retain both extras in the launch command so the notebook kernel has the same
packages. Prepare a complete local or cached Diffusers pipeline separately,
set `SD_MODEL`, and enable `RUN_DIFFUSION` only when ready. Installing the
packages does not download model weights; the generation cell loads cached
files only. The original prompt uses 30 steps and guidance scale 7.5.

The additional [extended tokenization notebook](lecture-01-exercise-tokenization.ipynb)
copies the Spring notebook's Ollama, Unicode, regular expression, spaCy, dataset,
and tokenizer sections, with minimum edit distance removed. The deck links to
it after the tokenizer comparison. The toolbar still opens E01–E05.

For extended practice, run `uv sync --extra tokenization` and start the preview
with `uv run --extra tokenization python scripts/slides.py serve`. The notebook
explains the separate Ollama and spaCy model setup and dataset downloads.
WikiText tokenizer training uses only the training split. The optional
BookCorpus example fetches prepared statistics from the original course;
the corpus archive itself is not required for that plot. Save generated files
in the personal working copy under `workspace/`.

The toy corpus is `low` × 5 and `lower` × 2. The first two merges are `l o` and
`lo w`, producing weighted token totals 25, 18, and 11. Pair ties use ascending
token-ID pairs. Count overlapping pairs, but replace non-overlapping occurrences
left to right. Each training string is a separate sequence, so no merge crosses
a document boundary. The notebook deliberately omits production pre-tokenizers
and chat-template processing and explains those limitations.

The launcher preserves existing notebook answers. To receive a revised handout,
rename the existing personal notebook in JupyterLab, then click **Notebook**
again to create a fresh copy. Keep the renamed copy for your previous work.
