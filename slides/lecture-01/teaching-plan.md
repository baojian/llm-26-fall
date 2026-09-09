# Lecture 01: Introduction and Tokenization

September 9, 2026 · CS40008.01 · Three 45-minute teaching periods

## Central question and learning objectives

How does an LLM application turn our text into something a model can predict?

Students should be able to inspect a model response against a concrete check,
explain what a preprocessing rule preserves or discards, distinguish code points
from bytes and tokens, trace BPE training and fixed-rank encoding, and compare
vocabularies on held-out text without equating compression with model quality.

The deck has **71 slides**, including five repeated outlines, a combined review
and sources slide, and a final reading page. **Development of NLP & LLMs has ten
historical pages and three September
2026 updates**, excluding its outline. Keep the shared Reveal.js template,
fonts, margins, and controls.
Minimum edit distance is outside this lecture and both companion notebooks.

## Teaching sequence

| Period | Minutes | Slides | Content and activity |
| --- | --- | --- | --- |
| 1 | 0–10 | 1–8 | Instructor introduction, four topics, assessment, learning strategy, app survey |
| 1 | 10–25 | 9–16 | Prepared Qwen call; five application groups; E01 (5 min), E02 (3 min); course website and Qizhi GPU introduction (1 min) |
| 1 | 25–39 | 17–27 | Development outline and ten historical pages; Turing test, changing methods, access and evaluation |
| 1 | 39–45 | 28–30 | 2026 models (1 min), Terminal-Bench Science and invitation to follow (2 min), Navier–Stokes announcement (3 min) |
| 2 | 0–12 | 31–39 | Sources, preprocessing decisions, regex patterns, P01 (2 min), lossless partitioning |
| 2 | 12–25 | 40–44 | Code points, UTF-8, live counter, E03 (3 min), normalization, decoding |
| 2 | 25–45 | 45–56 | Tokenizer interface, next-token prediction, P02 (2 min), subwords, weighted pair counts, two BPE merges, E04 (4 min), vocabulary |
| 3 | 0–17 | 57–64 | Fixed-rank encoding, E05 (5 min), larger Spring corpus, boundaries, special tokens, algorithm families |
| 3 | 17–37 | 65–69 | Vocabulary and sequence costs; E06 (8 min); compare held-out languages and interpret results |
| 3 | 37–45 | 70–71 | Exit questions, discussion, sources and extended-practice link, readings |

Breaks fall between periods and are outside the 135 teaching minutes. The six
E exercises total 28 minutes; P01 and P02 add four minutes. The authorship poll
and quick prediction prompts are short discussions within the indicated blocks.
These are **ungraded practices**. The Fall course website defines assessment:
quizzes 10%, assignments 45%, individual project 45%; A1 is released in Week 2.

Use the speaker notes for timing, expected responses, and common mistakes.
If a live model call is slow, use the checked response criteria and continue.
Do not spend the timed gallery installing software or waiting for optional
image generation. Keep the algorithm comparison brief when time is tight;
the extended notebook is linked from the closing slide for work after class.

## Course overview and notebook correspondence

The classroom notebook is [lecture-01-exercise.ipynb](lecture-01-exercise.ipynb).
Its four main sections match the deck. Its opening course map contains the
same learning sequence, assessment, instructor details, and local workflow.
The closing overview slide and notebook link the course website, CFFF registration,
and the Qizhi GPU platform for coursework and projects. Per-student compute budgets
follow the project handout, as stated on the course website.
The full four-book/seven-course Spring resource inventory and detailed course
map remain in the notebook. The deck introduces resources on the course website
slide and closes with selected sources and readings.

| Slide example | Notebook location | Evidence to discuss |
| --- | --- | --- |
| Example 01: Qwen | One prompt, one response | Shanghai; inspect actual response, token counts, and stop reason |
| Task 1: sentiment | E01; `sentiment` cell | Positive, positive, negative; the two uses of “light” have different contexts |
| Task 2: translation | E02; `original-translation-examples` | A's quantities and date; B's definition and prediction |
| Task 3: authorship | Article generation quiz | Both attributed to GPT-3 by the Spring source; fluency alone cannot verify authorship or facts |
| Task 4: image understanding | Image understanding; `vision` | Big Data illustration; separate visible labels, logos, and lines from inferred relationships |
| Task 5: images/video | Optional diffusion and recorded-video cells | Prompt adherence and visible evidence; the Johannesburg clip is locally hosted |

The slide's **Open the classroom notebook** link uses the shared launcher,
so it works with the local course server without a fixed Jupyter port or
workspace identifier. The toolbar opens the same notebook. Both retain the
student's personal copy. To receive a revised handout, rename an old personal
notebook and reopen the launcher; preserve the renamed copy for earlier work.

For E02, use A as the three-minute core and B as a follow-on comparison.
The checked quantities are **249 languages**, **over 200 million daily users**,
**April 2016**, **over 500 million total users**, and **over 100 billion words
per day**. Keep the date, units, and qualifiers. These are historical source
sentences for translation, not a current fact sheet. Both complete passages
and both complete article excerpts remain in the notebook; projected excerpts
are shortened for reading at classroom distance.

Task 4 follows the article quiz in the notebook and uses the bundled Big Data
image with `qwen3-vl:2b`. Its Thinking variant needs a larger
generation budget; prepare a response before class and compare its claims with
the image rather than waiting for inference during the gallery. Preserve the
notebook's model-specific settings and five-minute timeout. Live vision,
diffusion, thinking, log probabilities, and extra generation are optional.
The default offline run exercises the complete preprocessing and BPE material.

## Ten historical pages and three 2026 updates

| Page | Slide ID | Date / idea | Teaching question |
| --- | --- | --- | --- |
| 1 | `early-nlp` | 1949: Weaver and translation | Why does “bank” need context? |
| 2 | `turing-test` | 1950: anonymous text interaction | What can the judge observe? |
| 3 | `turing-evidence` | 1950: imitation and accuracy | Is a convincing reply a correct calculation? |
| 4 | `rules-and-statistics` | 1960s–1990s: rules and corpora | What supplies a system's behavior? |
| 5 | `development` | 2003–2014: learned representations | What can word vectors and sequence models share? |
| 6 | `milestones` | 2014–2017: attention and Transformer | How can a decoder consult relevant positions? |
| 7 | `pretraining` | 2018: BERT and GPT | What objective can text itself supply? |
| 8 | `in-context-learning` | 2020: GPT-3 | What changes when demonstrations enter the prompt? |
| 9 | `instruction-tuning` | 2022: instruction following | How do demonstrations and preferences change behavior? |
| 10 | `llm-landscape` | 2023–2025: access, tools, reasoning | What can we run, inspect, and independently check? |
| 11 | `models-2026` | 2026: Fable 5.1, GPT-6 Astra, GLM-5.3, Qwen3.8, Kimi K3, DeepSeek-V4-Pro | How should we compare their capabilities? |
| 12 | `terminal-bench-science` | 2026: scientific workflow evaluation | What output and verification does a research task require? |
| 13 | `navier-stokes-2026` | September 8, 2026: reported mathematical result | How do we assess the paper and formalization? |

The two Turing illustrations are explicitly identified as generated conceptual
scenes. The first depicts the simplified human-versus-machine adaptation;
Turing's original game begins with a man, a woman, and an interrogator. The
paper's arithmetic reply is 105,621; the correct sum is 105,721. The dates mark
selected papers, and approaches continue to coexist. Attention predates the
Transformer. BERT's preprint is 2018; publication is 2019. ReAct's preprint is
2022; publication is 2023. The first ten pages are historical landmarks through
2025. The three updates are a September 9, 2026 snapshot, without a model ranking.

Use the official model links and primary research sources in the matching notebook
update. Invite students to follow Terminal-Bench Science and inspect a task's
verification. Link the Anthropic and OpenAI release announcements reporting
Fable 5.1 and GPT-6 Astra evaluations. Present the Navier–Stokes
announcement as a reported solution with a public paper and Lean formalization;
Clay's problem page still says Unsolved at the snapshot date. Keep that status
visible and revisit it before a later presentation. Pause for the first break
after this discussion.

## Preprocessing and tokenization

P01 compresses the Spring regex task to `Senjō 3 can't 你好🙂`. The ASCII
word pattern yields `['Senj', "can't"]`. The next pattern partitions every
character and round-trips the input, but still does not establish linguistic
word boundaries. E03 checks precomposed and decomposed accents, Chinese, and
emoji. NFC, NFKC, lowercasing, and byte decoding are distinct operations.
The full Spring ambiguity, messy-text, and reasoning examples are preserved
as notebook discussion prompts beside the preprocessing material.

P02 distinguishes word occurrences, word types, and byte tokens: `low low lower`
has 3 whitespace-separated tokens, 2 types, and 13 UTF-8 bytes. BPE then uses
`low`×5 and `lower`×2 as separate training sequences. The first two merges
produce weighted totals **25 → 18 → 11** and a 258-entry byte vocabulary.
The browser demo computes each step using the same tie-breaking rule as the
notebook. Its overlap example demonstrates **10 → 8**, despite a winning
pair count of four: replacements cannot share a character.

The larger Spring corpus is `low`×5, `lowest`×2, `newer`×6, `wider`×3, `new`×2,
with an appended toy `_` marker. The first two merges are `e r → er` and
`er _ → er_`; totals are **96 → 87 → 78**. The notebook calculates these values.
Later tied choices may differ from the Spring illustration. The source's
incorrect `(low, er_) → low_` concatenation is not reproduced. Separate input
sequences prevent cross-word merging; a printed boundary marker alone is not
a full boundary policy.

E05 keeps the vocabulary fixed: `lowest` encodes as `[257, 101, 115, 116]`.
Chinese and emoji retain byte coverage even without useful learned merges.
E06 trains with 0, 8, and 32 merges and evaluates fixed held-out sentences by
language. For the two Chinese sentences, mixed training produces 54, 48, and
35 tokens; English-only training produces 54 throughout. These small synthetic
corpora are teaching examples, not language-model benchmarks.

## Preparation and sources

```sh
uv sync
uv run python scripts/slides.py serve
```

Prepare Ollama and the selected model separately before class. The core
preprocessing and tokenization require no model, GPU, or network. Optional
packages are documented in the notebooks. For the extended and diffusion work:

```sh
uv sync --extra tokenization --extra multimodal
uv run --extra tokenization --extra multimodal python scripts/slides.py serve
```

The [extended notebook](lecture-01-exercise-tokenization.ipynb) retains the
Spring regex, spaCy, vocabulary-growth, corpus, and pretrained-tokenizer work.
Its dependencies and downloads are optional; the toolbar still opens the
classroom notebook. Student files belong in `workspace/`.

Primary adaptation source: [Fudan Spring Lecture 01](https://baojian.github.io/llm-26/slides/lecture-01-slides/),
local sibling checkout `llm-26/slides/lecture-01-slides/index.md` and
`llm-26/lecture-01-tokenization/lecture-01-exercise-tokenization.ipynb`.
The Fall course page governs logistics. The notebook history table and each
slide's speaker notes link the relevant original papers and documentation.
[CS336 Lecture 1](https://cs336.stanford.edu/lectures/?trace=lecture_01) supplements
the tokenizer interface, byte implementation, building perspective, and budgets;
its assessment rules do not apply. Read [Sennrich et al., §3.2](../../papers/sennrich-2016-subword-units.pdf)
for BPE and [SentencePiece](../../papers/kudo-2018-sentencepiece.pdf) for additional
text handling. Visual provenance and editable sources are in
[assets/README.md](assets/README.md); image prompts are in
[assets/image-prompts.md](assets/image-prompts.md).
