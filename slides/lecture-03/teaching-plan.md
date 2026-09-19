# Lecture 03: Embeddings and PyTorch for Language Models

September 23, 2026 · CS40008.01 · Three 45-minute teaching periods

## Central question and learning objectives

How do discrete tokens become trainable representations?

Students should be able to state the distributional hypothesis, define an
embedding matrix $E \in \mathbb{R}^{|V| \times d}$, compute a PPMI value from a
co-occurrence table, write the skip-gram negative-sampling loss for one
positive and $k$ negative pairs and carry out one SGD update, compare
embeddings with cosine similarity and the analogy query, explain an embedding
lookup as a one-hot product, predict the tensor shapes from token IDs
$(B, T)$ to logits $(B, T, |V|)$, explain what `loss.backward()` computes and
which embedding rows receive a gradient, and count the parameters and memory
of the input and output tables with and without weight sharing.

## Status: first version for instructor review

This deck has two parts.

1. **A complete reproduction of the instructor's
   [Spring 2026 Lecture 03](https://baojian.github.io/llm-26/slides/lecture-03-slides/)**
   (59 slides: text classification, counting-based methods, word2vec, bridge
   to LLM embeddings). Every sentence, formula, table, figure, and link is
   kept. The Spring slides used 0.6–0.8em text inside two-column boxes; the
   shared template enforces 30px body text, so Spring slides 2–59 became
   **127 Fall slides**, split at the Spring column and box boundaries. Nothing
   was shortened. The instructor will recheck, reformat, and reduce this part;
   the text-classification section is the first candidate.
2. **Fall additions (13 slides):** a Fall title slide, exercises E01 and E02
   inside the Spring material, and a closing section "PyTorch for Language
   Models" with exercises E03 and E04 and the exit questions.

The deck has **140 slides**. That is more than three periods can hold at full
depth; the teaching sequence below marks what to skim until the reduction is
done.

## Teaching sequence

| Period | Minutes | Slides | Content and activity |
| --- | --- | --- | --- |
| 1 | 0–3 | 1–2 | Title and outline; the question of the day |
| 1 | 3–18 | 3–32 | Text classification as review: task and examples, Naive Bayes in one pass, logistic regression from score to sigmoid/softmax to cross-entropy. Keep slides 16–22 (sigmoid, softmax, loss), because the PyTorch section reuses them; skim the GD-step arithmetic (26–28) |
| 1 | 18–45 | 33–57 | Counting-based methods: word meaning, distributional hypothesis, formal definition of $E$, term–document and co-occurrence matrices, TF-IDF, PMI/PPMI. **E01** (4 min): PPMI(information, data) |
| 2 | 0–30 | 58–88 | word2vec: core idea, training pairs, negative-sampling objective, parameters $W$ and $C$, running example, softmax versus negative sampling, SGD objective, explicit gradients. **E02** (5 min): one update by hand |
| 2 | 30–38 | 89–100 | Forward and backward propagation figures (seven images, one minute each at most: autograd replaces them in period 3), initialization, window size, Embedding Projector |
| 2 | 38–45 | 101–106 | Evaluation: similarity benchmarks, analogy query, benchmark table |
| 3 | 0–5 | 107–127 | Skim: semantic change, document embeddings, CBOW versus skip-gram, SVD and GloVe, fastText, classic resources; bridge to Qwen3-Embedding and EmbeddingGemma (demos are described, not run) |
| 3 | 5–27 | 128–136 | PyTorch for language models: lookup table and one-hot product, `nn.Embedding` and shapes, output projection and cross-entropy, **E03** (4 min), autograd on the E02 numbers, the five-line training step, weight sharing and memory, **E04** (3 min) |
| 3 | 27–30 | 137–140 | Exit questions, references, preview of Week 4 |
| 3 | 30–45 | — | **Quiz 1** (15 minutes, closed book), as published on the course website. The quiz is prepared and kept outside this repository |

Breaks fall between periods and are outside the 135 teaching minutes. The four
E exercises total 16 minutes; the notebook's P01–P03 are for after class.
These are **ungraded practices**, distinct from Quiz 1 and from Assignment A1,
which are defined by the course website and the instructors' repository.

## Notebook correspondence

`lecture-03-exercise.ipynb` runs offline on a CPU in a few seconds with the
core `uv sync` environment (PyTorch arrives through `edtrace`). It uses toy
data made for this lecture; it shares no data, model class, or code with
Assignment A1.

| ID | Slide | Notebook content | Numbers shown on the slide |
| --- | --- | --- | --- |
| E01 | `exercise-01` | PMI and PPMI from the Jurafsky and Martin word–context counts (Appendix J, Figure J.2) | 0.0944 bits; totals 7703, 5673, 11716 |
| E02 | `exercise-02`, `autograd` | One skip-gram negative-sampling update by hand, then the same gradients with `loss.backward()` | $\sigma(1)=0.7311$, $\sigma(0.5)=0.6225$, loss 1.2873, gradients $(-0.2689,-0.1345)$ and $(0.4880,-0.8914)$ |
| E03 | `lookup-code`, `output-projection`, `exercise-03` | Lookup shapes, one-hot equivalence, logits, cross-entropy at $\log 10 = 2.3026$, rows of $E$ with a gradient | shapes $(2,3,4)$ and $(2,3,10)$; rows 0, 1, 5, 7 |
| E04 | `weight-sharing`, `exercise-04` | `TinyLM` parameter counts with and without sharing; GPT-2 small and Qwen3-0.6B tables; memory in fp32 and bf16 | 38,597,376 of 124,439,808; 155,582,464; 593.5 and 296.8 MiB |
| P01 | — | Skip-gram with negative sampling in PyTorch on a template-generated toy corpus (window 3, $k=5$, word2vec initialization) | first-minibatch loss $6\log 2 = 4.1589$ (notebook only) |
| P02 | — | Nearest neighbours and the analogy query on the toy vectors | — |
| P03 | `training-step` | The five-line training loop memorizes one batch | starts at 2.3026, below 0.5 after 200 steps |

`tests/test_lecture_03.py` executes every code cell with network access
blocked and checks these numbers against the notebook and the slide source.

## Spring-to-Fall slide map

| Spring slide | Spring title | Fall slides |
| ---: | --- | --- |
| 1 | Title (Text Classification and Word Embeddings) | 1 (Fall title; the Spring title is recorded in the note) |
| 2 | Outline | 2 |
| 3 | Assign probabilities to sentences | 3–4 |
| 4 | Classification methods | 5–6 |
| 5–7 | Naive Bayes: classifier, bag of words, training and prediction | 7–12 |
| 8–13 | Logistic regression: classifier, binary and multinomial, loss, objective and GD, one GD step, training in practice | 13–30 |
| 14 | From NB and LR to Neural Text Classification | 31–32 |
| 15 | Outline | 33 |
| 16–19 | Word meaning, distributional hypothesis, formal definition, how to obtain embeddings | 34–43 |
| 20–24 | Counting-based representations, raw counts, TF-IDF, co-occurrence, PMI/PPMI | 44–56 |
| — | Fall E01 | 57 |
| 25 | Outline | 58 |
| 26–37 | word2vec: background to one SGD step | 59–87 |
| — | Fall E02 | 88 |
| 38–44 | word2vec forward and backward propagation figures | 89–95 |
| 45–46 | Initialization; window and visualization | 96–100 |
| 47–49 | Embedding evaluation | 101–106 |
| 50–51 | Semantic change; sentence and document embeddings | 107–110 |
| 52–55 | CBOW versus skip-gram; SVD and GloVe; fastText; classic resources | 111–122 |
| 56 | Outline | 123 |
| 57–58 | Bridge to LLMs: Qwen embeddings, EmbeddingGemma | 124–127 |
| — | Fall section: PyTorch for Language Models, E03, E04, exit questions | 128–137 |
| 59 | References and Next Lecture | 138–140 |

Each slide's speaker note names its Spring slide and, for a split slide, its
part number.

## Differences from the Spring deck

- **Layout only:** Spring's coloured boxes became plain paragraphs, lists, or
  blockquotes; blue sub-headings and closing lines became bold paragraphs;
  red and green emphasis use the template's `text-bad` and `text-good`. Where
  a Spring fragment revealed a whole column, that column is now its own slide,
  so advancing the slide is the reveal. Three display equations are set inline
  to fit (cosine on the evaluation procedure, the two analogy examples); the
  notes of those slides say so.
- **Outline:** the four Spring topics plus the new fifth topic, "PyTorch for
  Language Models", on every outline slide.
- **Defects fixed:** the title typo "wor2vec" on the seven figure slides; an
  `http://` GloVe link; two broken relative links to the word2vec papers, now
  absolute; an unclosed `<b>` on the TF-IDF slide and stray `</b>` tags on the
  parameters slide. Left as written: two literal `---` sequences (Spring
  slides 3 and 4) and the carried-over title "Assign probabilities to
  sentences" on Spring slide 3.
- **Next lecture:** the Spring closing box named "Neural Language Models and
  Sequence Labeling"; it now names the Fall Week 4 lecture. The Spring wording
  is in the slide's note.
- **Figures:** the seven word2vec figures are wide (2:1). The template's
  `.diagram` class fixes image height at 410px, so they render at about
  820×410 with small in-image text. A wider figure class would be a change to
  the shared theme and is left for the review.
- **Not ported:** the Spring notebook (scikit-learn Naive Bayes and logistic
  regression on newsgroups and IMDB, gensim word2vec, a Qwen3-Embedding
  download, an Ollama EmbeddingGemma call). It needs an out-of-band dataset
  archive, packages outside the course environment, and network access. The
  Fall notebook is new and offline.

## Candidates for reduction (for the instructor's review)

`docs/course-revision.md` keeps the distributional hypothesis, embedding
lookup, one skip-gram example, similarity, and contextual representations, and
makes these optional: repeated NB/LR derivations, the full word2vec gradient
sequence, the GloVe/SVD/fastText survey; TF-IDF moves to the retrieval week.
In this deck that corresponds to slides 7–12 (Naive Bayes), 23–30 (logistic
regression objective, gradient descent, and training), 48–49 (TF-IDF), 89–95 (propagation figures, superseded by the autograd
slide), and 111–122 (other static embeddings).
