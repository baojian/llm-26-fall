# Lecture 02: N-gram Language Models

September 16, 2026 · CS40008.01 · Three 45-minute teaching periods

## Central question and learning objectives

How can a model assign a probability to a sentence, and how do we know whether
one model is better than another?

Students should be able to write the chain-rule factorization of a sentence
probability, state the Markov assumption behind an $N$-gram model, estimate
bigram parameters by maximum likelihood, explain perplexity as a normalized
inverse probability and as a branching factor, apply additive smoothing and
interpolation to a count table, and describe why a neural probabilistic LM
replaces count tables with embeddings.

The deck is a port of the instructor's
[Spring 2026 Lecture 02](https://baojian.github.io/llm-26/slides/lecture-02-slides/)
into the shared Reveal.js template. The content is kept; dense Spring slides
are split so that each slide holds one idea at the template's font sizes.
The deck has **55 slides**, including four repeated outlines and a final
reading page.

## Teaching sequence

| Period | Minutes | Slides | Content and activity |
| --- | --- | --- | --- |
| 1 | 0–10 | 1–4 | Title, outline, why sentences need probabilities: speech recognition, MT, spell correction |
| 1 | 10–25 | 5–9 | Unknown data distribution, KL objective, empirical log-likelihood, factorization, training samples |
| 1 | 25–45 | 10–15 | Chain rule, next-token prediction clip (play 2 min), unigram/bigram/trigram, Markov assumption, build steps, parameter matrix and MLE |
| 2 | 0–15 | 16–20 | E01 (5 min): toy bigram MLE; restaurant-review counts; sentence boundaries; OOV and UNK |
| 2 | 15–35 | 21–26 | Outline; data split and extrinsic evaluation; E02 (3 min): propose a metric; perplexity; E03 (3 min): digits example; WSJ perplexities |
| 2 | 35–45 | 27–29 | Unigram sampling, interval figure, WSJ samples; bigram sampling question |
| 3 | 0–20 | 30–41 | Outline; zero probabilities; smoothing intuition and the two charts; additive smoothing, pseudo-counts, add-δ; count tables; E04 (4 min); reconstituted counts |
| 3 | 20–33 | 42–48 | Linear interpolation; held-out tuning; Katz backoff; Good–Turing; E05 (4 min); Kneser–Ney idea |
| 3 | 33–45 | 49–55 | Summary; outline; four NPLM pages; toolkits and readings |

Breaks fall between periods and are outside the 135 teaching minutes. The five
E exercises total 19 minutes; the notebook's P01 is for after class. These are
**ungraded practices**. Assignment A1 (text and probability) is released this
week and defined by the course website and the instructors' repository.

## Notebook correspondence

The classroom notebook is [lecture-02-exercise.ipynb](lecture-02-exercise.ipynb).
It runs offline with the standard library only.

| Slide | Notebook | Evidence to discuss |
| --- | --- | --- |
| Toy example of training a bigram LM | E01 | Six MLE fractions; the column for history `I` sums to one |
| Intrinsic evaluation | E02 | One unseen bigram makes the test log-likelihood $-\infty$ |
| Perplexity: interpretation | E03 | Uniform digits give perplexity 10 for every length |
| Using Laplace smoothing | E04 | The 8×8 probability and reconstituted-count tables recomputed from the counts |
| Good–Turing example | E05 | The estimates sum to exactly one; the most frequent word gets zero |
| Sentence sampling | P01 | Bigram sampling from BOS until EOS |

## Differences from the Spring deck

- Dense Spring slides were split (objective, chain rule, $N$-gram definitions,
  practical issues, evaluation, perplexity, sampling, smoothing methods, the
  8×8 tables) so each slide fits the shared template without inline styles.
- The HTML bar charts on the smoothing-intuition slide are Plotly charts.
- The bigram MLE on the parameter-matrix slide is written with the history in
  the denominator, $C(v_j v_i)/C(v_j)$; the Spring slide had $C(v_i v_j)/C(v_i)$.
- The closing slide lists Chapter 3 as this lecture's reading; the Spring slide
  pointed to Chapters 4–5 for its next lecture.
- Whiteboard prompts became timed exercises E01–E05 with revealed answers.
