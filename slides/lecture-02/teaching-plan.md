# Lecture 02: N-gram Language Models

September 16, 2026 · CS40008.01 · Three 45-minute teaching periods

## Central question and learning objectives

How can a model assign a probability to a sentence, and how do we know whether
one model is better than another?

Students should be able to write the chain-rule factorization of a sentence
probability, state the Markov assumption behind an $N$-gram model, estimate
bigram parameters by maximum likelihood, explain perplexity as a normalized
inverse probability and as a branching factor, apply additive smoothing and
interpolation to unseen $n$-grams, and describe why a neural probabilistic LM
replaces count tables with embeddings.

The deck is a port of the instructor's
[Spring 2026 Lecture 02](https://baojian.github.io/llm-26/slides/lecture-02-slides/)
into the shared Reveal.js template. The content is kept; dense Spring slides
are split so that each slide holds one idea at the template's font sizes.
The deck has **38 slides**, including four repeated outlines and a final
reading page. Smoothing is one page; the Spring section of nine slides is
kept in the notebook as optional practices P02 and P03.

## Teaching sequence

| Period | Minutes | Slides | Content and activity |
| --- | --- | --- | --- |
| 1 | 0–10 | 1–4 | Title, outline, why sentences need probabilities: speech recognition, MT, spell correction |
| 1 | 10–25 | 5–9 | Unknown data distribution, KL objective, empirical log-likelihood, factorization, training samples |
| 1 | 25–45 | 10–15 | Chain rule, next-token prediction clip (play 2 min), unigram/bigram/trigram, Markov assumption, build steps, parameter matrix and MLE |
| 2 | 0–15 | 16–20 | E01 (5 min): toy bigram MLE; restaurant-review counts; sentence boundaries; OOV and UNK |
| 2 | 15–35 | 21–26 | Outline; data split and extrinsic evaluation; E02 (3 min): propose a metric; perplexity; E03 (3 min): digits example; WSJ perplexities |
| 2 | 35–45 | 27–29 | Unigram sampling, interval figure, WSJ samples; bigram sampling question |
| 3 | 0–12 | 30–32 | Outline; smoothing in one page (zero probabilities, add-δ, interpolation); N-gram summary |
| 3 | 12–40 | 33–37 | Outline; four NPLM pages: task and embeddings, forward inference, training, improvements over N-grams |
| 3 | 40–45 | 38 | Toolkits and readings; preview of Week 3 (embeddings) |

Breaks fall between periods and are outside the 135 teaching minutes. The three
E exercises total 11 minutes; the notebook's P01–P03 are for after class. These are
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
| Sentence sampling | P01 | Bigram sampling from BOS until EOS |
| Smoothing N-gram LMs in one page | P02 (optional) | The 8×8 Laplace probability and reconstituted-count tables recomputed from the counts |
| Smoothing N-gram LMs in one page | P03 (optional) | Good–Turing estimates sum to exactly one; the most frequent word gets zero |

## Differences from the Spring deck

- Dense Spring slides were split (objective, chain rule, $N$-gram definitions,
  practical issues, evaluation, perplexity, sampling) so each slide fits the
  shared template without inline styles.
- The smoothing section (Spring slides 20–27: intuition, additive smoothing
  with the 8×8 tables, interpolation, Katz backoff, Good–Turing, Kneser–Ney)
  is one page, per the instructor's decision on September 11, 2026. The
  worked examples live in notebook practices P02 and P03.
- The bigram MLE on the parameter-matrix slide is written with the history in
  the denominator, $C(v_j v_i)/C(v_j)$; the Spring slide had $C(v_i v_j)/C(v_i)$.
- The closing slide lists Chapter 3 as this lecture's reading; the Spring slide
  pointed to Chapters 4–5 for its next lecture.
- Whiteboard prompts became timed exercises E01–E03 with revealed answers.
