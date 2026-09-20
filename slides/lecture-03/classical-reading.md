# Classical background for Lecture 03

This optional reading extends the ten-slide classical bridge in the 60-slide
lecture. It supplies longer explanations of the briefly introduced static
methods and the classification background outside the core lecture.
The core lesson follows token IDs through
lookup, next-token loss, autograd, and a training step. No additional graded
work is introduced here.

The instructor's full
[Spring Lecture 03](https://baojian.github.io/llm-26/slides/lecture-03-slides/)
remains available for the longer explanations. The original Fall port is also
preserved in the history of [PR #157](https://github.com/baojian/llm-26-fall/pull/157).

## Classification background

A classifier maps an input to scores for a set of labels. A next-token language
model also makes a classification decision, with the vocabulary as the label
set. The core lecture therefore retains logits, softmax, and cross-entropy.

Naive Bayes provides a counting-based baseline, while logistic regression
learns a linear scoring function. They are useful background for text
classification, but their full derivations are not prerequisites for tracing
today's neural language-model computation. Review Spring slides 3–14 if these
ideas are unfamiliar.

## Counting and PPMI

A word–context matrix records co-occurrence counts. A term–document matrix
instead records words across documents. These matrices make the distributional
hypothesis concrete: a representation is a profile of where a word occurs.

Pointwise mutual information compares a pair's joint frequency with its
frequency under independence:

$$\operatorname{PMI}(w,c)=\log_2\frac{p(w,c)}{p(w)p(c)}.$$

Positive PMI clips negative values to zero. Notebook **P01** uses the published
counts in Jurafsky and Martin, [Appendix J, Figure J.2](https://web.stanford.edu/~jurafsky/slp3/J.pdf).
The frequent pair information/data has PPMI about 0.0944 bits, whereas
cherry/pie has about 4.38 bits. Frequency alone does not measure association.

Zero co-occurrence produces negative infinity before clipping. Sparse counts
and rare events make these estimates noisy. See
[Chapter 5](https://web.stanford.edu/~jurafsky/slp3/5.pdf) for the distributional
view and vector similarity.

TF-IDF weights term counts using document frequency. It remains useful in
retrieval; we revisit it alongside search rather than treating it as a step in
neural LM training.

## Static embedding methods

| Method | Useful idea | Reading |
| --- | --- | --- |
| SVD | Factor a count-based matrix into lower-dimensional factors. The result depends on the weighting and matrix being factored. | Jurafsky and Martin, [Appendix J](https://web.stanford.edu/~jurafsky/slp3/J.pdf) |
| Skip-gram | Learn vectors by predicting contexts around a center word. | [Mikolov et al. (2013)](https://arxiv.org/abs/1301.3781) |
| CBOW | Predict a center word from a representation of surrounding words. | Same paper |
| GloVe | Learn vectors using a weighted objective over global co-occurrence counts. | [Pennington et al. (2014)](https://aclanthology.org/D14-1162/) |
| fastText | Include character n-gram representations in word vectors. This is distinct from an LLM's subword tokenizer. | [Bojanowski et al. (2017)](https://aclanthology.org/Q17-1010/) |

The core lecture introduces these methods in slides 11–20 and keeps one
skip-gram gradient calculation to show how a training objective changes
representations. PPMI/SVD are demonstrated on an invented three-word count
table; detailed comparisons and historical derivations remain optional.
The goal is to explain what each representation preserves and how an LLM's
context network changes the representation of a token occurrence.

## One skip-gram update

For one positive pair and one sampled negative pair:

$$\ell=-\log\sigma(\mathbf w^\top\mathbf c_+)
       -\log\sigma(-\mathbf w^\top\mathbf c_-).$$

Writing $s_+=\sigma(\mathbf w^\top\mathbf c_+)$ and
$s_-=\sigma(\mathbf w^\top\mathbf c_-)$ gives

$$\nabla_{\mathbf c_+}\ell=(s_+-1)\mathbf w,\qquad
\nabla_{\mathbf c_-}\ell=s_-\mathbf w,$$

$$\nabla_{\mathbf w}\ell=(s_+-1)\mathbf c_+ + s_-\mathbf c_-.$$

Notebook **E02** checks this calculation against autograd. **P02** trains and
inspects toy vectors using the same objective. Negative sampling scores
observed versus sampled pairs; it does not define the normalized next-token
probabilities needed for language-model perplexity. See
[Mikolov et al. (2013), Section 2.2](https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf).

The Spring port also includes a longer illustrated derivation, credited to
Eric Kim: [forward 1](assets/word2vec-1.png), [forward 2](assets/word2vec-2.png),
[backward 1](assets/word2vec-3.png), [backward 2](assets/word2vec-4.png),
[backward 3](assets/word2vec-5.png), [backward 4](assets/word2vec-6.png),
[backward 5](assets/word2vec-7.png). These images are better read at full size
than projected in a classroom. [Asset provenance](assets/README.md) records
their origin.

## Inspecting representations

Cosine neighbors, analogy queries, and low-dimensional projections can help
inspect embeddings. Their results depend on data, preprocessing, and the
objective. A successful analogy is not a sufficient evaluation of a language
model. The notebook's template-generated corpus is deliberately easy to
separate; its results do not measure real-world performance.

Static vectors can also be compared across periods to study semantic change;
see [Hamilton et al. (2016)](https://arxiv.org/abs/1605.09096) and the
[Spring illustration](assets/embedding-semantic-change.png).
For an early document-level representation, see
[Le and Mikolov (2014)](https://proceedings.mlr.press/v32/le14.html) and the
[paragraph-vector illustration](assets/embedding-paragraph-vector.png).

A modern retrieval vector is trained and constructed for comparing queries or
documents. It should not be confused with the raw token embedding table or an
arbitrary hidden state. The [Qwen3-Embedding report](https://arxiv.org/abs/2506.05176)
provides a later-course connection; no live model download is part of this
lecture.
