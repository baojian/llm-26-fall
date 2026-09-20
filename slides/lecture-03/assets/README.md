# Lecture 03 media

The PNG images are copied from the instructor's
[Spring 2026 Lecture 03](https://baojian.github.io/llm-26/slides/lecture-03-slides/)
(`slides/lecture-03-slides/media/` in the `baojian/llm-26` repository). The
seven `word2vec-*.png` images were downscaled from about 3460 to 2400 pixels
wide so the lecture folder stays small (6.2 MB to 4.4 MB); the content is
unchanged. The other images are byte-identical copies.

| File | Content and source |
| --- | --- |
| `models-text-classification.png` | Four classifier families in a two-by-two grid: Naive Bayes (Venn diagram), logistic regression (sigmoid curve), support vector machines (margin), and neural networks (layer diagram). Spring slide image; the Spring deck gives no source for the four pictures. |
| `word2vec-1.png`, `word2vec-2.png` | Skip-gram forward propagation of the loss, in two steps. Screenshots of the word2vec walkthrough by Eric Kim, [aegis4048.github.io](https://aegis4048.github.io/), credited on the Spring reference slide. |
| `word2vec-3.png` to `word2vec-7.png` | Backward propagation of the gradients through the output and input tables, in five steps. Same source. |
| `tensorflow-embedding.png` | Screenshot of the [TensorFlow Embedding Projector](https://projector.tensorflow.org/) with nearest neighbours of a query word. |
| `embedding-semantic-change.png` | Semantic change of *gay*, *broadcast*, and *awful* over time. Figure from Hamilton, Leskovec, and Jurafsky (2016), [Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change](https://arxiv.org/abs/1605.09096). |
| `embedding-paragraph-vector.png` | Paragraph vector framework. Figure from Le and Mikolov (2014), [Distributed Representations of Sentences and Documents](https://proceedings.mlr.press/v32/le14.html). |

The PNGs support the optional [classical reading](../classical-reading.md);
they are not projected in the revised 60-slide core deck. The ten-slide
classical bridge uses readable native tables, equations, and runnable examples.

## Revised lecture assets

| File | Content and source |
| --- | --- |
| `model-configs.json` | Selected fields from official Qwen3-0.6B and Qwen3-8B configurations, plus unique token-ID counts and maximum IDs from their tokenizer JSON artifacts. Each source has an immutable repository revision, URL, SHA-256 hash, and check date. No model weights or full tokenizer files are included. |
| `tiny-lm-loss.json` | Editable Plotly teaching figure, sampling notebook E04 at updates 0, 1, 5, 10, 20, 50, 100, 150, and 200. Eight toy input–target pairs, vocabulary 10, width 4, CPU float32, seed 0, nonzero input initialization, zero output initialization, untied tables, SGD learning rate 0.5. This illustrates a training loop and is not a benchmark. |

The notebook reads `model-configs.json` offline; P03 explains how to check the
primary sources separately. Unique tokenizer IDs are the union of
`model.vocab` values and `added_tokens` IDs, so duplicate entries are not
counted twice. Table arithmetic uses `config.vocab_size`, not the tokenizer
entry count. Both tokenizer snapshots have the same SHA-256 hash.

To refresh the teaching curve after intentionally changing E04, run that cell
and use `e04_losses[step]` for the listed steps. Index 0 is the loss before any
update; index 200 is after 200 updates. `tests/test_lecture_03.py` compares the
figure with the executable notebook, allowing small floating-point differences.

The PPMI counts in optional notebook P01 come from Jurafsky and Martin,
*Speech and Language Processing*, [Appendix J](https://web.stanford.edu/~jurafsky/slp3/J.pdf),
Figure J.2. The main deck's three-word co-occurrence table is invented toy data.
The notebook recomputes its total (35), PMI(tea, drink), and a rank-two SVD of
its PPMI matrix. It does not reuse the published P01 counts for that example.

The E04 held-out pairs 0 → 1 and 9 → 0 are constructed coverage probes, not a
representative evaluation dataset. They do not alter the training curve.
E06's memory demonstration measures the payload of a separate small CPU
model's tensors after one Adam update; no checkpoints or run logs are saved.
