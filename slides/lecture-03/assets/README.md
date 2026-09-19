# Lecture 03 media

All files are copied from the instructor's
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

The Fall additions (exercises E01–E04 and the PyTorch section) use no media.
Their numbers come from `../lecture-03-exercise.ipynb` and are checked by
`tests/test_lecture_03.py`. The co-occurrence counts in E01 are from Jurafsky
and Martin, *Speech and Language Processing* (3rd ed. draft of August 19, 2026),
[Appendix J](https://web.stanford.edu/~jurafsky/slp3/J.pdf), Figure J.2.
