# Lecture 02 media and charts

All files are copied from the instructor's
[Spring 2026 Lecture 02](https://baojian.github.io/llm-26/slides/lecture-02-slides/)
(`slides/lecture-02-slides/media/` in the `baojian/llm-26` repository) unless
noted. The images were downscaled to at most 2400 pixels wide so the lecture
folder stays small; the content is unchanged.

| File | Content and source |
| --- | --- |
| `sentence-sampling.png` | Unigram sampling on the unit interval. Figure from Jurafsky and Martin, Chapter 3, as used in the Spring deck. |
| `Katz-backoff.png` | Backoff strategy and the Katz backoff formula. Spring slide image. |
| `nplm-1.png` | Neural probabilistic LM introduction with Figure 1 and the header of Bengio et al. (2003), JMLR. Spring slide image. |
| `nplm-2.png` | Forward inference in a feedforward neural LM; diagram from Jurafsky and Martin, Chapter 7. Spring slide image. |
| `nplm-3.png` | Training the neural LM with embeddings as parameters; diagram from Jurafsky and Martin, Chapter 7. Spring slide image. |
| `nplm-4.png` | Improvements over N-gram LMs, the AP News perplexity table from Bengio et al. (2003), and a window-of-four NPLM diagram. Spring slide image. |
| `predicting-next-word.mp4` + `-poster.jpg` | Interview clip on next-word prediction, 2 min 36 s. The Spring deck bundled this clip without a stated source or license; it was re-encoded from 32 MB (1280×720) to 3.8 MB (854×480) for this repository. The poster is the frame at one second. |
| `smoothing-raw-counts.json` | Plotly bar chart of the toy counts after “denied the”: allegations 3, reports 2, claims 1, request 1, and three zeros. Recreated from the Spring slide's HTML bars. |
| `smoothing-after.json` | Plotly bar chart of the same counts after moving two counts to “other.” Recreated from the Spring slide's HTML bars. |

The Berkeley Restaurant Project bigram tables in `slides.md` are typed in
directly and were recomputed in `lecture-02-exercise.ipynb` (E04) from the
counts and unigram totals given in Jurafsky and Martin, Chapter 3.
