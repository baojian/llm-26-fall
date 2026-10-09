# Lecture 04: Neural language models and recurrent attention

**Class date:** September 30, 2026 (Asia/Shanghai).
**Revision:** October 9, 2026 (Asia/Shanghai), following the instructor's decision
to teach attention within recurrent translation and leave causal self-attention
to Lecture 05.

**Central question:** How can a model retain and select useful context?

The revised package has **42 slides**, five ungraded exercises, and an offline
CPU notebook. All students have the same practices and expectations. The
shared course theme, notebook launcher, and original NPLM fitting example
are retained. Original preparation: [issue #234](https://github.com/baojian/llm-26-fall/issues/234).

## Learning objectives and lecture boundary

Students train a fixed-window neural LM, trace recurrent states and gradients,
calculate a modern LSTM cell update, and explain additive alignment in an RNN
encoder–decoder. They distinguish the full source sentence from the target
prefix available at a decoding step.

Lecture 05 starts with a four-minute retrieval recap, then develops Q/K/V,
scaled dot products, causal self-attention, multiple heads, positions,
Transformer blocks, and complete-model verification. It does not require a
Lecture 04 self-attention implementation.

## Changes from the original 56-slide deck

These old slide numbers refer to the deck before this revision.

| Original slides | Revision and destination |
| --- | --- |
| 1–20: fixed-window LM and training | Retain; update objectives and outline |
| 21–25: RNN/LSTM bridge | Expand into recurrent-state, gradient, and gate practice in current slides 21–30; combine cell memory with gates on slide 29, and E04 with remaining limitations on slide 30 |
| 26: encoder context selection | Combine the encoder–decoder update with source Transformer slide 7's bottleneck illustration in current slide 32; develop additive alignment in current slides 31–41 |
| 27–37: weighted values, Q/K/V, scaling, matrix attention | Replace with recurrent alignment; self-attention is developed in Lecture 05, slides 6–19 |
| 38–42: causal masks and shifted targets | Remove from Lecture 04; covered by Lecture 05, slides 9–11 and 38–39 |
| 43: causal browser demonstration | Move code, fixture, and interaction checks to Lecture 05, slide 10 |
| 44–50: head implementation, output/gradient references, causality | Remove duplicate notebook section; covered by Lecture 05's references and E05 |
| 51–53: attention cost, LM path, positions | Remove from Lecture 04; covered by Lecture 05, slides 21–26, 35, and 51 |
| 54–56: exit questions and reading | Combine recurrence review, continuation, and readings in current slide 42 |

Lecture 04's former E03–E05 and optional P01–P02 attention tasks are replaced.
Current E03 traces memory, E04 calculates a gated update, and E05 calculates
source alignment. Existing personal notebooks are preserved by the launcher;
rename an old working copy before requesting the revised one.

## Three-period sequence

Each period is 45 minutes including exercises. Breaks are separate. These are
planning allocations, not measured classroom completion times.

| Period | Minutes | Slides | Teaching and activity |
| --- | --- | --- | --- |
| 1 | 0–5 | 1–5 | Objectives, bigram recap, and two prefixes sharing the last token |
| 1 | 5–15 | 6–10 | Vocabulary, windows, feedforward path, and shapes |
| 1 | 15–19 | 11 | E01, 4 min: context windows and shapes |
| 1 | 19–29 | 12–15 | PyTorch, loss, stability, and one update |
| 1 | 29–35 | 16 | E02, 6 min: repair the detached loss and fit the batch |
| 1 | 35–45 | 17–20 | Loss curve, predictions, diagnosis, and generalization |
| 2 | 0–10 | 21–24 | Fixed windows, recurrent states, and update shapes |
| 2 | 10–17 | 25 | E03, 7 min: state trace and input derivative |
| 2 | 17–29 | 26–28 | Recurrent LM, gradient products, and shared-weight derivatives |
| 2 | 29–35 | 29 | Cell memory, gates, and exposure |
| 2 | 35–42 | 30 | E04, 7 min: memory, exposure, and direct derivative |
| 2 | 42–45 | 30 | Reveal the remaining recurrent limitations after the exercise answer |
| 3 | 0–9 | 31–33 | Machine translation task; encoder–decoder update and bottleneck; attribution |
| 3 | 9–22 | 34–37 | Source annotations, additive scores, context, decoder update |
| 3 | 22–25 | 38 | Numerical alignment example |
| 3 | 25–32 | 39 | E05, 7 min: two contexts from the same annotations |
| 3 | 32–39 | 40–41 | Source versus target visibility and interpretation |
| 3 | 39–45 | 42 | Combined recap, continuation, and readings |

Preserve practice and feedback. If discussion runs long, leave the supplied
library-comparison code for reading rather than typing it in class.

## Notebook correspondence and computation

| Exercise | Cells | Expected evidence |
| --- | --- | --- |
| E01 | `window-data`, `context-model` | Six sentence-local windows; lookup/flat/hidden/logit shapes `(2,2,4)`, `(2,8)`, `(2,8)`, `(2,7)` |
| E02 | `broken-loop`, `fit-tiny-batch` | Detached loss fails; connected training fits all six targets |
| E03 | `recurrent-trace`, `rnn-reference` | States `(1,0.5,1.25)`; input derivatives `(0.25,0.5,1)`; shared-weight derivative 1 |
| E04 | `lstm-update`, `lstm-cell-reference` | Cell 1.25, hidden state about 0.4241, direct cell derivative 0.75 |
| E05 | `alignment-numbers`, `additive-attention` | Weights `(1/4,3/4)` give context `(2.5,0.5)`; reversed weights give `(1.5,1.5)` |

E03's scalar recurrence is deliberately linear. A separate tanh recurrence
matches `torch.nn.RNN`. E04 holds supplied gates fixed for its direct-path
derivative, then compares a full modern cell with `torch.nn.LSTMCell`. Tests
compare outputs and gradients. Additive attention has an independent scalar
reference, finite-difference checks, and a one-source edge case.

The NPLM fit retains CPU float32, seed 7, vocabulary 7, context 2, embedding
width 4, hidden width 8, and 200 SGD updates at learning rate 0.5 on six windows.
There is no held-out evaluation. New reference examples use float64, seeds
23 and 29, and no training. No external data or models are downloaded.

## Historical and technical precision

[Bahdanau, Cho, and Bengio](../../papers/2015-iclr-bahdanau-neural-machine-translation-align-translate.pdf) introduced additive
soft alignment for neural machine translation: September 2014 preprint, ICLR
2015. Avoid a universal first-attention claim. The paper's §6.1 discusses
[Graves's earlier handwriting alignment](https://arxiv.org/abs/1308.0850);
[Mnih et al.'s visual-attention preprint](https://arxiv.org/abs/1406.6247) also
appeared earlier, in June 2014.

Use the paper's actual dependencies: a bidirectional gated RNN produces source
annotations, the previous decoder state scores them, and their weighted context
feeds the decoder update. The paper uses GRUs; the LSTM section is a separate
memory lesson. Its additive score differs from the Transformer's scaled dot product.

All valid source annotations are available at each target step. Source and
target indices refer to different sequences. A target-index triangular mask
would wrongly hide available source words. Padding masks are separate.

## Spring continuity and validation

The [Spring slides](https://github.com/baojian/llm-26/blob/main/slides/lecture-04-slides/lecture-04-slides.pdf)
and [notebook](https://github.com/baojian/llm-26/blob/main/lecture-04-neural-lms/lecture-04-neural-lms.ipynb)
remain optional. Current material retains NPLM (source slides 17–24) and gives
more room to recurrence/BPTT/gates (38–46 and 60–73). Long micrograd and LSTM
training remain optional. Bahdanau §§2–3 and Appendix A.1.2 ground period 3.

```sh
uv run python -m pytest tests/test_lecture_04.py tests/test_lecture_05.py
node --test tests/test_attention_demo.mjs tests/test_position_demo.mjs
npm --prefix slides run check -- lecture-04
npm --prefix slides run check -- lecture-05
```

Inspect every slide screenshot. Causal interaction checks now run with Lecture
05. Lecture 04's loss curve uses the shared Plotly loader without a custom demo
module. Export and inspect a PDF with the standard `pdf` script before
distributing one. Review artifacts remain in ignored `slides/.checks/`.
