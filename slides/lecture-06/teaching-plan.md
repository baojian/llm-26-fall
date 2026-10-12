# Lecture 06: Pretraining and decoding in practice

**Date:** October 14, 2026 (Asia/Shanghai). Three 45-minute teaching periods.

**Central question:** What evidence shows that a training run is working?

## Teaching package

The deck has **52 slides**. Present slides 1–46 and 52; slides 47–51 are optional
appendix material. The six ungraded exercises share one `practice.ipynb` and
appear in order. The cover, learning objectives, repeated outlines, and exercise
headers follow Lecture 05 and the shared course layouts. There is no quiz in
this class. The scope follows the approved
[preparation issue #293](https://github.com/baojian/llm-26-fall/issues/293).

Students trace batches into optimizer updates, check fitting and development
loss, recover a training state in a fresh process, and compare decoding policies.
Lecture 05 supplies the decoder; this lecture spends time on operating and
checking it. Lecture 07 develops data quality, Lecture 08 scaling, Lecture 09
evaluation, Lecture 10 adaptation, and Lecture 13 inference/KV caching.

The final course reminder uses A2's **October 31, 23:59** deadline and the
project proposal's **October 21, 23:59** deadline. All times are Asia/Shanghai.
A2 is submitted privately through eLearning. This package contains supplied
ungraded toy examples; graded reference implementations and hidden tests remain
in the private instructor repository. All students have identical expectations.

## Classroom route


Times below are minutes **within each 45-minute period**. Breaks are outside these 135 teaching minutes. Exercise time is included.

| Period | Minutes | Slides | Teaching activity |
| --- | --- | --- | --- |
| 1 | 0–6 | 1–6 | Title, objectives, outline, pretraining motivation, and the Noa model. |
| 1 | 6–13 | 7–8 | Shifted batches and E01, including four minutes of student work. |
| 1 | 13–20 | 9–11 | Mean loss, one update, and autograd checks. |
| 1 | 20–30 | 12–13 | AdamW intuition and decoupled weight decay. |
| 1 | 30–39 | 14–15 | Microbatches, accumulation, and four-minute E02. |
| 1 | 39–45 | 16 | Read a run record, questions, and transition buffer. |
| 2 | 0–7 | 17–19 | Outline, warmup, cosine decay, and update indexing. |
| 2 | 7–10 | 20 | Global gradient clipping. |
| 2 | 10–15 | 21 | E03: learning-rate endpoints and clipping calculation. |
| 2 | 15–22 | 22–24 | Tiny fitting check, held-out data, token-weighted evaluation. |
| 2 | 22–28 | 25–26 | Read loss/norm curves and choose a diagnostic control. |
| 2 | 28–35 | 27–29 | Checkpoint contents, restart order, and comparison protocol. |
| 2 | 35–41 | 30 | E04: run or inspect a fresh-process restart check. |
| 2 | 41–45 | 31 | Interpret the result, questions, and transition buffer. |
| 3 | 0–6 | 32–34 | Outline, five-minute E05 retrieval practice, and brief debrief. |
| 3 | 6–14 | 35–37 | Generation loop, greedy choice, and sequence-probability counterexample. |
| 3 | 14–19 | 38–39 | Temperature and fixed-k versus adaptive-mass filtering. |
| 3 | 19–27 | 40–41 | Top-p mechanism and five-minute E06. |
| 3 | 27–34 | 42–43 | Controlled sampling demonstration and stopping/context behavior. |
| 3 | 34–40 | 44–45 | Interpret generation evidence and connect to A2. |
| 3 | 40–45 | 46, 52 | Exit check, readings, questions, and buffer. |

If discussion takes longer, shorten the historical motivation and show the prepared restart evidence instead of waiting for a live process. Retain the shift, clipping, checkpoint, and top-p exercises. Appendix slides remain outside the timed route. Rehearse before treating these allocations as feasible classroom timings.


## Spring source map


The instructor-supplied **`lecture-06-slides-gpts.pptx`** contains **130 slides**, dated April 16, 2026. Its text and slide order have been inventoried, and selected training and decoding figures inspected.

- Source SHA-256: `0a252f6b509ef10b910c58d447c4ffeda48b2acc4854271e5ea90f8e5445c6c5`.
- A [public Spring copy](https://github.com/baojian/llm-26/blob/main/slides/lecture-06-slides/lecture-06-slides-gpts.pptx) is available, but its bytes differ from the supplied file. The page numbers below refer to the supplied file. Preserve it unchanged and record figure-level attribution during extraction.
- The [Spring code collection](https://github.com/baojian/llm-26/tree/main/lecture-06-gpts) is an additional implementation reference. Its full GPT-2 configuration and dependencies need adaptation for a short CPU classroom demonstration.

| Spring slides | Material | Treatment in the Fall course |
| --- | --- | --- |
| 1–2, 10, 19, 40, 64, 76, 89, 95, 104, 130 | Title, repeated outlines, references | Replace with the Fall question, three-period route, and verified readings. |
| 3–9 | History, static/contextual representations, ELMo | Optional background; connect to Lecture 03 without repeating its main sequence. |
| 11–18 | Teacher forcing, Transformer LM, pretraining and transfer | Redraw the useful training diagram with Noa; explain parameter updates and conditional prediction. |
| 20–24 | Autoregressive generation and greedy search | Retain the generation loop and a small probability tree. |
| 25–31 | Beam search, repetition, length effects | One optional appendix slide; keep the main route focused on the published decoding topics. |
| 32–39 | Top-k, nucleus sampling, generation examples | Core decoding material; add an explicit temperature calculation and a small interactive distribution. |
| 41–42 | Architecture/objective comparison | Brief reminder in notes; detailed architecture comparison already belongs to Lecture 05. |
| 43–52, 54–63 | GPT-1/2/3, transfer, zero/few-shot use | One short opening motivation and one optional historical comparison. Date historical claims and retain original experiment settings. |
| 53 | Bycroft visualization | Link back to the Lecture 05 resource list. |
| 65–73 | Corpus sources, filtering, data selection | Carry forward to Lecture 07; recheck dataset and rights claims before reuse. |
| 74 | Perplexity | Retain next to held-out cross-entropy, with tokenization and split caveats. |
| 75 | Resources, energy, fairness | Brief run-cost reminder; longer discussion in Lectures 08–09. |
| 77–88 | Scaling laws, FLOPs, compute allocation | Lecture 08 and the pretraining planning issues. |
| 90–94 | KV cache | Lecture 13; a forward pointer is sufficient here. |
| 96–101 | PEFT and LoRA | Lecture 10. |
| 102–103 | Hallucinations and harms | Lectures 09 and 11; keep sample quality claims modest here. |
| 105–125 | BERT, MLM, NSP, RoBERTa, SpanBERT | Optional architecture/objective reading; selected links can support later evaluation and masked-generation topics. |
| 126–129 | Encoder–decoder models and T5 | Optional objective comparison, with later adaptation and masked-generation connections. |


## Exercises and worked calculations


All six exercises are **ungraded** and have the same expectations for all students. Their IDs and order must match `practice.ipynb`. Put detailed answers and common mistakes in speaker notes and notebook reveals.

| Exercise | Time | Prompt | Expected response / check |
| --- | --- | --- | --- |
| E01 | 4 min | Two Noa windows are `[0,1,2,3,4]` and `[4,5,6,7,8]`. Build inputs and next-token targets. | `x=[[0,1,2,3],[4,5,6,7]]`, `y=[[1,2,3,4],[5,6,7,8]]`; `B=2`, `T=4`, eight scored positions. |
| E02 | 4 min | One process uses microbatch `B=4`, `T=8`, accumulation `K=2`, and 32 optimizer updates. All positions are scored. | 64 tokens/update; 2,048 processed tokens; 64 microbatches. Divide each equal-size microbatch mean loss by two before accumulating. |
| E03 | 5 min | With eight updates, two warmup updates, maximum LR `0.001`, and minimum ratio `0.1`, find LR at zero-based updates 0, 1, and 7. Clip gradient `(3,4)` to global norm 1. | LRs `0.0005`, `0.001`, `0.0001`; original norm 5; ideal clipped vector `(0.6,0.8)`. State that this hand calculation omits the implementation's numerical epsilon. |
| E04 | 6 min | Compare the supplied continuous and restarted toy runs. Identify the state needed to reproduce the next update. | Same future batch IDs and LR; compare next loss and final parameters within the declared tolerance. Weights alone omit AdamW history. The recorded control has final parameter difference 0.12780. |
| E05 | 5 min | Inspect three records: falling train/rising dev loss; nonfinite loss with a norm spike; unexpectedly good dev loss with overlapping document IDs. | Plausible hypotheses: overfitting or split mismatch, instability, leakage. Pair each with a concrete check; curves alone do not uniquely identify causes. |
| E06 | 5 min | For probabilities `(0.4,0.3,0.2,0.1)` and top-p `0.6`, which tokens remain and with what probabilities? | Retain the first two; cumulative mass `0.7`; renormalized distribution `(4/7,3/7,0,0)`. Greedy chooses the first token. |

Additional worked examples:

- Token-weighted development loss: eight targets at mean NLL 1 and two at mean NLL 3 give `(8×1+2×3)/10 = 1.4 nats/token`.
- Greedy counterexample: first-token probabilities `P(A)=0.6`, `P(B)=0.4`; second-token probabilities after A are `(0.5,0.5)` and after B `(0.9,0.1)`. For exactly two generated tokens, greedy obtains `0.6×0.5=0.30`; the B-first alternative obtains `0.4×0.9=0.36`.
- Noa fitting floor: two ambiguous targets among twenty give `2 ln(2)/20 = ln(2)/10 ≈ 0.069315`. This is the Lecture 05 fixture. A2 uses a different fixture and loss floor.

**Schedule convention:** for zero-based update `s < w`, use `η_max(s+1)/w`. Otherwise use `q=(s-w+1)/(N-w)` and `η_min + (η_max−η_min)(1+cos(πq))/2`, with `0 < w < N` for this exercise. The warmup maximum occurs at `w−1`; the last update reaches `η_min`. Persist the number of completed updates so the next LR is unambiguous.


## Exact experiments

The model is adapted from Lecture 05's implementation notebook. Vocabulary IDs
0–11 are Noa, can, be, annoying, but, she, is, a, great, cat, friend, EOS.
It uses two pre-LN blocks, width 16, four heads, ReLU FFN width 32, learned
positions for ten inputs, affine LayerNorm (epsilon 1e-5), final normalization,
no Linear biases or dropout, and a tied embedding/readout. It has **4,608**
unique parameters. This is a teaching configuration, with deliberate differences
from historical GPT-2.

| Experiment | Data and update protocol | Interpretation |
| --- | --- | --- |
| Tiny fit | Two Lecture 05 Noa documents; B=2,T=10; 200 full-batch updates; constant LR .02; no clipping/decay | Debugging fixture; no held-out set; final NLL 0.069822 versus infimum ln(2)/10 |
| Restart | Same two documents, randomly sampled B=1,T=10; 40 updates; warmup 4; maximum LR .02; minimum ratio .1; clip 1 | 40 continuous versus 20 + fresh-process restart + 20; parameter difference 0 on the recorded CPU |
| Development | 16 train/8 dev authored documents; one uniform random document/update; 200 updates; warmup 20; otherwise same scheduled settings | Exact documents disjoint; train/dev words and prefixes overlap; measured final token-weighted NLL .5153/1.4027 |

All runs use CPU float32, two threads, model seed 7, AdamW betas (.9,.999),
epsilon 1e-8, zero weight decay. Random-document runs use sampler seed 17.
Every whole document receives EOS; no BOS, padding, or cross-document windows.
Tiny fit processes 4,000 targets; restart 400; the recorded development run
1,021. Full train/dev evaluations score 82/48 targets respectively and weight
by tokens. Uniform-document stochastic training weights documents equally.
These are different aggregation conventions and must be described accurately.

The restart control restores the same weights, future batches, and schedule
but resets AdamW history. Its first forward loss still agrees. After the next
optimizer update, the second forward-loss difference is 0.02846 and the final
maximum parameter difference is 0.12780. This isolates a real reason to save
optimizer state. Absolute parameter tolerance is 1e-7 with relative tolerance
zero, under the recorded environment; no cross-device equality is promised.

The notebook aligns measured loss, LR, and pre-clipping norm on the update axis.
The slides show the loss and LR figures separately for readability; the full
norm figure is linked in optional reading. E05 curves are explicitly schematic.
Do not substitute schematic points into measured evidence.

The three prepared generations use the tiny-fit checkpoint and the prefix
ending in great. Seed 29 and a 12-token budget give `friend EOS` for greedy and
`cat EOS` for both shown sampling settings. These short completions are limited
evidence about quality. Browser controls use four invented logits and a separate
small seeded RNG; they do not claim to run the PyTorch model in JavaScript.

## Presentation and notebook use

Start the local server with the PyTorch extra installed:

```sh
uv sync --extra tokenization
uv run --extra tokenization python scripts/slides.py serve
```

Open `/slides/lecture-06/` and choose **Notebook**. The shared launcher preserves
personal copies in `workspace/`. Assets, including the small Python utilities,
are copied with the notebook. Existing saved student work is not overwritten.
The notebook runs offline after dependency setup. The browser deck uses locally
hosted Reveal, KaTeX, Plotly, diagrams, and chart data. Reset restores temperature
1, top-p .6, and browser RNG seed 29. PDF export uses a fresh page so its chart
shows the initial example.

The measurements and executable demonstrations have been checked. The 135-minute
allocation includes student exercises, questions, and transition buffers, but
has not been validated with a live class. Use the prepared restart summary if
local setup would consume the six-minute exercise. Keep the five appendix slides
outside the normal route.

## Validation and reproducibility

See [asset provenance](assets/README.md) for commands, source hashes, exact
settings, measured runtimes, and memory boundaries. See [optional reading](optional-reading.md)
for primary sources and later-lecture connections. Review files are ignored under
`slides/.checks/lecture-06/`.

```sh
uv run --extra tokenization python -m pytest -q tests/test_lecture_06.py
node tests/test_lecture_06_sampling.mjs
node slides/lecture-06/check-demo.mjs http://127.0.0.1:8000
npm --prefix slides run check -- lecture-06
npm --prefix slides run pdf -- lecture-06
```

Execute the notebook from a fresh kernel and inspect every rendered slide and
PDF page after substantive edits. Exercise IDs must remain E01–E06 in order;
the final reading slide has ID `references`.
