# Lecture 05: Attention and the Transformer

**Date:** Saturday, October 10, 2026 (Asia/Shanghai), make-up class.
Time and room follow the university notice.

**Central question:** How does attention select context, how does it form a
Transformer, and what evidence supports the design?

## Package and teaching boundary

The deck is a **60-slide material bank**. It integrates all **31 component and
paper topics** from lecture-05-test, plus 29 original or consolidated pages.
The first merge integrated the source verbatim. The instructor then authorized
adapting the main deck and notebooks to a single Noa example. The current
revision keeps all 60 IDs, topics, citations, and reveal sequences; the test
deck stays unchanged. The [README](README.md) records this revision boundary.

### Running-example convention

Use “Noa can be annoying but she is a great cat.” The working prefix has six
toy word tokens at positions 0–5: Noa, can, be, annoying, but, she. The query
is she at position 5; its next-token target is is at position 6. A real tokenizer
can split words. The drawn vectors and attention patterns are invented.

Q/K/V and multi-head diagrams keep six token rows. The numeric causal demo
shows a seventh future word solely to test masking. The permutation control
swaps Noa and she under deliberately unmasked attention. RoPE starts with
she at 5 and Noa at 0; controls hold content fixed while changing hypothetical
positions. FFN and LayerNorm exercises use supplied small states at she.
The full-model notebook extends the sentence, adds the alternative ending
friend, and appends EOS. Its ten input positions are a different scope from
the six-position prefix diagram, with the same word convention.

Point to the word order above Q/K/V, then use the implementation captions to
explain the change from six prefix rows to ten input rows. The logit row at
she (5) predicts is (6); the row at cat or friend (9) predicts EOS (10).
EOS is outside this batch's input positions 0–9 and needs no eleventh position
embedding in this batch.

Lecture 04 supplies recurrent memory and additive alignment. Lecture 03
supplies embeddings, raw-logit loss, autograd, and weight tying. Lecture 01
supplies tokenization. This lecture introduces self-attention, Q/K/V, causal
masking, multiple heads, sinusoidal positions, RoPE, residuals, LayerNorm,
and the FFN, then reads the 2017 paper's experiments in their stated settings.

All practices have the same requirements for all students and are ungraded.
No current quiz questions, graded reference solutions, hidden tests, or
student data are included. Quiz 2 is separate private material and keeps its
published **15-minute** duration.

## Classroom route: 120 teaching minutes plus Quiz 2

The whole bank needs more time than one class. Use the following route through
**all 31 component and paper topics** and selected original pages. The original
implementation pages omitted from this route remain in the deck and notebook
for follow-up. Use stable-ID navigation or the Reveal overview to pass them.

These are preparation allocations, not observed completion times. Core practice
times are included. Bring the diagrams and prepared numerical checks up before
class; do not type or train the full model during the timed route.

| Period | Minutes | Slides / stable IDs | Activity |
| --- | --- | --- | --- |
| 1 | 0–5 | 1–4 | Objectives, outline, recurrent-attention recall |
| 1 | 5–19 | 5–9 | Context, self-attention, roles, projections, GPT overview |
| 1 | 19–26 | 10–12 | Position inputs, sinusoidal frequencies, core E01 |
| 1 | 26–29 | 13, `position-demo` | Predict the permutation result; prepared Implementation E02 demonstration |
| 1 | 29–40 | 14–16 | RoPE rotation, core E02, insertion-point comparison |
| 1 | 40–45 | Current component diagrams | Questions and transition buffer |
| 2 | 0–9 | 17–20 | Single head, numeric causal demonstration, Implementation P01 |
| 2 | 9–19 | 21–23 | Multiple-head intuition, mechanism, core E03, architecture uses |
| 2 | 19–27 | 24–27 | Residuals, LayerNorm, norm order, core E04 |
| 2 | 27–34 | 28–30 | FFN mechanism, parameter share, core E05 |
| 2 | 34–43.5 | 34–39; skip 31–33 | Block code, input recap, tiny configuration, learned table, block ledger |
| 2 | 43.5–45 | Current block diagram | Questions and transition buffer |
| 3 | 0–7 | 40, 42–44; skip 41 | Complete model, shifted targets, loss, prepared Implementation E05 causality check |
| 3 | 7–25 | 50–56; skip 45–49 | Original paper, experiment setup, core E06, results, ablations, parsing, later efficiency work |
| 3 | 25–29 | 57–58, 60; skip 59 | Recap, course dates, final evidence and reading slide |
| 3 | 29–30 | Questions | Finish the teaching route |
| 3 | 30–45 | Separate private material | **Quiz 2 — 15 minutes** |

Implementation E05 uses three minutes for a prediction and prepared evidence
in this route; another three minutes of notebook work remain for follow-up.
Implementation E02 similarly uses a three-minute prepared browser comparison.
Keep the component explanations and core practices when allocating time for extra implementation work.
If discussion consumes the buffers, defer original implementation details to
the supplied notebook and begin the quiz on time. A full rehearsal is needed
to establish whether this route fits the class's actual discussion pace.

### Follow-up pages

- **31–33:** batched tensor axes, split/join code, and Implementation E01.
- **41:** Implementation E04, counting the tied complete model.
- **45–49:** irreducible toy loss, fitting loop, measured norm comparison,
  materialized attention storage, and the three architecture configurations.
- **59:** implementation and further reading links.

These pages preserve the executable-course detail without requiring a live
coding session inside the same class as the full component and paper sequence.
The code is already supplied; the activities ask for predictions and checks.

## Notebooks and exercise correspondence

The default [core notebook](../shared/notebook.html?lecture=lecture-05&notebook=practice.ipynb)
retains E01–E06 and their order from lecture-05-test, with the main lecture's
Noa positions, six-row head example, and supplied she states. It needs Python's
standard library and no trained model.

| Core exercise | Slide / ID | Expected evidence |
| --- | --- | --- |
| E01 | 12 / `sinusoidal-example` | PE(0) = (0,1,0,1); PE(1) added coordinate-wise to the supplied can embedding |
| E02 | 15 / `rope-relative-offset` | she at 5 and Noa at 0 give −√3/2 for the toy pair; shared shifts preserve the fixed-content dot product |
| E03 | 22 / `multi-head-attention` | Width 512 split into 8 heads gives 64 per head; six prefix rows stay six rows |
| E04 | 27 / `add-norm-practice` | Mean 2, variance 1, LN approximately (−1,1), residual output (3,2), per-token feature statistics |
| E05 | 30 / `ffn-practice` | (1,−3,4) → ReLU (1,0,4) → (5,−4); isolated token independence; nonlinearity and parameter-share checks |
| E06 | 52 / `paper-parameter-count` | Encoder count under the slide's explicit bias, normalization, vocabulary, and position assumptions |

The separate [implementation notebook](../shared/notebook.html?lecture=lecture-05&notebook=lecture-05-exercise.ipynb)
now uses the same Noa vocabulary and sentences. Its learning objectives, cell
IDs, architecture, seed, and 200-update budget remain, while the data, table
sizes, checks, and measured loss figure are updated together. The archived
previous notebook and figure retain their own hashes. Slide labels say
**Implementation** and carry an explicit notebook link and
`data-exercise-notebook` attribute.

| Implementation activity | Slide / ID | Code cells and evidence |
| --- | --- | --- |
| E02 | 13 / `position-demo` | `e02-permutation`: unmasked restored outputs agree without positions; fixed positions give max difference about 0.449204 |
| P01 | 20 / `practice-01` | `single-head-practice`: causal weights (2,1,1,2,1,3,0)/10, output (1,1), zero future-is contribution |
| E01 | 33 / `exercise-01` | `multi-head-attention`, `e01-trace`, `e01-wrong-reshape`: shapes (2,4,10,4), (2,4,10,10), (2,10,16); values/gradients agree with a head loop; wrong reshape fails |
| E04 | 41 / `exercise-04` | `decoder-model`, `e04-count`: 2,112 per block; 4,608 tied or 4,800 untied; tying reuses the Parameter object |
| E05 | 44 / `exercise-05` | `e05-causality`, `e05-fit`: changing is at 6 leaves prefix logits 0–5 unchanged and their gradients to states 6–9 zero; an unmasked control leaks |

The duplicated original Implementation E03 slide is removed because core E04
uses the same calculation. Its `e03-block` cells remain in the implementation notebook
and define modules used later. Its FFN isolation check remains available.
The implementation labels are section references, so their appearance follows
the component teaching sequence rather than numerical order. Run that notebook
top-to-bottom for a complete execution; the default core notebook follows the
slide exercise order exactly.

Use the shared launcher for separate personal working copies. Existing student
notebooks are not overwritten. Rename an older personal copy before requesting
a fresh handout, retaining earlier answers.

## Exact teaching model

| Setting | Value |
| --- | --- |
| Vocabulary | Noa, can, be, annoying, but, she, is, a, great, cat, friend, EOS; IDs 0–11 |
| Documents | [0,1,2,3,4,5,6,7,8,9,11] and [0,1,2,3,4,5,6,7,8,10,11] |
| Input / target | First / last ten tokens of each document; EOS appended, no BOS, no padding or cross-document windows |
| Shape | Batch 2, length 10, width 16; 4 heads of width 4 |
| Blocks | 2 pre-LN blocks; ReLU FFN width 32 |
| Positions | Learned absolute table, 10 × 16; no embedding scaling |
| Norms | Feature axis; affine scale and bias; epsilon 1e−5; final LayerNorm |
| Linear layers / readout | No Linear biases; token embedding Parameter reused for output |
| Regularization | No dropout, label smoothing, weight decay, or clipping |
| Initialization | Seed 7; embedding/Linear weights normal, std 0.02; norm scale 1 and bias 0 |
| Training | CPU float32; full batch; 200 AdamW updates; lr 0.02; betas (0.9,0.999); epsilon 1e−8 |
| Independent checks | Float64 head outputs and gradients, tolerance 1e−10; full-model causality also checked independently |

One block has 4d² + 2d·d_ff + 4d = 2,112 parameters. Two blocks,
token and position tables, and final normalization give
4,224 + 192 + 160 + 32 = **4,608** unique parameters. Untying adds 192,
giving **4,800**.

The GPT-2-style overview uses learned positions, pre-LN, GELU,
and a standard fourfold FFN expansion. The tiny model deliberately uses ReLU,
no Linear biases, and a twofold expansion. Its FFN therefore has **one half**
of the block matrix weights; the **two-thirds** explanation assumes
fourfold expansion. Counting embeddings, norms, biases, cross-attention, or
prediction heads changes total-model fractions. Core E06 is another explicit
encoder-only configuration and must not be presented as an exact BERT count.

The two documents share the prefix through great, then target cat or friend
equally often. Two ambiguous targets among twenty give a mean-loss infimum
of log(2)/10 ≈ **0.069315 nats/token**. The new reference run ends at about
**0.069822 pre-LN** and **0.069761 post-LN** after 200 updates per variant.
All eighteen deterministic transitions are correct; the pre-LN probabilities
after great are approximately 0.49965 for cat and 0.49970 for friend. These
are observations of the stated CPU run, not cross-platform guarantees.
There is no held-out set, and one deterministic top-1 choice cannot satisfy
both ambiguous targets.

The pre/post-LN comparison keeps the final LayerNorm in both models and changes
only sublayer order from the same initial state. Preserve all recorded updates,
including spikes and unfavorable results. It is one small fitting experiment,
not a universal ranking or a complete reproduction of the 2017 model. Existing
command, revision/dirty-state record, configuration, and measured outputs for
the new Noa run are in [asset provenance](assets/README.md). The earlier
experiment is archived with its original data and notebook hash. Do not compare
the two curves as if they used the same corpus or loss floor.

### Attention cost

The retained `attention-cost` page counts one materialized score or weight
tensor: 4BhT² bytes in float32. With B=2 and h=4, lengths 256, 1024, and
2048 need 2, 32, and 128 MiB for that tensor. This is not total training
memory. The later-paper slide compares Reformer and Linformer;
it changes the attention computation rather than merely its implementation.

## Integration decisions

First-merge source equality is a historical record. The second revision,
authorized by the instructor, changes the main deck's running sentence,
indices, matrix dimensions, numerical fixtures, and full-model corpus. It does
not change the original test deck or historical paper results. Source-page
references describe the Desktop PowerPoint snapshot used for each topic;
they are not current slide numbers. The paper section uses the later 17-slide
source snapshot.

Required adjacent bundles are retained: context → self-attention → roles →
projections; GPT overview → position inputs; multi-head intuition → mechanism
→ architecture; FFN mechanism → motivation → exercise; paper setup → encoder
count. Original tensor implementation pages follow the FFN bundle, so they do
not interrupt the multi-head architecture explanation. The final
`references` section is the source paper conclusion section; the old main references
become `implementation-readings`.

| Original main material | Treatment |
| --- | --- |
| Recurrent/self-attention comparison | Fold into recurrent recap notes |
| Q/K/V recap and attention contract | Replaced by integrated roles, projections, and single-head slides |
| Numeric attention and mask-before-softmax | Consolidate into numeric causal demo; preserve incorrect post-softmax masking counterexample |
| Architecture families and Q/K/V-source table | Covered by integrated attention-in-architecture |
| Head motivation, equations, and static path | Replaced by integrated multi-head sequence |
| Split and join code | Combine into `split-heads`; retain shape and independent-reference practice |
| Position theory, demo, original E02 | Combine into the permutation demo with explicit implementation link |
| Sinusoidal formula and frequency plot | Replaced by integrated frequency and calculation slides |
| Residual, norm order, feature axis, original E03 | Replaced by complete Add & Norm sequence and core E04 |
| Original FFN and token/feature mixing | Replaced by integrated three-slide FFN sequence |
| Shifted target table | Incorporate into causal visibility diagram and notes |
| BERT contrast | Supplementary reading and implementation context |
| Complete-model checklist and evidence table | Combine into Implementation E05; retain negative control and input-state gradient distinction |
| Normalization comparison limits | Keep with the measured figure |
| Original references | Separate implementation-reading page before the final source references |

## Map from the original 85-slide 2025 source

The earlier port used the instructor's `lecture-05-Transformers-I/lecture-05-slides.pptx`,
titled Lecture 05 – Attention and Transformers, dated October 15, 2025.
The following inclusive ranges cover that source once. Destinations now use
stable IDs rather than stale page numbers from the 55-slide port.

| 2025 source slides | Current destination or treatment |
| --- | --- |
| 1–2 | Current title/objectives and section outlines |
| 3–8 | Lecture 04 development; `recurrent-attention-recap`; integrated architecture paths |
| 9 | Shared active-topic outlines |
| 10–16 | Weighted-average analogy in optional reading |
| 17 | Shared active-topic outlines |
| 18–20 | `context-for-text` and optional contextualization reading |
| 21 | Lecture 03 prerequisite; integrated `input-tokenization` |
| 22–28 | Integrated self-attention/roles/projections; `attention-demo`, P01, permutation demo |
| 29 | Correct lookup and dimensions in reading; integrated role/projection notes |
| 30–31 | Integrated single-head mechanism and numeric notebook examples |
| 32–35 | Integrated multi-head sequence; implementation axes and references |
| 36 | Consolidated component sequence |
| 37 | Shared active-topic outlines |
| 38–41 | Integrated GPT and attention architectures; `full-model`; original-paper section |
| 42 | Correct jointly trained embeddings in `input-tokenization` and learned-table code |
| 43–50 | Lecture 01 and integrated input/tokenizer recap |
| 51 | Integrated GPT overview and implementation `full-model` |
| 52–54 | Integrated sinusoidal section and E01 |
| 55 | Integrated RoPE and E02; NoPos qualification remains in notes/reading |
| 56–57 | Integrated position input branches and implementation learned table |
| 58 | `causal-visibility`, shifted loss, implementation notebook |
| 59–61 | Integrated single head, numeric causal demo, P01, complete-model checks |
| 62–63 | Optional attention visualization reading, with interpretation limits |
| 64 | Head-pruning reading in integrated multi-head notes |
| 65–67 | Complete integrated Add & Norm sequence |
| 68–69 | Complete integrated FFN sequence and core E05 |
| 70–71 | `full-model`, `logits-and-loss`, fitting loop |
| 72–74 | Integrated paper setup and translation results |
| 75 | Integrated results separate estimated training FLOPs from speed |
| 76 | `toy-configuration`, `original-and-baseline`, integrated paper setup |
| 77 | Separate ungraded encoder count E06 and decoder Implementation E04 |
| 78 | Integrated `paper-ablations`, with development/test distinction |
| 79–80 | Integrated `paper-constituency-parsing` |
| 81 | `attention-cost` and integrated `paper-efficient-attention` |
| 82 | Integrated Linformer explanation and further reading |
| 83 | Integrated final modification-transfer discussion and comparison limits |
| 84 | Qualified paths/parallelism in integrated paper overview and residual notes |
| 85 | Integrated final references plus implementation reading |

## External walkthroughs

These are optional follow-up demonstrations with local fallbacks, not additional
requirements on top of the timed route:

| Resource | Prompt | Local fallback |
| --- | --- | --- |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Which keys remain available; where do the weighted values go? | `attention-demo` and `single-head-mask` |
| [Bycroft's LLM Visualization](https://bbycroft.net/llm) | Identify token and feature axes and the vocabulary readout | `full-model` and `decoder-model` |
| [LLM-Visualized](https://www.llm-visualized.com/?token=4&generation=0&kvCache=0) | How does the prefix change after one generated token? | `causal-visibility` and local decoder diagram |

Keep KV caching off for this introduction. These sites show different models;
return to the specified teaching configuration for all calculations. Allow at
most ten seconds for an unavailable page before using the local fallback.

## Validation and release review

```sh
uv run python -m pytest tests/test_lecture_05.py
node --test tests/test_position_demo.mjs
node --test tests/test_lecture_05_test_positions.mjs
npm --prefix slides run check -- lecture-05
npm --prefix slides run pdf -- lecture-05
```

Validate the documented Noa adaptations, source-topic order, adjacent bundles, unique IDs,
exercise/notebook links, all reveal controls, deep links, reset, diagrams,
causal and permutation controls, and PDF page count. Inspect all slide images;
automated checks do not establish legibility or classroom timing. Run both
notebooks from a clean kernel with the documented environment. Check the core notebook against the main Noa slides and the implementation
notebook against the current measured figure's hash and data. Verify the
untouched test deck against its historical snapshot. Review outputs stay under ignored
`slides/.checks/lecture-05/`. Prepare a PR and perform the classroom rehearsal
before describing this material bank as a completed lecture release.
