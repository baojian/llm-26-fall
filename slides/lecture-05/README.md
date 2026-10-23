# Lecture 05: Attention and the Transformer

The deck contains **62 slides**: 31 component and paper topics integrated from
`lecture-05-test`, 29 original or consolidated Lecture 05 pages, and two
visual explanations accompanying EleutherAI’s RoPE blog. The first
merge copied the test content verbatim. The instructor then authorized a second
revision of the main deck to use **one Noa example throughout**. All prior 60 stable
IDs, topics, source citations, and reveal sequences remain. The test deck is left unchanged during this revision.

## One running example

**Noa can be annoying but she is a great cat**

The main mechanism uses the prefix **Noa / can / be / annoying / but / she**,
with zero-based positions 0–5. The query at **she** predicts **is** at position 6.
Each displayed word is one **toy token**; a real tokenizer may split it.
Vectors, projection illustrations, and head patterns are supplied teaching data.

- Self-attention, Q/K/V, multiple heads, and tokenizer diagrams keep six rows.
- The causal numeric demonstration displays **is** as a seventh, masked position.
- Position controls hold content fixed while testing actual or hypothetical slots.
- FFN and LayerNorm calculations use small supplied states at **she**.
- The executable model trains on the full sentence and one variant ending
  **friend**, with EOS appended to each. Its batch is **2 × 10** token inputs.

The Q/K/V diagram lists the six words in matrix-row order. The implementation
slides show the continuation to ten input words and the final shift from
cat or friend to EOS. EOS is a target beyond the ten input positions.

Reusing the sentence lets students follow changes in the computation without
also interpreting a new sentence. Historical translation and parsing datasets,
reported results, and paper configurations retain their original scope.

## Open the materials

- [Integrated slides](index.html)
- [Classroom route and source map](teaching-plan.md)
- [Core notebook: E01–E06](../shared/notebook.html?lecture=lecture-05&notebook=practice.ipynb)
- [Implementation notebook: P01 and E01–E05](../shared/notebook.html?lecture=lecture-05&notebook=lecture-05-exercise.ipynb)
- [Further reading](optional-reading.md)
- [Asset provenance](assets/README.md)

The toolbar's **Notebook** link opens the core notebook. Original coding
activities say **Implementation E01/E02/E04/E05** and link to the separate
PyTorch notebook. Implementation P01 also identifies its notebook. The two
notebooks use separate personal working copies. Both main notebooks now use
the Noa examples; the implementation notebook and measured figure have been
updated together, with the earlier experiment retained in the archive.

Use Space, arrows, Next, or the diagram to advance supported reveals; Previous
reverses them. Reset restores the current mechanism. Self-attention and Q/K/V
use the same Noa artwork in both query-design modes; their Original switch is
hidden. The GPT architecture keeps its Reference/Modern comparison. RoPE,
numeric causal, and permutation controls remain interactive, and printing
shows complete examples and staged answers.

## What changed

The component walkthrough is now the core sequence: context → self-attention
→ Q/K/V → GPT overview → sinusoidal positions and RoPE → one head and multiple
heads → Add & Norm → FFN. The input recap leads into the executable model and
the original paper's experiments. The final `references` slide remains at the
end. The parsing illustration now identifies a noun phrase in the Noa sentence;
its WSJ data and results remain unchanged.

The original main deck's repeated Q/K/V, head, position, normalization, and FFN
explanations were replaced by the integrated component sequence. Its numerical masking,
permutation demonstration, tensor implementation, model count, complete-model
causality, toy fitting result, and measured normalization comparison are retained
or consolidated. The old reading slide becomes `implementation-readings`,
before the final source reading slide.

This is a **material bank**, not a promise that every page can be taught in
120 minutes. The teaching plan gives a route through every component and paper topic,
selected implementation checks, and the original paper. Longer code walkthroughs
and experiment discussions remain available for follow-up. Quiz 2 keeps its
separate **15-minute** slot.

## Configuration boundaries

| Example | Positions / norms | FFN | Counting convention |
| --- | --- | --- | --- |
| Original 2017 translation model | Sinusoidal / post-LN | ReLU, width 4d, biases | Encoder and decoder; decoder also has cross-attention |
| GPT-2-style overview | Learned / pre-LN | GELU, width 4d, biases | Standard causal decoder blocks |
| Noa implementation | Ten learned slots / pre-LN | ReLU, width 2d, no Linear biases | 4,608 tied parameters; affine LayerNorms |

The FFN occupies **two thirds** of standard decoder-block matrix weights at
width 4d. It occupies **one half** at the tiny implementation's width 2d.
Neither fraction is an exact universal share of whole-model parameters.
The sentence stays fixed while explicitly stated widths vary to keep the
individual calculations manageable; the small arithmetic vectors and width-512
head exercise are not activations from the width-16 trained model.
Core E06 uses a separate stated encoder-only configuration; it is not an exact
BERT count or the tiny decoder count.

## Integrated slide map

| Slide | Stable ID | Origin |
| --- | --- | --- |
| 1 | `title` | Original / consolidated |
| 2 | `learning-objectives` | Original / consolidated |
| 3 | `outline-heads` | Original / consolidated |
| 4 | `recurrent-attention-recap` | Original / consolidated |
| 5 | `context-for-text` | Component / paper source |
| 6 | `self-attention` | Component / paper source |
| 7 | `attention-roles` | Component / paper source |
| 8 | `learned-projections` | Component / paper source |
| 9 | `gpt-architecture` | Component / paper source |
| 10 | `position-inputs` | Component / paper source |
| 11 | `sinusoidal-frequencies` | Component / paper source |
| 12 | `sinusoidal-example` | Component / paper source |
| 13 | `position-demo` | Original / consolidated |
| 14 | `rope-rotation` | Component / paper source |
| 15 | `rope-complex-pairs` | RoPE reading extension |
| 16 | `rope-pair-frequencies` | RoPE reading extension |
| 17 | `rope-relative-offset` | Component / paper source |
| 18 | `position-methods` | Component / paper source |
| 19 | `outline-blocks` | Original / consolidated |
| 20 | `scaled-dot-product` | Component / paper source |
| 21 | `attention-demo` | Original / consolidated |
| 22 | `practice-01` | Original / consolidated |
| 23 | `why-multiple-heads` | Component / paper source |
| 24 | `multi-head-attention` | Component / paper source |
| 25 | `attention-in-architecture` | Component / paper source |
| 26 | `residual-addition` | Component / paper source |
| 27 | `layernorm-features` | Component / paper source |
| 28 | `normalization-order` | Component / paper source |
| 29 | `add-norm-practice` | Component / paper source |
| 30 | `feedforward-network` | Component / paper source |
| 31 | `why-ffn` | Component / paper source |
| 32 | `ffn-practice` | Component / paper source |
| 33 | `head-shapes` | Original / consolidated |
| 34 | `split-heads` | Original / consolidated |
| 35 | `exercise-01` | Original / consolidated |
| 36 | `block-code` | Original / consolidated |
| 37 | `outline-working` | Original / consolidated |
| 38 | `input-tokenization` | Component / paper source |
| 39 | `toy-configuration` | Original / consolidated |
| 40 | `learned-positions` | Original / consolidated |
| 41 | `block-parameters` | Original / consolidated |
| 42 | `full-model` | Original / consolidated |
| 43 | `exercise-04` | Original / consolidated |
| 44 | `causal-visibility` | Original / consolidated |
| 45 | `logits-and-loss` | Original / consolidated |
| 46 | `exercise-05` | Original / consolidated |
| 47 | `irreducible-toy-loss` | Original / consolidated |
| 48 | `tiny-training-loop` | Original / consolidated |
| 49 | `norm-comparison` | Original / consolidated |
| 50 | `attention-cost` | Original / consolidated |
| 51 | `original-and-baseline` | Original / consolidated |
| 52 | `original-transformer-paper` | Component / paper source |
| 53 | `paper-experimental-setup` | Component / paper source |
| 54 | `paper-parameter-count` | Component / paper source |
| 55 | `paper-translation-results` | Component / paper source |
| 56 | `paper-ablations` | Component / paper source |
| 57 | `paper-constituency-parsing` | Component / paper source |
| 58 | `paper-efficient-attention` | Component / paper source |
| 59 | `recap` | Original / consolidated |
| 60 | `next-steps` | Original / consolidated |
| 61 | `implementation-readings` | Original / consolidated |
| 62 | `references` | Component / paper source |

## Revision record and validation

**First merge, October 10, 2026 (Asia/Shanghai):** all 31 source sections were
copied byte-for-byte, with their relative order and required adjacent groups.
The source-deck SHA-256 for that historical snapshot is:

```text
716d6b95b0758aa16394392da4e68b34ba9dbb8028e733b6c07fc2b4df35d82f
```

**Second revision, same date:** the instructor authorized adapting the main
deck and its notebooks to the Noa example. Source-deck equality is therefore a
historical first-merge result, not a requirement for the adapted sections.
Current checks retain all prior 60 IDs and source-topic order, required adjacent
bundles, exercise routing, reveal counts, and unchanged historical evidence.
The original test deck remains untouched by the Noa revision. Its second-round
baseline SHA-256 is `72c7c1b0c9432f103ac5c0893084ee3da7e830250153fe11dfd3b83c8921dc14`.
Paper layout/wording differences already present between that baseline and the
first integration are retained; the only new paper illustration change is the
Noa parsing example. Numerical historical reports are unchanged.

The training-data change required a new measured normalization curve. See
[asset provenance](assets/README.md) for the current notebook hash, run settings,
results, and archived earlier experiment. The new model has 4,608 tied or 4,800
untied parameters and an irreducible mean loss of log(2)/10 on its two sentences.

Reproduce numerical and rendering checks with:

```sh
npm --prefix slides run check -- lecture-05
npm --prefix slides run pdf -- lecture-05
uv run python -m pytest tests/test_lecture_05.py
node --test tests/test_lecture_05_test_positions.mjs
```

**RoPE reading extension, October 14, 2026 (Asia/Shanghai):** two new slides
connect complex multiplication and per-pair frequencies to the existing
Noa/she example. Core E02 adds independent complex-arithmetic checks; the
implementation model and recorded training curves are unchanged. The timed
route moves the permutation demonstration to follow-up to make room.

Validation: 75 focused Python checks and 10 JavaScript checks passed. Both
notebooks ran in fresh kernels (14 core and 15 implementation code cells).
Browser checks passed for all 62 slides at three viewport sizes, including
the new reveals, Previous, and Reset. All slide images and the 62-page PDF
were reviewed. The original 60 IDs retain their order, and the archived
source, implementation notebook, and training curves are unchanged.
Review output belongs under ignored `slides/.checks/lecture-05/` and
`slides/.checks/lecture-05-rope/`. Classroom timing still needs a rehearsal.
