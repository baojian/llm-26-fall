# Lecture 05 Transformer component test

Thirty-one slides extending selected topics from the instructor's Spring 2026 PowerPoint.
The opening language example motivates the self-attention calculation.
Positional encoding is the first component walkthrough after the architecture overview.

| Test slide | Source page | Topic | Reveal steps |
| --- | --- | --- | --- |
| 1 | Updated Desktop pages 4–6 | “she” refers to Noa; causal visibility; context vector | 3 |
| 2 | 24 | Self-attention for “the” in “Bank of the river” | 8 |
| 3 | Updated Desktop pages 5–7 | Three roles for the same embeddings; why learn projections | 2 |
| 4 | 29 | Learned query, key, and value projections | 5 |
| 5 | 40, adapted | GPT-style causal Transformer architecture | 5 (new) |
| 6 | 51–54, redesigned | Token identity and position as separate inputs | 3 |
| 7 | 51–53, redesigned | Sinusoidal frequencies: fast and slow clocks | Hover over curves |
| 8 | New worked example | Practice E01: build and add a position vector | 2 |
| 9 | RoFormer, expanding page 54 | RoPE: rotate each query/key coordinate pair | Position slider |
| 10 | RoFormer, new example | Practice E02: shared shifts and relative offsets | 2 + shift/gap controls |
| 11 | New comparison | Where sinusoidal addition and RoPE enter the model | Complete diagram |
| 12 | Desktop page 38, left | One head: scores, scaling, causal mask, softmax, values | 4 |
| 13 | Updated Desktop pages 2–3 | Why multiple heads: different mixtures of the same prefix | 2 |
| 14 | Desktop pages 38–39 | Parallel heads, concatenation, output projection; E03 | 5 |
| 15 | Desktop pages 39–40 | Attention in the original architecture and GPT-style model | 3 |
| 16 | Updated Desktop page 10 | Residual addition: the direct path and its gradient | 1 |
| 17 | Updated Desktop page 11 | LayerNorm statistics over one token's features | 1 |
| 18 | Pages 10–11, clarified | Post-LN versus the course's pre-LN model | 1 |
| 19 | New worked example | Practice E04: normalize one token, add an update, check the axis | 1 |
| 20 | Updated Desktop page 6 | FFN expansion, activation, projection, and shared parameters | 3 |
| 21 | New explanation | Why the FFN matters; two thirds of standard block matrix parameters | 1 |
| 22 | New worked example | Practice E05: FFN arithmetic, token independence, and nonlinearity | 1 |
| 23 | Desktop pages 41–43 | Left: inputs/subwords; right: tokenizer algorithms and two stages | 4 |
| 24 | Latest snapshot pages 4 and 16 | Original 2017 Transformer and the supported RNN comparison | Complete slide |
| 25 | Latest snapshot pages 5 and 8 | Translation datasets, base/big settings, and training budget | Complete slide |
| 26 | Latest snapshot page 9 | Practice E06: encoder-only parameter count with explicit assumptions | 2 |
| 27 | Latest snapshot pages 6–7 | Test BLEU and estimated training FLOPs; single versus ensemble comparison | 1 |
| 28 | Latest snapshot page 10 | Development-set ablations and the PPL/BLEU distinction | Complete slide |
| 29 | Latest snapshot pages 11–12 | Constituency parsing and results in different data regimes | Complete slide |
| 30 | Latest snapshot pages 13–14 | Reformer and Linformer as later efficiency methods | Complete slide |
| 31 | Latest snapshot pages 15 and 17 | Modification transfer, experimental interpretation, and readings | Complete slide |

The final eight slides use pages 4–17 of the 17-slide Desktop snapshot read on
October 10, 2026 (Asia/Shanghai), SHA-256
`484b3f8ea6386c40e50abaf9b21f530371df1099680afd6595386b30b2022eb6`.
They summarize the original paper's evidence after the component walkthrough.
Allow about 20 minutes including the three-minute encoder count. Longer
tables, training details, and supplementary resources remain in speaker notes.
The dated citation-count screenshot and obsolete next-lecture footer are omitted.

The source's “50 times faster” statement is clarified as about 52 times fewer
estimated training FLOPs than the ConvS2S **ensemble**, or about 6.5 times fewer
than its single model, using the English–French column in Table 2. Neither is
an inference-speed measurement. Table 2, rather than the source's inconsistent
bar chart, supplies the displayed BLEU values. The RNN comparison no longer
claims that Transformers eliminate vanishing/exploding gradients or always
need fewer training steps. The 2020 and 2021 follow-ups are labeled separately
from the original 2017 contribution.

E06 retains the instructor's 12-layer, width-768, 12-head, 30,522-token
encoder example. Its answer, **108,495,360**, includes linear biases and two
affine LayerNorms per layer, fixed positions, and no prediction head or extra
embedding components. The notebook counts explicit tensor shapes; the Python
check compares the result with PyTorch modules on the meta device, without
allocating model weights. The earlier E01–E05 remain unchanged.

- [Start the eight-slide paper section](http://127.0.0.1:8005/slides/lecture-05-test/#/original-transformer-paper)
- [Encoder count](http://127.0.0.1:8005/slides/lecture-05-test/#/paper-parameter-count)
- [Translation evidence](http://127.0.0.1:8005/slides/lecture-05-test/#/paper-translation-results)

Source page numbers describe the revision used when preparing each section.
The opener uses the 49-slide Desktop revision; the three-role bridge uses the
later 43-slide revision, both inspected on October 10, 2026 (Asia/Shanghai).
The multi-head intuition uses the later 26-slide Desktop revision inspected
on the same date. The Add & Norm section uses pages 10–11 of the subsequent
25-slide revision. The FFN section uses page 6 of the subsequent 19-slide
revision. In the 43-slide revision, page 4 is the existing self-attention calculation and
page 9 is the existing learned-projection diagram. Earlier rows retain the page numbers from the original source
and the earlier 83-slide Desktop copy. The builders reproduce their artwork
without reading the changing PowerPoint file.

The modern version uses bullet explanations, SVG shapes, and KaTeX equations
with the course's existing fonts. Page 29 groups the original per-word
projections into three matrix branches. All matrices and vectors are schematic.
Speaker notes explain the dimensions and the row-vector convention.
The page 40 adaptation uses a GPT-2-style causal decoder with learned position
embeddings, LayerNorm before attention and the MLP, two residual connections
per block, a final LayerNorm, and a vocabulary head. A return arrow explains
how generation appends a token and repeats. The illustrative continuation is
“river”; it is not a model result.
The architecture runs upward: separate token and positional embeddings at
the bottom are added before the stack, and the language-model head is at the top.

```sh
uv run python scripts/slides.py serve --port 8005
```

- [Slide 1: context for a token](http://127.0.0.1:8005/slides/lecture-05-test/#/context-for-text)
- [Slide 2: self-attention](http://127.0.0.1:8005/slides/lecture-05-test/#/self-attention)
- [Slide 3: one input, three roles](http://127.0.0.1:8005/slides/lecture-05-test/#/attention-roles)
- [Slide 4: learned projections](http://127.0.0.1:8005/slides/lecture-05-test/#/learned-projections)
- [Slide 5: GPT architecture](http://127.0.0.1:8005/slides/lecture-05-test/#/gpt-architecture)
- [Start the position section](http://127.0.0.1:8005/slides/lecture-05-test/#/position-inputs)
- [Sinusoidal frequencies](http://127.0.0.1:8005/slides/lecture-05-test/#/sinusoidal-frequencies)
- [RoPE rotation](http://127.0.0.1:8005/slides/lecture-05-test/#/rope-rotation)
- [RoPE relative offsets](http://127.0.0.1:8005/slides/lecture-05-test/#/rope-relative-offset)
- [Start the multi-head section](http://127.0.0.1:8005/slides/lecture-05-test/#/scaled-dot-product)
- [Why multiple heads?](http://127.0.0.1:8005/slides/lecture-05-test/#/why-multiple-heads)
- [Parallel attention heads](http://127.0.0.1:8005/slides/lecture-05-test/#/multi-head-attention)
- [Residual addition](http://127.0.0.1:8005/slides/lecture-05-test/#/residual-addition)
- [LayerNorm](http://127.0.0.1:8005/slides/lecture-05-test/#/layernorm-features)
- [Pre-LN and post-LN](http://127.0.0.1:8005/slides/lecture-05-test/#/normalization-order)
- [Add & Norm practice](http://127.0.0.1:8005/slides/lecture-05-test/#/add-norm-practice)
- [Feedforward network](http://127.0.0.1:8005/slides/lecture-05-test/#/feedforward-network)
- [Why the FFN matters](http://127.0.0.1:8005/slides/lecture-05-test/#/why-ffn)
- [FFN practice](http://127.0.0.1:8005/slides/lecture-05-test/#/ffn-practice)
- [Input/tokenizer recap](http://127.0.0.1:8005/slides/lecture-05-test/#/input-tokenization)
- [Original page 29 reproduction](http://127.0.0.1:8005/slides/lecture-05-test/?design=original#/learned-projections)

Use **Space / Right**, **Next**, or click the diagram to advance.
**Left / Previous** reverses a stage. These controls cross to the adjacent slide
at the end or beginning of its reveals. **Reset** restarts the current slide.
**Original / Modern** switches designs and keeps the current slide.
On slide 5, **Reference** opens the linked full GPT diagram as a static
comparison; **Modern** returns to the simplified teaching diagram.
The position section uses its new diagrams in either design. Its RoPE slider
changes the rotation angle. **Shift both +1** moves both absolute positions;
**Gap +1** changes their separation. **Reset** also restores the local example.
**Print** shows all thirty-one slides with answers, one per page, with a complete
default example for each interactive diagram.

## Opening language example and Lecture 05 integration

Allow 1–2 minutes. Ask which earlier word identifies who “she” refers to, then
reveal Noa, the causal boundary, and the weighted context. The link is a human
interpretation of the sentence; model attention weights are learned. The words
stand for toy token positions. “is a great cat” lies beyond the available prefix.
The following four-word worked example deliberately uses unmasked attention to
isolate the arithmetic. Speaker notes make that transition explicit.

For the eventual move into `lecture-05`, keep the stable `context-for-text` ID
immediately before `self-attention`. Move its Markdown section,
`build-context-intuition.mjs`, and `assets/context-intuition.{svg,json}` with the
existing diagram loader, KaTeX styles, and Reveal controls. All paths are relative
to the deck, and the new builder does not depend on the `lecture-05-test` folder
name. The browser checks locate slides by ID so inserting or moving this section
preserves its navigation checks. Exercise IDs and the notebook are unchanged.

## From the attention calculation to learned projections

Allow 1 minute for the `attention-roles` bridge after `self-attention`.
One reveal names the query, key, and value uses in the existing four-word
example. The second motivates learning a separate representation for each role.
The following `learned-projections` slide develops the matrices and dimensions.

The source's “No weights for model to train” is qualified: this simplified
attention operation has no learned Q/K/V projections. The embeddings can still
be trained, and attention weights are computed from the input representations.
Keep this bridge, `build-attention-roles.mjs`, and
`assets/attention-roles.{svg,json}` together when integrating Lecture 05.
It uses the same local diagram loader and KaTeX styles as the opener.

## Position section: teaching plan

Allow roughly 20 minutes including the two ungraded exercises. Speaker notes
include the reveal sequence, checked answers, source sections, and common mistakes.

1. Identify two inputs: token identity and position. Read the diagram upward.
2. Build the 2017 sinusoidal code from pairs with different frequencies.
3. Calculate a four-dimensional code and add it to an invented embedding.
4. Rotate each query/key pair; leave values unchanged. Move the position slider.
5. Hold content vectors fixed and compare equal gaps at different absolute positions.
6. Return to the architecture: additive input vectors versus rotations inside attention.

The original Transformer scales token embeddings by `sqrt(d_model)` before
adding its fixed sinusoidal code. The worked example calls that scaled vector
`e`. The earlier GPT-2 overview uses learned additive position embeddings;
RoPE instead belongs after Q/K projections in each attention layer. The
comparison diagram retains the causal mask and shows V bypassing rotation.

RoPE's circle examples explicitly use a **toy frequency of 30° per token**.
The general formula uses the standard baseline `10000^(-2i/d_h)`. Equal
offsets preserve the positional dot product for fixed content vectors; this
does not claim that changing a prompt leaves a whole model's outputs unchanged.

[Practice notebook](practice.ipynb) contains E01–E06 in slide order. Use
the toolbar's Notebook link to open a personal working copy in JupyterLab.
It needs only Python's standard library and includes the numerical checks.
Existing personal notebooks are preserved. To open the updated handout, rename
an older working copy in JupyterLab, then use Notebook again.

## Multi-head attention and input recap

The section combines **pages 38–43 of the earlier Desktop copy** with
pages 2–3 of its updated 26-slide revision. Allow roughly 16 minutes,
including a one-minute dimension check.

1. Unpack one head, keeping the source's upward order. The causal mask acts on
   logits before softmax; the value path goes directly to the weighted sum.
2. Motivate multiple heads with the source's “I gave my dog Charlie some food”
   sentence. The query is at the final word so all illustrated keys are causally
   available. Three invented weight rows emphasize different parts of the prefix.
   Each row sums to one; heads' linguistic roles are illustrative and can overlap.
3. Repeat the operation with separate Q/K/V projections, concatenate features
   per token, and apply `W^O`. E03 checks the original base model's 512 / 8 = 64
   head width and the unchanged number of token rows.
4. Show encoder self-attention, decoder causal self-attention, and cross-attention,
   alongside the course's GPT-style causal path. Residuals and normalization are
   omitted in this attention-only comparison and explained in speaker notes.
5. Combine source pages 41–42 on the left (text, subwords, IDs, embeddings) and
   page 43 on the right (BPE, Unigram, WordPiece; learner and segmenter).

Page 41's wording is corrected: embeddings can be trained with the Transformer;
they do not require a separate pretrained embedding model. The token example
and IDs are explicitly invented. The right-hand vocabulary/rules output feeds
the segmenter; tokenizer fitting is distinct from neural embedding training.

### Coverage of the updated Desktop pages 2–7

This map refers to the **26-slide revision** inspected on October 10, 2026
(Asia/Shanghai), before further source deletions or reordering.

| Source pages | Decision | Location in the test deck |
| --- | --- | --- |
| 2–3: motivation and multiple relations | Condensed into one new slide with a causal example | `why-multiple-heads` |
| 4: per-head projections and combination | Already covered | `multi-head-attention` |
| 5: stacked layers | Already drawn; notes now distinguish parallel heads from sequential layers | `gpt-architecture`, `attention-in-architecture` |
| 6: scaling and mask | Already covered, including the current token in the causal prefix | `scaled-dot-product` |
| 7: head pruning | Retained as optional discussion and a verified paper link | Notes under `multi-head-attention` |

The optional paper is [Michel, Levy, and Neubig (NeurIPS 2019)](https://papers.neurips.cc/paper_files/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf),
§§3–5. Its pruning results concern trained models. Keep the scope of its
single-layer ablations explicit when discussing the result.

Move `why-multiple-heads`, `build-multihead-intuition.mjs`, and
`assets/multihead-intuition.{svg,json}` together when integrating Lecture 05.
The builder uses relative paths and needs no access to the source PowerPoint.

## Residual addition and LayerNorm

The four-slide section after the attention architectures adapts **pages 10–11
of the 25-slide Desktop revision**, inspected on October 10, 2026
(Asia/Shanghai). Allow 8 minutes including the two-minute E04 practice.
It uses the shared text, equation, table, and reveal layouts.

1. **Residual addition:** learn an update while keeping the input as a direct
   path. Shapes stay `(B,T,d)`. The Jacobian has an identity contribution;
   this does not guarantee stable total gradients.
2. **LayerNorm:** compute population mean and variance over one token's
   features, then apply learned scale and bias. Statistics are independent
   across tokens, and affine outputs need not have zero mean or unit variance.
3. **Placement:** compare `LN(x + F(x))` with `x + F(LN(x))`. The source code
   is pre-LN, although the original Transformer's “Add & Norm” is post-LN.
   Our GPT-style model keeps a final LayerNorm after the stack.
4. **E04:** normalize `(1,3)` and add the supplied update `(2,-1)` to the
   original input. The notebook checks the output `(3,2)`, token independence,
   constant features, and learned affine parameters, using only the CPU.

Correct the source's ResNet venue to **CVPR 2016** (the preprint is from 2015).
The source's historical `std + eps` code is not an exact implementation of
PyTorch LayerNorm: use population variance and `sqrt(var + eps)`. Keep the
training motivation, with its dependence on model and optimization settings;
omit blanket promises of faster training or better performance.

Primary readings:

- [He et al., residual learning, §3.2](https://arxiv.org/abs/1512.03385).
- [Ba et al., Layer Normalization, §3](https://arxiv.org/html/1607.06450v1#S3).
- [Vaswani et al., §3.1](https://arxiv.org/html/1706.03762v7#S3.S1) and
  [Xiong et al., §§2–4](https://proceedings.mlr.press/v119/xiong20b.html).
- [PyTorch LayerNorm formula and implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L88).

For eventual integration into `lecture-05`, replace or expand its existing
residual/normalization section with these explanations rather than duplicating
it. Keep the stable slide IDs and adapt E04 to that deck's existing E03
normalization exercise. The original Desktop PowerPoint is unchanged.

## Feedforward network

Allow 7 minutes, including E05. The three-slide section explains the FFN
named in the architecture, using page 6 of the 19-slide Desktop revision.

1. Apply the same expansion, activation, and projection to every token.
   The diagram labels ReLU for the original Transformer and GELU for GPT-2.
2. Explain nonlinear feature computation and derive the parameter share:
   with one standard attention sublayer and `d_ff = 4d`, attention matrices
   have `4d²` weights and FFN matrices `8d²`. The FFN thus has two thirds of
   these block weights. Whole-model counts also include other components.
3. E05 computes a 2 → 3 → 2 FFN. The notebook verifies output `(5,-4)`, token
   independence, nonlinearity, and the parameter counts.

Keep `feedforward-network`, `why-ffn`, and `ffn-practice` together during
integration. Copy the builder and its three assets, and preserve the existing
E01–E04 handout content when adding E05. The original working PowerPoint is
unchanged. Source equations and architecture choices are linked in the notes.

## Editable sources

- [Context introduction builder](build-context-intuition.mjs): one editable SVG
  and three-step manifest, adapted from updated Desktop pages 4–6. The diagram
  uses native text and vector paths, with a KaTeX context equation.
- [Three-role bridge builder](build-attention-roles.mjs): one editable SVG and
  two-step manifest, adapted from pages 5–7 of the updated 43-slide Desktop
  revision. It reuses the previous example and leads into the learned matrices.
- [Multi-head intuition builder](build-multihead-intuition.mjs): a portable SVG
  with two reveals and three checked, invented attention-weight rows. Adapted
  from pages 2–3 of the 26-slide Desktop revision.
- [FFN builder](build-ffn.mjs): a token-wise SVG, three-reveal manifest, and
  a vector bar showing the 1:2 attention/FFN block parameter ratio.
- [Modern builder](build-modern.mjs): all diagrams, bullets, and TeX equations.
  It generates `assets/self-attention-modern.svg` and
  `assets/self-attention-29-modern.svg`, and `assets/self-attention-40-modern.svg`, rendered inside the course page with
  local KaTeX styles. No raster artwork is used in the modern version.
- [Modern page 24 sequence](assets/animation-modern.json) and
  [page 29 sequence](assets/animation-29-modern.json), plus the new
  [GPT architecture sequence](assets/animation-40-modern.json).
- [Full GPT reference](assets/gpt-architecture-reference.svg): the unmodified
  diagram by Marxav / Mrmw, downloaded from Wikimedia Commons (CC0 1.0).
- [Native source page 24](assets/source-slide24.pptx) and
  [native source page 29](assets/source-slide29.pptx): each retains its original
  slide XML, editable objects, and animation timing.
- [Original importer](import-slide.py): extracts source artwork and click groups
  without changing the input PowerPoint. Use `--slide 24` or `--slide 29`.
  These page numbers require the original source revision, before reordering.
  The original browser diagrams combine vector shapes with the source's
  embedded PNG text and equation artwork.
- [Page 24 provenance](assets/animation.json) and
  [page 29 provenance](assets/animation-29.json): source object IDs and hashes.
- [Position builder](build-positions.mjs): four editable SVG diagrams and a
  Plotly specification computed from the sinusoidal formula.
- [Position mathematics](position-math.js) and
  [browser controls](position-interactions.js): rotations, dot products, and
  the shared-shift demonstration. All artwork is native SVG, HTML, or Plotly;
  the new section uses no raster screenshots.
- [Multi-head and tokenizer builder](build-multihead.mjs): four editable SVG
  redraws with KaTeX, click manifests, and source-page provenance. Rebuilding
  these assets does not need the instructor's PowerPoint file.

```sh
node slides/lecture-05-test/build-context-intuition.mjs
node slides/lecture-05-test/build-attention-roles.mjs
node slides/lecture-05-test/build-multihead-intuition.mjs
node slides/lecture-05-test/build-ffn.mjs
node slides/lecture-05-test/build-modern.mjs
node slides/lecture-05-test/build-positions.mjs
node slides/lecture-05-test/build-multihead.mjs
node --test tests/test_lecture_05_test_positions.mjs
uv run python -m pytest -q tests/test_lecture_05_test_paper.py
npm --prefix slides run check -- lecture-05-test
npm --prefix slides run pdf -- lecture-05-test
npm --prefix slides run check:notebook -- http://127.0.0.1:8005 lecture-05-test
```

Checks cover every reveal in both designs, backward navigation, slide
boundaries, diagram clicks, Reset, and switching designs. They also check
KaTeX rendering, clipping, and the absence of raster assets in the modern
version. The position checks cover exercise answers, keyboard operation of
the slider, shared shifts, changing gaps, Reset, and deep fragment links.
The four newer slides also check forward/backward reveals, diagram clicks,
Reset, deep links, equation clipping, and the value/cross-attention/tokenizer paths.
The opener also checks each reveal, the first-slide Previous state, navigation
into the calculation, Reset, and restoring a deep fragment link.
The three-role bridge checks both reveals and navigation between the calculation
and learned projections in either design.
The multi-head intuition checks its reveals, causal key positions, normalized
weight rows, Reset, and navigation into the projection diagram.
The Add & Norm slides check their answer reveals with the same toolbar;
notebook E04 also checks the numerical example and normalization axis.
The FFN section checks its reveals, native equations, parameter-share figure,
and E05's arithmetic, nonlinearity, and token independence.
Review screenshots and the thirty-one-page PDF are saved under
`slides/.checks/lecture-05-test/`.

The source is Baojian Zhou's Spring 2026
`lecture-05-slides-transformers.pptx`, pages 24 and 29. Both slides attribute
the example to [Rasa Algorithm Whiteboard 2](https://www.youtube.com/watch?v=tIvKXrEDMhk).

For page 40, the visual reference is
[Full GPT architecture](https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.svg).
The normalization order, residual additions, causal mask, learned position
embeddings, and vocabulary projection follow
[OpenAI's GPT-2 implementation](https://github.com/openai/gpt-2/blob/master/src/model.py).
The simplified diagram groups attention and MLP internals and omits dropout.

The position section adapts the teaching topic of Spring source pages 51–54.
Its figures and numerical examples are newly created. Primary readings:

- [Vaswani et al., Attention Is All You Need, §§3.4–3.5](https://arxiv.org/html/1706.03762v7#S3.S5).
- [Su et al., RoFormer, §§3.2.1–3.2.2, equations 13–16](https://arxiv.org/html/2104.09864v5).
- Caveat from source page 54: [Haviv et al., NoPos](https://arxiv.org/abs/2203.16634).

The multi-head and tokenizer section uses the Desktop copy's page numbering.
It redraws the diagrams in the established style; the original input file is
unchanged. Primary readings for this section:

- [Attention Is All You Need, §§3.1–3.2.3 and §3.4](https://arxiv.org/html/1706.03762v7#S3.S2).
- [Sennrich et al. (2016), neural MT with subword units](https://aclanthology.org/P16-1162/).
- [Kudo (2018), subword regularization](https://aclanthology.org/P18-1007/).
- [Schuster & Nakajima (2012), Japanese and Korean voice search](https://research.google/pubs/japanese-and-korean-voice-search/).
