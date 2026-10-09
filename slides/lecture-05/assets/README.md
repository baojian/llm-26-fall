# Lecture 05 assets

All diagrams and numerical figures are original course teaching assets. The
sentences, token IDs, and browser vectors are invented toy data. No third-party
figure screenshots, datasets, model weights, or student submissions are copied.

| Files | Content and provenance |
| --- | --- |
| `encoder-context.svg` and `.excalidraw` | Fixed context versus a context selected at each target step; follows Bahdanau et al. (2015), §§2–3 |
| `multihead-path.svg` and `.excalidraw` | Q/K/V projections, head split, per-head attention, merge, and output projection; follows the mechanism in Vaswani et al., §3.2.2 |
| `residual-block.svg` and `.excalidraw` | Two pre-LN residual sublayers; distinguishes normalization placement from the original post-LN model |
| `decoder-model.svg` and `.excalidraw` | Token/position embeddings, two decoder blocks, final norm, and tied vocabulary readout used in the notebook |
| `causal-mask.svg` and `.excalidraw` | The four input slots and shifted next-token targets; the diagonal is allowed |
| `attention-values.json` and `attention-demo.json` | Three-position causal example moved from Lecture 04; values updated to match the Lecture 05 worked example |
| `position-values.json` | Four word labels, identity-matrix vectors, and the permutation `[3,1,2,0]`; matches E02 |
| `position-demo.json` | Printable initial Plotly state: original order, positions off |
| `position-frequencies.json` | Formula-generated sine coordinates for width 8 over positions 0–31; frequencies 1, 0.1, and 0.01 radians per position |
| `norm-comparison.json` | Measured pre/post-LN loss at every update 0–200, plus the `log(2)/4` floor; includes machine-readable provenance in `layout.meta` |

The diagrams are editable Excalidraw scenes, exported with
`@excalidraw/excalidraw` 0.18.1. They follow the previous lectures' canvas,
font sizes, and palette. Edit the scene and export its SVG alongside it; the
classroom browser loads only the SVG. Sources for the adapted concepts are
[Bahdanau et al. (2015), §3](https://arxiv.org/abs/1409.0473),
[Luong et al. (2015), §3](https://aclanthology.org/D15-1166/),
[Vaswani et al. (2017), §§3.1–3.5](https://arxiv.org/abs/1706.03762),
[Xiong et al. (2020)](https://proceedings.mlr.press/v119/xiong20b.html), and
the instructor's 2025 Lecture 05. The [source map](../teaching-plan.md#map-from-all-85-source-slides)
accounts for that whole deck.

## Browser causal example

[attention-demo.js](../attention-demo.js) computes the head and figure;
[causal-demo.js](../causal-demo.js) wires the query/mask/value/reset controls.
Query 2 uses one-based chart labels and corresponds to zero-based slot 1 in
slides 8–9. Its scores are `(1,0,1)` and values `(1,0)`, `(0,2)`, `(2,1)`.
The initial masked output is `(0.731059,0.537883)`. Changing V3 adds `(10,10)`;
Q2 is unchanged with the mask and changes without it. Reset restores the initial
state, also used for printing. All assets are local and both demos initialize
through [demo.js](../demo.js).

## Browser position example

[position-demo.js](../position-demo.js) computes the attention values and
figure. [demo.js](../demo.js) wires native buttons to Plotly. Reset restores
both position and swap state, and the accessible description reports the
current order and numerical difference.

The vectors are rows of `I4`, with identity Q/K/V projections, scale 2, and no
mask. The initial self weight is `exp(0.5)/(exp(0.5)+3) ≈ 0.354661`, and each
other weight is about 0.215113. Swapping bank/river and restoring output order
gives zero change without positions. Adding sinusoidal vectors at fixed slots
gives maximum change about 1.365050. The illustration tests order information;
the word labels imply no trained semantics. `position-demo.json` is generated
by `figureFor(fixture, initialState)`; the Node test verifies exact agreement.

## Measured normalization comparison

The [teaching plan](../teaching-plan.md#exact-teaching-model) states the complete
configuration. Both variants receive the same copied initial parameters,
including the final LayerNorm. Only block normalization order changes.
The initial reference run used Python 3.11.14, PyTorch 2.14.0, macOS on arm64,
CPU float32, one thread, seed 7, and 200 AdamW updates per variant. No model or
data files are downloaded. Full-batch size is 2, sequence length 4, model width
16, heads 4, FFN width 32, and depth 2. The learning rate is 0.02, betas are
(0.9, 0.999), epsilon is 1e-8, and weight decay is zero.

| Update | Pre-LN loss (nats/token) | Post-LN loss (nats/token) |
| ---: | ---: | ---: |
| 0 | 1.9754 | 1.9730 |
| 20 | 0.3644 | 0.3636 |
| 50 | 0.3465 | 0.3473 |
| 100 | 0.3599 | 0.1744 |
| 200 | 0.1739 | 0.1735 |

Index 0 is before training; index 200 follows 200 completed updates. The figure
retains all 201 points, including transient loss spikes. Post-LN reached a low
loss earlier in this run. Neither this observation nor the final losses
establishes a general ranking. The two equally frequent targets after BOS
impose the loss floor `log(2)/4 ≈ 0.173287`. There is no held-out evaluation.
Small numerical differences can change the trajectory around the plateaus;
do not interpret one platform's exact transition update as a robust result.

## Reproduce and review

```sh
uv run python slides/lecture-05/prepare-figures.py
uv run python -m pytest tests/test_lecture_05.py
node --test tests/test_position_demo.mjs
npm --prefix slides run pdf -- lecture-05
```

The preparation script executes the committed notebook and refreshes the loss
figure. It records the command, Git revision and dirty state, notebook hash,
Asia/Shanghai timestamp, environment, seed, compute budget, and elapsed time in
the figure's metadata. Full loss/gradient histories and cell output stay in
ignored `slides/.checks/lecture-05/figure-run.json`. This small CPU run usually
takes a few seconds; that elapsed time is a reproduction aid, not a benchmark.
Review the resulting chart before committing a regenerated asset.

Numerical tests independently verify the position formula, outputs, shapes,
parameter counts, PyTorch attention agreement, and full-model causal gradients.
The browser checker verifies the controls and all three viewport sizes. The
PDF retains the initial position example and measured training figure.
