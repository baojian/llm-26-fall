<!-- .slide: class="title-slide" id="title" -->

<p class="eyebrow">CS40008.01</p>

# Attention and the Transformer

<p class="subtitle">Lecture 05 – NLP and LLMs</p>

<p class="byline">Baojian Zhou<br>School of Data Science<br>Fudan University<br>October 10, 2026</p>

Note:
Allow 1 minute. Saturday make-up class. Three 45-minute periods, with 15 minutes reserved for Quiz 2. The five E exercises are ungraded practices. Quiz 2 assesses previously taught material and is administered separately. Time and room follow the university notice. Adapted from the instructor's 2025 Lecture 05, with the source map in teaching-plan.md.

---

<!-- .slide: id="learning-objectives" -->

## What you will be able to do

**How does attention select useful context, and how does it lead to the Transformer?**

- Connect the encoder–decoder bottleneck to learned alignment.
- Trace multiple heads and the complete decoder block.
- Explain positions, residual paths, normalization, and the FFN.
- Check a small model's parameters, gradients, and causality.
- Interpret a controlled comparison on toy data.

Note:
Allow 2 minutes. The notebook contains the full code; slides isolate the important operations. No datasets or model weights need downloading. Predictions before execution are the expected classroom response.

---

<!-- .slide: class="outline-slide" id="outline-heads" -->

## Outline

<ul class="outline-topics">
<li aria-current="step">From recurrence to attention</li>
<li>Positions and Transformer blocks</li>
<li>A working language model</li>
</ul>

<p class="caption">Period 1 · 45 minutes, including E01</p>

Note:
Allow 1 minute. The revised first-four-lecture recap ends with RNNs, LSTMs, and their limitations. Introduce attention here without requiring students to have studied the attention extension in the published Lecture 04 deck. Start from recurrent context, build one head numerically, and then introduce multiple projections.

---

<!-- .slide: id="recurrence-limits" -->

## A recurrent state carries the past forward

$$h_t=f(h_{t-1},x_t)$$

- Computing $h_t$ requires the previous state.
- Distant information passes through many recurrent steps.
- LSTM gates help retain information, but the recurrence remains.

<p class="caption">The next step: let a decoder select from several source states.</p>

Note:
Allow 2 minutes. Recap Lecture 04 rather than deriving LSTM gates again. Sequential state dependencies restrict parallelism across positions in a conventional RNN; long paths can make credit assignment difficult. These are architectural and optimization considerations, not a claim that RNNs cannot learn long dependencies. Sources: 2025 slides 3–8 and 84; Bahdanau et al. (ICLR 2015), §§2–3, https://arxiv.org/abs/1409.0473; Vaswani et al., §4.

---

<!-- .slide: id="encoder-bottleneck" -->

## Replace one summary with selectable context

<img class="diagram" src="assets/encoder-context.svg" data-excalidraw-source="assets/encoder-context.excalidraw" alt="A fixed-context encoder compresses the source into one context vector for the decoder. With attention, the decoder selects a weighted combination of encoder states for each target step.">

<p class="caption">The context is still a vector, but its weights can change with the target step.</p>

Note:
Allow 3 minutes. The fixed-context case uses a final recurrent state as its summary. The attentional case keeps the source states available and produces a separate context c_t for each target step. In Bahdanau et al., source annotations come from a bidirectional RNN; the decoder and alignment model are trained jointly. This is still a recurrent encoder–decoder, before the Transformer removes recurrence from its blocks. Sources: 2025 slides 3–8; Bahdanau et al., §§2.1–3.1.

---

<!-- .slide: id="learned-alignment" -->

## Learn which source states to combine

$$e_{t,s}=a(q_t,h_s),\qquad
\alpha_{t,s}=\frac{\exp(e_{t,s})}{\sum_j\exp(e_{t,j})}$$

$$c_t=\sum_s\alpha_{t,s}h_s$$

$q_t$ is the decoder query; $h_s$ is a source state.

The alignment function $a$ is learned with the translation model.

Note:
Allow 3 minutes. Softmax is over source positions s. Bahdanau's query is the previous decoder state; its alignment network is additive, for example v^T tanh(W_q q_t + W_h h_s). Luong et al. (§3.1) compare dot, general/bilinear, and concat scores; their query uses the current decoder state. Do not conflate those state-update conventions. The common mechanism is score, normalize, and combine. Gradients pass through the weights; hard alignment labels are not required by these translation objectives. Sources: Bahdanau et al., §3.1 and Appendix A.1.2; Luong et al., §3, https://aclanthology.org/D15-1166/.

---

<!-- .slide: id="alignment-numbers" -->

## Calculate one weighted context

Use an unscaled dot score, $q=(1,0)$, and $h_1=(0,1)$, $h_2=(1,0)$.

| Source state | Score $q^\top h_s$ | Softmax weight |
| --- | ---: | ---: |
| $h_1$ | 0 | 0.2689 |
| $h_2$ | 1 | 0.7311 |

$$c=0.2689h_1+0.7311h_2\approx(0.7311,\ 0.2689)$$

<p class="caption">Invented vectors; dot scoring is one of Luong et al.'s alignment choices.</p>

Note:
Allow 3 minutes. Compute exp(0)/(exp(0)+exp(1)) and its complement, then combine both coordinates. The source states serve as both keys and values here. This example isolates aggregation; the vectors are supplied rather than produced by trained encoders. The notebook reproduces the exact calculation before E01. The Transformer head next adds learned Q/K/V projections and sqrt(d_k) scaling. Sources: Luong et al., §3.1; 2025 source slides 8 and 25.

---

<!-- .slide: id="context-recap" -->

## Queries select; values are combined

| Vector | Role in one attention head |
| --- | --- |
| Query $q_t$ | Scores candidate source positions |
| Key $k_s$ | Is compared with the query |
| Value $v_s$ | Contributes to the weighted output |

Self-attention obtains all three from the same sequence of states.

The three projection matrices are learned jointly with the model.

Note:
Allow 3 minutes. In the preceding alignment example, the same source state supplied both the key and the value. Separate projections let scoring and returned content use different learned features. The query is also a learned projection, not a human-written question. For example, bank in bank of the river can use river in an unmasked encoder; a causal decoder cannot use that future word at bank. No grammatical role or semantic coordinate is assigned to a head. Sources: 2025 slides 22–31; Vaswani et al., §§3.2.1–3.2.3.

---

<!-- .slide: id="attention-contract" -->

## One head: project, score, normalize, combine

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V$$

$$A=\operatorname{softmax}_{\text{keys}}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad O=AV$$

- A causal mask allows the current and earlier input positions.
- Queries and keys share $d_k$; values may have another width.

Note:
Allow 3 minutes. Read the computation left to right, without assuming prior attention work. With T input rows and model width d, W_Q and W_K have shape (d,d_k), and W_V has shape (d,d_v). QK^T is (T,T), softmax normalizes each query row over keys, and O is (T,d_v). The projections are shared across token positions. M is zero on allowed entries and negative infinity elsewhere; every query must retain an allowed key. Under independent zero-mean unit-variance Q/K coordinates, the unscaled dot product has variance d_k, motivating division by sqrt(d_k). This is an initialization argument, not a constraint maintained during training. Correct the source slide 29 dictionary lookup: d['b'] is 2, not 3. Source: Vaswani et al. (2017), §§3.2.1 and 3.2.3, https://arxiv.org/abs/1706.03762.

---

<!-- .slide: id="scaled-attention-numbers" -->

## Work through one query

Already-projected toy vectors: $q=(\sqrt{2},0)$ and $d_k=2$.

| Slot | Key $k_s$ | Value $v_s$ | Score $q^\top k_s/\sqrt{2}$ |
| --- | --- | --- | ---: |
| 0 | $(1,0)$ | $(1,0)$ | 1 |
| 1 | $(0,1)$ | $(0,2)$ | 0 |
| 2 | $(1,1)$ | $(2,1)$ | 1 |

$$\alpha=\operatorname{softmax}(1,0,1)
\approx(0.4223,0.1554,0.4223)$$

$$o=\sum_s\alpha_sv_s\approx(1.2670,0.7330)$$

Note:
Allow 3 minutes. The denominator is 2e+1. Derive the first output coordinate as 3e/(2e+1) and the second as (2+e)/(2e+1). Keys set the weights, while values determine what is averaged; they need not be the same vectors. Use the supplied projected vectors rather than inventing semantic meanings for their coordinates. The notebook's single-head-example reproduces these values in float64. Source for the operation: Vaswani et al., §3.2.1. The numbers are original teaching data.

---

<!-- .slide: id="mask-before-softmax" -->

## Mask future scores before softmax

The query is at input slot 1. Slots 0 and 1 are available; slot 2 is future.

$$\operatorname{softmax}(1,0,-\infty)
\approx(0.7311,0.2689,0)$$

$$o\approx0.7311(1,0)+0.2689(0,2)=(0.7311,0.5379)$$

Only allowed positions contribute to this query.

<div class="answer fragment">Changing the future value leaves this output unchanged. The allowed weights sum to one.</div>

<p class="source">Interactive example: <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener noreferrer">Transformer Explainer</a>.</p>

Note:
Allow 4 minutes. This is the same supplied query and the same K/V table, now with causal visibility. Slot 1 may read its own input while predicting slot 2's target: the input and next-token target are shifted. Setting the future weight to zero after the unmasked softmax would leave total weight about 0.5777 and is not the same operation. Explicit renormalization could repair that construction, but masking logits first is the direct stable implementation. Every row needs at least one allowed key; all-negative-infinity logits yield undefined softmax. The notebook verifies both weights, the future-value perturbation, an unmasked control, and zero derivatives to the future key and value. Source: Vaswani et al., §3.2.3.

Use the first 3 minutes for the numerical example and its counterexample. In the final minute, open the prepared Transformer Explainer tab: https://poloclub.github.io/transformer-explainer/. Follow one query through its allowed keys, normalized weights, and weighted values. Its GPT-2 small model has different dimensions and parameters from our toy head. Keep sampling controls for Lecture 06. If the page is not ready within 10 seconds, continue with this slide and the local single-head-mask notebook cell. Return here before advancing.

---

<!-- .slide: id="architecture-families" -->

## Three uses of Transformer blocks

| Architecture | Available context | Example objective |
| --- | --- | --- |
| Encoder | Both directions | Predict a masked token |
| Decoder language model | Current prefix | Predict the next token |
| Encoder–decoder | Source plus target prefix | Translate a sentence |

**Our executable model is a decoder language model.**

Note:
Allow 1 minute. The 2025 deck develops the original translation architecture (slides 3–8 and 38–41). Keep it as the source of the building blocks and use a decoder language model for the course's working implementation. An encoder–decoder Transformer also has decoder-to-encoder cross-attention. Sources: Vaswani et al. (2017), §3; Devlin et al. (2019), §3, https://aclanthology.org/N19-1423/.

---

<!-- .slide: id="multiple-views" -->

## Multiple learned views of context

**I gave my dog Charlie some food.**

The representation at <strong>gave</strong> may benefit from several relations.

Each head learns its own query, key, and value projections.

<p class="source">The sentence illustrates possible relations. Heads are not assigned grammatical roles.</p>

Note:
Allow 1 minute. Adapt 2025 slides 32–35. This example uses bidirectional context, as in the original encoder. Do not draw a causal arrow from gave to food in our language model. Multiple heads permit different learned projections; neither unique specialization nor improved quality is guaranteed. At fixed model width with d_h=d/h, increasing h does not itself increase the four dense projection matrices' total parameters. Source: Vaswani et al. (2017), §3.2.2.

---

<!-- .slide: id="head-equations" -->

## Separate projections, one output

$$H_i=\operatorname{Attention}
(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)})$$

$$\operatorname{MHA}(X)
=[H_1;\ldots;H_h]W_O$$

Each head has width $d_h=d/h$ in our model.

Concatenate along features, then mix with $W_O$.

Note:
Allow 2 minutes. The semicolon denotes feature concatenation, not addition. All heads receive the same sequence but use different learned projections. Our baseline uses d_k=d_v=d_h and no projection biases. More general dimension choices are possible. Source: 2025 slides 34 and 39; Vaswani et al. (2017), §3.2.2.

---

<!-- .slide: id="multihead-path" -->

## The multi-head computation

<img class="diagram" src="assets/multihead-path.svg" data-excalidraw-source="assets/multihead-path.excalidraw" alt="Input states are projected to queries, keys, and values, split into heads, processed by attention separately, then joined and projected to the model width.">

<p class="caption">Splitting heads rearranges features. It creates no new parameters.</p>

Note:
Allow 1 minute. Trace one token through every stage, then trace one head across tokens. The projections do the learning; reshape and transpose do bookkeeping. Original editable diagram, following 2025 slides 34 and 39.

---

<!-- .slide: id="head-shapes" -->

## Shapes through multiple heads

| Quantity | Shape |
| --- | --- |
| Input and projected Q/K/V | $(B,T,d)$ |
| Q/K/V after splitting | $(B,h,T,d_h)$ |
| Scores and weights | $(B,h,T,T)$ |
| Per-head outputs | $(B,h,T,d_h)$ |
| Joined and projected output | $(B,T,d)$ |

Note:
Allow 2 minutes. Key positions are the last axis of the score tensor. The query and key lengths happen to agree for self-attention; they can differ in cross-attention. Our example has B=2, T=4, d=16, h=4, and d_h=4. Use labels, not just the repeated number 4, to identify axes.

---

<!-- .slide: id="split-heads" -->

## Split features, then move the head axis

~~~python
B, T, d = x.shape
q = q_proj(x)
q = q.reshape(B, T, h, d // h)
q = q.transpose(1, 2)
# q: (batch, head, query_position, head_feature)
~~~

The transpose changes which axis is a token position.

Note:
Allow 2 minutes. Apply the same operations to K and V. A direct reshape to (B,h,T,d_h) generally mixes token and head indices; matching output shapes cannot establish correctness. The notebook compares with a loop over heads. PyTorch uses row-vector batches, while nn.Linear stores each matrix as (out_features,in_features).

---

<!-- .slide: id="join-heads" -->

## Join the head outputs

~~~python
scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
scores = scores.masked_fill(~allowed, -torch.inf)
weights = scores.softmax(dim=-1)
z = weights @ v
z = z.transpose(1, 2).contiguous().view(B, T, d)
out = out_proj(z)
~~~

<p class="caption">The allowed mask broadcasts across the batch and heads.</p>

Note:
Allow 3 minutes. Here allowed is a lower triangular (T,T) Boolean tensor, with True meaning permitted. Explain contiguous before view: transpose changes strides. This explicit version materializes scores for teaching. The notebook supplies imports, projections, shape validation, and mask construction on the input device; this slide is an excerpt.

---

<!-- .slide: class="exercise" id="exercise-01" -->

<p class="exercise-meta">Exercise E01 · 5 minutes · predict, then check</p>

## Trace and verify the heads

Use $B=2,\ T=4,\ d=16,\ h=4$.

Give the split-Q, score, and joined-output shapes.
Then compare batched attention with a loop over heads.

<div class="answer fragment">

$(2,4,4,4)$; $(2,4,4,4)$; $(2,4,16)$.
Equal shapes hide different axis meanings. Check values and gradients.

</div>

Note:
Allow 5 minutes. Notebook E01 runs the independently expressed head reference in float64. Students identify every axis and inspect the agreement. A wrong reshape is a useful negative control. Do not merely verify that the output has the right shape.

---

<!-- .slide: class="outline-slide" id="outline-blocks" -->

## Outline

<ul class="outline-topics">
<li>From recurrence to attention</li>
<li aria-current="step">Positions and Transformer blocks</li>
<li>A working language model</li>
</ul>

<p class="caption">Period 2 · 45 minutes, including E02–E04</p>

Note:
Allow 1 minute. The head mixes context; a block also needs feature transformations and a trainable path through depth. The original source returns to the architecture repeatedly in slides 51,56,59,65,68,70,71; this section follows that progression in a simpler decoder diagram.

---

<!-- .slide: id="why-positions" -->

## What changes when tokens move?

Without positions or a mask, self-attention satisfies

$$\operatorname{SA}(PX)=P\operatorname{SA}(X)$$

$P$ permutes the token rows.

The output moves with each token; it does not identify its original slot.

Note:
Allow 2 minutes. This is permutation equivariance, not invariance of the whole output tensor. It assumes shared projections and no position-dependent mask or bias. A causal mask already constrains ordering, so do not apply this identity to a fixed causal mask under arbitrary permutations. Permuting key/value pairs together preserves their correspondence; the output follows the permuted query rows. Source: 2025 slides 23 and 52–55; Vaswani et al. (2017), §3.5.

---

<!-- .slide: id="sinusoidal-positions" -->

## Sinusoidal positions

$$\operatorname{PE}(p,2i)
=\sin\left(p/10000^{2i/d}\right)$$

$$\operatorname{PE}(p,2i+1)
=\cos\left(p/10000^{2i/d}\right)$$

Add this vector to the token embedding at position $p$.

<p class="caption">$i=0,\ldots,d/2-1$; the teaching example uses an even width.</p>

Note:
Allow 2 minutes. A pair of coordinates uses the same frequency. Correct the duplicate final sine in 2025 slide 54 and use zero-based frequency indices consistently. At p=0 each pair is (0,1), not (0,0). The original model used sinusoidal positions and also evaluated learned positions. Source: Vaswani et al. (2017), §3.5.

---

<!-- .slide: id="position-frequencies" -->

## Different coordinates change at different rates

<div class="plot" data-plotly="assets/position-frequencies.json" role="img" aria-label="Three sine coordinates of an eight-dimensional sinusoidal encoding over positions zero through thirty-one. Frequencies are 1, 0.1, and 0.01 radians per position."></div>

<p class="caption">Deterministic formula values, with $d=8$. This is not a training result.</p>

Note:
Allow 1 minute. The source deck compares positional frequencies with binary digits (slides 53–54). The graph makes the frequency change visible without suggesting that sine coordinates are bits. The omitted cosine partners have the same frequency. Explicit position features do not guarantee extrapolation to arbitrary sequence lengths.

---

<!-- .slide: id="learned-positions" -->

## Learned absolute positions

~~~python
token_table = nn.Embedding(vocab_size, d)
position_table = nn.Embedding(max_length, d)
positions = torch.arange(ids.shape[1], device=ids.device)
x = token_table(ids) + position_table(positions)
~~~

Both tables are learned jointly with the model.

The position table covers a fixed range of slots.

Note:
Allow 1 minute. Our baseline uses a four-row position table. Reject sequences beyond max_length rather than silently recycling indices. Learned positions add max_length*d parameters. They do not require a pretrained word-vector model. This resolves the inconsistent descriptions in 2025 slides 42 and 57. Source: Vaswani et al. (2017), §3.5, learned-position comparison.

---

<!-- .slide: id="position-demo" -->

## Position changes a token's output

<div class="demo-form">
<button id="position-toggle" type="button" aria-pressed="false">Positions: off</button>
<button id="position-swap" type="button" aria-pressed="false">Swap bank / river</button>
<button id="position-reset" type="button">Reset</button>
</div>

<div id="position-visual" class="plot" data-plotly="assets/position-demo.json" role="img" aria-label="An unmasked four-token toy attention example, initially without positions."></div>

Note:
Allow 1 minute. The four word labels reuse the source sentence, but their vectors are the rows of the identity matrix, not trained embeddings. Q/K/V projections are identity and the scale is sqrt(4). Swap the first and last token and compare each token's output after undoing that permutation. Position vectors, when enabled, remain attached to slots. There is deliberately no causal mask. Reset restores the PDF's initial state.

---

<!-- .slide: class="exercise" id="exercise-02" -->

<p class="exercise-meta">Exercise E02 · 5 minutes · predict, then check</p>

## Does the output move with the token?

Swap <strong>bank</strong> and <strong>river</strong>, then undo that permutation on the output.

Predict the difference with positions <strong>off</strong>, then <strong>on</strong>.

<div class="answer fragment">

Off: the outputs agree up to rounding.
On: fixed slot vectors change the token's input and output.

</div>

Note:
Allow 5 minutes. Use the browser before running the corresponding notebook cells. Both use X=I4 and permutation [3,1,2,0], with sinusoidal positions at slots 0–3. Check the actual tensors, not only the attention visualization. The statement is specific to unmasked attention. In our causal language model, position embeddings and the mask both contribute ordering information.

---

<!-- .slide: id="residual-paths" -->

## Keep a residual path through each sublayer

<img class="diagram" src="assets/residual-block.svg" data-excalidraw-source="assets/residual-block.excalidraw" alt="The pre-normalized attention sublayer adds its update to x, producing a. The pre-normalized feedforward sublayer adds its update to a, producing the block output. Each addition has a direct residual path.">

<p class="caption">Every update and its residual have shape $(B,T,d)$.</p>

Note:
Allow 3 minutes. Trace the direct path and the learned update separately. The attention sublayer mixes allowed positions; the FFN transforms features at each position. Addition requires identical shapes. Residual paths provide a direct gradient contribution but do not guarantee stable gradients at all depths. Sources: 2025 slides 66–67; Vaswani et al. (2017), §3.1; Xiong et al. (2020).

---

<!-- .slide: id="norm-location" -->

## Where does normalization go?

**Original post-LN sublayer**

$$y=\operatorname{LN}\bigl(x+F(x)\bigr)$$

**Our pre-LN sublayer**

$$y=x+F\bigl(\operatorname{LN}(x)\bigr)$$

The placement changes the computation and its gradients.

Note:
Allow 3 minutes. F is attention or the FFN. The original 2017 Transformer also uses dropout; the teaching model has none. Our pre-LN decoder includes a final normalization before readout. For the controlled comparison later, both variants retain that final norm, so only sublayer norm order changes. Do not describe the comparison model as a full reproduction of the 2017 architecture. Source: Xiong et al., On Layer Normalization in the Transformer Architecture, https://proceedings.mlr.press/v119/xiong20b.html.

---

<!-- .slide: id="layernorm-axis" -->

## LayerNorm works across one token's features

$$\mu=\frac1d\sum_jx_j,\qquad
v=\frac1d\sum_j(x_j-\mu)^2$$

$$\operatorname{LN}(x)_j
=\gamma_j\frac{x_j-\mu}{\sqrt{v+\epsilon}}+\beta_j$$

For $(B,T,d)$ states, normalize the last axis.

<p class="caption">Each token has its own statistics; $\gamma,\beta$ are shared across tokens.</p>

Note:
Allow 3 minutes. Use the population variance (divide by d), as in nn.LayerNorm. Our epsilon is 1e-5, with learned affine scale and bias. Normalizing across time would mix statistics from future positions and can break causality. Source: Ba et al., Layer Normalization (2016), https://arxiv.org/abs/1607.06450; 2025 slide 67.

---

<!-- .slide: class="exercise" id="exercise-03" -->

<p class="exercise-meta">Exercise E03 · 4 minutes · predict, then check</p>

## Normalize and add

For $x=(1,3)$, find $\mu$, $v$, and $\operatorname{LN}(x)$ with $\gamma=1$, $\beta=0$, $\epsilon=10^{-5}$.

A sublayer returns $(2,-1)$. What is the residual sum?
For $(B,T,d)$ states, which axis does LayerNorm normalize?

<div class="answer fragment">

$\mu=2,\ v=1$; $\operatorname{LN}(x)\approx(-1,1)$.
The residual sum is $(3,2)$.
Normalize over the last feature axis, $d$.

</div>

Note:
Allow 4 minutes. The exact normalized vector is (-1,1)/sqrt(1+epsilon), approximately(-0.999995,0.999995). The residual sum is x+F(...), not LN(x)+F(...), in our pre-LN model. The sublayer output in the question is given; students need not infer F. Notebook checks this calculation and the feature axis.

---

<!-- .slide: id="feedforward" -->

## A feedforward network at each position

$$\operatorname{FFN}(x)
=\operatorname{ReLU}(xW_1)W_2$$

$$d\ \longrightarrow\ d_{\mathrm{ff}}\ \longrightarrow\ d$$

The same weights are applied independently to every token.

Our teaching FFN omits biases.

Note:
Allow 3 minutes. The original Transformer uses two affine layers with biases and a ReLU. State our deliberate simplification before counting parameters. The inner width need not be 4d; our tiny model uses d=16 and d_ff=32. No new context is mixed by a position-wise FFN alone. Source: 2025 slide69; Vaswani et al. (2017), §3.3.

---

<!-- .slide: id="token-feature-mixing" -->

## Attention and the FFN do different work

| Sublayer | Information combined | Output shape |
| --- | --- | --- |
| Causal attention | Features from allowed token positions | $(B,T,d)$ |
| Position-wise FFN | Features within one token state | $(B,T,d)$ |

Changing another token can affect attention's output.

It cannot affect a standalone FFN's output at this token.

Note:
Allow 2 minutes. Test this on already supplied input states to an isolated FFN. In a complete block, the FFN input already contains context from attention. Avoid implying the full model's FFN states are context-free. Notebook includes a position-isolation check.

---

<!-- .slide: id="block-code" -->

## A complete pre-LN block

~~~python
def forward(self, x, causal=True):
    update, _ = self.attention(self.attention_norm(x), causal)
    x = x + update
    return x + self.ffn(self.ffn_norm(x))
~~~

Each block has its own attention, FFN, and normalization parameters.

The notebook defines every component explicitly.

Note:
Allow 3 minutes. This is the pre-LN branch of the notebook's Block.forward. Attention also returns its weights for inspection; the underscore discards them here. Inputs and outputs keep shape (B,T,d). Distinct blocks have distinct parameters; the same block parameters apply to all time positions. No recurrence over positions is introduced.

---

<!-- .slide: id="block-parameters" -->

## Count one block's parameters

Use $d=16$, $h=4$, and $d_{\mathrm{ff}}=32$.

| Component | Count | Value |
| --- | --- | ---: |
| Q, K, V, output projections | $4d^2$ | 1,024 |
| Two FFN matrices | $2d\,d_{\mathrm{ff}}$ | 1,024 |
| Two affine LayerNorms | $4d$ | 64 |
| **One block** | | **2,112** |

<p class="caption">All Linear layers are bias-free. LayerNorm has both scale and bias.</p>

Note:
Allow 2 minutes. Four heads of width 4 partition each full-width projection. There is no extra factor of h in 4d². The operations softmax, reshape, mask, residual addition, and ReLU add no trainable parameters. This makes the conventions missing from 2025 slide 77 explicit.

---

<!-- .slide: id="full-model" -->

## Assemble the language model

<img class="diagram" src="assets/decoder-model.svg" data-excalidraw-source="assets/decoder-model.excalidraw" alt="Token IDs select token embeddings and add learned positions. Two decoder blocks and a final LayerNorm produce states. A vocabulary readout reuses the token embedding table to produce next-token logits.">

<p class="caption">Every position produces a vocabulary logit vector: $(B,T,7)$.</p>

<p class="source">Interactive model: <a href="https://bbycroft.net/llm" target="_blank" rel="noopener noreferrer">Brendan Bycroft's LLM Visualization</a>.</p>

Note:
Allow 4 minutes. Match each box to a notebook module. The repeated blocks change representations but preserve shape. Only the token embedding table is reused for readout; the position table is independent. The source architecture's encoder and cross-attention are omitted because this model predicts a continuation from one prefix.

Spend 3 minutes on our diagram, then at most 1 minute in the prepared https://bbycroft.net/llm tab. Trace an input token through embeddings, attention, the feedforward computation, and the output. The working small model sorts letters; it has its own dimensions and parameters. Ask students to identify the token axis and feature axis, then return to our seven-token, two-block model before E04. Source and implementation: https://github.com/bbycroft/llm-viz. If the visualization does not load promptly, trace the same path in the local diagram.

---

<!-- .slide: class="exercise" id="exercise-04" -->

<p class="exercise-meta">Exercise E04 · 5 minutes · predict, then check</p>

## Count the whole language model

Two blocks; $d=16$; 7 tokens; 4 learned positions.

Add one final affine LayerNorm.
Tie the bias-free output matrix to the token table.

<div class="answer fragment">

$2(2112)+7(16)+4(16)+2(16)=\mathbf{4432}$.
Untying the readout adds 112 parameters.

</div>

Note:
Allow 5 minutes. Verify 4432 unique trainable parameters, or 4544 with an untied output. Weight tying means the same Parameter object is reused; equal initial values in distinct Parameters are not tying. This is a new ungraded practice inspired by the parameter-count concept in 2025 slide 77, not the current private quiz.

---

<!-- .slide: class="outline-slide" id="outline-working" -->

## Outline

<ul class="outline-topics">
<li>From recurrence to attention</li>
<li>Positions and Transformer blocks</li>
<li aria-current="step">A working language model</li>
</ul>

<p class="caption">Period 3 · 30 minutes of teaching, then Quiz 2 (15 minutes)</p>

Note:
Allow 1 minute. The complete code is already in the notebook. Run small checks before interpreting a training curve. Quiz 2 is separate private material on previous lectures. The teaching plan reserves the published 15-minute quiz duration.

---

<!-- .slide: id="shifted-targets" -->

## The input and target are shifted

| Position | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Input | BOS | red | key | opens |
| Target | red | key | opens | EOS |

The logit vector at position $t$ predicts token $t+1$.

Teacher forcing supplies the observed input prefix during training.

Note:
Allow 1 minute. Continue Lecture 04's seven-token vocabulary and two invented sentences. The second is BOS blue key closes EOS. Inputs and targets are sliced within each document; nothing crosses a sentence boundary. There is no padding in this example. Source: 2025 slide58 and original Transformer shifted-right decoder input.

---

<!-- .slide: id="causal-visibility" -->

## The diagonal is allowed

<img class="diagram" src="assets/causal-mask.svg" data-excalidraw-source="assets/causal-mask.excalidraw" alt="Four causal attention rows. The BOS row can use BOS; red can use BOS and red; key can use BOS, red, and key; opens can use all four inputs. The targets are red, key, opens, and EOS.">

<p class="caption">Position $t$ can read input $t$ while predicting target $t+1$.</p>

Note:
Allow 1 minute. Revisit the off-by-one ambiguity in 2025 slide61. The token at the diagonal is an observed input, not the next target. A strict lower triangle would create a fully masked first row unless another special convention were used. Our lower triangle includes the diagonal and every row has at least one allowed key.

---

<!-- .slide: id="attention-inputs" -->

## Where do Q, K, and V come from?

| Attention use | Q | K and V | Visibility |
| --- | --- | --- | --- |
| Encoder self-attention | Source states | Source states | All valid source tokens |
| Decoder self-attention | Target states | Target states | Target prefix |
| Decoder cross-attention | Target states | Encoder outputs | All valid source tokens |

Note:
Allow 1 minute. This table completes the bridge from the original source's translation diagram to the implemented model. Source and target lengths can differ in cross-attention, so its score matrix can be rectangular. Padding masks still apply when sequences are padded. Our no-padding decoder-only notebook needs just the causal self-attention row. Source: Vaswani et al. (2017), §3.2.3.

---

<!-- .slide: id="bert-contrast" -->

## Bidirectional masked prediction

**The cat sat on the [MASK].**

A BERT-style encoder can use context on both sides of a selected token.

The training objective predicts selected corrupted positions.

A causal language model instead predicts the next token from its prefix.

Note:
Allow 1 minute. In BERT's original recipe, selected prediction positions are corrupted using a mixture of masking, replacement, and unchanged tokens; this sentence shows only the mask-token case. Do not imply BERT is obtained by removing a causal mask while keeping next-token targets, which would leak target information. Source: Devlin et al. (2019), §3.1. Keep the broader model-family survey for other material.

---

<!-- .slide: id="toy-configuration" -->

## A model small enough to inspect

| Setting | Teaching baseline |
| --- | --- |
| Data | Two invented sentences, seven token IDs |
| Shape | Batch 2, length 4, width 16 |
| Blocks | 2; each has 4 heads and FFN width 32 |
| Positions / normalization | Learned absolute / pre-LN |
| Readout / total parameters | Tied / 4,432 |

Note:
Allow 1 minute. No dropout or Linear biases; affine LayerNorm with epsilon 1e-5. Parameters use seed 7 and a documented initialization. This is a transparent mechanism and debugging example; no generalization or benchmark-quality claim is possible from two sentences.

---

<!-- .slide: id="logits-and-loss" -->

## Compute next-token loss from logits

~~~python
inputs = documents[:, :-1]
targets = documents[:, 1:]
logits = model(inputs)              # (2, 4, 7)
loss = F.cross_entropy(
    logits.reshape(-1, 7), targets.reshape(-1)
)
~~~

Cross-entropy receives raw logits.

Note:
Allow 1 minute. Flatten batch and position in the same order for logits and targets. The library computes log-softmax internally; applying softmax first changes the expected input. Our targets include EOS, while BOS is an input-only token. There is no ignored padding in this toy batch.

---

<!-- .slide: id="check-complete-model" -->

## Check the complete computation

1. Compare multiple heads with an independent reference.
2. Perturb a future token and compare earlier logits.
3. Differentiate an early-output loss with respect to input states.

Run comparisons with dropout disabled and stated tolerances.

Note:
Allow 1 minute. A correct head can still be embedded in an incorrect model. A wrong normalization axis or reshape can introduce leakage. The gradient test must use per-position input states, not embedding-table gradients: the same token embedding parameters can be used at several positions. No dropout is present here; eval mode remains a useful habit for deterministic comparisons.

---

<!-- .slide: class="exercise" id="exercise-05" -->

<p class="exercise-meta">Exercise E05 · 6 minutes · predict, then check</p>

## Can the future change the prefix?

Change only input position 3 in the two-block model.

Which logits can change? Where may an early-output loss send gradients?

<div class="answer fragment">

Logits at positions 0–2 stay unchanged.
Their loss has zero gradient to the input state at position 3.
An unmasked negative control can leak.

</div>

Note:
Allow 6 minutes. The notebook compares complete-model logits and uses a prefix loss on positions 0–2. It checks zero future gradients and nonzero allowed-prefix gradients. The future token replacement is legal but deliberately different. Then inspect the unmasked negative control so a trivially constant implementation cannot satisfy the lesson.

---

<!-- .slide: id="verification-evidence" -->

## Evidence for correctness

| Check | Expected evidence |
| --- | --- |
| Batched heads vs head loop | Outputs and gradients agree in float64 |
| Full-model future perturbation | Earlier logits agree |
| Loss on earlier logits | Future input-state gradients are zero |
| Unmasked negative control | Earlier logits can change |

<p class="caption">A loss curve alone cannot establish causality or tensor correctness.</p>

Note:
Allow 1 minute. Forward and gradient comparisons use explicit numerical tolerances in the tests; inspect the notebook's printed errors. Causality tests cover two blocks, feature-wise norms, residual additions, and final readout. A finite-difference or library reference complements the independently expressed head loop when checking derivatives.

---

<!-- .slide: id="irreducible-toy-loss" -->

## This toy batch cannot reach zero loss

Both sentences start with <strong>BOS</strong>.
The next token is <strong>red</strong> or <strong>blue</strong>, equally often.

$$L_{\min}=\frac{\log 2}{4}\approx0.1733
\quad\text{nats per token}$$

The other three targets per sentence can be predicted consistently.

Note:
Allow 2 minutes. There are eight target positions in the two-sentence batch. The identical BOS prefix contributes two copies of log 2 at optimum; other terms can approach zero. We do not mask away the ambiguous targets or claim perfect top-1 accuracy. A fitting criterion below 0.20 nats is meaningful here. The lower bound is an infimum for finite softmax logits; the other target losses approach zero with increasing margins.

---

<!-- .slide: id="tiny-training-loop" -->

## Fit the checked model on the tiny batch

~~~python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.02, weight_decay=0.0
)
for step in range(200):
    optimizer.zero_grad()
    logits = model(inputs)
    loss = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten()
    )
    loss.backward()
    optimizer.step()
~~~

<p class="caption">CPU, seed 7, full batch, no dropout. This is a fitting check.</p>

Note:
Allow 1 minute. Parameters are initialized with standard deviation 0.02 for embeddings and Linear weights; norms start with scale 1 and bias 0. AdamW uses betas (0.9,0.999), epsilon 1e-8, and zero weight decay. Record losses before training and after completed updates; do not confuse update index with an epoch over a large dataset. See assets/README.md for the exact run and caveats.

---

<!-- .slide: id="norm-comparison" -->

## Compare normalization order

<div class="plot" data-plotly="assets/norm-comparison.json" role="img" aria-label="Measured mean next-token cross-entropy for pre-LN and post-LN teaching models over 200 CPU updates on the same two-sentence toy batch, with the theoretical loss floor shown."></div>

<p class="caption">Same initial parameters and training budget; only block norm order changes.</p>

Note:
Allow 3 minutes. Both variants retain the same final LayerNorm, tied readout, learned position table, and training configuration. The graph is recomputed from the notebook; it is not a claim that one placement always trains faster. Values at 0 are before any update; values at 200 follow 200 updates. Historical motivation: Xiong et al. (2020). The measured comparison is our own controlled toy experiment.

---

<!-- .slide: id="comparison-limits" -->

## What does this comparison establish?

- Both models use the same data, initialization, optimizer, and budget.
- Their losses and gradient norms describe this small fitting run.
- One seed and two blocks cannot establish a general ranking.

Next: change one factor and repeat the checks.

Note:
Allow 1 minute. The notebook records initial/final losses and gradient norms even if post-LN performs as well as or better than pre-LN. Preserve that result. Training stability can depend on depth, initialization, learning rate, and normalization details; the source deck's blanket claim of no vanishing/exploding gradients is removed. Do not infer held-out performance because there is no held-out set. If extending to a corpus, split documents before forming overlapping windows.

---

<!-- .slide: id="attention-cost" -->

## A materialized attention matrix grows quadratically

$$\text{score storage}=4BhT^2\ \text{bytes in float32}$$

| Length $T$ | One matrix, $B=2,\ h=4$ |
| --- | ---: |
| 256 | 2 MiB |
| 1,024 | 32 MiB |
| 2,048 | 128 MiB |

<p class="caption">One layer's score or weight tensor; this is not total training memory.</p>

<p class="source">Generation walkthrough: <a href="https://www.llm-visualized.com/?token=4&amp;generation=0&amp;kvCache=0" target="_blank" rel="noopener noreferrer">LLM-Visualized</a>.</p>

Note:
Allow 2 minutes. MiB means 2^20 bytes. The formula counts a materialized (B,h,T,T) tensor, not every kernel's implementation or total model state. Our tiny T=4 tensor needs 512 bytes. At fixed width, QK and AV together take approximately 4BT²d floating-point operations when a multiply-add counts as 2. Original slides 81–82 survey approximate alternatives; later efficiency material will distinguish algorithm changes from memory-efficient exact attention. Training positions can be parallelized, but autoregressive generation still depends on preceding generated tokens.

Use 1 minute for the storage table and 1 minute for the prepared https://www.llm-visualized.com/?token=4&generation=0&kvCache=0 tab. Preserve this starting URL and keep KV caching off. Trace one forward pass to the next-token probabilities and show how the next pass extends the prefix. Keep KV caching for Lecture 13. The site illustrates a separate model; its numerical values do not reproduce our notebook. If it is unavailable, return to shifted-targets and trace the next prediction with the local decoder diagram. Keep the entire walkthrough within this slide's 2-minute allocation.

---

<!-- .slide: id="original-and-baseline" -->

## The original architecture and our baseline

| Choice | 2017 Transformer | Teaching decoder |
| --- | --- | --- |
| Architecture | Encoder–decoder | Two decoder blocks |
| Positions | Sinusoidal | Learned absolute |
| Sublayer norm | Post-LN | Pre-LN |
| Regularization | Dropout and label smoothing | Neither in the toy fit |

<p class="caption">State each choice before interpreting a result or counting parameters.</p>

Note:
Allow 1 minute. This table identifies deliberate adaptation rather than silently mixing diagrams from incompatible designs. The original FFN also has biases, while ours does not. Our controlled post-LN comparison changes only sublayer placement and keeps a final norm, so it is not the complete original model. Sources: Vaswani et al. §§3,5.4; source deck 76; teaching-plan.md configuration.

---

<!-- .slide: id="recap" -->

## Check your model of the Transformer

1. Which operation combines information across token positions?
2. Why can a causal model read its diagonal input?
3. Why can this two-sentence batch never reach zero mean loss?

**A working model needs a correct computation and evidence that checks it.**

Note:
Allow 1 minute. Expected responses: attention combines permitted positions; the current input predicts the next target; identical BOS prefixes have two equally frequent next tokens. The FFN and LayerNorm act within features at a position. This recap is ungraded; it does not expose the private quiz.

---

<!-- .slide: id="next-steps" -->

## Before Lecture 06

- Reopen E05 and inspect the complete-model checks.
- A2 and candidate projects are scheduled for release today.
- A2 is due October 31; project proposals are due October 14.

**October 14:** pretraining and decoding in practice.

<p class="source">Submission deadlines: 23:59, Asia/Shanghai. Follow the course announcements and eLearning.</p>

Note:
Allow 1 minute. These are the published course dates at authoring; the slide introduces no new release or deadline. Quiz 2 uses the 15-minute slot and covers prior material through the private instructor workflow. Keep instructions to students in the published assignment/project announcements. Later Lecture 06 covers optimizer schedules, checkpoint/resume behavior, and decoding. References: ../../docs/schedule.md and ../../index.html.

---

<!-- .slide: class="references" id="references" -->

## Reading

- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762): §§3.1–3.5; §4.
- [Xiong et al., LayerNorm in Transformers](https://proceedings.mlr.press/v119/xiong20b.html): pre-LN and post-LN.
- [Devlin et al., BERT](https://aclanthology.org/N19-1423/): §3.1, masked prediction.
- [Bahdanau et al.](https://arxiv.org/abs/1409.0473) and [Luong et al.](https://aclanthology.org/D15-1166/): §3, learned alignment.
- [Historical examples and further reading](optional-reading.md)
- [Interactive Transformer walkthroughs](optional-reading.md#interactive-transformer-walkthroughs)

Note:
Allow 3 minutes. The 2025 Lecture 05 PowerPoint supplied the content sequence and examples; teaching-plan.md maps every source slide and records corrections. All new diagrams are editable and documented in assets/README.md. The core notebook is offline. Figures and reported values must be checked against their cited versions rather than carried forward from a screenshot.
