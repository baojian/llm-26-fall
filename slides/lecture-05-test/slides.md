<!-- .slide: id="context-for-text" -->

## Context for a token

<div class="teaching-diagram" data-diagram="context-intuition" data-source-url="https://www.youtube.com/watch?v=tIvKXrEDMhk"></div>

Note:
Allow 1–2 minutes. This single slide condenses “Get more context for text,” pages 4–6 of the 49-slide Desktop lecture-05-slides-transformers.pptx inspected on October 10, 2026 (Asia/Shanghai). Those pages attribute the example to Rasa Algorithm Whiteboard. The sentence and linguistic motivation are retained; the three reveals are redesigned for the course's causal GPT focus.

Start with the complete example sentence and the highlighted word “she.” Ask which earlier word identifies its referent. Give students a few seconds to respond; the expected answer is “Noa.” This oral check introduces the mechanism and adds no notebook exercise.

1. Reveal the backward link from “she” to “Noa.” Intervening words do not prevent a useful relation. The link represents a human reading of this sentence. It does not prescribe a model's attention weights or claim a particular head will learn a coreference explanation.
2. Reveal the causal boundary. At the position of “she,” a GPT model can use the prefix including “she.” The later words “is a great cat” are masked at this position, even when training provides the complete sentence. Treat the ten displayed words as toy token positions; a real tokenizer may split them differently.
3. Reveal the context equation. t is the position of “she,” v_j is a value vector at available position j, and alpha_(t,j) is its attention weight. In standard softmax attention these weights are nonnegative and sum to one over the available positions. The model learns the score computation through training and computes the weights from the current representations. It can combine information from the whole available prefix, including the current token.

The next slide isolates the arithmetic with an unmasked four-word example, initially using the input vectors as values. State that simplification explicitly: all four words participate there. Learned Q/K/V projections follow; the later one-head slide applies the causal mask before softmax. Keep this distinction when moving the section into Lecture 05. Preserve the stable ID context-for-text and its adjacent self-attention example.

---

<!-- .slide: id="self-attention" -->

## Self-attention

<div id="attention-diagram" class="attention-diagram" data-source-slide="24"></div>

Note:
Source: Baojian Zhou, Spring 2026, lecture-05-slides-transformers.pptx, slide 24.
Original attribution: [Rasa Algorithm Whiteboard 2](https://www.youtube.com/watch?v=tIvKXrEDMhk).

Use Space, Right, Next, or click the diagram to reveal the next group.
Left and Previous reverse a group. Reset returns to the title-only start.

1. Compare the third embedding with the four input embeddings.
2. Show the dot-product scores.
3. Normalize with softmax.
4. Show the attention weights.
5. Bring in the embeddings to be combined.
6. Multiply each embedding by its weight.
7. Sum the products to obtain the third output.
8. Repeat for the other output positions.

This is the source's introductory attention mechanism without learned Q/K/V
projections. Violet identifies the query word and teal identifies weights and
context vectors. The four-cell bars are schematic embedding vectors, not data.
This example uses unmasked attention to show the full score-and-mix calculation;
the opener's GPT example restricts the available positions to the causal prefix.
The later one-head slide adds that mask to the same calculation.
For each query position, recompute all scores and weights before combining the
input embeddings. The input and output dimensions match.

The default design uses aligned word columns, bullet explanations, and KaTeX. The Original
toolbar link opens the faithful Spring reproduction with its source artwork.

---

<!-- .slide: id="attention-roles" -->

## One input, three roles

<div class="teaching-diagram" data-diagram="attention-roles" data-source-url="https://www.youtube.com/watch?v=tIvKXrEDMhk"></div>

Note:
Allow 1 minute. This bridge condenses pages 5–7 of the 43-slide Desktop lecture-05-slides-transformers.pptx inspected on October 10, 2026 (Asia/Shanghai). Current page 4 is the calculation already shown on the preceding slide; avoid replaying it. The source introduces the missing projection parameters, observes three uses of an embedding, and names query, key, and value.

1. Name the three roles in the same “Bank of the river” example. For the output at position 3, x_3 is the query. Each x_j, for j = 1, 2, 3, 4, is used as a key in x_3^T x_j and as a value in the weighted sum. The colors label roles; the equations show that the vectors have not been transformed. At another output position, select that position's input as the query and recompute the weights. The diagram shows an unmasked introductory example, as on the preceding slide.
2. Motivate separate learned representations for these roles. The source says “No weights for model to train”; qualify this as no learned Q/K/V projection parameters in this simplified operation. The embedding table and upstream layers can already be trainable. Attention weights are computed from the current inputs and are distinct from the learned projection matrices. The next slide introduces those matrices and their shared use across positions.

Keep the stable attention-roles ID between self-attention and learned-projections when moving the section into Lecture 05. This bridge introduces no additional notebook exercise; E03 later checks the projected head dimensions.

---

<!-- .slide: id="learned-projections" -->

## Learned queries, keys, and values

<div id="attention-qkv-diagram" class="attention-diagram" data-source-slide="29"></div>

Note:
Source: Baojian Zhou, Spring 2026, lecture-05-slides-transformers.pptx, slide 29.
Original attribution: [Rasa Algorithm Whiteboard 2](https://www.youtube.com/watch?v=tIvKXrEDMhk).

This modern layout groups the source's per-word projections into three matrix
branches. Each matrix contains one row per word; the cell patterns are schematic.
Use the Original link to compare with the original per-word diagram.

Keep the source's five clicks:

1. Query projection: what each word looks for in the other words.
2. Key projection: what each word offers for matching against a query.
3. Value projection: the information passed into the weighted sum.
4. All three projection matrices are learned during training. The same query
   matrix is applied at every position; likewise for the key and value matrices.
   The three matrices have separate parameters.
5. Stack token embeddings as rows of X. Then Q = X W_Q, K = X W_K, V = X W_V.

For dimensions, X is n by d_model, W_Q and W_K are d_model by d_k, and W_V is
d_model by d_v. Dot products form Q K^T; apply softmax across each row (over
keys), then multiply the weights by V to obtain Z, of shape n by d_v.
This uses row vectors, the transpose of the earlier example's column-vector
notation. The computation is the same. Scaling, masking, and multiple heads
are introduced later; this slide isolates the role of the learned projections.

Reset restarts the current slide. Next after the three-role bridge's last reveal
advances here. Previous from this slide's initial state returns to that bridge.

---

<!-- .slide: id="gpt-architecture" -->

## GPT-style causal Transformer

<div id="gpt-architecture-diagram" class="attention-diagram" data-source-slide="40" data-reference-label="Reference" data-source-url="https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.svg"></div>

Note:
This is the causal language-model architecture to build in the course. It
adapts the architecture topic from Spring 2026 source page 40, which shows the
2017 encoder-decoder Transformer. The five reveals on this slide are new.
Read the data flow upward, from the two embedding components at the bottom
to the final LayerNorm and language-model head at the top. The reveal order
follows this computation. The two embedding branches meet at an addition node.

Visual reference: [Full GPT architecture](https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.svg),
original by Marxav, vectorization by Mrmw (2024), CC0 1.0. The reference is
available through the toolbar. The teaching diagram combines attention's
internal operations and the MLP's internal projections into named modules.

Architecture source: [OpenAI GPT-2 implementation](https://github.com/openai/gpt-2/blob/master/src/model.py),
functions `attention_mask`, `block`, and `model`. This example uses GPT-2-style
learned position embeddings and LayerNorm before each sublayer. The final
LayerNorm follows the last block; see also the
[GPT-2 report, section 2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
GPT variants may use other position schemes
and normalization layers. Dropout is omitted from this overview.

1. Map token IDs to embeddings and add learned position embeddings. The three
   displayed words stand for token positions; an actual tokenizer may split
   words into several tokens. The hidden states have shape n by d_model.
2. Normalize each position, compute causal multi-head self-attention, and add
   the result to the unnormalized residual stream. Position i can attend to
   positions j <= i, including itself. Set future attention logits to negative
   infinity before softmax, so their attention weights are zero.
3. Normalize again, apply the same MLP independently at every position, and
   add its output to the residual stream. Both additions preserve d_model.
4. Repeat the complete block L times. Blocks have separate learned parameters;
   each block shares its own parameters across token positions. The bypass
   arrows and plus symbols are residual connections.
5. Apply final LayerNorm, a linear vocabulary projection (the LM head), and
   softmax. At generation time, use the final input position's distribution,
   choose a token, append it to the prefix, and repeat. “river” is an illustrative
   continuation, not a measured model prediction.

The input to each block is H. Its two updates are
U = H + CausalMHA(LN_1(H)) and H_next = U + MLP(LN_2(U)).
The vocabulary head acts at every position during training. Train in parallel
against shifted next-token targets; sampling and appending are generation
steps. The causal mask prevents access to each target token. A KV cache can
reuse earlier keys and values during generation; it is omitted here.

The first component walkthrough starts next: positional information. The two
input branches here illustrate additive positions (GPT-2 learns those vectors).
We will first replace the learned table with the original Transformer's fixed
sinusoidal code. Then we will show a different insertion point for RoPE:
inside attention, after the query/key projections. RoPE is not an additional
vector to add at the bottom of this diagram.

---

<!-- .slide: id="position-inputs" -->

## Token identity and position

<div class="position-diagram" data-diagram="position-inputs" data-source-url="https://arxiv.org/html/1706.03762v7#S3.S5"></div>

Note:
Component 01 · 2 minutes. New design based on the topic of Spring 2026 source
pages 51–54. Learning objective: distinguish token identity, absolute position,
and the place where a position method enters the computation.

Use the two occurrences of “the” in “the cat saw the dog”: p = 0 and p = 3.
Assume these displayed words are single tokens for this example. Their lookup
embedding is identical; adding different position vectors produces different
inputs. Read upward, as on the architecture slide. The colored cells are
schematic, not numeric measurements. The two branches are kept separate to
make the addition explicit. p is a zero-based token position.

Reveal 1: token lookup. Reveal 2: position lookup/function and addition.
Reveal 3: model input. Position vectors have width d_model, the same as token
embeddings. In the 2017 paper, e_t = sqrt(d_model) E[t] is the scaled token
embedding, so the full input is x_p = sqrt(d_model) E[t_p] + PE(p).
Dropout is omitted here. The addition happens at the input to the stack.

Do not claim that a causal Transformer has no position information without a
position embedding: the causal mask itself introduces asymmetry. The purpose
here is to introduce an explicit position representation. Unmasked attention
without any positional mechanism is permutation equivariant. The NoPos paper
on source page 54 is a useful caveat, not a third method for this section.

Source: [Vaswani et al., Attention Is All You Need, §§3.4–3.5](https://arxiv.org/html/1706.03762v7#S3.S5).
Caveat: [Haviv et al., Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634).

---

<!-- .slide: id="sinusoidal-frequencies" -->

## Sinusoidal encoding: many clocks

<div class="position-formulas">
  <span>$\omega_i=10000^{-2i/d}$</span>
  <span>$PE(p,2i)=\sin(p\omega_i)$</span>
  <span>$PE(p,2i+1)=\cos(p\omega_i)$</span>
</div>

<div id="position-frequency-plot" class="plot" data-plotly="assets/position-frequencies.json" role="img" aria-label="Three sine channels for embedding width eight. Dimension zero oscillates quickly; dimensions two and four vary more slowly across positions zero to sixty-four." data-source-url="https://arxiv.org/html/1706.03762v7#S3.S5"></div>

<p class="caption">Example: embedding width $d=8$. Pair $i$ shares one frequency; its sine channel is shown.</p>

Note:
3 minutes. Replaces the frequency plots on source pages 51–52. Horizontal axis:
token position p; vertical axis: the value of one position-encoding coordinate.
Hover to read values. Curves use dense samples for legibility; real token
positions are integers. No parameters are learned in this position function.

d abbreviates d_model, an even embedding width. Pair i = 0, …, d/2 − 1 uses
coordinates 2i and 2i+1. With d = 8, the angular frequencies are 1, 0.1, 0.01,
and 0.001 radians per token. The figure shows sine coordinates 0, 2, and 4.
Their cosine partners have the same frequencies and a quarter-cycle phase
offset. Dimension 6 and its cosine partner 7 are omitted to keep the chart clear.
All eight coordinates together make one vector for one position.

The binary-counter analogy on source page 53 motivates combining fast and
slow changes. These are smooth rotations, not binary digits; avoid claiming
an exact binary representation. At p = 0 all sine entries are zero and all
cosine entries are one. The next slide asks students to calculate that vector.

Source: [Attention Is All You Need, §3.5](https://arxiv.org/html/1706.03762v7#S3.S5).
The chart is computed from the formula by build-positions.mjs; it is original
artwork. Do not promise that a defined code at unseen positions guarantees
accurate extrapolation by a trained model.

---

<!-- .slide: id="sinusoidal-example" -->

## Build one position vector

<p class="exercise-meta">Practice E01 · 2 min · Predict $PE(0)$, then add $PE(1)$ to the embedding.</p>

<p>For $d=4$: $\quad PE(p)=[\sin p,\cos p,\sin(p/100),\cos(p/100)]$.</p>

<table class="position-table">
  <thead><tr><th>Position</th><th>Dim. 0</th><th>Dim. 1</th><th>Dim. 2</th><th>Dim. 3</th></tr></thead>
  <tbody>
    <tr><td>$p=0$</td><td><span class="fragment position-answer" data-fragment-index="0">0</span></td><td><span class="fragment position-answer" data-fragment-index="0">1</span></td><td><span class="fragment position-answer" data-fragment-index="0">0</span></td><td><span class="fragment position-answer" data-fragment-index="0">1</span></td></tr>
    <tr><td>$p=1$</td><td>0.841</td><td>0.540</td><td>0.010</td><td>1.000</td></tr>
  </tbody>
</table>

<p class="position-equation">Toy embedding: $e_{t_1}=[1,0,1,0]$.</p>

<div class="fragment" data-fragment-index="1">
  <p class="position-equation position-answer">$x_1=e_{t_1}+PE(1)\approx[1.841,\;0.540,\;1.010,\;1.000]$</p>
  <p class="caption">Coordinate-wise addition; the vector still has four coordinates.</p>
</div>

Note:
Practice E01. Pause before the first reveal. Students write the four entries
for p = 0, then compute the four coordinate-wise sums for p = 1. Use radians.
The first click reveals [0, 1, 0, 1]; the second reveals the sum.

Checked values: PE(1) = [0.8414709848078965, 0.5403023058681398,
0.009999833334166664, 0.9999500004166653]. The last entry rounds to 1.000;
it is not exactly one. The two frequencies are 1 and 0.01 radians per token.
Do not reuse the d = 8 denominator from the preceding illustration: we chose
d = 4 here so students can calculate the entire vector.

e is the already-scaled embedding from the preceding slide. With d = 4,
the invented raw lookup E[t_1] would be [0.5, 0, 0.5, 0]. No trained model
or dataset is used in this example. Notebook E01 implements the formula and
checks these numbers using Python's standard math library. This is ungraded
classroom practice, not an assignment or a grading rubric.

Source formula: [Attention Is All You Need, §§3.4–3.5](https://arxiv.org/html/1706.03762v7#S3.S5).

---

<!-- .slide: id="rope-rotation" -->

## RoPE: rotate each query/key pair

<div id="rope-rotation-visual" class="position-diagram" data-diagram="position-rotation" data-interactive="rotation" data-source-url="https://arxiv.org/html/2104.09864v5"></div>

<div class="position-controls">
  <label for="rope-position">Position $p$ <input id="rope-position" type="range" min="0" max="6" step="1" value="2"><output id="rope-position-value" for="rope-position">2</output></label>
  <span class="caption">Move $p$; the vector keeps its length.</span>
</div>

Note:
4 minutes. RoPE = rotary position embedding. Source page 54 names RoFormer;
this slide adds the mechanism. The formulas here use column vectors. q and k
are content vectors after the learned projections, before applying RoPE.

Start with the unit vector (1, 0). At p = 2, the teaching frequency θ = π/6
radians gives a 60° rotation: (0.5, sqrt(3)/2). Drag the position slider.
At p = 0 it is unchanged; at p = 3 it becomes (0, 1). Rotation preserves
length. Reset returns p to 2. The toy frequency makes angles easy to read;
it is not the standard RoPE frequency of the first pair, which is 1 radian.

For a head of even width d_h, apply a 2D rotation independently to each
coordinate pair (2i, 2i+1), with θ_i = 10000^(−2i/d_h). Other coordinate
pairing conventions can implement the same construction. The shown formula
uses the baseline from RoFormer; model-specific bases and scaling can differ.

Apply RoPE to each head's queries and keys in each attention layer. The
standard construction leaves V unchanged. RoPE does not replace the learned
Q/K projections or the causal mask, and it does not require an additive
position vector at the model input. It is a different insertion point from
the previous sinusoidal addition and the GPT-2 architecture's learned table.

Source: [Su et al., RoFormer, §§3.2.1–3.2.2, equations 13–16](https://arxiv.org/html/2104.09864v5).

---

<!-- .slide: id="rope-relative-offset" -->

## RoPE: relative offsets in the score

<div id="rope-relative-visual" class="position-diagram" data-diagram="position-relative" data-interactive="relative" data-source-url="https://arxiv.org/html/2104.09864v5"></div>

<div class="position-controls">
  <button id="rope-shift" type="button">Shift both +1</button>
  <button id="rope-gap" type="button">Gap +1</button>
  <span class="caption">Next reveals the scores, then the identity. Reset restores the example.</span>
</div>

Note:
Practice E02 · 2 minutes for prediction, then 3 minutes for the explanation.
Ask students to predict both dot products before the first reveal. The
content vectors q = k = (1, 0) are held fixed. Use the same toy θ = 30° as
the preceding slide. The query is purple and the key is teal.

Initially (m, n) = (3, 1) and (m+s, n+s) = (6, 4). In the first circle,
q rotates to (0, 1) and k to (sqrt(3)/2, 1/2). In the second, q is (−1, 0)
and k is (−1/2, sqrt(3)/2). Both dot products are 1/2. The key always precedes
the query, so these examples refer to entries allowed by a causal mask.

First reveal: both scores. Second reveal: the general identity. For one pair,
q_tilde_m^T k_tilde_n = q_m^T R(mθ)^T R(nθ) k_n
= q_m^T R((n−m)θ) k_n. The sign is n minus m for this column-vector
rotation convention. In our example n−m = −2. Although the controls display
the nonnegative gap m−n for readability, the formula keeps the signed offset.

Shift both +1 increases the right circle's shared shift s up to 6. Gap +1
increases both circles' query-key gap up to 4. Their scores remain equal at
any shared shift; increasing the gap from 2 to 3 changes both scores from
0.5 to 0, then gap 4 gives −0.5. Reset restores gap = 2 and s = 3 and hides
the answers. The scores shown are raw dot products, before scaling, masking,
and softmax. A cosine is not a monotonic distance-decay function.

For multiple pairs, sum these identities using each pair's own θ_i. The
relative-offset claim holds for the positional operation with fixed content
q_m and k_n. It does not assert invariance of an entire language model's
outputs when the prompt content or context changes. Notebook E02 checks
norm preservation and the multi-pair identity on nonidentical vectors.

Source: [RoFormer, §3.2, especially equation 16](https://arxiv.org/html/2104.09864v5).

---

<!-- .slide: id="position-methods" -->

## Two methods, two insertion points

<table class="position-compare">
  <thead><tr><th></th><th>Sinusoidal · 2017</th><th>RoPE · 2021</th></tr></thead>
  <tbody>
    <tr><th>Acts on</th><td>Token embeddings</td><td>Queries and keys</td></tr>
    <tr><th>Operation</th><td>Add a position vector</td><td>Rotate coordinate pairs</td></tr>
    <tr><th>Applied</th><td>At the stack input</td><td>In every attention layer</td></tr>
  </tbody>
</table>

<div class="position-diagram" data-diagram="position-placement" data-source-url="https://arxiv.org/html/2104.09864v5"></div>

Note:
2 minutes. Return to the architecture overview. Read both simplified paths
upward. The left path can use the positional input branch already drawn on
that slide. In the RoPE version, remove that additive positional branch and
place rotations on Q and K inside each attention head. The V branch bypasses
rotation. Projection boxes summarize the surrounding normalization/attention
block; the sketch does not change the pre-LN architecture shown earlier.

The 2017 Transformer uses fixed sinusoidal position vectors. The earlier GPT-2
overview instead has a learned position table; both are additive input
schemes. RoPE, introduced in the 2021 RoFormer preprint, uses fixed rotation
frequencies in its baseline form. Both leave the learned embedding and
projection parameters trainable. d_model controls the additive code's width;
d_h controls the per-head RoPE rotations. Full-width RoPE is illustrated.

Both causal language-model variants need a causal mask. Neither method by
itself guarantees accurate behavior beyond the training context length.
End by asking students to point to the addition node or the Q/K rotation
node on the appropriate architecture. The next component can now unpack
scaled dot-product attention and its mask.

Readings: [Attention Is All You Need, §§3.4–3.5](https://arxiv.org/html/1706.03762v7#S3.S5)
and [RoFormer, §§3.2.1–3.2.2](https://arxiv.org/html/2104.09864v5).

---

<!-- .slide: id="scaled-dot-product" -->

## One head: scaled dot-product attention

<div class="teaching-diagram" data-diagram="mha-single-head" data-source-url="https://arxiv.org/html/1706.03762v7#S3.S2"></div>

Note:
Component 02 · 3 minutes. Source: page 38, left diagram, of the instructor's
Desktop copy of lecture-05-slides-transformers.pptx. This redraw follows its
upward computation. The next slides motivate multiple heads, unpack their
computation, and connect them to the architectures on source pages 39–40.

Read the arrows upward. Q and K form a score matrix. Divide by sqrt(d_k),
apply the mask to the logits, run softmax across keys, and multiply by V.
The value branch enters only at the final weighted sum. The mask is optional
in the source's generic attention block; our causal language model uses it.

Four clicks reveal: (1) dot products and scaling, (2) causal mask, (3) row-wise
softmax, (4) value mixing and output. With token rows, Q and K are n by d_k,
V is n by d_v, the scores and weights are n by n, and Z is n by d_v. For a
query at t, keys j <= t remain available, including the query position itself.
Future logits become negative infinity before softmax, giving zero weights.
All retained weights in one query row sum to one. Padding masks are omitted.

Equivalently, Z = softmax_row(Q K^T / sqrt(d_k) + M) V, where M[t,j] is zero
for j <= t and negative infinity otherwise. If using RoPE, Q and K here are
already rotated separately within each head. Values are not rotated.
This is the row-vector notation used on the learned-projections slide.

The sqrt(d_k) factor moderates the scale of dot products; under independent,
zero-mean, unit-variance coordinate assumptions, the dot-product variance is
d_k. Do not present these assumptions as universal properties of trained
query/key vectors. We next motivate repeating this operation with separate projections.

Source: [Attention Is All You Need, §3.2.1 and Figure 2](https://arxiv.org/html/1706.03762v7#S3.S2).

---

<!-- .slide: id="why-multiple-heads" -->

## Why use multiple heads?

<div class="teaching-diagram" data-diagram="multihead-intuition" data-source-url="https://arxiv.org/html/1706.03762v7#S3.S2"></div>

Note:
Allow 2 minutes. Condenses pages 2–3 of the instructor's 26-slide Desktop lecture-05-slides-transformers.pptx inspected on October 10, 2026 (Asia/Shanghai). Keep the source's sentence and three questions: who gives, to whom, and what? Move the query from “gave” to the final word “food.” A causal query at “gave” cannot use the later recipient or object. At “food,” all seven shown positions are available, including the current position. The displayed words are toy tokens and the prefix can continue after “food.”

1. Reveal one head's row. It is one weighting of the available positions for one query. Its entries are nonnegative and sum to one. In this invented pattern, “I” gets the largest weight and suggests the giver.
2. Reveal two additional rows. They emphasize “Charlie” and “food,” illustrating the recipient and the thing given. Each head forms a separate context mixture before the outputs are combined. One head can assign weight to several words, but it still supplies one distribution over positions for this query. Multiple heads provide multiple distributions and separate value projections.

All three rows are invented normalized examples, not weights from a trained model or a claim that heads have assigned linguistic jobs. Actual patterns can overlap, and attention weights alone do not establish a causal explanation of a prediction. The numerical rows have sums 1.00, 1.00, and 1.00. Only their weights are shown; each head has its own projected query, keys, and values, developed next.

The benefit is separate learned mixtures. Avoid the source's suggestion that the issue is simply having too few parameters: with fixed d_model and d_k = d_v = d_model / h, the total Q/K/V and output-projection parameter count is independent of h, ignoring biases. The next slide's E03 checks that width split. Preserve why-multiple-heads between scaled-dot-product and multi-head-attention when integrating Lecture 05.

Source for the mechanism: [Attention Is All You Need, §3.2.2](https://arxiv.org/html/1706.03762v7#S3.S2).

---

<!-- .slide: id="multi-head-attention" -->

## Multi-head attention: project, attend, combine

<div class="teaching-diagram" data-diagram="mha-parallel-heads" data-source-url="https://arxiv.org/html/1706.03762v7#S3.S2"></div>

Note:
4 minutes plus Practice E03 · 1 minute. Source: page 38, right diagram, and
the multi-head block on page 39 of the Desktop PowerPoint. Read upward from
the same input matrix X. Heads 1, 2, and h stand for an arbitrary number of
heads; the ellipsis matters. The three drawn boxes do not mean h is three.
The preceding slide showed possible weight patterns; this slide explains
the learned projections that produce them and combine their outputs.

1. Each head has its own W_i^Q, W_i^K, and W_i^V. Every projection can use all
   d_model input features: this is not simply cutting raw X into fixed slices.
2. H_i = Attention(X W_i^Q, X W_i^K, X W_i^V). Each head independently forms
   its score matrix and normalizes its rows. In the causal model, every head
   applies the same causal visibility rule. Head patterns can differ, but
   distinct interpretable specializations are not guaranteed.
3. Concatenate H_1, …, H_h along features, preserving each token's row.
4. Multiply by the learned output matrix: Y = Concat(H_1, …, H_h) W^O.
   W^O mixes the heads' features. Concatenation itself does not average them.
5. Reveal the exercise answer after students calculate it.

Practice E03: use the original base-model setting d_model = 512, h = 8,
and d_k = d_v = d_model / h. Ask for the width per head; expected answer: 64.
Then ask verbally for the concatenated width: 8 × 64 = 512. For n = 4 input
positions, each H_i is 4 × 64, the concatenation is 4 × 512, and Y is 4 × 512.
Each W_i^Q, W_i^K, and W_i^V is 512 × 64; W^O is 512 × 512. n remains four.
The notebook checks row-preserving feature concatenation in this example.

This is standard multi-head self-attention with equal key and value widths.
The general output projection is (h d_v) by d_model; other configurations
need not make h d_v equal d_model. Grouped-query attention and multi-query
attention are separate variants, outside the scope of this introduction.

Optional discussion from page 7 of the updated 26-slide Desktop source:
“Do we need every trained head?” Michel, Levy, and Neubig, NeurIPS 2019,
[Are Sixteen Heads Really Better than One?, §§3–5](https://papers.neurips.cc/paper_files/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf).
Their trained translation and BERT models tolerated some head pruning;
translation cross-attention was more sensitive. The one-head-per-layer
ablation changed one layer at a time. A pruning result measures dependence
after training; it does not establish how many heads suffice during training
or justify reducing every layer to one head. Treat this as optional reading.

Source: [Attention Is All You Need, §3.2.2 and Figure 2](https://arxiv.org/html/1706.03762v7#S3.S2).

---

<!-- .slide: id="attention-in-architecture" -->

## Where multi-head attention is used

<div class="teaching-diagram" data-diagram="mha-architecture" data-source-url="https://arxiv.org/html/1706.03762v7#S3.S2"></div>

Note:
3 minutes. Source: model architecture on Desktop pages 39–40. The original
2017 diagram is simplified to show the origins of Q, K, and V. The course's
GPT-style model is alongside it. Every attention box represents multi-head
attention, with the mechanism just unpacked. Read both models upward.

Retain the distinction from page 5 of the updated 26-slide Desktop source:
heads operate in parallel within one attention sublayer; Transformer layers
run sequentially. Each layer uses the preceding layer's hidden states and
has its own learned projections. The repeated-block labels show depth,
while the previous slide's h labels the number of heads within one layer.

Reveal 1: encoder self-attention. Each source position can read all source
positions, aside from any padding mask. Q, K, and V come from the encoder's
current hidden states. The drawing groups the repeated encoder stack; its
final outputs supply the cross-attention memory.

Reveal 2: the original decoder has two attention sublayers. Causal self-
attention reads its own target prefix. Cross-attention takes Q from the
decoder states following that self-attention sublayer and K/V from the
encoder's final outputs. The cross-attention keys cover the source sequence;
they do not use the target sequence's causal triangular mask. A source
padding mask can still apply. Repeat the decoder block N times. During
training, target inputs are shifted right to predict the next target token.

Reveal 3: our GPT-style causal decoder attends within a single prefix and
has no encoder cross-attention sublayer. Repeat its attention/MLP block L
times, then apply the vocabulary head. The source/prefix states at the bottom
already include the embedding stage and the appropriate positional treatment.

This diagram deliberately isolates attention paths. Residual additions,
normalization, dropout, and embedding internals are omitted and remain as
described on the existing architecture overview. The 2017 paper uses
post-LayerNorm, whereas the earlier GPT-2-style overview uses pre-LayerNorm;
the simplified boxes here do not imply that these normalization orders match.

Source: [Attention Is All You Need, §§3.1–3.2.3 and Figure 1](https://arxiv.org/html/1706.03762v7#S3.S2).
GPT-style example: [OpenAI GPT-2 model implementation](https://github.com/openai/gpt-2/blob/master/src/model.py).

---

<!-- .slide: id="residual-addition" -->

## Residual addition keeps a direct path

<p data-source-url="https://arxiv.org/abs/1512.03385">A sublayer learns an update to the current representation.</p>

<p>$$y=\underbrace{x}_{\text{direct path}}+\underbrace{G(x)}_{\text{learned update}}$$</p>

<div class="columns">
<div>
<h3>Forward</h3>
<p>Add matching coordinates.<br>The shape stays $(B,T,d)$.</p>
<p>If $G(x)=0$, then $y=x$.</p>
</div>
<div class="fragment" data-fragment-index="0">
<h3>Backward</h3>
<p>$\displaystyle \frac{\partial y}{\partial x}=I+J_G(x)$</p>
<p>The identity term gives gradients a direct route.</p>
</div>
</div>

<p class="caption">In our pre-LN model, the update is $G(x)=F(\operatorname{LN}(x))$.</p>

Note:
Allow 2 minutes. Adapted from page 10 of the 25-slide Desktop
lecture-05-slides-transformers.pptx inspected on October 10, 2026
(Asia/Shanghai). Preserve the residual-learning motivation and replace the
paper/code screenshots with the forward and backward computations.

Start with the two terms in y = x + G(x). G denotes the complete update branch;
in our pre-LN model it includes normalization followed by attention or the FFN.
The input and update must have the same shape. The operation adds corresponding
features at corresponding token positions. It introduces no trainable
parameters and does not concatenate vectors or add token positions together.
Attention has already mixed the permitted positions inside its own branch.

Ask: if the update is zero, what leaves this addition? Expected response: x.
Then reveal the backward column. J_G is the Jacobian of the update with
respect to x, with the tensor regarded as a vector. For column gradients,
grad_x L = grad_y L + J_G(x)^T grad_y L. The identity contribution bypasses the
learned branch. It does not guarantee that total gradients stay large or small:
the two contributions can cancel or amplify. For example, G(x) = -x cancels
the identity derivative. The near-identity interpretation assumes a small
update; it is not an assertion about every trained block.

Source: [He et al., Deep Residual Learning for Image Recognition, §3.2](https://arxiv.org/abs/1512.03385).
The preprint appeared in 2015; the conference paper is CVPR 2016, correcting
the source slide's “CVPR, 2015.” The Transformer applies residual connections
around each attention and FFN sublayer; see
[Vaswani et al., §3.1](https://arxiv.org/html/1706.03762v7#S3.S1).

---

<!-- .slide: id="layernorm-features" -->

## LayerNorm normalizes one token's features

<p data-source-url="https://arxiv.org/abs/1607.06450">For each token vector $x\in\mathbb{R}^{d}$, compute its own statistics.</p>

$$\mu=\frac1d\sum_{j=1}^{d}x_j,\qquad
\sigma^2=\frac1d\sum_{j=1}^{d}(x_j-\mu)^2$$

$$\operatorname{LN}(x)_j=\gamma_j\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_j$$

<div class="fragment" data-fragment-index="0">
<ul>
<li>For $(B,T,d)$, normalize the last axis; keep tokens separate.</li>
<li>Learned $\gamma,\beta\in\mathbb{R}^{d}$ are shared across tokens.</li>
<li>Recenter and rescale features; $\epsilon$ keeps the denominator positive.</li>
</ul>
</div>

Note:
Allow 2 minutes. Adapted from page 11 of the same 25-slide Desktop revision.
Here d = d_model. The statistics belong to one batch item and one token
position: every (b,t) gets its own mean and variance over j. An affine
LayerNorm module has 2d parameters, reused at all token positions. Different
normalization modules in a block have separate parameters.

The motivation is to control shifts and scale in the features seen by the
next computation, while the learned scale and bias retain flexibility.
Avoid promising better accuracy, faster iterations, or stable training for
every configuration. After the affine transform, the output need not have
mean zero or variance one. Before it, the variance is sigma^2/(sigma^2+eps),
so even that variance is only approximately one for nonzero epsilon.

Use population variance: divide by d. With eps = 1e-5, place epsilon inside
the square root. The source's historical code uses x.std(-1) and adds epsilon
after the standard deviation; it should not be copied as an exact equivalent
of PyTorch LayerNorm. Its unqualified std call also uses the sample correction
in PyTorch. An explicit implementation is:

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, correction=0, keepdim=True)
    y = gamma * (x - mean) / torch.sqrt(var + eps) + beta

LayerNorm uses the input's statistics in both training and evaluation. No
batch running averages are needed. Normalizing across time would couple token
statistics, potentially letting future tokens change an earlier position.
The exercise after the order comparison checks this distinction on two rows.

Sources: [Ba et al., Layer Normalization, §3](https://arxiv.org/html/1607.06450v1#S3);
[PyTorch LayerNorm implementation and documented formula](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/normalization.py#L88).

---

<!-- .slide: id="normalization-order" -->

## Where does normalization go?

<p data-source-url="https://proceedings.mlr.press/v119/xiong20b.html">Let $F$ be attention or the feedforward network. Dropout is omitted.</p>

<table class="position-table">
<thead><tr><th>Architecture</th><th>One sublayer</th></tr></thead>
<tbody>
<tr><td>Original Transformer: post-LN</td><td>$y=\operatorname{LN}(x+F(x))$</td></tr>
<tr><td>Our GPT-style model: pre-LN</td><td>$y=x+F(\operatorname{LN}(x))$</td></tr>
</tbody>
</table>

<div class="fragment" data-fragment-index="0">
<p>Pre-LN leaves the direct path unchanged through each sublayer.</p>
<p>Our model applies a final LayerNorm after the block stack.</p>
</div>

<p class="caption">Normalization placement changes gradient flow and training behavior.</p>

Note:
Allow 2 minutes. This comparison resolves an inconsistency in source pages
10–11: “Add & Norm” follows the 2017 architecture, but the code screenshot
on page 10 normalizes the input before calling the sublayer. Read that code
as pre-LN; it is not the same computation as LN(x + F(x)).

Ask students to trace the unchanged x term in each formula before revealing
the explanation. In pre-LN, normalization lies inside the update branch.
In post-LN, normalization also acts on the result of the direct addition;
its full Jacobian is J_LN(x + F(x)) (I + J_F(x)). Thus the preceding slide's
identity derivative describes the addition node, not an unchanged path
through the whole post-LN sublayer. A final normalization in our pre-LN
model is applied once after the stack, before the vocabulary head.

The original model also applies dropout to sublayer outputs before adding
them; it is omitted here to isolate the order. The two sublayers in our
block use distinct LayerNorm modules, as on the GPT architecture overview.
Xiong et al. analyze initialization and warm-up under stated assumptions;
their findings motivate this comparison, not a universal ranking of all
pre-LN and post-LN models. Architecture, initialization, depth, learning
rate, and the training setup still matter.

Sources: [Vaswani et al., §3.1 and §5.4](https://arxiv.org/html/1706.03762v7#S3.S1);
[OpenAI GPT-2 implementation, block and model](https://github.com/openai/gpt-2/blob/master/src/model.py);
[Xiong et al., §§2–4](https://proceedings.mlr.press/v119/xiong20b.html).

---

<!-- .slide: id="add-norm-practice" -->

## Normalize, then add the update

<p class="exercise-meta" data-source-url="https://arxiv.org/abs/1607.06450">Practice E04 · 2 min · One token in a pre-LN sublayer</p>

<p>$x=(1,3)$, $\gamma=(1,1)$, $\beta=(0,0)$, $\epsilon=10^{-5}$.</p>
<p>The update is given: $F(\operatorname{LN}(x))=(2,-1)$.</p>

<ol>
<li>Find $\mu$, $\sigma^2$, $\operatorname{LN}(x)$, and the output $y$.</li>
<li>Can another token change $\operatorname{LN}(x)$?</li>
</ol>

<div class="fragment answer" data-fragment-index="0">
<p>$\mu=2,\;\sigma^2=1,\;\operatorname{LN}(x)\approx(-1,1)$;<br>$y=(1,3)+(2,-1)=(3,2)$.</p>
<p>No: each token's normalization uses only its own features.</p>
</div>

Note:
Practice E04, ungraded. Students calculate the two statistics and the final
sum before the reveal. The exact normalized vector is (-1,1)/sqrt(1.00001),
approximately (-0.999995,0.999995). The supplied update is already the result
of F(LN(x)); do not infer F or apply F again. The residual term is the original
x, so the output is (3,2), not LN(x) + (2,-1).

The last question isolates LayerNorm, whose statistics are local to one
token. An allowed earlier token can still affect this token's attention
update through F; independence of LayerNorm does not imply independence
of the complete attention sublayer.

Notebook E04 uses Python's standard library to check the values, the last
feature axis, a constant vector, and learned scale/bias. It perturbs another
token in a toy (B,T,d) tensor and verifies that this token's normalized vector
is unchanged. All inputs are invented; no model, training run, or GPU is used.
Source formula: [Ba et al., §3](https://arxiv.org/html/1607.06450v1#S3), with the
epsilon and population-variance convention of PyTorch LayerNorm.

---

<!-- .slide: id="feedforward-network" -->

## Feedforward network: transform each token

<div class="teaching-diagram" data-diagram="ffn-tokenwise" data-source-url="https://arxiv.org/html/1706.03762v7#S3.S3"></div>

Note:
Allow 3 minutes. Adapt page 6 of the 19-slide Desktop lecture-05-slides-transformers.pptx inspected on October 10, 2026 (Asia/Shanghai). The test deck previously named the MLP in its architecture without unpacking it. Read the three token columns upward; each applies the same parameters.

1. Expand with W_1 of shape d_model by d_ff and bias b_1 of width d_ff.
2. Apply sigma coordinate by coordinate. Use ReLU for the 2017 model and GELU for the GPT-2 architecture already shown. The source's code also has dropout, omitted here to isolate the computation.
3. Project with W_2 of shape d_ff by d_model and bias b_2 of width d_model. The output keeps the batch and token axes: (B,T,d_model). The 512 → 2048 → 512 example is the 2017 base model; the expansion ratio is a design choice.

Use row vectors, as in the source formula. Parameters are shared across positions within this FFN and differ between Transformer layers. In the pre-LN block, x_t here denotes the input after its LayerNorm; the residual addition remains outside the FFN. The supplied state can already contain context from attention. For fixed input states, this FFN introduces no additional communication across token positions.

Sources: [Vaswani et al., §3.3](https://arxiv.org/html/1706.03762v7#S3.S3), and
[GPT-2 implementation, mlp and block](https://github.com/openai/gpt-2/blob/master/src/model.py).

---

<!-- .slide: id="why-ffn" -->

## Why the FFN matters

<p data-source-url="https://arxiv.org/html/1706.03762v7#S3.S3">Attention gathers context. The FFN learns nonlinear features from it.</p>

<p class="caption">One GPT-style block, standard multi-head attention, $d_{\mathrm{ff}}=4d$.</p>

<table class="position-table">
<thead><tr><th>Module</th><th>Weight matrices</th><th>Parameters</th></tr></thead>
<tbody>
<tr><td>Attention</td><td>$W_Q,W_K,W_V,W_O$</td><td>$4d^2$</td></tr>
<tr><td>FFN</td><td>$d\times4d$ and $4d\times d$</td><td>$8d^2$</td></tr>
</tbody>
</table>

<div class="fragment" data-fragment-index="0">
<p>FFN share: $\frac{8d^2}{4d^2+8d^2}=\frac23\approx67\%$.</p>
<img class="ffn-parameter-share" src="assets/ffn-parameter-share.svg" width="1152" height="56" alt="Block matrix parameters at FFN width 4d: attention one third, FFN two thirds.">
</div>

<p class="caption">Block matrix parameters. Embeddings, biases, and normalization are excluded.</p>

Note:
Allow 2 minutes. Explain why the architecture dedicates a substantial module to processing each token after attention has gathered context. W_1 builds learned combinations of features, the activation changes their response nonlinearly, and W_2 combines the resulting features into an update of the original width. A wider hidden space supplies more such nonlinear features. Removing the activation would collapse these two affine maps into one affine map. Attention's softmax is already nonlinear; do not claim that a Transformer without its FFN would become entirely linear.

Derive the displayed ratio from matrix shapes. Standard multi-head attention has four aggregate d by d projections: Q, K, V, and output. Summing over the heads introduces no extra factor of h. A two-matrix FFN has d*d_ff + d_ff*d weights. At d_ff = 4d, attention contributes 4d^2 and the FFN 8d^2, so the FFN occupies two thirds of these block matrix parameters. For d = 512, the counts are 1,048,576 and 2,097,152. These are calculated counts, not a measured training result.

“Capacity” here means parameter count, not an exact measure of reasoning ability, stored facts, memory use, or inference time. The ratio can approximate the whole-model parameter share when the repeated blocks dominate. Counting embeddings, position tables, output heads, biases, and normalization changes that share. Changing d_ff, using gated FFNs or grouped-query attention, or adding cross-attention also changes the accounting. In particular, the original encoder-decoder's decoder has an additional attention sublayer; the displayed denominator describes our GPT-style block.

This calculation follows the shapes in [Vaswani et al., §§3.2.2–3.3](https://arxiv.org/html/1706.03762v7#S3.S2)
and the fourfold MLP expansion in [GPT-2's block implementation](https://github.com/openai/gpt-2/blob/master/src/model.py).

---

<!-- .slide: id="ffn-practice" -->

## Apply the FFN to one token

<p class="exercise-meta" data-source-url="https://arxiv.org/html/1706.03762v7#S3.S3">Practice E05 · 2 min · Invented weights, ReLU, zero biases</p>

<p>$$x=(1,-2),\qquad W_1=\begin{pmatrix}1&-1&2\\0&1&-1\end{pmatrix},\qquad W_2=\begin{pmatrix}1&0\\0&1\\1&-1\end{pmatrix}$$</p>

<ol>
<li>Compute $xW_1$, then $\operatorname{ReLU}(xW_1)W_2$.</li>
<li>Can changing another token's FFN input change this output?</li>
</ol>

<div class="fragment answer" data-fragment-index="0">
<p>$xW_1=(1,-3,4)$; after ReLU: $(1,0,4)$.<br>$\operatorname{FFN}(x)=(5,-4)$.</p>
<p>No. This FFN processes each supplied token vector independently.</p>
</div>

Note:
Practice E05, ungraded. Use the row-vector convention from the preceding diagram. The toy widths are 2 → 3 → 2, showing that the hidden width need not be four times the model width. Both biases are zero. First multiply (1,-2) by W_1 to obtain (1,-3,4); zero the negative coordinate; then (1,0,4) W_2 = (5,-4). This is the isolated FFN output before any residual addition.

Hold this token's supplied input fixed when changing another token. The FFN's lack of cross-token mixing is a statement about this isolated operation. Earlier attention can still change the contextual input received by the FFN.

Notebook E05 verifies the intermediate and final vectors, token isolation, the effect of the nonlinearity, and the two-thirds block parameter count from the previous slide. Its small deterministic examples use the standard library and no trained model. With the opposite input (-1,2), the FFN output is (0,3); the two outputs do not sum to FFN(0), illustrating nonlinearity.

Keep feedforward-network, why-ffn, and ffn-practice together when integrating Lecture 05, and reconcile E05 with that deck's existing exercise numbering. Source formula: [Vaswani et al., §3.3](https://arxiv.org/html/1706.03762v7#S3.S3).

---

<!-- .slide: id="input-tokenization" -->

## From raw text to model inputs

<div class="teaching-diagram" data-diagram="tokenizer-inputs" data-source-url="https://aclanthology.org/P16-1162/"></div>

Note:
3 minutes. Layout requested by the instructor: combine Desktop pages 41–42
on the left and page 43 on the right. This is a recap of Lecture 01's
tokenization and Lecture 03's embedding lookup, now placed in the Transformer.

Left: raw text is segmented into tokens, mapped to token IDs, and used to
look up trainable embedding vectors. Tokens can be words, characters, or
subwords. The illustrative segmentation “We / are / play / ing” and IDs
10, 20, 30, 40 are invented, not output from BPE, WordPiece, or a named model.
Spacing markers and tokenizer-specific normalization are omitted. Subword
vocabularies can contain both whole words and word fragments. The 2017-style
additive position formula recalls the source page 41 figure; e denotes the
scaled token embedding used earlier. RoPE uses the separate Q/K insertion
point already discussed, rather than this additive position branch.

Corrected from source page 41: input embeddings are not necessarily obtained
from a separate pretrained embedding model. A Transformer normally has an
embedding table trained with its other parameters; a pretrained model comes
with an already-trained table. Vocabulary learning is a different training
process from learning neural embedding weights. The diagram's colored vector
cells are schematic, with no claimed numeric encoding values.

Right: list the three algorithms from source page 43. BPE builds units by
iterative pair merging; the Unigram tokenizer fits a probabilistic subword
model; WordPiece builds a subword vocabulary and segments using it. These
are three alternatives, not three stages of one tokenizer. Avoid implying
that the slide title “Input (BPE)” makes Unigram and WordPiece BPE algorithms.

The learner consumes a training corpus and produces a vocabulary plus any
associated rules or model parameters: BPE merge ranks, Unigram probabilities,
or the WordPiece vocabulary/segmentation rules. The segmenter applies that
fitted tokenizer to new text and returns token IDs. Reuse the same tokenizer
configuration used for the model; do not refit it on each test sentence.

Reveal 1: text, tokens, IDs, embeddings. Reveal 2: three algorithms. Reveal 3:
learner and its output. Reveal 4: segmenter, new text, and use of that output.

Sources: [Sennrich et al. (2016), §3](https://aclanthology.org/P16-1162/),
[Kudo (2018), §3.2](https://aclanthology.org/P18-1007/), and
[Schuster & Nakajima (2012)](https://research.google/pubs/japanese-and-korean-voice-search/).
Embedding treatment: [Attention Is All You Need, §3.4](https://arxiv.org/html/1706.03762v7#S3.S4).

---

<!-- .slide: id="original-transformer-paper" -->

## Attention Is All You Need · 2017

<p data-source-url="https://arxiv.org/html/1706.03762v7">Vaswani et al., NeurIPS. An encoder–decoder for translation without recurrent or convolutional layers.</p>

<div class="columns">
<div>
<h3>The original model</h3>
<ul>
<li>Six encoder and six decoder layers.</li>
<li>Self-attention, cross-attention, and position-wise ReLU networks.</li>
<li>Post-LN residual blocks and sinusoidal positions.</li>
</ul>
</div>
<div>
<h3>Why remove recurrence?</h3>
<table>
<thead><tr><th>For $n$ tokens</th><th>RNN</th><th>Attention</th></tr></thead>
<tbody>
<tr><td>Sequential steps</td><td>$O(n)$</td><td>$O(1)$</td></tr>
<tr><td>Path length</td><td>$O(n)$</td><td>$O(1)$</td></tr>
</tbody>
</table>
</div>
</div>

<p class="caption">Parallel training, sequential generation. Shorter paths do not guarantee stable gradients.</p>

Note:
Allow 2 minutes. This section asks what evidence established the architecture's usefulness, after the preceding slides explain its components. The original paper is about sequence-to-sequence translation; the earlier GPT-style causal decoder is a later adaptation. In the original model each decoder block also reads encoder states through cross-attention. Its feedforward sublayer uses two affine maps with ReLU between them, independently at each position. No recurrent or convolutional layers does not mean no feedforward layers.

The comparison concerns sequence-axis dependencies for a layer, not a promise of constant wall-clock runtime or fewer optimization updates. Projection and feedforward costs remain; dense attention materializes quadratic scores. Teacher forcing supplies target input tokens together for training, whereas autoregressive generation still generates one new token at a time. Transformers can still have vanishing or exploding gradients; initialization, residuals, normalization, depth, and optimization matter.

Source mapping: pages 4 and 16 of the 17-slide Desktop PowerPoint snapshot used on October 10, 2026 (Asia/Shanghai). Its historical citation-count screenshot is omitted because it is a dated popularity statistic, not experimental evidence. Sources: Vaswani et al., §§3–4, Figure 1 and Table 1, https://arxiv.org/html/1706.03762v7. Table 1 holds the layer type and representation width fixed while comparing sequence length. The source's absolute claims about gradient problems and fewer training steps are replaced by the narrower supported claims here.

---

<!-- .slide: id="paper-experimental-setup" -->

## Translation data and model configurations

<div class="columns" data-source-url="https://arxiv.org/html/1706.03762v7#S5">
<div>
<h3>WMT 2014</h3>
<p><strong>English–German:</strong> 4.5M pairs; about 37k shared BPE tokens.</p>
<p><strong>English–French:</strong> 36M pairs; 32k wordpieces.</p>
<p>Length-based batches: about 25k source + 25k target tokens.</p>
<p>Eight P100 GPUs.<br>Base: 100k steps / 12 hours.<br>Big: 300k steps / 3.5 days.</p>
</div>
<div>
<h3>Base and big</h3>
<table>
<thead><tr><th>Setting</th><th>Base</th><th>Big</th></tr></thead>
<tbody>
<tr><td>$d_{\mathrm{model}}$</td><td>512</td><td>1024</td></tr>
<tr><td>$d_{\mathrm{ff}}$</td><td>2048</td><td>4096</td></tr>
<tr><td>Heads</td><td>8</td><td>16</td></tr>
<tr><td>Dropout</td><td>0.1</td><td>0.3*</td></tr>
</tbody>
</table>
<p>Both: six layers per stack;<br>$d_k=d_v=64$; label smoothing $0.1$.</p>
</div>
</div>

<p class="caption">*Big uses dropout 0.1 for English–French. Training settings: §5 and Table 3.</p>

Note:
Allow 3 minutes. Source pages 5 and 8. The datasets, vocabularies, batching, hardware, and optimization budget define the experiment. M means million sentence pairs, not tokens. N counts layers in each stack, so N=6 means six encoder layers and six decoder layers. The attention projections split the model width into h equal heads; d_model/h remains 64 for base and big. The FFN expands features to four times the model width and projects back. Table 3 reports about 65M and 213M parameters for the full base and big translation models; the next slide is a different encoder-only exercise.

Training details for follow-up reading: Adam beta1=0.9, beta2=0.98, epsilon=1e-9; 4,000 linear warm-up steps, then inverse-square-root decay. The learning rate is d_model^(-1/2) min(step^(-1/2), step * 4000^(-3/2)). Dropout is applied to sublayer outputs and embedding-plus-position sums. Label smoothing uses epsilon_ls=0.1. The English–French big run uses dropout 0.1 rather than the 0.3 general big configuration. The paper reports about 0.4 seconds per base step and 1 second per big step on its hardware, not expected performance on a student's machine.

Sources: Vaswani et al., §§3.2–3.5 and 5, Table 3, https://arxiv.org/html/1706.03762v7#S5. These are historical paper experiments, not experiments performed for this lecture.

---

<!-- .slide: class="exercise" id="paper-parameter-count" -->

<p class="exercise-meta" data-source-url="https://arxiv.org/html/1706.03762v7#S3">Practice E06 · 3 minutes · count an encoder</p>

## Parameters in an encoder-only model

<p>$N=12$, $d=768$, $h=12$, $|V|=30{,}522$; $d_k=d_v=64$, $d_{\mathrm{ff}}=4d$.</p>

<p>Include biases and LayerNorm; sinusoidal positions; no prediction head.</p>

<div class="columns">
<div class="fragment" data-fragment-index="0">
<table>
<thead><tr><th>Per layer</th><th>Parameters</th></tr></thead>
<tbody>
<tr><td>Q/K/V + output</td><td>$4d^2+4d$</td></tr>
<tr><td>FFN</td><td>$8d^2+5d$</td></tr>
<tr><td>Two LayerNorms</td><td>$4d$</td></tr>
</tbody>
</table>
</div>
<div class="answer fragment" data-fragment-index="1">
<h3>Combine the terms</h3>
<p>$P=|V|d+N(12d^2+13d)$</p>
<p>Embeddings $|V|d$: <strong>23,440,896</strong><br>Encoder stack: <strong>85,054,464</strong></p>
<p>Total: <strong>108,495,360 ≈ 108.50M</strong></p>
</div>
</div>

Note:
Allow 3 minutes, using the component tally as a scaffold. Source page 9 specifies N=12, d=768, h=12, V=30522, dk=d/h, and d_ff=4d. This is not the paper's six-layer base model. The original prompt does not specify biases, learned positions, segment embeddings, or an output head; an exact parameter count requires those choices. Here we include all linear biases, learned gamma/beta in both LayerNorms, one token-embedding table, and fixed sinusoidal positions. We exclude embedding normalization, segment embeddings, pooling, final extra normalization, and all prediction heads. Do not label this as an exact BERT parameter count.

Each attention layer has three d-by-d projections and one d-by-d output map, each with d bias entries. Splitting a fixed width into h heads does not multiply this count by h again. The FFN is ReLU(xW1+b1)W2+b2: shapes d-by-4d and 4d-by-d, with 4d and d biases. Each LayerNorm contributes d scale and d shift entries. Residual additions, softmax, dropout, and sinusoidal positions add no learned parameters.

One layer has 7,087,872 parameters; twelve have 85,054,464. The token table adds 23,440,896, giving 108,495,360. If all linear and LayerNorm bias vectors were omitted but LayerNorm scales retained, the count would change; keep the stated scope when comparing answers. Notebook E06 counts explicit tensor shapes; the numerical checks compare with an independent PyTorch module count without allocating model weights.

Source: instructor's encoder-count exercise, page 9; architecture definitions in Vaswani et al., §§3.1–3.4. These dimensions define a classroom calculation, not a measured model run.

---

<!-- .slide: id="paper-translation-results" -->

## Translation quality and training cost

<p data-source-url="https://arxiv.org/html/1706.03762v7#S6.S1">WMT <strong>newstest2014 test set</strong> · BLEU, higher is better</p>

<table>
<thead><tr><th>Model</th><th>EN–DE</th><th>EN–FR</th><th>EN–FR training FLOPs</th></tr></thead>
<tbody>
<tr><td>ConvS2S, single</td><td>25.16</td><td>40.46</td><td>$1.5\times10^{20}$</td></tr>
<tr><td>ConvS2S, ensemble</td><td>26.36</td><td>41.29</td><td>$1.2\times10^{21}$</td></tr>
<tr><td>Transformer base</td><td>27.3</td><td>38.1</td><td>$3.3\times10^{18}$</td></tr>
<tr><td><strong>Transformer big</strong></td><td><strong>28.4</strong></td><td><strong>41.8</strong></td><td>$2.3\times10^{19}$</td></tr>
</tbody>
</table>

<p class="fragment" data-fragment-index="0">ConvS2S / big compute ratio: <strong>6.5 (single), 52.2 (ensemble)</strong>.</p>

<p class="caption">Table 2. Estimated training FLOPs across systems; these ratios do not measure inference speed.</p>

Note:
Allow 3 minutes. Source pages 6–7. The numeric table replaces two bar charts and the full comparison screenshot. Use the values in the paper's Table 2: the source's EN–FR chart does not exactly match that table, and versions of the prose report a different EN–FR score. These slide values explicitly follow Table 2, including 41.8 for big and 38.1 for base. Do not silently combine numbers from the chart or narrative with the table.

The source's “50 times faster” statement conflates compute and time and leaves the comparator ambiguous. Calculate 1.5e20 / 2.3e19 = 6.5217 for the single ConvS2S model, and 1.2e21 / 2.3e19 = 52.1739 for the ensemble. The paper estimates training FLOPs from wall-clock training time, GPU count, and estimated sustained single-precision FLOPs per GPU. These ratios are not per-token inference throughput and are not a matched-hardware speed measurement. The base row also makes clear that the smaller model does not beat every baseline on both tasks.

The original Table 2 additionally includes ByteNet, Deep-Att + PosUnk, GNMT + RL, and MoE; the selected rows retain both single-model and ensemble comparisons needed for the source's claim. Its best prior EN–DE ensemble is 26.36 versus big's 28.4. BLEU compares generated translations with references; it is not perplexity and its absolute value should not be compared across the two language pairs as if they were the same test.

The translation evaluation averages the last 5 checkpoints for base or 20 for big, then uses beam size 4 and length penalty 0.6. Source: Vaswani et al., §6.1 and Table 2, https://arxiv.org/html/1706.03762v7#S6.S1. Numerical cost ratios are also checked in the notebook.

---

<!-- .slide: id="paper-ablations" -->

## Which choices mattered in the paper?

<p data-source-url="https://arxiv.org/html/1706.03762v7#S6.S2">English–German <strong>newstest2013 development set</strong> · Table 3</p>

<div class="columns">
<div>
<h3>Architecture</h3>
<table>
<thead><tr><th>Setting</th><th>PPL ↓</th><th>BLEU ↑</th></tr></thead>
<tbody>
<tr><td>Base: 8 heads</td><td>4.92</td><td>25.8</td></tr>
<tr><td>One head</td><td>5.29</td><td>24.9</td></tr>
<tr><td>FFN width 4096</td><td>4.75</td><td>26.2</td></tr>
</tbody>
</table>
</div>
<div>
<h3>Training and positions</h3>
<table>
<thead><tr><th>Change</th><th>PPL ↓</th><th>BLEU ↑</th></tr></thead>
<tbody>
<tr><td>No dropout</td><td>5.77</td><td>24.6</td></tr>
<tr><td>No smoothing</td><td>4.67</td><td>25.3</td></tr>
<tr><td>Learned positions</td><td>4.92</td><td>25.7</td></tr>
</tbody>
</table>
</div>
</div>

<p class="caption">Lower perplexity need not mean better translation BLEU. These are results under this training setup, not universal rankings.</p>

Note:
Allow 3 minutes. Source page 10. This is an ablation table on the development set, with no checkpoint averaging; do not compare its 25.8 base BLEU directly with Table 2's 27.3 test BLEU. PPL is per wordpiece under this BPE vocabulary, not per word. Each listed row changes the named setting while holding unspecified base settings fixed. The larger FFN also increases parameters and compute, so its improvement is not a free architectural gain.

Other rows in the source table remain useful for discussion: head counts 4, 16, and 32 give BLEU 25.5, 25.8, and 25.4; more heads are not monotonically better. Reducing key width to 16 or 32 gives 25.1 or 25.4. Encoder/decoder depth 2, 4, or 8 gives 23.7, 25.3, or 25.5. Model width 256 or 1024 gives 24.5 or 26.0. FFN width 1024 gives 25.4. Dropout 0.2 gives 25.5; label smoothing 0.2 gives 25.7. Big uses more capacity and 300K steps, so its 26.4 dev BLEU and 4.33 PPL are not a single-factor ablation of the 100K-step base.

Ask why removing smoothing gives lower PPL but lower BLEU. Label smoothing changes the objective and confidence of the predicted distribution; the metrics measure different things. Learned positions and sinusoids perform almost identically here, which is not a proof of equal length extrapolation. Source: Vaswani et al., §6.2 and Table 3, https://arxiv.org/html/1706.03762v7#S6.S2.

---

<!-- .slide: id="paper-constituency-parsing" -->

## Beyond translation: constituency parsing

<div class="columns" data-source-url="https://arxiv.org/html/1706.03762v7#S6.S3">
<div>
<h3>Sentence → phrase structure</h3>
<p>“I drink coffee with milk”</p>
<p>Is “with milk” part of the noun phrase, or attached to the verb phrase?</p>
<p>A sequence of brackets and labels can encode the tree.</p>
<p>Four-layer Transformer; $d=1024$.</p>
</div>
<div>
<h3>WSJ section 23 · F1 ↑</h3>
<table>
<thead><tr><th>Training setting</th><th>F1</th></tr></thead>
<tbody>
<tr><td>Transformer, WSJ only (~40k sentences)</td><td>91.3</td></tr>
<tr><td>Transformer, semi-supervised (~17M sentences)</td><td>92.7</td></tr>
<tr><td>Dyer et al., WSJ-only discriminative</td><td>91.7</td></tr>
</tbody>
</table>
</div>
</div>

<p class="caption">Table 4. Evidence of transfer to structured prediction; training data and modeling assumptions differ across rows.</p>

Note:
Allow 2 minutes. Source pages 11–12. The source contrasts attaching a prepositional phrase inside a noun phrase (“coffee with milk”) with attaching it to an action (“drink coffee with friends”). The invented example here motivates the structure; it is not a model prediction or a reported test case. Constituency F1 evaluates labeled spans against reference parses, rather than n-gram overlap as BLEU does. An output sequence can serialize a tree with brackets and labels.

The paper trains a four-layer model of width 1024 on about 40K WSJ training sentences, and also in a semi-supervised setting with approximately 17M sentences. It uses vocabularies of 16K and 32K, respectively. Development choices use WSJ section 22; the table reports section 23. Parsing uses beam size 21 and length penalty 0.3, with a longer maximum output length to accommodate trees. These are adaptations to a different task, not an unchanged pretrained translation model evaluated without training.

The remaining original table provides context: earlier WSJ-only discriminative systems report 88.3, 90.4, 90.4, and 91.7; semi-supervised systems report 91.3, 91.3, 92.1, and 92.1; the Transformer is 91.3 or 92.7 in its two settings. Multi-task Luong et al. reports 93.0, and generative Dyer et al. reports 93.3. Thus the experiment demonstrates broader applicability, not that the Transformer wins every parsing setting. Source: Vaswani et al., §6.3 and Table 4, https://arxiv.org/html/1706.03762v7#S6.S3.

---

<!-- .slide: id="paper-efficient-attention" -->

## Later work on long-sequence cost

<p data-source-url="https://arxiv.org/abs/2001.04451">Dense attention has $n\times n$ scores per head. These papers appeared in <strong>2020</strong>.</p>

<table>
<thead><tr><th>Method</th><th>Main idea</th><th>Cost in $n$</th></tr></thead>
<tbody>
<tr><td>Dense</td><td>Compare every query with every key</td><td>$O(n^2)$</td></tr>
<tr><td><a href="https://arxiv.org/abs/2001.04451">Reformer</a></td><td>Hash vectors; attend within buckets</td><td>$O(n\log n)$</td></tr>
<tr><td><a href="https://arxiv.org/abs/2006.04768">Linformer</a></td><td>Compress the K/V sequence axis: $n\to k$</td><td>$O(nk)$</td></tr>
</tbody>
</table>

<p>Reversible residuals reduce Reformer's saved activations across depth.</p>

<p class="caption">Widths and method settings held fixed; Linformer is linear in $n$ for fixed $k$. These methods change the attention computation.</p>

Note:
Allow 2 minutes. Source pages 13–14. Distinguish a long-sequence limitation of the 2017 architecture from later responses to it. With feature width fixed, materialized dense score storage is quadratic in n; the attention multiplication cost also carries a feature-width factor. Reformer uses locality-sensitive hashing, sorting, and bucket/chunk attention to avoid evaluating every pair. Its complexity statement assumes the method's bucket and hash settings; hashing is an approximation to all-pairs attention, not a guarantee that every relevant pair survives. Its reversible residual layers reconstruct intermediate states to reduce the layer-depth contribution to stored activations; this does not make all training memory independent of depth.

Linformer projects the sequence dimension of K and V from n to k, so a query attends to k projected positions. At fixed feature widths this gives O(nk) work and score storage, linear in n if k is held fixed. Low-rank approximations can change model outputs; neither paper is an exact implementation of arbitrary dense attention with an unconditional speed guarantee. The original source's comparison table also lists sparse attention at O(n sqrt(n)) and recurrence at O(n) for fixed width; the three rows here focus on the two named follow-up papers.

Sources: Kitaev, Kaiser, and Levskaya, Reformer (ICLR 2020), §§2–3, https://arxiv.org/abs/2001.04451; Wang et al., Linformer (2020), §3, https://arxiv.org/abs/2006.04768. These are separate papers, not sections of Attention Is All You Need.

---

<!-- .slide: id="references" -->

## Architecture changes and evidence

<div class="columns" data-source-url="https://aclanthology.org/2021.emnlp-main.465/">
<div>
<h3>Read the evidence with the design</h3>
<ul>
<li>Architecture, data, and training budget jointly determine the result.</li>
<li>Keep test results separate from development ablations.</li>
<li>Compare a modification in the target implementation and task.</li>
</ul>
<p>Narang et al. (EMNLP 2021) found that most tested modifications did not improve performance meaningfully in their shared evaluation.</p>
</div>
<div class="references">
<h3>Reading</h3>
<ul>
<li><a href="https://arxiv.org/html/1706.03762v7">Attention Is All You Need</a>: §§3–6</li>
<li><a href="https://arxiv.org/abs/2001.04451">Reformer</a>; <a href="https://arxiv.org/abs/2006.04768">Linformer</a></li>
<li><a href="https://aclanthology.org/2021.emnlp-main.465/">Do Transformer Modifications Transfer?</a></li>
<li><a href="http://nlp.seas.harvard.edu/annotated-transformer/">The Annotated Transformer</a>: code walkthrough</li>
</ul>
</div>
</div>

Note:
Allow 2 minutes. Source pages 15 and 17, plus the qualified RNN comparison already covered at this section's start. Narang et al. study many modifications across implementations and applications in a common experimental setting. Their conclusion is scoped to those comparisons; it does not say that no later improvement works. Improvements that did carry over were often relatively small or developed in the same codebase, which motivates controlled reproduction and re-evaluation rather than adopting every proposed change. The source's large result screenshot is available in the linked paper; read it by task and metric, not as a universal leaderboard.

Ask students to name the comparison behind a claim: BLEU on which test, F1 in which data regime, training FLOPs or inference speed, fixed compute or larger model? The key paper contribution combines parallel sequence processing, an encoder–decoder built around attention, and evidence from translation and parsing. Later long-context methods and modification studies test different questions.

Primary source: Narang et al., Do Transformer Modifications Transfer Across Implementations and Applications?, EMNLP 2021, https://aclanthology.org/2021.emnlp-main.465/. Earlier source sections and exact tables are cited in the preceding slide notes.

Source-page 17 optional resources, grouped for self-study: Harvard's Annotated Transformer (http://nlp.seas.harvard.edu/annotated-transformer/); Jay Alammar's Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/) and recurrent-translation visualization (https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/); Rasa videos on self-attention (https://www.youtube.com/watch?v=yGTUuEx3GkA), Q/K/V (https://www.youtube.com/watch?v=tIvKXrEDMhk), multiple heads (https://www.youtube.com/watch?v=23XUv0T9L5c), and architecture (https://www.youtube.com/watch?v=EXNBy8G43MM). Additional historical links in that source: https://cs.uwaterloo.ca/~ppoupart/teaching/cs480-spring19/schedule.html, https://primo.ai/index.php?title=Attention, https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html, https://koaning.io/, https://colah.github.io/posts/2015-08-Understanding-LSTMs/, and https://people.cs.umass.edu/~miyyer/cs685_f22/slides/04-attention.pdf. These are supplementary explanations, not sources for the reported numerical results. The old “Next lecture: LLMs and Benchmarks” footer is omitted because the current course sequence continues with pretraining and decoding.
