# Lecture 05: Historical examples and further reading

These notes retain background from the instructor's 2025 *Attention and
Transformers* lecture. They supplement the [current slides](index.html) and
[notebook](lecture-05-exercise.ipynb); they add no graded work.

## Interactive Transformer walkthroughs

Use these resources alongside the lecture's numerical examples. Each site
illustrates its own model and parameter choices. Use the course notebook to
reproduce the seven-token model and its 4,432-parameter count.

| Resource | What to follow | Lecture connection |
| --- | --- | --- |
| [Transformer Explainer, Georgia Tech / Polo Club](https://poloclub.github.io/transformer-explainer/) | In GPT-2 small, follow one query through Q/K scores, the causal mask, softmax weights, and the weighted values. | [Masking before softmax](index.html#/mask-before-softmax), after the hand calculation |
| [LLM Visualization, Brendan Bycroft](https://bbycroft.net/llm) | Explore a GPT-style network in 3D. The working small model sorts letters. Follow the token and feature axes through embeddings, attention, the feedforward computation, and readout. | [Complete decoder](index.html#/full-model), before the parameter-count exercise |
| [LLM-Visualized](https://www.llm-visualized.com/?token=4&generation=0&kvCache=0) | Use the supplied starting view with KV caching off. Follow the matrix operations of one forward pass, then the next-token generation loop. | [Attention storage and generation](index.html#/attention-cost); caching is a later Lecture 13 topic |

For each visualization, identify the current token, the positions it can
attend to, and the operation that produces vocabulary logits. Distinguish
softmax over attention keys from softmax over the output vocabulary.
For Bycroft's model, the [author's code and explanation](https://github.com/bbycroft/llm-viz)
describe the letter-sorting example and the larger model views.

The external sites need a network connection. The lecture's diagrams,
position demonstration, and CPU notebook remain available locally. If a
site cannot load, use the matching slide and notebook calculation.

## From a context vector to attention

An encoder–decoder system maps a source sequence to a target sequence. In a
recurrent system, using only the last encoder state forces that state to carry
all the information needed by the decoder. Attention lets each decoder query
combine several encoder states. Lecture 05 introduces this motivation and the weighted-context calculation,
then distinguishes encoder self-attention, decoder self-attention, and
decoder-to-encoder cross-attention. In cross-attention, Q comes from the
decoder, while K and V come from the encoder. See
[Vaswani et al., §§3.1–3.2.3](https://arxiv.org/abs/1706.03762).

For the original learned-alignment formulation, read
[Bahdanau et al., §3](https://arxiv.org/abs/1409.0473). The alignment network
uses the previous decoder state and source annotations. Compare it with
[Luong et al., §3](https://aclanthology.org/D15-1166/): the current decoder
state supplies the query, with dot, general, or concat scoring. Distinguish
these recurrent update conventions from the shared score/softmax/weighted-sum
mechanism. Our two-state worked example uses unscaled dot scoring.

The source lecture's noisy time-series example illustrates weighted averaging:
weights can emphasize observations relevant to a query. A local smoothing
kernel emphasizes nearby observations by design. Learned text attention can
instead connect distant tokens. Neither physical proximity nor a predefined
linguistic role determines the learned weights. Our `bank of the river` example
illustrates contextualization; its browser vectors are invented, with no claim
that they encode word meanings.

## Tokenization is a prerequisite

The source's eight-slide BPE walkthrough is already covered in
[Lecture 01](../lecture-01/index.html). Review how training learns an ordered
merge list and how tokenization applies those merges to new text. Embedding
lookup is a separate operation: token IDs select trainable rows. A Transformer
does not require a separately pretrained word-vector model. The original
architecture describes learned embeddings in
[Vaswani et al., §3.4](https://arxiv.org/abs/1706.03762).

## Position information and head behavior

The notebook's permutation identity assumes **unmasked** attention with no
position-dependent term. A causal mask already supplies an ordering constraint.
Consequently, the identity does not prove that every causal language model
needs explicit position embeddings. Haviv et al. study causal models trained
without them and find that position information can still be learned. Our
baseline uses a learned position table to make the mechanism easy to inspect.
See [Haviv et al. (2022)](https://arxiv.org/abs/2203.16634), especially the model
comparison and position-probing experiments. RoPE is a separate way to place
relative offsets in Q/K dot products;
[Su et al., §3.2](https://arxiv.org/abs/2104.09864) gives the rotation derivation.

For a pair of column-vector coordinates, let `R_p` rotate by a frequency
times position `p`. Then `q'_p = R_p q_p` and `k'_s = R_s k_s` give
`q'_p^T k'_s = q_p^T R_(s-p) k_s`, because `R_p^T R_s = R_(s-p)`.
Standard RoPE rotates Q and K, while V is unchanged. This optional extension
is separate from the notebook's learned absolute position table.

The original Transformer's appendix includes attention visualizations for
particular encoder heads and sentences. Such examples can suggest a hypothesis
about a head; they do not establish a fixed grammatical role for every head.
Michel et al. found that many heads could be pruned with little performance
loss in the models and tasks they studied. The result motivates measurement,
not a universal instruction to use one head. See
[Vaswani et al., attention visualizations](https://arxiv.org/abs/1706.03762) and
[Michel et al. (2019), pruning experiments](https://arxiv.org/abs/1905.10650).

## Translation and parsing in the original paper

The 2017 model has encoder and decoder stacks. Its decoder includes causal
self-attention and cross-attention. It uses sinusoidal positions, post-LN
sublayers, dropout, and label smoothing. Our small pre-LN decoder omits the
encoder and cross-attention, uses learned positions, and has no dropout or
label smoothing. These choices must be stated when comparing results.

This optional historical English–French comparison is pinned to
[arXiv v7, Table 2](https://arxiv.org/html/1706.03762v7#S6.T2): single-model
ConvS2S reports 40.46 BLEU and an estimated 1.5 × 10²⁰ training FLOPs;
Transformer big reports 41.8 BLEU and 2.3 × 10¹⁹ FLOPs. The ratio of estimated
computation is about 6.5. It is not a measured same-hardware runtime ratio.
The older lecture's “50 times faster” wording mixed an ensemble comparison
with a wall-clock claim.

Read §6.2 for architecture variations and §6.3 for the English constituency
parsing experiment. They extend the evidence beyond one translation result,
but do not make the two-sentence notebook a benchmark or establish that every
Transformer variant transfers equally well.

## Efficiency and transfer of modifications

For a materialized float32 score tensor, storage is `4 * B * h * T**2` bytes.
This counts one tensor in one layer. It excludes other activations, gradients,
optimizer state, and implementation-specific temporary buffers.

| Source topic | Mechanism to read about | Limit on the claim |
| --- | --- | --- |
| [Reformer](https://arxiv.org/abs/2001.04451), §§2–3 | Locality-sensitive hashing for attention and reversible residual layers | Changes the attention computation and activation-storage strategy; evaluate quality and implementation costs |
| [Linformer](https://arxiv.org/abs/2006.04768), §3 | Low-rank projection along the sequence dimension | An approximation with a chosen projection size; its complexity claim has those assumptions |
| [Do Transformer Modifications Transfer?](https://aclanthology.org/2021.emnlp-main.465/), experiments and analysis | Re-evaluate modifications across implementations and applications | Gains in one setup need independent verification in another |

Training can process known sequence positions in parallel. Autoregressive
generation still depends on preceding generated tokens. Residual connections
and normalization also do not guarantee that gradients remain well behaved.
Later course material treats efficiency and training at larger scales.

## Code reading

[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
is a useful companion implementation. Compare its encoder, decoder,
cross-attention, and normalization conventions with the original paper before
copying a component. Use the course notebook for the exact model counted and
tested in E01–E05. The [teaching plan](teaching-plan.md) maps every source slide
and explains the reductions in classroom scope.
