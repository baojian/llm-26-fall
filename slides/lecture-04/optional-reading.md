# Lecture 04 optional reading

The core lecture follows a fixed-window neural LM, then builds and checks one
causal attention head. These readings extend the same material. They are
ungraded and available to every student.

## The original Spring materials

- [Spring Lecture 04 slides](https://github.com/baojian/llm-26/blob/main/slides/lecture-04-slides/lecture-04-slides.pdf)
- [Spring Lecture 04 notebook](https://github.com/baojian/llm-26/blob/main/lecture-04-neural-lms/lecture-04-neural-lms.ipynb)

The slides contain 84 pages on neural networks and sequence learning.
Pages 4–14 develop nonlinear hidden representations and XOR. Pages 17–24
introduce NPLM and deeper feedforward networks. Pages 25–35 discuss training
and computation graphs. Pages 38–46 and 60–73 develop recurrence, BPTT, and
LSTM gates. The remaining sequence-model examples provide historical context.

The notebook's section 2 implements micrograd components. Section 3 builds a
Bengio-style NPLM, including an optional direct input-to-output connection.
Section 4 trains an LSTM. Those external experiments can require additional
dependencies, tokenizer/data downloads, and more compute than the Fall
classroom notebook. Their code is separate from the offline core exercises.

For a focused extension, trace the derivative through one micrograd operation
or explain how an LSTM forget gate affects its cell-state path. There is no
requirement to rebuild a complete differentiation engine or run an LSTM
training pipeline.

## Neural language modeling

[Bengio et al. (2003), Section 2](https://www.jmlr.org/papers/v3/bengio03a.html)
describes the context embeddings and learned prediction function. Compare its
optional direct connection with our simpler `ContextLM`. The classroom example
uses only a hidden nonlinear path and a vocabulary readout.

In a separate experiment, increase the context size while holding data and
training budget fixed. Construct any train/dev split at the document level
before making windows. Successful fitting of our six examples is a debugging
result, not an estimate of generalization.

## Attention and implementation

[Bahdanau et al. (ICLR 2015), Sections 2–3](https://arxiv.org/abs/1409.0473)
motivate selecting source states in an encoder–decoder system. This is the
historical bridge to our self-attention computation, not the same score
function or architecture.

[Vaswani et al. (2017), Sections 3.2.1 and 3.2.3](https://arxiv.org/abs/1706.03762)
specify scaled dot-product attention and the model's attention masks. Section
3.5 previews position information. Lecture 06 assembles the full Transformer.

[CS336 Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
provides examples of named tensor axes, explicit matrix-product gradients,
memory, and FLOPs. Apply those methods to the head in our notebook before
moving to larger models. [CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics)
shows how component interfaces and correctness tests support an implementation
from scratch. Consult its own course instructions if completing that assignment.

Our notebook's P01 computes dense score storage and product FLOPs. P02 verifies
that reordering paired allowed keys and values leaves one fixed query's output
unchanged. Neither claim says that a complete causally masked network ignores
order, nor that every attention implementation stores the full score matrix.
