# Lecture 04 optional reading

The core lecture develops a fixed-window neural LM, recurrent memory, and
additive alignment in an RNN encoder–decoder. These readings extend the same material. They are
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

## Additive alignment in recurrent translation

[Bahdanau et al.](https://arxiv.org/abs/1409.0473), §§2–3 and Appendix A.1.2,
develops the fixed-vector bottleneck, bidirectional source annotations, additive
alignment, and a decoder-dependent context. The preprint appeared in September
2014 and the paper at ICLR 2015. The model uses GRUs; the LSTM section teaches
a separate recurrent memory mechanism.

Read §6.1 for the connection to [Graves's 2013 handwriting alignment](https://arxiv.org/abs/1308.0850).
[Mnih et al.'s visual-attention model](https://arxiv.org/abs/1406.6247) also
predates the NMT preprint. Attribute additive soft alignment for NMT to
Bahdanau et al., without claiming the first attention mechanism of every kind.

Explain why target step 2 can read source position 5: the complete source is
already supplied. Future target words remain unknown. Source and target
indices refer to different sequences.

## Recurrent gradients and modern LSTM

[Pascanu et al. (2013), §2](https://proceedings.mlr.press/v28/pascanu13.html)
analyzes recurrent gradient paths. Expand the linear E03 recurrence before
locating the tanh derivatives in its nonlinear counterpart.

Our gate calculation uses the [modern LSTM equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
The forget gate was added after the original 1997 LSTM. Distinguish the direct
cell-state derivative, with gates fixed, from the complete recurrent gradient.

## Continuing in Lecture 05

[Lecture 05](../lecture-05/index.html) develops Q/K/V, scaled dot products,
causal masking, positions, and the Transformer decoder. Its
[notebook](../lecture-05/lecture-05-exercise.ipynb) contains independent
head references and complete-model causality checks. The interactive causal
matrix now belongs to that lecture.

[Vaswani et al. (2017), §3.2](https://arxiv.org/abs/1706.03762) is the primary
reading. Compare the query source, context source, scoring function, and
available positions with Bahdanau's recurrent alignment.
