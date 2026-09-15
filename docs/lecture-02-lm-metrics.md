# Lecture 02 Note: How Good Is a Language Model? The Metrics in Use

**Course:** NLP and LLMs, Fall 2026, Fudan University  
**Part of:** Lecture 02, *N-gram Language Models* (slides 22–26 on evaluation)  
**Papers checked:** September 15, 2026; PDFs in [papers/](../papers/README.md)

Lecture 02 defines perplexity. This note answers the question students ask
next: is that what the people training GPT, Llama, DeepSeek, and Qwen look
at? Yes, in one of four units. Every intrinsic metric below is the same
quantity, the average negative log probability the model assigns to held-out
text, expressed per token, per byte, per character, or exponentiated.

## 1. One quantity, four units

Let a test corpus have $T$ tokens and $B$ UTF-8 bytes under the model's
tokenizer, and let

$$\ell = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\left(w_t \mid w_{1:t-1}\right)$$

be the average negative log probability per token in nats (natural log).
Then:

| Metric | Formula | Unit | Depends on the tokenizer? |
| --- | --- | --- | --- |
| Loss (cross-entropy, negative log-likelihood) | $\ell$ | nats per token | yes |
| Bits per token | $\ell / \ln 2$ | bits per token | yes |
| Perplexity | $\exp(\ell)$ | tokens (effective branching factor) | yes |
| Bits per byte (BPB) | $\dfrac{T}{B}\cdot\dfrac{\ell}{\ln 2}$ | bits per byte | no |
| Bits per character (BPC) | same with characters in place of bytes | bits per character | no, but "character" is ambiguous in Unicode |

The conversion to bits per byte is the one The Pile introduced,
$\mathrm{BPB} = (L_T/L_B)\,\ell/\ln 2$, where $L_T/L_B$ is the model's
tokens-per-byte ratio on that corpus (Gao et al., 2020, Section 3.1; the
GPT-2 tokenizer gives 0.293 tokens per byte on the Pile). It is the unit this
course reports on its held-out shards, because a larger vocabulary lowers
perplexity without the model getting better, while bits per byte stays
comparable across tokenizers and model families.

Two properties worth remembering:

1. **Lower is better for all of them**, and the ordering of models is
   identical under loss, bits per token, and perplexity. Only the byte and
   character units can change an ordering, and only between models with
   different tokenizers.
2. **The metric is computed on held-out text**: a validation split during
   training, a fixed test set for the paper. Training loss is plotted for
   diagnosis (spikes, divergence); the number that is reported is measured
   on text the model never saw.

## 2. What the well-known reports actually plot

The right-hand column names the figure or table in each paper so a student
can open the PDF and find it.

| Metric | Where it appears | Paper, figure or section |
| --- | --- | --- |
| **Cross-entropy validation loss (nats)** vs. training compute | The scaling-law curve everyone reproduces | GPT-3, Figure 3.1: "performance (measured in terms of cross-entropy validation loss) follows a power-law trend with the amount of compute" (Brown et al., 2020) |
| **Test loss (nats)** vs. parameters, data, compute | The original scaling laws; "$L$ – the cross entropy loss in nats" is the principal metric | Kaplan et al., 2020, Figure 1 and Section 1.3 |
| **Training loss** vs. FLOPs; **final pre-training loss** $L(N, D)$ | Compute-optimal model size (Chinchilla) | Hoffmann et al., 2022, Figures 2–3, Section 2; smoothed training loss used as an unbiased estimate of test loss in the single-epoch regime |
| **Validation loss** vs. training tokens, one curve per compute budget | IsoFLOP curves for the 405B model | Llama 3, Figure 2 (Section 3.2.1) |
| **Normalized negative log-likelihood per character** of the correct answer | Bridge from pre-training loss to downstream accuracy | Llama 3, Figure 4: NLL on ARC Challenge vs. FLOPs, then accuracy vs. NLL |
| **Bits per byte on a validation set** as the scaling-law metric | Chosen so that models with different vocabularies compare fairly | DeepSeek LLM, Figures 4–6 (IsoFLOP and performance scaling curves, "the metric is the bits-per-byte on the validation set") |
| **Pile-test bits per byte** in the results table | Reported next to MMLU, BBH, and the rest | DeepSeek-V3, Table 3 (0.548 for V3-Base, 0.542 for Llama 3.1 405B) and Section 4.4.1: BPB "to guarantee fair comparison among models using different tokenizers"; DeepSeek LLM, Table 4 |
| **Validation loss** for architecture ablations | Auxiliary-loss-free load balancing: 2.253 vs. 2.258 on 1B MoE | DeepSeek-V3, Section 4.5.1 |
| **Bits per byte on the Pile subsets**; **WikiText-103 perplexity** | Chinchilla vs. Gopher: BPB decrease per subset; PPL 7.16 vs. 7.75 | Hoffmann et al., 2022, Figure 5 and Table A5 |
| **Zero-shot perplexity on Penn Treebank** | GPT-3's language-modeling result: 20.50 | Brown et al., 2020, Section 3.1.1 |
| **Training loss and gradient norm** vs. steps | Stability diagnosis: loss spikes, divergence, the effect of QK-norm and $\epsilon$ | OLMo 2, Figures 2 and 9, Section 3.4 |
| **Perplexity under a reference model** as a data filter | Not a model metric at all: documents are scored and pruned (Lecture 02, slide 32) | DCLM, Section 4.2 and Table 3 ("Perplexity filtering" after CCNet); Ankner et al., 2024 |
| **No loss curve at all** | Downstream benchmark tables only | Qwen3 technical report: accuracy on MMLU, GPQA, LiveCodeBench, and others; the only "loss" in the report is the MoE load-balancing loss |

Reading the table top to bottom is the history: nats per token for scaling
laws (2020–2022), bits per byte once vocabularies diverged (2020 for
benchmarks, 2024 for scaling laws), and downstream accuracy tables as the
public face of a release (2024 onward), with loss curves kept for the
training team.

## 3. What this means for the course

- The lecture's perplexity slides are correct and current; only the unit on
  the axis has changed. Slide 26 shows the WSJ word perplexities and the
  course's bits-per-byte rows side by side for that reason.
- Every model the class trains reports the same three numbers on the same
  held-out shards: loss in nats per token, perplexity, and bits per byte.
  `pipeline/eval.py` computes them for an n-gram model today and for the
  30M–350M proxies and the 1B model later.
- Perplexity has a second life as a **data filter**: a small reference model
  scores documents and the tail is dropped (CCNet, RedPajama-V2, DCLM). That
  use is stage 2b of the course pipeline and is why the n-gram lecture is
  still worth teaching in 2026.
- Downstream accuracy is not a replacement. Llama 3's Figure 4 is the clearest
  statement of the relation: accuracy is a function of the negative
  log-likelihood of the correct answer, which is a loss on a different text.

## 4. Papers

All PDFs are in [papers/](../papers/README.md).

- Brown et al. 2020. *Language Models are Few-Shot Learners* (GPT-3). [PDF](../papers/2020-neurips-brown-language-models-few-shot-learners-gpt3.pdf) · [arXiv 2005.14165](https://arxiv.org/abs/2005.14165)
- Kaplan et al. 2020. *Scaling Laws for Neural Language Models*. [PDF](../papers/2020-arxiv-kaplan-scaling-laws-neural-language-models.pdf) · [arXiv 2001.08361](https://arxiv.org/abs/2001.08361)
- Gao et al. 2020. *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. [PDF](../papers/2020-arxiv-gao-the-pile-800gb-dataset-diverse-text.pdf) · [arXiv 2101.00027](https://arxiv.org/abs/2101.00027)
- Hoffmann et al. 2022. *Training Compute-Optimal Large Language Models* (Chinchilla). [PDF](../papers/2022-neurips-hoffmann-training-compute-optimal-large-language-models-chinchilla.pdf) · [arXiv 2203.15556](https://arxiv.org/abs/2203.15556)
- DeepSeek-AI 2024. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*. [PDF](../papers/2024-arxiv-deepseek-deepseek-llm-scaling-open-source-language-models.pdf) · [arXiv 2401.02954](https://arxiv.org/abs/2401.02954)
- Llama Team 2024. *The Llama 3 Herd of Models*. [PDF](../papers/2024-arxiv-llama-team-llama-3-herd-of-models.pdf) · [arXiv 2407.21783](https://arxiv.org/abs/2407.21783)
- Li et al. 2024. *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models*. [PDF](../papers/2024-neurips-li-datacomp-lm-next-generation-training-sets.pdf) · [arXiv 2406.11794](https://arxiv.org/abs/2406.11794)
- DeepSeek-AI 2024. *DeepSeek-V3 Technical Report*. [PDF](../papers/2024-arxiv-deepseek-deepseek-v3-technical-report.pdf) · [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- OLMo Team 2025. *2 OLMo 2 Furious*. [PDF](../papers/2025-arxiv-olmo-team-2-olmo-2-furious.pdf) · [arXiv 2501.00656](https://arxiv.org/abs/2501.00656)
- Qwen Team 2025. *Qwen3 Technical Report*. [PDF](../papers/2025-arxiv-qwen-team-qwen3-technical-report.pdf) · [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- Not in the folder: Ankner et al. 2024, *Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models*, [arXiv 2405.20541](https://arxiv.org/abs/2405.20541); Wenzek et al. 2020, *CCNet*, [arXiv 1911.00359](https://arxiv.org/abs/1911.00359).
