# Course papers

Keep one PDF per paper in this shared folder so multiple lectures can link to
the same file. Name files `YYYY-venue-firstauthor-short-title.pdf`, all
lowercase with hyphens, so a listing sorts into a timeline and shows the venue:
for example `2020-neurips-brown-language-models-few-shot-learners-gpt3.pdf` or
`2025-arxiv-qwen-team-qwen3-technical-report.pdf` (use `arxiv` for preprints).
Record the authors, publication, original URL, and license below when adding
a paper. Keep published filenames stable for existing lecture links; PDFs stay
plain Git files, not Git LFS, because GitHub Pages does not serve LFS objects.

## Lecture 01: Tokenization

- Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016.
  *Neural Machine Translation of Rare Words with Subword Units*. ACL.
  [Course PDF](2016-acl-sennrich-neural-machine-translation-rare-words-subword-units.pdf) ·
  [Original publication](https://aclanthology.org/P16-1162/).
  Focus on Section 3.2 for BPE.
- Taku Kudo and John Richardson. 2018.
  *SentencePiece: A simple and language independent subword tokenizer and
  detokenizer for Neural Text Processing*. EMNLP System Demonstrations.
  [Course PDF](2018-emnlp-kudo-sentencepiece-language-independent-subword-tokenizer.pdf) ·
  [Original publication](https://aclanthology.org/D18-2012/).
  Further reading on tokenizer implementation and language independence.
- Beyond these two, the [tokenizer reading list](../docs/tokenizer-reading-list.md)
  links recent papers (2022–2026) on whether the tokenizer changes model
  quality and what text to train it on. It links to the papers; it holds no PDFs.

## Lecture 02: N-gram language models and how LMs are measured

Cited in the [Lecture 02 metrics note](../docs/lecture-02-lm-metrics.md),
which says which figure or table of each paper reports loss, perplexity, or
bits per byte. All ten are the authors' arXiv versions, retrieved on
September 15, 2026, and redistributed under the license shown on each arXiv
abstract page (linked below).

- Tom B. Brown et al. 2020. *Language Models are Few-Shot Learners* (GPT-3). NeurIPS.
  [Course PDF](2020-neurips-brown-language-models-few-shot-learners-gpt3.pdf) · [arXiv 2005.14165](https://arxiv.org/abs/2005.14165).
  Figure 3.1 (validation loss vs. compute), Section 3.1.1 (PTB perplexity).
- Jared Kaplan et al. 2020. *Scaling Laws for Neural Language Models*.
  [Course PDF](2020-arxiv-kaplan-scaling-laws-neural-language-models.pdf) · [arXiv 2001.08361](https://arxiv.org/abs/2001.08361).
  Figure 1, Section 1.3 (loss in nats as the metric).
- Leo Gao et al. 2020. *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*.
  [Course PDF](2020-arxiv-gao-the-pile-800gb-dataset-diverse-text.pdf) · [arXiv 2101.00027](https://arxiv.org/abs/2101.00027).
  Section 3.1 (bits per byte and its conversion formula).
- Jordan Hoffmann et al. 2022. *Training Compute-Optimal Large Language Models* (Chinchilla). NeurIPS.
  [Course PDF](2022-neurips-hoffmann-training-compute-optimal-large-language-models-chinchilla.pdf) · [arXiv 2203.15556](https://arxiv.org/abs/2203.15556).
  Figures 2–3 (training loss vs. FLOPs), Figure 5 (Pile bits per byte).
- DeepSeek-AI. 2024. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*.
  [Course PDF](2024-arxiv-deepseek-deepseek-llm-scaling-open-source-language-models.pdf) · [arXiv 2401.02954](https://arxiv.org/abs/2401.02954).
  Figures 4–6 (bits per byte as the scaling-law metric), Table 4 (Pile-test BPB).
- Llama Team, AI @ Meta. 2024. *The Llama 3 Herd of Models*.
  [Course PDF](2024-arxiv-llama-team-llama-3-herd-of-models.pdf) · [arXiv 2407.21783](https://arxiv.org/abs/2407.21783).
  Figure 2 (IsoFLOP validation loss), Figure 4 (normalized NLL per character vs. accuracy).
- Jeffrey Li et al. 2024. *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models*. NeurIPS.
  [Course PDF](2024-neurips-li-datacomp-lm-next-generation-training-sets.pdf) · [arXiv 2406.11794](https://arxiv.org/abs/2406.11794).
  Section 4.2 and Table 3 (perplexity filtering among the quality filters).
- DeepSeek-AI. 2024. *DeepSeek-V3 Technical Report*.
  [Course PDF](2024-arxiv-deepseek-deepseek-v3-technical-report.pdf) · [arXiv 2412.19437](https://arxiv.org/abs/2412.19437).
  Section 4.4.1 and Table 3 (Pile-test BPB), Section 4.5.1 (validation loss ablations).
- Team OLMo. 2025. *2 OLMo 2 Furious*.
  [Course PDF](2025-arxiv-olmo-team-2-olmo-2-furious.pdf) · [arXiv 2501.00656](https://arxiv.org/abs/2501.00656).
  Figures 2 and 9 (training loss and gradient norm as stability diagnostics).
- Qwen Team. 2025. *Qwen3 Technical Report*.
  [Course PDF](2025-arxiv-qwen-team-qwen3-technical-report.pdf) · [arXiv 2505.09388](https://arxiv.org/abs/2505.09388).
  Downstream benchmark tables only; no loss or perplexity curve.

These PDFs are unmodified copies from the ACL Anthology, retrieved on
September 8, 2026. Both are distributed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/),
as stated by the original publication pages.
