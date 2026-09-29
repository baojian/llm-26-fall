# Course pretraining plan: a model between GPT-2 and Marin

Draft, September 11, 2026. A plan for the model the class trains during the
semester: what size, how many tokens, which data, on what schedule, and what
must be verified first. Numbers marked *measured* come from tests run on the
course cluster on September 11; numbers marked *assumed* or *estimated* are
stated assumptions to be replaced by measurements.

## 1. Target in one sentence

Train a **1B-parameter decoder** on **about 35B tokens** of English, Chinese,
code, and math with the Qwen3 tokenizer, plus a **ladder of 30M / 125M /
350M proxies** on the same data, so that every lecture from Week 6 on has a
real run to look at and every student ablation is cheap.

Why this point: GPT-2 XL is 1.5B parameters on about 10B tokens (2019);
Marin 8B is 8B parameters on about 12T tokens on TPU pods; Qwen3-0.6B/1.7B
are trained on 36T tokens. A 1B model on 35B tokens is a few times past the
compute-optimal 20 tokens/parameter, takes about a day on 32 cards of the
course cluster, and is directly comparable to public baselines of the same
size (Qwen3-0.6B-Base, SmolLM2-360M/1.7B, GPT-2 XL). Anything Qwen3-scale in
data (trillions of tokens) is out of reach by two orders of magnitude and
would not teach more.

## 2. Compute budget

Course cluster (shared; K8s "distributed training" jobs have no per-job cap):

| Pool | Cards | Memory per card | Single-card results |
| --- | ---: | ---: | --- |
| Alibaba PPU (ZW810 class) | 56 (7 nodes × 8) | 96 GB | *measured*: 397 TFLOP/s bf16 and 107 fp32 on an 8192² matmul; torch 2.6 with a CUDA-12.6-compatible SDK. End-to-end training not yet measured |
| MetaX C550 | 120 (15 nodes × 8) | 64 GB | *measured*: 274 TFLOP/s bf16 and 36 fp32 on an 8192² matmul; causal SDPA 106 TFLOP/s with the flash path enabled; vendor torch 2.1.2 with flash_attn 2.6.3 and apex. End-to-end training measured below |

Training throughput, *measured* on one MetaX C550: a GPT with SDPA attention,
bf16 autocast, AdamW, sequence length 2048, tied 152K-entry Qwen vocabulary,
no `torch.compile`:

| Config | Params incl. embeddings | Tokens/s per card | Achieved TFLOP/s | Share of bf16 peak | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| 125M body | 203M | 66,157 | 96 | 35% | 18 GB |
| 350M body | 460M | 32,523 | 109 | 40% | 26 GB |
| 1B | 1,121M | 13,902 | 105 | 38% | 27 GB |
| 1.5B | 1,524M | 10,266 | 106 | 39% | 35 GB |

Wall-clock for the planned runs, from the measured tokens/s and assuming
linear scaling across cards (multi-card scaling is not yet measured):

| Run | Tokens | Card-hours | Hours on 8 cards | Hours on 32 cards | Hours on 120 cards |
| --- | ---: | ---: | ---: | ---: | ---: |
| 125M proxy | 2.5B | 10 | 1.3 | 0.3 | — |
| 350M proxy | 7B | 60 | 7.5 | 1.9 | — |
| **1B flagship** | **35B** | **700** | **87** | **22** | **5.8** |
| 1.5B stretch | 50B | 1,350 | 169 | 42 | 11 |

The PPU pool should be about 1.4× faster per card if it reaches the same
utilization, since its bf16 matmul peak is 397 against 274 TFLOP/s
(*estimated*, to confirm with the same script). The 30M proxy was not
measured; it is a few minutes on one card.

An earlier version of this table used the 6·N·D rule without the output layer
and a 35% utilization assumption, and gave 49 hours on 8 cards for the
flagship. With a 152K vocabulary the output layer adds about a third of the
compute, so that figure was too low; the measured 87 hours replaces it.

## 3. Data: about 35B tokens, five sources

All sources are public, downloadable through the Hugging Face mirror that the
cluster can reach, and already filtered by their publishers; we add our own
light pass (Section 4) so students see the pipeline, not to improve on it.

| Domain | Source | Tokens used | Why this source |
| --- | --- | ---: | --- |
| English web | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-100BT`) | 14B | classifier-filtered educational web text; the current default for small models (SmolLM) |
| Chinese web | CCI3-HQ (`BAAI/CCI3-HQ`) and FineWeb-2 `zho_Hans` | 10B (7B + 3B) | the two largest openly licensed high-quality Chinese web corpora; two sources so students can compare them |
| Code | The Stack v2 smol (`bigcode/the-stack-v2-train-smol-ids`) or StarCoderData, Python-heavy | 4B | small, deduplicated, permissively licensed |
| Math | FineMath 4+ (`HuggingFaceTB/finemath`) and OpenWebMath | 4B (2B + 2B) | the two math corpora whose extraction we study in the regex tasks |
| Reference | Wikipedia EN and ZH (`wikimedia/wikipedia`), Cosmopedia v2 (synthetic textbooks) | 3B (2B + 1B) | clean, encyclopedic; Cosmopedia is the only synthetic source and is labelled as such |

Mixture: English 40%, Chinese 29%, code 11%, math 11%, reference 9%. The
Chinese share is deliberately high for a Fudan class and is the main lever
students can vary in the Week 7 data-policy ablation.

Sizes: 35B tokens is about 140 GB of text and 140 GB tokenized as uint32
(uint16 cannot hold the 151,669-entry Qwen vocabulary), well inside the 3 TB
project store. Throughput on the preprocessing host was *measured* at 27 MB/s
for tiktoken encoding across 32 processes, so tokenizing the whole corpus is
under two hours; regex filters run at similar speed; MinHash deduplication is
the slow step and should run once on the CPU pool with datatrove.

Tokenizer: reuse the **Qwen3 tokenizer** for the main runs, so perplexities
and token counts are comparable with Qwen3-0.6B-Base and the vocabulary
covers Chinese well. Students still train their own BPE in Lecture 01–02
tasks; a 32K in-house BPE is an optional ablation, not the main line. A
smaller vocabulary would also cut the output-layer cost noted in Section 2.

## 4. Pipeline (what students build, in task form)

1. **Download and inspect** each source through the mirror; record license,
   size, and a 20-document sample (Week 3–4 tasks).
2. **Filter**: the C4/Gopher line and document rules, the language ID check,
   and the PII regexes (Weeks 2–4 regex tasks are exactly these functions).
3. **Deduplicate**: exact hashes per document, then MinHash near-duplicates
   within each source (Week 7, data lecture).
4. **Decontaminate**: 10-gram overlap against every benchmark we will report
   (DeepSeekMath's rule), before tokenization.
5. **Tokenize and pack** into fixed 2048-token sequences with document
   boundaries marked; write shards with a manifest of source, tokens, and
   the filter versions used.
6. **Mix** by sampling shards according to the table above; one config file
   per policy so ablations differ only in that file.

Every stage writes a small report (documents in, documents out, top removal
reasons, three examples of each) that goes on the course site, the same way
the survey chart does.

## 5. Schedule aligned with the lectures

Lecture dates follow the [revised course schedule](schedule.md). Cluster work
remains a preparation target, subject to the checks in Section 7.

| Week | Lecture | Cluster work | Students |
| --- | --- | --- | --- |
| 2–4 | N-grams, embeddings, attention | finish the measurements in Section 6; download sources; tokenizer counts | regex and tokenizer tasks |
| 5 | Transformer (October 10 make-up class) | dedup, decontaminate, tokenize; train the 30M and 125M proxies on the full mixture | shape checks on the decoder |
| 6 | Pretraining and decoding | **launch the 1B flagship** (about one day on 32 cards); publish the training curve live | resumed-run and failure-diagnosis exercises on the 125M |
| 7 | Data preparation | 125M ablations: two data policies (Chinese share, with/without Cosmopedia) at equal token budget | compare the two policies |
| 8 | Compute and scaling | fit the 30M–350M ladder; predict the 1B loss, then compare with the real run; dense versus MoE at equal data (Section 6) | resource estimate versus measured value |
| 9 | Evaluation | evaluate the 1B against Qwen3-0.6B-Base, SmolLM2, GPT-2 XL; contamination audit | audit one benchmark item |
| 10–11 | SFT, preferences | SFT and a DPO pass on the 1B with open instruction data | before/after on one prompt set |
| 12–16 | RAG, inference, diffusion, agents, synthesis | serve the model (vllm is installed on the PPU image); latency tables | projects use the checkpoints |

## 6. MoE option: measured cost

The minimal canonical mixture-of-experts follows OLMoE (1.3B active, 6.9B
total, 64 experts, top-8) scaled down: 16 layers, width 1024, 64 SwiGLU
experts of width 512, top-8 dropless token-choice routing, no shared expert,
load-balancing loss 0.01 and router z-loss 0.001. It has 1.83B total and
0.43B active parameters and needs 1.0e20 FLOPs for 35B tokens, 2.6× fewer
than the dense 1B. Sources: OLMoE Sections 2 and 4.1 and Table 1; the
Qwen3-30B-A3B config (128 experts, top-8, no shared experts); DeepSeek-V2-Lite
config (64 routed + 2 shared experts, top-6).

FLOPs do not translate into time without a grouped-matmul kernel. One MoE
layer, 16,384 tokens, forward and backward, bf16, *measured* on one MetaX
C550 in plain PyTorch; all shapes have the same total and active expert
parameters:

| Layer | Loop over experts | Padded batched matmul |
| --- | ---: | ---: |
| dense SwiGLU, width 4096 (reference) | 169 TFLOP/s | — |
| 64 experts, top-8, width 512 | 9 TFLOP/s | 40 TFLOP/s |
| 16 experts, top-2, width 2048 | 39 TFLOP/s | 88 TFLOP/s |
| 8 experts, top-1, width 4096 | 64 TFLOP/s | 102 TFLOP/s |

*Estimated* end to end, scaling these layer rates by the measured dense
training efficiency: the 64 × top-8 model runs at about 15,000 tokens/s per
card, no faster than the dense 1B, so it saves nothing in wall-clock; the
16 × top-2 shape at the same size runs at about 26,000 tokens/s, 35B tokens in
about 12 hours on 32 cards.

Decision proposed: keep the dense 1B as the flagship. Train the MoE on the
same data as the Week 8 comparison, in the 16 × top-2 shape with padded
dispatch; switch to the canonical 64 × top-8 only if a grouped-matmul kernel
is available on the vendor stack.

## 7. Verify before committing

1. **End-to-end throughput on one card.** Done on the MetaX C550 (Section 2).
   Still to run on a PPU with the same script, once its session is up.
2. **Multi-card scaling**: the 350M config on 8 cards (one node) and on 16
   cards (two nodes) with PyTorch DDP/FSDP through each vendor's collective
   library. If cross-node scaling is poor, the flagship shrinks to what one
   node can do in about two days (1B × 20B tokens on 8 MetaX cards at the
   measured rate), which is still twice GPT-2 XL's token count.
3. **Which pool for what.** Both pools run the same PyTorch code on single
   cards. Proposed: the flagship on whichever pool scales better; proxies,
   ablations, and student jobs on the other.

Also to settle: a checkpoint policy (every 1B tokens, kept on the project
store, with the exact data manifest and commit hash), and data-loader CPU
needs (the MetaX test container had only 4 CPU cores).

## 8. Risks and the fallback

- **Vendor accelerators, not NVIDIA.** Two different stacks: the PPU image
  has torch 2.6; the MetaX image has torch 2.1.2 with vendor builds of
  flash_attn, apex, and triton. Code must run on both without `torch.compile`
  and without NVIDIA-only kernels. Marin's Levanter stack is JAX on TPU and is
  not an option here; the code base is PyTorch (nanoGPT/CS336 style).
- **Network differs by pool.** The PPU image cannot reach GitHub; the MetaX
  image can. Datasets come through the Hugging Face mirror and packages through
  the Aliyun PyPI mirror on both.
- **Session storage is ephemeral.** Everything lives under the project store;
  jobs are written to resume from the last checkpoint.
- **Licensing.** Every source above is redistributable for research; Chinese
  web corpora should be checked once more against their cards before use.
- **Fallback.** If scaling or stability fails, the 350M proxy on 7B tokens
  becomes the class model. It still supports every lecture from Week 6 on.
