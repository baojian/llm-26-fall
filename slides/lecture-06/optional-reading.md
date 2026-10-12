# Lecture 06: Reading and follow-up

Use the [teaching plan](teaching-plan.md) for the classroom route and the
[notebook](../shared/notebook.html?lecture=lecture-06&notebook=practice.ipynb)
for E01–E06. The [asset notes](assets/README.md) identify all measurements and
reproduction commands. Readings have no fixed reference-count requirement.

## Optimizer, clipping, and restart

- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101),
  §2: distinguish an L2 loss penalty from decoupled decay with adaptive updates.
- [PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html):
  algorithm, bias correction, parameter groups, and optimizer state dictionaries.
- [Global gradient clipping](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html):
  the norm is global across parameters; the function returns the norm before clipping.
- [General checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training):
  saving model and optimizer state for continuation. The lecture adds explicit
  data identity, RNG, schedule, and fresh-process comparison checks.
- [AMP clipping and accumulation](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-clipping):
  unscale before clipping when a gradient scaler is used. This is optional;
  the measured lecture experiments use CPU float32.

The notebook displays train/dev loss, learning rate, and gradient norm with
aligned update axes. The prepared [gradient-norm figure data](assets/gradient-norm.json)
comes from the same complete 200-update development run as the slide's loss
curves. The horizontal threshold is a clipping setting, not a claim about the
norm of AdamW's parameter change.

## Decoding

Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751):
read the decoding-method definitions and inspect the paper's evaluation setup.
Nucleus sampling retains a prefix of sorted probabilities, including the token
that reaches the threshold. The class's exact boundary/tie rules follow the
[released A2 handout](../../assignments/a2-build-a-small-lm/README.md).

The beam-search appendix uses only the complete fixed-length two-step tree
shown in class. Search scores and subjective text quality are different
criteria; claims about another task need its own evidence.

## Historical GPT reports

| Reading | Useful question |
| --- | --- |
| [GPT-1, §§3.1–3.2](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | How do pretraining and supervised fine-tuning connect? |
| [GPT-2, §§2–3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | How are tasks represented as language-model contexts? |
| [GPT-3, §2.1](https://arxiv.org/abs/2005.14165) | What changes in zero-, one-, and few-shot in-context evaluation? |

Keep model sizes, data, and reported scores in their original historical
settings. The Spring lecture is a source of teaching material, not a current
leaderboard. Its [public copy](https://github.com/baojian/llm-26/blob/main/slides/lecture-06-slides/lecture-06-slides-gpts.pptx)
differs from the instructor-supplied local snapshot; the exact source hash and
130-page reuse map are in the [teaching plan](teaching-plan.md).

## Later lectures and our training plan

- Lecture 07: corpus sources, filtering, deduplication, and contamination.
- Lecture 08: scaling, compute allocation, and prediction from small runs.
- Lecture 09: evaluation beyond a single loss or sample.
- Lecture 10: fine-tuning, PEFT, and LoRA.
- Lecture 13: KV caching and inference cost.
- [Pretraining baseline/profiling work, #272](https://github.com/baojian/llm-26-fall/issues/272).
- [Validate the scaling ladder, #279](https://github.com/baojian/llm-26-fall/issues/279).

The classroom toy needs no GPU. Larger experiments require a separate measured
compute budget and evaluation plan; the small run does not establish their
quality or resource requirements.
