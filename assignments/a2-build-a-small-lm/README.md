# A2: Build and investigate a small language model

**Release version: 2026-10-10.**

Build a causal decoder, establish that its training is correct, and investigate
how architecture, sampling, and data affect the same small language model.

- Release: October 10, 2026.
- Due: **October 31, 2026, 23:59 Asia/Shanghai (UTC+08:00)**.
- Weight: 15% of the course grade; 100 assignment points.
- Submit one ZIP privately through eLearning. All students have identical work,
  criteria, and maximum scores. There are no bonus or degree-specific parts.
- AI assistance is allowed with disclosure. You are responsible for correctness
  and interpretation; no chat transcript is required.

## 1. Setup and supplied files

Copy this assignment folder into the course repository's ignored `workspace/`
directory or a separate private local copy before editing. Keep solutions and
results out of public commits; submit them only through eLearning.

From your copied assignment directory, use the supplied standalone environment:

```sh
uv sync --locked
uv run python -m pytest -q
```

The initial tests should fail with `NotImplementedError`. Implement the marked
functions in `a2/model.py`, `a2/train.py`, `a2/generate.py`, and `a2/data_policy.py`.
Keep names and signatures unchanged. Constructors, initialization, CLI parsing,
tokenizer, corpus, sampler, evaluation, plotting, and ZIP packaging are supplied.
You may add private helpers inside the four submitted modules. They must import
using only the standard library and PyTorch; do not depend on `support.py` or
your local output files from within submitted modules.

Use PyTorch tensor operations, `Linear`, `Embedding`, `LayerNorm`, cross-entropy,
autograd, and AdamW. Do not substitute `nn.MultiheadAttention`, `nn.Transformer`,
or fused attention for the attention functions you implement. You may use them
in your own independent checks. Do not rebuild a tokenizer or MinHash/LSH.

Run all commands here. `outputs/` contains local checkpoints, curves, and logs;
keep it out of your submission. The bundled data requires no network access.
See [data provenance](data/README.md), [runtime notes](runtime.md), and the exact
[submission contract](manifest.toml).

## 2. Part 1 — Can the decoder use only its allowed context? (40 points)

Implement the four forward paths in `a2/model.py`:

| Function | Input → output | Contract |
| --- | --- | --- |
| `scaled_dot_product_attention` | Q `(...,Q,D)`, K `(...,K,D)`, V `(...,K,Dv)` → `(...,Q,Dv)` | Scale by `sqrt(D)`; Boolean True permits a key; mask before softmax over keys; broadcast mask; every row must permit a key |
| `CausalSelfAttention.forward` | `(B,T,width)` → same | Independent Q/K/V projections; split heads preserving token order; permit the diagonal; join heads and apply output projection |
| `DecoderBlock.forward` | `(B,T,width)` → same | Two residual sublayers; feature-wise LayerNorm; position-wise ReLU FFN; support the supplied pre/post norm switch |
| `TinyLM.forward` | integer IDs `(B,T)` → raw logits `(B,T,V)` | Token plus learned position embeddings, blocks, final LayerNorm, tied readout; require `1 <= T <= context` |

The supplied constructors use bias-free linears, affine LayerNorm with epsilon
`1e-5`, no embedding scaling, and no dropout. They initialize shared matrix
parameters once from a normal distribution with standard deviation `0.02`.

For pre-norm, each sublayer applies normalization before its transformation,
then adds the residual. For post-norm, normalize after the residual addition.
Keep the final LayerNorm for both models. This compares block normalization
placement; it is not a reproduction of the full 2017 Transformer.

Show a small numerical attention check. Run the supplied diagnostics:

```sh
uv run python run.py diagnostics
```

Explain why changing only future tokens preserves earlier logits, and why a
logit has zero gradient with respect to future input states. The supplied
unmasked control must expose leakage. Matching tensor shapes alone is not a
correctness argument. Because targets are shifted, position `t` may see input
position `t` when predicting token `t+1`.

## 3. Part 2 — What evidence shows training works? (30 points)

Complete `a2/train.py`:

- `make_batch`: a `(B,T+1)` window produces inputs without the last token and
  targets without the first; return contiguous tensors.
- `train_step`: training mode, clear old gradients, raw-logit mean cross-entropy,
  backward, global gradient clipping, and exactly one optimizer update. Return
  `loss` and `grad_norm` as floats; the latter is measured **before** clipping.
- `learning_rate`: use the documented zero-based schedule. For update `s < w`,
  return `max_lr*(s+1)/w`. Otherwise cosine progress is `(s-w+1)/(N-w)`, ending
  at `min_ratio*max_lr` on update `N-1`. Thus `w-1` is the warmup maximum.
- `save_checkpoint` / `load_checkpoint`: preserve model, optimizer, completed
  updates (also the schedule position), Python/torch/sampler RNG states, config,
  and active accelerator RNG when applicable. Reject incompatible config,
  device type, or version before loading. Use `torch.load(..., weights_only=True)`.

The common path runs on CPU. All required runs use float32 and two threads.
The tiny fitting fixture contains two equally frequent sequences whose first
target differs after the same BOS. Its attainable mean loss floor is
`ln(2)/4`, not zero. Interpret your fitting curve against that floor.

Check a small end-to-end run before the required budget:

```sh
uv run python run.py suite --config configs/smoke.json --out outputs/smoke
uv run python run.py suite --out outputs/required
uv run python plot_results.py --out outputs/required
```

The required suite runs **baseline pre-norm, post-norm, and deduplicated data**,
512 updates each, batch 16, context 128: **1,048,576 training tokens per run**.
The model has two blocks, width 128, four heads, FFN width 512, vocabulary 4,096,
and 935,168 trainable parameters with the supplied tying convention. The same
seed, initial parameter values, optimizer, schedule, and tokenizer are used.
Baseline and post-norm use exactly the same sampled windows.

Record curves, learning rates, unclipped gradient norms, actual tokens processed,
wall time, and the memory measurement method. The supplied meter reports sampled
process RSS, which may miss peaks between samples; it is not GPU memory.
The driver writes config, environment, hashes, and artifact paths to each run.

The suite also compares **100 continuous CPU updates** with **50 + save + a
fresh process + load + 50**. Losses and final parameters should agree within
absolute tolerance `1e-7` in the pinned CPU environment. The supplied weights-only
control retains future batches but loses AdamW state and should disagree.
Bitwise agreement across different devices or library versions is not required.

To exercise interruption and restart separately:

```sh
uv run python run.py train --condition baseline --out outputs/restart-demo --stop-after 256
uv run python run.py train --condition baseline --out outputs/restart-demo --resume
```

Use a new output directory for independent runs. Existing checkpoints are not
silently overwritten. Reuse the full-budget baseline for Parts 3 and 4; a smoke
run does not replace a required run. No learning-rate sweep or extra seed is
required. Neither a particular winner nor post-norm divergence earns points.

## 4. Part 3 — How does decoding change the same model? (10 points)

Implement `sampling_probs`, `choose_token`, and the marked generation loop.
Temperature must be positive; greedy is a separate mode and chooses the
smallest token ID among tied maxima. For top-p, sort by decreasing probability
with ties broken by smaller token ID. Keep the smallest prefix reaching `p`,
**including the crossing token**, then renormalize. Require `0 < p <= 1`; at
`p=1`, retain every token. Sample with the supplied CPU `torch.Generator`.

The supplied context manager crops to the last `context` tokens and restarts
learned position IDs from zero. Return new IDs only, including a generated EOS,
and `stop='eos'` or `'length'`. A zero generation budget produces no new tokens.
Restore the model's original train/eval mode.

After training, finalization produces nine generations from one baseline
checkpoint: three fixed prompts, each with greedy, `(temperature=0.7, p=0.9)`,
and `(temperature=1.0, p=1.0)`, seed 29, at most 64 new tokens. Explain their
coherence, repetition, and stopping reason. The reported repeated-bigram
fraction is `1 - unique_bigrams / total_bigrams` (zero with no bigrams).
Fluency and a specific random sample are not correctness thresholds.

## 5. Part 4 — Can a data policy help at a fixed budget? (20 points)

Implement deterministic exact-document deduplication and its audit:

1. Normalize to Unicode NFC, convert CRLF and CR to LF, and strip outer
   whitespace. Preserve case and internal whitespace/punctuation.
2. Keep the original record with the **lexicographically smallest ID** per group.
   Sort retained records by ID and return every removed-ID → retained-ID mapping.
3. Reject repeated IDs, invalid fields, and blank normalized text with `ValueError`.
   Empty input returns `([], {})`. Do not mutate the input.

The baseline deliberately repeats 256 of the 2,048 distinct training documents
four times each, giving 2,816 records. This is a teaching construction, not a
measurement of repetition in natural web text. Related but different documents
must remain distinct. Read [data/README.md](data/README.md) for split lineage.

Both conditions use the same frozen tokenizer and evaluate the same text.
Training concatenates each ID-sorted document as `[BOS] text [EOS]`. Full windows
may cross document boundaries; there is no padding or boundary attention reset.
Sample starts uniformly with replacement. Evaluation scores each next token
once in disjoint context blocks, including a short final block; loss is weighted
by scored tokens. Both protocols and all comparisons use the same conventions.

Match **tokens processed**, not epochs: removing duplicates changes the number
of passes over the retained corpus. Report corpus tokens and the ratio of
processed tokens to corpus tokens. Reinitialize before the deduplicated run.
Inspect at least three retained and three removed examples, including a reason
why repetition could sometimes be useful. Correct negative or inconclusive
results can receive full credit.

Make decisions using development results. When your code and comparison are
fixed, evaluate the baseline and deduplicated checkpoints on the final split:

```sh
uv run python run.py finalize --out outputs/required --results results.json
uv run python validate_results.py results.json
```

Finalization validates config, initial weights, budgets, data/tokenizer hashes,
and checkpoint identities. It records the final evaluation and reuses that
record on later calls; it rejects changed checkpoints. Do not use final scores
to choose another model, seed, data policy, or stopping point.

## 6. Evidence, grading, and submission

Complete [report.md](report.md), targeting 600–900 words. Use figures locally to
inspect your work; the submitted `results.json` contains the numeric curves.
Do not reference plots or raw logs as the only evidence: they are not in the ZIP.
[results-schema.json](results-schema.json) specifies the machine-readable record;
[the synthetic example](examples/results-example.json) illustrates structure and
must not be presented as a measured run. Structural validation is not scientific
validation. Small independent fixtures let Parts 2–4 be tested without a working
decoder; clearly identify any supplied comparison evidence in your report.
The [supplied development traces](examples/comparison-traces.json) support
interpretation and partial credit if your model is not yet working. Label them
as instructor evidence; they do not replace your own execution or final results.
If you cannot produce a required evidence section, use
[the partial-results format](examples/results-partial.json): set
`submission_status` to `partial`, mark unavailable sections `null`, and give a
specific reason for each in `unavailable_evidence`. Submit your working code
and report normally. Never invent measurements to satisfy the packager. Missing
evidence affects its own criteria; it does not prevent grading independent parts.

| Part | Automatic | TA interpretation | Total |
| --- | ---: | ---: | ---: |
| Architecture | 30 | 10 | 40 |
| Training and diagnosis | 18 | 12 | 30 |
| Decoding | 6 | 4 | 10 |
| Data experiment | 6 | 14 | 20 |

Automatic tests use small CPU fixtures; they do not retrain the full assignment.
TA points assess your checks, experimental controls, evidence, interpretation,
limitations, and disclosure. Hardware speed and additional training carry no bonus.

```sh
uv run python make_submission.py YOUR_STUDENT_ID
```

Upload `a2-YOUR_STUDENT_ID.zip` privately to eLearning. It contains exactly:

```text
a2/__init__.py
a2/model.py
a2/train.py
a2/generate.py
a2/data_policy.py
results.json
report.md
```

Keep data, test files, environments, checkpoints, plots, and large logs out of
the ZIP. The staff harness supplies tests and dependencies. Your modules must
work after extraction without your original checkout. Keep the original ZIP
filename. Missing disclosure incurs one five-point deduction after ordinary
grading, clamped to 0–100; AI use itself is not penalized.

### Late submissions

The penalty multiplies your earned assignment score after ordinary grading,
including any disclosure deduction. Count each started 24-hour period after
October 31, 2026, 23:59 (Asia/Shanghai) as one late day, using the eLearning
submission timestamp.

| Late days | Final score |
| --- | ---: |
| 0 | 100% of the earned score |
| 1 | 90% |
| 2 | 80% |
| 3 | 70% |
| 4 | 60% |
| 5 | 50% |
| 6 or more | 0 |

## Sources

The exercises adapt selected topics from [CS336 A1](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf)
and [CS336 A4](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/cs336_assignment4_data.pdf).
The model follows our Lecture 05 conventions; the budgets, fixtures, restart
check, and exact-document policy are course-specific. TinyStories is attributed
and licensed separately in `data/`.
