# NanoGPT training in 39.9 seconds

Recorded for the course on **September 30, 2026**.

Deven Pietrzak of Hyperstition reported training a language model to the
NanoGPT speedrun's target quality in **39.914 seconds on eight NVIDIA H100
GPUs**. The [submission, PR #360][submission], was merged on September 28,
2026 (UTC). This is a useful classroom example of how algorithms and GPU
engineering can reduce the time needed to reach a fixed language-modeling
quality target.

The instructor's original link was the [X trending story][original], titled
“Deven Pietrzak Sets NanoGPT Speedrun Record at 39.9 Seconds,” last updated
September 29, 2026. The measurements below come from the primary sources.

## What was measured

The [record documentation][record] and [submitted statistics][statistics]
report these results:

| Item | Recorded value |
| --- | --- |
| Timed training | **39.914 seconds**, mean of 18 runs; sample standard deviation 0.120 seconds |
| Hardware | One node with **8 × NVIDIA H100 GPUs, 80 GB each** |
| Data | FineWeb10B training data and the benchmark's fixed FineWeb validation tokens |
| Quality target | Mean validation cross-entropy **≤ 3.28 nats per token**; lower is better |
| Achieved quality | Mean **3.27731**, sample standard deviation 0.00099 |
| Training schedule | **1,194 training steps** |
| Model size | A modified GPT-2-scale transformer described as 124M-class, plus sparse embeddings; **approximately 65.3 billion total parameters**, over 99% in the hashed n-gram table |

The public statistics include all 18 runs. Seventeen raw run logs remain
available; the authors disclose that one log was lost when a rented instance
expired, and retain its recorded metrics. This course note checks the published
evidence; we have not rerun the GPU experiment.

## How much faster

Different comparisons answer different questions:

| Comparison | Earlier time | Result and interpretation |
| --- | --- | --- |
| Author's controlled comparison with record 89 | **73.889 s**, mean of nine runs interleaved on the same machine | 39.914 s saves **33.975 s**, a **46.0%** reduction or **1.85×** speedup. [Source][submission] |
| Immediately preceding leaderboard entry, record 91 | **1.126 min**, about **67.6 s** | The new entry is 0.665 min, about 39.9 s: **about 41%** less time, calculated from the rounded leaderboard values. [Source][history] |
| Original leaderboard baseline from May 28, 2024 | **45 min** | About **68×** the new training time. This historical comparison spans changes in architecture, training recipes, software, and timing rules. [Source][history] |

The 46% figure specifically uses the author's record-89 baseline. Always name
the baseline when presenting a speedup.

## Why it became faster

The author's [technical explanation][blog] describes three main changes:

- **ANVIL II and weight averaging:** improve optimization so the model reaches
  the required loss sooner.
- **Sampled softmax:** score a subset of vocabulary entries during much of
  training, then expand to the full vocabulary. Validation uses the full
  vocabulary throughout.
- **Larger sparse bigram and trigram embeddings:** divide the table across
  GPUs and update only the rows used by a batch. This makes much more capacity
  practical for learning local token patterns.

FP8 arithmetic, smaller attention components, and CUDA graphs also reduce
the work and overhead of training. Together, these changes improve both
the cost of a step and the number of steps needed to reach the target.

## How to present the result in class

**The 39.9 seconds measures a defined training section.** Compilation,
kernel warmup, graph capture, and validation are outside the clock. Training
includes data loading within the run, final weight averaging, and
synchronization. Setup, data download, and the research needed to discover
the recipe add time beyond this measurement. See the [timing convention][timing].

**The quality target and parameter structure matter.** The benchmark measures
next-token prediction on a fixed validation set. Its large embedding table
uses sparse lookups and updates; the total parameter count therefore does
not imply the compute of a dense 65B transformer. The 124M-class description
refers to the transformer component. General chat capability and training
time at frontier scale require separate measurements.

Discussion prompt: when someone says a language model can be trained in
40 seconds, what hardware, quality target, model structure, and timing
boundaries would you ask them to specify?

## Source dates and permanent references

- [Original X story][original]: updated September 29, 2026.
- [Hyperstition article by Deven Pietrzak][blog]: dated September 27, 2026.
- [Upstream PR #360][submission]: opened August 31; merged September 28,
  2026 at 21:19:54 UTC (**September 29 at 05:19:54 Asia/Shanghai**).
- GitHub documentation links in this note are pinned to the
  [merge commit][commit], preserving the record as accepted. The record
  directory is dated August 30 and its reported run pool August 31; these
  dates precede the article, merge, and X coverage.

[original]: https://x.com/i/trending/2104757152657166545
[blog]: https://hyperstition.cc/training-nanogpt-in-39-9-seconds
[submission]: https://github.com/KellerJordan/modded-nanogpt/pull/360
[commit]: https://github.com/KellerJordan/modded-nanogpt/commit/f9b6266f0b0aa3f4316b86394bce9304c84afdf8
[record]: https://github.com/KellerJordan/modded-nanogpt/blob/f9b6266f0b0aa3f4316b86394bce9304c84afdf8/records/track_1_short/2026-08-30_ANVIL2/README.md
[statistics]: https://github.com/KellerJordan/modded-nanogpt/blob/f9b6266f0b0aa3f4316b86394bce9304c84afdf8/records/track_1_short/2026-08-30_ANVIL2/this_pr/statistics.md
[history]: https://github.com/KellerJordan/modded-nanogpt/blob/f9b6266f0b0aa3f4316b86394bce9304c84afdf8/README.md#world-record-history
[timing]: https://github.com/KellerJordan/modded-nanogpt/blob/f9b6266f0b0aa3f4316b86394bce9304c84afdf8/records/track_1_short/2026-08-30_ANVIL2/README.md#timing-convention
