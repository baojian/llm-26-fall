# Running A1 on a laptop

The supplied data has 10 MB train, 1 MB dev, and 1 MB test for each corpus.
The full experiment trains two 1,000-token BPE tokenizers, unigram through
trigram models on both corpora, and a small neural model on TinyStories only.
It runs on CPU. A GPU is not required for A1.

## Measured reference run

Measured September 16, 2026 on an **Apple M1 laptop with 16 GiB RAM and
8 logical CPUs**, using Python 3.11.15 and PyTorch 2.14.0.
PyTorch used two intra-operation threads and one inter-operation thread; all
models ran on CPU. Neural settings: context 3, embedding/hidden dimensions 64,
batch size 256, three epochs of Adam; development/test scoring was batched.

| Stage | TinyStories English | Chinese web |
| --- | ---: | ---: |
| Train BPE | 3 s | 4 min 06 s |
| Encode train/dev/test text, including round-trip checks | 36 s | 1 min 10 s |
| Fit, tune, and evaluate count models | 1 min 05 s | 1 min 26 s |
| Neural training and evaluation | 3 min 30 s | Not required |
| Entire corpus experiment | 5 min 15 s | 6 min 43 s |

**Whole run: 12 min 00 s; peak process resident memory: 1.03 GiB.**
The process used 12.0 CPU-minutes in total, averaging
1.00 CPU cores over the run (100% CPU
in a process monitor commonly means one full core). Python tokenization and
count models mostly use one core; the neural part can use two.

An 8 GB RAM laptop should have sufficient room for this reference workload;
16 GB is comfortable with other applications open. Allow at least an hour for
an initial full run, and more on a slower laptop or with a less efficient
implementation. Setup, debugging, and report writing are separate from this
experiment time. Peak resident memory is a process measurement, not the
machine's total RAM requirement.

These measurements describe one run of the completed reference implementation,
with other applications running. They are a planning guide, not a runtime limit
or a grading criterion. Student implementations and laptop hardware vary.
Debug with the small public tests before running the full experiment.

## Why Chinese BPE training takes longer here

The pre-tokenizer splits text into chunks before BPE learns its merges. English
spaces usually separate short words: `" a little cat"` becomes three chunks.
A Chinese phrase such as `"一只小猫"` stays in one chunk until a space,
punctuation mark, or another character category ends it. BPE does not merge
across these boundaries.

On the actual 10 MB training samples, before any BPE merges:

| Measurement | TinyStories English | Chinese web |
| --- | ---: | ---: |
| Distinct chunks | 10,169 | 266,064 |
| Mean distinct-chunk length, UTF-8 bytes | 6.84 | 32.00 |
| Adjacent byte positions summed over distinct chunks | 59,393 | 8,247,666 |

The original trainer counted repeated chunks once, with a frequency weight,
but scanned all distinct chunks again for each merge. Chinese has about 26 times
as many distinct chunks and 139 times as many initial byte-pair positions to
scan. Those positions shrink as merges proceed, so the initial ratio is not
a prediction of the final time ratio.

This comparison also mixes languages with different data sources. TinyStories
deliberately repeats a small vocabulary, while Chinese web documents cover
more varied subjects. Both segmentation and corpus diversity matter.

Most Chinese characters occupy three UTF-8 bytes, but both training files
already contain approximately the same number of bytes. That fact alone does
not explain the runtime gap. The original costly full scans were specific to that
implementation; the observation does not mean all Chinese tokenizers are
intrinsically slower. The current trainer keeps an index of affected chunks
and updates only their counts after each merge. Chinese still has a much
larger index and more affected text for common early merges. Learning BPE merges and encoding text with an already
trained tokenizer are separate operations with different costs.

## Practical settings

- Use the supplied fixed train/dev/test files. Do not shrink or resplit them
  for the final reported experiment.
- Limit PyTorch to two CPU threads to leave the laptop responsive. On macOS
  or Linux: `OMP_NUM_THREADS=2 uv run python a1/evaluate.py data/`.
  On PowerShell, first run `$env:OMP_NUM_THREADS="2"`, then
  `uv run python a1/evaluate.py data/`.
- Evaluate neural development and test examples in minibatches. Creating all
  vocabulary logits for a whole split at once can use much more memory than
  the text files themselves.
- The tokenizer and count-model code is ordinary Python and mainly uses one
  CPU core. Adding GPU access does not accelerate these functions as written.
- When choosing the report's vocabulary-size experiment, run it on TinyStories
  first. Larger vocabularies require more BPE merges; a 4,000-token Chinese run
  can take substantially longer. Other report choices have the same points.
