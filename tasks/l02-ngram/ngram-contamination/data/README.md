# Duplicate and overlap teaching data

Version: `lecture02-contamination-v2`. Generator seed: `20261003`.

Original synthetic text, generated offline: **256 training and 64 evaluation
documents**, up to **1,024 tokens** each. IDs are unique; repeated content is
intentional. The corpus includes:

- Exact copies and light/heavier edits, including a duplicate within training.
- Short passages contained in much longer documents.
- Shared boilerplate, including evaluation passages consisting only of that notice.
- Scattered overlap: evaluation trigrams placed in shuffled training windows,
  separated by another token. All evaluation trigrams occur, but the whole passage
  does not appear contiguously.
- Independent text and two evaluation texts too short for trigrams.

## Files and construction

- `examples.json`: three small prediction inputs, without answers.
- `checks.json`: separate public correctness cases with expected outputs.
- `corpus.json`: documents, settings, transformation annotations, groups, and
  exhaustive threshold-based pairs for the supplied metrics.
- `evaluation-audit.json`: three short documents and invented loss/count records
  illustrating a ranking reversal; these are not model measurements.

The initial 40 training documents contain 60 tokens each. Additional backgrounds
cycle through lengths 64, 128, 256, 512, and 1,024 before overlap transformations.
The generator searches at most 1,000 candidates to construct an edited pair
missed by four bands and found by eight. It also constructs a contained passage
that LSH retrieves but exact Jaccard rejects. Search counts and transformations
are recorded in `corpus.json`; `expanded_transformations` names the source used
for each new evaluation construction (independent cases do not copy that source).

These deliberately constructed cases illustrate failures; their retrieval rates
do not estimate production performance. A match may reflect boilerplate or
scattered local overlap, so it needs inspection before any claim of leakage.

## Comparison rules

Keep the corpus, hashes, shingle length, and thresholds fixed when comparing
band counts. At 16 hashes, going from four bands to eight splits each old band
in two, so candidate recall cannot decrease in this comparison. Precision may
move either way. Exhaustive containment is independent of band count.

The public runner reports actual retrieval counts and full/clean-subset audit
scores. It does not delete documents. The private generator is
`tasks/l02-ngram/build_data.py`. V1 contained 40 training and 12 evaluation documents;
its results are retained as historical instructor examples.
