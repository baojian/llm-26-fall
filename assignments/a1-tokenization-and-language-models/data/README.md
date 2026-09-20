# A1 tokenization and language models dataset v1

Fixed laptop-scale samples for every undergraduate and master's student. No bonus or extension dataset is required.

| File | UTF-8 bytes | Documents |
| --- | ---: | ---: |
| `tinystories-train.txt` | 10,000,399 | 12,352 |
| `tinystories-dev.txt` | 1,000,253 | 1,235 |
| `tinystories-test.txt` | 1,000,561 | 1,276 |
| `chinese-web-train.txt` | 10,000,520 | 2,307 |
| `chinese-web-dev.txt` | 1,001,784 | 229 |
| `chinese-web-test.txt` | 1,000,171 | 232 |

One MB is 1,000,000 bytes. Complete documents are retained, so file sizes slightly exceed the requested target. Each document is at most 8,192 bytes; internal empty lines are removed and a blank line separates documents. This matches the A1 reader. Never split documents further before assigning train/dev/test.

## Sources and scope

- TinyStories V2 GPT-4, `roneneldan/TinyStories`: simple synthetic English stories useful for learning tokenization, count models, and small neural language models. Train comes from the full official training file; dev and test are disjoint samples of the official validation file. The three TinyStories sample files remain under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). These are modified subsets: the teaching team selected documents, stripped line edges, removed empty lines, and added blank-line record separators. The [original source card](sources/tinystories/README.md) is preserved.
- Chinese web text, `opencsg/Fineweb-Edu-Chinese-V2.1`, 4-5 score bucket: samples from 64 deterministically selected shards of the course mirror. The [upstream source card](sources/chinese-web/README.md) declares [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) and contains additional OpenCSG licensing text; preserve and consult both. These three sample files use the same document selection and whitespace transformations described above.

These are length-filtered teaching samples, not representative benchmarks of all English or Chinese. Direct scores across languages and sources do not isolate language difficulty.

## Reproducibility

Seed: `20260915`. `manifest.json` records source hashes, split hashes, document counts, filters, and the exact recipe. `provenance/` maps every output document to its source document index or parquet row. Exact and normalized duplicate texts never occur in two splits. Semantic near-duplicates may remain.

The teaching team can regenerate this sample using `grading/sample_a1_data.py` from the private instructor repository, into a new, nonexistent output directory:

```bash
python3 grading/sample_a1_data.py --source-root /path/to/pretraining --output /path/to/new-version
```

This command needs pyarrow. Students do not run it or select their own samples.

## Student use

The six text files are bundled with the assignment repository and student release; students do not need SSH or access to the full source mirrors. Read the six fixed files; train tokenizers and models on train, select settings on dev, and report final results on test. Use the explicit `chinese-web-dev.txt`; do not carve a new development split out of Chinese training.

Download the released assignment repository or the same starter archive attached on eLearning, then run locally. Submit only the documented solution files, results, and report; do not re-upload the supplied data. The archive contains no model outputs, grading answers, or hidden tests.
