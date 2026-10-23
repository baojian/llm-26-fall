# Explorations

Open the [explorations collection](index.html) in a browser.

Instructor-authored investigations of concrete questions about NLP and language
models. Each subfolder contains one explorable topic: the question, supplied
inputs, reproducible measurements, and an analysis of what the evidence does
and does not establish. Students can read the results and inspect the examples
without submitting work.

| Topic | Question | Read and explore |
| --- | --- | --- |
| [Tokenizer comparison](tokenizer-comparison/README.md) | What changes when the same text goes through six published tokenizers? | [Interactive report](tokenizer-comparison/index.html) · [Measured tables](tokenizer-comparison/results.md) |

## Adding a topic

Use a descriptive lowercase folder name. Give each topic a README with a
question, method, interpretation, limitations, primary sources, and reproduction
commands. Keep the topic's small input fixtures, source manifest, and authoring
code together. Keep reusable repository tests under `tests/`.

Published tables and pages are curated teaching material. Raw run records,
downloaded artifacts, logs, and exploratory outputs stay in ignored `workspace/`.
Document the hardware needed and distinguish measured observations from
explanations and extrapolations. Add a row here when a topic is ready to read.

These resources are separate from the course's [participation activities](../tasks/README.md)
and [graded assignments](../assignments/README.md).
