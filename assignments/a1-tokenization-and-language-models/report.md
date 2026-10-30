# A1 report

Student ID: TODO

Replace every `TODO` (the packager refuses a report that still contains one).
Keep the headings. About 400 words for sections 2 to 4 together; the table and
the disclosure box do not count. Every number you quote must come from your own
`results.json` or your own runs.

## 1. Results table (Part 4)

Orders 1 to 3 with MLE plus floor, add-δ, and interpolation on both test sets,
and the neural model on TinyStories. Perplexity and bits per byte for each.

| Corpus | Vocabulary size | Bytes per token |
| --- | ---: | ---: |
| TinyStories | TODO | TODO |
| Chinese web | TODO | TODO |

| Corpus | Order | Model | Perplexity | Bits per byte |
| --- | ---: | --- | ---: | ---: |
| TinyStories | 1 | MLE + floor | TODO | TODO |
| TinyStories | 1 | add-δ | TODO | TODO |
| TinyStories | 1 | interpolation | TODO | TODO |
| TinyStories | 2 | MLE + floor | TODO | TODO |
| TinyStories | 2 | add-δ | TODO | TODO |
| TinyStories | 2 | interpolation | TODO | TODO |
| TinyStories | 3 | MLE + floor | TODO | TODO |
| TinyStories | 3 | add-δ | TODO | TODO |
| TinyStories | 3 | interpolation | TODO | TODO |
| TinyStories | neural (context 3) | untied / tied | TODO | TODO |
| Chinese web | 1 | MLE + floor | TODO | TODO |
| Chinese web | 1 | add-δ | TODO | TODO |
| Chinese web | 1 | interpolation | TODO | TODO |
| Chinese web | 2 | MLE + floor | TODO | TODO |
| Chinese web | 2 | add-δ | TODO | TODO |
| Chinese web | 2 | interpolation | TODO | TODO |
| Chinese web | 3 | MLE + floor | TODO | TODO |
| Chinese web | 3 | add-δ | TODO | TODO |
| Chinese web | 3 | interpolation | TODO | TODO |

## 2. Samples

Three stories sampled from your interpolated trigram (`sample`). Optionally
three more from the neural model, labelled as such.

```text
TODO sample 1
```

```text
TODO sample 2
```

```text
TODO sample 3
```

Two sentences on what the samples get right and what they get wrong: TODO

## 3. One design choice

Choice (one of: vocabulary size 1000 vs 4000, the digit rule, δ, the
interpolation weights, the embedding dimension): TODO

Claim: TODO

Evidence (at least two numbers from `results.json` or a second run, with the
setting each number came from): TODO

What these numbers cannot tell you (one sentence): TODO

## 4. Embeddings

Nearest neighbours of a few frequent tokens (from `results.json`): TODO

Parameter counts: untied TODO, tied TODO.

Why counts cannot generalize the way embeddings do (one sentence): TODO

## AI-use disclosure

Fill in every line; write "none" where nothing applies. A report without this
box loses 5 points. Using assistants is allowed; hiding it is not.

- Tools used (name, version if known): TODO
- What you used them for (ideas, explanations, debugging, code, prose): TODO
- What you checked yourself, and how (tests you ran, numbers you recomputed): TODO
- What you adopted, changed, or rejected, and why: TODO
- I confirm that every number in this report was measured by my own code: TODO (yes / no, and if no, which)
