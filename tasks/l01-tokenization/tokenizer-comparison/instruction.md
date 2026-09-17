# Compare public tokenizers on the same text

**Voluntary and ungraded · about 45 minutes · rolling contributions · work
alone or with a classmate.** Everyone uses the same instructions.

## 1. Goal

In Lecture 01, a tokenizer turns text into a sequence of IDs drawn from a
vocabulary $V$. Here we ask: **what changes when the same text goes through
different tokenizers, and which part of the pipeline explains the change?**

We will build a shared collection of short, reproducible comparisons for
future lectures. Start with [the worked example](worked-example.md), then
choose **two** of the six tokenizers in [tokenizers.json](tokenizers.json).

| ID | Published artifact | Why it is in the pilot | Sources and notes |
| --- | --- | --- | --- |
| `r50k_base` | tiktoken vocabulary and encoding configuration | GPT-2-style byte BPE | [Published encoding definitions](https://github.com/openai/tiktoken/blob/0.14.0/tiktoken_ext/openai_public.py); download contains mergeable byte tokens, while regex and special tokens are separate |
| `cl100k_base` | tiktoken vocabulary and encoding configuration | Compare a later byte BPE vocabulary and splitting rule | [OpenAI's encoding examples](https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken); published encoding used by closed-weight models |
| `o200k_base` | tiktoken vocabulary and encoding configuration | Compare another vocabulary and multilingual splitting rule | [OpenAI's encoding examples](https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken); use this exact encoding name, not an inferred mapping to a current product |
| `qwen3` | `Qwen/Qwen3-8B/tokenizer.json` | BPE with NFC normalization and a different digit rule | [Pinned artifact manifest](tokenizers.json); public tokenizer for an open-weight model |
| `bert` | `google-bert/bert-base-uncased/tokenizer.json` | WordPiece, lowercasing, accent handling, unknown tokens | [Pinned artifact manifest](tokenizers.json); public tokenizer for an open-weight model |
| `t5` | `google-t5/t5-small/tokenizer.json` | Unigram, normalization, unknown tokens | [Pinned artifact manifest](tokenizers.json); public tokenizer for an open-weight model |

This is a representative sample, not a claim to collect every model. Some
closed-weight models publish a tokenizer; others do not. A tokenizer may
be shared by many models, and its files may be named `vocab.json`,
`merges.txt`, `.model`, or `.tiktoken` instead of `tokenizer.json`.
Unpublished or inaccessible artifacts belong in a future inventory as
**unavailable**, not as guessed vocabularies. No paid API or model weights
are needed for this task.

## 2. Predict before measuring

Copy [contribution-template.json](contribution-template.json) to
`submissions/<your-lowercase-github-username>.json`. For a pair, use the
first contributor's username for the filename and list both contributors.
The progress board currently counts the filename owner; the file and PR
credit both people.

Choose one question, for example:

- How do long digit sequences split in `r50k_base` and `qwen3`?
- What happens to Chinese mixed with English in `cl100k_base` and `o200k_base`?
- Does decoding recover capitalization, accents, and spacing in `bert` and `qwen3`?
- Can a smaller token count hide lost information in `t5` and `o200k_base`?

Keep the three required cases exactly as supplied: `digits`, `mixed-script`,
and `normalization`. The last case contains precomposed `É` and a separate
combining accent after the final `e`. Add at least **one original input**
not already in [inputs.json](inputs.json), such as a small change that tests
your explanation. Use short invented or public text; no personal messages.

For each case, write a prediction about the two tokenizers **before** running
the measurements. An incorrect prediction is useful: keep it and explain
what the result taught you. Set `question`, your chosen `tokenizers`, and
`contributors`. You will replace the `notes` and `sources_and_notes`
prompts after measuring.

## 3. Run the shared comparison

From the repository root, install the small optional dependency group and
download the pinned public tokenizer artifacts once:

```sh
uv sync --locked --extra tokenizer-comparison
uv run --locked --extra tokenizer-comparison python scripts/tokenizer_comparison.py fetch
```

Downloads go to the ignored `workspace/tokenizer-comparison/` directory.
The manifest pins each artifact's SHA-256 and its repository revision or
published encoding configuration. A hash mismatch stops the download;
do not change the expected hash merely to make it pass.

Measure your file, substituting your GitHub username for `alice`:

```sh
uv run --locked --extra tokenizer-comparison python scripts/tokenizer_comparison.py measure tasks/l01-tokenization/tokenizer-comparison/submissions/alice.json
```

This adds `observed` measurements and `provenance` to the file, preserving
your predictions and notes. It records normalization, a pre-tokenizer
preview, token pieces and IDs, token counts, decoded text, and exact
round-trip results. Run it again only when you change your input or chosen
tokenizers, then update your explanation.

To explore all ten supplied cases and all six tokenizers, save a local report:

```sh
uv run --locked --extra tokenizer-comparison python scripts/tokenizer_comparison.py compare > workspace/tokenizer-comparison/report.json
```

Or inspect one input and two tokenizers:

```sh
uv run --locked --extra tokenizer-comparison python scripts/tokenizer_comparison.py compare --tokenizers bert qwen3 --text 'CAFÉ café'
```

Only `fetch` uses the network. Other commands read the verified cache.
Missing files produce a fetch instruction rather than an implicit download.

## 4. Explain the evidence

Replace `notes` with **3–5 sentences** that answer your question. Identify a
specific observed difference, explain it using the vocabulary or pipeline,
and state one limit of your conclusion. In **Sources and notes**
(`sources_and_notes`), link the two pinned artifact sources and any supporting
documentation, with a short explanation of what each source establishes.

Keep these distinctions clear:

- **Pre-tokenizer chunks are not final tokens.** The preview calls the
  pre-tokenizer on normalized text in isolation. Added-token matching can
  occur before that stage, so the preview is not a full trace of encoding.
- **Token IDs are local to a vocabulary.** Equal IDs across tokenizers do
  not imply equal text. The displayed pieces also use different conventions:
  byte representations, `##`, `▁`, or a byte-level alphabet. Inspect the
  decoded text; individual byte tokens may split a UTF-8 character.
- **Vocabulary entries, ID capacity, and embedding rows differ.** The report
  counts actual vocabulary entries including added/special entries, and
  separately reports `highest_token_id_plus_one`. Special entries may
  already be in a base vocabulary, so do not always add the two counts.
  Embedding dimension and matrix shape require the model configuration or
  weights; they cannot be inferred from a tokenizer alone.
- **Normalization and unknown tokens can lose information.** A false
  `round_trip_exact` is an observation to explain, not automatically a bug.
  Fewer tokens on one string does not establish better language coverage,
  model quality, arithmetic ability, or a better training objective.
- **Compare ordinary text consistently.** The runner omits automatic BOS,
  EOS, padding, truncation, and chat templates. Special-looking input is
  processed as ordinary text; other added tokens keep their published
  matching rules. Counts describe this policy, not a complete API request.

Useful pipeline reference: [Hugging Face's tokenizer pipeline](https://huggingface.co/docs/tokenizers/main/en/pipeline).

## 5. Check and contribute

After writing your explanation, verify your file:

```sh
uv run --locked --extra tokenizer-comparison python scripts/tasks.py check tasks/l01-tokenization/tokenizer-comparison alice
```

The checker independently re-encodes every case with the pinned tokenizers.
It checks the filename, required inputs, distinct cases, source/runtime
provenance, and measured outputs. It does **not** grade your predictions or
judge whether your explanation is convincing; a human reviewer does that.

Open a PR containing only your JSON file, titled
`l01-tokenization/tokenizer-comparison: <username>`. State the pair of
tokenizers and your question in the body. Link this task and use
`Related to #154` for the companion model-table project. Several students
may investigate the same pair; no exclusive claim is needed. Graded A1
implementations and reports are submitted privately through eLearning.

Ask a classmate to review one observation: reproduce the result, inspect the
source, and say whether the explanation follows from the evidence. The
teaching team reviews the file before merging. A successful contribution
adds a useful example or a well-supported correction, even if its initial
prediction was wrong.

## 6. How we will use the results

The instructor supplies this runner, the verified pilot, and common inputs.
Students supply questions, examples, and peer reviews. We will select a few
contrasting cases for a short class discussion and use verified vocabulary
counts in the [model table and timeline project](https://github.com/baojian/llm-26-fall/issues/154).
That project records release dates, embedding dimensions, and context windows
with separate model-level sources.

After the pilot, we can propose another tokenizer with a public source,
exact revision, artifact hash, and an explanation of what new comparison it
enables. Adding a backend or changing the manifest is a separate teaching-team
change: review the source and license, verify the counts, refresh the worked
example, and re-measure existing contributions together. Do not mix those
changes into an individual contribution PR.
