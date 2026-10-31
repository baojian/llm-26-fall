# Model and tokenizer statistics

[Download CSV](models.csv) · [Canonical JSON](models.json)

This initial collection contains **84 model releases from 2018–2026**:
65 with open weights and 19 with closed weights. It includes historically
significant model families and a few encoder/encoder-decoder tokenizer
baselines, such as BERT and T5. It is a curated starting point, not an exhaustive
catalog or a ranking measured by downloads, users, or benchmark popularity.
Collection date: **2026-10-31**.

The [expanded Hub catalog](expanded-catalog.md) adds a reproducible discovery
and measurement workflow for **11,210 Hugging Face repositories**, including
fine-tunes, quantized copies, merges, and adapters. Its local CSV/JSON includes
these 84 curated records and has **11,229 rows** in total. A compressed
[data snapshot](snapshots/2026-10-31/) is committed for use on another computer.
Working copies, tokenizer caches, and generated figures remain in ignored
`workspace/`.

Each row describes one named model variant or API snapshot. Parameter sizes,
base models, instruction variants, and updated checkpoints are not assumed to
share identical training histories. The first seven columns contain the main
analysis fields; later columns record definitions, provenance, and limitations.

## Coverage and missing values

| Tokenizer status | Model rows | Meaning |
| --- | ---: | --- |
| `downloaded` | 64 | Public tokenizer artifacts were downloaded, hashed, and measured locally |
| `gated` | 12 | Publisher access controls prevented local collection |
| `unavailable_at_collection` | 2 | The original MPT/DBRX repositories returned HTTP 401; publisher metadata is retained |
| `not_verified` | 6 | No exact public tokenizer mapping was verified for this row |

Unknown or unverified values are **JSON `null` / empty CSV cells**, never zero.
An unverified field does not establish that the information is undisclosed or
that no public implementation exists. For example, Anthropic publishes a
[legacy tokenizer](https://github.com/anthropics/anthropic-tokenizer-typescript),
but its mapping to the specific Claude snapshots here has not been established.
No community tokenizer is silently substituted for a provider tokenizer.

`open_weights` describes access to weights, not an OSI license judgment or
public availability of training data. Gated Llama and Gemma releases still
belong in that category. License identifiers are copied from repository
metadata when available; the linked provider terms remain authoritative.

## What the counts mean

- **Tokenizer vocabulary:** distinct assigned token IDs in the serialized
  tokenizer plus explicitly configured added tokens. Base tokens that are also
  marked special are counted once. Reserved entries present in the inspected
  representation are included; missing IDs are not invented.
- **ID capacity:** largest assigned token ID plus one. This can exceed the
  number of entries when there are gaps. Yi 6B has 63,992 entries but an ID
  capacity of 64,000 in the inspected artifact.
- **Configuration vocabulary:** `vocab_size` in the model's text configuration.
  This is recorded separately and may include padding. It is not a measurement
  of tensor shapes: no model weights were downloaded.
- **Special tokens:** tokens explicitly marked special by the artifact,
  added-token configuration, reviewed implementation, or SentencePiece control/
  unknown classification. This is an annotation count, not a cross-provider
  semantic classification of every possible control marker.
- **Dataset tokens:** a published pretraining corpus size. **Tokens processed:**
  the published training exposure, which can include repeated data. These are
  separate columns. Words, bytes, documents, and dataset size are not converted
  into training tokens by assumption.

Counts are integers in token units: `1000000000000` means one trillion tokens.
Always retain `pretraining_count_qualifier` and `pretraining_count_notes` in an
analysis. Some figures describe a family maximum, a lower bound, one training
stage, inherited base-model training, or continued pretraining. They are not
interchangeable totals. Multimodal token accounting is not normalized across
providers. Nominal parameter counts use billions and total rather than active
MoE parameters; precision varies with the publisher's description.

### Public tokenizers for closed models

For OpenAI rows, the vocabulary is the mapped public **text encoding** from
tiktoken 0.14.0. For example, `cl100k_base` has 100,261 assigned entries and an
ID capacity of 100,277; `o200k_base` has 200,000 entries and an ID capacity of
200,019. Neither value establishes a closed model's embedding size, private
control tokens, multimodal vocabulary, or complete API token accounting.

The Gemini rows use the local text-tokenizer mappings in pinned official
Google SDK source. The mapping source and tokenizer artifact source are both
recorded. A current SDK mapping does not prove that the initial preview used
the same tokenizer or that the complete multimodal service uses only these
entries. `tokenizer_count_scope` preserves this distinction.

The gpt-oss rows measure the published Hugging Face representation, which has
200,019 entries. Their configuration has 201,088 vocabulary rows; tiktoken's
`o200k_harmony` reserved-token representation is another distinct object.
Kimi K2's pinned code allocates 258 special/reserved entries, whereas K2.5
allocates 256. The K2 artifact/config discrepancy is retained in its row.

## Dates, revisions, and historical analysis

`release_date` describes the named release, with `day`, `month`, or `year`
precision. A partial date remains partial; it is not filled with January 1 or
the first day of a month. The collection does not use Hugging Face repository
creation time as a model release date. A preview, a stable API snapshot, a
model announcement, and the later publication of weights can have different
dates; row names and notes identify the selected event.

`timeline_date` supplies a separate day-level plotting coordinate: a documented
release day takes priority; otherwise the earliest inspected repository record
is used and explicitly labeled as a proxy. The original partial `release_date`
is preserved. `timeline_date_basis` and its source URL distinguish the events.
The fallback considers repository creation and, for the curated records and
legacy repositories, paginated commit history. It does **not** establish the
first public access to model weights. Hugging Face's synthetic March 2, 2022
creation-date floor is excluded as a candidate. A commit can predate a public
release, and a later Hub migration can postdate it.

There are 82 dated curated rows, including 63 with measured vocabularies.
GPT-1, GPT-3's API debut, and Gemini 1.0 have been assigned exact announcement
days from first-party dated announcements. Unresolved dates stay blank.

Tokenizer files are pinned to **revisions inspected at collection time**.
These are not necessarily launch-day revisions. Preserve that distinction
when plotting vocabulary size against model release year. Several modern
families share tokenizers, so treating every checkpoint as an independent
tokenizer design can also bias a trend. Community derivatives are included in
the expanded catalog with declared ancestry and shared-vocabulary identifiers.

## Column dictionary

| Column | Definition |
| --- | --- |
| `release_date` | ISO `YYYY-MM-DD`, `YYYY-MM`, or `YYYY`; blank if unverified |
| `model_name` | Human-readable name identifying the variant/update |
| `architecture` | Published implementation class, or a sourced architecture description when configuration is unavailable; not a normalized architecture taxonomy |
| `tokenizer_local_path` | Directory relative to the repository root; populated only for collected artifacts |
| `tokenizer_vocab_size` | Total distinct assigned token IDs, including added tokens |
| `pretraining_dataset_tokens` | Published corpus-token count, interpreted with the qualifier and notes |
| `pretraining_tokens_processed` | Published training-token exposure, interpreted with the qualifier and notes |
| `model_id` | Unique provider/repository identifier or provider-prefixed API model name |
| `developer` | Publisher namespace |
| `weights_access` | `open_weights` or `closed_weights` |
| `license` | Hugging Face license metadata when collected; blank does not imply no license |
| `parameters_billion` | Published nominal total parameter count, in billions |
| `release_date_precision` | `day`, `month`, `year`, or blank |
| `release_date_source_url` | First-party release record, model card, or original paper supporting the date |
| `model_source_url` | Model card or official model description |
| `architecture_source_url` | Configuration or publication supporting the architecture |
| `config_model_type` | `model_type` from the text configuration |
| `config_hidden_size` | Text hidden dimension from `hidden_size`, `n_embd`, or `d_model` |
| `config_vocab_size` | Vocabulary rows declared by the text configuration |
| `tokenizer_status` | Collection status defined above |
| `tokenizer_count_scope` | `serialized_artifact_and_added_tokens`, `public_text_encoding_only`, or `official_sdk_text_tokenizer` |
| `tokenizer_base_vocab_size` | Entries serialized in the base vocabulary before separate added-token records |
| `tokenizer_added_vocab_size` | Additional IDs absent from that base vocabulary |
| `tokenizer_special_token_count` | Distinct IDs annotated special under the definition above |
| `tokenizer_id_capacity` | Maximum assigned ID plus one |
| `tokenizer_algorithm` | Algorithm identified in the inspected tokenizer, e.g. BPE, Unigram, WordPiece |
| `tokenizer_format` | File representation inspected by the measurement script |
| `tokenizer_encoding_name` | tiktoken encoding name, where applicable |
| `tokenizer_primary_file` | Primary vocabulary filename inside the local directory |
| `tokenizer_profile` | Reviewed custom rank-file rule: `qwen1`, `kimi_k2`, or `kimi_k25`; otherwise blank |
| `tokenizer_revision` | Pinned repository commit or tiktoken package version |
| `tokenizer_sha256` | SHA-256 of the primary artifact; companion hashes are in the manifest |
| `tokenizer_source_url` | Artifact or official encoding documentation |
| `tokenizer_mapping_source_url` | Official model-to-tokenizer mapping when a separately published encoding is used |
| `reported_vocab_size_text` | Published, potentially rounded or conflicting vocabulary description, separate from measured counts |
| `reported_vocab_source_url` | Source of that published description |
| `pretraining_dataset` | Published corpus name or description; not a claim of dataset availability |
| `pretraining_count_qualifier` | Scope/precision of the reported counts, such as `greater_than` or `stage_1_only` |
| `pretraining_count_notes` | How the source's training figures were interpreted |
| `pretraining_source_url` | Source supporting training-token figures |
| `collected_at` | Date this record was curated and its artifacts inspected |
| `notes` | Remaining variant, provenance, and comparison limitations |
| `hf_repository_created_at` | Original Hub timestamp, retained even when it is the legacy synthetic floor |
| `earliest_repository_record_at` | Earliest usable timestamp among the inspected repository records; not proof of public weight availability |
| `earliest_repository_record_kind` | `earliest_hf_commit` or `hf_repository_creation` |
| `earliest_repository_record_source_url` | Commit page or repository API supporting that timestamp |
| `timeline_date` | Day-level plotting coordinate, distinct from the original release date |
| `timeline_date_basis` | `documented_release`, `earliest_hf_commit`, or `hf_repository_creation` |
| `timeline_date_source_url` | Source supporting the selected plotting coordinate |

## Read, reproduce, and extend

The JSON array is canonical. The CSV has the same columns and values, UTF-8
encoding, a header row, and conventional quoting. For example:

```python
import json
from pathlib import Path

rows = json.loads(Path("docs/model-tokenizer-statistics/models.json").read_text())
for row in rows:
    if row["tokenizer_vocab_size"] is not None:
        print(row["release_date"], row["model_name"], row["tokenizer_vocab_size"])
```

Run from the repository root:

```sh
# Offline structure, provenance, and JSON/CSV consistency checks.
uv run python scripts/llm_tokenizer_statistics.py check

# Rebuild CSV after a reviewed edit to the canonical JSON.
uv run python scripts/llm_tokenizer_statistics.py export

# Download the pinned tokenizer/configuration/card files, then verify hashes.
uv run python scripts/llm_tokenizer_statistics.py fetch

# Recompute all measured counts from those files, without model inference.
uv run --with sentencepiece==0.2.1 --with protobuf==6.33.6 \
  python scripts/llm_tokenizer_statistics.py measure

uv run --group dev python -m pytest -q tests/test_llm_tokenizer_statistics.py
```

The [file manifest](tokenizer-files.json) contains 307 files totaling about
524 MB (decimal), including shared files only once. Files live in ignored
`workspace/llm-tokenizers/`; tokenizer assets are not committed. The downloader
retrieves only manifest entries, checks SHA-256, and never downloads model
weights or executes publisher Python files. Small tiktoken encoding definitions
are retained in [encodings/](encodings/) to reproduce their special-token sets.
Files with access failures are not part of the download manifest. This command
does not authenticate to gated repositories.

An existing cache can be used with `--cache-root /path/to/repository`. A
different checkout can reproduce it with `fetch`. Source availability and
publisher terms can change; hashes prevent silently accepting changed files.

To extend the collection, add a fully sourced model record, keep unknowns
blank, pin and hash any new artifacts, and update the manifest. Review custom
tokenizer implementations before adding a counting profile. Run `export`,
`check`, and `measure`; update the coverage figures above. Add release variants
as separate rows when their training history or tokenizer differs. A future
popularity study should add an explicit metric, observation date, and sampling
rule instead of treating this selection as a popularity ranking.

Related course work: [vocabulary timeline issue #154](https://github.com/baojian/llm-26-fall/issues/154)
and [tokenizer comparison PR #156](https://github.com/baojian/llm-26-fall/pull/156).
