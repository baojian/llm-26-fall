# Expanded model and tokenizer catalog

The October 31, 2026 collection contains **11,229 entries**: 11,210 public
Hugging Face repository records and 19 curated closed-model records. One row
represents a repository or named API snapshot, not an independently trained
model or tokenizer. Fine-tunes, quantized copies, GGUF conversions, adapters,
and merges are included. Test fixtures and specialized scientific models
returned by the task queries are retained too.

The full JSON/CSV has 56 columns. Large generated files, cached API responses,
tokenizer artifacts, and figures live in ignored `workspace/`; the collection,
export, and plotting scripts are versioned. The separate [84-row curated
collection](README.md) retains more detailed manually checked release and
training-data provenance.

## Selection and coverage

Discovery sorts by current 30-day downloads and requests these task samples:

| Hub pipeline tag | Requested | Returned |
| --- | ---: | ---: |
| `text-generation` | 10,000 | 10,000 |
| `image-text-to-text` | 1,000 | 1,000 |
| `text2text-generation` | 500 | 0 |
| `any-to-any` | 200 | 200 |

After deduplication, the queries cover 11,198 repositories; twelve curated
repositories outside those samples are added. The zero-result
query is preserved in the discovery summary; it does not establish coverage
of the encoder-decoder population. These are current popularity samples, not
a census of all LLMs or of historically popular releases. Pagination spans
several minutes rather than an atomic Hub snapshot. Downloads are a repository
usage signal and can include automated or test activity. Each row retains its
selection task, rank, downloads, likes, and observation time.

| Measurement result | Rows |
| --- | ---: |
| Directly measured root `tokenizer.json` | 5,084 |
| Other directly measured curated artifacts | 8 |
| Closed-model public text encodings | 13 |
| No root `tokenizer.json` collected | 5,932 |
| Gated | 178 |
| Unsupported or conflicting tokenizer definitions | 8 |
| Unverified closed-model tokenizer | 6 |
| **Total** | **11,229** |

All 5,092 ungated repositories listing a root `tokenizer.json` were attempted.
There are **5,105 measured rows**, of which **5,104 have a timeline date**.
The JSON measurements cover **1,127 distinct files** and **637 distinct
token-to-ID mappings**. The plot collapses those mappings plus other-format
artifact groups into **649 dated groups**.

Many quantized repositories package tokenizers inside GGUF weights or refer
to a parent repository. This pass does not download weight files, unpack GGUF,
or assign a parent's tokenizer count to its descendants. Absence of a root
JSON is a collection limitation, not proof that a tokenizer is unavailable.
Unknown counts remain `null` / empty CSV cells. The cache occupies roughly
ten GB; downloads deduplicate identical Hub blob identifiers.

## Definitions and provenance

| Fields | Interpretation |
| --- | --- |
| `release_date`, precision, source | Documented model event from the curated record; otherwise unknown |
| `timeline_date`, basis, source | Exact documented day when available; otherwise a labeled repository-date proxy |
| `hf_repository_created_at` | Raw creation timestamp, including legacy synthetic values |
| `earliest_repository_record_*` | Earliest inspected usable creation/commit record and its evidence |
| `declared_base_models` | Parent identifiers explicitly present in Hub tags or model-card metadata |
| `declared_lineage_roots` | Terminal ancestors in the sampled declared graph, possibly outside the sample; not independently verified original models |
| `variant_types` | Nonexclusive metadata relations; library/format indicators add adapter, GGUF-conversion, or merge labels |
| `lineage_cycle_detected` | True when declared ancestry loops; no origin is invented to break the loop |
| `shared_tokenizer_json_sha256` | Equal primary JSON bytes, including merges and serialized configuration; excludes companion files |
| `shared_vocabulary_sha256` | Equal sorted ID-to-token mapping including serialized added tokens; ignores formatting, merges, scores, and special-token flags |
| `shared_*_repositories` | Number of directly measured repositories sharing that identifier in this sample |
| `declared_base_tokenizer_json_sha256` | Known parent artifact references, never a substituted measurement of the child |
| `tokenizer_count_scope` | Bulk counts use `serialized_tokenizer_json_only`; curated and closed-model scopes retain their explicit definitions |
| `tokenizer_local_path` | Cache directory relative to the checkout root; content-addressed JSON folders are shared |
| `revision`, artifact SHA-256, source URL | Pinned inspected repository revision and downloaded artifact evidence |
| `architecture`, `config_model_type` | Hub configuration metadata, with curated fallback; no repository Python is executed |
| `parameters_total` | Hub-reported safetensors parameter total, when present; can describe an adapter rather than a complete model |
| `weights_access` | Curated access classification, or `hub_repository_unverified` for newly discovered repositories; discovery does not verify their weights |
| `pretraining_*` | Exact curated matches only; corpus size and training exposure remain separate and unknown values are not inherited |

Other scalar count fields follow the [curated definitions](README.md#what-the-counts-mean).
The broad pass reads the root JSON only. Companion files can add entries or
change behavior, so its scope can differ from a Transformers loader or a
curated measurement that includes those files. Equal vocabulary hashes do not
prove equal tokenization behavior or common training ancestry. Equal vocabulary
sizes alone are never used to identify shared tokenizers.

Arrays remain arrays in JSON and use compact JSON strings inside CSV cells.
CSV booleans are `true`/`false`. Published pretraining figures remain sparse;
the collector does not infer them from parameter counts or model size.

## Dates and figures

Repository creation is used for ordinary recent records. Full paginated commit
history is also inspected for curated repositories and records whose creation
time is on the Hub's March 2, 2022 storage floor. That floor is never treated
as an actual date. The [Hub API documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)
documents this creation-date limitation and reverse-chronological commit lists.
An earliest commit may contain only repository scaffolding, while imported Git
history may predate the Hub repository. Neither creation nor commit time is
proof of the first public availability of model weights. Later repository
uploads can also postdate the true model release.

The original partial release date is retained alongside the selected plotting
date. Filled markers indicate documented release days; hollow markers indicate
proxies. The figure contains linear and logarithmic y-axes, both for all dated
measurements and for grouped vocabularies/artifacts. Each group's x-coordinate
is the earliest model date observed in this sample, not the tokenizer's proven
invention date. The plot contains no date-range bars, regression, or uncertainty
intervals. Small-vocabulary scientific and test models remain visible.

These observations alone do not establish a scaling law. Shared ancestry,
current-download selection, repository migrations, current tokenizer revisions,
missing gated measurements, and heterogeneous model purposes affect a time
trend. Dates and measurements remain traceable so a later analysis can define
its population and filters explicitly.

## Reproduce

From the repository root, use a fresh `--cache-dir` for a new observation date.
Without it the commands reuse cached responses and measurements. The discovery
summary records actual query sizes. These commands need no Hub authentication:

```sh
# Populate the pinned curated artifacts used by other-format and closed rows.
uv run python scripts/llm_tokenizer_statistics.py fetch

uv run --with requests==2.34.2 python scripts/collect_hf_tokenizer_catalog.py discover
uv run --with requests==2.34.2 python scripts/collect_hf_tokenizer_catalog.py history
uv run --with requests==2.34.2 python scripts/collect_hf_tokenizer_catalog.py measure
uv run python scripts/collect_hf_tokenizer_catalog.py fingerprint
uv run python scripts/export_hf_tokenizer_catalog.py

# Optional: write the date-enriched 84-row canonical files after reviewing sources.
uv run python scripts/export_hf_tokenizer_catalog.py --update-curated

uv run --with matplotlib==3.11.1 python scripts/plot_hf_tokenizer_catalog.py
uv run --group dev python -m pytest -q \
  tests/test_llm_tokenizer_statistics.py tests/test_hf_tokenizer_catalog.py
```

Generated outputs are `workspace/hf-tokenizer-catalog/model-catalog.{json,csv}`,
`tokenizer-groups.json`, `vocabulary-groups.json`, and collection summaries.
Figures go to `workspace/model-tokenizer-statistics/figures/`. Use the plot's
`--catalog` and `--output-dir` options with a different cache location.
The collector validates Hub content hashes, retries transient requests, and
respects HTTP 429 retry information. It never loads model weights or executes
publisher code. Eight unsuccessful measurements remain explicitly recorded.

To view local files, serve only the intended output directory on loopback with
`python -m http.server --bind 127.0.0.1 --directory <output-directory> 8767`.
Local HTTP links require that server to remain running.
