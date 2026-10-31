# Model tokenizer catalog: October 31, 2026

[Download catalog.zip](catalog.zip) · [Checksums and sizes](manifest.json)

This versioned data snapshot contains **11,229 rows and 56 columns** in CSV
and JSON. It is about **4.55 MB compressed / 44.12 MB extracted** and includes:

- `model-catalog.csv`: the complete table, ready for a dataframe or spreadsheet.
- `model-catalog.json`: the same records with numeric, boolean, and array types.
- `catalog-summary.json`: coverage and missing-value counts.
- `tokenizer-groups.json`: groups sharing identical primary tokenizer JSON bytes.
- `vocabulary-groups.json`: groups sharing identical token-to-ID mappings.

The snapshot is an analysis input. Tokenizer assets, model weights, API caches,
and generated figures are not bundled. Counts and provenance are unchanged
from the verified October 31 collection. See the [methodology and field
definitions](../../expanded-catalog.md) before interpreting the timeline.

From the repository root:

```sh
uv run --no-project --python 3.11 python -m zipfile -e \
  docs/model-tokenizer-statistics/snapshots/2026-10-31/catalog.zip \
  workspace/hf-tokenizer-catalog

uv run --no-project --python 3.11 --with matplotlib==3.11.1 \
  python scripts/plot_hf_tokenizer_catalog.py
```

Use a new destination directory if you already have edited working copies;
the extraction command replaces files with the same names. The plot accepts
`--catalog <path-to-model-catalog.json>` and `--output-dir <figure-directory>`.
Actual tokenizer files are not required for these plots. Relative tokenizer
cache paths in the records become usable after their corresponding assets
have been downloaded.

For the exact source snapshot, retain this dated archive. Collecting again
can change download rankings, repository coverage, and inspected revisions.
