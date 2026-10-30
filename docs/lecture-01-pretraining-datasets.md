# Lecture 01 Companion: Pretraining Corpora for Tokenizers and LLMs

**Course:** NLP and LLMs, Fall 2026, Fudan University  
**Dataset cards and repositories checked:** September 12, 2026  
**Scope:** Which public corpora real language models are pretrained on, what
the course mirrors for hands-on tokenization work, and how to read the files.

The [Lecture 01 preprocessing notes](lecture-01-pre-tokenization.md) explain
*how* text is prepared before and inside a tokenizer. This companion answers
the question that comes next: *which text?* Every large open model release
since 2023 documents a mixture of web crawls, curated sources, code, math,
and non-English corpora. Most of those corpora are public. A bounded sample
of each is enough to train a tokenizer, compare compression across
languages, and inspect the artifacts of corpus preparation (quality scores,
duplicate flags, language tags) that production pipelines leave behind.

The download script `scripts/download_pretraining_datasets.py` mirrors the
samples described here. `--list` prints the catalog; `--dry-run` shows
exactly which files a run would fetch.

## Contents

1. [How to read the catalog](#1-how-to-read-the-catalog)
2. [English web-scale corpora](#2-english-web-scale-corpora)
3. [Multi-source, encyclopedic, PDF, and synthetic corpora](#3-multi-source-encyclopedic-pdf-and-synthetic-corpora)
4. [Math and code](#4-math-and-code)
5. [Chinese corpora](#5-chinese-corpora)
6. [Small classroom sets](#6-small-classroom-sets)
7. [Gated or unmirrored datasets](#7-gated-or-unmirrored-datasets)
8. [The course mirror](#8-the-course-mirror)
9. [Reading the files](#9-reading-the-files)
10. [Exercises that use these samples](#10-exercises-that-use-these-samples)
11. [Licensing](#11-licensing)

## 1. How to read the catalog

Each table has the same columns.

| Column | Meaning |
| --- | --- |
| **Repository** | The Hugging Face dataset id. Most also exist on ModelScope under the same id; the script records which mirror it used in `manifest.json`. |
| **Reported size** | What the dataset card states, quoted as found on the check date. Token counts depend on the tokenizer the authors used (GPT-2, GPT-NeoX, Llama, ...) and are not comparable across rows. |
| **Format** | File type of the data shards. |
| **Mirrored sample** | The sub-tree and byte budget the course script selects. Sizes are measured from the repository listing during the dry run. |

Three evidence categories apply, as in the preprocessing notes: **reported**
(dataset card or paper), **artifact-verified** (file listing or local probe),
and **recommended** (a course choice). Sizes in the "Mirrored sample"
column are artifact-verified; "Reported size" is reported.

## 2. English web-scale corpora

These are the Common Crawl derivatives that dominate open pretraining
mixtures. They differ mainly in extraction, filtering, deduplication, and
how much metadata is kept per document.

| Dataset | Repository | Reported size | Format | Mirrored sample |
| --- | --- | --- | --- | --- |
| FineWeb | `HuggingFaceFW/fineweb` | 18.5T tokens (originally 15T), 96+ Common Crawl snapshots, English only | parquet | `sample/10BT`, the official 10B-token random subset, 28.6 GB |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 1.3T tokens kept by an educational-quality classifier (score >= 3); a 5.4T "score-2" variant exists | parquet | `sample/10BT`, 26.6 GB |
| DCLM-baseline | `mlfoundations/dclm-baseline-1.0` (+ `-parquet`) | 4T tokens, 3B documents, fastText quality filter trained on OpenHermes vs. Reddit ELI5 | jsonl.zst or parquet | first local shard of global shard 1, 9.8 GB |
| Dolma 3 pool | `allenai/dolma3` | 9T-token pool (8.14T from Common Crawl) behind OLMo 3; the 6T training mix is a separate repository | jsonl.zst | one topic partition, 10 GB (relayed; no ModelScope mirror) |
| RedPajama-V2 | `togethercomputer/RedPajama-Data-V2` | 100B+ documents, 30B with quality signals, 20B after deduplication; English alone 37T tokens | json.gz + signal files | the official `sample` tree with documents, quality signals, minhash, duplicates, 4.9 GB |
| Ultra-FineWeb (English) | `openbmb/Ultra-FineWeb` | 1T English tokens (plus 120B Chinese), MiniCPM4 corpus | parquet | first 4 English shards, 4.8 GB |
| Zyda-2 | `Zyphra/Zyda-2` | 5T tokens combining DCLM, FineWeb-Edu, Zyda-1, Dolma-CC with cross-deduplication | parquet | first shards of the cross-deduplicated DCLM component, 4.9 GB |
| Essential-Web v1.0 | `EssentialAI/essential-web-v1.0` | 24T tokens, 23.6B documents, every document labeled with a 12-category taxonomy | parquet | first shards of one crawl, 5 GB (relayed) |
| TxT360 | `LLM360/TxT360` | 99 Common Crawl snapshots plus 14 curated sources, globally deduplicated; the recommended recipe yields 15T+ tokens | jsonl.gz | first Common Crawl shards, 5 GB (relayed) |
| C4 | `allenai/c4` | The T5 corpus (2019 crawl, English `en` config about 365M documents) | json.gz | 10 train shards + 1 validation shard, 1.8 GB |
| OpenWebText | `Skylion007/openwebtext` | Reddit-linked pages reproducing GPT-2's WebText, about 8M documents | parquet | complete, 12 GB (relayed) |
| The Pile (uncopyrighted) | `monology/pile-uncopyrighted` | The 2020 EleutherAI 825 GB mixture with Books3, BookCorpus2, OpenSubtitles, YouTube subtitles, and OWT2 removed | jsonl.zst | train shard 00 plus val and test, 11 GB (relayed) |

**What to look at.** FineWeb and FineWeb-Edu shards carry `language_score`,
`token_count`, and the crawl `dump` per row; RedPajama-V2's sample includes
about 40 quality signals per document as a separate file; Essential-Web adds
taxonomy labels. These fields show what a corpus-preparation pipeline
computes *before* any tokenizer sees the text.

## 3. Multi-source, encyclopedic, PDF, and synthetic corpora

| Dataset | Repository | Reported size | Format | Mirrored sample |
| --- | --- | --- | --- | --- |
| FinePDFs (English) | `HuggingFaceFW/finepdfs` | About 3T tokens from 475M PDF documents in 1,733 languages; English 1.19T tokens | parquet | English test split + first train shard, 4.5 GB |
| Wikipedia (English) | `wikimedia/wikipedia` | 2023-11-01 dump, all articles | parquet | complete `20231101.en`, 41 shards, 10.8 GB |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus` | 39M synthetic textbooks, blog posts, and stories generated by Mixtral-8x7B | parquet | first 4 shards, 4.4 GB |
| Python-Edu | `HuggingFaceTB/smollm-corpus` | Educational Python files scored by a classifier (SmolLM corpus) | parquet | complete, 614 MB |
| Nemotron pretraining sample | `nvidia/Nemotron-Pretraining-Dataset-sample` | Small sample of the Nemotron Nano 2 corpus: Common Crawl, synthetic QA, math, code, SFT | parquet | complete, 45 MB (relayed) |
| Dolma v1.x URL lists | `allenai/dolma` | Dolma v1.7 is 3T tokens, 4.5 TB; shards are hosted outside Hugging Face | txt | the URL lists only, 1.3 MB |
| RedPajama-1T URL lists | `togethercomputer/RedPajama-Data-1T` | 1.2T tokens across seven sources (Common Crawl, C4, GitHub, arXiv, books, Wikipedia, StackExchange) | txt | the URL lists only, 207 KB |

The two URL-list entries are included on purpose. Dolma and RedPajama-1T
show a common pattern: the Hugging Face repository is a loader plus an
index, and the data lives on a separate object store. Reading the lists
tells you the shard naming scheme and size of a real corpus without
downloading it.

## 4. Math and code

| Dataset | Repository | Reported size | Format | Mirrored sample |
| --- | --- | --- | --- | --- |
| Proof-Pile-2 (arXiv) | `EleutherAI/proof-pile-2` | 55B tokens total; arXiv 29B, OpenWebMath 15B, AlgebraicStack 11B (Llemma corpus) | jsonl.zst | first 12 arXiv train shards, 2.8 GB |
| Proof-Pile-2 (AlgebraicStack) | `EleutherAI/proof-pile-2` | 11B tokens of mathematical code, numerical computing, formal proofs | jsonl.zst | train shards across languages, 2.9 GB |
| OpenWebMath | `open-web-math/open-web-math` | 14.7B tokens, 6.3M documents of mathematical web text with LaTeX preserved | parquet | first shards, 4.9 GB (ModelScope copy taken from Proof-Pile-2) |
| FineMath-4+ | `HuggingFaceTB/finemath` | 9.6B tokens, 6.7M documents (the 3+ tier is 34B tokens) | parquet | first 18 shards, 4.8 GB |
| StarCoderData (Python) | `bigcode/starcoderdata` | About 250B tokens across 86 languages from The Stack v1.2, deduplicated | parquet | first 27 Python shards, 9.9 GB (gated on Hugging Face; ModelScope serves it openly) |

Math and code are where tokenizer choices show up most clearly: digit
grouping, whitespace runs, LaTeX commands, and indentation all interact with
the pre-tokenization regex discussed in the preprocessing notes.

## 5. Chinese corpora

Chinese is the second-largest language in most modern mixtures and the one
that stresses byte-level tokenizers most: three bytes per character, no
spaces to pre-tokenize on, and a large character inventory.

| Dataset | Repository | Reported size | Format | Mirrored sample |
| --- | --- | --- | --- | --- |
| FineWeb-2 (Chinese) | `HuggingFaceFW/fineweb-2` | 1,000+ languages; the `cmn_Hani` train split is 370 shards of about 4.8 GB | parquet | first 2 train shards (9 GB) plus the complete test split (83 MB) |
| Ultra-FineWeb (Chinese) | `openbmb/Ultra-FineWeb` | 120B Chinese tokens | parquet | first 4 shards, 4.7 GB |
| SkyPile-150B | `Skywork/SkyPile-150B` | 150B tokens, 233M pages, 620 GB of plain text | jsonl | first 2 shards, 7.8 GB |
| Chinese Fineweb Edu (v1) | `opencsg/chinese-fineweb-edu` | Educational filter over CCI2, SkyPile, TeleChat-PTD, IndustryCorpus, MAP-CC | parquet | one or two shards from each of the five sources, 7.3 GB |
| Fineweb-Edu-Chinese V2.1 | `opencsg/Fineweb-Edu-Chinese-V2.1` | Score buckets 4-5 (70 GB, about 46B tokens), 3-4 (800 GB), 2-3 (1.4 TB) | parquet | first 1,141 files of the 4-5 bucket, 10 GB |
| ChineseWebText 2.0 | `CASIA-LM/ChineseWebText2.0` | Web text with quality, domain, and toxicity scores; 150 train parts of about 9 GB | jsonl.gz | test split only (relayed) |
| MAP-CC (encyclopedia) | `m-a-p/MAP-CC` | 800B tokens across web, books, papers, encyclopedia, other | jsonl.gz, 40 GB parts | the complete `zh_baike` part, 2.6 GB |
| FinePDFs (Chinese) | `HuggingFaceFW/finepdfs` | Chinese subset of FinePDFs | parquet | test split + first train shard, 4.5 GB |
| CCI3-HQ | `BAAI/CCI3-HQ` | BAAI's high-quality Chinese corpus, 672 parts of about 1 GB | jsonl | first 10 parts, 9.8 GB (gated on Hugging Face) |
| CCI2-Data | `BAAI/CCI2-Data` | 178 parquet shards of about 1.8 GB | parquet | first 2 shards, 3.6 GB (gated on Hugging Face) |
| IndustryCorpus2 | `BAAI/IndustryCorpus2` | Chinese and English text organized by industry and quality tier | parquet | one high-quality Chinese shard from each of the first industries, 4.9 GB (gated on Hugging Face) |
| TeleChat-PTD | `Tele-AI/TeleChat-PTD` | 189 shards of about 2.4 GB from TeleChat pretraining | jsonl.gz | first 2 shards, 4.9 GB (gated on Hugging Face) |
| Chinese Cosmopedia | `opencsg/chinese-cosmopedia` | Synthetic Chinese textbooks in the Cosmopedia style, 58 shards | parquet | first 4 shards, 4.5 GB |
| Wikipedia (Chinese) | `wikimedia/wikipedia` | 2023-11-01 dump | parquet | complete `20231101.zh`, 1.6 GB |

## 6. Small classroom sets

| Dataset | Repository | Reported size | Format | Mirrored sample |
| --- | --- | --- | --- | --- |
| TinyStories | `roneneldan/TinyStories` | Synthetic children's stories (v1 GPT-3.5/4, v2 GPT-4 only); the CS336 tokenizer assignment corpus | txt + parquet | v1 and v2 train/valid text plus parquet, 4.8 GB |
| CS336 OpenWebText sample | `stanford-cs336/owt-sample` | The OpenWebText subset used by Stanford CS336 Assignment 1 | txt.gz | complete, 4.4 GB (relayed) |
| WikiText | `Salesforce/wikitext` | WikiText-2 (2M tokens) and WikiText-103 (103M tokens), raw and tokenized | parquet | complete, 614 MB |

Start here for anything that must run on a laptop. TinyStories and the
OpenWebText sample are the two corpora the BPE-from-scratch exercise expects.

## 7. Gated or unmirrored datasets

Worth knowing, but not in the mirror. "Gated" means the Hugging Face
repository requires accepting terms with a logged-in account.

| Dataset | Repository | Why it matters | Status on the check date |
| --- | --- | --- | --- |
| The Stack v2 | `bigcode/the-stack-v2` | 67 TB of source code (StarCoder2); Hugging Face holds metadata, contents are served from Software Heritage | gated |
| Nemotron-CC v2 | `nvidia/Nemotron-CC-v2` | 6.3T-token English crawl with synthetic rephrasing (Nemotron Nano 2) | gated (manual approval) |
| Nemotron-Pretraining-Code v1 | `nvidia/Nemotron-Pretraining-Code-v1` | Curated code with metadata for a 747B-token corpus | gated (manual approval) |
| SlimPajama-627B | `cerebras/SlimPajama-627B` | Deduplicated RedPajama-1T, 627B tokens | returned 401 without login; ModelScope copies hold only the card |
| Dolma 3 mix | `allenai/dolma3_mix-6T-1025` | The 6T-token mixture actually used for OLMo 3 stage 1 | open, but no ModelScope mirror; the pool sample above stands in |
| WuDaoCorpora | (BAAI download portal) | 3 TB Chinese corpus, historically important | not on Hugging Face without login |
| Common Crawl itself | `data.commoncrawl.org` | The raw WARC/WET crawls every web corpus above starts from | open on the public internet; not reachable from the course preprocessing machine |

## 8. The course mirror

The course keeps one directory per entry under a shared data root, referred
to below as `$LLM26_DATA_ROOT`. Ask the instructor for the actual path; it
is not part of this public repository.

```text
$LLM26_DATA_ROOT/datasets/pretraining/
    fineweb/sample/10BT/000_00000.parquet ...
    fineweb/README.md                 # the dataset card as published
    fineweb/manifest.json             # source, repo, files, sizes, status, timestamp
    fineweb-edu/...
    ...
$LLM26_DATA_ROOT/tools/data/download_pretraining_datasets.py
$LLM26_DATA_ROOT/system/logs/download-*.log
```

Repository paths are preserved under each entry, so a path in a dataset
card or paper maps directly onto the mirror. `manifest.json` lists every
file with its byte size and whether it was downloaded or already present.

Entries marked "relayed" in the tables have no ModelScope mirror. They were
fetched from Hugging Face on a machine with CDN access and copied over; the
manifest records `huggingface (relayed)` as the source.

To recreate the mirror elsewhere:

```bash
# Everything the script can reach from ModelScope (about 208 GB)
uv run python scripts/download_pretraining_datasets.py --root "$LLM26_DATA_ROOT/datasets/pretraining"

# Everything, from Hugging Face, where huggingface.co and its CDN are reachable
uv run python scripts/download_pretraining_datasets.py --root "$LLM26_DATA_ROOT/datasets/pretraining" --source huggingface

# Only the laptop-sized sets
uv run python scripts/download_pretraining_datasets.py --root data/pretraining --group classroom
```

Budgets and sub-trees are declared in the `ENTRIES` table at the top of the
script. Raise a `max_bytes` value and re-run to extend a sample; files
already present are skipped.

## 9. Reading the files

Four container formats cover the whole catalog. `pyarrow` and `zstandard`
are not project dependencies; run one-off scripts with `uv run --with`.

```bash
uv run --with pyarrow --with zstandard python inspect_sample.py
```

```python
import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
import zstandard

root = Path("/path/to/pretraining")  # $LLM26_DATA_ROOT/datasets/pretraining

# parquet: FineWeb, FineWeb-Edu, FineWeb-2, Wikipedia, FineMath, ...
table = pq.read_table(root / "fineweb-edu/sample/10BT/000_00000.parquet",
                      columns=["text", "language_score", "token_count"])
print(table.num_rows, table.column_names)
print(table.column("text")[0].as_py()[:500])

# jsonl.zst: DCLM, Dolma 3, Proof-Pile-2, The Pile
with open(root / "proof-pile-2-arxiv/arxiv/train/arXiv_000.jsonl.zst", "rb") as fh:
    reader = zstandard.ZstdDecompressor().stream_reader(fh)
    for line in map(json.loads, __import__("io").TextIOWrapper(reader, encoding="utf-8")):
        print(line["text"][:500])
        break

# json.gz / jsonl.gz: C4, RedPajama-V2, TxT360, TeleChat-PTD, MAP-CC
with gzip.open(root / "c4-en/en/c4-validation.00000-of-00008.json.gz", "rt", encoding="utf-8") as fh:
    print(json.loads(next(fh))["text"][:500])

# plain text: TinyStories, the CS336 OpenWebText sample (after gunzip)
print((root / "tinystories/TinyStories-valid.txt").read_text(encoding="utf-8")[:500])
```

Do not decompress the multi-gigabyte archives in place on the shared store;
stream them as above, or extract into your own working directory.

## 10. Exercises that use these samples

1. **Bytes per token across corpora.** Encode the first 10 MB of FineWeb,
   FineWeb-2 Chinese, Python-Edu, and OpenWebMath with `gpt2`,
   `cl100k_base`, and one of the tokenizers inspected in the preprocessing
   notes. Report bytes per token and characters per token. Explain the
   Chinese and code rows using the pre-tokenization regexes.
2. **Pre-tokenizer spans on real documents.** Run the GPT-2 and the
   `\p{N}{1,3}` digit-grouping regexes over 1,000 FineMath documents. Count
   how often a number longer than three digits is split, and whether that
   changes the BPE merge statistics.
3. **What the quality signals encode.** For a RedPajama-V2 sample shard, join
   documents to their quality signals and plot document length against the
   fastText quality score and the duplicate flag. Compare with FineWeb-Edu's
   classifier score on documents of similar length.
4. **Train a BPE tokenizer on a mixture.** Sample 200 MB each from FineWeb,
   FineWeb-2 Chinese, and Python-Edu. Train a 32K byte-level BPE on the
   mixture and on each source alone. Measure compression on held-out shards
   from all three sources and on the FineWeb-2 Chinese test split.
5. **Corpus-preparation archaeology.** Read the Dolma v1.7 URL list and the
   RedPajama-1T URL lists. Reconstruct each corpus's source mixture from
   shard names alone, then check your estimate against the dataset card.

## 11. Licensing

The samples are mirrored for teaching. Check the card before redistributing
anything or training a model you intend to release.

| License | Datasets |
| --- | --- |
| ODC-By 1.0 | FineWeb, FineWeb-Edu, FineWeb-2, FinePDFs, Dolma, Dolma 3, TxT360, Zyda-2, Essential-Web, SmolLM corpus, C4, OpenWebMath, FineMath |
| CC-BY 4.0 | DCLM-baseline |
| Apache 2.0 | Ultra-FineWeb, Chinese Fineweb Edu, Fineweb-Edu-Chinese V2.1, ChineseWebText 2.0, Chinese Cosmopedia |
| CC0 1.0 | OpenWebText |
| CC-BY-SA 3.0 + GFDL | Wikipedia, WikiText |
| CDLA-Sharing 1.0 | TinyStories |
| CC-BY-NC-ND 4.0 (non-commercial, no derivatives) | MAP-CC |
| Vendor community licenses | SkyPile-150B (Skywork), CCI2/CCI3/IndustryCorpus2 (BAAI), TeleChat-PTD (TeleAI), StarCoderData (BigCode OpenRAIL), Nemotron sample (NVIDIA) |
| Mixed, see card | The Pile (uncopyrighted), RedPajama-V2 (Common Crawl terms of use), Proof-Pile-2 (per-subset) |

Underlying web text remains subject to its original authors' rights
regardless of the dataset license; ODC-By and similar licenses cover the
collection, not the pages.

## 12. Sources

Every repository id in the tables resolves as
`https://huggingface.co/datasets/<id>`; the same id usually works as
`https://www.modelscope.cn/datasets/<id>`. Papers and posts that describe
the corpora:

- FineWeb and FineWeb-Edu: Penedo et al., "The FineWeb Datasets", 2024. <https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1>
- FineWeb-2: Penedo et al., "FineWeb2: One Pipeline to Scale Them All", 2025. <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2>
- DCLM: Li et al., "DataComp-LM", 2024. <https://www.datacomp.ai/dclm/>
- Dolma and Dolma 3: Soldaini et al., "Dolma", 2024; the OLMo 3 report, 2025. <https://allenai.org/dolma>
- RedPajama-V2: Weber et al., "RedPajama", 2024. <https://www.together.ai/blog/redpajama-data-v2>
- TxT360: LLM360, 2024. <https://huggingface.co/spaces/LLM360/TxT360>
- Zyda-2: Zyphra, 2024. <https://www.zyphra.com/post/building-zyda-2>
- Essential-Web: Essential AI, 2025. <https://huggingface.co/datasets/EssentialAI/essential-web-v1.0>
- FinePDFs: Hugging Face, 2025. <https://huggingface.co/datasets/HuggingFaceFW/finepdfs>
- The Pile: Gao et al., 2020. <https://pile.eleuther.ai/>
- C4: Raffel et al., "Exploring the Limits of Transfer Learning", 2020. <https://www.tensorflow.org/datasets/catalog/c4>
- Proof-Pile-2 and OpenWebMath: Azerbayev et al., "Llemma", 2023; Paster et al., 2023.
- FineMath and SmolLM corpus: Hugging Face, 2024. <https://huggingface.co/datasets/HuggingFaceTB/finemath>
- StarCoderData: Li et al., "StarCoder", 2023. <https://huggingface.co/datasets/bigcode/starcoderdata>
- Ultra-FineWeb: OpenBMB, 2025. <https://huggingface.co/datasets/openbmb/Ultra-FineWeb>
- SkyPile-150B: Skywork technical report, 2023. <https://huggingface.co/datasets/Skywork/SkyPile-150B>
- Chinese Fineweb Edu: OpenCSG, 2024. <https://huggingface.co/datasets/opencsg/chinese-fineweb-edu>
- MAP-CC: "Chinese Tiny LLM", 2024. <https://huggingface.co/datasets/m-a-p/MAP-CC>
- CCI3-HQ and IndustryCorpus2: BAAI, 2024. <https://huggingface.co/datasets/BAAI/CCI3-HQ>
- TinyStories: Eldan and Li, 2023. <https://huggingface.co/datasets/roneneldan/TinyStories>
- Stanford CS336 Assignment 1 data: <https://huggingface.co/datasets/stanford-cs336/owt-sample>
