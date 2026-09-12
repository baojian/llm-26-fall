"""Mirror bounded samples of commonly used LLM pretraining corpora.

Companion to ``docs/lecture-01-pretraining-datasets.md``. Each entry below
names one dataset repository, the sub-tree to look at, and a byte budget. The
script lists that sub-tree, keeps files in name order until the budget is
reached, downloads them into ``<root>/<name>/``, and writes ``manifest.json``
next to them. Re-running skips files that already exist with the expected
size, so an interrupted run can simply be restarted.

Two sources are supported. ``huggingface`` uses ``huggingface_hub`` and works
wherever huggingface.co and its CDN are reachable. ``modelscope`` (default)
uses the public ModelScope REST API and ``curl``; it is the working option on
machines that can reach modelscope.cn but not the Hugging Face CDN. Entries
without a ModelScope mirror are skipped there and reported at the end.

Examples::

    uv run python scripts/download_pretraining_datasets.py --list
    uv run python scripts/download_pretraining_datasets.py --root data/pretraining --dry-run
    uv run python scripts/download_pretraining_datasets.py --root data/pretraining --only tinystories,wikitext
    uv run python scripts/download_pretraining_datasets.py --root data/pretraining --source huggingface

Repositories that are gated on Hugging Face (The Stack v2, Nemotron-CC,
CCI3-HQ, ...) are described in the lecture notes; the few that ModelScope
serves openly are included here with ``hf_gated=True`` and are skipped by the
Hugging Face source unless ``HF_TOKEN`` is set and the terms were accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

GB = 1024**3
MS_API = "https://www.modelscope.cn/api/v1/datasets"


@dataclass(frozen=True)
class Entry:
    name: str  # directory name under --root
    hf_repo: str  # Hugging Face dataset repository id
    path: str  # sub-tree to list ("" = repository root)
    max_bytes: int  # byte budget for data files (dataset card is extra)
    group: str  # category used in --list
    note: str  # one line for --list and the manifest
    pattern: str = r".*"  # regex on the path relative to the repo root
    round_robin: bool = False  # interleave first-level sub-directories
    ms_repo: str | None = None  # ModelScope mirror id (None = not mirrored)
    ms_path: str | None = None  # sub-tree on ModelScope when it differs
    hf_gated: bool = False  # needs accepted terms + token on Hugging Face


DCLM_PREFIX = (
    "filtered/OH_eli5_vs_rw_v2_bigram_200k_train/"
    "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/"
)

ENTRIES: list[Entry] = [
    # ---- English web-scale corpora -------------------------------------------
    Entry("fineweb", "HuggingFaceFW/fineweb", "sample/10BT", 32 * GB, "english-web",
          "FineWeb 10BT random sample (official subset, ~28 GB parquet)",
          ms_repo="swift/fineweb"),
    Entry("fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample/10BT", 32 * GB, "english-web",
          "FineWeb-Edu 10BT random sample (official subset)",
          ms_repo="HuggingFaceFW/fineweb-edu"),
    Entry("dclm-baseline", "mlfoundations/dclm-baseline-1.0-parquet",
          DCLM_PREFIX + "global-shard_01_of_10/local-shard_0_of_10", 10 * GB, "english-web",
          "DCLM-baseline, first local shard of global shard 1 (parquet re-upload)",
          ms_repo="mlfoundations/dclm-baseline-1.0-parquet"),
    Entry("dolma3", "allenai/dolma3", "data/common_crawl-education_and_jobs-0003", 10 * GB,
          "english-web", "Dolma 3 pool, one Common Crawl topic partition (jsonl.zst)"),
    Entry("redpajama-v2-sample", "togethercomputer/RedPajama-Data-V2", "sample", 30 * GB,
          "english-web", "RedPajama-V2 official sample with quality signals, minhash, duplicates",
          ms_repo="togethercomputer/RedPajama-Data-V2"),
    Entry("ultra-fineweb-en", "openbmb/Ultra-FineWeb", "data/ultrafineweb_en", 5 * GB,
          "english-web", "Ultra-FineWeb English (MiniCPM4 corpus), first shards",
          ms_repo="openbmb/Ultra-FineWeb"),
    Entry("zyda-2", "Zyphra/Zyda-2", "data/dclm_crossdeduped/global-shard_01_of_10", 5 * GB,
          "english-web", "Zyda-2 cross-deduplicated DCLM component, first shards",
          ms_repo="Zyphra/Zyda-2"),
    Entry("essential-web", "EssentialAI/essential-web-v1.0", "data/crawl=CC-MAIN-2013-20",
          5 * GB, "english-web", "Essential-Web v1.0 with taxonomy metadata, one crawl"),
    Entry("txt360-common-crawl", "LLM360/TxT360", "data/common-crawl", 5 * GB, "english-web",
          "TxT360 globally deduplicated Common Crawl, first shards"),
    Entry("c4-en", "allenai/c4", "en", 5 * GB, "english-web",
          "C4 English: 10 train shards + one validation shard (json.gz)",
          pattern=r"^en/c4-(train\.0000\d-of-01024|validation\.00000-of-00008)\.json\.gz$",
          ms_repo="allenai/c4"),
    Entry("openwebtext", "Skylion007/openwebtext", "plain_text", 15 * GB, "english-web",
          "OpenWebText, complete (GPT-2 style corpus, ~12 GB parquet)"),
    Entry("pile-uncopyrighted", "monology/pile-uncopyrighted", "", 20 * GB, "english-web",
          "The Pile (uncopyrighted subset): train shard 00, val, test",
          pattern=r"^(train/00\.jsonl\.zst|val\.jsonl\.zst|test\.jsonl\.zst)$"),
    # ---- multi-source, PDF, encyclopedic, synthetic --------------------------------
    Entry("finepdfs-en", "HuggingFaceFW/finepdfs", "data/eng_Latn", 5 * GB, "multi-source",
          "FinePDFs English (PDF-extracted text): test split + first train shards",
          ms_repo="HuggingFaceFW/finepdfs"),
    Entry("wikipedia-en", "wikimedia/wikipedia", "20231101.en", 25 * GB, "multi-source",
          "English Wikipedia 2023-11-01 dump, all 41 parquet shards",
          ms_repo="wikimedia/wikipedia"),
    Entry("wikipedia-zh", "wikimedia/wikipedia", "20231101.zh", 5 * GB, "multi-source",
          "Chinese Wikipedia 2023-11-01 dump, complete", ms_repo="wikimedia/wikipedia"),
    Entry("smollm-cosmopedia-v2", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 5 * GB,
          "multi-source", "Cosmopedia v2 synthetic textbooks (SmolLM corpus), first shards",
          ms_repo="HuggingFaceTB/smollm-corpus"),
    Entry("smollm-python-edu", "HuggingFaceTB/smollm-corpus", "python-edu", 5 * GB,
          "multi-source", "Python-Edu educational code (SmolLM corpus), complete",
          ms_repo="HuggingFaceTB/smollm-corpus"),
    Entry("nemotron-pretraining-sample", "nvidia/Nemotron-Pretraining-Dataset-sample", "",
          5 * GB, "multi-source", "Nemotron pretraining sample: CC, synthetic QA, math, code, SFT",
          pattern=r"\.parquet$"),
    Entry("dolma-v1-url-lists", "allenai/dolma", "urls", 1 * GB, "multi-source",
          "Dolma v1.5-v1.7 shard URL lists only (data hosted outside Hugging Face)",
          ms_repo="allenai/dolma"),
    Entry("redpajama-1t-url-lists", "togethercomputer/RedPajama-Data-1T", "urls", 1 * GB,
          "multi-source", "RedPajama-1T shard URL lists only",
          ms_repo="togethercomputer/RedPajama-Data-1T"),
    # ---- math and code ------------------------------------------------------------
    Entry("proof-pile-2-arxiv", "EleutherAI/proof-pile-2", "arxiv/train", 3 * GB, "math-code",
          "Proof-Pile-2 arXiv subset, first train shards", ms_repo="EleutherAI/proof-pile-2"),
    Entry("proof-pile-2-algebraic-stack", "EleutherAI/proof-pile-2", "algebraic-stack/train",
          3 * GB, "math-code", "Proof-Pile-2 AlgebraicStack (math code), first train shards",
          ms_repo="EleutherAI/proof-pile-2"),
    Entry("open-web-math", "open-web-math/open-web-math", "data", 5 * GB, "math-code",
          "OpenWebMath, first shards (ModelScope copy comes from Proof-Pile-2)",
          ms_repo="EleutherAI/proof-pile-2", ms_path="open-web-math/train"),
    Entry("finemath-4plus", "HuggingFaceTB/finemath", "finemath-4plus", 5 * GB, "math-code",
          "FineMath-4+ (9.6B tokens), first shards", ms_repo="HuggingFaceTB/finemath"),
    Entry("starcoderdata-python", "bigcode/starcoderdata", "python", 10 * GB, "math-code",
          "StarCoderData Python split (The Stack v1 derived), first shards",
          ms_repo="bigcode/starcoderdata", hf_gated=True),
    # ---- Chinese corpora ------------------------------------------------------------
    Entry("fineweb-2-zh-train", "HuggingFaceFW/fineweb-2", "data/cmn_Hani/train", 10 * GB,
          "chinese", "FineWeb-2 Chinese (cmn_Hani) train, first shards",
          ms_repo="HuggingFaceFW/fineweb-2"),
    Entry("fineweb-2-zh-test", "HuggingFaceFW/fineweb-2", "data/cmn_Hani/test", 5 * GB,
          "chinese", "FineWeb-2 Chinese held-out test split, complete",
          ms_repo="HuggingFaceFW/fineweb-2"),
    Entry("ultra-fineweb-zh", "openbmb/Ultra-FineWeb", "data/ultrafineweb_zh", 5 * GB,
          "chinese", "Ultra-FineWeb Chinese (120B tokens total), first shards",
          ms_repo="openbmb/Ultra-FineWeb"),
    Entry("skypile-150b", "Skywork/SkyPile-150B", "data", 10 * GB, "chinese",
          "SkyPile-150B Chinese web text, first jsonl shards",
          ms_repo="AI-ModelScope/SkyPile-150B"),
    Entry("chinese-fineweb-edu", "opencsg/chinese-fineweb-edu", "", 10 * GB, "chinese",
          "OpenCSG Chinese Fineweb Edu v1, interleaved across its five sources",
          pattern=r"\.parquet$", round_robin=True, ms_repo="opencsg/chinese-fineweb-edu"),
    Entry("fineweb-edu-chinese-v2.1-4_5", "opencsg/Fineweb-Edu-Chinese-V2.1", "4_5", 10 * GB,
          "chinese", "OpenCSG Fineweb-Edu-Chinese V2.1, top score bucket (4-5), first shards",
          ms_repo="opencsg/Fineweb-Edu-Chinese-V2.1"),
    Entry("chinesewebtext-2.0", "CASIA-LM/ChineseWebText2.0", "", 5 * GB, "chinese",
          "ChineseWebText 2.0 test split only (each train part is ~9 GB)",
          pattern=r"^test\.jsonl$"),
    Entry("map-cc-baike", "m-a-p/MAP-CC", "", 10 * GB, "chinese",
          "MAP-CC Chinese encyclopedia (zh_baike) split, complete single part",
          pattern=r"^zh_baike\.jsonl\.gz\.part01$", ms_repo="m-a-p/MAP-CC"),
    Entry("finepdfs-zh", "HuggingFaceFW/finepdfs", "data/cmn_Hani", 5 * GB, "chinese",
          "FinePDFs Chinese (PDF-extracted text): test split + first train shards",
          ms_repo="HuggingFaceFW/finepdfs"),
    Entry("cci3-hq", "BAAI/CCI3-HQ", "data", 10 * GB, "chinese",
          "BAAI CCI3-HQ high-quality Chinese web corpus, first jsonl shards",
          ms_repo="BAAI/CCI3-HQ", hf_gated=True),
    Entry("cci2-data", "BAAI/CCI2-Data", "data", 5 * GB, "chinese",
          "BAAI CCI2 Chinese corpus, first shards", ms_repo="BAAI/CCI2-Data", hf_gated=True),
    Entry("industrycorpus2", "BAAI/IndustryCorpus2", "", 5 * GB, "chinese",
          "BAAI IndustryCorpus2, interleaved across industry sub-directories",
          pattern=r"\.parquet$", round_robin=True, ms_repo="BAAI/IndustryCorpus2", hf_gated=True),
    Entry("telechat-ptd", "Tele-AI/TeleChat-PTD", "data", 5 * GB, "chinese",
          "TeleChat-PTD Chinese pretraining corpus, first shards",
          ms_repo="TeleAI/TeleChat-PTD", hf_gated=True),
    Entry("chinese-cosmopedia", "opencsg/chinese-cosmopedia", "data", 5 * GB, "chinese",
          "OpenCSG Chinese Cosmopedia synthetic textbooks, first shards",
          ms_repo="opencsg/chinese-cosmopedia"),
    # ---- small classroom sets ------------------------------------------------------
    Entry("tinystories", "roneneldan/TinyStories", "", 5 * GB, "classroom",
          "TinyStories v1 and v2 text + parquet (CS336 tokenizer exercise corpus)",
          pattern=r"^(TinyStories.*\.txt|data/.*\.parquet)$", ms_repo="roneneldan/TinyStories"),
    Entry("owt-sample-cs336", "stanford-cs336/owt-sample", "", 5 * GB, "classroom",
          "Stanford CS336 OpenWebText sample (owt_train/owt_valid)", pattern=r"\.gz$"),
    Entry("wikitext", "Salesforce/wikitext", "", 5 * GB, "classroom",
          "WikiText-2 and WikiText-103, raw and tokenized variants", pattern=r"\.parquet$",
          ms_repo="Salesforce/wikitext"),
]


def entry_map() -> dict[str, Entry]:
    return {e.name: e for e in ENTRIES}


# ----------------------------------------------------------------------------- sources
class HuggingFaceSource:
    name = "huggingface"

    def __init__(self):
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.endpoint = self.api.endpoint

    def repo(self, e: Entry):
        return e.hf_repo

    def available(self, e: Entry) -> bool:
        return not e.hf_gated or bool(os.environ.get("HF_TOKEN"))

    def list_files(self, e: Entry):
        tree = self.api.list_repo_tree(e.hf_repo, path_in_repo=e.path or None,
                                       repo_type="dataset", recursive=True)
        out = []
        for item in tree:
            size = getattr(item, "size", None)
            if size is not None:
                out.append((item.path, int(size)))
        return sorted(out)

    def fetch(self, e: Entry, rel_path: str, dest: Path):
        from huggingface_hub import hf_hub_download

        hf_hub_download(e.hf_repo, rel_path, repo_type="dataset", local_dir=str(dest))


class ModelScopeSource:
    name = "modelscope"
    endpoint = MS_API

    def repo(self, e: Entry):
        return e.ms_repo

    def available(self, e: Entry) -> bool:
        return e.ms_repo is not None

    @staticmethod
    def _get_json(url: str):
        req = urllib.request.Request(url, headers={"User-Agent": "llm-26-fall-datasets/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def list_files(self, e: Entry):
        root = e.path if e.ms_path is None else e.ms_path
        out, page, page_size = [], 1, 1000
        while True:
            url = (f"{MS_API}/{e.ms_repo}/repo/tree?Revision=master&Recursive=true"
                   f"&Root={urllib.parse.quote(root)}&PageSize={page_size}&PageNumber={page}")
            data = self._get_json(url).get("Data") or {}
            files = data.get("Files") or []
            for f in files:
                if f.get("Type") == "blob":
                    out.append((f["Path"], int(f.get("Size", 0))))
            total = data.get("TotalCount") or 0
            if not files or page * page_size >= total:
                break
            page += 1
        return sorted(out)

    def fetch(self, e: Entry, rel_path: str, dest: Path):
        url = (f"{MS_API}/{e.ms_repo}/repo?Revision=master"
               f"&FilePath={urllib.parse.quote(rel_path, safe='')}")
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        cmd = ["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "10",
               "--speed-time", "120", "--speed-limit", "10000", "-C", "-", "-o", str(part), url]
        subprocess.run(cmd, check=True)
        part.replace(target)


# ----------------------------------------------------------------------------- helpers
def select_files(files, e: Entry):
    """Apply the regex and byte budget; interleave sub-directories if asked."""
    rx = re.compile(e.pattern)
    cands = [(p, s) for p, s in files if rx.search(p)]
    if e.round_robin:
        buckets: dict[str, list] = {}
        for p, s in cands:
            buckets.setdefault(p.split("/")[0], []).append((p, s))
        ordered, queues = [], list(buckets.values())
        while any(queues):
            for q in queues:
                if q:
                    ordered.append(q.pop(0))
        cands = ordered
    chosen, total = [], 0
    for p, s in cands:
        if chosen and total + s > e.max_bytes:
            break
        chosen.append((p, s))
        total += s
    return chosen, total


def download_one(source, e: Entry, rel_path: str, size: int, dest: Path, retries: int = 4):
    target = dest / rel_path
    if target.exists() and (size == 0 or target.stat().st_size == size):
        return rel_path, size, "cached"
    last = None
    for attempt in range(1, retries + 1):
        try:
            source.fetch(e, rel_path, dest)
            got = target.stat().st_size
            if size and got != size:
                raise RuntimeError(f"size mismatch: expected {size}, got {got}")
            return rel_path, size, "downloaded"
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last = exc
            time.sleep(min(120, 15 * attempt))
    raise RuntimeError(f"{source.repo(e)}/{rel_path}: {last}")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="destination directory (one sub-directory per entry)")
    ap.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    ap.add_argument("--only", help="comma-separated entry names to process (default: all)")
    ap.add_argument("--group", help="comma-separated groups to process (see --list)")
    ap.add_argument("--list", action="store_true", help="print the catalog and exit")
    ap.add_argument("--dry-run", action="store_true", help="list files and sizes, download nothing")
    ap.add_argument("--workers", type=int, default=16, help="parallel file downloads")
    args = ap.parse_args(argv)

    if args.list:
        for e in ENTRIES:
            ms = e.ms_repo or "-"
            print(f"{e.group:12s} {e.name:30s} <= {human(e.max_bytes):8s} hf={e.hf_repo}"
                  f"{' (gated)' if e.hf_gated else ''}  ms={ms}\n{'':54s}{e.note}")
        return 0
    if args.root is None:
        ap.error("--root is required unless --list is given")

    selected = ENTRIES
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - set(entry_map())
        if unknown:
            ap.error(f"unknown entries: {sorted(unknown)}")
        selected = [e for e in ENTRIES if e.name in wanted]
    if args.group:
        groups = set(args.group.split(","))
        selected = [e for e in selected if e.group in groups]

    source = ModelScopeSource() if args.source == "modelscope" else HuggingFaceSource()
    if args.source == "modelscope" and shutil.which("curl") is None:
        ap.error("the modelscope source needs curl on PATH")
    print(f"source: {source.name} ({source.endpoint})")
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    grand_total, failures, skipped = 0, [], []

    for e in selected:
        if not source.available(e):
            skipped.append(e.name)
            print(f"[{e.name}] skipped: not available from {source.name}", flush=True)
            continue
        t0 = time.time()
        try:
            files = source.list_files(e)
        except Exception as exc:  # noqa: BLE001
            print(f"[{e.name}] LIST FAILED: {exc}", flush=True)
            failures.append(e.name)
            continue
        chosen, total = select_files(files, e)
        chosen.append(("README.md", 0))
        grand_total += total
        print(f"[{e.name}] {source.repo(e)}: {len(chosen)} files, {human(total)} "
              f"(listed {len(files)} files in {time.time() - t0:.0f}s)", flush=True)
        if args.dry_run:
            for p, s in chosen[:3]:
                print(f"    {human(s):>10s}  {p}")
            if len(chosen) > 3:
                print(f"    ... {chosen[-2][0]}")
            continue

        dest = root / e.name
        dest.mkdir(parents=True, exist_ok=True)
        results, errors = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(download_one, source, e, p, s, dest): p for p, s in chosen}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
                    print(f"    ERROR {exc}", flush=True)
        manifest = {
            "name": e.name,
            "source": source.name,
            "repo": source.repo(e),
            "hf_repo": e.hf_repo,
            "path": e.path if e.ms_path is None or source.name == "huggingface" else e.ms_path,
            "pattern": e.pattern,
            "max_bytes": e.max_bytes,
            "note": e.note,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "bytes": sum(s for _, s, _ in results),
            "files": sorted(({"path": p, "bytes": s, "status": st} for p, s, st in results),
                            key=lambda d: d["path"]),
            "errors": errors,
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        done = sum(1 for r in results if r[2] == "downloaded")
        cached = sum(1 for r in results if r[2] == "cached")
        print(f"    done: {done} downloaded, {cached} cached, {len(errors)} errors, "
              f"{time.time() - t0:.0f}s", flush=True)
        if errors:
            failures.append(e.name)

    print(f"\nselected data volume: {human(grand_total)} across "
          f"{len(selected) - len(skipped)} entries")
    if skipped:
        print(f"skipped (no {source.name} mirror or gated): {skipped}")
    if failures:
        print(f"entries with errors: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
