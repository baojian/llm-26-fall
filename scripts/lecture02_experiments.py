"""Compute the Lecture 02 numbers and figures from held-out shards.

Produces, for the deck and the teaching plan:
  1. bits per byte of interpolated 1/2/3-gram models on three held-out shards
     (TinyStories, OpenWebText, Chinese FineWeb-Edu) with the Qwen3 tokenizer;
  2. the reference-model filter histogram (Wikipedia 3-gram scoring web documents);
  3. the self-training loop: a bigram retrained on its own samples for five rounds.

    uv run python scripts/lecture02_experiments.py --data DIR --tokenizer tokenizer.json \
        --owt owt_shard0.parquet --out slides/lecture-02/assets

Writes ``lecture02-results.json`` plus Plotly chart specifications
``lm-filter.json`` and ``self-training-loop.json``. Install the tokenization
extra for ``tokenizers`` and ``pyarrow``; the notebook itself is standard-library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.eval import make_tokenizer  # noqa: E402
from pipeline.filters.lm_score import ReferenceScorer, histogram, reference_split  # noqa: E402
from pipeline.ngram_lm import NGramLM, bits_per_byte, perplexity  # noqa: E402

MB = 1024 * 1024


def document_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", " ".join(text.split())).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(file=path.name, bytes=path.stat().st_size, sha256=digest)


def split_manifest(source: dict) -> dict:
    """Ordered document fingerprints and normalized overlap counts for the supplied slices."""
    out, keys = {}, {}
    for name in ("train", "dev", "test"):
        docs = source[name]
        hashes = [hashlib.sha256(d.encode("utf-8")).hexdigest() for d in docs]
        keys[name] = {document_key(d) for d in docs}
        out[name] = dict(documents=len(docs), utf8_bytes=sum(len(d.encode("utf-8")) for d in docs),
                         ordered_document_hashes_sha256=hashlib.sha256("\n".join(hashes).encode()).hexdigest())
    out["normalized_cross_split_duplicates"] = {
        f"{a}/{b}": len(keys[a] & keys[b]) for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    return out


# ----------------------------------------------------------------------------- data
def parquet_texts(path: Path, column: str = "text", limit: int | None = None) -> list[str]:
    import pyarrow.parquet as pq

    out = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=[column]):
        out.extend(t for t in batch.column(0).to_pylist() if t)
        if limit and len(out) >= limit:
            return out[:limit]
    return out


def cap_bytes(docs: list[str], limit: int) -> list[str]:
    out, total = [], 0
    for d in docs:
        out.append(d)
        total += len(d.encode("utf-8"))
        if total >= limit:
            break
    return out


def tinystories(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [s.strip() for s in text.split("<|endoftext|>") if s.strip()]


def load_sources(data: Path, owt: Path, train_cap: int) -> dict:
    ts_train = tinystories(data / "tinystories-train-head.txt")
    ts_valid = tinystories(data / "tinystories-valid.txt")
    owt_docs = parquet_texts(owt)
    zh_train = [d for i in range(10) for d in parquet_texts(data / f"zh-0000{i:02d}.parquet")]
    zh_dev = parquet_texts(data / "zh-000010.parquet")
    zh_test = parquet_texts(data / "zh-000011.parquet")
    return {
        "TinyStories": dict(train=cap_bytes(ts_train[:-1], train_cap), dev=ts_valid[:400], test=ts_valid[400:2400],
                            label="TinyStories v2 (stories)"),
        "OpenWebText": dict(train=cap_bytes(owt_docs[:-1500], train_cap), dev=owt_docs[-1500:-1200], test=owt_docs[-1200:],
                            label="OpenWebText (English web)"),
        "Chinese": dict(train=cap_bytes(zh_train, train_cap), dev=zh_dev[:300], test=zh_test[:1200],
                        label="Fineweb-Edu-Chinese V2.1 (Chinese web)"),
    }


# ------------------------------------------------------------------------- part 1
def held_out_table(sources: dict, encode, vocab_size: int, orders=(1, 2, 3)) -> dict:
    table = {}
    for name, src in sources.items():
        train_ids = [encode(d) for d in src["train"]]
        dev_ids = [encode(d) for d in src["dev"]]
        test_ids = [encode(d) for d in src["test"]]
        test_bytes = sum(len(d.encode("utf-8")) for d in src["test"])
        content_tokens = sum(len(x) for x in test_ids)
        test_tokens = content_tokens + len(test_ids)  # one scored EOS per document
        rows = {}
        for order in orders:
            t0 = time.time()
            lm = NGramLM(order=order, vocab_size=vocab_size).fit(train_ids)
            lm.tune(dev_ids)
            bpb = bits_per_byte(lm, test_ids, test_bytes)
            rows[order] = dict(bits_per_byte=round(bpb, 3),
                               loss_nats_per_token=round(bpb * test_bytes * math.log(2) / test_tokens, 6),
                               token_perplexity=round(perplexity(lm, test_ids), 1),
                               weights=[round(w, 3) for w in lm.weights], seconds=round(time.time() - t0, 1))
            print(f"  {name:12s} order {order}: {rows[order]}", flush=True)
        table[name] = dict(label=src["label"], train_docs=len(src["train"]),
                           train_tokens=sum(len(x) for x in train_ids), test_docs=len(src["test"]),
                           test_bytes=test_bytes, test_tokens=test_tokens, test_content_tokens=content_tokens,
                           bytes_per_token=round(test_bytes / test_tokens, 2),
                           bytes_per_content_token=round(test_bytes / content_tokens, 2),
                           split_manifest=split_manifest(src), orders=rows)
    return table


# ------------------------------------------------------------------------- part 2
def filter_histogram(data: Path, sources: dict, encode, vocab_size: int, ref_cap: int) -> dict:
    wiki = parquet_texts(data / "wikitext103-train0.parquet")
    # wikitext rows are lines; glue into documents on " = Title = " headings
    docs, cur = [], []
    for line in wiki:
        if line.startswith(" = ") and not line.startswith(" = = ") and cur:
            docs.append("".join(cur))
            cur = []
        cur.append(line)
    docs.append("".join(cur))
    selected = [d.strip() for d in cap_bytes(docs, ref_cap) if d.strip()]
    split = reference_split(len(selected), 0.05)
    scorer = ReferenceScorer.from_documents(selected, encode, vocab_size, order=3)
    web = sources["OpenWebText"]["test"]
    scores = scorer.score_many(web)
    stories = scorer.score_many(sources["TinyStories"]["test"][:300])
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranked = lambda i: dict(bits_per_byte=round(scores[i], 3), snippet=web[i][:160].replace("\n", " "))
    cut = sorted(scores)[int(len(scores) * 2 / 3)]  # illustrative course policy, not a CCNet default
    bounds = (min(scores + stories), max(scores + stories))
    return dict(reference_loaded_docs=len(docs), reference_selected_docs=len(selected),
                reference_train_docs=split, reference_dev_docs=len(selected) - split,
                reference_bytes=sum(len(d.encode("utf-8")) for d in selected), web_docs=len(web),
                stories_docs=len(stories), reference_split="last 5% of selected whole articles for development",
                histogram_bounds=list(bounds), histogram_bins=30,
                cut_bits_per_byte=round(cut, 3), web_median=round(sorted(scores)[len(scores) // 2], 3),
                stories_median=round(sorted(stories)[len(stories) // 2], 3),
                examples=dict(lowest=ranked(order[0]), median=ranked(order[len(order) // 2]), highest=ranked(order[-1])),
                histogram=histogram(scores, bins=30, bounds=bounds),
                stories_histogram=histogram(stories, bins=30, bounds=bounds))


# ------------------------------------------------------------------------- part 3
def self_training_loop(sources: dict, encode, decode, vocab_size: int, rounds: int, n_samples: int) -> dict:
    src = sources["TinyStories"]
    test_ids = [encode(d) for d in src["test"]]
    test_bytes = sum(len(d.encode("utf-8")) for d in src["test"])
    corpus = [encode(d) for d in src["train"]]
    dev = [encode(d) for d in src["dev"]]
    rng = random.Random(2026)
    series = []
    for r in range(rounds + 1):
        lm = NGramLM(order=2, vocab_size=vocab_size).fit(corpus)
        lm.tune(dev)
        bpb = bits_per_byte(lm, test_ids, test_bytes)
        types = len({t for ids in corpus for t in ids})
        sample_text = decode(lm.sample(rng, max_tokens=60))
        series.append(dict(round=r, bits_per_byte=round(bpb, 3), corpus_docs=len(corpus),
                           corpus_tokens=sum(len(ids) for ids in corpus),
                           distinct_tokens=types, sample=sample_text[:160]))
        print(f"  round {r}: bpb={bpb:.3f} distinct={types} sample={sample_text[:60]!r}", flush=True)
        corpus = [lm.sample(rng, max_tokens=128) for _ in range(n_samples)]
    return dict(rounds=series, n_samples=n_samples, seed=2026, max_sample_tokens=128,
                design="replace all training documents with capped model samples; keep original dev/test",
                limitation="training size and length distribution change; no filtering or equal-budget control")


def plotly_figures(results: dict) -> dict:
    """Shared-template charts with readable labels and identical histogram bins."""
    f = results["filter"]
    width = (f["histogram_bounds"][1] - f["histogram_bounds"][0]) / f["histogram_bins"]
    traces = []
    for key, n, name, color in (("histogram", f["web_docs"], "Web", "#4285f4"),
                                ("stories_histogram", f["stories_docs"], "TinyStories", "#e88925")):
        traces.append(dict(type="bar", x=[left + width / 2 for left, _ in f[key]],
                           y=[100 * count / n for _, count in f[key]], width=width * 0.95,
                           name=f"{name} (n={n:,})", marker=dict(color=color), opacity=0.65,
                           hovertemplate="BPB: %{x:.2f}<br>Documents: %{y:.1f}%<extra>%{fullData.name}</extra>"))
    cut = f["cut_bits_per_byte"]
    histogram_plot = dict(data=traces, layout=dict(
        barmode="overlay", margin=dict(l=85, r=25, t=55, b=70),
        hoverlabel=dict(font=dict(size=24)),
        xaxis=dict(title=dict(text="Reference-model bits per byte")),
        yaxis=dict(title=dict(text="Documents (%)"), rangemode="tozero"),
        legend=dict(orientation="h", x=0, y=1.2),
        shapes=[dict(type="line", x0=cut, x1=cut, y0=0, y1=1, yref="paper",
                     line=dict(color="#c62828", width=3, dash="dash"))],
        annotations=[dict(x=cut, y=0.95, yref="paper", text=f"Cut {cut:.2f}",
                          xanchor="left", xshift=8, showarrow=False, font=dict(size=24, color="#c62828"))]))
    rounds = results["self_training"]["rounds"]
    loop_plot = dict(data=[dict(type="scatter", mode="lines+markers",
        x=[r["round"] for r in rounds], y=[r["bits_per_byte"] for r in rounds],
        line=dict(color="#c62828", width=4), marker=dict(size=12),
        customdata=[r["corpus_tokens"] for r in rounds],
        hovertemplate="Round %{x}<br>BPB: %{y:.3f}<br>Training tokens: %{customdata:,}<extra></extra>")],
        layout=dict(showlegend=False, margin=dict(l=90, r=25, t=15, b=70),
                    hoverlabel=dict(font=dict(size=24)),
                    xaxis=dict(title=dict(text="Round (0 = original corpus)"), dtick=1),
                    yaxis=dict(title=dict(text="Held-out bits per byte"))))
    return {"lm-filter.json": histogram_plot, "self-training-loop.json": loop_plot}


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--owt", type=Path, required=True, help="one OpenWebText parquet shard")
    ap.add_argument("--tokenizer", required=True, help="Qwen3 tokenizer.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-mb", type=int, default=24)
    ap.add_argument("--ref-mb", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args(argv)

    encode, vocab_size = make_tokenizer(args.tokenizer)
    from tokenizers import Tokenizer as HFTokenizer

    tok = HFTokenizer.from_file(args.tokenizer)
    decode = lambda ids: tok.decode([i for i in ids if i >= 0])
    print("loading sources", flush=True)
    sources = load_sources(args.data, args.owt, args.train_mb * MB)
    results = dict(tokenizer="Qwen/Qwen3-0.6B tokenizer.json", vocab_size=vocab_size, train_cap_mb=args.train_mb)

    print("part 1: held-out bits per byte", flush=True)
    results["held_out"] = held_out_table(sources, encode, vocab_size)
    print("part 2: reference-model filter", flush=True)
    results["filter"] = filter_histogram(args.data, sources, encode, vocab_size, args.ref_mb * MB)
    print("part 3: self-training loop", flush=True)
    results["self_training"] = self_training_loop(sources, encode, decode, vocab_size, args.rounds, args.samples)

    args.out.mkdir(parents=True, exist_ok=True)
    inputs = [args.tokenizer, args.owt, *sorted(args.data.glob("*.parquet")),
              args.data / "tinystories-train-head.txt", args.data / "tinystories-valid.txt"]
    results["provenance"] = dict(
        inputs=[file_fingerprint(Path(p)) for p in inputs],
        token_count_policy="test_tokens includes EOS; test_content_tokens and train_tokens exclude EOS",
        byte_unit="MiB = 1048576 bytes; cap includes the final complete document",
        note="Input hashes identify the local September 15 snapshot; upstream commit revisions were not recorded.")
    (args.out / "lecture02-results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, figure in plotly_figures(results).items():
        (args.out / name).write_text(json.dumps(figure, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
