"""Compute the Lecture 02 numbers and figures from held-out shards.

Produces, for the deck and the teaching plan:
  1. bits per byte of interpolated 1/2/3-gram models on three held-out shards
     (TinyStories, OpenWebText, Chinese FineWeb-Edu) with the Qwen3 tokenizer;
  2. the reference-model filter histogram (Wikipedia 3-gram scoring web documents);
  3. the self-training loop: a bigram retrained on its own samples for five rounds.

    uv run python scripts/lecture02_experiments.py --data DIR --tokenizer tokenizer.json \
        --owt owt_shard0.parquet --out slides/lecture-02/assets

Writes ``lecture02-results.json`` plus ``held-out-bpb.svg``, ``lm-filter.svg``,
and ``self-training-loop.svg``. Standard library plus ``tokenizers`` and
``pyarrow`` (both in the project environment).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.eval import make_tokenizer  # noqa: E402
from pipeline.filters.lm_score import ReferenceScorer, histogram  # noqa: E402
from pipeline.ngram_lm import NGramLM, bits_per_byte, perplexity  # noqa: E402

MB = 1024 * 1024


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
        test_tokens = sum(len(x) for x in test_ids)
        rows = {}
        for order in orders:
            t0 = time.time()
            lm = NGramLM(order=order, vocab_size=vocab_size).fit(train_ids)
            lm.tune(dev_ids)
            rows[order] = dict(bits_per_byte=round(bits_per_byte(lm, test_ids, test_bytes), 3),
                               token_perplexity=round(perplexity(lm, test_ids), 1),
                               weights=[round(w, 3) for w in lm.weights], seconds=round(time.time() - t0, 1))
            print(f"  {name:12s} order {order}: {rows[order]}", flush=True)
        table[name] = dict(label=src["label"], train_docs=len(src["train"]),
                           train_tokens=sum(len(x) for x in train_ids), test_docs=len(src["test"]),
                           test_bytes=test_bytes, test_tokens=test_tokens,
                           bytes_per_token=round(test_bytes / test_tokens, 2), orders=rows)
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
    reference = "\n\n".join(cap_bytes(docs, ref_cap))
    scorer = ReferenceScorer.from_text(reference, encode, vocab_size, order=3)
    web = sources["OpenWebText"]["test"]
    scores = scorer.score_many(web)
    stories = scorer.score_many(sources["TinyStories"]["test"][:300])
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranked = lambda i: dict(bits_per_byte=round(scores[i], 3), snippet=web[i][:160].replace("\n", " "))
    cut = sorted(scores)[int(len(scores) * 2 / 3)]  # CCNet keeps head and middle thirds
    return dict(reference_docs=len(docs), reference_bytes=len(reference.encode()), web_docs=len(web),
                cut_bits_per_byte=round(cut, 3), web_median=round(sorted(scores)[len(scores) // 2], 3),
                stories_median=round(sorted(stories)[len(stories) // 2], 3),
                examples=dict(lowest=ranked(order[0]), median=ranked(order[len(order) // 2]), highest=ranked(order[-1])),
                histogram=[(round(l, 3), c) for l, c in histogram(scores, bins=30)],
                stories_histogram=[(round(l, 3), c) for l, c in histogram(stories, bins=30)])


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
                           distinct_tokens=types, sample=sample_text[:160]))
        print(f"  round {r}: bpb={bpb:.3f} distinct={types} sample={sample_text[:60]!r}", flush=True)
        corpus = [lm.sample(rng, max_tokens=128) for _ in range(n_samples)]
    return dict(rounds=series, n_samples=n_samples)


# ------------------------------------------------------------------------- svg
def svg_bars(groups: list[tuple[str, list[float]]], series_names: list[str], ylabel: str, title: str,
             width=1152, height=560) -> str:
    left, right, top, bottom = 90, 30, 60, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = max(v for _, vals in groups for v in vals) * 1.15
    colors = ["#8ab4f8", "#4285f4", "#1a56c4"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-size="26" font-weight="bold">{escape(title)}</text>']
    for i in range(6):
        y = top + plot_h - i * plot_h / 5
        v = ymax * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-8}" y="{y+6:.1f}" text-anchor="end" font-size="18">{v:.1f}</text>')
    gw = plot_w / len(groups)
    bw = gw * 0.7 / len(series_names)
    for g, (name, vals) in enumerate(groups):
        x0 = left + g * gw + gw * 0.15
        for s, v in enumerate(vals):
            h = v / ymax * plot_h
            x = x0 + s * bw
            parts.append(f'<rect x="{x:.1f}" y="{top+plot_h-h:.1f}" width="{bw-4:.1f}" height="{h:.1f}" fill="{colors[s % 3]}"/>')
            parts.append(f'<text x="{x+bw/2-2:.1f}" y="{top+plot_h-h-6:.1f}" text-anchor="middle" font-size="17">{v:.2f}</text>')
        parts.append(f'<text x="{x0+gw*0.35:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-size="20">{escape(name)}</text>')
    for s, sname in enumerate(series_names):
        x = left + s * 160
        parts.append(f'<rect x="{x}" y="{height-34}" width="18" height="18" fill="{colors[s % 3]}"/>')
        parts.append(f'<text x="{x+24}" y="{height-19}" font-size="18">{escape(sname)}</text>')
    parts.append(f'<text x="24" y="{top+plot_h/2}" transform="rotate(-90 24 {top+plot_h/2})" text-anchor="middle" font-size="20">{escape(ylabel)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_histogram(hist: list, cut: float, title: str, xlabel: str, second: list | None = None,
                  labels=("web documents", "TinyStories"), width=1152, height=560) -> str:
    left, right, top, bottom = 80, 30, 60, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    all_bins = hist + (second or [])
    xmin = min(l for l, _ in all_bins)
    xmax = max(l for l, _ in all_bins) + (hist[1][0] - hist[0][0])
    ymax = max(c for _, c in all_bins) * 1.1
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * plot_w
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-size="26" font-weight="bold">{escape(title)}</text>']
    for series, color, opacity in ((hist, "#4285f4", 0.85), (second or [], "#f4a142", 0.7)):
        if not series:
            continue
        bw = (series[1][0] - series[0][0])
        for l, c in series:
            h = c / ymax * plot_h
            parts.append(f'<rect x="{sx(l):.1f}" y="{top+plot_h-h:.1f}" width="{sx(l+bw)-sx(l)-1:.1f}" height="{h:.1f}" fill="{color}" opacity="{opacity}"/>')
    parts.append(f'<line x1="{sx(cut):.1f}" y1="{top}" x2="{sx(cut):.1f}" y2="{top+plot_h}" stroke="#c62828" stroke-width="3" stroke-dasharray="8 6"/>')
    parts.append(f'<text x="{sx(cut)+8:.1f}" y="{top+22}" font-size="19" fill="#c62828">cut at {cut:.2f} (drop the worst third)</text>')
    for i in range(6):
        x = xmin + (xmax - xmin) * i / 5
        parts.append(f'<text x="{sx(x):.1f}" y="{top+plot_h+26}" text-anchor="middle" font-size="18">{x:.1f}</text>')
    parts.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="20">{escape(xlabel)}</text>')
    parts.append(f'<text x="24" y="{top+plot_h/2}" transform="rotate(-90 24 {top+plot_h/2})" text-anchor="middle" font-size="20">documents</text>')
    for i, (lab, color) in enumerate(zip(labels, ("#4285f4", "#f4a142"))):
        parts.append(f'<rect x="{width-330}" y="{top+10+i*28}" width="18" height="18" fill="{color}"/>')
        parts.append(f'<text x="{width-306}" y="{top+25+i*28}" font-size="18">{escape(lab)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_line(points: list[tuple[int, float]], title: str, xlabel: str, ylabel: str, width=1152, height=560) -> str:
    left, right, top, bottom = 90, 30, 60, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    ys = [y for _, y in points]
    ymin, ymax = min(ys) * 0.9, max(ys) * 1.1
    sx = lambda x: left + x / (len(points) - 1) * plot_w
    sy = lambda y: top + plot_h - (y - ymin) / (ymax - ymin) * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-size="26" font-weight="bold">{escape(title)}</text>']
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        parts.append(f'<line x1="{left}" y1="{sy(y):.1f}" x2="{width-right}" y2="{sy(y):.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-8}" y="{sy(y)+6:.1f}" text-anchor="end" font-size="18">{y:.2f}</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
    parts.append(f'<path d="{path}" fill="none" stroke="#c62828" stroke-width="4"/>')
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="#c62828"/>')
        parts.append(f'<text x="{sx(x):.1f}" y="{sy(y)-14:.1f}" text-anchor="middle" font-size="18">{y:.2f}</text>')
        parts.append(f'<text x="{sx(x):.1f}" y="{top+plot_h+26}" text-anchor="middle" font-size="18">{x}</text>')
    parts.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="20">{escape(xlabel)}</text>')
    parts.append(f'<text x="24" y="{top+plot_h/2}" transform="rotate(-90 24 {top+plot_h/2})" text-anchor="middle" font-size="20">{escape(ylabel)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


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
    (args.out / "lecture02-results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    groups = [(name, [results["held_out"][name]["orders"][o]["bits_per_byte"] for o in (1, 2, 3)])
              for name in results["held_out"]]
    (args.out / "held-out-bpb.svg").write_text(svg_bars(
        groups, ["unigram", "bigram", "trigram"], "bits per byte (lower is better)",
        f"Interpolated n-gram models, Qwen3 tokens, {args.train_mb} MB of training text per source"))
    f = results["filter"]
    (args.out / "lm-filter.svg").write_text(svg_histogram(
        f["histogram"], f["cut_bits_per_byte"], "Web documents scored by a Wikipedia 3-gram model",
        "bits per byte under the reference model (lower = closer to Wikipedia)", f["stories_histogram"]))
    pts = [(r["round"], r["bits_per_byte"]) for r in results["self_training"]["rounds"]]
    (args.out / "self-training-loop.svg").write_text(svg_line(
        pts, "A bigram retrained on its own samples", "round (0 = trained on real TinyStories)",
        "held-out bits per byte"))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
