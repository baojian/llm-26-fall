"""A1 Part 4: perplexity and bits per byte of every model on the held-out files.

    python a1/evaluate.py data/          # writes results.json next to the a1/ package

This file is mostly scaffolding that calls your Parts 1–3; the only piece you
write is ``score``. The layout of ``results.json`` is fixed by ``SCHEMA`` below
and by the code in ``evaluate_source``; the graders read exactly these keys.

Byte counts: ``bits_per_byte`` divides by the UTF-8 length of the documents
that are actually modelled, i.e. each document stripped of surrounding
whitespace, summed over documents; the blank-line separators are not counted.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a1.bpe import decode, encode, train_bpe  # noqa: E402
from a1.ngram import NGramLM  # noqa: E402
from a1.nplm import NeuralNGramLM, log2_prob as nplm_log2_prob, nearest_neighbours, parameter_count  # noqa: E402
from a1.nplm import train as train_nplm  # noqa: E402

VOCAB_SIZE = 1000
ORDERS = (1, 2, 3)
DELTAS = (0.01, 0.1, 0.5, 1.0)
NPLM = dict(context=3, dim=64, hidden=64, epochs=3)  # Part 3 runs on TinyStories only
SOURCES = {
    "tinystories": ("tinystories-train.txt", "tinystories-dev.txt", "tinystories-test.txt"),
    "chinese": ("chinese-web-train.txt", "chinese-web-dev.txt", "chinese-web-test.txt"),
}

SCHEMA = {  # the exact shape of results.json (values are examples)
    "tinystories": {
        "vocab_size": 1000,
        "bytes_per_token": 3.14,  # test bytes / test tokens under your tokenizer
        "orders": {
            "1": {  # one entry per order in ORDERS, keyed by the order as a string
                "mle": {"perplexity": 304.3, "bits_per_byte": 2.636},
                "add": {"delta": 0.5, "perplexity": 304.2, "bits_per_byte": 2.636},
                "interp": {"weights": [0.001, 0.999], "perplexity": 304.3, "bits_per_byte": 2.636},
            },
        },
        "nplm": {  # TinyStories only
            "context": 3, "dim": 64, "hidden": 64,
            "epochs": [{"epoch": 1, "train_loss": 3.83, "dev_loss": 3.38}],
            "parameters": 0, "parameters_tied": 0,
            "perplexity": 24.2, "bits_per_byte": 1.469,
            "neighbours": {" happy": [[" angry", 0.45], [" sad", 0.39]]},
        },
    },
    "chinese": {"vocab_size": 1000, "bytes_per_token": 2.48, "orders": {"...": "as above"}},
}


def read_docs(path: Path) -> list[str]:
    return [d.strip() for d in path.read_text(encoding="utf-8").split("\n\n") if d.strip()]


def byte_length(docs: list[str]) -> int:
    return sum(len(d.encode("utf-8")) for d in docs)


def score(lm: NGramLM, sequences: list[list[int]], test_bytes: int, kind: str, delta: float = 1.0) -> dict:
    """Token perplexity and bits per byte of ``sequences`` under ``lm.log2_prob(..., kind, delta)``.

    perplexity = 2 ** (-(total log2 probability) / (number of predicted tokens, EOS included))
    bits_per_byte = -(total log2 probability) / test_bytes
    Round to 2 and 4 decimals respectively.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def evaluate_source(data: Path, train_file: str, dev_file: str | None, test_file: str) -> dict:
    """Train the tokenizer and every model on train, tune on dev, report on test."""
    train_docs = read_docs(data / train_file)
    test_docs = read_docs(data / test_file)
    if dev_file is None:  # carve a dev split off the end of train
        cut = int(len(train_docs) * 0.8)
        train_docs, dev_docs = train_docs[:cut], train_docs[cut:]
    else:
        dev_docs = read_docs(data / dev_file)

    print(f"{train_file}: training {VOCAB_SIZE}-token BPE...", flush=True)
    vocab, merges = train_bpe("\n\n".join(train_docs), VOCAB_SIZE)
    print(f"{train_file}: encoding train/dev/test...", flush=True)
    enc = lambda d: encode(d, vocab, merges)  # noqa: E731
    train_ids = [enc(d) for d in train_docs]
    dev_ids = [enc(d) for d in dev_docs]
    test_ids = [enc(d) for d in test_docs]
    assert all(decode(enc(d), vocab) == d for d in test_docs[:5]), "decode(encode(x)) must equal x"
    test_bytes, dev_bytes = byte_length(test_docs), byte_length(dev_docs)
    result: dict = {
        "vocab_size": VOCAB_SIZE,
        "bytes_per_token": round(test_bytes / sum(len(x) for x in test_ids), 3),
        "orders": {},
    }
    for order in ORDERS:
        print(f"{train_file}: fitting and evaluating order {order}...", flush=True)
        lm = NGramLM(order, VOCAB_SIZE).fit(train_ids)
        best_delta = min(DELTAS, key=lambda d: score(lm, dev_ids, dev_bytes, "add", d)["bits_per_byte"])
        weights = lm.tune(dev_ids)  # dev only, never test
        result["orders"][str(order)] = {
            "mle": score(lm, test_ids, test_bytes, "mle"),
            "add": {"delta": best_delta, **score(lm, test_ids, test_bytes, "add", best_delta)},
            "interp": {"weights": [round(w, 4) for w in weights], **score(lm, test_ids, test_bytes, "interp")},
        }

    if train_file.startswith("tinystories"):
        print(f"{train_file}: training and evaluating the neural model...", flush=True)
        torch.manual_seed(0)  # seed before construction so the initial weights are reproducible
        model = NeuralNGramLM(VOCAB_SIZE, NPLM["context"], NPLM["dim"], NPLM["hidden"])
        history = train_nplm(model, train_ids, dev_ids, epochs=NPLM["epochs"], seed=0)
        log2p, n = nplm_log2_prob(model, test_ids)
        tied = NeuralNGramLM(VOCAB_SIZE, NPLM["context"], NPLM["dim"], NPLM["hidden"], tie_output=True)
        frequent = [t for t, _ in __import__("collections").Counter(t for ids in train_ids for t in ids).most_common(40)][-5:]
        result["nplm"] = {
            "context": NPLM["context"], "dim": NPLM["dim"], "hidden": NPLM["hidden"], "epochs": history,
            "parameters": parameter_count(model), "parameters_tied": parameter_count(tied),
            "perplexity": round(2 ** (-log2p / n), 2), "bits_per_byte": round(-log2p / test_bytes, 4),
            "neighbours": {decode([t], vocab): [(decode([i], vocab), sim) for i, sim in nearest_neighbours(model, t)]
                           for t in frequent},
        }
    return result


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    data = Path(args[0]) if args else Path(__file__).resolve().parents[1] / "data"
    results = {name: evaluate_source(data, *files) for name, files in SOURCES.items()}
    out = Path(__file__).resolve().parents[1] / "results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, r in results.items():
        for order, row in r["orders"].items():
            print(f"{name:12s} order {order}: mle {row['mle']['bits_per_byte']:.3f}  add {row['add']['bits_per_byte']:.3f}  "
                  f"interp {row['interp']['bits_per_byte']:.3f} bits/byte")
        if "nplm" in r:
            print(f"{name:12s} nplm   : {r['nplm']['bits_per_byte']:.3f} bits/byte, {r['nplm']['parameters']:,} parameters "
                  f"({r['nplm']['parameters_tied']:,} tied), dev loss {[e['dev_loss'] for e in r['nplm']['epochs']]}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
