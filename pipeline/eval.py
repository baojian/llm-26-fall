"""Bits per byte on fixed held-out shards: the one evaluation every model in the course reports.

    uv run python -m pipeline.eval --train train.txt --dev dev.txt --test test.txt --order 3
    uv run python -m pipeline.eval ... --tokenizer path/to/tokenizer.json   # Qwen3 or any HF tokenizer.json

Without ``--tokenizer`` the text is split into UTF-8 bytes, which needs no
extra package. Documents are separated by blank lines; each document is one
sequence with its own BOS and EOS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from pipeline.ngram_lm import NGramLM, bits_per_byte, perplexity

Tokenizer = Callable[[str], list[int]]


def make_tokenizer(path: str | None) -> tuple[Tokenizer, int]:
    """Return (encode, vocab_size). ``path`` is an HF ``tokenizer.json``; None = bytes."""
    if path is None:
        return (lambda text: list(text.encode("utf-8"))), 256
    from tokenizers import Tokenizer as HFTokenizer  # optional dependency

    tok = HFTokenizer.from_file(path)
    return (lambda text: tok.encode(text, add_special_tokens=False).ids), tok.get_vocab_size()


def read_documents(path: Path, limit_bytes: int | None = None) -> list[str]:
    """Split a text file into documents on blank lines, optionally capped in bytes."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if limit_bytes is not None:
        text = text[:limit_bytes]
    return [doc.strip() for doc in text.split("\n\n") if doc.strip()]


def evaluate(train: Path, dev: Path, test: Path, order: int, tokenizer: str | None,
             train_bytes: int | None = None) -> dict:
    encode, vocab_size = make_tokenizer(tokenizer)
    train_docs = read_documents(train, train_bytes)
    dev_docs = read_documents(dev)
    test_docs = read_documents(test)
    lm = NGramLM(order=order, vocab_size=vocab_size)
    lm.fit(encode(d) for d in train_docs)
    lm.tune([encode(d) for d in dev_docs])
    test_ids = [encode(d) for d in test_docs]
    test_bytes = sum(len(d.encode("utf-8")) for d in test_docs)
    test_tokens = sum(len(ids) for ids in test_ids)
    return {
        "order": order,
        "tokenizer": tokenizer or "bytes",
        "train_docs": len(train_docs),
        "test_docs": len(test_docs),
        "test_bytes": test_bytes,
        "test_tokens": test_tokens,
        "bytes_per_token": round(test_bytes / test_tokens, 3),
        "weights": [round(w, 4) for w in lm.weights],
        "bits_per_byte": round(bits_per_byte(lm, test_ids, test_bytes), 4),
        "token_perplexity": round(perplexity(lm, test_ids), 2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--tokenizer", help="HF tokenizer.json; default splits into bytes")
    ap.add_argument("--train-bytes", type=int, help="use only the first N bytes of the training file")
    args = ap.parse_args(argv)
    result = evaluate(args.train, args.dev, args.test, args.order, args.tokenizer, args.train_bytes)
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
