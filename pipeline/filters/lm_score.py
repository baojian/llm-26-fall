"""Score documents with a CCNet-inspired reference n-gram model.

CCNet scores documents with a Wikipedia 5-gram KenLM and assigns perplexity
buckets; RedPajama-V2 exposes that score as ``ccnet_perplexity``. Selection is
a separate policy. This teaching implementation uses interpolated n-grams and
bits per byte; its scores and thresholds are not the original CCNet signal.

    from pipeline.filters.lm_score import ReferenceScorer
    scorer = ReferenceScorer.from_text(wikipedia_text, encode, vocab_size, order=3)
    scores = [scorer.bits_per_byte(doc) for doc in web_documents]
    keep = [doc for doc, s in zip(web_documents, scores) if s <= threshold]

Lower is closer to the reference. Choose the threshold on a labelled sample or
as a percentile of the score distribution; report the histogram either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from pipeline.ngram_lm import NGramLM

Tokenizer = Callable[[str], list[int]]


@dataclass
class ReferenceScorer:
    lm: NGramLM
    encode: Tokenizer

    @classmethod
    def from_text(cls, reference_text: str, encode: Tokenizer, vocab_size: int,
                  order: int = 3, dev_fraction: float = 0.05) -> "ReferenceScorer":
        docs = [d.strip() for d in reference_text.split("\n\n") if d.strip()]
        return cls.from_documents(docs, encode, vocab_size, order, dev_fraction)

    @classmethod
    def from_documents(cls, docs: Sequence[str], encode: Tokenizer, vocab_size: int,
                       order: int = 3, dev_fraction: float = 0.05) -> "ReferenceScorer":
        """Split whole reference documents without treating internal paragraphs as documents."""
        split = reference_split(len(docs), dev_fraction)
        lm = NGramLM(order=order, vocab_size=vocab_size)
        lm.fit(encode(d) for d in docs[:split])
        lm.tune([encode(d) for d in docs[split:]])
        return cls(lm, encode)

    def bits_per_byte(self, document: str) -> float:
        ids = self.encode(document)
        log2p, _ = self.lm.log2_prob_sequence(ids)
        return -log2p / max(1, len(document.encode("utf-8")))

    def score_many(self, documents: Iterable[str]) -> list[float]:
        return [self.bits_per_byte(d) for d in documents]


def reference_split(n_docs: int, dev_fraction: float) -> int:
    if n_docs < 2 or not 0 < dev_fraction < 1:
        raise ValueError("reference scoring requires at least two documents and 0 < dev_fraction < 1")
    return min(n_docs - 1, max(1, int(n_docs * (1 - dev_fraction))))


def histogram(scores: list[float], bins: int = 30,
              bounds: tuple[float, float] | None = None) -> list[tuple[float, int]]:
    """(bin left edge, count) pairs for a quick text or SVG histogram."""
    lo, hi = bounds if bounds is not None else (min(scores), max(scores))
    width = (hi - lo) / bins or 1.0
    counts = [0] * bins
    for s in scores:
        counts[min(bins - 1, int((s - lo) / width))] += 1
    return [(lo + i * width, c) for i, c in enumerate(counts)]
