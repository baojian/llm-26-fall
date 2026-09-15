"""Score documents with a reference n-gram model, the CCNet / RedPajama-V2 quality signal.

CCNet trains a 5-gram KenLM on Wikipedia and keeps the web documents whose
perplexity under that model is low; RedPajama-V2 ships the same number as
``ccnet_perplexity``. This module does the same job with
``pipeline.ngram_lm`` so it runs anywhere.

    from pipeline.filters.lm_score import ReferenceScorer
    scorer = ReferenceScorer.from_text(wikipedia_text, encode, vocab_size, order=3)
    scores = [scorer.bits_per_byte(doc) for doc in web_documents]
    keep = [doc for doc, s in zip(web_documents, scores) if s <= threshold]

Lower is closer to the reference. Choose the threshold on a labelled sample or
as a percentile of the score distribution; report the histogram either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

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
        split = max(1, int(len(docs) * (1 - dev_fraction)))
        lm = NGramLM(order=order, vocab_size=vocab_size)
        lm.fit(encode(d) for d in docs[:split])
        lm.tune([encode(d) for d in docs[split:]] or [encode(docs[0])])
        return cls(lm, encode)

    def bits_per_byte(self, document: str) -> float:
        ids = self.encode(document)
        log2p, _ = self.lm.log2_prob_sequence(ids)
        return -log2p / max(1, len(document.encode("utf-8")))

    def score_many(self, documents: Iterable[str]) -> list[float]:
        return [self.bits_per_byte(d) for d in documents]


def histogram(scores: list[float], bins: int = 30) -> list[tuple[float, int]]:
    """(bin left edge, count) pairs for a quick text or SVG histogram."""
    lo, hi = min(scores), max(scores)
    width = (hi - lo) / bins or 1.0
    counts = [0] * bins
    for s in scores:
        counts[min(bins - 1, int((s - lo) / width))] += 1
    return [(lo + i * width, c) for i, c in enumerate(counts)]
