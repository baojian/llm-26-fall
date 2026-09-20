"""Interpolated n-gram language model over integer token ids.

The same estimator serves three jobs in the course: the Week 2 baseline on the
held-out shards, the rung-zero point on the Week 9 scaling plot, and the
perplexity filter that scores web documents (``pipeline.filters.lm_score``).

    lm = NGramLM(order=3, vocab_size=tokenizer_vocab_size)
    lm.fit(train_ids)                     # counts
    lm.tune(dev_ids)                      # interpolation weights on held-out data
    bpb = bits_per_byte(lm, test_ids, test_bytes)

Token ids are plain integers; ``BOS`` and ``EOS`` are reserved negative ids so
they never collide with a tokenizer's vocabulary.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

BOS = -1
EOS = -2


def _pad(ids: Sequence[int], order: int) -> list[int]:
    return [BOS] * (order - 1) + list(ids) + [EOS]


@dataclass
class NGramLM:
    """Counts up to ``order`` and linear interpolation across orders.

    Predicted outcomes are ids ``0..vocab_size-1`` plus EOS, never BOS.
    If omitted, the size is inferred as the largest training id plus one.
    Pass the full tokenizer vocabulary size to cover ids absent from training.
    A positive uniform weight gives every in-vocabulary outcome positive mass.
    """

    order: int = 3
    vocab_size: int | None = None
    counts: list[dict] = field(default_factory=list)  # counts[k][history] -> Counter(next)
    context_totals: list[dict] = field(default_factory=list)  # context_totals[k][history] -> int
    weights: list[float] = field(default_factory=list)  # weights[0] uniform, weights[k] order k
    seen: set = field(default_factory=set)
    _configured_vocab_size: int | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be positive")
        if self.vocab_size is not None and self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        self._configured_vocab_size = self.vocab_size

    def _validate_ids(self, ids: Sequence[int], limit: int | None) -> None:
        if any(not isinstance(t, int) or t < 0 or (limit is not None and t >= limit) for t in ids):
            raise ValueError("content ids must be nonnegative integers within the tokenizer vocabulary")

    # ------------------------------------------------------------------ train
    def fit(self, sequences: Iterable[Sequence[int]]) -> "NGramLM":
        self.counts = [defaultdict(Counter) for _ in range(self.order + 1)]
        self.context_totals = [defaultdict(int) for _ in range(self.order + 1)]
        self.seen = set()
        self.vocab_size = self._configured_vocab_size
        for ids in sequences:
            ids = list(ids)
            self._validate_ids(ids, self.vocab_size)
            padded = _pad(ids, self.order)
            self.seen.update(ids)
            for i in range(self.order - 1, len(padded)):
                token = padded[i]
                for k in range(1, self.order + 1):
                    history = tuple(padded[i - k + 1 : i]) if k > 1 else ()
                    self.counts[k][history][token] += 1
                    self.context_totals[k][history] += 1
        if not self.context_totals[1]:
            raise ValueError("training requires at least one document")
        if self.vocab_size is None:
            self.vocab_size = max(self.seen, default=-1) + 1
        self.weights = [1.0 / (self.order + 1)] * (self.order + 1)
        return self

    # ------------------------------------------------------------------ score
    def _order_probs(self, history: Sequence[int], token: int) -> list[float]:
        """Probability of ``token`` under each order 0..N.

        Every entry is a proper distribution over the ``vocab_size`` ids plus EOS:
        order 0 is uniform over those ``vocab_size + 1`` outcomes, and an order
        whose history was never seen backs off to the next lower order's
        estimate. Without that, the mixture would drop the weight of the
        unseen order and no longer sum to one.
        """
        probs = [1.0 / (self.vocab_size + 1)]
        for k in range(1, self.order + 1):
            h = tuple(history[len(history) - (k - 1) :]) if k > 1 else ()
            total = self.context_totals[k].get(h, 0)
            probs.append(self.counts[k][h][token] / total if total else probs[-1])
        return probs

    def prob(self, history: Sequence[int], token: int) -> float:
        if token != EOS and not 0 <= token < self.vocab_size:
            return 0.0
        probs = self._order_probs(history, token)
        return sum(w * p for w, p in zip(self.weights, probs))

    def log2_prob_sequence(self, ids: Sequence[int]) -> tuple[float, int]:
        """Total log2 probability of a sequence (including EOS) and the token count."""
        self._validate_ids(ids, self.vocab_size)
        padded = _pad(ids, self.order)
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            history = padded[i - self.order + 1 : i]
            total += math.log2(self.prob(history, padded[i]))
        return total, len(padded) - (self.order - 1)

    # ------------------------------------------------------------------- tune
    def tune(self, sequences: Iterable[Sequence[int]], iterations: int = 20) -> list[float]:
        """Set interpolation weights by expectation-maximization on held-out data."""
        rows = []
        for ids in sequences:
            self._validate_ids(ids, self.vocab_size)
            padded = _pad(ids, self.order)
            for i in range(self.order - 1, len(padded)):
                rows.append(self._order_probs(padded[i - self.order + 1 : i], padded[i]))
        if not rows:
            raise ValueError("tuning requires at least one development document")
        weights = [1.0 / (self.order + 1)] * (self.order + 1)
        for _ in range(iterations):
            posterior = [0.0] * (self.order + 1)
            for probs in rows:
                mix = sum(w * p for w, p in zip(weights, probs))
                for k, (w, p) in enumerate(zip(weights, probs)):
                    posterior[k] += w * p / mix
            total = sum(posterior)
            weights = [p / total for p in posterior]
        self.weights = weights
        return weights

    # ----------------------------------------------------------------- sample
    def sample(self, rng, max_tokens: int = 64) -> list[int]:
        """Draw one sequence from the interpolated model (BOS to EOS).

        Samples the mixture exactly as ``prob`` defines it: pick an order with
        probability ``weights[k]``, then draw from that order's conditional;
        an order whose history was never seen backs off to the next lower
        order (the rule ``_order_probs`` uses), and order 0 draws uniformly
        from the ``vocab_size`` ids plus EOS.
        """
        history = [BOS] * (self.order - 1)
        out: list[int] = []
        while len(out) < max_tokens:
            k = rng.choices(range(self.order + 1), weights=self.weights)[0]
            token = None
            while k > 0:
                h = tuple(history[len(history) - (k - 1) :]) if k > 1 else ()
                counter = self.counts[k].get(h)
                if counter:
                    token = rng.choices(list(counter.keys()), weights=list(counter.values()))[0]
                    break
                k -= 1
            if token is None:
                draw = rng.randrange(self.vocab_size + 1)
                token = EOS if draw == self.vocab_size else draw
            if token == EOS:
                break
            out.append(token)
            history = (history + [token])[-(self.order - 1) :] if self.order > 1 else []
        return out


def bits_per_byte(lm: NGramLM, sequences: Iterable[Sequence[int]], total_bytes: int) -> float:
    """Cross-entropy of the model per UTF-8 byte of the original text.

    Compare tokenizers or model families on identical evaluated text and
    boundary conventions. Pass the UTF-8 byte length of that text, excluding
    separators or whitespace removed before tokenization. EOS contributes
    probability but no text bytes.
    """
    total = 0.0
    for ids in sequences:
        log2p, _ = lm.log2_prob_sequence(ids)
        total += log2p
    return -total / total_bytes


def perplexity(lm: NGramLM, sequences: Iterable[Sequence[int]]) -> float:
    """Token-level perplexity (2 ** average negative log2 probability)."""
    total, count = 0.0, 0
    for ids in sequences:
        log2p, n = lm.log2_prob_sequence(ids)
        total += log2p
        count += n
    return 2 ** (-total / count)
