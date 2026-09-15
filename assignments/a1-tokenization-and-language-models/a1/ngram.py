"""A1 Part 2: n-gram language models over token ids with MLE, add-delta, and interpolation.

BOS = -1 and EOS = -2 are reserved. Every sequence is padded with order-1 BOS
tokens and one EOS. Standard library only.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

BOS = -1
EOS = -2


def pad(ids: list[int], order: int) -> list[int]:
    return [BOS] * (order - 1) + list(ids) + [EOS]


class NGramLM:
    def __init__(self, order: int, vocab_size: int):
        self.order = order
        self.vocab_size = vocab_size  # ids 0..vocab_size-1 plus EOS are the possible next tokens
        self.counts: list[dict] = [defaultdict(Counter) for _ in range(order + 1)]  # counts[k][history] -> Counter
        self.totals: list[dict] = [defaultdict(int) for _ in range(order + 1)]  # totals[k][history] -> int
        self.weights: list[float] = [1.0 / (order + 1)] * (order + 1)  # weights[0] = uniform floor

    # ------------------------------------------------------------------ counts
    def fit(self, sequences: list[list[int]]) -> "NGramLM":
        """Count n-grams of every order 1..self.order over padded sequences."""
        # YOUR CODE HERE
        raise NotImplementedError
        return self

    def _history(self, history: list[int], k: int) -> tuple:
        h = tuple(history[len(history) - (k - 1) :]) if k > 1 else ()
        return h

    # ------------------------------------------------------------ estimators
    def prob_mle(self, history: list[int], token: int) -> float:
        """C(history, token) / C(history) at the model's full order; 0.0 if unseen."""
        # YOUR CODE HERE
        raise NotImplementedError

    def prob_add(self, history: list[int], token: int, delta: float) -> float:
        """(C(history, token) + delta) / (C(history) + delta * (vocab_size + 1)); the +1 is EOS."""
        # YOUR CODE HERE
        raise NotImplementedError

    def _order_probs(self, history: list[int], token: int) -> list[float]:
        """Probability of ``token`` under each order 0..N; every entry is a proper distribution over
        the vocab_size + 1 outcomes (ids plus EOS).

        Order 0 is uniform. If the history was never seen at order k, that order's
        estimate is the order k-1 estimate (back-off inside the mixture); otherwise
        the mixture would lose the weight of that order and stop summing to one.
        """
        probs = [1.0 / (self.vocab_size + 1)]
        for k in range(1, self.order + 1):
            h = self._history(history, k)
            total = self.totals[k].get(h, 0)
            probs.append(self.counts[k][h][token] / total if total else probs[-1])
        return probs

    def prob_interp(self, history: list[int], token: int) -> float:
        """Sum_k weights[k] * P_k(token | history), with the uniform floor at k = 0."""
        # YOUR CODE HERE
        raise NotImplementedError

    def tune(self, dev_sequences: list[list[int]], iterations: int = 20) -> list[float]:
        """EM for the interpolation weights on held-out data; returns the weights."""
        # YOUR CODE HERE
        raise NotImplementedError
        return self.weights

    # ------------------------------------------------------------- scoring
    def log2_prob(self, ids: list[int], kind: str = "interp", delta: float = 1.0) -> tuple[float, int]:
        """Total log2 probability of a padded sequence and the number of predicted tokens.

        kind: "mle" (with a 1e-6 uniform floor so nothing is -inf), "add", or "interp".
        """
        # YOUR CODE HERE
        raise NotImplementedError

    # ------------------------------------------------------------- sampling
    def sample(self, rng: random.Random, max_tokens: int = 60) -> list[int]:
        """Draw one sequence from the interpolated model, BOS to EOS (EOS not included in the output).

        Sample the mixture exactly as ``prob_interp`` defines it: pick an order k
        with probability weights[k], then draw from that order's conditional
        counts; if the history was never seen at that order, fall back to the
        next lower order (the same rule ``_order_probs`` uses); order 0 draws
        uniformly from the vocab_size ids plus EOS.
        """
        history = [BOS] * (self.order - 1)
        out: list[int] = []
        # YOUR CODE HERE
        raise NotImplementedError
        return out
