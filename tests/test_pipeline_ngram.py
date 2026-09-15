"""Tests for the course pipeline's n-gram estimator, evaluation, and reference scorer."""

import math
import random

from pipeline.eval import make_tokenizer
from pipeline.filters.lm_score import ReferenceScorer, histogram
from pipeline.ngram_lm import BOS, EOS, NGramLM, bits_per_byte, perplexity

TOY = ["I am Sam", "Sam I am", "I do not like eggs and ham"]


def words(sentence):
    return sentence.split()


def encode_words(sentences):
    vocab = sorted({w for s in sentences for w in words(s)})
    index = {w: i for i, w in enumerate(vocab)}
    return [[index[w] for w in words(s)] for s in sentences], len(vocab)


def test_bigram_mle_matches_the_lecture_toy_example():
    ids, vocab_size = encode_words(TOY)
    lm = NGramLM(order=2, vocab_size=vocab_size).fit(ids)
    lm.weights = [0.0, 0.0, 1.0]  # pure bigram
    i, am, sam = (TOY[0].split().index(w) for w in ("I", "am", "Sam"))
    vocab = sorted({w for s in TOY for w in words(s)})
    I, AM, SAM = vocab.index("I"), vocab.index("am"), vocab.index("Sam")
    assert math.isclose(lm.prob([BOS], I), 2 / 3)
    assert math.isclose(lm.prob([I], AM), 2 / 3)
    assert math.isclose(lm.prob([SAM], EOS), 1 / 2)
    assert math.isclose(lm.prob([AM], SAM), 1 / 2)


def test_interpolation_keeps_every_probability_positive_and_normalized():
    ids, vocab_size = encode_words(TOY)
    lm = NGramLM(order=3, vocab_size=vocab_size).fit(ids)
    lm.tune(ids)
    assert all(w > 0 for w in lm.weights) and math.isclose(sum(lm.weights), 1.0)
    unseen_history = [vocab_size - 1, vocab_size - 1]
    assert lm.prob(unseen_history, 0) > 0
    # the mixture over all ids plus EOS sums to one for seen, partially seen, and unseen histories
    for history in ([BOS, BOS], [0, 1], [vocab_size - 1], unseen_history):
        total = sum(lm.prob(history, t) for t in range(vocab_size)) + lm.prob(history, EOS)
        assert math.isclose(total, 1.0, abs_tol=1e-9), history


def test_bits_per_byte_and_perplexity_agree_on_uniform_digits():
    digits = [list(range(10))] * 5
    lm = NGramLM(order=1, vocab_size=10)
    lm.fit(digits)
    lm.weights = [1.0, 0.0]  # uniform over the 10 ids plus EOS: perplexity 11 per token
    assert math.isclose(perplexity(lm, digits), 11.0)
    text_bytes = sum(len("0123456789") for _ in digits)
    assert math.isclose(bits_per_byte(lm, digits, text_bytes), 11 * math.log2(11) / 10)


def test_sampler_returns_training_tokens_and_stops():
    ids, vocab_size = encode_words(TOY)
    lm = NGramLM(order=2, vocab_size=vocab_size).fit(ids)
    lm.weights = [0.0, 0.0, 1.0]
    rng = random.Random(2026)
    for _ in range(20):
        sample = lm.sample(rng, max_tokens=30)
        assert all(0 <= t < vocab_size for t in sample)
        assert len(sample) <= 30
    lm.weights = [1.0, 0.0, 0.0]  # pure uniform over ids plus EOS: some samples must stop before the cap
    assert min(len(lm.sample(random.Random(s), max_tokens=50)) for s in range(30)) < 50


def test_byte_tokenizer_and_reference_scorer_prefer_in_domain_text():
    encode, vocab_size = make_tokenizer(None)
    reference = "\n\n".join(["the cat sat on the mat"] * 50 + ["the dog sat on the rug"] * 50)
    scorer = ReferenceScorer.from_text(reference, encode, vocab_size, order=3)
    close = scorer.bits_per_byte("the cat sat on the rug")
    far = scorer.bits_per_byte("quantum chromodynamics lattice")
    assert close < far
    bins = histogram([close, far, close], bins=4)
    assert sum(c for _, c in bins) == 3
