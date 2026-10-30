"""Public tests for A1. The graders run a larger hidden suite that includes everything checked here;
passing here is necessary, not sufficient."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a1.bpe import decode, encode, pretokenize, train_bpe  # noqa: E402
from a1.ngram import BOS, EOS, NGramLM  # noqa: E402
from a1.nplm import NeuralNGramLM, make_examples, nearest_neighbours  # noqa: E402

TOY = ["I am Sam", "Sam I am", "I do not like eggs and ham"]


def toy_ids():
    vocab = sorted({w for s in TOY for w in s.split()})
    index = {w: i for i, w in enumerate(vocab)}
    return [[index[w] for w in s.split()] for s in TOY], index, len(vocab)


def test_pretokenize_matches_gpt2_examples():
    assert pretokenize("Hello world") == ["Hello", " world"]
    assert pretokenize("I'm 25 years old.") == ["I", "'m", " 25", " years", " old", "."]
    assert pretokenize("") == []
    assert pretokenize("snake_case") == ["snake", "_", "case"]  # every character must land in some chunk
    for text in ["a_b", "x²y", "  two  spaces", "复旦 NLP!"]:
        assert "".join(pretokenize(text)) == text


def test_bpe_round_trip_and_vocab_size():
    text = "low lower lowest newer wider new new new"
    vocab, merges = train_bpe(text, 270)
    assert len(vocab) == 270 and len(merges) == 14
    assert all(isinstance(v, bytes) for v in vocab.values())
    for sample in [text, "newest lowest", "unseen words too", ""]:
        assert decode(encode(sample, vocab, merges), vocab) == sample


def test_bigram_mle_on_the_lecture_toy_corpus():
    ids, index, vocab_size = toy_ids()
    lm = NGramLM(2, vocab_size).fit(ids)
    assert math.isclose(lm.prob_mle([BOS], index["I"]), 2 / 3)
    assert math.isclose(lm.prob_mle([index["I"]], index["am"]), 2 / 3)
    assert math.isclose(lm.prob_mle([index["Sam"]], EOS), 1 / 2)
    assert lm.prob_mle([index["ham"]], index["I"]) == 0.0


def test_add_delta_is_a_distribution_per_history():
    ids, index, vocab_size = toy_ids()
    lm = NGramLM(2, vocab_size).fit(ids)
    for h in [[BOS], [index["I"]], [index["ham"]], [999]]:
        total = sum(lm.prob_add(h, t, 0.5) for t in range(vocab_size)) + lm.prob_add(h, EOS, 0.5)
        assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_interpolation_weights_sum_to_one_and_score_is_finite():
    ids, index, vocab_size = toy_ids()
    lm = NGramLM(3, vocab_size).fit(ids)
    weights = lm.tune([[index["Sam"], index["do"], index["not"], index["like"], index["ham"]]])
    assert math.isclose(sum(weights), 1.0) and all(w >= 0 for w in weights)
    log2p, n = lm.log2_prob([index["I"], index["do"], index["not"], index["like"], index["Sam"]])
    assert math.isfinite(log2p) and n == 6


def test_uniform_model_over_ten_ids_plus_eos_has_perplexity_eleven():
    lm = NGramLM(1, 10).fit([list(range(10))] * 3)
    lm.weights = [1.0, 0.0]  # pure uniform floor over 11 outcomes (ten ids + EOS)
    log2p, n = lm.log2_prob(list(range(10)), "interp")
    assert math.isclose(2 ** (-log2p / n), 11.0)


def test_sampler_returns_ids_in_range_and_stops():
    import random

    ids, index, vocab_size = toy_ids()
    lm = NGramLM(2, vocab_size).fit(ids)
    lm.weights = [0.0, 0.0, 1.0]
    for seed in range(5):
        sample = lm.sample(random.Random(seed), max_tokens=20)
        assert all(0 <= t < vocab_size for t in sample) and len(sample) <= 20


def test_nplm_shapes_and_example_count():
    import torch

    model = NeuralNGramLM(vocab_size=10, context=3, dim=8, hidden=16)
    x, y = make_examples([[1, 2, 3], [4, 5]], model)
    assert x.shape == (4 + 3, 3) and y.shape == (7,)  # (3 tokens + EOS) + (2 tokens + EOS)
    logits = model(x)
    assert logits.shape == (7, 12)  # ten ids plus BOS and EOS rows
    assert torch.isfinite(logits).all()
    neighbours = nearest_neighbours(model, 3, k=4)
    assert len(neighbours) == 4 and all(i != 3 for i, _ in neighbours)
