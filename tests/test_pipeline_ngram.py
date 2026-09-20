"""Tests for the course pipeline's n-gram estimator, evaluation, and reference scorer."""

import math
import random
from collections import Counter

import pytest

from pipeline.eval import evaluate, make_tokenizer, read_documents
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


@pytest.mark.parametrize("order", [1, 2, 3])
def test_inferred_vocabulary_covers_sparse_ids_without_counting_boundaries(order):
    lm = NGramLM(order=order).fit([[100, 200]])
    assert lm.vocab_size == 201
    assert lm.seen == {100, 200}
    for history in ([BOS] * (order - 1), [199] * (order - 1)):
        mass = sum(lm.prob(history, t) for t in range(201)) + lm.prob(history, EOS)
        assert mass == pytest.approx(1)
        assert lm.prob(history, BOS) == 0
        assert lm.prob(history, 201) == 0


def test_refit_resets_inferred_support_counts_and_tuned_weights():
    lm = NGramLM(order=2).fit([[100, 200]])
    lm.tune([[100, 200]])
    lm.fit([[0, 1]])
    assert lm.vocab_size == 2 and lm.seen == {0, 1}
    assert lm.weights == pytest.approx([1 / 3] * 3)
    assert lm.prob([BOS], 200) == 0
    fixed = NGramLM(order=2, vocab_size=256).fit([[100, 200]])
    fixed.fit([[0, 1]])
    assert fixed.vocab_size == 256


def test_default_sampler_matches_scored_mixture_including_eos():
    lm = NGramLM(order=2).fit([[0, 1], [1, 2], [0, 0]])
    rng = random.Random(37)
    draws = Counter()
    for _ in range(12000):
        sample = lm.sample(rng, max_tokens=1)
        draws[sample[0] if sample else EOS] += 1
    for token in [0, 1, 2, EOS]:
        assert draws[token] / 12000 == pytest.approx(lm.prob([BOS], token), abs=0.015)


def test_reserved_or_out_of_vocabulary_content_ids_are_rejected():
    for ids in ([BOS], [EOS], [3]):
        with pytest.raises(ValueError, match="content ids"):
            NGramLM(vocab_size=3).fit([ids])
    lm = NGramLM(vocab_size=3).fit([[0]])
    with pytest.raises(ValueError, match="content ids"):
        lm.tune([[3]])
    with pytest.raises(ValueError, match="development"):
        lm.tune([])


@pytest.mark.parametrize("text,cap,expected", [("abcdef", 3, ["abc"]),
    ("你好世界", 2, []), ("你好世界", 3, ["你"]), ("你好世界", 4, ["你"]),
    ("你好世界", 6, ["你好"]), ("你好世界", 12, ["你好世界"])])
def test_training_cap_counts_utf8_bytes_at_codepoint_boundaries(tmp_path, text, cap, expected):
    path = tmp_path / "text.txt"
    path.write_text(text, encoding="utf-8")
    docs = read_documents(path, cap)
    assert docs == expected
    assert sum(len(d.encode("utf-8")) for d in docs) <= cap


def test_windows_newlines_and_blank_document_separators(tmp_path):
    path = tmp_path / "text.txt"
    path.write_bytes(b"first\r\nline\r\n\r\nsecond\r\n")
    assert read_documents(path) == ["first\nline", "second"]


def test_evaluation_metrics_use_the_same_eos_count(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("a", encoding="utf-8")
    result = evaluate(path, path, path, 1, None)
    assert result["test_content_tokens"] == 1
    assert result["test_tokens"] == 2 and result["test_bytes"] == 1
    assert result["loss_nats_per_token"] == pytest.approx(math.log(2), abs=1e-6)
    assert result["bits_per_byte"] == pytest.approx(
        result["test_tokens"] / result["test_bytes"] * math.log2(result["token_perplexity"]))


def test_reference_split_preserves_internal_paragraphs():
    docs = ["one\n\narticle", "another\n\narticle"]
    encoded = []
    def encode(text):
        encoded.append(text)
        return list(text.encode())
    scorer = ReferenceScorer.from_documents(docs, encode, 256)
    assert encoded == docs
    assert scorer.lm.context_totals[1][()] == len(docs[0].encode()) + 1
    with pytest.raises(ValueError, match="at least two"):
        ReferenceScorer.from_documents(docs[:1], encode, 256)


def test_histograms_can_share_bins_for_different_populations():
    web = histogram([1, 2, 3], bins=4, bounds=(1, 5))
    stories = histogram([4, 5], bins=4, bounds=(1, 5))
    assert [x for x, _ in web] == [x for x, _ in stories]
    assert sum(c for _, c in web) == 3 and sum(c for _, c in stories) == 2
