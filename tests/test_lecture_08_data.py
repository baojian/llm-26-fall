"""Independent checks of the small Lecture 08 curation examples."""

import importlib.util
from itertools import permutations, product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "lecture08_data_quality", ROOT / "slides/lecture-08/assets/data_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_minhash_collision_matches_jaccard_over_every_permutation():
    orders = list(permutations("ABCD"))
    left, right = set("ABC"), set("BCD")
    a = quality.minhash_signature(left, orders)
    b = quality.minhash_signature(right, orders)
    assert sum(x == y for x, y in zip(a, b)) == 12
    assert sum(x == y for x, y in zip(a, b)) / len(orders) == len(left & right) / len(left | right)
    assert quality.minhash_signature({"A"}, orders) == ("A",) * 24
    for items, invalid in [(set(), orders), ({"E"}, orders), ({"A"}, []),
                           ({"A"}, ["ABCD", "AACD"]), ({"A"}, ["ABCD", "ABCE"])]:
        with pytest.raises(ValueError):
            quality.minhash_signature(items, invalid)


def test_lsh_probability_matches_enumerated_independent_rows():
    # Two bands of two Bernoulli(1/2) rows: 7 of the 16 outcomes share a band.
    outcomes = list(product((0, 1), repeat=4))
    empirical = sum((a and b) or (c and d) for a, b, c, d in outcomes) / len(outcomes)
    assert quality.candidate_probability(.5, bands=2, rows=2) == empirical == 7 / 16
    assert quality.candidate_probability(0, bands=14, rows=8) == 0
    assert quality.candidate_probability(1, bands=14, rows=8) == 1
    with pytest.raises(ValueError):
        quality.candidate_probability(float("nan"), bands=2, rows=2)
    with pytest.raises(ValueError):
        quality.candidate_probability(.5, bands=True, rows=2)


def test_connected_families_preserve_transitive_matches_and_isolated_records():
    assert quality.duplicate_components(["A", "B", "C", "D"], [("A", "B"), ("B", "C")]) == [
        ["A", "B", "C"], ["D"]]
    assert quality.duplicate_components(["A", "B", "C", "D"], [("C", "B"), ("B", "A")]) == [
        ["A", "B", "C"], ["D"]]
    assert quality.duplicate_components([], []) == []
    with pytest.raises(ValueError):
        quality.duplicate_components(["A", "A"], [])
    with pytest.raises(ValueError):
        quality.duplicate_components(["A"], [("A", "B")])


def test_sampling_weights_conserve_budget_and_expose_small_source_repetition():
    natural = quality.mixture_exposure([9e6, 1e6], 60e6, exponent=1)
    tempered = quality.mixture_exposure([9e6, 1e6], 60e6, exponent=.5)
    assert natural["probabilities"] == pytest.approx([.9, .1])
    assert natural["expected_passes"] == pytest.approx([6, 6])
    assert tempered["probabilities"] == pytest.approx([.75, .25])
    assert tempered["processed_tokens"] == pytest.approx([45e6, 15e6])
    assert tempered["expected_passes"] == pytest.approx([5, 15])
    for result in (natural, tempered):
        assert sum(result["processed_tokens"]) == pytest.approx(60e6)
    assert quality.mixture_exposure([9e6, 1e6], 60e6, exponent=0)["probabilities"] == [.5, .5]
    for counts, budget, exponent in [([], 1, 1), ([0, 2], 1, 1), ([1], 0, 1),
                                    ([1], 1, -1), ([float("inf")], 1, 1)]:
        with pytest.raises(ValueError):
            quality.mixture_exposure(counts, budget, exponent=exponent)
