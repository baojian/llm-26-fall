"""Regression checks for vocabulary identity and exact-repository popularity."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tokenizer_catalog_analysis import grouped_representatives, popularity_ranking


def row(identifier, count, **extra):
    return {"model_id": identifier, "tokenizer_vocab_size": count,
            "timeline_date": "2025-01-01", **extra}


def test_shared_kimi_rank_file_does_not_merge_different_added_tokens():
    base = row("Kimi-K2-Base", 163842, tokenizer_artifact_sha256="same-ranks")
    newer = row("Kimi-K2.5", 163840, tokenizer_artifact_sha256="same-ranks")
    assert len(grouped_representatives([base, newer])) == 2


def test_same_artifact_and_count_are_still_not_full_mapping_evidence():
    rows = [row(name, 100, tokenizer_artifact_sha256="same-ranks") for name in ("a", "b")]
    assert len(grouped_representatives(rows)) == 2


def test_complete_mapping_hash_collapses_to_earliest_observation():
    earlier = row("a", 100, shared_vocabulary_sha256="complete")
    later = row("b", 100, shared_vocabulary_sha256="complete", timeline_date="2026-01-01")
    assert grouped_representatives([later, earlier]) == [earlier]


def test_conflicting_counts_under_one_mapping_hash_stop_the_plot():
    rows = [row(str(n), n, shared_vocabulary_sha256="complete") for n in (100, 101)]
    with pytest.raises(ValueError, match="Conflicting vocabulary sizes"):
        grouped_representatives(rows)


def test_popularity_distinguishes_zero_unknown_and_stable_ties():
    rows = [row("unknown", 100, downloads_30d=None), row("zero", 100, downloads_30d=0),
            row("b", 100, downloads_30d=10), row("a", 100, downloads_30d=10)]
    assert [r["model_id"] for r in popularity_ranking(rows)] == ["a", "b", "zero"]
