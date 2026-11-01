"""Check population selection, provenance, missing values, and snapshot integrity."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_tokenizer_note import read_snapshot, select_models


def selection():
    return {"snapshot_date": "2026-10-31", "models": [
        {"model_id": "publisher/base", "stage": "base", "label": "Base",
         "base_evidence_url": "https://example.org/base", "release_year": None}]}


def test_explicit_population_does_not_promote_unclassified_hub_repositories():
    catalog = [{"model_id": "publisher/base", "downloads_30d": 10,
                "tokenizer_vocab_size": 32000, "timeline_date": "2023-01-01"},
               {"model_id": "random/test", "downloads_30d": 999999,
                "tokenizer_vocab_size": 26, "variant_types": ["not_declared"]}]
    result = select_models(selection(), catalog, [])
    assert [row["model_id"] for row in result] == ["publisher/base"]
    assert result[0]["release_year"] is None  # Do not turn a repository date into a release.
    assert result[0]["tokenizer_vocab_size"] == 32000  # No 50k threshold.


def test_missing_counts_are_retained_and_known_downloads_still_rank():
    result = select_models(selection(), [{"model_id": "publisher/base", "downloads_30d": 8}], [])
    assert result[0]["tokenizer_vocab_size"] is None
    assert result[0]["popularity_rank"] == 1
    result = select_models(selection(), [], [{"model_id": "publisher/base"}])
    assert result[0]["downloads_30d"] is None
    assert result[0]["popularity_rank"] is None


def test_curated_companion_aware_measurement_keeps_its_scope_and_source():
    snapshot = [{"model_id": "publisher/base", "downloads_30d": 8, "tokenizer_vocab_size": 100}]
    curated = [{"model_id": "publisher/base", "tokenizer_vocab_size": 102,
                "tokenizer_count_scope": "serialized_artifact_and_added_tokens",
                "tokenizer_source_url": "https://example.org/pinned-tokenizer"}]
    result = select_models(selection(), snapshot, curated)[0]
    assert result["tokenizer_vocab_size"] == 102
    assert result["tokenizer_count_scope"] == "serialized_artifact_and_added_tokens"
    assert result["tokenizer_source_url"] == "https://example.org/pinned-tokenizer"


def test_documented_repository_rename_preserves_measurement_and_ranks_canonical_id():
    config = selection()
    config["models"][0].update(curated_model_id="publisher/old-name",
                               alias_evidence_url="https://example.org/redirect")
    snapshot = [{"model_id": "publisher/base", "downloads_30d": 100}]
    curated = [{"model_id": "publisher/old-name", "tokenizer_vocab_size": None}]
    result = select_models(config, snapshot, curated)[0]
    assert result["downloads_30d"] == 100
    assert result["popularity_rank"] == 1
    assert result["tokenizer_vocab_size"] is None
    config["models"][0].pop("alias_evidence_url")
    with pytest.raises(ValueError, match="Repository-rename evidence"):
        select_models(config, snapshot, curated)


@pytest.mark.parametrize("change,match", [
    ({"stage": "instruct"}, "Base-checkpoint evidence"),
    ({"base_evidence_url": None}, "Base-checkpoint evidence"),
    ({"release_year": 2024}, "Release-year source"),
])
def test_selection_requires_training_stage_and_date_evidence(change, match):
    config = selection()
    config["models"][0].update(change)
    with pytest.raises(ValueError, match=match):
        select_models(config, [{"model_id": "publisher/base"}], [])


def test_duplicate_selection_is_not_double_counted():
    config = selection()
    config["models"].append(deepcopy(config["models"][0]))
    with pytest.raises(ValueError, match="Duplicate selection"):
        select_models(config, [{"model_id": "publisher/base"}], [])


def test_corrupted_snapshot_is_rejected_before_reading(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"archive": "catalog.zip", "archive_sha256": "bad"}))
    (tmp_path / "catalog.zip").write_bytes(b"not the reviewed snapshot")
    with pytest.raises(ValueError, match="checksum"):
        read_snapshot(tmp_path)


def test_committed_selection_uses_base_variants_and_retains_small_vocabularies():
    data = ROOT / "docs/model-tokenizer-statistics"
    config = json.loads((data / "foundation-selection.json").read_text())
    catalog, _ = read_snapshot(data / "snapshots/2026-10-31")
    rows = select_models(config, catalog, json.loads((data / "models.json").read_text()))
    by_id = {row["model_id"]: row for row in rows}
    assert "Qwen/Qwen3-8B-Base" in by_id
    assert "Qwen/Qwen3-8B" not in by_id
    assert "deepseek-ai/DeepSeek-R1" not in by_id
    assert by_id["mistralai/Mistral-7B-v0.1"]["tokenizer_vocab_size"] == 32000
    assert by_id["ibm-granite/granite-3.0-8b-base"]["tokenizer_vocab_size"] == 49152
    assert by_id["HuggingFaceTB/SmolLM2-135M"]["tokenizer_vocab_size"] == 49152
    assert by_id["meta-llama/Llama-3.1-8B"]["downloads_30d"] == 575876
    assert all(row["release_year"] is None or row["release_year_source_url"] for row in rows)
