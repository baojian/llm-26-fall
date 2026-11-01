"""Protect source completeness, exact joins, missing values, and date provenance."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_tokenizer_note import assemble_inventory, csv_text, read_snapshot, select_models
from import_wikipedia_llms import extract_inventory


def source():
    return {"entries": [{"entry_id": "wiki-2025-01", "listed_name": "Example",
                         "listed_year": 2025, "listed_date": "Jan", "listed_developer": "Publisher",
                         "source_url": "https://example.org/list", "article_url": None,
                         "reference_urls": []}]}


def decisions(identifier="publisher/exact"):
    return {"mappings": [{"entry_id": "wiki-2025-01", "listed_name": "Example",
                          "representative_id": identifier, "category": "publisher_release",
                          "mapping_evidence_url": "https://example.org/model" if identifier else None,
                          "mapping_note": "Exact representative, not its parent."}]}


def measured(identifier="publisher/exact"):
    return {"model_id": identifier, "tokenizer_vocab_size": 32000,
            "tokenizer_source_url": "https://example.org/tokenizer", "tokenizer_revision": "abc",
            "tokenizer_sha256": "digest", "tokenizer_count_scope": "serialized_tokenizer_json_only"}


def test_rowspans_preserve_names_dates_developers_and_every_row():
    raw = '''<script>{"wgRevisionId":123}</script><h3>2026</h3><table class="wikitable">
    <tr><th>Model</th><th>Date</th><th>Developer</th><th>Notes</th></tr>
    <tr><td>First</td><td rowspan="2">Jan 27</td><td rowspan="2">Publisher</td><td>Prose</td></tr>
    <tr><td>Second<sup>[1]</sup></td><td>Other prose</td></tr>
    <tr><td>Third</td><td>Feb</td><td>Another</td><td>Not copied</td></tr></table>'''
    data = extract_inventory(raw)
    assert len(data["entries"]) == 3
    assert data["source"]["rows_by_year"] == {"2026": 3}
    assert [row["listed_name"] for row in data["entries"]] == ["First", "Second", "Third"]
    assert data["entries"][1]["listed_date"] == "Jan 27"
    assert data["entries"][1]["listed_developer"] == "Publisher"
    assert "prose" not in json.dumps(data).lower()


def test_complete_inventory_retains_unmapped_entries_without_zero_imputation():
    rows = assemble_inventory(source(), decisions(None), [], [], [], [])
    assert len(rows) == 1
    assert rows[0]["model_id"] is None
    assert rows[0]["measurement_status"] == "not_mapped"
    assert rows[0]["tokenizer_vocab_size"] is None
    assert rows[0]["popularity_rank"] is None


def test_exact_join_does_not_copy_a_parent_or_new_download_snapshot():
    catalog = [{**measured("publisher/parent"), "downloads_30d": 999999},
               {"model_id": "publisher/exact", "declared_base_models": ["publisher/parent"]}]
    extra = [{"model_id": "publisher/exact", "downloads_30d": 1234}]
    row = assemble_inventory(source(), decisions(), [], catalog, [], extra)[0]
    assert row["measurement_status"] == "mapped_not_measured"
    assert row["tokenizer_vocab_size"] is None
    assert row["downloads_30d"] is None
    assert row["downloads_observed_on"] is None


def test_conflicting_listed_and_verified_years_are_both_retained():
    curated = [{**measured(), "release_date": "2024-09-19", "release_date_source_url": "https://example.org/release"}]
    row = assemble_inventory(source(), decisions(), [], [], curated, [])[0]
    assert (row["listed_year"], row["release_year"], row["timeline_year"]) == (2025, 2024, 2024)
    assert row["year_differs_from_listing"]
    assert row["timeline_year_basis"] == "primary_source"


def test_repository_date_cannot_become_verified_release_year():
    catalog = [{**measured(), "timeline_date": "2020-01-01", "hf_repository_created_at": "2020-01-01"}]
    row = assemble_inventory(source(), decisions(), [], catalog, [], [])[0]
    assert row["release_year"] is None
    assert row["timeline_year"] == 2025
    assert row["timeline_year_basis"] == "wikipedia_only"


def test_api_encoding_is_separate_and_missing_vocabulary_still_ranks():
    encoding = {**measured(), "tokenizer_count_scope": "public_text_encoding_only"}
    row = assemble_inventory(source(), decisions(), [], [], [encoding], [])[0]
    assert row["measurement_status"] == "public_text_encoding"
    row = assemble_inventory(source(), decisions(), [], [{"model_id": "publisher/exact", "downloads_30d": 0}], [], [])[0]
    assert row["tokenizer_vocab_size"] is None
    assert row["popularity_rank"] == 1


def test_supplemental_artifact_can_fill_a_gap_with_its_own_counting_scope():
    row = assemble_inventory(source(), decisions(), [], [],
                             [{"model_id": "publisher/exact", "tokenizer_status": "gated"}], [measured()])[0]
    assert row["tokenizer_vocab_size"] == 32000
    assert row["tokenizer_count_scope"] == "serialized_tokenizer_json_only"


def test_missing_or_stale_mapping_cannot_silently_drop_an_entry():
    with pytest.raises(ValueError, match="every inventory row"):
        assemble_inventory(source(), {"mappings": []}, [], [], [], [])
    config = decisions()
    config["mappings"][0]["listed_name"] = "Different generation"
    with pytest.raises(ValueError, match="Source row changed"):
        assemble_inventory(source(), config, [], [], [], [])


def test_two_source_rows_cannot_silently_double_count_a_representative():
    inventory, config = source(), decisions()
    inventory["entries"].append({**deepcopy(inventory["entries"][0]), "entry_id": "wiki-2025-02"})
    config["mappings"].append({**deepcopy(config["mappings"][0]), "entry_id": "wiki-2025-02"})
    with pytest.raises(ValueError, match="Duplicate representative"):
        assemble_inventory(inventory, config, [], [], [], [])


def test_csv_retains_array_references_nulls_and_boolean_fields():
    record = {"name": 'A, "B"', "references": ["https://example.org/a", "https://example.org/b"],
              "vocabulary": None, "verified": False}
    exported = next(csv.DictReader(io.StringIO(csv_text([record]))))
    assert exported["name"] == record["name"]
    assert json.loads(exported["references"]) == record["references"]
    assert exported["vocabulary"] == ""
    assert exported["verified"] == "false"


def test_pinned_inventory_covers_all_189_source_rows_and_keeps_original_cohort():
    data = ROOT / "docs/model-tokenizer-statistics"
    load = lambda name: json.loads((data / name).read_text())
    inventory = load("wikipedia-inventory.json")
    catalog, _ = read_snapshot(data / "snapshots/2026-10-31")
    curated = load("models.json")
    bases = select_models(load("foundation-selection.json"), catalog, curated)
    rows = assemble_inventory(inventory, load("inventory-mappings.json"), bases, catalog,
                              curated, load("inventory-additional-measurements.json")["models"])
    assert len(rows) == 189 == sum(inventory["source"]["rows_by_year"].values())
    assert {r["entry_id"] for r in rows} == {r["entry_id"] for r in inventory["entries"]}
    assert len(bases) == 42
    by_name = {r["listed_name"]: r for r in rows}
    assert by_name["Qwen2.5"]["listed_year"] == 2025
    assert by_name["Qwen2.5"]["release_year"] == 2024
    assert by_name["DeepSeek-V4-Flash"]["model_id"].endswith("-0731")
    assert by_name["DeepSeek-V4-Pro"]["model_id"].endswith("-0813")
    assert by_name["Mistral 7B"]["tokenizer_vocab_size"] == 32000
    assert by_name["Mistral Large"]["tokenizer_vocab_size"] == 32768
    assert by_name["GPT-4"]["measurement_status"] == "public_text_encoding"
    assert by_name["HyperCLOVA"]["measurement_status"] == "not_mapped"
    assert all(r["downloads_observed_on"] in (None, "2026-10-31") for r in rows)
