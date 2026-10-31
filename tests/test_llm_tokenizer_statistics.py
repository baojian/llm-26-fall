"""Offline checks for vocabulary accounting and the curated model inventory."""

import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "llm_tokenizer_statistics", ROOT / "scripts/llm_tokenizer_statistics.py"
)
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def test_special_tokens_are_not_double_counted_and_id_gaps_are_preserved():
    result = stats.vocabulary_statistics(
        {0: "[UNK]", 2: "word"},
        [{"id": 0, "content": "[UNK]", "special": True},
         {"id": 7, "content": "[END]", "special": True}], set()
    )
    assert result == {
        "tokenizer_vocab_size": 3, "tokenizer_base_vocab_size": 2,
        "tokenizer_added_vocab_size": 1, "tokenizer_special_token_count": 2,
        "tokenizer_id_capacity": 8,
    }


def test_conflicting_definitions_and_nonexistent_special_ids_fail():
    with pytest.raises(ValueError, match="Conflicting"):
        stats.vocabulary_statistics({0: "a"}, [{"id": 0, "content": "b"}], set())
    with pytest.raises(ValueError, match="must exist"):
        stats.vocabulary_statistics({0: "a"}, [], {1})


def test_configuration_additions_are_merged_with_serialized_vocabulary(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"a": 0, "[UNK]": 1}},
        "added_tokens": [{"id": 1, "content": "[UNK]", "special": True}],
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": {"4": {"content": "[END]", "special": True}}
    }))
    measured = stats.measure(tmp_path)
    assert measured["tokenizer_vocab_size"] == 3
    assert measured["tokenizer_id_capacity"] == 5
    assert measured["tokenizer_special_token_count"] == 2


@pytest.mark.parametrize("profile,reserved", [("kimi_k2", 258), ("kimi_k25", 256)])
def test_kimi_versions_have_different_reserved_token_counts(tmp_path, profile, reserved):
    (tmp_path / "tiktoken.model").write_text("YQ== 0\nYg== 1\n")
    measured = stats.measure(tmp_path, profile)
    assert measured["tokenizer_vocab_size"] == 2 + reserved
    assert measured["tokenizer_special_token_count"] == reserved


def test_rank_file_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "ranks.tiktoken"
    path.write_text("YQ== 0\nYg== 0\n")
    with pytest.raises(ValueError, match="Duplicate rank"):
        stats.read_ranks(path)


def test_csv_keeps_null_distinct_from_zero_and_escapes_text():
    rows = [{"missing": None, "zero": 0, "text": 'a,"b"\nc'}]
    parsed = list(csv.DictReader(io.StringIO(stats.csv_text(rows))))
    assert parsed == [{"missing": "", "zero": "0", "text": 'a,"b"\nc'}]


def test_manifest_rejects_missing_primary_file_and_checksum_mismatch(tmp_path):
    payload = b"a tokenizer fixture"
    relative = "workspace/llm-tokenizers/test/tokenizer.model"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    manifest = [{"local_path": relative, "url": "https://example.org/tokenizer.model",
                 "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}]
    stats.verify_files(manifest, tmp_path)
    path.write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="checksum mismatch"):
        stats.verify_files(manifest, tmp_path)
    with pytest.raises(ValueError, match="primary artifact"):
        stats.validate_manifest([{
            "model_id": "test", "tokenizer_status": "downloaded",
            "tokenizer_local_path": "workspace/llm-tokenizers/missing",
            "tokenizer_primary_file": "tokenizer.model", "tokenizer_sha256": "a" * 64,
        }], manifest)


def test_fetch_reconstructs_encoding_definition_and_rejects_changed_download(tmp_path, monkeypatch):
    source = stats.DATA / "encodings/0.14.0/r50k_base.json"
    payload = source.read_bytes()
    relative = "workspace/llm-tokenizers/test/encoding.json"
    entry = {"local_path": relative, "source_path": str(source.relative_to(ROOT)),
             "url": None, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    stats.verify_files([entry], tmp_path, fetch=True)
    assert (tmp_path / relative).read_bytes() == payload

    download = {"local_path": "workspace/llm-tokenizers/test/new.model",
                "url": "https://example.org/new.model", "sha256": "0" * 64, "bytes": 7}
    monkeypatch.setattr(stats.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"changed"))
    with pytest.raises(ValueError, match="Downloaded checksum mismatch"):
        stats.verify_files([download], tmp_path, fetch=True)
    assert not (tmp_path / download["local_path"]).exists()


@pytest.mark.parametrize("path", ["/tmp/tokenizer.json", "workspace/llm-tokenizers/../../../bad", "docs/tokenizer.json"])
def test_manifest_rejects_paths_outside_cache(path):
    with pytest.raises(ValueError, match="inside the tokenizer cache"):
        stats.validate_manifest([], [{"local_path": path}])


@pytest.mark.parametrize("date,precision", [("2025-02-30", "day"), ("2024-13", "month"), ("2024", "day")])
def test_invalid_dates_and_false_precision_fail(date, precision):
    row = copy.deepcopy(stats.read_json(stats.DATA / "models.json")[0])
    row.update(release_date=date, release_date_precision=precision)
    with pytest.raises(ValueError):
        stats.validate([row])


def test_curated_dataset_and_export_agree():
    rows = stats.read_json(stats.DATA / "models.json")
    stats.validate(rows)
    stats.validate_manifest(rows, stats.read_json(stats.DATA / "tokenizer-files.json"))
    assert (stats.DATA / "models.csv").read_text() == stats.csv_text(rows)
    assert {row["weights_access"] for row in rows} == {"open_weights", "closed_weights"}
