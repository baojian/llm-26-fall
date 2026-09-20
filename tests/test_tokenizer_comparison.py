"""Test artifact integrity, independent checking, and tokenizer boundary cases.

The normal suite uses tiny local fixtures and never downloads artifacts.
Set TOKENIZER_COMPARISON_INTEGRATION=1 after fetch to verify the six real files.
"""

import base64
import copy
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

from scripts import tokenizer_comparison as comparison


def byte_fixture(tmp_path, name="bytes"):
    content = b"".join(base64.b64encode(bytes([value])) + f" {value}\n".encode() for value in range(256))
    (tmp_path / f"{name}.artifact").write_bytes(content)
    return {
        "id": name, "backend": "tiktoken", "url": "https://example.invalid/tokenizer",
        "sha256": hashlib.sha256(content).hexdigest(), "revision": "test-fixture",
        "pattern": r"\S+|\s+", "special_tokens": {"<|end|>": 300},
    }


def contribution(names):
    record = comparison.read_json(comparison.TASK / "contribution-template.json")
    record.update(contributors=["alice"], tokenizers=names)
    record["cases"][-1].update(text="A1β", prediction="A byte tokenizer will split β into two byte tokens.")
    return record


def test_missing_or_corrupt_cache_fails_without_network(tmp_path, monkeypatch):
    spec = byte_fixture(tmp_path)
    def no_network(*args, **kwargs):
        pytest.fail("Offline commands must not request a download")
    monkeypatch.setattr(comparison.urllib.request, "urlopen", no_network)
    with pytest.raises(ValueError, match="Missing.*fetch"):
        comparison.Engine(spec, tmp_path / "absent")
    (tmp_path / "bytes.artifact").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        comparison.Engine(spec, tmp_path)


def test_fetch_checks_hash_before_replacing_and_reuses_verified_cache(tmp_path, monkeypatch):
    spec = byte_fixture(tmp_path)
    original = (tmp_path / "bytes.artifact").read_bytes()
    (tmp_path / "bytes.artifact").write_bytes(b"old corrupt copy")
    monkeypatch.setattr(comparison.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong download"))
    with pytest.raises(ValueError, match="cache was not replaced"):
        comparison.fetch(spec, tmp_path)
    assert (tmp_path / "bytes.artifact").read_bytes() == b"old corrupt copy"
    monkeypatch.setattr(comparison.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(original))
    comparison.fetch(spec, tmp_path)
    def no_network(*args, **kwargs):
        pytest.fail("A verified cache must be reused")
    monkeypatch.setattr(comparison.urllib.request, "urlopen", no_network)
    comparison.fetch(spec, tmp_path)
    assert comparison.verified_bytes(spec, tmp_path) == original


def test_byte_tokens_preserve_unicode_and_do_not_count_id_holes(tmp_path):
    engine = comparison.Engine(byte_fixture(tmp_path), tmp_path)
    assert engine.metadata["vocabulary_entries"] == 257
    assert engine.metadata["highest_token_id_plus_one"] == 301
    observed = engine.observe("β ")
    assert observed["input_codepoints"] == 2 and observed["input_utf8_bytes"] == 3
    assert observed["token_ids"] == [206, 178, 32]
    assert observed["token_pieces"][:2] == ["b'\\xce'", "b'\\xb2'"]
    assert observed["decoded_text"] == "β " and observed["round_trip_exact"]
    assert 300 not in engine.observe("<|end|>")["token_ids"]
    empty = engine.observe("")
    assert empty["token_ids"] == [] and empty["token_count"] == 0
    assert empty["input_bytes_per_token"] is None and empty["round_trip_exact"]


def test_huggingface_pipeline_normalizes_and_disables_wrappers_padding_truncation(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece(
        {"[UNK]": 0, "[SEP]": 1, "hello": 2, "cafe": 3, "[": 4, "]": 5, "sep": 6},
        unk_token="[UNK]",
    ))
    tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = tokenizers.decoders.WordPiece()
    tokenizer.add_special_tokens(["[SEP]"])
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[SEP] $A [SEP]", special_tokens=[("[SEP]", 1)],
    )
    tokenizer.enable_padding(length=10, pad_id=1, pad_token="[SEP]")
    tokenizer.enable_truncation(max_length=3)
    content = tokenizer.to_str().encode()
    spec = {"id": "fixture", "backend": "huggingface", "sha256": hashlib.sha256(content).hexdigest()}
    (tmp_path / "fixture.artifact").write_bytes(content)
    engine = comparison.Engine(spec, tmp_path)
    observed = engine.observe("HELLO CAFÉ")
    assert observed["normalized_text"] == "hello cafe"
    assert observed["token_ids"] == [2, 3]  # no padding, wrappers, or truncation to one content token
    assert observed["decoded_text"] == "hello cafe" and not observed["round_trip_exact"]
    assert engine.observe("[SEP]")["token_ids"] == [4, 6, 5]
    assert not engine.observe("🙂")["round_trip_exact"]
    # Added special tokens may already be entries in the base vocabulary.
    assert engine.metadata["base_vocabulary_entries"] == engine.metadata["vocabulary_entries"] == 7
    assert engine.metadata["special_token_entries"] == 1


def test_measure_preserves_predictions_and_check_rejects_self_consistent_wrong_results(tmp_path, monkeypatch):
    specs = [byte_fixture(tmp_path, name) for name in ("first", "second")]
    data = {"tokenizers": specs, "runtime_versions": {}}
    monkeypatch.setattr(comparison, "manifest", lambda: data)
    record = contribution([spec["id"] for spec in specs])
    predictions = [case["prediction"] for case in record["cases"]]
    notes = record["notes"]
    path = tmp_path / "alice.json"
    comparison.write_json(path, record)
    monkeypatch.setattr(sys, "argv", ["runner", "--cache", str(tmp_path), "measure", str(path)])
    comparison.main()
    measured = comparison.read_json(path)
    assert [case["prediction"] for case in measured["cases"]] == predictions
    assert measured["notes"] == notes
    comparison.check_contribution(measured, data, tmp_path)
    wrong = copy.deepcopy(measured)
    observation = wrong["cases"][0]["observed"]["first"]
    observation.update(token_ids=[], token_pieces=[], token_count=0, input_bytes_per_token=None,
                       decoded_text="", round_trip_exact=False)
    with pytest.raises(ValueError, match="independent re-encoding"):
        comparison.check_contribution(wrong, data, tmp_path)
    measured["provenance"]["artifacts"]["first"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Provenance"):
        comparison.check_contribution(measured, data, tmp_path)


@pytest.mark.parametrize("change, message", [
    (lambda r: r["tokenizers"].__setitem__(1, r["tokenizers"][0]), "distinct tokenizer"),
    (lambda r: r["cases"][0].update(text="changed"), "shared case"),
    (lambda r: r["cases"][-1].update(text=r["cases"][0]["text"]), "distinct strings"),
    (lambda r: r["cases"][-1].update(text=""), "original input"),
    (lambda r: r["cases"][0].update(prediction=""), "prediction"),
    (lambda r: r.update(contributors=["Alice"]), "lowercase GitHub"),
])
def test_invalid_contributions_are_rejected(change, message):
    record = contribution(["r50k_base", "qwen3"])
    change(record)
    with pytest.raises(ValueError, match=message):
        comparison.validate_contribution(record, comparison.manifest())


def test_runtime_drift_is_explicit():
    with pytest.raises(ValueError, match="expected.*uv sync --locked"):
        comparison.check_runtime({"runtime_versions": {"tiktoken": "wrong-version"}})


@pytest.mark.skipif(os.environ.get("TOKENIZER_COMPARISON_INTEGRATION") != "1",
                    reason="Opt-in: fetch the six real artifacts before running")
def test_pinned_pilot_matches_worked_example_without_network(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("The downloaded pilot must work offline")
    monkeypatch.setattr(comparison.urllib.request, "urlopen", no_network)
    data = comparison.manifest()
    comparison.check_runtime(data)
    engines = {spec["id"]: comparison.Engine(spec, comparison.DEFAULT_CACHE) for spec in data["tokenizers"]}
    example = comparison.read_json(comparison.TASK / "worked-example.json")
    assert example["provenance"] == comparison.provenance(data, data["tokenizers"])
    for name, expected in example["vocabulary_entries"].items():
        assert engines[name].metadata["vocabulary_entries"] == expected
    for case in example["cases"]:
        for name, expected in case["observed"].items():
            actual = engines[name].observe(case["text"])
            assert {key: actual[key] for key in expected} == expected
    for case in comparison.read_json(comparison.TASK / "inputs.json"):
        for engine in engines.values():
            observed = engine.observe(case["text"])
            assert observed["token_count"] == len(observed["token_ids"]) == len(observed["token_pieces"])
