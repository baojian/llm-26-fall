"""Test artifact integrity, independent checking, and tokenizer boundary cases.

The normal suite uses tiny local fixtures and never downloads artifacts.
Set TOKENIZER_COMPARISON_INTEGRATION=1 after fetch to verify the six real files.
"""

import base64
import hashlib
import io
import json
import os
import importlib.util
from pathlib import Path

import pytest

TOPIC = Path(__file__).resolve().parents[1] / "explorations/tokenizer-comparison"
spec = importlib.util.spec_from_file_location("tokenizer_comparison", TOPIC / "run.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def byte_fixture(tmp_path, name="bytes"):
    content = b"".join(base64.b64encode(bytes([value])) + f" {value}\n".encode() for value in range(256))
    (tmp_path / f"{name}.artifact").write_bytes(content)
    return {
        "id": name, "backend": "tiktoken", "url": "https://example.invalid/tokenizer",
        "sha256": hashlib.sha256(content).hexdigest(), "revision": "test-fixture",
        "pattern": r"\S+|\s+", "special_tokens": {"<|end|>": 300},
    }


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
    unknown = engine.observe("🙂")
    assert unknown["fidelity"] == "unknown-tokens"
    assert unknown["unknown_token_count"] == 1
    assert observed["fidelity"] == "text-changed"
    # Added special tokens may already be entries in the base vocabulary.
    assert engine.metadata["base_vocabulary_entries"] == engine.metadata["vocabulary_entries"] == 7
    assert engine.metadata["special_token_entries"] == 1


def test_runtime_drift_is_explicit():
    with pytest.raises(ValueError, match="expected.*uv sync --locked"):
        comparison.check_runtime({"runtime_versions": {"tiktoken": "wrong-version"}})


def test_canonical_normalization_is_distinguished_from_unknown_content(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"café": 0, "[UNK]": 1}, unk_token="[UNK]"))
    tokenizer.normalizer = tokenizers.normalizers.NFC()
    content = tokenizer.to_str().encode()
    (tmp_path / "nfc.artifact").write_bytes(content)
    engine = comparison.Engine({"id": "nfc", "backend": "huggingface",
                                "sha256": hashlib.sha256(content).hexdigest()}, tmp_path)
    composed = engine.observe("café")
    decomposed = engine.observe("cafe\u0301")
    assert composed["fidelity"] == "exact"
    assert decomposed["fidelity"] == "nfc-equivalent"
    assert decomposed["decoded_text"] == "café"
    assert decomposed["unknown_token_count"] == 0
    assert engine.observe("unseen")["fidelity"] == "unknown-tokens"


@pytest.mark.skipif(os.environ.get("TOKENIZER_COMPARISON_INTEGRATION") != "1",
                    reason="Opt-in: fetch the six real artifacts before running")
def test_pinned_pilot_matches_worked_example_without_network(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("The downloaded pilot must work offline")
    monkeypatch.setattr(comparison.urllib.request, "urlopen", no_network)
    data = comparison.manifest()
    comparison.check_runtime(data)
    engines = {spec["id"]: comparison.Engine(spec, comparison.DEFAULT_CACHE) for spec in data["tokenizers"]}
    expected_counts = {
        "digits": [5, 4, 4, 10, 7, 5],
        "mixed-script": [19, 10, 6, 6, 9, 7],
        "normalization": [6, 4, 4, 3, 2, 4],
        "emoji": [12, 12, 8, 6, 1, 4],
    }
    assert [e.metadata["vocabulary_entries"] for e in engines.values()] == [50257, 100261, 200000, 151669, 30522, 32100]
    cases = comparison.load_cases()
    assert len(cases) == 32
    report = comparison.compare(data, data["tokenizers"], cases, comparison.DEFAULT_CACHE)
    for case in report["cases"]:
        observations = case["observed"]
        if case["id"] in expected_counts:
            assert [v["token_count"] for v in observations.values()] == expected_counts[case["id"]]
        for observed in observations.values():
            assert observed["token_count"] == len(observed["token_ids"]) == len(observed["token_pieces"])
        if case["id"] == "normalization":
            assert [v["fidelity"] for v in observations.values()] == ["exact", "exact", "exact", "nfc-equivalent", "text-changed", "nfc-equivalent"]
        if case["id"] == "emoji":
            assert observations["bert"]["unknown_token_count"] == 1
            assert observations["bert"]["fidelity"] == "unknown-tokens"
            assert observations["t5"]["unknown_token_count"] == 2
        if case["id"] == "empty":
            assert all(v["fidelity"] == "exact" and v["input_bytes_per_token"] is None for v in observations.values())
        if case["id"] == "literal-special":
            assert 50256 not in observations["r50k_base"]["token_ids"]
    assert len(report["provenance"]["inputs_sha256"]) == 64
