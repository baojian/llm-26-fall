"""Regression checks for supplied data loading across checkout line endings."""
import hashlib
import json
from pathlib import Path

import pytest

import support


DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def data_copy(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name in ("manifest.json", *manifest["files"]):
        content = (DATA / name).read_bytes().replace(b"\r\n", b"\n")
        (data / name).write_bytes(content)
    monkeypatch.setattr(support, "HERE", tmp_path)
    return data


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_bundled_data_loads_with_checkout_line_endings(data_copy, newline):
    for path in data_copy.iterdir():
        path.write_bytes(path.read_bytes().replace(b"\n", newline))
    manifest, tokenizer, documents = support.load_data(("train", "dev", "test"))
    assert tokenizer.get_vocab_size() == manifest["tokenizer"]["vocab_size"]
    for split, rows in documents.items():
        expected = [json.loads(line) for line in
                    (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows == expected
        assert len(rows) == manifest["files"][f"{split}.jsonl"]["documents"]
    assert len({row["text"] for row in documents["train"]}) == manifest["unique_train_documents"]


@pytest.mark.parametrize("filename", ["train.jsonl", "dev.jsonl", "test.jsonl", "tokenizer.json"])
def test_data_content_change_is_rejected(data_copy, filename):
    path = data_copy / filename
    path.write_bytes(path.read_bytes().replace(b'"', b"'", 1))
    with pytest.raises(ValueError, match=f"data/tokenizer checksum mismatch: {filename}"):
        support.load_data()


def test_binary_checksum_preserves_raw_bytes(tmp_path):
    path = tmp_path / "checkpoint.pt"
    content = b"\x00\xff\r\n\x80\n"
    path.write_bytes(content)
    assert support.sha256(path) == hashlib.sha256(content).hexdigest()
    assert support.sha256(path) != hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def test_text_checksum_normalizes_only_crlf(tmp_path):
    path = tmp_path / "manifest.json"
    canonical = '{"text": "caf\u00e9"}\n'.encode("utf-8")
    path.write_bytes(canonical.replace(b"\n", b"\r\n"))
    assert support.text_sha256(path) == hashlib.sha256(canonical).hexdigest()
    for changed in (canonical + b" ", canonical.replace(b"\n", b"\r"),
                    canonical.replace(b"text", b"other")):
        path.write_bytes(changed)
        assert support.text_sha256(path) != hashlib.sha256(canonical).hexdigest()
