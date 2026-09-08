"""Check the imported notebook's navigation and offline teaching examples."""

import ast
import json
import re
import time
from pathlib import Path

import nbformat
import pytest
from IPython.core.inputtransformer2 import TransformerManager


LECTURE = Path(__file__).resolve().parents[1] / "slides/lecture-01"
NOTEBOOK = json.loads((LECTURE / "lecture-01-exercise-tokenization.ipynb").read_text())


def test_notebook_navigation_and_distribution():
    nbformat.validate(nbformat.from_dict(NOTEBOOK))
    source = "\n".join("".join(cell["source"]) for cell in NOTEBOOK["cells"])
    assert not re.search(r"(?:minimum|minimal)[ _-]edit[ _-]distance|test_med|med-example|#5[12]", source, re.I)
    anchors = set(re.findall(r'<a id="([^"]+)"', source))
    targets = set(re.findall(r"\]\(#([^)]*)\)", source))
    assert targets <= anchors
    assert {"00", "10", "20", "30", "40", "50", "60"} <= anchors
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
    for asset in re.findall(r'<img[^>]+src="([^"]+)"', source):
        assert (LECTURE / asset).is_file()


def test_code_cells_support_the_course_python_minimum():
    transform = TransformerManager().transform_cell
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(transform("".join(cell["source"])), filename=cell["id"], feature_version=(3, 11))


def test_unicode_and_regex_examples_run_offline(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    monkeypatch.chdir(tmp_path)
    namespace = {"re": re}
    in_section = False
    executed = 0
    for cell in NOTEBOOK["cells"]:
        source = "".join(cell["source"])
        if '<a id="21">' in source:
            in_section = True
        if '<a id="23">' in source:
            break
        if in_section and cell["cell_type"] == "code":
            exec(compile(source, cell["id"], "exec"), namespace)
            executed += 1
    assert executed >= 15
    assert namespace["decoded_string"] == "Hello, World! 你好，世界！"
    assert (tmp_path / "unicode_file.txt").read_text() == "Some text with a special character: é, 你好，世界！"
    assert namespace["word_re"].findall("don't 2011 real-time Senjō") == ["don't", "real", "time", "Senj"]


def test_bpe_and_wordpiece_training_exclude_heldout_text(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets").mkdir()

    class TrainingRows:
        texts = ["hello world", "hello tokenizer"]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, key):
            return {"text": self.texts[key]}

    class TrainingOnly(dict):
        def __getitem__(self, key):
            assert key == "train", "Tokenizer training accessed a held-out split"
            return TrainingRows()

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    namespace = {
        "time": time, "tokenizer": tokenizer,
        "BpeTrainer": tokenizers.trainers.BpeTrainer, "wiki103_ds": TrainingOnly()
    }
    training_cells = [cell for cell in NOTEBOOK["cells"]
                      if cell["cell_type"] == "code" and "def batch_iterator" in "".join(cell["source"])]
    assert len(training_cells) == 2
    for cell in training_cells:
        exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    assert tokenizer.encode("hello").tokens == ["hello"]
    assert namespace["bert_tokenizer"].encode("hello").tokens == ["[CLS]", "hello", "[SEP]"]
    restored = tokenizers.Tokenizer.from_file(str(tmp_path / "assets/wikitext103-bert.json"))
    assert restored.encode("hello").ids == namespace["bert_tokenizer"].encode("hello").ids
