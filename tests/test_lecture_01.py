"""Check the executable teaching material directly from the distributed notebook."""

import base64
import builtins
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import nbformat
import pytest


LECTURE = Path(__file__).resolve().parents[1] / "slides/lecture-01"
NOTEBOOK = json.loads((LECTURE / "lecture-01-exercise.ipynb").read_text())


def execute_cell(cell, namespace):
    exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)


@pytest.fixture
def lesson(monkeypatch):
    monkeypatch.setenv("COURSE_RUN_OLLAMA", "0")
    namespace = {}
    for cell in NOTEBOOK["cells"]:
        tags = cell["metadata"].get("tags", [])
        if cell["cell_type"] == "code" and set(tags) & {"definitions", "bpe-definition"}:
            execute_cell(cell, namespace)
    return namespace


def test_notebook_runs_offline_without_network(monkeypatch, capsys):
    def no_network(*args, **kwargs):
        pytest.fail("The offline notebook attempted a network request")

    original_import = builtins.__import__

    def no_model_imports(name, *args, **kwargs):
        if name in {"torch", "diffusers"}:
            pytest.fail("The offline notebook imported an optional model package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("COURSE_RUN_OLLAMA", "0")
    monkeypatch.setattr("urllib.request.OpenerDirector.open", no_network)
    monkeypatch.setattr(builtins, "__import__", no_model_imports)
    nbformat.validate(nbformat.from_dict(NOTEBOOK))
    namespace = {}
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            execute_cell(cell, namespace)
    assert namespace["OLLAMA_READY"] is False
    figure = json.loads((LECTURE / "assets/sequence-length.json").read_text())
    assert namespace["totals"] == figure["data"][0]["y"]
    assert len(namespace["comparison_rows"]) == 12
    assert all(row["roundtrip"] for row in namespace["comparison_rows"])
    heldout_figure = json.loads((LECTURE / "assets/heldout-chinese.json").read_text())
    for trace, training in zip(heldout_figure["data"], ["EN + ZH", "EN only"]):
        rows = [row for row in namespace["comparison_rows"]
                if row["training"] == training and row["language"] == "Chinese"]
        assert trace["x"] == [row["learned"] for row in rows]
        assert trace["y"] == [row["tokens"] for row in rows]
    assert "NFC can make these strings equal" in capsys.readouterr().out


def test_displayed_python_examples_run(capsys):
    source = (LECTURE / "slides.md").read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", source, re.DOTALL)
    assert blocks
    for index, block in enumerate(blocks):
        exec(compile(block, f"slides.md Python block {index + 1}", "exec"), {})
    assert "你好 2 6" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["", "lowest", "你好🙂", "e\u0301", "\n\t  ", "a\x00b", "<|endoftext|>"])
def test_unseen_text_roundtrips_without_changing_vocabulary(lesson, text):
    vocab, merges, _ = lesson["train_bpe"]([("low", 5), ("lower", 2)], 2)
    tokenizer = lesson["ByteBPETokenizer"](vocab, merges)
    before = dict(tokenizer.vocab)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab == before
    assert len(tokenizer.vocab) == 258


def test_pair_counts_include_overlap_but_replacements_do_not(lesson):
    counts = lesson["count_pairs"]([([97, 97, 97, 98], 2), ([97, 98], 1)])
    assert counts == {(97, 97): 4, (97, 98): 3}
    vocab, merges, totals = lesson["train_bpe"]([("aaab", 2), ("ab", 1)], 1)
    tokenizer = lesson["ByteBPETokenizer"](vocab, merges)
    assert tokenizer.pieces("aaab") == [b"aa", b"a", b"b"]
    assert totals == [10, 8]


def test_tie_breaking_is_independent_of_corpus_order(lesson):
    first = lesson["train_bpe"]([("low", 5), ("lower", 2)], 2)
    second = lesson["train_bpe"]([("lower", 2), ("low", 5)], 2)
    assert first == second
    assert [first[0][index] for _, index in first[1]] == [b"lo", b"low"]


def test_encoding_obeys_learned_rank_not_leftmost_position(lesson):
    vocab, merges, _ = lesson["train_bpe"]([("bc", 3), ("ab", 2)], 2)
    tokenizer = lesson["ByteBPETokenizer"](vocab, merges)
    assert tokenizer.pieces("abc") == [b"a", b"bc"]


@pytest.mark.parametrize("corpus", [[], [("", 1)], [("a", 2), ("b", 3)]])
def test_no_pairs_stops_without_merging_document_boundaries(lesson, corpus):
    vocab, merges, totals = lesson["train_bpe"](corpus, 100)
    assert len(vocab) == 256
    assert merges == []
    assert len(totals) == 1


@pytest.mark.parametrize("corpus,budget", [([("x", 0)], 2), ([("x", -1)], 2),
                                          ([("x", 1.5)], 2), ([(b"x", 1)], 2),
                                          ([("x", 1)], -1), ([("x", 1)], 1.5)])
def test_invalid_training_inputs_are_rejected(lesson, corpus, budget):
    with pytest.raises(ValueError):
        lesson["train_bpe"](corpus, budget)


def test_decoding_incomplete_utf8_fails_explicitly(lesson):
    vocab, merges, _ = lesson["train_bpe"]([], 0)
    tokenizer = lesson["ByteBPETokenizer"](vocab, merges)
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([228])
    with pytest.raises(KeyError):
        tokenizer.decode([256])


@pytest.mark.parametrize("corpus,budget", [
    ([("low", 5), ("lower", 2)], 2),
    ([("aaab", 2), ("ab", 1)], 1),
    ([("aaaaa", 3)], 4),
    ([("你好🙂", 2), ("你好", 1)], 5),
    ([("a", 1), ("b", 1)], 10),
    ([("a b\n", 1)], 4),
    ([], 3),
    ([("abc", 1)], 0),
])
def test_browser_bpe_trace_matches_notebook_at_every_step(lesson, corpus, budget):
    script = """
        import { buildBpeTrace } from './slides/lecture-01/demo.js';
        let input = '';
        for await (const chunk of process.stdin) input += chunk;
        const { corpus, budget } = JSON.parse(input);
        const trace = buildBpeTrace(corpus, budget);
        console.log(JSON.stringify(trace.map(state => ({
            total: state.total,
            ids: state.sequences.map(sequence => sequence.ids),
            vocabulary: state.vocabulary,
        }))));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps({"corpus": corpus, "budget": budget}),
        capture_output=True, text=True, check=True, cwd=LECTURE.parents[1],
    )
    trace = json.loads(result.stdout)
    _, _, all_totals = lesson["train_bpe"](corpus, budget)
    assert [state["total"] for state in trace] == all_totals
    for step, state in enumerate(trace):
        vocab, merges, _ = lesson["train_bpe"](corpus, step)
        tokenizer = lesson["ByteBPETokenizer"](vocab, merges)
        assert state["ids"] == [tokenizer.encode(text) for text, _ in corpus]
        assert state["vocabulary"] == [list(vocab[index]) for index in range(len(vocab))]


def test_ollama_client_uses_local_json_request(lesson, monkeypatch):
    requests = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(json.dumps({"response": "Shanghai"}).encode())

    monkeypatch.setattr(lesson["local_http"], "open", open_request)
    payload = {"model": "qwen3:0.6b", "prompt": "你好", "stream": False}
    result = lesson["ollama_request"]("/api/generate", payload)
    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:11434/api/generate"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == payload
    assert timeout == 120
    assert result["response"] == "Shanghai"


@pytest.mark.parametrize("failure,expected", [
    (URLError("connection refused"), "Cannot reach Ollama"),
    (TimeoutError(), "timed out"),
    (HTTPError("http://127.0.0.1:11434", 404, "missing", {},
               io.BytesIO(b'{"error":"model not found"}')), "model not found"),
])
def test_ollama_failures_report_actionable_errors(lesson, monkeypatch, failure, expected):
    def open_request(*args, **kwargs):
        raise failure

    monkeypatch.setattr(lesson["local_http"], "open", open_request)
    with pytest.raises(RuntimeError, match=expected):
        lesson["ollama_request"]("/api/tags")


def test_generation_keeps_thinking_options_at_the_api_top_level(lesson, monkeypatch):
    payloads = []

    def request(endpoint, payload):
        payloads.append(payload)
        return {"response": "Shanghai", "done_reason": "stop"}

    monkeypatch.setitem(lesson, "ollama_request", request)
    lesson["OLLAMA_READY"] = True
    lesson["generate"]("A prompt", think=True, logprobs=True, max_tokens=20)
    assert payloads[0]["think"] is True
    assert payloads[0]["logprobs"] is True
    assert payloads[0]["stream"] is False
    assert payloads[0]["options"]["num_predict"] == 20
    assert "think" not in payloads[0]["options"]


def test_vision_request_sends_image_bytes_in_the_user_message(lesson):
    image_bytes = b"test image payload"
    payload = lesson["vision_payload"](image_bytes, "installed-vision-model")
    message = payload["messages"][0]
    assert message["role"] == "user"
    assert base64.b64decode(message["images"][0]) == image_bytes
    assert payload["model"] == "installed-vision-model"
    assert payload["stream"] is False
    assert "think" not in payload
    assert "think" not in payload["options"]
    assert payload["options"] == {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "seed": 42, "num_predict": 2048,
    }


def test_vision_request_preserves_the_selected_question(lesson):
    prompt = "Describe the image in one paragraph."
    payload = lesson["vision_payload"](b"image", "installed-vision-model", prompt)
    assert payload["messages"][0]["content"] == prompt


@pytest.mark.parametrize("example,filename,question", [
    ("rainfall", "vision-rainfall.jpg", "axes, red line, and green bar"),
    ("big data", "vision-big-data.png", "central label, surrounding logos"),
])
def test_vision_example_selects_matching_image_and_question(lesson, monkeypatch, example, filename, question):
    requests = []

    def request(endpoint, payload=None, timeout=120):
        requests.append((endpoint, payload))
        if endpoint == "/api/tags":
            return {"models": [{"name": "qwen3-vl:2b"}]}
        if endpoint == "/api/show":
            return {"capabilities": ["vision"]}
        if endpoint == "/api/chat":
            assert timeout == 300
            return {"message": {"content": "A mocked image description."}}
        pytest.fail(f"Unexpected Ollama request: {endpoint}")

    monkeypatch.chdir(LECTURE)
    lesson["RUN_OLLAMA"] = True
    monkeypatch.setitem(lesson, "ollama_request", request)
    cell = next(cell for cell in NOTEBOOK["cells"] if cell["id"] == "vision")
    code = "".join(cell["source"]).replace("RUN_VISION = False", "RUN_VISION = True")
    code = code.replace('VISION_EXAMPLE = "rainfall"', f'VISION_EXAMPLE = "{example}"')
    execute_cell({**cell, "source": [code]}, lesson)

    assert [endpoint for endpoint, _ in requests] == ["/api/tags", "/api/show", "/api/chat"]
    payload = requests[-1][1]
    assert payload["model"] == "qwen3-vl:2b"
    message = payload["messages"][0]
    assert question in message["content"]
    if example == "big data":
        assert "axes" not in message["content"]
    assert base64.b64decode(message["images"][0]) == (LECTURE / "assets" / filename).read_bytes()


@pytest.mark.parametrize("answer", ["", "A partial chart description."])
def test_vision_reports_truncation_without_retrying(lesson, monkeypatch, capsys, answer):
    requests = []

    def request(endpoint, payload=None, timeout=120):
        requests.append(endpoint)
        if endpoint == "/api/tags":
            return {"models": [{"name": "qwen3-vl:2b"}]}
        if endpoint == "/api/show":
            return {"capabilities": ["vision", "thinking"]}
        if endpoint == "/api/chat":
            assert timeout == 300
            return {"message": {"content": answer}, "done_reason": "length"}
        pytest.fail(f"Unexpected Ollama request: {endpoint}")

    monkeypatch.chdir(LECTURE)
    lesson["RUN_OLLAMA"] = True
    monkeypatch.setitem(lesson, "ollama_request", request)
    cell = next(cell for cell in NOTEBOOK["cells"] if cell["id"] == "vision")
    code = "".join(cell["source"]).replace("RUN_VISION = False", "RUN_VISION = True")
    execute_cell({**cell, "source": [code]}, lesson)

    output = capsys.readouterr().out
    assert "Generation reached the token limit" in output
    assert "rerun both vision code cells" in output
    if answer:
        assert answer in output
        assert "No visible answer" not in output
    else:
        assert "No visible answer" in output
    assert requests.count("/api/chat") == 1


@pytest.mark.parametrize("device,dtype", [("cpu", "float32"), ("mps", "float32"),
                                          ("cuda", "float16")])
def test_diffusion_uses_local_weights_and_retains_pipeline_defaults(lesson, monkeypatch, device, dtype):
    calls = {}
    expected_image = object()

    class Generator:
        def __init__(self, device):
            calls["generator_device"] = device

        def manual_seed(self, seed):
            calls["seed"] = seed
            return self

    class Pipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls["model"] = model
            calls["load"] = kwargs
            return cls()

        def to(self, device):
            calls["device"] = device
            return self

        def __call__(self, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(images=[expected_image])

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float16="float16", float32="float32", Generator=Generator,
    ))
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(DiffusionPipeline=Pipeline))
    image = lesson["generate_local_image"]("cached-model", "A campus gate", device=device)

    assert image is expected_image
    assert calls["model"] == "cached-model"
    assert calls["load"] == {"local_files_only": True, "use_safetensors": True, "dtype": dtype}
    assert calls["device"] == device
    assert calls["generator_device"] == "cpu"
    assert calls["seed"] == 42
    assert calls["generate"]["prompt"] == "A campus gate"
    assert calls["generate"]["num_inference_steps"] == 30


def test_missing_diffusion_cache_has_no_download_fallback(lesson, monkeypatch):
    attempts = []

    class MissingPipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            attempts.append(kwargs)
            raise OSError("Model files are absent from the local cache")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16", float32="float32"))
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(DiffusionPipeline=MissingPipeline))
    with pytest.raises(OSError, match="local cache"):
        lesson["generate_local_image"]("missing-model", "A campus gate")
    assert len(attempts) == 1
    assert attempts[0]["local_files_only"] is True
