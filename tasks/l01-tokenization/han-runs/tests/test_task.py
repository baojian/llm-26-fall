"""Public Han-runs checks for every file in submissions/.

Run one student:  uv run python scripts/tasks.py check tasks/l01-tokenization/han-runs <username>
Run everyone:     uv run python scripts/tasks.py check tasks/l01-tokenization/han-runs
"""

import importlib.util
import json
from pathlib import Path

import pytest

FUNCTION = "solve"
DATA = Path(__file__).resolve().parents[1] / "data"
CASES = [
    # (input, expected output)
    ('用GPT写代码', ['用', 'GPT', '写代码']),
    ('你好, world', ['你好', ',', ' ', 'world']),
    ('', []),
    ('复旦大学 NLP 2026', ['复旦大学', ' ', 'NLP', ' ', '2026']),
    ('。。。', ['。。。']),
    ('K2中文2026', ['K2', '中文', '2026']),
    ('GPT4', ['GPT4']),
    ('\t \n', ['\t \n']),
    ('a\u00a0\u3000b', ['a', '\u00a0\u3000', 'b']),
    ('Ａ１é🙂', ['Ａ１é🙂']),
    ('㐀𠀀中', ['㐀𠀀', '中']),
    ('\u4dff\u4e00\u9fff\ua000', ['\u4dff', '\u4e00\u9fff', '\ua000']),
    ('中_文', ['中', '_', '文']),
    ('A_!?🙂é中', ['A', '_!?🙂é', '中']),
    ('Kimi K2 的 tokenizer', ['Kimi', ' ', 'K2', ' ', '的', ' ', 'tokenizer']),
    ('2026年9月16日', ['2026', '年', '9', '月', '16', '日']),
    ('GPT-4o很强', ['GPT', '-', '4o', '很强']),
    (DATA.joinpath('sample.txt').read_text(encoding='utf-8'),
     json.loads(DATA.joinpath('sample-chunks.json').read_text(encoding='utf-8'))),
]
PREDICTION_INPUTS = ['Kimi K2 的 tokenizer', '2026年9月16日', 'GPT-4o很强']  # listed in instruction.md; students predict these
MIN_OWN_CASES = 2
MIN_NOTES_CHARS = 200

SUBMISSIONS = sorted(p for p in (Path(__file__).resolve().parents[1] / "submissions").glob("*.py") if not p.name.startswith((".", "_")))


def test_raw_text_fixture_preserves_every_character():
    text, expected = CASES[-1]
    assert text.endswith("\n")
    assert "".join(expected) == text


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path):
    return getattr(load_module(path), FUNCTION)


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_filename_is_a_lowercase_username(submission):
    assert submission.stem == submission.stem.lower(), "name the file <username>.py in lowercase"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
@pytest.mark.parametrize("text,expected", CASES)
def test_cases(submission, text, expected):
    assert load(submission)(text) == expected


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_predictions_match_solve(submission):
    module = load_module(submission)
    predictions = getattr(module, "PREDICTIONS", {})
    assert set(predictions) == set(PREDICTION_INPUTS), "PREDICTIONS must cover exactly the three inputs in instruction.md"
    for text, expected in predictions.items():
        assert getattr(module, FUNCTION)(text) == expected, f"solve({text!r}) differs from your prediction"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_own_cases_are_new_and_pass(submission):
    module = load_module(submission)
    own = getattr(module, "MY_CASES", [])
    assert len(own) >= MIN_OWN_CASES, f"add at least {MIN_OWN_CASES} cases of your own to MY_CASES"
    assert len({text for text, _ in own}) >= MIN_OWN_CASES, "MY_CASES must contain at least two distinct inputs"
    given = {text for text, _ in CASES} | set(PREDICTION_INPUTS)
    for text, expected in own:
        assert text not in given, f"{text!r} is one of the given inputs; find a new one"
        assert getattr(module, FUNCTION)(text) == expected, f"solve({text!r}) fails your own case"


@pytest.mark.parametrize("submission", SUBMISSIONS, ids=[f"user.{p.stem}" for p in SUBMISSIONS])
def test_notes_answer_the_question(submission):
    notes = getattr(load_module(submission), "NOTES", "")
    assert len(notes.strip()) >= MIN_NOTES_CHARS and "REPLACE" not in notes, "write 3–5 sentences in NOTES"
