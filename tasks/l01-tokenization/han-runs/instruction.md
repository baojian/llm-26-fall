# Keep Chinese character runs together

**Lecture:** l01-tokenization · **Difficulty:** medium · **Time:** about 40 minutes ·
**Deadline:** Tuesday, September 22, 23:59 (Asia/Shanghai)

## Goal

Write the chunking rule that keeps a run of Chinese characters in one chunk and never lets it merge with Latin letters or digits, as Kimi K2's tokenizer does.

## What you submit

One file: `submissions/<your-github-username>.py` containing a function

```python
def solve(text: str) -> list[str]:
    ...
```

Input: any string. Output: the chunks in order, concatenating back to the input. Each chunk is a maximal run of one kind: (a) Chinese characters in the CJK Unified Ideographs block `U+4E00`–`U+9FFF`; (b) ASCII letters or digits `[A-Za-z0-9]`; (c) whitespace; (d) anything else (punctuation, symbols, other scripts). An empty string gives an empty list.

## Examples

| Input | Output |
| :--- | :--- |
| `"用GPT写代码"` | `["用", "GPT", "写代码"]` |
| `"你好, world"` | `["你好", ",", " ", "world"]` |
| `""` | `[]` |

## Before you code: predict, then break it

Your file also contains three things the checker reads:

```python
PREDICTIONS = {  # write these BEFORE writing solve; the checker compares solve to them
    'Kimi K2 的 tokenizer': [...],
    '2026年9月16日': [...],
    'GPT-4o很强': [...],
}
MY_CASES = [  # two inputs of your own where a naive solution fails, with the expected output
    ("...", [...]),
    ("...", [...]),
]
NOTES = """
Answer in 3–5 sentences: Without this rule, BPE could learn a merge that spans a Chinese character and a Latin letter (for example the end of `用` with `G` in `用GPT`). What would such merged tokens do to the vocabulary and to the model's handling of mixed-script text, and why does Kimi K2 make the Han run the *first* alternative in its pattern?
"""
```

The three prediction inputs are: `'Kimi K2 的 tokenizer'`, `'2026年9月16日'`, `'GPT-4o很强'`.

## How it is checked

`tests/test_task.py` calls your `solve` on the examples above and on a few
similar cases, checks that `solve` agrees with your `PREDICTIONS`, that
`MY_CASES` are new and pass, and that `NOTES` is not empty. Run it yourself
before opening the PR:

```sh
uv run python scripts/tasks.py check tasks/l01-tokenization/han-runs <username>
```

## Rules

- Standard library only. Python's `re` has no `\p{Han}`; use the code-point range `[\u4e00-\u9fff]`.
- Use your own words and code; discussing the approach with classmates is fine.
- PR title `l01-tokenization/han-runs: <username>`, body `Related to #<issue>`.

## Why this matters

Lecture 01 showed that a visible character can be several bytes and that merges are only allowed inside a chunk. For Chinese text the chunking rule decides whether the model's vocabulary spends entries on cross-script fragments. Kimi K2's pattern starts with `[\p{Han}]+` for exactly this reason (docs/regex-in-llm-training.md, Section 1).
