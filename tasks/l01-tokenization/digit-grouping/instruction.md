# Group digits like GPT-4 before BPE

**Lecture:** l01-tokenization · **Difficulty:** easy · **Time:** about 30 minutes ·
**Deadline:** Tuesday, September 22, 23:59 (Asia/Shanghai)

## Goal

Apply GPT-4's digit rule, at most three digits per chunk, and explain what it changes for numbers.

## What you submit

One file: `submissions/<your-github-username>.py` containing a function

```python
def solve(text: str) -> list[str]:
    ...
```

Input: any string. Output: the chunks in order, concatenating back to the input. Every maximal run of digits is cut into groups of at most three from the left (`2026` becomes `202`, `6`); every maximal run of non-digits is one chunk, unchanged. An empty string gives an empty list.

## Examples

| Input | Output |
| :--- | :--- |
| `"2026"` | `["202", "6"]` |
| `"Year 2026: 1,000,000"` | `["Year ", "202", "6", ": ", "1", ",", "000", ",", "000"]` |
| `""` | `[]` |

## Before you code: predict, then break it

Your file also contains three things the checker reads:

```python
PREDICTIONS = {  # write these BEFORE writing solve; the checker compares solve to them
    '3.14159': [...],
    'Room 101': [...],
    '2024-09-16': [...],
}
MY_CASES = [  # two inputs of your own where a naive solution fails, with the expected output
    ("...", [...]),
    ("...", [...]),
]
NOTES = """
Answer in 3–5 sentences: GPT-4's `cl100k_base` groups digits in threes (`\p{N}{1,3}`); Qwen's tokenizer takes one digit at a time (`\p{N}`). For a table of years and for a list of prices, which rule produces fewer tokens, and why might a model builder still prefer the one-digit rule?
"""
```

The three prediction inputs are: `'3.14159'`, `'Room 101'`, `'2024-09-16'`.

## How it is checked

`tests/test_task.py` calls your `solve` on the examples above and on a few
similar cases, checks that `solve` agrees with your `PREDICTIONS`, that
`MY_CASES` are new and pass, and that `NOTES` is not empty. Run it yourself
before opening the PR:

```sh
uv run python scripts/tasks.py check tasks/l01-tokenization/digit-grouping <username>
```

## Rules

- Standard library only (`re.findall` with two alternatives is enough).
- Use your own words and code; discussing the approach with classmates is fine.
- PR title `l01-tokenization/digit-grouping: <username>`, body `Related to #<issue>`.

## Why this matters

Lecture 01's pre-tokenizer comparison: the same number becomes a different token sequence under GPT-4, Qwen, and GPT-2 (which keeps the whole run). Token counts, vocabulary use, and arithmetic behaviour all follow from this one rule.
