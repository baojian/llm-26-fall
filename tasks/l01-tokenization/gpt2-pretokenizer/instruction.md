# Chunk text with the GPT-2 pre-tokenizer rules

**Lecture:** l01-tokenization · **Difficulty:** medium · **Time:** about 45 minutes ·
**Deadline:** Tuesday, September 22, 23:59 (Asia/Shanghai)

## Goal

Reproduce the regular expression that runs before BPE in GPT-2 (and, with small changes, in GPT-4, Qwen, and Llama 3), so you can say exactly where merges are allowed.

## What you submit

One file: `submissions/<your-github-username>.py` containing a function

```python
def solve(text: str) -> list[str]:
    ...
```

Input: any string. Output: the chunks the GPT-2 pre-tokenizer would produce, in order, concatenating back to the input. The rules, applied left to right, first match wins:

1. a contraction suffix `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`;
2. an optional space followed by a run of letters (Python's `re` has no `\p{L}`; use `[^\W\d_]` for a Unicode letter);
3. an optional space followed by a run of digits (`\d`);
4. an optional space followed by a run of other non-space characters, i.e. characters that are neither a letter nor a digit (punctuation, symbols, and also `_`, which Python's `\w` would swallow; write this branch as `(?:(?![^\W\d_]|\d)\S)+`);
5. a run of whitespace that is not followed by a non-space character (`\s+(?!\S)`);
6. any remaining run of whitespace.

Every character of the input lands in exactly one chunk, so the chunks concatenate back to the input. An empty string gives an empty list.

## Examples

| Input | Output |
| :--- | :--- |
| `"Hello world"` | `["Hello", " world"]` |
| `"I'm 25 years old."` | `["I", "'m", " 25", " years", " old", "."]` |
| `"snake_case"` | `["snake", "_", "case"]` |
| `""` | `[]` |

## Before you code: predict, then break it

Your file also contains three things the checker reads:

```python
PREDICTIONS = {  # write these BEFORE writing solve; the checker compares solve to them
    "don't stop": [...],
    'x2 + 3x = 0': [...],
    "It's 3.14": [...],
}
MY_CASES = [  # two inputs of your own where a naive solution fails, with the expected output
    ("...", [...]),
    ("...", [...]),
]
NOTES = """
Answer in 3–5 sentences: Rule 2 attaches the space to the *following* word (" world" is one chunk), not to the preceding one. What does the model gain from that choice, and what would change for the vocabulary if the space were attached to the preceding word instead?
"""
```

The three prediction inputs are: `"don't stop"`, `'x2 + 3x = 0'`, `"It's 3.14"`.

## How it is checked

`tests/test_task.py` calls your `solve` on the examples above and on a few
similar cases, checks that `solve` agrees with your `PREDICTIONS`, that
`MY_CASES` are new and pass, and that `NOTES` is not empty. Run it yourself
before opening the PR:

```sh
uv run python scripts/tasks.py check tasks/l01-tokenization/gpt2-pretokenizer <username>
```

## Rules

- Standard library only (`re` is enough). Python's `re` does not support `\p{L}`, `\p{N}`, or possessive quantifiers; the six rules above are already written in `re` syntax. One known difference from tiktoken: Python's `\d` matches only decimal digits, and its `\w` counts numerals like `²` and `½` as word characters, so they join the letter run in rule 2 here, while GPT-2's `\p{N}` treats them as digits.
- Use your own words and code; discussing the approach with classmates is fine.
- PR title `l01-tokenization/gpt2-pretokenizer: <username>`, body `Related to #<issue>`.

## Why this matters

Lecture 01 showed that BPE merges are only learned inside a chunk. This pattern is the boundary: it decides that `'s` is its own token, that a space belongs to the word after it, and that digits never merge with letters. Every model family ships a variant of it (see docs/regex-in-llm-training.md, Section 1).
