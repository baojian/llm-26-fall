# Count words with a regular expression

**Lecture:** example · **Difficulty:** easy · **Time:** about 20 minutes ·
**Deadline:** none (this is the practice task)

## Goal

Use `re.findall` to split English text into words, the first step of every
tokenizer in Lecture 01.

## What you submit

One file: `submissions/<your-github-username>.py` containing a function

```python
def solve(text: str) -> list[str]:
    ...
```

Input: any string. Output: the words in order, lowercased, where a word is a
run of ASCII letters, digits, or apostrophes. Punctuation and whitespace are
not words. An empty string gives an empty list.

## Examples

| Input | Output |
| :--- | :--- |
| `"It's hard to recognize speech."` | `["it's", "hard", "to", "recognize", "speech"]` |
| `"GPT-4 has 2 tokens"` | `["gpt", "4", "has", "2", "tokens"]` |
| `""` | `[]` |

## How it is checked

`tests/test_task.py` calls your `solve` on the examples above and on a few
similar cases. Run it yourself before opening the PR:

```sh
uv run python scripts/tasks.py check tasks/example/word-count <username>
```

## Rules

- Standard library only (`re` is enough).
- Use your own words and code; discussing the approach with classmates is fine.
- PR title `example/word-count: <username>`, body `Related to #<issue>`.

## Why this matters

Lecture 01 tokenizes text before anything else happens. A regular expression
is the simplest tokenizer, and its choices (what counts as a word) already
change the vocabulary the model sees.
