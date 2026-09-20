# Group digits into chunks of at most three

**Lecture:** l01-tokenization · **Difficulty:** easy · **Time:** about 30 minutes ·
**Deadline:** 23:59 (Asia/Shanghai), seven calendar days after
[release PR #131](https://github.com/baojian/llm-26-fall/pull/131) merges into
`main`, using its Shanghai merge date.

**Participation:** Optional; see [Lecture 01 activities](../README.md).

## Goal

Implement digit grouping to explore a tokenizer's tradeoff between sequence
length and vocabulary size: larger groups can shorten sequences, but representing
every possible group requires more vocabulary entries. GPT-4's
[`cl100k_base`](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)
limits numeric chunks to three characters before BPE. Grouping also affects
numerical reasoning: [Singh and Strouse (2024), *Tokenization counts*](../../../papers/2024-arxiv-singh-tokenization-counts-arithmetic-frontier-llms.pdf)
found that changing grouping direction affected addition accuracy.

## Specification

Implement one function using only the Python standard library:

```python
def solve(text: str) -> list[str]:
    ...
```

- Split each consecutive digit run **from the left** into groups of three,
  keeping any final one or two digits.
- Keep each consecutive non-digit run as **one unchanged chunk**, even across
  line breaks.
- Preserve all characters, including leading zeros and whitespace:
  `"".join(solve(text)) == text`. Return `[]` for empty input.
- A digit means a Unicode decimal digit (`str.isdecimal()` or Python `re`'s
  `\d`), including `0`–`9` and `０`–`９`; do not convert to integers.

The checker supplies raw text; do not read files or print results. Return
**pre-tokenization chunks**, which BPE may split further. This is a simplified
rule, not a complete GPT-4 tokenizer; `cl100k_base` uses the broader `\p{N}`
number category.

## Examples

| Input | Output |
| :--- | :--- |
| `"2026"` | `["202", "6"]` |
| `"0000123"` | `["000", "012", "3"]` |
| `"Year 2026: 1,000,000"` | `["Year ", "202", "6", ": ", "1", ",", "000", ",", "000"]` |
| `"A1234B"` | `["A", "123", "4", "B"]` |
| `"no  digits\n"` | `["no  digits\n"]` |
| `""` | `[]` |

The checker also uses the entire two-line [sample.txt](data/sample.txt),
including its final newline, and [expected chunks](data/sample-chunks.json).
For example, `".\nTotal: -"` remains one non-digit chunk.

## Before you code

Include these variables in the same file. Write `PREDICTIONS` before `solve`;
replace the placeholders with expected outputs and your explanation.

```python
PREDICTIONS = {
    "3.14159": [...],
    "Room 101": [...],
    "2024-09-16": [...],
}
MY_CASES = [("...", [...]), ("...", [...])]
NOTES = """Your explanation here."""
```

Choose two distinct new inputs for `MY_CASES` where a naive solution fails. In `NOTES`,
write **3–5 sentences, at least 200 characters**: if every ASCII digit string
of length 1 through *k*, including leading zeros, has its own token, how many
numeric vocabulary entries are needed for *k* = 1, 3, and 4? How many tokens
represent `123456789012` in each case? Explain the tradeoff and why a
pre-tokenization limit alone does not guarantee one BPE token per chunk.
Your `solve` still uses groups of three.

## Submission and checks

Submit **one file**, `submissions/<username>.py`, where `<username>` is the
PR author's GitHub username in lowercase (for example, `OctoCat` → `octocat.py`).
**Other filenames are not accepted, even if the tests pass.**

Run from the repository root, replacing `<username>` with your lowercase
username without `.py`:

```sh
uv run python scripts/tasks.py check tasks/l01-tokenization/digit-grouping <username>
```

The public checker compares exact chunks, including whitespace, on supplied
and edge cases; it also checks lowercase filenames, predictions, personal
cases, and `NOTES` length. The teaching team verifies the filename against
your account, reviews your reasoning, and checks additional inputs.

Use your own code and words; discussing approaches is fine. Open the PR from
your own account, changing only your submission file. Keep supplied data and
tests unchanged. Use title `l01-tokenization/digit-grouping: <username>` and
body `Related to #<issue>`.
