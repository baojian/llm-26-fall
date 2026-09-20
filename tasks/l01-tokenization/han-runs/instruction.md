# Keep Chinese character runs together

**Lecture:** l01-tokenization · **Difficulty:** medium · **Time:** about 40 minutes ·
**Deadline:** 23:59 (Asia/Shanghai), seven calendar days after
[release PR #131](https://github.com/baojian/llm-26-fall/pull/131) merges into
`main`, using its Shanghai merge date.

**Participation:** Optional; see [Lecture 01 activities](../README.md).

## Goal

Explore how script boundaries constrain BPE's vocabulary. Kimi K2's
[tokenizer](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/tokenization_kimi.py)
separates Han runs from other letters, preventing merges across those
boundaries. Implement the simplified rule below. For model background, see
[Kimi Team (2025), *Kimi K2: Open Agentic Intelligence*](../../../papers/2025-arxiv-kimi-team-kimi-k2-open-agentic-intelligence.pdf).
Your output contains **pre-tokenization chunks**, which BPE may split further.

## Specification

Implement one function using only the Python standard library:

```python
def solve(text: str) -> list[str]:
    ...
```

Return maximal consecutive runs of these four disjoint categories, in order:

1. Characters in `U+4E00`–`U+9FFF`, inclusive (`[\u4e00-\u9fff]`).
2. ASCII letters and digits (`[A-Za-z0-9]`), grouped together: `GPT4` is one run.
3. Whitespace recognized by `str.isspace()` or Python `re`'s Unicode `\s`,
   including spaces, tabs, and line breaks.
4. Everything else, including punctuation, symbols, accented letters,
   non-ASCII digits, and Han characters outside the specified range.

A run ends only when the category changes. Preserve every character:
`"".join(solve(text)) == text`. Return `[]` for empty input. Do not normalize
text. The checker supplies raw text; do not read files or print results.

This exercise uses one Unicode block, not Kimi's full `\p{Han}` script set;
for example, `㐀` (`U+3400`) and `𠀀` (`U+20000`) belong to category 4 here.
Python's standard `re` does not support `\p{Han}`.

## Examples

| Input | Output |
| :--- | :--- |
| `"用GPT写代码"` | `["用", "GPT", "写代码"]` |
| `"你好, world"` | `["你好", ",", " ", "world"]` |
| `"K2中文2026"` | `["K2", "中文", "2026"]` |
| `"\t \n"` | `["\t \n"]` |
| `"Ａ１é🙂"` | `["Ａ１é🙂"]` |
| `""` | `[]` |

The checker also uses the entire two-line [sample.txt](data/sample.txt),
including its final newline, and [expected chunks](data/sample-chunks.json).

## Before you code

Include these variables in the same file. Write `PREDICTIONS` before `solve`;
replace the placeholders with expected outputs and your explanation.

```python
PREDICTIONS = {
    "Kimi K2 的 tokenizer": [...],
    "2026年9月16日": [...],
    "GPT-4o很强": [...],
}
MY_CASES = [("...", [...]), ("...", [...])]
NOTES = """Your explanation here."""
```

Choose two distinct new inputs for `MY_CASES` where a naive solution fails. In `NOTES`,
write **3–5 sentences, at least 200 characters**: how could merges across
Han–Latin boundaries affect vocabulary reuse and mixed-script token lengths?
Using `用GPT` and `GPT用`, explain the roles of putting Han first and excluding
Han from other letter branches in Kimi's pattern.

## Submission and checks

Submit **one file**, `submissions/<username>.py`, where `<username>` is the
PR author's GitHub username in lowercase (for example, `OctoCat` → `octocat.py`).
**Other filenames are not accepted, even if the tests pass.**

Run from the repository root, replacing `<username>` with your lowercase
username without `.py`:

```sh
uv run python scripts/tasks.py check tasks/l01-tokenization/han-runs <username>
```

The checker compares exact chunks, including whitespace, and checks lowercase
filenames, predictions, personal cases, and `NOTES` length. The teaching team
verifies the filename against your account and reviews your reasoning.

Use your own code and words; discussing approaches is fine. Open the PR from
your own account, changing only your submission file. Keep supplied data and
tests unchanged. Use title `l01-tokenization/han-runs: <username>` and
body `Related to #<issue>`.
