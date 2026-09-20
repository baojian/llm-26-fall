# Chunk text with the GPT-2 pre-tokenizer rules

**Lecture:** l01-tokenization · **Difficulty:** medium · **Time:** about 45 minutes ·
**Deadline:** Sunday, September 27, 2026, 23:59 (Asia/Shanghai, UTC+08:00).

**Participation:** Optional; see [Lecture 01 activities](../README.md).

## Goal

Implement a simplified GPT-2 pre-tokenizer to understand where BPE can merge.
Chunk boundaries keep words and punctuation separate, avoiding vocabulary
entries for every word–punctuation combination; allowing a leading space
improves compression. See [Radford et al. (2019), Section 2.2](../../../papers/2019-openai-radford-language-models-unsupervised-multitask-learners-gpt2.pdf#page=4)
and the [original encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
Your output contains **pre-tokenization chunks**, which BPE may split further.

## Specification

Implement one function using only the Python standard library:

```python
def solve(text: str) -> list[str]:
    ...
```

Scan left to right; at each position, use the first matching rule with greedy
runs and normal regex backtracking:

1. A lowercase contraction suffix: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`.
2. An optional ASCII space followed by a letter-like run (`[^\W\d_]+`).
3. An optional ASCII space followed by decimal digits (`\d+`).
4. An optional ASCII space followed by other non-whitespace characters:
   `(?:(?![^\W\d_]|\d)\S)+`. This includes punctuation, symbols, and `_`.
5. Whitespace matching `\s+(?!\S)`; backtracking may leave one space for
   the following word.
6. Any remaining whitespace run (`\s+`).

Use Python `re`'s default Unicode, case-sensitive behavior. Its classes are
an approximation of GPT-2's `\p{L}` and `\p{N}`: numerals such as `²` and `½`
join letter-like runs here, but number runs in GPT-2. Do not lowercase or
normalize input.

Preserve every character: `"".join(solve(text)) == text`. Return `[]` for
empty input. The checker supplies raw text; do not read files or print results.

## Examples

| Input | Output |
| :--- | :--- |
| `"Hello world"` | `["Hello", " world"]` |
| `"I'm 25 years old."` | `["I", "'m", " 25", " years", " old", "."]` |
| `"snake_case"` | `["snake", "_", "case"]` |
| `"  two  spaces"` | `[" ", " two", " ", " spaces"]` |
| `"tab\tsep\nline"` | `["tab", "\t", "sep", "\n", "line"]` |
| `""` | `[]` |

The checker also uses the entire two-line [sample.txt](data/sample.txt),
including its final newline, and [expected chunks](data/sample-chunks.json).

## Before you code

Include these variables in the same file. Write `PREDICTIONS` before `solve`;
replace the placeholders with expected outputs and your explanation.

```python
PREDICTIONS = {
    "don't stop": [...],
    "x2 + 3x = 0": [...],
    "It's 3.14": [...],
}
MY_CASES = [("...", [...]), ("...", [...])]
NOTES = """Your explanation here."""
```

Choose two distinct new inputs for `MY_CASES` where a naive solution fails. In `NOTES`,
write **3–5 sentences, at least 200 characters**: why attach a space to the
following word (`" world"`) rather than the preceding word? What does the
model gain, and how would attaching trailing spaces change the vocabulary?

## Submission and checks

Submit **one file**, `submissions/<username>.py`, where `<username>` is the
PR author's GitHub username in lowercase (for example, `OctoCat` → `octocat.py`).
**Other filenames are not accepted, even if the tests pass.**

Run from the repository root, replacing `<username>` with your lowercase
username without `.py`:

```sh
uv run python scripts/tasks.py check tasks/l01-tokenization/gpt2-pretokenizer <username>
```

The checker compares exact chunks, including whitespace, and checks lowercase
filenames, predictions, personal cases, and `NOTES` length. The teaching team
verifies the filename against your account and reviews your reasoning.

Use your own code and words; discussing approaches is fine. Open the PR from
your own account, changing only your submission file. Keep supplied data and
tests unchanged. Use title `l01-tokenization/gpt2-pretokenizer: <username>` and
body `Related to #202`.

Open your PR by the deadline. Solutions merge in **one batch per task after
the deadline**; public PRs and forks remain visible before merging.
