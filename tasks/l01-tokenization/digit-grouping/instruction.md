# Group digits into chunks of at most three

**Lecture:** l01-tokenization · **Difficulty:** easy · **Time:** about 30 minutes ·
**Deadline:** Tuesday, September 22, 23:59 (Asia/Shanghai)

## Goal

A finite vocabulary must represent arbitrarily long numbers by reusing smaller
pieces. GPT-4's `cl100k_base`
[limits numeric chunks to at most three characters](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)
before BPE, restricting which digits can merge together. A useful way to analyze
this choice is the tradeoff between sequence length and vocabulary size: larger
groups can shorten sequences, but covering every possible group costs more
entries. There are 1,000 ASCII digit strings of exactly three digits, compared
with 10,000 of exactly four digits, including leading zeros; shorter strings
need additional entries. The boundaries also affect how the model processes
numbers: [Singh and Strouse (2024)](https://arxiv.org/abs/2402.14903) found that
changing the grouping direction affected addition accuracy in GPT-3.5 and
GPT-4. Implementing this rule connects a small text-processing function to
choices that shape a language model's vocabulary and numerical reasoning.

Implement one function that groups digits from left to right, with at most
three digits per chunk. Use it to understand how a pre-tokenization rule
affects the vocabulary needed to represent numbers.

## What you submit

Submit **one Python file**:
`submissions/<your-github-username-in-lowercase>.py`.

Use the GitHub username in your profile URL, converted to lowercase. For
example, the account `OctoCat` submits `octocat.py`. GitHub may display capital
letters in a username; it does not lowercase your file for you. The filename
must match the account that opens the PR. **We will not accept a submission
with a different filename, even if its code passes the tests.** Rename the
file in the same PR to correct it.

The only function you need to implement is:

```python
def solve(text: str) -> list[str]:
    ...
```

The checker passes raw text to `solve` as a Python string. Return a
`list[str]`; file reading and printing are not part of the function's job.
The group size is fixed at **three** for this task.

1. For each consecutive run of digits, take groups of three **from the left**.
   Keep a final group of one or two digits: `12345678` becomes
   `["123", "456", "78"]`.
2. Keep each consecutive run of non-digits as **one unchanged chunk**.
   Spaces, punctuation, tabs, and line breaks belong to these runs; a run
   may span multiple lines.
3. Preserve every character, including leading zeros and repeated spaces.
   The result must satisfy `"".join(solve(text)) == text`.
4. Return `[]` for an empty input.

Here, a digit means a Unicode decimal digit, as recognized by Python's
`str.isdecimal()` or `re`'s `\d`. This includes `0`–`9` and full-width digits
such as `１`. Keep the original characters; do not convert them to integers.

These outputs are **pre-tokenization chunks**, before BPE and token-ID
lookup. This exercise isolates the digit rule; it is not a complete GPT-4
tokenizer. The three predictions, two personal examples, and short explanation
below are data and prose in the same file, not additional functions.

## Examples

| Input | Output |
| :--- | :--- |
| `"2026"` | `["202", "6"]` |
| `"0000123"` | `["000", "012", "3"]` |
| `"Year 2026: 1,000,000"` | `["Year ", "202", "6", ": ", "1", ",", "000", ",", "000"]` |
| `"A1234B"` | `["A", "123", "4", "B"]` |
| `"no  digits\n"` | `["no  digits\n"]` |
| `""` | `[]` |

### Raw-text example

The supplied [sample.txt](data/sample.txt) contains exactly these two lines,
including a final newline:

```text
Batch 0000123 ran on 2026-09-22.
Total: -1234.0500 USD; code A12345678B.
```

Passing the **entire text** to `solve` must produce:

```python
[
    "Batch ", "000", "012", "3", " ran on ", "202", "6", "-", "09", "-", "22",
    ".\nTotal: -", "123", "4", ".", "050", "0", " USD; code A",
    "123", "456", "78", "B.\n",
]
```

Notice that `".\nTotal: -"` is a single non-digit chunk. The checker loads
this text and its [expected chunks](data/sample-chunks.json) automatically;
you only submit your Python file.

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
Answer in 3–5 sentences (at least 200 characters): Suppose every possible ASCII
digit string of length 1 through k has its own token, including leading zeros.
How many numeric vocabulary entries are needed for k = 1, 3, and 4? How many
tokens represent the 12-digit string "123456789012" in each case? Explain the
tradeoff, and why setting a pre-tokenization group size alone does not guarantee
that every group is one BPE token. This comparison is a written explanation;
your solve function still uses the fixed group size of three.
"""
```

The three prediction inputs are: `'3.14159'`, `'Room 101'`, `'2024-09-16'`.

## How it is checked

`tests/test_task.py` calls your `solve` on the examples, the complete raw-text
sample, the prediction inputs, and edge cases. It compares the exact ordered
list of strings, including whitespace. It also checks that your filename is
lowercase, that `solve` agrees with your `PREDICTIONS`, that `MY_CASES` are new
and pass, and that `NOTES` has at least 200 characters. The teaching team
checks the filename against the PR author's GitHub username and checks the
function on additional text with the same rules. A passing output check does
not waive the filename requirement. Run the public checks from the repository
root before opening the PR:

```sh
uv run python scripts/tasks.py check tasks/l01-tokenization/digit-grouping <username>
```

Replace `<username>` with your GitHub username in lowercase, without `.py`.
The checker checks correctness on the supplied cases; the teaching team reviews
the reasoning in `NOTES`.

## Rules

- Standard library only. A regular expression or a loop is sufficient.
- Use your own words and code; discussing the approach with classmates is fine.
- Open the PR from your own GitHub account and change only your correctly named
  submission file. Keep the supplied data and tests unchanged.
- PR title `l01-tokenization/digit-grouping: <username>`, body `Related to #<issue>`.

## Why this matters

GPT-4's `cl100k_base` uses a numeric pre-tokenization limit of three; its actual
pattern uses the broader Unicode number category `\p{N}`. See the
[encoding definition](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py).
Larger groups can shorten numeric sequences but require more vocabulary entries
if every possible group is represented by one token. BPE operates within each
chunk, so chunk counts and final token counts must be distinguished. See the
[educational BPE implementation](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py).

Suggested reading: Aaditya K. Singh and DJ Strouse (2024),
[Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs](https://arxiv.org/abs/2402.14903).
The introduction and Figure 3 explain how number tokenization differs between
GPT-3 and GPT-4; Section 3 examines the effect of grouping direction on addition.
