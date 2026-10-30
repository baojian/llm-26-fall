"""Instructor demo submission for the digit-grouping exercise."""

import re

PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

MY_CASES = [
    # Fullwidth decimal digits and leading zeros must be preserved.
    ("PIN ０００１２３４", ["PIN ", "０００", "１２３", "４"]),
    # A superscript is not a decimal digit; non-digit runs span line breaks.
    ("A12²345\n\tB00007", ["A", "12", "²", "345", "\n\tB", "000", "07"]),
]

NOTES = """Counting every ASCII digit string of length 1 through k, including
leading zeros, requires sum(10**i for i in range(1, k + 1)) numeric vocabulary
entries: 10 for k = 1, 1110 for k = 3, and 11110 for k = 4.
If every permitted chunk has its own token, 123456789012 takes 12, 4, and 3
tokens for those respective limits.
Larger groups shorten the sequence but require more vocabulary entries and,
at a fixed embedding width, a larger embedding table.
A pre-tokenization limit only sets chunk boundaries; BPE may split a chunk
further when the learned vocabulary does not represent it as one token.
"""


def solve(text: str) -> list[str]:
    """Group decimal digits from the left and preserve each non-digit run."""
    return re.findall(r"\d{1,3}|\D+", text)
