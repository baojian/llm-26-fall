"""Sunny-hs submission for the optional digit-grouping activity."""

PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

MY_CASES = [
    ("编号１２3456X", ["编号", "１２3", "456", "X"]),
    ("99\n\nA００００", ["99", "\n\nA", "０００", "０"]),
]

NOTES = """If every ASCII digit string of lengths 1 through k has its own token, k=1 needs 10 numeric entries, k=3 needs 10+100+1000=1,110, and k=4 needs 11,110. The twelve-digit string 123456789012 then takes 12, 4, and 3 tokens respectively when grouped to those maximum lengths. Longer groups improve compression, but the numeric vocabulary grows exponentially and spends capacity on many rare strings. A pre-tokenization limit only sets merge boundaries: a chunk becomes one BPE token only if training actually learned the required byte merges."""


def solve(text: str) -> list[str]:
    """Split decimal runs from the left into groups of at most three."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        decimal = text[start].isdecimal()
        end = start + 1
        while end < len(text) and text[end].isdecimal() == decimal:
            end += 1

        run = text[start:end]
        if decimal:
            chunks.extend(run[i : i + 3] for i in range(0, len(run), 3))
        else:
            chunks.append(run)
        start = end
    return chunks
