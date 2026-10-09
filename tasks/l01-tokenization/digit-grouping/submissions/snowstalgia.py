"""Group decimal digit runs into chunks of at most three characters."""

PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

MY_CASES = [
    ('Where is my “baby girl"?', ['Where is my “baby girl"?']),
    ("A0000123B", ["A", "000", "012", "3", "B"]),
]

NOTES = (
    "With all ASCII digit strings of lengths 1 through k, including leading "
    "zeros, the numeric vocabulary needs 10 entries for k=1, 1,110 for k=3, "
    "and 11,110 for k=4. "
    "The string 123456789012 takes 12, 4, or 3 chunks with maximum group "
    "lengths of 1, 3, or 4. "
    "Longer groups can shorten the sequence, but they need many more entries "
    "in the vocabulary. "
    "Pre-tokenization only sets the chunk boundaries; BPE may still represent "
    "one chunk with several tokens when it has not learned the needed merges."
)


def solve(text: str) -> list[str]:
    chunks: list[str] = []
    position = 0
    while position < len(text):
        if text[position].isdecimal():
            end = position
            while end < len(text) and text[end].isdecimal():
                end += 1
            for start in range(position, end, 3):
                chunks.append(text[start:min(start + 3, end)])
        else:
            end = position
            while end < len(text) and not text[end].isdecimal():
                end += 1
            chunks.append(text[position:end])
        position = end
    return chunks
