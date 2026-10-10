PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

MY_CASES = [
    ("ID００１2345 end", ["ID", "００１", "234", "5", " end"]),
    ("a12\n3456b", ["a", "12", "\n", "345", "6", "b"]),
]

# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """If every ASCII digit string of length 1 through k has its own vocabulary entry, including leading zeros, k=1 needs 10 entries, k=3 needs 10+100+1000=1110, and k=4 needs 11110. The string 123456789012 uses 12 numeric tokens for k=1, four three-digit tokens for k=3, and three four-digit tokens for k=4. A larger k shortens sequences but rapidly expands the vocabulary and spends model capacity on rare numeric combinations. A three-digit pre-tokenization limit only sets BPE merge boundaries, so a chunk whose merges were not learned may still split into several BPE tokens."""


def solve(text: str) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        is_digit = text[i].isdecimal()
        j = i + 1
        while j < len(text) and text[j].isdecimal() == is_digit:
            j += 1

        run = text[i:j]
        if is_digit:
            chunks.extend(run[k : k + 3] for k in range(0, len(run), 3))
        else:
            chunks.append(run)
        i = j

    return chunks
