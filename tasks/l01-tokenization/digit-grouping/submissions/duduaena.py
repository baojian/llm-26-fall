PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}


def solve(text: str) -> list[str]:
    """Split text into non-digit runs and digit chunks of at most 3."""
    if not text:
        return []

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        if text[i].isdecimal():
            start = i

            while i < n and text[i].isdecimal() and i - start < 3:
                i += 1

            chunks.append(text[start:i])

        else:
            start = i

            while i < n and not text[i].isdecimal():
                i += 1

            chunks.append(text[start:i])

    return chunks


MY_CASES = [
    (
        "ID: 00123456",
        ["ID: ", "001", "234", "56"],
    ),
    (
        "Version 1234567-beta",
        ["Version ", "123", "456", "7", "-beta"],
    ),
]


NOTES = """
Allowing digit strings of length 1 through k to be separate tokens requires 10 entries for k=1, 
1110 entries for k=3, and 11110 entries for k=4. For the 12-digit string 123456789012, the corresponding 
numbers of tokens are 12, 4, and 3. Therefore, larger digit groups can reduce the number of tokens and
shorten the sequence, but they also require a much larger vocabulary. This shows a trade-off between vocabulary 
size and sequence length. Also, a pre-tokenized chunk is not necessarily one final BPE token, because BPE may 
split the chunk further if the whole chunk was not learned as a vocabulary item.
"""
