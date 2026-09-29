PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}

def solve(text: str) -> list[str]:
    chunks = []
    digit_buffer = ""
    other_buffer = ""

    for character in text:
        if character.isdecimal():
            if other_buffer:
                chunks.append(other_buffer)
                other_buffer = ""

            digit_buffer += character

            if len(digit_buffer) == 3:
                chunks.append(digit_buffer)
                digit_buffer = ""
        else:
            if digit_buffer:
                chunks.append(digit_buffer)
                digit_buffer = ""

            other_buffer += character

    if digit_buffer:
        chunks.append(digit_buffer)

    if other_buffer:
        chunks.append(other_buffer)

    return chunks

MY_CASES = [
    (
        "Student 26210980124",
        ["Student ", "262", "109", "801", "24"],
    ),
    (
        "v2\nbuild0000456!",
        ["v", "2", "\nbuild", "000", "045", "6", "!"],
    ),
]

NOTES = """
If every ASCII digit string from length one through k has its own token, k=1 requires 10 numeric vocabulary entries, k=3 requires 1,110, and k=4 requires 11,110. The number 123456789012 needs 12 tokens when k=1, 4 tokens when k=3, and 3 tokens when k=4. Larger groups shorten the sequence, but the number of required vocabulary entries grows very quickly. A pre-tokenization limit does not guarantee one BPE token per chunk, because BPE may not have learned the merges needed to represent the complete chunk as a single vocabulary entry.
"""