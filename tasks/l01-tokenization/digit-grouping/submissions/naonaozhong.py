PREDICTIONS = {
    "3.14159": ["3", ".", "141", "59"],
    "Room 101": ["Room ", "101"],
    "2024-09-16": ["202", "4", "-", "09", "-", "16"],
}
MY_CASES = [
    ("00786", ["007", "86"]),
    ("１２３４５", ["１２３", "４５"]),
]
NOTES = """If every digit string of length 1 to k has its own token, 
then for each length there are 10^L possible strings,
since each digit has 10 choices and leading zeros count too. 
So k=1 needs 10 entries, k=3 needs 10+100+1000=1110 entries, and k=4 needs 1110+10000=11110 entries. 
The number 123456789012 has 12 digits, so it becomes 12 tokens when k=1, 4 tokens when k=3 (123, 456, 789, 012), and 3 tokens when k=4 (1234, 5678, 9012). 
Bigger groups mean fewer tokens but a much bigger vocabulary,
because the number of possible groups grows by a factor of 10 for each extra digit.
A three-character limit only says a chunk has at most three characters; 
it does not force BPE to keep the chunk as one token, since BPE merges the most common pairs and may split a three-digit chunk into smaller pieces. 
My solve still groups into threes, which matches k=3."""

def solve(text: str) -> list[str]:
    output = list()
    digit = ""
    digit_len = 0
    non_digit = ""
    for ch in text:
        if ch.isdecimal():
            if non_digit:
                output.append(non_digit)
                non_digit = ""

            digit = digit + ch
            digit_len += 1
            if digit_len >= 3:
                output.append(digit)
                digit = ""
                digit_len = 0
        else:
            if non_digit:
                non_digit = non_digit + ch
            else:
                if digit_len > 0:
                    output.append(digit)
                    digit_len = 0
                    digit = ""
                non_digit = non_digit + ch
    if digit:
        output.append(digit)

    if non_digit:
        output.append(non_digit)

    return output
