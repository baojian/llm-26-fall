PREDICTIONS = {
    "don't stop": ["don","'t"," stop"],
    "x2 + 3x = 0": ["x","2"," +"," 3","x"," ="," 0"],
    "It's 3.14": ["It","'s"," 3",".","14"],
}
MY_CASES = [("llm-26-fall", ["llm","-","26","-","fall"]), ("tasks/l01-tokenization/gpt2-pretokenizer/instruction.md", ["tasks","/","l","01","-","tokenization","/","gpt","2","-","pretokenizer","/","instruction",".","md"])]
# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """Attaching an ASCII space to the following word gives the model an explicit word-start cue in a form such as " world". BPE can learn frequent leading-space forms, while unspaced forms may also occur and either chunk can split into smaller tokens. Attaching a space to the preceding word would instead favor forms such as "world ", marking the boundary after that word and changing which variants the vocabulary learns. Both conventions can preserve every input character; leading spaces are not needed to prevent BPE from losing whitespace, and neither convention alone establishes better generation quality."""

def solve(text: str) -> list[str]:
    import re
    res_str_list = []
    pattern = re.compile(r"""
    ('s|'t|'re|'ve|'m|'ll|'d)
    |([ ]?[^\W\d_]+)
    |([ ]?\d+)
    |([ ]?(?:(?![^\W\d_]|\d)\S)+)
    |(\s+(?!\S))
    |(\s+)
    """,re.X)
    for match in pattern.finditer(text):
        res_str_list.append(match.group())
    return res_str_list
