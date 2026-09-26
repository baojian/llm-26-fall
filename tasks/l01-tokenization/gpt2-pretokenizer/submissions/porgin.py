PREDICTIONS = {
    "don't stop": ["don","'t"," stop"],
    "x2 + 3x = 0": ["x","2"," +"," 3","x"," ="," 0"],
    "It's 3.14": ["It","'s"," 3",".","14"],
}
MY_CASES = [("llm-26-fall", ["llm","-","26","-","fall"]), ("tasks/l01-tokenization/gpt2-pretokenizer/instruction.md", ["tasks","/","l","01","-","tokenization","/","gpt","2","-","pretokenizer","/","instruction",".","md"])]
NOTES = """这种做法主要为了让模型能清晰地区分单词在句子中的位置，
            并带来了一系列技术和效率上的好处.将空格附加到后续单词，
            是一项兼顾了 BPE 算法兼容性、模型处理效率和边界清晰度的设计。
            它通过" "标记，让模型能明确感知单词的起始位置，简化了处理逻辑，
            并避免了 BPE 合并时可能出现的空格丢失问题。如果改为尾随空格，
            不仅会使词汇表结构大变，还可能增加模型预测的负担，并可能对生成性能产生负面影响。"""

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