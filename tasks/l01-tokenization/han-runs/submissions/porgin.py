PREDICTIONS = {
    "Kimi K2 的 tokenizer": ["Kimi"," ","K2"," ","的"," ","tokenizer"],
    "2026年9月16日": ["2026","年","9","月","16","日"],
    "GPT-4o很强": ["GPT","-","4o","很强"],
}
MY_CASES = [("Ａ１é", ["Ａ１é"]), ("Grok和Gemini", ["Grok","和","Gemini"])]
NOTES = """如果Han-Latin边界允许合并,比如说让"GPT用"或者"用GPT"预分为一个块。
            好处是BPE token数可能更少,坏处是这种混合串频率低,难复用,还会占用词表。
            因此把Han放第一可以防止"用GPT"被整体吞掉,而从其他字母分支排除Han。
            则可以防止"GPT用"被整体吞掉,两者合起来目的就是建立对称硬边界,提高纯汉字
            和纯拉丁token复用率。"""

def solve(text: str) -> list[str]:
    import re
    res_str_list = []
    pattern = re.compile(r"""
    ([\u4e00-\u9fff]+)
    |([A-Za-z0-9]+)
    |(\s+)
    |([^\u4e00-\u9fffA-Za-z0-9\s]+)
    """,re.X)
    for m in pattern.finditer(text):
        res_str_list.append(m.group())
    return res_str_list