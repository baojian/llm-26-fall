PREDICTIONS = {
    "Kimi K2 的 tokenizer": ["Kimi"," ","K2"," ","的"," ","tokenizer"],
    "2026年9月16日": ["2026","年","9","月","16","日"],
    "GPT-4o很强": ["GPT","-","4o","很强"],
}
MY_CASES = [("Ａ１é", ["Ａ１é"]), ("Grok和Gemini", ["Grok","和","Gemini"])]
# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """Allowing merges across Han-Latin boundaries could make mixed forms such as 用GPT and GPT用 use fewer tokens, but these specific combinations may be rare and occupy vocabulary entries with limited reuse. Putting the Han branch first prevents a broad letter branch from consuming 用GPT as one run when matching starts at 用. Excluding Han from other letter branches also stops a Latin-first run at the boundary in GPT用. Together, these choices create boundaries in both directions so the same Han and Latin pieces can be reused across contexts."""

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
