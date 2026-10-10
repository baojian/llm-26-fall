PREDICTIONS = {
    "3.14159": ["3",".","141","59"],
    "Room 101": ["Room ","101"],
    "2024-09-16": ["202","4","-","09","-","16"],
}
MY_CASES = [("a1b2c3", ["a","1","b","2","c","3"]), ("a\n1\n", ["a\n","1","\n"])]
# Teaching-team edit: translated or corrected NOTES during the authorized October 1 review.
NOTES = """For k=1, k=3, and k=4, including all ASCII digit strings and leading zeros requires 10, 10+100+1000=1110, and 10+100+1000+10000=11110 vocabulary entries, respectively. The string 123456789012 then takes 12, 4, and 3 tokens, respectively. Increasing k makes long numbers shorter in tokens, but the vocabulary grows exponentially. Pre-tokenization only sets chunk boundaries, so a chunk such as 123 may still become 1 and 23 if the learned BPE merges do not combine the whole chunk."""

def solve(text: str) -> list[str]:
    temp_str = ''
    res_str_list = []
    count = 0
    for i in range (0, len(text)):
        if i < len(text)-1:
            if not text[i].isdecimal():
                temp_str += text[i]
                if not text[i+1].isdecimal():
                    continue
                else:
                    res_str_list.append(temp_str)
                    temp_str = ''
            else:
                temp_str += text[i]
                count += 1
                if count < 3 and text[i+1].isdecimal():
                    continue
                else:
                    res_str_list.append(temp_str)
                    count = 0
                    temp_str = ''
        else:
            res_str_list.append(temp_str+text[i])
    return res_str_list
