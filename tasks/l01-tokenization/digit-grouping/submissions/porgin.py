PREDICTIONS = {
    "3.14159": ["3",".","141","59"],
    "Room 101": ["Room ","101"],
    "2024-09-16": ["202","4","-","09","-","16"],
}
MY_CASES = [("a1b2c3", ["a","1","b","2","c","3"]), ("a\n1\n", ["a\n","1","\n"])]
NOTES = """问题1:在k=1时,entries=10;在k=3时,entries=10+100+1000;在k=4时,entries=10+100+1000+10000。
            问题2:k=1,3,4的情况下tokens数量分别为12,4,3。
            问题3:k越大,词表大小指数级增长,好处是长数字会被拆成更少的token,推理速度更快;
            预分词只决定候选块的边界,而合并规则存在先后顺序,即使是123,也可能被拆分为1|23两个token，无法保证一个数据块对应一个BPE token。"""

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