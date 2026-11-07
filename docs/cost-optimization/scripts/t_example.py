"""Ví dụ §2.2: Sonnet 5.5, prefix 6.000 token, mỗi lượt user 100, trả lời 300.
Breakpoint ở cuối prompt mỗi lượt: lượt i đọc cache toàn bộ prompt lượt i-1,
ghi phần mới (câu trả lời trước + câu hỏi mới)."""
from decimal import Decimal as D
from anthropic.types import Usage
from cost import request_cost

M = "claude-sonnet-5-5"

def conversation(turns):
    unc = cached = D(0)
    prev = 0
    for i in range(1, turns + 1):
        prompt = 6000 + i * 100 + (i - 1) * 300
        unc += request_cost(Usage(input_tokens=prompt, output_tokens=300), M)
        cached += request_cost(Usage(input_tokens=0, cache_read_input_tokens=prev,
                                     cache_creation_input_tokens=prompt - prev, output_tokens=300), M)
        prev = prompt
    return unc, cached

u8, c8 = conversation(8); u40, c40 = conversation(40)
print(u8, c8, u40, c40, round(u8 / c8, 2), round(u40 / c40, 2))
assert (u8, c8, u40, c40) == (D("0.1440"), D("0.05136"), D("1.2320"), D("0.22768"))
print("EXAMPLE OK")
