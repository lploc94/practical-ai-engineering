from decimal import Decimal as D
import anthropic
from anthropic.types import Usage
from anthropic.types.beta import BetaUsage
from cost import request_cost

# 1. ví dụ trong bảng giá Claude (Managed Agents, phần token): 10k input, 40k cache read, 15k output trên Opus 5.5
u = Usage(input_tokens=10_000, cache_read_input_tokens=40_000, output_tokens=15_000)
assert request_cost(u, "claude-opus-5-5") == D("0.348"), request_cost(u, "claude-opus-5-5")
# 2. trang cost optimization: ghi 100k token cache 1h trên Opus 5.5 = $0.80; đọc 100k = $0.02
u = Usage(input_tokens=0, cache_creation_input_tokens=100_000,
          cache_creation={"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 100_000}, output_tokens=0)
assert request_cost(u, "claude-opus-5-5") == D("0.8")
assert request_cost(Usage(input_tokens=0, cache_read_input_tokens=100_000, output_tokens=0), "claude-opus-5-5") == D("0.02")
# 3. cache write 5m không có breakdown = 1.25x
assert request_cost(Usage(input_tokens=0, cache_creation_input_tokens=1_000_000, output_tokens=0), "claude-sonnet-5-5") == D("2.5")
# 4. Haiku 5.5 vượt 100k: cả input lẫn output tính giá cao; cache read cũng tính vào ngưỡng
u = Usage(input_tokens=20_000, cache_read_input_tokens=90_000, output_tokens=1_000)
assert request_cost(u, "claude-haiku-5-5") == (D(20_000)*D("0.5") + D(90_000)*D("0.05") + D(1_000)*D("2.5"))/D(1_000_000)
u = Usage(input_tokens=20_000, cache_read_input_tokens=80_000, output_tokens=1_000)   # đúng 100k: giá thường
assert request_cost(u, "claude-haiku-5-5") == (D(20_000)*D("0.1") + D(80_000)*D("0.01") + D(1_000)*D("0.5"))/D(1_000_000)
# 5. batch: một nửa
u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
assert request_cost(u, "claude-sonnet-5-5", batch=True) == D(6)
# 6. fast mode Opus 5.5: $8/$40, cache read 0.05x của giá fast
u = Usage(input_tokens=1_000_000, cache_read_input_tokens=1_000_000, output_tokens=1_000_000, speed="fast")
assert request_cost(u, "claude-opus-5-5") == D(8) + D("0.4") + D(40)
# 7. iterations: fallback và compaction chỉ có trong iterations; mỗi mục theo model của nó
u = BetaUsage.model_validate({"input_tokens": 1000, "output_tokens": 500, "iterations": [
    {"type": "message", "input_tokens": 1000, "output_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "model": "claude-fable-5-1"},
    {"type": "fallback_message", "input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "model": "claude-opus-4-8"}]})
exp = (D(1000)*10 + D(10)*50 + D(1000)*5 + D(500)*25) / D(1_000_000)
assert request_cost(u, "claude-fable-5-1") == exp, (request_cost(u, "claude-fable-5-1"), exp)
assert request_cost(u, "claude-fable-5-1") > request_cost(Usage(input_tokens=1000, output_tokens=500), "claude-opus-4-8")
# 8. dict thô (đọc từ log JSON) cũng chạy
assert request_cost({"input_tokens": 1_000_000, "output_tokens": 0}, "claude-opus-5-5") == D(4)
print("COST OK")
