import threading
from datetime import datetime, timezone, timedelta
from decimal import Decimal as D
from budget import SpendLedger, BudgetExceeded, max_request_cost

now = [datetime(2026, 11, 6, 23, 0, tzinfo=timezone.utc)]
L = SpendLedger({"acme": D("1.00")}, clock=lambda: now[0])
# 1. trần chi phí: Sonnet 5.5, 10k input ghi cache 5m, max_tokens 4k = 10k*2*1.25 + 4k*10 = 0.065
assert max_request_cost("claude-sonnet-5-5", 10_000, 4_000) == D("0.065")
assert max_request_cost("claude-sonnet-5-5", 10_000, 4_000, D(1)) == D("0.06")
# Haiku 5.5: prompt vượt 100k thì tính theo giá prompt dài; output không tính vào ngưỡng
assert max_request_cost("claude-haiku-5-5", 101_000, 2_000, D(1)) == (D(101_000)*D("0.5") + D(2_000)*D("2.5"))/D(1_000_000)
assert max_request_cost("claude-haiku-5-5", 99_000, 2_000, D(1)) == (D(99_000)*D("0.1") + D(2_000)*D("0.5"))/D(1_000_000)
# 2. trừ chi phí thật sau khi gọi
with L.hold("acme", D("0.50")) as a:
    a["usd"] = D("0.10")
assert L.spent[("acme", "2026-11-06")] == D("0.10") and ("acme", "2026-11-06") not in L.reserved
# 3. phần đang giữ chỗ chặn request khác dù chưa tiêu
with L.hold("acme", D("0.80")) as a:
    try:
        with L.hold("acme", D("0.20")): pass
        raise SystemExit("phải bị chặn")
    except BudgetExceeded: pass
    a["usd"] = D("0.05")
# 4. lỗi trong lúc gọi mà không gán chi phí: tính đủ phần giữ chỗ
try:
    with L.hold("acme", D("0.30")):
        raise RuntimeError("mất kết nối")
except RuntimeError: pass
assert L.spent[("acme", "2026-11-06")] == D("0.45")
# 5. vượt hạn mức
try:
    with L.hold("acme", D("0.56")): pass
    raise SystemExit("phải bị chặn")
except BudgetExceeded as e: assert "hạn mức 1.00" in str(e)
with L.hold("acme", D("0.55")) as a: a["usd"] = D("0.55")
# 6. sang ngày mới (UTC) thì hạn mức làm mới
now[0] += timedelta(hours=2)
with L.hold("acme", D("0.90")) as a: a["usd"] = D("0.01")
assert set(L.spent) == {("acme", "2026-11-07")} and L.reserved == {}, (L.spent, L.reserved)
# 6b. chạy lâu: số khoá không tăng theo số ngày
for _ in range(1000):
    now[0] += timedelta(days=1)
    with L.hold("acme", D("0.01")) as a: a["usd"] = D("0.01")
assert len(L.spent) == 1 and L.reserved == {}
# 6c. request kéo qua nửa đêm: ngày mới không bị trừ, và không tạo lại khoá của ngày cũ
with L.hold("acme", D("0.50")) as a:
    now[0] += timedelta(days=1)
    with L.hold("acme", D("0.90")) as b: b["usd"] = D("0.90")
    a["usd"] = D("0.50")
today = now[0].strftime("%Y-%m-%d")
assert L.spent == {("acme", today): D("0.90")} and L.reserved == {}, (L.spent, L.reserved)
# 7. nhiều thread: tổng giữ chỗ không bao giờ vượt hạn mức
L2 = SpendLedger({"t": D("1.00")}); ok = []; barrier = threading.Barrier(20)
def worker():
    barrier.wait()
    try:
        with L2.hold("t", D("0.10")) as a:
            ok.append(1); import time; time.sleep(0.05); a["usd"] = D("0.10")
    except BudgetExceeded: pass
ts = [threading.Thread(target=worker) for _ in range(20)]
[t.start() for t in ts]; [t.join() for t in ts]
assert len(ok) == 10 and sum(L2.spent.values()) == D("1.00"), (len(ok), L2.spent)
print("BUDGET OK")
