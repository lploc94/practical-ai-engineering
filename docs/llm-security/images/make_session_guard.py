"""Sinh session-guard.svg: SVG tĩnh minh hoạ `Session` ở §2.1 "Code: áp Rule of Two cho từng phiên".

Ba hàng là ba test trong danh sách "Test đã chạy" của §2.1:
- Test 1: get_orders (B) rồi send_email (C): không hỏi.
- Test 3: get_orders (B), fetch_url (A), rồi send_email (C): bị hỏi.
- Test 5: send_email (C) trước, rồi read_email (A và B), rồi send_email (C): lần đầu không hỏi,
  lần sau bị hỏi.
Nhãn A/B/C của từng tool lấy theo danh sách TOOLS trong bài. Dưới mỗi ô là `seen` sau lời gọi đó:
thuộc tính chỉ được cộng thêm, không bao giờ gỡ khỏi phiên.

Script import `Session` thật từ ../scripts/session_guard.py và chạy đúng ba chuỗi lời gọi
(approve luôn trả True, tool giả không làm gì), rồi assert kết quả hỏi/không hỏi và `seen`
khớp với những gì hình vẽ.

Màu: xanh dương là lời gọi C chạy không cần hỏi, cam là lời gọi C phải chờ người duyệt;
tool đọc dùng ô trung tính. Không dùng đỏ để không đặt cạnh cam. Hình không có số liệu.

Chạy: python3 make_session_guard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
sys.dont_write_bytecode = True     # không để __pycache__ trong ../scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import svgkit  # noqa: E402
from session_guard import Cap, Session, Tool  # noqa: E402

A, B, C = Cap.UNTRUSTED, Cap.PRIVATE, Cap.OUTWARD
TOOLS = [
    Tool("read_email", A | B, lambda ctx: None),
    Tool("get_orders", B, lambda ctx: None),
    Tool("fetch_url", A, lambda ctx: None),
    Tool("send_email", C, lambda ctx: None),
]
CAPS = {t.name: t.caps for t in TOOLS}
ROWS = [
    ("Test 1", "B rồi C", ["get_orders", "send_email"]),
    ("Test 3", "B, A rồi C", ["get_orders", "fetch_url", "send_email"]),
    ("Test 5", "C trước, rồi A+B", ["send_email", "read_email", "send_email"]),
]


def letters(caps: Cap) -> str:
    return " ".join(ch for ch, c in (("A", A), ("B", B), ("C", C)) if c in caps) or "trống"


# Chạy code thật: mỗi bước ghi lại (tên tool, có bị hỏi không, seen sau lời gọi)
runs = []
for _, _, calls in ROWS:
    asked: list[str] = []
    s = Session(TOOLS, ctx=None, approve=lambda name, shown: asked.append(name) or True)
    steps = []
    for name in calls:
        before = len(asked)
        s.call(name, {})
        steps.append((name, len(asked) > before, s.seen))
    runs.append(steps)

# Kết quả phải khớp với các test trong bài
assert [q for _, q, _ in runs[0]] == [False, False]
assert [q for _, q, _ in runs[1]] == [False, False, True]
assert [q for _, q, _ in runs[2]] == [False, False, True]
assert runs[1][-1][2] == A | B | C and runs[2][0][2] == C

W = 900
LX, X0, CW, GAP, CH = 20, 200, 190, 24, 58
RY0, RSTEP = 112, 112
H = RY0 + RSTEP * len(ROWS) + 40
xs = [X0 + i * (CW + GAP) for i in range(3)]
assert xs[-1] + CW <= W - 30

body: list[str] = []
body.append('<text class="title" x="20" y="38">Session cộng dồn A, B, C; chỉ hỏi khi gọi C trong phiên có cả A và B</text>')
body.append('<text class="cap2" x="20" y="62">Thuộc tính không bao giờ được gỡ khỏi phiên. '
            'Dưới mỗi ô: seen sau lời gọi.</text>')
assert 9.6 * len("Session cộng dồn A, B, C; chỉ hỏi khi gọi C trong phiên có cả A và B") <= W - 50

for r, ((name, sub, _), steps) in enumerate(zip(ROWS, runs)):
    y = RY0 + r * RSTEP
    body.append(f'<text class="row" x="{LX}" y="{y + 26}">{name}</text>')
    body.append(f'<text class="sub" x="{LX}" y="{y + 44}">{sub}</text>')
    for i, (tool, asked, seen) in enumerate(steps):
        x = xs[i]
        outward = C in CAPS[tool]
        cls = ("orange" if asked else "blue") if outward else "tok"
        body.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{CW}" height="{CH}" rx="7"/>')
        body.append(f'<text class="lbl" x="{x + CW / 2}" y="{y + 23}">{tool}</text>')
        if outward:
            verdict = "chờ người duyệt" if asked else "chạy, không hỏi"
            body.append(f'<text class="cap mid" x="{x + CW / 2}" y="{y + 44}">{letters(CAPS[tool])}: {verdict}</text>')
        else:
            body.append(f'<text class="cap mid" x="{x + CW / 2}" y="{y + 44}">mang {letters(CAPS[tool])}</text>')
        body.append(f'<text class="sub mid" x="{x + CW / 2}" y="{y + CH + 18}">seen = {letters(seen)}</text>')
        if i:
            body.append(f'<path class="lfg" marker-end="url(#afg)" d="M{x - GAP + 2},{y + CH / 2}H{x - 3}"/>')

body.append(svgkit.legend(20, H - 18, [("blue", "C chạy luôn"), ("orange", "C chờ người duyệt"),
                                       ("tok", "tool chỉ đọc")]))

svgkit.write(__file__, "session-guard.svg", svgkit.svg(
    W, H, "Session cộng dồn thuộc tính của mọi tool đã gọi. Đọc đơn (B) rồi gửi email (C): không hỏi. "
    "Đọc đơn (B), đọc web (A) rồi gửi email: phải chờ người duyệt. Gửi email trước thì không hỏi, nhưng "
    "sau khi đọc email (A và B) thì lần gửi tiếp theo phải chờ duyệt", body))
