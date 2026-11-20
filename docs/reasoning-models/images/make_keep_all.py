"""Sinh keep-all.svg: SVG tĩnh minh hoạ thinking block cũ được giữ hay bỏ ở Claude (§1.5).

Request của lượt người dùng thứ ba, đang giữa một tool loop. Bạn gửi lại mọi block
nguyên văn trong cả hai trường hợp; khác nhau ở phần model thực sự đọc:
- Keep-all (Opus 4.5 trở lên, Sonnet 4.6 trở lên, Haiku 5.5, Fable, Mythos): giữ thinking
  của mọi lượt, nên thinking cũ còn nằm trong context và cache, tính tiền input hoặc
  cache read ở mỗi lượt sau.
- Last-turn-only (Opus/Sonnet cũ hơn, mọi Haiku đến 4.5): API tự bỏ thinking của các lượt
  trước, chỉ giữ thinking của lượt đang chạy, nơi tool loop cần nó.
Nguồn: docs Claude Thinking [5], tóm tắt ở bảng §1.5. Hội thoại là ví dụ, không có số liệu.

Chạy: python3 make_keep_all.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

# (nhãn, độ rộng, loại); loại "old" là thinking của lượt trước, "cur" là thinking lượt đang chạy
ITEMS = [
    ("user 1", 62, "msg"), ("thinking", 80, "old"), ("text", 46, "msg"),
    ("user 2", 62, "msg"), ("thinking", 80, "old"), ("text", 46, "msg"),
    ("user 3", 62, "msg"), ("thinking", 80, "cur"), ("tool_use", 78, "msg"), ("tool_result", 100, "msg"),
]
X0, GAP, BH = 196, 5, 36
xs = []
xc = X0
for _, w, _ in ITEMS:
    xs.append(xc)
    xc += w + GAP
W = int(xc - GAP + 24)
ROWS = [  # (tiêu đề, dòng phụ, y)
    ("Bạn gửi", "giống nhau ở cả hai kiểu", 118),
    ("Keep-all", "Opus 4.5+, Sonnet 4.6+…", 206),
    ("Last-turn-only", "Opus/Sonnet cũ hơn…", 294),
]
H = 384

body: list[str] = []
body.append('<text class="title" x="20" y="38">Claude: thinking của lượt trước được giữ hay bỏ</text>')
body.append('<text class="cap2" x="20" y="62">Request ở lượt người dùng thứ 3, giữa tool loop. '
            'Bạn luôn gửi lại mọi block; API quyết định model đọc gì.</text>')
body.append(svgkit.legend(W - 360, 38, [("blue", "thinking model đọc"), ("slot", "thinking API tự bỏ")]))

# Ngoặc đánh dấu lượt đang chạy (user 3 trở đi), đặt trên hàng đầu
b0, b1 = xs[6], xs[-1] + ITEMS[-1][1]
by = ROWS[0][2] - 12
body.append(f'<path class="lmuted" d="M{b0},{by + 6} V{by} H{b1} V{by + 6}"/>')
body.append(f'<text class="sub mid" x="{(b0 + b1) / 2}" y="{by - 6}">lượt đang chạy (tool loop)</text>')

for r, (title, sub, y) in enumerate(ROWS):
    body.append(f'<text class="row" x="20" y="{y + 16}">{title}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 33}">{sub}</text>')
    for (label, w, kind), x in zip(ITEMS, xs):
        if r == 0 or kind == "msg":
            cls = "tok"
        elif r == 2 and kind == "old":
            cls = "slot"
        else:
            cls = "blue"
        body.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
                    f'<text class="lbl" style="font-size:13px" x="{x + w / 2}" y="{y + BH / 2 + 5}">{label}</text>')

notes = {
    1: "Thinking cũ ở lại trong context và cache: lượt sau trả tiền input hoặc cache read cho nó.",
    2: "Thinking của lượt trước bị bỏ; thinking của lượt đang chạy được giữ cho tool loop.",
}
for r, text in notes.items():
    body.append(f'<text class="cap" x="{X0}" y="{ROWS[r][2] + BH + 24}">{text}</text>')

assert xs[-1] + ITEMS[-1][1] <= W - 20

svgkit.write(__file__, "keep-all.svg", svgkit.svg(
    W, H, "Claude: bạn luôn gửi lại mọi thinking block. Model keep-all đọc thinking của mọi lượt, "
    "nên thinking cũ tiếp tục tốn input hoặc cache read ở các lượt sau; với model last-turn-only, "
    "API tự bỏ thinking của các lượt trước và chỉ giữ thinking của lượt đang chạy trong tool loop", body))
