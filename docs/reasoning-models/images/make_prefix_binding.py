"""Sinh prefix-binding.svg: SVG tĩnh minh hoạ sửa lịch sử làm mất hiệu lực thinking block (§1.5).

Từ Claude 5.1, signature của mỗi thinking block ghi lại prefix đã sinh ra nó: system,
tools và mọi message trước đó [9]. Sửa một phần trong prefix thì block đó và mọi block
phía sau mất hiệu lực; API trả 400 hoặc bỏ block, tuỳ tài khoản và cấu hình
thinking.block_binding.prefix_mismatch_behavior. Nối thêm message mới ở cuối không đụng
tới prefix của block nào. Hội thoại là ví dụ, không có số liệu.

Chạy: python3 make_prefix_binding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

# (nhãn, độ rộng); vị trí thinking block được suy ra từ nhãn
ITEMS = [("system", 68), ("tools", 58), ("user 1", 62), ("thinking 1", 94), ("text", 46),
         ("user 2", 62), ("thinking 2", 94), ("text", 46), ("user 3", 62)]
THINK = [i for i, (lbl, _) in enumerate(ITEMS) if lbl.startswith("thinking")]
X0, GAP, BH = 196, 5, 36
xs = []
xc = X0
for _, w in ITEMS:
    xs.append(xc)
    xc += w + GAP
W = int(xc - GAP + 24)

# (tiêu đề, vị trí bị sửa hoặc None, vị trí mới nối thêm hoặc None, y)
ROWS = [
    ("Chỉ nối thêm", None, 8, 116),
    ("Sửa user 2", 5, None, 214),
    ("Sửa system", 0, None, 312),
]
H = 412

body: list[str] = []
body.append('<text class="title" x="20" y="38">Claude 5.1+: sửa phần trước làm hỏng thinking phía sau</text>')
body.append('<text class="cap2" x="20" y="62">Signature của thinking block ghi lại prefix sinh ra nó: '
            'system, tools và mọi message trước nó.</text>')
body.append(svgkit.legend(W - 300, 92, [("blue", "còn hợp lệ"), ("hatch", "mất hiệu lực")]))

for title, edited, new, y in ROWS:
    body.append(f'<text class="row" x="20" y="{y + 23}">{title}</text>')
    for i, ((label, w), x) in enumerate(zip(ITEMS, xs)):
        if i == new:
            continue
        if i in THINK:
            bad = edited is not None and edited < i
            cls = "hatch" if bad else "blue"
        else:
            cls = "tok"
        tcls = "lbl halo" if cls == "hatch" else "lbl"
        body.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
                    f'<text class="{tcls}" style="font-size:13px" x="{x + w / 2}" y="{y + BH / 2 + 5}">{label}</text>')
        if i == edited:
            # Phần tử bị sửa: viền đậm màu chữ và nhãn ngay bên trên
            body.append(f'<rect class="lfg" style="stroke-width:3" x="{x - 3}" y="{y - 3}" width="{w + 6}" '
                        f'height="{BH + 6}" rx="9"/>'
                        f'<text class="sub mid" style="fill:var(--fg)" x="{x + w / 2}" y="{y - 9}">sửa</text>')
    if new is not None:
        x, (label, w) = xs[new], ITEMS[new]
        body.append(f'<rect class="tok" style="stroke-width:2.5" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
                    f'<text class="lbl" style="font-size:13px" x="{x + w / 2}" y="{y + BH / 2 + 5}">{label}</text>'
                    f'<text class="sub mid" style="fill:var(--fg)" x="{x + w / 2}" y="{y - 9}">+ mới</text>')

notes = {
    0: "Prefix của mọi thinking block giữ nguyên: cả hai còn hợp lệ.",
    1: "Thinking 1 sinh trước user 2 nên không bị ảnh hưởng; thinking 2 mất hiệu lực.",
    2: "System nằm trong prefix của mọi block: cả hai thinking mất hiệu lực.",
}
for r, text in notes.items():
    body.append(f'<text class="cap" x="{X0}" y="{ROWS[r][3] + BH + 24}">{text}</text>')
body.append(f'<text class="cap2" x="20" y="{H - 16}">Block mất hiệu lực: API trả 400 hoặc bỏ block, '
            'tuỳ tài khoản và thinking.block_binding.</text>')

assert xs[-1] + ITEMS[-1][1] <= W - 20

svgkit.write(__file__, "prefix-binding.svg", svgkit.svg(
    W, H, "Claude 5.1 trở lên: nối thêm message ở cuối giữ mọi thinking block hợp lệ; sửa user 2 làm "
    "mất hiệu lực thinking sinh sau nó nhưng thinking sinh trước vẫn hợp lệ; sửa system prompt làm mất hiệu lực mọi thinking block", body))
