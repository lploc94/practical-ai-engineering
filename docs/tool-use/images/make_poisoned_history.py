"""Sinh poisoned-history.svg: SVG tĩnh minh hoạ lịch sử bị "nhiễm độc" (§3.4).

Một hội thoại năm lượt. Ở lượt 2, code lưu assistant message có tool call rồi ném
exception trước khi lưu tool result. Hàng trên: không sửa gì, lịch sử còn một tool
call mồ côi nên mọi request từ lượt 3 đều trả 400. Hàng dưới: load_history thêm
một kết quả lỗi INTERRUPTED cho call mồ côi, hội thoại chạy tiếp bình thường.

Chạy: python3 make_poisoned_history.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

N, CRASH = 5, 1               # năm lượt, crash ở lượt thứ 2 (chỉ số 1)
BW, GAP, BH = 132, 14, 54
X0 = 150
xs = [X0 + i * (BW + GAP) for i in range(N)]
W, H = xs[-1] + BW + 24, 352
R1, R2 = 118, 248

body: list[str] = []


def turn(i: int, y: float, cls: str, top: str, bottom: str = "", tcls: str = "") -> str:
    x = xs[i]
    out = (f'<rect class="{cls}" x="{x}" y="{y}" width="{BW}" height="{BH}" rx="7"/>'
           f'<text class="row mid halo" x="{x + BW / 2}" y="{y + 23}">{top}</text>')
    if bottom:
        out += f'<text class="sub mid halo {tcls}" x="{x + BW / 2}" y="{y + 42}">{bottom}</text>'
    return out


body.append('<text class="title" x="20" y="38">Một exception làm hỏng cả hội thoại</text>')
body.append(svgkit.legend(W - 330, 38, [("blue", "kết quả lỗi được thêm"), ("hatch", "request trả 400")]))
body.append('<text class="cap2" x="20" y="62">Lượt 2: đã lưu tool call, exception xảy ra trước khi lưu tool result.</text>')

for i in range(N):
    body.append(f'<text class="sub mid" x="{xs[i] + BW / 2}" y="{R1 - 12}">lượt {i + 1}</text>')

# Hàng 1: không sửa lịch sử
body.append(f'<text class="row" x="20" y="{R1 + 23}">Không sửa</text>')
body.append(f'<text class="sub" x="20" y="{R1 + 41}">lịch sử</text>')
for i in range(N):
    if i < CRASH:
        body.append(turn(i, R1, "tok", "OK"))
    elif i == CRASH:
        body.append(turn(i, R1, "tok", "tool call", "crash, thiếu result", "tred"))
    else:
        body.append(turn(i, R1, "hatch", "400", "call mồ côi"))
body.append(f'<text class="cap" x="{X0}" y="{R1 + BH + 26}">Mọi request sau đó đều 400, dù lỗi gốc chỉ xảy ra một lần.</text>')

# Hàng 2: load_history thêm kết quả INTERRUPTED
body.append(f'<text class="row" x="20" y="{R2 + 23}">load_history</text>')
body.append(f'<text class="sub" x="20" y="{R2 + 41}">vá call mồ côi</text>')
for i in range(N):
    if i < CRASH:
        body.append(turn(i, R2, "tok", "OK"))
    elif i == CRASH:
        body.append(turn(i, R2, "tok", "tool call", "crash, thiếu result", "tred"))
    elif i == CRASH + 1:
        body.append(turn(i, R2, "blue", "OK", "+ INTERRUPTED"))
    else:
        body.append(turn(i, R2, "tok", "OK"))
body.append(f'<text class="cap" x="{X0}" y="{R2 + BH + 26}">Model đọc "chưa rõ đã thực hiện hay chưa" và kiểm tra '
            'trạng thái trước khi thử lại.</text>')

svgkit.write(__file__, "poisoned-history.svg", svgkit.svg(
    W, H, "Tool call được lưu mà không có tool result: nếu không sửa, mọi request sau đó của hội thoại đều trả 400; "
    "load_history thêm kết quả lỗi INTERRUPTED thì hội thoại chạy tiếp", body))
