"""Sinh cascade-verify.svg: SVG tĩnh, ví dụ cascade trích xuất ở §3.7.

Số liệu lấy từ "Kết quả tham khảo" ở §3.7 (cookbook TypeSafe [28]): gpt-5.4-mini để
trống registration_open_date (đúng) nhưng chép ví dụ trong schema vào description.
Jev cho description 0.98 (hallucinated), 0.97 (off_target), 0.86 (unreasonable);
field ngày để trống 0.05 (absence_wrong). Max 0.98 > FIRE_T 0.7 nên cascade gọi
gpt-5.5 với reasoning_effort="high", model này trả description rỗng.

Chạy: python3 make_cascade_verify.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402
from svgkit import MONO  # noqa: E402

FIRE_T = 0.7
flags = [
    ("description", "hallucinated", 0.98),
    ("description", "off_target", 0.97),
    ("description", "unreasonable", 0.86),
    ("registration_open_date", "absence_wrong", 0.05),
]
top = max(p for *_, p in flags)
assert top == 0.98 and top > FIRE_T
assert sum(p > FIRE_T for *_, p in flags) == 3

W, H = 820, 486
body: list[str] = []
body.append(f'<style>.code{{font-family:{MONO};font-size:12.5px}}</style>')
body.append('<text class="title" x="20" y="38">Cascade: model nhỏ trích, decision model kiểm, model lớn chỉ khi cần</text>')
body.append('<text class="cap2" x="20" y="62">Ví dụ ở §3.7: trang lịch sự kiện không có ngày mở đăng ký. '
            'Record sai nhưng vẫn hợp lệ với schema.</text>')
body.append(svgkit.legend(20, 92, [("blue", "đầu vào"), ("hatch", "cờ vượt ngưỡng")]))

# Hàng 1: source → model nhỏ → record
R1, RH = 112, 64
body.append(svgkit.box(20, R1, 130, RH, "blue"))
body.append(f'<text class="row mid" x="85" y="{R1 + 28}">source</text>'
            f'<text class="sub mid" x="85" y="{R1 + 47}">trang sự kiện</text>')
body.append(f'<line class="lfg" x1="154" y1="{R1 + RH / 2}" x2="186" y2="{R1 + RH / 2}" marker-end="url(#afg)"/>')
body.append(svgkit.box(192, R1, 150, RH, "tok"))
body.append(f'<text class="row mid" x="267" y="{R1 + 28}">gpt-5.4-mini</text>'
            f'<text class="sub mid" x="267" y="{R1 + 47}">trích xuất, rẻ</text>')
body.append(f'<line class="lfg" x1="346" y1="{R1 + RH / 2}" x2="378" y2="{R1 + RH / 2}" marker-end="url(#afg)"/>')
body.append(svgkit.box(384, R1, 416, RH, "tok"))
body.append(f'<text class="code" x="398" y="{R1 + 26}">registration_open_date: ""</text>'
            f'<text class="code" x="398" y="{R1 + 47}">description: "Registration opens for the fall…"</text>')

# Hàng 2: Jev chấm từng field × kiểu lỗi
R2 = 256
BX0, BX1, BH, BSTEP = 330, 610, 22, 34
body.append(f'<line class="lfg" x1="592" y1="{R1 + RH + 4}" x2="592" y2="{R2 - 40}" marker-end="url(#afg)"/>')
body.append(f'<text class="row" x="20" y="{R2 - 30}">Jev kiểm từng field, mỗi kiểu lỗi một câu yes/no</text>')
body.append(f'<text class="sub" x="20" y="{R2 - 12}">true nghĩa là có vấn đề</text>')


def bx(p: float) -> float:
    return BX0 + p * (BX1 - BX0)


for i, (field, head, p) in enumerate(flags):
    y = R2 + i * BSTEP
    body.append(f'<text class="code end" x="{BX0 - 12}" y="{y + 16}">{field} · {head}</text>')
    body.append(f'<rect class="slot" x="{BX0}" y="{y}" width="{BX1 - BX0}" height="{BH}" rx="3"/>')
    cls = "hatch" if p > FIRE_T else "tok"
    body.append(f'<rect class="{cls}" x="{BX0}" y="{y}" width="{bx(p) - BX0:.1f}" height="{BH}" rx="3"/>')
    body.append(f'<text class="sub halo" x="{bx(p) + 6:.1f}" y="{y + 16}">{p:.2f}</text>')
yb = R2 + len(flags) * BSTEP
body.append(f'<line class="lfg dash" x1="{bx(FIRE_T)}" y1="{R2 - 6}" x2="{bx(FIRE_T)}" y2="{yb}"/>')
body.append(f'<text class="sub mid" x="{bx(FIRE_T)}" y="{yb + 16}">FIRE_T = {FIRE_T}</text>')

# Bên phải: quyết định theo max
DX = 676
body.append(f'<line class="lfg" x1="{BX1 + 38}" y1="{R2 + 78}" x2="{DX - 6}" y2="{R2 + 78}" marker-end="url(#afg)"/>')
body.append(f'<text class="row" x="{DX}" y="{R2 + 14}">max = {top:.2f}</text>')
body.append(f'<text class="sub" x="{DX}" y="{R2 + 32}">&gt; {FIRE_T}: escalate</text>')
body.append(svgkit.box(DX, R2 + 46, 124, 64, "tok"))
body.append(f'<text class="row mid" x="{DX + 62}" y="{R2 + 74}">gpt-5.5</text>'
            f'<text class="sub mid" x="{DX + 62}" y="{R2 + 93}">reasoning high</text>')
body.append(f'<text class="sub" x="{DX}" y="{R2 + 132}">trả description rỗng</text>')

body.append(f'<text class="cap" x="20" y="{H - 38}">Escalate theo max, không theo trung bình: một cờ chắc chắn là đủ.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 14}">Câu hỏi tách theo field và kiểu lỗi nên biết field nào sai, '
            'sai kiểu gì. Số liệu từ cookbook TypeSafe.</text>')

svgkit.write(__file__, "cascade-verify.svg", svgkit.svg(
    W, H, "Cascade trích xuất: gpt-5.4-mini chép ví dụ của schema vào description; Jev chấm description 0.98, 0.97 và "
    "0.86 ở ba kiểu lỗi, field ngày để trống 0.05; max 0.98 vượt ngưỡng 0.7 nên cascade gọi gpt-5.5 và nhận "
    "description rỗng", body))
