"""Sinh thinking-switch.svg: SVG tĩnh về thinking block khi fallback rồi quay về (§1.5 d).

Theo trang Preserved thinking của Claude ([7], truy cập tháng 11/2026):
- Opus 5.5 đọc block của chính nó và của Opus 4.8 ("Every Claude model reads
  blocks from Claude Opus 4.8...").
- Opus 4.8 chỉ đọc block của Opus 4.8, Sonnet 5, Haiku 4.5 và các model cũ hơn,
  nên không đọc block của Opus 5.5.
- Block model đích không đọc được bị API bỏ cho request đó, không lỗi, không tính
  tiền; API không sửa mảng messages nên quay về model cũ thì block được đọc lại.
Kịch bản: lượt 1 và 2 do Opus 5.5 trả lời, lượt 3 fallback sang Opus 4.8, lượt 4
quay về Opus 5.5. Mỗi cột là một request, mỗi hàng là thinking block của một
lượt trước đó trong lịch sử.
Hình tĩnh: đây là bảng trạng thái theo từng request, nhìn cùng lúc dễ so hơn.

Chạy: python3 make_thinking_switch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
assert "ví dụ Opus 5.5 sang Opus 4.8, và đó đúng là chiều mất reasoning" in README
assert "dùng lại được nếu hội thoại quay về model cũ [7]" in README

READS = {"Opus 5.5": {"Opus 5.5", "Opus 4.8"}, "Opus 4.8": {"Opus 4.8"}}
TURNS = ["Opus 5.5", "Opus 5.5", "Opus 4.8", "Opus 5.5"]       # model trả lời lượt 1..4
REQS = [2, 3, 4]                                                # request của lượt 2, 3, 4

grid = {}
for r in REQS:
    target = TURNS[r - 1]
    for prev in range(1, r):
        grid[(prev, r)] = TURNS[prev - 1] in READS[target]
assert grid == {(1, 2): True, (1, 3): False, (2, 3): False, (1, 4): True, (2, 4): True, (3, 4): True}

W, H = 900, 400
CX0, CWD, CGAP = 230, 190, 20
RY0, RH, RGAP = 150, 40, 12
body: list[str] = []
body.append('<text class="title" x="20" y="38">Fallback làm mất reasoning, quay về thì đọc lại</text>')
body.append(svgkit.legend(20, 66, [("blue", "model đọc block"), ("slot", "API bỏ block cho request này")]))

for ci, r in enumerate(REQS):
    x = CX0 + ci * (CWD + CGAP)
    target = TURNS[r - 1]
    tag = " (fallback)" if target != TURNS[0] else ""
    body.append(f'<text class="row mid" x="{x + CWD / 2}" y="{RY0 - 34}">Request lượt {r}</text>')
    body.append(f'<text class="sub mid" x="{x + CWD / 2}" y="{RY0 - 14}">gửi tới {target}{tag}</text>')
    for prev in range(1, r):
        y = RY0 + (prev - 1) * (RH + RGAP)
        read = grid[(prev, r)]
        cls, lab = ("blue", "đọc") if read else ("slot", "bỏ, không lỗi")
        body.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{CWD}" height="{RH}" rx="6"/>'
                    f'<text class="cap mid" x="{x + CWD / 2}" y="{y + RH / 2 + 5}">{lab}</text>')

for prev in range(1, len(TURNS)):
    y = RY0 + (prev - 1) * (RH + RGAP)
    body.append(f'<text class="cap" x="20" y="{y + 17}">Thinking lượt {prev}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 33}">do {TURNS[prev - 1]} tạo</text>')

yb = RY0 + 3 * (RH + RGAP) + 14
body.append(f'<text class="cap" x="20" y="{yb + 10}">Ba request dùng chung một lịch sử, mỗi lượt chỉ nối thêm. Opus 4.8 không đọc '
            'block của Opus 5.5,</text>')
body.append(f'<text class="cap" x="20" y="{yb + 30}">nên trả lời lượt 3 mà không có reasoning trước đó. Quay về Opus 5.5, '
            'mọi block được đọc lại.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 16}">Block bị bỏ không tính tiền cho request đó. Đừng xoá thinking khỏi '
            'lịch sử khi đổi model.</text>')
assert CX0 + 3 * CWD + 2 * CGAP <= W - 30

svgkit.write(__file__, "thinking-switch.svg", svgkit.svg(
    W, H, "Lượt 1 và 2 do Opus 5.5 trả lời, lượt 3 fallback sang Opus 4.8: Opus 4.8 không đọc thinking block "
    "của Opus 5.5 nên API bỏ chúng cho request đó, không báo lỗi. Lượt 4 quay về Opus 5.5 với cùng lịch sử: "
    "mọi block, kể cả block của Opus 4.8, được đọc lại", body))
