"""Sinh cache-threshold.svg: hình tĩnh minh họa vì sao không ngưỡng tĩnh nào tách sạch hai nhóm (§2.11).

Số liệu GIẢ ĐỊNH, chỉ để minh họa hình dạng vấn đề: 20 cặp câu hỏi dùng chung được câu trả lời
và 13 cặp cần câu trả lời khác nhau (đảo chiều, phủ định, khác con số...), mỗi cặp có một cosine.
Hai nhóm nằm xen kẽ trong cùng khoảng điểm, nên nâng ngưỡng từ 0.90 lên 0.95 giảm trúng nhầm
nhưng vẫn không về 0, đồng thời bỏ lỡ nhiều cặp dùng lại được. Các con số đếm được assert bên dưới.

Chạy: python3 make_cache_threshold.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

# Không đặt điểm nào đúng bằng ngưỡng để khỏi mơ hồ. Cosine nhân 1000 để so sánh số nguyên, tránh sai số dấu phẩy động
SAME = [810, 840, 860, 870, 880, 890, 906, 912, 918, 924, 930, 935, 940, 945, 956, 963, 970, 977, 984, 991]
DIFF = [780, 800, 820, 830, 850, 860, 875, 890, 905, 920, 940, 955, 965]
THRESHOLDS = [900, 950]
NOTES = {920: ("hủy ↔ đổi gói", "middle"), 965: ("quý 2 ↔ quý 3", "start")}


def stats(t: int) -> tuple[int, int, int]:
    hit = sum(v >= t for v in SAME)
    false_hit = sum(v >= t for v in DIFF)
    return hit, false_hit, len(SAME) - hit


# Mỗi điểm cách ngưỡng ít nhất 0.004 (> bán kính dấu) để thấy rõ nằm bên nào
assert all(abs(v - t) >= 4 for v in SAME + DIFF for t in THRESHOLDS)
# Hai dấu cạnh nhau trong cùng hàng không chồng lên nhau (≥ 0.005 ≈ 13 px, dấu rộng 12 px)
assert all(b - a >= 5 for row in (SAME, DIFF) for a, b in zip(sorted(row), sorted(row)[1:]))
assert stats(900) == (14, 5, 6)
assert stats(950) == (6, 2, 14)

W, H = 860, 390
PX0, PX1, LO, HI = 200, 830, 760, 1000
Y_SAME, Y_DIFF = 150, 225


def x(v: float) -> float:
    return PX0 + (v - LO) / (HI - LO) * (PX1 - PX0)


def num(v: int) -> str:
    return f"{v / 1000:.2f}"


body: list[str] = []
body.append('<text class="title" x="20" y="38">Không ngưỡng tĩnh nào tách sạch “cùng ý” và “khác ý”</text>')
body.append('<text class="cap2" x="20" y="62">Mỗi dấu là một cặp câu hỏi, đặt theo cosine giữa chúng. '
            'Số liệu giả định.</text>')

body.append(f'<text class="row" x="20" y="{Y_SAME + 5}">Dùng chung đáp án</text>')
body.append(f'<text class="sub" x="20" y="{Y_SAME + 22}">nên trả từ cache</text>')
body.append(f'<text class="row" x="20" y="{Y_DIFF + 5}">Cần đáp án khác</text>')
body.append(f'<text class="sub" x="20" y="{Y_DIFF + 22}">trả từ cache là sai</text>')

AX = Y_DIFF + 50
for v in range(LO, HI + 1, 40):
    body.append(f'<line class="grid" x1="{x(v)}" y1="{Y_SAME - 30}" x2="{x(v)}" y2="{AX}"/>')
    body.append(f'<text class="sub mid" x="{x(v)}" y="{AX + 18}">{num(v)}</text>')
body.append(f'<line class="axis" x1="{PX0}" y1="{AX}" x2="{PX1}" y2="{AX}"/>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{AX + 36}">cosine giữa câu hỏi mới và câu hỏi đã cache</text>')

for t in THRESHOLDS:
    body.append(f'<line class="lfg dash" x1="{x(t)}" y1="{Y_SAME - 40}" x2="{x(t)}" y2="{AX}"/>')
    body.append(f'<text class="sub mid halo" x="{x(t)}" y="{Y_SAME - 46}">ngưỡng {num(t)}</text>')

for v in SAME:
    body.append(f'<circle class="blue" cx="{x(v):.1f}" cy="{Y_SAME}" r="6"/>')
# Dấu × có nền sọc: mã hoá phụ ngoài màu đỏ cho "trả sai"
for v in DIFF:
    cx, cy, s = x(v), Y_DIFF, 6
    body.append(f'<rect class="hatch" x="{cx - s:.1f}" y="{cy - s}" width="{2 * s}" height="{2 * s}" rx="2"/>')
    body.append(f'<path class="lred" d="M{cx - 3:.1f},{cy - 3}L{cx + 3:.1f},{cy + 3}M{cx - 3:.1f},{cy + 3}L{cx + 3:.1f},{cy - 3}"/>')
for v, (label, anchor) in NOTES.items():
    assert v in DIFF
    dx = 8 if anchor == "start" else 0
    body.append(f'<text class="sub halo" x="{x(v) + dx:.1f}" y="{Y_DIFF + 24}" style="text-anchor:{anchor}">{label}</text>')

# Đếm theo từng ngưỡng
y = AX + 70
for t in THRESHOLDS:
    hit, false_hit, miss = stats(t)
    body.append(f'<text class="cap" x="20" y="{y}"><tspan style="font-weight:600">Ngưỡng {num(t)}:</tspan> '
                f'{hit} lần trả đúng từ cache, {false_hit} lần trúng nhầm, '
                f'{miss}/{len(SAME)} cặp dùng lại được bị bỏ lỡ.</text>')
    y += 24

svgkit.write(__file__, "cache-threshold.svg", svgkit.svg(
    W, H, "Cosine của các cặp câu hỏi dùng chung đáp án và các cặp cần đáp án khác nằm xen kẽ nhau; "
    "nâng ngưỡng từ 0.90 lên 0.95 giảm trúng nhầm từ 5 xuống 2 nhưng bỏ lỡ 14 trên 20 cặp dùng lại được",
    body))
