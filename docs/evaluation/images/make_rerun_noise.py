"""Sinh rerun-noise.svg: hình tĩnh, ba pass rate ở tình huống đầu bài kèm khoảng tin cậy 95% (§1.4).

Số liệu GIẢ ĐỊNH lấy từ tình huống thứ hai ở đầu bài: bộ 200 câu hỏi, model cũ 86%,
chạy lại model cũ 83%, model mới 84%. Khoảng tin cậy tính bằng công thức ở §1.4:
SE = sqrt(p(1 − p) / n), 95% CI ≈ p ± 1,96 × SE. Các con số in trong hình được assert.

Chạy: python3 make_rerun_noise.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

N = 200
ROWS = [("Model cũ, lần 1", 0.86, "blue"), ("Model cũ, lần 2", 0.83, "blue"), ("Model mới", 0.84, "orange")]


def half(p: float, n: int = N) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n)


HALF = {p: half(p) for _, p, _ in ROWS}
assert [round(HALF[p] * 100, 1) for p in (0.86, 0.83, 0.84)] == [4.8, 5.2, 5.1]
assert round(half(0.85) * 100) == 5      # §1.4: 200 ca, pass rate 85% → khoảng ±5 điểm


def fmt(v: float) -> str:
    return f"{v:.1f}".replace(".", ",")


W, H = 820, 340
X0, X1 = 230, 790
LO, HI = 76, 92                       # trục hoành, điểm %
assert all(LO < (p - HALF[p]) * 100 and (p + HALF[p]) * 100 < HI for p in HALF)


def x(pct: float) -> float:
    return X0 + (pct - LO) / (HI - LO) * (X1 - X0)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Cùng bộ 200 câu, ba lần chạy: chênh 2 điểm nằm trong nhiễu</text>')
body.append('<text class="cap2" x="20" y="62">Số liệu giả định từ tình huống đầu bài. '
            'Vạch ngang: khoảng tin cậy 95%.</text>')
body.append(svgkit.legend(20, 92, [("blue", "model cũ"), ("orange", "model mới")]))

PY0, PY1 = 112, 262
for v in range(LO, HI + 1, 4):
    body.append(f'<line class="grid" x1="{x(v)}" y1="{PY0}" x2="{x(v)}" y2="{PY1}"/>')
    body.append(f'<text class="sub mid" x="{x(v)}" y="{PY1 + 18}">{v}%</text>')
body.append(f'<text class="sub mid" x="{(X0 + X1) / 2}" y="{PY1 + 38}">Pass rate trên 200 câu</text>')

for i, (name, p, c) in enumerate(ROWS):
    yy = PY0 + 28 + i * 48
    lo, hi = (p - HALF[p]) * 100, (p + HALF[p]) * 100
    body.append(f'<text class="row" x="20" y="{yy + 5}">{name}</text>')
    body.append(f'<line class="l{c}" x1="{x(lo):.1f}" y1="{yy}" x2="{x(hi):.1f}" y2="{yy}" style="stroke-width:3"/>')
    for e in (lo, hi):
        body.append(f'<line class="l{c}" x1="{x(e):.1f}" y1="{yy - 8}" x2="{x(e):.1f}" y2="{yy + 8}" '
                    'style="stroke-width:3"/>')
    body.append(f'<circle class="{c}" cx="{x(p * 100):.1f}" cy="{yy}" r="7"/>')
    body.append(f'<text class="cap mid halo" x="{x(p * 100):.1f}" y="{yy - 13}">'
                f'{round(p * 100)}% ± {fmt(HALF[p] * 100)}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Ba khoảng chồng lên nhau gần hết. Hai lần chạy cùng model '
            'đã lệch 3 điểm, hơn cả 2 điểm giữa hai model.</text>')

svgkit.write(__file__, "rerun-noise.svg", svgkit.svg(
    W, H, "Trên 200 câu, model cũ đạt 86% rồi 83% khi chạy lại, model mới 84%; khoảng tin cậy 95% rộng khoảng "
    "±5 điểm nên ba khoảng chồng lên nhau và chênh 2 điểm giữa hai model nằm trong nhiễu (số liệu giả định)", body))
