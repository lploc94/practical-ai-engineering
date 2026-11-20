"""Sinh paired-diff.svg: hình tĩnh, so hai phiên bản như hai số độc lập và so theo cặp (§1.4).

Số liệu GIẢ ĐỊNH, dựa trên tình huống đầu bài: 200 câu, model cũ đúng 172 câu (86%),
model mới đúng 168 câu (84%). Giả định thêm cách hai model đúng sai trên từng câu:
162 câu cả hai đúng, 22 câu cả hai sai, 10 câu chỉ model cũ đúng, 6 câu chỉ model mới đúng.

Hiệu số từng câu d = mới − cũ ∈ {−1, 0, +1}; trung bình = (6 − 10) / 200 = −0,02.
Khoảng tin cậy 95% xấp xỉ chuẩn (đủ chính xác với n = 200; bootstrap theo cặp ở §1.4 cho
kết quả gần như vậy):
- Hai số độc lập: SE = sqrt(p₁(1 − p₁)/n + p₂(1 − p₂)/n) → −2 ± 7,0 điểm.
- Theo cặp: SE = sqrt(Var(d)/n) → −2 ± 3,9 điểm.
Các con số in trong hình được assert.

Chạy: python3 make_paired_diff.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

N = 200
BOTH, NONE, OLD_ONLY, NEW_ONLY = 162, 22, 10, 6
assert BOTH + NONE + OLD_ONLY + NEW_ONLY == N
assert BOTH + OLD_ONLY == 172 and BOTH + NEW_ONLY == 168      # 86% và 84%
p_old, p_new = 172 / N, 168 / N

d = [0] * (BOTH + NONE) + [-1] * OLD_ONLY + [1] * NEW_ONLY
mean = sum(d) / N
var = sum((v - mean) ** 2 for v in d) / N
half_pair = 1.96 * math.sqrt(var / N)
half_ind = 1.96 * math.sqrt(p_old * (1 - p_old) / N + p_new * (1 - p_new) / N)
assert round(mean * 100, 1) == -2.0
assert round(half_ind * 100, 1) == 7.0 and round(half_pair * 100, 1) == 3.9
ROWS = [("Hai số độc lập", half_ind), ("So theo cặp", half_pair)]
assert [round((mean + s * h) * 100, 1) for _, h in ROWS for s in (-1, 1)] == [-9.0, 5.0, -5.9, 1.9]


def fmt(v: float) -> str:
    s = f"{abs(v):.1f}".replace(".", ",")
    return ("−" if v < 0 else "+") + s if round(v, 1) else "0"


W, H = 820, 500
body: list[str] = []
body.append('<text class="title" x="20" y="38">So theo cặp: cùng 200 câu, khoảng tin cậy hẹp hơn</text>')
body.append('<text class="cap2" x="20" y="62">Số liệu giả định: model cũ đúng 172/200 (86%), '
            'model mới đúng 168/200 (84%).</text>')

# Phần trên: lưới 200 câu, xếp theo nhóm
COLS, CELL, GAP = 25, 13, 3
GX, GY = 20, 92
cats = [("tok", BOTH), ("blue", OLD_ONLY), ("orange", NEW_ONLY), ("slot", NONE)]
k = 0
for cls, cnt in cats:
    for _ in range(cnt):
        r, c = divmod(k, COLS)
        body.append(f'<rect class="{cls}" x="{GX + c * (CELL + GAP)}" y="{GY + r * (CELL + GAP)}" '
                    f'width="{CELL}" height="{CELL}" rx="2"/>')
        k += 1
grid_w = COLS * (CELL + GAP) - GAP
grid_h = (N // COLS) * (CELL + GAP) - GAP

LX = GX + grid_w + 34
items = [("tok", f"Cả hai đúng: {BOTH} câu", "hiệu số 0"),
         ("blue", f"Chỉ model cũ đúng: {OLD_ONLY} câu", "hiệu số −1"),
         ("orange", f"Chỉ model mới đúng: {NEW_ONLY} câu", "hiệu số +1"),
         ("slot", f"Cả hai sai: {NONE} câu", "hiệu số 0")]
for i, (cls, a, b) in enumerate(items):
    yy = GY + 6 + i * 30
    body.append(f'<rect class="{cls}" x="{LX}" y="{yy - 1}" width="{CELL}" height="{CELL}" rx="2"/>')
    body.append(f'<text class="cap" x="{LX + 22}" y="{yy + 11}">{a}</text>')
    body.append(f'<text class="cap2 end" x="{W - 20}" y="{yy + 11}">{b}</text>')
body.append(f'<text class="cap" x="{LX}" y="{GY + 6 + 4 * 30 + 14}">Hiệu số chỉ khác 0 ở 16 câu bất đồng.</text>')

# Phần dưới: khoảng tin cậy của hiệu số
X0, X1, LO, HI = 200, 780, -10, 6
PY0 = GY + 6 + 4 * 30 + 52


def x(pct: float) -> float:
    return X0 + (pct - LO) / (HI - LO) * (X1 - X0)


body.append(f'<text class="row" x="20" y="{PY0}">Hiệu số mới − cũ, khoảng tin cậy 95%</text>')
PT, PB = PY0 + 16, PY0 + 136
for v in range(LO, HI + 1, 2):
    cls = "lfg dash" if v == 0 else "grid"
    body.append(f'<line class="{cls}" x1="{x(v)}" y1="{PT}" x2="{x(v)}" y2="{PB}"/>')
    lab = "0" if v == 0 else ("−" if v < 0 else "+") + str(abs(v))
    body.append(f'<text class="sub mid" x="{x(v)}" y="{PB + 18}">{lab}</text>')
body.append(f'<text class="sub mid" x="{(X0 + X1) / 2}" y="{PB + 36}">điểm %</text>')

for i, (name, h) in enumerate(ROWS):
    yy = PT + 34 + i * 58
    lo, hi = (mean - h) * 100, (mean + h) * 100
    body.append(f'<text class="row" x="20" y="{yy + 5}">{name}</text>')
    body.append(f'<line class="lfg" x1="{x(lo):.1f}" y1="{yy}" x2="{x(hi):.1f}" y2="{yy}" style="stroke-width:3"/>')
    for e in (lo, hi):
        body.append(f'<line class="lfg" x1="{x(e):.1f}" y1="{yy - 8}" x2="{x(e):.1f}" y2="{yy + 8}" '
                    'style="stroke-width:3"/>')
        body.append(f'<text class="sub mid halo" x="{x(e):.1f}" y="{yy + 24}">{fmt(e)}</text>')
    body.append(f'<circle class="ffg" cx="{x(mean * 100):.1f}" cy="{yy}" r="6"/>')
    body.append(f'<text class="cap mid halo" x="{x(mean * 100):.1f}" y="{yy - 13}">'
                f'−2 ± {fmt(h * 100)[1:]}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Cả hai khoảng đều chứa 0: chưa đủ bằng chứng. '
            'Nhưng theo cặp chỉ ±3,9 thay vì ±7,0, không tốn thêm lần gọi nào.</text>')

svgkit.write(__file__, "paired-diff.svg", svgkit.svg(
    W, H, "200 câu, model cũ đúng 86%, model mới 84%: 184 câu hai model cùng đúng hoặc cùng sai nên hiệu số bằng 0, "
    "chỉ 16 câu bất đồng; so như hai số độc lập khoảng tin cậy của hiệu số là −2 ± 7,0 điểm, so theo cặp là "
    "−2 ± 3,9 điểm (số liệu giả định)", body))
