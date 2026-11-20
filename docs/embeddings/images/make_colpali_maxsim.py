"""Sinh colpali-maxsim.svg: hình tĩnh minh họa MaxSim của ColPali so với single-vector (§2.8.5).

Ví dụ lấy từ bài: câu hỏi "doanh thu quý 3 miền Nam"; mỗi token khớp mạnh nhất với một patch ở
một góc khác nhau của trang. Lưới được thu nhỏ còn 6 × 8 = 48 patch để vẽ được (ColPali thật có
khoảng 1.030 patch mỗi trang). Điểm khớp của từng token là GIẢ ĐỊNH; tổng được assert.

Chạy: python3 make_colpali_maxsim.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

COLS, ROWS = 6, 8
PX, PY, PW, PH = 250, 120, 70, 40  # góc trang và kích thước một patch
TOKENS = [  # (token, patch (cột, hàng) khớp nhất, điểm khớp giả định × 100)
    ("miền Nam", (5, 0), 71),
    ("quý 3", (4, 2), 77),
    ("doanh thu", (0, 5), 82),
]
total = sum(s for _, _, s in TOKENS)
assert total == 230
N_PATCH = COLS * ROWS
assert N_PATCH == 48

W, H = 960, 560
RIGHT = PX + COLS * PW


def pc(c: float, r: float) -> tuple[float, float]:
    """Toạ độ tâm patch (c, r)."""
    return PX + c * PW + PW / 2, PY + r * PH + PH / 2


def num(v: int) -> str:
    return f"{v / 100:.2f}".replace(".", ",")


body: list[str] = []
body.append('<text class="title" x="20" y="38">MaxSim: mỗi token của câu hỏi tìm vùng khớp nhất trên trang</text>')
body.append(svgkit.legend(20, 70, [("orange", "token của câu hỏi"), ("blue", "patch khớp nhất với token")]))

# Trang tài liệu
body.append(f'<rect class="tok" x="{PX}" y="{PY}" width="{COLS * PW}" height="{ROWS * PH}" rx="4"/>')
txt = []
txt.append(('Báo cáo kinh doanh 2025', 0, 0, "start", "font-weight:600"))
txt.append(("Miền Nam", 5, 0, "middle", ""))
for k, qn in enumerate(["Q1", "Q2", "Q3", "Q4"]):
    txt.append((qn, 2 + k, 2, "middle", "font-weight:600"))
values = {3: ("Chi phí", [8.1, 8.4, 9.0, 9.3]), 4: ("Lợi nhuận", [2.2, 2.5, 3.1, 3.0]),
          5: ("Doanh thu", [10.3, 10.9, 12.1, 12.3])}
for r, (label, vals) in values.items():
    txt.append((label, 0, r, "start", ""))
    for k, v in enumerate(vals):
        txt.append((f"{v:.1f}".replace(".", ","), 2 + k, r, "middle", "fill:var(--muted)"))
# Patch khớp nhất (vẽ dưới chữ và lưới)
for _, (c, r), _ in TOKENS:
    body.append(f'<rect class="blue" x="{PX + c * PW + 2}" y="{PY + r * PH + 2}" width="{PW - 4}" height="{PH - 4}" '
                'rx="4" />')
for t, c, r, anchor, style in txt:
    x = PX + c * PW + (5 if anchor == "start" else PW / 2)
    body.append(f'<text class="sub" x="{x}" y="{PY + r * PH + PH / 2 + 4}" '
                f'style="text-anchor:{anchor};fill:var(--fg);{style}">{t}</text>')
# Đường kẻ bảng và vài dòng chữ mờ phía dưới (đoạn văn, chân trang)
body.append(f'<line class="lmuted" x1="{PX + 6}" y1="{PY + 3 * PH}" x2="{RIGHT - 6}" y2="{PY + 3 * PH}"/>')
for r in (6, 7):
    for k, wfrac in enumerate([0.92, 0.7]):
        y = PY + r * PH + 13 + k * 13
        body.append(f'<line class="lmuted" x1="{PX + 8}" y1="{y}" x2="{PX + 8 + (COLS * PW - 16) * wfrac}" y2="{y}" '
                    'style="stroke-width:5;opacity:.7"/>')

# Lưới patch
for c in range(1, COLS):
    body.append(f'<line class="grid" x1="{PX + c * PW}" y1="{PY}" x2="{PX + c * PW}" y2="{PY + ROWS * PH}"/>')
for r in range(1, ROWS):
    body.append(f'<line class="grid" x1="{PX}" y1="{PY + r * PH}" x2="{RIGHT}" y2="{PY + r * PH}"/>')
body.append(f'<text class="sub mid" x="{PX + COLS * PW / 2}" y="{PY + ROWS * PH + 20}">'
            f'Lưới thu nhỏ: {COLS} × {ROWS} = {N_PATCH} patch (ColPali thật: ~1.030 patch/trang)</text>')

# Token và đường nối tới patch khớp nhất
TX, TW, TH = 20, 130, 34
ty = {"miền Nam": 132, "quý 3": 222, "doanh thu": PY + 5 * PH + PH / 2 - TH / 2}
route_y = {"miền Nam": (PY - 30, 70), "quý 3": (PY - 14, 85)}  # (độ cao, khoảng cách cột dọc)
for tok, (c, r), s in TOKENS:
    y0 = ty[tok]
    body.append(svgkit.box(TX, y0, TW, TH, "orange", tok))
    sx, sy = TX + TW, y0 + TH / 2
    tx, tyy = pc(c, r)
    if tok in route_y:  # đi vòng phía trên trang để không đè lên chữ
        ry, off = route_y[tok]
        d = f"M{sx},{sy} L{sx + off},{sy} L{sx + off},{ry} L{tx},{ry} L{tx},{tyy - PH / 2 + 2}"
    else:
        d = f"M{sx},{sy} L{PX + c * PW - 2},{tyy}"
    lx, ly = sx + 6, sy - 7
    body.append(f'<path class="lorange" d="{d}" marker-end="url(#aorange)"/>')
    body.append(f'<text class="sub halo" x="{lx}" y="{ly}" style="fill:var(--fg)">'
                f'max {num(s)}</text>')

body.append(f'<text class="cap" x="20" y="{H - 52}"><tspan style="font-weight:600">MaxSim</tspan> = '
            + " + ".join(num(s) for _, _, s in TOKENS) + f' = {num(total)}: ba vùng ở ba chỗ khác nhau trên trang đều góp điểm.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 28}">Điểm khớp là giả định. Mỗi token so với mọi patch, '
            'nên chi phí gấp |q| × |d| lần một phép dot product.</text>')

# So sánh: single-vector gộp cả trang thành một vector
RX = RIGHT + 50
body.append(f'<text class="row" x="{RX}" y="{PY + 14}">Single-vector</text>')
body.append(f'<text class="sub" x="{RX}" y="{PY + 34}">gộp cả trang thành 1 vector</text>')
mini, mx, my = 8, RX, PY + 56
for r in range(ROWS):
    for c in range(COLS):
        hit = any((c, r) == p for _, p, _ in TOKENS)
        body.append(f'<rect class="{"blue" if hit else "tok"}" x="{mx + c * (mini + 3)}" y="{my + r * (mini + 3)}" '
                    f'width="{mini}" height="{mini}" rx="1.5" style="stroke-width:1"/>')
gy = my + ROWS * (mini + 3)
body.append(f'<line class="lfg" x1="{mx + 30}" y1="{gy + 6}" x2="{mx + 30}" y2="{gy + 40}" marker-end="url(#afg)"/>')
body.append(f'<text class="sub" x="{mx + 42}" y="{gy + 28}">lấy trung bình</text>')
body.append(svgkit.box(mx, gy + 48, 150, 34, "tok", "1 vector"))
body.append(f'<text class="cap2" x="{RX}" y="{gy + 108}">3 vùng khớp bị trộn với</text>')
body.append(f'<text class="cap2" x="{RX}" y="{gy + 126}">{N_PATCH - 3} vùng khác (thực tế ~1.027):</text>')
body.append(f'<text class="cap2" x="{RX}" y="{gy + 144}">tín hiệu bị pha loãng.</text>')

svgkit.write(__file__, "colpali-maxsim.svg", svgkit.svg(
    W, H, "Câu hỏi doanh thu quý 3 miền Nam: mỗi token khớp với một patch ở một góc khác nhau của trang "
    "và MaxSim cộng ba điểm khớp tốt nhất; single-vector lấy trung bình cả trang nên ba vùng này bị pha loãng",
    body))
