"""Sinh similarity-metrics.svg: hình tĩnh minh họa cosine, dot product và L2 xếp hạng khác nhau (§1.3).

Ví dụ 2 chiều, số liệu giả định để thấy cơ chế:
- query q: góc 45°, độ dài 3
- A (đoạn ngắn) và B (bài dài) cùng hướng 30°, độ dài 2 và 5
- C (chủ đề lệch hơn): góc 70°, độ dài 3

Kết quả (script assert): cosine xếp A = B > C, dot product xếp B > C > A, L2 xếp A gần nhất, rồi C, B xa nhất (khoảng cách A < C < B).

Chạy: python3 make_similarity_metrics.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

VEC = {"q": (45, 3.0), "A": (30, 2.0), "B": (30, 5.0), "C": (70, 3.0)}


def xy(name: str) -> tuple[float, float]:
    ang, r = VEC[name]
    return r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))


q = xy("q")
docs = ["A", "B", "C"]
cos = {d: (q[0] * xy(d)[0] + q[1] * xy(d)[1]) / (VEC["q"][1] * VEC[d][1]) for d in docs}
dot = {d: q[0] * xy(d)[0] + q[1] * xy(d)[1] for d in docs}
l2 = {d: math.dist(q, xy(d)) for d in docs}

assert abs(cos["A"] - cos["B"]) < 1e-9 and cos["A"] > cos["C"]
assert dot["B"] > dot["C"] > dot["A"]
assert l2["A"] < l2["C"] < l2["B"]

W, H = 900, 420
OX, OY, S = 70, 370, 62  # gốc toạ độ và số px cho 1 đơn vị


def px(p: tuple[float, float]) -> tuple[float, float]:
    return OX + p[0] * S, OY - p[1] * S


body: list[str] = []
body.append('<text class="title" x="20" y="38">Ba phép đo, ba thứ hạng khác nhau</text>')
body.append('<text class="cap2" x="20" y="62">Ví dụ 2 chiều, số liệu giả định. A và B cùng hướng; '
            'B dài hơn vì là bài dài. Nét đứt: khoảng cách L2.</text>')

# Trục
body.append(f'<line class="axis" x1="{OX}" y1="{OY}" x2="{OX + 5.4 * S}" y2="{OY}"/>')
body.append(f'<line class="axis" x1="{OX}" y1="{OY}" x2="{OX}" y2="{OY - 4.6 * S}"/>')
body.append(f'<text class="sub" x="{OX - 12}" y="{OY + 18}">0</text>')

# Góc 15° giữa q và hướng của A/B: chỉ thứ này quyết định cosine
r_arc = 1.25 * S
a1, a2 = math.radians(30), math.radians(45)
p1 = (OX + r_arc * math.cos(a1), OY - r_arc * math.sin(a1))
p2 = (OX + r_arc * math.cos(a2), OY - r_arc * math.sin(a2))
body.append(f'<path class="lfg" d="M{p1[0]:.1f},{p1[1]:.1f} A{r_arc},{r_arc} 0 0,0 {p2[0]:.1f},{p2[1]:.1f}"/>')
body.append(f'<text class="sub halo" x="{p1[0] + 4:.1f}" y="{p1[1] + 16:.1f}">15°</text>')

# Khoảng cách L2 từ đầu q đến đầu mỗi vector tài liệu (nét đứt)
for d in docs:
    a, b = px(q), px(xy(d))
    body.append(f'<line class="lmuted dash" x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')

# Vector tài liệu (xanh), vẽ B trước để A nằm trên
for d in ["B", "C", "A"]:
    e = px(xy(d))
    body.append(f'<line class="lblue" x1="{OX}" y1="{OY}" x2="{e[0]:.1f}" y2="{e[1]:.1f}" '
                f'marker-end="url(#ablue)" style="stroke-width:2.5"/>')
labels = {"A": ("A · đoạn ngắn", 8, 18), "B": ("B · bài dài", 10, 6),
          "C": ("C · chủ đề lệch hơn", 10, -6)}
for d, (t, dx, dy) in labels.items():
    e = px(xy(d))
    body.append(f'<text class="cap halo" x="{e[0] + dx:.1f}" y="{e[1] + dy:.1f}">{t}</text>')
# Query (cam)
e = px(q)
body.append(f'<line class="lorange" x1="{OX}" y1="{OY}" x2="{e[0]:.1f}" y2="{e[1]:.1f}" '
            f'marker-end="url(#aorange)" style="stroke-width:3"/>')
body.append(f'<text class="cap halo" x="{e[0] - 28:.1f}" y="{e[1] - 10:.1f}" style="font-weight:600">q · query</text>')

# Bảng thứ hạng bên phải
RX = 510


def rank(scores: dict[str, float], reverse: bool, fmt: str) -> str:
    order = sorted(docs, key=lambda d: scores[d], reverse=reverse)
    parts = []
    for i, d in enumerate(order):
        sep = ""
        if i:
            sep = " = " if abs(scores[d] - scores[order[i - 1]]) < 1e-9 else (" &gt; " if reverse else " &lt; ")
        parts.append(f"{sep}{d} ({format(scores[d], fmt)})")
    return "".join(parts)


rows = [
    ("Cosine — chỉ nhìn góc", rank(cos, True, ".2f"), "A và B ngang nhau: độ dài bị bỏ qua"),
    ("Dot product — góc × độ dài", rank(dot, True, ".1f"), "B dài nên vượt lên; A ngắn tụt xuống cuối"),
    ("L2 — khoảng cách hai đầu mút", rank(l2, False, ".2f"), "Nhỏ hơn là gần hơn; B xa nhất vì quá dài"),
]
y0 = 118
for i, (head, order, note) in enumerate(rows):
    y = y0 + i * 88
    body.append(f'<text class="row" x="{RX}" y="{y}">{head}</text>')
    body.append(f'<text class="lbl" x="{RX}" y="{y + 26}" style="text-anchor:start">{order}</text>')
    body.append(f'<text class="cap2" x="{RX}" y="{y + 48}">{note}</text>')
body.append(f'<text class="cap2" x="{RX}" y="{y0 + 3 * 88 - 4}">Khi mọi vector được chuẩn hoá về độ dài 1,</text>')
body.append(f'<text class="cap2" x="{RX}" y="{y0 + 3 * 88 + 14}">ba phép đo cho cùng một thứ hạng.</text>')

svgkit.write(__file__, "similarity-metrics.svg", svgkit.svg(
    W, H, "Query q và ba tài liệu: A và B cùng hướng nhưng B dài hơn. Cosine xếp A bằng B, "
    "dot product xếp B cao nhất, L2 xếp A gần nhất và B xa nhất", body))
