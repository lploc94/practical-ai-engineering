"""Sinh hnsw-search.svg: SVG động minh họa cách HNSW tìm kiếm từ tầng thưa xuống tầng dày (§2.1).

Không gian vector được vẽ thu gọn thành một trục ngang: node càng gần dấu q thì càng gần query.
Đồ thị gồm 16 vector, 3 tầng; vị trí node và cạnh là giả định để minh họa cơ chế greedy search,
không phải dữ liệu thật. Script assert rằng mỗi bước nhảy thực sự là bước greedy (đến láng giềng
gần q nhất, dừng khi không còn láng giềng nào gần hơn).

Chạy: python3 make_hnsw_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

N = 16
Q = 10.6  # vị trí của query trên trục
LAYERS = {  # tầng: (các node, các cạnh)
    2: ([1, 8, 14], [(1, 8), (8, 14)]),
    1: ([1, 4, 8, 11, 14], [(1, 4), (4, 8), (8, 11), (11, 14)]),
    0: (list(range(N)), [(i, i + 1) for i in range(N - 1)] + [(2, 5), (7, 10), (11, 13)]),
}
K = 3


def neighbors(layer: int, n: int) -> list[int]:
    return [b if a == n else a for a, b in LAYERS[layer][1] if n in (a, b)]


def greedy(layer: int, start: int) -> list[int]:
    path = [start]
    while True:
        best = min(neighbors(layer, path[-1]), key=lambda m: abs(m - Q))
        if abs(best - Q) >= abs(path[-1] - Q):
            return path
        path.append(best)


path2 = greedy(2, 1)
path1 = greedy(1, path2[-1])
path0 = greedy(0, path1[-1])
assert path2 == [1, 8] and path1 == [8, 11] and path0 == [11]
topk = sorted(range(N), key=lambda m: abs(m - Q))[:K]
assert sorted(topk) == [10, 11, 12]

X0, STEP, R = 220, 40, 9
LY = {2: 120, 1: 205, 0: 290}
W, H = X0 + (N - 1) * STEP + 60, 400


def x(i: float) -> float:
    return X0 + i * STEP


a = svgkit.Anim(dur=20, fade_out=96)  # 70% → 96%: giữ trạng thái cuối khoảng 5 giây
body: list[str] = []
body.append('<text class="title" x="20" y="40">HNSW: tìm từ tầng thưa xuống tầng dày</text>')
body.append(svgkit.legend(W - 330, 40, [("orange", "đường đi của truy vấn"), ("blue", f"top-{K}")]))

names = {2: ("Tầng 2", "ít node, cạnh dài"), 1: ("Tầng 1", ""), 0: ("Tầng 0", "mọi vector")}
for lv, y in LY.items():
    body.append(f'<text class="row" x="20" y="{y + 5}">{names[lv][0]}</text>')
    if names[lv][1]:
        body.append(f'<text class="sub" x="20" y="{y + 23}">{names[lv][1]}</text>')

# Đường nối cùng một vector giữa các tầng
for lv in (2, 1):
    for n in LAYERS[lv][0]:
        body.append(f'<line class="lmuted dash" x1="{x(n)}" y1="{LY[lv] + R}" x2="{x(n)}" y2="{LY[lv - 1] - R}"/>')

# Vị trí query: đường dọc qua cả ba tầng
qx = x(Q)
body.append(f'<line class="lfg dash" x1="{qx}" y1="{LY[2] - 34}" x2="{qx}" y2="{LY[0] + 22}" style="opacity:.6"/>')
body.append(f'<text class="cap mid" x="{qx}" y="{LY[2] - 40}" style="font-weight:600">q</text>')

# Cạnh trong từng tầng: liền kề vẽ thẳng, cạnh xa vẽ cung phía dưới
for lv, (nodes, edges) in LAYERS.items():
    y = LY[lv]
    for p, r in edges:
        if lv == 0 and r - p > 1:
            body.append(f'<path class="lmuted" d="M{x(p)},{y + R} Q{(x(p) + x(r)) / 2},{y + 26} {x(r)},{y + R}"/>')
        else:
            body.append(f'<line class="lmuted" x1="{x(p) + R}" y1="{y}" x2="{x(r) - R}" y2="{y}"/>')
    for n in nodes:
        body.append(f'<circle class="tok" cx="{x(n)}" cy="{y}" r="{R}"/>')


def node(lv: int, n: int, cls: str, attrs: str) -> str:
    return f'<g {attrs}><circle class="{cls}" cx="{x(n)}" cy="{LY[lv]}" r="{R}"/></g>'


def hop(lv: int, p: int, r: int, attrs: str) -> str:
    y = LY[lv] - R - 2
    return (f'<g {attrs}><path class="lorange" d="M{x(p)},{y} Q{(x(p) + x(r)) / 2},{y - 34} {x(r) - 4},{y - 2}" '
            f'marker-end="url(#aorange)"/></g>')


def descend(lv: int, n: int, attrs: str) -> str:
    return (f'<g {attrs}><line class="lorange" x1="{x(n) + 4}" y1="{LY[lv] + R + 2}" x2="{x(n) + 4}" '
            f'y2="{LY[lv - 1] - R - 3}" marker-end="url(#aorange)"/></g>')


on = lambda t: a([(t, 0), (t + 1.5, 1)])  # noqa: E731  hiện ra ở mốc t (phần trăm chu kỳ)

# Mốc thời gian (phần trăm của chu kỳ 20 s, mỗi bước cách nhau ít nhất 1 giây)
T_ENTRY, T_HOP2, T_DOWN2, T_HOP1, T_DOWN1, T_LOOK, T_TOPK = 6, 14, 24, 32, 42, 52, 62
body.append(node(2, path2[0], "orange", on(T_ENTRY)))
body.append(f'<g {on(T_ENTRY)}><text class="sub mid halo" x="{x(path2[0])}" y="{LY[2] - 18}">điểm vào</text></g>')
body.append(hop(2, path2[0], path2[1], on(T_HOP2)))
body.append(node(2, path2[1], "orange", on(T_HOP2 + 3)))
body.append(descend(2, path2[1], on(T_DOWN2)))
body.append(node(1, path1[0], "orange", on(T_DOWN2 + 3)))
body.append(hop(1, path1[0], path1[1], on(T_HOP1)))
body.append(node(1, path1[1], "orange", on(T_HOP1 + 3)))
body.append(descend(1, path1[1], on(T_DOWN1)))
body.append(node(0, path0[0], "orange", on(T_DOWN1 + 3)))
# Tầng 0: xem các láng giềng quanh điểm vừa đến
look = [n for n in neighbors(0, path0[0])]
body.append(f'<g {a([(T_LOOK, 0), (T_LOOK + 1.5, 1), (T_TOPK, 1), (T_TOPK + 1.5, 0)], static=False)}>' + "".join(
    f'<circle class="lorange dash" cx="{x(n)}" cy="{LY[0]}" r="{R + 5}"/>' for n in look) + "</g>")
body.append(f'<g {on(T_TOPK)}>' + "".join(
    f'<circle class="blue" cx="{x(n)}" cy="{LY[0]}" r="{R + 3}"/>' for n in topk)
    + f'<text class="sub mid halo" x="{x(sum(topk) / K)}" y="{LY[0] + 40}">top-{K} gần q nhất</text></g>')

# Chú thích theo pha: mỗi lúc chỉ một dòng
CY = 352
caps = [
    (0, T_DOWN1, "Tầng trên thưa: mỗi bước nhảy đi xa, nhanh chóng đến vùng gần q.",
     "Nhảy tới láng giềng gần q nhất; khi không còn ai gần hơn thì xuống tầng dưới."),
    (T_DOWN1, T_TOPK, "Xuống tầng dưới, tiếp tục từ chính node vừa dừng.",
     "Tầng 0 có mọi vector nên xét kỹ các láng giềng quanh đó."),
    (T_TOPK, 100, f"Kết quả: top-{K} vector gần q nhất.",
     "Chỉ duyệt vài node thay vì so q với cả 16 vector; đổi lại có thể bỏ sót (approximate)."),
]
for i, (t0, t1, l1, l2) in enumerate(caps):
    pts = [(t0, 0), (t0 + 1.5, 1)] if t0 else [(0, 1)]
    if t1 < 100:
        pts += [(t1 - 1.5, 1), (t1, 0)]
    body.append(f'<g {a(pts, static=False)}><text class="cap" x="20" y="{CY}">{l1}</text>'
                f'<text class="cap2" x="20" y="{CY + 22}">{l2}</text></g>')
body.append(f'<g class="static-only"><text class="cap" x="20" y="{CY}">Bắt đầu ở tầng thưa, nhảy tới láng giềng gần q '
            'nhất, hết đường thì xuống tầng dưới.</text>'
            f'<text class="cap2" x="20" y="{CY + 22}">Tầng 0 chứa mọi vector: xét kỹ quanh đó để lấy top-{K}, '
            'thay vì so q với cả 16 vector.</text></g>')

svgkit.write(__file__, "hnsw-search.svg", svgkit.svg(
    W, H, "HNSW tìm kiếm: bắt đầu ở tầng trên thưa, nhảy về phía query, xuống dần từng tầng, "
    "đến tầng 0 thì xét kỹ láng giềng để lấy top-K, chỉ duyệt vài node thay vì toàn bộ", body, a))
