"""Sinh score-shapes.svg: SVG tĩnh, ba phân phối của một câu Score ba mức (§1.4, §1.5).

Ba phân phối lấy từ văn bản §1.4 và §1.5:
- (0, 1, 0): toàn bộ xác suất ở mức 1 → score 1.0, confidence 1.0.
- (0.5, 0, 0.5): chia hai đầu thang → score 1.0, confidence 0.
- (0, 0.5, 0.5): chia hai mức kề → score 1.5, confidence 0.25.
Score và confidence tính bằng công thức Jev trong §1.5; assert kiểm tra lại.

Chạy: python3 make_score_shapes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402


def score(p: list[float]) -> float:
    return sum(i * v for i, v in enumerate(p))


def confidence(p: list[float], m: int | None = None) -> float:
    n = len(p)
    if m is None:
        m = max(range(n), key=lambda i: p[i])
    spread = sum(v * abs(i - m) for i, v in enumerate(p))
    mad_unif = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / mad_unif)


panels = [
    ("Dồn vào mức 1", [0.0, 1.0, 0.0]),
    ("Chia hai đầu thang", [0.5, 0.0, 0.5]),
    ("Chia hai mức kề", [0.0, 0.5, 0.5]),
]
results = [(score(p), confidence(p)) for _, p in panels]
assert [round(s, 2) for s, _ in results] == [1.0, 1.0, 1.5]
assert [round(c, 2) for _, c in results] == [1.0, 0.0, 0.25]
# Panel 3 có hai mức cùng cao nhất; chọn m = 1 hay m = 2 đều cho cùng confidence
assert round(confidence([0.0, 0.5, 0.5], 1), 2) == round(confidence([0.0, 0.5, 0.5], 2), 2) == 0.25


def num(v: float) -> str:
    """0.25 → "0.25", 1.0 → "1.0", 0 → "0" (dấu chấm như trong JSON của văn bản)."""
    if v == 0:
        return "0"
    s = f"{v:.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


W, H = 820, 400
PW, PGAP, PX0 = 220, 36, 60          # rộng panel, khe, lề trái
PY0, PY1 = 162, 290                   # vùng vẽ cột: xác suất 1 ở PY0, 0 ở PY1
BW = 52

body: list[str] = []
body.append('<text class="title" x="20" y="38">Cùng score, khác hình dạng phân phối</text>')
body.append('<text class="cap2" x="20" y="62">Câu Score ba mức (0, 1, 2). Score là trung bình có trọng số của số '
            'thứ tự mức; confidence đo độ tập trung.</text>')
body.append(svgkit.legend(20, 92, [("fblue", "xác suất của mức"), ("ffg", "vị trí score")]))

for k, (title, p) in enumerate(panels):
    s, c = results[k]
    x0 = PX0 + k * (PW + PGAP)
    slot = PW / 3

    def cx(i: float) -> float:
        return x0 + slot * (i + 0.5)

    def y(v: float) -> float:
        return PY1 - v * (PY1 - PY0)

    body.append(f'<text class="row" x="{x0}" y="{PY0 - 34}">{title}</text>')
    for v in (0.5, 1.0):
        body.append(f'<line class="grid" x1="{x0}" y1="{y(v)}" x2="{x0 + PW}" y2="{y(v)}"/>')
    if k == 0:
        for v in (0, 0.5, 1.0):
            body.append(f'<text class="sub end" x="{x0 - 6}" y="{y(v) + 4}">{num(v)}</text>')
    for i, v in enumerate(p):
        if v:
            body.append(f'<rect class="fblue" x="{cx(i) - BW / 2}" y="{y(v)}" width="{BW}" '
                        f'height="{y(0) - y(v)}" rx="3"/>')
            body.append(f'<text class="sub mid halo" x="{cx(i)}" y="{y(v) - 6}">{num(v)}</text>')
        body.append(f'<text class="sub mid" x="{cx(i)}" y="{PY1 + 18}">mức {i}</text>')
    body.append(f'<line class="axis" x1="{x0}" y1="{PY1}" x2="{x0 + PW}" y2="{PY1}"/>')
    # Tam giác chỉ vị trí score trên trục mức
    sx = cx(s)
    body.append(f'<path class="ffg" d="M{sx},{PY1 + 26} l-7,12 h14 z"/>')
    body.append(f'<text class="cap" x="{x0}" y="{PY1 + 66}">score {num(s)} · confidence {num(c)}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 14}">Hai panel đầu cùng score 1.0: một bên chắc chắn, một bên '
            'phân vân giữa hai đầu thang.</text>')

svgkit.write(__file__, "score-shapes.svg", svgkit.svg(
    W, H, "Ba phân phối trên thang ba mức: dồn vào mức 1 cho score 1.0 và confidence 1.0; chia đôi hai đầu thang "
    "cũng cho score 1.0 nhưng confidence 0; chia đôi hai mức kề cho score 1.5 và confidence 0.25", body))
