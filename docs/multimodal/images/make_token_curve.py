"""Sinh token-curve.svg: SVG tĩnh, token của một ảnh 16:9 theo chiều rộng ảnh (§1.2, "So sánh trên cùng ảnh").

Claude: tính bằng hàm claude_image_tokens chép từ §1.2 (ước lượng theo công thức patch 28 px,
khớp tám con số trong bảng docs [4]); assert lại hai điểm 1920×1080 và 3840×2160 của bảng.
Gemini 3, 4: 1.120 token ở mức media_resolution mặc định, bất kể kích thước [9].
Ảnh 16:9: cao = round(rộng × 9 / 16). Không vẽ OpenAI vì cột OpenAI trong bài là phép tính
chưa đối chiếu với request thật.

Chạy: python3 make_token_curve.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402


def claude_image_tokens(w: int, h: int, high_res: bool) -> int:
    """Chép từ §1.2 của README."""
    max_edge, max_tokens = (2576, 4784) if high_res else (1568, 1568)

    def tokens(scale: float) -> int:
        return math.ceil(math.floor(w * scale) / 28) * math.ceil(math.floor(h * scale) / 28)

    scale = min(1.0, max_edge / max(w, h))
    if tokens(scale) > max_tokens:
        lo, hi = 0.0, scale
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tokens(mid) <= max_tokens else (lo, mid)
        scale = lo
    return tokens(scale)


GEMINI = 1120
XS = list(range(100, 4001, 50))


def hgt(w: int) -> int:
    return round(w * 9 / 16)


std = [claude_image_tokens(w, hgt(w), False) for w in XS]
hr = [claude_image_tokens(w, hgt(w), True) for w in XS]
# Điểm trong bảng docs ở §1.2
assert claude_image_tokens(1920, 1080, False) == 1560 and claude_image_tokens(1920, 1080, True) == 2691
assert claude_image_tokens(3840, 2160, False) == 1560 and claude_image_tokens(3840, 2160, True) == 4784
assert hgt(1920) == 1080 and hgt(3840) == 2160
assert max(std) == 1560 and max(hr) == 4784

W, H = 820, 420
PX0, PX1, PY0, PY1 = 70, 790, 110, 330
XMAX, YMAX = 4000, 5000


def x(v: float) -> float:
    return PX0 + v / XMAX * (PX1 - PX0)


def y(v: float) -> float:
    return PY1 - v / YMAX * (PY1 - PY0)


def fmt(v: int) -> str:
    return f"{v:,}".replace(",", ".")


def poly(vals: list[int], cls: str) -> str:
    pts = " ".join(f"{x(w):.1f},{y(t):.1f}" for w, t in zip(XS, vals))
    return f'<polyline class="{cls}" points="{pts}" style="stroke-width:2.5"/>'


body: list[str] = []
body.append('<text class="title" x="20" y="38">Token của một ảnh 16:9 theo chiều rộng ảnh gốc</text>')
body.append('<text class="cap2" x="20" y="62">Claude tăng theo diện tích rồi chạm trần; '
            'Gemini 3 tính cố định theo mức media_resolution.</text>')
body.append(svgkit.legend(PX0, 92, [("fblue", "Claude standard"), ("forange", "Claude high-res (4.7+)"),
                                    ("ffg", "Gemini 3, mặc định")]))

for v in range(0, YMAX + 1, 1000):
    body.append(f'<line class="grid" x1="{PX0}" y1="{y(v)}" x2="{PX1}" y2="{y(v)}"/>')
    body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(v) + 4}">{fmt(v)}</text>')
for v in range(0, XMAX + 1, 1000):
    body.append(f'<text class="sub mid" x="{x(v)}" y="{PY1 + 18}">{fmt(v)}</text>')
body.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}"/>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{PY1 + 38}">Chiều rộng ảnh gốc (px), ảnh 16:9</text>')

body.append(f'<line class="lfg dash" x1="{x(XS[0])}" y1="{y(GEMINI)}" x2="{x(XS[-1])}" y2="{y(GEMINI)}" '
            'style="stroke-width:2"/>')
# Hai tier trùng nhau khi chưa chạm trần: vẽ high-res trước, standard đè lên
body.append(poly(hr, "lorange"))
body.append(poly(std, "lblue"))

# Nhãn trực tiếp: hai trần và hai điểm của bảng docs
body.append(f'<text class="sub end halo" x="{PX1}" y="{y(4784) - 8}">trần 4.784</text>')
body.append(f'<text class="sub end halo" x="{PX1}" y="{y(1560) - 8}">trần 1.560 (ảnh 16:9)</text>')
body.append(f'<text class="sub end halo" x="{PX1}" y="{y(GEMINI) + 18}">1.120</text>')
for w, t, dot in ((1920, 2691, "forange"), (1920, 1560, "fblue")):
    body.append(f'<circle class="{dot}" cx="{x(w):.1f}" cy="{y(t):.1f}" r="4.5"/>')
body.append(f'<text class="sub halo" x="{x(1920) + 10:.1f}" y="{y(2691) + 4:.1f}">1920×1080: 2.691</text>')
body.append(f'<text class="sub halo" x="{x(1920) + 10:.1f}" y="{y(1560) - 8:.1f}">1920×1080: 1.560</text>')
assert all(a == b for w, a, b in zip(XS, std, hr) if w <= 1400)
body.append(f'<text class="sub halo" x="{x(450):.1f}" y="{y(2000):.1f}">hai tier trùng nhau</text>')
body.append(f'<text class="sub halo" x="{x(450):.1f}" y="{y(2000) + 16:.1f}">khi chưa chạm trần</text>')

body.append(f'<text class="cap2" x="20" y="{H - 14}">Claude tính bằng hàm ước lượng ở §1.2, khớp bảng ví dụ '
            'trong docs. Đối chiếu usage trên request thật.</text>')

svgkit.write(__file__, "token-curve.svg", svgkit.svg(
    W, H, "Token của ảnh 16:9 theo chiều rộng: Claude tăng theo diện tích rồi chạm trần 1.560 token ở tier standard "
    "và 4.784 ở tier high-res, ảnh 1920×1080 tốn 1.560 và 2.691; Gemini 3 mặc định luôn 1.120 token", body))
