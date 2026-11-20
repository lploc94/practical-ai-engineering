"""Sinh pass-at-k.svg: hình tĩnh, pass@k và pass^k theo k cho một agent đúng 90% mỗi lần (§2.4).

Số liệu GIẢ ĐỊNH lấy từ ví dụ ở §2.4: agent thành công 90% mỗi lần, các lần thử độc lập.
pass@k = 1 − 0,1^k (ít nhất một lần đúng trong k lần), pass^k = 0,9^k (cả k lần đều đúng).
Với k = 8: pass@8 gần 100%, pass^8 ≈ 43%. Các con số in trong hình được assert.

Chạy: python3 make_pass_at_k.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

P, K = 0.9, 8
ks = list(range(1, K + 1))
at = [1 - (1 - P) ** k for k in ks]
hat = [P ** k for k in ks]
assert round(hat[-1] * 100) == 43 and at[-1] > 0.9999


def pct(v: float) -> str:
    return f"{round(v * 100)}%"


W, H = 820, 400
X0, X1, Y0, Y1 = 80, 670, 100, 320     # vùng vẽ


def x(k: int) -> float:
    return X0 + (k - 1) / (K - 1) * (X1 - X0)


def y(v: float) -> float:
    return Y1 - v * (Y1 - Y0)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Agent đúng 90% mỗi lần: pass@k và pass^k đi ngược chiều</text>')
body.append('<text class="cap2" x="20" y="62">Giả định các lần thử độc lập. pass@k: ít nhất một lần đúng '
            'trong k lần. pass^k: cả k lần đều đúng.</text>')
body.append(svgkit.legend(X0, 86, [("fblue", "pass@k"), ("forange", "pass^k")]))

for v in (0, 0.25, 0.5, 0.75, 1.0):
    body.append(f'<line class="grid" x1="{X0}" y1="{y(v)}" x2="{X1}" y2="{y(v)}"/>')
    body.append(f'<text class="sub end" x="{X0 - 8}" y="{y(v) + 4}">{pct(v)}</text>')
body.append(f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>')
for k in ks:
    body.append(f'<text class="sub mid" x="{x(k)}" y="{Y1 + 18}">{k}</text>')
body.append(f'<text class="sub mid" x="{(X0 + X1) / 2}" y="{Y1 + 38}">k (số lần thử)</text>')

for cls, vals in (("blue", at), ("orange", hat)):
    pts = " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in zip(ks, vals))
    body.append(f'<polyline class="l{cls}" points="{pts}" style="stroke-width:2.5"/>')
    for k, v in zip(ks, vals):
        body.append(f'<circle class="f{cls}" cx="{x(k):.1f}" cy="{y(v):.1f}" r="4"/>')

# Nhãn trực tiếp: điểm xuất phát chung và hai điểm cuối
body.append(f'<text class="cap halo" x="{x(1) + 6}" y="{y(P) + 48}">cả hai 90% ở k = 1</text>')
body.append(f'<text class="cap halo" x="{x(K) + 12}" y="{y(at[-1]) + 5}">pass@8 ≈ 100%</text>')
body.append(f'<text class="cap halo" x="{x(K) + 12}" y="{y(hat[-1]) + 5}">pass^8 ≈ {pct(hat[-1])}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Agent viết code cho người tự review: pass@k hợp lý. '
            'Agent phục vụ khách hàng: pass^k, vì mỗi khách chỉ có một lần.</text>')

svgkit.write(__file__, "pass-at-k.svg", svgkit.svg(
    W, H, "Agent đúng 90% mỗi lần, các lần thử độc lập: khi k tăng từ 1 đến 8, pass@k tăng từ 90% lên gần 100% "
    "còn pass^k giảm từ 90% xuống khoảng 43% (số liệu giả định)", body))
