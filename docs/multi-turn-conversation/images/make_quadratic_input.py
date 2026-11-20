"""Sinh quadratic-input.svg: biểu đồ input của từng request trong 30 lượt chat (§1.1).

Số liệu lấy từ ví dụ ở §1.4: tools + system = 8.000 token, mỗi lượt user 300 token,
assistant 700 token. Input của request thứ n = 8.300 + 1.000·(n−1).

Chạy: python3 make_quadratic_input.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

N = 30
inputs = [8300 + 1000 * (n - 1) for n in range(1, N + 1)]
repeated = [0] + inputs[:-1]          # phần đã có trong request trước
new = [i - r for i, r in zip(inputs, repeated)]
total, total_rep = sum(inputs), sum(repeated)
assert total == 684_000

W, H = 820, 380
PX0, PX1, PY0, PY1 = 70, 800, 96, 306   # vùng vẽ
YMAX = 40_000
slot = (PX1 - PX0) / N
bw = slot - 6


def y(v: float) -> float:
    return PY1 - v / YMAX * (PY1 - PY0)


def fmt(v: int) -> str:
    return f"{v:,}".replace(",", ".")


body: list[str] = []
body.append('<text class="title" x="20" y="38">Input của mỗi request trong 30 lượt chat</text>')
body.append(f'<text class="cap2" x="20" y="62">Tổng {fmt(total)} token; {fmt(total_rep)} ({total_rep / total:.0%}) '
            'là phần lịch sử gửi lại từ request trước.</text>')
body.append(svgkit.legend(PX0, 86, [("fblue", "lặp lại từ request trước"), ("forange", "mới trong request này")]))

for v in range(0, YMAX + 1, 10_000):
    body.append(f'<line class="grid" x1="{PX0}" y1="{y(v)}" x2="{PX1}" y2="{y(v)}"/>')
    body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(v) + 4}">{f"{v // 1000}K" if v else "0"}</text>')

for n in range(N):
    x = PX0 + n * slot + 3
    r, w = repeated[n], new[n]
    # Phần lặp lại ở đáy, phần mới ở trên; khe 2px giữa hai đoạn
    if r:
        body.append(f'<rect class="fblue" x="{x:.1f}" y="{y(r):.1f}" width="{bw:.1f}" height="{y(0) - y(r):.1f}"/>')
    top = y(r + w)
    h = y(r) - top - (2 if r else 0)
    body.append(f'<rect class="forange" x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3"/>')
    if n + 1 in (1, 10, 20, 30):
        body.append(f'<text class="sub mid" x="{x + bw / 2:.1f}" y="{PY1 + 18}">{n + 1}</text>')

body.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}"/>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{PY1 + 36}">Request thứ n</text>')
# Nhãn trực tiếp cho hai đầu
last_x = PX0 + (N - 1) * slot + 3 + bw / 2
body.append(f'<text class="sub end halo" x="{last_x + bw / 2}" y="{y(inputs[-1]) - 8}">{fmt(inputs[-1])}</text>')
body.append(f'<text class="sub halo" x="{PX0 + 3}" y="{y(inputs[0]) - 8}">{fmt(inputs[0])}</text>')
body.append(f'<text class="cap2" x="20" y="{H - 10}">Phần mới gần như không đổi (~1.000 token/lượt); '
            'phần lặp lại tăng tuyến tính nên tổng input tăng theo N².</text>')

svgkit.write(__file__, "quadratic-input.svg", svgkit.svg(
    W, H, f"Biểu đồ cột 30 request: input tăng từ 8.300 lên 37.300 token; tổng {fmt(total)} token, "
    f"trong đó {total_rep / total:.0%} là lịch sử gửi lại", body))
