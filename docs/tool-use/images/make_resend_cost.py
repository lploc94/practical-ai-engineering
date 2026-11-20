"""Sinh resend-cost.svg: biểu đồ phần tool result trong input của từng request (§1.1).

Số liệu giả định lấy từ ví dụ ở §1.1: tác vụ 20 bước, mỗi tool result 2.000 token.
API stateless nên request thứ i mang theo i kết quả: i − 1 kết quả cũ gửi lại và
1 kết quả mới. Chỉ tính phần tool result, bỏ qua tools, system và tin nhắn user.
Tổng = 2.000 × (1 + 2 + … + 20) = 420.000 token; tool chỉ trả về 40.000 token.

Chạy: python3 make_resend_cost.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

N, R = 20, 2000
new = [R] * N                                   # kết quả mới của bước i
repeated = [(i - 1) * R for i in range(1, N + 1)]   # kết quả cũ gửi lại
inputs = [r + n for r, n in zip(repeated, new)]
total, total_new = sum(inputs), sum(new)
assert total == 420_000 and total_new == 40_000
assert inputs[-1] == 40_000
assert total / total_new == 10.5   # §1.1 ghi "gấp 10,5 lần"

W, H = 820, 380
PX0, PX1, PY0, PY1 = 70, 800, 96, 306   # vùng vẽ
YMAX = 40_000
slot = (PX1 - PX0) / N
bw = slot - 8


def y(v: float) -> float:
    return PY1 - v / YMAX * (PY1 - PY0)


def fmt(v: int) -> str:
    return f"{v:,}".replace(",", ".")


body: list[str] = []
body.append('<text class="title" x="20" y="38">Tool result trong input của mỗi request, tác vụ 20 bước</text>')
body.append(f'<text class="cap2" x="20" y="62">Tổng {fmt(total)} token tool result được gửi đi, trong khi tool chỉ '
            f'trả về {fmt(total_new)} token ({str(round(total / total_new, 1)).replace(".", ",")} lần).</text>')
body.append(svgkit.legend(PX0, 86, [("fblue", "kết quả cũ gửi lại"), ("forange", "kết quả mới của bước này")]))

for v in range(0, YMAX + 1, 10_000):
    body.append(f'<line class="grid" x1="{PX0}" y1="{y(v)}" x2="{PX1}" y2="{y(v)}"/>')
    body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(v) + 4}">{f"{v // 1000}K" if v else "0"}</text>')

for n in range(N):
    x = PX0 + n * slot + 4
    r, w = repeated[n], new[n]
    # Phần gửi lại ở đáy, phần mới ở trên; khe 2px giữa hai đoạn
    if r:
        body.append(f'<rect class="fblue" x="{x:.1f}" y="{y(r):.1f}" width="{bw:.1f}" height="{y(0) - y(r):.1f}"/>')
    top = y(r + w)
    h = y(r) - top - (2 if r else 0)
    body.append(f'<rect class="forange" x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3"/>')
    if n + 1 in (1, 5, 10, 15, 20):
        body.append(f'<text class="sub mid" x="{x + bw / 2:.1f}" y="{PY1 + 18}">{n + 1}</text>')

body.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}"/>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{PY1 + 36}">Request thứ i</text>')
# Nhãn trực tiếp cho hai đầu
first_x = PX0 + 4
last_x = PX0 + (N - 1) * slot + 4
body.append(f'<text class="sub halo" x="{first_x}" y="{y(inputs[0]) - 8}">{fmt(inputs[0])}</text>')
body.append(f'<text class="sub end halo" x="{last_x + bw}" y="{y(inputs[-1]) - 8}">{fmt(inputs[-1])}</text>')
body.append(f'<text class="cap2" x="20" y="{H - 10}">Số liệu giả định: mỗi tool result 2.000 token. '
            'Chỉ tính phần tool result, chưa gồm tools, system và tin nhắn user.</text>')

svgkit.write(__file__, "resend-cost.svg", svgkit.svg(
    W, H, f"Biểu đồ cột 20 request: phần tool result tăng từ {fmt(inputs[0])} lên {fmt(inputs[-1])} token; "
    f"tổng {fmt(total)} token được gửi đi trong khi tool chỉ trả về {fmt(total_new)} token", body))
