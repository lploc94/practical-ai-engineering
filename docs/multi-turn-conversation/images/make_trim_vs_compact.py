"""Sinh trim-vs-compact.svg: so cửa sổ trượt với compact theo ngưỡng (§1.3, §3.8).

Mô hình đơn giản hoá, số liệu giả định: tools + system = 8.000 token, mỗi lượt thêm
1.000 token lịch sử, 40 request, giới hạn lịch sử 24.000 token cho cả hai cách.
- Cửa sổ trượt: giữ 24 lượt gần nhất; khi bắt đầu bỏ lượt cũ, message đầu tiên
  sau system đổi mỗi request nên chỉ tools + system còn khớp cache.
- Compact theo ngưỡng: append-only đến 24.000 token rồi tóm tắt còn 3.000 token;
  request ngay sau compact chỉ khớp tools + system, các request sau lại khớp gần hết.

Chạy: python3 make_trim_vs_compact.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

S, TURN, N, LIMIT, SUMMARY = 8000, 1000, 40, 24_000, 3000


def sliding() -> list[tuple[int, int]]:
    """(đọc từ cache, phải prefill) cho từng request."""
    out = []
    for n in range(1, N + 1):
        hist = min(n * TURN, LIMIT)
        if n == 1:
            out.append((0, S + hist))
        elif n * TURN <= LIMIT:           # chưa bỏ lượt nào: append-only
            out.append((S + hist - TURN, TURN))
        else:                             # đầu lịch sử đổi mỗi request
            out.append((S, hist))
    return out


def compact() -> list[tuple[int, int]]:
    out, hist = [], 0
    for n in range(1, N + 1):
        if n == 1:
            hist = TURN
            out.append((0, S + hist))
        elif hist + TURN > LIMIT:         # chạm ngưỡng: tóm tắt, prefix mới
            hist = SUMMARY + TURN
            out.append((S, hist))
        else:
            out.append((S + hist, TURN))
            hist += TURN
    return out


panels = [("Cửa sổ trượt: bỏ lượt cũ nhất ở mỗi request", sliding()),
          ("Compact theo ngưỡng: append-only, chạm 24K thì tóm tắt còn 3K", compact())]

W, H = 820, 512
PX0, PX1 = 70, 800
PH, GAP, TOP = 130, 62, 142
YMAX = 32_000
slot = (PX1 - PX0) / N
bw = slot - 4


def fmt(v: int) -> str:
    return f"{v:,}".replace(",", ".")


body: list[str] = []
body.append('<text class="title" x="20" y="38">Dọn context mỗi lượt hay theo ngưỡng</text>')
totals = [sum(u for _, u in data) for _, data in panels]
body.append(f'<text class="cap2" x="20" y="62">40 request, cùng giới hạn lịch sử 24K token. Tổng token phải prefill: '
            f'cửa sổ trượt {fmt(totals[0])}, compact theo ngưỡng {fmt(totals[1])}.</text>')
body.append(svgkit.legend(PX0, 92, [("fblue", "đọc từ cache"), ("hatch", "phải prefill")]))

for p, (title, data) in enumerate(panels):
    py0 = TOP + p * (PH + GAP)
    py1 = py0 + PH

    def y(v: float) -> float:
        return py1 - v / YMAX * PH

    body.append(f'<text class="row" x="{PX0}" y="{py0 - 12}">{title}</text>')
    for v in (0, 16_000, 32_000):
        body.append(f'<line class="grid" x1="{PX0}" y1="{y(v)}" x2="{PX1}" y2="{y(v)}"/>')
        body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(v) + 4}">{f"{v // 1000}K" if v else "0"}</text>')
    for n, (c, u) in enumerate(data):
        x = PX0 + n * slot + 2
        if c:
            body.append(f'<rect class="fblue" x="{x:.1f}" y="{y(c):.1f}" width="{bw:.1f}" height="{y(0) - y(c):.1f}"/>')
        top = y(c + u)
        body.append(f'<rect class="hatch" style="stroke-width:1" x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                    f'height="{y(c) - top - (2 if c else 0):.1f}" rx="2"/>')
    body.append(f'<line class="axis" x1="{PX0}" y1="{py1}" x2="{PX1}" y2="{py1}"/>')
    for n in (1, 10, 20, 30, 40):
        body.append(f'<text class="sub mid" x="{PX0 + (n - 1) * slot + 2 + bw / 2:.1f}" y="{py1 + 16}">{n}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Số liệu giả định: tools + system 8K token, mỗi lượt thêm 1K token. '
            'Request thứ n trên trục ngang.</text>')

svgkit.write(__file__, "trim-vs-compact.svg", svgkit.svg(
    W, H, f"So sánh 40 request: cửa sổ trượt phải prefill {fmt(totals[0])} token vì đầu lịch sử đổi mỗi lượt; "
    f"compact theo ngưỡng chỉ phải prefill {fmt(totals[1])} token", body))
