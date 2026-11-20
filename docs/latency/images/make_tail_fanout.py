"""Sinh tail-fanout.svg: biểu đồ tĩnh, tỉ lệ request chạm phần đuôi theo số server phải chờ (§1.4).

Mô hình của Dean và Barroso, "The Tail at Scale" [20]: một server thường trả lời
trong 10 ms nhưng p99 là 1 giây, tức 1% lần trả lời chậm hơn 1 giây. Request phải
chờ n server như thế thì tỉ lệ request chậm hơn 1 giây là 1 − 0,99^n (giả định các
lần chậm độc lập). Bài báo nêu hai điểm, được ghim bằng assert bên dưới:
n = 1 → 1 trên 100 request; n = 100 → 63%.

Chạy: python3 make_tail_fanout.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

P_SLOW = 0.01
NMAX = 100


def slow(n: int) -> float:
    return 1 - (1 - P_SLOW) ** n


assert round(slow(1) * 100) == 1                 # §1.4: 1 trên 100 request
assert round(slow(100) * 100) == 63              # §1.4: 63% request chậm hơn 1 giây

W, H = 820, 380
PX0, PX1, PY0, PY1 = 80, 770, 96, 296


def x(n: float) -> float:
    return PX0 + n / NMAX * (PX1 - PX0)


def y(p: float) -> float:
    return PY1 - p * (PY1 - PY0)


def pct(p: float) -> str:
    return f"{p * 100:.0f}%"


body: list[str] = []
body.append('<text class="title" x="20" y="38">Chờ càng nhiều server, càng dễ gặp phần đuôi</text>')
body.append('<text class="cap2" x="20" y="62">Mỗi server chỉ chậm hơn 1 giây ở 1% số lần (p99 = 1 s). '
            'Tỉ lệ request chậm hơn 1 giây = 1 − 0,99^n.</text>')

for p in (0, 0.25, 0.5, 0.75, 1.0):
    body.append(f'<line class="grid" x1="{PX0}" y1="{y(p):.1f}" x2="{PX1}" y2="{y(p):.1f}"/>')
    body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(p) + 4:.1f}">{pct(p)}</text>')
for n in (1, 20, 40, 60, 80, 100):
    body.append(f'<text class="sub mid" x="{x(n):.1f}" y="{PY1 + 18}">{n}</text>')
body.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}"/>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{PY1 + 38}">'
            'n: số server một request phải chờ</text>')

pts = " ".join(f"{x(n):.1f},{y(slow(n)):.1f}" for n in range(1, NMAX + 1))
body.append(f'<polyline class="lblue" style="stroke-width:2.5" points="{pts}"/>')

# Nhãn trực tiếp cho hai điểm bài báo nêu
# (nhãn, dx, dy, căn lề): nhãn n = 1 đặt cao hơn để không đè lên đường cong
labels = {1: ("n = 1: 1% request chậm", 8, -44, ""),
          100: (f"n = 100: {pct(slow(100))} request chậm", -10, -12, " end")}
for n, (text, dx, dy, anchor) in labels.items():
    px, py = x(n), y(slow(n))
    body.append(f'<circle class="fblue" cx="{px:.1f}" cy="{py:.1f}" r="5"/>')
    body.append(f'<text class="cap halo{anchor}" x="{px + dx:.1f}" y="{py + dy:.1f}">{text}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Mô hình và hai điểm n = 1, n = 100 theo "The Tail at Scale" '
            '(Dean, Barroso, 2013); giả định các lần chậm độc lập.</text>')

svgkit.write(__file__, "tail-fanout.svg", svgkit.svg(
    W, H, "Khi mỗi server chỉ chậm hơn 1 giây ở 1% số lần, request chờ 1 server thì 1% request chậm, "
    "chờ 100 server thì 63% request chậm hơn 1 giây", body))
