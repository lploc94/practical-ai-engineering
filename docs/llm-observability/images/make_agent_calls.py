"""Sinh agent-calls.svg: phân phối gen_ai.invoke_agent.inference_calls (§3.2).

Số liệu giả định: 1.000 agent run. Đa số run gọi model 2–10 lượt; 20 run (2%)
chạy vòng tới giới hạn cứng 60 lượt. Khi đó p50 = 4 và p99 = 60, đúng ví dụ trong
bài. Bỏ 20 run chạy vòng thì p50 vẫn là 4, p99 về 10: chỉ đuôi phân phối lộ ra vòng lặp.
Percentile tính theo nearest-rank.
Hình tĩnh vì ý chính là hình dạng phân phối, không có diễn biến theo thời gian.

Chạy: python3 make_agent_calls.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

LIMIT = 60
COUNTS = {2: 120, 3: 220, 4: 250, 5: 170, 6: 100, 7: 60, 8: 35, 9: 15, 10: 10, LIMIT: 20}
N = sum(COUNTS.values())
assert N == 1000


def percentile(q: float, counts: dict[int, int] = COUNTS) -> int:
    rank = math.ceil(q * sum(counts.values()))
    acc = 0
    for v in sorted(counts):
        acc += counts[v]
        if acc >= rank:
            return v
    raise AssertionError


P50, P99 = percentile(0.50), percentile(0.99)
assert (P50, P99) == (4, 60)
NO_LOOP = {v: c for v, c in COUNTS.items() if v != LIMIT}
assert percentile(0.50, NO_LOOP) == P50       # bỏ run chạy vòng, p50 vẫn là 4
assert percentile(0.99, NO_LOOP) == 10        # và p99 về 10
LOOP_PCT = COUNTS[LIMIT] / N * 100
assert LOOP_PCT == 2

W, H = 820, 380
PX0, PX1, PY0, PY1 = 70, 790, 100, 300
XMAX, YMAX = 62, 250
slot = (PX1 - PX0) / XMAX
bw = slot - 2


def x(v: float) -> float:
    return PX0 + (v - 0.5) * slot


def y(v: float) -> float:
    return PY1 - v / YMAX * (PY1 - PY0)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Số lượt gọi model mỗi agent run: p50 không thấy vòng lặp, p99 thấy</text>')
body.append(f'<text class="cap2" x="20" y="62">1.000 run giả định; {COUNTS[LIMIT]} run ({LOOP_PCT:.0f}%) chạy vòng tới '
            f'giới hạn cứng {LIMIT} lượt.</text>')

for v in range(0, YMAX + 1, 50):
    body.append(f'<line class="grid" x1="{PX0}" y1="{y(v)}" x2="{PX1}" y2="{y(v)}"/>')
    body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(v) + 4}">{v}</text>')
body.append(f'<text class="sub" x="{PX0 - 8}" y="{PY0 - 14}" style="text-anchor:start">số run</text>')

for v, c in COUNTS.items():
    cls = "hatch" if v == LIMIT else "fblue"
    body.append(f'<rect class="{cls}" x="{x(v) + 1:.1f}" y="{y(c):.1f}" width="{bw:.1f}" height="{y(0) - y(c):.1f}"/>')

body.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}"/>')
for v in (1, 10, 20, 30, 40, 50, 60):
    body.append(f'<text class="sub mid" x="{x(v) + slot / 2:.1f}" y="{PY1 + 18}">{v}</text>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{PY1 + 38}">gen_ai.invoke_agent.inference_calls '
            '(số lượt gọi model trong một run)</text>')

# Đường percentile
for v, name, anchor, dx in ((P50, f"p50 = {P50}", "", 8), (P99, f"p99 = {P99}", "end", -8)):
    xc = x(v) + slot / 2
    body.append(f'<line class="lfg dash" x1="{xc:.1f}" y1="{PY0 - 8}" x2="{xc:.1f}" y2="{PY1}"/>')
    body.append(f'<text class="cap halo {anchor}" x="{xc + dx:.1f}" y="{PY0 + 4}" style="font-weight:600">{name}</text>')
xl = x(LIMIT) + slot / 2
body.append(f'<text class="sub end halo" x="{xl - 8:.1f}" y="{y(COUNTS[LIMIT]) - 26}">{COUNTS[LIMIT]} run chạm giới hạn</text>')
body.append(f'<text class="sub end halo" x="{xl - 8:.1f}" y="{y(COUNTS[LIMIT]) - 10}">{LIMIT} lượt: đang chạy vòng</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Bỏ 2% run chạy vòng thì p50 vẫn là 4, p99 về 10; '
            'alert trên p99 và tỉ lệ run chạm giới hạn.</text>')

svgkit.write(__file__, "agent-calls.svg", svgkit.svg(
    W, H, f"Histogram 1.000 agent run giả định: đa số gọi model 2 đến 10 lượt, p50 = {P50}; {COUNTS[LIMIT]} run "
    f"({LOOP_PCT:.0f}%) chạy vòng tới giới hạn {LIMIT} lượt nên p99 = {P99}", body))
