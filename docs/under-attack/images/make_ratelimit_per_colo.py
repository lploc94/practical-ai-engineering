"""Sinh ratelimit-per-colo.svg: biểu đồ cột tĩnh minh hoạ mục A4 (counter đếm riêng từng data center).

Số liệu GIẢ ĐỊNH: rule rate limiting "100 request / 10 giây" đếm theo JA4 fingerprint
(characteristic chỉ có ở Enterprise với Advanced Rate Limiting và cần mua thêm Bot Management,
theo bảng Availability ở developers.cloudflare.com/waf/rate-limiting-rules/). Mọi máy của botnet
dùng chung một JA4.
Botnet trải khắp thế giới nên request rơi vào ba data center: 80, 70 và 90 request trong cùng
10 giây. Mỗi data center giữ counter riêng, nên không counter nào vượt 100 dù tổng là 240.
Không nơi nào đếm cột "tổng".

Không dùng rule đếm theo IP: traffic của một IP bình thường chỉ tới một data center, nên
counter riêng từng data center chủ yếu ảnh hưởng rule đếm theo characteristic nhiều client cùng có.

Chạy: python3 make_ratelimit_per_colo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

LIMIT = 100
COUNTS = {"Data center 1": 80, "Data center 2": 70, "Data center 3": 90}
TOTAL = sum(COUNTS.values())
assert TOTAL == 240
assert all(c <= LIMIT for c in COUNTS.values()), "không counter nào được vượt ngưỡng"
assert TOTAL > LIMIT

W, H = 760, 480
PX0, PY0, PH = 90, 160, 230          # vùng vẽ: trái, trên, cao
YMAX = 250
BW, STEP = 90, 150
BASE = PY0 + PH


def y(v: float) -> float:
    return BASE - v / YMAX * PH


body: list[str] = []
body.append('<text class="title" x="20" y="38">Rate limiting đếm riêng ở từng data center</text>')
body.append(f'<text class="cap2" x="20" y="62">Rule {LIMIT} request / 10 giây đếm theo JA4 '
            '(Enterprise, cần Advanced Rate Limiting và Bot Management).</text>')
body.append('<text class="cap2" x="20" y="82">Botnet chung một JA4, rải ra ba data center. Số liệu giả định.</text>')
body.append(svgkit.legend(20, 110, [("blue", "counter của một data center"), ("slot", "tổng (không nơi nào đếm)")]))

xs = [PX0 + 40 + i * STEP for i in range(len(COUNTS) + 1)]
PX1 = xs[-1] + BW + 30
assert PX1 <= W - 20

for v in range(0, YMAX + 1, 50):
    body.append(f'<line class="grid" x1="{PX0}" y1="{y(v)}" x2="{PX1}" y2="{y(v)}"/>')
    body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(v) + 4}">{v}</text>')
body.append(f'<line class="axis" x1="{PX0}" y1="{BASE}" x2="{PX1}" y2="{BASE}"/>')
body.append(f'<text class="sub" x="20" y="{PY0 - 14}">request / 10 giây</text>')

for x, (name, c) in zip(xs, COUNTS.items()):
    body.append(f'<rect class="blue" x="{x}" y="{y(c)}" width="{BW}" height="{BASE - y(c)}" rx="3"/>')
    body.append(f'<text class="cap mid" x="{x + BW / 2}" y="{BASE + 22}">{name}</text>')

xt = xs[-1]
body.append(f'<rect class="slot" x="{xt}" y="{y(TOTAL)}" width="{BW}" height="{BASE - y(TOTAL)}" rx="3"/>')
body.append(f'<text class="row mid halo" x="{xt + BW / 2}" y="{y(TOTAL) - 8}">{TOTAL}</text>')
body.append(f'<text class="cap mid" x="{xt + BW / 2}" y="{BASE + 22}">Tổng</text>')
body.append(f'<text class="sub mid" x="{xt + BW / 2}" y="{BASE + 40}">'
            + " + ".join(str(c) for c in COUNTS.values()) + "</text>")

# Ngưỡng
body.append(f'<line class="lfg dash" x1="{PX0}" y1="{y(LIMIT)}" x2="{PX1}" y2="{y(LIMIT)}"/>')
body.append(f'<text class="sub halo" x="{PX0 + 6}" y="{y(LIMIT) - 6}">ngưỡng {LIMIT}</text>')

body.append(f'<text class="cap" x="20" y="{H - 20}">Không counter nào vượt {LIMIT} nên không bị chặn, '
            f'dù tổng {TOTAL} request đã vượt ngưỡng.</text>')

svgkit.write(__file__, "ratelimit-per-colo.svg", svgkit.svg(
    W, H, f"Rule {LIMIT} request trong 10 giây đếm theo JA4 (Enterprise, cần Advanced Rate Limiting và "
    f"Bot Management); botnet dùng chung một JA4 gửi 80, 70 và 90 request vào ba data center: "
    f"không counter nào vượt {LIMIT} nên không bị chặn, dù tổng {TOTAL} request đã vượt ngưỡng (số liệu giả định)",
    body))
