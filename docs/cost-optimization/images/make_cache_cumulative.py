"""Sinh cache-cumulative.svg: chi phí cộng dồn của hội thoại hỗ trợ khách hàng ở §2.2.

Giả định lấy đúng ví dụ §2.2 (và scripts/t_example.py): Sonnet 5.5 (input 2 USD,
output 10 USD, đọc cache 0,10 USD mỗi triệu token, ghi cache TTL 5 phút 1,25 lần
input), system prompt và tool 6.000 token, mỗi lượt user 100 token, model trả lời
300 token. Prompt của lượt i = 6.000 + 100·i + 300·(i−1).
- Không cache: cả prompt tính giá input ở mọi lượt.
- Có cache: breakpoint ở cuối prompt; lượt i đọc cache toàn bộ prompt lượt i−1 và
  ghi phần mới (câu trả lời trước cộng câu hỏi mới).
Output giống nhau ở hai cách: 300 token × 10 USD / triệu mỗi lượt.

Chạy: python3 make_cache_cumulative.py
"""

import sys
from decimal import Decimal as D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

M = D(1_000_000)
P_IN, P_OUT, P_READ, WRITE = D(2), D(10), D("0.10"), D("1.25")
N = 40


def prompt_tokens(i: int) -> int:
    return 6000 + 100 * i + 300 * (i - 1)


unc, cached, out = [D(0)], [D(0)], [D(0)]
for i in range(1, N + 1):
    prompt, prev = prompt_tokens(i), (prompt_tokens(i - 1) if i > 1 else 0)
    o = 300 * P_OUT / M
    unc.append(unc[-1] + prompt * P_IN / M + o)
    cached.append(cached[-1] + (prev * P_READ + (prompt - prev) * P_IN * WRITE) / M + o)
    out.append(out[-1] + o)

# Khớp bảng §2.2 và scripts/t_example.py
assert (unc[8], cached[8], unc[40], cached[40]) == (D("0.1440"), D("0.05136"), D("1.2320"), D("0.22768"))
assert out[8] == D("0.024") and out[40] == D("0.12")
r8, r40 = unc[8] / cached[8], unc[40] / cached[40]
assert (round(r8, 1), round(r40, 1)) == (D("2.8"), D("5.4"))


def vn(v, nd: int) -> str:
    return f"{D(str(v)):.{nd}f}".replace(".", ",")


W, H = 820, 436
PX0, PX1, PY0, PY1 = 80, 640, 122, 346
YMAX = 1.3


def x(n: float) -> float:
    return PX0 + n / N * (PX1 - PX0)


def y(v) -> float:
    return PY1 - float(v) / YMAX * (PY1 - PY0)


def path(series: list[D]) -> str:
    return "M" + " L".join(f"{x(n):.1f},{y(v):.1f}" for n, v in enumerate(series))


RED = "stroke-width:3;stroke-dasharray:6 4"
BLUE = "stroke-width:3"
OUT = "stroke:var(--muted);stroke-width:2.5;stroke-dasharray:2 4"

body: list[str] = []
body.append('<text class="title" x="20" y="38">Chi phí cộng dồn của một hội thoại, có và không có prompt cache</text>')
body.append('<text class="cap2" x="20" y="62">Sonnet 5.5, prefix 6.000 token, mỗi lượt 100 token hỏi và 300 token trả lời '
            '(ví dụ §2.2).</text>')
lx = PX0
for cls, style, text in (("lred", RED, "không cache"), ("lblue", BLUE, "có cache"),
                         ("lmuted", OUT, "riêng phần output (như nhau ở hai cách)")):
    body.append(f'<line class="{cls}" x1="{lx}" y1="86" x2="{lx + 24}" y2="86" style="{style}"/>')
    body.append(f'<text class="sub" x="{lx + 30}" y="90">{text}</text>')
    lx += 30 + 7 * len(text) + 28

for v in (0, 0.25, 0.5, 0.75, 1.0, 1.25):
    body.append(f'<line class="grid" x1="{PX0}" y1="{y(v):.1f}" x2="{PX1}" y2="{y(v):.1f}"/>')
    body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(v) + 4:.1f}">{vn(v, 2)}</text>')
body.append(f'<text class="sub end" x="{PX0 - 8}" y="{PY0 - 8}">USD</text>')
body.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}"/>')
for n in (1, 8, 20, 30, 40):
    body.append(f'<text class="sub mid" x="{x(n):.1f}" y="{PY1 + 18}">{n}</text>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{PY1 + 36}">Số lượt</text>')

body.append(f'<path class="lmuted" d="{path(out)}" style="{OUT}"/>')
body.append(f'<path class="lred" d="{path(unc)}" style="{RED}"/>')
body.append(f'<path class="lblue" d="{path(cached)}" style="{BLUE}"/>')

# Mốc 8 lượt: đường dóng lên vùng trống phía trên, nhãn đặt ở đó
x8 = x(8)
body.append(f'<line class="axis dash" x1="{x8:.1f}" y1="{y(unc[8]) - 6:.1f}" x2="{x8:.1f}" y2="{PY0 + 52}"/>')
for v, cls in ((unc[8], "fred"), (cached[8], "fblue")):
    body.append(f'<circle class="{cls}" cx="{x8:.1f}" cy="{y(v):.1f}" r="4"/>')
body.append(f'<text class="cap halo" x="{x8 - 6:.1f}" y="{PY0 + 26}">8 lượt: {vn(unc[8], 4)} so với '
            f'{vn(cached[8], 4)} USD, rẻ hơn {vn(r8, 1)} lần</text>')
body.append(f'<text class="sub halo" x="{x8 - 6:.1f}" y="{PY0 + 44}">output {vn(out[8], 3)} USD là gần nửa '
            'chi phí có cache</text>')

# Mốc 40 lượt: nhãn bên phải, ngoặc giữa hai điểm
x40 = x(40)
for v, cls in ((unc[40], "fred"), (cached[40], "fblue")):
    body.append(f'<circle class="{cls}" cx="{x40:.1f}" cy="{y(v):.1f}" r="4"/>')
body.append(f'<text class="cap" x="{x40 + 12:.1f}" y="{y(unc[40]) + 5:.1f}">{vn(unc[40], 4)} USD</text>')
body.append(f'<text class="cap" x="{x40 + 12:.1f}" y="{y(cached[40]) - 4:.1f}">{vn(cached[40], 4)} USD</text>')
body.append(f'<text class="sub" x="{x40 + 12:.1f}" y="{y(out[40]) + 14:.1f}">output {vn(out[40], 2)}</text>')
bx = x40 + 24
yt, yb = y(unc[40]) + 16, y(cached[40]) - 24
body.append(f'<path class="lfg" d="M{bx - 6:.1f},{yt:.1f} h6 V{yb:.1f} h-6"/>')
body.append(f'<text class="cap" x="{bx + 8:.1f}" y="{(yt + yb) / 2 + 5:.1f}" style="font-weight:600">'
            f'{vn(r40, 1)} lần</text>')

body.append(f'<text class="cap2" x="20" y="{H - 30}">Không cache, prompt gửi lại dài thêm mỗi lượt nên chi phí '
            'cộng dồn tăng theo bình phương số lượt.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">Có cache, phần gửi lại tính giá đọc cache; output thành phần '
            'lớn của hoá đơn.</text>')

svgkit.write(__file__, "cache-cumulative.svg", svgkit.svg(
    W, H, f"Chi phí cộng dồn của hội thoại trên Sonnet 5.5: không cache tăng theo bình phương số lượt, tới "
    f"{vn(unc[40], 4)} USD ở 40 lượt; có cache gần như tuyến tính, {vn(cached[40], 4)} USD, rẻ hơn {vn(r40, 1)} lần "
    f"(ở 8 lượt là {vn(r8, 1)} lần). Output như nhau ở hai cách và chiếm gần nửa chi phí có cache ở 8 lượt", body))
