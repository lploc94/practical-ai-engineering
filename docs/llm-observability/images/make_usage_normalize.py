"""Sinh usage-normalize.svg: SVG tĩnh về chuẩn hoá token giữa provider (§1.3).

Quy ước OTel: gen_ai.usage.input_tokens gồm cả cache read và cache write,
gen_ai.usage.output_tokens gồm cả reasoning. Hai chỗ dễ sai nhất trong bài:
- Claude: input_tokens chỉ là phần sau cache breakpoint cuối; tổng input =
  input_tokens + cache_read_input_tokens + cache_creation_input_tokens.
- Gemini: candidatesTokenCount không gồm thoughtsTokenCount.

Số liệu giả định cho một request:
- Claude: input_tokens 500, cache_read_input_tokens 9.000,
  cache_creation_input_tokens 500. Ghi thẳng 500 làm cache hit rate
  9.000 / 500 = 1.800%; chuẩn hoá thì 9.000 / 10.000 = 90%.
- Gemini: candidatesTokenCount 400, thoughtsTokenCount 1.600. Ghi thẳng 400
  bỏ sót 1.600 token thinking; chuẩn hoá thì output = 2.000.
Hình tĩnh vì ý chính là so thành phần của tổng, không có diễn biến theo thời gian.

Chạy: python3 make_usage_normalize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

C_IN, C_READ, C_WRITE = 500, 9_000, 500
C_TOTAL = C_IN + C_READ + C_WRITE
assert C_TOTAL == 10_000
HIT_WRONG, HIT_RIGHT = C_READ / C_IN, C_READ / C_TOTAL
assert round(HIT_WRONG * 100) == 1800 and round(HIT_RIGHT * 100) == 90

G_CAND, G_THOUGHT = 400, 1_600
G_TOTAL = G_CAND + G_THOUGHT
assert G_TOTAL == 2_000

W, H = 820, 516
X0, X1 = 200, 790
BH = 28
MONO = f'style="font-family:{svgkit.MONO}"'


def fmt(v: int) -> str:
    return f"{v:,}".replace(",", ".")


PCT_WRONG = fmt(round(HIT_WRONG * 100)) + "%"   # 1.800%
PCT_RIGHT = fmt(round(HIT_RIGHT * 100)) + "%"


def seg(x: float, y: float, w: float, cls: str) -> str:
    return f'<rect class="{cls}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{BH}" rx="4"/>'


def row_label(y: float, main: str, sub: str = "") -> str:
    out = f'<text class="row" x="20" y="{y + 19}">{main}</text>'
    if sub:
        out += f'<text class="sub" x="20" y="{y + 35}">{sub}</text>'
    return out


body: list[str] = []
body.append('<text class="title" x="20" y="38">Chuẩn hoá token trước khi ghi vào gen_ai.usage.*</text>')
body.append(svgkit.legend(20, 66, [("blue", "đọc từ cache"), ("tok", "token khác"),
                                    ("hatch", "ghi thẳng field của provider: sai")]))

# Panel 1: Claude, input. Thang: C_TOTAL token = X1 - X0 px
sc = (X1 - X0) / C_TOTAL
y0 = 112
body.append(f'<text class="cap" x="20" y="{y0 - 10}" style="font-weight:600">Claude: input_tokens chỉ là phần sau cache breakpoint</text>')
ya = y0 + 14
body.append(row_label(ya, "usage trả về"))
# Thứ tự theo prompt: phần đọc từ cache (prefix) trước, phần sau breakpoint cuối cùng
xa = X0
body.append(seg(xa, ya, C_READ * sc - 2, "blue"))
xb = xa + C_READ * sc
body.append(seg(xb, ya, C_WRITE * sc - 2, "tok"))
xc = xb + C_WRITE * sc
body.append(seg(xc, ya, C_IN * sc, "tok"))
body.append(f'<text class="lbl halo" x="{(xa + xb) / 2:.1f}" y="{ya + 19}" style="font-size:13px">'
            f'cache_read_input_tokens {fmt(C_READ)}</text>')
body.append(f'<text class="sub end" x="{X1}" y="{ya - 6}" {MONO}>input_tokens {fmt(C_IN)}</text>')
body.append(f'<text class="sub end" x="{xc - 2:.1f}" y="{ya + BH + 15}" {MONO}>cache_creation_input_tokens {fmt(C_WRITE)}</text>')

yb = ya + BH + 30
body.append(row_label(yb, "Ghi thẳng", "input_tokens"))
body.append(seg(xc, yb, C_IN * sc, "hatch"))
body.append(f'<text class="cap end" x="{xc - 10:.1f}" y="{yb + 19}">input = {fmt(C_IN)} → cache hit rate '
            f'{fmt(C_READ)} / {fmt(C_IN)} = <tspan style="font-weight:600">{PCT_WRONG}</tspan></text>')

yc = yb + BH + 22
body.append(row_label(yc, "Chuẩn hoá", "cộng cả cache"))
body.append(f'<rect class="lfg" x="{X0}" y="{yc}" width="{X1 - X0}" height="{BH}" rx="4"/>')
body.append(f'<text class="cap" x="{X0 + 10}" y="{yc + 19}">input = {fmt(C_IN)} + {fmt(C_READ)} + {fmt(C_WRITE)} = '
            f'{fmt(C_TOTAL)} → cache hit rate <tspan style="font-weight:600">{PCT_RIGHT}</tspan></text>')

# Panel 2: Gemini, output. Thang riêng: G_TOTAL token = X1 - X0 px
sg = (X1 - X0) / G_TOTAL
y1 = yc + BH + 62
body.append(f'<text class="cap" x="20" y="{y1 - 10}" style="font-weight:600">Gemini: candidatesTokenCount không gồm thinking</text>')
yd = y1 + 14
body.append(row_label(yd, "usage trả về"))
body.append(seg(X0, yd, G_CAND * sg - 2, "tok"))
body.append(seg(X0 + G_CAND * sg, yd, G_THOUGHT * sg, "tok"))
body.append(f'<text class="lbl" x="{X0 + G_CAND * sg / 2:.1f}" y="{yd + 19}" style="font-size:13px">candidates {fmt(G_CAND)}</text>')
body.append(f'<text class="lbl" x="{X0 + G_CAND * sg + G_THOUGHT * sg / 2:.1f}" y="{yd + 19}" style="font-size:13px">'
            f'thoughtsTokenCount {fmt(G_THOUGHT)}</text>')

ye = yd + BH + 18
body.append(row_label(ye, "Ghi thẳng", "candidatesTokenCount"))
body.append(seg(X0, ye, G_CAND * sg, "hatch"))
body.append(f'<text class="cap" x="{X0 + G_CAND * sg + 10:.1f}" y="{ye + 19}">output = {fmt(G_CAND)} → bỏ sót '
            f'{fmt(G_THOUGHT)} token thinking vẫn bị tính tiền</text>')

yf = ye + BH + 22
body.append(row_label(yf, "Chuẩn hoá", "cộng thoughts"))
body.append(f'<rect class="lfg" x="{X0}" y="{yf}" width="{X1 - X0}" height="{BH}" rx="4"/>')
body.append(f'<text class="cap" x="{X0 + 10}" y="{yf + 19}">output = {fmt(G_CAND)} + {fmt(G_THOUGHT)} = {fmt(G_TOTAL)}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Số token giả định cho một request. Mỗi panel có thang riêng.</text>')

svgkit.write(__file__, "usage-normalize.svg", svgkit.svg(
    W, H, f"Claude trả input_tokens {fmt(C_IN)}, cache_read_input_tokens {fmt(C_READ)}, cache_creation_input_tokens "
    f"{fmt(C_WRITE)}: ghi thẳng input_tokens làm cache hit rate thành {PCT_WRONG}, cộng cả cache thì input "
    f"{fmt(C_TOTAL)} và hit rate {PCT_RIGHT}. Gemini trả candidatesTokenCount {fmt(G_CAND)} và thoughtsTokenCount "
    f"{fmt(G_THOUGHT)}: ghi thẳng candidates bỏ sót phần thinking, chuẩn hoá thì output {fmt(G_TOTAL)}", body))
