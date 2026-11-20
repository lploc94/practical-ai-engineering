"""Sinh traffic-paths.svg: SVG tĩnh minh hoạ mục "Phút 0–2: Traffic tấn công đi qua đâu?".

Hai đường traffic tấn công tới origin. Nhánh A: botnet đi qua Cloudflare (zone đã proxy),
nên DDoS ruleset, custom rule, rate limiting, Under Attack lọc được trước khi tới origin.
Nhánh B: kẻ tấn công đã biết IP origin, đánh thẳng vào IP đó, không qua lớp nào của
Cloudflare, nên mọi rule trên Cloudflare không có tác dụng.

Màu: xanh dương là traffic đã qua rule của zone; đỏ là traffic tấn công chưa bị lọc.
Hình không có số liệu.

Chạy: python3 make_traffic_paths.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 860, 420
SX, SW, SH = 20, 170, 54              # cột nguồn traffic
CX, CY, CW, CH = 300, 100, 250, 170   # khung Cloudflare
OX, OY, OW, OH = 680, 150, 160, 200   # khung origin
Y_USER, Y_BOT, Y_DIRECT = 140, 225, 320

assert CY + CH < Y_DIRECT - 30, "đường nhánh B phải đi dưới khung Cloudflare"
assert OY < CY + CH / 2 < Y_DIRECT < OY + OH

body: list[str] = []
body.append('<text class="title" x="20" y="38">Rule Cloudflare chỉ lọc traffic đi qua Cloudflare</text>')
body.append(svgkit.legend(W - 360, 38, [("blue", "đã qua rule của zone"), ("red", "traffic tấn công")]))
body.append('<text class="cap2" x="20" y="62">Biết IP origin thì kẻ tấn công không cần đi qua Cloudflare.</text>')


def source(y: float, title: str, sub: str) -> str:
    return (f'<rect class="tok" x="{SX}" y="{y - SH / 2}" width="{SW}" height="{SH}" rx="7"/>'
            f'<text class="row mid" x="{SX + SW / 2}" y="{y - 3}">{title}</text>'
            f'<text class="sub mid" x="{SX + SW / 2}" y="{y + 15}">{sub}</text>')


body.append(source(Y_USER, "Người dùng thật", "trình duyệt, app"))
body.append(source(Y_BOT, "Botnet", "HTTP flood"))
body.append(source(Y_DIRECT, "Kẻ biết IP origin", "SYN/UDP/HTTP flood"))

# Khung Cloudflare
body.append(f'<rect class="blue" x="{CX}" y="{CY}" width="{CW}" height="{CH}" rx="9"/>')
body.append(f'<text class="row mid" x="{CX + CW / 2}" y="{CY + 26}">Cloudflare (zone của bạn)</text>')
for i, t in enumerate(["DDoS ruleset, custom rule", "rate limiting, Under Attack"]):
    body.append(f'<text class="sub mid" x="{CX + CW / 2}" y="{CY + 48 + 17 * i}">{t}</text>')
# Phần bị chặn ngay trong Cloudflare
BX, BY, BW, BH = CX + 24, Y_BOT - 22, 140, 44
body.append(f'<rect class="hatch" x="{BX}" y="{BY}" width="{BW}" height="{BH}" rx="6"/>')
body.append(f'<text class="cap mid halo" x="{BX + BW / 2}" y="{BY + 27}">chặn / challenge</text>')

# Origin
body.append(f'<rect class="tok" x="{OX}" y="{OY}" width="{OW}" height="{OH}" rx="9"/>')
body.append(f'<text class="row mid" x="{OX + OW / 2}" y="{OY + 26}">Origin</text>')
body.append(f'<text class="sub mid" x="{OX + OW / 2}" y="{OY + 44}">server vật lý</text>')

# Mũi tên
body.append(f'<path class="lblue" marker-end="url(#ablue)" d="M{SX + SW},{Y_USER}H{CX - 4}"/>')
body.append(f'<path class="lred" marker-end="url(#ared)" d="M{SX + SW},{Y_BOT}H{BX - 4}"/>')
body.append(f'<text class="sub halo" x="{SX + SW + 14}" y="{Y_BOT - 8}">Nhánh A</text>')
YCO = OY + 30
assert CY < YCO < CY + CH
body.append(f'<path class="lblue" marker-end="url(#ablue)" d="M{CX + CW},{YCO}H{OX - 4}"/>')
body.append(f'<text class="sub mid" x="{(CX + CW + OX) / 2}" y="{YCO - 10}">phần đã lọc</text>')
body.append(f'<path class="lred" marker-end="url(#ared)" d="M{SX + SW},{Y_DIRECT}H{OX - 4}"/>')
body.append(f'<text class="sub halo" x="{SX + SW + 14}" y="{Y_DIRECT - 8}">Nhánh B</text>')
body.append(f'<text class="cap mid" x="{CX + CW / 2}" y="{Y_DIRECT - 10}">đánh thẳng vào IP origin</text>')

body.append(f'<text class="cap" x="20" y="{H - 44}">Nhánh A: bật Under Attack, viết rule trên Cloudflare.</text>')
body.append(f'<text class="cap" x="20" y="{H - 20}">Nhánh B: rule Cloudflare vô dụng; phải chặn ngay trên origin '
            '(firewall), hoặc nhờ nhà cung cấp chặn phía trên.</text>')

svgkit.write(__file__, "traffic-paths.svg", svgkit.svg(
    W, H, "Botnet đi qua Cloudflare bị DDoS ruleset và custom rule lọc trước khi tới origin; "
    "kẻ tấn công biết IP origin đánh thẳng vào server, không đi qua lớp nào của Cloudflare nên "
    "rule Cloudflare không có tác dụng", body))
