"""Sinh allowlist-gap.svg: SVG tĩnh minh hoạ ghi chú cuối mục B1 (allowlist IP Cloudflare).

Firewall trên origin chỉ cho dải IP Cloudflare vào 80/443. Ba đường tới origin:
1. Visitor đi qua zone của bạn: IP nguồn là IP Cloudflare, được nhận.
2. Kẻ tấn công đánh thẳng từ IP của họ: không phải IP Cloudflare, bị drop.
3. Kẻ tấn công tạo zone trong tài khoản Cloudflare riêng, trỏ vào IP origin của bạn:
   IP nguồn vẫn là IP Cloudflare nên được nhận, dù không đi qua rule nào của zone bạn.
Đường 3 chỉ bị chặn khi dùng Tunnel hoặc Authenticated Origin Pulls với certificate riêng.

Màu: xanh dương là traffic đã qua rule của zone bạn; đỏ là traffic tấn công.
Hình không có số liệu.

Chạy: python3 make_allowlist_gap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 880, 430
SX, SW, SH = 20, 170, 54          # cột nguồn
MX, MW, MH = 250, 190, 54         # cột "qua Cloudflare"
FX = 560                          # firewall (đường dọc)
OX, OW = 690, 170                 # origin
ROWS = [130, 225, 320]            # ba đường
OY, OH = ROWS[0] - 40, ROWS[-1] - ROWS[0] + 80

assert SX + SW < MX and MX + MW < FX < OX and OX + OW <= W - 20

body: list[str] = []
body.append('<text class="title" x="20" y="38">Allowlist IP Cloudflare: kiểm IP nguồn, không kiểm zone</text>')
body.append('<text class="cap2" x="20" y="62">Firewall trên origin chỉ cho IP Cloudflare vào 80/443 (script B1).</text>')
body.append(svgkit.legend(20, 86, [("blue", "đã qua rule của zone bạn"), ("red", "traffic tấn công"),
                                   ("hatch", "bị drop")]))


def node(x: float, y: float, w: float, cls: str, title: str, sub: str) -> str:
    return (f'<rect class="{cls}" x="{x}" y="{y - SH / 2}" width="{w}" height="{SH}" rx="7"/>'
            f'<text class="row mid" x="{x + w / 2}" y="{y - 3}">{title}</text>'
            f'<text class="sub mid" x="{x + w / 2}" y="{y + 15}">{sub}</text>')


y1, y2, y3 = ROWS
body.append(node(SX, y1, SW, "tok", "Người dùng thật", "vào domain của bạn"))
body.append(node(SX, y2, SW, "tok", "Kẻ tấn công", "đánh thẳng IP origin"))
body.append(node(SX, y3, SW, "tok", "Kẻ tấn công", "zone riêng trỏ vào IP bạn"))
body.append(node(MX, y1, MW, "blue", "Zone của bạn", "rule của bạn chạy ở đây"))
body.append(node(MX, y3, MW, "tok", "Zone của kẻ tấn công", "cũng trên Cloudflare"))

# Firewall
body.append(f'<line class="lfg" x1="{FX}" y1="{y1 - 48}" x2="{FX}" y2="{y3 + 40}" style="stroke-width:4"/>')
body.append(f'<text class="sub mid" x="{FX}" y="{y3 + 60}">firewall origin</text>')

# Origin
body.append(f'<rect class="tok" x="{OX}" y="{OY}" width="{OW}" height="{OH}" rx="9"/>')
body.append(f'<text class="row mid" x="{OX + OW / 2}" y="{(y1 + y3) / 2 - 4}">Origin</text>')
body.append(f'<text class="sub mid" x="{OX + OW / 2}" y="{(y1 + y3) / 2 + 14}">cổng 80/443</text>')

# Đường 1: qua zone bạn, IP Cloudflare, được nhận
body.append(f'<path class="lblue" marker-end="url(#ablue)" d="M{SX + SW},{y1}H{MX - 4}"/>')
body.append(f'<path class="lblue" marker-end="url(#ablue)" d="M{MX + MW},{y1}H{OX - 4}"/>')
body.append(f'<text class="sub mid halo" x="{(MX + MW + FX) / 2}" y="{y1 - 10}">IP Cloudflare</text>')
body.append(f'<text class="sub mid halo" x="{(FX + OX) / 2}" y="{y1 - 10}">nhận</text>')

# Đường 2: IP lạ, bị drop
DW = 70
body.append(f'<path class="lred" marker-end="url(#ared)" d="M{SX + SW},{y2}H{FX - DW - 4}"/>')
body.append(f'<text class="sub mid halo" x="{(SX + SW + FX - DW) / 2}" y="{y2 - 10}">IP không phải Cloudflare</text>')
body.append(f'<rect class="hatch" x="{FX - DW}" y="{y2 - 18}" width="{DW}" height="36" rx="6"/>')
body.append(f'<text class="cap mid halo" x="{FX - DW / 2}" y="{y2 + 5}">drop</text>')

# Đường 3: qua zone khác, vẫn là IP Cloudflare, lọt qua
body.append(f'<path class="lred" marker-end="url(#ared)" d="M{SX + SW},{y3}H{MX - 4}"/>')
body.append(f'<path class="lred" marker-end="url(#ared)" d="M{MX + MW},{y3}H{OX - 4}"/>')
body.append(f'<text class="sub mid halo" x="{(MX + MW + FX) / 2}" y="{y3 - 10}">IP Cloudflare</text>')
body.append(f'<text class="sub mid halo" x="{(FX + OX) / 2}" y="{y3 - 10}">lọt qua</text>')

body.append(f'<text class="cap" x="20" y="{H - 20}">Đường dưới không qua rule nào của bạn. Chặn nó bằng Tunnel, hoặc '
            'Authenticated Origin Pulls với certificate riêng.</text>')

svgkit.write(__file__, "allowlist-gap.svg", svgkit.svg(
    W, H, "Firewall chỉ nhận IP Cloudflare: traffic đánh thẳng từ IP lạ bị drop, nhưng kẻ tấn công dùng "
    "zone trong tài khoản Cloudflare riêng trỏ vào IP origin vẫn mang IP Cloudflare nên lọt qua mà không "
    "qua rule nào của zone bạn; chặn đường này cần Tunnel hoặc Authenticated Origin Pulls với certificate riêng",
    body))
