"""Sinh oauth-flow.svg: SVG tĩnh minh hoạ §2.3 "Luồng khi client gặp server lần đầu".

Sáu bước theo danh sách trong bài (đặc tả Authorization bản 2026-07-28):
1. Client gọi server không token; server trả 401 với WWW-Authenticate có resource_metadata.
2. Client đọc Protected Resource Metadata (RFC 9728), lấy authorization_servers.
3. Client lấy metadata của AS (RFC 8414 hoặc OpenID Connect Discovery).
4. Client đăng ký, mở trình duyệt với PKCE và tham số resource; nhận code.
5. Client đổi code lấy token, cũng kèm resource.
6. Mọi request sau mang Authorization: Bearer <token>; server chỉ nhận token phát cho
   chính nó (audience), token có aud khác bị 401 (bảng kết quả trong bài).

Màu: xanh dương đánh dấu sợi dây audience (tham số resource ở bước 4, 5 và bước server
kiểm aud); đỏ có sọc là 401. Không dùng cam để không đặt cạnh đỏ. Hình không có số liệu.

Chạy: python3 make_oauth_flow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 900, 730
XC, XS, XA = 120, 440, 760          # lifeline: client, MCP server, authorization server
TOP, BOT = 126, 650
STEP = [160, 244, 328, 412, 496, 580]     # y của request ở mỗi bước; response ở y + RESP
RESP = 40
assert all(b - a >= RESP + 44 for a, b in zip(STEP, STEP[1:]))
assert STEP[-1] + RESP < BOT
assert XA + 100 <= W - 30

body: list[str] = []
body.append('<text class="title" x="20" y="38">Client gặp remote server lần đầu: từ 401 tới Bearer token</text>')
body.append('<text class="cap2" x="20" y="62">resource là URI của MCP server; '
            'server chỉ nhận token phát cho chính nó.</text>')

for x, name, sub in ((XC, "MCP client", "OAuth client"), (XS, "MCP server", "resource server"),
                     (XA, "Authorization server", "IdP hoặc dịch vụ của bạn")):
    body.append(f'<rect class="tok" x="{x - 95}" y="{TOP - 50}" width="190" height="46" rx="7"/>')
    body.append(f'<text class="row mid" x="{x}" y="{TOP - 31}">{name}</text>')
    body.append(f'<text class="sub mid" x="{x}" y="{TOP - 13}">{sub}</text>')
    body.append(f'<line class="lmuted dash" x1="{x}" y1="{TOP - 4}" x2="{x}" y2="{BOT}"/>')


def arrow(x1: float, x2: float, y: float, cls: str = "lfg", marker: str = "afg") -> str:
    end = x2 - 4 if x2 > x1 else x2 + 4
    return f'<path class="{cls}" marker-end="url(#{marker})" d="M{x1},{y}H{end}"/>'


def label(x: float, y: float, text: str, cls: str = "cap") -> str:
    return f'<text class="{cls} mid halo" x="{x}" y="{y}">{text}</text>'


def chip(x: float, y: float, text: str, cls: str) -> str:
    w = 8.4 * len(text) + 16
    return (f'<rect class="{cls}" x="{x - w / 2:.1f}" y="{y - 12}" width="{w:.1f}" height="24" rx="5"/>'
            f'<text class="lbl" style="font-size:13px" x="{x}" y="{y + 5}">{text}</text>')


MS = (XC + XS) / 2
MA = (XC + XA) / 2

# 1. không token -> 401
y = STEP[0]
body.append(arrow(XC, XS, y))
body.append(label(MS, y - 10, "1. request không token"))
body.append(arrow(XS, XC, y + RESP, "lred dash", "ared"))
body.append(chip(XC + 40, y + RESP, "401", "hatch"))
body.append(f'<text class="sub halo" x="{XC + 70}" y="{y + RESP - 8}">WWW-Authenticate có resource_metadata</text>')
# 2. Protected Resource Metadata
y = STEP[1]
body.append(arrow(XC, XS, y))
body.append(label(MS, y - 10, "2. Protected Resource Metadata (RFC 9728)"))
body.append(arrow(XS, XC, y + RESP, "lfg dash"))
body.append(label(MS, y + RESP - 8, "authorization_servers", "sub"))
# 3. metadata của AS
y = STEP[2]
body.append(arrow(XC, XA, y))
body.append(label(MA, y - 10, "3. metadata của AS (RFC 8414 hoặc OpenID Connect Discovery)"))
body.append(arrow(XA, XC, y + RESP, "lfg dash"))
body.append(label(MA, y + RESP - 8, "các endpoint của AS", "sub"))
# 4. đăng ký, trình duyệt, PKCE + resource
y = STEP[3]
body.append(arrow(XC, XA, y))
body.append(label(MA, y - 10, "4. đăng ký client, mở trình duyệt với PKCE"))
body.append(chip(XA - 80, y, "resource", "blue"))
body.append(arrow(XA, XC, y + RESP, "lfg dash"))
body.append(label(MA, y + RESP - 8, "code (người dùng đã đăng nhập, đồng ý)", "sub"))
# 5. đổi code lấy token
y = STEP[4]
body.append(arrow(XC, XA, y))
body.append(label(MA, y - 10, "5. đổi code lấy token"))
body.append(chip(XA - 80, y, "resource", "blue"))
body.append(arrow(XA, XC, y + RESP, "lfg dash"))
body.append(label(MA, y + RESP - 8, "access token phát cho MCP server", "sub"))
# 6. Bearer token, server kiểm aud
y = STEP[5]
body.append(arrow(XC, XS, y))
body.append(label(MS, y - 10, "6. Authorization: Bearer &lt;token&gt;"))
body.append(arrow(XS, XC, y + RESP, "lfg dash"))
body.append(label(MS, y + RESP - 8, "200, tool chạy với subject của token", "sub"))
KX, KW, KH = XS + 24, 250, 58
body.append(f'<rect class="blue" x="{KX}" y="{y - 22}" width="{KW}" height="{KH}" rx="7"/>')
body.append(f'<text class="cap mid" x="{KX + KW / 2}" y="{y - 1}">kiểm aud = chính server</text>')
body.append(f'<text class="sub mid" x="{KX + KW / 2}" y="{y + 18}">token phát cho server khác: 401</text>')

body.append(svgkit.legend(20, H - 52, [("blue", "ràng buộc audience"), ("hatch", "bị từ chối")]))
body.append(f'<text class="cap2" x="20" y="{H - 16}">Client MUST gửi resource ở cả bước 4 và 5; '
            'token không bao giờ nằm trong query string.</text>')

svgkit.write(__file__, "oauth-flow.svg", svgkit.svg(
    W, H, "Client gọi MCP server không token và nhận 401 có resource_metadata; đọc Protected Resource "
    "Metadata để biết authorization server; lấy metadata của AS; mở trình duyệt với PKCE và resource để "
    "lấy code; đổi code lấy token, cũng kèm resource; rồi gửi Bearer token. Server chỉ nhận token có "
    "audience là chính nó, token phát cho server khác bị 401", body))
