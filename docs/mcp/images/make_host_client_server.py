"""Sinh host-client-server.svg: SVG tĩnh minh hoạ §1.1 "MCP là gì: host, client, server".

Host chứa model và vòng lặp tool call. Host nối ba server nên có ba client, mỗi client
giữ kết nối tới đúng một server. Model không nói chuyện với server: host lấy tool qua
client, đưa vào prompt như tool definition, nhận tool call từ model rồi chuyển tới đúng
server; kết quả đi ngược lại vào context. Theo bản 2026-07-28, server không gửi request
ngược cho client: mọi request do client khởi đầu, server chỉ trả result.

Màu: cam là yêu cầu đi ra (tool call model sinh, request client gửi); xanh dương là dữ
liệu về model (tool definition, tool result). Ba server (orders, GitHub, Sentry) lấy từ
ví dụ trong bài. Hình không có số liệu.

Chạy: python3 make_host_client_server.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 920, 470
HX, HY, HW, HH = 20, 96, 600, 300          # khung host
MX, MW = 40, 120                            # model
LX, LW = 260, 160                           # vòng lặp tool call
CX, CW, CH = 470, 130, 54                   # client
SX, SW = 720, 160                           # server
ROWS = [182, 262, 342]                      # tâm y của từng cặp client–server
MID = ROWS[1]
BOXH = 96

assert HX + HW < SX - 60, "phải chừa chỗ cho mũi tên client–server"
assert SX + SW <= W - 40, "lề phải ≥ 30 px"
assert HY + 50 < ROWS[0] - CH / 2 and ROWS[-1] + CH / 2 < HY + HH - 10

body: list[str] = []
body.append('<text class="title" x="20" y="38">Model không nói chuyện với server: mọi lời gọi đi qua host</text>')
body.append('<text class="cap2" x="20" y="62">Mỗi client giữ kết nối tới đúng một server. '
            'Host nối ba server thì có ba client.</text>')
body.append(svgkit.legend(560, 86, [("orange", "yêu cầu đi ra"), ("blue", "dữ liệu về model")]))

# Khung host
body.append(f'<rect class="lmuted" x="{HX}" y="{HY}" width="{HW}" height="{HH}" rx="10"/>')
body.append(f'<text class="row" x="{HX + 16}" y="{HY + 26}">Host</text>')
body.append(f'<text class="sub" x="{HX + 60}" y="{HY + 26}">Claude Desktop, Claude Code, IDE, agent của bạn</text>')


def card(x: float, y: float, w: float, h: float, title: str, subs: list[str], cls: str = "tok") -> str:
    out = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="7"/>'
    top = y + h / 2 - 8 * len(subs) + 4
    out += f'<text class="row mid" x="{x + w / 2}" y="{top}">{title}</text>'
    for i, s in enumerate(subs):
        out += f'<text class="sub mid" x="{x + w / 2}" y="{top + 18 + 16 * i}">{s}</text>'
    return out


body.append(card(MX, MID - BOXH / 2, MW, BOXH, "Model", ["chọn tool,", "sinh tool call"]))
body.append(card(LX, MID - BOXH / 2, LW, BOXH, "Vòng lặp", ["quyết định có gọi,", "có hỏi người dùng"]))

# Model <-> vòng lặp
ya, yb = MID - 14, MID + 14
body.append(f'<path class="lblue" marker-end="url(#ablue)" d="M{LX},{ya}H{MX + MW + 4}"/>')
body.append(f'<text class="sub mid" x="{(MX + MW + LX) / 2}" y="{ya - 24}">tool definition,</text>')
body.append(f'<text class="sub mid" x="{(MX + MW + LX) / 2}" y="{ya - 9}">tool result</text>')
body.append(f'<path class="lorange" marker-end="url(#aorange)" d="M{MX + MW},{yb}H{LX - 4}"/>')
body.append(f'<text class="sub mid" x="{(MX + MW + LX) / 2}" y="{yb + 20}">tool call</text>')

body.append(f'<text class="sub mid" x="{CX + CW / 2}" y="{HY + 52}">mỗi client một server</text>')
body.append(f'<text class="row mid" x="{SX + SW / 2}" y="{HY + 26}">Server</text>')
body.append(f'<text class="sub mid" x="{SX + SW / 2}" y="{HY + 52}">tool, resource, prompt</text>')

servers = [("orders", "process local, stdio"), ("GitHub", "remote, HTTP"), ("Sentry", "remote, HTTP")]
for i, (y, (name, sub)) in enumerate(zip(ROWS, servers)):
    # vòng lặp -> client: chỉ là kết nối trong cùng process
    body.append(f'<path class="lmuted" d="M{LX + LW},{MID}L{CX},{y}"/>')
    body.append(card(CX, y - CH / 2, CW, CH, f"Client {i + 1}", []))
    body.append(card(SX, y - CH / 2, SW, CH, name, [sub]))
    body.append(f'<path class="lorange" marker-end="url(#aorange)" d="M{CX + CW},{y - 9}H{SX - 4}"/>')
    body.append(f'<path class="lblue dash" marker-end="url(#ablue)" d="M{SX},{y + 9}H{CX + CW + 4}"/>')
    if i == 0:
        body.append(f'<text class="sub mid" x="{(CX + CW + SX) / 2}" y="{y - 16}">request</text>')
        body.append(f'<text class="sub mid" x="{(CX + CW + SX) / 2}" y="{y + 29}">result</text>')

body.append(f'<text class="cap" x="20" y="{H - 44}">Host lấy tool qua client (tools/list), đưa vào prompt; '
            'model sinh tool call; host chuyển tới đúng server (tools/call).</text>')
body.append(f'<text class="cap2" x="20" y="{H - 20}">Bản 2026-07-28: server không gửi request ngược cho client; '
            'mọi luồng do client khởi đầu.</text>')

svgkit.write(__file__, "host-client-server.svg", svgkit.svg(
    W, H, "Host chứa model, vòng lặp tool call và ba client, mỗi client nối một server (orders, GitHub, "
    "Sentry). Model không nói chuyện với server: host lấy tool qua client, đưa vào prompt, nhận tool call "
    "từ model rồi chuyển tới đúng server; kết quả đi ngược lại vào context. Mọi request do client khởi đầu",
    body))
