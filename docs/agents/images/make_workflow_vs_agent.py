"""Sinh workflow-vs-agent.svg: SVG tĩnh so workflow với agent (§1.1).

Hai ví dụ lấy từ §1.1:
- Workflow: pipeline trích xuất hoá đơn gọi model ba lần theo thứ tự cố định.
  Mũi tên màu chữ: code quyết định bước kế tiếp, đường đi biết trước.
- Agent: chatbot hỗ trợ có hai tool (tra đơn hàng, đọc chính sách), tự chọn gọi
  tool nào, rồi quyết định có hoàn tiền không. Mũi tên cam: model quyết định.
  Vòng lặp dừng khi model trả lời mà không gọi tool nào (§1.2).
Hình tĩnh vì đây là cấu trúc, không có diễn biến theo thời gian cần thấy.

Chạy: python3 make_workflow_vs_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 980, 430
BH = 46
body: list[str] = []


def node(x: float, y: float, w: float, cls: str, top: str, sub: str = "") -> str:
    out = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
    if sub:
        out += (f'<text class="cap mid" x="{x + w / 2}" y="{y + 20}">{top}</text>'
                f'<text class="sub mid" x="{x + w / 2}" y="{y + 37}">{sub}</text>')
    else:
        out += f'<text class="cap mid" x="{x + w / 2}" y="{y + BH / 2 + 5}">{top}</text>'
    return out


def arrow(x1, y1, x2, y2, cls="lfg", m="afg", extra="") -> str:
    return f'<path class="{cls} {extra}" d="M{x1},{y1}L{x2},{y2}" marker-end="url(#{m})"/>'


body.append('<text class="title" x="20" y="38">Workflow và agent: ai quyết định bước kế tiếp</text>')
body.append(svgkit.legend(W - 300, 38, [("orange", "lần gọi model"), ("tok", "code, tool, dữ liệu")]))

# Đường chia hai panel
MID = 470
body.append(f'<line class="grid" x1="{MID}" y1="70" x2="{MID}" y2="{H - 20}"/>')

# ---------------- Workflow ----------------
LX = 30
body.append(f'<text class="row" x="{LX}" y="86">Workflow</text>')
body.append(f'<text class="sub" x="{LX}" y="104">trích xuất hoá đơn: 3 lần gọi theo thứ tự cố định</text>')
NW, NX = 250, LX + 80
ys = [130, 196, 262]
body.append(node(NX, ys[0], NW, "orange", "Lần gọi 1"))
body.append(node(NX, ys[1], NW, "orange", "Lần gọi 2"))
body.append(node(NX, ys[2], NW, "orange", "Lần gọi 3"))
for a, b in zip(ys, ys[1:]):
    body.append(arrow(NX + NW / 2, a + BH + 2, NX + NW / 2, b - 4))
# Ngoặc bên trái: code điều khiển
bx = NX - 18
body.append(f'<path class="lfg" d="M{bx + 8},{ys[0] + 6}H{bx}V{ys[2] + BH - 6}H{bx + 8}"/>')
body.append(f'<text class="sub end" x="{bx - 6}" y="{(ys[0] + ys[2] + BH) / 2 - 2}">code</text>')
body.append(f'<text class="sub end" x="{bx - 6}" y="{(ys[0] + ys[2] + BH) / 2 + 14}">nối</text>')
body.append(f'<text class="cap" x="{LX}" y="{H - 22}">Đường đi biết trước: test từng nhánh như code.</text>')

# ---------------- Agent ----------------
RX = MID + 30
body.append(f'<text class="row" x="{RX}" y="86">Agent</text>')
body.append(f'<text class="sub" x="{RX}" y="104">chatbot hỗ trợ: 2 tool, model tự chọn gọi gì</text>')
MW, MX, MY = 150, RX + 10, 196
body.append(node(MX, MY, MW, "orange", "Model", "chọn bước tiếp"))
TW, TX = 190, RX + 250
T1, T2 = 140, 252
body.append(node(TX, T1, TW, "tok", "tra đơn hàng", "tool"))
body.append(node(TX, T2, TW, "tok", "đọc chính sách", "tool"))
# Model -> tool (cam, model quyết định) và kết quả quay về (nét đứt màu chữ).
# Mỗi tool một cặp đường gần song song, đầu ở model lệch nhau để không cắt nhau.
mx2 = MX + MW
for ty, (y_out, y_ret) in ((T1, (MY + 8, MY + 18)), (T2, (MY + 28, MY + 38))):
    cy = ty + BH / 2
    body.append(f'<path class="lorange" d="M{mx2 + 2},{y_out}L{TX - 6},{cy - 7}" marker-end="url(#aorange)"/>')
    body.append(f'<path class="lfg dash" d="M{TX - 2},{cy + 7}L{mx2 + 8},{y_ret}" marker-end="url(#afg)"/>')
body.append(f'<text class="sub torange halo end" x="{TX - 40}" y="{T1 + 6}">gọi tool</text>')
body.append(f'<text class="sub halo end" x="{TX - 40}" y="{T2 + BH + 4}">kết quả</text>')
# Lối ra: trả lời khi không gọi tool nào
OY = 318
body.append(f'<path class="lorange" d="M{MX + MW / 2},{MY + BH + 2}L{MX + MW / 2},{OY - 4}" marker-end="url(#aorange)"/>')
body.append(node(MX - 10, OY, MW + 20, "tok", "trả lời", "hoàn tiền hay không"))
body.append(f'<text class="sub" x="{MX + MW + 22}" y="{OY + 20}">dừng khi model không</text>')
body.append(f'<text class="sub" x="{MX + MW + 22}" y="{OY + 36}">gọi tool nào nữa</text>')
body.append(f'<text class="cap" x="{RX}" y="{H - 22}">Đường đi do model chọn: chấm kết quả cuối.</text>')

svgkit.write(__file__, "workflow-vs-agent.svg", svgkit.svg(
    W, H, "Workflow: code nối ba lần gọi model theo thứ tự cố định nên đường đi biết trước. "
    "Agent: model tự chọn gọi tool nào và khi nào dừng. Khác biệt là ai quyết định bước kế tiếp", body))
