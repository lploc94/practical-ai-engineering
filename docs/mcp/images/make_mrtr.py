"""Sinh mrtr.svg: SVG tĩnh minh hoạ §1.3 "Server cần hỏi giữa chừng: MRTR".

Theo ví dụ trong bài: client gửi tools/call (id 2); server cần GitHub username nên trả
result có resultType "input_required", kèm inputRequests (elicitation/create, mode form)
và requestState. Client hỏi người dùng, người dùng trả "octocat". Client gửi lại cùng
tool call với id mới (id 3), kèm inputResponses và requestState nguyên văn. Server
không giữ gì giữa hai lượt: mọi trạng thái cần thiết nằm trong requestState.

Màu: xanh dương là requestState (dữ liệu server đã tạo, client gửi lại nguyên văn); cam
là inputResponses (dữ liệu mới người dùng vừa nhập). Mũi tên request và result trung
tính. Hình không có số liệu. id 2 lấy từ ví dụ JSON trong bài; id 3 của lượt gửi lại lấy từ
ví dụ "Retry with Input Responses" trong trang Tools của đặc tả (nguồn [4] của bài).

Chạy: python3 make_mrtr.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 880, 560
XU, XC, XS = 110, 390, 720          # lifeline: người dùng, client, server
TOP, BOT = 120, 470                 # đầu và cuối lifeline
Y1, Y2, Y3, Y4, Y5, Y6 = 162, 222, 282, 322, 392, 452
assert TOP < Y1 < Y2 < Y3 < Y4 < Y5 < Y6 < BOT + 10
assert XS + 80 <= W - 40, "hộp tiêu đề server phải chừa lề phải"

body: list[str] = []
body.append('<text class="title" x="20" y="38">MRTR: server hỏi thêm bằng result, không mở kênh ngược</text>')
body.append('<text class="cap2" x="20" y="62">Server không giữ gì giữa hai lượt: '
            'trạng thái nằm trong requestState mà client gửi lại nguyên văn.</text>')

for x, name in ((XU, "Người dùng"), (XC, "Client"), (XS, "Server")):
    body.append(f'<rect class="tok" x="{x - 70}" y="{TOP - 38}" width="140" height="34" rx="7"/>')
    body.append(f'<text class="row mid" x="{x}" y="{TOP - 16}">{name}</text>')
    body.append(f'<line class="lmuted dash" x1="{x}" y1="{TOP - 4}" x2="{x}" y2="{BOT}"/>')


def arrow(x1: float, x2: float, y: float, dashed: bool = False) -> str:
    end = x2 - 4 if x2 > x1 else x2 + 4
    cls = "lfg dash" if dashed else "lfg"
    return f'<path class="{cls}" marker-end="url(#afg)" d="M{x1},{y}H{end}"/>'


def label(x: float, y: float, text: str, cls: str = "cap") -> str:
    return f'<text class="{cls} mid halo" x="{x}" y="{y}">{text}</text>'


def chip(x: float, y: float, text: str, cls: str) -> str:
    """Ô màu chứa tên field, tâm ở (x, y)."""
    w = 8.4 * len(text) + 16
    return (f'<rect class="{cls}" x="{x - w / 2:.1f}" y="{y - 12}" width="{w:.1f}" height="24" rx="5"/>'
            f'<text class="lbl" style="font-size:13px" x="{x}" y="{y + 5}">{text}</text>')


MCS = (XC + XS) / 2
MUC = (XU + XC) / 2

# 1. tools/call id 2
body.append(arrow(XC, XS, Y1))
body.append(label(MCS, Y1 - 10, "tools/call, id: 2"))
# 2. input_required
body.append(arrow(XS, XC, Y2, dashed=True))
body.append(label(MCS, Y2 - 10, 'resultType: "input_required"'))
body.append(chip(MCS - 70, Y2 + 22, "inputRequests", "tok"))
body.append(chip(MCS + 75, Y2 + 22, "requestState", "blue"))
# 3–4. hỏi người dùng
body.append(arrow(XC, XU, Y3))
body.append(label(MUC, Y3 - 10, "form: GitHub username?"))
body.append(arrow(XU, XC, Y4, dashed=True))
body.append(label(MUC, Y4 + 20, '"octocat"', "sub"))
# Server không giữ trạng thái giữa hai lượt
SY0, SY1 = Y2 + 46, Y5 - 22
body.append(f'<rect class="slot" x="{XS - 9}" y="{SY0}" width="18" height="{SY1 - SY0}" rx="4"/>')
body.append(f'<text class="sub mid" x="{MCS + 20}" y="{(SY0 + SY1) / 2 - 2}">server không lưu gì</text>')
body.append(f'<text class="sub mid" x="{MCS + 20}" y="{(SY0 + SY1) / 2 + 14}">giữa hai lượt</text>')
# 5. gửi lại với id mới
body.append(arrow(XC, XS, Y5))
body.append(label(MCS, Y5 - 10, "cùng tools/call, id mới: 3"))
body.append(chip(MCS - 75, Y5 + 22, "inputResponses", "orange"))
body.append(chip(MCS + 75, Y5 + 22, "requestState", "blue"))
# 6. complete
body.append(arrow(XS, XC, Y6 + 20, dashed=True))
body.append(label(MCS, Y6 + 10, 'resultType: "complete"'))

body.append(svgkit.legend(20, H - 52, [("blue", "requestState: server tạo, client gửi lại nguyên văn"),
                                        ("orange", "inputResponses: người dùng vừa nhập")]))
body.append(f'<text class="cap2" x="20" y="{H - 16}">requestState đi qua client nên là input không tin cậy: '
            'ký hoặc mã hoá nếu client không được sửa.</text>')

svgkit.write(__file__, "mrtr.svg", svgkit.svg(
    W, H, "Client gửi tools/call id 2; server trả resultType input_required kèm inputRequests và "
    "requestState; client hỏi người dùng rồi gửi lại cùng tool call với id mới 3, kèm inputResponses "
    "và requestState nguyên văn; server trả complete. Server không lưu gì giữa hai lượt", body))
