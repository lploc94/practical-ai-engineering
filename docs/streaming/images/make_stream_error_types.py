"""Sinh stream-error-types.svg: SVG tĩnh đặt ba loại lỗi lên dòng thời gian của stream (§3.6).

Mỗi hàng là một response theo thứ tự byte đến client, dùng tên event của Claude làm ví dụ:
- Thành công: HTTP 200, các delta, rồi message_stop.
- Loại 1, lỗi trước stream: status khác 200 (ví dụ 529), không có byte nào của stream.
- Loại 2, lỗi trong stream: 200 đã gửi, vài delta, rồi event: error; không có message_stop.
- Loại 3, đứt kết nối: 200, vài delta, kết nối đóng; không có event lỗi lẫn event kết thúc.
Ý chính: status 200 đi trước token đầu tiên, nên chỉ event kết thúc mới xác nhận thành công.
Hình tĩnh vì cần so bốn hàng cạnh nhau, không có diễn biến cần xem theo thời gian.

Chạy: python3 make_stream_error_types.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 820, 420
BH = 32
SX, SW = 230, 92                 # ô status
DX, DW, DG = SX + SW + 14, 66, 8  # ô delta
EX = DX + 3 * (DW + DG) + 6      # ô thứ tư: event kết thúc / lỗi
EW = 170
assert EX + EW <= W - 20
ROW0, ROWH = 112, 66

body: list[str] = []


def blk(x: float, y: float, w: float, cls: str, label: str, size: int = 13) -> str:
    return (f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="6"/>'
            f'<text class="lbl halo" style="font-size:{size}px" x="{x + w / 2}" y="{y + BH / 2 + 5}">{label}</text>')


def slot(x: float, y: float, w: float, label: str) -> str:
    return (f'<rect class="slot" x="{x}" y="{y}" width="{w}" height="{BH}" rx="6"/>'
            f'<text class="sub mid" x="{x + w / 2}" y="{y + BH / 2 + 4}">{label}</text>')


body.append('<text class="title" x="20" y="38">Status 200 đến trước token đầu, nên lỗi sau đó nằm trong body</text>')
body.append(svgkit.legend(20, 66, [("tok", "status, event kết thúc"), ("blue", "delta nội dung"),
                                   ("hatch", "lỗi")]))

rows = [
    ("Thành công", "có event kết thúc"),
    ("1. Lỗi trước stream", "status khác 200"),
    ("2. Lỗi trong stream", "200, rồi event lỗi"),
    ("3. Đứt kết nối", "không lỗi, không kết thúc"),
]
for i, (name, sub) in enumerate(rows):
    y = ROW0 + i * ROWH
    body.append(f'<text class="row" x="20" y="{y + 14}">{name}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 31}">{sub}</text>')


def deltas(y: float, n: int) -> str:
    return "".join(blk(DX + k * (DW + DG), y, DW, "blue", "delta") for k in range(n))


y = ROW0
body.append(blk(SX, y, SW, "tok", "HTTP 200") + deltas(y, 3) + blk(EX, y, EW, "tok", "message_stop"))
y += ROWH
body.append(blk(SX, y, SW, "hatch", "HTTP 529")
            + f'<text class="cap2" x="{DX}" y="{y + BH / 2 + 5}">chưa có byte nào của stream; SDK xử lý theo status</text>')
y += ROWH
body.append(blk(SX, y, SW, "tok", "HTTP 200") + deltas(y, 2)
            + blk(EX, y, EW, "hatch", "event: error")
            + f'<text class="sub halo" x="{EX + EW}" y="{y + BH + 15}" style="text-anchor:end">'
            'sau đó không có message_stop</text>')
y += ROWH
cut = DX + 2 * (DW + DG) + 4
body.append(blk(SX, y, SW, "tok", "HTTP 200") + deltas(y, 2)
            + f'<path class="lred" style="stroke-width:3" d="M{cut},{y - 6} l8,{BH / 3 + 2} l-8,{BH / 3 + 2} '
            f'l8,{BH / 3 + 4}"/>'
            + slot(cut + 22, y, EX + EW - cut - 22, "kết nối đóng, không có event nào"))

# Mốc: status đã gửi trước mọi delta
lx = SX + SW + 7
body.append(f'<line class="lfg dash" x1="{lx}" y1="{ROW0 - 18}" x2="{lx}" y2="{ROW0 + 3 * ROWH + BH + 6}"/>')
body.append(f'<text class="sub halo" x="{lx + 6}" y="{ROW0 - 10}">sau mốc này status không đổi được nữa</text>')

ay = ROW0 + 3 * ROWH + BH + 26
body.append(f'<line class="axis" x1="{SX}" y1="{ay}" x2="{EX + EW}" y2="{ay}" marker-end="url(#afg)"/>')
body.append(f'<text class="sub end" x="{EX + EW}" y="{ay + 16}">thứ tự byte đến client</text>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">Tên event theo Claude. Chỉ coi là xong khi có event kết thúc '
            '(message_stop, [DONE], ...).</text>')

svgkit.write(__file__, "stream-error-types.svg", svgkit.svg(
    W, H, "Status 200 được gửi trước token đầu tiên: lỗi trước stream có status khác 200; lỗi trong stream đến "
    "sau status 200 dưới dạng event error; đứt kết nối thì không có event lỗi lẫn event kết thúc; chỉ event "
    "kết thúc như message_stop xác nhận stream thành công", body))
