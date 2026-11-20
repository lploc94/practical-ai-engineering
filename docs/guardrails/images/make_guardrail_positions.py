"""Sinh guardrail-positions.svg: SVG tĩnh, ba vị trí đặt guardrail và mỗi vị trí thấy gì (§1.2).

Ví dụ lấy từ §1.2 và phần mở đầu:
- Input guardrail chỉ thấy câu hỏi. "Giao trễ thì shop hoàn tiền không?" là câu hỏi
  hợp lệ nên phải cho qua.
- Output guardrail thấy đúng câu khách sẽ đọc, nên bắt được câu hứa
  "Shop cam kết hoàn tiền 100%" trước khi nó hiện ra.
- Guardrail trên hành động: code của tool `issue_refund` tự kiểm điều kiện đơn hàng
  (ngày giao dự kiến, ngày giao thật, trạng thái thanh toán) và trần số tiền, bất kể
  model nói gì.
Màu: xanh dương là bước kiểm tra, đỏ sọc chéo là vi phạm bị chặn. Model chính và
khách để màu trung tính.
Hình tĩnh vì đây là cấu trúc, không có diễn biến theo thời gian cần thấy.

Chạy: python3 make_guardrail_positions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
Q = "giao trễ thì shop hoàn tiền không?"
BAD = "Shop cam kết hoàn tiền 100%"
assert f"Câu \"{Q}\" hoàn toàn hợp lệ" in README
assert "shop cam kết hoàn tiền 100%" in README
assert "Nếu bot có tool `issue_refund`" in README
assert "ngày giao dự kiến, ngày giao thật, trạng thái thanh toán" in README
assert "giới hạn số tiền, bất kể model nói gì" in README

W, H = 980, 440
BH = 52
body: list[str] = []


def node(x: float, y: float, w: float, cls: str, top: str, sub: str = "") -> str:
    out = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
    if sub:
        out += (f'<text class="cap mid" x="{x + w / 2}" y="{y + 22}">{top}</text>'
                f'<text class="sub mid" x="{x + w / 2}" y="{y + 40}">{sub}</text>')
    else:
        out += f'<text class="cap mid" x="{x + w / 2}" y="{y + BH / 2 + 5}">{top}</text>'
    return out


def arrow(x1, y1, x2, y2) -> str:
    return f'<path class="lfg" d="M{x1},{y1}L{x2},{y2}" marker-end="url(#afg)"/>'


body.append('<text class="title" x="20" y="38">Ba chỗ đặt guardrail, mỗi chỗ thấy một thứ khác nhau</text>')
body.append(svgkit.legend(20, 66, [("blue", "guardrail: bước kiểm tra"), ("hatch", "vi phạm bị chặn"),
                                    ("tok", "khách, model, tool")]))

# Hàng chính: khách -> input guardrail -> model -> output guardrail -> màn hình
Y = 110
cols = [(20, 110, "tok", "Khách", "gửi câu hỏi"),
        (175, 170, "blue", "Input guardrail", "trước model"),
        (390, 170, "tok", "Model chính", "sinh câu trả lời"),
        (605, 170, "blue", "Output guardrail", "trước khi hiện"),
        (820, 130, "tok", "Màn hình", "khách đọc")]
for x, w, cls, top, sub in cols:
    body.append(node(x, Y, w, cls, top, sub))
for (x1, w1, *_), (x2, *_) in zip(cols, cols[1:]):
    body.append(arrow(x1 + w1 + 2, Y + BH / 2, x2 - 5, Y + BH / 2))

# Nhánh hành động: model -> tool guardrail -> hoàn tiền thật
AY = 300
MX = 390 + 85
body.append(arrow(MX, Y + BH + 2, MX, AY - 5))
body.append(f'<text class="sub halo" x="{MX + 8}" y="{(Y + BH + AY) / 2 + 4}">gọi tool issue_refund</text>')
body.append(node(390, AY, 170, "blue", "Kiểm trong code tool", "trước khi tool chạy"))
body.append(arrow(390 + 170 + 2, AY + BH / 2, 605 - 5, AY + BH / 2))
body.append(node(605, AY, 170, "tok", "Hoàn tiền thật", "tác dụng phụ"))

# Mỗi guardrail thấy gì
body.append(f'<text class="cap" x="175" y="{Y + BH + 26}">Chỉ thấy câu hỏi:</text>')
body.append(f'<text class="sub" x="175" y="{Y + BH + 44}">"{Q}"</text>')
body.append(f'<text class="sub" x="175" y="{Y + BH + 60}">hợp lệ, phải cho qua</text>')

body.append(f'<text class="cap" x="605" y="{Y + BH + 26}">Thấy đúng câu khách sẽ đọc:</text>')
body.append(f'<rect class="hatch" x="605" y="{Y + BH + 36}" width="230" height="28" rx="5"/>'
            f'<text class="cap halo" x="615" y="{Y + BH + 55}">"{BAD}"</text>')
body.append(f'<text class="sub" x="605" y="{Y + BH + 82}">bị chặn trước khi hiện ra</text>')

body.append(f'<text class="cap" x="120" y="{AY + 18}">Điều kiện là dữ liệu đơn hàng:</text>')
body.append(f'<text class="sub" x="120" y="{AY + 36}">ngày giao dự kiến, ngày giao thật,</text>')
body.append(f'<text class="sub" x="120" y="{AY + 52}">trạng thái thanh toán, trần số tiền</text>')
body.append(f'<text class="sub" x="120" y="{AY + 68}">kiểm bằng code, không bằng classifier</text>')

body.append(f'<text class="cap2" x="20" y="{H - 18}">Chính sách kinh doanh nằm ở câu trả lời, nên phải kiểm '
            'ở output; hành động có tác dụng phụ kiểm trong code của tool.</text>')

svgkit.write(__file__, "guardrail-positions.svg", svgkit.svg(
    W, H, "Input guardrail chỉ thấy câu hỏi nên phải cho qua câu hỏi hợp lệ về hoàn tiền. Output guardrail "
    "thấy đúng câu khách sẽ đọc nên chặn được câu hứa hoàn tiền 100% trước khi hiện. Hành động hoàn tiền "
    "được code của tool kiểm theo dữ liệu đơn hàng và trần số tiền, không dựa vào classifier", body))
