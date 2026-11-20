"""Sinh doc-routing.svg: SVG tĩnh, cách chia tài liệu giữa đường parse và đường VLM (§3.2).

Các bước lấy từ danh sách "Cách chọn dựa trên dữ liệu thật" ở §3.2: phân loại theo có text
layer hay không, kiểm tra chất lượng text layer, chỉ gửi trang khó cho VLM, giữ số trang và
vị trí. Ví dụ báo cáo 200 trang có 15 trang bảng phức tạp cũng lấy từ §3.2.

Chạy: python3 make_doc_routing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

PAGES, HARD = 200, 15
EASY = PAGES - HARD
assert EASY == 185

W, H = 820, 480
body: list[str] = []
body.append('<text class="title" x="20" y="38">Parse trước hay đưa thẳng cho VLM: chia theo tài liệu và theo trang</text>')
body.append('<text class="cap2" x="20" y="62">Các bước kiểm tra rẻ chạy trước, VLM chỉ nhận phần parse không làm tốt.</text>')
body.append(svgkit.legend(20, 92, [("blue", "đường parse: dùng text layer"), ("orange", "đường VLM: gửi ảnh trang")]))


def node(x: float, y: float, w: float, h: float, cls: str, title: str, sub: str) -> str:
    return (svgkit.box(x, y, w, h, cls)
            + f'<text class="row mid" x="{x + w / 2}" y="{y + 27}">{title}</text>'
            + f'<text class="sub mid" x="{x + w / 2}" y="{y + 46}">{sub}</text>')


def arrow(x1: float, y1: float, x2: float, y2: float, c: str) -> str:
    return f'<line class="l{c}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#a{c})"/>'


R1, RH = 122, 62
R2 = 262
R3 = 372
# Hàng 1: tài liệu và ba câu hỏi
body.append(node(20, R1, 130, RH, "tok", "Tài liệu", "PDF, scan, ảnh chụp"))
body.append(node(186, R1, 170, RH, "tok", "Có text layer?", "theo tài liệu, vài ms"))
body.append(node(392, R1, 190, RH, "tok", "Text layer sạch?", "ký tự lạ, dấu, trang rỗng"))
body.append(node(618, R1, 182, RH, "tok", "Trang khó?", "theo trang: bảng phức tạp"))
cy = R1 + RH / 2
for x1, x2 in ((150, 186), (356, 392), (582, 618)):
    body.append(arrow(x1 + 2, cy, x2 - 4, cy, "fg"))
for xm in (371, 600):
    body.append(f'<text class="sub mid halo" x="{xm}" y="{cy - 9}">có</text>')

# Hàng 2: hai đường
body.append(node(186, R2, 396, RH, "orange", "Đưa thẳng cho VLM", "ảnh trang + schema"))
body.append(node(618, R2, 182, RH, "blue", "Parse trước", "Docling, Marker"))
yb = R1 + RH
body.append(arrow(271, yb + 2, 271, R2 - 4, "orange"))
body.append(f'<text class="sub halo end" x="263" y="{yb + 44}">không: scan, ảnh chụp</text>')
body.append(arrow(487, yb + 2, 487, R2 - 4, "orange"))
body.append(f'<text class="sub halo end" x="479" y="{yb + 44}">không: font nhúng hỏng</text>')
body.append(arrow(640, yb + 2, 560, R2 - 4, "orange"))
body.append(arrow(730, yb + 2, 730, R2 - 4, "blue"))
body.append(f'<text class="sub halo" x="606" y="{yb + 44}">có: {HARD} trang</text>')
body.append(f'<text class="sub halo" x="738" y="{yb + 36}">không:</text>')
body.append(f'<text class="sub halo" x="738" y="{yb + 52}">{EASY} trang</text>')

# Hàng 3: gộp lại
body.append(node(186, R3, 614, RH, "tok", "Trích xuất theo schema", "giữ số trang và vị trí để trích dẫn, để người kiểm tra mở đúng chỗ"))
body.append(arrow(384, R2 + RH + 2, 384, R3 - 4, "fg"))
body.append(arrow(709, R2 + RH + 2, 709, R3 - 4, "fg"))

body.append(f'<text class="cap2" x="20" y="{H - 14}">Ví dụ ở §3.2: báo cáo {PAGES} trang có {HARD} trang bảng phức tạp; '
            f'chỉ {HARD} trang đó gửi ảnh cho VLM.</text>')

svgkit.write(__file__, "doc-routing.svg", svgkit.svg(
    W, H, f"Tài liệu không có text layer hoặc text layer hỏng đi thẳng tới VLM; tài liệu có text layer sạch được xét "
    f"theo trang: báo cáo {PAGES} trang gửi {HARD} trang bảng phức tạp cho VLM, {EASY} trang còn lại parse thành text; "
    "mọi đường giữ số trang và vị trí", body))
