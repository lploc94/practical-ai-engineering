"""Sinh field-order.svg: SVG tĩnh so hai cách đặt field đáp án và field suy luận (§1.5).

Model sinh token theo thứ tự và mỗi token chỉ dựa vào những gì đứng trước nó. Nếu
"answer" đứng đầu object, đáp án được chọn khi mới có prompt; nếu "reasoning" đứng
trước, đáp án được chọn sau khi đã có lập luận. Bài toán và lập luận trong hình là ví
dụ minh hoạ, không phải số liệu đo.

Chạy: python3 make_field_order.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402
from svgkit import box  # noqa: E402

W, H = 820, 340
BH, GAP = 36, 8
# (nhãn, độ rộng, class); class "orange" là token đáp án
ROWS = [
    ("Đáp án trước", [("prompt", 80, "tok"), ('{"answer":', 104, "tok"), ("12", 44, "orange"),
                      (', "reasoning":', 130, "tok"), ('"3 hộp × 4 quả…"', 170, "tok"), ("}", 32, "tok")]),
    ("Lập luận trước", [("prompt", 80, "tok"), ('{"reasoning":', 122, "tok"), ('"3 hộp × 4 quả…"', 170, "tok"),
                        (', "answer":', 104, "tok"), ("12", 44, "orange"), ("}", 32, "tok")]),
]
X0, Y0, DY = 150, 112, 106

body: list[str] = []
body.append('<text class="title" x="20" y="40">Thứ tự field quyết định đáp án dựa vào đâu</text>')
body.append('<text class="cap2" x="20" y="64">Model sinh từng token từ trái sang phải; mỗi token chỉ dựa vào '
            'các token đứng trước nó.</text>')
body.append(svgkit.legend(W - 330, 92, [("blue", "ngữ cảnh khi chọn đáp án"), ("orange", "đáp án")]))

for r, (label, toks) in enumerate(ROWS):
    y = Y0 + r * DY
    body.append(f'<text class="row" x="20" y="{y + 23}">{label}</text>')
    x = X0
    xs = []
    for text, w, cls in toks:
        xs.append(x)
        x += w + GAP
    assert x - GAP <= W - 20, "hàng token tràn khung"
    ans = next(i for i, (_, _, c) in enumerate(toks) if c == "orange")
    # Khung bao các token đứng trước đáp án: phần model nhìn thấy khi chọn đáp án
    right = xs[ans] - GAP / 2
    body.append(f'<rect class="blue" style="fill-opacity:.5" x="{X0 - 6}" y="{y - 6}" '
                f'width="{right - X0 + 6 - 2}" height="{BH + 12}" rx="10"/>')
    for (text, w, cls), xi in zip(toks, xs):
        body.append(box(xi, y, w, BH, cls, text.replace('"', "&quot;")))
    note = ("đáp án được chọn khi mới có prompt; lập luận sinh sau không đổi được đáp án"
            if r == 0 else "đáp án được chọn sau khi lập luận đã nằm trong ngữ cảnh")
    body.append(f'<text class="cap" x="{X0}" y="{y + BH + 30}">{note}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 20}">Thứ tự key thực tế tuỳ provider (§2.1). '
            'Reasoning model có thinking nằm ngoài vùng bị ràng buộc nên không cần cách này.</text>')

svgkit.write(__file__, "field-order.svg", svgkit.svg(
    W, H, "Khi answer đứng đầu object, model chọn đáp án lúc mới chỉ có prompt; khi reasoning đứng trước, "
    "đáp án được chọn sau khi lập luận đã nằm trong ngữ cảnh", body))
