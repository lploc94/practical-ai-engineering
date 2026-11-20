"""Sinh in-batch-negatives.svg: hình tĩnh minh họa in-batch negatives và false negative do trùng lặp (§3.9).

Batch giả định B = 4 cặp (query, tài liệu đúng). Cặp 2 và cặp 4 có cùng một tài liệu đúng,
nên hai ô ngoài đường chéo của chúng là false negative.

Chạy: python3 make_in_batch_negatives.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

QUERIES = ["đổi mật khẩu?", "hủy gói cước?", "giờ mở cửa?", "ngừng gói cước?"]
DOCS = ["HD đặt lại mật khẩu", "HD hủy gói cước", "Giờ làm việc", "HD hủy gói cước"]
B = len(QUERIES)
assert len(DOCS) == B and B - 1 == 3


def false_negative(i: int, j: int) -> bool:
    return i != j and DOCS[i] == DOCS[j]


fn_cells = [(i, j) for i in range(B) for j in range(B) if false_negative(i, j)]
assert fn_cells == [(1, 3), (3, 1)]

CX0, CY0, CW, CH, GAP = 220, 128, 128, 46, 6
W = CX0 + B * (CW + GAP) + 30
H = CY0 + B * (CH + GAP) + 92

body: list[str] = []
body.append('<text class="title" x="20" y="38">In-batch negatives: tài liệu đúng của query khác làm negative</text>')
body.append(svgkit.legend(20, 70, [("blue", "positive: kéo lại gần"), ("tok", "negative: đẩy ra xa"),
                                   ("hatch", "false negative: đẩy nhầm cặp liên quan")]))

for j, d in enumerate(DOCS):
    cx = CX0 + j * (CW + GAP) + CW / 2
    body.append(f'<text class="row mid" x="{cx}" y="{CY0 - 26}">d{j + 1}</text>')
    dup = next((k for k in range(j) if DOCS[k] == d), None)
    note = f" (= d{dup + 1})" if dup is not None else ""
    body.append(f'<text class="sub mid" x="{cx}" y="{CY0 - 9}">{d}{note}</text>')
for i, q in enumerate(QUERIES):
    y = CY0 + i * (CH + GAP)
    body.append(f'<text class="row" x="20" y="{y + 20}">q{i + 1}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 37}">{q}</text>')
    for j in range(B):
        x = CX0 + j * (CW + GAP)
        if i == j:
            cls, label = "blue", "kéo gần"
        elif false_negative(i, j):
            cls, label = "hatch", "đẩy xa (sai)"
        else:
            cls, label = "tok", "đẩy xa"
        body.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{CW}" height="{CH}" rx="6"/>')
        body.append(f'<text class="cap mid{" halo" if cls == "hatch" else ""}" x="{x + CW / 2}" y="{y + CH / 2 + 5}">'
                    f'{label}</text>')

y = CY0 + B * (CH + GAP) + 26
body.append(f'<text class="cap" x="20" y="{y}">Mỗi hàng là bài phân loại {B} lựa chọn; '
            f'mỗi query có B − 1 = {B - 1} negative mà không cần gán nhãn.</text>')
body.append(f'<text class="cap2" x="20" y="{y + 24}">q2 và q4 trùng tài liệu đúng: hai ô sọc dạy model '
            'đẩy xa một cặp thực sự liên quan.</text>')

svgkit.write(__file__, "in-batch-negatives.svg", svgkit.svg(
    W, H, "Ma trận 4 query nhân 4 tài liệu trong một batch: đường chéo là cặp đúng được kéo lại gần, "
    "các ô còn lại là negative bị đẩy ra xa; hai query có cùng tài liệu đúng tạo ra hai ô false negative",
    body))
