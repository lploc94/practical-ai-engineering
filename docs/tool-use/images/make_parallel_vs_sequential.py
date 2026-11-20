"""Sinh parallel-vs-sequential.svg: SVG động so gọi tool song song với tuần tự (§1.4).

Hai hàng chạy trên cùng một đồng hồ, cột thẳng hàng nghĩa là cùng thời điểm:
- Song song: "so sánh đơn DH-208431 và DH-208432". Request 1 trả hai tool call,
  code chạy đồng thời, cả hai kết quả đi trong request 2 → 2 request tới model.
- Tuần tự: "tìm đơn gần nhất rồi huỷ". Kết quả list_recent_orders là input của
  cancel_order nên phải qua thêm một lượt → 3 request tới model.
Độ dài các khối chỉ minh hoạ thứ tự, không phải thời gian đo được.
Khung tĩnh cho prefers-reduced-motion là trạng thái cuối (cả hai hàng đầy đủ).

Chạy: python3 make_parallel_vs_sequential.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

LW, TW, GAP, BH = 110, 184, 14, 34   # rộng khối LLM, rộng khối tool, khe, cao
X0 = 130
# Năm cột thời gian: LLM, tool, LLM, tool, LLM
cols = []
x = X0
for w in (LW, TW, LW, TW, LW):
    cols.append((x, w))
    x += w + GAP
W, H = x - GAP + 24, 400

PAR_Y = 104                 # hàng song song: hai tool xếp chồng
PAR_MID = PAR_Y + BH + 4    # tâm dọc của hàng song song
SEQ_Y = 262
STEP0, STEP = 8, 11         # cột k xuất hiện ở STEP0 + k·STEP (%)
END = STEP0 + 5 * STEP      # sau cột cuối: hiện tổng kết

a = svgkit.Anim(dur=18, fade_out=95)   # giữ trạng thái cuối khoảng 6 giây
body: list[str] = []


def t(k: int) -> float:
    return STEP0 + k * STEP


def blk(col: int, y: float, cls: str, label: str, size: int = 13) -> str:
    bx, bw = cols[col]
    return (f'<rect class="{cls}" x="{bx}" y="{y}" width="{bw}" height="{BH}" rx="7"/>'
            f'<text class="lbl" style="font-size:{size}px" x="{bx + bw / 2}" y="{y + BH / 2 + 5}">{label}</text>')


def show(k: int, inner: str) -> str:
    return f"<g {a([(t(k), 0), (t(k) + 2, 1)])}>{inner}</g>"


body.append('<text class="title" x="20" y="38">Song song và tuần tự: bao nhiêu request tới model</text>')
body.append(svgkit.legend(W - 330, 38, [("orange", "request tới model"), ("tok", "tool chạy trong code")]))
body.append('<text class="cap2" x="20" y="62">Hai hàng chạy cùng một đồng hồ; khối thẳng cột là cùng thời điểm.</text>')

# Nhãn hàng
body.append(f'<text class="row" x="20" y="{PAR_MID - 4}">Song song</text>')
body.append(f'<text class="sub" x="20" y="{PAR_MID + 14}">so sánh 2 đơn</text>')
body.append(f'<text class="row" x="20" y="{SEQ_Y + 15}">Tuần tự</text>')
body.append(f'<text class="sub" x="20" y="{SEQ_Y + 33}">tìm rồi huỷ</text>')

# Ô trống cho mọi vị trí để thấy trước bố cục
for col, (bx, bw) in enumerate(cols):
    body.append(f'<rect class="slot" x="{bx}" y="{SEQ_Y}" width="{bw}" height="{BH}" rx="7"/>')
for col in (0, 2):
    bx, bw = cols[col]
    body.append(f'<rect class="slot" x="{bx}" y="{PAR_MID - BH / 2}" width="{bw}" height="{BH}" rx="7"/>')
bx, bw = cols[1]
for yy in (PAR_Y, PAR_Y + BH + 8):
    body.append(f'<rect class="slot" x="{bx}" y="{yy}" width="{bw}" height="{BH}" rx="7"/>')

# Hàng song song: request 1 → hai tool cùng lúc → request 2 (trả lời)
body.append(show(0, blk(0, PAR_MID - BH / 2, "orange", "request 1")))
body.append(show(1, blk(1, PAR_Y, "tok", "get_order DH-208431")
                 + blk(1, PAR_Y + BH + 8, "tok", "get_order DH-208432")))
body.append(show(2, blk(2, PAR_MID - BH / 2, "orange", "request 2")
                 + f'<text class="sub mid" x="{cols[2][0] + LW / 2}" y="{PAR_MID + BH / 2 + 16}">'
                 'gửi cả 2 kết quả</text>'))
# Hàng tuần tự: mỗi tool cần kết quả của bước trước
body.append(show(0, blk(0, SEQ_Y, "orange", "request 1")))
body.append(show(1, blk(1, SEQ_Y, "tok", "list_recent_orders")))
body.append(show(2, blk(2, SEQ_Y, "orange", "request 2")))
body.append(show(3, blk(3, SEQ_Y, "tok", "cancel_order DH-208431", 12)
                 + f'<text class="sub mid" x="{cols[3][0] + TW / 2}" y="{SEQ_Y - 10}">'
                 'mã đơn lấy từ kết quả trước</text>'))
body.append(show(4, blk(4, SEQ_Y, "orange", "request 3")))

# Tổng kết sau khi cả hai hàng xong
done = a([(END, 0), (END + 2, 1)])
body.append(f'<g {done}>'
            f'<text class="cap" x="{cols[2][0] + LW + GAP}" y="{PAR_MID + 5}">Xong sau 2 request tới model.</text>'
            f'<text class="cap" x="{X0}" y="{SEQ_Y + BH + 30}">Xong sau 3 request; '
            'mỗi request gửi lại toàn bộ lịch sử.</text></g>')

# Trục thời gian
ay = H - 40
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{W - 30}" y2="{ay}" marker-end="url(#afg)"/>')
body.append(f'<text class="sub end" x="{W - 30}" y="{ay + 18}">thời gian</text>')
body.append(f'<text class="cap2" x="20" y="{H - 10}">Độ dài khối chỉ minh hoạ thứ tự, không phải thời gian đo được.</text>')

svgkit.write(__file__, "parallel-vs-sequential.svg", svgkit.svg(
    W, H, "Gọi song song: hai get_order chạy cùng lúc, xong sau 2 request tới model. "
    "Gọi tuần tự: cancel_order cần mã đơn từ list_recent_orders, phải qua 3 request", body, a))
