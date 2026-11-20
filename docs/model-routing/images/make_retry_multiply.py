"""Sinh retry-multiply.svg: SVG tĩnh về retry chồng retry (§1.3, Tra cứu nhanh mục 4).

Số lấy từ bài:
- Ba tầng (ứng dụng, gateway, SDK), mỗi tầng ba lần thử: một lần bấm thành
  3 * 3 * 3 = 27 request lên provider đang quá tải.
- Một tầng làm chủ: SDK max_retries=0, gateway không retry, chuỗi fallback của
  ứng dụng có trần max_attempts=4 với attempts_per_target=2 (ví dụ ở §3.1), nên
  tối đa 4 request: 2 lần ở model chính, 2 lần ở model fallback.
Hình tĩnh vì đây là phép nhân giữa các tầng, không có diễn biến cần thấy.
Script đọc README để chắc các số ví dụ ở §3.1 còn khớp.

Chạy: python3 make_retry_multiply.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
PER_LAYER = 3
LAYERS = ["Ứng dụng", "Gateway", "SDK"]
TOTAL = PER_LAYER ** len(LAYERS)
assert TOTAL == 27 and "Ba tầng mỗi tầng ba lần thử là 27 request" in README
assert f"ba tầng, mỗi tầng ba lần thử, thì một request lỗi có thể thành {TOTAL} request" in README   # §1.3
MAX_ATTEMPTS, PER_TARGET = 4, 2
assert f"max_attempts={MAX_ATTEMPTS}" in README and f"attempts_per_target={PER_TARGET}" in README
assert "`max_retries=0`" in README

W, H = 980, 400
BH = 34
body: list[str] = []
body.append('<text class="title" x="20" y="38">Retry chồng retry: ba tầng nhân với nhau</text>')
body.append(svgkit.legend(20, 66, [("tok", "một lần thử ở tầng đó"), ("hatch", "request lên provider đang quá tải"),
                                    ("blue", "request lên provider, có trần")]))

# ---------------- Trái: ba tầng ----------------
LX, CELL, GAP = 120, 16, 2
LW = TOTAL * CELL
YS = [130, 180, 230]
body.append(f'<text class="row" x="20" y="108">Mỗi tầng tự thử 3 lần</text>')
count = 1
for li, name in enumerate(LAYERS):
    n = PER_LAYER ** (li + 1)
    w = LW / n
    y = YS[li]
    body.append(f'<text class="cap" x="20" y="{y + 15}">{name}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 31}">{count} × {PER_LAYER} = {n}</text>')
    count = n
    cls = "hatch" if li == len(LAYERS) - 1 else "tok"
    for k in range(n):
        body.append(f'<rect class="{cls}" x="{LX + k * w + GAP / 2:.1f}" y="{y}" width="{w - GAP:.1f}" '
                    f'height="{BH}" rx="3"/>')
body.append(f'<text class="cap" x="{LX}" y="{YS[-1] + BH + 26}" style="font-weight:600">'
            f'{PER_LAYER} × {PER_LAYER} × {PER_LAYER} = {TOTAL} request cho một lần bấm</text>')
body.append(f'<text class="sub" x="{LX}" y="{YS[-1] + BH + 44}">nhân thêm với số người dùng đang bấm "thử lại"</text>')

# ---------------- Phải: một tầng ----------------
MID = LX + LW + 34
body.append(f'<line class="grid" x1="{MID}" y1="90" x2="{MID}" y2="{H - 70}"/>')
RX = MID + 26
RCELL = 80
body.append(f'<text class="row" x="{RX}" y="108">Một tầng làm chủ</text>')
labels = ["chính"] * PER_TARGET + ["fallback"] * (MAX_ATTEMPTS - PER_TARGET)
for i, lab in enumerate(labels):
    x = RX + i * (RCELL + GAP)
    cx = x + RCELL / 2
    body.append(f'<rect class="tok" x="{x}" y="{YS[0]}" width="{RCELL}" height="{BH}" rx="3"/>'
                f'<text class="sub mid" x="{cx}" y="{YS[0] + BH / 2 + 4}">{lab}</text>')
    body.append(f'<rect class="slot" x="{x}" y="{YS[1]}" width="{RCELL}" height="{BH}" rx="3"/>'
                f'<text class="sub mid" x="{cx}" y="{YS[1] + BH / 2 + 4}">1 lần</text>')
    body.append(f'<rect class="blue" x="{x}" y="{YS[2]}" width="{RCELL}" height="{BH}" rx="3"/>')
    for ya, yb in ((YS[0], YS[1]), (YS[1], YS[2])):
        body.append(f'<path class="lfg" d="M{cx},{ya + BH + 2}L{cx},{yb - 4}" marker-end="url(#afg)"/>')
body.append(f'<text class="cap" x="{RX}" y="{YS[-1] + BH + 26}" style="font-weight:600">'
            f'tối đa {MAX_ATTEMPTS} request</text>')
body.append(f'<text class="sub" x="{RX}" y="{YS[-1] + BH + 44}">SDK đặt max_retries=0</text>')

body.append(f'<text class="cap2" x="20" y="{H - 34}">SDK, gateway và ứng dụng đều có thể retry. Chọn một tầng làm chủ, '
            'tắt retry ở các tầng còn lại,</text>')
body.append(f'<text class="cap2" x="20" y="{H - 16}">và đặt trần cho cả chuỗi: tổng số lần thử và tổng thời gian.</text>')

assert MID + 26 + MAX_ATTEMPTS * (RCELL + GAP) <= W - 30   # lề phải ≥ 30 px

svgkit.write(__file__, "retry-multiply.svg", svgkit.svg(
    W, H, f"Ứng dụng, gateway và SDK mỗi tầng thử {PER_LAYER} lần thì một lần bấm thành {TOTAL} request lên "
    f"provider đang quá tải. Chỉ một tầng retry, SDK đặt max_retries=0 và chuỗi fallback có trần "
    f"max_attempts={MAX_ATTEMPTS}, thì tối đa {MAX_ATTEMPTS} request", body))
