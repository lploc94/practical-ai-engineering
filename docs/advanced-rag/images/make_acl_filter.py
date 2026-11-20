"""Sinh acl-filter.svg: SVG tĩnh minh hoạ filter quyền trong truy vấn và trong prompt (§3.3).

Ví dụ minh hoạ: retrieval tìm được ba tài liệu A, B và một tài liệu mật người dùng
không có quyền đọc.
- Hàng trên: search không filter, cả ba vào context, system prompt dặn "không dùng
  tài liệu mật". Model đã đọc tài liệu mật nên nội dung có thể lộ qua tóm tắt,
  diễn giải hoặc prompt injection.
- Hàng dưới: luồng một request của §3.3. Nhóm của người dùng lấy từ hệ thống định
  danh lúc request, search với filter nhóm (pre-filter), kiểm tra lại với hệ thống
  nguồn, chỉ tài liệu qua cả hai lớp mới vào reranker, cache và context.

Chạy: python3 make_acl_filter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 860, 420
BH = 64
R1, R2 = 140, 290
CW, CH = 54, 28

body: list[str] = []


def stage(x: float, y: float, w: float, title: str, sub: str = "") -> str:
    out = (f'<rect class="tok" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
           f'<text class="row mid" x="{x + w / 2}" y="{y + (27 if sub else 38)}">{title}</text>')
    if sub:
        out += f'<text class="sub mid" x="{x + w / 2}" y="{y + 47}">{sub}</text>'
    return out


def arrow(x1: float, x2: float, y: float) -> str:
    return f'<line class="lfg" x1="{x1 + 4}" y1="{y}" x2="{x2 - 6}" y2="{y}" marker-end="url(#afg)"/>'


def chips(x: float, y: float, names: list[str]) -> str:
    out = []
    for name in names:
        cls = "hatch" if name == "mật" else "blue"
        out.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{CW}" height="{CH}" rx="5"/>'
                   f'<text class="lbl halo" x="{x + CW / 2}" y="{y + CH / 2 + 5}">{name}</text>')
        x += CW + 8
    return "".join(out)


body.append('<text class="title" x="20" y="36">Filter quyền phải nằm trong truy vấn, không nằm trong prompt</text>')
body.append('<text class="cap2" x="20" y="58">Ví dụ: retrieval tìm được A, B và một tài liệu mật người dùng không được đọc.</text>')
body.append(svgkit.legend(20, 88, [("blue", "người dùng được đọc"), ("hatch", "không được đọc")]))

# Hàng 1: lọc trong prompt
body.append(f'<text class="row" x="20" y="{R1 - 12}">Lọc trong prompt</text>')
body.append(stage(20, R1, 150, "Search", "không filter"))
CX1, CW1 = 200, 336
body.append(f'<rect class="slot" x="{CX1}" y="{R1}" width="{CW1}" height="{BH}" rx="7"/>')
body.append(f'<text class="sub" x="{CX1 + 12}" y="{R1 + 18}">context</text>')
body.append(chips(CX1 + 12, R1 + 26, ["A", "B", "mật"]))
body.append(f'<text class="sub" x="{CX1 + 200}" y="{R1 + 36}">+ prompt “đừng</text>')
body.append(f'<text class="sub" x="{CX1 + 200}" y="{R1 + 52}">dùng tài liệu mật”</text>')
body.append(arrow(170, CX1, R1 + BH / 2))
body.append(stage(566, R1, 130, "Model", "đã đọc cả ba"))
body.append(arrow(CX1 + CW1, 566, R1 + BH / 2))
body.append(f'<text class="cap" x="20" y="{R1 + BH + 26}">Nội dung mật có thể lộ qua tóm tắt, diễn giải hoặc prompt injection.</text>')

# Hàng 2: lọc trong truy vấn, luồng một request của §3.3
body.append(f'<text class="row" x="20" y="{R2 - 12}">Lọc trong truy vấn</text>')
xs = [(20, 150, "Nhóm của user", "từ hệ thống định danh"),
      (200, 150, "Search", "filter theo nhóm"),
      (380, 150, "Kiểm tra lại", "với hệ thống nguồn")]
for i, (x, w, t, s) in enumerate(xs):
    body.append(stage(x, R2, w, t, s))
    if i:
        prev = xs[i - 1]
        body.append(arrow(prev[0] + prev[1], x, R2 + BH / 2))
CX2, CW2 = 560, 260
body.append(f'<rect class="slot" x="{CX2}" y="{R2}" width="{CW2}" height="{BH}" rx="7"/>')
body.append(f'<text class="sub" x="{CX2 + 12}" y="{R2 + 18}">reranker, cache, context</text>')
body.append(chips(CX2 + 12, R2 + 26, ["A", "B"]))
body.append(arrow(530, CX2, R2 + BH / 2))
body.append(f'<text class="cap" x="20" y="{R2 + BH + 26}">Tài liệu mật bị loại ở truy vấn hoặc ở bước kiểm tra lại, không tới reranker, cache hay context.</text>')

svgkit.write(__file__, "acl-filter.svg", svgkit.svg(
    W, H, "Lọc trong prompt: tài liệu mật vẫn vào context, model đã đọc nên nội dung có thể lộ. "
    "Lọc trong truy vấn: lấy nhóm của người dùng từ hệ thống định danh, search với filter nhóm, kiểm tra lại "
    "với hệ thống nguồn, nên chỉ tài liệu được phép mới tới reranker, cache và context", body))
