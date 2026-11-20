"""Sinh cache-layout.svg: SVG tĩnh minh hoạ thứ tự nội dung và prompt caching (§3.3).

Một request thứ hai trong cùng một phiên trên Claude, nơi cache khớp theo prefix
theo thứ tự tools → system → messages. Hai bố cục chứa cùng các khối nội dung:

- Hàng trên: dòng giờ `14:32:05` ở đầu system prompt. Giờ đổi mỗi request nên
  byte khác đầu tiên nằm ngay sau tools; chỉ dẫn, ví dụ, tài liệu tĩnh và dữ liệu
  phiên dù giống hệt lần trước vẫn phải tính lại.
- Hàng dưới: phần ổn định trước, ngày render một lần lúc đầu phiên; chỉ retrieval
  và câu hỏi là mới.

Độ rộng các khối chỉ để vừa nhãn, không tỷ lệ với số token.

Chạy: python3 make_cache_layout.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

GAP, BH, X0 = 6, 46, 150
# (nhãn, độ rộng, trạng thái): hit = đọc từ cache, miss = lẽ ra cache được nhưng phải
# tính lại, new = phần động mỗi request, clock = dòng giờ đổi mỗi request
BAD = [("tools", 64, "hit"), ("giờ 14:32:05", 112, "clock"), ("chỉ dẫn", 82, "miss"), ("ví dụ", 70, "miss"),
       ("tài liệu tĩnh", 112, "miss"), ("ngày, hồ sơ", 104, "miss"), ("retrieval", 92, "new"),
       ("câu hỏi", 82, "new")]
GOOD = [("tools", 64, "hit"), ("chỉ dẫn", 82, "hit"), ("ví dụ", 70, "hit"), ("tài liệu tĩnh", 112, "hit"),
        ("ngày, hồ sơ", 104, "hit"), ("retrieval", 92, "new"), ("câu hỏi", 82, "new")]

# Hai bố cục chứa cùng nội dung, chỉ khác dòng giờ và vị trí của nó
assert [b for b in BAD if b[2] != "clock"] == [(n, w, "miss" if s == "hit" and n != "tools" else s)
                                               for n, w, s in GOOD]
# Hàng trên: mọi khối giữa dòng giờ và phần động đều phải tính lại
first_diff = next(i for i, b in enumerate(BAD) if b[2] == "clock")
assert all(b[2] == "hit" for b in BAD[:first_diff])
assert all(b[2] in ("miss", "new") for b in BAD[first_diff + 1:])

CLS = {"hit": "blue", "miss": "hatch", "new": "tok", "clock": "tok"}
W = X0 + sum(w for _, w, _ in BAD) + GAP * (len(BAD) - 1) + 20
H = 384
R1, R2 = 140, 284

body: list[str] = []


def row(blocks: list[tuple[str, int, str]], y: float) -> tuple[list[str], dict[str, tuple[float, float]]]:
    out, pos = [], {}
    x = X0
    for name, w, state in blocks:
        out.append(f'<rect class="{CLS[state]}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>')
        out.append(f'<text class="cap mid halo" x="{x + w / 2}" y="{y + BH / 2 + 5}">{name}</text>')
        pos[name] = (x, x + w)
        x += w + GAP
    return out, pos


body.append('<text class="title" x="20" y="38">Một dòng giờ ở đầu prompt làm mất cache phía sau</text>')
body.append('<text class="cap2" x="20" y="62">Request thứ hai trong cùng một phiên, Claude: cache khớp theo prefix '
            'tools → system → messages.</text>')
body.append(svgkit.legend(20, 92, [("blue", "đọc từ cache"), ("hatch", "giống lần trước, vẫn phải tính lại"),
                                   ("tok", "phần mới của request")]))

# Hàng 1: giờ ở đầu system
items, pos = row(BAD, R1)
body.append(f'<text class="row" x="20" y="{R1 + 20}">Giờ ở đầu</text>')
body.append(f'<text class="sub" x="20" y="{R1 + 38}">system prompt</text>')
body += items
cx0, cx1 = pos["giờ 14:32:05"]
body.append(f'<text class="sub mid" x="{(cx0 + cx1) / 2}" y="{R1 - 10}">đổi mỗi request</text>')
body.append(f'<text class="cap" x="{X0}" y="{R1 + BH + 26}">Byte khác đầu tiên là dòng giờ: các khối sọc '
            'không đổi một chữ nhưng vẫn phải tính lại.</text>')

# Hàng 2: ổn định trước, động sau
items, pos = row(GOOD, R2)
body.append(f'<text class="row" x="20" y="{R2 + 20}">Ổn định trước</text>')
body.append(f'<text class="sub" x="20" y="{R2 + 38}">động sau</text>')
body += items
px0, px1 = pos["tools"][0], pos["ngày, hồ sơ"][1]
body.append(f'<path class="lblue" d="M{px0},{R2 - 6} V{R2 - 12} H{px1} V{R2 - 6}"/>')
body.append(f'<text class="sub mid" x="{(px0 + px1) / 2}" y="{R2 - 18}">prefix giống hệt request trước</text>')
body.append(f'<text class="cap" x="{X0}" y="{R2 + BH + 26}">Ngày render một lần lúc đầu phiên nên nằm trong prefix; '
            'chỉ retrieval và câu hỏi là mới.</text>')

svgkit.write(__file__, "cache-layout.svg", svgkit.svg(
    W, H, "Dòng giờ đổi mỗi request đặt ở đầu system prompt làm chỉ dẫn, ví dụ, tài liệu tĩnh và dữ liệu phiên phía "
    "sau phải tính lại dù không đổi; đặt phần ổn định trước, ngày render một lần đầu phiên, thì chỉ retrieval và "
    "câu hỏi là phần mới", body))
