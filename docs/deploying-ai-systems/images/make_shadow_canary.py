"""Sinh shadow-canary.svg: SVG tĩnh, hai bước giữa eval offline và chuyển toàn bộ (§3.3).

Theo "Thứ tự an toàn cho một thay đổi" ở §3.3 (script đọc README để chắc còn khớp):
- Shadow: gateway gửi request cho bản cũ như thường, bản cũ trả lời người dùng;
  bản mới nhận bản sao, kết quả không trả cho người dùng, chỉ dùng để so độ trễ,
  lỗi, điểm judge. Tắt tool có tác dụng phụ ở bản shadow.
- Canary: 1%, rồi 5%, 25% traffic sang bản mới, chia theo user hoặc phiên; sau đó
  chuyển toàn bộ và giữ bản cũ chạy thêm để rollback không phải chờ khởi động.
Màu: xanh dương là bản cũ (đang có), cam là bản mới (phần mới).
Hình tĩnh vì đây là cấu trúc luồng traffic và các mức chia, không có diễn biến
liên tục cần thấy.

Chạy: python3 make_shadow_canary.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
STAGES = [1, 5, 25, 100]
assert "Canary: một phần nhỏ traffic (1%, rồi 5%, 25%) sang bản mới" in README
assert "kết quả không trả cho người dùng, chỉ dùng để so độ trễ, lỗi, điểm judge" in README
assert "tắt các tool có tác dụng phụ" in README
assert "Chia theo user hoặc phiên, không theo request" in README
assert "Chuyển toàn bộ, giữ bản cũ chạy thêm một thời gian để rollback không phải chờ khởi động" in README

W, H = 980, 440
BH = 46
body: list[str] = []


def node(x: float, y: float, w: float, cls: str, top: str, sub: str = "") -> str:
    out = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="7"/>'
    if sub:
        out += (f'<text class="cap mid" x="{x + w / 2}" y="{y + 20}">{top}</text>'
                f'<text class="sub mid" x="{x + w / 2}" y="{y + 37}">{sub}</text>')
    else:
        out += f'<text class="cap mid" x="{x + w / 2}" y="{y + BH / 2 + 5}">{top}</text>'
    return out


def arrow(x1, y1, x2, y2, cls="lfg", m="afg", extra="") -> str:
    return f'<path class="{cls} {extra}" d="M{x1},{y1}L{x2},{y2}" marker-end="url(#{m})"/>'


body.append('<text class="title" x="20" y="38">Shadow rồi canary: bản mới gặp traffic thật trước khi gặp mọi người</text>')
body.append(svgkit.legend(20, 70, [("blue", "bản cũ"), ("orange", "bản mới")]))

MID = 460
body.append(f'<line class="grid" x1="{MID}" y1="88" x2="{MID}" y2="{H - 20}"/>')

# ---------------- Shadow ----------------
LX = 20
body.append(f'<text class="row" x="{LX}" y="110">Bước 2: shadow</text>')
body.append(f'<text class="sub" x="{LX}" y="128">bản mới nhận bản sao traffic thật</text>')
UX, UW, UY = LX, 96, 170
GX, GW = LX + 160, 96
OX, OW = LX + 280, 140
OY, NY = 150, 260
body.append(node(UX, UY, UW, "tok", "Người dùng"))
body.append(node(GX, UY, GW, "tok", "Gateway"))
body.append(node(OX, OY, OW, "blue", "Bản cũ"))
body.append(node(OX, NY, OW, "orange", "Bản mới", "tắt tool tác dụng phụ"))
body.append(arrow(UX + UW + 2, UY + 16, GX - 4, UY + 16))
body.append(arrow(GX - 2, UY + 32, UX + UW + 6, UY + 32, extra="dash"))
body.append(f'<text class="sub halo mid" x="{(UX + UW + GX) / 2}" y="{UY + 52}">trả lời</text>')
body.append(arrow(GX + GW + 2, UY + 14, OX - 4, OY + BH / 2, "lblue", "ablue"))
body.append(arrow(GX + GW / 2, UY + BH + 2, OX - 4, NY + BH / 2, "lorange", "aorange", "dash"))
body.append(f'<text class="sub torange halo" x="{GX + 10}" y="{NY + 34}">bản sao</text>')
body.append(f'<text class="sub halo" x="{GX}" y="{NY + BH + 24}">kết quả của bản mới không trả cho người dùng,</text>')
body.append(f'<text class="sub halo" x="{GX}" y="{NY + BH + 40}">chỉ để so độ trễ, lỗi, điểm judge với bản cũ</text>')
assert OX + OW <= MID - 20
body.append(f'<text class="cap" x="{LX}" y="{H - 22}">Tốn gấp đôi chi phí model cho phần traffic được sao.</text>')

# ---------------- Canary ----------------
RX = MID + 30
body.append(f'<text class="row" x="{RX}" y="110">Bước 3, 4: canary rồi chuyển toàn bộ</text>')
body.append(f'<text class="sub" x="{RX}" y="128">chia theo user hoặc phiên, không theo request</text>')
BX, BW, RH = RX + 60, 330, 30
assert BX + BW <= W - 30   # lề phải ≥ 30 px
for i, pct in enumerate(STAGES):
    y = 150 + i * (RH + 18)
    body.append(f'<text class="cap end" x="{BX - 12}" y="{y + RH / 2 + 5}">{pct}%</text>')
    nw = max(BW * pct / 100, 3)
    old_w = BW - nw
    if pct < 100:
        body.append(f'<rect class="blue" x="{BX}" y="{y}" width="{old_w - 2:.1f}" height="{RH}" rx="4"/>')
        body.append(f'<rect class="orange" x="{BX + old_w:.1f}" y="{y}" width="{nw:.1f}" height="{RH}" rx="3"/>')
    else:
        body.append(f'<rect class="orange" x="{BX}" y="{y}" width="{BW}" height="{RH}" rx="4"/>')
yk = 150 + len(STAGES) * (RH + 18)
body.append(f'<rect class="blue" x="{BX}" y="{yk}" width="{BW}" height="{RH}" rx="4" style="stroke-dasharray:6 4;fill:none"/>'
            f'<text class="sub mid" x="{BX + BW / 2}" y="{yk + RH / 2 + 4}">bản cũ vẫn chạy, không nhận traffic</text>')
body.append(f'<text class="cap" x="{RX}" y="{H - 40}">Mỗi mức: so lỗi, TTFT, refusal, điểm judge giữa hai nhóm.</text>')
body.append(f'<text class="cap" x="{RX}" y="{H - 22}">Bản cũ còn chạy thì rollback không chờ khởi động.</text>')

ALT = ("Shadow: bản mới nhận bản sao traffic thật, kết quả không trả cho người dùng, chỉ để so độ trễ, lỗi, điểm "
       f"judge. Canary: {STAGES[0]}%, {STAGES[1]}%, {STAGES[2]}% rồi toàn bộ traffic sang bản mới, chia theo user "
       "hoặc phiên; bản cũ vẫn chạy để rollback ngay")
assert f"![{ALT}](images/shadow-canary.svg)" in README   # alt trong bài khớp số trên hình
svgkit.write(__file__, "shadow-canary.svg", svgkit.svg(W, H, ALT, body))
