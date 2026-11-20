"""Sinh github-mcp.svg: SVG tĩnh minh hoạ vụ GitHub MCP ở đầu bài (Invariant Labs, 05/2025 [15]).

Diễn biến theo hai đoạn mở đầu bài:
1. Kẻ tấn công mở một issue trong repo public, nội dung là chỉ dẫn viết cho AI.
2. Nạn nhân nhờ agent "xem các issue đang mở".
3. Agent đọc issue: nội dung do người lạ viết vào context (thuộc tính A).
4. Agent làm theo chỉ dẫn, đọc repo private (thuộc tính B).
5. Agent đăng thông tin đó vào một pull request ở repo public (thuộc tính C), ai cũng đọc được,
   kể cả kẻ tấn công (bước 6).
Không có lỗi code nào: lỗ hổng là tổ hợp A, B, C trong cùng một phiên (bảng "Ba thuộc tính").

Màu: đỏ cho nội dung của kẻ tấn công và chỗ dữ liệu lộ ra; xanh dương cho dữ liệu riêng.
Không dùng cam để không đặt cạnh đỏ. Hình không có số liệu.

Chạy: python3 make_github_mcp.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 900, 520
BW, BH = 220, 60
X1, X2, X3 = 30, 340, 650
Y1, Y2, Y3 = 110, 260, 410
assert X3 + BW <= W - 30
assert X1 + BW + 90 <= X2 and X2 + BW + 90 <= X3
CHAR = 7.6                      # ước lượng bề rộng một ký tự cỡ 14px (Verdana/DejaVu rộng hơn Helvetica)


def node(x: float, y: float, cls: str, name: str, sub: str) -> str:
    assert 8.5 * len(name) <= BW - 16 and 7 * len(sub) <= BW - 16, (name, sub)
    return (f'<rect class="{cls}" x="{x}" y="{y}" width="{BW}" height="{BH}" rx="7"/>'
            f'<text class="row mid halo" x="{x + BW / 2}" y="{y + 25}">{name}</text>'
            f'<text class="sub mid halo" x="{x + BW / 2}" y="{y + 45}">{sub}</text>')


def chip(x: float, y: float, text: str) -> str:
    return (f'<circle class="tok" cx="{x}" cy="{y}" r="13" style="stroke:var(--fg)"/>'
            f'<text class="lbl" style="font-size:13px;font-weight:600" x="{x}" y="{y + 5}">{text}</text>')


body: list[str] = []
body.append('<text class="title" x="20" y="38">GitHub MCP: một phiên agent có đủ A, B và C</text>')
body.append('<text class="cap2" x="20" y="62">Không có dòng code nào sai. Chỉ dẫn nằm trong issue, '
            'agent chạy với quyền của nạn nhân.</text>')

body.append(node(X1, Y1, "red", "Kẻ tấn công", "không cần quyền gì"))
body.append(node(X2, Y1, "red", "Issue, repo public", "chỉ dẫn viết cho AI"))
body.append(node(X1, Y2, "tok", "Nạn nhân", "“xem các issue đang mở”"))
body.append(node(X2, Y2, "tok", "Agent của nạn nhân", "quyền của nạn nhân"))
body.append(node(X3, Y2, "blue", "Repo private", "dữ liệu riêng"))
body.append(node(X2, Y3, "red", "PR, repo public", "ai cũng đọc được"))

cx = X2 + BW / 2
mid12 = (X1 + BW + X2) / 2
mid23 = (X2 + BW + X3) / 2
# 1. kẻ tấn công -> issue
body.append(f'<path class="lred" marker-end="url(#ared)" d="M{X1 + BW},{Y1 + BH / 2}H{X2 - 4}"/>')
body.append(f'<text class="cap mid halo" x="{mid12}" y="{Y1 + BH / 2 - 10}">1. mở issue</text>')
# 2. nạn nhân -> agent
body.append(f'<path class="lfg" marker-end="url(#afg)" d="M{X1 + BW},{Y2 + BH / 2}H{X2 - 4}"/>')
body.append(f'<text class="cap mid halo" x="{mid12}" y="{Y2 + BH / 2 - 10}">2. nhờ việc</text>')
# 3. issue -> agent (A)
ya = (Y1 + BH + Y2) / 2
body.append(f'<path class="lred" marker-end="url(#ared)" d="M{cx},{Y1 + BH}V{Y2 - 4}"/>')
body.append(chip(cx + 24, ya, "A"))
body.append(f'<text class="cap halo" x="{cx + 44}" y="{ya + 5}">3. agent đọc issue</text>')
# 4. repo private -> agent (B)
body.append(f'<path class="lblue" marker-end="url(#ablue)" d="M{X3},{Y2 + BH / 2}H{X2 + BW + 4}"/>')
body.append(chip(mid23, Y2 + BH / 2 - 26, "B"))
body.append(f'<text class="cap mid halo" x="{mid23}" y="{Y2 + BH / 2 + 22}">4. đọc</text>')
# 5. agent -> PR công khai (C)
yc = (Y2 + BH + Y3) / 2
body.append(f'<path class="lred" marker-end="url(#ared)" d="M{cx},{Y2 + BH}V{Y3 - 4}"/>')
body.append(chip(cx + 24, yc, "C"))
TXT5 = "5. tạo PR chứa dữ liệu riêng"
body.append(f'<text class="cap halo" x="{cx + 44}" y="{yc + 5}">{TXT5}</text>')
assert cx + 44 + CHAR * len(TXT5) <= W - 30
# 6. kẻ tấn công đọc PR: mũi tên nét đứt ra phía trái, nhãn nằm cuối mũi tên
TXT6 = "6. kẻ tấn công đọc"
body.append(f'<path class="lred dash" marker-end="url(#ared)" d="M{X2},{Y3 + BH / 2}H{X1 + BW - 40}"/>')
body.append(f'<text class="cap end halo" x="{X1 + BW - 48}" y="{Y3 + BH / 2 + 5}">{TXT6}</text>')
assert X1 + BW - 48 - CHAR * len(TXT6) >= 20

body.append(f'<text class="cap2" x="20" y="{H - 16}">A: nội dung không tin cậy. B: dữ liệu riêng. '
            'C: gửi ra ngoài. Thiếu một trong ba, chuỗi này đứt.</text>')

svgkit.write(__file__, "github-mcp.svg", svgkit.svg(
    W, H, "Kẻ tấn công mở issue chứa chỉ dẫn trong repo public. Nạn nhân nhờ agent xem các issue; "
    "agent đọc issue (A), làm theo chỉ dẫn đọc repo private (B), rồi đăng dữ liệu đó vào pull request "
    "công khai (C) mà kẻ tấn công đọc được. Không có lỗi code: lỗ hổng là ba thuộc tính trong cùng một phiên",
    body))
