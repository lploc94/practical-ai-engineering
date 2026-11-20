"""Sinh patch-tiers.svg: SVG tĩnh minh hoạ patch 28 px và việc thu nhỏ ảnh (§1.1, số liệu ở §1.2).

Số liệu lấy từ bảng ví dụ của docs Claude ở §1.2 [4]: ảnh 1920×1080 trên tier standard
(cạnh dài ≤ 1.568 px, tối đa 1.568 token) bị thu về 1456×819, còn 1.560 token; trên tier
high-res (Claude 4.7 trở lên, ≤ 2.576 px, tối đa 4.784 token) giữ nguyên, 2.691 token.
Hàm claude_image_tokens chép nguyên từ §1.2; các con số được assert với hàm đó.

Lưới vẽ đúng tỉ lệ: mỗi ô là một patch 28×28 px của ảnh mà model thực sự nhìn thấy.

Chạy: python3 make_patch_tiers.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402


def claude_image_tokens(w: int, h: int, high_res: bool) -> int:
    """Chép từ §1.2 của README."""
    max_edge, max_tokens = (2576, 4784) if high_res else (1568, 1568)

    def tokens(scale: float) -> int:
        return math.ceil(math.floor(w * scale) / 28) * math.ceil(math.floor(h * scale) / 28)

    scale = min(1.0, max_edge / max(w, h))
    if tokens(scale) > max_tokens:
        lo, hi = 0.0, scale
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tokens(mid) <= max_tokens else (lo, mid)
        scale = lo
    return tokens(scale)


P = 28
W0, H0 = 1920, 1080
STD_W, STD_H = 1456, 819          # kích thước sau khi thu nhỏ, theo bảng docs
std_cols, std_rows = math.ceil(STD_W / P), math.ceil(STD_H / P)
hr_cols, hr_rows = math.ceil(W0 / P), math.ceil(H0 / P)
std_tok, hr_tok = std_cols * std_rows, hr_cols * hr_rows
assert (std_cols, std_rows, std_tok) == (52, 30, 1560)
assert (hr_cols, hr_rows, hr_tok) == (69, 39, 2691)
assert claude_image_tokens(W0, H0, high_res=False) == std_tok
assert claude_image_tokens(W0, H0, high_res=True) == hr_tok
assert max(W0, H0) <= 2576            # high-res không phải thu nhỏ ảnh này

S = 0.185                              # px của hình cho mỗi px ảnh
W, H = 820, 430
TOP = 130
LX, RX = 20, 405


def fmt(v: int) -> str:
    return f"{v:,}".replace(",", ".")


def grid(x: float, y: float, cols: int, rows: int, img_w: int, img_h: int) -> str:
    """Ảnh (khung xanh) phủ lưới patch; ô cuối mỗi hàng/cột có thể lấn ra ngoài ảnh."""
    gw, gh = cols * P * S, rows * P * S
    out = [f'<rect class="blue" x="{x}" y="{y}" width="{img_w * S:.1f}" height="{img_h * S:.1f}" rx="2"/>']
    lines = []
    for c in range(1, cols):
        xx = x + c * P * S
        lines.append(f"M{xx:.2f},{y}V{y + gh:.2f}")
    for r in range(1, rows):
        yy = y + r * P * S
        lines.append(f"M{x},{yy:.2f}H{x + gw:.2f}")
    out.append(f'<path d="{"".join(lines)}" style="stroke:var(--blue);stroke-width:.6;opacity:.55;fill:none"/>')
    return "".join(out)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Cùng ảnh 1920×1080, hai tier của Claude</text>')
body.append('<text class="cap2" x="20" y="62">Mỗi ô là một patch 28×28 px của ảnh model thực sự nhìn thấy, '
            'một ô là một token.</text>')
body.append(svgkit.legend(20, 92, [("blue", "ảnh đưa vào encoder"), ("slot", "khung ảnh gốc")]))

# Tier standard: khung gốc nét đứt, ảnh đã thu nhỏ ở góc trên trái
body.append(f'<text class="row" x="{LX}" y="{TOP - 10}">Standard: ≤ 1.568 px, tối đa 1.568 token</text>')
body.append(f'<rect class="slot" x="{LX}" y="{TOP}" width="{W0 * S:.1f}" height="{H0 * S:.1f}" rx="2"/>')
body.append(grid(LX, TOP, std_cols, std_rows, STD_W, STD_H))
body.append(f'<text class="sub end" x="{LX + W0 * S - 6:.1f}" y="{TOP + H0 * S - 8:.1f}">gốc 1920×1080</text>')

# Tier high-res: giữ nguyên ảnh
body.append(f'<text class="row" x="{RX}" y="{TOP - 10}">High-res (4.7+): ≤ 2.576 px, tối đa 4.784 token</text>')
body.append(grid(RX, TOP, hr_cols, hr_rows, W0, H0))

BY = TOP + H0 * S + 30
body.append(f'<text class="cap" x="{LX}" y="{BY}">Thu về {STD_W}×{STD_H}: {std_cols} × {std_rows} = '
            f'<tspan style="font-weight:600">{fmt(std_tok)} token</tspan></text>')
body.append(f'<text class="cap" x="{RX}" y="{BY}">Giữ nguyên: {hr_cols} × {hr_rows} = '
            f'<tspan style="font-weight:600">{fmt(hr_tok)} token</tspan></text>')
body.append(f'<text class="cap2" x="{LX}" y="{BY + 22}">Ít token hơn, chữ nhỏ có thể khó đọc hơn</text>')
ratio = round(hr_tok / std_tok, 1)
assert ratio == 1.7
body.append(f'<text class="cap2" x="{RX}" y="{BY + 22}">Gấp {str(ratio).replace(".", ",")} lần token '
            'với cùng ảnh</text>')

body.append(f'<text class="cap2" x="20" y="{H - 16}">Giới hạn px áp cho cạnh dài; token = ⌈w/28⌉ × ⌈h/28⌉ sau khi thu nhỏ. Số liệu: docs Claude.</text>')

svgkit.write(__file__, "patch-tiers.svg", svgkit.svg(
    W, H, f"Ảnh 1920×1080 trên Claude: tier standard thu ảnh về {STD_W}×{STD_H} rồi chia {std_cols} × {std_rows} "
    f"patch 28 px, tức {fmt(std_tok)} token; tier high-res giữ nguyên ảnh, {hr_cols} × {hr_rows} = {fmt(hr_tok)} token",
    body))
