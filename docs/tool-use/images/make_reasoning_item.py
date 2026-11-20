"""Sinh reasoning-item.svg: SVG tĩnh minh hoạ lỗi thiếu reasoning item (§1.6).

Request thứ hai của một vòng lặp trên OpenAI Responses API với reasoning model,
dựng theo hai cách: gửi lại nguyên các item của response, hoặc dựng lại lịch sử
từ những field code cũ biết (role, content, tool_calls). Cách thứ hai làm rơi
reasoning item nằm trước function_call, và API trả 400 với thông báo trích ở §1.6.

Chạy: python3 make_reasoning_item.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

IW, GAP, BH = 182, 12, 40
X0 = 24
xs = [X0 + i * (IW + GAP) for i in range(4)]
W, H = xs[-1] + IW + 24, 392
R1, R2 = 132, 270          # y của hai hàng item

body: list[str] = []


def item(i: int, y: float, cls: str, label: str, sub: str = "") -> str:
    x = xs[i]
    ty = y + (BH / 2 + 5 if not sub else 17)
    out = (f'<rect class="{cls}" x="{x}" y="{y}" width="{IW}" height="{BH}" rx="7"/>'
           f'<text class="lbl" style="font-size:13px" x="{x + IW / 2}" y="{ty}">{label}</text>')
    if sub:
        out += f'<text class="sub mid" x="{x + IW / 2}" y="{y + 33}">{sub}</text>'
    return out


body.append('<text class="title" x="20" y="38">Request 2 của vòng lặp với reasoning model</text>')
body.append(svgkit.legend(W - 330, 38, [("blue", "item do API trả về"), ("hatch", "bị bỏ mất")]))
body.append('<text class="cap2" x="20" y="62">OpenAI Responses API. Reasoning item không đọc được, '
            'nhưng function_call đi sau nó bắt buộc phải có nó.</text>')

# Ngoặc đánh dấu các item thuộc response 1 (item 1..2)
bx0, bx1 = xs[1], xs[2] + IW
by = R1 - 12
body.append(f'<path class="lmuted" d="M{bx0},{by + 6} V{by} H{bx1} V{by + 6}"/>')
body.append(f'<text class="sub mid" x="{(bx0 + bx1) / 2}" y="{by - 6}">output của response 1</text>')

# Hàng 1: gửi lại nguyên văn
body.append(f'<text class="row" x="20" y="{R1 - 34}">Gửi lại nguyên các item của response</text>')
body.append(item(0, R1, "tok", "user"))
body.append(item(1, R1, "blue", "reasoning", "rs_… (mờ, có thể mã hoá)"))
body.append(item(2, R1, "blue", "function_call", "fc_…, call_id"))
body.append(item(3, R1, "tok", "function_call_output", "kết quả của code bạn"))
body.append(f'<text class="cap" x="20" y="{R1 + BH + 28}">→ Hợp lệ: model tiếp tục từ phần suy luận đã làm.</text>')

# Hàng 2: dựng lại từ field quen thuộc, reasoning rơi mất
body.append(f'<text class="row" x="20" y="{R2 - 16}">Dựng lại từ role, content, tool_calls</text>')
body.append(item(0, R2, "tok", "user"))
body.append(item(1, R2, "hatch", "", ""))
body.append(f'<text class="lbl halo tred" style="font-size:13px" x="{xs[1] + IW / 2}" y="{R2 + BH / 2 + 5}">'
            'thiếu reasoning</text>')
body.append(item(2, R2, "blue", "function_call", "fc_…, call_id"))
body.append(item(3, R2, "tok", "function_call_output", "kết quả của code bạn"))
body.append(f'<text class="cap" x="20" y="{R2 + BH + 28}">→ 400:</text>')
body.append(f'<text class="lbl tred" style="font-size:12px;text-anchor:start" x="72" y="{R2 + BH + 28}">'
            "Item 'fc_…' of type 'function_call' was provided without its required 'reasoning' item</text>")
body.append(f'<text class="cap2" x="20" y="{H - 12}">Claude (thinking block + signature) và Gemini 3+ '
            '(thought_signature) có quy tắc tương tự.</text>')

svgkit.write(__file__, "reasoning-item.svg", svgkit.svg(
    W, H, "Gửi lại nguyên các item thì request hợp lệ; dựng lại lịch sử từ role, content, tool_calls "
    "làm rơi reasoning item đứng trước function_call và API trả 400", body))
