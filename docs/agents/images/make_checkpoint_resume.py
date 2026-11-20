"""Sinh checkpoint-resume.svg: SVG động, lưu checkpoint trước hay sau khi chạy tool (§3.2).

Kịch bản theo các gạch đầu dòng sau code ở §3.2: model trả một tool call refund,
process chết ngay sau khi refund đã chạy xong ở hệ thống thanh toán nhưng trước khi
kết quả tool được lưu. Khoá idempotency là f"{run_id}:{tool_use_id}" như trong code.
- Hàng trên, lưu sau khi nhận tool call, trước khi chạy tool (cách của §3.2):
  checkpoint có toolu_1 chưa có kết quả, resume chạy lại đúng toolu_1 với cùng khoá;
  phía nhận khử trùng theo khoá nên trả lại kết quả cũ, chỉ hoàn tiền một lần.
- Hàng dưới, chỉ lưu sau khi chạy tool: checkpoint chưa có toolu_1, resume gọi lại
  model, model sinh toolu_2 với khoá mới, hệ thống thanh toán hoàn tiền lần hai.
Id rút gọn (id thật dạng toolu_01...), run id "r42" là giả định.
Hai hàng chạy trên cùng đồng hồ để thấy crash xảy ra ở cùng thời điểm.
Khung tĩnh cho prefers-reduced-motion là trạng thái cuối.

Chạy: python3 make_checkpoint_resume.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

DUR = 30.0
a = svgkit.Anim(dur=DUR, fade_out=95)
STEP0, STEP, NCOL = 1.0, 3.0, 6
T_END = STEP0 + NCOL * STEP                      # hiện kết luận sau cột cuối
HOLD = DUR * a.fade_out / 100 - T_END
assert STEP >= 1.0 and HOLD >= 3.0, HOLD


def pct(t: float) -> float:
    return round(t / DUR * 100, 2)


def show(t: float, inner: str) -> str:
    return f'<g {a([(pct(t), 0), (pct(t + 0.4), 1)])}>{inner}</g>'


BW, GAP, BH = 142, 12, 54
X0 = 176
xs = [X0 + i * (BW + GAP) for i in range(NCOL)]
W = xs[-1] + BW + 24
RA, RB = 132, 280
H = RB + BH + 128
CRASH = 3

body: list[str] = []
body.append('<text class="title" x="20" y="38">Lưu checkpoint trước hay sau khi chạy tool</text>')
body.append(svgkit.legend(W - 380, 38, [("blue", "lấy từ checkpoint"), ("hatch", "tác dụng phụ lặp lại")]))
body.append('<text class="cap2" x="20" y="64">Process chết sau khi refund đã hoàn tiền, trước khi lưu kết quả tool. '
            'Khoá = run id + tool_use id.</text>')


def cell(i: int, y: float, cls: str, top: str, code: str = "") -> str:
    x = xs[i]
    out = f'<rect class="{cls}" x="{x}" y="{y}" width="{BW}" height="{BH}" rx="7"/>'
    if code:
        out += (f'<text class="cap mid halo" x="{x + BW / 2}" y="{y + 22}">{top}</text>'
                f'<text class="lbl halo" style="font-size:12px" x="{x + BW / 2}" y="{y + 42}">{code}</text>')
    else:
        out += f'<text class="cap mid halo" x="{x + BW / 2}" y="{y + BH / 2 + 5}">{top}</text>'
    return out


# Nhãn hàng
for y, name, sub in ((RA, "Lưu trước tool", "như vòng lặp §3.2"), (RB, "Lưu sau tool", "chỉ lưu khi có kết quả")):
    body.append(f'<text class="row" x="20" y="{y + 24}">{name}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 42}">{sub}</text>')
    for i in range(NCOL):
        if i != CRASH:
            body.append(f'<rect class="slot" x="{xs[i]}" y="{y}" width="{BW}" height="{BH}" rx="7"/>')

# Các cột, mỗi cột hiện cùng lúc ở hai hàng
cols = [
    (cell(0, RA, "tok", "model trả", "tool_use toolu_1"),
     cell(0, RB, "tok", "model trả", "tool_use toolu_1")),
    (cell(1, RA, "blue", "lưu checkpoint", "có toolu_1"),
     f'<text class="sub mid" x="{xs[1] + BW / 2}" y="{RB + BH / 2 + 5}">chưa lưu</text>'),
    (cell(2, RA, "tok", "refund: hoàn tiền", "r42:toolu_1"),
     cell(2, RB, "tok", "refund: hoàn tiền", "r42:toolu_1")),
    None,                                   # cột crash, vẽ riêng
    (cell(4, RA, "blue", "resume: chạy lại", "toolu_1"),
     cell(4, RB, "tok", "resume: gọi model", "tool_use toolu_2")),
    (cell(5, RA, "blue", "refund: khoá cũ", "r42:toolu_1"),
     cell(5, RB, "hatch", "refund lần 2", "r42:toolu_2")),
]
for i, pair in enumerate(cols):
    t = STEP0 + i * STEP
    if pair is None:
        cx = xs[i] + BW / 2
        body.append(show(t, f'<line class="lred dash" x1="{cx}" y1="{RA - 8}" x2="{cx}" y2="{RB + BH + 8}"/>'
                            f'<text class="cap mid tred" x="{cx}" y="{RA - 16}">process chết</text>'))
        continue
    body.append(show(t, pair[0] + pair[1]))

# Kết luận mỗi hàng, hiện sau cột cuối
body.append(show(T_END,
                 f'<text class="cap" x="{X0}" y="{RA + BH + 28}">Cùng khoá: phía nhận trả lại kết quả cũ. '
                 'Hoàn tiền 1 lần.</text>'
                 f'<text class="cap" x="{X0}" y="{RB + BH + 28}">Khoá mới: phía nhận coi là yêu cầu mới. '
                 'Hoàn tiền 2 lần.</text>'))

# Trục thời gian và chú thích tĩnh
ay = H - 50
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{W - 24}" y2="{ay}" marker-end="url(#afg)"/>')
body.append(f'<text class="sub end" x="{W - 24}" y="{ay + 18}">thời gian</text>')
body.append(f'<text class="cap2" x="20" y="{H - 14}">Hàng trên chỉ an toàn khi phía nhận khử trùng theo khoá. '
            'Id rút gọn, run id giả định.</text>')

svgkit.write(__file__, "checkpoint-resume.svg", svgkit.svg(
    W, H, "Process chết sau khi refund đã chạy. Lưu checkpoint trước khi chạy tool: resume chạy lại đúng "
    "toolu_1 với cùng khoá, phía nhận trả kết quả cũ, hoàn tiền 1 lần. Chỉ lưu sau khi chạy tool: "
    "resume gọi lại model, toolu_2 mang khoá mới, hoàn tiền 2 lần", body, a))
