"""Sinh orchestrator-workers.svg: SVG động cho hệ thống research nhiều agent (§2.2).

Theo bài của Anthropic [10] và §1.4 [3]:
1. Lead agent đọc câu hỏi, lập kế hoạch.
2. Sinh các sub-agent chạy song song, mỗi sub-agent nhận một brief.
3. Mỗi sub-agent tìm kiếm trong context riêng; context của nó dài ra theo từng
   tool call. Ba sub-agent xuất hiện và dài ra cùng lúc vì chúng chạy song song.
4. Chỉ bản tóm tắt (thường 1.000–2.000 token theo [3]) quay về context của lead.
5. Một agent riêng gắn trích dẫn ở cuối, rồi trả báo cáo.
Số sub-agent (3), số khối context và độ dài khối chỉ minh hoạ, không phải số đo.
Khung tĩnh cho prefers-reduced-motion là trạng thái cuối.

Chạy: python3 make_orchestrator_workers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

DUR = 22.0
a = svgkit.Anim(dur=DUR, fade_out=95)

# Mốc thời gian (giây). Mỗi pha một thay đổi; chú thích mỗi pha ≤ ~15 ký tự/giây.
T_LEAD, T_SPAWN, T_GROW, T_SUM, T_CITE = 0.5, 4.0, (7.5, 8.7, 9.9), 11.5, 15.0
GROW_STEP = T_GROW[1] - T_GROW[0]
assert GROW_STEP >= 1.0                      # mỗi phần tử đổi trạng thái tối đa 1 lần/giây
HOLD = DUR * a.fade_out / 100 - (T_CITE + 0.4)
assert HOLD >= 3.0, HOLD                     # giữ trạng thái cuối ít nhất 3 giây


def pct(t: float) -> float:
    return round(t / DUR * 100, 2)


def on(t: float) -> str:
    """Hiện từ giây t tới hết chu kỳ."""
    return a([(pct(t), 0), (pct(t + 0.4), 1)])


def window(t0: float, t1: float) -> str:
    """Hiện trong [t0, t1), ẩn trong khung tĩnh."""
    return a([(pct(t0), 0), (pct(t0 + 0.3), 1), (pct(t1 - 0.3), 1), (pct(t1), 0)])


CAPS = [  # (bắt đầu, kết thúc, chú thích)
    (T_LEAD, T_SPAWN, "Lead agent đọc câu hỏi, lập kế hoạch."),
    (T_SPAWN, T_GROW[0], "Sinh 3 sub-agent chạy song song, mỗi cái một brief."),
    (T_GROW[0], T_SUM, "Mỗi sub-agent tìm kiếm trong context riêng."),
    (T_SUM, T_CITE, "Chỉ bản tóm tắt quay về context của lead."),
]
for t0, t1, text in CAPS:
    assert len(text) / (t1 - t0) <= 15, (text, len(text) / (t1 - t0))
FINAL_CAP = "Lead chỉ giữ bản tóm tắt; agent trích dẫn gắn nguồn rồi trả báo cáo."
assert len(FINAL_CAP) / HOLD <= 15

W, H = 980, 560
BH = 40
body: list[str] = []
body.append('<text class="title" x="20" y="38">Orchestrator–worker: song song, mỗi sub-agent một context</text>')
body.append(svgkit.legend(W - 360, 66, [("orange", "context sub-agent"), ("blue", "bản tóm tắt gửi về")]))

# ---------- Lead agent ----------
LX, LY, LW, LH = 30, 90, 650, 96
body.append(f'<g {on(T_LEAD)}>'
            f'<rect class="tok" x="{LX}" y="{LY}" width="{LW}" height="{LH}" rx="9"/>'
            f'<text class="row" x="{LX + 14}" y="{LY + 24}">Lead agent</text>'
            f'<text class="sub" x="{LX + 110}" y="{LY + 24}">context của lead</text></g>')
SLOT_Y, SLOT_H = LY + 40, BH
plan_w = 170
body.append(f'<g {on(T_LEAD)}><rect class="slot" x="{LX + 14}" y="{SLOT_Y}" width="{plan_w}" height="{SLOT_H}" rx="6"/>'
            f'<text class="cap mid" x="{LX + 14 + plan_w / 2}" y="{SLOT_Y + 25}">câu hỏi + kế hoạch</text></g>')
SUM_W, SUM_GAP = 120, 20
sum_x = [LX + 14 + plan_w + SUM_GAP + i * (SUM_W + SUM_GAP) for i in range(3)]
assert sum_x[-1] + SUM_W <= LX + LW - 10

# ---------- Agent trích dẫn và báo cáo ----------
CX, CW = LX + LW + 34, 130
RX, RW = CX + CW + 30, 86
cite_y = LY + (LH - 56) / 2
body.append(f'<g {on(T_CITE)}>'
            f'<path class="lfg" d="M{LX + LW + 4},{LY + LH / 2}L{CX - 6},{LY + LH / 2}" marker-end="url(#afg)"/>'
            f'<rect class="tok" x="{CX}" y="{cite_y}" width="{CW}" height="56" rx="7"/>'
            f'<text class="cap mid" x="{CX + CW / 2}" y="{cite_y + 24}">Agent trích dẫn</text>'
            f'<text class="sub mid" x="{CX + CW / 2}" y="{cite_y + 42}">gắn nguồn</text>'
            f'<path class="lfg" d="M{CX + CW + 4},{LY + LH / 2}L{RX - 6},{LY + LH / 2}" marker-end="url(#afg)"/>'
            f'<rect class="tok" x="{RX}" y="{cite_y}" width="{RW}" height="56" rx="7"/>'
            f'<text class="cap mid" x="{RX + RW / 2}" y="{cite_y + 33}">Báo cáo</text></g>')

# ---------- Sub-agent ----------
SW = SUM_W
SX = sum_x            # mỗi sub-agent nằm ngay dưới ô tóm tắt của nó
SY = 250
names = ["A", "B", "C"]
work = [["brief", "tìm kiếm", "đọc trang", "tìm kiếm"],
        ["brief", "tìm kiếm", "đọc trang", "đọc trang"],
        ["brief", "tìm kiếm", "tìm kiếm", "đọc trang"]]
spawn, grow = [], [[] for _ in T_GROW]
for i, x in enumerate(SX):
    cx = x + SW / 2
    # Mũi tên giao việc từ lead xuống, ở nửa trái; tóm tắt quay về ở nửa phải
    spawn.append(f'<path class="lfg" d="M{x + 32},{LY + LH + 4}L{x + 32},{SY - 6}" marker-end="url(#afg)"/>'
                 f'<rect class="tok" x="{x}" y="{SY}" width="{SW}" height="34" rx="7"/>'
                 f'<text class="cap mid" x="{cx}" y="{SY + 22}">Sub-agent {names[i]}</text>'
                 f'<rect class="slot" x="{x}" y="{SY + 44}" width="{SW}" height="{4 * (BH - 8) + 3 * 6 + 16}" rx="7"/>')
    for k, label in enumerate(work[i]):
        by = SY + 52 + k * (BH - 8 + 6)
        blk = (f'<rect class="orange" x="{x + 8}" y="{by}" width="{SW - 16}" height="{BH - 8}" rx="5"/>'
               f'<text class="sub mid" style="fill:var(--fg)" x="{cx}" y="{by + 21}">{label}</text>')
        if k == 0:
            spawn.append(blk)
        else:
            grow[k - 1].append(blk)
body.append(f'<g {on(T_SPAWN)}>{"".join(spawn)}</g>')
for t, blocks in zip(T_GROW, grow):
    body.append(f'<g {on(t)}>{"".join(blocks)}</g>')
ctx_bottom = SY + 44 + 4 * (BH - 8) + 3 * 6 + 16
body.append(f'<g {on(T_GROW[0])}><text class="sub" x="{LX + 14}" y="{SY + 80}">mỗi tool result</text>'
            f'<text class="sub" x="{LX + 14}" y="{SY + 96}">nằm lại trong context</text>'
            f'<text class="sub" x="{LX + 14}" y="{SY + 112}">của sub-agent đó</text></g>')

# ---------- Tóm tắt quay về lead ----------
ret = []
for i, x in enumerate(SX):
    sx = x + SW - 32
    ret.append(f'<rect class="blue" x="{sum_x[i]}" y="{SLOT_Y}" width="{SUM_W}" height="{SLOT_H}" rx="6"/>'
               f'<text class="cap mid" x="{sum_x[i] + SUM_W / 2}" y="{SLOT_Y + 25}">tóm tắt {names[i]}</text>'
               f'<path class="lblue dash" d="M{sx},{SY - 4}L{sx},{SLOT_Y + SLOT_H + 6}" '
               f'marker-end="url(#ablue)"/>')
body.append(f'<g {on(T_SUM)}>{"".join(ret)}'
            f'<text class="sub" x="{LX + 14}" y="{SLOT_Y + SLOT_H + 42}">mỗi bản tóm tắt</text>'
            f'<text class="sub" x="{LX + 14}" y="{SLOT_Y + SLOT_H + 58}">thường 1.000–2.000 token</text></g>')

# ---------- Chú thích theo pha ----------
CY = ctx_bottom + 44
for t0, t1, text in CAPS:
    body.append(f'<g {window(t0, t1)}><text class="cap" x="30" y="{CY}">{text}</text></g>')
body.append(f'<g {on(T_CITE)}><text class="cap" x="30" y="{CY}">{FINAL_CAP}</text></g>')
body.append(f'<text class="cap2" x="30" y="{H - 16}">Số sub-agent và độ dài khối chỉ minh hoạ, không phải số đo.</text>')
assert CY < H - 36

svgkit.write(__file__, "orchestrator-workers.svg", svgkit.svg(
    W, H, "Lead agent lập kế hoạch rồi sinh các sub-agent chạy song song; mỗi sub-agent tìm kiếm trong "
    "context riêng và chỉ gửi bản tóm tắt, thường 1.000 đến 2.000 token, về context của lead; "
    "một agent riêng gắn trích dẫn trước khi trả báo cáo", body, a))
