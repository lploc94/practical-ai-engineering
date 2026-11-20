"""Sinh hedged-request.svg: SVG động, hedge theo chữ đầu tiên (§3.4).

Diễn biến theo đúng thứ tự thời gian của hàm `hedged` + `open_until_first_text`:
1. Giây 0: gửi bản chính. Nó kẹt trong hàng đợi, chưa có chữ đầu.
2. Giây HEDGE_AFTER (= p95 TTFT của loại request này): bản chính vẫn chưa có
   chữ đầu, budget còn, nên gửi bản thứ hai.
3. Hai bản cùng chờ.
4. Bản hai có chữ đầu trước: lấy bản này, stream cho người dùng, huỷ bản chính
   (cùng một sự kiện trong `hedged`).
5. So với không hedge (mốc giây 6) và tổng kết.
Số liệu giả định, chỉ để minh hoạ: p95 TTFT 2 s; bản hai có TTFT 1,2 s nên chữ đầu
ở giây 3,2; nếu không hedge, bản chính có chữ đầu ở giây 6.
Chờ tới p95 nên chỉ khoảng 5% request đi tới bước gửi bản hai [20].

Mỗi bước thêm một thay đổi và giữ lại (không có chú thích biến mất). Khung tĩnh
cho prefers-reduced-motion là trạng thái cuối.

Chạy: python3 make_hedged_request.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

HEDGE_AFTER = 2.0          # p95 TTFT
HEDGE_TTFT = 1.2
PRIMARY_FIRST = 6.0        # chữ đầu của bản chính nếu không hedge
WIN = HEDGE_AFTER + HEDGE_TTFT
assert WIN == 3.2 and WIN < PRIMARY_FIRST

W, H = 820, 400
X0, X1 = 190, 770          # trục 0 → TMAX giây
TMAX = 7
BH = 34
YP, YH = 130, 214          # hàng bản chính, hàng bản hai


def x(s: float) -> float:
    return X0 + s / TMAX * (X1 - X0)


def fmt(s: float) -> str:
    return f"{s:g}".replace(".", ",")


# Lịch animation (giây trong chu kỳ). Mỗi bước hiện trong FADE giây rồi giữ.
DUR, FADE_OUT, FADE = 22.0, 95, 0.5
STEPS = [0.5, 2.0, 5.5, 7.0, 12.5]               # bước 1..5
a = svgkit.Anim(dur=DUR, fade_out=FADE_OUT)


def at(k: int) -> str:
    t = STEPS[k - 1]
    return a([(t / DUR * 100, 0), ((t + FADE) / DUR * 100, 1)])


def bar(s0: float, s1: float, y: float, cls: str, label: str = "") -> str:
    out = f'<rect class="{cls}" x="{x(s0) + 1:.1f}" y="{y}" width="{x(s1) - x(s0) - 2:.1f}" height="{BH}" rx="5"/>'
    if label:
        out += f'<text class="lbl" style="font-size:12px" x="{(x(s0) + x(s1)) / 2:.1f}" y="{y + BH / 2 + 5}">{label}</text>'
    return out


# Mọi chữ mới xuất hiện ở mỗi bước. SVG lấy chữ từ đây nên assert tốc độ đọc phủ hết.
TXT = {1: {"wait": "chờ chữ đầu"},
       2: {"p95": f"giây {fmt(HEDGE_AFTER)} = p95 TTFT: chưa có chữ đầu, gửi bản 2"},
       3: {},
       4: {"stream": "stream cho người dùng", "cancel": "huỷ", "first": f"chữ đầu: giây {fmt(WIN)}"},
       5: {"nohedge": f"nếu không hedge: chữ đầu ở giây {fmt(PRIMARY_FIRST)}",
           "sum": "Chỉ khoảng 5% request chậm hơn p95 nên chỉ khoảng 5% request phải gửi bản 2."}}


def chars(k: int) -> int:
    return sum(len(v) for v in TXT[k].values())


# Kiểm nhịp theo docs/ILLUSTRATIONS.md
end_hold = DUR * FADE_OUT / 100 - (STEPS[-1] + FADE)
assert end_hold >= 3, end_hold                               # trạng thái cuối giữ ≥ 3 s
for k in range(1, len(STEPS)):
    gap = STEPS[k] - STEPS[k - 1]
    assert gap >= 1, gap                                     # mỗi bước ≥ 1 s
    assert chars(k) / gap <= 15, (k, chars(k) / gap)         # ≤ 15 ký tự/giây
assert chars(5) / end_hold <= 15, chars(5) / end_hold

body: list[str] = []
body.append('<text class="title" x="20" y="38">Hedge theo chữ đầu tiên: bản nào có chữ trước thì dùng</text>')
body.append(svgkit.legend(20, 70, [("blue", "chờ chữ đầu: hàng đợi, prefill"), ("orange", "chữ trả lời")]))

body.append(f'<text class="row" x="20" y="{YP + 15}">Bản chính</text>')
body.append(f'<text class="sub" x="20" y="{YP + 32}">gửi ở giây 0</text>')
body.append(f'<text class="row" x="20" y="{YH + 15}">Bản 2 (hedge)</text>')
body.append(f'<text class="sub" x="20" y="{YH + 32}">gửi khi quá p95</text>')
for y in (YP, YH):
    body.append(f'<rect class="slot" x="{X0}" y="{y}" width="{X1 - X0}" height="{BH}" rx="5"/>')

# Bước 1: gửi bản chính, chờ chữ đầu
body.append(f'<g {at(1)}>{bar(0, HEDGE_AFTER, YP, "blue", TXT[1]["wait"])}</g>')

# Bước 2: tới p95 mà chưa có chữ đầu: gửi bản 2
xp = x(HEDGE_AFTER)
body.append(f'<g {at(2)}><line class="lfg dash" x1="{xp:.1f}" y1="{YP - 22}" x2="{xp:.1f}" y2="{YH + BH + 8}"/>'
            f'<text class="sub halo" x="{xp + 6:.1f}" y="{YP - 10}">{TXT[2]["p95"]}</text></g>')

# Bước 3: hai bản cùng chờ
body.append(f'<g {at(3)}>{bar(HEDGE_AFTER, WIN, YP, "blue")}{bar(HEDGE_AFTER, WIN, YH, "blue")}</g>')

# Bước 4: bản 2 có chữ đầu trước: stream bản 2 và huỷ bản chính là cùng một sự kiện
xw = x(WIN)
body.append(f'<g {at(4)}>{bar(WIN, TMAX, YH, "orange", TXT[4]["stream"])}'
            f'<rect class="bg" x="{xw + 1:.1f}" y="{YP + 1}" width="{x(PRIMARY_FIRST) - xw - 2:.1f}" height="{BH - 2}"/>'
            f'<line class="lfg" x1="{xw:.1f}" y1="{YP - 4}" x2="{xw:.1f}" y2="{YH + BH + 8}"/>'
            f'<text class="lbl" style="font-size:12px" x="{xw + 26:.1f}" y="{YP + BH / 2 + 5}">{TXT[4]["cancel"]}</text>'
            f'<text class="sub halo" x="{xw + 6:.1f}" y="{YH + BH + 16}">{TXT[4]["first"]}</text></g>')

# Bước 5: so với không hedge (mốc giây 6) và tổng kết
xf = x(PRIMARY_FIRST)
body.append(f'<g {at(5)}><line class="lfg dash" x1="{xf:.1f}" y1="{YP - 4}" x2="{xf:.1f}" y2="{YP + BH + 4}"/>'
            f'<text class="sub halo end" x="{xf - 4:.1f}" y="{YP + BH + 16}">{TXT[5]["nohedge"]}</text>'
            f'<text class="cap" x="20" y="{H - 34}">{TXT[5]["sum"]}</text></g>')

# Trục thời gian
ay = YH + BH + 40
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for s in range(0, TMAX + 1):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')

body.append(f'<text class="cap2" x="20" y="{H - 10}">Giả định: p95 TTFT {fmt(HEDGE_AFTER)} s; bản 2 có TTFT '
            f'{fmt(HEDGE_TTFT)} s.</text>')

svgkit.write(__file__, "hedged-request.svg", svgkit.svg(
    W, H, f"Bản chính chưa có chữ đầu sau p95 TTFT {fmt(HEDGE_AFTER)} giây nên gửi bản thứ hai; bản hai có chữ đầu ở "
    f"giây {fmt(WIN)}, được stream cho người dùng, bản chính bị huỷ; không hedge thì chữ đầu ở giây "
    f"{fmt(PRIMARY_FIRST)}", body, a))
