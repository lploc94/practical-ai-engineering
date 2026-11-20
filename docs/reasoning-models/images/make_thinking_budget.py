"""Sinh thinking-budget.svg: SVG động minh hoạ token suy luận đi trước câu trả lời (§1.2).

Output của reasoning model sinh tuần tự: token suy luận trước, câu trả lời sau, và
cả hai cùng tính vào output token lẫn giới hạn max_tokens.

Số liệu:
- Hàng trên dùng ví dụ trong docs Claude [8] mà §1.2 trích: 312 thinking token trên
  tổng 348 output token, tức 36 token câu trả lời và gần 90% output là suy luận.
- Hàng dưới là giả định: cùng request, cùng lượng suy luận, nhưng max_tokens = 256.
  Trần thấp hơn phần suy luận nên response dừng với stop_reason "max_tokens" trước
  khi có chữ nào của câu trả lời (§1.4, §3.5).

Chạy: python3 make_thinking_budget.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

THINK, TOTAL = 312, 348
ANSWER = TOTAL - THINK
CAP = 256                       # giả định cho hàng dưới
assert ANSWER == 36
assert round(THINK / TOTAL * 100) == 90  # "gần 90%" trong §1.2
assert CAP < THINK

N_THINK = 8                     # chia phần suy luận thành 8 đoạn để hiện dần
CH = THINK / N_THINK            # 39 token mỗi đoạn
N_ANS = 2
CA = ANSWER / N_ANS             # 18 token mỗi đoạn

X0, SC = 190, 1.45              # x của token 0, px mỗi token
AXMAX = 400
W, H = int(X0 + AXMAX * SC + 40), 372
RA, RB, BH = 112, 236, 40       # y của hai hàng

DUR = 16
STEP = 0.8                      # mỗi đoạn mới hiện sau 0,8 giây
T0 = 1.0


def pct(t: float) -> float:
    return round(t / DUR * 100, 2)


def x(tok: float) -> float:
    return X0 + tok * SC


a = svgkit.Anim(dur=DUR, fade_out=95)   # trạng thái cuối giữ từ ~10 s đến 15,2 s
body: list[str] = []
body.append("<style>.green{fill:var(--green-bg);stroke:var(--green);stroke-width:2}</style>")

body.append('<text class="title" x="20" y="38">Suy luận sinh trước, câu trả lời sinh sau</text>')
body.append('<text class="cap2" x="20" y="62">Cả hai là output token: cùng tính tiền, '
            'cùng tính vào max_tokens.</text>')
body.append(svgkit.legend(W - 300, 38, [("orange", "token suy luận"), ("green", "câu trả lời")]))

# Trục token: lưới mờ, nhãn phía dưới
for t in range(0, AXMAX + 1, 100):
    body.append(f'<line class="grid" x1="{x(t)}" y1="{RA - 14}" x2="{x(t)}" y2="{RB + BH + 10}"/>')
    body.append(f'<text class="sub mid" x="{x(t)}" y="{RB + BH + 28}">{t}</text>')
body.append(f'<text class="sub" x="{X0}" y="{RB + BH + 48}">output token, sinh lần lượt theo thời gian →</text>')

# Nhãn hàng
body.append(f'<text class="row" x="20" y="{RA + 18}">max_tokens đủ lớn</text>')
body.append(f'<text class="sub" x="20" y="{RA + 36}">ví dụ trong docs Claude</text>')
body.append(f'<text class="row" x="20" y="{RB + 18}">max_tokens = {CAP}</text>')
body.append(f'<text class="sub" x="20" y="{RB + 36}">giả định</text>')


def seg(t0: float, t1: float, y: float, cls: str, when: float) -> str:
    w = (t1 - t0) * SC - 2      # khe 2px giữa các đoạn
    return (f'<g {a([(pct(when), 0), (pct(when + 0.3), 1)])}><rect class="{cls}" x="{x(t0)}" y="{y}" '
            f'width="{w}" height="{BH}" rx="4"/></g>')


# Phần suy luận: hai hàng chạy cùng lúc vì là cùng một request
cut_time = None
for k in range(N_THINK):
    s, e = k * CH, (k + 1) * CH
    when = T0 + k * STEP
    body.append(seg(s, e, RA, "orange", when))
    if s < CAP:
        body.append(seg(s, min(e, CAP), RB, "orange", when))
        if e >= CAP:
            cut_time = when
assert cut_time is not None

# Câu trả lời chỉ bắt đầu sau khi suy luận xong
ans_start = T0 + N_THINK * STEP
for k in range(N_ANS):
    body.append(seg(THINK + k * CA, THINK + (k + 1) * CA, RA, "green", ans_start + k * STEP))

# Hàng trên: mốc chữ đầu tiên của câu trả lời
mark = a([(pct(ans_start), 0), (pct(ans_start + 0.4), 1)])
body.append(f'<g {mark}><line class="lfg dash" x1="{x(THINK)}" y1="{RA - 10}" x2="{x(THINK)}" y2="{RA + BH + 6}"/>'
            f'<text class="sub end" x="{x(THINK) - 6}" y="{RA - 16}">chữ đầu tiên của câu trả lời</text></g>')
sum_a = a([(pct(ans_start + N_ANS * STEP + 0.6), 0), (pct(ans_start + N_ANS * STEP + 1.0), 1)])
body.append(f'<g {sum_a}><text class="cap" x="{X0}" y="{RA + BH + 24}">{THINK} suy luận + {ANSWER} trả lời '
            f'= {TOTAL} output token; gần 90% là suy luận</text></g>')

# Hàng dưới: trần max_tokens, dừng giữa lúc suy luận
body.append(f'<line class="lfg" style="stroke-width:2.5" x1="{x(CAP)}" y1="{RB - 14}" x2="{x(CAP)}" '
            f'y2="{RB + BH + 8}"/>')
body.append(f'<text class="sub mid" x="{x(CAP)}" y="{RB - 20}">trần max_tokens</text>')
stop = cut_time + STEP
empty = a([(pct(stop), 0), (pct(stop + 0.4), 1)])
body.append(f'<g {empty}><rect class="slot" x="{x(THINK)}" y="{RB}" width="{ANSWER * SC - 2}" height="{BH}" rx="4"/>'
            f'<text class="sub" x="{x(THINK) + ANSWER * SC + 8}" y="{RB + 17}">câu trả lời:</text>'
            f'<text class="sub" x="{x(THINK) + ANSWER * SC + 8}" y="{RB + 33}">không có</text></g>')
sum_b = a([(pct(stop + 0.8), 0), (pct(stop + 1.2), 1)])
body.append(f'<g {sum_b}><text class="cap" x="{X0}" y="{RB + BH + 70}">stop_reason: "max_tokens" '
            f'khi mới suy luận được {CAP} token; {CAP} token đó vẫn tính tiền</text></g>')

assert pct(ans_start + N_ANS * STEP + 1.0) < 70, "trạng thái cuối phải giữ ít nhất vài giây"

svgkit.write(__file__, "thinking-budget.svg", svgkit.svg(
    W, H, f"Ví dụ trong docs Claude: {THINK} token suy luận sinh trước, {ANSWER} token câu trả lời sinh sau, "
    f"gần 90% trong {TOTAL} output token là suy luận. Với max_tokens = {CAP} (giả định), "
    "request dừng với stop_reason max_tokens giữa lúc suy luận và không có chữ nào của câu trả lời", body, a))
