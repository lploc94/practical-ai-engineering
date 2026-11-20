"""Sinh tail-sampling.svg: SVG động về thời điểm quyết định của tail sampling (§3.1).

Hai trace đi qua Collector với cấu hình ở §3.1: decision_wait 30 s, policy giữ
trace có app.llm.keep=true, còn lại lấy mẫu 5%. Span được export khi kết thúc, nên
Collector nhận span đầu tiên của trace khi span đó kết thúc; tail sampling processor
giữ trace trong bộ nhớ decision_wait kể từ lúc nhận nó rồi mới quyết định
(tailsamplingprocessor README [12]; span đến sau quyết định không được xét lại).
- Trace A bắt đầu ở giây 0: chat (0–5 s), execute_tool (5–9 s), chat (9–14 s) kết
  thúc với finish reason bị cắt (max_tokens) nên wrapper đặt app.llm.keep=true.
  Span đầu tới Collector ở giây 5, quyết định ở giây 35: giữ.
- Trace B bắt đầu ở giây 3: một span chat (3–10 s), kết thúc bình thường. Span tới
  ở giây 10, quyết định ở giây 40: không khớp policy nào và rơi ngoài mẫu 5%, nên bị
  bỏ. Vài giờ sau người dùng bấm thumbs down: quyết định đã xong, trace không còn.
- Head sampling quyết định ngay khi trace bắt đầu (ô thoi ở giây bắt đầu của mỗi
  hàng), lúc chưa có span nào nên không biết trace A sẽ bị cắt.
Thời lượng span là giả định, chỉ để minh hoạ thứ tự.

Animation đi theo thứ tự thời gian thật của các sự kiện (congruence): span hiện khi
tới Collector. Khung tĩnh cho prefers-reduced-motion là trạng thái cuối.

Chạy: python3 make_tail_sampling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

WAIT = 30                                   # decision_wait (s)
A_START, B_START = 0, 3
A_SPANS = [("chat", 0, 5), ("tool", 5, 9), ("chat", 9, 14)]
B_SPANS = [("chat", 3, 10)]
A_FIRST = min(e for _, _, e in A_SPANS)     # span đầu tiên tới Collector
B_FIRST = min(e for _, _, e in B_SPANS)
A_DEC, B_DEC = A_FIRST + WAIT, B_FIRST + WAIT
assert (A_DEC, B_DEC) == (35, 40)
assert all(e <= A_DEC for _, _, e in A_SPANS)   # mọi span của A tới trước quyết định

W, H = 820, 426
X0, XT = 190, 640                           # trục 0..TEND giây
TEND = 42
XF0, XF1 = 680, 800                         # vùng "vài giờ sau"
BH = 30
YA, YB = 140, 266


def x(s: float) -> float:
    return X0 + s / TEND * (XT - X0)


DUR = 28.5
a = svgkit.Anim(dur=DUR, fade_out=95)


def at(t: float) -> str:
    """Hiện ở giây t của chu kỳ (mờ dần 0,6 s), giữ tới cuối."""
    p0, p1 = t / DUR * 100, (t + 0.6) / DUR * 100
    return a([(p0, 0), (p1, 1)])


body: list[str] = []
body.append('<text class="title" x="20" y="38">Tail sampling quyết định sau decision_wait, feedback đến còn muộn hơn</text>')
body.append(svgkit.legend(20, 66, [("blue", "span đã tới Collector"), ("slot", "chờ quyết định"),
                                    ("hatch", "đến sau quyết định")]))


def row_title(y: float, name: str, sub: str) -> None:
    body.append(f'<text class="row" x="20" y="{y + 14}">{name}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 31}">{sub}</text>')


def head(y: float, start: float, t: float) -> None:
    body.append(f'<g {at(t)}><path class="ffg" d="M{x(start) - 6:.1f},{y - 18} l6,-6 l6,6 l-6,6z"/></g>')


def span(y: float, name: str, s: float, e: float, t: float, window: float | None = None) -> None:
    """Span hiện khi tới Collector; span đầu tiên mở cửa sổ decision_wait."""
    win = ""
    if window is not None:
        win = (f'<rect class="slot" x="{x(window):.1f}" y="{y - 6}" width="{x(window + WAIT) - x(window):.1f}" '
               f'height="{BH + 12}" rx="6"/>')
    body.append(f'<g {at(t)}>{win}<rect class="blue" x="{x(s) + 1:.1f}" y="{y}" width="{x(e) - x(s) - 2:.1f}" '
                f'height="{BH}" rx="4"/><text class="lbl" style="font-size:12px" x="{(x(s) + x(e)) / 2:.1f}" '
                f'y="{y + 19}">{name}</text></g>')


def decision(y: float, s: float, word: str, t: float, extra: str = "") -> None:
    d = x(s)
    body.append(f'<g {at(t)}><line class="lfg" x1="{d:.1f}" y1="{y - 12}" x2="{d:.1f}" y2="{y + BH + 12}" '
                f'style="stroke-width:3"/><text class="cap" x="{d + 8:.1f}" y="{y + 12}" style="font-weight:600">'
                f'{word}</text><text class="sub" x="{d + 8:.1f}" y="{y + 28}">giây {s}</text>{extra}</g>')


# Trục thời gian
ay = YB + BH + 56
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{XT}" y2="{ay}"/>')
for s in range(0, TEND + 1, 6):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')
body.append(f'<line class="axis dash" x1="{XT + 6}" y1="{ay}" x2="{XF0 - 6}" y2="{ay}"/>')
body.append(f'<line class="axis" x1="{XF0}" y1="{ay}" x2="{XF1}" y2="{ay}"/>')
body.append(f'<text class="sub mid" x="{(XF0 + XF1) / 2}" y="{ay + 18}">vài giờ sau</text>')

# Thứ tự hiện theo thời gian thật: head A (0 s), head B (3 s), span A1 (5 s), A2 (9 s),
# B1 (10 s), A3 (14 s), quyết định A (35 s), quyết định B (40 s), feedback (vài giờ sau)
row_title(YA, "Trace A", "có lượt bị cắt")
row_title(YB, "Trace B", "lượt bình thường")
head(YA, A_START, 1)
body.append(f'<g {at(1)}><text class="sub" x="{x(A_START) + 12:.1f}" y="{YA - 14}">'
            f'head sampling quyết định ở ô thoi, khi chưa có span nào</text></g>')
head(YB, B_START, 5)
span(YA, *A_SPANS[0], t=6.5, window=A_FIRST)
body.append(f'<g {at(6.5)}><text class="sub end" x="{x(A_DEC) - 8:.1f}" y="{YA + BH - 4}">decision_wait {WAIT} s</text></g>')
span(YA, *A_SPANS[1], t=8)
span(YB, *B_SPANS[0], t=9.5, window=B_FIRST)
span(YA, *A_SPANS[2], t=11)
body.append(f'<g {at(11)}><text class="sub" x="{x(A_SPANS[2][1]):.1f}" y="{YA + BH + 24}" '
            f'style="font-family:{svgkit.MONO}">max_tokens → app.llm.keep</text></g>')
decision(YA, A_DEC, "giữ", 13.5)
decision(YB, B_DEC, "bỏ", 15.5,
         f'<text class="sub end" x="{x(B_DEC) - 8:.1f}" y="{YB + BH + 24}">không có cờ, rơi ngoài mẫu 5%</text>')
body.append(f'<g {at(18.5)}><rect class="hatch" x="{XF0}" y="{YB}" width="{XF1 - XF0}" height="{BH}" rx="4"/>'
            f'<text class="cap halo mid" x="{(XF0 + XF1) / 2}" y="{YB + 20}" style="font-weight:600">thumbs down</text>'
            f'<text class="sub" x="{XF0}" y="{YB + BH + 18}">trace đã bị bỏ</text>'
            f'<text class="sub" x="{XF0}" y="{YB + BH + 34}">từ giây {B_DEC}</text></g>')

body.append(f'<text class="cap2" x="20" y="{H - 30}">Collector chờ decision_wait kể từ khi nhận span đầu tiên. Muốn đọc '
            'trace có feedback tiêu cực,</text>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">phải lưu ngắn hạn 100% request rồi chuyển bản ghi sang vùng lưu dài. '
            'Thời lượng span là giả định.</text>')

svgkit.write(__file__, "tail-sampling.svg", svgkit.svg(
    W, H, f"Tail sampling chờ decision_wait {WAIT} giây kể từ khi nhận span đầu tiên rồi mới quyết định: trace A có "
    f"lượt bị cắt mang app.llm.keep nên được giữ ở giây {A_DEC}; trace B bình thường rơi ngoài mẫu 5% nên bị bỏ ở "
    f"giây {B_DEC}. Feedback tiêu cực cho trace B đến vài giờ sau, khi trace đã bị bỏ. Head sampling quyết định ngay "
    "lúc trace bắt đầu, khi chưa biết trace sẽ bị cắt", body, a))
