"""Sinh proxy-buffering.svg: SVG động so nginx bật và tắt proxy_buffering (§3.2).

Hai hàng chạy cùng một đồng hồ. Origin gửi 5 event SSE lần lượt.
- proxy_buffering on (mặc định): nginx giữ event trong buffer; với stream nhỏ như
  SSE, trình duyệt nhận cả khối khi upstream kết thúc (chữ hiện cùng lúc ở cuối).
- proxy_buffering off hoặc header X-Accel-Buffering: no: nginx chuyển mỗi event
  ngay khi nhận được.
Số event và khoảng cách giữa chúng chỉ minh hoạ thứ tự, không phải thời gian đo.
Khung tĩnh cho prefers-reduced-motion: hàng trên vẫn hiện 5 event bị giữ trong
buffer cùng 5 event đến trình duyệt ở cuối, kèm nhãn giải thích.

Chạy: python3 make_proxy_buffering.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

WORDS = ["Hợp", "đồng", "gồm", "ba", "phần"]
N = len(WORDS)
SW, SG, SH = 36, 6, 30                  # rộng ô event, khe, cao
BOXW, BOXH = N * SW + (N - 1) * SG + 16, 70
BX = [20, 300, 580]                      # origin, nginx, trình duyệt
assert BX[2] + BOXW <= 810
W, H = 820, 436
ROW_Y = {"on": 114, "off": 278}

STEP0, STEP = 8, 9                       # event k gửi ở STEP0 + k·STEP (%)
END = STEP0 + N * STEP                   # upstream kết thúc
a = svgkit.Anim(dur=20, fade_out=95)     # giữ trạng thái cuối khoảng 5 giây
body: list[str] = []


def t(k: int) -> float:
    return STEP0 + k * STEP


def slot_x(box: int, k: int) -> float:
    return BX[box] + 8 + k * (SW + SG)


def ev(box: int, k: int, y: float, cls: str) -> str:
    sx = slot_x(box, k)
    return (f'<rect class="{cls}" x="{sx}" y="{y}" width="{SW}" height="{SH}" rx="5"/>'
            f'<text class="lbl halo" style="font-size:12px" x="{sx + SW / 2}" y="{y + SH / 2 + 4}">{WORDS[k]}</text>')


def frame(y: float, nginx_note: str) -> None:
    names = ["Origin (backend)", "nginx", "Trình duyệt"]
    for i, name in enumerate(names):
        body.append(f'<rect class="tok" x="{BX[i]}" y="{y}" width="{BOXW}" height="{BOXH}" rx="8"/>')
        body.append(f'<text class="sub" x="{BX[i] + 8}" y="{y + 17}">{name}</text>')
        for k in range(N):
            body.append(f'<rect class="slot" x="{slot_x(i, k)}" y="{y + 30}" width="{SW}" height="{SH}" rx="5"/>')
    body.append(f'<text class="sub end" x="{BX[1] + BOXW - 8}" y="{y + 17}">{nginx_note}</text>')
    for i in (0, 1):
        x1, x2 = BX[i] + BOXW + 6, BX[i + 1] - 8
        body.append(f'<line class="lfg" x1="{x1}" y1="{y + 45}" x2="{x2}" y2="{y + 45}" marker-end="url(#afg)"/>')


body.append('<text class="title" x="20" y="38">nginx giữ event SSE trong buffer cho tới khi upstream kết thúc</text>')
body.append(svgkit.legend(20, 66, [("blue", "event đã gửi hoặc đã nhận"), ("hatch", "bị giữ trong buffer")]))

# Hàng 1: buffering bật
yo = ROW_Y["on"]
body.append(f'<text class="row" x="20" y="{yo - 6}" style="font-family:{svgkit.MONO}">proxy_buffering on</text>')
body.append(f'<text class="sub" x="210" y="{yo - 6}">(mặc định)</text>')
frame(yo, "buffer")
for k in range(N):
    body.append(f'<g {a([(t(k), 0), (t(k) + 1.5, 1)])}>{ev(0, k, yo + 30, "blue")}</g>')
    body.append(f'<g {a([(t(k) + 1.5, 0), (t(k) + 3, 1), (END + 2, 1), (END + 3.5, 0)], static=True)}>'
                f'{ev(1, k, yo + 30, "hatch")}</g>')
# Upstream kết thúc: cả 5 event đến trình duyệt cùng lúc
body.append(f'<g {a([(END + 2, 0), (END + 3.5, 1)])}>'
            + "".join(ev(2, k, yo + 30, "blue") for k in range(N)) + "</g>")
body.append(f'<g {a([(t(0) + 1.5, 0), (t(0) + 3, 1), (END, 1), (END + 1.5, 0)], static=False)}>'
            f'<text class="cap2" x="{BX[2]}" y="{yo + BOXH + 22}">Trình duyệt chưa nhận gì.</text></g>')
body.append(f'<g {a([(END + 2, 0), (END + 3.5, 1)])}>'
            f'<text class="cap" x="{BX[2]}" y="{yo + BOXH + 22}">Nhận cả 5 event cùng lúc,</text>'
            f'<text class="cap" x="{BX[2]}" y="{yo + BOXH + 41}">khi upstream kết thúc.</text></g>')

body.append(f'<text class="static-only sub" x="{BX[1] + 8}" y="{yo + BOXH + 18}">giữ cả 5 event tới lúc cuối</text>')

# Hàng 2: buffering tắt
yf = ROW_Y["off"]
body.append(f'<text class="row" x="20" y="{yf - 6}" style="font-family:{svgkit.MONO}">proxy_buffering off</text>')
body.append(f'<text class="sub" x="220" y="{yf - 6}">hoặc X-Accel-Buffering: no</text>')
frame(yf, "chuyển ngay")
for k in range(N):
    body.append(f'<g {a([(t(k), 0), (t(k) + 1.5, 1)])}>{ev(0, k, yf + 30, "blue")}</g>')
    body.append(f'<g {a([(t(k) + 1.5, 0), (t(k) + 3, 1)])}>{ev(2, k, yf + 30, "blue")}</g>')
body.append(f'<g {a([(t(0) + 1.5, 0), (t(0) + 3, 1)])}>'
            f'<text class="cap" x="{BX[2]}" y="{yf + BOXH + 22}">Mỗi event tới ngay sau khi</text>'
            f'<text class="cap" x="{BX[2]}" y="{yf + BOXH + 41}">origin gửi.</text></g>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Số event và nhịp gửi chỉ để minh hoạ thứ tự, '
            'không phải thời gian đo được.</text>')

svgkit.write(__file__, "proxy-buffering.svg", svgkit.svg(
    W, H, "Khi proxy_buffering bật, nginx giữ 5 event SSE trong buffer và trình duyệt nhận cả 5 cùng lúc "
    "khi upstream kết thúc; khi tắt buffering hoặc gửi X-Accel-Buffering: no, mỗi event tới trình duyệt "
    "ngay sau khi origin gửi", body, a))
