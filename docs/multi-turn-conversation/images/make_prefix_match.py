"""Sinh prefix-match.svg: SVG động minh họa vì sao cache chỉ khớp theo prefix (§1.2).

Chạy: python3 make_prefix_match.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402
from svgkit import box  # noqa: E402

N, DIFF = 8, 3  # 8 token, request B khác A ở vị trí 3
X0, STEP, BW, BH = 170, 78, 64, 36
A_Y, B_Y, KV_Y = 74, 140, 226
W, H = X0 + N * STEP + 20, 350

SCAN0, SCAN_STEP = 8, 6        # so từng cột từ trái sang
MISS = SCAN0 + DIFF * SCAN_STEP  # thời điểm gặp X
CASCADE0, CASCADE_STEP = MISS + 8, 4

a = svgkit.Anim(dur=16, fade_out=95)
body: list[str] = []


def x(i: int) -> float:
    return X0 + i * STEP


def cx(i: int) -> float:
    return x(i) + BW / 2


body.append('<text class="title" x="20" y="40">Cache chỉ khớp theo prefix</text>')
body.append(svgkit.legend(W - 300, 38, [("blue", "dùng lại từ cache"), ("hatch", "phải tính lại")]))
body.append(f'<text class="row" x="20" y="{A_Y + 23}">Request A</text>')
body.append(f'<text class="row" x="20" y="{B_Y + 23}">Request B</text>')
body.append(f'<text class="row" x="20" y="{KV_Y + 16}">KV của B</text>')
body.append(f'<text class="sub" x="20" y="{KV_Y + 33}">khi A đã có cache</text>')

for i in range(N):
    body.append(box(x(i), A_Y, BW, BH, "tok", f"t{i}"))
    body.append(box(x(i), B_Y, BW, BH, "tok", "X" if i == DIFF else f"t{i}"))
    body.append(f'<rect class="slot" x="{x(i)}" y="{KV_Y}" width="{BW}" height="{BH}" rx="7"/>')

# Quét từng cột: khớp thì KV được đọc từ cache
for i in range(DIFF):
    t = SCAN0 + i * SCAN_STEP
    body.append(f'<g {a([(t, 0), (t + 1.5, 1)])}><text class="lbl tblue" x="{cx(i)}" y="{B_Y - 9}">=</text></g>')
    body.append(box(x(i), B_Y, BW, BH, "blue", f"t{i}", a([(t, 0), (t + 1.5, 1)])))
    body.append(box(x(i), KV_Y, BW, BH, "blue", f"KV{i}", a([(t + 2, 0), (t + 3.5, 1)])))

# Gặp X: khác token
body.append(f'<g {a([(MISS, 0), (MISS + 1.5, 1)])}><text class="lbl tred" x="{cx(DIFF)}" y="{B_Y - 9}">≠</text></g>')
body.append(box(x(DIFF), B_Y, BW, BH, "red", "X", a([(MISS, 0), (MISS + 1.5, 1)])))

# Mọi KV từ X trở đi phụ thuộc X nên phải tính lại, kể cả khi token giống A
for i in range(DIFF, N):
    t = CASCADE0 + (i - DIFF) * CASCADE_STEP
    body.append(box(x(i), KV_Y, BW, BH, "hatch", f"KV{i}", a([(t, 0), (t + 1.5, 1)])))
end_cascade = CASCADE0 + (N - DIFF) * CASCADE_STEP
# Token sau X giống A (dấu = mờ), nhưng KV của chúng vẫn phải tính lại
same = a([(end_cascade, 0), (end_cascade + 2, 1)])
body.append(f"<g {same}>" + "".join(
    f'<text class="lbl" style="fill:var(--muted)" x="{cx(i)}" y="{B_Y - 9}">=</text>' for i in range(DIFF + 1, N))
    + f'<text class="sub mid" x="{(x(DIFF + 1) + x(N - 1) + BW) / 2}" y="{(B_Y + BH + KV_Y) / 2 + 4}">'
    'token giống A nhưng KV vẫn phải tính lại</text></g>')

cap1 = a([(MISS, 0), (MISS + 2, 1)])
cap2 = a([(end_cascade, 0), (end_cascade + 2, 1)])
body.append(f'<g {cap1}><text class="cap" x="20" y="306">So từ đầu: t0..t2 khớp nên KV đọc từ cache. '
            'Gặp X ở vị trí 3 thì dừng khớp.</text></g>')
body.append(f'<g {cap2}><text class="cap" x="20" y="328">KV của X và mọi token sau nó phụ thuộc X '
            '(attention causal), nên phải prefill lại.</text></g>')

svgkit.write(__file__, "prefix-match.svg", svgkit.svg(
    W, H, "Hai request khác nhau ở token thứ 3: KV trước đó dùng lại từ cache, "
    "KV từ đó trở đi phải tính lại dù các token sau giống hệt", body, a))
