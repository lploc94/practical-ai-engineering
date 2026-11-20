"""Sinh prefill-decode.svg: SVG động minh họa hai pha prefill và decode (§1.2).

Chạy: python3 make_prefill_decode.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402
from svgkit import box  # noqa: E402

N_PROMPT, N_GEN = 6, 3
X0, STEP, BW, BH = 170, 78, 64, 36
TOK_Y, KV_Y = 92, 206
W, H = X0 + (N_PROMPT + N_GEN) * STEP + 20, 330

# Mốc thời gian (phần trăm của chu kỳ)
PREFILL_ON, PREFILL_FULL = 8, 16
DECODE_START, DECODE_STEP = 28, 17

a = svgkit.Anim(dur=16, fade_out=95)  # trạng thái cuối giữ khoảng 3 giây
body: list[str] = []


def x(i: int) -> float:
    return X0 + i * STEP


def cx(i: int) -> float:
    return x(i) + BW / 2


body.append(f'<text class="row" x="20" y="{TOK_Y + 23}">Token</text>')
body.append(f'<text class="row" x="20" y="{KV_Y + 16}">KV cache</text>')
body.append(f'<text class="sub" x="20" y="{KV_Y + 33}">(bộ nhớ GPU)</text>')
for i in range(N_PROMPT + N_GEN):
    body.append(f'<rect class="slot" x="{x(i)}" y="{TOK_Y}" width="{BW}" height="{BH}" rx="7"/>')
    body.append(f'<rect class="slot" x="{x(i)}" y="{KV_Y}" width="{BW}" height="{BH}" rx="7"/>')

# Prefill: prompt sáng lên đồng loạt, KV của cả prompt được ghi một lượt
for i in range(N_PROMPT):
    body.append(box(x(i), TOK_Y, BW, BH, "tok", f"t{i}"))
hl = a([(PREFILL_ON, 0), (PREFILL_ON + 2, 1), (DECODE_START - 4, 1), (DECODE_START, 0)])
body.append(f"<g {hl}>" + "".join(box(x(i), TOK_Y, BW, BH, "blue", f"t{i}") for i in range(N_PROMPT)) + "</g>")
arr = a([(PREFILL_ON + 2, 0), (PREFILL_ON + 4, 1), (DECODE_START - 4, 1), (DECODE_START, 0)])
body.append(f"<g {arr}>" + "".join(
    f'<line class="lblue" x1="{cx(i)}" y1="{TOK_Y + BH + 4}" x2="{cx(i)}" y2="{KV_Y - 8}" marker-end="url(#ablue)"/>'
    for i in range(N_PROMPT)) + "</g>")
kvp = a([(PREFILL_FULL - 4, 0), (PREFILL_FULL, 1)])
body.append(f"<g {kvp}>" + "".join(box(x(i), KV_Y, BW, BH, "blue", f"KV{i}") for i in range(N_PROMPT)) + "</g>")

# Decode: mỗi bước đọc toàn bộ KV đã có, sinh một token, thêm KV của nó
for k in range(N_GEN):
    pos = N_PROMPT + k
    s = DECODE_START + k * DECODE_STEP
    att_on, att_off = s + 1, s + 9
    tok_on, kv_on = s + 9, s + 13
    right = x(pos - 1) + BW + 5
    att = a([(att_on, 0), (att_on + 2, 1), (att_off, 1), (att_off + 2, 0)])
    body.append(
        f'<g {att}><rect class="lorange dash" x="{X0 - 5}" y="{KV_Y - 5}" width="{right - X0 + 5}" '
        f'height="{BH + 10}" rx="10"/>'
        f'<line class="lorange" x1="{cx(pos)}" y1="{TOK_Y + BH + 4}" x2="{right - 18}" y2="{KV_Y - 9}" '
        f'marker-end="url(#aorange)"/>'
        f'<text class="sub mid torange" x="{(X0 + right) / 2}" y="{KV_Y - 14}">đọc KV của mọi token trước</text></g>')
    q = a([(att_on, 0), (att_on + 2, 1), (tok_on, 1), (tok_on + 0.5, 0)])
    body.append(f'<g {q}><rect class="lorange dash" x="{x(pos)}" y="{TOK_Y}" width="{BW}" height="{BH}" rx="7" '
                f'style="fill:var(--bg)"/><text class="lbl" x="{cx(pos)}" y="{TOK_Y + BH / 2 + 5}">?</text></g>')
    body.append(box(x(pos), TOK_Y, BW, BH, "orange", f"t{pos}", a([(tok_on, 0), (tok_on + 1.5, 1)])))
    a2 = a([(tok_on + 1, 0), (tok_on + 2, 1), (kv_on + 3, 1), (kv_on + 5, 0)])
    body.append(f'<g {a2}><line class="lorange" x1="{cx(pos)}" y1="{TOK_Y + BH + 4}" '
                f'x2="{cx(pos)}" y2="{KV_Y - 8}" marker-end="url(#aorange)"/></g>')
    body.append(box(x(pos), KV_Y, BW, BH, "orange", f"KV{pos}", a([(kv_on, 0), (kv_on + 1.5, 1)])))

# Tiêu đề và chú thích theo pha
t1 = a([(DECODE_START - 3, 1), (DECODE_START - 1, 0)])
t2 = a([(DECODE_START - 1, 0), (DECODE_START + 1, 1)], static=False)
body.append(f'<g {t1}><text class="title" x="20" y="44">Pha 1 · Prefill</text>'
            '<text class="cap" x="20" y="292">Đọc toàn bộ prompt trong một lượt, tính KV cho mọi token song song.</text>'
            '<text class="cap2" x="20" y="314">Tốn compute; quyết định time-to-first-token. Prompt caching giúp bỏ bớt pha này.</text></g>')
body.append(f'<g {t2}><text class="title" x="20" y="44">Pha 2 · Decode</text>'
            '<text class="cap" x="20" y="292">Mỗi bước: token mới attend vào KV của mọi token trước → sinh 1 token → thêm KV của nó.</text>'
            '<text class="cap2" x="20" y="314">Tuần tự từng token; bị giới hạn bởi băng thông bộ nhớ khi đọc KV cache.</text></g>')
# Khung tĩnh cho prefers-reduced-motion: tóm tắt cả hai pha
body.append('<g class="static-only"><text class="title" x="20" y="44">Prefill → Decode</text>'
            '<text class="cap" x="20" y="292">Xanh: KV của prompt, tính song song trong một lượt (prefill).</text>'
            '<text class="cap" x="20" y="314">Cam: token sinh tuần tự (decode); mỗi token đọc toàn bộ KV rồi thêm KV của nó.</text></g>')
body.append(svgkit.legend(W - 260, 42, [("blue", "từ prompt"), ("orange", "token sinh ra")]))

svgkit.write(__file__, "prefill-decode.svg", svgkit.svg(
    W, H, "Minh họa hai pha prefill và decode của LLM inference", body, a))
