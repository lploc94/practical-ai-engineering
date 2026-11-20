"""Sinh cache-routing.svg: SVG động minh họa vì sao cache phụ thuộc routing (§2.1).

Hai cảnh nối tiếp: request đến đúng máy đang giữ KV (hit), rồi một request cùng
prefix bị đưa sang máy khác (miss). Khung tĩnh cho prefers-reduced-motion hiện cả
hai đường cùng lúc.

Chạy: python3 make_cache_routing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 820, 360
a = svgkit.Anim(dur=16, fade_out=95)
body: list[str] = []

REQ = (20, 140, 170, 64)       # x, y, w, h
ROUTER = (270, 140, 150, 64)
MA = (515, 62, 290, 84)
MB = (515, 206, 290, 84)
S1, S2 = 4, 50                 # thời điểm bắt đầu hai cảnh (%)


def rect(b, cls, extra=""):
    x, y, w, h = b
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" {extra}/>'


def g(attrs: str, inner: str) -> str:
    return f"<g {attrs}>{inner}</g>"


body.append('<text class="title" x="20" y="38">Cache nằm trên một máy cụ thể</text>')
body.append(svgkit.legend(W - 250, 36, [("blue", "cache hit"), ("hatch", "miss: prefill lại")]))

# Khung cố định
body.append(rect(REQ, "tok"))
body.append(f'<text class="row mid" x="{REQ[0] + REQ[2] / 2}" y="{REQ[1] + 28}">Request</text>')
body.append(f'<text class="sub mid" x="{REQ[0] + REQ[2] / 2}" y="{REQ[1] + 47}">prefix P</text>')
body.append(rect(ROUTER, "tok"))
body.append(f'<text class="row mid" x="{ROUTER[0] + ROUTER[2] / 2}" y="{ROUTER[1] + 28}">Router</text>')
body.append(f'<text class="sub mid" x="{ROUTER[0] + ROUTER[2] / 2}" y="{ROUTER[1] + 47}">hash(prefix, cache key)</text>')
for b, name in ((MA, "Máy A"), (MB, "Máy B")):
    body.append(rect(b, "tok"))
    body.append(f'<text class="row" x="{b[0] + 14}" y="{b[1] + 26}">{name}</text>')
# KV đang có trên máy A; máy B chưa có
kv_a = (MA[0] + 14, MA[1] + 38, 130, 34)
kv_b = (MB[0] + 14, MB[1] + 38, 130, 34)
body.append(svgkit.box(*kv_a, "blue", "KV của P"))
body.append(f'<rect class="slot" x="{kv_b[0]}" y="{kv_b[1]}" width="{kv_b[2]}" height="{kv_b[3]}" rx="7"/>')
body.append(f'<text class="sub mid" x="{kv_b[0] + kv_b[2] / 2}" y="{kv_b[1] + 21}">chưa có KV của P</text>')

# Mũi tên request → router dùng chung cho hai cảnh
rx1 = REQ[0] + REQ[2]
body.append(g(a([(S1, 0), (S1 + 3, 1)]),
              f'<line class="lfg" x1="{rx1 + 6}" y1="{REQ[1] + 32}" '
              f'x2="{ROUTER[0] - 8}" y2="{ROUTER[1] + 32}" marker-end="url(#afg)"/>'))

# Cảnh 1: đến máy A, hit
ox, oy = ROUTER[0] + ROUTER[2] + 6, ROUTER[1] + 20
hit_path = (f'<path class="lblue" d="M{ox},{oy} C{ox + 60},{oy} {MA[0] - 70},{MA[1] + 42} {MA[0] - 8},{MA[1] + 42}" '
            f'marker-end="url(#ablue)"/>')
hit = (f'<rect class="lblue" x="{kv_a[0] - 4}" y="{kv_a[1] - 4}" width="{kv_a[2] + 8}" height="{kv_a[3] + 8}" rx="9"/>'
       f'<text class="cap" x="{kv_a[0] + kv_a[2] + 14}" y="{kv_a[1] + 15}">hit: đọc KV,</text>'
       f'<text class="cap" x="{kv_a[0] + kv_a[2] + 14}" y="{kv_a[1] + 33}">bỏ qua prefill P</text>')
body.append(g(a([(S1 + 6, 0), (S1 + 9, 1), (S2 - 6, 1), (S2 - 3, 0)], static=True), hit_path))
body.append(g(a([(S1 + 12, 0), (S1 + 15, 1), (S2 - 6, 1), (S2 - 3, 0)], static=True), hit))
body.append(g(a([(S1 + 6, 0), (S1 + 9, 1), (S2 - 6, 1), (S2 - 3, 0)], static=False),
              '<text class="cap" x="20" y="326">Router đưa request về máy đã giữ KV của prefix P '
              '(cùng prefix, cùng cache key) → cache hit.</text>'))

# Cảnh 2: cùng prefix nhưng sang máy B, miss
oy2 = ROUTER[1] + 44
miss_path = (f'<path class="lred dash" d="M{ox},{oy2} C{ox + 60},{oy2} {MB[0] - 70},{MB[1] + 42} {MB[0] - 8},{MB[1] + 42}" '
             f'marker-end="url(#ared)"/>')
miss = (svgkit.box(*kv_b, "hatch", "prefill lại P")
        + f'<text class="cap" x="{kv_b[0] + kv_b[2] + 14}" y="{kv_b[1] + 15}">miss: prefix giống</text>'
        f'<text class="cap" x="{kv_b[0] + kv_b[2] + 14}" y="{kv_b[1] + 33}">nhưng KV ở máy A</text>')
body.append(g(a([(S2 + 6, 0), (S2 + 9, 1)], static=True), miss_path))
body.append(g(a([(S2 + 12, 0), (S2 + 15, 1)], static=True), miss))
body.append(g(a([(S2 + 6, 0), (S2 + 9, 1)], static=False),
              '<text class="cap" x="20" y="326">Cùng prefix P nhưng request sang máy khác (traffic tràn, '
              'thiếu cache key) → miss.</text>'))

body.append('<g class="static-only"><text class="cap" x="20" y="326">Cùng prefix P: đến máy đang giữ KV thì hit, '
            'sang máy khác thì miss và phải prefill lại.</text></g>')
body.append('<text class="cap2" x="20" y="348">Gợi ý routing: prompt_cache_key (OpenAI, xAI Responses), '
            'x-grok-conv-id (xAI Chat Completions).</text>')

svgkit.write(__file__, "cache-routing.svg", svgkit.svg(
    W, H, "Hai request cùng prefix: request đến máy đang giữ KV thì cache hit, "
    "request bị đưa sang máy khác thì miss và phải prefill lại", body, a))
