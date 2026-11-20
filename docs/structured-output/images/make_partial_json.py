"""Sinh partial-json.svg: SVG động minh họa parse partial JSON khi streaming (§3.4).

Dùng hai chunk trong ví dụ của §3.4 và thêm chunk 3 = '0}]}' để minh hoạ câu
"120 có thể mới là phần đầu của 1200": giá trị tạm đã hiển thị hoá ra sai. Chunk 3 là
giả định cho minh hoạ. Script kiểm tra mỗi partial object đều parse được và giá trị
revenue đúng như hình vẽ.

Chạy: python3 make_partial_json.py
"""

import json
import sys
import unicodedata
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

CHUNKS = ['{"title": "Báo cáo Q3", "rows": [{"name": "Hà N', 'ội", "revenue": 120', "0}]}"]
CLOSE = ['"}]}', "}]}", ""]  # phần parser đóng tạm sau mỗi chunk
CHUNKS = [unicodedata.normalize("NFC", c) for c in CHUNKS]

received = ["".join(CHUNKS[: k + 1]) for k in range(3)]
partials = [json.loads(received[k] + CLOSE[k]) for k in range(3)]
assert partials[0]["rows"][0] == {"name": "Hà N"}
assert partials[1]["rows"][0] == {"name": "Hà Nội", "revenue": 120}
assert partials[2]["rows"][0] == {"name": "Hà Nội", "revenue": 1200}
REV = received[1].rindex("120")  # vị trí của 120 trong partial 2

W, H = 820, 420
CW, X0, GAP = 7.8, 170, 8  # độ rộng một ký tự mono 13px; textLength giữ đúng độ rộng này
RAW_Y, P_Y, P_DY = 96, 186, 58
assert X0 + (len(received[2]) + 4) * CW + 2 * GAP < W - 20, "chuỗi tràn khung"

# Mốc thời gian (phần trăm của chu kỳ 28 giây): mỗi chunk ~7,5 s để kịp đọc chú thích;
# trạng thái cuối giữ từ ~71% đến 95% (~6,7 s)
T = [3, 30, 57]
PARTIAL_DELAY, HATCH = 7, 69
a = svgkit.Anim(dur=28, fade_out=95)
body: list[str] = ['<style>.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
                   'font-size:13px}.muted{fill:var(--muted)}</style>']


def show(start: float) -> str:
    return a([(start, 0), (start + 2, 1)])


def mono(x: float, y: float, s: str, cls: str = "mono") -> str:
    return (f'<text class="{cls}" x="{x:.1f}" y="{y}" textLength="{len(s) * CW:.1f}" '
            f'lengthAdjust="spacingAndGlyphs">{escape(s, quote=False)}</text>')


# Hàng "Đã nhận": các chunk nối nhau theo thời gian
body.append(f'<text class="row" x="20" y="{RAW_Y + 5}">Đã nhận</text>')
body.append(f'<text class="sub" x="20" y="{RAW_Y + 22}">byte từ stream</text>')
x = X0
for k, c in enumerate(CHUNKS):
    w = len(c) * CW
    body.append(f'<g {show(T[k])}><rect class="blue" x="{x - 2:.1f}" y="{RAW_Y - 15}" width="{w + 4:.1f}" '
                f'height="22" rx="4"/>{mono(x, RAW_Y, c)}'
                f'<text class="sub mid" x="{x + w / 2:.1f}" y="{RAW_Y + 26}">chunk {k + 1}</text></g>')
    x += w + GAP  # khe nhỏ chỉ để tách khung từng chunk; trong buffer chúng liền nhau

# Partial object sau mỗi chunk
labels = ["sau chunk 1", "sau chunk 2", "sau chunk 3"]
body.append(f'<text class="row" x="20" y="{P_Y - 34}">Partial object</text>')
for k in range(3):
    y = P_Y + k * P_DY
    g = show(T[k] + PARTIAL_DELAY)
    s, close = received[k], CLOSE[k]
    parts = [f'<text class="sub" x="20" y="{y}">{labels[k]}</text>']
    if k == 1:
        # 120 đã hiển thị nhưng chunk 3 cho thấy giá trị thật là 1200
        hx = X0 + REV * CW
        parts.append(f'<g {show(HATCH)}><rect class="hatch" x="{hx - 3:.1f}" y="{y - 15}" '
                     f'width="{3 * CW + 3:.1f}" height="22" rx="4"/>'
                     f'<text class="cap end" x="{hx + 3 * CW + 3:.1f}" y="{y + 26}">giá trị tạm đã hiện; thật ra là 1200</text></g>')
    parts.append(mono(X0, y, s))
    if close:
        cx = X0 + len(s) * CW
        parts.append(f'<rect class="slot" x="{cx:.1f}" y="{y - 15}" width="{len(close) * CW + 4:.1f}" '
                     f'height="22" rx="4"/>{mono(cx + 2, y, close, "mono muted")}')
    else:
        parts.append(f'<text class="cap" x="{X0}" y="{y + 26}">JSON đã đóng đủ: lúc này mới validate rồi dùng</text>')
    body.append(f'<g {g}>' + "".join(parts) + "</g>")

body.append(svgkit.legend(W - 430, 40, [("blue", "đã nhận"), ("slot", "parser đóng tạm"), ("hatch", "giá trị tạm sai")]))

CAPS = [
    (T[0], T[1], "Chunk 1 · Dừng giữa chữ",
     "Parser đóng tạm chuỗi, object và mảng còn mở để có object hợp lệ cho UI.",
     "Tên mới có &quot;Hà N&quot;: chỉ dùng để hiển thị."),
    (T[1], T[2], "Chunk 2 · Có revenue",
     "Partial object có revenue = 120, nhưng số chưa kết thúc.",
     "Field bắt buộc cũng có thể chưa đến."),
    (T[2], None, "Chunk 3 · Giá trị thật là 1200",
     "120 đã hiện ở bước trước chỉ là phần đầu của 1200.",
     "Chờ JSON hoàn chỉnh và validate trước khi tính toán hay ghi database."),
]
for start, end, title, c1, c2 in CAPS:
    g = a([(start, 0), (start + 2, 1)] + ([(end - 1, 1), (end, 0)] if end else []), static=False)
    body.append(f'<g {g}><text class="title" x="20" y="40">{title}</text>'
                f'<text class="cap" x="20" y="{H - 44}">{c1}</text>'
                f'<text class="cap2" x="20" y="{H - 20}">{c2}</text></g>')
body.append('<g class="static-only"><text class="title" x="20" y="40">Partial JSON chỉ để hiển thị tạm</text>'
            f'<text class="cap" x="20" y="{H - 44}">Parser đóng tạm ngoặc sau mỗi chunk; revenue = 120 sau chunk 2 '
            'hoá ra là phần đầu của 1200.</text>'
            f'<text class="cap2" x="20" y="{H - 20}">Chờ JSON hoàn chỉnh và validate trước khi tính toán '
            'hay ghi database.</text></g>')

svgkit.write(__file__, "partial-json.svg", svgkit.svg(
    W, H, "Ba chunk stream: parser đóng tạm ngoặc để có partial object sau mỗi chunk; revenue = 120 sau chunk 2 "
    "hoá ra là phần đầu của 1200, nên chỉ dùng object khi JSON đã hoàn chỉnh và validate", body, a))
