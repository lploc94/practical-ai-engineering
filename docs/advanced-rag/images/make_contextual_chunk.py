"""Sinh contextual-chunk.svg: SVG tĩnh minh hoạ contextual retrieval (§2.3).

Ví dụ minh hoạ, không phải output thật của model: chunk "Điều khoản này không áp
dụng cho đơn hàng dưới 500.000 đồng" (câu mở đầu §2.3) nằm trong mục 3 của
"Chính sách hoàn tiền 2026" (cùng tài liệu với ví dụ search_result ở §2.8).
LLM đọc toàn bộ tài liệu cùng chunk, viết một đoạn ngữ cảnh ngắn (50–100 token
theo bài của Anthropic), ghép vào đầu chunk; bản ghép này đi vào cả index
embedding lẫn index BM25. Phần <document> đặt đầu prompt và được cache, dùng
chung cho mọi chunk của tài liệu.

Chạy: python3 make_contextual_chunk.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

W, H = 860, 450
CHUNK = ["Điều khoản này không áp dụng", "cho đơn hàng dưới 500.000 đồng."]
CONTEXT = ["Trích Chính sách hoàn tiền 2026, mục 3:", "điều kiện hoàn tiền của gói Pro."]

body: list[str] = []
body.append('<text class="title" x="20" y="36">Contextual retrieval: ghép ngữ cảnh vào chunk trước khi index</text>')
body.append('<text class="cap2" x="20" y="58">Ví dụ minh hoạ. Đoạn ngữ cảnh do LLM viết thường dài 50–100 token.</text>')
body.append(svgkit.legend(20, 90, [("blue", "nguyên văn tài liệu"), ("orange", "ngữ cảnh do LLM viết")]))

# Tài liệu nguồn, chunk nằm trong mục 3
DX, DY, DW, DH = 20, 116, 266, 228
body.append(f'<rect class="tok" x="{DX}" y="{DY}" width="{DW}" height="{DH}" rx="7"/>')
body.append(f'<text class="row" x="{DX + 14}" y="{DY + 26}">Chính sách hoàn tiền 2026</text>')
y = DY + 52
for name in ("Mục 1. Phạm vi", "Mục 2. Gói Basic", "Mục 3. Gói Pro"):
    body.append(f'<text class="sub" x="{DX + 14}" y="{y}">{name}</text>')
    y += 34
CY = y - 22
body.append(f'<rect class="blue" x="{DX + 14}" y="{CY}" width="{DW - 28}" height="52" rx="5"/>')
for i, line in enumerate(CHUNK):
    body.append(f'<text class="cap2" x="{DX + 22}" y="{CY + 21 + i * 18}" style="fill:var(--fg)">{line}</text>')
body.append(f'<text class="sub" x="{DX + 14}" y="{CY + 74}">Chunk không có “hoàn tiền”, “Pro”.</text>')

# LLM
LX, LY, LW, LH = 340, 178, 166, 92
body.append(f'<rect class="tok" x="{LX}" y="{LY}" width="{LW}" height="{LH}" rx="7"/>')
body.append(f'<text class="row mid" x="{LX + LW / 2}" y="{LY + 30}">LLM</text>')
body.append(f'<text class="sub mid" x="{LX + LW / 2}" y="{LY + 52}">đọc cả tài liệu</text>')
body.append(f'<text class="sub mid" x="{LX + LW / 2}" y="{LY + 70}">(cache) + chunk</text>')
body.append(f'<line class="lfg" x1="{DX + DW + 4}" y1="{LY + LH / 2}" x2="{LX - 6}" y2="{LY + LH / 2}" '
            'marker-end="url(#afg)"/>')

# Chunk sau khi ghép
RX, RY, RW = 560, 140, 280
body.append(f'<text class="sub" x="{RX}" y="{RY - 10}">Bản dùng để index</text>')
body.append(f'<rect class="orange" x="{RX}" y="{RY}" width="{RW}" height="52" rx="5"/>')
for i, line in enumerate(CONTEXT):
    body.append(f'<text class="cap2" x="{RX + 10}" y="{RY + 21 + i * 18}" style="fill:var(--fg)">{line}</text>')
body.append(f'<rect class="blue" x="{RX}" y="{RY + 58}" width="{RW}" height="52" rx="5"/>')
for i, line in enumerate(CHUNK):
    body.append(f'<text class="cap2" x="{RX + 10}" y="{RY + 79 + i * 18}" style="fill:var(--fg)">{line}</text>')
body.append(f'<line class="lfg" x1="{LX + LW + 4}" y1="{LY + LH / 2}" x2="{RX - 6}" y2="{LY + LH / 2}" '
            'marker-end="url(#afg)"/>')

# Hai index
IY, IW, IH = 326, 130, 44
bottom = RY + 110
for i, name in enumerate(("Embedding", "BM25")):
    ix = RX + i * (RW - IW)
    body.append(f'<line class="lfg" x1="{ix + IW / 2}" y1="{bottom + 4}" x2="{ix + IW / 2}" y2="{IY - 6}" '
                'marker-end="url(#afg)"/>')
    body.append(f'<rect class="tok" x="{ix}" y="{IY}" width="{IW}" height="{IH}" rx="7"/>')
    body.append(f'<text class="row mid" x="{ix + IW / 2}" y="{IY + 28}">{name}</text>')

body.append(f'<text class="cap" x="20" y="{H - 48}">Cùng một bản ghép đi vào cả hai index: BM25 giờ khớp được “hoàn tiền”, “gói Pro”.</text>')
body.append(f'<text class="cap" x="20" y="{H - 24}">Chỉ tốn thêm lúc index; lúc truy vấn không có bước nào mới.</text>')

svgkit.write(__file__, "contextual-chunk.svg", svgkit.svg(
    W, H, "Chunk “Điều khoản này không áp dụng cho đơn hàng dưới 500.000 đồng” không nói đó là chính sách gì; "
    "LLM đọc cả tài liệu rồi viết một câu ngữ cảnh ghép vào đầu chunk, và bản ghép đó được đưa vào cả index "
    "embedding lẫn BM25", body))
