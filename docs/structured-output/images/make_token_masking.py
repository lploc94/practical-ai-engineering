"""Sinh token-masking.svg: SVG động minh họa constrained decoding che token trước khi sample (§1.3).

Dùng đúng ví dụ trong mục "Che token trước khi sample": schema {"anger": integer 1..5},
output đến lúc này là {"anger":, model muốn " \\"" (0,41), " 4" (0,38), " \\"kh" (0,12).
Số liệu giả định thêm: " 3" = 0,06 và " 5" = 0,03 (để 5 token cộng lại bằng 1, bỏ qua
phần còn lại của vocabulary; ba giá trị sau chuẩn hoá làm tròn vẫn cộng đúng 1,00).
Hình ghi rõ phần giả định trong một dòng chú thích hiện suốt chu kỳ. Bước cuối giả định engine không cho thêm khoảng trắng,
nên sau " 4" chỉ còn "}" hợp lệ và engine ghi luôn (jump-forward).

Chạy: python3 make_token_masking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402
from svgkit import box  # noqa: E402

# (token như trong văn bản, xác suất model, hợp lệ theo schema?)
TOKENS = [
    ('" \\""', 0.41, False),
    ('" 4"', 0.38, True),
    ('" \\"kh"', 0.12, False),
    ('" 3"', 0.06, True),
    ('" 5"', 0.03, True),
]
assert abs(sum(p for _, p, _ in TOKENS) - 1.0) < 1e-9
VALID_SUM = sum(p for _, p, ok in TOKENS if ok)
RENORM = {t: p / VALID_SUM for t, p, ok in TOKENS if ok}
assert round(RENORM['" 4"'], 2) == 0.81  # 0,38 / 0,47
assert round(sum(round(v, 2) for v in RENORM.values()), 2) == 1.0  # nhãn hiển thị cộng đúng 1
CHOSEN = 1  # " 4"

W, H = 820, 440
X0, SCALE, ROW0, ROWH, BH = 170, 440, 150, 42, 26
OUT_Y, OBH = 66, 36

# Mốc thời gian (phần trăm của chu kỳ 36 giây): mỗi bước ~5,8 s để kịp đọc một dòng chú thích;
# trạng thái cuối giữ từ 66% đến 95% (~10 s)
P1, P2, P3, P4, P5 = 2, 18, 34, 50, 66
a = svgkit.Anim(dur=36, fade_out=95)
body: list[str] = ['<style>.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
                   'font-size:14px}</style>']


def fmt(p: float) -> str:
    return f"{p:.2f}".replace(".", ",")


def ry(i: int) -> float:
    return ROW0 + i * ROWH


def phase(start: float, end: float | None) -> str:
    """Hiện từ start đến end (None = giữ đến cuối chu kỳ)."""
    if end is None:
        return a([(start, 0), (start + 2, 1)])
    return a([(start, 0), (start + 2, 1), (end - 1, 1), (end, 0)])


# Hàng output
body.append(f'<text class="row" x="20" y="{OUT_Y + 23}">Output</text>')
body.append(box(110, OUT_Y, 104, OBH, "tok", '{"anger":'))
body.append(f'<rect class="slot" x="222" y="{OUT_Y}" width="52" height="{OBH}" rx="7"/>')
body.append(f'<rect class="slot" x="282" y="{OUT_Y}" width="40" height="{OBH}" rx="7"/>')
body.append(box(222, OUT_Y, 52, OBH, "tok", "4", phase(P4 + 2, None)))
body.append(box(282, OUT_Y, 40, OBH, "tok", "}", phase(P5 + 2, None)))
body.append(f'<g {phase(P1, P4 + 2)}><text class="cap2" x="344" y="{OUT_Y + 23}">'
            'automaton: chờ số nguyên 1–5</text></g>')
body.append(f'<g {phase(P4 + 2, None)}><text class="cap2" x="344" y="{OUT_Y + 23}">'
            'automaton: chỉ còn } hợp lệ</text></g>')

# Biểu đồ xác suất
body.append(f'<text class="row" x="20" y="{ROW0 - 16}">Token ứng viên</text>')
body.append(svgkit.legend(W - 300, ROW0 - 16, [("fblue", "xác suất"), ("hatch", "bị che (−inf)")]))
for i, (tok, p, ok) in enumerate(TOKENS):
    y = ry(i)
    body.append(f'<text class="mono end" x="{X0 - 12}" y="{y + 18}">{tok}</text>')
    body.append(f'<line class="axis" x1="{X0}" y1="{y - 4}" x2="{X0}" y2="{y + BH + 4}"/>')
    w0 = p * SCALE
    # Bước 1: phân phối gốc của model; với token hợp lệ, biến mất khi chuẩn hoá lại
    body.append(f'<g {phase(P1, P3 + 1) if ok else phase(P1, None)}>'
                f'<rect class="fblue" x="{X0}" y="{y}" width="{w0:.1f}" height="{BH}" rx="3"/></g>')
    body.append(f'<g {phase(P1, P2 + 1 if not ok else P3 + 1)}>'
                f'<text class="cap" x="{X0 + w0 + 8:.1f}" y="{y + 18}">{fmt(p)}</text></g>')
    if not ok:
        # Bước 2: token không hợp lệ bị che
        body.append(f'<g {phase(P2, None)}><rect class="hatch" x="{X0}" y="{y}" width="{w0:.1f}" '
                    f'height="{BH}" rx="3"/></g>')
        body.append(f'<g {phase(P2 + 1, None)}><text class="cap" x="{X0 + w0 + 8:.1f}" y="{y + 18}">'
                    f'{fmt(p)} → bị che, xác suất 0</text></g>')
    else:
        # Bước 3: chuẩn hoá lại trên các token còn lại
        w1 = RENORM[tok] * SCALE
        body.append(f'<g {phase(P3, None)}><rect class="fblue" x="{X0}" y="{y}" width="{w1:.1f}" '
                    f'height="{BH}" rx="3"/></g>')
        label = f"{fmt(p)} → {fmt(RENORM[tok])}"
        body.append(f'<g {phase(P3 + 1, None)}><text class="cap" x="{X0 + w1 + 8:.1f}" y="{y + 18}">'
                    f'{label}</text></g>')

# Bước 4: sample " 4"
y = ry(CHOSEN)
body.append(f'<g {phase(P4, None)}><rect class="lfg" x="{X0 - 150}" y="{y - 6}" '
            f'width="{W - X0 + 130}" height="{BH + 12}" rx="8"/>'
            f'<text class="cap end" x="{W - 32}" y="{y + 18}">được chọn</text></g>')

# Chú thích theo bước
CAPS = [
    (P1, P2, "Bước 1 · Model trả xác suất cho mọi token",
     "Chưa có ràng buộc: token mở chuỗi \" \\\"\" đang cao nhất."),
    (P2, P3, "Bước 2 · Che token không hợp lệ",
     "Ở vị trí này chỉ số nguyên 1–5 hợp lệ: token mở chuỗi bị đặt logit −inf."),
    (P3, P4, "Bước 3 · Chuẩn hoá lại",
     "Xác suất dồn sang các token còn lại: \" 4\" từ 0,38 lên 0,81."),
    (P4, P5, "Bước 4 · Sample và cập nhật trạng thái",
     "Sampler chọn \" 4\"; automaton chuyển sang trạng thái chỉ nhận }."),
    (P5, None, "Bước 5 · Ghi luôn token cố định",
     "Chỉ còn } hợp lệ nên engine ghi luôn, không chạy model (jump-forward)."),
]
for start, end, title, c1 in CAPS:
    g = a([(start, 0), (start + 2, 1)] + ([(end - 1, 1), (end, 0)] if end else []),
          static=False)
    c1 = c1.replace('"', "&quot;")
    body.append(f'<g {g}><text class="title" x="20" y="40">{title}</text>'
                f'<text class="cap" x="20" y="{H - 44}">{c1}</text></g>')
# Giả định hiện suốt chu kỳ và trong khung tĩnh
body.append(f'<text class="cap2" x="20" y="{H - 20}">Số liệu giả định: &quot; 3&quot; = 0,06, &quot; 5&quot; = 0,03, '
            'bỏ qua phần còn lại của vocabulary; engine không cho khoảng trắng sau số.</text>')
# Khung tĩnh cho prefers-reduced-motion: tóm tắt cả quá trình
body.append('<g class="static-only"><text class="title" x="20" y="40">Constrained decoding: che token rồi mới sample</text>'
            f'<text class="cap" x="20" y="{H - 44}">Token mở chuỗi bị che; &quot; 4&quot; từ 0,38 lên 0,81 và được chọn; '
            'sau đó engine ghi luôn }.</text></g>')

svgkit.write(__file__, "token-masking.svg", svgkit.svg(
    W, H, "Constrained decoding: sau {\"anger\": token mở chuỗi bị che dù model chấm cao nhất, xác suất dồn sang "
    "số 1 đến 5 nên 4 tăng từ 0,38 lên 0,81 và được chọn; sau đó chỉ còn } hợp lệ nên engine ghi luôn"
    .replace('"', "&quot;"), body, a))
