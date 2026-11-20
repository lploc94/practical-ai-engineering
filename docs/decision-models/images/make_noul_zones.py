"""Sinh noul-zones.svg: SVG tĩnh, ba vùng ngưỡng của câu yes/no (§4.4, số liệu từ §1.4).

Sáu tin nhắn và xác suất `noul` lấy nguyên từ bảng ở §1.4 (docs Jev, jev-1.13.0,
câu hỏi "Khách có đang muốn gặp người thật không?"). Hai ngưỡng khởi đầu 0.3 và
0.7 theo §1.5 và §4.4. Câu dài nhất được rút gọn bằng dấu "…".

Chạy: python3 make_noul_zones.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

LOW, HIGH = 0.3, 0.7
rows = [
    ("Thanks, that fixed it!", 0.02),
    ("How do I reset my password?", 0.07),
    ("I need this sorted today, whatever it takes.", 0.26),
    ("Are you a bot?", 0.40),
    ("Is there any way to speak to someone about my invoice?", 0.84),
    ("I have asked three times now. … talk to a real person?", 0.99),
]
zone = ["không" if p < LOW else "có" if p > HIGH else "giữa" for _, p in rows]
assert zone == ["không", "không", "không", "giữa", "có", "có"]

W, H = 820, 404
PX0, PX1 = 440, 790               # trục xác suất 0 → 1
RY0, RH = 150, 32                 # hàng đầu, chiều cao mỗi hàng
PY0, PY1 = RY0 - 22, RY0 + len(rows) * RH - 8


def x(p: float) -> float:
    return PX0 + p * (PX1 - PX0)


body: list[str] = []
body.append('<style>.mid-zone{fill:var(--orange-bg);opacity:.7}.dot{fill:var(--fg)}</style>')
body.append('<text class="title" x="20" y="38">Hai ngưỡng chia xác suất yes/no thành ba vùng</text>')
body.append('<text class="cap2" x="20" y="62">"Khách có đang muốn gặp người thật không?" trên jev-1.13.0, '
            'số liệu từ docs Jev (bảng ở §1.4).</text>')

# Vùng giữa tô nền cam nhạt; hai ngưỡng là đường nét đứt
body.append(f'<rect class="mid-zone" x="{x(LOW)}" y="{PY0}" width="{x(HIGH) - x(LOW)}" height="{PY1 - PY0}"/>')
for t in (LOW, HIGH):
    body.append(f'<line class="lorange dash" x1="{x(t)}" y1="{PY0 - 6}" x2="{x(t)}" y2="{PY1}"/>')
    body.append(f'<text class="sub mid" x="{x(t)}" y="{PY1 + 18}">{t}</text>')
for v in (0, 1):
    body.append(f'<text class="sub mid" x="{x(v)}" y="{PY1 + 18}">{v}</text>')
body.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}"/>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{PY1 + 36}">xác suất "có" (noul)</text>')

# Nhãn ba vùng ở trên
for lo, hi, top, bottom in ((0, LOW, "không", "không hành động"),
                            (LOW, HIGH, "xử lý riêng", "hỏi lại, gắn cờ"),
                            (HIGH, 1, "có", "hành động")):
    cxm = (x(lo) + x(hi)) / 2
    body.append(f'<text class="row mid" x="{cxm}" y="{PY0 - 30}">{top}</text>')
    body.append(f'<text class="sub mid" x="{cxm}" y="{PY0 - 12}">{bottom}</text>')

for i, (msg, p) in enumerate(rows):
    cy = RY0 + i * RH + 8
    body.append(f'<line class="grid" x1="{PX0}" y1="{cy}" x2="{PX1}" y2="{cy}"/>')
    body.append(f'<text class="cap end" x="{PX0 - 14}" y="{cy + 5}">{msg}</text>')
    dot_cls = "forange" if zone[i] == "giữa" else "dot"
    body.append(f'<circle class="{dot_cls}" cx="{x(p)}" cy="{cy}" r="6"/>')
    # Nhãn số đặt phía không bị đường ngưỡng cắt
    near_line = any(t - 0.08 < p < t for t in (LOW, HIGH))
    anchor, dx = ("end", -12) if p > 0.5 or near_line else ("", 12)
    body.append(f'<text class="sub halo {anchor}" x="{x(p) + dx}" y="{cy + 4}">{p:.2f}</text>')

body.append(f'<text class="cap2" x="20" y="{H - 14}">0.3 và 0.7 là điểm khởi đầu; ngưỡng thật đặt từ dữ liệu có nhãn '
            'theo chi phí của từng loại sai.</text>')

svgkit.write(__file__, "noul-zones.svg", svgkit.svg(
    W, H, "Sáu tin nhắn trên trục xác suất muốn gặp người thật: ba tin dưới 0.3 là không, hai tin trên 0.7 là có, "
    "riêng Are you a bot với 0.40 rơi vào vùng giữa 0.3 đến 0.7 cần đường xử lý riêng", body))
