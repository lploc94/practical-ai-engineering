"""Sinh speculative-moderation.svg: hình tĩnh, kiểm duyệt input tuần tự và speculative (§2.4).

Ba hàng chung một trục thời gian:
- Tuần tự: kiểm duyệt input xong mới bắt đầu sinh câu trả lời.
- Speculative, input hợp lệ: kiểm duyệt và sinh chạy cùng lúc; kiểm duyệt xong
  trước chữ đầu tiên nên bước kiểm duyệt gần như không thêm độ trễ [9].
- Speculative, input bị chặn: kiểm duyệt báo chặn thì huỷ nhánh sinh ngay;
  request của nhánh bị huỷ vẫn có thể bị tính tiền (Claude tính tiền request bị
  client ngắt giữa chừng, §2.4).
Số liệu giả định, chỉ để minh hoạ: kiểm duyệt 1,0 s; TTFT của câu trả lời 1,5 s
(cùng con số TTFT ở §3.2). Thanh output kéo tới mép phải nghĩa là còn tiếp.

Chạy: python3 make_speculative_moderation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

MOD, TTFT = 1.0, 1.5
FIRST_SEQ = MOD + TTFT                 # tuần tự: chữ đầu sau kiểm duyệt + TTFT
FIRST_SPEC = max(MOD, TTFT)            # speculative: chữ đầu khi cả hai đã xong phần cần chờ
assert MOD < TTFT                      # kiểm duyệt xong trước chữ đầu: không phải giữ output
assert (FIRST_SEQ, FIRST_SPEC) == (2.5, 1.5)

W, H = 820, 440
X0, X1 = 230, 770                      # trục 0 → TMAX giây
TMAX = 4
BH, GAP = 26, 6
ROWS = {"seq": 100, "ok": 186, "block": 286}


def x(s: float) -> float:
    return X0 + s / TMAX * (X1 - X0)


def fmt(s: float) -> str:
    return f"{s:.1f}".replace(".", ",")


def bar(s0: float, s1: float, y: float, cls: str, label: str = "") -> str:
    out = f'<rect class="{cls}" x="{x(s0) + 1:.1f}" y="{y}" width="{x(s1) - x(s0) - 2:.1f}" height="{BH}" rx="4"/>'
    if label:
        out += f'<text class="lbl" style="font-size:12px" x="{(x(s0) + x(s1)) / 2:.1f}" y="{y + BH / 2 + 4}">{label}</text>'
    return out


def mark(s: float, y0: float, y1: float, text: str) -> str:
    return (f'<line class="lfg dash" x1="{x(s):.1f}" y1="{y0}" x2="{x(s):.1f}" y2="{y1}"/>'
            f'<text class="sub halo" x="{x(s) + 6:.1f}" y="{y1 + 2}">{text}</text>')


body: list[str] = []
body.append('<text class="title" x="20" y="38">Kiểm duyệt input: chạy trước rồi sinh, hay chạy cùng lúc</text>')
body.append(svgkit.legend(20, 70, [("tok", "kiểm duyệt input"), ("blue", "TTFT"), ("orange", "chữ trả lời")]))

# Tuần tự
y = ROWS["seq"]
body.append(f'<text class="row" x="20" y="{y + 12}">Tuần tự</text>')
body.append(f'<text class="sub" x="20" y="{y + 29}">kiểm xong mới sinh</text>')
body.append(bar(0, MOD, y, "tok", "kiểm duyệt"))
body.append(bar(MOD, FIRST_SEQ, y, "blue"))
body.append(bar(FIRST_SEQ, TMAX, y, "orange"))
body.append(mark(FIRST_SEQ, y - 6, y + BH + 16, f"chữ đầu: giây {fmt(FIRST_SEQ)}"))

# Speculative, input hợp lệ
y = ROWS["ok"]
body.append(f'<text class="row" x="20" y="{y + 20}">Speculative</text>')
body.append(f'<text class="sub" x="20" y="{y + 37}">input hợp lệ</text>')
body.append(bar(0, MOD, y, "tok", "kiểm duyệt: ok"))
body.append(bar(0, TTFT, y + BH + GAP, "blue"))
body.append(bar(TTFT, TMAX, y + BH + GAP, "orange"))
body.append(mark(FIRST_SPEC, y - 6, y + 2 * BH + GAP + 16, f"chữ đầu: giây {fmt(FIRST_SPEC)}"))

# Speculative, input bị chặn
y = ROWS["block"]
body.append(f'<text class="row" x="20" y="{y + 20}">Speculative</text>')
body.append(f'<text class="sub" x="20" y="{y + 37}">input bị chặn</text>')
body.append(bar(0, MOD, y, "tok", "kiểm duyệt: chặn"))
body.append(bar(0, MOD, y + BH + GAP, "blue"))
body.append(bar(MOD, TTFT, y + BH + GAP, "slot"))
body.append(f'<text class="sub" x="{x(TTFT) + 8:.1f}" y="{y + BH + GAP + BH / 2 + 4}">'
            f'giây {fmt(MOD)}: huỷ nhánh sinh; request đã gửi vẫn tính tiền</text>')

# Trục thời gian
ay = ROWS["block"] + 2 * BH + GAP + 22
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for s in range(0, TMAX + 1):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')

body.append(f'<text class="cap2" x="20" y="{H - 30}">Nếu kiểm duyệt xong sau chữ đầu, giữ output lại tới khi kiểm xong '
            'rồi mới gửi cho người dùng.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">Giả định: kiểm duyệt {fmt(MOD)} s, TTFT {fmt(TTFT)} s.</text>')

svgkit.write(__file__, "speculative-moderation.svg", svgkit.svg(
    W, H, f"Kiểm duyệt rồi mới sinh thì chữ đầu đến ở giây {fmt(FIRST_SEQ)}; chạy kiểm duyệt cùng lúc với sinh thì "
    f"chữ đầu đến ở giây {fmt(FIRST_SPEC)}; nếu input bị chặn, nhánh sinh bị huỷ ngay khi kiểm duyệt xong", body))
