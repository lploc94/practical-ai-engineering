"""Sinh guardrail-latency.svg: SVG tĩnh, guardrail cộng thêm bao nhiêu thời gian tuỳ cách xếp (§3.1).

Hai ý của §3.1:
- Input guardrail chạy trước, tuần tự: chữ đầu tiên tới khách muộn thêm toàn bộ thời
  gian guardrail. Chạy song song với lần sinh: gần như không thêm gì, nhưng nếu
  guardrail chặn thì phải huỷ lần sinh, token đã sinh vẫn tính tiền và tool có thể
  đã chạy.
- Ba output guardrail độc lập (moderation, PII, policy check) chạy song song thì xong
  sau guardrail chậm nhất, không phải sau tổng của ba.
Độ dài các thanh là giả định, chỉ để thấy hình dạng; không có đơn vị thời gian.
Riêng thanh PII được vẽ ngắn nhất vì §2.4 đo `redact` mất khoảng 1,7 ms, còn
moderation là một lần gọi mạng và policy check là một lần gọi LLM.
Giả định cho input guardrail: guardrail xong trước khi model sinh ra chữ đầu tiên,
nên khi chạy song song TTFT không đổi.
Hình tĩnh: các thanh cần nhìn cùng lúc để so độ dài, không có ý nào mất khi bỏ
chuyển động.

Chạy: python3 make_guardrail_latency.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
assert "| Input guardrail chạy trước, tuần tự | Toàn bộ thời gian guardrail | Không |" in README
assert "| Input guardrail song song với lần sinh | Gần như không | Tốn token khi bị chặn; tool có thể đã chạy |" in README
assert "tổng thời gian bằng guardrail chậm nhất, không bằng tổng" in README
assert "mất khoảng 1,7 ms" in README

# Giả định (đơn vị tuỳ ý)
G_IN = 1.5          # input guardrail
TTFT = 2.0          # model sinh chữ đầu tiên sau từng này
GEN = 6.0           # model sinh xong
assert G_IN < TTFT  # giả định để song song không thêm gì vào TTFT
OUT = [("moderation", 1.8), ("PII", 0.15), ("policy check", 3.0)]
assert min(OUT, key=lambda o: o[1])[0] == "PII"
SEQ_END = sum(d for _, d in OUT)
PAR_END = max(d for _, d in OUT)

W, H = 980, 490
X0, X1 = 210, 940
TMAX = 8.5
BAR = 22


def x(t: float) -> float:
    return X0 + t / TMAX * (X1 - X0)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Guardrail cộng thêm bao nhiêu thời gian, tuỳ cách xếp</text>')
body.append(svgkit.legend(20, 66, [("orange", "model đang sinh"), ("tok", "guardrail chạy")]))
body.append(f'<path class="lblue" d="M{X1 - 260},55V71"/><text class="sub" x="{X1 - 252}" y="66">'
            'chữ đầu tiên tới khách</text>')


def bar(t0: float, t1: float, y: float, cls: str, label: str, h: float = BAR) -> str:
    w = x(t1) - x(t0)
    lab = (f'<text class="sub mid" x="{x(t0) + w / 2:.1f}" y="{y + h / 2 + 4}">{label}</text>'
           if w > 60 else f'<text class="sub" x="{x(t1) + 6:.1f}" y="{y + h / 2 + 4}">{label}</text>')
    return f'<rect class="{cls}" x="{x(t0):.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="4"/>' + lab


def tick(t: float, y0: float, y1: float, label: str) -> str:
    return (f'<path class="lblue" d="M{x(t):.1f},{y0}V{y1}"/>'
            f'<text class="sub tblue" x="{x(t) + 5:.1f}" y="{y0 - 1}">{label}</text>')


# ---------- Panel A: input guardrail ----------
YA = 104
body.append(f'<text class="row" x="20" y="{YA}">Input guardrail</text>')
r1, r2 = YA + 20, YA + 100
body.append(f'<text class="cap" x="20" y="{r1 + 16}">chạy trước, tuần tự</text>')
body.append(bar(0, G_IN, r1, "tok", "guardrail"))
body.append(bar(G_IN, G_IN + GEN, r1, "orange", "model sinh"))
body.append(tick(G_IN + TTFT, r1 - 6, r1 + BAR + 6, "chữ đầu"))
body.append(f'<path class="lfg" d="M{x(0):.1f},{r1 + BAR + 10}v5H{x(G_IN):.1f}v-5"/>'
            f'<text class="sub" x="{x(0):.1f}" y="{r1 + BAR + 30}">chữ đầu muộn thêm toàn bộ thời gian guardrail</text>')

body.append(f'<text class="cap" x="20" y="{r2 + 16}">song song lần sinh</text>')
body.append(bar(0, G_IN, r2, "tok", "guardrail", BAR - 4))
body.append(bar(0, GEN, r2 + BAR, "orange", "model sinh", BAR - 4))
body.append(tick(TTFT, r2 - 6, r2 + 2 * BAR + 2, "chữ đầu"))
body.append(f'<text class="sub" x="{x(GEN) + 10:.1f}" y="{r2 + 12}">bị chặn thì huỷ lần sinh:</text>'
            f'<text class="sub" x="{x(GEN) + 10:.1f}" y="{r2 + 28}">token đã sinh vẫn tính tiền,</text>'
            f'<text class="sub" x="{x(GEN) + 10:.1f}" y="{r2 + 44}">tool có thể đã chạy</text>')

# ---------- Panel B: ba output guardrail ----------
YB = 312
body.append(f'<line class="grid" x1="20" y1="{YB - 30}" x2="{W - 30}" y2="{YB - 30}"/>')
body.append(f'<text class="row" x="20" y="{YB}">Ba output guardrail độc lập, sau khi có câu trả lời</text>')
b1, b2 = YB + 20, YB + 70
body.append(f'<text class="cap" x="20" y="{b1 + 16}">lần lượt</text>')
t = 0.0
for name, d in OUT:
    if x(t + d) - x(t) > 60:
        body.append(bar(t, t + d, b1, "tok", name))
    else:                                   # thanh quá ngắn: nhãn đặt phía trên
        body.append(f'<rect class="tok" x="{x(t):.1f}" y="{b1}" width="{x(t + d) - x(t):.1f}" height="{BAR}" rx="4"/>'
                    f'<text class="sub mid" x="{(x(t) + x(t + d)) / 2:.1f}" y="{b1 - 5}">{name}</text>')
    t += d
body.append(f'<path class="lfg" d="M{x(SEQ_END):.1f},{b1 - 4}V{b1 + BAR + 4}"/>'
            f'<text class="sub" x="{x(SEQ_END) + 6:.1f}" y="{b1 + 16}">xong sau tổng ba lần</text>')

body.append(f'<text class="cap" x="20" y="{b2 + 16}">song song</text>')
SH = 18
for i, (name, d) in enumerate(OUT):
    body.append(bar(0, d, b2 + i * (SH + 3), "tok", name, SH))
body.append(f'<path class="lfg" d="M{x(PAR_END):.1f},{b2 - 4}V{b2 + 3 * (SH + 3) + 1}"/>'
            f'<text class="sub" x="{x(PAR_END) + 6:.1f}" y="{b2 + 45}">xong sau lần chậm nhất</text>')

body.append(f'<text class="cap2" x="20" y="{H - 16}">Độ dài các thanh là giả định, chỉ để so hình dạng. '
            'PII ngắn nhất: §2.4 đo khoảng 1,7 ms.</text>')

svgkit.write(__file__, "guardrail-latency.svg", svgkit.svg(
    W, H, "Input guardrail chạy trước thì chữ đầu tiên tới khách muộn thêm toàn bộ thời gian guardrail; "
    "chạy song song với lần sinh thì gần như không thêm gì, nhưng bị chặn thì token đã sinh vẫn tính tiền "
    "và tool có thể đã chạy. Ba output guardrail độc lập chạy song song thì xong sau guardrail chậm nhất, "
    "không phải sau tổng của ba", body))
