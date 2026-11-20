"""Sinh decode-vs-score.svg: SVG động so LLM sinh JSON với decision model (§1.3).

Hai hàng chạy trên cùng một đồng hồ, cùng state và cùng câu hỏi `department`:
- LLM + structured output: prefill rồi decode tuần tự từng token của JSON
  {"department": "returns"} (5 token, cách tách token chỉ để minh hoạ). Kết quả là
  một chuỗi, không có xác suất cho từng option.
- Decision model: prefill rồi chấm mọi option trong một lượt, không sinh token nào.
  Phân phối lấy từ response mẫu ở §1.2: shipping 0.04, billing 0.35, returns 0.61.
Độ dài các khối chỉ minh hoạ thứ tự, không phải thời gian đo được.
Khung tĩnh cho prefers-reduced-motion là trạng thái cuối của cả hai hàng.

Chạy: python3 make_decode_vs_score.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402
from svgkit import MONO  # noqa: E402

TOKENS = [("{", 30), ('"department"', 124), (":", 30), ('"returns"', 100), ("}", 30)]
PROBS = [("shipping", 0.04), ("billing", 0.35), ("returns", 0.61)]
assert abs(sum(p for _, p in PROBS) - 1) < 1e-9
assert max(PROBS, key=lambda o: o[1])[0] == "returns"

W, H = 820, 372
LX, PX, PW, BH = 20, 196, 110, 40     # cột nhãn, khối prefill, cao khối
TX0, TGAP = PX + PW + 16, 6           # token đầu tiên
R1, R2 = 108, 236                     # hàng LLM, hàng decision model

PRE, OUT, STEP = 6, 20, 8             # % chu kỳ: prefill, kết quả đầu tiên, khoảng cách token
LLM_DONE = OUT + len(TOKENS) * STEP
a = svgkit.Anim(dur=16, fade_out=95)  # chú thích cuối hiện đủ ở 62%, giữ khoảng 5 giây
# Thay đổi cuối cùng là chú thích ở LLM_DONE (hiện đủ sau 2%); trạng thái cuối phải giữ ≥ 3 giây
assert (a.fade_out - (LLM_DONE + 2)) / 100 * a.dur >= 3


def show(t: float, inner: str) -> str:
    return f"<g {a([(t, 0), (t + 2, 1)])}>{inner}</g>"


body: list[str] = []
body.append(f'<style>.code{{font-family:{MONO};font-size:13px}}.fmuted{{fill:var(--muted)}}</style>')
body.append('<text class="title" x="20" y="38">LLM viết từng token; decision model chấm mọi option một lượt</text>')
body.append('<text class="cap2" x="20" y="62">Cùng state, cùng câu hỏi department. Hai hàng chạy cùng một đồng hồ; '
            'độ dài khối chỉ minh hoạ thứ tự.</text>')
body.append(svgkit.legend(W - 300, 92, [("blue", "prefill"), ("orange", "token sinh ra")]))

# Nhãn hàng
body.append(f'<text class="row" x="{LX}" y="{R1 + 18}">LLM</text>'
            f'<text class="sub" x="{LX}" y="{R1 + 35}">structured output</text>')
body.append(f'<text class="row" x="{LX}" y="{R2 + 18}">Decision model</text>'
            f'<text class="sub" x="{LX}" y="{R2 + 35}">chỉ prefill</text>')

# Prefill: hai hàng cùng lúc (cùng state)
for r in (R1, R2):
    body.append(f'<rect class="slot" x="{PX}" y="{r}" width="{PW}" height="{BH}" rx="7"/>')
    body.append(show(PRE, f'<rect class="blue" x="{PX}" y="{r}" width="{PW}" height="{BH}" rx="7"/>'
                          f'<text class="lbl" x="{PX + PW / 2}" y="{r + BH / 2 + 5}">prefill</text>'))

# Hàng LLM: mỗi bước decode thêm đúng một token
x = TX0
for k, (tok, w) in enumerate(TOKENS):
    body.append(f'<rect class="slot" x="{x}" y="{R1}" width="{w}" height="{BH}" rx="7"/>')
    body.append(show(OUT + k * STEP,
                     f'<rect class="orange" x="{x}" y="{R1}" width="{w}" height="{BH}" rx="7"/>'
                     f'<text class="code mid" x="{x + w / 2}" y="{R1 + BH / 2 + 5}">{tok}</text>'))
    x += w + TGAP
x_end = x - TGAP
body.append(show(LLM_DONE, f'<text class="cap" x="{PX}" y="{R1 + BH + 26}">{len(TOKENS)} bước decode tuần tự; kết quả là '
                           'chuỗi "returns", không kèm xác suất.</text>'))

# Hàng decision model: cả ba option được chấm cùng lúc
BAR_X, BAR_W, BAR_H, BAR_STEP = TX0 + 90, 260, 18, 26
bars = []
for i, (name, p) in enumerate(PROBS):
    y = R2 - 12 + i * BAR_STEP
    bars.append(f'<text class="code end" x="{BAR_X - 10}" y="{y + 14}">{name}</text>'
                f'<rect class="fmuted" x="{BAR_X}" y="{y}" width="{max(p * BAR_W, 2):.1f}" height="{BAR_H}" rx="3"/>'
                f'<text class="sub" x="{BAR_X + p * BAR_W + 8:.1f}" y="{y + 14}">{p:.2f}</text>')
body.append(show(OUT, "".join(bars)))
body.append(show(OUT + STEP, f'<text class="cap" x="{PX}" y="{R2 + 3 * BAR_STEP + 16}">0 token sinh ra; trả phân phối '
                             'trên đúng các option bạn đưa.</text>'))

svgkit.write(__file__, "decode-vs-score.svg", svgkit.svg(
    W, H, "Cùng một câu hỏi chọn đội: LLM prefill rồi decode tuần tự 5 token để viết chuỗi returns, không có xác suất; "
    "decision model chỉ prefill rồi chấm cả ba option một lượt: shipping 0.04, billing 0.35, returns 0.61", body, a))
