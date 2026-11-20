"""Sinh rerun-failures.svg: chạy low rồi chạy lại bài lỗi ở high, so với một mức effort cố định (§3.1).

Số liệu của Anthropic [2] trên tập con SWE-bench Pro (478 bài) với Opus 5.5, chi phí
tính trên mỗi tác vụ (không phải mỗi bài giải được), như bảng effort ở §3.3:
- low 87,4% · 0,11 USD; medium 92,8% · 0,21 USD; high 95,3% · 0,29 USD.
- low rồi chạy lại các bài lỗi ở high: khoảng 97% · khoảng 0,17 USD, đã tính cả lần
  thử low thất bại. Anthropic tính điểm này từng bài từ các lượt chạy cố định, không
  chạy riêng.
Trục tung bắt đầu từ 85%.

Chạy: python3 make_rerun_failures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

FIXED = [("low", 0.11, 87.4), ("medium", 0.21, 92.8), ("high", 0.29, 95.3)]
RERUN = (0.17, 97.0)
low_fail = 100 - FIXED[0][2]
saving = 1 - RERUN[0] / FIXED[2][1]
assert round(low_fail, 1) == 12.6
SAVING_PCT = 40                           # bảng Tra cứu nhanh và [2]: "khoảng 40%"
assert abs(saving * 100 - SAVING_PCT) < 2  # 1 − 0,17/0,29 ≈ 41,4%
assert RERUN[1] >= FIXED[2][2] and RERUN[0] < FIXED[2][1]


def vn(v: float, nd: int) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


# Số chép từ nguồn phải khớp bài: bảng effort §3.3 và đoạn §3.1
README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
assert "| SWE-bench Pro (tập con) | Opus 5.5 | " + " | ".join(
    f"{vn(p, 1)}% · {vn(c, 2)} USD" for _, c, p in FIXED) + " |" in README
assert f"giải khoảng {vn(RERUN[1], 0)}% với khoảng {vn(RERUN[0], 2)} USD mỗi bài" in README
assert f"Khoảng {SAVING_PCT}% rẻ hơn chạy hết ở `high`" in README


W, H = 820, 440
PX0, PX1, PY0, PY1 = 90, 620, 96, 350
XMAX, Y0, Y1 = 0.35, 85, 100


def x(c: float) -> float:
    return PX0 + c / XMAX * (PX1 - PX0)


def y(p: float) -> float:
    return PY1 - (p - Y0) / (Y1 - Y0) * (PY1 - PY0)


body: list[str] = []
body.append('<text class="title" x="20" y="38">Chạy low trước, chạy lại bài lỗi ở high: rẻ hơn ≈40%, tỉ lệ giải không kém</text>')
body.append('<text class="cap2" x="20" y="62">Opus 5.5 trên tập con SWE-bench Pro, số đo của Anthropic. '
            'Chi phí trên mỗi tác vụ, trục tung bắt đầu từ 85%.</text>')

for p in range(85, 101, 5):
    body.append(f'<line class="grid" x1="{PX0}" y1="{y(p):.1f}" x2="{PX1}" y2="{y(p):.1f}"/>')
    body.append(f'<text class="sub end" x="{PX0 - 8}" y="{y(p) + 4:.1f}">{p}%</text>')
for c in (0, 0.1, 0.2, 0.3):
    body.append(f'<text class="sub mid" x="{x(c):.1f}" y="{PY1 + 18}">{vn(c, 2)}</text>')
body.append(f'<line class="axis" x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}"/>')
body.append(f'<text class="sub mid" x="{(PX0 + PX1) / 2}" y="{PY1 + 36}">Chi phí mỗi tác vụ (USD)</text>')
body.append(f'<text class="sub" x="20" y="{PY0 - 14}">Tỉ lệ bài giải được</text>')

# Đường của các mức effort cố định
pts = " L".join(f"{x(c):.1f},{y(p):.1f}" for _, c, p in FIXED)
body.append(f'<path class="lblue" d="M{pts}"/>')
for name, c, p in FIXED:
    body.append(f'<circle class="fblue" cx="{x(c):.1f}" cy="{y(p):.1f}" r="6"/>')
    body.append(f'<text class="cap halo" x="{x(c) + 12:.1f}" y="{y(p) + 18:.1f}">'
                f'<tspan style="font-family:{svgkit.MONO}">{name}</tspan> {vn(p, 1)}% · {vn(c, 2)}</text>')

# Chính sách chạy lại: mũi tên từ low (phần lỗi được chạy lại) tới điểm kết quả
rc, rp = RERUN
lc, lp = FIXED[0][1], FIXED[0][2]
body.append(f'<line class="lorange dash" x1="{x(lc) + 6:.1f}" y1="{y(lp) - 6:.1f}" x2="{x(rc) - 9:.1f}" '
            f'y2="{y(rp) + 9:.1f}" marker-end="url(#aorange)"/>')
body.append(f'<rect class="orange" x="{x(rc) - 7:.1f}" y="{y(rp) - 7:.1f}" width="14" height="14" rx="2"/>')
body.append(f'<text class="cap halo" x="{x(rc) - 14:.1f}" y="{y(rp) - 26:.1f}" style="text-anchor:end;font-weight:600">'
            f'low, rồi chạy lại bài lỗi ở high</text>')
body.append(f'<text class="cap halo" x="{x(rc) - 14:.1f}" y="{y(rp) - 8:.1f}" style="text-anchor:end">'
            f'≈{vn(rp, 0)}% · ≈{vn(rc, 2)}</text>')
mx, my = (x(lc) + x(rc)) / 2, (y(lp) + y(rp)) / 2
body.append(f'<text class="sub halo end" x="{mx - 10:.1f}" y="{my + 4:.1f}">{vn(low_fail, 1)}% bài lỗi ở low</text>')

# So với chạy hết ở high
hc, hp = FIXED[2][1], FIXED[2][2]
body.append(f'<line class="lfg dash" x1="{x(hc):.1f}" y1="{y(rp):.1f}" x2="{x(rc) + 12:.1f}" y2="{y(rp):.1f}" '
            f'marker-end="url(#afg)"/>')
body.append(f'<line class="axis dash" x1="{x(hc):.1f}" y1="{y(rp):.1f}" x2="{x(hc):.1f}" y2="{y(hp) - 8:.1f}"/>')
body.append(f'<text class="sub halo" x="{x(hc) + 8:.1f}" y="{y(rp) + 4:.1f}">rẻ hơn ≈{SAVING_PCT}% so với chạy hết ở high</text>')

body.append(f'<text class="cap2" x="20" y="{H - 30}">Chi phí của chính sách chạy lại đã gồm lần thử low thất bại. '
            'Điều kiện: có tín hiệu lỗi đáng tin (test, validator);</text>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">bài lỗi mất hai lần thời gian chạy.</text>')

svgkit.write(__file__, "rerun-failures.svg", svgkit.svg(
    W, H, f"Opus 5.5 trên tập con SWE-bench Pro: low {vn(FIXED[0][2], 1)}% với {vn(FIXED[0][1], 2)} USD mỗi tác vụ, "
    f"medium {vn(FIXED[1][2], 1)}% với {vn(FIXED[1][1], 2)} USD, high {vn(FIXED[2][2], 1)}% với "
    f"{vn(FIXED[2][1], 2)} USD. Chạy low rồi chạy lại {vn(low_fail, 1)}% bài lỗi ở high giải khoảng "
    f"{vn(rp, 0)}% với khoảng {vn(rc, 2)} USD: rẻ hơn khoảng {SAVING_PCT}% so với chạy hết ở high, tỉ lệ giải "
    "không thấp hơn",
    body))
