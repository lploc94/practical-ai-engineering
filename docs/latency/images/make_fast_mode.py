"""Sinh fast-mode.svg: hình tĩnh, fast mode rút ngắn decode nhưng không đổi TTFT (§2.3).

Số liệu giả định ghép từ các ví dụ trong bài:
- §1.2: câu trả lời 600 token ở 60 tokens/giây là 10 giây decode;
  tổng ≈ TTFT + số output token ÷ tokens/giây.
- §3.2: TTFT 1,5 s mỗi lượt.
- §2.3: Claude fast mode cho tới 2,5 lần output tokens/giây, không cải thiện TTFT [2].
  Hình lấy đúng mức tối đa 2,5 lần: 150 tokens/giây, 4 giây decode.
Hai hàng chung một trục thời gian.

Chạy: python3 make_fast_mode.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

TTFT, N_OUT, TPS, SPEEDUP = 1.5, 600, 60, 2.5
DEC_STD = N_OUT / TPS
DEC_FAST = N_OUT / (TPS * SPEEDUP)
TOT_STD, TOT_FAST = TTFT + DEC_STD, TTFT + DEC_FAST
assert DEC_STD == 10 and DEC_FAST == 4          # §1.2: 600 token ở 60 tok/s = 10 s
assert (TOT_STD, TOT_FAST) == (11.5, 5.5)

W, H = 820, 330
X0, X1 = 200, 710                              # trục 0 → TMAX giây
TMAX = 12
BH = 34
ROWS = {"std": 106, "fast": 186}


def x(s: float) -> float:
    return X0 + s / TMAX * (X1 - X0)


def fmt(s: float) -> str:
    return f"{s:g}".replace(".", ",")


body: list[str] = []
body.append('<text class="title" x="20" y="38">Fast mode: chữ chạy nhanh hơn, chữ đầu không đến sớm hơn</text>')
body.append(svgkit.legend(20, 70, [("blue", "TTFT: tới chữ đầu tiên"), ("orange", "decode: 600 token output")]))

rows = {"std": ("Tốc độ thường", f"{TPS} tokens/giây", DEC_STD, TOT_STD),
        "fast": ("Fast mode", f"{int(TPS * SPEEDUP)} tokens/giây (×{fmt(SPEEDUP)})", DEC_FAST, TOT_FAST)}
for key, (name, sub, dec, tot) in rows.items():
    y = ROWS[key]
    body.append(f'<text class="row" x="20" y="{y + 15}">{name}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 32}">{sub}</text>')
    body.append(f'<rect class="blue" x="{X0}" y="{y}" width="{x(TTFT) - X0 - 2:.1f}" height="{BH}" rx="4"/>')
    body.append(f'<rect class="orange" x="{x(TTFT):.1f}" y="{y}" width="{x(tot) - x(TTFT):.1f}" height="{BH}" rx="4"/>')
    body.append(f'<text class="lbl" x="{(x(TTFT) + x(tot)) / 2:.1f}" y="{y + BH / 2 + 5}">{fmt(dec)} s</text>')
    done = f"xong: {fmt(tot)} s"
    assert x(tot) + 8 + 7 * len(done) <= W - 30          # ước lượng 7 px/ký tự: lề phải ≥ 30 px
    body.append(f'<text class="sub" x="{x(tot) + 8:.1f}" y="{y + BH / 2 + 4}">{done}</text>')

# Mốc chữ đầu tiên: cùng một vị trí ở cả hai hàng
yt, yb = ROWS["std"] - 10, ROWS["fast"] + BH + 10
body.append(f'<line class="lfg dash" x1="{x(TTFT):.1f}" y1="{yt}" x2="{x(TTFT):.1f}" y2="{yb}"/>')
body.append(f'<text class="sub halo" x="{x(TTFT) + 6:.1f}" y="{yb + 6}">chữ đầu tiên: giây {fmt(TTFT)} ở cả hai</text>')

# Trục thời gian
ay = yb + 30
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for s in range(0, TMAX + 1, 2):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')

body.append(f'<text class="cap2" x="20" y="{H - 12}">Giả định: TTFT {fmt(TTFT)} s, {N_OUT} token output, '
            f'{TPS} tokens/giây; fast mode lấy mức tối đa {fmt(SPEEDUP)} lần.</text>')

svgkit.write(__file__, "fast-mode.svg", svgkit.svg(
    W, H, f"Với {N_OUT} token output, fast mode rút decode từ {fmt(DEC_STD)} xuống {fmt(DEC_FAST)} giây và tổng từ "
    f"{fmt(TOT_STD)} xuống {fmt(TOT_FAST)} giây, nhưng chữ đầu tiên vẫn đến ở giây {fmt(TTFT)}", body))
