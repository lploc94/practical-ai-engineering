"""Sinh ttft-vs-e2e.svg: SVG động so stream với không stream trên cùng đồng hồ (§1.1).

Số liệu giả định lấy từ ví dụ ở §1.1: TTFT 0,8 giây, ITL 20 ms, output 1.500 token.
e2e ≈ TTFT + (output_tokens − 1) × ITL = 0,8 + 1.499 × 0,02 ≈ 30,8 giây ở cả hai chế độ.
- Hàng "Model sinh": giống nhau ở cả hai chế độ (TTFT rồi các token).
- Hàng "Stream": người dùng thấy chữ ngay khi token được sinh, từ giây 0,8.
- Hàng "Không stream": người dùng chỉ thấy chữ khi token cuối đã sinh, ở giây 30,8.
Output được vẽ thành 8 đoạn bằng nhau chỉ để dễ nhìn. Hai hàng "Model sinh" và
"Stream" lộ ra liên tục theo con trỏ thời gian (một tấm che màu nền trượt cùng con
trỏ), nên chữ đầu tiên ở hàng Stream hiện đúng lúc con trỏ qua giây 0,8.
30,8 giây mô phỏng chạy trong khoảng 10 giây animation.
Khung tĩnh cho prefers-reduced-motion là trạng thái cuối (mọi hàng đầy đủ).

Chạy: python3 make_ttft_vs_e2e.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

TTFT, ITL, N_OUT = 0.8, 0.02, 1500
E2E = TTFT + (N_OUT - 1) * ITL
assert abs(E2E - 30.78) < 1e-9 and round(E2E, 1) == 30.8   # §1.1 ghi ≈ 30,8 giây

W, H = 820, 390
X0, X1 = 200, 780                    # trục thời gian 0 → E2E
ROWS = {"model": 100, "stream": 170, "batch": 240}
BH = 30
NSEG = 8
DUR = 18
P0, P1 = 6, 62                       # % chu kỳ ứng với giây 0 và giây E2E


def x(s: float) -> float:
    return X0 + s / E2E * (X1 - X0)


def p(s: float) -> float:
    return P0 + s / E2E * (P1 - P0)


def fmt(s: float) -> str:
    return f"{s:.1f}".replace(".", ",")


a = svgkit.Anim(dur=DUR, fade_out=95)   # giữ trạng thái cuối khoảng 5,5 giây
body: list[str] = []

body.append('<text class="title" x="20" y="38">Stream và không stream: cùng tổng thời gian, khác lúc thấy chữ</text>')
body.append(svgkit.legend(20, 66, [("blue", "TTFT: xếp hàng, prefill"), ("orange", "token output")]))

labels = {"model": ("Model sinh", "giống nhau ở hai chế độ"),
          "stream": ("Stream", "người dùng thấy"),
          "batch": ("Không stream", "người dùng thấy")}
for key, (name, sub) in labels.items():
    y = ROWS[key]
    body.append(f'<text class="row" x="20" y="{y + 14}">{name}</text>')
    body.append(f'<text class="sub" x="20" y="{y + 31}">{sub}</text>')
    body.append(f'<rect class="slot" x="{X0}" y="{y}" width="{X1 - X0}" height="{BH}" rx="5"/>')

# Hàng model và stream vẽ đầy đủ; tấm che bên dưới làm chúng lộ ra theo con trỏ
ym = ROWS["model"]
seg = (E2E - TTFT) / NSEG
full = [f'<rect class="blue" x="{X0}" y="{ym}" width="{x(TTFT) - X0:.1f}" height="{BH}" rx="3"/>']
for k in range(NSEG):
    s0, s1 = TTFT + k * seg, TTFT + (k + 1) * seg
    for key in ("model", "stream"):
        full.append(f'<rect class="orange" x="{x(s0) + 1:.1f}" y="{ROWS[key]}" '
                    f'width="{x(s1) - x(s0) - 2:.1f}" height="{BH}" rx="3"/>')
body.append(f'<g {a([(0, 1)])}>' + "".join(full) + "</g>")

# Tấm che màu nền trượt cùng con trỏ: phần bên phải con trỏ chưa xảy ra nên chưa thấy.
# Ở cuối trượt thêm 4px để không che viền phải của đoạn cuối.
# Ẩn trong khung tĩnh (a-hide) để reduced-motion thấy cả hai hàng đầy đủ.
dx = X1 - X0
body.append('<style>@keyframes cover{'
            f'0%{{transform:translateX(0)}}{P0}%{{transform:translateX(0)}}'
            f'{P1}%{{transform:translateX({dx + 4}px)}}100%{{transform:translateX({dx + 4}px)}}}}</style>')
for key in ("model", "stream"):
    y = ROWS[key]
    body.append(f'<g class="a-hide" style="animation:cover {DUR}s linear infinite">'
                f'<rect class="bg" x="{X0 - 2}" y="{y - 3}" width="{dx + 6}" height="{BH + 6}"/></g>')

# Không stream: chờ đến token cuối, rồi cả câu trả lời hiện một lần
yb = ROWS["batch"]
body.append(f'<g {a([(P0, 0), (P0 + 1, 1), (P1 - 0.5, 1), (P1, 0)])}>'
            f'<text class="cap2" x="{(X0 + X1) / 2}" y="{yb + 20}" style="text-anchor:middle">đang chờ…</text></g>')
body.append(f'<g {a([(P1 - 0.5, 0), (P1 + 0.5, 1)])}><rect class="orange" x="{x(TTFT) + 1:.1f}" y="{yb}" '
            f'width="{X1 - x(TTFT) - 2:.1f}" height="{BH}" rx="3"/></g>')

# Mốc "chữ đầu tiên" cho từng hàng người dùng
ys = ROWS["stream"]
first = p(TTFT)
body.append(f'<g {a([(first - 0.5, 0), (first + 0.5, 1)])}>'
            f'<line class="lfg dash" x1="{x(TTFT):.1f}" y1="{ys - 6}" x2="{x(TTFT):.1f}" y2="{ys + BH + 6}"/>'
            f'<text class="sub halo" x="{x(TTFT) + 6:.1f}" y="{ys + BH + 17}">chữ đầu tiên: giây {fmt(TTFT)}</text></g>')
body.append(f'<g {a([(P1 - 0.5, 0), (P1 + 0.5, 1)])}>'
            f'<line class="lfg dash" x1="{X1}" y1="{yb - 6}" x2="{X1}" y2="{yb + BH + 6}"/>'
            f'<text class="sub end halo" x="{X1 - 6}" y="{yb + BH + 17}">chữ đầu tiên: giây {fmt(E2E)}</text></g>')

# Trục thời gian
ay = yb + BH + 40
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for s in (0, 10, 20, 30):
    body.append(f'<line class="axis" x1="{x(s):.1f}" y1="{ay}" x2="{x(s):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(s):.1f}" y="{ay + 18}">{s} s</text>')

# Con trỏ thời gian: chạy từ giây 0 tới giây E2E, ẩn trong khung tĩnh
body.append('<style>@keyframes cur{'
            f'0%{{transform:translateX(0);opacity:0}}{P0}%{{transform:translateX(0);opacity:1}}'
            f'{P1}%{{transform:translateX({dx}px);opacity:1}}{P1 + 2}%{{transform:translateX({dx}px);opacity:0}}'
            f'100%{{transform:translateX({dx}px);opacity:0}}}}</style>')
body.append(f'<g class="a-hide" style="animation:cur {DUR}s linear infinite">'
            f'<line class="lfg" x1="{X0}" y1="{ROWS["model"] - 10}" x2="{X0}" y2="{ay}"/></g>')

# Tổng kết sau khi cả hai chế độ xong
body.append(f'<g {a([(P1 + 2, 0), (P1 + 4, 1)])}>'
            f'<text class="cap" x="20" y="{H - 34}">Cả hai chế độ xong ở giây {fmt(E2E)}; stream chỉ đổi lúc người dùng '
            'thấy chữ đầu tiên.</text></g>')
body.append(f'<text class="cap2" x="20" y="{H - 12}">Số liệu giả định từ ví dụ §1.1: TTFT {fmt(TTFT)} s, ITL 20 ms, '
            '1.500 token output.</text>')

svgkit.write(__file__, "ttft-vs-e2e.svg", svgkit.svg(
    W, H, f"Với TTFT {fmt(TTFT)} giây, ITL 20 ms và 1.500 token, cả stream và không stream đều xong ở giây "
    f"{fmt(E2E)}; khi stream người dùng thấy chữ từ giây {fmt(TTFT)}, khi không stream phải chờ đến giây "
    f"{fmt(E2E)}", body, a))
