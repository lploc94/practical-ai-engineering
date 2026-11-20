"""Sinh vram-budget.svg: bộ nhớ GPU chia cho weights, overhead và KV cache (§3.1).

Số liệu lấy đúng theo hai dòng Llama 3.3 70B FP8 của bảng ở §3.1, cùng giả định với
hàm plan() trong mem_calc.py:
- H100 80GB HBM3: 81.559 MiB theo nvidia-smi [44], tức khoảng 79,6 GiB.
- utilization 0,9 (như --gpu-memory-utilization), overhead 2 GiB mỗi GPU.
- Weights FP8: 70,55 tỉ tham số [43] × 1 byte = 65,7 GiB; tensor parallel chia weights
  cho các GPU nên tính trên tổng bộ nhớ.
- KV cache BF16 của Llama 3.3 70B: 320 KiB mỗi token, 10,0 GiB cho chuỗi 32.768 token.
Kết quả: 1× H100 còn 4,0 GiB cho KV (không chứa nổi một chuỗi 32k), 2× H100 còn 73,7 GiB
(7 chuỗi 32k). Phần 10% ngoài utilization vẽ nét đứt.

Chạy: python3 make_vram_budget.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

GiB = 2**30
GPU = 81559 / 1024                       # GiB mỗi H100
UTIL, OVER = 0.9, 2.0
PARAMS = 70.553706496e9                  # safetensors.total của Llama 3.3 70B [43]
WEIGHTS = PARAMS / GiB                   # FP8: 1 byte mỗi tham số
KV_TOKEN = 2 * 80 * 8 * 128 * 2          # 2 × 80 lớp × 8 KV head × head_dim 128 × 2 byte
KV_SEQ = KV_TOKEN * 32768 / GiB

assert KV_TOKEN == 320 * 1024
assert round(KV_SEQ, 1) == 10.0
assert round(GPU, 1) == 79.6 and round(WEIGHTS, 1) == 65.7


def budget(gpus: int) -> float:
    return gpus * (GPU * UTIL - OVER) - WEIGHTS


B1, B2 = budget(1), budget(2)
assert round(B1, 1) == 4.0 and round(B2, 1) == 73.7, (B1, B2)
assert math.floor(B1 / KV_SEQ) == 0 and math.floor(B2 / KV_SEQ) == 7
assert math.floor(B1 * GiB / KV_TOKEN) == 13023   # cột tổng token của bảng §3.1


def vn(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}".replace(".", ",")


W, H = 900, 420
X0, X1 = 170, 840
XMAX = 2 * GPU
BH = 44
Y1, Y2 = 120, 230


def x(v: float) -> float:
    return X0 + v / XMAX * (X1 - X0)


def seg(y: float, a: float, b: float, cls: str, label: str = "") -> str:
    x0, x1 = x(a) + 1, x(b) - 1
    out = f'<rect class="{cls}" x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" height="{BH}" rx="5"/>'
    if label:
        out += f'<text class="lbl" style="font-size:13px" x="{(x0 + x1) / 2:.1f}" y="{y + 27}">{label}</text>'
    return out


body: list[str] = []
body.append('<text class="title" x="20" y="38">Llama 3.3 70B FP8: KV cache là phần còn lại</text>')
body.append(svgkit.legend(20, 72, [("blue", "weights"), ("tok", "overhead"), ("orange", "KV cache"),
                                    ("slot", "10% không dùng")]))

for y, n, b in ((Y1, 1, B1), (Y2, 2, B2)):
    body.append(f'<text class="row" x="20" y="{y + 20}">{n}× H100</text>')
    body.append(f'<text class="sub" x="20" y="{y + 38}">{vn(n * GPU)} GiB</text>')
    used = n * GPU * UTIL
    body.append(seg(y, 0, WEIGHTS, "blue", f"weights {vn(WEIGHTS)} GiB"))
    body.append(seg(y, WEIGHTS, WEIGHTS + n * OVER, "tok"))
    body.append(seg(y, WEIGHTS + n * OVER, used, "orange", "" if n == 1 else f"KV {vn(b)} GiB"))
    body.append(seg(y, used, n * GPU, "slot"))

# Nhãn KV của 1× H100: đoạn quá hẹp nên đặt bên phải thanh, có mũi tên chỉ vào
kx = (x(WEIGHTS + OVER) + x(GPU * UTIL)) / 2
body.append(f'<line class="lorange" x1="{x(GPU) + 46:.1f}" y1="{Y1 + BH / 2}" x2="{kx + 6:.1f}" '
            f'y2="{Y1 + BH / 2}" marker-end="url(#aorange)"/>')
body.append(f'<text class="cap" x="{x(GPU) + 52:.1f}" y="{Y1 + 18}" style="font-weight:600">'
            f'KV {vn(B1)} GiB: không đủ một chuỗi 32k</text>')
body.append(f'<text class="sub" x="{x(GPU) + 52:.1f}" y="{Y1 + 36}">khoảng 13.000 token cho mọi request</text>')

# Thước một chuỗi 32k trên hàng 2
n32 = math.floor(B2 / KV_SEQ)
s0 = WEIGHTS + 2 * OVER
for k in range(1, n32 + 1):
    tx = x(s0 + k * KV_SEQ)
    body.append(f'<line class="axis" x1="{tx:.1f}" y1="{Y2 + BH + 2}" x2="{tx:.1f}" y2="{Y2 + BH + 10}"/>')
body.append(f'<text class="sub" x="{x(s0):.1f}" y="{Y2 + BH + 26}">mỗi vạch = một chuỗi 32k (10 GiB KV): '
            f'{n32} chuỗi</text>')

# Trục
ay = Y2 + BH + 52
body.append(f'<line class="axis" x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}"/>')
for v in range(0, 161, 40):
    body.append(f'<line class="axis" x1="{x(v):.1f}" y1="{ay}" x2="{x(v):.1f}" y2="{ay + 5}"/>'
                f'<text class="sub mid" x="{x(v):.1f}" y="{ay + 20}">{v}</text>')
body.append(f'<text class="sub end" x="{X1}" y="{ay + 38}">GiB</text>')
body.append(f'<text class="cap2" x="20" y="{H - 14}">Giả định: utilization 0,9, overhead 2 GiB mỗi GPU, '
            'KV BF16. Weights chia đều cho hai GPU.</text>')

svgkit.write(__file__, "vram-budget.svg", svgkit.svg(
    W, H, "Llama 3.3 70B FP8 trên H100: weights chiếm 65,7 GiB. Một GPU chỉ còn 4,0 GiB cho KV cache, "
    "không đủ một chuỗi 32k token; hai GPU còn 73,7 GiB, đủ 7 chuỗi 32k, vì weights không tăng theo số GPU",
    body))
