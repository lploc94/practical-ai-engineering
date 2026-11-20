"""Sinh nondeterminism-fork.svg: SVG tĩnh minh hoạ vì sao temperature 0 vẫn không deterministic (§1.1).

Số liệu lấy từ thí nghiệm của Thinking Machines Lab [41] được trích trong bài:
Qwen3-235B chạy 1.000 lần cùng prompt, temperature 0, ra 80 output khác nhau;
mọi output giống nhau ở 102 token đầu và tách ra từ token 103. Ở token 103 chỉ
có 2 hướng đi; 80 output là kết quả của những lần rẽ tiếp ở các token sau. Với
kernel batch-invariant, cả 1.000 output giống hệt nhau.

Số nhánh và vị trí các lần rẽ sau token 103 chỉ tượng trưng, không phải số thật.
Ví dụ phép cộng số thực không kết hợp được kiểm tra bằng assert bên dưới.

Chạy: python3 make_nondeterminism_fork.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))
import svgkit  # noqa: E402

RUNS, DISTINCT, SAME_PREFIX = 1000, 80, 102
FORK_AT = SAME_PREFIX + 1
assert FORK_AT == 103
RUNS_VI = f"{RUNS:,}".replace(",", ".")  # 1.000

# Phép cộng số thực không có tính kết hợp: đổi thứ tự cộng là đổi kết quả
assert (0.1 + 1e20) - 1e20 == 0.0
assert 0.1 + (1e20 - 1e20) == 0.1

W, H = 900, 420
X0, XF, XE = 200, 560, 700         # đầu thân, chỗ rẽ, đầu mút các nhánh
R1, R2, BH = 172, 290, 40          # tâm dọc hàng 1, đỉnh hàng 2, chiều cao thanh
# Chỗ rẽ tiếp theo (x, y) và đầu mút các nhánh của nó; chỉ tượng trưng cho 80 output
SPLITS = [((XF + 50, R1 - 30), [R1 - 62, R1 - 44, R1 - 26]),
          ((XF + 80, R1 + 30), [R1 + 26, R1 + 44, R1 + 62])]
assert len(SPLITS) == 2  # token 103 chỉ có 2 hướng đi

body: list[str] = []
body.append(f'<text class="title" x="20" y="38">Cùng prompt, temperature 0, vẫn ra {DISTINCT} output khác nhau</text>')
body.append(f'<text class="cap2" x="20" y="62">Qwen3-235B chạy {RUNS_VI} lần với cùng một prompt '
            '(Thinking Machines Lab).</text>')
body.append(svgkit.legend(20, 94, [("blue", "giống nhau ở mọi output"), ("orange", "bắt đầu khác nhau")]))

# Hàng 1: kernel thông thường, batch size đổi theo tải
body.append(f'<text class="row" x="20" y="{R1 - 2}">Kernel thường</text>')
body.append(f'<text class="sub" x="20" y="{R1 + 16}">batch size đổi theo tải</text>')
body.append(f'<rect class="blue" x="{X0}" y="{R1 - BH / 2}" width="{XF - X0}" height="{BH}" rx="7"/>')
body.append(f'<text class="cap mid" x="{(X0 + XF) / 2}" y="{R1 + 5}">token 1–{SAME_PREFIX}: '
            f'giống nhau ở cả {RUNS_VI} output</text>')


def curve(x0: float, y0: float, x1: float, y1: float) -> str:
    k = (x1 - x0) / 2
    return f'<path class="lorange" d="M{x0},{y0} C{x0 + k},{y0} {x1 - k},{y1} {x1},{y1}"/>'


# Token 103 chỉ rẽ thành 2 hướng; mỗi hướng rẽ tiếp ở các token sau (vị trí tượng trưng)
for (sx, sy), ends in SPLITS:
    body.append(curve(XF, R1, sx, sy))
    body.append(f'<circle class="forange" cx="{sx}" cy="{sy}" r="3"/>')
    for ey in ends:
        body.append(curve(sx, sy, XE, ey))
        body.append(f'<circle class="forange" cx="{XE}" cy="{ey}" r="3.5"/>')
body.append(f'<circle class="forange" cx="{XF}" cy="{R1}" r="3.5"/>')
body.append(f'<text class="sub end halo" x="{XF}" y="{R1 + BH / 2 + 18}">rẽ lần đầu: token {FORK_AT}</text>')
body.append(f'<text class="sub halo" x="{XF + 60}" y="{R1 + 84}">rẽ tiếp ở các token sau</text>')
body.append(f'<text class="row" x="{XE + 16}" y="{R1 - 2}">{DISTINCT} output</text>')
body.append(f'<text class="sub" x="{XE + 16}" y="{R1 + 16}">khác nhau khi kết thúc</text>')

# Hàng 2: kernel batch-invariant
body.append(f'<text class="row" x="20" y="{R2 + 16}">Kernel</text>')
body.append(f'<text class="row" x="20" y="{R2 + 34}">batch-invariant</text>')
body.append(f'<rect class="blue" x="{X0}" y="{R2}" width="{XE - X0}" height="{BH}" rx="7"/>')
body.append(f'<text class="cap mid" x="{(X0 + XE) / 2}" y="{R2 + 25}">mọi token: cả {RUNS_VI} '
            'output giống hệt nhau</text>')

# Nguyên nhân
body.append(f'<text class="cap" x="20" y="{H - 48}">Tải đổi → batch size đổi → thứ tự cộng số thực trong kernel '
            'đổi → logit lệch rất nhỏ → có lúc đổi token được chọn.</text>')
body.append(f'<text class="cap2" x="20" y="{H - 24}">Cộng số thực không kết hợp: (0.1 + 1e20) − 1e20 = 0, '
            'còn 0.1 + (1e20 − 1e20) = 0.1.</text>')

svgkit.write(__file__, "nondeterminism-fork.svg", svgkit.svg(
    W, H, f"Chạy {RUNS_VI} lần cùng prompt ở temperature 0: mọi output giống nhau đến token {SAME_PREFIX} rồi bắt đầu "
    f"tách từ token {FORK_AT}, rẽ tiếp ở các token sau thành {DISTINCT} output khác nhau, vì batch size đổi theo tải làm đổi thứ tự cộng số thực; "
    f"kernel batch-invariant cho {RUNS_VI} output giống hệt", body))
